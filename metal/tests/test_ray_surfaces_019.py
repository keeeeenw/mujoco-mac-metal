# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned engine-oracle qualification for deformable ray query utilities."""
import ctypes
import itertools
from dataclasses import replace
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_metal.ray_queries import lower_ray_flex, lower_ray_skin, MetalRaySurface

_GPU = pytest.mark.skipif(os.environ.get('MUJOCO_METAL_RUN_GPU') != '1',
                         reason='serialized native surface-ray qualification')


def _model(dim, dof=None):
  count = {1:'4 1 1',2:'4 4 1',3:'4 4 4'}[dim]
  extra = '' if dof is None else f' dof="{dof}"'
  return mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
    <worldbody><flexcomp name="surface" type="grid" count="{count}"
      spacing=".12 .12 .12" dim="{dim}" radius=".01" mass="1"{extra}>
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <edge stiffness="0" damping="0"/>
    </flexcomp></worldbody></mujoco>''')


@pytest.mark.parametrize('dim',[1,2,3])
def test_flex_ray_topology_retains_source_face_order_local_ids_and_layers(dim):
  model = _model(dim)
  surface = lower_ray_flex(model,0)
  assert surface.nvert == model.flex_vertnum[0] and surface.dim == dim
  np.testing.assert_array_equal(surface.edges,model.flex_edge)
  if dim>1:
    elements = model.flex_elem.reshape(-1,dim+1)
    mappings = [(0,1,2)] if dim==2 else [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    expected = np.asarray([e[list(ids)] for e in elements for ids in mappings])
    np.testing.assert_array_equal(surface.faces,expected)
    np.testing.assert_array_equal(surface.layers,np.repeat(model.flex_elemlayer,len(mappings)))
  for a in (surface.edges,surface.faces,surface.layers):
    assert not a.flags.writeable
  with pytest.raises(ValueError,match='before device allocation'):
    MetalRaySurface(surface,2,memory_budget_bytes=1)


def test_skin_topology_validation_and_empty_surface_budget():
  for bad in (np.array([[0.,1.,2.]]),np.array([[0,1,3]]),np.array([[-1,1,2]])):
    with pytest.raises(ValueError):
      lower_ray_skin(3,bad)
  empty = lower_ray_skin(0,np.empty((0,3),np.int32))
  assert empty.nvert == 0 and empty.nbytes == 0
  with pytest.raises(ValueError,match='before device allocation'):
    MetalRaySurface(empty,1,memory_budget_bytes=79)
  with pytest.raises(ValueError,match='signed int32'):
    MetalRaySurface(empty,2**31)
  with pytest.raises(ValueError,match='local vertex topology'):
    MetalRaySurface(replace(empty,faces=np.array([[0,1,2]],np.int32)),1)
  with pytest.raises(ValueError,match='radius'):
    MetalRaySurface(replace(empty,radius=np.nan),1)


def _skin_oracle(faces,vertices,pnt,vec):
  # The pinned Python binding incorrectly exposes these C pointers as scalars.
  # Call the exported pinned CPU function with explicit array pointer types.
  library = ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob('libmujoco.*.dylib'))))
  pi,pf,pd = (ctypes.POINTER(t) for t in (ctypes.c_int,ctypes.c_float,ctypes.c_double))
  function = library.mju_raySkin
  function.argtypes = [ctypes.c_int,ctypes.c_int,pi,pf,pd,pd,pi]
  function.restype = ctypes.c_double
  faces = np.ascontiguousarray(faces,np.int32)
  vertices = np.ascontiguousarray(vertices,np.float32)
  pnt,vec = np.ascontiguousarray(pnt,float),np.ascontiguousarray(vec,float)
  vertex = np.array([-1],np.int32)
  distance = function(len(faces),len(vertices),faces.ctypes.data_as(pi),
      vertices.ctypes.data_as(pf),pnt.ctypes.data_as(pd),vec.ctypes.data_as(pd),
      vertex.ctypes.data_as(pi))
  return distance,vertex[0]


def test_pinned_skin_oracle_hits_face_and_retains_nearest_vertex_ties():
  faces = np.array([[0,1,2]],np.int32)
  vertices = np.array([[-1,-1,0],[1,-1,0],[0,1,0]],np.float32)
  assert _skin_oracle(faces,vertices,[0,-1,1],[0,0,-2]) == (.5,0)
  assert _skin_oracle(faces,vertices,[3,3,1],[0,0,-1]) == (-1,-1)


@_GPU
@pytest.mark.parametrize('dim',[1,2,3])
def test_native_flex_rays_flags_layers_normals_and_deformed_vertices_match_pinned(dim):
  torch = pytest.importorskip('torch')
  model = _model(dim)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world,ref in enumerate(refs):
    if world:
      ref.qpos[:] += np.linspace(-.012,.014,model.nq)
    mujoco.mj_forward(model,ref)
  vertices = np.stack([r.flexvert_xpos for r in refs]).astype(np.float32)
  # Exactly match CPU oracle geometry to the actual device input.
  for ref,v in zip(refs,vertices):
    ref.flexvert_xpos[:] = v
  tensor = lambda a:torch.as_tensor(np.asarray(a).copy(),device='mps')
  program = MetalRaySurface(lower_ray_flex(model,0),2)
  inputs = [(np.array([[.015,.007,1],[-.02,.018,1]],np.float32),
             np.array([[0,0,-2],[0,0,-.7]],np.float32)),
            (np.array([[3,3,1],[3,3,1]],np.float32),
             np.array([[0,0,-1],[0,0,-1]],np.float32))]
  for origins,vectors in inputs:
    for flags in itertools.product([False,True],repeat=4):
      for layer in ([0,1,-1] if dim==3 else [0]):
        out = program.run_device(tensor(vertices),tensor(origins),tensor(vectors),
            flex_layer=layer,flg_vert=flags[0],flg_edge=flags[1],
            flg_face=flags[2],flg_skin=flags[3])
        np.testing.assert_array_equal(out['status'].cpu(),0)
        for world,ref in enumerate(refs):
          vid,n = np.array([-1],np.int32),np.zeros(3)
          expected = mujoco.mj_rayFlex(model,ref,layer,*flags,0,
              origins[world].astype(float),vectors[world].astype(float),vid,n)
          np.testing.assert_allclose(out['distance'][world].cpu(),expected,atol=2e-6,rtol=2e-5)
          np.testing.assert_array_equal(out['vertid'][world].cpu(),vid[0])
          np.testing.assert_allclose(out['normal'][world].cpu(),n,atol=3e-6,rtol=3e-5)


@_GPU
def test_native_skin_ray_batch_changes_ties_degenerate_faces_and_invalid_worlds():
  torch = pytest.importorskip('torch')
  # Reversed winding and repeated vertex/degenerate triangles.
  faces = np.array([[0,1,2],[2,1,3],[0,0,1]],np.int32)
  vertices = np.array([[[-1,-1,0],[1,-1,0],[-1,1,0],[1,1,0]],
                       [[-1,-1,.3],[1,-1,.3],[-1,1,.3],[1,1,.3]]],np.float32)
  program = MetalRaySurface(lower_ray_skin(4,faces),2)
  tensor = lambda a:torch.as_tensor(np.asarray(a).copy(),device='mps')
  for origins in (np.array([[.2,.3,1],[-.2,.3,1]],np.float32),
                  np.array([[0,-1,1],[0,-1,1]],np.float32),
                  np.array([[3,3,1],[3,3,1]],np.float32)):
    directions = np.array([[0,0,-2],[0,0,-.5]],np.float32)
    out = program.run_device(tensor(vertices),tensor(origins),tensor(directions))
    for world in range(2):
      expected,vid = _skin_oracle(faces,vertices[world],origins[world],directions[world])
      np.testing.assert_allclose(out['distance'][world].cpu(),expected,rtol=3e-6,atol=1e-6)
      assert int(out['vertid'][world].cpu()) == vid
    np.testing.assert_array_equal(out['status'].cpu(),0)
  invalid = vertices.copy()
  invalid[1,0,0] = np.nan
  out = program.run_device(tensor(invalid),tensor([[0,0,1],[0,0,1]]).float(),
                           tensor([[0,0,-1],[0,0,-1]]).float())
  np.testing.assert_array_equal(out['status'].cpu(),[0,1])
  np.testing.assert_array_equal(out['distance'][1].cpu(),-1)
  np.testing.assert_array_equal(out['normal'][1].cpu(),0)
  empty = MetalRaySurface(lower_ray_skin(0,np.empty((0,3),np.int32)),2)
  out = empty.run_device(tensor(np.empty((2,0,3),np.float32)),
      tensor(np.zeros((2,3),np.float32)),tensor(np.array([[0,0,1],[0,0,1]],np.float32)))
  np.testing.assert_array_equal(out['distance'].cpu(),-1)
  np.testing.assert_array_equal(out['status'].cpu(),0)


@_GPU
@pytest.mark.parametrize('dim,dof',[(2,None),(3,'trilinear'),(3,'quadratic')])
def test_public_surface_ray_queries_preserve_prepared_state_and_use_current_flex_positions(dim,dof):
  torch = pytest.importorskip('torch')
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.native_api import mj_fwdPosition,mj_rayFlex,mju_raySkin
  model = _model(dim,dof)
  sim = MetalSimulation(model,batch_size=1,profile='integrated_scalable_v1')
  record = mj_fwdPosition(sim,return_record=True,skipsensor=True)
  before = sim.state._qpos.clone()
  out = mj_rayFlex(sim,0,False,False,True,False,0,[[.015,.007,1]],[[0,0,-1]])
  ref = mujoco.MjData(model)
  mujoco.mj_forward(model,ref)
  vid,n = np.array([-1],np.int32),np.zeros(3)
  expected = mujoco.mj_rayFlex(model,ref,0,False,False,True,False,0,
      np.array([.015,.007,1]),np.array([0.,0.,-1.]),vid,n)
  np.testing.assert_allclose(out['distance'].cpu(),expected,rtol=2e-5,atol=2e-6)
  np.testing.assert_array_equal(out['vertid'].cpu(),vid)
  torch.testing.assert_close(sim.state._qpos,before,atol=0,rtol=0)
  assert sim._forward_stages._record is record
  # Explicit skin input must remain independent of the simulation's flex.
  skin = mju_raySkin(sim,[[0,1,2]],[[[-1,-1,.5],[1,-1,.5],[0,1,.5]]],
                    [[0,0,1]],[[0,0,-1]])
  np.testing.assert_allclose(skin['distance'].cpu(),.5,atol=1e-6)
  torch.testing.assert_close(sim.state._qpos,before,atol=0,rtol=0)
  # A subsequent accepted step changes geometry; a default query must rebuild
  # positions from current state rather than reuse an earlier ray's vertices.
  sim.state._qvel.fill_(.04)
  for _ in range(2):
    np.testing.assert_array_equal(sim.step().cpu(),0)
  current = sim.state._qpos.detach().cpu().numpy()[0].astype(float)
  ref.qpos[:] = current
  mujoco.mj_forward(model,ref)
  out = mj_rayFlex(sim,0,False,False,True,False,0,[[.015,.007,1]],[[0,0,-1]])
  expected = mujoco.mj_rayFlex(model,ref,0,False,False,True,False,0,
      np.array([.015,.007,1]),np.array([0.,0.,-1.]),vid,n)
  np.testing.assert_allclose(out['distance'].cpu(),expected,rtol=2e-5,atol=2e-6)
  np.testing.assert_array_equal(out['vertid'].cpu(),vid)
  # Outputs must remain owned after subsequent calls overwrite program scratch.
  saved = {name:value.clone() for name,value in out.items()}
  mj_rayFlex(sim,0,True,True,True,True,0,[[3,3,1]],[[0,0,-1]])
  for name,value in saved.items():
    torch.testing.assert_close(out[name],value,rtol=0,atol=0)
