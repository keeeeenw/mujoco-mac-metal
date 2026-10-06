# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Independent engine-oracle gates for native complete-scene ray queries."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.ray_queries import lower_ray_scene, MetalRayQueries, MetalRayPrimitives


def _scene():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <asset>
      <material name="hidden" rgba="1 1 1 0"/>
      <mesh name="cube" vertex="-.2 -.2 -.2  .2 -.2 -.2  .2 .2 -.2  -.2 .2 -.2
          -.2 -.2 .2  .2 -.2 .2  .2 .2 .2  -.2 .2 .2"/>
      <hfield name="hill" nrow="3" ncol="3" size=".3 .3 .25 .1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 .1" group="0"/>
      <geom type="sphere" size=".25" pos="-2 0 .5" group="1"/>
      <geom type="capsule" size=".15 .3" pos="-1 0 .5" euler="20 30 10"/>
      <geom type="ellipsoid" size=".25 .18 .3" pos="0 0 .5"/>
      <geom type="cylinder" size=".2 .3" pos="1 0 .5" euler="10 20 30"/>
      <geom type="box" size=".2 .3 .25" pos="2 0 .5" euler="15 25 5"/>
      <geom type="mesh" mesh="cube" pos="3 0 .5" euler="15 25 5" contype="0" conaffinity="0"/>
      <geom type="hfield" hfield="hill" pos="4 0 .2"/>
      <geom type="sphere" size=".3" pos="5 0 .5" rgba="1 1 1 0"/>
      <geom type="sphere" size=".3" pos="6 0 .5" material="hidden"/>
      <body pos="7 0 .5"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".25" mass="1"/>
      </body>
    </worldbody></mujoco>''')
  model.hfield_data[:] = [.1, .3, .1, .2, 1, .2, .1, .4, .1]
  return model


def test_ray_scene_lowering_contains_all_geoms_and_immutable_assets():
  model = _scene()
  scene = lower_ray_scene(model)
  assert scene.ngeom == model.ngeom
  assert scene.metadata[8, 4] == scene.metadata[9, 4] == 0
  assert scene.metadata[-1, 2] == 0
  assert scene.hull_info[6, 4] == model.mesh_facenum[0]
  assert scene.hull_info[7, 6] == scene.hull_info[7, 7] == 3
  for value in vars(scene).values():
    if isinstance(value, np.ndarray):
      assert not value.flags.writeable
  assert scene.nbytes > 0


def test_ray_scene_memory_rejection_precedes_torch_import(monkeypatch):
  import builtins
  original = builtins.__import__
  def guarded(name, *args, **kwargs):
    if name == 'torch':
      raise AssertionError('device runtime imported before capacity rejection')
    return original(name, *args, **kwargs)
  model = _scene()
  monkeypatch.setattr(builtins, '__import__', guarded)
  with pytest.raises(ValueError, match='memory budget'):
    MetalRayQueries(model, 2, memory_budget_bytes=1)


def test_ray_scene_heightfield_packing_has_no_contact_grid_dimension_cap():
  model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
    <hfield name="large" nrow="67" ncol="3" size="1 1 .2 .1"/>
    </asset><worldbody><geom type="hfield" hfield="large"/></worldbody></mujoco>''')
  scene = lower_ray_scene(model)
  np.testing.assert_array_equal(scene.hull_info[0,6:8], [67,3])
  assert scene.hull.size == 4 + 67*3


def test_primitive_ray_capacity_rejection_precedes_device_runtime(monkeypatch):
  import builtins
  original = builtins.__import__
  def guarded(name, *args, **kwargs):
    if name == 'torch':
      raise AssertionError('primitive ray budget rejection imported device runtime')
    return original(name, *args, **kwargs)
  monkeypatch.setattr(builtins, '__import__', guarded)
  with pytest.raises(ValueError, match='memory budget'):
    MetalRayPrimitives(2, memory_budget_bytes=47)
  with pytest.raises(ValueError, match='int32'):
    MetalRayPrimitives(2**31, memory_budget_bytes=2**40)


_GPU = pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1',
                         reason='serialized native scene query gate')


@_GPU
@pytest.mark.parametrize('filter_mode', ['all', 'groups', 'dynamic', 'exclude'])
def test_native_scene_rays_match_pinned_geometry_filters_and_normals(filter_mode):
  torch = pytest.importorskip('torch')
  model = _scene()
  data = [mujoco.MjData(model) for _ in range(2)]
  for world, d in enumerate(data):
    d.qpos[0] = .1*world
    mujoco.mj_forward(model, d)
  origins = np.array([[g, .04, 2] for g in range(-2, 8)], np.float32)
  # Oblique and non-unit directions exercise the original parameter semantics.
  vectors = np.array([[.02, -.03, -2]]*len(origins), np.float32)
  origins, vectors = np.stack([origins]*2), np.stack([vectors]*2)
  groups = np.array([0, 1, 0, 0, 0, 0], np.uint8) if filter_mode=='groups' else None
  static = filter_mode != 'dynamic'
  exclude = 1 if filter_mode=='exclude' else -1
  tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), device='mps')
  poses = {'geom_pos': tensor(np.asarray([d.geom_xpos for d in data], np.float32)),
           'geom_quat': tensor(np.asarray([[
               _matrix_quat(mat) for mat in d.geom_xmat] for d in data], np.float32))}
  query = MetalRayQueries(model, 2)
  out = query.run_device(poses, tensor(origins), tensor(vectors),
                         geomgroup=groups, flg_static=static, bodyexclude=exclude)
  expected_distance = np.zeros((2,len(origins[0])))
  expected_id = np.zeros_like(expected_distance, dtype=np.int32)
  expected_normal = np.zeros((2,len(origins[0]),3))
  for w, d in enumerate(data):
    for i, (origin, vector) in enumerate(zip(origins[w],vectors[w])):
      gid, normal = np.array([-1],np.int32), np.zeros(3)
      expected_distance[w,i] = mujoco.mj_ray(model,d,origin.astype(float),
          vector.astype(float),groups,static,exclude,gid,normal)
      expected_id[w,i], expected_normal[w,i] = gid[0],normal
  np.testing.assert_array_equal(out['status'].cpu(), 0)
  np.testing.assert_array_equal(out['geomid'].cpu(), expected_id)
  np.testing.assert_allclose(out['distance'].cpu(), expected_distance,rtol=2e-5,atol=3e-6)
  np.testing.assert_allclose(out['normal'].cpu(), expected_normal,rtol=3e-5,atol=5e-6)


def _matrix_quat(matrix):
  out = np.zeros(4)
  mujoco.mju_mat2Quat(out, np.asarray(matrix, np.float64))
  return out


@_GPU
def test_native_ray_queries_isolate_bad_inputs_and_own_outputs():
  torch = pytest.importorskip('torch')
  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
  query = MetalRayQueries(model,2)
  poses = dict(geom_pos=torch.empty((2,0,3),device='mps'),
               geom_quat=torch.empty((2,0,4),device='mps'))
  origins = torch.zeros((2,2,3),device='mps')
  vectors = torch.tensor([[[0,0,1],[0,0,0]],[[0,0,1],[float('nan'),0,1]]],device='mps')
  result = query.run_device(poses,origins,vectors)
  np.testing.assert_array_equal(result['status'].cpu(),[[0,1],[0,1]])
  np.testing.assert_array_equal(result['distance'].cpu(),-1)
  np.testing.assert_array_equal(result['normal'].cpu(),0)
  saved = result['status'].clone()
  query.run_device(poses,origins,torch.ones_like(vectors))
  torch.testing.assert_close(result['status'],saved,rtol=0,atol=0)
  empty = query.run_device(poses,origins[:,:0].contiguous(),vectors[:,:0].contiguous())
  assert tuple(empty['distance'].shape)==(2,0)


@_GPU
@pytest.mark.parametrize('provided_dynamics', [False, True])
def test_public_ray_api_preserves_prepared_stages_and_checkpoint_trajectory(provided_dynamics):
  torch = pytest.importorskip('torch')
  from mujoco_metal import native_api
  from mujoco_metal.simulation import MetalSimulation
  model = _scene()
  sim = MetalSimulation(model, batch_size=2,
      qpos=np.asarray([[0.0],[0.1]],np.float32), profile='integrated_euler_v1')
  record = native_api.mj_fwdPosition(sim,return_record=True,skipsensor=True)
  dynamics = native_api.mj_fwdVelocity(sim,record=record,skipsensor=True)
  query_dynamics = dynamics if provided_dynamics else None
  original = {name:value.clone() for name,value in dynamics['poses'].items()}
  before = sim.snapshot()
  origins = np.asarray([[3,.04,2],[4,.04,2]],np.float32)
  vectors = np.asarray([[.02,-.03,-2],[.02,-.03,-2]],np.float32)
  first = native_api.mj_ray(sim,origins,vectors,dynamics=query_dynamics)
  multiple = native_api.mj_multiRay(sim,origins,vectors[:,None,:],dynamics=query_dynamics)
  for name,value in first.items():
    torch.testing.assert_close(value,multiple[name][:,0],rtol=0,atol=0)
  for name,value in original.items():
    torch.testing.assert_close(dynamics['poses'][name],value,rtol=0,atol=0)
  # Subsequent ACT/ACC consumes the same borrowed generation-scoped prefix.
  native_api.mj_fwdActuation(sim,record=record)
  native_api.mj_fwdAcceleration(sim,record=record)
  after=sim.snapshot()
  for name in ('qpos','qvel','time'):
    np.testing.assert_array_equal(getattr(after['device'],name),getattr(before['device'],name))
  for geomid,func in ((6,native_api.mj_rayMesh),(7,native_api.mj_rayHfield)):
    result=func(sim,geomid,origins,vectors,dynamics=query_dynamics)
    for world in range(2):
      d=mujoco.MjData(model);d.qpos[:]=[.1*world];mujoco.mj_forward(model,d)
      normal=np.zeros(3)
      cpu=mujoco.mj_rayMesh if geomid==6 else mujoco.mj_rayHfield
      expected=cpu(model,d,geomid,origins[world].astype(float),vectors[world].astype(float),normal)
      np.testing.assert_allclose(result['distance'][world].cpu(),expected,rtol=2e-5,atol=3e-6)
      np.testing.assert_allclose(result['normal'][world].cpu(),normal,rtol=3e-5,atol=5e-6)


@_GPU
def test_native_scene_plugin_sdf_rays_match_pinned_distance_and_normal():
  torch = pytest.importorskip('torch')
  names=('bolt','bowl','gear','nut','torus')
  extension=''.join(f'<plugin plugin="mujoco.sdf.{name}"><instance name="p{i}"/></plugin>'
                    for i,name in enumerate(names))
  assets=''.join(f'<mesh name="m{i}"><plugin instance="p{i}"/></mesh>' for i in range(5))
  geoms=''.join(f'<geom type="sdf" mesh="m{i}" pos="{4*i} 0 0"><plugin instance="p{i}"/></geom>'
               for i in range(5))
  model=mujoco.MjModel.from_xml_string(f'<mujoco><extension>{extension}</extension>'
      f'<asset>{assets}</asset><worldbody>{geoms}</worldbody></mujoco>')
  data=mujoco.MjData(model);mujoco.mj_forward(model,data)
  origins=np.asarray([[[4*i+.19,.07,2] for i in range(5)]],np.float32)
  vectors=np.asarray([[[.015,-.005,-2] for _ in range(5)]],np.float32)
  tensor=lambda value:torch.as_tensor(np.asarray(value).copy(),device='mps')
  poses={'geom_pos':tensor(data.geom_xpos.astype(np.float32)[None]),
         'geom_quat':tensor(np.asarray([[_matrix_quat(mat) for mat in data.geom_xmat]],np.float32))}
  output=MetalRayQueries(model,1).run_device(poses,tensor(origins),tensor(vectors))
  for i in range(5):
    gid=np.array([-1],np.int32);normal=np.zeros(3)
    expected=mujoco.mj_ray(model,data,origins[0,i].astype(float),vectors[0,i].astype(float),
                           None,True,-1,gid,normal)
    assert int(output['status'][0,i].cpu())==0
    assert int(output['geomid'][0,i].cpu())==gid[0]
    np.testing.assert_allclose(output['distance'][0,i].cpu(),expected,rtol=2e-5,atol=5e-6)
    np.testing.assert_allclose(output['normal'][0,i].cpu(),normal,rtol=2e-4,atol=3e-5)


@_GPU
def test_native_multi_ray_cutoff_uses_pinned_bounding_sphere_elimination():
  torch=pytest.importorskip('torch')
  model=_scene();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
  origin=np.asarray([[0,.04,2]],np.float32)
  vectors=np.asarray([[[.02,-.03,-2],[2,0,-2],[0,0,-1]]],np.float32)
  tensor=lambda value:torch.as_tensor(np.asarray(value).copy(),device='mps')
  poses={'geom_pos':tensor(data.geom_xpos.astype(np.float32)[None]),
         'geom_quat':tensor(np.asarray([[_matrix_quat(mat) for mat in data.geom_xmat]],np.float32))}
  query=MetalRayQueries(model,1)
  for cutoff in (.5,1.3,4.0):
    expected_id=np.full(3,-1,np.int32);distance=np.full(3,-1.0);normal=np.zeros((3,3))
    mujoco.mj_multiRay(model,data,origin[0].astype(float),vectors[0].astype(float).reshape(-1),
        None,True,-1,expected_id,distance,normal.reshape(-1),3,cutoff)
    actual=query.run_device(poses,tensor(np.tile(origin[:,None],(1,3,1))),tensor(vectors),cutoff=cutoff)
    np.testing.assert_array_equal(actual['status'].cpu(),0)
    np.testing.assert_array_equal(actual['geomid'].cpu()[0],expected_id)
    np.testing.assert_allclose(actual['distance'].cpu()[0],distance,atol=3e-6,rtol=2e-5)
    np.testing.assert_allclose(actual['normal'].cpu()[0],normal,atol=5e-6,rtol=3e-5)


@_GPU
def test_native_nonconvex_mesh_ray_keeps_hole_and_exceeds_contact_vertex_cap():
  torch=pytest.importorskip('torch')
  vertices=[];faces=[]
  for i in range(16):
    u=2*np.pi*i/16
    for j in range(8):
      v=2*np.pi*j/8
      vertices.append(((.7+.2*np.cos(v))*np.cos(u),
                       (.7+.2*np.cos(v))*np.sin(u),.2*np.sin(v)))
      a=8*i+j;b=8*((i+1)%16)+j;c=8*((i+1)%16)+(j+1)%8;d=8*i+(j+1)%8
      faces.extend(((a,b,c),(a,c,d)))
  vertex_text=' '.join(str(x) for row in vertices for x in row)
  face_text=' '.join(str(x) for row in faces for x in row)
  model=mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="ring" vertex="{vertex_text}" face="{face_text}"/>
      </asset><worldbody><geom type="mesh" mesh="ring" contype="0" conaffinity="0"/>
      </worldbody></mujoco>''')
  data=mujoco.MjData(model);mujoco.mj_forward(model,data)
  assert model.mesh_vertnum[0]==128
  origins=np.asarray([[[0,0,2],[.67,.08,2],[-.6,.15,2]]],np.float32)
  vectors=np.asarray([[[0,0,-1]]*3],np.float32)
  tensor=lambda value:torch.as_tensor(np.asarray(value).copy(),device='mps')
  poses={'geom_pos':tensor(data.geom_xpos.astype(np.float32)[None]),
         'geom_quat':tensor(np.asarray([[_matrix_quat(mat) for mat in data.geom_xmat]],np.float32))}
  out=MetalRayQueries(model,1).run_device(poses,tensor(origins),tensor(vectors))
  for i in range(3):
    gid=np.array([-1],np.int32);normal=np.zeros(3)
    expected=mujoco.mj_ray(model,data,origins[0,i].astype(float),vectors[0,i].astype(float),
                           None,True,-1,gid,normal)
    if i==0:
      assert expected==-1 and gid[0]==-1
    else:
      assert expected>0 and gid[0]==0
    assert int(out['status'][0,i].cpu())==0
    assert int(out['geomid'][0,i].cpu())==gid[0]
    np.testing.assert_allclose(out['distance'][0,i].cpu(),expected,rtol=2e-5,atol=3e-6)
    np.testing.assert_allclose(out['normal'][0,i].cpu(),normal,rtol=3e-5,atol=5e-6)


@_GPU
def test_native_ray_small_direction_thresholds_follow_pinned_primitive_and_multi_rules():
  torch = pytest.importorskip('torch')
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="plane" size="0 0 .1"/>
    <geom type="sphere" size=".25" pos="3 0 .5"/>
    </worldbody></mujoco>''')
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), device='mps')
  poses = dict(geom_pos=tensor(data.geom_xpos.astype(np.float32)[None]),
      geom_quat=tensor(np.asarray([[_matrix_quat(mat) for mat in data.geom_xmat]], np.float32)))
  origins = np.asarray([[[0,0,1], [3,0,1], [0,0,1]]], np.float32)
  vectors = np.asarray([[[0,0,-2e-13], [0,0,-1e-9], [0,0,-1e-6]]], np.float32)
  query = MetalRayQueries(model, 1)
  single = query.run_device(poses, tensor(origins), tensor(vectors))
  for i in range(3):
    gid, normal = np.array([-1], np.int32), np.zeros(3)
    expected = mujoco.mj_ray(model, data, origins[0,i].astype(float),
        vectors[0,i].astype(float), None, True, -1, gid, normal)
    np.testing.assert_allclose(single['distance'][0,i].cpu(), expected, rtol=3e-6, atol=1e-6)
    np.testing.assert_array_equal(single['geomid'][0,i].cpu(), gid[0])
    np.testing.assert_allclose(single['normal'][0,i].cpu(), normal, rtol=0, atol=1e-6)
  np.testing.assert_array_equal(single['status'].cpu(), 0)
  # Each upstream multiRay call shares one origin; run independent one-ray
  # calls to cover each origin without changing the pinned API contract.
  multiple = query.run_device(poses, tensor(origins), tensor(vectors), multiray=True)
  for i in range(3):
    gid, distance, normal = np.array([-1], np.int32), np.array([-1.0]), np.zeros(3)
    mujoco.mj_multiRay(model, data, origins[0,i].astype(float), vectors[0,i].astype(float),
        None, True, -1, gid, distance, normal, 1, 1e14)
    np.testing.assert_allclose(multiple['distance'][0,i].cpu(), distance[0], rtol=3e-6, atol=1e-6)
  np.testing.assert_array_equal(multiple['status'].cpu(), [[1,1,0]])


@_GPU
def test_native_heightfield_side_hits_use_raised_top_box():
  torch = pytest.importorskip('torch')
  model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
    <hfield name="plateau" nrow="3" ncol="3" size=".5 .5 .4 .1"/>
    </asset><worldbody><geom type="hfield" hfield="plateau"/></worldbody></mujoco>''')
  model.hfield_data[:] = [.8, .9, 1, .8, .9, 1, .8, .9, 1]
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), device='mps')
  poses = dict(geom_pos=tensor(data.geom_xpos.astype(np.float32)[None]),
      geom_quat=tensor(np.asarray([[_matrix_quat(mat) for mat in data.geom_xmat]], np.float32)))
  origins = np.asarray([[[.8,.1,.3], [-.8,.1,.25], [.2,.8,.3], [.2,-.8,.3]]], np.float32)
  vectors = np.asarray([[[-1,0,0], [1,0,0], [0,-1,0], [0,1,0]]], np.float32)
  output = MetalRayQueries(model, 1).run_device(poses, tensor(origins), tensor(vectors))
  for i in range(4):
    normal = np.zeros(3)
    expected = mujoco.mj_rayHfield(model, data, 0, origins[0,i].astype(float),
                                 vectors[0,i].astype(float), normal)
    assert expected > 0
    np.testing.assert_array_equal(output['status'][0,i].cpu(), 0)
    np.testing.assert_allclose(output['distance'][0,i].cpu(), expected, rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(output['normal'][0,i].cpu(), normal, rtol=0, atol=1e-6)


@_GPU
def test_native_ray_budget_accounts_for_parameter_bindings_before_output_allocation(monkeypatch):
  torch = pytest.importorskip('torch')
  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
  scene = lower_ray_scene(model)
  # One ray's outputs plus every binding, with one byte deliberately missing.
  query = MetalRayQueries(model, 1, memory_budget_bytes=scene.nbytes+24+64-1)
  poses = dict(geom_pos=torch.empty((1,0,3), device='mps'),
               geom_quat=torch.empty((1,0,4), device='mps'))
  origins, vectors = torch.zeros((1,1,3), device='mps'), torch.ones((1,1,3), device='mps')
  def forbidden_allocation(*args, **kwargs):
    raise AssertionError('output allocation preceded capacity rejection')
  monkeypatch.setattr(torch, 'empty', forbidden_allocation)
  with pytest.raises(ValueError, match='memory budget'):
    query.run_device(poses, origins, vectors)


@_GPU
@pytest.mark.parametrize('geomtype', [0,2,3,4,5,6])
def test_native_primitive_ray_utility_matches_pinned_explicit_poses(geomtype):
  torch = pytest.importorskip('torch')
  quat = np.asarray([.81,.19,-.31,.37], np.float64)
  quat /= np.linalg.norm(quat)
  rotation = np.zeros(9)
  mujoco.mju_quat2Mat(rotation, quat)
  rotation = rotation.reshape(3,3).astype(np.float32)
  pos = np.asarray([[.1,.2,.3], [-.7,.4,.6], [.2,-.3,1]], np.float32)
  mat = np.stack([rotation]*3)
  size = np.asarray([[.4,.3,.2]]*3, np.float32)
  local_origins = np.asarray([[.06,.04,2], [0,0,0], [4,4,2]], np.float32)
  local_vectors = np.asarray([[0,0,-2], [1,.2,.1], [1,1,1]], np.float32)
  origins = np.einsum('bij,bj->bi',mat,local_origins)+pos
  vectors = np.einsum('bij,bj->bi',mat,local_vectors)
  tensor = lambda value: torch.as_tensor(value.copy(),device='mps')
  query = MetalRayPrimitives(3)
  out = query.run_device(tensor(pos),tensor(mat),tensor(size),
                         tensor(origins),tensor(vectors),geomtype)
  np.testing.assert_array_equal(out['status'].cpu(),0)
  for i in range(3):
    normal = np.zeros(3)
    expected = mujoco.mju_rayGeom(pos[i].astype(float),mat[i].astype(float).reshape(-1),
        size[i].astype(float),origins[i].astype(float),vectors[i].astype(float),geomtype,normal)
    np.testing.assert_allclose(out['distance'][i].cpu(),expected,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(out['normal'][i].cpu(),normal,rtol=3e-5,atol=3e-6)
  saved = out['distance'].clone()
  query.run_device(tensor(pos),tensor(mat),tensor(size),tensor(origins+5),tensor(vectors),geomtype)
  torch.testing.assert_close(out['distance'],saved,rtol=0,atol=0)


@_GPU
def test_public_primitive_ray_utility_isolates_invalid_input_and_preserves_state():
  torch = pytest.importorskip('torch')
  from mujoco_metal import native_api
  from mujoco_metal.simulation import MetalSimulation
  model = _scene()
  sim = MetalSimulation(model,batch_size=2,profile='integrated_euler_v1')
  before = sim.snapshot()
  versions = (sim.state._qpos._version,sim.state._qvel._version,sim.state._time._version)
  pos,mat,size = np.zeros((2,3),np.float32),np.stack([np.eye(3,dtype=np.float32)]*2),np.ones((2,3),np.float32)
  origins = np.asarray([[0,0,2],[0,0,2]],np.float32)
  vectors = np.asarray([[0,0,-1],[float('nan'),0,-1]],np.float32)
  out = native_api.mju_rayGeom(sim,pos,mat,size,origins,vectors,2)
  np.testing.assert_array_equal(out['status'].cpu(),[0,1])
  np.testing.assert_allclose(out['distance'].cpu(),[1,-1],rtol=0,atol=0)
  np.testing.assert_allclose(out['normal'].cpu(),[[0,0,1],[0,0,0]],rtol=0,atol=0)
  with pytest.raises(ValueError,match='geomtype'):
    native_api.mju_rayGeom(sim,pos,mat,size,origins,vectors,1)
  with pytest.raises(ValueError,match='shape'):
    native_api.mju_rayGeom(sim,pos,mat,size,origins,vectors[:1],2)
  after = sim.snapshot()
  assert versions==(sim.state._qpos._version,sim.state._qvel._version,sim.state._time._version)
  for name in ('qpos','qvel','qacc','time'):
    np.testing.assert_array_equal(getattr(before['device'],name),getattr(after['device'],name))
