# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public source geometry distance owns its pipeline independently of step/sensors."""
import builtins
from dataclasses import replace
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
from mujoco_metal.geometry_queries import lower_geometry_distance,GeometryDistanceQuery


def _model(kind='sphere',libccd=False):
  flag='<flag nativeccd="disable"/>' if libccd else ''
  vertices='-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 -.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1'
  geom=('<geom name="moving" type="mesh" mesh="m"/>' if kind=='mesh' else
        '<geom name="moving" type="sdf" mesh="m"/>' if kind=='sdf' else
        f'<geom name="moving" type="{kind}" size=".1 .07 .1"/>' if kind=='ellipsoid' else
        '<geom name="moving" type="sphere" size=".1"/>')
  return mujoco.MjModel.from_xml_string(f'<mujoco><option gravity="0 0 0" sdf_initpoints="8">{flag}</option><default><geom contype="0" conaffinity="0"/></default><asset><mesh name="m" vertex="{vertices}"/></asset><worldbody><geom name="fixed" type="sphere" size=".1"/><body pos=".25 0 0"><freejoint/>{geom}</body></worldbody></mujoco>')

@pytest.mark.parametrize('kind',['sphere','mesh','ellipsoid','sdf'])
def test_geometry_distance_preflight_counts_all_owned_backings_before_device_import(kind,monkeypatch):
  model=_model(kind);desc=lower_geometry_distance(model,2,0,1,.5)
  assert desc['bytes_total']==4*sum(desc['workspace_sizes'].values())
  assert 'fk.position_words' in desc['workspace_sizes']
  original=builtins.__import__
  def guarded(name,*args,**kwargs):
    if name=='torch':raise AssertionError('GPU import before budget admission')
    return original(name,*args,**kwargs)
  monkeypatch.setattr(builtins,'__import__',guarded)
  with pytest.raises(CapacityOverflow,match='geometry-distance workspace'):
    GeometryDistanceQuery(model,2,0,1,.5,
        limits=replace(CapacityLimits(),memory_budget_bytes=desc['bytes_total']-1))

@pytest.mark.parametrize('ids',[(-1,1),(0,2),(True,1),(0,.5)])
def test_invalid_distance_ids_rejected_on_host(ids):
  with pytest.raises(ValueError,match='compiled geom'):lower_geometry_distance(_model(),2,*ids,.5)

@pytest.mark.parametrize('bound',[np.nan,np.inf,-np.inf,True,'1',1+1j,1e100])
def test_invalid_distance_bound_rejected_on_host(bound):
  with pytest.raises(ValueError):lower_geometry_distance(_model(),2,0,1,bound)

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native public geometry query')
@pytest.mark.parametrize('kind',['sphere','mesh','ellipsoid','sdf'])
@pytest.mark.parametrize('libccd',[False,True])
def test_public_distance_source_oracle_order_bounds_and_replay(kind,libccd):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_geomDistance
  model=_model(kind,libccd)
  q=np.tile(model.qpos0.astype(np.float32),(3,1));q[:,0]=[.0625,.25,2]
  # Unique skew witnesses qualify source-order endpoints; the exactly
  # symmetric EPA case has its own independent surface-distance gate below.
  q[:,1:3]=[.03125,.0625]
  # Standalone queries remain available with stored sensors disabled.
  model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  sim=MetalSimulation(model,3,qpos=q,profile='integrated_euler_v1')
  refs=[mujoco.MjData(model) for _ in range(3)]
  for world,ref in enumerate(refs):ref.qpos[:]=q[world];mujoco.mj_forward(model,ref)
  snapshot=sim.snapshot()
  initial_words=sim.state._precision.words.clone()
  prior=sim.assembled_system(recompute=True)
  prior_values={name:value.clone() for name,value in prior.items() if isinstance(value,torch.Tensor)}
  witnesses=[]
  for bound in (.5,.03125,-.03125):
    for a,b in ((0,1),(1,0)):
      actual=mj_geomDistance(sim,a,b,bound)
      assert actual['status'].dtype==torch.int32
      # Independently count physical backings including output aliases once.
      backings={};seen=set();program=sim._geometry_distance_query[1]
      def visit(value):
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,torch.Tensor):
          allocation=value.untyped_storage()
          backings[allocation.data_ptr()]=allocation.nbytes()
        elif isinstance(value,dict):
          for child in value.values():visit(child)
        elif isinstance(value,(tuple,list)):
          for child in value:visit(child)
        elif type(value).__module__.startswith('mujoco_metal.') and hasattr(value,'__dict__'):
          for child in vars(value).values():visit(child)
      visit(program);visit(actual)
      assert sum(backings.values())==4*sum(program.workspace_sizes.values())
      expected=[];fromto=[]
      for ref in refs:
        witness=np.zeros(6)
        expected.append(mujoco.mj_geomDistance(model,ref,a,b,bound,witness));fromto.append(witness)
      np.testing.assert_allclose(actual['distance'].cpu().numpy(),expected,atol=3e-5,rtol=3e-5)
      np.testing.assert_allclose(actual['fromto'].cpu().numpy(),fromto,atol=3e-5,rtol=3e-5)
      np.testing.assert_array_equal(actual['status'].cpu().numpy(),0)
      witnesses.append((actual,actual['fromto'].clone()))
  for result,expected in witnesses:torch.testing.assert_close(result['fromto'],expected,rtol=0,atol=0)
  for name,expected in prior_values.items():torch.testing.assert_close(prior[name],expected,rtol=0,atol=0)
  torch.testing.assert_close(sim.state._precision.words,initial_words,rtol=0,atol=0)
  sim.restore(snapshot)
  restored=mj_geomDistance(sim,0,1,.5)
  torch.testing.assert_close(restored['fromto'],witnesses[0][0]['fromto'],rtol=0,atol=0)
  sim.step();sim.restore(snapshot)
  torch.testing.assert_close(mj_geomDistance(sim,0,1,.5)['fromto'],restored['fromto'],rtol=0,atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native degenerate EPA witness qualification')
def test_symmetric_mesh_sphere_witnesses_satisfy_independent_surface_bounds():
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_geomDistance
  model=_model('mesh',False)
  q=model.qpos0.astype(np.float32)[None];q[0,0]=.0625
  sim=MetalSimulation(model,1,qpos=q,profile='integrated_euler_v1')
  ref=mujoco.MjData(model);ref.qpos[:]=q[0];mujoco.mj_forward(model,ref)
  # Source EPA stops within ccd_tolerance. At an axis-symmetric face its
  # approximate tangential witness is not unique: the captured source and
  # native endpoints differ by .000214 tangentially while both radial errors
  # are <3e-7. Check each against the independent sphere/box surfaces and
  # exact signed depth, retaining the strict unique-witness matrix above.
  source=np.zeros(6);source_distance=mujoco.mj_geomDistance(model,ref,0,1,.5,source)
  result=mj_geomDistance(sim,0,1,.5)
  native=result['fromto'][0].cpu().numpy().astype(float)
  np.testing.assert_allclose(result['distance'][0].cpu().item(),-.1375,atol=model.opt.ccd_tolerance,rtol=0)
  np.testing.assert_allclose(source_distance,-.1375,atol=model.opt.ccd_tolerance,rtol=0)
  for witness in (source,native):
    assert abs(np.linalg.norm(witness[:3])-.1)<model.opt.ccd_tolerance
    local=witness[3:]-np.array([.0625,0,0])
    assert np.max(abs(local))<=.1+model.opt.ccd_tolerance
    assert abs(local[0]+.1)<model.opt.ccd_tolerance
    assert abs(np.linalg.norm(witness[3:]-witness[:3])-.1375)<model.opt.ccd_tolerance
  np.testing.assert_array_equal(result['status'].cpu().numpy(),0)
