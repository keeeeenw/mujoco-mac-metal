# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned engine-oracle gates for native position manifold utilities."""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_metal.position_queries import MetalPositionQueries, lower_position_topology

_GPU = pytest.mark.skipif(os.environ.get('MUJOCO_METAL_RUN_GPU') != '1',
                         reason='serialized MPS position query gate')


def _model():
  return mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <default><geom contype="0" conaffinity="0" size=".1" mass="1"/></default>
    <worldbody>
      <body pos="0 0 1"><freejoint/><geom/></body>
      <body pos="1 0 1"><joint type="ball"/><geom/></body>
      <body pos="2 0 1"><joint type="hinge"/><geom/></body>
      <body pos="3 0 1"><joint type="slide"/><geom/></body>
    </worldbody></mujoco>''')


def _inputs(model):
  q = np.tile(model.qpos0, (3,1)).astype(np.float32)
  q[:, :3] = [[.12,-.24,.36],[-.4,.1,.2],[.7,-.6,.5]]
  q[0,3:7] = [.87,.22,-.3,.1]
  q[1,3:7] = [-.7,.11,.23,-.36]
  q[2,3:7] = 0
  q[0,7:11] = [1.,.3,-.21,.09]
  q[1,7:11] = [1e-20,0,0,0]
  q[2,7:11] = [.0001,.99,-.02,.03]
  q[:,11:] = [[.3,-.1],[-.7,.2],[.1,.4]]
  v = np.linspace(-.3,.5,3*model.nv,dtype=np.float32).reshape(3,model.nv)
  return q,v


def test_position_topology_is_complete_owned_and_budgeted_before_mps_allocation():
  model = _model()
  topology = lower_position_topology(model)
  assert (topology.nq,topology.nv) == (13,11)
  np.testing.assert_array_equal(topology.joints[:,0],[0,1,3,2])
  with pytest.raises(ValueError):
    topology.joints[0,0] = 99
  with pytest.raises(ValueError,match='before device allocation'):
    MetalPositionQueries(model,3,memory_budget_bytes=1)
  for batch in (0,-1,True,1.5):
    with pytest.raises(ValueError):
      MetalPositionQueries(model,batch)
  with pytest.raises(ValueError,match='signed int32'):
    MetalPositionQueries(model,2**31)
  fake = SimpleNamespace(nq=1,nv=1,njnt=1,jnt_type=np.array([3]),
      jnt_qposadr=np.array([1]),jnt_dofadr=np.array([0]))
  with pytest.raises(ValueError,match='joint addresses'):
    lower_position_topology(fake)


@_GPU
@pytest.mark.parametrize('dt',[0.,.02,-.02,1.,-1.])
def test_native_integrate_and_normalize_match_every_joint_and_pinned_zero_quat(dt):
  torch = pytest.importorskip('torch')
  model = _model()
  q,v = _inputs(model)
  t = lambda a:torch.as_tensor(a.copy(),device='mps')
  program = MetalPositionQueries(model,3)
  out = program.integrate(t(q),t(v),dt)
  expected = q.astype(float)
  for w in range(3):
    mujoco.mj_integratePos(model,expected[w],v[w].astype(float),dt)
  np.testing.assert_array_equal(out['status'].cpu(),0)
  np.testing.assert_allclose(out['qpos'].cpu(),expected,rtol=3e-6,atol=3e-7)
  normalized = program.normalize(t(q))
  expected = q.astype(float)
  for row in expected:
    mujoco.mj_normalizeQuat(model,row)
  np.testing.assert_allclose(normalized['qpos'].cpu(),expected,rtol=3e-6,atol=3e-7)
  np.testing.assert_array_equal(normalized['status'].cpu(),0)


@_GPU
@pytest.mark.parametrize('dt',[.02,-.02,.5])
def test_native_differentiate_uses_local_quaternion_tangent_and_shortest_rotation(dt):
  torch = pytest.importorskip('torch')
  model = _model()
  q,v = _inputs(model)
  normalized = q.astype(float)
  for row in normalized:
    mujoco.mj_normalizeQuat(model,row)
  q = normalized.astype(np.float32)
  q2 = q.astype(float)
  for w in range(3):
    mujoco.mj_integratePos(model,q2[w],v[w].astype(float),dt)
  q2 = q2.astype(np.float32)
  # Quaternion sign must not change orientation difference.
  q2[1,3:7] *= -1
  q2[2,7:11] *= -1
  expected = np.zeros_like(v,dtype=float)
  for w in range(3):
    mujoco.mj_differentiatePos(model,expected[w],dt,q[w].astype(float),q2[w].astype(float))
  t = lambda a:torch.as_tensor(a.copy(),device='mps')
  program = MetalPositionQueries(model,3)
  out = program.differentiate(t(q),t(q2),dt)
  np.testing.assert_array_equal(out['status'].cpu(),0)
  np.testing.assert_allclose(out['qvel'].cpu(),expected,rtol=3e-5,atol=5e-6)
  # Equal positions must yield zero, including non-unit and zero quaternions.
  original,_ = _inputs(model)
  same = program.differentiate(t(original),t(original),dt)
  np.testing.assert_allclose(same['qvel'].cpu(),0,rtol=0,atol=5e-6)


@_GPU
def test_position_queries_empty_model_invalid_rows_admission_and_owned_outputs():
  torch = pytest.importorskip('torch')
  empty = mujoco.MjModel.from_xml_string('<mujoco/>')
  program = MetalPositionQueries(empty,2)
  zero = torch.empty((2,0),dtype=torch.float32,device='mps')
  for out in (program.normalize(zero),program.integrate(zero,zero,0),
              program.differentiate(zero,zero,-1)):
    assert next(iter(out.values())).shape == (2,0)
    np.testing.assert_array_equal(out['status'].cpu(),0)
  model = _model()
  q,v = _inputs(model)
  program = MetalPositionQueries(model,3)
  tq,tv = torch.as_tensor(q,device='mps'),torch.as_tensor(v,device='mps')
  saved = program.integrate(tq,tv,.02)
  copies = {k:x.clone() for k,x in saved.items()}
  program.normalize(tq)
  program.differentiate(tq,tq,1)
  for k,x in copies.items():
    torch.testing.assert_close(saved[k],x,rtol=0,atol=0)
  torch.testing.assert_close(tq.cpu(),torch.from_numpy(q),rtol=0,atol=0)
  bad = tq.clone()
  bad[1,2] = float('nan')
  result = program.integrate(bad,tv,.02)
  np.testing.assert_array_equal(result['status'].cpu(),[0,1,0])
  torch.testing.assert_close(result['qpos'][1],bad[1],rtol=0,atol=0,equal_nan=True)
  huge = tv.clone()
  huge[1,0] = float(np.finfo(np.float32).max)
  result = program.integrate(tq,huge,2.)
  np.testing.assert_array_equal(result['status'].cpu(),[0,2,0])
  torch.testing.assert_close(result['qpos'][1],tq[1],rtol=0,atol=0)
  for dt in (0,float('nan'),float('inf'),1e-60,True):
    with pytest.raises(ValueError):
      program.differentiate(tq,tq,dt)
  with pytest.raises(ValueError):
    program.normalize(tq.cpu())
  with pytest.raises(ValueError):
    program.normalize(torch.zeros((3,model.nq*2),device='mps')[:,::2])


@_GPU
def test_public_position_queries_preserve_simulation_and_prepared_records():
  torch = pytest.importorskip('torch')
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.native_api import (mj_fwdPosition,mj_integratePos,
      mj_differentiatePos,mj_normalizeQuat)
  model = _model()
  sim = MetalSimulation(model,batch_size=1,profile='integrated_scalable_v1')
  record = mj_fwdPosition(sim,return_record=True,skipsensor=True)
  before = sim.state._qpos.clone()
  velocity = np.linspace(-.1,.1,model.nv,dtype=np.float32)[None]
  out = mj_integratePos(sim,before,velocity,.02)
  tangent = mj_differentiatePos(sim,before,out['qpos'],.02)
  expected = np.empty(model.nv)
  mujoco.mj_differentiatePos(model,expected,.02,
      before[0].cpu().numpy().astype(float),out['qpos'][0].cpu().numpy().astype(float))
  np.testing.assert_allclose(tangent['qvel'][0].cpu(),expected,rtol=3e-5,atol=5e-6)
  mj_normalizeQuat(sim,out['qpos'])
  assert sim._forward_stages._record is record
  torch.testing.assert_close(sim.state._qpos,before,rtol=0,atol=0)
