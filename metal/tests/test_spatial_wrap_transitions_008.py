# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual sphere/cylinder wrap transitions, sides, one-sided slopes and replay."""
import os
from pathlib import Path
import mujoco
import numpy as np
import pytest


def _model(kind,side):
  size='.1' if kind=='sphere' else '.1 .2'
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".0005" gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><geom name="obstacle" type="{kind}" size="{size}"/>
      <site name="side" pos="0 {side*.2} .03"/>
      <body><joint name="y" type="slide" axis="0 1 0"/><geom size=".03" mass="1"/>
        <site name="a" pos="-.3 0 .03"/><site name="b" pos=".3 0 .03"/>
      </body></worldbody><tendon><spatial name="rope" stiffness="2" damping=".1"
          springlength=".59 .6"><site site="a"/><geom geom="obstacle"
          sidesite="side"/><site site="b"/></spatial></tendon></mujoco>''')


def _oracle(model,q,v=.07):
  data=mujoco.MjData(model);data.qpos[0]=q;data.qvel[0]=v
  mujoco.mj_forward(model,data)
  return data


def _row(model,data):
  return float(data.ten_J[model.ten_J_rowadr[0]]) if model.ten_J_rownnz[0] else 0.


@pytest.mark.parametrize('kind',['sphere','cylinder'])
@pytest.mark.parametrize('side',[-1,1])
def test_wrap_sides_transition_and_independent_local_slopes(kind,side):
  model=_model(kind,side);boundary=np.sqrt(.1**2-.03**2) if kind=='sphere' else .1
  qs=side*np.asarray([.02,boundary-2e-4,boundary+2e-4,.14])
  counts=[]
  for q in qs:
    data=_oracle(model,q);counts.append(int(data.ten_wrapnum[0]))
    h=1e-6
    centered=(_oracle(model,q+h).ten_length[0]-_oracle(model,q-h).ten_length[0])/(2*h)
    np.testing.assert_allclose(_row(model,data),centered,atol=2e-6,rtol=2e-5)
  assert counts[0]>2 and counts[1]>2 and counts[2]==counts[3]==2,counts
  # Each boundary is sampled from its own side, never across its branch switch.
  before=_oracle(model,side*boundary-side*2e-4)
  after=_oracle(model,side*boundary+side*2e-4)
  for data,direction in [(before,-side),(after,side)]:
    shifted=_oracle(model,float(data.qpos[0])+direction*1e-7)
    one_sided=(shifted.ten_length[0]-data.ten_length[0])/(direction*1e-7)
    np.testing.assert_allclose(_row(model,data),one_sided,atol=2e-6,rtol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native wrap transitions')
@pytest.mark.parametrize('kind',['sphere','cylinder'])
@pytest.mark.parametrize('side',[-1,1])
def test_native_wrap_branch_motion_and_full_checkpoint_replay(kind,side,monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'examples'))
  from magnetic_crane import _same_checkpoint
  model=_model(kind,side);boundary=np.sqrt(.1**2-.03**2) if kind=='sphere' else .1
  q=side*np.asarray([[.02],[boundary-2e-4],[boundary+2e-4],[.14]],np.float32)
  v=np.asarray([[.03],[-.07],[.05],[-.02]],np.float32)
  refs=[_oracle(model,q[w,0],v[w,0]) for w in range(4)]
  sim=MetalSimulation(model,4,profile='integrated_euler_v1');sim.reset(qpos=q,qvel=v)
  def geometry():
    dynamics=sim._smooth.run_device(sim.state._qpos,sim.state._qvel)
    return sim._spatial_tendons.run_kinematics(sim.state._qvel,dynamics['poses'])
  with monkeypatch.context() as context:
    def forbidden(*_args,**_kwargs):raise AssertionError('native wrap used CPU physics')
    for name in ('mj_forward','mj_tendon','mj_kinematics'):
      context.setattr(mujoco,name,forbidden)
    kin=geometry()
  lengths=kin['length'].cpu().numpy()[:,0]
  for world,data in enumerate(refs):
    np.testing.assert_allclose(lengths[world],data.ten_length[0],atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(kin['jacobian'][world,0,0].cpu().numpy(),_row(model,data),
                               atol=2e-6,rtol=2e-5)
  assert lengths[0]>.6001
  np.testing.assert_allclose(lengths[2:],.6,atol=2e-6,rtol=0)
  # Reset swaps wrapped/unwrapped worlds; no old branch can survive.
  sim.reset(qpos=q[::-1].copy(),qvel=v[::-1].copy())
  changed=geometry()
  np.testing.assert_allclose(changed['length'].cpu().numpy()[:,0],lengths[::-1],atol=2e-6)
  live=[_oracle(model,q[3-w,0],v[3-w,0]) for w in range(4)]
  checkpoint=sim.snapshot()
  for step in range(6):
    sim.step()
    for world,data in enumerate(live):
      mujoco.mj_step(model,data)
      for field in ("qpos","qvel","qacc"):
        np.testing.assert_allclose(getattr(sim.state,field)[world].cpu().numpy(),
                                   getattr(data,field),rtol=3e-5,atol=3e-6)
  expected=sim.snapshot()
  sim.restore(checkpoint);sim.step(6)
  _same_checkpoint(expected,sim.snapshot())


def _inside_model(kind):
  model=_model(kind,1)
  model.site_pos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'side')]=[0,0,.01]
  model.site_pos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'a')]=[.3,-.12,.03]
  model.site_pos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'b')]=[.4,.12,.04]
  mujoco.mj_setConst(model,mujoco.MjData(model))
  return model


@pytest.mark.parametrize('kind',['sphere','cylinder'])
def test_inside_side_oracle_has_actual_routed_contacts(kind):
  model=_inside_model(kind)
  for q in (.0,.05,.18):
    data=_oracle(model,q)
    assert data.ten_wrapnum[0]==4
    assert data.ten_length[0]>.26
    assert np.isfinite(_row(model,data))
    h=1e-5
    fd=(_oracle(model,q+h).ten_length[0]-_oracle(model,q-h).ten_length[0])/(2*h)
    np.testing.assert_allclose(_row(model,data),fd,rtol=2e-5,atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native inside wraps')
@pytest.mark.parametrize('kind',['sphere','cylinder'])
def test_native_inside_side_geometry_and_replay(kind,monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'examples'))
  from magnetic_crane import _same_checkpoint
  model=_inside_model(kind);q=np.asarray([[0],[.05],[.18]],np.float32)
  v=np.asarray([[.04],[-.05],[.03]],np.float32)
  refs=[_oracle(model,q[w,0],v[w,0]) for w in range(3)]
  sim=MetalSimulation(model,3,profile='integrated_euler_v1');sim.reset(qpos=q,qvel=v)
  dynamics=sim._smooth.run_device(sim.state._qpos,sim.state._qvel)
  kin=sim._spatial_tendons.run_kinematics(sim.state._qvel,dynamics['poses'])
  for world,data in enumerate(refs):
    np.testing.assert_allclose(kin['length'][world,0].cpu().numpy(),data.ten_length[0],rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(kin['jacobian'][world,0,0].cpu().numpy(),_row(model,data),rtol=2e-5,atol=2e-6)
  before=sim.snapshot();sim.step(6);after=sim.snapshot()
  sim.restore(before);sim.step(6);_same_checkpoint(after,sim.snapshot())
