# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compensated real tendon geometry, multiple pulleys and state ownership."""
import os
from pathlib import Path
import mujoco
import numpy as np
import pytest


def _model():
  return mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".0005" gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><site name="a" pos="-.3 .2 .4"/>
      <site name="b" pos=".6 -.2 .7"/><site name="c" pos="-.2 -.4 .2"/>
      <body pos=".1 .2 .6"><joint name="h" axis="0 1 0"/>
        <geom size=".05" mass="1"/><site name="s1" pos=".2 .03 -.02"/>
        <body pos=".3 0 0"><joint name="ball" type="ball"/>
          <geom size=".05" mass=".5"/><site name="s2" pos=".07 -.02 .12"/>
        </body></body>
      <body pos="-.1 -.2 .5"><freejoint name="free"/>
        <geom size=".05" mass=".5"/><site name="s3" pos=".12 .06 -.04"/>
      </body></worldbody><tendon>
      <spatial name="rope" armature=".03"><site site="a"/><site site="s1"/>
        <pulley divisor="2.5"/><site site="b"/><site site="s2"/>
        <pulley divisor=".75"/><site site="s3"/><site site="c"/>
      </spatial></tendon>
    <keyframe><key name="initial"/></keyframe></mujoco>''')


def _states(model):
  q=np.tile(model.qpos0,(2,1))
  for world in range(2):
    mujoco.mj_integratePos(model,q[world],np.linspace(.03,.17,model.nv)*(1 if world==0 else -.7),1)
  return q.astype(np.float32), np.stack([np.linspace(.02,.2,model.nv),np.linspace(-.12,-.03,model.nv)]).astype(np.float32)


def _oracle(model,q,v):
  data=mujoco.MjData(model);data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(model,data)
  row=np.zeros(model.nv);start=int(model.ten_J_rowadr[0]);n=int(model.ten_J_rownnz[0])
  row[model.ten_J_colind[start:start+n]]=data.ten_J[start:start+n]
  return data,row


def _finite_difference(model,q,v):
  result=np.empty(model.nv)
  for dof in range(model.nv):
    direction=np.eye(model.nv)[dof];plus=q.astype(float).copy();minus=plus.copy()
    mujoco.mj_integratePos(model,plus,direction,1e-6)
    mujoco.mj_integratePos(model,minus,direction,-1e-6)
    dp,_=_oracle(model,plus,v);dm,_=_oracle(model,minus,v)
    result[dof]=(dp.ten_length[0]-dm.ten_length[0])/2e-6
  return result


def test_multiple_pulley_ball_free_fixture_has_independent_nonzero_jacobian():
  model=_model();q,v=_states(model)
  assert model.nv==10
  for world in range(2):
    data,row=_oracle(model,q[world],v[world])
    assert abs(data.ten_velocity[0])>.001 and np.count_nonzero(abs(row)>.001)>=8
    np.testing.assert_allclose(row,_finite_difference(model,q[world],v[world]),rtol=2e-5,atol=3e-6)


def test_source_site_capacity_accounts_persistent_and_transient_storage():
  from mujoco_metal.capacity import _runtime_buffer_sizes
  model=_model();sizes=dict(_runtime_buffer_sizes(model,2))
  assert sizes['spatial_tendon.source_site_poses']==4*(3*model.nsite+7*model.nbody+6*model.njnt)
  assert sizes['spatial_tendon.source_site_constants']==2*model.nwrap


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU')!='1',reason='native site geometry')
def test_native_multiple_pulleys_ball_free_fd_mask_reset_restore(monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'examples'))
  from magnetic_crane import _same_checkpoint
  model=_model();q,v=_states(model)
  refs=[_oracle(model,q[w],v[w]) for w in range(2)]
  sim=MetalSimulation(model,2,profile='integrated_euler_v1');sim.reset(qpos=q,qvel=v)
  def native(mask=None):
    dynamics=sim._smooth.run_device(sim.state._qpos,sim.state._qvel)
    return sim._spatial_tendons.run_kinematics(sim.state._qvel,dynamics['poses'],world_mask=mask)
  with monkeypatch.context() as context:
    def forbidden(*_args,**_kwargs):raise AssertionError('site geometry used CPU physics')
    for name in ('mj_forward','mj_kinematics','mj_comPos','mj_tendon'):
      context.setattr(mujoco,name,forbidden)
    got=native()
  for world,(data,row) in enumerate(refs):
    np.testing.assert_allclose(got['jacobian'][world,0].cpu().numpy(),row,rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(got['length'][world,0].cpu().numpy(),data.ten_length[0],rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(got['velocity'][world,0].cpu().numpy(),row@v[world],rtol=2e-6,atol=2e-7)
  saved={key:value.cpu().numpy().copy() for key,value in got.items()}
  poses=sim._smooth.run_device(sim.state._qpos,sim.state._qvel)["poses"]
  invalid=dict(poses);invalid["site_pos_low"]=poses["site_pos_low"][:,:-1].contiguous()
  with pytest.raises(ValueError,match="site_pos_low"):
    sim._spatial_tendons.run_kinematics(sim.state._qvel,invalid)
  for key,value in got.items():assert value.cpu().numpy().tobytes()==saved[key].tobytes()
  replaced={key:value.clone() for key,value in poses.items()}
  fresh=sim._spatial_tendons.run_kinematics(sim.state._qvel,replaced)
  for key,value in fresh.items():np.testing.assert_array_equal(value.cpu().numpy(),saved[key])
  kept={key:value[1].cpu().numpy().copy() for key,value in got.items()}
  # A real in-place tensor mutation followed by fresh FK must update the
  # selected world while leaving the other world's outputs byte-identical.
  sim.state._qpos[0,0].add_(.2)
  mask=torch.tensor([1,0],dtype=torch.int32,device='mps')
  updated=native(mask)
  data,row=_oracle(model,sim.state.qpos[0].cpu().numpy(),v[0])
  np.testing.assert_allclose(updated['jacobian'][0,0].cpu().numpy(),row,rtol=2e-6,atol=2e-7)
  for key,value in updated.items():assert value[1].cpu().numpy().tobytes()==kept[key].tobytes()
  sim.reset(qpos=q,qvel=v);checkpoint=sim.snapshot()
  sim.step(3);expected=sim.snapshot()
  sim.restore(checkpoint);sim.step(3)
  _same_checkpoint(expected,sim.snapshot())
  sim.reset_to_keyframe(0);reset=native()
  data,row=_oracle(model,model.qpos0,np.zeros(model.nv))
  np.testing.assert_allclose(reset['jacobian'][0,0].cpu().numpy(),row,rtol=2e-6,atol=2e-7)
  np.testing.assert_array_equal(reset['velocity'].cpu().numpy(),0)
