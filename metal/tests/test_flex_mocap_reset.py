# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Reset publishes flex kinematics from the same persistent mocap transaction."""
import os

import mujoco
import numpy as np
import pytest


def _model(dim):
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" timestep=".001"/>
    <worldbody><body mocap="true" name="driver" pos=".1 .2 .3">
      <flexcomp name="cargo" type="grid" dim="{dim}"
          count="2 2 {2 if dim == 3 else 1}" spacing=".1 .1 .1"
          radius=".001" mass=".5">
        <contact contype="0" conaffinity="0"/>
      </flexcomp>
    </body></worldbody>
    <keyframe><key name="moved" mpos=".5 .6 .7"
        mquat=".9238795325 0 .3826834324 0"/></keyframe>
    </mujoco>''')


@pytest.mark.parametrize('dim', [2,3])
def test_mocap_motion_changes_all_compiled_flex_vertices(dim):
  model = _model(dim)
  data = mujoco.MjData(model); mujoco.mj_forward(model,data)
  initial = data.flexvert_xpos.copy()
  mujoco.mj_resetDataKeyframe(model,data,0); mujoco.mj_forward(model,data)
  assert model.nmocap == 1 and model.nv >= 12
  assert np.all(np.linalg.norm(data.flexvert_xpos-initial,axis=1) > .2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('dim', [2,3])
def test_partial_reset_keyframe_restore_and_next_step_publish_current_mocap_flex(dim):
  from mujoco_metal.simulation import MetalSimulation
  model = _model(dim)
  sim = MetalSimulation(model,batch_size=2,profile='integrated_scalable_v1')
  qvel = np.full((2,model.nv),.02,np.float32)
  initial_pos = np.asarray([[[.2,.3,.4]], [[-.3,.4,.6]]],np.float32)
  initial_quat = np.asarray([[[1.,0,0,0]], [[.9238795,.3826834,0,0]]],np.float32)
  sim.reset(qvel=qvel,mocap_pos=initial_pos,mocap_quat=initial_quat)
  host = lambda tensor: tensor.detach().cpu().numpy().copy()

  def check_vertices():
    positions, velocities = host(sim.state._qpos),host(sim.state._qvel)
    mocap_pos,mocap_quat = host(sim.state._mpos),host(sim.state._mquat)
    actual = host(sim._flex.flexvert_xpos)
    expected = []
    for world in range(2):
      data=mujoco.MjData(model)
      data.qpos[:],data.qvel[:]=positions[world],velocities[world]
      data.mocap_pos[:],data.mocap_quat[:]=mocap_pos[world],mocap_quat[world]
      mujoco.mj_forward(model,data)
      expected.append(data.flexvert_xpos.copy())
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    return actual

  first = check_vertices()
  np.testing.assert_array_equal(host(sim.step()),0)
  check_vertices()
  peer = host(sim._flex.flexvert_xpos)[1].copy()
  sim.reset(env_ids=[0],mocap_pos=np.asarray([[.7,-.2,.5]],np.float32),
            mocap_quat=np.asarray([[.9238795,0,.3826834,0]],np.float32))
  changed = check_vertices()
  assert np.all(np.linalg.norm(changed[0]-first[0],axis=1) > .2)
  np.testing.assert_array_equal(changed[1],peer)
  snapshot=sim.snapshot()
  sim.reset_to_keyframe(0,env_ids=[0])
  keyed=check_vertices()
  np.testing.assert_array_equal(keyed[1],peer)
  assert np.max(np.abs(keyed[0]-changed[0])) > .1
  sim.restore(snapshot)
  np.testing.assert_array_equal(check_vertices(),changed)
  np.testing.assert_array_equal(host(sim.step()),0)
  expected=check_vertices()
  sim.restore(snapshot); np.testing.assert_array_equal(host(sim.step()),0)
  np.testing.assert_array_equal(check_vertices(),expected)
