# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""RK4 tests against MuJoCo 3.10, including quaternion stage semantics."""

import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.stepping import validate_stepping_profile

XML = """<mujoco><option integrator='RK4' timestep='.002'><flag contact='disable'/></option>
<worldbody><body pos='0 0 1'><freejoint/><geom type='box' size='.1 .2 .3'/></body>
<body pos='1 0 1'><joint type='ball' damping='.1'/><geom type='capsule' size='.08 .3' pos='.1 .2 0'/></body>
<body pos='2 0 1'><joint name='j' axis='0 1 0' damping='.2'/><geom type='box' size='.1 .2 .3' pos='.1 0 -.2'/></body>
</worldbody><actuator><motor joint='j' gear='2'/></actuator></mujoco>"""


def test_rk4_contract():
  m = mujoco.MjModel.from_xml_string(XML)
  profile = validate_stepping_profile(m, profile='contact_free_motor_rk4_v1')
  assert not profile.implicit_euler_damping
  assert profile.passive_damping_enabled
  with pytest.raises(ValueError, match='Euler'):
    validate_stepping_profile(m, profile='contact_free_motor_euler_v1')
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
  with pytest.raises(ValueError, match='RK4'):
    validate_stepping_profile(m, profile='contact_free_motor_rk4_v1')


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
def test_rk4_trajectory_restore_and_row_failure():
  import torch
  from mujoco_metal.simulation import MetalSimulation

  m = mujoco.MjModel.from_xml_string(XML)
  rng = np.random.default_rng(321)
  q = np.tile(m.qpos0, (3, 1)).astype('float32')
  v = rng.normal(0, 0.4, (3, m.nv)).astype('float32')
  sim = MetalSimulation(m, 3, q, v, profile='contact_free_motor_rk4_v1')
  refs = [mujoco.MjData(m) for _ in range(3)]
  for i, d in enumerate(refs):
    d.qpos[:] = q[i]
    d.qvel[:] = v[i]
  for step in range(500):
    ctrl = np.full((3, 1), 0.1 * np.sin(step / 50), dtype='float32')
    force = np.zeros((3, m.nv), dtype='float32')
    force[:, :3] = [0.2, -0.1, 0.3]
    sim.step(ctrl=ctrl, qfrc_applied=force)
    for i, d in enumerate(refs):
      d.ctrl[:] = ctrl[i]
      d.qfrc_applied[:] = force[i]
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for i, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[i], d.qpos, atol=6e-5, rtol=2e-4)
    np.testing.assert_allclose(snap.qvel[i], d.qvel, atol=8e-5, rtol=2e-4)
    np.testing.assert_allclose(snap.qacc[i], d.qacc, atol=1e-3, rtol=3e-4)
  sim.step(3)
  replay = sim.state.snapshot()
  sim.state.restore(snap)
  sim.step(3)
  np.testing.assert_array_equal(sim.state.snapshot().qpos, replay.qpos)
  force = torch.zeros((3, m.nv), device='mps')
  force[1, 0] = float('nan')
  before = sim.state.snapshot()
  sim.step(qfrc_applied=force)
  after = sim.state.snapshot()
  assert after.status[1] != 0
  np.testing.assert_array_equal(after.qpos[1], before.qpos[1])
  assert after.status[0] == 0 and after.status[2] == 0
