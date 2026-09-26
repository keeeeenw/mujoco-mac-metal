# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Joint constraint trajectories, failure isolation and checkpoint replay."""

import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.simulation import MetalSimulation

XML = """<mujoco><compiler angle="radian"/><option timestep=".001" gravity="0 0 0"><flag contact="disable"/></option>
<worldbody><body><joint name="a" range="-.2 .2" limited="true" frictionloss=".03" damping=".02"/><geom type="box" size=".1 .2 .1" mass="1"/>
<body pos=".2 .1 .2"><joint name="b" range="-.25 .25" limited="true" axis="0 1 0" frictionloss=".02"/><geom type="box" size=".1 .2 .1" mass="1"/></body></body></worldbody>
<equality><joint joint1="a" joint2="b" polycoef="0 1 .2 0 0"/></equality><actuator><motor joint="a"/></actuator></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
@pytest.mark.parametrize(
    'disable',
    [
        0,
        int(mujoco.mjtDisableBit.mjDSBL_EQUALITY),
        int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT),
    ],
)
def test_joint_constraint_trajectory_and_replay(disable):
  m = mujoco.MjModel.from_xml_string(XML)
  m.opt.disableflags |= disable
  q = np.array([[0.23, 0.21], [-0.18, -0.28], [0, 0]], dtype='float32')
  v = np.array([[0.1, -0.2], [-0.2, 0.1], [0, 0]], dtype='float32')
  sim = MetalSimulation(m, 3, q, v, profile='joint_constraints_euler_v1')
  refs = [mujoco.MjData(m) for _ in range(3)]
  for row, d in enumerate(refs):
    d.qpos[:] = q[row]
    d.qvel[:] = v[row]
  for step in range(300):
    ctrl = np.array([[0.06], [-0.03], [0.01]], dtype='float32')
    sim.step(ctrl=ctrl)
    for row, d in enumerate(refs):
      d.ctrl[:] = ctrl[row]
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for row, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[row], d.qpos, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(snap.qvel[row], d.qvel, atol=1e-3, rtol=1e-3)
  sim.step(10)
  expected = sim.state.snapshot()
  sim.state.restore(snap)
  sim.step(10)
  np.testing.assert_array_equal(sim.state.snapshot().qpos, expected.qpos)
