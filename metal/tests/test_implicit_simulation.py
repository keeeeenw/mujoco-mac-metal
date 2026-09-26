# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Implicitfast non-free rigid joint rollout parity."""

import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.simulation import MetalSimulation

XML = """<mujoco><option timestep=".002" integrator="implicitfast" gravity="0 0 0"><flag contact="disable"/></option>
<worldbody><body><joint name="a" type="ball" damping=".3" stiffness=".2"/><geom type="box" size=".1 .2 .3" mass="1"/>
<body pos=".3 .1 .2"><joint name="b" damping=".4" stiffness=".3" axis="0 1 0"/><geom type="box" size=".1 .2 .1" mass="1"/>
<body pos=".1 .2 .3"><joint name="c" type="slide" damping=".2" axis="1 0 0"/><geom type="sphere" size=".1" mass=".3"/></body></body></body></worldbody><actuator><motor joint="b"/></actuator></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
@pytest.mark.parametrize(
    'flags',
    [
        0,
        int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
        int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
    ],
)
def test_implicitfast_rollout_and_checkpoint(flags):
  m = mujoco.MjModel.from_xml_string(XML)
  m.opt.disableflags |= flags
  q = np.tile(m.qpos0, (2, 1))
  v = np.tile(np.linspace(-0.3, 0.2, m.nv), (2, 1)).astype('float32')
  v[1] *= -1
  for row in range(2):
    mujoco.mj_integratePos(m, q[row], v[row].astype('float64'), 0.4)
  q = q.astype('float32')
  sim = MetalSimulation(m, 2, q, v, profile='contact_free_implicitfast_v1')
  refs = [mujoco.MjData(m) for _ in range(2)]
  for row, d in enumerate(refs):
    d.qpos[:] = q[row]
    d.qvel[:] = v[row]
  for step in range(400):
    ctrl = np.array([[0.01], [-0.02]], dtype='float32')
    sim.step(ctrl=ctrl)
    for row, d in enumerate(refs):
      d.ctrl[:] = ctrl[row]
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for row, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[row], d.qpos, atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qvel[row], d.qvel, atol=5e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qacc[row], d.qacc, atol=1e-4, rtol=1e-3)
  sim.step(5)
  expected = sim.state.snapshot()
  sim.state.restore(snap)
  sim.step(5)
  np.testing.assert_array_equal(sim.state.snapshot().qpos, expected.qpos)
