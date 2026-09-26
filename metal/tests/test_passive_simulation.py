# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Integrated passive force trajectories against CPU MuJoCo."""

import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.simulation import MetalSimulation

XML = """<mujoco><option timestep='.001' gravity='0 0 0'><flag contact='disable'/></option>
<worldbody><body><freejoint/><geom type='box' size='.1 .2 .3' mass='1'/></body>
<body pos='1 0 0'><joint type='ball' stiffness='.4' damping='.05'/><geom type='box' size='.1 .2 .3' mass='1'/></body>
<body pos='2 0 0'><joint name='h' stiffness='.5' damping='.05' axis='0 1 0'/><geom type='box' size='.1 .2 .3' mass='1'/></body>
</worldbody><actuator><motor joint='h'/></actuator></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
@pytest.mark.parametrize('integrator', ['Euler', 'RK4'])
@pytest.mark.parametrize(
    'flags',
    [
        0,
        int(mujoco.mjtDisableBit.mjDSBL_SPRING),
        int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
        int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
    ],
)
def test_passive_rollout(integrator, flags):
  m = mujoco.MjModel.from_xml_string(
      XML.replace(
          "timestep='.001'", f"timestep='.001' integrator='{integrator}'"
      )
  )
  m.opt.disableflags |= flags
  m.jnt_stiffness[0] = 0.3
  m.jnt_stiffnesspoly[:] = [0.02, 0.01]
  m.dof_dampingpoly[:] = [0.01, 0.005]
  q = np.tile(m.qpos0, (2, 1))
  for row in range(2):
    mujoco.mj_integratePos(m, q[row], np.linspace(-0.4, 0.4, m.nv), 0.3)
  q = q.astype('float32')
  v = np.tile(np.linspace(-0.2, 0.2, m.nv), (2, 1)).astype('float32')
  v[1] *= -1
  sim = MetalSimulation(
      m, 2, q, v, profile='contact_free_passive_' + integrator.lower() + '_v1'
  )
  refs = [mujoco.MjData(m) for _ in range(2)]
  for i, d in enumerate(refs):
    d.qpos[:] = q[i]
    d.qvel[:] = v[i]
  for _ in range(400):
    sim.step(ctrl=np.full((2, 1), 0.01, dtype='float32'))
    for d in refs:
      d.ctrl[:] = 0.01
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for i, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[i], d.qpos, atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qvel[i], d.qvel, atol=6e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qacc[i], d.qacc, atol=5e-4, rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
def test_projected_wrench_and_gravity_compensation_trajectory():
  m = mujoco.MjModel.from_xml_string(
      """<mujoco><option timestep='.002'><flag contact='disable'/></option>
  <worldbody><body gravcomp='.6' quat='.9 .1 .2 .3'><freejoint/><geom type='box' size='.1 .2 .3' mass='2' pos='.05 -.03 .1'/></body></worldbody></mujoco>"""
  )
  sim = MetalSimulation(m, 2, profile='contact_free_passive_euler_v1')
  refs = [mujoco.MjData(m) for _ in range(2)]
  for step in range(300):
    wrench = np.zeros((2, m.nbody, 6), dtype='float32')
    wrench[:, 1] = [0.5, -0.2, 1, 0.01, 0.02, -0.03]
    wrench[1, 1] *= -1
    sim.step(xfrc_applied=wrench)
    for i, d in enumerate(refs):
      d.xfrc_applied[:] = wrench[i]
      mujoco.mj_step(m, d)
  s = sim.state.snapshot()
  assert not np.any(s.status)
  for i, d in enumerate(refs):
    np.testing.assert_allclose(s.qpos[i], d.qpos, atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(s.qvel[i], d.qvel, atol=3e-5, rtol=1e-4)
  before = sim.state.snapshot()
  with pytest.raises(ValueError, match='finite'):
    sim.step(xfrc_applied=np.full((2, m.nbody, 6), np.nan))
  np.testing.assert_array_equal(sim.state.snapshot().qpos, before.qpos)
