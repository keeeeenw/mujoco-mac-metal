# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native inertia-box fluid trajectories against pinned CPU MuJoCo."""

import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.simulation import MetalSimulation

XML = """<mujoco><option timestep=".001" gravity="0 0 0" density="1.7" viscosity=".09" wind=".6 -.2 .35"><flag contact="disable"/></option>
<worldbody><body pos=".2 -.3 1" quat=".8 .1 -.3 .2"><freejoint/><inertial pos=".08 -.04 .03" mass="1.3" diaginertia=".08 .11 .14"/><geom type="box" size=".12 .08 .1"/>
<body pos=".15 .1 .05"><joint name="h" axis=".3 .8 -.2" damping=".02"/><geom type="ellipsoid" size=".1 .16 .07" mass=".6"/></body></body></worldbody><actuator><motor joint="h"/></actuator></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
@pytest.mark.parametrize('integrator', ['Euler', 'RK4'])
@pytest.mark.parametrize('disable_passive', [False, True])
def test_fluid_trajectory(integrator, disable_passive):
  m = mujoco.MjModel.from_xml_string(
      XML.replace(
          'timestep=".001"', f'timestep=".001" integrator="{integrator}"'
      )
  )
  if disable_passive:
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING) | int(
        mujoco.mjtDisableBit.mjDSBL_DAMPER
    )
  q = np.tile(m.qpos0, (2, 1)).astype('float32')
  v = np.tile(np.linspace(-0.7, 0.9, m.nv), (2, 1)).astype('float32')
  v[1] *= -1
  sim = MetalSimulation(
      m, 2, q, v, profile=f'contact_free_fluid_{integrator.lower()}_v1'
  )
  refs = [mujoco.MjData(m) for _ in range(2)]
  for row, d in enumerate(refs):
    d.qpos[:] = q[row]
    d.qvel[:] = v[row]
  for step in range(400):
    ctrl = np.array([[0.01], [-0.01]], dtype='float32')
    sim.step(ctrl=ctrl)
    for row, d in enumerate(refs):
      d.ctrl[:] = ctrl[row]
      mujoco.mj_step(m, d)
  snap = sim.state.snapshot()
  assert not np.any(snap.status)
  for row, d in enumerate(refs):
    np.testing.assert_allclose(snap.qpos[row], d.qpos, atol=3e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qvel[row], d.qvel, atol=5e-5, rtol=1e-4)
    np.testing.assert_allclose(snap.qacc[row], d.qacc, atol=5e-4, rtol=1e-3)
