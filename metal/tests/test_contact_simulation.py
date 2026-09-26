# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Contact active-set trajectories against CPU MuJoCo."""

import os
import mujoco
import numpy as np
import pytest

XML = """<mujoco><option timestep='.002' tolerance='1e-10' iterations='100'/><default><geom condim='1'/></default><worldbody>
<geom type='plane' size='2 2 .1'/>
<body pos='0 0 .8'><freejoint/><geom type='sphere' size='.15' mass='1'/></body>
<body pos='0 0 1.3'><freejoint/><geom type='sphere' size='.15' mass='1.2'/></body>
</worldbody></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
def test_falling_stacking_balls_match_cpu_and_resume():
  from mujoco_metal.simulation import MetalSimulation

  m = mujoco.MjModel.from_xml_string(XML)
  q = np.tile(m.qpos0, (2, 1)).astype('float32')
  q[1, 0] = 0.12
  sim = MetalSimulation(m, 2, qpos=q, profile='normal_contact_euler_v1')
  refs = [mujoco.MjData(m) for _ in range(2)]
  for i, d in enumerate(refs):
    d.qpos[:] = q[i]
  errors = np.zeros(2)
  for step in range(500):
    sim.step()
    for d in refs:
      mujoco.mj_step(m, d)
    s = sim.state.snapshot()
    assert not np.any(s.status), (step, s.status)
    for i, d in enumerate(refs):
      errors = np.maximum(
          errors,
          [np.max(abs(s.qpos[i] - d.qpos)), np.max(abs(s.qvel[i] - d.qvel))],
      )
  assert errors[0] < 2e-3 and errors[1] < 2e-2, errors
  saved = sim.state.snapshot()
  sim.step(5)
  qafter = sim.state.snapshot().qpos
  sim.state.restore(saved)
  sim.step(5)
  np.testing.assert_array_equal(sim.state.snapshot().qpos, qafter)
