# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Retained original assembly state: sparse Newton residual inside the L2 bound.

The original 600-step scene failed at step 412 even though the recomputed
inner residual norm was below its unchanged requested limit. A component
limit divided by sqrt(n) incorrectly rejected the sparse residual. These
literal project-generated states keep the regression independent of logs.
"""
import os
from pathlib import Path
import mujoco
import numpy as np
import pytest

_QPOS = np.array([-0.1296180635690689, 6.900041171320481e-06, 0.03964091092348099, 0.9999997615814209, -0.00013419897004496306, 0.0006521888426505029, 0.00028631542227230966, 0.12999999523162842, 0.05000000074505806, 0.05234108120203018, 1.0, 4.399378639163842e-09, 8.943739845945764e-12, -1.937227938828001e-12, 0.12999999523162842, 0.05000000074505806, 0.08936117589473724, 1.0, -3.6702751948780588e-09, -6.29754625183776e-12, -2.5707418364439548e-21], dtype=np.float32)
_QVEL = np.array([-0.00011805506073869765, -0.00026718678418546915, 0.005147792864590883, -0.2354728877544403, 0.12127936631441116, 0.0010296286782249808, 1.6480957223952153e-12, -1.0677930067926056e-10, 1.7256032713852676e-10, 5.932354163462605e-09, 1.2765428991645678e-10, -7.3357665591318e-17, 1.1495772866618181e-12, -1.092452309170433e-10, 2.3243044988685568e-10, -6.96213930950762e-09, -9.114923399389241e-11, -3.4823411984384556e-19], dtype=np.float32)
_WARM = np.array([0.027391133829951286, 0.054801590740680695, 0.14984817802906036, 35.20756530761719, -20.216020584106445, -0.17429915070533752, 9.138610007619619e-13, -7.687580572124375e-10, -1.0712335907214765e-08, 6.396847851419807e-08, 3.424738270751959e-11, 6.791758351738526e-15, 9.545569629870743e-13, -5.109977441186686e-10, -1.3465707660031967e-08, -4.0487353913931656e-08, -7.563127901732969e-11, -2.4630427187411588e-20], dtype=np.float32)

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native Newton regression")
def test_loaded_original_assembly_newton_step_and_full_checkpoint_replay(monkeypatch):
  from mujoco_metal import MetalSimulation
  from test_analytic_material_lifecycle_010 import _native
  from magnetic_crane import _same_checkpoint
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).parents[1] / "examples" / "assembly_puzzle.xml"))
  sim = MetalSimulation(model, 1, qpos=_QPOS[None], qvel=_QVEL[None], profile="integrated_euler_v1")
  before = sim.snapshot()
  before["native_state"]["qacc_warmstart"][:] = _WARM[None]
  sim.restore(before)
  source = mujoco.MjData(model)
  source.qpos[:], source.qvel[:], source.qacc_warmstart[:] = _QPOS, _QVEL, _WARM
  mujoco.mj_step(model, source)
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
  assert sim._last_coupled["contact_mask"].any().item()
  assert np.linalg.norm(sim._last_coupled["qfrc_constraint"].cpu().numpy()) > .01
  np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), source.qpos, rtol=0, atol=5e-6)
  np.testing.assert_allclose(sim.state.qvel[0].cpu().numpy(), source.qvel, rtol=0, atol=5e-6)
  sim.restore(before)
  commands = [(.0001 * np.cos(np.arange(model.nv) + i)).astype(np.float32)[None] for i in range(20)]
  for command in commands:
    np.testing.assert_array_equal(_native(monkeypatch, lambda: sim.step(qfrc_applied=command)).cpu().numpy(), [0])
  expected = sim.snapshot()
  sim.restore(before)
  for command in commands:
    np.testing.assert_array_equal(_native(monkeypatch, lambda: sim.step(qfrc_applied=command)).cpu().numpy(), [0])
  _same_checkpoint(sim.snapshot(), expected)
