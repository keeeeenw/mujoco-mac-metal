# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU-side contract checks for the free-flight demonstration."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

_PATH = Path(__file__).parents[1] / "examples" / "space_docking.py"
_SPEC = importlib.util.spec_from_file_location("space_docking_demo", _PATH)
_DEMO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEMO)


def test_three_craft_freeflight_uses_force_profile_assumptions():
  sim = _DEMO.SpaceDocking("cpu")
  m = sim.model
  assert (m.nq, m.nv, m.nu, m.ntendon, m.neq) == (21, 18, 0, 0, 0)
  assert m.opt.gravity.tolist() == [0.0, 0.0, 0.0]
  assert m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  damping = m.dof_damping.reshape(3, 6)
  np.testing.assert_allclose(damping[:, 0], _DEMO._DAMPING)
  np.testing.assert_allclose(
      damping, np.repeat(_DEMO._DAMPING, 6).reshape(3, 6)
  )

  wrench, targets = sim.compute_force()
  assert wrench.shape == (18,)
  assert np.all(np.isfinite(wrench))
  assert targets.shape == (3, 3)
  assert not np.array_equal(targets, sim.actual.xpos[1:4])
  sim.step()
  assert sim.time == m.opt.timestep
  assert sim.max_qpos_error == sim.max_qvel_error == 0
  assert sim.report()["target_markers_are_visual_only"] is True

  end_qpos = sim.actual.qpos.copy()
  sim.reset()
  np.testing.assert_array_equal(sim.actual.qpos, sim.initial_qpos)
  sim.step()
  np.testing.assert_array_equal(sim.actual.qpos, end_qpos)


def test_headless_check_cli_runs_without_native_device():
  result = subprocess.run(
      [sys.executable, str(_PATH), "--mode", "cpu", "--headless", "--check", "--steps", "200"],
      check=True,
      capture_output=True,
      text=True,
  )
  assert '"simulated_seconds": 0.4000000000000003' in result.stdout
  assert '"max_qpos_error": 0.0' in result.stdout
  assert "same qfrc_applied" in result.stdout


def test_gif_duration_is_bounded_by_cli():
  result = subprocess.run(
      [
          sys.executable,
          str(_PATH),
          "--headless",
          "--record",
          "/tmp/unused.gif",
          "--record-seconds",
          "13",
      ],
      capture_output=True,
      text=True,
  )
  assert result.returncode == 2
  assert "between 0.5 and 12" in result.stderr
