# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A tiny crossed-cable XY plotter with native scalar transmission forces.

Two fixed-joint tendons route orthogonal carriage coordinates as x+y and x-y.
Stateless affine position servos drive those lengths along a bounded looping
calligraphy path. The native example composes Metal smooth dynamics, the
transmission force stage, a dense solve, and semi-implicit Euler; MuJoCo's CPU
``mj_step`` advances an independent reference.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _path(t):
  """Smooth figure-eight calligraphy command in carriage coordinates."""
  return .22 * np.sin(.72 * t), .17 * np.sin(1.44 * t + .35)


class _NativeStages:
  """Compose the native primitives for this explicit demo model."""

  def __init__(self, model, qpos, qvel):
    import torch

    from mujoco_metal.integration import MetalEulerIntegration
    from mujoco_metal.model import load_model
    from mujoco_metal.smooth_metal import MetalSmoothDynamics
    from mujoco_metal.smooth_solve import MetalDenseSolve
    from mujoco_metal.transmissions import MetalTransmissions

    self.torch = torch
    self.descriptor = load_model(model)
    self.dynamics = MetalSmoothDynamics(self.descriptor, batch_size=1)
    self.transmission = MetalTransmissions(model)
    self.solve = MetalDenseSolve(model.nv, batch_size=1)
    self.integrator = MetalEulerIntegration(
        self.descriptor, batch_size=1, timestep=model.opt.timestep
    )
    device = torch.device("mps")
    self.qpos = torch.tensor(qpos[None, :], dtype=torch.float32, device=device)
    self.qvel = torch.tensor(qvel[None, :], dtype=torch.float32, device=device)
    self.control = torch.zeros((1, model.nu), dtype=torch.float32, device=device)
    self.time = torch.zeros((1,), dtype=torch.float32, device=device)
    self.status = torch.zeros((1,), dtype=torch.int32, device=device)
    self.nv = model.nv

  def step(self, control):
    self.control.copy_(self.torch.tensor(control[None, :], dtype=self.torch.float32, device="mps"))
    dynamics = self.dynamics.run_device(self.qpos, self.qvel)
    actuator_force = self.transmission.run_device(self.qpos, self.qvel, self.control)
    rhs = actuator_force - dynamics["qfrc_bias"]
    acceleration, solve_status = self.solve.run_device(dynamics["mass_matrix"], rhs)
    qpos, qvel, time, status = self.integrator.run_device(
        self.qpos, self.qvel, acceleration, self.time, solve_status
    )
    self.qpos, self.qvel, self.time, self.status = qpos, qvel, time, status
    if int(status[0].item()) != 0:
      raise RuntimeError(f"native solver/integrator status {int(status[0].item())}")
    return qpos[0].cpu().numpy().copy(), qvel[0].cpu().numpy().copy()


def run(steps=200, mode="metal", check=False, record=None):
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if model.nu != 2 or model.ntendon != 2 or model.nq != 2 or model.nv != 2:
    raise RuntimeError("Expected a two-axis carriage driven by two fixed tendons")
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")

  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  actual = mujoco.MjData(model)
  reference = mujoco.MjData(model)
  actual.qpos[:] = reference.qpos[:] = qpos
  actual.qvel[:] = reference.qvel[:] = qvel
  native = _NativeStages(model, qpos, qvel) if mode == "metal" else None
  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model,
        record,
        "Crossed-cable calligraphy plotter",
        lookat=[0, 0, .58],
        distance=1.9,
        azimuth=135,
        elevation=-28,
    )

  max_qpos_error = max_qvel_error = 0.0
  dt = float(model.opt.timestep)
  try:
    for index in range(steps):
      command_x, command_y = _path(index * dt)
      # Tendon lengths are x+y and x-y, matching the fixed-joint wrap maps.
      control = np.array([command_x + command_y, command_x - command_y])
      actual.ctrl[:] = reference.ctrl[:] = control
      mujoco.mj_step(model, reference)
      if native is None:
        mujoco.mj_step(model, actual)
      else:
        actual.qpos[:], actual.qvel[:] = native.step(control)
        actual.time = (index + 1) * dt
      max_qpos_error = max(max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos))))
      max_qvel_error = max(max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel))))
      if recorder:
        recorder.frame(index, actual, reference, "Tendon position servos | 1x playback")
  finally:
    if recorder:
      recorder.close()

  report = {
      "demo": "cable_plotter",
      "mode": mode,
      "steps": steps,
      "transmissions": ["x + y", "x - y"],
      "actuators": "stateless fixed-gain affine-bias position servos",
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "recording": str(record) if record else None,
  }
  if check and mode == "metal" and (max_qpos_error > 1e-3 or max_qvel_error > 1e-2):
    raise AssertionError(json.dumps(report, indent=2))
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true", help="Run without an interactive viewer")
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--check", action="store_true", help="Check a short native-versus-CPU rollout (at most 200 steps)")
  parser.add_argument("--record", type=Path, help="Record actual native/CPU renders as a GIF (requires OpenGL and Pillow)")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and (not args.headless or args.steps > 200):
    parser.error("Use positive --steps; --check requires --headless and at most 200 steps")
  if args.check and args.mode != "metal":
    parser.error("--check compares native Metal against the independent CPU reference; use --mode metal")
  print(json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2))


if __name__ == "__main__":
  main()
