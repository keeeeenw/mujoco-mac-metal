# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A driven, colored hinge/ball lattice that carries a mechanical wave.

The native profile assembles and solves MuJoCo implicitfast's damping-modified
mass matrix. A fixed-gain motor drives the first link; coupled articulated
inertia, springs, and damping carry the motion down the lattice. The CPU panel
is a separate MuJoCo rollout. Geom contacts are explicitly disabled.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "contact_free_implicitfast_v1"


def _load_model():
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if model.njnt != 8 or model.nu != 1 or model.nv != 16:
    raise RuntimeError("Expected an eight-joint hinge/ball wave lattice and one motor")
  if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
    raise RuntimeError("The model must use the implicitfast integrator")
  if not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
    raise RuntimeError("The wave lattice must keep geom contacts disabled")
  return model


def _drive(at_time):
  # A bright two-tone forcing pattern keeps the traveling mode visually legible.
  return np.array([.112*np.sin(2.45*at_time) + .025*np.sin(4.9*at_time)], dtype=np.float32)


def run(steps=2000, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  actual, reference = mujoco.MjData(model), mujoco.MjData(model)
  for data in (actual, reference):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    native = MetalSimulation(
        model, qpos=qpos[None, :], qvel=qvel[None, :], profile=PROFILE
    )
  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model, record, "Mechanical wave lattice | implicitfast",
        [-.15, 0, 1.36], 4.9, azimuth=145, elevation=-16,
    )
  qpos_error = qvel_error = 0.0
  max_wave_angle = 0.0
  dt = float(model.opt.timestep)
  try:
    for step in range(steps):
      ctrl = _drive(float(reference.time))
      reference.ctrl[:] = ctrl
      mujoco.mj_step(model, reference)
      if native is None:
        actual.ctrl[:] = ctrl
        mujoco.mj_step(model, actual)
      else:
        native.step(ctrl=ctrl[None, :])
        state = native.state.snapshot()
        if np.any(state.status):
          raise RuntimeError(f"native implicitfast status: {state.status.tolist()}")
        actual.qpos[:] = state.qpos[0]
        actual.qvel[:] = state.qvel[0]
        actual.time = float(state.time[0])
      qpos_error = max(qpos_error, float(np.max(np.abs(actual.qpos-reference.qpos))))
      qvel_error = max(qvel_error, float(np.max(np.abs(actual.qvel-reference.qvel))))
      wave_angle = float(np.max(np.abs(actual.qpos - qpos)))
      max_wave_angle = max(max_wave_angle, wave_angle)
      if recorder:
        recorder.frame(
            step, actual, reference,
            extra=f"measured lattice excursion {wave_angle:.3f} rad-coordinate   motor {ctrl[0]*float(model.actuator_gear[0, 0]):+.3f} N·m",
        )
  finally:
    if recorder:
      recorder.close()
  result = {
      "demo": "wave_lattice",
      "mode": mode,
      "profile": PROFILE if native is not None else "MuJoCo CPU",
      "steps": steps,
      "simulated_seconds": steps*dt,
      "features": [
          "implicitfast damping-modified mass solve",
          "four ball joints and four hinge joints with coupled articulated inertia",
          "joint springs and linear damping",
          "fixed-gain motor with smooth two-tone drive",
          "independent CPU reference; geom contacts disabled",
      ],
      "max_qpos_error": qpos_error,
      "max_qvel_error": qvel_error,
      "maximum_measured_lattice_excursion": max_wave_angle,
  }
  if check and mode == "metal" and (qpos_error > 2e-3 or qvel_error > 2e-2):
    raise AssertionError(json.dumps(result, indent=2))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument("--check", action="store_true", help="compare a short native rollout with CPU MuJoCo")
  parser.add_argument("--record", help="record side-by-side simulation frames as a GIF")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and (not args.headless or args.steps > 200):
    parser.error("--check requires --headless and 1..200 steps")
  if args.check and args.mode != "metal":
    parser.error("--check compares native Metal with the independent CPU reference")
  print(json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2))


if __name__ == "__main__":
  main()
