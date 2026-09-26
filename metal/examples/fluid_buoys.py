# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Colored free bodies drifting and rotating in MuJoCo's inertia-box fluid model."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _make_model(integrator):
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  model.opt.integrator = int(
      mujoco.mjtIntegrator.mjINT_EULER
      if integrator == "Euler"
      else mujoco.mjtIntegrator.mjINT_RK4
  )
  return model


def run(steps=600, mode="metal", integrator="Euler", check=False, record=None):
  model = _make_model(integrator)
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  for name, angular in (("ray_free", [.45, .15, -.25]),
                        ("amber_free", [-.2, .55, .12]),
                        ("violet_free", [.1, -.35, .5])):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    dof = int(model.jnt_dofadr[joint])
    qvel[dof+3:dof+6] = angular
  actual = mujoco.MjData(model)
  reference = mujoco.MjData(model)
  actual.qpos[:] = reference.qpos[:] = qpos
  actual.qvel[:] = reference.qvel[:] = qvel
  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    native = MetalSimulation(
        model,
        qpos=qpos[None, :],
        qvel=qvel[None, :],
        profile=f"contact_free_fluid_{integrator.lower()}_v1",
    )
  elif mode != "cpu":
    raise ValueError(f"Unknown mode {mode!r}")

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model, record, "Inertia-box current drift | MuJoCo model",
        lookat=[0, 0, .48], distance=2.1, azimuth=135, elevation=-18,
    )

  dt = float(model.opt.timestep)
  max_qpos_error = max_qvel_error = 0.0
  try:
    for index in range(steps):
      mujoco.mj_step(model, reference)
      if native is None:
        mujoco.mj_step(model, actual)
      else:
        native.step()
        snapshot = native.state.snapshot()
        if np.any(snapshot.status):
          raise RuntimeError(f"native simulation status: {snapshot.status.tolist()}")
        actual.qpos[:] = snapshot.qpos[0]
        actual.qvel[:] = snapshot.qvel[0]
        actual.time = snapshot.time[0]
      max_qpos_error = max(max_qpos_error, float(np.max(np.abs(actual.qpos-reference.qpos))))
      max_qvel_error = max(max_qvel_error, float(np.max(np.abs(actual.qvel-reference.qvel))))
      if recorder:
        recorder.frame(index, actual, reference, "Uniform current | 1x playback")
  finally:
    if recorder:
      recorder.close()

  bodies = []
  for name, joint in (("ray_buoy", "ray_free"), ("amber_glider", "amber_free"), ("violet_drifter", "violet_free")):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    address = int(model.jnt_qposadr[joint_id])
    bodies.append({
        "name": name,
        "position": actual.qpos[address:address+3].astype(float).tolist(),
        "orientation_wxyz": actual.qpos[address+3:address+7].astype(float).tolist(),
    })
  report = {
      "demo": "fluid_buoys",
      "mode": mode,
      "integrator": integrator,
      "steps": steps,
      "model": "MuJoCo 3.10 inertia-box fluid forces with density, viscosity, and wind",
      "bodies": bodies,
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "recording": str(record) if record else None,
  }
  if check and mode == "metal" and (max_qpos_error > 2e-3 or max_qvel_error > 2e-2):
    raise AssertionError(json.dumps(report, indent=2))
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--integrator", choices=("Euler", "RK4"), default="Euler")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", type=Path)
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and (not args.headless or args.steps > 200):
    parser.error("Use positive --steps; --check requires --headless and at most 200 steps")
  if args.check and args.mode != "metal":
    parser.error("--check compares native Metal against the independent CPU reference; use --mode metal")
  print(json.dumps(run(args.steps,args.mode,args.integrator,args.check,args.record),indent=2))


if __name__ == "__main__":
  main()
