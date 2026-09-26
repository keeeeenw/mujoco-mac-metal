# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Three marbles roll down a tilted plane and exchange momentum by contact.

The left view is the native normal-contact profile. The right view is an
independent MuJoCo ``mj_step`` reference from the same state and inputs.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _load_model():
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if model.nv != 18 or model.ngeom != 4 or model.npair or model.nexclude:
    raise RuntimeError("Expected three free marbles and one tilted plane")
  if np.any(model.geom_condim != 1):
    raise RuntimeError("The marble cascade requires condim=1")
  return model


def _initial_state(model):
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  # The plane normal is rotated 20 degrees about world Y. The ramp direction
  # is its tangent; each following marble begins faster than the one ahead.
  angle = np.deg2rad(20.0)
  downhill = np.array([-np.cos(angle), 0.0, np.sin(angle)])
  for body, speed in ((1, 0.2), (2, 1.1), (3, 1.9)):
    qa = int(model.jnt_qposadr[int(model.body_jntadr[body])])
    da = int(model.jnt_dofadr[int(model.body_jntadr[body])])
    qvel[da : da + 3] = speed * downhill
  return qpos, qvel


def run(steps=600, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos, qvel = _initial_state(model)
  actual = mujoco.MjData(model)
  actual.qpos[:], actual.qvel[:] = qpos, qvel
  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos, qvel
  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    native = MetalSimulation(
        model,
        qpos=qpos[None, :],
        qvel=qvel[None, :],
        profile="normal_contact_euler_v1",
    )

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model, record, "Marble cascade", [0.1, 0.0, 0.2], 2.4,
        azimuth=145, elevation=-22,
    )
  max_qpos_error = max_qvel_error = 0.0
  contact_steps = 0
  peak_contacts = 0
  for step in range(steps):
    mujoco.mj_step(model, reference)
    if native is None:
      mujoco.mj_step(model, actual)
    else:
      native.step()
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"native status: {state.status.tolist()}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
    max_qpos_error = max(
        max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos)))
    )
    max_qvel_error = max(
        max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel)))
    )
    peak_contacts = max(peak_contacts, reference.ncon)
    contact_steps += int(reference.ncon > 0)
    if recorder:
      recorder.frame(
          step, actual, reference,
          extra=f"3 free marbles | {reference.ncon} CPU contacts | 2 ms steps",
      )
  if recorder:
    recorder.close()
  result = {
      "demo": "marble_cascade",
      "mode": mode,
      "steps": steps,
      "profile": "normal_contact_euler_v1" if native else "MuJoCo CPU",
      "contact_family": "plane-sphere and sphere-sphere, condim=1",
      "peak_contacts": peak_contacts,
      "contact_steps": contact_steps,
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "record": str(record) if record else None,
  }
  if check and mode == "metal":
    if contact_steps < 10:
      raise AssertionError("marble cascade did not exercise sustained contact")
    if max_qpos_error > 0.08 or max_qvel_error > 0.8:
      raise AssertionError(json.dumps(result, indent=2))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true", help="Run without a viewer")
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--check", action="store_true", help="Check native state against CPU MuJoCo")
  parser.add_argument("--record", help="Optionally save side-by-side actual simulation frames as a GIF")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and (not args.headless or args.mode != "metal"):
    parser.error("Use positive --steps; --check requires --headless and --mode metal")
  print(json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2))


if __name__ == "__main__":
  main()
