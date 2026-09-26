# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A spring-folding flower compared against MuJoCo's CPU stepper.

Four colored petals oscillate around a fixed stem. Joint springs and dampers
provide the motion; a smooth external body wrench gives the east and west
petals a small periodic breeze. The native mode runs the bounded passive-force
profile, while the CPU reference advances with ``mj_step``.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _load_model():
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if (
      model.njnt != 4
      or model.nu
      or not (
          int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
      )
  ):
    raise RuntimeError("Expected four unactuated petals without contacts")
  return model


def _inputs(model, at_time):
  wrench = np.zeros((1, model.nbody, 6), dtype=np.float32)
  east = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "petal_east")
  west = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "petal_west")
  # xfrc_applied stores world-frame force then torque at each body's COM.
  pulse = np.sin(1.4 * at_time)
  wrench[0, east, :3] = [0.12 * pulse, 0.0, 0.03 * np.cos(at_time)]
  wrench[0, west, 3:] = [0.0, 0.018 * pulse, 0.0]
  return wrench


def run(steps=200, mode="metal", check=False, record=None):
  model = _load_model()
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  for joint, position, velocity in (
      ("east_fold", 0.18, 1.1),
      ("north_fold", -0.12, -0.8),
      ("west_fold", 0.1, 0.7),
      ("south_fold", -0.16, -1.0),
  ):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qpos[int(model.jnt_qposadr[jid])] = position
    qvel[int(model.jnt_dofadr[jid])] = velocity

  reference = mujoco.MjData(model)
  reference.qpos[:] = qpos
  reference.qvel[:] = qvel
  actual = mujoco.MjData(model)
  actual.qpos[:] = qpos
  actual.qvel[:] = qvel
  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    native = MetalSimulation(
        model,
        qpos=qpos[None, :],
        qvel=qvel[None, :],
        profile="contact_free_passive_euler_v1",
    )

  max_qpos_error = max_qvel_error = 0.0
  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model, record, "Spring flower", [0, 0, 0.9], 2.6
    )
  for step in range(steps):
    wrench = _inputs(model, float(reference.time))
    reference.xfrc_applied[:] = wrench[0]
    mujoco.mj_step(model, reference)
    if native is None:
      actual.xfrc_applied[:] = wrench[0]
      mujoco.mj_step(model, actual)
    else:
      native.step(xfrc_applied=wrench)
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"native simulation status: {state.status.tolist()}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
    max_qpos_error = max(
        max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos)))
    )
    max_qvel_error = max(
        max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel)))
    )

    if recorder:
      recorder.frame(step, actual, reference)
  if recorder:
    recorder.close()

  result = {
      "demo": "kinetic_sculpture",
      "mode": mode,
      "steps": steps,
      "profile": "contact_free_passive_euler_v1" if native else "MuJoCo CPU",
      "features": [
          "joint springs",
          "linear damping",
          "body gravity compensation",
          "applied body wrenches",
      ],
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
  }
  if check and mode == "metal":
    if max_qpos_error > 1e-3 or max_qvel_error > 1e-2:
      raise AssertionError(json.dumps(result, indent=2))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument(
      "--headless", action="store_true", help="Run without a viewer"
  )
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument(
      "--check",
      action="store_true",
      help="Check a short native-versus-CPU rollout (at most 200 steps)",
  )
  parser.add_argument("--record", help="Save actual simulation frames as GIF")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and (not args.headless or args.steps > 200):
    parser.error(
        "Use positive --steps; --check requires --headless and at most 200 steps"
    )
  print(
      json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2)
  )


if __name__ == "__main__":
  main()
