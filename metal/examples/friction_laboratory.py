# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compare low, medium, and high friction as identical balls slide and roll."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

_BODY_NAMES = ("low_friction", "medium_friction", "high_friction")


def _load_model():
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if model.nv != 18 or model.ngeom != 4 or model.npair or model.nexclude:
    raise RuntimeError("Expected three free balls on one floor")
  if np.any(model.geom_condim != 3):
    raise RuntimeError("All laboratory contacts must use condim=3")
  if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
    raise RuntimeError("The friction laboratory requires the pyramidal cone")
  return model


def _initial_state(model):
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  for body in range(1, 4):
    jnt = int(model.body_jntadr[body])
    dof = int(model.jnt_dofadr[jnt])
    qvel[dof] = 1.3
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
        profile="friction_contact_euler_v1",
    )

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model,
        record,
        "Friction laboratory",
        [0.25, 0.0, 0.12],
        3.1,
        azimuth=110,
        elevation=-42,
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
          step,
          actual,
          reference,
          extra=f"μ = 0.01 / 0.12 / 1.00 | {reference.ncon} CPU contacts",
      )
  if recorder:
    recorder.close()
  body_states = {}
  for name in _BODY_NAMES:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    jnt = int(model.body_jntadr[body])
    dof = int(model.jnt_dofadr[jnt])
    body_states[name] = {
        "x_displacement": float(
            actual.qpos[int(model.jnt_qposadr[jnt])]
            - qpos[int(model.jnt_qposadr[jnt])]
        ),
        "slide_speed": float(np.linalg.norm(actual.qvel[dof : dof + 3])),
        "spin_speed": float(np.linalg.norm(actual.qvel[dof + 3 : dof + 6])),
        "x_slip_speed": float(
            abs(actual.qvel[dof] - 0.12 * actual.qvel[dof + 4])
        ),
    }
  result = {
      "demo": "friction_laboratory",
      "mode": mode,
      "steps": steps,
      "profile": "friction_contact_euler_v1" if native else "MuJoCo CPU",
      "contact_family": "plane-sphere and sphere-sphere, pyramidal condim=3",
      "peak_contacts": peak_contacts,
      "contact_steps": contact_steps,
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "balls": body_states,
      "record": str(record) if record else None,
  }
  if check and mode == "metal":
    if contact_steps < steps // 2 or peak_contacts < 3:
      raise AssertionError(
          "friction laboratory did not exercise sustained contacts"
      )
    if max_qpos_error > 2e-4 or max_qvel_error > 2e-4:
      raise AssertionError(json.dumps(result, indent=2))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument(
      "--headless", action="store_true", help="Run without a viewer"
  )
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument(
      "--check",
      action="store_true",
      help="Check native state against CPU MuJoCo",
  )
  parser.add_argument(
      "--record",
      help="Optionally save side-by-side actual simulation frames as a GIF",
  )
  args = parser.parse_args(argv)
  if (
      args.steps <= 0
      or args.check
      and (not args.headless or args.mode != "metal")
  ):
    parser.error(
        "Use positive --steps; --check requires --headless and --mode metal"
    )
  print(
      json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2)
  )


if __name__ == "__main__":
  main()
