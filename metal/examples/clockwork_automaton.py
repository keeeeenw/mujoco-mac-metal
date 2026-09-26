# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A clockwork automaton driven by native joint constraints and a small sequencer.

The visible toothed wheels are decorative: contacts are disabled. Their measured
rotation ratio comes from a polynomial joint equality. The pawl slider is driven
through a bounded limit and friction-loss row; no tooth-impact or ratchet contact
physics is claimed.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "joint_constraints_euler_v1"


def _load_model():
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if (
      model.nq != 4 or model.nv != 4 or model.neq != 1 or model.nu != 1
      or int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT) == 0
  ):
    raise RuntimeError("Expected two linked gears, one limited pawl, and no contacts")
  if model.eq_type[0] != int(mujoco.mjtEq.mjEQ_JOINT):
    raise RuntimeError("Expected the polynomial joint equality gear ratio")
  return model


def _sequencer(at_time):
  """Four-state winding/release/reverse/release motor and pawl-force cycle."""
  phase = int(np.floor(max(0.0, at_time) / 0.22)) % 4
  motor = (0.14, 0.0, -0.14, 0.0)[phase]
  pawl = (0.28, 0.0, -0.28, 0.0)[phase]
  return phase, np.array([motor], dtype=np.float32), pawl


def run(steps=1000, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos = model.qpos0.copy().astype(np.float32)
  qpos[0], qpos[1], qpos[2], qpos[3] = 0.08, -0.08, -0.025, 0.12
  qvel = np.array([0.0, 0.0, 0.0, -0.1], dtype=np.float32)
  actual, reference = mujoco.MjData(model), mujoco.MjData(model)
  for data in (actual, reference):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    native = MetalSimulation(
        model, qpos=qpos[None], qvel=qvel[None], profile=PROFILE
    )
  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model, record, "Clockwork automaton | measured gear ratio",
        [0, 0.0, .8], 2.5, azimuth=135, elevation=-23,
    )
  max_qpos_error = max_qvel_error = max_ratio_error = 0.0
  phase_counts = np.zeros(4, dtype=np.int64)
  pawl_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pawl_slide")])
  qadr = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
          for name in ("drive", "follower")]
  dt = float(model.opt.timestep)
  for step in range(steps):
    phase, ctrl, pawl_force = _sequencer(float(reference.time))
    phase_counts[phase] += 1
    reference.ctrl[:] = ctrl
    reference.qfrc_applied[:] = 0
    reference.qfrc_applied[pawl_dof] = pawl_force
    mujoco.mj_step(model, reference)
    if native is None:
      actual.ctrl[:] = ctrl
      actual.qfrc_applied[:] = 0
      actual.qfrc_applied[pawl_dof] = pawl_force
      mujoco.mj_step(model, actual)
    else:
      force = np.zeros((1, model.nv), dtype=np.float32)
      force[0, pawl_dof] = pawl_force
      native.step(ctrl=ctrl[None], qfrc_applied=force)
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"native joint-constraint status: {state.status.tolist()}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
      actual.time = float(state.time[0])
    max_qpos_error = max(max_qpos_error, float(np.max(np.abs(actual.qpos-reference.qpos))))
    max_qvel_error = max(max_qvel_error, float(np.max(np.abs(actual.qvel-reference.qvel))))
    gear_ratio = actual.qpos[qadr[0]] + actual.qpos[qadr[1]]
    max_ratio_error = max(max_ratio_error, abs(float(gear_ratio)))
    if recorder:
      recorder.frame(step, actual, reference,
          extra=f"sequencer state {phase+1}/4   measured q_drive + q_follower = {gear_ratio:+.5f} rad   pawl {actual.qpos[2]:+.3f} m")
  if recorder:
    recorder.close()
  result = {
      "demo": "clockwork_automaton",
      "mode": mode,
      "profile": PROFILE if native else "MuJoCo CPU",
      "steps": steps,
      "simulated_seconds": steps * dt,
      "features": [
          "polynomial joint equality sets counter-rotating gear ratio",
          "joint frictionloss bounds motor-driven motion",
          "slide-joint limit and friction constrain escapement pawl travel",
          "four-state host sequencer controls a fixed-gain motor",
          "disabled geom contacts; no tooth-impact model",
      ],
      "sequencer_state_counts": phase_counts.tolist(),
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "max_measured_gear_ratio_error_rad": max_ratio_error,
  }
  if check and mode == "metal" and (
      max_qpos_error > 2e-3 or max_qvel_error > 2e-2 or max_ratio_error > 2e-3
  ):
    raise AssertionError(json.dumps(result, indent=2))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--check", action="store_true", help="check a short independent native/CPU rollout")
  parser.add_argument("--record", help="record actual native and CPU renders to a GIF")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and (not args.headless or args.steps > 200):
    parser.error("--check requires --headless and 1..200 steps")
  if args.check and args.mode != "metal":
    parser.error("--check compares native Metal with the independent CPU reference")
  print(json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2))


if __name__ == "__main__":
  main()
