# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Marble plinko factory: capacity demo beyond the old pair/slot ceilings.

A marble drops through a 25-peg static field (26 candidate pairs, 26 slots,
~85 rows) with broadphase pruning skipping distant pegs every step.
Headless verification (--check) and side-by-side GIF recording (--record).
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  return model


def run(steps=600, mode="metal", check=False, record=None):
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  model = _load_model()
  desc = lower_coupled_constraints(model)
  peak = {"pairs": int(desc.npairs), "slots": int(desc.ncontacts_max),
          "rows": int(desc.nr)}
  assert peak["pairs"] > 16 and peak["slots"] > 24, peak  # beyond old caps
  assert peak["rows"] <= 96, peak
  qpos = np.asarray(model.qpos0, dtype=np.float64)
  qvel = np.zeros(model.nv)
  reference = mujoco.MjData(model)
  reference.qpos[:] = qpos
  actual = mujoco.MjData(model)
  actual.qpos[:] = qpos

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(
        model, batch_size=1, qpos=qpos[None].astype(np.float32),
        qvel=qvel[None].astype(np.float32), profile=PROFILE)

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(
        model, record, "Marble Plinko Factory | 25 Pairs, Pruned Narrowphase",
        [0.0, 0.0, 0.6], 2.2, azimuth=135, elevation=-25)

  max_qpos_error = 0.0
  min_ball_z = 1e9
  peg_hits = 0
  for step in range(steps):
    mujoco.mj_step(model, reference)
    if native is None:
      mujoco.mj_step(model, actual)
    else:
      native.step()
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(
            f"Native status {state.status.tolist()} at step {step}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
      actual.time = float(state.time[0])
    max_qpos_error = max(
        max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos))))
    min_ball_z = min(min_ball_z, float(actual.qpos[2]))
    for c in range(reference.ncon):
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[0]))
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[1]))
      if "peg" in (g1 or "") or "peg" in (g2 or ""):
        peg_hits += 1
        break
    if recorder is not None:
      extra = f"t={float(actual.time):.2f}s"
      recorder.frame(step, actual, reference, extra=extra)

  out = {
      "steps": steps,
      "peak_pairs": peak["pairs"],
      "peak_slots": peak["slots"],
      "peak_rows": peak["rows"],
      "max_qpos_error": max_qpos_error,
      "min_ball_z": min_ball_z,
      "end_ball_z": float(actual.qpos[2]),
      "peg_hit_steps": peg_hits,
  }
  if recorder is not None:
    recorder.close()
  if check:
    # Marble falls through the peg field (progress), never tunnels the
    # floor, hits pegs along the way, and tracks the CPU trajectory.
    assert out["end_ball_z"] < 1.0, out["end_ball_z"]
    assert out["min_ball_z"] > 0.03, out["min_ball_z"]
    assert out["peg_hit_steps"] > 50, out["peg_hit_steps"]
    assert max_qpos_error < 5e-2, max_qpos_error
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  args = parser.parse_args()
  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")
  out = run(steps=args.steps, mode=args.mode, check=args.check,
            record=args.record)
  print(out)


if __name__ == "__main__":
  main()
