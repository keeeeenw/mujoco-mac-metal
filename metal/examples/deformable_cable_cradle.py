# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Deformable cable cradle demo: native MPS flex physics (Milestone 018).

Demonstrates native Apple Silicon GPU simulation of a 1D deformable cable:
- Lowers flexcomp elements and bilateral distance equality constraints (mjEQ_FLEX).
- Vectorized forward kinematics, point Jacobians, and edge Jacobians on MPS.
- Delassus block-row constraint solve with analytic compliance and reference acceleration.
- Parity qualification against pinned CPU MuJoCo 3.10.0 oracle.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.nflex != 1:
    raise RuntimeError(f"Expected model with 1 flex object, got {model.nflex}")
  return model


def run(steps=300, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")

  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float64)
  qvel0 = np.zeros(model.nv, dtype=np.float64)

  cpu_ref = mujoco.MjData(model)
  cpu_ref.qpos[:] = qpos0
  cpu_ref.qvel[:] = qvel0
  mujoco.mj_forward(model, cpu_ref)

  actual = mujoco.MjData(model)
  actual.qpos[:] = qpos0
  actual.qvel[:] = qvel0
  mujoco.mj_forward(model, actual)

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(
        model, batch_size=1,
        qpos=qpos0[None, :].astype(np.float32),
        qvel=qvel0[None, :].astype(np.float32),
        profile=PROFILE,
    )

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(
        model, record, "Deformable Cable Cradle | Native GPU Flex Physics",
        [0.0, 0.0, 0.8], 2.2, azimuth=135, elevation=-20,
    )

  max_qpos_error = 0.0
  max_edge_error = 0.0
  max_excursion = 0.0
  rest_length = float(np.sum(model.flexedge_length0))

  for step in range(steps):
    mujoco.mj_step(model, cpu_ref)
    if native is None:
      mujoco.mj_step(model, actual)
    else:
      native.step()
      gq = native.state.qpos[0].cpu().numpy()
      gv = native.state.qvel[0].cpu().numpy()
      actual.qpos[:] = gq
      actual.qvel[:] = gv
      actual.time = float(native.state.time[0])

    err = float(np.max(np.abs(actual.qpos - cpu_ref.qpos)))
    max_qpos_error = max(max_qpos_error, err)

    # Track swing excursion (displacement of tip vertex)
    tip_x = abs(float(actual.qpos[-3]))
    tip_z = abs(float(actual.qpos[-1]))
    max_excursion = max(max_excursion, tip_x, tip_z)

    if native is not None:
      nat_edges = native.flexedge_length[0].cpu().numpy()
      edge_err = float(np.max(np.abs(nat_edges - cpu_ref.flexedge_length)))
      max_edge_error = max(max_edge_error, edge_err)

    if recorder is not None:
      extra = f"t={float(actual.time):.2f}s | swing={tip_y:+.2f}m"
      recorder.frame(step, actual, cpu_ref, extra=extra)

  if recorder is not None:
    recorder.close()

  tip_pos = [float(actual.qpos[-3]), float(actual.qpos[-2]), float(actual.qpos[-1])]
  current_length = float(np.sum(actual.flexedge_length if native is None else native.flexedge_length[0].cpu().numpy()))

  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_edge_error": max_edge_error,
      "max_excursion": max_excursion,
      "rest_length": rest_length,
      "current_length": current_length,
      "tip_position": tip_pos,
  }

  if check:
    assert max_qpos_error < 2.0e-3, f"max_qpos_error {max_qpos_error:.3e} exceeds 2e-3"
    assert max_excursion > 0.08, f"cable failed to swing (excursion {max_excursion:.3e})"
    assert abs(current_length - rest_length) < 0.015, f"cable stretch {abs(current_length - rest_length):.3e} exceeded tolerance"

  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  parser.add_argument("--json", help="write numerical report to JSON")
  args = parser.parse_args()

  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")

  out = run(steps=args.steps, mode=args.mode, check=args.check, record=args.record)
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
