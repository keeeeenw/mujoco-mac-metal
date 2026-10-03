# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Gravity assembly puzzle: convex mesh pieces settle into a tray.

Three procedural convex hulls (tetrahedron, brick, octahedron; authored
vertex lists, clearly licensed as project originals) drop into a walled
tray under gravity with Coulomb friction. Piece/tray and piece/piece
contacts exercise native mesh-plane, mesh-box and mesh-mesh narrow phase;
piece rest poses come from rigid-body dynamics (rest equilibria with
friction are non-unique, so checks gate containment and CPU parity
envelopes, not exact poses).

Headless verification (--check) and side-by-side GIF recording (--record).
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
  if model.nq != 21 or model.nv != 18:
    raise RuntimeError(
        f"Expected 3 free pieces (21q/18v), got {model.nq}q/{model.nv}v")
  return model


def run(steps=600, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)

  actual = mujoco.MjData(model)
  reference = mujoco.MjData(model)
  for data in (actual, reference):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(
        model, batch_size=1, qpos=qpos[None], qvel=qvel[None],
        profile=PROFILE)

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(
        model, record, "Assembly Puzzle | Convex Mesh Settling",
        [0.0, 0.0, 0.25], 1.1, azimuth=135, elevation=-25)

  max_qpos_error = 0.0
  max_quat_error = 0.0
  pre_qpos_error = 0.0
  end_qpos_error = 0.0
  max_penetration = 0.0
  piece_geoms = {"tetra_geom", "brick_geom", "octa_geom"}
  pos_idx = [0, 1, 2, 7, 8, 9, 14, 15, 16]
  quat_idx = [3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20]
  for step in range(steps):
    reference.ctrl[:] = 0
    mujoco.mj_step(model, reference)
    for c in range(reference.ncon):
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[0]))
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[1]))
      if g1 in piece_geoms or g2 in piece_geoms:
        max_penetration = max(max_penetration,
                              -float(reference.contact[c].dist))
    if native is None:
      actual.ctrl[:] = 0
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
        max_qpos_error, float(np.max(np.abs(
            actual.qpos[pos_idx] - reference.qpos[pos_idx]))))
    max_quat_error = max(
        max_quat_error, float(np.max(np.abs(
            actual.qpos[quat_idx] - reference.qpos[quat_idx]))))
    if step < 30:
      pre_qpos_error = max(
          pre_qpos_error, float(np.max(np.abs(
              actual.qpos[pos_idx] - reference.qpos[pos_idx]))))
    if step == steps - 1:
      end_qpos_error = float(np.max(np.abs(
          actual.qpos[pos_idx] - reference.qpos[pos_idx])))
    if recorder is not None:
      extra = f"t={float(actual.time):.2f}s"
      recorder.frame(step, actual, reference, extra=extra)

  q = actual.qpos
  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_quat_error": max_quat_error,
      "pre_qpos_error": pre_qpos_error,
      "end_qpos_error": end_qpos_error,
      "max_penetration": max_penetration,
      "pieceA_xyz": [float(q[0]), float(q[1]), float(q[2])],
      "pieceB_xyz": [float(q[7]), float(q[8]), float(q[9])],
      "pieceC_xyz": [float(q[14]), float(q[15]), float(q[16])],
  }
  if recorder is not None:
    recorder.close()
  if check:
    # Pieces rest in their tray lanes (divider at x=0.02 separates A on
    # the left from B/C on the right) below rim height.
    ax, ay, az = float(q[0]), float(q[1]), float(q[2])
    bx, by, bz = float(q[7]), float(q[8]), float(q[9])
    cx, cy, cz = float(q[14]), float(q[15]), float(q[16])
    assert -0.24 < ax < 0.0, (ax,)
    assert 0.04 < bx < 0.24, (bx,)
    assert 0.04 < cx < 0.24, (cx,)
    for name, y in (("A", ay), ("B", by), ("C", cy)):
      assert abs(y) < 0.18, (name, y)
    for name, z in (("A", az), ("B", bz), ("C", cz)):
      assert z < 0.15, (name, z)
    # Native/CPU trajectories agree on positions (contact-transient
    # envelope); orientations accumulate contact-phase differences and
    # use a phase envelope, as in the roller workshop. Pre-impact flight
    # agrees tightly; the settled end state agrees to lane precision.
    assert pre_qpos_error < 2e-4, pre_qpos_error
    assert end_qpos_error < 2e-2, end_qpos_error
    assert max_qpos_error < 2e-2, max_qpos_error
    assert max_quat_error < 5e-2, max_quat_error
    # Compliant impact transient on light hulls (measured ~16 mm on the
    # first tray strike); the bound documents it, not zero penetration.
    assert max_penetration < 0.02, max_penetration
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  parser.add_argument("--json", help="write numerical report to JSON")
  args = parser.parse_args()
  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")
  out = run(steps=args.steps, mode=args.mode, check=args.check,
            record=args.record)
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
