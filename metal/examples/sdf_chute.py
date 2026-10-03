# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""SDF screw chute: sculpted track guides objects with contact overlay.

A static signed-distance chute (procedural helical tube mesh, authored
vertex lists) guides three spheres from pre-placed bore positions to the
floor. Ball/chute and floor contacts exercise the native Halton/descent
SDF narrow phase with live contact-count and penetration overlay in the
caption band. Cargo-cargo pairs are excluded (staggered starts, budget);
rest equilibria with friction are non-unique, so checks gate traverse/exit,
impact-transient and settle envelopes plus early parity, not exact spots.

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
        f"Expected 3 free bodies (21q/18v), got {model.nq}q/{model.nv}v")
  return model


def run(steps=900, mode="metal", check=False, record=None):
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
        model, record, "SDF Screw Chute",
        [0.0, 0.0, 0.3], 1.6, azimuth=135, elevation=-25)

  max_qpos_error = 0.0
  max_quat_error = 0.0
  pre_qpos_error = 0.0
  end_qpos_error = 0.0
  max_penetration = 0.0
  max_ncon = 0
  sdf_geoms = {"chute"}
  pos_idx = [0, 1, 2, 7, 8, 9, 14, 15, 16]
  quat_idx = [3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20]
  for step in range(steps):
    reference.ctrl[:] = 0
    mujoco.mj_step(model, reference)
    ncon = 0
    for c in range(reference.ncon):
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[0]))
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[1]))
      if g1 in sdf_geoms or g2 in sdf_geoms:
        ncon += 1
        max_penetration = max(max_penetration,
                              -float(reference.contact[c].dist))
    max_ncon = max(max_ncon, ncon)
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
      extra = (f"t={float(actual.time):.2f}s sdf_n={ncon} "
               f"pen={max_penetration * 1000:.1f}mm")
      recorder.frame(step, actual, reference, extra=extra)

  q = actual.qpos
  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_quat_error": max_quat_error,
      "pre_qpos_error": pre_qpos_error,
      "end_qpos_error": end_qpos_error,
      "max_penetration": max_penetration,
      "max_sdf_contacts": max_ncon,
      "ballA_xyz": [float(q[0]), float(q[1]), float(q[2])],
      "ballB_xyz": [float(q[7]), float(q[8]), float(q[9])],
      "ballC_xyz": [float(q[14]), float(q[15]), float(q[16])],
  }
  if recorder is not None:
    recorder.close()
  if check:
    # All pieces exit the chute and land on the floor below the tube.
    # Free floor rolling spreads settle spots (descent-chaos envelopes),
    # so ends gate by floor contact, not exact spots.
    for name, idx in (("A", 0), ("B", 7), ("C", 14)):
      assert float(q[idx + 2]) < 0.15, (name, float(q[idx + 2]))
    # Balls leave the helix laterally (travel away from the drop line).
    assert abs(float(q[0])) + abs(float(q[1])) > 0.05, (float(q[0]), float(q[1]))
    assert max_ncon > 0, "no SDF contact seen"
    # Native/CPU envelopes: clean free-fall start, SDF sliding uses
    # descent-chaos envelopes (matrix tiers); settle spots agree tightly
    # here (guided short traverse, no free floor-roll phase divergence).
    assert pre_qpos_error < 1e-3, pre_qpos_error
    assert end_qpos_error < 1e-2, end_qpos_error
    assert max_qpos_error < 1e-2, max_qpos_error
    assert max_quat_error < 1e-2, max_quat_error
    # Compliant SDF impact transient on light bodies (measured 27 mm when
    # the box negotiates the helix curve); the bound documents it.
    assert max_penetration < 0.035, max_penetration
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=900)
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
