# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Rock-garden axle cart: a passive two-wheel cart crosses procedural terrain.

A free body carries two sphere wheels (fixed to the body, rolling as a rigid
axle) joined by a visual axle capsule. The heightfield combines a start pad,
a sine rock garden, a catch flat and a stop berm; gravity plus a small launch
push drive the cart across multiple terrain cells with Coulomb friction.
Wheel/terrain contacts exercise native per-prism heightfield narrow phase;
the thin axle is intentionally collision-free (contype 0, documented) so the
pair budget covers the two 8-slot wheel pairs. Rest equilibria with friction
are non-unique, so checks gate traverse distance, clearance, impact
transients and settle envelopes plus early parity, not exact rest poses.

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
  if model.nq != 7 or model.nv != 6:
    raise RuntimeError(
        f"Expected single free axle body (7q/6v), got {model.nq}q/{model.nv}v")
  return model


def run(steps=800, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos = model.qpos0.copy().astype(np.float32)
  # Passive launch push along +x; gravity + downhill grade carry the cart
  # through the garden to the catch berm. Modest speed keeps wheels rolling
  # on the ground instead of ballistically overshooting features.
  qvel = np.zeros(model.nv, dtype=np.float32)
  qvel[0] = 0.3

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
        model, record, "Rock-Garden Rover | Heightfield Traverse",
        [0.0, 0.0, 0.3], 3.0, azimuth=135, elevation=-25)

  max_qpos_error = 0.0
  max_quat_error = 0.0
  pre_qpos_error = 0.0
  end_qpos_error = 0.0
  end_quat_error = 0.0
  max_penetration = 0.0
  min_axle_z = float("inf")
  wheel_geoms = {"wheelL_geom", "wheelR_geom"}
  pos_idx = [0, 1, 2]
  quat_idx = [3, 4, 5, 6]
  for step in range(steps):
    reference.ctrl[:] = 0
    mujoco.mj_step(model, reference)
    for c in range(reference.ncon):
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[0]))
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[1]))
      if g1 in wheel_geoms or g2 in wheel_geoms:
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
      end_quat_error = float(np.max(np.abs(
          actual.qpos[quat_idx] - reference.qpos[quat_idx])))
    min_axle_z = min(min_axle_z, float(actual.qpos[2]))
    if recorder is not None:
      extra = f"t={float(actual.time):.2f}s x={float(actual.qpos[0]):.2f}m"
      recorder.frame(step, actual, reference, extra=extra)

  q = actual.qpos
  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_quat_error": max_quat_error,
      "pre_qpos_error": pre_qpos_error,
      "end_qpos_error": end_qpos_error,
      "end_quat_error": end_quat_error,
      "max_penetration": max_penetration,
      "min_axle_z": min_axle_z,
      "start_x": float(qpos[0]),
      "end_x": float(q[0]),
      "travel": float(q[0] - qpos[0]),
      "end_z": float(q[2]),
      "roll_angle": float(2.0 * np.arccos(min(1.0, abs(float(q[3]))))),
  }
  if recorder is not None:
    recorder.close()
  if check:
    # Traverse: rolls at least half a meter down the garden.
    assert out["travel"] > 0.5, out["travel"]
    # Axle never tunnels the solid (axle height stays above terrain base).
    assert out["min_axle_z"] > -0.05, out["min_axle_z"]
    # Body rolled (rotation from identity wound up rolling).
    assert out["roll_angle"] > 1.0, out["roll_angle"]
    # Native/CPU envelopes: early parity tight, rolling orientation uses a
    # phase envelope (contact-phase differences integrate over the
    # traverse, as in the roller workshop), settle positions lane-tight.
    assert pre_qpos_error < 5e-4, pre_qpos_error
    assert end_qpos_error < 2e-2, end_qpos_error
    assert end_quat_error < 0.15, end_quat_error
    assert max_qpos_error < 2e-2, max_qpos_error
    assert max_quat_error < 0.2, max_quat_error
    assert max_penetration < 0.02, max_penetration
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=800)
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
