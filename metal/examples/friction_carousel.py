# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Friction carousel: stacked boxes ride a turntable through grip stations.

A motor-driven turntable (hinge + velocity servo, constant yaw rate) carries
a stacked pair on its high-grip half and a single box on its low-grip half
(different floor materials). The high-grip stack sticks and rides around;
the low-grip box slips and spins relative to the platform. Slip speed (each
rider vs platform surface) and stack gap are overlaid live.
(A mocap-driven platform was tried first: MuJoCo exposes no mocap surface
velocity to contacts, so kinematic platforms transmit no drag on either
engine; the dynamic hinge drive is the correct design.) Cargo-cargo pairs
other than the stack itself are excluded (budget); rest equilibria are
friction-non-unique, so checks gate station contrast, traverse,
impact-transient and settle envelopes plus early parity.

Headless verification (--check) and side-by-side GIF recording (--record).
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"
OMEGA = 1.5


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.nq != 22 or model.nv != 19 or model.nu != 1:
    raise RuntimeError(
        f"Expected hinge + 3 free boxes + motor (22q/19v/1u), got "
        f"{model.nq}q/{model.nv}v/{model.nu}u")
  return model


def run(steps=700, mode="metal", check=False, record=None):
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
        model, record, "Friction Carousel | Grip Stations",
        [0.0, 0.0, 0.25], 1.4, azimuth=135, elevation=-25)

  max_qpos_error = 0.0
  max_quat_error = 0.0
  pre_qpos_error = 0.0
  end_qpos_error = 0.0
  max_penetration = 0.0
  slip_high = []
  slip_low = []
  stack_gap_max = 0.0
  rider_geoms = {"rider_high_geom", "rider_low_geom", "stacker_geom"}
  # qpos layout: hinge angle [0], rider_high xyz [1:4] + quat [4:8],
  # rider_low xyz [8:11] + quat [11:15], stacker xyz [15:18] + quat [18:22].
  # qvel layout: hinge [0], rider_high xyz [1:4] + ang [4:7],
  # rider_low xyz [7:10] + ang [10:13], stacker xyz [13:16] + ang [16:19].
  pos_idx = [1, 2, 3, 8, 9, 10, 15, 16, 17]
  quat_idx = [4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21]

  def drive(qvel_state):
    # Velocity servo on the turntable hinge (dof 0).
    return np.array([4.0 * (OMEGA - float(qvel_state[0]))], dtype=float)

  for step in range(steps):
    ctrl = drive(reference.qvel)
    if native is not None:
      native.step(1, ctrl=np.asarray(ctrl, dtype=np.float32).reshape(1, 1))
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(
            f"Native status {state.status.tolist()} at step {step}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
      actual.time = float(state.time[0])
    reference.ctrl[:] = ctrl
    mujoco.mj_step(model, reference)
    if native is None:
      actual.qpos[:] = reference.qpos
      actual.qvel[:] = reference.qvel
      actual.time = reference.time
    for c in range(reference.ncon):
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[0]))
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                             int(reference.contact[c].geom[1]))
      if g1 in rider_geoms or g2 in rider_geoms:
        max_penetration = max(max_penetration,
                              -float(reference.contact[c].dist))
    # Slip per rider: COM velocity vs platform surface velocity below it.
    # Platform speed comes from the hinge state itself.
    om = float(actual.qvel[0])
    for tag, qx, qy, vx, vy in (
        ("HIGH", float(actual.qpos[1]), float(actual.qpos[2]),
         float(actual.qvel[1]), float(actual.qvel[2])),
        ("LOW", float(actual.qpos[8]), float(actual.qpos[9]),
         float(actual.qvel[7]), float(actual.qvel[8])),
    ):
      surf = np.array([-om * qy, om * qx])
      slip = float(np.linalg.norm(np.array([vx, vy]) - surf))
      (slip_high if tag == "HIGH" else slip_low).append(slip)
    zone = f"H={slip_high[-1]:.2f}/L={slip_low[-1]:.2f}"
    stack_gap_max = max(stack_gap_max,
                        float(actual.qpos[17] - actual.qpos[3]))
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
      extra = (f"t={float(actual.time):.2f}s {zone}m/s")
      recorder.frame(step, actual, reference, extra=extra)

  q = actual.qpos
  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_quat_error": max_quat_error,
      "pre_qpos_error": pre_qpos_error,
      "end_qpos_error": end_qpos_error,
      "max_penetration": max_penetration,
      "slip_high_mean": float(np.mean(slip_high)) if slip_high else 0.0,
      "slip_low_mean": float(np.mean(slip_low)) if slip_low else 0.0,
      "stack_gap_max": stack_gap_max,
      "rider_high_end": [float(q[1]), float(q[2]), float(q[3])],
      "rider_low_end": [float(q[8]), float(q[9]), float(q[10])],
  }
  if recorder is not None:
    recorder.close()
  if check:
    # Grip contrast: low rider slips several times more than high stack.
    assert out["slip_low_mean"] > 3 * out["slip_high_mean"] + 0.02, out
    # Stack survives (stacker stays above rider deck height).
    assert out["stack_gap_max"] < 0.15, out["stack_gap_max"]
    # High stack swept with the platform (traverse, not stuck at start).
    assert abs(float(q[1]) - float(qpos[1])) + abs(float(q[2]) - float(qpos[2])) > 0.05, \
        (float(q[1]), float(q[2]))
    assert pre_qpos_error < 5e-4, pre_qpos_error
    assert end_qpos_error < 3e-2, end_qpos_error
    assert max_qpos_error < 3e-2, max_qpos_error
    assert max_quat_error < 0.2, max_quat_error
    assert max_penetration < 0.02, max_penetration
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=700)
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
