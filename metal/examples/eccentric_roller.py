# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Eccentric roller workshop: cylinders and ellipsoids sort on a tilted table.

Rollers (axes across the slope, high sliding friction) roll downhill into the
far catch bin, while nonuniform ellipsoids (low sliding friction) slide and
tumble into the near bin. Rolling motion emerges from rigid-body dynamics
with sliding (Coulomb) friction; no rolling-resistance torque is modeled
(all contacts use the default contact dimensionality).

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


def run(steps=900, mode="metal", check=False, record=None, replay_check=False,
        viewer_seconds=0):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  if record and mode != "metal":
    raise ValueError("Native/CPU comparison recording requires --mode metal")
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
  if viewer_seconds:
    if native is None:
      raise ValueError("The Mac viewer requires --mode metal")
    from demo_viewer import run_native_viewer
    return run_native_viewer(
        model, native, steps=steps, seconds=viewer_seconds,
        title="Eccentric Roller Workshop | Native Metal",
        lookat=(.15, 0, .02), distance=1.9, azimuth=120, elevation=-35)

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(
        model, record, "Eccentric Roller Workshop | Integrated Euler",
        [.15, 0, .02], 1.9, azimuth=120, elevation=-35)

  max_qpos_error = 0.0
  max_quat_error = 0.0
  max_penetration = 0.0
  native_max_penetration = 0.0
  native_contact_steps = 0
  replay_steps = []
  from demo_clearance import ClearanceMonitor
  clearance = ClearanceMonitor(model, [(name, "ridge") for name in
                                      ("roll1", "tumb1", "tumb2")])
  piece_geoms = {"roll1", "tumb1", "tumb2"}
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
      contacts = native._last_coupled
      mask = contacts["contact_mask"][0].cpu().numpy().astype(bool)
      native_contact_steps += int(mask.any())
      if mask.any():
        native_max_penetration = max(native_max_penetration, -float(
            contacts["contact_distance"][0].cpu().numpy()[mask].min()))
      if replay_check and step + 1 in (300, 700):
        from magnetic_crane import _same_checkpoint
        before = native.snapshot()
        for _ in range(20):
          status = native.step()
          np.testing.assert_array_equal(status.cpu().numpy(), [0])
        expected = native.snapshot()
        native.restore(before)
        for _ in range(20):
          status = native.step()
          np.testing.assert_array_equal(status.cpu().numpy(), [0])
        _same_checkpoint(native.snapshot(), expected)
        native.restore(before)
        replay_steps.append(step + 1)
    clearance.sample(actual.qpos)
    max_qpos_error = max(
        max_qpos_error, float(np.max(np.abs(
            actual.qpos[pos_idx] - reference.qpos[pos_idx]))))
    max_quat_error = max(
        max_quat_error, float(np.max(np.abs(
            actual.qpos[quat_idx] - reference.qpos[quat_idx]))))
    if recorder is not None:
      recorder.frame(step, actual, reference)

  # Free-joint x addresses: roller 0, tumbler1 7, tumbler2 14.
  q = actual.qpos
  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_quat_error": max_quat_error,
      "max_penetration": max_penetration,
      "cpu_max_penetration": max_penetration,
      "native_max_penetration": native_max_penetration if native else None,
      "native_contact_steps": native_contact_steps if native else None,
      "minimum_divider_clearance": clearance.minimum,
      "divider_clearance_lower_bound": clearance.reported,
      "full_checkpoint_replay_steps": replay_steps,
      "roller_x": float(q[0]),
      "tumbler1_x": float(q[7]),
      "tumbler2_x": float(q[14]),
      "roller_y": float(q[1]),
      "tumbler1_y": float(q[8]),
      "tumbler2_y": float(q[15]),
  }
  if recorder is not None:
    recorder.close()
  if check:
    # Roller reaches the far rail; tumblers stay in the near rail zone.
    assert 0.70 < float(q[0]) < 0.95, q[0]
    assert 0.10 < float(q[7]) < 0.45, q[7]
    assert 0.05 < float(q[14]) < 0.45, q[14]
    # Lanes kept: no cross-lane travel (justifies pair filtering).
    for y, lane in ((q[1], -0.2), (q[8], 0.2), (q[15], 0.2)):
      assert abs(float(y) - lane) < 0.08, (y, lane)
    assert max_qpos_error < 5e-3, max_qpos_error
    # Rolling/spinning attitude accumulates phase at 0.05%/radian rates;
    # positions (above) carry trajectory parity, quaternions get a phase
    # envelope instead.
    assert max_quat_error < 5e-2, max_quat_error
    clearance.check()
    if native is not None:
      assert native_contact_steps > steps // 2, native_contact_steps
      if replay_check:
        assert replay_steps == [300, 700], replay_steps
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=900)
  parser.add_argument("--mode", default="metal")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", default=None)
  parser.add_argument("--replay-check", action="store_true")
  parser.add_argument("--viewer-seconds", type=float, default=0)
  args = parser.parse_args()
  out = run(steps=args.steps, mode=args.mode, check=args.check,
            record=args.record, replay_check=args.replay_check,
            viewer_seconds=args.viewer_seconds)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
