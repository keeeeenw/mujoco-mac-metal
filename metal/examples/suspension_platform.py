# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Suspended-platform demo: mixed rigid constraints under one coupled solve.

A mocap carrier tilts sinusoidally. A platform hangs from four spatial-tendon
cables (one travel-limited, one pair coupled by a tendon equality) with a
loose cargo crate riding aboard on contact. A ball-jointed pendulum strut
swings into its angular-stop limit. Hinge-free composition: ball limits,
tendon limits, tendon equality, dry contact (pyramidal cone) and mocap-driven
motion all participate in the same coupled problem. A paired level-hold run
shows the constraints never spuriously engage.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"
TILT_AMP = 0.35
TILT_PERIOD = 400


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.ntendon != 4 or model.neq != 1:
    raise RuntimeError(
        f"Expected suspension rig with 4 tendons and 1 equality, "
        f"got ntendon={model.ntendon} neq={model.neq}")
  return model


def _carrier_quat(step):
  ang = TILT_AMP * math.sin(2 * math.pi * step / TILT_PERIOD)
  return np.array([math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0], dtype=np.float32)


def _strut_angle(qpos):
  q = np.asarray(qpos[0:4], dtype=float)
  return 2.0 * math.acos(min(1.0, abs(q[0])))


def run(steps=1200, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)
  base_pos = np.asarray(model.key_mpos).reshape(-1, 3)[0].astype(np.float32)

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    native.reset_to_keyframe(0)

  cpu_swing = mujoco.MjData(model)
  cpu_level = mujoco.MjData(model)
  for d in (cpu_swing, cpu_level):
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)

  from demo_clearance import ClearanceMonitor
  clearance = ClearanceMonitor(model, [
      (strut, cargo) for strut in ("strut_geom", "bob_geom")
      for cargo in ("platform_geom", "cargo_geom")
  ])

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(model, record, "Suspension Platform | Mixed Constraints",
                                  [0.0, -0.1, 1.0], 2.8, azimuth=120, elevation=-20)

  max_qpos_err = 0.0
  max_pos_err = 0.0
  max_quat_err = 0.0
  end_pos_err = 0.0
  end_quat_err = 0.0
  strut_peak = 0.0
  cable_peak = 0.0
  cargo_contacts = 0
  strut_hits = 0
  cable_hits = 0

  for step in range(steps):
    quat = _carrier_quat(step)
    if native is not None:
      native.set_mocap(base_pos.reshape(1, 3), quat.reshape(1, 4))
      native.step(1)
    cpu_swing.mocap_pos[0] = base_pos
    cpu_swing.mocap_quat[0] = quat
    mujoco.mj_step(model, cpu_swing)
    cpu_level.mocap_pos[0] = base_pos
    cpu_level.mocap_quat[0] = np.array([1, 0, 0, 0], dtype=float)
    mujoco.mj_step(model, cpu_level)
    strut_peak = max(strut_peak, _strut_angle(cpu_swing.qpos))
    cable_peak = max(cable_peak, float(np.asarray(cpu_swing.ten_length)[3]))
    if _strut_angle(cpu_swing.qpos) > 0.28 - 1e-9:
      strut_hits += 1
    if float(np.asarray(cpu_swing.ten_length)[3]) > 0.66 - 1e-9:
      cable_hits += 1
    for c in range(cpu_swing.ncon):
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(cpu_swing.contact[c].geom[0]))
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(cpu_swing.contact[c].geom[1]))
      if "cargo" in g1 or "cargo" in g2:
        cargo_contacts += 1
        break
    if native is not None:
      gq = native.state.qpos[0].cpu().numpy()
      # qpos layout: ball quat[0:4] + platform xyz[4:7] + quat[7:11] +
      # cargo xyz[11:14] + quat[14:18]. Translations vs orientations
      # compared separately: contact transients hit rotation hardest.
      epos = float(np.max(np.abs(gq[[4, 5, 6, 11, 12, 13]]
                                   - cpu_swing.qpos[[4, 5, 6, 11, 12, 13]])))
      equat = float(np.max(np.abs(gq[[0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17]]
                                    - cpu_swing.qpos[[0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17]])))
      max_pos_err = max(max_pos_err, epos)
      max_quat_err = max(max_quat_err, equat)
      max_qpos_err = max(max_qpos_err, float(np.max(np.abs(gq - cpu_swing.qpos))))
      if step == steps - 1:
        end_pos_err, end_quat_err = epos, equat
    clearance.sample(
        native.state.qpos[0].cpu().numpy() if native is not None else cpu_swing.qpos,
        cpu_swing.mocap_pos, cpu_swing.mocap_quat)
    if recorder and native is not None:
      actual = mujoco.MjData(model)
      actual.qpos[:] = native.state.qpos[0].cpu().numpy()
      actual.qvel[:] = native.state.qvel[0].cpu().numpy()
      actual.mocap_pos[0] = base_pos
      actual.mocap_quat[0] = quat
      actual.time = float(native.state.time[0].cpu().numpy())
      extra = f"t={actual.time:.2f}s | strut={_strut_angle(actual.qpos):.2f}"
      recorder.frame(step, actual, cpu_swing, extra=extra)

  if recorder:
    recorder.close()

  plat_idx, cargo_idx = 4, 11
  if mode == "metal" and native is not None:
    rel = native.state.qpos[0].cpu().numpy()
    swing = (float(np.max(np.abs(rel[plat_idx:plat_idx + 1]))), float(rel[cargo_idx + 2]))
    from mujoco_metal import MetalSimulation as MS
    lat_sim = MS(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    lat_sim.reset_to_keyframe(0)
    for _ in range(steps):
      lat_sim.set_mocap(base_pos.reshape(1, 3), np.array([[1, 0, 0, 0]], dtype=np.float32))
      lat_sim.step(1)
    lat = lat_sim.state.qpos[0].cpu().numpy()
    level = (float(np.max(np.abs(lat[plat_idx:plat_idx + 1]))), float(lat[cargo_idx + 2]))
  else:
    swing = (float(np.max(np.abs(cpu_swing.qpos[plat_idx:plat_idx + 1]))), float(cpu_swing.qpos[cargo_idx + 2]))
    level = (float(np.max(np.abs(cpu_level.qpos[plat_idx:plat_idx + 1]))), float(cpu_level.qpos[cargo_idx + 2]))

  result = {
      "max_qpos_err": max_qpos_err,
      "max_pos_err": max_pos_err,
      "max_quat_err": max_quat_err,
      "end_pos_err": end_pos_err,
      "end_quat_err": end_quat_err,
      "strut_peak": strut_peak,
      "cable_peak": cable_peak,
      "cargo_contacts": cargo_contacts,
      "strut_hits": strut_hits,
      "cable_hits": cable_hits,
      "swing": swing,
      "level": level,
      "steps": steps,
      "minimum_geometry_distance": clearance.minimum,
  }
  if check:
    clearance.check()
  if check and mode == "metal":
    assert max_pos_err < 5e-3, max_pos_err
    assert max_quat_err < 1.5e-2, max_quat_err
    assert end_pos_err < 5e-3 and end_quat_err < 5e-3, (end_pos_err, end_quat_err)
    assert strut_peak > 0.28, strut_peak  # ball stop engaged
    assert cable_peak > 0.66, cable_peak  # cable limit engaged
    assert cargo_contacts > 100, cargo_contacts
    assert swing[0] > 0.02 or swing[1] > 1.0, swing  # platform moves
    assert level[0] < 0.035, level  # level hold stays put (settle transient)
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=1200)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true", help="check native/CPU rollout")
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
