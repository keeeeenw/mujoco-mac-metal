# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Suspended-platform demo: mixed rigid constraints under one coupled solve.

A mocap carrier tilts sinusoidally. A platform hangs from four spatial-tendon
cables (one travel-limited, one pair coupled by a tendon equality) with a
loose cargo crate riding aboard on contact. A ball-jointed pendulum strut
swings into its angular-stop limit. Hinge-free composition: ball limits,
tendon limits, tendon equality, dry contact (pyramidal cone) and mocap-driven
motion all participate in the same coupled problem. A paired level-hold run
isolates carrier-induced motion from the platform settling under its load.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"
TILT_AMP = 0.35
TILT_PERIOD = 1600
NOMINAL_STRUT_STOP = 0.28
NOMINAL_CABLE_LIMIT = 0.66
STRUT_CONFIGURED_STOP = 0.2788
CABLE_CONFIGURED_LIMIT = 0.6578


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


def _tendon_length_at(model, data, qpos, mocap_pos, mocap_quat, tendon_id):
  """Query a tendon length at an accepted state without advancing physics."""
  data.qpos[:] = qpos
  data.mocap_pos[:] = mocap_pos
  data.mocap_quat[:] = mocap_quat
  mujoco.mj_kinematics(model, data)
  mujoco.mj_tendon(model, data)
  return float(data.ten_length[tendon_id])


def run(steps=4800, mode="metal", check=False, record=None):
  if check and steps < 4800:
    raise ValueError("suspension --check requires the complete 4800-step schedule")
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)
  base_pos = np.asarray(model.key_mpos).reshape(-1, 3)[0].astype(np.float32)

  native = None
  native_fk = None
  native_tendon_query = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    native.reset_to_keyframe(0)
    # Independent query workspaces must not overwrite physics FK/row caches.
    from mujoco_metal.metal_kinematics import MetalKinematics
    from mujoco_metal.model import load_model
    from mujoco_metal.spatial_tendons import MetalSpatialTendonDynamics
    native_fk = MetalKinematics(load_model(model), batch_size=1)
    native_tendon_query = MetalSpatialTendonDynamics(model, 1)

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
  # The visible strut/bob geometry is axisymmetric, so compare its axis
  # separately from raw quaternion components. The angular stop still uses
  # the FULL relative rotation angle: axial twist is not unconstrained while
  # that stop is active. Both the axis trajectory and full-angle travel bound
  # are gated below; raw quaternion differences remain separately reported.
  max_tilt_err = 0.0
  end_tilt_err = 0.0
  max_rigid_quat_err = 0.0
  end_rigid_quat_err = 0.0
  nat_strut_peak = 0.0
  nat_cable_peak = 0.0
  cpu_query_cable_peak = 0.0
  max_native_cable_query_err = 0.0
  nat_strut_hits = 0
  _AXIS = np.array([0.0, 0.0, -1.0])
  native_query = mujoco.MjData(model) if native is not None else None

  for step in range(steps):
    quat = _carrier_quat(step)
    if native is not None:
      native.set_mocap(base_pos.reshape(1, 3), quat.reshape(1, 4))
      status = native.step(1)
      np.testing.assert_array_equal(status.cpu().numpy(), [0])
    cpu_swing.mocap_pos[0] = base_pos
    cpu_swing.mocap_quat[0] = quat
    mujoco.mj_step(model, cpu_swing)
    cpu_level.mocap_pos[0] = base_pos
    cpu_level.mocap_quat[0] = np.array([1, 0, 0, 0], dtype=float)
    mujoco.mj_step(model, cpu_level)
    strut_peak = max(strut_peak, _strut_angle(cpu_swing.qpos))
    cable_peak = max(cable_peak, float(np.asarray(cpu_swing.ten_length)[3]))
    if _strut_angle(cpu_swing.qpos) > STRUT_CONFIGURED_STOP - 1e-9:
      strut_hits += 1
    if float(np.asarray(cpu_swing.ten_length)[3]) > CABLE_CONFIGURED_LIMIT - 1e-9:
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
      rigid = float(np.max(np.abs(gq[[7, 8, 9, 10, 14, 15, 16, 17]]
                                    - cpu_swing.qpos[[7, 8, 9, 10, 14, 15, 16, 17]])))
      # Strut-axis tilt parity (twist-excluded): direction of the strut
      # long axis under each engine's ball-quat, compared by angle.
      w0, x0, y0, z0 = (float(v) for v in gq[0:4])
      w1, x1, y1, z1 = (float(v) for v in cpu_swing.qpos[0:4])
      dn = np.array([2 * (x0 * z0 + y0 * w0), 2 * (y0 * z0 - x0 * w0),
                     1 - 2 * (x0 * x0 + y0 * y0)])
      dc = np.array([2 * (x1 * z1 + y1 * w1), 2 * (y1 * z1 - x1 * w1),
                     1 - 2 * (x1 * x1 + y1 * y1)])
      tilt = float(np.arccos(np.clip(dn @ dc / (np.linalg.norm(dn) * np.linalg.norm(dc)),
                             -1.0, 1.0)))
      max_pos_err = max(max_pos_err, epos)
      max_quat_err = max(max_quat_err, equat)
      max_tilt_err = max(max_tilt_err, tilt)
      max_rigid_quat_err = max(max_rigid_quat_err, rigid)
      max_qpos_err = max(max_qpos_err, float(np.max(np.abs(gq - cpu_swing.qpos))))
      ns = _strut_angle(gq)
      nat_strut_peak = max(nat_strut_peak, ns)
      # Measure accepted native qpos using separate Metal FK/tendon queries.
      # The CPU query is a diagnostic at that same pose, not the physics
      # trajectory or the native cable-limit measurement.
      query_poses = native_fk.run_device(
          native.state._qpos,
          mocap_pos=native.state._mpos,
          mocap_quat=native.state._mquat)
      query = native_tendon_query.run_kinematics(native.state._qvel, query_poses)
      native_cable = float(query["length"][0, 3].cpu())
      cpu_query_cable = _tendon_length_at(
          model, native_query, gq, np.asarray([base_pos]),
          np.asarray([quat]), 3)
      cpu_query_cable_peak = max(cpu_query_cable_peak, cpu_query_cable)
      max_native_cable_query_err = max(max_native_cable_query_err,
                                      abs(native_cable-cpu_query_cable))
      nat_cable_peak = max(nat_cable_peak, native_cable)
      if ns > STRUT_CONFIGURED_STOP - 1e-9:
        nat_strut_hits += 1
      if step == steps - 1:
        end_pos_err, end_quat_err = epos, equat
        end_tilt_err, end_rigid_quat_err = tilt, rigid
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
      status = lat_sim.step(1)
      np.testing.assert_array_equal(status.cpu().numpy(), [0])
    lat = lat_sim.state.qpos[0].cpu().numpy()
    level = (float(np.max(np.abs(lat[plat_idx:plat_idx + 1]))), float(lat[cargo_idx + 2]))
  else:
    swing = (float(np.max(np.abs(cpu_swing.qpos[plat_idx:plat_idx + 1]))), float(cpu_swing.qpos[cargo_idx + 2]))
    level = (float(np.max(np.abs(cpu_level.qpos[plat_idx:plat_idx + 1]))), float(cpu_level.qpos[cargo_idx + 2]))

  strut_tip_overrun = max(0.0, strut_peak - NOMINAL_STRUT_STOP) * 0.5
  cable_limit_overrun = max(0.0, cable_peak - NOMINAL_CABLE_LIMIT)
  native_strut_tip_overrun = max(
      0.0, nat_strut_peak - NOMINAL_STRUT_STOP) * 0.5
  native_cable_limit_overrun = max(
      0.0, nat_cable_peak - NOMINAL_CABLE_LIMIT)
  result = {
      "max_qpos_err": max_qpos_err,
      "max_pos_err": max_pos_err,
      "max_quat_err": max_quat_err,
      "max_tilt_err": max_tilt_err,
      "max_rigid_quat_err": max_rigid_quat_err,
      "end_pos_err": end_pos_err,
      "end_quat_err": end_quat_err,
      "end_tilt_err": end_tilt_err,
      "end_rigid_quat_err": end_rigid_quat_err,
      "nat_strut_peak": nat_strut_peak,
      "nat_cable_peak": nat_cable_peak if native is not None else None,
      "native_cable_measurement": ("independent native Metal FK and tendon query"
          if native is not None else None),
      "cpu_query_cable_peak_on_native_state": cpu_query_cable_peak if native is not None else None,
      "max_native_cable_query_error_m": max_native_cable_query_err if native is not None else None,
      "nat_strut_hits": nat_strut_hits,
      "strut_configured_stop": STRUT_CONFIGURED_STOP,
      "cable_configured_limit": CABLE_CONFIGURED_LIMIT,
      "strut_tip_overrun_m": strut_tip_overrun,
      "cable_limit_overrun_m": cable_limit_overrun,
      "native_strut_tip_overrun_m": native_strut_tip_overrun,
      "native_cable_limit_overrun_m": (
          native_cable_limit_overrun if native is not None else None),
      "strut_peak": strut_peak,
      "cable_peak": cable_peak,
      "cargo_contacts": cargo_contacts,
      "strut_hits": strut_hits,
      "cable_hits": cable_hits,
      "swing": swing,
      "level": level,
      "steps": steps,
      "minimum_geometry_distance": clearance.minimum,
      "conservative_geometry_distance_lower_bound": clearance.reported,
      "geometry_distance_uncertainty_bound": clearance.uncertainty,
      # These strict distances are evaluated at accepted qpos (native qpos in
      # Metal mode); the lower-bound ensemble is reported separately.
      "max_actual_hardware_overlap_m": max(
          0.0, -min(float(value) for value in clearance.minimum.values())),
  }
  if check:
    clearance.check()
  if check and mode == "metal":
    assert max_pos_err < 5e-3, max_pos_err
    # Gate platform/cargo orientations, the observable strut-axis trajectory
    # and the full relative ball-angle travel bound. Report raw quaternion
    # components separately rather than implying exact long-horizon parity.
    assert max_rigid_quat_err < 1.5e-2, max_rigid_quat_err
    assert max_tilt_err < 8e-2, max_tilt_err
    assert end_pos_err < 5e-3 and end_rigid_quat_err < 5e-3, (end_pos_err, end_rigid_quat_err)
    assert end_tilt_err < 5e-2, end_tilt_err
    assert nat_strut_peak > STRUT_CONFIGURED_STOP, nat_strut_peak
    assert nat_strut_hits > 100, nat_strut_hits
    assert max_native_cable_query_err < 2e-6, max_native_cable_query_err
    assert nat_cable_peak > CABLE_CONFIGURED_LIMIT, nat_cable_peak
    assert strut_peak > STRUT_CONFIGURED_STOP, strut_peak
    assert cable_peak > CABLE_CONFIGURED_LIMIT, cable_peak
    assert native_strut_tip_overrun <= 0.001, native_strut_tip_overrun
    assert strut_tip_overrun <= 0.001, strut_tip_overrun
    assert cable_limit_overrun <= 0.001, cable_limit_overrun
    assert native_cable_limit_overrun <= 0.001, native_cable_limit_overrun
    assert cargo_contacts > 100, cargo_contacts
    assert swing[0] > 0.02 or swing[1] > 1.0, swing  # platform moves
    assert level[0] < 0.035, level  # level hold stays put (settle transient)
  return result


def run_viewer(steps=4800, seconds=0):
  from mujoco_metal import MetalSimulation
  from demo_viewer import run_native_viewer
  model=_load_model()
  sim=MetalSimulation(model,1,profile=PROFILE)
  sim.reset_to_keyframe(0)
  base=np.asarray(model.key_mpos).reshape(-1,3)[0].astype(np.float32)
  def before_step(sim,step):
    sim.set_mocap(base.reshape(1,3),_carrier_quat(step).reshape(1,4))
  return run_native_viewer(model,sim,steps=steps,seconds=seconds,
      title="Suspension platform — native Metal",lookat=(0,-.1,1),
      distance=2.8,azimuth=120,elevation=-20,before_step=before_step)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=4800)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true", help="check native/CPU rollout")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  parser.add_argument("--json", help="write numerical report to JSON")
  parser.add_argument("--viewer-seconds", type=float, default=0,
                      help="auto-close the native Mac viewer after this wall time")
  args = parser.parse_args()
  if not np.isfinite(args.viewer_seconds) or args.viewer_seconds<0:
    raise SystemExit("--viewer-seconds must be finite and nonnegative")
  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")
  out = (run_viewer(steps=args.steps, seconds=args.viewer_seconds)
         if not args.headless and args.record is None and args.mode=="metal"
         else run(steps=args.steps, mode=args.mode, check=args.check, record=args.record))
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
