# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Antagonistic muscle-powered gripper demo with mocap gantry.

A mocap gantry carries two muscle-driven fingers (antagonistic flexor/
extensor pairs with MuJoCo muscle dynamics, force-length-velocity gains and
passive biases). Parallel jaws include supporting ledges; the schedule clears
the bin walls and pedestal. Phase A delivers a sphere to a bin (grasp, carry,
release). Phase B delivers a second sphere onto a pedestal (grasp,
carry, set-down release). Gantry motion and finger schedules are deterministic
open-loop inputs identical in CPU/native runs; a paired always-open run shows
the physical effect of grasping.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"

OPEN = [0.05, 0.9, 0.05, 0.9]
TOUCH = [0.6, 0.2, 0.6, 0.2]
FIRM = [1.0, 0.0, 1.0, 0.0]
RELEASE = [0.05, 0.5, 0.05, 0.5]
CHANNEL = [0.35, 0.45, 0.35, 0.45]

# (step, x, z, ctrl)
WPS = [
    (0, 0.0, 0.62, OPEN),
    (100, 0.0, 0.62, OPEN),
    (250, 0.0, 0.15, OPEN),
    (350, 0.0, 0.15, TOUCH),
    (500, 0.0, 0.15, FIRM),
    (700, 0.53, 0.40, FIRM),
    (1000, 0.53, 0.40, FIRM),
    (1100, 0.53, 0.37, FIRM),
    (1150, 0.53, 0.37, OPEN),
    (1300, 0.53, 0.45, OPEN),
    (1450, 0.30, 0.45, OPEN),
    (1600, 0.30, 0.15, OPEN),
    (1750, 0.30, 0.15, FIRM),
    (1900, 0.30, 0.40, FIRM),
    (2400, -0.5, 0.40, FIRM),
    (2500, -0.5, 0.38, FIRM),
    (2550, -0.5, 0.38, OPEN),
    (2700, -0.5, 0.50, OPEN),
    (3000, -0.5, 0.50, OPEN),
]


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.nu != 4 or model.na != 4 or model.nmocap != 1:
    raise RuntimeError(
        f"Expected muscle gripper with 4 actuators, 4 slots, 1 mocap, "
        f"got nu={model.nu} na={model.na} nmocap={model.nmocap}")
  return model


def _sched(step):
  if step <= WPS[0][0]:
    return np.array([WPS[0][1], 0, WPS[0][2]]), WPS[0][3]
  for k in range(len(WPS) - 1):
    t0, x0, z0, _ = WPS[k]
    t1, x1, z1, _ = WPS[k + 1]
    if step <= t1:
      s = (step - t0) / max(1, t1 - t0)
      e = s * s * (3 - 2 * s)
      u = WPS[k + 1][3] if s > 0 else WPS[k][3]
      return np.array([x0 + (x1 - x0) * e, 0, z0 + (z1 - z0) * e]), u
  return np.array([WPS[-1][1], 0, WPS[-1][2]]), WPS[-1][3]


def _clearance_monitor(model):
  """Monitor moving hardware/furniture and jaw self-clearance pairs."""
  from demo_clearance import ClearanceMonitor
  return ClearanceMonitor(
      model,
      [(moving, fixed)
       for moving in ("palm_geom", "fingerL_geom", "fingerR_geom",
                      "ledgeL", "ledgeR")
       for fixed in ("bin_geom", "wallL_geom", "wallR_geom",
                     "pedestal_geom", "floor")]
      + [("fingerL_geom", "fingerR_geom"),
         ("ledgeL", "ledgeR"),
         ("ledgeL", "fingerR_geom"), ("ledgeR", "fingerL_geom"),
         ("ball_geom", "ball2_geom"), ("ball_geom", "pedestal_geom"),
         ("ball2_geom", "bin_geom"), ("ball2_geom", "wallL_geom"),
         ("ball2_geom", "wallR_geom")],
      [(ball, other)
       for ball in ("ball_geom", "ball2_geom")
       for other in ("palm_geom", "fingerL_geom", "fingerR_geom",
                     "ledgeL", "ledgeR", "floor")]
      + [("ball_geom", "bin_geom"), ("ball_geom", "wallL_geom"),
         ("ball_geom", "wallR_geom"), ("ball2_geom", "pedestal_geom")])


def _state_error_breakdown(model, qpos_a, qvel_a, qpos_b, qvel_b):
  """Return unit-separated free-body and scalar-joint state errors."""
  errors = {
      "translation_m": 0.0,
      "orientation_geodesic_rad": 0.0,
      "linear_velocity_m_s": 0.0,
      "angular_velocity_rad_s": 0.0,
      "scalar_joint_position": 0.0,
      "scalar_joint_velocity": 0.0,
  }
  free = int(mujoco.mjtJoint.mjJNT_FREE)
  ball = int(mujoco.mjtJoint.mjJNT_BALL)
  for joint in range(model.njnt):
    kind = int(model.jnt_type[joint])
    qa = int(model.jnt_qposadr[joint])
    va = int(model.jnt_dofadr[joint])
    if kind == free:
      errors["translation_m"] = max(
          errors["translation_m"],
          float(np.linalg.norm(qpos_a[qa:qa + 3] - qpos_b[qa:qa + 3])))
      qa_quat = np.asarray(qpos_a[qa + 3:qa + 7], dtype=np.float64).copy()
      qb_quat = np.asarray(qpos_b[qa + 3:qa + 7], dtype=np.float64).copy()
      qa_quat /= max(float(np.linalg.norm(qa_quat)), 1e-30)
      qb_quat /= max(float(np.linalg.norm(qb_quat)), 1e-30)
      dot = float(np.clip(abs(np.dot(qa_quat, qb_quat)), 0.0, 1.0))
      errors["orientation_geodesic_rad"] = max(
          errors["orientation_geodesic_rad"], 2.0 * float(np.arccos(dot)))
      errors["linear_velocity_m_s"] = max(
          errors["linear_velocity_m_s"],
          float(np.linalg.norm(qvel_a[va:va + 3] - qvel_b[va:va + 3])))
      errors["angular_velocity_rad_s"] = max(
          errors["angular_velocity_rad_s"],
          float(np.linalg.norm(qvel_a[va + 3:va + 6] - qvel_b[va + 3:va + 6])))
    elif kind == ball:
      qa_quat = np.asarray(qpos_a[qa:qa + 4], dtype=np.float64).copy()
      qb_quat = np.asarray(qpos_b[qa:qa + 4], dtype=np.float64).copy()
      qa_quat /= max(float(np.linalg.norm(qa_quat)), 1e-30)
      qb_quat /= max(float(np.linalg.norm(qb_quat)), 1e-30)
      dot = float(np.clip(abs(np.dot(qa_quat, qb_quat)), 0.0, 1.0))
      errors["orientation_geodesic_rad"] = max(
          errors["orientation_geodesic_rad"], 2.0 * float(np.arccos(dot)))
      errors["angular_velocity_rad_s"] = max(
          errors["angular_velocity_rad_s"],
          float(np.linalg.norm(qvel_a[va:va + 3] - qvel_b[va:va + 3])))
    else:
      qpos_width = 1
      qvel_width = 1
      errors["scalar_joint_position"] = max(
          errors["scalar_joint_position"],
          float(np.max(np.abs(qpos_a[qa:qa + qpos_width]
                               - qpos_b[qa:qa + qpos_width]))))
      errors["scalar_joint_velocity"] = max(
          errors["scalar_joint_velocity"],
          float(np.max(np.abs(qvel_a[va:va + qvel_width]
                               - qvel_b[va:va + qvel_width]))))
  return errors


def run(steps=3000, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    native.reset_to_keyframe(0)

  cpu_grasp = mujoco.MjData(model)
  cpu_open = mujoco.MjData(model)
  for d in (cpu_grasp, cpu_open):
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)

  ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
  bin_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_geom")
  block_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball2_geom")
  finger_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                  for n in ("fingerL_geom", "fingerR_geom")]

  # Clearance (must stay separated): gripper hardware vs furniture, jaw
  # self-pairs (finger-finger, ledge-ledge across the staggered fork,
  # ledges vs opposite jaws), cross-station balls, and ball-ball.
  # Mechanical slide joints (palm shafts through finger tracks) and rigid
  # same-body attachments are excluded by design (documented, not contacts).
  # Functional grasp/rest contacts are allowlisted with penetration bounds
  # reported separately (compliant solref contact legitimately penetrates).
  clearance = _clearance_monitor(model)

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(model, record, "Muscle Gripper | Grasp and Carry",
                                  [0.1, 0, 0.30], 1.8, azimuth=110, elevation=-30)

  max_qpos_err = 0.0; max_qvel_err = 0.0
  max_act_err = 0.0
  pre_qpos_err = 0.0; pre_qvel_err = 0.0
  max_state_err = {
      "translation_m": 0.0,
      "orientation_geodesic_rad": 0.0,
      "linear_velocity_m_s": 0.0,
      "angular_velocity_rad_s": 0.0,
      "scalar_joint_position": 0.0,
      "scalar_joint_velocity": 0.0,
  }
  first_large_divergence = {
      "translation_step_1cm": None,
      "orientation_step_0p05rad": None,
      "linear_velocity_step_0p1m_s": None,
      "angular_velocity_step_0p1rad_s": None,
  }
  ball_bin_hits = 0
  latched_bin_hits = 0
  finger_contacts = 0
  first_finger_contact_step = None
  first_red_ball_finger_contact_step = None
  first_red_ball_bin_contact_step = None
  gripL_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripL")
  gripR_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripR")
  qadr_L = int(model.jnt_qposadr[gripL_id])
  qadr_R = int(model.jnt_qposadr[gripR_id])
  lo_L, hi_L = float(model.jnt_range[gripL_id][0]), float(model.jnt_range[gripL_id][1])
  lo_R, hi_R = float(model.jnt_range[gripR_id][0]), float(model.jnt_range[gripR_id][1])
  min_limit_margin = 1.0

  for step in range(steps):
    target, ctrl = _sched(step)
    t32 = target.astype(np.float32)
    if native is not None:
      native.set_mocap(t32.reshape(1, 3), np.array([[1, 0, 0, 0]], dtype=np.float32))
      native.step(1, ctrl=np.asarray(ctrl, dtype=np.float32).reshape(1, 4))
    cpu_grasp.mocap_pos[0] = target
    cpu_grasp.ctrl[:] = ctrl
    mujoco.mj_step(model, cpu_grasp)
    # Paired always-open counterfactual (never grasps).
    cpu_open.mocap_pos[0] = target
    cpu_open.ctrl[:] = OPEN
    mujoco.mj_step(model, cpu_open)
    grasped = step >= 350
    if native is not None:
      gq = native.state.qpos[0].cpu().numpy()
      gv = native.state.qvel[0].cpu().numpy()
      ga = native.state.act.cpu().numpy()
      step_qpos_err = float(np.max(np.abs(gq - cpu_grasp.qpos)))
      step_qvel_err = float(np.max(np.abs(gv - cpu_grasp.qvel)))
      max_qpos_err = max(max_qpos_err, step_qpos_err)
      max_qvel_err = max(max_qvel_err, step_qvel_err)
      state_err = _state_error_breakdown(model, gq, gv,
                                         cpu_grasp.qpos, cpu_grasp.qvel)
      for name, value in state_err.items():
        max_state_err[name] = max(max_state_err[name], value)
      for name, key, threshold in (
          ("translation_step_1cm", "translation_m", 0.01),
          ("orientation_step_0p05rad", "orientation_geodesic_rad", 0.05),
          ("linear_velocity_step_0p1m_s", "linear_velocity_m_s", 0.1),
          ("angular_velocity_step_0p1rad_s", "angular_velocity_rad_s", 0.1),
      ):
        if first_large_divergence[name] is None and state_err[key] > threshold:
          first_large_divergence[name] = step
      max_act_err = max(max_act_err, float(np.max(np.abs(ga - cpu_grasp.act))))
      if step < 100:
        pre_qpos_err = max(pre_qpos_err, step_qpos_err)
        pre_qvel_err = max(pre_qvel_err, step_qvel_err)
    for c in range(cpu_grasp.ncon):
      g1, g2 = int(cpu_grasp.contact[c].geom[0]), int(cpu_grasp.contact[c].geom[1])
      if (g1 == ball_geom and g2 == bin_geom) or (g2 == ball_geom and g1 == bin_geom):
        ball_bin_hits += 1
        if first_red_ball_bin_contact_step is None:
          first_red_ball_bin_contact_step = step
        break
    for c in range(cpu_open.ncon):
      g1, g2 = int(cpu_open.contact[c].geom[0]), int(cpu_open.contact[c].geom[1])
      if (g1 == ball_geom and g2 == bin_geom) or (g2 == ball_geom and g1 == bin_geom):
        latched_bin_hits += 1
        break
    for c in range(cpu_grasp.ncon):
      g1, g2 = int(cpu_grasp.contact[c].geom[0]), int(cpu_grasp.contact[c].geom[1])
      if (g1 in finger_geoms or g2 in finger_geoms):
        finger_contacts += 1
        if first_finger_contact_step is None:
          first_finger_contact_step = step
        other = g2 if g1 in finger_geoms else g1
        if other == ball_geom and first_red_ball_finger_contact_step is None:
          first_red_ball_finger_contact_step = step
        break
    clearance.sample(
        native.state.qpos[0].cpu().numpy() if native is not None else cpu_grasp.qpos,
        cpu_grasp.mocap_pos, cpu_grasp.mocap_quat)
    ql = float((native.state.qpos[0].cpu().numpy() if native is not None
                else cpu_grasp.qpos)[qadr_L])
    qr = float((native.state.qpos[0].cpu().numpy() if native is not None
                else cpu_grasp.qpos)[qadr_R])
    min_limit_margin = min(min_limit_margin, ql - lo_L, hi_L - ql,
                           qr - lo_R, hi_R - qr)
    if recorder and native is not None:
      actual = mujoco.MjData(model)
      actual.qpos[:] = native.state.qpos[0].cpu().numpy()
      actual.qvel[:] = native.state.qvel[0].cpu().numpy()
      actual.mocap_pos[0] = target
      actual.time = float(native.state.time[0].cpu().numpy())
      extra = f"{'GRASP' if grasped else 'OPEN'} | t={actual.time:.2f}s | ball x={float(actual.qpos[2]):+.2f}"
      recorder.frame(step, actual, cpu_grasp, extra=extra)

  if recorder:
    recorder.close()

  if mode == "metal" and native is not None:
    rel = native.state.qpos[0].cpu().numpy()
    released = (float(rel[2]), float(rel[4]))
    from mujoco_metal import MetalSimulation as MS
    lat_sim = MS(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    lat_sim.reset_to_keyframe(0)
    for step in range(steps):
      target, _ = _sched(step)
      lat_sim.set_mocap(target.astype(np.float32).reshape(1, 3),
                        np.array([[1, 0, 0, 0]], dtype=np.float32))
      lat_sim.step(1, ctrl=np.asarray(OPEN, dtype=np.float32).reshape(1, 4))
    lat = lat_sim.state.qpos[0].cpu().numpy()
    latched = (float(lat[2]), float(lat[4]))
    block_rel = (float(rel[9]), float(rel[11]))
    block_lat = (float(lat[9]), float(lat[11]))
  else:
    released = (float(cpu_grasp.qpos[2]), float(cpu_grasp.qpos[4]))
    latched = (float(cpu_open.qpos[2]), float(cpu_open.qpos[4]))
    block_rel = (float(cpu_grasp.qpos[9]), float(cpu_grasp.qpos[11]))
    block_lat = (float(cpu_open.qpos[9]), float(cpu_open.qpos[11]))

  result = {
      "max_qpos_err": max_qpos_err, "max_qvel_err": max_qvel_err,
      "max_act_err": max_act_err,
      "pre_qpos_err": pre_qpos_err, "pre_qvel_err": pre_qvel_err,
      "max_state_error_components": max_state_err,
      "first_large_divergence_steps": first_large_divergence,
      "ball_bin_hits": ball_bin_hits, "latched_bin_hits": latched_bin_hits,
      "finger_contacts": finger_contacts,
      "first_finger_contact_step": first_finger_contact_step,
      "first_red_ball_finger_contact_step": first_red_ball_finger_contact_step,
      "first_red_ball_bin_contact_step": first_red_ball_bin_contact_step,
      "released_ball": released, "latched_ball": latched,
      "block_rel": block_rel, "block_lat": block_lat,
      "steps": steps,
      "minimum_geometry_distance": clearance.minimum,
      "conservative_geometry_distance_lower_bound": clearance.reported,
      "geometry_distance_uncertainty_bound": clearance.uncertainty,
      "max_functional_penetration": clearance.max_pen,
      "min_joint_limit_margin": min_limit_margin,
  }
  if check:
    clearance.check()
    for pair, pen in clearance.max_pen.items():
      assert pen < 0.015, (pair, pen)
  if check and mode == "metal":
    assert pre_qpos_err < 2e-3, pre_qpos_err
    assert pre_qvel_err < 0.2, pre_qvel_err
    assert finger_contacts > 100, finger_contacts
    assert ball_bin_hits > 20, ball_bin_hits
    assert latched_bin_hits == 0, latched_bin_hits
    # Ball delivered into the bin footprint at bin-top height.
    assert 0.45 <= released[0] <= 0.75, released
    assert released[1] < 0.25, released
    # Always-open counterfactual never delivers (ball stays on the floor outside).
    assert not (0.45 <= latched[0] <= 0.75 and latched[1] < 0.25), latched
    # Second ball set down onto the pedestal (top z=0.2, rest center 0.255).
    assert abs(block_rel[0] - -0.5) < 0.10, block_rel
    assert abs(block_rel[1] - 0.255) < 0.03, block_rel
    assert abs(block_rel[0] - block_lat[0]) > 0.1
  return result


def run_viewer(steps=3000, seconds=0):
  """Interactive native viewer (requires a display; headless uses --headless).

  Steps the native simulation with the deterministic gantry/finger schedule
  and shows native state live, tracking the CPU oracle for the report.
  With seconds > 0 the viewer auto-closes after that wall time (used for
  explicit visual checks); 0 runs the full finite rollout (steps), then closes.
  """
  import time
  from mujoco import viewer as mj_viewer
  from mujoco_metal import MetalSimulation
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)
  native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :],
                           profile=PROFILE)
  native.reset_to_keyframe(0)
  cpu = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, cpu, 0)
  mujoco.mj_forward(model, cpu)
  actual = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, actual, 0)
  mujoco.mj_forward(model, actual)
  max_err = 0.0
  with mj_viewer.launch_passive(model, actual) as viewer:
    viewer.cam.lookat[:] = [0.1, 0, 0.30]
    viewer.cam.distance = 1.8
    viewer.cam.azimuth = 110
    viewer.cam.elevation = -30
    deadline = time.monotonic() + seconds
    for step in range(steps):
      if not viewer.is_running():
        break
      if seconds > 0 and time.monotonic() >= deadline:
        break
      target, ctrl = _sched(step)
      native.set_mocap(target.astype(np.float32).reshape(1, 3),
                       np.array([[1, 0, 0, 0]], dtype=np.float32))
      native.step(1, ctrl=np.asarray(ctrl, dtype=np.float32).reshape(1, 4))
      cpu.mocap_pos[0] = target
      cpu.ctrl[:] = ctrl
      mujoco.mj_step(model, cpu)
      gq = native.state.qpos[0].cpu().numpy()
      max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
      with viewer.lock():
        actual.qpos[:] = gq
        actual.qvel[:] = native.state.qvel[0].cpu().numpy()
        actual.mocap_pos[0] = target
        actual.time = float(native.state.time[0].cpu().numpy())
      viewer.sync()
  return {"max_qpos_err": max_err, "steps": steps}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=3000)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true", help="check native/CPU rollout")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  parser.add_argument("--json", help="write numerical report to JSON")
  parser.add_argument("--viewer-seconds", type=float, default=0,
                      help="interactive native viewer; auto-close after this wall time (0 runs the finite rollout, requires a display)")
  args = parser.parse_args()
  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")
  if not args.headless and args.record is None and args.mode == "metal":
    out = run_viewer(steps=args.steps, seconds=args.viewer_seconds)
    print(json.dumps(out, indent=2))
    if args.json:
      Path(args.json).write_text(json.dumps(out, indent=2))
    return
  out = run(steps=args.steps, mode=args.mode, check=args.check, record=args.record)
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
