# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Robotic marble music machine driven by the integrated native Metal Euler pipeline.

Combines motor actuation, fixed-joint tendons, passive spring/damping resonators,
fluid drag, pyramidal sphere-plane and sphere-sphere contacts, joint limits,
dry frictionloss, polynomial joint equality, and live current-state sensor queries—all
solved in one coupled constraint solve per Euler step.
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
  if (
      model.nq != 18
      or model.nv != 16
      or model.neq != 1
      or model.nu != 1
      or model.ntendon != 1
      or model.nsensor != 4
  ):
    raise RuntimeError(
        "Expected 16-DOF marble music machine with dual gates, dual chimes, two marbles, and sensors"
    )
  return model


def _rhythm_controller(at_time):
  """Rhythmic harmonic gate excitation to direct marbles and excite chimes."""
  # 1.25 Hz rhythmic switching pattern
  freq = 1.25
  omega = 2.0 * np.pi * freq
  motor = 1.8 * np.sin(omega * at_time)
  return np.array([motor], dtype=np.float32)


def run(steps=800, mode="metal", check=False, record=None):
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
        model, batch_size=1, qpos=qpos[None], qvel=qvel[None], profile=PROFILE
    )

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model,
        record,
        "Robotic Marble Music Machine | Integrated Euler",
        [0.1, 0.0, 0.6],
        2.4,
        azimuth=140,
        elevation=-26,
    )

  max_qpos_error = 0.0
  max_qvel_error = 0.0
  max_stage_sensor_error = 0.0
  max_trajectory_sensor_error = 0.0
  equality_max_residual = 0.0
  active_joint_limit_steps = 0
  near_limit_steps = 0
  unique_contact_pairs = set()
  cpu_detected_contact_pairs = set()
  dt = float(model.opt.timestep)

  chime1_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chime1_hinge")])
  chime2_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chime2_hinge")])
  gate1_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gate1_hinge")])
  gate2_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gate2_hinge")])

  max_chime1_vib = 0.0
  max_chime2_vib = 0.0

  for step in range(steps):
    t = step * dt
    ctrl = _rhythm_controller(t)

    # Reference CPU step
    reference.ctrl[:] = ctrl
    mujoco.mj_step(model, reference)

    if native is None:
      actual.ctrl[:] = ctrl
      mujoco.mj_step(model, actual)
      sensor_vals = actual.sensordata.copy()
    else:
      native.step(ctrl=ctrl[None])
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"Native integrated Euler status: {state.status.tolist()}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
      actual.time = float(state.time[0])
      sensor_vals = native.sensor_values().cpu().numpy()[0]

    max_qpos_error = max(max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos))))
    max_qvel_error = max(max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel))))

    # Equality constraint residual: gate2 + gate1 == 0
    eq_res = abs(float(actual.qpos[gate2_dof] + actual.qpos[gate1_dof]))
    equality_max_residual = max(equality_max_residual, eq_res)

    # Joint limit events:
    # 1. Near limit states (|q| >= 0.58)
    if (abs(float(actual.qpos[gate1_dof])) >= 0.58 or
        abs(float(actual.qpos[gate2_dof])) >= 0.58 or
        abs(float(actual.qpos[chime1_dof])) >= 0.58 or
        abs(float(actual.qpos[chime2_dof])) >= 0.58):
      near_limit_steps += 1

    # Stage sensor comparison (sensor-stage parity at current Metal state)
    d_check = mujoco.MjData(model)
    d_check.qpos[:] = actual.qpos
    d_check.qvel[:] = actual.qvel
    d_check.time = actual.time
    mujoco.mj_forward(model, d_check)
    max_stage_sensor_error = max(max_stage_sensor_error, float(np.max(np.abs(sensor_vals - d_check.sensordata))))

    # 2. Active limit constraint rows in MuJoCo forward
    if np.sum(d_check.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT) > 0:
      active_joint_limit_steps += 1

    # Trajectory sensor comparison (against independent CPU trajectory rollout)
    d_traj = mujoco.MjData(model)
    d_traj.qpos[:] = reference.qpos
    d_traj.qvel[:] = reference.qvel
    d_traj.time = reference.time
    mujoco.mj_fwdPosition(model, d_traj)
    mujoco.mj_fwdVelocity(model, d_traj)
    mujoco.mj_sensorPos(model, d_traj)
    mujoco.mj_sensorVel(model, d_traj)
    max_trajectory_sensor_error = max(max_trajectory_sensor_error, float(np.max(np.abs(sensor_vals - d_traj.sensordata))))

    # Record active contact pairs from CPU collision queries at native states
    for c_idx in range(d_check.ncon):
      g1 = model.geom(d_check.contact[c_idx].geom1).name
      g2 = model.geom(d_check.contact[c_idx].geom2).name
      pair_name = f"{min(g1, g2)} <-> {max(g1, g2)}"
      cpu_detected_contact_pairs.add(pair_name)
      unique_contact_pairs.add(pair_name)

    max_chime1_vib = max(max_chime1_vib, abs(float(actual.qpos[chime1_dof])))
    max_chime2_vib = max(max_chime2_vib, abs(float(actual.qpos[chime2_dof])))

    if recorder:
      g1 = actual.qpos[gate1_dof]
      c1 = actual.qpos[chime1_dof]
      c2 = actual.qpos[chime2_dof]
      recorder.frame(
          step,
          actual,
          reference,
          extra=f"gate {g1:+.2f} rad | chime1 {c1:+.3f} | chime2 {c2:+.3f} | t={actual.time:.2f}s",
      )

  if recorder:
    recorder.close()

  result = {
      "demo": "marble_music_machine",
      "mode": mode,
      "profile": PROFILE if native else "MuJoCo CPU",
      "steps": steps,
      "simulated_seconds": steps * dt,
      "features": [
          "stateless motor actuator driving primary selector gate",
          "polynomial joint equality synchronizing counter-rotating dual gates",
          "joint frictionloss and limits on gate and chime mechanisms",
          "fixed-joint tendon with spring, damping, and armature coupling gate to chime",
          "passive spring/damper resonant chime bars",
          "fluid drag and viscosity resisting falling marble paths",
          "pyramidal condim=3 sphere-plane and sphere-sphere collision contacts",
          "stateless current-state sensor queries read on GPU during rollout",
          "single unified Delassus PGS constraint solve coupling all contacts and limits",
      ],
      "max_chime1_vibration_rad": max_chime1_vib,
      "max_chime2_vibration_rad": max_chime2_vib,
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "max_stage_sensor_error": max_stage_sensor_error,
      "max_trajectory_sensor_error": max_trajectory_sensor_error,
      "max_sensor_error": max_stage_sensor_error,
      "equality_max_residual": equality_max_residual,
      "active_joint_limit_steps": active_joint_limit_steps,
      "near_limit_steps": near_limit_steps,
      "joint_limit_events": active_joint_limit_steps,
      "cpu_detected_contact_pairs": sorted(list(cpu_detected_contact_pairs)),
      "unique_contact_pairs": sorted(list(unique_contact_pairs)),
  }

  if check and mode == "metal":
    if max_qpos_error > 5e-5:
      raise AssertionError(f"qpos error {max_qpos_error} exceeded 5e-5")
    if max_qvel_error > 1e-3:
      raise AssertionError(f"qvel error {max_qvel_error} exceeded 1e-3")
    if max_stage_sensor_error > 1e-6:
      raise AssertionError(f"stage sensor error {max_stage_sensor_error} exceeded 1e-6")
    if max_trajectory_sensor_error > 5e-4:
      raise AssertionError(f"trajectory sensor error {max_trajectory_sensor_error} exceeded 5e-4")
    if steps >= 300:
      if max_chime1_vib < 0.05:
        raise AssertionError(f"chime 1 did not vibrate sufficiently: {max_chime1_vib}")
      if max_chime2_vib < 0.03:
        raise AssertionError(f"chime 2 did not vibrate sufficiently: {max_chime2_vib}")
      if len(unique_contact_pairs) < 4:
        raise AssertionError(f"expected 4 unique contact pairs, found {len(unique_contact_pairs)}")
      if active_joint_limit_steps < 100:
        raise AssertionError(f"expected active joint limit constraints, found {active_joint_limit_steps}")

  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument(
      "--check", action="store_true", help="check an independent native/CPU rollout"
  )
  parser.add_argument("--record", help="record actual native and CPU renders to a GIF")
  args = parser.parse_args(argv)

  if args.steps <= 0 or (args.check and not args.headless):
    parser.error("--check requires --headless")
  if args.check and args.mode != "metal":
    parser.error("--check compares native Metal with the independent CPU reference")

  print(json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2))


if __name__ == "__main__":
  main()
