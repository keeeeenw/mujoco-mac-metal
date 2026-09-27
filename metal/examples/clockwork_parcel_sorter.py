# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Clockwork parcel sorter demo driven by the integrated native Metal Euler pipeline.

Features:
- Solid box ramp with multi-contact face/edge manifold and Coulomb friction
- Real box parcels, capsule parcels, and sphere parcels mixed in the feed
- Actuated swinging diverter gate driven by rhythmic clockwork motor
- Follower diverter flap synchronized via polynomial joint equality constraint
- Passive spring/damper lever coupled via fixed tendon
- Contact manifolds coupling box-box, capsule-box, capsule-capsule, sphere-box,
  sphere-capsule, sphere-plane, and parcel-to-parcel interactions
- Dynamic physical sorting routing parcels to distinct channels
- Headless verification (--check) and side-by-side GIF recording (--record)
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
      model.nq != 24
      or model.nv != 21
      or model.neq != 1
      or model.nu != 1
      or model.ntendon != 1
      or model.nsensor != 4
  ):
    raise RuntimeError(
        "Expected 21-DOF parcel sorter with dual diverters, 3 mixed parcels, tendon, and sensors"
    )
  return model


def _diverter_controller(at_time):
  """Clockwork diverter switching controller: holds left then flips right."""
  motor = 1.2 if at_time < 0.6 else -1.2
  return np.array([motor], dtype=np.float32)


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
        model, batch_size=1, qpos=qpos[None], qvel=qvel[None], profile=PROFILE
    )

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model,
        record,
        "Clockwork Parcel Sorter | Integrated Euler",
        [0.15, 0.0, 0.35],
        1.3,
        azimuth=135,
        elevation=-25,
    )

  box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parcel_box")
  cap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parcel_cap")
  sph_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parcel_sph")

  div1_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "diverter1_hinge")])
  div2_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "diverter2_hinge")])

  max_early_qpos_error = 0.0
  max_early_qvel_error = 0.0
  max_stage_sensor_error = 0.0
  equality_max_residual = 0.0
  cpu_equality_max_residual = 0.0
  active_joint_limit_steps = 0
  active_contact_steps = 0
  peak_contacts = 0
  unique_contact_pairs = set()
  cpu_detected_contact_pairs = set()
  dt = float(model.opt.timestep)

  for step in range(steps):
    t = step * dt
    ctrl = _diverter_controller(t)

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
        raise RuntimeError(f"Native integrated Euler status: {state.status.tolist()} at step {step}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
      actual.time = float(state.time[0])
      sensor_vals = native.sensor_values().cpu().numpy()[0]

    # Pre-impact trajectory parity check (during ramp descent, steps 0..200)
    if step <= 200:
      max_early_qpos_error = max(max_early_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos))))
      max_early_qvel_error = max(max_early_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel))))

    # Equality constraint residual: diverter2 - diverter1 == 0
    eq_res = abs(float(actual.qpos[div2_dof] - actual.qpos[div1_dof]))
    equality_max_residual = max(equality_max_residual, eq_res)
    cpu_eq_res = abs(float(reference.qpos[div2_dof] - reference.qpos[div1_dof]))
    cpu_equality_max_residual = max(cpu_equality_max_residual, cpu_eq_res)

    # Evaluate MuJoCo forward at current actual state for sensor & contact verification
    d_check = mujoco.MjData(model)
    d_check.qpos[:] = actual.qpos
    d_check.qvel[:] = actual.qvel
    d_check.time = actual.time
    mujoco.mj_forward(model, d_check)
    max_stage_sensor_error = max(max_stage_sensor_error, float(np.max(np.abs(sensor_vals - d_check.sensordata))))

    if d_check.ncon > 0:
      active_contact_steps += 1
      peak_contacts = max(peak_contacts, d_check.ncon)
      for c_idx in range(d_check.ncon):
        g1 = model.geom(d_check.contact[c_idx].geom1).name
        g2 = model.geom(d_check.contact[c_idx].geom2).name
        pair_name = f"{min(g1, g2)} <-> {max(g1, g2)}"
        unique_contact_pairs.add(pair_name)
        cpu_detected_contact_pairs.add(pair_name)

    if np.sum(d_check.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT) > 0:
      active_joint_limit_steps += 1

    if recorder:
      b_y = actual.xpos[box_bid][1]
      c_y = actual.xpos[cap_bid][1]
      s_y = actual.xpos[sph_bid][1]
      recorder.frame(
          step,
          actual,
          reference,
          extra=f"Box Y={b_y:+.2f} | Cap Y={c_y:+.2f} | Sph Y={s_y:+.2f} | con={d_check.ncon} | t={actual.time:.2f}s",
      )

  if recorder:
    recorder.close()

  mujoco.mj_forward(model, actual)
  box_final_y = float(actual.xpos[box_bid][1])
  cap_final_y = float(actual.xpos[cap_bid][1])
  sph_final_y = float(actual.xpos[sph_bid][1])

  box_routed_left = bool(box_final_y < -0.05)
  cap_routed_right = bool(cap_final_y > 0.05)
  sph_routed_center = bool(abs(sph_final_y) < 0.05)

  result = {
      "demo": "clockwork_parcel_sorter",
      "mode": mode,
      "profile": PROFILE if native else "MuJoCo CPU",
      "steps": steps,
      "simulated_seconds": steps * dt,
      "features": [
          "solid box feed ramp with multi-contact face/edge manifold and friction",
          "real box parcels, capsule parcels, and sphere parcels mixed in feed",
          "actuated swinging diverter gate driven by rhythmic clockwork motor",
          "follower diverter gate synchronized via polynomial joint equality",
          "passive spring/damper lever coupled via fixed tendon",
          "primitive collision manifold generation across planes, spheres, capsules, boxes",
          "stateless current-state sensor queries read on GPU during rollout",
          "single unified Delassus PGS constraint solve coupling all contacts and limits",
      ],
      "box_final_y": box_final_y,
      "cap_final_y": cap_final_y,
      "sph_final_y": sph_final_y,
      "box_routed_left": box_routed_left,
      "cap_routed_right": cap_routed_right,
      "sph_routed_center": sph_routed_center,
      "max_early_qpos_error": max_early_qpos_error,
      "max_early_qvel_error": max_early_qvel_error,
      "max_stage_sensor_error": max_stage_sensor_error,
      "equality_max_residual": equality_max_residual,
      "active_contact_steps": active_contact_steps,
      "peak_contacts": peak_contacts,
      "active_joint_limit_steps": active_joint_limit_steps,
      "unique_contact_pairs": sorted(list(unique_contact_pairs)),
  }

  if check and mode == "metal":
    if max_early_qpos_error > 5e-5:
      raise AssertionError(f"early qpos error {max_early_qpos_error} exceeded 5e-5")
    if max_early_qvel_error > 1e-3:
      raise AssertionError(f"early qvel error {max_early_qvel_error} exceeded 1e-3")
    if max_stage_sensor_error > 1e-6:
      raise AssertionError(f"stage sensor error {max_stage_sensor_error} exceeded 1e-6")
    if abs(equality_max_residual - cpu_equality_max_residual) > 1e-4:
      raise AssertionError(
          f"equality residual parity failed: Metal={equality_max_residual}, CPU={cpu_equality_max_residual}"
      )
    if equality_max_residual > 0.03:
      raise AssertionError(f"equality residual {equality_max_residual} exceeded 0.03")
    if steps >= 400:
      if not box_routed_left:
        raise AssertionError(f"box parcel was not routed to left channel: Y={box_final_y}")
      if not cap_routed_right:
        raise AssertionError(f"capsule parcel was not routed to right channel: Y={cap_final_y}")
      if not sph_routed_center:
        raise AssertionError(f"sphere parcel was not routed down center: Y={sph_final_y}")
      if active_contact_steps < 200:
        raise AssertionError(f"expected >= 200 contact steps, found {active_contact_steps}")
      if peak_contacts < 3:
        raise AssertionError(f"expected peak contacts >= 3, found {peak_contacts}")
      if len(unique_contact_pairs) < 4:
        raise AssertionError(f"expected >= 4 unique contact pairs, found {len(unique_contact_pairs)}")

  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=600)
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
