# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A tabletop arcade for sliding, torsional and rolling friction."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

_MODEL_PATH = Path(__file__).with_suffix(".xml")
_ROLL_STEPS = 400
_PRESS_STEPS = 60
_RELEASE_STEP = 180


def _load_model(rolling=".18", torsional=".48"):
  xml = _MODEL_PATH.read_text()
  # These exact strings are limited to the named demo geoms. Counterfactuals
  # change one friction axis at a time while preserving every other parameter.
  if rolling != ".18":
    xml = xml.replace('name="roll_geom" type="sphere" size=".115" mass=".35"\n            condim="6" friction=".9 .12 .18"',
                      f'name="roll_geom" type="sphere" size=".115" mass=".35"\n            condim="6" friction=".9 .12 {rolling}"')
  if torsional != ".48":
    xml = xml.replace('name="spin_geom" type="sphere" size=".105" mass=".5"\n            condim="4" friction=".9 .48 .04"',
                      f'name="spin_geom" type="sphere" size=".105" mass=".5"\n            condim="4" friction=".9 {torsional} .04"')
  model = mujoco.MjModel.from_xml_string(xml)
  if model.nv != 14 or model.ngeom != 6 or model.npair or model.nexclude:
    raise RuntimeError("Spin-and-Grip scene bounds changed unexpectedly")
  if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
    raise RuntimeError("Spin-and-Grip requires the elliptic cone")
  return model


def _indices(model):
  return {
      "roll_q": int(model.jnt_qposadr[model.joint("roll_free").id]),
      "roll_v": int(model.jnt_dofadr[model.joint("roll_free").id]),
      "spin_v": int(model.jnt_dofadr[model.joint("spin_free").id]),
      "grip_q": int(model.jnt_qposadr[model.joint("grip_slide").id]),
      "grip_v": int(model.jnt_dofadr[model.joint("grip_slide").id]),
      "pad_q": int(model.jnt_qposadr[model.joint("pad_slide").id]),
      "pad_v": int(model.jnt_dofadr[model.joint("pad_slide").id]),
      "pad_geom": int(model.geom("pad_geom").id),
      "grip_geom": int(model.geom("grip_geom").id),
      "roll_body": int(model.body("roll_runner").id),
      "spin_body": int(model.body("spin_top").id),
      "grip_body": int(model.body("grip_object").id),
  }


def _initial_state(model):
  idx = _indices(model)
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  qvel[idx["roll_v"] : idx["roll_v"] + 6] = [1.8, 0, 0, 0, 1.8, 0]
  qvel[idx["spin_v"] : idx["spin_v"] + 6] = [0, 0, 0, 0, 0, 8]
  qvel[idx["grip_v"]] = 1.2
  return qpos, qvel


def _controls(step, model, qfrc):
  idx = _indices(model)
  ctrl = np.zeros(model.nu, dtype=np.float32)
  # Close the vertical press, hold it through step 180, then retract it.
  ctrl[0] = 25.0 if step < _RELEASE_STEP else -25.0
  qfrc.fill(0.0)
  # A small physical push demonstrates that the object slides again after the
  # press releases. It is applied as generalized force, not scripted motion.
  if step >= _RELEASE_STEP:
    qfrc[idx["grip_v"]] = 0.5
  return ctrl


def _run_cpu_counterfactual(rolling=".18", torsional=".48", steps=_ROLL_STEPS):
  model = _load_model(rolling=rolling, torsional=torsional)
  qpos, qvel = _initial_state(model)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  idx = _indices(model)
  first_pad_contact = None
  contact_steps = 0
  hold_start_x = hold_end_x = None
  for step in range(steps):
    qfrc = np.zeros(model.nv, dtype=np.float64)
    data.ctrl[:] = _controls(step, model, qfrc)
    data.qfrc_applied[:] = qfrc
    mujoco.mj_step(model, data)
    pad_touching = any(
        {data.contact[c].geom1, data.contact[c].geom2}
        == {idx["pad_geom"], idx["grip_geom"]}
        for c in range(data.ncon)
    )
    if pad_touching:
      contact_steps += 1
      if first_pad_contact is None:
        first_pad_contact = step
    if step == _PRESS_STEPS - 1:
      hold_start_x = float(data.qpos[idx["grip_q"]])
    if step == _RELEASE_STEP - 1:
      hold_end_x = float(data.qpos[idx["grip_q"]])
  return {
      "rolling_travel_m": float(data.qpos[idx["roll_q"]] - qpos[idx["roll_q"]]),
      "spin_rate_rad_s": float(np.linalg.norm(data.qvel[idx["spin_v"] + 3 : idx["spin_v"] + 6])),
      "pad_first_contact_step": first_pad_contact,
      "pad_contact_steps": contact_steps,
      "grip_hold_drift_m": float(abs(hold_end_x - hold_start_x)),
      "grip_release_travel_m": float(data.qpos[idx["grip_q"]] - hold_end_x),
  }


def run(steps=_ROLL_STEPS, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos, qvel = _initial_state(model)
  idx = _indices(model)
  actual = mujoco.MjData(model)
  actual.qpos[:], actual.qvel[:] = qpos, qvel
  reference = mujoco.MjData(model)
  reference.qpos[:], reference.qvel[:] = qpos, qvel
  simulation = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    simulation = MetalSimulation(
        model,
        qpos=qpos[None, :],
        qvel=qvel[None, :],
        profile="integrated_euler_v1",
    )

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model,
        record,
        "SPIN / GRIP / RELEASE",
        [0.0, 0.0, 0.08],
        3.2,
        azimuth=90,
        elevation=-35,
    )

  max_qpos_error = max_qvel_error = 0.0
  first_pad_contact = None
  pad_contact_steps = 0
  hold_start_x = hold_end_x = None
  max_contacts = 0
  for step in range(steps):
    qfrc = np.zeros(model.nv, dtype=np.float32)
    ctrl = _controls(step, model, qfrc)
    reference.ctrl[:] = ctrl
    reference.qfrc_applied[:] = qfrc
    mujoco.mj_step(model, reference)
    if simulation is None:
      actual.ctrl[:] = ctrl
      actual.qfrc_applied[:] = qfrc
      mujoco.mj_step(model, actual)
    else:
      simulation.step(1, ctrl=ctrl[None, :], qfrc_applied=qfrc[None, :])
      state = simulation.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(
            f"native status at step {step}: {state.status.tolist()} "
            f"solver={simulation._coupled_constraints._workspace['out_diagnostics'].tolist()}"
        )
      actual.qpos[:], actual.qvel[:] = state.qpos[0], state.qvel[0]

    max_qpos_error = max(max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos))))
    max_qvel_error = max(max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel))))
    max_contacts = max(max_contacts, reference.ncon)
    pad_touching = any(
        {reference.contact[c].geom1, reference.contact[c].geom2}
        == {idx["pad_geom"], idx["grip_geom"]}
        for c in range(reference.ncon)
    )
    if pad_touching:
      pad_contact_steps += 1
      if first_pad_contact is None:
        first_pad_contact = step
    if step == _PRESS_STEPS - 1:
      hold_start_x = float(actual.qpos[idx["grip_q"]])
    if step == _RELEASE_STEP - 1:
      hold_end_x = float(actual.qpos[idx["grip_q"]])
    if recorder:
      phase = "PRESS" if step < _PRESS_STEPS else ("HOLD" if step < _RELEASE_STEP else "RELEASE")
      recorder.frame(step, actual, reference, extra=f"{phase} | elliptic condim 4/6")
  if recorder:
    recorder.close()

  actual_metrics = {
      "rolling_travel_m": float(actual.qpos[idx["roll_q"]] - qpos[idx["roll_q"]]),
      "spin_rate_rad_s": float(np.linalg.norm(actual.qvel[idx["spin_v"] + 3 : idx["spin_v"] + 6])),
      "pad_first_contact_step": first_pad_contact,
      "pad_contact_steps": pad_contact_steps,
      "grip_hold_drift_m": float(abs(hold_end_x - hold_start_x)) if hold_end_x is not None else None,
      "grip_release_travel_m": float(actual.qpos[idx["grip_q"]] - hold_end_x) if hold_end_x is not None else None,
  }
  result = {
      "demo": "spin_and_grip",
      "mode": mode,
      "profile": "integrated_euler_v1" if simulation else "MuJoCo CPU",
      "cone": "elliptic",
      "steps": steps,
      "contact_max": max_contacts,
      "max_qpos_error_mixed_coordinates": max_qpos_error,
      "max_qvel_error_mixed_units": max_qvel_error,
      "actual": actual_metrics,
      "record": str(record) if record else None,
  }
  if check and mode == "metal":
    low_roll = _run_cpu_counterfactual(rolling=".001", steps=steps)
    low_spin = _run_cpu_counterfactual(torsional=".001", steps=steps)
    result["cpu_counterfactual_low_rolling_friction"] = low_roll
    result["cpu_counterfactual_low_torsional_friction"] = low_spin
    if first_pad_contact is None or pad_contact_steps < 30:
      raise AssertionError("actuated pad never formed a sustained object contact")
    if actual_metrics["grip_hold_drift_m"] is None or actual_metrics["grip_hold_drift_m"] > 0.02:
      raise AssertionError("press failed to hold the sliding object during the hold phase")
    if actual_metrics["grip_release_travel_m"] is None or actual_metrics["grip_release_travel_m"] < 0.02:
      raise AssertionError("released object did not resume sliding under applied force")
    if not abs(actual_metrics["rolling_travel_m"]) < 0.5 * abs(low_roll["rolling_travel_m"]):
      raise AssertionError("rolling resistance did not reduce travel versus the low-friction case")
    if not actual_metrics["spin_rate_rad_s"] < 0.5 * low_spin["spin_rate_rad_s"]:
      raise AssertionError("torsional friction did not damp spin versus the low-friction case")
    if max_qpos_error > 0.03 or max_qvel_error > 0.5:
      raise AssertionError(json.dumps(result, indent=2))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=_ROLL_STEPS)
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", help="record a native/CPU comparison GIF")
  args = parser.parse_args(argv)
  if args.steps < _RELEASE_STEP + 20:
    parser.error(f"Use at least {_RELEASE_STEP + 20} steps so release is observable")
  if args.check and (not args.headless or args.mode != "metal"):
    parser.error("--check requires --headless and --mode metal")
  print(json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2))


if __name__ == "__main__":
  main()
