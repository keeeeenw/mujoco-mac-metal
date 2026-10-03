# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Instrumented pinball inspection table: pads, accelerometers, range array.

Two free balls drop onto force pads while motor-driven flippers sweep; touch
pads report contact forces, body sites report acceleration/force, and a
downward range array scans ball heights. Overlay displays live sensor outputs.

Headless verification (--check) and side-by-side GIF recording (--record).
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.nsensor == 0:
    raise RuntimeError("Expected instrumented sensors")
  return model


def _sensor_summary(model, sensordata):
  adr = np.asarray(model.sensor_adr)
  out = {}
  # Order: touchA(0), touchB(1), accelA(2), accelB(3), range0(4), range1(5),
  # range2(6: dist+normal), jointL, jointR, actFrc, limFrc.
  out["touchA"] = float(sensordata[adr[0]])
  out["touchB"] = float(sensordata[adr[1]])
  out["accelA"] = float(np.linalg.norm(sensordata[adr[2]:adr[2] + 3]))
  out["forceA"] = float(np.linalg.norm(sensordata[adr[4]:adr[4] + 3]))
  out["range0"] = float(sensordata[adr[6]])
  out["range1"] = float(sensordata[adr[7]])
  return out, None


def run(steps=600, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  # Lively drop: balls fall onto pads, flippers sweep.
  # qpos layout: flipL[0], flipR[1], ballA xyz[2:5]+quat[5:9], ballB xyz[9:12]+quat[12:16].
  qvel[:] = 0

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
        model, record, "Pinball Inspection Table | Pads + IMU + Range Array",
        [0.0, -0.4, 1.0], 2.2, azimuth=90, elevation=-30)

  max_qpos_error = 0.0
  max_qvel_error = 0.0
  max_sens_error = 0.0
  touch_peak = 0.0
  accel_peak = 0.0
  range_hits = 0
  for step in range(steps):
    t = 0.002 * step
    ctrl = np.array([0.8 * math.sin(2 * math.pi * 1.2 * t),
                     0.8 * math.sin(2 * math.pi * 1.2 * t + 1.0)], dtype=float)
    if native is not None:
      native.step(1, ctrl=ctrl.reshape(1, 2).astype(np.float32))
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"Native status {state.status.tolist()} at step {step}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
      actual.time = float(state.time[0])
      # Stored step-stage sample matches mj_step timing; query re-evaluates
      # at the post-step state (see REPORT), so parity uses stored.
      sens_native = native.step_sensordata()[0]
      query_native = native.sensor_values().cpu().numpy()[0]
    reference.ctrl[:] = ctrl
    mujoco.mj_step(model, reference)
    if native is None:
      actual.qpos[:] = reference.qpos
      actual.qvel[:] = reference.qvel
      actual.time = reference.time
      sens_native = np.asarray(reference.sensordata)
      query_native = sens_native
    sens_ref = np.asarray(reference.sensordata)
    max_qpos_error = max(max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos))))
    max_qvel_error = max(max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel))))
    max_sens_error = max(max_sens_error, float(np.max(np.abs(sens_native - sens_ref))))
    summ, _ = _sensor_summary(model, query_native)
    touch_peak = max(touch_peak, summ["touchA"], summ["touchB"])
    accel_peak = max(accel_peak, summ["accelA"])
    if summ["range0"] >= 0:
      range_hits += 1
    if recorder is not None:
      extra = (f"t={float(actual.time):.2f}s touchA={summ['touchA']:.1f} "
               f"accA={summ['accelA']:.1f} r0={summ['range0']:.2f}")
      recorder.frame(step, actual, reference, extra=extra)

  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "max_sens_error": max_sens_error,
      "touch_peak": touch_peak,
      "accel_peak": accel_peak,
      "range_hit_fraction": range_hits / max(steps, 1),
  }
  if recorder is not None:
    recorder.close()
  if check:
    assert touch_peak > 0.5, out
    assert accel_peak > 1.0, out
    assert out["range_hit_fraction"] > 0.5, out
    assert max_sens_error < 5e-3, out
    assert max_qpos_error < 3e-2, out
    assert max_qvel_error < 5e-2, out
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true")
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
