# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Submerged jellyfish: filter-driven fins flap in modeled fluid.

Two stateful filter-actuated ellipsoid fins (per-geom ellipsoid fluid model
with Magnus/Kutta lift) flap in antiphase on a fixed mount under reduced
gravity, viscosity and a steady current; a passive sphere drifts with the
current via the inertia-box model. The modeled fluid is MuJoCo's analytic
approximation (drag + lift on equivalent ellipsoids), not a full fluid
simulator.
Rest equilibria are not unique in drift, so checks gate flap amplitude,
drift, impact-transient and settle envelopes plus early parity.

Headless verification (--check) and side-by-side GIF recording (--record).
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"
FREQ = 2.0 * math.pi * 1.5


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.nq != 9 or model.nv != 8 or model.nu != 2 or model.na != 2:
    raise RuntimeError(
        f"Expected 2 hinges + free sphere + 2 filter actuators (9q/8v/2u/2a), got "
        f"{model.nq}q/{model.nv}v/{model.nu}u/{model.na}a")
  if not np.any(np.asarray(model.geom_fluid)[:, 0] > 0):
    raise RuntimeError("Expected interacting ellipsoid fluid geoms")
  return model


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
        model, batch_size=1, qpos=qpos[None], qvel=qvel[None],
        profile=PROFILE)

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(
        model, record, "Tide Jellyfish | Filter Fins in Fluid",
        [0.1, 0.0, 0.55], 1.8, azimuth=135, elevation=-20)

  max_qpos_error = 0.0
  max_qvel_error = 0.0
  pre_qpos_error = 0.0
  end_qpos_error = 0.0
  flap_amp = 0.0
  # qpos: flapL [0], flapR [1], floater xyz [2:5] + quat [5:9].
  pos_idx = list(range(9))
  for step in range(steps):
    t = 0.002 * step
    ctrl = np.array([math.sin(FREQ * t), -math.sin(FREQ * t)], dtype=float)
    if native is not None:
      native.step(1, ctrl=ctrl.reshape(1, 2).astype(np.float32))
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
    flap_amp = max(flap_amp, abs(float(actual.qpos[0])), abs(float(actual.qpos[1])))
    max_qpos_error = max(
        max_qpos_error, float(np.max(np.abs(
            actual.qpos[pos_idx] - reference.qpos[pos_idx]))))
    max_qvel_error = max(
        max_qvel_error, float(np.max(np.abs(
            actual.qvel - reference.qvel))))
    if step < 30:
      pre_qpos_error = max(
          pre_qpos_error, float(np.max(np.abs(
              actual.qpos[pos_idx] - reference.qpos[pos_idx]))))
    if step == steps - 1:
      end_qpos_error = float(np.max(np.abs(
          actual.qpos[pos_idx] - reference.qpos[pos_idx])))
    if recorder is not None:
      extra = (f"t={float(actual.time):.2f}s flap={float(actual.qpos[0]):+.2f}")
      recorder.frame(step, actual, reference, extra=extra)

  q = actual.qpos
  out = {
      "steps": steps,
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "pre_qpos_error": pre_qpos_error,
      "end_qpos_error": end_qpos_error,
      "flap_amplitude": flap_amp,
      "floater_end": [float(q[2]), float(q[3]), float(q[4])],
      "floater_drift": float(np.linalg.norm(q[2:5] - qpos[2:5])),
  }
  if recorder is not None:
    recorder.close()
  if check:
    # Fins flap with real amplitude; floater drifts with the current.
    assert out["flap_amplitude"] > 0.15, out
    assert out["floater_drift"] > 0.05, out
    assert pre_qpos_error < 5e-4, pre_qpos_error
    assert end_qpos_error < 3e-2, end_qpos_error
    assert max_qpos_error < 3e-2, max_qpos_error
    assert max_qvel_error < 5e-2, max_qvel_error
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
  out = run(steps=args.steps, mode=args.mode, check=args.check,
            record=args.record)
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
