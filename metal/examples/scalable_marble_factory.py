# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A scalable multi-lane marble feed and contact-capacity demonstration.

Fifteen independent marble lanes share one compiled model. Each lane has a
moving free sphere in a three-sided chute, so every sphere makes floor and two
rail contacts while rolling down the lane. Per-lane collision groups reflect
the physical dividers and keep the broad phase proportional to real contacts.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np


LANE_COUNT = 15
MARBLE_RADIUS = 0.05
LANE_PITCH = 0.16
ROLL_SPEED = 0.12


def build_model():
  """Compile the fixed lane array with contact margins and no overlaps."""
  surfaces = []
  marbles = []
  for lane in range(LANE_COUNT):
    bit = 1 << lane
    x = lane * LANE_PITCH
    surfaces.extend((
        f'<geom name="floor{lane}" type="plane" size=".1 .1 .1" '
        f'pos="{x} 0 0" contype="{bit}" conaffinity="0" '
        'condim="6" margin=".001" friction=".8 .01 .001"/>',
        f'<geom name="left_rail{lane}" type="plane" size=".1 .1 .1" '
        f'pos="{x - MARBLE_RADIUS} 0 {MARBLE_RADIUS}" '
        'quat=".7071068 0 .7071068 0" '
        f'contype="{bit}" conaffinity="0" condim="6" margin=".001" '
        'friction=".8 .01 .001"/>',
        f'<geom name="right_rail{lane}" type="plane" size=".1 .1 .1" '
        f'pos="{x + MARBLE_RADIUS} 0 {MARBLE_RADIUS}" '
        'quat=".7071068 0 -.7071068 0" '
        f'contype="{bit}" conaffinity="0" condim="6" margin=".001" '
        'friction=".8 .01 .001"/>',
    ))
    marbles.append(
        f'<body name="marble{lane:02d}" pos="{x} 0 {MARBLE_RADIUS}">'
        '<freejoint/>'
        f'<geom name="marble_geom{lane:02d}" type="sphere" '
        f'size="{MARBLE_RADIUS}" mass=".1" contype="{bit}" '
        f'conaffinity="{bit}" condim="6" margin=".001" '
        'friction=".8 .01 .001" rgba=".92 .38 .09 1"/>'
        '</body>')
  xml = f'''<mujoco model="scalable_marble_factory">
    <compiler angle="radian"/>
    <option timestep=".002" integrator="Euler" iterations="100"
            tolerance="1e-8" gravity="0 0 -9.81" cone="elliptic"/>
    <visual><global offwidth="1280" offheight="720"/><map znear=".01" zfar="20"/></visual>
    <worldbody>
      <light directional="true" pos="0 -2 5" dir="0 .3 -1"
             diffuse=".9 .9 .9" specular=".3 .3 .3"/>
      {''.join(surfaces)}
      {''.join(marbles)}
    </worldbody>
  </mujoco>'''
  return mujoco.MjModel.from_xml_string(xml)


def _initial_velocity(model):
  qvel = np.zeros(model.nv, dtype=np.float64)
  for jid in range(model.njnt):
    qvel[int(model.jnt_dofadr[jid]) + 1] = ROLL_SPEED
  return qvel


def _capacity_limits(model):
  """Set explicit lane-derived caps and a visible one-GiB allocation budget."""
  from mujoco_metal.capacity import CapacityLimits

  return CapacityLimits(
      max_nv=int(model.nv),
      max_pairs=3 * LANE_COUNT,
      max_slots=3 * LANE_COUNT,
      max_rows=390,
      max_batch=1,
      memory_budget_bytes=1 << 30)


def _quat_error(model, qpos, ref_qpos):
  """Return maximum sign-invariant angle error for the freejoint rotations."""
  maximum = 0.0
  for jid in range(model.njnt):
    adr = int(model.jnt_qposadr[jid])
    qa = np.asarray(qpos[adr + 3:adr + 7], dtype=np.float64)
    qb = np.asarray(ref_qpos[adr + 3:adr + 7], dtype=np.float64)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    maximum = max(maximum, 2.0 * float(np.arccos(dot)))
  return maximum


def run(steps=250, mode="cpu", check=False, record=None,
        restore_check=False):
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.capacity import estimate_capacity

  if int(steps) <= 0:
    raise ValueError("steps must be positive")
  model = build_model()
  limits = _capacity_limits(model)
  descriptor = lower_coupled_constraints(model, limits=limits)
  structural = {
      "nv": int(model.nv),
      "candidate_pairs": int(descriptor.npairs),
      "allocated_slots": int(descriptor.ncontacts_max),
      "allocated_rows": int(descriptor.nr),
  }
  assert structural == {
      "nv": 90,
      "candidate_pairs": 45,
      "allocated_slots": 45,
      "allocated_rows": 390,
  }, structural
  estimate = estimate_capacity(
      model, 1, descriptor.npairs, descriptor.ncontacts_max, descriptor.nr,
      nr_joint=descriptor.nr_joint, neq=model.neq,
      mass_storage="block_sparse")
  if estimate.memory_bytes > limits.memory_budget_bytes:
    raise MemoryError(
        f"marble workspace estimate {estimate.memory_bytes} exceeds "
        f"the configured {limits.memory_budget_bytes}-byte budget")

  qpos0 = np.asarray(model.qpos0, dtype=np.float64).copy()
  qvel0 = _initial_velocity(model)
  reference = mujoco.MjData(model)
  actual = mujoco.MjData(model)
  reference.qpos[:] = qpos0
  reference.qvel[:] = qvel0
  actual.qpos[:] = qpos0
  actual.qvel[:] = qvel0

  native = None
  if mode == "metal":
    from mujoco_metal.simulation import MetalSimulation
    native = MetalSimulation(
        model, batch_size=1, qpos=qpos0[None].astype(np.float32),
        qvel=qvel0[None].astype(np.float32),
        profile="integrated_scalable_v1", limits=limits)
  elif mode != "cpu":
    raise ValueError("mode must be 'cpu' or 'metal'")

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(
        model, record,
        "Scalable Marble Feed | 90 DOF, 45 Pair Candidates",
        [1.1, 0., .25], 3.3, azimuth=135, elevation=-24)

  max_qpos_error = 0.0
  max_qvel_error = 0.0
  max_qacc_error = 0.0
  max_orientation_error = 0.0
  peak_active_pairs = 0
  peak_contacts = 0
  peak_active_rows = 0
  peak_native_broadphase_pairs = 0
  peak_native_contact_slots = 0
  peak_native_active_rows = 0
  for step in range(int(steps)):
    mujoco.mj_step(model, reference)
    native_qacc = None
    if native is None:
      mujoco.mj_step(model, actual)
    else:
      native.step()
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"native status {state.status.tolist()} at step {step}")
      actual.qpos[:] = state.qpos[0]
      actual.qvel[:] = state.qvel[0]
      actual.qacc[:] = state.qacc[0]
      native_qacc = state.qacc[0].copy()
      actual.time = float(state.time[0])
      mujoco.mj_forward(model, actual)
      if native._coupled_constraints is None:
        raise RuntimeError("scalable marble scene unexpectedly bypassed contact rows")
      native_pairs, native_slots, native_rows = (
          native._coupled_constraints.active_counts()[0])
      peak_native_broadphase_pairs = max(
          peak_native_broadphase_pairs, int(native_pairs))
      peak_native_contact_slots = max(
          peak_native_contact_slots, int(native_slots))
      peak_native_active_rows = max(peak_native_active_rows, int(native_rows))

    max_qpos_error = max(
        max_qpos_error,
        float(np.max(np.abs(actual.qpos - reference.qpos))))
    max_qvel_error = max(
        max_qvel_error,
        float(np.max(np.abs(actual.qvel - reference.qvel))))
    max_qacc_error = max(
        max_qacc_error,
        float(np.max(np.abs((actual.qacc if native_qacc is None else native_qacc)
                            - reference.qacc))))
    max_orientation_error = max(
        max_orientation_error, _quat_error(model, actual.qpos, reference.qpos))
    active_pairs = {tuple(sorted(map(int, reference.contact[i].geom)))
                    for i in range(reference.ncon)}
    peak_active_pairs = max(peak_active_pairs, len(active_pairs))
    peak_contacts = max(peak_contacts, int(reference.ncon))
    peak_active_rows = max(peak_active_rows, int(reference.nefc))
    if recorder is not None:
      recorder.frame(step, actual, reference,
                     extra=f"t={float(reference.time):.2f}s")

  if recorder is not None:
    recorder.close()
  checkpoint_replay_bitwise = None
  if restore_check:
    if native is None:
      raise ValueError("restore_check requires mode='metal'")
    checkpoint = native.snapshot()
    future = []
    for _ in range(2):
      native.step()
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"native replay status {state.status.tolist()}")
      future.append((state.qpos.copy(), state.qvel.copy(), state.qacc.copy(),
                     state.time.copy()))
    native.restore(checkpoint)
    replay = []
    for _ in range(2):
      native.step()
      state = native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"native restored replay status {state.status.tolist()}")
      replay.append((state.qpos.copy(), state.qvel.copy(), state.qacc.copy(),
                     state.time.copy()))
    checkpoint_replay_bitwise = all(
        np.array_equal(a, b)
        for left, right in zip(future, replay)
        for a, b in zip(left, right))
    if not checkpoint_replay_bitwise:
      raise AssertionError("native checkpoint replay was not bitwise exact")
  result = {
      **structural,
      "estimated_workspace_bytes": int(estimate.memory_bytes),
      "steps": int(steps),
      "peak_active_pairs": peak_active_pairs,
      "peak_contacts": peak_contacts,
      "peak_active_rows": peak_active_rows,
      "peak_native_broadphase_pairs": peak_native_broadphase_pairs,
      "peak_native_contact_slots": peak_native_contact_slots,
      "peak_native_active_rows": peak_native_active_rows,
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "max_qacc_error": max_qacc_error,
      "max_orientation_error_rad": max_orientation_error,
      "checkpoint_replay_bitwise": checkpoint_replay_bitwise,
      "finite": bool(np.all(np.isfinite(actual.qpos))
                     and np.all(np.isfinite(actual.qvel))
                     and np.all(np.isfinite(
                         actual.qacc if native_qacc is None else native_qacc))),
  }
  if check:
    assert result["nv"] > 64, result
    assert result["candidate_pairs"] > 16, result
    assert result["allocated_slots"] > 24, result
    assert result["allocated_rows"] > 256, result
    assert result["peak_active_pairs"] > 16, result
    assert result["peak_contacts"] > 16, result
    assert result["peak_active_rows"] > 256, result
    assert result["finite"], result
    if mode == "metal":
      assert peak_native_broadphase_pairs > 16, result
      assert peak_native_contact_slots > 16, result
      assert peak_native_active_rows > 256, result
      assert max_qpos_error < .01, result
      assert max_qvel_error < .05, result
      assert max_qacc_error < .5, result
      assert max_orientation_error < .02, result
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=250)
  parser.add_argument("--mode", choices=("cpu", "metal"), default="cpu")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record")
  parser.add_argument("--restore-check", action="store_true")
  args = parser.parse_args()
  print(run(args.steps, args.mode, args.check, args.record,
            args.restore_check))


if __name__ == "__main__":
  main()
