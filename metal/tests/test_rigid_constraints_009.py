# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Remaining rigid-constraint qualification (CPU admission + GPU parity)."""

import math
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _ball_model(extra="", option='<option timestep="0.002"/>'):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><compiler angle="radian"/>{option}<worldbody>'
      f'<body pos="0 0 1"><joint name="b" type="ball" {extra}/>'
      '<geom type="sphere" size="0.1" mass="1"/></body>'
      '</worldbody></mujoco>')


def _unit_quat(angle, axis=(1.0, 0.0, 0.0)):
  q = np.array([math.cos(angle / 2), axis[0] * math.sin(angle / 2),
                axis[1] * math.sin(angle / 2), axis[2] * math.sin(angle / 2)])
  return q / np.linalg.norm(q)


def test_ball_limit_admission_cpu():
  d = lower_coupled_constraints(_ball_model('limited="true" range="0 0.5"'))
  assert d.nr > 0
  # Free joints accept no limit attributes at all (compiler-owned); the
  # lowering's limited-free guard agrees for programmatic models.
  with pytest.raises(ValueError, match="unrecognized attribute"):
    mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body pos="0 0 1"><freejoint limited="true"/>'
        '<geom type="sphere" size="0.1" mass="1"/></body>'
        '</worldbody></mujoco>')
  # Unlimited free joints are fine.
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><compiler angle="radian"/><worldbody>'
      '<body pos="0 0 1"><freejoint/>'
      '<geom type="sphere" size="0.1" mass="1"/></body>'
      '</worldbody></mujoco>')
  lower_coupled_constraints(m)


def test_distance_equality_rejected_as_removed_cpu():
  # The distance element was removed upstream (2.2.2): the schema itself
  # rejects it, and the lowering agrees for any programmatic model.
  with pytest.raises(ValueError, match="unrecognized element"):
    mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>'
        '<body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>'
        '</worldbody><equality><distance body1="b1" body2="b2"/></equality></mujoco>')


def test_ball_limit_row_matches_pinned_math_cpu():
  # Independent math: dist = max(range) - angle, J = -unit axis.
  # (Ball ranges start at 0 by compiler contract.)
  m = _ball_model('limited="true" range="0 0.5"')
  d = mujoco.MjData(m)
  d.qpos[:4] = _unit_quat(0.6, (0.0, 1.0, 0.0))
  mujoco.mj_forward(m, d)
  assert d.nefc == 1
  np.testing.assert_allclose(d.efc_J[:3], [0.0, -1.0, 0.0], atol=1e-7)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_ball_limit_native_parity_gpu():
  from mujoco_metal import MetalSimulation
  m = _ball_model('limited="true" range="0 0.5"')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[:4] = _unit_quat(0.5236)
  cpu.qvel[0] = 2.0
  mujoco.mj_forward(m, cpu)
  assert cpu.nefc == 1
  sim.reset(qpos=np.asarray(cpu.qpos, dtype=np.float32).reshape(1, -1),
            qvel=np.asarray(cpu.qvel, dtype=np.float32).reshape(1, -1))
  max_err = 0.0
  for _ in range(60):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    gq = sim.state.qpos.cpu().numpy()[0]
    max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert max_err < 1e-4, max_err
  # Limit holds: angle stays within range + small tolerance.
  gq = sim.state.qpos.cpu().numpy()[0]
  ang = 2.0 * math.acos(min(1.0, abs(float(gq[0]))))
  assert ang < 0.55, ang


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_ball_limit_boundary_holds_gpu():
  from mujoco_metal import MetalSimulation
  m = _ball_model('limited="true" range="0 0.3"')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.qpos[:4] = _unit_quat(0.29)  # just inside
  cpu.qvel[0] = 3.0  # driving outward
  mujoco.mj_forward(m, cpu)
  sim.reset(qpos=np.asarray(cpu.qpos, dtype=np.float32).reshape(1, -1),
            qvel=np.asarray(cpu.qvel, dtype=np.float32).reshape(1, -1))
  for _ in range(30):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq, np.asarray(cpu.qpos), rtol=1e-5, atol=5e-5)
  ang = 2.0 * math.acos(min(1.0, abs(float(gq[0]))))
  assert ang < 0.35, ang


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_composition_both_cones_gpu():
  from mujoco_metal import MetalSimulation
  for cone in ("pyramidal", "elliptic"):
    world = ('<geom name="floor" type="plane" size="5 5 0.1"/>'
             '<body pos="0 0 1"><joint name="b" type="ball" limited="true" range="0 0.6"/>'
             '<geom type="sphere" size="0.1" mass="1"/></body>'
             '<body pos="0.8 0 0.5"><joint name="s" type="slide" axis="1 0 0" frictionloss="0.2"/>'
             '<geom type="sphere" size="0.05" mass="0.5"/></body>'
             '<body pos="0.8 0 0.05"><freejoint name="f"/>'
             '<geom name="box" type="box" size="0.05 0.05 0.05" mass="0.3"/></body>')
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><option timestep="0.002" cone="{cone}"/>'
        f'<worldbody>{world}</worldbody>'
        '<contact><pair geom1="box" geom2="floor"/></contact>'
        '<tendon><fixed name="t1"><joint joint="s" coef="1.0"/></fixed>'
        '<fixed name="t2"><joint joint="s" coef="-1.0"/></fixed></tendon>'
        '<equality><tendon tendon1="t1" tendon2="t2" polycoef="0 1 0 0 0"/></equality></mujoco>')
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    cpu = mujoco.MjData(m)
    cpu.qpos[:4] = _unit_quat(0.7)  # violates the 0.6 ball limit
    cpu.qvel[0] = 1.0
    mujoco.mj_forward(m, cpu)
    sim.reset(qpos=np.asarray(cpu.qpos, dtype=np.float32).reshape(1, -1),
              qvel=np.asarray(cpu.qvel, dtype=np.float32).reshape(1, -1))
    max_err = 0.0
    for _ in range(40):
      sim.step(1)
      mujoco.mj_step(m, cpu)
      gq = sim.state.qpos.cpu().numpy()[0]
      max_err = max(max_err, float(np.max(np.abs(gq - cpu.qpos))))
    assert max_err < 1e-3, (cone, max_err)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_redundant_and_failed_neighbor_isolation_gpu():
  from mujoco_metal import MetalSimulation
  # Two identical tendon equalities (redundant) + batch isolation.
  world = ('<body pos="0 0 1"><joint name="j1" type="slide" axis="1 0 0"/>'
           '<geom type="sphere" size="0.05" mass="1"/></body>'
           '<body pos="1 0 1"><joint name="j2" type="slide" axis="1 0 0"/>'
           '<geom type="sphere" size="0.05" mass="1"/></body>')
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><compiler angle="radian"/><option timestep="0.002"/>'
      f'<worldbody>{world}</worldbody>'
      '<tendon><fixed name="f1"><joint joint="j1" coef="1.0"/></fixed>'
      '<fixed name="f2"><joint joint="j2" coef="1.0"/></fixed></tendon>'
      '<equality><tendon tendon1="f1" tendon2="f2" polycoef="0 1 0 0 0"/>'
      '<tendon tendon1="f1" tendon2="f2" polycoef="0 1 0 0 0"/></equality></mujoco>')
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.3, -0.1], [10.0, 10.0]], dtype=np.float32),
            qvel=np.array([[0.5, -0.2], [0.0, 0.0]], dtype=np.float32))
  st = sim.step(5)
  status = st.cpu().numpy() if hasattr(st, "cpu") else np.asarray(st)
  gq = sim.state.qpos.cpu().numpy()
  assert np.all(np.isfinite(gq[0])), gq[0]
