# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned Euler advances the forward iterate unless joint damping applies."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.stepping import _euler_damping_dofs, validate_stepping_profile


def _model(damping=0, flag="", solver="Newton", iterations=0):
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="{solver}"
            iterations="{iterations}" tolerance="1e-12">{flag}</option>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0" damping="{damping}"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody><equality>
      <joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/>
    </equality>
  </mujoco>''')


@pytest.mark.parametrize("kind,enabled", [
    ("none", False), ("positive", True), ("negative", False),
    ("polynomial", True), ("inherited", True),
    ("damper_disabled", False), ("eulerdamp_disabled", False)])
def test_compiled_euler_damping_admission_matches_pinned_fields(kind, enabled):
  model = _model(damping=2 if kind in (
      "positive", "damper_disabled", "eulerdamp_disabled") else
      -1 if kind == "negative" else 0)
  if kind == "polynomial":
    model.dof_dampingpoly[0, 1] = .4
  elif kind == "inherited":
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="a"/><geom size=".1"/></body></worldbody>
      <actuator><general joint="a" damping="1"/></actuator></mujoco>''')
    assert model.jnt_actuatorid[0] != -1
  elif kind == "damper_disabled":
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  elif kind == "eulerdamp_disabled":
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
  assert bool(np.any(_euler_damping_dofs(model))) == enabled
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert profile.implicit_euler_damping == enabled
  assert profile.execution_plan.is_stage_enabled("euler_damping") == enabled


def test_pinned_zero_budget_euler_does_not_resolve_constraint_force():
  model = _model()
  data = mujoco.MjData(model)
  data.qpos[:] = [.02, 0]
  mujoco.mj_step(model, data)
  assert np.max(np.abs(data.qfrc_constraint)) > 100
  np.testing.assert_array_equal(data.qacc, 0)
  np.testing.assert_array_equal(data.qvel, 0)
  np.testing.assert_array_equal(data.qpos, [.02, 0])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("damping,flag", [
    (0, ""), (2, ""), (-1, ""), (2, '<flag damper="disable"/>'),
    (2, '<flag eulerdamp="disable"/>')])
def test_native_euler_preserves_pinned_finite_budget_acceleration(damping, flag):
  from mujoco_metal import MetalSimulation
  model = _model(damping, flag)
  qpos = np.array([[.02, 0]], dtype=np.float32)
  qvel = np.array([[.1, -.05]], dtype=np.float32)
  sim = MetalSimulation(model, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  data.qvel[:] = qvel[0]
  for _ in range(5):
    sim.step()
    mujoco.mj_step(model, data)
    snap = sim.state.snapshot()
    np.testing.assert_array_equal(snap.status, 0)
    np.testing.assert_allclose(snap.qpos[0], data.qpos, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(snap.qvel[0], data.qvel, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(snap.qacc[0], data.qacc, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                    "contact_free_forces_euler_v1"])
def test_negative_damping_remains_an_explicit_passive_force(profile):
  model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0">
      <flag contact="disable"/></option>
    <worldbody><body><joint damping="-.5" type="slide"/>
      <geom size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>''')
  plan = validate_stepping_profile(model, profile=profile)
  assert plan.passive_damping_enabled
  assert not plan.implicit_euler_damping
  data = mujoco.MjData(model)
  data.qvel[:] = .2
  mujoco.mj_forward(model, data)
  np.testing.assert_allclose(data.qfrc_passive, .1)
  mujoco.mj_step(model, data)
  assert data.qvel[0] > .2


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                    "contact_free_forces_euler_v1"])
def test_native_signed_damping_explicit_trajectory(profile):
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0">
      <flag contact="disable"/></option><worldbody>
    <body><joint damping="-.5" type="slide"/>
      <geom size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>''')
  sim = MetalSimulation(model, qvel=np.array([[.2]], dtype=np.float32),
                        profile=profile)
  cpu = mujoco.MjData(model)
  cpu.qvel[:] = .2
  assert sim._euler_solver is None
  for _ in range(20):
    sim.step()
    mujoco.mj_step(model, cpu)
  snap = sim.state.snapshot()
  np.testing.assert_array_equal(snap.status, 0)
  np.testing.assert_allclose(snap.qpos[0], cpu.qpos, atol=2e-7, rtol=2e-6)
  np.testing.assert_allclose(snap.qvel[0], cpu.qvel, atol=2e-7, rtol=2e-6)
