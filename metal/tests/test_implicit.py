"""Pinned MuJoCo CPU oracle and support boundaries for implicitfast."""
import numpy as np
import pytest
import mujoco

from mujoco_metal.implicit import (
    ImplicitFastDescriptor, implicitfast_oracle, lower_implicitfast,
)

XML = '''<mujoco><option integrator="implicitfast" timestep=".004" gravity="0 0 0"><flag contact="disable"/></option>
<worldbody><body><joint name="hinge" type="hinge" axis="0 1 0" damping=".3" stiffness="2" springref=".1"/>
<geom type="box" size=".1 .2 .3" mass="1"/><body pos=".3 0 0"><joint name="ball" type="ball" damping=".12"/>
<geom type="box" size=".08 .09 .1" mass=".3"/></body><body pos="0 0 .7"><joint name="slide" type="slide" axis="0 0 1" damping=".2"/>
<geom type="capsule" size=".07 .2" mass=".4"/></body></body></worldbody>
<actuator><motor joint="hinge" gear=".6"/></actuator></mujoco>'''


def _state(model, flip_damper=False):
  data = mujoco.MjData(model)
  data.qpos[:] = [.25, 1., 0., 0., 0., .2]
  data.qvel[:] = np.linspace(-.4, .5, model.nv)
  data.ctrl[:] = .15
  mujoco.mj_forward(model, data)
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  return data, mass


def test_lowering_snapshots_damping_and_requires_implicitfast():
  model = mujoco.MjModel.from_xml_string(XML)
  descriptor = lower_implicitfast(model)
  assert isinstance(descriptor, ImplicitFastDescriptor)
  assert descriptor.nv == model.nv
  assert descriptor.dof_damping.flags.writeable is False
  with pytest.raises(ValueError):
    descriptor.dof_damping[0] = 0
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
  with pytest.raises(ValueError, match="implicitfast integrator"):
    lower_implicitfast(model)


@pytest.mark.parametrize("disable_damper", [False, True])
def test_oracle_matches_real_mj_step_velocity_and_position_for_scalar_and_ball(disable_damper):
  model = mujoco.MjModel.from_xml_string(XML)
  if disable_damper:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  data, mass = _state(model)
  qpos0, qvel0 = data.qpos.copy(), data.qvel.copy()
  oracle = implicitfast_oracle(model, mass, data.qfrc_smooth)
  assert np.all(oracle["status"] == 0)
  expected_vel = qvel0 + model.opt.timestep * oracle["qacc"][0]
  expected_pos = qpos0.copy()
  mujoco.mj_integratePos(model, expected_pos, expected_vel, model.opt.timestep)
  mujoco.mj_step(model, data)
  # MuJoCo data.qacc is the forward acceleration; implicitfast's solved
  # acceleration is observed in the velocity increment at mj_advance.
  np.testing.assert_allclose(data.qvel, expected_vel, rtol=1e-8, atol=1e-9)
  np.testing.assert_allclose(data.qpos, expected_pos, rtol=1e-8, atol=1e-9)


def test_custom_full_velocity_derivative_builds_source_exact_effective_mass():
  model = mujoco.MjModel.from_xml_string(XML)
  data, mass = _state(model)
  derivative = np.diag(np.linspace(-.1, -.5, model.nv))
  derivative[0, -1] = 87.0  # upper triangle is ignored by implicitfast's mapD2M gather
  derivative[-1, 0] = -.25
  result = implicitfast_oracle(model, mass, data.qfrc_smooth, derivative)
  mirrored = np.tril(derivative) + np.tril(derivative, -1).T
  expected_matrix = mass - model.opt.timestep*mirrored
  np.testing.assert_allclose(result["effective_mass"][0], expected_matrix)
  np.testing.assert_allclose(expected_matrix @ result["qacc"][0], data.qfrc_smooth, rtol=1e-12, atol=1e-12)


def test_empty_world_is_valid():
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option integrator="implicitfast"><flag contact="disable"/></option><worldbody/></mujoco>'
  )
  result = implicitfast_oracle(model, np.empty((0, 0)), np.empty(0))
  assert result["effective_mass"].shape == (1, 0, 0)
  assert result["qacc"].shape == (1, 0)
  assert result["status"][0] == 0


def test_batched_shapes_and_nonfinite_values_are_rejected():
  model = mujoco.MjModel.from_xml_string(XML)
  data, mass = _state(model)
  with pytest.raises(ValueError, match="shapes"):
    implicitfast_oracle(model, mass[None], data.qfrc_smooth[None, :2])
  with pytest.raises(ValueError, match="finite"):
    implicitfast_oracle(model, np.full_like(mass, np.nan), data.qfrc_smooth)


def test_freejoint_is_allowed_for_midpoint_profile():
  free = '''<mujoco><option integrator="implicitfast"><flag contact="disable"/></option>
  <worldbody><body><freejoint/><geom type="box" size=".1 .2 .3" mass="1"/></body></worldbody></mujoco>'''
  descriptor = lower_implicitfast(mujoco.MjModel.from_xml_string(free))
  assert descriptor.nv == 6


def test_external_full_derivative_unblocks_other_velocity_force_families():
  model = mujoco.MjModel.from_xml_string(XML)
  model.dof_dampingpoly[0, 0] = .02
  # R06/D2: polynomial damping assembles natively (auto derivative flag);
  # the external path still works and agrees on the solve.
  descriptor = lower_implicitfast(model)
  assert descriptor.nv == model.nv
  assert descriptor.auto_derivative is True
  descriptor = lower_implicitfast(model, external_derivative=True)
  assert descriptor.nv == model.nv
  data, mass = _state(model)
  derivative = -np.eye(model.nv)
  result = implicitfast_oracle(model, mass, data.qfrc_smooth, derivative)
  np.testing.assert_allclose(
      result["effective_mass"][0], mass + model.opt.timestep*np.eye(model.nv)
  )


def test_wrong_integrator_fails_explicitly():
  with pytest.raises(ValueError, match="implicitfast integrator"):
    lower_implicitfast(mujoco.MjModel.from_xml_string(
        XML.replace('integrator="implicitfast"', 'integrator="Euler"')
    ))


def test_shader_argument_abi_is_dense():
  import re
  from pathlib import Path
  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "implicit.metal"
  indices = [int(x) for x in re.findall(r"\[\[buffer\((\d+)\)\]\]", shader.read_text())]
  # Sixth output retains the full nonsymmetric operator for stiffness CG.
  assert indices == list(range(6))
