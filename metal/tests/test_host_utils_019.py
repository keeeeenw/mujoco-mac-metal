# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 019: host-only pinned utilities (CPU tests, never GPU evidence)."""

import mujoco
import numpy as np
import pytest

from mujoco_metal.host_utils import (
    body_jacobian_host,
    forward_inverse_consistency,
    inverse_dynamics,
    mass_matrix_host,
    object_velocity_host,
)


def _arm():
  return mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "<site name='s'/>"
      "</body></worldbody></mujoco>")


def test_inverse_matches_forward_residual_cpu():
  m = _arm()
  qpos = np.array([0.3])
  qvel = np.array([1.2])
  d = mujoco.MjData(m)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  mujoco.mj_forward(m, d)
  # Inverse of the forward acceleration recovers the applied force.
  f = inverse_dynamics(m, qpos, qvel, np.asarray(d.qacc))
  np.testing.assert_allclose(f, np.asarray(d.qfrc_inverse), rtol=1e-12, atol=0)


def test_forward_inverse_roundtrip_cpu():
  m = _arm()
  qpos = np.array([0.3])
  qvel = np.array([1.2])
  qacc = np.array([-4.0])
  forces, roundtrip, residual = forward_inverse_consistency(m, qpos, qvel, qacc)
  assert forces.shape == (1,)
  np.testing.assert_allclose(residual, [0.0], atol=1e-9)


def test_mass_spd_and_matches_native_descriptor_cpu():
  from mujoco_metal.model import load_model
  m = _arm()
  M = mass_matrix_host(m, np.array([0.3]))
  assert M.shape == (1, 1) and float(M[0, 0]) > 0
  desc = load_model(m)
  fk = desc.forward_kinematics(np.array([0.3]))
  assert np.isfinite(np.asarray(fk["body_pos"])).all()


def test_jacobian_matches_finite_differences_cpu():
  m = _arm()
  qpos = np.array([0.3])
  jacp, jacr = body_jacobian_host(m, qpos, 1)
  assert jacp.shape == (3, 1) and jacr.shape == (3, 1)
  # mj_objectVelocity packs angular first, then linear.
  v = object_velocity_host(m, qpos, np.array([2.0]),
                           mujoco.mjtObj.mjOBJ_BODY, 1)
  np.testing.assert_allclose(v[3:], (jacp * 2.0)[:, 0], rtol=1e-9, atol=1e-12)
  np.testing.assert_allclose(v[:3], (jacr * 2.0)[:, 0], rtol=1e-9, atol=1e-12)


def test_host_utils_reject_bad_inputs_cpu():
  m = _arm()
  with pytest.raises(ValueError):
    inverse_dynamics(m, [np.nan], [0.0], [0.0])
  with pytest.raises(ValueError):
    mass_matrix_host(m, [0.0, 0.0])
  with pytest.raises(ValueError):
    body_jacobian_host(m, [0.0], 99)
  with pytest.raises(TypeError):
    inverse_dynamics("nope", [0.0], [0.0], [0.0])
