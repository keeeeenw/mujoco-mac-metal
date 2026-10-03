# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source-derived checks for acceleration-space constraint costs.

These checks use the pinned scalar cost/force equations from MuJoCo 3.10's
``mj_constraintUpdate_impl`` and are intentionally independent of the Metal
solver helpers. They guard the native primal port's row semantics before its
GPU trajectory qualification.
"""

import numpy as np
import os
import pytest


def _equality(residual, compliance):
  stiffness = 1.0 / compliance
  return 0.5 * stiffness * residual * residual, -stiffness * residual


def _inequality(residual, compliance):
  if residual < 0.0:
    return _equality(residual, compliance)
  return 0.0, 0.0


def _friction(residual, compliance, loss):
  stiffness = 1.0 / compliance
  threshold = compliance * loss
  if residual <= -threshold:
    return loss * (-0.5 * threshold - residual), loss
  if residual >= threshold:
    return loss * (-0.5 * threshold + residual), -loss
  return 0.5 * stiffness * residual * residual, -stiffness * residual


def _elliptic(residual, compliance, mu, friction):
  """Pinned mj_constraintUpdate_impl elliptic-contact piecewise cost/force."""
  residual = np.asarray(residual, dtype=np.float64)
  friction = np.asarray(friction, dtype=np.float64)
  D = 1.0 / np.asarray(compliance, dtype=np.float64)
  U = residual.copy()
  U[0] *= mu
  U[1:] *= friction[:len(residual) - 1]
  N = U[0]
  T = np.linalg.norm(U[1:])
  if N >= mu * T or (T <= 0 and N >= 0):
    return 0.0, np.zeros_like(residual)
  if mu * N + T <= 0 or (T <= 0 and N < 0):
    return 0.5 * np.sum(D * residual * residual), -D * residual
  Dm = D[0] / (mu * mu * (1 + mu * mu))
  NmT = N - mu * T
  force = np.zeros_like(residual)
  force[0] = -Dm * NmT * mu
  if T > 0:
    force[1:] = -force[0] / T * U[1:] * friction[:len(residual) - 1]
  return 0.5 * Dm * NmT * NmT, force


def _elliptic_hessian(residual, compliance, mu, friction):
  """Source-derived active middle-zone Hessian of the elliptic cost."""
  residual = np.asarray(residual, dtype=np.float64)
  friction = np.asarray(friction, dtype=np.float64)
  D = 1.0 / np.asarray(compliance, dtype=np.float64)
  U = residual.copy()
  U[0] *= mu
  U[1:] *= friction[:len(residual) - 1]
  N, T = U[0], np.linalg.norm(U[1:])
  if N >= mu * T or (T <= 0 and N >= 0):
    return np.zeros((len(residual), len(residual)))
  if mu * N + T <= 0 or (T <= 0 and N < 0):
    return np.diag(D)
  scale = np.r_[mu, friction[:len(residual) - 1]]
  raw = np.zeros((len(residual), len(residual)))
  raw[0, 0] = 1.0
  raw[0, 1:] = raw[1:, 0] = -mu / T * U[1:]
  tangent = mu * N / (T ** 3) * np.outer(U[1:], U[1:])
  tangent += np.eye(len(residual) - 1) * (mu * mu - mu * N / T)
  raw[1:, 1:] = tangent
  Dm = D[0] / (mu * mu * (1 + mu * mu))
  return Dm * scale[:, None] * raw * scale[None, :]


@pytest.mark.parametrize("residual", [-0.8, -0.03, 0.0, 0.03, 0.8])
def test_primal_equality_cost_force_and_gradient(residual):
  cost, force = _equality(residual, 0.2)
  eps = 1e-6
  plus = _equality(residual + eps, 0.2)[0]
  minus = _equality(residual - eps, 0.2)[0]
  assert np.isfinite(cost)
  np.testing.assert_allclose((plus - minus) / (2 * eps), -force,
                             atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("residual", [-0.8, -0.03, 0.0, 0.03, 0.8])
def test_primal_unilateral_cost_has_signed_active_force(residual):
  cost, force = _inequality(residual, 0.2)
  eps = 1e-6
  plus = _inequality(residual + eps, 0.2)[0]
  minus = _inequality(residual - eps, 0.2)[0]
  np.testing.assert_allclose((plus - minus) / (2 * eps), -force,
                             atol=2e-6, rtol=2e-9)
  assert cost >= 0.0
  if residual > 0:
    assert cost == force == 0.0
  elif residual < 0:
    assert force > 0.0


@pytest.mark.parametrize("residual", [-0.8, -0.03, 0.0, 0.03, 0.8])
def test_primal_dry_friction_huber_cost_matches_force_gradient(residual):
  cost, force = _friction(residual, 0.2, 0.4)
  eps = 1e-6
  plus = _friction(residual + eps, 0.2, 0.4)[0]
  minus = _friction(residual - eps, 0.2, 0.4)[0]
  np.testing.assert_allclose((plus - minus) / (2 * eps), -force,
                             atol=2e-9, rtol=2e-9)
  assert cost >= 0.0
  assert -0.4 <= force <= 0.4


@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("normal,tangent", [(1.0, 0.0), (-1.0, 0.0),
                                             (-0.1, 1.0), (0.1, 1.0)])
def test_elliptic_cone_cost_and_force_cover_all_zones(dim, normal, tangent):
  friction = np.array([0.5, 0.5, 0.03, 0.1, 0.2])
  residual = np.zeros(dim)
  residual[0] = normal
  if dim > 1:
    residual[1] = tangent
  compliance = np.linspace(0.1, 0.2, dim)
  cost, force = _elliptic(residual, compliance, 0.5, friction)
  assert cost >= 0 and np.isfinite(cost)
  assert np.isfinite(force).all()
  eps = 1e-7
  for k in range(dim):
    step = np.zeros(dim)
    step[k] = eps
    plus = _elliptic(residual + step, compliance, 0.5, friction)[0]
    minus = _elliptic(residual - step, compliance, 0.5, friction)[0]
    np.testing.assert_allclose((plus - minus) / (2 * eps), -force[k],
                               atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("dim", [3, 4, 6])
def test_elliptic_middle_zone_hessian_matches_force_derivative(dim):
  friction = np.array([0.6, 0.6, 0.04, 0.12, 0.2])
  residual = np.zeros(dim)
  residual[0] = -0.08
  residual[1:] = np.linspace(0.7, 0.25, dim - 1)
  compliance = np.linspace(0.08, 0.19, dim)
  H = _elliptic_hessian(residual, compliance, 0.6, friction)
  eps = 1e-6
  numerical = np.zeros_like(H)
  for k in range(dim):
    delta = np.zeros(dim)
    delta[k] = eps
    fp = _elliptic(residual + delta, compliance, 0.6, friction)[1]
    fm = _elliptic(residual - delta, compliance, 0.6, friction)[1]
    numerical[:, k] = -(fp - fm) / (2 * eps)
  np.testing.assert_allclose(H, numerical, atol=3e-7, rtol=2e-6)


def test_default_cg_direction_uses_hager_zhang_truncation():
  # A small positive d·y yields the pinned HZ expression and its dynamic lower
  # bound. The PR+ value differs, so a later native source test can catch an
  # accidental substitution of the more common PR+ update.
  direction = np.array([-1.0, -0.5])
  grad_old = np.array([1.0, 0.2])
  grad_new = np.array([0.4, -0.1])
  mgrad_old = np.array([0.5, 0.1])
  mgrad_new = np.array([0.2, -0.05])
  y = grad_new - grad_old
  my = mgrad_new - mgrad_old
  d_dot_y = float(direction @ y)
  assert d_dot_y > 0.0
  beta_hz = (float(y @ mgrad_new)
             - 2.0 * float(y @ my) / d_dot_y * float(direction @ grad_new)) / d_dot_y
  eta_k = -1.0 / (np.linalg.norm(direction)
                  * min(0.01, np.linalg.norm(grad_new)))
  beta = max(eta_k, beta_hz)
  pr_plus = max(0.0, float(grad_new @ (mgrad_new - mgrad_old))
                / max(1e-15, float(grad_old @ mgrad_old)))
  assert beta != pytest.approx(pr_plus)


@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_solver_low_budget_equality_matches_pinned(solver, iterations):
  """The public step must retain MuJoCo's finite acceleration-space iterate."""
  import mujoco
  import torch  # noqa: F401  (ensures the opt-in test has its runtime)
  from mujoco_metal import MetalSimulation

  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" solver="{solver}" iterations="{iterations}"
      tolerance="1e-12"/>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody>
    <equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
  </mujoco>""")
  qpos = np.array([[0.02, 0.0]], dtype=np.float32)
  qvel = np.array([[0.0, 0.0]], dtype=np.float32)
  native = MetalSimulation(model, qpos=qpos, qvel=qvel,
                           profile="integrated_euler_v1")
  native.step()
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos[0]
  cpu.qvel[:] = qvel[0]
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(native.state.qpos.detach().cpu().numpy()[0],
                             cpu.qpos, atol=2e-5, rtol=2e-5)
  np.testing.assert_allclose(native.state.qvel.detach().cpu().numpy()[0],
                             cpu.qvel, atol=2e-4, rtol=2e-4)
  assert np.isfinite(native.state.qacc.detach().cpu().numpy()).all()
  assert np.max(np.abs(cpu.qfrc_constraint)) > 1.0


@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_solver_consumes_and_advances_qacc_warmstart(solver):
  """WARMSTART is an acceleration state and is selected by pinned cost."""
  import mujoco
  import torch  # noqa: F401
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import StateSpec, mj_getState, mj_setState

  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="{solver}"
      iterations="0" tolerance="1e-12"/>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody>
    <equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
  </mujoco>""")
  qpos = np.array([[0.02, 0.0]], dtype=np.float32)
  warm = np.array([[-25.0, 25.0]], dtype=np.float32)
  native = MetalSimulation(model, qpos=qpos,
                           profile="integrated_euler_v1")
  mj_setState(native, {"qacc_warmstart": warm})
  actual_warm = mj_getState(native, StateSpec.WARMSTART)["warmstart"]
  np.testing.assert_array_equal(
      actual_warm.detach().cpu().numpy(), warm)
  native.step()

  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos[0]
  cpu.qacc_warmstart[:] = warm[0]
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(native.state.qpos.detach().cpu().numpy()[0],
                             cpu.qpos, atol=2e-5, rtol=2e-5)
  np.testing.assert_allclose(native.state.qvel.detach().cpu().numpy()[0],
                             cpu.qvel, atol=2e-5, rtol=2e-5)
  np.testing.assert_allclose(native.state.qacc.detach().cpu().numpy()[0],
                             cpu.qacc, atol=2e-5, rtol=2e-5)
  actual_warm = mj_getState(native, StateSpec.WARMSTART)["warmstart"]
  np.testing.assert_allclose(actual_warm.detach().cpu().numpy()[0],
                             cpu.qacc_warmstart, atol=2e-5, rtol=2e-5)

  cold = mujoco.MjData(model)
  cold.qpos[:] = qpos[0]
  mujoco.mj_step(model, cold)
  assert np.max(np.abs(cpu.qpos - cold.qpos)) > 1e-5


@pytest.mark.parametrize("solver", ["CG", "Newton"])
@pytest.mark.parametrize("warm", ["zero", "worse_than_cold"])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_primal_solver_ignores_retained_row_multiplier_for_qacc_warmstart(solver, warm):
  """CG/Newton select the pinned acceleration state, never retained lambda."""
  import mujoco
  import torch  # noqa: F401
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_setState

  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" timestep=".002" solver="{solver}"
      iterations="0" tolerance="1e-12"/>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/></body>
    </worldbody>
    <equality><joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/></equality>
  </mujoco>""")
  qpos = np.array([[0.02, 0.0]], dtype=np.float32)
  native = MetalSimulation(model, qpos=qpos, profile="integrated_euler_v1")
  nr = native._coupled_constraints.descriptor.nr
  retained = np.full((1, nr), 1000.0, dtype=np.float32)
  native._coupled_constraints.set_warmstart(retained)
  warm_acc = (np.zeros((1, model.nv), dtype=np.float32) if warm == "zero"
              else np.full((1, model.nv), 1000.0, dtype=np.float32))
  mj_setState(native, {"qacc_warmstart": warm_acc})
  native.step()

  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos[0]
  cpu.qacc_warmstart[:] = warm_acc[0]
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(native.state.qpos.detach().cpu().numpy()[0],
                             cpu.qpos, atol=2e-5, rtol=2e-5)
  np.testing.assert_allclose(native.state.qvel.detach().cpu().numpy()[0],
                             cpu.qvel, atol=2e-5, rtol=2e-5)
  np.testing.assert_allclose(native.state.qacc.detach().cpu().numpy()[0],
                             cpu.qacc, atol=2e-5, rtol=2e-5)
