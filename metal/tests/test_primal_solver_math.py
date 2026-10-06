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
from pathlib import Path
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


def _mixed_primal(q0, acc, mass, J, aref, kinds, compliance, cone):
  """Acceleration-space objective, gradient, Hessian for mixed row blocks."""
  delta = acc - q0
  residual = J @ acc - aref
  force = np.zeros(J.shape[0], dtype=np.float64)
  Hrow = np.zeros((J.shape[0], J.shape[0]), dtype=np.float64)
  cost = 0.5 * delta @ mass @ delta
  for row, kind in enumerate(kinds):
    r = residual[row]
    if kind == "equality":
      cost_row, force[row] = _equality(r, compliance[row])
      Hrow[row, row] = 1.0 / compliance[row]
    elif kind == "limit":
      cost_row, force[row] = _inequality(r, compliance[row])
      Hrow[row, row] = 1.0 / compliance[row] if r < 0 else 0.0
    elif kind == "friction":
      loss = 0.4
      cost_row, force[row] = _friction(r, compliance[row], loss)
      Hrow[row, row] = 1.0 / compliance[row] if abs(r) < compliance[row] * loss else 0.0
    else:
      raise AssertionError(kind)
    cost += cost_row
  start = len(kinds)
  dim = len(cone[0])
  cost_cone, force_cone = _elliptic(
      residual[start:start + dim], *cone)
  Hrow[start:start + dim, start:start + dim] = _elliptic_hessian(
      residual[start:start + dim], *cone)
  force[start:start + dim] = force_cone
  cost += cost_cone
  gradient = mass @ delta - J.T @ force
  hessian = mass + J.T @ Hrow @ J
  return cost, gradient, hessian


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


@pytest.mark.parametrize("dim", [4, 6])
@pytest.mark.parametrize("target", ["top", "middle"])
def test_elliptic_bottom_zone_transition_subtracts_initial_cost(dim, target):
  """A zone-crossing line delta includes the initial bottom-zone cost.

  MuJoCo 3.10's `ellipticCostDif` uses the bottom quadratic's constant at
  alpha zero when the endpoint leaves the bottom zone.  The shader's shifted
  bottom polynomial therefore needs its constant for zone-transition
  fallback, even though a bottom-to-bottom delta uses only its linear and
  quadratic coefficients.
  """
  friction = np.array([0.8, 0.6, 0.04, 0.1, 0.2])
  residual = np.zeros(dim, dtype=np.float64)
  residual[0] = -2.0
  compliance = np.linspace(0.1, 0.4, dim)
  mu = 0.8
  start_cost, _ = _elliptic(residual, compliance, mu, friction)
  assert start_cost > 0.0

  direction = np.zeros(dim, dtype=np.float64)
  if target == "top":
    direction[0] = 4.0
    end = residual + direction
    end_cost, _ = _elliptic(end, compliance, mu, friction)
    # The normal rises above the cone boundary; source top-zone cost is zero.
    assert end_cost == 0.0
  else:
    direction[1] = 8.0
    end = residual + direction
    end_cost, _ = _elliptic(end, compliance, mu, friction)
    # This endpoint is in the middle zone and retains a positive cone cost.
    assert end_cost > 0.0
  expected_delta = end_cost - start_cost
  assert np.isfinite(expected_delta) and abs(expected_delta) > 1e-8

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  helper = shader.split("inline float primal_elliptic_cost_at", 1)[1].split(
      "inline float primal_elliptic_cost_delta", 1)[0]
  delta = shader.split("inline float primal_elliptic_cost_delta", 1)[1].split(
      "// Compute the source's shifted primal", 1)[0]
  assert "bottom_constant + alpha * alpha * bottom_quadratic" in helper
  assert "bottom_constant, bottom_linear, bottom_quadratic" in delta


def test_primal_bracket_numerical_fallback_keeps_midpoint():
  """Pinned PrimalSearch returns its midpoint even without cost decrease."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  solve = shader.split("inline int solve_primal_accel_scalar", 1)[1].split(
      "inline void seed_pgs_warmstart_qacc", 1)[0]
  fallback = solve.split("if (!b1 && !b2)", 1)[1].split("line_done = true;", 1)[0]
  assert "accepted_alpha = pmid.alpha;" in fallback
  assert "if (pmid.cost < 0.0f)" not in fallback


def test_primal_linepoint_uses_prepared_cost_and_derivatives():
  """Line search must not recover slopes from rounded candidate qacc."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  evaluator = shader.split("inline PrimalLinePoint primal_line_eval_elliptic",
                           1)[1].split("// Pinned MuJoCo CG operates", 1)[0]
  assert "primal_accel_cost_delta_elliptic" in evaluator
  assert "PrimalFloatPair slope" in evaluator
  assert "PrimalFloatPair curvature" in evaluator
  assert "n_eq_rows, slope, curvature," in evaluator
  assert "return {alpha, cost_delta.hi, cost_delta.lo," in evaluator
  assert "slope.hi, slope.lo, curvature.hi, curvature.lo" in evaluator
  assert "primal_accel_eval_elliptic" not in evaluator
  assert "primal_accel_curvature_elliptic" not in evaluator
  assert "primal_elliptic_line_derivatives" in shader


def test_primal_accepted_state_increment_preserves_sub_ulp_updates():
  """The 3.10 Ma/Jaref update retains increments rounded qacc cannot hold."""
  q0 = np.float32(1_000_000.0)
  aref = np.float32(q0 - np.float32(0.25))
  alpha = np.float32(0.25)
  direction = np.float32(0.125)
  mass = np.float32(3.0)
  row_j = np.float32(1.0)

  qacc = np.float32(q0 + alpha * direction)
  assert qacc == q0  # alpha*direction is below qacc's local ULP.
  recomputed_mass_delta = np.float32(mass * np.float32(qacc - q0))
  updated_mass_delta = np.float32(
      np.float32(0.0) + alpha * np.float32(mass * direction))
  assert recomputed_mass_delta == 0.0
  assert updated_mass_delta == np.float32(0.09375)

  initial_row_residual = np.float32(row_j * qacc - aref)
  recomputed_row_residual = np.float32(row_j * qacc - aref)
  updated_row_residual = np.float32(
      initial_row_residual + alpha * np.float32(row_j * direction))
  assert recomputed_row_residual == initial_row_residual
  assert updated_row_residual == np.float32(0.28125)

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  solve = shader.split("inline int solve_primal_accel_scalar", 1)[1].split(
      "inline void seed_pgs_warmstart_qacc", 1)[0]
  assert "primal_mass_product_pair(mass, sparse_mass, debug, dims, direction" in solve
  assert "PrimalFloatPair accepted_alpha" in solve
  assert "primal_pair_fma_pair(accepted_alpha," in solve
  assert "{old_gradient[i], mass_delta_low[i]}" in solve
  assert "{residual[row], row_residual_low[row]}" in solve
  assert "{direction[i], direction_low[i]}, {acc[i], acc_low[i]}" in solve
  assert "old_gradient[i] = updated_mass_delta.hi;" in solve
  assert "mass_delta_low[i] = updated_mass_delta.lo;" in solve
  assert "row_residual_low[row] = updated.lo;" in solve


def test_pair_mass_preconditioner_clears_low_word_before_matrix_product():
  """Residual formation must not read stale low words from prior PCG H*v."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  helper = shader.split(
      "__attribute__((noinline)) bool primal_hessian_mass_precondition_pair_elliptic(",
      1)[1].split("inline PrimalLinePoint primal_line_eval_elliptic", 1)[0]
  clear = helper.index("for (int dof = 0; dof < nv; ++dof) solution_low[dof] = 0.0f;")
  residual_rows = helper.index(
      "for (int dof = 0; dof < nv; ++dof) {", clear + len(
          "for (int dof = 0; dof < nv; ++dof) solution_low[dof] = 0.0f;"))
  assert clear < residual_rows
  residual_product = helper.index("primal_compensated_product_add(mij, solution_low[other]")
  assert residual_rows < residual_product
  assert "!solver_dof_awake(dims, awake_tree, other)" in helper
  assert "bool direct = !partitioned && (sparse_mass || !has_inactive_dof);" in helper


def test_final_dual_certificate_matches_retained_row_system():
  """The published diagnostic is the projected residual of retained rows."""
  # Equality row has an intentionally small but real residual.  This guards
  # against reporting the unrelated, already-small primal objective gradient.
  W = np.asarray([[2.0, -0.5], [-0.5, 1.5]], dtype=np.float64)
  R = np.asarray([0.25, 0.1], dtype=np.float64)
  rhs = np.asarray([1.0, -0.4], dtype=np.float64)
  ar = np.asarray([0.2, -0.1], dtype=np.float64)
  lam = np.asarray([0.4, 0.2 + 2.0e-6], dtype=np.float64)
  lo = np.asarray([-np.inf, 0.0], dtype=np.float64)
  hi = np.asarray([np.inf, np.inf], dtype=np.float64)
  gradient = (W + np.diag(R)) @ lam - rhs
  projected = np.clip(lam - gradient / np.maximum(np.diag(W) + R, 1e-15), lo, hi)
  scale = np.maximum(1.0, np.abs(ar) + np.abs(R * lam)
                     + np.sum(np.abs(W * lam[None, :]), axis=1))
  expected = np.max(np.abs(projected - lam) * np.maximum(np.diag(W) + R, 1e-15)
                    / scale)
  assert expected > 1.0e-7

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  helper = shader.split("inline float cert_retained_system_residual", 1)[1].split(
      "// Scalable block-row coupled constraint solver", 1)[0]
  assert "W[row * nr + col] * lam[col]" in helper
  assert "R[row] * lam[row]" in helper
  assert "abs(projected - lam[row]) * diag" in helper
  assert shader.count("= cert_retained_system_residual(W, R, rhs, ar,") == 2
  assert "solver_diagnostics[0] certifies the multipliers" in shader


def test_outer_solver_stopping_uses_requested_tolerance_slot():
  """The float32 certification floor must not replace MuJoCo's opt.tolerance."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  assert shader.count("float tol = params[6];") == 2
  assert "params[2] remains the public float32 certification floor" in shader
  assert "certification floor must never relax that criterion" in shader


@pytest.mark.parametrize("dim", [3, 4, 6])
def test_mixed_primal_elliptic_objective_gradient_and_hessian(dim):
  """Cone curvature composes with scalar rows and the mass Gauss term."""
  rng = np.random.default_rng(900 + dim)
  nv = dim + 1
  raw = rng.normal(size=(nv, nv))
  mass = raw.T @ raw + np.eye(nv)
  q0 = rng.normal(size=nv) * 0.02
  acc = q0 + rng.normal(size=nv) * 0.1
  nrow = 3 + dim
  J = rng.normal(size=(nrow, nv))
  desired = np.zeros(nrow)
  desired[:3] = [-0.03, -0.04, 0.01]
  desired[3] = -0.08
  desired[4:3 + dim] = np.linspace(0.7, 0.25, dim - 1)
  aref = J @ acc - desired
  kinds = ["equality", "limit", "friction"]
  compliance = np.r_[0.2, 0.3, 0.4]
  cone = (np.linspace(0.08, 0.19, dim), 0.6,
          np.array([0.6, 0.6, 0.04, 0.12, 0.2]))
  cost, gradient, hessian = _mixed_primal(
      q0, acc, mass, J, aref, kinds, compliance, cone)
  eps = 2e-6
  numerical_gradient = np.zeros(nv)
  numerical_hessian = np.zeros((nv, nv))
  for col in range(nv):
    delta = np.zeros(nv)
    delta[col] = eps
    cp = _mixed_primal(q0, acc + delta, mass, J, aref,
                       kinds, compliance, cone)[0]
    cm = _mixed_primal(q0, acc - delta, mass, J, aref,
                       kinds, compliance, cone)[0]
    gp = _mixed_primal(q0, acc + delta, mass, J, aref,
                       kinds, compliance, cone)[1]
    gm = _mixed_primal(q0, acc - delta, mass, J, aref,
                       kinds, compliance, cone)[1]
    numerical_gradient[col] = (cp - cm) / (2 * eps)
    numerical_hessian[:, col] = (gp - gm) / (2 * eps)
  assert cost > 0 and np.isfinite(cost)
  np.testing.assert_allclose(gradient, numerical_gradient, atol=2e-6, rtol=2e-6)
  np.testing.assert_allclose(hessian, numerical_hessian, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("seed", [41, 47])
def test_matrix_free_primal_hessian_operator_and_cg_residual(seed):
  """Independent large-dimension oracle for Newton's Hessian-vector solve."""
  rng = np.random.default_rng(seed)
  batch, nv, nr = 2, 73, 261
  blocks = [(0, 6), (nr // 2, 4), (nr - 6, 6)]
  for world in range(batch):
    raw = rng.normal(size=(nv, nv))
    spectrum = np.geomspace(1e-3, 1e2, nv)
    mass = (raw / np.sqrt(nv)) @ np.diag(spectrum) @ (raw / np.sqrt(nv)).T
    mass += np.diag(spectrum)
    J = rng.normal(size=(nr, nv)) / np.sqrt(nv)
    row_curvature = rng.uniform(0.0, 2.0, size=nr)
    row_curvature[[row for start, dim in blocks for row in range(start, start + dim)]] = 0
    hessian = mass + J.T @ (row_curvature[:, None] * J)
    for block_index, (start, dim) in enumerate(blocks):
      if block_index == 0:
        residual = np.r_[1.0, np.full(dim - 1, 0.01)]  # cone interior
      elif block_index == 1:
        residual = np.r_[-0.08, np.linspace(0.7, 0.25, dim - 1)]  # cone side
      else:
        residual = np.r_[-1.0, np.full(dim - 1, 0.01)]  # polar region
      compliance = np.linspace(0.08, 0.19, dim)
      friction = np.array([0.6, 0.6, 0.04, 0.12, 0.2])
      block_hessian = _elliptic_hessian(residual, compliance, 0.6, friction)
      hessian += J[start:start + dim].T @ block_hessian @ J[start:start + dim]

    awake = np.ones(nv, dtype=bool)
    awake[::11] = False
    # Pinned sleeping-DOF assembly uses an identity diagonal and zero cross
    # terms, so compare against that exact principal operator.
    active_hessian = np.zeros_like(hessian)
    active_hessian[np.ix_(awake, awake)] = hessian[np.ix_(awake, awake)]
    active_hessian[np.diag_indices(nv)] += (~awake).astype(float)
    rhs = rng.normal(size=nv) * awake
    diagonal = np.diag(active_hessian)
    x = np.zeros(nv)
    r = rhs.copy()
    z = r / diagonal
    p = z.copy()
    rho = r @ z
    for _ in range(nv):
      hp = active_hessian @ p
      alpha = rho / (p @ hp)
      x += alpha * p
      r -= alpha * hp
      if (r @ r) <= 1e-12 * (rhs @ rhs):
        break
      z = r / diagonal
      next_rho = r @ z
      p = z + (next_rho / rho) * p
      rho = next_rho
    relative_residual = np.linalg.norm(active_hessian @ x - rhs) / np.linalg.norm(rhs)
    assert relative_residual <= 1e-6
    expected = np.linalg.solve(active_hessian, rhs)
    np.testing.assert_allclose(x, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("zone", ["top", "middle", "bottom"])
def test_elliptic_cpu_oracle_matches_mj_constraint_update(dim, zone):
  """Compare source-derived cone formulas to MuJoCo's actual CPU update."""
  import mujoco

  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 -9.81" cone="elliptic" impratio="3.5"/>
    <worldbody>
      <geom type="plane" size="2 2 .1" friction=".8 .03 .01"
        condim="{dim}"/>
      <body pos="0 0 .04"><freejoint/>
        <geom type="sphere" size=".05" mass="1"
          friction=".8 .03 .01" condim="{dim}"/>
      </body>
    </worldbody>
  </mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.nefc == dim and data.ncon == 1
  if zone == "top":
    residual = np.zeros(dim, dtype=np.float64)
    residual[0] = 0.2
  elif zone == "bottom":
    residual = np.zeros(dim, dtype=np.float64)
    residual[0] = -0.2
  else:
    residual = np.zeros(dim, dtype=np.float64)
    residual[0] = -0.08
    residual[1:] = np.linspace(0.7, 0.25, dim - 1)

  compliance = data.efc_R[:dim].copy()
  friction = data.contact[0].friction.copy()
  mu = data.contact[0].mu
  cost = np.zeros(1, dtype=np.float64)
  mujoco.mj_constraintUpdate(model, data, residual, cost, 1)
  expected_cost, expected_force = _elliptic(
      residual, compliance, mu, friction)
  np.testing.assert_allclose(cost[0], expected_cost, atol=2e-10, rtol=2e-10)
  np.testing.assert_allclose(data.efc_force[:dim], expected_force,
                             atol=2e-9, rtol=2e-9)
  if zone == "middle":
    expected_hessian = _elliptic_hessian(residual, compliance, mu, friction)
    # The CPU API exposes the dense symmetric local contact Hessian.
    actual_hessian = np.asarray(data.contact[0].H)[:dim * dim].reshape(dim, dim)
    np.testing.assert_allclose(
        actual_hessian, expected_hessian, atol=2e-8, rtol=2e-8)


def test_same_position_velocity_stages_preserve_rows_and_refresh_references():
  """Pinned RK4 POS reuse keeps J/R but changes velocity-dependent ar."""
  import mujoco

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option timestep=".002" gravity="0 0 -9.81" cone="elliptic"
        impratio="3.5" iterations="2"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1" condim="6"
          friction=".9 .5 .2"/>
      <body name="a" pos="0 0 .095"><freejoint/>
        <geom type="sphere" size=".1" mass="1" condim="6"
            friction=".9 .5 .2"/>
      </body>
      <body name="b" pos=".5 0 .5"><freejoint/>
        <geom type="sphere" size=".1" mass="1"
            contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <equality><weld body1="a" body2="b" solref=".02 1"
        solimp=".9 .95 .001 .5 2"/></equality>
  </mujoco>""")
  qpos = np.asarray(model.qpos0).copy()
  velocities = (np.zeros(model.nv),
                np.arange(model.nv, dtype=np.float64) * .17 - .4)
  rows = []
  for qvel in velocities:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    assert data.ncon == 1 and data.contact[0].dim == 6
    assert data.nefc >= 12  # six weld rows plus six elliptic contact rows
    rows.append((np.asarray(data.efc_J).reshape(data.nefc, model.nv).copy(),
                 np.asarray(data.efc_R[:data.nefc]).copy(),
                 np.asarray(data.efc_aref[:data.nefc]).copy()))
  np.testing.assert_allclose(rows[0][0], rows[1][0], atol=1e-12, rtol=1e-12)
  np.testing.assert_allclose(rows[0][1], rows[1][1], atol=1e-12, rtol=1e-12)
  assert np.linalg.norm(rows[0][2] - rows[1][2]) > 1.0


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


@pytest.mark.parametrize("solver", ["CG", "Newton"])
@pytest.mark.parametrize("iterations", [0, 1, 2])
@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("warm_kind", ["accepted", "rejected"])
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_elliptic_primal_solver_low_budgets_match_pinned(
    solver, iterations, dim, warm_kind):
  """Low-budget cone solves retain MuJoCo's finite qacc iterate."""
  import mujoco
  import torch  # noqa: F401
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_setState

  def make_model(budget):
    return mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option gravity="0 0 -9.81" timestep=".002" cone="elliptic"
        impratio="3.5" solver="{solver}" iterations="{budget}"
        tolerance="1e-12"/>
      <worldbody>
        <geom type="plane" size="2 2 .1" friction=".8 .03 .01"
          condim="{dim}"/>
        <body pos="0 0 .04"><freejoint/>
          <geom type="sphere" size=".05" mass="1"
            friction=".8 .03 .01" condim="{dim}"/>
        </body>
      </worldbody>
    </mujoco>""")

  qpos = np.array([0, 0, .04, 1, 0, 0, 0], dtype=np.float64)
  qvel = np.array([.7, -.4, .1, .3, -.2, .5], dtype=np.float64)
  seed_model = make_model(100)
  seed_data = mujoco.MjData(seed_model)
  seed_data.qpos[:] = qpos
  seed_data.qvel[:] = qvel
  mujoco.mj_forward(seed_model, seed_data)
  accepted_seed = seed_data.qacc.copy()
  assert np.isfinite(accepted_seed).all()

  model = make_model(iterations)
  cold = mujoco.MjData(model)
  cold.qpos[:] = qpos
  cold.qvel[:] = qvel
  mujoco.mj_step(model, cold)
  warm_seed = (accepted_seed if warm_kind == "accepted"
               else np.full_like(accepted_seed, 1000.0))
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qpos
  cpu.qvel[:] = qvel
  cpu.qacc_warmstart[:] = warm_seed
  mujoco.mj_step(model, cpu)
  if iterations == 0:
    if warm_kind == "accepted":
      assert np.linalg.norm(cpu.qacc - cold.qacc) > 1e-3
    else:
      assert np.linalg.norm(cpu.qacc - warm_seed) > 1e-3
  assert cpu.ncon == 1
  expected_mu = (cpu.contact[0].friction[0]
                 * np.sqrt(cpu.efc_R[1] / cpu.efc_R[0]))
  np.testing.assert_allclose(cpu.contact[0].mu, expected_mu,
                             atol=1e-12, rtol=1e-12)
  assert not np.isclose(cpu.contact[0].mu, cpu.contact[0].friction[0])

  native = MetalSimulation(
      model, qpos=qpos[None, :].astype(np.float32),
      qvel=qvel[None, :].astype(np.float32),
      profile="integrated_euler_v1")
  mj_setState(native, {"qacc_warmstart": warm_seed[None, :].astype(np.float32)})
  native.step()
  np.testing.assert_allclose(native.state.qacc.detach().cpu().numpy()[0],
                             cpu.qacc, atol=3e-3, rtol=3e-4)
  np.testing.assert_allclose(native.state.qvel.detach().cpu().numpy()[0],
                             cpu.qvel, atol=3e-5, rtol=3e-5)
  assert np.isfinite(native.state.qacc.detach().cpu().numpy()).all()


@pytest.mark.parametrize("solver", ["CG", "Newton"])
@pytest.mark.parametrize("iterations", [0, 1, 2])
@pytest.mark.parametrize("dim", [3, 4, 6])
@pytest.mark.parametrize("warm_kind", ["accepted", "rejected"])
def test_pinned_elliptic_contact_cpu_low_budget_oracle(
    solver, iterations, dim, warm_kind):
  """Pin CPU acceleration references for the native low-budget cone matrix."""
  import mujoco

  def make_model(budget):
    return mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option gravity="0 0 -9.81" timestep=".002" cone="elliptic"
        impratio="3.5" solver="{solver}" iterations="{budget}"
        tolerance="1e-12"/>
      <worldbody>
        <geom type="plane" size="2 2 .1" friction=".8 .03 .01"
          condim="{dim}"/>
        <body pos="0 0 .04"><freejoint/>
          <geom type="sphere" size=".05" mass="1"
            friction=".8 .03 .01" condim="{dim}"/>
        </body>
      </worldbody>
    </mujoco>""")

  qpos = np.array([0, 0, .04, 1, 0, 0, 0], dtype=np.float64)
  qvel = np.array([.7, -.4, .1, .3, -.2, .5], dtype=np.float64)
  seed_model = make_model(100)
  seed = mujoco.MjData(seed_model)
  seed.qpos[:] = qpos
  seed.qvel[:] = qvel
  mujoco.mj_forward(seed_model, seed)
  accepted_seed = seed.qacc.copy()

  model = make_model(iterations)
  cold = mujoco.MjData(model)
  cold.qpos[:] = qpos
  cold.qvel[:] = qvel
  mujoco.mj_forward(model, cold)
  warm = mujoco.MjData(model)
  warm.qpos[:] = qpos
  warm.qvel[:] = qvel
  warm_seed = (accepted_seed if warm_kind == "accepted"
               else np.full_like(accepted_seed, 1000.0))
  warm.qacc_warmstart[:] = warm_seed
  mujoco.mj_forward(model, warm)
  assert cold.nefc == warm.nefc == dim
  assert np.isfinite(warm.qacc).all()
  assert warm.ncon == 1
  expected_mu = (warm.contact[0].friction[0]
                 * np.sqrt(warm.efc_R[1] / warm.efc_R[0]))
  np.testing.assert_allclose(warm.contact[0].mu, expected_mu,
                             atol=1e-12, rtol=1e-12)
  assert not np.isclose(warm.contact[0].mu, warm.contact[0].friction[0])
  if iterations == 0 and warm_kind == "accepted":
    assert np.linalg.norm(warm.qacc - cold.qacc) > 1e-3
  if iterations == 0 and warm_kind == "rejected":
    # A rejected seed falls back to qacc_smooth, which need not be the cold
    # qacc chosen from a different qacc_warmstart cost comparison.
    assert np.linalg.norm(warm.qacc - warm_seed) > 1e-3
