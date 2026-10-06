# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""CPU operator and ABI checks for matrix-free scalable implicit solves."""

import os

import numpy as np
import pytest

from mujoco_metal.implicit_effective import (
    effective_gmres_workspace_bytes,
    effective_gmres_workspace_sizes,
    effective_operator_coo_cpu,
    effective_operator_pair_cpu,
    solve_effective_cpu,
    select_gmres_krylov_dimension,
    MetalEffectiveImplicitSolve,
)


def _native_three_dof_component_solver(batch=2):
  import mujoco
  from mujoco_metal.component_solve import MetalComponentMassSolver
  xml = """<mujoco><worldbody>
    <body><joint name='j0' type='hinge'/><geom type='sphere' size='.1'/>
    <body><joint name='j1' type='hinge'/><geom type='sphere' size='.1'/>
    <body><joint name='j2' type='hinge'/><geom type='sphere' size='.1'/>
    </body></body></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  layout = {
      "nv": 3, "ncomponent": 1, "nnz": 9,
      "component_dof_offsets": np.asarray([0, 3], np.int32),
      "component_dof_ids": np.asarray([0, 1, 2], np.int32),
      "component_mass_offsets": np.asarray([0], np.int32),
  }
  return model, MetalComponentMassSolver(
      model, layout, batch_size=batch, rhs_capacity=1), layout


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_restarted_gmres_b2_residual_failure_isolation_and_recovery():
  import torch
  from mujoco_metal.implicit_effective import MetalEffectiveImplicitSolve

  _, component_solver, _ = _native_three_dof_component_solver(batch=2)
  matrices = np.asarray([
      [[2.0, .1, 0.], [.1, 1.5, .2], [0., .2, 1.]],
      [[1.2, 0., .1], [0., 1.7, .15], [.1, .15, 1.3]],
  ], dtype=np.float32)
  mass_blocks = torch.as_tensor(
      matrices.reshape(2, 9).copy(), dtype=torch.float32, device="mps")
  rows = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], np.int32)
  cols = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], np.int32)
  values_np = np.asarray([
      [.3, -.2, .1, .05, .25, -.15, .1, .07, .2],
      [-.1, .15, .25, -.2, .1, .05, .12, -.08, .3],
  ], dtype=np.float32)
  values = torch.as_tensor(values_np.copy(), device="mps")
  rhs_np = np.asarray([[1., -2., .5], [.2, 1.5, -1.]], np.float32)
  rhs = torch.as_tensor(rhs_np.copy(), device="mps")
  ids, counts = component_solver._all_dof_ids, component_solver._all_counts
  solver = MetalEffectiveImplicitSolve(
      component_solver, edge_rows=rows, edge_cols=cols,
      krylov_dimension=2, max_iterations=30)
  got, status, residual, iterations = solver.solve_device(
      mass_blocks, rhs, values, .5, dof_ids=ids, counts=counts,
      relative_tolerance=2e-5, absolute_tolerance=1e-6)
  expected = solve_effective_cpu(
      matrices.astype(np.float64), rows, cols, values_np.astype(np.float64),
      rhs_np.astype(np.float64), .5)
  assert status.cpu().tolist() == [0, 0]
  assert np.all(iterations.cpu().numpy() > 2)  # more than one restart basis
  assert np.all(residual.cpu().numpy() <= 1e-6 + 2e-5 * np.linalg.norm(rhs_np, axis=1))
  np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=3e-4, atol=3e-5)

  # Non-finite data in one world must not contaminate its neighbor, and a
  # subsequent valid call reuses the same persistent shader/workspace safely.
  invalid_np = np.zeros_like(values_np)
  invalid_np[0, 1] = np.nan
  invalid = torch.as_tensor(invalid_np, device="mps")
  recovered, failed_status, _, _ = solver.solve_device(
      mass_blocks, rhs, invalid, .01, dof_ids=ids, counts=counts)
  assert failed_status.cpu().tolist() == [1, 0]
  np.testing.assert_array_equal(recovered.cpu().numpy()[0], np.zeros(3))
  valid_zero = torch.zeros_like(values)
  recovered, recovered_status, recovered_residual, _ = solver.solve_device(
      mass_blocks, rhs, valid_zero, .01, dof_ids=ids, counts=counts)
  assert recovered_status.cpu().tolist() == [0, 0]
  assert np.all(recovered_residual.cpu().numpy() < 1e-5)
  np.testing.assert_allclose(recovered.cpu().numpy(),
                             np.linalg.solve(matrices, rhs_np[..., None])[..., 0],
                             rtol=3e-5, atol=2e-6)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_gmres_happy_breakdown_small_rhs_diagonal_operator():
  import torch
  from mujoco_metal.implicit_effective import MetalEffectiveImplicitSolve

  _, component_solver, _ = _native_three_dof_component_solver(batch=2)
  solver = MetalEffectiveImplicitSolve(
      component_solver,
      edge_rows=np.asarray([0, 1, 2], np.int32),
      edge_cols=np.asarray([0, 1, 2], np.int32),
      krylov_dimension=3, max_iterations=3)
  mass = torch.as_tensor(
      np.broadcast_to(np.eye(3, dtype=np.float32) * .125, (2, 3, 3)).copy(),
      dtype=torch.float32, device="mps").reshape(2, 9).contiguous()
  derivative = torch.zeros((2, 3), dtype=torch.float32, device="mps")
  ids, counts = component_solver._all_dof_ids, component_solver._all_counts
  for scale in (1.0, 1e-3):
    rhs_np = np.asarray([
        [-3.96247953e-4, -2.30804086e-4, -1.47361308e-4],
        [-3.45299020e-4, -6.77518547e-5, -2.90721655e-4],
    ], dtype=np.float32) * scale
    rhs = torch.as_tensor(rhs_np.copy(), dtype=torch.float32, device="mps")
    got, status, residual, iterations = solver.solve_device(
        mass, rhs, derivative, .002, dof_ids=ids, counts=counts,
        relative_tolerance=1e-6, absolute_tolerance=1e-8)
    assert status.cpu().tolist() == [0, 0]
    assert iterations.cpu().tolist() == [1, 1]
    got_np = got.cpu().numpy()
    np.testing.assert_allclose(got_np, 8.0 * rhs_np, rtol=2e-5, atol=2e-8)
    physical_residual = np.linalg.norm(.125 * got_np - rhs_np, axis=1)
    np.testing.assert_allclose(residual.cpu().numpy(), physical_residual,
                               rtol=2e-4, atol=2e-10)
    assert np.all(physical_residual <=
                  1e-8 + 1e-6 * np.linalg.norm(rhs_np, axis=1))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_gmres_tiny_preconditioned_residual_uses_physical_residual():
  import torch
  from mujoco_metal.implicit_effective import MetalEffectiveImplicitSolve

  _, component_solver, _ = _native_three_dof_component_solver(batch=2)
  solver = MetalEffectiveImplicitSolve(
      component_solver,
      edge_rows=np.asarray([0, 1, 2], np.int32),
      edge_cols=np.asarray([0, 1, 2], np.int32),
      krylov_dimension=2, max_iterations=4)
  rhs = torch.as_tensor([[1., -1., .5], [.3, .2, -.1]],
                        dtype=torch.float32, device="mps")
  derivative = torch.zeros((2, 3), dtype=torch.float32, device="mps")
  ids, counts = component_solver._all_dof_ids, component_solver._all_counts
  huge_mass = torch.as_tensor(
      np.broadcast_to(np.diag([1e31, 1e31, 1e31]), (2, 3, 3)).copy(),
      dtype=torch.float32, device="mps").reshape(2, 9).contiguous()
  failed, status, residual, iterations = solver.solve_device(
      huge_mass, rhs, derivative, .01, dof_ids=ids, counts=counts,
      relative_tolerance=1e-6, absolute_tolerance=1e-8)
  # M^-1*b has norm below the kernel's preconditioned-breakdown threshold,
  # while the original-space equation still has an O(1) residual at x=0.
  # The solver must return a local convergence failure instead of accepting
  # the tiny left-preconditioned residual as a physical solution.
  assert status.cpu().tolist() == [2, 2]
  np.testing.assert_array_equal(failed.cpu().numpy(), np.zeros((2, 3), np.float32))
  rhs_norm = torch.linalg.vector_norm(rhs, dim=1).cpu().numpy()
  np.testing.assert_allclose(residual.cpu().numpy(), rhs_norm, rtol=2e-6, atol=2e-7)
  assert np.all(residual.cpu().numpy() > np.asarray([1e-6, 1e-6]))
  assert iterations.cpu().tolist() == [0, 0]

  recovery_rhs = torch.as_tensor([[1., 0., 0.], [0., 1., 0.]],
                                 dtype=torch.float32, device="mps")
  identity = torch.as_tensor(
      np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
      dtype=torch.float32, device="mps").reshape(2, 9).contiguous()
  recovered, recovered_status, recovered_residual, _ = solver.solve_device(
      identity, recovery_rhs, derivative, .01, dof_ids=ids, counts=counts,
      relative_tolerance=1e-6, absolute_tolerance=1e-8)
  assert recovered_status.cpu().tolist() == [0, 0]
  assert np.all(recovered_residual.cpu().numpy() < 1e-6)
  np.testing.assert_allclose(recovered.cpu().numpy(), recovery_rhs.cpu().numpy(),
                             rtol=0, atol=1e-6)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_restarted_gmres_budget_failure_is_world_local_and_recovers():
  import torch
  from mujoco_metal.implicit_effective import MetalEffectiveImplicitSolve

  _, component_solver, _ = _native_three_dof_component_solver(batch=2)
  mass = np.asarray([
      [[2., .1, 0.], [.1, 1.5, .2], [0., .2, 1.]],
      [[1.2, 0., .1], [0., 1.7, .15], [.1, .15, 1.3]],
  ], np.float32)
  mass_blocks = torch.as_tensor(mass.reshape(2, 9).copy(), device="mps")
  rows = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], np.int32)
  cols = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], np.int32)
  solver = MetalEffectiveImplicitSolve(
      component_solver, edge_rows=rows, edge_cols=cols,
      krylov_dimension=1, max_iterations=1)
  values_np = np.asarray([
      [.3, -.2, .1, .05, .25, -.15, .1, .07, .2],
      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
  ], np.float32)
  values = torch.as_tensor(values_np, device="mps")
  rhs_np = np.asarray([[1., -2., .5], [.2, 1.5, -1.]], np.float32)
  rhs = torch.as_tensor(rhs_np, device="mps")
  ids, counts = component_solver._all_dof_ids, component_solver._all_counts
  got, status, residual, iters = solver.solve_device(
      mass_blocks, rhs, values, .01, dof_ids=ids, counts=counts,
      relative_tolerance=1e-7, absolute_tolerance=1e-9)
  assert status.cpu().tolist() == [2, 0]
  assert iters.cpu().tolist() == [1, 1]
  assert residual.cpu().numpy()[0] > 1e-7
  np.testing.assert_array_equal(got.cpu().numpy()[0], np.zeros(3))
  np.testing.assert_allclose(got.cpu().numpy()[1],
                             np.linalg.solve(mass[1], rhs_np[1]),
                             rtol=2e-5, atol=2e-6)
  values.zero_()
  got, status, residual, _ = solver.solve_device(
      mass_blocks, rhs, values, .01, dof_ids=ids, counts=counts,
      relative_tolerance=1e-7, absolute_tolerance=1e-9)
  assert status.cpu().tolist() == [0, 0]
  assert np.all(residual.cpu().numpy() < 1e-5)
  np.testing.assert_allclose(got.cpu().numpy(),
                             np.linalg.solve(mass, rhs_np[..., None])[..., 0],
                             rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_restarted_gmres_zero_dof_returns_empty_success():
  import mujoco
  import torch
  from mujoco_metal.component_solve import MetalComponentMassSolver
  from mujoco_metal.implicit_effective import MetalEffectiveImplicitSolve

  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
  layout = {
      "nv": 0, "ncomponent": 0, "nnz": 0,
      "component_dof_offsets": np.asarray([0], np.int32),
      "component_dof_ids": np.zeros(0, np.int32),
      "component_mass_offsets": np.zeros(0, np.int32),
  }
  component_solver = MetalComponentMassSolver(
      model, layout, batch_size=2, rhs_capacity=1)
  solver = MetalEffectiveImplicitSolve(
      component_solver, edge_rows=np.zeros(0, np.int32),
      edge_cols=np.zeros(0, np.int32), krylov_dimension=1,
      max_iterations=1)
  result = solver.solve_device(
      torch.zeros((2, 0), dtype=torch.float32, device="mps"),
      torch.zeros((2, 0), dtype=torch.float32, device="mps"),
      torch.zeros((2, 1), dtype=torch.float32, device="mps"), .01)
  assert tuple(result[0].shape) == (2, 0)
  assert result[1].cpu().tolist() == [0, 0]


def test_coo_effective_operator_matches_independent_dense_small_oracle():
  rows = np.asarray([0, 0, 1, 2], dtype=np.int32)
  cols = np.asarray([0, 2, 0, 2], dtype=np.int32)
  values = np.asarray([[.4, -.1, .2, .3], [.1, .5, -.3, .2]])
  vector = np.asarray([[1., -2., .5], [-.5, 1.5, 2.]])
  active = np.asarray([[1, 1, 1], [1, 0, 1]], dtype=bool)
  got = effective_operator_coo_cpu(
      vector, rows, cols, values, active_dof=active)
  expected = np.zeros_like(vector)
  for b in range(2):
    for r, c, value in zip(rows, cols, values[b]):
      if active[b, r] and active[b, c]:
        expected[b, r] += value * vector[b, c]
  np.testing.assert_allclose(got, expected)


def test_effective_mass_solve_has_small_residual_and_world_local_active_sets():
  mass = np.asarray([
      [[2.0, .2, 0.], [.2, 1.5, .1], [0., .1, 1.]],
      [[1.1, 0., .3], [0., 2., .4], [.3, .4, 1.8]],
  ])
  rows = np.asarray([0, 0, 1, 2, 2], dtype=np.int32)
  cols = np.asarray([0, 1, 2, 0, 2], dtype=np.int32)
  values = np.asarray([[.1, .05, -.2, .3, .1], [.2, -.1, .25, .4, -.05]])
  rhs = np.asarray([[1., -.5, .2], [.1, 2., -1.]])
  active = np.asarray([[1, 1, 1], [1, 0, 1]], dtype=bool)
  dt = .01
  solution = solve_effective_cpu(
      mass, rows, cols, values, rhs, dt, active_dof=active)
  for world in range(2):
    D = np.zeros((3, 3))
    np.add.at(D, (rows, cols), values[world])
    indices = np.flatnonzero(active[world])
    residual = ((mass[world] - dt * D)[np.ix_(indices, indices)]
                @ solution[world, indices] - rhs[world, indices])
    assert np.linalg.norm(residual) < 1e-12
    np.testing.assert_array_equal(solution[world, ~active[world]], 0.0)


def test_gmres_workspace_scales_from_actual_nv_and_coo_support():
  sizes = effective_gmres_workspace_sizes(batch=2, nv=70, edge_count=311)
  assert sizes["implicit_gmres_basis"] == 2 * 33 * 70
  assert sizes["implicit_gmres_hessenberg"] == 2 * 33 * 32
  assert sizes["implicit_gmres_edge_values"] == 2 * 311
  assert sizes["implicit_gmres_edge_rows"] == 311
  assert sizes["implicit_gmres_params"] == 3
  empty = effective_gmres_workspace_sizes(batch=1, nv=0, edge_count=0)
  assert empty["implicit_gmres_edge_values"] == 1
  assert empty["implicit_gmres_edge_rows"] == 1


def test_gmres_restart_workspace_tracks_configured_memory_budget():
  sizes = effective_gmres_workspace_sizes(
      batch=3, nv=130, edge_count=500, krylov_dimension=12,
      max_iterations=80)
  assert sizes["implicit_gmres_basis"] == 3 * 13 * 130
  assert sizes["implicit_gmres_hessenberg"] == 3 * 13 * 12
  assert sizes["implicit_gmres_coefficients"] == 3 * 12
  assert sizes["implicit_gmres_residual_vector"] == 3 * 130
  assert sizes["implicit_gmres_residual_rhs"] == 3 * 13
  assert sizes["implicit_gmres_dims"] == 5
  byte_count = effective_gmres_workspace_bytes(
      batch=3, nv=130, edge_count=500, krylov_dimension=12,
      max_iterations=80)
  selected = select_gmres_krylov_dimension(
      batch=3, nv=130, edge_count=500, available_bytes=byte_count,
      max_iterations=80)
  assert selected == 12
  assert select_gmres_krylov_dimension(
      batch=3, nv=130, edge_count=500, available_bytes=byte_count - 1,
      max_iterations=80) == 11
  minimum = effective_gmres_workspace_bytes(
      batch=3, nv=130, edge_count=500, krylov_dimension=1,
      max_iterations=80)
  with pytest.raises(ValueError, match="at least"):
    select_gmres_krylov_dimension(
        batch=3, nv=130, edge_count=500, available_bytes=minimum - 1,
        max_iterations=80)


@pytest.mark.parametrize("kwargs", [
    {"batch": True, "nv": 2, "edge_count": 0},
    {"batch": 1.5, "nv": 2, "edge_count": 0},
    {"batch": 1, "nv": 2.5, "edge_count": 0},
    {"batch": 1, "nv": 2, "edge_count": False},
    {"batch": 1, "nv": 2, "edge_count": 0, "krylov_dimension": True},
    {"batch": 1, "nv": 2, "edge_count": 0, "max_iterations": 1.5},
])
def test_gmres_workspace_rejects_nonintegral_and_boolean_dimensions(kwargs):
  with pytest.raises(ValueError, match="integer"):
    effective_gmres_workspace_sizes(**kwargs)


def test_effective_solver_oracle_handles_ill_conditioned_spd_mass_and_cross_edges():
  mass = np.diag([1e-3, 1., 1e3])
  rows = np.asarray([0, 1, 1, 2], np.int32)
  cols = np.asarray([1, 0, 2, 1], np.int32)
  values = np.asarray([[.02, -.03, .04, .05], [-.01, .02, -.03, .04]])
  rhs = np.asarray([[.1, -2., 100.], [-.2, 1., -50.]])
  result = solve_effective_cpu(mass, rows, cols, values, rhs, 1e-3)
  assert np.all(np.isfinite(result))
  for world in range(2):
    D = np.zeros((3, 3))
    np.add.at(D, (rows, cols), values[world])
    residual = (mass - 1e-3 * D) @ result[world] - rhs[world]
    assert np.linalg.norm(residual) <= 2e-10 * np.linalg.norm(rhs[world])


def _independent_restarted_gmres(matrix, rhs, restart, max_iterations,
                                 tolerance):
  """Small textbook left-preconditioned GMRES oracle for restart tests."""
  n = rhs.size
  x = np.zeros(n, dtype=np.float64)
  total = 0
  while total < max_iterations:
    residual = rhs - matrix @ x
    beta = np.linalg.norm(residual)
    if beta <= tolerance:
      return x, 0, beta, total
    width = min(restart, max_iterations - total)
    basis = np.zeros((n, width + 1), dtype=np.float64)
    hessenberg = np.zeros((width + 1, width), dtype=np.float64)
    basis[:, 0] = residual / beta
    used = 0
    for j in range(width):
      work = matrix @ basis[:, j]
      for i in range(j + 1):
        hessenberg[i, j] = basis[:, i] @ work
        work -= hessenberg[i, j] * basis[:, i]
      hessenberg[j + 1, j] = np.linalg.norm(work)
      used = j + 1
      if hessenberg[j + 1, j] > 1e-30:
        basis[:, j + 1] = work / hessenberg[j + 1, j]
      total += 1
      target = np.zeros(j + 2)
      target[0] = beta
      coeff, *_ = np.linalg.lstsq(
          hessenberg[:j + 2, :j + 1], target, rcond=None)
      if np.linalg.norm(target - hessenberg[:j + 2, :j + 1] @ coeff) <= tolerance:
        break
    target = np.zeros(used + 1)
    target[0] = beta
    coeff, *_ = np.linalg.lstsq(
        hessenberg[:used + 1, :used], target, rcond=None)
    x += basis[:, :used] @ coeff
    true_residual = np.linalg.norm(rhs - matrix @ x)
    if true_residual <= tolerance:
      return x, 0, true_residual, total
  return np.zeros_like(x), 2, np.linalg.norm(rhs - matrix @ x), total


def test_restarted_gmres_oracle_checks_true_residual_and_budget_per_world():
  # Nonsymmetric coupled operators exercise restart residuals and isolated
  # convergence budgets independently of the production Metal kernels.
  matrices = [
      np.asarray([[2.0, -1.2, .2], [.1, 1.5, -.7], [0.0, .3, 1.1]]),
      np.asarray([[1.4, .5, 0.0], [-.8, 2.1, .6], [.2, -.1, .9]]),
  ]
  rhs = [np.asarray([1.0, -2.0, .5]), np.asarray([.2, 1.5, -1.0])]
  for matrix, vector in zip(matrices, rhs):
    solution, status, residual, iterations = _independent_restarted_gmres(
        matrix, vector, restart=2, max_iterations=40, tolerance=1e-10)
    assert status == 0
    assert 0 < iterations <= 40
    assert residual <= 1e-10
    np.testing.assert_allclose(matrix @ solution, vector, rtol=1e-9, atol=1e-10)
  _, status, residual, iterations = _independent_restarted_gmres(
      matrices[0], rhs[0], restart=1, max_iterations=1, tolerance=1e-14)
  assert status == 2
  assert iterations == 1
  assert residual > 1e-14


def test_preconditioned_identity_gmres_is_scale_invariant_for_small_rhs():
  # Replays the c003 structure independently: M=0.125 I, D=0, so left
  # preconditioning makes the Arnoldi operator exactly I even though the
  # physical RHS is only around 1e-3. The acceptance check remains in original
  # equation units (M*x-rhs), not in the much smaller preconditioned units.
  mass = np.eye(3, dtype=np.float64) * 0.125
  rhs_base = np.asarray([
      [-3.96247953e-4, -2.30804086e-4, -1.47361308e-4],
      [-3.45299020e-4, -6.77518547e-5, -2.90721655e-4],
  ], dtype=np.float64)
  for scale in (1.0, 1e-3):
    for rhs in rhs_base * scale:
      preconditioned_rhs = np.linalg.solve(mass, rhs)
      solution, status, _, iterations = _independent_restarted_gmres(
          np.eye(3), preconditioned_rhs, restart=3, max_iterations=3,
          tolerance=1e-12)
      assert status == 0
      assert iterations == 1
      np.testing.assert_allclose(mass @ solution, rhs, rtol=1e-12, atol=1e-15)


def test_gmres_coo_layout_rejects_unsorted_and_duplicate_support_before_compile():
  class Solver:
    batch_size = 1
    nv = 3
    factorize_device = lambda *args, **kwargs: None
    solve_factored_vector_device = lambda *args, **kwargs: None
    merge_world_status = lambda *args, **kwargs: None

  with pytest.raises(ValueError, match="sorted"):
    MetalEffectiveImplicitSolve(
        Solver(), edge_rows=np.asarray([1, 0], np.int32),
        edge_cols=np.asarray([0, 2], np.int32))
  with pytest.raises(ValueError, match="duplicate"):
    MetalEffectiveImplicitSolve(
        Solver(), edge_rows=np.asarray([1, 1], np.int32),
        edge_cols=np.asarray([0, 0], np.int32))


def test_effective_operator_pair_keeps_roundoff_from_one_operator_application():
  mass = np.asarray([
      [[1.0000001, -1.0], [-.2, .3]],
      [[2.0, .5], [.25, 1.5]],
  ], dtype=np.float32)
  high = np.asarray([[1.0e8, 1.0e8], [4.0, -3.0]], dtype=np.float32)
  low = np.asarray([[1.0, -1.0], [1.0e-6, -2.0e-6]], dtype=np.float32)
  rows = np.asarray([0, 1, 1], dtype=np.int32)
  cols = np.asarray([0, 0, 1], dtype=np.int32)
  values = np.asarray([[.25, -.1, .2], [.5, .3, -.2]], dtype=np.float32)
  active = np.asarray([[True, True], [True, False]])
  out_hi, out_low = effective_operator_pair_cpu(
      mass, high, low, rows, cols, values, .01, active_dof=active)
  x = high.astype(np.float64) + low.astype(np.float64)
  expected = np.zeros_like(x)
  for world in range(2):
    for row in range(2):
      if not active[world, row]:
        continue
      for col in range(2):
        if active[world, col]:
          expected[world, row] += float(mass[world, row, col]) * x[world, col]
      for r, c, value in zip(rows, cols, values[world]):
        if r == row and active[world, c]:
          expected[world, row] -= .01 * float(value) * x[world, c]
  reconstructed = out_hi.astype(np.float64) + out_low.astype(np.float64)
  np.testing.assert_allclose(reconstructed, expected, rtol=0, atol=1e-7)
  assert out_hi[0, 0] != np.float32(expected[0, 0]) or out_low[0, 0] != 0
  assert out_hi[1, 1] == 0 and out_low[1, 1] == 0


