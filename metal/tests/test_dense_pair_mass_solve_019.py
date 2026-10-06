"""Dense acceleration two-word solve oracle and native witness.

The NumPy tests model the float32 factor plus the production residual
correction using an independent float64 reference. The opt-in MPS test runs
the actual dense paired Metal kernel, including its awake-principal path.
"""

import math
import inspect

import numpy as np
import pytest

from mujoco_metal.smooth_solve import MetalDenseSolve


def _f32(value):
  return np.float32(value)


def _pair_normalize(hi, lo):
  exact = float(hi) + float(lo)
  rounded = _f32(exact)
  return rounded, _f32(exact - float(rounded))


def _pair_add(a, b):
  return _pair_normalize(_f32(a[0] + b[0]),
                         _f32((float(a[0]) + float(a[1])
                               + float(b[0]) + float(b[1]))
                              - float(_f32(a[0] + b[0]))))


def _pair_mul_float(scalar, pair):
  exact_hi = float(_f32(scalar)) * float(pair[0])
  product = _f32(exact_hi)
  error = _f32(exact_hi - float(product))
  low_product = _f32(float(_f32(scalar)) * float(pair[1]))
  return _pair_add((product, error), (low_product, _f32(0)))


def _pair_div_float(pair, divisor):
  exact = (float(pair[0]) + float(pair[1])) / float(_f32(divisor))
  quotient = _f32(exact)
  return quotient, _f32(exact - float(quotient))


def _triangular_pair_solve(lower, rhs_hi, rhs_low):
  n = len(rhs_hi)
  y = [(_f32(0), _f32(0)) for _ in range(n)]
  for row in range(n):
    value = (_f32(rhs_hi[row]), _f32(rhs_low[row]))
    for col in range(row):
      product = _pair_mul_float(lower[row, col], y[col])
      value = _pair_add(value, (-product[0], -product[1]))
    y[row] = _pair_div_float(value, lower[row, row])
  x = [(_f32(0), _f32(0)) for _ in range(n)]
  for row in range(n - 1, -1, -1):
    value = y[row]
    for col in range(row + 1, n):
      product = _pair_mul_float(lower[col, row], x[col])
      value = _pair_add(value, (-product[0], -product[1]))
    x[row] = _pair_div_float(value, lower[row, row])
  return (np.asarray([v[0] for v in x], dtype=np.float32),
          np.asarray([v[1] for v in x], dtype=np.float32))


def _production_shaped_pair_solve(matrix, rhs_hi, rhs_low):
  """CPU witness for float32 Cholesky plus represented-operator correction."""
  matrix = np.asarray(matrix, dtype=np.float32)
  rhs_hi = np.asarray(rhs_hi, dtype=np.float32)
  rhs_low = np.asarray(rhs_low, dtype=np.float32)
  lower = np.linalg.cholesky(matrix.astype(np.float64)).astype(np.float32)
  high, low = _triangular_pair_solve(lower, rhs_hi, rhs_low)
  residual_hi = np.zeros_like(high)
  residual_low = np.zeros_like(low)
  for row in range(len(high)):
    product = (_f32(0), _f32(0))
    for col in range(len(high)):
      term = _pair_mul_float(matrix[row, col], (high[col], low[col]))
      product = _pair_add(product, term)
    residual = _pair_add((rhs_hi[row], rhs_low[row]),
                         (-product[0], -product[1]))
    residual_hi[row], residual_low[row] = residual
  correction_hi, correction_low = _triangular_pair_solve(
      lower, residual_hi, residual_low)
  out_hi, out_low = zip(*(
      _pair_add((high[i], low[i]),
                (correction_hi[i], correction_low[i]))
      for i in range(len(high))))
  return np.asarray(out_hi), np.asarray(out_low)


def test_float32_mass_cancellation_gets_true_residual_correction():
  matrix = np.asarray(
      [[1.0, .999, .998], [.999, 1.0, .997], [.998, .997, 1.0]],
      dtype=np.float32)
  expected = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
  represented_rhs = matrix.astype(np.float64) @ expected
  rhs_hi = represented_rhs.astype(np.float32)
  rhs_low = (represented_rhs - rhs_hi.astype(np.float64)).astype(np.float32)
  lower = np.linalg.cholesky(matrix.astype(np.float64)).astype(np.float32)
  high_only = np.linalg.solve(
      lower.astype(np.float64).T,
      np.linalg.solve(lower.astype(np.float64), rhs_hi.astype(np.float64)))
  paired_hi, paired_low = _production_shaped_pair_solve(
      matrix, rhs_hi, rhs_low)
  paired = paired_hi.astype(np.float64) + paired_low.astype(np.float64)
  high_residual = np.linalg.norm(matrix.astype(np.float64) @ high_only
                                 - represented_rhs)
  paired_residual = np.linalg.norm(matrix.astype(np.float64) @ paired
                                   - represented_rhs)
  assert np.linalg.cond(matrix.astype(np.float64)) > 1000.0
  assert paired_residual < high_residual * 1e-5
  np.testing.assert_allclose(paired, expected, rtol=0, atol=2e-6)


def test_pair_solve_preserves_independent_low_force_word():
  matrix = np.diag(np.asarray([2.0, 8.0], dtype=np.float32))
  high = np.asarray([0.0, -16.0], dtype=np.float32)
  low = np.asarray([2.0 ** -24, 0.0], dtype=np.float32)
  out_hi, out_low = _production_shaped_pair_solve(matrix, high, low)
  np.testing.assert_array_equal(
      out_hi, np.asarray([2.0 ** -25, -2.0], np.float32))
  assert math.isclose(float(out_hi[0]) + float(out_low[0]),
                      2.0 ** -25, rel_tol=0, abs_tol=1e-14)


def test_dense_acceleration_calls_pair_solver_and_charges_workspace():
  from pathlib import Path

  root = Path(__file__).resolve().parents[1]
  simulation = (root / "mujoco_metal" / "simulation.py").read_text()
  capacity = (root / "mujoco_metal" / "capacity.py").read_text()
  dense_solver = inspect.getsource(MetalDenseSolve.run_pair_device)
  from mujoco_metal.simulation import MetalSimulation
  acceleration = inspect.getsource(MetalSimulation.prepare_forward_acceleration)
  ordinary_acceleration = inspect.getsource(MetalSimulation._acceleration)
  assert "self._solver.run_pair_device(" in simulation
  assert "self._rhs, self._rhs_low" in simulation
  assert "retained_low=(self._qacc_low_for_sensor(retained_qacc)" in acceleration
  assert "retained_low=(self._qacc_low_for_sensor(retained_qacc)" in ordinary_acceleration
  assert '"rhs_low", b * nv' in capacity
  assert '"dense_pair_work"' in capacity
  assert "self._pair_kernel(" in dense_solver


def test_retained_dense_pair_uses_only_matching_qacc_low_provenance():
  """Awake retention carries low only with the exact retained high tensor."""
  torch = pytest.importorskip("torch")
  from types import SimpleNamespace
  from mujoco_metal.simulation import MetalSimulation

  sim = object.__new__(MetalSimulation)
  high = torch.tensor([[1.25, -2.5]], dtype=torch.float32)
  low = torch.tensor([[2.0 ** -23, -2.0 ** -22]], dtype=torch.float32)
  sim._state = SimpleNamespace(_torch=torch, generation=7)
  sim._last_sensor_qacc = high
  sim._last_sensor_qacc_version = int(high._version)
  sim._last_sensor_qacc_generation = 7
  sim._last_sensor_qacc_low = low
  assert sim._qacc_low_for_sensor(high) is low

  # Mutating the high word invalidates its cached low residual immediately.
  high.add_(1.0)
  assert sim._qacc_low_for_sensor(high) is None
  sim._last_sensor_qacc_version = int(high._version)
  sim._state.generation = 8
  assert sim._qacc_low_for_sensor(high) is None
  assert sim._qacc_low_for_sensor(torch.zeros_like(high)) is None


@pytest.mark.gpu
def test_native_dense_pair_mass_solver_matches_float64_operator():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("requires native MPS")
  matrix_np = np.asarray(
      [[1.0, .999, .998], [.999, 1.0, .997], [.998, .997, 1.0]],
      dtype=np.float32)
  solution_np = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
  rhs64 = matrix_np.astype(np.float64) @ solution_np
  rhs_hi = rhs64.astype(np.float32)
  rhs_low = (rhs64 - rhs_hi.astype(np.float64)).astype(np.float32)
  solve = MetalDenseSolve(3, 1)
  mass = torch.as_tensor(matrix_np[None], dtype=torch.float32, device="mps")
  high = torch.as_tensor(rhs_hi[None], dtype=torch.float32, device="mps")
  low = torch.as_tensor(rhs_low[None], dtype=torch.float32, device="mps")
  out_hi, out_low, status = solve.run_pair_device(mass, high, low)
  assert int(status.detach().cpu().numpy()[0]) == 0
  reconstructed = (out_hi.detach().cpu().numpy()[0].astype(np.float64)
                   + out_low.detach().cpu().numpy()[0].astype(np.float64))
  residual = np.linalg.norm(matrix_np.astype(np.float64) @ reconstructed - rhs64)
  assert residual < 1e-7


@pytest.mark.gpu
def test_native_dense_pair_awake_solve_retains_both_words_for_asleep_dof():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("requires native MPS")
  solve = MetalDenseSolve(2, 1)
  mass = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]],
                      dtype=torch.float32, device="mps")
  rhs_hi = torch.tensor([[2.0, 0.0]], dtype=torch.float32, device="mps")
  rhs_low = torch.tensor([[2.0 ** -24, 0.0]], dtype=torch.float32, device="mps")
  awake = {
      "dof_ids": torch.tensor([[0, 0]], dtype=torch.int32, device="mps"),
      "counts": torch.tensor([[1, 1, 1]], dtype=torch.int32, device="mps"),
  }
  retained_hi = torch.tensor([[0.0, 7.0]], dtype=torch.float32, device="mps")
  retained_low = torch.tensor([[0.0, 0.125]], dtype=torch.float32, device="mps")
  out_hi, out_low, status = solve.run_pair_device(
      mass, rhs_hi, rhs_low, awake_lists=awake,
      retained=retained_hi, retained_low=retained_low)
  assert int(status.detach().cpu().numpy()[0]) == 0
  hi = out_hi.detach().cpu().numpy()[0]
  low = out_low.detach().cpu().numpy()[0]
  assert hi[0] == 1.0
  assert low[0] == 2.0 ** -25
  assert hi[1] == 7.0
  assert low[1] == 0.125
