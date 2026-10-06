"""Dense-pair host dispatch and per-world Metal contract regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_metal.smooth_solve import MetalDenseSolve, factored_workspace_elements


class _FakeTensor:
  """Small contiguous tensor stand-in accepted by the host dispatch only."""

  def __init__(self, data, dtype):
    self.data = np.asarray(data)
    self.dtype = dtype
    self.device = SimpleNamespace(type="mps")
    self.shape = self.data.shape

  def is_contiguous(self):
    return bool(self.data.flags.c_contiguous)

  def reshape(self, *shape):
    return _FakeTensor(self.data.reshape(*shape), self.dtype)

  def __getitem__(self, key):
    return _FakeTensor(self.data[key], self.dtype)


class _FakeScalar:
  def __init__(self, array, index):
    self.array, self.index = array, index

  def fill_(self, value):
    self.array[self.index] = value


class _FakeDims:
  def __init__(self, values):
    self.values = np.asarray(values, dtype=np.int32)

  def __getitem__(self, index):
    return _FakeScalar(self.values, index)


def _host_dispatch_solver(batch=1, nv=2):
  solver = object.__new__(MetalDenseSolve)
  solver.batch_size = batch
  solver.nv = nv
  solver.nrhs = 1
  solver._torch = SimpleNamespace(
      Tensor=_FakeTensor, int32="int32", float32="float32")
  solver._all_world_mask = _FakeTensor(np.ones(batch, np.int32), "int32")
  solver._empty_int = _FakeTensor(np.zeros(1, np.int32), "int32")
  solver._empty_input = _FakeTensor(np.zeros(1, np.float32), "float32")
  solver._zero_pair_low = _FakeTensor(
      np.zeros(batch * max(nv, 1), np.float32), "float32")
  solver._factor = _FakeTensor(np.zeros(batch * nv * nv, np.float32), "float32")
  solver._pair_work = _FakeTensor(
      np.zeros(batch * 6 * max(nv, 1), np.float32), "float32")
  solver._solution = _FakeTensor(np.full(batch * nv, 31, np.float32), "float32")
  solver._solution_low = _FakeTensor(
      np.full(batch * max(nv, 1), 32, np.float32), "float32")
  solver._status = _FakeTensor(np.full(batch, 33, np.int32), "int32")
  solver._pair_dims = _FakeDims([nv, batch, 0, 0])
  solver.calls = []
  solver._pair_kernel = lambda *args, **kwargs: solver.calls.append((args, kwargs))
  return solver


def test_host_pair_dispatch_passes_borrowed_high_and_low_aliases_directly():
  solver = _host_dispatch_solver()
  mass = _FakeTensor(np.eye(2, dtype=np.float32).reshape(1, 2, 2), "float32")
  rhs_hi = _FakeTensor(np.zeros((1, 2), np.float32), "float32")
  rhs_low = _FakeTensor(np.zeros((1, 2), np.float32), "float32")
  ids = _FakeTensor(np.asarray([[0, 0]], np.int32), "int32")
  counts = _FakeTensor(np.asarray([[1, 1, 1]], np.int32), "int32")
  awake = {"dof_ids": ids, "counts": counts}

  # The solver's output buffers are also the previous-call borrowed values.
  # The host passes their views through without replacing/copying either word.
  retained_hi = solver._solution.reshape(1, 2)
  retained_low = solver._solution_low.reshape(1, 2)
  solver.run_pair_device(mass, rhs_hi, rhs_low, awake_lists=awake,
                         retained=retained_hi, retained_low=retained_low)
  args, kwargs = solver.calls[-1]
  assert kwargs["threads"] == (1,)
  assert solver._pair_dims.values.tolist() == [2, 1, 1, 1]
  assert np.shares_memory(args[5].data, retained_hi.data)
  assert np.shares_memory(args[6].data, retained_low.data)


def test_nv_zero_pair_workspace_uses_nonempty_indexing_backings():
  workspace = factored_workspace_elements(0, 3)
  assert workspace == {"factor": 0, "pivots": 0,
                       "solution": 0, "status": 3}
  source = __import__("inspect").getsource(MetalDenseSolve.__init__)
  assert "batch_size * 6 * max(nv, 1)" in source
  assert "max(batch_size * nv * nv, 1)" in source
  assert "max(batch_size * nv, 1)" in source


def test_primal_q0_low_snapshot_and_row_low_planes_are_disjoint():
  from pathlib import Path

  root = Path(__file__).resolve().parents[1]
  shader = (root / "mujoco_metal" / "shaders" /
            "coupled_constraints.metal").read_text()
  nv, nr = 5, 7
  intervals = {
      "vector_pair_scratch": (0, 6 * nv),
      "mass_delta_low": (6 * nv, 7 * nv),
      "q0_low_snapshot": (7 * nv, 8 * nv),
      "row_residual_low": (8 * nv, 8 * nv + nr),
      "row_force_low": (8 * nv + nr, 8 * nv + 2 * nr),
      "aref_low": (8 * nv + 2 * nr, 8 * nv + 3 * nr),
  }
  ordered = list(intervals.values())
  assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))
  assert ordered[-1][1] == 8 * nv + 3 * nr
  for name, (start, stop) in intervals.items():
    assert start >= 0 and stop >= start, name

  scalar_start = shader.index("inline int solve_primal_accel_scalar(")
  scalar_end = shader.index("// Candidate narrowphase status", scalar_start)
  scalar_solver = shader[scalar_start:scalar_end]
  assert "acc[i] = q0[i];" in scalar_solver
  assert "acc_low[i] = q0_low[i];" in scalar_solver
  assert "q0_low[" not in scalar_solver.replace("q0_low[i]", "")
  assert "device float* qacc_smooth_low = high_low_scratch + 7 * max(nv, 1);" in shader
  assert "qacc_smooth_low[i] = out_acc[batch * nv + qb + i];" in shader


@pytest.mark.gpu
def test_native_pair_inactive_world_keeps_outputs_and_status_untouched():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("requires native MPS")
  solve = MetalDenseSolve(1, 2)
  solve._solution.fill_(11.0)
  solve._solution_low.fill_(12.0)
  solve._status.fill_(13)
  mass = torch.ones((2, 1, 1), dtype=torch.float32, device="mps")
  rhs_hi = torch.tensor([[2.0], [float("nan")]], dtype=torch.float32, device="mps")
  rhs_low = torch.zeros((2, 1), dtype=torch.float32, device="mps")
  awake = {
      "dof_ids": torch.tensor([[-1], [-1]], dtype=torch.int32, device="mps"),
      "counts": torch.tensor([[1, 1, 1], [-1, -1, -1]],
                              dtype=torch.int32, device="mps"),
  }
  mask = torch.tensor([1, 0], dtype=torch.int32, device="mps")
  high, low, status = solve.run_pair_device(
      mass, rhs_hi, rhs_low, awake_lists=awake,
      retained=torch.ones((2, 1), dtype=torch.float32, device="mps"),
      world_mask=mask)
  np.testing.assert_array_equal(high.detach().cpu().numpy().reshape(-1), [0.0, 11.0])
  np.testing.assert_array_equal(low.detach().cpu().numpy().reshape(-1), [0.0, 12.0])
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [1, 13])


@pytest.mark.gpu
def test_native_pair_retained_inputs_may_alias_prior_borrowed_outputs():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("requires native MPS")
  solve = MetalDenseSolve(2, 1)
  mass = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]],
                      dtype=torch.float32, device="mps")
  first_hi, first_low, status = solve.run_pair_device(
      mass,
      torch.tensor([[2.0, 12.0]], dtype=torch.float32, device="mps"),
      torch.tensor([[0.0, 2.0 ** -22]], dtype=torch.float32, device="mps"))
  assert int(status.detach().cpu().numpy()[0]) == 0
  retained_sleep_hi = float(first_hi[0, 1].detach().cpu())
  retained_sleep_low = float(first_low[0, 1].detach().cpu())
  awake = {
      "dof_ids": torch.tensor([[0, 0]], dtype=torch.int32, device="mps"),
      "counts": torch.tensor([[1, 1, 1]], dtype=torch.int32, device="mps"),
  }
  # first_hi/first_low are borrowed views of this same solver's output stores.
  second_hi, second_low, status = solve.run_pair_device(
      mass,
      torch.tensor([[4.0, 0.0]], dtype=torch.float32, device="mps"),
      torch.zeros((1, 2), dtype=torch.float32, device="mps"),
      awake_lists=awake, retained=first_hi, retained_low=first_low)
  assert int(status.detach().cpu().numpy()[0]) == 0
  hi = second_hi.detach().cpu().numpy()[0]
  low = second_low.detach().cpu().numpy()[0]
  assert hi[0] == 2.0 and low[0] == 0.0
  assert hi[1] == retained_sleep_hi
  assert low[1] == retained_sleep_low




@pytest.mark.gpu
def test_native_pair_malformed_awake_worlds_fail_without_poisoning_valid_world():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("requires native MPS")
  solve = MetalDenseSolve(1, 3)
  mass = torch.ones((3, 1, 1), dtype=torch.float32, device="mps")
  rhs_hi = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device="mps")
  rhs_low = torch.zeros((3, 1), dtype=torch.float32, device="mps")
  awake = {
      "dof_ids": torch.tensor([[-1], [0], [0]], dtype=torch.int32, device="mps"),
      "counts": torch.tensor([[1, 1, 1], [2, 2, 2], [1, 1, 1]],
                              dtype=torch.int32, device="mps"),
  }
  high, low, status = solve.run_pair_device(mass, rhs_hi, rhs_low,
                                           awake_lists=awake)
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [1, 1, 0])
  np.testing.assert_array_equal(high.detach().cpu().numpy().reshape(-1),
                                [0.0, 0.0, 3.0])
  np.testing.assert_array_equal(low.detach().cpu().numpy().reshape(-1),
                                [0.0, 0.0, 0.0])


@pytest.mark.gpu
def test_native_pair_nv_zero_returns_empty_successful_batch():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("requires native MPS")
  solve = MetalDenseSolve(0, 2)
  mass = torch.empty((2, 0, 0), dtype=torch.float32, device="mps")
  rhs_hi = torch.empty((2, 0), dtype=torch.float32, device="mps")
  rhs_low = torch.empty((2, 0), dtype=torch.float32, device="mps")
  awake = {
      "dof_ids": torch.zeros((2, 1), dtype=torch.int32, device="mps"),
      "counts": torch.zeros((2, 3), dtype=torch.int32, device="mps"),
  }
  high, low, status = solve.run_pair_device(mass, rhs_hi, rhs_low,
                                           awake_lists=awake)
  assert tuple(high.shape) == tuple(low.shape) == (2, 0)
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
