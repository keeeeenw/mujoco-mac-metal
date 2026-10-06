"""Opt-in witness for the actual paired-CG restricted mass preconditioner."""
from pathlib import Path
import os

import numpy as np
import pytest


def _cases():
  # L is exactly representable in float32 and M=L L^T is exactly integral.
  # Cut cases select strict principal subsets while M has nonzero cross-island
  # entries, so production must use its restricted fallback.
  factor = np.asarray([[2, 0, 0], [1, 2, 0], [0, 1, 2]], dtype=np.float32)
  mass = factor.astype(np.float64) @ factor.astype(np.float64).T
  rhs_hi = np.asarray([5.125, -2.75, 1.375], dtype=np.float32)
  rhs_lo = np.asarray([2**-20, -2**-18, 2**-19], dtype=np.float32)
  assert np.any(rhs_hi != 0) and np.any(rhs_lo != 0)
  assert mass[0, 1] != 0.0
  return [
      ("cut-singleton", mass, factor, rhs_hi, rhs_lo,
       np.asarray([0, 1, 1], dtype=np.int32), np.asarray([0], dtype=np.int32)),
      ("cut-two-dof", mass, factor, rhs_hi, rhs_lo,
       np.asarray([1, 0, 0], dtype=np.int32), np.asarray([1, 2], dtype=np.int32)),
      ("complete", mass, factor, rhs_hi, rhs_lo,
       np.asarray([0, 0, 0], dtype=np.int32), np.asarray([0, 1, 2], dtype=np.int32)),
  ]


def test_cut_fixture_really_exercises_restricted_mass_operator():
  cases = _cases()
  for cut in cases[:2]:
    assert np.all(cut[4] != 0.0)
    rhs = cut[3].astype(np.float64) + cut[4].astype(np.float64)
    selected = cut[6]
    restricted_mass = cut[1][np.ix_(selected, selected)]
    restricted = np.linalg.solve(restricted_mass, rhs[selected])
    unrestricted = np.linalg.solve(cut[1], rhs)[selected]
    high_only = np.linalg.solve(
        restricted_mass, cut[3][selected].astype(np.float64))
    assert np.max(np.abs(restricted - unrestricted)) > 1.0e-2
    # Negative control: dropping the low RHS misses the source result by much
    # more than the required two-word residual budget.
    assert np.max(np.abs(high_only - restricted)) > 1.0e-8
  two_dof = cases[1]
  assert np.linalg.det(two_dof[1][np.ix_(two_dof[6], two_dof[6])]) == 21.0


def test_native_paired_preconditioner_matches_restricted_binary64_system():
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native MPS execution is opt-in via MUJOCO_METAL_RUN_GPU=1")
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")

  shader_root = Path(__file__).parents[1] / "mujoco_metal" / "shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=shader_root)
  library = torch.mps.compile_shader(source)

  for name, mass, factor, rhs_hi, rhs_low, islands, selected in _cases():
    packed = np.concatenate((
        mass.astype(np.float32).reshape(-1), factor.reshape(-1), rhs_hi, rhs_low
    )).astype(np.float32)
    dims = np.zeros((27,), dtype=np.int32)
    dims[20] = 0  # dense mass mode
    dims[22] = 3  # awake-tree count
    dims[23] = 24  # per-DOF tree-id map
    dims[24:27] = np.arange(3, dtype=np.int32)
    awake = np.ones((3,), dtype=np.int32)
    dof_island = islands
    device_input = torch.tensor(packed, dtype=torch.float32, device="mps")
    device_dims = torch.tensor(dims, dtype=torch.int32, device="mps")
    device_awake = torch.tensor(awake, dtype=torch.int32, device="mps")
    device_island = torch.tensor(dof_island, dtype=torch.int32, device="mps")
    output = torch.full((31,), float("nan"), dtype=torch.float32, device="mps")
    library.primal_cg_mass_precondition_pair_witness(
        device_input, device_dims, device_awake, device_island, output,
        threads=(1,), group_size=(1,))
    actual = output.cpu().numpy().astype(np.float64)
    assert actual[24] == 1.0, f"production helper rejected {name} witness"

    selected = np.flatnonzero(islands == 0)
    rhs = rhs_hi.astype(np.float64) + rhs_low.astype(np.float64)
    expected = np.linalg.solve(mass[np.ix_(selected, selected)], rhs[selected])
    paired = actual[selected] + actual[3 + selected]
    np.testing.assert_allclose(paired, expected, rtol=0.0, atol=2.0e-12)
    residual_hi = actual[25 + selected]
    residual_low = actual[28 + selected]
    residual = residual_hi + residual_low
    true_residual = rhs[selected] - mass[np.ix_(selected, selected)] @ paired
    np.testing.assert_allclose(residual, true_residual, rtol=0.0, atol=2.0e-12)
    assert np.max(np.abs(true_residual)) <= 2.0e-12
    print("PRIMAL_CG_PRECONDITION", name,
          "solution=", paired.tolist(), "expected=", expected.tolist(),
          "residual=", true_residual.tolist(), flush=True)
