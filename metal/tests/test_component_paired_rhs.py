# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual component solves retain residual inputs for every constraint row."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from test_component_mass_solver import _layout


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native paired RHS")
@pytest.mark.parametrize("indefinite", [False, True])
def test_full_paired_rows_fused_retained_factors_and_legacy_input(indefinite):
  import torch
  from mujoco_metal.component_solve import MetalComponentMassSolver

  solver = MetalComponentMassSolver(SimpleNamespace(nv=3, nbody=4), _layout(),
                                   batch_size=2, rhs_capacity=4)
  matrices = np.asarray([[2., .75, .75, 1.25, 3.],
                         [1.5, -.25, -.25, 2.5, 4.]], dtype=np.float32)
  if indefinite:
    matrices[:, 4] *= -1
  dense = np.zeros((2, 3, 3))
  dense[:, 0, 0] = matrices[:, 0]; dense[:, 0, 2] = matrices[:, 1]
  dense[:, 2, 0] = matrices[:, 2]; dense[:, 2, 2] = matrices[:, 3]
  dense[:, 1, 1] = matrices[:, 4]
  rng = np.random.default_rng(3464759)
  hi = rng.normal(size=(2, 4, 3)).astype(np.float32)
  lo = (rng.normal(size=hi.shape) * 2e-8).astype(np.float32)
  mass = torch.as_tensor(matrices, device="mps")
  rhs = torch.as_tensor(hi, device="mps"); low = torch.as_tensor(lo, device="mps")
  expected = np.stack([np.linalg.solve(dense[w],
                       (hi[w].astype(float) + lo[w]).T).T for w in range(2)])
  _, status = solver.run_device(mass, rhs, rhs_low=low,
                                allow_indefinite=indefinite)
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  pair = solver.paired_output.cpu().numpy().astype(float)
  np.testing.assert_allclose(pair[0] + pair[1], expected, rtol=2e-13, atol=2e-13)
  # No host factorization or oracle value enters either native solve.
  np.testing.assert_array_equal(solver.factorize_device(
      mass, allow_indefinite=indefinite).cpu().numpy(), 0)
  solver.solve_factored_device(rhs, rhs_low=low)
  np.testing.assert_array_equal(solver.paired_output.cpu().numpy(), pair)
  # The original 2-D input still applies only to the final smooth-force row.
  final_low = low[:, -1].contiguous()
  solver.solve_factored_device(rhs, rhs_low=final_low)
  legacy = solver.paired_output.cpu().numpy().copy()
  zeros = torch.zeros_like(low); zeros[:, -1].copy_(final_low)
  solver.solve_factored_device(rhs, rhs_low=zeros)
  np.testing.assert_array_equal(solver.paired_output.cpu().numpy(), legacy)
  np.testing.assert_array_equal(rhs.cpu().numpy(), hi)
  np.testing.assert_array_equal(low.cpu().numpy(), lo)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native paired RHS isolation")
def test_nonfinite_first_row_low_isolated_and_invalid_layout_atomic():
  import torch
  from mujoco_metal.component_solve import MetalComponentMassSolver

  solver = MetalComponentMassSolver(SimpleNamespace(nv=3, nbody=4), _layout(),
                                   batch_size=2, rhs_capacity=4)
  mass = torch.tensor([[2., .75, .75, 1.25, 3.]] * 2, device="mps")
  rhs = torch.ones((2, 4, 3), device="mps")
  low = torch.full_like(rhs, 1e-8)
  solver.run_device(mass, rhs, rhs_low=low)
  before = solver.paired_output.cpu().numpy().copy()
  dims = solver._dims.cpu().numpy().copy()
  phase = solver._phase.cpu().numpy().copy()
  for bad in (low[:, :2], torch.zeros((2, 3), dtype=torch.int32, device="mps"),
              torch.zeros((2, 4, 3)), torch.zeros((2, 4, 4), device="mps")):
    with pytest.raises(ValueError, match="rhs_low"):
      solver.run_device(mass, rhs, rhs_low=bad)
    np.testing.assert_array_equal(solver.paired_output.cpu().numpy(), before)
    np.testing.assert_array_equal(solver._dims.cpu().numpy(), dims)
    np.testing.assert_array_equal(solver._phase.cpu().numpy(), phase)
  low[0, 0, 2] = float("nan")
  _, status = solver.run_device(mass, rhs, rhs_low=low,
      world_mask=torch.tensor([1, 0], dtype=torch.int32, device="mps"))
  np.testing.assert_array_equal(status.cpu().numpy(), [[1, 0], [0, 0]])
  pair = solver.paired_output.cpu().numpy()
  np.testing.assert_array_equal(pair[:, 1], before[:, 1])
  np.testing.assert_array_equal(pair[:, 0, :, [0, 2]], 0)
  np.testing.assert_array_equal(pair[:, 0, :, 1], before[:, 0, :, 1])
