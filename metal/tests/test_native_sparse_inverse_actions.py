# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CSR native reductions retain operand range, row order and sparse ownership."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_native_inverse_map_rejects_invalid_topology_before_allocation(monkeypatch):
  from mujoco_metal.inverse_constraints import _native_csr_map
  from mujoco_metal.constraint_jacobian import PackedJacobianLayout, PACKED_J_CSR
  layout = PackedJacobianLayout.create(2, 3, pattern=SimpleNamespace(
      nr=2, nv=3, nnz=3), mode=PACKED_J_CSR, paired=True)
  monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs:
                      pytest.fail("invalid metadata reached device allocation"))
  for offsets, columns in (([0, 2, 3], [1, 1, 2]),
                            ([0, 3, 2], [0, 1, 2]),
                            ([0, 1, 3], [0, 1, 3]),
                            ([1, 2, 3], [0, 1, 2])):
    with pytest.raises(ValueError, match="CSR"):
      _native_csr_map(layout, SimpleNamespace(row_offsets=np.asarray(offsets),
                         columns=np.asarray(columns)), 2, "mps")
  with pytest.raises(ValueError, match="signed int32"):
    _native_csr_map(layout, SimpleNamespace(row_offsets=np.array([0, 1, 3]),
                   columns=np.array([0, 1, 2])), 2**30, "mps")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_sparse_paired_actions_extreme_operands_empty_rows_and_isolation():
  from mujoco_metal.constraint_jacobian import (
      ConstraintJacobianPattern, PackedJacobianLayout, PACKED_J_CSR,
      initialize_packed_jacobian_records)
  from mujoco_metal.inverse_constraints import (
      _packed_jacobian_pair_matvec, _packed_jacobian_transpose_pair_matvec)
  offsets = np.array([0, 2, 2, 4, 6], np.int32)
  columns = np.array([0, 3, 1, 2, 0, 2], np.int32)
  pattern = ConstraintJacobianPattern(offsets, columns, 4, 4)
  layout = PackedJacobianLayout.create(4, 4, pattern=pattern,
                                       mode=PACKED_J_CSR, paired=True)
  buffer = np.zeros(2 * layout.stride_words, np.float32)
  initialize_packed_jacobian_records(buffer.view(np.int32), 2, layout, pattern)
  values = np.array([1e35, .7, 1e-35, -.2, .3, -.5], np.float32)
  matrices = []
  for world in range(2):
    base = world * layout.stride_words
    correction = values * np.float32((-1 if world else 1) * 2**-26)
    buffer[base+layout.values_offset:base+layout.values_offset+6] = values
    buffer[base+layout.lo_values_offset():base+layout.lo_values_offset()+6] = correction
    matrix = np.zeros((4, 4), np.float64)
    for row in range(4):
      start, end = offsets[row:row+2]
      matrix[row, columns[start:end]] = (values[start:end].astype(np.float64)
                                        + correction[start:end].astype(np.float64))
    matrices.append(matrix)
  tensor = lambda v: torch.tensor(v, dtype=torch.float32, device="mps")
  packed = tensor(buffer)
  before = packed.clone()
  acceleration = tensor([[1e-20, 1e20, .2, .4], [-1e-20, -1e20, -.2, -.4]])
  high, low = _packed_jacobian_pair_matvec(packed, layout, pattern, acceleration)
  expected = np.stack([matrices[w] @ acceleration[w].cpu().double().numpy()
                       for w in range(2)])
  np.testing.assert_allclose(high.cpu().double().numpy() + low.cpu().double().numpy(),
                             expected, rtol=3e-15, atol=3e-15)
  force = tensor([[1e-20, 0, 1e20, .4], [-1e-20, 0, -1e20, -.4]])
  force_low = force * 2**-26
  live = torch.tensor([[True, False, True, True], [False, False, True, True]], device="mps")
  # A disabled row must not load a poisoned high or residual entry.
  base = layout.stride_words
  packed[base+layout.values_offset:base+layout.values_offset+2] = float("nan")
  packed[base+layout.lo_values_offset():base+layout.lo_values_offset()+2] = float("nan")
  poisoned = packed.clone()
  out, out_low = _packed_jacobian_transpose_pair_matvec(
      packed, layout, pattern, force, force_low, live)
  effective_force = force.cpu().double().numpy() + force_low.cpu().double().numpy()
  expected = np.stack([matrices[w].T @ (effective_force[w] * live[w].cpu().numpy())
                       for w in range(2)])
  np.testing.assert_allclose(out.cpu().double().numpy() + out_low.cpu().double().numpy(),
                             expected, rtol=3e-15, atol=3e-15)
  torch.testing.assert_close(packed.view(torch.int32), poisoned.view(torch.int32), rtol=0, atol=0)
  assert torch.equal(before[:layout.stride_words], packed[:layout.stride_words])
