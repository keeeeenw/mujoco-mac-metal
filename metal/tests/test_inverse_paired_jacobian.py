# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Both words of the forward Jacobian must enter the inverse cost gradient."""
import os
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=[
    pytest.mark.gpu, pytest.mark.skipif(
        os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")])])
@pytest.mark.parametrize("storage", ["dense", "csr"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
def test_inverse_consumes_packed_jacobian_residual_in_both_actions(
    device, storage, paired, inactive):
  torch = pytest.importorskip("torch")
  from mujoco_metal.constraint_jacobian import (
      ConstraintJacobianPattern, PackedJacobianLayout, PACKED_J_CSR,
      PACKED_J_DENSE, initialize_packed_jacobian_records)
  from mujoco_metal.inverse_constraints import inverse_constraint_force
  hi = np.array([[1, .5, 0, 0], [.125, 1, 0, 1], [0, 0, 1, .25]], np.float32)
  offsets = np.array([0, 2, 5, 7], np.int32)
  columns = np.array([0, 1, 0, 1, 3, 2, 3], np.int32)
  pattern = ConstraintJacobianPattern(offsets, columns, 4, 3)
  csr = storage == "csr"
  layout = PackedJacobianLayout.create(3, 4, pattern=pattern if csr else None,
      mode=PACKED_J_CSR if csr else PACKED_J_DENSE, paired=paired)
  backing = np.zeros(2 * layout.stride_words, np.float32)
  initialize_packed_jacobian_records(backing.view(np.int32), 2, layout,
                                     pattern if csr else None)
  low = (hi != 0).astype(np.float32) * np.float32(2**-26)
  effective = []
  active = np.ones((2, 3), np.float32)
  for world in range(2):
    correction = low * (1 if world == 0 else -1)
    values = hi[hi != 0] if csr else hi.ravel()
    corrections = correction[hi != 0] if csr else correction.ravel()
    base = world * layout.stride_words
    backing[base + layout.values_offset:base + layout.values_offset + layout.nnz] = values
    if paired:
      start = base + layout.lo_values_offset()
      backing[start:start + layout.nnz] = corrections
    effective.append(hi.astype(np.float64) + (correction if paired else 0))
    if inactive:
      row = world + 1
      active[world, row] = 0
      begin, end = (int(offsets[row]), int(offsets[row+1])) if csr else (row*4, (row+1)*4)
      start = base + layout.values_offset
      backing[start + begin:start + end] = np.nan
      if paired:
        start = base + layout.lo_values_offset()
        backing[start + begin:start + end] = np.nan
      effective[-1][row] = 0
  tensor = lambda value: torch.tensor(value, dtype=torch.float32, device=device)
  acceleration = np.array([[2, -1, .25, .5], [-.5, .25, 2, -1]], np.float32)
  acceleration_low = np.array([[2**-25, -2**-26, 0, 2**-27],
                               [-2**-26, 0, 2**-25, -2**-27]], np.float32)
  reference = np.array([[4, 1, 2], [1, 2, 4]], np.float32)
  regulator = np.tile(np.array([.5, .25, 1], np.float32), (2, 1))
  rows = dict(J_packed=tensor(backing), jacobian_layout=layout,
      jacobian_pattern=pattern if csr else None, R=tensor(regulator),
      ar=tensor(reference), lo=tensor(np.full((2, 3), -np.inf)),
      hi=tensor(np.full((2, 3), np.inf)), active=tensor(active))
  descriptor = SimpleNamespace(nr=3, nv=4, n_eq_rows=3,
                               ncontacts_max=0, cone_type=0)
  original = rows["J_packed"].clone()
  out_hi, out_low = inverse_constraint_force(rows, tensor(acceleration),
      descriptor, qacc_low=tensor(acceleration_low), return_low=True)
  expected = []
  # Pinned bilateral constraint cost: C = .5 * (J*a-aref)' R^-1 (J*a-aref).
  # Its generalized negative gradient is -J' R^-1 (J*a-aref).
  for world, J in enumerate(effective):
    a = acceleration[world].astype(np.float64) + acceleration_low[world]
    expected.append(-J.T @ (((J @ a - reference[world]) / regulator[world])
                            * active[world]))
  actual = out_hi.cpu().double().numpy() + out_low.cpu().double().numpy()
  np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-13)
  torch.testing.assert_close(rows["J_packed"].view(torch.int32),
                             original.view(torch.int32), rtol=0, atol=0)
