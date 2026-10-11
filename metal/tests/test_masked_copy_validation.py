# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Validation contract for masked world copies and fills.

`_copy_world_masked` must validate every dimension/offset/layout before
performing (or skipping) the copy, so malformed empty calls fail instead of
passing silently. `_fill_world_masked` has the matching integer contract.
Uses real method execution on CPU torch; no source-string checks.
"""

import pytest

torch = pytest.importorskip("torch")

from mujoco_metal.coupled_constraints import MetalCoupledConstraints


def _solver(batch=2):
  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  solver._torch = torch
  solver._device = torch.device("cpu")
  solver.batch_size = batch
  solver._world_mask_offset = 0
  solver._constants = {"solver_dims": torch.zeros(32, dtype=torch.int32)}
  solver._position_refresh_copy_dims = torch.zeros(16, dtype=torch.int32)

  class _Lib:
    def __getattr__(self, name):
      raise AssertionError(f"masked shader must not run on CPU: {name}")

  solver._library = _Lib()
  return solver


def _bufs(batch=2, s_stride=4, d_stride=5):
  src = torch.arange(batch * s_stride, dtype=torch.float32).reshape(batch, s_stride)
  dst = torch.full((batch, d_stride), -7.0, dtype=torch.float32)
  return src, dst


def test_negative_width_rejected_without_mutation():
  solver = _solver()
  src, dst = _bufs()
  before = dst.clone()
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width=-1,
                              source_stride=4, destination_stride=5)
  assert torch.equal(dst, before)


def test_zero_width_with_negative_stride_rejected():
  solver = _solver()
  src, dst = _bufs()
  before = dst.clone()
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width=0,
                              source_stride=-4, destination_stride=5)
  assert torch.equal(dst, before)


def test_fractional_dimensions_rejected_not_truncated():
  solver = _solver()
  src, dst = _bufs()
  before = dst.clone()
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width=2.0,
                              source_stride=4, destination_stride=5)
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width=2,
                              source_stride=4.5, destination_stride=5)
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width="2",
                              source_stride=4, destination_stride=5)
  assert torch.equal(dst, before)


def test_zero_width_with_bad_offset_rejected():
  solver = _solver()
  src, dst = _bufs()
  before = dst.clone()
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width=0,
                              source_stride=4, destination_stride=5,
                              source_offset=-1)
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width=0,
                              source_stride=4, destination_stride=5,
                              destination_offset=9)
  assert torch.equal(dst, before)


def test_valid_empty_is_noop_and_keeps_words():
  solver = _solver()
  src, dst = _bufs()
  before = dst.clone()
  solver._copy_world_masked(dst, src, None, width=0,
                            source_stride=4, destination_stride=5)
  assert torch.equal(dst, before)


def test_bounds_contiguity_and_type_checked():
  solver = _solver()
  src, dst = _bufs()
  before = dst.clone()
  # width exceeds stride
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, src, None, width=5,
                              source_stride=4, destination_stride=5)
  # undersized backing
  small = torch.zeros(3, dtype=torch.float32)
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, small, None, width=2,
                              source_stride=4, destination_stride=5)
  # non-contiguous
  noncontig = torch.zeros(4, 4, dtype=torch.float32).t()
  assert not noncontig.is_contiguous()
  with pytest.raises(ValueError):
    solver._copy_world_masked(dst, noncontig, None, width=2,
                              source_stride=4, destination_stride=5)
  assert torch.equal(dst, before)


def test_fill_rejects_fractional_and_negative():
  solver = _solver()
  dst = torch.full((2, 5), -7.0, dtype=torch.float32)
  before = dst.clone()
  with pytest.raises(ValueError):
    solver._fill_world_masked(dst, None, width=2.0, stride=5, offset=0)
  with pytest.raises(ValueError):
    solver._fill_world_masked(dst, None, width=-1, stride=5, offset=0)
  with pytest.raises(ValueError):
    solver._fill_world_masked(dst, None, width=0, stride=5, offset=-1)
  with pytest.raises(ValueError):
    solver._fill_world_masked(dst, None, width=2, stride=1, offset=0)
  assert torch.equal(dst, before)


def test_fill_valid_empty_is_noop():
  solver = _solver()
  dst = torch.full((2, 5), -7.0, dtype=torch.float32)
  before = dst.clone()
  solver._fill_world_masked(dst, None, width=0, stride=5, offset=2)
  assert torch.equal(dst, before)


def test_unmasked_copy_writes_selected_slice_only():
  solver = _solver()
  src, dst = _bufs()
  solver._copy_world_masked(dst, src, None, width=2,
                            source_stride=4, destination_stride=5,
                            source_offset=1, destination_offset=3)
  assert torch.equal(dst[:, 3:5], src[:, 1:3])
  assert torch.equal(dst[:, :3], torch.full((2, 3), -7.0))
