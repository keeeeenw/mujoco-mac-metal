# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU reference contract for deterministic 017 compact execution maps."""

import os
import numpy as np
import pytest

from mujoco_metal.row_compaction import compact_constraints, compact_flags


def test_stable_compaction_roundtrips_and_keeps_batch_worlds_independent():
  flags = np.asarray([
      [0, 1, 1, 0, 1, 0, 0],
      [1, 0, 0, 1, 0, 1, 1],
      [0, 0, 0, 0, 0, 0, 0],
  ], dtype=np.int32)
  mapping = compact_flags(flags, capacity=4)
  np.testing.assert_array_equal(mapping.active_count, (3, 4, 0))
  np.testing.assert_array_equal(mapping.overflow, (0, 0, 0))
  np.testing.assert_array_equal(mapping.packed_to_logical[0], (1, 2, 4, -1))
  np.testing.assert_array_equal(mapping.packed_to_logical[1], (0, 3, 5, 6))
  np.testing.assert_array_equal(mapping.logical_to_packed[0], (-1, 0, 1, -1, 2, -1, -1))
  np.testing.assert_array_equal(mapping.logical_to_packed[1], (0, -1, -1, 1, -1, 2, 3))
  np.testing.assert_array_equal(mapping.logical_to_packed[2], (-1,) * 7)
  for world in range(flags.shape[0]):
    for packed, logical in enumerate(mapping.packed_to_logical[world]):
      if logical >= 0:
        assert mapping.logical_to_packed[world, logical] == packed
        assert flags[world, logical]


def test_overflow_refuses_partial_map_without_losing_requested_count():
  flags = np.asarray([[1, 0, 1, 1], [1, 1, 0, 0]], dtype=np.int32)
  mapping = compact_flags(flags, capacity=2)
  np.testing.assert_array_equal(mapping.active_count, (3, 2))
  np.testing.assert_array_equal(mapping.overflow, (1, 0))
  np.testing.assert_array_equal(mapping.packed_to_logical[0], (-1, -1))
  np.testing.assert_array_equal(mapping.logical_to_packed[0], (-1, -1, -1, -1))
  np.testing.assert_array_equal(mapping.packed_to_logical[1], (0, 1))


def test_all_numeric_nonzero_masks_are_active_including_nonfinite_values():
  flags = np.asarray([[0.0, -0.5, 0.25, 0.5, 1.0, np.nan, np.inf]], dtype=np.float32)
  mapping = compact_flags(flags)
  np.testing.assert_array_equal(mapping.active_count, (6,))
  np.testing.assert_array_equal(mapping.packed_to_logical[0], (1, 2, 3, 4, 5, 6, -1))
  np.testing.assert_array_equal(mapping.logical_to_packed[0], (-1, 0, 1, 2, 3, 4, 5))


@pytest.mark.parametrize("bad_capacity", [-1, 1.5, True])
def test_capacity_must_be_an_explicit_nonnegative_integer(bad_capacity):
  with pytest.raises(ValueError, match="capacity"):
    compact_flags(np.ones((1, 3), dtype=np.int32), bad_capacity)


def test_compaction_keeps_pair_slot_and_logical_row_namespaces_separate():
  comp = compact_constraints(
      pair_flags=np.asarray([[1, 0, 1]], dtype=np.int32),
      slot_flags=np.asarray([[0, 1, 0, 1, 1]], dtype=np.int32),
      row_flags=np.asarray([[1, 0, 1, 1, 0, 0]], dtype=np.int32),
      pair_capacity=2, slot_capacity=3, row_capacity=3)
  np.testing.assert_array_equal(comp.pairs.packed_to_logical[0], (0, 2))
  np.testing.assert_array_equal(comp.slots.packed_to_logical[0], (1, 3, 4))
  np.testing.assert_array_equal(comp.rows.packed_to_logical[0], (0, 2, 3))
  assert comp.rows.logical_count == 6
  np.testing.assert_array_equal(comp.rows.logical_to_packed[0], (0, -1, 1, 2, -1, -1))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_metal_scan_crosses_block_boundary_and_reports_exact_overflow():
  import torch
  from mujoco_metal.row_compaction import compact_flags
  flags = np.zeros((2, 513), dtype=np.float32)
  flags[0, [0, 255, 256, 512]] = (1.0, -0.5, 0.25, np.inf)
  flags[1, :5] = 1.0
  mapping = compact_flags(torch.as_tensor(flags, device="mps"), capacity=4)
  np.testing.assert_array_equal(mapping.active_count.cpu().numpy(), (4, 5))
  np.testing.assert_array_equal(mapping.overflow.cpu().numpy(), (0, 1))
  np.testing.assert_array_equal(mapping.packed_to_logical[0].cpu().numpy(), (0, 255, 256, 512))
  np.testing.assert_array_equal(mapping.logical_to_packed[0, [0, 255, 256, 512]].cpu().numpy(), (0, 1, 2, 3))
  np.testing.assert_array_equal(mapping.packed_to_logical[1].cpu().numpy(), (-1,) * 4)
  np.testing.assert_array_equal(mapping.logical_to_packed[1, :5].cpu().numpy(), (-1,) * 5)
