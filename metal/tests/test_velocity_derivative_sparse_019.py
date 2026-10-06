# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned sparse velocity-derivative support and COO producer tests."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.velocity_derivative import (
    MetalVelocityDerivativeValues,
    compile_velocity_derivative_layout,
    signed_passive_diagonal,
)


def _model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <worldbody><body><joint type="slide"/><geom type="sphere" size=".1" mass="1"/>
      <body><joint type="hinge"/><geom type="sphere" size=".1" mass="1"/>
      </body></body></worldbody></mujoco>""")


def test_layout_is_exact_sorted_unique_compiled_muJoCo_D_pattern():
  model = _model()
  layout = compile_velocity_derivative_layout(model)
  expected = [(row, int(col)) for row in range(model.nv)
              for col in model.D_colind[
                  model.D_rowadr[row]:model.D_rowadr[row] + model.D_rownnz[row]]]
  assert list(zip(layout.edge_rows.tolist(), layout.edge_cols.tolist())) == expected
  assert list(zip(layout.edge_rows.tolist(), layout.edge_cols.tolist())) == sorted(set(expected))
  assert layout.edge_count == int(model.nD)
  assert layout.diagonal_slots.shape == (model.nv,)
  for dof, slot in enumerate(layout.diagonal_slots):
    assert (layout.edge_rows[slot], layout.edge_cols[slot]) == (dof, dof)
  assert not layout.edge_rows.flags.writeable
  assert not layout.edge_cols.flags.writeable


def test_zero_dof_layout_has_no_logical_edges():
  model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
  layout = compile_velocity_derivative_layout(model)
  assert model.nv == 0
  assert layout.edge_count == 0
  assert layout.edge_rows.shape == layout.edge_cols.shape == (0,)
  assert layout.diagonal_slots.shape == (0,)
  offsets, rows, slots = layout.column_edge_map()
  np.testing.assert_array_equal(offsets, [0])
  # The maps keep one physical sentinel for zero-sized shader bindings while
  # preserving a zero logical edge count.
  np.testing.assert_array_equal(rows, [0])
  np.testing.assert_array_equal(slots, [0])


def test_column_edge_map_partitions_exact_coo_slots_without_reordering():
  layout = compile_velocity_derivative_layout(_model())
  offsets, rows, slots = layout.column_edge_map()
  nv = int(layout.diagonal_slots.size)
  assert offsets.shape == (nv + 1,)
  assert offsets[0] == 0
  assert offsets[-1] == layout.edge_count
  assert np.all(offsets[1:] >= offsets[:-1])
  assert rows.shape == slots.shape == (layout.edge_count,)

  seen = []
  for col in range(nv):
    start, stop = map(int, offsets[col:col + 2])
    actual = list(zip(rows[start:stop].tolist(), slots[start:stop].tolist()))
    expected = [(int(row), int(slot))
                for slot, (row, edge_col) in enumerate(
                    zip(layout.edge_rows, layout.edge_cols))
                if int(edge_col) == col]
    assert actual == expected
    assert [row for row, _ in actual] == sorted(row for row, _ in actual)
    seen.extend(slot for _, slot in actual)
  assert sorted(seen) == list(range(layout.edge_count))
  assert len(set(seen)) == layout.edge_count


def test_dense_projection_is_explicit_cpu_oracle_and_passive_sign_matches_qderiv():
  layout = compile_velocity_derivative_layout(_model())
  matrix = np.arange(2 * layout.diagonal_slots.size**2, dtype=np.float32).reshape(
      2, layout.diagonal_slots.size, layout.diagonal_slots.size)
  projected = layout.project_dense_oracle(matrix)
  np.testing.assert_array_equal(
      projected,
      matrix[:, layout.edge_rows, layout.edge_cols])
  damping = np.array([[1., 0.], [2., 3.]], dtype=np.float32)
  np.testing.assert_array_equal(signed_passive_diagonal(damping), -damping)


def test_implicitfast_projection_mirrors_pinned_lower_triangle_values():
  model = _model()
  layout = compile_velocity_derivative_layout(model)
  nv = int(layout.diagonal_slots.size)
  matrix = np.arange(2 * nv * nv, dtype=np.float32).reshape(2, nv, nv)
  projected = layout.project_symmetric_lower_oracle(matrix)
  expected = matrix[:, np.maximum(layout.edge_rows, layout.edge_cols),
                    np.minimum(layout.edge_rows, layout.edge_cols)]
  np.testing.assert_array_equal(projected, expected)
  source = layout.symmetric_lower_source_slots()
  lookup = {(int(row), int(col)): slot for slot, (row, col) in enumerate(
      zip(layout.edge_rows, layout.edge_cols))}
  for slot, (row, col) in enumerate(zip(layout.edge_rows, layout.edge_cols)):
    assert int(source[slot]) == lookup[(max(int(row), int(col)),
                                         min(int(row), int(col)))]
  # Verify every actual M-pattern qH gather uses the same D entry as the
  # mirror map (the pinned engine_forward.c mapD2M route).
  dense = np.arange(nv * nv, dtype=np.float32).reshape(nv, nv)
  coo = layout.project_symmetric_lower_oracle(dense[None])[0]
  for row in range(nv):
    start = int(model.M_rowadr[row])
    for offset in range(int(model.M_rownnz[row])):
      col = int(model.M_colind[start + offset])
      d_slot = int(model.mapD2M[start + offset])
      assert int(layout.edge_rows[d_slot]) == row
      assert int(layout.edge_cols[d_slot]) == col
      projected_slot = lookup[(row, col)]
      assert coo[projected_slot] == dense[max(row, col), min(row, col)]


def test_compiled_layout_rejects_unsorted_duplicate_or_missing_diagonals():
  model = _model()
  model.D_colind[0], model.D_colind[1] = model.D_colind[1], model.D_colind[0]
  with pytest.raises(ValueError, match="sorted and unique"):
    compile_velocity_derivative_layout(model)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_passive_diagonal_writes_directly_to_batched_coo_slots():
  import torch

  layout = compile_velocity_derivative_layout(_model())
  values = torch.full((2, max(layout.edge_count, 1)), 99.0,
                      dtype=torch.float32, device="mps")
  writer = MetalVelocityDerivativeValues(layout, 2, values)
  damping = torch.tensor([[.5, 2.], [3., 4.]], dtype=torch.float32,
                         device="mps")
  writer.clear_device()
  writer.add_passive_diagonal_device(damping)
  torch.mps.synchronize()
  expected = np.zeros((2, max(layout.edge_count, 1)), dtype=np.float32)
  for world in range(2):
    expected[world, layout.diagonal_slots] = -damping.cpu().numpy()[world]
  np.testing.assert_array_equal(values.cpu().numpy(), expected)
  mirrored = writer.symmetrize_lower_device()
  torch.mps.synchronize()
  expected_mirrored = np.zeros_like(expected)
  source = layout.symmetric_lower_source_slots()
  valid = source >= 0
  expected_mirrored[:, np.flatnonzero(valid)] = expected[:, source[valid]]
  np.testing.assert_array_equal(mirrored.cpu().numpy(), expected_mirrored)
