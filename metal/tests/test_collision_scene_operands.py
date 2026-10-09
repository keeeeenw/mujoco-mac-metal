# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Collision masks cannot hide assets required by geometry queries."""
import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import (
    lower_collision_geometry, lower_coupled_constraints)
from test_dynamic_hfield_capacity import _model


def test_query_and_physics_pack_identical_complete_terrain_operands():
  model = _model(count=1)
  physical = lower_coupled_constraints(model)
  pairs = tuple(zip(physical.geom1.tolist(), physical.geom2.tolist()))
  query = lower_collision_geometry(model, pairs)
  np.testing.assert_array_equal(query.mesh_hull, physical.mesh_hull)
  np.testing.assert_array_equal(query.mesh_hull_info, physical.mesh_hull_info)
  np.testing.assert_array_equal(query.geom_size, physical.geom_size)
  assert not query.mesh_hull.flags.writeable
  assert query.byte_count == query.mesh_hull.nbytes + query.mesh_hull_info.nbytes


def test_queries_pack_static_masked_assets_without_changing_model():
  model = _model(count=9)
  model.geom_contype[:] = 0
  model.geom_conaffinity[:] = 0
  before = model.geom_contype.copy(), model.geom_conaffinity.copy()
  physical = lower_coupled_constraints(model)
  assert physical.npairs == 0
  sphere = model.geom('sphere').id
  query = lower_collision_geometry(model, [(g, sphere) for g in range(9)])
  for geom in range(9):
    offset, rows, cols = query.mesh_hull_info[9*geom+5:9*geom+8]
    assert (rows, cols) == (80, 96)
    np.testing.assert_array_equal(query.mesh_hull[offset:offset+rows*cols],
                                 model.hfield_data)
  np.testing.assert_array_equal(model.geom_contype, before[0])
  np.testing.assert_array_equal(model.geom_conaffinity, before[1])


@pytest.mark.parametrize('pairs', [[(-1, 0)], [(0, 100)], [(False, 0)], [(0., 1)]])
def test_invalid_scene_pair_ids_fail_before_packing(pairs):
  with pytest.raises(ValueError, match='geom IDs'):
    lower_collision_geometry(_model(), pairs)
