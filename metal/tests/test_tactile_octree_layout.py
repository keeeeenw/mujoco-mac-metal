# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""The actual tactile constant builder and its admission inventory agree."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mujoco_metal.sensors import SensorProgram, tactile_workspace_sizes
from mujoco_metal.capacity import _runtime_buffer_sizes
from test_tactile_parity import _model


@pytest.mark.parametrize("octree", [False, True])
def test_actual_tactile_builder_preserves_octree_words_and_deduplicates(octree):
  model = _model("mesh", mesh_octree=octree)
  if octree:
    # Exercise low and tail words rather than an accidentally exact head.
    for field in (model.oct_aabb, model.oct_coeff):
      nonzero = field != 0
      field[nonzero] = np.nextafter(field[nonzero], np.inf)
  program = SensorProgram.__new__(SensorProgram)
  program._torch, program._device, program.batch_size = torch, "cpu", 2
  program._tactile_sensors = [0]
  program._build_tactile_constants(model)
  sizes = tactile_workspace_sizes(model)
  inventory = dict(_runtime_buffer_sizes(model, 2))
  for name, count in sizes.items():
    actual = getattr(program, "_tactile_" + name)
    assert actual.numel() == count
    assert inventory["sensor.tactile." + name] == actual.numel()
  if not octree:
    assert program._tactile_oct.numel() == 1
    assert np.all(program._tactile_oct_info.numpy() == -1)
    return
  info = program._tactile_oct_info.numpy().reshape(-1, 2)
  used = info[info[:, 0] >= 0]
  assert len(used) == 2  # Mesh and SDF geoms share the same compiled asset.
  np.testing.assert_array_equal(used[0], used[1])
  offset, count = used[0]
  assert offset == 0
  words = program._tactile_oct.numpy()
  assert words.size == 50 * count
  mesh = int(model.geom_dataid[np.flatnonzero(info[:, 0] >= 0)[0]])
  adr = int(model.mesh_octadr[mesh])
  np.testing.assert_array_equal(words[:8*count], model.oct_child[adr:adr+count].reshape(-1))
  for field, width, starts in (("oct_aabb", 6, (8, 22, 36)),
                              ("oct_coeff", 8, (14, 28, 42))):
    planes = [words[start*count:(start+width)*count].astype(np.float64)
              for start in starts]
    source = getattr(model, field)[adr:adr+count].reshape(-1)
    assert np.any(planes[1] != 0)
    reconstructed = planes[0] + planes[1] + planes[2]
    np.testing.assert_array_equal(reconstructed.view(np.uint64), source.view(np.uint64))


def test_tactile_builder_rejects_nonfinite_compiled_octree_before_upload():
  model = _model("mesh", mesh_octree=True)
  model.oct_coeff[0, 0] = np.nan
  program = SensorProgram.__new__(SensorProgram)
  program._torch, program._device, program.batch_size = torch, "cpu", 1
  program._tactile_sensors = [0]
  with pytest.raises(ValueError, match="finite"):
    program._build_tactile_constants(model)
