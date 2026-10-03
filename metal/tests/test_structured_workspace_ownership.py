# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Retained compaction maps and query workspace restoration own their values."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.row_compaction import CompactionMap, ConstraintCompaction
from mujoco_metal.simulation import _clone_system_dict, _restore_system_buffers


def test_structured_maps_clone_without_alias_and_restore_borrowed_storage():
  torch = pytest.importorskip("torch")
  def mapping():
    return CompactionMap(torch.tensor([[2, 3, -1]]),
                         torch.tensor([[-1, -1, 0, 1]]),
                         torch.tensor([2]), torch.tensor([0]))
  maps = ConstraintCompaction(mapping(), mapping(), mapping())
  source = {"maps": maps, "tensor": torch.tensor([3.]), "label": "logical"}
  saved = _clone_system_dict(source)
  assert saved["maps"] is not source["maps"]
  assert saved["maps"].slots is not source["maps"].slots
  borrowed = maps.slots.packed_to_logical
  pointer = borrowed.data_ptr()
  borrowed.fill_(-1)
  source["tensor"].fill_(8)
  assert saved["maps"].slots.packed_to_logical[0, 0] == 2
  assert saved["tensor"].item() == 3
  _restore_system_buffers(source, saved)
  assert maps.slots.packed_to_logical.data_ptr() == pointer
  torch.testing.assert_close(maps.slots.packed_to_logical,
                             saved["maps"].slots.packed_to_logical)
  torch.testing.assert_close(source["tensor"], saved["tensor"])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_sensor_query_preserves_structured_maps_without_host_warmstart(monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="plane" size="1 1 .1"/>
    <body pos="0 0 .08"><freejoint/><geom type="sphere" size=".1" mass="1"/>
      <site name="imu"/>
    </body></worldbody><sensor><accelerometer site="imu"/></sensor></mujoco>''')
  simulation = MetalSimulation(model, profile="integrated_euler_v1")
  simulation.step()
  cc = simulation._coupled_constraints
  before = _clone_system_dict(cc._workspace)
  def forbid(*args, **kwargs):
    raise AssertionError("native current-state query must not roundtrip warmstarts through host")
  for name in ("get_warmstart", "set_warmstart", "clear_warmstart"):
    monkeypatch.setattr(cc, name, forbid)
  value = simulation.sensor_values()
  assert value.device.type == "mps"
  assert np.all(np.isfinite(value.detach().cpu().numpy()))
  torch.testing.assert_close(cc._workspace["workspace_debug"],
                             before["workspace_debug"], rtol=0, atol=0)
  torch.testing.assert_close(cc._workspace["slot_maps"].packed_to_logical,
                             before["slot_maps"].packed_to_logical, rtol=0, atol=0)
