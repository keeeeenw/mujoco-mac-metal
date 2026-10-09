# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned contact fields preserve slot boundaries and sidecar storage."""
import torch
from mujoco_metal.contact_precision import axis_data, RECORD_WORDS


def test_public_axes_borrow_each_owned_slot_without_precision_aliasing():
  records = torch.arange(2 * 3 * RECORD_WORDS, dtype=torch.float32)
  axes = axis_data(records, 2, 3)
  assert axes.shape == (2, 3, 6, 7)
  for world in range(2):
    for slot in range(3):
      base = (world * 3 + slot) * RECORD_WORDS
      torch.testing.assert_close(axes[world, slot].flatten(), records[base:base+42])
  sidecar = records.reshape(2, 3, RECORD_WORDS)[..., 42:].clone()
  axes[..., 0, 0].fill_(-7)
  torch.testing.assert_close(records.reshape(2, 3, RECORD_WORDS)[..., 42:], sidecar)
  assert records[RECORD_WORDS] == -7


def test_empty_contact_axes_preserve_public_shape():
  assert axis_data(torch.empty(0), 2, 0).shape == (2, 0, 6, 7)


import os
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native cache qualification")
@pytest.mark.parametrize("profile", ["integrated_euler_v1", "integrated_scalable_v1"])
def test_actual_contact_precision_cache_survives_workspace_overwrite_and_restore(profile):
  from test_paired_aref_refinement_022 import _torus_model
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model()
  sim = MetalSimulation(model, 1, profile=profile)
  result = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  cc = sim._coupled_constraints
  nr = cc.descriptor.nr
  offset = cc._regularizer_low_debug_offset
  debug = cc._workspace["workspace_debug"].reshape(1, cc._debug_stride)
  expected = debug[:, offset:offset + 6 * max(nr, 1)].clone()
  assert bool(torch.any(expected != 0))
  context = cc.capture_position_context()
  debug[:, offset:offset + 6 * max(nr, 1)].fill_(float("nan"))
  cc.refresh_velocity_context(context, sim.state._qpos, sim.state._qvel)
  torch.testing.assert_close(debug[:, offset:offset + 6 * max(nr, 1)],
                             expected, rtol=0, atol=0)
  checkpoint = sim.snapshot()
  np.testing.assert_array_equal(sim.step().cpu().numpy(), 0)
  sim.assembled_system(recompute=True)
  sim.restore(checkpoint)
  restored = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(restored["status"].cpu().numpy(), 0)
  torch.testing.assert_close(debug[:, offset:offset + 6 * max(nr, 1)],
                             expected, rtol=0, atol=0)
