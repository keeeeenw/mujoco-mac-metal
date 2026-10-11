# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native prism EPA words must reach actual contact rows without projection loss."""
import os

import mujoco
import numpy as np
import pytest

from test_primitive_hfield_full_manifold import _LIMITS, _model


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('shape,size', [('cylinder', '.25 .15'), ('box', '.3 .22 .2')])
@pytest.mark.parametrize('reversed_order', [False, True])
def test_native_prism_contact_retains_source_witness_words(shape, size, reversed_order, monkeypatch):
  from mujoco_metal.simulation import MetalSimulation

  model = _model(shape, size, False, reversed_order)
  qpos = np.tile(model.qpos0.astype(np.float32), (2, 1))
  qpos[1, 2] = .8
  qvel = np.tile(np.asarray([.02, -.01, .03, .04, -.02, .01], np.float32), (2, 1))
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
  mujoco.mj_forward(model, data)
  source_points = np.asarray([c.pos.copy() for c in data.contact[:data.ncon]])
  source_normals = np.asarray([c.frame[:3].copy() for c in data.contact[:data.ncon]])
  source_distances = np.asarray([c.dist for c in data.contact[:data.ncon]])
  sim = MetalSimulation(model, 2, qpos=qpos, qvel=qvel,
                        profile='integrated_scalable_v1', limits=_LIMITS)

  def forbidden(*args, **kwargs):
    raise AssertionError('CPU forward/mass physics during native prism publication')

  monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
  monkeypatch.setattr(mujoco, 'mj_fullM', forbidden)
  system = sim.assembled_system(recompute=True)
  cc = sim._coupled_constraints
  host = lambda value: value.detach().cpu().numpy().astype(float)
  np.testing.assert_array_equal(host(system['status']), 0)
  mask = host(system['contact_mask']) > .5
  assert mask[0].sum() == data.ncon and not mask[1].any()
  records = host(cc._workspace['contact_pair_records']).reshape(
      2, cc.descriptor.ncontacts_max, 37)[0, mask[0]]
  points = records[:, 13:16]+records[:, 25:28]+records[:, 31:34]
  normals = records[:, 16:19]+records[:, 28:31]+records[:, 34:37]
  distances = records[:, 12]+records[:, 9]+records[:, 10]
  first_geom = int(cc.descriptor.geom1[0])
  source_normals *= np.asarray([1 if int(c.geom[0]) == first_geom else -1
                                for c in data.contact[:data.ncon]])[:, None]
  assert np.any(records[:, 25:37] != 0), 'the source-word payload was cleared'
  np.testing.assert_allclose(points, source_points, rtol=0, atol=2e-12)
  np.testing.assert_allclose(normals, source_normals, rtol=0, atol=2e-12)
  np.testing.assert_allclose(distances, source_distances, rtol=0, atol=2e-12)
