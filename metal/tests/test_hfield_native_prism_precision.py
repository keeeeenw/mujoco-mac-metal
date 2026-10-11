# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native prism EPA words must reach actual contact rows without projection loss."""
import os

import mujoco
import numpy as np
import pytest

from test_primitive_hfield_full_manifold import _LIMITS, _model
from test_contact_operator_precision import _effective_rows


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('shape,size', [('cylinder', '.25 .15'), ('box', '.3 .22 .2')])
@pytest.mark.parametrize('reversed_order', [False, True])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_native_prism_contact_retains_source_witness_words(shape, size, reversed_order, profile, monkeypatch):
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
  source_jacobian = np.asarray(data.efc_J).reshape(data.nefc, model.nv).copy()
  source_aref = np.asarray(data.efc_aref).copy()
  source_regularizer = np.asarray(data.efc_R).copy()
  sim = MetalSimulation(model, 2, qpos=qpos, qvel=qvel,
                        profile=profile, limits=_LIMITS)

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
  # Accurate collision words must reach the actual solve, including the
  # native CCD route and an adjacent contact-free world. A producer-only
  # witness check would miss the host's former alternate-CCD-only gate.
  offsets = np.asarray(cc.descriptor.contact_condim_packed).reshape(-1, 3)
  rows, source_rows = [], []
  for slot, contact in zip(np.flatnonzero(mask[0]), data.contact[:data.ncon]):
    base = int(cc.descriptor.nr_joint) + int(offsets[slot, 1])
    rows.extend(range(base, base + 3))
    source_rows.extend(range(contact.efc_address, contact.efc_address + 3))
  np.testing.assert_allclose(_effective_rows(cc)[rows], source_jacobian[source_rows],
                             rtol=2e-12, atol=2e-12)
  np.testing.assert_array_equal(_effective_rows(cc, 1), 0)
  aref = host(system['ar'])[0] + host(system['ar_low'])[0]
  np.testing.assert_allclose(aref[rows], source_aref[source_rows],
                             rtol=2e-11, atol=2e-10)
  nr = cc.descriptor.nr
  debug = host(cc._workspace['workspace_debug']).reshape(2, -1)
  low = debug[:, cc._regularizer_low_debug_offset:cc._regularizer_low_debug_offset + nr]
  regularizer = host(system['R'])[0] + low[0]
  np.testing.assert_allclose(regularizer[rows], source_regularizer[source_rows],
                             rtol=2e-12, atol=2e-12)
  np.testing.assert_array_equal(low[1], 0)
