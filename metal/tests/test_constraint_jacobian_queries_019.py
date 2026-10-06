# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Prepared-row multiplication, sparse storage, ownership, and native oracles."""
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tests.test_contact_force_query_019 import _prepared


@pytest.mark.parametrize('storage', ['dense', 'packed_dense', 'csr'])
def test_canonical_jacobian_queries_mask_rows_own_results_and_never_materialize(storage, monkeypatch):
  torch = pytest.importorskip('torch')
  from mujoco_metal import native_api as api
  from mujoco_metal import constraint_jacobian as cj
  matrix = torch.tensor([[[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]],
                         [[2., 0., 1.], [0., 5., 0.], [6., 0., 7.]]])
  active = torch.tensor([[1., 0., 1.], [1., 1., 1.]])
  rows = {'active': active, 'J': matrix, 'status': torch.tensor([0, 2], dtype=torch.int32)}
  if storage != 'dense':
    pattern = cj.ConstraintJacobianPattern(
        row_offsets=np.array([0, 2, 3, 5], np.int32),
        columns=np.array([0, 2, 1, 0, 2], np.int32), nv=3, nr=3)
    packed, layout = cj.packed_jacobian_device_storage(torch, 'cpu', 2,
        pattern if storage == 'csr' else None, nr=3, nv=3)
    for world in range(2):
      base = world*layout.stride_words + layout.values_offset
      if storage == 'csr':
        for row in range(3):
          begin, end = pattern.row_offsets[row:row+2]
          packed[base+begin:base+end] = matrix[world, row, pattern.columns[begin:end]]
      else:
        packed[base:base+9] = matrix[world].reshape(-1)
    rows.pop('J')
    rows.update(J_packed=packed, jacobian_layout=layout,
                jacobian_pattern=pattern if storage == 'csr' else None)
  sim, record = _prepared(rows)
  sim._mjmodel = SimpleNamespace(nv=3)
  sim.state._device = torch.device('cpu')
  def forbidden(*args, **kwargs):
    raise AssertionError('query materialized sparse Jacobian')
  monkeypatch.setattr(cj, 'materialize_packed_jacobian_torch', forbidden)
  v = torch.tensor([[.4, -.7, .1], [.2, .3, .6]])
  f = torch.tensor([[1., 99., -2.], [.3, .1, .2]])
  expected_jv = torch.tensor([[.6, 0., 2.1], [0., 0., 0.]])
  expected_jtf = torch.tensor([[-7., 0., -8.], [0., 0., 0.]])
  before = {key: value.clone() for key, value in rows.items() if isinstance(value, torch.Tensor)}
  jv = api.mj_mulJacVec(sim, v, record=record)
  jtf = api.mj_mulJacTVec(sim, f, record=record)
  torch.testing.assert_close(jv, expected_jv)
  torch.testing.assert_close(jtf, expected_jtf)
  # This relation independently checks the transpose and inactive-row rule.
  torch.testing.assert_close((f*jv).sum(-1), (v*jtf).sum(-1))
  for key, tensor in before.items():
    assert torch.equal(rows[key], tensor)
  retained_jv, retained_jtf = jv.clone(), jtf.clone()
  active.zero_()
  assert not api.mj_mulJacVec(sim, v).any()
  assert torch.equal(jv, retained_jv) and torch.equal(jtf, retained_jtf)
  record.qpos.add_(1)
  with pytest.raises(ValueError, match='mutated'):
    api.mj_mulJacVec(sim, v, record=record)


def test_canonical_jacobian_queries_handle_empty_rows_and_reject_bad_vectors():
  torch = pytest.importorskip('torch')
  from mujoco_metal import native_api as api
  sim, record = _prepared({'status': torch.zeros(2, dtype=torch.int32)})
  sim._mjmodel = SimpleNamespace(nv=3)
  sim.state._device = torch.device('cpu')
  assert api.mj_mulJacVec(sim, torch.zeros((2, 3))).shape == (2, 0)
  assert api.mj_mulJacTVec(sim, torch.zeros((2, 0))).shape == (2, 3)
  assert not api.mj_mulJacTVec(sim, torch.zeros((2, 0))).any()
  for vector in (None, torch.zeros((1, 3)), torch.zeros((2, 4)),
                 torch.full((2, 3), float('nan'))):
    with pytest.raises((TypeError, ValueError)):
      api.mj_mulJacVec(sim, vector, record=record)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_native_prepared_jacobian_products_match_pinned_compact_equality_rows(profile, monkeypatch):
  import torch
  from mujoco_metal import native_api as api
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0" jacobian="dense" solver="PGS">
      <flag contact="disable"/></option>
    <worldbody><body><joint name="a" type="slide" axis="1 0 0"/>
      <geom size=".05" mass="1"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="0 1 0"/>
        <geom size=".05" mass="1"/></body></worldbody>
    <equality><joint joint1="a" joint2="b" polycoef=".1 .7 .2 0 0"/></equality>
  </mujoco>''')
  qp = np.array([[.2, .3], [-.1, -.2]], np.float32)
  qv = np.array([[.1, -.2], [.3, .4]], np.float32)
  vectors = np.array([[.4, -.7], [.1, .5]], np.float32)
  forces = np.array([[1.3], [-.8]], np.float32)
  expected_jv, expected_jtf = [], []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qp[world]
    data.qvel[:] = qv[world]
    mujoco.mj_forward(model, data)
    assert data.nefc == 1
    jv, jtf = np.zeros(1), np.zeros(model.nv)
    mujoco.mj_mulJacVec(model, data, jv, vectors[world].astype(np.float64))
    mujoco.mj_mulJacTVec(model, data, jtf, forces[world].astype(np.float64))
    expected_jv.append(jv); expected_jtf.append(jtf)
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv, profile=profile)
  result = api.mj_forward(sim, skipsensor=True)
  record = result['record']
  def forbidden(*args, **kwargs):
    raise AssertionError('Jacobian query ran CPU physics/assembly or a solver')
  for name in ('mj_forward', 'mj_mulJacVec', 'mj_mulJacTVec'):
    monkeypatch.setattr(mujoco, name, forbidden)
  monkeypatch.setattr(sim._coupled_constraints, 'run_velocity_device', forbidden)
  actual_jv = api.mj_mulJacVec(sim, vectors, record=record)
  # The backend has reserved canonical rows. Here the only equality is active.
  nr = actual_jv.shape[1]
  force = np.zeros((2, nr), np.float32)
  force[:, :1] = forces
  actual_jtf = api.mj_mulJacTVec(sim, force, record=record)
  np.testing.assert_allclose(actual_jv[:, :1].cpu().numpy(), expected_jv,
                             atol=3e-6, rtol=3e-5)
  np.testing.assert_allclose(actual_jtf.cpu().numpy(), expected_jtf, atol=3e-6, rtol=3e-5)
  saved = actual_jv.clone(), actual_jtf.clone()
  sim.state._qpos.add_(.1)
  with pytest.raises(ValueError, match='mutated'):
    api.mj_mulJacVec(sim, vectors, record=record)
  assert torch.equal(actual_jv, saved[0]) and torch.equal(actual_jtf, saved[1])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('cone', ['pyramidal', 'elliptic'])
@pytest.mark.parametrize('condim', [1, 3, 4, 6])
def test_native_contact_jacobian_products_match_source_rows_and_inactive_world(cone, condim, monkeypatch):
  import torch
  from mujoco_metal import native_api as api
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 -9.81" jacobian="dense" solver="PGS" cone="{cone}"/>
    <worldbody><geom type="plane" size="2 2 .1" condim="{condim}"/>
      <body pos=".2 -.1 .095"><freejoint/>
        <geom type="sphere" size=".1" mass="1" condim="{condim}"
          friction=".7 .1 .05"/></body></worldbody>
  </mujoco>''')
  qp = np.repeat(model.qpos0[None], 2, axis=0).astype(np.float32)
  qp[1, 2] = .3
  qv = np.array([[.1, -.2, -.05, .3, .1, -.2], [.2, .1, .3, -.1, .2, .1]], np.float32)
  vector = np.array([[.4, -.7, .1, .6, -.3, .2], [.1, .5, .2, .3, .7, -.1]], np.float32)
  expected_jv, expected_jtf = [], []
  compact_force = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qp[world]; data.qvel[:] = qv[world]
    mujoco.mj_forward(model, data)
    assert data.ncon == (1 if world == 0 else 0)
    force = np.arange(1, data.nefc+1, dtype=np.float64)*.3
    jv, jtf = np.zeros(data.nefc), np.zeros(model.nv)
    mujoco.mj_mulJacVec(model, data, jv, vector[world].astype(np.float64))
    mujoco.mj_mulJacTVec(model, data, jtf, force)
    expected_jv.append(jv); expected_jtf.append(jtf); compact_force.append(force)
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv,
                        profile='integrated_euler_v1')
  result = api.mj_forward(sim, skipsensor=True)
  rows = result['constraint']['canonical_rows']
  live_rows = rows['active'].cpu().numpy() > .5
  # Independent source order is the candidate's contact-row block, not a
  # selected subset of source columns or a reordered diagnostic matrix.
  descriptor = sim._coupled_constraints.descriptor
  assert descriptor.ncontacts_max == 1
  block = np.asarray(descriptor.contact_condim_packed).reshape(-1, 3)[0]
  count = 1 if condim == 1 else (condim if cone == 'elliptic' else 2*(condim-1))
  first = int(descriptor.nr_joint)+int(block[1])
  np.testing.assert_array_equal(np.flatnonzero(live_rows[0]), np.arange(first, first+count))
  assert not live_rows[1].any()
  force = np.full(live_rows.shape, 99., np.float32)
  force[0, first:first+count] = compact_force[0]
  def forbidden(*args, **kwargs):
    raise AssertionError('query ran physics/assembly')
  monkeypatch.setattr(sim._coupled_constraints, 'run_velocity_device', forbidden)
  for name in ('mj_forward', 'mj_mulJacVec', 'mj_mulJacTVec'):
    monkeypatch.setattr(mujoco, name, forbidden)
  actual_jv = api.mj_mulJacVec(sim, vector, record=result['record'])
  actual_jtf = api.mj_mulJacTVec(sim, force, record=result['record'])
  np.testing.assert_allclose(actual_jv[0, first:first+count].cpu().numpy(), expected_jv[0],
                             atol=4e-6, rtol=4e-5)
  assert not actual_jv[1].any()
  np.testing.assert_allclose(actual_jtf.cpu().numpy(), expected_jtf, atol=4e-6, rtol=4e-5)
  torch.testing.assert_close((actual_jv*torch.as_tensor(force, device=sim.device)).sum(-1),
      (actual_jtf*torch.as_tensor(vector, device=sim.device)).sum(-1), atol=4e-6, rtol=4e-5)
