# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Recovery assembly retains healthy mixed rows and their borrowed workspaces."""
import os

import mujoco
import numpy as np
import pytest

from test_cached_mixed_row_ownership import _mixed_model, _mixed_states


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('selected', [(), (0,), (1,)])
def test_mixed_recovery_preserves_every_unselected_constraint_record(profile, selected):
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _mixed_model()
  if profile == 'integrated_scalable_v1':
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  qpos, qvel = _mixed_states(model)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel, profile=profile)
  initial_qacc, status, _ = sim._acceleration(sim.state._qpos, sim.state._qvel)
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
  cc = sim._coupled_constraints
  d = cc.descriptor
  assert cc._jacobian_layout.mode == (profile == 'integrated_scalable_v1')
  rows = cc._assembly_views(include_optimizer_outputs=False)
  active = rows['active'].cpu().numpy()
  for start, count in ((0, d.n_eq_rows),
                       (d.ten_base, d.ten_friction_rows + d.ten_limit_rows),
                       (d.flex_contact_base, d.n_flex_contact_rows)):
    assert np.any(active[:, start:start + count] > .5)
  assert np.linalg.norm(cc._workspace['out_force'].cpu().numpy()) > 1
  for world in range(2):
    oracle = mujoco.MjData(model)
    oracle.qpos[:], oracle.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, oracle)
    np.testing.assert_allclose(initial_qacc[world].cpu().numpy(), oracle.qacc,
                               rtol=3e-3, atol=3e-3)

  names = ('workspace_debug', 'workspace_J', 'out_status', 'out_acc', 'out_force',
           'out_diagnostics', 'out_joint_force',
           'position_current_qvel', 'position_current_cvel', 'position_current_cdof',
           'position_current_cdof_dot', 'position_current_body_pos',
           'position_current_body_quat', 'position_current_root_com',
           'position_current_eq_active', 'position_assembly_context',
           'position_surface_velocity')
  # Integer metadata shares float scratch, so compare raw words, including
  # NaN payloads and signed zero; floating equal_nan is insufficient here.
  def words(name):
    raw = cc._workspace[name].cpu().numpy().copy().view(np.uint32)
    if name == 'out_acc':
      return raw.reshape(2, 2, model.nv).transpose(1, 0, 2).reshape(2, -1)
    return raw.reshape(2, -1)
  before = {name: words(name) for name in names}
  qp = torch.tensor(qpos, device='mps')
  qv = torch.tensor(qvel + np.float32(.013), device='mps')
  mask = torch.zeros(2, dtype=torch.bool, device='mps')
  mask[list(selected)] = True
  healthy = sorted(set(range(2)) - set(selected))
  qp[healthy] = float('nan')
  qv[healthy] = float('nan')
  _, status, _ = sim._acceleration(qp, qv, forward_mask=mask)
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
  for name, expected in before.items():
    actual = words(name)
    np.testing.assert_array_equal(actual[healthy], expected[healthy], err_msg=name)
  for world in selected:
    oracle = mujoco.MjData(model)
    oracle.qpos[:], oracle.qvel[:] = qpos[world], qvel[world] + np.float32(.013)
    mujoco.mj_forward(model, oracle)
    np.testing.assert_allclose(cc._workspace['out_acc'][:2 * model.nv]
                               .reshape(2, model.nv)[world].cpu().numpy(),
                               oracle.qacc, rtol=3e-3, atol=3e-3)


def test_masked_sparse_publication_preserves_healthy_values_and_error_words():
  import torch
  from test_constraint_jacobian_pattern_017 import _model
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.constraint_jacobian import (
      compile_constraint_jacobian_pattern, packed_jacobian_device_storage,
      packed_jacobian_scatter_rows_torch)
  model = _model()
  desc = lower_coupled_constraints(model)
  pattern = compile_constraint_jacobian_pattern(model, desc)
  storage, layout = packed_jacobian_device_storage(torch, 'cpu', 2, pattern,
                                                   paired=True)
  for world in range(2):
    base = world * layout.stride_words
    storage[base + layout.values_offset:base + layout.values_offset + layout.nnz] = 3.1
    storage[base + layout.lo_values_offset():base + layout.lo_values_offset() + layout.nnz] = 2**-20
  before = storage.numpy().copy().view(np.uint32)
  values = torch.ones((2, desc.nr, desc.nv), dtype=torch.float32)
  values[0] = float('nan')
  packed_jacobian_scatter_rows_torch(
      torch, storage, 2, layout, pattern, values, 0,
      row_count=desc.nr, world_mask=torch.tensor([0, 1], dtype=torch.int32))
  np.testing.assert_array_equal(storage.numpy().view(np.uint32)[:layout.stride_words],
                                before[:layout.stride_words])
  selected = layout.stride_words
  assert int(storage.view(torch.int32)[selected + 10]) == 1
  np.testing.assert_array_equal(storage[selected + layout.values_offset:
                                       selected + layout.values_offset + layout.nnz], 1)
  np.testing.assert_array_equal(storage[selected + layout.lo_values_offset():
                                       selected + layout.lo_values_offset() + layout.nnz], 2**-20)
