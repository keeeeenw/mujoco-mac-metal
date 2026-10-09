# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Published generalized forces consume the actual accepted multiplier pair."""
import os
import mujoco
import numpy as np
import pytest
from test_primitive_hfield_full_manifold import _model, _LIMITS


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native force coherence')
@pytest.mark.parametrize('solver', [mujoco.mjtSolver.mjSOL_PGS,
                                  mujoco.mjtSolver.mjSOL_CG,
                                  mujoco.mjtSolver.mjSOL_NEWTON])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_contact_cancellation_uses_retained_multipliers_across_mass_routes(solver, profile):
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.constraint_jacobian import PACKED_J_CSR
  model = _model('cylinder', '.25 .15', True)
  model.opt.solver = solver
  q = np.tile(model.qpos0.astype(np.float32), (2, 1)); q[1, 2] = .8
  v = np.tile(np.array([.02, -.01, .03, .04, -.02, .01], np.float32), (2, 1))
  sim = MetalSimulation(model, 2, qpos=q, qvel=v, profile=profile, limits=_LIMITS)
  result = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(result['status'].cpu().numpy(), 0)
  cc = sim._coupled_constraints
  layout = cc._jacobian_layout; nr = cc.descriptor.nr; nv = model.nv
  raw = cc._workspace['workspace_J'].cpu().numpy().reshape(2, -1)[0]
  values = raw[layout.values_offset:layout.values_offset + layout.nnz].astype(float)
  lows = raw[layout.lo_values_offset():layout.lo_values_offset() + layout.nnz].astype(float)
  if layout.mode == PACKED_J_CSR:
    indices = raw.view(np.int32)
    pointers = indices[layout.rowptr_offset:layout.rowptr_offset + nr + 1]
    columns = indices[layout.columns_offset:layout.columns_offset + layout.nnz]
    jhi = np.zeros((nr, nv)); jlo = np.zeros_like(jhi)
    for row in range(nr):
      span = slice(pointers[row], pointers[row + 1])
      jhi[row, columns[span]] = values[span]
      jlo[row, columns[span]] = lows[span]
  else:
    jhi = values.reshape(nr, nv); jlo = lows.reshape(nr, nv)
  # The same effective operator must drive accepted acceleration and force.
  # Terrain witnesses and FK now retain nonzero Jacobian residuals.
  assert np.any(jlo != 0)
  debug = cc._workspace['workspace_debug'].cpu().numpy().reshape(2, -1)[0]
  low_base = (nr * nr + 7 * nr + cc._high_low_workspace_offset
              + 8 * max(nv, 1) + max(nr, 1))
  lamhi = result['lambda'][0].cpu().numpy().astype(float)
  lamlo = debug[low_base:low_base + nr].astype(float)
  if solver != mujoco.mjtSolver.mjSOL_PGS:
    assert np.any(lamlo != 0)
  assert np.count_nonzero(result['contact_mask'][0].cpu().numpy()) > 8
  expected = (jhi + jlo).T @ (lamhi + lamlo)
  actual = result['qfrc_constraint'][0].cpu().numpy().astype(float)
  np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
  np.testing.assert_array_equal(result['qfrc_constraint'][1].cpu().numpy(), 0)
  # Independent source oracle guards against a coherent but incorrect force.
  cpu = mujoco.MjData(model); cpu.qpos[:] = q[0]; cpu.qvel[:] = v[0]
  mujoco.mj_forward(model, cpu)
  np.testing.assert_allclose(actual, cpu.qfrc_constraint, rtol=2e-3, atol=2e-3)
  original_force = result['qfrc_constraint'].cpu().numpy().copy()
  checkpoint = sim.snapshot()
  sim.step(); sim.restore(checkpoint)
  replay = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(replay['qfrc_constraint'].cpu().numpy(),
                                original_force)
