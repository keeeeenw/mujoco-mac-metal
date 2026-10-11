# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Large cached row arenas must not be mistaken for component-only mass."""
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal import MetalSimulation, native_api as api
from mujoco_metal.capacity import model_component_capacity_limits, DENSE_ROW_THRESHOLD
from test_energy_lifecycle_006 import _native_only
from test_sensor_spatial_016 import _XML, _states

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native cached block-row solve')]


def test_admitted_dense_mass_cached_large_row_solve_matches_cpu_without_reassembly(monkeypatch):
  model = mujoco.MjModel.from_xml_string(_XML)
  q, v, _ = _states(model, batch=3)
  q[:, 2] = [.08, .4, .085]
  q, v = q.astype(np.float32), v.astype(np.float32)
  refs = []
  for pos, vel in zip(q, v):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = pos, vel
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
    refs.append(data)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile='integrated_euler_v1',
                        limits=model_component_capacity_limits(model, 3))
  assert sim._coupled_constraints.descriptor.nr > DENSE_ROW_THRESHOLD
  assert not sim._coupled_constraints.descriptor.dense_path
  assert not sim._component_mass_enabled and sim._component_solver is None
  result = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))
  record, solved = result['record'], result['record'].values[5]
  mass = record.values[1]['dynamics']['mass_matrix'].detach().cpu().numpy().copy()
  host = lambda value: value.detach().cpu().numpy().copy()
  np.testing.assert_array_equal(host(solved['status']), 0)
  np.testing.assert_allclose(host(solved['qacc']), [d.qacc for d in refs], rtol=2e-4, atol=2e-4)
  np.testing.assert_allclose(host(solved['qfrc_constraint']), [d.qfrc_constraint for d in refs],
                             rtol=2e-4, atol=2e-4)
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1
  initial_qacc, initial_force = host(solved['qacc']), host(solved['qfrc_constraint'])
  def forbidden(*args, **kwargs):
    raise AssertionError('cached block rows must not redispatch POS or full assembly')
  with monkeypatch.context() as guard:
    guard.setattr(sim._smooth, 'run_position_device', forbidden)
    guard.setattr(sim._coupled_constraints, 'run_device', forbidden)
    qacc, status, _ = _native_only(guard, lambda: api.mj_fwdConstraint(
        sim, record=record, skipsensor=True))
  np.testing.assert_array_equal(host(status), 0)
  np.testing.assert_array_equal(host(qacc), initial_qacc)
  np.testing.assert_array_equal(host(record.values[5]['qfrc_constraint']), initial_force)
  np.testing.assert_array_equal(host(record.values[1]['dynamics']['mass_matrix']), mass)
