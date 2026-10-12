# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Raw failure rollback uses admitted, reusable simulation-owned storage."""
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal.capacity import estimate_capacity


def _model(sensors=True):
  sensor = '<sensor><jointpos joint="j"/><jointvel joint="j"/></sensor>' if sensors else ''
  return mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
    <worldbody><body><joint name="j" type="slide"/><geom size=".1" mass="1"/>
    </body></worldbody>{sensor}</mujoco>''')


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('storage', ['dense', 'block_sparse'])
@pytest.mark.parametrize('sensors', [False, True])
def test_raw_rollback_exact_named_capacity_for_actual_model(batch, storage, sensors):
  model = _model(sensors)
  sizes = dict(estimate_capacity(model, batch, 0, 0, 0, mass_storage=storage).memory_breakdown)
  assert sizes['simulation.forward_stage_raw_sensor_rollback'] == 4*batch*model.nsensordata


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native rollback allocation')
@pytest.mark.parametrize('profile,storage', [('integrated_euler_v1', 'dense'),
                                          ('integrated_scalable_v1', 'block_sparse')])
def test_actual_rollback_storage_is_charged_and_reused_across_native_updates(profile, storage, monkeypatch):
  from mujoco_metal import MetalSimulation
  from test_energy_lifecycle_006 import _native_only
  from test_sensor_stage_publication import _host
  model = _model()
  sim = MetalSimulation(model, 3, qvel=np.asarray([[.1], [.2], [.3]], np.float32), profile=profile)
  buffer = sim._forward_stage_raw_sensor_rollback
  pointer = buffer.data_ptr()
  inventory = dict(estimate_capacity(model, 3, 0, 0, 0, mass_storage=storage).memory_breakdown)
  assert inventory['simulation.forward_stage_raw_sensor_rollback'] == buffer.numel()*buffer.element_size()
  for _ in range(8):
    previous = (_host(sim._raw_sensordata) if sim._raw_sensordata is not None
                else sim.step_sensordata() if sim._sensordata is not None
                else np.zeros((3, model.nsensordata), np.float32))
    _native_only(monkeypatch, sim.step)
    assert sim._forward_stage_raw_sensor_rollback is buffer and buffer.data_ptr() == pointer
    np.testing.assert_array_equal(_host(buffer), previous)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native derivative rollback ownership')
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_actual_transition_derivative_preserves_raw_rollback_owner_and_next_trajectory(profile, monkeypatch):
  from mujoco_metal import MetalSimulation, native_api as api
  from test_energy_lifecycle_006 import _native_only
  from test_sensor_stage_publication import _host
  from test_lifecycle_commit_rollback import _same
  model = _model()
  sim = MetalSimulation(model, 3, qvel=np.asarray([[.1], [.2], [.3]], np.float32), profile=profile)
  _native_only(monkeypatch, lambda: sim.step(3))
  before = sim.snapshot()
  owner = sim._forward_stage_raw_sensor_rollback
  previous = _host(owner)
  generation = sim.state.generation
  result = _native_only(monkeypatch, lambda: api.mjd_transitionFD(sim, eps=1e-4, flg_centered=True))
  np.testing.assert_array_equal(_host(result['status']), 0)
  expected_a = np.broadcast_to([[1., float(model.opt.timestep)], [0., 1.]], (3, 2, 2))
  np.testing.assert_allclose(_host(result['A']), expected_a, rtol=2e-4, atol=2e-4)
  np.testing.assert_allclose(_host(result['C']), np.broadcast_to(np.eye(2), (3, 2, 2)),
                             rtol=2e-4, atol=2e-4)
  assert sim._forward_stage_raw_sensor_rollback is owner
  assert sim.state.generation == generation
  np.testing.assert_array_equal(_host(owner), previous)
  _same(sim.snapshot(), before)
  _native_only(monkeypatch, lambda: sim.step(2))
  expected = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(before))
  _native_only(monkeypatch, lambda: sim.step(2))
  _same(sim.snapshot(), expected)
