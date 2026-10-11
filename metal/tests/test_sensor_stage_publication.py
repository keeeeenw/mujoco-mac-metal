# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Standalone sensor stages publish only their addresses from prepared physics."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api
from test_energy_lifecycle_006 import _native_only
from test_sensor_spatial_016 import _XML as SPATIAL, _states
from test_sensor_remaining_016 import _XML as LIMITS

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native sensor publication')]


def _host(value):
  return value.detach().cpu().numpy().copy()


def _fixture(family, profile, mode='direct', disabled=False):
  extra = ('<velocimeter site="imu"/>' if family == 'spatial' else
           '<jointpos joint="h"/><jointvel joint="h"/>')
  xml = (SPATIAL if family == 'spatial' else LIMITS).replace(
      '</sensor>', extra+'<e_potential/><e_kinetic/></sensor>')
  spec = mujoco.MjSpec.from_string(xml)
  if mode != 'direct':
    for sensor in spec.sensors:
      sensor.nsample, sensor.interp = 8, 1
      sensor.delay = .003 if mode == 'delay' else 0
      if mode == 'interval':
        sensor.interval = [.004, -.001]
  model = spec.compile()
  if disabled:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  if family == 'spatial':
    q, v, _ = _states(model, batch=3)
    q[:, 2] = [.08, .4, .085]
    control = np.empty((3, 0), np.float32)
  else:
    q = np.array([[1.1, .1], [0, 1.2], [0, -.1]])
    v = np.array([[-.5, .7], [.3, -.4], [.2, .1]])
    control = np.array([[.5, .3], [-.2, .8], [.1, -.2]], np.float32)
  q, v = q.astype(np.float32), v.astype(np.float32)
  from mujoco_metal.capacity import model_component_capacity_limits
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile,
                        limits=model_component_capacity_limits(model, 3))
  api.mj_setState(sim, {'ctrl': control})
  refs = []
  for pos, vel, ctrl in zip(q, v, control):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = pos, vel, ctrl
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
    refs.append(data)
  if mode != 'direct':
    # Qualify live interpolation/held values, not only empty reset histories.
    for _ in range(4):
      with pytest.MonkeyPatch.context() as guard:
        _native_only(guard, sim.step)
      for data in refs:
        mujoco.mj_step(model, data)
    for data in refs:
      mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
  return model, sim, refs


@pytest.mark.parametrize('family', ['limits', 'spatial'])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('mode', ['direct', 'delay', 'interval'])
@pytest.mark.parametrize('disabled', [False, True])
def test_actual_public_sensor_stage_matches_pinned_source_without_forward_redispatch(
    family, profile, stage, mode, disabled, monkeypatch):
  model, sim, refs = _fixture(family, profile, mode, disabled)
  record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
  np.testing.assert_array_equal(_host(record.values[5]['status']), 0)
  np.testing.assert_allclose(_host(record.values[5]['qacc']), [d.qacc for d in refs],
                             rtol=2e-4, atol=2e-4)
  sentinel = np.arange(3*model.nsensordata, dtype=np.float32).reshape(3, -1)+20
  sim._sensordata = sim.state._torch.as_tensor(sentinel, device=sim.state._device).clone()
  sim._raw_sensordata = sim._sensordata.clone()
  for data, values in zip(refs, sentinel):
    data.sensordata[:] = values
  for world in (2, 0):
    getattr(mujoco, 'mj_sensor'+stage)(model, refs[world])
  expected = np.stack([d.sensordata.copy() for d in refs])
  before = sim.snapshot()
  borrowed = record.values[1]['dynamics']
  mass = borrowed['mass_blocks' if sim._component_mass_enabled else 'mass_matrix']
  mass_before = _host(mass)
  def forbidden(*args, **kwargs):
    raise AssertionError('standalone sensors must not redispatch forward physics')
  with monkeypatch.context() as guard:
    for name in ('prepare_forward_position', 'prepare_forward_velocity',
                 'prepare_forward_actuation', 'prepare_forward_acceleration',
                 'prepare_forward_constraint', '_acceleration'):
      guard.setattr(sim, name, forbidden)
    result = _native_only(guard, lambda: getattr(api, 'mj_sensor'+stage)(
        sim, record, env_ids=[2, 0]))
  np.testing.assert_allclose(_host(result), expected, rtol=2e-4, atol=2e-4)
  np.testing.assert_allclose(sim.step_sensordata(), expected, rtol=2e-4, atol=2e-4)
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs], rtol=2e-5, atol=2e-6)
  np.testing.assert_array_equal(_host(sim.flg_energypos), [d.flg_energypos for d in refs])
  np.testing.assert_array_equal(_host(sim.flg_energyvel), [d.flg_energyvel for d in refs])
  assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
  np.testing.assert_array_equal(_host(mass), mass_before)
  after = sim.snapshot()
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status', 'precision_words'):
    np.testing.assert_array_equal(getattr(after['device'], name), getattr(before['device'], name))
  np.testing.assert_array_equal(after['native_state']['history'], before['native_state']['history'])
  result.fill_(-1)
  np.testing.assert_allclose(sim.step_sensordata(), expected, rtol=2e-4, atol=2e-4)
  _native_only(monkeypatch, sim.step)
  final = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(after))
  _native_only(monkeypatch, sim.step)
  replay = sim.snapshot()
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status', 'precision_words'):
    np.testing.assert_array_equal(getattr(replay['device'], name), getattr(final['device'], name))
  np.testing.assert_array_equal(replay['sensordata'], final['sensordata'])
