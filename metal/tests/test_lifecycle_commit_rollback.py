# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual reset/keyframe/copy failures preserve the complete prepared owner."""
from dataclasses import fields, is_dataclass, replace
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api, sensor_lifecycle
from mujoco_metal.capacity import CapacityOverflow
from mujoco_metal.extensions import default_registry, PluginType
from test_energy_lifecycle_006 import _native_only
from test_sensor_stage_plugin_lifecycle import CounterSensor
from test_sensor_stage_publication import _host

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native lifecycle commit rollback')]


def _same(actual, expected):
  if isinstance(expected, np.ndarray):
    np.testing.assert_array_equal(actual, expected)
  elif is_dataclass(expected):
    assert type(actual) is type(expected)
    for field in fields(expected):
      _same(getattr(actual, field.name), getattr(expected, field.name))
  elif isinstance(expected, dict):
    assert actual.keys() == expected.keys()
    for name in expected:
      _same(actual[name], expected[name])
  elif isinstance(expected, (tuple, list)):
    assert len(actual) == len(expected)
    for item, reference in zip(actual, expected):
      _same(item, reference)
  else:
    assert actual == expected


@pytest.fixture
def prepared(monkeypatch):
  name = 'lifecycle_counter'
  default_registry.register(CounterSensor(name))
  try:
    spec = mujoco.MjSpec.from_string(f'''<mujoco>
      <option gravity="0 0 0" timestep=".002"/>
      <worldbody><body><joint name="j" type="slide" range="-1 1"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
      <actuator><general joint="j" dyntype="filter" dynprm=".03"/></actuator>
      <sensor><jointpos joint="j"/><jointvel joint="j"/>
      <user name="{name}" dim="1" needstage="pos"/></sensor>
      <keyframe><key qpos=".2" qvel=".3" act=".1" ctrl=".4" time=".25"/>
      </keyframe></mujoco>''')
    for sensor in list(spec.sensors)[:2]:
      sensor.nsample, sensor.delay, sensor.interp = 8, .003, 1
    model = spec.compile()
    sim = MetalSimulation(model, 3, qpos=np.asarray([[1.1], [.3], [-1.1]], np.float32),
        qvel=np.asarray([[.2], [-.4], [.1]], np.float32), profile='integrated_scalable_v1')
    api.mj_setState(sim, {'ctrl': np.asarray([[.5], [-.2], [.3]], np.float32)})
    _native_only(monkeypatch, lambda: sim.step(4))
    assert np.all(sim.snapshot()['plugins'][('sensor', name)] > 0)
    yield sim
  finally:
    default_registry.unregister(name, PluginType.SENSOR)


def _operation(sim, operation, ids):
  if operation == 'reset':
    return sim.reset(env_ids=ids)
  if operation == 'keyframe':
    return sim.reset_to_keyframe(0, env_ids=ids)
  return sim.copy_environment(0, 2)


@pytest.mark.parametrize('operation', ['reset', 'keyframe', 'copy'])
@pytest.mark.parametrize('point', ['physical', 'diagnostics'])
@pytest.mark.parametrize('ids', [None, [2, 0]])
@pytest.mark.parametrize('coherent', [True, False])
def test_failure_after_actual_commit_rolls_back_all_state_bindings_adapters_and_records(
    prepared, operation, point, ids, coherent, monkeypatch):
  sim = prepared
  record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
  if not coherent:
    sim.state._qvel.add_(.01)
  before = sim.snapshot()
  generation = sim.state.generation
  owners = ((sim, ('_sensordata', '_raw_sensordata', '_sensor_plugin_status',
                   '_energy', '_energy_computed', '_control')),
            (sim.state, ('_qpos', '_qvel', '_qacc', '_time', '_status', '_history')))
  bindings = [(owner, name, getattr(owner, name)) for owner, names in owners for name in names]
  mass = record.values[1]['dynamics']['mass_blocks']
  borrowed_mass = _host(mass)
  calls = []
  with monkeypatch.context() as guard:
    if point == 'physical':
      owner = type(sim.state)
      method = 'copy_environment' if operation == 'copy' else 'reset'
      original = getattr(owner, method)
      def fail_after_actual_state_commit(state, *args, **kwargs):
        result = original(state, *args, **kwargs)
        if state is sim.state:
          calls.append(True)
          raise RuntimeError('actual lifecycle commit fault')
        return result
      guard.setattr(owner, method, fail_after_actual_state_commit)
    else:
      method = 'copy_sensor_diagnostic_row' if operation == 'copy' else 'clear_sensor_diagnostics'
      original = getattr(sensor_lifecycle, method)
      def fail_after_actual_diagnostic_commit(owner, *args, **kwargs):
        original(owner, *args, **kwargs)
        calls.append(True)
        raise RuntimeError('actual lifecycle commit fault')
      guard.setattr(sensor_lifecycle, method, fail_after_actual_diagnostic_commit)
    with pytest.raises(RuntimeError, match='actual lifecycle commit fault'):
      _native_only(guard, lambda: _operation(sim, operation, ids))
  assert calls == [True]
  _same(sim.snapshot(), before)
  assert sim.state.generation == generation
  assert all(getattr(owner, name) is value for owner, name, value in bindings)
  np.testing.assert_array_equal(_host(mass), borrowed_mass)
  if coherent:
    assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
  else:
    with pytest.raises(ValueError, match='mutated|stale'):
      sim.validate_forward_stage_record(record, 'CONSTRAINT')
  # The actual failed operation must not alter subsequent accepted motion or
  # canonical history/adapters, including unselected worlds.
  _native_only(monkeypatch, lambda: sim.step(2))
  actual = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(before))
  _native_only(monkeypatch, lambda: sim.step(2))
  _same(sim.snapshot(), actual)


@pytest.mark.parametrize('operation', ['reset', 'keyframe', 'copy'])
def test_lifecycle_rollback_budget_rejects_before_any_adapter_or_physical_commit(
    prepared, operation, monkeypatch):
  from mujoco_metal import simulation
  sim = prepared
  sim.limits = replace(sim.limits, memory_budget_bytes=1)
  before = sim.snapshot()
  with monkeypatch.context() as guard:
    def forbidden(*args, **kwargs):
      raise AssertionError('memory rejection must precede adapter or physical commit')
    guard.setattr(simulation, '_reset_plugins_transactionally', forbidden)
    guard.setattr(simulation, '_preflight_plugin_restores', forbidden)
    guard.setattr(simulation, '_apply_plugin_restores', forbidden)
    guard.setattr(type(sim.state), 'reset', forbidden)
    guard.setattr(type(sim.state), 'copy_environment', forbidden)
    with pytest.raises(CapacityOverflow, match='lifecycle rollback'):
      _native_only(guard, lambda: _operation(sim, operation, [2, 0]))
  _same(sim.snapshot(), before)
