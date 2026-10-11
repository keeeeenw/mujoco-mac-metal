# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Raw, stored and status sensor ownership through real simulation lifecycle."""
import copy
import os
from dataclasses import replace
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityOverflow
from test_sensor_stage_publication import _fixture, _host
from test_energy_lifecycle_006 import _native_only

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native sensor lifecycle')]


def _prepared(monkeypatch, mode='delay'):
  _, sim, refs = _fixture('limits', 'integrated_scalable_v1', mode)
  # Real timed stepping populates different raw versus delayed samples.
  _native_only(monkeypatch, sim.step)
  checkpoint = sim.snapshot()
  assert checkpoint['sensor_diagnostics_version'] == 1
  assert checkpoint['raw_sensordata'] is not None
  if mode == 'delay':
    assert not np.array_equal(checkpoint['raw_sensordata'], checkpoint['sensordata'])
  return sim, checkpoint, refs


def _assert_same(sim, expected):
  actual = sim.snapshot()
  for name in ('sensordata', 'raw_sensordata', 'sensor_plugin_status'):
    if expected[name] is None:
      assert actual[name] is None
    else:
      np.testing.assert_array_equal(actual[name], expected[name])
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status', 'precision_words'):
    np.testing.assert_array_equal(getattr(actual['device'], name), getattr(expected['device'], name))
  for name in expected['native_state']:
    np.testing.assert_array_equal(actual['native_state'][name], expected['native_state'][name])


@pytest.mark.parametrize('mode', ['direct', 'delay', 'interval'])
@pytest.mark.parametrize('ids', [None, [2, 0]])
def test_raw_diagnostic_checkpoint_restores_selected_values_and_exact_following_updates(mode, ids, monkeypatch):
  sim, saved, _ = _prepared(monkeypatch, mode)
  _native_only(monkeypatch, lambda: sim.step(3))
  later = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(saved, env_ids=ids))
  actual = sim.snapshot()
  for name in ('sensordata', 'raw_sensordata', 'sensor_plugin_status'):
    expected = saved[name].copy() if ids is None else later[name].copy()
    if ids is not None:
      expected[ids] = saved[name][ids]
    np.testing.assert_array_equal(actual[name], expected)
  checkpoint = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.step(3))
  expected = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(checkpoint))
  _native_only(monkeypatch, lambda: sim.step(3))
  _assert_same(sim, expected)


@pytest.mark.parametrize('ids', [None, [1]])
@pytest.mark.parametrize('keyframe', [False, True])
def test_cold_reset_clears_raw_status_and_preserves_unselected_samples(ids, keyframe, monkeypatch):
  import mujoco
  from mujoco_metal import MetalSimulation
  spec = mujoco.MjSpec.from_string('''<mujoco><option gravity="0 0 0"/>
    <worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
    <sensor><jointpos joint="j"/><jointvel joint="j"/></sensor>
    <keyframe><key qpos=".2" qvel=".3"/></keyframe></mujoco>''')
  model = spec.compile()
  sim = MetalSimulation(model, 3, qpos=np.asarray([[.4], [.5], [.6]], np.float32),
      qvel=np.asarray([[.7], [.8], [.9]], np.float32), profile='integrated_scalable_v1')
  _native_only(monkeypatch, sim.step)
  sim._sensor_plugin_status[:] = sim.state._torch.as_tensor([1, 2, 3], device=sim.state._device)
  old = sim.snapshot()
  if keyframe:
    _native_only(monkeypatch, lambda: sim.reset_to_keyframe(0, env_ids=ids))
  else:
    _native_only(monkeypatch, lambda: sim.reset(env_ids=ids))
  selected = list(range(3)) if ids is None else ids
  after = sim.snapshot()
  for name in ('sensordata', 'raw_sensordata', 'sensor_plugin_status'):
    expected = old[name].copy()
    expected[selected] = 0
    np.testing.assert_array_equal(after[name], expected)
  # Follow the reset with source-oracle motion, not only buffer assertions.
  refs = [mujoco.MjData(model) for _ in selected]
  for data in refs:
    if keyframe:
      mujoco.mj_resetDataKeyframe(model, data, 0)
  _native_only(monkeypatch, sim.step)
  for world, data in zip(selected, refs):
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(_host(sim.state._qpos)[world], data.qpos, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(sim.step_sensordata()[world], data.sensordata, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('bad', ['raw_shape', 'raw_nan', 'raw_overflow', 'status_shape',
    'status_negative', 'status_float', 'status_overflow', 'missing_raw', 'missing_status', 'version'])
def test_bad_diagnostics_reject_before_any_restore_callback_or_mutation(bad, monkeypatch):
  sim, saved, _ = _prepared(monkeypatch)
  broken = copy.deepcopy(saved)
  if bad == 'raw_shape': broken['raw_sensordata'] = np.zeros((2, 1))
  elif bad == 'raw_nan': broken['raw_sensordata'][0, 0] = np.nan
  elif bad == 'raw_overflow': broken['raw_sensordata'] = np.full_like(saved['raw_sensordata'], 1e300, dtype=np.float64)
  elif bad == 'status_shape': broken['sensor_plugin_status'] = np.zeros((3, 1), np.int32)
  elif bad == 'status_negative': broken['sensor_plugin_status'][0] = -1
  elif bad == 'status_float': broken['sensor_plugin_status'] = np.zeros(3, np.float32)
  elif bad == 'status_overflow': broken['sensor_plugin_status'] = np.full(3, 2**32, np.uint64)
  elif bad == 'missing_raw': del broken['raw_sensordata']
  elif bad == 'missing_status': del broken['sensor_plugin_status']
  else: broken['sensor_diagnostics_version'] = True
  from mujoco_metal import simulation
  def forbidden(*args, **kwargs):
    raise AssertionError('bad payload reached extension callback')
  monkeypatch.setattr(simulation, '_preflight_plugin_restores', forbidden)
  with pytest.raises(ValueError):
    _native_only(monkeypatch, lambda: sim.restore(broken, env_ids=[2, 0]))
  _assert_same(sim, saved)


def test_diagnostic_staging_budget_rejects_before_physical_restore(monkeypatch):
  sim, saved, _ = _prepared(monkeypatch)
  sim.limits = replace(sim.limits, memory_budget_bytes=1)
  owner = type(sim.state)
  original = owner.restore
  def forbid_live_commit(state, *args, **kwargs):
    if state is sim.state:
      raise AssertionError('budget failure reached physical restore')
    # validate_snapshot deliberately executes restore on a host-only probe.
    return original(state, *args, **kwargs)
  monkeypatch.setattr(owner, 'restore', forbid_live_commit)
  with pytest.raises(CapacityOverflow):
    _native_only(monkeypatch, lambda: sim.restore(saved))
  _assert_same(sim, saved)


@pytest.mark.parametrize('ids', [None, [1]])
def test_legacy_checkpoint_uses_saved_samples_instead_of_future_raw_data(ids, monkeypatch):
  sim, saved, _ = _prepared(monkeypatch)
  legacy = copy.deepcopy(saved)
  for name in ('sensor_diagnostics_version', 'raw_sensordata', 'sensor_plugin_status'):
    del legacy[name]
  _native_only(monkeypatch, lambda: sim.step(3))
  later = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(legacy, env_ids=ids))
  selected = list(range(3)) if ids is None else ids
  expected = later['raw_sensordata'].copy()
  expected[selected] = saved['sensordata'][selected]
  np.testing.assert_array_equal(_host(sim._raw_sensordata), expected)
  np.testing.assert_array_equal(_host(sim._sensor_plugin_status)[selected], 0)


def test_full_environment_copy_includes_distinct_raw_and_diagnostic_state(monkeypatch):
  sim, _, _ = _prepared(monkeypatch)
  sim._sensor_plugin_status[0] = 3
  before = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.copy_environment(0, 2))
  after = sim.snapshot()
  for name in ('sensordata', 'raw_sensordata', 'sensor_plugin_status'):
    expected = before[name].copy()
    expected[2] = before[name][0]
    np.testing.assert_array_equal(after[name], expected)


@pytest.mark.parametrize('ids', [None, [1]])
def test_lazy_checkpoint_restores_absent_raw_storage_without_retaining_future_rows(ids, monkeypatch):
  _, sim, _ = _fixture('limits', 'integrated_scalable_v1', 'direct')
  untouched = sim.snapshot()
  assert untouched['raw_sensordata'] is None
  _native_only(monkeypatch, lambda: sim.step(2))
  later = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(untouched, env_ids=ids))
  if ids is None:
    assert sim._raw_sensordata is None
  else:
    expected = later['raw_sensordata'].copy()
    expected[ids] = 0
    np.testing.assert_array_equal(_host(sim._raw_sensordata), expected)


@pytest.mark.parametrize('ids', [None, [2, 0]])
def test_restored_diagnostic_bindings_are_mutable_outside_inference_context(ids, monkeypatch):
  sim, saved, _ = _prepared(monkeypatch)
  _native_only(monkeypatch, lambda: sim.step(2))
  with sim.state._torch.inference_mode():
    _native_only(monkeypatch, lambda: sim.restore(saved, env_ids=ids))
  for value in (sim._raw_sensordata, sim._sensor_plugin_status):
    assert not sim.state._torch.is_inference(value)
    value.copy_(value.clone())
  _native_only(monkeypatch, sim.step)


@pytest.mark.parametrize('ids', [None, [2, 0]])
def test_failure_after_actual_physical_restore_keeps_all_previous_sensor_and_physics_state(ids, monkeypatch):
  sim, saved, _ = _prepared(monkeypatch)
  _native_only(monkeypatch, lambda: sim.step(3))
  before = sim.snapshot()
  owner, original = type(sim.state), type(sim.state).restore
  def fail_after_commit(state, *args, **kwargs):
    result = original(state, *args, **kwargs)
    if state is sim.state:
      raise RuntimeError('after actual physical restore')
    return result
  monkeypatch.setattr(owner, 'restore', fail_after_commit)
  with pytest.raises(RuntimeError, match='after actual physical restore'):
    _native_only(monkeypatch, lambda: sim.restore(saved, env_ids=ids))
  _assert_same(sim, before)


@pytest.mark.parametrize('ids', [None, [2, 0]])
@pytest.mark.parametrize('coherent', [True, False])
def test_late_diagnostic_publication_failure_preserves_old_storage_and_record_coherence(ids, coherent, monkeypatch):
  from mujoco_metal import native_api as api, sensor_lifecycle
  sim, saved, _ = _prepared(monkeypatch)
  _native_only(monkeypatch, lambda: sim.step(3))
  record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
  if not coherent:
    sim.state._qvel.add_(.01)
  old_raw, old_status = sim._raw_sensordata, sim._sensor_plugin_status
  before = sim.snapshot()
  original = sensor_lifecycle.publish_sensor_diagnostics
  def fail_after_binding(owner, values):
    original(owner, values)
    assert owner._raw_sensordata is not old_raw
    raise RuntimeError('after diagnostic binding publication')
  monkeypatch.setattr(sensor_lifecycle, 'publish_sensor_diagnostics', fail_after_binding)
  with pytest.raises(RuntimeError, match='after diagnostic binding publication'):
    _native_only(monkeypatch, lambda: sim.restore(saved, env_ids=ids))
  assert sim._raw_sensordata is old_raw and sim._sensor_plugin_status is old_status
  _assert_same(sim, before)
  if coherent:
    assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
  else:
    with pytest.raises(ValueError, match='mutated|stale'):
      sim.validate_forward_stage_record(record, 'CONSTRAINT')
