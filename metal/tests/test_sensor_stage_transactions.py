# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Real sensor execution must preserve the prepared owner after failed publication."""
from dataclasses import replace
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api
from mujoco_metal import sensor_stages
from mujoco_metal.capacity import CapacityOverflow
from test_energy_lifecycle_006 import _native_only
from test_sensor_stage_publication import _fixture, _host

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native sensor transactions')]


def _prepared():
  model, sim, refs = _fixture('limits', 'integrated_scalable_v1')
  record = _native_only_prepare(sim)
  values = sim.state._torch.arange(3*model.nsensordata, device=sim.state._device,
                                  dtype=sim.state._torch.float32).reshape(3, -1)+20
  sim._sensordata, sim._raw_sensordata = values.clone(), values.clone()
  return model, sim, record, refs


def _native_only_prepare(sim):
  with pytest.MonkeyPatch.context() as guard:
    return _native_only(guard, lambda: api.mj_forward(sim, skipsensor=True))['record']


def _assert_preserved(sim, record, before, bindings):
  after = sim.snapshot()
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status', 'precision_words'):
    np.testing.assert_array_equal(getattr(after['device'], name), getattr(before['device'], name))
  for name in ('sensordata', 'energy', 'energy_computed'):
    np.testing.assert_array_equal(after[name], before[name])
  np.testing.assert_array_equal(after['native_state']['history'], before['native_state']['history'])
  assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
  assert all(getattr(sim, name) is value for name, value in bindings.items())


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('failure', ['evaluation', 'publication'])
def test_failure_after_actual_evaluation_or_binding_commit_restores_all_owners(
    stage, failure, monkeypatch):
  _, sim, record, _ = _prepared()
  before = sim.snapshot()
  bindings = {name: getattr(sim, name) for name in sensor_stages._FIELDS}
  mass = record.values[1]['dynamics']['mass_blocks']
  mass_before, raw_before = _host(mass), _host(sim._raw_sensordata)
  original = getattr(sensor_stages, '_evaluate_stage' if failure == 'evaluation' else '_publish')
  calls = []
  def fail_after_real_operation(*args, **kwargs):
    result = original(*args, **kwargs)
    calls.append(True)
    raise RuntimeError('actual sensor publication fault')
  monkeypatch.setattr(sensor_stages, '_evaluate_stage' if failure == 'evaluation' else '_publish',
                      fail_after_real_operation)
  with pytest.raises(RuntimeError, match='actual sensor publication fault'):
    _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record, env_ids=[2, 0]))
  assert calls == [True]
  _assert_preserved(sim, record, before, bindings)
  np.testing.assert_array_equal(_host(mass), mass_before)
  np.testing.assert_array_equal(_host(sim._raw_sensordata), raw_before)


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('ids', [[], [0, 0], [-1], [3], [True], [.5], [[0]]])
def test_invalid_selection_rejects_before_actual_evaluation(stage, ids, monkeypatch):
  _, sim, record, _ = _prepared()
  before = sim.snapshot()
  bindings = {name: getattr(sim, name) for name in sensor_stages._FIELDS}
  def forbidden(*args, **kwargs):
    raise AssertionError('invalid selection reached device sensor evaluation')
  monkeypatch.setattr(sensor_stages, '_evaluate_stage', forbidden)
  with pytest.raises((ValueError, TypeError, IndexError)):
    _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record, env_ids=ids))
  _assert_preserved(sim, record, before, bindings)


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('change', ['mutation', 'replacement', 'reset', 'restore'])
def test_stale_prepared_records_reject_before_evaluation(stage, change, monkeypatch):
  _, sim, record, _ = _prepared()
  if change == 'mutation':
    sim.state._qpos.add_(0)
  elif change == 'replacement':
    sim.state._qvel = sim.state._qvel.clone()
  elif change == 'reset':
    _native_only(monkeypatch, sim.reset)
  else:
    _native_only(monkeypatch, lambda: sim.restore(sim.snapshot()))
  before = sim.snapshot()
  def forbidden(*args, **kwargs):
    raise AssertionError('stale record reached device sensor evaluation')
  monkeypatch.setattr(sensor_stages, '_evaluate_stage', forbidden)
  with pytest.raises(ValueError):
    _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  after = sim.snapshot()
  for name in ('sensordata', 'energy', 'energy_computed'):
    np.testing.assert_array_equal(after[name], before[name])


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
def test_memory_rejection_precedes_transaction_and_exact_bound_admits_execution(stage, monkeypatch):
  _, sim, record, _ = _prepared()
  needed = sensor_stages._admit(sim)
  original_limits = sim.limits
  sim.limits = replace(original_limits, memory_budget_bytes=needed-1)
  before = sim.snapshot()
  bindings = {name: getattr(sim, name) for name in sensor_stages._FIELDS}
  with monkeypatch.context() as guard:
    def forbidden(*args, **kwargs):
      raise AssertionError('memory rejection must precede rollback allocation/evaluation')
    guard.setattr(api, '_inverse_query_workspaces', forbidden)
    guard.setattr(sensor_stages, '_evaluate_stage', forbidden)
    with pytest.raises(CapacityOverflow, match='sensor stage publication'):
      _native_only(guard, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  _assert_preserved(sim, record, before, bindings)
  sim.limits = replace(original_limits, memory_budget_bytes=needed)
  result = _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  assert tuple(result.shape) == (sim.batch_size, sim._mjmodel.nsensordata)


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
def test_owned_mutable_result_under_inference_mode(stage, monkeypatch):
  _, sim, record, _ = _prepared()
  with sim.state._torch.inference_mode():
    result = _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  assert not sim.state._torch.is_inference(result)
  before = sim.step_sensordata()
  result.fill_(-7)
  np.testing.assert_array_equal(sim.step_sensordata(), before)


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
def test_zero_sensor_model_returns_owned_empty_samples_without_advancing(stage, monkeypatch):
  model = mujoco.MjModel.from_xml_string('<mujoco/>')
  sim = MetalSimulation(model, 2, profile='integrated_scalable_v1')
  record = _native_only_prepare(sim)
  before = sim.snapshot()
  result = _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  assert tuple(result.shape) == (2, 0)
  assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
  np.testing.assert_array_equal(sim.snapshot()['device'].time, before['device'].time)


@pytest.mark.parametrize('stage', ['Pos', 'Vel'])
def test_public_stage_consumes_minimum_prefix_without_forward_completion(stage, monkeypatch):
  model, sim, refs = _fixture('limits', 'integrated_scalable_v1')
  record = _native_only(monkeypatch, lambda: api.mj_fwdPosition(
      sim, skipsensor=True, return_record=True))
  for data in refs:
    mujoco.mj_fwdPosition(model, data)
  if stage == 'Vel':
    _native_only(monkeypatch, lambda: api.mj_fwdVelocity(sim, record=record, skipsensor=True))
    for data in refs:
      mujoco.mj_fwdVelocity(model, data)
  previous_stage = record.stage
  for data in refs:
    data.sensordata[:] = 20
    getattr(mujoco, 'mj_sensor'+stage)(model, data)
  sim._sensordata = sim.state._torch.full((3, model.nsensordata), 20., device=sim.state._device)
  sim._raw_sensordata = sim._sensordata.clone()
  def forbidden(*args, **kwargs):
    raise AssertionError('sensor stage must not complete a missing forward prefix')
  with monkeypatch.context() as guard:
    for name in ('prepare_forward_position', 'prepare_forward_velocity',
                 'prepare_forward_actuation', 'prepare_forward_acceleration',
                 'prepare_forward_constraint'):
      guard.setattr(sim, name, forbidden)
    result = _native_only(guard, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  np.testing.assert_allclose(_host(result), [d.sensordata for d in refs], rtol=2e-4, atol=2e-4)
  assert record.stage == previous_stage


@pytest.mark.parametrize('stage', ['Vel', 'Acc'])
def test_missing_required_prefix_rejects_without_sensor_writes(stage, monkeypatch):
  _, sim, _ = _fixture('limits', 'integrated_scalable_v1')
  record = _native_only(monkeypatch, lambda: api.mj_fwdPosition(
      sim, skipsensor=True, return_record=True))
  before = sim.snapshot()
  with pytest.raises(ValueError, match='needs'):
    _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  np.testing.assert_array_equal(sim.snapshot()['sensordata'], before['sensordata'])


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
def test_truly_sleeping_worlds_preserve_source_sensor_samples(stage, monkeypatch):
  from test_awake_simulation_017 import DAMPED_XML
  xml = DAMPED_XML.replace('</body>', '<site name="imu"/></body>').replace(
      '</mujoco>', '<sensor><jointpos joint="slider"/><jointvel joint="slider"/>'
      '<accelerometer site="imu"/></sensor></mujoco>')
  model = mujoco.MjModel.from_xml_string(xml)
  v = np.asarray([[.06], [.04]], np.float32)
  sim = MetalSimulation(model, 2, qvel=v, profile='integrated_scalable_v1')
  refs = [mujoco.MjData(model), mujoco.MjData(model)]
  for data, value in zip(refs, v):
    data.qvel[:] = value
  for _ in range(15):
    _native_only(monkeypatch, sim.step)
    for data in refs:
      mujoco.mj_step(model, data)
  assert all(not d.tree_awake.any() for d in refs)
  np.testing.assert_array_equal(_host(sim._sleep_schedule.tree_awake), 0)
  record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
  for data in refs:
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
    data.sensordata[:] = 20
    getattr(mujoco, 'mj_sensor'+stage)(model, data)
  sim._sensordata = sim.state._torch.full((2, model.nsensordata), 20., device=sim.state._device)
  sim._raw_sensordata = sim._sensordata.clone()
  result = _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  np.testing.assert_array_equal(_host(result), [d.sensordata for d in refs])
  np.testing.assert_array_equal(_host(result), 20)


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('peer', ['sticky_failure', 'selected_reset'])
def test_bad_or_invalidated_peer_does_not_change_healthy_sensor_publication(stage, peer, monkeypatch):
  model, sim, record, refs = _prepared()
  if peer == 'sticky_failure':
    sim.state._status[1] = 2
  else:
    _native_only(monkeypatch, lambda: sim.reset(env_ids=[1]))
    with pytest.raises(ValueError, match='stale'):
      _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
    record = _native_only_prepare(sim)
  before = sim.snapshot()
  expected = sim.step_sensordata()
  for world in (0, 2):
    refs[world].sensordata[:] = expected[world]
    getattr(mujoco, 'mj_sensor'+stage)(model, refs[world])
    expected[world] = refs[world].sensordata
  result = _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(
      sim, record, env_ids=[0, 2] if peer == 'selected_reset' else None))
  np.testing.assert_allclose(_host(result), expected, rtol=2e-4, atol=2e-4)
  after = sim.snapshot()
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status', 'precision_words'):
    np.testing.assert_array_equal(getattr(after['device'], name), getattr(before['device'], name))
  np.testing.assert_array_equal(_host(result)[1], before['sensordata'][1])


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
def test_actual_inference_input_without_mutation_counter_rejects_before_sampling(stage, monkeypatch):
  _, sim, _ = _fixture('limits', 'integrated_scalable_v1')
  with sim.state._torch.inference_mode():
    value = (sim.state._qpos if stage == 'Pos' else sim.state._qvel).clone()
  record = _native_only(monkeypatch, lambda: api.mj_fwdPosition(
      sim, qpos=value if stage == 'Pos' else None, skipsensor=True, return_record=True))
  if stage != 'Pos':
    _native_only(monkeypatch, lambda: api.mj_fwdVelocity(
        sim, qvel=value, record=record, skipsensor=True))
  before = sim.snapshot()
  def forbidden(*args, **kwargs):
    raise AssertionError('input without a mutation counter reached sensor evaluation')
  monkeypatch.setattr(sensor_stages, '_evaluate_stage', forbidden)
  # The full ACC prefix cannot be published from unversioned VEL inputs.
  with pytest.raises(ValueError, match='mutation counter' if stage != 'Acc' else 'needs CONSTRAINT'):
    _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
  np.testing.assert_array_equal(sim.snapshot()['sensordata'], before['sensordata'])
