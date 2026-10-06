# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit device-query rollback preserves coherent record storage/tokens."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('stale', [False, True])
def test_query_restores_paired_acceleration_values_aliases_and_coherence(fail, stale):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  from mujoco_metal.simulation import MetalSimulation
  high = torch.tensor([[1., 2.]])
  low = torch.tensor([[1e-7, -2e-7]])
  rhs_low = torch.tensor([[3e-7, 4e-7]])
  sim = SimpleNamespace(
      _state=SimpleNamespace(_torch=torch, generation=7),
      _component_solution_vector=high,
      _component_solution_low_vector=low, _rhs_low=rhs_low,
      _last_sensor_qacc=high, _last_sensor_qacc_low=low,
      _last_sensor_qacc_generation=7, _last_sensor_qacc_version=int(high._version),
      _last_coupled={'qacc': high, 'qacc_low': low},
      _last_coupled_generation=7, _last_coupled_qacc_version=int(high._version))
  sim._forward_position_buffers = {'actuators': {'length': rhs_low}}
  if stale:
    high.add_(1.)
  saved = [value.clone() for value in (high, low, rhs_low)]
  def query():
    with _inverse_query_workspaces(sim):
      high.add_(10.)
      low.zero_()
      rhs_low.fill_(99.)
      sim._forward_position_buffers['actuators'] = {'length': torch.zeros_like(rhs_low)}
      sim._last_sensor_qacc = torch.zeros_like(high)
      sim._last_sensor_qacc_low = None
      sim._component_solution_low_vector = torch.zeros_like(low)
      sim._last_sensor_qacc_generation = 100
      sim._last_coupled = None
      if fail:
        raise RuntimeError('paired query failed')
  if fail:
    with pytest.raises(RuntimeError, match='paired query failed'):
      query()
  else:
    query()
  assert sim._last_sensor_qacc is high
  assert sim._last_sensor_qacc_low is low
  assert sim._component_solution_low_vector is low
  assert sim._last_coupled['qacc'] is high
  assert sim._last_coupled['qacc_low'] is low
  assert sim._forward_position_buffers['actuators']['length'] is rhs_low
  for value, before in zip((high, low, rhs_low), saved):
    torch.testing.assert_close(value, before, rtol=0, atol=0)
  assert sim._last_sensor_qacc_generation == 7
  assert MetalSimulation._qacc_low_for_sensor(sim, high) is (None if stale else low)


@pytest.mark.parametrize('fail', [False, True])
def test_query_restores_record_values_owners_tokens_and_bookkeeping(fail):
  torch = pytest.importorskip('torch')
  from mujoco_metal.forward_stages import ForwardStage as Stage, ForwardStageCoordinator
  from mujoco_metal.native_api import _inverse_query_workspaces
  sim = SimpleNamespace(_forward_position_epoch=12,
      _last_actuation_kin={'force': torch.tensor([[.7]])},
      _rhs=torch.tensor([[.5]]))
  sim._forward_stages = ForwardStageCoordinator(sim)
  qpos, qvel = torch.tensor([[.2]]), torch.tensor([[.3]])
  record = sim._forward_stages.begin(generation=0, qpos=qpos,
                                   position={'mass': torch.tensor([[[2.]]])})
  sim._forward_stages.publish(record, Stage.VEL, {'qvel': qvel})
  sim._step1_record = record
  original_force = sim._last_actuation_kin['force']
  def query():
    with _inverse_query_workspaces(sim):
      qpos.add_(9)
      qvel.add_(8)
      sim._forward_stages.invalidate()
      sim._step1_record = None
      sim._forward_position_epoch = 99
      sim._last_actuation_kin['force'] = torch.tensor([[99.]])
      sim._rhs.add_(10)
      if fail:
        raise RuntimeError('query failed')
  if fail:
    with pytest.raises(RuntimeError, match='query failed'):
      query()
  else:
    query()
  assert sim._forward_position_epoch == 12
  assert sim._last_actuation_kin['force'] is original_force
  torch.testing.assert_close(original_force, torch.tensor([[.7]]))
  torch.testing.assert_close(sim._rhs, torch.tensor([[.5]]))
  assert sim._forward_stages._record is record
  assert sim._step1_record is record
  assert record.qpos is qpos and record.values[Stage.VEL]['qvel'] is qvel
  torch.testing.assert_close(qpos, torch.tensor([[.2]]))
  torch.testing.assert_close(qvel, torch.tensor([[.3]]))
  sim._forward_stages.validate(record, generation=0, minimum=Stage.VEL)


def test_query_never_rehabilitates_a_record_already_stale_on_entry():
  torch = pytest.importorskip('torch')
  from mujoco_metal.forward_stages import ForwardStageCoordinator
  from mujoco_metal.native_api import _inverse_query_workspaces
  sim = SimpleNamespace()
  sim._forward_stages = ForwardStageCoordinator(sim)
  qpos = torch.tensor([[.2]])
  record = sim._forward_stages.begin(generation=0, qpos=qpos, position={'mass': object()})
  qpos.add_(1)
  with _inverse_query_workspaces(sim):
    pass
  with pytest.raises(ValueError, match='mutated'):
    sim._forward_stages.validate(record, generation=0)


@pytest.mark.parametrize('fail', [False, True])
def test_query_restores_legacy_row_producers_and_combined_scratch(fail):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  contact_rows = torch.arange(12, dtype=torch.float32).reshape(2, 6)
  joint_rows = torch.arange(6, dtype=torch.float32).reshape(2, 3)
  detection = {'pos': torch.tensor([[[.1, .2, .3]], [[.4, .5, .6]]])}
  contact = SimpleNamespace(_workspace={'canonical_rows': contact_rows,
                                        'detector': detection})
  joint = SimpleNamespace(_outputs={'canonical_rows': joint_rows})
  combined = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
  rhs = torch.tensor([[.2, .4], [.6, .8]])
  sim = SimpleNamespace(_contact=contact, _joint_constraints=joint,
      _legacy_canonical_rows=combined, _legacy_constraint_rhs=rhs,
      _component_solution_vector=torch.tensor([[.1, .3], [.5, .7]]))
  vector_owner = sim._component_solution_vector
  vector_before = vector_owner.clone()
  saved = [value.clone() for value in
           (contact_rows, joint_rows, detection['pos'], combined, rhs)]
  def query():
    with _inverse_query_workspaces(sim):
      contact_rows.fill_(9)
      contact._workspace['canonical_rows'] = torch.zeros_like(contact_rows)
      detection['pos'].add_(7)
      contact._workspace['detector'] = {'pos': torch.zeros_like(detection['pos'])}
      joint_rows.mul_(3)
      joint._outputs.clear()
      combined.fill_(8)
      sim._legacy_canonical_rows = None
      rhs.add_(6)
      sim._component_solution_vector.fill_(9)
      sim._component_solution_vector = torch.zeros_like(vector_owner)
      if fail:
        raise RuntimeError('legacy row query failed')
  if fail:
    with pytest.raises(RuntimeError, match='legacy row query failed'):
      query()
  else:
    query()
  assert contact._workspace['canonical_rows'] is contact_rows
  assert contact._workspace['detector'] is detection
  assert joint._outputs['canonical_rows'] is joint_rows
  assert sim._legacy_canonical_rows is combined
  assert sim._legacy_constraint_rhs is rhs
  assert sim._component_solution_vector is vector_owner
  torch.testing.assert_close(vector_owner, vector_before, rtol=0, atol=0)
  for actual, expected in zip(
      (contact_rows, joint_rows, detection['pos'], combined, rhs), saved):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('capacity', [0, 3])
def test_query_restores_real_frozen_compaction_map_without_masking_errors(
    fail, capacity):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  from mujoco_metal.row_compaction import CompactionMap
  packed = torch.full((2, capacity), -1, dtype=torch.int32)
  reverse = torch.full((2, capacity), -1, dtype=torch.int32)
  count = torch.zeros(2, dtype=torch.int32)
  overflow = torch.zeros(2, dtype=torch.int32)
  mapping = CompactionMap(packed, reverse, count, overflow)
  program = SimpleNamespace(_slot_map=mapping, _workspace={'slot_maps': mapping})
  sim = SimpleNamespace(_coupled_constraints=program)
  def query():
    with _inverse_query_workspaces(sim):
      packed.fill_(2)
      reverse.fill_(1)
      count.fill_(capacity)
      overflow.fill_(1)
      program._slot_map = None
      program._workspace.clear()
      if fail:
        raise RuntimeError('original compaction producer failure')
  if fail:
    with pytest.raises(RuntimeError, match='original compaction producer failure'):
      query()
  else:
    query()
  assert program._slot_map is mapping
  assert program._workspace['slot_maps'] is mapping
  assert mapping.packed_to_logical is packed
  assert mapping.logical_to_packed is reverse
  assert mapping.active_count is count and mapping.overflow is overflow
  torch.testing.assert_close(packed, torch.full_like(packed, -1), rtol=0, atol=0)
  torch.testing.assert_close(reverse, torch.full_like(reverse, -1), rtol=0, atol=0)
  torch.testing.assert_close(count, torch.zeros_like(count), rtol=0, atol=0)
  torch.testing.assert_close(overflow, torch.zeros_like(overflow), rtol=0, atol=0)


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('allocated', [False, True])
def test_query_restores_energy_values_and_owner_even_if_buffer_is_replaced(
    fail, allocated):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  owner = torch.tensor([[2., .5], [3., .7]]) if allocated else None
  sim = SimpleNamespace(_energy=owner)
  saved = owner.clone() if allocated else None
  def query():
    with _inverse_query_workspaces(sim):
      if allocated:
        sim._energy.add_(9)
      sim._energy = torch.zeros(2, 2)
      if fail:
        raise RuntimeError('energy query failed')
  if fail:
    with pytest.raises(RuntimeError, match='energy query failed'):
      query()
  else:
    query()
  assert sim._energy is owner
  if allocated:
    torch.testing.assert_close(owner, saved, rtol=0, atol=0)


@pytest.mark.parametrize('fail', [False, True])
def test_query_restores_sleep_state_sensor_status_and_plugin_device_state(fail):
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  from mujoco_metal.forward_stages import ForwardStage as Stage, ForwardStageCoordinator

  class Plugin:
    def __init__(self):
      self.values = torch.tensor([[1.], [2.]])
      self.restores = 0
    def snapshot(self):
      raise AssertionError('host plugin snapshot must not be used')
    def device_snapshot(self):
      return self.values.clone()
    def restore_device(self, payload):
      self.values.copy_(payload)
      self.restores += 1

  plugin = Plugin()
  original_qpos = torch.tensor([[.2], [.3]])
  masked_qvel = torch.tensor([[.4], [0.]])
  state = SimpleNamespace(_qpos=original_qpos, _generation=12)
  scheduler = SimpleNamespace(tree_state=torch.tensor([[-11], [0]]),
      tree_awake=torch.tensor([[1], [0]], dtype=torch.int32), epoch=3)
  sim = SimpleNamespace(_state=state, _sleep_schedule=scheduler,
      _sleep_qvel=masked_qvel, _native_plugins=(plugin,),
      _sensor_plugin_status=torch.tensor([0, 2], dtype=torch.int32))
  sim._forward_stages = ForwardStageCoordinator(sim)
  record = sim._forward_stages.begin(generation=12, qpos=original_qpos, position={})
  sim._forward_stages.publish(record, Stage.VEL, {'qvel': masked_qvel})
  before = scheduler.tree_state.clone()
  def query():
    with _inverse_query_workspaces(sim):
      state._qpos.add_(7)
      state._generation += 1
      scheduler.tree_state.fill_(-11)
      scheduler.tree_awake.fill_(1)
      scheduler.epoch += 1
      sim._sleep_qvel.add_(8)
      sim._sensor_plugin_status.fill_(7)
      plugin.values.add_(9)
      sim._forward_stages.invalidate()
      if fail:
        raise RuntimeError('temporary sleeping query failed')
  if fail:
    with pytest.raises(RuntimeError, match='temporary sleeping query failed'):
      query()
  else:
    query()
  assert state._qpos is original_qpos and state._generation == 12
  assert sim._sleep_qvel is masked_qvel
  torch.testing.assert_close(original_qpos, torch.tensor([[.2], [.3]]))
  torch.testing.assert_close(masked_qvel, torch.tensor([[.4], [0.]]))
  torch.testing.assert_close(scheduler.tree_state, before)
  torch.testing.assert_close(scheduler.tree_awake, torch.tensor([[1], [0]], dtype=torch.int32))
  torch.testing.assert_close(sim._sensor_plugin_status, torch.tensor([0, 2], dtype=torch.int32))
  torch.testing.assert_close(plugin.values, torch.tensor([[1.], [2.]]))
  assert plugin.restores == 1 and scheduler.epoch == 3
  sim._forward_stages.validate(record, generation=12, minimum=Stage.VEL)


def test_query_restores_other_plugins_and_owned_buffers_if_one_rollback_fails():
  torch = pytest.importorskip('torch')
  from mujoco_metal.native_api import _inverse_query_workspaces
  calls = []
  class Plugin:
    def __init__(self, name, fail):
      self.name, self.fail = name, fail
    def device_snapshot(self):
      return None
    def restore_device(self, value):
      calls.append(self.name)
      if self.fail:
        raise ValueError('rollback refused')
  sim = SimpleNamespace(_native_plugins=(Plugin('good', False), Plugin('bad', True)),
                        _rhs=torch.tensor([[.5]]))
  with pytest.raises(RuntimeError, match='plugin device rollback failed'):
    with _inverse_query_workspaces(sim):
      sim._rhs.fill_(9)
  assert calls == ['bad', 'good']
  torch.testing.assert_close(sim._rhs, torch.tensor([[.5]]))
