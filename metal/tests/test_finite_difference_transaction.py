# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU fakes exercise the native derivative rollback boundary without MPS."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mujoco_metal.finite_difference import (
    DeviceQueryTransaction, estimate_transaction_bytes,
)


class _Program:
  __module__ = "mujoco_metal.fake_query_program"

  def __init__(self, state):
    self.state = state
    self.work = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    array_root = np.arange(8, dtype=np.float32)
    self.nested = {"array_root": array_root,
                   "array": array_root[::2],
                   "alias": self.work[:, 1:]}
    self.nested["array_view"] = self.nested["array"][::-1]
    self.nested["empty"] = np.empty((0,), dtype=np.float32)
    self.generation = 5

  def evaluate(self, fail=False):
    self.work.add_(3)
    self.nested["array"][:] += 2
    self.generation += 1
    self.state.sensor.copy_(self.state.qpos[:, :1])
    self.state.qpos.add_(1)
    self.state.record = SimpleNamespace(tag="temporary")
    if fail:
      raise RuntimeError("injected query failure")


class _State:
  __module__ = "mujoco_metal.fake_query_state"

  def __init__(self):
    self.backing = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    self.qpos = self.backing[:, :2]
    self.sensor = self.backing[:, 2:3]
    self.record = SimpleNamespace(tag="prepared")
    self.generation = 11


class _Plugin:
  def __init__(self):
    self.calls = torch.zeros(2, dtype=torch.int32)
    self.snapshot_calls = 0

  def device_snapshot(self):
    self.snapshot_calls += 1
    return self.calls.clone()

  def device_snapshot_bytes(self):
    return self.calls.numel() * self.calls.element_size()

  def restore_device(self, payload):
    self.calls.copy_(payload)


class _Record:
  __module__ = "mujoco_metal.fake_query_record"

  def __init__(self, value, *, stale=False):
    self.input_tensors = {"qpos": value}
    version = value._version
    self.input_versions = {"qpos": version - int(stale)}


class _Stages:
  __module__ = "mujoco_metal.fake_query_stages"

  def __init__(self, value, *, stale=False):
    self._record = _Record(value, stale=stale)

  @staticmethod
  def _version(value):
    return value._version

  def capture_input(self, record, name, value):
    record.input_versions[name] = value._version


class _SetOnlyMember:
  __module__ = "mujoco_metal.fake_query_set_member"

  def __init__(self, value, owner_set):
    self.value = value
    self.alias = value
    self.owner_set = owner_set


class _FakeSimulation:
  __module__ = "mujoco_metal.fake_simulation"

  def __init__(self):
    self._state = _State()
    self._program = _Program(self._state)
    self._native_plugins = (_Plugin(),)
    self.limits = SimpleNamespace(memory_budget_bytes=1 << 20)
    self._prepared_record = SimpleNamespace(generation=11)
    self._alias = self._program.work[:, 1:]
    self._forward_stages = _Stages(self._state.qpos)
    self._program._owner = self
    self._identity_tuple = (self._program, self._program.nested)


def _capture(sim):
  return {
      "qpos": sim._state.qpos.clone(),
      "sensor": sim._state.sensor.clone(),
      "backing": sim._state.backing.clone(),
      "work": sim._program.work.clone(),
      "array": sim._program.nested["array"].copy(),
      "array_view": sim._program.nested["array_view"].copy(),
      "array_root": sim._program.nested["array_root"].copy(),
      "record": sim._state.record,
      "generation": sim._state.generation,
      "program_generation": sim._program.generation,
      "prepared": sim._prepared_record,
      "alias": sim._alias,
      "plugin_calls": sim._native_plugins[0].calls.clone(),
      "stage_record": sim._forward_stages._record,
      "identity_tuple": sim._identity_tuple,
  }


def _assert_restored(sim, saved):
  torch.testing.assert_close(sim._state.qpos, saved["qpos"])
  torch.testing.assert_close(sim._state.sensor, saved["sensor"])
  torch.testing.assert_close(sim._state.backing, saved["backing"])
  torch.testing.assert_close(sim._program.work, saved["work"])
  np.testing.assert_array_equal(sim._program.nested["array"], saved["array"])
  np.testing.assert_array_equal(sim._program.nested["array_view"], saved["array_view"])
  np.testing.assert_array_equal(sim._program.nested["array_root"], saved["array_root"])
  torch.testing.assert_close(sim._program.nested["alias"], saved["work"][:, 1:])
  assert sim._state.record is saved["record"]
  assert sim._state.generation == saved["generation"]
  assert sim._program.generation == saved["program_generation"]
  assert sim._prepared_record is saved["prepared"]
  assert sim._alias is saved["alias"]
  assert sim._forward_stages._record is saved["stage_record"]
  assert sim._identity_tuple is saved["identity_tuple"]
  assert (sim._forward_stages._version(sim._state.qpos) ==
          sim._forward_stages._record.input_versions["qpos"])
  torch.testing.assert_close(sim._native_plugins[0].calls, saved["plugin_calls"])


def test_transaction_restores_complete_fake_device_query_and_reuses_checkpoint():
  sim = _FakeSimulation()
  saved = _capture(sim)
  original_program = sim._program
  # Aliased views share one saved storage allocation in the preflight estimate.
  assert estimate_transaction_bytes(sim) == 124
  with DeviceQueryTransaction(sim, extra_bytes=4096) as transaction:
    sim._program.evaluate()
    sim._native_plugins[0].calls.add_(1)
    sim._program = _Program(sim._state)
    transaction.restore()
    assert sim._program is original_program
    _assert_restored(sim, saved)
    sim._program.evaluate()
    sim._native_plugins[0].calls.add_(2)
  _assert_restored(sim, saved)


def test_transaction_restores_mutations_and_replacements_after_exception():
  sim = _FakeSimulation()
  saved = _capture(sim)
  with pytest.raises(RuntimeError, match="injected query failure"):
    with DeviceQueryTransaction(sim):
      sim._program.evaluate(fail=True)
      sim._native_plugins[0].calls.add_(1)
  _assert_restored(sim, saved)


def test_transaction_requires_explicit_plugin_device_rollback_contract():
  sim = _FakeSimulation()
  sim._native_plugins = (object(),)
  with pytest.raises(TypeError, match="device rollback methods"):
    DeviceQueryTransaction(sim)


def test_transaction_admits_plugin_payload_before_calling_snapshot():
  sim = _FakeSimulation()
  plugin = sim._native_plugins[0]
  snapshot_bytes = estimate_transaction_bytes(sim)
  sim.limits.memory_budget_bytes = snapshot_bytes - 1
  with pytest.raises(ValueError, match="memory budget"):
    DeviceQueryTransaction(sim)
  assert plugin.snapshot_calls == 0

  sim.limits.memory_budget_bytes = snapshot_bytes
  with DeviceQueryTransaction(sim):
    assert plugin.snapshot_calls == 1


def test_transaction_rejects_unbounded_plugin_snapshot_before_allocating():
  sim = _FakeSimulation()
  plugin = sim._native_plugins[0]
  plugin.device_snapshot_bytes = lambda: -1
  with pytest.raises(ValueError, match="nonnegative int"):
    DeviceQueryTransaction(sim)
  assert plugin.snapshot_calls == 0


def test_transaction_reports_plugin_restore_failure_after_restoring_simulation():
  sim = _FakeSimulation()
  saved = _capture(sim)
  plugin = sim._native_plugins[0]
  original_restore = plugin.restore_device

  def broken_restore(payload):
    original_restore(payload)
    raise RuntimeError("injected plugin restore failure")

  plugin.restore_device = broken_restore
  with pytest.raises(RuntimeError, match="plugin rollback failed"):
    with DeviceQueryTransaction(sim):
      sim._program.evaluate()
      sim._native_plugins[0].calls.add_(1)
  _assert_restored(sim, saved)


def test_preflight_never_probes_dynamic_shader_function_attributes():
  class ShaderLibrary:
    __module__ = "torch.mps.shader"

    def __getattr__(self, name):
      if name in ("device_snapshot", "restore_device", "__dict__"):
        raise RuntimeError("dynamic shader lookup would compile")
      raise AttributeError(name)

  sim = _FakeSimulation()
  sim._shader_library = ShaderLibrary()
  assert estimate_transaction_bytes(sim) == 124


def test_transaction_restores_zero_stride_view_through_its_unique_storage():
  sim = _FakeSimulation()
  root = torch.tensor([1., 2., 3.])
  broadcast = root.as_strided((4,), (0,), storage_offset=1)
  sim._zero_stride_views = (root, broadcast)
  with DeviceQueryTransaction(sim):
    root.add_(7)
  torch.testing.assert_close(root, torch.tensor([1., 2., 3.]))
  torch.testing.assert_close(broadcast, torch.tensor([2., 2., 2., 2.]))


def test_transaction_preflight_accounts_for_dtype_alias_clone_duplication():
  sim = _FakeSimulation()
  root = torch.arange(4, dtype=torch.float32)
  reinterpret = root.view(torch.int32)
  sim._dtype_alias = (root, reinterpret)
  before = estimate_transaction_bytes(sim)
  assert before - estimate_transaction_bytes(_FakeSimulation()) == 2 * root.numel() * 4
  saved_bits = reinterpret.clone()
  with DeviceQueryTransaction(sim):
    root.add_(1)
  torch.testing.assert_close(root, torch.arange(4, dtype=torch.float32))
  torch.testing.assert_close(reinterpret, saved_bits)


def test_transaction_restores_set_only_tensor_members_and_cycles():
  sim = _FakeSimulation()
  owner_set = set()
  value = torch.arange(3, dtype=torch.float32)
  member = _SetOnlyMember(value, owner_set)
  owner_set.update((member, value))
  sim._set_only = owner_set
  saved_value = value.clone()
  saved_bytes = estimate_transaction_bytes(sim)
  assert saved_bytes == estimate_transaction_bytes(_FakeSimulation()) + value.numel() * 4

  with DeviceQueryTransaction(sim):
    value.add_(11)
    owner_set.clear()
    owner_set.add(torch.tensor([99.], dtype=torch.float32))

  assert sim._set_only is owner_set
  assert len(owner_set) == 2
  assert member in owner_set
  assert value in owner_set
  assert member.owner_set is owner_set
  assert member.value is value
  assert member.alias is value
  torch.testing.assert_close(value, saved_value)


def test_transaction_does_not_revalidate_an_already_stale_prepared_record():
  sim = _FakeSimulation()
  sim._forward_stages._record = _Record(sim._state.qpos, stale=True)
  stale_version = sim._forward_stages._record.input_versions["qpos"]
  with DeviceQueryTransaction(sim):
    sim._program.evaluate()
  assert sim._forward_stages._record.input_versions["qpos"] == stale_version
  assert sim._forward_stages._version(sim._state.qpos) != stale_version


def test_transaction_preflights_saved_and_output_bytes_before_cloning():
  sim = _FakeSimulation()
  sim.limits.memory_budget_bytes = 1
  with pytest.raises(ValueError, match="memory budget"):
    DeviceQueryTransaction(sim, extra_bytes=1)


def test_transaction_restores_compensated_bias_workspace():
  sim = _FakeSimulation()
  bias_low = torch.tensor([[0.125, -0.25]], dtype=torch.float32)
  sim._program.bias_low = bias_low
  saved = bias_low.clone()
  base_bytes = estimate_transaction_bytes(_FakeSimulation())
  assert estimate_transaction_bytes(sim) == base_bytes + bias_low.numel() * 4
  with DeviceQueryTransaction(sim):
    bias_low.add_(1)
    sim._program.bias_low = torch.full_like(bias_low, 9)
  assert sim._program.bias_low is bias_low
  torch.testing.assert_close(bias_low, saved, rtol=0, atol=0)


def test_transaction_recoheres_only_entry_coherent_paired_qacc_tags():
  sim = _FakeSimulation()
  qacc = sim._state.qpos
  low = torch.full_like(qacc, 0.125)
  sim._last_coupled = {"qacc": qacc, "qacc_low": low}
  sim._last_coupled_qacc_version = qacc._version
  sim._last_sensor_qacc = qacc
  sim._last_sensor_qacc_low = low
  sim._last_sensor_qacc_version = qacc._version
  saved = qacc.clone()
  with DeviceQueryTransaction(sim) as transaction:
    qacc.add_(4)
    sim._last_coupled = {"qacc": torch.zeros_like(qacc), "qacc_low": low}
    transaction.restore()
  assert sim._last_coupled["qacc"] is qacc
  assert sim._last_sensor_qacc is qacc
  assert sim._last_coupled_qacc_version == qacc._version
  assert sim._last_sensor_qacc_version == qacc._version
  torch.testing.assert_close(qacc, saved)

  stale = _FakeSimulation()
  stale_qacc = stale._state.qpos
  stale_low = torch.full_like(stale_qacc, 0.25)
  stale._last_coupled = {"qacc": stale_qacc, "qacc_low": stale_low}
  stale._last_coupled_qacc_version = stale_qacc._version - 1
  stale._last_sensor_qacc = stale_qacc
  stale._last_sensor_qacc_low = stale_low
  stale._last_sensor_qacc_version = stale_qacc._version - 1
  stale_coupled_version = stale._last_coupled_qacc_version
  stale_sensor_version = stale._last_sensor_qacc_version
  with pytest.raises(RuntimeError, match="injected query failure"):
    with DeviceQueryTransaction(stale):
      stale_qacc.add_(8)
      stale._last_sensor_qacc = torch.zeros_like(stale_qacc)
      raise RuntimeError("injected query failure")
  assert stale._last_coupled["qacc"] is stale_qacc
  assert stale._last_sensor_qacc is stale_qacc
  assert stale._last_coupled_qacc_version == stale_coupled_version
  assert stale._last_sensor_qacc_version == stale_sensor_version
  assert stale_qacc._version != stale_sensor_version
