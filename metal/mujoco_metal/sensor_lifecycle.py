# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned raw sensor samples and adapter diagnostics at lifecycle boundaries."""
import numpy as np
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import copy


@contextmanager
def lifecycle_transaction(sim):
  """Rollback owned state, bindings, caches and adapters after actual failure.

  This explicit host lifecycle boundary can serialize adapter state. Numerical
  storage is copied on its owning device; no physics is run for rollback.
  """
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  from mujoco_metal.native_api import (
      _capture_query_storage, _contains_device_tensor,
      _restore_query_workspaces, _owned_native_backing_bytes)
  torch, seen = sim.state._torch, set()
  def count(value):
    if id(value) in seen:
      return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
      return value.numel()*value.element_size()
    if isinstance(value, dict):
      return sum(count(v) for v in value.values())
    if isinstance(value, (tuple, list)):
      return sum(count(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
      return sum(count(getattr(value, field.name)) for field in fields(value))
    if type(value).__module__.startswith('mujoco_metal.') and hasattr(value, '__dict__'):
      return count(vars(value))
    return 0
  plugins = tuple(getattr(sim, '_native_plugins', ()))
  rollback_bytes = count(sim)+sum(count(vars(plugin)) for plugin in plugins)
  declared = sum(int(plugin.device_snapshot_bytes()) for plugin in plugins
                 if callable(getattr(plugin, 'device_snapshot_bytes', None)))
  limit = (getattr(sim, 'limits', None) or CapacityLimits()).memory_budget_bytes
  if _owned_native_backing_bytes(sim)+2*rollback_bytes+3*declared > limit:
    raise CapacityOverflow('lifecycle rollback exceeds owned memory budget')
  saved, visited, memo = [], set(), {}
  def collect(program):
    if id(program) in visited:
      return
    visited.add(id(program))
    attributes = dict(vars(program))
    storage = {name: _capture_query_storage(value, memo)
        for name, value in attributes.items()
        if _contains_device_tensor(value) or isinstance(value, (dict, list))
        or (isinstance(value, np.ndarray) and value.flags.writeable)}
    saved.append((program, attributes, storage))
    for name, value in attributes.items():
      if (name != '_owner' and type(value).__module__.startswith('mujoco_metal.')
          and hasattr(value, '__dict__')):
        collect(value)
  stages = getattr(sim, '_forward_stages', None)
  record = getattr(stages, '_record', None)
  coherent = []
  if record is not None:
    for name, value in record.input_tensors.items():
      version = record.input_versions.get(name)
      if version is not None and stages._version(value) == version:
        coherent.append((name, value))
  paired = []
  system = getattr(sim, '_last_coupled', None)
  for value, name in ((getattr(sim, '_last_sensor_qacc', None), '_last_sensor_qacc_version'),
                     (system.get('qacc') if isinstance(system, dict) else None, '_last_coupled_qacc_version')):
    version = getattr(sim, name, None)
    if isinstance(value, torch.Tensor) and isinstance(version, int):
      try:
        if value._version == version:
          paired.append((value, name))
      except RuntimeError:
        pass
  with torch.inference_mode(False), torch.no_grad():
    collect(sim)
    before = [(plugin, copy.deepcopy(plugin.snapshot())) for plugin in plugins]
    try:
      yield
    except BaseException:
      _restore_query_workspaces(saved, (), {}, {}, sim, stages, record, coherent)
      for value, name in paired:
        setattr(sim, name, int(value._version))
      failures = []
      for plugin, payload in reversed(before):
        try:
          plugin.restore(payload)
        except Exception as error:
          failures.append(error)
      if failures:
        raise RuntimeError('lifecycle adapter rollback failed') from failures[0]
      raise


def snapshot_sensor_diagnostics(sim):
  result = {'sensor_diagnostics_version': 1}
  for key, attr in (('raw_sensordata', '_raw_sensordata'),
                    ('sensor_plugin_status', '_sensor_plugin_status')):
    value = getattr(sim, attr, None)
    result[key] = None if value is None else value.detach().cpu().numpy().copy()
  return result


def validate_sensor_diagnostics(sim, snapshot):
  """Validate complete host payload before callbacks or any device mutation."""
  modern = 'sensor_diagnostics_version' in snapshot
  if modern:
    version = snapshot['sensor_diagnostics_version']
    if (isinstance(version, (bool, np.bool_))
        or not isinstance(version, (int, np.integer)) or int(version) != 1):
      raise ValueError('unsupported sensor diagnostics version')
    if not {'raw_sensordata', 'sensor_plugin_status'} <= snapshot.keys():
      raise ValueError('snapshot sensor diagnostics fields are missing')
  # Legacy snapshots had stored samples but no separate raw high words.
  # Initialize from their stored sample instead of retaining future raw data.
  value = snapshot.get('raw_sensordata', snapshot.get('sensordata'))
  raw = None
  if value is not None:
    array = np.asarray(value, dtype=np.float64)
    expected = (sim.batch_size, int(sim._mjmodel.nsensordata))
    if array.shape != expected or not np.isfinite(array).all():
      raise ValueError('snapshot raw_sensordata has invalid shape or nonfinite values')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
      raw = array.astype(np.float32)
    if not np.isfinite(raw).all():
      raise ValueError('snapshot raw_sensordata values overflow float32')
    raw = raw.copy()
  current = getattr(sim, '_sensor_plugin_status', None)
  value = snapshot.get('sensor_plugin_status')
  if value is None:
    if modern and current is not None:
      raise ValueError('snapshot sensor_plugin_status is missing')
    status = np.zeros(sim.batch_size, np.int32) if current is not None else None
  else:
    array = np.asarray(value)
    if (array.shape != (sim.batch_size,) or array.dtype.kind not in 'biu'
        or np.any(array < 0) or np.any(array > np.iinfo(np.int32).max)):
      raise ValueError('snapshot sensor_plugin_status must be nonnegative int32[batch]')
    if current is None:
      raise ValueError('snapshot sensor_plugin_status has no diagnostic owner')
    status = array.astype(np.int32, copy=True)
  return raw, status


def stage_sensor_diagnostics(sim, checked, ids=None):
  """Admit and stage owned replacement bindings before physical publication."""
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  from mujoco_metal.native_api import _owned_native_backing_bytes
  raw, status = checked
  batch, width = sim.batch_size, int(sim._mjmodel.nsensordata)
  # Host upload, retained destination and publication/selection reserve.
  needed = 12*batch*(width+1)+8*batch
  budget = (getattr(sim, 'limits', None) or CapacityLimits()).memory_budget_bytes
  if _owned_native_backing_bytes(sim)+needed > budget:
    raise CapacityOverflow('sensor diagnostic restore exceeds owned memory budget')
  torch, device = sim.state._torch, sim.state._device
  with torch.inference_mode(False), torch.no_grad():
    index = None if ids is None else torch.as_tensor(ids, dtype=torch.long, device=device)
    replacements = {}
    for attr, payload, shape, dtype in (
        ('_raw_sensordata', raw, (batch, width), torch.float32),
        ('_sensor_plugin_status', status, (batch,), torch.int32)):
      current = getattr(sim, attr, None)
      if payload is None and ids is None:
        replacements[attr] = None
        continue
      if payload is None and current is None:
        replacements[attr] = None
        continue
      staged = torch.zeros(shape, dtype=dtype, device=device) if current is None else current.clone()
      source = torch.zeros(shape, dtype=dtype, device=device) if payload is None else torch.as_tensor(
          payload, dtype=dtype, device=device)
      if index is None:
        staged.copy_(source)
      else:
        staged.index_copy_(0, index, source.index_select(0, index))
      replacements[attr] = staged
  return replacements


def clear_sensor_diagnostics(sim, env_ids=None):
  """Cold reset only selected raw/status rows, retaining lazy allocation."""
  ids = None if env_ids is None else sim.state._env_ids(env_ids)
  for attr in ('_raw_sensordata', '_sensor_plugin_status'):
    value = getattr(sim, attr, None)
    if value is not None:
      if ids is None:
        value.zero_()
      elif ids.size:
        value[ids.tolist()] = 0


def publish_sensor_diagnostics(sim, values):
  """Publish staged bindings inside the enclosing lifecycle transaction."""
  vars(sim).update(values)


def copy_sensor_diagnostic_row(sim, source, destination):
  for attr in ('_raw_sensordata', '_sensor_plugin_status'):
    value = getattr(sim, attr, None)
    if value is not None:
      value[destination] = value[source].clone()
