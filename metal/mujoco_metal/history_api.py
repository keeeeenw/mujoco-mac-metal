# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned native history queries and atomic canonical-buffer initialization."""
import numbers
import numpy as np
from mujoco_metal.model_access import prepared_model


def history_entry(model, kind, identifier):
  """Host configuration admission without device allocation."""
  count = int(model.nu if kind == 0 else model.nsensor)
  if (isinstance(identifier, (bool, np.bool_)) or
      not isinstance(identifier, numbers.Integral) or not 0 <= identifier < count):
    raise ValueError('history id must identify a compiled actuator or sensor')
  identifier = int(identifier)
  nsample, interp = map(int, (model.actuator_history if kind == 0 else model.sensor_history)[identifier])
  return dict(n=nsample, dim=1 if kind == 0 else int(model.sensor_dim[identifier]),
              adr=int((model.actuator_historyadr if kind == 0 else model.sensor_historyadr)[identifier]),
              output=identifier if kind == 0 else int(model.sensor_adr[identifier]),
              interp=interp, delay=float((model.actuator_delay if kind == 0 else model.sensor_delay)[identifier]))


def _budget(sim, words, operation):
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow, _check_i32_elements
  from mujoco_metal.native_api import _owned_native_backing_bytes
  _check_i32_elements('history API temporary workspace', words)
  peak = _owned_native_backing_bytes(sim) + 4 * words
  limit = (sim.limits or CapacityLimits()).memory_budget_bytes
  if peak > limit:
    raise CapacityOverflow(f'{operation} total owned memory peak {peak} bytes exceeds budget {limit}')


def _host_values(value, shape, count, what):
  raw = np.asarray(value)
  if raw.dtype.kind not in 'fiu' or raw.dtype.kind == 'b':
    raise ValueError(f'{what} must contain real finite values')
  data = np.asarray(raw, np.float64)
  if data.shape == shape: data = np.broadcast_to(data, (count, *shape))
  if data.shape != (count, *shape):
    raise ValueError(f'{what} must have shape {shape} or {(count, *shape)}')
  with np.errstate(over='ignore', invalid='ignore'):
    high = data.astype(np.float32)
  if not np.all(np.isfinite(high)):
    raise ValueError(f'{what} must be finite float32-representable values')
  return high.copy()


def _input(value, shape, count, sim, what):
  import torch
  if isinstance(value, torch.Tensor):
    if (value.device != sim.state._history.device or value.dtype != torch.float32 or
        not value.is_contiguous() or tuple(value.shape) not in (shape, (count, *shape))):
      raise ValueError(f'{what} must be contiguous float32 on the simulation device with compiled shape')
    if not bool(torch.isfinite(value).all()):
      raise ValueError(f'{what} must be finite')
    return value.expand(count, *shape) if tuple(value.shape) == shape else value
  return torch.as_tensor(_host_values(value, shape, count, what), device=sim.state._device)


def read_history(sim, kind, identifier, time, interp=-1):
  import torch
  model = prepared_model(sim)
  entry = history_entry(model, kind, identifier)
  if isinstance(interp, (bool, np.bool_)) or not isinstance(interp, numbers.Integral) or interp not in (-1, 0, 1, 2):
    raise ValueError('interp must be -1 (compiled), 0, 1 or 2')
  b, dim = sim.batch_size, entry['dim']
  # Count outputs, query head/low, metadata, delay words and a possible empty
  # raw-storage placeholder before any MPS allocation. Input validation may
  # synchronize explicitly; interpolation never reads numerical data on host.
  absent = kind == 1 and sim._sensordata is None
  rawstride = int(model.nu if kind == 0 else model.nsensordata)
  _budget(sim, b*dim + 5*b + 8 + 2 + 1 + (b*rawstride if absent else 0), 'history read')
  if isinstance(time, torch.Tensor):
    high = _input(time, (), b, sim, 'time')
    low = torch.zeros_like(high)
    query = torch.stack((high, low), dim=1).contiguous()
  else:
    high = _host_values(time, (), b, 'time')
    source = np.asarray(time, np.float64)
    source = np.broadcast_to(source, (b,))
    low = (source - high.astype(np.float64)).astype(np.float32)
    query = torch.as_tensor(np.stack((high, low), axis=1), device=sim.state._device)
  raw = sim._control if kind == 0 else sim._sensordata
  if raw is None:
    raw = torch.zeros((b, rawstride), dtype=torch.float32, device=sim.state._device)
  output = torch.empty((b, dim), dtype=torch.float32, device=sim.state._device)
  metadata = torch.tensor([b, int(model.nhistory), rawstride, entry['adr'], entry['n'], dim,
                           entry['interp'] if interp == -1 else int(interp), entry['output']],
                          dtype=torch.int32, device=sim.state._device)
  delayhi = np.float32(entry['delay'])
  delay = torch.tensor([delayhi, np.float32(entry['delay']-float(delayhi))],
                       dtype=torch.float32, device=sim.state._device)
  dummy = torch.zeros(1, device=sim.state._device)
  sim._history_program._library.history_read_query(
      sim.state._history.reshape(-1) if model.nhistory else dummy,
      raw.reshape(-1), query.reshape(-1), output.reshape(-1), metadata, delay,
      threads=(b*dim,), group_size=(1,))
  return output[:, 0] if kind == 0 else output


def init_history(sim, kind, identifier, times=None, values=None, phase=0.0, env_ids=None):
  import torch
  from mujoco_metal.native_api import _selected_ids
  model = prepared_model(sim)
  e = history_entry(model, kind, identifier)
  n, dim, adr = e['n'], e['dim'], e['adr']
  if not n: raise ValueError('actuator or sensor has no compiled history buffer')
  ids, count = _selected_ids(sim, env_ids)
  # Conservative named peak: selected canonical row clone, staged inputs,
  # indexing, validation differences/masks, and phase; independent of n on stack.
  _budget(sim, count*(int(model.nhistory) + n*(dim+1) + 4*n + 8), 'history initialization')
  times_input = None if times is None else _input(times, (n,), count, sim, 'times')
  values_input = None if values is None else _input(values, (n,dim), count, sim, 'values')
  phase_input = _input(phase, (), count, sim, 'phase') if kind else None
  ids_t = None if ids is None else torch.as_tensor(ids, dtype=torch.long, device=sim.state._device)
  # Work only on a private transaction. Pinned NULL-times uses physical order,
  # not logical ring order; a rotated unsorted buffer must fail atomically.
  rows = sim.state._history.clone() if ids_t is None else sim.state._history.index_select(0, ids_t)
  stamps = rows[:, adr+2:adr+2+n] if times_input is None else times_input
  if not bool(torch.isfinite(stamps).all()) or (n > 1 and not bool((stamps[:,1:]-stamps[:,:-1] >= 1e-15).all())):
    raise ValueError('times must be strictly increasing in physical buffer order')
  rows[:, adr+1] = n-1
  if times_input is not None: rows[:, adr+2:adr+2+n].copy_(times_input)
  if values_input is not None: rows[:, adr+2+n:adr+2+n+n*dim].copy_(values_input.reshape(count,-1))
  if phase_input is not None: rows[:,adr].copy_(phase_input)
  if ids_t is None: sim.state._history.copy_(rows)
  else: sim.state._history.index_copy_(0, ids_t, rows)
  sim._sync_delay_from_history(ids_t)
  sim._invalidate_after_state_write({'history'})
