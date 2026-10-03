# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Canonical device-resident history buffers for pinned MuJoCo 3.10.

The storage ABI is ``[user, cursor, times[n], values[n, dim]]`` inside
``mjData.history``. Compiled addresses are authoritative. This module does not
invent a second checkpoint representation or perform CPU numerical decisions.
"""

from pathlib import Path

import mujoco
import numpy as np


def lower_history(model):
  """Validate compiled layout and return entry metadata and reset template."""
  entries, parameters, occupied = [], [], []
  initial = mujoco.MjData(model)
  for kind, count in ((0, int(model.nu)), (1, int(model.nsensor))):
    for i in range(count):
      hist = model.actuator_history[i] if kind == 0 else model.sensor_history[i]
      n, interp = map(int, hist)
      delay = float(model.actuator_delay[i] if kind == 0 else model.sensor_delay[i])
      period = float(model.sensor_interval[i, 0]) if kind else 0.0
      if n < 0 or n >= 2**24 or interp not in (0, 1, 2):
        raise ValueError("invalid history count or interpolation order")
      if not np.isfinite(delay) or delay < 0 or not np.isfinite(period) or period < 0:
        raise ValueError("history delay and interval must be finite and nonnegative")
      if not n:
        if delay > 0 or period > 0:
          raise ValueError("delay or interval requires compiled history storage")
        continue
      adr = int(model.actuator_historyadr[i] if kind == 0 else model.sensor_historyadr[i])
      dim = 1 if kind == 0 else int(model.sensor_dim[i])
      outadr = i if kind == 0 else int(model.sensor_adr[i])
      stop = adr + 2 + n*(1+dim)
      if adr < 0 or dim <= 0 or stop > int(model.nhistory):
        raise ValueError("history configuration requires recompiling the model")
      if any(adr < end and stop > start for start, end in occupied):
        raise ValueError("compiled history buffers overlap")
      occupied.append((adr, stop))
      entries.append([kind, adr, n, dim, interp, outadr])
      # Keep the low part of compiled clock constants. Tick eligibility is
      # discrete: rounding an interval before comparing can insert a sample
      # one step earlier than pinned double-precision MuJoCo.
      phase = float(initial.history[adr]) if kind else 0.0
      parameters.append([delay, period, period-float(np.float32(period)),
                         phase, phase-float(np.float32(phase)),
                         delay-float(np.float32(delay))])
  template = np.asarray(initial.history, dtype=np.float32).copy()
  return (np.asarray(entries, dtype=np.int32).reshape(-1, 6),
          np.asarray(parameters, dtype=np.float32).reshape(-1, 6), template)


class DeviceHistory:
  """Read and insert canonical actuator/sensor samples on MPS.

  ``sensor_samples`` reads delayed/held values without modifying history;
  ``record`` inserts raw forward-stage sensor values and held controls only for
  successfully advanced worlds. Public arbitrary-time reads and writes retain
  pinned interpolation and out-of-order insertion semantics.
  """

  def __init__(self, model, batch_size=1, device="mps"):
    import torch
    self._torch = torch
    self.batch_size, self.nhistory = int(batch_size), int(model.nhistory)
    self.nu, self.nsensordata = int(model.nu), int(model.nsensordata)
    self.device = torch.device(device)
    info, params, self.reset_template = lower_history(model)
    self.entries = info
    self._meta = torch.as_tensor(info.reshape(-1), device=self.device)
    self._parameters = torch.as_tensor(params.reshape(-1), device=self.device)
    self._dims = torch.tensor([self.batch_size, len(info), self.nhistory,
                               self.nu, self.nsensordata],
                              dtype=torch.int32, device=self.device)
    # Torch resolves the unindexed MPS alias to mps:0 on allocated tensors.
    self.device = self._dims.device
    source = Path(__file__).with_name("shaders").joinpath("history.metal").read_text()
    self._library = torch.mps.compile_shader(source)

  def _validate(self, history, time):
    torch = self._torch
    if (not isinstance(history, torch.Tensor) or history.device != self.device
        or history.dtype != torch.float32 or not history.is_contiguous()
        or tuple(history.shape) != (self.batch_size, self.nhistory)):
      raise ValueError("history must be contiguous float32 MPS with compiled shape")
    if (not isinstance(time, torch.Tensor) or time.device != self.device
        or time.dtype != torch.float32 or not time.is_contiguous()
        or tuple(time.shape) != (self.batch_size,)):
      raise ValueError("history time must be contiguous float32 MPS per world")

  def samples(self, raw, history, time, *, kind, force_read=False):
    """Read actuator controls or delayed/held sensor values; history unchanged."""
    torch = self._torch
    self._validate(history, time)
    size = self.nu if kind == 0 else self.nsensordata
    if kind not in (0, 1) or (raw.device != self.device or raw.dtype != torch.float32
                             or tuple(raw.shape) != (self.batch_size, size)):
      raise ValueError("history samples require float32 device raw data of matching kind")
    result = raw.contiguous().clone()
    if self.entries.size:
      flags = torch.tensor([kind, int(force_read)], dtype=torch.int32, device=self.device)
      self._library.history_samples(
          history.reshape(-1), time, result.reshape(-1), self._meta,
          self._parameters, self._dims, flags,
          threads=(self.batch_size*len(self.entries),), group_size=(1,))
    return result

  def record(self, controls, sensors, history, time, success):
    """Advance enabled entries, exactly once at the pre-integration time."""
    torch = self._torch
    self._validate(history, time)
    for values, shape in ((controls, (self.batch_size, self.nu)),
                          (sensors, (self.batch_size, self.nsensordata))):
      if (values.device != self.device or values.dtype != torch.float32
          or tuple(values.shape) != shape or not values.is_contiguous()):
        raise ValueError("history recording requires contiguous raw device samples")
    if success.device != self.device or tuple(success.shape) != (self.batch_size,):
      raise ValueError("history recording requires a per-world device success mask")
    if self.entries.size:
      # Empty arrays cannot be passed as MPS shader buffers. No entry of that
      # kind dereferences these placeholders.
      dummy = torch.zeros(1, device=self.device)
      self._library.history_record(
          history.reshape(-1), time, controls.reshape(-1) if self.nu else dummy,
          sensors.reshape(-1) if self.nsensordata else dummy,
          success.to(dtype=torch.int32), self._meta, self._parameters, self._dims,
          threads=(self.batch_size*len(self.entries),), group_size=(1,))
