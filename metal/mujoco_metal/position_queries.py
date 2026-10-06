# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned native position-manifold utilities for the pinned joint conventions."""
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import math

import numpy as np

_SHADER = Path(__file__).parent / 'shaders' / 'position_queries.metal'
_I32 = 2**31 - 1


@dataclass(frozen=True)
class PositionTopology:
  nq: int
  nv: int
  joints: np.ndarray


def lower_position_topology(model):
  """Copy the complete compiled joint map without initializing GPU physics."""
  nq, nv, nj = int(model.nq), int(model.nv), int(model.njnt)
  if any(x < 0 or x > _I32 for x in (nq, nv, nj, nj * 3)):
    raise ValueError('position topology exceeds signed int32')
  joints = np.column_stack((model.jnt_type, model.jnt_qposadr,
                           model.jnt_dofadr)).astype(np.int32)
  if joints.shape != (nj, 3):
    raise ValueError('joint metadata must have shape [njnt,3]')
  qcovered, vcovered = 0, 0
  for kind, qadr, vadr in joints:
    if kind not in (0, 1, 2, 3):
      raise ValueError('unknown joint type')
    qwidth, vwidth = ((7, 6) if kind == 0 else
                      (4, 3) if kind == 1 else (1, 1))
    if (qadr != qcovered or vadr != vcovered
        or qadr + qwidth > nq or vadr + vwidth > nv):
      raise ValueError('invalid or overlapping joint addresses')
    qcovered += qwidth
    vcovered += vwidth
  if qcovered != nq or vcovered != nv:
    raise ValueError('joint addresses must cover every qpos and DOF')
  # Immutable bytes also prevent mutation through a writable base array.
  joints = np.frombuffer(joints.tobytes(), np.int32).reshape(nj, 3)
  return PositionTopology(nq, nv, joints)


class MetalPositionQueries:
  """Native integrate/differentiate/normalize with owned per-call outputs.

  Methods consume contiguous float32 MPS [B,nq]/[B,nv] inputs and return
  {'qpos' or 'qvel': owned tensor, 'status': int32[B]}. No simulation state
  changes. Status 1 denotes nonfinite input and 2 arithmetic overflow. Failed
  position rows preserve their input; failed velocity rows are zero. dt is a
  finite float32 scalar, negative values are supported, and differentiation
  rejects zero. Zero quaternions follow pinned normalization to identity.
  """

  def __init__(self, model, batch_size, *, memory_budget_bytes=1 << 30):
    if (isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, Integral) or batch_size < 1):
      raise ValueError('batch_size must be a positive integer')
    if (isinstance(memory_budget_bytes, (bool, np.bool_))
        or not isinstance(memory_budget_bytes, Integral)
        or memory_budget_bytes < 0):
      raise ValueError('memory_budget_bytes must be a nonnegative integer')
    self.topology = lower_position_topology(model)
    self.batch_size = int(batch_size)
    self.nq, self.nv = self.topology.nq, self.topology.nv
    nj = len(self.topology.joints)
    if any(x > _I32 for x in (self.batch_size, self.nq, self.nv, nj,
                             self.batch_size * self.nq,
                             self.batch_size * self.nv, nj * 3)):
      raise ValueError('position query addressing exceeds signed int32')
    # Metadata, dispatch constants/dummy, largest owned result, and status.
    self.required_bytes = (max(3 * nj, 1) + 4 + 1 + 3 + 1
                           + max(self.batch_size * max(self.nq, self.nv), 1)
                           + self.batch_size) * 4
    if self.required_bytes > memory_budget_bytes:
      raise ValueError('position query memory budget exceeded before device allocation')
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, 'compile_shader'):
      raise RuntimeError('PyTorch MPS shader compilation is unavailable')
    self._torch = torch
    self._device = torch.device('mps')
    self._joints = torch.as_tensor(
        self.topology.joints.copy().reshape(-1) if nj else np.zeros(1, np.int32),
        dtype=torch.int32, device=self._device)
    self._dims = torch.tensor([self.batch_size, self.nq, self.nv, nj],
                              dtype=torch.int32, device=self._device)
    self._dt = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._modes = tuple(torch.tensor([mode], dtype=torch.int32, device=self._device)
                        for mode in range(3))
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.position_queries

  def _input(self, value, width, name):
    torch = self._torch
    if (not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (self.batch_size, width)
        or value.dtype != torch.float32 or value.device.type != 'mps'
        or not value.is_contiguous()):
      raise ValueError(f'{name} must be contiguous MPS float32 [B,{width}]')

  def _run(self, first, second, dt, mode):
    self._input(first, self.nq, 'qpos')
    if mode < 2:
      self._input(second, self.nv if mode == 0 else self.nq,
                  'qvel' if mode == 0 else 'qpos2')
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real):
      raise ValueError('dt must be a finite float32 scalar')
    with np.errstate(over='ignore', under='ignore'):
      rounded = np.float32(dt)
    if not math.isfinite(float(rounded)) or (dt != 0 and rounded == 0):
      raise ValueError('dt must be representable as finite float32')
    if mode == 1 and rounded == 0:
      raise ValueError('differentiate dt must be nonzero')
    torch = self._torch
    width = self.nv if mode == 1 else self.nq
    backing = torch.empty(max(self.batch_size * width, 1), dtype=torch.float32,
                          device=self._device)
    output = backing[:self.batch_size * width].reshape(self.batch_size, width)
    status = torch.empty(self.batch_size, dtype=torch.int32, device=self._device)
    self._dt.fill_(float(rounded))
    self._kernel(first.reshape(-1),
        second.reshape(-1) if second is not None else self._dummy,
        self._joints, self._dims, self._dt, self._modes[mode], backing, status,
        threads=(self.batch_size,), group_size=(min(self.batch_size, 128),))
    return {'qvel' if mode == 1 else 'qpos': output, 'status': status}

  def integrate(self, qpos, qvel, dt):
    return self._run(qpos, qvel, dt, 0)

  def differentiate(self, qpos1, qpos2, dt):
    return self._run(qpos1, qpos2, dt, 1)

  def normalize(self, qpos):
    return self._run(qpos, None, 0, 2)
