# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native tangent-coordinate quaternion derivatives from pinned MuJoCo 3.10."""
from numbers import Integral, Real
from pathlib import Path
import math

import numpy as np

_SHADER = Path(__file__).parent / 'shaders' / 'quaternion_derivatives.metal'


def validate_scale(scale):
  """Validate a public scalar without starting the device runtime."""
  if isinstance(scale,(bool,np.bool_)) or not isinstance(scale,Real):
    raise ValueError('scale must be a finite float32 scalar')
  with np.errstate(over='ignore',under='ignore'):
    rounded = np.float32(scale)
  if not math.isfinite(float(rounded)) or (scale != 0 and rounded == 0):
    raise ValueError('scale must be representable as finite float32')
  return float(rounded)


class MetalQuaternionDerivatives:
  """Owned analytical matrices; no simulation or finite-difference rollout.

  sub_quat returns Da/Db[B,3,3] w.r.t local tangent perturbations, not four
  ambient quaternion components. quat_integrate returns Dquat/Dvel[B,3,3]
  and Dscale[B,3]. Dvel is w.r.t scaled angular velocity, following the pinned
  function; multiply by scale to obtain derivative w.r.t unscaled velocity.
  Each result includes int32 status[B]: 1 input, 2 arithmetic nonfinite.
  """

  def __init__(self, batch_size, *, memory_budget_bytes=1 << 30):
    if (isinstance(batch_size, (bool,np.bool_)) or not isinstance(batch_size,Integral)
        or not 0 < batch_size <= (2**31-1)//21):
      raise ValueError('batch_size must be positive and addressing must fit signed int32')
    if (isinstance(memory_budget_bytes,(bool,np.bool_))
        or not isinstance(memory_budget_bytes,Integral) or memory_budget_bytes<0):
      raise ValueError('memory_budget_bytes must be a nonnegative integer')
    self.batch_size = int(batch_size)
    self.required_bytes = (4 + self.batch_size * 22) * 4
    if self.required_bytes > memory_budget_bytes:
      raise ValueError('quaternion derivative budget exceeded before device allocation')
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
      raise RuntimeError('PyTorch MPS shader compilation is unavailable')
    self._torch = torch
    self._device = torch.device('mps')
    self._batch = torch.tensor([self.batch_size],dtype=torch.int32,device=self._device)
    self._modes = tuple(torch.tensor([i],dtype=torch.int32,device=self._device) for i in (0,1))
    self._scale = torch.zeros(1,dtype=torch.float32,device=self._device)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.quaternion_derivatives

  def _input(self, value, width, name):
    torch = self._torch
    if (not isinstance(value,torch.Tensor) or value.dtype != torch.float32
        or value.device.type != 'mps' or not value.is_contiguous()
        or tuple(value.shape) != (self.batch_size,width)):
      raise ValueError(f'{name} must be contiguous MPS float32 [B,{width}]')

  def _run(self, first, second, scale, mode):
    if mode == 0:
      self._input(first,4,'qa')
      self._input(second,4,'qb')
    else:
      self._input(first,3,'vel')
      second = first  # Valid unused binding; no dummy allocation.
    rounded = validate_scale(scale)
    torch = self._torch
    packed = torch.empty((self.batch_size,21),dtype=torch.float32,device=self._device)
    status = torch.empty(self.batch_size,dtype=torch.int32,device=self._device)
    self._scale.fill_(float(rounded))
    self._kernel(first.reshape(-1),second.reshape(-1),self._batch,self._scale,
        self._modes[mode],packed.reshape(-1),status,
        threads=(self.batch_size,),group_size=(min(self.batch_size,128),))
    # Views share an owned per-call allocation, never reusable program scratch.
    if mode == 0:
      return {'Da':packed[:,:9].reshape(-1,3,3),
              'Db':packed[:,9:18].reshape(-1,3,3),'status':status}
    return {'Dquat':packed[:,:9].reshape(-1,3,3),
            'Dvel':packed[:,9:18].reshape(-1,3,3),
            'Dscale':packed[:,18:21],'status':status}

  def sub_quat(self,qa,qb):
    return self._run(qa,qb,0,0)

  def quat_integrate(self,vel,scale):
    return self._run(vel,None,scale,1)
