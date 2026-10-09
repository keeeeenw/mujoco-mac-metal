# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Owned residual words for persistent position and velocity state.

High tensors remain the public float32 projection. Packed residual planes are
qpos low, qpos tail, qvel low and qvel tail, per world. No device value is read
on the host when reconciling direct high-tensor writes. A replacement tensor
starts a new represented input; controlled integration commits bind its
residuals explicitly instead of inferring them from tensor identity.
"""
from pathlib import Path
import numpy as np


def pack_residuals(qpos, qvel):
  planes=[]
  for source in (qpos, qvel):
    from mujoco_metal.model import require_triple_word_domain
    raw=np.asarray(source,dtype=np.float64)
    value=require_triple_word_domain(raw,what='retained state').reshape(raw.shape)
    high=value.astype(np.float32).astype(np.float64)
    low=(value-high).astype(np.float32)
    tail=(value-high-low.astype(np.float64)).astype(np.float32)
    planes.extend((low,tail))
  return np.concatenate(planes,axis=1)


def validate_residuals(qpos, qvel, words):
  nq,nv=qpos.shape[1],qvel.shape[1]
  raw=np.asarray(words)
  if raw.shape!=(qpos.shape[0],2*nq+2*nv) or raw.dtype.kind not in 'fiu':
    raise ValueError('precision_words has invalid shape or dtype')
  raw=np.asarray(raw,dtype=np.float32)
  if not np.all(np.isfinite(raw)):
    raise ValueError('precision_words must be finite')
  pos=qpos.astype(np.float64)+raw[:,:nq].astype(np.float64)+raw[:,nq:2*nq].astype(np.float64)
  vel=qvel.astype(np.float64)+raw[:,2*nq:2*nq+nv].astype(np.float64)+raw[:,2*nq+nv:].astype(np.float64)
  if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
    raise ValueError('represented state must be finite')
  for value,high in ((pos,qpos),(vel,qvel)):
    if not np.array_equal(value.astype(np.float32).view(np.uint32),high.view(np.uint32)):
      raise ValueError('precision_words do not preserve the high projection')
  if not np.array_equal(pack_residuals(pos,vel).view(np.uint32),raw.view(np.uint32)):
    raise ValueError('precision_words must use canonical residual planes')
  return raw.copy(),pos,vel


class RetainedState:
  """A fixed backing allocation, including device-side mutation witnesses."""
  def __init__(self,qpos,qvel,source_qpos=None,source_qvel=None):
    import torch
    self._torch=torch
    self.nq,self.nv=qpos.shape[1],qvel.shape[1]
    self.batch=qpos.shape[0]
    self.words=torch.zeros((self.batch,2*self.nq+2*self.nv),
                          dtype=torch.float32,device=qpos.device)
    if source_qpos is not None:
      self.words.copy_(torch.as_tensor(pack_residuals(source_qpos,source_qvel),
                                      device=qpos.device))
    self.qpos_shadow=qpos.clone();self.qvel_shadow=qvel.clone()
    self._qpos,self._qvel=qpos,qvel
    self._kernel=None
    self._dummy=torch.zeros(1,dtype=torch.float32,device=qpos.device)
    self.dims=torch.tensor([self.nq,self.nv,self.batch,0],dtype=torch.int32,device=qpos.device)
    if qpos.device.type=='mps':
      self._library=torch.mps.compile_shader(
          Path(__file__).with_name('shaders').joinpath('retained_state.metal').read_text())
      self._kernel=self._library.reconcile_retained_state

  def synchronize(self,qpos,qvel):
    """Invalidate only changed rows, even without mutation counters."""
    replacement=int(qpos is not self._qpos)+2*int(qvel is not self._qvel)
    if self._kernel is not None:
      self.dims[3]=replacement
      self._kernel(qpos.reshape(-1) if self.nq else self._dummy,
                   qvel.reshape(-1) if self.nv else self._dummy,
                   self.words.reshape(-1) if self.words.numel() else self._dummy,
                   self.qpos_shadow.reshape(-1) if self.nq else self._dummy,
                   self.qvel_shadow.reshape(-1) if self.nv else self._dummy,self.dims,
                   threads=(self.batch,),group_size=(1,))
    else:
      torch=self._torch
      pos_changed=(qpos.view(torch.int32)!=self.qpos_shadow.view(torch.int32)).any(dim=1)
      vel_changed=(qvel.view(torch.int32)!=self.qvel_shadow.view(torch.int32)).any(dim=1)
      if replacement&1: pos_changed.fill_(True)
      if replacement&2: vel_changed.fill_(True)
      p=self.words[:,:2*self.nq];v=self.words[:,2*self.nq:]
      p.copy_(torch.where(pos_changed[:,None],0.,p))
      v.copy_(torch.where(vel_changed[:,None],0.,v))
      self.qpos_shadow.copy_(qpos);self.qvel_shadow.copy_(qvel)
    self._qpos,self._qvel=qpos,qvel
    return self.words

  def bind(self,qpos,qvel):
    """Record a controlled commit whose residuals have already been written."""
    self._qpos,self._qvel=qpos,qvel
    self.qpos_shadow.copy_(qpos);self.qvel_shadow.copy_(qvel)

  def commit(self,qpos,qvel,words,success):
    self.words.copy_(self._torch.where(success[:,None],words,self.words))
    self.bind(qpos,qvel)
