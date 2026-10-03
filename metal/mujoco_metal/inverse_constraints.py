# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned inverse constraint-cost gradient on canonical device row views.

MuJoCo 3.10 engine_core_constraint.c:mj_constraintUpdate defines bilateral,
bounded scalar and elliptic contact costs. This query reduction does not run a
forward optimizer, rebuild rows, or reconstruct the generalized mass matrix.
"""
import mujoco
import numpy as np
import torch


def inverse_constraint_force(rows, qacc, descriptor):
  """Return owned generalized constraint force ``J.T @ -grad(cost)``.

  Row inputs are borrowed device views from one coherent assembly. Activity
  selects the reserved rows; positive R alone does not make a row live.
  Static compiled contact metadata lives on the host; numerical work remains
  on the input tensors' device. No CPU physics or numerical readback occurs.
  """
  if not isinstance(rows, dict) or not isinstance(qacc, torch.Tensor):
    raise TypeError("rows must be a canonical dictionary and qacc a tensor")
  nr, nv = int(descriptor.nr), int(descriptor.nv)
  if qacc.ndim != 2 or qacc.shape[1] != nv or qacc.dtype != torch.float32:
    raise ValueError("qacc must be float32 [batch,nv]")
  b = int(qacc.shape[0])
  if nr == 0:
    return torch.zeros_like(qacc)
  for name, shape in (("J", (b, nr, nv)), ("R", (b, nr)), ("ar", (b, nr)),
                      ("lo", (b, nr)), ("hi", (b, nr)), ("active", (b, nr))):
    value = rows.get(name)
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
        or value.device != qacc.device or tuple(value.shape) != shape):
      raise ValueError(f"rows.{name} must be float32 {shape} on the query device")
  J, R = rows["J"], rows["R"]
  live = (rows["active"] > 0.5) & (R > 0)
  jar = torch.bmm(J, qacc.unsqueeze(-1)).squeeze(-1) - rows["ar"]
  safe_R = torch.where(live, R, torch.ones_like(R))
  raw = torch.where(live, -jar/safe_R, torch.zeros_like(jar))
  bounded = torch.minimum(torch.maximum(raw, rows["lo"]), rows["hi"])
  ne = int(descriptor.n_eq_rows)
  if not 0 <= ne <= nr:
    raise ValueError("equality span is outside the canonical row layout")
  if ne:
    bounded[:, :ne] = raw[:, :ne]
  force = torch.where(live, bounded, torch.zeros_like(raw))
  nc = int(descriptor.ncontacts_max)
  elliptic = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
  if nc and int(descriptor.cone_type) == elliptic:
    packed = np.asarray(descriptor.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
    if len(packed) != nc:
      raise ValueError("contact metadata must contain one record per slot")
    friction, mask = rows.get("contact_friction"), rows.get("contact_mask")
    for name, value, shape in (("contact_friction", friction, (nc, 5)),
                              ("contact_mask", mask, (b, nc))):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.device != qacc.device or value.dtype != torch.float32):
        raise ValueError(f"rows.{name} must be float32 {shape} on the query device")
    for slot, (condim, offset, cone) in enumerate(packed.tolist()):
      if cone != elliptic or condim == 1:
        continue
      if condim not in (3, 4, 6):
        raise ValueError("unsupported elliptic contact dimension")
      row0 = int(descriptor.nr_joint) + int(offset)
      if row0 < 0 or row0+condim > nr:
        raise ValueError("elliptic contact span is outside the canonical row layout")
      active = live[:, row0] & (mask[:, slot] > .5)
      rn = torch.where(active, R[:, row0], torch.ones_like(R[:, row0]))
      rt = torch.where(active, R[:, row0+1], rn)
      # Pinned mj_makeImpedance scales the cone slope with its row R ratio.
      mu = friction[slot, 0] * torch.sqrt(rt.clamp_min(0)/rn)
      coeff = friction[slot, :condim-1]
      span = slice(row0, row0+condim)
      local = jar[:, span]
      U = torch.cat((local[:, :1]*mu[:, None], local[:, 1:]*coeff[None, :]), dim=1)
      N, T = U[:, 0], torch.linalg.vector_norm(U[:, 1:], dim=1)
      top = (N >= mu*T) | ((T <= 0) & (N >= 0))
      bottom = (mu*N+T <= 0) | ((T <= 0) & (N < 0))
      scale = -(N-mu*T)*mu/(rn*(mu.square()*(1+mu.square())).clamp_min(1e-24))
      cone_force = torch.cat((scale[:, None],
          -scale[:, None]/T.clamp_min(1e-20)[:, None]*U[:, 1:]*coeff[None, :]), dim=1)
      cone_force = torch.where(top[:, None], torch.zeros_like(cone_force),
          torch.where(bottom[:, None], raw[:, span], cone_force))
      force[:, span] = torch.where(active[:, None], cone_force, torch.zeros_like(cone_force))
  return torch.bmm(J.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
