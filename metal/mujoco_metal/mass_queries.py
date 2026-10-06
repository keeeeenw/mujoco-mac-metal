# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Device mass queries with the pinned descending L' D L convention.

Inputs are validated smooth-stage tensors. Queries allocate owned outputs;
these are not a stepping solver or a reconstruction of global sparse mass for
training. Component inputs remain component-local except explicit ``full_mass``.
"""
import numpy as np


def _mass_blocks(dynamics):
  import torch
  if 'mass_matrix' in dynamics:
    mass = dynamics['mass_matrix']
    yield None, mass
    return
  layout, flat = dynamics['mass_block_layout'], dynamics['mass_blocks']
  armature = dynamics.get('tendon_armature_blocks')
  for component in range(int(layout['ncomponent'])):
    width = int(layout['component_dofnum'][component])
    begin = int(layout['component_dof_offsets'][component])
    ids = np.asarray(layout['component_dof_ids'][begin:begin+width], dtype=np.int64)
    offset = int(layout['component_mass_offsets'][component])
    block = flat[:, offset:offset+width*width].reshape(flat.shape[0], width, width)
    if armature is not None:
      block = block + armature[:, offset:offset+width*width].reshape_as(block)
    yield torch.as_tensor(ids.copy(), device=flat.device), block


def _shape(dynamics):
  if 'mass_matrix' in dynamics:
    value = dynamics['mass_matrix']
    return value, value.shape[-1]
  return dynamics['mass_blocks'], len(dynamics['mass_block_layout']['dof_component'])


def _rhs(dynamics, value):
  import torch
  reference, nv = _shape(dynamics)
  if (not isinstance(value, torch.Tensor) or value.ndim not in (2, 3)
      or value.shape[0] != reference.shape[0] or value.shape[-1] != nv
      or value.device != reference.device or value.dtype != reference.dtype):
    raise ValueError('vector must be on the mass device with shape [batch,nv] or [batch,nrhs,nv]')
  return value.unsqueeze(1) if value.ndim == 2 else value


def full_mass(dynamics):
  """Return an owned dense matrix only at this explicit query boundary."""
  import torch
  reference, nv = _shape(dynamics)
  result = torch.zeros((reference.shape[0], nv, nv), dtype=reference.dtype,
                       device=reference.device)
  for ids, mass in _mass_blocks(dynamics):
    if ids is None:
      result.copy_(mass)
    else:
      result[:, ids[:, None], ids[None, :]] = mass
  return result


def mass_product(dynamics, vector):
  """Apply the selected mass layout; never construct global sparse mass."""
  import torch
  rhs = _rhs(dynamics, vector)
  result = torch.zeros_like(rhs)
  for ids, mass in _mass_blocks(dynamics):
    local = rhs if ids is None else rhs.index_select(2, ids)
    value = torch.bmm(local, mass.transpose(1, 2))
    if ids is None:
      result.copy_(value)
    else:
      result.index_copy_(2, ids, value)
  return result[:, 0] if vector.ndim == 2 else result


def _factor(mass):
  """Factor descending, masking failed worlds before any pivot division."""
  import torch
  b, nv, _ = mass.shape
  factor = torch.eye(nv, dtype=mass.dtype, device=mass.device).expand(b, nv, nv).clone()
  work, diagonal = mass.clone(), torch.zeros((b, nv), dtype=mass.dtype, device=mass.device)
  if not nv:
    # An empty mass has no invalid pivots. MPS empty ``all`` reductions do
    # not reliably implement this vacuous truth, so return it explicitly.
    return factor, diagonal, torch.ones(b, dtype=torch.bool, device=mass.device)
  scale = mass.abs().amax(dim=(1, 2)) if nv else torch.zeros(b, device=mass.device)
  good = torch.isfinite(mass).all(dim=(1, 2))
  if nv:
    asym = (mass - mass.transpose(1, 2)).abs().amax(dim=(1, 2))
    good &= asym <= scale * (32 * torch.finfo(mass.dtype).eps)
  for k in range(nv-1, -1, -1):
    pivot = work[:, k, k]
    good &= torch.isfinite(pivot) & (pivot > 0)
    safe = torch.where(good, pivot, torch.ones_like(pivot))
    diagonal[:, k] = safe
    if k:
      row = torch.where(good[:, None], work[:, k, :k] / safe[:, None],
                        torch.zeros_like(work[:, k, :k]))
      good &= torch.isfinite(row).all(dim=1)
      row = torch.where(good[:, None], row, torch.zeros_like(row))
      factor[:, k, :k] = row
      work[:, :k, :k] -= safe[:, None, None] * row[:, :, None] * row[:, None, :]
  return factor, diagonal, good


def factor_mass(dynamics):
  """Return owned component factors with the pinned ``M = L' D L`` order.

  Each entry in ``blocks`` contains device ``dof_ids``, unit lower-triangular
  ``L``, ``D`` and ``Dinv``. Dense inputs produce one block; component inputs
  retain their local blocks, including tendon armature, without constructing
  a global dense matrix. ``status`` is int32[batch]: 0 for a finite SPD factor
  and 2 for failure. All factor outputs of a failed world are zero, even when
  only one of its components failed. Inputs and prepared solver factors are
  never mutated. This explicit query does not change the stepping factor.
  """
  import torch
  reference, nv = _shape(dynamics)
  good_world = torch.ones(reference.shape[0], dtype=torch.bool,
                          device=reference.device)
  blocks = []
  for ids, mass in _mass_blocks(dynamics):
    lower, diagonal, good = _factor(mass)
    inverse = torch.reciprocal(diagonal)
    if diagonal.shape[1]:
      good &= torch.isfinite(inverse).all(dim=1)
    good_world &= good
    blocks.append({
        'dof_ids': (torch.arange(nv, device=reference.device) if ids is None
                    else ids.clone()),
        'L': lower, 'D': diagonal, 'Dinv': inverse,
    })
  for block in blocks:
    block['L'] = torch.where(good_world[:, None, None], block['L'],
                             torch.zeros_like(block['L']))
    for name in ('D', 'Dinv'):
      block[name] = torch.where(good_world[:, None], block[name],
                                torch.zeros_like(block[name]))
  status = torch.where(good_world,
      torch.zeros_like(good_world, dtype=torch.int32),
      torch.full_like(good_world, 2, dtype=torch.int32))
  return {'blocks': tuple(blocks), 'status': status, 'nv': nv}


def mass_factor_query(dynamics, vector, operation):
  """Return owned (value,status) for solve, sqrt product, or half solve.

  ``sqrt`` is sqrt(D)*L*v, not the symmetric principal square root.
  ``half_solve`` is sqrt(inv(D))*inv(L')*v, matching pinned mj_solveM2.
  Status 2 means nonfinite/non-SPD mass or overflow; failed worlds return zero.
  """
  import torch
  if operation not in ('solve', 'sqrt', 'half_solve'):
    raise ValueError('unknown mass factor query')
  rhs = _rhs(dynamics, vector)
  result = torch.zeros_like(rhs)
  good_world = (torch.isfinite(rhs).all(dim=(1, 2))
                if rhs.shape[1] and rhs.shape[2] else
                torch.ones(rhs.shape[0], dtype=torch.bool, device=rhs.device))
  for ids, mass in _mass_blocks(dynamics):
    local = rhs if ids is None else rhs.index_select(2, ids)
    factor, diagonal, good = _factor(mass)
    nv = mass.shape[-1]
    value = local.clone()
    if operation == 'sqrt':
      value = torch.bmm(value, factor.transpose(1, 2)) * diagonal.sqrt()[:, None]
    else:
      # inv(L')*rhs: descend in the pinned generalized-coordinate order.
      for k in range(nv-1, 0, -1):
        value[:, :, :k] -= value[:, :, k:k+1].clone() * factor[:, None, k, :k]
      if operation == 'half_solve':
        value /= diagonal.sqrt()[:, None]
      else:
        value /= diagonal[:, None]
        for k in range(1, nv):
          value[:, :, k] -= (value[:, :, :k] * factor[:, None, k, :k]).sum(dim=2)
    if value.shape[1] and value.shape[2]:
      good &= torch.isfinite(value).all(dim=(1, 2))
    good_world &= good
    if ids is None:
      result.copy_(value)
    else:
      result.index_copy_(2, ids, value)
  result = torch.where(good_world[:, None, None], result, torch.zeros_like(result))
  status = torch.where(good_world, torch.zeros_like(good_world, dtype=torch.int32),
                        torch.full_like(good_world, 2, dtype=torch.int32))
  return (result[:, 0] if vector.ndim == 2 else result), status
