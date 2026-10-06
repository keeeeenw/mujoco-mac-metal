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


def inverse_constraint_force(rows, qacc, descriptor, *, qacc_low=None,
                            return_low=False):
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
  if qacc_low is None:
    qacc_low = torch.zeros_like(qacc)
  elif (not isinstance(qacc_low, torch.Tensor)
        or tuple(qacc_low.shape) != tuple(qacc.shape)
        or qacc_low.dtype != torch.float32 or qacc_low.device != qacc.device
        or not qacc_low.is_contiguous()):
    raise ValueError("qacc_low must be contiguous float32 with qacc shape")
  if not isinstance(return_low, bool):
    raise TypeError("return_low must be boolean")
  if nr == 0:
    high = torch.zeros_like(qacc)
    return (high, torch.zeros_like(high)) if return_low else high
  packed = rows.get("J_packed")
  layout = rows.get("jacobian_layout")
  pattern = rows.get("jacobian_pattern")
  J = rows.get("J")
  if packed is None:
    if (not isinstance(J, torch.Tensor) or J.dtype != torch.float32
        or J.device != qacc.device or tuple(J.shape) != (b, nr, nv)):
      raise ValueError(f"rows.J must be float32 {(b, nr, nv)} on the query device")
  else:
    from mujoco_metal.constraint_jacobian import (
        PACKED_J_CSR, PACKED_J_DENSE,
        materialize_packed_jacobian_torch)
    if (layout is None or layout.nr != nr or layout.nv != nv
        or not isinstance(packed, torch.Tensor) or packed.dtype != torch.float32
        or packed.device != qacc.device
        or packed.numel() < b * layout.stride_words):
      raise ValueError("rows.J_packed has an invalid compiled layout")
    if layout.mode == PACKED_J_DENSE:
      J = materialize_packed_jacobian_torch(
          torch, packed, b, layout, pattern)
    elif layout.mode != PACKED_J_CSR or pattern is None:
      raise ValueError("sparse query rows require their compiled CSR pattern")
    elif (pattern.nr != nr or pattern.nv != nv
          or pattern.nnz != layout.nnz):
      raise ValueError("rows.jacobian_pattern does not match the packed layout")
  for name, shape in (("R", (b, nr)), ("ar", (b, nr)),
                      ("lo", (b, nr)), ("hi", (b, nr)), ("active", (b, nr))):
    value = rows.get(name)
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
        or value.device != qacc.device or tuple(value.shape) != shape):
      raise ValueError(f"rows.{name} must be float32 {shape} on the query device")
  R = rows["R"]
  ar_low = rows.get("ar_low")
  if ar_low is None:
    ar_low = torch.zeros_like(rows["ar"])
  elif (not isinstance(ar_low, torch.Tensor) or ar_low.dtype != torch.float32
        or ar_low.device != qacc.device or tuple(ar_low.shape) != (b, nr)):
    raise ValueError(f"rows.ar_low must be float32 {(b, nr)} on the query device")
  live = (rows["active"] > 0.5) & (R > 0)
  if packed is not None and layout.mode == PACKED_J_CSR:
    dot_hi, dot_low = _packed_jacobian_pair_matvec(
        packed, layout, pattern, qacc)
    if qacc_low is not None:
      low_hi, low_lo = _packed_jacobian_pair_matvec(
          packed, layout, pattern, qacc_low)
      dot_hi, dot_low = _pair_add(dot_hi, dot_low, low_hi, low_lo)
  else:
    dot_hi, dot_low = _dense_jacobian_pair_matvec(J, qacc)
    if qacc_low is not None:
      low_hi, low_lo = _dense_jacobian_pair_matvec(J, qacc_low)
      dot_hi, dot_low = _pair_add(dot_hi, dot_low, low_hi, low_lo)
  jar, jar_low = _pair_add(dot_hi, dot_low, -rows["ar"], -ar_low)
  safe_R = torch.where(live, R, torch.ones_like(R))
  raw, raw_low = _pair_divide(-jar, -jar_low, safe_R)
  raw = torch.where(live, raw, torch.zeros_like(raw))
  raw_low = torch.where(live, raw_low, torch.zeros_like(raw_low))
  bounded = torch.minimum(torch.maximum(raw, rows["lo"]), rows["hi"])
  above_lower = ((raw > rows["lo"])
                 | ((raw == rows["lo"]) & (raw_low > 0)))
  below_upper = ((raw < rows["hi"])
                 | ((raw == rows["hi"]) & (raw_low < 0)))
  bounded_low = torch.where(above_lower & below_upper, raw_low,
                            torch.zeros_like(raw_low))
  ne = int(descriptor.n_eq_rows)
  if not 0 <= ne <= nr:
    raise ValueError("equality span is outside the canonical row layout")
  if ne:
    bounded[:, :ne] = raw[:, :ne]
    bounded_low[:, :ne] = raw_low[:, :ne]
  force = torch.where(live, bounded, torch.zeros_like(raw))
  force_low = torch.where(live, bounded_low, torch.zeros_like(raw_low))
  nc = int(descriptor.ncontacts_max)
  elliptic = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
  if nc and int(descriptor.cone_type) == elliptic:
    contact_metadata = np.asarray(
        descriptor.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
    if len(contact_metadata) != nc:
      raise ValueError("contact metadata must contain one record per slot")
    friction, mask = rows.get("contact_friction"), rows.get("contact_mask")
    for name, value, shape in (("contact_friction", friction, (nc, 5)),
                              ("contact_mask", mask, (b, nc))):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.device != qacc.device or value.dtype != torch.float32):
        raise ValueError(f"rows.{name} must be float32 {shape} on the query device")
    for slot, (condim, offset, cone) in enumerate(contact_metadata.tolist()):
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
      local_low = jar_low[:, span]
      normal_hi, normal_product_low = _two_prod(local[:, :1], mu[:, None])
      normal_lo_hi, normal_lo_low = _two_prod(
          local_low[:, :1], mu[:, None])
      normal_low_hi, normal_low = _pair_add(
          normal_product_low, torch.zeros_like(normal_product_low),
          normal_lo_hi, normal_lo_low)
      tangent_hi, tangent_product_low = _two_prod(
          local[:, 1:], coeff[None, :])
      tangent_lo_hi, tangent_lo_low = _two_prod(
          local_low[:, 1:], coeff[None, :])
      tangent_low_hi, tangent_low = _pair_add(
          tangent_product_low, torch.zeros_like(tangent_product_low),
          tangent_lo_hi, tangent_lo_low)
      U = torch.cat((normal_hi, tangent_hi), dim=1)
      U_low = torch.cat((normal_low_hi + normal_low,
                         tangent_low_hi + tangent_low), dim=1)
      N, T = U[:, 0], torch.linalg.vector_norm(U[:, 1:], dim=1)
      T_low = ((U[:, 1:] * U_low[:, 1:]).sum(dim=1)
               / T.clamp_min(1e-20))
      muT_hi, muT_low = _two_prod(mu, T)
      top_hi, top_lo = _pair_add(
          N, U_low[:, 0], -muT_hi, -muT_low - mu*T_low)
      top = ((top_hi > 0) | ((top_hi == 0) & (top_lo >= 0))
             | ((T <= 0) & ((N > 0) | ((N == 0) & (U_low[:, 0] >= 0)))))
      bottom_hi, bottom_prod_low = _two_prod(mu, N)
      bottom_hi, bottom_low = _pair_add(
          bottom_hi, bottom_prod_low, T, T_low)
      bottom_hi, bottom_low = _pair_add(
          bottom_hi, bottom_low, torch.zeros_like(bottom_low),
          mu*U_low[:, 0])
      bottom = ((bottom_hi < 0) | ((bottom_hi == 0) & (bottom_low <= 0))
                | ((T <= 0) & ((N < 0) | ((N == 0) & (U_low[:, 0] < 0)))))
      scale = -(N-mu*T)*mu/(rn*(mu.square()*(1+mu.square())).clamp_min(1e-24))
      cone_force = torch.cat((scale[:, None],
          -scale[:, None]/T.clamp_min(1e-20)[:, None]*U[:, 1:]*coeff[None, :]), dim=1)
      cone_force = torch.where(top[:, None], torch.zeros_like(cone_force),
          torch.where(bottom[:, None], raw[:, span], cone_force))
      dU = torch.cat((local_low[:, :1]*mu[:, None],
                      local_low[:, 1:]*coeff[None, :]), dim=1)
      dN = dU[:, 0]
      dT = (U[:, 1:] * dU[:, 1:]).sum(dim=1) / T.clamp_min(1e-20)
      scale_slope = -mu / (rn * (mu.square() * (1 + mu.square())).clamp_min(1e-24))
      dscale = scale_slope * (dN - mu*dT)
      cone_force_low = torch.cat((dscale[:, None],
          -coeff[None, :] * (
              dscale[:, None] * U[:, 1:] / T.clamp_min(1e-20)[:, None]
              + scale[:, None] * dU[:, 1:] / T.clamp_min(1e-20)[:, None]
              - scale[:, None] * U[:, 1:] * dT[:, None]
                / T.clamp_min(1e-20).square()[:, None])), dim=1)
      cone_force_low = torch.where(top[:, None], torch.zeros_like(cone_force_low),
          torch.where(bottom[:, None], raw_low[:, span], cone_force_low))
      force[:, span] = torch.where(active[:, None], cone_force, torch.zeros_like(cone_force))
      force_low[:, span] = torch.where(active[:, None], cone_force_low,
                                       torch.zeros_like(cone_force_low))
  if packed is not None and layout.mode == PACKED_J_CSR:
    out_hi, out_low = _packed_jacobian_transpose_pair_matvec(
        packed, layout, pattern, force, force_low, live)
  else:
    J = torch.where(live[:, :, None], J, torch.zeros_like(J))
    out_hi, out_low = _dense_jacobian_transpose_pair_matvec(
        J, force, force_low)
  return (out_hi, out_low) if return_low else out_hi


def _two_sum(a, b):
  """Error-free sum transform for finite float32 operands."""
  total = a + b
  b_virtual = total - a
  a_virtual = total - b_virtual
  b_roundoff = b - b_virtual
  a_roundoff = a - a_virtual
  return total, a_roundoff + b_roundoff


def _pair_add(a_hi, a_low, b_hi, b_low):
  hi, err = _two_sum(a_hi, b_hi)
  tail = a_low + b_low
  tail = tail + err
  return _two_sum(hi, tail)


def _two_prod(a, b):
  """Float32 product plus a range-scaled Dekker residual."""
  large_limit, small_limit = 1.0e34, 1.0e-18
  large_a, large_b = torch.abs(a) > large_limit, torch.abs(b) > large_limit
  small_a = (torch.abs(a) < small_limit) & (a != 0)
  small_b = (torch.abs(b) < small_limit) & (b != 0)
  a_scale = torch.where(large_a, 2.0**-32,
      torch.where(small_a, 2.0**32, 1.0))
  b_scale = torch.where(large_b, 2.0**-32,
      torch.where(small_b, 2.0**32, 1.0))
  a_scaled, b_scaled = a * a_scale, b * b_scale
  product = a * b
  scaled_product = a_scaled * b_scaled
  splitter = 4097.0
  a_split, b_split = splitter * a_scaled, splitter * b_scaled
  a_hi = a_split - (a_split - a_scaled)
  a_low = a_scaled - a_hi
  b_hi = b_split - (b_split - b_scaled)
  b_low = b_scaled - b_hi
  scaled_error = (((a_hi * b_hi - scaled_product) + a_hi * b_low)
                  + a_low * b_hi) + a_low * b_low
  scale_back = (1.0 / a_scale) * (1.0 / b_scale)
  rescaled_product = scaled_product * scale_back
  error = scaled_error * scale_back + (rescaled_product - product)
  return product, error


def _dense_jacobian_pair_matvec(J, qacc):
  batch, nr, nv = J.shape
  hi = torch.zeros((batch, nr), dtype=qacc.dtype, device=qacc.device)
  low = torch.zeros_like(hi)
  for dof in range(nv):
    product, product_low = _two_prod(J[:, :, dof], qacc[:, dof:dof+1])
    hi, low = _pair_add(hi, low, product, product_low)
  return hi, low


def _pair_divide(numerator_hi, numerator_low, denominator):
  """Return a two-float quotient for a two-float numerator."""
  quotient = numerator_hi / denominator
  product, product_low = _two_prod(quotient, denominator)
  remainder_hi, remainder_low = _pair_add(
      numerator_hi, torch.zeros_like(numerator_hi), -product, -product_low)
  remainder_hi, remainder_low = _pair_add(
      remainder_hi, remainder_low, numerator_low,
      torch.zeros_like(numerator_low))
  correction = (remainder_hi + remainder_low) / denominator
  return _two_sum(quotient, correction)


def _packed_jacobian_pair_matvec(packed, layout, pattern, qacc):
  batch = qacc.shape[0]
  hi = torch.zeros((batch, layout.nr), dtype=qacc.dtype, device=qacc.device)
  low = torch.zeros_like(hi)
  for row in range(layout.nr):
    begin, end = int(pattern.row_offsets[row]), int(pattern.row_offsets[row+1])
    if begin < 0 or end < begin or end > layout.nnz:
      raise ValueError("compiled CSR row pointer is outside its value array")
    columns = pattern.columns[begin:end]
    if (np.any(columns < 0) or np.any(columns >= layout.nv)
        or np.any(columns[1:] <= columns[:-1])):
      raise ValueError("compiled CSR columns are invalid or not strictly sorted")
    if begin == end:
      continue
    for world in range(batch):
      base = world * layout.stride_words + layout.values_offset
      values = packed[base + begin:base + end]
      acc_hi = torch.zeros((), dtype=qacc.dtype, device=qacc.device)
      acc_low = torch.zeros_like(acc_hi)
      for k, col in enumerate(columns.tolist()):
        p, e = _two_prod(values[k], qacc[world, int(col)])
        acc_hi, acc_low = _pair_add(acc_hi, acc_low, p, e)
      hi[world, row], low[world, row] = acc_hi, acc_low
  return hi, low


def _dense_jacobian_transpose_pair_matvec(J, force_hi, force_low):
  batch, nr, nv = J.shape
  hi = torch.zeros((batch, nv), dtype=force_hi.dtype, device=force_hi.device)
  low = torch.zeros_like(hi)
  for row in range(nr):
    p, pe = _two_prod(J[:, row, :], force_hi[:, row:row+1])
    pl, ple = _two_prod(J[:, row, :], force_low[:, row:row+1])
    hi, low = _pair_add(hi, low, p, pe)
    hi, low = _pair_add(hi, low, pl, ple)
  return hi, low


def _packed_jacobian_transpose_pair_matvec(packed, layout, pattern,
                                           force_hi, force_low, live):
  batch = force_hi.shape[0]
  hi = torch.zeros((batch, layout.nv), dtype=force_hi.dtype, device=force_hi.device)
  low = torch.zeros_like(hi)
  for row in range(layout.nr):
    begin, end = int(pattern.row_offsets[row]), int(pattern.row_offsets[row+1])
    if begin < 0 or end < begin or end > layout.nnz:
      raise ValueError("compiled CSR row pointer is outside its value array")
    columns = pattern.columns[begin:end]
    if (np.any(columns < 0) or np.any(columns >= layout.nv)
        or np.any(columns[1:] <= columns[:-1])):
      raise ValueError("compiled CSR columns are invalid or not strictly sorted")
    if begin == end:
      continue
    for world in range(batch):
      base = world * layout.stride_words + layout.values_offset
      for k, col in enumerate(columns.tolist()):
        col = int(col)
        j = torch.where(live[world, row], packed[base + begin + k],
                        torch.zeros((), dtype=packed.dtype,
                                    device=packed.device))
        ph, pl = _two_prod(j, force_hi[world, row])
        lh, ll = _two_prod(j, force_low[world, row])
        sum_hi, sum_low = _pair_add(hi[world, col], low[world, col], ph, pl)
        hi[world, col], low[world, col] = _pair_add(sum_hi, sum_low, lh, ll)
  return hi, low


def _packed_jacobian_matvec(packed, layout, pattern, qacc, aref):
  """Compute J*qacc for a query without materializing sparse J."""
  b, nr = qacc.shape[0], layout.nr
  result = torch.zeros((b, nr), dtype=qacc.dtype, device=qacc.device)
  for row in range(nr):
    begin = int(pattern.row_offsets[row])
    end = int(pattern.row_offsets[row + 1])
    if begin < 0 or end < begin or end > layout.nnz:
      raise ValueError("compiled CSR row pointer is outside its value array")
    if begin == end:
      continue
    host_columns = pattern.columns[begin:end]
    if (np.any(host_columns < 0) or np.any(host_columns >= layout.nv)
        or np.any(host_columns[1:] <= host_columns[:-1])):
      raise ValueError("compiled CSR columns are invalid or not strictly sorted")
    columns = torch.as_tensor(host_columns.copy(),
                              dtype=torch.int64, device=qacc.device)
    gathered = qacc.index_select(1, columns)
    for world in range(b):
      base = world * layout.stride_words + layout.values_offset
      values = packed[base + begin:base + end]
      result[world, row] = torch.sum(gathered[world] * values)
  return result - aref


def _packed_jacobian_transpose_matvec(packed, layout, pattern, force):
  """Compute J.T*force by compiled row supports, without dense scratch."""
  b = force.shape[0]
  result = torch.zeros((b, layout.nv), dtype=force.dtype, device=force.device)
  for row in range(layout.nr):
    begin = int(pattern.row_offsets[row])
    end = int(pattern.row_offsets[row + 1])
    if begin < 0 or end < begin or end > layout.nnz:
      raise ValueError("compiled CSR row pointer is outside its value array")
    if begin == end:
      continue
    host_columns = pattern.columns[begin:end]
    if (np.any(host_columns < 0) or np.any(host_columns >= layout.nv)
        or np.any(host_columns[1:] <= host_columns[:-1])):
      raise ValueError("compiled CSR columns are invalid or not strictly sorted")
    columns = torch.as_tensor(host_columns.copy(),
                              dtype=torch.int64, device=force.device)
    for world in range(b):
      base = world * layout.stride_words + layout.values_offset
      values = packed[base + begin:base + end] * force[world, row]
      result[world].index_add_(0, columns, values)
  return result
