# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned mjtState vector layout; model sizes are legitimate host metadata."""
import numbers

import mujoco


def _signature(value):
  if isinstance(value, bool) or not isinstance(value, (numbers.Integral, mujoco.mjtState)):
    raise TypeError("state signature must be an integer mjtState bitmask")
  value = int(value)
  if not 0 <= value < (1 << int(mujoco.mjtState.mjNSTATE)):
    raise ValueError("state signature contains bits outside pinned mjtState")
  return value


def state_vector_layout(model, signature):
  """Ordered (bit, offset, width) tuples, including zero-width elements."""
  signature = _signature(signature)
  widths = (1, model.nq, model.nv, model.na, model.nhistory, model.nv,
            model.nu, model.nv, 6*model.nbody, model.neq, 3*model.nmocap,
            4*model.nmocap, model.nuserdata, model.npluginstate)
  layout, offset = [], 0
  for index, width in enumerate(widths):
    if signature & (1 << index):
      width = int(width)
      layout.append((1 << index, offset, width))
      offset += width
  return tuple(layout), offset


def extract_state_vector(model, source, source_signature, destination_signature):
  """Owned [batch,size(dst)] device extraction in pinned state-field order."""
  import torch
  source_signature = _signature(source_signature)
  destination_signature = _signature(destination_signature)
  if source_signature & destination_signature != destination_signature:
    raise ValueError("destination state signature is not a subset of source")
  layout, width = state_vector_layout(model, source_signature)
  if (not isinstance(source, torch.Tensor) or source.ndim != 2
      or source.shape[1] != width or not source.is_contiguous()
      or source.dtype not in (torch.float32, torch.float64)):
    raise ValueError("source must be a contiguous floating [batch,state_size] tensor")
  selected = [source[:, offset:offset+size] for bit, offset, size in layout
              if destination_signature & bit]
  # No value readback or numeric decisions: even nonfinite state values are
  # copied exactly, matching the upstream extraction utility.
  return torch.cat(selected, dim=1) if selected else source.new_empty((source.shape[0], 0))
