# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Stable device compaction maps for broadphase, slots, and solver rows.

The logical descriptor order remains the public identity for contact records,
retained multipliers, force queries, and certificates. These maps only choose
the compact execution order. A set bit at logical index ``i`` appears once at
its monotonically increasing position in ``packed_to_logical``; the inverse
map contains ``-1`` for every inactive identity.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "row_compaction.metal"
_METAL_LIBRARY = None
_SCAN_BLOCK_SIZE = 256


class CompactionOverflow(ValueError):
  """A requested compact map cannot retain every active logical identity."""


@dataclass(frozen=True)
class CompactionMap:
  """One fixed-capacity compact map for a batch of logical identities."""

  packed_to_logical: object  # [batch, capacity], -1 padded
  logical_to_packed: object  # [batch, logical_count], -1 when inactive
  active_count: object       # [batch]
  overflow: object           # [batch] int32 0/1; true forbids execution

  @property
  def capacity(self):
    return int(self.packed_to_logical.shape[1])

  @property
  def logical_count(self):
    return int(self.logical_to_packed.shape[1])


@dataclass(frozen=True)
class ConstraintCompaction:
  """Pair, slot, and row maps with a shared per-world overflow status."""

  pairs: CompactionMap
  slots: CompactionMap
  rows: CompactionMap


class CompactionWorkspace:
  """Reusable fixed-shape MPS buffers for one logical map domain."""

  def __init__(self, batch, logical_count, capacity, device="mps"):
    import torch
    if device != "mps" and getattr(device, "type", None) != "mps":
      raise ValueError("native compaction workspace requires MPS")
    for name, value in (("batch", batch), ("logical_count", logical_count),
                        ("capacity", capacity)):
      if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if int(batch) == 0:
      raise ValueError("batch must be positive")
    self.batch = int(batch)
    self.logical_count = int(logical_count)
    self.capacity = int(capacity)
    self.device = torch.device(device)
    blocks = max((self.logical_count + _SCAN_BLOCK_SIZE - 1) // _SCAN_BLOCK_SIZE, 1)
    self.output = CompactionMap(
        torch.full((self.batch, self.capacity), -1, dtype=torch.int32, device=self.device),
        torch.full((self.batch, self.logical_count), -1, dtype=torch.int32, device=self.device),
        torch.zeros((self.batch,), dtype=torch.int32, device=self.device),
        torch.zeros((self.batch,), dtype=torch.int32, device=self.device),
    )
    self.local_prefix = torch.zeros(
        (self.batch, max(self.logical_count, 1)), dtype=torch.int32, device=self.device)
    self.block_sum = torch.zeros((self.batch, blocks), dtype=torch.int32, device=self.device)
    self.block_offset = torch.zeros((self.batch, blocks), dtype=torch.int32, device=self.device)
    self.dims = torch.tensor(
        (self.logical_count, self.capacity, self.batch, blocks),
        dtype=torch.int32, device=self.device)

  def run(self, flags):
    import torch
    if flags.ndim == 1:
      flags = flags.reshape(1, -1)
    if tuple(flags.shape) != (self.batch, self.logical_count):
      raise ValueError(
          "flags shape must match preallocated compaction workspace "
          f"({self.batch}, {self.logical_count}); got {tuple(flags.shape)}")
    if flags.device.type != "mps":
      raise ValueError("native compaction flags must be on MPS")
    if not flags.is_contiguous():
      flags = flags.contiguous()
    flags = flags.to(dtype=torch.float32).contiguous()
    self.output.packed_to_logical.fill_(-1)
    self.output.logical_to_packed.fill_(-1)
    _launch_compaction_kernel(flags, self, torch)
    return self.output


def compact_flags_cpu(flags, capacity=None):
  """Reference stable compactor for a 1-D or 2-D CPU flag array.

  Flag semantics are strictly nonzero-is-active, shared with the Metal
  kernels. This accepts fractional or negative numeric masks without silently
  changing their meaning; NaN and infinities also compare nonzero and thus
  remain active conservatively.
  """
  flags = np.asarray(flags)
  if flags.ndim == 1:
    flags = flags.reshape(1, -1)
  if flags.ndim != 2:
    raise ValueError("flags must have shape [batch, logical_count]")
  if flags.dtype.kind not in "biuf":
    raise TypeError("flags must be numeric or boolean")
  batch, logical_count = flags.shape
  if capacity is None:
    capacity = logical_count
  if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
    raise ValueError("capacity must be a nonnegative integer")
  capacity = int(capacity)
  packed = np.full((batch, capacity), -1, dtype=np.int32)
  reverse = np.full((batch, logical_count), -1, dtype=np.int32)
  count = np.zeros(batch, dtype=np.int32)
  overflow = np.zeros(batch, dtype=np.int32)
  for world in range(batch):
    active = np.flatnonzero(flags[world] != 0)
    count[world] = len(active)
    if len(active) > capacity:
      overflow[world] = 1
      # Do not return a partially valid map that could be mistaken for an
      # executable truncation. The nonzero count/status are retained.
      continue
    packed[world, :len(active)] = active
    reverse[world, active] = np.arange(len(active), dtype=np.int32)
  return CompactionMap(packed, reverse, count, overflow)


def compact_flags(flags, capacity=None, out=None):
  """Compact flags on CPU (oracle) or MPS (native scan/pack kernel).

  The MPS kernel is one deterministic invocation per world and never reads a
  count back to the host. Output buffers have fixed shapes, and an overflow
  status prevents the caller from launching a truncated solve.
  """
  try:
    import torch
  except ImportError:
    torch = None
  if torch is not None and isinstance(flags, torch.Tensor):
    if flags.ndim == 1:
      flags = flags.reshape(1, -1)
    if flags.ndim != 2 or flags.device.type != "mps":
      raise ValueError("tensor flags must be a 2-D MPS tensor")
    if not flags.is_contiguous():
      flags = flags.contiguous()
    if capacity is None:
      capacity = out.capacity if isinstance(out, CompactionWorkspace) else int(flags.shape[1])
    if isinstance(out, CompactionWorkspace):
      if int(capacity) != out.capacity:
        raise ValueError("capacity differs from the preallocated workspace")
      return out.run(flags)
    if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
      raise ValueError("capacity must be a nonnegative integer")
    return _compact_flags_metal(flags.to(dtype=torch.float32).contiguous(), int(capacity), torch)
  return compact_flags_cpu(flags, capacity)


def compact_constraints(pair_flags, slot_flags, row_flags, *, pair_capacity=None,
                        slot_capacity=None, row_capacity=None):
  """Build deterministic pair/slot/row maps for a complete constraint step."""
  pairs = compact_flags(pair_flags, pair_capacity)
  slots = compact_flags(slot_flags, slot_capacity)
  rows = compact_flags(row_flags, row_capacity)
  return ConstraintCompaction(pairs, slots, rows)


def _compact_flags_metal(flags, capacity, torch):
  workspace = CompactionWorkspace(*flags.shape, capacity, device=flags.device)
  return workspace.run(flags)


def _launch_compaction_kernel(flags, workspace, torch):
  global _METAL_LIBRARY
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    raise RuntimeError("native row compaction requires PyTorch MPS compile_shader")
  if _METAL_LIBRARY is None:
    _METAL_LIBRARY = torch.mps.compile_shader(_SHADER.read_text())
  batch, logical_count = workspace.batch, workspace.logical_count
  output = workspace.output
  blocks = int(workspace.block_sum.shape[1])
  _METAL_LIBRARY.compact_flag_blocks(
      flags, workspace.local_prefix, workspace.block_sum, workspace.dims,
      threads=(batch * blocks * _SCAN_BLOCK_SIZE,),
      group_size=(_SCAN_BLOCK_SIZE,))
  _METAL_LIBRARY.scan_compaction_blocks(
      workspace.block_sum, workspace.block_offset,
      output.active_count, output.overflow, workspace.dims,
      threads=(batch,), group_size=(1,))
  _METAL_LIBRARY.scatter_compaction_maps(
      flags, workspace.local_prefix, workspace.block_offset,
      output.overflow, output.packed_to_logical, output.logical_to_packed,
      workspace.dims,
      threads=(batch * max(logical_count, 1),), group_size=(1,))
