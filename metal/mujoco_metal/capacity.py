# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit capacity and allocation architecture (milestone 017).

Fixed small-model ceilings become named, budgeted capacities resolved at
host construction boundaries. Device execution never grows allocations:
overflow is a host lowering error with requested-vs-allowed detail (no
silent truncation), and per-environment runtime failures stay isolated via
the existing status channel.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

# Baseline ceilings carried over from milestones 004-016. Pair/slot
# ceilings were raised by the F1 scalable path (broadphase-fed pruning:
# inactive pairs skip narrowphase work, so cost follows active slots and
# rows, both still hard-capped below) — never a bare constant bump. nv/row
# ceilings still need tiled layouts and the block solver (follow-up).
BASE_NVIDIA_NV = 32
BASE_MAX_PAIRS = 32
BASE_MAX_SLOTS = 48
BASE_MAX_ROWS = 96
# Dense solver threshold: at or below this row count the native solver runs
# the exact small-model dense path (unchanged math). Above it the block
# path takes over (017 follow-up commits).
DENSE_ROW_THRESHOLD = 96
AUTO_JACOBIAN_DENSE_NV = 60


class CapacityOverflow(ValueError):
  """Host lowering error: model needs more than the budgeted capacity."""


@dataclass(frozen=True)
class CapacityLimits:
  """User-visible budgets. Growth happens only at host boundaries."""

  max_nv: int = BASE_NVIDIA_NV
  max_pairs: int = BASE_MAX_PAIRS
  max_slots: int = BASE_MAX_SLOTS
  max_rows: int = BASE_MAX_ROWS
  max_batch: int = 16
  memory_budget_bytes: int = 1 << 30


@dataclass(frozen=True)
class CapacityEstimate:
  """Deterministic host-side estimate for one model and batch size."""

  nv: int
  nbody: int
  ngeom: int
  npairs: int
  nslots: int
  nr: int
  batch: int
  dense_path: bool
  jacobian_kind: str
  jacobian_auto_threshold: int
  memory_bytes: int
  memory_breakdown: tuple


def _bytes(n):
  return int(n) * 4


def _bytes_i32(n):
  return int(n) * 4


def selected_jacobian_kind(model: mujoco.MjModel, nv=None):
  """Return the pinned MuJoCo 3.10 Jacobian storage selection.

  AUTO selects dense storage through 60 DOFs and sparse storage above 60.
  This decision is independent of solver-row path selection.
  """
  jac = int(model.opt.jacobian)
  if jac == int(mujoco.mjtJacobian.mjJAC_DENSE):
    return "dense"
  if jac == int(mujoco.mjtJacobian.mjJAC_SPARSE):
    return "sparse"
  if jac != int(mujoco.mjtJacobian.mjJAC_AUTO):
    raise ValueError(f"unknown MuJoCo Jacobian mode: {jac}")
  dofs = int(model.nv if nv is None else nv)
  return "dense" if dofs <= AUTO_JACOBIAN_DENSE_NV else "sparse"


def estimate_workspace(nv, npairs, nslots, nr, nr_joint, neq, batch):
  """Budget solver buffers and the fixed-capacity milestone-017 maps.

  Returns ``(parts, total)`` with exact sizes for current coupled solver
  allocations plus the reserved pair/slot/row maps and scan scratch required
  by 017. Counts use ``max(..., 1)`` empty guards. The compact-map entries
  are reservations until the packing path binds them into
  ``prepare_workspace``; they must remain here so the allocation budget
  continues to cover the integrated implementation.
  """
  b, v = int(batch), int(max(nv, 1))
  nc = int(nslots)
  # Both ``empty`` and torch.zeros in prepare_workspace allocate one element
  # for an otherwise empty buffer. Keep that guard explicit so the estimate
  # covers the zero-contact/zero-row cases too.
  def guarded(n):
    return max(int(n), 1)
  blocks = max((int(nr) + 255) // 256, 1)
  parts = (
      ("contact_row_data", _bytes(guarded(b * nc * 6 * 6))),
      ("contact_frame", _bytes(guarded(b * nc * 12))),
      ("contact_jacobian", _bytes(guarded(b * nc * 6 * v))),
      ("pair_mask", _bytes(guarded(b * max(int(npairs), 1)))),
      ("workspace_J", _bytes(guarded(b * int(nr) * v))),
      ("workspace_debug", _bytes(guarded(b * (int(nr) * int(nr) + 7 * int(nr))))),
      ("out_force", _bytes(guarded(b * v))),
      ("out_acc", _bytes(guarded(b * v))),
      ("out_status", _bytes_i32(b)),
      ("out_diagnostics", _bytes(guarded(b * 10))),
      ("out_contact_force", _bytes(guarded(b * nc * 11))),
      ("out_joint_force", _bytes(guarded(b * max(int(nr_joint), 1)))),
      ("eq_active", _bytes_i32(b * max(int(neq), 1))),
      ("eq_active_default", _bytes_i32(b * max(int(neq), 1))),
      # Compaction maps preserve canonical logical identity while reducing
      # pair, slot, and row execution spans. The conservative budget assumes
      # every candidate can be active and accounts for independent scan
      # scratch for all three domains.
      ("packed_to_pair", _bytes_i32(b * int(npairs))),
      ("pair_to_packed", _bytes_i32(b * int(npairs))),
      ("packed_to_slot", _bytes_i32(b * nc)),
      ("slot_to_packed", _bytes_i32(b * nc)),
      ("packed_to_logical_row", _bytes_i32(b * int(nr))),
      ("logical_to_packed_row", _bytes_i32(b * int(nr))),
      ("compaction_counts_overflow", _bytes_i32(b * 4)),
      ("pair_scan_prefix", _bytes_i32(b * guarded(int(npairs)))),
      ("pair_scan_blocks", _bytes_i32(b * max((int(npairs) + 255) // 256, 1) * 2)),
      ("slot_scan_prefix", _bytes_i32(b * guarded(nc))),
      ("slot_scan_blocks", _bytes_i32(b * max((nc + 255) // 256, 1) * 2)),
      ("row_scan_prefix", _bytes_i32(b * guarded(int(nr)))),
      ("row_scan_blocks", _bytes_i32(b * blocks * 2)),
  )
  return parts, sum(value for _, value in parts)


def estimate_capacity(model, batch_size, npairs, nslots, nr, *, neq=0, nr_joint=0) -> CapacityEstimate:
  """Build a deterministic estimate from lowering counts.

  `npairs`/`nslots`/`nr` come from the coupled lowering (candidate counts,
  before broadphase pruning); `neq`/`nr_joint` size the equality/joint
  outputs. Memory includes the current solver workspace and full compact-map
  reservation (see :func:`estimate_workspace`) at the REAL batch size: lowering-time calls
  must pass the construction batch, never a hardcoded 1. Deterministic:
  pure function of the inputs, no device state.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  batch = int(batch_size)
  if batch <= 0:
    raise ValueError("batch_size must be positive")
  nv = int(model.nv)
  nbody, ngeom = int(model.nbody), int(model.ngeom)
  nr = int(nr)
  parts, total = estimate_workspace(nv, npairs, nslots, nr, nr_joint, neq, batch)
  return CapacityEstimate(
      nv=nv, nbody=nbody, ngeom=ngeom, npairs=int(npairs),
      nslots=int(nslots), nr=nr, batch=batch,
      dense_path=(nr <= DENSE_ROW_THRESHOLD and nv <= 32),
      jacobian_kind=selected_jacobian_kind(model),
      jacobian_auto_threshold=AUTO_JACOBIAN_DENSE_NV,
      memory_bytes=total, memory_breakdown=parts)


def check_capacity(estimate, limits=None) -> CapacityEstimate:
  """Enforce budgets; raise CapacityOverflow (a ValueError) on excess.

  Keeps the historical error text so existing boundary tests keep reading
  the same diagnostics, now with explicit requested-vs-allowed detail.
  """
  limits = limits or CapacityLimits()
  if estimate.batch > limits.max_batch:
    raise CapacityOverflow(
        f"batch size ({estimate.batch}) exceeds capacity {limits.max_batch}")
  if estimate.nv > limits.max_nv:
    raise CapacityOverflow(
        f"coupled constraint stage bounds nv to {limits.max_nv}; found {estimate.nv}")
  if estimate.npairs > limits.max_pairs:
    raise CapacityOverflow(
        f"candidate contact pairs ({estimate.npairs}) exceeds capacity {limits.max_pairs}")
  if estimate.nslots > limits.max_slots:
    raise CapacityOverflow(
        f"total candidate contact slots ({estimate.nslots}) exceeds capacity {limits.max_slots}")
  if estimate.nr > limits.max_rows:
    raise CapacityOverflow(
        f"total candidate constraint rows ({estimate.nr}) exceeds capacity {limits.max_rows}")
  if estimate.memory_bytes > limits.memory_budget_bytes:
    raise CapacityOverflow(
        f"estimated device memory ({estimate.memory_bytes} bytes) exceeds budget "
        f"{limits.memory_budget_bytes} bytes")
  return estimate
