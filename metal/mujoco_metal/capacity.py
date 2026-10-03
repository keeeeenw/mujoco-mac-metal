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

# Baseline ceilings carried over from milestones 004-016. Raising any of
# these requires the matching scalable path (broadphase/compaction/block
# solver/tiled kernels), never a bare constant bump.
BASE_NVIDIA_NV = 32
BASE_MAX_PAIRS = 16
BASE_MAX_SLOTS = 24
BASE_MAX_ROWS = 96
# Dense solver threshold: at or below this row count the native solver runs
# the exact small-model dense path (unchanged math). Above it the block
# path takes over (017 follow-up commits).
DENSE_ROW_THRESHOLD = 96


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
  memory_bytes: int
  memory_breakdown: tuple


def _bytes(n):
  return int(n) * 4


def estimate_capacity(model, batch_size, npairs, nslots, nr) -> CapacityEstimate:
  """Build a deterministic estimate from lowering counts.

  `npairs`/`nslots`/`nr` come from the coupled lowering (candidate counts,
  before broadphase pruning); memory covers the dense Delassus, Jacobian,
  contact frames/Jacobians/forces, and retained warmstarts. Deterministic:
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
  parts = (
      ("delassus_W", _bytes(batch * max(nr, 0) * max(nr, 0))),
      ("delassus_vectors", _bytes(batch * 7 * max(nr, 0))),
      ("jacobian_J", _bytes(batch * max(nr, 0) * max(nv, 1))),
      ("contact_row_data", _bytes(batch * max(int(nslots), 0) * 36)),
      ("contact_frame", _bytes(batch * max(int(nslots), 0) * 12)),
      ("contact_jacobian", _bytes(batch * max(int(nslots), 0) * 6 * max(nv, 1))),
      ("contact_force", _bytes(batch * max(int(nslots), 0) * 11)),
      ("warmstart", _bytes(batch * max(nr, 0))),
  )
  total = sum(value for _, value in parts)
  return CapacityEstimate(
      nv=nv, nbody=nbody, ngeom=ngeom, npairs=int(npairs),
      nslots=int(nslots), nr=nr, batch=batch,
      dense_path=nr <= DENSE_ROW_THRESHOLD,
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
