# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compiled sparse support for velocity-derivative operators.

MuJoCo 3.10 stores ``qDeriv`` in the model's compiled D CSR.  That pattern is
the authoritative union for passive, tendon, actuator, fluid, and bias
velocity derivatives.  This module lowers that exact pattern to a sorted COO
view for the matrix-free effective-implicit consumer; it does not allocate or
assemble a dense derivative matrix.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "velocity_derivative.metal"
_I32_MAX = (1 << 31) - 1


def _frozen(array, dtype):
  value = np.asarray(array, dtype=dtype, order="C")
  return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class VelocityDerivativeLayout:
  """Immutable row-major COO projection of MuJoCo's compiled D pattern."""

  edge_rows: np.ndarray
  edge_cols: np.ndarray
  diagonal_slots: np.ndarray

  @property
  def edge_count(self):
    return int(self.edge_rows.size)

  def project_dense_oracle(self, matrix):
    """Gather a small dense CPU oracle into COO order.

    This is only an oracle/conformance utility.  Production device derivative
    producers must write their values directly to COO slots.
    """
    value = np.asarray(matrix)
    if value.ndim != 3 or value.shape[1:] != (self.diagonal_slots.size,) * 2:
      raise ValueError("oracle matrix must have shape [batch,nv,nv]")
    if not np.all(np.isfinite(value)):
      raise ValueError("oracle matrix must be finite")
    return np.ascontiguousarray(
        value[:, self.edge_rows, self.edge_cols], dtype=np.float32)

  def symmetric_lower_source_slots(self):
    """Return the compiled-D slot supplying MuJoCo's implicitfast qH entry.

    MuJoCo gathers the lower triangle of qDeriv into the lower M pattern and
    factors that result as symmetric LDL.  On the full D COO support, each
    entry therefore reads its lower counterpart; a missing counterpart is a
    structural zero (the corresponding mass-pattern gather is zero as well).
    """
    lookup = {(int(row), int(col)): slot
              for slot, (row, col) in enumerate(
                  zip(self.edge_rows, self.edge_cols))}
    source = np.fromiter(
        (lookup.get((max(int(row), int(col)), min(int(row), int(col))), -1)
         for row, col in zip(self.edge_rows, self.edge_cols)),
        dtype=np.int32, count=self.edge_count)
    return _frozen(source, np.int32)

  def project_symmetric_lower_oracle(self, matrix):
    """Mirror the pinned implicitfast lower-triangle gather in COO order."""
    value = np.asarray(matrix)
    if value.ndim != 3 or value.shape[1:] != (self.diagonal_slots.size,) * 2:
      raise ValueError("oracle matrix must have shape [batch,nv,nv]")
    return np.ascontiguousarray(
        value[:, np.maximum(self.edge_rows, self.edge_cols),
              np.minimum(self.edge_rows, self.edge_cols)], dtype=np.float32)

  def column_edge_map(self):
    """Return deterministic CSC-style COO slots grouped by derivative column.

    RNE bias differentiation computes one velocity column per device thread.
    This map lets that thread emit only compiled D entries without materializing
    a dense ``[batch,nv,nv]`` derivative matrix.
    """
    nv, edges = int(self.diagonal_slots.size), int(self.edge_count)
    if nv + 1 > _I32_MAX or edges > _I32_MAX:
      raise ValueError("compiled D column map exceeds signed int32 addressing")
    buckets = [[] for _ in range(nv)]
    for slot, (row, col) in enumerate(zip(self.edge_rows, self.edge_cols)):
      buckets[int(col)].append((int(row), slot))
    offsets = np.zeros(nv + 1, dtype=np.int32)
    for col, bucket in enumerate(buckets):
      offsets[col + 1] = offsets[col] + len(bucket)
    rows = np.empty(max(edges, 1), dtype=np.int32)
    slots = np.empty(max(edges, 1), dtype=np.int32)
    for col, bucket in enumerate(buckets):
      start = int(offsets[col])
      for offset, (row, slot) in enumerate(bucket):
        rows[start + offset] = row
        slots[start + offset] = slot
    if edges == 0:
      rows[0] = slots[0] = 0
    return (_frozen(offsets, np.int32), _frozen(rows, np.int32),
            _frozen(slots, np.int32))


def compile_velocity_derivative_layout(model):
  """Lower exact compiled MuJoCo D CSR support into sorted unique COO.

  Accepts the compiled pinned ``MjModel`` because the immutable descriptor
  intentionally does not carry D's numeric-derivative sparsity metadata.
  ``D_rowadr``/``D_rownnz`` have one row per DOF, and ``D_colind`` holds the
  compiled columns.  Every row must contain its diagonal; otherwise a
  producer's compiled pattern and the source matrix ABI disagree.
  """
  nv = int(model.nv)
  if nv < 0 or nv > _I32_MAX:
    raise ValueError("compiled D nv exceeds signed int32 addressing")
  rowadr = np.asarray(model.D_rowadr, dtype=np.int64)
  rownnz = np.asarray(model.D_rownnz, dtype=np.int64)
  colind = np.asarray(model.D_colind, dtype=np.int64)
  if rowadr.shape != (nv,) or rownnz.shape != (nv,):
    raise ValueError("compiled D row arrays must have shape [nv]")
  if np.any(rowadr < 0) or np.any(rownnz < 0):
    raise ValueError("compiled D row metadata must be nonnegative")
  if (np.any(rowadr > _I32_MAX) or np.any(rownnz > _I32_MAX)
      or colind.size > _I32_MAX):
    raise ValueError("compiled D metadata exceeds signed int32 addressing")
  if rowadr.size and np.any(rowadr + rownnz > colind.size):
    raise ValueError("compiled D row metadata exceeds D_colind")

  edges = []
  logical_edge_count = int(np.sum(rownnz, dtype=np.int64))
  if logical_edge_count > _I32_MAX:
    raise ValueError("compiled D edge count exceeds signed int32 addressing")
  for row in range(nv):
    columns = colind[rowadr[row]:rowadr[row] + rownnz[row]]
    if columns.size and (np.any(columns < 0) or np.any(columns >= nv)):
      raise ValueError("compiled D columns must be in [0,nv)")
    if columns.size > 1 and np.any(columns[1:] <= columns[:-1]):
      raise ValueError("compiled D rows must be sorted and unique")
    edges.extend((row, int(column)) for column in columns)

  rows = np.fromiter((row for row, _ in edges), dtype=np.int32,
                     count=len(edges))
  cols = np.fromiter((column for _, column in edges), dtype=np.int32,
                     count=len(edges))
  lookup = {(int(row), int(col)): slot
            for slot, (row, col) in enumerate(zip(rows, cols))}
  diagonal = np.fromiter((lookup.get((dof, dof), -1) for dof in range(nv)),
                         dtype=np.int32, count=nv)
  if np.any(diagonal < 0):
    raise ValueError("compiled D CSR must contain every DOF diagonal")
  return VelocityDerivativeLayout(
      edge_rows=_frozen(rows, np.int32),
      edge_cols=_frozen(cols, np.int32),
      diagonal_slots=_frozen(diagonal, np.int32))


def signed_passive_diagonal(damping_magnitude):
  """Return MuJoCo qDeriv's negative passive-damper diagonal contribution."""
  values = np.asarray(damping_magnitude)
  if values.ndim != 2 or not np.all(np.isfinite(values)):
    raise ValueError("passive damping derivative must be finite [batch,nv]")
  return np.ascontiguousarray(-values, dtype=np.float32)


class MetalVelocityDerivativeValues:
  """Direct COO value assembly for currently supported source contributions.

  The edge-value output is borrowed from ``MetalEffectiveImplicitSolve`` so
  this writer adds no duplicate B×E backing. Call ``clear_device`` once per
  forward derivative stage, then each source-specific writer contributes its
  values. Device execution stays fixed-shape and never reads counts to host.
  """

  def __init__(self, layout, batch_size, device_values, *,
               edge_rows_device=None, edge_cols_device=None):
    import torch
    if (isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or int(batch_size) <= 0):
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = int(batch_size)
    self.nv = int(layout.diagonal_slots.size)
    self.edge_count = int(layout.edge_count)
    if (not isinstance(device_values, torch.Tensor)
        or device_values.dtype != torch.float32
        or device_values.device.type != "mps"
        or tuple(device_values.shape) !=
        (self.batch_size, max(self.edge_count, 1))
        or not device_values.is_contiguous()):
      raise ValueError("device_values must be contiguous float32 MPS [B,max(E,1)]")
    from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
    _validate_workspace_index_capacity(
        self.batch_size,
        {"velocity_derivative.diagonal_slots": max(self.nv, 1),
         "velocity_derivative.dims": 3,
         "velocity_derivative.edge_values": self.batch_size * max(self.edge_count, 1),
         "velocity_derivative.symmetric_lower_source_slots": max(self.edge_count, 1),
         "velocity_derivative.symmetric_values": self.batch_size * max(self.edge_count, 1)},
        {"nv": self.nv, "edge_count": self.edge_count})
    self._torch, self._values = torch, device_values
    self._device = device_values.device
    # Layout arrays are immutable host metadata. Give torch owned writable
    # buffers before the CPU->MPS transfer; as_tensor on read-only NumPy
    # storage is unsupported and can warn or alias unexpectedly.
    slots = np.array(layout.diagonal_slots, dtype=np.int32, copy=True,
                     order="C")
    self._diagonal_slots = torch.as_tensor(
        slots if slots.size else np.zeros(1, np.int32),
        dtype=torch.int32, device=self._device)
    edge_rows = np.array(layout.edge_rows, dtype=np.int32, copy=True,
                         order="C")
    edge_cols = np.array(layout.edge_cols, dtype=np.int32, copy=True,
                         order="C")
    if edge_rows_device is None:
      self._edge_rows = torch.as_tensor(
          edge_rows if edge_rows.size else np.zeros(1, np.int32),
          dtype=torch.int32, device=self._device)
    else:
      self._edge_rows = edge_rows_device
    if edge_cols_device is None:
      self._edge_cols = torch.as_tensor(
          edge_cols if edge_cols.size else np.zeros(1, np.int32),
          dtype=torch.int32, device=self._device)
    else:
      self._edge_cols = edge_cols_device
    for name, index in (("edge_rows_device", self._edge_rows),
                        ("edge_cols_device", self._edge_cols)):
      if (not isinstance(index, torch.Tensor)
          or index.dtype != torch.int32
          or index.device.type != "mps"
          or tuple(index.shape) != (max(self.edge_count, 1),)
          or not index.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS int32 [max(E,1)]")
    self._dims = torch.tensor(
        [self.nv, self.edge_count, self.batch_size],
        dtype=torch.int32, device=self._device)
    self._column_dims = torch.tensor(
        [self.nv, self.edge_count, self.batch_size, 0],
        dtype=torch.int32, device=self._device)
    source_slots = np.array(layout.symmetric_lower_source_slots(),
                            dtype=np.int32, copy=True, order="C")
    if source_slots.size == 0:
      source_slots = np.zeros(1, dtype=np.int32)
    self._symmetric_lower_source_slots = torch.as_tensor(
        source_slots, dtype=torch.int32, device=self._device)
    self._symmetric_values = torch.zeros_like(self._values)
    self._library = torch.mps.compile_shader(_SHADER.read_text())

  @property
  def values(self):
    """Borrowed shared `[B,max(E,1)]` COO accumulation target."""
    return self._values

  @property
  def symmetric_values(self):
    """Borrowed COO values after MuJoCo's implicitfast lower-triangle gather."""
    return self._symmetric_values

  def symmetrize_lower_device(self):
    """Mirror lower compiled-D values for MuJoCo implicitfast qH products."""
    self._library.velocity_derivative_mirror_lower(
        self._values.reshape(-1), self._symmetric_lower_source_slots,
        self._dims, self._symmetric_values.reshape(-1),
        threads=self.batch_size * max(self.edge_count, 1))
    return self._symmetric_values

  def clear_device(self):
    """Clear the logical COO values before source contributions are added."""
    self._library.velocity_derivative_clear(
        self._values.reshape(-1), self._dims,
        threads=self.batch_size * max(self.edge_count, 1))

  def add_passive_diagonal_device(self, damping_magnitude):
    """Add the pinned negative damping derivative on compiled D diagonals."""
    torch = self._torch
    expected = (self.batch_size, self.nv)
    if (not isinstance(damping_magnitude, torch.Tensor)
        or damping_magnitude.dtype != torch.float32
        or damping_magnitude.device.type != "mps"
        or tuple(damping_magnitude.shape) != expected
        or not damping_magnitude.is_contiguous()):
      raise ValueError(f"damping_magnitude must be contiguous float32 MPS {expected}")
    if self.nv == 0:
      return
    self._library.velocity_derivative_add_diagonal(
        self._values.reshape(-1), damping_magnitude.reshape(-1),
        self._diagonal_slots, self._dims,
        threads=self.batch_size * self.nv)

  def add_column_device(self, column, derivative_column):
    """Add one exact source column directly to its compiled COO slots.

    This supports producers whose natural operation is a velocity-column
    perturbation or directional derivative while avoiding a persistent dense
    ``[B,nv,nv]`` temporary. The caller clears the shared values once before
    assembling all columns and sources.
    """
    torch = self._torch
    if (isinstance(column, (bool, np.bool_))
        or not isinstance(column, (int, np.integer))
        or int(column) < 0 or int(column) >= self.nv):
      raise ValueError("column must be an integer in [0,nv)")
    expected = (self.batch_size, self.nv)
    if (not isinstance(derivative_column, torch.Tensor)
        or derivative_column.dtype != torch.float32
        or derivative_column.device.type != "mps"
        or tuple(derivative_column.shape) != expected
        or not derivative_column.is_contiguous()):
      raise ValueError(f"derivative_column must be contiguous float32 MPS {expected}")
    if self.edge_count == 0:
      return
    # This scalar is static dispatch metadata, not a device-derived branch.
    self._column_dims[3] = int(column)
    self._library.velocity_derivative_add_column(
        self._values.reshape(-1), derivative_column.reshape(-1),
        self._edge_rows, self._edge_cols, self._column_dims,
        threads=self.batch_size * self.edge_count)

  def add_dense_source_device(self, derivative, *, sign=1):
    """Gather one transitional dense source into the compiled D-CSR union.

    This avoids constructing a combined dense derivative or effective mass.
    Source-specific kernels should replace this gather as they become
    available; their source backing remains independently capacity-accounted.
    """
    torch = self._torch
    expected = (self.batch_size, self.nv, self.nv)
    if (not isinstance(derivative, torch.Tensor)
        or derivative.dtype != torch.float32
        or derivative.device.type != "mps"
        or tuple(derivative.shape) != expected
        or not derivative.is_contiguous()):
      raise ValueError(
          f"derivative must be contiguous float32 MPS {expected}")
    if isinstance(sign, (bool, np.bool_)) or sign not in (-1, 1):
      raise ValueError("sign must be +1 or -1")
    if self.edge_count == 0:
      return
    kernel = (self._library.velocity_derivative_add_dense_positive
              if sign == 1 else
              self._library.velocity_derivative_add_dense_negative)
    kernel(self._values.reshape(-1), derivative.reshape(-1),
           self._edge_rows, self._edge_cols, self._dims,
           threads=self.batch_size * self.edge_count)
