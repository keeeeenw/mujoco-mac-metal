# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0
"""Exact constraint impedance updates for MuJoCo 3.10's DIAGEXACT flag.

Pinned source: ``src/engine/engine_core_constraint.c``,
``mj_projectConstraint`` and ``mj_makeImpedance``. Exact mass solves are
provided by the caller; this module contracts canonical ``J`` and ``M^-1 J^T``
rows, updates R/diagA and applies the source's elliptic/pyramidal cone scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

_SHADER = Path(__file__).with_name("shaders") / "constraint_impedance.metal"


class ConeGroup(NamedTuple):
  """Contact cone metadata; ``start`` indexes its normal canonical row."""
  start: int
  condim: int
  cone: str
  friction: tuple[float, ...]


def constraint_impedance_workspace_sizes(batch_size, row_capacity,
                                         dof_capacity, cone_capacity=1,
                                         *, mass_storage="none"):
  """Return exact tensor element counts for the prepared helper.

  The returned mapping names each independent backing rather than reporting
  only a combined byte total, so callers can enforce signed-index limits
  before importing Torch or allocating MPS storage.
  """
  values = (batch_size, row_capacity, dof_capacity, cone_capacity)
  if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
         for v in values):
    raise ValueError("all capacities must be positive integers")
  if mass_storage not in ("none", "dense", "block_sparse"):
    raise ValueError("mass_storage must be none, dense, or block_sparse")
  b, nr, nv, ng = (int(v) for v in values)
  sizes = {
      "diag": b * nr,
      "active_rows": b * nr,
      "world_status": b,
      "row_impedance": b * nr,
      "R": b * nr,
      "diagA": b * nr,
      "cone_mu": b * ng,
      "cone_groups": ng * 3,
      "cone_friction": ng * 5,
      "cone_dims": 4,
      "row_dims": 6,
      "pack_dims": 5,
  }
  if mass_storage == "block_sparse":
    sizes["component_rhs"] = b * (nr + 1) * nv
  elif mass_storage == "dense":
    sizes["dense_rhs"] = b * nv * nr
  bad = {name: count for name, count in sizes.items()
         if count > np.iinfo(np.int32).max}
  if bad:
    name, count = next(iter(bad.items()))
    raise ValueError(f"{name} exceeds signed int32 address range: {count}")
  return sizes


def exact_constraint_diagonal_cpu(jacobian, mass, *, active_dof=None,
                                  active_rows=None):
  """Oracle for ``diag(J M^-1 J.T)`` on awake principal DOFs.

  Shapes are [B,nr,nv] and [B,nv,nv]. This CPU implementation is used only
  by tests; production computes each supplied M^-1 row on device.
  """
  J = np.asarray(jacobian, dtype=np.float64)
  M = np.asarray(mass, dtype=np.float64)
  if J.ndim != 3 or M.ndim != 3 or M.shape[0] != J.shape[0]:
    raise ValueError("jacobian and mass must be [batch,nr,nv] and [batch,nv,nv]")
  batch, nr, nv = J.shape
  if M.shape[1:] != (nv, nv):
    raise ValueError("mass shape does not match Jacobian DOFs")
  awake = (np.ones((batch, nv), dtype=bool) if active_dof is None
           else np.asarray(active_dof, dtype=bool))
  rows = (np.ones((batch, nr), dtype=bool) if active_rows is None
          else np.asarray(active_rows, dtype=bool))
  if awake.shape != (batch, nv) or rows.shape != (batch, nr):
    raise ValueError("active masks must match the Jacobian batch dimensions")
  result = np.zeros((batch, nr), dtype=np.float64)
  for world in range(batch):
    ids = np.flatnonzero(awake[world])
    selected = np.flatnonzero(rows[world])
    if not ids.size or not selected.size:
      continue
    block = M[world][np.ix_(ids, ids)]
    if not np.all(np.isfinite(block)):
      raise ValueError(f"nonfinite active mass in world {world}")
    j = J[world, selected][:, ids]
    solved = np.linalg.solve(block, j.T)
    result[world, selected] = np.einsum("ij,ji->i", j, solved)
  return result


def recompute_impedance_cpu(exact_diagonal, row_impedance, *,
                            cone_groups: Sequence[ConeGroup] = (),
                            impratio=1.0, minval=1e-15):
  """Oracle for pinned ``mj_makeImpedance`` R, diagA and contact scaling.

  Cone friction is clamped to MuJoCo's ``mjMINMU`` (1e-5), matching
  ``mj_assignFriction`` before the contact reaches ``mj_makeImpedance``.
  """
  diag = np.asarray(exact_diagonal, dtype=np.float64)
  imp = np.asarray(row_impedance, dtype=np.float64)
  if diag.ndim != 2 or imp.shape != diag.shape:
    raise ValueError("exact diagonal and row impedance must share [batch,nr]")
  if not np.isfinite(impratio) or impratio <= 0:
    raise ValueError("impratio must be finite and positive")
  if np.any(~np.isfinite(diag)) or np.any(diag < 0):
    raise ValueError("exact diagonal must be finite and nonnegative")
  if np.any(~np.isfinite(imp)) or np.any((imp <= 0) | (imp >= 1)):
    raise ValueError("row impedance must lie strictly between zero and one")
  R = np.maximum(float(minval), (1.0 - imp) * diag / imp)
  mu = np.zeros((diag.shape[0], len(cone_groups)), dtype=np.float64)
  for group_index, group in enumerate(cone_groups):
    start, condim = int(group.start), int(group.condim)
    friction = np.asarray(group.friction, dtype=np.float64)
    if start < 0 or condim not in (3, 4, 6):
      raise ValueError("invalid contact cone row group")
    if group.cone not in ("elliptic", "pyramidal"):
      raise ValueError("cone must be elliptic or pyramidal")
    nrows = condim if group.cone == "elliptic" else 2 * (condim - 1)
    if (start + nrows > diag.shape[1]
        or friction.shape != (5,)
        or np.any(~np.isfinite(friction))):
      raise ValueError("contact cone rows/friction exceed supplied arrays")
    normal = R[:, start].copy()
    R1 = normal / max(float(impratio), float(minval))
    friction = np.maximum(friction, 1e-5)  # pinned mj_assignFriction
    mu[:, group_index] = friction[0] * np.sqrt(
        R1 / np.maximum(normal, float(minval)))
    if group.cone == "elliptic":
      R[:, start + 1] = R1
      for j in range(1, condim - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
          R[:, start + j + 1] = R1 * friction[0] ** 2 / friction[j] ** 2
    else:
      Rpy = 2.0 * mu[:, group_index] ** 2 * normal
      R[:, start:start + nrows] = Rpy[:, None]
  adjusted_diag = R * imp / (1.0 - imp)
  return R, adjusted_diag, mu


def compile_contact_cone_groups(descriptor):
  """Lower static rigid and flex contact cone groups to packed CPU arrays.

  Returns ``(groups, friction)`` with shapes ``[G,3]`` and ``[G,5]``.
  Group rows contain ``(canonical_row_start, condim, kind)`` where kind is 1
  for elliptic and 2 for pyramidal. Activity is deliberately not compiled:
  the device kernel reads it from each world's canonical active-row mask.
  Capacity-zero models retain a single inert group sentinel.
  """
  import mujoco

  elliptic = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
  pyramidal = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
  groups, frictions = [], []

  packed = np.asarray(descriptor.contact_condim_packed, dtype=np.int64)
  offsets = np.asarray(descriptor.pair_contact_offset, dtype=np.int64)
  contact_friction = np.asarray(descriptor.contact_friction, dtype=np.float32)
  if packed.size % 3 or offsets.shape != (int(descriptor.npairs) + 1,):
    raise ValueError("contact cone descriptor arrays have inconsistent lengths")
  nslots = packed.size // 3
  if contact_friction.shape != (nslots, 5):
    raise ValueError("contact friction metadata must be [ncontacts,5]")
  for slot in range(nslots):
    condim, row_offset, cone = (int(x) for x in packed[3 * slot:3 * slot + 3])
    if condim == 1:
      continue
    if cone not in (elliptic, pyramidal):
      raise ValueError(f"unsupported contact cone enum {cone}")
    pair = int(np.searchsorted(offsets[1:], slot, side="right"))
    if pair >= int(descriptor.npairs):
      raise ValueError("contact slot does not map to a compiled pair")
    start = int(descriptor.nr_joint) + row_offset
    groups.append((start, condim, 1 if cone == elliptic else 2))
    frictions.append(np.maximum(contact_friction[slot], 1e-5))

  flex = descriptor.flex_contact_descriptor
  if flex is not None:
    for start, condim, cone, friction in zip(
        np.asarray(flex.row_start), np.asarray(flex.condim),
        np.asarray(flex.cone), np.asarray(flex.friction)):
      condim, cone = int(condim), int(cone)
      if condim == 1:
        continue
      if cone not in (elliptic, pyramidal):
        raise ValueError(f"unsupported flex contact cone enum {cone}")
      groups.append((int(descriptor.flex_contact_base) + int(start), condim,
                     1 if cone == elliptic else 2))
      frictions.append(np.maximum(np.asarray(friction, dtype=np.float32), 1e-5))

  if groups:
    return (np.asarray(groups, dtype=np.int32),
            np.asarray(frictions, dtype=np.float32).reshape(-1, 5))
  return (np.zeros((1, 3), dtype=np.int32),
          np.zeros((1, 5), dtype=np.float32))


class MetalConstraintImpedance:
  """Device-side canonical row diagonal and impedance kernels.

  The caller supplies the exact row solve output from a dense or component
  mass factor. Static cone grouping is an immutable descriptor ``[G,3]`` of
  ``(row_start, condim, cone_kind)`` with a matching friction array ``[G,5]``.
  Activity is read from the already-produced canonical normal-row active bit.
  Kind 1 is elliptic, 2 pyramidal.
  """

  def __init__(self, *, batch_size, row_capacity, dof_capacity,
               cone_capacity=1, cone_groups=None, friction=None,
               mass_storage="none"):
    values = (batch_size, row_capacity, dof_capacity, cone_capacity)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
           for v in values):
      raise ValueError("all capacities must be positive integers")
    if cone_groups is not None:
      raw_groups = np.asarray(cone_groups)
      if (raw_groups.ndim != 2 or raw_groups.shape[1] != 3
          or raw_groups.shape[0] < 1
          or not np.issubdtype(raw_groups.dtype, np.integer)
          or np.issubdtype(raw_groups.dtype, np.bool_)):
        raise ValueError("cone_groups must have shape [G,3]")
      rows_checked = []
      int32 = np.iinfo(np.int32)
      for raw_start, raw_condim, raw_kind in raw_groups:
        start, condim, kind = int(raw_start), int(raw_condim), int(raw_kind)
        nrows = condim if kind == 1 else 2 * (condim - 1)
        if (not int32.min <= start <= int32.max
            or not int32.min <= condim <= int32.max
            or not int32.min <= kind <= int32.max
            or condim not in (3, 4, 6) or kind not in (1, 2)
            or start < 0 or start + nrows > int(row_capacity)):
          raise ValueError("cone_groups contain an invalid row span or cone kind")
        rows_checked.append((start, start + nrows))
      rows_checked.sort()
      if any(rows_checked[i][1] > rows_checked[i + 1][0]
             for i in range(len(rows_checked) - 1)):
        raise ValueError("cone_groups row spans must not overlap")
      groups = raw_groups.astype(np.int32, copy=True)
      cone_capacity = int(groups.shape[0])
    else:
      groups = np.zeros((int(cone_capacity), 3), dtype=np.int32)
    if friction is not None:
      friction_array = np.asarray(friction, dtype=np.float32)
      if (friction_array.shape != (int(cone_capacity), 5)
          or not np.all(np.isfinite(friction_array))):
        raise ValueError("friction must have shape [G,5]")
      friction_array = np.maximum(friction_array, np.float32(1e-5))
    else:
      friction_array = np.zeros((int(cone_capacity), 5), dtype=np.float32)
    self.batch_size, self.row_capacity = int(batch_size), int(row_capacity)
    self.dof_capacity, self.cone_capacity = int(dof_capacity), int(cone_capacity)
    self._workspace_sizes = constraint_impedance_workspace_sizes(
        self.batch_size, self.row_capacity, self.dof_capacity,
        self.cone_capacity, mass_storage=mass_storage)
    self.mass_storage = mass_storage
    import torch
    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._row_dims = torch.tensor(
        [self.batch_size, self.row_capacity, self.dof_capacity, 0, 0, 0],
        dtype=torch.int32, device=self._device)
    self._cone_dims = torch.tensor(
        [self.batch_size, self.row_capacity, self.cone_capacity, 0],
        dtype=torch.int32, device=self._device)
    self._pack_dims = torch.tensor(
        [self.batch_size, self.row_capacity, self.dof_capacity,
         self.row_capacity + 1, 0], dtype=torch.int32, device=self._device)
    self._cone_groups = torch.as_tensor(
        groups.copy(), dtype=torch.int32, device=self._device).contiguous()
    self._cone_friction = torch.as_tensor(
        friction_array.copy(), dtype=torch.float32, device=self._device).contiguous()
    self._workspace = {
        "diagonal": torch.zeros((self.batch_size, self.row_capacity),
                                 dtype=torch.float32, device=self._device),
        "active_rows": torch.ones((self.batch_size, self.row_capacity),
                                   dtype=torch.int32, device=self._device),
        "world_status": torch.zeros(self.batch_size, dtype=torch.int32,
                                     device=self._device),
        "row_impedance": torch.empty((self.batch_size, self.row_capacity),
                                      dtype=torch.float32, device=self._device),
        "R": torch.zeros((self.batch_size, self.row_capacity),
                         dtype=torch.float32, device=self._device),
        "diagA": torch.zeros((self.batch_size, self.row_capacity),
                             dtype=torch.float32, device=self._device),
        "cone_mu": torch.zeros((self.batch_size, self.cone_capacity),
                               dtype=torch.float32, device=self._device),
    }
    if mass_storage == "block_sparse":
      self._workspace["component_rhs"] = torch.empty(
          (self.batch_size, self.row_capacity + 1, self.dof_capacity),
          dtype=torch.float32, device=self._device)
    elif mass_storage == "dense":
      self._workspace["dense_rhs"] = torch.empty(
          (self.batch_size, self.dof_capacity, self.row_capacity),
          dtype=torch.float32, device=self._device)

  def solve_mass_rows_device(self, jacobian, active_rows, *,
                             mass_storage, solver, mass,
                             mass_blocks=None, dof_ids=None, counts=None,
                             tendon_armature_blocks=None,
                             packed_jacobian=False):
    """Apply the prepared mass factor to each active canonical row RHS.

    The returned layout matches the native diagonal contraction: dense solves
    return ``[B,nv,nr]``; block-sparse solves return the fixed component ABI
    ``[B,nr+1,nv]`` with a zero smooth-force sentinel row. RHS packing is one
    preallocated MSL dispatch, including active-row zeroing. Returned solution
    and status views remain owned by the supplied solver.
    """
    if mass_storage != self.mass_storage:
      raise ValueError("mass_storage differs from prepared impedance workspace")
    torch = self._torch
    expected_j = (self.batch_size, self.row_capacity, self.dof_capacity)
    if not isinstance(jacobian, torch.Tensor) or jacobian.dtype != torch.float32:
      raise ValueError("jacobian must be a float32 MPS tensor")
    if packed_jacobian:
      if jacobian.ndim != 1 or jacobian.device.type != "mps" or not jacobian.is_contiguous():
        raise ValueError("packed jacobian must be contiguous flat float32 MPS storage")
      self._pack_dims[4] = 1
    elif (tuple(jacobian.shape) != expected_j
          or jacobian.device.type != "mps" or not jacobian.is_contiguous()):
      raise ValueError("jacobian must be contiguous float32 MPS [B,nr,nv]")
    else:
      self._pack_dims[4] = 0
    if (not isinstance(active_rows, torch.Tensor)
        or tuple(active_rows.shape) != (self.batch_size, self.row_capacity)
        or active_rows.dtype not in (torch.int32, torch.bool, torch.float32)
        or active_rows.device.type != "mps"):
      raise ValueError("active_rows must be MPS int32/bool [B,nr]")
    # Debug-row activity is stored as float32 in the coupled workspace. Copy
    # through the owned contiguous integer backing so the MSL RHS packer sees
    # a canonical stride-free ABI.
    self._workspace["active_rows"].copy_(active_rows)
    if mass_storage == "block_sparse":
      if (solver is None or mass_blocks is None or dof_ids is None
          or counts is None or not hasattr(solver, "run_device")):
        raise ValueError("component row solve requires its factor and awake mass inputs")
      rhs = self._workspace["component_rhs"]
      self._library.pack_component_inverse_rhs(
          jacobian.reshape(-1), self._workspace["active_rows"].reshape(-1),
          rhs.reshape(-1),
          self._pack_dims,
          threads=(self.batch_size, self.row_capacity, self.dof_capacity),
          group_size=(1, 1, 1))
      return solver.run_device(
          mass_blocks, rhs, dof_ids=dof_ids, counts=counts,
          tendon_armature_blocks=tendon_armature_blocks)
    if mass_storage == "dense":
      if solver is None or mass is None or not hasattr(solver, "run_device"):
        raise ValueError("dense row solve requires its factor and mass matrix")
      rhs = self._workspace["dense_rhs"]
      self._library.pack_dense_inverse_rhs(
          jacobian.reshape(-1), self._workspace["active_rows"].reshape(-1),
          rhs.reshape(-1),
          self._pack_dims,
          threads=(self.batch_size, self.row_capacity, self.dof_capacity),
          group_size=(1, 1, 1))
      return solver.run_device(mass, rhs)
    raise ValueError("no prepared mass-row workspace is available")

  def mask_active_rows_device(self, active_rows, world_status):
    """Suppress rows belonging to failed worlds without host inspection."""
    torch = self._torch
    if (not isinstance(active_rows, torch.Tensor)
        or tuple(active_rows.shape) != (self.batch_size, self.row_capacity)
        or active_rows.dtype != torch.float32
        or active_rows.device.type != "mps"):
      raise ValueError("active_rows must be MPS float32 [B,nr]")
    if (not isinstance(world_status, torch.Tensor)
        or tuple(world_status.shape) != (self.batch_size,)
        or world_status.dtype != torch.int32
        or world_status.device.type != "mps"
        or not world_status.is_contiguous()):
      raise ValueError("world_status must be contiguous MPS int32 [B]")
    output = self._workspace["active_rows"]
    self._library.mask_active_rows(
        active_rows.reshape(-1), world_status, output.reshape(-1),
        self._row_dims,
        threads=(self.batch_size, self.row_capacity), group_size=(1, 1))
    return output

  def exact_diagonal_device(self, jacobian, inverse_jacobian, *,
                            active_rows=None, rhs_layout="row_major"):
    torch = self._torch
    expected = (self.batch_size, self.row_capacity)
    packed_jacobian = isinstance(jacobian, torch.Tensor) and jacobian.ndim == 1
    if (not isinstance(jacobian, torch.Tensor) or jacobian.dtype != torch.float32
        or jacobian.device.type != "mps" or not jacobian.is_contiguous()):
      raise ValueError("jacobian must be contiguous float32 MPS storage")
    if packed_jacobian:
      if jacobian.numel() < self.batch_size * 12:
        raise ValueError("packed jacobian is shorter than its per-world header")
    elif (jacobian.ndim != 3 or tuple(jacobian.shape[:2]) != expected
          or jacobian.shape[2] != self.dof_capacity):
      raise ValueError("jacobian must be [B,nr,nv] or packed CSR storage")
    if rhs_layout not in ("row_major", "dof_major"):
      raise ValueError("rhs_layout must be row_major or dof_major")
    inverse_shape = (tuple(inverse_jacobian.shape)
                     if isinstance(inverse_jacobian, torch.Tensor) else ())
    if rhs_layout == "row_major":
      inverse_shape_ok = (
          len(inverse_shape) == 3
          and inverse_shape[0] == self.batch_size
          and inverse_shape[1] >= self.row_capacity
          and inverse_shape[2] == self.dof_capacity)
    else:
      inverse_shape_ok = inverse_shape == (
          self.batch_size, self.dof_capacity, self.row_capacity)
    if (not isinstance(inverse_jacobian, torch.Tensor)
        or inverse_jacobian.dtype != torch.float32
        or inverse_jacobian.device.type != "mps"
        or not inverse_shape_ok
        or not inverse_jacobian.is_contiguous()):
      raise ValueError("inverse_jacobian does not match its contiguous MPS layout")
    if active_rows is None:
      self._workspace["active_rows"].fill_(1)
    else:
      if (not isinstance(active_rows, torch.Tensor)
          or tuple(active_rows.shape) != expected
          or active_rows.device.type != "mps"
          or active_rows.dtype not in (torch.int32, torch.bool, torch.float32)):
        raise ValueError("active_rows must be MPS bool/int32 [B,nr]")
      self._workspace["active_rows"].copy_(active_rows)
    self._row_dims[3] = int(rhs_layout == "dof_major")
    self._row_dims[4] = (int(inverse_shape[1])
                         if rhs_layout == "row_major" else self.row_capacity)
    self._row_dims[5] = int(packed_jacobian)
    self._library.exact_constraint_diagonal(
        jacobian.reshape(-1), inverse_jacobian.reshape(-1),
        self._workspace["active_rows"].reshape(-1),
        self._workspace["diagonal"].reshape(-1), self._row_dims,
        threads=(self.batch_size, self.row_capacity), group_size=(1, 1))
    return self._workspace["diagonal"]

  def update_impedance_device(self, row_impedance, *, cone_groups=None,
                              friction=None, impratio):
    torch = self._torch
    shape = (self.batch_size, self.row_capacity)
    if (not isinstance(row_impedance, torch.Tensor)
        or tuple(row_impedance.shape) != shape
        or row_impedance.dtype != torch.float32
        or row_impedance.device.type != "mps"):
      raise ValueError("row_impedance must be float32 MPS [B,nr]")
    cone_groups = self._cone_groups if cone_groups is None else cone_groups
    friction = self._cone_friction if friction is None else friction
    if not np.isfinite(impratio) or impratio <= 0:
      raise ValueError("impratio must be finite and positive")
    group_shape = (self.cone_capacity, 3)
    if (not isinstance(cone_groups, torch.Tensor)
        or tuple(cone_groups.shape) != group_shape
        or cone_groups.dtype != torch.int32 or cone_groups.device.type != "mps"
        or not cone_groups.is_contiguous()):
      raise ValueError("cone_groups must be contiguous MPS int32 [G,3]")
    friction_shape = (self.cone_capacity, 5)
    if (not isinstance(friction, torch.Tensor) or tuple(friction.shape) != friction_shape
        or friction.dtype != torch.float32 or friction.device.type != "mps"
        or not friction.is_contiguous()):
      raise ValueError("friction must be contiguous float32 MPS [G,5]")
    ratio_bits = int(np.asarray(np.float32(impratio)).view(np.int32))
    self._cone_dims[3] = ratio_bits
    w = self._workspace
    # Canonical row impedance may be borrowed from the per-world debug prefix.
    # Its batch stride is the full debug stride, not nr, so never flatten-bind
    # that view to Metal. Copy through this owned contiguous backing.
    w["row_impedance"].copy_(row_impedance)
    self._library.recompute_row_impedance(
        w["diagonal"].reshape(-1), w["row_impedance"].reshape(-1),
        w["active_rows"].reshape(-1), w["R"].reshape(-1),
        self._row_dims, threads=(self.batch_size, self.row_capacity),
        group_size=(1, 1))
    self._library.scale_contact_cones(
        w["R"].reshape(-1), w["active_rows"].reshape(-1),
        cone_groups.reshape(-1), friction.reshape(-1),
        w["cone_mu"].reshape(-1), self._cone_dims,
        threads=(self.batch_size, self.cone_capacity), group_size=(1, 1))
    self._library.adjusted_constraint_diagonal(
        w["R"].reshape(-1), w["row_impedance"].reshape(-1),
        w["active_rows"].reshape(-1), w["diagA"].reshape(-1),
        self._row_dims, threads=(self.batch_size, self.row_capacity),
        group_size=(1, 1))
    return {"R": w["R"], "diagA": w["diagA"], "contact_mu": w["cone_mu"]}

  def run_device(self, jacobian, inverse_jacobian, row_impedance, *,
                 active_rows, rhs_layout="row_major", impratio=1.0):
    """Compute exact diagonals then update impedance using retained buffers.

    ``inverse_jacobian`` is the exact result of applying the current mass
    factor to the canonical row right-hand sides. Row-major component output
    may have a padded second dimension (for example ``nr + 1``); the native
    contraction reads only the canonical ``nr`` rows. Dense multi-RHS output
    uses ``[B,nv,nr]``. Returned views are owned by this helper and are
    overwritten by its next call.
    """
    diagonal = self.exact_diagonal_device(
        jacobian, inverse_jacobian, active_rows=active_rows,
        rhs_layout=rhs_layout)
    updated = self.update_impedance_device(
        row_impedance, impratio=impratio)
    return {"diagonal": diagonal, **updated}
