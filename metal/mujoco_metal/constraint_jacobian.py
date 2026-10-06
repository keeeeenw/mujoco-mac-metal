# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Host-compiled structural sparsity for canonical constraint Jacobians.

The pattern is model-only and includes every DOF that can contribute to each
reserved logical row. Runtime values remain world-local. This module is kept
free of Torch so model validation and capacity planning work in CPU-only
environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


PACKED_J_MAGIC = 0x4D4A4353  # "MJCS"
PACKED_J_VERSION = 1
PACKED_J_HEADER_WORDS = 12
PACKED_J_DENSE = 0
PACKED_J_CSR = 1


@dataclass(frozen=True)
class PackedJacobianLayout:
  """Typed per-world storage record for a canonical constraint Jacobian.

  ``workspace_J`` remains an MPS float32 allocation so it can share the
  existing kernel binding.  The header and CSR maps occupy the same 32-bit
  words, but are initialized and accessed through int32 views; no integer is
  encoded through float arithmetic.  Numeric values follow the maps in the
  same record and are always float32.
  """

  mode: int
  nr: int
  nv: int
  nnz: int
  rowptr_offset: int
  columns_offset: int
  values_offset: int
  stride_words: int

  @classmethod
  def create(cls, nr, nv, *, pattern=None, mode=None):
    nr, nv = int(nr), int(nv)
    if nr < 0 or nv < 0:
      raise ValueError("packed Jacobian dimensions must be nonnegative")
    if mode is None:
      mode = PACKED_J_CSR if pattern is not None else PACKED_J_DENSE
    mode = int(mode)
    if mode not in (PACKED_J_DENSE, PACKED_J_CSR):
      raise ValueError("packed Jacobian mode must be dense or CSR")
    if mode == PACKED_J_CSR:
      if pattern is None or int(pattern.nr) != nr or int(pattern.nv) != nv:
        raise ValueError("CSR layout requires a matching compiled pattern")
      nnz = int(pattern.nnz)
      rowptr = PACKED_J_HEADER_WORDS
      columns = rowptr + nr + 1
      values = columns + nnz
      stride = values + nnz
    else:
      nnz = nr * nv
      rowptr = columns = 0
      values = PACKED_J_HEADER_WORDS
      stride = values + nnz
    if stride > np.iinfo(np.int32).max:
      raise ValueError("packed Jacobian record exceeds signed int32 addressing")
    return cls(mode, nr, nv, nnz, rowptr, columns, values, stride)

  def header(self):
    return np.asarray((PACKED_J_MAGIC, PACKED_J_VERSION, self.mode,
                       self.nr, self.nv, self.nnz, self.rowptr_offset,
                       self.columns_offset, self.values_offset,
                       self.stride_words, 0, 0), dtype=np.int32)


def initialize_packed_jacobian_records(storage_i32, batch_size, layout,
                                        pattern=None):
  """Initialize persistent packed records using a typed integer view.

  ``storage_i32`` must be a flat writable int32 view over the float32 kernel
  allocation.  Numeric value words are cleared; sparse maps are copied once
  from the immutable host pattern for each world record.
  """
  batch_size = int(batch_size)
  if batch_size <= 0:
    raise ValueError("batch_size must be positive")
  required = batch_size * layout.stride_words
  if required > np.iinfo(np.int32).max:
    raise ValueError("batched packed Jacobian exceeds signed int32 addressing")
  if storage_i32.dtype != np.int32 or storage_i32.ndim != 1:
    raise TypeError("packed Jacobian initializer requires flat int32 storage")
  if storage_i32.size < required:
    raise ValueError("packed Jacobian storage is smaller than its ABI record")
  if layout.mode == PACKED_J_CSR:
    if pattern is None or pattern.nr != layout.nr or pattern.nv != layout.nv:
      raise ValueError("CSR record initialization requires its matching pattern")
    if pattern.nnz != layout.nnz:
      raise ValueError("CSR pattern nnz changed after layout compilation")
  header = layout.header()
  for world in range(batch_size):
    base = world * layout.stride_words
    storage_i32[base:base + PACKED_J_HEADER_WORDS] = header
    if layout.mode == PACKED_J_CSR:
      storage_i32[base + layout.rowptr_offset:
                  base + layout.rowptr_offset + layout.nr + 1] = pattern.row_offsets
      storage_i32[base + layout.columns_offset:
                  base + layout.columns_offset + layout.nnz] = pattern.columns
      storage_i32[base + layout.values_offset:
                  base + layout.stride_words] = 0
    else:
      storage_i32[base + layout.values_offset:
                  base + layout.stride_words] = 0
  return required


def packed_jacobian_device_storage(torch, device, batch_size, pattern=None,
                                  *, nr=None, nv=None, mode=None):
  """Allocate and initialize per-world CSR records on ``device``.

  Each world owns a complete typed header, row pointer, and column map. This
  duplication is intentional: the Metal binding remains one flat float32
  buffer and the existing kernels derive each world base from the record
  header. No dense ``B*nr*nv`` stepping allocation is created.
  """
  batch_size = int(batch_size)
  if pattern is not None:
    nr, nv = int(pattern.nr), int(pattern.nv)
    mode = PACKED_J_CSR if mode is None else mode
  elif nr is None or nv is None:
    raise ValueError("dense packed storage requires explicit nr and nv")
  if mode is None:
    mode = PACKED_J_DENSE
  layout = PackedJacobianLayout.create(
      int(nr), int(nv), pattern=pattern, mode=mode)
  words = batch_size * layout.stride_words
  if words > np.iinfo(np.int32).max:
    raise ValueError("batched packed Jacobian exceeds signed int32 addressing")
  host_i32 = np.zeros(words, dtype=np.int32)
  initialize_packed_jacobian_records(host_i32, batch_size, layout, pattern)
  # View the initialized typed words as floats only for the established MSL
  # buffer argument ABI. The header/maps are never represented arithmetically.
  host_f32 = host_i32.view(np.float32).copy()
  storage = torch.as_tensor(host_f32, dtype=torch.float32, device=device).contiguous()
  return storage, layout


def materialize_packed_jacobian_torch(torch, storage, batch_size, layout,
                                      pattern, *, dtype=None):
  """Explicitly materialize a dense query view from packed device records.

  This helper is for APIs that explicitly request Jacobian diagnostics or
  inverse-query inputs. Solver stepping must consume ``storage`` directly.
  """
  b = int(batch_size)
  if int(storage.numel()) < b * layout.stride_words:
    raise ValueError("packed Jacobian storage is shorter than its batch layout")
  dtype = torch.float32 if dtype is None else dtype
  dense = torch.zeros((b, layout.nr, layout.nv), dtype=dtype,
                      device=storage.device)
  if layout.mode == PACKED_J_DENSE:
    result = []
    for world in range(b):
      base = world * layout.stride_words + layout.values_offset
      result.append(storage[base:base + layout.nr * layout.nv]
                    .reshape(layout.nr, layout.nv))
    return torch.stack(result, dim=0)
  if (pattern is None or pattern.nr != layout.nr
      or pattern.nnz != layout.nnz):
    raise ValueError("materialization pattern does not match packed layout")
  for row in range(layout.nr):
    begin = int(pattern.row_offsets[row])
    end = int(pattern.row_offsets[row + 1])
    if begin == end:
      continue
    cols = torch.as_tensor(pattern.columns[begin:end].copy(),
                           dtype=torch.int64, device=storage.device)
    slot = torch.arange(begin, end, dtype=torch.int64, device=storage.device)
    values_by_world = []
    for world in range(b):
      base = world * layout.stride_words + layout.values_offset
      values_by_world.append(storage[base + slot])
    vals = torch.stack(values_by_world, dim=0)
    dense[:, row, cols] = vals
  return dense


def packed_jacobian_scatter_rows_torch(torch, storage, batch_size, layout,
                                       pattern, source_rows, row_start,
                                       *, source_row_start=0, row_count=None,
                                       active=None):
  """Scatter a producer-local dense row block into the packed global CSR.

  ``source_rows`` may be only the producer's bounded row block (for example
  flex contact candidates), never the full canonical ``[B,nr,nv]`` matrix.
  The static gather/scatter maps are built from the compiled CSR pattern.
  """
  if layout.mode != PACKED_J_CSR or pattern.nnz != layout.nnz:
    raise ValueError("scatter requires matching packed CSR metadata")
  b = int(batch_size)
  count = int(pattern.nr if row_count is None else row_count)
  row_start, source_row_start = int(row_start), int(source_row_start)
  if (row_start < 0 or count < 0 or row_start + count > pattern.nr
      or tuple(source_rows.shape) != (b, count, layout.nv)):
    raise ValueError("producer row block does not match its canonical span")
  if active is not None and tuple(active.shape) != (b, count):
    raise ValueError("active mask does not match producer row block")
  source_indices, destination_slots = [], []
  destination_rows = []
  # Enforce the compiled structural support even though a producer may expose
  # a small dense row block. Silently gathering only allowed columns would
  # hide an incomplete support compiler and drop a real Jacobian term. Reduce
  # unsupported nonzeros per world on-device into the typed header status;
  # the normal solver entry validates that status before consuming rows.
  unsupported = torch.zeros((b,), dtype=torch.int32, device=storage.device)
  for local in range(count):
    row = row_start + local
    begin = int(pattern.row_offsets[row])
    end = int(pattern.row_offsets[row + 1])
    if begin < 0 or end < begin or end > layout.nnz:
      raise ValueError("compiled CSR row pointer is outside its value array")
    support = np.asarray(pattern.columns[begin:end], dtype=np.int64)
    if support.size and (np.any(support < 0) or np.any(support >= layout.nv)
                         or np.any(support[1:] <= support[:-1])):
      raise ValueError("compiled CSR columns are invalid or not strictly sorted")
    if support.size < layout.nv:
      allowed = torch.zeros(layout.nv, dtype=torch.bool, device=storage.device)
      if support.size:
        allowed[torch.as_tensor(support.copy(), dtype=torch.int64,
                                device=storage.device)] = True
      outside = source_rows[:, local, :][:, ~allowed]
      if outside.shape[1]:
        unsupported.bitwise_or_(torch.any(outside != 0.0, dim=1).to(
            torch.int32))
    for slot in range(begin, end):
      source_indices.append(local * layout.nv + int(pattern.columns[slot]))
      destination_slots.append(slot)
      destination_rows.append(local)
  records = storage.view(torch.int32).reshape(b, layout.stride_words)
  records[:, 10].bitwise_or_(unsupported)
  if not destination_slots:
    return
  src = torch.as_tensor(source_indices, dtype=torch.int64,
                        device=storage.device)
  dst = torch.as_tensor(destination_slots, dtype=torch.int64,
                        device=storage.device)
  gathered = source_rows.reshape(b, -1).index_select(1, src)
  if active is not None:
    row_id_tensor = torch.as_tensor(destination_rows, dtype=torch.int64,
                                    device=storage.device)
    gathered = gathered * active.index_select(1, row_id_tensor).to(gathered.dtype)
  for world in range(b):
    base = world * layout.stride_words + layout.values_offset
    storage[base + dst] = gathered[world]


def packed_jacobian_read(record_f32, layout, row, column):
  """Read one logical entry from a host-side packed record (test/oracle)."""
  row, column = int(row), int(column)
  if not (0 <= row < layout.nr and 0 <= column < layout.nv):
    raise IndexError("Jacobian index out of range")
  if layout.mode == PACKED_J_DENSE:
    return float(record_f32[layout.values_offset + row * layout.nv + column])
  # Lower_bound is used rather than a linear search so this mirrors the
  # bounded logarithmic MSL accessor used by device consumers.
  i32 = np.asarray(record_f32).view(np.int32)
  lo = int(i32[layout.rowptr_offset + row])
  hi = int(i32[layout.rowptr_offset + row + 1])
  target = column
  base = layout.columns_offset
  while lo < hi:
    mid = (lo + hi) // 2
    if int(i32[base + mid]) < target:
      lo = mid + 1
    else:
      hi = mid
  end = int(i32[layout.rowptr_offset + row + 1])
  if lo >= end or int(i32[base + lo]) != column:
    return 0.0
  return float(record_f32[layout.values_offset + lo])


def packed_jacobian_write(record_f32, layout, row, column, value):
  """Set one logical entry; return False for an unsupported nonzero write."""
  row, column, value = int(row), int(column), float(value)
  if not (0 <= row < layout.nr and 0 <= column < layout.nv):
    raise IndexError("Jacobian index out of range")
  if layout.mode == PACKED_J_DENSE:
    record_f32[layout.values_offset + row * layout.nv + column] = value
    return True
  i32 = np.asarray(record_f32).view(np.int32)
  lo = int(i32[layout.rowptr_offset + row])
  end = int(i32[layout.rowptr_offset + row + 1])
  hi, base = end, layout.columns_offset
  while lo < hi:
    mid = (lo + hi) // 2
    if int(i32[base + mid]) < column:
      lo = mid + 1
    else:
      hi = mid
  if lo >= end or int(i32[base + lo]) != column:
    return value == 0.0
  record_f32[layout.values_offset + lo] = value
  return True


@dataclass(frozen=True)
class ConstraintJacobianPattern:
  """Immutable CSR support for the coupled solver's canonical row order."""

  row_offsets: np.ndarray
  columns: np.ndarray
  nv: int
  nr: int

  @property
  def nnz(self):
    return int(self.columns.size)

  def row_columns(self, row):
    row = int(row)
    if row < 0 or row >= self.nr:
      raise IndexError("constraint row is out of range")
    return self.columns[int(self.row_offsets[row]):int(self.row_offsets[row + 1])]


def _dof_support_for_body(model, body_id):
  """DOFs on the compiled ancestor chain that affect ``body_id``."""
  body_id = int(body_id)
  nbody = int(model.nbody)
  if body_id < 0 or body_id >= nbody:
    return set()
  ancestors = set()
  while body_id > 0:
    ancestors.add(body_id)
    body_id = int(model.body_parentid[body_id])
  dof_body = np.asarray(model.dof_bodyid, dtype=np.int32)
  return {int(dof) for dof, owner in enumerate(dof_body)
          if int(owner) in ancestors}


def _body_for_object(model, objtype, objid):
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_BODY):
    return int(objid)
  if int(objtype) == int(mujoco.mjtObj.mjOBJ_SITE):
    return int(model.site_bodyid[int(objid)])
  return -1


def _flex_vertex_bodies(model, flex_id, vertices):
  """Return a conservative body support for compiled flex vertices.

  An interpolated candidate can move over an element interior, so sampling
  interpolation weights at the corner vertices is not a support proof. Use
  every compiled interpolation node body for that flex. Direct vertices use
  their compiled rigid body IDs.
  """
  vertices = np.asarray(vertices, dtype=np.int32).reshape(-1)
  if not vertices.size:
    return set()
  flex_id = int(flex_id)
  if int(model.flex_interp[flex_id]) == 0:
    vertadr = int(model.flex_vertadr[flex_id])
    vertnum = int(model.flex_vertnum[flex_id])
    valid = vertices[(vertices >= vertadr) & (vertices < vertadr + vertnum)]
    return set(map(int, np.asarray(model.flex_vertbodyid)[valid]))
  nodeadr = int(model.flex_nodeadr[flex_id])
  nodenum = int(model.flex_nodenum[flex_id])
  node_bodies = np.asarray(model.flex_nodebodyid, dtype=np.int32)
  return set(map(int, node_bodies[nodeadr:nodeadr + nodenum]))


def _joint_dofs(model, joint_id):
  joint_id = int(joint_id)
  if joint_id < 0 or joint_id >= int(model.njnt):
    return set()
  adr = int(model.jnt_dofadr[joint_id])
  kind = int(model.jnt_type[joint_id])
  width = {
      int(mujoco.mjtJoint.mjJNT_FREE): 6,
      int(mujoco.mjtJoint.mjJNT_BALL): 3,
      int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
      int(mujoco.mjtJoint.mjJNT_HINGE): 1,
  }.get(kind, 0)
  return set(range(adr, adr + width))


def _spatial_tendon_support(model, tendon_id):
  """Compile generalized-coordinate support from a spatial tendon's wraps.

  A spatial tendon length depends on the articulated transforms of its wrap
  sites/geometries and directly on any joint wrap.  The static union of those
  ancestor chains is conservative across every configuration while avoiding
  the previous all-DOF fallback.  Pulley wraps add a scalar path ratio; their
  adjacent site/geom wraps carry the moving-body support.
  """
  tendon_id = int(tendon_id)
  adr = int(model.tendon_adr[tendon_id])
  count = int(model.tendon_num[tendon_id])
  wrap_type = np.asarray(model.wrap_type, dtype=np.int32)
  wrap_objid = np.asarray(model.wrap_objid, dtype=np.int32)
  support = set()
  site_type = int(mujoco.mjtWrap.mjWRAP_SITE)
  geom_types = {int(mujoco.mjtWrap.mjWRAP_SPHERE),
                int(mujoco.mjtWrap.mjWRAP_CYLINDER)}
  joint_type = int(mujoco.mjtWrap.mjWRAP_JOINT)
  for wrap in range(adr, adr + count):
    kind, obj = int(wrap_type[wrap]), int(wrap_objid[wrap])
    if kind == site_type and 0 <= obj < int(model.nsite):
      body = int(model.site_bodyid[obj])
      support.update(_dof_support_for_body(model, body))
    elif kind in geom_types and 0 <= obj < int(model.ngeom):
      body = int(model.geom_bodyid[obj])
      support.update(_dof_support_for_body(model, body))
    elif kind == joint_type:
      support.update(_joint_dofs(model, obj))
    elif kind not in (int(mujoco.mjtWrap.mjWRAP_NONE),
                      int(mujoco.mjtWrap.mjWRAP_PULLEY)):
      raise ValueError(
          f"spatial tendon {tendon_id} has unsupported compiled wrap type {kind}")
  return support


def _tendon_support(model, descriptor, tendon_id):
  """Return the compiled DOF support for one fixed or spatial tendon."""
  tendon_id = int(tendon_id)
  if tendon_id < 0 or tendon_id >= int(model.ntendon):
    return set()
  moment = getattr(descriptor, "ten_moment_map", None)
  if moment is not None:
    values = np.asarray(moment)
    if values.ndim == 2 and tendon_id < values.shape[0]:
      support = set(map(int, np.flatnonzero(values[tendon_id] != 0)))
    else:
      support = set()
  else:
    support = set()
  from mujoco_metal.spatial_tendons import SpatialTendonModel
  if SpatialTendonModel(model).has_spatial:
    support.update(_spatial_tendon_support(model, tendon_id))
  return support


def _compiled_flex_jacobian_row(model, prefix, row):
  """Return one row's exact compiled flex-J support, or an empty set."""
  adr_name, nnz_name, col_name = (
      f"{prefix}_J_rowadr", f"{prefix}_J_rownnz", f"{prefix}_J_colind")
  try:
    rowadr = np.asarray(getattr(model, adr_name), dtype=np.int64).reshape(-1)
    rownnz = np.asarray(getattr(model, nnz_name), dtype=np.int64).reshape(-1)
    colind = np.asarray(getattr(model, col_name), dtype=np.int64).reshape(-1)
  except AttributeError:
    return set()
  row = int(row)
  if row < 0 or row >= rowadr.size or row >= rownnz.size:
    return set()
  start, count = int(rowadr[row]), int(rownnz[row])
  if start < 0 or count < 0 or start + count > colind.size:
    raise ValueError(f"compiled {prefix} Jacobian row {row} exceeds its column map")
  cols = colind[start:start + count]
  if np.any(cols < 0) or np.any(cols >= int(model.nv)):
    raise ValueError(f"compiled {prefix} Jacobian row {row} has an invalid column")
  return set(map(int, cols))


def _flexstrain_node_bodies(model, equality_id):
  """Compile the nodal body support for one flexstrain element equality."""
  flex_id = int(model.eq_obj1id[int(equality_id)])
  if flex_id < 0 or flex_id >= int(model.nflex):
    return set()
  order = abs(int(model.flex_interp[flex_id]))
  if order not in (1, 2):
    return set()
  nodeadr = int(model.flex_nodeadr[flex_id])
  nodenum = int(model.flex_nodenum[flex_id])
  if nodeadr < 0 or nodenum <= 0:
    return set()
  # The signed interpolation order and compiled flexstrain record determine
  # a cell or shell face. Volume cells have an exact rectangular nodal map;
  # shell faces use the union of nodes in that flex, which is conservative
  # across every compiled face and therefore cannot omit a moving DOF.
  signed_order = int(model.flex_interp[flex_id])
  if signed_order < 0:
    local_nodes = np.arange(nodeadr, nodeadr + nodenum, dtype=np.int32)
  else:
    cell = np.asarray(model.eq_data[int(equality_id), :3], dtype=np.int64)
    cellnum = np.asarray(model.flex_cellnum[flex_id], dtype=np.int64)
    ci, cj, ck = map(int, cell)
    cx, cy, cz = map(int, cellnum)
    if not (0 <= ci < cx and 0 <= cj < cy and 0 <= ck < cz):
      raise ValueError(f"flexstrain equality {equality_id} has an invalid cell")
    ny, nz = cy * order + 1, cz * order + 1
    local_nodes = np.asarray([
        nodeadr + (ci * order + li) * ny * nz
        + (cj * order + lj) * nz + ck * order + lk
        for li in range(order + 1)
        for lj in range(order + 1)
        for lk in range(order + 1)], dtype=np.int32)
  node_bodies = np.asarray(model.flex_nodebodyid, dtype=np.int32)
  if local_nodes.size and (local_nodes.min() < 0 or local_nodes.max() >= node_bodies.size):
    raise ValueError(f"flexstrain equality {equality_id} node support is out of range")
  return set(map(int, node_bodies[local_nodes]))


def compile_constraint_jacobian_pattern(model, descriptor):
  """Compile conservative exact structural supports for each logical row.

  Rows whose support is topology-local use body ancestry, compiled joint and
  wrap maps, or flex interpolation support. The resulting CSR is a static
  superset of configuration-dependent numeric support; runtime row values may
  be zero at a particular pose without changing their compiled slots.
  """
  nv, nr = int(descriptor.nv), int(descriptor.nr)
  rows = [set() for _ in range(nr)]
  all_dofs = set(range(nv))

  def add(start, count, support):
    support = {int(d) for d in support if 0 <= int(d) < nv}
    for row in range(max(int(start), 0), min(int(start) + int(count), nr)):
      rows[row].update(support)

  # Equality rows are ordered by compiled equality ID.
  for eid in range(int(model.neq)):
    start, count = int(descriptor.eq_rowadr[eid]), int(descriptor.eq_rownum[eid])
    kind = int(model.eq_type[eid])
    objtype = int(model.eq_objtype[eid])
    obj1, obj2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
    if kind == int(mujoco.mjtEq.mjEQ_JOINT):
      support = _joint_dofs(model, obj1) | _joint_dofs(model, obj2)
    elif kind in (int(mujoco.mjtEq.mjEQ_CONNECT),
                  int(mujoco.mjtEq.mjEQ_WELD)):
      bodies = (_body_for_object(model, objtype, obj1),
                _body_for_object(model, objtype, obj2))
      support = set().union(*(_dof_support_for_body(model, body)
                              for body in bodies))
    elif kind == int(mujoco.mjtEq.mjEQ_TENDON):
      support = (_tendon_support(model, descriptor, obj1)
                 | _tendon_support(model, descriptor, obj2))
    elif kind == int(mujoco.mjtEq.mjEQ_FLEX):
      # The compiled flex-edge CSR is the pinned structural support. Each
      # equality row ID is an edge ID, and rigid edges have no solver row.
      row_ids = descriptor.eq_row_ids[eid]
      for ordinal, edge in enumerate(np.asarray(row_ids, dtype=np.int32)):
        row = start + ordinal
        if row < min(start + count, nr):
          rows[row].update(_compiled_flex_jacobian_row(model, "flexedge", edge))
      continue
    elif kind == int(mujoco.mjtEq.mjEQ_FLEXVERT):
      # flexvert CSR rows are flattened as (vertex, invariant) in source
      # order, which is also the equality descriptor's row ID convention.
      row_ids = descriptor.eq_row_ids[eid]
      for ordinal, source_row in enumerate(np.asarray(row_ids, dtype=np.int32)):
        row = start + ordinal
        if row < min(start + count, nr):
          rows[row].update(_compiled_flex_jacobian_row(
              model, "flexvert", source_row))
      continue
    elif kind == int(mujoco.mjtEq.mjEQ_FLEXSTRAIN):
      bodies = _flexstrain_node_bodies(model, eid)
      support = set().union(*(_dof_support_for_body(model, body)
                              for body in bodies)) if bodies else set()
    else:
      support = all_dofs
    add(start, count, support)

  # DOF friction rows and two reserved rows per joint limit.
  friction_start = int(descriptor.n_eq_rows)
  for dof, frictionloss in enumerate(np.asarray(model.dof_frictionloss)):
    if float(frictionloss) > 0.0:
      add(friction_start + dof, 1, (dof,))
  limit_start = int(descriptor.n_eq_rows) + nv
  for joint in range(int(model.njnt)):
    if bool(model.jnt_limited[joint]):
      add(limit_start + 2 * joint, 2, _joint_dofs(model, joint))

  # Tendon friction and limit rows use their own fixed moment map or spatial
  # wrap ancestry. Distinct tendons may occupy disjoint generalized trees.
  tendon_start = int(descriptor.ten_base)
  total_friction = int(np.count_nonzero(
      np.asarray(model.tendon_frictionloss) > 0.0))
  friction_slot = 0
  limit_slot = 0
  for tendon in range(int(model.ntendon)):
    support = _tendon_support(model, descriptor, tendon)
    if float(model.tendon_frictionloss[tendon]) > 0.0:
      add(tendon_start + friction_slot, 1, support)
      friction_slot += 1
    if bool(model.tendon_limited[tendon]):
      add(tendon_start + total_friction + 2 * limit_slot, 2, support)
      limit_slot += 1

  # Rigid contact rows: every side can depend on the articulated ancestors of
  # its geometry body; all reserved manifold rows share that support.
  pair_offsets = np.asarray(descriptor.pair_contact_offset, dtype=np.int32)
  packed = np.asarray(descriptor.contact_condim_packed, dtype=np.int32)
  pair_rows = int(descriptor.nr_joint)
  for pair in range(int(descriptor.npairs)):
    body_ids = (int(model.geom_bodyid[int(descriptor.geom1[pair])]),
                int(model.geom_bodyid[int(descriptor.geom2[pair])]))
    support = set().union(*(_dof_support_for_body(model, body)
                            for body in body_ids))
    begin, end = int(pair_offsets[pair]), int(pair_offsets[pair + 1])
    for slot in range(begin, end):
      row_start = int(packed[3 * slot + 1])
      condim = int(packed[3 * slot])
      cone = int(packed[3 * slot + 2])
      span = (1 if condim == 1 else
              2 * (condim - 1) if cone == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
              else condim)
      add(pair_rows + row_start, span, support)

  # Flex contact row slots have static feature topology, including weighted
  # interpolation-node bodies for every source vertex/element pair.
  flex = getattr(descriptor, "flex_contact_descriptor", None)
  flex_base = int(getattr(descriptor, "flex_contact_base", nr))
  if flex is not None:
    for slot in range(int(flex.slot_count)):
      bodies = set()
      f1, f2 = int(flex.flex1[slot]), int(flex.flex2[slot])
      if f1 >= 0:
        verts = ([int(flex.vert1[slot])] if int(flex.vert1[slot]) >= 0 else
                 np.asarray(flex.nodes1[slot], dtype=np.int32))
        bodies.update(_flex_vertex_bodies(model, f1, verts))
      if f2 >= 0:
        verts = ([int(flex.vert2[slot])] if int(flex.vert2[slot]) >= 0 else
                 np.asarray(flex.nodes2[slot], dtype=np.int32))
        bodies.update(_flex_vertex_bodies(model, f2, verts))
      geom = int(flex.geom[slot])
      if geom >= 0:
        bodies.add(int(model.geom_bodyid[geom]))
      support = set().union(*(_dof_support_for_body(model, body)
                              for body in bodies)) if bodies else set()
      start = flex_base + int(flex.row_start[slot])
      add(start, int(flex.row_span[slot]), support)

  row_offsets = np.zeros(nr + 1, dtype=np.int64)
  for row, support in enumerate(rows):
    row_offsets[row + 1] = row_offsets[row] + len(support)
  nnz = int(row_offsets[-1])
  if nnz > np.iinfo(np.int32).max:
    raise ValueError("constraint Jacobian pattern exceeds signed int32 addressing")
  columns = np.empty(nnz, dtype=np.int32)
  cursor = 0
  for support in rows:
    ordered = sorted(support)
    columns[cursor:cursor + len(ordered)] = ordered
    cursor += len(ordered)
  row_offsets = row_offsets.astype(np.int32)
  row_offsets.setflags(write=False)
  columns.setflags(write=False)
  return ConstraintJacobianPattern(row_offsets, columns, nv, nr)
