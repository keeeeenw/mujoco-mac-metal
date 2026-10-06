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

"""Canonical active constraint-island partitioning reference.

This CPU implementation is the oracle for the fixed-shape device producer.
It follows MuJoCo's solver-island row ownership rules, which are distinct
from sleep-cycle IDs and kinematic-tree labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SolverIslandMap:
  """One-world logical island labels and stable row packing."""

  dof_island: np.ndarray       # [nv], -1 when its tree has no active row
  row_island: np.ndarray       # [nr], -1 for inactive or unowned rows
  row_order: np.ndarray        # [nr], active logical rows grouped by island
  row_inverse: np.ndarray     # [nr], packed position or -1
  island_offsets: np.ndarray  # [nisland + 1], offsets into row_order
  island_tree_count: np.ndarray  # [nisland]
  midpoint_blocked_tree: np.ndarray  # [ntree], tendon-limit/friction shortcut
  midpoint_blocked_body: np.ndarray  # [nbody], exact contact/equality bodies

  @property
  def island_count(self) -> int:
    return int(self.island_offsets.size - 1)


@dataclass(frozen=True)
class DeviceSolverIslandMap:
  """Borrowed fixed-shape MPS island maps (valid until the next run)."""

  dof_island: object
  row_island: object
  row_order: object
  row_inverse: object
  island_offsets: object
  island_count: object


def _pgs_pcg32_next(state, increment):
  """One MuJoCo 3.10 PCG32 transition for source-order oracles."""
  mask64 = (1 << 64) - 1
  mask32 = (1 << 32) - 1
  old = state
  state = (old * 6364136223846793005 + (increment | 1)) & mask64
  xorshifted = (((old >> 18) ^ old) >> 27) & mask32
  rotation = old >> 59
  value = ((xorshifted >> rotation)
           | (xorshifted << ((-rotation) & 31))) & mask32
  return state, value


def pgs_island_sweep_orders(row_order, island_offsets, iterations,
                            elliptic_blocks=()):
  """Return pinned PGS block visitation order independently per island.

  ``row_order`` is the compact, ascending logical-row order from the island
  producer and ``island_offsets`` partitions it. ``elliptic_blocks`` contains
  ``(start, dimension)`` pairs in logical-row coordinates; each elliptic
  contact occupies one PGS block, while pyramidal rows remain scalar blocks.
  MuJoCo initializes a fresh PCG32 state per call to ``solPGS_island``.
  """
  rows = _int_vector("row_order", row_order)
  offsets = _int_vector("island_offsets", island_offsets)
  if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != rows.size:
    raise ValueError("island_offsets must span row_order from zero to its length")
  if np.any(offsets[1:] < offsets[:-1]):
    raise ValueError("island_offsets must be nondecreasing")
  if (isinstance(iterations, (bool, np.bool_))
      or not isinstance(iterations, (int, np.integer)) or iterations < 0):
    raise ValueError("iterations must be a non-negative integer")
  starts = {}
  members = set()
  for start, dimension in elliptic_blocks:
    if (isinstance(start, (bool, np.bool_))
        or not isinstance(start, (int, np.integer))
        or isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))):
      raise ValueError("elliptic block start and dimension must be integers")
    start, dimension = int(start), int(dimension)
    if dimension <= 1 or start < 0 or start + dimension > (int(rows.max()) + 1 if rows.size else 0):
      raise ValueError("elliptic block range is invalid")
    block_rows = set(range(start, start + dimension))
    if members & block_rows:
      raise ValueError("elliptic blocks may not overlap")
    members |= block_rows
    starts[start] = dimension

  all_islands = []
  for island in range(offsets.size - 1):
    segment = rows[int(offsets[island]):int(offsets[island + 1])]
    block_order = []
    for row in segment:
      row = int(row)
      if row in members:
        if row in starts:
          block_order.append(row)
      else:
        block_order.append(row)
    state, increment = 0, 1
    state, _ = _pgs_pcg32_next(state, increment)
    sweeps = []
    for _ in range(int(iterations)):
      for index in range(len(block_order) - 1, 0, -1):
        state, random_value = _pgs_pcg32_next(state, increment)
        other = random_value % (index + 1)
        block_order[index], block_order[other] = block_order[other], block_order[index]
      sweeps.append(tuple(block_order))
    all_islands.append(tuple(sweeps))
  return tuple(all_islands)


def lower_solver_row_metadata(model, descriptor):
  """Lower static MuJoCo row identity and structural tree shortcuts.

  Returns int32 ``[nr,10]`` rows containing
  ``(type,id,tree0,tree1,exempt,shortcut,body0,body1,mid_tree0,mid_tree1)``.
  Dynamic activity and generic-J tree contributors are determined by the
  solver after it assembles the current world. The metadata preserves source
  identity for repeated equality/contact/joint blocks, and the tree iterator
  shortcuts for contacts, joint friction/limits, and connect/weld equalities.
  """
  import mujoco

  nr = int(descriptor.nr)
  out = np.zeros((nr, 10), dtype=np.int32)
  out[:, 0] = -1
  out[:, 1] = np.arange(nr, dtype=np.int32)
  out[:, 2:4] = -1
  out[:, 4] = 0
  out[:, 5] = 0
  out[:, 6:10] = -1
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)

  def fill(start, count, ctype, ident, tree0=-1, tree1=-1, exempt=False,
           shortcut=False, body0=-1, body1=-1, mid_tree0=-1, mid_tree1=-1):
    lo, hi = max(0, int(start)), min(nr, int(start) + int(count))
    if hi <= lo:
      return
    out[lo:hi, 0] = int(ctype)
    out[lo:hi, 1] = int(ident)
    out[lo:hi, 2] = int(tree0)
    out[lo:hi, 3] = int(tree1)
    out[lo:hi, 4] = int(bool(exempt))
    out[lo:hi, 5] = int(bool(shortcut))
    out[lo:hi, 6] = int(body0)
    out[lo:hi, 7] = int(body1)
    out[lo:hi, 8] = int(mid_tree0)
    out[lo:hi, 9] = int(mid_tree1)

  # Equality row spans are compiled into the canonical prefix. Only connect
  # and weld have structural shortcuts in treeIterInit; flex equality rows
  # deliberately scan each row and are exempt from same-id reuse.
  eq_type = np.asarray(model.eq_type, dtype=np.int32)
  eq_objtype = np.asarray(model.eq_objtype, dtype=np.int32)
  eq_obj1 = np.asarray(model.eq_obj1id, dtype=np.int32)
  eq_obj2 = np.asarray(model.eq_obj2id, dtype=np.int32)
  eq_adr = np.asarray(descriptor.eq_rowadr, dtype=np.int32)
  eq_num = np.asarray(descriptor.eq_rownum, dtype=np.int32)
  connect = int(mujoco.mjtEq.mjEQ_CONNECT)
  weld = int(mujoco.mjtEq.mjEQ_WELD)
  flex_eq_types = {int(getattr(mujoco.mjtEq, name)) for name in
                   ("mjEQ_FLEX", "mjEQ_FLEXVERT", "mjEQ_FLEXSTRAIN")
                   if hasattr(mujoco.mjtEq, name)}
  for eid in range(int(model.neq)):
    tree0 = tree1 = -1
    body0 = body1 = -1
    if int(eq_type[eid]) in (connect, weld):
      obj_type = int(eq_objtype[eid])
      b0, b1 = int(eq_obj1[eid]), int(eq_obj2[eid])
      if obj_type == int(mujoco.mjtObj.mjOBJ_SITE):
        b0 = int(model.site_bodyid[b0])
        b1 = int(model.site_bodyid[b1])
      body0, body1 = b0, b1
      if 0 <= b0 < body_tree.size and 0 <= b1 < body_tree.size:
        tree0, tree1 = int(body_tree[b0]), int(body_tree[b1])
    fill(eq_adr[eid], eq_num[eid], mujoco.mjtConstraint.mjCNSTR_EQUALITY,
         eid, tree0, tree1, exempt=int(eq_type[eid]) in flex_eq_types,
         shortcut=int(eq_type[eid]) in (connect, weld),
         body0=body0, body1=body1)

  # One potential dry-friction row per DOF.
  frictionloss = np.asarray(model.dof_frictionloss)
  friction_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
  for dof in range(int(model.nv)):
    if frictionloss[dof] > 0:
      fill(descriptor.n_eq_rows + dof, 1, friction_type, dof,
           int(dof_tree[dof]), shortcut=True)

  # Two reserved side rows per joint. Ball/free limit blocks share the joint
  # id and are visited as one same-constraint island edge set.
  joint_adr = np.asarray(model.jnt_dofadr, dtype=np.int32)
  joint_limited = np.asarray(model.jnt_limited, dtype=bool)
  joint_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
  limit_base = descriptor.n_eq_rows + int(model.nv)
  for jid in range(int(model.njnt)):
    if not joint_limited[jid]:
      continue
    first_dof = int(joint_adr[jid])
    tree = int(dof_tree[first_dof]) if 0 <= first_dof < dof_tree.size else -1
    fill(limit_base + 2 * jid, 2, joint_type, jid, tree, shortcut=True)

  # Tendon limit/friction rows use numerical J for island construction and
  # retain one shared logical group per tendon id.
  if descriptor.ntendon:
    n_friction = int(descriptor.ten_friction_rows)
    n_before = 0
    friction_t = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
    limit_t = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
    tendon_treeid = np.asarray(model.tendon_treeid, dtype=np.int32)
    for tendon in range(int(descriptor.ntendon)):
      if float(model.tendon_frictionloss[tendon]) > 0:
        trees = tendon_treeid[tendon] if tendon_treeid.ndim == 2 else ()
        mid0 = int(trees[0]) if len(trees) > 0 else -1
        mid1 = int(trees[1]) if len(trees) > 1 else -1
        fill(descriptor.ten_base + n_before, 1, friction_t, tendon,
             mid_tree0=mid0, mid_tree1=mid1)
        n_before += 1
    limited_before = 0
    for tendon in range(int(descriptor.ntendon)):
      if bool(model.tendon_limited[tendon]):
        trees = tendon_treeid[tendon] if tendon_treeid.ndim == 2 else ()
        mid0 = int(trees[0]) if len(trees) > 0 else -1
        mid1 = int(trees[1]) if len(trees) > 1 else -1
        fill(descriptor.ten_base + n_friction + 2 * limited_before, 2,
             limit_t, tendon, mid_tree0=mid0, mid_tree1=mid1)
        limited_before += 1

  # Candidate slots retain stable logical identity by pair offsets. The
  # contact rows use the same static pair-tree shortcut as engine_island.c.
  contact_base = int(descriptor.nr_joint)
  offsets = np.asarray(descriptor.pair_contact_offset, dtype=np.int64)
  packed = np.asarray(descriptor.contact_condim_packed, dtype=np.int32)
  cone_pyramid = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
  for slot in range(int(descriptor.ncontacts_max)):
    cdim, row_offset, cone = (int(x) for x in packed[3 * slot:3 * slot + 3])
    rows = (1 if cdim == 1 else
            2 * (cdim - 1) if cone == cone_pyramid else cdim)
    pair = int(np.searchsorted(offsets[1:], slot, side="right"))
    if pair >= int(descriptor.npairs):
      continue
    body0 = int(model.geom_bodyid[int(descriptor.geom1[pair])])
    body1 = int(model.geom_bodyid[int(descriptor.geom2[pair])])
    tree0 = int(body_tree[body0]) if 0 <= body0 < body_tree.size else -1
    tree1 = int(body_tree[body1]) if 0 <= body1 < body_tree.size else -1
    if cdim == 1:
      ctype = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
    elif cone == cone_pyramid:
      ctype = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
    else:
      ctype = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)
    fill(contact_base + row_offset, rows, ctype, slot, tree0, tree1,
         shortcut=True, body0=body0, body1=body1)

  # Flex contact candidates reserve complete, stable row spans independently
  # of their per-world narrowphase activity.  Preserve each candidate slot as
  # the logical contact identity and use its compiled flex/geom support when
  # labeling the rows for island ordering and midpoint exclusion.
  flex_contacts = getattr(descriptor, "flex_contact_descriptor", None)
  if flex_contacts is not None:
    from mujoco_metal.flex_contact import (
        _KIND_ELEMENT_PAIR, _KIND_GEOM_ELEMENT,
        _KIND_INTERNAL_VERTEX_ELEMENT, _KIND_PLANE_VERTEX)

    flex_contact_base = int(descriptor.flex_contact_base)
    vert_body = np.asarray(model.flex_vertbodyid, dtype=np.int32)
    for slot in range(int(flex_contacts.slot_count)):
      start = flex_contact_base + int(flex_contacts.row_start[slot])
      span = int(flex_contacts.row_span[slot])
      if start < flex_contact_base or span <= 0 or start + span > nr:
        raise ValueError("flex contact row span is outside the canonical solver rows")
      cdim = int(flex_contacts.condim[slot])
      cone = int(flex_contacts.cone[slot])
      if cdim == 1:
        ctype = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
      elif cone == cone_pyramid:
        ctype = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
      else:
        ctype = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)

      kind = int(flex_contacts.kind[slot])
      geom = int(flex_contacts.geom[slot])
      body0 = body1 = -1
      trees0, trees1 = [], []
      if geom >= 0:
        geom_body = int(model.geom_bodyid[geom])
        body0 = geom_body
        if 0 <= geom_body < body_tree.size:
          trees0 = [int(body_tree[geom_body])]
      nodes1 = np.asarray(flex_contacts.nodes1[slot], dtype=np.int32)
      nodes2 = np.asarray(flex_contacts.nodes2[slot], dtype=np.int32)
      if kind in (_KIND_PLANE_VERTEX, _KIND_GEOM_ELEMENT):
        flex_nodes = nodes1 if np.any(nodes1 >= 0) else nodes2
        flex_trees = sorted({int(body_tree[int(vert_body[v])])
                             for v in flex_nodes if 0 <= int(v) < vert_body.size
                             and 0 <= int(vert_body[int(v)]) < body_tree.size})
        trees1 = flex_trees
        bodies = sorted({int(vert_body[v]) for v in flex_nodes
                         if 0 <= int(v) < vert_body.size
                         and int(vert_body[int(v)]) > 0})
        body1 = bodies[0] if bodies else -1
      elif kind == _KIND_INTERNAL_VERTEX_ELEMENT:
        side1 = [int(v) for v in nodes1 if 0 <= int(v) < vert_body.size]
        side2 = [int(v) for v in nodes2 if 0 <= int(v) < vert_body.size]
        trees0 = sorted({int(body_tree[int(vert_body[v])]) for v in side1
                         if 0 <= int(vert_body[v]) < body_tree.size})
        trees1 = sorted({int(body_tree[int(vert_body[v])]) for v in side2
                         if 0 <= int(vert_body[v]) < body_tree.size})
        body0 = min((int(vert_body[v]) for v in side1
                     if int(vert_body[v]) > 0), default=-1)
        body1 = min((int(vert_body[v]) for v in side2
                     if int(vert_body[v]) > 0), default=-1)
      elif kind == _KIND_ELEMENT_PAIR:
        side1 = [int(v) for v in nodes1 if 0 <= int(v) < vert_body.size]
        side2 = [int(v) for v in nodes2 if 0 <= int(v) < vert_body.size]
        trees0 = sorted({int(body_tree[int(vert_body[v])]) for v in side1
                         if 0 <= int(vert_body[v]) < body_tree.size})
        trees1 = sorted({int(body_tree[int(vert_body[v])]) for v in side2
                         if 0 <= int(vert_body[v]) < body_tree.size})
        body0 = min((int(vert_body[v]) for v in side1
                     if int(vert_body[v]) > 0), default=-1)
        body1 = min((int(vert_body[v]) for v in side2
                     if int(vert_body[v]) > 0), default=-1)

      # Rigid contact rows use the engine's two-body structural shortcut.
      # A flex contact is different: one interpolated vertex can depend on
      # several compiled node bodies/trees, while flex_vertbodyid names only
      # a placeholder body.  Let the assembled row Jacobian enumerate its
      # actual generalized-coordinate contributors so the island contains
      # every interpolation tree.  Flex wake links are exported separately
      # from the contact candidate and are not represented by these two slots.
      t0 = trees0[0] if trees0 else -1
      t1 = trees1[0] if trees1 else -1
      fill(start, span, ctype, int(descriptor.ncontacts_max) + slot,
           t0, t1, shortcut=False, body0=body0, body1=body1)

  return out


def pack_solver_dimension_metadata(scalar_dims, dof_tree, row_metadata,
                                  ntree, core_scratch, disable_island_bit,
                                  *, nbody=0):
  """Append signed-int32 tree/row metadata to the solver dimension ABI."""
  scalars = _int_vector("scalar_dims", scalar_dims)
  trees = _int_vector("dof_tree", dof_tree)
  rows = np.asarray(row_metadata)
  if rows.ndim != 2 or rows.shape[1] != 10 or rows.dtype.kind not in "iu":
    raise ValueError("row_metadata must be an integer [nr,10] array")
  if rows.size and (np.any(rows < np.iinfo(np.int32).min)
                    or np.any(rows > np.iinfo(np.int32).max)):
    raise ValueError("row_metadata contains a value outside signed int32")
  for name, value in (("ntree", ntree), ("core_scratch", core_scratch),
                      ("disable_island_bit", disable_island_bit),
                      ("nbody", nbody)):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))):
      raise ValueError(f"{name} must be an integer")
  ntree, core_scratch, disable_island_bit, nbody = (
      int(ntree), int(core_scratch), int(disable_island_bit), int(nbody))
  if ntree < 0 or core_scratch < 0 or disable_island_bit < 0 or nbody < 0:
    raise ValueError("island metadata dimensions must be non-negative")
  if max(ntree, core_scratch, disable_island_bit, nbody) > np.iinfo(np.int32).max:
    raise OverflowError("island metadata dimension exceeds signed int32")
  if trees.size == 0:
    trees = np.zeros(1, dtype=np.int32)
  if trees.size and np.any(trees < -1):
    raise ValueError("dof_tree entries must be -1 or nonnegative")
  dof_offset = scalars.size + 6
  row_offset = dof_offset + trees.size
  total = row_offset + rows.size
  if total > (1 << 31) - 1:
    raise OverflowError("solver dimension metadata exceeds signed int32 addressing")
  # The awake-tree mask follows the island/midpoint maps in each world's
  # int32 view of workspace_debug. It is before the independent qacc tail.
  awake_offset = (core_scratch + max(trees.size, 1) + max(rows.shape[0], 1)
                  + 4 * max(ntree, 1) + max(nbody, 1))
  if awake_offset > np.iinfo(np.int32).max:
    raise OverflowError("awake-tree scratch offset exceeds signed int32")
  header = np.asarray([ntree, dof_offset, row_offset, core_scratch,
                       disable_island_bit, awake_offset], dtype=np.int32)
  return np.concatenate((scalars, header, trees.astype(np.int32, copy=False),
                         rows.astype(np.int32, copy=False).reshape(-1)))


def _int_vector(name, value, *, length=None):
  arr = np.asarray(value)
  if arr.ndim != 1 or arr.dtype.kind not in "iu":
    raise ValueError(f"{name} must be a one-dimensional integer array")
  if arr.size and (np.any(arr < np.iinfo(np.int32).min)
                   or np.any(arr > np.iinfo(np.int32).max)):
    raise ValueError(f"{name} contains an index outside signed int32")
  out = arr.astype(np.int32, copy=False)
  if length is not None and out.size != length:
    raise ValueError(f"{name} must have length {length}")
  return out


def build_solver_islands(dof_tree, jacobian, row_active, *, row_links=None,
                         group_key=None, group_exempt=None,
                         midpoint_tree_ids=None, midpoint_body_ids=None,
                         body_tree=None):
  """Partition active rows using structural shortcuts and numerical J.

  Args:
    dof_tree: Compiled MuJoCo tree ID for each generalized DOF.
    jacobian: Canonical dense row Jacobian ``[nr, nv]`` for this world.
    row_active: Canonical activation flags ``[nr]``.
    row_links: Optional structural body-tree endpoint pairs ``[nr, K, 2]``.
      Contact, connect, and weld rows use these even when their current J row
      is zero, matching the structural shortcuts in ``mj_island.c``.
    midpoint_tree_ids: Optional tendon shortcut tree IDs ``[nr, K]`` for
      disabled-islands midpoint eligibility. Pinned tendon limit/friction
      rows inspect the first two tendon trees.
    midpoint_body_ids: Optional exact endpoint body IDs ``[nr, K]`` for
      contact and connect/weld midpoint eligibility. Pinned checks the free
      body ID, not every body in its kinematic tree.
    group_key: Optional signed integer key ``[nr]``. Active rows with the
      same nonnegative key share the first row's tree set, as pinned solver
      grouping requires for ordinary same-type/same-ID constraints.
    group_exempt: Optional bool ``[nr]``. Flex equality families that are
      explicitly exempt from same-key grouping set this bit.

  Returns a deterministic compact labeling ordered by each island's minimum
  tree ID, then logical row ID. No input is modified.
  """
  dof_tree = _int_vector("dof_tree", dof_tree)
  J = np.asarray(jacobian)
  if J.ndim != 2 or J.shape[1] != dof_tree.size:
    raise ValueError("jacobian must have shape [nr, nv]")
  if J.dtype.kind not in "fiu":
    raise ValueError("jacobian must be numeric")
  nr, nv = J.shape
  active_arr = np.asarray(row_active)
  if active_arr.shape != (nr,) or active_arr.dtype.kind not in "biu":
    raise ValueError("row_active must be a bool/integer vector of length nr")
  active = active_arr != 0
  if np.any(dof_tree < -1):
    raise ValueError("dof_tree entries must be -1 or nonnegative")
  if body_tree is None:
    body_tree_arr = np.empty(0, dtype=np.int32)
  else:
    body_tree_arr = _int_vector("body_tree", body_tree)
  ntree = max(int(np.max(dof_tree, initial=-1)),
              int(np.max(body_tree_arr, initial=-1))) + 1
  if np.any(body_tree_arr < -1):
    raise ValueError("body_tree entries must be -1 or nonnegative")

  def row_index_matrix(name, value, upper):
    if value is None:
      return np.empty((nr, 0), dtype=np.int32)
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[0] != nr or arr.dtype.kind not in "iu":
      raise ValueError(f"{name} must have shape [nr, K] and integer dtype")
    if arr.dtype.kind == "u" and arr.size and np.any(arr > np.iinfo(np.int32).max):
      raise ValueError(f"{name} contains an index outside signed int32")
    result = arr.astype(np.int32, copy=False)
    if np.any(result < -1) or np.any(result >= upper):
      raise ValueError(f"{name} contains an out-of-range ID")
    return result

  midpoint_trees = row_index_matrix(
      "midpoint_tree_ids", midpoint_tree_ids, ntree)
  midpoint_bodies = row_index_matrix(
      "midpoint_body_ids", midpoint_body_ids, len(body_tree_arr))

  if row_links is None:
    links = np.empty((nr, 0, 2), dtype=np.int32)
  else:
    raw_links = np.asarray(row_links)
    if raw_links.ndim != 3 or raw_links.shape[0] != nr or raw_links.shape[2] != 2:
      raise ValueError("row_links must have shape [nr, K, 2]")
    if raw_links.dtype.kind not in "iu":
      raise ValueError("row_links must contain integer tree IDs")
    if raw_links.dtype.kind == "u" and raw_links.size and np.any(raw_links > np.iinfo(np.int32).max):
      raise ValueError("row_links contains an index outside signed int32")
    links = raw_links.astype(np.int32, copy=False)
    if np.any(links < -1) or np.any(links >= ntree):
      raise ValueError("row_links contains a tree ID outside dof_tree")

  if group_key is None:
    groups = np.full((nr, 2), -1, dtype=np.int32)
  else:
    groups = np.asarray(group_key)
    if groups.shape == (nr,):
      groups = np.column_stack((groups, np.zeros(nr, dtype=np.int32)))
    if groups.shape != (nr, 2) or groups.dtype.kind not in "iu":
      raise ValueError("group_key must be an integer vector or [nr,2] array")
    if groups.size and (np.any(groups < np.iinfo(np.int32).min)
                        or np.any(groups > np.iinfo(np.int32).max)):
      raise ValueError("group_key contains an ID outside signed int32")
    groups = groups.astype(np.int32, copy=False)
  if group_exempt is None:
    exempt = np.zeros(nr, dtype=bool)
  else:
    exempt_arr = np.asarray(group_exempt)
    if exempt_arr.shape != (nr,) or exempt_arr.dtype.kind not in "biu":
      raise ValueError("group_exempt must be a bool vector of length nr")
    exempt = exempt_arr != 0

  parent = np.arange(ntree, dtype=np.int32)

  def find(tree):
    root = int(tree)
    while int(parent[root]) != root:
      root = int(parent[root])
    while int(parent[tree]) != tree:
      nxt = int(parent[tree])
      parent[tree] = root
      tree = nxt
    return root

  def union(a, b):
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
      # Canonical minimum root makes labels stable independent of row traversal.
      parent[max(ra, rb)] = min(ra, rb)

  row_trees: list[list[int]] = [[] for _ in range(nr)]
  midpoint_blocked_tree = np.zeros(ntree, dtype=bool)
  midpoint_blocked_body = np.zeros(len(body_tree_arr), dtype=bool)
  first_for_group = {}
  for row in range(nr):
    if not active[row]:
      continue
    key = tuple(int(value) for value in groups[row])
    if key[0] >= 0 and not exempt[row] and key in first_for_group:
      # findEdges copies efc_tree[i-1] and skips treeIterInit for repeated
      # type/ID rows. Do not scan this row's J or structural metadata.
      row_trees[row] = list(row_trees[first_for_group[key]])
      continue
    trees = set()
    for dof in np.flatnonzero(J[row] != 0):
      tree = int(dof_tree[dof])
      if tree >= 0:
        trees.add(tree)
    for t0, t1 in links[row]:
      if t0 >= 0:
        trees.add(int(t0))
      if t1 >= 0:
        trees.add(int(t1))
      if t0 >= 0 and t1 >= 0:
        union(int(t0), int(t1))
    ordered = sorted(trees)
    for left, right in zip(ordered, ordered[1:]):
      union(left, right)
    row_trees[row] = ordered
    if key[0] >= 0 and not exempt[row]:
      first_for_group.setdefault(key, row)

  for row in range(nr):
    if not active[row]:
      continue
    for tree in midpoint_trees[row]:
      if tree >= 0:
        midpoint_blocked_tree[int(tree)] = True
    for body in midpoint_bodies[row]:
      if body >= 0:
        midpoint_blocked_body[int(body)] = True

  # Compress all rows after every structural and grouping edge is known.
  # MuJoCo materializes singleton islands for otherwise-unconstrained trees
  # whenever there is at least one active constraint. With no rows the island
  # builder is skipped and DOFs remain marked unconstrained.
  active_trees = (sorted({find(tree) for trees in row_trees for tree in trees})
                  if any(active) else [])
  root_to_island = {root: island for island, root in enumerate(active_trees)}
  tree_active_island = np.full(ntree, -1, dtype=np.int32)
  tree_counts = np.zeros(len(active_trees), dtype=np.int32)
  for tree in range(ntree):
    root = find(tree)
    if root in root_to_island:
      island = root_to_island[root]
      tree_active_island[tree] = island
      tree_counts[island] += 1

  dof_island = np.full(nv, -1, dtype=np.int32)
  for dof, tree in enumerate(dof_tree):
    if tree >= 0:
      dof_island[dof] = tree_active_island[int(tree)]
  row_island = np.full(nr, -1, dtype=np.int32)
  rows_by_island: list[list[int]] = [[] for _ in active_trees]
  for row, trees in enumerate(row_trees):
    if not active[row] or not trees:
      continue
    island = int(tree_active_island[trees[0]])
    row_island[row] = island
    rows_by_island[island].append(row)

  row_order = np.full(nr, -1, dtype=np.int32)
  row_inverse = np.full(nr, -1, dtype=np.int32)
  offsets = [0]
  cursor = 0
  for rows in rows_by_island:
    for row in rows:
      row_order[cursor] = row
      row_inverse[row] = cursor
      cursor += 1
    offsets.append(cursor)
  return SolverIslandMap(
      dof_island=dof_island,
      row_island=row_island,
      row_order=row_order,
      row_inverse=row_inverse,
      island_offsets=np.asarray(offsets, dtype=np.int32),
      island_tree_count=tree_counts,
      midpoint_blocked_tree=midpoint_blocked_tree,
      midpoint_blocked_body=midpoint_blocked_body)


class MetalSolverIslandWorkspace:
  """Reusable fixed-shape MPS connected-component producer.

  ``row_tree_links`` is static per canonical logical row and contains
  structural shortcut endpoints (contact, joint friction/limit,
  connect/weld, and tendon first-two-tree links). Numerical Jacobian
  contributors are scanned in the device kernel for generic row families.
  """

  _SHADER = Path(__file__).with_name("shaders") / "solver_islands.metal"
  _I32_MAX = (1 << 31) - 1

  def __init__(self, dof_tree, row_tree_links, row_group, row_group_exempt,
               *, batch_size, device="mps"):
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("solver islands require PyTorch MPS compile_shader")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    self._torch = torch
    self.batch_size = int(batch_size)
    self._device = torch.device(device)
    if self._device.type != "mps":
      raise ValueError("MetalSolverIslandWorkspace requires an MPS device")
    self._dof_tree = _int_vector("dof_tree", dof_tree)
    self.nv = int(self._dof_tree.size)
    if np.any(self._dof_tree < -1):
      raise ValueError("dof_tree entries must be -1 or nonnegative")
    self.ntree = int(np.max(self._dof_tree, initial=-1)) + 1
    links = np.asarray(row_tree_links)
    if links.ndim != 2 or links.shape[1] != 2 or links.dtype.kind not in "iu":
      raise ValueError("row_tree_links must have shape [nr,2] and integer dtype")
    if links.size and (np.any(links < np.iinfo(np.int32).min)
                       or np.any(links > np.iinfo(np.int32).max)):
      raise ValueError("row_tree_links contains an index outside signed int32")
    self.nr = int(links.shape[0])
    self._row_links = links.astype(np.int32, copy=False)
    if np.any(self._row_links < -1) or np.any(self._row_links >= self.ntree):
      raise ValueError("row_tree_links contains an invalid tree ID")
    group = np.asarray(row_group)
    if group.shape != (self.nr, 2) or group.dtype.kind not in "iu":
      raise ValueError("row_group must have shape [nr,2] and integer dtype")
    if group.size and (np.any(group < np.iinfo(np.int32).min)
                       or np.any(group > np.iinfo(np.int32).max)):
      raise ValueError("row_group contains an ID outside signed int32")
    self._row_group = group.astype(np.int32, copy=False)
    exempt = np.asarray(row_group_exempt)
    if exempt.shape != (self.nr,) or exempt.dtype.kind not in "biu":
      raise ValueError("row_group_exempt must have shape [nr] and bool/integer dtype")
    self._row_group_exempt = (exempt != 0).astype(np.int32)
    for name, elements in (
        ("dof_tree", max(self.nv, 1)),
        ("row_tree_links", max(self.nr * 2, 1)),
        ("row_group", max(self.nr * 2, 1)),
        ("row_group_exempt", max(self.nr, 1)),
        ("jacobian", max(self.batch_size * self.nr * self.nv, 1)),
        ("row_island", max(self.batch_size * self.nr, 1)),
        ("row_order", max(self.batch_size * self.nr, 1)),
        ("row_inverse", max(self.batch_size * self.nr, 1)),
        ("dof_island", max(self.batch_size * self.nv, 1)),
        ("island_offsets", max(self.batch_size * (self.ntree + 1), 1)),
        ("parent", max(self.batch_size * self.ntree, 1)),
        ("used_tree", max(self.batch_size * self.ntree, 1)),
        ("root_label", max(self.batch_size * self.ntree, 1)),
        ("row_first_tree", max(self.batch_size * self.nr, 1)),
    ):
      if elements > self._I32_MAX:
        raise OverflowError(f"{name} element count exceeds signed int32 addressing")
    self._library = torch.mps.compile_shader(self._SHADER.read_text())
    self._kernel = self._library.build_solver_island_partition

    def int_tensor(values, minimum=1):
      arr = np.asarray(values, dtype=np.int32).reshape(-1)
      if arr.size < minimum:
        arr = np.zeros(minimum, dtype=np.int32)
      return torch.as_tensor(arr.copy(), dtype=torch.int32, device=self._device)

    self._dof_tree_device = int_tensor(self._dof_tree)
    self._row_links_device = int_tensor(self._row_links)
    self._row_group_device = int_tensor(self._row_group)
    self._row_exempt_device = int_tensor(self._row_group_exempt)
    self._dims = torch.tensor(
        [self.nr, self.nv, self.ntree, self.batch_size, max(self.nr, 1), 0],
        dtype=torch.int32, device=self._device)
    b, nr, nv, ntree = self.batch_size, max(self.nr, 1), max(self.nv, 1), max(self.ntree, 1)
    self._dof_island = torch.empty((b, nv), dtype=torch.int32, device=self._device)
    self._row_island = torch.empty((b, nr), dtype=torch.int32, device=self._device)
    self._row_order = torch.empty((b, nr), dtype=torch.int32, device=self._device)
    self._row_inverse = torch.empty((b, nr), dtype=torch.int32, device=self._device)
    self._island_offsets = torch.empty((b, ntree + 1), dtype=torch.int32, device=self._device)
    self._island_count = torch.empty((b,), dtype=torch.int32, device=self._device)
    self._parent = torch.empty((b, ntree), dtype=torch.int32, device=self._device)
    self._used_tree = torch.empty((b, ntree), dtype=torch.int32, device=self._device)
    self._root_label = torch.empty((b, ntree), dtype=torch.int32, device=self._device)
    self._row_first_tree = torch.empty((b, nr), dtype=torch.int32, device=self._device)

  def run_device(self, jacobian, row_activity, *, activity_offset=0):
    """Run connectivity from J and an active-row float slice in debug storage.

    ``row_activity`` is a contiguous per-world float32 buffer; the active
    logical-row flags begin at ``activity_offset`` within each world row.
    This allows direct reuse of the coupled solver's packed workspace_debug
    without allocating a converted row-mask tensor.
    """
    torch = self._torch
    if (not isinstance(jacobian, torch.Tensor) or jacobian.dtype != torch.float32
        or tuple(jacobian.shape) != (self.batch_size, self.nr, self.nv)
        or jacobian.device.type != self._device.type
        or (self._device.index is not None
            and jacobian.device.index != self._device.index)
        or not jacobian.is_contiguous()):
      raise ValueError(
          f"jacobian must be contiguous float32 MPS with shape "
          f"{(self.batch_size, self.nr, self.nv)}")
    if (isinstance(activity_offset, (bool, np.bool_))
        or not isinstance(activity_offset, (int, np.integer))
        or activity_offset < 0):
      raise ValueError("activity_offset must be a non-negative integer")
    if (not isinstance(row_activity, torch.Tensor)
        or row_activity.dtype != torch.float32
        or row_activity.ndim != 2 or row_activity.shape[0] != self.batch_size
        or row_activity.shape[1] < int(activity_offset) + self.nr
        or row_activity.device.type != self._device.type
        or (self._device.index is not None
            and row_activity.device.index != self._device.index)
        or not row_activity.is_contiguous()):
      raise ValueError(
          "row_activity must be contiguous float32 MPS with per-world stride "
          f"at least {int(activity_offset) + self.nr}")
    self._dims[4] = int(row_activity.shape[1])
    self._dims[5] = int(activity_offset)
    self._kernel(
        self._dof_tree_device, jacobian.reshape(-1), row_activity.reshape(-1),
        self._row_links_device, self._row_group_device,
        self._row_exempt_device, self._dims, self._dof_island.reshape(-1),
        self._row_island.reshape(-1), self._row_order.reshape(-1),
        self._row_inverse.reshape(-1), self._island_offsets.reshape(-1),
        self._island_count, self._parent.reshape(-1),
        self._used_tree.reshape(-1), self._root_label.reshape(-1),
        self._row_first_tree.reshape(-1), threads=(self.batch_size,),
        group_size=(1,))
    return DeviceSolverIslandMap(
        dof_island=self._dof_island[:, :self.nv],
        row_island=self._row_island[:, :self.nr],
        row_order=self._row_order[:, :self.nr],
        row_inverse=self._row_inverse[:, :self.nr],
        island_offsets=self._island_offsets[:, :self.ntree + 1],
        island_count=self._island_count)
