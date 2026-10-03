# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Stable active-contact to kinematic-tree link mapping.

The map is generated from the *narrowphase slot flags*, not broadphase
candidates. Logical pair order is retained so the result is deterministic and
can be combined with equality links without a device count readback.
"""

from pathlib import Path

import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "active_contact_links.metal"
_LIBRARY = None


def pair_tree_ids(model, pair_geoms):
  """Return [npairs, 2] tree IDs for the descriptor's ordered geom pairs."""
  pairs = np.asarray(pair_geoms, dtype=np.int32)
  if pairs.ndim != 2 or pairs.shape[1] != 2:
    raise ValueError("pair_geoms must have shape [npairs, 2]")
  geom_body = np.asarray(model.geom_bodyid, dtype=np.int32)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  if pairs.size and (int(pairs.min()) < 0 or int(pairs.max()) >= geom_body.size):
    raise ValueError("pair_geoms contains a geom outside the model")
  body_ids = geom_body[pairs] if pairs.size else np.zeros((0, 2), dtype=np.int32)
  return body_tree[body_ids].astype(np.int32, copy=True)


def equality_tree_ids(model):
  """Tree pairs for body-body connect/weld equality descriptors.

  Equality families with pinned pairwise wake semantics are mapped directly.
  Tendon/flex equalities use hyperedge dependencies; they are marked with the
  reserved -2 pair so the device mapper fails open and keeps every tree awake.
  """
  import mujoco
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  pairs = np.full((int(model.neq), 2), -1, dtype=np.int32)
  supported = (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD))
  for eq in range(int(model.neq)):
    eq_type = int(model.eq_type[eq])
    obj_type = int(model.eq_objtype[eq])
    body1 = body2 = -1
    if eq_type in supported:
      if obj_type == int(mujoco.mjtObj.mjOBJ_BODY):
        body1, body2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
      elif obj_type == int(mujoco.mjtObj.mjOBJ_SITE):
        site1, site2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        site_body = np.asarray(model.site_bodyid, dtype=np.int32)
        if 0 <= site1 < site_body.size and 0 <= site2 < site_body.size:
          body1, body2 = int(site_body[site1]), int(site_body[site2])
    elif eq_type == int(mujoco.mjtEq.mjEQ_JOINT):
      joint_body = np.asarray(model.jnt_bodyid, dtype=np.int32)
      joint1, joint2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
      tree1 = (int(body_tree[joint_body[joint1]])
               if 0 <= joint1 < joint_body.size else -1)
      tree2 = (int(body_tree[joint_body[joint2]])
               if 0 <= joint2 < joint_body.size else -1)
      if (joint1 < -1 or joint2 < -1
          or (joint1 >= joint_body.size) or (joint2 >= joint_body.size)):
        pairs[eq] = (-2, -2)
      else:
        pairs[eq] = (tree1, tree2)
      continue
    elif eq_type in (int(mujoco.mjtEq.mjEQ_TENDON),
                     int(mujoco.mjtEq.mjEQ_FLEX),
                     int(mujoco.mjtEq.mjEQ_FLEXVERT),
                     int(mujoco.mjtEq.mjEQ_FLEXSTRAIN)):
      pairs[eq] = (-2, -2)
      continue
    else:
      continue
    if not (0 <= body1 < body_tree.size and 0 <= body2 < body_tree.size):
      pairs[eq] = (-2, -2)
      continue
    pairs[eq] = body_tree[[body1, body2]]
    if pairs[eq, 0] < 0 or pairs[eq, 1] < 0 or pairs[eq, 0] == pairs[eq, 1]:
      pairs[eq] = -1
  return pairs


def equality_tree_links(model):
  """Expand equality tree dependencies, including flex hyperedges.

  Returns fixed model metadata as ``(tree_pairs, equality_ids)``. Each tree
  pair is active exactly when the corresponding ``eq_active`` entry is set.
  Flex equalities are represented by a stable star over their distinct dynamic
  vertex/node trees. Tendon equalities retain MuJoCo's unsupported sleep
  behavior through a fail-open ``(-2, -2)`` marker.
  """
  import mujoco
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  links, owners = [], []

  def add_pair(eq, tree1, tree2):
    tree1, tree2 = int(tree1), int(tree2)
    if tree1 == -2 or tree2 == -2:
      links.append((-2, -2))
      owners.append(eq)
    elif tree1 >= 0 and tree2 >= 0 and tree1 != tree2:
      links.append((tree1, tree2))
      owners.append(eq)

  pairwise = (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD))
  flex_eq = (int(mujoco.mjtEq.mjEQ_FLEX), int(mujoco.mjtEq.mjEQ_FLEXVERT),
             int(mujoco.mjtEq.mjEQ_FLEXSTRAIN))
  for eq in range(int(model.neq)):
    eq_type = int(model.eq_type[eq])
    obj_type = int(model.eq_objtype[eq])
    id1, id2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
    if eq_type in pairwise:
      if obj_type == int(mujoco.mjtObj.mjOBJ_BODY):
        body1, body2 = id1, id2
      elif obj_type == int(mujoco.mjtObj.mjOBJ_SITE):
        site_body = np.asarray(model.site_bodyid, dtype=np.int32)
        if 0 <= id1 < site_body.size and 0 <= id2 < site_body.size:
          body1, body2 = int(site_body[id1]), int(site_body[id2])
        else:
          add_pair(eq, -2, -2)
          continue
      else:
        add_pair(eq, -2, -2)
        continue
      if 0 <= body1 < body_tree.size and 0 <= body2 < body_tree.size:
        add_pair(eq, body_tree[body1], body_tree[body2])
      else:
        add_pair(eq, -2, -2)
    elif eq_type == int(mujoco.mjtEq.mjEQ_JOINT):
      joint_body = np.asarray(model.jnt_bodyid, dtype=np.int32)
      trees = []
      valid = True
      for joint in (id1, id2):
        if joint == -1:
          trees.append(-1)
        elif 0 <= joint < joint_body.size:
          trees.append(int(body_tree[joint_body[joint]]))
        else:
          valid = False
      add_pair(eq, *(trees if valid else (-2, -2)))
    elif eq_type in flex_eq:
      fid = id1
      if not 0 <= fid < int(model.nflex):
        add_pair(eq, -2, -2)
        continue
      if bool(model.flex_interp[fid]):
        adr, count = int(model.flex_nodeadr[fid]), int(model.flex_nodenum[fid])
        bodyids = np.asarray(model.flex_nodebodyid, dtype=np.int32)[adr:adr + count]
      else:
        adr, count = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
        bodyids = np.asarray(model.flex_vertbodyid, dtype=np.int32)[adr:adr + count]
      # Keep first-occurrence order from the pinned body/node/vertex scan.
      # mj_wakeEquality selects the first awake tree and then the first
      # sleeping tree in this order; sorting would silently change which
      # island wakes when a flex equality spans more than two trees.
      trees, seen = [], set()
      for body in bodyids:
        body = int(body)
        if 0 <= body < body_tree.size:
          tree = int(body_tree[body])
          if tree >= 0 and tree not in seen:
            trees.append(tree)
            seen.add(tree)
      if len(trees) > 1:
        for tree in trees[1:]:
          add_pair(eq, trees[0], tree)
    elif eq_type == int(mujoco.mjtEq.mjEQ_TENDON):
      add_pair(eq, -2, -2)
  return (np.asarray(links, dtype=np.int32).reshape(-1, 2),
          np.asarray(owners, dtype=np.int32))


def equality_flex_hyperedges(model):
  """Return ordered flex-equality tree lists in pinned body scan order.

  The result is ``(offsets, tree_ids, equality_ids)``. The flattened member
  list deliberately preserves body order and repeated tree IDs, matching
  ``mj_wakeEquality``'s first-awake/first-sleeping scan. Ordinary equality
  pairs and tendon equalities are not included.
  """
  import mujoco
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  flex_eq = (int(mujoco.mjtEq.mjEQ_FLEX), int(mujoco.mjtEq.mjEQ_FLEXVERT),
             int(mujoco.mjtEq.mjEQ_FLEXSTRAIN))
  offsets = [0]
  flat, owners = [], []
  for eq in range(int(model.neq)):
    if int(model.eq_type[eq]) in flex_eq:
      fid = int(model.eq_obj1id[eq])
      if 0 <= fid < int(model.nflex):
        if bool(model.flex_interp[fid]):
          adr, count = int(model.flex_nodeadr[fid]), int(model.flex_nodenum[fid])
          bodyids = np.asarray(model.flex_nodebodyid, dtype=np.int32)[adr:adr + count]
        else:
          adr, count = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
          bodyids = np.asarray(model.flex_vertbodyid, dtype=np.int32)[adr:adr + count]
        for body in bodyids:
          body = int(body)
          if 0 <= body < body_tree.size and body_tree[body] >= 0:
            flat.append(int(body_tree[body]))
      owners.append(eq)
      offsets.append(len(flat))
  return (np.asarray(offsets, dtype=np.int32),
          np.asarray(flat, dtype=np.int32),
          np.asarray(owners, dtype=np.int32))


def equality_wake_plan(model):
  """Lower the equality wake cases implemented by pinned ``mj_wakeEquality``.

  Returns a dictionary of immutable model metadata:

  * ``pair_trees``/``pair_eq_ids``: connect, weld, and joint pairs in model
    order, excluding static and same-tree operands.
  * ``flex_offsets``/``flex_tree_ids``/``flex_eq_ids``: ordered dynamic tree
    members for each flex/flexvert/flexstrain equality, preserving body order.
  * ``unsupported_tendon_eq_ids``: tendon equalities, for which pinned 3.10
    calls ``mjERROR`` rather than defining sleep wake semantics.
  * ``unsupported_eq_ids``: unknown/invalid equality lowering, which must fail
    explicitly rather than silently treating it as a contact link.
  """
  import mujoco
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  pair_trees, pair_ids = [], []
  flex_offsets, flex_trees, flex_ids = [0], [], []
  unsupported_tendon, unsupported = [], []
  pair_types = (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD))
  flex_types = (int(mujoco.mjtEq.mjEQ_FLEX), int(mujoco.mjtEq.mjEQ_FLEXVERT),
                int(mujoco.mjtEq.mjEQ_FLEXSTRAIN))
  for eq in range(int(model.neq)):
    eq_type = int(model.eq_type[eq])
    obj_type = int(model.eq_objtype[eq])
    id1, id2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
    if eq_type in pair_types:
      bodies = None
      if obj_type == int(mujoco.mjtObj.mjOBJ_BODY):
        bodies = (id1, id2)
      elif obj_type == int(mujoco.mjtObj.mjOBJ_SITE):
        site_body = np.asarray(model.site_bodyid, dtype=np.int32)
        if 0 <= id1 < site_body.size and 0 <= id2 < site_body.size:
          bodies = (int(site_body[id1]), int(site_body[id2]))
      if bodies is None or any(b < 0 or b >= body_tree.size for b in bodies):
        unsupported.append(eq)
        continue
      a, b = int(body_tree[bodies[0]]), int(body_tree[bodies[1]])
    elif eq_type == int(mujoco.mjtEq.mjEQ_JOINT):
      joint_body = np.asarray(model.jnt_bodyid, dtype=np.int32)
      trees = []
      for joint in (id1, id2):
        if joint == -1:
          trees.append(-1)
        elif 0 <= joint < joint_body.size:
          trees.append(int(body_tree[joint_body[joint]]))
        else:
          trees = []
          break
      if len(trees) != 2:
        unsupported.append(eq)
        continue
      a, b = trees
    elif eq_type in flex_types:
      fid = id1
      if not 0 <= fid < int(model.nflex):
        unsupported.append(eq)
        flex_ids.append(eq)
        flex_offsets.append(len(flex_trees))
        continue
      if bool(model.flex_interp[fid]):
        adr, count = int(model.flex_nodeadr[fid]), int(model.flex_nodenum[fid])
        bodyids = np.asarray(model.flex_nodebodyid, dtype=np.int32)[adr:adr + count]
      else:
        adr, count = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
        bodyids = np.asarray(model.flex_vertbodyid, dtype=np.int32)[adr:adr + count]
      for body in bodyids:
        body = int(body)
        if 0 <= body < body_tree.size and body_tree[body] >= 0:
          flex_trees.append(int(body_tree[body]))
      flex_ids.append(eq)
      flex_offsets.append(len(flex_trees))
      continue
    elif eq_type == int(mujoco.mjtEq.mjEQ_TENDON):
      unsupported_tendon.append(eq)
      continue
    else:
      unsupported.append(eq)
      continue
    # Pinned mj_wakeEquality ignores static operands and equalities internal
    # to one tree. Both are deliberately absent from the wake-pair list.
    if a >= 0 and b >= 0 and a != b:
      pair_trees.append((a, b))
      pair_ids.append(eq)
  return {
      "pair_trees": np.asarray(pair_trees, dtype=np.int32).reshape(-1, 2),
      "pair_eq_ids": np.asarray(pair_ids, dtype=np.int32),
      "flex_offsets": np.asarray(flex_offsets, dtype=np.int32),
      "flex_tree_ids": np.asarray(flex_trees, dtype=np.int32),
      "flex_eq_ids": np.asarray(flex_ids, dtype=np.int32),
      "unsupported_tendon_eq_ids": np.asarray(unsupported_tendon, dtype=np.int32),
      "unsupported_eq_ids": np.asarray(unsupported, dtype=np.int32),
  }


def active_contact_tree_links_cpu(slot_flags, pair_contact_offset,
                                  pair_trees, capacity=None, *,
                                  equality_trees=None, equality_active=None,
                                  equality_activity_ids=None):
  """CPU oracle: expose one link for each pair owning an active narrowphase slot."""
  flags = np.asarray(slot_flags)
  if flags.ndim == 1:
    flags = flags.reshape(1, -1)
  offsets = np.asarray(pair_contact_offset, dtype=np.int64)
  trees = np.asarray(pair_trees, dtype=np.int32)
  if flags.ndim != 2 or flags.dtype.kind not in "biuf":
    raise ValueError("slot_flags must be numeric [batch, ncontacts]")
  if not np.all(np.isfinite(flags)):
    raise ValueError("slot_flags must be finite")
  if offsets.ndim != 1 or offsets.size != trees.shape[0] + 1:
    raise ValueError("pair_contact_offset must have npairs + 1 entries")
  if trees.ndim != 2 or trees.shape[1] != 2:
    raise ValueError("pair_trees must have shape [npairs, 2]")
  if offsets[0] != 0 or offsets[-1] != flags.shape[1] or np.any(np.diff(offsets) < 0):
    raise ValueError("pair contact offsets must partition all logical slots")
  npairs = trees.shape[0]
  eq_trees = (np.zeros((0, 2), dtype=np.int32) if equality_trees is None
              else np.asarray(equality_trees, dtype=np.int32))
  if eq_trees.ndim != 2 or eq_trees.shape[1] != 2:
    raise ValueError("equality_trees must have shape [nlinks, 2]")
  eq_ids = (np.arange(eq_trees.shape[0], dtype=np.int32)
            if equality_activity_ids is None
            else np.asarray(equality_activity_ids, dtype=np.int32))
  if eq_ids.shape != (eq_trees.shape[0],) or np.any(eq_ids < 0):
    raise ValueError("equality_activity_ids must have shape [nlinks]")
  if equality_active is None:
    neq = int(eq_ids.max()) + 1 if eq_ids.size else 0
    eq_active = np.ones((flags.shape[0], neq), dtype=bool)
  else:
    raw_eq_active = np.asarray(equality_active)
    if raw_eq_active.dtype.kind not in "biuf" or not np.all(np.isfinite(raw_eq_active)):
      raise ValueError("equality_active must be finite numeric data")
    eq_active = raw_eq_active != 0
    if eq_active.ndim == 1:
      eq_active = np.broadcast_to(eq_active[None, :],
                                  (flags.shape[0], eq_active.size))
    if eq_active.ndim != 2 or eq_active.shape[0] != flags.shape[0]:
      raise ValueError("equality_active must have shape [batch, neq]")
  if eq_ids.size and np.any(eq_ids >= eq_active.shape[1]):
    raise ValueError("equality_activity_ids reference an absent equality")
  eq_link_active = eq_active[:, eq_ids] if eq_ids.size else np.zeros((flags.shape[0], 0), dtype=bool)
  linked_eq = eq_link_active & (eq_trees[None, :, 0] >= 0) & (eq_trees[None, :, 1] >= 0)
  linked_eq &= (eq_trees[None, :, 0] != eq_trees[None, :, 1])
  unsupported_eq = eq_link_active & np.all(eq_trees[None, :, :] == -2, axis=2)
  all_count = npairs + eq_trees.shape[0]
  if capacity is None:
    capacity = all_count
  if isinstance(capacity, bool) or int(capacity) != capacity or int(capacity) < 0:
    raise ValueError("capacity must be a nonnegative integer")
  capacity = int(capacity)
  batch = flags.shape[0]
  links = np.full((batch, capacity, 2), -1, dtype=np.int32)
  overflow = np.zeros((batch,), dtype=np.int32)
  active_pairs = np.zeros((batch, npairs), dtype=bool)
  for pair in range(npairs):
    active_pairs[:, pair] = np.any(
        flags[:, int(offsets[pair]):int(offsets[pair + 1])] != 0, axis=1)
    # A contact against the world does not link two moving trees. Likewise,
    # same-tree contacts cannot merge/wake independent sleeping cycles.
    active_pairs[:, pair] &= ((trees[pair, 0] >= 0)
                              & (trees[pair, 1] >= 0)
                              & (trees[pair, 0] != trees[pair, 1]))
  for world in range(batch):
    if np.any(unsupported_eq[world]):
      overflow[world] = 1
      continue
    active = np.concatenate((active_pairs[world], linked_eq[world]))
    if np.any(active[capacity:]):
      overflow[world] = 1
      continue
    if capacity:
      kept = min(capacity, all_count)
      all_trees = np.concatenate((trees, eq_trees), axis=0)
      links[world, :kept][active[:kept]] = all_trees[:kept][active[:kept]]
  return links, overflow


class ActiveContactLinkWorkspace:
  """Persistent MPS mapping from slot compaction to active tree links."""

  def __init__(self, batch, pair_contact_offset, pair_trees, device="mps",
               capacity=None, equality_trees=None, equality_active0=None,
               equality_activity_ids=None):
    import torch
    offsets = np.asarray(pair_contact_offset, dtype=np.int32)
    trees = np.asarray(pair_trees, dtype=np.int32)
    if offsets.ndim != 1 or trees.ndim != 2 or trees.shape[1] != 2:
      raise ValueError("expected offsets [npairs + 1] and pair_trees [npairs, 2]")
    if offsets.size != trees.shape[0] + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
      raise ValueError("pair contact offsets must partition logical slot space")
    self.batch = int(batch)
    self.npairs = int(trees.shape[0])
    self.ncontacts = int(offsets[-1])
    eq_trees = (np.zeros((0, 2), dtype=np.int32) if equality_trees is None
                else np.asarray(equality_trees, dtype=np.int32))
    if eq_trees.ndim != 2 or eq_trees.shape[1] != 2:
      raise ValueError("equality_trees must have shape [nlinks, 2]")
    self.nequality_links = int(eq_trees.shape[0])
    if equality_activity_ids is None:
      eq_ids = np.arange(self.nequality_links, dtype=np.int32)
    else:
      eq_ids = np.asarray(equality_activity_ids, dtype=np.int32)
    if eq_ids.shape != (self.nequality_links,) or np.any(eq_ids < 0):
      raise ValueError("equality_activity_ids must have shape [nlinks]")
    eq0 = (np.ones((int(eq_ids.max()) + 1,), dtype=np.int32) if eq_ids.size
           else np.zeros((0,), dtype=np.int32)) if equality_active0 is None else \
        np.asarray(equality_active0, dtype=np.int32)
    if eq0.ndim != 1:
      raise ValueError("equality_active0 must have shape [neq]")
    if eq0.size == 1 and eq0[0] == 0 and self.nequality_links == 0:
      # Compiled descriptors store empty arrays as one zero sentinel.
      eq0 = eq0[:0]
    self.neq = int(eq0.size)
    if eq_ids.size and np.any(eq_ids >= self.neq):
      raise ValueError("equality_activity_ids reference an absent equality")
    if self.batch <= 0:
      raise ValueError("batch must be positive")
    if capacity is None:
      capacity = self.npairs + self.nequality_links
    if isinstance(capacity, bool) or int(capacity) != capacity or int(capacity) < 0:
      raise ValueError("capacity must be a nonnegative integer")
    self.capacity = int(capacity)
    self.device = torch.device(device)
    if self.device.type != "mps":
      raise ValueError("active contact link workspace requires MPS")
    self.pair_contact_offset = torch.as_tensor(
        offsets.copy(), dtype=torch.int32, device=self.device)
    self.pair_trees = torch.as_tensor(trees.copy(), dtype=torch.int32,
                                      device=self.device)
    self.equality_trees = torch.as_tensor(eq_trees.copy(), dtype=torch.int32,
                                          device=self.device)
    self.equality_activity_ids = torch.as_tensor(
        eq_ids.copy(), dtype=torch.int32, device=self.device)
    self.equality_active0 = torch.as_tensor(eq0.copy(), dtype=torch.int32,
                                            device=self.device)
    self.equality_active_default = torch.as_tensor(
        np.broadcast_to(eq0, (self.batch, self.neq)).copy(),
        dtype=torch.int32, device=self.device)
    self.links = torch.full((self.batch, self.capacity, 2), -1,
                            dtype=torch.int32, device=self.device)
    self.overflow = torch.zeros((self.batch,), dtype=torch.int32,
                                device=self.device)
    self.dims = torch.tensor(
        (self.ncontacts, self.npairs, self.capacity, self.batch, self.neq,
         self.nequality_links),
        dtype=torch.int32, device=self.device)

  def run(self, slot_map, equality_active=None):
    """Map active slots from a ``CompactionMap`` without host synchronization."""
    import torch
    logical_to_packed = slot_map.logical_to_packed
    if (tuple(logical_to_packed.shape) != (self.batch, self.ncontacts)
        or logical_to_packed.device.type != "mps"
        or logical_to_packed.dtype != torch.int32
        or not logical_to_packed.is_contiguous()):
      raise ValueError("slot map must provide contiguous MPS int32 logical_to_packed")
    if equality_active is None:
      equality_active = self.equality_active_default
    if not isinstance(equality_active, torch.Tensor):
      raise ValueError("equality_active must be an MPS tensor")
    if (tuple(equality_active.shape) != (self.batch, self.neq)
        or equality_active.device.type != "mps"
        or equality_active.dtype != torch.int32
        or not equality_active.is_contiguous()):
      if self.neq == 0 and tuple(equality_active.shape) == (self.batch, 0):
        pass
      else:
        raise ValueError("equality_active must be contiguous MPS int32 [batch, neq]")
    global _LIBRARY
    if _LIBRARY is None:
      _LIBRARY = torch.mps.compile_shader(_SHADER.read_text())
    self.links.fill_(-1)
    self.overflow.zero_()
    _LIBRARY.map_active_contact_tree_links(
        logical_to_packed, self.pair_contact_offset, self.pair_trees,
        equality_active, self.equality_trees, self.equality_activity_ids,
        self.links, self.overflow, self.dims,
        threads=(self.batch * max(self.npairs + self.nequality_links, 1),), group_size=(1,))
    return self.links, self.overflow
