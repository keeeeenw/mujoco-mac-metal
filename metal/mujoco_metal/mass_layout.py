# Copyright 2026 The MuJoCo Metal contributors.
# Licensed under the Apache License, Version 2.0.
"""Pure compiled metadata for exact tree/component mass block layouts."""

import numpy as np


def compile_tree_mass_layout(body_treeid, dof_bodyid, *, tendon_treeid=None,
                             tendon_treenum=None, tendon_armature=None,
                             tendon_j_rowadr=None, tendon_j_rownnz=None,
                             tendon_j_colind=None, mass_rowadr=None,
                             mass_rownnz=None, mass_colind=None):
  """Compile exact dense component blocks for rigid-body inertia.

  Rigid-body inertia is block diagonal by kinematic tree. A tendon with
  nonzero effective armature contributes ``a * J.T @ J`` and can couple the
  trees in its compiled contributor set. Such trees are merged into one
  dense component block before allocating values. The component may contain
  zero cross-tree entries, but no nonzero mass term can fall outside its
  compiled component.
  """
  body_treeid = np.asarray(body_treeid, dtype=np.int32)
  dof_bodyid = np.asarray(dof_bodyid, dtype=np.int32)
  if body_treeid.ndim != 1 or dof_bodyid.ndim != 1:
    raise ValueError("body_treeid and dof_bodyid must be one-dimensional")
  if np.any(dof_bodyid < 0) or np.any(dof_bodyid >= body_treeid.size):
    raise ValueError("dof_bodyid contains an invalid body index")
  dof_treeid = body_treeid[dof_bodyid]
  if np.any(dof_treeid < 0):
    raise ValueError("every generalized DOF must belong to a kinematic tree")
  nonnegative = body_treeid[body_treeid >= 0]
  ntree = int(nonnegative.max()) + 1 if nonnegative.size else 0
  if dof_treeid.size and (ntree == 0 or int(dof_treeid.max()) >= ntree):
    raise ValueError("DOF tree ids are outside the compiled tree range")
  tree_dofadr = np.zeros(ntree, dtype=np.int32)
  tree_dofnum = np.zeros(ntree, dtype=np.int32)
  for tree in range(ntree):
    ids = np.flatnonzero(dof_treeid == tree)
    if ids.size:
      first, count = int(ids[0]), int(ids.size)
      if not np.array_equal(ids, np.arange(first, first + count)):
        raise ValueError("compiled tree DOFs must form contiguous block ranges")
      tree_dofadr[tree], tree_dofnum[tree] = first, count
  if tendon_treeid is None:
    tendon_treeid = np.zeros((0, 2), dtype=np.int32)
    tendon_treenum = np.zeros((0,), dtype=np.int32)
    tendon_armature = np.zeros((0,), dtype=np.float64)
  else:
    tendon_treeid = np.asarray(tendon_treeid, dtype=np.int32)
    tendon_treenum = np.asarray(tendon_treenum, dtype=np.int32)
    tendon_armature = np.asarray(tendon_armature, dtype=np.float64)
    if (tendon_treeid.ndim != 2 or tendon_treeid.shape[1] != 2
        or tendon_treenum.shape != (tendon_treeid.shape[0],)
        or tendon_armature.shape != (tendon_treeid.shape[0],)):
      raise ValueError("tendon tree metadata must have shapes [ntendon,2], [ntendon], [ntendon]")
    if not np.all(np.isfinite(tendon_armature)):
      raise ValueError("effective tendon armature must be finite")
  if tendon_j_rowadr is not None:
    tendon_j_rowadr = np.asarray(tendon_j_rowadr, dtype=np.int32)
    tendon_j_rownnz = np.asarray(tendon_j_rownnz, dtype=np.int32)
    tendon_j_colind = np.asarray(tendon_j_colind, dtype=np.int32)
    if (tendon_j_rowadr.shape != tendon_treenum.shape
        or tendon_j_rownnz.shape != tendon_treenum.shape
        or tendon_j_colind.ndim != 1
        or np.any(tendon_j_rowadr < 0) or np.any(tendon_j_rownnz < 0)
        or np.any(tendon_j_rowadr.astype(np.int64)
                  + tendon_j_rownnz.astype(np.int64) > tendon_j_colind.size)
        or np.any(tendon_j_colind < 0) or np.any(tendon_j_colind >= dof_treeid.size)):
      raise ValueError("tendon Jacobian sparsity metadata is invalid")
  if mass_rowadr is not None:
    mass_rowadr = np.asarray(mass_rowadr, dtype=np.int32)
    mass_rownnz = np.asarray(mass_rownnz, dtype=np.int32)
    mass_colind = np.asarray(mass_colind, dtype=np.int32)
    if (mass_rowadr.shape != (dof_treeid.size,)
        or mass_rownnz.shape != (dof_treeid.size,)
        or mass_colind.ndim != 1
        or np.any(mass_rowadr < 0) or np.any(mass_rownnz < 0)
        or np.any(mass_rowadr.astype(np.int64)
                  + mass_rownnz.astype(np.int64) > mass_colind.size)
        or np.any(mass_colind < 0) or np.any(mass_colind >= dof_treeid.size)):
      raise ValueError("compiled mass matrix sparsity metadata is invalid")
  parent = np.arange(ntree, dtype=np.int32)

  def find(tree):
    root = int(tree)
    while int(parent[root]) != root:
      parent[root] = parent[int(parent[root])]
      root = int(parent[root])
    return root

  for tendon, armature in enumerate(tendon_armature):
    count = int(tendon_treenum[tendon])
    if count < 0:
      raise ValueError(f"tendon {tendon} has an invalid compiled tree count")
    if tendon_j_rowadr is not None:
      start = int(tendon_j_rowadr[tendon])
      width = int(tendon_j_rownnz[tendon])
      dofs = np.unique(tendon_j_colind[start:start + width])
      trees = np.unique(dof_treeid[dofs])
      if armature != 0 and mass_rowadr is not None:
        support = set(map(int, dofs))
        for dof in dofs:
          first = find(int(dof_treeid[int(dof)]))
          adr, nnz = int(mass_rowadr[dof]), int(mass_rownnz[dof])
          for col in mass_colind[adr:adr + nnz]:
            if int(col) not in support:
              continue
            other = find(int(dof_treeid[int(col)]))
            if first != other:
              parent[max(first, other)] = min(first, other)
              first = find(first)
        continue
    elif count <= 2:
      trees = tendon_treeid[tendon, :count]
    else:
      # The pinned model stores at most two tree IDs even when the tendon spans
      # more. Until a model has the structural ten_J column map, merge all trees
      # conservatively so no armature cross-term can escape its component.
      trees = np.arange(ntree, dtype=np.int32)
    if np.any(trees < 0) or np.any(trees >= ntree):
      raise ValueError(f"tendon {tendon} has an invalid dynamic tree contributor")
    if armature != 0 and trees.size > 1:
      first = find(int(trees[0]))
      for tree in trees[1:]:
        other = find(int(tree))
        if first != other:
          parent[max(first, other)] = min(first, other)
          first = find(first)
  roots = [find(tree) for tree in range(ntree)]
  root_order = sorted(set(roots), key=lambda root: min(
      tree for tree, candidate in enumerate(roots) if candidate == root))
  root_to_component = {root: index for index, root in enumerate(root_order)}
  tree_component = np.asarray(
      [root_to_component[root] for root in roots], dtype=np.int32)
  ncomponent = len(root_order)
  component_dofs = [[] for _ in range(ncomponent)]
  dof_component = np.empty(dof_treeid.size, dtype=np.int32)
  dof_local_index = np.empty(dof_treeid.size, dtype=np.int32)
  for dof, tree in enumerate(dof_treeid):
    component = int(tree_component[int(tree)])
    dof_component[dof] = component
    component_dofs[component].append(dof)
  component_dof_offsets = np.zeros(ncomponent + 1, dtype=np.int32)
  component_dof_ids_parts = []
  component_mass_offsets = np.zeros(ncomponent, dtype=np.int32)
  cursor = 0
  mass_cursor = 0
  for component, ids in enumerate(component_dofs):
    ids = np.asarray(ids, dtype=np.int32)
    component_dof_ids_parts.append(ids)
    component_dof_offsets[component] = cursor
    if ids.size:
      dof_local_index[ids] = np.arange(ids.size, dtype=np.int32)
    cursor += int(ids.size)
    component_dof_offsets[component + 1] = cursor
    block_size = int(ids.size) * int(ids.size)
    if mass_cursor > np.iinfo(np.int32).max or block_size > np.iinfo(np.int32).max - mass_cursor:
      raise ValueError("component mass block offsets exceed the Metal int32 ABI")
    component_mass_offsets[component] = mass_cursor
    mass_cursor += block_size
  component_dof_ids = (np.concatenate(component_dof_ids_parts)
                       if component_dof_ids_parts else np.zeros(0, np.int32))
  legacy_tree_mass_offsets = np.zeros(ntree, dtype=np.int32)
  tree_cursor = 0
  for tree, count in enumerate(tree_dofnum):
    legacy_tree_mass_offsets[tree] = tree_cursor
    tree_cursor += int(count) * int(count)
  packed = np.concatenate((tree_dofadr, tree_dofnum,
                           dof_treeid if dof_treeid.size else np.zeros(1, np.int32),
                           legacy_tree_mass_offsets)).astype(np.int32, copy=False)
  # Packed component ABI: component DOF offsets, component-ordered DOF ids,
  # per-DOF component, per-DOF local index, and component matrix offsets.
  component_packed = np.concatenate((
      component_dof_offsets,
      component_dof_ids if component_dof_ids.size else np.zeros(1, np.int32),
      dof_component if dof_component.size else np.zeros(1, np.int32),
      dof_local_index if dof_local_index.size else np.zeros(1, np.int32),
      component_mass_offsets)).astype(np.int32, copy=False)
  old_packed = packed
  return {
      "ntree": ntree,
      "nv": int(dof_treeid.size),
      "dof_treeid": dof_treeid.astype(np.int32, copy=True),
      "tree_dofadr": tree_dofadr,
      "tree_dofnum": tree_dofnum,
      "tree_mass_offsets": legacy_tree_mass_offsets,
      "tree_nnz": tree_cursor,
      "packed": old_packed,
      "ncomponent": ncomponent,
      "tree_component": tree_component,
      "component_dof_offsets": component_dof_offsets,
      "component_dof_ids": component_dof_ids,
      "component_dofnum": np.diff(component_dof_offsets).astype(np.int32),
      "component_mass_offsets": component_mass_offsets,
      "dof_component": dof_component.astype(np.int32, copy=True),
      "dof_local_index": dof_local_index.astype(np.int32, copy=True),
      "component_packed": component_packed,
      "nnz": mass_cursor,
  }
