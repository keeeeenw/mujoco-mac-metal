# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Device-side kinematic-tree sleep eligibility and wake scheduling.

The scheduler is called before dynamics/constraint work. It keeps MuJoCo's
signed ``tree_asleep`` countdown on-device: fully awake is ``-11`` (pinned
MuJoCo 3.10 ``-(1 + mjMINAWAKE)``), ``-1`` is eligible to sleep, and
nonnegative values are asleep. Solver/contact row support is supplied as
active tree links so newly coupled sleeping trees wake before solving.
"""

from pathlib import Path

import numpy as np

from mujoco_metal.islands import build_island_partition
from mujoco_metal.active_contact_links import equality_wake_plan

_SHADER = Path(__file__).parent / "shaders" / "sleep_schedule.metal"
_LIBRARY = None
_FULLY_AWAKE = -11


def _tree_link_count(value, batch, capacity, device, torch):
  """Validate a borrowed link tensor and return its logical link count.

  Inputs may use a shorter statically bounded link axis than the scheduler's
  prepared backing. The tail is filled with ``(-1, -1)`` before dispatch; an
  input larger than the prepared capacity is rejected rather than truncated.
  """
  shape = tuple(getattr(value, "shape", ()))
  if (len(shape) != 3 or shape[0] != int(batch) or shape[2] != 2
      or shape[1] > int(capacity)):
    raise ValueError(
        "active_tree_pairs must have shape [batch, links, 2] with "
        f"links <= {int(capacity)}; got {shape}")
  actual = getattr(value, "device", None)
  expected = torch.device(device)
  if (actual is None or actual.type != expected.type
      or (expected.index is not None and actual.index != expected.index)):
    raise ValueError(
        f"active_tree_pairs must be on {expected}; got {actual}")
  if getattr(value, "dtype", None) != torch.int32:
    raise ValueError("active_tree_pairs must have int32 dtype")
  contiguous = getattr(value, "is_contiguous", None)
  if not callable(contiguous) or not contiguous():
    raise ValueError("active_tree_pairs must be contiguous")
  return int(shape[1])


def sleep_scheduler_workspace_sizes(model, batch_size, max_tree_links, *,
                                    validate=True):
  """Host-only exact tensor inventory plus preallocation address check."""
  from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
  for name, value in (("batch_size", batch_size),
                      ("max_tree_links", max_tree_links)):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) or int(value) <= 0):
      raise ValueError(f"{name} must be a positive integer")
  b, links = int(batch_size), int(max_tree_links)
  nt, nb, nv, neq = (int(model.ntree), int(model.nbody), int(model.nv),
                     int(model.neq))
  flex_ids = int(equality_wake_plan(model)["flex_tree_ids"].size)
  sizes = {
      "sleep.initial_tree_state": max(nt, 1),
      "sleep.tree_state": b * max(nt, 1),
      "sleep.tree_awake": b * max(nt, 1),
      "sleep.pose_mismatch": b * max(nt, 1),
      "sleep.tree_dofadr": nt, "sleep.tree_dofnum": nt,
      "sleep.tree_bodyadr": nt, "sleep.tree_bodynum": nt,
      "sleep.tree_policy": nt, "sleep.dof_length": nv,
      "sleep.zero_qfrc": b * max(nv, 1), "sleep.zero_xfrc": b * nb * 6,
      "sleep.contact_equality_links": b * links * 2,
      "sleep.constraints_active": b, "sleep.links_overflow": b,
      "sleep.status": b, "sleep.eq_active_default": b * neq,
      "sleep.eq_active": b * neq, "sleep.eq_kinds": neq,
      "sleep.eq_pair_trees": 2 * neq, "sleep.eq_flex_offsets": neq + 1,
      "sleep.eq_flex_tree_ids": flex_ids,
      "sleep.body_treeid": nb, "sleep.body_parentid": nb,
      "sleep.dof_treeid": nv,
      "sleep.body_awake_ids": b * max(nb, 1),
      "sleep.parent_awake_ids": b * max(nb, 1),
      "sleep.body_awake_mask": b * max(nb, 1),
      "sleep.dof_awake_ids": b * max(nv, 1),
      "sleep.awake_counts": b * 3, "sleep.awake_list_dims": 4,
      "sleep.active_worlds_default": b,
      "sleep.active_worlds_converted": b,
      "sleep.zero_acceleration_dims": 3, "sleep.integer_dims": 11,
      "sleep.reset_rows_dims": 6,
      "sleep.tolerance": 1, "sleep.mode_wake": 1,
      "sleep.mode_advance": 1,
  }
  if validate:
    _validate_workspace_index_capacity(
        b, sizes, {"ntree": nt, "nbody": nb, "nv": nv, "neq": neq,
                   "max_tree_links": links, "flex_tree_ids": flex_ids})
    for name, elements in sizes.items():
      if elements > (1 << 31) - 1:
        raise ValueError(
            f"{name} exceeds the Metal int32 address range ({elements})")
  return sizes


def initial_sleep_state(model):
  """Lower pinned reset-time sleep state as immutable model metadata.

  Host model initialization is a preprocessing boundary, not a simulation
  fallback. With INIT trees, mj_resetData runs mj_sleep once on all trees:
  other eligible trees start at -10 and INIT islands form actual cycles.
  Reconstructing only singleton INIT cycles loses both behaviors.
  Runtime resets reuse this template without host physics or readback.
  """
  import mujoco
  state = np.full(max(int(model.ntree), 1), _FULLY_AWAKE, dtype=np.int32)
  enabled = bool(int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
  if enabled and np.any(np.asarray(model.tree_sleep_policy)
                        == int(mujoco.mjtSleepPolicy.mjSLEEP_INIT)):
    data = mujoco.MjData(model)
    state[:model.ntree] = data.tree_asleep
  return np.frombuffer(state.tobytes(), dtype=np.int32)


def validate_sleep_equalities(model, enabled):
  """Return pinned wake metadata or reject equality families with no policy."""
  plan = equality_wake_plan(model)
  if enabled and plan["unsupported_tendon_eq_ids"].size:
    ids = plan["unsupported_tendon_eq_ids"].tolist()
    raise NotImplementedError(
        f"MuJoCo 3.10 does not support sleeping for tendon equalities; "
        f"sleep-enabled model contains equality ids {ids}")
  if enabled and plan["unsupported_eq_ids"].size:
    ids = plan["unsupported_eq_ids"].tolist()
    raise NotImplementedError(
        f"sleep scheduling cannot lower equality ids {ids} to pinned wake semantics")
  return plan


def awake_lists_cpu(model, tree_awake):
  """Return MuJoCo 3.10 body/parent/DOF awake lists for test oracles.

  Lists are fixed-width, stable model-index order with ``-1`` padding. Static
  bodies (``body_treeid < 0``) participate in body/parent passes; DOFs are
  included only when their dynamic tree is awake. Parent entries follow the
  pinned rule and are included when their parent is not asleep, even if the
  child itself is asleep.
  """
  awake = np.asarray(tree_awake)
  if awake.ndim == 1:
    awake = awake[None, :]
  if awake.ndim != 2 or awake.shape[1] != max(int(model.ntree), 1):
    raise ValueError("tree_awake must have shape [batch, max(ntree, 1)]")
  if np.any((awake != 0) & (awake != 1)):
    raise ValueError("tree_awake values must be binary")
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  parent = np.asarray(model.body_parentid, dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  batch, nbody, nv = awake.shape[0], int(model.nbody), int(model.nv)
  bodies = np.full((batch, nbody), -1, dtype=np.int32)
  parents = np.full((batch, nbody), -1, dtype=np.int32)
  dofs = np.full((batch, nv), -1, dtype=np.int32)
  body_count = np.zeros(batch, dtype=np.int32)
  parent_count = np.zeros(batch, dtype=np.int32)
  dof_count = np.zeros(batch, dtype=np.int32)
  for world in range(batch):
    body_ids = [body for body in range(nbody)
                if body_tree[body] < 0 or awake[world, body_tree[body]] != 0]
    parent_ids = [body for body in range(1, nbody)
                  if (body_tree[parent[body]] < 0
                      or awake[world, body_tree[parent[body]]] != 0)]
    dof_ids = [dof for dof in range(nv)
               if dof_tree[dof] >= 0 and awake[world, dof_tree[dof]] != 0]
    body_count[world], parent_count[world], dof_count[world] = (
        len(body_ids), len(parent_ids), len(dof_ids))
    bodies[world, :len(body_ids)] = body_ids
    parents[world, :len(parent_ids)] = parent_ids
    dofs[world, :len(dof_ids)] = dof_ids
  return {
      "body_ids": bodies, "parent_ids": parents, "dof_ids": dofs,
      "body_mask": np.asarray([
          [(tree < 0 or awake[world, tree] != 0) for tree in body_tree]
          for world in range(batch)], dtype=np.int32),
      "body_count": body_count, "parent_count": parent_count,
      "dof_count": dof_count,
  }


def tree_sleep_transition_cpu(model, state, qvel, qfrc_applied, xfrc_applied,
                              active_tree_pairs=(), constraints_active=None,
                              active_equality_pairs=None, equality_active=None,
                              active_flex_equalities=(),
                              flex_equality_active=None, active_worlds=None):
  """Pinned-policy CPU reference for one sleep/wake scheduling pass.

  Inputs are batched NumPy arrays. Nonzero applied forces prevent sleep;
  speed uses MuJoCo's DOF-length-weighted infinity norm. This mirrors the
  pinned tree policy: controls are not direct sleep or wake criteria.
  """
  import mujoco
  partition = build_island_partition(model)
  state = np.asarray(state, dtype=np.int32).copy()
  qvel = np.asarray(qvel)
  qfrc_applied = np.asarray(qfrc_applied)
  xfrc_applied = np.asarray(xfrc_applied)
  batch, ntree = state.shape
  if active_worlds is None:
    active_worlds = np.ones(batch, dtype=np.int32)
  else:
    active_worlds = np.asarray(active_worlds)
    if active_worlds.shape != (batch,) or np.any((active_worlds != 0) & (active_worlds != 1)):
      raise ValueError("active_worlds must be a binary vector with shape [batch]")
  if ntree != partition.ntree:
    raise ValueError("state tree dimension does not match model")
  if not (qvel.shape == qfrc_applied.shape == (batch, int(model.nv))):
    raise ValueError("qvel/qfrc_applied must have shape [batch, nv]")
  if xfrc_applied.shape != (batch, int(model.nbody), 6):
    raise ValueError("xfrc_applied must have shape [batch, nbody, 6]")
  enabled = bool(int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
  disabled_islands = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ISLAND))
  if not enabled:
    state[active_worlds != 0] = _FULLY_AWAKE
    return state
  policy = partition.tree_sleep_policy
  velocity_tol = float(model.opt.sleep_tolerance)
  links = np.asarray(active_tree_pairs, dtype=np.int32)
  if links.size == 0:
    links = np.zeros((batch, 0, 2), dtype=np.int32)
  elif links.ndim == 2 and links.shape[-1] == 2:
    links = np.broadcast_to(links[None, :, :], (batch,) + links.shape)
  elif links.ndim != 3 or links.shape[0] != batch or links.shape[-1] != 2:
    raise ValueError("active_tree_pairs must have shape [links, 2] or [batch, links, 2]")
  if active_equality_pairs is None:
    equality_pairs = np.zeros((batch, 0, 2), dtype=np.int32)
  else:
    equality_pairs = np.asarray(active_equality_pairs, dtype=np.int32)
    if equality_pairs.ndim == 2 and equality_pairs.shape[-1] == 2:
      equality_pairs = np.broadcast_to(
          equality_pairs[None, :, :], (batch,) + equality_pairs.shape)
    if (equality_pairs.ndim != 3 or equality_pairs.shape[0] != batch
        or equality_pairs.shape[-1] != 2):
      raise ValueError("active_equality_pairs must have shape [links,2] or [batch,links,2]")
  eq_active = (np.ones((batch, equality_pairs.shape[1]), dtype=bool)
               if equality_active is None else np.asarray(equality_active) != 0)
  if eq_active.ndim == 1:
    eq_active = np.broadcast_to(eq_active[None, :],
                                (batch, eq_active.size))
  if eq_active.shape != (batch, equality_pairs.shape[1]):
    raise ValueError("equality_active must have shape [batch, equality links]")
  flex_edges = tuple(np.asarray(edge, dtype=np.int32).reshape(-1)
                     for edge in active_flex_equalities)
  flex_active = (np.ones((batch, len(flex_edges)), dtype=bool)
                 if flex_equality_active is None
                 else np.asarray(flex_equality_active) != 0)
  if flex_active.ndim == 1:
    flex_active = np.broadcast_to(flex_active[None, :],
                                  (batch, flex_active.size))
  if flex_active.shape != (batch, len(flex_edges)):
    raise ValueError("flex_equality_active must have shape [batch, hyperedges]")

  def cycle_members(world_state, tree):
    members = []
    current = tree
    for _ in range(ntree + 1):
      if current < 0 or current >= ntree or current_state_is_awake(world_state, current):
        raise ValueError("invalid sleeping-tree cycle")
      members.append(current)
      current = int(world_state[current])
      if current == tree:
        return members
    raise ValueError("sleeping-tree cycle does not close")

  def current_state_is_awake(world_state, tree):
    return int(world_state[tree]) < 0

  def wake_cycle(world_state, tree, wake_value):
    if int(world_state[tree]) < 0:
      world_state[tree] = min(int(wake_value), int(world_state[tree]))
      return
    members = cycle_members(world_state, tree)
    world_state[members] = int(wake_value)

  for world in range(batch):
    if active_worlds[world] == 0:
      continue
    if disabled_islands and constraints_active is not None and bool(constraints_active[world]):
      state[world].fill(_FULLY_AWAKE)
      continue
    can_sleep = np.ones((ntree,), dtype=bool)
    for tree in range(ntree):
      if int(policy[tree]) in (
          int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER),
          int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER)):
        can_sleep[tree] = False
        continue
      dofs = partition.tree_dofs[tree]
      bodies = partition.tree_bodies[tree]
      if dofs:
        dof_idx = np.asarray(dofs, dtype=np.int64)
        weighted_speed = np.max(
            np.abs(qvel[world, dof_idx]) * np.asarray(model.dof_length)[dof_idx])
        zero_velocity = np.all(qvel[world, dof_idx] == 0)
        moving = weighted_speed >= velocity_tol if velocity_tol else weighted_speed != 0
        applied_qfrc = np.any(qfrc_applied[world, dof_idx] != 0)
      else:
        moving = False
        zero_velocity = True
        applied_qfrc = False
      applied_xfrc = np.any(xfrc_applied[world, bodies] != 0) if bodies else False
      can_sleep[tree] = not (moving or applied_qfrc or applied_xfrc)
      # The pre-solve mj_wake pass uses treeCanSleep(..., 0), while the later
      # mj_sleep transition uses sleep_tolerance. A nonzero qvel below that
      # tolerance must wake an existing sleeping cycle before integration.
      can_wake = (zero_velocity and not applied_qfrc and not applied_xfrc
                  and int(policy[tree]) not in (
                      int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER),
                      int(mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER)))
      if state[world, tree] >= 0 and not can_wake:
        wake_cycle(state[world], tree, _FULLY_AWAKE)
      elif state[world, tree] < 0 and not can_sleep[tree]:
        state[world, tree] = _FULLY_AWAKE
    # Existing sleepers wake as a complete pinned island cycle when a
    # perturbation reaches any member.
    for tree in range(ntree):
      if state[world, tree] >= 0 and not can_sleep[tree]:
        wake_cycle(state[world], tree, _FULLY_AWAKE)

    # Active cross-tree rows preserve/propagate the pinned cycle association.
    # A newly coupled pair of distinct sleeping cycles must wake before solve.
    for _ in range(max(ntree, 1)):
      changed = False
      for a, b in links[world]:
        a, b = int(a), int(b)
        if not (0 <= a < ntree and 0 <= b < ntree and a != b):
          if a == -1 and b == -1:
            continue
          raise ValueError(f"invalid active tree pair {(a, b)}")
        asleep_a, asleep_b = state[world, a] >= 0, state[world, b] >= 0
        if asleep_a and not asleep_b:
          cycle = cycle_members(state[world], a)
          wake_cycle(state[world], a, int(state[world, b]))
          changed |= any(state[world, tree] < 0 for tree in cycle)
        elif asleep_b and not asleep_a:
          cycle = cycle_members(state[world], b)
          wake_cycle(state[world], b, int(state[world, a]))
          changed |= any(state[world, tree] < 0 for tree in cycle)
        elif asleep_a and asleep_b:
          # mj_wakeCollision treats any contact between sleeping trees as an
          # invariant violation, including trees in the same encoded cycle.
          raise ValueError(f"active contact joins sleeping trees {(a, b)}")
      if not changed:
        break

    # Pinned mj_wakeEquality processes ordinary pairwise equalities after
    # contacts. Both distinct sleeping cycles wake fully; an awake operand
    # wakes the sleeping operand fully; static and same-tree operands are
    # ignored. This is deliberately separate from contact wake propagation,
    # whose wake value follows the awake contact partner's countdown.
    for link, (a, b) in enumerate(equality_pairs[world]):
      if not eq_active[world, link]:
        continue
      a, b = int(a), int(b)
      if a == b or a < 0 or b < 0:
        continue
      if a >= ntree or b >= ntree:
        raise ValueError(f"invalid equality tree pair {(a, b)}")
      asleep_a, asleep_b = state[world, a] >= 0, state[world, b] >= 0
      if asleep_a and asleep_b:
        cycle_a = set(cycle_members(state[world], a))
        cycle_b = set(cycle_members(state[world], b))
        if cycle_a.isdisjoint(cycle_b):
          wake_cycle(state[world], a, _FULLY_AWAKE)
          wake_cycle(state[world], b, _FULLY_AWAKE)
      elif asleep_a:
        wake_cycle(state[world], a, _FULLY_AWAKE)
      elif asleep_b:
        wake_cycle(state[world], b, _FULLY_AWAKE)

    # A flex equality is a hyperedge over its dynamic node/vertex trees. Pinned
    # mj_wakeEquality scans body order, selects the first awake tree and wakes
    # only the first sleeping island to that tree's current countdown value.
    # If every member sleeps, it does not wake merely because cycles differ.
    for edge, trees in enumerate(flex_edges):
      if not flex_active[world, edge]:
        continue
      dynamic = [int(tree) for tree in trees if 0 <= int(tree) < ntree]
      awake_tree = next((tree for tree in dynamic if state[world, tree] < 0), None)
      if awake_tree is None:
        continue
      sleeping_tree = next((tree for tree in dynamic if state[world, tree] >= 0), None)
      if sleeping_tree is not None:
        wake_cycle(state[world], sleeping_tree, int(state[world, awake_tree]))

    # Advance the quiet countdown for awake trees. Non-sleepable trees stay
    # fully awake, as in mj_sleep after the pre-solve mj_wake sweep.
    for tree in range(ntree):
      if not can_sleep[tree]:
        state[world, tree] = _FULLY_AWAKE
      elif state[world, tree] < -1:
        state[world, tree] += 1

    # MuJoCo encodes each sleeping island as a closed cycle of tree ids.
    # At this point any component linked to a tree still counting down is not
    # eligible; isolated -1 trees become one-node cycles.
    sleep_links = [tuple(map(int, pair)) for pair in links[world]]
    sleep_links.extend(tuple(map(int, pair)) for i, pair in enumerate(equality_pairs[world])
                       if eq_active[world, i])
    for i, trees in enumerate(flex_edges):
      if flex_active[world, i]:
        dynamic = [int(tree) for tree in trees if 0 <= int(tree) < ntree]
        if dynamic:
          sleep_links.extend((dynamic[0], tree) for tree in dynamic[1:])
    adjacency = [[] for _ in range(ntree)]
    for a, b in sleep_links:
      a, b = int(a), int(b)
      if 0 <= a < ntree and 0 <= b < ntree and a != b:
        adjacency[a].append(b)
        adjacency[b].append(a)
    visited = set()
    for root in range(ntree):
      if root in visited or state[world, root] != -1:
        continue
      component, stack = [], [root]
      visited.add(root)
      while stack:
        tree = stack.pop()
        component.append(tree)
        for neighbor in adjacency[tree]:
          if neighbor not in visited and state[world, neighbor] < 0:
            visited.add(neighbor)
            stack.append(neighbor)
      if all(state[world, tree] == -1 for tree in component):
        ordered = sorted(component)
        for index, tree in enumerate(ordered):
          state[world, tree] = ordered[(index + 1) % len(ordered)]
  return state


class DeviceSleepScheduler:
  """Persistent MPS buffers and kernel for pre-solve tree scheduling."""

  def __init__(self, model, batch_size=1, device="mps", max_tree_links=None):
    import mujoco
    if (isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0):
      raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    ntree = int(model.ntree)
    if max_tree_links is None:
      max_tree_links = max(ntree * 8, 1)
    if (isinstance(max_tree_links, (bool, np.bool_))
        or not isinstance(max_tree_links, (int, np.integer))
        or int(max_tree_links) <= 0):
      raise ValueError("max_tree_links must be a positive integer")
    max_tree_links = int(max_tree_links)
    # Check every actual tensor shape before importing Torch or compiling.
    self._workspace_sizes = sleep_scheduler_workspace_sizes(
        model, batch_size, max_tree_links)
    enabled = bool(int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
    partition = build_island_partition(model)
    equality_plan = validate_sleep_equalities(model, enabled)
    expected_plan_shapes = {
        "eq_kinds": (int(model.neq),),
        "pair_trees": (int(model.neq), 2),
        "flex_offsets": (int(model.neq) + 1,),
        "flex_tree_ids": (self._workspace_sizes["sleep.eq_flex_tree_ids"],),
    }
    for name, shape in expected_plan_shapes.items():
      if np.asarray(equality_plan[name]).shape != shape:
        raise ValueError(
            f"lowered equality plan {name} has shape "
            f"{np.asarray(equality_plan[name]).shape}, expected {shape}")
    if np.asarray(model.eq_active0).shape != (int(model.neq),):
      raise ValueError("model eq_active0 must have shape [neq]")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("native sleep scheduling requires PyTorch MPS compile_shader")
    self.model = model
    self.batch_size = batch_size
    self.device = torch.device(device)
    if self.device.type != "mps":
      raise ValueError("DeviceSleepScheduler requires an MPS device")
    self.partition = partition
    self.ntree = self.partition.ntree
    self.enabled = enabled
    self.island_disabled = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ISLAND))
    self.equality_plan = equality_plan
    self.max_tree_links = max_tree_links
    global _LIBRARY
    if _LIBRARY is None:
      _LIBRARY = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = _LIBRARY.sleep_transition
    self._build_awake_lists_kernel = _LIBRARY.build_awake_lists
    self._copy_active_worlds_kernel = _LIBRARY.copy_active_worlds_bool
    self._reset_rows_kernel = _LIBRARY.reset_sleep_rows
    self._zero_asleep_acceleration_kernel = _LIBRARY.zero_asleep_acceleration
    batch, ntree = self.batch_size, max(self.ntree, 1)
    self._initial_tree_state = self._const(initial_sleep_state(model))
    self.tree_state = self._initial_tree_state.expand(batch, ntree).clone()
    self.tree_awake = torch.ones((batch, ntree), dtype=torch.int32, device=self.device)
    self.tree_awake.copy_((self.tree_state < 0).to(dtype=torch.int32))
    self._zero_pose_mismatch = torch.zeros(
        (batch, ntree), dtype=torch.int32, device=self.device)
    self.tree_dofadr = self._const(np.asarray(model.tree_dofadr, dtype=np.int32))
    self.tree_dofnum = self._const(np.asarray(model.tree_dofnum, dtype=np.int32))
    self.tree_bodyadr = self._const(np.asarray(model.tree_bodyadr, dtype=np.int32))
    self.tree_bodynum = self._const(np.asarray(model.tree_bodynum, dtype=np.int32))
    self.tree_policy = self._const(np.asarray(model.tree_sleep_policy, dtype=np.int32))
    # The transition shader consumes dof_length as float32. _const is the
    # integer metadata helper; using it here silently truncated the pinned
    # kinetic-distance weights to zero and let moving trees count toward sleep.
    self.dof_length = torch.as_tensor(
        np.asarray(model.dof_length, dtype=np.float32).copy(),
        dtype=torch.float32, device=self.device)
    self._zero_qfrc = torch.zeros((batch, max(int(model.nv), 1)),
                                  dtype=torch.float32, device=self.device)
    self._zero_xfrc = torch.zeros((batch, int(model.nbody), 6),
                                  dtype=torch.float32, device=self.device)
    self.links = torch.full((batch, self.max_tree_links, 2), -1,
                            dtype=torch.int32, device=self.device)
    self.constraints_active = torch.zeros((batch,), dtype=torch.int32, device=self.device)
    self.links_overflow = torch.zeros((batch,), dtype=torch.int32, device=self.device)
    self.status = torch.zeros((batch,), dtype=torch.int32, device=self.device)
    self.neq = int(model.neq)
    eq_active0 = np.asarray(model.eq_active0, dtype=np.int32).reshape(-1)
    if eq_active0.shape != (self.neq,):
      raise ValueError("model eq_active0 must have shape [neq]")
    self.eq_active_default = torch.as_tensor(
        np.broadcast_to(eq_active0, (batch, self.neq)).copy(),
        dtype=torch.int32, device=self.device)
    self.eq_active = self.eq_active_default.clone()
    self.eq_kinds = self._const(self.equality_plan["eq_kinds"])
    self.eq_pair_trees = self._const(self.equality_plan["pair_trees"])
    self.eq_flex_offsets = self._const(self.equality_plan["flex_offsets"])
    self.eq_flex_tree_ids = self._const(self.equality_plan["flex_tree_ids"])
    # MuJoCo's sleep-filtered smooth/RNE stages consume packed, ascending
    # body, parent, and DOF indices. Keep fixed-width storage and device counts
    # so sleeping trees require no host readback or dynamic allocation.
    nb, nv = int(model.nbody), int(model.nv)
    self.body_treeid = self._const(np.asarray(model.body_treeid, dtype=np.int32))
    self.body_parentid = self._const(np.asarray(model.body_parentid, dtype=np.int32))
    self.dof_treeid = self._const(np.asarray(model.dof_treeid, dtype=np.int32))
    self.body_awake_ids = torch.full(
        (batch, max(nb, 1)), -1, dtype=torch.int32, device=self.device)
    self.parent_awake_ids = torch.full_like(self.body_awake_ids, -1)
    self.body_awake_mask = torch.zeros(
        (batch, max(nb, 1)), dtype=torch.int32, device=self.device)
    self.dof_awake_ids = torch.full(
        (batch, max(nv, 1)), -1, dtype=torch.int32, device=self.device)
    self.awake_counts = torch.zeros((batch, 3), dtype=torch.int32,
                                    device=self.device)
    self.awake_list_dims = torch.tensor(
        (nb, nv, self.ntree, batch), dtype=torch.int32, device=self.device)
    self.active_worlds_default = torch.ones((batch,), dtype=torch.int32,
                                            device=self.device)
    self._active_worlds_converted = torch.empty_like(self.active_worlds_default)
    self._zero_acceleration_dims = torch.tensor(
        (nv, self.ntree, batch), dtype=torch.int32, device=self.device)
    # Integer dimensions share a separate tensor so no float conversion can
    # round counts on large worlds.
    self.int_dims = torch.tensor(
        (int(model.nv), int(model.nbody), int(model.nu), self.ntree,
         self.max_tree_links, self.batch_size, int(self.enabled),
         int(self.island_disabled), 1, self.neq,
         int(self.equality_plan["flex_tree_ids"].size)),
        dtype=torch.int32, device=self.device)
    self._reset_rows_dims = torch.tensor(
        (batch, max(self.ntree, 1), max(self.max_tree_links, 1), self.neq,
         nv, nb),
        dtype=torch.int32, device=self.device)
    self.sleep_tolerance = torch.tensor(
        (float(model.opt.sleep_tolerance),), dtype=torch.float32, device=self.device)
    self._mode_wake = torch.zeros((1,), dtype=torch.int32, device=self.device)
    self._mode_advance = torch.ones((1,), dtype=torch.int32, device=self.device)
    self._build_awake_lists()

  def _build_awake_lists(self, active_worlds=None):
    """Build pinned fixed-width awake lists from the current device mask."""
    if active_worlds is None:
      active_worlds = self.active_worlds_default
    self._build_awake_lists_kernel(
        self.tree_awake, self.body_treeid, self.body_parentid, self.dof_treeid,
        self.body_awake_ids, self.parent_awake_ids, self.dof_awake_ids,
        self.body_awake_mask,
        self.awake_counts, self.awake_list_dims, active_worlds,
        threads=(self.batch_size,), group_size=(1,))

  def awake_lists(self):
    """Return borrowed fixed buffers and per-world counts (no readback)."""
    return {
        "body_ids": self.body_awake_ids,
        "parent_ids": self.parent_awake_ids,
        "dof_ids": self.dof_awake_ids,
        "body_mask": self.body_awake_mask,
        "body_count": self.awake_counts[:, 0],
        "parent_count": self.awake_counts[:, 1],
        "dof_count": self.awake_counts[:, 2],
        "counts": self.awake_counts,
        "tree_awake": self.tree_awake,
        "list_dims": self.awake_list_dims,
    }

  def _const(self, array):
    import torch
    return torch.as_tensor(array.copy(), dtype=torch.int32, device=self.device)

  def reset(self, env_ids=None):
    import torch
    rows = (np.arange(self.batch_size, dtype=np.int64) if env_ids is None
            else np.asarray(env_ids, dtype=np.int64).reshape(-1))
    if np.any(rows < 0) or np.any(rows >= self.batch_size):
      raise IndexError("sleep reset environment index is outside batch")
    for world in rows.tolist():
      self.tree_state[world].copy_(self._initial_tree_state)
      self.tree_awake[world].copy_(
          (self.tree_state[world] < 0).to(dtype=torch.int32))
      self.links[world].fill_(-1)
      self.constraints_active[world].zero_()
      self.links_overflow[world].zero_()
      self.status[world].zero_()
      self.eq_active[world].copy_(self.eq_active_default[world])
    self._zero_qfrc.zero_()
    self._zero_xfrc.zero_()
    self._build_awake_lists()

  def reset_masked(self, mask):
    """Reset sleep cycles selected by an on-device boolean world mask."""
    import torch
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
        or tuple(mask.shape) != (self.batch_size,)
        or mask.device != self.tree_awake.device or not mask.is_contiguous()):
      raise ValueError("mask must be contiguous bool[batch_size] on scheduler device")
    if not hasattr(self, "_reset_rows_kernel"):
      def select(current, replacement):
        expanded = mask.reshape(self.batch_size,
                                *([1] * (current.ndim - 1)))
        return torch.where(expanded, replacement, current)
      initial = self._initial_tree_state.expand_as(self.tree_state)
      self.tree_state.copy_(select(self.tree_state, initial))
      self.tree_awake.copy_(select(
          self.tree_awake, (initial < 0).to(torch.int32)))
      self.links.copy_(select(self.links, torch.full_like(self.links, -1)))
      self.constraints_active.copy_(select(
          self.constraints_active, torch.zeros_like(self.constraints_active)))
      self.links_overflow.copy_(select(
          self.links_overflow, torch.zeros_like(self.links_overflow)))
      self.status.copy_(select(self.status, torch.zeros_like(self.status)))
      self.eq_active.copy_(select(self.eq_active, self.eq_active_default))
      self._zero_qfrc.copy_(select(self._zero_qfrc,
                                   torch.zeros_like(self._zero_qfrc)))
      self._zero_xfrc.copy_(select(self._zero_xfrc,
                                   torch.zeros_like(self._zero_xfrc)))
      self._build_awake_lists()
      return
    self._copy_active_worlds_kernel(
        mask, self._active_worlds_converted, self.awake_list_dims,
        threads=(self.batch_size,), group_size=(1,))
    self._reset_rows_kernel(
        self._active_worlds_converted, self._initial_tree_state,
        self.eq_active_default, self.tree_state, self.tree_awake, self.links,
        self.constraints_active, self.links_overflow, self.status,
        self.eq_active, self._zero_qfrc.reshape(-1),
        self._zero_xfrc.reshape(-1), self._reset_rows_dims,
        threads=(self.batch_size,), group_size=(1,))
    self._build_awake_lists(self._active_worlds_converted)

  def _run(self, qvel, *, qfrc_applied=None, xfrc_applied=None,
           active_tree_pairs=None, constraints_active=None, link_overflow=None,
           equality_active=None, active_worlds=None, pose_mismatch=None,
           mode):
    """Run one device wake or post-advance sleep pass."""
    import torch
    b, nv = self.batch_size, int(self.model.nv)
    if active_worlds is None:
      active_worlds = self.active_worlds_default
    elif (tuple(active_worlds.shape) != (b,)
          or active_worlds.device.type != "mps"
          or not active_worlds.is_contiguous()):
      raise ValueError("active_worlds must be a contiguous MPS vector [batch]")
    elif active_worlds.dtype == torch.bool:
      self._copy_active_worlds_kernel(
          active_worlds, self._active_worlds_converted,
          self.awake_list_dims, threads=(b,), group_size=(1,))
      active_worlds = self._active_worlds_converted
    elif active_worlds.dtype != torch.int32:
      raise ValueError("active_worlds must have bool or int32 dtype")
    if tuple(qvel.shape) != (b, nv) or qvel.device.type != "mps" or qvel.dtype != torch.float32:
      raise ValueError(f"qvel must be MPS with shape ({b}, {nv})")
    if not qvel.is_contiguous():
      raise ValueError("qvel must be contiguous")
    if pose_mismatch is None:
      pose_mismatch = self._zero_pose_mismatch
    elif (tuple(pose_mismatch.shape) != (b, max(self.ntree, 1))
          or pose_mismatch.device.type != "mps"
          or pose_mismatch.dtype != torch.int32
          or not pose_mismatch.is_contiguous()):
      raise ValueError(
          "pose_mismatch must be contiguous MPS int32 "
          "[batch, max(ntree, 1)]")
    if qfrc_applied is None:
      self._zero_qfrc.zero_()
      qfrc_applied = self._zero_qfrc[:, :nv] if nv else self._zero_qfrc
    if tuple(qfrc_applied.shape) != (b, nv) or qfrc_applied.device.type != "mps" or qfrc_applied.dtype != torch.float32:
      raise ValueError("qfrc_applied must be MPS with shape [batch, nv]")
    if not qfrc_applied.is_contiguous():
      raise ValueError("qfrc_applied must be contiguous")
    if xfrc_applied is None:
      self._zero_xfrc.zero_()
      xfrc_applied = self._zero_xfrc
    if tuple(xfrc_applied.shape) != (b, int(self.model.nbody), 6) or xfrc_applied.device.type != "mps" or xfrc_applied.dtype != torch.float32:
      raise ValueError("xfrc_applied must be MPS with shape [batch, nbody, 6]")
    if not xfrc_applied.is_contiguous():
      raise ValueError("xfrc_applied must be contiguous")
    if equality_active is None:
      equality_active = self.eq_active_default
    if (tuple(equality_active.shape) != (b, self.neq)
        or equality_active.device.type != "mps"
        or equality_active.dtype != torch.int32
        or not equality_active.is_contiguous()):
      raise ValueError("equality_active must be contiguous MPS int32 [batch, neq]")
    if equality_active is not self.eq_active:
      self.eq_active.copy_(equality_active)
    if active_tree_pairs is None:
      self.links.fill_(-1)
    else:
      nlinks = _tree_link_count(
          active_tree_pairs, b, self.max_tree_links, self.device, torch)
      if active_tree_pairs is not self.links:
        self.links.fill_(-1)
        if nlinks:
          self.links[:, :nlinks].copy_(active_tree_pairs)
    if constraints_active is None:
      self.constraints_active.zero_()
    else:
      if (tuple(constraints_active.shape) != (b,)
          or constraints_active.device.type != "mps"
          or constraints_active.dtype != torch.int32
          or not constraints_active.is_contiguous()):
        raise ValueError("constraints_active must be an MPS [batch] tensor")
      if constraints_active is not self.constraints_active:
        self.constraints_active.copy_(constraints_active)
    if link_overflow is None:
      self.links_overflow.zero_()
    else:
      if (tuple(link_overflow.shape) != (b,)
          or link_overflow.device.type != "mps"
          or link_overflow.dtype != torch.int32
          or not link_overflow.is_contiguous()):
        raise ValueError("link_overflow must be an MPS [batch] tensor")
      if link_overflow is not self.links_overflow:
        self.links_overflow.copy_(link_overflow)
    # Zero-DOF and zero-actuator models use a one-element storage sentinel.
    qv = qvel if nv else self._zero_qfrc
    qfrc = qfrc_applied if nv else self._zero_qfrc
    xfrc = xfrc_applied
    self._kernel(
        qv, qfrc, xfrc,
        self.dof_length, self.tree_dofadr, self.tree_dofnum, self.tree_bodyadr,
        self.tree_bodynum, self.tree_policy, self.links,
        self.constraints_active, self.sleep_tolerance, self.tree_state,
        self.tree_awake, self.status, self.links_overflow, self.int_dims,
        mode, self.eq_active, self.eq_kinds, self.eq_pair_trees,
        self.eq_flex_offsets, self.eq_flex_tree_ids, active_worlds,
        pose_mismatch,
        threads=(b,), group_size=(1,))
    self._build_awake_lists(active_worlds)
    return self.tree_awake

  def wake_before_solve(self, qvel, *, qfrc_applied=None, xfrc_applied=None,
                        active_tree_pairs=None, constraints_active=None,
                        link_overflow=None, equality_active=None,
                        active_worlds=None, pose_mismatch=None):
    """Wake cycles from this position stage without advancing sleep timers."""
    return self._run(
        qvel, qfrc_applied=qfrc_applied, xfrc_applied=xfrc_applied,
        active_tree_pairs=active_tree_pairs,
        constraints_active=constraints_active, link_overflow=link_overflow,
        equality_active=equality_active,
        active_worlds=active_worlds,
        pose_mismatch=pose_mismatch,
        mode=self._mode_wake)

  def advance_after_step(self, qvel, *, qfrc_applied=None, xfrc_applied=None):
    """Compatibility alias; qvel must be the pre-integration velocity.

    MuJoCo calls ``mj_sleep`` before qvel/qpos integration. New callers should
    use ``advance_before_integration`` to make that ordering explicit.
    """
    return self.advance_before_integration(
        qvel, qfrc_applied=qfrc_applied, xfrc_applied=xfrc_applied)

  def advance_before_integration(self, qvel, *, qfrc_applied=None,
                                 xfrc_applied=None, active_tree_pairs=None,
                                 constraints_active=None, link_overflow=None,
                                 equality_active=None, active_worlds=None):
    """Run pinned ``mj_sleep`` policy on pre-integration velocity.

    Call after activation advancement and the current forward/contact/equality
    wake pass, but before qvel/qpos integration. The rebuilt awake lists are
    then authoritative for the integration kernel.
    """
    return self._run(
        qvel, qfrc_applied=qfrc_applied, xfrc_applied=xfrc_applied,
        active_tree_pairs=(self.links if active_tree_pairs is None else active_tree_pairs),
        constraints_active=(self.constraints_active if constraints_active is None
                            else constraints_active),
        link_overflow=(self.links_overflow if link_overflow is None
                       else link_overflow),
        equality_active=(self.eq_active if equality_active is None
                         else equality_active),
        active_worlds=active_worlds,
        mode=self._mode_advance)

  def zero_asleep_acceleration(self, qacc):
    """Clear qacc entries for trees put to sleep in the pre-integration pass.

    MuJoCo's ``mj_sleepTrees`` zeros qvel and qacc when an awake tree enters
    its sleeping cycle. Integration separately retains asleep qpos and emits
    zero asleep qvel; this method applies the corresponding qacc ownership.
    """
    import torch
    nv, batch = int(self.model.nv), self.batch_size
    if (not isinstance(qacc, torch.Tensor) or tuple(qacc.shape) != (batch, nv)
        or qacc.dtype != torch.float32 or qacc.device.type != "mps"
        or not qacc.is_contiguous()):
      raise ValueError("qacc must be contiguous float32 MPS [batch,nv]")
    source = qacc if nv else self._zero_qfrc
    self._zero_asleep_acceleration_kernel(
        source, self.dof_treeid, self.tree_awake,
        self._zero_acceleration_dims,
        threads=(batch,), group_size=(1,))
    return qacc

  def step(self, qvel, *, qfrc_applied=None, xfrc_applied=None,
           active_tree_pairs=None, constraints_active=None, link_overflow=None,
           equality_active=None, active_worlds=None):
    """Combined reference transition, retained for direct policy tests."""
    return self._run(
        qvel, qfrc_applied=qfrc_applied, xfrc_applied=xfrc_applied,
        active_tree_pairs=active_tree_pairs, constraints_active=constraints_active,
        link_overflow=link_overflow, equality_active=equality_active,
        active_worlds=active_worlds,
        mode=self._mode_advance)

  def wake_all(self, env_ids=None):
    """Wake all trees in selected worlds after a host state/config change."""
    if env_ids is None:
      self.tree_state.fill_(_FULLY_AWAKE)
      self.tree_awake.fill_(1)
      self.status.zero_()
      self.links_overflow.zero_()
      self._build_awake_lists()
      return
    for world in env_ids:
      world = int(world)
      if world < 0 or world >= self.batch_size:
        raise IndexError(f"world {world} is outside batch size {self.batch_size}")
      self.tree_state[world].fill_(_FULLY_AWAKE)
      self.tree_awake[world].fill_(1)
      self.status[world] = 0
      self.links_overflow[world] = 0
    self._build_awake_lists()

  def check_status(self):
    """Raise for pinned-policy invariant failures at an explicit sync boundary."""
    status = self.status.detach().cpu().numpy()
    bad_contact = np.flatnonzero(status == 3)
    if bad_contact.size:
      raise RuntimeError(
          "MuJoCo sleep invariant failed: active contact between sleeping "
          f"trees in worlds {bad_contact.tolist()}")
    bad_cycle = np.flatnonzero(status == 1)
    if bad_cycle.size:
      raise RuntimeError(
          f"MuJoCo sleep state contains an invalid island cycle in worlds {bad_cycle.tolist()}")
    overflow = np.flatnonzero(status == 2)
    if overflow.size:
      raise RuntimeError(
          f"active tree-link capacity overflow in worlds {overflow.tolist()}")

  def snapshot(self):
    """Copy persistent device scheduler state at an explicit snapshot boundary."""
    return {
        "tree_state": self.tree_state.detach().cpu().numpy().copy(),
    }

  def restore(self, snapshot, env_ids=None):
    """Restore scheduler state after validated native state restoration."""
    import torch
    state = np.asarray(snapshot["tree_state"], dtype=np.int32)
    if state.shape != tuple(self.tree_state.shape):
      raise ValueError("sleep scheduler snapshot shape mismatch")
    state_t = torch.as_tensor(state, dtype=torch.int32, device=self.device)
    if env_ids is None:
      self.tree_state.copy_(state_t)
      self.tree_awake.copy_((state_t < 0).to(dtype=torch.int32))
      self.status.zero_()
      self.links_overflow.zero_()
    else:
      for world in env_ids:
        world = int(world)
        if world < 0 or world >= self.batch_size:
          raise IndexError(f"world {world} is outside batch size {self.batch_size}")
        self.tree_state[world].copy_(state_t[world])
        self.tree_awake[world].copy_((state_t[world] < 0).to(dtype=torch.int32))
        self.status[world] = 0
        self.links_overflow[world] = 0
    self._build_awake_lists()
