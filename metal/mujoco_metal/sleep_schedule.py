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

_SHADER = Path(__file__).parent / "shaders" / "sleep_schedule.metal"
_LIBRARY = None
_FULLY_AWAKE = -11


def tree_sleep_transition_cpu(model, state, qvel, qfrc_applied, xfrc_applied,
                              active_tree_pairs=(), constraints_active=None,
                              active_equality_pairs=None, equality_active=None,
                              active_flex_equalities=(),
                              flex_equality_active=None):
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
  if ntree != partition.ntree:
    raise ValueError("state tree dimension does not match model")
  if not (qvel.shape == qfrc_applied.shape == (batch, int(model.nv))):
    raise ValueError("qvel/qfrc_applied must have shape [batch, nv]")
  if xfrc_applied.shape != (batch, int(model.nbody), 6):
    raise ValueError("xfrc_applied must have shape [batch, nbody, 6]")
  enabled = bool(int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
  disabled_islands = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ISLAND))
  if not enabled:
    state.fill(_FULLY_AWAKE)
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
        moving = weighted_speed >= velocity_tol if velocity_tol else weighted_speed != 0
        applied_qfrc = np.any(qfrc_applied[world, dof_idx] != 0)
      else:
        moving = False
        applied_qfrc = False
      applied_xfrc = np.any(xfrc_applied[world, bodies] != 0) if bodies else False
      can_sleep[tree] = not (moving or applied_qfrc or applied_xfrc)
      if not can_sleep[tree]:
        if state[world, tree] >= 0:
          wake_cycle(state[world], tree, _FULLY_AWAKE)
        else:
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
          cycle_a = set(cycle_members(state[world], a))
          cycle_b = set(cycle_members(state[world], b))
          if cycle_a.isdisjoint(cycle_b):
            wake_cycle(state[world], a, _FULLY_AWAKE)
            wake_cycle(state[world], b, _FULLY_AWAKE)
            changed = True
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
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("native sleep scheduling requires PyTorch MPS compile_shader")
    self.model = model
    self.batch_size = int(batch_size)
    if self.batch_size <= 0:
      raise ValueError("batch_size must be positive")
    self.device = torch.device(device)
    if self.device.type != "mps":
      raise ValueError("DeviceSleepScheduler requires an MPS device")
    self.partition = build_island_partition(model)
    self.ntree = self.partition.ntree
    self.enabled = bool(int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
    self.island_disabled = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ISLAND))
    if max_tree_links is None:
      max_tree_links = max(self.ntree * 8, 1)
    self.max_tree_links = int(max_tree_links)
    if self.max_tree_links <= 0:
      raise ValueError("max_tree_links must be positive")
    global _LIBRARY
    if _LIBRARY is None:
      _LIBRARY = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = _LIBRARY.sleep_transition
    batch, ntree = self.batch_size, max(self.ntree, 1)
    self.tree_state = torch.full((batch, ntree), _FULLY_AWAKE,
                                 dtype=torch.int32, device=self.device)
    init_policy = np.asarray(self.partition.tree_sleep_policy)
    init_trees = np.flatnonzero(init_policy == int(mujoco.mjtSleepPolicy.mjSLEEP_INIT))
    if init_trees.size and self.enabled:
      # mj_resetData marks init trees -1, then mj_sleep turns isolated trees
      # into one-node cycles (tree_asleep[i] == i).
      self.tree_state[:, init_trees] = torch.as_tensor(
          init_trees, dtype=torch.int32, device=self.device)
    self.tree_awake = torch.ones((batch, ntree), dtype=torch.int32, device=self.device)
    self.tree_awake.copy_((self.tree_state < 0).to(dtype=torch.int32))
    self.tree_dofadr = self._const(np.asarray(model.tree_dofadr, dtype=np.int32))
    self.tree_dofnum = self._const(np.asarray(model.tree_dofnum, dtype=np.int32))
    self.tree_bodyadr = self._const(np.asarray(model.tree_bodyadr, dtype=np.int32))
    self.tree_bodynum = self._const(np.asarray(model.tree_bodynum, dtype=np.int32))
    self.tree_policy = self._const(np.asarray(model.tree_sleep_policy, dtype=np.int32))
    self.dof_length = self._const(np.asarray(model.dof_length, dtype=np.float32))
    self._zero_qfrc = torch.zeros((batch, max(int(model.nv), 1)),
                                  dtype=torch.float32, device=self.device)
    self._zero_xfrc = torch.zeros((batch, int(model.nbody), 6),
                                  dtype=torch.float32, device=self.device)
    self.links = torch.full((batch, self.max_tree_links, 2), -1,
                            dtype=torch.int32, device=self.device)
    self.constraints_active = torch.zeros((batch,), dtype=torch.int32, device=self.device)
    self.links_overflow = torch.zeros((batch,), dtype=torch.int32, device=self.device)
    self.status = torch.zeros((batch,), dtype=torch.int32, device=self.device)
    # Integer dimensions share a separate tensor so no float conversion can
    # round counts on large worlds.
    self.int_dims = torch.tensor(
        (int(model.nv), int(model.nbody), int(model.nu), self.ntree,
         self.max_tree_links, self.batch_size, int(self.enabled),
         int(self.island_disabled), 1), dtype=torch.int32, device=self.device)
    self.sleep_tolerance = torch.tensor(
        (float(model.opt.sleep_tolerance),), dtype=torch.float32, device=self.device)
    self._mode_wake = torch.zeros((1,), dtype=torch.int32, device=self.device)
    self._mode_advance = torch.ones((1,), dtype=torch.int32, device=self.device)

  def _const(self, array):
    import torch
    return torch.as_tensor(array.copy(), dtype=torch.int32, device=self.device)

  def reset(self, env_ids=None):
    import mujoco
    import torch
    rows = (np.arange(self.batch_size, dtype=np.int64) if env_ids is None
            else np.asarray(env_ids, dtype=np.int64).reshape(-1))
    if np.any(rows < 0) or np.any(rows >= self.batch_size):
      raise IndexError("sleep reset environment index is outside batch")
    for world in rows.tolist():
      self.tree_state[world].fill_(_FULLY_AWAKE)
    if self.enabled:
      init = np.flatnonzero(
          self.partition.tree_sleep_policy == int(mujoco.mjtSleepPolicy.mjSLEEP_INIT))
      if init.size and rows.size:
        values = torch.as_tensor(init, dtype=torch.int32, device=self.device)
        for world in rows.tolist():
          self.tree_state[world, init.tolist()] = values
    for world in rows.tolist():
      self.tree_awake[world].copy_(
          (self.tree_state[world] < 0).to(dtype=torch.int32))
    self._zero_qfrc.zero_()
    self._zero_xfrc.zero_()
    self.links.fill_(-1)
    self.constraints_active.zero_()
    self.links_overflow.zero_()
    self.status.zero_()

  def _run(self, qvel, *, qfrc_applied=None, xfrc_applied=None,
           active_tree_pairs=None, constraints_active=None, link_overflow=None,
           mode):
    """Run one device wake or post-advance sleep pass."""
    import torch
    b, nv = self.batch_size, int(self.model.nv)
    if tuple(qvel.shape) != (b, nv) or qvel.device.type != "mps" or qvel.dtype != torch.float32:
      raise ValueError(f"qvel must be MPS with shape ({b}, {nv})")
    if not qvel.is_contiguous():
      raise ValueError("qvel must be contiguous")
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
    if active_tree_pairs is None:
      self.links.fill_(-1)
    else:
      if (tuple(active_tree_pairs.shape) != tuple(self.links.shape)
          or active_tree_pairs.device.type != "mps"
          or active_tree_pairs.dtype != torch.int32
          or not active_tree_pairs.is_contiguous()):
        raise ValueError(f"active_tree_pairs must have shape {tuple(self.links.shape)} on MPS")
      if active_tree_pairs is not self.links:
        self.links.copy_(active_tree_pairs)
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
        mode,
        threads=(b,), group_size=(1,))
    return self.tree_awake

  def wake_before_solve(self, qvel, *, qfrc_applied=None, xfrc_applied=None,
                        active_tree_pairs=None, constraints_active=None,
                        link_overflow=None):
    """Wake cycles from this position stage without advancing sleep timers."""
    return self._run(
        qvel, qfrc_applied=qfrc_applied, xfrc_applied=xfrc_applied,
        active_tree_pairs=active_tree_pairs,
        constraints_active=constraints_active, link_overflow=link_overflow,
        mode=self._mode_wake)

  def advance_after_step(self, qvel, *, qfrc_applied=None, xfrc_applied=None):
    """Advance quiet timers after accepted integration, matching mj_advance."""
    return self._run(
        qvel, qfrc_applied=qfrc_applied, xfrc_applied=xfrc_applied,
        active_tree_pairs=self.links, constraints_active=self.constraints_active,
        link_overflow=self.links_overflow, mode=self._mode_advance)

  def step(self, qvel, *, qfrc_applied=None, xfrc_applied=None,
           active_tree_pairs=None, constraints_active=None, link_overflow=None):
    """Combined reference transition, retained for direct policy tests."""
    return self._run(
        qvel, qfrc_applied=qfrc_applied, xfrc_applied=xfrc_applied,
        active_tree_pairs=active_tree_pairs, constraints_active=constraints_active,
        link_overflow=link_overflow, mode=self._mode_advance)

  def wake_all(self, env_ids=None):
    """Wake all trees in selected worlds after a host state/config change."""
    if env_ids is None:
      self.tree_state.fill_(_FULLY_AWAKE)
      self.tree_awake.fill_(1)
      return
    for world in env_ids:
      world = int(world)
      if world < 0 or world >= self.batch_size:
        raise IndexError(f"world {world} is outside batch size {self.batch_size}")
      self.tree_state[world].fill_(_FULLY_AWAKE)
      self.tree_awake[world].fill_(1)

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
    else:
      for world in env_ids:
        world = int(world)
        if world < 0 or world >= self.batch_size:
          raise IndexError(f"world {world} is outside batch size {self.batch_size}")
        self.tree_state[world].copy_(state_t[world])
        self.tree_awake[world].copy_((state_t[world] < 0).to(dtype=torch.int32))
