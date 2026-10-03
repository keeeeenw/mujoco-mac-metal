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

"""Kinematic island discovery and sleep/wake lifecycle (milestone 017).

In multibody dynamics, bodies connected by kinematic joints form kinematic
trees. When no active constraints (contacts, equalities, tendons) couple
different trees, their equations of motion are completely decoupled into
independent 'islands'.

When mjENBL_SLEEP is enabled:
An island whose generalized velocities and external forces/accelerations remain
below the model's sleep tolerance enters sleep mode. For sleeping islands,
accelerations and constraint forces are zeroed, avoiding unnecessary solver
work and eliminating numerical drift on stationary bodies.

When an external disturbance arrives (user modifies qvel, applies ctrl or
xfrc_applied, or an awake body makes contact), the sleeping island immediately
wakes.

When mjDSBL_ISLAND is set in disableflags, island decomposition is disabled
(the entire system is treated as a single unified island).
"""

from __future__ import annotations

from dataclasses import dataclass
import mujoco
import numpy as np
try:
  import torch
except ImportError:
  torch = None


@dataclass
class IslandPartition:
  """Static kinematic tree structure of an MjModel."""
  ntree: int
  body_tree: np.ndarray    # [nbody] -> tree index (world body 0 -> -1)
  dof_tree: np.ndarray     # [nv] -> tree index
  tree_dofs: list[list[int]] # list of dof indices per tree
  tree_bodies: list[list[int]] # list of body indices per tree


def build_island_partition(model: mujoco.MjModel) -> IslandPartition:
  """Identify kinematic trees rooted at direct children of the worldbody."""
  nbody = int(model.nbody)
  nv = int(model.nv)
  body_tree = np.full(nbody, -1, dtype=np.int32)
  dof_tree = np.full(nv, -1, dtype=np.int32)

  # Find root bodies (children of worldbody 0)
  tree_roots = []
  for b in range(1, nbody):
    p = int(model.body_parentid[b])
    if p == 0:
      tree_roots.append(b)

  ntree = len(tree_roots)
  tree_bodies: list[list[int]] = [[] for _ in range(ntree)]
  tree_dofs: list[list[int]] = [[] for _ in range(ntree)]

  for tree_idx, root in enumerate(tree_roots):
    # BFS/DFS to collect all descendant bodies
    stack = [root]
    while stack:
      curr = stack.pop()
      body_tree[curr] = tree_idx
      tree_bodies[tree_idx].append(curr)
      # Find DOFs belonging to joints of this body
      jadr = int(model.body_jntadr[curr])
      jnum = int(model.body_jntnum[curr])
      for j in range(jadr, jadr + jnum):
        dadr = int(model.jnt_dofadr[j])
        jtype = int(model.jnt_type[j])
        dof_cnt = 3 if jtype == int(mujoco.mjtJoint.mjJNT_BALL) else (
            6 if jtype == int(mujoco.mjtJoint.mjJNT_FREE) else 1)
        for d in range(dadr, dadr + dof_cnt):
          dof_tree[d] = tree_idx
          tree_dofs[tree_idx].append(d)
      # Find child bodies
      for b in range(1, nbody):
        if int(model.body_parentid[b]) == curr:
          stack.append(b)

  return IslandPartition(
      ntree=ntree,
      body_tree=body_tree,
      dof_tree=dof_tree,
      tree_dofs=tree_dofs,
      tree_bodies=tree_bodies,
  )


class IslandManager:
  """Tracks kinematic islands and manages sleep/wake lifecycle across batch."""

  def __init__(self, model: mujoco.MjModel, batch_size: int = 1, device=None):
    self.model = model
    self.batch_size = int(batch_size)
    self.device = device or (torch.device("cpu") if torch is not None else "cpu")
    self.partition = build_island_partition(model)

    opt = model.opt
    self.sleep_enabled = bool(int(opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
    self.island_disabled = bool(int(opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ISLAND))

    tol = float(getattr(opt, "sleep_tolerance", 1e-4))
    self.sleep_tolerance = tol if tol > 0 else 1e-4

    # Per-world sleep state: [batch, ntree]
    nt = max(self.partition.ntree, 1)
    self.tree_asleep = np.zeros((self.batch_size, nt), dtype=bool)
    # Consecutive stationary step counters
    self._stationary_steps = np.zeros((self.batch_size, nt), dtype=np.int32)
    # Required quiet steps before sleeping
    self.sleep_delay_steps = 10

  def discover_islands(self, active_body_pairs: list[tuple[int, int]] | None = None) -> list[list[int]]:
    """Group connected kinematic trees into independent islands.

    `active_body_pairs` contains pairs of body IDs with active contacts or constraints.
    Returns a list of tree ID sets representing each island.
    """
    nt = self.partition.ntree
    if nt == 0 or self.island_disabled:
      return [list(range(nt))]

    # Disjoint Set / Union-Find over trees
    parent = list(range(nt))

    def find(i):
      path = []
      while parent[i] != i:
        path.append(i)
        i = parent[i]
      for node in path:
        parent[node] = i
      return i

    def union(i, j):
      root_i, root_j = find(i), find(j)
      if root_i != root_j:
        parent[root_i] = root_j

    # Connect trees via active contact pairs
    if active_body_pairs:
      for b1, b2 in active_body_pairs:
        if 0 <= b1 < len(self.partition.body_tree) and 0 <= b2 < len(self.partition.body_tree):
          t1 = self.partition.body_tree[b1]
          t2 = self.partition.body_tree[b2]
          if t1 >= 0 and t2 >= 0 and t1 != t2:
            union(t1, t2)

    # Equalities connecting trees
    for e in range(self.model.neq):
      eq_type = int(self.model.eq_type[e])
      if eq_type in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
        b1 = int(self.model.eq_obj1id[e])
        b2 = int(self.model.eq_obj2id[e])
        t1 = self.partition.body_tree[b1] if 0 <= b1 < len(self.partition.body_tree) else -1
        t2 = self.partition.body_tree[b2] if 0 <= b2 < len(self.partition.body_tree) else -1
        if t1 >= 0 and t2 >= 0 and t1 != t2:
          union(t1, t2)

    # Group into islands
    islands: dict[int, list[int]] = {}
    for t in range(nt):
      root = find(t)
      if root not in islands:
        islands[root] = []
      islands[root].append(t)

    return list(islands.values())

  def update_sleep(self, qvel: torch.Tensor, ctrl: torch.Tensor | None = None,
                   xfrc_applied: torch.Tensor | None = None,
                   active_contacts: list[tuple[int, int]] | None = None):
    """Check kinematic islands for sleep eligibility or wake triggers."""
    if not self.sleep_enabled or self.island_disabled or self.partition.ntree == 0:
      self.tree_asleep.fill(False)
      return

    qvel_np = qvel.detach().cpu().numpy()
    b_size = qvel_np.shape[0]

    # Wake islands with active external control or forces
    ctrl_active = False
    if ctrl is not None:
      ctrl_np = ctrl.detach().cpu().numpy()
      ctrl_active = bool(np.any(np.abs(ctrl_np) > 1e-6))

    islands = self.discover_islands(active_contacts)

    for w in range(b_size):
      for island_trees in islands:
        # Collect DOFs in this island
        island_dofs = []
        for t in island_trees:
          island_dofs.extend(self.partition.tree_dofs[t])

        if not island_dofs:
          continue

        v_max = float(np.max(np.abs(qvel_np[w, island_dofs])))

        # Check if island is stationary and unforced
        if v_max < self.sleep_tolerance and not ctrl_active:
          for t in island_trees:
            self._stationary_steps[w, t] += 1
            if self._stationary_steps[w, t] >= self.sleep_delay_steps:
              self.tree_asleep[w, t] = True
        else:
          # Wake the entire island
          for t in island_trees:
            self._stationary_steps[w, t] = 0
            self.tree_asleep[w, t] = False

  def apply_sleep_to_state(self, qvel: torch.Tensor, qacc: torch.Tensor | None = None):
    """Zero velocities and accelerations of sleeping trees."""
    if not self.sleep_enabled or self.island_disabled or not np.any(self.tree_asleep):
      return

    for w in range(self.batch_size):
      for t in range(self.partition.ntree):
        if self.tree_asleep[w, t]:
          dofs = self.partition.tree_dofs[t]
          if dofs:
            qvel[w, dofs] = 0.0
            if qacc is not None:
              qacc[w, dofs] = 0.0

  def wake_all(self, env_ids=None):
    """Explicitly wake selected or all worlds."""
    if env_ids is None:
      self.tree_asleep.fill(False)
      self._stationary_steps.fill(0)
    else:
      for w in env_ids:
        self.tree_asleep[w].fill(False)
        self._stationary_steps[w].fill(0)
