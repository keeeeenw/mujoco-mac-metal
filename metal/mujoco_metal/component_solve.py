# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Exact per-component mass factorization and multi-RHS application."""

from pathlib import Path

import numpy as np

from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity

_SHADER = Path(__file__).parent / "shaders" / "component_solve.metal"


def _component_layout_array(layout, nv):
  """Validate and pack the static layout consumed by the factor shader."""
  i32max = np.iinfo(np.int32).max
  for name in ("ncomponent", "nnz"):
    value = layout[name]
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))):
      raise ValueError(f"{name} must be an integer")
    if int(value) > i32max:
      raise ValueError(f"{name} exceeds the Metal int32 dimension limit")
  ncomponent, nnz = int(layout["ncomponent"]), int(layout["nnz"])
  raw = {name: np.asarray(layout[key]) for name, key in (
      ("component DOF offsets", "component_dof_offsets"),
      ("component DOF ids", "component_dof_ids"),
      ("component mass offsets", "component_mass_offsets"))}
  for name, value in raw.items():
    if value.dtype.kind not in "iu":
      raise ValueError(f"{name} must use integer storage")
    if value.size and (np.any(value < 0) or np.any(value > i32max)):
      raise ValueError(f"{name} exceeds the Metal int32 index range")
  offsets = raw["component DOF offsets"].astype(np.int64, copy=False)
  dof_ids = raw["component DOF ids"].astype(np.int64, copy=False)
  mass_offsets = raw["component mass offsets"].astype(np.int64, copy=False)
  if ncomponent < 0 or nnz < 0 or nv < 0:
    raise ValueError("component layout dimensions must be nonnegative")
  if (offsets.shape != (ncomponent + 1,) or offsets[0] != 0
      or offsets[-1] != nv or np.any(np.diff(offsets) < 0)):
    raise ValueError("component DOF offsets must partition all nv DOFs")
  if (dof_ids.shape != (nv,) or np.any(dof_ids < 0)
      or np.any(dof_ids >= nv)
      or not np.array_equal(np.sort(dof_ids), np.arange(nv))):
    raise ValueError("component DOF ids must be a permutation of [0,nv)")
  if mass_offsets.shape != (ncomponent,) or np.any(mass_offsets < 0):
    raise ValueError("component mass offsets must have shape [ncomponent]")
  expected_mass_offset = 0
  for component in range(ncomponent):
    width = int(offsets[component + 1] - offsets[component])
    start = int(mass_offsets[component])
    if start != expected_mass_offset:
      raise ValueError("component mass blocks must densely partition nnz")
    if start + width * width > nnz:
      raise ValueError("component mass block exceeds its compiled backing")
    expected_mass_offset += width * width
  if expected_mass_offset != nnz:
    raise ValueError("component mass blocks must densely partition nnz")
  ids = (dof_ids.astype(np.int32) if nv else np.zeros(1, dtype=np.int32))
  mass = (mass_offsets.astype(np.int32) if ncomponent
          else np.zeros(1, dtype=np.int32))
  packed = np.concatenate((offsets.astype(np.int32), ids, mass))
  return ncomponent, nnz, packed


def component_solver_workspace_sizes(*, batch, nv, ncomponent, nnz,
                                     rhs_capacity, layout_size):
  """Actual and input backing sizes used by ``MetalComponentMassSolver``."""
  values = {}
  for name, value in (("batch", batch), ("nv", nv),
                      ("ncomponent", ncomponent), ("nnz", nnz),
                      ("rhs_capacity", rhs_capacity),
                      ("layout_size", layout_size)):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))):
      raise ValueError(f"{name} must be an integer")
    values[name] = int(value)
  batch, nv = values["batch"], values["nv"]
  ncomponent, nnz = values["ncomponent"], values["nnz"]
  rhs_capacity, layout_size = values["rhs_capacity"], values["layout_size"]
  if (batch <= 0 or min(nv, ncomponent, nnz, rhs_capacity, layout_size) < 0):
    raise ValueError("component workspace dimensions must be nonnegative with positive batch")
  maxnv = max(nv, 1)
  return {
      "component_mass_dof_mask": batch * maxnv,
      "component_mass_factor_dof_mask": batch * maxnv,
      "component_mass_factor": max(batch * nnz, 1),
      "component_mass_output": batch * rhs_capacity * maxnv,
      "component_mass_single_output": batch * maxnv,
      "component_mass_matvec_output": batch * maxnv,
      "component_mass_status": batch * max(ncomponent, 1),
      "component_mass_factor_status": batch * max(ncomponent, 1),
      "component_mass_zero_blocks": max(batch * nnz, 1),
      "component_mass_layout": max(layout_size, 1),
      "component_mass_default_dof_ids": batch * maxnv,
      "component_mass_default_counts": batch * 3,
      "component_mass_diagonal_add": batch * maxnv,
      "component_mass_damping_mask": maxnv,
      "component_mass_euler_dt": 1,
      "component_mass_allow_indefinite": 1,
      "component_mass_phase": 1,
      "component_mass_dims": 5,
      "component_mass_single_dims": 5,
      "component_mass_merge_dims": 2,
  }


def build_euler_diagonal_cpu(q_deriv, timestep, euler_eligible, awake_dof):
  """CPU oracle for the pinned awake Euler ``dt*qDeriv`` diagonal.

  Eligibility is a per-world branch trigger. Once any awake DOF triggers the
  implicit damping solve, MuJoCo adds the derivative for every awake DOF.
  """
  q_deriv = np.asarray(q_deriv, dtype=np.float64)
  eligible = np.asarray(euler_eligible, dtype=bool)
  awake = np.asarray(awake_dof, dtype=bool)
  if q_deriv.ndim != 2 or eligible.shape != (q_deriv.shape[1],):
    raise ValueError("q_deriv and euler_eligible shapes do not match")
  if awake.shape != q_deriv.shape:
    raise ValueError("awake_dof must have shape [batch,nv]")
  if (isinstance(timestep, (bool, np.bool_))
      or not isinstance(timestep, (int, float, np.integer, np.floating))
      or not np.isfinite(timestep) or float(timestep) <= 0):
    raise ValueError("timestep must be finite and positive")
  trigger = np.any(awake & eligible[None, :], axis=1)
  return np.where(awake & trigger[:, None], float(timestep) * q_deriv, 0.0)


def solve_component_blocks_cpu(mass_blocks, layout, rhs, *, active_dof=None,
                               tendon_armature_blocks=None,
                               diagonal_add=None, allow_indefinite=False):
  """CPU oracle for the exact packed-component ``M^-1 rhs`` operation."""
  mass_blocks = np.asarray(mass_blocks, dtype=np.float64)
  rhs = np.asarray(rhs, dtype=np.float64)
  offsets = np.asarray(layout["component_dof_offsets"], dtype=np.int64)
  dofs = np.asarray(layout["component_dof_ids"], dtype=np.int64)
  mass_offsets = np.asarray(layout["component_mass_offsets"], dtype=np.int64)
  nv, ncomponent = int(layout["nv"]), int(layout["ncomponent"])
  if rhs.ndim != 3 or rhs.shape[0] != mass_blocks.shape[0] or rhs.shape[2] != nv:
    raise ValueError("rhs must have shape [batch,nrhs,nv]")
  batch, nrhs = rhs.shape[:2]
  nnz = int(layout["nnz"])
  if mass_blocks.shape != (batch, nnz):
    raise ValueError("mass_blocks shape does not match the compiled layout")
  if tendon_armature_blocks is None:
    tendon_armature_blocks = np.zeros_like(mass_blocks)
  else:
    tendon_armature_blocks = np.asarray(tendon_armature_blocks, dtype=np.float64)
    if tendon_armature_blocks.shape != mass_blocks.shape:
      raise ValueError("tendon armature block shape does not match mass_blocks")
  if active_dof is None:
    active_dof = np.ones((batch, nv), dtype=bool)
  else:
    active_dof = np.asarray(active_dof, dtype=bool)
    if active_dof.shape != (batch, nv):
      raise ValueError("active_dof must have shape [batch,nv]")
  if diagonal_add is None:
    diagonal_add = np.zeros((batch, nv), dtype=np.float64)
  else:
    diagonal_add = np.asarray(diagonal_add, dtype=np.float64)
    if diagonal_add.shape != (batch, nv):
      raise ValueError("diagonal_add must have shape [batch,nv]")
  output = np.zeros_like(rhs)
  status = np.zeros((batch, ncomponent), dtype=np.int32)
  float32_limit = np.finfo(np.float32).max
  symmetry_factor = 32.0 * np.finfo(np.float32).eps
  for world in range(batch):
    for component in range(ncomponent):
      begin, end = int(offsets[component]), int(offsets[component + 1])
      component_dofs = dofs[begin:end]
      awake = active_dof[world, component_dofs]
      if not np.any(awake):
        continue
      width = end - begin
      block_offset = int(mass_offsets[component])
      mass_block = mass_blocks[
          world, block_offset:block_offset + width * width].reshape(width, width)
      armature_block = tendon_armature_blocks[
          world, block_offset:block_offset + width * width].reshape(width, width)
      active_mass = mass_block[np.ix_(awake, awake)]
      active_armature = armature_block[np.ix_(awake, awake)]
      active_diagonal = diagonal_add[world, component_dofs[awake]]
      active_rhs = rhs[world][:, component_dofs[awake]]
      if not (np.all(np.isfinite(active_mass))
              and np.all(np.isfinite(active_armature))
              and np.all(np.isfinite(active_diagonal))
              and np.all(np.isfinite(active_rhs))):
        status[world, component] = 1
        continue
      with np.errstate(over="ignore", invalid="ignore"):
        active_block = active_mass + active_armature
        active_block[np.diag_indices_from(active_block)] += active_diagonal
      if (not np.all(np.isfinite(active_block))
          or np.any(np.abs(active_block) > float32_limit)):
        status[world, component] = 1
        continue
      scale = float(np.max(np.abs(active_block), initial=0.0))
      tolerance = scale * symmetry_factor
      if np.any(np.abs(active_block - active_block.T) > tolerance):
        status[world, component] = 1
        continue
      active_block = (0.5 * active_block + 0.5 * active_block.T).astype(
          np.float64)
      try:
        if allow_indefinite:
          # Match pinned mj_factorI: descending L' D L factorization, then
          # inv(L'), D^-1, L^-1. np.linalg.solve would accept the same matrix
          # but would not exercise the production pivot ordering.
          width_active = active_block.shape[0]
          with np.errstate(over="ignore", invalid="ignore"):
            factor = active_block.astype(np.float32)
            rhs32 = active_rhs.astype(np.float32)
          if not (np.all(np.isfinite(factor)) and np.all(np.isfinite(rhs32))):
            status[world, component] = 1
            continue
          failed = False
          with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for k in range(width_active - 1, -1, -1):
              pivot = np.float32(factor[k, k])
              if not np.isfinite(pivot) or pivot == 0:
                failed = True
                break
              for i in range(k - 1, -1, -1):
                multiplier = np.float32(factor[k, i] / pivot)
                if not np.isfinite(multiplier):
                  failed = True
                  break
                for j in range(i + 1):
                  product = np.float32(multiplier * factor[k, j])
                  value = np.float32(factor[i, j] - product)
                  if not np.isfinite(product) or not np.isfinite(value):
                    failed = True
                    break
                  factor[i, j] = value
                if failed:
                  break
              if failed:
                break
              for j in range(k):
                value = np.float32(factor[k, j] / pivot)
                if not np.isfinite(value):
                  failed = True
                  break
                factor[k, j] = value
              if failed:
                break
          if failed:
            status[world, component] = 1
            continue
          solved = np.zeros_like(active_rhs, dtype=np.float32)
          with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for row in range(active_rhs.shape[0]):
              values = rhs32[row].copy()
              # inv(L') descending scatter.
              for i in range(width_active - 1, -1, -1):
                value = values[i]
                for j in range(i):
                  product = np.float32(factor[i, j] * value)
                  updated = np.float32(values[j] - product)
                  if not np.isfinite(product) or not np.isfinite(updated):
                    failed = True
                    break
                  values[j] = updated
                if failed:
                  break
              if failed:
                break
              # D^-1.
              for i in range(width_active):
                value = np.float32(values[i] / factor[i, i])
                if not np.isfinite(value):
                  failed = True
                  break
                values[i] = value
              if failed:
                break
              # L^-1 ascending.
              for i in range(width_active):
                value = values[i]
                for j in range(i):
                  product = np.float32(factor[i, j] * values[j])
                  value = np.float32(value - product)
                  if not np.isfinite(product) or not np.isfinite(value):
                    failed = True
                    break
                if failed:
                  break
                values[i] = value
              if failed or not np.all(np.isfinite(values)):
                failed = True
                break
              solved[row] = values
          if failed:
            status[world, component] = 1
            continue
          solved = solved.astype(np.float64)
        else:
          np.linalg.cholesky(active_block)
          solved = np.linalg.solve(active_block, active_rhs.T).T
      except np.linalg.LinAlgError:
        status[world, component] = 1
        continue
      if (not np.all(np.isfinite(solved))
          or np.any(np.abs(solved) > float32_limit)):
        status[world, component] = 1
        continue
      output[world][:, component_dofs[awake]] = solved
  return output, status


def component_mass_matvec_cpu(mass_blocks, layout, vector, *, active_dof=None,
                              tendon_armature_blocks=None,
                              diagonal_add=None):
  """Apply compiled component mass blocks to a vector on the CPU oracle."""
  mass_blocks = np.asarray(mass_blocks, dtype=np.float64)
  vector = np.asarray(vector, dtype=np.float64)
  offsets = np.asarray(layout["component_dof_offsets"], dtype=np.int64)
  dofs = np.asarray(layout["component_dof_ids"], dtype=np.int64)
  mass_offsets = np.asarray(layout["component_mass_offsets"], dtype=np.int64)
  nv, ncomponent, nnz = (int(layout["nv"]), int(layout["ncomponent"]),
                         int(layout["nnz"]))
  if (mass_blocks.ndim != 2 or mass_blocks.shape[1] != nnz
      or vector.ndim != 2 or vector.shape != (mass_blocks.shape[0], nv)):
    raise ValueError("mass blocks/vector shapes do not match the compiled layout")
  batch = mass_blocks.shape[0]
  if tendon_armature_blocks is None:
    tendon_armature_blocks = np.zeros_like(mass_blocks)
  else:
    tendon_armature_blocks = np.asarray(tendon_armature_blocks, dtype=np.float64)
    if tendon_armature_blocks.shape != mass_blocks.shape:
      raise ValueError("tendon armature block shape does not match mass_blocks")
  if active_dof is None:
    active_dof = np.ones((batch, nv), dtype=bool)
  else:
    active_dof = np.asarray(active_dof, dtype=bool)
    if active_dof.shape != (batch, nv):
      raise ValueError("active_dof must have shape [batch,nv]")
  if diagonal_add is None:
    diagonal_add = np.zeros((batch, nv), dtype=np.float64)
  else:
    diagonal_add = np.asarray(diagonal_add, dtype=np.float64)
    if diagonal_add.shape != (batch, nv):
      raise ValueError("diagonal_add must have shape [batch,nv]")
  output = np.zeros_like(vector)
  for world in range(batch):
    for component in range(ncomponent):
      begin, end = int(offsets[component]), int(offsets[component + 1])
      ids = dofs[begin:end]
      active = active_dof[world, ids]
      width = end - begin
      base = int(mass_offsets[component])
      block = (mass_blocks[world, base:base + width * width]
               + tendon_armature_blocks[world, base:base + width * width]
               ).reshape(width, width)
      if np.any(active):
        active_ids = ids[active]
        active_block = block[np.ix_(active, active)]
        output[world, active_ids] = (
            active_block @ vector[world, active_ids]
            + diagonal_add[world, active_ids] * vector[world, active_ids])
  return output


class MetalComponentMassSolver:
  """Apply exact component-wise ``M^-1`` to a fixed-capacity RHS batch.

  The static block layout comes from :func:`compile_tree_mass_layout` and
  includes every sparse mass coupling the pinned compiler can emit. ``rhs``
  may contain a gathered set of canonical constraint rows; its second axis is
  caller-owned and is never reordered here.
  """

  def __init__(self, model, layout, *, batch_size, rhs_capacity,
               euler_damping_dofs=None):
    if (isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
      raise ValueError("batch_size must be a positive integer")
    if (isinstance(rhs_capacity, (bool, np.bool_))
        or not isinstance(rhs_capacity, (int, np.integer))
        or rhs_capacity <= 0):
      raise ValueError("rhs_capacity must be a positive integer")
    batch_size, rhs_capacity = int(batch_size), int(rhs_capacity)
    self.model = model
    self.nv = int(model.nv)
    self.ncomponent, self.nnz, packed = _component_layout_array(
        layout, self.nv)
    self.batch_size = batch_size
    self.rhs_capacity = rhs_capacity
    self._workspace_sizes = component_solver_workspace_sizes(
        batch=batch_size, nv=self.nv, ncomponent=self.ncomponent,
        nnz=self.nnz, rhs_capacity=rhs_capacity, layout_size=packed.size)
    _validate_workspace_index_capacity(
        batch_size, self._workspace_sizes,
        {"nv": self.nv, "nbody": int(model.nbody),
         "ncomponent": self.ncomponent,
         "nnz": self.nnz, "rhs_capacity": rhs_capacity})
    for name, elements in self._workspace_sizes.items():
      if elements > np.iinfo(np.int32).max:
        raise ValueError(
            f"{name} exceeds the Metal int32 address range ({elements})")
    # Validate and lower this optional public metadata before importing Torch,
    # compiling a shader, or allocating any device workspace. Malformed host
    # metadata must remain a pure preflight failure.
    if euler_damping_dofs is not None:
      damping_eligible = np.asarray(euler_damping_dofs)
      if (damping_eligible.dtype.kind != "b"
          or damping_eligible.shape != (self.nv,)):
        raise ValueError("euler_damping_dofs must be boolean [nv]")
      has_euler_damping_mask = True
    elif all(hasattr(model, name) for name in (
        "dof_damping", "dof_dampingpoly", "dof_jntid", "jnt_actuatorid",
        "opt")):
      from mujoco_metal.stepping import _euler_damping_dofs
      damping_eligible = _euler_damping_dofs(model)
      has_euler_damping_mask = True
    else:
      damping_eligible = np.zeros(self.nv, dtype=bool)
      has_euler_damping_mask = False

    import torch

    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._merge_dims = torch.tensor(
        [self.ncomponent, batch_size], dtype=torch.int32, device=self._device)
    self._component_layout = torch.as_tensor(
        packed,
        dtype=torch.int32, device=self._device)
    self._dims = torch.tensor(
        [self.nv, self.ncomponent, batch_size, self.nnz,
         rhs_capacity], dtype=torch.int32, device=self._device)
    self._single_dims = torch.tensor(
        [self.nv, self.ncomponent, batch_size, self.nnz, 1],
        dtype=torch.int32, device=self._device)
    maxnv = max(self.nv, 1)
    self._workspace = {
        "dof_mask": torch.zeros((batch_size, maxnv), dtype=torch.int32,
                                device=self._device),
        "factor": torch.empty(max(batch_size * self.nnz, 1),
                              dtype=torch.float32, device=self._device),
        "output": torch.zeros((batch_size, rhs_capacity, maxnv),
                              dtype=torch.float32, device=self._device),
        "single_output": torch.zeros((batch_size, maxnv),
                                     dtype=torch.float32, device=self._device),
        "matvec_output": torch.empty((batch_size, maxnv),
                                      dtype=torch.float32, device=self._device),
        "status": torch.zeros((batch_size, max(self.ncomponent, 1)),
                              dtype=torch.int32, device=self._device),
        "factor_status": torch.zeros((batch_size, max(self.ncomponent, 1)),
                                      dtype=torch.int32, device=self._device),
        "zero_blocks": torch.zeros(max(batch_size * self.nnz, 1),
                                    dtype=torch.float32, device=self._device),
    }
    self._all_dof_ids = torch.arange(maxnv, dtype=torch.int32,
                                     device=self._device).expand(batch_size, -1).contiguous()
    self._all_counts = torch.tensor(
        np.tile(np.asarray([[model.nbody - 1, model.nbody - 1, self.nv]],
                           dtype=np.int32), (batch_size, 1)),
        dtype=torch.int32, device=self._device)
    self._has_euler_damping_mask = has_euler_damping_mask
    damping_mask = np.zeros(maxnv, dtype=np.int32)
    damping_mask[:self.nv] = damping_eligible.astype(np.int32)
    self._euler_damping_mask = torch.as_tensor(
        damping_mask, dtype=torch.int32, device=self._device)
    self._euler_dt = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._allow_indefinite = torch.zeros(1, dtype=torch.int32,
                                         device=self._device)
    self._phase = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._factorized = False
    self._workspace["diagonal_add"] = torch.zeros(
        (batch_size, maxnv), dtype=torch.float32, device=self._device)
    self._workspace["factor_dof_mask"] = torch.zeros(
        (batch_size, maxnv), dtype=torch.int32, device=self._device)

  def run_device(self, mass_blocks, rhs, *, dof_ids=None, counts=None,
                 tendon_armature_blocks=None, diagonal_add=None,
                 allow_indefinite=False):
    """Return borrowed ``(solution, per-component status)`` MPS tensors."""
    torch = self._torch
    if (not isinstance(mass_blocks, torch.Tensor)
        or tuple(mass_blocks.shape) != (self.batch_size, self.nnz)
        or mass_blocks.dtype != torch.float32 or mass_blocks.device.type != "mps"
        or not mass_blocks.is_contiguous()):
      raise ValueError("mass_blocks must be contiguous float32 MPS [batch,nnz]")
    if (not isinstance(rhs, torch.Tensor) or rhs.ndim != 3
        or tuple(rhs.shape) != (self.batch_size, self.rhs_capacity, self.nv)
        or rhs.dtype != torch.float32 or rhs.device.type != "mps"
        or not rhs.is_contiguous()):
      raise ValueError("rhs must be contiguous float32 MPS [batch,rhs_capacity,nv]")
    if dof_ids is None:
      dof_ids, counts = self._all_dof_ids, self._all_counts
    if (not isinstance(dof_ids, torch.Tensor)
        or tuple(dof_ids.shape) != (self.batch_size, max(self.nv, 1))
        or dof_ids.dtype != torch.int32 or dof_ids.device.type != "mps"
        or not dof_ids.is_contiguous()):
      raise ValueError("dof_ids must be contiguous MPS int32 [batch,max(nv,1)]")
    if (not isinstance(counts, torch.Tensor)
        or tuple(counts.shape) != (self.batch_size, 3)
        or counts.dtype != torch.int32 or counts.device.type != "mps"
        or not counts.is_contiguous()):
      raise ValueError("counts must be contiguous MPS int32 [batch,3]")
    if tendon_armature_blocks is None:
      tendon_armature_blocks = self._workspace["zero_blocks"][:
          self.batch_size * self.nnz].reshape(self.batch_size, self.nnz)
    elif (not isinstance(tendon_armature_blocks, torch.Tensor)
          or tuple(tendon_armature_blocks.shape) != (self.batch_size, self.nnz)
          or tendon_armature_blocks.dtype != torch.float32
          or tendon_armature_blocks.device.type != "mps"
          or not tendon_armature_blocks.is_contiguous()):
      raise ValueError("tendon_armature_blocks must be contiguous float32 MPS [batch,nnz]")
    diagonal_add = self._validate_diagonal_add(diagonal_add)
    if not isinstance(allow_indefinite, (bool, np.bool_)):
      raise ValueError("allow_indefinite must be a boolean")
    self._factorized = False
    self._allow_indefinite.fill_(int(allow_indefinite))
    self._phase.fill_(2)
    output, status, workspace = self._workspace["output"], self._workspace["status"], self._workspace
    if self.nv:
      self._library.build_dof_awake_mask(
          dof_ids.reshape(-1), counts.reshape(-1), workspace["dof_mask"].reshape(-1),
          self._dims, threads=(self.batch_size,), group_size=(1,))
    output.zero_()
    if self.ncomponent:
      self._library.factor_apply_components(
          mass_blocks.reshape(-1), tendon_armature_blocks.reshape(-1),
          rhs.reshape(-1), workspace["dof_mask"].reshape(-1),
          self._component_layout, workspace["factor"], output.reshape(-1),
          status.reshape(-1), self._dims,
          diagonal_add.reshape(-1),
          self._allow_indefinite,
          self._phase,
          status.reshape(-1),
          threads=(self.batch_size * self.ncomponent,), group_size=(1,))
    return (output[:, :, :self.nv],
            status[:, :self.ncomponent])

  @property
  def factor_active_dof_mask(self):
    """Borrowed `[batch,max(nv,1)]` int mask used by retained factors."""
    return self._workspace["factor_dof_mask"]

  def factorize_device(self, mass_blocks, *, dof_ids=None, counts=None,
                       tendon_armature_blocks=None, diagonal_add=None,
                       allow_indefinite=False):
    """Factor the active component mass blocks once for repeated RHS solves.

    The returned status is borrowed device memory. The retained factors are
    valid until this solver performs another factorization or fused
    ``run_device`` call. The caller must keep the mass, armature, diagonal,
    and active-DOF set unchanged while using ``solve_factored_device``.
    """
    torch = self._torch
    if (not isinstance(mass_blocks, torch.Tensor)
        or tuple(mass_blocks.shape) != (self.batch_size, self.nnz)
        or mass_blocks.dtype != torch.float32 or mass_blocks.device.type != "mps"
        or not mass_blocks.is_contiguous()):
      raise ValueError("mass_blocks must be contiguous float32 MPS [batch,nnz]")
    if dof_ids is None:
      dof_ids, counts = self._all_dof_ids, self._all_counts
    if (not isinstance(dof_ids, torch.Tensor)
        or tuple(dof_ids.shape) != (self.batch_size, max(self.nv, 1))
        or dof_ids.dtype != torch.int32 or dof_ids.device.type != "mps"
        or not dof_ids.is_contiguous()):
      raise ValueError("dof_ids must be contiguous MPS int32 [batch,max(nv,1)]")
    if (not isinstance(counts, torch.Tensor)
        or tuple(counts.shape) != (self.batch_size, 3)
        or counts.dtype != torch.int32 or counts.device.type != "mps"
        or not counts.is_contiguous()):
      raise ValueError("counts must be contiguous MPS int32 [batch,3]")
    if tendon_armature_blocks is None:
      tendon_armature_blocks = self._workspace["zero_blocks"][
          :self.batch_size * self.nnz].reshape(self.batch_size, self.nnz)
    elif (not isinstance(tendon_armature_blocks, torch.Tensor)
          or tuple(tendon_armature_blocks.shape) != (self.batch_size, self.nnz)
          or tendon_armature_blocks.dtype != torch.float32
          or tendon_armature_blocks.device.type != "mps"
          or not tendon_armature_blocks.is_contiguous()):
      raise ValueError(
          "tendon_armature_blocks must be contiguous float32 MPS [batch,nnz]")
    diagonal_add = self._validate_diagonal_add(diagonal_add)
    if not isinstance(allow_indefinite, (bool, np.bool_)):
      raise ValueError("allow_indefinite must be a boolean")
    self._factorized = False
    self._allow_indefinite.fill_(int(allow_indefinite))
    self._phase.fill_(0)
    workspace = self._workspace
    if self.nv:
      self._library.build_dof_awake_mask(
          dof_ids.reshape(-1), counts.reshape(-1),
          workspace["dof_mask"].reshape(-1), self._dims,
          threads=(self.batch_size,), group_size=(1,))
    workspace["factor_dof_mask"].copy_(workspace["dof_mask"])
    if self.ncomponent:
      self._library.factor_apply_components(
          mass_blocks.reshape(-1), tendon_armature_blocks.reshape(-1),
          workspace["output"].reshape(-1),
          workspace["factor_dof_mask"].reshape(-1), self._component_layout,
          workspace["factor"], workspace["output"].reshape(-1),
          workspace["factor_status"].reshape(-1), self._dims,
          diagonal_add.reshape(-1), self._allow_indefinite, self._phase,
          workspace["factor_status"].reshape(-1),
          threads=(self.batch_size * self.ncomponent,), group_size=(1,))
    self._factorized = True
    return workspace["factor_status"][:, :self.ncomponent]

  def solve_factored_device(self, rhs):
    """Apply retained component factors to a new fixed-capacity RHS batch."""
    torch = self._torch
    if not self._factorized:
      raise RuntimeError("component factors are unavailable; call factorize_device first")
    if (not isinstance(rhs, torch.Tensor) or rhs.ndim != 3
        or tuple(rhs.shape) != (self.batch_size, self.rhs_capacity, self.nv)
        or rhs.dtype != torch.float32 or rhs.device.type != "mps"
        or not rhs.is_contiguous()):
      raise ValueError(
          "rhs must be contiguous float32 MPS [batch,rhs_capacity,nv]")
    self._phase.fill_(1)
    output, status, workspace = (
        self._workspace["output"], self._workspace["status"], self._workspace)
    output.zero_()
    if self.ncomponent:
      self._library.factor_apply_components(
          self._workspace["zero_blocks"].reshape(-1),
          self._workspace["zero_blocks"].reshape(-1), rhs.reshape(-1),
          workspace["factor_dof_mask"].reshape(-1), self._component_layout,
          workspace["factor"], output.reshape(-1), status.reshape(-1),
          self._dims, self._workspace["diagonal_add"].reshape(-1),
          self._allow_indefinite, self._phase,
          workspace["factor_status"].reshape(-1),
          threads=(self.batch_size * self.ncomponent,), group_size=(1,))
    return output[:, :, :self.nv], status[:, :self.ncomponent]

  def solve_factored_vector_device(self, rhs):
    """Apply retained factors to one canonical ``[batch,nv]`` RHS.

    This fixed single-vector path lets restarted or full GMRES reuse the same
    factorization without requiring its Krylov dimension to fit the caller's
    row-batch RHS capacity.
    """
    torch = self._torch
    if not self._factorized:
      raise RuntimeError("component factors are unavailable; call factorize_device first")
    if (not isinstance(rhs, torch.Tensor)
        or tuple(rhs.shape) != (self.batch_size, self.nv)
        or rhs.dtype != torch.float32 or rhs.device.type != "mps"
        or not rhs.is_contiguous()):
      raise ValueError("rhs must be contiguous float32 MPS [batch,nv]")
    self._phase.fill_(1)
    output = self._workspace["single_output"]
    output.zero_()
    status = self._workspace["status"]
    if self.ncomponent:
      self._library.factor_apply_components(
          self._workspace["zero_blocks"].reshape(-1),
          self._workspace["zero_blocks"].reshape(-1), rhs.reshape(-1),
          self._workspace["factor_dof_mask"].reshape(-1),
          self._component_layout, self._workspace["factor"],
          output.reshape(-1), status.reshape(-1), self._single_dims,
          self._workspace["diagonal_add"].reshape(-1),
          self._allow_indefinite, self._phase,
          self._workspace["factor_status"].reshape(-1),
          threads=(self.batch_size * self.ncomponent,), group_size=(1,))
    return output[:, :self.nv], status[:, :self.ncomponent]

  def merge_world_status(self, component_status, world_status):
    """Max-reduce component failures into the caller's per-world status.

    This is a device-only merge so a staged component solve can contribute to
    an existing transaction status without a host readback.
    """
    torch = self._torch
    if (not isinstance(component_status, torch.Tensor)
        or tuple(component_status.shape) != (self.batch_size, self.ncomponent)
        or component_status.dtype != torch.int32
        or component_status.device.type != "mps"
        or not component_status.is_contiguous()):
      raise ValueError(
          "component_status must be contiguous MPS int32 [batch,ncomponent]")
    if (not isinstance(world_status, torch.Tensor)
        or tuple(world_status.shape) != (self.batch_size,)
        or world_status.dtype != torch.int32
        or world_status.device.type != "mps"
        or not world_status.is_contiguous()):
      raise ValueError("world_status must be contiguous MPS int32 [batch]")
    if self.ncomponent:
      self._library.merge_component_world_status(
          component_status.reshape(-1), world_status, self._merge_dims,
          threads=(self.batch_size,), group_size=(1,))
    return world_status

  def _validate_diagonal_add(self, diagonal_add):
    torch = self._torch
    if diagonal_add is None:
      diagonal_add = self._workspace["diagonal_add"]
      diagonal_add.zero_()
    elif (not isinstance(diagonal_add, torch.Tensor)
          or tuple(diagonal_add.shape) != (self.batch_size, self.nv)
          or diagonal_add.dtype != torch.float32
          or diagonal_add.device.type != "mps"
          or not diagonal_add.is_contiguous()):
      raise ValueError(
          "diagonal_add must be contiguous float32 MPS [batch,nv]")
    return diagonal_add

  def build_euler_diagonal_device(self, q_deriv, timestep, *, dof_ids=None,
                                  counts=None):
    """Build awake ``dt*qDeriv`` entries admitted by pinned ``mj_EulerSkip``."""
    torch = self._torch
    if not self._has_euler_damping_mask:
      raise ValueError(
          "Euler diagonal needs euler_damping_dofs when model metadata "
          "does not include the pinned eligibility inputs")
    if (not isinstance(q_deriv, torch.Tensor)
        or tuple(q_deriv.shape) != (self.batch_size, self.nv)
        or q_deriv.dtype != torch.float32 or q_deriv.device.type != "mps"
        or not q_deriv.is_contiguous()):
      raise ValueError("q_deriv must be contiguous float32 MPS [batch,nv]")
    if (isinstance(timestep, (bool, np.bool_))
        or not isinstance(timestep, (int, float, np.integer, np.floating))
        or not np.isfinite(timestep) or float(timestep) <= 0):
      raise ValueError("timestep must be finite and positive")
    if dof_ids is None:
      dof_ids, counts = self._all_dof_ids, self._all_counts
    if (not isinstance(dof_ids, torch.Tensor)
        or tuple(dof_ids.shape) != (self.batch_size, max(self.nv, 1))
        or dof_ids.dtype != torch.int32 or dof_ids.device.type != "mps"
        or not dof_ids.is_contiguous()):
      raise ValueError("dof_ids must be contiguous MPS int32 [batch,max(nv,1)]")
    if (not isinstance(counts, torch.Tensor)
        or tuple(counts.shape) != (self.batch_size, 3)
        or counts.dtype != torch.int32 or counts.device.type != "mps"
        or not counts.is_contiguous()):
      raise ValueError("counts must be contiguous MPS int32 [batch,3]")
    self._euler_dt.fill_(float(timestep))
    output = self._workspace["diagonal_add"]
    output.zero_()
    if self.nv:
      self._library.build_euler_damping_diagonal(
          dof_ids.reshape(-1), counts.reshape(-1),
          self._euler_damping_mask, q_deriv.reshape(-1), output.reshape(-1),
          self._dims, self._euler_dt,
          threads=(self.batch_size,), group_size=(1,))
    return output[:, :self.nv]

  def run_mass_matvec_device(self, mass_blocks, vector, *, dof_ids=None,
                             counts=None, tendon_armature_blocks=None,
                             diagonal_add=None):
    """Apply the exact compiled sparse mass operator to one vector per world.

    This operator is a reusable primitive for future acceleration-space CG/
    Newton work. Its presence does not claim that those solver paths consume
    sparse storage yet.
    """
    torch = self._torch
    if (not isinstance(mass_blocks, torch.Tensor)
        or tuple(mass_blocks.shape) != (self.batch_size, self.nnz)
        or mass_blocks.dtype != torch.float32 or mass_blocks.device.type != "mps"
        or not mass_blocks.is_contiguous()):
      raise ValueError("mass_blocks must be contiguous float32 MPS [batch,nnz]")
    if (not isinstance(vector, torch.Tensor)
        or tuple(vector.shape) != (self.batch_size, self.nv)
        or vector.dtype != torch.float32 or vector.device.type != "mps"
        or not vector.is_contiguous()):
      raise ValueError("vector must be contiguous float32 MPS [batch,nv]")
    if dof_ids is None:
      dof_ids, counts = self._all_dof_ids, self._all_counts
    if (not isinstance(dof_ids, torch.Tensor)
        or tuple(dof_ids.shape) != (self.batch_size, max(self.nv, 1))
        or dof_ids.dtype != torch.int32 or dof_ids.device.type != "mps"
        or not dof_ids.is_contiguous()):
      raise ValueError("dof_ids must be contiguous MPS int32 [batch,max(nv,1)]")
    if (not isinstance(counts, torch.Tensor)
        or tuple(counts.shape) != (self.batch_size, 3)
        or counts.dtype != torch.int32 or counts.device.type != "mps"
        or not counts.is_contiguous()):
      raise ValueError("counts must be contiguous MPS int32 [batch,3]")
    if tendon_armature_blocks is None:
      tendon_armature_blocks = self._workspace["zero_blocks"][
          :self.batch_size * self.nnz].reshape(self.batch_size, self.nnz)
    elif (not isinstance(tendon_armature_blocks, torch.Tensor)
          or tuple(tendon_armature_blocks.shape) != (self.batch_size, self.nnz)
          or tendon_armature_blocks.dtype != torch.float32
          or tendon_armature_blocks.device.type != "mps"
          or not tendon_armature_blocks.is_contiguous()):
      raise ValueError(
          "tendon_armature_blocks must be contiguous float32 MPS [batch,nnz]")
    diagonal_add = self._validate_diagonal_add(diagonal_add)
    output = self._workspace["matvec_output"]
    output.zero_()
    if self.nv:
      self._library.build_dof_awake_mask(
          dof_ids.reshape(-1), counts.reshape(-1),
          self._workspace["dof_mask"].reshape(-1), self._dims,
          threads=(self.batch_size,), group_size=(1,))
    if self.ncomponent:
      self._library.apply_component_mass(
          mass_blocks.reshape(-1), tendon_armature_blocks.reshape(-1),
          vector.reshape(-1), self._workspace["dof_mask"].reshape(-1),
          self._component_layout, output.reshape(-1), self._dims,
          diagonal_add.reshape(-1),
          threads=(self.batch_size * self.ncomponent,), group_size=(1,))
    return output[:, :self.nv]
