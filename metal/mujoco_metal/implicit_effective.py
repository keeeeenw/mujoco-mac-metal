# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Matrix-free effective implicit operator and device-resident GMRES."""

from pathlib import Path

import numpy as np

from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity

_SHADER = Path(__file__).parent / "shaders" / "implicit_effective.metal"
DEFAULT_GMRES_RESTART_DIMENSION = 32


def effective_gmres_workspace_sizes(*, batch, nv, edge_count,
                                   krylov_dimension=None,
                                   max_iterations=None):
  """Exact element counts for the persistent matrix-free GMRES backings."""
  values = {}
  for name, value in (("batch", batch), ("nv", nv),
                      ("edge_count", edge_count)):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))):
      raise ValueError(f"{name} must be an integer")
    values[name] = int(value)
  batch, nv, edge_count = (values["batch"], values["nv"],
                           values["edge_count"])
  if batch <= 0 or nv < 0 or edge_count < 0:
    raise ValueError("GMRES dimensions must have positive batch and nonnegative nv/E")
  m = (min(max(nv, 1), DEFAULT_GMRES_RESTART_DIMENSION)
       if krylov_dimension is None else krylov_dimension)
  iterations = max(nv, 1) if max_iterations is None else max_iterations
  for name, value in (("krylov_dimension", m), ("max_iterations", iterations)):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))):
      raise ValueError(f"{name} must be an integer")
  m, iterations = int(m), int(iterations)
  if m < 1 or m > max(nv, 1) or iterations < 1:
    raise ValueError("GMRES dimension must be in [1,max(nv,1)] and iterations positive")
  if iterations > np.iinfo(np.int32).max:
    raise ValueError("GMRES iteration budget exceeds signed int32 range")
  maxnv = max(nv, 1)
  sizes = {
      "implicit_gmres_basis": batch * (m + 1) * maxnv,
      "implicit_gmres_hessenberg": batch * (m + 1) * m,
      "implicit_gmres_coefficients": batch * m,
      "implicit_gmres_work": batch * maxnv,
      "implicit_gmres_work_low": batch * maxnv,
      "implicit_gmres_coo_product": batch * maxnv,
      "implicit_gmres_coo_product_low": batch * maxnv,
      "implicit_gmres_solution": batch * maxnv,
      "implicit_gmres_residual_vector": batch * maxnv,
      "implicit_gmres_residual": batch,
      "implicit_gmres_residual_rhs": batch * (m + 1),
      "implicit_gmres_givens": 2 * batch * m,
      "implicit_gmres_rhs_norms": batch,
      "implicit_gmres_status": batch,
      "implicit_gmres_done": batch,
      "implicit_gmres_iterations": batch,
      "implicit_gmres_cycle_iterations": batch,
      "implicit_gmres_cycle_stop": batch,
      "implicit_gmres_edge_values": batch * max(edge_count, 1),
      "implicit_gmres_edge_rows": max(edge_count, 1),
      "implicit_gmres_edge_cols": max(edge_count, 1),
      "implicit_gmres_diagonal_slots": maxnv,
      "implicit_gmres_active_dof": batch * maxnv,
      "implicit_gmres_dims": 5,
      "implicit_gmres_params": 3,
      "implicit_gmres_writer_dims": 3,
  }
  _validate_workspace_index_capacity(
      batch, sizes,
      {"nv": nv, "krylov_dimension": m,
       "max_iterations": iterations, "edge_count": edge_count,
       "workspace_nv_plus_one": nv + 1})
  for name, count in sizes.items():
    if count > np.iinfo(np.int32).max:
      raise ValueError(f"{name} exceeds the Metal int32 address range ({count})")
  return sizes


def effective_gmres_workspace_bytes(*, batch, nv, edge_count,
                                    krylov_dimension=None,
                                    max_iterations=None):
  """Bytes for actual persistent GMRES tensors, including typed int buffers."""
  sizes = effective_gmres_workspace_sizes(
      batch=batch, nv=nv, edge_count=edge_count,
      krylov_dimension=krylov_dimension, max_iterations=max_iterations)
  integer = {"implicit_gmres_status", "implicit_gmres_done",
             "implicit_gmres_iterations", "implicit_gmres_cycle_iterations",
             "implicit_gmres_cycle_stop",
             "implicit_gmres_edge_rows",
             "implicit_gmres_edge_cols", "implicit_gmres_active_dof",
             "implicit_gmres_dims",
             "implicit_gmres_writer_dims"}
  return sum(count * 4 for count in sizes.values())


def select_gmres_krylov_dimension(*, batch, nv, edge_count, available_bytes,
                                  max_iterations=None,
                                  krylov_dimension=None):
  """Largest restart dimension fitting an explicit remaining-memory budget."""
  if (isinstance(available_bytes, (bool, np.bool_))
      or not isinstance(available_bytes, (int, np.integer))
      or int(available_bytes) <= 0):
    raise ValueError("available_bytes must be a positive integer")
  if krylov_dimension is not None:
    sizes = effective_gmres_workspace_sizes(
        batch=batch, nv=nv, edge_count=edge_count,
        krylov_dimension=krylov_dimension, max_iterations=max_iterations)
    required = sum(count * 4 for count in sizes.values())
    if required > int(available_bytes):
      raise ValueError(f"GMRES workspace needs at least {required} bytes")
    return int(krylov_dimension)
  upper = min(max(int(nv), 1), DEFAULT_GMRES_RESTART_DIMENSION)
  low, high, best = 1, upper, 0
  while low <= high:
    middle = (low + high) // 2
    required = effective_gmres_workspace_bytes(
        batch=batch, nv=nv, edge_count=edge_count,
        krylov_dimension=middle, max_iterations=max_iterations)
    if required <= int(available_bytes):
      best = middle
      low = middle + 1
    else:
      high = middle - 1
  if best == 0:
    minimum = effective_gmres_workspace_bytes(
        batch=batch, nv=nv, edge_count=edge_count,
        krylov_dimension=1, max_iterations=max_iterations)
    raise ValueError(f"GMRES workspace needs at least {minimum} bytes")
  return best


def effective_operator_coo_cpu(vector, edge_rows, edge_cols, edge_values,
                               *, nv=None, active_dof=None):
  """Small CPU oracle for the device COO derivative operator `D @ vector`."""
  x = np.asarray(vector, dtype=np.float64)
  rows = np.asarray(edge_rows)
  cols = np.asarray(edge_cols)
  values = np.asarray(edge_values, dtype=np.float64)
  if x.ndim != 2:
    raise ValueError("vector must have shape [batch,nv]")
  batch, inferred_nv = x.shape
  nv = inferred_nv if nv is None else int(nv)
  if nv != inferred_nv:
    raise ValueError("nv does not match vector width")
  if rows.dtype.kind not in "iu" or cols.dtype.kind not in "iu":
    raise ValueError("COO row/column indices must be integer arrays")
  if rows.ndim != 1 or cols.shape != rows.shape:
    raise ValueError("COO row and column arrays must have matching [E] shape")
  if values.shape != (batch, rows.size):
    raise ValueError("edge_values must have shape [batch,E]")
  if rows.size and (np.any(rows < 0) or np.any(rows >= nv)
                    or np.any(cols < 0) or np.any(cols >= nv)):
    raise ValueError("COO indices must be in [0,nv)")
  if active_dof is None:
    active = np.ones((batch, nv), dtype=bool)
  else:
    active = np.asarray(active_dof, dtype=bool)
    if active.shape != (batch, nv):
      raise ValueError("active_dof must have shape [batch,nv]")
  result = np.zeros_like(x)
  for edge, (row, col) in enumerate(zip(rows, cols)):
    result[:, row] += np.where(active[:, row] & active[:, col],
                               values[:, edge] * x[:, col], 0.0)
  return result


def effective_operator_pair_cpu(mass, vector_hi, vector_low, edge_rows,
                                edge_cols, edge_values, timestep, *,
                                active_dof=None):
  """Independent f64 oracle split into the production two-word ABI.

  All operands are first rounded to their actual binary32 device inputs. The
  exact ``(M-hD) (x_hi+x_low)`` value is then split into a binary32 high word
  and the residual low word, which is the contract of the paired device path.
  """
  mass = np.asarray(mass, dtype=np.float32)
  hi = np.asarray(vector_hi, dtype=np.float32)
  low = np.asarray(vector_low, dtype=np.float32)
  values = np.asarray(edge_values, dtype=np.float32)
  rows, cols = np.asarray(edge_rows), np.asarray(edge_cols)
  if (mass.ndim != 3 or mass.shape[1] != mass.shape[2]
      or hi.ndim != 2 or low.shape != hi.shape
      or mass.shape != (hi.shape[0], hi.shape[1], hi.shape[1])):
    raise ValueError("mass and vector pair must have matching [batch,nv] shapes")
  batch, nv = hi.shape
  if (rows.dtype.kind not in "iu" or cols.dtype.kind not in "iu"
      or rows.ndim != 1 or cols.shape != rows.shape
      or values.shape != (batch, rows.size)
      or (rows.size and (np.any(rows < 0) or np.any(rows >= nv)
                         or np.any(cols < 0) or np.any(cols >= nv)))):
    raise ValueError("COO derivative inputs have invalid shapes or indices")
  if not np.isfinite(timestep) or timestep <= 0:
    raise ValueError("timestep must be finite and positive")
  if active_dof is None:
    active = np.ones((batch, nv), dtype=bool)
  else:
    active = np.asarray(active_dof, dtype=bool)
    if active.shape != (batch, nv):
      raise ValueError("active_dof must have shape [batch,nv]")
  x = hi.astype(np.float64) + low.astype(np.float64)
  exact = np.zeros((batch, nv), dtype=np.float64)
  for world in range(batch):
    for row in range(nv):
      if not active[world, row]:
        continue
      acc = np.float64(0.)
      for col in range(nv):
        if active[world, col]:
          acc += np.float64(mass[world, row, col]) * x[world, col]
      for edge, (edge_row, edge_col) in enumerate(zip(rows, cols)):
        if (edge_row == row and active[world, edge_col]):
          acc -= (np.float64(timestep) * np.float64(values[world, edge])
                  * x[world, edge_col])
      exact[world, row] = acc
  out_hi = exact.astype(np.float32)
  out_low = (exact - out_hi.astype(np.float64)).astype(np.float32)
  return out_hi, out_low


def solve_effective_cpu(mass, edge_rows, edge_cols, edge_values, rhs,
                        timestep, *, active_dof=None):
  """Independent dense small-system oracle for `(M - hD) x = rhs`."""
  M = np.asarray(mass, dtype=np.float64)
  b = np.asarray(rhs, dtype=np.float64)
  if b.ndim != 2:
    raise ValueError("rhs must have shape [batch,nv]")
  if M.ndim == 2:
    M = np.broadcast_to(M, (b.shape[0], *M.shape))
  if M.shape != (b.shape[0], b.shape[1], b.shape[1]):
    raise ValueError("mass and rhs dimensions do not match")
  if (isinstance(timestep, (bool, np.bool_))
      or not isinstance(timestep, (int, float, np.integer, np.floating))
      or not np.isfinite(timestep) or float(timestep) <= 0):
    raise ValueError("timestep must be finite and positive")
  batch, nv = b.shape
  # Build the source COO matrix independently of a dense derivative input.
  matrix = np.zeros((nv, nv), dtype=np.float64)
  rows, cols = np.asarray(edge_rows), np.asarray(edge_cols)
  vals = np.asarray(edge_values, dtype=np.float64)
  if vals.ndim == 1:
    vals = np.broadcast_to(vals, (batch, vals.size))
  if vals.shape != (batch, rows.size):
    raise ValueError("edge_values must have shape [batch,E]")
  if (rows.ndim != 1 or cols.shape != rows.shape
      or rows.dtype.kind not in "iu" or cols.dtype.kind not in "iu"):
    raise ValueError("COO row/column indices must be matching integer vectors")
  if rows.size and (np.any(rows < 0) or np.any(rows >= nv)
                    or np.any(cols < 0) or np.any(cols >= nv)):
    raise ValueError("COO indices must be in [0,nv)")
  if active_dof is not None:
    active_dof = np.asarray(active_dof, dtype=bool)
    if active_dof.shape != (batch, nv):
      raise ValueError("active_dof must have shape [batch,nv]")
  outputs = np.empty_like(b)
  for world in range(batch):
    matrix.fill(0)
    np.add.at(matrix, (rows, cols), vals[world])
    active = np.ones(nv, dtype=bool) if active_dof is None else np.asarray(active_dof, dtype=bool)[world]
    effective = M[world].copy()
    effective[np.ix_(active, active)] -= float(timestep) * matrix[np.ix_(active, active)]
    effective[~active, :] = 0
    effective[:, ~active] = 0
    effective[np.flatnonzero(~active), np.flatnonzero(~active)] = 1
    outputs[world] = np.linalg.solve(effective, np.where(active, b[world], 0.0))
  return outputs


class MetalEffectiveImplicitSolve:
  """Device-resident restarted left-preconditioned GMRES for `M-hD`.

  The component solver owns and factors the exact sparse M (+ tendon
  armature) blocks. Each Arnoldi product gathers D*x from a fixed COO support
  union, then applies the retained component factors. Restart dimension is an
  explicit memory-budget choice and max_iterations is an explicit solve
  budget. Every restart checks the true physical residual; a world that misses
  tolerance at the budget boundary returns status 2 and a zero solution.
  """

  def __init__(self, component_solver, *, edge_rows, edge_cols,
               krylov_dimension=None, max_iterations=None):
    if not all(hasattr(component_solver, name) for name in (
        "factorize_device", "solve_factored_vector_device",
        "merge_world_status", "batch_size", "nv")):
      raise TypeError("component_solver must provide retained component factors")
    rows = np.asarray(edge_rows)
    cols = np.asarray(edge_cols)
    if (rows.ndim != 1 or cols.shape != rows.shape
        or rows.dtype.kind not in "iu" or cols.dtype.kind not in "iu"):
      raise ValueError("edge_rows/edge_cols must be matching integer vectors")
    if rows.size and (np.any(rows < 0) or np.any(rows >= component_solver.nv)
                      or np.any(cols < 0) or np.any(cols >= component_solver.nv)):
      raise ValueError("COO support indices exceed component nv")
    if rows.size:
      order = np.lexsort((cols, rows))
      if not np.array_equal(order, np.arange(rows.size)):
        raise ValueError("COO support must be sorted by row then column")
      if np.any((rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])):
        raise ValueError("COO support must not contain duplicate entries")
    import torch
    packed_rows = rows.astype(np.int32, copy=True)
    packed_cols = cols.astype(np.int32, copy=True)
    if not rows.size:
      packed_rows = np.zeros(1, dtype=np.int32)
      packed_cols = np.zeros(1, dtype=np.int32)
    self._torch = torch
    self._solver = component_solver
    self.batch_size = int(component_solver.batch_size)
    self.nv = int(component_solver.nv)
    self.edge_count = int(rows.size)
    self._device = torch.device("mps")
    self.krylov_dimension = (
        min(max(self.nv, 1), DEFAULT_GMRES_RESTART_DIMENSION)
        if krylov_dimension is None else krylov_dimension)
    self.max_iterations = (max(self.nv, 1) if max_iterations is None
                           else max_iterations)
    self.workspace_sizes = effective_gmres_workspace_sizes(
        batch=self.batch_size, nv=self.nv, edge_count=self.edge_count,
        krylov_dimension=self.krylov_dimension,
        max_iterations=self.max_iterations)
    self.krylov_dimension = int(self.krylov_dimension)
    self.max_iterations = int(self.max_iterations)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._dims = torch.tensor(
        [self.nv, self.edge_count, self.batch_size, self.krylov_dimension,
         self.max_iterations],
        dtype=torch.int32, device=self._device)
    self._params = torch.zeros(3, dtype=torch.float32, device=self._device)
    self._edge_rows = torch.as_tensor(packed_rows, dtype=torch.int32,
                                      device=self._device)
    self._edge_cols = torch.as_tensor(packed_cols, dtype=torch.int32,
                                      device=self._device)
    B, n, E, m = (self.batch_size, self.nv, max(self.edge_count, 1),
                  self.krylov_dimension)
    self._workspace = {
        "edge_values": torch.zeros((B, E), dtype=torch.float32, device=self._device),
        "active_dof": torch.zeros((B, max(n, 1)), dtype=torch.int32, device=self._device),
        "basis": torch.zeros((B, m + 1, max(n, 1)), dtype=torch.float32, device=self._device),
        "hessenberg": torch.zeros((B, m + 1, m), dtype=torch.float32, device=self._device),
        "coefficients": torch.zeros((B, m), dtype=torch.float32, device=self._device),
        "work": torch.zeros((B, max(n, 1)), dtype=torch.float32, device=self._device),
        "work_low": torch.zeros((B, max(n, 1)), dtype=torch.float32, device=self._device),
        "coo_product": torch.zeros((B, max(n, 1)), dtype=torch.float32, device=self._device),
        "coo_product_low": torch.zeros((B, max(n, 1)), dtype=torch.float32, device=self._device),
        "solution": torch.zeros((B, max(n, 1)), dtype=torch.float32, device=self._device),
        "residual_vector": torch.zeros((B, max(n, 1)), dtype=torch.float32, device=self._device),
        "residual_rhs": torch.zeros((B, m + 1), dtype=torch.float32, device=self._device),
        "givens_cos": torch.zeros((B, m), dtype=torch.float32, device=self._device),
        "givens_sin": torch.zeros((B, m), dtype=torch.float32, device=self._device),
        "rhs_norms": torch.zeros((B,), dtype=torch.float32, device=self._device),
        "status": torch.zeros((B,), dtype=torch.int32, device=self._device),
        "done": torch.zeros((B,), dtype=torch.int32, device=self._device),
        "iterations": torch.zeros((B,), dtype=torch.int32, device=self._device),
        "cycle_iterations": torch.zeros((B,), dtype=torch.int32, device=self._device),
        "cycle_stop": torch.zeros((B,), dtype=torch.int32, device=self._device),
        "residual": torch.zeros((B,), dtype=torch.float32, device=self._device),
    }

  def apply_effective_operator_device(self, mass_blocks, vector, edge_values,
                                      timestep, *, dof_ids=None, counts=None,
                                      tendon_armature_blocks=None,
                                      diagonal_add=None):
    """Apply `M-hD` without factorization or solve, for inverse queries."""
    torch = self._torch
    B, n = self.batch_size, self.nv
    if n == 0:
      status = torch.zeros((B,), dtype=torch.int32, device=self._device)
      return torch.zeros((B, 0), dtype=torch.float32, device=self._device), status
    for name, value, shape, dtype in (
        ("vector", vector, (B, n), torch.float32),
        ("edge_values", edge_values, (B, max(self.edge_count, 1)), torch.float32)):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != dtype or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS {dtype} with shape {shape}")
    if (isinstance(timestep, (bool, np.bool_))
        or not isinstance(timestep, (int, float, np.integer, np.floating))
        or not np.isfinite(timestep) or timestep <= 0
        or float(timestep) > np.finfo(np.float32).max):
      raise ValueError("timestep must be positive and representable as float32")
    if dof_ids is None:
      dof_ids, counts = self._solver._all_dof_ids, self._solver._all_counts
    for name, value, shape, dtype in (
        ("dof_ids", dof_ids, (B, max(n, 1)), torch.int32),
        ("counts", counts, (B, 3), torch.int32)):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != dtype or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS {dtype} with shape {shape}")
    self._params[0] = float(timestep)
    self._workspace["edge_values"].copy_(edge_values)
    status = self._workspace["status"]
    status.zero_()
    self._library.effective_validate_edge_values(
        self._workspace["edge_values"].reshape(-1), self._dims, status,
        threads=(B,), group_size=(1,))
    mass_product = self._solver.run_mass_matvec_device(
        mass_blocks, vector, dof_ids=dof_ids, counts=counts,
        tendon_armature_blocks=tendon_armature_blocks,
        diagonal_add=diagonal_add)
    self._workspace["active_dof"].copy_(self._solver._workspace["dof_mask"])
    self._library.effective_coo_vector_matvec(
        vector.reshape(-1), self._workspace["edge_values"].reshape(-1),
        self._edge_rows, self._edge_cols,
        self._workspace["active_dof"].reshape(-1), self._dims,
        self._workspace["coo_product"].reshape(-1),
        threads=(B * n,), group_size=(1,))
    output = self._workspace["work"]
    self._library.effective_combine_operator(
        mass_product.reshape(-1), self._workspace["coo_product"].reshape(-1),
        self._workspace["active_dof"].reshape(-1), self._dims,
        self._params, status, output.reshape(-1),
        threads=(B,), group_size=(1,))
    return output[:, :n], status

  def apply_effective_operator_pair_device(
      self, mass_blocks, vector_hi, vector_low, edge_values, timestep, *,
      dof_ids=None, counts=None, tendon_armature_blocks=None,
      diagonal_add=None):
    """Apply ``M-hD`` to a two-word vector and retain operator roundoff.

    Unlike two separate rounded operator calls, this forms the mass and COO
    derivative products from both words before reducing, so the low result is
    the residual of the same accepted operator application as the high result.
    """
    torch = self._torch
    B, n = self.batch_size, self.nv
    if n == 0:
      status = torch.zeros((B,), dtype=torch.int32, device=self._device)
      empty = torch.zeros((B, 0), dtype=torch.float32, device=self._device)
      return empty, empty.clone(), status
    for name, value in (("vector_hi", vector_hi), ("vector_low", vector_low)):
      if (not isinstance(value, torch.Tensor)
          or tuple(value.shape) != (B, n) or value.dtype != torch.float32
          or value.device.type != "mps" or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous float32 MPS [batch,nv]")
    if (not isinstance(edge_values, torch.Tensor)
        or tuple(edge_values.shape) != (B, max(self.edge_count, 1))
        or edge_values.dtype != torch.float32 or edge_values.device.type != "mps"
        or not edge_values.is_contiguous()):
      raise ValueError("edge_values must be contiguous float32 MPS [batch,E]")
    if (isinstance(timestep, (bool, np.bool_))
        or not isinstance(timestep, (int, float, np.integer, np.floating))
        or not np.isfinite(timestep) or timestep <= 0
        or float(timestep) > np.finfo(np.float32).max):
      raise ValueError("timestep must be positive and representable as float32")
    if dof_ids is None:
      dof_ids, counts = self._solver._all_dof_ids, self._solver._all_counts
    self._params[0] = float(timestep)
    self._workspace["edge_values"].copy_(edge_values)
    status = self._workspace["status"]
    status.zero_()
    self._library.effective_validate_edge_values(
        self._workspace["edge_values"].reshape(-1), self._dims, status,
        threads=(B,), group_size=(1,))
    mass_hi, mass_low = self._solver.run_mass_matvec_pair_device(
        mass_blocks, vector_hi, vector_low, dof_ids=dof_ids, counts=counts,
        tendon_armature_blocks=tendon_armature_blocks,
        diagonal_add=diagonal_add)
    self._workspace["active_dof"].copy_(self._solver._workspace["dof_mask"])
    self._library.effective_coo_vector_matvec_pair(
        vector_hi.reshape(-1), vector_low.reshape(-1),
        self._workspace["edge_values"].reshape(-1), self._edge_rows,
        self._edge_cols, self._workspace["active_dof"].reshape(-1),
        self._dims, self._workspace["coo_product"].reshape(-1),
        self._workspace["coo_product_low"].reshape(-1),
        threads=(B * n,), group_size=(1,))
    self._library.effective_combine_operator_pair(
        mass_hi.reshape(-1), mass_low.reshape(-1),
        self._workspace["coo_product"].reshape(-1),
        self._workspace["coo_product_low"].reshape(-1),
        self._workspace["active_dof"].reshape(-1), self._dims, self._params,
        status, self._workspace["work"].reshape(-1),
        self._workspace["work_low"].reshape(-1),
        threads=(B,), group_size=(1,))
    return self._workspace["work"][:, :n], self._workspace["work_low"][:, :n], status

  def solve_device(self, mass_blocks, rhs, edge_values, timestep, *,
                   dof_ids=None, counts=None, tendon_armature_blocks=None,
                   diagonal_add=None, relative_tolerance=1e-5,
                   absolute_tolerance=1e-7):
    """Solve `(M - timestep*D)x=rhs`; return solution, status, residual, iters."""
    torch = self._torch
    n, B = self.nv, self.batch_size
    if n == 0:
      status = torch.zeros((B,), dtype=torch.int32, device=self._device)
      return torch.zeros((B, 0), dtype=torch.float32, device=self._device), status, self._workspace["residual"], self._workspace["iterations"]
    def tensor_ok(name, value, shape, dtype):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
          or value.dtype != dtype or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS {dtype} with shape {shape}")
    tensor_ok("rhs", rhs, (B, n), torch.float32)
    tensor_ok("edge_values", edge_values, (B, max(self.edge_count, 1)), torch.float32)
    if (isinstance(timestep, (bool, np.bool_))
        or not isinstance(timestep, (int, float, np.integer, np.floating))
        or not np.isfinite(timestep) or float(timestep) <= 0):
      raise ValueError("timestep must be finite and positive")
    scalar_values = (("relative_tolerance", relative_tolerance, False),
                     ("absolute_tolerance", absolute_tolerance, True))
    for name, value, allow_zero in scalar_values:
      if (isinstance(value, (bool, np.bool_))
          or not isinstance(value, (int, float, np.integer, np.floating))
          or not np.isfinite(value)
          or (value < 0 if allow_zero else value <= 0)):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    float32_limit = np.finfo(np.float32).max
    if (abs(float(timestep)) > float32_limit
        or float(relative_tolerance) > float32_limit
        or float(absolute_tolerance) > float32_limit):
      raise ValueError("GMRES scalar parameters must be representable as float32")
    if dof_ids is None:
      dof_ids, counts = self._solver._all_dof_ids, self._solver._all_counts
    tensor_ok("dof_ids", dof_ids, (B, max(n, 1)), torch.int32)
    tensor_ok("counts", counts, (B, 3), torch.int32)
    w = self._workspace
    self._dims[0] = n
    self._dims[1] = self.edge_count
    self._dims[2] = B
    self._dims[3] = self.krylov_dimension
    self._dims[4] = self.max_iterations
    self._params[0] = float(timestep)
    self._params[1] = float(relative_tolerance)
    self._params[2] = float(absolute_tolerance)
    w["edge_values"].copy_(edge_values)
    self._solver.factorize_device(
        mass_blocks, dof_ids=dof_ids, counts=counts,
        tendon_armature_blocks=tendon_armature_blocks,
        diagonal_add=diagonal_add)
    w["active_dof"].copy_(self._solver.factor_active_dof_mask)
    w["status"].zero_(); w["iterations"].zero_(); w["residual"].zero_()
    w["done"].zero_(); w["solution"].zero_()
    component_status = self._solver._workspace["factor_status"][:, :self._solver.ncomponent]
    self._solver.merge_world_status(component_status, w["status"])
    self._library.effective_validate_edge_values(
        w["edge_values"].reshape(-1), self._dims, w["status"],
        threads=(B,), group_size=(1,))
    preconditioned_rhs, component_status = self._solver.solve_factored_vector_device(rhs)
    self._solver.merge_world_status(component_status, w["status"])
    w["cycle_stop"].fill_(2)
    self._library.effective_gmres_prepare_cycle(
        preconditioned_rhs.reshape(-1), w["active_dof"].reshape(-1),
        self._dims, w["status"], w["done"], w["cycle_iterations"],
        w["cycle_stop"], w["basis"].reshape(-1),
        w["hessenberg"].reshape(-1), w["givens_cos"].reshape(-1),
        w["givens_sin"].reshape(-1), w["residual_rhs"].reshape(-1),
        w["rhs_norms"], threads=(B,), group_size=(1,))
    def complete_stopped_cycles():
      self._library.effective_gmres_finish_cycle(
          w["basis"].reshape(-1), w["hessenberg"].reshape(-1),
          w["residual_rhs"].reshape(-1), w["cycle_iterations"],
          w["cycle_stop"], w["done"], w["status"], self._dims,
          w["coefficients"].reshape(-1), w["solution"].reshape(-1),
          w["residual"], threads=(B,), group_size=(1,))
      mass_product = self._solver.run_mass_matvec_device(
          mass_blocks, w["solution"][:, :n], dof_ids=dof_ids, counts=counts,
          tendon_armature_blocks=tendon_armature_blocks,
          diagonal_add=diagonal_add)
      self._library.effective_coo_vector_matvec(
          w["solution"].reshape(-1), w["edge_values"].reshape(-1),
          self._edge_rows, self._edge_cols, w["active_dof"].reshape(-1),
          self._dims, w["coo_product"].reshape(-1),
          threads=(B * n,), group_size=(1,))
      self._library.effective_gmres_true_residual(
          rhs.reshape(-1), mass_product.reshape(-1),
          w["coo_product"].reshape(-1), w["active_dof"].reshape(-1),
          self._dims, self._params, w["status"], w["done"],
          w["iterations"], w["residual_vector"].reshape(-1),
          w["residual"], w["rhs_norms"], w["solution"].reshape(-1),
          w["cycle_iterations"], w["cycle_stop"],
          threads=(B,), group_size=(1,))
      preconditioned_residual, component_status = self._solver.solve_factored_vector_device(
          w["residual_vector"][:, :n])
      self._solver.merge_world_status(component_status, w["status"])
      self._library.effective_gmres_prepare_cycle(
          preconditioned_residual.reshape(-1), w["active_dof"].reshape(-1),
          self._dims, w["status"], w["done"], w["cycle_iterations"],
          w["cycle_stop"], w["basis"].reshape(-1),
          w["hessenberg"].reshape(-1), w["givens_cos"].reshape(-1),
          w["givens_sin"].reshape(-1), w["residual_rhs"].reshape(-1),
          w["rhs_norms"], threads=(B,), group_size=(1,))

    # One host slot consumes at most one Arnoldi iteration per world. Device
    # cycle_stop tracks independent restart boundaries; it is never read back.
    # This bounds total Arnoldi work by max_iterations, not max_iterations*m.
    for _slot in range(self.max_iterations):
      complete_stopped_cycles()
      self._library.effective_coo_matvec(
          w["basis"].reshape(-1), w["edge_values"].reshape(-1),
          self._edge_rows, self._edge_cols, w["active_dof"].reshape(-1),
          w["cycle_iterations"], self._dims, w["status"], w["done"],
          w["cycle_stop"], w["coo_product"].reshape(-1),
          threads=(B * n,), group_size=(1,))
      preconditioned_dv, component_status = self._solver.solve_factored_vector_device(
          w["coo_product"][:, :n])
      self._solver.merge_world_status(component_status, w["status"])
      self._library.effective_gmres_arnoldi(
          w["basis"].reshape(-1), preconditioned_dv.reshape(-1),
          w["active_dof"].reshape(-1), self._dims,
          self._params, w["status"], w["done"], w["cycle_iterations"],
          w["cycle_stop"], w["iterations"], w["hessenberg"].reshape(-1),
          w["givens_cos"].reshape(-1), w["givens_sin"].reshape(-1),
          w["residual_rhs"].reshape(-1), w["work"].reshape(-1),
          w["rhs_norms"], threads=(B,), group_size=(1,))
    # The last slot can itself finish a cycle, so always check its physical
    # residual before returning. No restart is prepared after the budget ends.
    complete_stopped_cycles()
    return w["solution"][:, :n], w["status"], w["residual"], w["iterations"]
