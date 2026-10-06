# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Matrix-free pinned flex stiffness correction for component mass profiles."""

import numpy as np


def sparse_flex_correction_workspace_sizes(batch, nv, edge_count=0):
  """Exact persistent vector/scalar buffers for the 50-step flex PCG."""
  for name, value in (("batch", batch), ("nv", nv)):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) or int(value) < 0):
      raise ValueError(f"{name} must be a nonnegative integer")
  if (isinstance(edge_count, (bool, np.bool_))
      or not isinstance(edge_count, (int, np.integer))
      or int(edge_count) < 0):
    raise ValueError("edge_count must be a nonnegative integer")
  b, n, e = int(batch), int(nv), int(edge_count)
  if b <= 0:
    raise ValueError("batch must be positive")
  result = {f"sparse_flex.{name}": b * n for name in
            ("rhs", "qacc", "residual", "direction", "product", "tmp",
             "preconditioned")}
  result.update({f"sparse_flex.{name}": b for name in
                 ("rz", "next_rz", "pap", "rhs_norm", "residual_norm",
                  "alpha", "beta", "status", "iterations", "active")})
  # The effective solver owns its own mutable COO work buffer. These copies
  # preserve the full nonsymmetric D and the lower-symmetric qH preconditioner
  # across repeated operator/preconditioner dispatches.
  edge_width = max(e, 1)
  result["sparse_flex.full_derivative_values"] = b * edge_width
  result["sparse_flex.preconditioner_derivative_values"] = b * edge_width
  return result


class SparseFlexImplicitCorrection:
  """Pinned 50-step flex correction with no global mass or stiffness matrix.

  The effective sparse operator applies the exact compiled mass and D-COO
  terms. Flex material products are produced from a frozen flex context.
  Each H-inverse preconditioner application uses the bounded effective GMRES
  solve, leaving all numerical decisions on device.
  """

  def __init__(self, batch_size, nv, edge_count=0):
    sizes = sparse_flex_correction_workspace_sizes(batch_size, nv,
                                                    edge_count)
    from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
    _validate_workspace_index_capacity(
        int(batch_size), sizes,
        {"nv": int(nv), "edge_count": int(edge_count)})
    import torch
    self.batch_size, self.nv = int(batch_size), int(nv)
    self.edge_count = int(edge_count)
    self._torch = torch
    self._device = torch.device("mps")
    b, n = self.batch_size, self.nv
    self._vectors = {name: torch.zeros((b, n), dtype=torch.float32,
                                      device=self._device)
                     for name in ("rhs", "qacc", "residual", "direction",
                                  "product", "tmp", "preconditioned")}
    self._scalars = {name: torch.zeros((b,), dtype=torch.float32,
                                      device=self._device)
                     for name in ("rz", "next_rz", "pap", "rhs_norm",
                                  "residual_norm", "alpha", "beta")}
    self._status = torch.zeros((b,), dtype=torch.int32, device=self._device)
    self._iterations = torch.zeros((b,), dtype=torch.int32, device=self._device)
    self._active = torch.zeros((b,), dtype=torch.bool, device=self._device)
    edge_width = max(self.edge_count, 1)
    self._full_derivative_values = torch.zeros(
        (b, edge_width), dtype=torch.float32, device=self._device)
    self._preconditioner_derivative_values = torch.zeros(
        (b, edge_width), dtype=torch.float32, device=self._device)

  def capture_full_derivative_values(self, edge_values):
    """Snapshot compiled full D before an effective solver can reuse scratch."""
    torch = self._torch
    expected = (self.batch_size, max(self.edge_count, 1))
    if (not isinstance(edge_values, torch.Tensor)
        or tuple(edge_values.shape) != expected
        or edge_values.dtype != torch.float32
        or edge_values.device.type != "mps"
        or not edge_values.is_contiguous()):
      raise ValueError(
          f"edge_values must be contiguous float32 MPS {expected}")
    self._full_derivative_values.copy_(edge_values)

  @property
  def full_derivative_values(self):
    """Borrowed preserved nonsymmetric D values for the flex correction."""
    return self._full_derivative_values

  def _dot(self, left, right, out):
    self._torch.sum(left * right, dim=1, out=out)

  @staticmethod
  def _mask_direction_(direction, candidate, active):
    """Copy active PCG directions and clear inactive rows without NaN*0."""
    direction.copy_(candidate)
    direction.masked_fill_(~active[:, None], 0.0)

  def _apply(self, vector, *, flex, context, effective_solver, mass_blocks,
             edge_values, timestep, dof_ids, counts, armature_blocks):
    base, status = effective_solver.apply_effective_operator_device(
        mass_blocks, vector, edge_values, timestep, dof_ids=dof_ids,
        counts=counts, tendon_armature_blocks=armature_blocks)
    out = self._vectors["product"]
    out.copy_(base)
    products = flex.apply_material_operator_context_device(context, vector)
    h = float(timestep)
    out.add_(products["interp_stiffness"], alpha=-(h * h))
    out.add_(products["interp_damped_stiffness"], alpha=-h)
    out.add_(products["bend_stiffness"], alpha=h * h)
    out.add_(products["bend_damped_stiffness"], alpha=h)
    return out, status

  def _precondition(self, residual, *, effective_solver, mass_blocks,
                    edge_values, timestep, dof_ids, counts, armature_blocks,
                    active):
    solution, status, _, _ = effective_solver.solve_device(
        mass_blocks, residual, edge_values, timestep, dof_ids=dof_ids,
        counts=counts, tendon_armature_blocks=armature_blocks,
        relative_tolerance=1e-6, absolute_tolerance=1e-8)
    out = self._vectors["preconditioned"]
    out.copy_(solution)
    self._status.copy_(self._torch.where(
        active & (status != 0), status, self._status))
    return out

  def run_device(self, *, flex, context, effective_solver, mass_blocks,
                 edge_values, preconditioner_values, armature_blocks,
                 qfrc_total, qvel, initial_qacc, timestep, dof_ids=None,
                 counts=None, enabled=None):
    """Apply the pinned flex PCG stage and return borrowed result buffers."""
    torch = self._torch
    b, n = self.batch_size, self.nv
    for name, value in (("qfrc_total", qfrc_total), ("qvel", qvel),
                        ("initial_qacc", initial_qacc)):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (b, n)
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous float32 MPS [batch,nv]")
    if enabled is None:
      enabled = torch.ones((b,), dtype=torch.bool, device=self._device)
    if (not isinstance(enabled, torch.Tensor) or tuple(enabled.shape) != (b,)
        or enabled.dtype != torch.bool or enabled.device.type != "mps"
        or not enabled.is_contiguous()):
      raise ValueError("enabled must be contiguous MPS bool [batch]")
    edge_shape = (b, max(self.edge_count, 1))
    for name, value in (("edge_values", edge_values),
                        ("preconditioner_values", preconditioner_values)):
      if (not isinstance(value, torch.Tensor)
          or tuple(value.shape) != edge_shape
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(
            f"{name} must be contiguous float32 MPS {edge_shape}")
    # Freeze both operators before the first effective solve. Those methods
    # reuse one internal COO tensor and would otherwise overwrite an aliased
    # writer backing between full-D matvecs and lower-qH preconditioning.
    self._full_derivative_values.copy_(edge_values)
    self._preconditioner_derivative_values.copy_(preconditioner_values)
    edge_values = self._full_derivative_values
    preconditioner_values = self._preconditioner_derivative_values
    w, s = self._vectors, self._scalars
    self._status.zero_()
    self._iterations.zero_()
    w["qacc"].copy_(initial_qacc)
    if n == 0:
      return {"qacc": w["qacc"], "status": self._status,
              "iterations": self._iterations}

    h = float(timestep)
    terms = flex.apply_material_operator_context_device(context, qvel)
    w["rhs"].copy_(qfrc_total)
    w["rhs"].add_(terms["interp_stiffness"], alpha=h)
    w["rhs"].add_(terms["bend_stiffness"], alpha=-h)
    product, op_status = self._apply(
        w["qacc"], flex=flex, context=context,
        effective_solver=effective_solver, mass_blocks=mass_blocks,
        edge_values=edge_values, timestep=h, dof_ids=dof_ids, counts=counts,
        armature_blocks=armature_blocks)
    w["residual"].copy_(w["rhs"]).sub_(product)
    finite = (torch.isfinite(w["rhs"]).all(dim=1)
              & torch.isfinite(w["residual"]).all(dim=1)
              & torch.isfinite(w["qacc"]).all(dim=1))
    self._status.copy_(torch.where(enabled & ~finite, 4, 0).to(torch.int32))
    self._status.copy_(torch.where(enabled & (op_status != 0), op_status,
                                   self._status))
    self._dot(w["rhs"], w["rhs"], s["rhs_norm"])
    self._dot(w["residual"], w["residual"], s["residual_norm"])
    self._active.copy_(
        enabled & finite & (self._status == 0)
        & (s["residual_norm"] >= 1e-10 * s["rhs_norm"])
        & (s["residual_norm"] >= 1e-15))
    z = self._precondition(
        w["residual"], effective_solver=effective_solver,
        mass_blocks=mass_blocks, edge_values=preconditioner_values,
        timestep=h, dof_ids=dof_ids, counts=counts,
        armature_blocks=armature_blocks, active=self._active)
    w["direction"].copy_(z)
    self._dot(w["residual"], z, s["rz"])
    for _ in range(50):
      product, op_status = self._apply(
          w["direction"], flex=flex, context=context,
          effective_solver=effective_solver, mass_blocks=mass_blocks,
          edge_values=edge_values, timestep=h, dof_ids=dof_ids, counts=counts,
          armature_blocks=armature_blocks)
      self._status.copy_(torch.where(
          self._active & (op_status != 0), op_status, self._status))
      self._dot(w["direction"], product, s["pap"])
      good = torch.isfinite(s["pap"]) & torch.isfinite(s["rz"])
      self._status.copy_(torch.where(self._active & ~good, 4, self._status))
      self._active.logical_and_(good & (torch.abs(s["pap"]) >= 1e-15)
                                & (self._status == 0))
      s["alpha"].copy_(torch.where(
          self._active,
          s["rz"] / torch.where(self._active, s["pap"], 1.0), 0.0))
      torch.mul(w["direction"], s["alpha"][:, None], out=w["tmp"])
      w["qacc"].add_(w["tmp"])
      torch.mul(product, s["alpha"][:, None], out=w["tmp"])
      w["residual"].sub_(w["tmp"])
      self._iterations.add_(self._active.to(torch.int32))
      self._dot(w["residual"], w["residual"], s["residual_norm"])
      finite_iterate = (torch.isfinite(s["residual_norm"])
                        & torch.isfinite(w["qacc"]).all(dim=1))
      self._status.copy_(torch.where(
          self._active & ~finite_iterate, 4, self._status))
      self._active.logical_and_(
          finite_iterate & (self._status == 0)
          & (s["residual_norm"] >= 1e-10 * s["rhs_norm"])
          & (s["residual_norm"] >= 1e-15))
      z = self._precondition(
          w["residual"], effective_solver=effective_solver,
          mass_blocks=mass_blocks, edge_values=preconditioner_values,
          timestep=h, dof_ids=dof_ids, counts=counts,
          armature_blocks=armature_blocks, active=self._active)
      self._dot(w["residual"], z, s["next_rz"])
      denom = torch.where(torch.abs(s["rz"]) >= 1e-15, s["rz"], 1.0)
      s["beta"].copy_(torch.where(self._active, s["next_rz"] / denom, 0.0))
      torch.mul(w["direction"], s["beta"][:, None], out=w["tmp"])
      w["tmp"].add_(z)
      self._mask_direction_(w["direction"], w["tmp"], self._active)
      s["rz"].copy_(s["next_rz"])

    retain = (~enabled | (self._status != 0))[:, None]
    w["qacc"].copy_(torch.where(retain, initial_qacc, w["qacc"]))
    return {"qacc": w["qacc"], "status": self._status,
            "iterations": self._iterations}
