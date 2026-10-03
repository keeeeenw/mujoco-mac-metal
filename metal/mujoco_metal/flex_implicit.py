# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned implicit flex stiffness correction in native device workspaces.

MuJoCo 3.10 engine_forward.c flexInterp_cgsolve supplies a separate stiffness
correction after its ordinary implicit velocity solve. These frozen material
operators differ from the full position derivative of corotational force.
"""
import mujoco

from mujoco_metal.implicit import implicit_workspace_elements


OPERATOR_NAMES = ("interp_stiffness", "interp_damped_stiffness",
                  "bend_stiffness", "bend_damped_stiffness")


class FlexImplicitCorrection:
  """Source-derived 50-iteration preconditioned CG with per-world stopping.

  Inputs stay on MPS. ``base_matrix`` is M-h*qDeriv with its full compiled D
  structure, while ``preconditioner`` is the factorization input of the normal
  implicit solve (implicitfast uses the gathered symmetric lower triangle).
  A false enabled mask preserves that world's initial acceleration. Returned
  tensors borrow persistent storage; the next call overwrites them.
  """
  def __init__(self, model, batch_size=1):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("flex implicit correction requires a MuJoCo MjModel")
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError("flex implicit correction requires MuJoCo 3.10.0")
    integrator = int(model.opt.integrator)
    if integrator not in (int(mujoco.mjtIntegrator.mjINT_IMPLICIT),
                           int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)):
      raise ValueError("flex correction requires implicit or implicitfast")
    implicit_workspace_elements(int(model.nv), batch_size)
    self.nv, self.batch_size = int(model.nv), int(batch_size)
    self.timestep = float(model.opt.timestep)
    if not (self.timestep > 0 and self.timestep < float("inf")):
      raise ValueError("timestep must be finite and positive")
    import torch
    from mujoco_metal.smooth_solve import MetalFactorizedSolve
    self._torch, self._device = torch, torch.device("mps")
    self._preconditioner = MetalFactorizedSolve(
        self.nv, self.batch_size,
        general=integrator == int(mujoco.mjtIntegrator.mjINT_IMPLICIT))
    b,n = self.batch_size,self.nv
    self._ws = {name: torch.empty((b,n),dtype=torch.float32,device=self._device)
                for name in ("rhs","qacc","residual","direction","product")}
    self._matrix = torch.empty((b,n,n),dtype=torch.float32,device=self._device)
    self._status = torch.zeros(b,dtype=torch.int32,device=self._device)
    self._iterations = torch.zeros(b,dtype=torch.int32,device=self._device)
    self._enabled_all = torch.ones(b,dtype=torch.bool,device=self._device)

  def _check(self, value, name, shape, dtype=None):
    torch = self._torch
    if (not isinstance(value,torch.Tensor) or tuple(value.shape) != shape
        or value.dtype != (torch.float32 if dtype is None else dtype)
        or value.device.type != "mps" or not value.is_contiguous()):
      raise ValueError(f"{name} must be contiguous MPS with shape {shape} and matching dtype")

  def run_device(self, preconditioner, base_matrix, qfrc_total, qvel,
                 initial_qacc, operators, *, enabled=None):
    torch,w,b,n,h = self._torch,self._ws,self.batch_size,self.nv,self.timestep
    for name,value in (("preconditioner",preconditioner),("base_matrix",base_matrix)):
      self._check(value,name,(b,n,n))
    for name,value in (("qfrc_total",qfrc_total),("qvel",qvel),("initial_qacc",initial_qacc)):
      self._check(value,name,(b,n))
    for name in OPERATOR_NAMES:
      if name not in operators:
        raise ValueError(f"missing flex operator {name}")
      self._check(operators[name],name,(b,n,n))
    if enabled is None:
      enabled = self._enabled_all
    self._check(enabled,"enabled",(b,),torch.bool)
    self._status.zero_(); self._iterations.zero_()
    w["qacc"].copy_(initial_qacc)
    if n == 0:
      return {"qacc":w["qacc"],"status":self._status,"iterations":self._iterations}
    ki,kdi,kb,kdb = (operators[name] for name in OPERATOR_NAMES)
    self._matrix.copy_(base_matrix-h*h*ki-h*kdi+h*h*kb+h*kdb)
    def matvec(matrix,vector):
      return torch.bmm(matrix,vector.unsqueeze(-1)).squeeze(-1)
    w["rhs"].copy_(qfrc_total+h*matvec(ki-kb,qvel))
    w["residual"].copy_(w["rhs"]-matvec(self._matrix,w["qacc"]))
    finite = (torch.isfinite(self._matrix).all(dim=2).all(dim=1)
              & torch.isfinite(w["rhs"]).all(dim=1)
              & torch.isfinite(initial_qacc).all(dim=1))
    self._status.copy_(torch.where(enabled & ~finite,1,0).to(torch.int32))
    tolerance = 1e-10*(w["rhs"]*w["rhs"]).sum(dim=1)
    norm = (w["residual"]*w["residual"]).sum(dim=1)
    active = enabled & finite & (norm >= tolerance) & (norm >= 1e-15)
    self._preconditioner.factor_device(preconditioner)
    z,status = self._preconditioner.solve_factored_device(w["residual"])
    self._status.copy_(torch.where(active & (status != 0),status,self._status))
    active = active & (status == 0)
    w["direction"].copy_(z)
    rz = (w["residual"]*z).sum(dim=1)
    for _ in range(50):
      w["product"].copy_(matvec(self._matrix,w["direction"]))
      pap = (w["direction"]*w["product"]).sum(dim=1)
      usable = torch.isfinite(pap) & torch.isfinite(rz)
      self._status.copy_(torch.where(active & ~usable,4,self._status))
      active = active & usable & (torch.abs(pap) >= 1e-15)
      alpha = torch.where(active,rz/torch.where(active,pap,1),0)
      w["qacc"].add_(alpha[:,None]*w["direction"])
      w["residual"].sub_(alpha[:,None]*w["product"])
      self._iterations.add_(active.to(torch.int32))
      norm = (w["residual"]*w["residual"]).sum(dim=1)
      finite_iterate = torch.isfinite(norm) & torch.isfinite(w["qacc"]).all(dim=1)
      self._status.copy_(torch.where(active & ~finite_iterate,4,self._status))
      active = active & finite_iterate & (norm >= tolerance) & (norm >= 1e-15)
      z,status = self._preconditioner.solve_factored_device(w["residual"])
      self._status.copy_(torch.where(active & (status != 0),status,self._status))
      active = active & (status == 0)
      next_rz = (w["residual"]*z).sum(dim=1)
      beta = torch.where(active,next_rz/torch.clamp(rz,min=1e-15),0)
      w["direction"].copy_(torch.where(active[:,None],z+beta[:,None]*w["direction"],0))
      rz = next_rz
    # Inactive worlds and failed neighbors retain their input acceleration.
    retain = (~enabled | (self._status != 0))[:,None]
    w["qacc"].copy_(torch.where(retain,initial_qacc,w["qacc"]))
    return {"qacc":w["qacc"],"status":self._status,"iterations":self._iterations}
