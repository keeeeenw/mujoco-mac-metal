# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Bounded native implicitfast velocity solve for rigid joints."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "implicit.metal"


def _frozen(value, dtype):
  array = np.asarray(value, dtype=dtype, order="C")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ImplicitFastDescriptor:
  """Source-derived constants for the explicitly bounded implicitfast stage."""
  nv: int
  timestep: float
  disableflags: int
  dof_damping: np.ndarray


def lower_implicitfast(model, *, external_derivative=False):
  """Validate the subset whose velocity derivative and advance are exact here.

  Free bodies are handled by ``FreeBodyMidpointProgram`` after this solve;
  that stage applies MuJoCo's eligibility test and leaves all other DOFs on the
  ordinary implicitfast velocity path.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("implicitfast lowering requires a MuJoCo MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("implicitfast lowering requires MuJoCo 3.10.0")
  if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
    raise ValueError("implicitfast stage requires MuJoCo's implicitfast integrator")
  if not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
    raise ValueError("implicitfast stage requires contact explicitly disabled")
  if model.nv > 24:
    raise ValueError("implicitfast stage currently bounds nv to 24")
  if model.neq or np.any(model.jnt_limited) or np.any(model.dof_frictionloss):
    raise ValueError("implicitfast stage currently excludes joint constraints")
  if model.nflex or model.nflexvert or model.nflexelem:
    raise ValueError("implicitfast stage currently excludes flex dynamics")
  if model.nplugin or model.nmocap:
    raise ValueError("implicitfast stage excludes MuJoCo plugins and mocap bodies")
  if not external_derivative and model.ntendon:
    raise ValueError("implicitfast tendon derivatives require an external full qDeriv")
  if not external_derivative and (model.opt.density != 0 or model.opt.viscosity != 0):
    raise ValueError("implicitfast fluid velocity derivatives are unsupported")
  if not external_derivative and np.any(np.asarray(model.dof_dampingpoly) != 0):
    raise ValueError("implicitfast stage currently supports constant DOF damping only")
  if model.nu and not external_derivative:
    from mujoco_metal.actuation import ScalarMotorModel
    ScalarMotorModel.from_model(model)
  if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
    raise ValueError("implicitfast timestep must be finite and positive")
  return ImplicitFastDescriptor(
      nv=int(model.nv), timestep=float(model.opt.timestep),
      disableflags=int(model.opt.disableflags),
      dof_damping=_frozen(model.dof_damping, np.float32),
  )


def _baseline_derivative(descriptor):
  d = descriptor
  damping = d.dof_damping.copy()
  if d.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER):
    damping.fill(0)
  return -np.diag(damping).astype(np.float32)


def implicitfast_oracle(model, mass_matrix, qfrc_smooth,
                        force_velocity_derivative=None):
  """CPU mathematical reference for ``H qacc = qfrc_smooth``.

  The optional derivative is the full MuJoCo 3.10 ``qDeriv`` block
  ``d(qfrc_passive + qfrc_actuator)/d(qvel)``; the bias derivative is omitted
  exactly as in ``mjd_smooth_vel(..., flg_bias=0)``. If omitted, this bounded
  helper uses only constant DOF damping from the model.
  """
  d = lower_implicitfast(model, external_derivative=force_velocity_derivative is not None)
  mass = np.asarray(mass_matrix, dtype=np.float64)
  force = np.asarray(qfrc_smooth, dtype=np.float64)
  if mass.ndim == 2:
    mass = mass[None]
  if force.ndim == 1:
    force = force[None]
  batch = force.shape[0] if force.ndim == 2 else 0
  if batch < 1 or force.shape != (batch, d.nv) or mass.shape != (batch, d.nv, d.nv):
    raise ValueError(f"mass_matrix and qfrc_smooth must have shapes (B, {d.nv}, {d.nv}) and (B, {d.nv})")
  if force_velocity_derivative is None:
    deriv = np.broadcast_to(_baseline_derivative(d), (batch, d.nv, d.nv)).astype(np.float64)
  else:
    deriv = np.asarray(force_velocity_derivative, dtype=np.float64)
    if deriv.ndim == 2:
      deriv = np.broadcast_to(deriv, (batch, d.nv, d.nv))
    if deriv.shape != (batch, d.nv, d.nv):
      raise ValueError(f"force_velocity_derivative must have shape ({batch}, {d.nv}, {d.nv})")
  if not all(np.all(np.isfinite(v)) for v in (mass, force, deriv)):
    raise ValueError("implicitfast inputs must be finite")
  # mj_implicitfast gathers the sparse qDeriv block into the lower-triangular
  # mass pattern; that lower triangle is authoritative for factorization.
  lower = np.tril(deriv)
  deriv_symmetric = lower + np.swapaxes(np.tril(deriv, -1), -1, -2)
  hessian = mass - d.timestep*deriv_symmetric
  acceleration = np.zeros_like(force)
  status = np.zeros(batch, dtype=np.int32)
  for world in range(batch):
    try:
      acceleration[world] = np.linalg.solve(hessian[world], force[world])
    except np.linalg.LinAlgError:
      status[world] = 2
  return {"effective_mass": hessian, "qacc": acceleration, "status": status}


class ImplicitFastProgram:
  """MPS assembly of ``M - h*qDeriv`` followed by a native dense SPD solve.

  Optional ``force_velocity_derivative`` input to ``run_device`` must be the
  complete current-state derivative of passive plus actuator generalized force
  with respect to generalized velocity, with MuJoCo's bias derivative omitted.
  As in ``mju_gather(qH, qDeriv, mapD2M)``, its lower triangle is authoritative
  and is mirrored before the dense symmetric solve.
  Without it the constructor's supported baseline is constant DOF damping only.
  """
  def __init__(self, model, batch_size=1, *, external_derivative=False):
    self.descriptor = lower_implicitfast(model, external_derivative=external_derivative)
    self._requires_derivative = bool(external_derivative)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = int(batch_size)
    if self.batch_size > 2**31-1 or self.batch_size*max(self.descriptor.nv**2, 1) > 2**32-1:
      raise ValueError("batch dimensions exceed shader indexing capacity")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("implicitfast stage requires PyTorch MPS compile_shader")
    self._torch, self._device = torch, torch.device("mps")
    self._empty_input = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._assemble = self._library.assemble_implicit_mass
    from mujoco_metal.smooth_solve import MetalDenseSolve
    self._solver = MetalDenseSolve(self.descriptor.nv, self.batch_size)
    self._mass_storage = torch.empty(
        max(self.batch_size*self.descriptor.nv*self.descriptor.nv, 1),
        dtype=torch.float32, device=self._device,
    )
    self._mass = self._mass_storage[:self.batch_size*self.descriptor.nv*self.descriptor.nv].reshape(
        self.batch_size, self.descriptor.nv, self.descriptor.nv
    )
    self._baseline = torch.as_tensor(
        np.array(_baseline_derivative(self.descriptor), copy=True),
        dtype=torch.float32, device=self._device,
    ).contiguous()
    self._dims = torch.tensor([self.batch_size, self.descriptor.nv], dtype=torch.int32, device=self._device)
    self._timestep = torch.tensor([self.descriptor.timestep], dtype=torch.float32, device=self._device)

  def run_device(self, mass_matrix, qfrc_smooth, force_velocity_derivative=None):
    """Return borrowed ``effective_mass``, ``qacc`` and per-world status tensors."""
    torch, nv, b = self._torch, self.descriptor.nv, self.batch_size
    for name, tensor, shape in (
        ("mass_matrix", mass_matrix, (b, nv, nv)),
        ("qfrc_smooth", qfrc_smooth, (b, nv)),
    ):
      if not isinstance(tensor, torch.Tensor) or tensor.device.type != "mps" or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    derivative = self._baseline.expand(b, nv, nv).contiguous()
    if force_velocity_derivative is not None:
      derivative = force_velocity_derivative
      if not isinstance(derivative, torch.Tensor) or derivative.device.type != "mps" or derivative.dtype != torch.float32 or tuple(derivative.shape) != (b, nv, nv) or not derivative.is_contiguous():
        raise ValueError(f"force_velocity_derivative must be contiguous float32 MPS with shape ({b}, {nv}, {nv})")
    elif self._requires_derivative:
      raise ValueError("this program requires the caller's full force velocity derivative")
    if nv == 0:
      mass_buffer = derivative_buffer = self._empty_input
    else:
      mass_buffer, derivative_buffer = mass_matrix.reshape(-1), derivative.reshape(-1)
    self._assemble(mass_buffer, derivative_buffer, self._mass_storage, self._dims, self._timestep, threads=(b*max(nv*nv, 1),), group_size=(1,))
    acceleration, status = self._solver.run_device(self._mass, qfrc_smooth)
    return {"effective_mass": self._mass, "qacc": acceleration, "status": status}
