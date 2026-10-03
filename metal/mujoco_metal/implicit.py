# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Bounded native implicitfast velocity solve for rigid joints."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "implicit.metal"
_FULL_SHADER = Path(__file__).parent / "shaders" / "implicit_full.metal"


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
  auto_derivative: bool = False


def implicitfast_auto_servable(model):
  """Whether native code assembles velocity derivatives (R06/D2).

  True when every smooth-force velocity dependence is natively covered:
  passive dampers (linear + polynomial, device diagonal), fixed-tendon
  damping tangents (device rank updates), and fluid forces (device
  derivative via MetalInertiaBoxFluid); no activation state, and only
  zero-derivative scalar motors. Tendon stiffness/limits are position-only.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("implicitfast lowering requires a MuJoCo MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("implicitfast lowering requires MuJoCo 3.10.0")
  if int(model.na) > 0:
    return False
  if int(model.nu) > 0:
    try:
      from mujoco_metal.actuation import ScalarMotorModel
      ScalarMotorModel.from_model(model)
    except ValueError:
      return False
  return True


def implicitfast_supports_automatic(model):
  """Whether velocity derivatives assemble automatically (R06b, no guards moved).

  True when the smooth generalized force has no velocity dependence outside
  the passive damper (linear + polynomial), fixed-tendon damping and fluid
  drag (inertia-box and per-geom ellipsoid models): no activation state,
  and only fixed-gain/no-bias/no-dynamics scalar motors (zero velocity
  derivative). Tendon stiffness/limits are position-only and contribute a
  zero block. This is an admission predicate only; assembly lives in
  :func:`implicit_derivative_reference`.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("implicitfast lowering requires a MuJoCo MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("implicitfast lowering requires MuJoCo 3.10.0")
  if int(model.na) > 0:
    return False
  if int(model.nu) > 0:
    try:
      from mujoco_metal.actuation import ScalarMotorModel
      ScalarMotorModel.from_model(model)
    except ValueError:
      return False
  return True


def implicit_derivative_reference(model, qpos, qvel):
  """Assemble ``d(qfrc_passive + qfrc_tendon + qfrc_actuator)/d(qvel)`` (R06b).

  Host float64 reference for the exact subset admitted by
  :func:`implicitfast_supports_automatic`: passive joint damping (linear +
  polynomial, pinned ``mju_polyForce`` derivative), fixed-tendon damping
  tangents (``-J' diag(c) J`` with the pinned polynomial tangent), fluid
  drag (R06d analytic Jacobian), and a zero block for admitted scalar
  motors. Activation-state dynamics and velocity-dependent actuator forces
  raise instead of returning a silently incomplete matrix. Disable flags
  are honored.
  """
  if not implicitfast_supports_automatic(model):
    raise ValueError("automatic implicitfast derivatives exclude this model")
  nv = int(model.nv)
  qpos = np.asarray(qpos, dtype=np.float64)
  qvel = np.asarray(qvel, dtype=np.float64)
  if qpos.ndim != 2 or qpos.shape[1] != int(model.nq) or qvel.shape != (qpos.shape[0], nv):
    raise ValueError(f"qpos/qvel must have shapes (B, {int(model.nq)}) and (B, {nv})")
  if not qpos.shape[0]:
    raise ValueError("batch must be nonempty")
  if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
    raise ValueError("qpos and qvel must be finite")
  from mujoco_metal.passive import PassiveForceModel
  from mujoco_metal.tendons import FixedTendonModel
  from mujoco_metal.fluid import fluid_derivative_reference
  deriv = np.zeros((qpos.shape[0], nv, nv), dtype=np.float64)
  if nv:
    damper = PassiveForceModel(model).damping_derivative(qvel)
    for b in range(qpos.shape[0]):
      deriv[b] -= np.diag(damper[b])
    if int(model.ntendon) > 0:
      _, tangent, _ = FixedTendonModel(model).run(qpos, qvel)
      deriv -= tangent
    if (model.opt.density != 0 or model.opt.viscosity != 0):
      deriv += fluid_derivative_reference(model, qpos, qvel)
  return deriv


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
  if int(model.na) > 0:
    raise ValueError("implicitfast stage excludes activation state (owned by milestone 015)")
  # Nonlinear velocity dependence (tendon damping, polynomial dampers)
  # Nonlinear velocity dependence (tendon damping, polynomial dampers, fluid)
  # is served by native automatic assembly when auto-servable; otherwise
  # an external full qDeriv (or constant damping) is required.
  if not external_derivative and (
      model.ntendon or np.any(np.asarray(model.dof_dampingpoly) != 0)
      or model.opt.density != 0 or model.opt.viscosity != 0):
    if not implicitfast_auto_servable(model):
      raise ValueError(
          "implicitfast nonlinear velocity derivatives require an external "
          "full qDeriv")
  if model.nu and not external_derivative:
    from mujoco_metal.actuation import ScalarMotorModel
    ScalarMotorModel.from_model(model)
  if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
    raise ValueError("implicitfast timestep must be finite and positive")
  # Native automatic assembly engages where the baseline is inexact
  # (tendon damping, polynomial dampers, or fluid drag present).
  auto = (implicitfast_auto_servable(model)
          and (int(model.ntendon) > 0
               or bool(np.any(np.asarray(model.dof_dampingpoly) != 0))
               or bool(model.opt.density != 0 or model.opt.viscosity != 0)))
  return ImplicitFastDescriptor(
      nv=int(model.nv), timestep=float(model.opt.timestep),
      disableflags=int(model.opt.disableflags),
      dof_damping=_frozen(model.dof_damping, np.float32),
      auto_derivative=bool(auto),
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

  def run_device_auto(self, mass_matrix, qfrc_smooth, passive_diag, tendon_tangent=None, fluid_jacobian=None):
    """Solve with natively assembled velocity derivatives (R06/D2).

    ``passive_diag`` is the device per-dof damping tangent magnitudes
    ``(b, nv)`` (passive stage ``deriv`` output, already damper-gated) and
    ``tendon_tangent`` the device ``(b, nv, nv)`` tendon damping tangent
    (or None without fixed tendons). If fluid is present, ``fluid_jacobian``
    is the ``(b, nv, nv)`` fluid generalized velocity derivative.
    The assembled derivative is ``-(diag(passive) + tendon) + fluid``,
    matching :func:`implicit_derivative_reference`.
    """
    torch, nv, b = self._torch, self.descriptor.nv, self.batch_size
    if not self.descriptor.auto_derivative:
      raise ValueError("this program was lowered without automatic derivatives")
    if passive_diag is None:
      passive_diag = torch.zeros((b, nv), dtype=torch.float32, device=self._device)
    for name, tensor, shape in (
        ("passive_diag", passive_diag, (b, nv)),
    ):
      if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "mps"
              or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape
              or not tensor.is_contiguous()):
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    if tendon_tangent is None:
      combined = -torch.diag_embed(passive_diag)
    else:
      if (not isinstance(tendon_tangent, torch.Tensor) or tendon_tangent.device.type != "mps"
              or tendon_tangent.dtype != torch.float32
              or tuple(tendon_tangent.shape) != (b, nv, nv) or not tendon_tangent.is_contiguous()):
        raise ValueError(f"tendon_tangent must be contiguous float32 MPS with shape ({b}, {nv}, {nv})")
      combined = -(torch.diag_embed(passive_diag) + tendon_tangent)
    if fluid_jacobian is not None:
      if (not isinstance(fluid_jacobian, torch.Tensor) or fluid_jacobian.device.type != "mps"
              or fluid_jacobian.dtype != torch.float32
              or tuple(fluid_jacobian.shape) != (b, nv, nv) or not fluid_jacobian.is_contiguous()):
        raise ValueError(f"fluid_jacobian must be contiguous float32 MPS with shape ({b}, {nv}, {nv})")
      combined = combined + fluid_jacobian
    return self.run_device(mass_matrix, qfrc_smooth, combined)

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


@dataclass(frozen=True)
class ImplicitDescriptor:
  """Source-derived constants for the general implicit stage."""
  nv: int
  timestep: float
  disableflags: int
  dof_damping: np.ndarray
  auto_derivative: bool = False


def lower_implicit(model, *, external_derivative=False):
  """Validate the subset for the general implicit velocity solve."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("implicit lowering requires a MuJoCo MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("implicit lowering requires MuJoCo 3.10.0")
  if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICIT):
    raise ValueError("implicit stage requires MuJoCo's implicit integrator")
  if not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
    raise ValueError("implicit stage requires contact explicitly disabled")
  if model.nv > 32:
    raise ValueError("implicit stage currently bounds nv to 32")
  if model.neq or np.any(model.jnt_limited) or np.any(model.dof_frictionloss):
    raise ValueError("implicit stage currently excludes joint constraints")
  if model.nflex or model.nflexvert or model.nflexelem:
    raise ValueError("implicit stage currently excludes flex dynamics")
  if model.nplugin or model.nmocap:
    raise ValueError("implicit stage excludes MuJoCo plugins and mocap bodies")
  if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
    raise ValueError("implicit timestep must be finite and positive")
  auto = True
  return ImplicitDescriptor(
      nv=int(model.nv), timestep=float(model.opt.timestep),
      disableflags=int(model.opt.disableflags),
      dof_damping=_frozen(model.dof_damping, np.float32),
      auto_derivative=bool(auto),
  )


def implicit_oracle(model, mass_matrix, qfrc_smooth,
                    force_velocity_derivative=None):
  """CPU mathematical reference for nonsymmetric H qacc = qfrc_smooth."""
  d = lower_implicit(model, external_derivative=force_velocity_derivative is not None)
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
    raise ValueError("implicit inputs must be finite")
  hessian = mass - d.timestep * deriv
  acceleration = np.zeros_like(force)
  status = np.zeros(batch, dtype=np.int32)
  for world in range(batch):
    try:
      acceleration[world] = np.linalg.solve(hessian[world], force[world])
    except np.linalg.LinAlgError:
      status[world] = 2
  return {"effective_mass": hessian, "qacc": acceleration, "status": status}


class ImplicitProgram:
  """MPS assembly of nonsymmetric ``M - h*qDeriv`` followed by a native dense LU solve."""

  def __init__(self, model, batch_size=1, *, external_derivative=False):
    self.descriptor = lower_implicit(model, external_derivative=external_derivative)
    self._requires_derivative = bool(external_derivative)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = int(batch_size)
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("implicit stage requires PyTorch MPS compile_shader")
    self._torch, self._device = torch, torch.device("mps")
    self._empty_input = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._library = torch.mps.compile_shader(_FULL_SHADER.read_text())
    self._assemble = self._library.assemble_nonsymmetric_implicit_mass
    from mujoco_metal.smooth_solve import MetalGeneralDenseSolve
    self._solver = MetalGeneralDenseSolve(self.descriptor.nv, self.batch_size)
    self._mass_storage = torch.empty(
        max(self.batch_size * self.descriptor.nv * self.descriptor.nv, 1),
        dtype=torch.float32, device=self._device,
    )
    self._mass = self._mass_storage[:self.batch_size * self.descriptor.nv * self.descriptor.nv].reshape(
        self.batch_size, self.descriptor.nv, self.descriptor.nv
    )
    self._baseline = torch.as_tensor(
        np.array(_baseline_derivative(self.descriptor), copy=True),
        dtype=torch.float32, device=self._device,
    ).contiguous()
    self._dims = torch.tensor([self.batch_size, self.descriptor.nv], dtype=torch.int32, device=self._device)
    self._timestep = torch.tensor([self.descriptor.timestep], dtype=torch.float32, device=self._device)

  def run_device_auto(
      self,
      mass_matrix,
      qfrc_smooth,
      passive_diag,
      tendon_tangent=None,
      fluid_jacobian=None,
      bias_derivative=None,
  ):
    torch, nv, b = self._torch, self.descriptor.nv, self.batch_size
    if passive_diag is None:
      passive_diag = torch.zeros((b, nv), dtype=torch.float32, device=self._device)
    combined = -torch.diag_embed(passive_diag)
    if tendon_tangent is not None:
      combined = combined - tendon_tangent
    if fluid_jacobian is not None:
      combined = combined + fluid_jacobian
    if bias_derivative is not None:
      combined = combined + bias_derivative
    return self.run_device(mass_matrix, qfrc_smooth, combined)

  def run_device(self, mass_matrix, qfrc_smooth, force_velocity_derivative=None):
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

