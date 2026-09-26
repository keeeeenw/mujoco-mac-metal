# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""MuJoCo 3.10 eligible free-body midpoint correction for implicitfast."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mujoco_metal.implicit import lower_implicitfast

_SHADER = Path(__file__).parent / "shaders" / "implicit_midpoint.metal"
_MAX_NEWTON = 100
_MAX_LINESEARCH = 20
_NONCONVERGENCE = 20
_NONFINITE = 21


def _frozen(value, dtype):
  array = np.asarray(value, dtype=dtype, order="C")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class FreeBodyMidpointDescriptor:
  """Eligible standalone free bodies from MuJoCo's implicitfast source guards."""
  nq: int
  nv: int
  nbody: int
  nfree: int
  timestep: float
  gravity: np.ndarray
  gravity_enabled: bool
  dofadr: np.ndarray
  bodyid: np.ndarray
  mass: np.ndarray
  inertia: np.ndarray
  ipos: np.ndarray
  iquat: np.ndarray
  aligned: np.ndarray


def lower_free_body_midpoints(model):
  """Lower only joints satisfying MuJoCo 3.10 midpoint_eligible guards.

  Eligible means: implicitfast; inverse-discrete dynamics disabled; zero
  density/viscosity; sleep disabled; free joint; exactly six DOFs in its tree;
  and exact equality between that body's subtree mass and its own body mass.
  The implicit profile also guarantees no constraints, so every such tree is
  unconstrained. Other free joints deliberately use ordinary implicitfast
  integration, as MuJoCo does.
  """
  lower_implicitfast(model)
  if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
    raise ValueError("free-body midpoint stage requires sleep mode disabled")
  eligible = []
  inverse_discrete = bool(
      int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  )
  free = int(mujoco.mjtJoint.mjJNT_FREE)
  for jid in range(model.njnt):
    if int(model.jnt_type[jid]) != free or inverse_discrete:
      continue
    body = int(model.jnt_bodyid[jid])
    dof = int(model.jnt_dofadr[jid])
    tree = int(model.dof_treeid[dof])
    if int(model.tree_dofnum[tree]) != 6:
      continue
    if float(model.body_subtreemass[body]) != float(model.body_mass[body]):
      continue
    eligible.append((jid, body, dof))
  bodies = [body for _, body, _ in eligible]
  gravity_disabled = bool(
      int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  )
  return FreeBodyMidpointDescriptor(
      nq=int(model.nq), nv=int(model.nv), nbody=int(model.nbody),
      nfree=len(eligible), timestep=float(model.opt.timestep),
      gravity=_frozen(model.opt.gravity, np.float32),
      gravity_enabled=not gravity_disabled,
      dofadr=_frozen([dof for _, _, dof in eligible], np.int32),
      bodyid=_frozen(bodies, np.int32),
      mass=_frozen(model.body_mass[bodies], np.float32),
      inertia=_frozen(model.body_inertia[bodies], np.float32).reshape(len(bodies), 3),
      ipos=_frozen(model.body_ipos[bodies], np.float32).reshape(len(bodies), 3),
      iquat=_frozen(model.body_iquat[bodies], np.float32).reshape(len(bodies), 4),
      aligned=_frozen(
          np.all(np.asarray(model.body_ipos)[bodies] == 0, axis=1), np.uint8
      ),
  )


def _rotate(q, v):
  u = q[1:]
  return v + 2*np.cross(u, np.cross(u, v) + q[0]*v)


def _quat_mul(a, b):
  return np.r_[a[0]*b[0]-np.dot(a[1:], b[1:]),
               a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def _residual(inertia, wmid, w, tau, i2h):
  return i2h*inertia*(wmid-w) + np.cross(wmid, inertia*wmid) - tau


def _midpoint_newton(inertia, w, tau, h, *, max_iterations=_MAX_NEWTON):
  """Double-precision transcription of engine_forward.c midpointNewton."""
  i2h = 2.0/h
  dI = np.array([inertia[2]-inertia[1], inertia[0]-inertia[2], inertia[1]-inertia[0]])
  wmid = w.copy()
  for iteration in range(max_iterations):
    Iw = inertia*wmid
    f = _residual(inertia, wmid, w, tau, i2h)
    fnorm = float(np.linalg.norm(f))
    if fnorm < 1e-13*(1+i2h*float(np.linalg.norm(Iw))):
      return wmid, iteration, True
    jac = np.array([
        [i2h*inertia[0], wmid[2]*dI[0], wmid[1]*dI[0]],
        [wmid[2]*dI[1], i2h*inertia[1], wmid[0]*dI[1]],
        [wmid[1]*dI[2], wmid[0]*dI[2], i2h*inertia[2]],
    ])
    try:
      delta = np.linalg.solve(jac, -f)
    except np.linalg.LinAlgError:
      return wmid, iteration, False
    step = 1.0
    for _ in range(_MAX_LINESEARCH):
      candidate = wmid + step*delta
      if np.linalg.norm(_residual(inertia, candidate, w, tau, i2h)) < fnorm:
        wmid = candidate
        break
      step *= .5
  return wmid, max_iterations, False


def _one_free_midpoint(d, old, qforce, bodyquat, slot, max_iterations=_MAX_NEWTON):
  h = d.timestep
  dof = int(d.dofadr[slot])
  mass, inertia = float(d.mass[slot]), d.inertia[slot].astype(np.float64)
  ipos, iquat = d.ipos[slot].astype(np.float64), d.iquat[slot].astype(np.float64)
  bodyquat = np.asarray(bodyquat, dtype=np.float64)
  iquat_neg = iquat.copy(); iquat_neg[1:] *= -1
  w = _rotate(iquat_neg, old[dof+3:dof+6])
  tau = _rotate(iquat_neg, qforce[dof+3:dof+6])
  aligned = bool(d.aligned[slot])
  if aligned:
    tau_com = tau
    rot_x2i = force = r_com = None
  else:
    xquat_neg = bodyquat.copy(); xquat_neg[1:] *= -1
    rot_x2i = _quat_mul(iquat_neg, xquat_neg)
    force = _rotate(rot_x2i, qforce[dof:dof+3])
    r_com = _rotate(iquat_neg, ipos)
    tau_com = tau - np.cross(r_com, force)
  wmid, iterations, converged = _midpoint_newton(
      inertia, w, tau_com, h, max_iterations=max_iterations
  )
  if not converged:
    return None, iterations, False
  wnew = 2*wmid-w
  wnew_body, wmid_body = _rotate(iquat, wnew), _rotate(iquat, wmid)
  new, mid = old[dof:dof+6].copy(), old[dof:dof+6].copy()
  new[3:] = wnew_body
  mid[3:] = .5*(old[dof+3:dof+6]+wnew_body)
  if not aligned:
    v = _rotate(rot_x2i, old[dof:dof+3])
    vcom = v + np.cross(w, r_com)
    i2h = 2.0/h
    b = force/mass+i2h*vcom
    if d.gravity_enabled:
      b += _rotate(rot_x2i, d.gravity.astype(np.float64))
    norm2 = float(np.dot(wmid, wmid))
    denom = i2h*i2h+norm2
    vcommid = (i2h*b+(float(np.dot(wmid,b))/i2h)*wmid-np.cross(wmid,b))/denom
    vmid = vcommid-np.cross(wmid,r_com)
    vnew = 2*vmid-v
    wnorm = float(np.linalg.norm(wmid_body))
    axis = wmid_body/wnorm if wnorm > 0 else np.zeros(3)
    angle = h*wnorm
    qrot = np.r_[np.cos(.5*angle), axis*np.sin(.5*angle)]
    xquat_new = _quat_mul(bodyquat, qrot)
    new[:3] = _rotate(xquat_new, _rotate(iquat, vnew))
    mid[:3] = .5*(old[dof:dof+3]+new[:3])
  return (new, mid), iterations, True


def free_midpoint_oracle(model, qvel, effective_acceleration, physical_qacc,
                         qfrc_total, body_quat, *, max_iterations=_MAX_NEWTON):
  """CPU reference for midpoint overrides and ordinary fallback DOFs.

  qfrc_total is current smooth generalized force with qfrc_bias added back,
  matching MuJoCo's midpoint helper input. qacc starts with physical forward
  acceleration; MuJoCo overwrites it only on eligible midpoint DOFs.
  """
  d = lower_free_body_midpoints(model)
  old = np.asarray(qvel, dtype=np.float64)
  effective = np.asarray(effective_acceleration, dtype=np.float64)
  physical = np.asarray(physical_qacc, dtype=np.float64)
  force = np.asarray(qfrc_total, dtype=np.float64)
  quat = np.asarray(body_quat, dtype=np.float64)
  if old.ndim == 1:
    old = old[None]
  if effective.ndim == 1:
    effective = effective[None]
  if physical.ndim == 1:
    physical = physical[None]
  if force.ndim == 1:
    force = force[None]
  if quat.ndim == 2:
    quat = quat[None]
  batch = old.shape[0]
  expected = ((batch, d.nv), (batch, d.nv), (batch, d.nv), (batch, d.nv), (batch, d.nbody, 4))
  if tuple(x.shape for x in (old, effective, physical, force, quat)) != expected:
    raise ValueError("free midpoint input shapes do not match the model and batch")
  if not all(np.all(np.isfinite(x)) for x in (old, effective, physical, force, quat)):
    raise ValueError("free midpoint inputs must be finite")
  next_vel = old + d.timestep*effective
  position_velocity = next_vel.copy()
  reported_qacc = physical.copy()
  status = np.zeros(batch, dtype=np.int32)
  for world in range(batch):
    for slot in range(d.nfree):
      dof = int(d.dofadr[slot])
      body = int(d.bodyid[slot])
      result, _, converged = _one_free_midpoint(
          d, old[world], force[world], quat[world, body], slot,
          max_iterations=max_iterations,
      )
      if not converged:
        status[world] = _NONCONVERGENCE
        next_vel[world] = old[world]
        position_velocity[world] = old[world]
        reported_qacc[world] = physical[world]
        break
      new, midpoint = result
      # MuJoCo overwrites rotational DOFs for aligned bodies and all six
      # free-joint DOFs when the inertial COM is offset.
      start = 3 if d.aligned[slot] else 0
      next_vel[world, dof+start:dof+6] = new[start:]
      position_velocity[world, dof+start:dof+6] = midpoint[start:]
      reported_qacc[world, dof+start:dof+6] = (
          new[start:]-old[world, dof+start:dof+6]
      )/d.timestep
  return {"qvel_next": next_vel, "position_velocity": position_velocity,
          "qacc": reported_qacc, "status": status}


class FreeBodyMidpointProgram:
  """Batched MPS correction for MuJoCo-eligible implicitfast free bodies.

  Returned buffers are borrowed. qvel_next is the true post-step velocity;
  position_velocity is the midpoint vector for MuJoCo pose integration;
  qacc follows MuJoCo's convention for the final MjData qacc.
  """
  def __init__(self, model, batch_size=1):
    self.descriptor = lower_free_body_midpoints(model)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = int(batch_size)
    if self.batch_size > 2**31-1 or self.batch_size*max(self.descriptor.nv, self.descriptor.nbody*4, 1) > 2**31-1:
      raise ValueError("batch dimensions exceed shader indexing capacity")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("free-body midpoint stage requires PyTorch MPS compile_shader")
    self._torch, self._device = torch, torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.free_body_midpoint
    d = self.descriptor
    self._arrays = {}
    for name, dtype in (
        ("dofadr", np.int32), ("bodyid", np.int32), ("mass", np.float32),
        ("inertia", np.float32), ("ipos", np.float32), ("iquat", np.float32),
        ("aligned", np.uint8),
    ):
      array = np.array(getattr(d, name), dtype=dtype, copy=True)
      if array.size == 0:
        array = np.zeros(1, dtype=dtype)
      self._arrays[name] = torch.as_tensor(array, device=self._device)
    self._gravity = torch.as_tensor(np.array(d.gravity, copy=True), dtype=torch.float32, device=self._device)
    self._dims = torch.tensor([d.nv, d.nbody, d.nfree, self.batch_size, int(d.gravity_enabled)], dtype=torch.int32, device=self._device)
    self._timestep = torch.tensor([d.timestep], dtype=torch.float32, device=self._device)
    self._outputs = {
        "qvel_next": torch.empty((self.batch_size, d.nv), dtype=torch.float32, device=self._device),
        "position_velocity": torch.empty((self.batch_size, d.nv), dtype=torch.float32, device=self._device),
        "qacc": torch.empty((self.batch_size, d.nv), dtype=torch.float32, device=self._device),
        "status": torch.empty((self.batch_size,), dtype=torch.int32, device=self._device),
    }
    self._empty = torch.zeros(1, dtype=torch.float32, device=self._device)

  def run_device(self, qvel, effective_acceleration, physical_qacc,
                 qfrc_total, body_quat):
    """Return qvel_next, midpoint position_velocity, reported qacc, and status."""
    torch, d, b = self._torch, self.descriptor, self.batch_size
    if not np.isfinite(d.timestep) or d.timestep <= 0:
      raise ValueError("free midpoint timestep must be finite and positive")
    inputs = (
        ("qvel", qvel, (b, d.nv)),
        ("effective_acceleration", effective_acceleration, (b, d.nv)),
        ("physical_qacc", physical_qacc, (b, d.nv)),
        ("qfrc_total", qfrc_total, (b, d.nv)),
        ("body_quat", body_quat, (b, d.nbody, 4)),
    )
    for name, tensor, shape in inputs:
      if not isinstance(tensor, torch.Tensor) or tensor.device.type != "mps" or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    nv = d.nv
    state_buffers = [x.reshape(-1) if x.numel() else self._empty for x in
                     (qvel, effective_acceleration, physical_qacc, qfrc_total, body_quat)]
    outputs = [self._outputs[n].reshape(-1) if nv else self._empty for n in
               ("qvel_next", "position_velocity", "qacc")]
    self._kernel(
        *state_buffers, self._arrays["dofadr"], self._arrays["bodyid"],
        self._arrays["mass"], self._arrays["inertia"], self._arrays["ipos"],
        self._arrays["iquat"], self._arrays["aligned"], self._gravity,
        *outputs, self._outputs["status"], self._dims, self._timestep,
        threads=(b,), group_size=(1,),
    )
    return self._outputs
