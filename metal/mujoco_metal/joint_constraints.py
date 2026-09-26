# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Bounded scalar-joint constraint lowering and a native coupled solve."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "joint_constraints.metal"
_MINVAL = 1e-15
_MAX_NV = 24
_MAX_ROWS = 32
_MAX_ITERATIONS = 512
_TOLERANCE = 2e-6
_DISABLE_CONSTRAINT = int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
_DISABLE_EQUALITY = int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
_DISABLE_FRICTION = int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
_DISABLE_LIMIT = int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)


def _frozen(value, dtype):
  array = np.asarray(value, dtype=dtype, order="C")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class JointConstraintDescriptor:
  """Readonly MuJoCo 3.10 constants for scalar limits/friction/joint equalities."""

  nq: int
  nv: int
  njnt: int
  neq: int
  nrow: int
  timestep: float
  disableflags: int
  refsafe: bool
  joint_type: np.ndarray
  joint_qposadr: np.ndarray
  qpos0: np.ndarray
  joint_dofadr: np.ndarray
  joint_limited: np.ndarray
  joint_range: np.ndarray
  joint_margin: np.ndarray
  joint_solref: np.ndarray
  joint_solimp: np.ndarray
  dof_frictionloss: np.ndarray
  dof_invweight0: np.ndarray
  dof_solref: np.ndarray
  dof_solimp: np.ndarray
  equality_type: np.ndarray
  equality_obj1: np.ndarray
  equality_obj2: np.ndarray
  equality_data: np.ndarray
  equality_solref: np.ndarray
  equality_solimp: np.ndarray
  equality_active0: np.ndarray


def lower_joint_constraints(model):
  """Snapshot metadata, rejecting constraint families this stage cannot solve."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("joint constraint lowering requires a MuJoCo MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("joint constraint lowering requires MuJoCo 3.10.0")
  if model.nv > _MAX_NV:
    raise ValueError(f"joint constraint stage currently bounds nv to {_MAX_NV}")
  if model.neq and np.any(np.asarray(model.eq_type) != _EQ_JOINT):
    bad = int(np.flatnonzero(np.asarray(model.eq_type) != _EQ_JOINT)[0])
    raise ValueError(f"equality {bad}: only polynomial joint equality is supported")
  scalar_types = (_HINGE, _SLIDE)
  limited = np.asarray(model.jnt_limited, dtype=bool)
  for jid in range(model.njnt):
    if limited[jid] and int(model.jnt_type[jid]) not in scalar_types:
      raise ValueError(f"joint {jid}: only scalar hinge/slide limits are supported")
  for eid in range(model.neq):
    j1, j2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
    if not 0 <= j1 < model.njnt or int(model.jnt_type[j1]) not in scalar_types:
      raise ValueError(f"equality {eid}: object 1 must be a scalar hinge/slide joint")
    if j2 >= 0 and (j2 >= model.njnt or int(model.jnt_type[j2]) not in scalar_types):
      raise ValueError(f"equality {eid}: object 2 must be a scalar hinge/slide joint")
  if model.ntendon and (np.any(model.tendon_limited) or np.any(model.tendon_frictionloss)):
    raise ValueError("tendon limits and frictionloss are unsupported")
  if model.npair or model.nexclude:
    raise ValueError("explicit geom pairs and exclusions are outside the joint constraint stage")
  # Reject potentially active contacts unless the model explicitly disables
  # them; an independent joint stage must not imply full constraint coverage.
  if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)):
    for a in range(model.ngeom):
      for b in range(a + 1, model.ngeom):
        if (int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) or (
            int(model.geom_contype[b]) & int(model.geom_conaffinity[a])
        ):
          raise ValueError("potential geom contacts are unsupported by the joint constraint stage")
  nrow = int(model.neq + model.nv + 2*model.njnt)
  if nrow > _MAX_ROWS:
    raise ValueError(f"joint constraint stage currently bounds candidate rows to {_MAX_ROWS}")
  if int(model.opt.integrator) not in (
      int(mujoco.mjtIntegrator.mjINT_EULER),
      int(mujoco.mjtIntegrator.mjINT_IMPLICIT),
      int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
  ):
    raise ValueError("joint constraint stage does not support RK4 constraint timing")
  return JointConstraintDescriptor(
      nq=int(model.nq), nv=int(model.nv), njnt=int(model.njnt),
      neq=int(model.neq), nrow=nrow, timestep=float(model.opt.timestep),
      disableflags=int(model.opt.disableflags),
      refsafe=not bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)),
      joint_type=_frozen(model.jnt_type, np.int32),
      joint_qposadr=_frozen(model.jnt_qposadr, np.int32),
      qpos0=_frozen(model.qpos0, np.float32),
      joint_dofadr=_frozen(model.jnt_dofadr, np.int32),
      joint_limited=_frozen(model.jnt_limited, np.uint8),
      joint_range=_frozen(model.jnt_range.reshape(model.njnt, 2), np.float32),
      joint_margin=_frozen(model.jnt_margin, np.float32),
      joint_solref=_frozen(model.jnt_solref.reshape(model.njnt, 2), np.float32),
      joint_solimp=_frozen(model.jnt_solimp.reshape(model.njnt, 5), np.float32),
      dof_frictionloss=_frozen(model.dof_frictionloss, np.float32),
      dof_invweight0=_frozen(model.dof_invweight0, np.float32),
      dof_solref=_frozen(model.dof_solref.reshape(model.nv, 2), np.float32),
      dof_solimp=_frozen(model.dof_solimp.reshape(model.nv, 5), np.float32),
      equality_type=_frozen(model.eq_type, np.int32),
      equality_obj1=_frozen(model.eq_obj1id, np.int32),
      equality_obj2=_frozen(model.eq_obj2id, np.int32),
      equality_data=_frozen(np.asarray(model.eq_data).reshape(model.neq, 11), np.float32),
      equality_solref=_frozen(model.eq_solref.reshape(model.neq, 2), np.float32),
      equality_solimp=_frozen(model.eq_solimp.reshape(model.neq, 5), np.float32),
      equality_active0=_frozen(model.eq_active0, np.uint8),
  )


def _impedance(solimp, pos, margin):
  d0, d1, width, midpoint, power = map(float, solimp)
  if d0 == d1 or width <= _MINVAL:
    return .5*(d0+d1)
  x = abs((pos-margin)/width)
  if x >= 1:
    return d1
  if x <= 0:
    return d0
  if power == 1:
    y = x
  elif x <= midpoint:
    y = x**power / midpoint**(power-1)
  else:
    y = 1 - (1-x)**power / (1-midpoint)**(power-1)
  return d0 + y*(d1-d0)


def joint_constraint_oracle(model, qpos, qvel, mass_matrix, qfrc_smooth,
                            eq_active=None, *, max_iterations=_MAX_ITERATIONS,
                            tolerance=_TOLERANCE):
  """Float64 projected coupled solve; reference only, never used by MPS stage."""
  d = lower_joint_constraints(model)
  qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
  mass = np.asarray(mass_matrix, dtype=np.float64)
  force = np.asarray(qfrc_smooth, dtype=np.float64)
  if qpos.ndim != 2 or qpos.shape[1] != d.nq or not len(qpos):
    raise ValueError(f"qpos must have shape (batch, {d.nq}) with batch > 0")
  batch = len(qpos)
  if qvel.shape != (batch, d.nv) or force.shape != (batch, d.nv) or mass.shape != (batch, d.nv, d.nv):
    raise ValueError("qvel, mass_matrix, and qfrc_smooth batch shapes do not match")
  if not all(np.all(np.isfinite(a)) for a in (qpos, qvel, mass, force)):
    raise ValueError("joint constraint inputs must be finite")
  if eq_active is None:
    active = np.broadcast_to(d.equality_active0.astype(bool), (batch, d.neq))
  else:
    active = np.asarray(eq_active, dtype=bool)
    if active.shape == (d.neq,):
      active = np.broadcast_to(active, (batch, d.neq))
    if active.shape != (batch, d.neq):
      raise ValueError(f"eq_active must have shape ({d.neq},) or ({batch}, {d.neq})")
  qfrc_constraint = np.zeros((batch, d.nv), dtype=np.float64)
  qacc = np.zeros_like(qfrc_constraint)
  status = np.zeros(batch, dtype=np.int32)
  residual = np.zeros(batch, dtype=np.float64)
  iterations = np.zeros(batch, dtype=np.int32)
  for world in range(batch):
    if d.disableflags & _DISABLE_CONSTRAINT:
      qacc[world] = np.linalg.solve(mass[world], force[world]) if d.nv else np.empty(0)
      continue
    J = np.zeros((d.nrow, d.nv), dtype=np.float64)
    R = np.ones(d.nrow, dtype=np.float64)
    aref = np.zeros(d.nrow, dtype=np.float64)
    lo = np.zeros(d.nrow, dtype=np.float64)
    hi = np.zeros(d.nrow, dtype=np.float64)
    enabled = np.zeros(d.nrow, dtype=bool)
    for eid in range(d.neq):
      row = eid
      if d.disableflags & _DISABLE_EQUALITY or not active[world, eid]:
        continue
      j1, j2 = int(d.equality_obj1[eid]), int(d.equality_obj2[eid])
      qa1, da1 = int(d.joint_qposadr[j1]), int(d.joint_dofadr[j1])
      value1 = qpos[world, qa1]
      ref1 = float(d.qpos0[qa1])
      data = d.equality_data[eid]
      if j2 >= 0:
        qa2, da2 = int(d.joint_qposadr[j2]), int(d.joint_dofadr[j2])
        dif = qpos[world, qa2] - float(d.qpos0[qa2])
        poly = sum(float(data[k+1])*dif**(k+1) for k in range(4))
        derivative = sum((k+1)*float(data[k+1])*dif**k for k in range(4))
        pos = value1-ref1-float(data[0])-poly
        J[row, da1], J[row, da2] = 1., -derivative
        vel = qvel[world, da1] - derivative*qvel[world, da2]
        diag_a = d.dof_invweight0[da1] + d.dof_invweight0[da2]
      else:
        pos = value1-ref1-float(data[0])
        J[row, da1] = 1.
        vel = qvel[world, da1]
        diag_a = d.dof_invweight0[da1]
      R[row], aref[row] = _reference_and_r(
          d, d.equality_solref[eid], d.equality_solimp[eid], pos, 0.,
          vel, diag_a, friction=False,
      )
      lo[row], hi[row], enabled[row] = -np.inf, np.inf, True
    friction_disabled = bool(d.disableflags & _DISABLE_FRICTION)
    for dof in range(d.nv):
      row = d.neq+dof
      loss = float(d.dof_frictionloss[dof])
      if friction_disabled or loss <= 0:
        continue
      J[row, dof] = 1.
      R[row], aref[row] = _reference_and_r(
          d, d.dof_solref[dof], d.dof_solimp[dof], 0., 0.,
          qvel[world, dof], d.dof_invweight0[dof], friction=True,
      )
      lo[row], hi[row], enabled[row] = -loss, loss, True
    limit_disabled = bool(d.disableflags & _DISABLE_LIMIT)
    for jid in range(d.njnt):
      if not d.joint_limited[jid]:
        continue
      dof, qa = int(d.joint_dofadr[jid]), int(d.joint_qposadr[jid])
      for side_index, side in enumerate((-1, 1)):
        row = d.neq+d.nv+2*jid+side_index
        value = qpos[world, qa]
        dist = side * (float(d.joint_range[jid, side_index])-value)
        margin = float(d.joint_margin[jid])
        if limit_disabled or dist >= margin:
          continue
        J[row, dof] = -float(side)
        R[row], aref[row] = _reference_and_r(
            d, d.joint_solref[jid], d.joint_solimp[jid], dist, margin,
            J[row] @ qvel[world], d.dof_invweight0[dof], friction=False,
        )
        lo[row], hi[row], enabled[row] = 0., np.inf, True
    q0 = np.linalg.solve(mass[world], force[world]) if d.nv else np.empty(0)
    if d.nrow:
      minv_jt = np.linalg.solve(mass[world], J.T) if d.nv else np.zeros((0, d.nrow))
      W = J @ minv_jt + np.diag(R)
      b = aref - J @ q0
      for row in range(d.nrow):
        if not enabled[row]:
          W[row, :] = 0.; W[:, row] = 0.; W[row, row] = 1.; b[row] = 0.
          lo[row] = hi[row] = 0.
      lam = np.zeros(d.nrow, dtype=np.float64)
      for sweep in range(max_iterations):
        for row in range(d.nrow):
          value = b[row] - W[row] @ lam + W[row, row]*lam[row]
          candidate = value / W[row, row]
          lam[row] = min(hi[row], max(lo[row], candidate))
        grad = W @ lam - b
        projected = np.minimum(hi, np.maximum(lo, lam-grad))
        residual[world] = float(np.max(np.abs(projected-lam)))
        iterations[world] = sweep+1
        if residual[world] <= tolerance:
          break
      else:
        status[world] = 1
      qfrc_constraint[world] = J.T @ lam
    qacc[world] = np.linalg.solve(mass[world], force[world]+qfrc_constraint[world]) if d.nv else np.empty(0)
  return dict(qfrc_constraint=qfrc_constraint, qacc=qacc, status=status,
              residual=residual, iterations=iterations)


def _reference_and_r(d, solref, solimp, pos, margin, vel, diag_a, friction):
  impedance = _impedance(solimp, pos, margin)
  ref = np.array(solref, dtype=np.float64, copy=True)
  if d.refsafe and ref[0] > 0:
    ref[0] = max(float(ref[0]), 2*d.timestep)
  # MuJoCo 3.10 getsolparam/efc_KBIP use d_width=solimp[1]. The separate
  # solimp[2] parameter is the impedance transition width used by _impedance.
  width = max(_MINVAL, float(solimp[1]))
  if ref[0] > 0:
    stiffness = 1/max(_MINVAL, width*width*ref[0]*ref[0]*ref[1]*ref[1])
  else:
    stiffness = -ref[0]/max(_MINVAL, width*width)
  if friction:
    stiffness = 0.
  damping = 2/max(_MINVAL, width*ref[0]) if ref[1] > 0 else -ref[1]/width
  compliance = max(_MINVAL, (1-impedance)*diag_a/impedance)
  return compliance, -damping*vel-stiffness*impedance*(pos-margin)


class JointConstraintProgram:
  """Explicit MPS wrapper for the bounded coupled joint constraint solve."""

  def __init__(self, model, batch_size=1):
    self.descriptor = lower_joint_constraints(model)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = int(batch_size)
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("joint constraint stage requires PyTorch MPS compile_shader")
    self._torch, self._device = torch, torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.solve_joint_constraints
    d = self.descriptor
    self._arrays = {}
    for name in (
        "joint_type", "joint_qposadr", "qpos0", "joint_dofadr", "joint_limited",
        "joint_range", "joint_margin", "joint_solref", "joint_solimp",
        "dof_frictionloss", "dof_invweight0", "dof_solref", "dof_solimp",
        "equality_type", "equality_obj1", "equality_obj2", "equality_data",
        "equality_solref", "equality_solimp",
    ):
      host = np.array(getattr(d, name), copy=True)
      if host.size == 0:
        host = np.zeros(1, dtype=host.dtype)
      self._arrays[name] = torch.as_tensor(host, device=self._device)
    self._dims = torch.tensor(
        [d.nq, d.nv, d.njnt, d.neq, d.nrow, self.batch_size,
         d.disableflags, int(d.refsafe), _MAX_ITERATIONS],
        dtype=torch.int32, device=self._device,
    )
    self._timestep = torch.tensor([d.timestep], dtype=torch.float32, device=self._device)
    self._outputs = {
        "qfrc_constraint": torch.empty((self.batch_size, d.nv), dtype=torch.float32, device=self._device),
        "qacc": torch.empty((self.batch_size, d.nv), dtype=torch.float32, device=self._device),
        "status": torch.empty((self.batch_size,), dtype=torch.int32, device=self._device),
        "residual": torch.empty((self.batch_size,), dtype=torch.float32, device=self._device),
        "iterations": torch.empty((self.batch_size,), dtype=torch.int32, device=self._device),
    }
    self._eq_active0 = torch.as_tensor(np.array(d.equality_active0, dtype=np.int32, copy=True), device=self._device)

  def run_device(self, mass_matrix, qfrc_smooth, qpos, qvel, eq_active=None):
    """Solve one coupled batch; output tensors borrow this program's workspace."""
    torch, d = self._torch, self.descriptor
    expected = {
        "mass_matrix": (self.batch_size, d.nv, d.nv),
        "qfrc_smooth": (self.batch_size, d.nv),
        "qpos": (self.batch_size, d.nq),
        "qvel": (self.batch_size, d.nv),
    }
    values = {"mass_matrix": mass_matrix, "qfrc_smooth": qfrc_smooth, "qpos": qpos, "qvel": qvel}
    for name, shape in expected.items():
      tensor = values[name]
      if not isinstance(tensor, torch.Tensor) or tensor.device.type != "mps" or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    if eq_active is None:
      active = self._eq_active0.unsqueeze(0).expand(self.batch_size, d.neq).contiguous()
    else:
      if eq_active.device.type != "mps" or eq_active.dtype not in (torch.int32, torch.bool) or not eq_active.is_contiguous() or tuple(eq_active.shape) != (self.batch_size, d.neq):
        raise ValueError(f"eq_active must be contiguous int32/bool MPS with shape ({self.batch_size}, {d.neq})")
      active = eq_active.to(dtype=torch.int32).contiguous()
    args = [mass_matrix.reshape(-1), qfrc_smooth.reshape(-1), qpos.reshape(-1), qvel.reshape(-1), active.reshape(-1)]
    for name in (
        "joint_type", "joint_qposadr", "qpos0", "joint_dofadr", "joint_limited",
        "joint_range", "joint_margin", "joint_solref", "joint_solimp",
        "dof_frictionloss", "dof_invweight0", "dof_solref", "dof_solimp",
        "equality_obj1", "equality_obj2", "equality_data", "equality_solref",
        "equality_solimp",
    ):
      args.append(self._arrays[name].reshape(-1))
    args.extend([self._dims, self._timestep])
    args.extend(self._outputs[name] for name in ("qfrc_constraint", "qacc", "status", "residual", "iterations"))
    self._kernel(*args, threads=(self.batch_size,), group_size=(1,))
    return self._outputs
