# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0

"""Native joint spring and damper force stage for supported rigid joints."""

from pathlib import Path

import mujoco
import numpy as np
from mujoco_metal.metal_kinematics import MetalKinematics
from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
from mujoco_metal.model import load_model

_SHADER = Path(__file__).parent / "shaders" / "passive.metal"


def _frozen(value, dtype):
  array = np.ascontiguousarray(value, dtype=dtype)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError(
        "passive model constants must be finite and float32-representable"
    )
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class PassiveForceModel:
  """Joint-level passive force constants copied from a MuJoCo 3.10 model."""

  def __init__(self, model):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(
          f"passive force lowering requires MuJoCo 3.10.0; found {mujoco.__version__}"
      )
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    self.model = model
    self.nq, self.nv, self.njnt, self.nbody = (
        int(model.nq),
        int(model.nv),
        int(model.njnt),
        int(model.nbody),
    )
    self.jnt_type = _frozen(model.jnt_type, np.int32)
    self.qadr = _frozen(model.jnt_qposadr, np.int32)
    self.dadr = _frozen(model.jnt_dofadr, np.int32)
    self.springref = _frozen(model.qpos_spring, np.float32)
    self.stiffness = _frozen(model.jnt_stiffness, np.float32)
    self.springpoly = _frozen(
        np.asarray(model.jnt_stiffnesspoly).reshape(self.njnt, 2), np.float32
    )
    self.damping = _frozen(model.dof_damping, np.float32)
    self.damperpoly = _frozen(
        np.asarray(model.dof_dampingpoly).reshape(self.nv, 2), np.float32
    )
    # R06-1: joint-targeted actuator damping folds into the dof damper
    # (pinned mj_actuatorDamping gear^2 scan); tendon-targeted raises.
    from mujoco_metal.model import actuator_joint_inheritance
    _, _damp_fold, _dpoly_fold = actuator_joint_inheritance(model, tendon_ok=True)
    self.damping = _frozen(
        np.asarray(self.damping, dtype=np.float64) + _damp_fold, np.float32)
    self.damperpoly = _frozen(
        np.asarray(self.damperpoly, dtype=np.float64).reshape(self.nv, 2)
        + _dpoly_fold, np.float32)
    self.disableflags = int(model.opt.disableflags)
    self._num_poly = self.springpoly.shape[1]
    self.body_mass = _frozen(model.body_mass, np.float32)
    self.body_gravcomp = _frozen(model.body_gravcomp, np.float32)
    self.gravity = _frozen(model.opt.gravity, np.float32)
    self.jnt_actgravcomp = _frozen(model.jnt_actgravcomp, np.uint8)
    rigid = {int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)}
    scalar = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    supported = rigid | scalar
    if any(int(t) not in supported for t in self.jnt_type):
      raise ValueError("unsupported joint type in passive force model")
    if self._num_poly != 2:
      raise ValueError(
          "MuJoCo 3.10 polynomial passive terms require two coefficients"
      )
    if np.any(self.jnt_actgravcomp):
      raise ValueError("joint actuator gravity compensation is unsupported")

  def _poly(self, linear, poly, x, odd=False):
    xx = abs(x) if odd else x
    return linear + poly[0] * xx + poly[1] * xx * xx

  def force(self, qpos, qvel, xfrc_applied=None):
    """Compute native-supported joint spring and dof damper forces on CPU."""
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if (
        qpos.ndim != 2
        or qpos.shape[1] != self.nq
        or qvel.shape != (qpos.shape[0], self.nv)
        or not qpos.shape[0]
    ):
      raise ValueError(
          f"qpos/qvel must have shapes (B, {self.nq}) and (B, {self.nv})"
      )
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
      raise ValueError("qpos and qvel must be finite")
    if xfrc_applied is None:
      xfrc = np.zeros((len(qpos), int(self.model.nbody), 6), dtype=np.float64)
    else:
      xfrc = np.asarray(xfrc_applied, dtype=np.float64)
      if xfrc.shape != (len(qpos), int(self.model.nbody), 6) or not np.all(
          np.isfinite(xfrc)
      ):
        raise ValueError(
            f"xfrc_applied must be finite with shape (batch, {self.model.nbody}, 6)"
        )
    out = np.zeros((len(qpos), self.nv), dtype=np.float64)
    if not self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING):
      for j, typ in enumerate(self.jnt_type):
        da, qa = int(self.dadr[j]), int(self.qadr[j])
        if typ in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
          x = qpos[:, qa] - float(self.springref[qa])
          out[:, da] += -x * (
              self.stiffness[j]
              + self.springpoly[j, 0] * x
              + self.springpoly[j, 1] * x * x
          )
        elif typ == int(mujoco.mjtJoint.mjJNT_FREE):
          diff = qpos[:, qa : qa + 3] - self.springref[qa : qa + 3]
          radius = np.linalg.norm(diff, axis=1)
          k = (
              self.stiffness[j]
              + self.springpoly[j, 0] * radius
              + self.springpoly[j, 1] * radius**2
          )
          out[:, da : da + 3] += -diff * k[:, None]
          current = np.asarray(qpos[:, qa + 3 : qa + 7], dtype=np.float64)
          reference = np.asarray(
              self.springref[qa + 3 : qa + 7], dtype=np.float64
          )
          for row in range(len(qpos)):
            quat = current[row].copy()
            mujoco.mju_normalize4(quat)
            diff = np.zeros(3, dtype=np.float64)
            mujoco.mju_subQuat(diff, quat, reference)
            radius = np.linalg.norm(diff)
            k = self._poly(float(self.stiffness[j]), self.springpoly[j], radius)
            out[row, da + 3 : da + 6] += -diff * k
        elif typ == int(mujoco.mjtJoint.mjJNT_BALL):
          for row in range(len(qpos)):
            quat = np.asarray(qpos[row, qa : qa + 4], dtype=np.float64).copy()
            mujoco.mju_normalize4(quat)
            diff = np.zeros(3, dtype=np.float64)
            mujoco.mju_subQuat(diff, quat, self.springref[qa : qa + 4])
            radius = np.linalg.norm(diff)
            k = self._poly(float(self.stiffness[j]), self.springpoly[j], radius)
            out[row, da : da + 3] += -diff * k
    if not self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER):
      for d in range(self.nv):
        v = qvel[:, d]
        out[:, d] += -v * (
            self.damping[d]
            + self.damperpoly[d, 0] * np.abs(v)
            + self.damperpoly[d, 1] * v * v
        )
    gravity_disabled = bool(
        self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    ) or (
        bool(self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
        and bool(self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    )
    for row in range(len(qpos)):
      data = mujoco.MjData(self.model)
      data.qpos[:] = qpos[row]
      mujoco.mj_forward(self.model, data)
      for body in range(1, int(self.model.nbody)):
        force = xfrc[row, body, :3].copy()
        if not gravity_disabled and self.body_gravcomp[body] != 0:
          force -= (
              self.body_mass[body] * self.body_gravcomp[body] * self.gravity
          )
        torque = xfrc[row, body, 3:].copy()
        if not np.any(force) and not np.any(torque):
          continue
        jacp = np.zeros((3, self.nv), dtype=np.float64)
        jacr = np.zeros((3, self.nv), dtype=np.float64)
        mujoco.mj_jacBodyCom(self.model, data, jacp, jacr, body)
        out[row] += jacp.T @ force + jacr.T @ torque
    return out

  def damping_derivative(self, qvel):
    """Return the positive diagonal derivative of the damper force magnitude."""
    qvel = np.asarray(qvel, dtype=np.float64)
    if (
        qvel.ndim != 2
        or qvel.shape[1] != self.nv
        or not np.all(np.isfinite(qvel))
    ):
      raise ValueError(f"qvel must be finite with shape (batch, {self.nv})")
    if self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER):
      return np.zeros_like(qvel)
    return (
        self.damping[None, :]
        + 2 * self.damperpoly[None, :, 0] * np.abs(qvel)
        + 3 * self.damperpoly[None, :, 1] * qvel * qvel
    )


class MetalPassiveForces:
  """MPS launch wrapper for linear/polynomial rigid-joint springs and dampers."""

  def __init__(self, model):
    self.model = model
    self._meta = PassiveForceModel(model)
    import torch

    self._torch = torch
    self._device = torch.device("mps")
    self._lib = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._lib.passive_joint_force
    self._projection_kernel = self._lib.project_body_wrenches

    def tensor(a, dtype=torch.float32):
      arr = np.array(
          a,
          dtype=np.int32 if dtype == torch.int32 else np.float32,
          order="C",
          copy=True,
      )
      if arr.dtype == np.float32 and not np.all(np.isfinite(arr)):
        raise ValueError(
            "passive constants must be finite and float32-representable"
        )
      return torch.as_tensor(
          arr if arr.size else np.zeros(1, dtype=arr.dtype),
          dtype=dtype,
          device=self._device,
      )

    self._qadr = tensor(self._meta.qadr, torch.int32)
    self._dadr = tensor(self._meta.dadr, torch.int32)
    self._type = tensor(self._meta.jnt_type, torch.int32)
    self._ref = tensor(self._meta.springref)
    self._stiff = tensor(self._meta.stiffness)
    self._spoly = tensor(self._meta.springpoly.reshape(-1))
    self._damp = tensor(self._meta.damping)
    self._dpoly = tensor(self._meta.damperpoly.reshape(-1))
    spring_disabled = bool(
        self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    )
    damper_disabled = bool(
        self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    )
    self._dims = tensor(
        [
            self._meta.nq,
            self._meta.nv,
            self._meta.njnt,
            self._meta._num_poly,
            int(spring_disabled),
            int(damper_disabled),
        ],
        torch.int32,
    )
    self._descriptor = load_model(model)
    self._fk = MetalKinematics(self._descriptor, batch_size=1)
    for name in ("body_parentid", "body_jntadr", "body_jntnum", "jnt_bodyid"):
      setattr(
          self, "_" + name, tensor(getattr(self._descriptor, name), torch.int32)
      )
    self._body_mass = tensor(self._meta.body_mass)
    self._body_gravcomp = tensor(self._meta.body_gravcomp)
    self._unit_gravcomp = tensor(
        np.ones_like(np.asarray(self._meta.body_gravcomp, dtype=np.float32)))
    self._gravity = tensor(self._meta.gravity)
    gravity_disabled = bool(
        self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    ) or (spring_disabled and damper_disabled)
    self._projection_dims = tensor(
        [
            self._meta.nbody,
            self._meta.njnt,
            self._meta.nv,
            int(gravity_disabled),
        ],
        torch.int32,
    )
    self._gravcomp_dims = tensor(
        [
            self._meta.nbody,
            self._meta.njnt,
            self._meta.nv,
            int(bool(
                self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))),
        ],
        torch.int32,
    )
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._wrench = torch.zeros(
        (1, self._meta.nbody, 6), dtype=torch.float32, device=self._device
    )

  def run_device(self, qpos, qvel, xfrc_applied=None, return_damping=False,
                 mocap_pos=None, mocap_quat=None):
    """Return borrowed MPS generalized passive forces for device state tensors."""
    torch = self._torch
    if not isinstance(qpos, torch.Tensor) or not isinstance(qvel, torch.Tensor):
      raise TypeError("qpos and qvel must be torch.Tensor values")
    b = qpos.shape[0] if qpos.ndim == 2 else -1
    if (
        b <= 0
        or qpos.shape != (b, self._meta.nq)
        or qvel.shape != (b, self._meta.nv)
    ):
      raise ValueError("qpos and qvel have invalid batch dimensions")
    _validate_workspace_index_capacity(
        b,
        {
            "qpos": b * self._meta.nq,
            "qvel": b * self._meta.nv,
            "wrenches": b * self._meta.nbody * 6,
            "forces": b * self._meta.nv,
        },
        {
            "nq": self._meta.nq,
            "nv": self._meta.nv,
            "nbody": self._meta.nbody,
            "njnt": self._meta.njnt,
        },
    )
    for name, value in (("qpos", qpos), ("qvel", qvel)):
      if (
          value.device.type != "mps"
          or value.dtype != torch.float32
          or not value.is_contiguous()
      ):
        raise ValueError(f"{name} must be contiguous float32 MPS")
    if xfrc_applied is not None:
      if (
          not isinstance(xfrc_applied, torch.Tensor)
          or xfrc_applied.device.type != "mps"
          or xfrc_applied.dtype != torch.float32
          or not xfrc_applied.is_contiguous()
          or tuple(xfrc_applied.shape) != (b, self._meta.nbody, 6)
      ):
        raise ValueError(
            f"xfrc_applied must be contiguous float32 MPS with shape ({b}, {self._meta.nbody}, 6)"
        )
      wrench = xfrc_applied.reshape(-1)
    else:
      if self._wrench.shape[0] != b:
        self._wrench = torch.zeros(
            (b, self._meta.nbody, 6), dtype=torch.float32, device=self._device
        )
      wrench = self._wrench.reshape(-1)
    if self._fk._workspace["batch_size"] != b:
      self._fk.prepare_workspace(b)
    out = torch.empty(
        (b, self._meta.nv), dtype=torch.float32, device=self._device
    )
    deriv = torch.empty_like(out)
    self._kernel(
        qpos.reshape(-1) if self._meta.nq else self._dummy,
        qvel.reshape(-1) if self._meta.nv else self._dummy,
        self._qadr,
        self._dadr,
        self._type,
        self._ref,
        self._stiff,
        self._spoly,
        self._damp,
        self._dpoly,
        self._dims,
        out.reshape(-1) if self._meta.nv else self._dummy,
        deriv.reshape(-1) if self._meta.nv else self._dummy,
        threads=(b,),
        group_size=(1,),
    )
    if self._meta.nv and self._meta.nbody > 1:
      poses = self._fk.run_device(qpos, mocap_pos, mocap_quat)
      self._projection_kernel(
          self._body_parentid,
          self._body_jntadr,
          self._body_jntnum,
          self._jnt_bodyid,
          self._type,
          self._dadr,
          poses["body_quat"].reshape(-1),
          poses["inertial_pos"].reshape(-1),
          poses["joint_anchor"].reshape(-1),
          poses["joint_axis"].reshape(-1),
          self._body_mass,
          self._body_gravcomp,
          self._gravity,
          wrench,
          self._projection_dims,
          out.reshape(-1),
          threads=(b,),
          group_size=(1,),
      )
    return (out, deriv) if return_damping else out

  def gravcomp_device(self, qpos, mocap_pos=None, mocap_quat=None):
    """Return borrowed MPS per-dof gravity generalized forces (qfrc_gravcomp).

    Unit-scale projection of body weights, matching pinned qfrc_gravcomp used
    for actuator gravity-compensation routing (milestone 007). Zeros when
    gravity is disabled, zero, or the model has no DOFs. Never reads values
    back to the host.
    """
    torch = self._torch
    b = qpos.shape[0] if qpos.ndim == 2 else -1
    nv = self._meta.nv
    if b <= 0 or tuple(qpos.shape) != (b, self._meta.nq):
      raise ValueError("qpos has invalid batch dimensions")
    if qpos.device.type != "mps" or qpos.dtype != torch.float32 or not qpos.is_contiguous():
      raise ValueError("qpos must be contiguous float32 MPS")
    out = torch.zeros((b, max(nv, 1)), dtype=torch.float32, device=self._device)
    gravity = np.asarray(self._meta.gravity, dtype=np.float64)
    gravity_disabled = bool(
        self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    )
    if nv == 0 or gravity_disabled or not np.any(gravity != 0):
      return out.reshape(b, max(nv, 1)) if nv else out.reshape(b, 1) * 0
    if self._fk._workspace["batch_size"] != b:
      self._fk.prepare_workspace(b)
    poses = self._fk.run_device(qpos, mocap_pos, mocap_quat)
    if self._meta.nbody <= 1:
      return out
    self._projection_kernel(
        self._body_parentid,
        self._body_jntadr,
        self._body_jntnum,
        self._jnt_bodyid,
        self._type,
        self._dadr,
        poses["body_quat"].reshape(-1),
        poses["inertial_pos"].reshape(-1),
        poses["joint_anchor"].reshape(-1),
        poses["joint_axis"].reshape(-1),
        self._body_mass,
        self._unit_gravcomp,
        self._gravity,
        torch.zeros((b * self._meta.nbody * 6,), dtype=torch.float32, device=self._device),
        self._gravcomp_dims,
        out.reshape(-1),
        threads=(b,),
        group_size=(1,),
    )
    return out
