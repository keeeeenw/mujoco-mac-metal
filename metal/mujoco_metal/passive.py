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


def passive_workspace_sizes(meta, batch_size, *, validate=True):
  """Exact independently-addressed passive tensors for a prepared batch."""
  from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
  if (isinstance(batch_size, (bool, np.bool_))
      or not isinstance(batch_size, (int, np.integer))
      or int(batch_size) <= 0):
    raise ValueError("batch_size must be a positive integer")
  b = int(batch_size)
  nq, nv, nj, nb = (int(meta.nq), int(meta.nv), int(meta.njnt),
                    int(meta.nbody))
  model = getattr(meta, "model", meta)
  ng, ns = int(model.ngeom), int(model.nsite)
  nmocap = int(model.nmocap)
  ntree = int(np.max(np.asarray(model.body_treeid), initial=-1)) + 1
  if hasattr(meta, "_num_poly"):
    num_poly = int(meta._num_poly)
  elif nj == 0:
    # MuJoCo's passive polynomial ABI still uses two coefficients when the
    # model has no joints; the zero-length tensor receives a one-value guard.
    num_poly = 2
  else:
    num_poly = int(np.asarray(meta.jnt_stiffnesspoly).reshape(nj, -1).shape[1])
  sizes = {
      "passive.qadr": max(nj, 1),
      "passive.dadr": max(nj, 1),
      "passive.jnt_type": max(nj, 1),
      "passive.springref": max(nq, 1),
      "passive.stiffness": max(nj, 1),
      "passive.springpoly": max(nj * num_poly, 1),
      "passive.damping": max(nv, 1),
      "passive.damperpoly": max(nv * 2, 1),
      "passive.dims": 7 + b,
      "passive.body_parentid": max(nb, 1),
      "passive.body_jntadr": max(nb, 1),
      "passive.body_jntnum": max(nb, 1),
      "passive.jnt_bodyid": max(nj, 1),
      "passive.jnt_actgravcomp": max(nj, 1),
      "passive.body_mass": max(nb, 1),
      "passive.body_gravcomp": max(nb, 1),
      "passive.gravity": 3,
      "passive.projection_dims": 4 + b,
      "passive.gravcomp_dims": 4 + b,
      "passive.dummy_float": 1,
      "passive.dummy_int": 1,
      "passive.sleep_filter": 1,
      "passive.projection_accumulate": 1,
      "passive.projection_clear": 1,
      "passive.wrench": b * nb * 6,
      "passive.force": b * nv,
      "passive.damping_derivative": b * nv,
      "passive.gravcomp_output": b * max(nv, 1),
      "passive.gravcomp_zero_wrench": b * nb * 6,
  }
  # MetalPassiveForces owns an independent MetalKinematics instance. Count
  # its immutable FK inputs and prepared outputs separately from Simulation's
  # primary FK workspace; sharing source model metadata does not share device
  # tensors.
  for name, elements in (
      ("body_parentid", max(nb, 1)), ("body_treeid", max(nb, 1)),
      ("body_mocapid", max(nb, 1)), ("body_pos", max(nb * 3, 1)),
      ("body_quat", max(nb * 4, 1)), ("body_ipos", max(nb * 3, 1)),
      ("body_iquat", max(nb * 4, 1)), ("jnt_type", max(nj, 1)),
      ("jnt_qposadr", max(nj, 1)), ("jnt_bodyid", max(nj, 1)),
      ("jnt_pos", max(nj * 3, 1)), ("jnt_axis", max(nj * 3, 1)),
      ("qpos0", max(nq, 1)), ("geom_bodyid", max(ng, 1)),
      ("geom_type", max(ng, 1)), ("geom_size", max(ng * 3, 1)),
      ("geom_pos", max(ng * 3, 1)), ("geom_quat", max(ng * 4, 1)),
      ("site_bodyid", max(ns, 1)), ("site_pos", max(ns * 3, 1)),
      ("site_quat", max(ns * 4, 1)),
      ("actuator_trntype", max(int(model.nu), 1)),
      ("actuator_trnid", max(int(model.nu) * 2, 1)),
      ("actuator_gear", max(int(model.nu) * 6, 1)),
      ("actuator_damping", max(int(model.nu), 1)),
      ("actuator_dampingpoly", max(int(model.nu) * 2, 1)),
      ("tendon_treeid", max(int(model.ntendon) * 2, 1)),
      ("tendon_treenum", max(int(model.ntendon), 1)),
      ("tendon_j_rowadr", max(int(model.ntendon), 1)),
      ("tendon_j_rownnz", max(int(model.ntendon), 1)),
      ("tendon_j_colind", max(int(np.asarray(model.ten_J_colind).size), 1)),
      ("mass_rowadr", max(nv, 1)), ("mass_rownnz", max(nv, 1)),
      ("mass_colind", max(int(np.asarray(model.M_colind).size), 1)),
  ):
    sizes[f"passive.fk.model.{name}"] = elements
  for name, count, width in (
      ("body_pos", nb, 3), ("body_quat", nb, 4),
      ("geom_pos", ng, 3), ("geom_quat", ng, 4),
      ("site_pos", ns, 3), ("site_quat", ns, 4),
      ("inertial_pos", nb, 3), ("inertial_quat", nb, 4),
      ("joint_anchor", nj, 3), ("joint_axis", nj, 3),
  ):
    sizes[f"passive.fk.{name}"] = max(b * count * width, 1)
  sizes.update({
      "passive.fk.qpos": max(b * nq, 1),
      "passive.fk.dims": 8,
      "passive.fk.auxiliary": max(
          (b * nmocap * 7 if nmocap else 1) + nb
          + b * max(ntree, 1) + b, 1),
      "passive.fk.tree_awake": b * max(ntree, 1),
  })
  if validate:
    _validate_workspace_index_capacity(
        b, sizes, {"nq": nq, "nv": nv, "njnt": nj, "nbody": nb})
    for name, elements in sizes.items():
      if elements > (1 << 31) - 1:
        raise ValueError(
            f"{name} exceeds the Metal int32 address range ({elements})")
  return sizes


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
      # Pinned mj_passive adds body gravity compensation to passive force
      # only on DOFs whose joint is not marked actuatorgravcomp. The same
      # qfrc_gravcomp vector is routed to qfrc_actuator for flagged joints.
      gravcomp_disabled = gravity_disabled
      if not gravcomp_disabled:
        for dof in range(self.nv):
          joint = int(self.model.dof_jntid[dof])
          if not self.jnt_actgravcomp[joint]:
            out[row, dof] += float(data.qfrc_gravcomp[dof])
      for body in range(1, int(self.model.nbody)):
        # Gravity compensation is already supplied by the source
        # qfrc_gravcomp vector above, filtered per joint. This loop is only
        # the independently applied body wrench; adding body gravity here as
        # well would double-count it and leak flagged-joint compensation.
        force = xfrc[row, body, :3].copy()
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

  def __init__(self, model, batch_size=None):
    self.model = model
    self._meta = PassiveForceModel(model)
    if (batch_size is not None
        and (isinstance(batch_size, (bool, np.bool_))
             or not isinstance(batch_size, (int, np.integer))
             or int(batch_size) <= 0)):
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = None if batch_size is None else int(batch_size)
    passive_workspace_sizes(self._meta, self.batch_size or 1)
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
    self._dims_base = [
            self._meta.nq,
            self._meta.nv,
            self._meta.njnt,
            self._meta._num_poly,
            int(spring_disabled),
            int(damper_disabled),
            self._meta.nbody,
        ]
    self._dims = tensor(
        self._dims_base + [1] * (self.batch_size or 1),
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
    self._jnt_actgravcomp = tensor(self._meta.jnt_actgravcomp, torch.int32)
    self._gravity = tensor(self._meta.gravity)
    gravity_disabled = bool(
        self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    ) or (spring_disabled and damper_disabled)
    self._projection_dims_base = [
            self._meta.nbody,
            self._meta.njnt,
            self._meta.nv,
            int(gravity_disabled),
        ]
    self._projection_dims = tensor(
        self._projection_dims_base + [1] * (self.batch_size or 1),
        torch.int32,
    )
    self._gravcomp_dims = tensor(
        [
            self._meta.nbody,
            self._meta.njnt,
            self._meta.nv,
            self._projection_dims_base[3],
        ] + [1] * (self.batch_size or 1),
        torch.int32,
    )
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._dummy_int = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._sleep_filter = torch.zeros(1, dtype=torch.int32, device=self._device)
    # Shared projection accumulates passive force, but replaces gravcomp.
    self._projection_accumulate = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._projection_clear = torch.ones(1, dtype=torch.int32, device=self._device)
    self._force = None
    self._damping_derivative = None
    self._wrench = torch.zeros(
        (self.batch_size or 1, self._meta.nbody, 6), dtype=torch.float32,
        device=self._device)
    self._gravcomp_output = (
        torch.zeros((self.batch_size, max(self._meta.nv, 1)),
                    dtype=torch.float32, device=self._device)
        if self.batch_size is not None else None)
    self._gravcomp_zero_wrench = (
        torch.zeros((self.batch_size, self._meta.nbody, 6),
                    dtype=torch.float32, device=self._device)
        if self.batch_size is not None else None)

  def _ensure_gravcomp_batch(self, batch):
    passive_workspace_sizes(self._meta, batch)
    if self.batch_size is None:
      if (self._gravcomp_output is None
          or tuple(self._gravcomp_output.shape)
          != (int(batch), max(self._meta.nv, 1))):
        self._gravcomp_output = self._torch.zeros(
            (batch, max(self._meta.nv, 1)), dtype=self._torch.float32,
            device=self._device)
        self._gravcomp_zero_wrench = self._torch.zeros(
            (batch, self._meta.nbody, 6), dtype=self._torch.float32,
            device=self._device)
    elif self.batch_size != int(batch):
      raise ValueError("gravcomp batch differs from its prepared workspace")

  def run_device(self, qpos, qvel, xfrc_applied=None, return_damping=False,
                 mocap_pos=None, mocap_quat=None, awake_lists=None,
                 poses=None, world_mask=None):
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
    if self.batch_size is not None and self.batch_size != b:
      raise ValueError("passive force batch differs from its prepared workspace")
    if self._dims.numel() != 7 + b:
      self._dims = torch.tensor(
          self._dims_base + [1] * b, dtype=torch.int32, device=self._device)
    if self._projection_dims.numel() != 4 + b:
      self._projection_dims = torch.tensor(
          self._projection_dims_base + [1] * b,
          dtype=torch.int32, device=self._device)
    if world_mask is None:
      self._dims[7:].fill_(1)
      self._projection_dims[4:].fill_(1)
    else:
      if (not isinstance(world_mask, torch.Tensor)
          or world_mask.dtype != torch.int32 or world_mask.device.type != "mps"
          or tuple(world_mask.shape) != (b,) or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous int32 MPS with shape (batch,)")
      self._dims[7:].copy_(world_mask)
      self._projection_dims[4:].copy_(world_mask)
    _validate_workspace_index_capacity(
        b,
        {
            **passive_workspace_sizes(self._meta, b),
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
    if self._force is None or tuple(self._force.shape) != (b, self._meta.nv):
      self._force = torch.zeros(
          (b, self._meta.nv), dtype=torch.float32, device=self._device)
      self._damping_derivative = torch.zeros_like(self._force)
    out, deriv = self._force, self._damping_derivative
    sleep_filter = awake_lists is not None and awake_lists.get("tree_awake") is not None
    if sleep_filter:
      required = ("body_ids", "dof_ids", "counts", "tree_awake")
      if not isinstance(awake_lists, dict) or any(k not in awake_lists for k in required):
        raise ValueError("awake_lists must contain body_ids, dof_ids, counts and tree_awake")
      for key in ("body_ids", "dof_ids", "tree_awake"):
        value = awake_lists[key]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int32
            or value.device.type != "mps" or not value.is_contiguous()):
          raise ValueError(f"awake_lists.{key} must be contiguous MPS int32")
      counts = awake_lists.get("counts")
      if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int32
          or counts.device.type != "mps" or tuple(counts.shape) != (b, 3)
          or not counts.is_contiguous()):
        raise ValueError("awake_lists.counts must be contiguous MPS int32 [batch,3]")
      if tuple(awake_lists["body_ids"].shape) != (b, max(self._meta.nbody, 1)):
        raise ValueError("awake_lists.body_ids has an invalid shape")
      if tuple(awake_lists["dof_ids"].shape) != (b, max(self._meta.nv, 1)):
        raise ValueError("awake_lists.dof_ids has an invalid shape")
      body_ids, body_count = awake_lists["body_ids"], counts
      dof_ids, dof_count = awake_lists["dof_ids"], counts
    else:
      # No-sleep callers use the original full-model path; optional buffers are
      # harmless because the shader ignores them when sleep_filter is zero.
      body_count = dof_ids = dof_count = self._dummy_int
      body_ids = self._dummy_int
    self._sleep_filter.fill_(int(sleep_filter))
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
        self._body_jntadr,
        self._body_jntnum,
        body_ids.reshape(-1) if sleep_filter else self._dummy_int,
        body_count,
        dof_ids.reshape(-1) if sleep_filter else self._dummy_int,
        dof_count,
        self._sleep_filter,
        out.reshape(-1) if self._meta.nv else self._dummy,
        deriv.reshape(-1) if self._meta.nv else self._dummy,
        threads=(b,),
        group_size=(1,),
    )
    if self._meta.nv and self._meta.nbody > 1:
      if poses is None:
        poses = self._fk.run_device(
            qpos, mocap_pos, mocap_quat,
            tree_awake=(awake_lists.get("tree_awake") if sleep_filter else None),
            world_mask=world_mask)
      else:
        required_pose_shapes = {
            "body_quat": (b, self._meta.nbody, 4),
            "inertial_pos": (b, self._meta.nbody, 3),
            "joint_anchor": (b, self._meta.njnt, 3),
            "joint_axis": (b, self._meta.njnt, 3),
        }
        for name, shape in required_pose_shapes.items():
          value = poses.get(name) if isinstance(poses, dict) else None
          if (not isinstance(value, torch.Tensor)
              or tuple(value.shape) != shape
              or value.device.type != "mps" or value.dtype != torch.float32
              or not value.is_contiguous()):
            raise ValueError(
                f"cached poses.{name} must be contiguous float32 MPS {shape}")
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
          body_ids.reshape(-1) if sleep_filter else self._dummy_int,
          body_count,
          self._sleep_filter,
          out.reshape(-1),
          self._projection_accumulate,
          self._jnt_actgravcomp,
          dof_ids.reshape(-1) if sleep_filter else self._dummy_int,
          dof_count,
          threads=(b,),
          group_size=(1,),
      )
    return (out, deriv) if return_damping else out

  def gravcomp_device(self, qpos, mocap_pos=None, mocap_quat=None, *,
                      poses=None, world_mask=None, awake_lists=None):
    """Return borrowed MPS per-dof gravity generalized forces (qfrc_gravcomp).

    Compiled body_gravcomp-scaled weights, matching pinned qfrc_gravcomp used
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
    self._ensure_gravcomp_batch(b)
    if self._gravcomp_dims.numel() != 4 + b:
      self._gravcomp_dims = torch.tensor(
          [self._meta.nbody, self._meta.njnt, self._meta.nv,
           self._projection_dims_base[3]] + [1] * b,
          dtype=torch.int32, device=self._device)
    if world_mask is None:
      self._gravcomp_dims[4:].fill_(1)
    else:
      if (not isinstance(world_mask, torch.Tensor)
          or world_mask.dtype != torch.int32 or world_mask.device.type != "mps"
          or tuple(world_mask.shape) != (b,) or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous int32 MPS with shape (batch,)")
      self._gravcomp_dims[4:].copy_(world_mask)
    out = self._gravcomp_output
    sleep_filter = (awake_lists is not None
                    and awake_lists.get("tree_awake") is not None)
    if sleep_filter:
      required = ("body_ids", "dof_ids", "counts", "tree_awake")
      if (not isinstance(awake_lists, dict)
          or any(name not in awake_lists for name in required)):
        raise ValueError("awake_lists must contain body_ids, dof_ids, counts and tree_awake")
      body_ids = awake_lists["body_ids"]
      dof_ids = awake_lists["dof_ids"]
      counts = awake_lists["counts"]
      for name, value in (("body_ids", body_ids), ("dof_ids", dof_ids),
                          ("tree_awake", awake_lists["tree_awake"]),
                          ("counts", counts)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int32
            or value.device.type != "mps" or not value.is_contiguous()):
          raise ValueError(f"awake_lists.{name} must be contiguous MPS int32")
      if (tuple(body_ids.shape) != (b, max(self._meta.nbody, 1))
          or tuple(dof_ids.shape) != (b, max(nv, 1))
          or tuple(counts.shape) != (b, 3)):
        raise ValueError("awake_lists has invalid body/dof/count dimensions")
    else:
      body_ids = dof_ids = counts = self._dummy_int
    self._sleep_filter.fill_(int(sleep_filter))
    if world_mask is None and not sleep_filter:
      out.zero_()
    if nv == 0:
      return out
    if poses is None:
      if self._fk._workspace["batch_size"] != b:
        self._fk.prepare_workspace(b)
      poses = self._fk.run_device(
          qpos, mocap_pos, mocap_quat, world_mask=world_mask)
    else:
      required_pose_shapes = {
          "body_quat": (b, self._meta.nbody, 4),
          "inertial_pos": (b, self._meta.nbody, 3),
          "joint_anchor": (b, self._meta.njnt, 3),
          "joint_axis": (b, self._meta.njnt, 3),
      }
      for name, shape in required_pose_shapes.items():
        value = poses.get(name) if isinstance(poses, dict) else None
        if (not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.device.type != "mps" or value.dtype != torch.float32
            or not value.is_contiguous()):
          raise ValueError(
              f"cached poses.{name} must be contiguous float32 MPS {shape}")
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
        self._body_gravcomp,
        self._gravity,
        self._gravcomp_zero_wrench.reshape(-1),
        self._gravcomp_dims,
        body_ids.reshape(-1) if sleep_filter else self._dummy_int,
        counts if sleep_filter else self._dummy_int,
        self._sleep_filter,
        out.reshape(-1),
        self._projection_clear,
        self._jnt_actgravcomp,  # real metadata; standalone mode ignores routing
        dof_ids.reshape(-1) if sleep_filter else self._dummy_int,
        counts if sleep_filter else self._dummy_int,
        threads=(b,),
        group_size=(1,),
    )
    return out
