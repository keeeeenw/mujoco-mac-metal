# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0

"""Native joint spring and damper force stage for supported rigid joints."""

from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "passive.metal"


class PassiveForceModel:
  """Joint-level passive force constants copied from a MuJoCo 3.10 model."""

  def __init__(self, model):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"passive force lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    self.model = model
    self.nq, self.nv, self.njnt = int(model.nq), int(model.nv), int(model.njnt)
    self.jnt_type = np.asarray(model.jnt_type, dtype=np.int32).copy()
    self.qadr = np.asarray(model.jnt_qposadr, dtype=np.int32).copy()
    self.dadr = np.asarray(model.jnt_dofadr, dtype=np.int32).copy()
    self.springref = np.asarray(model.qpos_spring, dtype=np.float32).copy()
    self.stiffness = np.asarray(model.jnt_stiffness, dtype=np.float32).copy()
    self.springpoly = np.asarray(model.jnt_stiffnesspoly, dtype=np.float32).reshape(self.njnt, -1).copy()
    self.damping = np.asarray(model.dof_damping, dtype=np.float32).copy()
    self.damperpoly = np.asarray(model.dof_dampingpoly, dtype=np.float32).reshape(self.nv, -1).copy()
    self.disableflags = int(model.opt.disableflags)
    self._num_poly = self.springpoly.shape[1]
    rigid = {int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)}
    scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    supported = rigid | scalar
    if any(int(t) not in supported for t in self.jnt_type):
      raise ValueError("unsupported joint type in passive force model")
    if self._num_poly != 2:
      raise ValueError("MuJoCo 3.10 polynomial passive terms require two coefficients")

  def _poly(self, linear, poly, x, odd=False):
    xx = abs(x) if odd else x
    return linear + poly[0] * xx + poly[1] * xx * xx

  def force(self, qpos, qvel):
    """Compute native-supported joint spring and dof damper forces on CPU."""
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != self.nq or qvel.shape != (qpos.shape[0], self.nv) or not qpos.shape[0]:
      raise ValueError(f"qpos/qvel must have shapes (B, {self.nq}) and (B, {self.nv})")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
      raise ValueError("qpos and qvel must be finite")
    out = np.zeros((len(qpos), self.nv), dtype=np.float64)
    if not self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING):
      for j, typ in enumerate(self.jnt_type):
        da, qa = int(self.dadr[j]), int(self.qadr[j])
        if typ in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
          x = qpos[:, qa] - float(self.springref[qa])
          out[:, da] += -x * (self.stiffness[j] + self.springpoly[j, 0] * x + self.springpoly[j, 1] * x * x)
        elif typ == int(mujoco.mjtJoint.mjJNT_FREE):
          diff = qpos[:, qa:qa+3] - self.springref[qa:qa+3]
          radius = np.linalg.norm(diff, axis=1)
          k = self.stiffness[j] + self.springpoly[j, 0] * radius + self.springpoly[j, 1] * radius**2
          out[:, da:da+3] += -diff * k[:, None]
          # Quaternion spring terms are deliberately rejected until their
          # MuJoCo quaternion-difference convention is represented natively.
        # Ball and free rotational springs are intentionally zero for now.
    if not self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER):
      for d in range(self.nv):
        v = qvel[:, d]
        out[:, d] += -v * (self.damping[d] + self.damperpoly[d, 0] * np.abs(v) + self.damperpoly[d, 1] * v * v)
    return out


class MetalPassiveForces:
  """MPS launch wrapper for linear/polynomial rigid-joint springs and dampers."""

  def __init__(self, model):
    import torch
    self.model = model
    self._torch = torch
    self._device = torch.device("mps")
    self._lib = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._lib.passive_joint_force
    self._meta = PassiveForceModel(model)
    def tensor(a, dtype=torch.float32):
      return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=self._device)
    self._qadr = tensor(self._meta.qadr, torch.int32)
    self._dadr = tensor(self._meta.dadr, torch.int32)
    self._type = tensor(self._meta.jnt_type, torch.int32)
    self._ref = tensor(self._meta.springref)
    self._stiff = tensor(self._meta.stiffness)
    self._spoly = tensor(self._meta.springpoly.reshape(-1))
    self._damp = tensor(self._meta.damping)
    self._dpoly = tensor(self._meta.damperpoly.reshape(-1))
    self._dims = tensor([self._meta.nq, self._meta.nv, self._meta.njnt, self._meta._num_poly, self._meta.disableflags], torch.int32)

  def run_device(self, qpos, qvel):
    """Return borrowed MPS generalized passive forces for device state tensors."""
    torch = self._torch
    if not isinstance(qpos, torch.Tensor) or not isinstance(qvel, torch.Tensor):
      raise TypeError("qpos and qvel must be torch.Tensor values")
    b = qpos.shape[0] if qpos.ndim == 2 else -1
    if b <= 0 or qpos.shape != (b, self._meta.nq) or qvel.shape != (b, self._meta.nv):
      raise ValueError("qpos and qvel have invalid batch dimensions")
    for name, value in (("qpos", qpos), ("qvel", qvel)):
      if value.device != self._device or value.dtype != torch.float32 or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS")
    out = torch.empty((b, self._meta.nv), dtype=torch.float32, device=self._device)
    self._kernel(qpos.reshape(-1), qvel.reshape(-1), self._qadr, self._dadr, self._type, self._ref, self._stiff, self._spoly, self._damp, self._dpoly, self._dims, out.reshape(-1), threads=(b,), group_size=(1,))
    return out
