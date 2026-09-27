# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Fixed-joint tendon passive forces, damping tangent, and armature matrix."""

from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "tendons.metal"
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


def _frozen(value, dtype=np.float32):
  array = np.array(value, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("tendon constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class FixedTendonModel:
  """CPU reference lowered from one pinned MuJoCo fixed-joint-tendon model.

  Tendon lengths use the MuJoCo 3.10 definition, ``sum(coef * qpos)``. Spatial
  wraps, tendon limits/friction, actuator-inherited armature/damping, and other
  passive subsystems are rejected explicitly.
  """

  def __init__(self, model):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"fixed tendon lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    self.nq, self.nv, self.ntendon = int(model.nq), int(model.nv), int(model.ntendon)
    if any(value < 0 or value > _INT32_MAX for value in (self.nq, self.nv, self.ntendon)):
      raise ValueError("tendon dimensions exceed int32")
    if self.ntendon * self.nq > _UINT32_MAX or self.ntendon * self.nv > _UINT32_MAX:
      raise ValueError("tendon model buffers exceed uint32 indexing")
    if np.any(np.asarray(model.actuator_armature) != 0):
      raise ValueError("actuator-inherited tendon armature is unsupported")
    if np.any(np.asarray(model.actuator_damping) != 0) or np.any(np.asarray(model.actuator_dampingpoly) != 0):
      raise ValueError("actuator-inherited tendon damping is unsupported")
    if np.any(np.asarray(model.tendon_limited)):
      raise ValueError("tendon limits are unsupported")
    if np.any(np.asarray(model.tendon_frictionloss) != 0):
      raise ValueError("tendon friction loss is unsupported")
    if np.any(np.asarray(model.tendon_actfrclimited)):
      raise ValueError("tendon actuator-force limits are unsupported")

    hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    wrap_joint = int(mujoco.mjtWrap.mjWRAP_JOINT)
    length_map = np.zeros((self.ntendon, self.nq), dtype=np.float64)
    moment_map = np.zeros((self.ntendon, self.nv), dtype=np.float64)
    for tendon in range(self.ntendon):
      start, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
      if count <= 0:
        raise ValueError("tendons must contain at least one fixed joint wrap")
      for wrap in range(start, start+count):
        if int(model.wrap_type[wrap]) != wrap_joint:
          raise ValueError("only fixed tendons made of joint wraps are supported")
        joint = int(model.wrap_objid[wrap])
        if joint < 0 or joint >= int(model.njnt) or int(model.jnt_type[joint]) not in (hinge, slide):
          raise ValueError("fixed tendon wraps must target hinge or slide joints")
        coefficient = float(model.wrap_prm[wrap])
        length_map[tendon, int(model.jnt_qposadr[joint])] += coefficient
        moment_map[tendon, int(model.jnt_dofadr[joint])] += coefficient

    self.length_map = _frozen(length_map)
    self.moment_map = _frozen(moment_map)
    self.stiffness = _frozen(model.tendon_stiffness)
    self.stiffnesspoly = _frozen(np.asarray(model.tendon_stiffnesspoly).reshape(self.ntendon, 2))
    self.damping = _frozen(model.tendon_damping)
    self.dampingpoly = _frozen(np.asarray(model.tendon_dampingpoly).reshape(self.ntendon, 2))
    self.spring_range = _frozen(np.asarray(model.tendon_lengthspring).reshape(self.ntendon, 2))
    self.armature = _frozen(model.tendon_armature)
    if np.any(self.armature < 0) or np.any(self.damping < 0):
      raise ValueError("tendon armature and damping must be nonnegative")
    self.disable_spring = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    self.disable_damper = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    ancestor_mask = np.zeros((self.nv, self.nv), dtype=np.float32)
    for dof in range(self.nv):
      anc = dof
      while anc >= 0:
        ancestor_mask[dof, anc] = 1.0
        ancestor_mask[anc, dof] = 1.0
        anc = int(model.dof_parentid[anc])
    self.ancestor_mask = _frozen(ancestor_mask)

  def _state(self, qpos, qvel):
    qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != self.nq or qvel.shape != (qpos.shape[0], self.nv) or not qpos.shape[0]:
      raise ValueError(f"qpos/qvel must have shapes (B, {self.nq}) and (B, {self.nv})")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
      raise ValueError("qpos and qvel must be finite")
    return qpos, qvel

  def run(self, qpos, qvel):
    """Return ``(qfrc_tendon, damping_matrix, armature_matrix)`` on CPU."""
    qpos, qvel = self._state(qpos, qvel)
    batch = len(qpos)
    lengths = qpos @ self.length_map.T
    velocities = qvel @ self.moment_map.T
    force = np.zeros((batch, self.nv), dtype=np.float64)
    damping_matrix = np.zeros((batch, self.nv, self.nv), dtype=np.float64)
    armature_matrix = np.zeros((self.nv, self.nv), dtype=np.float64)
    for tendon in range(self.ntendon):
      jac = self.moment_map[tendon].astype(np.float64)
      if not self.disable_spring:
        displacement = np.where(
            lengths[:, tendon] > self.spring_range[tendon, 1],
            lengths[:, tendon] - self.spring_range[tendon, 1],
            np.where(
                lengths[:, tendon] < self.spring_range[tendon, 0],
                lengths[:, tendon] - self.spring_range[tendon, 0],
                0.0,
            ),
        )
        spring_coef = (
            self.stiffness[tendon]
            + self.stiffnesspoly[tendon, 0] * displacement
            + self.stiffnesspoly[tendon, 1] * displacement * displacement
        )
        spring_force = -displacement * spring_coef
      else:
        spring_force = np.zeros(batch, dtype=np.float64)
      if not self.disable_damper:
        velocity = velocities[:, tendon]
        speed = np.abs(velocity)
        damper_coef = (
            self.damping[tendon]
            + self.dampingpoly[tendon, 0] * speed
            + self.dampingpoly[tendon, 1] * velocity * velocity
        )
        damper_force = -velocity * damper_coef
        damper_tangent = (
            self.damping[tendon]
            + 2 * self.dampingpoly[tendon, 0] * speed
            + 3 * self.dampingpoly[tendon, 1] * velocity * velocity
        )
      else:
        damper_force = np.zeros(batch, dtype=np.float64)
        damper_tangent = np.zeros(batch, dtype=np.float64)
      force += (spring_force + damper_force)[:, None] * jac[None, :]
      damping_matrix += damper_tangent[:, None, None] * np.outer(jac, jac)[None, :, :]
      armature_matrix += self.armature[tendon] * np.outer(jac, jac)
    armature_matrix *= self.ancestor_mask
    return force, damping_matrix, armature_matrix


class MetalFixedTendonDynamics:
  """Native MPS fixed-tendon passive-force and rank-one dynamics stage."""

  def __init__(self, model, batch_size=1):
    self._meta = FixedTendonModel(model)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = batch_size
    for value in (self._meta.nq, self._meta.nv, self._meta.ntendon, batch_size):
      if value > _INT32_MAX:
        raise ValueError("tendon dimensions exceed int32")
    if batch_size * max(self._meta.nq, self._meta.nv, self._meta.ntendon, 1) > _UINT32_MAX:
      raise ValueError("tendon workspace exceeds uint32 indexing")
    if batch_size * self._meta.nv * self._meta.nv > _UINT32_MAX:
      raise ValueError("tendon matrix workspace exceeds uint32 indexing")
    import torch
    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.fixed_tendon_dynamics
    def tensor(value, dtype=torch.float32):
      npdtype = np.int32 if dtype == torch.int32 else np.float32
      array = np.array(value, dtype=npdtype, order="C", copy=True)
      if array.dtype == np.float32 and not np.all(np.isfinite(array)):
        raise ValueError("tendon constants must be finite float32")
      if array.size == 0:
        array = np.zeros(1, dtype=npdtype)
      return torch.as_tensor(array, dtype=dtype, device=self._device)
    meta = self._meta
    self._length_map = tensor(meta.length_map.reshape(-1))
    self._moment_map = tensor(meta.moment_map.reshape(-1))
    self._stiffness = tensor(meta.stiffness)
    self._stiffnesspoly = tensor(meta.stiffnesspoly.reshape(-1))
    self._damping = tensor(meta.damping)
    self._dampingpoly = tensor(meta.dampingpoly.reshape(-1))
    self._spring_range = tensor(meta.spring_range.reshape(-1))
    self._armature = tensor(meta.armature)
    self._ancestor_mask = tensor(meta.ancestor_mask.reshape(-1))
    self._dims = tensor([meta.nq, meta.nv, meta.ntendon, batch_size, int(meta.disable_spring), int(meta.disable_damper)], torch.int32)
    self._qfrc = torch.empty((batch_size, meta.nv), dtype=torch.float32, device=self._device)
    self._damping_matrix = torch.empty((batch_size, meta.nv, meta.nv), dtype=torch.float32, device=self._device)
    self._armature_matrix = torch.empty((batch_size, meta.nv, meta.nv), dtype=torch.float32, device=self._device)
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)

  def run_device(self, qpos, qvel):
    """Return borrowed `(qfrc, damping matrix, armature matrix)` MPS tensors."""
    torch, meta = self._torch, self._meta
    for name, value, shape in (
        ("qpos", qpos, (self.batch_size, meta.nq)),
        ("qvel", qvel, (self.batch_size, meta.nv)),
    ):
      if not isinstance(value, torch.Tensor) or value.device.type != "mps" or value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    self._kernel(
        qpos.reshape(-1) if meta.nq else self._dummy,
        qvel.reshape(-1) if meta.nv else self._dummy,
        self._length_map, self._moment_map, self._stiffness,
        self._stiffnesspoly, self._damping, self._dampingpoly,
        self._spring_range, self._armature, self._dims,
        self._qfrc.reshape(-1) if meta.nv else self._dummy,
        self._damping_matrix.reshape(-1) if meta.nv else self._dummy,
        self._armature_matrix.reshape(-1) if meta.nv else self._dummy,
        self._ancestor_mask,
        threads=(self.batch_size,), group_size=(1,),
    )
    return self._qfrc, self._damping_matrix, self._armature_matrix
