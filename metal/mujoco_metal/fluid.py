# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""MuJoCo 3.10 inertia-box fluid forces, lowered to generalized force."""

from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "fluid.metal"
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


def _frozen(value):
  value = np.array(value, dtype=np.float32, order="C", copy=True)
  if not np.all(np.isfinite(value)):
    raise ValueError("fluid constants must be finite and float32-representable")
  return np.frombuffer(value.tobytes(), dtype=np.float32).reshape(value.shape)


def _supported_model(model):
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError(f"fluid lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  if np.any(np.asarray(model.geom_fluid)[:, 0] > 0):
    raise ValueError("geom-fluid ellipsoid/lift models are unsupported")
  if np.any(np.asarray(model.geom_fluid) != 0):
    raise ValueError("nonzero geom-fluid parameters are unsupported")
  if model.opt.density < 0 or model.opt.viscosity < 0:
    raise ValueError("fluid density and viscosity must be nonnegative")
  constants = np.r_[model.opt.density, model.opt.viscosity, model.opt.wind]
  if not np.all(np.isfinite(constants)) or not np.all(np.isfinite(constants.astype(np.float32))):
    raise ValueError("fluid coefficients and wind must be finite float32 values")
  dims = (int(model.nq), int(model.nv), int(model.nbody), int(model.njnt))
  if any(value < 0 or value > _INT32_MAX for value in dims):
    raise ValueError("fluid model dimensions exceed int32")
  return dims


class InertiaBoxFluidModel:
  """CPU reference for the pinned inertia-box force model.

  Returns only this stage's generalized force; fluid-inertia terms,
  ellipsoid geometry interactions, and lift are not part of this model.
  """

  def __init__(self, model):
    self.nq, self.nv, self.nbody, self.njnt = _supported_model(model)
    self.model = model
    self.mass = _frozen(model.body_mass)
    self.inertia = _frozen(model.body_inertia)
    self.wind = _frozen(model.opt.wind)
    self.density = float(model.opt.density)
    self.viscosity = float(model.opt.viscosity)
    self.disableflags = int(model.opt.disableflags)

  def run(self, qpos, qvel):
    qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != self.nq or qvel.shape != (qpos.shape[0], self.nv) or not qpos.shape[0]:
      raise ValueError(f"qpos/qvel must have shapes (B, {self.nq}) and (B, {self.nv})")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
      raise ValueError("qpos and qvel must be finite")
    result = np.zeros((len(qpos), self.nv), dtype=np.float64)
    # mj_passive returns before fluid evaluation when both bits are disabled.
    spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    if self.disableflags & spring and self.disableflags & damper:
      return result
    if not self.density and not self.viscosity:
      return result
    for row in range(len(qpos)):
      data = mujoco.MjData(self.model)
      data.qpos[:] = qpos[row]
      data.qvel[:] = qvel[row]
      mujoco.mj_forward(self.model, data)
      for body in range(1, self.nbody):
        mass = float(self.mass[body])
        if mass < mujoco.mjMINVAL:
          continue
        inertia = self.inertia[body].astype(np.float64)
        box = np.sqrt(np.maximum(mujoco.mjMINVAL,
            np.array([inertia[1]+inertia[2]-inertia[0], inertia[0]+inertia[2]-inertia[1], inertia[0]+inertia[1]-inertia[2]]) / mass * 6))
        local_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, local_velocity, 1)
        wind = np.r_[np.zeros(3), self.wind.astype(np.float64)]
        local_wind = np.zeros(6, dtype=np.float64)
        root = int(self.model.body_rootid[body])
        mujoco.mju_transformSpatial(local_wind, wind, 0, data.xipos[body], data.subtree_com[root], data.ximat[body].reshape(-1))
        local_velocity[3:] -= local_wind[3:]
        local_force = np.zeros(6, dtype=np.float64)
        if self.viscosity > 0:
          diameter = float(np.sum(box) / 3)
          local_force[:3] = -np.pi * diameter**3 * self.viscosity * local_velocity[:3]
          local_force[3:] = -3 * np.pi * diameter * self.viscosity * local_velocity[3:]
        if self.density > 0:
          for axis in range(3):
            other = [i for i in range(3) if i != axis]
            local_force[3+axis] -= .5 * self.density * box[other[0]] * box[other[1]] * abs(local_velocity[3+axis]) * local_velocity[3+axis]
            j, k = other
            local_force[axis] -= self.density * box[axis] * (box[j]**4 + box[k]**4) * abs(local_velocity[axis]) * local_velocity[axis] / 64
        rotation = data.ximat[body].reshape(3, 3)
        world_torque = rotation @ local_force[:3]
        world_force = rotation @ local_force[3:]
        mujoco.mj_applyFT(self.model, data, world_force, world_torque,
                          data.xipos[body], body, result[row])
    return result


class MetalInertiaBoxFluid:
  """Native MPS inertia-box fluid stage.

  `run_device` consumes qpos/qvel plus a native smooth-stage result containing
  `poses`, `cvel`, and `root_com`; this keeps kinematics owned by the caller's
  smooth stage and avoids a redundant mass/bias computation.
  """

  def __init__(self, model, batch_size=1):
    self._meta = InertiaBoxFluidModel(model)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0 or batch_size > _INT32_MAX:
      raise ValueError("batch_size must be a positive int32 integer")
    if max(batch_size * self._meta.nbody * 36,
           batch_size * self._meta.njnt * 3,
           batch_size * self._meta.nv) > _UINT32_MAX:
      raise ValueError("fluid workspace exceeds uint32 indexing")
    self.batch_size = batch_size
    import torch
    self._torch = torch
    self._device = torch.device("mps")
    meta = self._meta
    def tensor(value, dtype=torch.float32):
      arr = np.array(value, dtype=np.int32 if dtype == torch.int32 else np.float32, order="C", copy=True)
      if arr.dtype == np.float32 and not np.all(np.isfinite(arr)):
        raise ValueError("fluid constants must be finite float32")
      if not arr.size:
        arr = np.zeros(1, dtype=arr.dtype)
      return torch.as_tensor(arr, dtype=dtype, device=self._device)
    for name in ("body_parentid", "body_rootid", "body_jntadr", "body_jntnum", "jnt_type", "jnt_dofadr"):
      setattr(self, "_" + name, tensor(getattr(model, name), torch.int32))
    self._mass = tensor(meta.mass)
    self._inertia = tensor(meta.inertia.reshape(-1))
    self._fluid = tensor([meta.density, meta.viscosity, *meta.wind])
    spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    disabled = bool(meta.disableflags & spring and meta.disableflags & damper)
    self._dims = tensor([meta.nv, meta.nbody, meta.njnt, batch_size, int(disabled)], torch.int32)
    self._output = torch.empty((batch_size, meta.nv), dtype=torch.float32, device=self._device)
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.inertia_box_fluid

  def run_device(self, qpos, qvel, dynamics=None):
    torch, meta = self._torch, self._meta
    for name, value, shape in (("qpos", qpos, (self.batch_size, meta.nq)), ("qvel", qvel, (self.batch_size, meta.nv))):
      if not isinstance(value, torch.Tensor) or value.device.type != "mps" or value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    if dynamics is None:
      raise ValueError("pass the native smooth-stage result with poses, cvel, and root_com")
    required = {"poses", "cvel", "root_com"}
    if not isinstance(dynamics, dict) or not required.issubset(dynamics):
      raise ValueError("dynamics must contain native poses, cvel, and root_com")
    poses = dynamics["poses"]
    for name, value, shape in (
        ("poses.body_quat", poses["body_quat"], (self.batch_size, meta.nbody, 4)),
        ("poses.inertial_pos", poses["inertial_pos"], (self.batch_size, meta.nbody, 3)),
        ("poses.inertial_quat", poses["inertial_quat"], (self.batch_size, meta.nbody, 4)),
        ("poses.joint_anchor", poses["joint_anchor"], (self.batch_size, meta.njnt, 3)),
        ("poses.joint_axis", poses["joint_axis"], (self.batch_size, meta.njnt, 3)),
        ("cvel", dynamics["cvel"], (self.batch_size, meta.nbody, 6)),
        ("root_com", dynamics["root_com"], (self.batch_size, meta.nbody, 3)),
    ):
      if not isinstance(value, torch.Tensor) or value.device.type != "mps" or value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    nv = meta.nv
    self._kernel(self._body_parentid, self._body_jntadr, self._body_jntnum,
        self._jnt_type, self._jnt_dofadr, self._mass, self._inertia,
        poses["body_quat"].reshape(-1), poses["inertial_pos"].reshape(-1),
        poses["inertial_quat"].reshape(-1), dynamics["cvel"].reshape(-1),
        dynamics["root_com"].reshape(-1), poses["joint_anchor"].reshape(-1),
        poses["joint_axis"].reshape(-1), self._fluid, self._dims,
        self._output.reshape(-1) if nv else self._dummy,
        threads=(self.batch_size,), group_size=(1,))
    return self._output
