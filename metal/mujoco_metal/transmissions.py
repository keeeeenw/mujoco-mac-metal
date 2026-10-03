# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Stateless scalar actuator lowering and generalized-force evaluation."""

from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "transmissions.metal"
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


def _frozen(values, dtype):
  array = np.array(values, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("actuator constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class TransmissionModel:
  """Immutable model-derived scalar force map for MuJoCo 3.10.0.

  The accepted actuator family is stateless SISO with fixed or affine gain,
  none or affine bias, and scalar hinge/slide joint or fixed-joint-tendon
  transmission. Control
  clipping, actuator force clipping, actuator-group disable, and global
  actuation disable follow MuJoCo's source behavior.
  """

  def __init__(self, model, allow_inherited=False):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"transmission lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    self.model = model
    self.nq, self.nv, self.nu = int(model.nq), int(model.nv), int(model.nu)
    self.na = int(np.asarray(model.actuator_gaintype).size)
    if any(value < 0 or value > _INT32_MAX for value in (self.nq, self.nv, self.nu, self.na)):
      raise ValueError("transmission dimensions exceed int32")
    if self.nu != self.na:
      raise ValueError("only one-control, one-output scalar actuators are supported")
    if self.nu * self.nq > _UINT32_MAX or self.nu * self.nv > _UINT32_MAX:
      raise ValueError("transmission model buffers exceed uint32 indexing")
    if int(model.nplugin):
      raise ValueError("actuator plugins are unsupported")
    if np.any(np.asarray(model.actuator_actnum) != 0) or np.any(np.asarray(model.actuator_dyntype) != int(mujoco.mjtDyn.mjDYN_NONE)):
      raise ValueError("stateful actuator dynamics are unsupported")
    if not allow_inherited:
      if np.any(np.asarray(model.actuator_armature) != 0):
        raise ValueError("actuator armature is unsupported")
      if np.any(np.asarray(model.actuator_damping) != 0) or np.any(np.asarray(model.actuator_dampingpoly) != 0):
        raise ValueError("actuator damping is unsupported")
    else:
      from mujoco_metal.model import actuator_joint_inheritance, actuator_tendon_inheritance
      actuator_joint_inheritance(model, tendon_ok=True)
      actuator_tendon_inheritance(model)
    # Delay/history validated here; the stepping layer applies the delay
    # line above this scalar path (R06/D1).
    from mujoco_metal.stateful_actuation import actuator_delay_config
    actuator_delay_config(model)
    if np.any(np.asarray(model.jnt_actfrclimited)):
      raise ValueError("joint-level actuator force limits are unsupported")
    if np.any(np.asarray(model.jnt_actgravcomp)):
      raise ValueError("actuator gravity compensation routing is unsupported")

    fixed = int(mujoco.mjtGain.mjGAIN_FIXED)
    affine = int(mujoco.mjtGain.mjGAIN_AFFINE)
    no_bias = int(mujoco.mjtBias.mjBIAS_NONE)
    affine_bias = int(mujoco.mjtBias.mjBIAS_AFFINE)
    joint_trn = int(mujoco.mjtTrn.mjTRN_JOINT)
    parent_trn = int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    tendon_trn = int(mujoco.mjtTrn.mjTRN_TENDON)
    wrap_joint = int(mujoco.mjtWrap.mjWRAP_JOINT)
    hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    length_map = np.zeros((self.nu, self.nq), dtype=np.float64)
    moment_map = np.zeros((self.nu, self.nv), dtype=np.float64)
    for actuator in range(self.nu):
      gain_type = int(model.actuator_gaintype[actuator])
      bias_type = int(model.actuator_biastype[actuator])
      if gain_type not in (fixed, affine) or bias_type not in (no_bias, affine_bias):
        raise ValueError("actuators require fixed/affine gain and none/affine bias")
      if int(model.actuator_plugin[actuator]) >= 0:
        raise ValueError("actuator plugins are unsupported")
      transmission = int(model.actuator_trntype[actuator])
      gear = np.asarray(model.actuator_gear[actuator], dtype=np.float64)
      if np.any(gear[1:] != 0):
        raise ValueError("scalar actuator gear must use only its first component")
      target = int(model.actuator_trnid[actuator, 0])
      if transmission in (joint_trn, parent_trn):
        if target < 0 or target >= int(model.njnt) or int(model.jnt_type[target]) not in (hinge, slide):
          raise ValueError("actuator transmission must target a hinge or slide joint")
        qa, da = int(model.jnt_qposadr[target]), int(model.jnt_dofadr[target])
        length_map[actuator, qa] = gear[0]
        moment_map[actuator, da] = gear[0]
      elif transmission == tendon_trn:
        if target < 0 or target >= int(model.ntendon):
          raise ValueError("actuator transmission must target a valid tendon")
        if bool(model.tendon_actfrclimited[target]):
          raise ValueError("tendon-level actuator force limits are unsupported")
        start, count = int(model.tendon_adr[target]), int(model.tendon_num[target])
        if count <= 0 or any(int(t) != wrap_joint for t in model.wrap_type[start:start+count]):
          raise ValueError("actuator tendon must be a fixed tendon made only of joint wraps")
        for wrap in range(start, start+count):
          joint = int(model.wrap_objid[wrap])
          if joint < 0 or joint >= int(model.njnt) or int(model.jnt_type[joint]) not in (hinge, slide):
            raise ValueError("fixed tendon wraps must target hinge or slide joints")
          coefficient = gear[0] * float(model.wrap_prm[wrap])
          length_map[actuator, int(model.jnt_qposadr[joint])] += coefficient
          moment_map[actuator, int(model.jnt_dofadr[joint])] += coefficient
      else:
        raise ValueError("only scalar joint and fixed-joint-tendon transmissions are supported")

    self.length_map = _frozen(length_map, np.float32)
    self.moment_map = _frozen(moment_map, np.float32)
    gainprm = np.asarray(model.actuator_gainprm, dtype=np.float64)[:, :3]
    biasprm = np.asarray(model.actuator_biasprm, dtype=np.float64)[:, :3]
    bias_enabled = np.asarray(model.actuator_biastype == affine_bias, dtype=np.uint8)
    self.gaintype = _frozen(np.asarray(model.actuator_gaintype == affine, dtype=np.uint8), np.uint8)
    self.gainprm = _frozen(gainprm, np.float32)
    self.biasprm = _frozen(biasprm, np.float32)
    self.bias_enabled = _frozen(bias_enabled, np.uint8)
    self.ctrl_limited = _frozen(model.actuator_ctrllimited, np.uint8)
    self.ctrl_range = _frozen(model.actuator_ctrlrange, np.float32)
    self.force_limited = _frozen(model.actuator_forcelimited, np.uint8)
    self.force_range = _frozen(model.actuator_forcerange, np.float32)
    self.group = _frozen(model.actuator_group, np.int32)
    if np.any(self.group < 0) or np.any(self.group >= 31):
      raise ValueError("actuator groups must be in [0, 30]")
    for limited, ranges, label in (
        (self.ctrl_limited, self.ctrl_range, "control"),
        (self.force_limited, self.force_range, "force"),
    ):
      if np.any(limited & (ranges[:, 0] > ranges[:, 1])):
        raise ValueError(f"limited actuator {label} ranges must be ordered")
    self.disableflags = int(model.opt.disableflags)
    self.disableactuator = int(model.opt.disableactuator)

  @classmethod
  def from_model(cls, model):
    return cls(model)

  def transmission_state(self, qpos, qvel):
    """Compute actuator length, moment rows, and velocity on CPU."""
    qpos, qvel = self._validate_state(qpos, qvel)
    lengths = qpos @ self.length_map.T
    moments = np.broadcast_to(self.moment_map, (len(qpos), self.nu, self.nv))
    velocities = np.einsum("buv,bv->bu", moments, qvel)
    return lengths, moments, velocities

  def _validate_state(self, qpos, qvel):
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != self.nq or qvel.shape != (qpos.shape[0], self.nv) or not qpos.shape[0]:
      raise ValueError(f"qpos/qvel must have shapes (B, {self.nq}) and (B, {self.nv})")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
      raise ValueError("qpos and qvel must be finite")
    return qpos, qvel

  def generalized_force(self, qpos, qvel, ctrl):
    """Return MuJoCo-equivalent generalized actuator force on CPU."""
    qpos, qvel = self._validate_state(qpos, qvel)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if ctrl.shape != (len(qpos), self.nu) or not np.all(np.isfinite(ctrl)):
      raise ValueError(f"ctrl must be finite with shape (batch, {self.nu})")
    if not np.all(np.isfinite(ctrl.astype(np.float32))):
      raise ValueError("ctrl must be representable as float32")
    if self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
      return np.zeros((len(qpos), self.nv), dtype=np.float64)
    length, moment, velocity = self.transmission_state(qpos, qvel)
    controls = ctrl.copy()
    if not self.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL):
      controls = np.where(self.ctrl_limited[None, :], np.clip(controls, self.ctrl_range[:, 0], self.ctrl_range[:, 1]), controls)
    gain = self.gainprm[None, :, 0] + self.gaintype[None, :] * (
        self.gainprm[None, :, 1] * length + self.gainprm[None, :, 2] * velocity
    )
    force = gain * controls
    force += self.bias_enabled[None, :] * (
        self.biasprm[None, :, 0]
        + self.biasprm[None, :, 1] * length
        + self.biasprm[None, :, 2] * velocity
    )
    force = np.where(self.force_limited[None, :], np.clip(force, self.force_range[:, 0], self.force_range[:, 1]), force)
    enabled = ((self.disableactuator & (1 << self.group)) == 0)
    force *= enabled[None, :]
    return np.einsum("buv,bu->bv", moment, force)


class MetalTransmissions:
  """Native MPS stateless scalar actuator force stage."""

  def __init__(self, model, allow_inherited=True):
    self.model = model
    self._meta = TransmissionModel(model, allow_inherited=allow_inherited)
    self._last_force = None
    import torch
    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.scalar_transmission_force
    def tensor(values, dtype=torch.float32):
      npdtype = np.int32 if dtype == torch.int32 else np.float32
      arr = np.array(values, dtype=npdtype, order="C", copy=True)
      if arr.dtype == np.float32 and not np.all(np.isfinite(arr)):
        raise ValueError("transmission constants must be finite float32")
      if arr.size == 0:
        arr = np.zeros(1, dtype=npdtype)
      return torch.as_tensor(arr, dtype=dtype, device=self._device)
    self._length_map = tensor(self._meta.length_map.reshape(-1))
    self._moment_map = tensor(self._meta.moment_map.reshape(-1))
    self._gaintype = tensor(self._meta.gaintype, torch.int32)
    self._gainprm = tensor(self._meta.gainprm.reshape(-1))
    self._bias_enabled = tensor(self._meta.bias_enabled, torch.int32)
    self._biasprm = tensor(self._meta.biasprm.reshape(-1))
    self._ctrl_limited = tensor(self._meta.ctrl_limited, torch.int32)
    self._ctrl_range = tensor(self._meta.ctrl_range.reshape(-1))
    self._force_limited = tensor(self._meta.force_limited, torch.int32)
    self._force_range = tensor(self._meta.force_range.reshape(-1))
    self._group = tensor(self._meta.group, torch.int32)
    self._dims = tensor([
        self._meta.nq, self._meta.nv, self._meta.nu,
        int(bool(self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))),
        int(bool(self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))),
        self._meta.disableactuator,
    ], torch.int32)
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)

  def run_device(self, qpos, qvel, ctrl):
    """Return qfrc_actuator as an MPS tensor from persistent device state."""
    torch = self._torch
    for name, value in (("qpos", qpos), ("qvel", qvel), ("ctrl", ctrl)):
      if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"{name} must be a rank-2 torch.Tensor")
      if value.device.type != "mps" or value.dtype != torch.float32 or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS")
    batch = qpos.shape[0]
    if batch <= 0 or tuple(qpos.shape) != (batch, self._meta.nq) or tuple(qvel.shape) != (batch, self._meta.nv) or tuple(ctrl.shape) != (batch, self._meta.nu):
      raise ValueError("qpos, qvel and ctrl have invalid batch dimensions")
    dimensions = (self._meta.nq, self._meta.nv, self._meta.nu, batch)
    if any(value > _INT32_MAX for value in dimensions):
      raise ValueError("transmission dimensions exceed int32")
    elements = batch * max(1, self._meta.nq, self._meta.nv, self._meta.nu)
    if (
        elements > _UINT32_MAX
        or batch * self._meta.nu * self._meta.nv > _UINT32_MAX
        or batch * self._meta.nu * self._meta.nq > _UINT32_MAX
    ):
      raise ValueError("transmission kernel offsets exceed uint32")
    output = torch.empty((batch, self._meta.nv), dtype=torch.float32, device=self._device)
    force = torch.empty((batch, max(self._meta.nu, 1)), dtype=torch.float32, device=self._device)
    self._kernel(
        qpos.reshape(-1) if self._meta.nq else self._dummy,
        qvel.reshape(-1) if self._meta.nv else self._dummy,
        ctrl.reshape(-1) if self._meta.nu else self._dummy,
        self._length_map, self._moment_map, self._gaintype, self._gainprm,
        self._bias_enabled, self._biasprm, self._ctrl_limited, self._ctrl_range,
        self._force_limited, self._force_range, self._group, self._dims,
        output.reshape(-1) if self._meta.nv else self._dummy,
        force.reshape(-1) if self._meta.nu else self._dummy,
        threads=(batch,), group_size=(1,),
    )
    self._last_force = force
    return output
