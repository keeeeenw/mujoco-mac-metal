# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Stateless scalar actuator lowering and generalized-force evaluation."""

from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "transmissions.metal"
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


def transmission_workspace_sizes(meta, batch_size, *, validate=True):
  """Exact scalar-transmission tensor inventory for one prepared batch."""
  from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
  if (isinstance(batch_size, (bool, np.bool_))
      or not isinstance(batch_size, (int, np.integer))
      or int(batch_size) <= 0):
    raise ValueError("batch_size must be a positive integer")
  b = int(batch_size)
  nu, nq, nv = int(meta.nu), int(meta.nq), int(meta.nv)
  njnt = int(getattr(meta, "njnt", 0))
  sizes = {
      "transmission.length_map": max(nu * nq, 1),
      "transmission.moment_map": max(nu * nv, 1),
      "transmission.jnt_actgravcomp": max(njnt, 1),
      "transmission.jnt_dofadr": max(njnt, 1),
      "transmission.jnt_type": max(njnt, 1),
      "transmission.gravcomp_dims": 1,
      "transmission.gaintype": max(nu, 1),
      "transmission.gainprm": max(nu * 3, 1),
      "transmission.bias_enabled": max(nu, 1),
      "transmission.biasprm": max(nu * 3, 1),
      "transmission.ctrl_limited": max(nu, 1),
      "transmission.ctrl_range": max(nu * 2, 1),
      "transmission.force_limited": max(nu, 1),
      "transmission.force_range": max(nu * 2, 1),
      "transmission.group": max(nu, 1),
      "transmission.actuator_treeids": max(nu * 2, 1),
      "transmission.actuator_treenum": max(nu, 1),
      "transmission.dims": 8 + b,
      "transmission.cached_dims": 8,
      # cache_scalar_transmission_position has a dedicated ABI whose fourth
      # word is the world count; force dims[3] is ACTUATION-disable.
      "transmission.position_dims": 4 + b,
      "transmission.dummy_float": 1,
      "transmission.dummy_int": 1,
      "transmission.sleep_filter": 1,
      "transmission.position_length": b * max(nu, 1),
      "transmission.last_qfrc": b * nv,
      "transmission.last_force": b * max(nu, 1),
  }
  if validate:
    _validate_workspace_index_capacity(
        b, sizes, {"nq": nq, "nv": nv, "nu": nu})
    for name, elements in sizes.items():
      if elements > _INT32_MAX:
        raise ValueError(
            f"{name} exceeds the Metal int32 address range ({elements})")
  return sizes


def _validate_transmission_batch(meta, batch):
  if (isinstance(batch, (bool, np.bool_))
      or not isinstance(batch, (int, np.integer)) or int(batch) <= 0):
    raise ValueError("batch_size must be a positive integer")
  batch = int(batch)
  if batch > _INT32_MAX:
    raise ValueError("transmission batch exceeds int32")
  for name, elements in (
      ("position length", batch * max(int(meta.nu), 1)),
      ("qfrc", batch * max(int(meta.nv), 1)),
      ("qpos", batch * max(int(meta.nq), 1)),
      ("moment map", int(meta.nu) * int(meta.nv)),
      ("length map", int(meta.nu) * int(meta.nq)),
  ):
    if elements > _INT32_MAX:
      raise ValueError(f"transmission {name} exceeds signed int32 addressing")
  if batch * int(meta.nu) * int(meta.nv) > _UINT32_MAX:
    raise ValueError("transmission force-map product exceeds uint32 addressing")


def _frozen(values, dtype):
  array = np.array(values, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("actuator constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def actuator_awake_cpu(model, tree_awake):
  """Return the pinned MuJoCo 3.10 stateless-actuator sleep predicate."""
  awake = np.asarray(tree_awake, dtype=bool)
  if awake.ndim == 1:
    awake = awake[None, :]
  if awake.ndim != 2 or awake.shape[1] != max(int(model.ntree), 1):
    raise ValueError("tree_awake must have shape [batch, max(ntree, 1)]")
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  joint_tree = body_tree[np.asarray(model.jnt_bodyid, dtype=np.int32)]
  tendon_tree = np.asarray(model.tendon_treeid, dtype=np.int32).reshape(-1, 2)
  tendon_num = np.asarray(model.tendon_treenum, dtype=np.int32)
  trn = np.asarray(model.actuator_trntype, dtype=np.int32)
  trnid = np.asarray(model.actuator_trnid, dtype=np.int32)
  joint_types = {int(mujoco.mjtTrn.mjTRN_JOINT),
                 int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)}
  tendon_type = int(mujoco.mjtTrn.mjTRN_TENDON)
  result = np.ones((len(awake), len(trn)), dtype=bool)
  for actuator, kind in enumerate(trn):
    target = int(trnid[actuator, 0])
    if int(kind) in joint_types:
      tree = int(joint_tree[target])
      if tree >= 0:
        result[:, actuator] = awake[:, tree]
    elif int(kind) == tendon_type:
      count = int(tendon_num[target])
      if count in (1, 2):
        result[:, actuator] = np.any(awake[:, tendon_tree[target, :count]], axis=1)
      # Zero trees are static (awake); more than two is pinned always-awake.
  return result


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
    self.jnt_actgravcomp = _frozen(model.jnt_actgravcomp, np.int32)
    self.jnt_dofadr = _frozen(model.jnt_dofadr, np.int32)
    self.jnt_type = _frozen(model.jnt_type, np.int32)
    self.njnt = int(model.njnt)
    self.actuator_treeids = np.full((self.nu, 2), -1, dtype=np.int32)
    self.actuator_treenum = np.zeros(self.nu, dtype=np.int32)
    tree_pair = np.asarray(model.tendon_treeid, dtype=np.int32).reshape(-1, 2)
    tree_count = np.asarray(model.tendon_treenum, dtype=np.int32)
    for actuator, kind in enumerate(np.asarray(model.actuator_trntype)):
      target = int(model.actuator_trnid[actuator, 0])
      if int(kind) in (joint_trn, parent_trn):
        tree = int(model.body_treeid[int(model.jnt_bodyid[target])])
        if tree >= 0:
          self.actuator_treeids[actuator, 0] = tree
          self.actuator_treenum[actuator] = 1
      elif int(kind) == tendon_trn:
        count = int(tree_count[target])
        if count in (1, 2):
          self.actuator_treeids[actuator] = tree_pair[target]
          self.actuator_treenum[actuator] = count
        elif count > 2:
          self.actuator_treenum[actuator] = 3  # pinned always-awake case
    self.actuator_treeids = _frozen(self.actuator_treeids, np.int32)
    self.actuator_treenum = _frozen(self.actuator_treenum, np.int32)

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

  def generalized_force(self, qpos, qvel, ctrl, gravcomp=None):
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
    qfrc = np.einsum("buv,bu->bv", moment, force)
    if gravcomp is not None:
      gravcomp = np.asarray(gravcomp, dtype=np.float64)
      if gravcomp.shape != qfrc.shape or not np.all(np.isfinite(gravcomp)):
        raise ValueError("gravcomp must be finite with shape (batch, nv)")
      for joint in range(self.njnt):
        if not self.jnt_actgravcomp[joint]:
          continue
        dof = int(self.jnt_dofadr[joint])
        ndof = {0: 6, 1: 3, 2: 1, 3: 1}[int(self.jnt_type[joint])]
        qfrc[:, dof:dof + ndof] += gravcomp[:, dof:dof + ndof]
    return qfrc


class MetalTransmissions:
  """Native MPS stateless scalar actuator force stage."""

  def __init__(self, model, allow_inherited=True, *, batch_size=None):
    self.model = model
    self._meta = TransmissionModel(model, allow_inherited=allow_inherited)
    self._last_force = None
    self._last_qfrc = None
    if (batch_size is not None
        and (isinstance(batch_size, (bool, np.bool_))
             or not isinstance(batch_size, (int, np.integer))
             or int(batch_size) <= 0)):
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = None if batch_size is None else int(batch_size)
    # Static constants are allocated even for legacy unprepared callers, so
    # validate their addressability before importing/initializing torch too.
    transmission_workspace_sizes(self._meta, self.batch_size or 1)
    _validate_transmission_batch(self._meta, self.batch_size or 1)
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
    self._jnt_actgravcomp = tensor(self._meta.jnt_actgravcomp, torch.int32)
    self._jnt_dofadr = tensor(self._meta.jnt_dofadr, torch.int32)
    self._jnt_type = tensor(self._meta.jnt_type, torch.int32)
    self._gravcomp_dims = tensor([self._meta.njnt], torch.int32)
    self._gaintype = tensor(self._meta.gaintype, torch.int32)
    self._gainprm = tensor(self._meta.gainprm.reshape(-1))
    self._bias_enabled = tensor(self._meta.bias_enabled, torch.int32)
    self._biasprm = tensor(self._meta.biasprm.reshape(-1))
    self._ctrl_limited = tensor(self._meta.ctrl_limited, torch.int32)
    self._ctrl_range = tensor(self._meta.ctrl_range.reshape(-1))
    self._force_limited = tensor(self._meta.force_limited, torch.int32)
    self._force_range = tensor(self._meta.force_range.reshape(-1))
    self._group = tensor(self._meta.group, torch.int32)
    self._actuator_treeids = tensor(
        self._meta.actuator_treeids.reshape(-1), torch.int32)
    self._actuator_treenum = tensor(self._meta.actuator_treenum, torch.int32)
    dims_base = [
        self._meta.nq, self._meta.nv, self._meta.nu,
        int(bool(self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))),
        int(bool(self._meta.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))),
        self._meta.disableactuator,
        int(model.ntree),
        0,
    ]
    self._dims_base = dims_base
    self._dims = tensor(dims_base + [1] * (self.batch_size or 1), torch.int32)
    self._cached_dims = self._dims.clone()
    self._cached_dims[7] = 1
    self._position_dims = tensor([
        self._meta.nq, self._meta.nv, self._meta.nu,
        0 if self.batch_size is None else self.batch_size,
    ] + [1] * (self.batch_size or 1), torch.int32)
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._dummy_int = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._sleep_filter = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._position_length = (
        torch.empty((self.batch_size, max(self._meta.nu, 1)),
                    dtype=torch.float32, device=self._device)
        if self.batch_size is not None else None)
    self._position_context_epoch = 0
    self._position_kernel = self._library.cache_scalar_transmission_position

  def _ensure_batch(self, batch, *, position_context=False):
    if self.batch_size is not None and int(batch) != self.batch_size:
      raise ValueError("transmission batch differs from its prepared workspace")
    _validate_transmission_batch(self._meta, batch)
    transmission_workspace_sizes(self._meta, batch)
    if (position_context
        and (self._position_length is None
             or tuple(self._position_length.shape)
             != (int(batch), max(self._meta.nu, 1)))):
      self._position_length = self._torch.empty(
          (int(batch), max(self._meta.nu, 1)), dtype=self._torch.float32,
          device=self._device)
    if self._dims.numel() != 8 + int(batch):
      self._dims = self._torch.tensor(
          self._dims_base + [1] * int(batch), dtype=self._torch.int32,
          device=self._device)
      cached = self._dims.clone()
      cached[7] = 1
      self._cached_dims = cached
    if self._position_dims.numel() != 4 + int(batch):
      self._position_dims = self._torch.tensor(
          [self._meta.nq, self._meta.nv, self._meta.nu, int(batch)]
          + [1] * int(batch), dtype=self._torch.int32, device=self._device)

  def capture_position_context(self, qpos, *, world_mask=None):
    """Capture X-stage actuator lengths and the fixed model moment map."""
    torch = self._torch
    if (not isinstance(qpos, torch.Tensor) or qpos.ndim != 2
        or tuple(qpos.shape[1:]) != (self._meta.nq,)
        or qpos.dtype != torch.float32 or qpos.device.type != "mps"
        or not qpos.is_contiguous()):
      raise ValueError("qpos must be contiguous float32 MPS [batch,nq]")
    batch = int(qpos.shape[0])
    if batch <= 0:
      raise ValueError("batch must be positive")
    self._ensure_batch(batch, position_context=True)
    self._copy_mask(self._position_dims, 4, world_mask)
    if self._meta.nu:
      # The position kernel's ABI is [nq,nv,nu,batch]. Do not pass `_dims`:
      # that force-stage ABI stores mjDSBL_ACTUATION at index 3.
      self._position_dims[3] = batch
      self._position_kernel(
          qpos.reshape(-1) if self._meta.nq else self._dummy,
          self._length_map, self._position_length.reshape(-1),
          self._position_dims,
          threads=(batch,), group_size=(1,))
    self._position_context_epoch += 1
    return {"length": self._position_length, "moment_map": self._moment_map,
            "_owner": self, "_epoch": self._position_context_epoch}

  def invalidate_position_context(self):
    """Invalidate a cached actuator-length context after state replacement."""
    self._position_context_epoch += 1

  def run_device(self, qpos, qvel, ctrl, awake_lists=None, *,
                 position_context=None, world_mask=None, gravcomp=None):
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
    self._ensure_batch(batch)
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
    if (self._last_qfrc is None
        or tuple(self._last_qfrc.shape) != (batch, self._meta.nv)):
      self._last_qfrc = torch.zeros(
          (batch, self._meta.nv), dtype=torch.float32, device=self._device)
      self._last_force = torch.zeros(
          (batch, max(self._meta.nu, 1)), dtype=torch.float32,
          device=self._device)
    output, force = self._last_qfrc, self._last_force
    if gravcomp is not None:
      if (not isinstance(gravcomp, torch.Tensor)
          or gravcomp.device.type != "mps" or gravcomp.dtype != torch.float32
          or not gravcomp.is_contiguous()
          or tuple(gravcomp.shape) != (batch, max(self._meta.nv, 1))):
        raise ValueError(
            "gravcomp must be contiguous float32 MPS [batch, max(nv,1)]")
      gravcomp_values = gravcomp.reshape(-1)
    else:
      gravcomp_values = self._dummy
    sleep_filter = awake_lists is not None and awake_lists.get("tree_awake") is not None
    if sleep_filter:
      tree_awake = awake_lists["tree_awake"]
      dof_ids, dof_count = awake_lists["dof_ids"], awake_lists.get("counts")
      for name, value in (("tree_awake", tree_awake), ("dof_ids", dof_ids)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int32
            or value.device.type != "mps" or not value.is_contiguous()):
          raise ValueError(f"awake_lists.{name} must be contiguous MPS int32")
      ntree = int(self._meta.model.ntree)
      if tuple(tree_awake.shape) != (batch, max(ntree, 1)):
        raise ValueError("awake_lists.tree_awake has an invalid shape")
      if tuple(dof_ids.shape) != (batch, max(self._meta.nv, 1)):
        raise ValueError("awake_lists.dof_ids has an invalid shape")
      if (not isinstance(dof_count, torch.Tensor) or dof_count.dtype != torch.int32
          or dof_count.device.type != "mps" or tuple(dof_count.shape) != (batch, 3)
          or not dof_count.is_contiguous()):
        raise ValueError("awake_lists.counts must be contiguous MPS int32 [batch,3]")
    else:
      tree_awake = dof_ids = dof_count = self._dummy_int
    position_length = self._dummy
    dims = self._dims
    if position_context is not None:
      if (not isinstance(position_context, dict)
          or position_context.get("_owner") is not self
          or position_context.get("_epoch") != self._position_context_epoch
          or position_context.get("length") is not self._position_length
          or position_context.get("moment_map") is not self._moment_map):
        raise ValueError("position_context is stale or belongs to another workspace")
      position_length = self._position_length.reshape(-1)
      dims = self._cached_dims
    self._copy_mask(dims, 8, world_mask)
    self._sleep_filter.fill_(int(sleep_filter))
    self._kernel(
        qpos.reshape(-1) if self._meta.nq else self._dummy,
        qvel.reshape(-1) if self._meta.nv else self._dummy,
        ctrl.reshape(-1) if self._meta.nu else self._dummy,
        self._length_map, self._moment_map, self._gaintype, self._gainprm,
        self._bias_enabled, self._biasprm, self._ctrl_limited, self._ctrl_range,
        self._force_limited, self._force_range, self._group, dims,
        self._actuator_treeids, self._actuator_treenum,
        tree_awake.reshape(-1) if sleep_filter else self._dummy_int,
        dof_ids.reshape(-1) if sleep_filter else self._dummy_int,
        dof_count, self._sleep_filter,
        output.reshape(-1) if self._meta.nv else self._dummy,
        force.reshape(-1) if self._meta.nu else self._dummy,
        position_length,
        gravcomp_values,
        self._jnt_actgravcomp,
        self._jnt_dofadr,
        self._jnt_type,
        self._gravcomp_dims,
        threads=(batch,), group_size=(1,),
    )
    self._last_force = force
    return output

  def _copy_mask(self, dims, offset, world_mask):
    if world_mask is None:
      dims[offset:].fill_(1)
      return
    torch = self._torch
    if (not isinstance(world_mask, torch.Tensor)
        or world_mask.dtype != torch.int32 or world_mask.device.type != "mps"
        or tuple(world_mask.shape) != (dims.numel() - offset,)
        or not world_mask.is_contiguous()):
      raise ValueError("world_mask must be contiguous int32 MPS with shape (batch_size,)")
    dims[offset:].copy_(world_mask)
