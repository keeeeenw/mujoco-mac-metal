# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Narrow lowering and native force mapping for scalar joint motors.

Accepted models use MuJoCo 3.10.0, have no actuator activation state or
plugins, and contain only fixed-gain, no-bias, no-dynamics actuators with a
joint transmission targeting one hinge or slide DOF. Joint/tendon actuator
armature and damping, joint-level actuator-force limits, and all other
transmissions/gains/biases/dynamics are rejected. Control clipping (unless
``mjDSBL_CLAMPCTRL`` is set), actuator force clipping, global actuation disable,
and actuator-group disable are represented. This primitive is connected to the explicit
``contact_free_motor_euler_v1`` simulation profile. Other simulation profiles
continue to reject actuators.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "actuation.metal"


def _frozen(value, dtype):
  array = np.ascontiguousarray(value, dtype=dtype)
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ScalarMotorModel:
  """Immutable actuator-to-DOF map lowered from one compiled MjModel."""

  nv: int
  nu: int
  dof: np.ndarray
  gear: np.ndarray
  gain: np.ndarray
  ctrl_limited: np.ndarray
  ctrl_range: np.ndarray
  force_limited: np.ndarray
  force_range: np.ndarray
  actuator_group: np.ndarray
  disableflags: int
  disableactuator: int

  def __post_init__(self):
    """Validate direct construction too and own immutable metadata arrays."""
    for name in ("nv", "nu"):
      value = getattr(self, name)
      if (
          isinstance(value, (bool, np.bool_))
          or not isinstance(value, (int, np.integer))
          or not 0 <= value <= (1 << 31) - 1
      ):
        raise ValueError(f"{name} must be a nonnegative int32 dimension")
    specs = {
        "dof": ((self.nu,), np.int32),
        "gear": ((self.nu,), np.float32),
        "gain": ((self.nu,), np.float32),
        "ctrl_limited": ((self.nu,), np.uint8),
        "ctrl_range": ((self.nu, 2), np.float32),
        "force_limited": ((self.nu,), np.uint8),
        "force_range": ((self.nu, 2), np.float32),
        "actuator_group": ((self.nu,), np.int32),
    }
    for name, (shape, dtype) in specs.items():
      value = np.asarray(getattr(self, name))
      if value.shape != shape or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric with shape {shape}")
      if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
      if (
          dtype != np.float32
          and value.dtype.kind == "f"
          and not np.all(value == np.floor(value))
      ):
        raise ValueError(f"{name} must contain integer values")
      if dtype == np.uint8 and np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must contain only zero or one")
      if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if np.any(value < limits.min) or np.any(value > limits.max):
          raise ValueError(f"{name} exceeds its integer storage range")
      with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(value, dtype=dtype)
      if dtype == np.float32 and not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must be finite and representable as float32")
      if dtype != np.float32 and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
      object.__setattr__(self, name, _frozen(converted, dtype))
    if np.any(self.dof < 0) or np.any(self.dof >= self.nv):
      raise ValueError("dof indices must be in [0, nv)")
    if np.any(self.ctrl_limited > 1) or np.any(self.force_limited > 1):
      raise ValueError("limit flags must contain only zero or one")
    if np.any(self.actuator_group < 0) or np.any(self.actuator_group >= 31):
      raise ValueError("actuator groups must be in [0, 30]")
    for limited, ranges, label in (
        (self.ctrl_limited, self.ctrl_range, "control"),
        (self.force_limited, self.force_range, "force"),
    ):
      if np.any(limited & (ranges[:, 0] > ranges[:, 1])):
        raise ValueError(f"limited actuator {label} ranges must be ordered")

  @classmethod
  def from_model(cls, model):
    """Lower supported scalar motors or reject before allocating device state."""
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(
          f"scalar motor lowering requires MuJoCo 3.10.0; found {mujoco.__version__}"
      )
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")

    nu = int(model.nu)
    nv = int(model.nv)
    if int(model.nplugin):
      raise ValueError("actuator plugins are unsupported")

    # Reject unsupported generalized actuator terms, including terms that
    # could otherwise be silently added by the smooth dynamics stage.
    if np.any(np.asarray(model.actuator_armature) != 0):
      raise ValueError("actuator armature is unsupported")
    if np.any(np.asarray(model.actuator_damping) != 0) or np.any(
        np.asarray(model.actuator_dampingpoly) != 0
    ):
      raise ValueError("actuator damping is unsupported")
    if np.any(np.asarray(model.jnt_actfrclimited)):
      raise ValueError("joint-level actuator force limits are unsupported")

    trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    gain_fixed = int(mujoco.mjtGain.mjGAIN_FIXED)
    bias_none = int(mujoco.mjtBias.mjBIAS_NONE)
    dyn_none = int(mujoco.mjtDyn.mjDYN_NONE)
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    dof = np.empty(nu, dtype=np.int32)
    for actuator in range(nu):
      if int(model.actuator_plugin[actuator]) >= 0:
        raise ValueError("actuator plugins are unsupported")
      if (
          int(model.actuator_trntype[actuator]) != trn_joint
          or int(model.actuator_gaintype[actuator]) != gain_fixed
          or int(model.actuator_biastype[actuator]) != bias_none
          or int(model.actuator_dyntype[actuator]) != dyn_none
          or int(model.actuator_actnum[actuator]) != 0
      ):
        raise ValueError(
            "only fixed-gain, no-bias, no-dynamics joint motors are supported"
        )
      joint = int(model.actuator_trnid[actuator, 0])
      if joint < 0 or joint >= int(model.njnt):
        raise ValueError("actuator must target exactly one valid joint")
      if int(model.jnt_type[joint]) not in (hinge, slide):
        raise ValueError(
            "actuator joint transmission must target hinge or slide"
        )
      dof[actuator] = int(model.jnt_dofadr[joint])

    gear6 = np.asarray(model.actuator_gear, dtype=np.float64).reshape(nu, 6)
    gear = gear6[:, 0].copy()
    gain = (
        np.asarray(model.actuator_gainprm, dtype=np.float64).reshape(nu, -1)[
            :, 0
        ]
        if nu
        else np.empty((0,), dtype=np.float64)
    )
    ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64).reshape(
        nu, 2
    )
    force_range = np.asarray(
        model.actuator_forcerange, dtype=np.float64
    ).reshape(nu, 2)
    ctrl_limited = np.asarray(model.actuator_ctrllimited, dtype=np.uint8).copy()
    force_limited = np.asarray(
        model.actuator_forcelimited, dtype=np.uint8
    ).copy()
    groups = np.asarray(model.actuator_group, dtype=np.int32).copy()

    for name, values in (
        ("gear", gear),
        ("fixed gain", gain),
        ("control range", ctrl_range),
        ("force range", force_range),
    ):
      if not np.all(np.isfinite(values)):
        raise ValueError(f"actuator {name} must be finite")
    if not np.all(np.isfinite(gear.astype(np.float32))) or not np.all(
        np.isfinite(gain.astype(np.float32))
    ):
      raise ValueError(
          "actuator gear and gain must be representable as float32"
      )
    if np.any(ctrl_limited & (ctrl_range[:, 0] > ctrl_range[:, 1])):
      raise ValueError("limited actuator control ranges must be ordered")
    if np.any(force_limited & (force_range[:, 0] > force_range[:, 1])):
      raise ValueError("limited actuator force ranges must be ordered")
    if np.any(groups < 0) or np.any(groups >= 31):
      raise ValueError("actuator groups must be in [0, 30]")

    return cls(
        nv=nv,
        nu=nu,
        dof=_frozen(dof, np.int32),
        gear=_frozen(gear, np.float32),
        gain=_frozen(gain, np.float32),
        ctrl_limited=_frozen(ctrl_limited, np.uint8),
        ctrl_range=_frozen(ctrl_range, np.float32),
        force_limited=_frozen(force_limited, np.uint8),
        force_range=_frozen(force_range, np.float32),
        actuator_group=_frozen(groups, np.int32),
        disableflags=int(model.opt.disableflags),
        disableactuator=int(model.opt.disableactuator),
    )

  def generalized_force(self, ctrl):
    """CPU reference mapping for finite controls shaped ``[B, nu]``."""
    values = np.asarray(ctrl, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != self.nu or values.shape[0] <= 0:
      raise ValueError(
          f"ctrl must have shape (batch, {self.nu}) with batch > 0"
      )
    if not np.all(np.isfinite(values)):
      raise ValueError("ctrl must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
      values32 = values.astype(np.float32)
    if not np.all(np.isfinite(values32)):
      raise ValueError("ctrl cannot be represented as finite float32")
    return _generalized_force_numpy(self, values32)


def _generalized_force_numpy(model, ctrl):
  batch = ctrl.shape[0]
  if model.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
    return np.zeros((batch, model.nv), dtype=np.float32)
  ctrl = ctrl.copy()
  if not (model.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)):
    for actuator in range(model.nu):
      if model.ctrl_limited[actuator]:
        ctrl[:, actuator] = np.clip(
            ctrl[:, actuator], *model.ctrl_range[actuator]
        )
  with np.errstate(over="ignore", invalid="ignore"):
    force = ctrl * model.gain[None, :]
  for actuator in range(model.nu):
    group_disabled = bool(
        model.disableactuator & (1 << model.actuator_group[actuator])
    )
    if group_disabled:
      force[:, actuator] = 0
    elif model.force_limited[actuator]:
      force[:, actuator] = np.clip(
          force[:, actuator], *model.force_range[actuator]
      )
  result = np.zeros((batch, model.nv), dtype=np.float32)
  with np.errstate(over="ignore", invalid="ignore"):
    for actuator in range(model.nu):
      result[:, model.dof[actuator]] += (
          model.gear[actuator] * force[:, actuator]
      )
  if not np.all(np.isfinite(result)):
    raise ValueError("actuator controls produce nonfinite generalized force")
  return result


class MetalScalarMotorForce:
  """Batched scalar-joint motor force map; this is not a simulation step.

  Inputs to ``run_device`` are contiguous float32 MPS controls ``[B, nu]``.
  The returned generalized force is a borrowed MPS tensor ``[B, nv]`` that is
  overwritten by the next call. Device controls are checked in the shader;
  any nonfinite control makes that world's full force row NaN so the caller can
  fail it before committing state, without a CPU readback in this primitive.
  """

  def __init__(self, model: ScalarMotorModel, batch_size: int = 1):
    if not isinstance(model, ScalarMotorModel):
      raise TypeError("model must be a ScalarMotorModel")
    self.model = model
    self._batch_size = _positive_batch(batch_size)
    self._check_capacity(self._batch_size)
    import torch

    if not torch.backends.mps.is_available():
      raise RuntimeError(
          "MetalScalarMotorForce requires an available MPS device"
      )
    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.scalar_motor_force
    self._dof = self._device_array(model.dof, torch.int32)
    self._gear = self._device_array(model.gear, torch.float32)
    self._gain = self._device_array(model.gain, torch.float32)
    self._ctrl_limited = self._device_array(model.ctrl_limited, torch.int32)
    self._ctrl_range = self._device_array(model.ctrl_range, torch.float32)
    self._force_limited = self._device_array(model.force_limited, torch.int32)
    self._force_range = self._device_array(model.force_range, torch.float32)
    self._groups = self._device_array(model.actuator_group, torch.int32)
    self._dims = torch.tensor(
        [
            model.nu,
            model.nv,
            self._batch_size,
            int(
                bool(
                    model.disableflags
                    & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
                )
            ),
            int(
                bool(
                    model.disableflags
                    & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
                )
            ),
            model.disableactuator,
        ],
        dtype=torch.int32,
        device=self._device,
    )
    self._workspace = None
    self.prepare_workspace(self._batch_size)

  def prepare_workspace(self, batch_size):
    self._batch_size = _positive_batch(batch_size)
    self._check_capacity(self._batch_size)
    self._workspace = self._torch.empty(
        (self._batch_size, max(self.model.nv, 1)),
        dtype=self._torch.float32,
        device=self._device,
    )
    self._empty_ctrl = (
        self._torch.empty((1,), dtype=self._torch.float32, device=self._device)
        if self.model.nu == 0
        else None
    )
    self._dims[2] = self._batch_size

  def _check_capacity(self, batch):
    maximum = (2**31 - 1) // max(1, self.model.nv, self.model.nu)
    if batch > maximum:
      raise ValueError("batch exceeds int32 kernel index capacity")

  def _device_array(self, value, dtype):
    numpy_dtype = np.int32 if dtype == self._torch.int32 else np.float32
    array = np.asarray(value, dtype=numpy_dtype)
    if array.size == 0:
      array = np.zeros((1,), dtype=array.dtype)
    return self._torch.from_numpy(np.ascontiguousarray(array.copy())).to(
        self._device
    )

  def run_device(self, ctrl):
    """Compute ``qfrc_actuator = moment.T @ actuator_force`` on MPS."""
    torch = self._torch
    if not isinstance(ctrl, torch.Tensor) or ctrl.ndim != 2:
      raise TypeError("ctrl must be a rank-2 torch.Tensor")
    batch = ctrl.shape[0]
    if batch <= 0 or ctrl.shape[1] != self.model.nu:
      raise ValueError(
          f"ctrl must have shape (batch, {self.model.nu}) with batch > 0"
      )
    if ctrl.dtype != torch.float32 or not ctrl.is_contiguous():
      raise ValueError("ctrl must be contiguous float32")
    if batch != self._batch_size:
      raise ValueError("call prepare_workspace(batch_size) before this batch")
    if ctrl.device.type != "mps":
      raise ValueError("ctrl must be on the MPS device")
    self._check_capacity(batch)
    device_ctrl = self._empty_ctrl if self.model.nu == 0 else ctrl.reshape(-1)
    self._kernel(
        device_ctrl,
        self._dof,
        self._gear,
        self._gain,
        self._ctrl_limited,
        self._ctrl_range.reshape(-1),
        self._force_limited,
        self._force_range.reshape(-1),
        self._groups,
        self._dims,
        self._workspace.reshape(-1),
        threads=(batch,),
        group_size=(1,),
    )
    return self._workspace[:, : self.model.nv]


def _positive_batch(value):
  if (
      isinstance(value, (bool, np.bool_))
      or not isinstance(value, (int, np.integer))
      or value <= 0
  ):
    raise ValueError("batch_size must be a positive integer")
  return int(value)
