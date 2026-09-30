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

"""Owned batched device state and host checkpoint/reset boundaries."""

from dataclasses import dataclass
import hashlib
import math

import mujoco
import numpy as np

from mujoco_metal.lifecycle import _fingerprint
from mujoco_metal.model import ModelDescriptor
from mujoco_metal.model import load_model
from mujoco_metal.model import snapshot_descriptor
from mujoco_metal.stepping import SteppingProfile
from mujoco_metal.stepping import TARGET_MUJOCO_VERSION
from mujoco_metal.stepping import validate_stepping_profile


def _freeze_float32(value, shape, name):
  array = np.asarray(value)
  if array.shape != shape or array.dtype.kind not in "fiu":
    raise ValueError(f"{name} must be numeric with shape {shape}")
  with np.errstate(over="ignore", under="ignore", invalid="ignore"):
    array = np.asarray(array, dtype=np.float32, order="C")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must be finite and representable as float32")
  return np.frombuffer(array.tobytes(), dtype=np.float32).reshape(shape)


def _state_fingerprint(profile):
  digest = hashlib.sha256(b"mujoco-metal-device-state-v1\0")
  for value in (
      TARGET_MUJOCO_VERSION,
      profile.name,
      profile.model_fingerprint,
      profile.descriptor_fingerprint,
      repr(float(profile.timestep)),
      repr(profile.nq),
      repr(profile.nv),
      repr(profile.joint_types),
      repr(profile.supported),
      repr(profile.irrelevant),
      repr(profile.rejected),
      repr(profile.passive_damping_enabled),
      repr(profile.implicit_euler_damping),
  ):
    digest.update(value.encode("utf-8"))
    digest.update(b"\0")
  return digest.hexdigest()


@dataclass(frozen=True)
class StateSnapshot:
  """Detached float32 checkpoint tied to one model/profile and batch size."""

  model_fingerprint: str
  profile_fingerprint: str
  timestep: float
  nq: int
  nv: int
  batch_size: int
  qpos: np.ndarray
  qvel: np.ndarray
  qacc: np.ndarray
  time: np.ndarray
  status: np.ndarray
  schema_version: int = 1
  neq: int = 0
  eq_active: np.ndarray | None = None
  nmocap: int = 0
  mpos: np.ndarray | None = None
  mquat: np.ndarray | None = None
  nact: int = 0
  act: np.ndarray | None = None

  def __post_init__(self):
    if self.schema_version not in (1, 2, 3, 4):
      raise ValueError("unsupported state snapshot schema")
    if self.batch_size <= 0 or self.nq < 0 or self.nv < 0 or self.neq < 0:
      raise ValueError("invalid state snapshot dimensions")
    if not math.isfinite(float(self.timestep)) or self.timestep <= 0:
      raise ValueError("snapshot timestep must be finite and positive")
    batch = self.batch_size
    object.__setattr__(
        self, "qpos", _freeze_float32(self.qpos, (batch, self.nq), "qpos")
    )
    object.__setattr__(
        self, "qvel", _freeze_float32(self.qvel, (batch, self.nv), "qvel")
    )
    object.__setattr__(
        self, "qacc", _freeze_float32(self.qacc, (batch, self.nv), "qacc")
    )
    object.__setattr__(
        self, "time", _freeze_float32(self.time, (batch,), "time")
    )
    raw_status = np.asarray(self.status)
    if raw_status.shape != (batch,) or raw_status.dtype.kind not in "iu":
      raise ValueError("status must be an integer vector matching batch size")
    if np.any(raw_status < np.iinfo(np.int32).min) or np.any(
        raw_status > np.iinfo(np.int32).max
    ):
      raise ValueError("status values must fit in int32")
    status = np.asarray(raw_status, dtype=np.int32, order="C")
    frozen = np.frombuffer(status.tobytes(), dtype=np.int32).reshape((batch,))
    object.__setattr__(self, "status", frozen)
    # Equality activity (schema 2+). Version 1 never carries activity and must
    # never be silently upgraded for models with equalities.
    if self.schema_version == 1:
      if self.neq != 0 or self.eq_active is not None:
        raise ValueError("schema 1 snapshots must not carry equality activity")
      if self.nmocap != 0 or self.mpos is not None or self.mquat is not None:
        raise ValueError("schema 1 snapshots must not carry mocap poses")
      if self.nact != 0 or self.act is not None:
        raise ValueError("schema 1 snapshots must not carry activation state")
      object.__setattr__(self, "neq", 0)
      object.__setattr__(self, "eq_active", None)
      object.__setattr__(self, "nmocap", 0)
      object.__setattr__(self, "mpos", None)
      object.__setattr__(self, "mquat", None)
      object.__setattr__(self, "nact", 0)
      object.__setattr__(self, "act", None)
    else:
      if self.neq < 0:
        raise ValueError("invalid equality dimensions")
      if self.neq == 0:
        if self.eq_active is not None:
          raise ValueError("snapshots without equalities must not carry activity")
        object.__setattr__(self, "eq_active", None)
      else:
        raw_eq = np.asarray(self.eq_active) if self.eq_active is not None else None
        if raw_eq is None or raw_eq.shape != (batch, self.neq) or raw_eq.dtype.kind not in "iub":
          raise ValueError(f"eq_active must be boolean/integer with shape ({batch}, {self.neq})")
        eq = np.asarray(raw_eq, dtype=np.int32, order="C")
        if np.any((eq != 0) & (eq != 1)):
          raise ValueError("eq_active values must be 0 or 1")
        frozen_eq = np.frombuffer(eq.tobytes(), dtype=np.int32).reshape((batch, self.neq))
        object.__setattr__(self, "eq_active", frozen_eq)
      if self.schema_version == 2:
        if self.nmocap != 0 or self.mpos is not None or self.mquat is not None:
          raise ValueError("schema 2 snapshots must not carry mocap poses")
        object.__setattr__(self, "nmocap", 0)
        object.__setattr__(self, "mpos", None)
        object.__setattr__(self, "mquat", None)
        if self.nact != 0 or self.act is not None:
          raise ValueError("schema 2 snapshots must not carry activation state")
        object.__setattr__(self, "nact", 0)
        object.__setattr__(self, "act", None)
      else:
        if self.nmocap < 0:
          raise ValueError("invalid mocap dimensions")
        if self.nmocap == 0:
          if self.mpos is not None or self.mquat is not None:
            raise ValueError("snapshots without mocap bodies must not carry mocap poses")
          object.__setattr__(self, "mpos", None)
          object.__setattr__(self, "mquat", None)
        else:
          object.__setattr__(
              self, "mpos", _freeze_float32(self.mpos, (batch, self.nmocap, 3), "mpos")
          )
          raw_quat = np.asarray(self.mquat)
          if raw_quat.shape != (batch, self.nmocap, 4) or raw_quat.dtype.kind not in "fiu":
            raise ValueError(
                f"mquat must be numeric with shape ({batch}, {self.nmocap}, 4)"
            )
          with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            quat = np.asarray(raw_quat, dtype=np.float32, order="C")
          if not np.all(np.isfinite(quat)):
            raise ValueError("mquat must be finite and representable as float32")
          norms = np.linalg.norm(quat.astype(np.float64).reshape(-1, 4), axis=1)
          if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1) > 1e-5):
            raise ValueError("mquat must hold unit quaternions")
          object.__setattr__(
              self, "mquat",
              np.frombuffer(quat.tobytes(), dtype=np.float32).reshape(
                  (batch, self.nmocap, 4)),
          )
      # Activation state (schema 4). Schemas <4 never carry it.
      if self.schema_version < 4:
        if self.nact != 0 or self.act is not None:
          raise ValueError(f"schema {self.schema_version} snapshots must not carry activation state")
        object.__setattr__(self, "nact", 0)
        object.__setattr__(self, "act", None)
      else:
        if self.nact < 0:
          raise ValueError("invalid activation dimensions")
        if self.nact == 0:
          if self.act is not None:
            raise ValueError("snapshots without activation state must not carry act")
          object.__setattr__(self, "act", None)
        else:
          object.__setattr__(
              self, "act", _freeze_float32(self.act, (batch, self.nact), "act")
          )


class DeviceState:
  """Own persistent batched state tensors for one validated stepping profile.

  The default device is MPS. ``device='cpu'`` is provided for host-only state
  lifecycle verification; it does not enable CPU physics stepping. Device time
  uses float32, so its resolution decreases as elapsed time grows. Seeded
  randomization may be prepared as host arrays and passed to ``reset`` outside
  the step loop.
  """

  def __init__(
      self,
      model,
      profile: SteppingProfile,
      batch_size: int,
      qpos=None,
      qvel=None,
      device="mps",
  ):
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    if not isinstance(profile, SteppingProfile):
      raise TypeError("profile must be a validated SteppingProfile")
    if mujoco.__version__ != TARGET_MUJOCO_VERSION:
      raise RuntimeError(
          f"requires MuJoCo {TARGET_MUJOCO_VERSION}; found {mujoco.__version__}"
      )
    if isinstance(model, mujoco.MjModel):
      validated = validate_stepping_profile(
          model, profile.timestep, profile=profile.name
      )
      descriptor = load_model(model)
      if validated != profile:
        raise ValueError("profile does not match the compiled model")
      model_fingerprint = validated.model_fingerprint
    elif isinstance(model, ModelDescriptor):
      descriptor = model
      model_fingerprint = profile.model_fingerprint
      if _fingerprint(descriptor) != profile.descriptor_fingerprint:
        raise ValueError("profile does not match the immutable model descriptor")
      if (descriptor.nq, descriptor.nv) != (profile.nq, profile.nv):
        raise ValueError("profile dimensions do not match model descriptor")
      if tuple(int(x) for x in descriptor.jnt_type) != profile.joint_types:
        raise ValueError("profile joint types do not match model descriptor")
      descriptor = snapshot_descriptor(descriptor)
    else:
      raise TypeError("model must be an MjModel or immutable ModelDescriptor")

    self._model = descriptor
    self.profile = profile
    self.batch_size = int(batch_size)
    self._model_fingerprint = model_fingerprint
    self._profile_fingerprint = _state_fingerprint(profile)

    # Equality activity: one logical boolean per equality per environment,
    # initialized from compiled eq_active0. Models without equalities (including
    # ModelDescriptor inputs, which carry no equality metadata) own no mask.
    if isinstance(model, mujoco.MjModel):
      self._neq = int(model.neq)
      if self._neq > 0:
        eq0 = np.asarray(model.eq_active0).astype(np.int32, copy=True).reshape((self._neq,))
        if eq0.shape != (self._neq,) or np.any((eq0 != 0) & (eq0 != 1)):
          raise ValueError("compiled eq_active0 must be 0/1")
      else:
        eq0 = np.zeros(0, dtype=np.int32)
    else:
      self._neq = 0
      eq0 = np.zeros(0, dtype=np.int32)
    self._eq_active0 = np.frombuffer(eq0.tobytes(), dtype=np.int32).reshape(eq0.shape)

    # Prescribed mocap poses: one position/quaternion per mocap body per
    # environment, initialized from the compiled reference body frames (the
    # pinned `mj_resetData` defaults). Descriptor inputs carry no mocap
    # metadata beyond counts and also start from reference frames.
    if isinstance(model, mujoco.MjModel):
      self._nmocap = int(model.nmocap)
      mocapid = np.asarray(model.body_mocapid)
      ref_pos = np.asarray(model.body_pos)
      ref_quat = np.asarray(model.body_quat)
    else:
      self._nmocap = int(descriptor.nmocap)
      mocapid = np.asarray(descriptor.body_mocapid)
      ref_pos = np.asarray(descriptor.body_pos)
      ref_quat = np.asarray(descriptor.body_quat)
    mpos0 = np.zeros((self._nmocap, 3), dtype=np.float32)
    mquat0 = np.zeros((self._nmocap, 4), dtype=np.float32)
    mquat0[:, 0] = 1.0
    for bid in range(int(descriptor.nbody)):
      mid = int(mocapid[bid]) if bid < len(mocapid) else -1
      if 0 <= mid < self._nmocap:
        mpos0[mid] = ref_pos[bid]
        mquat0[mid] = ref_quat[bid]
    if self._nmocap and (
        not np.all(np.isfinite(mpos0))
        or not np.all(np.isfinite(mquat0))
        or np.any(np.abs(np.linalg.norm(mquat0, axis=1) - 1.0) > 1e-6)
    ):
      raise ValueError("compiled mocap reference frames must be finite with unit quaternions")
    self._mpos0 = np.frombuffer(mpos0.tobytes(), dtype=np.float32).reshape(mpos0.shape)
    self._mquat0 = np.frombuffer(mquat0.tobytes(), dtype=np.float32).reshape(mquat0.shape)

    initial_qpos = np.broadcast_to(
        descriptor.qpos0, (self.batch_size, descriptor.nq)
    ).copy()
    initial_qvel = np.zeros((self.batch_size, descriptor.nv), dtype=np.float64)
    if qpos is not None:
      initial_qpos = self._host_values(qpos, initial_qpos.shape, "qpos")
    if qvel is not None:
      initial_qvel = self._host_values(qvel, initial_qvel.shape, "qvel")
    initial_qpos = self._validate_qpos(
        self._host_values(initial_qpos, initial_qpos.shape, "qpos")
    )
    initial_qvel = self._host_values(
        initial_qvel, initial_qvel.shape, "qvel"
    )

    import torch

    self._torch = torch
    try:
      self._device = torch.device(device)
    except (TypeError, RuntimeError) as error:
      raise ValueError(f"invalid state device: {device!r}") from error
    if self._device.type == "mps" and not torch.backends.mps.is_available():
      raise RuntimeError("MPS is unavailable; pass device='cpu' only for state testing")
    if self._device.type not in ("mps", "cpu"):
      raise ValueError("device state supports only MPS or explicit CPU state tests")

    self._qpos = torch.as_tensor(
        initial_qpos, dtype=torch.float32, device=self._device
    ).clone()
    self._qvel = torch.as_tensor(
        initial_qvel, dtype=torch.float32, device=self._device
    ).clone()
    self._qacc = torch.zeros(
        (self.batch_size, descriptor.nv), dtype=torch.float32, device=self._device
    )
    self._time = torch.zeros(
        (self.batch_size,), dtype=torch.float32, device=self._device
    )
    self._status = torch.zeros(
        (self.batch_size,), dtype=torch.int32, device=self._device
    )
    if self._neq > 0:
      init_eq = np.broadcast_to(self._eq_active0, (self.batch_size, self._neq)).copy()
      self._eq_active = torch.as_tensor(init_eq, dtype=torch.int32, device=self._device).clone()
    else:
      self._eq_active = None
    if self._nmocap > 0:
      init_pos = np.broadcast_to(self._mpos0, (self.batch_size, self._nmocap, 3)).copy()
      init_quat = np.broadcast_to(self._mquat0, (self.batch_size, self._nmocap, 4)).copy()
      self._mpos = torch.as_tensor(init_pos, dtype=torch.float32, device=self._device).clone()
      self._mquat = torch.as_tensor(init_quat, dtype=torch.float32, device=self._device).clone()
    else:
      self._mpos = None
      self._mquat = None
    # Activation state: MuJoCo zeroes act on reset; keyframes override.
    if isinstance(model, mujoco.MjModel):
      self._na = int(model.na)
    else:
      self._na = 0
    if self._na > 0:
      init_act = np.zeros((self.batch_size, self._na), dtype=np.float32)
      self._act = torch.as_tensor(init_act, dtype=torch.float32, device=self._device).clone()
    else:
      self._act = None
    self._generation = 0

  @property
  def na(self):
    return self._na

  @property
  def act(self):
    if self._act is None:
      return None
    return self._act.detach().clone()

  @property
  def nmocap(self):
    return self._nmocap

  @property
  def mocap_pos(self):
    if self._mpos is None:
      return None
    return self._mpos.detach().clone()

  @property
  def mocap_quat(self):
    if self._mquat is None:
      return None
    return self._mquat.detach().clone()

  @property
  def neq(self):
    return self._neq

  @property
  def eq_active(self):
    if self._eq_active is None:
      return None
    return self._eq_active.detach().clone()

  @staticmethod
  def _host_values(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
      raise ValueError(f"{name} must be numeric with shape {shape}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      array = np.asarray(array, dtype=np.float32, order="C")
    if not np.all(np.isfinite(array)):
      raise ValueError(f"{name} must be finite and representable as float32")
    return array.copy()

  def _validate_qpos(self, qpos):
    for joint, kind in enumerate(self._model.jnt_type):
      start = int(self._model.jnt_qposadr[joint])
      if int(kind) == int(mujoco.mjtJoint.mjJNT_FREE):
        quat_slice = slice(start + 3, start + 7)
      elif int(kind) == int(mujoco.mjtJoint.mjJNT_BALL):
        quat_slice = slice(start, start + 4)
      else:
        continue
      norms = np.linalg.norm(qpos[:, quat_slice], axis=1)
      if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1) > 1e-5):
        raise ValueError("qpos free/ball quaternions must be unit length")
    return qpos

  @property
  def device(self):
    return str(self._device)

  @property
  def generation(self):
    return self._generation

  @property
  def qpos(self):
    return self._qpos.detach().clone()

  @property
  def qvel(self):
    return self._qvel.detach().clone()

  @property
  def qacc(self):
    return self._qacc.detach().clone()

  @property
  def time(self):
    return self._time.detach().clone()

  @property
  def status(self):
    return self._status.detach().clone()

  def _env_ids(self, env_ids):
    if env_ids is None:
      return np.arange(self.batch_size, dtype=np.int64)
    raw = np.asarray(env_ids)
    if raw.size == 0 and raw.ndim == 1:
      return np.empty((0,), dtype=np.int64)
    if raw.dtype.kind not in "iu":
      raise ValueError("env_ids must contain integers")
    ids = raw.astype(np.int64, copy=False)
    if ids.ndim != 1 or np.unique(ids).size != ids.size:
      raise ValueError("env_ids must be a vector of unique environment indices")
    if np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise ValueError("env_ids are out of range")
    return ids

  def _validate_eq_active(self, value, shape, name="eq_active"):
    arr = np.asarray(value)
    # Accept bool or integer 0/1; reject float/non-binary.
    if arr.dtype.kind == "f":
      raise ValueError(f"{name} must be boolean or integer 0/1, not float")
    if arr.shape != shape or arr.dtype.kind not in "bi u":
      raise ValueError(f"{name} must be boolean/integer with shape {shape}")
    try:
      as_int = np.asarray(arr, dtype=np.int32)
    except Exception as exc:
      raise ValueError(f"{name} must be representable as int32") from exc
    if np.any((as_int != 0) & (as_int != 1)):
      raise ValueError(f"{name} values must be 0 or 1")
    return np.ascontiguousarray(as_int, dtype=np.int32).copy()

  def set_equality_active(self, values, env_ids=None):
    """Update persistent equality activity for selected environments.

    `values` accepts host boolean/integer arrays with shape `(neq,)` (broadcast
    to all selected worlds) or `(len(env_ids), neq)` (per-world), or contiguous
    int32/bool MPS/CPU tensors with the same shapes (device tensors are copied,
    never borrowed or modified). `env_ids=None` selects all worlds. Validation
    is atomic: bad input leaves all worlds unchanged. Changes take effect on
    the next step/assembly and do not clear sticky failure status.
    """
    if self._neq == 0:
      raise ValueError("model has no equalities")
    ids = self._env_ids(env_ids)
    if not ids.size:
      return self._generation
    count = ids.size
    torch = self._torch
    # Device tensor fast path (copied, validated without host readback of state).
    if isinstance(values, torch.Tensor):
      if values.dtype not in (torch.int32, torch.bool):
        raise ValueError("eq_active tensor must have dtype torch.int32 or torch.bool")
      if not values.is_contiguous():
        raise ValueError("eq_active tensor must be contiguous")
      if tuple(values.shape) == (self._neq,):
        # Broadcast single row to all selected worlds.
        want = (count, self._neq)
        # Validate values are 0/1 without host readback of state (read input only).
        # For device inputs, nonfinite rows fail via solver; here check 0/1 via device ops?
        # To keep atomic without host sync, copy to host for validation (input readback
        # is allowed; hot-path stepping performs no readback).
        host = values.detach().to("cpu").numpy()
        checked = self._validate_eq_active(host, (self._neq,), "eq_active")
        checked = np.broadcast_to(checked, want).copy()
      elif tuple(values.shape) == (count, self._neq):
        host = values.detach().to("cpu").numpy()
        checked = self._validate_eq_active(host, (count, self._neq), "eq_active")
      elif env_ids is None and tuple(values.shape) == (self.batch_size, self._neq):
        host = values.detach().to("cpu").numpy()
        checked = self._validate_eq_active(host, (self.batch_size, self._neq), "eq_active")
        # ids is all worlds in order; use checked directly.
        ids = np.arange(self.batch_size, dtype=np.int64)
        count = self.batch_size
      else:
        raise ValueError(
            f"eq_active tensor must have shape ({self._neq},) or ({count}, {self._neq})"
        )
    else:
      arr = np.asarray(values)
      if arr.shape == (self._neq,):
        checked = self._validate_eq_active(arr, (self._neq,), "eq_active")
        checked = np.broadcast_to(checked, (count, self._neq)).copy()
      elif arr.shape == (count, self._neq):
        checked = self._validate_eq_active(arr, (count, self._neq), "eq_active")
      else:
        raise ValueError(
            f"eq_active must have shape ({self._neq},) or ({count}, {self._neq}), got {arr.shape}"
        )
    index = torch.as_tensor(ids, dtype=torch.int64, device=self._device)
    value_tensor = torch.as_tensor(checked, dtype=torch.int32, device=self._device)
    # Atomic commit via copy-then-swap (no partial updates on failure; validation
    # already passed, index_copy_ cannot fail for validated shapes).
    next_eq = self._eq_active.clone()
    next_eq.index_copy_(0, index, value_tensor)
    self._eq_active = next_eq
    self._generation += 1
    return self._generation

  def _checked_mocap_pair(self, pos, quat, count):
    """Validate one mocap pose batch; returns float32 copies or raises."""
    pos_arr = np.asarray(pos, dtype=np.float64)
    quat_arr = np.asarray(quat, dtype=np.float64)
    if pos_arr.shape != (count, self._nmocap, 3) or not np.all(np.isfinite(pos_arr)):
      raise ValueError(
          f"mocap_pos must be finite with shape ({count}, {self._nmocap}, 3)"
      )
    if quat_arr.shape != (count, self._nmocap, 4):
      raise ValueError(
          f"mocap_quat must have shape ({count}, {self._nmocap}, 4)"
      )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      pos32 = np.asarray(pos_arr, dtype=np.float32, order="C")
      quat32 = np.asarray(quat_arr, dtype=np.float32, order="C")
    if not np.all(np.isfinite(pos32)) or not np.all(np.isfinite(quat32)):
      raise ValueError("mocap poses must be representable as finite float32")
    norms = np.linalg.norm(quat32.astype(np.float64).reshape(-1, 4), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1) > 1e-5):
      raise ValueError("mocap_quat must hold unit quaternions")
    return pos32.copy(), quat32.copy()

  def set_mocap(self, pos, quat=None, env_ids=None):
    """Update prescribed mocap poses for selected environments.

    Accepts host arrays or contiguous float32 MPS/CPU tensors. Shapes follow
    the equality-activity convention: ``(nmocap, 3)``/``(nmocap, 4)`` broadcast
    to all selected worlds, or ``(len(env_ids), nmocap, 3/4)`` per world.
    Alternatively pass a single ``(len(env_ids), nmocap, 7)`` posquat array as
    ``pos`` with ``quat=None``. Inputs are copied, never borrowed or modified;
    validation is atomic and sticky failure status is preserved. Changes take
    effect on the next step/assembly.
    """
    if self._nmocap == 0:
      raise ValueError("model has no mocap bodies")
    ids = self._env_ids(env_ids)
    if not ids.size:
      return self._generation
    count = ids.size
    torch = self._torch
    if quat is None and isinstance(pos, torch.Tensor):
      if pos.dtype != torch.float32 or not pos.is_contiguous():
        raise ValueError("mocap posquat tensor must be contiguous float32")
      if tuple(pos.shape) == (count, self._nmocap, 7):
        host = pos.detach().to("cpu").numpy()
        pos_h, quat_h = host[..., :3], host[..., 3:]
      elif tuple(pos.shape) == (self._nmocap, 7):
        host = pos.detach().to("cpu").numpy()
        pos_h = np.broadcast_to(host[..., :3], (count, self._nmocap, 3)).copy()
        quat_h = np.broadcast_to(host[..., 3:], (count, self._nmocap, 4)).copy()
      else:
        raise ValueError(
            f"mocap posquat tensor must have shape ({count}, {self._nmocap}, 7)"
          f" or ({self._nmocap}, 7)"
        )
      checked_pos, checked_quat = self._checked_mocap_pair(pos_h, quat_h, count)
    elif quat is None:
      arr = np.asarray(pos)
      if arr.shape == (self._nmocap, 7):
        pos_h = np.broadcast_to(arr[..., :3], (count, self._nmocap, 3)).copy()
        quat_h = np.broadcast_to(arr[..., 3:], (count, self._nmocap, 4)).copy()
      elif arr.shape == (count, self._nmocap, 7):
        pos_h, quat_h = arr[..., :3].copy(), arr[..., 3:].copy()
      else:
        raise ValueError(
          f"mocap posquat must have shape ({self._nmocap}, 7) or "
          f"({count}, {self._nmocap}, 7), got {arr.shape}"
        )
      checked_pos, checked_quat = self._checked_mocap_pair(pos_h, quat_h, count)
    else:
      if isinstance(pos, torch.Tensor) or isinstance(quat, torch.Tensor):
        if not isinstance(pos, torch.Tensor) or not isinstance(quat, torch.Tensor):
          raise ValueError("mocap pos and quat must both be tensors or both arrays")
        for tensor, want, name in (
            (pos, (count, self._nmocap, 3), "mocap_pos"),
            (quat, (count, self._nmocap, 4), "mocap_quat"),
        ):
          if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError(f"{name} tensor must be contiguous float32")
        pos_shapes = [tuple(pos.shape)]
        quat_shapes = [tuple(quat.shape)]
        if pos_shapes[0] == (self._nmocap, 3) and quat_shapes[0] == (self._nmocap, 4):
          pos_h = pos.detach().to("cpu").numpy()
          quat_h = quat.detach().to("cpu").numpy()
          pos_h = np.broadcast_to(pos_h, (count, self._nmocap, 3)).copy()
          quat_h = np.broadcast_to(quat_h, (count, self._nmocap, 4)).copy()
        elif pos_shapes[0] == (count, self._nmocap, 3) and quat_shapes[0] == (count, self._nmocap, 4):
          pos_h = pos.detach().to("cpu").numpy()
          quat_h = quat.detach().to("cpu").numpy()
        else:
          raise ValueError(
            f"mocap pos/quat tensors must have shapes ({count}, {self._nmocap}, 3/4)"
            f" or ({self._nmocap}, 3/4)"
          )
        checked_pos, checked_quat = self._checked_mocap_pair(pos_h, quat_h, count)
      else:
        pos_h = np.asarray(pos)
        quat_h = np.asarray(quat)
        if pos_h.shape == (self._nmocap, 3) and quat_h.shape == (self._nmocap, 4):
          pos_h = np.broadcast_to(pos_h, (count, self._nmocap, 3)).copy()
          quat_h = np.broadcast_to(quat_h, (count, self._nmocap, 4)).copy()
        checked_pos, checked_quat = self._checked_mocap_pair(pos_h, quat_h, count)
    index = torch.as_tensor(ids, dtype=torch.int64, device=self._device)
    pos_tensor = torch.as_tensor(checked_pos, dtype=torch.float32, device=self._device)
    quat_tensor = torch.as_tensor(checked_quat, dtype=torch.float32, device=self._device)
    next_pos = self._mpos.clone()
    next_quat = self._mquat.clone()
    next_pos.index_copy_(0, index, pos_tensor)
    next_quat.index_copy_(0, index, quat_tensor)
    self._mpos, self._mquat = next_pos, next_quat
    self._generation += 1
    return self._generation

  def copy_environment(self, src, dst):
    """Copy every state row (qpos/qvel/qacc/time/status/eq/mocap/act) src -> dst."""
    for name, value in (("src", src), ("dst", dst)):
      raw = np.asarray(value)
      if raw.shape != () or raw.dtype.kind not in "iu":
        raise ValueError("src/dst must be integer environment indices")
    src_i, dst_i = int(np.asarray(src)), int(np.asarray(dst))
    for index in (src_i, dst_i):
      if not 0 <= index < self.batch_size:
        raise ValueError("environment index out of range")
    for tensor_name in ("_qpos", "_qvel", "_qacc", "_time", "_status"):
      tensor = getattr(self, tensor_name).clone()
      tensor[dst_i] = getattr(self, tensor_name)[src_i]
      setattr(self, tensor_name, tensor)
    if self._eq_active is not None:
      tensor = self._eq_active.clone()
      tensor[dst_i] = self._eq_active[src_i]
      self._eq_active = tensor
    if self._mpos is not None:
      pos = self._mpos.clone()
      quat = self._mquat.clone()
      pos[dst_i] = self._mpos[src_i]
      quat[dst_i] = self._mquat[src_i]
      self._mpos, self._mquat = pos, quat
    if self._act is not None:
      act_t = self._act.clone()
      act_t[dst_i] = self._act[src_i]
      self._act = act_t
    self._generation += 1
    return self._generation

  def reset(self, env_ids=None, qpos=None, qvel=None, eq_active=None,
            mocap_pos=None, mocap_quat=None, act=None):
    """Reset selected rows atomically from checked host arrays or model defaults.

    `eq_active=None` restores compiled defaults for selected worlds; pass an
    explicit `(neq,)` or `(len(env_ids), neq)` boolean/integer array to override.
    `mocap_pos`/`mocap_quat=None` restore compiled reference frames; pass
    explicit `(nmocap, 3/4)` or `(len(env_ids), nmocap, 3/4)` arrays to override.
    `act=None` zeroes activation state for selected worlds; pass explicit
    `(na,)` or `(len(env_ids), na)` arrays to override.
    Unselected worlds keep their activity. Validation is atomic.
    """
    ids = self._env_ids(env_ids)
    if not ids.size:
      return self._generation
    count = ids.size
    pos_default = np.broadcast_to(
        self._model.qpos0, (count, self._model.nq)
    ).copy()
    vel_default = np.zeros((count, self._model.nv), dtype=np.float32)
    if qpos is None:
      pos = pos_default
    else:
      arr = np.asarray(qpos)
      if arr.shape == (self._model.nq,):
        arr = np.broadcast_to(arr, (count, self._model.nq)).copy()
      pos = self._host_values(arr, pos_default.shape, "qpos")
    if qvel is None:
      vel = vel_default
    else:
      arr = np.asarray(qvel)
      if arr.shape == (self._model.nv,):
        arr = np.broadcast_to(arr, (count, self._model.nv)).copy()
      vel = self._host_values(arr, vel_default.shape, "qvel")
    pos = self._validate_qpos(pos)
    if self._nmocap == 0:
      if mocap_pos is not None or mocap_quat is not None:
        raise ValueError("model has no mocap bodies")
      mocap_checked = None
    else:
      if mocap_pos is None and mocap_quat is None:
        mocap_checked = (
          np.broadcast_to(self._mpos0, (count, self._nmocap, 3)).copy(),
          np.broadcast_to(self._mquat0, (count, self._nmocap, 4)).copy(),
        )
      elif mocap_pos is None or mocap_quat is None:
        raise ValueError("mocap_pos and mocap_quat must be given together")
      else:
        pos_h = np.asarray(mocap_pos)
        quat_h = np.asarray(mocap_quat)
        if pos_h.shape == (self._nmocap, 3) and quat_h.shape == (self._nmocap, 4):
          pos_h = np.broadcast_to(pos_h, (count, self._nmocap, 3)).copy()
          quat_h = np.broadcast_to(quat_h, (count, self._nmocap, 4)).copy()
        mocap_checked = self._checked_mocap_pair(pos_h, quat_h, count)
    if self._neq == 0:
      if eq_active is not None:
        raise ValueError("model has no equalities")
      eq_checked = None
    else:
      if eq_active is None:
        eq_checked = np.broadcast_to(self._eq_active0, (count, self._neq)).copy()
      else:
        arr = np.asarray(eq_active)
        if arr.shape == (self._neq,):
          eq_checked = self._validate_eq_active(arr, (self._neq,), "eq_active")
          eq_checked = np.broadcast_to(eq_checked, (count, self._neq)).copy()
        elif arr.shape == (count, self._neq):
          eq_checked = self._validate_eq_active(arr, (count, self._neq), "eq_active")
        else:
          raise ValueError(
              f"eq_active must have shape ({self._neq},) or ({count}, {self._neq}), got {arr.shape}"
          )
    if self._na == 0:
      if act is not None:
        raise ValueError("model has no activation state")
      act_checked = None
    else:
      if act is None:
        act_checked = np.zeros((count, self._na), dtype=np.float32)
      else:
        arr = np.asarray(act)
        if arr.shape == (self._na,):
          arr = np.broadcast_to(arr, (count, self._na)).copy()
        act_checked = self._host_values(arr, (count, self._na), "act")

    index = self._torch.as_tensor(ids, dtype=self._torch.int64, device=self._device)
    pos_tensor = self._torch.as_tensor(
        pos, dtype=self._torch.float32, device=self._device
    )
    vel_tensor = self._torch.as_tensor(
        vel, dtype=self._torch.float32, device=self._device
    )
    zero_acc = self._torch.zeros_like(vel_tensor)
    zero_time = self._torch.zeros(
        (count,), dtype=self._torch.float32, device=self._device
    )
    zero_status = self._torch.zeros(
        (count,), dtype=self._torch.int32, device=self._device
    )

    next_qpos = self._qpos.clone()
    next_qvel = self._qvel.clone()
    next_qacc = self._qacc.clone()
    next_time = self._time.clone()
    next_status = self._status.clone()
    next_qpos.index_copy_(0, index, pos_tensor)
    next_qvel.index_copy_(0, index, vel_tensor)
    next_qacc.index_copy_(0, index, zero_acc)
    next_time.index_copy_(0, index, zero_time)
    next_status.index_copy_(0, index, zero_status)
    self._qpos, self._qvel = next_qpos, next_qvel
    self._qacc, self._time, self._status = next_qacc, next_time, next_status
    if self._neq > 0:
      next_eq = self._eq_active.clone()
      eq_tensor = self._torch.as_tensor(eq_checked, dtype=self._torch.int32, device=self._device)
      next_eq.index_copy_(0, index, eq_tensor)
      self._eq_active = next_eq
    if self._nmocap > 0:
      next_pos = self._mpos.clone()
      next_quat = self._mquat.clone()
      next_pos.index_copy_(
          0, index,
          self._torch.as_tensor(mocap_checked[0], dtype=self._torch.float32, device=self._device))
      next_quat.index_copy_(
          0, index,
          self._torch.as_tensor(mocap_checked[1], dtype=self._torch.float32, device=self._device))
      self._mpos, self._mquat = next_pos, next_quat
    if self._na > 0:
      next_act = self._act.clone()
      next_act.index_copy_(
          0, index,
          self._torch.as_tensor(act_checked, dtype=self._torch.float32, device=self._device))
      self._act = next_act
    self._generation += 1
    return self._generation

  def snapshot(self):
    """Copy all state to an immutable host checkpoint outside the step loop."""
    if self._neq == 0 and self._nmocap == 0 and self._na == 0:
      return StateSnapshot(
        model_fingerprint=self._model_fingerprint,
        profile_fingerprint=self._profile_fingerprint,
        timestep=self.profile.timestep,
        nq=self._model.nq,
        nv=self._model.nv,
        batch_size=self.batch_size,
        qpos=self._qpos.detach().cpu().numpy(),
        qvel=self._qvel.detach().cpu().numpy(),
        qacc=self._qacc.detach().cpu().numpy(),
        time=self._time.detach().cpu().numpy(),
        status=self._status.detach().cpu().numpy(),
        schema_version=1,
        neq=0,
        eq_active=None,
      )
    if self._nmocap == 0 and self._na == 0:
      return StateSnapshot(
        model_fingerprint=self._model_fingerprint,
        profile_fingerprint=self._profile_fingerprint,
        timestep=self.profile.timestep,
        nq=self._model.nq,
        nv=self._model.nv,
        batch_size=self.batch_size,
        qpos=self._qpos.detach().cpu().numpy(),
        qvel=self._qvel.detach().cpu().numpy(),
        qacc=self._qacc.detach().cpu().numpy(),
        time=self._time.detach().cpu().numpy(),
        status=self._status.detach().cpu().numpy(),
        schema_version=2,
        neq=self._neq,
        eq_active=self._eq_active.detach().cpu().numpy() if self._neq else None,
      )
    if self._na == 0:
      return StateSnapshot(
        model_fingerprint=self._model_fingerprint,
        profile_fingerprint=self._profile_fingerprint,
        timestep=self.profile.timestep,
        nq=self._model.nq,
        nv=self._model.nv,
        batch_size=self.batch_size,
        qpos=self._qpos.detach().cpu().numpy(),
        qvel=self._qvel.detach().cpu().numpy(),
        qacc=self._qacc.detach().cpu().numpy(),
        time=self._time.detach().cpu().numpy(),
        status=self._status.detach().cpu().numpy(),
        schema_version=3,
        neq=self._neq,
        eq_active=self._eq_active.detach().cpu().numpy() if self._neq else None,
        nmocap=self._nmocap,
        mpos=self._mpos.detach().cpu().numpy(),
        mquat=self._mquat.detach().cpu().numpy(),
    )
    return StateSnapshot(
        model_fingerprint=self._model_fingerprint,
        profile_fingerprint=self._profile_fingerprint,
        timestep=self.profile.timestep,
        nq=self._model.nq,
        nv=self._model.nv,
        batch_size=self.batch_size,
        qpos=self._qpos.detach().cpu().numpy(),
        qvel=self._qvel.detach().cpu().numpy(),
        qacc=self._qacc.detach().cpu().numpy(),
        time=self._time.detach().cpu().numpy(),
        status=self._status.detach().cpu().numpy(),
        schema_version=4,
        neq=self._neq,
        eq_active=self._eq_active.detach().cpu().numpy() if self._neq else None,
        nmocap=self._nmocap,
        mpos=self._mpos.detach().cpu().numpy() if self._nmocap else None,
        mquat=self._mquat.detach().cpu().numpy() if self._nmocap else None,
        nact=self._na,
        act=self._act.detach().cpu().numpy(),
    )

  def restore(self, snapshot):
    """Restore a matching checkpoint only after validating every field."""
    if not isinstance(snapshot, StateSnapshot):
      raise TypeError("snapshot must be a StateSnapshot")
    if snapshot.schema_version not in (1, 2, 3, 4):
      raise ValueError("unsupported state snapshot schema")
    if (
        snapshot.model_fingerprint != self._model_fingerprint
        or snapshot.profile_fingerprint != self._profile_fingerprint
        or snapshot.timestep != self.profile.timestep
        or snapshot.nq != self._model.nq
        or snapshot.nv != self._model.nv
        or snapshot.batch_size != self.batch_size
    ):
      raise ValueError("snapshot model, profile, timestep, or dimensions do not match")
    # Equality compatibility: never silently lose activity.
    if self._neq == 0:
      if snapshot.schema_version >= 2 and snapshot.neq != 0:
        raise ValueError("snapshot equality dimensions do not match")
      eq_checked = None
    else:
      if snapshot.schema_version == 1:
        raise ValueError(
            "snapshot schema 1 has no equality activity; refusing to restore "
            "into a model with equalities (would silently lose activity)"
        )
      if snapshot.neq != self._neq:
        raise ValueError("snapshot equality dimensions do not match")
      eq_checked = self._validate_eq_active(
          snapshot.eq_active, (self.batch_size, self._neq), "eq_active"
      )
    # Mocap compatibility: never silently lose prescribed poses.
    if self._nmocap == 0:
      if snapshot.schema_version >= 3 and (
          snapshot.nmocap != 0 or snapshot.mpos is not None or snapshot.mquat is not None
      ):
        raise ValueError("snapshot mocap dimensions do not match")
      mocap_checked = None
    else:
      if snapshot.schema_version < 3:
        raise ValueError(
            "snapshot schema <3 has no mocap poses; refusing to restore "
            "into a model with mocap bodies (would silently lose poses)"
        )
      if snapshot.nmocap != self._nmocap:
        raise ValueError("snapshot mocap dimensions do not match")
      mocap_checked = self._checked_mocap_pair(
          snapshot.mpos, snapshot.mquat, self.batch_size
      )
    # Activation compatibility: never silently lose activation state.
    if self._na == 0:
      if snapshot.schema_version >= 4 and (
          snapshot.nact != 0 or snapshot.act is not None
      ):
        raise ValueError("snapshot activation dimensions do not match")
      act_checked = None
    else:
      if snapshot.schema_version < 4:
        raise ValueError(
            "snapshot schema <4 has no activation state; refusing to restore "
            "into a model with activation state (would silently lose act)"
        )
      if snapshot.nact != self._na:
        raise ValueError("snapshot activation dimensions do not match")
      act_checked = self._host_values(
          snapshot.act, (self.batch_size, self._na), "act")
    qpos = self._validate_qpos(
        self._host_values(snapshot.qpos, (self.batch_size, self._model.nq), "qpos")
    )
    qvel = self._host_values(
        snapshot.qvel, (self.batch_size, self._model.nv), "qvel"
    )
    qacc = self._host_values(
        snapshot.qacc, (self.batch_size, self._model.nv), "qacc"
    )
    time = self._host_values(snapshot.time, (self.batch_size,), "time")
    if np.any(time < 0):
      raise ValueError("snapshot time must be nonnegative")
    status = np.asarray(snapshot.status)
    if status.shape != (self.batch_size,) or status.dtype.kind not in "iu":
      raise ValueError("snapshot status has an invalid shape or dtype")
    if np.any(status < np.iinfo(np.int32).min) or np.any(
        status > np.iinfo(np.int32).max
    ):
      raise ValueError("snapshot status values must fit in int32")

    tensors = (
        self._torch.as_tensor(qpos, dtype=self._torch.float32, device=self._device),
        self._torch.as_tensor(qvel, dtype=self._torch.float32, device=self._device),
        self._torch.as_tensor(qacc, dtype=self._torch.float32, device=self._device),
        self._torch.as_tensor(time, dtype=self._torch.float32, device=self._device),
        self._torch.as_tensor(
            np.asarray(status, dtype=np.int32).copy(),
            dtype=self._torch.int32,
            device=self._device,
        ),
    )
    self._qpos, self._qvel, self._qacc, self._time, self._status = tensors
    if self._neq > 0:
      self._eq_active = self._torch.as_tensor(eq_checked, dtype=self._torch.int32, device=self._device)
    if self._nmocap > 0:
      self._mpos = self._torch.as_tensor(mocap_checked[0], dtype=self._torch.float32, device=self._device)
      self._mquat = self._torch.as_tensor(mocap_checked[1], dtype=self._torch.float32, device=self._device)
    if self._na > 0:
      self._act = self._torch.as_tensor(act_checked, dtype=self._torch.float32, device=self._device)
    self._generation += 1
    return self._generation

  def reset_to_default(self, env_ids=None):
    """Reset selected worlds to compiled qpos0/eq_active0/mocap reference frames."""
    return self.reset(env_ids=env_ids)
