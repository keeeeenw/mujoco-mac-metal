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

"""Explicitly opt-in Torch MPS launcher for the kinematics-only stage."""

from pathlib import Path

import numpy as np

from mujoco_metal.model import _validate_lowered
from mujoco_metal.model import ModelDescriptor
from mujoco_metal.model import snapshot_descriptor
from mujoco_metal.capacity import _fk_auxiliary_words

_SHADER = Path(__file__).parent / "shaders" / "kinematics.metal"
_INT32_MAX = (1 << 31) - 1
_UINT32_CAPACITY = 1 << 32
_FK_INT_INPUTS = (
    "body_parentid", "jnt_type", "jnt_qposadr", "jnt_bodyid",
    "body_sameframe", "geom_bodyid", "geom_sameframe", "site_bodyid",
    "site_sameframe",
)
_FK_FLOAT_INPUTS = (
    "body_pos_pair", "body_quat_pair", "jnt_pos_pair", "jnt_axis_pair",
    "qpos0_pair", "geom_pos_pair", "geom_quat_pair", "site_pos_pair",
    "site_quat_pair", "body_ipos_pair", "body_iquat_pair",
)
_FK_OUTPUTS = (
    ("body_pos", "nbody", 3), ("body_pos_low", "nbody", 3),
    ("body_pos_tail", "nbody", 3), ("body_quat", "nbody", 4),
    ("body_quat_low", "nbody", 4), ("body_quat_tail", "nbody", 4),
    ("geom_pos", "ngeom", 3), ("geom_pos_low", "ngeom", 3),
    ("geom_pos_tail", "ngeom", 3), ("geom_quat", "ngeom", 4),
    ("geom_quat_low", "ngeom", 4), ("geom_quat_tail", "ngeom", 4),
    ("geom_xmat", "ngeom", 9), ("geom_xmat_low", "ngeom", 9),
    ("geom_xmat_tail", "ngeom", 9), ("site_pos", "nsite", 3),
    ("site_pos_low", "nsite", 3), ("site_pos_tail", "nsite", 3),
    ("site_quat", "nsite", 4), ("site_quat_low", "nsite", 4),
    ("site_quat_tail", "nsite", 4),
    ("inertial_pos", "nbody", 3), ("inertial_pos_low", "nbody", 3),
    ("inertial_pos_tail", "nbody", 3),
    ("inertial_quat", "nbody", 4),
    ("inertial_quat_low", "nbody", 4),
    ("inertial_quat_tail", "nbody", 4),
    ("joint_anchor", "njnt", 3), ("joint_axis", "njnt", 3),
)


def _validate_workspace_index_capacity(batch_size, buffers, dimensions):
  """Reject dimensions and offsets that the MSL uint ABI cannot represent."""
  if (
      isinstance(batch_size, bool)
      or not isinstance(batch_size, int)
      or batch_size <= 0
  ):
    raise ValueError("batch_size must be a positive integer")
  if batch_size > _INT32_MAX:
    raise ValueError("batch_size exceeds the Metal int32 dimension limit")
  for name, value in dimensions.items():
    if value < 0 or value > _INT32_MAX:
      raise ValueError(f"{name} exceeds the Metal int32 dimension limit")
  for name, elements in buffers.items():
    if elements < 0 or elements > _UINT32_CAPACITY:
      raise ValueError(f"{name} exceeds the Metal uint32 index capacity")


def _prepare_host_arrays(model: ModelDescriptor):
  """Validate and pack float32/int32 constants without initializing a device."""
  counts = {
      "nq": model.nq,
      "nv": model.nv,
      "nu": model.nu,
      "nbody": model.nbody,
      "njnt": model.njnt,
      "ngeom": model.ngeom,
      "nsite": model.nsite,
      "ntendon": model.ntendon,
      "ntendon_jnnz": model.ntendon_jnnz,
      "nmass_nnz": model.nmass_nnz,
      "nmocap": model.nmocap,
      "disableflags": model.disableflags,
  }
  names = (
      "body_rootid",
      "body_dofadr",
      "body_dofnum",
      "body_subtreemass",
      "dof_parentid",
      "dof_bodyid",
      "dof_jntid",
      "gravity",
      "body_parentid",
      "body_treeid",
      "body_jntadr",
      "body_jntnum",
      "body_mocapid",
      "body_sameframe",
      "body_pos",
      "body_quat",
      "body_ipos",
      "body_iquat",
      "body_mass",
      "body_inertia",
      "dof_armature",
      "dof_damping",
      "actuator_armature",
      "actuator_trntype",
      "actuator_trnid",
      "actuator_gear",
      "actuator_damping",
      "actuator_dampingpoly",
      "tendon_armature",
      "tendon_treeid",
      "tendon_treenum",
      "tendon_j_rowadr",
      "tendon_j_rownnz",
      "tendon_j_colind",
      "mass_rowadr",
      "mass_rownnz",
      "mass_colind",
      "jnt_type",
      "jnt_qposadr",
      "jnt_dofadr",
      "jnt_bodyid",
      "jnt_pos",
      "jnt_axis",
      "qpos0",
      "geom_bodyid",
      "geom_type",
      "geom_size",
      "geom_pos",
      "geom_quat",
      "geom_sameframe",
      "site_bodyid",
      "site_pos",
      "site_quat",
      "site_sameframe",
  )
  values = {name: getattr(model, name) for name in names}
  _validate_lowered(counts, values)
  host = {}
  for name, value in values.items():
    dtype = np.int32 if np.asarray(value).dtype.kind in "iu" else np.float32
    converted = np.array(value, dtype=dtype, copy=True).reshape(-1)
    if converted.dtype.kind == "f" and not np.all(np.isfinite(converted)):
      raise ValueError(f"{name} cannot be represented as finite float32")
    if name not in ("body_jntadr", "body_jntnum") and converted.size == 0:
      converted = np.zeros(1, dtype=dtype)
    host[name] = converted
  # Keep two residual words alongside each float32 high word.  A single low
  # float retains only about 48 effective bits; source-rounded CCD predicates
  # can distinguish the remaining binary64 bits after repeated FK transforms.
  for name in ("body_pos", "jnt_pos", "qpos0", "geom_pos",
               "body_ipos", "site_pos"):
    raw = np.asarray(values[name], dtype=np.float64).reshape(-1)
    if raw.size:
      high = host[name][:raw.size].astype(np.float64, copy=False)
      low = (raw - high).astype(np.float32)
      tail = (raw - high - low.astype(np.float64)).astype(np.float32)
    else:
      low = np.zeros(1, dtype=np.float32)
      tail = np.zeros(1, dtype=np.float32)
    host[name + "_low"] = np.array(low, dtype=np.float32, copy=True)
    host[name + "_tail"] = np.array(tail, dtype=np.float32, copy=True)
    high_words = np.array(host[name][:max(raw.size, 1)],
                          dtype=np.float32, copy=True)
    host[name + "_pair"] = np.concatenate(
        (high_words, np.array(low, dtype=np.float32, copy=True),
         np.array(tail, dtype=np.float32, copy=True)))
  # Preserve compiled binary64 quaternion and axis constants across the
  # float32 host/device boundary.  The three planes are high, residual, and
  # residual-of-residual, each with the original flattened shape.
  for name in ("body_quat", "body_iquat", "geom_quat", "site_quat",
               "jnt_axis"):
    raw = np.asarray(values[name], dtype=np.float64).reshape(-1)
    high_words = np.array(host[name][:max(raw.size, 1)], dtype=np.float32,
                          copy=True)
    if raw.size:
      high = high_words[:raw.size].astype(np.float64, copy=False)
      low = (raw - high).astype(np.float32)
      tail = (raw - high - low.astype(np.float64)).astype(np.float32)
    else:
      low = np.zeros(1, dtype=np.float32)
      tail = np.zeros(1, dtype=np.float32)
    host[name + "_pair"] = np.concatenate((
        high_words, np.array(low, dtype=np.float32, copy=True),
        np.array(tail, dtype=np.float32, copy=True)))
  for name in ("body_quat", "body_iquat", "geom_quat", "site_quat"):
    if np.asarray(values[name]).size == 0:
      continue
    quat = host[name].reshape(-1, 4)
    norms = np.linalg.norm(quat.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
      raise ValueError(
          f"{name} cannot be represented as nonzero float32 quaternions"
      )
  return host


def _shape_output(buffer, batch, count, width):
  """Drop dummy storage from zero-length GPU outputs before reshaping."""
  return buffer[: batch * count * width].reshape(batch, count, width)


def _pack_fk_constants(host_arrays, names, dtype):
  offsets = {}
  pieces = []
  offset = 0
  for name in names:
    part = np.asarray(host_arrays[name], dtype=dtype).reshape(-1)
    if part.size == 0:
      part = np.zeros(1, dtype=dtype)
    offsets[name] = offset
    pieces.append(part)
    offset += int(part.size)
  if offset > _INT32_MAX:
    raise ValueError("packed FK constants exceed int32 offset capacity")
  return np.concatenate(pieces), offsets


def _fk_output_layout(model, batch_size):
  """Return field offsets and physical words for the single FK output arena."""
  _validate_workspace_index_capacity(batch_size, {}, {})
  offsets = {}
  cursor = 0
  for name, count_name, width in _FK_OUTPUTS:
    count = int(getattr(model, count_name))
    if count < 0:
      raise ValueError(f"{count_name} must be non-negative")
    offsets[name] = cursor
    cursor += max(batch_size * count * width, 1)
    if cursor > _INT32_MAX:
      raise ValueError("FK output arena exceeds the int32 offset capacity")
  return offsets, cursor


class MetalKinematics:
  """Batched native MSL forward kinematics; construction initializes MPS."""

  def __init__(self, model: ModelDescriptor, batch_size: int = 1):
    self.model = snapshot_descriptor(model)
    host_arrays = _prepare_host_arrays(self.model)
    # Importing this module remains host-only; construction is the explicit
    # device capability boundary.
    import torch

    self._torch = torch
    if not torch.backends.mps.is_available():
      raise RuntimeError("PyTorch MPS is unavailable")
    if not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("PyTorch does not provide torch.mps.compile_shader")
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.forward_kinematics
    self._prepare_rows_kernel = self._library.prepare_fk_rows
    packed_i, offsets_i = _pack_fk_constants(
        host_arrays, _FK_INT_INPUTS, np.int32)
    packed_f, offsets_f = _pack_fk_constants(
        host_arrays, _FK_FLOAT_INPUTS, np.float32)
    self._arrays = {
        "static_int": torch.from_numpy(packed_i).to(self._device),
        "static_float": torch.from_numpy(packed_f).to(self._device),
    }
    self._static_offsets = {**offsets_i, **offsets_f}
    self._workspace = None
    self.prepare_workspace(batch_size)

  def prepare_workspace(self, batch_size: int):
    """Preallocate FK outputs for ``batch_size`` device worlds.

    Buffers returned by :meth:`run_device` are borrowed workspace views and
    remain valid only until the next call that reuses this workspace.
    """
    _validate_workspace_index_capacity(batch_size, {}, {})
    torch = self._torch
    m = self.model
    _validate_workspace_index_capacity(
        batch_size,
        {
            "qpos": batch_size * m.nq,
            "body_pos": batch_size * m.nbody * 3,
            "body_pos_low": batch_size * m.nbody * 3,
            "body_pos_tail": batch_size * m.nbody * 3,
            "body_quat": batch_size * m.nbody * 4,
            "body_quat_low": batch_size * m.nbody * 4,
            "body_quat_tail": batch_size * m.nbody * 4,
            "geom_pos": batch_size * m.ngeom * 3,
            "geom_pos_low": batch_size * m.ngeom * 3,
            "geom_pos_tail": batch_size * m.ngeom * 3,
            "geom_quat": batch_size * m.ngeom * 4,
            "geom_quat_low": batch_size * m.ngeom * 4,
            "geom_quat_tail": batch_size * m.ngeom * 4,
            "geom_xmat": batch_size * m.ngeom * 9,
            "geom_xmat_low": batch_size * m.ngeom * 9,
            "geom_xmat_tail": batch_size * m.ngeom * 9,
            "site_pos": batch_size * m.nsite * 3,
            "site_pos_low": batch_size * m.nsite * 3,
            "site_pos_tail": batch_size * m.nsite * 3,
            "site_quat": batch_size * m.nsite * 4,
            "site_quat_low": batch_size * m.nsite * 4,
            "site_quat_tail": batch_size * m.nsite * 4,
            "inertial_pos": batch_size * m.nbody * 3,
            "inertial_pos_low": batch_size * m.nbody * 3,
            "inertial_pos_tail": batch_size * m.nbody * 3,
            "inertial_quat": batch_size * m.nbody * 4,
            "inertial_quat_low": batch_size * m.nbody * 4,
            "inertial_quat_tail": batch_size * m.nbody * 4,
            "joint_anchor": batch_size * m.njnt * 3,
            "joint_axis": batch_size * m.njnt * 3,
            "tree_awake": batch_size * max(
                int(np.max(np.asarray(m.body_treeid), initial=-1)) + 1, 1),
            "tree_awake_cast": batch_size * max(
                int(np.max(np.asarray(m.body_treeid), initial=-1)) + 1, 1),
            "all_world_mask": batch_size,
            "body_parentid": m.nbody,
            "geom_bodyid": m.ngeom,
            "site_bodyid": m.nsite,
        },
        {
            "nq": m.nq,
            "nbody": m.nbody,
            "njnt": m.njnt,
            "ngeom": m.ngeom,
            "nsite": m.nsite,
        },
    )
    outputs = {}
    out_offsets, out_cursor = _fk_output_layout(m, batch_size)
    _validate_workspace_index_capacity(
        batch_size, {"pose_output_arena": out_cursor},
        {"pose_output_offset": out_cursor})
    outputs["pose_output"] = torch.empty(
        out_cursor, dtype=torch.float32, device=self._device)
    for name, count_name, width in _FK_OUTPUTS:
      count = int(getattr(m, count_name))
      elements = max(batch_size * count * width, 1)
      outputs[name] = outputs["pose_output"].narrow(
          0, out_offsets[name], elements)

    dims_values = [
        m.nq, m.nbody, m.njnt, m.ngeom, m.nsite, batch_size, m.nmocap,
        int(np.max(np.asarray(m.body_treeid), initial=-1)) + 1,
        0, 0, 0, 0,
    ]
    dims_values.extend(self._static_offsets[name] for name in _FK_INT_INPUTS)
    dims_values.extend(self._static_offsets[name] for name in _FK_FLOAT_INPUTS)
    dims_values.extend(out_offsets[name] for name, _, _ in _FK_OUTPUTS)
    if any(value < 0 or value > _INT32_MAX for value in dims_values):
      raise ValueError("FK arena offset exceeds the int32 shader ABI")
    outputs["dims"] = torch.tensor(
        dims_values, dtype=torch.int32, device=self._device)

    ntree = int(np.max(np.asarray(m.body_treeid), initial=-1)) + 1
    mocap_values = batch_size * int(m.nmocap) * 7 if int(m.nmocap) else 1
    tree_offset = mocap_values
    mocap_id_offset = tree_offset + int(m.nbody)
    awake_offset = mocap_id_offset + int(m.nbody)
    valid_offset = awake_offset + batch_size * max(ntree, 1)
    mismatch_offset = valid_offset + batch_size
    auxiliary_words = _fk_auxiliary_words(
        batch_size, int(m.nmocap), int(m.nbody), ntree)
    if auxiliary_words != max(mismatch_offset + batch_size * max(ntree, 1), 1):
      raise RuntimeError("FK auxiliary layout disagrees with capacity helper")
    _validate_workspace_index_capacity(
        batch_size, {"auxiliary": auxiliary_words},
        {"auxiliary_words": auxiliary_words})
    outputs["auxiliary"] = torch.zeros(
        auxiliary_words, dtype=torch.float32, device=self._device)
    outputs["qpos"] = torch.empty(
        max(batch_size * m.nq, 1), dtype=torch.float32, device=self._device)
    if int(m.nbody):
      outputs["auxiliary"][tree_offset:mocap_id_offset].copy_(
          torch.as_tensor(np.asarray(m.body_treeid, dtype=np.float32).copy(),
                          dtype=torch.float32, device=self._device))
      outputs["auxiliary"][mocap_id_offset:awake_offset].copy_(
          torch.as_tensor(np.asarray(m.body_mocapid, dtype=np.float32).copy(),
                          dtype=torch.float32, device=self._device))
    outputs["tree_awake"] = torch.ones(
        (batch_size, max(ntree, 1)), dtype=torch.int32, device=self._device)
    outputs["tree_awake_cast"] = torch.empty(
        (batch_size, max(ntree, 1)), dtype=torch.int32, device=self._device)
    outputs["all_world_mask"] = torch.ones(
        (batch_size,), dtype=torch.int32, device=self._device)
    outputs["cache_valid"] = outputs["auxiliary"][valid_offset:valid_offset + batch_size]
    outputs["pose_mismatch"] = outputs["auxiliary"][
        mismatch_offset:mismatch_offset + batch_size * max(ntree, 1)].view(
            batch_size, max(ntree, 1)).view(torch.int32)
    outputs["ntree"] = ntree
    self._workspace = {"batch_size": batch_size, "outputs": outputs}

  def invalidate_cache(self, env_ids=None):
    """Invalidate cached world poses after externally supplied state changes.

    ``env_ids`` may be a host sequence of batch rows. The operation changes
    only a device validity flag; it never reads device data back to the host.
    """
    if self._workspace is None:
      return
    valid = self._workspace["outputs"]["cache_valid"]
    if env_ids is None:
      valid.zero_()
    else:
      rows = np.asarray(env_ids, dtype=np.int64).reshape(-1)
      if rows.size and (rows.min() < 0 or rows.max() >= valid.numel()):
        raise ValueError("env_ids contain a row outside the FK workspace")
      if rows.size:
        valid[rows.tolist()] = 0

  @staticmethod
  def _check_device_tensor(value, name, shape, torch, device):
    """Check only host-visible tensor metadata; never synchronizes MPS."""
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != device.type:
      raise ValueError(f"{name} must be on {device}")
    if value.dtype != torch.float32:
      raise ValueError(f"{name} must have dtype torch.float32")
    if tuple(value.shape) != tuple(shape):
      raise ValueError(f"{name} must have shape {tuple(shape)}")
    if not value.is_contiguous():
      raise ValueError(f"{name} must be contiguous")

  def run_device(self, qpos, mocap_pos=None, mocap_quat=None, *, tree_awake=None,
                 world_mask=None):
    """Run FK from a contiguous MPS float32 state without host readback.

    State values are trusted to be finite; free/ball quaternions must be
    nonzero (they are normalized by the shader). Values are validated when a
    device state is reset. This method performs metadata checks only. Returned
    tensors are borrowed workspace views, valid until the next workspace use.

    `mocap_pos`/`mocap_quat` are contiguous float32 MPS tensors with shapes
    ``(batch, nmocap, 3)`` and ``(batch, nmocap, 4)``. They are required when
    the model has mocap bodies and must be omitted otherwise.
    """
    torch = self._torch
    if not isinstance(qpos, torch.Tensor) or qpos.ndim != 2:
      raise TypeError("qpos must be a rank-2 torch.Tensor")
    batch = qpos.shape[0]
    if batch <= 0 or qpos.shape[1] != self.model.nq:
      raise ValueError(
          f"qpos must have shape (batch, {self.model.nq}) with batch > 0"
      )
    self._check_device_tensor(
        qpos, "qpos", (batch, self.model.nq), torch, self._device
    )
    nmocap = int(self.model.nmocap)
    if nmocap == 0:
      if mocap_pos is not None or mocap_quat is not None:
        raise ValueError("model has no mocap bodies")
    else:
      if mocap_pos is None or mocap_quat is None:
        raise ValueError(
            f"mocap_pos/mocap_quat are required for {nmocap} mocap bodies"
        )
      self._check_device_tensor(
          mocap_pos, "mocap_pos", (batch, nmocap, 3), torch, self._device
      )
      self._check_device_tensor(
          mocap_quat, "mocap_quat", (batch, nmocap, 4), torch, self._device
      )
    workspace = self._workspace
    if workspace is None or workspace["batch_size"] != batch:
      raise ValueError(
          "call prepare_workspace(batch_size) before using this batch size"
      )
    out = workspace["outputs"]
    if world_mask is None:
      world_mask = out["all_world_mask"]
    elif (not isinstance(world_mask, torch.Tensor)
          or tuple(world_mask.shape) != (batch,)
          or world_mask.dtype != torch.int32
          or world_mask.device.type != self._device.type
          or not world_mask.is_contiguous()):
      raise ValueError("world_mask must be contiguous int32 [batch] on MPS")
    auxiliary = out["auxiliary"]
    ntree = out["ntree"]
    if tree_awake is None:
      # The public default remains a complete FK evaluation. Simulation passes
      # its authoritative device scheduler mask when sleep is enabled.
      awake = out["tree_awake"]
    else:
      # DeviceSleepScheduler owns int32 state, while the FK auxiliary ABI is
      # float32. Accept its native mask directly and cast during the copy into
      # the preallocated auxiliary buffer; do not allocate a per-stage cast.
      if not isinstance(tree_awake, torch.Tensor):
        raise TypeError("tree_awake must be a torch.Tensor")
      if tree_awake.device.type != self._device.type:
        raise ValueError(f"tree_awake must be on {self._device}")
      if tree_awake.dtype not in (torch.int32, torch.float32):
        raise ValueError("tree_awake must have dtype torch.int32 or torch.float32")
      if tuple(tree_awake.shape) != (batch, max(ntree, 1)):
        raise ValueError(
            f"tree_awake must have shape {(batch, max(ntree, 1))}")
      if not tree_awake.is_contiguous():
        raise ValueError("tree_awake must be contiguous")
      awake = tree_awake
      if tree_awake.dtype == torch.float32:
        out["tree_awake_cast"].copy_(tree_awake)
        awake = out["tree_awake_cast"]
    # Auxiliary inputs are copied on the device only for selected worlds.
    # They share the exact same predicate as the subsequent FK producer.
    zero = out["qpos"]
    self._prepare_rows_kernel(
        world_mask,
        mocap_pos.reshape(-1) if nmocap else zero,
        mocap_quat.reshape(-1) if nmocap else zero,
        awake.reshape(-1), auxiliary, out["dims"],
        threads=(batch,), group_size=(1,))
    # A reshape is a device view. For nq=0, the kernel reads the dummy element;
    # this copy keeps the ABI buffer valid without allocating CPU state.
    if self.model.nq:
      qbuf = qpos.reshape(-1)
    else:
      qbuf = out["qpos"]
    self._kernel(
        self._arrays["static_int"], self._arrays["static_float"], qbuf,
        out["pose_output"], world_mask, auxiliary, out["dims"],
        threads=(batch,), group_size=(1,))
    result = {}
    for kind, count in (
        ("body", self.model.nbody),
        ("geom", self.model.ngeom),
        ("site", self.model.nsite),
        ("inertial", self.model.nbody),
    ):
      result[f"{kind}_pos"] = _shape_output(out[f"{kind}_pos"], batch, count, 3)
      result[f"{kind}_quat"] = _shape_output(
          out[f"{kind}_quat"], batch, count, 4
      )
      result[f"{kind}_quat_low"] = _shape_output(
          out[f"{kind}_quat_low"], batch, count, 4)
      result[f"{kind}_quat_tail"] = _shape_output(
          out[f"{kind}_quat_tail"], batch, count, 4)
      result[f"{kind}_pos_low"] = _shape_output(
          out[f"{kind}_pos_low"], batch, count, 3)
      result[f"{kind}_pos_tail"] = _shape_output(
          out[f"{kind}_pos_tail"], batch, count, 3)
    for name in ("geom_xmat", "geom_xmat_low", "geom_xmat_tail"):
      result[name] = _shape_output(out[name], batch, self.model.ngeom, 9)
    for name in ("joint_anchor", "joint_axis"):
      result[name] = _shape_output(out[name], batch, self.model.njnt, 3)
    result["body_pos_low"] = _shape_output(
        out["body_pos_low"], batch, self.model.nbody, 3)
    result["body_pos_tail"] = _shape_output(
        out["body_pos_tail"], batch, self.model.nbody, 3)
    result["geom_pos_low"] = _shape_output(
        out["geom_pos_low"], batch, self.model.ngeom, 3)
    result["geom_pos_tail"] = _shape_output(
        out["geom_pos_tail"], batch, self.model.ngeom, 3)
    return result

  def pose_mismatch(self):
    """Return the borrowed device mismatch mask from the most recent FK."""
    if self._workspace is None:
      raise RuntimeError("prepare_workspace(batch_size) before reading FK status")
    return self._workspace["outputs"]["pose_mismatch"]

  def run(self, qpos, mocap_pos=None, mocap_quat=None):
    """Compute full world poses for a CPU qpos batch; returns MPS tensors."""
    source = np.asarray(qpos, dtype=np.float64)
    if source.ndim == 1:
      source = source[None, :]
    if (
        source.ndim != 2
        or source.shape[1] != self.model.nq
        or source.shape[0] == 0
    ):
      raise ValueError(
          f"qpos must have shape (batch, {self.model.nq}) with batch > 0"
      )
    if not np.all(np.isfinite(source)):
      raise ValueError("qpos must be finite")
    batch = source.shape[0]
    nmocap = int(self.model.nmocap)
    if nmocap == 0:
      if mocap_pos is not None or mocap_quat is not None:
        raise ValueError("model has no mocap bodies")
      mocap_flat_host = np.zeros(1, dtype=np.float32)
    else:
      if mocap_pos is None or mocap_quat is None:
        checked_pos, checked_quat = self.model._checked_mocap(None, None)
        checked_pos = np.broadcast_to(checked_pos, (batch, nmocap, 3)).copy()
        checked_quat = np.broadcast_to(checked_quat, (batch, nmocap, 4)).copy()
      else:
        pos_in = np.asarray(mocap_pos, dtype=np.float64)
        quat_in = np.asarray(mocap_quat, dtype=np.float64)
        if pos_in.shape == (nmocap, 3):
          pos_in = np.broadcast_to(pos_in, (batch, nmocap, 3)).copy()
        if quat_in.shape == (nmocap, 4):
          quat_in = np.broadcast_to(quat_in, (batch, nmocap, 4)).copy()
        checked_pos, checked_quat = self.model._checked_mocap(pos_in, quat_in)
      mocap_flat_host = np.concatenate(
          [np.asarray(checked_pos, dtype=np.float32).reshape(-1),
           np.asarray(checked_quat, dtype=np.float32).reshape(-1)]
      )
    q = np.array(source, dtype=np.float32, order="C", copy=True)
    if not np.all(np.isfinite(q)):
      raise ValueError("qpos cannot be represented as finite float32")
    for typ, qa in zip(self.model.jnt_type, self.model.jnt_qposadr):
      start = int(qa) + (3 if int(typ) == 0 else 0)
      width = 4 if int(typ) in (0, 1) else 0
      if width:
        norms = np.linalg.norm(
            q[:, start : start + width].astype(np.float64), axis=1
        )
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
          raise ValueError(
              "free/ball quaternion must be finite and nonzero in float32"
          )
        q[:, start : start + width] = (
            q[:, start : start + width].astype(np.float64) / norms[:, None]
        ).astype(np.float32)
    torch = self._torch
    self.prepare_workspace(batch)
    q_tensor = torch.as_tensor(q, dtype=torch.float32, device=self._device).contiguous()
    if nmocap:
      mocap_pos_tensor = torch.as_tensor(
          np.array(checked_pos, dtype=np.float32, copy=True),
          dtype=torch.float32, device=self._device).contiguous()
      mocap_quat_tensor = torch.as_tensor(
          np.array(checked_quat, dtype=np.float32, copy=True),
          dtype=torch.float32, device=self._device).contiguous()
    else:
      mocap_pos_tensor = None
      mocap_quat_tensor = None
    return self.run_device(q_tensor, mocap_pos_tensor, mocap_quat_tensor)
