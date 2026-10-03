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

"""Explicit opt-in Metal dense mass-matrix stage; MPS output stays on device."""

from pathlib import Path

import numpy as np

from mujoco_metal.metal_kinematics import _prepare_host_arrays
from mujoco_metal.metal_kinematics import MetalKinematics
from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
from mujoco_metal.model import ModelDescriptor

_SHADER = Path(__file__).parent / "shaders" / "smooth_mass.metal"
_BIAS_SHADER = Path(__file__).parent / "shaders" / "smooth_bias.metal"
_DEVICE_ARRAYS = (
    "body_parentid",
    "body_rootid",
    "body_jntadr",
    "body_jntnum",
    "body_dofadr",
    "body_dofnum",
    "dof_parentid",
    "dof_bodyid",
    "jnt_type",
    "jnt_dofadr",
    "body_mass",
    "body_inertia",
    "dof_armature",
    "gravity",
)


def bias_derivative_workspace_sizes(nbody, nv, batch_size):
  """Checked sizes for analytical per-column RNE derivatives, without MPS."""
  for name, value in (("nbody", nbody), ("nv", nv)):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
      raise ValueError(f"{name} must be a nonnegative integer")
  stride = 6 * (3 * nbody + nv)
  sizes = {"bias_derivative_scratch": batch_size * nv * stride,
           "bias_derivative": batch_size * nv * nv}
  _validate_workspace_index_capacity(batch_size, sizes,
      {"nbody": nbody, "nv": nv, "derivative_stride": stride})
  return sizes


def validate_pose_dict(model, poses, batch, device, torch, *, check_finite=True):
  """Validate the full device pose ABI consumed by native smooth stages."""
  if not isinstance(poses, dict):
    raise TypeError("poses must be a position-stage pose dictionary")
  shapes = {
      "body_pos": (batch, model.nbody, 3),
      "body_quat": (batch, model.nbody, 4),
      "geom_pos": (batch, model.ngeom, 3),
      "geom_quat": (batch, model.ngeom, 4),
      "site_pos": (batch, model.nsite, 3),
      "site_quat": (batch, model.nsite, 4),
      "inertial_pos": (batch, model.nbody, 3),
      "inertial_quat": (batch, model.nbody, 4),
      "joint_anchor": (batch, model.njnt, 3),
      "joint_axis": (batch, model.njnt, 3),
  }
  if set(poses) != set(shapes):
    raise ValueError("poses must contain the complete position-stage ABI")
  for name, shape in shapes.items():
    value = poses[name]
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
        value.dtype != torch.float32 or not value.is_contiguous() or
        value.device.type != device.type or
        (device.index is not None and value.device.index != device.index)):
      raise ValueError(f"poses.{name} has an invalid shape, dtype, layout, or device")
    if check_finite and not bool(torch.isfinite(value).all()):
      raise ValueError(f"poses.{name} contains nonfinite values")


class MetalSmoothDynamics:
  """Batched native MPS smooth inertial dynamics stage.

  ``mass_matrix(qpos_batch)`` returns an MPS tensor with shape ``[B,nv,nv]``.
  ``run(qpos_batch, qvel_batch)`` returns the mass matrix and inertial/gravity
  bias as MPS tensors. It does not compute actuation, passive forces, contacts,
  constraints, or physics steps. Construction explicitly initializes MPS and
  compiles shaders.
  """

  def __init__(self, model: ModelDescriptor, batch_size: int = 1):
    from mujoco_metal.model import actuator_joint_inheritance
    # R06-1: joint-targeted actuator armature folds into the dof armature
    # (pinned mj_actuatorArmature gear^2 scan); tendon-targeted handled by tendon stage.
    arm_fold, _, _ = actuator_joint_inheritance(model, tendon_ok=True)
    if np.any(model.tendon_armature != 0):
      raise ValueError("Metal smooth stage does not support tendon armature")
    host = _prepare_host_arrays(model)
    host["dof_armature"] = np.asarray(
        np.asarray(host["dof_armature"], dtype=np.float64) + arm_fold,
        dtype=np.float32)
    self._fk = MetalKinematics(model, batch_size=batch_size)
    self.model = self._fk.model
    torch = self._fk._torch
    self._torch = torch
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.dense_mass_matrix
    self._pose_finite_kernel = self._library.pose_finite_status
    self._bias_library = torch.mps.compile_shader(_BIAS_SHADER.read_text())
    self._bias_kernel = self._bias_library.smooth_bias
    self._bias_deriv_kernel = self._bias_library.smooth_bias_derivative
    self._arrays = {
        name: torch.from_numpy(host[name]).to(self._fk._device)
        for name in _DEVICE_ARRAYS
    }
    self.prepare_workspace(batch_size)

  def prepare_workspace(self, batch_size: int):
    """Preallocate all smooth-stage buffers for an explicit world batch."""
    _validate_workspace_index_capacity(batch_size, {}, {})
    nb, nv = self.model.nbody, self.model.nv
    _validate_workspace_index_capacity(
        batch_size,
        {
            "mass": batch_size * nv * nv,
            "pose_status": batch_size,
            "root_com": batch_size * nb * 3,
            "cdof": batch_size * nv * 6,
            "crb": batch_size * nb * 36,
            "local_inertia": batch_size * nb * 36,
            "qvel": batch_size * nv,
            "bias": batch_size * nv,
            "cvel": batch_size * nb * 6,
            "cdof_dot": batch_size * nv * 6,
            "cacc": batch_size * nb * 6,
            "body_force": batch_size * nb * 6,
        },
        {"nbody": nb, "njnt": self.model.njnt, "nv": nv},
    )
    self._fk.prepare_workspace(batch_size)
    torch, device = self._torch, self._fk._device

    def buffer(size):
      return torch.empty(max(size, 1), dtype=torch.float32, device=device)

    self._workspace = {
        "batch_size": batch_size,
        "mass": buffer(batch_size * nv * nv),
        "pose_status": torch.zeros(batch_size, dtype=torch.int32, device=device),
        "pose_finite_dims": torch.tensor(
            [nb, self.model.ngeom, self.model.nsite, self.model.njnt, batch_size],
            dtype=torch.int32, device=device),
        "root_com": buffer(batch_size * nb * 3),
        "cdof": buffer(batch_size * nv * 6),
        "crb": buffer(batch_size * nb * 36),
        "local_inertia": buffer(batch_size * nb * 36),
        "qvel": buffer(batch_size * nv),
        "bias": buffer(batch_size * nv),
        "cvel": buffer(batch_size * nb * 6),
        "cdof_dot": buffer(batch_size * nv * 6),
        "cacc": buffer(batch_size * nb * 6),
        "body_force": buffer(batch_size * nb * 6),
        "mass_dims": torch.tensor(
            [nb, self.model.njnt, nv, batch_size],
            dtype=torch.int32,
            device=device,
        ),
        "bias_dims": torch.tensor(
            [nb, self.model.njnt, nv, batch_size],
            dtype=torch.int32,
            device=device,
        ),
        "disableflags": torch.tensor(
            [self.model.disableflags], dtype=torch.int32, device=device
        ),
    }

  def run_device(self, qpos, qvel, mocap_pos=None, mocap_quat=None, poses=None, *,
                 _trusted_internal_poses=False):
    """Compute M(q) and bias from borrowed MPS float32 state tensors.

    Inputs are trusted to be finite and have finite nonzero free/ball
    quaternions. Device-state reset owns value validation; this hot path checks
    tensor metadata only and never reads a device value back to the host.
    Internal poses report nonfinite worlds through device pose_status.
    Supplied external poses retain strict synchronous boundary validation.
    Results borrow persistent workspace and remain valid until its next use.

    Mocap poses follow the `MetalKinematics.run_device` contract (required
    when the model has mocap bodies, omitted otherwise).
    """
    torch = self._torch
    if not isinstance(qpos, torch.Tensor) or qpos.ndim != 2:
      raise TypeError("qpos must be a rank-2 torch.Tensor")
    if not isinstance(qvel, torch.Tensor) or qvel.ndim != 2:
      raise TypeError("qvel must be a rank-2 torch.Tensor")
    batch = qpos.shape[0]
    if batch <= 0 or qpos.shape[1] != self.model.nq:
      raise ValueError(
          f"qpos must have shape (batch, {self.model.nq}) with batch > 0"
      )
    self._fk._check_device_tensor(
        qpos, "qpos", (batch, self.model.nq), torch, self._fk._device
    )
    self._fk._check_device_tensor(
        qvel, "qvel", (batch, self.model.nv), torch, self._fk._device
    )
    if self._workspace["batch_size"] != batch:
      raise ValueError(
          "call prepare_workspace(batch_size) before using this batch size"
      )

    # The split-stage API may pass a position-stage result. Validate that
    # complete borrowed pose ABI and consume it directly; do not recompute FK
    # (which can overwrite the caller's position-stage workspace).
    internal_pose_values = poses is None or _trusted_internal_poses
    if poses is None:
      poses = self._fk.run_device(qpos, mocap_pos, mocap_quat)
    else:
      if mocap_pos is not None or mocap_quat is not None:
        raise ValueError("mocap inputs cannot accompany supplied poses")
      validate_pose_dict(self.model, poses, batch, self._fk._device, torch,
                         check_finite=not _trusted_internal_poses)
    w, arrays = self._workspace, self._arrays
    if internal_pose_values:
      pose_buffers = []
      for name in ("body_pos", "body_quat", "geom_pos", "geom_quat",
                   "site_pos", "site_quat", "inertial_pos", "inertial_quat",
                   "joint_anchor", "joint_axis"):
        value = poses[name]
        pose_buffers.append(value.reshape(-1) if value.numel() else w["bias"])
      self._pose_finite_kernel(
          *pose_buffers, w["pose_finite_dims"], w["pose_status"],
          threads=(batch,), group_size=(1,))
    else:
      w["pose_status"].zero_()
    qvel_flat = qvel.reshape(-1) if self.model.nv else w["qvel"]
    args = [
        arrays[name]
        for name in (
            "body_parentid",
            "body_rootid",
            "body_jntadr",
            "body_jntnum",
            "dof_parentid",
            "dof_bodyid",
            "jnt_type",
            "jnt_dofadr",
            "body_mass",
            "body_inertia",
            "dof_armature",
        )
    ]
    args.extend(
        [
            poses["body_quat"].reshape(-1),
            poses["inertial_pos"].reshape(-1),
            poses["inertial_quat"].reshape(-1),
            poses["joint_anchor"].reshape(-1),
            poses["joint_axis"].reshape(-1),
            w["mass"],
            w["root_com"],
            w["cdof"],
            w["crb"],
            w["mass_dims"],
            w["local_inertia"],
        ]
    )
    self._kernel(*args, threads=(batch,), group_size=(1,))
    bias_args = [
        arrays["body_parentid"],
        arrays["body_dofadr"],
        arrays["body_dofnum"],
        arrays["body_jntadr"],
        arrays["body_jntnum"],
        arrays["dof_bodyid"],
        arrays["jnt_type"],
        arrays["jnt_dofadr"],
        arrays["gravity"],
        w["cdof"],
        w["local_inertia"],
        w["disableflags"],
        qvel_flat,
        w["cvel"],
        w["cdof_dot"],
        w["cacc"],
        w["body_force"],
        w["bias"],
        w["bias_dims"],
    ]
    self._bias_kernel(*bias_args, threads=(batch,), group_size=(1,))
    nv = self.model.nv
    return {
        "mass_matrix": w["mass"][: batch * nv * nv].reshape(batch, nv, nv),
        "pose_status": w["pose_status"],
        "qfrc_bias": w["bias"][: batch * nv].reshape(batch, nv),
        "poses": poses,
        "cvel": w["cvel"][: batch * self.model.nbody * 6].reshape(
            batch, self.model.nbody, 6
        ),
        "root_com": w["root_com"][: batch * self.model.nbody * 3].reshape(
            batch, self.model.nbody, 3
        ),
        "cdof": w["cdof"][: batch * nv * 6].reshape(batch, nv, 6),
        "cdof_dot": w["cdof_dot"][: batch * nv * 6].reshape(batch, nv, 6),
    }

  def _compute_mass_matrix(self, qpos_batch, mocap_pos=None, mocap_quat=None):
    source = np.asarray(qpos_batch, dtype=np.float64)
    if source.ndim == 1:
      source = source[None, :]
    if (
        source.ndim != 2
        or source.shape[0] == 0
        or source.shape[1] != self.model.nq
        or not np.all(np.isfinite(source))
    ):
      raise ValueError(
          f"qpos must be finite with shape (batch, {self.model.nq}) "
          "and batch > 0"
      )
    qpos32 = np.asarray(source, dtype=np.float32)
    if not np.all(np.isfinite(qpos32)):
      raise ValueError("qpos cannot be represented as finite float32")

    # FK validates and normalizes its own float32 input before allocating MPS.
    poses = self._fk.run(source, mocap_pos, mocap_quat)
    batch = source.shape[0]
    nv, nb = self.model.nv, self.model.nbody
    torch = self._torch
    output = torch.empty(
        max(batch * nv * nv, 1), dtype=torch.float32, device=self._fk._device
    )
    root_com = torch.empty(
        max(batch * nb * 3, 1), dtype=torch.float32, device=self._fk._device
    )
    cdof = torch.empty(
        max(batch * nv * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    crb = torch.empty(
        max(batch * nb * 36, 1), dtype=torch.float32, device=self._fk._device
    )
    local_inertia = torch.empty(
        max(batch * nb * 36, 1), dtype=torch.float32, device=self._fk._device
    )
    arrays = self._arrays
    args = [
        arrays["body_parentid"],
        arrays["body_rootid"],
        arrays["body_jntadr"],
        arrays["body_jntnum"],
        arrays["dof_parentid"],
        arrays["dof_bodyid"],
        arrays["jnt_type"],
        arrays["jnt_dofadr"],
        arrays["body_mass"],
        arrays["body_inertia"],
        arrays["dof_armature"],
        poses["body_quat"].reshape(-1),
        poses["inertial_pos"].reshape(-1),
        poses["inertial_quat"].reshape(-1),
        poses["joint_anchor"].reshape(-1),
        poses["joint_axis"].reshape(-1),
        output,
        root_com,
        cdof,
        crb,
        torch.tensor(
            [nb, self.model.njnt, nv, batch],
            dtype=torch.int32,
            device=self._fk._device,
        ),
        local_inertia,
    ]
    if len(args) != 22:
      raise RuntimeError("native dense mass shader buffer ABI mismatch")
    self._kernel(*args, threads=(batch,), group_size=(1,))
    return {
        "mass_matrix": output[: batch * nv * nv].reshape(batch, nv, nv),
        "root_com": root_com,
        "cdof": cdof,
        "local_inertia": local_inertia,
        "poses": poses,
        "batch": batch,
    }

  def mass_matrix(self, qpos_batch):
    """Compute dense generalized inertia for a batch, returning MPS output."""
    return self._compute_mass_matrix(qpos_batch)["mass_matrix"]

  def run(self, qpos_batch, qvel_batch, mocap_pos=None, mocap_quat=None):
    """Return batched mass matrices and inertial/gravity bias on MPS."""
    qpos = np.asarray(qpos_batch, dtype=np.float64)
    if qpos.ndim == 1:
      qpos = qpos[None, :]
    if (
        qpos.ndim != 2
        or qpos.shape[0] == 0
        or qpos.shape[1] != self.model.nq
        or not np.all(np.isfinite(qpos))
    ):
      raise ValueError(
          f"qpos must be finite with shape (batch, {self.model.nq})"
      )
    if not np.all(np.isfinite(np.asarray(qpos, dtype=np.float32))):
      raise ValueError("qpos cannot be represented as finite float32")
    qvel = np.asarray(qvel_batch, dtype=np.float64)
    if qvel.ndim == 1:
      qvel = qvel[None, :]
    if (
        qvel.ndim != 2
        or qvel.shape != (qpos.shape[0], self.model.nv)
        or qpos.shape[0] == 0
        or not np.all(np.isfinite(qvel))
    ):
      raise ValueError(
          f"qvel must be finite with shape (batch, {self.model.nv})"
      )
    qvel32 = np.asarray(qvel, dtype=np.float32)
    if not np.all(np.isfinite(qvel32)):
      raise ValueError("qvel cannot be represented as finite float32")

    computed = self._compute_mass_matrix(qpos)
    batch, nb, nv = computed["batch"], self.model.nbody, self.model.nv
    torch = self._torch
    qvel_flat = qvel32.reshape(-1)
    if qvel_flat.size == 0:
      qvel_flat = np.zeros(1, dtype=np.float32)
    qvel_tensor = torch.from_numpy(qvel_flat).to(self._fk._device)
    bias = torch.empty(
        max(batch * nv, 1), dtype=torch.float32, device=self._fk._device
    )
    cvel = torch.empty(
        max(batch * nb * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    cdof_dot = torch.empty(
        max(batch * nv * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    cacc = torch.empty(
        max(batch * nb * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    body_force = torch.empty(
        max(batch * nb * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    args = [
        self._arrays["body_parentid"],
        self._arrays["body_dofadr"],
        self._arrays["body_dofnum"],
        self._arrays["body_jntadr"],
        self._arrays["body_jntnum"],
        self._arrays["dof_bodyid"],
        self._arrays["jnt_type"],
        self._arrays["jnt_dofadr"],
        self._arrays["gravity"],
        computed["cdof"],
        computed["local_inertia"],
        torch.tensor(
            [self.model.disableflags],
            dtype=torch.int32,
            device=self._fk._device,
        ),
        qvel_tensor,
        cvel,
        cdof_dot,
        cacc,
        body_force,
        bias,
        torch.tensor(
            [nb, self.model.njnt, nv, batch],
            dtype=torch.int32,
            device=self._fk._device,
        ),
    ]
    if len(args) != 19:
      raise RuntimeError("native smooth bias shader buffer ABI mismatch")
    self._bias_kernel(*args, threads=(batch,), group_size=(1,))
    return {
        "mass_matrix": computed["mass_matrix"],
        "qfrc_bias": bias[: batch * nv].reshape(batch, nv),
    }

  def bias_derivative_device(self, qpos, qvel, dynamics=None):
    """Compute analytical -d(qfrc_bias)/d(qvel) in reusable device storage.

    Held-position RNE is differentiated directly; there are no perturbed
    physics evaluations or fixed-size body/DOF arrays. Returned storage is
    borrowed and remains valid until the next derivative call.
    """
    torch = self._torch
    batch = qpos.shape[0]
    nv = self.model.nv
    if nv == 0:
      return torch.zeros((batch, 0, 0), dtype=torch.float32, device=self._fk._device)
    if dynamics is None:
      dynamics = self.run_device(qpos, qvel)
    w, arrays = self._workspace, self._arrays
    if batch != w["batch_size"]:
      raise ValueError("bias derivative batch does not match prepared smooth workspace")
    if "bias_derivative" not in w:
      sizes = bias_derivative_workspace_sizes(self.model.nbody, nv, batch)
      for key, count in sizes.items():
        w[key] = torch.empty(max(count, 1), dtype=torch.float32, device=self._fk._device)
    qderiv = w["bias_derivative"].reshape(batch, nv, nv)
    cdof = dynamics["cdof"] if "cdof" in dynamics else w["cdof"]
    local_inertia = dynamics["local_inertia"] if "local_inertia" in dynamics else w["local_inertia"]
    args = [
        arrays["body_parentid"],
        arrays["body_dofadr"],
        arrays["body_dofnum"],
        arrays["body_jntadr"],
        arrays["body_jntnum"],
        arrays["dof_bodyid"],
        arrays["jnt_type"],
        arrays["jnt_dofadr"],
        arrays["gravity"],
        cdof.reshape(-1),
        local_inertia.reshape(-1),
        w["disableflags"],
        qvel.reshape(-1),
        qderiv.reshape(-1),
        w["bias_dims"],
        dynamics.get("cvel", w["cvel"]).reshape(-1),
        dynamics.get("cdof_dot", w["cdof_dot"]).reshape(-1),
        w["bias_derivative_scratch"],
    ]
    self._bias_deriv_kernel(*args, threads=(nv, batch), group_size=(1, 1))
    return qderiv
