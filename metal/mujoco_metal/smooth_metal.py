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

from mujoco_metal.mass_layout import compile_tree_mass_layout

from mujoco_metal.metal_kinematics import _prepare_host_arrays
from mujoco_metal.metal_kinematics import MetalKinematics
from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity


def position_context_shapes(model, batch, *, mass_storage="dense", component_nnz=0):
  """Exact fixed backings for the internal retained forward position stage."""
  shapes = {}
  for kind, count in (("body", model.nbody), ("inertial", model.nbody),
                      ("geom", model.ngeom), ("site", model.nsite)):
    shapes[f"poses.{kind}_pos"] = (batch, count, 3)
    shapes[f"poses.{kind}_pos_low"] = (batch, count, 3)
    shapes[f"poses.{kind}_pos_tail"] = (batch, count, 3)
    shapes[f"poses.{kind}_quat"] = (batch, count, 4)
    shapes[f"poses.{kind}_quat_low"] = (batch, count, 4)
    shapes[f"poses.{kind}_quat_tail"] = (batch, count, 4)
  # FK publishes the source-order rotation matrix alongside its quaternion.
  # Preserve all five residual-bearing geom rotation words through POS->VEL.
  for name in ("geom_xmat", "geom_xmat_low", "geom_xmat_tail"):
    shapes[f"poses.{name}"] = (batch, model.ngeom, 9)
  for name in ("joint_anchor", "joint_axis"):
    shapes[f"poses.{name}"] = (batch, model.njnt, 3)
  shapes.update(root_com=(batch, model.nbody, 3), cdof=(batch, model.nv, 6),
                pose_status=(batch,), local_inertia=(batch, model.nbody, 36))
  if mass_storage == "dense":
    shapes["mass_matrix"] = (batch, model.nv, model.nv)
    shapes["tendon_armature_matrix"] = (batch, model.nv, model.nv)
  elif mass_storage == "block_sparse":
    shapes["mass_blocks"] = (batch, component_nnz)
    shapes["tendon_armature_blocks"] = (batch, component_nnz)
  else:
    raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
  return shapes


def position_context_workspace_sizes(model, batch, *, mass_storage="dense",
                                     component_nnz=0):
  return {f"position_cache.{name}": max(int(np.prod(shape, dtype=object)), 1)
          for name, shape in position_context_shapes(
              model, batch, mass_storage=mass_storage,
              component_nnz=component_nnz).items()}
from mujoco_metal.model import ModelDescriptor

_SHADER = Path(__file__).parent / "shaders" / "smooth_mass.metal"
_BIAS_SHADER = Path(__file__).parent / "shaders" / "smooth_bias.metal"
_DEVICE_ARRAYS = (
    "body_parentid",
    "body_rootid",
    "body_treeid",
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
    "tendon_armature",
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
      "joint_anchor": (batch, model.njnt, 3),
      "joint_axis": (batch, model.njnt, 3),
  }
  for kind, count in (("body", model.nbody), ("inertial", model.nbody),
                      ("geom", model.ngeom), ("site", model.nsite)):
    shapes[f"{kind}_pos"] = (batch, count, 3)
    shapes[f"{kind}_pos_low"] = (batch, count, 3)
    shapes[f"{kind}_pos_tail"] = (batch, count, 3)
    shapes[f"{kind}_quat"] = (batch, count, 4)
    shapes[f"{kind}_quat_low"] = (batch, count, 4)
    shapes[f"{kind}_quat_tail"] = (batch, count, 4)
  for name in ("geom_xmat", "geom_xmat_low", "geom_xmat_tail"):
    shapes[name] = (batch, model.ngeom, 9)
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

  def __init__(self, model: ModelDescriptor, batch_size: int = 1, *,
               mass_storage="dense", velocity_derivative_layout=None):
    from mujoco_metal.model import actuator_joint_inheritance, actuator_tendon_inheritance
    # R06-1: joint-targeted actuator armature folds into the dof armature
    # (pinned mj_actuatorArmature gear^2 scan); tendon-targeted handled by tendon stage.
    arm_fold, _, _ = actuator_joint_inheritance(model, tendon_ok=True)
    tendon_armature, _, _ = actuator_tendon_inheritance(model)
    effective_tendon_armature = np.asarray(
        model.tendon_armature, dtype=np.float64) + tendon_armature
    if np.any(~np.isfinite(effective_tendon_armature)):
      raise ValueError("effective tendon armature must be finite")
    host = _prepare_host_arrays(model)
    # ModelDescriptor intentionally does not mirror MjModel.ntree. Build this
    # immutable layout from its lowered body-tree and DOF-owner arrays.
    tree_layout = compile_tree_mass_layout(
        host["body_treeid"], host["dof_bodyid"][:int(model.nv)],
        tendon_treeid=np.asarray(model.tendon_treeid, dtype=np.int32),
        tendon_treenum=np.asarray(model.tendon_treenum, dtype=np.int32),
        tendon_armature=effective_tendon_armature,
        tendon_j_rowadr=np.asarray(model.tendon_j_rowadr, dtype=np.int32),
        tendon_j_rownnz=np.asarray(model.tendon_j_rownnz, dtype=np.int32),
        tendon_j_colind=np.asarray(model.tendon_j_colind, dtype=np.int32),
        mass_rowadr=np.asarray(model.mass_rowadr, dtype=np.int32),
        mass_rownnz=np.asarray(model.mass_rownnz, dtype=np.int32),
        mass_colind=np.asarray(model.mass_colind, dtype=np.int32))
    self.ntree = tree_layout["ntree"]
    self.ncomponent = tree_layout["ncomponent"]
    self._tree_dofadr = tree_layout["tree_dofadr"]
    self._tree_dofnum = tree_layout["tree_dofnum"]
    self._tree_mass_offsets = tree_layout["tree_mass_offsets"]
    self._tree_mass_nnz = tree_layout["nnz"]
    self._dof_treeid = tree_layout["dof_treeid"]
    self._tree_layout_device_size = int(tree_layout["component_packed"].size)
    self._component_dof_offsets = tree_layout["component_dof_offsets"]
    self._component_dof_ids = tree_layout["component_dof_ids"]
    self._component_dofnum = tree_layout["component_dofnum"]
    self._component_mass_offsets = tree_layout["component_mass_offsets"]
    self._dof_component = tree_layout["dof_component"]
    self._dof_local_index = tree_layout["dof_local_index"]
    self._effective_tendon_armature = np.asarray(
        effective_tendon_armature, dtype=np.float32)
    host["dof_armature"] = np.asarray(
        np.asarray(host["dof_armature"], dtype=np.float64) + arm_fold,
        dtype=np.float32)
    self._fk = MetalKinematics(model, batch_size=batch_size)
    self.model = self._fk.model
    torch = self._fk._torch
    self._torch = torch
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.dense_mass_matrix
    self._tendon_armature_kernel = self._library.add_tendon_armature
    self._pose_finite_kernel = self._library.pose_finite_status
    self._pose_finite_residual_kernel = self._library.pose_finite_residual_status
    self._bias_library = torch.mps.compile_shader(_BIAS_SHADER.read_text())
    self._bias_kernel = self._bias_library.smooth_bias
    self._bias_deriv_kernel = self._bias_library.smooth_bias_derivative
    self._bias_derivative_layout = velocity_derivative_layout
    if velocity_derivative_layout is None:
      self._bias_column_offsets = self._bias_empty_index = torch.zeros(
          (1,), dtype=torch.int32, device=self._fk._device)
      self._bias_edge_rows = self._bias_column_offsets
      self._bias_edge_slots = self._bias_column_offsets
      self._bias_edge_count = 0
    else:
      if int(velocity_derivative_layout.diagonal_slots.size) != int(model.nv):
        raise ValueError("velocity derivative layout nv does not match smooth model")
      offsets, rows, slots = velocity_derivative_layout.column_edge_map()
      from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
      _validate_workspace_index_capacity(
          batch_size,
          {"smooth.bias_coo_column_offsets": offsets.size,
           "smooth.bias_coo_edge_rows": rows.size,
           "smooth.bias_coo_edge_slots": slots.size},
          {"nv": int(model.nv),
           "edges": int(velocity_derivative_layout.edge_count)})
      self._bias_column_offsets = torch.as_tensor(
          offsets.copy(), dtype=torch.int32, device=self._fk._device)
      self._bias_edge_rows = torch.as_tensor(
          rows.copy(), dtype=torch.int32, device=self._fk._device)
      self._bias_edge_slots = torch.as_tensor(
          slots.copy(), dtype=torch.int32, device=self._fk._device)
      self._bias_edge_count = int(velocity_derivative_layout.edge_count)
    self._arrays = {
        name: torch.from_numpy(host[name]).to(self._fk._device)
        for name in _DEVICE_ARRAYS
    }
    self._tree_layout_device = torch.as_tensor(
        tree_layout["component_packed"].copy(), dtype=torch.int32,
        device=self._fk._device)
    self._dof_treeid_device = torch.as_tensor(
        self._dof_treeid if self._dof_treeid.size else np.zeros(1, np.int32),
        dtype=torch.int32, device=self._fk._device)
    self._tendon_treeid_device = torch.as_tensor(
        np.asarray(model.tendon_treeid, dtype=np.int32).reshape(-1).copy(),
        dtype=torch.int32, device=self._fk._device)
    self._tendon_treenum_device = torch.as_tensor(
        np.asarray(model.tendon_treenum, dtype=np.int32).copy(),
        dtype=torch.int32, device=self._fk._device)
    self._tendon_j_rowadr_device = torch.as_tensor(
        np.asarray(model.tendon_j_rowadr, dtype=np.int32).copy(),
        dtype=torch.int32, device=self._fk._device)
    self._tendon_j_rownnz_device = torch.as_tensor(
        np.asarray(model.tendon_j_rownnz, dtype=np.int32).copy(),
        dtype=torch.int32, device=self._fk._device)
    tendon_j_colind = np.asarray(model.tendon_j_colind, dtype=np.int32).copy()
    self._tendon_j_colind_device = torch.as_tensor(
        tendon_j_colind if tendon_j_colind.size else np.zeros(1, np.int32),
        dtype=torch.int32, device=self._fk._device)
    self._mass_rowadr_device = torch.as_tensor(
        np.asarray(model.mass_rowadr, dtype=np.int32).copy(),
        dtype=torch.int32, device=self._fk._device)
    self._mass_rownnz_device = torch.as_tensor(
        np.asarray(model.mass_rownnz, dtype=np.int32).copy(),
        dtype=torch.int32, device=self._fk._device)
    mass_colind = np.asarray(model.mass_colind, dtype=np.int32).copy()
    self._mass_colind_device = torch.as_tensor(
        mass_colind if mass_colind.size else np.zeros(1, np.int32),
        dtype=torch.int32, device=self._fk._device)
    armature_f32 = np.asarray(effective_tendon_armature, dtype=np.float32)
    self._effective_tendon_armature_device = torch.as_tensor(
        armature_f32 if armature_f32.size else np.zeros(1, np.float32),
        dtype=torch.float32, device=self._fk._device)
    self._has_tendon_armature = bool(np.any(armature_f32 != 0))
    if mass_storage not in ("dense", "block_sparse"):
      raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
    self._mass_storage = mass_storage
    self.prepare_workspace(batch_size, mass_storage=mass_storage)

  def prepare_workspace(self, batch_size: int, *, mass_storage=None):
    """Preallocate all smooth-stage buffers for an explicit world batch."""
    if mass_storage is None:
      mass_storage = self._mass_storage
    if mass_storage not in ("dense", "block_sparse"):
      raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
    self._mass_storage = mass_storage
    _validate_workspace_index_capacity(batch_size, {}, {})
    nb, nv = self.model.nbody, self.model.nv
    cache_sizes = position_context_workspace_sizes(
        self.model, batch_size, mass_storage=mass_storage,
        component_nnz=self._tree_mass_nnz)
    bias_sparse_scratch = (
        batch_size * nv * 6 * (3 * nb + nv)
        if self._bias_derivative_layout is not None else 0)
    _validate_workspace_index_capacity(
        batch_size,
        {
            "mass": batch_size * nv * nv if mass_storage == "dense" else 0,
            "mass_blocks": batch_size * self._tree_mass_nnz if mass_storage == "block_sparse" else 0,
            "mass_sparse_product": batch_size * nv,
            "mass_armature": batch_size * (nv * nv if mass_storage == "dense" else self._tree_mass_nnz),
            "tree_layout": self._tree_layout_device_size,
            "dof_treeid": max(nv, 1),
            "tendon_j_structure": int(
                self.model.ntendon_jnnz + 3 * self.model.ntendon),
            "mass_j_structure": int(
                self.model.nmass_nnz + 2 * self.model.nv),
            "root_com": batch_size * nb * 3,
            "cdof": batch_size * nv * 6,
            "crb": batch_size * nb * 36,
            "local_inertia": batch_size * nb * 36,
            "qvel": batch_size * nv,
            "bias": batch_size * nv,
            "bias_low": batch_size * nv,
            "cvel_low": batch_size * nb * 6,
            "cdof_dot_low": batch_size * nv * 6,
            "cacc_low": batch_size * nb * 6,
            "body_force_low": batch_size * nb * 6,
            "cvel": batch_size * nb * 6,
            "cdof_dot": batch_size * nv * 6,
            "cacc": batch_size * nb * 6,
            "body_force": batch_size * nb * 6,
            "awake_ids": batch_size * (2 * max(nb, 1) + max(nv, 1)),
            "awake_counts": batch_size * 3,
            "pose_status": batch_size,
            "bias_derivative_dims": 12,
            "bias_sparse_scratch": bias_sparse_scratch,
            **cache_sizes,
        },
        {"nbody": nb, "njnt": self.model.njnt, "nv": nv,
         "ntree": self.ntree, "tree_mass_nnz": self._tree_mass_nnz},
    )
    self._fk.prepare_workspace(batch_size)
    self._position_context_epoch = getattr(self, "_position_context_epoch", 0) + 1
    torch, device = self._torch, self._fk._device

    def buffer(size):
      return torch.empty(max(size, 1), dtype=torch.float32, device=device)

    self._workspace = {
        "batch_size": batch_size,
        "mass_storage": mass_storage,
        "mass": (buffer(batch_size * nv * nv)
                 if mass_storage == "dense" else None),
        "mass_blocks": (buffer(batch_size * self._tree_mass_nnz)
                        if mass_storage == "block_sparse" else None),
        "mass_sparse_product": buffer(batch_size * nv),
        "pose_status": torch.zeros(batch_size, dtype=torch.int32,
                                    device=device),
        "mass_zero_blocks": buffer(batch_size * self._tree_mass_nnz),
        "mass_armature": buffer(batch_size * (nv * nv if mass_storage == "dense" else self._tree_mass_nnz)),
        "root_com": buffer(batch_size * nb * 3),
        "cdof": buffer(batch_size * nv * 6),
        "crb": buffer(batch_size * nb * 36),
        "local_inertia": buffer(batch_size * nb * 36),
        "qvel": buffer(batch_size * nv),
        "bias": buffer(batch_size * nv),
        "bias_low": buffer(batch_size * nv),
        "cvel_low": buffer(batch_size * nb * 6),
        "cdof_dot_low": buffer(batch_size * nv * 6),
        "cacc_low": buffer(batch_size * nb * 6),
        "body_force_low": buffer(batch_size * nb * 6),
        "cvel": buffer(batch_size * nb * 6),
        "cdof_dot": buffer(batch_size * nv * 6),
        "cacc": buffer(batch_size * nb * 6),
        "body_force": buffer(batch_size * nb * 6),
        "mass_dims": torch.tensor(
            [nb, self.model.njnt, nv, batch_size],
            dtype=torch.int32,
            device=device,
        ),
        "pose_finite_dims": torch.tensor(
            [nb, self.model.ngeom, self.model.nsite, self.model.njnt,
             batch_size], dtype=torch.int32, device=device),
        "bias_derivative_dims_dense": torch.tensor(
            [nb, self.model.njnt, nv, batch_size, nv * nv, 0],
            dtype=torch.int32, device=device),
        "bias_derivative_dims_sparse": torch.tensor(
            [nb, self.model.njnt, nv, batch_size,
             self._bias_edge_count, 1], dtype=torch.int32, device=device),
        "bias_derivative_scratch": (
            buffer(bias_sparse_scratch)
            if self._bias_derivative_layout is not None else None),
        "mass_layout_dims": torch.tensor(
            [self._tree_mass_nnz if mass_storage == "block_sparse" else nv * nv,
             int(mass_storage == "block_sparse"), self.ncomponent, nv,
             self.model.ntendon, *([1] * batch_size)],
            dtype=torch.int32, device=device),
        "mass_matvec_dims": torch.tensor(
            [nv, self.ncomponent, batch_size, self._tree_mass_nnz]
            + [1] * batch_size,
            dtype=torch.int32, device=device),
        "energy_velocity_dims": torch.tensor(
            [nv, batch_size, 0] + [1] * batch_size,
            dtype=torch.int32, device=device),
        "tendon_armature_dims": torch.tensor(
            [nv, self.model.ntendon, batch_size, self.ntree,
             self._tree_mass_nnz if mass_storage == "block_sparse" else nv * nv,
             int(mass_storage == "block_sparse"), self.ncomponent],
            dtype=torch.int32, device=device),
        "bias_dims": torch.tensor(
            [nb, self.model.njnt, nv, batch_size],
            dtype=torch.int32,
            device=device,
        ),
        "disableflags": torch.tensor(
            [self.model.disableflags], dtype=torch.int32, device=device
        ),
    }
    # Public no-sleep calls use the same packed-kernel ABI with the complete
    # model lists. They are immutable device inputs and are reused across runs.
    all_body = np.arange(nb, dtype=np.int32)
    all_parent = np.arange(1, nb, dtype=np.int32)
    all_dof = np.arange(nv, dtype=np.int32)
    body_ids = np.full(max(nb, 1), -1, dtype=np.int32)
    parent_ids = np.full(max(nb, 1), -1, dtype=np.int32)
    dof_ids = np.full(max(nv, 1), -1, dtype=np.int32)
    body_ids[:all_body.size] = all_body
    parent_ids[:all_parent.size] = all_parent
    dof_ids[:all_dof.size] = all_dof
    self._workspace["all_awake_lists"] = {
        "body_ids": torch.as_tensor(np.tile(body_ids, (batch_size, 1)),
                                     dtype=torch.int32, device=device),
        "parent_ids": torch.as_tensor(np.tile(parent_ids, (batch_size, 1)),
                                       dtype=torch.int32, device=device),
        "dof_ids": torch.as_tensor(np.tile(dof_ids, (batch_size, 1)),
                                   dtype=torch.int32, device=device),
        "body_mask": torch.ones(
            (batch_size, max(nb, 1)), dtype=torch.int32, device=device),
        "body_count": torch.full((batch_size,), nb, dtype=torch.int32, device=device),
        "parent_count": torch.full((batch_size,), nb - 1, dtype=torch.int32, device=device),
        "dof_count": torch.full((batch_size,), nv, dtype=torch.int32, device=device),
        "counts": torch.tensor(
            np.tile(np.array([[nb, nb - 1, nv]], dtype=np.int32),
                    (batch_size, 1)), dtype=torch.int32, device=device),
        "tree_awake": None,
        "list_dims": torch.tensor(
            (nb, nv, self.ntree, batch_size), dtype=torch.int32,
            device=device),
    }
    self._workspace["all_tree_awake"] = torch.ones(
        (batch_size, max(self.ntree, 1)), dtype=torch.int32,
        device=device)
    if self._workspace["mass"] is not None:
      self._workspace["mass"].zero_()
    if self._workspace["mass_blocks"] is not None:
      self._workspace["mass_blocks"].zero_()
    self._workspace["mass_armature"].zero_()
    self._workspace["root_com"].zero_()
    self._workspace["mass_zero_blocks"].zero_()
    self._workspace["cdof"].zero_()
    self._workspace["crb"].zero_()
    self._workspace["local_inertia"].zero_()
    self._workspace["cvel"].zero_()
    self._workspace["cdof_dot"].zero_()
    self._workspace["bias"].zero_()
    self._workspace["bias_low"].zero_()
    self._workspace["cvel_low"].zero_()
    self._workspace["cdof_dot_low"].zero_()
    self._workspace["cacc_low"].zero_()
    self._workspace["body_force_low"].zero_()
    # One retained POS stage is sufficient for forwardSkip(POS). A later
    # capture supersedes its token; ordinary full forward dispatches do not.
    self._position_capture = 0
    self._position_shapes = position_context_shapes(
        self.model, batch_size, mass_storage=mass_storage,
        component_nnz=self._tree_mass_nnz)
    self._position_buffers = {}
    self._position_views = {}
    for name, shape in self._position_shapes.items():
      dtype = torch.int32 if name == "pose_status" else torch.float32
      backing = torch.empty(cache_sizes[f"position_cache.{name}"],
                            dtype=dtype, device=device)
      self._position_buffers[name] = backing
      self._position_views[name] = backing[:int(np.prod(shape))].reshape(shape)

  def _all_awake_lists(self, batch):
    """Build the legacy convenience API's complete, fixed-width worklists."""
    torch = self._torch
    nb, nv = self.model.nbody, self.model.nv
    body = np.full(max(nb, 1), -1, dtype=np.int32)
    parent = np.full(max(nb, 1), -1, dtype=np.int32)
    dof = np.full(max(nv, 1), -1, dtype=np.int32)
    body[:nb] = np.arange(nb, dtype=np.int32)
    if nb > 1:
      parent[:nb - 1] = np.arange(1, nb, dtype=np.int32)
    dof[:nv] = np.arange(nv, dtype=np.int32)
    return {
        "body_ids": torch.as_tensor(np.tile(body, (batch, 1)),
                                    dtype=torch.int32, device=self._fk._device),
        "parent_ids": torch.as_tensor(np.tile(parent, (batch, 1)),
                                       dtype=torch.int32, device=self._fk._device),
        "dof_ids": torch.as_tensor(np.tile(dof, (batch, 1)),
                                    dtype=torch.int32, device=self._fk._device),
        "counts": torch.tensor(np.tile([nb, nb - 1, nv], (batch, 1)),
                                dtype=torch.int32, device=self._fk._device),
        "body_mask": torch.ones((batch, max(nb, 1)), dtype=torch.int32,
                                 device=self._fk._device),
        "list_dims": torch.tensor((nb, nv, self.ntree, batch),
                                  dtype=torch.int32, device=self._fk._device),
    }

  @property
  def mass_block_layout(self):
    """Return the immutable host description of the compiled mass blocks.

    The returned arrays are defensive copies so a caller cannot corrupt the
    constructor's packed device ABI.  ``nnz`` is the flat backing length of
    the component-dense blocks; it is not the number of nonzero scalar
    entries in MuJoCo's compressed mass matrix.
    """
    names = (
        "tree_dofadr", "tree_dofnum", "tree_mass_offsets", "dof_treeid",
        "component_dof_offsets", "component_dof_ids", "component_dofnum",
        "component_mass_offsets", "dof_component", "dof_local_index",
    )
    result = {name: getattr(self, f"_{name}").copy() for name in names}
    result.update(ncomponent=self.ncomponent, nnz=self._tree_mass_nnz,
                  nv=int(self.model.nv), ntree=self.ntree)
    return result

  def project_tendon_armature_blocks(self, tendon_J, *, tree_awake=None,
                                     world_mask=None):
    """Project runtime tendon Jacobians through pinned mass sparsity.

    This is the sparse counterpart of the dense tendon-armature contribution.
    It writes the persistent component-block workspace and returns a borrowed
    ``[batch, nnz]`` MPS view. Entries touching an awake tree are refreshed;
    fully asleep entries retain their cached values. The first call starts
    from the zero-initialized workspace established by ``prepare_workspace``.

    The runtime Jacobian is the complete canonical tendon array, with fixed
    tendon rows and any spatial tendon rows in their model tendon indices.
    The projection follows ``mass_rowadr/mass_rownnz/mass_colind`` exactly,
    matching the sparse addition used by MuJoCo's compiled mass matrix.
    """
    torch = self._torch
    batch = int(self._workspace["batch_size"])
    shape = (batch, self.model.ntendon, self.model.nv)
    if (not isinstance(tendon_J, torch.Tensor)
        or tuple(tendon_J.shape) != shape
        or tendon_J.dtype != torch.float32
        or tendon_J.device.type != "mps"
        or not tendon_J.is_contiguous()):
      raise ValueError(f"tendon_J must be contiguous float32 MPS {shape}")
    if not self._has_tendon_armature:
      return None
    if self._mass_storage != "block_sparse":
      raise ValueError("tendon armature blocks require block_sparse mass storage")
    if tree_awake is None:
      tree_awake = self._workspace["all_tree_awake"]
    if world_mask is None:
      world_mask = self._fk._workspace["outputs"]["all_world_mask"]
    if (not isinstance(tree_awake, torch.Tensor)
        or tuple(tree_awake.shape) != (batch, max(self.ntree, 1))
        or tree_awake.dtype != torch.int32
        or tree_awake.device.type != "mps"
        or not tree_awake.is_contiguous()):
      raise ValueError("tree_awake must be contiguous MPS int32 [batch,max(ntree,1)]")
    w = self._workspace
    self._tendon_armature_kernel(
        tendon_J.reshape(-1) if self.model.ntendon and self.model.nv
        else w["mass_sparse_product"],
        self._effective_tendon_armature_device,
        self._tendon_treeid_device,
        self._tendon_treenum_device,
        self._tendon_j_rowadr_device,
        self._tendon_j_rownnz_device,
        self._tendon_j_colind_device,
        tree_awake,
        self._tree_layout_device,
        self._mass_rowadr_device,
        self._mass_rownnz_device,
        self._mass_colind_device,
        self._dof_treeid_device,
        w["mass_armature"],
        w["tendon_armature_dims"],
        world_mask,
        threads=(batch,), group_size=(1,))
    return w["mass_armature"][:batch * self._tree_mass_nnz].reshape(
        batch, self._tree_mass_nnz)

  def _run_velocity_bias(self, qvel, lists, *, cdof, local_inertia,
                         world_mask=None):
    """Refresh only comVel/RNE; position-dependent CRBA inputs stay cached."""
    w, arrays = self._workspace, self._arrays
    batch = int(w["batch_size"])
    qvel_flat = qvel.reshape(-1) if self.model.nv else w["qvel"]
    if world_mask is None:
      world_mask = self._fk._workspace["outputs"]["all_world_mask"]
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
        cdof.reshape(-1) if self.model.nv else w["cdof"],
        local_inertia.reshape(-1),
        w["disableflags"],
        qvel_flat,
        w["cvel"],
        w["cdof_dot"],
        w["cacc"],
        w["body_force"],
        w["bias"],
        w["bias_low"],
        w["cvel_low"],
        w["cdof_dot_low"],
        w["cacc_low"],
        w["body_force_low"],
        w["bias_dims"],
        lists["body_ids"], lists["parent_ids"], lists["dof_ids"],
        lists["body_mask"], lists["counts"], lists["list_dims"],
        world_mask,
    ]
    if len(bias_args) != 31:
      raise RuntimeError("native smooth bias shader buffer ABI mismatch")
    self._bias_kernel(*bias_args, threads=(batch,), group_size=(1,))
    nv, nb = self.model.nv, self.model.nbody
    return {
        "qfrc_bias": w["bias"][:batch * nv].reshape(batch, nv),
        "qfrc_bias_low": w["bias_low"][:batch * nv].reshape(batch, nv),
        "cvel": w["cvel"][:batch * nb * 6].reshape(batch, nb, 6),
        "cdof_dot": w["cdof_dot"][:batch * nv * 6].reshape(batch, nv, 6),
    }

  def position_context(self, dynamics):
    """Own a frozen position stage for an internal forwardSkip(POS) call.

    The context is tied to this prepared workspace. It includes the actual
    mass result after the caller's tendon additions, and is independent of
    subsequent borrowed-output dispatches. One persistent cache is reused;
    a new capture invalidates an older context. Runtime values stay on MPS
    and no device storage is allocated by this method.
    """
    names = ("poses", "root_com", "cdof", "pose_status", "mass_matrix",
             "mass_blocks", "mass_block_layout", "tendon_armature_matrix",
             "tendon_armature_blocks")
    source = {}
    for name in names:
      if name in dynamics and name != "mass_block_layout":
        if name == "poses":
          source.update({f"poses.{key}": value
                         for key, value in dynamics[name].items()})
        else:
          source[name] = dynamics[name]
    batch, nb = int(self._workspace["batch_size"]), int(self.model.nbody)
    source["local_inertia"] = self._workspace["local_inertia"][
        :batch * nb * 36].reshape(batch, nb, 36)
    pose_names = {name.split(".", 1)[1] for name in self._position_shapes
                  if name.startswith("poses.")}
    if ({name.split(".", 1)[1] for name in source
         if name.startswith("poses.")} != pose_names):
      raise ValueError("position stage has an incomplete cached pose ABI")
    required = set(self._position_shapes) - {
        "tendon_armature_matrix", "tendon_armature_blocks"}
    if not required <= source.keys():
      raise ValueError("position stage is missing a required smooth field")
    for name, value in source.items():
      if name not in self._position_views:
        raise ValueError(f"position stage has an unsupported field {name}")
      target = self._position_views[name]
      if (not isinstance(value, self._torch.Tensor)
          or tuple(value.shape) != self._position_shapes[name]
          or value.dtype != target.dtype or value.device != target.device
          or not value.is_contiguous()):
        raise ValueError(f"position stage {name} has invalid tensor metadata")
    for name, value in source.items():
      self._position_views[name].copy_(value)
    context = {"poses": {name.split(".", 1)[1]: self._position_views[name]
                         for name in source if name.startswith("poses.")}}
    context.update({name: self._position_views[name] for name in source
                    if not name.startswith("poses.") and name != "local_inertia"})
    if "mass_block_layout" in dynamics:
      context["mass_block_layout"] = dynamics["mass_block_layout"]
    context["_local_inertia"] = self._position_views["local_inertia"]
    self._position_capture += 1
    context["_capture"] = self._position_capture
    context["_owner"] = self
    context["_epoch"] = self._position_context_epoch
    return context

  def run_velocity_device(self, context, qvel, *, awake_lists=None,
                          world_mask=None):
    """Consume a frozen POS stage without FK, CRBA, or armature projection.

    Returns position tensors owned by ``context`` plus borrowed velocity-stage
    outputs. This is an internal stage API: the caller must retain the same
    pre-sleep awake indices until velocity/actuation/constraint refresh ends.
    """
    if (not isinstance(context, dict) or context.get("_owner") is not self
        or context.get("_epoch") != self._position_context_epoch
        or context.get("_capture") != getattr(self, "_position_capture", None)):
      raise ValueError("position context belongs to another or replaced workspace")
    batch = int(self._workspace["batch_size"])
    self._validate_cached_pose_fields(context, batch)
    self._validate_world_mask(world_mask, batch)
    self._fk._check_device_tensor(
        qvel, "qvel", (batch, self.model.nv), self._torch, self._fk._device)
    lists = self._workspace["all_awake_lists"] if awake_lists is None else awake_lists
    self._validate_awake_lists(lists, batch)
    result = {name: value for name, value in context.items()
              if not name.startswith("_")}
    result.update(self._run_velocity_bias(
        qvel, lists, cdof=context["cdof"],
        local_inertia=context["_local_inertia"], world_mask=world_mask))
    return result

  def _validate_cached_pose_fields(self, context, batch):
    """Reject replaced/corrupted borrowed FK pose views before VEL writes."""
    poses = context.get("poses") if isinstance(context, dict) else None
    pose_names = {name.split(".", 1)[1]
                  for name in self._position_shapes if name.startswith("poses.")}
    if not isinstance(poses, dict) or set(poses) != pose_names:
      raise ValueError("position context has an incomplete cached pose ABI")
    for name in sorted(pose_names):
      shape = self._position_shapes[f"poses.{name}"]
      value = poses[name]
      if (not isinstance(value, self._torch.Tensor)
          or tuple(value.shape) != shape
          or value.dtype != self._torch.float32
          or value.device.type != self._fk._device.type
          or (self._fk._device.index is not None
              and value.device.index != self._fk._device.index)
          or not value.is_contiguous()):
        raise ValueError(f"position context poses.{name} has invalid tensor metadata")

  def _validate_world_mask(self, world_mask, batch):
    """Validate an optional world mask before any smooth-stage dispatch."""
    if world_mask is None:
      return
    if (not isinstance(world_mask, self._torch.Tensor)
        or tuple(world_mask.shape) != (batch,)
        or world_mask.dtype != self._torch.int32
        or world_mask.device.type != self._fk._device.type
        or (self._fk._device.index is not None
            and world_mask.device.index != self._fk._device.index)
        or not world_mask.is_contiguous()):
      raise ValueError("world_mask must be contiguous MPS int32 [batch]")

  def _validate_awake_lists(self, lists, batch):
    torch = self._torch
    list_shapes = {
        "body_ids": (batch, max(self.model.nbody, 1)),
        "parent_ids": (batch, max(self.model.nbody, 1)),
        "dof_ids": (batch, max(self.model.nv, 1)),
        "counts": (batch, 3),
        "body_mask": (batch, max(self.model.nbody, 1)),
        "list_dims": (4,),
    }
    if not isinstance(lists, dict) or not list_shapes.keys() <= lists.keys():
      raise ValueError("awake_lists must contain fixed IDs and per-world counts")
    for name, shape in list_shapes.items():
      value = lists[name]
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != torch.int32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"awake_lists.{name} must be contiguous MPS int32 {shape}")
    # These aliases are column views into the contiguous [B,3] count buffer;
    # they are descriptive API fields and are not passed to native kernels.
    for name in ("body_count", "parent_count", "dof_count"):
      value = lists.get(name)
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch,)
          or value.dtype != torch.int32 or value.device.type != "mps"):
        raise ValueError(f"awake_lists.{name} must be an MPS int32 [batch] view")
    tree_awake = lists.get("tree_awake")
    if tree_awake is not None and (
        tuple(tree_awake.shape) != (batch, max(self.ntree, 1))
        or tree_awake.dtype != torch.int32 or tree_awake.device.type != "mps"
        or not tree_awake.is_contiguous()):
      raise ValueError("awake_lists.tree_awake must be contiguous MPS int32")

  def run_device(self, qpos, qvel, mocap_pos=None, mocap_quat=None, poses=None,
                 awake_lists=None, mass_storage=None, tendon_J=None, *,
                 _trusted_internal_poses=False, _position_only=False,
                 world_mask=None):
    """Compute M(q) and bias from borrowed MPS float32 state tensors.

    Inputs are trusted to be finite and have finite nonzero free/ball
    quaternions. Device-state reset owns value validation; this hot path checks
    tensor metadata only and never reads a device value back to the host.
    Results borrow persistent workspace and remain valid until its next use.

    Mocap poses follow the `MetalKinematics.run_device` contract (required
    when the model has mocap bodies, omitted otherwise).

    In dense mode, ``tendon_J`` is optional: omitting it returns rigid-body
    CRBA, leaving dense fixed/spatial tendon armature assembly to the caller's
    tendon stage. Block-sparse mode requires the canonical Jacobian whenever
    compiled tendon armature is nonzero because that projection is part of its
    packed mass operator.

    Public supplied pose dictionaries are finite-checked at the boundary.
    The private ``_trusted_internal_poses`` switch is only for dictionaries
    produced by this simulation's FK stage; it retains metadata validation and
    reports nonfinite values through the device-only ``pose_status`` result.
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
    self._validate_world_mask(world_mask, batch)
    internal_pose_values = poses is None or _trusted_internal_poses
    if poses is None:
      poses = self._fk.run_device(
          qpos, mocap_pos, mocap_quat,
          tree_awake=(awake_lists.get("tree_awake")
                      if awake_lists is not None else None),
          world_mask=world_mask)
    else:
      if mocap_pos is not None or mocap_quat is not None:
        raise ValueError("mocap inputs cannot accompany supplied poses")
      validate_pose_dict(
          self.model, poses, batch, self._fk._device, torch,
          check_finite=not _trusted_internal_poses)
    w, arrays = self._workspace, self._arrays
    if world_mask is None:
      world_mask = self._fk._workspace["outputs"]["all_world_mask"]
    if internal_pose_values:
      pose_buffers = []
      for name in ("body_pos", "body_quat", "geom_pos", "geom_quat",
                   "site_pos", "site_quat", "inertial_pos", "inertial_quat",
                   "joint_anchor", "joint_axis", "geom_xmat", "geom_xmat_low",
                   "geom_xmat_tail", "geom_quat_low", "geom_quat_tail"):
        value = poses[name]
        pose_buffers.append(value.reshape(-1) if value.numel()
                            else w["mass_sparse_product"])
      self._pose_finite_kernel(
          *pose_buffers, w["pose_finite_dims"], w["pose_status"], world_mask,
          threads=(batch,), group_size=(1,))
      # The primary pose check is already at the practical MSL binding limit
      # once it also validates geom_xmat words. Check the remaining retained
      # position/quaternion residuals in a second pass, using the same status
      # and mask buffers and no additional workspace.
      residual_buffers = []
      for kind in ("body", "geom", "site", "inertial"):
        for field in (f"{kind}_pos_low", f"{kind}_pos_tail",
                      f"{kind}_quat_low", f"{kind}_quat_tail"):
          value = poses[field]
          residual_buffers.append(
              value.reshape(-1) if value.numel()
              else w["mass_sparse_product"])
      self._pose_finite_residual_kernel(
          *residual_buffers, w["pose_finite_dims"], w["pose_status"],
          world_mask, threads=(batch,), group_size=(1,))
    else:
      # External poses are synchronously checked at their boundary above.
      # Report that validation as success without a second value readback.
      w["pose_status"].zero_()
    selected_mass_storage = w["mass_storage"] if "mass_storage" in w else self._mass_storage
    if mass_storage is not None and mass_storage != selected_mass_storage:
      raise ValueError("mass storage is fixed when the smooth workspace is prepared")
    mass_output = (w["mass_blocks"] if selected_mass_storage == "block_sparse"
                   else w["mass"])
    lists = w["all_awake_lists"] if awake_lists is None else awake_lists
    self._validate_awake_lists(lists, batch)
    qvel_flat = qvel.reshape(-1) if self.model.nv else w["qvel"]
    args = [
        arrays[name]
        for name in (
            "body_parentid",
            "body_rootid",
            "body_treeid",
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
            mass_output,
            w["root_com"],
            w["cdof"],
            w["crb"],
            w["mass_dims"],
            w["local_inertia"],
        ]
    )
    args.extend([lists["body_ids"], lists["parent_ids"], lists["dof_ids"],
                 lists["counts"],
                 lists["body_mask"], lists["list_dims"]])
    args.extend([self._tree_layout_device, w["mass_layout_dims"]])
    w["mass_layout_dims"][5:5 + batch].copy_(world_mask)
    if len(args) != 31:
      raise RuntimeError("native dense mass shader buffer ABI mismatch")
    self._kernel(*args, threads=(batch,), group_size=(1,))
    if _position_only:
      # POS owns FK/COM, cdof/local inertia and CRBA only.  In particular it
      # must not run comVel/RNE with a dummy velocity: mj_fwdVelocity is a
      # distinct later stage and consumes the saved position products.
      w["bias"].zero_()
      w["bias_low"].zero_()
      w["cvel_low"].zero_()
      w["cdof_dot_low"].zero_()
      w["cacc_low"].zero_()
      w["body_force_low"].zero_()
      w["cvel"].zero_()
      w["cdof_dot"].zero_()
    else:
      self._run_velocity_bias(
          qvel, lists, cdof=w["cdof"], local_inertia=w["local_inertia"],
          world_mask=world_mask)
    tendon_armature_output = None
    if self._has_tendon_armature and tendon_J is None:
      # Dense legacy callers assemble fixed/spatial tendon armature alongside
      # their tendon force stage, and native mass queries project it in their
      # own exact row path. Only the sparse storage contract requires this
      # projection here because its mass operator has no dense fallback.
      if selected_mass_storage == "block_sparse":
        raise ValueError(
            "tendon_J [batch, ntendon, nv] is required for sparse tendon armature")
    elif self._has_tendon_armature:
      if (not isinstance(tendon_J, torch.Tensor)
          or tuple(tendon_J.shape) != (batch, self.model.ntendon, self.model.nv)
          or tendon_J.dtype != torch.float32 or tendon_J.device.type != "mps"
          or not tendon_J.is_contiguous()):
        raise ValueError(
            f"tendon_J must be contiguous float32 MPS {(batch, self.model.ntendon, self.model.nv)}")
      tree_awake = lists.get("tree_awake")
      if tree_awake is None:
        tree_awake = w["all_tree_awake"]
      self._tendon_armature_kernel(
          tendon_J.reshape(-1) if self.model.ntendon and self.model.nv else w["mass_sparse_product"],
          self._effective_tendon_armature_device,
          self._tendon_treeid_device,
          self._tendon_treenum_device,
          self._tendon_j_rowadr_device,
          self._tendon_j_rownnz_device,
          self._tendon_j_colind_device,
          tree_awake,
          self._tree_layout_device,
          self._mass_rowadr_device,
          self._mass_rownnz_device,
          self._mass_colind_device,
          self._dof_treeid_device,
          w["mass_armature"],
          w["tendon_armature_dims"],
          world_mask,
          threads=(batch,), group_size=(1,))
      tendon_armature_output = w["mass_armature"]
    nv = self.model.nv
    result = {
        "qfrc_bias": w["bias"][: batch * nv].reshape(batch, nv),
        "qfrc_bias_low": w["bias_low"][: batch * nv].reshape(batch, nv),
        "pose_status": w["pose_status"],
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
    if selected_mass_storage == "dense":
      result["mass_matrix"] = w["mass"][: batch * nv * nv].reshape(batch, nv, nv)
      if tendon_armature_output is not None:
        result["tendon_armature_matrix"] = tendon_armature_output[
            :batch * nv * nv].reshape(batch, nv, nv)
    else:
      result["mass_blocks"] = w["mass_blocks"][
          :batch * self._tree_mass_nnz].reshape(batch, self._tree_mass_nnz)
      result["mass_block_layout"] = {
          "tree_dofadr": self._tree_dofadr.copy(),
          "tree_dofnum": self._tree_dofnum.copy(),
          "tree_mass_offsets": self._tree_mass_offsets.copy(),
          "dof_treeid": self._dof_treeid.copy(),
          "component_dof_offsets": self._component_dof_offsets.copy(),
          "component_dof_ids": self._component_dof_ids.copy(),
          "component_dofnum": self._component_dofnum.copy(),
          "component_mass_offsets": self._component_mass_offsets.copy(),
          "dof_component": self._dof_component.copy(),
          "dof_local_index": self._dof_local_index.copy(),
          "ncomponent": self.ncomponent,
          "nnz": self._tree_mass_nnz,
      }
      if tendon_armature_output is not None:
        result["tendon_armature_blocks"] = tendon_armature_output[
            :batch * self._tree_mass_nnz].reshape(batch, self._tree_mass_nnz)
    return result

  def run_position_device(self, qpos, mocap_pos=None, mocap_quat=None, *,
                          poses=None, awake_lists=None, tendon_J=None):
    """Run the native position stage without evaluating velocity dynamics.

    This is the POS half of the internal forward-skip pipeline.  The method
    reuses the already prepared qvel backing as an all-zero placeholder only
    for the existing input ABI; ``_position_only`` prevents that placeholder
    from reaching the bias/RNE kernel.  The returned mass, poses, COM and
    motion-subspace values are the ordinary ``run_device`` borrowed results.
    """
    torch = self._torch
    if not isinstance(qpos, torch.Tensor) or qpos.ndim != 2:
      raise TypeError("qpos must be a rank-2 torch.Tensor")
    batch = int(qpos.shape[0])
    if self._workspace.get("batch_size") != batch:
      raise ValueError("call prepare_workspace(batch_size) before using this batch size")
    zero_qvel = self._workspace["qvel"][:batch * self.model.nv].view(
        batch, self.model.nv)
    zero_qvel.zero_()
    return self.run_device(
        qpos, zero_qvel, mocap_pos, mocap_quat, poses=poses,
        awake_lists=awake_lists, tendon_J=tendon_J,
        _trusted_internal_poses=poses is not None, _position_only=True)

  def mass_blocks_matvec_device(self, mass_blocks, vector,
                                tendon_armature_blocks=None, *,
                                world_mask=None):
    """Multiply compiled dense component blocks by a vector.

    Component blocks preserve kinematic-tree inertia and merge trees coupled
    by tendon armature. The returned product borrows reusable smooth workspace.
    """
    torch = self._torch
    batch = self._workspace["batch_size"]
    if self._workspace["mass_storage"] != "block_sparse":
      raise RuntimeError("prepare a block_sparse smooth workspace first")
    for name, value, shape in (
        ("mass_blocks", mass_blocks, (batch, self._tree_mass_nnz)),
        ("vector", vector, (batch, self.model.nv)),
    ):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    if tendon_armature_blocks is None:
      tendon_armature_blocks = self._workspace["mass_zero_blocks"][:
          batch * self._tree_mass_nnz].reshape(batch, self._tree_mass_nnz)
    elif (not isinstance(tendon_armature_blocks, torch.Tensor)
          or tuple(tendon_armature_blocks.shape) != (batch, self._tree_mass_nnz)
          or tendon_armature_blocks.dtype != torch.float32
          or tendon_armature_blocks.device.type != "mps"
          or not tendon_armature_blocks.is_contiguous()):
      raise ValueError("tendon_armature_blocks must be contiguous float32 MPS [batch, nnz]")
    dims = self._workspace["mass_matvec_dims"]
    if world_mask is None:
      dims[4:].fill_(1)
    else:
      if (not isinstance(world_mask, torch.Tensor)
          or world_mask.dtype != torch.int32 or world_mask.device.type != "mps"
          or tuple(world_mask.shape) != (batch,) or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous int32 MPS [batch]")
      dims[4:].copy_(world_mask)
    self._library.tree_block_mass_matvec(
        mass_blocks.reshape(-1), vector.reshape(-1),
        tendon_armature_blocks.reshape(-1),
        self._tree_layout_device,
        self._workspace["mass_sparse_product"],
        dims,
        threads=(batch,), group_size=(1,))
    return self._workspace["mass_sparse_product"][:batch * self.model.nv].reshape(
        batch, self.model.nv)

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
    batch = source.shape[0]
    nv, nb = self.model.nv, self.model.nbody
    _validate_workspace_index_capacity(
        batch, {"dense_mass_query": batch * nv * nv},
        {"nv": nv, "dense_mass_stride": nv * nv, "batch": batch})
    poses = self._fk.run(source, mocap_pos, mocap_quat)
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
        arrays["body_treeid"],
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
    lists = self._all_awake_lists(batch)
    args.extend([lists["body_ids"], lists["parent_ids"], lists["dof_ids"],
                 lists["counts"], lists["body_mask"], lists["list_dims"]])
    args.extend([self._tree_layout_device,
                 torch.tensor([nv * nv, 0, self.ntree, nv],
                              dtype=torch.int32, device=self._fk._device)])
    if len(args) != 31:
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
    bias_low = torch.empty(
        max(batch * nv, 1), dtype=torch.float32, device=self._fk._device
    )
    cvel_low = torch.empty(
        max(batch * nb * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    cdof_dot_low = torch.empty(
        max(batch * nv * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    cacc_low = torch.empty(
        max(batch * nb * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    body_force_low = torch.empty(
        max(batch * nb * 6, 1), dtype=torch.float32, device=self._fk._device
    )
    for scratch in (bias_low, cvel_low, cdof_dot_low, cacc_low,
                    body_force_low):
      scratch.zero_()
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
        bias_low,
        cvel_low,
        cdof_dot_low,
        cacc_low,
        body_force_low,
        torch.tensor(
            [nb, self.model.njnt, nv, batch],
            dtype=torch.int32,
            device=self._fk._device,
        ),
    ]
    lists = self._all_awake_lists(batch)
    args.extend([lists["body_ids"], lists["parent_ids"], lists["dof_ids"],
                 lists["body_mask"], lists["counts"], lists["list_dims"]])
    args.append(self._fk._workspace["outputs"]["all_world_mask"])
    if len(args) != 31:
      raise RuntimeError("native smooth bias shader buffer ABI mismatch")
    self._bias_kernel(*args, threads=(batch,), group_size=(1,))
    return {
        "mass_matrix": computed["mass_matrix"],
        "qfrc_bias": bias[: batch * nv].reshape(batch, nv),
    }

  def bias_derivative_device(self, qpos, qvel, dynamics=None, *,
                             edge_values=None):
    """Compute analytical -d(qfrc_bias)/d(qvel) in reusable device storage.

    Held-position RNE is differentiated directly; there are no perturbed
    physics evaluations or fixed-size body/DOF arrays. Returned storage is
    borrowed and remains valid until the next derivative call.
    """
    torch = self._torch
    batch = qpos.shape[0]
    nv = self.model.nv
    sparse_output = edge_values is not None
    if nv == 0:
      if sparse_output:
        return edge_values
      return torch.zeros((batch, 0, 0), dtype=torch.float32, device=self._fk._device)
    if dynamics is None:
      dynamics = self.run_device(qpos, qvel)
    w, arrays = self._workspace, self._arrays
    if batch != w["batch_size"]:
      raise ValueError("bias derivative batch does not match prepared smooth workspace")
    if sparse_output:
      if self._bias_derivative_layout is None:
        raise ValueError("compiled velocity derivative layout is required for COO bias")
      expected = (batch, max(self._bias_edge_count, 1))
      if (not isinstance(edge_values, torch.Tensor)
          or edge_values.dtype != torch.float32
          or edge_values.device.type != "mps"
          or tuple(edge_values.shape) != expected
          or not edge_values.is_contiguous()):
        raise ValueError(f"edge_values must be contiguous MPS float32 {expected}")
      qderiv = edge_values
    elif "bias_derivative" not in w:
      sizes = bias_derivative_workspace_sizes(self.model.nbody, nv, batch)
      for key, count in sizes.items():
        w[key] = torch.empty(max(count, 1), dtype=torch.float32, device=self._fk._device)
    if not sparse_output:
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
        (w["bias_derivative_dims_sparse"] if sparse_output else
         w["bias_derivative_dims_dense"]),
        dynamics.get("cvel", w["cvel"]).reshape(-1),
        dynamics.get("cdof_dot", w["cdof_dot"]).reshape(-1),
        w["bias_derivative_scratch"],
        self._bias_column_offsets,
        self._bias_edge_rows,
        self._bias_edge_slots,
    ]
    self._bias_deriv_kernel(*args, threads=(nv, batch), group_size=(1, 1))
    return qderiv
