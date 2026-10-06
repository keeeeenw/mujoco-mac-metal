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

"""Milestone 019: Native GPU split-stage, inverse dynamics, and state query APIs.

Simulation queries consume prepared MPS/Metal stages without a CPU physics
fallback. Host arguments are validated and staged explicitly; compilation and
visualization remain upstream host utilities. The mju_* tensor helpers retain
their inputs' device and can also be used with CPU tensors. These backend APIs
have their own documented signatures and are not universal replacements for the
upstream C/Python bindings.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any
import mujoco
import numpy as np

try:
  import torch
except ImportError:
  torch = None


# -------------------------------------------------------------------------
# Native MPS Math Utilities
# -------------------------------------------------------------------------

def mju_mulMatVec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Batched or unbatched matrix-vector multiplication on device.
  
  mat: [batch, n, m] or [n, m]
  vec: [batch, m] or [m]
  Returns: [batch, n] or [n]
  """
  if torch is None:
    raise RuntimeError("PyTorch with MPS is required for native math utilities")
  if mat.dim() == 2 and vec.dim() == 1:
    return torch.matmul(mat, vec)
  elif mat.dim() == 3 and vec.dim() == 2:
    return torch.bmm(mat, vec.unsqueeze(-1)).squeeze(-1)
  elif mat.dim() == 3 and vec.dim() == 1:
    b = mat.shape[0]
    return torch.bmm(mat, vec.unsqueeze(0).expand(b, -1).unsqueeze(-1)).squeeze(-1)
  raise ValueError(f"Incompatible shapes for mju_mulMatVec: mat {mat.shape}, vec {vec.shape}")


def mju_transpose(mat: torch.Tensor) -> torch.Tensor:
  """Batched or unbatched matrix transpose on device."""
  if mat.dim() == 2:
    return mat.t().contiguous()
  elif mat.dim() == 3:
    return mat.transpose(1, 2).contiguous()
  raise ValueError(f"Incompatible shape for mju_transpose: {mat.shape}")


def mju_mulMatMat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Batched or unbatched matrix-matrix multiplication on device."""
  if a.dim() == 2 and b.dim() == 2:
    return torch.matmul(a, b)
  elif a.dim() == 3 and b.dim() == 3:
    return torch.bmm(a, b)
  raise ValueError(f"Incompatible shapes for mju_mulMatMat: a {a.shape}, b {b.shape}")


def mju_cholSolve(L: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Solve L L^T x = vec on device for lower triangular Cholesky factor L."""
  if torch is None:
    raise RuntimeError("PyTorch is required")
  if L.dim() == 2:
    # 2D case
    y = torch.linalg.solve_triangular(L, vec.unsqueeze(-1), upper=False).squeeze(-1)
    x = torch.linalg.solve_triangular(L.t(), y.unsqueeze(-1), upper=True).squeeze(-1)
    return x
  elif L.dim() == 3:
    # Batched 3D case
    y = torch.linalg.solve_triangular(L, vec.unsqueeze(-1), upper=False)
    x = torch.linalg.solve_triangular(L.transpose(1, 2), y, upper=True).squeeze(-1)
    return x
  raise ValueError(f"Incompatible shape for mju_cholSolve: L {L.shape}")


# -------------------------------------------------------------------------
# Native Inverse Dynamics (runs on device)
# -------------------------------------------------------------------------

def _contains_device_tensor(value):
  if torch is not None and isinstance(value, torch.Tensor):
    return True
  if isinstance(value, dict):
    return any(_contains_device_tensor(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return any(_contains_device_tensor(item) for item in value)
  if is_dataclass(value) and not isinstance(value, type):
    return any(_contains_device_tensor(getattr(value, field.name))
               for field in fields(value))
  return False


def _capture_query_storage(value, memo):
  """Retain original storage owners and container bindings, not just values."""
  if id(value) in memo:
    return memo[id(value)]
  if isinstance(value, torch.Tensor):
    node = ("tensor", value, value.detach().clone())
  elif isinstance(value, dict):
    children = {}
    node = ("dict", value, children)
    memo[id(value)] = node
    children.update({key: _capture_query_storage(item, memo) for key, item in value.items()})
  elif isinstance(value, (list, tuple)):
    node = ("sequence", value, [_capture_query_storage(item, memo) for item in value])
  elif is_dataclass(value) and not isinstance(value, type):
    node = ("record", value, {field.name: _capture_query_storage(getattr(value, field.name), memo)
                              for field in fields(value)})
  else:
    node = ("leaf", value, None)
  memo[id(value)] = node
  return node


def _restore_query_storage(node):
  kind, original, saved = node
  if kind == "tensor":
    original.copy_(saved)
  elif kind == "dict":
    original.clear()
    original.update({name: _restore_query_storage(value) for name, value in saved.items()})
  elif kind == "sequence":
    values = [_restore_query_storage(value) for value in saved]
    if isinstance(original, list):
      original[:] = values
  elif kind == "record":
    for name, value in saved.items():
      object.__setattr__(original, name, _restore_query_storage(value))
  return original


@contextmanager
def _inverse_query_workspaces(sim):
  """Preserve stage storage and cache identity at an explicit query boundary.

  Stage outputs are borrowed by assembled-system views, tendon sensors and
  cached forward contexts. Restoring only constraint rows leaves their smooth
  mass/pose buffers describing the temporary query. Snapshot device storage
  in place (including nested stage programs) without numerical readback.
  This allocation cost belongs to the explicit inverse query, not stepping.
  """
  roots = ("_smooth", "_passive", "_fluid", "_tendons", "_spatial_tendons",
           "_flex", "_coupled_constraints", "_component_solver", "_forward_stages",
           "_solver", "_implicit", "_implicitfast", "_effective_implicit",
           "_velocity_derivative_values", "_flex_implicit", "_sparse_flex_implicit",
           "_actuators", "_transmissions",
           "_motor", "_sensors", "_state", "_sleep_schedule", "_contact",
           "_joint_constraints", "_touch_grid", "_cable", "_history_program",
           "_delay")
  seen, saved, memo = set(), [], {}
  stages = getattr(sim, "_forward_stages", None)
  record = getattr(stages, "_record", None)
  coherent_inputs = []
  paired_versions = []
  coupled = getattr(sim, "_last_coupled", None)
  for value, version_name in (
      (getattr(sim, "_last_sensor_qacc", None), "_last_sensor_qacc_version"),
      (coupled.get("qacc") if isinstance(coupled, dict) else None,
       "_last_coupled_qacc_version"),
  ):
    version = getattr(sim, version_name, None)
    if isinstance(value, torch.Tensor) and isinstance(version, int):
      paired_versions.append((value, version_name, int(value._version) == version))
  if record is not None:
    for name, value in record.input_tensors.items():
      expected = record.input_versions.get(name)
      if expected is not None and stages._version(value) == expected:
        coherent_inputs.append((name, value))
  def collect(program):
    if program is None or id(program) in seen or not hasattr(program, "__dict__"):
      return
    seen.add(id(program))
    attributes = dict(vars(program))
    storage = {name: _capture_query_storage(value, memo)
               for name, value in attributes.items() if _contains_device_tensor(value)}
    saved.append((program, attributes, storage))
    for name, value in attributes.items():
      # Programs own subordinate FK, broadphase, compaction and flex stages;
      # arbitrary third-party objects and Torch internals are not traversed.
      if name != "_owner" and type(value).__module__.startswith("mujoco_metal."):
        collect(value)
  for name in roots:
    collect(getattr(sim, name, None))
  bookkeeping_names = ("_spatial_kin", "_spatial_cache_key", "_forward_position_epoch",
      "_accepted_step", "_last_actuation_kin", "_last_coupled",
      "_last_coupled_generation", "_last_coupled_qacc_version",
      "_last_sensor_qacc", "_last_sensor_qacc_low",
      "_last_sensor_qacc_generation", "_last_sensor_qacc_version",
      "_assembled_system_valid", "_step1_record",
      "_last_native_actuator_outputs")
  bookkeeping = {name: _capture_query_storage(getattr(sim, name), memo)
                 for name in bookkeeping_names if hasattr(sim, name)}
  scratch_names = ("_component_solve_rhs", "_component_world_status",
                   "_component_solution_vector",
                   "_component_solution_low_vector",
                   "_forward_position_buffers",
                   "_component_tendon_J", "_component_damping_deriv", "_rhs", "_rhs_low",
                   "_act_dot", "_actuator_velocity_derivative", "_sensordata",
                   "_act_vel", "_sen_act_force", "_sen_qfrc_act",
                   "_raw_sensordata", "_energy", "_sensor_plugin_status",
                   "_legacy_canonical_rows", "_legacy_constraint_rhs", "_control",
                   "_native_actuator_force", "_native_actuator_qfrc")
  scratch_names += tuple(name for name in vars(sim)
                         if name.startswith(("_forward_stage_", "_sleep_")) and
                         isinstance(getattr(sim, name), torch.Tensor))
  component = {name: _capture_query_storage(getattr(sim, name), memo)
               for name in scratch_names
               if hasattr(sim, name)}
  plugins = tuple(getattr(sim, "_native_plugins", ()))
  if plugins:
    for plugin in plugins:
      if (not callable(getattr(plugin, "device_snapshot", None)) or
          not callable(getattr(plugin, "restore_device", None))):
        raise TypeError("native queries require device rollback methods on plugins")
    plugin_state = [(plugin, plugin.device_snapshot()) for plugin in plugins]
  else:
    plugin_state = None
  try:
    try:
      yield
    finally:
      _restore_query_workspaces(saved, bookkeeping_names, bookkeeping, component,
                                sim, stages, record, coherent_inputs)
      # Restoring a tensor increments its mutation counter. Preserve whether
      # paired acceleration was coherent on entry, without reviving stale data.
      for value, version_name, coherent in paired_versions:
        if coherent:
          setattr(sim, version_name, int(value._version))
  finally:
    if plugin_state is not None:
      failures = []
      for plugin, payload in reversed(plugin_state):
        try:
          plugin.restore_device(payload)
        except Exception as error:
          failures.append(error)
      if failures:
        raise RuntimeError("native query plugin device rollback failed") from failures[0]


def _restore_query_workspaces(saved, bookkeeping_names, bookkeeping, component,
                              sim, stages, record, coherent_inputs):
  """Restore all simulation-owned storage before invoking plugin rollback."""
  for program, attributes, storage in reversed(saved):
    # Restore the schema and original storage owners before copying values;
    # a failed query may have replaced a buffer or published an extra map.
    vars(program).clear()
    vars(program).update(attributes)
    for name, value in storage.items():
      # Frozen dataclass maps are owned records, not mutable programs, but
      # their tensor storage still participates in rollback. Restore their
      # original field bindings through the same internal mechanism used by
      # `_restore_query_storage`, without invoking their public frozen setter.
      object.__setattr__(program, name, _restore_query_storage(value))
  for name in bookkeeping_names:
    if name in bookkeeping:
      setattr(sim, name, _restore_query_storage(bookkeeping[name]))
    elif hasattr(sim, name):
      delattr(sim, name)
  for name, value in component.items():
    setattr(sim, name, _restore_query_storage(value))
  # Restoring tensor values increments Torch mutation counters. Rebind only
  # inputs that were coherent on entry, after every borrowed buffer has been
  # restored. Never turn an already-stale record into a valid one.
  if record is not None and getattr(stages, "_record", None) is record:
    for name, value in coherent_inputs:
      if record.input_tensors.get(name) is value:
        stages.capture_input(record, name, value)



def mj_inverse(sim, qpos=None, qvel=None, qacc=None,
               mocap_pos=None, mocap_quat=None) -> torch.Tensor:
  """Compute inverse force through all native inverse stages.

  This is the NONE-skip entry point, including the model's INVDISCRETE
  acceleration conversion. Returns owned [batch,nv] force and preserves
  borrowed forward buffers/records on success and failure. Explicit input
  validation may synchronize device tensors at this query boundary.
  """
  return mj_inverseSkip(sim, qpos=qpos, qvel=qvel, qacc=qacc,
                        mocap_pos=mocap_pos, mocap_quat=mocap_quat)


def _inverse_impl(sim, qpos, qvel, qacc, mocap_pos, mocap_quat):
  """Internal stateless assembly reference; public queries use inverse stages.

  Retained for isolated mass/constraint reduction checks while stage producers
  are qualified. It does not implement integrator flags or cached skip stages.
  """
  if torch is None:
    raise RuntimeError("PyTorch with MPS is required")
  state = sim.state
  saved_spatial_kin = getattr(sim, "_spatial_kin", None)
  saved_spatial_key = getattr(sim, "_spatial_cache_key", None)
  qp = _stage_tensor(sim, qpos, (sim.batch_size, sim._mjmodel.nq), "qpos") if qpos is not None else state._qpos
  qv = _stage_tensor(sim, qvel, (sim.batch_size, sim._mjmodel.nv), "qvel") if qvel is not None else state._qvel
  qa = _stage_tensor(sim, qacc, (sim.batch_size, sim._mjmodel.nv), "qacc") if qacc is not None else state._qacc

  # 1. Run kinematics & smooth dynamics to obtain mass matrix and Coriolis bias
  mpos = _stage_tensor(sim, mocap_pos, (sim.batch_size, state._nmocap, 3), "mocap_pos") if mocap_pos is not None else getattr(state, "_mpos", None)
  mquat = _stage_tensor(sim, mocap_quat, (sim.batch_size, state._nmocap, 4), "mocap_quat") if mocap_quat is not None else getattr(state, "_mquat", None)
  if mquat is not None and not bool((torch.abs(torch.linalg.vector_norm(mquat, dim=-1) - 1) <= 1e-5).all()):
    raise ValueError("mocap_quat values must be unit quaternions")
  dynamics = _query_smooth(sim, qp, qv, mpos, mquat)
  sparse = getattr(sim, "_component_mass_enabled", False)
  M = None if sparse else dynamics["mass_matrix"]
  bias = dynamics["qfrc_bias"]      # [batch, nv]
  tendon_force = None
  if getattr(sim, "_tendons", None) is not None:
    tendon_force, _, tendon_armature = sim._tendons.run_device(qp, qv)
    if not sparse:
      M = M + tendon_armature
  spatial_J = spatial_length = None
  if getattr(sim, "_spatial_tendons", None) is not None:
    kin = sim._spatial_jacobian(qv, dynamics["poses"])
    spatial_J = kin
    skin = sim._spatial_kin
    sf, _, sa = sim._spatial_tendons.run_forces(skin, include_armature=not sparse)
    if not sparse:
      M = M + sa.reshape(M.shape)
    tendon_force = sf.reshape((sim.batch_size, sim._mjmodel.nv)) if tendon_force is None else tendon_force + sf.reshape((sim.batch_size, sim._mjmodel.nv))
    sbias, _ = sim._spatial_tendons.run_armature_bias(
        skin, qv, dynamics["poses"], dynamics.get("cvel"),
        dynamics.get("root_com"), dynamics.get("cdof"),
        dynamics.get("cdof_dot"))
    bias = bias + sbias.reshape(bias.shape)

  # 2. Compute M * qacc on MPS
  if sparse:
    M_qacc = sim._smooth.mass_blocks_matvec_device(
        dynamics["mass_blocks"], qa, dynamics.get("tendon_armature_blocks"))
  else:
    M_qacc = torch.bmm(M, qa.unsqueeze(-1)).squeeze(-1)

  # 3. Add bias forces
  qfrc_inv = M_qacc + bias

  # 4. Subtract passive forces (damping, spring) if active
  if sim._passive is not None:
    passive = sim._passive.run_device(qp, qv, mocap_pos=mpos, mocap_quat=mquat)
    qfrc_inv = qfrc_inv - passive

  # 5. Subtract fluid forces if active
  if getattr(sim, "_fluid", None) is not None:
    qfrc_inv = qfrc_inv - sim._fluid.run_device(qp, qv, dynamics)

  # 6. Subtract tendon forces if active
  if tendon_force is not None:
    qfrc_inv = qfrc_inv - tendon_force

  # 7. Subtract flex passive forces if active
  if getattr(sim, "_flex", None) is not None:
    flex_qfrc, _, _ = sim._flex.run_device(
        qp, qv, dict(dynamics["poses"], root_com=dynamics["root_com"]),
        dynamics.get("cvel", None)
    )
    qfrc_inv = qfrc_inv - flex_qfrc

  # 8. Subtract constraint forces (mj_invConstraint)
  if getattr(sim, "_coupled_constraints", None) is not None:
    cc = sim._coupled_constraints
    eq_act = getattr(state, "_eq_active", None)
    _ten_J, _ten_L = sim._spatial_for_coupled(qv, dynamics["poses"]) if hasattr(sim, "_spatial_for_coupled") else (None, None)
    constraint_result = cc.assemble_device(
        dict(dynamics["poses"], root_com=dynamics["root_com"]), qp, qv,
        eq_active=eq_act, cvel=dynamics.get("cvel"),
        cdof=dynamics.get("cdof"), cdof_dot=dynamics.get("cdof_dot"),
        tendon_J_spatial=_ten_J, tendon_length_spatial=_ten_L,
        flex=getattr(sim, "_flex", None))
    from mujoco_metal.inverse_constraints import inverse_constraint_force
    qfrc_inv = qfrc_inv - inverse_constraint_force(
        constraint_result, qa, cc.descriptor)

  if hasattr(sim, "_spatial_kin"):
    sim._spatial_kin = saved_spatial_kin
    sim._spatial_cache_key = saved_spatial_key
  return qfrc_inv


def _inverse_constraint_force_from_cached(sim, cc, qacc, record=None, *,
                                          qacc_low=None, return_low=False):
  """Reduce retained canonical rows; no POS/VEL assembly or optimizer call."""
  if record is not None:
    sim.validate_forward_stage_record(record, "VEL")
  from mujoco_metal.inverse_constraints import inverse_constraint_force
  return inverse_constraint_force(
      cc._assembly_views(include_optimizer_outputs=False), qacc, cc.descriptor,
      qacc_low=qacc_low, return_low=return_low)


# -------------------------------------------------------------------------
# Native Split-Stage APIs (runs on device)
# -------------------------------------------------------------------------

def _stage_tensor(sim, value, shape, name):
  """Validate and stage one explicit native stage input without mutation."""
  if value is None:
    return None
  if isinstance(value, torch.Tensor):
    if (value.device.type != sim.state._device.type or
        (sim.state._device.index is not None and value.device.index != sim.state._device.index)):
      raise ValueError(f"{name} tensor must be on {sim.state._device}")
    if value.dtype not in (torch.float32, torch.float64):
      raise TypeError(f"{name} tensor must be floating point")
    out = value.to(dtype=torch.float32).contiguous()
    if tuple(out.shape) != tuple(shape) or not bool(torch.isfinite(out).all()):
      raise ValueError(f"{name} must be finite with shape {tuple(shape)}")
    _validate_stage_quaternions(sim, out, name)
    return out
  arr = np.asarray(value)
  if arr.dtype.kind not in "fiu" or arr.shape != tuple(shape):
    raise ValueError(f"{name} must be numeric with shape {tuple(shape)}")
  with np.errstate(over="ignore", invalid="ignore"):
    arr = np.asarray(arr, dtype=np.float32)
  if not np.all(np.isfinite(arr)):
    raise ValueError(f"{name} must be finite and float32-representable")
  out = torch.as_tensor(arr, dtype=torch.float32, device=sim.state._device)
  _validate_stage_quaternions(sim, out, name)
  return out


def _validate_stage_quaternions(sim, value, name):
  model = sim._mjmodel
  if name == "qpos":
    q = value.reshape(sim.batch_size, int(model.nq))
    for jid in range(int(model.njnt)):
      jt, adr = int(model.jnt_type[jid]), int(model.jnt_qposadr[jid])
      if jt == int(mujoco.mjtJoint.mjJNT_FREE):
        quat = q[:, adr + 3:adr + 7]
      elif jt == int(mujoco.mjtJoint.mjJNT_BALL):
        quat = q[:, adr:adr + 4]
      else:
        continue
      if not bool((torch.abs(torch.linalg.vector_norm(quat, dim=-1) - 1) <= 1e-5).all()):
        raise ValueError("qpos contains a non-unit joint quaternion")
  elif name == "mocap_quat" and value.numel():
    quat = value.reshape(sim.batch_size, -1, 4)
    if not bool((torch.abs(torch.linalg.vector_norm(quat, dim=-1) - 1) <= 1e-5).all()):
      raise ValueError("mocap_quat values must be unit quaternions")


def _stage_dynamics(sim, dynamics):
  if dynamics is None:
    return None
  if not isinstance(dynamics, dict):
    raise TypeError("dynamics must be a dynamics dictionary from mj_fwdVelocity")
  b, nv = sim.batch_size, int(sim._mjmodel.nv)
  sparse = getattr(sim, "_component_mass_enabled", False)
  if sparse:
    mass = dynamics.get("mass_blocks")
    expected_layout = sim._smooth.mass_block_layout
    layout = dynamics.get("mass_block_layout")
    if not isinstance(layout, dict):
      raise ValueError("dynamics is missing the compiled mass_block_layout")
    for key in ("ncomponent", "nnz", "component_dof_offsets", "component_dof_ids",
                "component_dofnum", "component_mass_offsets", "dof_component", "dof_local_index"):
      if key not in layout or not np.array_equal(layout[key], expected_layout[key]):
        raise ValueError(f"dynamics.mass_block_layout.{key} does not match the model")
    mass_name, mass_shape = "mass_blocks", (b, int(expected_layout["nnz"]))
  else:
    mass = dynamics.get("mass_matrix")
    mass_name, mass_shape = "mass_matrix", (b, nv, nv)
  bias = dynamics.get("qfrc_bias")
  if not isinstance(mass, torch.Tensor) or tuple(mass.shape) != mass_shape:
    raise ValueError(f"dynamics.{mass_name} has invalid shape")
  if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (b, nv):
    raise ValueError("dynamics.qfrc_bias has invalid shape")
  def _same_device(value):
    return (value.device.type == sim.state._device.type and
            (sim.state._device.index is None or value.device.index == sim.state._device.index))
  if not _same_device(mass) or not _same_device(bias):
    raise ValueError("dynamics tensors must remain on the simulation device")
  if (mass.dtype != torch.float32 or bias.dtype != torch.float32 or
      not mass.is_contiguous() or not bias.is_contiguous()):
    raise ValueError("dynamics tensors must be contiguous float32")
  if not bool(torch.isfinite(mass).all() and torch.isfinite(bias).all()):
    raise ValueError("dynamics contains nonfinite values")
  if sparse:
    armature = dynamics.get("tendon_armature_blocks")
    if getattr(sim._smooth, "_has_tendon_armature", False) and armature is None:
      raise ValueError("dynamics is missing tendon_armature_blocks")
    if armature is not None and (
        not isinstance(armature, torch.Tensor) or tuple(armature.shape) != mass_shape
        or not _same_device(armature) or armature.dtype != torch.float32
        or not armature.is_contiguous() or not bool(torch.isfinite(armature).all())):
      raise ValueError("dynamics.tendon_armature_blocks has invalid native layout or values")
  if "poses" not in dynamics or not isinstance(dynamics["poses"], dict):
    raise ValueError("dynamics is missing the position-stage poses")
  from mujoco_metal.smooth_metal import validate_pose_dict
  validate_pose_dict(sim._mjmodel, dynamics["poses"], b,
                     sim.state._device, torch)
  # The remaining smooth buffers are consumed by inverse/constraint stages;
  # require their exact native layouts now instead of failing in a shader.
  expected = {
      "cvel": (b, int(sim._mjmodel.nbody), 6),
      "root_com": (b, int(sim._mjmodel.nbody), 3),
      "cdof": (b, nv, 6),
      "cdof_dot": (b, nv, 6),
  }
  for name, shape in expected.items():
    value = dynamics.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
      raise ValueError(f"dynamics.{name} has invalid shape")
    if not _same_device(value) or value.dtype != torch.float32 or not value.is_contiguous():
      raise ValueError(f"dynamics.{name} must be contiguous float32 on the simulation device")
    if not bool(torch.isfinite(value).all()):
      raise ValueError(f"dynamics.{name} contains nonfinite values")
  return dynamics


def _query_smooth(sim, qpos, qvel, mocap_pos=None, mocap_quat=None, *, poses=None):
  """Run the selected smooth layout with canonical tendon armature inputs.

  Sparse armature projection consumes the position-stage tendon Jacobian.
  Calling the smooth program directly omits that input for spatial tendons.
  Query calls use all DOFs rather than stepping's asleep-tree lists.
  """
  if not getattr(sim, "_component_mass_enabled", False):
    if poses is None:
      return sim._smooth.run_device(qpos, qvel, mocap_pos, mocap_quat)
    return sim._smooth.run_device(qpos, qvel, poses=poses)
  if poses is None:
    poses = sim._smooth._fk.run_device(qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)
  # Explicit inputs may reuse the same tensor object after an in-place write;
  # never reuse a previous spatial Jacobian for an explicit query boundary.
  sim._spatial_cache_key = None
  tendon_J = sim._component_tendon_jacobian(qvel, poses)
  return sim._smooth.run_device(qpos, qvel, poses=poses,
      awake_lists=sim._smooth._workspace["all_awake_lists"], tendon_J=tendon_J)

def _prepared_record(sim, minimum, record=None, *, qpos=None, qvel=None,
                     poses=None, dynamics=None):
  """Validate a published stage before allowing any downstream writes.

  Device inputs identify the captured storage. Repeated host inputs are
  accepted only after a value check at this explicit API boundary. Borrowed
  pose/dynamics dictionaries must be the actual published objects; shape
  compatibility alone cannot establish that their stages are coherent.
  """
  from mujoco_metal.forward_stages import ForwardStage
  stages, generation = sim._forward_stages, sim.state.generation
  if record is None:
    record = stages.current(generation=generation, minimum=minimum)
  record = sim.validate_forward_stage_record(record, minimum)
  position = stages.consume(record, ForwardStage.POS, generation=generation)
  if poses is not None and poses is not position["poses"]:
    raise ValueError("poses must be the published position-stage record")
  if qpos is not None:
    _match_stage_input(sim, qpos, record.qpos, "qpos")
  if qvel is not None or dynamics is not None:
    velocity = stages.consume(record, ForwardStage.VEL, generation=generation)
    if qvel is not None:
      _match_stage_input(sim, qvel, velocity["qvel"], "qvel")
    if dynamics is not None:
      _stage_dynamics(sim, dynamics)
      if dynamics is not velocity["dynamics"]:
        raise ValueError("dynamics must be the published velocity-stage record")
  return record


def _match_stage_input(sim, value, captured, name):
  if isinstance(value, torch.Tensor):
    if value is not captured:
      raise ValueError(f"{name} must identify the captured stage input tensor")
    return captured
  checked = _stage_tensor(sim, value, tuple(captured.shape), name)
  if not bool(torch.equal(checked, captured)):
    raise ValueError(f"{name} does not match the captured stage input")
  return captured


def mj_fwdPosition(sim, qpos=None, *, mocap_pos=None, mocap_quat=None,
                   skipsensor=False, return_record=False):
  """Compute only POS and publish its generation-scoped borrowed record.

  Returns poses for compatibility; ``return_record=True`` returns the record
  accepted explicitly by every subsequent stage. This call does not run RNE,
  actuation, acceleration, or the constraint optimizer.
  """
  state = sim.state
  qp = _stage_tensor(sim, qpos, (sim.batch_size, sim._mjmodel.nq), "qpos")
  mp = _stage_tensor(sim, mocap_pos, (sim.batch_size, state._nmocap, 3), "mocap_pos")
  mq = _stage_tensor(sim, mocap_quat, (sim.batch_size, state._nmocap, 4), "mocap_quat")
  record = sim.prepare_forward_position(qp, mocap_pos=mp, mocap_quat=mq,
                                        skipsensor=skipsensor)
  if return_record:
    return record
  from mujoco_metal.forward_stages import ForwardStage
  return record.values[ForwardStage.POS]["poses"]


def mj_fwdVelocity(sim, qpos=None, qvel=None, poses=None, dynamics=None,
                   *, record=None, skipsensor=False):
  """Refresh VEL from the captured POS without recomputing geometry or mass.

  Passing the current published ``dynamics`` requests its already computed
  value. Foreign dictionaries are rejected before a producer can overwrite
  borrowed buffers.
  """
  from mujoco_metal.forward_stages import ForwardStage
  record = _prepared_record(sim, ForwardStage.POS, record, qpos=qpos,
                            poses=poses, dynamics=dynamics)
  if dynamics is not None:
    if qvel is not None:
      velocity = sim._forward_stages.consume(
          record, ForwardStage.VEL, generation=sim.state.generation)
      _match_stage_input(sim, qvel, velocity["qvel"], "qvel")
    return dynamics
  qv = _stage_tensor(sim, qvel, (sim.batch_size, sim._mjmodel.nv), "qvel")
  return sim.prepare_forward_velocity(record, qv, skipsensor=skipsensor)["dynamics"]


def mj_fwdActuation(sim, qpos=None, qvel=None, poses=None, ctrl=None, *, record=None):
  """Compute ACT from prepared POS/VEL; a control override is temporary."""
  from mujoco_metal.forward_stages import ForwardStage
  record = _prepared_record(sim, ForwardStage.VEL, record, qpos=qpos,
                            qvel=qvel, poses=poses)
  control = _stage_tensor(sim, ctrl, (sim.batch_size, sim._mjmodel.nu), "ctrl")
  return sim.prepare_forward_actuation(record, ctrl=control)["qfrc_actuator"]


def mj_fwdAcceleration(sim, qpos=None, qvel=None, poses=None, dynamics=None,
                       qfrc_applied=None, *, record=None):
  """Solve unconstrained ACC from captured forces, without rerunning POS/VEL.

  If ACT has not yet been published, compute that missing prerequisite once
  with current controls. Explicit applied force does not change simulation
  input storage.
  """
  from mujoco_metal.forward_stages import ForwardStage
  force = _stage_tensor(sim, qfrc_applied, (sim.batch_size, sim._mjmodel.nv), "qfrc_applied")
  record = _prepared_record(sim, ForwardStage.VEL, record, qpos=qpos,
                            qvel=qvel, poses=poses, dynamics=dynamics)
  if record.stage < ForwardStage.ACT:
    sim.prepare_forward_actuation(record)
  result = sim.prepare_forward_acceleration(record, qfrc_applied=force)
  return result["qacc_smooth"], result["status"]


def mj_fwdConstraint(sim, qpos=None, qvel=None, poses=None, dynamics=None,
                     *, record=None, skipsensor=False):
  """Solve captured constraint rows using the published unconstrained ACC."""
  from mujoco_metal.forward_stages import ForwardStage
  record = _prepared_record(sim, ForwardStage.ACC, record, qpos=qpos,
                            qvel=qvel, poses=poses, dynamics=dynamics)
  result = sim.prepare_forward_constraint(record, skipsensor=skipsensor)
  velocity = sim._forward_stages.consume(
      record, ForwardStage.VEL, generation=sim.state.generation)
  return result["qacc"], result["status"], velocity["dynamics"]


def _constraint_jacobian_product(sim, vector, transpose, record):
  """Consume a coherent prepared canonical row record without dispatching."""
  from mujoco_metal.forward_stages import ForwardStage
  record = _prepared_record(sim, ForwardStage.CONSTRAINT, record)
  stage = sim._forward_stages.consume(
      record, ForwardStage.CONSTRAINT, generation=sim.state.generation)
  rows = stage.get("canonical_rows", stage)
  b, nv = sim.batch_size, int(sim._mjmodel.nv)
  active = None if rows is None else rows.get("active")
  if active is None:
    cc = getattr(sim, "_coupled_constraints", None)
    if cc is not None and cc.descriptor.nr:
      raise RuntimeError("constraint stage did not publish canonical row activity")
    nr = 0
  else:
    if (not isinstance(active, torch.Tensor) or active.ndim != 2
        or active.shape[0] != b or active.dtype != torch.float32
        or active.device.type != sim.state._device.type):
      raise ValueError("canonical row activity must be device float32 [B,nr]")
    nr = int(active.shape[1])
  if vector is None:
    raise TypeError("vector is required")
  vector = _stage_tensor(sim, vector, (b, nr if transpose else nv), "vector")
  if nr == 0:
    return torch.zeros((b, nv if transpose else 0),
                       dtype=torch.float32, device=sim.state._device)
  status = stage["status"]
  live = (active > .5) & (status == 0)[:, None]
  packed = rows.get("J_packed")
  if packed is not None:
    from mujoco_metal.constraint_jacobian import PACKED_J_CSR, PACKED_J_DENSE
    from mujoco_metal.inverse_constraints import (
        _packed_jacobian_matvec, _packed_jacobian_transpose_matvec)
    layout, pattern = rows.get("jacobian_layout"), rows.get("jacobian_pattern")
    if (layout is None or layout.nr != nr or layout.nv != nv
        or not isinstance(packed, torch.Tensor) or packed.dtype != torch.float32
        or packed.device != vector.device or packed.ndim != 1
        or packed.numel() < b * layout.stride_words):
      raise ValueError("canonical packed Jacobian has invalid layout")
    if layout.mode == PACKED_J_CSR:
      if (pattern is None or pattern.nr != nr or pattern.nv != nv
          or pattern.nnz != layout.nnz):
        raise ValueError("canonical sparse Jacobian requires its compiled pattern")
      if transpose:
        force = torch.where(live, vector, 0)
        result = _packed_jacobian_transpose_matvec(packed, layout, pattern, force)
      else:
        result = _packed_jacobian_matvec(
            packed, layout, pattern, vector, torch.zeros_like(active))
        result = torch.where(live, result, 0)
      return torch.where((status == 0)[:, None], result, 0).clone()
    if layout.mode != PACKED_J_DENSE:
      raise ValueError("unknown canonical Jacobian storage mode")
    matrices = []
    for world in range(b):
      base = world * layout.stride_words + layout.values_offset
      matrices.append(packed[base:base+nr*nv].reshape(nr, nv))
    jacobian = torch.stack(matrices)
  else:
    jacobian = rows.get("J")
  if (not isinstance(jacobian, torch.Tensor)
      or jacobian.dtype != torch.float32 or jacobian.device != vector.device
      or tuple(jacobian.shape) != (b, nr, nv)):
    raise ValueError("canonical dense Jacobian has invalid layout")
  jacobian = torch.where(live[:, :, None], jacobian, 0)
  result = (torch.bmm(jacobian.transpose(1, 2),
                      torch.where(live, vector, 0).unsqueeze(-1)).squeeze(-1)
            if transpose else
            torch.bmm(jacobian, vector.unsqueeze(-1)).squeeze(-1))
  return torch.where((status == 0)[:, None], result, 0).clone()


def mj_mulJacVec(sim, vector, *, record=None):
  """Owned J*vector [B,nr] from a completed native constraint stage.

  Rows use reserved canonical slots, with inactive/failed rows zeroed; CPU
  MuJoCo instead exposes compact active rows. Sparse storage is consumed
  directly without materializing a dense Jacobian or recomputing physics.
  """
  return _constraint_jacobian_product(sim, vector, False, record)


def mj_mulJacTVec(sim, vector, *, record=None):
  """Owned J.T*vector [B,nv] with reserved-row and record rules of mj_mulJacVec."""
  return _constraint_jacobian_product(sim, vector, True, record)


def mj_contactForce(sim, contact_id, *, record=None):
  """Return owned ``[batch,6]`` force/torque in a native contact's frame.

  ``contact_id`` is a native candidate slot, not the CPU engine's compact
  contact index. Consume a completed, coherent forward record; this query
  never detects contacts or runs a solver. Invalid slots, inactive contacts,
  and failed worlds return zero, matching the source's absent-force behavior.
  """
  from numbers import Integral
  from mujoco_metal.forward_stages import ForwardStage
  if isinstance(contact_id, (bool, np.bool_)) or not isinstance(contact_id, Integral):
    raise TypeError("contact_id must be an integer native contact slot")
  record = _prepared_record(sim, ForwardStage.CONSTRAINT, record)
  result = sim._forward_stages.consume(
      record, ForwardStage.CONSTRAINT, generation=sim.state.generation)
  wrench = result.get("contact_wrench")
  mask = result.get("contact_mask")
  if wrench is None:
    legacy = result.get("contact_result")
    if legacy is not None:
      # The legacy profile admits condim 1/3 pyramidal contacts. Its five
      # slots retain an unused normal row for condim3, then four edges.
      rows = legacy["force_rows"]
      friction = sim._contact._constants["friction"].reshape(-1, 2)
      condim = sim._contact._constants["condim"].reshape(-1)
      normal = torch.where(condim[None, :] == 3,
                           rows[:, :, 1:5].sum(dim=-1), rows[:, :, 0])
      tangent1 = friction[None, :, 0] * (rows[:, :, 1] - rows[:, :, 2])
      tangent2 = friction[None, :, 1] * (rows[:, :, 3] - rows[:, :, 4])
      zero = torch.zeros_like(normal)
      wrench = torch.stack((normal, tangent1, tangent2, zero, zero, zero), dim=-1)
      mask = legacy["mask"]
  output = torch.zeros((sim.batch_size, 6), dtype=torch.float32,
                       device=sim.state._device)
  if wrench is None:
    # A contact-free stage has no contact result; an admitted legacy contact
    # path must publish its actual solver result rather than hide a gap.
    if getattr(sim, "_contact", None) is not None:
      raise RuntimeError("constraint stage did not publish contact forces")
    return output
  if not 0 <= int(contact_id) < wrench.shape[1]:
    return output
  active = (mask[:, int(contact_id)] > 0) & (result["status"] == 0)
  output.copy_(torch.where(active[:, None], wrench[:, int(contact_id)], output))
  return output


def _skip_stage_inputs(sim, skipstage, skipsensor, record, qpos, qvel,
                       mocap_pos, mocap_quat):
  """Validate a shared forward/inverse prefix without producer writes."""
  from mujoco_metal.forward_stages import ForwardStage
  from numbers import Integral
  if (isinstance(skipstage, (bool, np.bool_)) or
      not isinstance(skipstage, (Integral, mujoco.mjtStage))):
    raise TypeError("skipstage must be a MuJoCo stage integer")
  skip = int(skipstage)
  if skip not in tuple(int(value) for value in mujoco.mjtStage.__members__.values()):
    raise ValueError(f"invalid skipstage {skip}")
  if not isinstance(skipsensor, (bool, np.bool_)):
    raise TypeError("skipsensor must be boolean")
  state, model = sim.state, sim._mjmodel
  if skip == int(mujoco.mjtStage.mjSTAGE_NONE):
    qp = _stage_tensor(sim, qpos, (sim.batch_size, model.nq), "qpos")
    qv = _stage_tensor(sim, qvel, (sim.batch_size, model.nv), "qvel")
    mp = _stage_tensor(sim, mocap_pos, (sim.batch_size, state._nmocap, 3), "mocap_pos")
    mq = _stage_tensor(sim, mocap_quat, (sim.batch_size, state._nmocap, 4), "mocap_quat")
  else:
    record = _prepared_record(sim, ForwardStage.POS if skip == 1 else ForwardStage.VEL,
        record, qpos=qpos, qvel=qvel if skip > 1 else None)
    qp, mp, mq = record.qpos, record.mocap_pos, record.mocap_quat
    for name, value, captured in (("mocap_pos", mocap_pos, mp),
                                  ("mocap_quat", mocap_quat, mq)):
      if value is not None:
        _match_stage_input(sim, value, captured, name)
    qv = (_stage_tensor(sim, qvel, (sim.batch_size, model.nv), "qvel") if skip == 1
          else record.values[ForwardStage.VEL]["qvel"])
  return skip, record, qp, qv, mp, mq


def mj_forwardSkip(sim, skipstage=mujoco.mjtStage.mjSTAGE_NONE,
                   skipsensor=False, *, record=None, qpos=None, qvel=None,
                   mocap_pos=None, mocap_quat=None, ctrl=None):
  """Evaluate forward dynamics with an explicitly validated cached prefix.

  Returns the full borrowed stage bundle. For POS/VEL/ACC skips, unchanged
  host inputs are matched at this boundary and device inputs identify the
  captured tensor. Skipping never guesses that arbitrary workspace contents
  form a valid prefix.
  """
  skip, record, qp, qv, mp, mq = _skip_stage_inputs(
      sim, skipstage, skipsensor, record, qpos, qvel, mocap_pos, mocap_quat)
  control = _stage_tensor(sim, ctrl, (sim.batch_size, sim._mjmodel.nu), "ctrl")
  return sim.forward_skip(skip, skipsensor, record=record, qpos=qp, qvel=qv,
                          mocap_pos=mp, mocap_quat=mq, ctrl=control)


def mj_forward(sim, **inputs):
  """Compute all native forward stages; return their borrowed result bundle."""
  return mj_forwardSkip(sim, mujoco.mjtStage.mjSTAGE_NONE, **inputs)


def mj_step1(sim, *, control_callback=None):
  """Prepare POS/VEL, sensors and energy without advancing simulation time.

  An optional Python callback is an explicit host extension invoked after
  those stages. The default has no callback. Its returned controls are
  installed for step2; the callback is not represented as GPU-native code.
  """
  if control_callback is not None and not callable(control_callback):
    raise TypeError("control_callback must be callable or None")
  return sim.step1(control_callback=control_callback)


def mj_step2(sim, *, ctrl=None, qfrc_applied=None, xfrc_applied=None):
  """Finish a pending step1 with ACT/ACC/constraints and one time step.

  Inputs use explicit batched shapes: ctrl [B,nu], qfrc_applied [B,nv],
  and xfrc_applied [B,nbody,6]. All inputs are validated before any producer
  writes. MuJoCo's split stepping uses Euler for RK4 models; implicit models
  retain their implicit integrator.
  """
  from mujoco_metal.forward_stages import ForwardStage
  record = getattr(sim, "_step1_record", None)
  if record is None:
    raise ValueError("step2 requires a successful step1 prefix")
  _prepared_record(sim, ForwardStage.VEL, record)
  model = sim._mjmodel
  control = _stage_tensor(sim, ctrl, (sim.batch_size, model.nu), "ctrl")
  force = _stage_tensor(sim, qfrc_applied, (sim.batch_size, model.nv),
                        "qfrc_applied")
  wrench = _stage_tensor(sim, xfrc_applied, (sim.batch_size, model.nbody, 6),
                         "xfrc_applied")
  return sim.step2(ctrl=control, qfrc_applied=force, xfrc_applied=wrench)


def mj_inverseSkip(sim, skipstage=mujoco.mjtStage.mjSTAGE_NONE,
                   skipsensor=False, *, record=None, qpos=None, qvel=None,
                   qacc=None, mocap_pos=None, mocap_quat=None, return_details=False):
  """Compute owned inverse force from a validated native POS/VEL prefix.

  This explicit query preserves the simulation's borrowed stages and stored
  sensors. ``return_details=True`` returns owned numerical outputs (including
  status) rather than the transient query's record. INVDISCRETE is handled by
  the simulation's pinned integrator-specific acceleration conversion.
  """
  if not isinstance(return_details, (bool, np.bool_)):
    raise TypeError("return_details must be boolean")
  skip, record, qp, qv, mp, mq = _skip_stage_inputs(
      sim, skipstage, skipsensor, record, qpos, qvel, mocap_pos, mocap_quat)
  qa = _stage_tensor(sim, qacc, (sim.batch_size, sim._mjmodel.nv), "qacc")
  with _inverse_query_workspaces(sim):
    result = sim.inverse_skip(skip, skipsensor, record=record, qpos=qp, qvel=qv,
                             qacc=qa, mocap_pos=mp, mocap_quat=mq)
    if return_details:
      return {name: value.clone() if isinstance(value, torch.Tensor) else value
              for name, value in result.items() if name != "record"}
    return result["qfrc_inverse"].clone()


def mj_invPosition(sim, qpos=None, *, mocap_pos=None, mocap_quat=None,
                   return_record=False):
  """Publish native inverse POS without evaluating sensors or later stages.

  MuJoCo's inverse POS shares kinematics, mass, collision, constraint assembly
  and transmissions with forward POS. This backend counterpart uses that
  shared producer and returns borrowed poses, or its generation-scoped record
  when ``return_record=True``. It does not advance state or run an optimizer.
  Existing prepared records are replaced, as for :func:`mj_fwdPosition`.
  """
  if not isinstance(return_record, (bool, np.bool_)):
    raise TypeError("return_record must be boolean")
  return mj_fwdPosition(sim, qpos, mocap_pos=mocap_pos,
      mocap_quat=mocap_quat, skipsensor=True, return_record=return_record)


def mj_invVelocity(sim, qpos=None, qvel=None, poses=None, dynamics=None,
                   *, record=None):
  """Publish inverse VEL from native POS, without sensors or actuation.

  The pinned engine's ``mj_invVelocity`` calls ``mj_fwdVelocity`` directly.
  This counterpart has the same native prepared-stage reuse contract as
  :func:`mj_fwdVelocity`; returned dynamics are borrowed until a producer
  overwrites them. Foreign, invalidated or mutated prefixes are rejected.
  """
  return mj_fwdVelocity(sim, qpos, qvel, poses, dynamics,
                        record=record, skipsensor=True)


def mj_invConstraint(sim, qacc=None, *, record=None, return_details=False):
  """Evaluate owned inverse constraint force from a prepared native VEL.

  Applies the pinned constraint-cost gradient to ``J*qacc-aref`` and reduces
  it with ``J.T``. No forward optimizer, ACT, mass solve, sensor evaluation or
  INVDISCRETE acceleration conversion runs in this standalone primitive.
  The full :func:`mj_inverse` applies that conversion before the constraint
  stage. Query scratch is restored on success and failure, so prepared
  forward/inverse views remain usable. ``return_details=True`` also returns
  an owned stage status; default output is owned ``[batch,nv]`` force.
  """
  from mujoco_metal.forward_stages import ForwardStage
  if not isinstance(return_details, (bool, np.bool_)):
    raise TypeError("return_details must be boolean")
  checked = _stage_tensor(sim, qacc, (sim.batch_size, sim._mjmodel.nv), "qacc")
  record = _prepared_record(sim, ForwardStage.VEL, record)
  velocity = sim._forward_stages.consume(
      record, ForwardStage.VEL, generation=sim.state.generation)
  acceleration = (sim.state._qacc if checked is None else checked).clone()
  with _inverse_query_workspaces(sim):
    if sim._coupled_constraints is not None:
      force = _inverse_constraint_force_from_cached(
          sim, sim._coupled_constraints, acceleration, record)
    elif sim._contact is not None or sim._joint_constraints is not None:
      rows, descriptor = sim._assemble_legacy_inverse_rows(record, velocity)
      from mujoco_metal.inverse_constraints import inverse_constraint_force
      force = inverse_constraint_force(rows, acceleration, descriptor)
    else:
      force = torch.zeros_like(acceleration)
    if return_details:
      return {"qfrc_constraint_inverse": force.clone(),
              "status": velocity["status"].clone()}
    return force.clone()


def mj_compareFwdInv(sim, *, record=None, forward_result=None):
  """Return owned [batch,2] forward/inverse residuals, preserving stage buffers."""
  from mujoco_metal.forward_stages import ForwardStage
  record = _prepared_record(sim, ForwardStage.CONSTRAINT, record)
  with _inverse_query_workspaces(sim):
    return sim.compare_forward_inverse(record, forward_result).clone()


def _mass_query(sim, operation, vector=None, *, dynamics=None):
  """Run an explicit mass query without invalidating borrowed forward views."""
  from mujoco_metal.mass_queries import (
      full_mass, mass_product, mass_factor_query, factor_mass)
  checked = None
  if vector is not None:
    shape = tuple(vector.shape) if hasattr(vector, "shape") else np.asarray(vector).shape
    if (len(shape) not in (2, 3) or shape[0] != sim.batch_size
        or shape[-1] != sim._mjmodel.nv):
      raise ValueError("vector must have shape [batch,nv] or [batch,nrhs,nv]")
    checked = _stage_tensor(sim, vector, shape, "vector")
  if dynamics is not None:
    _stage_dynamics(sim, dynamics)
  with _inverse_query_workspaces(sim):
    if dynamics is None:
      state = sim.state
      stage = _query_smooth(sim, state._qpos, state._qvel,
          getattr(state, "_mpos", None), getattr(state, "_mquat", None))
      # Sparse smooth has already projected canonical tendon armature. Dense
      # smooth intentionally returns rigid/joint mass; explicit mass queries
      # add fixed/spatial tendon armature just once, as forward positioning does.
      if not getattr(sim, "_component_mass_enabled", False):
        stage = dict(stage)
        mass = stage["mass_matrix"]
        if getattr(sim, "_tendons", None) is not None:
          _, _, armature = sim._tendons.run_device(state._qpos, state._qvel)
          mass = mass + armature
        if getattr(sim, "_spatial_tendons", None) is not None:
          sim._spatial_jacobian(state._qvel, stage["poses"])
          _, _, armature = sim._spatial_tendons.run_forces(
              sim._spatial_kin, include_armature=True)
          mass = mass + armature.reshape_as(mass)
        stage["mass_matrix"] = mass
    else:
      stage = dynamics
    if operation == "full":
      return full_mass(stage)
    if operation == "factor":
      return factor_mass(stage)
    if operation == "product":
      return mass_product(stage, checked)
    return mass_factor_query(stage, checked, operation)


def mj_fullM(sim, *, dynamics=None):
  """Return owned [batch,nv,nv] mass, including joint/tendon armature.

  Sparse layouts are explicitly expanded for this query only. Stepping and
  sparse solves never use this conversion as a dense solver fallback.
  """
  return _mass_query(sim, "full", dynamics=dynamics)


def mj_factorM(sim, *, dynamics=None):
  """Return owned device L' D L factors including joint/tendon armature.

  The result has ``blocks`` (each with ``dof_ids``, unit lower ``L``, ``D``
  and ``Dinv``), ``status`` (int32[batch], 0 success/2 failure), and ``nv``.
  Component mass remains component-local; the query never changes prepared
  factors or stages. Failed worlds return zero factors. This backend API
  returns explicit factors rather than writing upstream ``MjData.qLD``.
  """
  return _mass_query(sim, "factor", dynamics=dynamics)


def mj_mulM(sim, vector, *, dynamics=None):
  """Apply mass on device to one or several RHS per world; return owned output."""
  return _mass_query(sim, "product", vector, dynamics=dynamics)


def mj_mulM2(sim, vector, *, dynamics=None):
  """Return (sqrt(D)*L*vector, status), using pinned descending L' D L.

  This is MuJoCo's triangular mass square-root factor, not the symmetric
  principal square root. Status is [batch] int32; failed worlds return zero.
  """
  return _mass_query(sim, "sqrt", vector, dynamics=dynamics)


def mj_solveM(sim, vector, *, dynamics=None):
  """Return owned (inverse(M)*vector, status), preserving query workspace owners."""
  return _mass_query(sim, "solve", vector, dynamics=dynamics)


def mj_solveM2(sim, vector, *, dynamics=None):
  """Return (sqrt(inverse(D))*inverse(L')*vector, status), with D from this mass."""
  return _mass_query(sim, "half_solve", vector, dynamics=dynamics)


def _spatial_query(sim, operation, dynamics=None):
  """Query current smooth state while preserving borrowed forward records."""
  from mujoco_metal.spatial_queries import DeviceSpatialQueries
  program = getattr(sim, "_spatial_queries", None)
  if (program is None or program.model is not sim.model
      or program.batch_size != sim.batch_size or program.device != sim.state._device):
    program = DeviceSpatialQueries(sim.model, sim.batch_size, sim.state._device)
    sim._spatial_queries = program
  if dynamics is not None:
    _validate_spatial_stage(sim, dynamics)
  with _inverse_query_workspaces(sim):
    stage = dynamics if dynamics is not None else _query_smooth(sim,
        sim.state._qpos, sim.state._qvel,
        getattr(sim.state, "_mpos", None), getattr(sim.state, "_mquat", None))
    return operation(program, stage)


def _validate_spatial_stage(sim, dynamics):
  """Check a caller-supplied smooth motion record at the query boundary.

  Spatial queries consume no mass layout. A stage from either dense or sparse
  dynamics is valid, with the same full position and motion ABI.
  """
  if not isinstance(dynamics, dict):
    raise TypeError("dynamics must be a smooth motion dictionary")
  from mujoco_metal.smooth_metal import validate_pose_dict
  validate_pose_dict(sim._mjmodel, dynamics.get("poses"), sim.batch_size,
                     sim.state._device, torch)
  b, nb, nv = sim.batch_size, int(sim._mjmodel.nbody), int(sim._mjmodel.nv)
  for name, shape in (("root_com", (b, nb, 3)), ("cvel", (b, nb, 6)),
                      ("cdof", (b, nv, 6)), ("cdof_dot", (b, nv, 6))):
    value = dynamics.get(name)
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
        or value.device.type != sim.state._device.type
        or (sim.state._device.index is not None
            and value.device.index != sim.state._device.index)
        or value.dtype != torch.float32
        or not value.is_contiguous() or not bool(torch.isfinite(value).all())):
      raise ValueError(f"dynamics.{name} has invalid native layout or values")


def _ray_tensor(sim, value, shape, name):
  if isinstance(value, torch.Tensor):
    if (value.dtype != torch.float32 or value.device.type != sim.state._device.type
        or tuple(value.shape) != tuple(shape) or not value.is_contiguous()):
      raise ValueError(f"{name} must be contiguous float32 on the simulation device with shape {shape}")
    return value
  array = np.asarray(value, dtype=np.float32)
  if tuple(array.shape) != tuple(shape):
    raise ValueError(f"{name} must have shape {shape}")
  return torch.as_tensor(array.copy(), device=sim.state._device)


def _ray_query(sim, origins, vectors, *, dynamics=None, **filters):
  from mujoco_metal.ray_queries import MetalRayQueries
  program = getattr(sim, "_ray_queries", None)
  if program is None or program.model is not sim.model or program.batch_size != sim.batch_size:
    program = MetalRayQueries(sim.model, sim.batch_size,
        memory_budget_bytes=getattr(sim.limits, "memory_budget_bytes", 1 << 30))
    sim._ray_queries = program
  if dynamics is not None:
    _validate_spatial_stage(sim, dynamics)
  with _inverse_query_workspaces(sim):
    stage = dynamics if dynamics is not None else _query_smooth(sim,
        sim.state._qpos, sim.state._qvel,
        getattr(sim.state, "_mpos", None), getattr(sim.state, "_mquat", None))
    return program.run_device(stage["poses"], origins, vectors, **filters)


def mj_ray(sim, pnt, vec, geomgroup=None, flg_static=True, bodyexclude=-1,
           *, dynamics=None):
  """Query one ray per world; return owned distance/geom ID/normal/status.

  Inputs have shape [B,3]. Directions need not be unit. Missing hits return
  distance and geom ID -1; status distinguishes invalid rays/SDF arithmetic
  from ordinary misses. Upstream model compilation and asset loading remain
  host operations. This query preserves prepared simulation stages/state.
  """
  origin = _ray_tensor(sim, pnt, (sim.batch_size, 3), "pnt")[:, None, :]
  vector = _ray_tensor(sim, vec, (sim.batch_size, 3), "vec")[:, None, :]
  result = _ray_query(sim, origin, vector, geomgroup=geomgroup,
      flg_static=flg_static, bodyexclude=bodyexclude, dynamics=dynamics)
  return {name: value[:, 0] for name, value in result.items()}


def mj_multiRay(sim, pnt, vec, geomgroup=None, flg_static=True,
                bodyexclude=-1, cutoff=None, *, dynamics=None):
  """Query [B,N,3] directions from [B,3] origins, with owned [B,N] hits.

  Optional cutoff follows pinned bounding-sphere geom elimination rather than
  clipping intersection distances. Directions with squared norm below mjMINVAL
  and nonfinite directions fail only their own ray (status 1). Unlike upstream output pointers, results are returned in
  a dictionary of device tensors and never overwrite caller storage.
  """
  shape = tuple(vec.shape) if hasattr(vec, "shape") else np.asarray(vec).shape
  if len(shape) != 3 or shape[0] != sim.batch_size or shape[2] != 3:
    raise ValueError("vec must have shape [B,N,3]")
  vector = _ray_tensor(sim, vec, shape, "vec")
  point = _ray_tensor(sim, pnt, (sim.batch_size, 3), "pnt")
  origin = point[:, None, :].expand(shape).contiguous()
  return _ray_query(sim, origin, vector, geomgroup=geomgroup,
      flg_static=flg_static, bodyexclude=bodyexclude, cutoff=cutoff,
      dynamics=dynamics, multiray=True)


def _ray_asset(sim, geomid, kind, pnt, vec, dynamics):
  from numbers import Integral
  if (isinstance(geomid, (bool, np.bool_)) or not isinstance(geomid, Integral)
      or not 0 <= geomid < sim.model.ngeom):
    raise ValueError("geomid must identify a compiled geom")
  if int(sim.model.geom_type[geomid]) != int(kind):
    raise ValueError("geomid has the wrong asset geometry type")
  origin = _ray_tensor(sim, pnt, (sim.batch_size, 3), "pnt")[:, None, :]
  vector = _ray_tensor(sim, vec, (sim.batch_size, 3), "vec")[:, None, :]
  result = _ray_query(sim, origin, vector, geomid=int(geomid), dynamics=dynamics)
  return {name: value[:, 0] for name, value in result.items()}


def mj_rayMesh(sim, geomid, pnt, vec, *, dynamics=None):
  """Owned per-world mesh hits using every compiled face, independent of masks."""
  return _ray_asset(sim, geomid, mujoco.mjtGeom.mjGEOM_MESH, pnt, vec, dynamics)


def mj_rayHfield(sim, geomid, pnt, vec, *, dynamics=None):
  """Owned per-world heightfield hits, independent of visual/group filters."""
  return _ray_asset(sim, geomid, mujoco.mjtGeom.mjGEOM_HFIELD, pnt, vec, dynamics)


def mju_rayGeom(sim, pos, mat, size, pnt, vec, geomtype):
  """Owned per-world primitive ray distances/normals/status from explicit poses.

  ``pos``, ``size``, ``pnt`` and ``vec`` are [B,3]; ``mat`` is row-major [B,3,3].
  One analytic geom type applies to the batch. No scene, forward-stage storage
  or accepted simulation state is evaluated or modified. Host inputs are staged
  at this boundary; intersections execute on Metal.
  """
  from mujoco_metal.ray_queries import MetalRayPrimitives
  program = getattr(sim, '_ray_primitives', None)
  if program is None:
    program = MetalRayPrimitives(sim.batch_size,
        memory_budget_bytes=getattr(sim.limits, 'memory_budget_bytes', 1 << 30))
    sim._ray_primitives = program
  b = sim.batch_size
  return program.run_device(_ray_tensor(sim,pos,(b,3),'pos'),
      _ray_tensor(sim,mat,(b,3,3),'mat'), _ray_tensor(sim,size,(b,3),'size'),
      _ray_tensor(sim,pnt,(b,3),'pnt'), _ray_tensor(sim,vec,(b,3),'vec'), geomtype)


def mju_raySkin(sim, face, vert, pnt, vec):
  """Ray intersection with explicit skin triangles and local nearest vertex.

  ``face`` is immutable host integer[nface,3] topology; ``vert`` is current
  float32[B,nvert,3] world positions. Counts are derived from these shapes.
  Device outputs are distance/vertid/status plus an additional face normal;
  upstream's pointer outputs are replaced by owned tensors. No simulation
  state, compiled skin animation or OpenGL/Metal rendering is evaluated.
  """
  from mujoco_metal.ray_queries import lower_ray_skin, MetalRaySurface
  shape = tuple(vert.shape) if hasattr(vert,'shape') else np.asarray(vert).shape
  if len(shape) != 3 or shape[0] != sim.batch_size or shape[2] != 3:
    raise ValueError('vert must have shape [B,nvert,3]')
  surface = lower_ray_skin(shape[1],face)
  cached = getattr(sim,'_ray_skin',None)
  if (cached is None or cached.surface.nvert != surface.nvert
      or not np.array_equal(cached.surface.faces,surface.faces)):
    cached = MetalRaySurface(surface,sim.batch_size,
        memory_budget_bytes=getattr(sim.limits,'memory_budget_bytes',1 << 30))
    sim._ray_skin = cached
  return cached.run_device(_ray_tensor(sim,vert,shape,'vert'),
      _ray_tensor(sim,pnt,(sim.batch_size,3),'pnt'),
      _ray_tensor(sim,vec,(sim.batch_size,3),'vec'))


def mj_rayFlex(sim, flex_layer, flg_vert, flg_edge, flg_face, flg_skin,
               flexid, pnt, vec, *, dynamics=None):
  """Query compiled flex geometry at current state without advancing physics.

  Source flags/layers control vertices, capsule edges and triangle/tetrahedron
  faces. IDs are local to the selected flex. Current positions, intersections,
  nearest-hit reduction and normals execute on device; this boundary preserves
  prepared stages and accepted state. It is not a flex collision detector.
  """
  from mujoco_metal.ray_queries import lower_ray_flex, MetalRaySurface
  surface = lower_ray_flex(sim.model,flexid)
  if getattr(sim,'_flex',None) is None:
    raise ValueError('simulation has no native flex position pipeline')
  programs = getattr(sim,'_ray_flex',{})
  if int(flexid) not in programs:
    programs[int(flexid)] = MetalRaySurface(surface,sim.batch_size,
        memory_budget_bytes=getattr(sim.limits,'memory_budget_bytes',1 << 30))
    sim._ray_flex = programs
  program = programs[int(flexid)]
  if dynamics is not None:
    _validate_spatial_stage(sim,dynamics)
  origin = _ray_tensor(sim,pnt,(sim.batch_size,3),'pnt')
  vector = _ray_tensor(sim,vec,(sim.batch_size,3),'vec')
  with _inverse_query_workspaces(sim):
    stage = dynamics if dynamics is not None else _query_smooth(sim,
        sim.state._qpos,sim.state._qvel,
        getattr(sim.state,'_mpos',None),getattr(sim.state,'_mquat',None))
    sim._flex.update_kinematics(dict(stage['poses'],root_com=stage['root_com']),
                               stage['cvel'])
    va = int(sim.model.flex_vertadr[flexid])
    vertices = sim._flex.flexvert_xpos[:,va:va+surface.nvert,:].contiguous()
    return program.run_device(vertices,origin,vector,flex_layer=flex_layer,
        flg_vert=flg_vert,flg_edge=flg_edge,flg_face=flg_face,flg_skin=flg_skin)


def _position_query_program(sim):
  from mujoco_metal.position_queries import MetalPositionQueries, lower_position_topology
  program = getattr(sim, '_position_queries', None)
  topology = lower_position_topology(sim.model)
  if (program is None or program.batch_size != sim.batch_size
      or program.nq != topology.nq or program.nv != topology.nv
      or not np.array_equal(program.topology.joints, topology.joints)):
    program = MetalPositionQueries(sim.model, sim.batch_size,
        memory_budget_bytes=getattr(sim.limits, 'memory_budget_bytes', 1 << 30))
    sim._position_queries = program
  return program


def mj_integratePos(sim, qpos, qvel, dt):
  """Owned integrated [B,nq] position and status; does not update sim state.

  Finite dt may be negative or zero. Host inputs are staged explicitly. Joint
  quaternion normalization and right multiplication follow pinned MuJoCo.
  """
  program = _position_query_program(sim)
  return program.integrate(_ray_tensor(sim, qpos, (sim.batch_size, sim.model.nq), 'qpos'),
      _ray_tensor(sim, qvel, (sim.batch_size, sim.model.nv), 'qvel'), dt)


def mj_differentiatePos(sim, qpos1, qpos2, dt):
  """Owned [B,nv] tangent velocity between two [B,nq] positions and status.

  Quaternion subtraction uses the shortest signed rotation in local joint
  coordinates. A finite nonzero dt is required; accepted state is unchanged.
  """
  program = _position_query_program(sim)
  return program.differentiate(
      _ray_tensor(sim, qpos1, (sim.batch_size, sim.model.nq), 'qpos1'),
      _ray_tensor(sim, qpos2, (sim.batch_size, sim.model.nq), 'qpos2'), dt)


def mj_normalizeQuat(sim, qpos):
  """Owned [B,nq] positions with joint quaternions normalized, plus status.

  Scalar/translation fields are retained and zero quaternions become identity;
  caller inputs and simulation state are never overwritten.
  """
  return _position_query_program(sim).normalize(
      _ray_tensor(sim, qpos, (sim.batch_size, sim.model.nq), 'qpos'))


def mjd_subQuat(qa, qb, *, memory_budget_bytes=1 << 30):
  """Owned Da/Db[B,3,3] in local tangent coordinates, plus status[B].

  Inputs must be contiguous float32 MPS quaternions [B,4]. The matrices are
  derivatives with respect to three-dimensional rotational perturbations,
  following the pinned function rather than four quaternion components.
  """
  from mujoco_metal.quaternion_derivatives import MetalQuaternionDerivatives
  _quaternion_derivative_input(qa, 4, 'qa')
  _quaternion_derivative_input(qb, 4, 'qb', batch=qa.shape[0])
  program = MetalQuaternionDerivatives(qa.shape[0],
                                       memory_budget_bytes=memory_budget_bytes)
  return program.sub_quat(qa, qb)


def mjd_quatIntegrate(vel, scale, *, memory_budget_bytes=1 << 30):
  """Owned Dquat/Dvel[B,3,3], Dscale[B,3] and status[B] on MPS.

  vel is contiguous float32[B,3]. Pinned Dvel is the derivative with respect
  to scaled velocity; multiply it by scale to differentiate unscaled vel.
  These analytical utilities do not step physics or mutate any inputs.
  """
  from mujoco_metal.quaternion_derivatives import MetalQuaternionDerivatives
  _quaternion_derivative_input(vel, 3, 'vel')
  from mujoco_metal.quaternion_derivatives import validate_scale
  validate_scale(scale)
  program = MetalQuaternionDerivatives(vel.shape[0],
                                       memory_budget_bytes=memory_budget_bytes)
  return program.quat_integrate(vel, scale)


def _quaternion_derivative_input(value, width, name, *, batch=None):
  """Reject malformed public inputs before compiling or allocating on MPS."""
  if (torch is None or not isinstance(value, torch.Tensor)
      or value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != width
      or value.dtype != torch.float32 or value.device.type != 'mps'
      or not value.is_contiguous()
      or (batch is not None and value.shape[0] != batch)):
    raise ValueError(f'{name} must be contiguous MPS float32 [B,{width}]')


def _owned_stage_query(sim, operation, dynamics=None):
  """Admit conservative additional storage before an owned stage query.

  Resident simulation buffers and earlier retained outputs are separate. This
  bound includes the rollback snapshot, lazy spatial metadata and the maximum
  simultaneous output/scratch storage of the three Cartesian stage queries.
  """
  from mujoco_metal.finite_difference import estimate_transaction_bytes
  model, b = sim.model, int(sim.batch_size)
  nb, nv, ng, ns, nj = (int(getattr(model, name)) for name in
                        ("nbody", "nv", "ngeom", "nsite", "njnt"))
  counts = (nb * max(nv, 1), b * (128*nb + 24*nv + 48*ng + 48*ns + 12*nj),
            12*nb + 3 + 24*int(model.ncam))
  if any(count < 0 or count > (1 << 31)-1 for count in counts):
    raise ValueError("Cartesian stage query exceeds int32 address capacity")
  required = estimate_transaction_bytes(sim) + 4*sum(counts)
  budget = int(getattr(getattr(sim, "limits", None),
                       "memory_budget_bytes", 1 << 30))
  if required > budget:
    raise ValueError(f"Cartesian stage query needs {required} bytes; budget is {budget}")
  return _spatial_query(sim, operation, dynamics)


def mj_kinematics(sim, *, dynamics=None):
  """Owned current-state Cartesian body/inertia/geom/site/joint fields.

  Positions are [B,N,3], rotations [B,N,3,3], and xquat [B,nbody,4].
  This evaluates all bodies without updating sleep caches or simulation state.
  XML compilation and model metadata preparation remain host operations.
  """
  return _owned_stage_query(sim, lambda program, stage: program.kinematics(stage), dynamics)


def mj_comPos(sim, *, dynamics=None):
  """Owned subtree_com[B,nbody,3], cinert[B,nbody,10], cdof[B,nv,6].

  Inertia and motion use each root subtree COM as in pinned mj_comPos. The
  world inertia is zero and a zero-mass subtree falls back to its inertial
  position. This full-state query does not mutate persistent sleep caches.
  """
  return _owned_stage_query(sim, lambda program, stage:
      program.center_of_mass_position(stage), dynamics)


def mj_comVel(sim, *, dynamics=None):
  """Owned cvel[B,nbody,6] and cdof_dot[B,nv,6] in rot:lin convention.

  The complete native motion stage is evaluated without stepping or updating
  persistent sleep caches. A supplied validated smooth stage is also accepted.
  """
  return _owned_stage_query(sim, lambda program, stage:
      program.center_of_mass_velocity(stage), dynamics)


def mj_local2Global(sim, pos, quat, body, sameframe=0, *, dynamics=None):
  """Return owned world (position [B,3], rotation [B,3,3]) without stepping.

  All five pinned mjtSameFrame values are supported. A None position or
  quaternion omits that output. Batched local inputs are not broadcast or
  normalized; aligned outputs ignore their values as in the pinned source.
  """
  pos = None if pos is None else _stage_tensor(sim, pos, (sim.batch_size, 3), "pos")
  quat = None if quat is None else _stage_tensor(sim, quat, (sim.batch_size, 4), "quat")
  return _spatial_query(sim, lambda program, stage:
      program.local_to_global(stage, pos, quat, body, sameframe), dynamics)


def mj_jac(sim, point, body, *, dynamics=None):
  """Owned world point translation/rotation Jacobians [batch,3,nv]."""
  point = _stage_tensor(sim, point, (sim.batch_size, 3), "point")
  return _spatial_query(sim, lambda program, stage: program.jac(stage, point, body), dynamics)


def mj_jacDot(sim, point, body, *, dynamics=None):
  """Owned world time derivatives of the point/rotation Jacobians."""
  point = _stage_tensor(sim, point, (sim.batch_size, 3), "point")
  return _spatial_query(sim, lambda program, stage: program.jac_dot(stage, point, body), dynamics)


def mj_jacPointAxis(sim, point, axis, body, *, dynamics=None):
  """Owned world point and axis Jacobians [batch,3,nv].

  ``axis`` is a world vector; its magnitude is preserved. The axis result
  measures the vector's derivative, rather than the angular velocity itself.
  """
  point = _stage_tensor(sim, point, (sim.batch_size, 3), "point")
  axis = _stage_tensor(sim, axis, (sim.batch_size, 3), "axis")
  return _spatial_query(sim, lambda program, stage:
      program.jac_point_axis(stage, point, axis, body), dynamics)


def _object_jac(sim, objtype, objid, dynamics):
  def query(program, stage):
    body, point, _rotation = program.object_frame(stage, objtype, objid)
    return program.jac(stage, point, body)
  return _spatial_query(sim, query, dynamics)


def mj_jacBody(sim, body, *, dynamics=None):
  """World Jacobians at the regular body origin (not its inertial COM)."""
  return _object_jac(sim, mujoco.mjtObj.mjOBJ_XBODY, body, dynamics)


def mj_jacBodyCom(sim, body, *, dynamics=None):
  return _object_jac(sim, mujoco.mjtObj.mjOBJ_BODY, body, dynamics)


def mj_jacGeom(sim, geom, *, dynamics=None):
  return _object_jac(sim, mujoco.mjtObj.mjOBJ_GEOM, geom, dynamics)


def mj_jacSite(sim, site, *, dynamics=None):
  return _object_jac(sim, mujoco.mjtObj.mjOBJ_SITE, site, dynamics)


def mj_jacSubtreeCom(sim, body, *, dynamics=None):
  return _spatial_query(sim, lambda program, stage: program.jac_subtree_com(stage, body), dynamics)


def mj_angmomMat(sim, body, *, dynamics=None):
  """Owned [batch,3,nv] angular-momentum matrix about a body's subtree COM.

  Rows use world axes; multiplying by generalized velocity returns the
  subtree's angular momentum in kg m²/s. The optional smooth stage has the
  same validated device layout as other spatial queries. No step is taken.
  """
  return _spatial_query(sim, lambda program, stage:
      program.angmom_matrix(stage, body), dynamics)


def mj_subtreeVel(sim, *, dynamics=None):
  """Owned current-state subtree motion, each [batch,nbody,3].

  Returns ``subtree_linvel`` in m/s and ``subtree_angmom`` in kg m²/s,
  both in world axes and referenced to each subtree's center of mass.
  Evaluates all bodies without changing stored sensors, sleep scheduling
  or upstream's ``flg_subtreevel`` cache; it does not advance physics.
  """
  return _spatial_query(sim, lambda program, stage:
      program.subtree_velocity(stage), dynamics)


def mj_rne(sim, flg_acc=1, *, dynamics=None, qvel=None, qacc=None):
  """Owned [batch,nv] full-state recursive Newton–Euler generalized force.

  Includes rigid-body inertial, Coriolis and gravity terms, with the pinned
  gravity-disable convention. flg_acc=0 omits qacc. Joint/tendon armature,
  passive, actuation, applied and constraint forces are excluded, as in
  upstream mj_rne. All bodies are evaluated without mutating sleep caches.
  """
  if not isinstance(flg_acc, (bool, int, np.integer)) or int(flg_acc) not in (0, 1):
    raise ValueError("flg_acc must be 0 or 1")
  shape = (sim.batch_size, int(sim.model.nv))
  velocity = _stage_tensor(sim, sim.state._qvel if qvel is None else qvel,
                           shape, "qvel")
  acceleration = _stage_tensor(sim, sim.state._qacc if qacc is None else qacc,
                               shape, "qacc")
  return _spatial_query(sim, lambda program, stage:
      program.recursive_newton_euler(stage, velocity, acceleration, bool(flg_acc)),
      dynamics)


def _local_flag(value):
  if not isinstance(value, (bool, int, np.integer)) or int(value) not in (0, 1):
    raise ValueError("flg_local must be 0 or 1")
  return bool(value)


def mj_rnePostConstraint(sim, *, forward_result=None, qacc=None):
  """Owned cacc/cfrc_ext/cfrc_int/subtree_com/qacc/status from native RNE.

  A coherent complete forward bundle may be supplied; otherwise the native
  forward stages are evaluated inside a rollback transaction. A qacc override
  uses its exact float32 values rather than a solver residual side-channel.
  Applied body wrenches, geom contacts and connect/weld equalities contribute
  to cfrc_ext. Pinned source skips flex contacts and other equality families
  there; their dynamical effects remain represented in qacc/internal forces.
  Full forward-solver composition qualification remains in progress.
  """
  from mujoco_metal.rne_post_constraint import mj_rnePostConstraint as query
  return query(sim, forward_result=forward_result, qacc=qacc)


def mj_transmission(sim):
  """Owned native actuator lengths and dense transmission moment rows.

  Results have shapes [B,nu], [B,nu,nv] and per-world status [B]. This is a
  current-position query; actuator velocities belong to the VEL stage.
  """
  from mujoco_metal.transmission_query import mj_transmission as query
  return query(sim)


def mj_objectVelocity(sim, objtype, objid, flg_local=0, *, dynamics=None):
  """Owned [batch,6] rot:lin object-centered velocity, world/local axes."""
  local = _local_flag(flg_local)
  return _spatial_query(sim, lambda program, stage:
      program.object_velocity(stage, objtype, objid, local), dynamics)


def mj_objectAcceleration(sim, objtype, objid, flg_local=0, *, dynamics=None, qvel=None, qacc=None):
  """Pinned object acceleration from current qvel and supplied/owned qacc.

  Includes RNE's gravity convention and rotating-frame Coriolis correction.
  This evaluates a query; it does not perform a forward constraint solve.
  """
  local = _local_flag(flg_local)
  qa = (_stage_tensor(sim, qacc, (sim.batch_size, sim.model.nv), "qacc")
        if qacc is not None else sim.state._qacc)
  qv = (_stage_tensor(sim, qvel, (sim.batch_size, sim.model.nv), "qvel")
        if qvel is not None else sim.state._qvel)
  return _spatial_query(sim, lambda program, stage: program.object_acceleration(
      stage, qv, qa, objtype, objid, local), dynamics)


def mj_applyFT(sim, force, torque, point, body, *, dynamics=None):
  """Owned generalized wrench contribution, without mutating held inputs."""
  def tensor(value, name):
    if value is None:
      return torch.zeros((sim.batch_size, 3), dtype=torch.float32, device=sim.state._device)
    return _stage_tensor(sim, value, (sim.batch_size, 3), name)
  force, torque = tensor(force, "force"), tensor(torque, "torque")
  point = _stage_tensor(sim, point, (sim.batch_size, 3), "point")
  return _spatial_query(sim, lambda program, stage:
      program.apply_ft(stage, force, torque, point, body), dynamics)


# -------------------------------------------------------------------------
# Native State Selector APIs (mj_getState / mj_setState)
# -------------------------------------------------------------------------

class StateSpec:
  """Pinned MuJoCo 3.10 mjtState bitmasks."""
  TIME, QPOS, QVEL, ACT = 1, 2, 4, 8
  HISTORY, WARMSTART, CTRL = 16, 32, 64
  QFRC_APPLIED, XFRC_APPLIED, EQ_ACTIVE = 128, 256, 512
  MOCAP_POS, MOCAP_QUAT, USERDATA, PLUGIN = 1024, 2048, 4096, 8192
  PHYSICS = QPOS | QVEL | ACT | HISTORY
  USER = CTRL | QFRC_APPLIED | XFRC_APPLIED | EQ_ACTIVE | MOCAP_POS | MOCAP_QUAT | USERDATA
  FULLPHYSICS = TIME | PHYSICS | PLUGIN
  INTEGRATION = TIME | PHYSICS | WARMSTART | CTRL | QFRC_APPLIED | XFRC_APPLIED | EQ_ACTIVE | MOCAP_POS | MOCAP_QUAT | USERDATA | PLUGIN
  ALL = INTEGRATION


def mj_stateSize(model, sig):
  """Pinned vector width from host model metadata and an upstream mjtState mask.

  This describes upstream flat state vectors. mj_getState/mj_setState expose
  the separately documented backend dictionary interface and StateSpec mask.
  """
  from mujoco_metal.state_vector import state_vector_layout
  return state_vector_layout(model, sig)[1]


def mj_extractState(model, source, srcsig, dstsig):
  """Owned batched device subvector with the exact upstream mjtState ordering.

  srcsig/dstsig are upstream mjtState masks, distinct from backend StateSpec.
  The destination mask must be a subset; source has shape [batch,stateSize].
  """
  from mujoco_metal.state_vector import extract_state_vector
  return extract_state_vector(model, source, srcsig, dstsig)


_STATE_FIELDS = {
    StateSpec.TIME: ("time", "_time", (1,)),
    StateSpec.QPOS: ("qpos", "_qpos", None),
    StateSpec.QVEL: ("qvel", "_qvel", None),
    StateSpec.ACT: ("act", "_act", None),
    StateSpec.HISTORY: ("history", "_history", None),
    StateSpec.WARMSTART: ("warmstart", "_qacc_warmstart", None),
    StateSpec.CTRL: ("ctrl", "_control", None),
    StateSpec.QFRC_APPLIED: ("qfrc_applied", "_applied_force", None),
    StateSpec.XFRC_APPLIED: ("xfrc_applied", "_body_wrench", None),
    StateSpec.EQ_ACTIVE: ("eq_active", "_eq_active", None),
    StateSpec.MOCAP_POS: ("mocap_pos", "_mpos", None),
    StateSpec.MOCAP_QUAT: ("mocap_quat", "_mquat", None),
    StateSpec.USERDATA: ("userdata", "_userdata", None),
    StateSpec.PLUGIN: ("plugin_state", "_plugin_state", None),
}


def _selected_ids(sim, env_ids):
  b = int(sim.batch_size)
  if env_ids is None:
    return None, b
  raw = np.asarray(env_ids)
  if raw.ndim != 1:
    raise TypeError("env_ids must be a one-dimensional integer selection")
  if raw.size == 0:
    raise ValueError("env_ids must select at least one world")
  if raw.dtype.kind not in "iu":
    raise TypeError("env_ids must be a one-dimensional integer selection")
  ids = raw.astype(np.int64, copy=True)
  if np.any(ids < 0) or np.any(ids >= b):
    raise IndexError(f"env_ids must be in [0, {b})")
  if len(np.unique(ids)) != len(ids):
    raise ValueError("env_ids must not contain duplicates")
  return ids, len(ids)


def _validate_history(model, history):
  """Check the pinned [user,cursor,times,values] history record structure."""
  torch = __import__("torch")
  buffers = []
  for nvalues, dim, adr in (
      (np.asarray(model.actuator_history).reshape(-1, 2)[:, 0],
       np.ones(int(model.nu), dtype=np.int32),
       np.asarray(model.actuator_historyadr).reshape(-1)),
      (np.asarray(model.sensor_history).reshape(-1, 2)[:, 0],
       np.asarray(model.sensor_dim).reshape(-1),
       np.asarray(model.sensor_historyadr).reshape(-1)),
  ):
    for i, count in enumerate(np.asarray(nvalues).reshape(-1).tolist()):
      n = int(count)
      if n > 0:
        buffers.append((int(adr[i]), n))
  for start, n in buffers:
    cursor = history[:, start + 1]
    if not bool(((cursor >= 0) & (cursor < n) & (cursor == torch.round(cursor))).all()):
      raise ValueError("history cursor is outside the model buffer")
    logical = torch.arange(n, device=history.device, dtype=torch.float32)[None, :]
    idx = torch.remainder(cursor.to(torch.long)[:, None] + 1 + logical.to(torch.long), n)
    stamps = history[:, start + 2:start + 2 + n].gather(1, idx)
    if n > 1 and not bool((stamps[:, 1:] > stamps[:, :-1]).all()):
      raise ValueError("history timestamps must increase in logical order")


def _tensor_state(sim, field):
  _, attr, _ = field
  value = getattr(sim.state, attr, None) if attr.startswith("_") and attr not in (
      "_control", "_applied_force", "_body_wrench") else getattr(sim, attr, None)
  if value is None:
    # Zero-dimensional mjtState components still have an explicit empty view.
    tails = {"_act": (int(getattr(sim._mjmodel, "na", 0)),), "_history": (sim.state._nhistory,),
             "_control": (int(getattr(sim._mjmodel, "nu", 0)),),
             "_applied_force": (int(getattr(sim._mjmodel, "nv", 0)),),
             "_body_wrench": (int(getattr(sim._mjmodel, "nbody", 0)), 6),
             "_eq_active": (int(getattr(sim._mjmodel, "neq", 0)),),
             "_mpos": (sim.state._nmocap, 3), "_mquat": (sim.state._nmocap, 4),
             "_userdata": (sim.state._nuserdata,), "_plugin_state": (sim.state._npluginstate,)}
    return sim.state._torch.empty((sim.batch_size, *tails.get(attr, (0,))),
                                  dtype=sim.state._torch.float32, device=sim.state._device)
  return value


def mj_getState(sim, spec: int = StateSpec.ALL, env_ids=None):
  """Return copies of the selected pinned state groups, preserving device residency."""
  if isinstance(spec, bool) or not isinstance(spec, (int, np.integer)) or int(spec) < 0 or int(spec) & ~StateSpec.ALL:
    raise ValueError("spec contains bits outside pinned mjtState")
  ids, _ = _selected_ids(sim, env_ids)
  out = {}
  for bit, (name, _, _) in _STATE_FIELDS.items():
    if int(spec) & bit:
      value = _tensor_state(sim, _STATE_FIELDS[bit])
      out[name] = value.clone() if ids is None else value[torch.as_tensor(ids, device=value.device)].clone()
  return out


def mj_setState(sim, values: dict[str, Any], env_ids=None):
  """Validate all fields, then atomically write selected simulation state rows."""
  if not isinstance(values, dict):
    raise TypeError(f"values must be a dict, got {type(values)}")
  aliases = {"qacc_warmstart": "warmstart", "plugin": "plugin_state"}
  normalized = {}
  for key, value in values.items():
    name = aliases.get(key, key)
    if name in normalized:
      raise ValueError(f"duplicate state field alias for {name}")
    normalized[name] = value
  values = normalized
  accepted = {field[0] for field in _STATE_FIELDS.values()} | {"qacc"}
  if set(values) - accepted:
    raise ValueError(f"unknown state keys: {sorted(set(values) - accepted)}")
  ids, count = _selected_ids(sim, env_ids)
  state, model = sim.state, sim._mjmodel
  device = state._device
  staged = {}
  for name, value in values.items():
    if value is None:
      continue
    attr = "_qacc" if name == "qacc" else next(a for n, a, _ in _STATE_FIELDS.values() if n == name)
    live = getattr(state, attr, None) if attr.startswith("_") and attr not in ("_control", "_applied_force", "_body_wrench") else getattr(sim, attr, None)
    if live is None:
      dims = {"act": int(model.na), "eq_active": int(model.neq),
              "mocap_pos": (int(model.nmocap), 3),
              "mocap_quat": (int(model.nmocap), 4),
              "history": state._nhistory, "userdata": state._nuserdata,
              "plugin_state": state._npluginstate}
      dim = dims.get(name, 0)
      if isinstance(dim, tuple):
        tail = dim
      else:
        tail = (dim,)
      live = torch.empty((sim.batch_size, *tail), dtype=torch.float32, device=device)
    shape = tuple(live.shape)
    if len(shape) == 1:
      shape = (sim.batch_size,)
    target_shape = shape[1:]
    if isinstance(value, torch.Tensor):
      if (value.device.type != device.type or
          (device.index is not None and value.device.index != device.index)):
        raise ValueError(f"{name} tensor must be on {device}")
      raw = value
      if raw.dtype not in (torch.float32, torch.float64, torch.int32, torch.int64, torch.bool):
        raise TypeError(f"{name} tensor must have a real numeric dtype")
      raw = raw.to(dtype=torch.float32)
      if not bool(torch.isfinite(raw).all()):
        raise ValueError(f"{name} must be finite")
      t = raw
    else:
      arr = np.asarray(value)
      if arr.dtype.kind not in ("fiu" if name != "eq_active" else "fiub"):
        raise TypeError(f"{name} must be a real numeric array")
      with np.errstate(over="ignore", invalid="ignore"):
        arr = np.asarray(arr, dtype=np.float32, order="C")
      if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and float32-representable")
      t = torch.as_tensor(arr, dtype=torch.float32, device=device)
    # Scalar time has a useful broadcast; all other groups retain MuJoCo shape.
    if name == "time" and t.numel() == 1:
      t = t.reshape(1).expand(count)
    elif tuple(t.shape) == (sim.batch_size, *target_shape) and ids is not None:
      t = t[torch.as_tensor(ids, device=device)]
    elif tuple(t.shape) == target_shape and name != "time" and count == 1:
      t = t.unsqueeze(0)
    elif tuple(t.shape) != (count, *target_shape):
      raise ValueError(f"{name} shape mismatch: expected {(count, *target_shape)} or full batch, got {tuple(t.shape)}")
    if t.dtype != torch.float32:
      t = t.to(torch.float32)
    if not bool(torch.isfinite(t).all()):
      raise ValueError(f"{name} must be finite and float32-representable")
    if name == "eq_active":
      if not bool(((t == 0) | (t == 1)).all()):
        raise ValueError("eq_active values must be 0 or 1")
      t = t.to(torch.int32)
    if name == "qpos":
      q = t.reshape(count, int(model.nq))
      for jid in range(int(model.njnt)):
        jt = int(model.jnt_type[jid])
        adr = int(model.jnt_qposadr[jid])
        if jt == int(mujoco.mjtJoint.mjJNT_FREE):
          quat = q[:, adr + 3:adr + 7]
        elif jt == int(mujoco.mjtJoint.mjJNT_BALL):
          quat = q[:, adr:adr + 4]
        else:
          continue
        if not bool((torch.abs(torch.linalg.vector_norm(quat, dim=-1) - 1) <= 1e-5).all()):
          raise ValueError("qpos contains a non-unit joint quaternion")
    if name == "mocap_quat":
      q = t.reshape(count, -1, 4)
      if not bool((torch.abs(torch.linalg.vector_norm(q, dim=-1) - 1) <= 1e-5).all()):
        raise ValueError("mocap_quat values must be unit quaternions")
    if name == "history":
      _validate_history(model, t)
    staged[name] = (attr, t)

  # Stage delegated warmstart inputs before any writes; this is the most
  # failure-prone external operation in the old implementation.
  ids_t = None if ids is None else torch.as_tensor(ids, dtype=torch.long, device=device)
  with torch.no_grad():
    for name, (attr, value) in staged.items():
      live = state._qacc if attr == "_qacc" else (getattr(state, attr) if attr not in ("_control", "_applied_force", "_body_wrench") else getattr(sim, attr))
      if value.numel() == 0:
        continue
      if live.ndim == 1 and attr != "_time":
        live = live.reshape(sim.batch_size, -1)
      target = live if ids_t is None else live.index_select(0, ids_t).clone()
      v = value.reshape(target.shape)
      target.copy_(v)
      if ids_t is None:
        live.copy_(target)
      else:
        live.index_copy_(0, ids_t, target)
  if staged:
    if "history" in staged:
      sim._sync_delay_from_history(None if ids_t is None else ids_t)
    sim._invalidate_after_state_write(staged)


def mjd_stepFD(sim, eps=1e-6, flg_centered=False):
  """Owned native finite differences of one step, transposed by input.

  Returns ``DyDq/DyDv/DyDa/DyDu`` with shapes ``[B,nv|na|nu,2*nv+na]``,
  ``DsDq/DsDv/DsDa/DsDu`` with shapes ``[B,nv|na|nu,nsensordata]``, and
  ``status[B]``. Device stepping and differencing remain on the simulation's
  MPS device. Pinned MuJoCo's RK4 behavior and control-bound one-sided stencil
  are preserved. Models with actuator history are rejected by the source API.
  """
  from mujoco_metal.finite_difference import mjd_stepFD as implementation
  return implementation(sim, eps, flg_centered)


def mjd_transitionFD(sim, eps=1e-6, flg_centered=False):
  """Owned control-theory transition, control, observation and sensor matrices.

  Returns ``A[B,nx,nx]``, ``B[B,nx,nu]``, ``C[B,nsensordata,nx]``,
  ``D[B,nsensordata,nu]`` and ``status[B]``. Pinned source gates continue to
  reject RK4 and actuator history.
  """
  from mujoco_metal.finite_difference import mjd_transitionFD as implementation
  return implementation(sim, eps, flg_centered)


def mjd_inverseFD(sim, eps=1e-6, flg_actuation=False):
  """Owned native inverse-dynamics finite differences.

  Returns transposed ``DfDq/DfDv/DfDa[B,nv,nv]``, sensor derivatives
  ``DsDq/DsDv/DsDa[B,nv,nsensordata]``, packed lower sparse-mass derivative
  ``DmDq[B,nv,nM]`` and ``status[B]``. RK4 and noslip retain their pinned
  MuJoCo 3.10 source restrictions.
  """
  from mujoco_metal.finite_difference import mjd_inverseFD as implementation
  return implementation(sim, eps, flg_actuation)
