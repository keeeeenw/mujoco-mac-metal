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

Unlike host_utils.py, every function here executes NATIVELY ON DEVICE (MPS/Metal),
preserving full batching, device memory residency, and zero CPU fallbacks.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
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
           "_flex", "_coupled_constraints")
  seen, saved, memo = set(), [], {}
  def collect(program):
    if program is None or id(program) in seen or not hasattr(program, "__dict__"):
      return
    seen.add(id(program))
    attributes = dict(vars(program))
    storage = {name: _capture_query_storage(value, memo)
               for name, value in attributes.items() if _contains_device_tensor(value)}
    saved.append((program, attributes, storage))
    for value in attributes.values():
      # Programs own subordinate FK, broadphase, compaction and flex stages;
      # arbitrary third-party objects and Torch internals are not traversed.
      if type(value).__module__.startswith("mujoco_metal."):
        collect(value)
  for name in roots:
    collect(getattr(sim, name, None))
  spatial = {name: getattr(sim, name) for name in ("_spatial_kin", "_spatial_cache_key")
             if hasattr(sim, name)}
  try:
    yield
  finally:
    for program, attributes, storage in reversed(saved):
      # Restore the schema and original storage owners before copying values;
      # a failed query may have replaced a buffer or published an extra map.
      vars(program).clear()
      vars(program).update(attributes)
      for name, value in storage.items():
        setattr(program, name, _restore_query_storage(value))
    for name in ("_spatial_kin", "_spatial_cache_key"):
      if name in spatial:
        setattr(sim, name, spatial[name])
      elif hasattr(sim, name):
        delattr(sim, name)


def mj_inverse(sim, qpos=None, qvel=None, qacc=None,
               mocap_pos=None, mocap_quat=None) -> torch.Tensor:
  """NATIVE GPU inverse dynamics: compute qfrc_inverse on MPS.
  
  qfrc_inverse = M(qpos) * qacc + qfrc_bias(qpos, qvel) - qfrc_passive(qpos, qvel)
  Operates natively on device tensors without host roundtrips.
  Returns an owned MPS tensor [batch, nv]. Temporary stage workspaces and
  cached forward records are preserved on success and failure.
  """
  with _inverse_query_workspaces(sim):
    return _inverse_impl(sim, qpos, qvel, qacc, mocap_pos, mocap_quat)


def _inverse_impl(sim, qpos, qvel, qacc, mocap_pos, mocap_quat):
  """NATIVE GPU inverse dynamics: compute qfrc_inverse on MPS.
  
  qfrc_inverse = M(qpos) * qacc + qfrc_bias(qpos, qvel) - qfrc_passive(qpos, qvel)
  Operates natively on device tensors without host roundtrips.
  Returns borrowed/owned MPS tensor [batch, nv].
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
  dynamics = sim._smooth.run_device(qp, qv, mpos, mquat)
  M = dynamics["mass_matrix"]       # [batch, nv, nv]
  bias = dynamics["qfrc_bias"]      # [batch, nv]
  tendon_force = None
  if getattr(sim, "_tendons", None) is not None:
    tendon_force, _, tendon_armature = sim._tendons.run_device(qp, qv)
    M = M + tendon_armature
  spatial_J = spatial_length = None
  if getattr(sim, "_spatial_tendons", None) is not None:
    kin = sim._spatial_jacobian(qv, dynamics["poses"])
    spatial_J = kin
    skin = sim._spatial_kin
    sf, _, sa = sim._spatial_tendons.run_forces(skin)
    M = M + sa.reshape(M.shape)
    tendon_force = sf.reshape((sim.batch_size, sim._mjmodel.nv)) if tendon_force is None else tendon_force + sf.reshape((sim.batch_size, sim._mjmodel.nv))
    sbias, _ = sim._spatial_tendons.run_armature_bias(
        skin, qv, dynamics["poses"], dynamics.get("cvel"),
        dynamics.get("root_com"), dynamics.get("cdof"),
        dynamics.get("cdof_dot"))
    bias = bias + sbias.reshape(bias.shape)

  # 2. Compute M * qacc on MPS
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
        qp, qv, dynamics["poses"], dynamics.get("cvel", None)
    )
    qfrc_inv = qfrc_inv - flex_qfrc

  # 8. Subtract constraint forces (mj_invConstraint)
  if getattr(sim, "_coupled_constraints", None) is not None:
    cc = sim._coupled_constraints
    from mujoco_metal.simulation import _clone_system_dict, _restore_system_buffers
    saved_workspace = _clone_system_dict(cc._workspace)
    saved_last = getattr(sim, "_last_coupled", None)
    saved_last_generation = getattr(sim, "_last_coupled_generation", None)
    eq_act = getattr(state, "_eq_active", None)
    try:
      _ten_J, _ten_L = sim._spatial_for_coupled(qv, dynamics["poses"]) if hasattr(sim, "_spatial_for_coupled") else (None, None)
      constraint_result = cc.run_device(
          dynamics["poses"], M, -bias, qp, qv,
          eq_active=eq_act, cvel=dynamics.get("cvel", None),
          tendon_J_spatial=_ten_J, tendon_length_spatial=_ten_L,
          flex=getattr(sim, "_flex", None),
      )
      w = cc._workspace
      b = sim.batch_size
      nr = cc.descriptor.nr
      nv = sim.model.nv
      if nr > 0:
        J = w["workspace_J"].view(b, nr, nv)
        dbg = w["workspace_debug"].view(b, -1)
        R = dbg[:, nr * nr : nr * nr + nr]
        aref = dbg[:, nr * nr + nr : nr * nr + 2 * nr]
        jar = torch.bmm(J, qa.unsqueeze(-1)).squeeze(-1) - aref
        # MuJoCo's inverse constraint force is the gradient of the row cost.
        # Bilateral rows are quadratic for either sign. Remaining scalar rows
        # are bounded/unilateral; their bounds are assembled with the row.
        raw_force = -jar / torch.clamp(R, min=1e-12)
        n_eq = int(cc.descriptor.n_eq_rows)
        lo = w["workspace_debug"].view(b, -1)[:, nr * nr + 4 * nr:nr * nr + 5 * nr]
        hi = w["workspace_debug"].view(b, -1)[:, nr * nr + 5 * nr:nr * nr + 6 * nr]
        bounded = torch.minimum(torch.maximum(raw_force, lo), hi)
        if n_eq:
          bounded[:, :n_eq] = raw_force[:, :n_eq]
        force = torch.where(R > 0, bounded, torch.zeros_like(raw_force))
        # Elliptic contacts use MuJoCo's group cone cost (mj_constraintUpdate),
        # whose tangential gradient couples every row in a contact. Applying
        # independent scalar bounds here is correct for pyramidal friction but
        # gives a different force for the elliptic middle-cone region.
        desc = cc.descriptor
        if (int(desc.cone_type) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
            and int(desc.ncontacts_max) > 0):
          packed = np.asarray(desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
          friction = cc._constants["contact_friction"].reshape(-1, 5)
          contact_mask = constraint_result.get("contact_mask")
          contact_base = int(desc.nr_joint)
          for slot, (condim, offset, cone) in enumerate(packed.tolist()):
            if cone != int(mujoco.mjtCone.mjCONE_ELLIPTIC) or condim <= 1:
              continue
            row0 = contact_base + int(offset)
            if row0 + condim > nr:
              raise RuntimeError("elliptic contact rows exceed the native row layout")
            # `mj_makeImpedance` scales the regularized elliptic-cone slope
            # by sqrt(R_tangent / R_normal), so it depends on the assembled
            # soft-contact rows (and opt.impratio), not just material friction.
            mu = friction[slot, 0] * torch.sqrt(
                torch.clamp(R[:, row0 + 1], min=0.0) /
                torch.clamp(R[:, row0], min=1e-12))
            coeff = friction[slot, :condim - 1]
            rows = slice(row0, row0 + condim)
            local_jar = jar[:, rows]
            U = torch.cat((local_jar[:, :1] * mu[:, None],
                           local_jar[:, 1:] * coeff.reshape(1, -1)), dim=1)
            N = U[:, 0]
            T = torch.linalg.vector_norm(U[:, 1:], dim=1)
            top = (N >= mu * T) | ((T <= 0) & (N >= 0))
            bottom = (mu * N + T <= 0) | ((T <= 0) & (N < 0))
            Dm = (1.0 / torch.clamp(R[:, row0], min=1e-12)) / torch.clamp(
                mu * mu * (1.0 + mu * mu), min=1e-24)
            middle_scale = -Dm * (N - mu * T) * mu
            cone_force = torch.empty_like(local_jar)
            cone_force[:, 0] = middle_scale
            Tsafe = torch.clamp(T, min=1e-20)
            for axis in range(1, condim):
              cone_force[:, axis] = (
                  -middle_scale / Tsafe * U[:, axis] * coeff[axis - 1])
            quadratic_force = raw_force[:, rows]
            cone_force = torch.where(top[:, None], torch.zeros_like(cone_force),
                                     torch.where(bottom[:, None], quadratic_force,
                                                 cone_force))
            if contact_mask is not None:
              active = contact_mask[:, slot] > 0.5
              active &= R[:, row0] > 0
              cone_force = torch.where(active[:, None], cone_force,
                                       torch.zeros_like(cone_force))
            force[:, rows] = cone_force
        qfrc_constraint = torch.bmm(J.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
        qfrc_inv = qfrc_inv - qfrc_constraint
    finally:
      _restore_system_buffers(cc._workspace, saved_workspace)
      sim._last_coupled = saved_last
      sim._last_coupled_generation = saved_last_generation
      if hasattr(sim, "_spatial_kin"):
        sim._spatial_kin = saved_spatial_kin
        sim._spatial_cache_key = saved_spatial_key

  if hasattr(sim, "_spatial_kin"):
    sim._spatial_kin = saved_spatial_kin
    sim._spatial_cache_key = saved_spatial_key
  return qfrc_inv


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
  mass = dynamics.get("mass_matrix")
  bias = dynamics.get("qfrc_bias")
  if not isinstance(mass, torch.Tensor) or tuple(mass.shape) != (b, nv, nv):
    raise ValueError("dynamics.mass_matrix has invalid shape")
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

def mj_fwdPosition(sim, qpos=None):
  """NATIVE GPU position stage: kinematics, site/geom poses, spatial tendons."""
  state = sim.state
  qp = _stage_tensor(sim, qpos, (sim.batch_size, sim._mjmodel.nq), "qpos") if qpos is not None else state._qpos
  mpos = getattr(state, "_mpos", None)
  mquat = getattr(state, "_mquat", None)
  dynamics = sim._smooth.run_device(qp, state.qvel, mpos, mquat)
  poses = dynamics["poses"]
  if sim._spatial_tendons is not None:
    sim._spatial_kin = sim._spatial_tendons.run_kinematics(state.qvel, poses)
  return poses


def mj_fwdVelocity(sim, qpos=None, qvel=None, poses=None, dynamics=None):
  """NATIVE GPU velocity stage: smooth Coriolis/centrifugal bias and mass matrix."""
  state = sim.state
  qp = _stage_tensor(sim, qpos, (sim.batch_size, sim._mjmodel.nq), "qpos") if qpos is not None else state._qpos
  qv = _stage_tensor(sim, qvel, (sim.batch_size, sim._mjmodel.nv), "qvel") if qvel is not None else state._qvel
  supplied = _stage_dynamics(sim, dynamics)
  if supplied is not None:
    result = dict(supplied)
  else:
    result = sim._smooth.run_device(qp, qv,
                                    state._mpos if poses is None else None,
                                    state._mquat if poses is None else None,
                                    poses=poses)
  if poses is not None:
    # run_device validated complete pose shape/device/dtype; when supplied
    # dynamics were supplied too, preserve exact position-stage identity.
    result["poses"] = poses
  return result


def mj_fwdActuation(sim, qpos=None, qvel=None, poses=None, ctrl=None):
  """NATIVE GPU actuation stage: actuator kinematics, transmission, and forces."""
  state = sim.state
  qp = _stage_tensor(sim, qpos, (sim.batch_size, sim._mjmodel.nq), "qpos") if qpos is not None else state._qpos
  qv = _stage_tensor(sim, qvel, (sim.batch_size, sim._mjmodel.nv), "qvel") if qvel is not None else state._qvel
  p = poses if poses is not None else mj_fwdPosition(sim, qp)
  if ctrl is not None:
    checked = _stage_tensor(sim, ctrl, (sim.batch_size, sim._mjmodel.nu), "ctrl")
    saved_control = sim._control.clone()
    sim._control.copy_(checked)
  else:
    saved_control = None
  try:
    result = sim._state._torch.zeros((sim.batch_size, sim._mjmodel.nv),
        dtype=sim._state._torch.float32, device=sim._state._device)
    held_ctrl = sim._delayed_control(state._time)
    if sim._transmissions is not None:
      result = result + sim._transmissions.run_device(qp, qv, held_ctrl)
    if sim._actuators is not None:
      result = result + sim._actuation_force(qp, qv, p)
    if sim._motor is not None:
      result = result + sim._motor.run_device(held_ctrl)
    return result
  finally:
    if saved_control is not None:
      sim._control.copy_(saved_control)


def mj_fwdAcceleration(sim, qpos=None, qvel=None, poses=None, dynamics=None, qfrc_applied=None):
  """NATIVE GPU acceleration stage: unconstrained acceleration M a = sum(forces)."""
  state = sim.state
  qp = _stage_tensor(sim, qpos, (sim.batch_size, sim._mjmodel.nq), "qpos") if qpos is not None else state._qpos
  qv = _stage_tensor(sim, qvel, (sim.batch_size, sim._mjmodel.nv), "qvel") if qvel is not None else state._qvel
  dyn = _stage_dynamics(sim, dynamics)
  force = _stage_tensor(sim, qfrc_applied, (sim.batch_size, sim._mjmodel.nv), "qfrc_applied")
  acc, status, _ = sim._acceleration(qp, qv, constrained=False,
      dynamics_override=dyn, poses_override=poses,
      applied_force_override=force)
  return acc, status


def mj_fwdConstraint(sim, qpos=None, qvel=None, poses=None, dynamics=None):
  """NATIVE GPU constraint stage: solve contacts, limits, equalities."""
  state = sim.state
  qp = _stage_tensor(sim, qpos, (sim.batch_size, sim._mjmodel.nq), "qpos") if qpos is not None else state._qpos
  qv = _stage_tensor(sim, qvel, (sim.batch_size, sim._mjmodel.nv), "qvel") if qvel is not None else state._qvel
  dyn = _stage_dynamics(sim, dynamics)
  acc, status, dyn = sim._acceleration(qp, qv, dynamics_override=dyn,
                                      poses_override=poses)
  return acc, status, dyn


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
  if raw.ndim != 1 or raw.dtype.kind not in "iu":
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
