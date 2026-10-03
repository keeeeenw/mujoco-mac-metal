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

def mj_inverse(sim, qpos=None, qvel=None, qacc=None) -> torch.Tensor:
  """NATIVE GPU inverse dynamics: compute qfrc_inverse on MPS.
  
  qfrc_inverse = M(qpos) * qacc + qfrc_bias(qpos, qvel) - qfrc_passive(qpos, qvel)
  Operates natively on device tensors without host roundtrips.
  Returns borrowed/owned MPS tensor [batch, nv].
  """
  if torch is None:
    raise RuntimeError("PyTorch with MPS is required")
  state = sim.state
  qp = qpos if qpos is not None else state.qpos
  qv = qvel if qvel is not None else state.qvel
  qa = qacc if qacc is not None else state.qacc

  # 1. Run kinematics & smooth dynamics to obtain mass matrix and Coriolis bias
  dynamics = sim._smooth.run_device(qp, qv)
  M = dynamics["mass_matrix"]       # [batch, nv, nv]
  bias = dynamics["qfrc_bias"]      # [batch, nv]

  # 2. Compute M * qacc on MPS
  M_qacc = torch.bmm(M, qa.unsqueeze(-1)).squeeze(-1)

  # 3. Add bias forces
  qfrc_inv = M_qacc + bias

  # 4. Subtract passive forces (damping, spring) if active
  if sim._passive is not None:
    mpos = getattr(state, "_mpos", None)
    mquat = getattr(state, "_mquat", None)
    passive = sim._passive.run_device(qp, qv, mocap_pos=mpos, mocap_quat=mquat)
    qfrc_inv = qfrc_inv - passive

  # 5. Subtract fluid forces if active
  if getattr(sim, "_fluid", None) is not None:
    qfrc_inv = qfrc_inv - sim._fluid.run_device(qp, qv, dynamics)

  # 6. Subtract tendon forces if active
  if getattr(sim, "_tendons", None) is not None:
    t_force, _, _ = sim._tendons.run_device(qp, qv)
    qfrc_inv = qfrc_inv - t_force

  # 7. Subtract flex passive forces if active
  if getattr(sim, "_flex", None) is not None:
    flex_qfrc, _, _ = sim._flex.run_device(
        qp, qv, dynamics["poses"], dynamics.get("cvel", None)
    )
    qfrc_inv = qfrc_inv - flex_qfrc

  # 8. Subtract constraint forces (mj_invConstraint)
  if getattr(sim, "_coupled_constraints", None) is not None:
    cc = sim._coupled_constraints
    _ten_J, _ten_L = sim._spatial_for_coupled(qv, dynamics["poses"]) if hasattr(sim, "_spatial_for_coupled") else (None, None)
    eq_act = getattr(state, "_eq_active", None)
    cc.run_device(
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
      force = torch.where((R > 0) & (jar < 0), -jar / torch.clamp(R, min=1e-12), torch.zeros_like(jar))
      qfrc_constraint = torch.bmm(J.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
      qfrc_inv = qfrc_inv - qfrc_constraint

  return qfrc_inv


# -------------------------------------------------------------------------
# Native Split-Stage APIs (runs on device)
# -------------------------------------------------------------------------

def mj_fwdPosition(sim, qpos=None):
  """NATIVE GPU position stage: kinematics, site/geom poses, spatial tendons."""
  state = sim.state
  qp = qpos if qpos is not None else state.qpos
  mpos = getattr(state, "_mpos", None)
  mquat = getattr(state, "_mquat", None)
  dynamics = sim._smooth.run_device(qp, state.qvel, mpos, mquat)
  poses = dynamics["poses"]
  if sim._spatial_tendons is not None:
    sim._spatial_kin = sim._spatial_tendons.run_kinematics(state.qvel, poses)
  return poses


def mj_fwdVelocity(sim, qpos=None, qvel=None, poses=None):
  """NATIVE GPU velocity stage: smooth Coriolis/centrifugal bias and mass matrix."""
  state = sim.state
  qp = qpos if qpos is not None else state.qpos
  qv = qvel if qvel is not None else state.qvel
  dynamics = sim._smooth.run_device(qp, qv)
  return dynamics


def mj_fwdActuation(sim, qpos=None, qvel=None, poses=None, ctrl=None):
  """NATIVE GPU actuation stage: actuator kinematics, transmission, and forces."""
  state = sim.state
  qp = qpos if qpos is not None else state.qpos
  qv = qvel if qvel is not None else state.qvel
  p = poses if poses is not None else mj_fwdPosition(sim, qp)
  if ctrl is not None:
    sim._prepare_control(ctrl)
  if sim._actuators is not None:
    return sim._actuation_force(qp, qv, p)
  return sim._state._torch.zeros((sim.batch_size, sim._mjmodel.nv),
                                 dtype=sim._state._torch.float32,
                                 device=sim._state._device)


def mj_fwdAcceleration(sim, qpos=None, qvel=None, poses=None, dynamics=None, qfrc_applied=None):
  """NATIVE GPU acceleration stage: unconstrained acceleration M a = sum(forces)."""
  state = sim.state
  qp = qpos if qpos is not None else state.qpos
  qv = qvel if qvel is not None else state.qvel
  if qfrc_applied is not None:
    sim._prepare_force(qfrc_applied)
  acc, status, _ = sim._acceleration(qp, qv, constrained=False)
  return acc, status


def mj_fwdConstraint(sim, qpos=None, qvel=None, poses=None, dynamics=None):
  """NATIVE GPU constraint stage: solve contacts, limits, equalities."""
  state = sim.state
  qp = qpos if qpos is not None else state.qpos
  qv = qvel if qvel is not None else state.qvel
  acc, status, dyn = sim._acceleration(qp, qv)
  return acc, status, dyn


# -------------------------------------------------------------------------
# Native State Selector APIs (mj_getState / mj_setState)
# -------------------------------------------------------------------------

class StateSpec:
  """Bitmask specification matching mjtState for state selection."""
  TIME = 1
  QPOS = 2
  QVEL = 4
  ACT = 8
  HISTORY = 16
  WARMSTART = 32
  CTRL = 64
  QFRC_APPLIED = 128
  XFRC_APPLIED = 256
  EQ_ACTIVE = 512
  MOCAP_POS = 1024
  MOCAP_QUAT = 2048
  USERDATA = 4096
  PLUGIN = 8192
  PHYSICS = 30
  USER = 8128
  FULLPHYSICS = 8223
  INTEGRATION = 16383
  ALL = 16383


def mj_getState(sim, spec: int = StateSpec.ALL, env_ids=None) -> dict[str, torch.Tensor | np.ndarray]:
  """NATIVE GPU state extractor: return dictionary of requested state tensors."""
  res = {}
  state = sim.state
  ids = env_ids
  if spec & StateSpec.TIME:
    t = state.time
    res["time"] = t if ids is None else t[ids]
  if spec & StateSpec.QPOS:
    qp = state.qpos
    res["qpos"] = qp if ids is None else qp[ids]
  if spec & StateSpec.QVEL:
    qv = state.qvel
    res["qvel"] = qv if ids is None else qv[ids]
  if spec & StateSpec.ACT:
    act = state.act
    if act is not None:
      res["act"] = act if ids is None else act[ids]
  if spec & StateSpec.CTRL:
    ctrl = sim._control
    if ctrl is not None:
      res["ctrl"] = ctrl if ids is None else ctrl[ids]
  if spec & StateSpec.WARMSTART:
    cc = getattr(sim, "_coupled_constraints", None)
    if cc is not None and cc.descriptor.nr > 0:
      w = cc.get_warmstart()
      res["warmstart"] = w if ids is None else w[ids]
  if spec & StateSpec.EQ_ACTIVE:
    eq = state.eq_active
    if eq is not None:
      res["eq_active"] = eq if ids is None else eq[ids]
  if spec & StateSpec.MOCAP_POS:
    mp = state.mocap_pos
    if mp is not None:
      res["mocap_pos"] = mp if ids is None else mp[ids]
  if spec & StateSpec.MOCAP_QUAT:
    mq = state.mocap_quat
    if mq is not None:
      res["mocap_quat"] = mq if ids is None else mq[ids]
  return res


def mj_setState(sim, values: dict[str, Any], env_ids=None):
  """NATIVE GPU state setter: atomically apply provided state arrays."""
  if not isinstance(values, dict):
    raise TypeError(f"values must be a dict, got {type(values)}")

  recognized_keys = {
      "time", "qpos", "qvel", "act", "qacc", "ctrl", "warmstart",
      "eq_active", "xfrc_applied", "qfrc_applied", "mocap_pos", "mocap_quat"
  }
  for k in values:
    if k not in recognized_keys:
      raise ValueError(f"Unknown state key: {k}")

  model = getattr(sim, "model", None) or getattr(sim, "_mjmodel", None) or sim.state._model
  batch = sim.batch_size
  device = getattr(sim, "device", None) or getattr(sim.state, "device", "mps")

  if env_ids is None:
    target_idx = None
    target_count = batch
  else:
    target_idx = [int(i) for i in env_ids]
    target_count = len(target_idx)
    for idx in target_idx:
      if idx < 0 or idx >= batch:
        raise IndexError(f"env_id {idx} out of range [0, {batch})")

  to_apply = {}

  for k, val in values.items():
    if val is None:
      continue
    if isinstance(val, np.ndarray):
      t = torch.from_numpy(val).to(device=device)
    elif isinstance(val, torch.Tensor):
      t = val.to(device=device)
    else:
      t = torch.as_tensor(val, device=device)

    if k == "time":
      if t.numel() == 1:
        t = t.view(1).expand(target_count).to(torch.float32)
      elif t.shape == (target_count,) or t.shape == (target_count, 1):
        t = t.view(target_count).to(torch.float32)
      elif env_ids is not None and (t.shape == (batch,) or t.shape == (batch, 1)):
        t = t.view(batch)[target_idx].to(torch.float32)
      else:
        raise ValueError(f"time shape mismatch: got {t.shape}")
      to_apply["time"] = t

    elif k == "qacc":
      if t.shape == (target_count, model.nv):
        t = t.to(torch.float32)
      elif env_ids is not None and t.shape == (batch, model.nv):
        t = t[target_idx].to(torch.float32)
      elif t.numel() == 1 and model.nv == 1:
        t = t.view(1, 1).expand(target_count, 1).to(torch.float32)
      else:
        raise ValueError(f"qacc shape mismatch: expected (_, {model.nv}), got {t.shape}")
      to_apply["qacc"] = t

    elif k == "qpos":
      if t.shape == (target_count, model.nq):
        t = t.to(torch.float32)
      elif env_ids is not None and t.shape == (batch, model.nq):
        t = t[target_idx].to(torch.float32)
      else:
        raise ValueError(f"qpos shape mismatch: expected (_, {model.nq}), got {t.shape}")
      to_apply["qpos"] = t

    elif k == "qvel":
      if t.shape == (target_count, model.nv):
        t = t.to(torch.float32)
      elif env_ids is not None and t.shape == (batch, model.nv):
        t = t[target_idx].to(torch.float32)
      else:
        raise ValueError(f"qvel shape mismatch: expected (_, {model.nv}), got {t.shape}")
      to_apply["qvel"] = t

    elif k == "act":
      if model.na == 0:
        continue
      if t.shape == (target_count, model.na):
        t = t.to(torch.float32)
      elif env_ids is not None and t.shape == (batch, model.na):
        t = t[target_idx].to(torch.float32)
      else:
        raise ValueError(f"act shape mismatch: expected (_, {model.na}), got {t.shape}")
      to_apply["act"] = t

    elif k == "ctrl":
      if model.nu == 0:
        continue
      if t.shape == (target_count, model.nu):
        t = t.to(torch.float32)
      elif env_ids is not None and t.shape == (batch, model.nu):
        t = t[target_idx].to(torch.float32)
      else:
        raise ValueError(f"ctrl shape mismatch: expected (_, {model.nu}), got {t.shape}")
      to_apply["ctrl"] = t

    elif k == "warmstart":
      cc = getattr(sim, "_coupled_constraints", None)
      if cc is not None and cc.descriptor.nr > 0:
        nr = cc.descriptor.nr
        if t.shape == (target_count, nr):
          t = t.to(torch.float32)
        elif env_ids is not None and t.shape == (batch, nr):
          t = t[target_idx].to(torch.float32)
        else:
          raise ValueError(f"warmstart shape mismatch: expected (_, {nr}), got {t.shape}")
        to_apply["warmstart"] = t

    elif k == "eq_active":
      if model.neq > 0:
        if t.shape == (target_count, model.neq):
          t = t.to(torch.int32)
        elif env_ids is not None and t.shape == (batch, model.neq):
          t = t[target_idx].to(torch.int32)
        else:
          raise ValueError(f"eq_active shape mismatch: expected (_, {model.neq}), got {t.shape}")
        to_apply["eq_active"] = t

  state = sim.state
  if target_idx is None:
    if "time" in to_apply:
      state._time.copy_(to_apply["time"].view(state._time.shape))
    if "qacc" in to_apply:
      state._qacc.copy_(to_apply["qacc"])
    if "qpos" in to_apply:
      state._qpos.copy_(to_apply["qpos"])
      if hasattr(sim, "flex") and sim.flex is not None:
        dyn = sim._smooth.run_device(state._qpos, state._qvel)
        sim.flex.update_kinematics(dyn["poses"], dyn.get("cvel"))
    if "qvel" in to_apply:
      state._qvel.copy_(to_apply["qvel"])
    if "act" in to_apply and state._act is not None:
      state._act.copy_(to_apply["act"])
    if "ctrl" in to_apply and sim._control is not None:
      sim._control.copy_(to_apply["ctrl"])
    if "warmstart" in to_apply:
      sim._coupled_constraints.set_warmstart(to_apply["warmstart"], env_ids=None)
    if "eq_active" in to_apply and state._eq_active is not None:
      state._eq_active.copy_(to_apply["eq_active"])
  else:
    idx_tensor = torch.tensor(target_idx, device=device, dtype=torch.long)
    if "time" in to_apply:
      state._time[idx_tensor] = to_apply["time"].view(-1)
    if "qacc" in to_apply:
      state._qacc[idx_tensor] = to_apply["qacc"]
    if "qpos" in to_apply:
      state._qpos[idx_tensor] = to_apply["qpos"]
      if hasattr(sim, "flex") and sim.flex is not None:
        dyn = sim._smooth.run_device(state._qpos, state._qvel)
        sim.flex.update_kinematics(dyn["poses"], dyn.get("cvel"))
    if "qvel" in to_apply:
      state._qvel[idx_tensor] = to_apply["qvel"]
    if "act" in to_apply and state._act is not None:
      state._act[idx_tensor] = to_apply["act"]
    if "ctrl" in to_apply and sim._control is not None:
      sim._control[idx_tensor] = to_apply["ctrl"]
    if "warmstart" in to_apply:
      sim._coupled_constraints.set_warmstart(to_apply["warmstart"], env_ids=target_idx)
    if "eq_active" in to_apply and state._eq_active is not None:
      state._eq_active[idx_tensor] = to_apply["eq_active"]
