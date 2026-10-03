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
  dyn = dynamics if dynamics is not None else mj_fwdVelocity(sim, qp, qv)
  if qfrc_applied is not None:
    sim._prepare_force(qfrc_applied)
  acc, status, _ = sim._acceleration(qp, qv)
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
  TIME = 1 << 0
  QPOS = 1 << 1
  QVEL = 1 << 2
  ACT = 1 << 3
  QACC = 1 << 4
  CTRL = 1 << 5
  WARMSTART = 1 << 6
  ALL = (1 << 7) - 1


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
  if spec & StateSpec.QACC:
    qa = state.qacc
    res["qacc"] = qa if ids is None else qa[ids]
  if spec & StateSpec.CTRL:
    ctrl = sim._control
    if ctrl is not None:
      res["ctrl"] = ctrl if ids is None else ctrl[ids]
  if spec & StateSpec.WARMSTART:
    cc = getattr(sim, "_coupled_constraints", None)
    if cc is not None and cc.descriptor.nr > 0:
      w = cc.get_warmstart()
      res["warmstart"] = w if ids is None else w[ids]
  return res


def mj_setState(sim, values: dict[str, torch.Tensor | np.ndarray], env_ids=None):
  """NATIVE GPU state setter: atomically apply provided state arrays."""
  qpos = values.get("qpos", None)
  qvel = values.get("qvel", None)
  act = values.get("act", None)
  if qpos is not None or qvel is not None or act is not None:
    sim.reset(env_ids=env_ids, qpos=qpos, qvel=qvel, act=act)
  if "ctrl" in values:
    sim._prepare_control(values["ctrl"])
  if "warmstart" in values:
    cc = getattr(sim, "_coupled_constraints", None)
    if cc is not None:
      cc.set_warmstart(values["warmstart"], env_ids=env_ids)
