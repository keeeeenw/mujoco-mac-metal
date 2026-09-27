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

"""Coupled constraint lowering and native Metal coupled solve.

This stage unifies scalar joint limits, dry friction, polynomial joint
equalities, and sphere/plane contacts (condim=1 and pyramidal condim=3)
into one coupled constraint solve. Contacts and joint constraints share
the same generalized Delassus matrix, regularizer, and projected solve.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "coupled_constraints.metal"
_MINVAL = 1e-15
_MAX_NV = 32
_MAX_CONTACT_PAIRS = 16
_MAX_ROWS = 96
_MAX_ITERATIONS = 1024
_TOLERANCE = 1e-6

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)


def _frozen(value, dtype=np.float32):
  array = np.array(value, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("model constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class CoupledSolverSettings:
  """Explicit configuration for the native coupled Delassus constraint solver."""
  requested_iterations: int
  effective_iterations: int
  requested_tolerance: float
  effective_tolerance: float
  max_refinement_sweeps: int = 64
  metric: str = "max_normalized_projected_gradient"


@dataclass(frozen=True)
class CoupledConstraintDescriptor:
  """Model constants and candidate constraint rows for the coupled solve."""

  nq: int
  nv: int
  njnt: int
  neq: int
  nc: int
  nr_joint: int
  nr: int
  nbody: int
  ngeom: int
  timestep: float
  refsafe: bool
  impratio: float
  disableflags: int
  iterations: int
  tolerance: float
  solver_settings: CoupledSolverSettings

  # Joint constraint constants
  joint_type: np.ndarray
  joint_qposadr: np.ndarray
  qpos0: np.ndarray
  joint_dofadr: np.ndarray
  joint_limited: np.ndarray
  joint_range: np.ndarray
  joint_margin: np.ndarray
  joint_sol_params: np.ndarray
  dof_frictionloss: np.ndarray
  dof_invweight0: np.ndarray
  dof_sol_params: np.ndarray
  eq_obj: np.ndarray
  eq_data: np.ndarray
  eq_sol_params: np.ndarray
  eq_active0: np.ndarray

  # Contact constants
  geom1: np.ndarray
  geom2: np.ndarray
  radius1: np.ndarray
  radius2: np.ndarray
  margin: np.ndarray
  gap: np.ndarray
  solref: np.ndarray
  solimp: np.ndarray
  condim: np.ndarray
  friction: np.ndarray
  geom_bodyid: np.ndarray
  body_parentid: np.ndarray
  body_jntadr: np.ndarray
  body_jntnum: np.ndarray
  body_invweight0: np.ndarray


def _mix_contact_parameters(model, g1, g2):
  """MuJoCo 3.10 contact parameter mixing for geometry pairs."""
  p1, p2 = int(model.geom_priority[g1]), int(model.geom_priority[g2])
  if p1 > p2:
    return model.geom_solref[g1].copy(), model.geom_solimp[g1].copy(), model.geom_friction[g1].copy()
  if p2 > p1:
    return model.geom_solref[g2].copy(), model.geom_solimp[g2].copy(), model.geom_friction[g2].copy()
  m1, m2 = float(model.geom_solmix[g1]), float(model.geom_solmix[g2])
  if m1 >= _MINVAL and m2 >= _MINVAL:
    mix = m1 / (m1 + m2)
  elif m1 < _MINVAL and m2 < _MINVAL:
    mix = 0.5
  else:
    mix = 0.0 if m1 < _MINVAL else 1.0
  r1, r2 = model.geom_solref[g1], model.geom_solref[g2]
  ref = mix * r1 + (1 - mix) * r2 if r1[0] > 0 and r2[0] > 0 else np.minimum(r1, r2)
  imp = mix * model.geom_solimp[g1] + (1 - mix) * model.geom_solimp[g2]
  friction = np.maximum(model.geom_friction[g1], model.geom_friction[g2])
  return ref, imp, friction


def lower_coupled_constraints(model) -> CoupledConstraintDescriptor:
  """Validate and lower combined joint and contact constraints from an MjModel."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("coupled constraint lowering requires a MuJoCo MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError(f"coupled constraint lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
  if model.nv > _MAX_NV:
    raise ValueError(f"coupled constraint stage bounds nv to {_MAX_NV}; found {model.nv}")

  # 1. Joint constraint validation
  scalar_types = (_HINGE, _SLIDE)
  if model.neq and np.any(np.asarray(model.eq_type) != _EQ_JOINT):
    bad = int(np.flatnonzero(np.asarray(model.eq_type) != _EQ_JOINT)[0])
    raise ValueError(f"equality {bad}: only polynomial joint equality is supported")
  limited = np.asarray(model.jnt_limited, dtype=bool)
  for jid in range(model.njnt):
    if limited[jid] and int(model.jnt_type[jid]) not in scalar_types:
      raise ValueError(f"joint {jid}: only scalar hinge/slide limits are supported")
  for eid in range(model.neq):
    j1, j2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
    if not 0 <= j1 < model.njnt or int(model.jnt_type[j1]) not in scalar_types:
      raise ValueError(f"equality {eid}: object 1 must be a scalar hinge/slide joint")
    if j2 >= 0 and (j2 >= model.njnt or int(model.jnt_type[j2]) not in scalar_types):
      raise ValueError(f"equality {eid}: object 2 must be a scalar hinge/slide joint")
  if model.ntendon and (np.any(model.tendon_limited) or np.any(model.tendon_frictionloss)):
    raise ValueError("tendon limits and frictionloss are unsupported")

  # 2. Contact pairs filtering & lowering
  if model.npair or model.nexclude:
    raise ValueError("explicit geom pairs and exclusions are unsupported")
  if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
    raise ValueError("global contact overrides are unsupported")
  if mujoco.get_mjcb_contactfilter() is not None:
    raise ValueError("global contact filter callback is unsupported")

  pairs = []
  geoms = np.asarray(model.geom_type)
  for a in range(model.ngeom):
    for b in range(a + 1, model.ngeom):
      ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
      weld_a, weld_b = int(model.body_weldid[ba]), int(model.body_weldid[bb])
      parent_a = int(model.body_weldid[model.body_parentid[weld_a]])
      parent_b = int(model.body_weldid[model.body_parentid[weld_b]])
      filter_parent = not (
          int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
      )
      if weld_a == weld_b or (
          filter_parent and weld_a != 0 and weld_b != 0
          and (weld_a == parent_b or weld_b == parent_a)
      ):
        continue
      if not (
          (int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
          or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))
      ):
        continue
      ta, tb = int(geoms[a]), int(geoms[b])
      if (ta, tb) not in ((_SPHERE, _SPHERE), (_PLANE, _SPHERE), (_SPHERE, _PLANE)):
        raise ValueError(
            f"collidable geom pair ({a}, {b}) with types ({ta}, {tb}) is unsupported; "
            "only sphere-plane and sphere-sphere contacts are supported"
        )
      p1, p2 = int(model.geom_priority[a]), int(model.geom_priority[b])
      condim = (
          int(model.geom_condim[a]) if p1 > p2 else
          int(model.geom_condim[b]) if p2 > p1 else
          max(int(model.geom_condim[a]), int(model.geom_condim[b]))
      )
      if condim not in (1, 3):
        raise ValueError(f"contact pair ({a}, {b}) has unsupported condim {condim}; only 1 and 3 are supported")
      if condim == 3 and int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError("condim=3 contact requires pyramidal cone")
      solref, solimp, friction = _mix_contact_parameters(model, a, b)
      if bool(solref[0] > 0) != bool(solref[1] > 0):
        solref = np.asarray(model.opt.o_solref).copy()
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solref[0] > 0:
        solref[0] = max(float(solref[0]), 2.0 * float(model.opt.timestep))
      pairs.append((a, b, solref, solimp, condim, np.repeat(friction[0], 2)))

  nc = len(pairs)
  if nc > _MAX_CONTACT_PAIRS:
    raise ValueError(f"candidate contact pairs ({nc}) exceeds capacity {_MAX_CONTACT_PAIRS}")

  # 3. Capacity calculation
  nr_joint = int(model.neq + model.nv + 2 * model.njnt)
  nr_contact = int(nc * 4)
  nr = int(nr_joint + nr_contact)
  if nr > _MAX_ROWS:
    raise ValueError(f"total candidate constraint rows ({nr}) exceeds capacity {_MAX_ROWS}")

  # Assemble packed joint parameters
  joint_sol_params = (
      np.hstack([model.jnt_solref.reshape(model.njnt, 2), model.jnt_solimp.reshape(model.njnt, 5)]).astype(np.float32)
      if model.njnt
      else np.zeros((1, 7), dtype=np.float32)
  )
  dof_sol_params = (
      np.hstack([model.dof_solref.reshape(model.nv, 2), model.dof_solimp.reshape(model.nv, 5)]).astype(np.float32)
      if model.nv
      else np.zeros((1, 7), dtype=np.float32)
  )
  eq_sol_params = (
      np.hstack([model.eq_solref.reshape(model.neq, 2), model.eq_solimp.reshape(model.neq, 5)]).astype(np.float32)
      if model.neq
      else np.zeros((1, 7), dtype=np.float32)
  )
  eq_obj = (
      np.vstack([model.eq_obj1id, model.eq_obj2id]).T.astype(np.int32)
      if model.neq
      else np.zeros((1, 2), dtype=np.int32)
  )
  eq_data = (
      np.asarray(model.eq_data).reshape(model.neq, 11).astype(np.float32)
      if model.neq
      else np.zeros((1, 11), dtype=np.float32)
  )
  eq_active0 = (
      model.eq_active0.astype(np.uint8)
      if model.neq
      else np.zeros(1, dtype=np.uint8)
  )
  joint_range = (
      model.jnt_range.reshape(model.njnt, 2).astype(np.float32)
      if model.njnt
      else np.zeros((1, 2), dtype=np.float32)
  )

  # Assemble contact arrays
  if pairs:
    g1 = np.array([p[0] for p in pairs], dtype=np.int32)
    g2 = np.array([p[1] for p in pairs], dtype=np.int32)
    sr = np.array([p[2] for p in pairs], dtype=np.float32).reshape(-1, 2)
    si = np.array([p[3] for p in pairs], dtype=np.float32).reshape(-1, 5)
    cd = np.array([p[4] for p in pairs], dtype=np.int32)
    fr = np.array([p[5] for p in pairs], dtype=np.float32).reshape(-1, 2)
    r1 = np.where(geoms[g1] == _PLANE, -1.0, model.geom_size[g1, 0]).astype(np.float32)
    r2 = np.where(geoms[g2] == _PLANE, -1.0, model.geom_size[g2, 0]).astype(np.float32)
    margin = (model.geom_margin[g1] + model.geom_margin[g2]).astype(np.float32)
    gap = (model.geom_gap[g1] + model.geom_gap[g2]).astype(np.float32)
  else:
    g1 = g2 = np.empty(0, dtype=np.int32)
    sr = np.empty((0, 2), dtype=np.float32)
    si = np.empty((0, 5), dtype=np.float32)
    cd = np.empty(0, dtype=np.int32)
    fr = np.empty((0, 2), dtype=np.float32)
    r1 = r2 = margin = gap = np.empty(0, dtype=np.float32)

  iter_req = int(model.opt.iterations)
  if iter_req <= 0 or iter_req > 2048:
    raise ValueError(
        f"integrated_euler_v1 bounds iterations to [1, 2048]; found {iter_req}"
    )

  tol_req = float(model.opt.tolerance)
  if not math.isfinite(tol_req) or tol_req <= 0:
    raise ValueError("model.opt.tolerance must be finite and positive")

  # On single-precision float32 MPS, eps ~ 1.19e-7; tolerances below 1e-6 cannot be guaranteed
  # and are clamped to the hardware precision floor 1e-6. Both requested and effective values
  # are explicitly exposed in CoupledSolverSettings.
  eff_tol = max(tol_req, _TOLERANCE)

  solver_settings = CoupledSolverSettings(
      requested_iterations=iter_req,
      effective_iterations=iter_req,
      requested_tolerance=tol_req,
      effective_tolerance=eff_tol,
      max_refinement_sweeps=64,
      metric="max_normalized_projected_gradient",
  )

  return CoupledConstraintDescriptor(
      nq=int(model.nq), nv=int(model.nv), njnt=int(model.njnt),
      neq=int(model.neq), nc=nc, nr_joint=nr_joint, nr=nr,
      nbody=int(model.nbody), ngeom=int(model.ngeom),
      timestep=float(model.opt.timestep),
      refsafe=not bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)),
      impratio=float(model.opt.impratio),
      disableflags=int(model.opt.disableflags),
      iterations=iter_req,
      tolerance=eff_tol,
      solver_settings=solver_settings,
      joint_type=_frozen(model.jnt_type, np.int32),
      joint_qposadr=_frozen(model.jnt_qposadr, np.int32),
      qpos0=_frozen(model.qpos0, np.float32),
      joint_dofadr=_frozen(model.jnt_dofadr, np.int32),
      joint_limited=_frozen(model.jnt_limited, np.uint8),
      joint_range=_frozen(joint_range, np.float32),
      joint_margin=_frozen(model.jnt_margin, np.float32),
      joint_sol_params=_frozen(joint_sol_params, np.float32),
      dof_frictionloss=_frozen(model.dof_frictionloss, np.float32),
      dof_invweight0=_frozen(model.dof_invweight0, np.float32),
      dof_sol_params=_frozen(dof_sol_params, np.float32),
      eq_obj=_frozen(eq_obj, np.int32),
      eq_data=_frozen(eq_data, np.float32),
      eq_sol_params=_frozen(eq_sol_params, np.float32),
      eq_active0=_frozen(eq_active0, np.uint8),
      geom1=_frozen(g1, np.int32),
      geom2=_frozen(g2, np.int32),
      radius1=_frozen(r1, np.float32),
      radius2=_frozen(r2, np.float32),
      margin=_frozen(margin, np.float32),
      gap=_frozen(gap, np.float32),
      solref=_frozen(sr, np.float32),
      solimp=_frozen(si, np.float32),
      condim=_frozen(cd, np.int32),
      friction=_frozen(fr, np.float32),
      geom_bodyid=_frozen(model.geom_bodyid, np.int32),
      body_parentid=_frozen(model.body_parentid, np.int32),
      body_jntadr=_frozen(model.body_jntadr, np.int32),
      body_jntnum=_frozen(model.body_jntnum, np.int32),
      body_invweight0=_frozen(model.body_invweight0.reshape(-1, 2), np.float32),
  )


def coupled_constraint_oracle(
    model, qpos, qvel, mass_matrix=None, qfrc_smooth=None, eq_active=None
):
  """CPU oracle evaluating coupled constraints using MuJoCo reference forward.

  Returns dict with 'qacc', 'qfrc_constraint', 'status', and 'nefc'.
  """
  d = lower_coupled_constraints(model)
  qpos = np.asarray(qpos, dtype=np.float64)
  qvel = np.asarray(qvel, dtype=np.float64)
  if qpos.ndim != 2 or qpos.shape[1] != d.nq or not len(qpos):
    raise ValueError(f"qpos must have shape (batch, {d.nq}) with batch > 0")
  batch = len(qpos)
  if qvel.shape != (batch, d.nv):
    raise ValueError(f"qvel must have shape (batch, {d.nv})")
  if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
    raise ValueError("qpos and qvel must be finite")

  qfrc_constraint = np.zeros((batch, d.nv), dtype=np.float64)
  qacc = np.zeros((batch, d.nv), dtype=np.float64)
  status = np.zeros(batch, dtype=np.int32)
  nefc = np.zeros(batch, dtype=np.int32)

  for b in range(batch):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[b]
    data.qvel[:] = qvel[b]
    if eq_active is not None:
      act = np.asarray(eq_active, dtype=bool)
      if act.ndim == 1:
        data.eq_active[:] = act.astype(np.uint8)
      else:
        data.eq_active[:] = act[b].astype(np.uint8)

    if qfrc_smooth is not None:
      mujoco.mj_fwdPosition(model, data)
      mujoco.mj_fwdVelocity(model, data)
      data.qfrc_applied[:] = qfrc_smooth[b] - (data.qfrc_passive - data.qfrc_bias)

    mujoco.mj_forward(model, data)
    qacc[b] = data.qacc
    qfrc_constraint[b] = data.qfrc_constraint
    nefc[b] = data.nefc

  return {
      "qacc": qacc,
      "qfrc_constraint": qfrc_constraint,
      "status": status,
      "nefc": nefc,
  }


class MetalCoupledConstraints:
  """Batched native MPS execution for coupled joint and contact constraints."""

  def __init__(self, model, batch_size=1):
    self.descriptor = lower_coupled_constraints(model)
    self.solver_settings = self.descriptor.solver_settings
    self.batch_size = int(batch_size)
    if self.batch_size <= 0:
      raise ValueError("batch_size must be positive")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("native coupled constraints require PyTorch MPS compile_shader")
    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._contact_kernel = self._library.contact_normal
    self._solve_kernel = self._library.solve_coupled_constraints

    d = self.descriptor
    self._constants = {
        "joint_qposadr": self._tensor(d.joint_qposadr),
        "qpos0": self._tensor(d.qpos0),
        "joint_dofadr": self._tensor(d.joint_dofadr),
        "joint_limited": self._tensor(d.joint_limited),
        "joint_range": self._tensor(d.joint_range.reshape(-1)),
        "joint_margin": self._tensor(d.joint_margin),
        "joint_sol_params": self._tensor(d.joint_sol_params.reshape(-1)),
        "dof_frictionloss": self._tensor(d.dof_frictionloss),
        "dof_invweight0": self._tensor(d.dof_invweight0),
        "dof_sol_params": self._tensor(d.dof_sol_params.reshape(-1)),
        "eq_obj": self._tensor(d.eq_obj.reshape(-1)),
        "eq_data": self._tensor(d.eq_data.reshape(-1)),
        "eq_sol_params": self._tensor(d.eq_sol_params.reshape(-1)),
        "geom1": self._tensor(d.geom1),
        "geom2": self._tensor(d.geom2),
        "radius1": self._tensor(d.radius1),
        "radius2": self._tensor(d.radius2),
        "margin": self._tensor(d.margin),
        "gap": self._tensor(d.gap),
        "solref": self._tensor(d.solref.reshape(-1)),
        "solimp": self._tensor(d.solimp.reshape(-1)),
        "condim": self._tensor(d.condim),
        "friction": self._tensor(d.friction.reshape(-1)),
        "geom_bodyid": self._tensor(d.geom_bodyid),
        "body_parentid": self._tensor(d.body_parentid),
        "body_jntadr": self._tensor(d.body_jntadr),
        "body_jntnum": self._tensor(d.body_jntnum),
        "jnt_type": self._tensor(d.joint_type),
        "jnt_dofadr": self._tensor(d.joint_dofadr),
        "body_invweight0": self._tensor(d.body_invweight0.reshape(-1)),
        "c_dims": torch.tensor(
            [d.nv, d.nc, self.batch_size, d.nbody, d.njnt, d.ngeom],
            dtype=torch.int32, device=self._device,
        ),
        "solver_dims": torch.tensor(
            [d.nq, d.nv, d.njnt, d.neq, d.nc, self.batch_size, d.disableflags, 1 if d.refsafe else 0, d.iterations, d.nr],
            dtype=torch.int32, device=self._device,
        ),
        "solver_params": torch.tensor(
            [d.timestep, d.impratio, d.tolerance],
            dtype=torch.float32, device=self._device,
        ),
    }
    self._workspace = None
    self.prepare_workspace(self.batch_size)

  def _tensor(self, value):
    return self._torch.as_tensor(np.asarray(value).copy(), device=self._device)

  def prepare_workspace(self, batch_size):
    """Preallocate device workspace buffers for contacts, solver, and outputs."""
    d, torch = self.descriptor, self._torch
    b = int(batch_size)
    if b <= 0:
      raise ValueError("batch_size must be positive")
    self.batch_size = b
    nv, nc, nr = d.nv, d.nc, d.nr

    def empty(size):
      return torch.zeros(max(size, 1), dtype=torch.float32, device=self._device)

    self._workspace = {
        "contact_row_data": empty(b * nc * 5 * 6),
        "contact_frame": empty(b * nc * 12),
        "contact_jacobian": empty(b * nc * 5 * nv),
        "workspace_J": empty(b * nr * nv),
        "out_force": empty(b * nv),
        "out_acc": empty(b * nv),
        "out_status": torch.zeros(b, dtype=torch.int32, device=self._device),
        "out_diagnostics": empty(b * 2),
        "out_contact_force": empty(b * nc * 5),
        "out_joint_force": empty(b * max(d.nr_joint, 1)),
    }
    if d.neq > 0:
      init_eq = np.broadcast_to(d.eq_active0.astype(np.int32), (b, d.neq)).copy()
    else:
      init_eq = np.zeros((b, 1), dtype=np.int32)
    self._eq_active_default = torch.as_tensor(init_eq, dtype=torch.int32, device=self._device)
    self._constants["c_dims"][2] = b
    self._constants["solver_dims"][5] = b
    return self._workspace

  def run_device(self, poses, mass, qfrc_smooth, qpos, qvel, eq_active=None):
    """Solve coupled contacts, limits, dry friction, and equalities on MPS.

    `poses` is the dict of FK outputs from `MetalKinematics`.
    `mass` is batched generalized mass [batch, nv, nv] (including tendon armature).
    `qfrc_smooth` is unconstrained forces [batch, nv].
    `qpos` is [batch, nq], `qvel` is [batch, nv].
    """
    w, torch, d = self._workspace, self._torch, self.descriptor
    b, nv, nc, nr = self.batch_size, d.nv, d.nc, d.nr

    if eq_active is None:
      eq_active_tensor = self._eq_active_default
    else:
      if not isinstance(eq_active, torch.Tensor):
        raise TypeError("eq_active must be a torch.Tensor")
      expected_shape = (b, d.neq) if d.neq > 0 else (b, 1)
      if eq_active.shape != expected_shape:
        raise ValueError(
            f"eq_active must have shape {expected_shape}, got {eq_active.shape}"
        )
      if eq_active.dtype != torch.int32:
        raise TypeError("eq_active must have dtype torch.int32")
      if eq_active.device.type != "mps":
        raise ValueError("eq_active must be on MPS device")
      if not eq_active.is_contiguous():
        raise ValueError("eq_active must be contiguous")
      eq_active_tensor = eq_active

    # 1. Contact normal kernel (if candidate contact pairs exist)
    if nc > 0:
      self._contact_kernel(
          poses["geom_pos"], poses["geom_quat"], poses["body_pos"], poses["body_quat"],
          poses["joint_anchor"], poses["joint_axis"], qvel.reshape(-1),
          self._constants["geom1"], self._constants["geom2"],
          self._constants["radius1"], self._constants["radius2"],
          self._constants["margin"], self._constants["gap"],
          self._constants["solref"], self._constants["solimp"],
          self._constants["condim"], self._constants["friction"],
          self._constants["geom_bodyid"], self._constants["body_parentid"],
          self._constants["body_jntadr"], self._constants["body_jntnum"],
          self._constants["jnt_type"], self._constants["jnt_dofadr"],
          self._constants["body_invweight0"],
          w["contact_row_data"], w["contact_frame"], w["contact_jacobian"],
          self._constants["c_dims"],
          threads=(b * nc,), group_size=(1,),
      )

    # 2. Coupled constraint solver kernel
    self._solve_kernel(
        mass.reshape(-1), qfrc_smooth.reshape(-1), qpos.reshape(-1), qvel.reshape(-1),
        eq_active_tensor.reshape(-1),
        self._constants["joint_qposadr"], self._constants["qpos0"],
        self._constants["joint_dofadr"], self._constants["joint_limited"],
        self._constants["joint_range"], self._constants["joint_margin"],
        self._constants["joint_sol_params"],
        self._constants["dof_frictionloss"], self._constants["dof_invweight0"],
        self._constants["dof_sol_params"],
        self._constants["eq_obj"], self._constants["eq_data"],
        self._constants["eq_sol_params"],
        w["contact_jacobian"], w["contact_row_data"],
        self._constants["friction"], self._constants["condim"],
        self._constants["solver_dims"], self._constants["solver_params"],
        w["out_force"], w["out_acc"], w["out_status"], w["out_diagnostics"],
        w["out_contact_force"], w["out_joint_force"], w["workspace_J"],
        threads=(b,), group_size=(1,),
    )

    qacc = w["out_acc"][: b * nv].reshape(b, nv)
    qfrc_constraint = w["out_force"][: b * nv].reshape(b, nv)
    status = w["out_status"]
    diagnostics = w["out_diagnostics"][: b * 2].reshape(b, 2)

    result = {
        "qacc": qacc,
        "qfrc_constraint": qfrc_constraint,
        "status": status,
        "solver_diagnostics": diagnostics,
    }

    if nc > 0:
      contact_rows = w["contact_row_data"][: b * nc * 5 * 6].reshape(b, nc, 5, 6)
      contact_frames = w["contact_frame"][: b * nc * 12].reshape(b, nc, 12)
      contact_forces = w["out_contact_force"][: b * nc * 5].reshape(b, nc, 5)
      condim_tensor = self._constants["condim"].reshape(1, -1)
      detected = torch.where(condim_tensor == 3, contact_rows[:, :, 1, 0], contact_rows[:, :, 0, 0])
      result.update({
          "contact_mask": detected,
          "contact_distance": contact_rows[:, :, 0, 1],
          "contact_velocity": contact_rows[:, :, 0, 2],
          "contact_reference": contact_rows[:, :, 0, 3],
          "contact_impedance": contact_rows[:, :, 0, 4],
          "contact_normal": contact_frames[:, :, :3],
          "contact_tangent1": contact_frames[:, :, 3:6],
          "contact_tangent2": contact_frames[:, :, 6:9],
          "contact_position": contact_frames[:, :, 9:12],
          "contact_force": contact_forces[:, :, 0],
          "contact_force_rows": contact_forces,
      })

    if d.nr_joint > 0:
      joint_forces = w["out_joint_force"][: b * d.nr_joint].reshape(b, d.nr_joint)
      result["joint_force"] = joint_forces

    return result
