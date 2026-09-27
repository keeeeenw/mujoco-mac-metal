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
equalities, and primitive contacts (plane, sphere, capsule, box; condim=1 and
pyramidal condim=3) into one coupled constraint solve. Contacts and joint
constraints share the same generalized Delassus matrix, regularizer, and
projected solve.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "coupled_constraints.metal"
_COLLISION_SHADER = Path(__file__).parent / "shaders" / "collision_primitives.metal"
_MINVAL = 1e-15
_MAX_NV = 32
_MAX_PAIRS = 16
_MAX_CONTACTS = 24
_MAX_ROWS = 96
_MAX_ITERATIONS = 2048
_TOLERANCE = 1e-6

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)

_SUPPORTED_GEOM_TYPES = (_PLANE, _SPHERE, _CAPSULE, _BOX)


def pair_max_contacts(t1: int, t2: int) -> int:
  """Derive upper bound on contact points for a primitive geometry pair."""
  t_min, t_max = min(t1, t2), max(t1, t2)
  if (t_min, t_max) in (
      (_PLANE, _SPHERE),
      (_SPHERE, _SPHERE),
      (_SPHERE, _CAPSULE),
      (_SPHERE, _BOX),
  ):
    return 1
  if (t_min, t_max) in (
      (_PLANE, _CAPSULE),
      (_CAPSULE, _CAPSULE),
      (_CAPSULE, _BOX),
  ):
    return 2
  if (t_min, t_max) == (_PLANE, _BOX):
    return 4
  if (t_min, t_max) == (_BOX, _BOX):
    return 8
  return 0


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
  nc: int  # number of candidate pairs (for backward compatibility)
  npairs: int
  ncontacts_max: int
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
  pair_contact_offset: np.ndarray
  pair_max_contacts: np.ndarray
  contact_condim: np.ndarray
  contact_condim_packed: np.ndarray
  contact_friction: np.ndarray
  radius1: np.ndarray

  radius2: np.ndarray
  margin: np.ndarray
  gap: np.ndarray
  solref: np.ndarray
  solimp: np.ndarray
  condim: np.ndarray
  friction: np.ndarray
  geom_bodyid: np.ndarray
  geom_size: np.ndarray
  geom_type: np.ndarray
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
  if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
    raise ValueError("global contact overrides are unsupported")
  if mujoco.get_mjcb_contactfilter() is not None:
    raise ValueError("global contact filter callback is unsupported")

  pairs = []
  pair_signatures = set()
  geoms = np.asarray(model.geom_type)

  # Explicit pairs (<pair ...>)
  if model.npair > 0:
    for p in range(model.npair):
      g1 = int(model.pair_geom1[p])
      g2 = int(model.pair_geom2[p])
      t1, t2 = int(geoms[g1]), int(geoms[g2])
      if (t1, t2) == (_PLANE, _PLANE):
        continue
      if t1 not in _SUPPORTED_GEOM_TYPES or t2 not in _SUPPORTED_GEOM_TYPES:
        raise ValueError(
            f"explicit pair ({g1}, {g2}) with types ({t1}, {t2}) is unsupported; "
            "only plane, sphere, capsule, and box are supported"
        )
      # Canonicalize ordering matching MuJoCo C pushGeomGeom
      if t1 > t2 or (t1 == t2 and g1 > g2):
        g1, g2 = g2, g1
        t1, t2 = t2, t1
      cdim = int(model.pair_dim[p])
      if cdim not in (1, 3):
        raise ValueError(f"explicit pair ({g1}, {g2}) has unsupported condim {cdim}; only 1 and 3 are supported")
      if cdim == 3 and int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError("condim=3 contact requires pyramidal cone")

      solref = model.pair_solref[p].copy()
      if solref[0] == 0.0 and solref[1] == 0.0:
        solref = np.asarray(model.opt.o_solref).copy()
      if bool(solref[0] > 0) != bool(solref[1] > 0):
        solref = np.asarray(model.opt.o_solref).copy()
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solref[0] > 0:
        solref[0] = max(float(solref[0]), 2.0 * float(model.opt.timestep))

      solimp = model.pair_solimp[p].copy()
      margin = float(model.pair_margin[p])
      gap = float(model.pair_gap[p])
      friction = model.pair_friction[p, :2].copy()

      pairs.append((g1, g2, solref, solimp, cdim, friction, margin, gap))
      pair_signatures.add((min(g1, g2) << 16) + max(g1, g2))

  # Body exclusions (<exclude ...>)
  exclude_signatures = set()
  if model.nexclude > 0:
    for sig in model.exclude_signature:
      exclude_signatures.add(int(sig))

  # Dynamic geom pairs
  for a in range(model.ngeom):
    for b in range(a + 1, model.ngeom):
      sig_geom = (a << 16) + b
      if sig_geom in pair_signatures:
        continue

      ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
      sig_body = (min(ba, bb) << 16) + max(ba, bb)
      if sig_body in exclude_signatures:
        continue

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
      if (ta, tb) == (_PLANE, _PLANE):
        continue
      if ta not in _SUPPORTED_GEOM_TYPES or tb not in _SUPPORTED_GEOM_TYPES:
        raise ValueError(
            f"collidable geom pair ({a}, {b}) with types ({ta}, {tb}) is unsupported; "
            "only plane, sphere, capsule, and box contacts are supported"
        )

      # Canonical ordering
      if ta > tb or (ta == tb and a > b):
        g1, g2 = b, a
        t1, t2 = tb, ta
      else:
        g1, g2 = a, b
        t1, t2 = ta, tb

      p1, p2 = int(model.geom_priority[g1]), int(model.geom_priority[g2])
      condim = (
          int(model.geom_condim[g1]) if p1 > p2 else
          int(model.geom_condim[g2]) if p2 > p1 else
          max(int(model.geom_condim[g1]), int(model.geom_condim[g2]))
      )
      if condim not in (1, 3):
        raise ValueError(f"contact pair ({g1}, {g2}) has unsupported condim {condim}; only 1 and 3 are supported")
      if condim == 3 and int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError("condim=3 contact requires pyramidal cone")

      solref, solimp, friction = _mix_contact_parameters(model, g1, g2)
      if bool(solref[0] > 0) != bool(solref[1] > 0):
        solref = np.asarray(model.opt.o_solref).copy()
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solref[0] > 0:
        solref[0] = max(float(solref[0]), 2.0 * float(model.opt.timestep))

      margin = float(model.geom_margin[g1] + model.geom_margin[g2])
      gap = float(model.geom_gap[g1] + model.geom_gap[g2])

      pairs.append((g1, g2, solref, solimp, condim, np.repeat(friction[0], 2), margin, gap))

  npairs = len(pairs)
  if npairs > _MAX_PAIRS:
    raise ValueError(f"candidate contact pairs ({npairs}) exceeds capacity {_MAX_PAIRS}")

  # 3. Contact slots and offsets derivation
  pair_contact_offset = [0]
  pair_max_c = []
  total_candidate_contacts = 0
  contact_condim_list = []
  contact_friction_list = []
  contact_condim_packed = []

  row_offset = 0
  for p in pairs:
    g1, g2, solref, solimp, condim, friction, margin, gap = p
    mc = pair_max_contacts(int(geoms[g1]), int(geoms[g2]))
    pair_max_c.append(mc)
    total_candidate_contacts += mc
    pair_contact_offset.append(total_candidate_contacts)
    rows_per_con = 4 if condim == 3 else 1
    for _ in range(mc):
      contact_condim_list.append(condim)
      contact_friction_list.append(friction)
      contact_condim_packed.extend([condim, row_offset])
      row_offset += rows_per_con

  if total_candidate_contacts > _MAX_CONTACTS:
    raise ValueError(f"total candidate contact slots ({total_candidate_contacts}) exceeds capacity {_MAX_CONTACTS}")

  # 4. Capacity calculation
  nr_joint = int(model.neq + model.nv + 2 * model.njnt)
  nr_contact = int(row_offset)
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
    margin = np.array([p[6] for p in pairs], dtype=np.float32)
    gap = np.array([p[7] for p in pairs], dtype=np.float32)
    r1 = np.where(geoms[g1] == _PLANE, -1.0, model.geom_size[g1, 0]).astype(np.float32)
    r2 = np.where(geoms[g2] == _PLANE, -1.0, model.geom_size[g2, 0]).astype(np.float32)
    p_offset = np.array(pair_contact_offset, dtype=np.int32)
    p_max_c = np.array(pair_max_c, dtype=np.int32)
    c_condim = np.array(contact_condim_list, dtype=np.int32)
    c_friction = np.array(contact_friction_list, dtype=np.float32).reshape(-1, 2)
    c_condim_packed = np.array(contact_condim_packed, dtype=np.int32)
  else:
    g1 = g2 = np.empty(0, dtype=np.int32)
    sr = np.empty((0, 2), dtype=np.float32)
    si = np.empty((0, 5), dtype=np.float32)
    cd = np.empty(0, dtype=np.int32)
    fr = np.empty((0, 2), dtype=np.float32)
    r1 = r2 = margin = gap = np.empty(0, dtype=np.float32)
    p_offset = np.zeros(1, dtype=np.int32)
    p_max_c = np.empty(0, dtype=np.int32)
    c_condim = np.empty(0, dtype=np.int32)
    c_friction = np.empty((0, 2), dtype=np.float32)
    c_condim_packed = np.zeros(2, dtype=np.int32)


  geom_size_mat = np.asarray(model.geom_size, dtype=np.float32).reshape(model.ngeom, 3)

  iter_req = int(model.opt.iterations)
  if iter_req <= 0 or iter_req > _MAX_ITERATIONS:
    raise ValueError(
        f"integrated_euler_v1 bounds iterations to [1, {_MAX_ITERATIONS}]; found {iter_req}"
    )

  tol_req = float(model.opt.tolerance)
  if not math.isfinite(tol_req) or tol_req <= 0:
    raise ValueError("model.opt.tolerance must be finite and positive")

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
      neq=int(model.neq), nc=npairs, npairs=npairs, ncontacts_max=total_candidate_contacts,
      nr_joint=nr_joint, nr=nr,
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
      pair_contact_offset=_frozen(p_offset, np.int32),
      pair_max_contacts=_frozen(p_max_c, np.int32),
      contact_condim=_frozen(c_condim, np.int32),
      contact_condim_packed=_frozen(c_condim_packed, np.int32),
      contact_friction=_frozen(c_friction, np.float32),
      radius1=_frozen(r1, np.float32),
      radius2=_frozen(r2, np.float32),

      margin=_frozen(margin, np.float32),
      gap=_frozen(gap, np.float32),
      solref=_frozen(sr, np.float32),
      solimp=_frozen(si, np.float32),
      condim=_frozen(cd, np.int32),
      friction=_frozen(fr, np.float32),
      geom_bodyid=_frozen(model.geom_bodyid, np.int32),
      geom_size=_frozen(geom_size_mat, np.float32),
      geom_type=_frozen(model.geom_type, np.int32),
      body_parentid=_frozen(model.body_parentid, np.int32),
      body_jntadr=_frozen(model.body_jntadr, np.int32),
      body_jntnum=_frozen(model.body_jntnum, np.int32),
      body_invweight0=_frozen(model.body_invweight0.reshape(-1, 2), np.float32),
  )


def coupled_constraint_oracle(
    model, qpos, qvel, mass_matrix=None, qfrc_smooth=None, eq_active=None
):
  """CPU oracle evaluating coupled constraints using MuJoCo reference forward.

  Returns dict with 'qacc', 'qfrc_constraint', 'status', 'nefc', and intermediate
  quantities matching MuJoCo reference data.
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

  efc_J_list = []
  efc_D_list = []
  efc_b_list = []
  efc_force_list = []
  contacts_list = []

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
    efc_J_list.append(data.efc_J.reshape(data.nefc, model.nv).copy() if data.nefc else np.zeros((0, model.nv)))
    efc_D_list.append(data.efc_D.copy() if data.nefc else np.zeros(0))
    efc_b_list.append(data.efc_b.copy() if data.nefc else np.zeros(0))
    efc_force_list.append(data.efc_force.copy() if data.nefc else np.zeros(0))

    b_contacts = []
    for c_idx in range(data.ncon):
      con = data.contact[c_idx]
      b_contacts.append({
          "geom1": int(con.geom1),
          "geom2": int(con.geom2),
          "dist": float(con.dist),
          "pos": con.pos.copy(),
          "frame": con.frame.copy(),
          "dim": int(con.dim),
          "friction": con.friction.copy(),
          "solref": con.solref.copy(),
          "solimp": con.solimp.copy(),
      })
    contacts_list.append(b_contacts)

  return {
      "qacc": qacc,
      "qfrc_constraint": qfrc_constraint,
      "status": status,
      "nefc": nefc,
      "efc_J": efc_J_list,
      "efc_D": efc_D_list,
      "efc_b": efc_b_list,
      "efc_force": efc_force_list,
      "contacts": contacts_list,
  }


class MetalCoupledConstraints:
  """Batched native MPS execution for coupled joint and primitive contact constraints."""

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

    shader_source = _COLLISION_SHADER.read_text() + "\n" + _SHADER.read_text()
    self._library = torch.mps.compile_shader(shader_source)
    self._contact_kernel = self._library.contact_normal
    self._solve_kernel = self._library.solve_coupled_constraints

    d = self.descriptor
    joint_limit_params = (
        np.hstack([d.joint_range, d.joint_margin[:, None]]).astype(np.float32)
        if d.njnt
        else np.zeros((1, 3), dtype=np.float32)
    )

    # Interleaved pair_geoms [npairs, 2] -> [npairs * 2]
    pair_geoms = (
        np.column_stack([d.geom1, d.geom2]).astype(np.int32).reshape(-1)
        if d.npairs
        else np.zeros(2, dtype=np.int32)
    )
    # Interleaved pair_margin_gap [npairs, 2] -> [npairs * 2]
    pair_margin_gap = (
        np.column_stack([d.margin, d.gap]).astype(np.float32).reshape(-1)
        if d.npairs
        else np.zeros(2, dtype=np.float32)
    )

    contact_friction_arr = (
        d.contact_friction.reshape(-1)
        if d.ncontacts_max
        else np.zeros(2, dtype=np.float32)
    )
    contact_condim_arr = (
        d.contact_condim
        if d.ncontacts_max
        else np.zeros(1, dtype=np.int32)
    )

    self._constants = {
        "joint_qposadr": self._tensor(d.joint_qposadr),
        "qpos0": self._tensor(d.qpos0),
        "joint_dofadr": self._tensor(d.joint_dofadr),
        "joint_limited": self._tensor(d.joint_limited),
        "joint_limit_params": self._tensor(joint_limit_params.reshape(-1)),
        "joint_sol_params": self._tensor(d.joint_sol_params.reshape(-1)),
        "dof_frictionloss": self._tensor(d.dof_frictionloss),
        "dof_invweight0": self._tensor(d.dof_invweight0),
        "dof_sol_params": self._tensor(d.dof_sol_params.reshape(-1)),
        "eq_obj": self._tensor(d.eq_obj.reshape(-1)),
        "eq_data": self._tensor(d.eq_data.reshape(-1)),
        "eq_sol_params": self._tensor(d.eq_sol_params.reshape(-1)),
        "geom_size": self._tensor(d.geom_size.reshape(-1)),
        "geom_type": self._tensor(d.geom_type),
        "geom_bodyid": self._tensor(d.geom_bodyid),
        "body_parentid": self._tensor(d.body_parentid),
        "body_jntadr": self._tensor(d.body_jntadr),
        "body_jntnum": self._tensor(d.body_jntnum),
        "jnt_type": self._tensor(d.joint_type),
        "jnt_dofadr": self._tensor(d.joint_dofadr),
        "body_invweight0": self._tensor(d.body_invweight0.reshape(-1)),
        "pair_geoms": self._tensor(pair_geoms),
        "pair_margin_gap": self._tensor(pair_margin_gap),
        "pair_solref": self._tensor(d.solref.reshape(-1) if d.npairs else np.zeros(2, dtype=np.float32)),
        "pair_solimp": self._tensor(d.solimp.reshape(-1) if d.npairs else np.zeros(5, dtype=np.float32)),
        "pair_condim": self._tensor(d.condim if d.npairs else np.zeros(1, dtype=np.int32)),
        "pair_friction": self._tensor(d.friction.reshape(-1) if d.npairs else np.zeros(2, dtype=np.float32)),
        "pair_contact_offset": self._tensor(d.pair_contact_offset),
        "contact_friction": self._tensor(contact_friction_arr),
        "contact_condim": self._tensor(d.contact_condim_packed if d.ncontacts_max else np.zeros(2, dtype=np.int32)),
        "c_dims": torch.tensor(

            [d.nv, d.npairs, d.ncontacts_max, self.batch_size, d.nbody, d.njnt, d.ngeom],
            dtype=torch.int32, device=self._device,
        ),
        "solver_dims": torch.tensor(
            [d.nq, d.nv, d.njnt, d.neq, d.ncontacts_max, self.batch_size, d.disableflags, 1 if d.refsafe else 0, d.iterations, d.nr],
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
    nv, nc, nr = d.nv, d.ncontacts_max, d.nr

    def empty(size):
      return torch.zeros(max(size, 1), dtype=torch.float32, device=self._device)

    self._workspace = {
        "contact_row_data": empty(b * nc * 5 * 6),
        "contact_frame": empty(b * nc * 12),
        "contact_jacobian": empty(b * nc * 5 * nv),
        "workspace_J": empty(b * nr * nv),
        "workspace_debug": empty(b * (nr * nr + 4 * nr)),
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
    self._constants["c_dims"][3] = b
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
    b, nv, nc, nr = self.batch_size, d.nv, d.ncontacts_max, d.nr

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
    if d.npairs > 0 and nc > 0:
      self._contact_kernel(
          poses["geom_pos"], poses["geom_quat"],
          self._constants["geom_size"], self._constants["geom_type"],
          self._constants["geom_bodyid"], poses["body_pos"], poses["body_quat"],
          poses["joint_anchor"], poses["joint_axis"], qvel.reshape(-1),
          self._constants["body_parentid"], self._constants["body_jntadr"],
          self._constants["body_jntnum"], self._constants["jnt_type"],
          self._constants["jnt_dofadr"], self._constants["body_invweight0"],
          self._constants["pair_geoms"], self._constants["pair_margin_gap"],
          self._constants["pair_solref"], self._constants["pair_solimp"],
          self._constants["pair_condim"], self._constants["pair_friction"],
          self._constants["pair_contact_offset"],
          w["contact_row_data"], w["contact_frame"], w["contact_jacobian"],
          self._constants["c_dims"],
          threads=(b * d.npairs,), group_size=(1,),
      )

    # 2. Coupled constraint solver kernel
    self._solve_kernel(
        mass.reshape(-1), qfrc_smooth.reshape(-1), qpos.reshape(-1), qvel.reshape(-1),
        eq_active_tensor.reshape(-1),
        self._constants["joint_qposadr"], self._constants["qpos0"],
        self._constants["joint_dofadr"], self._constants["joint_limited"],
        self._constants["joint_limit_params"],
        self._constants["joint_sol_params"],
        self._constants["dof_frictionloss"], self._constants["dof_invweight0"],
        self._constants["dof_sol_params"],
        self._constants["eq_obj"], self._constants["eq_data"],
        self._constants["eq_sol_params"],
        w["contact_jacobian"], w["contact_row_data"],
        self._constants["contact_friction"], self._constants["contact_condim"],
        self._constants["solver_dims"], self._constants["solver_params"],
        w["out_force"], w["out_acc"], w["out_status"], w["out_diagnostics"],
        w["out_contact_force"], w["out_joint_force"], w["workspace_J"],
        w["workspace_debug"],
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

    if nr > 0:
      w_debug = w["workspace_debug"][: b * (nr * nr + 4 * nr)].reshape(b, nr * nr + 4 * nr)
      W = w_debug[:, : nr * nr].reshape(b, nr, nr)
      R = w_debug[:, nr * nr : nr * nr + nr].reshape(b, nr)
      ar = w_debug[:, nr * nr + nr : nr * nr + 2 * nr].reshape(b, nr)
      rhs = w_debug[:, nr * nr + 2 * nr : nr * nr + 3 * nr].reshape(b, nr)
      lam = w_debug[:, nr * nr + 3 * nr : nr * nr + 4 * nr].reshape(b, nr)
      J = w["workspace_J"][: b * nr * nv].reshape(b, nr, nv)
      result.update({
          "J": J,
          "W": W,
          "W_regularized": W + torch.diag_embed(R),
          "R": R,
          "ar": ar,
          "rhs": rhs,
          "lambda": lam,
      })

    if nc > 0:
      contact_rows = w["contact_row_data"][: b * nc * 5 * 6].reshape(b, nc, 5, 6)
      contact_frames = w["contact_frame"][: b * nc * 12].reshape(b, nc, 12)
      contact_forces = w["out_contact_force"][: b * nc * 5].reshape(b, nc, 5)
      condim_tensor = self._constants["contact_condim"][0::2].reshape(1, -1)
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
