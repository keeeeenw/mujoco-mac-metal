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
_BROADPHASE_SHADER = Path(__file__).parent / "shaders" / "broadphase.metal"
_EQUALITY_SHADER = Path(__file__).parent / "shaders" / "equality_assembly.metal"
_COLLISION_SHADER = Path(__file__).parent / "shaders" / "collision_primitives.metal"
_CONVEX_SHADER = Path(__file__).parent / "shaders" / "convex_narrowphase.metal"
_SDF_SHADER = Path(__file__).parent / "shaders" / "sdf_narrowphase.metal"
_MINVAL = 1e-15
_MAX_NV = 32
_MAX_PAIRS = 16
_MAX_CONTACTS = 24
_MAX_ROWS = 96
_MAX_ITERATIONS = 2048
_TOLERANCE = 1e-6
# G3: hard cap for adaptive PGS extension; must match the kernel constant.
_ADAPTIVE_MAX_ITERATIONS = 1024

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_SDF = int(mujoco.mjtGeom.mjGEOM_SDF)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_EQ_CONNECT = int(mujoco.mjtEq.mjEQ_CONNECT)
_EQ_WELD = int(mujoco.mjtEq.mjEQ_WELD)
_EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)
_EQ_TENDON = int(mujoco.mjtEq.mjEQ_TENDON)
_OBJ_BODY = int(mujoco.mjtObj.mjOBJ_BODY)
_OBJ_SITE = int(mujoco.mjtObj.mjOBJ_SITE)

_EQ_ROWS = {
    _EQ_JOINT: 1,
    _EQ_TENDON: 1,
    _EQ_CONNECT: 3,
    _EQ_WELD: 6,
}

_SUPPORTED_EQ_TYPES = (_EQ_JOINT, _EQ_TENDON, _EQ_CONNECT, _EQ_WELD)

_SUPPORTED_GEOM_TYPES = (_PLANE, _HFIELD, _SPHERE, _CAPSULE, _BOX,
                           _ELLIPSOID, _CYLINDER, _MESH, _SDF)

# Milestone 013 SDF bounds: octree nodes total, SDF geoms, per-pair slots.
# Contact counts track opt.sdf_initpoints (upstream mj_maxContact); native
# slots cap each SDF pair below, so lowering requires initpoints within it.
_SDF_MAX_NODES = 262144
_SDF_MAX_GEOMS = 8
_SDF_MAX_PER_PAIR = 8

# Milestone 011 convex-mesh bounds: per-geom hull verts and total hull store.
# Metal caps kernel buffers at 31, so faces pack into the same float store
# as the verts (indices-as-float are exact below 2^24): layout per model is
# [hull verts (3V)] [face normals (3F)] [face indices (3F as float)] with one
# 5-wide int descriptor per geom (hullOff, hullCnt, nrmOff, idxOff, faceCnt).
_MESH_MAX_VERTS = 64
_MESH_MAX_TOTAL = 256
# Face-snap store: triangle indices + outward unit normals (geom-local).
_MESH_MAX_FACES = 256
_MESH_DATA_FLOATS = 3 * (_MESH_MAX_TOTAL + 2 * _MESH_MAX_FACES)


def _rbound_with_mesh_hulls(model, extra):
  import numpy as np
  rb = np.asarray(model.geom_rbound, dtype=np.float32).reshape(int(model.ngeom)).copy()
  for g, v in extra.items():
    if not np.isfinite(rb[g]) or rb[g] < v:
      rb[g] = np.float32(v)
  return rb


def mesh_hull_is_convex(model, meshid, tol=1e-6):
  """Face-plane convexity check for a mesh asset (host preprocessing).

  Every face plane must leave all hull vertices on its interior side. This
  admits exactly the single-convex-piece assets the native vertex-support
  path implements; concave assets stay rejected at lowering (documented).
  """
  import numpy as np
  vadr = int(model.mesh_vertadr[meshid])
  vnum = int(model.mesh_vertnum[meshid])
  fadr = int(model.mesh_faceadr[meshid])
  fnum = int(model.mesh_facenum[meshid])
  verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float64)
  faces = np.asarray(model.mesh_face[fadr:fadr + fnum])
  for f in faces:
    p0, p1, p2 = verts[int(f[0])], verts[int(f[1])], verts[int(f[2])]
    n = np.cross(p1 - p0, p2 - p0)
    nl = float(np.linalg.norm(n))
    if nl < 1e-12:
      continue
    n = n / nl
    if bool(np.all(n @ (verts - p0).T <= tol)):
      continue
    return False
  return True


# Milestone 012 heightfield bounds: dims and total data floats.
_HF_MAX_N = 64
_HF_MAX_DATA = 4096
_HF_MAX_GEOMS = 8


def pair_max_contacts(t1: int, t2: int, sdf_initpoints: int = 8) -> int:
  """Derive upper bound on contact points for a primitive geometry pair.

  Bounds follow the pinned colliders' actual manifold sizes (3.10.0), not
  the conservative mj_maxContact allocation bounds: sphere/ellipsoid
  involvement caps at 1; capsule pairs cap at 2 (mjraw_CapsuleBox emits at
  most best+second); plane-cylinder/box cap at 4; box-box 8; remaining
  cylinder-involved convex pairs cap at 5 (multiccd native witnesses).
  Heightfield pairs emit one witness per overlapped terrain prism
  (mjc_ConvexHField, pinned cap mjMAXCONPAIR=50); native caps below cover
  the qualified demo budgets and are documented restrictions.
  SDF pairs emit one witness per Halton seed (mjc_SDF, pinned bound
  opt.sdf_initpoints); native caps each pair at _SDF_MAX_PER_PAIR.
  """
  t_min, t_max = min(t1, t2), max(t1, t2)
  if _SDF in (t_min, t_max):
    if t_min == _SDF and t_max == _SDF:
      return min(int(sdf_initpoints), _SDF_MAX_PER_PAIR)
    if t_min == _MESH:
      # Mesh-SDF needs BVH face processing + FPS selection (follow-up);
      # analytic and SDF-SDF pairs use the Halton/descent path.
      raise ValueError("mesh-SDF pairs are unsupported in 013")
    if t_min == _PLANE or t_min == _HFIELD:
      return 0
    return min(int(sdf_initpoints), _SDF_MAX_PER_PAIR)
  if _HFIELD in (t_min, t_max):
    if t_min == _HFIELD and t_max == _HFIELD:
      # Pinned mjCOLLISIONFUNC has no heightfield-heightfield entry (NULL):
      # rejection matches the engine, it is not a coverage gap.
      raise ValueError("heightfield-heightfield pairs are unsupported")
    if t_max == _MESH:
      # Faceted-prism vs hull EPA basins need a dedicated snap treatment
      # (measured 5x depth error on shallow presses); analytic types only.
      raise ValueError("mesh-heightfield pairs are unsupported in 012")
    if t_min == _PLANE:
      # Pinned mjCOLLISIONFUNC has no plane-heightfield entry (NULL):
      # zero slots match the engine, not a coverage gap.
      return 0
    if t_max == _SPHERE or t_max == _ELLIPSOID:
      return 8
    if t_max == _BOX:
      return 16
    return 8
  if _SPHERE in (t_min, t_max) or _ELLIPSOID in (t_min, t_max):
    # Pinned rule: any pair involving a sphere or an ellipsoid yields one
    # contact (sphere/python sphere pairs and all ellipsoid pairs). Checked
    # before the multi-contact rules below, matching mj_maxContact order.
    return 1
  if _MESH in (t_min, t_max):
    # R05-1 bounded mesh manifolds: convex mesh pairs yield up to 4 native
    # witnesses (face-clip expansion on planes, perturbed-restart multiCCD
    # on box/cylinder/capsule/mesh). Sphere/ellipsoid involvement returns
    # above (pinned single). Deterministic slot identity via spatial sort.
    return 4
  if (t_min, t_max) in (
      (_PLANE, _CAPSULE),
      (_CAPSULE, _CAPSULE),
      (_CAPSULE, _BOX),
  ):
    return 2
  if (t_min, t_max) in (
      (_PLANE, _CYLINDER),
      (_PLANE, _BOX),
  ):
    return 4
  if (t_min, t_max) == (_BOX, _BOX):
    return 8
  if t_min == _CYLINDER or t_max == _CYLINDER:
    # Remaining cylinder-involved convex pairs (capsule/sphere excluded
    # above): multiccd native path yields up to 5 witnesses.
    return 5
  return 0


def _frozen(value, dtype=np.float32):
  array = np.array(value, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("model constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class CoupledSolverSettings:
  """Explicit configuration for the native coupled Delassus constraint solver.

  `requested_iterations` is the model's outer PGS budget. When a budgeted
  window ends close to certification (residual <= 1e-3) and improved over
  the window start, the solver adaptively extends up to
  `adaptive_max_iterations` (G3 contract); `solver_diagnostics[1]` reports
  the actual iteration count, separately from the configured budgets.
  `max_refinement_sweeps` bounds the exact per-block contact refinement
  (64 rounds for blocks of <= 4 rows).
  """
  requested_iterations: int
  effective_iterations: int
  requested_tolerance: float
  effective_tolerance: float
  adaptive_max_iterations: int = 1024
  max_refinement_sweeps: int = 256
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
  nsite: int
  timestep: float
  refsafe: bool
  impratio: float
  disableflags: int
  iterations: int
  tolerance: float
  noslip_iterations: int
  noslip_tolerance: float
  mean_inertia: float
  solver_settings: CoupledSolverSettings
  n_eq_rows: int

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
  eq_type: np.ndarray
  eq_objtype: np.ndarray
  eq_rowadr: np.ndarray
  eq_rownum: np.ndarray

  # Contact constants
  cone_type: int
  geom1: np.ndarray
  geom2: np.ndarray
  pair_contact_offset: np.ndarray
  pair_max_contacts: np.ndarray
  contact_condim: np.ndarray
  contact_condim_packed: np.ndarray
  contact_friction: np.ndarray
  contact_solreffriction: np.ndarray
  radius1: np.ndarray

  radius2: np.ndarray
  margin: np.ndarray
  gap: np.ndarray
  solref: np.ndarray
  solimp: np.ndarray
  condim: np.ndarray
  friction: np.ndarray
  solreffriction: np.ndarray
  geom_bodyid: np.ndarray
  geom_size: np.ndarray
  geom_type: np.ndarray
  geom_rbound: np.ndarray
  mesh_hull: np.ndarray
  mesh_hull_info: np.ndarray
  body_parentid: np.ndarray
  body_jntadr: np.ndarray
  body_jntnum: np.ndarray
  body_invweight0: np.ndarray
  ntendon: int = 0
  ten_base: int = 0
  ten_friction_rows: int = 0
  ten_limit_rows: int = 0
  ten_limited: np.ndarray = None
  ten_range: np.ndarray = None
  ten_margin: np.ndarray = None
  ten_length0: np.ndarray = None
  ten_invweight0: np.ndarray = None
  ten_solref_lim: np.ndarray = None
  ten_solimp_lim: np.ndarray = None
  ten_frictionloss: np.ndarray = None
  ten_solref_fri: np.ndarray = None
  ten_solimp_fri: np.ndarray = None
  ten_length_map: np.ndarray = None
  ten_moment_map: np.ndarray = None
  dense_path: bool = True


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


def _fixed_tendon_maps(model):
  """Constant fixed-tendon length/moment maps; spatial rows stay zero.

  Mirrors FixedTendonModel math without its strict validation (the spatial
  tendon kernel supplies spatial rows at runtime).
  """
  nt, nq, nv = int(model.ntendon), int(model.nq), int(model.nv)
  wrap_joint = int(mujoco.mjtWrap.mjWRAP_JOINT)
  hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
  slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
  length_map = np.zeros((max(nt, 1), max(nq, 1)), dtype=np.float64)
  moment_map = np.zeros((max(nt, 1), max(nv, 1)), dtype=np.float64)
  for tendon in range(nt):
    start, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
    if count <= 0 or int(model.wrap_type[start]) != wrap_joint:
      continue
    ok = True
    for wrap in range(start, start + count):
      if int(model.wrap_type[wrap]) != wrap_joint:
        ok = False
        break
      joint = int(model.wrap_objid[wrap])
      if joint < 0 or joint >= int(model.njnt) or int(model.jnt_type[joint]) not in (hinge, slide):
        ok = False
        break
    if not ok:
      continue
    for wrap in range(start, start + count):
      joint = int(model.wrap_objid[wrap])
      coefficient = float(model.wrap_prm[wrap])
      length_map[tendon, int(model.jnt_qposadr[joint])] += coefficient
      moment_map[tendon, int(model.jnt_dofadr[joint])] += coefficient
  return length_map.astype(np.float32), moment_map.astype(np.float32)


def lower_coupled_constraints(model, limits=None) -> CoupledConstraintDescriptor:
  """Validate and lower combined joint and contact constraints from an MjModel."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("coupled constraint lowering requires a MuJoCo MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError(f"coupled constraint lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
  from mujoco_metal.capacity import CapacityOverflow
  max_nv = limits.max_nv if limits is not None else _MAX_NV
  if model.nv > max_nv:
    raise CapacityOverflow(f"coupled constraint stage bounds nv to {max_nv}; found {model.nv}")
  # Pinned no-slip post-stage (R04): admitted with validated budget; the
  # native solver runs exact friction subproblem sweeps after the main
  # solve (zero iterations = skipped, bit-identical to before).
  noslip_iters = int(model.opt.noslip_iterations)
  if noslip_iters < 0 or noslip_iters > _MAX_ITERATIONS:
    raise ValueError(
        f"integrated_euler_v1 bounds noslip_iterations to [0, {_MAX_ITERATIONS}]; found {noslip_iters}"
    )
  noslip_tol = float(model.opt.noslip_tolerance)
  if noslip_iters > 0 and (not math.isfinite(noslip_tol) or noslip_tol <= 0):
    raise ValueError("model.opt.noslip_tolerance must be finite and positive with nonzero noslip_iterations")
  if int(model.opt.solver) == int(mujoco.mjtSolver.mjSOL_CG):
    # Mapped like Newton (REQ-SOL-002): the native projected solver handles
    # the same coupled problem for every convex/cone combination; the
    # selection name is honored as a mapping, verified by parity tests.
    pass

  # 1. Equality and joint constraint validation (joint/connect/weld, mixed order)
  scalar_types = (_HINGE, _SLIDE)
  if model.neq:
    eq_types = np.asarray(model.eq_type)
    for eid in range(model.neq):
      et = int(eq_types[eid])
      if et not in _SUPPORTED_EQ_TYPES:
        raise ValueError(
            f"equality {eid}: only joint, tendon, connect and weld equalities are supported "
            f"(found type {et})"
        )
  limited = np.asarray(model.jnt_limited, dtype=bool)
  for jid in range(model.njnt):
    jt = int(model.jnt_type[jid])
    if jt == int(mujoco.mjtJoint.mjJNT_FREE) and limited[jid]:
      raise ValueError(f"joint {jid}: limited free joints are unsupported (upstream ignores them)")
    if limited[jid] and jt not in scalar_types + (_BALL,):
      raise ValueError(f"joint {jid}: only scalar hinge/slide and ball limits are supported")
  # Per-equality validation with explicit row spans; inactive equalities are
  # still validated and reserved (a model is not supported merely because a
  # new equality is initially inactive).
  eq_types_arr = np.asarray(model.eq_type) if model.neq else np.zeros(0, dtype=np.int32)
  eq_objtype_arr = np.asarray(model.eq_objtype) if model.neq else np.zeros(0, dtype=np.int32)
  eq_rowadr_list = []
  eq_rownum_list = []
  eq_row_cursor = 0
  for eid in range(model.neq):
    et = int(eq_types_arr[eid])
    ot = int(eq_objtype_arr[eid])
    span = int(_EQ_ROWS[et])
    eq_rowadr_list.append(eq_row_cursor)
    eq_rownum_list.append(span)
    eq_row_cursor += span
    if et == _EQ_JOINT:
      j1, j2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
      if not 0 <= j1 < model.njnt or int(model.jnt_type[j1]) not in scalar_types:
        raise ValueError(f"equality {eid}: object 1 must be a scalar hinge/slide joint")
      if j2 >= 0 and (j2 >= model.njnt or int(model.jnt_type[j2]) not in scalar_types):
        raise ValueError(f"equality {eid}: object 2 must be a scalar hinge/slide joint")
    elif et == _EQ_TENDON:
      t1, t2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
      if not 0 <= t1 < model.ntendon:
        raise ValueError(f"equality {eid}: tendon 1 id out of range")
      if t2 >= 0 and not 0 <= t2 < model.ntendon:
        raise ValueError(f"equality {eid}: tendon 2 id out of range")
      if not np.all(np.isfinite(np.asarray(model.tendon_length0, dtype=np.float64))):
        raise ValueError(f"equality {eid}: tendon_length0 must be finite")
    elif et == _EQ_CONNECT:
      if ot not in (_OBJ_BODY, _OBJ_SITE):
        raise ValueError(f"equality {eid}: connect requires body or site objects")
      o1, o2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
      if ot == _OBJ_BODY:
        if not 0 <= o1 < model.nbody:
          raise ValueError(f"equality {eid}: body1 id out of range")
        if not 0 <= o2 < model.nbody:
          raise ValueError(f"equality {eid}: body2 id out of range (world 0 allowed)")
        if o1 == o2:
          raise ValueError(f"equality {eid}: connect bodies must differ")
        # World (0), ordinary hinge/slide/ball/free bodies and mocap bodies
        # (kinematic anchors, exactly the crane-attach case) are in scope.
        # A both-mocap equality reserves rows but assembles zero rows/forces,
        # matching MuJoCo skipping its empty Jacobian.
      else:
        if not 0 <= o1 < model.nsite or not 0 <= o2 < model.nsite:
          raise ValueError(f"equality {eid}: site ids out of range")
        if o1 == o2:
          raise ValueError(f"equality {eid}: connect sites must differ")
        # Sites on mocap bodies are supported kinematic anchors (see above).
      # Connect eq_data anchors/ Hogan: validate finite float32 below via _frozen;
      # site-based connect ignores eq_data (must still be finite).
    else:  # _EQ_WELD
      if ot not in (_OBJ_BODY, _OBJ_SITE):
        raise ValueError(f"equality {eid}: weld requires body or site objects")
      o1, o2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
      if ot == _OBJ_BODY:
        if not 0 <= o1 < model.nbody:
          raise ValueError(f"equality {eid}: body1 id out of range")
        if not 0 <= o2 < model.nbody:
          raise ValueError(f"equality {eid}: body2 id out of range (world 0 allowed)")
        if o1 == o2:
          raise ValueError(f"equality {eid}: weld bodies must differ")
        # Mocap bodies allowed as above (crane-attach case).
      else:
        if not 0 <= o1 < model.nsite or not 0 <= o2 < model.nsite:
          raise ValueError(f"equality {eid}: site ids out of range")
        if o1 == o2:
          raise ValueError(f"equality {eid}: weld sites must differ")
        # Sites on mocap bodies allowed as above.
      # Weld torquescale (eq_data[10]) must be finite float32; zero is valid
      # (behaves like connect for rotation) and is explicitly reserved.
      try:
        ts = float(np.asarray(model.eq_data).reshape(model.neq, 11)[eid, 10])
      except Exception:
        ts = float("nan")
      if not math.isfinite(ts):
        raise ValueError(f"equality {eid}: weld torquescale must be finite")
      with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ts32 = np.float32(ts)
      if not np.isfinite(ts32):
        raise ValueError(f"equality {eid}: weld torquescale must be float32-representable")
  n_eq_rows = int(eq_row_cursor)
  eq_rowadr = np.asarray(eq_rowadr_list, dtype=np.int32) if model.neq else np.zeros(0, dtype=np.int32)
  eq_rownum = np.asarray(eq_rownum_list, dtype=np.int32) if model.neq else np.zeros(0, dtype=np.int32)
  # Tendon limits/frictionloss are natively supported rows (milestone 008);
  # validate their parameters here (mirrors SpatialTendonModel admission).
  if model.ntendon:
    if not np.all(np.isfinite(np.asarray(model.tendon_range, dtype=np.float64))):
      raise ValueError("tendon_range must be finite")
    if not np.all(np.isfinite(np.asarray(model.tendon_margin, dtype=np.float64))):
      raise ValueError("tendon_margin must be finite")
    if not np.all(np.isfinite(np.asarray(model.tendon_length0, dtype=np.float64))):
      raise ValueError("tendon_length0 must be finite")
    if not np.all(np.isfinite(np.asarray(model.tendon_invweight0, dtype=np.float64))):
      raise ValueError("tendon_invweight0 must be finite")
    lim = np.asarray(model.tendon_limited, dtype=bool)
    rng = np.asarray(model.tendon_range, dtype=np.float64).reshape(model.ntendon, 2)
    if np.any(lim & (rng[:, 0] > rng[:, 1])):
      raise ValueError("limited tendon ranges must be ordered")
    if np.any(np.asarray(model.tendon_frictionloss) < 0):
      raise ValueError("tendon friction loss must be nonnegative")
  ten_friction_rows = int(np.sum(np.asarray(model.tendon_frictionloss) > 0)) if model.ntendon else 0
  ten_limit_rows = int(2 * np.sum(np.asarray(model.tendon_limited, dtype=bool))) if model.ntendon else 0
  ten_base = int(n_eq_rows + model.nv + 2 * model.njnt)
  n_ten_rows = int(ten_friction_rows + ten_limit_rows)

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
            "only plane, heightfield, sphere, capsule, box, ellipsoid, cylinder "
            "and convex mesh are supported"
        )
      # Canonicalize ordering matching MuJoCo C pushGeomGeom
      if t1 > t2 or (t1 == t2 and g1 > g2):
        g1, g2 = g2, g1
        t1, t2 = t2, t1
      cdim = int(model.pair_dim[p])
      if cdim not in (1, 3, 4, 6):
        raise ValueError(f"explicit pair ({g1}, {g2}) has unsupported condim {cdim}; only 1, 3, 4 and 6 are supported")

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
      solreffriction = model.pair_solreffriction[p].copy()
      if bool(solreffriction[0] > 0) != bool(solreffriction[1] > 0):
        solreffriction[:] = 0.0
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solreffriction[0] > 0:
        solreffriction[0] = max(float(solreffriction[0]), 2.0 * float(model.opt.timestep))
      friction = model.pair_friction[p].copy()

      pairs.append((g1, g2, solref, solimp, cdim, friction, solreffriction, margin, gap))
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
            "only plane, heightfield, sphere, capsule, box, ellipsoid, cylinder "
            "and convex mesh contacts are supported"
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
      if condim not in (1, 3, 4, 6):
        raise ValueError(f"contact pair ({g1}, {g2}) has unsupported condim {condim}; only 1, 3, 4 and 6 are supported")

      solref, solimp, friction = _mix_contact_parameters(model, g1, g2)
      if bool(solref[0] > 0) != bool(solref[1] > 0):
        solref = np.asarray(model.opt.o_solref).copy()
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solref[0] > 0:
        solref[0] = max(float(solref[0]), 2.0 * float(model.opt.timestep))

      margin = float(model.geom_margin[g1] + model.geom_margin[g2])
      gap = float(model.geom_gap[g1] + model.geom_gap[g2])

      friction5 = np.array([friction[0], friction[0], friction[1], friction[2], friction[2]])
      pairs.append((g1, g2, solref, solimp, condim, friction5, np.zeros(2), margin, gap))

  npairs = len(pairs)
  # 017 capacity: candidate counts flow through the explicit estimator so
  # overflow carries requested-vs-allowed detail; default limits preserve the
  # historical ceilings until the matching scalable path lands.
  from mujoco_metal.capacity import check_capacity as _check_capacity
  from mujoco_metal.capacity import estimate_capacity as _estimate_capacity

  # 3. Contact slots and offsets derivation
  pair_contact_offset = [0]
  pair_max_c = []
  total_candidate_contacts = 0
  contact_condim_list = []
  contact_friction_list = []
  contact_solreffriction_list = []
  contact_condim_packed = []

  row_offset = 0
  sdf_ip = int(model.opt.sdf_initpoints)
  for p in pairs:
    g1, g2, solref, solimp, condim, friction, solreffriction, margin, gap = p
    mc = pair_max_contacts(int(geoms[g1]), int(geoms[g2]),
                           sdf_initpoints=sdf_ip)
    pair_max_c.append(mc)
    total_candidate_contacts += mc
    pair_contact_offset.append(total_candidate_contacts)
    cone_type = int(model.opt.cone)
    rows_per_con = (
        1 if condim == 1 else
        2 * (condim - 1) if cone_type == int(mujoco.mjtCone.mjCONE_PYRAMIDAL) else
        condim
    )
    for _ in range(mc):
      contact_condim_list.append(condim)
      contact_friction_list.append(friction)
      contact_solreffriction_list.append(solreffriction)
      contact_condim_packed.extend([condim, row_offset, cone_type])
      row_offset += rows_per_con

  nr_joint = int(n_eq_rows + model.nv + 2 * model.njnt + n_ten_rows)
  nr_contact = int(row_offset)
  nr = int(nr_joint + nr_contact)
  # Single capacity gate (pairs/slots/rows/nv/batch/memory) with
  # requested-vs-allowed diagnostics; messages preserve historical text.
  # The activity array still has one entry per equality; row mapping is
  # deterministic via eq_rowadr/eq_rownum. Do not replace all uses of neq
  # indiscriminately.
  _estimate = _estimate_capacity(model, 1, npairs, total_candidate_contacts, nr)
  _check_capacity(_estimate, limits=limits)

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
    fr = np.array([p[5] for p in pairs], dtype=np.float32).reshape(-1, 5)
    srfr = np.array([p[6] for p in pairs], dtype=np.float32).reshape(-1, 2)
    margin = np.array([p[7] for p in pairs], dtype=np.float32)
    gap = np.array([p[8] for p in pairs], dtype=np.float32)
    r1 = np.where(geoms[g1] == _PLANE, -1.0, model.geom_size[g1, 0]).astype(np.float32)
    r2 = np.where(geoms[g2] == _PLANE, -1.0, model.geom_size[g2, 0]).astype(np.float32)
    p_offset = np.array(pair_contact_offset, dtype=np.int32)
    p_max_c = np.array(pair_max_c, dtype=np.int32)
    c_condim = np.array(contact_condim_list, dtype=np.int32)
    c_friction = np.array(contact_friction_list, dtype=np.float32).reshape(-1, 5)
    c_solreffriction = np.array(contact_solreffriction_list, dtype=np.float32).reshape(-1, 2)
    c_condim_packed = np.array(contact_condim_packed, dtype=np.int32)
  else:
    g1 = g2 = np.empty(0, dtype=np.int32)
    sr = np.empty((0, 2), dtype=np.float32)
    si = np.empty((0, 5), dtype=np.float32)
    cd = np.empty(0, dtype=np.int32)
    fr = np.empty((0, 5), dtype=np.float32)
    srfr = np.empty((0, 2), dtype=np.float32)
    r1 = r2 = margin = gap = np.empty(0, dtype=np.float32)
    p_offset = np.zeros(1, dtype=np.int32)
    p_max_c = np.empty(0, dtype=np.int32)
    c_condim = np.empty(0, dtype=np.int32)
    c_friction = np.empty((0, 5), dtype=np.float32)
    c_solreffriction = np.empty((0, 2), dtype=np.float32)
    c_condim_packed = np.zeros(3, dtype=np.int32)


  geom_size_mat = np.asarray(model.geom_size, dtype=np.float32).reshape(model.ngeom, 3)

  # Milestone 011: convex-mesh hull store. Mesh geoms referenced by admitted
  # pairs must be single-convex-piece assets with bounded vertex counts;
  # hull verts are geom-local (world = geom frame composed with mesh_vert,
  # calibrated against CPU contacts) and uploaded once per model.
  mesh_used = sorted({g for pr in pairs for g in (pr[0], pr[1])
                      if int(geoms[g]) == _MESH})
  # Milestone 013 SDF pre-pass: exact oct sizing + validation. SDF geoms
  # reference mesh assets whose compiled octrees (child/aabb/coeff per node)
  # are uploaded below; third-party plugin SDFs have no native path (019).
  sdf_geoms = sorted({g for pr in pairs for g in (pr[0], pr[1])
                      if int(geoms[g]) == _SDF})
  if len(sdf_geoms) > _SDF_MAX_GEOMS:
    raise ValueError(f"at most {_SDF_MAX_GEOMS} SDF geoms; found "
                     f"{len(sdf_geoms)}")
  sdf_mesh_nodes = {}
  for g in sdf_geoms:
    if int(model.geom_plugin[g]) != -1:
      raise ValueError(
          f"SDF geom {g} uses a third-party CPU plugin with no native "
          "path (see 019 extension contract)")
    mid = int(model.geom_dataid[g])
    if mid < 0:
      raise ValueError(f"SDF geom {g} has no asset")
    if mid not in sdf_mesh_nodes:
      adr = int(model.mesh_octadr[mid])
      num = int(model.mesh_octnum[mid])
      if adr < 0 or num <= 0:
        raise ValueError(f"SDF geom {g} mesh {mid} has no compiled octree")
      sdf_mesh_nodes[mid] = (adr, num)
  sdf_oct_total = sum(num for _, num in sdf_mesh_nodes.values())
  if sdf_oct_total > _SDF_MAX_NODES:
    raise ValueError(f"total SDF octree nodes {sdf_oct_total} exceeds 013 "
                     f"capacity {_SDF_MAX_NODES}")
  sdf_ip_global = int(model.opt.sdf_initpoints)
  sdf_active = any(
      (ta == _SDF or tb == _SDF) and ta not in (_PLANE, _HFIELD) and
      tb not in (_PLANE, _HFIELD)
      for ta, tb in [(int(geoms[p[0]]), int(geoms[p[1]])) for p in pairs])
  if sdf_active and sdf_ip_global > _SDF_MAX_PER_PAIR:
    raise ValueError(
        f"opt.sdf_initpoints {sdf_ip_global} exceeds native SDF budget "
        f"{_SDF_MAX_PER_PAIR} per pair; lower it (upstream default 40 "
        "emits more contacts than device slots hold)")

  mesh_hull = np.zeros((_MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS + _HF_MAX_DATA +
                      22 * sdf_oct_total + (6 * int(model.ngeom) if sdf_geoms else 0),),
                     dtype=np.float32)
  mesh_hull_info = np.full((int(model.ngeom) * 9,), -1, dtype=np.int32)
  mesh_rbound_extra = {}
  cursor = 0
  ncursor = 3 * _MESH_MAX_TOTAL
  icursor = 3 * (_MESH_MAX_TOTAL + _MESH_MAX_FACES)
  for g in mesh_used:
    mid = int(model.geom_dataid[g])
    if mid < 0:
      raise ValueError(f"mesh geom {g} has no asset")
    vnum = int(model.mesh_vertnum[mid])
    if vnum <= 0 or vnum > _MESH_MAX_VERTS:
      raise ValueError(
          f"mesh geom {g} has {vnum} verts; 011 supports 1..{_MESH_MAX_VERTS}")
    if not mesh_hull_is_convex(model, mid):
      raise ValueError(
          f"mesh geom {g} is non-convex; 011 supports single-convex-piece "
          "mesh assets only (concave collision stays rejected)")
    if cursor + vnum > _MESH_MAX_TOTAL:
      raise ValueError("total mesh hull verts exceed 011 capacity "
                       f"{_MESH_MAX_TOTAL}")
    vadr = int(model.mesh_vertadr[mid])
    verts = np.asarray(model.mesh_vert[vadr:vadr + vnum], dtype=np.float32)
    mesh_hull[3 * cursor:3 * (cursor + vnum)] = verts.reshape(-1)
    mesh_hull_info[9 * g] = cursor
    mesh_hull_info[9 * g + 1] = vnum
    mesh_rbound_extra[g] = float(np.max(np.linalg.norm(verts, axis=1)))
    cursor += vnum
    # Face-snap data: outward normals from consistent winding (verified by
    # the convexity check above: all verts satisfy every face plane).
    fadr = int(model.mesh_faceadr[mid])
    fnum = int(model.mesh_facenum[mid])
    faces = np.asarray(model.mesh_face[fadr:fadr + fnum], dtype=np.int64)
    if ncursor + 3 * fnum > 3 * (_MESH_MAX_TOTAL + _MESH_MAX_FACES) or \
        icursor + 3 * fnum > _MESH_DATA_FLOATS:
      raise ValueError("total mesh faces exceed 011 capacity "
                       f"{_MESH_MAX_FACES}")
    vd = verts.astype(np.float64)
    interior = vd.mean(axis=0)
    mesh_hull_info[9 * g + 2] = ncursor // 3
    mesh_hull_info[9 * g + 3] = icursor // 3
    mesh_hull_info[9 * g + 4] = fnum
    for f in faces:
      i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
      n = np.cross(vd[i1] - vd[i0], vd[i2] - vd[i0])
      nl = float(np.linalg.norm(n))
      if nl < 1e-12:
        n = np.zeros(3)
      else:
        n = n / nl
        if float(n @ (vd[[i0, i1, i2]].mean(axis=0) - interior)) < 0.0:
          n = -n
      mesh_face_nrm = n.astype(np.float32)
      mesh_hull[ncursor:ncursor + 3] = mesh_face_nrm
      mesh_hull[icursor:icursor + 3] = np.array(
          [i0, i1, i2], dtype=np.float32)
      ncursor += 3
      icursor += 3

  # Milestone 012: heightfield store. Per hfield geom: 4 size floats plus
  # the compiled (normalized, row-flipped) data block. Like meshes, the
  # world height is data*size2 over the grid; prism bottoms sit at -size3.
  hf_used = sorted({g for pr in pairs for g in (pr[0], pr[1])
                    if int(geoms[g]) == _HFIELD})
  if len(hf_used) > _HF_MAX_GEOMS:
    raise ValueError(f"at most {_HF_MAX_GEOMS} heightfield geoms; found "
                     f"{len(hf_used)}")
  scursor = _MESH_DATA_FLOATS
  dcursor = _MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS
  for g in hf_used:
    hid = int(model.geom_dataid[g])
    if hid < 0:
      raise ValueError(f"heightfield geom {g} has no asset")
    nrow = int(model.hfield_nrow[hid])
    ncol = int(model.hfield_ncol[hid])
    if not (2 <= nrow <= _HF_MAX_N and 2 <= ncol <= _HF_MAX_N):
      raise ValueError(f"heightfield dims {(nrow, ncol)} outside 012 "
                       f"bounds [2, {_HF_MAX_N}]")
    ndata = nrow * ncol
    if dcursor + ndata > len(mesh_hull):
      raise ValueError("total heightfield data exceeds 012 capacity "
                       f"{_HF_MAX_DATA}")
    size = np.asarray(model.hfield_size[hid], dtype=np.float32)
    mesh_hull[scursor:scursor + 4] = size
    adr = int(model.hfield_adr[hid])
    mesh_hull[dcursor:dcursor + ndata] = np.asarray(
        model.hfield_data[adr:adr + ndata], dtype=np.float32)
    mesh_hull_info[9 * g + 5] = dcursor
    mesh_hull_info[9 * g + 6] = nrow
    mesh_hull_info[9 * g + 7] = ncol
    mesh_hull_info[9 * g + 8] = scursor
    scursor += 4
    dcursor += ndata

  # Milestone 013 SDF octree + local-AABB upload. Oct blocks pack per SDF
  # mesh as child[8N] (local indices as floats, exact below 2^24), then
  # aabb[6N], then coeff[8N]; per-SDF-geom info holds child base, node
  # count and sdf_iterations. Local AABBs (compiled, center+half) upload
  # for every geom at a fixed stride for the seed-box computation.
  oct_base = _MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS + _HF_MAX_DATA
  sdf_aabb_base = oct_base + 22 * sdf_oct_total
  ocursor = oct_base
  mesh_oct_base = {}
  for mid, (adr, num) in sorted(sdf_mesh_nodes.items()):
    child = np.asarray(model.oct_child[adr:adr + num]).reshape(-1)
    aabb = np.asarray(model.oct_aabb[adr:adr + num]).reshape(-1)
    coeff = np.asarray(model.oct_coeff[adr:adr + num]).reshape(-1)
    if child.size != 8 * num or aabb.size != 6 * num or coeff.size != 8 * num:
      raise ValueError(f"SDF mesh {mid} octree layout mismatch")
    mesh_hull[ocursor:ocursor + 8 * num] = child.astype(np.float32)
    mesh_hull[ocursor + 8 * num:ocursor + 14 * num] = aabb.astype(np.float32)
    mesh_hull[ocursor + 14 * num:ocursor + 22 * num] = coeff.astype(np.float32)
    mesh_oct_base[mid] = ocursor
    ocursor += 22 * num
  sdf_iters = int(model.opt.sdf_iterations)
  if sdf_geoms and sdf_iters <= 0:
    raise ValueError("opt.sdf_iterations must be positive with SDF pairs")
  for g in sdf_geoms:
    mid = int(model.geom_dataid[g])
    mesh_hull_info[9 * g] = mesh_oct_base[mid]
    mesh_hull_info[9 * g + 1] = sdf_mesh_nodes[mid][1]
    mesh_hull_info[9 * g + 2] = sdf_iters
    mesh_hull_info[9 * g + 3] = sdf_aabb_base
  if sdf_geoms:
    for g in range(int(model.ngeom)):
      mesh_hull[sdf_aabb_base + 6 * g:sdf_aabb_base + 6 * g + 6] = np.asarray(
          model.geom_aabb[g], dtype=np.float32)

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
      adaptive_max_iterations=_ADAPTIVE_MAX_ITERATIONS,
      max_refinement_sweeps=256,
      metric="max_normalized_projected_gradient",
  )

  eq_type_arr = (
      np.asarray(model.eq_type, dtype=np.int32)
      if model.neq
      else np.zeros(0, dtype=np.int32)
  )
  eq_objtype_arr = (
      np.asarray(model.eq_objtype, dtype=np.int32)
      if model.neq
      else np.zeros(0, dtype=np.int32)
  )
  return CoupledConstraintDescriptor(
      nq=int(model.nq), nv=int(model.nv), njnt=int(model.njnt),
      neq=int(model.neq), nc=npairs, npairs=npairs, ncontacts_max=total_candidate_contacts,
      nr_joint=nr_joint, nr=nr,
      nbody=int(model.nbody), ngeom=int(model.ngeom), nsite=int(model.nsite),
      timestep=float(model.opt.timestep),
      refsafe=not bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)),
      impratio=float(model.opt.impratio),
      cone_type=int(model.opt.cone),
      disableflags=int(model.opt.disableflags),
      iterations=iter_req,
      tolerance=eff_tol,
      noslip_iterations=noslip_iters,
      noslip_tolerance=noslip_tol if noslip_iters > 0 else 0.0,
      mean_inertia=(float(getattr(getattr(model, "stat", None), "meaninertia", 1.0))
                    if math.isfinite(float(getattr(getattr(model, "stat", None), "meaninertia", 1.0)))
                    and float(getattr(getattr(model, "stat", None), "meaninertia", 1.0)) > 0
                    else 1.0),
      solver_settings=solver_settings,
      n_eq_rows=n_eq_rows,
      ntendon=int(model.ntendon),
      ten_base=ten_base,
      ten_friction_rows=ten_friction_rows,
      ten_limit_rows=ten_limit_rows,
      ten_limited=_frozen(model.tendon_limited, np.int32) if model.ntendon else np.zeros(1, dtype=np.int32),
      ten_range=_frozen(np.asarray(model.tendon_range).reshape(model.ntendon, 2) if model.ntendon else np.zeros((1, 2)), np.float32),
      ten_margin=_frozen(model.tendon_margin, np.float32) if model.ntendon else np.zeros(1, dtype=np.float32),
      ten_length0=_frozen(model.tendon_length0, np.float32) if model.ntendon else np.zeros(1, dtype=np.float32),
      ten_invweight0=_frozen(model.tendon_invweight0, np.float32) if model.ntendon else np.zeros(1, dtype=np.float32),
      ten_solref_lim=_frozen(np.asarray(model.tendon_solref_lim).reshape(model.ntendon, 2) if model.ntendon else np.zeros((1, 2)), np.float32),
      ten_solimp_lim=_frozen(np.asarray(model.tendon_solimp_lim).reshape(model.ntendon, 5) if model.ntendon else np.zeros((1, 5)), np.float32),
      ten_frictionloss=_frozen(model.tendon_frictionloss, np.float32) if model.ntendon else np.zeros(1, dtype=np.float32),
      ten_solref_fri=_frozen(np.asarray(model.tendon_solref_fri).reshape(model.ntendon, 2) if model.ntendon else np.zeros((1, 2)), np.float32),
      ten_solimp_fri=_frozen(np.asarray(model.tendon_solimp_fri).reshape(model.ntendon, 5) if model.ntendon else np.zeros((1, 5)), np.float32),
      ten_length_map=_frozen(_fixed_tendon_maps(model)[0], np.float32),
      ten_moment_map=_frozen(_fixed_tendon_maps(model)[1], np.float32),
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
      eq_type=_frozen(eq_type_arr, np.int32),
      eq_objtype=_frozen(eq_objtype_arr, np.int32),
      eq_rowadr=_frozen(eq_rowadr, np.int32),
      eq_rownum=_frozen(eq_rownum, np.int32),
      geom1=_frozen(g1, np.int32),
      geom2=_frozen(g2, np.int32),
      pair_contact_offset=_frozen(p_offset, np.int32),
      pair_max_contacts=_frozen(p_max_c, np.int32),
      contact_condim=_frozen(c_condim, np.int32),
      contact_condim_packed=_frozen(c_condim_packed, np.int32),
      contact_friction=_frozen(c_friction, np.float32),
      contact_solreffriction=_frozen(c_solreffriction, np.float32),
      radius1=_frozen(r1, np.float32),
      radius2=_frozen(r2, np.float32),

      margin=_frozen(margin, np.float32),
      gap=_frozen(gap, np.float32),
      solref=_frozen(sr, np.float32),
      solimp=_frozen(si, np.float32),
      condim=_frozen(cd, np.int32),
      friction=_frozen(fr, np.float32),
      solreffriction=_frozen(srfr, np.float32),
      geom_bodyid=_frozen(model.geom_bodyid, np.int32),
      geom_size=_frozen(geom_size_mat, np.float32),
      geom_type=_frozen(model.geom_type, np.int32),
      geom_rbound=_frozen(_rbound_with_mesh_hulls(model, mesh_rbound_extra), np.float32),
      mesh_hull=_frozen(mesh_hull, np.float32),
      mesh_hull_info=_frozen(mesh_hull_info, np.int32),
      body_parentid=_frozen(model.body_parentid, np.int32),
      body_jntadr=_frozen(model.body_jntadr, np.int32),
      body_jntnum=_frozen(model.body_jntnum, np.int32),
      body_invweight0=_frozen(model.body_invweight0.reshape(-1, 2), np.float32),
      dense_path=bool(_estimate.dense_path),
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

  def __init__(self, model, batch_size=1, limits=None):
    from mujoco_metal.capacity import CapacityLimits
    self.limits = limits or CapacityLimits()
    self.descriptor = lower_coupled_constraints(model, limits=self.limits)
    self._mjmodel_ref = model
    self.solver_settings = self.descriptor.solver_settings
    self.batch_size = int(batch_size)
    if self.batch_size <= 0:
      raise ValueError("batch_size must be positive")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("native coupled constraints require PyTorch MPS compile_shader")
    self._torch = torch
    self._device = torch.device("mps")

    shader_source = (_COLLISION_SHADER.read_text() + "\n" + _CONVEX_SHADER.read_text()
                     + "\n" + _SDF_SHADER.read_text()
                     + "\n" + _EQUALITY_SHADER.read_text() + "\n" + _SHADER.read_text())
    self._library = torch.mps.compile_shader(shader_source)
    self._contact_kernel = self._library.contact_normal
    try:
      self._equality_kernel = self._library.equality_assembly
    except AttributeError:
      self._equality_kernel = None
    try:
      self._tendon_kernel = self._library.tendon_constraint_rows
    except AttributeError:
      self._tendon_kernel = None
    self._solve_kernel = self._library.solve_coupled_constraints
    try:
      self._solve_block_kernel = self._library.solve_coupled_constraints_block
    except AttributeError:
      self._solve_block_kernel = self._solve_kernel
    self._broadphase_lib = torch.mps.compile_shader(_BROADPHASE_SHADER.read_text())
    self._broadphase_kernel = self._broadphase_lib.broadphase_mask

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
        else np.zeros(5, dtype=np.float32)
    )
    contact_solreffriction_arr = (
        d.contact_solreffriction.reshape(-1)
        if d.ncontacts_max else np.zeros(2, dtype=np.float32)
    )
    contact_condim_arr = (
        d.contact_condim
        if d.ncontacts_max
        else np.zeros(1, dtype=np.int32)
    )

    self._constants = {
        "joint_qposadr": self._tensor(d.joint_qposadr if d.njnt else np.zeros(1, dtype=np.int32)),
        "qpos0": self._tensor(d.qpos0 if d.nq else np.zeros(1, dtype=np.float32)),
        "joint_dofadr": self._tensor(d.joint_dofadr if d.njnt else np.zeros(1, dtype=np.int32)),
        "joint_limited": self._tensor(
            (np.asarray(d.joint_limited, dtype=np.int32)
             | (np.asarray(d.joint_type, dtype=np.int32) << 1)).astype(np.uint8)
            if d.njnt else np.zeros(1, dtype=np.uint8)),
        "joint_limit_params": self._tensor(joint_limit_params.reshape(-1)),
        "joint_sol_params": self._tensor(d.joint_sol_params.reshape(-1)),
        "dof_frictionloss": self._tensor(d.dof_frictionloss if d.nv else np.zeros(1, dtype=np.float32)),
        "dof_invweight0": self._tensor(d.dof_invweight0 if d.nv else np.zeros(1, dtype=np.float32)),
        "dof_sol_params": self._tensor(d.dof_sol_params.reshape(-1)),
        "eq_obj": self._tensor(d.eq_obj.reshape(-1) if d.neq else np.zeros(2, dtype=np.int32)),
        "eq_data": self._tensor(d.eq_data.reshape(-1) if d.neq else np.zeros(11, dtype=np.float32)),
        "eq_sol_params": self._tensor(d.eq_sol_params.reshape(-1) if d.neq else np.zeros(7, dtype=np.float32)),
        "eq_type": self._tensor(d.eq_type if d.neq else np.zeros(1, dtype=np.int32)),
        "eq_objtype": self._tensor(d.eq_objtype if d.neq else np.zeros(1, dtype=np.int32)),
        "eq_rowadr": self._tensor(d.eq_rowadr if d.neq else np.zeros(1, dtype=np.int32)),
        "geom_size": self._tensor(d.geom_size.reshape(-1) if d.ngeom else np.zeros(3, dtype=np.float32)),
        "geom_type": self._tensor(d.geom_type if d.ngeom else np.zeros(1, dtype=np.int32)),
        "geom_rbound": self._tensor(d.geom_rbound if d.ngeom else np.zeros(1, dtype=np.float32)),
        "mesh_hull": self._tensor(d.mesh_hull),
        "mesh_hull_info": self._tensor(d.mesh_hull_info.astype(np.int32)),
        "geom_bodyid": self._tensor(d.geom_bodyid if d.ngeom else np.zeros(1, dtype=np.int32)),
        "body_parentid": self._tensor(d.body_parentid),
        "body_jntadr": self._tensor(d.body_jntadr),
        "body_jntnum": self._tensor(d.body_jntnum),
        "jnt_type": self._tensor(d.joint_type if d.njnt else np.zeros(1, dtype=np.int32)),
        "jnt_dofadr": self._tensor(d.joint_dofadr if d.njnt else np.zeros(1, dtype=np.int32)),
        "body_invweight0": self._tensor(d.body_invweight0.reshape(-1)),
        "site_bodyid": self._tensor(
            np.asarray(model.site_bodyid, dtype=np.int32) if model.nsite else np.zeros(1, dtype=np.int32)
        ),
        "body_rootid": self._tensor(np.asarray(model.body_rootid, dtype=np.int32)),
        "body_weldid": self._tensor(np.asarray(model.body_weldid, dtype=np.int32)),
        "body_dofadr": self._tensor(np.asarray(model.body_dofadr, dtype=np.int32)),
        "body_dofnum": self._tensor(np.asarray(model.body_dofnum, dtype=np.int32)),
        "dof_parentid": self._tensor(
            np.asarray(model.dof_parentid, dtype=np.int32) if model.nv else np.zeros(1, dtype=np.int32)
        ),
        "dof_bodyid": self._tensor(
            np.asarray(model.dof_bodyid, dtype=np.int32) if model.nv else np.zeros(1, dtype=np.int32)
        ),
        "dof_jntid": self._tensor(
            np.asarray(model.dof_jntid, dtype=np.int32) if model.nv else np.zeros(1, dtype=np.int32)
        ),
        "pair_geoms": self._tensor(pair_geoms),
        "pair_margin_gap": self._tensor(pair_margin_gap),
        "pair_solref": self._tensor(d.solref.reshape(-1) if d.npairs else np.zeros(2, dtype=np.float32)),
        "pair_solimp": self._tensor(d.solimp.reshape(-1) if d.npairs else np.zeros(5, dtype=np.float32)),
        "pair_condim": self._tensor(d.condim if d.npairs else np.zeros(1, dtype=np.int32)),
        "pair_friction": self._tensor(d.friction.reshape(-1) if d.npairs else np.zeros(5, dtype=np.float32)),
        "pair_solreffriction": self._tensor(d.solreffriction.reshape(-1) if d.npairs else np.zeros(2, dtype=np.float32)),
        "pair_contact_offset": self._tensor(d.pair_contact_offset),
        "contact_friction": self._tensor(contact_friction_arr),
        "contact_solreffriction": self._tensor(contact_solreffriction_arr),
        "contact_condim": self._tensor(d.contact_condim_packed if d.ncontacts_max else np.zeros(3, dtype=np.int32)),
        "ten_limited": self._tensor(d.ten_limited if d.ntendon else np.zeros(1, dtype=np.int32)),
        "ten_range": self._tensor(d.ten_range.reshape(-1) if d.ntendon else np.zeros(2, dtype=np.float32)),
        "ten_margin": self._tensor(d.ten_margin if d.ntendon else np.zeros(1, dtype=np.float32)),
        "ten_length0": self._tensor(d.ten_length0 if d.ntendon else np.zeros(1, dtype=np.float32)),
        "ten_invweight0": self._tensor(d.ten_invweight0 if d.ntendon else np.zeros(1, dtype=np.float32)),
        "ten_solref_lim": self._tensor(d.ten_solref_lim.reshape(-1) if d.ntendon else np.zeros(2, dtype=np.float32)),
        "ten_solimp_lim": self._tensor(d.ten_solimp_lim.reshape(-1) if d.ntendon else np.zeros(5, dtype=np.float32)),
        "ten_frictionloss": self._tensor(d.ten_frictionloss if d.ntendon else np.zeros(1, dtype=np.float32)),
        "ten_solref_fri": self._tensor(d.ten_solref_fri.reshape(-1) if d.ntendon else np.zeros(2, dtype=np.float32)),
        "ten_solimp_fri": self._tensor(d.ten_solimp_fri.reshape(-1) if d.ntendon else np.zeros(5, dtype=np.float32)),
        "ten_length_map": self._tensor(d.ten_length_map.reshape(-1)),
        "ten_moment_map": self._tensor(d.ten_moment_map.reshape(-1)),
        "c_dims": torch.tensor(

            [d.nv, d.npairs, d.ncontacts_max, self.batch_size, d.nbody, d.njnt, d.ngeom, d.cone_type,
             1 if (int(d.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)) else 0,
             # Tail index 9 toggles broadphase-fed narrowphase pruning
             # (R08/F1); earlier indices are frozen for every consumer.
             1],
            dtype=torch.int32, device=self._device,
        ),
        "body_dims": torch.tensor(
            [d.ncontacts_max, d.npairs, d.nv, 0, self.batch_size],
            dtype=torch.int32, device=self._device,
        ),
        "solver_dims": torch.tensor(
            # Appended tail carries the no-slip budget (R04); earlier
            # indices are frozen for every consumer of this layout.
            [d.nq, d.nv, d.njnt, d.neq, d.ncontacts_max, self.batch_size, d.disableflags, 1 if d.refsafe else 0, d.iterations, d.nr, d.cone_type, d.nbody, d.njnt, d.nsite, d.n_eq_rows, d.ntendon, d.ten_base, d.ten_friction_rows + d.ten_limit_rows, d.noslip_iterations],
            dtype=torch.int32, device=self._device,
        ),
        "solver_params": torch.tensor(
            [d.timestep, d.impratio, d.tolerance, d.noslip_tolerance, d.mean_inertia],
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
    # R08a allocation gate: enforce the real batch and the exact owned
    # workspace BEFORE partial allocation (lowering-time checks used
    # batch=1 and an approximate memory model).
    from mujoco_metal.capacity import check_capacity as _check_capacity
    from mujoco_metal.capacity import estimate_capacity as _estimate_capacity
    _check_capacity(_estimate_capacity(
        self._mjmodel_ref, b, d.npairs, d.ncontacts_max, d.nr,
        neq=d.neq, nr_joint=d.nr_joint), limits=self.limits)
    self.batch_size = b
    nv, nc, nr = d.nv, d.ncontacts_max, d.nr

    def empty(size):
      return torch.zeros(max(size, 1), dtype=torch.float32, device=self._device)

    self._workspace = {
        "contact_row_data": empty(b * nc * 6 * 6),
        "contact_frame": empty(b * nc * 12),
        "contact_jacobian": empty(b * nc * 6 * nv),
        "pair_mask": torch.zeros(max(b * max(d.npairs, 1), 1), dtype=torch.float32, device=self._device),
        "workspace_J": empty(b * nr * nv),
        "workspace_debug": empty(b * (nr * nr + 7 * nr)),
        "out_force": empty(b * nv),
        "out_acc": empty(b * nv),
        "out_status": torch.zeros(b, dtype=torch.int32, device=self._device),
        "out_diagnostics": empty(b * 10),
        "out_contact_force": empty(b * nc * 11),
        "out_joint_force": empty(b * max(d.nr_joint, 1)),
    }
    if d.neq > 0:
      init_eq = np.broadcast_to(d.eq_active0.astype(np.int32), (b, d.neq)).copy()
    else:
      init_eq = np.zeros((b, 1), dtype=np.int32)
    self._eq_active_default = torch.as_tensor(init_eq, dtype=torch.int32, device=self._device)
    self._constants["c_dims"][3] = b
    self._constants["solver_dims"][5] = b
    self._constants["body_dims"][3] = 0
    self._constants["body_dims"][4] = b
    return self._workspace

  @property
  def warm_size(self):
    """Number of retained constraint multipliers per world (nr, maybe 0)."""
    return int(self.descriptor.nr)

  def _warm_rows(self, env_ids=None):
    import numpy as _np
    b, nr = int(self.batch_size), int(self.descriptor.nr)
    if env_ids is None:
      return list(range(b))
    ids = _np.asarray(env_ids).reshape(-1)
    if ids.size == 0:
      raise ValueError("env_ids must select at least one world")
    out = []
    for v in ids.tolist():
      if isinstance(v, bool) or int(v) != v or not 0 <= int(v) < b:
        raise ValueError(f"env id {v!r} out of range for batch {b}")
      out.append(int(v))
    if len(set(out)) != len(out):
      raise ValueError("duplicate env ids are not allowed")
    return out

  def get_warmstart(self):
    """Return a host copy of retained multipliers, shape (batch, nr)."""
    import numpy as _np
    b, nr = int(self.batch_size), int(self.descriptor.nr)
    if nr == 0:
      return _np.zeros((b, 0), dtype=_np.float32)
    w = self._workspace["workspace_debug"].reshape(b, nr * nr + 7 * nr)
    return w[:, nr * nr + 3 * nr:nr * nr + 4 * nr].detach().cpu().numpy().copy()

  def set_warmstart(self, values, env_ids=None):
    """Store retained multipliers for selected worlds (validated, atomic).

    `values` accepts host arrays shaped `(nr,)` (broadcast) or
    `(len(env_ids), nr)`, finite float32-representable. Bad input leaves all
    worlds unchanged. Takes effect on the next solve; the cost check still
    rejects vectors that lose to a cold start.
    """
    import numpy as _np
    torch = self._torch
    b, nr = int(self.batch_size), int(self.descriptor.nr)
    ids = self._warm_rows(env_ids)
    arr = _np.asarray(values, dtype=_np.float64)
    if nr == 0:
      if arr.size != 0:
        raise ValueError("model has no constraint rows")
      return
    if arr.shape == (nr,):
      arr = _np.broadcast_to(arr, (len(ids), nr)).copy()
    if arr.shape != (len(ids), nr):
      raise ValueError(
          f"warmstart must have shape ({nr},) or ({len(ids)}, {nr}); "
          f"got {arr.shape}")
    if not _np.all(_np.isfinite(arr)):
      raise ValueError("warmstart must be finite")
    with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
      arr32 = _np.asarray(arr, dtype=_np.float32)
    if not _np.all(_np.isfinite(arr32)):
      raise ValueError("warmstart must be float32-representable")
    w = self._workspace["workspace_debug"].reshape(b, nr * nr + 7 * nr)
    w[:, nr * nr + 3 * nr:nr * nr + 4 * nr][ids] = torch.as_tensor(
        arr32, dtype=torch.float32, device=self._device)
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None

  def clear_warmstart(self, env_ids=None):
    """Zero retained multipliers (cold start) for selected worlds."""
    b, nr = int(self.batch_size), int(self.descriptor.nr)
    if nr == 0:
      self._warm_rows(env_ids)
      return
    ids = self._warm_rows(env_ids)
    w = self._workspace["workspace_debug"].reshape(b, nr * nr + 7 * nr)
    w[:, nr * nr + 3 * nr:nr * nr + 4 * nr][ids] = 0.0
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None

  def generate_candidates(self, poses, qvel):
    """Run only the contact candidate kernel into persistent workspace buffers.

    Used for same-step BODY-transmission adhesion moments (milestone 007),
    which require candidate contacts before actuator-force assembly. The
    buffers are zeroed first so unwritten slots read exact zero. `run_device`
    calls this internally; behavior is unchanged.
    """
    w, d = self._workspace, self.descriptor
    b, nc = self.batch_size, d.ncontacts_max
    if d.npairs > 0 and nc > 0:
      w["contact_row_data"].zero_()
      w["contact_frame"].zero_()
      w["contact_jacobian"].zero_()
      w["pair_mask"].zero_()
      self._broadphase_kernel(
          poses["geom_pos"].reshape(-1),
          self._constants["geom_rbound"],
          self._constants["pair_geoms"],
          self._constants["pair_margin_gap"],
          self._constants["geom_type"],
          w["pair_mask"],
          self._constants["c_dims"],
          threads=(b * d.npairs,), group_size=(1,),
      )
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
          self._constants["pair_solreffriction"], self._constants["pair_contact_offset"],
          w["contact_row_data"], w["contact_frame"], w["contact_jacobian"],
          self._constants["c_dims"], self._constants["geom_rbound"],
          self._constants["mesh_hull"], self._constants["mesh_hull_info"],
          threads=(b * d.npairs,), group_size=(1,),
      )

  def set_broadphase_pruning(self, enabled):
    """Toggle narrowphase broadphase-fed pruning (R08/F1 test hook).

    Enabled (default) pruned pairs skip the narrowphase and stamp a finite
    sentinel; disabled runs every pair. Physics must be identical either
    way (equivalence test).
    """
    self._constants["c_dims"][9] = 1 if enabled else 0

  def broadphase_counts(self):
    """Diagnostics-only readback: per-env (mask_active, slot_active) counts.

    `mask_active` counts broadphase sphere-overlap pairs; `slot_active`
    counts narrowphase slots with solver rows. Every slot-active pair must
    be mask-active (conservative superset). Host readback happens here only,
    never inside the step loop.
    """
    w, d = self._workspace, self.descriptor
    b, npairs, nc = self.batch_size, d.npairs, d.ncontacts_max
    mask = w["pair_mask"][: b * max(npairs, 1)].detach().cpu().numpy().reshape(b, max(npairs, 1))
    rows = w["contact_row_data"][: b * max(nc, 1) * 36].detach().cpu().numpy().reshape(b, max(nc, 1), 6, 6)
    out = []
    for env in range(b):
      m = int(np.sum(mask[env, :npairs] > 0.5)) if npairs else 0
      s = int(np.sum(rows[env, :nc, 0, 0] > 0.5)) if nc else 0
      out.append((m, s))
    return out

  def active_counts(self):
    """Return `(mask_active, slot_active, active_rows)` candidate counts per world."""
    counts = self.broadphase_counts()
    d = self.descriptor
    out = []
    cone_type = int(self._mjmodel_ref.opt.cone)
    rows_per_con = 4 if cone_type == 0 else 3
    for w in range(self.batch_size):
      mask_act, slot_act = counts[w]
      act_rows = min(d.nr, d.nr_joint + slot_act * rows_per_con)
      out.append((mask_act, slot_act, act_rows))
    return out

  def candidate_compaction_metrics(self):
    """Return telemetry demonstrating numerical work avoidance under compaction."""
    d = self.descriptor
    counts = self.active_counts()
    metrics = []
    for w in range(self.batch_size):
      mask_act, slot_act, act_rows = counts[w]
      alloc_rows = d.nr
      work_ratio = float(act_rows) / float(max(alloc_rows, 1))
      metrics.append({
          "allocated_slots": d.ncontacts_max,
          "active_slots": slot_act,
          "allocated_rows": alloc_rows,
          "active_rows": act_rows,
          "dense_path": d.dense_path,
          "work_ratio": work_ratio,
      })
    return metrics

  def contact_buffers(self):
    """Return borrowed candidate-contact workspace views for BODY adhesion."""
    w, d = self._workspace, self.descriptor
    b, nc, nv = self.batch_size, d.ncontacts_max, max(d.nv, 1)
    import torch as _torch
    zeros = _torch.zeros(1, dtype=_torch.float32, device=self._device)
    frame = w["contact_frame"] if nc else zeros
    jacobian = w["contact_jacobian"] if nc else zeros
    return {
        "frame": frame,
        "jacobian": jacobian,
        "pair_geoms": self._constants["pair_geoms"],
        "geom_bodyid": self._constants["geom_bodyid"],
        "pair_offset": self._constants["pair_contact_offset"],
        "dims": self._constants["body_dims"],
    }

  def run_device(self, poses, mass, qfrc_smooth, qpos, qvel, eq_active=None, cvel=None,
                 tendon_J_spatial=None, tendon_length_spatial=None):
    """Solve coupled contacts, limits, dry friction, and equalities on MPS.

    `poses` is the dict of FK outputs from `MetalKinematics`.
    `mass` is batched generalized mass [batch, nv, nv] (including tendon armature).
    `qfrc_smooth` is unconstrained forces [batch, nv].
    `qpos` is [batch, nq], `qvel` is [batch, nv].
    `cvel` is optional batched body spatial velocity [batch, nbody, 6] from
    smooth dynamics (angular, linear COM). It is required for connect/weld
    Jdot correction; joint-only models may omit it (Jdot is zero there).
    `tendon_J_spatial`/`tendon_length_spatial` are borrowed [batch, nt(, nv)]
    spatial tendon views from the simulation's once-per-assembly kinematics
    (None when the model has no spatial tendons; the stage substitutes zeros).
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
    self.generate_candidates(poses, qvel)

    # 1b. Equality assembly kernel (joint/connect; weld reserved as zeros).
    # Writes equality J into workspace_J rows [0, n_eq_rows) and R/ar into
    # workspace_debug R/ar slots for those rows. Solver preserves these
    # sections and treats all equality rows as bilateral.
    if d.neq > 0 and d.n_eq_rows > 0 and self._equality_kernel is not None:
      if cvel is None:
        cvel_tensor = torch.zeros((b, max(d.nbody, 1) * 6), dtype=torch.float32, device=self._device)
        # Reshape to [b, nbody, 6] with dummy when nbody==0 (never, nbody>=1).
        try:
          cvel_flat = cvel_tensor.reshape(-1)
        except Exception:
          cvel_flat = torch.zeros(max(b * max(d.nbody, 1) * 6, 1), dtype=torch.float32, device=self._device)
      else:
        if not isinstance(cvel, torch.Tensor):
          raise TypeError("cvel must be a torch.Tensor")
        cvel_flat = cvel.reshape(-1)
      # site_pos/site_quat may be empty when nsite==0; pass dummy non-null buffers.
      try:
        site_pos_tensor = poses["site_pos"]
      except KeyError:
        site_pos_tensor = torch.zeros((b, 1, 3), dtype=torch.float32, device=self._device)
      try:
        site_quat_tensor = poses["site_quat"]
      except KeyError:
        site_quat_tensor = torch.zeros((b, 1, 4), dtype=torch.float32, device=self._device)
      self._equality_kernel(
          self._constants["eq_obj"], self._constants["eq_data"],
          self._constants["eq_sol_params"], eq_active_tensor.reshape(-1),
          self._constants["eq_rowadr"],
          self._constants["eq_type"], self._constants["eq_objtype"],
          self._constants["joint_qposadr"], self._constants["qpos0"],
          self._constants["joint_dofadr"],
          self._constants["dof_invweight0"], self._constants["body_invweight0"],
          self._constants["site_bodyid"],
          self._constants["body_parentid"], self._constants["body_jntadr"],
          self._constants["body_jntnum"], self._constants["jnt_type"],
          self._constants["jnt_dofadr"],
          self._constants["solver_dims"], self._constants["solver_params"],
          qpos.reshape(-1), qvel.reshape(-1),
          poses["body_pos"].reshape(-1), poses["body_quat"].reshape(-1),
          poses["joint_anchor"].reshape(-1), poses["joint_axis"].reshape(-1),
          site_pos_tensor.reshape(-1), cvel_flat,
          w["workspace_J"], w["workspace_debug"],
          site_quat_tensor.reshape(-1),
          threads=(b,), group_size=(1,),
      )

    # 1b. Tendon constraint rows (limits, friction loss, tendon equalities).
    # Runs after equality assembly (which reserves zeros for tendon
    # equalities) and before the coupled solve (which reads flagged rows).
    if d.ntendon and (d.ten_friction_rows + d.ten_limit_rows > 0
                      or bool(np.any(np.asarray(d.eq_type) == 3))):
      if self._tendon_kernel is None:
        raise ValueError("tendon rows require the tendon_constraint_rows kernel")
      nt, nq = d.ntendon, d.nq
      if tendon_J_spatial is None:
        ten_J = torch.zeros((b * max(nt, 1) * max(nv, 1),), dtype=torch.float32, device=self._device)
      else:
        if tuple(tendon_J_spatial.shape) != (b, nt, max(nv, 1)):
          raise ValueError(f"tendon_J_spatial must have shape {(b, nt, max(nv, 1))}")
        ten_J = tendon_J_spatial.reshape(-1)
      if tendon_length_spatial is None:
        ten_L = torch.zeros((b * max(nt, 1),), dtype=torch.float32, device=self._device)
      else:
        if tuple(tendon_length_spatial.shape) != (b, max(nt, 1)):
          raise ValueError(f"tendon_length_spatial must have shape {(b, max(nt, 1))}")
        ten_L = tendon_length_spatial.reshape(-1)
      self._tendon_kernel(
          qpos.reshape(-1), qvel.reshape(-1), ten_J, ten_L,
          self._constants["ten_length_map"], self._constants["ten_moment_map"],
          self._constants["ten_limited"], self._constants["ten_range"],
          self._constants["ten_margin"], self._constants["ten_length0"],
          self._constants["ten_invweight0"],
          self._constants["ten_solref_lim"], self._constants["ten_solimp_lim"],
          self._constants["ten_frictionloss"],
          self._constants["ten_solref_fri"], self._constants["ten_solimp_fri"],
          self._constants["eq_type"], self._constants["eq_obj"],
          self._constants["eq_data"], self._constants["eq_sol_params"],
          self._constants["eq_rowadr"], eq_active_tensor.reshape(-1),
          self._constants["solver_dims"], self._constants["solver_params"],
          w["workspace_J"], w["workspace_debug"],
          threads=(b,), group_size=(1,),
      )

    # 2. Coupled constraint solver kernel
    solve_fn = self._solve_kernel if self.descriptor.dense_path else self._solve_block_kernel
    solve_fn(
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
    diagnostics = w["out_diagnostics"][: b * 10].reshape(b, 10)
    history = diagnostics[:, 2:]
    diagnostics = diagnostics[:, :2]

    result = {
        "qacc": qacc,
        "qfrc_constraint": qfrc_constraint,
        "status": status,
        "solver_diagnostics": diagnostics,
        "solver_history": history,
    }

    if nr > 0:
      w_debug = w["workspace_debug"][: b * (nr * nr + 7 * nr)].reshape(b, nr * nr + 7 * nr)
      W = w_debug[:, : nr * nr].reshape(b, nr, nr)
      R = w_debug[:, nr * nr : nr * nr + nr].reshape(b, nr)
      ar = w_debug[:, nr * nr + nr : nr * nr + 2 * nr].reshape(b, nr)
      rhs = w_debug[:, nr * nr + 2 * nr : nr * nr + 3 * nr].reshape(b, nr)
      lam = w_debug[:, nr * nr + 3 * nr : nr * nr + 4 * nr].reshape(b, nr)
      lo = w_debug[:, nr * nr + 4 * nr : nr * nr + 5 * nr].reshape(b, nr)
      hi = w_debug[:, nr * nr + 5 * nr : nr * nr + 6 * nr].reshape(b, nr)
      J = w["workspace_J"][: b * nr * nv].reshape(b, nr, nv)
      result.update({
          "J": J,
          "W": W,
          "W_regularized": W + torch.diag_embed(R),
          "R": R,
          "ar": ar,
          "rhs": rhs,
          "lambda": lam,
          "lo": lo,
          "hi": hi,
      })

    if nc > 0:
      contact_rows = w["contact_row_data"][: b * nc * 6 * 6].reshape(b, nc, 6, 6)
      contact_frames = w["contact_frame"][: b * nc * 12].reshape(b, nc, 12)
      contact_forces = w["out_contact_force"][: b * nc * 11].reshape(b, nc, 11)
      contact_jacobians = w["contact_jacobian"][: b * nc * 6 * nv].reshape(b, nc, 6, nv)
      friction_tensor = self._constants["contact_friction"].reshape(nc, 5)
      detected = contact_rows[:, :, 0, 0]
      if d.cone_type == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        contact_wrench = contact_forces[:, :, :6]
      else:
        wrench_axes = [contact_forces[:, :, 0]]
        for axis in range(5):
          plus = contact_forces[:, :, 1 + 2 * axis]
          minus = contact_forces[:, :, 2 + 2 * axis]
          wrench_axes.append(friction_tensor[None, :, axis] * (plus - minus))
        contact_wrench = torch.stack(wrench_axes, dim=-1)

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
          "contact_force_rows": contact_forces[:, :, :5],
          "contact_solver_rows": contact_forces,
          "contact_wrench": contact_wrench,
          "contact_jacobian": contact_jacobians,
      })

    if d.nr_joint > 0:
      joint_forces = w["out_joint_force"][: b * d.nr_joint].reshape(b, d.nr_joint)
      result["joint_force"] = joint_forces

    return result
