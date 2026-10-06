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

from dataclasses import dataclass, replace
from contextlib import contextmanager, nullcontext
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
_COMMON_CCD_CORE_SHADER = Path(__file__).parent / "shaders" / "common_ccd_core.metal"
_COMMON_CCD_SUPPORT_SHADER = Path(__file__).parent / "shaders" / "common_native_ccd_support.metal"
_COMMON_CCD_PRODUCTION_SHADER = Path(__file__).parent / "shaders" / "common_ccd_production.metal"


def _expand_flex_equality_reference_coefficients(k0, b0, impedance):
  """Broadcast source flex-equality coefficients over their canonical rows.

  FLEXVERT/FLEXSTRAIN producers provide scalar 0-D coefficients, while
  FLEX edge equalities provide one coefficient per row. Both represent raw
  source K/B; impedance affects the position term at use time, not these
  transported coefficients.
  """
  def broadcast(value):
    # FLEXVERT/FLEXSTRAIN may lower a shared coefficient as a Python scalar,
    # while FLEX edges can supply a row tensor. Normalize scalar and tensor
    # inputs to the runtime row tensor's dtype/device before broadcasting.
    if isinstance(value, type(impedance)):
      value = value.to(device=impedance.device, dtype=impedance.dtype)
    else:
      value = impedance.new_tensor(value)
    return value.expand_as(impedance)

  return broadcast(k0), broadcast(b0)
_PLUGIN_SDF_SHADER = Path(__file__).parent / "shaders" / "plugin_sdf.metal"
_SDF_STAGE_STATE_WORDS = 159


def _coupled_shader_source(*, expected_root=None):
  """Build the exact production unit, also used by native solver witnesses.

  An expected root verifies source ownership; it never redirects compilation
  to another checkout's shader files.
  """
  if (expected_root is not None
      and Path(expected_root).resolve() != _SHADER.parent.resolve()):
    raise AssertionError("coupled shader unit resolves outside the expected source root")
  return (
      _COLLISION_SHADER.read_text() + "\n"
      + _COMMON_CCD_CORE_SHADER.read_text() + "\n"
      + _COMMON_CCD_SUPPORT_SHADER.read_text() + "\n"
      + _CONVEX_SHADER.read_text() + "\n"
      + _PLUGIN_SDF_SHADER.read_text() + "\n"
      + _SDF_SHADER.read_text() + "\n"
      + _COMMON_CCD_PRODUCTION_SHADER.read_text()
      + "\n#define MUJOCO_METAL_EQUALITY_HELPERS_ONLY\n"
      + _EQUALITY_SHADER.read_text()
      + "\n#undef MUJOCO_METAL_EQUALITY_HELPERS_ONLY\n"
      + _SHADER.read_text())


def _pair_contact_cache_bytes(
    batch: int, ncontacts: int, npairs: int, sdf_seed_tile: int = 0) -> int:
  """Exact device storage for output, mesh-SDF, and analytic-SDF scratch."""
  count_words = max(int(batch) * int(npairs), 1)
  record_words = max(int(batch) * int(ncontacts) * 25, 1)
  # Preserve the pinned mesh-SDF candidate order without putting candidates
  # into output slots before deduplication. Each of 50 candidates has a
  # three-word point residual and three-word depth.
  candidate_words = max(int(batch) * int(npairs) * 50 * 12, 1)
  # Analytic/SDF pairs reuse a fixed per-start tile across the complete source
  # initpoint budget. Each tile slot stores a 25-word result, 159 words of
  # three-float staged descent state, and two int control/validity words.
  sdf_seed_words = (int(batch) * int(npairs) * int(sdf_seed_tile)
                    * (25 + _SDF_STAGE_STATE_WORDS + 2)
                    if sdf_seed_tile else 4)
  # Five int32 stage dimensions are allocated even for models with no SDF;
  # all four per-seed arrays also retain one-element dummies in that case.
  stage_dim_words = 5
  return 4 * (count_words + record_words + candidate_words + sdf_seed_words
              + stage_dim_words)


@contextmanager
def _solver_dimension_stage(solver_dims, stage, restore_value):
  """Temporarily select an internal dispatch stage, restoring on every exit."""
  solver_dims[20] = stage
  try:
    yield
  finally:
    solver_dims[20] = restore_value


@contextmanager
def _cached_position_stage(solver_dims, refsafe, trace_enabled=False,
                           detail_trace_enabled=False, detail_trace_flags=0):
  """Mark cached solve while preserving the optional trace observer bit."""
  original = (int(bool(refsafe)) | (32 if trace_enabled else 0)
              | ((64 | (int(detail_trace_flags) & 384))
                 if detail_trace_enabled else 0))
  solver_dims[7] = original | 2
  try:
    yield
  finally:
    solver_dims[7] = original


@contextmanager
def _provided_smooth_acceleration_stage(solver_dims):
  """Tell the solve kernel its force input already contains qacc_smooth."""
  original = int(solver_dims[7])
  solver_dims[7] = original | 4
  try:
    yield
  finally:
    solver_dims[7] = original


@contextmanager
def _paired_smooth_input_stage(solver_dims):
  """Mark dense coupled solves to read the smooth-force high/low planes."""
  original = int(solver_dims[7])
  solver_dims[7] = original | 16
  try:
    yield
  finally:
    solver_dims[7] = original


@contextmanager
def _preassembled_rows_stage(solver_dims):
  """Reuse canonical rows prepared by the immediately preceding dispatch."""
  original = int(solver_dims[7])
  solver_dims[7] = original | 8
  try:
    yield
  finally:
    solver_dims[7] = original


def _merge_exact_world_status(torch, out_status, exact_data):
  """Merge exact-mass status from its owned per-world result buffer.

  ``mass_status`` in ``exact_data`` may alias component-factor scratch. The
  following acceleration solve reuses that scratch, whereas ``status`` is the
  impedance helper's persistent, world-shaped summary.
  """
  if exact_data is not None:
    torch.maximum(out_status, exact_data["status"], out=out_status)


def _run_component_solve_after_prepare(prepare, solve):
  """Run borrowed-buffer preparation before the component solve overwrites it.

  Exact impedance assembly may use the component solver's persistent RHS/result
  backing for its mass-row solve. Keep this ordering explicit at both public
  solve entry points so the smooth/contact solve is the final writer before its
  result is dispatched.
  """
  prepared = prepare()
  solved = solve()
  return prepared, solved


@contextmanager
def _component_primal_stage(solver_dims, refsafe, stage, line_search_iterations,
                            provided_smooth_acceleration, trace_enabled=False,
                            detail_trace_enabled=False, detail_trace_flags=0):
  """Compose the cached component-primal ABI flags for one dispatch."""
  acceleration = (_provided_smooth_acceleration_stage(solver_dims)
                  if provided_smooth_acceleration else nullcontext())
  with (_cached_position_stage(solver_dims, refsafe, trace_enabled,
                               detail_trace_enabled, detail_trace_flags),
        _solver_dimension_stage(solver_dims, stage, line_search_iterations),
        acceleration):
    yield


def _clear_current_row_ownership(workspace_debug, batch, nr, stride):
  """Clear preassembly ownership before producers publish this solve's rows.

  The final seven-row debug prefix includes an activity/ownership bitmap.
  Tendon and flex row producers use it to mark rows they have preassembled;
  the solver also uses it to tell those rows from dynamically generated
  contacts. It is therefore current-stage metadata, unlike the retained
  multiplier slice, and must not carry contact activity across assemblies.
  """
  view = workspace_debug.reshape(int(batch), int(stride))
  start = int(nr) * int(nr) + 6 * int(nr)
  view[:, start:start + int(nr)] = 0


def _clear_current_aref_low(workspace_debug, batch, nr, stride, offset):
  """Reset low-word RHS corrections before any fresh row producer runs.

  The equality kernel initializes this region too, but it is skipped for
  models without rigid equality rows. Contact, flex, and other producers then
  intentionally leave their correction at zero. Resetting here makes that
  invariant hold when the device workspace is reused across assemblies.
  """
  if int(nr) <= 0:
    return
  view = workspace_debug.reshape(int(batch), int(stride))
  start = int(offset)
  stop = start + int(nr)
  if start < 0 or stop > int(stride):
    raise ValueError("aref-low workspace interval is outside the debug stride")
  view[:, start:stop].zero_()


def _validate_cached_tensor(torch, device, name, value, shape, dtype):
  """Validate one borrowed cached-stage input before touching workspaces."""
  if not isinstance(value, torch.Tensor):
    raise ValueError(f"{name} must be a torch.Tensor")
  if tuple(value.shape) != tuple(shape):
    raise ValueError(f"{name} must have shape {tuple(shape)}")
  if value.dtype != dtype:
    raise ValueError(f"{name} must have dtype {dtype}")
  if value.device.type != device.type:
    raise ValueError(f"{name} must be on {device.type}")
  if device.index is not None and value.device.index != device.index:
    raise ValueError(f"{name} must be on device {device}")
  if not value.is_contiguous():
    raise ValueError(f"{name} must be contiguous")


def _validate_geom_pose_residuals(torch, device, poses, batch, ngeom):
  """Validate FK's retained geom rotation words before solver mutation."""
  if not isinstance(poses, dict):
    raise ValueError("poses must be a mapping of FK tensors")
  for name, width in (("geom_xmat", 9), ("geom_xmat_low", 9),
                      ("geom_xmat_tail", 9), ("geom_quat_low", 4),
                      ("geom_quat_tail", 4)):
    _validate_cached_tensor(torch, device, f"poses[{name!r}]",
                            poses.get(name), (batch, ngeom, width),
                            torch.float32)


def _copy_compaction_map_storage(destination, logical_values):
  """Copy a logical compaction map into its one-element-safe backing.

  Capacity-zero maps have a genuine zero-element logical tensor while their
  retained cache uses one physical sentinel element for Metal allocation.
  Copy only logical elements and restore the sentinel independently.
  """
  if hasattr(destination, "fill_"):
    destination.fill_(-1)
  else:
    destination.fill(-1)
  if hasattr(logical_values, "numel"):
    count = int(logical_values.numel())
  else:
    count = int(np.asarray(logical_values).size)
  if count == 0:
    return
  target = destination.reshape(-1)
  source = logical_values.reshape(-1)
  capacity = int(target.numel()) if hasattr(target, "numel") else int(target.size)
  if capacity < count:
    raise ValueError("compaction cache backing is smaller than its logical map")
  if hasattr(target, "copy_"):
    target[:count].copy_(source)
  else:
    np.copyto(target[:count], np.asarray(source).reshape(-1))


def _position_reference_aref_cpu(jacobian, qvel, context,
                                 surface_velocity=None, extra_aref=None):
  """CPU oracle for replaying a cached MuJoCo row position context.

  ``context[..., :]`` is ``{K, B, impedance, position, margin}``; the optional
  surface term is projected relative velocity and ``extra_aref`` carries
  independently refreshed terms such as equality Jdot-v.
  """
  J = np.asarray(jacobian, dtype=np.float64)
  v = np.asarray(qvel, dtype=np.float64)
  c = np.asarray(context, dtype=np.float64)
  if J.ndim != 3 or v.ndim != 2 or c.shape != (*J.shape[:2], 5):
    raise ValueError("position context expects J[B,nr,nv], qvel[B,nv], ctx[B,nr,5]")
  if J.shape[0] != v.shape[0] or J.shape[2] != v.shape[1]:
    raise ValueError("position context Jacobian and qvel shapes are incompatible")
  velocity = np.einsum("brv,bv->br", J, v)
  if surface_velocity is not None:
    surface = np.asarray(surface_velocity, dtype=np.float64)
    if surface.shape != velocity.shape:
      raise ValueError("surface_velocity must have shape [B,nr]")
    velocity += surface
  result = -c[..., 1] * velocity - c[..., 0] * c[..., 2] * (c[..., 3] - c[..., 4])
  if extra_aref is not None:
    extra = np.asarray(extra_aref, dtype=np.float64)
    if extra.shape != velocity.shape:
      raise ValueError("extra_aref must have shape [B,nr]")
    result += extra
  return result


def _row_velocity_scales(model, descriptor, row_metadata, *, return_low=False):
  """Compile source-derived B coefficients for canonical constraint rows."""
  import mujoco

  nr = int(descriptor.nr)
  out = np.zeros(nr, dtype=np.float32)
  exact = np.zeros(nr, dtype=np.float64)
  meta = np.asarray(row_metadata, dtype=np.int32)
  refsafe = bool(descriptor.refsafe)
  timestep = float(descriptor.timestep)

  def coefficient(solref, dmax, *, safe):
    r0, r1 = (float(x) for x in solref[:2])
    dmax = max(1e-15, float(dmax))
    if safe and refsafe and r0 > 0.0:
      r0 = max(r0, 2.0 * timestep)
    return r1 > 0.0 and r0 > 0.0 and dmax * r0 > 1e-15 \
        and 2.0 / max(1e-15, dmax * r0) or -r1 / dmax

  def set_scale(row, value):
    exact[row] = value
    out[row] = value

  eq_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  friction_dof = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
  limit_joint = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
  friction_tendon = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
  limit_tendon = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
  c_frictionless = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
  c_pyramid = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
  c_ellipse = int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)
  for row, record in enumerate(meta):
    ctype, ident = int(record[0]), int(record[1])
    if ctype == eq_type and 0 <= ident < int(model.neq):
      params = np.asarray(descriptor.eq_sol_params[ident])
      source_params = (params[:7].astype(np.float64)
                       + params[7:14].astype(np.float64))
      set_scale(row, coefficient(source_params[:2], source_params[3], safe=True))
    elif ctype == friction_dof and 0 <= ident < int(model.nv):
      params = np.asarray(descriptor.dof_sol_params[ident])
      set_scale(row, coefficient(params[:2], params[3], safe=True))
    elif ctype == limit_joint and 0 <= ident < int(model.njnt):
      params = np.asarray(descriptor.joint_sol_params[ident])
      set_scale(row, coefficient(params[:2], params[3], safe=True))
    elif ctype in (friction_tendon, limit_tendon) and 0 <= ident < int(model.ntendon):
      params = (descriptor.ten_solref_fri[ident] if ctype == friction_tendon
                else descriptor.ten_solref_lim[ident])
      imp = (descriptor.ten_solimp_fri[ident] if ctype == friction_tendon
             else descriptor.ten_solimp_lim[ident])
      # The source B coefficient is normalized by solimp's upper activation
      # bound (dmax), not its geometric width (imp[2]).
      set_scale(row, coefficient(params, imp[1], safe=True))
    elif ctype in (c_frictionless, c_pyramid, c_ellipse):
      slot = ident
      if not (0 <= slot < int(descriptor.ncontacts_max)):
        continue
      offsets = np.asarray(descriptor.pair_contact_offset, dtype=np.int64)
      pair = int(np.searchsorted(offsets[1:], slot, side="right"))
      if pair >= int(descriptor.npairs):
        continue
      width = float(descriptor.solimp[pair, 1])
      base = coefficient(descriptor.solref[pair], width, safe=False)
      packed = np.asarray(descriptor.contact_condim_packed, dtype=np.int32)
      cdim, row_offset, cone = (int(x) for x in packed[3 * slot:3 * slot + 3])
      local = row - (int(descriptor.nr_joint) + row_offset)
      if local < 0:
        continue
      fr = np.asarray(descriptor.friction[pair], dtype=np.float64)
      frref = np.asarray(descriptor.solreffriction[pair], dtype=np.float64)
      frB = base
      if cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC) and np.any(frref != 0):
        fr0, fr1 = float(frref[0]), float(frref[1])
        frB = 2.0 / max(1e-15, width * fr0) if fr1 > 0 else -fr1 / max(1e-15, width)
      if cdim == 1 or ctype == c_frictionless:
        set_scale(row, base)
      elif cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        set_scale(row, base if local == 0 else frB)
      else:
        # Pyramid rows already contain Jn +/- mu*Jt.  MuJoCo applies the
        # contact's normal reference coefficient to each instantiated edge;
        # mixing the tangential coefficient into B here counts friction twice
        # during a cached-velocity refresh.
        set_scale(row, base)

  # Flex rows carry their exact per-world raw context from FlexContactRows at
  # assembly/capture time.  This static scale is only the generic capture
  # kernel's fallback before that row-owned context is copied over.
  flex_contacts = getattr(descriptor, "flex_contact_descriptor", None)
  if flex_contacts is not None:
    for slot in range(int(flex_contacts.slot_count)):
      params = np.asarray(flex_contacts.solref[slot])
      width = float(flex_contacts.solimp[slot, 2])
      scale = coefficient(params, width, safe=True)
      start = (int(descriptor.flex_contact_base)
               + int(flex_contacts.row_start[slot]))
      stop = min(nr, start + int(flex_contacts.row_span[slot]))
      if start < stop:
        for row in range(start, stop):
          set_scale(row, scale)
  if return_low:
    return out, (exact - out.astype(np.float64)).astype(np.float32)
  return out


def _pack_contact_dims_offsets(c_dims, pair_offsets, slot_row_starts=None):
  """Pack contact header, pair offsets, and canonical candidate row starts."""
  dims = np.asarray(c_dims)
  offsets = np.asarray(pair_offsets)
  for name, values in (("contact dims header", dims), ("pair contact offsets", offsets)):
    if values.dtype.kind not in "iu":
      raise TypeError(f"{name} must contain integers")
    if np.any(values < 0) or np.any(values > np.iinfo(np.int32).max):
      raise ValueError(f"{name} must be int32-representable and nonnegative")
  if dims.shape != (10,):
    raise ValueError(f"contact dims header must have 10 entries, got {dims.shape}")
  if dims[0] < 0 or dims[1] < 0 or dims[2] < 0 or dims[3] <= 0:
    raise ValueError("contact dims nv, npairs, and ncontacts must be nonnegative; batch must be positive")
  if dims[4] < 0 or dims[5] < 0 or dims[6] < 0:
    raise ValueError("contact dims body, joint, and geom counts must be nonnegative")
  if dims[7] not in (0, 1) or dims[8] not in (0, 1) or dims[9] not in (0, 1, 2, 3):
    raise ValueError("contact cone, boolean, and packed-J flags are invalid")
  if offsets.ndim != 1 or offsets.size != int(dims[1]) + 1 or offsets[0] != 0:
    raise ValueError("pair contact offsets must be a nonempty 1-D array starting at zero")
  if np.any(offsets[1:] < offsets[:-1]):
    raise ValueError("pair contact offsets must be monotone")
  if offsets[-1] != dims[2]:
    raise ValueError("final pair contact offset must equal ncontacts")
  if slot_row_starts is None:
    slot_row_starts = np.zeros(int(dims[2]), dtype=np.int32)
  slot_row_starts = np.asarray(slot_row_starts)
  if slot_row_starts.dtype.kind not in "iu":
    raise TypeError("contact candidate row starts must contain integers")
  if (slot_row_starts.shape != (int(dims[2]),)
      or np.any(slot_row_starts < 0)
      or np.any(slot_row_starts > np.iinfo(np.int32).max)):
    raise ValueError("contact candidate row starts must be valid int32 values")
  batch = int(dims[3])
  mask_pointer_index = int(10 + offsets.size + slot_row_starts.size)
  mask_offset = mask_pointer_index + 1
  packed = np.concatenate((dims, offsets, slot_row_starts,
                           np.asarray([mask_offset], dtype=np.int32),
                           np.ones(batch, dtype=np.int32)))
  return packed.astype(np.int32, copy=False)
_MINVAL = 1e-15
_MAX_NV = 32
_MAX_PAIRS = 16
_MAX_CONTACTS = 24
_MAX_ROWS = 96
_MAX_ITERATIONS = 2048
_MAX_LINE_SEARCH_ITERATIONS = 2048
_TOLERANCE = 1e-6
# Per-world device scratch for acceleration-space primal CG/Newton. Keep this
# in the existing debug-buffer binding so the Metal ABI remains at 31 slots.
# G3: hard cap for adaptive PGS extension; must match the kernel constant.
_ADAPTIVE_MAX_ITERATIONS = 0

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
_EQ_FLEX = int(mujoco.mjtEq.mjEQ_FLEX)
_EQ_FLEXVERT = int(mujoco.mjtEq.mjEQ_FLEXVERT)
_EQ_FLEXSTRAIN = int(mujoco.mjtEq.mjEQ_FLEXSTRAIN)
_OBJ_BODY = int(mujoco.mjtObj.mjOBJ_BODY)
_OBJ_SITE = int(mujoco.mjtObj.mjOBJ_SITE)

_EQ_ROWS = {
    _EQ_JOINT: 1,
    _EQ_TENDON: 1,
    _EQ_CONNECT: 3,
    _EQ_WELD: 6,
}

_SUPPORTED_EQ_TYPES = (_EQ_JOINT, _EQ_TENDON, _EQ_CONNECT, _EQ_WELD,
                       _EQ_FLEX, _EQ_FLEXVERT, _EQ_FLEXSTRAIN)

_SUPPORTED_GEOM_TYPES = (_PLANE, _HFIELD, _SPHERE, _CAPSULE, _BOX,
                           _ELLIPSOID, _CYLINDER, _MESH, _SDF)

# Pinned MuJoCo SDF output cap is mjMAXCONPAIR (50).  The source seed
# budget is opt.sdf_initpoints and can be larger; it is handled in scratch
# tiles, then unique accepted outputs are checked against this cap.
_SDF_MAX_NODES = 262144
_SDF_MAX_GEOMS = 8
_SDF_MAX_PER_PAIR = 50
_SDF_SEED_TILE = 32

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


def pair_max_contacts(t1: int, t2: int, sdf_initpoints: int = 40) -> int:
  """Derive upper bound on contact points for a primitive geometry pair.

  Bounds follow the pinned colliders' actual manifold sizes (3.10.0), not
  the conservative mj_maxContact allocation bounds: sphere/ellipsoid
  involvement caps at 1; capsule pairs cap at 2 (mjraw_CapsuleBox emits at
  most best+second); plane-cylinder/box cap at 4; box-box 8; remaining
  cylinder-involved convex pairs cap at 5 (multiccd native witnesses).
  Heightfield pairs emit one witness per overlapped terrain prism
  (mjc_ConvexHField, pinned cap mjMAXCONPAIR=50), including convex meshes
  through the host-compiled hull support map.
  SDF pairs visit every source Halton seed and compact unique accepted
  outputs in seed order up to the pinned mjMAXCONPAIR value of 50.
  """
  t_min, t_max = min(t1, t2), max(t1, t2)
  if _SDF in (t_min, t_max):
    if t_min == _SDF and t_max == _SDF:
      return min(max(0, int(sdf_initpoints)), _SDF_MAX_PER_PAIR)
    if t_min == _MESH:
      # Pinned mjc_MeshSDF collects penetrating face samples up to the
      # mjMAXCONPAIR candidate bound, then emits at most that many witnesses.
      return 50
    if t_min == _PLANE or t_min == _HFIELD:
      return 0
    return min(max(0, int(sdf_initpoints)), _SDF_MAX_PER_PAIR)
  if _HFIELD in (t_min, t_max):
    if t_min == _HFIELD and t_max == _HFIELD:
      # Pinned mjCOLLISIONFUNC has no heightfield-heightfield entry (NULL):
      # rejection matches the engine, it is not a coverage gap.
      raise ValueError("heightfield-heightfield pairs are unsupported")
    if t_max == _MESH:
      # Pinned mjc_ConvexHField traverses the overlapping triangular terrain
      # prisms and runs one convex penetration query per prism. The common
      # source-order GJK/EPA bridge consumes the compiled mesh support map.
      return 50
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


def _global_contact_override(model):
  """Preflight and return pinned mj_assign* global contact parameters.

  This is a host-only preflight so invalid mutable ``MjOption`` override
  payloads fail before descriptor allocation or device workspace setup.
  """
  flag = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
  if not (int(model.opt.enableflags) & flag):
    return None

  values = {
      "solref": (np.asarray(model.opt.o_solref, dtype=np.float64), (2,)),
      "solimp": (np.asarray(model.opt.o_solimp, dtype=np.float64), (5,)),
      "friction": (np.asarray(model.opt.o_friction, dtype=np.float64), (5,)),
  }
  checked = {}
  for name, (value, shape) in values.items():
    if value.shape != shape or not np.all(np.isfinite(value)):
      raise ValueError(f"global contact override {name} must be finite with shape {shape}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      narrowed = value.astype(np.float32)
    if not np.all(np.isfinite(narrowed)):
      raise ValueError(
          f"global contact override {name} must be float32-representable")
    checked[name] = value.copy()

  margin = float(model.opt.o_margin)
  if not math.isfinite(margin):
    raise ValueError("global contact override margin must be finite")
  with np.errstate(over="ignore", under="ignore", invalid="ignore"):
    margin32 = np.float32(margin)
  if not np.isfinite(margin32):
    raise ValueError(
        "global contact override margin must be float32-representable")
  checked["margin"] = margin
  # mj_assignFriction clamps each of the five coefficients independently.
  checked["friction"] = np.maximum(checked["friction"], 1.0e-5)
  return checked


def _solver_parameter_values(descriptor):
  """Return the stable float ABI for solver math and requested stopping.

  ``tolerance`` retains the float32 representability/certification floor used
  by diagnostics. The final word carries MuJoCo's requested outer stopping
  tolerance so a stricter request is not silently replaced by that floor.
  """
  return np.asarray((
      descriptor.timestep, descriptor.impratio, descriptor.tolerance,
      descriptor.noslip_tolerance, descriptor.mean_inertia,
      descriptor.line_search_tolerance,
      descriptor.solver_settings.requested_tolerance,
  ), dtype=np.float32)


@dataclass(frozen=True)
class CoupledSolverSettings:
  """Explicit configuration for the native coupled constraint solver.

  The configured MuJoCo outer iteration count is honored exactly, including
  zero.  A finite iterate is returned even when it does not meet tolerance;
  `solver_diagnostics[0]` reports the final retained-system projected dual
  residual after any configured no-slip pass, while `solver_diagnostics[1]`
  reports executed solver iterations.  `solver_history` samples the solver's
  own optimization metric and is not itself a dual-feasibility certificate.
  No hidden extension or refinement sweeps are added.
  """
  requested_iterations: int
  effective_iterations: int
  requested_tolerance: float
  effective_tolerance: float
  adaptive_max_iterations: int = 0
  max_refinement_sweeps: int = 0
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
  solver_type: int
  line_search_iterations: int
  line_search_tolerance: float
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
  eq_row_ids: tuple

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
  n_flex_contact_rows: int = 0
  flex_contact_base: int = 0
  flex_contact_descriptor: object = None
  # Immutable structural support used by the runtime CSR-J path. Dense models
  # leave this unset; sparse models compile and retain the pinned row layout.
  jacobian_kind: str = "dense"
  jacobian_pattern: object = None


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


def lower_coupled_constraints(model, limits=None, *, mass_storage="dense") -> CoupledConstraintDescriptor:
  """Validate and lower combined joint and contact constraints from an MjModel."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("coupled constraint lowering requires a MuJoCo MjModel")
  if mass_storage not in ("dense", "block_sparse"):
    raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError(f"coupled constraint lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
  contact_override = _global_contact_override(model)
  mean_inertia = float(model.stat.meaninertia)
  if not math.isfinite(mean_inertia) or mean_inertia <= 0.0:
    raise ValueError("model.stat.meaninertia must be finite and positive")
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
  solver_type = int(model.opt.solver)
  if solver_type not in (int(mujoco.mjtSolver.mjSOL_PGS),
                         int(mujoco.mjtSolver.mjSOL_CG),
                         int(mujoco.mjtSolver.mjSOL_NEWTON)):
    raise ValueError(f"unknown constraint solver type {solver_type}")

  # 1. Equality and joint constraint validation (including compiled flex rows)
  scalar_types = (_HINGE, _SLIDE)
  if model.neq:
    eq_types = np.asarray(model.eq_type)
    for eid in range(model.neq):
      et = int(eq_types[eid])
      if et not in _SUPPORTED_EQ_TYPES:
        raise ValueError(
            f"equality {eid}: unsupported equality type "
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
  eq_row_ids_list = []
  flex_eq_inventory = None
  if model.nflex:
    from .flex import lower_flex_descriptor
    flex_eq_inventory = lower_flex_descriptor(model)
  eq_row_cursor = 0
  for eid in range(model.neq):
    et = int(eq_types_arr[eid])
    ot = int(eq_objtype_arr[eid])
    if et == _EQ_FLEX:
      fid = int(model.eq_obj1id[eid])
      if not 0 <= fid < int(model.nflex):
        raise ValueError(f"equality {eid}: flex object id out of range")
      ids = (np.asarray(flex_eq_inventory.equality_row_ids_by_eqid[eid],
                        dtype=np.int32)
             if flex_eq_inventory is not None else np.zeros(0, dtype=np.int32))
      span = int(ids.size)
    elif et in (_EQ_FLEX, _EQ_FLEXVERT, _EQ_FLEXSTRAIN):
      if flex_eq_inventory is None or eid not in flex_eq_inventory.equality_row_counts:
        raise ValueError(f"equality {eid}: flex equality has no compiled row inventory")
      ids = np.asarray(flex_eq_inventory.equality_row_ids_by_eqid[eid],
                       dtype=np.int32)
      span = int(flex_eq_inventory.equality_row_counts[eid])
    else:
      span = int(_EQ_ROWS[et])
      ids = np.arange(span, dtype=np.int32)
    eq_rowadr_list.append(eq_row_cursor)
    eq_rownum_list.append(span)
    eq_row_ids_list.append(_frozen(ids, np.int32))
    eq_row_cursor += span
    if et == _EQ_JOINT:
      j1, j2 = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
      if not 0 <= j1 < model.njnt or int(model.jnt_type[j1]) not in scalar_types:
        raise ValueError(f"equality {eid}: object 1 must be a scalar hinge/slide joint")
      if j2 >= 0 and (j2 >= model.njnt or int(model.jnt_type[j2]) not in scalar_types):
        raise ValueError(f"equality {eid}: object 2 must be a scalar hinge/slide joint")
    elif et in (_EQ_FLEX, _EQ_FLEXVERT, _EQ_FLEXSTRAIN):
      fid = int(model.eq_obj1id[eid])
      if not 0 <= fid < int(model.nflex):
        raise ValueError(f"equality {eid}: flex object id out of range")
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

  # 2. Contact pairs filtering & lowering. Pinned mj_assignRef/Imp/Friction/
  # Margin run after contact-parameter mixing and pair overrides. Global
  # solreffriction is assigned from o_solref as well; gap and condim remain
  # pair/material properties.
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

      solimp = model.pair_solimp[p].copy()
      margin = float(model.pair_margin[p])
      gap = float(model.pair_gap[p])
      solreffriction = model.pair_solreffriction[p].copy()
      if bool(solreffriction[0] > 0) != bool(solreffriction[1] > 0):
        solreffriction[:] = 0.0
      friction = model.pair_friction[p].copy()

      if contact_override is not None:
        solref = contact_override["solref"].copy()
        solreffriction = contact_override["solref"].copy()
        solimp = contact_override["solimp"].copy()
        friction = contact_override["friction"].copy()
        margin = float(contact_override["margin"])

      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solref[0] > 0:
        solref[0] = max(float(solref[0]), 2.0 * float(model.opt.timestep))
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solreffriction[0] > 0:
        solreffriction[0] = max(float(solreffriction[0]), 2.0 * float(model.opt.timestep))

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
      solreffriction = np.zeros(2, dtype=np.float64)
      margin = float(model.geom_margin[g1] + model.geom_margin[g2])
      gap = float(model.geom_gap[g1] + model.geom_gap[g2])
      if contact_override is not None:
        solref = contact_override["solref"].copy()
        solimp = contact_override["solimp"].copy()
        friction = contact_override["friction"].copy()
        solreffriction = contact_override["solref"].copy()
        margin = float(contact_override["margin"])
      elif bool(solref[0] > 0) != bool(solref[1] > 0):
        solref = np.asarray(model.opt.o_solref).copy()
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solref[0] > 0:
        solref[0] = max(float(solref[0]), 2.0 * float(model.opt.timestep))
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solreffriction[0] > 0:
        solreffriction[0] = max(float(solreffriction[0]), 2.0 * float(model.opt.timestep))

      friction5 = np.array([friction[0], friction[0], friction[1], friction[2], friction[2]])
      if contact_override is not None:
        friction5 = np.asarray(friction, dtype=np.float64).copy()
      pairs.append((g1, g2, solref, solimp, condim, friction5, solreffriction, margin, gap))

  npairs = len(pairs)
  # 017 capacity: candidate counts flow through the explicit estimator so
  # overflow carries requested-vs-allowed detail; default limits preserve the
  # historical ceilings until the matching scalable path lands.
  from mujoco_metal.capacity import check_capacity as _check_capacity
  from mujoco_metal.capacity import estimate_capacity as _estimate_capacity
  from mujoco_metal.capacity import DENSE_ROW_THRESHOLD

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
  # Flex contacts own fixed logical candidate slots, each with the complete
  # cone/condim-derived row span.  Reserve the exact canonical descriptor
  # capacity instead of the old four-rows-per-vertex approximation.
  flex_contact_descriptor = None
  n_flex_contact_rows = 0
  if int(model.nflexvert) > 0 and int(model.nflex) > 0:
    from mujoco_metal.flex_contact import lower_flex_contacts
    flex_contact_descriptor = lower_flex_contacts(model)
    n_flex_contact_rows = int(flex_contact_descriptor.row_capacity)
  flex_contact_base = int(nr_joint + nr_contact)
  nr = int(nr_joint + nr_contact + n_flex_contact_rows)
  # Single capacity gate (pairs/slots/rows/nv/batch/memory) with
  # requested-vs-allowed diagnostics; messages preserve historical text.
  # The activity array still has one entry per equality; row mapping is
  # deterministic via eq_rowadr/eq_rownum. Do not replace all uses of neq
  # indiscriminately.
  # Defer the combined runtime memory gate until the selected mass storage
  # and exact structural J pattern are available below.
  # AUTO/SPARSE cannot be budgeted as dense here without rejecting models
  # whose compiled row support is sparse.

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
  if model.neq:
    # Keep the source precision needed by the acceleration reference.  The
    # first seven words retain the existing float32 ABI; the next seven are
    # the rounding residual, so the equality shader can form `aref` as a
    # two-float expansion without changing the model's runtime source values.
    eq_sol_source = np.hstack([
        np.asarray(model.eq_solref).reshape(model.neq, 2),
        np.asarray(model.eq_solimp).reshape(model.neq, 5),
    ])
    eq_sol_hi = eq_sol_source.astype(np.float32)
    eq_sol_lo = (eq_sol_source - eq_sol_hi.astype(np.float64)).astype(np.float32)
    eq_sol_params = np.concatenate([eq_sol_hi, eq_sol_lo], axis=1)
  else:
    eq_sol_params = np.zeros((1, 14), dtype=np.float32)
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
  # reference mesh assets whose compiled octrees are uploaded below; bundled
  # analytic plugin SDFs use compact per-geom high/low attributes.
  sdf_geoms = sorted({g for pr in pairs for g in (pr[0], pr[1])
                      if int(geoms[g]) == _SDF})
  if len(sdf_geoms) > _SDF_MAX_GEOMS:
    raise ValueError(f"at most {_SDF_MAX_GEOMS} SDF geoms; found "
                     f"{len(sdf_geoms)}")
  plugin_sdf_model = None
  plugin_sdf_rows = {}
  sdf_plugin_geoms = [g for g in sdf_geoms
                      if int(model.geom_plugin[g]) != -1]
  if sdf_plugin_geoms:
    from mujoco_metal.bundled_plugins import lower_bundled_plugins
    plugin_sdf_model = lower_bundled_plugins(model)
    plugin_sdf_rows = {
        int(g): row for row, g in enumerate(plugin_sdf_model.sdf_geom_id.tolist())}
    if any(g not in plugin_sdf_rows for g in sdf_plugin_geoms):
      raise ValueError("SDF plugin geom has no supported bundled analytic descriptor")
  sdf_mesh_nodes = {}
  for g in sdf_geoms:
    if g in sdf_plugin_geoms:
      continue
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
      (ta == _SDF or tb == _SDF) and ta not in (_PLANE, _HFIELD, _MESH) and
      tb not in (_PLANE, _HFIELD, _MESH)
      for ta, tb in [(int(geoms[p[0]]), int(geoms[p[1]])) for p in pairs])

  mesh_sdf_geoms = sorted({g for pr in pairs for g in pr[:2]
                           if int(geoms[g]) == _MESH and
                           int(geoms[pr[1] if pr[0] == g else pr[0]]) == _SDF})
  mesh_sdf_assets = sorted({int(model.geom_dataid[g]) for g in mesh_sdf_geoms})
  mesh_sdf_float_count = sum(
      3 * int(model.mesh_vertnum[mid]) +
      3 * int(model.mesh_facenum[mid]) +
      9 * int(model.mesh_bvhnum[mid]) for mid in mesh_sdf_assets)
  if mesh_sdf_float_count > (1 << 24):
    raise ValueError("mesh-SDF BVH data exceeds exact float-index capacity")
  plugin_sdf_float_count = 15 * len(sdf_plugin_geoms)
  sdf_geom_tail_floats = 15 * int(model.ngeom) if sdf_geoms else 0
  geometry_upload_floats = (
      _MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS + _HF_MAX_DATA +
      36 * sdf_oct_total + (18 * int(model.ngeom) if sdf_geoms else 0) +
      sdf_geom_tail_floats +
      mesh_sdf_float_count + plugin_sdf_float_count)
  geometry_upload_bytes = 4 * (
      geometry_upload_floats + 9 * int(model.ngeom))
  if geometry_upload_floats >= (1 << 31):
    from mujoco_metal.capacity import CapacityOverflow
    raise CapacityOverflow("collision geometry upload exceeds int32 addressing")
  from mujoco_metal.capacity import CapacityLimits, CapacityOverflow
  memory_budget = (CapacityLimits().memory_budget_bytes if limits is None else
                   int(limits.memory_budget_bytes))
  if geometry_upload_bytes > memory_budget:
    raise CapacityOverflow(
        "estimated device memory including collision geometry upload "
        f"({geometry_upload_bytes} bytes) exceeds "
        f"budget {memory_budget} bytes")
  mesh_hull = np.zeros((_MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS + _HF_MAX_DATA +
                      36 * sdf_oct_total + (18 * int(model.ngeom) if sdf_geoms else 0) +
                      sdf_geom_tail_floats +
                      mesh_sdf_float_count + plugin_sdf_float_count,),
                     dtype=np.float32)
  mesh_hull_info = np.full((int(model.ngeom) * 9,), -1, dtype=np.int32)
  mesh_rbound_extra = {}
  cursor = 0
  ncursor = 3 * _MESH_MAX_TOTAL
  icursor = 3 * (_MESH_MAX_TOTAL + _MESH_MAX_FACES)
  mesh_sdf_only = set(mesh_sdf_geoms)
  for pr in pairs:
    for g in pr[:2]:
      if int(geoms[g]) == _MESH and int(geoms[pr[1] if pr[0] == g else pr[0]]) != _SDF:
        mesh_sdf_only.discard(g)
  for g in mesh_used:
    mid = int(model.geom_dataid[g])
    if mid < 0:
      raise ValueError(f"mesh geom {g} has no asset")
    raw_vnum = int(model.mesh_vertnum[mid])
    if raw_vnum <= 0:
      raise ValueError(f"mesh geom {g} has no source vertices")
    if g in mesh_sdf_only:
      # Mesh-SDF consumes the full source triangle mesh and BVH below; it does
      # not use the convex support hull or face-snap data.
      continue
    from mujoco_metal.mesh_hull import collision_mesh_hull
    verts, faces = collision_mesh_hull(model, mid)
    vnum, fnum = len(verts), len(faces)
    if vnum <= 0 or vnum > _MESH_MAX_VERTS:
      raise ValueError(
          f"mesh geom {g} collision hull has {vnum} verts; 011 supports "
          f"1..{_MESH_MAX_VERTS} compiled hull verts")
    if cursor + vnum > _MESH_MAX_TOTAL:
      raise ValueError("total mesh hull verts exceed 011 capacity "
                       f"{_MESH_MAX_TOTAL}")
    mesh_hull[3 * cursor:3 * (cursor + vnum)] = verts.reshape(-1)
    mesh_hull_info[9 * g] = cursor
    mesh_hull_info[9 * g + 1] = vnum
    mesh_rbound_extra[g] = float(np.max(np.linalg.norm(verts, axis=1)))
    cursor += vnum
    # Snap planes come from the compiler collision hull, not the potentially
    # concave raw source surface retained separately for ray/SDF queries.
    faces = np.asarray(faces, dtype=np.int64)
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
  # aabb[6N], coeff[8N], aabb residual[6N], coeff residual[8N];
  # per-SDF-geom info holds child base, node
  # count and sdf_iterations. Local AABBs (compiled, center+half) upload
  # for every geom at a fixed stride for the seed-box computation.
  oct_base = _MESH_DATA_FLOATS + 4 * _HF_MAX_GEOMS + _HF_MAX_DATA
  sdf_aabb_base = oct_base + 36 * sdf_oct_total
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
    mesh_hull[ocursor + 22 * num:ocursor + 28 * num] = (
        aabb.astype(np.float64) - aabb.astype(np.float32).astype(np.float64))
    mesh_hull[ocursor + 28 * num:ocursor + 36 * num] = (
        coeff.astype(np.float64) - coeff.astype(np.float32).astype(np.float64))
    ocursor += 36 * num
  sdf_iters = int(model.opt.sdf_iterations)
  if sdf_geoms and sdf_iters <= 0:
    raise ValueError("opt.sdf_iterations must be positive with SDF pairs")
  for g in sdf_geoms:
    if g in sdf_plugin_geoms:
      continue
    mid = int(model.geom_dataid[g])
    mesh_hull_info[9 * g] = mesh_oct_base[mid]
    mesh_hull_info[9 * g + 1] = sdf_mesh_nodes[mid][1]
    mesh_hull_info[9 * g + 2] = sdf_iters
    mesh_hull_info[9 * g + 3] = sdf_aabb_base
    mesh_hull_info[9 * g + 6] = sdf_aabb_base + 18 * int(model.ngeom)
  if sdf_geoms:
    sdf_aabb_tail_base = sdf_aabb_base + 18 * int(model.ngeom)
    for g in range(int(model.ngeom)):
      # Pinned mjc_SDF seeds use m->geom_aabb for both geoms, including
      # plugin meshes. The plugin callback's construction bounds can have
      # a different center/extent from the compiled mesh bounds.
      aabb = model.geom_aabb[g]
      aabb_exact = np.asarray(aabb, dtype=np.float64)
      aabb_high = aabb_exact.astype(np.float32)
      aabb_low = (aabb_exact - aabb_high.astype(np.float64)).astype(np.float32)
      mesh_hull[sdf_aabb_base + 18 * g:sdf_aabb_base + 18 * g + 6] = aabb_high
      mesh_hull[sdf_aabb_base + 18 * g + 6:sdf_aabb_base + 18 * g + 12] = aabb_low
      size = np.asarray(model.geom_size[g], dtype=np.float64)
      size_high = size.astype(np.float32)
      mesh_hull[sdf_aabb_base + 18*g + 12:sdf_aabb_base + 18*g + 15] = size_high
      size_low = (size - size_high.astype(np.float64)).astype(np.float32)
      mesh_hull[sdf_aabb_base + 18*g + 15:sdf_aabb_base + 18*g + 18] = size_low
      tail = sdf_aabb_tail_base + 15 * g
      mesh_hull[tail:tail + 3] = (
          aabb_exact[:3] - aabb_high[:3].astype(np.float64)
          - aabb_low[:3].astype(np.float64)).astype(np.float32)
      mesh_hull[tail + 6:tail + 9] = (
          aabb_exact[3:] - aabb_high[3:].astype(np.float64)
          - aabb_low[3:].astype(np.float64)).astype(np.float32)
      mesh_hull[tail + 12:tail + 15] = (
          size - size_high.astype(np.float64)
          - size_low.astype(np.float64)).astype(np.float32)
    for g in sdf_geoms:
      mesh_hull_info[9 * g + 4] = max(1, int(model.opt.sdf_initpoints))

  # Mesh-SDF collision data is uploaded once per referenced mesh asset:
  # local vertices, triangle indices, then depth-first BVH records. Each BVH
  # record is [leaf face id, child0, child1, center xyz, half xyz]. The
  # shader traverses this compiled tree in the pinned callback order.
  mcursor = (sdf_aabb_base + (18 * int(model.ngeom) if sdf_geoms else 0)
             + sdf_geom_tail_floats)
  for mid in mesh_sdf_assets:
    verts = np.asarray(model.mesh_vert[int(model.mesh_vertadr[mid]):
                                       int(model.mesh_vertadr[mid]) +
                                       int(model.mesh_vertnum[mid])],
                       dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(model.mesh_face[int(model.mesh_faceadr[mid]):
                                       int(model.mesh_faceadr[mid]) +
                                       int(model.mesh_facenum[mid])],
                       dtype=np.int32).reshape(-1, 3)
    bad_index = np.any(faces < 0) or np.any(faces >= len(verts))
    if bad_index:
      raise ValueError(f"mesh-SDF asset {mid} has invalid face vertex indices")
    bvh_adr = int(model.mesh_bvhadr[mid])
    bvh_num = int(model.mesh_bvhnum[mid])
    if bvh_adr < 0 or bvh_num <= 0:
      raise ValueError(f"mesh-SDF asset {mid} has no compiled BVH")
    nodeids = np.asarray(model.bvh_nodeid[bvh_adr:bvh_adr + bvh_num],
                         dtype=np.int32)
    children = np.asarray(model.bvh_child[bvh_adr:bvh_adr + bvh_num],
                          dtype=np.int32).reshape(-1, 2)
    boxes = np.asarray(model.bvh_aabb[bvh_adr:bvh_adr + bvh_num],
                       dtype=np.float32).reshape(-1, 6)
    invalid_child = ((children < -1) | (children >= bvh_num)).any(axis=1)
    invalid_leaf = ((nodeids >= len(faces)) |
                    ((nodeids >= 0) & (children != -1).any(axis=1)))
    invalid_internal = ((nodeids < 0) & (children == -1).all(axis=1))
    if (invalid_child.any() or invalid_leaf.any() or invalid_internal.any()
        or not np.all(np.isfinite(boxes)) or np.any(boxes[:, 3:] < 0)):
      raise ValueError(f"mesh-SDF asset {mid} has invalid BVH records")
    # The native traversal uses a fixed 64-entry stack and expects a single
    # rooted tree. Validate the full compiled tree before upload so no node
    # can be silently skipped or overflow the shader's local stack.
    pending = [0]
    visited = set()
    max_pending = 1
    while pending:
      node = pending.pop()
      if node in visited:
        raise ValueError(f"mesh-SDF asset {mid} BVH contains a cycle/shared node")
      visited.add(node)
      if nodeids[node] < 0:
        for child in children[node]:
          if child >= 0:
            pending.append(int(child))
        max_pending = max(max_pending, len(pending))
        if max_pending > 64:
          raise ValueError(f"mesh-SDF asset {mid} BVH exceeds native stack depth")
    if len(visited) != bvh_num:
      raise ValueError(f"mesh-SDF asset {mid} BVH has unreachable nodes")
    vbase = mcursor
    mesh_hull[vbase:vbase + 3 * len(verts)] = verts.reshape(-1)
    mcursor += 3 * len(verts)
    fbase = mcursor
    mesh_hull[fbase:fbase + 3 * len(faces)] = faces.reshape(-1).astype(np.float32)
    mcursor += 3 * len(faces)
    bbase = mcursor
    records = np.concatenate([
        nodeids[:, None].astype(np.float32), children.astype(np.float32), boxes
    ], axis=1)
    mesh_hull[bbase:bbase + 9 * bvh_num] = records.reshape(-1)
    mcursor += 9 * bvh_num
    for g in mesh_sdf_geoms:
      if int(model.geom_dataid[g]) != mid:
        continue
      mesh_hull_info[9 * g + 4] = len(faces)
      mesh_hull_info[9 * g + 5] = vbase
      mesh_hull_info[9 * g + 6] = fbase
      mesh_hull_info[9 * g + 7] = bbase
      mesh_hull_info[9 * g + 8] = bvh_num
  # Bundled plugin-SDF data is packed as five high-word attributes, five
  # low-word residuals, and five third-word residuals per geom. The first info word uses -2 as
  # a descriptor tag; the remaining fields retain kind, iteration budget,
  # AABB base, seed count and attribute base.
  plugin_attr_cursor = mcursor
  for g in sdf_plugin_geoms:
    row = plugin_sdf_rows[g]
    kind = int(plugin_sdf_model.instance_kind[
        int(plugin_sdf_model.sdf_geom_instance[row])])
    attrs = np.asarray(plugin_sdf_model.sdf_geom_attributes[row], dtype=np.float32)
    attrs_low = np.asarray(plugin_sdf_model.sdf_geom_attributes_low[row], dtype=np.float32)
    attrs_tail = np.asarray(plugin_sdf_model.sdf_geom_attributes_tail[row],
                            dtype=np.float32)
    if attrs.shape != (5,) or attrs_low.shape != (5,) or attrs_tail.shape != (5,) or not (
        np.all(np.isfinite(attrs)) and np.all(np.isfinite(attrs_low)) and
        np.all(np.isfinite(attrs_tail))):
      raise ValueError(f"SDF plugin geom {g} has invalid attribute payload")
    mesh_hull[plugin_attr_cursor:plugin_attr_cursor + 5] = attrs
    mesh_hull[plugin_attr_cursor + 5:plugin_attr_cursor + 10] = attrs_low
    mesh_hull[plugin_attr_cursor + 10:plugin_attr_cursor + 15] = attrs_tail
    base = 9 * g
    mesh_hull_info[base] = -2
    mesh_hull_info[base + 1] = kind
    mesh_hull_info[base + 2] = sdf_iters
    mesh_hull_info[base + 3] = sdf_aabb_base
    mesh_hull_info[base + 4] = max(1, int(model.opt.sdf_initpoints))
    mesh_hull_info[base + 5] = plugin_attr_cursor
    mesh_hull_info[base + 6] = sdf_aabb_base + 18 * int(model.ngeom)
    mesh_hull_info[base + 7] = plugin_attr_cursor + 10
    plugin_attr_cursor += 15
  mcursor = plugin_attr_cursor
  if mcursor != len(mesh_hull):
    raise RuntimeError("collision geometry packed data length mismatch")

  iter_req = int(model.opt.iterations)
  if iter_req < 0 or iter_req > _MAX_ITERATIONS:
    raise ValueError(
        f"integrated_euler_v1 bounds iterations to [0, {_MAX_ITERATIONS}]; found {iter_req}"
    )

  tol_req = float(model.opt.tolerance)
  if not math.isfinite(tol_req) or tol_req <= 0:
    raise ValueError("model.opt.tolerance must be finite and positive")

  ls_iterations = int(model.opt.ls_iterations)
  if ls_iterations < 1 or ls_iterations > _MAX_LINE_SEARCH_ITERATIONS:
    raise ValueError(
        "model.opt.ls_iterations must be in "
        f"[1, {_MAX_LINE_SEARCH_ITERATIONS}]; found {ls_iterations}")
  ls_tolerance = float(model.opt.ls_tolerance)
  if not math.isfinite(ls_tolerance) or ls_tolerance <= 0:
    raise ValueError("model.opt.ls_tolerance must be finite and positive")

  eff_tol = max(tol_req, _TOLERANCE)

  solver_settings = CoupledSolverSettings(
      requested_iterations=iter_req,
      effective_iterations=iter_req,
      requested_tolerance=tol_req,
      effective_tolerance=eff_tol,
      adaptive_max_iterations=_ADAPTIVE_MAX_ITERATIONS,
      max_refinement_sweeps=0,
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
  descriptor = CoupledConstraintDescriptor(
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
      solver_type=solver_type,
      line_search_iterations=ls_iterations,
      line_search_tolerance=ls_tolerance,
      mean_inertia=mean_inertia,
      solver_settings=solver_settings,
      n_eq_rows=n_eq_rows,
      n_flex_contact_rows=n_flex_contact_rows,
      flex_contact_base=flex_contact_base,
      flex_contact_descriptor=flex_contact_descriptor,
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
      eq_row_ids=tuple(eq_row_ids_list),
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
      dense_path=bool(nr <= DENSE_ROW_THRESHOLD and int(model.nv) <= 32),
  )
  from mujoco_metal.capacity import selected_jacobian_kind
  jacobian_kind = str(selected_jacobian_kind(model))
  pattern = None
  if jacobian_kind == "sparse":
    from mujoco_metal.constraint_jacobian import compile_constraint_jacobian_pattern
    pattern = compile_constraint_jacobian_pattern(model, descriptor)
  final_estimate = _estimate_capacity(
      model, 1, npairs, total_candidate_contacts, nr,
      mass_storage=mass_storage,
      jacobian_kind=jacobian_kind,
      jacobian_nnz=(None if pattern is None else pattern.nnz))
  sdf_seed_tile = (_SDF_SEED_TILE if sdf_active else 0)
  contact_cache_bytes = _pair_contact_cache_bytes(
      1, total_candidate_contacts, npairs, sdf_seed_tile)
  final_estimate = replace(
      final_estimate,
      memory_bytes=final_estimate.memory_bytes + geometry_upload_bytes + contact_cache_bytes,
      memory_breakdown=final_estimate.memory_breakdown +
      (("collision_geometry_upload", geometry_upload_bytes),
       ("persistent_pair_contact_cache", contact_cache_bytes)))
  _check_capacity(final_estimate, limits=limits)
  return replace(descriptor, jacobian_kind=jacobian_kind,
                 jacobian_pattern=pattern)


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


@dataclass(frozen=True)
class _ComponentMassPlan:
  arrays: dict
  ncomponent: int
  nnz: int
  packed: np.ndarray


def _validate_component_layout(layout, nv):
  """Validate/pack a component layout on CPU, before any workspace allocation."""
  if layout is None:
    return None
  if not isinstance(layout, dict):
    raise TypeError("component mass layout must be a mapping")
  nv = int(nv)
  ncomponent, nnz = layout.get("ncomponent"), layout.get("nnz")
  for name, value in (("ncomponent", ncomponent), ("nnz", nnz)):
    if (isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) or int(value) < 0
        or int(value) > np.iinfo(np.int32).max):
      raise ValueError(f"component layout {name} must be a nonnegative int32")
  ncomponent, nnz = int(ncomponent), int(nnz)
  shapes = {
      "component_dof_offsets": (ncomponent + 1,),
      "component_dof_ids": (nv,),
      "component_mass_offsets": (ncomponent,),
      "dof_component": (nv,),
      "dof_local_index": (nv,),
  }
  limit = np.iinfo(np.int32).max
  arrays = {}
  for name, shape in shapes.items():
    value = np.asarray(layout.get(name))
    if value.shape != shape or value.dtype.kind not in "iu":
      raise ValueError(f"component layout {name} must be integer {shape}")
    if value.size and (np.any(value < 0) or np.any(value > limit)):
      raise ValueError(f"component layout {name} exceeds signed int32")
    arrays[name] = value.astype(np.int32, copy=False)
  offsets = arrays["component_dof_offsets"].astype(np.int64, copy=False)
  dofs = arrays["component_dof_ids"]
  if (offsets[0] != 0 or offsets[-1] != nv
      or np.any(np.diff(offsets) < 0)):
    raise ValueError("component offsets must partition all generalized DOFs")
  if dofs.size and (np.any(dofs >= nv)
                    or not np.array_equal(np.sort(dofs), np.arange(nv))):
    raise ValueError("component DOF IDs must permute [0,nv)")
  expected_mass_offset = 0
  for component in range(ncomponent):
    begin, end = int(offsets[component]), int(offsets[component + 1])
    component_dofs = dofs[begin:end]
    if (np.any(arrays["dof_component"][component_dofs] != component)
        or not np.array_equal(
            arrays["dof_local_index"][component_dofs],
            np.arange(end - begin, dtype=np.int32))):
      raise ValueError("component reverse DOF maps disagree with offsets")
    if int(arrays["component_mass_offsets"][component]) != expected_mass_offset:
      raise ValueError("component mass offsets must densely partition nnz")
    expected_mass_offset += (end - begin) ** 2
  if expected_mass_offset != nnz:
    raise ValueError("component mass blocks do not match the compiled nnz")
  packed = np.concatenate(tuple(arrays[name] for name in shapes))
  return _ComponentMassPlan(arrays, ncomponent, nnz, packed)


def _validate_component_layout_for_model(plan, model):
  """Require the supplied sparse plan to match the model's compiled blocks."""
  from mujoco_metal.model import actuator_tendon_inheritance
  from mujoco_metal.mass_layout import compile_tree_mass_layout
  inherited, _, _ = actuator_tendon_inheritance(model)
  expected = compile_tree_mass_layout(
      model.body_treeid, model.dof_bodyid,
      tendon_treeid=model.tendon_treeid,
      tendon_treenum=model.tendon_treenum,
      tendon_armature=(np.asarray(model.tendon_armature, dtype=np.float64)
                       + inherited),
      tendon_j_rowadr=model.ten_J_rowadr,
      tendon_j_rownnz=model.ten_J_rownnz,
      tendon_j_colind=model.ten_J_colind,
      mass_rowadr=model.M_rowadr,
      mass_rownnz=model.M_rownnz,
      mass_colind=model.M_colind)
  if plan.nnz != int(expected["nnz"]):
    raise ValueError("component layout block storage does not match the model")
  if plan.ncomponent != int(expected["ncomponent"]):
    raise ValueError("component layout partition does not match the model")
  for name in ("component_dof_offsets", "component_dof_ids",
               "component_mass_offsets", "dof_component", "dof_local_index"):
    if not np.array_equal(plan.arrays[name], expected[name]):
      raise ValueError(f"component layout {name} does not match the model")
  return plan


class MetalCoupledConstraints:
  """Batched native MPS execution for coupled joint and primitive contact constraints."""

  def __init__(self, model, batch_size=1, limits=None, *,
               mass_storage="dense", component_layout=None):
    from mujoco_metal.capacity import CapacityLimits, primal_scratch_floats
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("coupled constraint construction requires a MuJoCo MjModel")
    self.limits = limits or CapacityLimits()
    if mass_storage not in ("dense", "block_sparse"):
      raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
    # Resolve and validate the selected representation before lowering,
    # capacity admission, MPS setup, or any device workspace allocation.
    component_plan = _validate_component_layout(component_layout, int(model.nv))
    if (mass_storage == "block_sparse") != (component_plan is not None):
      raise ValueError(
          "block_sparse coupled constraints require a compiled component_layout; "
          "dense storage must not receive one")
    if component_plan is not None:
      component_plan = _validate_component_layout_for_model(component_plan, model)
    self.descriptor = lower_coupled_constraints(
        model, limits=self.limits, mass_storage=mass_storage)
    self._mjmodel_ref = model
    pair_types = [(int(model.geom_type[int(a)]), int(model.geom_type[int(b)]))
                  for a, b in zip(self.descriptor.geom1, self.descriptor.geom2)]
    self._has_mesh_sdf_pairs = any(set(pair) == {7, 8} for pair in pair_types)
    self._has_analytic_sdf_pairs = any(
        8 in pair and 7 not in pair for pair in pair_types)
    self._has_sdf_contact_pairs = (
        self._has_mesh_sdf_pairs or self._has_analytic_sdf_pairs)
    self._has_generic_contact_pairs = any(8 not in pair for pair in pair_types)
    self.solver_settings = self.descriptor.solver_settings
    self._line_search_iterations = int(self.descriptor.line_search_iterations)
    self.batch_size = int(batch_size)
    if self.batch_size <= 0:
      raise ValueError("batch_size must be positive")
    d = self.descriptor
    self._component_mass_storage = mass_storage
    self._common_ccd_layout = None
    self._common_ccd_mesh_hull = d.mesh_hull
    self._common_ccd_mesh_hull_info = d.mesh_hull_info
    self._common_ccd_hfield_mesh_requested = any(
        {int(d.geom_type[int(a)]), int(d.geom_type[int(b)])} == {_HFIELD, _MESH}
        for a, b in zip(d.geom1, d.geom2))
    from mujoco_metal.common_ccd_bridge import uses_pinned_mjc_convex
    self._common_ccd_rigid_requested = any(
        uses_pinned_mjc_convex(int(d.geom_type[int(a)]),
                               int(d.geom_type[int(b)]))
        for a, b in zip(d.geom1, d.geom2))
    self._common_ccd_requested = (
        self._common_ccd_hfield_mesh_requested or self._common_ccd_rigid_requested)
    if self._common_ccd_requested:
      from mujoco_metal.common_ccd_bridge import (
          append_production_workspace, estimate_production_upload_bytes)
      from mujoco_metal.capacity import check_capacity, estimate_capacity
      common_estimate = estimate_capacity(
          model, self.batch_size, d.npairs, d.ncontacts_max, d.nr,
          neq=d.neq, nr_joint=d.nr_joint, mass_storage=self._component_mass_storage,
          jacobian_kind=str(getattr(d, "jacobian_kind", "dense")),
          jacobian_nnz=(int(d.jacobian_pattern.nnz)
                        if getattr(d, "jacobian_pattern", None) is not None else None))
      common_bytes = estimate_production_upload_bytes(
          model, d.mesh_hull, d.mesh_hull_info,
          self.batch_size, d.npairs)
      common_estimate = replace(
          common_estimate, memory_bytes=common_estimate.memory_bytes + common_bytes,
          memory_breakdown=common_estimate.memory_breakdown +
          (("common_ccd_model_and_workspace", common_bytes),))
      # Reject before allocating the appended per-query output, EPA arena,
      # dynamic prism vertices, or feature-ID sidecar.
      check_capacity(common_estimate, limits=self.limits)
      common_hull, common_info, common_layout = append_production_workspace(
          model, d.mesh_hull, d.mesh_hull_info, self.batch_size, d.npairs)
      actual_common_bytes = int(common_hull.nbytes + common_info.nbytes)
      if actual_common_bytes != common_bytes:
        raise RuntimeError(
            "common CCD preflight disagrees with final upload size: "
            f"predicted {common_bytes}, allocated {actual_common_bytes}")
      self._common_ccd_mesh_hull = common_hull
      self._common_ccd_mesh_hull_info = common_info
      self._common_ccd_layout = common_layout
    # `lower_coupled_constraints` checks batch one because it is a reusable
    # host descriptor. The program constructor must admit its actual batch
    # before uploading the component-layout tensor or allocating any MPS
    # workspace. The common-CCD branch above already admitted this envelope
    # with its additional runtime arena bytes.
    from mujoco_metal.capacity import check_capacity as _check_capacity
    from mujoco_metal.capacity import estimate_capacity as _estimate_capacity
    _check_capacity(_estimate_capacity(
        model, self.batch_size, d.npairs, d.ncontacts_max, d.nr,
        neq=d.neq, nr_joint=d.nr_joint,
        mass_storage=self._component_mass_storage,
        jacobian_kind=str(getattr(d, "jacobian_kind", "dense")),
        jacobian_nnz=(int(d.jacobian_pattern.nnz)
                      if getattr(d, "jacobian_pattern", None) is not None
                      else None)), limits=self.limits)
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("native coupled constraints require PyTorch MPS compile_shader")
    self._torch = torch
    self._device = torch.device("mps")
    self._sdf_seed_count = max(0, int(model.opt.sdf_initpoints))
    self._sdf_seed_tile = _SDF_SEED_TILE
    self._sdf_stage_dims = torch.zeros(5, dtype=torch.int32, device=self._device)
    sdf_geom_ids = np.flatnonzero(
        np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_SDF))
    self._sdf_max_iterations = max((
        int(self.descriptor.mesh_hull_info[9 * int(g) + 2])
        for g in sdf_geom_ids), default=0)

    shader_source = _coupled_shader_source()
    self._library = torch.mps.compile_shader(shader_source)
    # Equality assembly is a standalone stage: compile it separately so its
    # entry point does not enlarge the collision/SDF/solver compilation unit.
    self._equality_library = torch.mps.compile_shader(_EQUALITY_SHADER.read_text())
    self._contact_kernel = self._library.contact_normal
    self._mask_failed_candidate_worlds_kernel = (
        self._library.mask_failed_candidate_worlds)
    self._common_ccd_hfield_mesh_kernel = (
        self._library.common_ccd_hfield_mesh_candidates
        if self._common_ccd_hfield_mesh_requested else None)
    self._common_ccd_rigid_kernel = (
        self._library.common_ccd_rigid_convex_candidates
        if self._common_ccd_rigid_requested else None)
    self._clear_contact_pair_counts = self._library.clear_contact_pair_counts
    self._contact_generic_producer = (
        self._library.contact_produce_generic
        if self._has_generic_contact_pairs else None)
    self._contact_sdf_seed_init = (
        self._library.contact_sdf_seed_init
        if self._has_analytic_sdf_pairs else None)
    self._contact_sdf_phase_reset = (
        self._library.contact_sdf_phase_reset
        if self._has_analytic_sdf_pairs else None)
    self._contact_sdf_descent_prepare = (
        self._library.contact_sdf_descent_prepare
        if self._has_analytic_sdf_pairs else None)
    self._contact_sdf_line_search = (
        self._library.contact_sdf_line_search
        if self._has_analytic_sdf_pairs else None)
    self._contact_sdf_publish_normal = (
        self._library.contact_sdf_publish_normal
        if self._has_analytic_sdf_pairs else None)
    self._contact_sdf_publish_contact = (
        self._library.contact_sdf_publish_contact
        if self._has_analytic_sdf_pairs else None)
    self._contact_sdf_finalize_producer = (
        self._library.contact_finalize_sdf
        if self._has_analytic_sdf_pairs else None)
    self._contact_mesh_sdf_producer = (
        self._library.contact_produce_mesh_sdf
        if self._has_mesh_sdf_pairs else None)
    try:
      self._equality_kernel = self._equality_library.equality_assembly
    except AttributeError:
      self._equality_kernel = None
    try:
      self._tendon_kernel = self._library.tendon_constraint_rows
    except AttributeError:
      self._tendon_kernel = None
    self._solve_kernel = self._library.solve_coupled_constraints
    self._merge_flex_narrowphase_status_kernel = (
        self._library.merge_flex_narrowphase_status)
    self._merge_sdf_narrowphase_status_kernel = (
        self._library.merge_sdf_narrowphase_status)
    self._pack_component_mass_rhs = self._library.pack_component_mass_rhs
    self._refresh_position_reference_kernel = (
        self._library.refresh_cached_constraint_aref)
    self._refresh_aref_low_kernel = (
        self._library.refresh_cached_constraint_aref_low)
    self._capture_position_context_kernel = (
        self._library.capture_cached_position_context)
    self._refresh_equality_jdot_kernel = (
        self._library.refresh_cached_equality_jdot)
    try:
      self._solve_block_kernel = self._library.solve_coupled_constraints_block
    except AttributeError:
      self._solve_block_kernel = self._solve_kernel
    self._broadphase_lib = torch.mps.compile_shader(_BROADPHASE_SHADER.read_text())
    self._broadphase_kernel = self._broadphase_lib.broadphase_mask
    from mujoco_metal.row_compaction import CompactionWorkspace
    self._CompactionWorkspace = CompactionWorkspace
    from mujoco_metal.active_contact_links import (
        ActiveContactLinkWorkspace, equality_tree_links, pair_tree_ids)
    self._ActiveContactLinkWorkspace = ActiveContactLinkWorkspace
    self._equality_tree_links = equality_tree_links
    self._pair_tree_ids = pair_tree_ids

    d = self.descriptor
    from mujoco_metal.solver_islands import lower_solver_row_metadata
    self._solver_island_ntree = int(model.ntree)
    self._solver_island_nbody = int(model.nbody)
    self._solver_island_row_metadata = lower_solver_row_metadata(model, d)
    (self._row_velocity_scales_cpu,
     self._row_velocity_scales_low_cpu) = _row_velocity_scales(
        model, d, self._solver_island_row_metadata, return_low=True)
    self._solver_island_dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
    if self._solver_island_dof_tree.size == 0:
      self._solver_island_dof_tree = np.zeros(1, dtype=np.int32)
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
        else np.zeros(0, dtype=np.float32)
    )
    contact_solreffriction_arr = (
        d.contact_solreffriction.reshape(-1)
        if d.ncontacts_max else np.zeros(2, dtype=np.float32)
    )
    # The row solver consumes contact cone metadata for both rigid and flex
    # candidate slots. Keep broadphase's rigid-only arrays untouched and
    # append flex slots to the solver-facing views in the same canonical slot
    # order used by FlexContactRows.
    flex_contacts = d.flex_contact_descriptor
    if flex_contacts is not None and int(flex_contacts.slot_count):
      flex_solver_friction = np.asarray(
          flex_contacts.friction, dtype=np.float32).reshape(-1, 5)
      flex_row_offset = int(d.flex_contact_base - d.nr_joint)
      flex_solver_condim = np.column_stack((
          np.asarray(flex_contacts.condim, dtype=np.int32),
          flex_row_offset + np.asarray(flex_contacts.row_start, dtype=np.int32),
          np.asarray(flex_contacts.cone, dtype=np.int32),
      )).reshape(-1)
      solver_contact_friction_arr = np.concatenate((
          np.asarray(contact_friction_arr, dtype=np.float32).reshape(-1),
          flex_solver_friction.reshape(-1)))
      solver_contact_condim_arr = np.concatenate((
          np.asarray(d.contact_condim_packed if d.ncontacts_max
                     else np.zeros(0, dtype=np.int32), dtype=np.int32),
          flex_solver_condim))
    else:
      solver_contact_friction_arr = np.asarray(
          contact_friction_arr if d.ncontacts_max else np.zeros(5, dtype=np.float32),
          dtype=np.float32)
      solver_contact_condim_arr = np.asarray(
          d.contact_condim_packed if d.ncontacts_max else np.zeros(3, dtype=np.int32),
          dtype=np.int32)

    c_dims_values = [d.nv, d.npairs, d.ncontacts_max, self.batch_size,
                     d.nbody, d.njnt, d.ngeom, d.cone_type,
                     1 if (int(d.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)) else 0,
                     (1 | (2 if str(getattr(d, "jacobian_kind", "dense")) == "sparse" else 0))]
    # The constants dictionary includes body_dims and solver_dims. Establish
    # their shared Jacobian ABI before either tensor is packed; prepare_workspace
    # later creates the matching per-world numeric backing.
    from mujoco_metal.constraint_jacobian import (
        PACKED_J_CSR, PackedJacobianLayout)
    jacobian_pattern = getattr(d, "jacobian_pattern", None)
    jacobian_kind = str(getattr(d, "jacobian_kind", "dense"))
    self._jacobian_pattern = jacobian_pattern
    self._jacobian_layout = PackedJacobianLayout.create(
        d.nr, d.nv, pattern=jacobian_pattern,
        mode=(PACKED_J_CSR if jacobian_kind == "sparse" else None))
    jacobian_layout = self._jacobian_layout
    # `_solver_dimension_tensor` is called while constructing `_constants`,
    # so the sparse-layout defaults must exist before that pack is evaluated.
    self._component_operator_layout_device = None
    self._component_operator_layout_words = 0
    self._component_operator_ncomponent = 0
    self._component_operator_nnz = 0
    self._component_operator_armature_offset = 0
    self._component_operator_layout_offset = 0
    self._component_mass_storage = mass_storage
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
        "eq_sol_params": self._tensor(d.eq_sol_params.reshape(-1) if d.neq else np.zeros(14, dtype=np.float32)),
        "eq_type": self._tensor(d.eq_type if d.neq else np.zeros(1, dtype=np.int32)),
        "eq_objtype": self._tensor(d.eq_objtype if d.neq else np.zeros(1, dtype=np.int32)),
        "eq_rowadr": self._tensor(d.eq_rowadr if d.neq else np.zeros(1, dtype=np.int32)),
        "eq_rownum": self._tensor(d.eq_rownum if d.neq else np.zeros(1, dtype=np.int32)),
        "row_velocity_scales": self._tensor(
            self._row_velocity_scales_cpu if d.nr else np.zeros(1, dtype=np.float32)),
        "row_velocity_scale_low": self._tensor(
            self._row_velocity_scales_low_cpu if d.nr else np.zeros(1, dtype=np.float32)),
        "geom_size": self._tensor(d.geom_size.reshape(-1) if d.ngeom else np.zeros(3, dtype=np.float32)),
        "geom_type": self._tensor(d.geom_type if d.ngeom else np.zeros(1, dtype=np.int32)),
        "geom_rbound": self._tensor(d.geom_rbound if d.ngeom else np.zeros(1, dtype=np.float32)),
        "mesh_hull": self._tensor(self._common_ccd_mesh_hull),
        "mesh_hull_info": self._tensor(
            np.asarray(self._common_ccd_mesh_hull_info, dtype=np.int32)),
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
        "contact_friction": self._tensor(solver_contact_friction_arr),
        "contact_solreffriction": self._tensor(contact_solreffriction_arr),
        "contact_condim": self._tensor(solver_contact_condim_arr),
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
        # Tail index 9 toggles broadphase-fed narrowphase pruning (R08/F1).
        "c_dims": torch.tensor(c_dims_values, dtype=torch.int32, device=self._device),
        "body_dims": torch.tensor(
            [d.ncontacts_max, d.npairs, d.nv, 0, self.batch_size,
             jacobian_layout.mode] + [1] * self.batch_size,
            dtype=torch.int32, device=self._device,
        ),
        "solver_dims": self._solver_dimension_tensor(),
        "solver_params": torch.tensor(
            _solver_parameter_values(d),
            dtype=torch.float32, device=self._device,
      ),
    }
    # Keep a fixed ten-word dimensions header followed by logical slot
    # offsets. The contact kernel can then read npairs from header[1] without
    # circularly deriving the header address from an offset value.
    packed_dims = self._tensor(
        _pack_contact_dims_offsets(
            c_dims_values, d.pair_contact_offset,
            (int(d.nr_joint)
             + np.asarray(d.contact_condim_packed, dtype=np.int32).reshape(-1, 3)[:, 1]
             if d.ncontacts_max else np.zeros(0, dtype=np.int32))))
    self._constants["pair_contact_offsets_dims"] = packed_dims
    self._constants["c_dims"] = packed_dims[:10]
    self._constants["pair_contact_offset"] = packed_dims[
        10:10 + int(d.npairs) + 1]
    mask_pointer_index = 10 + int(d.npairs) + 1 + int(d.ncontacts_max)
    self._pair_contact_world_mask = packed_dims[
        mask_pointer_index + 1:mask_pointer_index + 1 + self.batch_size]
    self._pair_contact_status_dims = torch.tensor(
        [self.batch_size, int(jacobian_layout.stride_words)]
        + [1] * self.batch_size,
        dtype=torch.int32, device=self._device)
    self._pair_contact_jclear_dims = torch.empty(
        (4 + self.batch_size,), dtype=torch.int32, device=self._device)
    self._position_refresh_copy_dims = torch.empty(
        (6 + self.batch_size,), dtype=torch.int32, device=self._device)
    self._workspace = None
    self._candidate_generation_epoch = 0
    self._candidate_generation_token = None
    self._assembly_generation = 0
    self._position_context_epoch = 0
    self._position_context_valid = False
    self._position_current_eq_jdot_inputs_valid = False
    self._position_context_eq_jdot_inputs_valid = False
    self._position_current_valid = False
    if component_plan is None:
      self.prepare_workspace(self.batch_size)
    else:
      self.configure_component_layout(component_plan)

  def configure_component_layout(self, layout):
    """Install immutable sparse mass metadata before the first assembly.

    The component layout is copied once into a typed persistent MPS tensor.
    Per-world copies are made into the already-bound debug buffer tail before
    sparse solver dispatches, keeping the Metal ABI at buffer indices 0..30.
    """
    if self._assembly_generation or self._position_context_valid:
      raise RuntimeError("component layout must be configured before assembly")
    if self._workspace is not None:
      raise RuntimeError(
          "component layout must be supplied before the initial workspace allocation")
    if self._component_operator_layout_device is not None:
      raise RuntimeError("component layout is already configured")
    plan = (layout if isinstance(layout, _ComponentMassPlan) else
            _validate_component_layout(layout, int(self.descriptor.nv)))
    if plan is None:
      raise TypeError("component mass layout must be a mapping")
    ncomponent, nnz, packed = plan.ncomponent, plan.nnz, plan.packed
    batch = int(self.batch_size)
    tail_words = nnz + int(packed.size)
    if max(batch * tail_words, 1) > np.iinfo(np.int32).max:
      raise ValueError("component operator tail exceeds Metal int32 addressing")
    self._component_operator_ncomponent = ncomponent
    self._component_operator_nnz = nnz
    self._component_mass_storage = "block_sparse"
    self._component_operator_layout_device = self._torch.as_tensor(
        packed.copy(), dtype=self._torch.int32, device=self._device)
    # The constructor has already admitted the exact batch before MPS setup;
    # prepare_workspace now allocates only this selected sparse backing.
    self.prepare_workspace(batch)
    self._constants["solver_dims"] = self._solver_dimension_tensor()

  def _tensor(self, value):
    return self._torch.as_tensor(np.asarray(value).copy(), device=self._device)

  def _solver_dimension_tensor(self):
    """Pack scalar solver dimensions and immutable island row metadata."""
    from mujoco_metal.capacity import primal_scratch_floats, solver_debug_layout
    from mujoco_metal.solver_islands import pack_solver_dimension_metadata
    from mujoco_metal.constraint_jacobian import (
        PACKED_J_CSR, PackedJacobianLayout)
    d = self.descriptor
    if getattr(self, "_jacobian_layout", None) is None:
      pattern = getattr(d, "jacobian_pattern", None)
      kind = str(getattr(d, "jacobian_kind", "dense"))
      self._jacobian_pattern = pattern
      self._jacobian_layout = PackedJacobianLayout.create(
          d.nr, d.nv, pattern=pattern,
          mode=(PACKED_J_CSR if kind == "sparse" else None))
    nv, nr, ntree = d.nv, d.nr, self._solver_island_ntree
    layout_words = (int(self._component_operator_layout_device.numel())
                    if self._component_operator_layout_device is not None else 0)
    debug_layout = solver_debug_layout(
        nv, nr, d.solver_type, ntree, self._solver_island_nbody,
        self._component_operator_nnz, layout_words,
        self._component_mass_storage)
    self._base_debug_stride = debug_layout["base_stride"]
    self._qacc_warmstart_offset = debug_layout["qacc_warmstart_offset"]
    self._equality_jacobian_offset = debug_layout["equality_jacobian_offset"]
    self._block_workspace_offset = debug_layout["block_workspace_offset"]
    self._primal_workspace_offset = debug_layout["primal_workspace_offset"]
    self._z_workspace_offset = debug_layout["z_workspace_offset"]
    self._high_low_workspace_offset = debug_layout["high_low_workspace_offset"]
    self._high_low_workspace_length = debug_layout["high_low_workspace_length"]
    self._aref_low_debug_offset = (
        nr * nr + 7 * nr + self._high_low_workspace_offset
        + 8 * max(nv, 1) + 2 * max(nr, 1))
    core_scratch = (primal_scratch_floats(
                        nv, nr, d.solver_type, 0, 0,
                        self._component_mass_storage)
                    - 2 * max(nv, 1) - max(nr, 1) - 6
                    - 12 * max(nv, 1))
    self._solver_island_core_scratch = core_scratch
    prefix = [
        d.nq, nv, d.njnt, d.neq, d.ncontacts_max, self.batch_size,
        d.disableflags, 1 if d.refsafe else 0, d.iterations, nr, d.cone_type,
        d.nbody, d.njnt, d.nsite, d.n_eq_rows, d.ntendon, d.ten_base,
        d.ten_friction_rows + d.ten_limit_rows, d.noslip_iterations,
        d.solver_type, d.line_search_iterations,
        debug_layout["final_stride"],
    ]
    island_disable_bit = int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
    packed = pack_solver_dimension_metadata(
        prefix, self._solver_island_dof_tree,
        self._solver_island_row_metadata, ntree, core_scratch,
        island_disable_bit, nbody=self._solver_island_nbody)
    metadata_header = int(packed[24]) + nr * 10
    self._solver_dimension_header = metadata_header
    self._solver_island_awake_offset = int(packed[27])
    packed = np.concatenate((packed, np.asarray((
        self._component_operator_ncomponent,
        self._component_operator_nnz,
        debug_layout["component_armature_offset"],
        debug_layout["component_layout_offset"],
        debug_layout["base_stride"],
        debug_layout["qacc_warmstart_offset"],
        debug_layout["equality_jacobian_offset"],
        debug_layout["component_operator_offset"],
        debug_layout["block_workspace_offset"],
        debug_layout["primal_workspace_offset"],
        debug_layout["z_workspace_offset"],
        (int(d.flex_contact_descriptor.slot_count)
         if d.flex_contact_descriptor is not None else 0),
        int(d.flex_contact_base - d.nr_joint),
        int(d.n_flex_contact_rows),
        int(d.line_search_iterations),
        debug_layout["high_low_workspace_offset"],
        debug_layout["high_low_workspace_length"],
        int(self._jacobian_layout.stride_words),
        int(self._jacobian_layout.mode),
        int(self._jacobian_layout.nnz),
    ), dtype=np.int32)))
    # Per-call recovery masks live in a typed int32 suffix of the existing
    # solver-dimensions binding.  This keeps the 31-buffer CC ABI unchanged
    # while allowing every per-world kernel to return before touching a
    # failed or unselected world's workspace.
    world_mask_offset = int(packed.size) + 1
    self._world_mask_offset = world_mask_offset
    packed = np.concatenate((packed, np.asarray((world_mask_offset,),
                                                dtype=np.int32)))
    packed = np.concatenate((packed, np.ones(self.batch_size, dtype=np.int32)))
    packed[metadata_header + 20] = world_mask_offset
    dims = self._torch.as_tensor(packed.copy(), dtype=self._torch.int32,
                                 device=self._device)
    if getattr(self, "_iteration_trace_enabled", False):
      dims[7:8].add_(32)
    if getattr(self, "_primal_detail_trace_enabled", False):
      dims[7:8].add_(64)
      if getattr(self, "_primal_detail_iteration", 1) == 2:
        dims[7:8].add_(128)
      if getattr(self, "_primal_detail_components", False):
        dims[7:8].add_(256)
    return dims

  def set_iteration_trace(self, enabled):
    """Enable a private per-world iteration-count observer in diagnostics.

    The observer uses only the spare dims[7] bit 32 and the existing
    diagnostics history words. It never changes solver inputs or allocates
    device storage. While enabled, diagnostics[2:10] holds either per-island
    primal counts (CG/Newton) or the PGS main/no-slip split in [2:4].
    """
    enabled = bool(enabled)
    previous = bool(getattr(self, "_iteration_trace_enabled", False))
    if enabled == previous:
      return
    # Increment/decrement the dedicated bit without reading an MPS scalar to
    # Python; repeated calls are idempotent through the host-side toggle.
    delta = 32 if enabled else -32
    self._constants["solver_dims"][7:8].add_(delta)
    self._iteration_trace_enabled = enabled

  def set_primal_detail_trace(self, enabled, *, accepted_iteration=1,
                              gradient_components=False):
    """Capture a selected accepted primal iteration in existing history.

    This diagnostic-only mode keeps the ordinary count observer enabled and
    reuses the fixed ten-word diagnostics row; it allocates no device memory.
    ``accepted_iteration=2`` selects the second accepted point. With
    ``gradient_components=True``, history[2:8] instead holds the first three
    gradient high/low words after that point; this bounded witness is useful
    for the two-DOF mixed-island regression.
    """
    enabled = bool(enabled)
    accepted_iteration = int(accepted_iteration)
    gradient_components = bool(gradient_components)
    if accepted_iteration not in (1, 2):
      raise ValueError("accepted_iteration must be 1 or 2")
    if gradient_components and accepted_iteration != 2:
      raise ValueError("gradient_components requires accepted_iteration=2")
    previous = bool(getattr(self, "_primal_detail_trace_enabled", False))
    old_iteration = int(getattr(self, "_primal_detail_iteration", 1))
    old_components = bool(getattr(self, "_primal_detail_components", False))
    if (enabled == previous and (not enabled or
        (accepted_iteration == old_iteration
         and gradient_components == old_components))):
      return
    if enabled and not getattr(self, "_iteration_trace_enabled", False):
      self.set_iteration_trace(True)
    old_bits = ((64 + (128 if old_iteration == 2 else 0)
                 + (256 if old_components else 0)) if previous else 0)
    new_bits = ((64 + (128 if accepted_iteration == 2 else 0)
                 + (256 if gradient_components else 0)) if enabled else 0)
    self._constants["solver_dims"][7:8].add_(new_bits - old_bits)
    self._primal_detail_trace_enabled = enabled
    self._primal_detail_iteration = accepted_iteration
    self._primal_detail_components = gradient_components

  def _primal_detail_extra_flags(self):
    """Host-side observer bits for cached dispatches; no device readback."""
    return ((128 if getattr(self, "_primal_detail_iteration", 1) == 2 else 0)
            | (256 if getattr(self, "_primal_detail_components", False) else 0))

  def _set_world_mask(self, world_mask):
    """Publish the current [B] int32 recovery mask in the dims suffix."""
    torch = self._torch
    b = int(self.batch_size)
    dims = self._constants["solver_dims"]
    offset = self._world_mask_offset
    target = dims[offset:offset + b]
    if world_mask is None:
      target.fill_(1)
      return
    if (not isinstance(world_mask, torch.Tensor)
        or world_mask.device != self._constants["solver_dims"].device
        or world_mask.dtype != torch.int32
        or tuple(world_mask.shape) != (b,)
        or not world_mask.is_contiguous()):
      raise ValueError("world_mask must be contiguous device int32 [batch]")
    target.copy_(world_mask)

  def _copy_world_masked(self, destination, source, world_mask, *, width,
                         source_stride, destination_stride,
                         source_offset=0, destination_offset=0, add=False):
    """Copy/add a per-world flat slice, leaving masked worlds untouched.

    When a recovery mask is present the operation uses a small shader and a
    persistent dimensions record. This avoids both device-to-host mask reads
    and a temporary batch-sized ``where`` result during cached VEL refresh.
    Offsets address slices inside each world's flat record, which also covers
    the strided workspace-debug view.
    """
    torch, b = self._torch, int(self.batch_size)
    width = int(width)
    source_stride, destination_stride = (int(source_stride),
                                         int(destination_stride))
    source_offset, destination_offset = (int(source_offset),
                                         int(destination_offset))
    if width < 0 or min(source_stride, destination_stride,
                        source_offset, destination_offset) < 0:
      raise ValueError("masked world copy dimensions must be nonnegative")
    if (source_offset + width > source_stride
        or destination_offset + width > destination_stride
        or source.numel() < b * source_stride
        or destination.numel() < b * destination_stride
        or not source.is_contiguous() or not destination.is_contiguous()):
      raise ValueError("masked world copy exceeds a per-world record")
    if width == 0:
      return
    if world_mask is None:
      src = source.view(b, source_stride)
      dst = destination.view(b, destination_stride)
      source_slice = src[:, source_offset:source_offset + width]
      destination_slice = dst[:, destination_offset:destination_offset + width]
      if add:
        destination_slice.add_(source_slice)
      else:
        destination_slice.copy_(source_slice)
      return
    dims = self._position_refresh_copy_dims
    dims[0] = b
    dims[1] = width
    dims[2] = source_stride
    dims[3] = destination_stride
    dims[4] = source_offset
    dims[5] = destination_offset
    mask_source = self._constants["solver_dims"][
        self._world_mask_offset:self._world_mask_offset + b]
    dims[6:].copy_(mask_source)
    if source.dtype == torch.float32 and destination.dtype == torch.float32:
      kernel = (self._library.add_selected_world_float if add else
                self._library.copy_selected_world_float)
    elif (source.dtype == torch.int32 and destination.dtype == torch.int32
          and not add):
      kernel = self._library.copy_selected_world_int
    else:
      raise TypeError("masked world copy supports float32 or int32 records")
    kernel(source.reshape(-1), destination.reshape(-1), dims,
           threads=(b * width,), group_size=(128,))

  def _fill_world_masked(self, destination, world_mask, *, width, stride,
                         offset=0, value=0):
    """Fill an owned per-world slice without touching masked records."""
    torch, b = self._torch, int(self.batch_size)
    width, stride, offset = int(width), int(stride), int(offset)
    if width < 0 or stride < 0 or offset < 0 or offset + width > stride:
      raise ValueError("masked world fill exceeds a per-world record")
    if destination.numel() < b * stride or not destination.is_contiguous():
      raise ValueError("masked world fill destination is undersized")
    if width == 0:
      return
    if world_mask is None:
      view = destination.view(b, stride)[:, offset:offset + width]
      view.fill_(value)
      return
    if destination.dtype not in (torch.int32, torch.float32):
      raise TypeError("masked world fill supports float32 or int32 records")
    if destination.dtype == torch.int32 and value not in (0, 1):
      raise TypeError("masked int32 world fill supports zero/one values")
    if destination.dtype == torch.float32 and value != 0:
      raise TypeError("masked float32 world fill supports zero values")
    dims = self._position_refresh_copy_dims
    dims[0] = b
    dims[1] = width
    dims[2] = 0
    dims[3] = stride
    dims[4] = 0
    dims[5] = offset
    mask_source = self._constants["solver_dims"][
        self._world_mask_offset:self._world_mask_offset + b]
    dims[6:].copy_(mask_source)
    if destination.dtype == torch.float32:
      kernel = self._library.clear_selected_world_float
    else:
      kernel = (self._library.fill_selected_world_int_one if value else
                self._library.clear_selected_world_int)
    kernel(destination.reshape(-1), dims, threads=(b * width,),
           group_size=(128,))

  def prepare_workspace(self, batch_size):
    """Preallocate device workspace buffers for contacts, solver, and outputs."""
    d, torch = self.descriptor, self._torch
    b = int(batch_size)
    if b <= 0:
      raise ValueError("batch_size must be positive")
    if self._common_ccd_layout is not None and b != self.batch_size:
      raise ValueError("common CCD workspace batch is fixed at construction")
    self._candidate_generation_token = None
    # R08a allocation gate: enforce the real batch and the exact owned
    # workspace BEFORE partial allocation (lowering-time checks used
    # batch=1 and an approximate memory model).
    from mujoco_metal.capacity import check_capacity as _check_capacity
    from mujoco_metal.capacity import estimate_capacity as _estimate_capacity
    workspace_estimate = _estimate_capacity(
        self._mjmodel_ref, b, d.npairs, d.ncontacts_max, d.nr,
        neq=d.neq, nr_joint=d.nr_joint,
        mass_storage=self._component_mass_storage,
        jacobian_kind=str(getattr(d, "jacobian_kind", "dense")),
        jacobian_nnz=(int(d.jacobian_pattern.nnz)
                      if getattr(d, "jacobian_pattern", None) is not None
                      else None))
    sdf_seed_tile = (_SDF_SEED_TILE if self._has_analytic_sdf_pairs else 0)
    sdf_work_slots = b * d.npairs * sdf_seed_tile
    if sdf_work_slots * _SDF_STAGE_STATE_WORDS > np.iinfo(np.int32).max:
      raise ValueError("staged SDF state exceeds Metal int32 addressing")
    contact_cache_bytes = _pair_contact_cache_bytes(
        b, d.ncontacts_max, d.npairs, sdf_seed_tile)
    workspace_estimate = replace(
        workspace_estimate,
        memory_bytes=workspace_estimate.memory_bytes + contact_cache_bytes,
        memory_breakdown=workspace_estimate.memory_breakdown +
        (("persistent_pair_contact_cache", contact_cache_bytes),))
    _check_capacity(workspace_estimate, limits=self.limits)
    self.batch_size = b
    nv, nc, nr = d.nv, d.ncontacts_max, d.nr
    from mujoco_metal.capacity import primal_scratch_floats, solver_debug_layout
    self._primal_scratch_floats = primal_scratch_floats(
        nv, nr, d.solver_type, self._solver_island_ntree,
        self._solver_island_nbody, self._component_mass_storage)
    self._debug_prefix = nr * nr + 7 * nr
    layout_words = (int(self._component_operator_layout_device.numel())
                    if self._component_operator_layout_device is not None else 0)
    debug_layout = solver_debug_layout(
        nv, nr, d.solver_type, self._solver_island_ntree,
        self._solver_island_nbody, self._component_operator_nnz,
        layout_words, self._component_mass_storage)
    self._base_debug_stride = debug_layout["base_stride"]
    self._qacc_warmstart_offset = debug_layout["qacc_warmstart_offset"]
    self._equality_jacobian_offset = debug_layout["equality_jacobian_offset"]
    self._block_workspace_offset = debug_layout["block_workspace_offset"]
    self._primal_workspace_offset = debug_layout["primal_workspace_offset"]
    self._z_workspace_offset = debug_layout["z_workspace_offset"]
    self._high_low_workspace_offset = debug_layout["high_low_workspace_offset"]
    self._high_low_workspace_length = debug_layout["high_low_workspace_length"]
    self._debug_stride = debug_layout["final_stride"]
    if self._component_operator_layout_device is not None:
      self._component_operator_armature_offset = (
          debug_layout["component_armature_offset"])
      self._component_operator_layout_offset = (
          debug_layout["component_layout_offset"])
      if b * self._debug_stride > np.iinfo(np.int32).max:
        raise ValueError("component solver debug workspace exceeds Metal int32 addressing")

    def empty(size):
      return torch.zeros(max(size, 1), dtype=torch.float32, device=self._device)

    contact_force_size = b * nc * 11
    contact_force_storage = max(contact_force_size, 1)
    joint_force_size = b * max(d.nr_joint, 1)
    force_outputs = empty(contact_force_storage + joint_force_size)
    from mujoco_metal.constraint_jacobian import (
        PACKED_J_CSR, packed_jacobian_device_storage)
    pattern = getattr(d, "jacobian_pattern", None)
    kind = str(getattr(d, "jacobian_kind", "dense"))
    if kind == "sparse" and pattern is None:
      raise ValueError("sparse Jacobian profile has no compiled row pattern")
    jacobian_storage, jacobian_layout = packed_jacobian_device_storage(
        torch, self._device, b, pattern, nr=nr, nv=nv,
        mode=(PACKED_J_CSR if kind == "sparse" else None))
    # Cache records have their own per-world immutable headers/maps and
    # independent numeric values. A plain clone also prevents later header
    # or support updates from aliasing retained POS state.
    position_cache_jacobian = jacobian_storage.clone()
    self._jacobian_layout = jacobian_layout
    self._jacobian_pattern = pattern
    contact_jacobian_size = (b * nc * 6 * nv
                             if jacobian_layout.mode == 0 else 1)
    self._workspace = {
        "contact_row_data": empty(b * nc * 6 * 6),
        "contact_frame": empty(b * nc * 12),
        "contact_pair_count": torch.zeros(
            max(b * d.npairs, 1), dtype=torch.int32, device=self._device),
        "contact_pair_records": empty(b * nc * 25),
        "contact_pair_scratch": empty(max(b * d.npairs * 50 * 12, 1)),
        # Reused source-order tile of per-seed exact contact records. The
        # host loops over all source seeds; tile width is only a storage bound.
        "contact_sdf_seed_records": empty(
            max(b * (d.npairs if self._has_analytic_sdf_pairs else 0)
                * self._sdf_seed_tile * 25, 1)),
        "contact_sdf_seed_valid": torch.zeros(
            max(b * (d.npairs if self._has_analytic_sdf_pairs else 0)
                * self._sdf_seed_tile, 1),
            dtype=torch.int32, device=self._device),
        "contact_sdf_seed_state": empty(
            max(b * (d.npairs if self._has_analytic_sdf_pairs else 0)
                * self._sdf_seed_tile * _SDF_STAGE_STATE_WORDS, 1)),
        "contact_sdf_seed_ctrl": torch.zeros(
            max(b * (d.npairs if self._has_analytic_sdf_pairs else 0)
                * self._sdf_seed_tile, 1),
            dtype=torch.int32, device=self._device),
        "contact_jacobian": empty(contact_jacobian_size),
        "pair_mask": torch.zeros(max(b * max(d.npairs, 1), 1), dtype=torch.float32, device=self._device),
        "workspace_J": jacobian_storage,
        "workspace_debug": empty(b * self._debug_stride),
        "position_context_zero": empty(b * nr),
        # Canonical `{K, B, impedance, position, margin}` context for cached
        # velocity-stage constraint refreshes. Row producers or the caller
        # populate it after the position assembly has completed.
        "position_context": empty(b * nr * 5),
        # Raw per-row source context produced by legacy flex-contact lowering;
        # the general coupled capture kernel fills the other row families.
        "position_assembly_context": empty(b * nr * 5),
        "position_surface_velocity": empty(b * nr),
        "position_extra_aref": empty(b * nr),
        "position_refresh_extra_aref": empty(b * nr),
        "position_qvel": empty(b * nv),
        "position_current_qvel": empty(b * nv),
        "position_current_eq_active": torch.zeros(
            max(b * max(d.neq, 1), 1), dtype=torch.int32, device=self._device),
        "position_cache_eq_active": torch.zeros(
            max(b * max(d.neq, 1), 1), dtype=torch.int32, device=self._device),
        "position_cache_cvel": empty(b * d.nbody * 6),
        "position_cache_cdof": empty(b * nv * 6),
        "position_cache_cdof_dot": empty(b * nv * 6),
        "position_current_cvel": empty(b * d.nbody * 6),
        "position_current_cdof": empty(b * nv * 6),
        "position_current_cdof_dot": empty(b * nv * 6),
        "position_current_body_pos": empty(b * d.nbody * 3),
        "position_current_body_quat": empty(b * d.nbody * 4),
        "position_current_root_com": empty(b * d.nbody * 3),
        "position_current_site_pos": empty(b * d.nsite * 3),
        "position_current_site_quat": empty(b * d.nsite * 4),
        "equality_pose_com": empty(2 * b * d.nbody * 3),
        "equality_motion": empty(b * nv * 12),
        "position_cache_body_pos": empty(b * d.nbody * 3),
        "position_cache_body_quat": empty(b * d.nbody * 4),
        "position_cache_root_com": empty(b * d.nbody * 3),
        "position_cache_site_pos": empty(b * d.nsite * 3),
        "position_cache_site_quat": empty(b * d.nsite * 4),
        "position_cache_J": position_cache_jacobian,
        "position_cache_rows": empty(b * 7 * nr),
        "position_current_aref_low": empty(b * nr),
        "position_cache_aref_low": empty(b * nr),
        "position_cache_qvel": empty(b * nv),
        "position_refresh_old_extra_aref": empty(b * nr),
        # Source impedance is kept independently from the canonical RHS
        # slice. The latter is reused by solver dispatches and is borrowed
        # from a padded per-world debug stride.
        "position_cache_impedance": empty(b * nr),
        "position_cache_contact_data": empty(b * nc * 36),
        "position_cache_contact_frame": empty(b * nc * 12),
        "position_cache_contact_jacobian": empty(contact_jacobian_size),
        "position_cache_slot_packed": torch.full(
            (max(b * nc, 1),), -1, dtype=torch.int32, device=self._device),
        "position_cache_slot_reverse": torch.full(
            (max(b * nc, 1),), -1, dtype=torch.int32, device=self._device),
        "position_cache_slot_count": torch.zeros(b, dtype=torch.int32, device=self._device),
        "position_cache_slot_overflow": torch.zeros(b, dtype=torch.int32, device=self._device),
        "position_cache_pair_packed": torch.full(
            (max(b * d.npairs, 1),), -1, dtype=torch.int32, device=self._device),
        "position_cache_pair_reverse": torch.full(
            (max(b * d.npairs, 1),), -1, dtype=torch.int32, device=self._device),
        "position_cache_pair_count": torch.zeros(b, dtype=torch.int32, device=self._device),
        "position_cache_pair_overflow": torch.zeros(b, dtype=torch.int32, device=self._device),
        "component_mass_rhs": (
            empty(b * (nr + 1) * nv)
            if d.solver_type == int(mujoco.mjtSolver.mjSOL_PGS) else None),
        # Cached component-PGS solves consume a high/low smooth-acceleration
        # pair through the existing qfrc binding. Keep that input contiguous
        # and independent from the row-RHS/factor output buffers.
        "component_smooth_acceleration_pair": (
            empty(2 * b * nv)
            if (nv > 0 and (d.dense_path
                or d.solver_type == int(mujoco.mjtSolver.mjSOL_PGS))) else None),
        "out_force": empty(b * nv),
        # Preserve the original contiguous high-word prefix and keep a
        # world-major low-word result for precision-sensitive consumers.
        "out_acc": empty(2 * b * nv),
        "out_status": torch.zeros(b, dtype=torch.int32, device=self._device),
        "out_diagnostics": empty(b * 10),
        "force_outputs": force_outputs,
        "out_contact_force": force_outputs[:contact_force_storage],
        "out_joint_force": force_outputs[
            contact_force_storage:contact_force_storage + joint_force_size],
    }
    self._position_context_valid = False
    self._position_context_epoch += 1
    self._position_current_eq_jdot_inputs_valid = False
    self._position_context_eq_jdot_inputs_valid = False
    self._position_current_valid = False
    # Stable active pair/slot maps are fixed-shape device workspaces. Logical
    # pair and slot identities remain the public order used by contacts,
    # retained row multipliers, tactile sensors and force queries.
    self._pair_compaction = (self._CompactionWorkspace(
        b, d.npairs, d.npairs, device=self._device) if d.npairs else None)
    self._slot_compaction = (self._CompactionWorkspace(
        b, nc, nc, device=self._device) if nc else None)
    # Canonical map storage exists before the first query/solve. This lets
    # query transactions restore both a mask and its corresponding packed
    # identities instead of leaving a lazily published map from the query.
    if self._pair_compaction is not None:
      self._workspace["pair_maps"] = self._pair_compaction.output
    if self._slot_compaction is not None:
      self._workspace["slot_maps"] = self._slot_compaction.output
    self._empty_compaction_map = torch.full((1,), -1, dtype=torch.int32,
                                            device=self._device)
    from mujoco_metal.row_compaction import CompactionMap
    empty_packed = torch.full((b, nc), -1, dtype=torch.int32, device=self._device)
    empty_reverse = torch.full((b, nc), -1, dtype=torch.int32, device=self._device)
    empty_counts = torch.zeros((b,), dtype=torch.int32, device=self._device)
    empty_overflow = torch.zeros((b,), dtype=torch.int32, device=self._device)
    self._empty_slot_map = CompactionMap(
        empty_packed, empty_reverse, empty_counts, empty_overflow)
    self._empty_eq_active = torch.zeros((b, 0), dtype=torch.int32, device=self._device)
    pair_geoms_host = (np.column_stack((d.geom1, d.geom2)).astype(np.int32)
                       if d.npairs else np.zeros((0, 2), dtype=np.int32))
    eq_trees, eq_activity_ids = self._equality_tree_links(self._mjmodel_ref)
    flex_link_capacity = int(
        d.flex_contact_descriptor.link_capacity
        if d.flex_contact_descriptor is not None else 0)
    self._flex_contact_link_offset = int(d.npairs + eq_trees.shape[0])
    self._flex_contact_link_capacity = flex_link_capacity
    self._active_contact_link_workspace = self._ActiveContactLinkWorkspace(
        b, d.pair_contact_offset, self._pair_tree_ids(self._mjmodel_ref, pair_geoms_host),
        device=self._device,
        capacity=max(d.npairs + eq_trees.shape[0] + flex_link_capacity, 1),
        equality_trees=eq_trees, equality_activity_ids=eq_activity_ids,
        equality_active0=d.eq_active0)
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
    w = self._workspace["workspace_debug"].reshape(b, self._debug_stride)
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
    w = self._workspace["workspace_debug"].reshape(b, self._debug_stride)
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
    w = self._workspace["workspace_debug"].reshape(b, self._debug_stride)
    w[:, nr * nr + 3 * nr:nr * nr + 4 * nr][ids] = 0.0
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None

  def generate_candidates(self, poses, qvel, eq_active=None, *, flex=None,
                          cvel=None, cdof=None, world_mask=None):
    """Run only the contact candidate kernel into persistent workspace buffers.

    Used for same-step BODY-transmission adhesion moments (milestone 007),
    which require candidate contacts before actuator-force assembly. The
    buffers are zeroed first so unwritten slots read exact zero. `run_device`
    calls this internally; behavior is unchanged.
    """
    w, d = self._workspace, self.descriptor
    b, nc = self.batch_size, d.ncontacts_max
    if (d.npairs > 0 and nc > 0
        and (self._common_ccd_hfield_mesh_kernel is not None
             or self._common_ccd_rigid_kernel is not None)):
      # Validate borrowed source-precision poses before clearing persistent
      # masks, rows or status. A rejected input must not partially mutate the
      # preceding assembly or dispatch a kernel with a foreign-device buffer.
      for pose_name, width in (
          ("geom_pos_low", 3), ("geom_pos_tail", 3),
          ("geom_xmat", 9), ("geom_xmat_low", 9),
          ("geom_xmat_tail", 9),
      ):
        pose_value = poses.get(pose_name)
        if pose_value is None:
          raise ValueError(
              f"common source CCD requires FK pose field {pose_name}")
        if not isinstance(pose_value, self._torch.Tensor):
          raise ValueError(
              f"common source CCD pose field {pose_name} must be a tensor")
        if tuple(pose_value.shape) != (b, int(d.ngeom), width):
          raise ValueError(
              f"common source CCD pose field {pose_name} has shape "
              f"{tuple(pose_value.shape)}, expected "
              f"{(b, int(d.ngeom), width)}")
        if (pose_value.dtype != self._torch.float32
            or pose_value.device != poses["geom_pos"].device
            or not pose_value.is_contiguous()):
          raise ValueError(
              f"common source CCD pose field {pose_name} must be contiguous "
              "float32 on the geometry pose device")
    if world_mask is None:
      self._pair_contact_world_mask.fill_(1)
    else:
      if (not isinstance(world_mask, self._torch.Tensor)
          or world_mask.dtype != self._torch.int32
          or world_mask.device != self._pair_contact_world_mask.device
          or tuple(world_mask.shape) != (b,)
          or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous MPS int32 [batch]")
      self._pair_contact_world_mask.copy_(world_mask)
    candidate_world_mask = self._pair_contact_world_mask
    self._reset_jacobian_write_status(world_mask=world_mask)
    if d.npairs > 0:
      self._clear_contact_pair_counts(
          w["contact_pair_count"], self._constants["pair_contact_offsets_dims"],
          threads=(b * d.npairs,), group_size=(128,))
    # Candidate generation opens the canonical CSR value epoch for every row
    # family. Empty contact topology still clears the prior step's values.
    self._clear_jacobian_rows(0, int(d.nr), world_mask=world_mask)
    if d.npairs > 0 and nc > 0:
      if world_mask is None:
        w["contact_row_data"].zero_()
        w["contact_frame"].zero_()
        if self._jacobian_layout.mode == 0:
          w["contact_jacobian"].zero_()
        w["pair_mask"].zero_()
      else:
        self._library.clear_contact_candidate_workspace(
            w["contact_row_data"], w["contact_frame"],
            self._constants["pair_contact_offsets_dims"],
            threads=(b * max(nc, 1) * 36,), group_size=(128,))
        if self._jacobian_layout.mode == 0:
          contact_width = int(nc) * 6 * int(d.nv)
          if contact_width:
            clear_dims = self._pair_contact_jclear_dims
            clear_dims[0] = b
            clear_dims[1] = contact_width
            clear_dims[2] = contact_width
            clear_dims[3] = 0
            clear_dims[4:].copy_(world_mask)
            self._library.clear_selected_jacobian_values(
                w["contact_jacobian"], clear_dims,
                threads=(b * contact_width,), group_size=(128,))
      self._broadphase_kernel(
          poses["geom_pos"].reshape(-1),
          self._constants["geom_rbound"],
          self._constants["pair_geoms"],
          self._constants["pair_margin_gap"],
          self._constants["geom_type"],
          w["pair_mask"],
          self._constants["pair_contact_offsets_dims"],
          threads=(b * d.npairs,), group_size=(1,),
      )
      pair_maps = self._pair_compaction.run(
          w["pair_mask"][:b * d.npairs].reshape(b, d.npairs),
          world_mask=world_mask)
      w["pair_maps"] = pair_maps
      if self._common_ccd_hfield_mesh_kernel is not None or self._common_ccd_rigid_kernel is not None:
        self._fill_world_masked(
            w["out_status"], world_mask, width=1, stride=1, value=0)
      if self._common_ccd_hfield_mesh_kernel is not None:
        self._common_ccd_hfield_mesh_kernel(
            poses["geom_pos"], poses["geom_quat"],
            poses["geom_pos_low"], poses["geom_pos_tail"],
            poses["geom_xmat"], poses["geom_xmat_low"],
            poses["geom_xmat_tail"],
            self._constants["geom_size"], self._constants["geom_type"],
            self._constants["geom_rbound"], self._constants["pair_geoms"],
            self._constants["pair_margin_gap"], self._constants["pair_contact_offsets_dims"],
            pair_maps.logical_to_packed.reshape(-1),
            self._constants["mesh_hull"], self._constants["mesh_hull_info"],
            w["out_status"], threads=(b * d.npairs,), group_size=(1,))
      if self._common_ccd_rigid_kernel is not None:
        self._common_ccd_rigid_kernel(
            poses["geom_pos"], poses["geom_quat"],
            poses["geom_pos_low"], poses["geom_pos_tail"],
            poses["geom_xmat"], poses["geom_xmat_low"],
            poses["geom_xmat_tail"],
            self._constants["geom_size"], self._constants["geom_type"],
            self._constants["pair_geoms"], self._constants["pair_margin_gap"],
            self._constants["pair_contact_offsets_dims"],
            pair_maps.logical_to_packed.reshape(-1),
            self._constants["mesh_hull"], self._constants["mesh_hull_info"],
            w["out_status"], threads=(b * d.npairs,), group_size=(1,))
      if (self._common_ccd_hfield_mesh_kernel is not None
          or self._common_ccd_rigid_kernel is not None):
        self._mask_failed_candidate_worlds_kernel(
            w["out_status"], self._pair_contact_world_mask,
            self._constants["pair_contact_offsets_dims"],
            threads=(b,), group_size=(128,))
        pair_maps = self._pair_compaction.run(
            w["pair_mask"][:b * d.npairs].reshape(b, d.npairs),
            world_mask=self._pair_contact_world_mask)
        w["pair_maps"] = pair_maps
      if self._contact_generic_producer is not None:
        self._contact_generic_producer(
            poses["geom_pos"], poses["geom_quat"],
            self._constants["geom_size"], self._constants["geom_type"],
            self._constants["geom_rbound"], self._constants["pair_geoms"],
            self._constants["pair_margin_gap"],
            self._constants["pair_contact_offsets_dims"],
            pair_maps.logical_to_packed.reshape(-1),
            self._constants["mesh_hull"], self._constants["mesh_hull_info"],
            w["contact_pair_count"], w["contact_pair_records"],
            threads=(b * d.npairs,), group_size=(1,))
      if self._contact_sdf_seed_init is not None:
        seed_capacity = self._sdf_seed_tile
        seed_starts = (range(0, self._sdf_seed_count, self._sdf_seed_tile)
                       if self._sdf_seed_count else (0,))
        for seed_start in seed_starts:
          tile_count = min(self._sdf_seed_tile,
                           max(0, self._sdf_seed_count - seed_start))
          thread_count = b * d.npairs * max(tile_count, 1)
          def set_sdf_stage(mode, step):
            self._sdf_stage_dims.copy_(self._torch.tensor(
                [tile_count, seed_start, seed_capacity, mode, step],
                dtype=self._torch.int32, device=self._device))
          if tile_count == 0:
            set_sdf_stage(0, 0)
            self._contact_sdf_finalize_producer(
                self._constants["geom_type"], self._constants["pair_geoms"],
                self._constants["pair_contact_offsets_dims"],
                pair_maps.logical_to_packed.reshape(-1),
                w["contact_sdf_seed_records"], w["contact_sdf_seed_valid"],
                w["contact_pair_count"], w["contact_pair_records"],
                self._sdf_stage_dims,
                threads=(b * d.npairs,), group_size=(1,))
            continue
          set_sdf_stage(2, 0)
          self._contact_sdf_seed_init(
              poses["geom_pos"], poses["geom_quat"],
              self._constants["geom_size"], self._constants["geom_type"],
              self._constants["pair_geoms"],
              self._constants["pair_contact_offsets_dims"],
              pair_maps.logical_to_packed.reshape(-1),
              self._constants["mesh_hull"], self._constants["mesh_hull_info"],
              w["contact_sdf_seed_valid"],
              w["contact_sdf_seed_state"], w["contact_sdf_seed_ctrl"],
              self._sdf_stage_dims,
              threads=(thread_count,), group_size=(1,))
          for step in range(self._sdf_max_iterations):
            set_sdf_stage(2, step)
            self._contact_sdf_descent_prepare(
                poses["geom_pos"], poses["geom_quat"],
                self._constants["geom_size"], self._constants["geom_type"],
                self._constants["pair_geoms"],
                self._constants["pair_contact_offsets_dims"],
                pair_maps.logical_to_packed.reshape(-1),
                self._constants["mesh_hull"], self._constants["mesh_hull_info"],
                w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
                w["contact_sdf_seed_ctrl"], self._sdf_stage_dims,
                threads=(thread_count,), group_size=(1,))
            self._contact_sdf_line_search(
                poses["geom_pos"], poses["geom_quat"],
                self._constants["geom_size"], self._constants["geom_type"],
                self._constants["pair_geoms"],
                self._constants["pair_contact_offsets_dims"],
                pair_maps.logical_to_packed.reshape(-1),
                self._constants["mesh_hull"], self._constants["mesh_hull_info"],
                w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
                w["contact_sdf_seed_ctrl"], self._sdf_stage_dims,
                threads=(thread_count,), group_size=(1,))
          set_sdf_stage(0, 0)
          self._contact_sdf_phase_reset(
              poses["geom_pos"], poses["geom_quat"],
              self._constants["geom_size"], self._constants["geom_type"],
              self._constants["pair_geoms"],
              self._constants["pair_contact_offsets_dims"],
              pair_maps.logical_to_packed.reshape(-1),
              w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
              w["contact_sdf_seed_ctrl"], self._sdf_stage_dims,
              threads=(thread_count,), group_size=(1,))
          self._contact_sdf_descent_prepare(
              poses["geom_pos"], poses["geom_quat"],
              self._constants["geom_size"], self._constants["geom_type"],
              self._constants["pair_geoms"],
              self._constants["pair_contact_offsets_dims"],
              pair_maps.logical_to_packed.reshape(-1),
              self._constants["mesh_hull"], self._constants["mesh_hull_info"],
              w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
              w["contact_sdf_seed_ctrl"], self._sdf_stage_dims,
              threads=(thread_count,), group_size=(1,))
          self._contact_sdf_line_search(
              poses["geom_pos"], poses["geom_quat"],
              self._constants["geom_size"], self._constants["geom_type"],
              self._constants["pair_geoms"],
              self._constants["pair_contact_offsets_dims"],
              pair_maps.logical_to_packed.reshape(-1),
              self._constants["mesh_hull"], self._constants["mesh_hull_info"],
              w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
              w["contact_sdf_seed_ctrl"], self._sdf_stage_dims,
              threads=(thread_count,), group_size=(1,))
          self._contact_sdf_publish_normal(
              poses["geom_pos"], poses["geom_quat"],
              self._constants["geom_size"], self._constants["geom_type"],
              self._constants["pair_geoms"],
              self._constants["pair_contact_offsets_dims"],
              pair_maps.logical_to_packed.reshape(-1),
              self._constants["mesh_hull"], self._constants["mesh_hull_info"],
              w["contact_sdf_seed_valid"], w["contact_sdf_seed_state"],
              self._sdf_stage_dims,
              threads=(thread_count,), group_size=(1,))
          self._contact_sdf_publish_contact(
              poses["geom_pos"], poses["geom_quat"], self._constants["geom_type"],
              self._constants["pair_geoms"],
              self._constants["pair_contact_offsets_dims"],
              pair_maps.logical_to_packed.reshape(-1),
              w["contact_sdf_seed_records"], w["contact_sdf_seed_valid"],
              w["contact_sdf_seed_state"], self._sdf_stage_dims,
              threads=(thread_count,), group_size=(1,))
          self._contact_sdf_finalize_producer(
              self._constants["geom_type"], self._constants["pair_geoms"],
              self._constants["pair_contact_offsets_dims"],
              pair_maps.logical_to_packed.reshape(-1),
              w["contact_sdf_seed_records"], w["contact_sdf_seed_valid"],
              w["contact_pair_count"], w["contact_pair_records"],
              self._sdf_stage_dims,
              threads=(b * d.npairs,), group_size=(1,))
      if self._contact_mesh_sdf_producer is not None:
        self._contact_mesh_sdf_producer(
            poses["geom_pos"], poses["geom_quat"],
            self._constants["geom_size"], self._constants["geom_type"],
            self._constants["pair_geoms"],
            self._constants["pair_contact_offsets_dims"],
            pair_maps.logical_to_packed.reshape(-1),
            self._constants["mesh_hull"], self._constants["mesh_hull_info"],
            w["contact_pair_count"], w["contact_pair_records"],
            w["contact_pair_scratch"],
            threads=(b * d.npairs,), group_size=(1,))
      self._contact_kernel(
          self._constants["geom_bodyid"], poses["body_pos"], poses["body_quat"],
          poses["joint_anchor"], poses["joint_axis"], qvel.reshape(-1),
          self._constants["body_parentid"], self._constants["body_jntadr"],
          self._constants["body_jntnum"], self._constants["jnt_type"],
          self._constants["jnt_dofadr"], self._constants["body_invweight0"],
          self._constants["pair_geoms"], self._constants["pair_margin_gap"],
          self._constants["pair_solref"], self._constants["pair_solimp"],
          self._constants["pair_condim"], self._constants["pair_friction"],
          self._constants["pair_solreffriction"],
          self._constants["pair_contact_offsets_dims"],
          w["contact_row_data"], w["contact_frame"],
          (w["workspace_J"] if self._jacobian_layout.mode == 1
           else w["contact_jacobian"]),
          pair_maps.logical_to_packed.reshape(-1),
          w["contact_pair_count"], w["contact_pair_records"],
          threads=(b * d.npairs,), group_size=(1,),
      )
      slot_flags = w["contact_row_data"][:b * nc * 36].reshape(b, nc, 36)[:, :, 0]
      slot_maps = self._slot_compaction.run(slot_flags,
                                             world_mask=candidate_world_mask)
      w["slot_maps"] = slot_maps
    else:
      slot_maps = self._empty_slot_map
      w["slot_maps"] = slot_maps

    # Keep only current narrowphase contacts in this graph. Equality waking is
    # a separate model-ordered pass in the sleep scheduler, matching pinned
    # mj_wakeEquality's pair and flex-hyperedge behavior.
    (w["active_tree_links"], w["active_tree_link_overflow"]
     ) = self._active_contact_link_workspace.run_contacts(
         slot_maps, world_mask=candidate_world_mask)

    # Flex contact slots are generated by their source-derived detector and
    # row assembler.  Keep the fixed candidate rows in a separate persistent
    # block; append its static link-capacity range after rigid/equality links
    # so sleep scheduling needs no device count readback or per-step allocation.
    self._flex_contact_current = None
    flex_desc = d.flex_contact_descriptor
    if (flex is not None and flex_desc is not None
        and int(flex_desc.row_capacity) > 0):
      flex_poses = dict(poses)
      if "cdof" not in flex_poses:
        if cdof is None:
          raise ValueError("flex contact assembly requires smooth cdof")
        flex_poses["cdof"] = cdof
      bundle = flex.run_native_contact_rows(
          flex_poses, qvel, cvel=cvel, include_wake_links=True,
          world_mask=candidate_world_mask,
          **({"packed_jacobian": w["workspace_J"],
              "canonical_row_offset": int(d.flex_contact_base)}
             if self._jacobian_layout.mode == 1 else {}))
      self._flex_contact_current = bundle
      contact_result = bundle["contact_result"]
      flex_links = contact_result.get("active_tree_links")
      flex_overflow = contact_result.get("active_tree_link_overflow")
      if flex_links is not None:
        expected = (int(self.batch_size), self._flex_contact_link_capacity, 2)
        if (tuple(flex_links.shape) != expected
            or flex_links.dtype != self._torch.int32
            or flex_links.device.type != self._device.type
            or not flex_links.is_contiguous()):
          raise ValueError(
              "flex active tree links must be contiguous MPS int32 with "
              f"shape {expected}")
        start = self._flex_contact_link_offset
        stop = start + self._flex_contact_link_capacity
        w["active_tree_links"][:, start:stop, :].copy_(flex_links)
      if flex_overflow is not None:
        if (tuple(flex_overflow.shape) != (int(self.batch_size),)
            or flex_overflow.dtype != self._torch.int32
            or flex_overflow.device.type != self._device.type
            or not flex_overflow.is_contiguous()):
          raise ValueError(
              "flex link overflow must be contiguous MPS int32 [batch]")
        w["active_tree_link_overflow"].copy_(self._torch.maximum(
            w["active_tree_link_overflow"], flex_overflow))
    elif self._flex_contact_link_capacity:
      start = self._flex_contact_link_offset
      stop = start + self._flex_contact_link_capacity
      w["active_tree_links"][:, start:stop, :].fill_(-1)
    self._candidate_generation_epoch += 1
    token = (self, self._candidate_generation_epoch, object())
    self._candidate_generation_token = token
    return token

  def set_broadphase_pruning(self, enabled):
    """Toggle narrowphase broadphase-fed pruning (R08/F1 test hook).

    Enabled (default) pruned pairs skip the narrowphase and stamp a finite
    sentinel; disabled runs every pair. Physics must be identical either
    way (equivalence test).
    """
    packed_rows = 2 if self._jacobian_layout.mode == 1 else 0
    self._constants["c_dims"][9] = packed_rows | (1 if enabled else 0)

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
    w = self._workspace
    nc = d.ncontacts_max
    cone_type = int(self._mjmodel_ref.opt.cone)
    rows = w["contact_row_data"][: self.batch_size * max(nc, 1) * 36].detach().cpu().numpy().reshape(self.batch_size, max(nc, 1), 6, 6)
    out = []
    for env in range(self.batch_size):
      mask_act, slot_act = counts[env]
      act_rows = d.nr_joint
      for s in range(nc):
        if rows[env, s, 0, 0] > 0.5:
          cdim = int(d.contact_condim[s]) if s < len(d.contact_condim) else 3
          r_con = 1 if cdim == 1 else (2 * (cdim - 1) if cone_type == 0 else cdim)
          act_rows += r_con
      act_rows = min(d.nr, act_rows)
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
    jacobian = (w["workspace_J"]
                if nc and self._jacobian_layout.mode == 1
                else w["contact_jacobian"] if nc else zeros)
    return {
        "frame": frame,
        "jacobian": jacobian,
        "pair_geoms": self._constants["pair_geoms"],
        "geom_bodyid": self._constants["geom_bodyid"],
        "pair_offset": self._constants["pair_contact_offset"],
        "dims": self._constants["body_dims"],
    }

  def _write_jacobian_rows(self, row_start, source_rows, *, active=None):
    """Copy one producer-local row block into canonical J storage.

    Sparse profiles gather only compiled support columns from the producer's
    local block and scatter those values into CSR slots. They never create a
    canonical dense ``[B,nr,nv]`` staging tensor.
    """
    torch, d, w = self._torch, self.descriptor, self._workspace
    b, nr, nv = int(self.batch_size), int(d.nr), int(d.nv)
    row_start = int(row_start)
    if source_rows.ndim != 3 or tuple(source_rows.shape)[0] != b:
      raise ValueError("Jacobian producer rows must have shape [B,R,nv]")
    count = int(source_rows.shape[1])
    if tuple(source_rows.shape[2:]) != (nv,):
      raise ValueError("Jacobian producer DOF width does not match model")
    if row_start < 0 or row_start + count > nr:
      raise ValueError("Jacobian producer span exceeds canonical rows")
    if active is not None and tuple(active.shape) != (b, count):
      raise ValueError("Jacobian producer activity does not match its row span")
    if self._jacobian_layout.mode == 1:
      from mujoco_metal.constraint_jacobian import (
          packed_jacobian_scatter_rows_torch)
      packed_jacobian_scatter_rows_torch(
          torch, w["workspace_J"], b, self._jacobian_layout,
          self._jacobian_pattern, source_rows, row_start,
          row_count=count, active=active)
      return
    values = source_rows if active is None else source_rows * active.unsqueeze(-1)
    layout = self._jacobian_layout
    for world in range(b):
      base = world * layout.stride_words + layout.values_offset
      begin = base + row_start * nv
      end = begin + count * nv
      w["workspace_J"][begin:end].copy_(values[world].reshape(-1))

  def _clear_jacobian_rows(self, row_start, row_stop, world_mask=None):
    """Clear a canonical row interval while preserving packed CSR metadata."""
    torch, w = self._torch, self._workspace
    b, nr, nv = int(self.batch_size), int(self.descriptor.nr), int(self.descriptor.nv)
    row_start, row_stop = int(row_start), int(row_stop)
    if not 0 <= row_start <= row_stop <= nr:
      raise ValueError("Jacobian clear interval exceeds canonical rows")
    pattern, layout = self._jacobian_pattern, self._jacobian_layout
    if world_mask is not None:
      if layout.mode == 0:
        first = row_start * nv
        width = (row_stop - row_start) * nv
      else:
        first = int(pattern.row_offsets[row_start])
        width = int(pattern.row_offsets[row_stop]) - first
      if width:
        dims = self._pair_contact_jclear_dims
        dims[0] = b
        dims[1] = width
        dims[2] = int(layout.stride_words)
        dims[3] = int(layout.values_offset + first)
        dims[4:].copy_(world_mask)
        storage = w["workspace_J"]
        if layout.mode == 1:
          storage = storage.view(torch.float32)
        self._library.clear_selected_jacobian_values(
            storage, dims, threads=(b * width,), group_size=(128,))
      return
    if layout.mode == 0:
      for world in range(b):
        base = world * layout.stride_words + layout.values_offset
        begin, end = base + row_start * nv, base + row_stop * nv
        w["workspace_J"][begin:end].zero_()
      return
    first = int(pattern.row_offsets[row_start])
    last = int(pattern.row_offsets[row_stop])
    if first == last:
      return
    for world in range(b):
      value_base = world * layout.stride_words + layout.values_offset
      # CSR row intervals map to a contiguous value interval. Advanced tensor
      # indexing would create a temporary and leave the persistent packed
      # values untouched, allowing the next contact assembly to accumulate
      # onto the previous step's Jacobian.
      w["workspace_J"][value_base + first:value_base + last].zero_()

  def materialize_jacobian(self, storage=None):
    """Explicitly materialize dense canonical J for query/diagnostic APIs."""
    w, d = self._workspace, self.descriptor
    layout = getattr(self, "_jacobian_layout", None)
    packed = w["workspace_J"] if storage is None else storage
    if layout is None:
      return packed[:self.batch_size * d.nr * d.nv].reshape(
          self.batch_size, d.nr, d.nv)
    from mujoco_metal.constraint_jacobian import materialize_packed_jacobian_torch
    return materialize_packed_jacobian_torch(
        self._torch, packed, self.batch_size, layout,
        getattr(self, "_jacobian_pattern", None))

  def _reset_jacobian_write_status(self, world_mask=None):
    """Clear per-world sparse-write errors without touching typed maps."""
    storage = self._workspace.get("workspace_J")
    layout = getattr(self, "_jacobian_layout", None)
    if storage is None or layout is None:
      return
    words = int(layout.stride_words)
    if words < 12 or storage.numel() < int(self.batch_size) * words:
      raise RuntimeError("packed Jacobian allocation no longer matches its ABI")
    records = storage.view(self._torch.int32)
    if world_mask is not None:
      dims = self._pair_contact_status_dims
      dims[2:].copy_(world_mask)
      self._library.reset_selected_jacobian_status(
          records, dims, threads=(int(self.batch_size),), group_size=(1,))
      return
    for world in range(int(self.batch_size)):
      records[world * words + 10] = 0

  def refresh_position_references(self, position_context, qvel,
                                 surface_velocity=None, extra_aref=None,
                                 world_mask=None):
    """Refresh canonical row references while preserving cached position rows.

    ``position_context`` is contiguous float32 MPS ``[B,nr,5]`` with each
    row ``{K, B, impedance, position, margin}``. This method updates only the
    cached ``ar`` view; it does not run FK, narrowphase, equality assembly, or
    alter cached ``J/R/bounds/activity``. Callers supply ``extra_aref`` for
    terms outside that affine reference law, notably equality Jdot-v, and
    ``surface_velocity`` for prescribed relative motion.
    """
    torch, d = self._torch, self.descriptor
    b, nr, nv = int(self.batch_size), int(d.nr), int(d.nv)
    self._set_world_mask(world_mask)

    def validate(name, tensor, shape):
      if (not isinstance(tensor, torch.Tensor)
          or tuple(tensor.shape) != shape
          or tensor.dtype != torch.float32
          or tensor.device.type != self._device.type
          or (self._device.index is not None
              and tensor.device.index != self._device.index)
          or not tensor.is_contiguous()):
        raise ValueError(
            f"{name} must be contiguous float32 MPS with shape {shape}")

    validate("position_context", position_context, (b, nr, 5))
    validate("qvel", qvel, (b, nv))
    zero = self._workspace["position_context_zero"]
    surface = zero
    extra = zero
    if surface_velocity is not None:
      validate("surface_velocity", surface_velocity, (b, nr))
      surface = surface_velocity
    if extra_aref is not None:
      validate("extra_aref", extra_aref, (b, nr))
      extra = extra_aref
    self._set_world_mask(world_mask)
    if nr:
      self._refresh_position_reference_kernel(
          self._workspace["workspace_J"], qvel, position_context,
          surface, extra, self._workspace["workspace_debug"],
          self._constants["solver_dims"],
          threads=(b * nr,), group_size=(128,))
    debug = self._workspace["workspace_debug"].reshape(b, self._debug_stride)
    return debug[:, nr * nr + nr:nr * nr + 2 * nr]

  def _refresh_cached_aref_low(self, qvel, surface_velocity, new_extra_aref,
                               old_extra_aref, world_mask=None):
    """Carry the captured two-float row reference to a new cached velocity.

    The high-word refresh above retains the public float32 ABI. This companion
    updates its residual word from the immutable POS reference using the
    velocity delta, source-derived B residuals, and the old/new extra terms.
    It does not rebuild a row or allocate a tensor.
    """
    torch, w = self._torch, self._workspace
    b, nv, nr = (int(self.batch_size), int(self.descriptor.nv),
                 int(self.descriptor.nr))
    if not nr:
      return
    debug = w["workspace_debug"].reshape(b, self._debug_stride)
    self._refresh_aref_low_kernel(
        w["position_cache_J"], qvel.reshape(-1),
        w["position_cache_qvel"],
        w["position_context"][:b * nr * 5],
        self._constants["row_velocity_scale_low"],
        old_extra_aref.reshape(-1), new_extra_aref.reshape(-1),
        w["position_cache_rows"], w["position_cache_aref_low"],
        debug.reshape(-1), self._constants["solver_dims"],
        threads=(b * nr,), group_size=(128,))
    self._copy_world_masked(
        w["position_current_aref_low"], debug.reshape(-1), world_mask,
        width=nr, source_stride=self._debug_stride,
        destination_stride=nr,
        source_offset=self._aref_low_debug_offset)

  def refresh_velocity_context(self, context, qpos, qvel, cvel=None, cdof=None,
                               cdof_dot=None, extra_aref=None, eq_active=None,
                               world_mask=None):
    """Refresh captured row ``aref`` at a new velocity without solving.

    This is the ``mj_fwdVelocity`` consumer for a retained POS assembly. It
    restores the canonical row/contact views from the captured cache, adds
    the source-derived connect/weld ``-Jdot*v`` term, and updates only row
    references. It does not rebuild contacts, FK, Jacobians, regularizers,
    bounds, mass factors, or optimizer output.
    """
    if (not isinstance(context, dict) or context.get("_owner") is not self
        or not self._position_context_valid
        or context.get("_epoch") != self._position_context_epoch):
      raise ValueError("position context is stale or belongs to another solver")
    torch, d = self._torch, self.descriptor
    b, nv, nr = int(self.batch_size), int(d.nv), int(d.nr)
    _validate_cached_tensor(torch, self._device, "qpos", qpos,
                            (b, int(d.nq)), torch.float32)
    _validate_cached_tensor(torch, self._device, "qvel", qvel,
                            (b, nv), torch.float32)
    has_differential_eq = bool(d.neq and np.any(
        np.isin(np.asarray(d.eq_type, dtype=np.int32), (0, 1))))
    if cvel is not None:
      _validate_cached_tensor(torch, self._device, "cvel", cvel,
                              (b, int(d.nbody), 6), torch.float32)
    if cdof is not None:
      _validate_cached_tensor(torch, self._device, "cdof", cdof,
                              (b, nv, 6), torch.float32)
    if cdof_dot is not None:
      _validate_cached_tensor(torch, self._device, "cdof_dot", cdof_dot,
                              (b, nv, 6), torch.float32)
    if has_differential_eq and (cvel is None or cdof is None or cdof_dot is None):
      raise ValueError("cached connect/weld refresh requires current cvel, cdof, and cdof_dot")
    if has_differential_eq and not (
        self._position_context_eq_jdot_inputs_valid
        and context.get("_eq_jdot_inputs_valid") is True):
      raise ValueError("captured position context lacks equality motion metadata")
    position = context.get("position_context")
    surface = context.get("surface_velocity")
    stored_extra = context.get("extra_aref")
    _validate_cached_tensor(torch, self._device, "context.position_context",
                            position, (b, nr, 5), torch.float32)
    _validate_cached_tensor(torch, self._device, "context.surface_velocity",
                            surface, (b, nr), torch.float32)
    _validate_cached_tensor(torch, self._device, "context.extra_aref",
                            stored_extra, (b, nr), torch.float32)
    if extra_aref is not None:
      _validate_cached_tensor(torch, self._device, "extra_aref", extra_aref,
                              (b, nr), torch.float32)
    if eq_active is not None:
      _validate_cached_tensor(torch, self._device, "eq_active", eq_active,
                              ((b, int(d.neq)) if d.neq else (b, 1)),
                              torch.int32)

    self._set_world_mask(world_mask)
    w = self._workspace
    debug = w["workspace_debug"].reshape(b, self._debug_stride)
    jacobian_stride = int(w["workspace_J"].numel()) // max(b, 1)
    self._copy_world_masked(
        w["workspace_J"], w["position_cache_J"], world_mask,
        width=jacobian_stride, source_stride=jacobian_stride,
        destination_stride=jacobian_stride)
    self._copy_world_masked(
        w["position_qvel"], w["position_cache_qvel"], world_mask,
        width=nv, source_stride=nv, destination_stride=nv)
    row_base = nr * nr
    self._copy_world_masked(
        w["workspace_debug"], w["position_cache_rows"], world_mask,
        width=7 * nr, source_stride=7 * nr,
        destination_stride=self._debug_stride, destination_offset=row_base)
    self._copy_world_masked(
        w["workspace_debug"], w["position_cache_impedance"], world_mask,
        width=nr, source_stride=nr, destination_stride=self._debug_stride,
        destination_offset=row_base + 2 * nr)
    self._copy_world_masked(
        w["workspace_debug"], w["position_cache_aref_low"], world_mask,
        width=nr, source_stride=nr, destination_stride=self._debug_stride,
        destination_offset=self._aref_low_debug_offset)
    self._copy_world_masked(
        w["position_current_aref_low"], w["workspace_debug"], world_mask,
        width=nr, source_stride=self._debug_stride,
        destination_stride=nr, source_offset=self._aref_low_debug_offset)
    self._copy_world_masked(
        w["contact_row_data"], w["position_cache_contact_data"], world_mask,
        width=w["contact_row_data"].numel() // b,
        source_stride=w["position_cache_contact_data"].numel() // b,
        destination_stride=w["contact_row_data"].numel() // b)
    self._copy_world_masked(
        w["contact_frame"], w["position_cache_contact_frame"], world_mask,
        width=w["contact_frame"].numel() // b,
        source_stride=w["position_cache_contact_frame"].numel() // b,
        destination_stride=w["contact_frame"].numel() // b)
    if getattr(getattr(self, "_jacobian_layout", None), "mode", 0) == 0:
      self._copy_world_masked(
          w["contact_jacobian"], w["position_cache_contact_jacobian"],
          world_mask,
          width=w["contact_jacobian"].numel() // b,
          source_stride=w["position_cache_contact_jacobian"].numel() // b,
          destination_stride=w["contact_jacobian"].numel() // b)
    for prefix, name in (("slot", "slot_maps"), ("pair", "pair_maps")):
      maps = w.get(name)
      if maps is not None:
        for destination, source_name in (
            (maps.packed_to_logical, f"position_cache_{prefix}_packed"),
            (maps.logical_to_packed, f"position_cache_{prefix}_reverse"),
            (maps.active_count, f"position_cache_{prefix}_count"),
            (maps.overflow, f"position_cache_{prefix}_overflow"),
        ):
          width = destination.numel() // b
          self._copy_world_masked(
              destination, w[source_name], world_mask, width=width,
              source_stride=width, destination_stride=width)
    old_extra = w["position_refresh_old_extra_aref"][:b * nr].view(b, nr)
    self._copy_world_masked(
        old_extra, stored_extra, world_mask, width=nr,
        source_stride=nr, destination_stride=nr)
    if has_differential_eq:
      old_jdot = w["position_context_zero"][:b * nr].view(b, nr)
      self._refresh_equality_jdot(
          w["position_cache_qvel"].view(b, nv),
          w["position_cache_cvel"].view(b, d.nbody, 6),
          w["position_cache_cdof"].view(b, nv, 6),
          w["position_cache_cdof_dot"].view(b, nv, 6), old_jdot,
          eq_active=w["position_cache_eq_active"])
      self._copy_world_masked(
          old_extra, old_jdot, world_mask, width=nr,
          source_stride=nr, destination_stride=nr, add=True)
    refresh_extra = w["position_refresh_extra_aref"][:b * nr].view(b, nr)
    self._copy_world_masked(
        refresh_extra, stored_extra, world_mask, width=nr,
        source_stride=nr, destination_stride=nr)
    if extra_aref is not None:
      self._copy_world_masked(
          refresh_extra, extra_aref, world_mask, width=nr,
          source_stride=nr, destination_stride=nr, add=True)
    if has_differential_eq:
      equality_aref = w["position_context_zero"][:b * nr].view(b, nr)
      self._refresh_equality_jdot(
          qvel, cvel, cdof, cdof_dot, equality_aref, eq_active=eq_active)
      self._copy_world_masked(
          refresh_extra, equality_aref, world_mask, width=nr,
          source_stride=nr, destination_stride=nr, add=True)
    result = self.refresh_position_references(
        position, qvel, surface, refresh_extra, world_mask=world_mask)
    self._refresh_cached_aref_low(
        qvel, surface, refresh_extra, old_extra, world_mask=world_mask)
    return result

  def _refresh_equality_jdot(self, qvel, cvel, cdof, cdof_dot, out, *,
                             assembly_current=False, eq_active=None):
    """Write the source-derived current equality ``-Jdot*v`` row vector."""
    b, nv, nr = self.batch_size, self.descriptor.nv, self.descriptor.nr
    if not self.descriptor.neq or not nr:
      out.zero_()
      return out
    d = self.descriptor
    pose_prefix = "position_current_" if assembly_current else "position_cache_"
    body_pos = self._workspace[pose_prefix + "body_pos"]
    body_quat = self._workspace[pose_prefix + "body_quat"]
    root_com = self._workspace[pose_prefix + "root_com"]
    site_pos = self._workspace[pose_prefix + "site_pos"]
    site_quat = self._workspace[pose_prefix + "site_quat"]
    jacobian = (self._workspace["workspace_J"] if assembly_current
                else self._workspace["position_cache_J"])
    if eq_active is None:
      eq_active = (self._workspace["position_current_eq_active"] if assembly_current
                   else self._workspace["position_cache_eq_active"])
    self._refresh_equality_jdot_kernel(
        qvel.reshape(-1), cvel.reshape(-1), cdof.reshape(-1), cdof_dot.reshape(-1),
        body_pos, body_quat, root_com, site_pos, site_quat,
        self._constants["eq_type"], self._constants["eq_objtype"],
        self._constants["eq_obj"], self._constants["eq_rowadr"],
        self._constants["eq_data"], self._constants["site_bodyid"],
        self._constants["body_rootid"], self._constants["body_weldid"],
        self._constants["body_dofadr"], self._constants["body_dofnum"],
        self._constants["dof_parentid"], self._constants["dof_bodyid"],
        self._constants["dof_jntid"], self._constants["jnt_type"],
        self._constants["joint_dofadr"], out.reshape(-1),
        jacobian,
        self._constants["body_invweight0"],
        eq_active.reshape(-1), self._constants["solver_dims"],
        threads=(b,), group_size=(1,))
    return out

  def capture_position_context(self, position_context=None, *,
                               surface_velocity=None, extra_aref=None):
    """Capture row references for later solver-only velocity stages.

    This must follow a completed :meth:`run_device` assembly. The context
    uses the pinned per-row `{K, B, impedance, position, margin}` ABI. Optional
    surface velocity and extra aref terms are borrowed inputs and must remain
    valid through the cached stage. The returned token is tied to this
    captured row cache and this program instance. A later ordinary forward may
    overwrite borrowed solver views without invalidating this retained cache;
    re-preparing the workspace or capturing a replacement invalidates it.
    """
    if self._assembly_generation <= 0:
      raise RuntimeError("position context requires a completed row assembly")
    # A capture attempt replaces the previous cache epoch, including when
    # validation or a later device operation fails. Never leave an older
    # token apparently valid after a partial replacement attempt.
    self._position_context_valid = False
    self._position_context_epoch += 1
    capture_epoch = self._position_context_epoch
    self._position_context_eq_jdot_inputs_valid = False
    if not self._position_current_valid:
      raise RuntimeError("position context requires a successful current assembly")
    torch, d = self._torch, self.descriptor
    b, nv, nr = int(self.batch_size), int(d.nv), int(d.nr)
    expected = (b, nr, 5)

    def validate(name, value, shape):
      if (not isinstance(value, torch.Tensor)
          or tuple(value.shape) != shape
          or value.dtype != torch.float32
          or value.device.type != self._device.type
          or (self._device.index is not None
              and value.device.index != self._device.index)
          or not value.is_contiguous()):
        raise ValueError(
            f"{name} must be contiguous float32 {self._device.type} with shape {shape}")

    # Validate every caller-owned tensor before the capture kernel can write
    # its destination or any persistent cache. This matters when callers
    # provide an explicit context together with malformed optional terms.
    if position_context is not None:
      validate("position_context", position_context, expected)
    if surface_velocity is not None:
      validate("surface_velocity", surface_velocity, (b, nr))
    if extra_aref is not None:
      validate("extra_aref", extra_aref, (b, nr))
    has_differential_eq = d.neq and np.any(
        np.isin(np.asarray(d.eq_type, dtype=np.int32), (0, 1)))
    if has_differential_eq and not self._position_current_eq_jdot_inputs_valid:
      raise ValueError(
          "cached connect/weld capture requires position-stage cvel, cdof, and cdof_dot")

    # Snapshot all inputs which define the captured POS stage. Normal
    # forwards write only position_current_* buffers and borrowed row views;
    # this cache is owned by the returned epoch until another capture or
    # workspace replacement.
    w = self._workspace
    for source_name, cache_name in (
        ("position_current_qvel", "position_qvel"),
        ("position_current_eq_active", "position_cache_eq_active"),
        ("position_current_cvel", "position_cache_cvel"),
        ("position_current_cdof", "position_cache_cdof"),
        ("position_current_cdof_dot", "position_cache_cdof_dot"),
        ("position_current_body_pos", "position_cache_body_pos"),
        ("position_current_body_quat", "position_cache_body_quat"),
        ("position_current_root_com", "position_cache_root_com"),
        ("position_current_site_pos", "position_cache_site_pos"),
        ("position_current_site_quat", "position_cache_site_quat"),
    ):
      w[cache_name].copy_(w[source_name])
    if position_context is None:
      position_context = self._workspace["position_context"][:b * nr * 5].view(expected)
      if nr:
        self._capture_position_context_kernel(
            self._workspace["workspace_J"], self._workspace["position_qvel"],
            self._constants["row_velocity_scales"],
            self._workspace["workspace_debug"], position_context,
            self._constants["solver_dims"], threads=(b * nr,), group_size=(128,))
        if int(getattr(d, "n_flex_contact_rows", 0)):
          start = int(d.flex_contact_base)
          stop = min(nr, start + int(d.n_flex_contact_rows))
          position_context[:, start:stop, :].copy_(
              self._workspace["position_assembly_context"].view(b, nr, 5)[:, start:stop, :])
        flex_eq = np.flatnonzero(np.isin(
            np.asarray(d.eq_type, dtype=np.int32),
            (_EQ_FLEX, _EQ_FLEXVERT, _EQ_FLEXSTRAIN)))
        for equality in flex_eq:
          start = int(d.eq_rowadr[equality])
          stop = min(nr, start + int(d.eq_rownum[equality]))
          if start < stop:
            position_context[:, start:stop, :].copy_(
                self._workspace["position_assembly_context"].view(
                    b, nr, 5)[:, start:stop, :])
      else:
        position_context.zero_()
      if has_differential_eq:
        self._workspace["position_cache_J"].copy_(self._workspace["workspace_J"])
        old_jdot_aref = self._workspace["position_context_zero"][:b * nr].view(b, nr)
        old_jdot_aref = self._refresh_equality_jdot(
            self._workspace["position_qvel"].view(b, d.nv),
            self._workspace["position_cache_cvel"].view(b, d.nbody, 6),
            self._workspace["position_cache_cdof"].view(b, d.nv, 6),
            self._workspace["position_cache_cdof_dot"].view(b, d.nv, 6),
            old_jdot_aref)
        # capture_cached_position_context encodes `-ar - B*Jv`, which is
        # `K*imp*(pos-margin) - old_extra`. Add old -Jdot*v back once to
        # recover the position-only residual before the next velocity stage
        # supplies its freshly computed -Jdot*v term.
        position_context[:, :, 3].add_(old_jdot_aref)
    stored = self._workspace["position_context"][:b * nr * 5].view(b, nr, 5)
    stored.copy_(position_context)
    stored_surface = self._workspace["position_surface_velocity"][:b * nr].view(b, nr)
    stored_extra = self._workspace["position_extra_aref"][:b * nr].view(b, nr)
    if surface_velocity is None:
      stored_surface.zero_()
    else:
      stored_surface.copy_(surface_velocity)
    if extra_aref is None:
      stored_extra.zero_()
    else:
      stored_extra.copy_(extra_aref)
    debug = w["workspace_debug"].reshape(b, self._debug_stride)
    row_base = nr * nr
    w["position_cache_J"].copy_(w["workspace_J"])
    w["position_cache_qvel"].view(b, nv).copy_(
        w["position_qvel"][:b * nv].view(b, nv))
    w["position_cache_rows"][:b * 7 * nr].view(b, 7 * nr).copy_(
        debug[:, row_base:row_base + 7 * nr])
    w["position_cache_impedance"].view(b, nr).copy_(
        debug[:, row_base + 2 * nr:row_base + 3 * nr])
    low_aref = debug[:, self._aref_low_debug_offset:
                     self._aref_low_debug_offset + nr]
    w["position_current_aref_low"].view(b, nr).copy_(low_aref)
    w["position_cache_aref_low"].view(b, nr).copy_(low_aref)
    w["position_cache_contact_data"].copy_(w["contact_row_data"])
    w["position_cache_contact_frame"].copy_(w["contact_frame"])
    if getattr(getattr(self, "_jacobian_layout", None), "mode", 0) == 0:
      w["position_cache_contact_jacobian"].copy_(w["contact_jacobian"])
    for prefix, name in (("slot", "slot_maps"), ("pair", "pair_maps")):
      cache_packed = w[f"position_cache_{prefix}_packed"]
      cache_reverse = w[f"position_cache_{prefix}_reverse"]
      cache_count = w[f"position_cache_{prefix}_count"]
      cache_overflow = w[f"position_cache_{prefix}_overflow"]
      maps = w.get(name)
      if maps is None:
        _copy_compaction_map_storage(cache_packed, cache_packed[:0])
        _copy_compaction_map_storage(cache_reverse, cache_reverse[:0])
        cache_count.zero_()
        cache_overflow.zero_()
      else:
        _copy_compaction_map_storage(cache_packed, maps.packed_to_logical)
        _copy_compaction_map_storage(cache_reverse, maps.logical_to_packed)
        cache_count.copy_(maps.active_count.reshape(-1))
        cache_overflow.copy_(maps.overflow.reshape(-1))
    self._position_context_valid = True
    self._position_context_eq_jdot_inputs_valid = bool(
        self._position_current_eq_jdot_inputs_valid)
    return {
        "_owner": self,
        "_epoch": capture_epoch,
        "_eq_jdot_inputs_valid": self._position_context_eq_jdot_inputs_valid,
        "position_context": stored,
        "surface_velocity": stored_surface,
        "extra_aref": stored_extra,
    }

  def run_velocity_device(self, context, poses, mass, qfrc_smooth, qpos, qvel,
                          eq_active=None, cvel=None, tendon_J_spatial=None,
                          tendon_length_spatial=None, flex=None,
                          qacc_warmstart=None, awake_tree=None,
                          qacc_smooth=None,
                          component_solver=None, mass_blocks=None,
                          tendon_armature_blocks=None, awake_dof_ids=None,
                          awake_counts=None, cdof=None, cdof_dot=None,
                          exact_impedance=None, exact_dense_solver=None,
                          qacc_smooth_low=None, qfrc_smooth_low=None,
                          world_mask=None):
    """Refresh and solve a captured position context without row assembly.

    Candidate generation, FK, equality/tendon/flex row construction and J
    updates are deliberately absent from this entry point. ``poses`` and the
    legacy row-builder arguments are accepted for caller symmetry but are not
    read. Connect/weld Jdot-v is refreshed from current smooth motion data and
    the retained POS snapshot. When exact impedance is configured, the cached
    row references and regularizers are refreshed from that same retained
    position context and current velocity; caller-provided extra_aref remains
    an additional term captured with the position context. If ``qacc_smooth`` is supplied,
    it is used directly as the unconstrained acceleration and the second mass
    solve is skipped; ``qfrc_smooth`` remains the fallback when omitted.
    """
    if (not isinstance(context, dict) or context.get("_owner") is not self
        or not self._position_context_valid
        or context.get("_epoch") != self._position_context_epoch):
      raise ValueError("position context is stale or belongs to another solver")
    torch, d = self._torch, self.descriptor
    b, nv, nr = int(self.batch_size), int(d.nv), int(d.nr)
    if poses is not None:
      _validate_geom_pose_residuals(torch, self._device, poses, b, int(d.ngeom))
    if not d.dense_path and component_solver is None:
      raise NotImplementedError(
          "cached velocity stages require dense mass or a component solver")
    expected_q = (b, nv)
    _validate_cached_tensor(torch, self._device, "qvel", qvel, expected_q,
                            torch.float32)
    _validate_cached_tensor(torch, self._device, "qfrc_smooth", qfrc_smooth,
                            expected_q, torch.float32)
    if qacc_smooth is not None:
      _validate_cached_tensor(torch, self._device, "qacc_smooth", qacc_smooth,
                              expected_q, torch.float32)
    if qacc_smooth_low is not None:
      if qacc_smooth is None:
        raise ValueError("qacc_smooth_low requires qacc_smooth high values")
      _validate_cached_tensor(torch, self._device, "qacc_smooth_low",
                              qacc_smooth_low, expected_q, torch.float32)
    if qfrc_smooth_low is not None:
      if (not d.dense_path
          and d.solver_type != int(mujoco.mjtSolver.mjSOL_PGS)):
        raise ValueError("qfrc_smooth_low requires a dense or PGS paired route")
      _validate_cached_tensor(torch, self._device, "qfrc_smooth_low",
                              qfrc_smooth_low, expected_q, torch.float32)
    _validate_cached_tensor(torch, self._device, "qpos", qpos, (b, int(d.nq)),
                            torch.float32)
    if poses is not None:
      if not isinstance(poses, dict):
        raise ValueError("poses must be a mapping of captured pose tensors")
      for name, shape in (("body_pos", (b, int(d.nbody), 3)),
                          ("body_quat", (b, int(d.nbody), 4))):
        if name in poses:
          _validate_cached_tensor(torch, self._device, f"poses[{name!r}]",
                                  poses[name], shape, torch.float32)
    if cvel is not None:
      _validate_cached_tensor(torch, self._device, "cvel", cvel,
                              (b, int(d.nbody), 6), torch.float32)
    has_differential_eq = d.neq and np.any(
        np.isin(np.asarray(d.eq_type, dtype=np.int32), (0, 1)))
    if cdof is not None:
      _validate_cached_tensor(torch, self._device, "cdof", cdof,
                              (b, nv, 6), torch.float32)
    if cdof_dot is not None:
      _validate_cached_tensor(torch, self._device, "cdof_dot", cdof_dot,
                              (b, nv, 6), torch.float32)
    if has_differential_eq and (cvel is None or cdof is None or cdof_dot is None):
      raise ValueError("cached connect/weld refresh requires current cvel, cdof, and cdof_dot")
    if has_differential_eq and not (
        self._position_context_eq_jdot_inputs_valid
        and context.get("_eq_jdot_inputs_valid") is True):
      raise ValueError("captured position context lacks equality motion metadata")
    if d.neq and cvel is None:
      eq_types = np.asarray(d.eq_type, dtype=np.int32)
      if np.any(np.isin(eq_types, (0, 1))):
        raise ValueError("cached connect/weld refresh requires current cvel")
    if eq_active is not None:
      expected_eq = (b, int(d.neq)) if d.neq else (b, 1)
      _validate_cached_tensor(torch, self._device, "eq_active", eq_active,
                              expected_eq, torch.int32)
    if qacc_warmstart is not None:
      _validate_cached_tensor(torch, self._device, "qacc_warmstart",
                              qacc_warmstart, expected_q, torch.float32)
    component_primal_route = (component_solver is not None
                              and d.solver_type != int(mujoco.mjtSolver.mjSOL_PGS))
    if component_primal_route and self._component_operator_layout_device is None:
      raise ValueError("cached sparse primal solve requires a configured component layout")
    if component_solver is None and mass is None:
      raise ValueError("cached dense velocity solve requires the generalized mass")
    if component_solver is not None:
      if mass_blocks is None or awake_dof_ids is None or awake_counts is None:
        raise ValueError("cached component solve requires mass blocks and awake DOF lists")
      if (component_solver.batch_size != b or component_solver.nv != nv
          or component_solver.nnz != self._component_operator_nnz):
        raise ValueError("cached component solver does not match the configured layout")
      _validate_cached_tensor(torch, self._device, "mass_blocks", mass_blocks,
                              (b, component_solver.nnz), torch.float32)
      _validate_cached_tensor(torch, self._device, "awake_dof_ids", awake_dof_ids,
                              (b, max(nv, 1)), torch.int32)
      _validate_cached_tensor(torch, self._device, "awake_counts", awake_counts,
                              (b, 3), torch.int32)
      if tendon_armature_blocks is not None:
        _validate_cached_tensor(
            torch, self._device, "tendon_armature_blocks", tendon_armature_blocks,
            (b, component_solver.nnz), torch.float32)
    else:
      _validate_cached_tensor(torch, self._device, "mass", mass, (b, nv, nv),
                              torch.float32)
    if exact_impedance is not None:
      if (getattr(exact_impedance, "batch_size", None) != b
          or getattr(exact_impedance, "row_capacity", None) != nr
          or getattr(exact_impedance, "dof_capacity", None) != nv):
        raise ValueError("exact impedance helper does not match this constraint workspace")
      if component_solver is None:
        if (exact_dense_solver is None
            or getattr(exact_impedance, "mass_storage", None) != "dense"
            or getattr(exact_dense_solver, "batch_size", None) != b
            or getattr(exact_dense_solver, "nv", None) != nv
            or getattr(exact_dense_solver, "nrhs", None) != nr):
          raise ValueError("dense exact impedance solver does not match its prepared workspace")
      elif (getattr(exact_impedance, "mass_storage", None) != "block_sparse"
            or getattr(component_solver, "rhs_capacity", 0) < nr + 1):
        raise ValueError("component exact impedance solver does not match its prepared workspace")

    debug = self._workspace["workspace_debug"].reshape(b, self._debug_stride)
    context_view = context["position_context"]
    surface = context.get("surface_velocity")
    extra = context.get("extra_aref")
    try:
      _validate_cached_tensor(torch, self._device, "context.position_context",
                              context_view, (b, nr, 5), torch.float32)
      _validate_cached_tensor(torch, self._device, "context.surface_velocity",
                              surface, (b, nr), torch.float32)
      _validate_cached_tensor(torch, self._device, "context.extra_aref",
                              extra, (b, nr), torch.float32)
    except ValueError:
      raise ValueError("captured position context was modified or is malformed") from None
    expected_awake = (b, max(self._solver_island_ntree, 1))
    if awake_tree is not None and (
        not isinstance(awake_tree, torch.Tensor)
        or tuple(awake_tree.shape) != expected_awake
        ):
      raise ValueError(f"awake_tree must be contiguous int32 with shape {expected_awake}")
    if awake_tree is not None:
      _validate_cached_tensor(torch, self._device, "awake_tree", awake_tree,
                              expected_awake, torch.int32)
    self._set_world_mask(world_mask)
    # Restore the captured position workspace after any ordinary forward has
    # overwritten the borrowed views. This cache remains valid until another
    # capture or workspace replacement, matching the retained RK4 POS stage.
    w = self._workspace
    jacobian_stride = int(w["workspace_J"].numel()) // max(b, 1)
    self._copy_world_masked(
        w["workspace_J"], w["position_cache_J"], world_mask,
        width=jacobian_stride, source_stride=jacobian_stride,
        destination_stride=jacobian_stride)
    self._copy_world_masked(
        w["position_qvel"], w["position_cache_qvel"], world_mask,
        width=nv, source_stride=nv, destination_stride=nv)
    row_base = nr * nr
    self._copy_world_masked(
        w["workspace_debug"], w["position_cache_rows"], world_mask,
        width=7 * nr, source_stride=7 * nr,
        destination_stride=self._debug_stride,
        destination_offset=row_base)
    self._copy_world_masked(
        w["workspace_debug"], w["position_cache_impedance"], world_mask,
        width=nr, source_stride=nr, destination_stride=self._debug_stride,
        destination_offset=row_base + 2 * nr)
    self._copy_world_masked(
        w["workspace_debug"], w["position_cache_aref_low"], world_mask,
        width=nr, source_stride=nr, destination_stride=self._debug_stride,
        destination_offset=self._aref_low_debug_offset)
    self._copy_world_masked(
        w["position_current_aref_low"], w["workspace_debug"], world_mask,
        width=nr, source_stride=self._debug_stride,
        destination_stride=nr, source_offset=self._aref_low_debug_offset)
    for destination_name, source_name in (
        ("contact_row_data", "position_cache_contact_data"),
        ("contact_frame", "position_cache_contact_frame")):
      destination, source = w[destination_name], w[source_name]
      width = destination.numel() // max(b, 1)
      self._copy_world_masked(
          destination, source, world_mask, width=width,
          source_stride=width, destination_stride=width)
    if self._jacobian_layout.mode == 0:
      destination = w["contact_jacobian"]
      source = w["position_cache_contact_jacobian"]
      width = destination.numel() // max(b, 1)
      self._copy_world_masked(
          destination, source, world_mask, width=width,
          source_stride=width, destination_stride=width)
    for prefix, name in (("slot", "slot_maps"), ("pair", "pair_maps")):
      maps = w.get(name)
      if maps is not None:
        for destination, source_name in (
            (maps.packed_to_logical, f"position_cache_{prefix}_packed"),
            (maps.logical_to_packed, f"position_cache_{prefix}_reverse"),
            (maps.active_count, f"position_cache_{prefix}_count"),
            (maps.overflow, f"position_cache_{prefix}_overflow"),
        ):
          width = destination.numel() // max(b, 1)
          self._copy_world_masked(
              destination, w[source_name], world_mask, width=width,
              source_stride=width, destination_stride=width)
    warm_offset = getattr(
        self, "_qacc_warmstart_offset",
        self._debug_prefix + self._primal_scratch_floats - max(nv, 1))
    warm_tail = debug[:, warm_offset:warm_offset + nv]
    if qacc_warmstart is None:
      self._fill_world_masked(
          w["workspace_debug"].view(torch.int32), world_mask,
          width=nv, stride=self._debug_stride,
          offset=warm_offset, value=0)
    else:
      self._copy_world_masked(
          w["workspace_debug"], qacc_warmstart, world_mask,
          width=nv, source_stride=nv, destination_stride=self._debug_stride,
          destination_offset=warm_offset)
    debug_i32 = debug.view(torch.int32)
    awake_offset = self._debug_prefix + self._solver_island_awake_offset
    awake_view = debug_i32[:, awake_offset:
                           awake_offset + max(self._solver_island_ntree, 1)]
    if awake_tree is None:
      self._fill_world_masked(
          debug_i32, world_mask,
          width=max(self._solver_island_ntree, 1),
          stride=self._debug_stride,
          offset=awake_offset, value=1)
    else:
      self._copy_world_masked(
          debug_i32, awake_tree, world_mask,
          width=max(self._solver_island_ntree, 1),
          source_stride=max(self._solver_island_ntree, 1),
          destination_stride=self._debug_stride,
          destination_offset=awake_offset)
    refresh_extra = extra
    eq_types = np.asarray(d.eq_type, dtype=np.int32)
    has_differential_eq = d.neq and np.any(np.isin(eq_types, (0, 1)))
    has_site_differential_eq = bool(
        has_differential_eq
        and np.any(np.isin(eq_types, (0, 1))
                   & (np.asarray(d.eq_objtype, dtype=np.int32) == 6)))
    old_extra = w["position_refresh_old_extra_aref"][:b * nr].view(b, nr)
    self._copy_world_masked(
        old_extra, w["position_extra_aref"][:b * nr].view(b, nr),
        world_mask, width=nr, source_stride=nr, destination_stride=nr)
    if has_differential_eq:
      old_jdot = w["position_context_zero"][:b * nr].view(b, nr)
      self._refresh_equality_jdot(
          w["position_cache_qvel"].view(b, nv),
          w["position_cache_cvel"].view(b, d.nbody, 6),
          w["position_cache_cdof"].view(b, nv, 6),
          w["position_cache_cdof_dot"].view(b, nv, 6), old_jdot,
          eq_active=w["position_cache_eq_active"])
      self._copy_world_masked(
          old_extra, old_jdot, world_mask, width=nr,
          source_stride=nr, destination_stride=nr, add=True)
    if d.neq:
      refresh_extra = w["position_refresh_extra_aref"][:b * nr].view(b, nr)
      self._copy_world_masked(
          refresh_extra, extra, world_mask, width=nr,
          source_stride=nr, destination_stride=nr)
      if has_differential_eq:
        equality_aref = w["position_context_zero"][:b * nr].view(b, nr)
        self._refresh_equality_jdot(
            qvel, cvel, cdof, cdof_dot, equality_aref)
        self._copy_world_masked(
            refresh_extra, equality_aref, world_mask, width=nr,
            source_stride=nr, destination_stride=nr, add=True)
    self.refresh_position_references(
        context_view, qvel, surface, refresh_extra, world_mask=world_mask)
    self._refresh_cached_aref_low(
        qvel, surface, refresh_extra, old_extra, world_mask=world_mask)

    solver_dims = self._constants["solver_dims"]
    solve_fn = self._solve_kernel if d.dense_path else self._solve_block_kernel
    w = self._workspace
    self._fill_world_masked(
        w["out_status"], world_mask, width=1, stride=1, value=0)
    if self._has_sdf_contact_pairs:
      self._merge_sdf_narrowphase_status_kernel(
          w["contact_pair_count"], w["out_status"],
          self._pair_contact_world_mask,
          self._constants["pair_contact_offsets_dims"],
          threads=(b,), group_size=(1,))
    slot_maps = w.get("slot_maps")
    packed_slots = (self._empty_compaction_map if slot_maps is None
                    else slot_maps.packed_to_logical.reshape(-1))

    solver_acceleration_input = (qfrc_smooth if qacc_smooth is None
                                 else qacc_smooth)
    paired_dense_input = (d.dense_path
        and (qacc_smooth is not None or qfrc_smooth_low is not None))
    if (d.solver_type == int(mujoco.mjtSolver.mjSOL_PGS)
        or paired_dense_input):
      pair = w.get("component_smooth_acceleration_pair")
      if pair is None:
        raise RuntimeError("paired smooth-acceleration workspace was not prepared")
      if qacc_smooth is not None:
        self._copy_world_masked(
            pair[:b * nv], qacc_smooth, world_mask, width=nv,
            source_stride=nv, destination_stride=nv)
        if qacc_smooth_low is None:
          self._fill_world_masked(
              pair[b * nv:2 * b * nv], world_mask, width=nv,
              stride=nv, value=0)
        else:
          self._copy_world_masked(
              pair[b * nv:2 * b * nv], qacc_smooth_low, world_mask,
              width=nv, source_stride=nv, destination_stride=nv)
      else:
        if paired_dense_input:
          self._copy_world_masked(
              pair[:b * nv], qfrc_smooth, world_mask, width=nv,
              source_stride=nv, destination_stride=nv)
        if qfrc_smooth_low is None:
          self._fill_world_masked(
              pair[b * nv:2 * b * nv], world_mask, width=nv,
              stride=nv, value=0)
        else:
          self._copy_world_masked(
              pair[b * nv:2 * b * nv], qfrc_smooth_low, world_mask,
              width=nv, source_stride=nv, destination_stride=nv)
      if qacc_smooth is not None or paired_dense_input:
        solver_acceleration_input = pair

    def dispatch_solver(mass_input, smooth_input=None):
      solve_fn(
          mass_input.reshape(-1),
          (solver_acceleration_input if smooth_input is None
           else smooth_input).reshape(-1), qpos.reshape(-1), qvel.reshape(-1),
          (self._eq_active_default if eq_active is None else eq_active).reshape(-1),
          self._constants["joint_qposadr"], self._constants["qpos0"],
          self._constants["joint_dofadr"], self._constants["joint_limited"],
          self._constants["joint_limit_params"], self._constants["joint_sol_params"],
          self._constants["dof_frictionloss"], self._constants["dof_invweight0"],
          self._constants["dof_sol_params"], self._constants["eq_obj"],
          self._constants["eq_data"], self._constants["eq_sol_params"],
          w["contact_jacobian"], w["contact_row_data"],
          self._constants["contact_friction"], self._constants["contact_condim"],
          solver_dims, self._constants["solver_params"], w["out_force"], w["out_acc"],
          w["out_status"], w["out_diagnostics"], w["out_contact_force"],
          packed_slots, w["workspace_J"], w["workspace_debug"],
          threads=(b,), group_size=(1,))

    exact_data = None
    if component_solver is None:
      if exact_impedance is not None:
        exact_data = self.recompute_exact_impedance_device(
            exact_impedance, mass=mass,
            exact_dense_solver=exact_dense_solver,
            component_solver=None, mass_blocks=mass_blocks,
            tendon_armature_blocks=tendon_armature_blocks)
      with _cached_position_stage(
          solver_dims, d.refsafe,
          bool(getattr(self, "_iteration_trace_enabled", False)),
          bool(getattr(self, "_primal_detail_trace_enabled", False)),
          self._primal_detail_extra_flags()), (
          _provided_smooth_acceleration_stage(solver_dims)
          if qacc_smooth is not None else nullcontext()), (
              _paired_smooth_input_stage(solver_dims)
              if paired_dense_input else nullcontext()), (
              _preassembled_rows_stage(solver_dims)
              if exact_data is not None else nullcontext()):
        dispatch_solver(mass)
    elif component_primal_route:
      self._stage_component_operator(tendon_armature_blocks)
      with _component_primal_stage(
          solver_dims, d.refsafe, -1, self._line_search_iterations,
          qacc_smooth is not None,
          bool(getattr(self, "_iteration_trace_enabled", False)),
          bool(getattr(self, "_primal_detail_trace_enabled", False)),
          self._primal_detail_extra_flags()):
        dispatch_solver(mass_blocks)
      if exact_impedance is not None:
        exact_data = self.recompute_exact_impedance_device(
            exact_impedance, mass=None,
            exact_dense_solver=exact_dense_solver,
            component_solver=component_solver, mass_blocks=mass_blocks,
            dof_ids=awake_dof_ids, counts=awake_counts,
            tendon_armature_blocks=tendon_armature_blocks)
      with _component_primal_stage(
          solver_dims, d.refsafe, -3, self._line_search_iterations,
          qacc_smooth is not None,
          bool(getattr(self, "_iteration_trace_enabled", False)),
          bool(getattr(self, "_primal_detail_trace_enabled", False)),
          self._primal_detail_extra_flags()), (
              _preassembled_rows_stage(solver_dims)
              if exact_data is not None else nullcontext()):
        dispatch_solver(mass_blocks)
    else:
      rhs_capacity = nr + 1
      component_rhs = w["component_mass_rhs"][:b * rhs_capacity * nv]
      def prepare_component_rhs():
        if exact_impedance is not None:
          # The exact mass-row solve borrows component_solver's prepared output
          # backing. It must finish before run_device overwrites that backing.
          exact = self.recompute_exact_impedance_device(
              exact_impedance, mass=None,
              exact_dense_solver=exact_dense_solver,
              component_solver=component_solver, mass_blocks=mass_blocks,
              dof_ids=awake_dof_ids, counts=awake_counts,
              tendon_armature_blocks=tendon_armature_blocks)
        else:
          exact = None
        with (_provided_smooth_acceleration_stage(solver_dims)
              if qacc_smooth is not None else nullcontext()):
          self._pack_component_mass_rhs(
              w["workspace_J"], qfrc_smooth.reshape(-1),
              w["workspace_debug"], component_rhs, solver_dims,
              threads=(b,), group_size=(1,))
        return exact

      exact_data, (component_result, component_status) = (
          _run_component_solve_after_prepare(
              prepare_component_rhs,
              lambda: component_solver.run_device(
                  mass_blocks,
                  component_rhs.reshape(b, rhs_capacity, nv),
                  dof_ids=awake_dof_ids, counts=awake_counts,
                  tendon_armature_blocks=tendon_armature_blocks,
                  rhs_low=(None if qacc_smooth is not None
                           else (w["component_smooth_acceleration_pair"]
                                 [b * nv:2 * b * nv].reshape(b, nv))))))
      with _cached_position_stage(
          solver_dims, d.refsafe,
          bool(getattr(self, "_iteration_trace_enabled", False)),
          bool(getattr(self, "_primal_detail_trace_enabled", False)),
          self._primal_detail_extra_flags()), _solver_dimension_stage(
          solver_dims, -2, self._line_search_iterations), (
              _provided_smooth_acceleration_stage(solver_dims)
              if qacc_smooth is not None else nullcontext()), (
                  _preassembled_rows_stage(solver_dims)
                  if exact_data is not None else nullcontext()):
        dispatch_solver(component_solver.paired_output)
      component_solver.merge_world_status(component_status, w["out_status"])

    if exact_data is not None:
      _merge_exact_world_status(torch, w["out_status"], exact_data)

    solver_diagnostics, solver_history = self._diagnostic_views()
    result = {
        "qacc": w["out_acc"][:b * nv].reshape(b, nv),
        "qacc_low": w["out_acc"][b * nv:2 * b * nv].reshape(b, nv),
        "qfrc_constraint": w["out_force"][:b * nv].reshape(b, nv),
        "status": w["out_status"],
        "solver_diagnostics": solver_diagnostics,
        "solver_history": solver_history,
        "position_context": context,
        "R": debug[:, nr * nr:nr * nr + nr],
        "ar": debug[:, nr * nr + nr:nr * nr + 2 * nr],
        "ar_low": debug[:, self._aref_low_debug_offset:
                        self._aref_low_debug_offset + nr],
        "rhs": debug[:, nr * nr + 2 * nr:nr * nr + 3 * nr],
        "lambda": debug[:, nr * nr + 3 * nr:nr * nr + 4 * nr],
        "lo": debug[:, nr * nr + 4 * nr:nr * nr + 5 * nr],
        "hi": debug[:, nr * nr + 5 * nr:nr * nr + 6 * nr],
        "W": debug[:, :nr * nr].reshape(b, nr, nr),
        "W_regularized": (
            debug[:, :nr * nr].reshape(b, nr, nr)
            + torch.diag_embed(debug[:, nr * nr:nr * nr + nr])),
    }
    jacobian_layout = getattr(self, "_jacobian_layout", None)
    if jacobian_layout is None:
      result["J"] = w["workspace_J"][:b * nr * nv].reshape(b, nr, nv)
    elif jacobian_layout.mode == 0:
      result["J"] = self.materialize_jacobian()
    else:
      result["J_packed"] = w["workspace_J"]
      result["jacobian_layout"] = jacobian_layout
    if nr:
      debug_i32 = debug.view(torch.int32)
      island_base = self._debug_prefix + self._solver_island_core_scratch
      result["dof_island"] = debug_i32[:, island_base:island_base + nv]
      result["row_island"] = debug_i32[
          :, island_base + nv:island_base + nv + nr]
      tree_base = island_base + nv + nr
      body_base = tree_base + 2 * max(self._solver_island_ntree, 1)
      result["midpoint_blocked_body"] = debug_i32[
          :, body_base:body_base + self._solver_island_nbody]
      result["midpoint_blocked_tree"] = debug_i32[
          :, body_base + max(self._solver_island_nbody, 1):
          body_base + max(self._solver_island_nbody, 1)
          + self._solver_island_ntree]
    nc = int(d.ncontacts_max)
    if nc:
      contact_rows = w["contact_row_data"][:b * nc * 36].reshape(b, nc, 6, 6)
      contact_frames = w["contact_frame"][:b * nc * 12].reshape(b, nc, 12)
      contact_forces = w["out_contact_force"][:b * nc * 11].reshape(b, nc, 11)
      contact_jacobians = (
          w["contact_jacobian"][:b * nc * 6 * nv].reshape(b, nc, 6, nv)
          if self._jacobian_layout.mode == 0 else None)
      friction_tensor = self._constants["contact_friction"][:nc * 5].reshape(nc, 5)
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
          "contact_mask": contact_rows[:, :, 0, 0],
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
      })
      if contact_jacobians is not None:
        result["contact_jacobian"] = contact_jacobians
    if int(d.nr_joint) > 0:
      result["joint_force"] = w["out_joint_force"][:b * d.nr_joint].reshape(
          b, d.nr_joint)
    if exact_data is not None:
      result["exact_impedance"] = exact_data
    return result

  def _stage_component_operator(self, tendon_armature_blocks):
    """Copy the per-step armature into the prebound sparse-operator tail."""
    if self._component_operator_layout_device is None:
      raise ValueError("sparse primal solve requires a configured component layout")
    torch = self._torch
    b, nnz = int(self.batch_size), self._component_operator_nnz
    debug = self._workspace["workspace_debug"].reshape(b, self._debug_stride)
    if nnz:
      armature = debug[:, self._component_operator_armature_offset:
                       self._component_operator_armature_offset + nnz]
      if tendon_armature_blocks is None:
        armature.zero_()
      else:
        _validate_cached_tensor(
            torch, self._device, "tendon_armature_blocks",
            tendon_armature_blocks, (b, nnz), torch.float32)
        armature.copy_(tendon_armature_blocks)
    debug_i32 = debug.view(torch.int32)
    start = self._component_operator_layout_offset
    end = start + int(self._component_operator_layout_device.numel())
    debug_i32[:, start:end].copy_(self._component_operator_layout_device[None, :])

  def run_device(self, poses, mass, qfrc_smooth, qpos, qvel, eq_active=None, cvel=None,
                 tendon_J_spatial=None, tendon_length_spatial=None, flex=None,
                 qacc_warmstart=None, awake_tree=None, component_solver=None,
                 mass_blocks=None, tendon_armature_blocks=None,
                 awake_dof_ids=None, awake_counts=None, cdof=None, cdof_dot=None,
                 assemble_only=False, candidate_token=None,
                 exact_impedance=None, exact_dense_solver=None,
                 qfrc_smooth_low=None, world_mask=None):
    """Solve coupled contacts, limits, dry friction, and equalities on MPS.

    `poses` is the dict of FK outputs from `MetalKinematics`.
    `mass` is batched generalized mass [batch, nv, nv] (including tendon armature).
    The PGS sparse path may pass ``mass=None`` together with a compiled
    component solver and block mass inputs; it assembles constraints first,
    then applies component-wise ``M^-1`` to live rows and smooth force.
    `qfrc_smooth` is unconstrained forces [batch, nv].
    `qpos` is [batch, nq], `qvel` is [batch, nv].
    `cvel` is batched body spatial velocity [batch, nbody, 6] from smooth
    dynamics (angular, linear COM). Differential connect/weld rows also need
    smooth `cdof` and `cdof_dot` to evaluate the pinned dense `mj_jacDot` term;
    scalar joint equalities do not consume these motion inputs.
    `tendon_J_spatial`/`tendon_length_spatial` are borrowed [batch, nt(, nv)]
    spatial tendon views from the simulation's once-per-assembly kinematics
    (None when the model has no spatial tendons; the stage substitutes zeros).
    """
    w, torch, d = self._workspace, self._torch, self.descriptor
    b, nv, nc, nr = self.batch_size, d.nv, d.ncontacts_max, d.nr
    component_route = nv > 0 and component_solver is not None
    component_primal_route = (component_route
                              and d.solver_type != int(mujoco.mjtSolver.mjSOL_PGS))
    supplied_component_args = (mass_blocks is not None
                               or tendon_armature_blocks is not None
                               or awake_dof_ids is not None
                               or awake_counts is not None)
    if component_primal_route and self._component_operator_layout_device is None:
      raise ValueError("sparse primal solve requires a configured component layout")
    if assemble_only and (component_route or supplied_component_args):
      raise ValueError("assembly-only mode does not accept component mass inputs")
    if mass is None and not component_route and not assemble_only:
      raise ValueError("dense solver path requires a mass matrix")
    if supplied_component_args and component_solver is None:
      raise ValueError("component mass inputs require component_solver")
    # This must precede token consumption, validity flags, status clearing,
    # debug staging, or any output workspace writes below.
    _validate_geom_pose_residuals(torch, self._device, poses, b, int(d.ngeom))
    if qfrc_smooth_low is not None:
      if (not d.dense_path
          and d.solver_type != int(mujoco.mjtSolver.mjSOL_PGS)):
        raise ValueError("qfrc_smooth_low requires a dense or PGS paired route")
      _validate_cached_tensor(torch, self._device, "qfrc_smooth_low",
                              qfrc_smooth_low, (b, nv), torch.float32)
    if candidate_token is not None:
      if candidate_token is not self._candidate_generation_token:
        raise ValueError("candidate-generation token is stale or foreign")
      # A generation describes one exact current-position narrowphase result.
      # Consume it before mutation so a failed downstream assembly cannot
      # accidentally replay partially overwritten candidate buffers.
      self._candidate_generation_token = None
    self._position_current_valid = False

    if exact_impedance is not None:
      if nr <= 0 or nv <= 0:
        raise ValueError("exact impedance requires nonempty constraint and DOF rows")
      if (getattr(exact_impedance, "batch_size", None) != b
          or getattr(exact_impedance, "row_capacity", None) != nr
          or getattr(exact_impedance, "dof_capacity", None) != nv):
        raise ValueError("exact impedance helper does not match this constraint workspace")
      if component_solver is None and (mass is None or exact_dense_solver is None):
        raise ValueError("dense exact impedance requires its prepared multi-RHS mass solver")
      if component_solver is not None and mass_blocks is None:
        raise ValueError("component exact impedance requires packed mass blocks")
    if component_route:
      if (mass_blocks is None or awake_dof_ids is None or awake_counts is None
          or not hasattr(component_solver, "run_device")
          or component_solver.batch_size != b
          or component_solver.nv != nv):
        raise ValueError(
            "component solver requires matching mass blocks and awake DOF lists")
      if (component_solver.nnz != self._component_operator_nnz
          or (not component_primal_route
              and component_solver.rhs_capacity != nr + 1)):
        raise ValueError("component solver does not match its configured workspace")
      if (not isinstance(mass_blocks, torch.Tensor)
          or mass_blocks.dtype != torch.float32
          or mass_blocks.device.type != self._device.type
          or (self._device.index is not None
              and mass_blocks.device.index != self._device.index)
          or not mass_blocks.is_contiguous()
          or tuple(mass_blocks.shape) != (b, component_solver.nnz)):
        raise ValueError("mass_blocks must be contiguous float32 MPS [batch,nnz]")
      if (not isinstance(awake_dof_ids, torch.Tensor)
          or tuple(awake_dof_ids.shape) != (b, max(nv, 1))
          or awake_dof_ids.dtype != torch.int32
          or awake_dof_ids.device.type != self._device.type
          or (self._device.index is not None
              and awake_dof_ids.device.index != self._device.index)
          or not awake_dof_ids.is_contiguous()):
        raise ValueError("awake_dof_ids must be contiguous MPS int32 [batch,max(nv,1)]")
      if (not isinstance(awake_counts, torch.Tensor)
          or tuple(awake_counts.shape) != (b, 3)
          or awake_counts.dtype != torch.int32
          or awake_counts.device.type != self._device.type
          or (self._device.index is not None
              and awake_counts.device.index != self._device.index)
          or not awake_counts.is_contiguous()):
        raise ValueError("awake_counts must be contiguous MPS int32 [batch,3]")
      if tendon_armature_blocks is not None and (
          not isinstance(tendon_armature_blocks, torch.Tensor)
          or tuple(tendon_armature_blocks.shape) != (b, component_solver.nnz)
          or tendon_armature_blocks.dtype != torch.float32
          or tendon_armature_blocks.device.type != self._device.type
          or (self._device.index is not None
              and tendon_armature_blocks.device.index != self._device.index)
          or not tendon_armature_blocks.is_contiguous()):
        raise ValueError(
            "tendon_armature_blocks must be contiguous float32 MPS [batch,nnz]")
      if w["component_mass_rhs"] is None and not component_primal_route:
        raise ValueError("component mass RHS workspace is unavailable for this solver")

    if awake_tree is not None:
      expected_awake = (b, max(self._solver_island_ntree, 1))
      if (not isinstance(awake_tree, torch.Tensor)
          or awake_tree.dtype != torch.int32
          or tuple(awake_tree.shape) != expected_awake
          or awake_tree.device.type != self._device.type
          or (self._device.index is not None
              and awake_tree.device.index != self._device.index)
          or not awake_tree.is_contiguous()):
        raise ValueError(
            f"awake_tree must be contiguous int32 MPS with shape {expected_awake}")

    # These exact position snapshots feed equality Jdot-v replay. Validate
    # them before warmstart/debug scratch is written, not after the solve.
    if not isinstance(poses, dict):
      raise ValueError("poses must be a mapping of FK tensors")
    for key, shape in (("body_pos", (b, d.nbody, 3)),
                       ("body_quat", (b, d.nbody, 4))):
      _validate_cached_tensor(torch, self._device, f"poses[{key!r}]",
                              poses.get(key), shape, torch.float32)
    eq_types = np.asarray(d.eq_type, dtype=np.int32)
    has_differential_eq = d.neq and np.any(np.isin(eq_types, (0, 1)))
    has_site_differential_eq = bool(
        has_differential_eq
        and np.any(np.isin(eq_types, (0, 1))
                   & (np.asarray(d.eq_objtype, dtype=np.int32) == 6)))
    if has_differential_eq:
      _validate_cached_tensor(torch, self._device, "poses['root_com']",
                              poses.get("root_com"), (b, d.nbody, 3),
                              torch.float32)
      if cvel is not None:
        _validate_cached_tensor(torch, self._device, "cvel", cvel,
                                (b, d.nbody, 6), torch.float32)
      if cdof is not None:
        _validate_cached_tensor(torch, self._device, "cdof", cdof,
                                (b, nv, 6), torch.float32)
      if cdof_dot is not None:
        _validate_cached_tensor(torch, self._device, "cdof_dot", cdof_dot,
                                (b, nv, 6), torch.float32)
      if has_differential_eq and (cvel is None or cdof is None or cdof_dot is None):
        raise ValueError(
            "connect/weld equality assembly requires cvel, cdof, and cdof_dot")
      if has_site_differential_eq:
        _validate_cached_tensor(torch, self._device, "poses['site_pos']",
                                poses.get("site_pos"), (b, d.nsite, 3),
                                torch.float32)
        _validate_cached_tensor(torch, self._device, "poses['site_quat']",
                                poses.get("site_quat"), (b, d.nsite, 4),
                                torch.float32)

    # MuJoCo's WARMSTART state is qacc_warmstart for every solver family.
    # PGS maps it to row forces before its cost check; CG/Newton compare the
    # acceleration-space cost with qacc_smooth. The final DOF-sized tail of
    # persistent scratch carries it without adding a Metal buffer binding.
    if qacc_warmstart is not None:
      if (not isinstance(qacc_warmstart, torch.Tensor)
          or qacc_warmstart.dtype != torch.float32
          or tuple(qacc_warmstart.shape) != (b, nv)
          or qacc_warmstart.device.type != self._device.type
          or (self._device.index is not None
              and qacc_warmstart.device.index != self._device.index)
          or not qacc_warmstart.is_contiguous()):
        raise ValueError(
            f"qacc_warmstart must be contiguous float32 MPS with shape {(b, nv)}"
        )
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
      if eq_active.device.type != self._device.type:
        raise ValueError(f"eq_active must be on {self._device.type} device")
      if not eq_active.is_contiguous():
        raise ValueError("eq_active must be contiguous")
      eq_active_tensor = eq_active

    self._set_world_mask(world_mask)
    # Snapshot current assembly inputs only after all motion and equality
    # inputs pass host-side validation. These writes are isolated from an
    # already retained position_cache_* epoch.
    self._workspace["position_current_qvel"].view(b, nv).copy_(qvel)
    for source, current, shape in (
        (cvel, "position_current_cvel", (b, d.nbody, 6)),
        (cdof, "position_current_cdof", (b, d.nv, 6)),
        (cdof_dot, "position_current_cdof_dot", (b, d.nv, 6)),
        (poses.get("body_pos"), "position_current_body_pos", (b, d.nbody, 3)),
        (poses.get("body_quat"), "position_current_body_quat", (b, d.nbody, 4)),
        (poses.get("root_com"), "position_current_root_com", (b, d.nbody, 3)),
        (poses.get("site_pos"), "position_current_site_pos", (b, d.nsite, 3)),
        (poses.get("site_quat"), "position_current_site_quat", (b, d.nsite, 4)),
    ):
      destination = self._workspace[current]
      count = int(np.prod(shape))
      if source is None or count == 0:
        destination.zero_()
      else:
        destination[:count].copy_(source.reshape(-1))
    self._workspace["position_current_eq_active"].copy_(
        eq_active_tensor.reshape(-1))
    self._position_current_eq_jdot_inputs_valid = bool(
        has_differential_eq and cvel is not None and cdof is not None
        and cdof_dot is not None)

    prefix = nr * nr + 7 * nr
    scratch = self._workspace["workspace_debug"].reshape(b, self._debug_stride)
    warm_offset = getattr(
        self, "_qacc_warmstart_offset", prefix
        + self._primal_scratch_floats - max(nv, 1))
    tail = scratch[:, warm_offset:warm_offset + nv]
    if qacc_warmstart is None:
      tail.zero_()
    else:
      tail.copy_(qacc_warmstart)
    self._workspace["position_assembly_context"].zero_()
    self._workspace["position_surface_velocity"].zero_()
    # All fresh row producers start with an exact zero correction unless they
    # explicitly publish an aref low word. The equality kernel also clears
    # this tail, but it is not dispatched for contact-only/flex-only models;
    # do the reset here so reusing a solver cannot inherit cached row words.
    _clear_current_aref_low(
        self._workspace["workspace_debug"], b, nr, self._debug_stride,
        self._aref_low_debug_offset)
    # Activity in this slice marks rows preassembled by the current tendon,
    # equality, or flex producers. It also feeds the solver's tendon-row
    # ownership test. Clear the previous assembly's bitmap before any
    # current producer runs; otherwise a disappeared contact can be mistaken
    # for a still-owned tendon row and replay its retained multiplier.
    _clear_current_row_ownership(
        self._workspace["workspace_debug"], b, nr, self._debug_stride)
    awake_offset = prefix + self._solver_island_awake_offset
    debug_i32 = scratch.view(torch.int32)
    awake_view = debug_i32[:, awake_offset:
                           awake_offset + max(self._solver_island_ntree, 1)]
    if awake_tree is None:
      awake_view.fill_(1)
    else:
      awake_view.copy_(awake_tree)

    # 1. Contact normal kernel (if candidate contact pairs exist)
    w["out_status"].zero_()
    if candidate_token is None:
      self.generate_candidates(
          poses, qvel, eq_active_tensor if d.neq > 0 else None,
          flex=flex, cvel=cvel, cdof=cdof, world_mask=world_mask)
    # Candidate generation owns the per-pair narrowphase counts, including
    # the fail-closed negative sentinel for an SDF output-capacity overflow.
    # Merge it after the status reset above and before any solver route can
    # consume the public status buffer. This also covers a current candidate
    # token produced by an earlier position stage.
    if self._has_sdf_contact_pairs:
      self._merge_sdf_narrowphase_status_kernel(
          w["contact_pair_count"], w["out_status"],
          self._pair_contact_world_mask,
          self._constants["pair_contact_offsets_dims"],
          threads=(b,), group_size=(1,))
    self._merge_flex_candidate_status()
    if d.n_flex_contact_rows and flex is None:
      raise ValueError(
          "the lowered model has flex contact rows but no native flex runtime")

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
          site_pos_tensor.reshape(-1), self._constants["eq_rownum"],
          w["workspace_J"], w["workspace_debug"],
          site_quat_tensor.reshape(-1),
          threads=(b,), group_size=(1,),
      )
      if has_differential_eq:
        # Equality assembly writes the position and -B*Jv terms. Apply the
        # exact source-derived dense mj_jacDot correction separately, using
        # the current prevalidated FK/smooth motion snapshot.
        jdot_aref = w["position_context_zero"][:b * nr].view(b, nr)
        self._refresh_equality_jdot(
            qvel, cvel, cdof, cdof_dot, jdot_aref,
            assembly_current=True, eq_active=eq_active_tensor)
        debug = w["workspace_debug"].reshape(b, self._debug_stride)
        ar = debug[:, nr * nr + nr:nr * nr + nr + d.n_eq_rows]
        ar.add_(jdot_aref[:, :d.n_eq_rows])
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

    # 1c. Compiled flex equality rows.  The descriptor owns stable row IDs
    # independent of efc offsets; flex producers provide the source row block
    # in their own canonical order.
    flex_eq_types = (_EQ_FLEX, _EQ_FLEXVERT, _EQ_FLEXSTRAIN)
    if (d.neq > 0 and flex is not None
        and bool(np.any(np.isin(np.asarray(d.eq_type), flex_eq_types)))):
      nr_val = int(d.nr)
      row_context = w["position_assembly_context"].view(b, nr_val, 5)
      for eid in range(d.neq):
        et = int(d.eq_type[eid])
        if et not in flex_eq_types:
          continue
        if et == _EQ_FLEX:
          # This legacy producer evaluates all flex edges together.  Its raw
          # coefficient context is indexed by compiled global edge ID.
          f_pos, f_aref, f_R, f_J = flex.run_equalities(
              poses, cvel=cvel, eq_active=eq_active_tensor,
              equality_id=eid)
          raw = flex.equality_position_context
          ids = np.asarray(d.eq_row_ids[eid], dtype=np.int32)
          rows = {
              "row_ids": ids,
              "J": f_J[:, ids, :], "R": f_R[:, ids],
              "aref": f_aref[:, ids], "pos": raw["pos"][:, ids],
              "impedance": raw["impedance"][:, ids],
              "b0": raw["b0"][:, ids], "k0": raw["k0"][:, ids],
          }
        elif et == _EQ_FLEXVERT:
          rows = flex.run_flexvert_equality_rows(eid, qvel)
        else:
          rows = flex.run_flexstrain_equality_rows(eid, qvel)

        row_start = int(d.eq_rowadr[eid])
        row_span = min(int(d.eq_rownum[eid]), len(rows["row_ids"]))
        dbg_w = w["workspace_debug"].view(b, -1)
        is_active = (eq_active_tensor[:, eid] != 0).float()
        impedance = rows["impedance"]
        for k in range(row_span):
          r = row_start + k
          source_row = k
          if r >= nr_val:
            break
          self._write_jacobian_rows(
              r, rows["J"][:, source_row:source_row + 1, :nv],
              active=is_active[:, None])
          dbg_w[:, nr_val * nr_val + r] = torch.where(
              is_active > 0, rows["R"][:, source_row], 0.0)
          dbg_w[:, nr_val * nr_val + nr_val + r] = (
              rows["aref"][:, source_row] * is_active)
          dbg_w[:, nr_val * nr_val + 2 * nr_val + r] = (
              impedance[:, source_row] * is_active)
          dbg_w[:, nr_val * nr_val + 4 * nr_val + r] = -float("inf")
          dbg_w[:, nr_val * nr_val + 5 * nr_val + r] = float("inf")
          dbg_w[:, nr_val * nr_val + 6 * nr_val + r] = is_active
          row_context[:, r, 0] = rows["k0"][:, source_row] / impedance[:, source_row]
          row_context[:, r, 1] = rows["b0"][:, source_row] / impedance[:, source_row]
          row_context[:, r, 2] = impedance[:, source_row]
          row_context[:, r, 3] = rows["pos"][:, source_row]
          row_context[:, r, 4].zero_()

    # 1d. Fixed canonical flex-contact rows. The flex-owned producer writes
    # every candidate's complete descriptor span, including inactive slots;
    # no per-world host readback or legacy vertex/geom loop is used here.
    if d.n_flex_contact_rows:
      start = int(d.flex_contact_base)
      stop = start + int(d.n_flex_contact_rows)
      if stop > nr:
        raise RuntimeError("flex contact row block exceeds coupled workspace")
      bundle = self._flex_contact_current
      if bundle is None:
        if flex is not None and d.flex_contact_descriptor is not None:
          raise RuntimeError("flex contact candidate rows were not generated")
        # No runtime flex object means these statically reserved candidates are
        # inactive. Keep every owned row and cached source coefficient zero.
        self._clear_jacobian_rows(start, stop)
        dbg_w = w["workspace_debug"].view(b, -1)
        for offset in (nr * nr, nr * nr + nr, nr * nr + 2 * nr,
                       nr * nr + 4 * nr,
                       nr * nr + 5 * nr, nr * nr + 6 * nr):
          dbg_w[:, offset + start:offset + stop].zero_()
        w["position_assembly_context"].view(b, nr, 5)[:, start:stop, :].zero_()
        w["position_surface_velocity"].view(b, nr)[:, start:stop].zero_()
      else:
        rows = bundle["rows"]
        count = int(d.n_flex_contact_rows)
        row_shape = (b, count)
        if self._jacobian_layout.mode == 0:
          _validate_cached_tensor(
              torch, self._device, "flex rows.workspace_J",
              rows.get("workspace_J"), (b, count, nv), torch.float32)
        else:
          packed_rows = rows.get("jacobian_packed")
          if (packed_rows is not w["workspace_J"]
              or int(rows.get("canonical_row_offset", -1)) != start):
            raise ValueError(
                "sparse flex rows must borrow the canonical CSR buffer and "
                "their compiled global row offset")
        for key in ("R", "aref", "lo", "hi", "active", "surface_velocity"):
          dtype = torch.int32 if key == "active" else torch.float32
          _validate_cached_tensor(
              torch, self._device, f"flex rows.{key}", rows.get(key),
              row_shape, dtype)
        _validate_cached_tensor(
            torch, self._device, "flex rows.position_context",
            rows.get("position_context"), (b, count, 5), torch.float32)
        if nv and self._jacobian_layout.mode == 0:
          self._write_jacobian_rows(start, rows["workspace_J"])
        debug = w["workspace_debug"].view(b, -1)
        debug[:, nr * nr + start:nr * nr + stop].copy_(rows["R"])
        debug[:, nr * nr + nr + start:nr * nr + nr + stop].copy_(rows["aref"])
        debug[:, nr * nr + 4 * nr + start:nr * nr + 4 * nr + stop].copy_(rows["lo"])
        debug[:, nr * nr + 5 * nr + start:nr * nr + 5 * nr + stop].copy_(rows["hi"])
        debug[:, nr * nr + 6 * nr + start:nr * nr + 6 * nr + stop].copy_(
            (rows["active"] != 0).to(dtype=torch.float32))
        # Stage -1 uses the RHS slice as a source-impedance channel. Keep the
        # raw flex material value there until the exact M^-1 contraction.
        debug[:, nr * nr + 2 * nr + start:
              nr * nr + 2 * nr + stop].copy_(
                  rows["position_context"][:, :, 2])
        w["position_assembly_context"].view(b, nr, 5)[:, start:stop, :].copy_(
            rows["position_context"])
        w["position_surface_velocity"].view(b, nr)[:, start:stop].copy_(
            rows["surface_velocity"])

    # 2. Coupled constraint solver kernel
    solve_fn = self._solve_kernel if self.descriptor.dense_path else self._solve_block_kernel
    slot_maps = w.get("slot_maps")
    if slot_maps is None:
      packed_slots = self._empty_compaction_map
    else:
      packed_slots = slot_maps.packed_to_logical.reshape(-1)
    paired_dense_input = d.dense_path and qfrc_smooth_low is not None
    smooth_solver_input = qfrc_smooth
    if paired_dense_input:
      pair = w.get("component_smooth_acceleration_pair")
      if pair is None:
        raise RuntimeError("dense paired smooth-force workspace was not prepared")
      pair[:b * nv].copy_(qfrc_smooth.reshape(-1))
      pair[b * nv:2 * b * nv].copy_(qfrc_smooth_low.reshape(-1))
      smooth_solver_input = pair

    def dispatch_solver(mass_input, smooth_input=None):
      solve_fn(
          mass_input.reshape(-1),
          (smooth_solver_input if smooth_input is None
           else smooth_input).reshape(-1), qpos.reshape(-1), qvel.reshape(-1),
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
          w["out_contact_force"], packed_slots, w["workspace_J"],
          w["workspace_debug"],
          threads=(b,), group_size=(1,),
      )

    def recompute_exact_impedance():
      if exact_impedance is None:
        return None
      return self.recompute_exact_impedance_device(
          exact_impedance, mass=mass,
          exact_dense_solver=exact_dense_solver,
          component_solver=component_solver, mass_blocks=mass_blocks,
          dof_ids=awake_dof_ids, counts=awake_counts,
          tendon_armature_blocks=tendon_armature_blocks)

    solver_dims = self._constants["solver_dims"]
    if assemble_only:
      # The inverse API needs the same canonical rows as forward dynamics,
      # but MuJoCo's inverse constraint path does not run an optimizer or
      # factor the generalized mass. The -1 stage assembles contact/equality/
      # limit/friction rows and exits before any mass or force read.
      with _solver_dimension_stage(
          self._constants["solver_dims"], -1, self._line_search_iterations):
        dispatch_solver(qfrc_smooth, qfrc_smooth)
      # Assembly-only is used internally by inverse/solver preparation. Keep
      # sparse canonical rows packed here; `assemble_device` is the explicit
      # dense diagnostic/query boundary when a caller requests a dense J.
      result = self._assembly_views(include_optimizer_outputs=False,
                                    materialize_jacobian=False)
      self._position_current_valid = True
      self._assembly_generation += 1
      return result

    if component_primal_route:
      solver_dims = self._constants["solver_dims"]
      self._stage_component_operator(tendon_armature_blocks)
      with _solver_dimension_stage(
          solver_dims, -1, self._line_search_iterations):
        dispatch_solver(mass_blocks)
      exact_data = recompute_exact_impedance()
      with _solver_dimension_stage(
          solver_dims, -3, self._line_search_iterations):
        if exact_data is None:
          dispatch_solver(mass_blocks)
        else:
          with _preassembled_rows_stage(solver_dims):
            dispatch_solver(mass_blocks)
      exact_world_status = None if exact_data is None else exact_data["status"]
    elif component_route:
      solver_dims = self._constants["solver_dims"]
      assembly_mass = mass if mass is not None else w["component_mass_rhs"]
      with _solver_dimension_stage(
          solver_dims, -1, self._line_search_iterations):
        dispatch_solver(assembly_mass)
      rhs_capacity = nr + 1
      component_rhs = w["component_mass_rhs"][:b * rhs_capacity * nv]
      smooth_pair = w.get("component_smooth_acceleration_pair")
      if d.solver_type == int(mujoco.mjtSolver.mjSOL_PGS):
        if smooth_pair is None:
          raise RuntimeError("component PGS force-pair workspace is unavailable")
        low_force = smooth_pair[b * nv:2 * b * nv].view(b, nv)
        if qfrc_smooth_low is None:
          low_force.zero_()
        else:
          low_force.copy_(qfrc_smooth_low)
      else:
        low_force = None
      def prepare_component_rhs():
        exact = recompute_exact_impedance()
        self._pack_component_mass_rhs(
            w["workspace_J"], qfrc_smooth.reshape(-1),
            w["workspace_debug"], component_rhs,
            solver_dims, threads=(b,), group_size=(1,))
        return exact

      exact_data, (component_result, component_status) = (
          _run_component_solve_after_prepare(
              prepare_component_rhs,
              lambda: component_solver.run_device(
                  mass_blocks,
                  component_rhs.reshape(b, rhs_capacity, nv),
                  dof_ids=awake_dof_ids, counts=awake_counts,
                  tendon_armature_blocks=tendon_armature_blocks,
                  rhs_low=low_force)))
      with _solver_dimension_stage(
          solver_dims, -2, self._line_search_iterations):
        if exact_data is None:
          dispatch_solver(component_solver.paired_output)
        else:
          with _preassembled_rows_stage(solver_dims):
            dispatch_solver(component_solver.paired_output)
      component_solver.merge_world_status(component_status, w["out_status"])
      exact_world_status = None if exact_data is None else exact_data["status"]
    elif exact_impedance is not None:
      solver_dims = self._constants["solver_dims"]
      with _solver_dimension_stage(
          solver_dims, -1, self._line_search_iterations):
        dispatch_solver(mass, qfrc_smooth)
      exact_data = recompute_exact_impedance()
      with (_preassembled_rows_stage(solver_dims),
            _paired_smooth_input_stage(solver_dims)
            if paired_dense_input else nullcontext()):
        dispatch_solver(mass)
      exact_world_status = exact_data["status"]
    else:
      with (_paired_smooth_input_stage(solver_dims)
            if paired_dense_input else nullcontext()):
        dispatch_solver(mass)
      exact_world_status = None

    if exact_world_status is not None:
      # The component mass-factor status returned in ``mass_status`` is a
      # borrowed solver buffer. The regular acceleration RHS solve reuses it,
      # so only the helper's independently owned per-world status is valid
      # after that dispatch.
      _merge_exact_world_status(
          torch, w["out_status"], {"status": exact_world_status})

    qacc = w["out_acc"][: b * nv].reshape(b, nv)
    qacc_low = w["out_acc"][b * nv:2 * b * nv].reshape(b, nv)
    qfrc_constraint = w["out_force"][: b * nv].reshape(b, nv)
    status = w["out_status"]
    diagnostics = w["out_diagnostics"][: b * 10].reshape(b, 10)
    history = diagnostics[:, 2:]
    diagnostics = diagnostics[:, :2]

    result = {
        "qacc": qacc,
        "qacc_low": qacc_low,
        "qfrc_constraint": qfrc_constraint,
        "status": status,
        "solver_diagnostics": diagnostics,
        "solver_history": history,
    }

    # Island labels are emitted into the byte-addressed scratch tail using
    # int32 payloads. Borrowed views preserve full row/tree IDs above 2^24.
    if nr > 0:
      debug_world = w["workspace_debug"].reshape(b, self._debug_stride)
      debug_i32 = debug_world.view(torch.int32)
      island_base = self._debug_prefix + self._solver_island_core_scratch
      result["dof_island"] = debug_i32[:, island_base:island_base + nv]
      result["row_island"] = debug_i32[
          :, island_base + nv:island_base + nv + nr]
      tree_base = island_base + nv + nr
      body_base = tree_base + 2 * max(self._solver_island_ntree, 1)
      result["midpoint_blocked_body"] = debug_i32[
          :, body_base:body_base + self._solver_island_nbody]
      result["midpoint_blocked_tree"] = debug_i32[
          :, body_base + max(self._solver_island_nbody, 1):
          body_base + max(self._solver_island_nbody, 1)
          + self._solver_island_ntree]

    if nr > 0:
      # Each world owns its full persistent debug + solver-scratch stride.
      # Reshape the complete allocation so borrowed prefix views keep the
      # correct batch offset when per-world scratch follows the debug prefix.
      w_debug = w["workspace_debug"].reshape(b, self._debug_stride)
      W = w_debug[:, : nr * nr].reshape(b, nr, nr)
      R = w_debug[:, nr * nr : nr * nr + nr].reshape(b, nr)
      ar = w_debug[:, nr * nr + nr : nr * nr + 2 * nr].reshape(b, nr)
      ar_low = w_debug[:, self._aref_low_debug_offset:
                       self._aref_low_debug_offset + nr].reshape(b, nr)
      rhs = w_debug[:, nr * nr + 2 * nr : nr * nr + 3 * nr].reshape(b, nr)
      lam = w_debug[:, nr * nr + 3 * nr : nr * nr + 4 * nr].reshape(b, nr)
      lo = w_debug[:, nr * nr + 4 * nr : nr * nr + 5 * nr].reshape(b, nr)
      hi = w_debug[:, nr * nr + 5 * nr : nr * nr + 6 * nr].reshape(b, nr)
      result.update({
          "W": W,
          "W_regularized": W + torch.diag_embed(R),
          "R": R,
          "ar": ar,
          "ar_low": ar_low,
          "rhs": rhs,
          "lambda": lam,
          "lo": lo,
          "hi": hi,
      })
    if self._jacobian_layout.mode == 0:
      result["J"] = self.materialize_jacobian()
    else:
        # Public query callers can request a dense view explicitly. Native
        # stepping keeps only the packed record live.
        result["J_packed"] = w["workspace_J"]
        result["jacobian_layout"] = self._jacobian_layout
        result["jacobian_pattern"] = self._jacobian_pattern

    if nc > 0:
      contact_rows = w["contact_row_data"][: b * nc * 6 * 6].reshape(b, nc, 6, 6)
      contact_frames = w["contact_frame"][: b * nc * 12].reshape(b, nc, 12)
      contact_forces = w["out_contact_force"][: b * nc * 11].reshape(b, nc, 11)
      contact_jacobians = (
          w["contact_jacobian"][: b * nc * 6 * nv].reshape(b, nc, 6, nv)
          if self._jacobian_layout.mode == 0 else None)
      friction_tensor = self._constants["contact_friction"][:nc * 5].reshape(nc, 5)
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
      })
      if contact_jacobians is not None:
        result["contact_jacobian"] = contact_jacobians

    if d.nr_joint > 0:
      joint_forces = w["out_joint_force"][: b * d.nr_joint].reshape(b, d.nr_joint)
      result["joint_force"] = joint_forces

    self._position_current_valid = True
    self._assembly_generation += 1
    return result

  def recompute_exact_impedance_device(
      self, exact_impedance, *, mass=None, exact_dense_solver=None,
      component_solver=None, mass_blocks=None, dof_ids=None, counts=None,
      tendon_armature_blocks=None):
    """Update canonical R from the current assembled rows and exact mass.

    This runs only after the assembly-only ``dims[20] = -1`` pass. Its mass
    solve uses the current awake principal system and the source row
    impedance; the borrowed RHS/debug view is copied by the helper before
    solver dispatch reuses that slice.
    """
    torch, d, w = self._torch, self.descriptor, self._workspace
    b, nv, nr = int(self.batch_size), int(d.nv), int(d.nr)
    if exact_impedance is None:
      raise ValueError("an exact impedance workspace is required")
    if (getattr(exact_impedance, "batch_size", None) != b
        or getattr(exact_impedance, "row_capacity", None) != nr
        or getattr(exact_impedance, "dof_capacity", None) != nv):
      raise ValueError("exact impedance helper does not match this constraint workspace")
    rows = self._assembly_views(include_optimizer_outputs=False,
                                materialize_jacobian=False)
    jacobian = rows.get("J_packed", rows.get("J"))
    if jacobian is None:
      raise RuntimeError("canonical Jacobian storage is unavailable")
    for key in ("active", "row_impedance"):
      if key not in rows:
        raise RuntimeError(f"DIAGEXACT assembly did not publish {key}")
    packed_jacobian = self._jacobian_layout is not None
    component = component_solver is not None
    helper_status = exact_impedance._workspace["world_status"]
    helper_status.copy_(w["out_status"])
    active_rows = exact_impedance.mask_active_rows_device(
        rows["active"], helper_status)
    if component:
      inverse, mass_status = exact_impedance.solve_mass_rows_device(
          jacobian, active_rows, mass_storage="block_sparse",
          solver=component_solver, mass=None, mass_blocks=mass_blocks,
          dof_ids=dof_ids, counts=counts,
          tendon_armature_blocks=tendon_armature_blocks,
          packed_jacobian=packed_jacobian)
    else:
      inverse, mass_status = exact_impedance.solve_mass_rows_device(
          jacobian, active_rows, mass_storage="dense",
          solver=exact_dense_solver, mass=mass,
          packed_jacobian=packed_jacobian)
    if component:
      component_solver.merge_world_status(mass_status, helper_status)
    else:
      torch.maximum(helper_status, mass_status, out=helper_status)
    active_rows = exact_impedance.mask_active_rows_device(
        rows["active"], helper_status)
    result = exact_impedance.run_device(
        jacobian, inverse, rows["row_impedance"],
        active_rows=active_rows,
        rhs_layout="row_major" if component else "dof_major",
        impratio=float(d.impratio))
    debug = w["workspace_debug"].reshape(b, self._debug_stride)
    debug[:, nr * nr:nr * nr + nr].copy_(result["R"])
    # Normalize the component solver's per-component failures to the
    # world-status ABI consumed by the prepared stage coordinator.
    return {**result, "mass_status": mass_status,
            "status": helper_status}

  def assemble_device(self, poses, qpos, qvel, *, eq_active=None, cvel=None,
                      tendon_J_spatial=None, tendon_length_spatial=None,
                      flex=None, awake_tree=None, cdof=None, cdof_dot=None,
                      candidate_token=None, exact_impedance=None, mass=None,
                      exact_dense_solver=None, component_solver=None,
                      mass_blocks=None, dof_ids=None, counts=None,
                      tendon_armature_blocks=None, world_mask=None):
    """Assemble canonical constraint rows without mass or solver work.

    This is the native inverse-constraint entry point. It refreshes contacts
    and rows for the supplied position/motion state, then returns borrowed
    views of ``J``, ``R``, ``ar``, bounds, activity and contact metadata. The
    returned tensors share this program's workspace and remain valid only
    until its next assembly/query. No generalized mass is formed and no
    optimizer is dispatched.
    """
    torch, d = self._torch, self.descriptor
    b, nv = int(self.batch_size), int(d.nv)
    _validate_cached_tensor(torch, self._device, "qpos", qpos,
                            (b, int(d.nq)), torch.float32)
    _validate_cached_tensor(torch, self._device, "qvel", qvel,
                            (b, nv), torch.float32)
    if not isinstance(poses, dict):
      raise ValueError("poses must be a mapping of FK tensors")
    required_pose_shapes = {
        "body_pos": (b, int(d.nbody), 3),
        "body_quat": (b, int(d.nbody), 4),
        "joint_anchor": (b, int(d.njnt), 3),
        "joint_axis": (b, int(d.njnt), 3),
    }
    if d.npairs:
      required_pose_shapes.update({
          "geom_pos": (b, int(d.ngeom), 3),
          "geom_quat": (b, int(d.ngeom), 4),
      })
    eq_types = np.asarray(d.eq_type, dtype=np.int32)
    has_differential_eq = bool(d.neq and np.any(np.isin(eq_types, (0, 1))))
    has_site_eq = bool(has_differential_eq and np.any(
        np.isin(eq_types, (0, 1))
        & (np.asarray(d.eq_objtype, dtype=np.int32) == 6)))
    if has_site_eq:
      required_pose_shapes.update({
          "site_pos": (b, int(d.nsite), 3),
          "site_quat": (b, int(d.nsite), 4),
      })
    if has_differential_eq:
      required_pose_shapes["root_com"] = (b, int(d.nbody), 3)
    for name, shape in required_pose_shapes.items():
      _validate_cached_tensor(torch, self._device, f"poses[{name!r}]",
                              poses.get(name), shape, torch.float32)
    if cvel is not None:
      _validate_cached_tensor(torch, self._device, "cvel", cvel,
                              (b, int(d.nbody), 6), torch.float32)
    if cdof is not None:
      _validate_cached_tensor(torch, self._device, "cdof", cdof,
                              (b, nv, 6), torch.float32)
    if cdof_dot is not None:
      _validate_cached_tensor(torch, self._device, "cdof_dot", cdof_dot,
                              (b, nv, 6), torch.float32)
    if has_differential_eq and (cvel is None or cdof is None or cdof_dot is None):
      raise ValueError("connect/weld assembly requires cvel, cdof, and cdof_dot")
    if eq_active is not None:
      expected_eq = (b, int(d.neq)) if d.neq else (b, 1)
      _validate_cached_tensor(torch, self._device, "eq_active", eq_active,
                              expected_eq, torch.int32)
    if tendon_J_spatial is not None:
      _validate_cached_tensor(torch, self._device, "tendon_J_spatial",
                              tendon_J_spatial,
                              (b, int(d.ntendon), max(nv, 1)), torch.float32)
    if tendon_length_spatial is not None:
      _validate_cached_tensor(torch, self._device, "tendon_length_spatial",
                              tendon_length_spatial,
                              (b, max(int(d.ntendon), 1)), torch.float32)
    if exact_impedance is not None:
      if (getattr(exact_impedance, "batch_size", None) != b
          or getattr(exact_impedance, "row_capacity", None) != int(d.nr)
          or getattr(exact_impedance, "dof_capacity", None) != nv):
        raise ValueError("exact impedance helper does not match this constraint workspace")
      if component_solver is None:
        if exact_dense_solver is None:
          raise ValueError("dense exact impedance requires a prepared mass solver")
        _validate_cached_tensor(torch, self._device, "mass", mass,
                                (b, nv, nv), torch.float32)
        if (getattr(exact_impedance, "mass_storage", None) != "dense"
            or getattr(exact_dense_solver, "batch_size", None) != b
            or getattr(exact_dense_solver, "nv", None) != nv
            or getattr(exact_dense_solver, "nrhs", None) != int(d.nr)):
          raise ValueError("dense exact impedance solver does not match its prepared workspace")
      else:
        if getattr(exact_impedance, "mass_storage", None) != "block_sparse":
          raise ValueError("sparse exact impedance helper has the wrong mass layout")
        if (component_solver.batch_size != b or component_solver.nv != nv
            or component_solver.rhs_capacity != int(d.nr) + 1):
          raise ValueError("component solver does not match exact impedance rows")
        _validate_cached_tensor(torch, self._device, "mass_blocks", mass_blocks,
                                (b, component_solver.nnz), torch.float32)
        _validate_cached_tensor(torch, self._device, "dof_ids", dof_ids,
                                (b, max(nv, 1)), torch.int32)
        _validate_cached_tensor(torch, self._device, "counts", counts,
                                (b, 3), torch.int32)
        if tendon_armature_blocks is not None:
          _validate_cached_tensor(
              torch, self._device, "tendon_armature_blocks",
              tendon_armature_blocks, (b, component_solver.nnz), torch.float32)
    self._set_world_mask(world_mask)
    result = self.run_device(
        poses, None, qvel, qpos, qvel, eq_active=eq_active, cvel=cvel,
        tendon_J_spatial=tendon_J_spatial,
        tendon_length_spatial=tendon_length_spatial, flex=flex,
        awake_tree=awake_tree, cdof=cdof, cdof_dot=cdof_dot,
        assemble_only=True, candidate_token=candidate_token)
    if (getattr(self, "_jacobian_layout", None) is not None
        and self._jacobian_layout.mode == 1):
      # assemble_device is an explicit row-query boundary used by inverse and
      # diagnostic APIs; it may materialize a dense compatibility result.
      result["J"] = self.materialize_jacobian()
    if exact_impedance is not None:
      exact = self.recompute_exact_impedance_device(
          exact_impedance, mass=mass,
          exact_dense_solver=exact_dense_solver,
          component_solver=component_solver, mass_blocks=mass_blocks,
          dof_ids=dof_ids, counts=counts,
          tendon_armature_blocks=tendon_armature_blocks)
      result["exact_impedance"] = exact
    return result

  def _diagnostic_views(self):
    """Return per-world diagnostics from their fixed ten-word ABI rows."""
    b = self.batch_size
    diagnostics = self._workspace["out_diagnostics"][:b * 10].reshape(b, 10)
    return diagnostics[:, :2], diagnostics[:, 2:]

  def _merge_flex_candidate_status(self):
    """Propagate detector failures without reading candidate status to host."""
    bundle = getattr(self, "_flex_contact_current", None)
    if bundle is None:
      return
    contact_result = bundle.get("contact_result")
    candidate_status = (None if contact_result is None else
                        contact_result.get("narrowphase_status"))
    if candidate_status is None:
      return
    torch, b = self._torch, int(self.batch_size)
    slot_count = int(self.descriptor.flex_contact_descriptor.slot_count)
    _validate_cached_tensor(
        torch, self._device, "flex narrowphase status", candidate_status,
        (b, slot_count), torch.int32)
    if slot_count:
      self._merge_flex_narrowphase_status_kernel(
          candidate_status.reshape(-1), self._workspace["out_status"],
          self._constants["solver_dims"], threads=(b,), group_size=(1,))

  def _assembly_views(self, *, include_optimizer_outputs,
                      materialize_jacobian=False):
    """Build borrowed canonical row/contact views after row assembly."""
    torch, w, d = self._torch, self._workspace, self.descriptor
    b, nv, nr, nc = self.batch_size, d.nv, d.nr, d.ncontacts_max
    debug = w["workspace_debug"].reshape(b, self._debug_stride)
    # Lightweight CPU stand-ins used by inverse-assembly callers may predate
    # the packed-J record ABI. Production constructors always install a
    # layout; an absent layout denotes their legacy dense test backing.
    jacobian_layout = getattr(self, "_jacobian_layout", None)
    jacobian_pattern = getattr(self, "_jacobian_pattern", None)
    result = {}
    if include_optimizer_outputs:
      solver_diagnostics, solver_history = self._diagnostic_views()
      result.update({
          "qacc": w["out_acc"][:b * nv].reshape(b, nv),
          "qacc_low": w["out_acc"][b * nv:2 * b * nv].reshape(b, nv),
          "qfrc_constraint": w["out_force"][:b * nv].reshape(b, nv),
          "status": w["out_status"],
          "solver_diagnostics": solver_diagnostics,
          "solver_history": solver_history,
      })
    if nr:
      result.update({
          "J_packed": w["workspace_J"],
          "jacobian_layout": jacobian_layout,
          "jacobian_pattern": jacobian_pattern,
          "R": debug[:, nr * nr:nr * nr + nr],
          "ar": debug[:, nr * nr + nr:nr * nr + 2 * nr],
          "ar_low": debug[:, self._aref_low_debug_offset:
                          self._aref_low_debug_offset + nr],
          "rhs": debug[:, nr * nr + 2 * nr:nr * nr + 3 * nr],
          # Stage -1 publishes source impedance in the canonical RHS slice.
          # This borrowed view has `debug_stride` batch spacing and is copied
          # to owned helper storage before a solver dispatch reuses the slice.
          "row_impedance": debug[:, nr * nr + 2 * nr:nr * nr + 3 * nr],
          "lambda": debug[:, nr * nr + 3 * nr:nr * nr + 4 * nr],
          "lo": debug[:, nr * nr + 4 * nr:nr * nr + 5 * nr],
          "hi": debug[:, nr * nr + 5 * nr:nr * nr + 6 * nr],
          "active": debug[:, nr * nr + 6 * nr:nr * nr + 7 * nr],
      })
      if jacobian_layout is None:
        result["J"] = w["workspace_J"][:b * nr * nv].reshape(b, nr, nv)
      elif jacobian_layout.mode == 0:
        result["J"] = self.materialize_jacobian()
      elif materialize_jacobian:
        result["J"] = self.materialize_jacobian()
    if nc:
      rows = w["contact_row_data"][:b * nc * 36].reshape(b, nc, 6, 6)
      frames = w["contact_frame"][:b * nc * 12].reshape(b, nc, 12)
      result.update({
          "contact_mask": rows[:, :, 0, 0],
          "contact_distance": rows[:, :, 0, 1],
          "contact_velocity": rows[:, :, 0, 2],
          "contact_reference": rows[:, :, 0, 3],
          "contact_impedance": rows[:, :, 0, 4],
          "contact_normal": frames[:, :, :3],
          "contact_tangent1": frames[:, :, 3:6],
          "contact_tangent2": frames[:, :, 6:9],
          "contact_position": frames[:, :, 9:12],
          "contact_friction": self._constants["contact_friction"][:nc * 5].reshape(nc, 5),
          "contact_condim": self._constants["contact_condim"][:nc * 3],
      })
      if getattr(getattr(self, "_jacobian_layout", None), "mode", 0) == 0:
        result["contact_jacobian"] = w["contact_jacobian"][:b * nc * 6 * nv].reshape(
            b, nc, 6, nv)
    bundle = getattr(self, "_flex_contact_current", None)
    if bundle is not None:
      flex_rows = bundle["rows"]
      result["flex_contact_result"] = bundle["contact_result"]
      result["flex_contact_rows"] = flex_rows
      if include_optimizer_outputs and nr:
        start = int(d.flex_contact_base)
        stop = start + int(d.n_flex_contact_rows)
        result["flex_contact_row_force"] = debug[
            :, nr * nr + 3 * nr + start:nr * nr + 3 * nr + stop]
    return result
