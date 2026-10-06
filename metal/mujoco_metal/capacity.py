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

"""Explicit capacity and allocation architecture (milestone 017).

Fixed small-model ceilings become named, budgeted capacities resolved at
host construction boundaries. Device execution never grows allocations:
overflow is a host lowering error with requested-vs-allowed detail (no
silent truncation), and per-environment runtime failures stay isolated via
the existing status channel.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

# Conservative default budgets carried over from earlier profiles. Callers
# can raise them explicitly; backing-size, int32-addressing, and memory-budget
# checks remain authoritative for the dynamically sized solver.
BASE_NVIDIA_NV = 32
BASE_MAX_PAIRS = 32
BASE_MAX_SLOTS = 50
# Pinned mjMAXCONPAIR witnesses need up to10pyramid rows each plus
# generalized rows. Model-sized capacity checks remain authoritative.
BASE_MAX_ROWS = 640
# Dense solver threshold: at or below this row count the native solver runs
# the exact small-model dense path (unchanged math). Above it the block
# path takes over (017 follow-up commits).
DENSE_ROW_THRESHOLD = 96
AUTO_JACOBIAN_DENSE_NV = 60
SIGNED_I32_ELEMENT_LIMIT = (1 << 31) - 1


def _integer_count(name, value, *, minimum=0):
  """Validate a public dimension without truncation or bool coercion."""
  if (isinstance(value, (bool, np.bool_)) or
      not isinstance(value, (int, np.integer))):
    raise ValueError(f"{name} must be an integer")
  result = int(value)
  if result < minimum:
    qualifier = "positive" if minimum == 1 else "non-negative"
    raise ValueError(f"{name} must be {qualifier}")
  return result


def primal_scratch_floats(nv, nr, solver_type, ntree=0, nbody=0,
                          mass_storage="dense"):
  """Exact per-world scratch for the currently implemented primal solver.

  PGS needs dynamic row-state/order scratch plus the supplied qacc warmstart.
  CG needs twelve DOF vectors and six row vectors, plus the retained row
  matrix and qacc warmstart.
  Newton uses four additional DOF vectors for a matrix-free, mass-
  preconditioned conjugate-gradient application of the primal Hessian.
  """
  nv = _integer_count("nv", nv)
  nr = _integer_count("nr", nr)
  ntree = _integer_count("ntree", ntree)
  nbody = _integer_count("nbody", nbody)
  if (isinstance(solver_type, (bool, np.bool_))
      or not isinstance(solver_type, (int, np.integer, mujoco.mjtSolver))):
    raise ValueError("solver_type must be an integer")
  if mass_storage not in ("dense", "block_sparse"):
    raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
  v, r = max(nv, 1), max(nr, 1)
  t = max(ntree, 1)
  solver = int(solver_type)
  if solver == int(mujoco.mjtSolver.mjSOL_PGS):
    # PGS retains active-row vectors for its source-order/island sweeps.
    core = 14 * r
  elif solver == int(mujoco.mjtSolver.mjSOL_CG):
    core = 13 * v + 5 * r
  elif solver == int(mujoco.mjtSolver.mjSOL_NEWTON):
    core = 16 * v + 5 * r
  else:
    core = 1
  # Dense Cholesky storage is dynamically indexed device scratch. Sparse
  # component layouts already own their packed factor blocks and must never
  # reserve a global nv-by-nv factor workspace.
  dense_factor_words = 0
  if mass_storage == "dense":
    dense_factor_words = v * v + 2 * v
    core += dense_factor_words
  # Typed per-row activation, contact/cone block identities, and friction
  # parameters live after the solver's vector/factor workspace. These maps
  # replace fixed thread arrays in the block kernel and remain live during
  # the complete outer solve.
  core += 12 * r
  # Dense Delassus construction retains M^-1 J^T for the solve. This is
  # sized from the actual row and DOF counts, not a fixed thread matrix.
  core += v * r
  # Islands need bit-preserving DOF/row maps and union-find parent/used/root
  # arrays plus exact midpoint eligibility masks. Keep those after the solver
  # core; retain an independent final nv-sized qacc_warmstart tail.
  b = max(int(nbody), 1)
  # One additional tree mask carries the precomputed awake set into the
  # coupled solver without consuming another Metal buffer binding.
  # Equality point-Jacobian scratch stores two 3xnv point/Jr blocks. Keeping
  # this dynamic workspace in the owned debug tail avoids fixed thread arrays
  # and local-memory ceilings in connect/weld assembly.
  equality_jacobian_scratch = 12 * v
  return core + 2 * v + r + 5 * t + b + equality_jacobian_scratch


def solver_debug_layout(nv, nr, solver_type, ntree=0, nbody=0,
                        component_nnz=0, component_layout_words=0,
                        mass_storage="dense"):
  """Return exact per-world debug intervals used by solver shaders.

  The typed component metadata lives in the int32 view of the final operator
  tail.  The float prefix/scratch interval ends before that tail; qacc and
  equality-Jacobian offsets are explicit so a larger final batch stride never
  moves them into component storage.
  """
  values = {}
  for name, value in (("nv", nv), ("nr", nr), ("ntree", ntree),
                      ("nbody", nbody), ("component_nnz", component_nnz),
                      ("component_layout_words", component_layout_words)):
    values[name] = _integer_count(name, value)
  if (isinstance(solver_type, (bool, np.bool_))
      or not isinstance(solver_type, (int, np.integer, mujoco.mjtSolver))):
    raise ValueError("solver_type must be an integer")
  if mass_storage not in ("dense", "block_sparse"):
    raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
  v, r = values["nv"], values["nr"]
  if solver_type == int(mujoco.mjtSolver.mjSOL_PGS):
    algorithm_words = 14 * max(r, 1)
  elif solver_type == int(mujoco.mjtSolver.mjSOL_CG):
    algorithm_words = 13 * max(v, 1) + 5 * max(r, 1)
  elif solver_type == int(mujoco.mjtSolver.mjSOL_NEWTON):
    algorithm_words = 16 * max(v, 1) + 5 * max(r, 1)
  else:
    algorithm_words = 1
  dense_factor_words = (max(v, 1) ** 2 + 2 * max(v, 1)
                        if mass_storage == "dense" else 0)
  block_workspace_offset = algorithm_words + dense_factor_words
  z_workspace_offset = block_workspace_offset + 12 * r
  prefix = r * r + 7 * r
  scratch = primal_scratch_floats(v, r, solver_type, values["ntree"],
                                  values["nbody"], mass_storage)
  base_stride = prefix + scratch
  qacc_offset = base_stride - max(v, 1)
  equality_scratch_offset = (base_stride - max(v, 1)
                             - 12 * max(v, 1))
  operator_offset = base_stride
  armature_offset = operator_offset
  layout_offset = armature_offset + values["component_nnz"]
  high_low_debug_offset = layout_offset + values["component_layout_words"]
  # Unlike the component metadata offsets, high/low vectors are addressed
  # from the solver_scratch pointer (debug base + debug_prefix).
  high_low_workspace_offset = high_low_debug_offset - prefix
  # Keep the independent high/low accumulators after all existing solver and
  # component regions so none of their offsets move. Kernels add this
  # per-world offset to the current debug record base.
  # The final nv plane retains the immutable smooth-acceleration low word while
  # primal islands update their own accepted acceleration pairs.
  high_low_workspace_length = 8 * max(v, 1) + 3 * max(r, 1)
  final_stride = high_low_debug_offset + high_low_workspace_length
  if min(qacc_offset, equality_scratch_offset) < prefix:
    raise ValueError("solver scratch layout overlaps the debug prefix")
  if final_stride > SIGNED_I32_ELEMENT_LIMIT:
    raise OverflowError("solver debug stride exceeds signed int32 addressing")
  return {
      "debug_prefix": prefix,
      "block_workspace_offset": block_workspace_offset,
      "primal_workspace_offset": dense_factor_words,
      "z_workspace_offset": z_workspace_offset,
      "base_stride": base_stride,
      "final_stride": final_stride,
      "qacc_warmstart_offset": qacc_offset,
      "equality_jacobian_offset": equality_scratch_offset,
      "component_operator_offset": operator_offset,
      "component_armature_offset": armature_offset,
      "component_layout_offset": layout_offset,
      "high_low_debug_offset": high_low_debug_offset,
      "high_low_workspace_offset": high_low_workspace_offset,
      "high_low_workspace_length": high_low_workspace_length,
  }


class CapacityOverflow(ValueError):
  """Host lowering error: model needs more than the budgeted capacity."""


@dataclass(frozen=True)
class CapacityLimits:
  """User-visible budgets. Growth happens only at host boundaries."""

  max_nv: int = BASE_NVIDIA_NV
  max_pairs: int = BASE_MAX_PAIRS
  max_slots: int = BASE_MAX_SLOTS
  max_rows: int = BASE_MAX_ROWS
  max_batch: int = 16
  memory_budget_bytes: int = 1 << 30
  # Device-linear implicit solve budgets. These are independent from the
  # contact solver's opt.iterations and never limit model nv admission.
  gmres_krylov_dimension: int | None = None
  gmres_max_iterations: int | None = None


@dataclass(frozen=True)
class CapacityEstimate:
  """Deterministic estimate for solver and runtime buffer envelope."""

  nv: int
  nbody: int
  ngeom: int
  npairs: int
  nslots: int
  nr: int
  batch: int
  dense_path: bool
  mass_storage: str
  jacobian_kind: str
  jacobian_auto_threshold: int
  memory_bytes: int
  memory_breakdown: tuple


def _bytes(n):
  return int(n) * 4


def _bytes_i32(n):
  return int(n) * 4


def _fk_auxiliary_words(batch, nmocap, nbody, ntree):
  """Exact flat-word count for MetalKinematics' auxiliary tensor.

  Pose high/low/tail fields live in ``pose_output_arena``; this buffer owns
  only mocap input, compiled body tree/mocap maps, awake/cache flags, and the
  per-tree mismatch flags.
  """
  b = _integer_count("batch", batch, minimum=1)
  nmocap = _integer_count("nmocap", nmocap)
  nbody = _integer_count("nbody", nbody)
  ntree = _integer_count("ntree", ntree)
  mocap_words = b * nmocap * 7 if nmocap else 1
  return max(mocap_words + 2 * nbody + b * max(ntree, 1) + b
             + b * max(ntree, 1), 1)


def _check_i32_elements(name, elements):
  """Reject any flat device allocation that kernels address with int32."""
  elements = int(elements)
  if elements > SIGNED_I32_ELEMENT_LIMIT:
    raise CapacityOverflow(
        f"{name} element count ({elements}) exceeds signed 32-bit addressing "
        f"limit {SIGNED_I32_ELEMENT_LIMIT}")


def _runtime_buffer_sizes(model, batch, *, npairs=None, rhs_capacity=None,
                          coupled_rows=0,
                          mass_storage="dense", legacy_contact_pairs=None,
                          legacy_joint_rows=None, legacy_total_rows=None,
                          effective_edge_count=None,
                          effective_gmres_dimension=None,
                          effective_gmres_iterations=None,
                          diag_exact_rows=None, diag_exact_cones=None,
                          touch_grid_contact_capacity=None,
                          touch_grid_pair_capacity=None,
                          bundled_cable_segments=None,
                          magnetic_force_plugins=0,
                          site_feedback_plugins=0,
                          native_actuator_user_workspace_bytes=0):
  """Return named runtime buffer shapes for pre-allocation address checks.

  Entries map to individual tensors in DeviceState, FK, smooth dynamics,
  Simulation, or SensorProgram; combined layouts are named only where the
  implementation truly uses a single backing buffer (FK auxiliary and sensor
  subtree runtime). Solver/collision buffers are estimated separately by
  :func:`estimate_workspace`. Some entries describe lazy/profile-conditional
  buffers, so their sum is a conservative memory envelope, not a peak-live
  memory profile.
  """
  b = int(batch)
  coupled_rows = _integer_count("coupled_rows", coupled_rows)
  native_actuator_user_workspace_bytes = _integer_count(
      "native_actuator_user_workspace_bytes",
      native_actuator_user_workspace_bytes)
  if mass_storage not in ("dense", "block_sparse"):
    raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
  # Standalone legacy profiles allocate their producer-owned canonical rows
  # as well as (sometimes) a combined Simulation view. Keep them conditional
  # so integrated profiles are not charged for backing they never construct.
  legacy_contact_pairs = (
      None if legacy_contact_pairs is None else
      _integer_count("legacy_contact_pairs", legacy_contact_pairs))
  legacy_joint_rows = (
      None if legacy_joint_rows is None else
      _integer_count("legacy_joint_rows", legacy_joint_rows))
  legacy_total_rows = (
      None if legacy_total_rows is None else
      _integer_count("legacy_total_rows", legacy_total_rows))
  effective_edge_count = (
      None if effective_edge_count is None else
      _integer_count("effective_edge_count", effective_edge_count))
  diag_exact_rows = (None if diag_exact_rows is None else
                     _integer_count("diag_exact_rows", diag_exact_rows))
  diag_exact_cones = (None if diag_exact_cones is None else
                      _integer_count("diag_exact_cones", diag_exact_cones,
                                     minimum=1))
  touch_grid_contact_capacity = (
      None if touch_grid_contact_capacity is None else
      _integer_count("touch_grid_contact_capacity",
                     touch_grid_contact_capacity))
  touch_grid_pair_capacity = (
      None if touch_grid_pair_capacity is None else
      _integer_count("touch_grid_pair_capacity", touch_grid_pair_capacity))
  bundled_cable_segments = (
      None if bundled_cable_segments is None else
      _integer_count("bundled_cable_segments", bundled_cable_segments))
  if (diag_exact_rows is None) != (diag_exact_cones is None):
    raise ValueError("diag_exact_rows and diag_exact_cones must be supplied together")
  nq, nv = int(model.nq), int(model.nv)
  nb, nj = int(model.nbody), int(model.njnt)
  ng, ns = int(model.ngeom), int(model.nsite)
  nsensor, nsensordata = int(model.nsensor), int(model.nsensordata)
  ntree = int(np.max(np.asarray(model.body_treeid), initial=-1)) + 1
  nmocap = int(model.nmocap)
  na, neq = int(model.na), int(model.neq)
  nhistory = int(getattr(model, "nhistory", 0))
  nuserdata = int(getattr(model, "nuserdata", 0))
  npluginstate = int(getattr(model, "npluginstate", 0))
  ncam, ntendon, nu = int(model.ncam), int(model.ntendon), int(model.nu)
  # MetalRungeKutta snapshots the already sleep-masked incoming velocity so
  # stage callbacks cannot alias/overwrite the X0 value. This allocation is
  # present only when Simulation constructs an RK4 runner.
  try:
    import mujoco
    is_rk4 = int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4)
  except (ImportError, AttributeError, TypeError, ValueError):
    is_rk4 = False
  nflexcontactslot = 0
  nflexcontactlinks = 0
  flex_ccd_float_stride = 0
  flex_ccd_int_stride = 0
  nflexradius = 0
  flex_geom_hull_words = 0
  flex_geom_hull_info_words = 0
  if int(getattr(model, "nflex", 0)) and int(getattr(model, "nflexvert", 0)):
    # Flex contact row metadata is a host-lowered immutable descriptor. It is
    # Torch-free and gives the exact solver-side appended friction/condim
    # arrays used by coupled_constraints.
    from mujoco_metal.flex_contact import lower_flex_contacts
    flex_contact_descriptor = lower_flex_contacts(model)
    nflexcontactslot = int(flex_contact_descriptor.slot_count)
    nflexcontactlinks = int(flex_contact_descriptor.link_capacity)
    nflexradius = int(model.nflex)
    # The flex detector owns one immutable packed geometry payload. Its
    # octree low/tail source words are appended to that same backing, so
    # account the exact expanded allocation here rather than creating an
    # untracked side buffer.
    from mujoco_metal.flex_contact import _flex_mesh_hull
    flex_hull, flex_hull_info = _flex_mesh_hull(
        model, flex_contact_descriptor)
    flex_geom_hull_words = max(int(flex_hull.size), 1)
    flex_geom_hull_info_words = int(flex_hull_info.size)
    # MetalFlexContactProgram keeps the pinned per-query CCD arena alive for
    # every batch/slot. Match its constructor's exact sizing here so the
    # runtime signed-index and byte-budget checks include the large detector
    # scratch as well as both parts of the compiled flex-radius operand.
    from mujoco_metal.flex_contact import _flex_ccd_workspace_capacity
    ccd_vertices, ccd_faces, ccd_horizon, ccd_stack = (
        _flex_ccd_workspace_capacity(
            model.opt.ccd_iterations, b, nflexcontactslot))
    flex_ccd_float_stride = (
        (ccd_vertices * 39 + ccd_faces * 20 + 3) & ~3)
    flex_ccd_int_stride = (
        ((ccd_faces + 1) & ~1) + ccd_horizon * 2 + ccd_stack * 6)
  # The smooth stage has an exact packed component layout. The block values
  # use dense storage only within each compiled component; this may be larger
  # than MuJoCo's scalar CSR but never exceeds nv**2. Use the same compiler as
  # the runtime constructor so this estimate follows tendon-armature coupling
  # and the pinned compressed-M sparsity, including cross-tree components.
  from mujoco_metal.model import actuator_tendon_inheritance
  from mujoco_metal.mass_layout import compile_tree_mass_layout
  inherited_tendon_armature, _, _ = actuator_tendon_inheritance(model)
  effective_tendon_armature = (
      np.asarray(model.tendon_armature, dtype=np.float64)
      + inherited_tendon_armature)
  has_tendon_armature = bool(np.any(effective_tendon_armature != 0.0))
  mass_layout = compile_tree_mass_layout(
      model.body_treeid, model.dof_bodyid,
      tendon_treeid=model.tendon_treeid,
      tendon_treenum=model.tendon_treenum,
      tendon_armature=effective_tendon_armature,
      tendon_j_rowadr=model.ten_J_rowadr,
      tendon_j_rownnz=model.ten_J_rownnz,
      tendon_j_colind=model.ten_J_colind,
      mass_rowadr=model.M_rowadr,
      mass_rownnz=model.M_rownnz,
      mass_colind=model.M_colind)
  component_nnz = int(mass_layout["nnz"])
  component_count = int(mass_layout["ncomponent"])

  # DeviceState fields. Zero-sized tensors use a one-element backing buffer
  # where torch.empty/zeros is called directly; optional tensors are absent.
  parts = [
      ("state.qpos", max(b * nq, 1)),
      ("state.qvel", max(b * nv, 1)),
      ("state.qacc", max(b * nv, 1)),
      ("state.qacc_warmstart", max(b * nv, 1)),
      ("state.time", b), ("state.status", b),
      ("state.warning_number", b * int(mujoco.mjtWarning.mjNWARNING)),
      ("state.warning_lastinfo", b * int(mujoco.mjtWarning.mjNWARNING)),
      ("state.row_reset_epoch", b),
      ("state.eq_active", b * neq if neq else 0),
      ("state.mocap_pos", b * nmocap * 3 if nmocap else 0),
      ("state.mocap_quat", b * nmocap * 4 if nmocap else 0),
      ("state.act", b * na if na else 0),
      ("state.history", b * nhistory),
      ("state.userdata", b * nuserdata),
      ("state.plugin_state", b * npluginstate),
      # Device-side AUTORESET defaults/mask and static dispatch metadata.
      ("state.reset.qpos0", max(nq, 1)),
      ("state.reset.eq_active0", max(neq, 1)),
      ("state.reset.mocap_pos0", max(nmocap * 3, 1)),
      ("state.reset.mocap_quat0", max(nmocap * 4, 1)),
      ("state.reset.history0", max(nhistory, 1)),
      ("state.reset.userdata0", max(nuserdata, 1)),
      ("state.reset.plugin_state0", max(npluginstate, 1)),
      ("state.reset.mask_i32", b),
      ("state.reset.row_operation_mask_i32", b),
      ("state.reset.dimensions", 10),
      ("state.reset.warning_dimensions", 4),
      ("spatial_tendon.dimensions", 11 + b),
      ("state.reset.clear_rows_dimensions", 3),
      ("state.reset.strided_copy_dimensions", 4),
      ("state.reset.packed_copy_dimensions", 6),
      ("state.reset.update_rows_dimensions", 4),
      ("state.reset.dummy_float", 1),
      ("state.reset.dummy_int", 1),
      # Appended solver-facing contact cone constants for canonical flex
      # candidate slots; the pre-existing rigid contact constants are
      # accounted by their own model inventory.
      ("model.solver_flex_contact_friction", nflexcontactslot * 5),
      ("model.solver_flex_contact_condim", nflexcontactslot * 3),
      ("flex_contact.radius_high", nflexradius),
      ("flex_contact.radius_mid", nflexradius),
      ("flex_contact.radius_low", nflexradius),
      ("flex_contact.geom_hull_payload", flex_geom_hull_words),
      ("flex_contact.geom_hull_info", flex_geom_hull_info_words),
      ("flex_contact.margin", nflexcontactslot),
      ("flex_contact.gap", nflexcontactslot),
      ("flex_contact.margin_gap_high", nflexcontactslot * 2),
      ("flex_contact.geom_size_mid", ng * 3 if nflexradius else 0),
      ("flex_contact.geom_size_low", ng * 3 if nflexradius else 0),
      ("flex_contact.margin_gap_mid", nflexcontactslot * 2),
      ("flex_contact.margin_gap_low", nflexcontactslot * 2),
      ("flex_contact.ccd_tolerance_mid", 1 if nflexradius else 0),
      ("flex_contact.ccd_tolerance_low", 1 if nflexradius else 0),
      ("flex_contact.vertex_high_low_pair",
       b * int(getattr(model, "nflexvert", 0)) * 9
       if nflexradius else 0),
      ("flex_contact.vertex_low_tail_dispatch",
       (b * int(getattr(model, "nflexvert", 0)) * 6
        + b * ng * 6) if nflexradius else 0),
      # The immutable zero fallbacks and shared dispatch slab are distinct
      # tensors from the caller/FK residual planes. The slab contains the
      # flex low/tail pair followed by the geometry low/tail pair.
      ("flex_contact.zero_vertex_residual",
       2 * b * int(getattr(model, "nflexvert", 0)) * 3
       if nflexradius else 0),
      ("flex_contact.zero_geom_residual",
       2 * b * ng * 3 if nflexradius else 0),
      # The direct dispatch slab has the flex low/tail planes followed by
      # geometry low/tail planes. Zero fallbacks are separately allocated.
      # FlexContactRows appends a per-world recovery predicate to both
      # persistent dispatch records (the base records are 7 and 5 int32s).
      ("flex_contact_rows.assemble_mask_dims",
       7 + b if nflexcontactslot else 0),
      ("flex_contact_rows.refresh_mask_dims",
       5 + b if nflexcontactslot else 0),
      ("flex_contact.epa_float_workspace",
       b * max(nflexcontactslot, 1) * flex_ccd_float_stride
       if nflexradius else 0),
      ("flex_contact.epa_int_workspace",
       b * max(nflexcontactslot, 1) * flex_ccd_int_stride
       if nflexradius else 0),
      # Source-ordered filterFlexContacts scratch: compact slot permutation,
      # positional selected bytes represented as int32, and three float32
      # words per binary64-equivalent min-distance score.
      ("flex_contact.filter_permutation", b * nflexcontactslot),
      ("flex_contact.filter_selected_position", b * nflexcontactslot),
      # Per-contact midpoint high values remain in contact_pos; this backing
      # stores the source binary64 residual words used by filterFlexContacts.
      ("flex_contact.contact_pos_low_tail", b * nflexcontactslot * 12),
      ("flex_contact.filter_min_distance_words",
       b * nflexcontactslot * 3),
  ]
  if is_rk4:
    parts.append(("rk4.zero", b * nv))
    parts.append(("rk4.initial_qvel", b * nv))
  if effective_edge_count is not None:
    from mujoco_metal.implicit_effective import effective_gmres_workspace_sizes
    edge_width = max(int(effective_edge_count), 1)
    nv_width = max(nv, 1)
    gmres_dimensions = {}
    if effective_gmres_dimension is not None:
      gmres_dimensions["krylov_dimension"] = effective_gmres_dimension
    if effective_gmres_iterations is not None:
      gmres_dimensions["max_iterations"] = effective_gmres_iterations
    parts.extend((f"implicit_effective.{name}", int(count)) for name, count in
                 effective_gmres_workspace_sizes(
                     batch=b, nv=nv, edge_count=effective_edge_count,
                     **gmres_dimensions).items())
    # Sparse derivative writers retain their own small metadata and scratch
    # backings in addition to the shared GMRES COO values/indices. Keep each
    # actual tensor separate so signed indexing guards match its constructor.
    parts.extend((
        ("velocity_derivative.diagonal_slots", nv_width),
        ("velocity_derivative.dims", 3),
        ("velocity_derivative.column_dims", 4),
        ("velocity_derivative.symmetric_lower_source_slots", edge_width),
        ("velocity_derivative.symmetric_values", b * edge_width),
        ("smooth.bias_coo_column_offsets", nv + 1),
        ("smooth.bias_coo_edge_rows", edge_width),
        ("smooth.bias_coo_edge_slots", edge_width),
    ))
    # The general actuator program owns independent pattern/index buffers and
    # an internal derivative scratch even when production writes into the
    # shared COO accumulator.
    if nu:
      parts.extend((
          # The scalar-motor compatibility program shares the actuator
          # profile's batch mask storage when that profile is selected.
          ("scalar_motor.dims", 6 + b),
          ("actuator.transmission_kinematics_dims", 8 + b),
          ("actuator.cached_velocity_dims", 3 + b),
          ("actuator.force_dot_dims", 5 + b),
          ("actuator.force_dims", 7 + b),
          ("actuator.qfrc_dims", 4 + b),
          ("actuator.velocity_derivative_pattern", 1),
          ("actuator.velocity_derivative_edge_rows", edge_width),
          ("actuator.velocity_derivative_edge_cols", edge_width),
          ("actuator.velocity_derivative_dims", 8 + b),
          ("actuator.velocity_derivative_scratch", b * edge_width),
      ))
    # Fixed and spatial tendon derivative entry points each own a compact
    # five-word ABI tensor. The fixed-tendon program is constructed for the
    # integrated sparse profile; the spatial one exists only for spatial
    # paths in the compiled model.
    parts.append(("fixed_tendon.coo_damping_dims", 5))
    from mujoco_metal.spatial_tendons import SpatialTendonModel
    if SpatialTendonModel(model).has_spatial:
      parts.append(("spatial_tendon.coo_damping_dims", 5))
    # The fluid finite-difference producer preallocates these three
    # batch-by-DOF vectors only when fluid forces are enabled for this model.
    opt = model.opt
    if (float(opt.density) != 0.0 or float(opt.viscosity) != 0.0
        or np.any(np.asarray(opt.wind) != 0.0)):
      parts.extend((
          ("fluid.dimensions", 6 + b),
          ("fluid.derivative_base", b * nv),
          ("fluid.derivative_velocity", b * nv),
          ("fluid.derivative_column", b * nv),
      ))
    # The sparse flex correction exists only for the component implicit
    # profile and models with compiled implicit flex stiffness. Its workspace
    # is linear in nv and contains no dense mass/stiffness matrix.
    from mujoco_metal.flex_implicit import requires_flex_implicit_correction
    if (mass_storage == "block_sparse"
        and requires_flex_implicit_correction(model)):
      from mujoco_metal.sparse_flex_implicit import (
          sparse_flex_correction_workspace_sizes)
      parts.extend((name, int(count)) for name, count in
                   sparse_flex_correction_workspace_sizes(
                       b, nv, edge_count=effective_edge_count).items())
  # MetalFlex retains this compiled support and the two per-edge coefficient
  # vectors for its ordinary operators too, independently of the selected
  # effective-implicit mass profile.  The column array has a one-int binding
  # sentinel for an empty compiled pattern; logical NNZ remains 0 in dims.
  nflexedge = int(getattr(model, "nflexedge", 0))
  nflex = int(getattr(model, "nflex", 0))
  if nflex:
    # Dedicated result owners used by selected-world checkAcc recovery. The
    # ordinary flex force/tangent buffers may be borrowed by a prepared
    # forward record and therefore cannot be cleared for a masked refresh.
    v = max(nv, 1)
    parts.extend((
        # Paired-coordinate residuals supplement, but do not replace, the
        # established float32 pose/material buffers. They remain live through
        # common flex support and the frozen POS context.
        ("flex.vertex_position_low", b * int(getattr(model, "nflexvert", 0)) * 3),
        ("flex.vertex_position_tail", b * int(getattr(model, "nflexvert", 0)) * 3),
        ("flex.node_position_low",
         b * int(np.asarray(getattr(model, "flex_node", [])).shape[0]) * 3),
        ("flex.node_position_tail",
         b * int(np.asarray(getattr(model, "flex_node", [])).shape[0]) * 3),
        ("flex.position_context_low",
         b * (int(getattr(model, "nbody", 0))
              + int(getattr(model, "nflexvert", 0))
              + int(np.asarray(getattr(model, "flex_node", [])).shape[0])) * 3),
        ("flex.position_context_tail",
         b * (int(getattr(model, "nbody", 0))
              + int(getattr(model, "nflexvert", 0))
              + int(np.asarray(getattr(model, "flex_node", [])).shape[0])) * 3),
        ("flex.local_vertex_low", int(getattr(model, "nflexvert", 0)) * 3),
        ("flex.local_vertex_tail", int(getattr(model, "nflexvert", 0)) * 3),
        ("flex.local_node_low",
         int(np.asarray(getattr(model, "flex_node", [])).shape[0]) * 3),
        ("flex.local_node_tail",
         int(np.asarray(getattr(model, "flex_node", [])).shape[0]) * 3),
        ("flex.interpolation_weight_low",
         int(getattr(model, "nflexvert", 0)) * 27
         + int(np.asarray(getattr(model, "flex_node", [])).shape[0]) * 26),
        ("flex.interpolation_weight_tail",
         int(getattr(model, "nflexvert", 0)) * 27
         + int(np.asarray(getattr(model, "flex_node", [])).shape[0]) * 26),
        ("flex.recovery_qfrc_passive", b * v),
        ("flex.recovery_damping_tangent", b * v * v),
        ("flex.recovery_stiffness_tangent", b * v * v),
        ("flex.recovery_body_joint_maps",
         3 * int(getattr(model, "nbody", 0))
         + 2 * int(getattr(model, "njnt", 0))),
        ("flex.recovery_vertex_bindings",
         5 * int(getattr(model, "nflexvert", 0))),
        ("flex.recovery_node_bindings",
         5 * int(np.asarray(getattr(model, "flex_node", [])).shape[0])),
        ("flex.recovery_edge_bindings", 2 * nflexedge),
        ("flex.recovery_point_mask_dims", 7 + b),
        ("flex.recovery_edge_mask_dims", 4 + b),
        ("flex.recovery_edge_force_dims", 5),
        ("flex.recovery_copy_mask_dims", 2 + b),
        ("flex.recovery_matvec_mask_dims", 3 + b),
        ("flex.recovery_interpolation_mask_dims", 7 + 2 * b),
        # Stretch/interp use six header words; bend/shell-bend use four.
        # Each header has a trailing int32[B] active-row predicate.
        ("flex.recovery_material_mask_dims", 2 * (6 + b) + 2 * (4 + b)),
    ))
  if int(getattr(model, "nflex", 0)) and nflexedge:
    flex_edge_nnz = int(np.asarray(model.flexedge_J_colind).size)
    parts.extend((
        ("flex.edge_j_rowadr", nflexedge),
        ("flex.edge_j_rownnz", nflexedge),
        ("flex.edge_j_colind", max(flex_edge_nnz, 1)),
        ("flex.edge_j_mask_dims", 4),
        ("flex.edge_spring_coeff", nflexedge),
        ("flex.edge_operator_coeff", nflexedge),
        ("flex.edge_operator_coo_dims", 4),
    ))
  if legacy_contact_pairs is not None:
    # ContactProgram.empty() guards its flat backing with max(size, 1).
    parts.append((
        "legacy.contact.canonical_rows",
        max(b * legacy_contact_pairs * 5 * (nv + 5), 1)))
    # Detection and solve descriptor tails carry one int32 world mask each.
    parts.append(("legacy.contact.world_mask_dims", 2 * (6 + b)))
  if legacy_joint_rows is not None:
    # JointConstraintProgram creates a two-dimensional [B, max(rows,1)*(nv+5)]
    # tensor, so even its empty-row case has B*(nv+5) physical elements.
    parts.append((
        "legacy.joint.canonical_rows",
        b * max(legacy_joint_rows, 1) * (nv + 5)))
    parts.append((
        "legacy.joint.lambda",
        b * max(legacy_joint_rows, 1)))
    # Assembly and solve descriptors each append a per-world int32 mask.
    parts.append(("legacy.joint.world_mask_dims", 2 * (11 + b)))
  if legacy_total_rows is not None and legacy_total_rows:
    parts.append((
        "simulation.legacy_canonical_rows",
        b * legacy_total_rows * (nv + 5)))
  if legacy_joint_rows is not None:
    # torch.empty_like(rhs) has zero elements when nv == 0.
    parts.append(("simulation.legacy_constraint_rhs", b * nv))

  # FK constants are uploaded as two physical packed arenas to stay within
  # Metal's per-kernel buffer-binding limit. Offsets are int32, so account
  # both the aggregate arena length and every packed field's dummy minimum.
  # Other model constants below belong to their own separately owned stages.
  for name, elements in (
      ("model.fk.static_int", sum(max(n, 1) for n in (
          nb, nj, nj, nj, nb, ng, ng, ns, ns))),
      ("model.fk.static_float", sum((
          3 * max(nb * 3, 1), 3 * max(nb * 4, 1),
          3 * max(nj * 3, 1), 3 * max(nj * 3, 1),
          3 * max(nq, 1), 3 * max(ng * 3, 1), 3 * max(ng * 4, 1),
          3 * max(ns * 3, 1), 3 * max(ns * 4, 1),
          3 * max(nb * 3, 1), 3 * max(nb * 4, 1)))),
      ("model.smooth.body_parentid", max(nb, 1)),
      ("model.smooth.body_rootid", max(nb, 1)),
      ("model.smooth.body_treeid", max(nb, 1)),
      ("model.smooth.dof_treeid", max(nv, 1)),
      ("model.smooth.body_jntadr", max(nb, 1)),
      ("model.smooth.body_jntnum", max(nb, 1)),
      ("model.smooth.body_dofadr", max(nb, 1)),
      ("model.smooth.body_dofnum", max(nb, 1)),
      ("model.smooth.dof_parentid", max(nv, 1)),
      ("model.smooth.dof_bodyid", max(nv, 1)),
      ("model.smooth.jnt_type", max(nj, 1)),
      ("model.smooth.jnt_dofadr", max(nj, 1)),
      ("model.smooth.body_mass", max(nb, 1)),
      ("model.smooth.body_inertia", max(nb * 3, 1)),
      ("model.smooth.dof_armature", max(nv, 1)),
      ("model.smooth.gravity", 3),
      ("model.smooth.component_layout",
       max(int(mass_layout["component_packed"].size), 1)),
      ("model.smooth.tendon_treeid", max(2 * ntendon, 1)),
      ("model.smooth.tendon_treenum", max(ntendon, 1)),
      ("model.smooth.tendon_j_rowadr", max(ntendon, 1)),
      ("model.smooth.tendon_j_rownnz", max(ntendon, 1)),
      ("model.smooth.tendon_j_colind",
       max(int(np.asarray(model.ten_J_colind).size), 1)),
      ("model.smooth.mass_rowadr", max(nv, 1)),
      ("model.smooth.mass_rownnz", max(nv, 1)),
      ("model.smooth.mass_colind",
       max(int(np.asarray(model.M_colind).size), 1)),
      ("model.coupled.eq_rownum", max(neq, 1)),
      ("model.coupled.eq_sol_params", max(neq, 1) * 14),
      ("model.coupled.row_velocity_scale_low", max(coupled_rows, 1)),
      ("model.smooth.effective_tendon_armature", max(ntendon, 1)),
      ("model.sensor.type", max(nsensor, 1)),
      ("model.sensor.datatype", max(nsensor, 1)),
      ("model.sensor.needstage", max(nsensor, 1)),
      ("model.sensor.objtype", max(nsensor, 1)),
      ("model.sensor.objid", max(nsensor, 1)),
      ("model.sensor.dim", max(nsensor, 1)),
      ("model.sensor.adr", max(nsensor, 1)),
      ("model.sensor.cutoff", max(nsensor, 1)),
      ("model.sensor.reftype", max(nsensor, 1)),
      ("model.sensor.refid", max(nsensor, 1)),
      ("model.sensor.joint_meta", max(nj, 1) * 4),
      ("model.sensor.joint_limits", max(nj, 1) * 3),
      ("model.sensor.tendon_limits", max(ntendon, 1) * 9),
      ("model.sensor.mass_body", max(nb, 1) * 5),
      ("model.sensor.body_tree", max(nb, 1) * 2),
      ("model.sensor.site_geom", max(ns, 1) * 4),
      ("model.sensor.econst", 11 + nq + 3 * nj + 17 * ncam),
      ("model.sensor.tendon_qpos_map", max(ntendon, 1) * max(nq, 1)),
      ("model.sensor.tendon_dof_map", max(ntendon, 1) * max(nv, 1)),
      ("model.sensor.actuator_qpos_map", max(nu, 1) * max(nq, 1)),
      ("model.sensor.actuator_dof_map", max(nu, 1) * max(nv, 1)),
      ("model.sensor.rne_body_jnt", max(nb, 1) * 4),
      ("model.sensor.rne_dof_jntid", max(nv, 1)),
      ("model.sensor.rne_jnt_type", max(nj, 1)),
      ("model.sensor.rne_jnt_dofadr", max(nj, 1)),
      ("model.sensor.rne_mass", max(nb, 1)),
      ("model.sensor.rne_body_tree", max(nb, 1) * 2),
      ("model.sensor.rne_inertia", max(nb, 1) * 3),
      ("model.sensor.rne_gravity", 3),
      ("model.sensor.rne_body_weld", max(nb, 1)),
      ("model.sensor.rne_geom_bodyid", max(ng, 1)),
      ("model.sensor.rne_site_bodyid", max(ns, 1)),
      ("model.sensor.rne_eq_meta", max(neq, 1) * 4),
      ("model.sensor.rne_eq_data", max(neq, 1) * 11),
      ("model.sensor.rne_site_lpos", max(ns, 1) * 3),
      ("model.sensor.rne_actuator_transmission", max(nu, 1) * 2),
      ("model.sleep.dof_treeid", max(nv, 1)),
  ):
    parts.append((name, max(elements, 1)))

  # These helpers own independent fixed-shape workspaces. Keep their exact
  # individual backing sizes in the admission envelope; in particular, the
  # component factor uses the compiled sparse component nnz, not nv**2. FK
  # outputs now share one physical arena. Count every high/low/tail view,
  # including the row-major geometry matrix needed by source-faithful CCD.
  fk_outputs = (
      (nb, 3), (nb, 3), (nb, 3), (nb, 4), (nb, 4), (nb, 4),
      (ng, 3), (ng, 3), (ng, 3), (ng, 4), (ng, 4), (ng, 4),
      (ng, 9), (ng, 9), (ng, 9),
      (ns, 3), (ns, 3), (ns, 3), (ns, 4), (ns, 4), (ns, 4),
      (nb, 3), (nb, 3), (nb, 3), (nb, 4), (nb, 4), (nb, 4),
      (nj, 3), (nj, 3),
  )
  fk_pose_output_words = sum(
      max(b * count * width, 1) for count, width in fk_outputs)
  parts.append(("fk.pose_output_arena", fk_pose_output_words))
  parts.extend((
      ("fk.auxiliary", _fk_auxiliary_words(b, nmocap, nb, ntree)),
      ("fk.tree_awake", b * max(ntree, 1)),
      ("fk.tree_awake_cast", b * max(ntree, 1)),
      ("fk.all_world_mask", b),
      ("fk.qpos", max(b * nq, 1)),
      # Header (12), packed-int offsets (9), packed-float offsets (11), and
      # output-arena offsets (29). Every shader-read offset is signed int32.
      ("fk.dims", 61),
  ))

  # Smooth mass/RNE workspaces. `crb` and `local_inertia` are 6x6 spatial
  # matrices per body; omitting these made fixed-heavy models evade bounds.
  derivative_stride = 6 * (3 * nb + nv)
  energy_enabled = bool(int(model.opt.enableflags)
                        & int(mujoco.mjtEnableBit.mjENBL_ENERGY))
  for name, count in (
      ("mass", b * nv * nv if mass_storage == "dense" else 0),
      ("root_com", b * nb * 3),
      ("mass_blocks", b * component_nnz if mass_storage == "block_sparse" else 0),
      ("mass_armature", b * (nv * nv if mass_storage == "dense" else component_nnz)),
      ("mass_zero_blocks", b * component_nnz),
      ("mass_sparse_product", b * nv),
      ("cdof", b * nv * 6), ("crb", b * nb * 36),
      ("local_inertia", b * nb * 36), ("qvel", b * nv),
      ("bias", b * nv), ("bias_low", b * nv),
      ("cvel_low", b * nb * 6), ("cdof_dot_low", b * nv * 6),
      ("cacc_low", b * nb * 6), ("body_force_low", b * nb * 6),
      ("cvel", b * nb * 6),
      ("cdof_dot", b * nv * 6), ("cacc", b * nb * 6),
      ("body_force", b * nb * 6),
      ("bias_derivative", b * nv * nv),
      ("bias_derivative_scratch", b * nv * derivative_stride),
      ("awake_body_ids", b * max(nb, 1)),
      ("awake_parent_ids", b * max(nb, 1)),
      ("awake_dof_ids", b * max(nv, 1)),
      ("awake_body_mask", b * max(nb, 1)),
      ("awake_body_count", b), ("awake_parent_count", b),
      ("awake_dof_count", b),
      ("awake_counts", b * 3),
      ("awake_list_dims", 4),
      ("tree_awake", b * max(ntree, 1)),
      ("pose_status", b),
      ("mass_dims", 4), ("pose_finite_dims", 5),
      ("mass_layout_dims", 5 + b), ("mass_matvec_dims", 4 + b),
      ("energy_velocity_dims", 3 + b),
      ("tendon_armature_dims", 7), ("bias_dims", 4),
      ("disableflags", 1),
  ):
    parts.append((f"smooth.{name}", max(count, 1) if count else 0))
  from mujoco_metal.smooth_metal import position_context_workspace_sizes
  parts.extend((f"smooth.{name}", max(int(count), 1)) for name, count in
               position_context_workspace_sizes(
                   model, b, mass_storage=mass_storage,
                   component_nnz=component_nnz).items())

  # Simulation-owned inputs and per-step device buffers. Effective mass is
  # created only by profiles that request implicit Euler damping; include the
  # conservative maximum because this estimator is profile-independent.
  for name, count in (
      ("rhs", b * nv), ("forward_stage_zero_constraint", b * nv),
      ("rhs_low", b * nv),
      ("dense_pair_solution_low", b * nv if mass_storage == "dense" else 0),
      ("dense_pair_zero_low", b * nv if mass_storage == "dense" else 0),
      ("dense_pair_work", 6 * b * max(nv, 1)
       if mass_storage == "dense" else 0),
      ("dense_pair_dims", 4 if mass_storage == "dense" else 0),
      ("dense_pair_empty_int", 1 if mass_storage == "dense" else 0),
      ("forward_stage_passive_force", b * nv),
      ("forward_stage_actuator_force", b * nv),
      ("solver_fwdinv", b * 2),
      ("applied_force", b * nv),
      ("body_wrench", b * nb * 6), ("control", b * nu),
      ("effective_mass", b * nv * nv), ("success", b),
      ("sensor_plugin_status", b),
      ("combined_status", b), ("next_qpos", b * nq),
      ("next_qvel", b * nv), ("next_qacc", b * nv),
      ("recovery_check_qacc_low", b * nv),
      ("next_time", b), ("next_status", b),
      ("act_dot", b * na), ("act_vel", b * na),
      ("next_act", b * na), ("sensordata", b * nsensordata),
      ("sensor_act_force", b * max(nu, 1)),
      ("sensor_qfrc_act", b * max(nv, 1)),
      ("energy_stage", b * 2 if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_body_mass_const", max(nb - 1, 1) if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_gravity_const", 3 if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_qpos_spring_const", max(nq, 1) if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_position_dims", 12 + b if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_joint_int", 3 * max(nj, 1) if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_joint_float", 3 * max(nj, 1) if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_tendon_int", 4 * max(ntendon, 1) if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_tendon_float", 5 * max(ntendon, 1) if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_flex_int", max(nflexedge, 1) if int(model.opt.enableflags)
       & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
      ("energy_flex_float", 2 * max(nflexedge, 1)
       if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_ENERGY)
       else 0),
      # Fixed-tendon RK4 forwardSkip(POS) cache. The current stage's lengths
      # are copied into a persistent buffer; separate kernel dimensions select
      # the cached-length VEL-only dispatch without mutating ordinary dims.
      ("fixed_tendon.last_length", b * max(ntendon, 1)),
      ("fixed_tendon.dims", 8 + b),
      ("fixed_tendon.velocity_dims", 8 + b),
      ("position_cache.fixed_tendon_length", b * max(ntendon, 1)),
      ("position_cache.spatial_tendons.length", b * max(ntendon, 1)),
      ("position_cache.spatial_tendons.jacobian",
       b * max(ntendon, 1) * max(nv, 1)),
      ("position_cache.actuators.length", b * max(nu, 1)),
      ("position_cache.actuators.moment",
       b * max(nu, 1) * max(nv, 1)),
      ("sleep_qvel", b * nv),
      ("sleep_fk_previous", b * max(ntree, 1)),
      ("sleep_fk_refresh", b * max(ntree, 1)),
      ("component_tendon_jacobian",
       b * max(ntendon, 1) * max(nv, 1)
       if mass_storage == "block_sparse" and has_tendon_armature else 0),
      ("component_damping_derivative", b * nv
       if mass_storage == "block_sparse" else 0),
      ("component_world_status", b if mass_storage == "block_sparse" else 0),
      ("dense_solver_world_mask", b if mass_storage == "dense" else 0),
  ):
    if name.startswith("energy_") and not energy_enabled:
      parts.append((f"simulation.{name}", 0))
    else:
      parts.append((f"simulation.{name}", max(count, 1)))
  if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
    # Immutable reset xpose seeds the first asleep-tree FK comparison; it is
    # model-sized and reused by full and selected-environment resets.
    parts.append(("simulation.sleep_reset_body_pos", max(nb * 3, 1)))
    parts.append(("simulation.sleep_reset_body_quat", max(nb * 4, 1)))

  # The signed Euler LDL helper has a private dense factor/solve workspace;
  # its awake IDs and [B,3] counts are prepared once and passed to the shader
  # as contiguous buffers. These are separate from Smooth's reusable awake
  # lists even though their contents are the all-awake case.
  from mujoco_metal.stepping import _euler_damping_dofs
  if np.any(_euler_damping_dofs(model)):
    for name, count in (
        ("euler_awake.dof_ids", b * max(nv, 1)),
        ("euler_awake.counts", b * 3),
        ("euler_solver.factor", b * nv * nv),
        ("euler_solver.solution", b * nv),
        ("euler_solver.status", b),
        ("euler_solver.world_mask", b),
        ("euler_solver.empty_input", 1),
        ("euler_solver.dims", 3),
        ("euler_solver.awake_work", b * nv),
        ("euler_solver.awake_dims", 3),
        ("euler_solver.awake_flags", 2),
    ):
      parts.append((f"simulation.{name}", max(count, 1)))
  parts.append(("simulation.euler_damping_eligibility", max(nv, 1)))

  if diag_exact_rows is not None:
    from mujoco_metal.constraint_impedance import (
        constraint_impedance_workspace_sizes)
    diag_sizes = constraint_impedance_workspace_sizes(
        b, max(diag_exact_rows, 1), max(nv, 1), diag_exact_cones,
        mass_storage=mass_storage)
    parts.extend((f"constraint_impedance.{name}", int(count))
                 for name, count in diag_sizes.items())
    parts.append(("simulation.diag_exact_status", max(b, 1)))
    if mass_storage == "dense":
      # MetalFactorizedSolve(nv,batch,nrhs=nr) stores each backing separately.
      for name, count in (
          ("dense_factor", b * nv * nv),
          ("dense_pivots", b * nv),
          ("dense_solution", b * nv * max(diag_exact_rows, 1)),
          ("dense_factor_status", b), ("dense_status", b),
          ("dense_empty_input", 1), ("dense_dims", 4)):
        parts.append((f"simulation.diag_exact_{name}", max(count, 1)))

  # These native wrappers own independent model constants and persistent
  # outputs. Count their physical tensors separately from the matching sensor
  # and smooth constants, since each shader receives its own buffer.
  from mujoco_metal.passive import passive_workspace_sizes
  parts.extend((name, max(int(count), 1)) for name, count in
               passive_workspace_sizes(model, b, validate=False).items())
  from mujoco_metal.transmissions import transmission_workspace_sizes
  parts.extend((name, max(int(count), 1)) for name, count in
               transmission_workspace_sizes(model, b, validate=False).items())

  # Sensor-owned buffers including the 128-byte-per-body subtree record,
  # RNE derivatives, and their shared scratch. Optional sensor families still
  # have the common output/runtime allocations below.
  parts.extend((
      ("sensor.stage_mask", 1 + b * max(nsensor, 1) + b),
      ("sensor.output", max(b * nsensordata, 1)),
      ("sensor.output_low", max(b * nsensordata, 1)),
      ("sensor.sensor_meta", max(nsensor, 1) * 10),
      ("sensor.subtree_runtime", 4 + b * nb * 32 + b * max(nsensor, 1)),
      ("sensor.state_dims", 15 + b),
      ("sensor.rne_assemble_dims", 8 + b),
      ("sensor.rne_post_dims", 5 + b),
      ("sensor.rne_acc_dims", 14 + b),
      ("sensor.rne_com_scratch", b * max(nb, 1) * 12),
      ("sensor.rne_cacc_pair", 2 * b * max(nb, 1) * 6),
      ("sensor.rne_qacc_low_zero", b * max(nv, 1)),
      ("sensor.rne_cfrc", b * max(nb, 1) * 6),
      ("sensor.rne_scom", b * max(nb, 1) * 3),
      ("sensor.rne_ext", b * max(nb, 1) * 6),
  ))
  if touch_grid_contact_capacity is not None and int(model.nplugin):
    from mujoco_metal.bundled_plugins import lower_bundled_plugins
    from mujoco_metal.bundled_touch_grid import touch_grid_workspace_sizes
    touch_sizes = touch_grid_workspace_sizes(
        model, b, touch_grid_contact_capacity,
        bundled_plugins=lower_bundled_plugins(model),
        pair_capacity=touch_grid_pair_capacity)
    parts.extend((f"sensor.touch_grid.{name}", max(int(count), 1))
                 for name, count in touch_sizes.items())
  if bundled_cable_segments:
    n = bundled_cable_segments
    parts.extend((f"plugin.cable.{name}", max(int(count), 1)) for name, count in (
        ("body_ids", n), ("previous", n), ("following", n), ("qadr", n),
        ("stiffness", 4*n), ("omega0", 3*n), ("local_quat", 4*n),
        ("body_parent", nb), ("dof_body", nv), ("dims", 5 + b), ("flags", 1)))
  # Scheduler and sensor-policy tensors are allocated after the FK, smooth,
  # simulation, and sensor workspaces. Keep their admission checks in that
  # construction order so an earlier first-failing backing is reported first.
  from mujoco_metal.sleep_schedule import sleep_scheduler_workspace_sizes
  from mujoco_metal.active_contact_links import equality_tree_links
  eq_tree_links, _ = equality_tree_links(model)
  pair_capacity = int(model.npair) if npairs is None else int(npairs)
  link_capacity = max(
      pair_capacity + int(eq_tree_links.shape[0]) + nflexcontactlinks, 1)
  parts.extend((name, max(int(count), 1)) for name, count in
               sleep_scheduler_workspace_sizes(
                   model, b, link_capacity, validate=False).items())
  from mujoco_metal.sensor_sleep import sensor_sleep_workspace_sizes
  parts.extend((name, max(int(count), 1)) for name, count in
               sensor_sleep_workspace_sizes(model, b, validate=False).items())
  if rhs_capacity is not None:
    rhs_capacity = _integer_count("rhs_capacity", rhs_capacity, minimum=1)
    from mujoco_metal.component_solve import component_solver_workspace_sizes
    # ComponentMassSolver guards its three independent static map arrays
    # when empty. This differs from CC's unguarded operator-layout payload.
    packed_size = (component_count + 1 + max(nv, 1)
                   + max(component_count, 1))
    component_sizes = component_solver_workspace_sizes(
        batch=b, nv=nv, ncomponent=component_count, nnz=component_nnz,
        rhs_capacity=rhs_capacity, layout_size=packed_size)
    parts.extend((name, max(int(count), 1))
                 for name, count in component_sizes.items())
    # Simulation copies the solver's strided first-RHS view before the next
    # coupled solve can reuse its backing. This is a separately owned [B,nv]
    # result tensor; it is empty when nv==0 and needs no one-element guard.
    parts.append(("simulation.component_solution_vector", b * nv))
    parts.append(("simulation.component_solution_low_vector", b * nv))
    # Public smooth acceleration uses a dedicated input backing so it cannot
    # alias the coupled solver's optional row RHS or retained solution views.
    parts.append(("simulation.component_solve_rhs",
                  b * rhs_capacity * nv))
  if nu:
    parts.append(("actuator.plugin_qfrc_dims", 3 + b))
  # One reusable spatial-force query workspace for each built-in FORCE
  # implementation (Magnetic and SiteFeedback). Their MSL row programs own
  # independent query metadata and output scratch; the Magnetic instance also
  # owns the static body map. Counts come from immutable registry descriptors
  # before DeviceState or plugin buffers are allocated.
  magnetic_force_plugins = _integer_count(
      "magnetic_force_plugins", magnetic_force_plugins)
  site_feedback_plugins = _integer_count(
      "site_feedback_plugins", site_feedback_plugins)
  spatial_instances = magnetic_force_plugins + site_feedback_plugins
  spatial_nv = max(nv, 1)
  parts.extend((
      ("spatial_plugin_force.chain", spatial_instances * nb * nv),
      ("spatial_plugin_force.roots", spatial_instances * nb),
      ("spatial_plugin_force.mask", spatial_instances * b),
      ("spatial_plugin_force.dimensions", spatial_instances * 6),
      ("spatial_plugin_force.output", spatial_instances * b * spatial_nv),
      ("spatial_plugin_force.query_parameters", spatial_instances * 3),
      ("spatial_plugin_force.magnetic_field",
       magnetic_force_plugins * 3),
      ("spatial_plugin_force.site_feedback_parameters",
       site_feedback_plugins * 14),
      ("spatial_plugin_force.magnetic_body_ids",
       magnetic_force_plugins * max(nb - 1, 1)),
  ))
  from mujoco_metal.spatial_tendons import SpatialTendonModel
  if SpatialTendonModel(model).has_spatial:
    parts.extend((
        ("spatial_tendon.actuator_state_rows", 3 * max(nu, 1)),
        ("spatial_tendon.actuator_state_dims", 5 + b),
    ))
  # Registered MuJoCo USER callbacks share the owned ACT-stage gain/bias
  # planes. Allocate both when any USER role exists: an ACT callback may be
  # registered for dynamics alone while the other planes still provide the
  # pinned unregistered defaults to the force kernel.
  has_user_actuator = bool(nu and (
      np.any(np.asarray(model.actuator_dyntype)
             == int(mujoco.mjtDyn.mjDYN_USER))
      or np.any(np.asarray(model.actuator_gaintype)
                == int(mujoco.mjtGain.mjGAIN_USER))
      or np.any(np.asarray(model.actuator_biastype)
                == int(mujoco.mjtBias.mjBIAS_USER))))
  if has_user_actuator:
    parts.extend((
        ("actuator.user_gain", b * max(nu, 1)),
        ("actuator.user_bias", b * max(nu, 1)),
    ))
  if native_actuator_user_workspace_bytes:
    parts.append(("actuator.user_plugin_workspace_elements",
                  (native_actuator_user_workspace_bytes + 3) // 4))
  names = [name for name, _ in parts]
  if len(names) != len(set(names)):
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    raise RuntimeError(
        "runtime capacity inventory contains duplicate backing names: "
        + ", ".join(duplicates))
  return tuple(parts)


def validate_runtime_buffers(model, batch_size, *, npairs=None,
                             rhs_capacity=None, coupled_rows=0,
                             mass_storage="dense",
                             legacy_contact_pairs=None,
                             legacy_joint_rows=None,
                             legacy_total_rows=None,
                             effective_edge_count=None,
                             effective_gmres_dimension=None,
                             effective_gmres_iterations=None,
                             diag_exact_rows=None, diag_exact_cones=None,
                             touch_grid_contact_capacity=None,
                             touch_grid_pair_capacity=None,
                             bundled_cable_segments=None,
                             magnetic_force_plugins=0,
                             site_feedback_plugins=0,
                             native_actuator_user_workspace_bytes=0):
  """Validate profile-independent runtime backing buffers before allocation."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  if (isinstance(batch_size, (bool, np.bool_)) or
      not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
    raise ValueError("batch_size must be a positive integer")
  batch = int(batch_size)
  if npairs is not None:
    npairs = _integer_count("npairs", npairs)
  if rhs_capacity is not None:
    rhs_capacity = _integer_count("rhs_capacity", rhs_capacity, minimum=1)
  if mass_storage not in ("dense", "block_sparse"):
    raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
  for name, value in (
      ("legacy_contact_pairs", legacy_contact_pairs),
      ("legacy_joint_rows", legacy_joint_rows),
      ("legacy_total_rows", legacy_total_rows),
      ("effective_edge_count", effective_edge_count),
      ("effective_gmres_dimension", effective_gmres_dimension),
      ("effective_gmres_iterations", effective_gmres_iterations),
      ("diag_exact_rows", diag_exact_rows),
      ("diag_exact_cones", diag_exact_cones),
      ("touch_grid_contact_capacity", touch_grid_contact_capacity),
      ("touch_grid_pair_capacity", touch_grid_pair_capacity),
      ("bundled_cable_segments", bundled_cable_segments),
      ("magnetic_force_plugins", magnetic_force_plugins),
      ("site_feedback_plugins", site_feedback_plugins),
  ):
    if value is not None:
      _integer_count(name, value,
                     minimum=1 if (name.startswith("effective_gmres_")
                                   or name == "diag_exact_cones") else 0)
  if (diag_exact_rows is None) != (diag_exact_cones is None):
    raise ValueError("diag_exact_rows and diag_exact_cones must be supplied together")
  for name, value in (
      ("nq", int(model.nq)), ("nv", int(model.nv)),
      ("nbody", int(model.nbody)), ("njnt", int(model.njnt)),
      ("ngeom", int(model.ngeom)), ("nsite", int(model.nsite)),
      ("nsensor", int(model.nsensor)),
      ("nsensordata", int(model.nsensordata)),
      ("nu", int(model.nu)), ("na", int(model.na)),
      ("neq", int(model.neq)), ("ntendon", int(model.ntendon)),
      ("nmocap", int(model.nmocap)), ("ncam", int(model.ncam)),
  ):
    if value > SIGNED_I32_ELEMENT_LIMIT:
      raise CapacityOverflow(
          f"{name} ({value}) exceeds signed 32-bit dimension limit "
          f"{SIGNED_I32_ELEMENT_LIMIT}")
  if (effective_gmres_dimension is not None
      and effective_gmres_dimension > max(int(model.nv), 1)):
    raise ValueError("effective_gmres_dimension must not exceed max(nv,1)")
  if (effective_gmres_iterations is not None
      and effective_gmres_iterations > SIGNED_I32_ELEMENT_LIMIT):
    raise CapacityOverflow("effective_gmres_iterations exceeds signed int32 range")
  for name, elements in _runtime_buffer_sizes(
      model, batch, npairs=npairs, rhs_capacity=rhs_capacity,
      coupled_rows=coupled_rows,
      mass_storage=mass_storage,
      legacy_contact_pairs=legacy_contact_pairs,
      legacy_joint_rows=legacy_joint_rows,
      legacy_total_rows=legacy_total_rows,
      effective_edge_count=effective_edge_count,
      effective_gmres_dimension=effective_gmres_dimension,
      effective_gmres_iterations=effective_gmres_iterations,
      diag_exact_rows=diag_exact_rows,
      diag_exact_cones=diag_exact_cones,
      touch_grid_contact_capacity=touch_grid_contact_capacity,
      touch_grid_pair_capacity=touch_grid_pair_capacity,
      bundled_cable_segments=bundled_cable_segments,
      magnetic_force_plugins=magnetic_force_plugins,
      site_feedback_plugins=site_feedback_plugins,
      native_actuator_user_workspace_bytes=native_actuator_user_workspace_bytes):
    _check_i32_elements(name, elements)
  return batch


def selected_jacobian_kind(model: mujoco.MjModel, nv=None):
  """Return the pinned MuJoCo 3.10 Jacobian storage selection.

  AUTO selects dense storage through 60 DOFs and sparse storage above 60.
  This decision is independent of solver-row path selection.
  """
  jac = int(model.opt.jacobian)
  if jac == int(mujoco.mjtJacobian.mjJAC_DENSE):
    return "dense"
  if jac == int(mujoco.mjtJacobian.mjJAC_SPARSE):
    return "sparse"
  if jac != int(mujoco.mjtJacobian.mjJAC_AUTO):
    raise ValueError(f"unknown MuJoCo Jacobian mode: {jac}")
  dofs = int(model.nv if nv is None else nv)
  return "dense" if dofs <= AUTO_JACOBIAN_DENSE_NV else "sparse"


def estimate_workspace(nv, npairs, nslots, nr, nr_joint, neq, batch,
                       solver_type=int(mujoco.mjtSolver.mjSOL_PGS), *, ntree=0,
                       nbody=0, nsite=0, mass_storage="dense",
                       component_nnz=0, ncomponent=0,
                       component_configured=False,
                       jacobian_kind="dense", jacobian_nnz=None):
  """Budget solver buffers and the fixed-capacity milestone-017 maps.

  Returns ``(parts, total)`` with exact sizes for current coupled solver
  allocations plus the reserved pair/slot/row maps and scan scratch required
  by 017. Counts use ``max(..., 1)`` empty guards. The compact-map entries
  are reservations until the packing path binds them into
  ``prepare_workspace``; they must remain here so the allocation budget
  continues to cover the integrated implementation.
  """
  b = _integer_count("batch", batch, minimum=1)
  nv = _integer_count("nv", nv)
  npairs = _integer_count("npairs", npairs)
  nc = _integer_count("nslots", nslots)
  nr = _integer_count("nr", nr)
  nr_joint = _integer_count("nr_joint", nr_joint)
  neq = _integer_count("neq", neq)
  if (isinstance(solver_type, (bool, np.bool_)) or
      not isinstance(solver_type, (int, np.integer, mujoco.mjtSolver))):
    raise ValueError("solver_type must be an integer")
  if mass_storage not in ("dense", "block_sparse"):
    raise ValueError("mass_storage must be 'dense' or 'block_sparse'")
  if jacobian_kind not in ("dense", "sparse"):
    raise ValueError("jacobian_kind must be 'dense' or 'sparse'")
  if jacobian_kind == "sparse":
    if jacobian_nnz is None:
      raise ValueError("sparse Jacobian capacity requires compiled nnz")
    jacobian_nnz = _integer_count("jacobian_nnz", jacobian_nnz)
    jacobian_words = (12 + nr + 1 + 2 * jacobian_nnz)
    if jacobian_words > SIGNED_I32_ELEMENT_LIMIT:
      raise CapacityOverflow("packed Jacobian record exceeds signed int32 range")
  else:
    jacobian_nnz = nv * nr
    jacobian_words = 12 + nv * nr
  v = max(nv, 1)
  primal_scratch = primal_scratch_floats(
      nv, nr, solver_type, ntree, nbody, mass_storage)
  # Both ``empty`` and torch.zeros in prepare_workspace allocate one element
  # for an otherwise empty buffer. Keep that guard explicit so the estimate
  # covers the zero-contact/zero-row cases too.
  def guarded(n):
    return max(int(n), 1)
  contact_jacobian_words = (b * nc * 6 * v
                            if jacobian_kind == "dense" else 1)
  blocks = max((int(nr) + 255) // 256, 1)
  parts = (
      ("contact_row_data", _bytes(guarded(b * nc * 6 * 6))),
      ("contact_frame", _bytes(guarded(b * nc * 12))),
      ("contact_jacobian", _bytes(guarded(contact_jacobian_words))),
      ("pair_mask", _bytes(guarded(b * max(int(npairs), 1)))),
      ("workspace_J", _bytes(guarded(b * jacobian_words))),
      ("position_context_zero", _bytes(guarded(b * int(nr)))),
      ("position_context", _bytes(guarded(b * int(nr) * 5))),
      ("position_assembly_context",
       _bytes(guarded(b * int(nr) * 5))),
      ("position_surface_velocity", _bytes(guarded(b * int(nr)))),
      ("position_extra_aref", _bytes(guarded(b * int(nr)))),
      ("position_qvel", _bytes(guarded(b * int(nv)))),
      ("position_current_qvel", _bytes(guarded(b * int(nv)))),
      # Persistent copies owned by the RK4 position-stage transaction. These
      # are independent of live assembly arrays because ordinary forwards
      # overwrite those arrays before the cached velocity stage.
      ("position_cache_J", _bytes(guarded(b * jacobian_words))),
      ("position_cache_rows", _bytes(guarded(b * 7 * int(nr)))),
      ("position_current_aref_low", _bytes(guarded(b * int(nr)))),
      ("position_cache_aref_low", _bytes(guarded(b * int(nr)))),
      ("position_cache_qvel", _bytes(guarded(b * int(nv)))),
      ("position_refresh_old_extra_aref", _bytes(guarded(b * int(nr)))),
      ("position_cache_impedance", _bytes(guarded(b * int(nr)))),
      ("position_cache_contact_data", _bytes(guarded(b * nc * 36))),
      ("position_cache_contact_frame", _bytes(guarded(b * nc * 12))),
      ("position_cache_contact_jacobian",
       _bytes(guarded(contact_jacobian_words))),
      ("position_cache_cvel", _bytes(guarded(b * int(nbody) * 6))),
      ("position_cache_cdof", _bytes(guarded(b * int(nv) * 6))),
      ("position_cache_cdof_dot", _bytes(guarded(b * int(nv) * 6))),
      ("position_current_cvel", _bytes(guarded(b * int(nbody) * 6))),
      ("position_current_cdof", _bytes(guarded(b * int(nv) * 6))),
      ("position_current_cdof_dot", _bytes(guarded(b * int(nv) * 6))),
      ("position_current_body_pos",
       _bytes(guarded(b * int(nbody) * 3))),
      ("position_current_body_pos_low",
       _bytes(guarded(b * int(nbody) * 3))),
      ("position_current_body_pos_tail",
       _bytes(guarded(b * int(nbody) * 3))),
      ("position_current_body_quat",
       _bytes(guarded(b * int(nbody) * 4))),
      ("position_current_root_com",
       _bytes(guarded(b * int(nbody) * 3))),
      ("position_current_site_pos",
       _bytes(guarded(b * int(nsite) * 3))),
      ("position_current_site_quat",
       _bytes(guarded(b * int(nsite) * 4))),
      ("equality_pose_com", _bytes(guarded(2 * b * int(nbody) * 3))),
      ("equality_motion", _bytes(guarded(b * int(nv) * 12))),
      ("position_cache_body_pos", _bytes(guarded(b * int(nbody) * 3))),
      ("position_cache_body_pos_low", _bytes(guarded(b * int(nbody) * 3))),
      ("position_cache_body_pos_tail", _bytes(guarded(b * int(nbody) * 3))),
      ("position_cache_body_quat", _bytes(guarded(b * int(nbody) * 4))),
      ("position_cache_site_pos", _bytes(guarded(b * int(nsite) * 3))),
      ("position_cache_site_quat", _bytes(guarded(b * int(nsite) * 4))),
      ("position_cache_root_com", _bytes(guarded(b * int(nbody) * 3))),
      ("position_refresh_extra_aref",
       _bytes(guarded(b * int(nr)))),
      ("position_refresh_copy_dims", _bytes_i32(6 + b)),
      ("position_cache_slot_packed", _bytes_i32(guarded(b * nc))),
      ("position_cache_slot_reverse", _bytes_i32(guarded(b * nc))),
      ("position_cache_slot_count", _bytes_i32(b)),
      ("position_cache_slot_overflow", _bytes_i32(b)),
      ("position_cache_pair_packed", _bytes_i32(guarded(b * int(npairs)))),
      ("position_cache_pair_reverse", _bytes_i32(guarded(b * int(npairs)))),
      ("position_cache_pair_count", _bytes_i32(b)),
      ("position_cache_pair_overflow", _bytes_i32(b)),
      # PGS may split canonical row assembly from sparse component-wise
      # application of M^-1. The RHS backing holds all logical rows plus the
      # smooth-force row; it is allocated only for the PGS solver path.
      ("component_mass_rhs", _bytes(
          guarded(b * (int(nr) + 1) * int(nv))
          if (component_configured
              and int(solver_type) == int(mujoco.mjtSolver.mjSOL_PGS)
              and int(nv) > 0) else 0)),
      ("component_smooth_acceleration_pair", _bytes(
          guarded(2 * b * int(nv))
          if ((mass_storage == "dense"
               or int(solver_type) == int(mujoco.mjtSolver.mjSOL_PGS))
              and int(nv) > 0) else 0)),
      # Reusable acceleration-space primal-solver scratch shares the debug
      # binding to preserve Metal's 31-buffer ABI.
      ("workspace_debug", _bytes(guarded(
          b * (int(nr) * int(nr) + 7 * int(nr) + primal_scratch
               + (int(component_nnz) + 2 * int(ncomponent) + 1
                  + 3 * nv
                  if mass_storage == "block_sparse" and component_configured
                  else 0)
               + 8 * max(int(nv), 1) + 3 * max(int(nr), 1))))),
      # One constructor-owned signed-int32 tensor contains the 22 scalar
      # dimensions, six island offsets, dof-tree map (guarded at nv=0), ten
      # canonical row words per row, and 20 appended solver/component
      # offsets. Count the complete backing rather than only its tail.
      ("solver_dims", _bytes_i32(
          22 + 6 + max(nv, 1) + 10 * int(nr) + 20)),
      ("contact.body_dims", _bytes_i32(6 + b)),
      ("out_force", _bytes(guarded(b * v))),
      # Keep an independently addressable low word for qacc beside the
      # unchanged high-word prefix consumed by existing callers.
      ("out_acc", _bytes(guarded(2 * b * int(nv)))),
      ("out_status", _bytes_i32(b)),
      ("out_diagnostics", _bytes(guarded(b * 10))),
      ("out_contact_force", _bytes(guarded(b * nc * 11))),
      ("out_joint_force", _bytes(guarded(b * max(int(nr_joint), 1)))),
      ("eq_active", _bytes_i32(b * max(int(neq), 1))),
      ("eq_active_default", _bytes_i32(b * max(int(neq), 1))),
      # Compaction maps preserve canonical logical identity while reducing
      # pair, slot, and row execution spans. The conservative budget assumes
      # every candidate can be active and accounts for independent scan
      # scratch for all three domains.
      ("packed_to_pair", _bytes_i32(b * int(npairs))),
      ("pair_to_packed", _bytes_i32(b * int(npairs))),
      ("packed_to_slot", _bytes_i32(b * nc)),
      ("slot_to_packed", _bytes_i32(b * nc)),
      ("packed_to_logical_row", _bytes_i32(b * int(nr))),
      ("logical_to_packed_row", _bytes_i32(b * int(nr))),
      ("compaction_counts_overflow", _bytes_i32(b * 4)),
      ("pair_slot_compaction_mask_dims",
       _bytes_i32((4 + b) * (int(npairs > 0) + int(nc > 0)))),
      ("active_contact_link_mask_dims", _bytes_i32(7 + b)),
      ("pair_contact_world_mask_suffix", _bytes_i32(1 + b)),
      ("pair_scan_prefix", _bytes_i32(b * guarded(int(npairs)))),
      ("pair_scan_blocks", _bytes_i32(b * max((int(npairs) + 255) // 256, 1) * 2)),
      ("slot_scan_prefix", _bytes_i32(b * guarded(nc))),
      ("slot_scan_blocks", _bytes_i32(b * max((nc + 255) // 256, 1) * 2)),
      ("row_scan_prefix", _bytes_i32(b * guarded(int(nr)))),
      ("row_scan_blocks", _bytes_i32(b * blocks * 2)),
  )
  for name, nbytes in parts:
    _check_i32_elements(name, nbytes // 4)
  # `force_outputs` is one combined backing allocation for the contact and
  # joint force views; its sum can overflow even when each view fits.
  contact_force = max(b * nc * 11, 1)
  joint_force = b * max(int(nr_joint), 1)
  _check_i32_elements("force_outputs", contact_force + joint_force)
  _check_i32_elements("pair_contact_offsets_dims", int(npairs) + 11)
  return parts, sum(value for _, value in parts)


def estimate_capacity(model, batch_size, npairs, nslots, nr, *, neq=0,
                      nr_joint=0, mass_storage="dense",
                      legacy_contact_pairs=None, legacy_joint_rows=None,
                      legacy_total_rows=None,
                      effective_edge_count=None,
                      effective_gmres_dimension=None,
                      effective_gmres_iterations=None,
                      diag_exact_rows=None, diag_exact_cones=None,
                      touch_grid_contact_capacity=None,
                      touch_grid_pair_capacity=None,
                      bundled_cable_segments=None,
                      magnetic_force_plugins=0,
                      site_feedback_plugins=0,
                      native_actuator_user_workspace_bytes=0,
                      jacobian_kind=None, jacobian_nnz=None) -> CapacityEstimate:
  """Build a deterministic estimate from lowering counts.

  `npairs`/`nslots`/`nr` come from the coupled lowering (candidate counts,
  before broadphase pruning); `neq`/`nr_joint` size the equality/joint
  outputs. Memory includes the solver workspace, compact-map reservation,
  and profile-independent runtime buffer envelope at the requested batch
  size. The envelope includes conditional/lazy derivative and sensor buffers,
  so it is conservative when the selected profile does not allocate them.
  Deterministic: pure function of the inputs, no device state.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  batch = _integer_count("batch_size", batch_size, minimum=1)
  nr = _integer_count("nr", nr)
  npairs = _integer_count("npairs", npairs)
  nslots = _integer_count("nslots", nslots)
  if touch_grid_contact_capacity is None and int(model.nplugin):
    # The coupled lowering's nslots is the fixed contact capacity consumed by
    # touch-grid, including SDF and flex candidate slots.
    touch_grid_contact_capacity = nslots
  if touch_grid_pair_capacity is None and int(model.nplugin):
    # The runtime touch-grid consumer receives the coupled solver's compiled
    # candidate-pair capacity, which may include generated pairs absent from
    # model.npair. Callers with the lowered descriptor should pass it exactly.
    touch_grid_pair_capacity = npairs
  validate_runtime_buffers(
      model, batch, npairs=npairs, rhs_capacity=max(nr + 1, 1),
      coupled_rows=nr,
      mass_storage=mass_storage,
      legacy_contact_pairs=legacy_contact_pairs,
      legacy_joint_rows=legacy_joint_rows,
      legacy_total_rows=legacy_total_rows,
      effective_edge_count=effective_edge_count,
      effective_gmres_dimension=effective_gmres_dimension,
      effective_gmres_iterations=effective_gmres_iterations,
      diag_exact_rows=diag_exact_rows,
      diag_exact_cones=diag_exact_cones,
      touch_grid_contact_capacity=touch_grid_contact_capacity,
      touch_grid_pair_capacity=touch_grid_pair_capacity,
      bundled_cable_segments=bundled_cable_segments,
      magnetic_force_plugins=magnetic_force_plugins,
      site_feedback_plugins=site_feedback_plugins,
      native_actuator_user_workspace_bytes=native_actuator_user_workspace_bytes)
  nv = int(model.nv)
  if jacobian_kind is None:
    jacobian_kind = selected_jacobian_kind(model)
  if jacobian_kind == "sparse" and jacobian_nnz is None:
    # Standalone estimates without a lowered row pattern remain safe by
    # charging dense support. The coupled lowerer/constructor pass exact nnz.
    jacobian_nnz = int(nv) * int(nr)
  nbody, ngeom = int(model.nbody), int(model.ngeom)
  nr_joint = _integer_count("nr_joint", nr_joint)
  neq = _integer_count("neq", neq)
  component_nnz = component_count = 0
  if mass_storage == "block_sparse":
    from mujoco_metal.model import actuator_tendon_inheritance
    from mujoco_metal.mass_layout import compile_tree_mass_layout
    inherited, _, _ = actuator_tendon_inheritance(model)
    layout = compile_tree_mass_layout(
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
    component_nnz = int(layout["nnz"])
    component_count = int(layout["ncomponent"])
  parts, total = estimate_workspace(
      nv, npairs, nslots, nr, nr_joint, neq, batch,
      solver_type=int(model.opt.solver), ntree=int(model.ntree),
      nbody=int(model.nbody), nsite=int(model.nsite),
      mass_storage=mass_storage, component_nnz=component_nnz,
      ncomponent=component_count,
      component_configured=(mass_storage == "block_sparse"),
      jacobian_kind=jacobian_kind, jacobian_nnz=jacobian_nnz)
  # Include profile-independent and conditional runtime buffers in the memory
  # envelope. Address guards already ran once through the same per-buffer
  # inventory in validate_runtime_buffers().
  runtime_parts = []
  for name, elements in _runtime_buffer_sizes(
      model, batch, npairs=npairs, rhs_capacity=max(nr + 1, 1),
      mass_storage=mass_storage,
      legacy_contact_pairs=legacy_contact_pairs,
      legacy_joint_rows=legacy_joint_rows,
      legacy_total_rows=legacy_total_rows,
      effective_edge_count=effective_edge_count,
      effective_gmres_dimension=effective_gmres_dimension,
      effective_gmres_iterations=effective_gmres_iterations,
      diag_exact_rows=diag_exact_rows,
      diag_exact_cones=diag_exact_cones,
      touch_grid_contact_capacity=touch_grid_contact_capacity,
      touch_grid_pair_capacity=touch_grid_pair_capacity,
      bundled_cable_segments=bundled_cable_segments,
      magnetic_force_plugins=magnetic_force_plugins,
      site_feedback_plugins=site_feedback_plugins,
      native_actuator_user_workspace_bytes=native_actuator_user_workspace_bytes):
    runtime_parts.append((name, _bytes(elements)))
  parts = parts + tuple(runtime_parts)
  total += sum(size for _, size in runtime_parts)
  return CapacityEstimate(
      nv=nv, nbody=nbody, ngeom=ngeom, npairs=int(npairs),
      nslots=int(nslots), nr=nr, batch=batch,
      dense_path=(nr <= DENSE_ROW_THRESHOLD and nv <= 32),
      mass_storage=mass_storage,
      jacobian_kind=selected_jacobian_kind(model),
      jacobian_auto_threshold=AUTO_JACOBIAN_DENSE_NV,
      memory_bytes=total, memory_breakdown=parts)


def check_capacity(estimate, limits=None) -> CapacityEstimate:
  """Enforce budgets; raise CapacityOverflow (a ValueError) on excess.

  Keeps the historical error text so existing boundary tests keep reading
  the same diagnostics, now with explicit requested-vs-allowed detail.
  """
  limits = limits or CapacityLimits()
  for name, value in (("gmres_krylov_dimension", limits.gmres_krylov_dimension),
                      ("gmres_max_iterations", limits.gmres_max_iterations)):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) or value < 1):
      raise CapacityOverflow(f"{name} must be a positive integer when specified")
  if (limits.gmres_krylov_dimension is not None
      and limits.gmres_krylov_dimension > max(estimate.nv, 1)):
    raise CapacityOverflow(
        "gmres_krylov_dimension must not exceed max(nv,1); use restart cycles")
  if (limits.gmres_max_iterations is not None
      and limits.gmres_max_iterations > SIGNED_I32_ELEMENT_LIMIT):
    raise CapacityOverflow("gmres_max_iterations exceeds signed int32 range")
  if estimate.batch > limits.max_batch:
    raise CapacityOverflow(
        f"batch size ({estimate.batch}) exceeds capacity {limits.max_batch}")
  max_nv = int(limits.max_nv)
  if estimate.nv > max_nv:
    raise CapacityOverflow(
        f"coupled constraint stage bounds nv to {max_nv}; found {estimate.nv}")
  if estimate.npairs > limits.max_pairs:
    raise CapacityOverflow(
        f"candidate contact pairs ({estimate.npairs}) exceeds capacity {limits.max_pairs}")
  if estimate.nslots > limits.max_slots:
    raise CapacityOverflow(
        f"total candidate contact slots ({estimate.nslots}) exceeds capacity {limits.max_slots}")
  max_rows = int(limits.max_rows)
  if estimate.nr > max_rows:
    raise CapacityOverflow(
        f"total candidate constraint rows ({estimate.nr}) exceeds capacity {max_rows}")
  if estimate.memory_bytes > limits.memory_budget_bytes:
    raise CapacityOverflow(
        f"estimated device memory ({estimate.memory_bytes} bytes) exceeds budget "
        f"{limits.memory_budget_bytes} bytes")
  return estimate
