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

# Baseline ceilings carried over from milestones 004-016. Pair/slot
# ceilings were raised by the F1 scalable path (broadphase-fed pruning:
# inactive pairs skip narrowphase work, so cost follows active slots and
# rows, both still hard-capped below) — never a bare constant bump. nv/row
# ceilings still need tiled layouts and the block solver (follow-up).
BASE_NVIDIA_NV = 32
BASE_MAX_PAIRS = 32
BASE_MAX_SLOTS = 48
BASE_MAX_ROWS = 96
# The current block-solver shader uses fixed per-thread arrays for at most 64
# DOFs and 256 rows. User budgets may tighten these limits, but cannot raise
# them past the compiled ABI.
NATIVE_MAX_NV = 64
NATIVE_MAX_ROWS = 256
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


def primal_scratch_floats(nv, nr, solver_type):
  """Exact per-world scratch for the currently implemented primal solver.

  PGS needs the supplied qacc warmstart. CG needs twelve DOF vectors, five
  row vectors, and qacc warmstart. Newton adds one dense primal
  Hessian/factor buffer.
  """
  v, r = max(int(nv), 1), max(int(nr), 1)
  solver = int(solver_type)
  if solver == int(mujoco.mjtSolver.mjSOL_PGS):
    return v
  if solver == int(mujoco.mjtSolver.mjSOL_CG):
    return 13 * v + 5 * r
  if solver == int(mujoco.mjtSolver.mjSOL_NEWTON):
    return 13 * v + 5 * r + v * v
  return 1


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
  jacobian_kind: str
  jacobian_auto_threshold: int
  memory_bytes: int
  memory_breakdown: tuple


def _bytes(n):
  return int(n) * 4


def _bytes_i32(n):
  return int(n) * 4


def _check_i32_elements(name, elements):
  """Reject any flat device allocation that kernels address with int32."""
  elements = int(elements)
  if elements > SIGNED_I32_ELEMENT_LIMIT:
    raise CapacityOverflow(
        f"{name} element count ({elements}) exceeds signed 32-bit addressing "
        f"limit {SIGNED_I32_ELEMENT_LIMIT}")


def _runtime_buffer_sizes(model, batch):
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

  # DeviceState fields. Zero-sized tensors use a one-element backing buffer
  # where torch.empty/zeros is called directly; optional tensors are absent.
  parts = [
      ("state.qpos", max(b * nq, 1)),
      ("state.qvel", max(b * nv, 1)),
      ("state.qacc", max(b * nv, 1)),
      ("state.qacc_warmstart", max(b * nv, 1)),
      ("state.time", b), ("state.status", b),
      ("state.eq_active", b * neq if neq else 0),
      ("state.mocap_pos", b * nmocap * 3 if nmocap else 0),
      ("state.mocap_quat", b * nmocap * 4 if nmocap else 0),
      ("state.act", b * na if na else 0),
      ("state.history", b * nhistory),
      ("state.userdata", b * nuserdata),
      ("state.plugin_state", b * npluginstate),
  ]

  # Immutable per-model device constants are independently addressed by
  # shaders even when a large batch is not requested. Keep each physical
  # tensor separate here; checking the sum of unrelated arrays can reject a
  # model whose individual shader-addressed buffers are all representable.
  for name, elements in (
      ("model.fk.body_pos", max(nb * 3, 1)),
      ("model.fk.body_quat", max(nb * 4, 1)),
      ("model.fk.body_ipos", max(nb * 3, 1)),
      ("model.fk.body_iquat", max(nb * 4, 1)),
      ("model.fk.body_parentid", max(nb, 1)),
      ("model.fk.body_treeid", max(nb, 1)),
      ("model.fk.body_mocapid", max(nb, 1)),
      ("model.fk.joint_type", max(nj, 1)),
      ("model.fk.joint_qposadr", max(nj, 1)),
      ("model.fk.joint_bodyid", max(nj, 1)),
      ("model.fk.joint_pos", max(nj * 3, 1)),
      ("model.fk.joint_axis", max(nj * 3, 1)),
      ("model.fk.qpos0", max(nq, 1)),
      ("model.fk.geom_bodyid", max(ng, 1)),
      ("model.fk.geom_type", max(ng, 1)),
      ("model.fk.geom_size", max(ng * 3, 1)),
      ("model.fk.geom_pos", max(ng * 3, 1)),
      ("model.fk.geom_quat", max(ng * 4, 1)),
      ("model.fk.site_bodyid", max(ns, 1)),
      ("model.fk.site_pos", max(ns * 3, 1)),
      ("model.fk.site_quat", max(ns * 4, 1)),
      ("model.smooth.body_parentid", max(nb, 1)),
      ("model.smooth.body_rootid", max(nb, 1)),
      ("model.smooth.body_treeid", max(nb, 1)),
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

  # FK output backing arrays and its mixed per-world/per-model auxiliary
  # layout: [mocap values][body tree ids][world x tree awake][cache valid].
  for name, count, width in (
      ("body_pos", nb, 3), ("body_quat", nb, 4),
      ("geom_pos", ng, 3), ("geom_quat", ng, 4),
      ("site_pos", ns, 3), ("site_quat", ns, 4),
      ("inertial_pos", nb, 3), ("inertial_quat", nb, 4),
      ("joint_anchor", nj, 3), ("joint_axis", nj, 3),
  ):
    parts.append((f"fk.{name}", max(b * count * width, 1)))
  parts.extend((
      ("fk.auxiliary", max((b * nmocap * 7 if nmocap else 1)
                            + nb + b * max(ntree, 1) + b, 1)),
      ("fk.tree_awake", b * max(ntree, 1)),
  ))

  # Smooth mass/RNE workspaces. `crb` and `local_inertia` are 6x6 spatial
  # matrices per body; omitting these made fixed-heavy models evade bounds.
  derivative_stride = 6 * (3 * nb + nv)
  for name, count in (
      ("mass", b * nv * nv), ("root_com", b * nb * 3),
      ("cdof", b * nv * 6), ("crb", b * nb * 36),
      ("local_inertia", b * nb * 36), ("qvel", b * nv),
      ("bias", b * nv), ("cvel", b * nb * 6),
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
  ):
    parts.append((f"smooth.{name}", max(count, 1)))

  # Simulation-owned inputs and per-step device buffers. Effective mass is
  # created only by profiles that request implicit Euler damping; include the
  # conservative maximum because this estimator is profile-independent.
  for name, count in (
      ("rhs", b * nv), ("applied_force", b * nv),
      ("body_wrench", b * nb * 6), ("control", b * nu),
      ("effective_mass", b * nv * nv), ("success", b),
      ("sensor_plugin_status", b),
      ("combined_status", b), ("next_qpos", b * nq),
      ("next_qvel", b * nv), ("next_qacc", b * nv),
      ("next_time", b), ("next_status", b),
      ("act_dot", b * na), ("act_vel", b * na),
      ("next_act", b * na), ("sensordata", b * nsensordata),
      ("sensor_act_force", b * max(nu, 1)),
      ("sensor_qfrc_act", b * max(nv, 1)),
      ("sleep_qvel", b * nv),
      ("sleep_fk_previous", b * max(ntree, 1)),
      ("sleep_fk_refresh", b * max(ntree, 1)),
  ):
    parts.append((f"simulation.{name}", max(count, 1)))

  # Sensor-owned buffers including the 128-byte-per-body subtree record,
  # RNE derivatives, and their shared scratch. Optional sensor families still
  # have the common output/runtime allocations below.
  parts.extend((
      ("sensor.stage_mask", 1 + b * max(nsensor, 1)),
      ("sensor.output", max(b * nsensordata, 1)),
      ("sensor.sensor_meta", max(nsensor, 1) * 10),
      ("sensor.subtree_runtime", 4 + b * nb * 32 + b * max(nsensor, 1)),
      ("sensor.rne_com_scratch", b * max(nb, 1) * 12),
      ("sensor.rne_cacc", b * max(nb, 1) * 6),
      ("sensor.rne_cfrc", b * max(nb, 1) * 6),
      ("sensor.rne_scom", b * max(nb, 1) * 3),
      ("sensor.rne_ext", b * max(nb, 1) * 6),
  ))
  return tuple(parts)


def validate_runtime_buffers(model, batch_size):
  """Validate profile-independent runtime backing buffers before allocation."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  if (isinstance(batch_size, (bool, np.bool_)) or
      not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
    raise ValueError("batch_size must be a positive integer")
  batch = int(batch_size)
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
  for name, elements in _runtime_buffer_sizes(model, batch):
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
                       solver_type=int(mujoco.mjtSolver.mjSOL_PGS)):
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
      not isinstance(solver_type, (int, np.integer))):
    raise ValueError("solver_type must be an integer")
  v = max(nv, 1)
  primal_scratch = primal_scratch_floats(nv, nr, solver_type)
  # Both ``empty`` and torch.zeros in prepare_workspace allocate one element
  # for an otherwise empty buffer. Keep that guard explicit so the estimate
  # covers the zero-contact/zero-row cases too.
  def guarded(n):
    return max(int(n), 1)
  blocks = max((int(nr) + 255) // 256, 1)
  parts = (
      ("contact_row_data", _bytes(guarded(b * nc * 6 * 6))),
      ("contact_frame", _bytes(guarded(b * nc * 12))),
      ("contact_jacobian", _bytes(guarded(b * nc * 6 * v))),
      ("pair_mask", _bytes(guarded(b * max(int(npairs), 1)))),
      ("workspace_J", _bytes(guarded(b * int(nr) * v))),
      # Reusable acceleration-space primal-solver scratch shares the debug
      # binding to preserve Metal's 31-buffer ABI.
      ("workspace_debug", _bytes(guarded(b * (int(nr) * int(nr) + 7 * int(nr) + primal_scratch)))),
      ("out_force", _bytes(guarded(b * v))),
      ("out_acc", _bytes(guarded(b * v))),
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


def estimate_capacity(model, batch_size, npairs, nslots, nr, *, neq=0, nr_joint=0) -> CapacityEstimate:
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
  validate_runtime_buffers(model, batch)
  nv = int(model.nv)
  nbody, ngeom = int(model.nbody), int(model.ngeom)
  npairs = _integer_count("npairs", npairs)
  nslots = _integer_count("nslots", nslots)
  nr = _integer_count("nr", nr)
  nr_joint = _integer_count("nr_joint", nr_joint)
  neq = _integer_count("neq", neq)
  parts, total = estimate_workspace(
      nv, npairs, nslots, nr, nr_joint, neq, batch,
      solver_type=int(model.opt.solver))
  # Include profile-independent and conditional runtime buffers in the memory
  # envelope. Address guards already ran once through the same per-buffer
  # inventory in validate_runtime_buffers().
  runtime_parts = []
  for name, elements in _runtime_buffer_sizes(model, batch):
    runtime_parts.append((name, _bytes(elements)))
  parts = parts + tuple(runtime_parts)
  total += sum(size for _, size in runtime_parts)
  return CapacityEstimate(
      nv=nv, nbody=nbody, ngeom=ngeom, npairs=int(npairs),
      nslots=int(nslots), nr=nr, batch=batch,
      dense_path=(nr <= DENSE_ROW_THRESHOLD and nv <= 32),
      jacobian_kind=selected_jacobian_kind(model),
      jacobian_auto_threshold=AUTO_JACOBIAN_DENSE_NV,
      memory_bytes=total, memory_breakdown=parts)


def check_capacity(estimate, limits=None) -> CapacityEstimate:
  """Enforce budgets; raise CapacityOverflow (a ValueError) on excess.

  Keeps the historical error text so existing boundary tests keep reading
  the same diagnostics, now with explicit requested-vs-allowed detail.
  """
  limits = limits or CapacityLimits()
  if estimate.batch > limits.max_batch:
    raise CapacityOverflow(
        f"batch size ({estimate.batch}) exceeds capacity {limits.max_batch}")
  max_nv = min(int(limits.max_nv), NATIVE_MAX_NV)
  if estimate.nv > max_nv:
    raise CapacityOverflow(
        f"coupled constraint stage bounds nv to {max_nv}; found {estimate.nv}")
  if estimate.npairs > limits.max_pairs:
    raise CapacityOverflow(
        f"candidate contact pairs ({estimate.npairs}) exceeds capacity {limits.max_pairs}")
  if estimate.nslots > limits.max_slots:
    raise CapacityOverflow(
        f"total candidate contact slots ({estimate.nslots}) exceeds capacity {limits.max_slots}")
  max_rows = min(int(limits.max_rows), NATIVE_MAX_ROWS)
  if estimate.nr > max_rows:
    raise CapacityOverflow(
        f"total candidate constraint rows ({estimate.nr}) exceeds capacity {max_rows}")
  if estimate.memory_bytes > limits.memory_budget_bytes:
    raise CapacityOverflow(
        f"estimated device memory ({estimate.memory_bytes} bytes) exceeds budget "
        f"{limits.memory_budget_bytes} bytes")
  return estimate
