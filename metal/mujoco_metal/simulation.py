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

"""All-device orchestration for explicitly validated native physics profiles."""

import numbers
import copy

import numpy as np

from dataclasses import replace, fields, is_dataclass

import mujoco

from mujoco_metal.device_state import DeviceState, _device_matches
from mujoco_metal.actuation import MetalScalarMotorForce
from mujoco_metal.actuation import ScalarMotorModel
from mujoco_metal.integration import MetalEulerIntegration
from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
from mujoco_metal.smooth_metal import MetalSmoothDynamics
from mujoco_metal.smooth_solve import MetalDenseSolve, MetalSymmetricLDLSolve
from mujoco_metal.stepping import validate_stepping_profile
from mujoco_metal.forward_stages import ForwardStageCoordinator

_AUTO_SENSOR_QACC_LOW = object()


def _assert_lifecycle_state_shapes_compatible(old_model, new_model):
  """Reject model swaps that cannot preserve model-sized DeviceState planes."""
  for attr in ("nhistory", "nuserdata", "npluginstate"):
    old_count = int(getattr(old_model, attr, 0))
    new_count = int(getattr(new_model, attr, 0))
    if old_count != new_count:
      raise ValueError(
          f"apply_lifecycle: structural count {attr} differs "
          f"({old_count} -> {new_count}); rebuild the simulation instead")


def _contact_override_option_signature(model):
  """Byte-exact binding for the mutable pinned global contact override."""
  flag = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
  enabled = int(model.opt.enableflags) & flag
  return (
      enabled,
      np.asarray(model.opt.o_margin, dtype=np.float64).tobytes(),
      np.asarray(model.opt.o_solref, dtype=np.float64).tobytes(),
      np.asarray(model.opt.o_solimp, dtype=np.float64).tobytes(),
      np.asarray(model.opt.o_friction, dtype=np.float64).tobytes(),
  )


def _clone_system_dict(d):
  """Clone structured device workspaces without retaining mutable map aliases."""
  import torch
  if isinstance(d, torch.Tensor):
    return d.detach().clone()
  if isinstance(d, dict):
    return {k: _clone_system_dict(v) for k, v in d.items()}
  if is_dataclass(d) and not isinstance(d, type):
    return replace(d, **{field.name: _clone_system_dict(getattr(d, field.name))
                         for field in fields(d) if field.init})
  if isinstance(d, (list, tuple)):
    return type(d)(_clone_system_dict(v) for v in d)
  return d


def _merge_accepted_system(target, source, accepted_mask, batch_size):
  """Update an owned accepted-system snapshot only for successful worlds."""
  import torch
  if source is None:
    return target
  if target is None:
    return _clone_system_dict(source)
  if isinstance(source, torch.Tensor):
    if (not isinstance(target, torch.Tensor)
        or tuple(target.shape) != tuple(source.shape)
        or target.dtype != source.dtype):
      return _clone_system_dict(source)
    if source.ndim and int(source.shape[0]) == int(batch_size):
      mask = accepted_mask.reshape(
          int(batch_size), *([1] * (source.ndim - 1)))
      torch.where(mask, source, target, out=target)
    return target
  if isinstance(source, dict):
    if not isinstance(target, dict):
      return _clone_system_dict(source)
    for key in tuple(target):
      if key not in source:
        del target[key]
    for key, value in source.items():
      target[key] = _merge_accepted_system(
          target.get(key), value, accepted_mask, batch_size)
    return target
  if is_dataclass(source) and not isinstance(source, type):
    if not (is_dataclass(target) and type(target) is type(source)):
      return _clone_system_dict(source)
    updates = {
        field.name: _merge_accepted_system(
            getattr(target, field.name), getattr(source, field.name),
            accepted_mask, batch_size)
        for field in fields(source) if field.init
    }
    return replace(target, **updates)
  if isinstance(source, list):
    if not isinstance(target, list) or len(target) != len(source):
      return _clone_system_dict(source)
    for index, value in enumerate(source):
      target[index] = _merge_accepted_system(
          target[index], value, accepted_mask, batch_size)
    return target
  if isinstance(source, tuple):
    if not isinstance(target, tuple) or len(target) != len(source):
      return _clone_system_dict(source)
    return type(source)(_merge_accepted_system(a, b, accepted_mask, batch_size)
                        for a, b in zip(target, source))
  return target


def _initial_accepted_system(source, accepted_mask, batch_size):
  """Own the first accepted-system snapshot, zeroing never-accepted rows."""
  import torch
  if isinstance(source, torch.Tensor):
    out = source.detach().clone()
    if source.ndim and int(source.shape[0]) == int(batch_size):
      mask = accepted_mask.reshape(
          int(batch_size), *([1] * (source.ndim - 1)))
      torch.where(mask, source, torch.zeros_like(source), out=out)
    return out
  if isinstance(source, dict):
    return {key: _initial_accepted_system(value, accepted_mask, batch_size)
            for key, value in source.items()}
  if is_dataclass(source) and not isinstance(source, type):
    return replace(source, **{
        field.name: _initial_accepted_system(
            getattr(source, field.name), accepted_mask, batch_size)
        for field in fields(source) if field.init
    })
  if isinstance(source, list):
    return [_initial_accepted_system(value, accepted_mask, batch_size)
            for value in source]
  if isinstance(source, tuple):
    return type(source)(_initial_accepted_system(value, accepted_mask, batch_size)
                        for value in source)
  return source


def _zero_invalid_rows(value, row_valid, state=None):
  """Clear the stale rows of a borrowed stage tensor in its current owner."""
  if value is None or row_valid is None or not hasattr(value, "shape"):
    return value
  if value.ndim == 0 or int(value.shape[0]) != int(row_valid.shape[0]):
    return value
  if state is not None:
    state.clear_masked_rows(value, row_valid, invert=True)
    return value
  mask = row_valid.reshape(row_valid.shape[0],
                            *([1] * (value.ndim - 1)))
  value.masked_fill_(~mask, 0)
  return value


def _select_euler_damping_input(passive_enabled, velocity_stage,
                                fresh_derivative, constant_damping):
  """Select damping owned by the stage used for the current step."""
  if not passive_enabled:
    return constant_damping
  source = (velocity_stage.get("passive_damping")
            if velocity_stage is not None else fresh_derivative)
  if source is None:
    raise RuntimeError(
        "Euler damping is enabled but its VEL-stage derivative is unavailable")
  return source


def _compiled_sparse_mass_nnz(model):
  """Return the exact block-sparse mass backing used by SmoothDynamics."""
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
  return int(layout["nnz"])


def _model_component_capacity_limits(model, batch_size):
  """Build a default scalable budget from exact compiled model counts.

  These are admission ceilings, not kernel-local truncation values. The
  separate memory-budget and signed-int32 checks remain in ``capacity.py``.
  """
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  nv = max(int(model.nv), 1)
  # Coupled lowering uses max_nv to select the dynamic descriptor path; its
  # other model dimensions are compiled and returned without fixed caps.
  signed_max = int(np.iinfo(np.int32).max)
  seed = CapacityLimits(
      max_nv=nv, max_pairs=signed_max, max_slots=signed_max,
      max_rows=signed_max, max_batch=max(int(batch_size), 1),
      memory_budget_bytes=(1 << 62))
  lowered = lower_coupled_constraints(model, limits=seed)
  return CapacityLimits(
      max_nv=nv,
      max_pairs=max(int(lowered.npairs), 1),
      max_slots=max(int(lowered.ncontacts_max), 1),
      max_rows=max(int(lowered.nr), 1),
      max_batch=max(int(batch_size), 1),
  )


def _restore_system_buffers(current, saved):
  """Restore borrowed tensor storage in place, including fixed compaction maps."""
  import torch
  if isinstance(saved, torch.Tensor):
    current.copy_(saved)
  elif isinstance(saved, dict):
    # A query may publish additional structured maps while evaluating its
    # temporary forward system. Restore the dictionary schema as well as
    # tensors so it cannot retain maps describing the discarded query.
    for key in tuple(current):
      if key not in saved:
        del current[key]
    for key, value in saved.items():
      _restore_system_buffers(current[key], value)
  elif is_dataclass(saved) and not isinstance(saved, type):
    for field in fields(saved):
      _restore_system_buffers(getattr(current, field.name), getattr(saved, field.name))
  elif isinstance(saved, (list, tuple)):
    for target, value in zip(current, saved):
      _restore_system_buffers(target, value)


def _plugin_state_key(plugin):
  return (plugin.plugin_type.value, plugin.name)


def _partition_force_plugins(plugins):
  """Return the passive-force and actuator plugin groups in registry order."""
  passive = tuple(p for p in plugins if p.plugin_type.value == "force")
  actuator = tuple(p for p in plugins if p.plugin_type.value == "actuator")
  return passive, actuator


def _plugin_stage_state(state, qpos, qvel, *, qacc=None, act=None, time=None):
  """Read-only callback view of the current forward stage, not live history.

  DeviceState's public tensor properties already return detached copies.
  A shallow view preserves those properties and compiled metadata while
  rebinding stage inputs without replacing the simulation's owned tensors.
  Plugins may mutate only their own registered, rollback-capable state.
  """
  stage = copy.copy(state)
  stage._qpos, stage._qvel = qpos, qvel
  if qacc is not None:
    stage._qacc = qacc
  if act is not None:
    stage._act = act
  if time is not None:
    stage._time = time
  return stage


def _check_plugin_device_rollback(plugins):
  """Reject stateful callbacks without device-native rollback support."""
  from mujoco_metal.extensions import NativePlugin
  for plugin in plugins:
    stateful = (type(plugin).snapshot is not NativePlugin.snapshot
                and plugin.snapshot() is not None)
    if not stateful:
      continue
    missing = [name for name in ("device_snapshot", "restore_device",
                                 "restore_masked", "reset_masked")
               if getattr(type(plugin), name, None) is getattr(NativePlugin, name)]
    if missing:
      raise TypeError(
          f"stateful native plugin {_plugin_state_key(plugin)!r} must implement "
          f"device-native rollback methods: {', '.join(missing)}")


def _poly_potential(stiffness, p0, p1, displacement):
  """Pinned mj_polyPotential with the two compiled polynomial terms."""
  return (0.5 * float(stiffness) * displacement.square()
          + (float(p0) / 3.0) * displacement.pow(3)
          + (float(p1) / 4.0) * displacement.pow(4))


def _quat_difference_angle(q, qref, torch):
  """Norm of MuJoCo's shortest q-minus-qref quaternion difference."""
  q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-30)
  qref = qref / torch.linalg.vector_norm(qref, dim=-1, keepdim=True).clamp_min(1e-30)
  w1, x1, y1, z1 = q.unbind(-1)
  w2, x2, y2, z2 = qref.unbind(-1)
  # q * conjugate(qref), in MuJoCo's [w,x,y,z] convention.
  w = w1*w2 + x1*x2 + y1*y2 + z1*z2
  x = -w1*x2 + x1*w2 - y1*z2 + z1*y2
  y = -w1*y2 + x1*z2 + y1*w2 - z1*x2
  z = -w1*z2 - x1*y2 + y1*x2 + z1*w2
  vnorm = torch.sqrt(x.square() + y.square() + z.square())
  return 2.0 * torch.atan2(vnorm, w.abs())


def _snapshot_plugins_device(plugins):
  """Capture plugin state on its owning device, without a host boundary."""
  return {plugin: plugin.device_snapshot() for plugin in plugins}


def _restore_plugins_device(snapshots):
  """Restore all plugin states after an exceptional step transaction."""
  errors = []
  for plugin, payload in reversed(tuple(snapshots.items())):
    try:
      plugin.restore_device(payload)
    except Exception as error:
      errors.append((_plugin_state_key(plugin), error))
  if errors:
    key, error = errors[0]
    raise RuntimeError(f"device rollback failed for plugin {key!r}") from error


def _restore_failed_plugin_rows(snapshots, accepted_mask):
  """Keep successful rows and restore callback state for rejected rows."""
  for plugin, payload in snapshots.items():
    plugin.restore_masked(payload, accepted_mask)


def _reset_plugins_device_masked(plugins, reset_mask):
  """Apply an all-or-nothing device-native reset to plugin-owned rows."""
  snapshots = _snapshot_plugins_device(plugins)
  try:
    for plugin in plugins:
      plugin.reset_masked(reset_mask)
  except Exception:
    _restore_plugins_device(snapshots)
    raise


def _copy_plugin_row(value, src, dst, batch):
  """Copy a plugin snapshot row when its public snapshot is batched data."""
  if isinstance(value, dict):
    return {k: _copy_plugin_row(v, src, dst, batch) for k, v in value.items()}
  if isinstance(value, tuple):
    return tuple(_copy_plugin_row(v, src, dst, batch) for v in value)
  if isinstance(value, list):
    return [_copy_plugin_row(v, src, dst, batch) for v in value]
  if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == batch:
    out = value.copy()
    out[dst] = value[src]
    return out
  try:
    import torch
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
      out = value.clone()
      out[dst] = value[src]
      return out
  except ImportError:
    pass
  return value


def _preflight_plugin_restores(plugin_map, payloads, env_ids=None):
  """Exercise restore validation before a surrounding state transaction commits.

  Plugin ABIs historically lacked a pure validation method. A restore probe
  gives rejecting implementations a chance to fail before simulation tensors
  are changed, and rolls every touched instance back to its captured payload.
  """
  active = {key: plugin for key, plugin in plugin_map.items()
            if payloads[key] is not None}
  old = {key: copy.deepcopy(plugin.snapshot())
         for key, plugin in active.items()}
  touched = []
  try:
    for key, plugin in active.items():
      touched.append(key)
      plugin.restore(copy.deepcopy(payloads[key]), env_ids=env_ids)
  except Exception:
    for key in reversed(touched):
      try:
        active[key].restore(copy.deepcopy(old[key]), env_ids=env_ids)
      except Exception:
        pass
    raise
  restore_errors = []
  for key in reversed(touched):
    try:
      active[key].restore(copy.deepcopy(old[key]), env_ids=env_ids)
    except Exception as error:
      restore_errors.append((key, error))
  if restore_errors:
    # A plugin can reject only its later restoration call. Retry the captured
    # value once before propagating, and never hide a failed rollback.
    retry_errors = []
    for key, _ in restore_errors:
      try:
        active[key].restore(copy.deepcopy(old[key]), env_ids=env_ids)
      except Exception as error:
        retry_errors.append(error)
    if retry_errors:
      raise RuntimeError("plugin preflight restore could not roll back its probes") from retry_errors[0]
    raise restore_errors[0][1]


def _apply_plugin_restores(plugin_map, payloads, env_ids=None):
  """Apply plugin payloads with rollback if an actual commit callback fails."""
  active = {key: plugin for key, plugin in plugin_map.items()
            if payloads[key] is not None}
  old = {key: copy.deepcopy(plugin.snapshot()) for key, plugin in active.items()}
  touched = []
  try:
    for key, plugin in active.items():
      touched.append(key)
      plugin.restore(copy.deepcopy(payloads[key]), env_ids=env_ids)
  except Exception:
    rollback_errors = []
    for key in reversed(touched):
      try:
        active[key].restore(copy.deepcopy(old[key]), env_ids=env_ids)
      except Exception as error:
        rollback_errors.append(error)
    if rollback_errors:
      raise RuntimeError("plugin restore failed and plugin rollback was rejected") from rollback_errors[0]
    raise


def _reset_plugins_transactionally(plugins, env_ids=None):
  """Reset plugin instances before native state commit and undo late errors."""
  active = tuple(plugins)
  old = tuple(copy.deepcopy(plugin.snapshot()) for plugin in active)
  touched = []
  try:
    for index, plugin in enumerate(active):
      touched.append(index)
      plugin.reset(env_ids=env_ids)
  except Exception:
    rollback_errors = []
    for index in reversed(touched):
      plugin = active[index]
      payload = copy.deepcopy(old[index])
      try:
        plugin.restore(payload, env_ids=env_ids)
      except Exception as error:
        rollback_errors.append(error)
    if rollback_errors:
      raise RuntimeError("plugin reset failed and plugin rollback was rejected") from rollback_errors[0]
    raise


class MetalSimulation:
  """Batched native MPS stepping for bounded, explicitly selected profiles.

  The default ``contact_free_euler_v1`` profile uses native smooth dynamics,
  a dense SPD solve and semi-implicit Euler with no applied forces. The opt-in
  ``contact_free_forces_euler_v1`` profile adds per-call generalized force and
  linear joint damping. ``contact_free_motor_euler_v1`` additionally supports
  the bounded scalar joint motor subset. Unsupported model features are
  rejected during construction before MPS initialization.
  ``state`` owns persistent qpos, qvel, qacc, time and status. Its reset and
  restore operations replace state tensors safely; every step reads the current
  tensors, so those operations do not leave stale cached references.
  A failed row keeps its first nonzero status and does not advance again until
  that row is reset or the state is restored.

  ``step`` returns a borrowed MPS int32 status tensor, valid until the next
  step call. Read or copy it before stepping again if it must be retained.
  State properties return detached copies, and snapshots perform host
  readback; use them outside the hot stepping loop.
  """

  def __init__(
      self,
      model,
      batch_size=1,
      qpos=None,
      qvel=None,
      *,
      profile="contact_free_euler_v1",
      limits=None,
  ):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    if isinstance(batch_size, bool) or not isinstance(
        batch_size, numbers.Integral
    ):
      raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    # Select the mass representation from the caller's profile name before
    # the first capacity pass.  In particular, the scalable profile must not
    # be rejected for a dense nv-by-nv backing it never allocates.
    requested_profile_name = getattr(profile, "name", profile)
    sparse_implicit_profiles = (
        "integrated_implicit_v1", "integrated_implicitfast_v1")
    component_mass_profiles = (
        "integrated_scalable_v1", "integrated_rk4_v1",
        *sparse_implicit_profiles)
    uses_component_mass = requested_profile_name in component_mass_profiles
    uses_sparse_implicit = requested_profile_name in sparse_implicit_profiles
    diag_exact_enabled = bool(
        int(model.opt.enableflags)
        & int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT))
    requested_mass_storage = "block_sparse" if uses_component_mass else "dense"
    velocity_derivative_layout = None
    if uses_sparse_implicit:
      # D is a compiled MuJoCo 3.10 CSR pattern. Lower its exact immutable
      # support on CPU before DeviceState initializes the MPS runtime.
      from mujoco_metal.velocity_derivative import compile_velocity_derivative_layout
      velocity_derivative_layout = compile_velocity_derivative_layout(model)
    # Guard profile-independent buffers after public type/range validation,
    # but before coercion or any component allocation/compilation.
    from mujoco_metal.capacity import validate_runtime_buffers
    from mujoco_metal.extensions import (
        default_registry, validate_actuator_user_bindings)
    plugin_registrations = default_registry.registration_snapshot()
    validate_actuator_user_bindings(model, plugin_registrations)
    actuator_user_bindings = tuple(
        (registration.name, registration.actuator_user_bindings)
        for registration in plugin_registrations
        if registration.actuator_user_bindings)
    actuator_user_workspace_bytes = sum(
        int(registration.device_workspace_bytes)
        for registration in plugin_registrations
        if registration.actuator_user_bindings)
    spatial_plugin_counts = default_registry.spatial_force_workspace_counts(
        plugin_registrations)
    validate_runtime_buffers(
        model, batch_size, mass_storage=requested_mass_storage,
        magnetic_force_plugins=spatial_plugin_counts["magnetic"],
        site_feedback_plugins=spatial_plugin_counts["site_feedback"],
        native_actuator_user_workspace_bytes=actuator_user_workspace_bytes)
    batch_size = int(batch_size)
    self.limits = limits
    if self.limits is None and uses_component_mass:
      # Resolve the no-override budget from this compiled model's immutable
      # contact/row counts. Scalable execution must not inherit the old small
      # profile's fixed pair, slot, or row ceilings.
      self.limits = _model_component_capacity_limits(model, batch_size)
    # This CPU-only contract check must finish before any constructor can
    # initialize MPS or compile a shader.
    profile = validate_stepping_profile(model, profile=profile, limits=self.limits)
    needs_component_row_preflight = profile.name in component_mass_profiles
    needs_dense_diag_preflight = (
        diag_exact_enabled and profile.name == "integrated_euler_v1")
    bundled_plugin_model = None
    touch_grid_sensors = ()
    bundled_cable_segments = None
    is_integrated = profile.name in (
        "integrated_euler_v1", "integrated_rk4_v1", "integrated_implicit_v1",
        "integrated_implicitfast_v1", "integrated_scalable_v1")
    if actuator_user_bindings and not is_integrated:
      raise ValueError(
          "actuator USER callbacks require an integrated native actuator profile")
    if is_integrated and int(model.nplugin):
      from mujoco_metal.bundled_plugins import lower_bundled_plugins
      from mujoco_metal.bundled_touch_grid import lower_touch_grid_sensors
      bundled_plugin_model = lower_bundled_plugins(model)
      touch_grid_sensors = lower_touch_grid_sensors(model, bundled_plugin_model)
      if any(instance.name == "mujoco.elasticity.cable"
             for instance in bundled_plugin_model.instances):
        from mujoco_metal.bundled_cable import lower_cable
        bundled_cable_segments = int(
            lower_cable(model, bundled_plugin_model)["body_ids"].size)
    needs_touch_grid_preflight = bool(touch_grid_sensors)
    needs_cable_preflight = bool(bundled_cable_segments)
    if (needs_component_row_preflight or needs_dense_diag_preflight
        or needs_touch_grid_preflight or needs_cable_preflight):
      # Lowering is host-only and gives the exact canonical RHS capacity the
      # component solver and coupled workspace will allocate.  Run this
      # second, profile-specific preflight before DeviceState touches MPS.
      from mujoco_metal.coupled_constraints import lower_coupled_constraints
      lowered_rows = lower_coupled_constraints(
          model, limits=self.limits, mass_storage=requested_mass_storage)
      sparse_rows = lowered_rows.nr
      diag_exact_rows = None
      diag_exact_cones = None
      if diag_exact_enabled and int(model.nv) > 0 and int(sparse_rows) > 0:
        from mujoco_metal.constraint_impedance import compile_contact_cone_groups
        exact_groups, _ = compile_contact_cone_groups(lowered_rows)
        diag_exact_rows = int(sparse_rows)
        diag_exact_cones = int(exact_groups.shape[0])
      gmres_dimension = None
      gmres_iterations = max(int(model.nv), 1)
      if uses_sparse_implicit:
        from mujoco_metal.capacity import estimate_capacity
        from mujoco_metal.implicit_effective import select_gmres_krylov_dimension
        if self.limits is not None and self.limits.gmres_max_iterations is not None:
          gmres_iterations = int(self.limits.gmres_max_iterations)
        if self.limits is not None and self.limits.gmres_krylov_dimension is not None:
          gmres_dimension = int(self.limits.gmres_krylov_dimension)
        base_estimate = estimate_capacity(
            model, batch_size, lowered_rows.npairs,
            lowered_rows.ncontacts_max, lowered_rows.nr,
            mass_storage="block_sparse",
            diag_exact_rows=diag_exact_rows,
            diag_exact_cones=diag_exact_cones,
            touch_grid_contact_capacity=(
                int(lowered_rows.ncontacts_max)
                if needs_touch_grid_preflight else None),
            touch_grid_pair_capacity=(
                int(lowered_rows.npairs) if needs_touch_grid_preflight else None),
            bundled_cable_segments=bundled_cable_segments,
            magnetic_force_plugins=spatial_plugin_counts["magnetic"],
            site_feedback_plugins=spatial_plugin_counts["site_feedback"],
            native_actuator_user_workspace_bytes=actuator_user_workspace_bytes)
        budget = (int(self.limits.memory_budget_bytes)
                  if self.limits is not None else 1 << 30)
        available = max(budget - int(base_estimate.memory_bytes), 0)
        gmres_dimension = select_gmres_krylov_dimension(
            batch=batch_size, nv=int(model.nv),
            edge_count=velocity_derivative_layout.edge_count,
            available_bytes=available, max_iterations=gmres_iterations,
            krylov_dimension=gmres_dimension)
      validate_runtime_buffers(
          model, batch_size, rhs_capacity=max(int(sparse_rows) + 1, 1),
          mass_storage=requested_mass_storage,
          effective_edge_count=(velocity_derivative_layout.edge_count
                                if uses_sparse_implicit else None),
          effective_gmres_dimension=(gmres_dimension
                                     if uses_sparse_implicit else None),
          effective_gmres_iterations=(gmres_iterations
                                      if uses_sparse_implicit else None),
          diag_exact_rows=diag_exact_rows,
          diag_exact_cones=diag_exact_cones,
          touch_grid_contact_capacity=(
              int(lowered_rows.ncontacts_max)
              if needs_touch_grid_preflight else None),
          touch_grid_pair_capacity=(
              int(lowered_rows.npairs) if needs_touch_grid_preflight else None),
          bundled_cable_segments=bundled_cable_segments,
          magnetic_force_plugins=spatial_plugin_counts["magnetic"],
          site_feedback_plugins=spatial_plugin_counts["site_feedback"],
          native_actuator_user_workspace_bytes=actuator_user_workspace_bytes)
      if needs_touch_grid_preflight or needs_cable_preflight:
        from mujoco_metal.capacity import CapacityLimits, check_capacity
        from mujoco_metal.capacity import estimate_capacity
        estimate = estimate_capacity(
            model, batch_size, int(lowered_rows.npairs),
            int(lowered_rows.ncontacts_max), int(lowered_rows.nr),
            nr_joint=int(lowered_rows.nr_joint),
            mass_storage=requested_mass_storage,
            effective_edge_count=(velocity_derivative_layout.edge_count
                                  if uses_sparse_implicit else None),
            effective_gmres_dimension=(gmres_dimension
                                       if uses_sparse_implicit else None),
            effective_gmres_iterations=(gmres_iterations
                                        if uses_sparse_implicit else None),
            touch_grid_contact_capacity=(
                int(lowered_rows.ncontacts_max)
                if needs_touch_grid_preflight else None),
            touch_grid_pair_capacity=(
                int(lowered_rows.npairs) if needs_touch_grid_preflight else None),
            bundled_cable_segments=bundled_cable_segments,
            magnetic_force_plugins=spatial_plugin_counts["magnetic"],
            site_feedback_plugins=spatial_plugin_counts["site_feedback"],
            native_actuator_user_workspace_bytes=actuator_user_workspace_bytes)
        check_capacity(estimate, self.limits or CapacityLimits())
      self._effective_gmres_dimension = gmres_dimension
      self._effective_gmres_iterations = gmres_iterations
    else:
      self._effective_gmres_dimension = None
      self._effective_gmres_iterations = None
    is_integrated = profile.name in ("integrated_euler_v1", "integrated_rk4_v1", "integrated_implicit_v1", "integrated_implicitfast_v1", "integrated_scalable_v1")
    legacy_joint_rows = (
        int(model.neq) + int(model.nv) + 2 * int(model.njnt)
        if profile.name == "joint_constraints_euler_v1" else 0)
    legacy_contact_rows = 0
    if profile.name in ("normal_contact_euler_v1", "friction_contact_euler_v1"):
      from mujoco_metal.contact import lower_contacts
      legacy_contact_rows = 5 * int(lower_contacts(model).pair_count)
    legacy_row_count = legacy_joint_rows + legacy_contact_rows
    with_transmissions = "transmission" in profile.name or is_integrated
    motor_model = (
        ScalarMotorModel.from_model(model)
        if (
            not is_integrated
            and profile.name.replace("rk4", "euler")
            in (
                "contact_free_motor_euler_v1",
                "joint_constraints_euler_v1",
                "contact_free_fluid_euler_v1",
                "contact_free_implicitfast_v1",
                "contact_free_implicit_v1",
                "contact_free_passive_euler_v1",
                "contact_free_sensor_euler_v1",
                "normal_contact_euler_v1",
                "friction_contact_euler_v1",
            )
        )
        else None
    )
    if motor_model is not None:
      maximum = (2**31 - 1) // max(1, motor_model.nv, motor_model.nu)
      if batch_size > maximum:
        raise ValueError(
            "batch exceeds scalar motor int32 kernel index capacity"
        )
    _validate_workspace_index_capacity(
        batch_size,
        {
            "qpos": batch_size * profile.nq,
            "qvel": batch_size * profile.nv,
            # Position-stage actuator kernels accept a [B,1] placeholder
            # when a source-valid model has actuators but no generalized DOFs.
            "zero_dof_actuator_qvel": (batch_size if model.nu and not model.nv else 0),
            "mass": (batch_size * profile.nv * profile.nv
                     if requested_mass_storage == "dense" else
                     batch_size * _compiled_sparse_mass_nnz(model)),
            "body_pos": batch_size * model.nbody * 3,
            "body_quat": batch_size * model.nbody * 4,
            "geom_pos": batch_size * model.ngeom * 3,
            "geom_quat": batch_size * model.ngeom * 4,
            "site_pos": batch_size * model.nsite * 3,
            "site_quat": batch_size * model.nsite * 4,
            "inertial_pos": batch_size * model.nbody * 3,
            "inertial_quat": batch_size * model.nbody * 4,
            "joint_anchor": batch_size * model.njnt * 3,
            "joint_axis": batch_size * model.njnt * 3,
            "root_com": batch_size * model.nbody * 3,
            "cdof": batch_size * profile.nv * 6,
            "crb": batch_size * model.nbody * 36,
            "local_inertia": batch_size * model.nbody * 36,
            "cvel": batch_size * model.nbody * 6,
            "cdof_dot": batch_size * profile.nv * 6,
            "cacc": batch_size * model.nbody * 6,
            "body_force": batch_size * model.nbody * 6,
            "ctrl": batch_size * model.nu,
            "energy_stage": (batch_size * 2 if int(model.opt.enableflags)
                             & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
            "energy_body_mass_const": (max(int(model.nbody) - 1, 1)
                                        if int(model.opt.enableflags)
                                        & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
            "energy_gravity_const": (3 if int(model.opt.enableflags)
                                     & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
            "energy_qpos_spring_const": (max(int(model.nq), 1)
                                         if int(model.opt.enableflags)
                                         & int(mujoco.mjtEnableBit.mjENBL_ENERGY) else 0),
            "legacy_constraint_rhs": (batch_size * int(model.nv)
                                      if legacy_joint_rows else 0),
            "legacy_joint_lambda": (batch_size * max(legacy_joint_rows, 1)
                                    if legacy_joint_rows else 0),
            "legacy_canonical_rows": (batch_size * max(legacy_row_count, 1)
                                      * (int(model.nv) + 5)
                                      if legacy_row_count else 0),
        },
        {
            "nq": profile.nq,
            "nv": profile.nv,
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
        },
    )
    self._state = DeviceState(
        model, profile, batch_size, qpos=qpos, qvel=qvel, limits=self.limits)
    # Split-stage records are tied to this simulation and the DeviceState
    # generation. Native API wrappers publish borrowed stage outputs here;
    # any public state mutation makes an earlier record fail validation.
    self._forward_stages = ForwardStageCoordinator(self)
    self.batch_size = int(batch_size)
    self._mjmodel = model
    self._contact_override_binding = _contact_override_option_signature(model)
    descriptor = self._state._model
    self.profile = profile
    smooth_descriptor = descriptor
    self._component_mass_enabled = profile.name in component_mass_profiles
    if with_transmissions and model.ntendon and not self._component_mass_enabled:
      # Fixed tendon armature is assembled separately as J.T @ armature @ J.
      smooth_descriptor = replace(
          descriptor, tendon_armature=np.zeros_like(descriptor.tendon_armature)
      )
    self._smooth = MetalSmoothDynamics(
        smooth_descriptor,
        batch_size=batch_size,
        mass_storage=("block_sparse" if self._component_mass_enabled else "dense"),
        velocity_derivative_layout=velocity_derivative_layout,
    )
    self._motor = (
        MetalScalarMotorForce(motor_model, batch_size=batch_size)
        if motor_model is not None
        else None
    )
    self._transmissions = None
    self._actuators = None
    self._tendons = None
    self._spatial_tendons = None
    self._spatial_kin = None
    self._spatial_cache_key = None
    self._tendon_damping = None
    self._passive = None
    self._damping_tangent = None
    self._fluid = None
    self._sensors = None
    self._sensordata = None
    self._joint_constraints = None
    self._contact = None
    self._coupled_constraints = None
    self._flex = None
    if hasattr(model, "nflex") and model.nflex > 0:
      from mujoco_metal.flex import MetalFlex
      self._flex = MetalFlex(model, batch_size=batch_size, device=self._state._device)
    self._accepted_step = None
    self._coupled_solve_dispatches = 0

    if is_integrated:
      import mujoco as _mj_init
      self._bundled_plugin_model = bundled_plugin_model
      plan = profile.execution_plan
      # Actuation state ownership requires the actuator objects whenever the
      # model has actuators, even with mjDSBL_ACTUATION: the disable bit
      # gates force output (kernels emit zeros) and freezes the advance
      # (next carries live state), matching pinned skip semantics while
      # keeping activation owned end-to-end (R02).
      _act_disabled = bool(int(model.opt.disableflags)
                           & int(_mj_init.mjtDisableBit.mjDSBL_ACTUATION))
      if plan.is_stage_enabled("actuation") or (_act_disabled and int(model.na) > 0):
        from mujoco_metal.stateful_actuation import ActuatorModel
        from mujoco_metal.stateful_actuation import MetalActuators
        actuator_meta = ActuatorModel(
            model, allow_inherited=True,
            bundled_plugins=self._bundled_plugin_model,
            actuator_user_bindings=actuator_user_bindings)
        has_bundled_pid = bool(np.any(actuator_meta.plugin_actuator_mask))
        if (actuator_meta.needs_general_path
            or has_bundled_pid
            or bool(actuator_user_bindings)
            or ("implicit" in profile.name and int(model.nu) > 0)):
          self._actuators = MetalActuators(
              model, batch_size,
              velocity_derivative_layout=velocity_derivative_layout,
              bundled_plugins=self._bundled_plugin_model,
              actuator_user_bindings=actuator_user_bindings)
        elif plan.is_stage_enabled("actuation"):
          from mujoco_metal.transmissions import MetalTransmissions
          self._transmissions = MetalTransmissions(model, batch_size=batch_size)
      if plan.is_stage_enabled("fixed_tendons"):
        from mujoco_metal.tendons import MetalFixedTendonDynamics
        self._tendons = MetalFixedTendonDynamics(
            model, batch_size, spatial_ok=True,
            armature_storage=("external" if self._component_mass_enabled else "dense"),
            velocity_derivative_layout=velocity_derivative_layout)
        from mujoco_metal.spatial_tendons import SpatialTendonModel
        from mujoco_metal.spatial_tendons import MetalSpatialTendonDynamics
        if SpatialTendonModel(model).has_spatial:
          self._spatial_tendons = MetalSpatialTendonDynamics(
              model, batch_size,
              armature_storage=("external" if self._component_mass_enabled else "dense"),
              velocity_derivative_layout=velocity_derivative_layout)
      if plan.is_stage_enabled("passive_forces"):
        from mujoco_metal.passive import MetalPassiveForces
        self._passive = MetalPassiveForces(model, batch_size=batch_size)
      if plan.is_stage_enabled("fluid_forces"):
        from mujoco_metal.fluid import MetalInertiaBoxFluid
        self._fluid = MetalInertiaBoxFluid(model, batch_size)
      if plan.is_stage_enabled("coupled_constraints"):
        from mujoco_metal.coupled_constraints import MetalCoupledConstraints
        self._coupled_constraints = MetalCoupledConstraints(
            model, batch_size, limits=self.limits,
            mass_storage=requested_mass_storage,
            component_layout=(self._smooth.mass_block_layout
                              if self._component_mass_enabled else None))
      if plan.is_stage_enabled("sensor_query"):
        from mujoco_metal.sensors import SensorProgram
        self._sensors = SensorProgram(model, batch_size)
    else:
      if with_transmissions and model.nu:
        from mujoco_metal.transmissions import MetalTransmissions
        self._transmissions = MetalTransmissions(model, batch_size=batch_size)
      if with_transmissions and model.ntendon:
        from mujoco_metal.tendons import MetalFixedTendonDynamics
        self._tendons = MetalFixedTendonDynamics(
            model, batch_size, spatial_ok=True,
            armature_storage=("external" if self._component_mass_enabled else "dense"),
            velocity_derivative_layout=velocity_derivative_layout)
      if (
          with_transmissions
          or "joint_constraints" in profile.name
          or "fluid" in profile.name
          or "implicitfast" in profile.name
          or "implicit" in profile.name
          or "passive" in profile.name
          or "sensor" in profile.name
          or profile.name in ("normal_contact_euler_v1", "friction_contact_euler_v1")
      ):
        from mujoco_metal.passive import MetalPassiveForces
        self._passive = MetalPassiveForces(model, batch_size=batch_size)
      if "fluid" in profile.name or (
          "implicit" in profile.name
          and (model.opt.density > 0 or model.opt.viscosity > 0 or np.any(model.opt.wind != 0))
      ):
        from mujoco_metal.fluid import MetalInertiaBoxFluid
        self._fluid = MetalInertiaBoxFluid(model, batch_size)
      if model.ntendon and "implicit" in profile.name:
        from mujoco_metal.tendons import MetalFixedTendonDynamics
        self._tendons = MetalFixedTendonDynamics(
            model, batch_size, spatial_ok=True,
            armature_storage=("external" if self._component_mass_enabled else "dense"),
            velocity_derivative_layout=velocity_derivative_layout)
      if "sensor" in profile.name:
        from mujoco_metal.sensors import SensorProgram
        self._sensors = SensorProgram(model, batch_size)
      if profile.name == "joint_constraints_euler_v1":
        from mujoco_metal.joint_constraints import JointConstraintProgram
        self._joint_constraints = JointConstraintProgram(model, batch_size)
      if profile.name in (
          "normal_contact_euler_v1",
          "friction_contact_euler_v1",
      ) and not (
          int(model.opt.disableflags)
          & (
              int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
              | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
          )
      ):
        from mujoco_metal.contact import MetalContact
        self._contact = MetalContact(model, batch_size)

    # Legacy contact-free transmission profiles use the same source spatial
    # force, inertia and velocity producers as integrated execution plans.
    if self._tendons is not None and self._spatial_tendons is None:
      from mujoco_metal.spatial_tendons import SpatialTendonModel
      from mujoco_metal.spatial_tendons import MetalSpatialTendonDynamics
      if SpatialTendonModel(model).has_spatial:
        self._spatial_tendons = MetalSpatialTendonDynamics(
            model, batch_size,
            armature_storage=("external" if self._component_mass_enabled else "dense"),
            velocity_derivative_layout=velocity_derivative_layout)

    self._component_solver = None
    self._component_solve_rhs = None
    self._component_solution_vector = None
    self._component_solution_low_vector = None
    self._component_world_status = None
    self._component_tendon_J = None
    self._effective_implicit = None
    self._velocity_derivative_values = None
    self._constraint_impedance = None
    self._exact_impedance_dense_solver = None
    self._exact_impedance_status = None
    if self._component_mass_enabled:
      torch = self._state._torch
      from mujoco_metal.component_solve import MetalComponentMassSolver
      from mujoco_metal.stepping import _euler_damping_dofs
      layout = self._smooth.mass_block_layout
      nr = (int(self._coupled_constraints.descriptor.nr)
            if self._coupled_constraints is not None else 0)
      self._component_solver = MetalComponentMassSolver(
          descriptor, layout, batch_size=batch_size,
          rhs_capacity=max(nr + 1, 1),
          euler_damping_dofs=_euler_damping_dofs(model))
      self._component_solution_vector = torch.empty(
          (batch_size, descriptor.nv), dtype=torch.float32,
          device=self._state._device)
      self._component_solution_low_vector = torch.empty(
          (batch_size, descriptor.nv), dtype=torch.float32,
          device=self._state._device)
      if (self._coupled_constraints is not None
          and self._coupled_constraints._component_mass_storage != "block_sparse"):
        raise RuntimeError(
            "coupled constraints were not initialized with component storage")
      # This is the independent smooth-force solve input. Do not depend on
      # the coupled solver's optional PGS row-RHS tail: CG/Newton dispatches
      # use that workspace differently and still need M^-1 qfrc_smooth here.
      self._component_solve_rhs = torch.empty(
          (batch_size, self._component_solver.rhs_capacity, descriptor.nv),
          dtype=torch.float32, device=self._state._device)
      self._component_world_status = torch.zeros(
          (batch_size,), dtype=torch.int32, device=self._state._device)
      if (model.ntendon and self._smooth._has_tendon_armature):
        self._component_tendon_J = torch.zeros(
            (batch_size, max(int(model.ntendon), 1),
             max(int(descriptor.nv), 1)), dtype=torch.float32,
            device=self._state._device)
      if uses_sparse_implicit:
        from mujoco_metal.implicit_effective import MetalEffectiveImplicitSolve
        from mujoco_metal.velocity_derivative import MetalVelocityDerivativeValues
        self._effective_implicit = MetalEffectiveImplicitSolve(
            self._component_solver,
            edge_rows=velocity_derivative_layout.edge_rows,
            edge_cols=velocity_derivative_layout.edge_cols,
            krylov_dimension=self._effective_gmres_dimension,
            max_iterations=self._effective_gmres_iterations)
        self._velocity_derivative_values = MetalVelocityDerivativeValues(
            velocity_derivative_layout, batch_size,
            self._effective_implicit._workspace["edge_values"],
            edge_rows_device=self._effective_implicit._edge_rows,
            edge_cols_device=self._effective_implicit._edge_cols)
    if (diag_exact_enabled and self._coupled_constraints is not None
        and int(self._coupled_constraints.descriptor.nr) > 0
        and int(descriptor.nv) > 0):
      from mujoco_metal.constraint_impedance import (
          MetalConstraintImpedance, compile_contact_cone_groups)
      cone_groups, cone_friction = compile_contact_cone_groups(
          self._coupled_constraints.descriptor)
      self._constraint_impedance = MetalConstraintImpedance(
          batch_size=batch_size,
          row_capacity=int(self._coupled_constraints.descriptor.nr),
          dof_capacity=int(descriptor.nv),
          cone_groups=cone_groups, friction=cone_friction,
          mass_storage=("block_sparse" if self._component_mass_enabled
                        else "dense"))
      if not self._component_mass_enabled:
        from mujoco_metal.smooth_solve import MetalFactorizedSolve
        self._exact_impedance_dense_solver = MetalFactorizedSolve(
          int(descriptor.nv), batch_size,
            nrhs=int(self._coupled_constraints.descriptor.nr))
      self._exact_impedance_status = self._state._torch.zeros(
          (batch_size,), dtype=self._state._torch.int32,
          device=self._state._device)
    self._solver = (None if self._component_mass_enabled
                    else MetalDenseSolve(descriptor.nv, batch_size))
    self._implicitfast = None
    self._implicit = None
    self._midpoint = None
    if profile.name in ("contact_free_implicitfast_v1", "integrated_implicitfast_v1") and not uses_sparse_implicit:
      from mujoco_metal.implicit import ImplicitFastProgram

      self._implicitfast = ImplicitFastProgram(model, batch_size)
      from mujoco_metal.implicit_midpoint import FreeBodyMidpointProgram
      from mujoco_metal.implicit_midpoint import lower_free_body_midpoints

      if lower_free_body_midpoints(model).nfree:
        self._midpoint = FreeBodyMidpointProgram(model, batch_size)
    elif profile.name in ("contact_free_implicit_v1", "integrated_implicit_v1") and not uses_sparse_implicit:
      from mujoco_metal.implicit import ImplicitProgram

      self._implicit = ImplicitProgram(model, batch_size)
    self._flex_implicit = None
    if self._flex is not None and (self._implicit is not None or self._implicitfast is not None):
      from mujoco_metal.flex_implicit import (
          FlexImplicitCorrection, requires_flex_implicit_correction)
      if requires_flex_implicit_correction(model):
        self._flex_implicit = FlexImplicitCorrection(model, batch_size)
    self._sparse_flex_implicit = None
    if (self._flex is not None and self._component_mass_enabled
        and self._effective_implicit is not None):
      from mujoco_metal.flex_implicit import requires_flex_implicit_correction
      if requires_flex_implicit_correction(model):
        from mujoco_metal.sparse_flex_implicit import (
            SparseFlexImplicitCorrection)
        self._sparse_flex_implicit = SparseFlexImplicitCorrection(
            batch_size, int(model.nv),
            int(velocity_derivative_layout.edge_count))
    self._euler_solver = (
        MetalSymmetricLDLSolve(descriptor.nv, batch_size)
        if (
            not self._component_mass_enabled
            and (
            profile.execution_plan.is_stage_enabled("euler_damping")
            if is_integrated
            else profile.implicit_euler_damping
            )
        )
        else None
    )
    self._integrator = MetalEulerIntegration(
        descriptor, batch_size, profile.timestep
    )

    torch = self._state._torch
    from mujoco_metal.stepping import _euler_damping_dofs
    self._euler_damping_dofs = torch.as_tensor(
        _euler_damping_dofs(model).copy(), dtype=torch.bool,
        device=self._state._device)
    self._rhs = torch.empty(
        (batch_size, descriptor.nv),
        dtype=torch.float32,
        device=self._state._device,
    )
    self._rhs_low = torch.zeros_like(self._rhs)
    self._forward_stage_zero_constraint = torch.zeros_like(self._rhs)
    self._forward_stage_passive_force = torch.empty_like(self._rhs)
    self._forward_stage_actuator_force = torch.empty_like(self._rhs)
    self._legacy_row_count = (
        (int(self._joint_constraints.descriptor.nrow)
         if self._joint_constraints is not None else 0)
        + (int(self._contact.descriptor.pair_count) * 5
           if self._contact is not None else 0))
    self._legacy_constraint_rhs = (
        torch.empty_like(self._rhs) if self._joint_constraints is not None else None)
    self._legacy_canonical_rows = (
        torch.empty((batch_size, self._legacy_row_count, int(model.nv) + 5),
                    dtype=torch.float32, device=self._state._device)
        if self._legacy_row_count else None)
    self._solver_fwdinv = torch.zeros(
        (batch_size, 2), dtype=torch.float32, device=self._state._device)
    self._energy = (
        torch.zeros((batch_size, 2), dtype=torch.float32,
                    device=self._state._device)
        if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        else None)
    self._energy_stage = (torch.empty_like(self._energy)
                          if self._energy is not None else None)
    self._energy_body_mass = None
    self._energy_gravity = None
    self._energy_qpos_spring = None
    if self._energy is not None:
      self._energy_body_mass = torch.zeros(
          (max(int(model.nbody) - 1, 1),), dtype=torch.float32,
          device=self._state._device)
      if model.nbody > 1:
        self._energy_body_mass[:model.nbody - 1].copy_(torch.as_tensor(
            np.asarray(model.body_mass[1:], dtype=np.float32).copy(),
            dtype=torch.float32, device=self._state._device))
      self._energy_gravity = torch.as_tensor(
          np.asarray(model.opt.gravity, dtype=np.float32).copy(),
          dtype=torch.float32, device=self._state._device)
      self._energy_qpos_spring = torch.zeros(
          (max(int(model.nq), 1),), dtype=torch.float32,
          device=self._state._device)
      if model.nq:
        self._energy_qpos_spring[:model.nq].copy_(torch.as_tensor(
            np.asarray(model.qpos_spring, dtype=np.float32).copy(),
            dtype=torch.float32, device=self._state._device))
      energy_jint = np.zeros((max(int(model.njnt), 1), 3), dtype=np.int32)
      energy_jfloat = np.zeros((max(int(model.njnt), 1), 3), dtype=np.float32)
      for jid in range(int(model.njnt)):
        energy_jint[jid] = (int(model.jnt_type[jid]),
                            int(model.jnt_qposadr[jid]),
                            int(model.jnt_bodyid[jid]))
        energy_jfloat[jid] = (
            float(model.jnt_stiffness[jid]),
            float(np.asarray(model.jnt_stiffnesspoly).reshape(-1, 2)[jid, 0]),
            float(np.asarray(model.jnt_stiffnesspoly).reshape(-1, 2)[jid, 1]))
      energy_tint = np.zeros((max(int(model.ntendon), 1), 4), dtype=np.int32)
      energy_tfloat = np.zeros((max(int(model.ntendon), 1), 5), dtype=np.float32)
      tendon_poly = np.asarray(model.tendon_stiffnesspoly).reshape(
          int(model.ntendon), 2) if int(model.ntendon) else np.zeros((0, 2))
      tendon_range = np.asarray(model.tendon_lengthspring).reshape(
          int(model.ntendon), 2) if int(model.ntendon) else np.zeros((0, 2))
      paths = (self._spatial_tendons._meta.paths
               if self._spatial_tendons is not None else [None] * int(model.ntendon))
      for tid in range(int(model.ntendon)):
        tree_num = int(np.asarray(model.tendon_treenum)[tid])
        treeids = np.asarray(model.tendon_treeid).reshape(
            int(model.ntendon), 2)[tid]
        energy_tint[tid] = (tree_num, int(treeids[0]), int(treeids[1]),
                            int(paths[tid] is not None))
        energy_tfloat[tid] = (
            float(model.tendon_stiffness[tid]),
            float(tendon_poly[tid, 0]), float(tendon_poly[tid, 1]),
            float(tendon_range[tid, 0]), float(tendon_range[tid, 1]))
      edge_pairs = []
      flex_dim = np.asarray(model.flex_dim)
      flex_rigid = np.asarray(model.flex_rigid)
      edge_adr = np.asarray(model.flex_edgeadr)
      edge_num = np.asarray(model.flex_edgenum)
      edge_rigid = np.asarray(model.flexedge_rigid)
      rest = np.asarray(model.flexedge_length0, dtype=np.float32)
      edge_stiffness = np.asarray(model.flex_edgestiffness, dtype=np.float32)
      for fid in range(int(model.nflex)):
        if bool(flex_rigid[fid]) or int(flex_dim[fid]) > 1:
          continue
        for edge in range(int(edge_adr[fid]),
                          int(edge_adr[fid]) + int(edge_num[fid])):
          if not bool(edge_rigid[edge]) and float(edge_stiffness[fid]) != 0.0:
            edge_pairs.append((edge, float(rest[edge]),
                               float(edge_stiffness[fid])))
      self._energy_joint_int = torch.as_tensor(
          energy_jint.reshape(-1), dtype=torch.int32, device=self._state._device)
      self._energy_joint_float = torch.as_tensor(
          energy_jfloat.reshape(-1), dtype=torch.float32, device=self._state._device)
      self._energy_tendon_int = torch.as_tensor(
          energy_tint.reshape(-1), dtype=torch.int32, device=self._state._device)
      self._energy_tendon_float = torch.as_tensor(
          energy_tfloat.reshape(-1), dtype=torch.float32, device=self._state._device)
      flex_int = np.zeros((max(len(edge_pairs), 1),), dtype=np.int32)
      flex_float = np.zeros((max(len(edge_pairs), 1), 2), dtype=np.float32)
      self._energy_flex_row_count = len(edge_pairs)
      if edge_pairs:
        flex_int[:len(edge_pairs)] = [row[0] for row in edge_pairs]
        flex_float[:len(edge_pairs)] = [row[1:] for row in edge_pairs]
      self._energy_flex_int = torch.as_tensor(
          flex_int, dtype=torch.int32, device=self._state._device)
      self._energy_flex_float = torch.as_tensor(
          flex_float.reshape(-1), dtype=torch.float32,
          device=self._state._device)
      self._energy_position_dims = torch.tensor(
          [int(model.nq), int(model.nv), int(model.nbody), int(model.njnt),
           int(model.ntendon), int(model.nflexedge), len(edge_pairs),
           max(int(model.ntree), 1),
           int(bool(int(model.opt.disableflags)
                    & int(mujoco.mjtDisableBit.mjDSBL_SPRING))),
           int(bool(int(model.opt.disableflags)
                    & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))),
           int(bool(int(model.opt.enableflags)
                    & int(mujoco.mjtEnableBit.mjENBL_SLEEP))),
           int(batch_size)] + [1] * int(batch_size),
          dtype=torch.int32, device=self._state._device)
    self._component_damping_deriv = (
        torch.empty_like(self._rhs) if self._component_mass_enabled else None)
    self._forward_position_epoch = 0
    self._forward_position_buffers = {}
    if self._tendons is not None:
      self._forward_position_buffers["fixed_tendon_length"] = torch.empty_like(
          self._tendons._last_length)
    if self._spatial_tendons is not None:
      nt = max(self._spatial_tendons._meta.ntendon, 1)
      self._forward_position_buffers["spatial_tendons"] = {
          "length": torch.empty((batch_size, nt), dtype=torch.float32,
                                device=self._state._device),
          "jacobian": torch.empty((batch_size, nt, max(descriptor.nv, 1)),
                                  dtype=torch.float32, device=self._state._device)}
    if self._actuators is not None:
      self._forward_position_buffers["actuators"] = {
          "length": torch.empty((batch_size, model.nu), dtype=torch.float32,
                                device=self._state._device),
          "moment": torch.empty((batch_size, model.nu, max(descriptor.nv, 1)),
                                dtype=torch.float32, device=self._state._device)}
      if not descriptor.nv:
        # The physical qvel has shape [B,0]. Keep an owned zero placeholder
        # for actuator kinematics' padded [B,max(nv,1)] input contract.
        # The `_forward_stage_` name makes native query transactions preserve
        # this reusable scratch by identity and contents.
        self._forward_stage_actuator_qvel = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=self._state._device)
    # mjtState selectors own these inputs independently of whether an
    # integrated profile has a corresponding force-producing model stage.
    self._applied_force = torch.zeros_like(self._rhs)
    self._body_wrench = torch.zeros(
        (batch_size, int(model.nbody), 6), dtype=torch.float32,
        device=self._state._device)
    self._control = torch.zeros(
        (batch_size, model.nu), dtype=torch.float32,
        device=self._state._device)
    self._damping = torch.tensor(
        descriptor.dof_damping
        if profile.passive_damping_enabled
        else np.zeros(descriptor.nv),
        dtype=torch.float32,
        device=self._state._device,
    )
    self._effective_mass = (
        torch.empty(
            (batch_size, descriptor.nv, descriptor.nv),
            dtype=torch.float32,
            device=self._state._device,
        )
        if profile.implicit_euler_damping and not self._component_mass_enabled
        else None
    )
    self._implicit_damping = (
        self.profile.timestep * self._damping
        if profile.implicit_euler_damping
        else None
    )
    self._success = torch.empty(
        (batch_size,), dtype=torch.bool, device=self._state._device
    )
    self._sensor_plugin_status = torch.zeros(
        (batch_size,), dtype=torch.int32, device=self._state._device)
    self._cached_act_force = None
    self._cached_qfrc_act = None
    self._sen_act_force = None
    self._sen_qfrc_act = None
    self._sen_jnt_map = None
    self._sen_ten_map = None
    self._sen_pair_live = None
    self._sen_slot_pair = None
    self._has_acc_sensors = False
    self._has_contact_sensors = False
    self._has_ray_sensors = False
    self._has_geomdist_sensors = False
    self._has_user_plugin_sensors = False
    self._touch_grid = None
    self._has_touch_grid_sensors = False
    self._cable = None
    if bundled_plugin_model is not None and any(
        instance.name == "mujoco.elasticity.cable"
        for instance in bundled_plugin_model.instances):
      from mujoco_metal.bundled_cable import MetalBundledCable
      self._cable = MetalBundledCable(
          model, bundled_plugin_model, batch_size=batch_size,
          device=self._state._device)
    if self._sensors is not None:
      ST = mujoco.mjtSensor
      need = np.asarray(self._sensors.descriptor.sensor_needstage)
      types = np.asarray(self._sensors.descriptor.sensor_type)
      ACC = int(mujoco.mjtStage.mjSTAGE_ACC)
      self._has_acc_sensors = bool(np.any(need == ACC))
      self._has_contact_sensors = bool(np.any(np.isin(types, [
          int(ST.mjSENS_CONTACT), int(ST.mjSENS_TOUCH), int(ST.mjSENS_TACTILE)])))
      self._has_user_plugin_sensors = bool(np.any(np.isin(types, [
          int(ST.mjSENS_USER), int(ST.mjSENS_PLUGIN)])))
      if bundled_plugin_model is not None:
        from mujoco_metal.bundled_touch_grid import (
            MetalBundledTouchGrid, lower_touch_grid_sensors)
        touch_sensors = lower_touch_grid_sensors(model, bundled_plugin_model)
        if touch_sensors:
          self._touch_grid = MetalBundledTouchGrid(
              model, bundled_plugin_model, batch_size=batch_size,
              device=self._state._device,
              contact_capacity=(int(self._coupled_constraints.descriptor.ncontacts_max)
                                if self._coupled_constraints is not None else 0),
              pair_capacity=(int(self._coupled_constraints.descriptor.npairs)
                             if self._coupled_constraints is not None else 0))
          self._has_touch_grid_sensors = True
      self._has_ray_sensors = bool(np.any(types == int(ST.mjSENS_RANGEFINDER)))
      self._has_geomdist_sensors = bool(np.any(np.isin(types, [
          int(ST.mjSENS_GEOMDIST), int(ST.mjSENS_GEOMNORMAL),
          int(ST.mjSENS_GEOMFROMTO)])))
      if (self._has_ray_sensors or self._has_geomdist_sensors) and profile.name != "integrated_euler_v1":
        GT = mujoco.mjtGeom
        stock = {int(GT.mjGEOM_MESH), int(GT.mjGEOM_HFIELD), int(GT.mjGEOM_SDF)}
        if bool(np.any(np.isin(np.asarray(model.geom_type), list(stock)))):
          raise ValueError(
              f"{profile.name} supports rays/geom-distance only for analytic "
              "scenes; mesh/heightfield/SDF spatial queries require "
              "integrated_euler_v1")
      cc = getattr(self, "_coupled_constraints", None)
      if cc is not None:
        d = cc.descriptor
        g1 = np.asarray(d.geom1).reshape(-1)
        g2 = np.asarray(d.geom2).reshape(-1)
        gb = np.asarray(model.geom_bodyid, dtype=np.int64)
        jn = np.asarray(model.body_jntnum, dtype=np.int64)
        par = np.asarray(model.body_parentid, dtype=np.int64)
        live = []
        for a, b in zip(g1.tolist(), g2.tolist()):
          ok = False
          for bb in (int(gb[int(a)]), int(gb[int(b)])):
            while bb > 0:
              if int(jn[bb]) > 0:
                ok = True
                break
              bb = int(par[bb])
            if ok:
              break
          live.append(1 if ok else 0)
        self._sen_pair_live = self._state._torch.as_tensor(
            np.array(live if live else [1], dtype=np.int32),
            dtype=self._state._torch.int32, device=self._state._device)
        off = np.asarray(d.pair_contact_offset, dtype=np.int64)
        slot = np.zeros(int(d.ncontacts_max), dtype=np.int32)
        for p in range(int(d.npairs)):
          slot[off[p]:off[p + 1]] = p
        self._sen_slot_pair = self._state._torch.as_tensor(
            slot.copy(), dtype=self._state._torch.int32,
            device=self._state._device)
      if profile.name != "integrated_euler_v1":
        S = mujoco.mjtSensor
        gated = {int(S.mjSENS_TOUCH), int(S.mjSENS_CONTACT),
                 int(S.mjSENS_JOINTLIMITFRC), int(S.mjSENS_TENDONLIMITFRC)}
        if bool(np.any(np.isin(np.asarray(
                self._sensors.descriptor.sensor_type), list(gated)))):
          raise ValueError(
              f"{profile.name} supports touch/contact/limit-force sensors "
              "only under integrated_euler_v1")
    self._combined_status = torch.empty_like(self._state._status)
    self._next_qpos = torch.empty_like(self._state._qpos)
    self._next_qvel = torch.empty_like(self._state._qvel)
    self._next_qacc = torch.empty_like(self._state._qacc)
    # A recovery check can synchronously run another forward ACC into the
    # same borrowed solver result planes. Preserve the source candidate's
    # low word across that nested, row-masked dispatch.
    self._check_qacc_low = torch.empty_like(self._state._qacc)
    self._next_time = torch.empty_like(self._state._time)
    self._next_status = torch.empty_like(self._state._status)
    na = int(getattr(self._state, "_na", 0))
    nu = int(model.nu)
    if na > 0 and self._actuators is None:
      raise ValueError("activation state requires the general actuator stage")
    self._act_dot = torch.zeros(
        (batch_size, max(na, 1)), dtype=torch.float32, device=self._state._device)
    self._act_vel = torch.zeros(
        (batch_size, max(nu, 1)), dtype=torch.float32, device=self._state._device)
    self._next_act = (torch.empty_like(self._state._act)
                      if na > 0 else None)
    if "rk4" in profile.name:
      from mujoco_metal.runge_kutta import MetalRungeKutta

      self._rk4 = MetalRungeKutta(descriptor, batch_size, profile.timestep)
      # Descriptors omit actuator counts; the live state owns na.
      self._rk4._na = na
    else:
      self._rk4 = None

    self._assembled_system_valid = False

    def _handle_state_snapshot(snap):
      # Attached solver seeds ride the documented StateSnapshot field
      # (validated by its __post_init__). The raw-device snapshot alone
      # owns kinematics/equality/mocap/activation only.
      if self._coupled_constraints is not None:
        warm = self._coupled_constraints.get_warmstart()
        object.__setattr__(snap, "warmstart_multiplier", warm)

    def _handle_state_validate_restore(snap, env_ids=None):
      # Runs BEFORE any live tensor is touched: a bad seed payload raises
      # with all state unchanged. None (unknown/old snapshots) is valid
      # and restores cold seeds.
      if self._coupled_constraints is None:
        if getattr(snap, "warmstart_multiplier", None) is not None:
          raise ValueError("snapshot seeds do not match this simulation")
        return
      nr = int(self._coupled_constraints.descriptor.nr)
      warm = getattr(snap, "warmstart_multiplier", None)
      if warm is None:
        return
      import numpy as _np
      arr = _np.asarray(warm, dtype=_np.float64)
      if arr.shape != (self.batch_size, nr) or not _np.all(_np.isfinite(arr)):
        raise ValueError("snapshot seeds have an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if not _np.all(_np.isfinite(arr.astype(_np.float32))):
          raise ValueError("snapshot seeds must be float32-representable")

    def _handle_state_restore(snap, env_ids=None):
      self._invalidate_forward_stage_record()
      self._assembled_system_valid = False
      if hasattr(self, "_last_coupled"):
        self._last_coupled = None
      if hasattr(self, "_last_coupled_generation"):
        self._last_coupled_generation = None
      self._accepted_step = None
      fk = getattr(getattr(self, "_smooth", None), "_fk", None)
      if fk is not None:
        fk.invalidate_cache(env_ids=env_ids)
      scheduler = getattr(self, "_sleep_schedule", None)
      if scheduler is not None:
        # A raw DeviceState restore has no sleep payload; wake selected rows.
        # Simulation.restore applies its validated saved tree cycles afterward.
        scheduler.wake_all(env_ids=env_ids)
      if self._coupled_constraints is not None:
        warm = getattr(snap, "warmstart_multiplier", None)
        if warm is not None:
          import numpy as _np
          self._coupled_constraints.set_warmstart(
              _np.asarray(warm, dtype=_np.float32)
              if env_ids is None else _np.asarray(warm, dtype=_np.float32)[_np.asarray(env_ids)],
              env_ids=env_ids)
        else:
          self._coupled_constraints.clear_warmstart(env_ids=env_ids)

    def _handle_state_reset(env_ids=None):
      self._invalidate_forward_stage_record()
      self._assembled_system_valid = False
      if hasattr(self, "_last_coupled"):
        self._last_coupled = None
      if hasattr(self, "_last_coupled_generation"):
        self._last_coupled_generation = None
      self._accepted_step = None
      fk = getattr(getattr(self, "_smooth", None), "_fk", None)
      if fk is not None:
        fk.invalidate_cache(env_ids=env_ids)
      scheduler = getattr(self, "_sleep_schedule", None)
      if scheduler is not None:
        scheduler.reset(env_ids=env_ids)
        self._seed_sleep_fk_reference(env_ids=env_ids)

    def _handle_state_masked_reset(mask):
      # Autorecovery is row scoped. Keep the batch-wide borrowed owners and
      # metadata alive; downstream code can distinguish reset rows through
      # the record's device row_valid mask. Public reset/restore still takes
      # the global invalidation path above.
      self._forward_stages.note_masked_reset(mask)
      fk = getattr(getattr(self, "_smooth", None), "_fk", None)
      if fk is not None:
        valid = fk._workspace["outputs"]["cache_valid"]
        self._state.clear_masked_rows(valid.reshape(self.batch_size, -1), mask)
      scheduler = getattr(self, "_sleep_schedule", None)
      if scheduler is not None:
        scheduler.reset_masked(mask)

    def _handle_eq_active_change(env_ids=None):
      # The next current-position prepass consumes this exact per-world mask;
      # equality-specific wake rules decide which island needs to run.
      self._invalidate_forward_stage_record()
      self._assembled_system_valid = False
      self._accepted_step = None
      self._last_coupled = None
      self._last_coupled_generation = None

    def _handle_environment_copy(src, dst):
      self._invalidate_forward_stage_record()
      fk = getattr(getattr(self, "_smooth", None), "_fk", None)
      if fk is not None:
        fk.invalidate_cache(env_ids=[dst])
        body_pos = fk._workspace["outputs"]["body_pos"].view(
            self.batch_size, self._mjmodel.nbody, 3)
        body_quat = fk._workspace["outputs"]["body_quat"].view(
            self.batch_size, self._mjmodel.nbody, 4)
        body_pos[dst].copy_(body_pos[src].clone())
        body_quat[dst].copy_(body_quat[src].clone())
      scheduler = getattr(self, "_sleep_schedule", None)
      if scheduler is not None:
        scheduler.tree_state[dst].copy_(scheduler.tree_state[src].clone())
        scheduler.tree_awake[dst].copy_(scheduler.tree_awake[src].clone())
        scheduler.status[dst] = 0
        scheduler.links_overflow[dst] = 0
      islands = getattr(self, "_islands", None)
      if islands is not None:
        islands.tree_asleep[dst] = islands.tree_asleep[src].copy()
        islands._stationary_steps[dst] = islands._stationary_steps[src].copy()
      self._assembled_system_valid = False
      self._accepted_step = None
      self._last_coupled = None
      self._last_coupled_generation = None

    self._state._on_snapshot = _handle_state_snapshot
    self._state._on_validate_restore = _handle_state_validate_restore
    self._state._on_restore = _handle_state_restore
    self._state._on_reset = _handle_state_reset
    self._state._on_masked_reset = _handle_state_masked_reset
    self._state._on_eq_active_change = _handle_eq_active_change
    self._state._on_environment_copy = _handle_environment_copy

    def _handle_mocap_change(env_ids=None):
      self._invalidate_forward_stage_record()
      ids = self._state._env_ids(env_ids)
      fk = getattr(getattr(self, "_smooth", None), "_fk", None)
      if fk is not None:
        fk.invalidate_cache(env_ids=ids)
      # Mocap bodies are static-tree participants in pinned mj_wakeCollision.
      # Recompute dependent world transforms, but let current contact/equality
      # policy decide whether any dynamic tree wakes.

    self._state._on_mocap_change = _handle_mocap_change
    # Canonical compiled history is the sole actuator/sensor ring storage.
    # Snapshot, partial reset and mjtState.HISTORY therefore share one ABI.
    from mujoco_metal.history import DeviceHistory
    self._history_program = DeviceHistory(model, batch_size, self._state._device)
    self._raw_sensordata = None
    self._raw_sensor_program = self._sensors
    if (self._sensors is not None and not self._history_program.sensor_enabled
        and bool(np.any(np.asarray(model.sensor_delay) > 0))):
      # Pinned mj_advance computes delayed raw samples even when the normal
      # forward sensor stage is disabled. Pure built-ins can share physics.
      reference = copy.copy(model)
      reference.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
      from mujoco_metal.sensors import SensorProgram
      self._raw_sensor_program = SensorProgram(reference, batch_size)
    self._delay = None  # Legacy snapshot key; no duplicate ring is maintained.

    from mujoco_metal.islands import IslandManager
    self._islands = IslandManager(model, batch_size=batch_size, device=self._state._device)
    # CoupledConstraints owns the canonical contact-to-tree map. Its capacity
    # includes the statically lowered flex-contact links as well as rigid
    # contacts and equality pairs. Do not replace it with a second rigid-only
    # map here: sleep scheduling consumes that same fixed link ABI.
    self._active_contact_link_workspace = (
        getattr(self._coupled_constraints,
                "_active_contact_link_workspace", None)
        if self._coupled_constraints is not None else None)

    self._sleep_schedule = None
    self._sensor_sleep_policy = None
    self._sleep_dof_treeid = None
    self._sleep_qvel = None
    if (int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        and self._state._device.type == "mps"):
      from mujoco_metal.sleep_schedule import DeviceSleepScheduler
      link_capacity = (self._active_contact_link_workspace.capacity
                       if self._active_contact_link_workspace is not None else 1)
      self._sleep_schedule = DeviceSleepScheduler(
          model, batch_size=batch_size, device=self._state._device,
          max_tree_links=link_capacity)
      # The first pinned kinematics pass compares candidate poses against
      # mj_resetData's retained xpose for articulated sleeping bodies.
      reset_data = mujoco.MjData(model)
      self._sleep_reset_body_pos = torch.as_tensor(
          np.asarray(reset_data.xpos, dtype=np.float32).copy(),
          dtype=torch.float32, device=self._state._device)
      self._sleep_reset_body_quat = torch.as_tensor(
          np.asarray(reset_data.xquat, dtype=np.float32).copy(),
          dtype=torch.float32, device=self._state._device)
      self._seed_sleep_fk_reference()
      self._sleep_dof_treeid = torch.as_tensor(
          np.asarray(model.dof_treeid, dtype=np.int32).copy(),
          dtype=torch.int32, device=self._state._device)
      self._sleep_qvel = torch.empty_like(self._state._qvel)
      self._sleep_zero = torch.zeros((), dtype=torch.float32, device=self._state._device)
      self._sleep_fk_previous = torch.empty_like(
          self._sleep_schedule.tree_awake)
      self._sleep_fk_refresh = torch.empty_like(
          self._sleep_schedule.tree_awake)
      if self._sensors is not None:
        from mujoco_metal.sensor_sleep import SensorSleepPolicy
        self._sensor_sleep_policy = SensorSleepPolicy(
            model, batch_size=batch_size, device=self._state._device)

    from mujoco_metal.extensions import (
        default_registry, NativeActuatorPlugin, NativeActuatorUserPlugin,
        PluginType)
    self._native_plugins = default_registry.instantiate(
        model, batch_size=batch_size, device=self._state._device,
        registrations=plugin_registrations)
    _check_plugin_device_rollback(self._native_plugins)
    # MuJoCo distinguishes passive-force plugins from actuator plugins. Keep
    # the roles separate so ordinary and prepared forward paths publish the
    # same qfrc_passive/qfrc_actuator decomposition.
    self._force_plugins, self._actuator_plugins = _partition_force_plugins(
        self._native_plugins)
    self._native_actuator_plugins = tuple(
        p for p in self._actuator_plugins
        if isinstance(p, NativeActuatorPlugin)
        and not isinstance(p, NativeActuatorUserPlugin))
    self._actuator_user_plugins = tuple(
        p for p in self._actuator_plugins
        if isinstance(p, NativeActuatorUserPlugin))
    self._legacy_actuator_plugins = tuple(
        p for p in self._actuator_plugins
        if not isinstance(p, (NativeActuatorPlugin,
                              NativeActuatorUserPlugin)))
    self._native_actuator_force = None
    self._native_actuator_qfrc = None
    self._last_native_actuator_outputs = {}
    if self._native_actuator_plugins:
      self._native_actuator_force = torch.zeros(
          (batch_size, int(model.nu)), dtype=torch.float32,
          device=self._state._device)
      self._native_actuator_qfrc = torch.zeros_like(self._rhs)
    if self._actuator_user_plugins:
      if self._actuators is None:
        raise ValueError("actuator USER callbacks require the native general actuator path")
      self._actuators.bind_user_callback_plugins(self._actuator_user_plugins)

  @property
  def islands(self):
    """Kinematic island discovery and sleep/wake manager (milestone 017)."""
    return self._islands

  @property
  def state(self):
    """The owned :class:`DeviceState` lifecycle and checkpoint interface."""
    return self._state

  @property
  def model(self):
    """The underlying MuJoCo MjModel."""
    return self._mjmodel

  @property
  def device(self):
    """The device backing simulation state."""
    return self._state.device

  @property
  def energy(self):
    """Pinned ``[potential, kinetic]`` energy from the latest evaluated stages."""
    return None if self._energy is None else self._energy.clone()

  def _invalidate_after_state_write(self, fields):
    """Advance state identity and invalidate records derived from a state transaction."""
    state = self._state
    self._invalidate_forward_stage_record()
    state._generation += 1
    self._assembled_system_valid = False
    self._accepted_step = None
    self._last_coupled = None
    self._last_coupled_generation = None
    self._spatial_cache_key = None
    self._spatial_kin = None
    if self._flex is not None and ("qpos" in fields or "qvel" in fields or
                                    "mocap_pos" in fields or "mocap_quat" in fields):
      dyn = self._smooth.run_device(
          state._qpos, state._qvel, state._mpos, state._mquat)
      self._flex.update_kinematics(dict(dyn["poses"], root_com=dyn["root_com"]), dyn.get("cvel"))
    if self._spatial_tendons is not None and ("qpos" in fields or "qvel" in fields or
                                               "mocap_pos" in fields or "mocap_quat" in fields):
      dyn = self._smooth.run_device(
          state._qpos, state._qvel, state._mpos, state._mquat)
      self._spatial_kin = self._spatial_tendons.run_kinematics(
          state._qvel, dyn["poses"])
      self._spatial_cache_key = (state.generation, id(state._qvel),
                                 id(dyn["poses"].get("body_pos")))

  def _invalidate_forward_stage_record(self):
    """Drop pending split-step state after an owned state mutation."""
    self._step1_record = None
    stages = getattr(self, "_forward_stages", None)
    if stages is not None:
      stages.invalidate()

  @property
  def execution_plan(self):
    """The model-derived execution plan for the active stepping profile."""
    return self.profile.execution_plan

  @property
  def solver_settings(self):
    """The explicit solver configuration for the coupled constraint solve."""
    if self._coupled_constraints is not None:
      return self._coupled_constraints.solver_settings
    return None

  def get_warmstart(self):
    """Return a host copy of ``qacc_warmstart`` with shape ``(batch, nv)``.

    This is the canonical pinned MuJoCo warm-start state used by PGS, CG, and
    Newton. Retained constraint multipliers are diagnostics, not an input to
    the native solver.
    """
    return self._state._qacc_warmstart.detach().cpu().numpy().copy()

  def set_warmstart(self, values, env_ids=None):
    """Set selected worlds' ``qacc_warmstart`` transactionally.

    `values` accepts host arrays or device tensors shaped `(nv,)` (broadcast),
    `(len(env_ids), nv)`, or `(batch, nv)`. All three native solver families
    use this acceleration state. For the pinned selector API, the equivalent
    operation is `mj_setState(sim, {"warmstart": values}, env_ids=...)`.
    """
    from mujoco_metal.native_api import _selected_ids, mj_setState
    _, count = _selected_ids(self, env_ids)
    if count == 0:
      raise ValueError("env_ids must select at least one world")
    torch = self._state._torch
    nv = int(self._mjmodel.nv)
    shape = tuple(values.shape) if hasattr(values, "shape") else tuple(np.asarray(values).shape)
    if shape == (nv,):
      if isinstance(values, torch.Tensor):
        values = values.reshape(1, nv).expand(count, nv)
      else:
        values = np.broadcast_to(np.asarray(values), (count, nv)).copy()
    mj_setState(self, {"warmstart": values}, env_ids=env_ids)
    return self._state.generation

  def clear_warmstart(self, env_ids=None):
    """Clear selected worlds' canonical acceleration warm-start state."""
    nv = int(self._mjmodel.nv)
    return self.set_warmstart(np.zeros((nv,), dtype=np.float32), env_ids=env_ids)

  def get_constraint_multipliers(self):
    """Return the latest coupled-solver multiplier diagnostics.

    These are the last assembled constraint forces, shaped `(batch, nr)`;
    they are not used as solver warm-start input. Use `get_warmstart` or the
    pinned `StateSpec.WARMSTART` selector for the canonical acceleration seed.
    """
    if self._coupled_constraints is None:
      if self._joint_constraints is not None:
        rows = int(self._joint_constraints.descriptor.nrow)
        return self._joint_constraints.get_multipliers()[:, :rows]
      return np.zeros((self.batch_size, 0), dtype=np.float32)
    return self._coupled_constraints.get_warmstart()

  def set_equality_active(self, values, env_ids=None):
    """Set persistent equality activity for selected environments.

    `values` accepts host boolean/integer arrays with shape `(neq,)` (broadcast
    to selected worlds) or `(len(env_ids), neq)`, or contiguous int32/bool MPS
    tensors with the same shapes (copied, never borrowed). `env_ids=None`
    selects all worlds. Validation is atomic: bad input leaves all worlds
    unchanged. Changes take effect on the next step/assembly, invalidate cached
    assembly, and do not clear sticky failure status. Works for joint, connect
    and weld equalities; reattaching uses the compiled reference, not a
    recaptured pose.
    """
    if self._coupled_constraints is None:
      # No coupled stage (e.g., constraints disabled): only allow empty/no-op?
      # Preserve no-equality support: raise unless model truly has no equalities.
      neq = int(self._mjmodel.neq) if isinstance(self._mjmodel, mujoco.MjModel) else 0
      if neq == 0:
        # Accept empty calls for uniformity? Reject non-empty to avoid silent loss.
        try:
          arr = values.detach().cpu().numpy() if hasattr(values, "detach") else __import__("numpy").asarray(values)
        except Exception:
          raise ValueError("model has no equalities")
        if np.asarray(arr).size != 0:
          raise ValueError("model has no equalities")
        return self._state.generation
      raise ValueError("set_equality_active requires a coupled constraint profile")
    gen = self._state.set_equality_active(values, env_ids=env_ids)
    # Invalidate cached assembly so next assembled_system recomputes with new
    # activity and fresh diagnostics. Do not touch failure status.
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return gen

  def set_mocap(self, pos, quat=None, env_ids=None):
    """Set prescribed mocap poses for selected environments.

    Mirrors `DeviceState.set_mocap`: host arrays or contiguous float32 MPS
    tensors, `(nmocap, 3/4)` broadcast or `(len(env_ids), nmocap, 3/4)` per
    world, or a single `(…, nmocap, 7)` posquat array with `quat=None`.
    Validation is atomic; sticky failure status is preserved; changes take
    effect on the next step/assembly and invalidate cached assembly.
    """
    if getattr(self._state, "_nmocap", 0) == 0:
      raise ValueError("model has no mocap bodies")
    gen = self._state.set_mocap(pos, quat, env_ids=env_ids)
    fk = getattr(getattr(self, "_smooth", None), "_fk", None)
    if fk is not None:
      fk.invalidate_cache(env_ids=(None if env_ids is None
                                   else self._state._env_ids(env_ids)))
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return gen

  def get_mocap(self):
    """Return `(pos, quat)` copies of prescribed mocap poses, or `(None, None)`."""
    state = self._state
    if getattr(state, "_nmocap", 0) == 0:
      return None, None
    return state.mocap_pos, state.mocap_quat

  def copy_environment(self, src, dst):
    """Copy all state rows (qpos/qvel/qacc/time/status/eq/mocap) src -> dst."""
    for name, value in (("src", src), ("dst", dst)):
      raw = np.asarray(value)
      if raw.shape != () or raw.dtype.kind not in "iu":
        raise ValueError("src/dst must be integer environment indices")
    src_i, dst_i = int(np.asarray(src)), int(np.asarray(dst))
    for index in (src_i, dst_i):
      if not 0 <= index < self.batch_size:
        raise ValueError("environment index out of range")
    plugin_map = {_plugin_state_key(p): p for p in getattr(self, "_native_plugins", ())}
    plugin_targets = {}
    for key, plugin in plugin_map.items():
      payload = plugin.snapshot()
      plugin_targets[key] = _copy_plugin_row(payload, src_i, dst_i, self.batch_size)
    active_plugins = {key: plugin_map[key] for key, value in plugin_targets.items()
                      if value is not None}
    active_targets = {key: plugin_targets[key] for key in active_plugins}
    _preflight_plugin_restores(active_plugins, active_targets, env_ids=[dst_i])
    # A restore implementation can still fail on its later, real commit call.
    # Apply it transactionally before touching any simulation-owned rows.
    _apply_plugin_restores(active_plugins, active_targets, env_ids=[dst_i])
    gen = self._state.copy_environment(src, dst)
    fk = getattr(getattr(self, "_smooth", None), "_fk", None)
    if fk is not None:
      fk.invalidate_cache(env_ids=[dst_i])
    # Full simulation state follows the device rows: retained warmstarts,
    # stored sensor samples and held per-call inputs. Caches invalidate.
    cc = getattr(self, "_coupled_constraints", None)
    if cc is not None and int(cc.descriptor.nr) > 0:
      w = cc._workspace["workspace_debug"].reshape(
          self.batch_size, cc._debug_stride)
      nr = cc.descriptor.nr
      w[dst_i, nr * nr + 3 * nr:nr * nr + 4 * nr] = w[src_i, nr * nr + 3 * nr:nr * nr + 4 * nr].clone()
    if getattr(self, "_sensordata", None) is not None:
      self._sensordata[dst_i] = self._sensordata[src_i].clone()
    seen_low = set()
    for program in (getattr(self, "_sensors", None),
                    getattr(self, "_raw_sensor_program", None)):
      if program is None or id(program) in seen_low:
        continue
      seen_low.add(id(program))
      low = getattr(program, "_s_state_out_low", None)
      if low is not None:
        low[dst_i] = low[src_i].clone()
    for held in ("_control", "_body_wrench", "_applied_force"):
      tensor = getattr(self, held, None)
      if tensor is not None:
        tensor[dst_i] = tensor[src_i].clone()
    if getattr(self, "_delay", None) is not None:
      self._delay.copy_row(src_i, dst_i)
      self._sync_history_from_delay(env_ids=[dst_i])
    if getattr(self, "_delay", None) is not None:
      self._delay.copy_row(src_i, dst_i)
    if getattr(self, "_islands", None) is not None:
      self._islands.tree_asleep[dst_i] = self._islands.tree_asleep[src_i].copy()
      self._islands._stationary_steps[dst_i] = self._islands._stationary_steps[src_i].copy()
    if getattr(self, "_sleep_schedule", None) is not None:
      self._sleep_schedule.tree_state[dst_i].copy_(
          self._sleep_schedule.tree_state[src_i].clone())
      self._sleep_schedule.tree_awake[dst_i].copy_(
          self._sleep_schedule.tree_awake[src_i].clone())
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return gen

  def _assert_contact_override_binding(self):
    """Reject live option drift before touching state or cached workspaces."""
    expected = getattr(self, "_contact_override_binding", None)
    if expected is None:
      # Narrow test doubles and legacy stand-ins do not own a compiled model.
      return
    if _contact_override_option_signature(self._mjmodel) != expected:
      raise ValueError(
          "global contact override options changed after compilation; "
          "call apply_lifecycle to rebuild the simulation")

  def snapshot(self):
    """Capture a versioned simulation-level checkpoint (host, immutable).

    Covers everything needed for exact replay: the device state, retained
    constraint multipliers, held per-call inputs, stored sensor samples
    and actuator delay rings. The narrower :meth:`DeviceState.snapshot`
    owns only kinematic/equality/mocap/activation rows; use this method
    for full trajectory replay.
    """
    self._assert_contact_override_binding()
    import numpy as _np
    torch = self._state._torch
    snap = {
        "schema_version": 1,
        "model_fingerprint": self._state._model_fingerprint,
        "profile_fingerprint": self._state._profile_fingerprint,
        "batch_size": self.batch_size,
        "nr": 0,
        "nsensordata": int(self._mjmodel.nsensordata),
        "nu": int(self._mjmodel.nu),
        "nv": int(self._mjmodel.nv),
        "nbody": int(self._mjmodel.nbody),
        "device": self._state.snapshot(),
        "native_state": {
            name: getattr(self._state, attr).detach().cpu().numpy().copy()
            for name, attr in (("history", "_history"),
                               ("qacc_warmstart", "_qacc_warmstart"),
                               ("userdata", "_userdata"),
                               ("plugin_state", "_plugin_state"))
        },
    }
    cc = getattr(self, "_coupled_constraints", None)
    if cc is not None and int(cc.descriptor.nr) > 0:
      nr = int(cc.descriptor.nr)
      snap["nr"] = nr
      snap["warmstart"] = cc.get_warmstart().astype(_np.float32, copy=True)
    else:
      snap["warmstart"] = None
    held = {}
    for name, tensor in (("control", self._control),
                         ("wrench", self._body_wrench),
                         ("force", self._applied_force)):
      held[name] = (tensor.detach().cpu().numpy().copy() if tensor is not None else None)
    snap["held"] = held
    snap["sensordata"] = (self._sensordata.detach().cpu().numpy().copy()
                          if getattr(self, "_sensordata", None) is not None else None)
    snap["sensordata_low"] = self._sensor_output_low_snapshot(
        getattr(self, "_sensors", None))
    raw_program = getattr(self, "_raw_sensor_program", None)
    snap["raw_sensordata_low"] = (
        self._sensor_output_low_snapshot(raw_program)
        if raw_program is not getattr(self, "_sensors", None) else None)
    snap["delay"] = (self._delay.snapshot()
                     if getattr(self, "_delay", None) is not None else None)
    snap["islands"] = ({
        "tree_asleep": self._islands.tree_asleep.copy(),
        "_stationary_steps": self._islands._stationary_steps.copy(),
    } if getattr(self, "_islands", None) is not None else None)
    snap["sleep_schedule"] = (
        self._sleep_schedule.snapshot()
        if getattr(self, "_sleep_schedule", None) is not None else None)
    snap["flex"] = (self._flex.get_state()
                    if getattr(self, "_flex", None) is not None else None)
    snap["plugins"] = {_plugin_state_key(p): p.snapshot()
                       for p in getattr(self, "_native_plugins", ())}
    return snap

  def restore(self, snapshot, env_ids=None):
    """Restore a simulation snapshot, fully or for selected worlds.

    Every field is validated before anything is committed: a bad snapshot
    (or bad env selection) raises with all live state untouched. Warm
    starts, held inputs and stored sensor samples follow the same rows as
    the device state. Restoring never clears retained seeds beyond what the
    snapshot carries; use :meth:`reset` for a cold start.
    """
    self._assert_contact_override_binding()
    import numpy as _np
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
      raise ValueError("unsupported simulation snapshot schema")
    if (snapshot.get("model_fingerprint") != self._state._model_fingerprint
            or snapshot.get("profile_fingerprint") != self._state._profile_fingerprint
            or snapshot.get("batch_size") != self.batch_size):
      raise ValueError("snapshot model, profile, or batch size do not match")
    ids = self._state._env_ids(env_ids) if env_ids is not None else None
    if ids is not None and ids.size == 0:
      raise ValueError("env_ids must select at least one world")
    nr = int(snapshot.get("nr", 0))
    cc = getattr(self, "_coupled_constraints", None)
    cc_nr = int(cc.descriptor.nr) if cc is not None else 0
    if nr != cc_nr:
      raise ValueError("snapshot constraint-row dimensions do not match")
    warm = snapshot.get("warmstart")
    warm32 = None
    if nr > 0:
      warm = _np.asarray(warm, dtype=_np.float64)
      if warm.shape != (self.batch_size, nr) or not _np.all(_np.isfinite(warm)):
        raise ValueError("snapshot warmstart has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        warm32 = _np.asarray(warm, dtype=_np.float32)
      if not _np.all(_np.isfinite(warm32)):
        raise ValueError("snapshot warmstart values overflow float32")
    elif warm is not None:
      raise ValueError("snapshot warmstart must be None without constraint rows")
    if int(snapshot.get("nsensordata", -1)) != int(self._mjmodel.nsensordata):
      raise ValueError("snapshot sensor dimensions do not match")
    if (int(snapshot.get("nu", -1)) != int(self._mjmodel.nu)
            or int(snapshot.get("nv", -1)) != int(self._mjmodel.nv)
            or int(snapshot.get("nbody", -1)) != int(self._mjmodel.nbody)):
      raise ValueError("snapshot model dimensions do not match")
    # Validate all device state before any extension restore callback.
    self._state.validate_snapshot(snapshot.get("device"), env_ids=ids)
    plugins = snapshot.get("plugins")
    plugin_map = {_plugin_state_key(p): p for p in getattr(self, "_native_plugins", ())}
    if not isinstance(plugins, dict) or set(plugins) != set(plugin_map):
      raise ValueError("snapshot plugin roles/names do not match this simulation")
    for key, plugin in plugin_map.items():
      validator = getattr(plugin, "validate_snapshot", None)
      if callable(validator):
        validator(plugins[key], env_ids=ids)
    # Some third-party restore implementations validate while writing. Probe
    # their payloads and restore their prior state before committing any
    # simulation-owned tensor, so a late rejecting plugin is transactional.
    _preflight_plugin_restores(plugin_map, plugins, env_ids=ids)
    held = snapshot.get("held", {})
    held_checked = {}
    for name, tensor in (("control", self._control),
                         ("wrench", self._body_wrench),
                         ("force", self._applied_force)):
      value = held.get(name)
      if tensor is None:
        if value is not None:
          raise ValueError(f"snapshot held {name} does not match this simulation")
        held_checked[name] = None
        continue
      if value is None:
        raise ValueError(f"snapshot held {name} is missing")
      arr = _np.asarray(value, dtype=_np.float64)
      if arr.shape != tuple(tensor.shape) or not _np.all(_np.isfinite(arr)):
        raise ValueError(f"snapshot held {name} has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        arr32 = _np.asarray(arr, dtype=_np.float32)
      if not _np.all(_np.isfinite(arr32)):
        raise ValueError(f"snapshot held {name} values overflow float32")
      held_checked[name] = arr32.copy()
    native_state = snapshot.get("native_state")
    native_checked = {}
    if not isinstance(native_state, dict) or set(native_state) != {
        "history", "qacc_warmstart", "userdata", "plugin_state"}:
      raise ValueError("snapshot native state groups are missing or unknown")
    for name, attr in (("history", "_history"),
                       ("qacc_warmstart", "_qacc_warmstart"),
                       ("userdata", "_userdata"),
                       ("plugin_state", "_plugin_state")):
      tensor = getattr(self._state, attr)
      arr = _np.asarray(native_state[name], dtype=_np.float64)
      if arr.shape != tuple(tensor.shape) or not _np.all(_np.isfinite(arr)):
        raise ValueError(f"snapshot {name} has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        arr32 = _np.asarray(arr, dtype=_np.float32)
      if not _np.all(_np.isfinite(arr32)):
        raise ValueError(f"snapshot {name} values overflow float32")
      native_checked[name] = arr32.copy()
    sens = snapshot.get("sensordata")
    sens_checked = None
    if sens is not None:
      sens_arr = _np.asarray(sens, dtype=_np.float64)
      if sens_arr.shape != (self.batch_size, int(self._mjmodel.nsensordata)) or not _np.all(_np.isfinite(sens_arr)):
        raise ValueError("snapshot sensor sample has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        sens32 = _np.asarray(sens_arr, dtype=_np.float32)
      if not _np.all(_np.isfinite(sens32)):
        raise ValueError("snapshot sensor sample values overflow float32")
      sens_checked = sens32.copy()

    def validate_sensor_low(name):
      value = snapshot.get(name)
      if value is None:
        return None
      array = _np.asarray(value, dtype=_np.float64)
      expected = (self.batch_size, int(self._mjmodel.nsensordata))
      if array.shape != expected or not _np.all(_np.isfinite(array)):
        raise ValueError(f"snapshot {name} has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value32 = _np.asarray(array, dtype=_np.float32)
      if not _np.all(_np.isfinite(value32)):
        raise ValueError(f"snapshot {name} values overflow float32")
      return value32.copy()

    sens_low_checked = validate_sensor_low("sensordata_low")
    raw_sens_low_checked = validate_sensor_low("raw_sensordata_low")

    # Delay rings (R06/D1): validated pre-commit like everything else. A
    # snapshot without delay state restores cold rings; a model without
    # delay lines must not carry ring state.
    delay_checked = None
    delay_snap = snapshot.get("delay", None)
    if getattr(self, "_delay", None) is None:
      if delay_snap is not None:
        raise ValueError("snapshot delay state does not match this simulation")
    else:
      if delay_snap is None:
        raise ValueError("snapshot delay state is missing")
      delay_checked = self._delay.checked_snapshot(delay_snap)

    # Apply actual callbacks before the native commit: implementations may
    # reject only on a later call after the preflight probe succeeded.
    _apply_plugin_restores(plugin_map, plugins, env_ids=ids)
    # Commit boundary: DeviceState validates its fields atomically before mutating.
    self._state.restore(snapshot["device"], env_ids=ids)
    torch = self._state._torch
    for name, attr in (("history", "_history"),
                       ("qacc_warmstart", "_qacc_warmstart"),
                       ("userdata", "_userdata"),
                       ("plugin_state", "_plugin_state")):
      tensor = getattr(self._state, attr)
      val = torch.as_tensor(native_checked[name], dtype=torch.float32,
                            device=self._state._device)
      if ids is None:
        tensor.copy_(val)
      else:
        tensor[ids] = val[ids]
    if nr > 0 and warm32 is not None:
      if ids is None:
        cc.set_warmstart(warm32)
      else:
        cc.set_warmstart(warm32[ids], env_ids=ids)
    for name, tensor in (("control", self._control),
                         ("wrench", self._body_wrench),
                         ("force", self._applied_force)):
      if tensor is not None:
        val_t = torch.as_tensor(held_checked[name], dtype=torch.float32, device=self._state._device)
        if ids is None:
          tensor.copy_(val_t)
        else:
          tensor[ids] = val_t[ids]
    if sens_checked is not None:
      if self._sensordata is None:
        self._sensordata = torch.zeros(
            (self.batch_size, int(self._mjmodel.nsensordata)),
            dtype=torch.float32, device=self._state._device)
      val_s = torch.as_tensor(sens_checked, dtype=torch.float32, device=self._state._device)
      if ids is None:
        self._sensordata.copy_(val_s)
      else:
        self._sensordata[ids] = val_s[ids]
    else:
      if ids is None:
        self._sensordata = None
      elif self._sensordata is not None:
        self._sensordata[ids] = 0.0
    self._restore_sensor_output_low(
        getattr(self, "_sensors", None), sens_low_checked, ids)
    raw_program = getattr(self, "_raw_sensor_program", None)
    if raw_program is not getattr(self, "_sensors", None):
      self._restore_sensor_output_low(
          raw_program, raw_sens_low_checked, ids)
    if delay_checked is not None:
      if ids is None:
        self._delay.restore(delay_checked)
      else:
        cur = self._delay.snapshot()
        nmax = self._delay.nmax
        nu = self._delay.nu
        for row in np.asarray(ids).tolist():
          cur["cursor"].reshape(self.batch_size, nu)[row] = \
              delay_checked["cursor"].reshape(self.batch_size, nu)[row].copy()
          cur["times"].reshape(self.batch_size, nu, nmax)[row] = \
              delay_checked["times"].reshape(self.batch_size, nu, nmax)[row].copy()
          cur["values"].reshape(self.batch_size, nu, nmax)[row] = \
              delay_checked["values"].reshape(self.batch_size, nu, nmax)[row].copy()
        self._delay.restore(cur)
      self._sync_delay_from_history(env_ids=ids)
    if getattr(self, "_islands", None) is not None and "islands" in snapshot and snapshot["islands"] is not None:
      if ids is None:
        self._islands.tree_asleep[:] = snapshot["islands"]["tree_asleep"]
        self._islands._stationary_steps[:] = snapshot["islands"]["_stationary_steps"]
      else:
        for row in np.asarray(ids).tolist():
          self._islands.tree_asleep[row] = snapshot["islands"]["tree_asleep"][row]
          self._islands._stationary_steps[row] = snapshot["islands"]["_stationary_steps"][row]
    if getattr(self, "_sleep_schedule", None) is not None:
      sleep_snapshot = snapshot.get("sleep_schedule")
      if sleep_snapshot is None:
        self._sleep_schedule.wake_all(env_ids=ids)
      else:
        self._sleep_schedule.restore(sleep_snapshot, env_ids=ids)
      # Restored sleep cycles can contain asleep trees while the FK workspace
      # cache belongs to a different trajectory. Rebuild every selected pose
      # before the next masked prepass, then retain the restored sleep state.
      refresh = self._sleep_fk_refresh
      refresh.fill_(1)
      self._smooth._fk.run_device(
          self._state._qpos, getattr(self._state, "_mpos", None),
          getattr(self._state, "_mquat", None), tree_awake=refresh)
    if getattr(self, "_flex", None) is not None:
      if "flex" in snapshot and snapshot["flex"] is not None:
        self._flex.set_state(snapshot["flex"], env_ids=env_ids)
      else:
        dyn = self._smooth.run_device(self._state.qpos, self._state.qvel)
        self._flex.update_kinematics(dict(dyn["poses"], root_com=dyn["root_com"]), dyn.get("cvel"))
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    self._assembled_system_valid = False
    self._accepted_step = None
    self._step1_record = None
    return self._state.generation

  def reset(self, env_ids=None, qpos=None, qvel=None, eq_active=None,
            mocap_pos=None, mocap_quat=None, act=None):
    """Reset selected worlds, clear held per-call inputs, invalidate cache.

    Reset is cold: retained warmstart multipliers and delay rings are
    cleared alongside state, matching pinned reset semantics.
    """
    prepared = self._state.prepare_reset(
        env_ids=env_ids, qpos=qpos, qvel=qvel, eq_active=eq_active,
        mocap_pos=mocap_pos, mocap_quat=mocap_quat, act=act,
    )
    if not prepared[0].size:
      return self._state.generation
    # Reset plugin-owned state before committing simulation-owned state. A
    # plugin may mutate its state and then raise; its prior snapshot is restored
    # transactionally while qpos, histories, warmstarts and held inputs remain
    # untouched.
    _reset_plugins_transactionally(getattr(self, "_native_plugins", ()),
                                   env_ids=env_ids)
    gen = self._state.reset(_prepared=prepared)
    if self._energy is not None:
      if env_ids is None:
        self._energy.zero_()
      else:
        ids = self._state._env_ids(env_ids)
        if ids.size:
          self._energy[ids.tolist()] = 0.0
    self._clear_held_inputs(env_ids)
    if getattr(self, "_coupled_constraints", None) is not None:
      self._coupled_constraints.clear_warmstart(env_ids=env_ids)
    if getattr(self, "_delay", None) is not None:
      self._delay.reset(env_ids=env_ids)
      self._sync_delay_from_history(env_ids=env_ids)
    if getattr(self, "_sensordata", None) is not None:
      # Stored step samples are state: reset zeroes selected worlds.
      if env_ids is None:
        self._sensordata.zero_()
      else:
        ids = self._state._env_ids(env_ids)
        if ids.size:
          self._sensordata[ids.tolist()] = 0.0
    self._clear_sensor_output_low(
        None if env_ids is None else self._state._env_ids(env_ids))
    if getattr(self, "_islands", None) is not None:
      self._islands.wake_all(env_ids=env_ids)
    # DeviceState.reset invokes the registered `_handle_state_reset` callback,
    # which resets the scheduler and seeds retained FK poses for these rows.
    # Do not reset it a second time here: that would clear the just-established
    # reset/cache epoch independently of the state transaction.
    if getattr(self, "_flex", None) is not None:
      dyn = self._smooth.run_device(self._state.qpos, self._state.qvel)
      self._flex.update_kinematics(dict(dyn["poses"], root_com=dyn["root_com"]), dyn.get("cvel"))
    self._assembled_system_valid = False
    self._accepted_step = None
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    self._step1_record = None
    return gen

  def _reset_from_check(self, reset_mask, *, forward_acc=False,
                        warning_commit=None):
    """Apply source AUTORESET to selected device worlds without host IDs."""
    torch = self._state._torch
    if (not isinstance(reset_mask, torch.Tensor)
        or reset_mask.dtype != torch.bool
        or tuple(reset_mask.shape) != (self.batch_size,)
        or reset_mask.device != self._state._qvel.device
        or not reset_mask.is_contiguous()):
      raise ValueError("reset mask must be contiguous bool[batch_size] on the state device")

    _reset_plugins_device_masked(
        tuple(getattr(self, "_native_plugins", ())), reset_mask)
    self._state.reset_mask(reset_mask)
    if warning_commit is not None:
      warning_commit()

    def clear_rows(value):
      if value is None or not isinstance(value, torch.Tensor) or not value.numel():
        return
      self._state.clear_masked_rows(value, reset_mask)

    for name in ("_control", "_body_wrench", "_applied_force", "_energy",
                 "_sensordata", "_raw_sensordata", "_next_act", "_act_dot",
                 "_act_vel", "_sen_act_force", "_sen_qfrc_act",
                 "_sensor_plugin_status", "_combined_status", "_next_status"):
      clear_rows(getattr(self, name, None))
    seen = set()
    for program in (getattr(self, "_sensors", None),
                    getattr(self, "_raw_sensor_program", None)):
      if program is None or id(program) in seen:
        continue
      seen.add(id(program))
      clear_rows(getattr(program, "_s_state_out_low", None))

    cc = getattr(self, "_coupled_constraints", None)
    if cc is not None and int(cc.descriptor.nr):
      nr = int(cc.descriptor.nr)
      workspace = cc._workspace["workspace_debug"].reshape(
          self.batch_size, cc._debug_stride)
      self._state.clear_masked_rows(workspace, reset_mask)
    delay = getattr(self, "_delay", None)
    if delay is not None:
      cursor = delay._cursor.reshape(self.batch_size, delay.nu)
      self._state.clear_masked_rows(cursor, reset_mask)
      self._state.clear_masked_rows(
          delay._times.reshape(self.batch_size, delay.nu, delay.nmax), reset_mask)
      self._state.clear_masked_rows(
          delay._values.reshape(self.batch_size, delay.nu, delay.nmax), reset_mask)
      clear_rows(delay._out.reshape(self.batch_size, delay.nu))

    # Row-local caches remain allocated. The row epoch/validity mask marks
    # reset rows stale; healthy rows retain their exact stage and solver views.

    if forward_acc:
      plugin_before = _snapshot_plugins_device(
          tuple(getattr(self, "_native_plugins", ())))
      warm_before = None
      if cc is not None and int(cc.descriptor.nr):
        nr = int(cc.descriptor.nr)
        workspace = cc._workspace["workspace_debug"].reshape(
            self.batch_size, cc._debug_stride)
        offset = nr * nr + 3 * nr
        warm_before = workspace[:, offset:offset + nr].clone()
      scheduler = getattr(self, "_sleep_schedule", None)
      awake_lists = scheduler.awake_lists() if scheduler is not None else None
      try:
        acceleration, status, dynamics = self._acceleration(
            self._state._qpos, self._state._qvel, forward_mask=reset_mask,
            skip_sleep_prepare=True, awake_lists_override=awake_lists,
            awake_solve_tree=(scheduler.tree_awake if scheduler is not None
                              else None))
      except Exception:
        _restore_plugins_device(plugin_before)
        if warm_before is not None:
          workspace[:, offset:offset + nr].copy_(warm_before)
        raise
      _restore_failed_plugin_rows(plugin_before, reset_mask)
      if warm_before is not None:
        self._state.copy_masked_strided_rows(
            workspace, warm_before, ~reset_mask, offset=offset)
      self._update_energy_position(self._state._qpos, dynamics, None,
                                   world_mask=reset_mask)
      self._update_energy_velocity({"dynamics": dynamics,
                                    "qvel": self._state._qvel},
                                   world_mask=reset_mask)
      self._state.copy_masked_rows(
          self._state._qacc, acceleration, reset_mask)
      self._state.copy_masked_rows(
          self._state._status, status, reset_mask)
      if self._sensordata is not None:
        self._store_step_sensors(
            self._state._qpos, self._state._qvel, dynamics,
            qacc=acceleration,
            qacc_low=(getattr(self, "_last_recovery_qacc_low", None)
                      if getattr(self, "_last_recovery_qacc_low", None) is not None
                      else self._qacc_low_for_sensor(acceleration)),
            compute_worlds=reset_mask,
            stages=(mujoco.mjtStage.mjSTAGE_POS,
                    mujoco.mjtStage.mjSTAGE_VEL,
                    mujoco.mjtStage.mjSTAGE_ACC))
    return self._state._generation

  def _check_step_acceleration(self, acceleration, qvel, status):
    """Run pinned post-forward ACC check and merge reset rows for integration."""
    from mujoco_metal.state_checks import mj_checkAcc
    torch = self._state._torch
    # `mj_checkAcc` may invoke `_reset_from_check`, which runs `_acceleration`
    # again and reuses the borrowed result/status views passed here. Keep the
    # accepted source values in step-owned buffers before entering that path.
    # `_next_qacc` and `_next_status` are not consumed until integration has
    # finished with this checked result.
    self._next_qacc.copy_(acceleration)
    self._next_status.copy_(status)
    source_low = self._qacc_low_for_sensor(acceleration)
    if source_low is None:
      self._check_qacc_low.zero_()
    else:
      self._check_qacc_low.copy_(source_low)
    result = mj_checkAcc(self, qacc=acceleration)
    reset_enabled = not bool(
        int(self._mjmodel.opt.disableflags)
        & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET))
    if not reset_enabled:
      return acceleration, qvel, status, self._state._torch.zeros_like(
          result["bad_world"])
    mask = result["bad_world"]
    acceleration = torch.where(mask[:, None], self._state._qacc,
                                self._next_qacc)
    qvel = torch.where(mask[:, None], self._state._qvel, qvel)
    status = torch.where(mask, self._state._status, self._next_status)
    if source_low is not None:
      recovery_low = getattr(self, "_last_recovery_qacc_low", None)
      if (not isinstance(recovery_low, torch.Tensor)
          or tuple(recovery_low.shape) != tuple(acceleration.shape)):
        recovery_low = torch.zeros_like(self._check_qacc_low)
      merged_low = torch.where(mask[:, None], recovery_low,
                               self._check_qacc_low)
      self._cache_sensor_acceleration(acceleration, merged_low)
    return acceleration, qvel, status, mask

  def reset_to_keyframe(self, key_id, env_ids=None):
    """Reset selected worlds from a compiled keyframe (pinned mj_resetDataKeyframe).

    Sets time to `key_time`, qpos/qvel/mocap to the key values, qacc/status to
    zero, equality activity to compiled defaults, and held controls to `key_ctrl`
    for the selected worlds. Invalid key ids fail atomically before any mutation.
    """
    model = self._mjmodel
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("keyframe reset requires a compiled MjModel")
    if isinstance(key_id, bool) or not isinstance(key_id, (int, np.integer)):
      raise TypeError("key_id must be an integer")
    key_id = int(key_id)
    if not 0 <= key_id < int(model.nkey):
      raise ValueError(f"key_id {key_id} out of range for nkey={int(model.nkey)}")
    nq, nv, nmocap, nu = int(model.nq), int(model.nv), int(model.nmocap), int(model.nu)
    key_qpos = np.asarray(model.key_qpos).reshape(int(model.nkey), nq)[key_id]
    key_qvel = np.asarray(model.key_qvel).reshape(int(model.nkey), nv)[key_id] if nv else np.zeros(0)
    if nmocap:
      key_mpos = np.asarray(model.key_mpos).reshape(int(model.nkey), nmocap, 3)[key_id]
      key_mquat = np.asarray(model.key_mquat).reshape(int(model.nkey), nmocap, 4)[key_id]
    else:
      key_mpos, key_mquat = None, None
    key_time = float(np.asarray(model.key_time).reshape(int(model.nkey))[key_id])
    if not np.isfinite(key_time) or key_time < 0:
      raise ValueError("keyframe time must be finite and nonnegative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      key_time32 = np.float32(key_time)
    if not np.isfinite(key_time32):
      raise ValueError("keyframe time must be float32-representable")
    # R4: validate held-control payload BEFORE any state mutation so that an
    # invalid keyframe fails atomically (state, held inputs, generation and
    # caches all untouched). _apply_key_ctrl re-checks defensively.
    if nu and self._control is not None:
      pre_ctrl = np.asarray(model.key_ctrl).reshape(int(model.nkey), nu)[key_id]
      if not np.all(np.isfinite(pre_ctrl)):
        raise ValueError("keyframe ctrl must be finite")
      with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        pre_ctrl32 = np.asarray(pre_ctrl, dtype=np.float32)
      if not np.all(np.isfinite(pre_ctrl32)):
        raise ValueError("keyframe ctrl must be float32-representable")
    na = int(model.na)
    if na:
      key_act = np.asarray(model.key_act).reshape(int(model.nkey), na)[key_id]
      if not np.all(np.isfinite(np.asarray(key_act, dtype=np.float32))):
        raise ValueError("keyframe act must be finite float32")
    else:
      key_act = None
    # Validate key payloads through the same paths as reset before mutating.
    prepared = self._state.prepare_reset(
        env_ids=env_ids, qpos=key_qpos, qvel=key_qvel,
        mocap_pos=key_mpos, mocap_quat=key_mquat, act=key_act,
    )
    if not prepared[0].size:
      return self._state.generation
    _reset_plugins_transactionally(getattr(self, "_native_plugins", ()),
                                   env_ids=env_ids)
    gen = self._state.reset(_prepared=prepared)
    if self._energy is not None:
      if env_ids is None:
        self._energy.zero_()
      else:
        ids = self._state._env_ids(env_ids)
        if ids.size:
          self._energy[ids.tolist()] = 0.0
    # mj_resetDataKeyframe sets time after _resetData; mirror per selected world.
    if env_ids is None:
      self._state._time.zero_().add_(key_time)
    else:
      ids = self._state._env_ids(env_ids)
      if ids.size:
        index = self._state._torch.as_tensor(ids, dtype=self._state._torch.int64, device=self._state._device)
        time_tensor = self._state._torch.full((ids.size,), key_time, dtype=self._state._torch.float32, device=self._state._device)
        next_time = self._state._time.clone()
        next_time.index_copy_(0, index, time_tensor)
        self._state._time = next_time
    self._state._generation += 1
    # Held controls follow key_ctrl for selected worlds; other held inputs zeroed.
    self._apply_key_ctrl(key_id, env_ids)
    # Keyframe reset is cold like reset.
    if getattr(self, "_coupled_constraints", None) is not None:
      self._coupled_constraints.clear_warmstart(env_ids=env_ids)
    if getattr(self, "_delay", None) is not None:
      self._delay.reset(env_ids=env_ids)
      self._sync_history_from_delay(env_ids=env_ids)
    if getattr(self, "_sensordata", None) is not None:
      if env_ids is None:
        self._sensordata.zero_()
      else:
        ids = self._state._env_ids(env_ids)
        if ids.size:
          self._sensordata[ids.tolist()] = 0.0
    self._clear_sensor_output_low(
        None if env_ids is None else self._state._env_ids(env_ids))
    if getattr(self, "_islands", None) is not None:
      self._islands.wake_all(env_ids=env_ids)
    if getattr(self, "_sleep_schedule", None) is not None:
      self._sleep_schedule.reset(
          env_ids=(None if env_ids is None else self._state._env_ids(env_ids)))
    if getattr(self, "_flex", None) is not None:
      dyn = self._smooth.run_device(self._state.qpos, self._state.qvel)
      self._flex.update_kinematics(dict(dyn["poses"], root_com=dyn["root_com"]), dyn.get("cvel"))
    self._assembled_system_valid = False
    self._accepted_step = None
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return self._state._generation

  def apply_lifecycle(self, lifecycle):
    """Adopt a rebuilt model into this live simulation, atomically (R7).

    `lifecycle` is a host-preparation `ModelLifecycle` (or a compiled
    `MjModel`) whose model must be structurally compatible: identical
    nq/nv/neq/nmocap/na counts, body/joint/geom/site/tendon/actuator counts
    and the same stepping profile (revalidated). All device stages are
    rebuilt from the new model first (raising before any mutation), then
    swapped in; live state (qpos/qvel/time/activity/held inputs) is
    preserved, fingerprints/generation advance, and assembled-system/spatial
    caches are invalidated. The model (and its batch-sharing: one model for
    all environments) applies to every world. On failure the simulation is
    untouched and keeps stepping.
    """
    from mujoco_metal.lifecycle import ModelLifecycle as _LC
    if isinstance(lifecycle, _LC):
      new_model = lifecycle._model
    elif isinstance(lifecycle, mujoco.MjModel):
      new_model = lifecycle
    else:
      raise TypeError("apply_lifecycle requires a ModelLifecycle or MjModel")
    old = self._mjmodel
    _assert_lifecycle_state_shapes_compatible(old, new_model)
    for attr in ("nq", "nv", "nbody", "njnt", "ngeom", "nsite", "ntendon",
                 "nu", "na", "neq", "nmocap"):
      if int(getattr(new_model, attr)) != int(getattr(old, attr)):
        raise ValueError(
            f"apply_lifecycle: structural count {attr} differs "
            f"({int(getattr(old, attr))} -> {int(getattr(new_model, attr))}); "
            f"rebuild the simulation instead")
    # Full validation + stage construction happens here, before any mutation
    # of this simulation (atomicity).
    fresh = type(self)(new_model, self.batch_size, profile=self.profile.name)
    keep = {"_state", "batch_size", "_success", "_combined_status",
            "_next_qpos", "_next_qvel", "_next_qacc", "_next_time",
            "_next_status", "_next_act", "_control", "_applied_force",
            "_body_wrench", "_tendon_damping", "_damping_tangent",
            "_act_dot", "_act_vel", "_spatial_kin", "_spatial_cache_key",
            "_last_coupled", "_last_coupled_generation"}
    for key, value in fresh.__dict__.items():
      if key not in keep:
        setattr(self, key, value)
    self._state.adopt_descriptor(fresh._state)
    self._state.reset_model_history(fresh._state)
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is not None:
      # Contact/equality metadata changed with the model. The first current
      # position sweep must rebuild an awake graph against the new lowering.
      scheduler.wake_all()
    self._assembled_system_valid = False
    self._accepted_step = None
    self._last_coupled = None
    self._last_coupled_generation = None
    self._spatial_cache_key = None
    self._spatial_kin = None
    # Delay rings are model-owned: a rebuilt model starts cold history
    # (fresh construction above replaces them; not in `keep`).
    self._invalidate_forward_stage_record()
    self._state._generation += 1
    return self._state.generation

  def _clear_sensor_output_low(self, ids=None):
    """Clear paired sensor residuals whenever stored samples are replaced."""
    seen = set()
    for program in (getattr(self, "_sensors", None),
                    getattr(self, "_raw_sensor_program", None)):
      if program is None or id(program) in seen:
        continue
      seen.add(id(program))
      low = getattr(program, "_s_state_out_low", None)
      if low is None:
        continue
      if ids is None:
        low.zero_()
      elif len(ids):
        low[list(ids)] = 0.0

  @staticmethod
  def _sensor_output_low_snapshot(program):
    low = getattr(program, "_s_state_out_low", None) if program is not None else None
    return (low.detach().cpu().numpy().copy()
            if low is not None else None)

  def _restore_sensor_output_low(self, program, values, ids=None):
    low = getattr(program, "_s_state_out_low", None) if program is not None else None
    if low is None:
      return
    if values is None:
      if ids is None:
        low.zero_()
      elif len(ids):
        low[list(ids)] = 0.0
      return
    value = self._state._torch.as_tensor(
        values, dtype=self._state._torch.float32,
        device=self._state._device)
    if ids is None:
      low.copy_(value)
    else:
      low[ids] = value[ids]

  def _clear_held_inputs(self, env_ids):
    ids = None if env_ids is None else self._state._env_ids(env_ids)
    if self._applied_force is not None:
      if ids is None:
        self._applied_force.zero_()
      elif ids.size:
        self._applied_force[ids] = 0
    if self._control is not None:
      if ids is None:
        self._control.zero_()
      elif ids.size:
        self._control[ids] = 0
    if self._body_wrench is not None:
      if ids is None:
        self._body_wrench.zero_()
      elif ids.size:
        self._body_wrench[ids] = 0

  def _apply_key_ctrl(self, key_id, env_ids):
    model = self._mjmodel
    nu = int(model.nu)
    key_ctrl = np.asarray(model.key_ctrl).reshape(int(model.nkey), nu)[key_id] if nu else None
    if key_ctrl is not None and self._control is not None:
      # R4: validate before clearing/applying so failures leave held inputs
      # and state untouched.
      if not np.all(np.isfinite(key_ctrl)):
        raise ValueError("keyframe ctrl must be finite")
      with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ctrl32 = np.asarray(key_ctrl, dtype=np.float32)
      if not np.all(np.isfinite(ctrl32)):
        raise ValueError("keyframe ctrl must be float32-representable")
    self._clear_held_inputs(env_ids)
    if key_ctrl is not None and self._control is not None:
      ids = None if env_ids is None else self._state._env_ids(env_ids)
      ctrl32 = np.asarray(key_ctrl, dtype=np.float32)
      if ids is None:
        self._control.copy_(
            self._state._torch.as_tensor(
                np.broadcast_to(ctrl32, (self.batch_size, nu)).copy(),
                dtype=self._state._torch.float32, device=self._state._device,
            )
        )
      elif ids.size:
        index = self._state._torch.as_tensor(ids, dtype=self._state._torch.int64, device=self._state._device)
        vals = self._state._torch.as_tensor(
            np.broadcast_to(ctrl32, (ids.size, nu)).copy(),
            dtype=self._state._torch.float32, device=self._state._device,
        )
        nxt = self._control.clone()
        nxt.index_copy_(0, index, vals)
        self._control = nxt

  def buffer_audit(self):
    """Return a tuple of buffer specifications derived from real device allocations."""
    b = self.batch_size
    entries = [
        {"name": "state.qpos", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._model.nq})", "dtype": str(self._state._qpos.dtype).replace("torch.", "")},
        {"name": "state.qvel", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._model.nv})", "dtype": str(self._state._qvel.dtype).replace("torch.", "")},
        {"name": "state.status", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b},)", "dtype": str(self._state._status.dtype).replace("torch.", "")},
        {"name": "state.time", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b},)", "dtype": str(self._state._time.dtype).replace("torch.", "")},
        {"name": "mass_matrix", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv}, {self._state._model.nv})", "dtype": "float32"},
        {"name": "qfrc_bias", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv})", "dtype": "float32"},
        {"name": "qfrc_smooth", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv})", "dtype": "float32"},
    ]
    if self._energy is not None:
      entries.append({"name": "energy", "residency": "MPS device-resident",
                      "lifetime": "persistent stage diagnostics",
                      "shape": f"({b}, 2)", "dtype": "float32"})
      for name, value in (
          ("energy_body_mass_const", self._energy_body_mass),
          ("energy_gravity_const", self._energy_gravity),
          ("energy_qpos_spring_const", self._energy_qpos_spring),
      ):
        entries.append({"name": name, "residency": "MPS device-resident",
                        "lifetime": "persistent compiled model constants",
                        "shape": str(tuple(value.shape)), "dtype": "float32"})
    if getattr(self._state, "_neq", 0) > 0 and getattr(self._state, "_eq_active", None) is not None:
      entries.append({"name": "state.eq_active", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._neq})", "dtype": str(self._state._eq_active.dtype).replace("torch.", "")})
    if getattr(self._state, "_nmocap", 0) > 0:
      entries.append({"name": "state.mocap_pos", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._nmocap}, 3)", "dtype": "float32"})
      entries.append({"name": "state.mocap_quat", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._nmocap}, 4)", "dtype": "float32"})
    if getattr(self._state, "_na", 0) > 0:
      entries.append({"name": "state.act", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._na})", "dtype": "float32"})
      entries.append({"name": "act_dot", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._na})", "dtype": "float32"})
    if self._euler_solver is not None:
      entries.append({"name": "effective_mass", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv}, {self._state._model.nv})", "dtype": "float32"})
    if self._coupled_constraints is not None:
      d = self._coupled_constraints.descriptor
      jac_layout = self._coupled_constraints._jacobian_layout
      jac_shape = (f"({b}, {jac_layout.stride_words})"
                   if int(jac_layout.mode) == 1
                   else f"({b}, {d.nr}, {d.nv})")
      jac_lifetime = ("typed per-world packed CSR workspace; no dense canonical J"
                      if int(jac_layout.mode) == 1
                      else "preallocated dense canonical-J workspace")
      entries.extend([
          {"name": "_eq_active_default", "residency": "MPS device-resident", "lifetime": "persistent preallocated", "shape": f"({b}, {max(d.neq, 1)})", "dtype": str(self._coupled_constraints._eq_active_default.dtype).replace("torch.", "")},
          {"name": "workspace_J", "residency": "MPS device-resident", "lifetime": jac_lifetime, "shape": jac_shape, "dtype": "float32"},
          {"name": "position_cache_J", "residency": "MPS device-resident", "lifetime": jac_lifetime.replace("workspace", "cache"), "shape": jac_shape, "dtype": "float32"},
          {"name": "workspace_debug", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {self._coupled_constraints._debug_stride})", "dtype": "float32"},
          {"name": "out_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.nv})", "dtype": "float32"},
          {"name": "out_acc", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.nv})", "dtype": "float32"},
          {"name": "out_status", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b},)", "dtype": str(self._coupled_constraints._workspace["out_status"].dtype).replace("torch.", "")},
          {"name": "out_diagnostics", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, 10)", "dtype": "float32"},
      ])
      if d.nc > 0:
        entries.extend([
            {"name": "contact_row_data", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.ncontacts_max}, 5, 6)", "dtype": "float32"},
            {"name": "contact_jacobian", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.ncontacts_max}, 5, {d.nv})", "dtype": "float32"},
            {"name": "out_contact_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.ncontacts_max * 5})", "dtype": "float32"},
        ])

      if d.nr_joint > 0:
        entries.append({"name": "out_joint_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {max(d.nr_joint, 1)})", "dtype": "float32"})
    if self._legacy_canonical_rows is not None:
      entries.append({"name": "legacy_canonical_rows",
                      "residency": "MPS device-resident",
                      "lifetime": "persistent inverse row assembly",
                      "shape": str(tuple(self._legacy_canonical_rows.shape)),
                      "dtype": "float32"})
    if self._legacy_constraint_rhs is not None:
      entries.append({"name": "legacy_constraint_rhs",
                      "residency": "MPS device-resident",
                      "lifetime": "scratch/legacy constraint solve",
                      "shape": str(tuple(self._legacy_constraint_rhs.shape)),
                      "dtype": "float32"})
    if self._contact is not None:
      entries.append({"name": "contact.canonical_rows",
                      "residency": "MPS device-resident",
                      "lifetime": "preallocated contact workspace",
                      "shape": str(tuple(self._contact._workspace[
                          "canonical_rows"].shape)), "dtype": "float32"})
    if self._joint_constraints is not None:
      entries.append({"name": "joint.canonical_rows",
                      "residency": "MPS device-resident",
                      "lifetime": "preallocated joint workspace",
                      "shape": str(tuple(self._joint_constraints._outputs[
                          "canonical_rows"].shape)), "dtype": "float32"})
      entries.append({"name": "joint.lambda",
                      "residency": "MPS device-resident",
                      "lifetime": "last scalar constraint force diagnostics",
                      "shape": str(tuple(self._joint_constraints._outputs[
                          "lambda"].shape)), "dtype": "float32"})
    if self._sensors is not None:
      entries.append({"name": "sensordata", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._mjmodel.nsensordata})", "dtype": "float32"})
    if self._component_solution_vector is not None:
      entries.append({"name": "component_solution_vector",
                      "residency": "MPS device-resident",
                      "lifetime": "persistent first-RHS result",
                      "shape": str(tuple(self._component_solution_vector.shape)),
                      "dtype": "float32"})
      entries.append({"name": "component_solution_low_vector",
                      "residency": "MPS device-resident",
                      "lifetime": "persistent first-RHS residual correction",
                      "shape": str(tuple(self._component_solution_low_vector.shape)),
                      "dtype": "float32"})
    return tuple(entries)

  def sensor_values(self):
    """Evaluate a query without changing persistent sparse-solve scratch."""
    if not getattr(self, "_component_mass_enabled", False):
      return self._sensor_values_impl()
    owned_names = ("_component_solve_rhs", "_component_solution_vector",
                   "_component_solution_low_vector",
                   "_component_world_status",
                   "_component_tendon_J", "_component_damping_deriv")
    owned = {name: (getattr(self, name).clone()
                    if getattr(self, name, None) is not None else None)
             for name in owned_names}
    component_workspace = _clone_system_dict(
        self._component_solver._workspace)
    component_controls = {
        name: getattr(self._component_solver, name).clone()
        for name in ("_euler_dt", "_allow_indefinite")
        if isinstance(getattr(self._component_solver, name, None),
                      self._state._torch.Tensor)
    }
    try:
      return self._sensor_values_impl()
    finally:
      for name, value in owned.items():
        current = getattr(self, name, None)
        if current is not None and value is not None:
          current.copy_(value)
      _restore_system_buffers(self._component_solver._workspace,
                              component_workspace)
      for name, value in component_controls.items():
        getattr(self._component_solver, name).copy_(value)

  def _sensor_values_impl(self):
    """Evaluate supported stateless sensors at CURRENT state on MPS.

    This is an explicit forward-stage query, not the pre-integration sample
    left by MuJoCo mj_step (see step_sensordata). Re-evaluation makes
    reset/restore immediately visible. Returned values are detached copies
    owned by the caller.
    """
    if self._sensors is None:
      raise ValueError("sensor_values requires a sensor stepping profile")
    state = self._state
    mpos = getattr(state, "_mpos", None)
    mquat = getattr(state, "_mquat", None)
    scheduler = getattr(self, "_sleep_schedule", None)
    awake_lists = scheduler.awake_lists() if scheduler is not None else None
    sparse_poses = None
    tendon_J = None
    if self._component_mass_enabled:
      if awake_lists is None:
        awake_lists = self._smooth._workspace["all_awake_lists"]
      sparse_poses = self._smooth._fk.run_device(
          state._qpos, mpos, mquat,
          tree_awake=(scheduler.tree_awake if scheduler is not None else None))
      tendon_J = self._component_tendon_jacobian(state._qvel, sparse_poses)
    dynamics = self._smooth.run_device(
        state._qpos, state._qvel, mpos if sparse_poses is None else None,
        mquat if sparse_poses is None else None, poses=sparse_poses,
        awake_lists=awake_lists, tendon_J=tendon_J,
        _trusted_internal_poses=self._component_mass_enabled)
    sensor_dynamics = dynamics
    if (not self._component_mass_enabled
        and np.any(np.asarray(self._sensors.descriptor.sensor_type)
                   == int(mujoco.mjtSensor.mjSENS_E_KINETIC))):
      # Dense SmoothDynamics intentionally returns rigid-body CRBA when no
      # tendon Jacobian is supplied. Kinetic-energy queries need the full M,
      # but must not mutate its borrowed stepping/assembled-system backing.
      sensor_mass = dynamics["mass_matrix"].clone()
      if self._tendons is not None:
        _, _, fixed_armature = self._tendons.run_device(
            state._qpos, state._qvel)
        if fixed_armature is not None:
          sensor_mass.add_(fixed_armature)
      if self._spatial_tendons is not None:
        sensor_kin = self._spatial_tendons.run_kinematics(
            state._qvel, dynamics["poses"])
        _, _, spatial_armature = self._spatial_tendons.run_forces(
            sensor_kin, include_armature=True)
        if spatial_armature is not None:
          sensor_mass.add_(spatial_armature.reshape(sensor_mass.shape))
      sensor_dynamics = dict(dynamics)
      sensor_dynamics["mass_matrix"] = sensor_mass
    poses = dict(
        dynamics["poses"], cvel=dynamics["cvel"], root_com=dynamics["root_com"]
    )
    torch = state._torch
    b = state.batch_size
    base = (self._sensordata.clone() if self._sensordata is not None else
            torch.zeros((b, self._mjmodel.nsensordata), dtype=torch.float32,
                        device=state._device))
    out = self._sensors.run_device(
        state._qpos, state._qvel, state._time, poses, sensordata=base)
    out = self._run_state_sensors(state._qpos, state._qvel, poses, sensor_dynamics,
                                  out=out)
    need_acc_or_contact = self._has_acc_sensors or self._has_contact_sensors
    if need_acc_or_contact:
      # ACC and CONTACT families need the solved forward state (pinned sensorAcc runs
      # post-constraint); evaluate the full forward, then the ACC/contact kernels.
      # Queries must not perturb future trajectory: the forward solve
      # refreshes retained warmstarts and the coupled cache, so both are
      # saved and restored around the query (R03 isolation).
      cc = getattr(self, "_coupled_constraints", None)
      last_coupled = getattr(self, "_last_coupled", None)
      last_gen = getattr(self, "_last_coupled_generation", None)
      accepted_step = getattr(self, "_accepted_step", None)
      asm_valid = getattr(self, "_assembled_system_valid", True)
      ws_saved = _clone_system_dict(cc._workspace) if cc is not None else None
      rhs_saved = self._rhs.clone() if getattr(self, "_rhs", None) is not None else None
      try:
        acceleration, _, acc_dyn = self._acceleration(state._qpos, state._qvel)
        acc_poses = dict(acc_dyn["poses"], cvel=acc_dyn["cvel"],
                         root_com=acc_dyn["root_com"])
        if self._has_acc_sensors:
          out = self._run_acc_into(state._qpos, state._qvel, acceleration,
                                   acc_poses, acc_dyn, out,
                                   qacc_low=self._qacc_low_for_sensor(acceleration))
        if self._has_touch_grid_sensors:
          out = self._run_touch_grid_into(acc_poses, out)
        if self._has_contact_sensors or self._has_ray_sensors or self._has_geomdist_sensors:
          out = self._run_spatial_into(acc_poses, out)
      finally:
        if ws_saved is not None:
          _restore_system_buffers(cc._workspace, ws_saved)
        if rhs_saved is not None:
          self._rhs.copy_(rhs_saved)
        if hasattr(self, "_last_coupled"):
          self._last_coupled = last_coupled
        if hasattr(self, "_last_coupled_generation"):
          self._last_coupled_generation = last_gen
        self._accepted_step = accepted_step
        self._assembled_system_valid = asm_valid
    elif self._has_ray_sensors or self._has_geomdist_sensors:
      out = self._run_spatial_into(poses, out)
    return (self._history_program.samples(
        out, state._history, state._time, kind=1) if
        self._history_program.sensor_enabled else out).clone()

  def _state_sensor_inputs(self, qpos, qvel, poses, dynamics,
                           position_context=None, world_mask=None):
    """Borrowed (ten_spa_len, ten_spa_vel, act_dyn_len, act_dyn_vel, kind).

    Computes spatial/general kinematics scratch synchronously (no readback)
    so tendon/actuator state sensors observe the given state. Returns
    ``None`` views when the driving stage is absent.
    """
    ten_l = ten_v = act_l = act_v = None
    kind = 0
    if getattr(self, "_spatial_tendons", None) is not None:
      if position_context is None:
        self._spatial_jacobian(qvel, poses, world_mask=world_mask)
        kin = self._spatial_kin
      else:
        kin = self._spatial_tendons.run_velocity_kinematics(
            position_context["spatial_tendons"], qvel,
            world_mask=world_mask)
      ten_l = kin["length"].reshape(self._state.batch_size, -1)
      ten_v = kin["velocity"].reshape(self._state.batch_size, -1)
    if getattr(self, "_actuators", None) is not None:
      kind = 1
      actuators = self._actuators
      contacts = None
      if position_context is None and actuators.meta.has_body_transmission:
        if self._coupled_constraints is None:
          raise ValueError("BODY transmissions require the coupled constraint stage")
        self._coupled_constraints._constants["body_dims"][3] = actuators.meta.nu
        self._coupled_constraints.generate_candidates(
            poses, qvel, world_mask=world_mask)
        contacts = self._coupled_constraints.contact_buffers()
      kin = (actuators.run_kinematics(
                 qpos, qvel, poses, contacts, world_mask=world_mask)
             if position_context is None else
             actuators.run_velocity_kinematics(
                 position_context["actuators"], qvel,
                 world_mask=world_mask))
      if position_context is None and getattr(self, "_spatial_tendons", None) is not None:
        self._spatial_tendons.apply_spatial_tendon_state(
            actuators.meta, self._spatial_kin, kin,
            world_mask=world_mask)
      act_l = kin["length"].reshape(self._state.batch_size, -1)
      act_v = kin["velocity"].reshape(self._state.batch_size, -1)
    return ten_l, ten_v, act_l, act_v, kind

  def _run_state_sensors(self, qpos, qvel, poses, dynamics, out, *, program=None,
                         stages=(mujoco.mjtStage.mjSTAGE_POS, mujoco.mjtStage.mjSTAGE_VEL),
                         position_context=None, world_mask=None):
    """Evaluate milestone-016 state families into ``out`` (borrowed)."""
    program = self._sensors if program is None else program
    producer_mask = world_mask
    if world_mask is not None and world_mask.dtype == self._state._torch.bool:
      producer_mask = self._state._reset_mask_i32
      producer_mask.copy_(world_mask)
    ten_l, ten_v, act_l, act_v, kind = self._state_sensor_inputs(
        qpos, qvel, poses, dynamics, position_context=position_context,
        world_mask=producer_mask)
    mm = dynamics.get("mass_matrix", None)
    mass_qvel = None
    if (mm is None and getattr(self, "_component_mass_enabled", False)
        and np.any(np.asarray(program.descriptor.sensor_type)
                   == int(mujoco.mjtSensor.mjSENS_E_KINETIC))):
      mass_blocks = dynamics.get("mass_blocks")
      if mass_blocks is None:
        raise RuntimeError(
            "sparse kinetic-energy sensor requires prepared component mass blocks")
      armature_blocks = dynamics.get("tendon_armature_blocks")
      if armature_blocks is None and self._smooth._has_tendon_armature:
        tendon_J = self._component_tendon_jacobian(qvel, poses)
        armature_blocks = self._smooth.project_tendon_armature_blocks(
            tendon_J, tree_awake=(getattr(
                getattr(self, "_sleep_schedule", None), "tree_awake", None)))
      mass_qvel = self._smooth.mass_blocks_matvec_device(
          mass_blocks, qvel, tendon_armature_blocks=armature_blocks,
          world_mask=producer_mask)
    return program.run_state_device(
        qpos, qvel, poses, mass_matrix=mm, mass_qvel=mass_qvel,
        ten_spa_len=ten_l, ten_spa_vel=ten_v,
        act_dyn_len=act_l, act_dyn_vel=act_v, act_kind=kind, out=out,
        stages=stages, world_mask=world_mask)

  def assembled_system(self, *, ctrl=None, qfrc_applied=None, recompute=False):
    """Return the coupled constraint system tensors on MPS.

    If `recompute` is True, `ctrl` is given, `qfrc_applied` is given, or no step
    has been run yet, evaluates the forward smooth dynamics and coupled constraint
    assembly at the current device state with the provided control and applied forces.
    Returns dict containing 'J', 'W', 'W_regularized', 'R', 'ar', 'rhs', 'lambda',
    'qacc', 'qfrc_constraint', 'mass_matrix', 'status'.
    Component profiles return `mass_matrix=None`, rigid `mass_blocks`, immutable
    `mass_block_layout`, and optional `tendon_armature_blocks` with the same
    packed layout. The full mass operator is the sum of those two block arrays;
    the solver includes that armature contribution exactly once. Device arrays
    are borrowed workspace views; clone them to retain values across dispatches.
    Changing equality activity, reset or restore invalidates the cache.
    """
    self._assert_contact_override_binding()
    if self._coupled_constraints is None:
      raise ValueError("assembled_system requires a coupled constraint stepping profile")
    if ctrl is not None:
      self._prepare_control(ctrl)
      recompute = True
    if qfrc_applied is not None:
      self._prepare_force(qfrc_applied)
      recompute = True
    if not recompute and getattr(self, "_assembled_system_valid", False):
      if (
          getattr(self, "_accepted_step", None) is not None
          and self._accepted_step["generation"] == self._state.generation
      ):
        return self._accepted_step["system"]
      if (
          getattr(self, "_last_coupled", None) is not None
          and getattr(self, "_last_coupled_generation", None) == self._state.generation
      ):
        return self._last_coupled
    state = self._state
    qpos, qvel = state._qpos, state._qvel
    mpos = getattr(state, "_mpos", None)
    mquat = getattr(state, "_mquat", None)
    scheduler = getattr(self, "_sleep_schedule", None)
    awake_lists = scheduler.awake_lists() if scheduler is not None else None
    if self._component_mass_enabled and awake_lists is None:
      awake_lists = self._smooth._workspace["all_awake_lists"]
    # The next full position assembly invalidates reuse of a borrowed spatial
    # tendon view even when qvel/pose tensor identities have been reused.
    self._spatial_cache_key = None
    sparse_tendon_J = None
    sparse_poses = None
    if self._component_mass_enabled:
      sparse_poses = self._smooth._fk.run_device(
          qpos, mpos, mquat,
          tree_awake=(scheduler.tree_awake if scheduler is not None else None))
      sparse_tendon_J = self._component_tendon_jacobian(qvel, sparse_poses)
    dynamics = self._smooth.run_device(
        qpos, qvel, mpos, mquat, poses=sparse_poses,
        awake_lists=awake_lists, tendon_J=sparse_tendon_J)
    torch = state._torch
    rhs = torch.neg(dynamics["qfrc_bias"])
    if self._applied_force is not None:
      rhs.add_(self._applied_force)
    if self._passive is not None:
      passive, _ = self._passive.run_device(
          qpos, qvel, xfrc_applied=self._body_wrench, return_damping=True,
          mocap_pos=mpos, mocap_quat=mquat, awake_lists=awake_lists,
      )
      rhs.add_(passive)
    if self._flex is not None:
      flex_qfrc, _, _ = self._flex.run_device(
          qpos, qvel, dict(dynamics["poses"], root_com=dynamics["root_com"]),
          dynamics.get("cvel", None)
      )
      rhs.add_(flex_qfrc)
    if self._fluid is not None:
      rhs.add_(self._fluid.run_device(qpos, qvel, dynamics))
    if self._tendons is not None:
      tendon_force, _, tendon_armature = self._tendons.run_device(qpos, qvel)
      rhs.add_(tendon_force)
      if not self._component_mass_enabled:
        dynamics["mass_matrix"].add_(tendon_armature)
    if self._spatial_tendons is not None:
      sjac = self._spatial_jacobian(qvel, dynamics["poses"])
      skin = self._spatial_kin
      sforce, _, sarm = self._spatial_tendons.run_forces(
          skin, include_armature=not self._component_mass_enabled)
      rhs.add_(sforce.reshape(rhs.shape))
      sbias, _ = self._spatial_tendons.run_armature_bias(
          skin, qvel, dynamics["poses"], dynamics.get("cvel", None),
          dynamics.get("root_com", None), dynamics.get("cdof", None),
          dynamics.get("cdof_dot", None))
      # Armature bias is a bias force (pinned mj_tendonBias accumulates into
      # qfrc_bias), so it subtracts from rhs like Coriolis/gravity.
      rhs.sub_(sbias.reshape(rhs.shape))
      if not self._component_mass_enabled:
        dynamics["mass_matrix"].add_(sarm.reshape(dynamics["mass_matrix"].shape))
    if self._transmissions is not None:
      rhs.add_(self._transmissions.run_device(
          qpos, qvel, self._delayed_control(state._time),
          awake_lists=awake_lists,
          gravcomp=self._transmission_gravcomp(
              qpos, dynamics["poses"], awake_lists=awake_lists)))
    if self._actuators is not None:
      rhs.add_(self._actuation_force(qpos, qvel, dynamics["poses"]))
    if self._motor is not None:
      rhs.add_(self._motor.run_device(self._delayed_control(state._time)))
    eq_active = getattr(state, "_eq_active", None)
    _ten_J, _ten_L = self._spatial_for_coupled(qvel, dynamics["poses"])
    self._coupled_solve_dispatches += 1
    coupled = self._coupled_constraints.run_device(
        dict(dynamics["poses"], root_com=dynamics["root_com"]),
        (None if self._component_mass_enabled else dynamics["mass_matrix"]), rhs, qpos, qvel,
        eq_active=eq_active, cvel=dynamics.get("cvel", None),
        cdof=dynamics.get("cdof", None), cdof_dot=dynamics.get("cdof_dot", None),
        tendon_J_spatial=_ten_J, tendon_length_spatial=_ten_L,
        flex=self._flex, qacc_warmstart=state._qacc_warmstart,
        awake_tree=getattr(getattr(self, "_sleep_schedule", None), "tree_awake", None),
        component_solver=(self._component_solver
                          if self._component_mass_enabled else None),
        mass_blocks=(dynamics.get("mass_blocks")
                     if self._component_mass_enabled else None),
        tendon_armature_blocks=(dynamics.get("tendon_armature_blocks")
                                if self._component_mass_enabled else None),
        awake_dof_ids=(awake_lists["dof_ids"]
                       if self._component_mass_enabled and awake_lists is not None
                       else None),
        awake_counts=(awake_lists["counts"]
                      if self._component_mass_enabled and awake_lists is not None
                      else None),
    )
    coupled["mass_matrix"] = dynamics.get("mass_matrix")
    if self._component_mass_enabled:
      coupled["mass_blocks"] = dynamics["mass_blocks"]
      coupled["mass_block_layout"] = self._smooth.mass_block_layout
      coupled["tendon_armature_blocks"] = dynamics.get(
          "tendon_armature_blocks")
    if "pose_status" in dynamics:
      coupled["status"] = torch.where(
          dynamics["pose_status"] == 0, coupled["status"], dynamics["pose_status"])
    self._last_coupled = coupled
    self._last_coupled_generation = self._state.generation
    self._last_coupled_qacc_version = int(coupled["qacc"]._version)
    self._assembled_system_valid = True
    return coupled

  @property
  def accepted_step(self):
    """Owned per-world snapshot of the latest successful trajectory rows.

    ``accepted_mask`` marks worlds with retained successful history and is
    cumulative across partial failures. ``generation`` is the newest global
    generation containing any successful world; ``input_time`` is per-world.
    """
    if self._accepted_step is None:
      return None
    record_mask = self._accepted_step.get("accepted_mask")
    if record_mask is not None and not bool(record_mask.any().item()):
      return None
    generation = self._accepted_step["generation"]
    input_generation = self._accepted_step["input_generation"]
    return {
        "system": _clone_system_dict(self._accepted_step["system"]),
        "generation": int(generation.item()) if hasattr(generation, "item") else generation,
        "input_generation": (int(input_generation.item())
                             if hasattr(input_generation, "item") else input_generation),
        "input_time": self._accepted_step["input_time"].clone(),
        "status": self._accepted_step["status"].clone(),
        "acceleration": self._accepted_step["acceleration"].clone(),
        "accepted_mask": (record_mask.clone() if record_mask is not None else None),
    }

  def _record_accepted_step(self, accepted_mask, input_generation, input_time,
                            status, acceleration):
    """Merge accepted world rows into owned history without a device readback.

    A failed world retains its last successful snapshot. An all-failed call
    therefore leaves the previous accepted record intact; on the first
    all-failed call the zero mask makes the public property return ``None``.
    """
    torch = self._state._torch
    success = accepted_mask.to(dtype=torch.bool)
    source_system = None
    if (getattr(self, "_last_coupled", None) is not None
        and getattr(self, "_last_coupled_generation", None) == input_generation):
      source_system = self._last_coupled
    previous = self._accepted_step
    if previous is None:
      system = _initial_accepted_system(
          source_system, success, self.batch_size)
      old_mask = torch.zeros_like(success)
      old_status = torch.zeros_like(status)
      old_acceleration = torch.zeros_like(acceleration)
      old_time = torch.zeros_like(input_time)
      generation = torch.zeros((), dtype=torch.int64, device=self._state._device)
      old_input_generation = torch.zeros_like(generation)
    else:
      system = _merge_accepted_system(
          previous["system"], source_system, success, self.batch_size)
      old_mask = previous["accepted_mask"]
      old_status = previous["status"]
      old_acceleration = previous["acceleration"]
      old_time = previous["input_time"]
      generation = previous["generation"]
      old_input_generation = previous["input_generation"]
    new_generation = torch.as_tensor(
        self._state.generation, dtype=torch.int64, device=self._state._device)
    new_input_generation = torch.as_tensor(
        input_generation, dtype=torch.int64, device=self._state._device)
    any_success = success.any()
    self._accepted_step = {
        "system": system,
        "generation": torch.where(any_success, new_generation, generation),
        "input_generation": torch.where(
            any_success, new_input_generation, old_input_generation),
        "input_time": torch.where(success, input_time, old_time),
        "status": torch.where(success, status, old_status),
        "acceleration": torch.where(success[:, None], acceleration,
                                    old_acceleration),
        "accepted_mask": old_mask | success,
    }

  @property
  def solver_fwdinv(self):
    """Per-world ``mj_compareFwdInv`` residuals from the latest step.

    The two columns are the constraint-force residual and applied-force
    residual, matching ``mjData.solver_fwdinv``. Values are zero when the
    enable flag is off or the model has no active constraints.
    """
    return self._solver_fwdinv.clone()

  def _prepare_force(self, qfrc_applied):
    torch = self._state._torch
    shape = (self._state.batch_size, self._state._model.nv)
    if qfrc_applied is None:
      # Like mj_step, omitted inputs preserve the simulation's held applied
      # force. Pass an explicit zero array to clear it.
      return
    if self._applied_force is None:
      raise ValueError("qfrc_applied requires contact_free_forces_euler_v1")
    if isinstance(qfrc_applied, torch.Tensor):
      if qfrc_applied.dtype != torch.float32:
        raise TypeError("qfrc_applied tensor must have dtype torch.float32")
      if tuple(qfrc_applied.shape) != shape:
        raise ValueError(f"qfrc_applied must have shape {shape}")
      if qfrc_applied.device.type != "mps":
        raise ValueError("qfrc_applied tensor must be on MPS")
      if not qfrc_applied.is_contiguous():
        raise ValueError("qfrc_applied tensor must be contiguous")
      self._applied_force.copy_(qfrc_applied)
      return
    array = np.asarray(qfrc_applied)
    if array.shape != shape or array.dtype.kind not in "fiu":
      raise ValueError(f"qfrc_applied must be numeric with shape {shape}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      array = np.asarray(array, dtype=np.float32, order="C")
    if not np.all(np.isfinite(array)):
      raise ValueError(
          "qfrc_applied must be finite and representable as float32"
      )
    self._applied_force.copy_(torch.tensor(array, device=self._state._device))

  def _prepare_wrench(self, xfrc_applied):
    if self._body_wrench is None:
      if xfrc_applied is not None:
        raise ValueError("xfrc_applied requires a passive or sensor profile")
      return
    if xfrc_applied is None:
      # Preserve values installed through mj_setState or a prior step call.
      return
    torch = self._state._torch
    shape = tuple(self._body_wrench.shape)
    if isinstance(xfrc_applied, torch.Tensor):
      if (
          xfrc_applied.device.type != "mps"
          or xfrc_applied.dtype != torch.float32
          or not xfrc_applied.is_contiguous()
          or tuple(xfrc_applied.shape) != shape
      ):
        raise ValueError(
            f"xfrc_applied must be contiguous float32 MPS with shape {shape}"
        )
      self._body_wrench.copy_(xfrc_applied)
    else:
      array = np.asarray(xfrc_applied)
      if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"xfrc_applied must be numeric with shape {shape}")
      with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float32)
      if not np.all(np.isfinite(array)):
        raise ValueError("xfrc_applied must be finite float32")
      self._body_wrench.copy_(torch.tensor(array, device=self._state._device))

  def _prepare_control(self, ctrl):
    if self._control is None:
      if ctrl is not None:
        raise ValueError("ctrl requires contact_free_motor_euler_v1")
      return
    torch = self._state._torch
    shape = tuple(self._control.shape)
    if ctrl is None:
      # Control is persistent MuJoCo state; None means leave it unchanged.
      return
    if isinstance(ctrl, torch.Tensor):
      if ctrl.dtype != torch.float32:
        raise TypeError("ctrl tensor must have dtype torch.float32")
      if tuple(ctrl.shape) != shape:
        raise ValueError(f"ctrl must have shape {shape}")
      if ctrl.device.type != "mps":
        raise ValueError("ctrl tensor must be on MPS")
      if not ctrl.is_contiguous():
        raise ValueError("ctrl tensor must be contiguous")
      self._control.copy_(ctrl)
      return
    array = np.asarray(ctrl)
    if array.shape != shape or array.dtype.kind not in "fiu":
      raise ValueError(f"ctrl must be numeric with shape {shape}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      array = np.asarray(array, dtype=np.float32, order="C")
    if not np.all(np.isfinite(array)):
      raise ValueError("ctrl must be finite and representable as float32")
    self._control.copy_(torch.tensor(array, device=self._state._device))

  def _spatial_jacobian(self, qvel, poses, world_mask=None):
    """Run spatial tendon kinematics once per assembly; stash for reuse.

    Returns the borrowed dense Jacobian [B, nt, nv] or None. Both the
    actuator-moment overlay and the spatial passive forces share this result
    within one assembly (pinned mj_tendon runs once in mj_fwdPosition).
    """
    if self._spatial_tendons is None:
      return None
    gen = (self._state.generation, id(qvel), id(poses.get("body_pos", None)))
    cached = getattr(self, "_spatial_cache_key", None) == gen
    if world_mask is not None:
      # A masked refresh always updates selected rows in place. It can retain
      # the batch cache key only when the existing healthy rows already belong
      # to the same inputs; otherwise leave the cache globally untrusted so a
      # later full-stage request recomputes every row.
      kin = self._spatial_tendons.run_kinematics(
          qvel, poses, world_mask=self._state._reset_mask_i32)
      self._spatial_kin = kin
      self._spatial_cache_key = gen if cached else None
    elif not cached:
      kin = self._spatial_tendons.run_kinematics(qvel, poses)
      self._spatial_cache_key = gen
      self._spatial_kin = kin
    return self._spatial_kin["jacobian"]

  def _component_tendon_jacobian(self, qvel, poses, world_mask=None):
    """Build the canonical fixed+spatial tendon J for block armature packing."""
    if (not getattr(self, "_component_mass_enabled", False)
        or self._component_tendon_J is None):
      return None
    output = self._component_tendon_J
    if world_mask is None:
      output.zero_()
    else:
      self._state.clear_masked_rows(output, world_mask)
    nt, nv = int(self._mjmodel.ntendon), int(self._mjmodel.nv)
    if self._tendons is not None and nt and nv:
      template = self._tendons.fixed_jacobian_template.reshape(nt, nv)
      if world_mask is None:
        output[:, :nt, :nv].copy_(template.unsqueeze(0))
      else:
        expanded = template.unsqueeze(0).expand(self._state.batch_size, -1, -1).contiguous()
        self._state.update_masked_rows(output[:, :nt, :nv], expanded, world_mask,
                                       add=False)
    if self._spatial_tendons is not None and nt and nv:
      spatial = self._spatial_jacobian(qvel, poses, world_mask=world_mask)
      if world_mask is None:
        output[:, :nt, :nv].add_(spatial[:, :nt, :nv])
      else:
        self._state.update_masked_rows(output[:, :nt, :nv],
                                       spatial[:, :nt, :nv], world_mask)
    # The device allocation is padded for empty dimensions, while the smooth
    # projection consumes the exact compiled [batch,ntendon,nv] view.
    return output[:, :nt, :nv]

  def _spatial_for_coupled(self, qvel, poses, world_mask=None):
    """Borrowed (J, length) spatial tendon views for the coupled stage."""
    if self._spatial_tendons is None:
      return None, None
    self._spatial_jacobian(qvel, poses, world_mask=world_mask)
    kin = self._spatial_kin
    return kin["jacobian"], kin["length"]

  def _component_rhs_workspace(self):
    """Return the dedicated smooth-force RHS workspace."""
    if self._component_solver is None:
      raise RuntimeError("component mass solver is not prepared")
    if self._component_solve_rhs is None:
      raise RuntimeError("component smooth-force RHS workspace was not prepared")
    return self._component_solve_rhs

  def _solve_component_rhs(self, dynamics, rhs, awake_lists=None, *,
                           rhs_low=None, diagonal_add=None,
                           allow_indefinite=False, world_mask=None):
    """Apply the prepared component inverse to one canonical vector/world."""
    if self._component_solver is None:
      raise RuntimeError("component mass solver is not prepared")
    workspace = self._component_rhs_workspace()
    rhs_index = self._component_solver.rhs_capacity - 1
    if world_mask is None:
      world_mask = self._component_solver._workspace["all_world_mask"]
    self._component_solver.stage_rhs_rows(
        rhs, workspace, rhs_index=rhs_index, world_mask=world_mask)
    solution, component_status = self._component_solver.run_device(
        dynamics["mass_blocks"], workspace,
        dof_ids=(awake_lists["dof_ids"] if awake_lists is not None else None),
        counts=(awake_lists["counts"] if awake_lists is not None else None),
        tendon_armature_blocks=dynamics.get("tendon_armature_blocks"),
        diagonal_add=diagonal_add,
        allow_indefinite=allow_indefinite, rhs_low=rhs_low,
        world_mask=world_mask)
    self._component_solver.merge_world_status(
        component_status, self._component_world_status,
        world_mask=world_mask)
    return (self._copy_component_first_rhs(
        solution, rhs_index, world_mask=world_mask),
            self._component_world_status)

  def _copy_component_first_rhs(self, solution, rhs_index=0, *,
                                world_mask=None):
    """Publish one component RHS in owned contiguous `[B,nv]` storage.

    The solver returns a borrowed `[B,nrhs,max(nv,1)]` tensor; selecting the
    first RHS is strided across worlds and is overwritten by the next solve.
    Copy before any constraint stage can reuse the component RHS workspace.
    """
    torch = self._state._torch
    nv = int(self._mjmodel.nv)
    if (not isinstance(solution, torch.Tensor) or solution.ndim != 3
        or solution.shape[0] != self.batch_size
        or solution.shape[1] < 1 or solution.shape[2] != nv):
      raise ValueError(
          "component solution must have shape [batch,nrhs,nv] with nrhs>=1")
    if self._component_solution_vector is None:
      raise RuntimeError("component solution output was not prepared")
    if nv:
      if world_mask is None:
        world_mask = self._component_solver._workspace.get("all_world_mask")
      publish = getattr(self._component_solver, "publish_solution_rows", None)
      if world_mask is None or publish is None:
        self._component_solution_vector.copy_(solution[:, rhs_index, :])
        self._component_solution_low_vector.copy_(
            self._component_solver._workspace["output_low"][:, rhs_index, :nv])
      else:
        publish(
            solution, self._component_solver._workspace["output_low"],
            self._component_solution_vector,
            self._component_solution_low_vector,
            rhs_index=rhs_index, world_mask=world_mask)
    return self._component_solution_vector

  def _delayed_control(self, time):
    """Read held controls through the canonical device history, without writes."""
    if self._control is None:
      return None
    return self._history_program.samples(
        self._control, self._state._history, time, kind=0)

  def _sync_delay_from_history(self, env_ids=None):
    """Compatibility hook: canonical history needs no shadow synchronization."""

  def _sync_history_from_delay(self, env_ids=None):
    """Compatibility hook: canonical history needs no shadow synchronization."""

  def _record_delay(self, time, success):
    """Record raw forward samples for accepted worlds at pre-step time."""
    torch = self._state._torch
    controls = self._control
    if controls is None:
      controls = torch.zeros((self.batch_size, self._mjmodel.nu),
                             device=self._state._device)
    sensors = self._raw_sensordata
    if sensors is None:
      sensors = torch.zeros((self.batch_size, self._mjmodel.nsensordata),
                            device=self._state._device)
    self._history_program.record(controls, sensors, self._state._history,
                                 time, success)

  def _actuation_force(self, qpos, qvel, poses, act_override=None, time_override=None,
                       position_context=None, world_mask=None, row_mask=None):
    """General actuator force stage with pinned mj_fwdActuation ordering.

    Runs candidate-contact generation first when BODY adhesion transmissions
    exist (same-step contacts, mirroring mj_fwdPosition order), then
    transmission kinematics, gravity-compensation routing, and the fused
    dynamics/force kernel. Stashes per-step act_dot/velocity for the
    activation advance. Returns borrowed MPS qfrc_actuator.
    ``act_override`` supplies stage activation (RK4); default is live state.
    ``time_override`` supplies per-stage query time (RK4).
    """
    actuators = self._actuators
    state = self._state
    contacts = None
    if position_context is None and actuators.meta.has_body_transmission:
      if self._coupled_constraints is None:
        raise ValueError("BODY transmissions require the coupled constraint stage")
      self._coupled_constraints._constants["body_dims"][3] = actuators.meta.nu
      self._coupled_constraints.generate_candidates(poses, qvel)
      contacts = self._coupled_constraints.contact_buffers()
    kin = (actuators.run_kinematics(qpos, qvel, poses, contacts,
                                   world_mask=world_mask)
           if position_context is None else
           actuators.run_velocity_kinematics(position_context, qvel))
    spatial_jac = (self._spatial_jacobian(qvel, poses, world_mask=row_mask)
                   if position_context is None else None)
    if spatial_jac is not None:
      # Complete spatial-tendon actuator inputs (R1): the actuator
      # kinematics stage only holds fixed-tendon maps, so overwrite
      # length/velocity rows and add gear*ten_J moment rows for actuators
      # targeting spatial tendons.
      self._spatial_tendons.apply_spatial_tendon_state(
          actuators.meta, self._spatial_kin, kin,
          world_mask=world_mask)
    self._last_actuation_kin = kin
    gravcomp = None
    if (self._passive is not None
        and not (int(actuators.meta.disableflags)
                 & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)) and bool(
        np.any(np.asarray(actuators.meta.jnt_actgravcomp)))):
      gravcomp = self._passive.gravcomp_device(
          qpos, getattr(state, "_mpos", None), getattr(state, "_mquat", None),
          # `poses` already belongs to this ACT stage. Reusing it avoids a
          # second private FK pass that could read/write unselected worlds
          # during masked AUTORESET recovery.
          poses=poses,
          world_mask=world_mask,
          awake_lists=(self._sleep_schedule.awake_lists()
                       if getattr(self, "_sleep_schedule", None) is not None
                       else None))
    act = act_override if act_override is not None else getattr(state, "_act", None)
    if actuators.meta.na > 0 and act is None:
      raise ValueError("activation state is missing")
    eval_time = time_override if time_override is not None else state._time
    controls = self._delayed_control(eval_time)
    out = actuators.run_forces(
        controls, act, kin, gravcomp, time=eval_time,
        plugin_ctrl=self._control if self._control is not None else controls,
        world_mask=world_mask,
        callback_mask=row_mask,
        callback_state=_plugin_stage_state(
            state, qpos, qvel, act=act, time=eval_time))
    if (row_mask is None and
        (self._implicit is not None or self._implicitfast is not None
         or self._effective_implicit is not None)):
      edge_values = (self._velocity_derivative_values.values
                     if self._effective_implicit is not None else None)
      self._actuator_velocity_derivative = actuators.run_velocity_derivative(
          controls, act, kin, edge_values=edge_values,
          world_mask=world_mask)
    if self._sensors is not None and self._has_acc_sensors:
      self._lazy_sensor_scratch()
      force = out["force"].reshape(self._sen_act_force.shape)
      qfrc = out["qfrc"].reshape(self._sen_qfrc_act.shape)
      if row_mask is None:
        self._sen_act_force.add_(force)
        self._sen_qfrc_act.add_(qfrc)
      else:
        self._state.update_masked_rows(self._sen_act_force, force, row_mask)
        self._state.update_masked_rows(self._sen_qfrc_act, qfrc, row_mask)
    na, nu = actuators.meta.na, actuators.meta.nu
    if na > 0:
      act_dot = out["act_dot"].reshape(self._act_dot.shape)
      if row_mask is None:
        self._act_dot.copy_(act_dot)
      else:
        self._state.copy_masked_rows(self._act_dot, act_dot, row_mask)
    if nu > 0:
      velocity = kin["velocity"].reshape(self._act_vel.shape)
      if row_mask is None:
        self._act_vel.copy_(velocity)
      else:
        self._state.copy_masked_rows(self._act_vel, velocity, row_mask)
    if actuators.meta.nv == 0:
      return self._state._torch.zeros_like(self._rhs)
    return out["qfrc"].reshape(self._rhs.shape)

  def _transmission_gravcomp(self, qpos, poses, world_mask=None,
                             awake_lists=None):
    """Return source qfrc_gravcomp for stateless transmission actuation."""
    transmissions = self._transmissions
    passive = self._passive
    if transmissions is None or passive is None:
      return None
    if (not np.any(transmissions._meta.jnt_actgravcomp)
        or transmissions._meta.disableflags
        & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)):
      return None
    return passive.gravcomp_device(
        qpos, getattr(self._state, "_mpos", None),
        getattr(self._state, "_mquat", None), poses=poses,
        world_mask=world_mask, awake_lists=awake_lists)

  def _lazy_sensor_scratch(self):
    """Allocate (no-op after first call) ACC capture scratch."""
    if self._sen_act_force is None:
      torch = self._state._torch
      dev = self._state._device
      b, nu, nv = self.batch_size, int(self._mjmodel.nu), int(self._mjmodel.nv)
      self._sen_act_force = torch.zeros((b, max(nu, 1)), dtype=torch.float32, device=dev)
      self._sen_qfrc_act = torch.zeros((b, max(nv, 1)), dtype=torch.float32, device=dev)
      self._sen_jnt_map = None
      self._sen_ten_map = None

  def _limit_row_maps(self):
    """Host (nj,3)/(nt,2) candidate limit-row maps for ACC sensors.

    Joint rows follow the coupled layout (ball joints use the first
    reserved row only); tendon rows follow ten_base + friction-count
    ordering. Models without the integrated coupled stage get all -1
    (limit-force sensors there are rejected at construction).
    """
    if self._sen_jnt_map is not None:
      return self._sen_jnt_map, self._sen_ten_map
    import mujoco as _mj
    torch = self._state._torch
    dev = self._state._device
    m = self._mjmodel
    nj, nt = int(m.njnt), int(m.ntendon)
    jmap = np.full((max(nj, 1), 3), -1, dtype=np.int32)
    tmap = np.full((max(nt, 1), 2), -1, dtype=np.int32)
    cc = getattr(self, "_coupled_constraints", None)
    if cc is not None:
      d = cc.descriptor
      HINGE, SLIDE, BALL = (int(_mj.mjtJoint.mjJNT_HINGE),
                            int(_mj.mjtJoint.mjJNT_SLIDE),
                            int(_mj.mjtJoint.mjJNT_BALL))
      for j in range(nj):
        t = int(np.asarray(m.jnt_type)[j])
        if t in (HINGE, SLIDE):
          jmap[j] = [d.n_eq_rows + d.nv + 2 * j, d.n_eq_rows + d.nv + 2 * j + 1,
                     int(np.asarray(m.jnt_dofadr)[j])]
        elif t == BALL:
          jmap[j] = [d.n_eq_rows + d.nv + 2 * j, -1,
                     int(np.asarray(m.jnt_dofadr)[j])]
      nfric = int(d.ten_friction_rows)
      pre = 0
      for t in range(nt):
        if bool(np.asarray(m.tendon_limited)[t]):
          tmap[t] = [d.ten_base + nfric + 2 * pre, d.ten_base + nfric + 2 * pre + 1]
          pre += 1
    self._sen_jnt_map = torch.as_tensor(jmap.copy(), dtype=torch.int32, device=dev)
    self._sen_ten_map = torch.as_tensor(tmap.copy(), dtype=torch.int32, device=dev)
    return self._sen_jnt_map, self._sen_ten_map

  def _contact_views(self):
    """Borrowed coupled contact views for ACC sensors, or None."""
    cc = getattr(self, "_coupled_constraints", None)
    if cc is None:
      return None
    d = cc.descriptor
    nc = int(d.ncontacts_max)
    if nc == 0:
      return None
    w = cc._workspace
    b = self.batch_size
    row_data = w["contact_row_data"][: b * nc * 36].reshape(b, nc, 6, 6)
    return {
        "frame": w["contact_frame"][: b * nc * 12].reshape(b, nc, 12),
        "force": w["out_contact_force"][: b * nc * 11].reshape(b, nc, 11),
        "contact_mask": row_data[:, :, 0, 0],
        "row": row_data[:, :, 0, 0],
        "packed": cc._constants["contact_condim"].reshape(-1),
        "contact_condim": cc._constants["contact_condim"].reshape(nc, 3),
        "mu": cc._constants["contact_friction"].reshape(nc, 5),
        "pair_geoms": cc._constants["pair_geoms"].reshape(-1),
        "pair_offset": cc._constants["pair_contact_offset"].reshape(-1),
        "slot_pair": self._sen_slot_pair,
        "pair_live": self._sen_pair_live,
        "npairs": int(d.npairs),
    }

  def _touch_grid_contact_views(self):
    """Return contact inputs with the mask packed into owned contiguous storage.

    Canonical CC row views are strided because the activity bit is one field in
    a larger row record. The touch-grid MSL binding consumes a flat contiguous
    mask, so stage that one field into the sensor-owned persistent backing.
    """
    contact = self._contact_views()
    if contact is None:
      return None
    mask = contact["contact_mask"]
    target = self._touch_grid._contact_mask_source
    target.zero_()
    if mask.numel():
      target[:, :mask.shape[1]].copy_(mask)
    contact["contact_mask"] = target
    return contact

  def _run_touch_grid_into(self, poses, out, *, compute_worlds=None,
                           record_status=False):
    """Run bundled touch_grid sensors from this stage's canonical contacts."""
    if not self._has_touch_grid_sensors:
      return out
    compute = self._history_program.sensor_compute_mask(
        self._state._history, self._state._time)
    masks = []
    for sensor in self._touch_grid.sensors:
      tick = compute[:, sensor.address:sensor.address + sensor.dimension].any(
          dim=1).contiguous()
      if compute_worlds is not None:
        tick = (tick & compute_worlds).contiguous()
      masks.append(tick)
      result = self._touch_grid.run_device(
        self._touch_grid_contact_views(), poses, out,
        compute_masks=tuple(masks))
    if record_status:
      bad = self._touch_grid._status != 0
      self._sensor_plugin_status.copy_(self._state._torch.where(
          bad, self._touch_grid._status, self._sensor_plugin_status))
    return result

  def _run_spatial_into(self, poses, out, *, program=None, world_mask=None):
    """Evaluate CONTACT/ray/geomdist families into ``out`` (borrowed)."""
    program = self._sensors if program is None else program
    # Rays/geomdist use the sensor hull store (all scene meshes/hfields,
    # including contype-0 geoms that never appear in contact pairs).
    hull = program._sp_hull.reshape(-1)
    hinfo = program._sp_hull_info.reshape(-1)
    if self._has_contact_sensors:
      out = program.run_contact_device(
          poses, self._contact_views(), out=out, world_mask=world_mask)
    if self._has_ray_sensors or self._has_geomdist_sensors:
      if self._has_ray_sensors:
        out = program.run_rays_device(
            poses, hull, hinfo, out=out, world_mask=world_mask)
      if self._has_geomdist_sensors:
        out = program.run_geomdist_device(
            poses, hull, hinfo, out=out, world_mask=world_mask)
    return out

  def _cache_sensor_acceleration(self, qacc, qacc_low=None):
    """Bind paired ACC provenance to the exact returned high tensor version."""
    self._last_sensor_qacc = qacc
    self._last_sensor_qacc_version = int(qacc._version)
    self._last_sensor_qacc_generation = self._state.generation
    if qacc_low is not None:
      if (not isinstance(qacc_low, self._state._torch.Tensor)
          or tuple(qacc_low.shape) != tuple(qacc.shape)):
        raise ValueError("qacc_low must match the published qacc tensor shape")
    self._last_sensor_qacc_low = qacc_low

  def _qacc_low_for_sensor(self, qacc):
    """Return only the paired residual belonging to this exact solver result."""
    if (getattr(self, "_last_sensor_qacc", None) is qacc
        and getattr(self, "_last_sensor_qacc_generation", None)
            == self._state.generation
        and getattr(self, "_last_sensor_qacc_version", None)
            == int(qacc._version)):
      candidate = getattr(self, "_last_sensor_qacc_low", None)
      if (isinstance(candidate, self._state._torch.Tensor)
          and tuple(candidate.shape) == tuple(qacc.shape)):
        return candidate
    coupled_result = getattr(self, "_last_coupled", None)
    if (not isinstance(coupled_result, dict)
        or getattr(self, "_last_coupled_generation", None)
            != self._state.generation
        or coupled_result.get("qacc") is not qacc
        or getattr(self, "_last_coupled_qacc_version", None)
            != int(qacc._version)):
      return None
    candidate = coupled_result.get("qacc_low")
    if (isinstance(candidate, self._state._torch.Tensor)
        and tuple(candidate.shape) == tuple(qacc.shape)):
      return candidate
    return None

  def _run_acc_into(self, qpos, qvel, qacc, poses, dynamics, out, *,
                    qacc_low=_AUTO_SENSOR_QACC_LOW, program=None,
                    world_mask=None):
    """Evaluate ACC force families into ``out`` (borrowed, merged)."""
    program = self._sensors if program is None else program
    if qacc_low is _AUTO_SENSOR_QACC_LOW:
      qacc_low = self._qacc_low_for_sensor(qacc)
    cc = getattr(self, "_coupled_constraints", None)
    lam = None
    lam_nr = lam_s = 0
    eqr = None
    if cc is not None:
      d = cc.descriptor
      lam_nr, lam_s = int(d.nr), int(cc._debug_stride)
      lam = cc._workspace["workspace_debug"].reshape(-1)
      eqr = cc._constants["eq_rowadr"].reshape(-1)
    jm, tm = self._limit_row_maps()
    contact = self._contact_views()
    return program.run_acc_device(
        qpos, qvel, qacc, poses, qacc_low=qacc_low,
        xfrc=self._body_wrench, contact=contact,
        eq_rowadr=eqr, jnt_map=jm, ten_map=tm, slot_pair=self._sen_slot_pair,
        lam_raw=lam, lam_nr=lam_nr, lam_stride=lam_s,
        act_force=self._sen_act_force, qfrc_act=self._sen_qfrc_act, out=out,
        world_mask=world_mask)

  def _capture_forward_position(self, qpos, dynamics):
    """Retain an actual forward POS stage for a later velocity-only refresh.

    MuJoCo's RK4 restores qpos before its sleep refresh but leaves stage-four
    position fields in mjData. Retaining only FK poses is insufficient: mass,
    tendon paths, transmission moments and canonical constraint rows must all
    describe that same stage. This internal context is invalid after another
    capture, reset, restore or replacement of a prepared workspace.
    """
    # Reject incomplete producers before overwriting any existing cache.
    for program, method in ((self._coupled_constraints, "capture_position_context"),
                            (self._flex, "capture_position_context")):
      if program is not None and not callable(getattr(program, method, None)):
        raise NotImplementedError(
            f"{type(program).__name__} has no complete cached position producer")
    if self._actuators is not None and getattr(self, "_last_actuation_kin", None) is None:
      raise RuntimeError("actuator position stage has not been evaluated")
    # Invalidate the preceding context before the first producer writes, so
    # an exception cannot leave a seemingly valid partially replaced stage.
    epoch = getattr(self, "_forward_position_epoch", 0) + 1
    self._forward_position_epoch = epoch
    context = {"smooth": self._smooth.position_context(dynamics)}
    if self._tendons is not None:
      target = self._forward_position_buffers["fixed_tendon_length"]
      target.copy_(self._tendons._last_length)
      context["fixed_tendon_length"] = target
    if self._spatial_tendons is not None:
      target = self._forward_position_buffers["spatial_tendons"]
      for name, value in target.items():
        value.copy_(self._spatial_kin[name])
      context["spatial_tendons"] = target
    if self._actuators is not None:
      target = self._forward_position_buffers["actuators"]
      for name, value in target.items():
        value.copy_(self._last_actuation_kin[name])
      context["actuators"] = target
    if self._transmissions is not None:
      context["transmissions"] = self._transmissions.capture_position_context(qpos)
    if self._coupled_constraints is not None:
      context["coupled"] = self._coupled_constraints.capture_position_context()
    if self._flex is not None:
      context["flex"] = self._flex.capture_position_context()
    context.update(_owner=self, _epoch=epoch,
                   _generation=self._state.generation)
    return context

  def _update_energy_position(self, qpos, dynamics, context, *, world_mask=None):
    """Refresh pinned ``mj_energyPos`` into the owned two-value stage buffer.

    Static model metadata is read on the host; all state-dependent arithmetic
    remains on MPS. This mirrors engine_sensor.c: gravity, joint springs,
    tendon springs, and one-dimensional flex edge springs.
    """
    if self._energy is None:
      return None
    torch, model = self._state._torch, self._mjmodel
    if world_mask is None:
      out = self._energy[:, 0]
    else:
      if (not isinstance(world_mask, torch.Tensor)
          or world_mask.dtype != torch.bool
          or tuple(world_mask.shape) != (self.batch_size,)
          or world_mask.device != self._state._qvel.device
          or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous bool[batch]")
      self._energy_stage.copy_(self._energy)
      dims = self._energy_position_dims
      dims[12:12 + self.batch_size].copy_(world_mask)
      scheduler = getattr(self, "_sleep_schedule", None)
      awake = scheduler.awake_lists() if scheduler is not None else None
      body_awake = (awake.get("body_mask") if awake is not None
                    else self._energy_joint_int)
      body_count = (awake.get("body_count") if awake is not None
                    else self._energy_joint_int)
      tree_awake = (scheduler.tree_awake if scheduler is not None
                    else self._energy_tendon_int)
      fixed_length = (
          context.get("fixed_tendon_length") if context is not None
          and "fixed_tendon_length" in context else
          self._tendons._last_length if self._tendons is not None else
          self._energy_tendon_float)
      spatial_length = (
          context["spatial_tendons"]["length"] if context is not None
          and "spatial_tendons" in context else
          self._spatial_kin["length"] if self._spatial_tendons is not None else
          fixed_length)
      flex_length = (
          context["flex"]["buffers"]["_flexedge_length"]
          if context is not None and "flex" in context else
          self._flex._flexedge_length if self._flex is not None else
          self._energy_flex_float)
      self._smooth._library.masked_energy_position(
          qpos.reshape(-1), dynamics["poses"]["inertial_pos"].reshape(-1),
          self._energy_body_mass, self._energy_gravity,
          self._energy_qpos_spring, self._energy_joint_int,
          self._energy_joint_float, fixed_length.reshape(-1),
          spatial_length.reshape(-1), self._energy_tendon_int,
          self._energy_tendon_float, flex_length.reshape(-1),
          self._energy_flex_int, self._energy_flex_float,
          body_awake.reshape(-1), body_count.reshape(-1),
          tree_awake.reshape(-1), dims, self._energy_stage.reshape(-1),
          threads=(self.batch_size,), group_size=(1,))
      self._state.copy_masked_rows(
          self._energy, self._energy_stage, world_mask)
      return self._energy[:, 0]
    out.zero_()
    disabled = int(model.opt.disableflags)
    dflags = mujoco.mjtDisableBit
    poses = dynamics["poses"]
    if not disabled & int(dflags.mjDSBL_GRAVITY):
      if model.nbody > 1:
        out.sub_((poses["inertial_pos"][:, 1:] * self._energy_gravity).sum(-1).mul(
            self._energy_body_mass[:model.nbody - 1].unsqueeze(0)).sum(1))
    if not disabled & int(dflags.mjDSBL_SPRING):
      scheduler = getattr(self, "_sleep_schedule", None)
      awake_lists = (scheduler.awake_lists() if scheduler is not None else None)
      body_awake_mask = (awake_lists.get("body_mask")
                         if awake_lists is not None else None)
      body_count = (awake_lists.get("body_count")
                    if awake_lists is not None else None)
      sleep_filter = (body_count < int(model.nbody)
                      if body_count is not None else None)
      jtype = np.asarray(model.jnt_type)
      jq = np.asarray(model.jnt_qposadr)
      jb = np.asarray(model.jnt_bodyid)
      qspring = self._energy_qpos_spring
      stiff = np.asarray(model.jnt_stiffness, dtype=np.float32)
      poly = np.asarray(model.jnt_stiffnesspoly, dtype=np.float32).reshape(
          int(model.njnt), 2)
      free = int(mujoco.mjtJoint.mjJNT_FREE)
      ball = int(mujoco.mjtJoint.mjJNT_BALL)
      slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
      hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
      for jid in range(int(model.njnt)):
        k, p0, p1 = float(stiff[jid]), float(poly[jid, 0]), float(poly[jid, 1])
        if k == 0.0 and p0 == 0.0 and p1 == 0.0:
          continue
        adr, jt = int(jq[jid]), int(jtype[jid])
        if jt in (slide, hinge):
          x = qpos[:, adr] - qspring[adr]
          value = _poly_potential(k, p0, p1, x)
        elif jt == ball:
          x = _quat_difference_angle(qpos[:, adr:adr+4],
                                     qspring[adr:adr+4], torch)
          value = _poly_potential(k, p0, p1, x)
        elif jt == free:
          delta = qpos[:, adr:adr+3] - qspring[adr:adr+3]
          xlin = torch.linalg.vector_norm(delta, dim=-1)
          xang = _quat_difference_angle(
              qpos[:, adr+3:adr+7],
              qspring[adr+3:adr+7], torch)
          value = (_poly_potential(k, p0, p1, xlin)
                   + _poly_potential(k, p0, p1, xang))
        else:
          continue
        if body_awake_mask is not None and sleep_filter is not None:
          sleeping = sleep_filter & (body_awake_mask[:, int(jb[jid])] == 0)
          value = torch.where(sleeping, torch.zeros_like(value), value)
        out.add_(value)

      if self._mjmodel.ntendon and (self._tendons is not None
                                    or self._spatial_tendons is not None):
        if context is not None and "fixed_tendon_length" in context:
          fixed_length = context["fixed_tendon_length"]
        elif self._tendons is not None:
          fixed_length = self._tendons._last_length
        else:
          fixed_length = self._spatial_kin["length"]
        lengths = fixed_length
        if self._spatial_tendons is not None:
          spatial = (context["spatial_tendons"]["length"]
                     if context is not None and "spatial_tendons" in context
                     else self._spatial_kin["length"])
          paths = self._spatial_tendons._meta.paths
          if any(path is not None for path in paths):
            lengths = fixed_length.clone()
            for tid, path in enumerate(paths):
              if path is not None:
                lengths[:, tid] = spatial[:, tid]
        springs = np.asarray(model.tendon_stiffness, dtype=np.float32)
        springpoly = np.asarray(model.tendon_stiffnesspoly,
                                dtype=np.float32).reshape(int(model.ntendon), 2)
        limits = np.asarray(model.tendon_lengthspring, dtype=np.float32).reshape(
            int(model.ntendon), 2)
        ntrees = np.asarray(model.tendon_treenum)
        treeids = np.asarray(model.tendon_treeid).reshape(int(model.ntendon), 2)
        tree_awake = getattr(scheduler, "tree_awake", None)
        for tid in range(int(model.ntendon)):
          k = float(springs[tid]); p0 = float(springpoly[tid, 0]); p1 = float(springpoly[tid, 1])
          if k == 0.0 and p0 == 0.0 and p1 == 0.0:
            continue
          lo, hi = map(float, limits[tid])
          length = lengths[:, tid]
          x = torch.where(length > hi, length-hi,
                          torch.where(length < lo, length-lo,
                                      torch.zeros_like(length)))
          value = _poly_potential(k, p0, p1, x)
          if tree_awake is not None and sleep_filter is not None:
            count = int(ntrees[tid])
            if count == 0:
              sleeping = sleep_filter
            elif count == 1:
              sleeping = ((tree_awake[:, int(treeids[tid, 0])] == 0)
                          & sleep_filter)
            elif count == 2:
              a = tree_awake[:, int(treeids[tid, 0])] != 0
              b = tree_awake[:, int(treeids[tid, 1])] != 0
              sleeping = (~(a | b)) & sleep_filter
            else:
              sleeping = torch.zeros_like(sleep_filter)
            # Pinned mj_tendonSleepState treats a tendon spanning more than
            # two trees as awake unconditionally.
            value = torch.where(sleeping, torch.zeros_like(value), value)
          out.add_(value)

      if self._flex is not None:
        flex_dim = np.asarray(model.flex_dim)
        flex_rigid = np.asarray(model.flex_rigid)
        edge_adr = np.asarray(model.flex_edgeadr)
        edge_num = np.asarray(model.flex_edgenum)
        edge_rigid = np.asarray(model.flexedge_rigid)
        rest = np.asarray(model.flexedge_length0, dtype=np.float32)
        edge_k = np.asarray(model.flex_edgestiffness, dtype=np.float32)
        lengths = (context["flex"]["buffers"]["_flexedge_length"]
                   if context is not None and "flex" in context
                   else self._flex._flexedge_length)
        for fid in range(int(model.nflex)):
          k = float(edge_k[fid])
          if bool(flex_rigid[fid]) or int(flex_dim[fid]) > 1 or k == 0.0:
            continue
          first, count = int(edge_adr[fid]), int(edge_num[fid])
          for edge in range(first, first + count):
            if bool(edge_rigid[edge]):
              continue
            dx = float(rest[edge]) - lengths[:, edge]
            out.add_(0.5 * k * dx.square())
    if world_mask is not None:
      self._state.update_masked_rows(
          self._energy, self._energy_stage, world_mask, add=False)
      return self._energy[:, 0]
    return out

  def _update_energy_velocity(self, velocity, *, world_mask=None):
    """Refresh pinned ``mj_energyVel`` using the complete prepared mass."""
    if self._energy is None:
      return None
    dynamics = velocity["dynamics"]
    qvel = velocity["qvel"]
    nv, batch = int(self._mjmodel.nv), self.batch_size
    if world_mask is not None:
      if (not isinstance(world_mask, self._state._torch.Tensor)
          or world_mask.dtype != self._state._torch.bool
          or tuple(world_mask.shape) != (batch,)
          or world_mask.device != self._state._qvel.device
          or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous bool[batch]")
      self._state._reset_mask_i32.copy_(world_mask)
    if self._component_mass_enabled:
      mass_product = self._smooth.mass_blocks_matvec_device(
          dynamics["mass_blocks"], qvel,
          dynamics.get("tendon_armature_blocks"),
          world_mask=(None if world_mask is None
                      else self._state._reset_mask_i32))
      mass_input = (mass_product.reshape(-1) if nv else
                    self._smooth._workspace["mass_sparse_product"])
      has_product = 1
    else:
      mass = dynamics["mass_matrix"]
      mass_input = (mass.reshape(-1) if nv else
                    self._smooth._workspace["mass_sparse_product"])
      has_product = 0
    dims = self._smooth._workspace["energy_velocity_dims"]
    dims[0], dims[1], dims[2] = nv, batch, has_product
    if world_mask is None:
      dims[3:].fill_(1)
      output = self._energy
    else:
      dims[3:].copy_(self._state._reset_mask_i32)
      self._energy_stage.copy_(self._energy)
      output = self._energy_stage
    qvel_input = (qvel.reshape(-1) if nv else
                  self._smooth._workspace["mass_sparse_product"])
    self._smooth._library.masked_energy_velocity(
        mass_input, qvel_input, dims, output.reshape(-1),
        threads=(batch,), group_size=(1,))
    if world_mask is not None:
      self._state.copy_masked_rows(self._energy, self._energy_stage, world_mask)
    return self._energy[:, 1]

  def prepare_forward_position(self, qpos=None, *, mocap_pos=None,
                               mocap_quat=None, skipsensor=False):
    """Evaluate and publish the native POS stage without velocity physics.

    The operation prepares FK/COM/CRBA, tendon and actuator position
    kinematics, and the available coupled row assembly before capturing all
    reusable position contexts. It intentionally supplies an owned zero
    velocity view to position-dependent producers; SmoothDynamics skips RNE
    entirely in this entry point. Profiles with legacy constraint builders or
    unadmitted flex contacts fail closed in their producer methods.
    """
    state, torch = self._state, self._state._torch
    qpos = state._qpos if qpos is None else qpos
    expected_qpos = (self.batch_size, int(self._mjmodel.nq))
    if (not isinstance(qpos, torch.Tensor) or tuple(qpos.shape) != expected_qpos
        or qpos.dtype != torch.float32 or qpos.device != state._qpos.device
        or not qpos.is_contiguous()):
      raise ValueError(f"qpos must be contiguous float32 MPS {expected_qpos}")
    # A failed replacement must not leave an older record pointing at
    # position workspaces that this attempt is about to overwrite.
    self._step1_record = None
    self._forward_stages.invalidate()
    self._forward_position_epoch = getattr(self, "_forward_position_epoch", 0) + 1
    if mocap_pos is None:
      mocap_pos = getattr(state, "_mpos", None)
    if mocap_quat is None:
      mocap_quat = getattr(state, "_mquat", None)
    scheduler = getattr(self, "_sleep_schedule", None)
    cc = getattr(self, "_coupled_constraints", None)
    needs_flex_wake = bool(
        scheduler is not None and self._flex is not None and cc is not None
        and int(cc.descriptor.n_flex_contact_rows) > 0)
    zero_qvel = self._smooth._workspace["qvel"][:
        self.batch_size * self._mjmodel.nv].view(
            self.batch_size, self._mjmodel.nv)
    zero_qvel.zero_()
    presolve_poses = None
    candidate_token = None
    if needs_flex_wake:
      # Flex contact wake links need current point Jacobians. Build their
      # position inputs from a complete all-awake POS pass before the wake
      # scheduler chooses the final per-world body/DOF lists. This reuses the
      # existing smooth workspace; its outputs are deliberately replaced by
      # the final masked POS pass below.
      presolve_poses = self._probe_sleep_pose_mismatch(
          qpos, state._qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat)
      presolve_tendon_J = self._component_tendon_jacobian(
          zero_qvel, presolve_poses)
      all_awake = self._smooth._workspace["all_awake_lists"]
      presolve_dynamics = self._smooth.run_position_device(
          qpos, poses=presolve_poses, awake_lists=all_awake,
          tendon_J=presolve_tendon_J)
      velocity_inputs = self._smooth._run_velocity_bias(
          state._qvel, all_awake, cdof=presolve_dynamics["cdof"],
          local_inertia=self._smooth._workspace["local_inertia"])
      presolve_poses, candidate_token = self._prepare_sleep_schedule(
          qpos, state._qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat,
          poses=presolve_poses, cvel=velocity_inputs["cvel"],
          cdof=presolve_dynamics["cdof"],
          root_com=presolve_dynamics["root_com"],
          return_candidate_token=True,
          pose_mismatch_checked=True,
          pose_status=presolve_dynamics.get("pose_status"))
    elif scheduler is not None:
      presolve_poses, candidate_token = self._prepare_sleep_schedule(
          qpos, state._qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat,
          return_candidate_token=True)
    awake_lists = scheduler.awake_lists() if scheduler is not None else None
    if self._component_mass_enabled and awake_lists is None:
      awake_lists = self._smooth._workspace["all_awake_lists"]
    tree_awake = (scheduler.tree_awake if scheduler is not None else None)
    poses = (presolve_poses if presolve_poses is not None else
             self._smooth._fk.run_device(
                 qpos, mocap_pos, mocap_quat, tree_awake=tree_awake))
    self._spatial_cache_key = None
    tendon_J = self._component_tendon_jacobian(zero_qvel, poses)
    dynamics = self._smooth.run_position_device(
        qpos, poses=poses, awake_lists=awake_lists, tendon_J=tendon_J)
    if self._tendons is not None:
      _, _, fixed_armature = self._tendons.run_device(qpos, zero_qvel)
      # The dense POS mass is the operator consumed by every later stage,
      # including cached VEL/ACC and inverse queries. Tendon armature is part
      # of MuJoCo's position-stage M, so add it before capturing the stage.
      # The component path projects the same contribution into its packed
      # tree blocks in _component_tendon_jacobian instead.
      if not self._component_mass_enabled and fixed_armature is not None:
        dynamics["mass_matrix"].add_(fixed_armature)
    if self._spatial_tendons is not None:
      self._spatial_kin = self._spatial_tendons.run_kinematics(
          zero_qvel, dynamics["poses"])
      _, _, spatial_armature = self._spatial_tendons.run_forces(
          self._spatial_kin, include_armature=not self._component_mass_enabled)
      if (not self._component_mass_enabled and spatial_armature is not None):
        dynamics["mass_matrix"].add_(
            spatial_armature.reshape(dynamics["mass_matrix"].shape))
      self._spatial_cache_key = (state.generation, id(zero_qvel),
                                 id(dynamics["poses"].get("body_pos")))
    contacts = None
    if (self._actuators is not None
        and self._actuators.meta.has_body_transmission):
      if self._coupled_constraints is None:
        raise ValueError("BODY transmissions require the coupled constraint stage")
      self._coupled_constraints._constants["body_dims"][3] = self._actuators.meta.nu
      if candidate_token is None:
        flex_for_candidates = (
            self._flex if int(self._coupled_constraints.descriptor
                              .n_flex_contact_rows) > 0 else None)
        candidate_poses = dynamics["poses"]
        if flex_for_candidates is not None:
          candidate_poses = dict(candidate_poses)
          candidate_poses["root_com"] = dynamics["root_com"]
          candidate_poses["cdof"] = dynamics["cdof"]
        candidate_token = self._coupled_constraints.generate_candidates(
            candidate_poses, zero_qvel, flex=flex_for_candidates,
            cvel=dynamics["cvel"], cdof=dynamics["cdof"])
      contacts = self._coupled_constraints.contact_buffers()
    if self._actuators is not None:
      actuator_qvel = zero_qvel
      if not self._mjmodel.nv:
        actuator_qvel = self._forward_stage_actuator_qvel
        actuator_qvel.zero_()
      kin = self._actuators.run_kinematics(
          qpos, actuator_qvel, dynamics["poses"], contacts)
      if self._spatial_tendons is not None:
        self._spatial_tendons.apply_spatial_tendon_state(
            self._actuators.meta, self._spatial_kin, kin)
      self._last_actuation_kin = kin
    if self._flex is not None:
      self._flex.update_kinematics(
          dict(dynamics["poses"], root_com=dynamics["root_com"]),
          dynamics["cvel"])
    exact_impedance_status = None
    if self._coupled_constraints is not None:
      if self._contact is not None or self._joint_constraints is not None:
        raise NotImplementedError(
            "split POS is unavailable for legacy constraint builders")
      _ten_j, _ten_l = self._spatial_for_coupled(zero_qvel, dynamics["poses"])
      assembled_rows = self._coupled_constraints.assemble_device(
          dict(dynamics["poses"], root_com=dynamics["root_com"]), qpos,
          zero_qvel, eq_active=getattr(state, "_eq_active", None),
          cvel=dynamics["cvel"], cdof=dynamics["cdof"],
          cdof_dot=dynamics["cdof_dot"], tendon_J_spatial=_ten_j,
          tendon_length_spatial=_ten_l, flex=self._flex,
          awake_tree=tree_awake, candidate_token=candidate_token,
          exact_impedance=self._constraint_impedance,
          mass=(None if self._component_mass_enabled
                else dynamics.get("mass_matrix")),
          exact_dense_solver=self._exact_impedance_dense_solver,
          component_solver=(self._component_solver
                            if self._component_mass_enabled else None),
          mass_blocks=(dynamics.get("mass_blocks")
                       if self._component_mass_enabled else None),
          dof_ids=(awake_lists["dof_ids"]
                   if self._component_mass_enabled and awake_lists is not None
                   else None),
          counts=(awake_lists["counts"]
                  if self._component_mass_enabled and awake_lists is not None
                  else None),
          tendon_armature_blocks=(dynamics.get("tendon_armature_blocks")
                                  if self._component_mass_enabled else None))
      exact = assembled_rows.get("exact_impedance")
      if exact is not None:
        self._exact_impedance_status.copy_(exact["status"])
        exact_impedance_status = self._exact_impedance_status
    for plugin in getattr(self, "_native_plugins", ()):
      if getattr(plugin, "stage", None) == "position" or hasattr(
          plugin, "run_position_device"):
        raise NotImplementedError(
            "native plug-in position-stage callbacks are not admitted")
    context = self._capture_forward_position(qpos, dynamics)
    energy_pos = self._update_energy_position(qpos, dynamics, context)
    if not skipsensor and self._sensors is not None:
      self._store_step_sensors(
          qpos, zero_qvel, dynamics, stages=(mujoco.mjtStage.mjSTAGE_POS,),
          position_context=context, include_spatial=False)
    record = self._forward_stages.begin(
        generation=state.generation, qpos=qpos, mocap_pos=mocap_pos,
        mocap_quat=mocap_quat,
        position={"dynamics": dynamics, "poses": dynamics["poses"],
                  "context": context, "awake_lists": awake_lists,
                  "energy_pos": energy_pos,
                  "exact_impedance_status": exact_impedance_status})
    return record

  def validate_forward_stage_record(self, record, minimum="POS", *,
                                    qpos=None, mocap_pos=None,
                                    mocap_quat=None):
    """Validate a split-stage record against the current state generation."""
    from mujoco_metal.forward_stages import ForwardStage
    stage = ForwardStage[minimum] if isinstance(minimum, str) else minimum
    record = self._forward_stages.validate(
        record, generation=self._state.generation, minimum=stage)
    for name, supplied, captured in (
        ("qpos", qpos, record.qpos),
        ("mocap_pos", mocap_pos, record.mocap_pos),
        ("mocap_quat", mocap_quat, record.mocap_quat)):
      if supplied is not None and supplied is not captured:
        raise ValueError(
            f"{name} does not match the position-stage input record")
    return record

  def _accumulate_plugin_forces(self, plugins, plugin_state, qpos, qvel,
                                dynamics, force_target, *, rhs_target=None,
                                compute_mask=None,
                                require_contiguous=False):
    """Run one plugin role and add its force to its canonical force bucket."""
    if not plugins:
      return
    torch = self._state._torch
    for plugin in plugins:
      kwargs = {} if compute_mask is None else {"compute_mask": compute_mask}
      plugin_force = plugin.run_device(
          plugin_state, qpos=qpos, qvel=qvel, dynamics=dynamics, **kwargs)
      if plugin_force is None:
        continue
      if (not isinstance(plugin_force, torch.Tensor)
          or plugin_force.device != force_target.device
          or plugin_force.dtype != torch.float32
          or tuple(plugin_force.shape) != tuple(force_target.shape)
          or ((require_contiguous or compute_mask is not None)
              and not plugin_force.is_contiguous())):
        raise ValueError(
            f"plugin {_plugin_state_key(plugin)!r} returned an invalid force tensor")
      if compute_mask is None:
        force_target.add_(plugin_force)
      else:
        self._state.update_masked_rows(
            force_target, plugin_force, compute_mask)
      if rhs_target is not None:
        if compute_mask is None:
          rhs_target.add_(plugin_force)
        else:
          self._state.update_masked_rows(
              rhs_target, plugin_force, compute_mask)

  def _run_native_actuator_plugins(self, plugin_state, control, dynamics,
                                  qvel, act, *, force_target,
                                  compute_mask=None):
    """Evaluate typed actuator-space callbacks and project through moment.

    Evaluation is pure with respect to plugin-owned state; stateful callbacks
    advance once in ``_advance_native_actuator_plugins`` after integration.
    The force callback may be evaluated at each RK4 substage.
    """
    plugins = getattr(self, "_native_actuator_plugins", ())
    if not plugins:
      return None
    if self._actuators is None or self._last_actuation_kin is None:
      raise ValueError("NativeActuatorPlugin requires prepared actuator kinematics")
    torch = self._state._torch
    kin = self._last_actuation_kin
    total = self._native_actuator_force
    if compute_mask is None:
      total.zero_()
    else:
      self._state.clear_masked_rows(total, compute_mask)
    outputs = {}
    na = int(getattr(self._state, "_na", 0))
    for plugin in plugins:
      output = plugin.run_actuator_device(
          plugin_state, control=control, kinematics=kin, dynamics=dynamics,
          act=act, act_dot=(self._act_dot if na else None),
          compute_mask=compute_mask)
      from mujoco_metal.extensions import ActuatorPluginOutput
      if not isinstance(output, ActuatorPluginOutput):
        raise TypeError(
            f"native actuator plugin {plugin.name!r} must return ActuatorPluginOutput")
      force = output.force
      expected_force = (self.batch_size, int(self._mjmodel.nu))
      if (not isinstance(force, torch.Tensor)
          or tuple(force.shape) != expected_force
          or force.dtype != torch.float32
          or force.device != self._state._qvel.device
          or not force.is_contiguous()):
        raise ValueError(
            f"native actuator plugin {plugin.name!r} force must be contiguous float32 {expected_force}")
      if compute_mask is None:
        total.add_(force)
      else:
        self._state.update_masked_rows(total, force, compute_mask)
      derivative = output.activation_derivative
      if derivative is not None:
        if not na or (not isinstance(derivative, torch.Tensor)
            or tuple(derivative.shape) != (self.batch_size, na)
            or derivative.dtype != torch.float32
            or derivative.device != self._state._qvel.device
            or not derivative.is_contiguous()):
          raise ValueError(
              f"native actuator plugin {plugin.name!r} activation_derivative must be contiguous float32 {(self.batch_size, na)}")
        if compute_mask is None:
          self._act_dot.add_(derivative)
        else:
          self._state.update_masked_rows(
              self._act_dot, derivative, compute_mask)
      outputs[plugin] = output
    qfrc = self._native_actuator_qfrc
    nv = int(self._mjmodel.nv)
    moment = kin["moment"]
    if (tuple(moment.shape) != (self.batch_size, int(self._mjmodel.nu),
                                max(nv, 1))
        or moment.dtype != torch.float32
        or moment.device != self._state._qvel.device
        or not moment.is_contiguous()):
      raise ValueError("actuator moment is invalid for typed plugin projection")
    self._actuators.project_plugin_force(
        total, moment, qfrc,
        world_mask=(None if compute_mask is None else
                    self._state._reset_mask_i32))
    if compute_mask is None:
      force_target.add_(qfrc)
    else:
      self._state.update_masked_rows(force_target, qfrc, compute_mask)
    self._last_native_actuator_outputs = outputs
    return {"force": total, "qfrc": qfrc,
            "activation_derivative": self._act_dot if na else None}

  def _advance_native_actuator_plugins(self, accepted_mask, *, qpos=None,
                                       qvel=None, qacc=None, act=None,
                                       time=None):
    """Advance typed actuator state at the tentative accepted-step boundary.

    Optional values are the computed next-state tensors. Calling before the
    simulation commits those references keeps a rejecting plugin callback
    inside the surrounding step transaction: physics state is still X0 and
    the exception path can restore plugin-owned state without rolling back a
    partially committed simulation step.
    """
    plugins = getattr(self, "_native_actuator_plugins", ())
    user_plugins = getattr(self, "_actuator_user_plugins", ())
    if not plugins and not user_plugins:
      return
    state, torch = self._state, self._state._torch
    if (not isinstance(accepted_mask, torch.Tensor)
        or accepted_mask.dtype != torch.bool
        or tuple(accepted_mask.shape) != (self.batch_size,)
        or accepted_mask.device != state._qvel.device):
      raise ValueError("accepted_mask must be bool[batch_size] on the simulation device")
    stage = _plugin_stage_state(
        state,
        state._qpos if qpos is None else qpos,
        state._qvel if qvel is None else qvel,
        qacc=state._qacc if qacc is None else qacc,
        act=getattr(state, "_act", None) if act is None else act,
        time=state._time if time is None else time)
    try:
      for plugin in plugins:
        output = self._last_native_actuator_outputs.get(plugin)
        plugin.advance_actuator_device(
            stage, accepted_mask=accepted_mask,
            timestep=float(self.profile.timestep), output=output)
      for plugin in user_plugins:
        if self._actuators is None:
          continue
        outputs = self._actuators._last_user_callback_outputs
        if plugin not in outputs:
          continue
        output = self._actuators.last_user_callback_output(plugin)
        plugin.advance_user_actuator_device(
            stage, accepted_mask=accepted_mask,
            timestep=float(self.profile.timestep), output=output)
    finally:
      # Outputs are borrowed from the preceding evaluation. Never let a
      # failed accepted-step callback reuse them on a later step.
      self._last_native_actuator_outputs = {}
      if self._actuators is not None:
        self._actuators._last_user_callback_outputs = {}

  def prepare_forward_velocity(self, record, qvel=None, *, skipsensor=False):
    """Refresh VEL-owned products from one current captured POS stage."""
    from mujoco_metal.forward_stages import ForwardStage
    record = self.validate_forward_stage_record(record, ForwardStage.POS)
    state, torch = self._state, self._state._torch
    qvel = state._qvel if qvel is None else qvel
    shape = (self.batch_size, int(self._mjmodel.nv))
    if (not isinstance(qvel, torch.Tensor) or tuple(qvel.shape) != shape
        or qvel.dtype != torch.float32 or qvel.device != state._qvel.device
        or not qvel.is_contiguous()):
      raise ValueError(f"qvel must be contiguous float32 MPS {shape}")
    position = self._forward_stages.consume(
        record, ForwardStage.POS, generation=state.generation)
    forward_mask = getattr(record, "row_valid", None)
    world_mask_i32 = None
    if forward_mask is not None:
      world_mask_i32 = state._reset_mask_i32
      world_mask_i32.copy_(forward_mask)
    dynamics = self._smooth.run_velocity_device(
        position["context"]["smooth"], qvel,
        awake_lists=position["awake_lists"], world_mask=world_mask_i32)
    # VEL refreshes motion-dependent fields in the prepared POS owner. Keep
    # the exact published pose dictionary object; callers may hold borrowed
    # POS views whose identity is part of the split-stage contract.
    dynamics["poses"] = position["poses"]
    context = position["context"]
    qfrc_passive = self._forward_stage_passive_force
    if forward_mask is None:
      qfrc_passive.zero_()
    else:
      state.clear_masked_rows(qfrc_passive, forward_mask)
    passive_damping = None
    if self._passive is not None:
      qfrc, passive_damping = self._passive.run_device(
          record.qpos, qvel, xfrc_applied=self._body_wrench,
          return_damping=True, mocap_pos=record.mocap_pos,
          mocap_quat=record.mocap_quat,
          awake_lists=position["awake_lists"], poses=dynamics["poses"],
          world_mask=world_mask_i32)
      if forward_mask is None:
        qfrc_passive.add_(qfrc)
      else:
        state.update_masked_rows(qfrc_passive, qfrc, forward_mask)
    if self._flex is not None:
      qfrc, _, _ = self._flex.run_velocity_device(
          context["flex"], record.qpos, qvel,
          dict(dynamics["poses"], root_com=dynamics["root_com"]),
          dynamics["cvel"], world_mask=world_mask_i32)
      if forward_mask is None:
        qfrc_passive.add_(qfrc)
      else:
        state.update_masked_rows(qfrc_passive, qfrc, forward_mask)
    if self._fluid is not None:
      qfrc = self._fluid.run_device(
          record.qpos, qvel, dynamics, world_mask=world_mask_i32)
      if forward_mask is None:
        qfrc_passive.add_(qfrc)
      else:
        state.update_masked_rows(qfrc_passive, qfrc, forward_mask)
    tendon_damping = None
    if self._tendons is not None:
      qfrc, tendon_damping, _ = self._tendons.run_device(
          record.qpos, qvel,
          length_override=context["fixed_tendon_length"],
          world_mask=world_mask_i32)
      if forward_mask is None:
        qfrc_passive.add_(qfrc)
      else:
        state.update_masked_rows(qfrc_passive, qfrc, forward_mask)
    spatial_damping = None
    if self._spatial_tendons is not None:
      self._spatial_kin = self._spatial_tendons.run_velocity_kinematics(
          context["spatial_tendons"], qvel, world_mask=world_mask_i32)
      qfrc, spatial_damping, _ = self._spatial_tendons.run_forces(
          self._spatial_kin, include_armature=False,
          world_mask=world_mask_i32)
      if forward_mask is None:
        qfrc_passive.add_(qfrc.reshape(shape))
      else:
        state.update_masked_rows(qfrc_passive, qfrc.reshape(shape), forward_mask)
      armature_bias, _ = self._spatial_tendons.run_armature_bias(
          self._spatial_kin, qvel, dynamics["poses"], dynamics["cvel"],
          dynamics["root_com"], dynamics["cdof"], dynamics["cdof_dot"],
          world_mask=world_mask_i32)
      if forward_mask is None:
        dynamics["qfrc_bias"].add_(armature_bias.reshape(shape))
      else:
        state.update_masked_rows(
            dynamics["qfrc_bias"], armature_bias.reshape(shape), forward_mask)
    if self._cable is not None:
      self._cable.run_device(record.qpos, dynamics["poses"], dynamics["cdof"],
                             qfrc_passive, world_mask=world_mask_i32)
    force_plugins = tuple(getattr(self, "_force_plugins", ()))
    before_plugins = _snapshot_plugins_device(force_plugins)
    plugin_state = _plugin_stage_state(
        state, record.qpos, qvel, time=state._time)
    try:
      self._accumulate_plugin_forces(
          force_plugins, plugin_state, record.qpos, qvel, dynamics,
          qfrc_passive, compute_mask=record.row_valid)
    except Exception:
      _restore_plugins_device(before_plugins)
      raise
    if self._coupled_constraints is not None:
      refresh = getattr(self._coupled_constraints,
                        "refresh_velocity_context", None)
      if refresh is None:
        raise NotImplementedError(
            "coupled VEL requires the row-cache velocity refresh stage")
      refresh(context["coupled"], record.qpos, qvel,
              cvel=dynamics["cvel"], cdof=dynamics["cdof"],
              cdof_dot=dynamics["cdof_dot"], world_mask=world_mask_i32)
    row_status = self._forward_stages.invalid_row_status(
        dynamics.get("pose_status"), record)
    if record.row_valid is not None:
      _zero_invalid_rows(qfrc_passive, record.row_valid, self._state)
      _zero_invalid_rows(dynamics.get("qfrc_bias"), record.row_valid,
                         self._state)
      _zero_invalid_rows(dynamics.get("qfrc_bias_low"), record.row_valid,
                         self._state)
    if not skipsensor and self._sensors is not None:
      self._store_step_sensors(
          record.qpos, qvel, dynamics,
          stages=(mujoco.mjtStage.mjSTAGE_VEL,),
          position_context=context, include_spatial=False,
          compute_worlds=record.row_valid)
    result = {
        "dynamics": dynamics,
        "qfrc_bias": dynamics["qfrc_bias"],
        "qfrc_passive": qfrc_passive,
        "passive_damping": passive_damping,
        "tendon_damping": tendon_damping,
        "spatial_damping": spatial_damping,
        "qvel": qvel,
        "position_context": context,
        "status": row_status,
    }
    result["energy_vel"] = self._update_energy_velocity(
        result, world_mask=forward_mask)
    self._forward_stages.publish(record, ForwardStage.VEL, result)
    return result

  def prepare_forward_actuation(self, record, *, ctrl=None, act=None, time=None):
    """Run actuator/transmission force production from the captured stage."""
    saved_control = self._control.clone() if ctrl is not None else None
    try:
      if ctrl is not None:
        self._prepare_control(ctrl)
      return self._prepare_forward_actuation_current(
          record, act=act, time=time)
    finally:
      if saved_control is not None:
        self._control.copy_(saved_control)

  def _prepare_forward_actuation_current(self, record, *, act=None, time=None):
    """Evaluate ACT while its caller owns any temporary control override."""
    from mujoco_metal.forward_stages import ForwardStage
    record = self.validate_forward_stage_record(record, ForwardStage.VEL)
    state, torch = self._state, self._state._torch
    velocity = self._forward_stages.consume(
        record, ForwardStage.VEL, generation=state.generation)
    forward_mask = getattr(record, "row_valid", None)
    world_mask_i32 = None
    if forward_mask is not None:
      # Reuse the device mask backing shared by masked reset and stage kernels.
      # A forward record can remain live for healthy rows after AUTORESET has
      # invalidated only the selected rows.
      world_mask_i32 = state._reset_mask_i32
      world_mask_i32.copy_(forward_mask)
    control = self._delayed_control(state._time if time is None else time)
    dynamics = velocity["dynamics"]
    context = velocity["position_context"]
    qfrc = self._forward_stage_actuator_force
    qfrc.zero_()
    # Prepared forward ACT is a fresh sensor-producing stage, just like the
    # ordinary `_acceleration` path below.  The actuator-force sensor scratch
    # is otherwise persistent across queries/steps, so appending this stage's
    # forces would report the previous sample again (and can double values on
    # repeated `mj_forward` calls).
    if self._sensors is not None and self._has_acc_sensors:
      self._lazy_sensor_scratch()
      if forward_mask is None:
        self._sen_act_force.zero_()
        self._sen_qfrc_act.zero_()
      else:
        self._state.clear_masked_rows(self._sen_act_force, forward_mask)
        self._state.clear_masked_rows(self._sen_qfrc_act, forward_mask)
    if self._transmissions is not None:
      transmission_qfrc = self._transmissions.run_device(
          record.qpos, velocity["qvel"], control,
          awake_lists=self._sleep_schedule.awake_lists()
          if getattr(self, "_sleep_schedule", None) is not None else None,
          position_context=context.get("transmissions"),
          world_mask=world_mask_i32,
          gravcomp=self._transmission_gravcomp(
              record.qpos, velocity["dynamics"]["poses"], world_mask_i32,
              self._sleep_schedule.awake_lists()
              if getattr(self, "_sleep_schedule", None) is not None else None))
      if forward_mask is None:
        qfrc.add_(transmission_qfrc)
      else:
        state.update_masked_rows(qfrc, transmission_qfrc, forward_mask)
      if self._sensors is not None and self._has_acc_sensors:
        force_rows = self._transmissions._last_force.reshape(
            self._sen_act_force.shape)
        qfrc_rows = transmission_qfrc.reshape(self._sen_qfrc_act.shape)
        if forward_mask is None:
          self._sen_act_force.add_(force_rows)
          self._sen_qfrc_act.add_(qfrc_rows)
        else:
          state.update_masked_rows(self._sen_act_force, force_rows, forward_mask)
          state.update_masked_rows(self._sen_qfrc_act, qfrc_rows, forward_mask)
    actuator_result = None
    if self._actuators is not None:
      actuator_result = self._actuation_force(
          record.qpos, velocity["qvel"], dynamics["poses"],
          act_override=act, time_override=time,
          world_mask=world_mask_i32, row_mask=forward_mask,
          position_context=context.get("actuators"))
      if forward_mask is None:
        qfrc.add_(actuator_result)
      else:
        state.update_masked_rows(qfrc, actuator_result, forward_mask)
    if self._motor is not None:
      motor_qfrc = self._motor.run_device(control, world_mask=world_mask_i32)
      if forward_mask is None:
        qfrc.add_(motor_qfrc)
      else:
        state.update_masked_rows(qfrc, motor_qfrc, forward_mask)
      if self._sensors is not None and self._has_acc_sensors:
        if self._motor._last_force is not None:
          force_rows = self._motor._last_force.reshape(
              self._sen_act_force.shape)
          if forward_mask is None:
            self._sen_act_force.add_(force_rows)
          else:
            state.update_masked_rows(self._sen_act_force, force_rows, forward_mask)
        qfrc_rows = motor_qfrc.reshape(self._sen_qfrc_act.shape)
        if forward_mask is None:
          self._sen_qfrc_act.add_(qfrc_rows)
        else:
          state.update_masked_rows(self._sen_qfrc_act, qfrc_rows, forward_mask)
    force_plugins = tuple(getattr(self, "_legacy_actuator_plugins", ()))
    typed_plugins = tuple(getattr(self, "_native_actuator_plugins", ()))
    before_plugins = _snapshot_plugins_device(force_plugins + typed_plugins)
    plugin_state = _plugin_stage_state(
        state, record.qpos, velocity["qvel"], act=act,
        time=state._time if time is None else time)
    try:
      typed_output = self._run_native_actuator_plugins(
          plugin_state, control, dynamics, velocity["qvel"], act,
          force_target=qfrc,
          compute_mask=getattr(record, "row_valid", None)) if typed_plugins else None
      if typed_output is not None and self._sensors is not None and self._has_acc_sensors:
        typed_force = typed_output["force"].reshape(self._sen_act_force.shape)
        typed_qfrc = typed_output["qfrc"].reshape(self._sen_qfrc_act.shape)
        if record.row_valid is None:
          self._sen_act_force.add_(typed_force)
          self._sen_qfrc_act.add_(typed_qfrc)
        else:
          self._state.update_masked_rows(
              self._sen_act_force, typed_force, record.row_valid)
          self._state.update_masked_rows(
              self._sen_qfrc_act, typed_qfrc, record.row_valid)
      self._accumulate_plugin_forces(
          force_plugins, plugin_state, record.qpos, velocity["qvel"],
          dynamics, qfrc, compute_mask=getattr(record, "row_valid", None),
          require_contiguous=True)
    except Exception:
      _restore_plugins_device(before_plugins)
      raise
    _zero_invalid_rows(qfrc, getattr(record, "row_valid", None), self._state)
    result = {"qfrc_actuator": qfrc,
              "actuator_result": actuator_result,
              "native_actuator_output": typed_output,
              "act_dot": (self._act_dot if self._actuators is not None else None),
              "control": control, "time": state._time if time is None else time}
    self._forward_stages.publish(record, ForwardStage.ACT, result)
    return result

  def prepare_forward_acceleration(self, record, *, qfrc_applied=None):
    """Solve unconstrained acceleration from already published force stages."""
    from mujoco_metal.forward_stages import ForwardStage
    record = self.validate_forward_stage_record(record, ForwardStage.ACT)
    state = self._state
    velocity = self._forward_stages.consume(
        record, ForwardStage.VEL, generation=state.generation)
    actuation = self._forward_stages.consume(
        record, ForwardStage.ACT, generation=state.generation)
    dynamics = velocity["dynamics"]
    rhs = self._rhs
    rhs_low = self._rhs_low
    torch = state._torch
    row_valid = record.row_valid
    world_mask_i32 = None
    if row_valid is not None and state._device.type == "mps":
      world_mask_i32 = state._reset_mask_i32
      world_mask_i32.copy_(row_valid)
      state.update_masked_rows(rhs, velocity["qfrc_bias"], row_valid,
                               add=False, sign=-1)
      state.update_masked_rows(
          rhs_low, velocity["dynamics"]["qfrc_bias_low"], row_valid,
          add=False, sign=-1)
      state.update_masked_rows(rhs, velocity["qfrc_passive"], row_valid)
      if (self._passive is None and self.profile.passive_damping_enabled
          and rhs.numel()):
        state.add_masked_scaled_rows(
            rhs, velocity["qvel"], self._damping, row_valid, sign=-1)
    else:
      torch.neg(velocity["qfrc_bias"], out=rhs)
      torch.neg(velocity["dynamics"]["qfrc_bias_low"], out=rhs_low)
      rhs.add_(velocity["qfrc_passive"])
      if (self._passive is None and self.profile.passive_damping_enabled
          and rhs.numel()):
        rhs.addcmul_(velocity["qvel"], self._damping.unsqueeze(0), value=-1.0)
    force = self._applied_force if qfrc_applied is None else qfrc_applied
    if force is not None:
      if world_mask_i32 is None:
        rhs.add_(force)
      else:
        state.update_masked_rows(rhs, force, row_valid)
    if world_mask_i32 is None:
      rhs.add_(actuation["qfrc_actuator"])
    else:
      state.update_masked_rows(rhs, actuation["qfrc_actuator"], row_valid)
    awake_lists = record.values[ForwardStage.POS]["awake_lists"]
    if self._component_mass_enabled:
      qacc, status = self._solve_component_rhs(
          dynamics, rhs, awake_lists, rhs_low=rhs_low,
          world_mask=world_mask_i32)
      qacc_low = self._component_solution_low_vector
    else:
      retained_qacc = state._qacc if awake_lists is not None else None
      qacc, qacc_low, status = self._solver.run_pair_device(
          dynamics["mass_matrix"], rhs, rhs_low, awake_lists=awake_lists,
          retained=retained_qacc,
          retained_low=(self._qacc_low_for_sensor(retained_qacc)
                        if retained_qacc is not None else None),
          world_mask=world_mask_i32)
    if velocity["status"] is not None:
      status = torch.where(velocity["status"] == 0, status, velocity["status"])
    exact_status = record.values[ForwardStage.POS].get(
        "exact_impedance_status")
    if exact_status is not None:
      status = torch.where(exact_status == 0, status, exact_status)
    status = self._forward_stages.invalid_row_status(status, record)
    _zero_invalid_rows(qacc, record.row_valid, self._state)
    _zero_invalid_rows(qacc_low, record.row_valid, self._state)
    _zero_invalid_rows(rhs, record.row_valid, self._state)
    _zero_invalid_rows(rhs_low, record.row_valid, self._state)
    result = {"qfrc_smooth": rhs, "qacc_smooth": qacc,
              "qacc_low": qacc_low, "status": status}
    self._forward_stages.publish(record, ForwardStage.ACC, result)
    return result

  def prepare_forward_constraint(self, record, *, skipsensor=False):
    """Apply the captured constraint stage to an existing ACC result."""
    from mujoco_metal.forward_stages import ForwardStage
    record = self.validate_forward_stage_record(record, ForwardStage.ACC)
    state = self._state
    acceleration = self._forward_stages.consume(
        record, ForwardStage.ACC, generation=state.generation)
    position = self._forward_stages.consume(
        record, ForwardStage.POS, generation=state.generation)
    velocity = self._forward_stages.consume(
        record, ForwardStage.VEL, generation=state.generation)
    context = position["context"]
    world_mask_i32 = None
    if record.row_valid is not None and state._device.type == "mps":
      world_mask_i32 = state._reset_mask_i32
      world_mask_i32.copy_(record.row_valid)
    if self._coupled_constraints is None:
      result = self._run_legacy_constraint_stage(
          record, position, velocity, acceleration)
    else:
      coupled = self._coupled_constraints.run_velocity_device(
          context["coupled"], velocity["dynamics"]["poses"],
          (None if self._component_mass_enabled
           else velocity["dynamics"]["mass_matrix"]),
          acceleration["qfrc_smooth"], record.qpos, velocity["qvel"],
          eq_active=getattr(state, "_eq_active", None),
          cvel=velocity["dynamics"]["cvel"],
          tendon_J_spatial=(self._spatial_kin["jacobian"]
                            if self._spatial_tendons is not None else None),
          tendon_length_spatial=(self._spatial_kin["length"]
                                 if self._spatial_tendons is not None else None),
          flex=self._flex, qacc_warmstart=state._qacc_warmstart,
          awake_tree=getattr(getattr(self, "_sleep_schedule", None),
                             "tree_awake", None),
          component_solver=(self._component_solver
                            if self._component_mass_enabled else None),
          mass_blocks=(velocity["dynamics"].get("mass_blocks")
                       if self._component_mass_enabled else None),
          tendon_armature_blocks=(velocity["dynamics"].get(
              "tendon_armature_blocks") if self._component_mass_enabled else None),
          awake_dof_ids=(position["awake_lists"]["dof_ids"]
                         if self._component_mass_enabled else None),
          awake_counts=(position["awake_lists"]["counts"]
                        if self._component_mass_enabled else None),
          cdof=velocity["dynamics"]["cdof"],
          cdof_dot=velocity["dynamics"]["cdof_dot"],
          qacc_smooth=acceleration["qacc_smooth"],
          qacc_smooth_low=acceleration.get("qacc_low"),
          world_mask=world_mask_i32,
          exact_impedance=self._constraint_impedance,
          exact_dense_solver=self._exact_impedance_dense_solver)
      status = state._torch.where(
          acceleration["status"] == 0, coupled["status"],
          acceleration["status"])
      result = dict(coupled)
      result["status"] = status
      # Publish the row owner's coherent canonical view alongside the
      # optimizer outputs. Queries must consume this prepared record, rather
      # than reconstructing rows or reading an unrelated later workspace.
      result["canonical_rows"] = self._coupled_constraints._assembly_views(
          include_optimizer_outputs=False)
      result["canonical_descriptor"] = self._coupled_constraints.descriptor
    result["status"] = self._forward_stages.invalid_row_status(
        result.get("status"), record)
    _zero_invalid_rows(result.get("qacc"), record.row_valid, self._state)
    _zero_invalid_rows(result.get("qfrc_constraint"), record.row_valid,
                       self._state)
    if (not skipsensor and self._sensors is not None
        and self._sensordata is not None):
      self._store_step_sensors(
          record.qpos, velocity["qvel"], velocity["dynamics"],
          qacc=result["qacc"], stages=(mujoco.mjtStage.mjSTAGE_ACC,),
          qacc_low=result.get("qacc_low"),
          position_context=context, act_override=getattr(state, "_act", None),
          compute_worlds=record.row_valid)
    self._forward_stages.publish(record, ForwardStage.CONSTRAINT, result)
    return result

  def _legacy_rows_view(self):
    """Return packed legacy scalar rows in the shared inverse-cost ABI."""
    if self._legacy_canonical_rows is None:
      return None, None
    from types import SimpleNamespace
    from mujoco_metal.forward_stages import ForwardStage
    packed = self._legacy_canonical_rows
    b, nr, width = packed.shape
    nv = int(self._mjmodel.nv)
    rows = {
        "J": packed[..., :nv], "R": packed[..., nv],
        "ar": packed[..., nv + 1], "lo": packed[..., nv + 2],
        "hi": packed[..., nv + 3], "active": packed[..., nv + 4],
    }
    joint = self._joint_constraints
    descriptor = SimpleNamespace(
        nr=int(nr), nv=nv,
        n_eq_rows=(int(joint.descriptor.neq) if joint is not None else 0),
        nr_joint=int(joint.descriptor.nrow) if joint is not None else 0,
        ncontacts_max=0,
        cone_type=int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
        contact_condim_packed=np.empty((0, 3), dtype=np.int32),
    )
    return rows, descriptor

  def _publish_legacy_rows(self, contact_rows=None, joint_rows=None, *,
                           world_mask=None):
    """Copy assembled legacy family rows into the canonical joint-first view."""
    if self._legacy_canonical_rows is None:
      return None, None
    joint_count = (int(self._joint_constraints.descriptor.nrow)
                   if self._joint_constraints is not None else 0)
    contact_count = (int(self._contact.descriptor.pair_count) * 5
                     if self._contact is not None else 0)
    if world_mask is not None and self._state._device.type == "mps":
      width = int(self._mjmodel.nv) + 5
      target = self._legacy_canonical_rows
      self._state.clear_masked_rows(target, world_mask)
      if joint_count:
        if not isinstance(joint_rows, dict):
          raise RuntimeError("legacy joint rows were not published by their builder")
        source = self._joint_constraints._outputs["canonical_rows"][:
            self.batch_size * joint_count * width].reshape(
                self.batch_size, joint_count, width)
        self._state.copy_masked_packed_rows(
            target, source, world_mask, row_offset=0)
      if contact_count:
        if not isinstance(contact_rows, dict):
          raise RuntimeError("legacy contact rows were not published by their detector")
        source = self._contact._workspace["canonical_rows"][:
            self.batch_size * contact_count * width].reshape(
                self.batch_size, contact_count, width)
        self._state.copy_masked_packed_rows(
            target, source, world_mask, row_offset=joint_count)
      self._state.clear_masked_rows(target, world_mask, invert=True)
      return self._legacy_rows_view()
    fields = ("J", "R", "ar", "lo", "hi", "active")
    if joint_count:
      if not isinstance(joint_rows, dict):
        raise RuntimeError("legacy joint rows were not published by their builder")
      # The packed row backing is `[J[nv], R, ar, lo, hi, active]`.
      self._legacy_canonical_rows[:, :joint_count, :int(
          self._mjmodel.nv)].copy_(joint_rows["J"])
      for offset, name in enumerate(fields[1:], start=int(self._mjmodel.nv)):
        self._legacy_canonical_rows[:, :joint_count, offset].copy_(
            joint_rows[name])
    if contact_count:
      if not isinstance(contact_rows, dict):
        raise RuntimeError("legacy contact rows were not published by their detector")
      self._legacy_canonical_rows[:, joint_count:joint_count+contact_count, :int(
          self._mjmodel.nv)].copy_(contact_rows["J"])
      for offset, name in enumerate(fields[1:], start=int(self._mjmodel.nv)):
        self._legacy_canonical_rows[:, joint_count:joint_count+contact_count,
                                    offset].copy_(contact_rows[name])
    return self._legacy_rows_view()

  def _assemble_legacy_inverse_rows(self, record, velocity):
    """Run only legacy row builders, never their forward optimizers."""
    if self._legacy_canonical_rows is None:
      return None, None
    dynamics = velocity["dynamics"]
    joint = self._joint_constraints
    joint_count = int(joint.descriptor.nrow) if joint is not None else 0
    contact_count = (int(self._contact.descriptor.pair_count) * 5
                     if self._contact is not None else 0)
    if joint is not None:
      joint.run_device(
          dynamics["mass_matrix"], self._forward_stage_zero_constraint,
          record.qpos, velocity["qvel"],
          eq_active=getattr(self._state, "_eq_active", None),
          assemble_only=True)
      source = joint._outputs["canonical_rows"].reshape(
          self.batch_size, max(joint_count, 1), int(self._mjmodel.nv) + 5)
      self._legacy_canonical_rows[:, :joint_count].copy_(
          source[:, :joint_count])
    if self._contact is not None:
      detected = self._contact.run_device(
          dynamics["poses"], None, None, velocity["qvel"],
          assemble_only=True)
      source = self._contact._workspace["canonical_rows"].reshape(
          self.batch_size, max(contact_count, 1), int(self._mjmodel.nv) + 5)
      self._legacy_canonical_rows[:, joint_count:].copy_(
          source[:, :contact_count])
    return self._legacy_rows_view()

  def _run_legacy_constraint_stage(self, record, position, velocity,
                                   acceleration):
    """Preserve legacy forward solvers while publishing their actual rows."""
    state, torch = self._state, self._state._torch
    dynamics, qvel = velocity["dynamics"], velocity["qvel"]
    qacc = acceleration["qacc_smooth"]
    status = acceleration["status"]
    qfrc_constraint = self._forward_stage_zero_constraint
    row_valid = record.row_valid
    world_mask_i32 = None
    if row_valid is None or state._device.type != "mps":
      qfrc_constraint.zero_()
    else:
      world_mask_i32 = state._reset_mask_i32
      world_mask_i32.copy_(row_valid)
      state.clear_masked_rows(qfrc_constraint, row_valid)
    contact_rows = None
    joint_rows = None
    if self._contact is not None:
      contact = self._contact.run_device(
          dynamics["poses"], dynamics["mass_matrix"], qacc, qvel,
          world_mask=world_mask_i32)
      if world_mask_i32 is None:
        qacc = contact["qacc"]
        qfrc_constraint.add_(contact["qfrc_contact"])
      else:
        state.update_masked_rows(qacc, contact["qacc"], row_valid, add=False)
        state.update_masked_rows(
            qfrc_constraint, contact["qfrc_contact"], row_valid)
      status = torch.where(status == 0, contact["status"], status)
      contact_rows = contact["canonical_rows"]
    if self._joint_constraints is not None:
      rhs = self._legacy_constraint_rhs
      if world_mask_i32 is None:
        rhs.copy_(acceleration["qfrc_smooth"])
        rhs.add_(qfrc_constraint)
      else:
        state.update_masked_rows(
            rhs, acceleration["qfrc_smooth"], row_valid, add=False)
        state.update_masked_rows(rhs, qfrc_constraint, row_valid)
      joint = self._joint_constraints.run_device(
          dynamics["mass_matrix"], rhs, record.qpos, qvel,
          eq_active=getattr(state, "_eq_active", None),
          qacc_warmstart=state._qacc_warmstart,
          world_mask=world_mask_i32)
      if world_mask_i32 is None:
        qacc = joint["qacc"]
        qfrc_constraint.add_(joint["qfrc_constraint"])
      else:
        state.update_masked_rows(qacc, joint["qacc"], row_valid, add=False)
        state.update_masked_rows(
            qfrc_constraint, joint["qfrc_constraint"], row_valid)
      status = torch.where(status == 0, joint["status"], status)
      joint_rows = joint["canonical_rows_view"]
    rows, descriptor = self._publish_legacy_rows(
        contact_rows, joint_rows, world_mask=row_valid)
    return {"qacc": qacc, "qfrc_constraint": qfrc_constraint,
            "status": status, "canonical_rows": rows,
            "canonical_descriptor": descriptor,
            "contact_result": contact if self._contact is not None else None}

  def forward_skip(self, skipstage=mujoco.mjtStage.mjSTAGE_NONE,
                   skipsensor=False, *, record=None, qpos=None, qvel=None,
                   mocap_pos=None, mocap_quat=None, ctrl=None):
    """Execute the pinned ``mj_forwardSkip`` stage sequence.

    ``skipstage`` controls which cached stages are consumed: NONE computes a
    fresh POS and VEL prefix, POS reuses POS, and VEL/ACC reuse both POS and
    VEL.  Every call recomputes ACT, acceleration, and constraints.  Cached
    stages are explicit generation-scoped records; this method never guesses
    that arbitrary workspace contents are valid.

    Results are borrowed device views in a dictionary.  POS/VEL sensors run
    only when their stage is executed, and ACC sensors run after constraints,
    matching the pinned forward ordering.
    """
    from mujoco_metal.forward_stages import ForwardStage

    if isinstance(skipstage, (bool, np.bool_)):
      raise TypeError("skipstage must be a MuJoCo stage integer")
    if isinstance(skipstage, mujoco.mjtStage):
      skipstage = int(skipstage)
    elif isinstance(skipstage, numbers.Integral):
      skipstage = int(skipstage)
    else:
      raise TypeError("skipstage must be a MuJoCo stage integer")
    allowed = {
        int(mujoco.mjtStage.mjSTAGE_NONE),
        int(mujoco.mjtStage.mjSTAGE_POS),
        int(mujoco.mjtStage.mjSTAGE_VEL),
        int(mujoco.mjtStage.mjSTAGE_ACC),
    }
    if skipstage not in allowed:
      raise ValueError(f"invalid forward skipstage {skipstage}")
    if not isinstance(skipsensor, (bool, np.bool_)):
      raise TypeError("skipsensor must be boolean")

    needs_position = skipstage == int(mujoco.mjtStage.mjSTAGE_NONE)
    needs_velocity = skipstage in (
        int(mujoco.mjtStage.mjSTAGE_NONE),
        int(mujoco.mjtStage.mjSTAGE_POS),
    )
    if needs_position:
      record = self.prepare_forward_position(
          qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat,
          skipsensor=bool(skipsensor))
    else:
      if record is None:
        record = self._forward_stages.current(
            generation=self._state.generation,
            minimum=(ForwardStage.POS if needs_velocity else ForwardStage.VEL))
      record = self.validate_forward_stage_record(
          record, ForwardStage.POS,
          qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)

    if needs_velocity:
      velocity = self.prepare_forward_velocity(
          record, qvel=qvel, skipsensor=bool(skipsensor))
    else:
      velocity = self._forward_stages.consume(
          record, ForwardStage.VEL, generation=self._state.generation)
      if qvel is not None and qvel is not velocity["qvel"]:
        raise ValueError("qvel does not match the cached velocity-stage input")

    actuation = self.prepare_forward_actuation(record, ctrl=ctrl)
    acceleration = self.prepare_forward_acceleration(record)
    constraint = self.prepare_forward_constraint(
        record, skipsensor=bool(skipsensor))
    position = self._forward_stages.consume(
        record, ForwardStage.POS, generation=self._state.generation)
    energy_buffer = getattr(self, "_energy", None)
    energy = energy_buffer.clone() if energy_buffer is not None else None
    return {
        "record": record,
        "position": position,
        "velocity": velocity,
        "actuation": actuation,
        "acceleration": acceleration,
        "constraint": constraint,
        "qacc": constraint["qacc"],
        "qfrc_constraint": constraint["qfrc_constraint"],
        "status": constraint["status"],
        "energy": energy,
    }

  def inverse_skip(self, skipstage=mujoco.mjtStage.mjSTAGE_NONE,
                   skipsensor=False, *, record=None, qpos=None, qvel=None,
                   qacc=None, mocap_pos=None, mocap_quat=None,
                   return_components=False, sensor_qacc_low=None):
    """Run pinned inverse dynamics using an optional prepared POS/VEL prefix.

    The returned ``qfrc_inverse`` is owned by this call. A supplied stage
    record must belong to this simulation and state generation; POS/VEL are
    evaluated only when the selected skip stage says they are missing.
    Constraint force is reduced from the cached canonical rows without a
    forward optimizer pass.
    """
    from mujoco_metal.forward_stages import ForwardStage

    if isinstance(skipstage, (bool, np.bool_)):
      raise TypeError("skipstage must be a MuJoCo stage integer")
    if isinstance(skipstage, mujoco.mjtStage):
      skipstage = int(skipstage)
    elif isinstance(skipstage, numbers.Integral):
      skipstage = int(skipstage)
    else:
      raise TypeError("skipstage must be a MuJoCo stage integer")
    allowed = {
        int(mujoco.mjtStage.mjSTAGE_NONE),
        int(mujoco.mjtStage.mjSTAGE_POS),
        int(mujoco.mjtStage.mjSTAGE_VEL),
        int(mujoco.mjtStage.mjSTAGE_ACC),
    }
    if skipstage not in allowed:
      raise ValueError(f"invalid inverse skipstage {skipstage}")
    if not isinstance(skipsensor, (bool, np.bool_)):
      raise TypeError("skipsensor must be boolean")
    if not isinstance(return_components, (bool, np.bool_)):
      raise TypeError("return_components must be boolean")
    state = self._state
    torch = state._torch
    qacc_from_state = qacc is None
    qacc = state._qacc if qacc_from_state else qacc
    expected = (self.batch_size, int(self._mjmodel.nv))
    if (not isinstance(qacc, torch.Tensor) or tuple(qacc.shape) != expected
        or qacc.dtype != torch.float32 or not _device_matches(qacc.device, state._device)
        or not qacc.is_contiguous()):
      raise ValueError(f"qacc must be contiguous float32 MPS {expected}")
    # The requested acceleration can be the simulation-owned output of a
    # legacy constraint program.  Assembly-only row capture reuses that
    # program's output backing, so retain the query input before any stage
    # producer is allowed to write to its scratch.
    if sensor_qacc_low is not None and (
        not isinstance(sensor_qacc_low, torch.Tensor)
        or tuple(sensor_qacc_low.shape) != expected
        or sensor_qacc_low.dtype != torch.float32
        or not _device_matches(sensor_qacc_low.device, state._device)
        or not sensor_qacc_low.is_contiguous()):
      raise ValueError(f"sensor_qacc_low must be contiguous float32 MPS {expected}")
    # Preserve the solver residual only when qacc is the state-owned result.
    # Stage assembly can reuse its borrowed backing, so clone before producers.
    if sensor_qacc_low is None and qacc_from_state:
      sensor_qacc_low = self._qacc_low_for_sensor(qacc)
    if sensor_qacc_low is not None:
      sensor_qacc_low = sensor_qacc_low.clone()
    qacc_used = qacc.clone()
    needs_position = skipstage == int(mujoco.mjtStage.mjSTAGE_NONE)
    needs_velocity = skipstage in (
        int(mujoco.mjtStage.mjSTAGE_NONE),
        int(mujoco.mjtStage.mjSTAGE_POS),
    )
    if needs_position:
      record = self.prepare_forward_position(
          qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat,
          skipsensor=bool(skipsensor))
    else:
      if record is None:
        record = self._forward_stages.current(
            generation=self._state.generation,
            minimum=(ForwardStage.POS if needs_velocity else ForwardStage.VEL))
      self.validate_forward_stage_record(
          record, ForwardStage.POS, qpos=qpos,
          mocap_pos=mocap_pos, mocap_quat=mocap_quat)
    if needs_velocity:
      velocity = self.prepare_forward_velocity(
          record, qvel=qvel, skipsensor=bool(skipsensor))
    else:
      velocity = self._forward_stages.consume(
          record, ForwardStage.VEL, generation=self._state.generation)
      if qvel is not None and qvel is not velocity["qvel"]:
        raise ValueError("qvel does not match the cached velocity-stage input")

    dynamics = velocity["dynamics"]
    discrete_status = None
    qacc_used_low = sensor_qacc_low
    if int(self._mjmodel.opt.enableflags) & int(
        mujoco.mjtEnableBit.mjENBL_INVDISCRETE):
      if qacc_used_low is None:
        qacc_used_low = torch.zeros_like(qacc_used)
      discrete_result = (
          self._inverse_discrete_with_preserved_actuator_scratch(
              record, velocity, qacc_used, qacc_used_low))
      if len(discrete_result) == 2:
        # Compatibility with test doubles and older internal extensions.
        qacc_used, discrete_status = discrete_result
      else:
        qacc_used, qacc_used_low, discrete_status = discrete_result
    mass_input_low = (qacc_used_low if qacc_used_low is not None
                      else torch.zeros_like(qacc_used))
    awake = record.values[ForwardStage.POS].get("awake_lists")
    mass_qacc, mass_qacc_low = self._inverse_discrete_mass_pair_product(
        dynamics, qacc_used, mass_input_low, awake_lists=awake)
    body_wrench_force = None
    if self._passive is not None and self._body_wrench.numel():
      # Forward VEL folds xfrc_applied into its passive-force workspace for
      # the stepping RHS. Inverse dynamics treats body wrenches as applied
      # forces, so add their generalized projection back. Preserve the
      # borrowed forward force output while the passive projector is reused.
      saved_passive = self._passive._force.clone()
      try:
        with_wrench = self._passive.run_device(
            record.qpos, velocity["qvel"], xfrc_applied=self._body_wrench,
            mocap_pos=record.mocap_pos, mocap_quat=record.mocap_quat,
            awake_lists=record.values[ForwardStage.POS].get("awake_lists"),
            poses=dynamics["poses"]).clone()
        without_wrench = self._passive.run_device(
            record.qpos, velocity["qvel"],
            mocap_pos=record.mocap_pos, mocap_quat=record.mocap_quat,
            awake_lists=record.values[ForwardStage.POS].get("awake_lists"),
            poses=dynamics["poses"]).clone()
        body_wrench_force = with_wrench - without_wrench
      finally:
        self._passive._force.copy_(saved_passive)
    inverse_constraint = torch.zeros_like(mass_qacc)
    inverse_constraint_low = torch.zeros_like(mass_qacc)
    if self._coupled_constraints is not None:
      from mujoco_metal.native_api import _inverse_constraint_force_from_cached
      inverse_constraint, inverse_constraint_low = (
          _inverse_constraint_force_from_cached(
          self, self._coupled_constraints, qacc_used, record,
          **({"qacc_low": qacc_used_low} if qacc_used_low is not None else {}),
          return_low=True))
    elif self._contact is not None or self._joint_constraints is not None:
      rows, legacy_descriptor = self._assemble_legacy_inverse_rows(
          record, velocity)
      from mujoco_metal.inverse_constraints import inverse_constraint_force
      inverse_constraint, inverse_constraint_low = inverse_constraint_force(
          rows, qacc_used, legacy_descriptor,
          **({"qacc_low": qacc_used_low} if qacc_used_low is not None else {}),
          return_low=True)
    # Match mj_inverseSkip's accumulation association: it first forms
    # (Ma - qfrc_passive - qfrc_constraint), then adds that term to the
    # RNE/tendon bias force.  Keeping the large bias out of the subtraction
    # avoids dropping a small but representable Ma/passive delta in float32.
    passive_force = velocity["qfrc_passive"]
    if body_wrench_force is not None:
      # VEL includes applied body-wrench projection in its passive workspace;
      # inverse dynamics' passive stage does not. Remove that projection
      # before applying the source accumulation order.
      passive_force = passive_force - body_wrench_force
    from mujoco_metal.inverse_constraints import _pair_add
    inverse_terms, inverse_terms_low = _pair_add(
        mass_qacc, mass_qacc_low, -passive_force,
        torch.zeros_like(passive_force))
    inverse_terms, inverse_terms_low = _pair_add(
        inverse_terms, inverse_terms_low, -inverse_constraint,
        -inverse_constraint_low)
    qfrc_inverse, qfrc_inverse_low = _pair_add(
        velocity["qfrc_bias"], dynamics["qfrc_bias_low"],
        inverse_terms, inverse_terms_low)
    if discrete_status is not None:
      velocity_status = velocity.get("status")
      status = (discrete_status if velocity_status is None else
                torch.where(velocity_status == 0, discrete_status,
                            velocity_status))
    else:
      status = velocity["status"]
    result = {"qfrc_inverse": qfrc_inverse,
              "qfrc_inverse_low": qfrc_inverse_low,
              "qfrc_constraint_inverse": inverse_constraint,
              "qfrc_constraint_inverse_low": inverse_constraint_low,
              "qfrc_body_wrench": body_wrench_force, "qacc": qacc,
              "energy": (self._energy.clone() if self._energy is not None else None),
              "status": status, "record": record}
    if return_components:
      result["qfrc_inverse_components"] = {
          "mass_qacc": mass_qacc.clone(),
          "mass_qacc_low": mass_qacc_low.clone(),
          "bias": velocity["qfrc_bias"].clone(),
          "bias_low": dynamics["qfrc_bias_low"].clone(),
          "passive": passive_force.clone(),
          "constraint": inverse_constraint.clone(),
          "constraint_low": inverse_constraint_low,
      }
    sensor_disabled = bool(int(getattr(self._mjmodel.opt, "disableflags", 0)) & int(
        mujoco.mjtDisableBit.mjDSBL_SENSOR))
    if (not skipsensor and not sensor_disabled and self._sensors is not None
        and self._sensordata is not None):
      self._store_step_sensors(
          record.qpos, velocity["qvel"], dynamics, qacc=qacc_used,
          qacc_low=qacc_used_low,
          stages=(mujoco.mjtStage.mjSTAGE_ACC,),
          position_context=record.values[ForwardStage.POS]["context"],
          act_override=getattr(state, "_act", None))
    return result


  def step1(self, *, control_callback=None):
    """Run the pinned POS/VEL prefix and leave ACT/ACC for :meth:`step2`.

    The optional callback is an explicit host opt-in. It receives a read-only
    stage-state view and the coherent POS/VEL record after sensors and energy
    have been updated; returning a control array installs it for step2. With
    no callback, callers may set controls between step1 and step2 directly.
    """
    if control_callback is not None and not callable(control_callback):
      raise TypeError("control_callback must be callable or None")
    if getattr(self, "_step1_record", None) is not None:
      self.validate_forward_stage_record(self._step1_record, "VEL")
      raise RuntimeError("step1 is already pending; call step2 before another step1")
    from mujoco_metal.state_checks import mj_checkPos, mj_checkVel
    mj_checkPos(self)
    mj_checkVel(self)
    record = self.prepare_forward_position()
    scheduler = getattr(self, "_sleep_schedule", None)
    qvel = (self._sleep_mask_velocity(self._state._qvel)
            if scheduler is not None else self._state._qvel)
    self.prepare_forward_velocity(record, qvel=qvel)
    if control_callback is not None:
      stage_state = _plugin_stage_state(
          self._state, record.qpos, qvel, time=self._state._time)
      control = control_callback(stage_state, record)
      if control is not None:
        self._prepare_control(control)
    self._step1_record = record
    return record

  def step2(self, *, ctrl=None, qfrc_applied=None, xfrc_applied=None):
    """Complete a prepared step1 with ACT/ACC/constraint and integration.

    Controls and applied forces may be changed at this boundary, as in
    ``mj_step1``/``mj_step2``. Xfrc changes refresh VEL from the retained POS
    context before force/constraint stages are republished.
    """
    from mujoco_metal.forward_stages import ForwardStage
    record = getattr(self, "_step1_record", None)
    if record is None:
      raise ValueError("step2 requires a successful step1 prefix")
    self.validate_forward_stage_record(record, ForwardStage.VEL)
    try:
      if xfrc_applied is not None:
        self._prepare_wrench(xfrc_applied)
        qvel = record.values[ForwardStage.VEL]["qvel"]
        self.prepare_forward_velocity(record, qvel=qvel)
      return self.step(1, qfrc_applied=qfrc_applied, ctrl=ctrl)
    finally:
      self._step1_record = None

  def _inverse_discrete_with_preserved_actuator_scratch(
      self, record, velocity, qacc, qacc_low=None):
    """Run the derivative query without transferring actuator workspace ownership."""
    preserve_actuator = int(self._mjmodel.opt.integrator) in (
        int(mujoco.mjtIntegrator.mjINT_IMPLICIT),
        int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST))
    scratch = (self._snapshot_inverse_actuator_scratch()
               if preserve_actuator else None)
    try:
      if qacc_low is None:
        return self._inverse_discrete_acceleration(record, velocity, qacc)
      return self._inverse_discrete_acceleration(
          record, velocity, qacc, qacc_low=qacc_low)
    finally:
      if scratch is not None:
        self._restore_inverse_actuator_scratch(scratch)


  def _snapshot_inverse_actuator_scratch(self):
    """Copy derivative-query outputs that must remain owned by forward state."""
    saved = {}
    if self._actuators is not None:
      saved["actuator_ws"] = _clone_system_dict(self._actuators._ws)
      for name in ("_last_actuation_kin", "_actuator_velocity_derivative"):
        saved[name] = getattr(self, name, None)
    for program, names in (
        (self._passive, ("_force", "_damping_derivative", "_wrench",
                         "_gravcomp_output", "_gravcomp_zero_wrench")),
        (self, ("_act_dot", "_act_vel", "_sen_act_force", "_sen_qfrc_act")),
    ):
      if program is None:
        continue
      target = saved.setdefault("tensors", [])
      for name in names:
        value = getattr(program, name, None)
        if value is not None and hasattr(value, "clone"):
          target.append((program, name, value.clone()))
    return saved

  def _restore_inverse_actuator_scratch(self, saved):
    """Restore borrowed actuator/passive workspaces even after query failure."""
    if self._actuators is not None and "actuator_ws" in saved:
      for name, value in saved["actuator_ws"].items():
        current = self._actuators._ws.get(name)
        if current is not None:
          current.copy_(value)
      for name in ("_last_actuation_kin", "_actuator_velocity_derivative"):
        if name in saved:
          setattr(self, name, saved[name])
    for program, name, value in saved.get("tensors", ()):
      current = getattr(program, name, None)
      if current is not None and tuple(current.shape) == tuple(value.shape):
        current.copy_(value)

  def compare_forward_inverse(self, record, forward_result=None):
    """Return the two pinned ``mj_compareFwdInv`` residuals for one record.

    The caller supplies the current forward result; inverse evaluation reuses
    its POS/VEL row cache and does not run a second forward optimizer. With no
    active constraint rows, MuJoCo defines both residuals as zero.
    """
    from mujoco_metal.forward_stages import ForwardStage
    record = self.validate_forward_stage_record(record, ForwardStage.CONSTRAINT)
    if forward_result is None:
      forward_result = {
          "record": record,
          "constraint": record.values[ForwardStage.CONSTRAINT],
          "actuation": record.values[ForwardStage.ACT],
      }
    if (not isinstance(forward_result, dict)
        or forward_result.get("record") is not record
        or forward_result.get("constraint") is not
        record.values.get(ForwardStage.CONSTRAINT)
        or forward_result.get("actuation") is not
        record.values.get(ForwardStage.ACT)):
      raise ValueError(
          "forward_result must reference this record's published ACT and CONSTRAINT stages")
    state, torch = self._state, self._state._torch
    constraint = forward_result["constraint"]
    out = torch.zeros((self.batch_size, 2), dtype=torch.float32,
                      device=state._device)
    has_rows = (
        int(self._coupled_constraints.descriptor.nr) > 0
        if self._coupled_constraints is not None
        else int(getattr(self, "_legacy_row_count", 0)) > 0)
    if not has_rows:
      return out
    from mujoco_metal.native_api import _inverse_query_workspaces
    with _inverse_query_workspaces(self):
      inverse = self.inverse_skip(
          mujoco.mjtStage.mjSTAGE_VEL, skipsensor=True, record=record,
          qacc=constraint["qacc"])
      # Inverse/derivative programs borrow the forward solver workspaces.
      # Clone only the products consumed below before the transaction restores
      # those owners in place.
      inverse_constraint = inverse["qfrc_constraint_inverse"].clone()
      inverse_force = inverse["qfrc_inverse"].clone()
      inverse_wrench = (inverse["qfrc_body_wrench"].clone()
                        if inverse.get("qfrc_body_wrench") is not None else None)
    if self._coupled_constraints is not None:
      active = self._coupled_constraints._assembly_views(
          include_optimizer_outputs=False)["active"]
    else:
      rows = constraint.get("canonical_rows")
      active = (rows["active"] if rows is not None else
                torch.zeros((self.batch_size, 0), dtype=torch.float32,
                            device=state._device))
    has_constraint = torch.any(active > 0.5, dim=1)
    forward_constraint = constraint["qfrc_constraint"]
    qforce = torch.zeros_like(inverse_force)
    if self._applied_force is not None:
      qforce.add_(self._applied_force)
    qforce.add_(forward_result["actuation"]["qfrc_actuator"])
    if inverse_wrench is not None:
      qforce.add_(inverse_wrench)
    first = torch.linalg.vector_norm(forward_constraint - inverse_constraint, dim=1)
    second = torch.linalg.vector_norm(qforce - inverse_force, dim=1)
    out[:, 0] = torch.where(has_constraint, first, torch.zeros_like(first))
    out[:, 1] = torch.where(has_constraint, second, torch.zeros_like(second))
    return out

  def _inverse_discrete_mass_pair_solve(self, dynamics, rhs, rhs_low,
                                        awake_lists=None, retained=None):
    """Solve a two-word mass RHS and return a normalized solution pair."""
    torch = self._state._torch
    if self._component_mass_enabled:
      high, status = self._solve_component_rhs(
          dynamics, rhs, awake_lists, rhs_low=rhs_low)
      high = high.clone()
      low = self._component_solution_low_vector.clone()
      return high, low, status
    mass = dynamics["mass_matrix"]
    from mujoco_metal.inverse_constraints import (
        _dense_jacobian_pair_matvec, _pair_add)

    def solve_pair(rhs_hi, rhs_lo, retained_value=None):
      first, status_hi = self._solver.run_device(
          mass, rhs_hi, awake_lists=awake_lists,
          retained=(retained_value if awake_lists is not None else None))
      first = first.clone()
      correction, status_lo = self._solver.run_device(
          mass, rhs_lo, awake_lists=awake_lists)
      correction = correction.clone()
      out_hi, out_low = _pair_add(
          first, torch.zeros_like(first), correction,
          torch.zeros_like(correction))
      return out_hi, out_low, torch.maximum(status_hi, status_lo)

    high, low, status = solve_pair(rhs, rhs_low, retained)
    # The dense solver publishes one float per solve. Refine its factorization
    # residual once in the same operator so the returned low word represents
    # both RHS low bits and the first solve's rounding residual.
    correction_status = status
    for _ in range(2):
      product_hi, product_low = _dense_jacobian_pair_matvec(mass, high)
      low_product_hi, low_product_low = _dense_jacobian_pair_matvec(mass, low)
      product_hi, product_low = _pair_add(
          product_hi, product_low, low_product_hi, low_product_low)
      residual_hi, residual_low = _pair_add(
          rhs, rhs_low, -product_hi, -product_low)
      delta_hi, delta_low, current_status = solve_pair(
          residual_hi, residual_low)
      high, low = _pair_add(high, low, delta_hi, delta_low)
      correction_status = torch.maximum(correction_status, current_status)
    return high, low, torch.maximum(status, correction_status)


  def _inverse_discrete_mass_pair_product(self, dynamics, vector,
                                          vector_low=None, *, awake_lists=None):
    """Apply dense or compiled component mass blocks to a vector pair."""
    torch = self._state._torch
    if vector_low is None:
      vector_low = torch.zeros_like(vector)
    if not self._component_mass_enabled:
      from mujoco_metal.inverse_constraints import (
          _dense_jacobian_pair_matvec, _pair_add)
      high, low = _dense_jacobian_pair_matvec(
          dynamics["mass_matrix"], vector)
      low_hi, low_lo = _dense_jacobian_pair_matvec(
          dynamics["mass_matrix"], vector_low)
      return _pair_add(high, low, low_hi, low_lo)

    if self._component_solver is None:
      raise RuntimeError("component mass pair requires a compiled solver")
    blocks = dynamics["mass_blocks"]
    armature = dynamics.get("tendon_armature_blocks")
    if armature is None:
      armature = torch.zeros_like(blocks)
    dof_ids = awake_lists.get("dof_ids") if awake_lists is not None else None
    counts = awake_lists.get("counts") if awake_lists is not None else None
    return self._component_solver.run_mass_matvec_pair_device(
        blocks, vector, vector_low, dof_ids=dof_ids, counts=counts,
        tendon_armature_blocks=armature,
        diagonal_add=dynamics.get("mass_diagonal_add"))


  def _inverse_discrete_acceleration(self, record, velocity, qacc,
                                     qacc_low=None):
    """Convert discrete inverse acceleration to continuous acceleration.

    This follows ``engine_inverse.c:mj_discreteAcc``. Euler adds the exact
    compiled damper derivative to ``M*qacc`` before solving the original mass
    operator. Implicit profiles form their pinned velocity derivative operator
    through the same prepared analytical backends used by stepping.
    """
    torch, model = self._state._torch, self._mjmodel
    integrator = int(model.opt.integrator)
    if qacc_low is None:
      qacc_low = torch.zeros_like(qacc)
    elif (not isinstance(qacc_low, torch.Tensor)
          or tuple(qacc_low.shape) != tuple(qacc.shape)
          or qacc_low.dtype != torch.float32
          or qacc_low.device != qacc.device or not qacc_low.is_contiguous()):
      raise ValueError("inverse discrete qacc_low must match qacc")
    if integrator == int(mujoco.mjtIntegrator.mjINT_RK4):
      raise ValueError(
          "discrete inverse dynamics is unsupported by the RK4 integrator")
    dynamics, qvel = velocity["dynamics"], velocity["qvel"]
    from mujoco_metal.forward_stages import ForwardStage
    awake = record.values[ForwardStage.POS].get("awake_lists")
    if integrator == int(mujoco.mjtIntegrator.mjINT_EULER):
      if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP):
        return qacc, qacc_low, torch.zeros((self.batch_size,), dtype=torch.int32,
                                           device=self._state._device)
      eligible = (np.asarray(model.dof_damping) > 0) | np.any(
          np.asarray(model.dof_dampingpoly) != 0, axis=1)
      if model.nv:
        eligible |= np.asarray(model.jnt_actuatorid)[model.dof_jntid] != -1
      if not np.any(eligible):
        return qacc, qacc_low, torch.zeros((self.batch_size,), dtype=torch.int32,
                                           device=self._state._device)
      if self._passive is None:
        raise RuntimeError("inverse discrete Euler requires lowered damper constants")
      damp = self._passive._damp[:model.nv].reshape(1, model.nv)
      dpoly = self._passive._dpoly[:2 * model.nv].reshape(1, model.nv, 2)
      speed = torch.abs(qvel)
      derivative = (damp + 2.0 * dpoly[:, :, 0] * speed
                    + 3.0 * dpoly[:, :, 1] * speed.square())
      from mujoco_metal.inverse_constraints import (
          _dense_jacobian_pair_matvec, _pair_add, _two_prod)
      mass_hi, mass_low = self._inverse_discrete_mass_pair_product(
          dynamics, qacc, qacc_low, awake_lists=awake)
      dt_hi = torch.full_like(derivative, float(model.opt.timestep))
      coefficient_hi, coefficient_low = _two_prod(dt_hi, derivative)
      damp_hi, damp_low = _two_prod(coefficient_hi, qacc)
      term_hi, term_low = _two_prod(coefficient_low, qacc)
      damp_hi, damp_low = _pair_add(damp_hi, damp_low, term_hi, term_low)
      term_hi, term_low = _two_prod(coefficient_hi, qacc_low)
      damp_hi, damp_low = _pair_add(damp_hi, damp_low, term_hi, term_low)
      term_hi, term_low = _two_prod(coefficient_low, qacc_low)
      damp_hi, damp_low = _pair_add(damp_hi, damp_low, term_hi, term_low)
      rhs, rhs_low = _pair_add(mass_hi, mass_low, damp_hi, damp_low)
      solution, solution_low, status = self._inverse_discrete_mass_pair_solve(
          dynamics, rhs, rhs_low, awake, retained=qacc)
      return solution, solution_low, status

    if integrator not in (int(mujoco.mjtIntegrator.mjINT_IMPLICIT),
                          int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)):
      raise ValueError(f"unsupported inverse discrete integrator {integrator}")
    if self._component_mass_enabled:
      if (self._effective_implicit is None
          or self._velocity_derivative_values is None):
        raise RuntimeError(
            "sparse discrete inverse requires its compiled derivative operator")
      writer = self._velocity_derivative_values
      writer.clear_device()
      if self._passive is not None and velocity.get("passive_damping") is not None:
        writer.add_passive_diagonal_device(velocity["passive_damping"])
      if self._tendons is not None:
        self._tendons.run_damping_derivative_coo_device(qvel, writer)
      if self._spatial_tendons is not None and self._spatial_kin is not None:
        self._spatial_tendons.run_damping_derivative_coo_device(
            self._spatial_kin, writer)
      if self._fluid is not None:
        self._fluid.run_derivative_device(
            record.qpos, qvel, dynamics, self._smooth, edge_writer=writer)
      if self._actuators is not None:
        self._actuation_force(
            record.qpos, qvel, dynamics["poses"],
            position_context=record.values[ForwardStage.POS]["context"].get(
                "actuators"))
      if integrator == int(mujoco.mjtIntegrator.mjINT_IMPLICIT):
        self._smooth.bias_derivative_device(
            record.qpos, qvel, dynamics, edge_values=writer.values)
      if self._flex is not None:
        self._flex.run_edge_velocity_derivative_coo_device(
            dict(dynamics["poses"], root_com=dynamics["root_com"]), writer,
            dynamics.get("cvel"))
      dof_ids = (awake["dof_ids"] if awake is not None else None)
      counts = (awake["counts"] if awake is not None else None)
      operator_values = (writer.symmetrize_lower_device()
                         if integrator == int(
                             mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
                         else writer.values)
      effective_rhs, low_rhs, effective_status = (
          self._effective_implicit.apply_effective_operator_pair_device(
              dynamics["mass_blocks"], qacc, qacc_low, operator_values,
              float(model.opt.timestep), dof_ids=dof_ids, counts=counts,
              tendon_armature_blocks=dynamics.get("tendon_armature_blocks")))
      effective_rhs = effective_rhs.clone()
      low_rhs = low_rhs.clone()
      effective_status = effective_status.clone()
      solution, solution_low, solve_status = (
          self._inverse_discrete_mass_pair_solve(
              dynamics, effective_rhs, low_rhs, awake))
      status = torch.where(effective_status == 0, solve_status,
                           effective_status)
      return solution, solution_low, status
    passive_diag = velocity.get("passive_damping")
    tendon_tangent = velocity.get("tendon_damping")
    if velocity.get("spatial_damping") is not None:
      spatial = velocity["spatial_damping"]
      tendon_tangent = spatial if tendon_tangent is None else tendon_tangent + spatial
    fluid_jacobian = (self._fluid.run_derivative_device(
        record.qpos, qvel, dynamics, self._smooth)
        if self._fluid is not None else None)
    actuator_jacobian = None
    if self._actuators is not None:
      self._actuation_force(
          record.qpos, qvel, dynamics["poses"],
          position_context=record.values[ForwardStage.POS]["context"].get(
              "actuators"))
      actuator_jacobian = getattr(self, "_actuator_velocity_derivative", None)
    flex_edge_derivative = None
    if self._flex is not None:
      flex_ops = self._flex.run_material_operators(
          record.qpos, qvel,
          dict(dynamics["poses"], root_com=dynamics["root_com"]),
          dynamics.get("cvel"))
      flex_edge_derivative = flex_ops.get("edge_velocity_derivative")
    if integrator == int(mujoco.mjtIntegrator.mjINT_IMPLICIT):
      backend = self._implicit
      bias_derivative = self._smooth.bias_derivative_device(
          record.qpos, qvel, dynamics)
      effective = backend.run_device_auto(
          dynamics["mass_matrix"], qacc, passive_diag,
          tendon_tangent=tendon_tangent, fluid_jacobian=fluid_jacobian,
          bias_derivative=bias_derivative,
          actuator_jacobian=actuator_jacobian,
          flex_edge_derivative=flex_edge_derivative)
    else:
      backend = self._implicitfast
      effective = backend.run_device_auto(
          dynamics["mass_matrix"], qacc, passive_diag,
          tendon_tangent=tendon_tangent, fluid_jacobian=fluid_jacobian,
          actuator_jacobian=actuator_jacobian,
          flex_edge_derivative=flex_edge_derivative)
    from mujoco_metal.inverse_constraints import (
        _dense_jacobian_pair_matvec, _pair_add)
    rhs, rhs_low = _dense_jacobian_pair_matvec(
        effective["effective_mass"], qacc)
    low_hi, low_lo = _dense_jacobian_pair_matvec(
        effective["effective_mass"], qacc_low)
    rhs, rhs_low = _pair_add(rhs, rhs_low, low_hi, low_lo)
    solution, solution_low, status = self._inverse_discrete_mass_pair_solve(
        dynamics, rhs, rhs_low)
    status = torch.where(effective["status"] == 0, status, effective["status"])
    return solution, solution_low, status


  def _validate_forward_position(self, context):
    """Check stage ownership before any scheduler, force or solver writes."""
    if (not isinstance(context, dict) or context.get("_owner") is not self
        or context.get("_epoch") != getattr(self, "_forward_position_epoch", None)
        or context.get("_generation") != self._state.generation):
      raise ValueError("forward position context is foreign, stale or restored")
    expected = {"smooth"}
    for program, name in ((self._tendons, "fixed_tendon_length"),
                          (self._spatial_tendons, "spatial_tendons"),
                          (self._actuators, "actuators"),
                          (self._transmissions, "transmissions"),
                          (self._coupled_constraints, "coupled"),
                          (self._flex, "flex")):
      if program is not None:
        expected.add(name)
    if not expected <= context.keys():
      raise ValueError("forward position context is missing a required stage")

  def _acceleration(self, qpos, qvel, act_override=None, time_override=None, constrained=True,
                   dynamics_override=None, poses_override=None,
                   applied_force_override=None, awake_lists_override=None,
                   skip_sleep_prepare=False, forward_mask=None,
                   awake_solve_tree=None, position_context=None,
                   capture_fwdinv=False):
    if forward_mask is None:
      self._last_sensor_qacc = None
      self._last_sensor_qacc_low = None
      self._last_sensor_qacc_version = None
      self._last_sensor_qacc_generation = None
      self._last_recovery_qacc_low = None
    else:
      self._last_recovery_qacc_low = None
    if position_context is not None:
      self._validate_forward_position(position_context)
      if (not skip_sleep_prepare or dynamics_override is not None
          or poses_override is not None):
        raise ValueError("cached position refresh must skip position preparation and overrides")
    # Position/velocity cache reuse is resolved row-wise in
    # _spatial_jacobian; do not globally invalidate healthy rows merely
    # because this invocation carries a recovery mask.
    torch = self._state._torch
    presolve_poses = (None if skip_sleep_prepare else
                     self._prepare_sleep_schedule(qpos, qvel))
    if getattr(self, "_sleep_schedule", None) is not None and not skip_sleep_prepare:
      qvel = self._sleep_mask_velocity(qvel)
    mpos = getattr(self._state, "_mpos", None)
    mquat = getattr(self._state, "_mquat", None)
    stage_poses = poses_override if poses_override is not None else presolve_poses
    scheduler = getattr(self, "_sleep_schedule", None)
    world_mask_i32 = None
    if forward_mask is not None:
      # Reuse the fixed int32 mask backing also consumed by the reset kernel.
      # Each producer predicates its per-world work in MSL; no host decision
      # is made from the mask values.
      world_mask_i32 = self._state._reset_mask_i32
      world_mask_i32.copy_(forward_mask)
    awake_lists = (awake_lists_override if awake_lists_override is not None else
                   scheduler.awake_lists() if scheduler is not None else None)
    if self._component_mass_enabled and awake_lists is None:
      awake_lists = self._smooth._workspace["all_awake_lists"]
    if (self._component_mass_enabled and position_context is None
        and stage_poses is None):
      stage_poses = self._smooth._fk.run_device(
          qpos, mpos, mquat,
          tree_awake=(scheduler.tree_awake if scheduler is not None else None),
          world_mask=world_mask_i32)
    if position_context is not None:
      dynamics = self._smooth.run_velocity_device(
          position_context["smooth"], qvel, awake_lists=awake_lists,
          world_mask=world_mask_i32)
    elif dynamics_override is not None:
      dynamics = dict(dynamics_override)
    else:
      tendon_J = self._component_tendon_jacobian(qvel, stage_poses,
                                                 world_mask=forward_mask)
      dynamics = self._smooth.run_device(
          qpos, qvel, mpos if stage_poses is None else None,
          mquat if stage_poses is None else None, poses=stage_poses,
          awake_lists=awake_lists, tendon_J=tendon_J,
          _trusted_internal_poses=True, world_mask=world_mask_i32)
    if dynamics_override is not None:
      dynamics["mass_matrix"] = dynamics["mass_matrix"].clone()
    if poses_override is not None:
      dynamics["poses"] = poses_override
    eval_time = time_override if time_override is not None else self._state._time
    if self._sensors is not None and self._has_acc_sensors:
      self._lazy_sensor_scratch()
      if forward_mask is None:
        self._sen_act_force.zero_()
        self._sen_qfrc_act.zero_()
      else:
        self._state.clear_masked_rows(self._sen_act_force, forward_mask)
        self._state.clear_masked_rows(self._sen_qfrc_act, forward_mask)
    if capture_fwdinv:
      if forward_mask is None:
        self._forward_stage_passive_force.zero_()
        self._forward_stage_actuator_force.zero_()
      else:
        self._state.clear_masked_rows(
            self._forward_stage_passive_force, forward_mask)
        self._state.clear_masked_rows(
            self._forward_stage_actuator_force, forward_mask)
    def add_rhs(value, sign=1):
      if world_mask_i32 is None:
        if sign < 0:
          self._rhs.sub_(value)
        else:
          self._rhs.add_(value)
      else:
        self._state.update_masked_rows(
            self._rhs, value, forward_mask,
            add=True, sign=-1 if sign < 0 else 1)

    if world_mask_i32 is None:
      self._state._torch.neg(dynamics["qfrc_bias"], out=self._rhs)
      self._state._torch.neg(dynamics["qfrc_bias_low"], out=self._rhs_low)
    else:
      self._state.update_masked_rows(
          self._rhs, dynamics["qfrc_bias"], forward_mask,
          add=False, sign=-1)
      self._state.update_masked_rows(
          self._rhs_low, dynamics["qfrc_bias_low"], forward_mask,
          add=False, sign=-1)
    applied_force = (self._applied_force if applied_force_override is None
                     else applied_force_override)
    if applied_force is not None:
      if (
          self._passive is None
          and self.profile.passive_damping_enabled
          and self._rhs.numel()
      ):
        if forward_mask is None:
          self._rhs.addcmul_(qvel, self._damping.unsqueeze(0), value=-1.0)
        else:
          damped = -qvel * self._damping.unsqueeze(0)
          self._state.update_masked_rows(self._rhs, damped, forward_mask)
      add_rhs(applied_force)
    if self._passive is not None:
      passive, self._damping_tangent = self._passive.run_device(
          qpos, qvel, xfrc_applied=self._body_wrench, return_damping=True,
          mocap_pos=mpos, mocap_quat=mquat, awake_lists=awake_lists,
          world_mask=world_mask_i32,
          poses=dynamics["poses"],
      )
      add_rhs(passive)
      if capture_fwdinv:
        self._forward_stage_passive_force.add_(passive)
    if self._flex is not None:
      flex_poses = dict(dynamics["poses"], root_com=dynamics["root_com"])
      if position_context is None:
        flex_qfrc, _, _ = self._flex.run_device(
            qpos, qvel, flex_poses, dynamics.get("cvel", None),
            world_mask=world_mask_i32)
      else:
        flex_qfrc, _, _ = self._flex.run_velocity_device(
            position_context["flex"], qpos, qvel, flex_poses,
            dynamics.get("cvel", None), world_mask=world_mask_i32)
      add_rhs(flex_qfrc)
      if capture_fwdinv:
        if forward_mask is None:
          self._forward_stage_passive_force.add_(flex_qfrc)
        else:
          self._state.update_masked_rows(
              self._forward_stage_passive_force, flex_qfrc, forward_mask)
      # Euler's qH contains joint/actuator DOF damping only. Material damping
      # is already in flex_qfrc; implicit profiles use the separate full edge
      # qDeriv and stiffness correction instead of a material diagonal.
    if self._fluid is not None:
      fluid_force = self._fluid.run_device(
          qpos, qvel, dynamics, world_mask=world_mask_i32)
      add_rhs(fluid_force)
      if capture_fwdinv:
        self._forward_stage_passive_force.add_(fluid_force)
    if self._tendons is not None:
      tendon_force, self._tendon_damping, tendon_armature = (
          self._tendons.run_device(
              qpos, qvel, length_override=(position_context["fixed_tendon_length"]
                                           if position_context is not None else None),
              world_mask=world_mask_i32)
      )
      add_rhs(tendon_force)
      if capture_fwdinv:
        self._forward_stage_passive_force.add_(tendon_force)
      if position_context is None and not self._component_mass_enabled:
        if forward_mask is None:
          dynamics["mass_matrix"].add_(tendon_armature)
        else:
          self._state.update_masked_rows(
              dynamics["mass_matrix"], tendon_armature, forward_mask)
    if self._spatial_tendons is not None:
      if position_context is None:
        self._spatial_jacobian(qvel, dynamics["poses"],
                               world_mask=forward_mask)
      else:
        spatial_key = (self._state.generation, id(qvel),
                       id(dynamics["poses"].get("body_pos")))
        spatial_cache_valid = getattr(self, "_spatial_cache_key", None) == spatial_key
        self._spatial_kin = self._spatial_tendons.run_velocity_kinematics(
            position_context["spatial_tendons"], qvel,
            world_mask=world_mask_i32)
        self._spatial_cache_key = (
            spatial_key if world_mask_i32 is None or spatial_cache_valid
            else None)
      skin = self._spatial_kin
      sforce, sdamp, sarm = self._spatial_tendons.run_forces(
          skin, include_armature=(position_context is None
                                  and not self._component_mass_enabled),
          world_mask=world_mask_i32)
      add_rhs(sforce.reshape(self._rhs.shape))
      if capture_fwdinv:
        self._forward_stage_passive_force.add_(sforce.reshape(self._rhs.shape))
      sbias, _ = self._spatial_tendons.run_armature_bias(
          skin, qvel, dynamics["poses"], dynamics.get("cvel", None),
          dynamics.get("root_com", None), dynamics.get("cdof", None),
          dynamics.get("cdof_dot", None), world_mask=world_mask_i32)
      # Armature bias is a bias force (pinned mj_tendonBias accumulates into
      # qfrc_bias), so it subtracts from rhs like Coriolis/gravity.
      add_rhs(sbias.reshape(self._rhs.shape), sign=-1)
      if position_context is None and not self._component_mass_enabled:
        if forward_mask is None:
          dynamics["mass_matrix"].add_(sarm.reshape(dynamics["mass_matrix"].shape))
        else:
          self._state.update_masked_rows(
              dynamics["mass_matrix"],
              sarm.reshape(dynamics["mass_matrix"].shape), forward_mask)
    if self._cable is not None:
      self._cable.run_device(
          qpos, dynamics["poses"], dynamics["cdof"], self._rhs,
          capture_target=(self._forward_stage_passive_force
                          if capture_fwdinv else None),
          world_mask=world_mask_i32)
    if self._transmissions is not None:
      t_qfrc = self._transmissions.run_device(
          qpos, qvel, self._delayed_control(eval_time),
          awake_lists=awake_lists,
          position_context=(position_context["transmissions"]
                            if position_context is not None else None),
          world_mask=world_mask_i32,
          gravcomp=self._transmission_gravcomp(
              qpos, dynamics["poses"], world_mask_i32, awake_lists))
      add_rhs(t_qfrc)
      if capture_fwdinv:
        self._forward_stage_actuator_force.add_(t_qfrc)
      if self._sensors is not None and self._has_acc_sensors:
        self._lazy_sensor_scratch()
        tforce = self._transmissions._last_force.reshape(self._sen_act_force.shape)
        if forward_mask is None:
          self._sen_act_force.add_(tforce)
          self._sen_qfrc_act.add_(t_qfrc.reshape(self._sen_qfrc_act.shape))
        else:
          self._state.update_masked_rows(self._sen_act_force, tforce,
                                         forward_mask)
          self._state.update_masked_rows(
              self._sen_qfrc_act, t_qfrc.reshape(self._sen_qfrc_act.shape),
              forward_mask)
    if self._actuators is not None:
      actuator_force = self._actuation_force(
          qpos, qvel, dynamics["poses"], act_override=act_override,
          time_override=eval_time,
          world_mask=world_mask_i32,
          row_mask=forward_mask,
          position_context=(position_context["actuators"]
                            if position_context is not None else None))
      add_rhs(actuator_force)
      if capture_fwdinv:
        self._forward_stage_actuator_force.add_(actuator_force)
    if self._motor is not None:
      m_qfrc = self._motor.run_device(
          self._delayed_control(eval_time), world_mask=world_mask_i32)
      add_rhs(m_qfrc)
      if capture_fwdinv:
        self._forward_stage_actuator_force.add_(m_qfrc)
      if self._sensors is not None and self._has_acc_sensors:
        self._lazy_sensor_scratch()
        if self._motor._last_force is not None:
          mforce = self._motor._last_force.reshape(self._sen_act_force.shape)
          if forward_mask is None:
            self._sen_act_force.add_(mforce)
          else:
            self._state.update_masked_rows(self._sen_act_force, mforce,
                                           forward_mask)
        if forward_mask is None:
          self._sen_qfrc_act.add_(m_qfrc.reshape(self._sen_qfrc_act.shape))
        else:
          self._state.update_masked_rows(
              self._sen_qfrc_act, m_qfrc.reshape(self._sen_qfrc_act.shape),
              forward_mask)
    force_plugins = tuple(getattr(self, "_force_plugins", ()))
    actuator_plugins = tuple(getattr(self, "_legacy_actuator_plugins", ()))
    typed_actuator_plugins = tuple(getattr(self, "_native_actuator_plugins", ()))
    all_force_plugins = force_plugins + actuator_plugins + typed_actuator_plugins
    before_plugins = _snapshot_plugins_device(all_force_plugins)
    plugin_state = _plugin_stage_state(
        self._state, qpos, qvel, act=act_override, time=eval_time)
    passive_plugin_target = (self._forward_stage_passive_force
                             if capture_fwdinv else self._rhs)
    actuator_plugin_target = (self._forward_stage_actuator_force
                              if capture_fwdinv else self._rhs)
    try:
      self._accumulate_plugin_forces(
          force_plugins, plugin_state, qpos, qvel, dynamics,
          passive_plugin_target,
          rhs_target=self._rhs if capture_fwdinv else None,
          compute_mask=forward_mask)
      typed_output = self._run_native_actuator_plugins(
          plugin_state, self._delayed_control(eval_time), dynamics, qvel,
          act_override, force_target=actuator_plugin_target,
          compute_mask=forward_mask) if typed_actuator_plugins else None
      if capture_fwdinv and typed_output is not None:
        add_rhs(typed_output["qfrc"])
      self._accumulate_plugin_forces(
          actuator_plugins, plugin_state, qpos, qvel, dynamics,
          actuator_plugin_target,
          rhs_target=self._rhs if capture_fwdinv else None,
          compute_mask=forward_mask)
      if forward_mask is not None:
        _restore_failed_plugin_rows(before_plugins, forward_mask)
    except Exception:
      _restore_plugins_device(before_plugins)
      raise
    if self._component_mass_enabled:
      acceleration, status = self._solve_component_rhs(
          dynamics, self._rhs, awake_lists, rhs_low=self._rhs_low,
          world_mask=world_mask_i32)
      qacc_low = self._component_solution_low_vector
    else:
      retained_qacc = self._state._qacc if awake_lists is not None else None
      acceleration, qacc_low, status = self._solver.run_pair_device(
          dynamics["mass_matrix"], self._rhs, self._rhs_low,
          awake_lists=awake_lists,
          retained=retained_qacc,
          retained_low=(self._qacc_low_for_sensor(retained_qacc)
                        if retained_qacc is not None else None),
          world_mask=world_mask_i32
      )
    if capture_fwdinv:
      qacc_smooth_for_compare = acceleration.clone()
      qfrc_smooth_for_compare = self._rhs.clone()
      constraint_force_for_compare = self._forward_stage_zero_constraint
      actuation_for_compare = self._forward_stage_actuator_force
    if "pose_status" in dynamics:
      status = torch.where(dynamics["pose_status"] == 0, status,
                           dynamics["pose_status"])
    coupled = None
    if not constrained:
      if forward_mask is None:
        self._cache_sensor_acceleration(acceleration, qacc_low)
      else:
        self._last_recovery_qacc_low = qacc_low
      return acceleration, status, dynamics
    if self._coupled_constraints is not None:
      self._coupled_solve_dispatches += 1
      eq_active = getattr(self._state, "_eq_active", None)
      _ten_J, _ten_L = self._spatial_for_coupled(
          qvel, dynamics["poses"], world_mask=forward_mask)
      coupled_solver = self._coupled_constraints.run_device
      coupled_args = ()
      if position_context is not None:
        coupled_solver = self._coupled_constraints.run_velocity_device
        coupled_args = (position_context["coupled"],)
      coupled = coupled_solver(*coupled_args,
          dict(dynamics["poses"], root_com=dynamics["root_com"]),
          (None if self._component_mass_enabled else dynamics["mass_matrix"]),
          self._rhs, qpos, qvel,
          eq_active=eq_active, cvel=dynamics.get("cvel", None),
          cdof=dynamics.get("cdof", None), cdof_dot=dynamics.get("cdof_dot", None),
          tendon_J_spatial=_ten_J, tendon_length_spatial=_ten_L,
          flex=self._flex, qacc_warmstart=self._state._qacc_warmstart,
          awake_tree=(awake_solve_tree if awake_solve_tree is not None else
                      getattr(scheduler, "tree_awake", None)),
          component_solver=(self._component_solver
                            if self._component_mass_enabled else None),
          mass_blocks=(dynamics.get("mass_blocks")
                       if self._component_mass_enabled else None),
          tendon_armature_blocks=(dynamics.get("tendon_armature_blocks")
                                  if self._component_mass_enabled else None),
          awake_dof_ids=(awake_lists["dof_ids"]
                         if self._component_mass_enabled and awake_lists is not None
                         else None),
          awake_counts=(awake_lists["counts"]
                        if self._component_mass_enabled and awake_lists is not None
                        else None),
          exact_impedance=self._constraint_impedance,
          exact_dense_solver=self._exact_impedance_dense_solver,
          qfrc_smooth_low=(self._rhs_low
              if (self._coupled_constraints.descriptor.dense_path
                  or self._coupled_constraints.descriptor.solver_type
                     == int(mujoco.mjtSolver.mjSOL_PGS)) else None),
          world_mask=world_mask_i32,
      )
      coupled["mass_matrix"] = dynamics.get("mass_matrix")
      if self._component_mass_enabled:
        coupled["mass_blocks"] = dynamics["mass_blocks"]
        coupled["mass_block_layout"] = self._smooth.mass_block_layout
        coupled["tendon_armature_blocks"] = dynamics.get(
            "tendon_armature_blocks")
      if forward_mask is None:
        self._last_coupled = coupled
        self._last_coupled_generation = self._state.generation
        self._last_coupled_qacc_version = int(coupled["qacc"]._version)
      status = self._state._torch.where(
          status == 0, coupled["status"], status
      )
      if world_mask_i32 is None:
        acceleration = coupled["qacc"]
      else:
        self._state.update_masked_rows(
            acceleration, coupled["qacc"], forward_mask,
            add=False, sign=1)
      if capture_fwdinv:
        constraint_force_for_compare = coupled["qfrc_constraint"]
      add_rhs(coupled["qfrc_constraint"])
    legacy_contact_rows = None
    legacy_joint_rows = None
    if self._contact is not None:
      contact = self._contact.run_device(
          dynamics["poses"], dynamics["mass_matrix"], acceleration, qvel,
          world_mask=world_mask_i32
      )
      torch = self._state._torch
      status = torch.where(status == 0, contact["status"], status)
      if world_mask_i32 is None:
        acceleration = contact["qacc"]
      else:
        self._state.update_masked_rows(
            acceleration, contact["qacc"], forward_mask,
            add=False, sign=1)
      if capture_fwdinv:
        constraint_force_for_compare = contact["qfrc_contact"]
      add_rhs(contact["qfrc_contact"])
      legacy_contact_rows = contact["canonical_rows"]
    if self._joint_constraints is not None:
      constrained = self._joint_constraints.run_device(
          dynamics["mass_matrix"], self._rhs, qpos, qvel,
          qacc_warmstart=self._state._qacc_warmstart,
          world_mask=world_mask_i32
      )
      status = self._state._torch.where(
          status == 0, constrained["status"], status
      )
      if world_mask_i32 is None:
        acceleration = constrained["qacc"]
      else:
        self._state.update_masked_rows(
            acceleration, constrained["qacc"], forward_mask,
            add=False, sign=1)
      if capture_fwdinv:
        constraint_force_for_compare = constrained["qfrc_constraint"]
      add_rhs(constrained["qfrc_constraint"])
      legacy_joint_rows = constrained["canonical_rows_view"]
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is not None and int(self._mjmodel.nv) > 0:
      solver_awake = (scheduler.tree_awake if awake_solve_tree is None
                      else awake_solve_tree)
      awake = solver_awake.index_select(1, self._sleep_dof_treeid) != 0
      if forward_mask is None:
        acceleration = torch.where(awake, acceleration, 0.0)
      else:
        asleep_zero = torch.where(awake, acceleration, 0.0)
        self._state.update_masked_rows(
            acceleration, asleep_zero, forward_mask, add=False)
    if capture_fwdinv:
      from mujoco_metal.forward_stages import ForwardStage
      context = (position_context if position_context is not None else
                 self._capture_forward_position(qpos, dynamics))
      record = self._forward_stages.begin(
          generation=self._state.generation, qpos=qpos,
          mocap_pos=mpos, mocap_quat=mquat,
          position={"dynamics": dynamics, "poses": dynamics["poses"],
                    "context": context, "awake_lists": awake_lists})
      self._forward_stages.publish(record, ForwardStage.VEL, {
          "dynamics": dynamics, "qfrc_bias": dynamics["qfrc_bias"],
          "qfrc_passive": self._forward_stage_passive_force,
          "qvel": qvel, "position_context": context, "status": status})
      self._forward_stages.publish(record, ForwardStage.ACT, {
          "qfrc_actuator": actuation_for_compare})
      self._forward_stages.publish(record, ForwardStage.ACC, {
          "qfrc_smooth": qfrc_smooth_for_compare,
          "qacc_smooth": qacc_smooth_for_compare, "status": status})
      legacy_rows = (self._publish_legacy_rows(
          legacy_contact_rows, legacy_joint_rows,
          world_mask=(forward_mask if world_mask_i32 is not None else None))[0]
          if self._legacy_canonical_rows is not None else None)
      constraint = {"qacc": acceleration,
                    "qfrc_constraint": constraint_force_for_compare,
                    "status": status,
                    "canonical_rows": legacy_rows}
      self._forward_stages.publish(record, ForwardStage.CONSTRAINT, constraint)
      forward_result = {
          "record": record,
          "constraint": record.values[ForwardStage.CONSTRAINT],
          "actuation": record.values[ForwardStage.ACT],
      }
      self._solver_fwdinv.copy_(self.compare_forward_inverse(
          record, forward_result))
    low = coupled.get("qacc_low") if isinstance(coupled, dict) else None
    if forward_mask is None:
      self._cache_sensor_acceleration(acceleration, low)
    else:
      self._last_recovery_qacc_low = low
    return acceleration, status, dynamics

  def _solve_sparse_effective_acceleration(self, qpos, qvel, dynamics):
    """Assemble compiled qDeriv sources and solve sparse implicit dynamics.

    Each native producer adds directly into MuJoCo's compiled D COO support.
    Rigid mass and fixed/spatial tendon armature stay in component blocks;
    this path does not construct an nv-by-nv mass or derivative matrix.
    """
    if self._effective_implicit is None or self._velocity_derivative_values is None:
      raise RuntimeError("sparse effective-implicit workspace is not prepared")
    writer = self._velocity_derivative_values
    self._accumulate_sparse_velocity_derivative(qpos, qvel, dynamics, writer)
    if self._sparse_flex_implicit is not None:
      # `writer.values` aliases the effective solver's mutable COO backing.
      # Preserve full nonsymmetric D before implicitfast projects qH and before
      # the first GMRES/apply dispatch reuses that backing.
      self._sparse_flex_implicit.capture_full_derivative_values(writer.values)
    scheduler = getattr(self, "_sleep_schedule", None)
    awake = scheduler.awake_lists() if scheduler is not None else None
    operator_values = (writer.symmetrize_lower_device()
                       if int(self._mjmodel.opt.integrator) ==
                       int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
                       else writer.values)
    solution, status, residual, iterations = self._effective_implicit.solve_device(
        dynamics["mass_blocks"], self._rhs, operator_values,
        float(self.profile.timestep),
        dof_ids=(awake["dof_ids"] if awake is not None else None),
        counts=(awake["counts"] if awake is not None else None),
        tendon_armature_blocks=dynamics.get("tendon_armature_blocks"))
    return {"qacc": solution, "status": status,
            "residual": residual, "iterations": iterations}

  def _accumulate_sparse_velocity_derivative(self, qpos, qvel, dynamics,
                                             writer):
    """Add every admitted source to the prepared compiled-D COO values."""
    values = writer.values
    if self._passive is not None and self._damping_tangent is not None:
      writer.add_passive_diagonal_device(self._damping_tangent)
    if self._tendons is not None:
      self._tendons.run_damping_derivative_coo_device(qvel, writer)
    if self._spatial_tendons is not None and self._spatial_kin is not None:
      self._spatial_tendons.run_damping_derivative_coo_device(
          self._spatial_kin, writer)
    if self._fluid is not None:
      self._fluid.run_derivative_device(
          qpos, qvel, dynamics, self._smooth, edge_writer=writer)
    # Full implicit differentiates the RNE bias at held position. Implicitfast
    # intentionally omits this Coriolis/bias derivative.
    if self.profile.name == "integrated_implicit_v1":
      self._smooth.bias_derivative_device(
          qpos, qvel, dynamics, edge_values=values)
    if self._flex is not None:
      poses = dict(dynamics["poses"], root_com=dynamics["root_com"])
      self._flex.run_edge_velocity_derivative_coo_device(
          poses, writer, dynamics.get("cvel"))

  def _midpoint_eligibility(self):
    from mujoco_metal.implicit_midpoint import midpoint_eligibility
    current = getattr(self, "_last_coupled", None)
    jacobian = (current.get("J") if current is not None
                and getattr(self, "_last_coupled_generation", None) == self._state.generation
                else None)
    # The native sleep scheduler, when installed, owns the authoritative awake
    # mask. A host mirror must never decide a per-world integration path.
    scheduler = getattr(self, "_sleep_schedule", None)
    valid_current = (current if jacobian is not None else {})
    return midpoint_eligibility(
        self._midpoint.descriptor, self.batch_size, torch=self._state._torch,
        device=self._state._device, constraint_jacobian=jacobian,
        tree_awake=getattr(scheduler, "tree_awake", None),
        dof_treeid=self._mjmodel.dof_treeid,
        dof_island=valid_current.get("dof_island"),
        midpoint_blocked_body=valid_current.get("midpoint_blocked_body"),
        midpoint_blocked_tree=valid_current.get("midpoint_blocked_tree"),
        islands_disabled=bool(int(self._mjmodel.opt.disableflags)
                              & int(mujoco.mjtDisableBit.mjDSBL_ISLAND)),
    )

  def _seed_sleep_fk_reference(self, env_ids=None):
    """Seed retained articulated poses from pinned reset data, by world."""
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None or not hasattr(self, "_sleep_reset_body_pos"):
      return
    outputs = self._smooth._fk._workspace["outputs"]
    body_pos = outputs["body_pos"].view(
        self.batch_size, int(self._mjmodel.nbody), 3)
    body_quat = outputs["body_quat"].view(
        self.batch_size, int(self._mjmodel.nbody), 4)
    if env_ids is None:
      body_pos.copy_(self._sleep_reset_body_pos.expand_as(body_pos))
      body_quat.copy_(self._sleep_reset_body_quat.expand_as(body_quat))
      return
    rows = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if rows.size and (np.any(rows < 0) or np.any(rows >= self.batch_size)):
      raise IndexError("sleep FK reference environment index is outside batch")
    for world in rows.tolist():
      body_pos[world].copy_(self._sleep_reset_body_pos)
      body_quat[world].copy_(self._sleep_reset_body_quat)

  def _probe_sleep_pose_mismatch(self, qpos, qvel, *, mocap_pos=None,
                                 mocap_quat=None, pose_status=None):
    """Run pinned kinematics1's retained-pose comparison and cycle wake."""
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None:
      return None
    state = self._state
    if mocap_pos is None:
      mocap_pos = getattr(state, "_mpos", None)
    if mocap_quat is None:
      mocap_quat = getattr(state, "_mquat", None)
    poses = self._smooth._fk.run_device(
        qpos, mocap_pos, mocap_quat, tree_awake=scheduler.tree_awake)
    active_worlds = state._status == 0
    if pose_status is not None:
      active_worlds = active_worlds & (pose_status == 0)
    scheduler.wake_before_solve(
        qvel, qfrc_applied=self._applied_force, xfrc_applied=self._body_wrench,
        active_worlds=active_worlds.contiguous(),
        pose_mismatch=self._smooth._fk.pose_mismatch())
    # mj_kinematics2 and all position consumers run after mj_wake has updated
    # body_awake/tree_awake. Rebuild the borrowed pose families with that mask;
    # no CPU pose or state readback is involved.
    return self._smooth._fk.run_device(
        qpos, mocap_pos, mocap_quat, tree_awake=scheduler.tree_awake)

  def _prepare_sleep_schedule(self, qpos, qvel, *, mocap_pos=None,
                              mocap_quat=None, poses=None, cvel=None,
                              cdof=None, root_com=None, pose_status=None,
                              return_candidate_token=False,
                              pose_mismatch_checked=False):
    """Run the current-position wake pass before force and solver assembly.

    Pinned MuJoCo performs kinematics, collision, and equality wake checks
    before making the awake lists used by smooth dynamics and constraints.
    The native profile mirrors that order with the position-only FK kernel,
    narrowphase slot generation, and the device tree scheduler. Contact links
    are sourced from actual active rows; unsupported links fail open through
    the workspace overflow bit.
    """
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None:
      return None
    state = self._state
    if mocap_pos is None:
      mocap_pos = getattr(state, "_mpos", None)
    if mocap_quat is None:
      mocap_quat = getattr(state, "_mquat", None)
    if poses is None:
      poses = self._probe_sleep_pose_mismatch(
          qpos, qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat,
          pose_status=pose_status)
    elif not pose_mismatch_checked:
      # Supplied position arrays may already have overwritten the retained
      # FK cache, so callers that do work before sleep preparation must run
      # the mismatch probe first and explicitly identify that ordering.
      raise ValueError(
          "sleep scheduling with supplied poses requires a prior retained-pose probe")
    cc = getattr(self, "_coupled_constraints", None)
    needs_flex_links = bool(
        self._flex is not None and cc is not None
        and int(cc.descriptor.n_flex_contact_rows) > 0)
    if needs_flex_links and (cvel is None or cdof is None
                             or "root_com" not in poses):
      # Other callers (for example an explicit acceleration refresh) enter
      # sleep preparation directly. Give their current pose a complete
      # all-awake smooth motion basis before asking the flex detector for
      # source-derived point Jacobians.
      zero_qvel = self._smooth._workspace["qvel"][:
          self.batch_size * self._mjmodel.nv].view(
              self.batch_size, self._mjmodel.nv)
      zero_qvel.zero_()
      tendon_J = self._component_tendon_jacobian(zero_qvel, poses)
      all_awake = self._smooth._workspace["all_awake_lists"]
      prepass = self._smooth.run_position_device(
          qpos, poses=poses, awake_lists=all_awake, tendon_J=tendon_J)
      velocity_inputs = self._smooth._run_velocity_bias(
          qvel, all_awake, cdof=prepass["cdof"],
          local_inertia=self._smooth._workspace["local_inertia"])
      cvel, cdof = velocity_inputs["cvel"], prepass["cdof"]
      root_com = prepass["root_com"]
      if pose_status is None:
        pose_status = prepass.get("pose_status")
    links = None
    overflow = None
    contact_active = None
    candidate_token = None
    if cc is not None:
      flex = self._flex if needs_flex_links else None
      candidate_poses = poses
      if flex is not None:
        candidate_poses = dict(poses)
        candidate_poses["cdof"] = cdof
        candidate_poses["root_com"] = root_com
      candidate_token = cc.generate_candidates(
          candidate_poses, qvel, getattr(state, "_eq_active", None),
          flex=flex, cvel=cvel, cdof=cdof)
      links = cc._workspace.get("active_tree_links")
      overflow = cc._workspace.get("active_tree_link_overflow")
      slot_map = cc._workspace.get("slot_maps")
      if slot_map is not None:
        import torch
        contact_active = torch.any(
            slot_map.logical_to_packed >= 0, dim=1).to(dtype=torch.int32)
    eq_active = getattr(state, "_eq_active", None)
    constraints_active = contact_active
    if eq_active is not None:
      import torch
      eq_rows_active = torch.any(eq_active != 0, dim=1).to(dtype=torch.int32)
      if constraints_active is None:
        constraints_active = eq_rows_active
      else:
        constraints_active = torch.maximum(constraints_active, eq_rows_active)
    active_worlds = state._status == 0
    if pose_status is not None:
      active_worlds = active_worlds & (pose_status == 0)
    scheduler.wake_before_solve(
        qvel, qfrc_applied=self._applied_force, xfrc_applied=self._body_wrench,
        active_tree_pairs=links, constraints_active=constraints_active,
        link_overflow=overflow, equality_active=eq_active,
        active_worlds=active_worlds.contiguous())
    return ((poses, candidate_token) if return_candidate_token else poses)

  def _sleep_mask_velocity(self, qvel):
    """Return device velocity with currently sleeping tree DOFs set to zero."""
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None or int(self._mjmodel.nv) == 0:
      return qvel
    tree = self._sleep_dof_treeid
    awake = scheduler.tree_awake.index_select(1, tree) != 0
    self._state._torch.where(awake, qvel, self._sleep_zero, out=self._sleep_qvel)
    return self._sleep_qvel

  def _snapshot_sleep_device(self):
    """Retain scheduler buffers for per-world failed-step rollback on device."""
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None:
      return None
    names = ("tree_state", "tree_awake", "status", "body_awake_ids",
             "parent_awake_ids", "dof_awake_ids", "body_awake_mask",
             "awake_counts", "links", "constraints_active", "links_overflow",
             "eq_active")
    return {name: getattr(scheduler, name).clone() for name in names}

  def _restore_failed_sleep_rows(self, saved, accepted):
    if saved is None:
      return
    scheduler, torch = self._sleep_schedule, self._state._torch
    for name, previous in saved.items():
      current = getattr(scheduler, name)
      mask = accepted.reshape((self.batch_size,) + (1,) * (current.ndim - 1))
      current.copy_(torch.where(mask, current, previous))

  def _prepare_sleep_integration(self, qvel, acceleration, active_worlds):
    """Apply mj_sleep to pre-integration velocity, then expose awake indices."""
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None:
      return None
    self._sleep_forward_lists = _clone_system_dict(scheduler.awake_lists())
    self._sleep_fk_previous.copy_(scheduler.tree_awake)
    scheduler.advance_before_integration(
        qvel, qfrc_applied=self._applied_force,
        xfrc_applied=self._body_wrench, active_worlds=active_worlds)
    scheduler.zero_asleep_acceleration(acceleration)
    return scheduler.awake_lists()

  def _sleep_forward_buffers(self):
    """Retain stage storage for masked velocity-forward transactions on device.

    The extra forward is logically conditional per world. Kernels may execute
    across the batch, but worlds without a sleep transition must retain their
    finite-iteration warm start, force caches and sensor samples exactly.
    """
    torch = self._state._torch
    seen, tensors = set(), []
    def visit(value):
      key = id(value)
      if key in seen:
        return
      seen.add(key)
      if isinstance(value, torch.Tensor):
        tensors.append((value, value.clone()))
      elif isinstance(value, dict):
        for item in value.values():
          visit(item)
      elif isinstance(value, (tuple, list)):
        for item in value:
          visit(item)
      elif type(value).__module__.startswith("mujoco_metal") and hasattr(value, "__dict__"):
        for item in vars(value).values():
          visit(item)
    for name in ("_smooth", "_passive", "_fluid", "_tendons", "_spatial_tendons",
                 "_transmissions", "_actuators", "_motor", "_solver",
                 "_coupled_constraints", "_flex", "_sensors", "_raw_sensor_program",
                 "_rhs", "_sen_act_force", "_sen_qfrc_act", "_act_dot"):
      visit(getattr(self, name, None))
    return tensors

  def _refresh_sleep_velocity(self, qpos, qvel, acceleration, accepted,
                              position_context=None):
    """Pinned mj_forwardSkip(POS) after mj_sleep, before updating indices.

    The scheduler already rebuilt the next integration lists. This refresh
    deliberately uses the saved pre-sleep body/DOF indices and tree_awake.
    mj_sleep changes tree_asleep and velocity; mj_updateSleep subsequently
    rebuilds tree_awake and body_awake after this forward refresh.
    """
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None:
      return acceleration
    torch = self._state._torch
    changed = ((self._sleep_fk_previous != 0) & (scheduler.tree_awake == 0)).any(dim=1)
    changed &= accepted
    original_acceleration = acceleration.clone()
    original_status = self._combined_status.clone()
    lists = self._sleep_forward_lists
    stored = self._sleep_forward_buffers()
    masked_velocity = self._sleep_mask_velocity(qvel).clone()
    refreshed, status, dynamics = self._acceleration(
        qpos, masked_velocity, act_override=getattr(self, "_next_act", None),
        awake_lists_override=lists, skip_sleep_prepare=True, forward_mask=changed,
        awake_solve_tree=self._sleep_forward_lists["tree_awake"],
        position_context=position_context)
    refreshed = refreshed.clone()
    if self._sensors is not None:
      self._store_step_sensors(
          qpos, masked_velocity, dynamics, qacc=refreshed,
          stages=(mujoco.mjtStage.mjSTAGE_VEL, mujoco.mjtStage.mjSTAGE_ACC),
          compute_worlds=changed, sleep_counts=lists["counts"],
          sleep_tree_awake=self._sleep_forward_lists["tree_awake"],
          act_override=getattr(self, "_next_act", None),
          position_context=position_context)
    for current, previous in stored:
      if self.batch_size and current.numel() % self.batch_size == 0:
        mask = changed[:, None]
        current.copy_(torch.where(mask, current.reshape(self.batch_size, -1),
                                   previous.reshape(self.batch_size, -1)).reshape(current.shape))
      else:
        current.copy_(previous)
    self._combined_status.copy_(torch.where(changed, status, original_status))
    return torch.where(changed[:, None], refreshed, original_acceleration)

  def _refresh_sleep_poses(self):
    """Refresh accepted awake poses without advancing sleep timers again."""
    scheduler = getattr(self, "_sleep_schedule", None)
    if scheduler is None:
      return None
    self._state._torch.maximum(
        self._sleep_fk_previous, scheduler.tree_awake,
        out=self._sleep_fk_refresh)
    return self._smooth._fk.run_device(
        self._state._qpos, getattr(self._state, "_mpos", None),
        getattr(self._state, "_mquat", None), tree_awake=self._sleep_fk_refresh)

  def step(self, steps=1, *, qfrc_applied=None, ctrl=None, xfrc_applied=None):
    """Advance all worlds by a positive number of native profile steps.

    ``qfrc_applied``, ``ctrl`` and ``xfrc_applied`` are optional host arrays or
    contiguous float32 MPS tensors with shapes ``[batch_size, nv]`` and
    the corresponding MuJoCo dimensions. Each supplied input is held constant
    for all steps in this call and remains installed as simulation state;
    omitting one preserves its current value. Pass explicit zeros to clear it.
    Controls require a profile supporting the model
    actuator family. Host arrays are validated and copied before device stepping;
    MPS tensors stay on device and nonfinite rows fail independently.

    The fixed workspace is reused. Failed worlds retain their previous state
    and acceleration while status records the failure. Failure remains sticky
    until reset or restore. Device stepping performs no CPU physics, solve,
    integration or readback.
    """
    self._assert_contact_override_binding()
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral):
      raise TypeError("steps must be a positive integer")
    if steps <= 0:
      raise ValueError("steps must be a positive integer")
    pending_step1 = getattr(self, "_step1_record", None)
    if pending_step1 is not None:
      if int(steps) != 1:
        raise ValueError("a pending step1 record can only be consumed by one step2")
      from mujoco_metal.forward_stages import ForwardStage
      self.validate_forward_stage_record(pending_step1, ForwardStage.VEL)
    self._prepare_wrench(xfrc_applied)
    self._prepare_control(ctrl)
    self._prepare_force(qfrc_applied)
    torch = self._state._torch
    state = self._state
    fwdinv_enabled = bool(
        int(self._mjmodel.opt.enableflags)
        & int(mujoco.mjtEnableBit.mjENBL_FWDINV))
    cc = getattr(self, "_coupled_constraints", None)
    for _ in range(int(steps)):
      if pending_step1 is None:
        from mujoco_metal.state_checks import mj_checkPos, mj_checkVel
        mj_checkPos(self)
        mj_checkVel(self)
      if fwdinv_enabled:
        self._solver_fwdinv.zero_()
      plugin_step_snapshot = _snapshot_plugins_device(self._native_plugins)
      sleep_step_snapshot = self._snapshot_sleep_device()
      pre_warm = None
      pre_sens = (self._sensordata.clone() if self._sensordata is not None
                  else None)
      pre_act_dot = (self._act_dot.clone()
                     if getattr(self, "_act_dot", None) is not None else None)
      try:
        pre_step_gen = state.generation
        pre_step_time = state._time.clone()
        self._sensor_plugin_status.zero_()
        split_step2 = pending_step1 is not None
        if cc is not None and int(cc.descriptor.nr) > 0:
          nr = int(cc.descriptor.nr)
          w_dbg = cc._workspace["workspace_debug"].reshape(self.batch_size, cc._debug_stride)
          pre_warm = w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr].clone()
        if pending_step1 is not None:
          from mujoco_metal.forward_stages import ForwardStage
          self.validate_forward_stage_record(pending_step1, ForwardStage.VEL)
          step_qvel = pending_step1.values[ForwardStage.VEL]["qvel"]
        else:
          self._prepare_sleep_schedule(state._qpos, state._qvel)
          step_qvel = self._sleep_mask_velocity(state._qvel)
        if self._effective_implicit is not None:
          # One current-step derivative transaction. Actuator force assembly
          # below may immediately add its exact velocity derivative; the
          # passive/tendon/fluid/RNE/flex sources add after force assembly.
          self._velocity_derivative_values.clear_device()
        if self._rk4 is not None and pending_step1 is None:
          scheduler = getattr(self, "_sleep_schedule", None)
          if scheduler is not None:
            # RK4 leaves X3 position-stage caches in mjData, restores the
            # public state to X0, then runs mj_sleep and forwardSkip(POS)
            # before integrating the weighted RK rates. Preserve that order
            # with an owned position context; rebuilding at X0 here would
            # change contacts, equality Jdot, actuator moments and tendons.
            na_here = int(getattr(state, "_na", 0))
            fwd_acc, fwd_status, fwd_dyn = self._acceleration(
                state._qpos, step_qvel, capture_fwdinv=fwdinv_enabled)
            self._update_energy_position(state._qpos, fwd_dyn, None)
            self._update_energy_velocity({"dynamics": fwd_dyn,
                                          "qvel": step_qvel})
            fwd_acc, step_qvel, fwd_status, check_reset = (
                self._check_step_acceleration(fwd_acc, step_qvel, fwd_status))
            initial_stage = (fwd_acc.clone(),
                             self._act_dot.clone() if na_here > 0 else None,
                             fwd_status.clone())
            if self._sensors is not None:
              self._store_step_sensors(
                  state._qpos, state._qvel, fwd_dyn, qacc=fwd_acc,
                  qacc_low=self._qacc_low_for_sensor(fwd_acc),
                  compute_worlds=(~check_reset).contiguous())
            rk4_act = (state._act if (self._actuators is not None and na_here > 0)
                       else None)
            callback_index = 0
            final_callback_index = 2 if initial_stage is not None else 3

            def _rk4_sleep_accel(q, v, a, t=None):
              nonlocal callback_index
              acc, solve_status, dyn = self._acceleration(
                  q, v, act_override=a, time_override=t)
              adot = self._act_dot.clone() if na_here > 0 else None
              result = (acc, adot, solve_status)
              if callback_index == final_callback_index:
                result += (self._capture_forward_position(q, dyn),)
              callback_index += 1
              return result

            deferred = self._rk4.run_device(
                state._qpos, step_qvel, rk4_act, state._time, state._status,
                _rk4_sleep_accel, initial_stage=initial_stage,
                defer_integration=True)
            # run_device copied the already masked X0 velocity into its owned
            # reusable buffer before any intermediate callback could overwrite
            # the caller's sleep-mask output.
            rk4_initial_qvel = self._rk4._initial_qvel
            status = deferred["status"]
            status = torch.where(self._sensor_plugin_status == 0, status,
                                 self._sensor_plugin_status)
            success = status == 0
            self._combined_status.copy_(status)
            if na_here > 0 and deferred["act_dot"] is not None:
              self._act_dot.copy_(deferred["act_dot"].reshape(self._act_dot.shape))
              self._success.copy_(success)
              self._advance_activations(state)

            # The sleep state transition uses X0 qvel and the RK-weighted
            # acceleration. Its zeroing applies only to the integration input;
            # state.qacc remains X3's forward acceleration after a sleep
            # transition refresh below.
            integration_acceleration = deferred["weighted_acceleration"].clone()
            integration_awake = self._prepare_sleep_integration(
                rk4_initial_qvel, integration_acceleration, success)
            sleep_status = scheduler.status
            sleep_failure = torch.where(
                (sleep_status == 1) | (sleep_status == 3), sleep_status,
                torch.zeros_like(sleep_status))
            torch.where(self._combined_status == 0, sleep_failure,
                        self._combined_status, out=self._combined_status)
            success = self._combined_status == 0
            acceleration = self._refresh_sleep_velocity(
                state._qpos, rk4_initial_qvel, deferred["final_acceleration"],
                success, position_context=deferred["final_context"])
            qpos, qvel, time, status = self._integrator.run_device(
                state._qpos, rk4_initial_qvel, integration_acceleration,
                state._time, self._combined_status,
                next_velocity=deferred["next_velocity"],
                position_velocity=deferred["position_velocity"],
                awake_lists=integration_awake)
            status = torch.where(self._sensor_plugin_status == 0, status,
                                 self._sensor_plugin_status)
            success = status == 0
            self._advance_native_actuator_plugins(
                success, qpos=qpos, qvel=qvel, qacc=acceleration,
                act=(self._next_act if self._actuators is not None
                     and na_here > 0 else None), time=time)
            state._qpos = torch.where(success[:, None], qpos, state._qpos)
            state._qvel = torch.where(success[:, None], qvel, state._qvel)
            state._qacc = torch.where(success[:, None], acceleration,
                                      state._qacc)
            state._qacc_warmstart.copy_(torch.where(
                success[:, None], state._qacc, state._qacc_warmstart))
            state._time = torch.where(success, time, state._time)
            state._status = status.clone()
            if self._actuators is not None and na_here > 0:
              state._act, self._next_act = self._next_act, state._act
            self._record_delay(pre_step_time, success)
            self._restore_failed_sleep_rows(sleep_step_snapshot, success)
            # mj_advance does not run kinematics after integration. Preserve
            # the final RK forward stage's position cache (X3), even though
            # the public state is restored to X0 before integration. The next
            # kinematics1 pass compares its current qpos with that retained
            # cache and wakes any sleeper whose pose changed.
            state._generation += 1
            if (getattr(self, "_flex", None) is not None
                and getattr(self, "_sleep_schedule", None) is None):
              fk_dyn = self._smooth.run_device(
                  state._qpos, state._qvel, getattr(state, "_mpos", None),
                  getattr(state, "_mquat", None))
              self._flex.update_kinematics(
                  dict(fk_dyn["poses"], root_com=fk_dyn["root_com"]),
                  fk_dyn.get("cvel", None))
            if self._sensordata is not None:
              rollback_sens = (pre_sens if pre_sens is not None
                               else torch.zeros_like(self._sensordata))
              self._sensordata.copy_(torch.where(
                  success.unsqueeze(1), self._sensordata, rollback_sens))
            if pre_warm is not None and cc is not None:
              failed_mask = ~success
              nr = int(cc.descriptor.nr)
              w_dbg = cc._workspace["workspace_debug"].reshape(
                  self.batch_size, cc._debug_stride)
              slice_w = w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr]
              slice_w.copy_(torch.where(failed_mask.unsqueeze(1), pre_warm,
                                        slice_w))
            self._assembled_system_valid = (
                getattr(self, "_last_coupled", None) is not None)
            self._record_accepted_step(
                success, pre_step_gen, pre_step_time, state._status,
                state._qacc)
            _restore_failed_plugin_rows(plugin_step_snapshot, success)
            continue
          na_here = int(getattr(state, "_na", 0))
          initial_stage = None
          # mj_step always performs the outer forward/checkAcc before selecting
          # RK4; feed that exact X0 stage into the four-stage integrator.
          fwd_acc, fwd_status, fwd_dyn = self._acceleration(
              state._qpos, step_qvel, capture_fwdinv=fwdinv_enabled)
          self._update_energy_position(state._qpos, fwd_dyn, None)
          self._update_energy_velocity({"dynamics": fwd_dyn,
                                        "qvel": step_qvel})
          fwd_acc, step_qvel, fwd_status, check_reset = (
              self._check_step_acceleration(fwd_acc, step_qvel, fwd_status))
          initial_stage = (fwd_acc.clone(),
                           self._act_dot.clone() if na_here > 0 else None,
                           fwd_status.clone())
          if self._sensors is not None:
            self._store_step_sensors(
                state._qpos, state._qvel, fwd_dyn, qacc=fwd_acc,
                qacc_low=self._qacc_low_for_sensor(fwd_acc),
                compute_worlds=(~check_reset).contiguous())
          rk4_act = (state._act if (self._actuators is not None and na_here > 0)
                     else None)
          def _rk4_accel(q, v, a, t=None):
            acc, solve_status, _ = self._acceleration(
                q, v, act_override=a, time_override=t)
            adot = self._act_dot.clone() if na_here > 0 else None
            return acc, adot, solve_status
          qpos, qvel, acceleration, weighted_dot, time, status = self._rk4.run_device(
              state._qpos,
              step_qvel,
              rk4_act,
              state._time,
              state._status,
              _rk4_accel,
              initial_stage=initial_stage,
          )
          status = torch.where(self._sensor_plugin_status == 0, status,
                               self._sensor_plugin_status)
          success = status == 0
          if self._actuators is not None and na_here > 0 and weighted_dot is not None:
            # Pinned mj_advance applies mj_nextActivation to the RK-weighted
            # act_dot from the step-start activation; failed worlds keep
            # theirs via the accepted-step mask (R02).
            self._act_dot.copy_(torch.where(
                success.unsqueeze(1), weighted_dot.reshape(self._act_dot.shape),
                self._act_dot))
            self._success.copy_(success)
            self._advance_activations(state)
          self._advance_native_actuator_plugins(
              success, qpos=qpos, qvel=qvel, qacc=acceleration,
              act=(self._next_act if self._actuators is not None
                   and na_here > 0 else None), time=time)
          state._qpos = torch.where(success.unsqueeze(1), qpos, state._qpos)
          state._qvel = torch.where(success.unsqueeze(1), qvel, state._qvel)
          state._qacc = torch.where(success[:, None], acceleration, state._qacc)
          state._qacc_warmstart.copy_(torch.where(
              success[:, None], state._qacc, state._qacc_warmstart))
          state._time = torch.where(success, time, state._time)
          state._status = status.clone()
          if self._actuators is not None and na_here > 0 and weighted_dot is not None:
            state._act, self._next_act = self._next_act, state._act
          self._record_delay(pre_step_time, success)
          # Like pinned mj_advance, leave FK outputs from RK's last forward
          # substage untouched after integrating the weighted rate.
          state._generation += 1
          if (getattr(self, "_flex", None) is not None
              and getattr(self, "_sleep_schedule", None) is None):
            fk_dyn = self._smooth.run_device(state._qpos, state._qvel, getattr(state, "_mpos", None), getattr(state, "_mquat", None))
            self._flex.update_kinematics(dict(fk_dyn["poses"], root_com=fk_dyn["root_com"]), fk_dyn.get("cvel", None))
          if self._sensordata is not None:
            rollback_sens = (pre_sens if pre_sens is not None
                             else torch.zeros_like(self._sensordata))
            self._sensordata.copy_(torch.where(
                success.unsqueeze(1), self._sensordata, rollback_sens))
          if pre_warm is not None and cc is not None:
            failed_mask = ~success
            nr = int(cc.descriptor.nr)
            w_dbg = cc._workspace["workspace_debug"].reshape(self.batch_size, cc._debug_stride)
            slice_w = w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr]
            slice_w.copy_(torch.where(failed_mask.unsqueeze(1), pre_warm, slice_w))
          self._assembled_system_valid = (
              getattr(self, "_last_coupled", None) is not None)
          self._record_accepted_step(
              success, pre_step_gen, pre_step_time, state._status,
              state._qacc)
          _restore_failed_plugin_rows(plugin_step_snapshot, success)
          continue
        velocity_stage = None
        if pending_step1 is None:
          acceleration, solve_status, dynamics = self._acceleration(
              state._qpos, step_qvel, capture_fwdinv=fwdinv_enabled
          )
          self._update_energy_position(state._qpos, dynamics, None)
          self._update_energy_velocity({"dynamics": dynamics,
                                        "qvel": step_qvel})
        else:
          from mujoco_metal.forward_stages import ForwardStage
          record = self.validate_forward_stage_record(
              pending_step1, ForwardStage.VEL)
          self.prepare_forward_actuation(record)
          self.prepare_forward_acceleration(
              record, qfrc_applied=self._applied_force)
          constraint = self.prepare_forward_constraint(record)
          velocity_stage = record.values[ForwardStage.VEL]
          dynamics = velocity_stage["dynamics"]
          acceleration = constraint["qacc"]
          solve_status = constraint["status"]
          if fwdinv_enabled:
            self._solver_fwdinv.copy_(self.compare_forward_inverse(
                record, {"record": record, "constraint": constraint,
                         "actuation": record.values[ForwardStage.ACT]}))
          pending_step1 = None
          self._step1_record = None
        scheduler = getattr(self, "_sleep_schedule", None)
        if scheduler is not None:
          # Preserve fail-open capacity behavior (status 2 wakes all trees),
          # but make impossible sleeping-contact/cycle states explicit and
          # sticky in the ordinary per-world simulation status channel.
          sleep_status = scheduler.status
          sleep_failure = torch.where(
              (sleep_status == 1) | (sleep_status == 3), sleep_status,
              torch.zeros_like(sleep_status))
          solve_status = torch.where(
              solve_status == 0, sleep_failure, solve_status)
        solved_acceleration = acceleration
        if self._sensors is not None and not split_step2:
          # Stored step-stage sample at the pre-integration (forward) state,
          # matching pinned mj_step sensor timing; see step_sensordata.
          # ACC sensors observe the solved acceleration (pre-midpoint).
          self._store_step_sensors(state._qpos, state._qvel, dynamics,
                                   qacc=solved_acceleration,
                                   qacc_low=self._qacc_low_for_sensor(
                                       solved_acceleration))
        acceleration, step_qvel, solve_status, _check_reset = (
            self._check_step_acceleration(
                acceleration, step_qvel, solve_status))
        integration_acceleration = acceleration
        next_velocity = position_velocity = None
        flex_operators = None
        if self._flex is not None and (self._implicitfast is not None or self._implicit is not None):
          flex_operators = self._flex.run_material_operators(
              state._qpos, step_qvel,
              dict(dynamics["poses"], root_com=dynamics["root_com"]),
              dynamics.get("cvel"))
        if self._effective_implicit is not None:
          implicit = self._solve_sparse_effective_acceleration(
              state._qpos, step_qvel, dynamics)
          integration_acceleration = implicit["qacc"]
          solve_status = torch.where(
              solve_status == 0, implicit["status"], solve_status)
        elif self._implicitfast is not None:
          if bool(getattr(self._implicitfast.descriptor, "auto_derivative", False)):
            # R06/D2: natively assembled passive/tendon derivatives from the
            # stage tangents already computed above (None stages read as zero).
            fluid_jac = None
            if getattr(self, "_fluid", None) is not None:
              fluid_jac = self._fluid.run_derivative_device(
                  state._qpos, state._qvel, dynamics, self._smooth
              )
            implicit = self._implicitfast.run_device_auto(
                dynamics["mass_matrix"], self._rhs,
                getattr(self, "_damping_tangent", None)
                if getattr(self, "_passive", None) is not None else None,
                getattr(self, "_tendon_damping", None)
                if getattr(self, "_tendons", None) is not None else None,
                fluid_jacobian=fluid_jac,
                actuator_jacobian=getattr(self, "_actuator_velocity_derivative", None),
                flex_edge_derivative=(flex_operators["edge_velocity_derivative"]
                                      if flex_operators is not None else None),
            )
          else:
            implicit = self._implicitfast.run_device(
                dynamics["mass_matrix"], self._rhs
            )
          integration_acceleration = implicit["qacc"]
          solve_status = torch.where(
              solve_status == 0, implicit["status"], solve_status
          )
        elif self._implicit is not None:
          if bool(getattr(self._implicit.descriptor, "auto_derivative", False)):
            fluid_jac = None
            if getattr(self, "_fluid", None) is not None:
              fluid_jac = self._fluid.run_derivative_device(
                  state._qpos, state._qvel, dynamics, self._smooth
              )
            bias_deriv = None
            if hasattr(self._smooth, "bias_derivative_device"):
              bias_deriv = self._smooth.bias_derivative_device(
                  state._qpos, state._qvel, dynamics
              )
            implicit = self._implicit.run_device_auto(
                dynamics["mass_matrix"], self._rhs,
                getattr(self, "_damping_tangent", None)
                if getattr(self, "_passive", None) is not None else None,
                getattr(self, "_tendon_damping", None)
                if getattr(self, "_tendons", None) is not None else None,
                fluid_jacobian=fluid_jac,
                bias_derivative=bias_deriv,
                actuator_jacobian=getattr(self, "_actuator_velocity_derivative", None),
                flex_edge_derivative=(flex_operators["edge_velocity_derivative"]
                                      if flex_operators is not None else None),
            )
          else:
            implicit = self._implicit.run_device(
                dynamics["mass_matrix"], self._rhs
            )
          integration_acceleration = implicit["qacc"]
          solve_status = torch.where(
              solve_status == 0, implicit["status"], solve_status
          )
        if self._flex_implicit is not None:
          # Pinned mj_implicitSkip omits stiffness CG when any DOFs in that
          # world are sleep-filtered. No host readback decides this mask.
          enabled = None
          scheduler = getattr(self, "_sleep_schedule", None)
          if scheduler is not None:
            enabled = (scheduler.tree_awake.index_select(
                1, self._sleep_dof_treeid) != 0).all(dim=1).contiguous()
          corrected = self._flex_implicit.run_device(
              implicit["effective_mass"], implicit["full_effective_mass"],
              self._rhs, step_qvel, integration_acceleration, flex_operators,
              enabled=enabled)
          integration_acceleration = corrected["qacc"]
          # Pinned mj_implicitSkip applies flexInterp_cgsolve to its stack
          # qacc integration vector. mj_advance consumes that correction but
          # retains d->qacc from the final forward stage for public qacc and
          # qacc_warmstart (except midpoint's explicit post-advance override).
          solve_status = torch.where(
              solve_status == 0, corrected["status"], solve_status)
        elif self._sparse_flex_implicit is not None:
          scheduler = getattr(self, "_sleep_schedule", None)
          enabled = None
          dof_ids = counts = None
          if scheduler is not None:
            awake_lists = scheduler.awake_lists()
            dof_ids, counts = awake_lists["dof_ids"], awake_lists["counts"]
            enabled = (scheduler.tree_awake.index_select(
                1, self._sleep_dof_treeid) != 0).all(dim=1).contiguous()
          material_context = self._flex.capture_material_operator_context(
              dict(dynamics["poses"], root_com=dynamics["root_com"]),
              dynamics.get("cvel"))
          writer = self._velocity_derivative_values
          preconditioner_values = (
              writer.symmetric_values
              if int(self._mjmodel.opt.integrator) ==
              int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
              else writer.values)
          corrected = self._sparse_flex_implicit.run_device(
              flex=self._flex, context=material_context,
              effective_solver=self._effective_implicit,
              mass_blocks=dynamics["mass_blocks"],
              # Pinned flexInterp_cgsolve always forms A with full qDeriv;
              # implicitfast uses the lower-symmetric qH only in its
              # preconditioner solve.
              edge_values=self._sparse_flex_implicit.full_derivative_values,
              preconditioner_values=preconditioner_values,
              armature_blocks=dynamics.get("tendon_armature_blocks"),
              qfrc_total=self._rhs, qvel=step_qvel,
              initial_qacc=integration_acceleration,
              timestep=float(self.profile.timestep),
              dof_ids=dof_ids, counts=counts, enabled=enabled)
          integration_acceleration = corrected["qacc"]
          solve_status = torch.where(
              solve_status == 0, corrected["status"], solve_status)
        if self._midpoint is not None:
          midpoint = self._midpoint.run_device(
              state._qvel,
              integration_acceleration,
              acceleration,
              self._rhs + dynamics["qfrc_bias"],
              dynamics["poses"]["body_quat"],
              eligible_mask=self._midpoint_eligibility(),
          )
          next_velocity = midpoint["qvel_next"]
          position_velocity = midpoint["position_velocity"]
          acceleration = midpoint["qacc"]
          solve_status = torch.where(
              solve_status == 0, midpoint["status"], solve_status
          )
        if (self._euler_solver is not None
            or (self._component_mass_enabled
                and self.profile.execution_plan.is_stage_enabled("euler_damping"))):
          scheduler = getattr(self, "_sleep_schedule", None)
          euler_awake = scheduler.awake_lists() if scheduler is not None else None
          damping_source = _select_euler_damping_input(
              self._passive is not None, velocity_stage,
              self._damping_tangent,
              self._damping.unsqueeze(0) if self._passive is None else None)
          if self._component_mass_enabled:
            # Split step2 consumes the already-published VEL record. Its
            # passive workspace may since have been reused, so use the
            # derivative owned by that record rather than a global scratch.
            # Passive outputs may be borrowed from padded/interleaved source
            # workspaces. The component operator accepts only its prepared
            # contiguous [B,nv] input; reuse this owned vector rather than
            # materializing a per-step contiguous copy.
            self._component_damping_deriv.copy_(damping_source)
            q_deriv = self._component_damping_deriv
            diagonal_add = self._component_solver.build_euler_diagonal_device(
                q_deriv, self.profile.timestep,
                dof_ids=(euler_awake["dof_ids"]
                         if euler_awake is not None else None),
                counts=(euler_awake["counts"]
                        if euler_awake is not None else None))
            euler_acceleration, euler_status = self._solve_component_rhs(
                dynamics, self._rhs, euler_awake,
                diagonal_add=diagonal_add, allow_indefinite=True)
          else:
            self._effective_mass.copy_(dynamics["mass_matrix"])
            self._effective_mass.diagonal(dim1=1, dim2=2).add_(
                self.profile.timestep * damping_source
                if self._passive is not None
                else self._implicit_damping
            )
            euler_acceleration, euler_status = self._euler_solver.run_device(
                self._effective_mass, self._rhs, awake_lists=euler_awake,
                retained=integration_acceleration if euler_awake is not None else None
            )
          # mj_EulerSkip checks only awake DOFs. A world whose damping lives
          # exclusively in sleeping trees must retain the forward iterate,
          # including at a finite/zero constraint-solver iteration budget.
          scheduler = getattr(self, "_sleep_schedule", None)
          if scheduler is not None:
            awake = scheduler.tree_awake.index_select(
                1, self._sleep_dof_treeid) != 0
            enabled = (awake & self._euler_damping_dofs).any(dim=1)
            integration_acceleration = torch.where(
                enabled[:, None], euler_acceleration, integration_acceleration)
            euler_status = torch.where(enabled, euler_status, 0)
          else:
            integration_acceleration = euler_acceleration
          torch.where(
              (solve_status == 0),
              euler_status,
              solve_status,
              out=self._combined_status,
          )
          solve_status = self._combined_status
        torch.eq(state._status, 0, out=self._success)
        torch.where(
            self._success, solve_status, state._status, out=self._combined_status
        )
        torch.where(
            self._sensor_plugin_status == 0, self._combined_status,
            self._sensor_plugin_status, out=self._combined_status)
        self._success.copy_(self._combined_status == 0)
        if self._actuators is not None and getattr(state, "_na", 0) > 0:
          self._advance_activations(state)
        integration_awake = self._prepare_sleep_integration(
            step_qvel, acceleration, self._success)
        if getattr(self, "_sleep_schedule", None) is not None:
          integration_acceleration = integration_acceleration.clone()
          next_velocity = next_velocity.clone() if next_velocity is not None else None
          position_velocity = position_velocity.clone() if position_velocity is not None else None
        acceleration = self._refresh_sleep_velocity(
            state._qpos, step_qvel, acceleration, self._success)
        qpos, qvel, time, status = self._integrator.run_device(
            state._qpos,
            step_qvel,
            integration_acceleration,
            state._time,
            self._combined_status,
            next_velocity=next_velocity,
            position_velocity=position_velocity,
            awake_lists=integration_awake,
        )
        torch.eq(status, 0, out=self._success)
        torch.where(
            self._success.unsqueeze(1),
            acceleration,
            state._qacc,
            out=self._next_qacc,
        )
        torch.where(
            self._success.unsqueeze(1),
            qpos,
            state._qpos,
            out=self._next_qpos,
        )
        torch.where(
            self._success.unsqueeze(1),
            qvel,
            state._qvel,
            out=self._next_qvel,
        )
        torch.where(
            self._success,
            time,
            state._time,
            out=self._next_time,
        )
        self._next_status.copy_(status)
        if self._actuators is not None and getattr(state, "_na", 0) > 0:
          self._next_act.copy_(torch.where(
              self._success[:, None], self._next_act, state._act))

        self._advance_native_actuator_plugins(
            self._success, qpos=self._next_qpos, qvel=self._next_qvel,
            qacc=self._next_qacc,
            act=(self._next_act if self._actuators is not None
                 and getattr(state, "_na", 0) > 0 else None),
            time=self._next_time)

        # Ping-pong owned tensors keep live state disjoint from borrowed stage
        # outputs and make each new current-state reference safe for the next
        # invocation. DeviceState reset/restore may replace these references.
        state._qpos, self._next_qpos = self._next_qpos, state._qpos
        state._qvel, self._next_qvel = self._next_qvel, state._qvel
        state._qacc, self._next_qacc = self._next_qacc, state._qacc
        state._qacc_warmstart.copy_(torch.where(
            self._success.unsqueeze(1), state._qacc,
            state._qacc_warmstart))
        state._time, self._next_time = self._next_time, state._time
        state._status, self._next_status = self._next_status, state._status
        if self._actuators is not None and getattr(state, "_na", 0) > 0:
          state._act, self._next_act = self._next_act, state._act
        # Pinned history advance (mj_advance records ctrl at the pre-step
        # time); successful worlds only, failed rows freeze (R02).
        self._record_delay(pre_step_time, self._success)
        self._restore_failed_sleep_rows(sleep_step_snapshot, self._success)
        # The prepared FK pose is the last forward state X0. mj_advance
        # integrates qpos but does not refresh xpos/xquat, so keep this
        # baseline for next-step sleeping-pose mismatch detection.
        state._generation += 1
        if (getattr(self, "_flex", None) is not None
            and getattr(self, "_sleep_schedule", None) is None):
          fk_dyn = self._smooth.run_device(state._qpos, state._qvel, getattr(state, "_mpos", None), getattr(state, "_mquat", None))
          self._flex.update_kinematics(dict(fk_dyn["poses"], root_com=fk_dyn["root_com"]), fk_dyn.get("cvel", None))

        # Failure atomicity (R02): rollback sensor samples and warmstarts for failed worlds
        if self._sensordata is not None:
          rollback_sens = (pre_sens if pre_sens is not None
                           else torch.zeros_like(self._sensordata))
          self._sensordata.copy_(torch.where(
              self._success.unsqueeze(1), self._sensordata, rollback_sens))
        if pre_warm is not None and cc is not None:
          failed_mask = ~self._success
          nr = int(cc.descriptor.nr)
          w_dbg = cc._workspace["workspace_debug"].reshape(self.batch_size, cc._debug_stride)
          slice_w = w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr]
          slice_w.copy_(torch.where(failed_mask.unsqueeze(1), pre_warm, slice_w))

        # Accepted step record (R03/R04): exact immutable snapshot of the solved system
        self._assembled_system_valid = (
            getattr(self, "_last_coupled", None) is not None)
        self._record_accepted_step(
            self._success, pre_step_gen, pre_step_time, state._status,
            state._qacc)
        _restore_failed_plugin_rows(plugin_step_snapshot, self._success)
      except Exception:
        _restore_plugins_device(plugin_step_snapshot)
        if pre_sens is not None and self._sensordata is not None:
          self._sensordata.copy_(pre_sens)
        if pre_warm is not None and cc is not None:
          nr = int(cc.descriptor.nr)
          w_dbg = cc._workspace["workspace_debug"].reshape(
              self.batch_size, cc._debug_stride)
          w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr].copy_(pre_warm)
        if pre_act_dot is not None:
          self._act_dot.copy_(pre_act_dot)
        self._restore_failed_sleep_rows(
            sleep_step_snapshot,
            torch.zeros(self.batch_size, dtype=torch.bool, device=state._device))
        raise
    return state._status

  def step_sensordata(self):
    """Return a host copy of the stored step-stage sensor sample.

    The sample is taken at every step's pre-integration forward state for
    the model's position/velocity stages (acceleration stages join in the
    force-family milestone), exactly the values MuJoCo's ``mj_step`` leaves
    in ``d->sensordata``. This differs from :meth:`sensor_values`, which
    re-evaluates at the current (post-step) state. Reset zeroes the sample;
    a state restore leaves it untouched until the next step recomputes it.
    """
    if self._sensors is None or self._sensordata is None:
      raise ValueError("step_sensordata requires a sensor stepping profile")
    return self._sensordata.detach().cpu().numpy().copy()

  def _store_step_sensors(self, qpos, qvel, dynamics, qacc=None, *,
                          qacc_low=_AUTO_SENSOR_QACC_LOW,
                          stages=(mujoco.mjtStage.mjSTAGE_POS, mujoco.mjtStage.mjSTAGE_VEL,
                                  mujoco.mjtStage.mjSTAGE_ACC),
                          compute_worlds=None, sleep_counts=None,
                          sleep_tree_awake=None, act_override=None,
                          position_context=None, include_spatial=True):
    """Evaluate position/velocity (+ACC, when present) stages into storage."""
    import mujoco as _mj
    torch = self._state._torch
    if self._sensordata is None:
      self._sensordata = torch.zeros(
          (self._state.batch_size, self._mjmodel.nsensordata),
          dtype=torch.float32, device=self._state._device)
    previous = self._sensordata.clone()
    previous_raw = (self._raw_sensordata.clone() if self._raw_sensordata is not None
                    else previous.clone())
    program = self._raw_sensor_program
    if self._sensor_sleep_policy is not None:
      awake = self._sensor_sleep_policy.run_device(
          self._sleep_schedule.tree_awake if sleep_tree_awake is None
          else sleep_tree_awake)
      counts = self._sleep_schedule.awake_counts if sleep_counts is None else sleep_counts
      # Pinned sensors consult tree sleep state only when body indices are
      # already filtered. A newly sleeping world still has its old indices
      # during the velocity-forward refresh.
      awake = torch.where((counts[:, :1] < int(self._mjmodel.nbody)),
                           awake, torch.ones_like(awake))
      if compute_worlds is not None:
        awake = awake * compute_worlds[:, None].to(dtype=awake.dtype)
      program.set_sensor_awake_mask(awake)
      if self._sensors is not program:
        self._sensors.set_sensor_awake_mask(awake)
    elif hasattr(program, "set_sensor_awake_mask"):
      program.set_sensor_awake_mask(None)
    poses = dict(dynamics["poses"], cvel=dynamics["cvel"],
                 root_com=dynamics["root_com"])
    ordinary_stages = tuple(stage for stage in stages if stage != _mj.mjtStage.mjSTAGE_ACC)
    old = program.run_device(
        qpos, qvel, self._state._time, poses, stages=ordinary_stages,
        sensordata=previous.clone(), world_mask=compute_worlds)
    merged = self._run_state_sensors(qpos, qvel, poses, dynamics, out=old,
                                     program=program, stages=ordinary_stages,
                                     position_context=position_context,
                                     world_mask=compute_worlds)
    if self._has_acc_sensors and qacc is not None and _mj.mjtStage.mjSTAGE_ACC in stages:
      # _acceleration just ran at this exact state; its cached actuator
      # rows plus the solved constraint rows feed the ACC kernel.
      merged = self._run_acc_into(qpos, qvel, qacc, poses, dynamics, merged,
                                  qacc_low=qacc_low, program=program,
                                  world_mask=compute_worlds)
    if (self._has_touch_grid_sensors and qacc is not None
        and _mj.mjtStage.mjSTAGE_ACC in stages):
      merged = self._run_touch_grid_into(
          poses, merged, compute_worlds=compute_worlds,
          record_status=True)
    if (include_spatial and (self._has_contact_sensors or self._has_ray_sensors
                             or self._has_geomdist_sensors)):
      merged = self._run_spatial_into(
          poses, merged, program=program, world_mask=compute_worlds)
    if getattr(self, "_has_user_plugin_sensors", False):
      import mujoco as _mj
      sensor_plugins = [p for p in self._native_plugins
                        if p.plugin_type.value == "sensor"]
      sensor_plugin_states = _snapshot_plugins_device(sensor_plugins)
      sensor_compute = self._history_program.sensor_compute_mask(
          self._state._history, self._state._time)
      plugin_state = _plugin_stage_state(
          self._state, qpos, qvel, qacc=qacc,
          act=act_override)
      for i, typ in enumerate(self._sensors.descriptor.sensor_type):
        if typ in (int(_mj.mjtSensor.mjSENS_USER), int(_mj.mjtSensor.mjSENS_PLUGIN)):
          adr = int(self._sensors.descriptor.sensor_adr[i])
          dim = int(self._sensors.descriptor.sensor_dim[i])
          sname = _mj.mj_id2name(self._mjmodel, int(_mj.mjtObj.mjOBJ_SENSOR), i) or "default"
          plugin = next((p for p in sensor_plugins if p.name == sname), None)
          if plugin is None:
            plugin = next((p for p in sensor_plugins if p.name == "default"), None)
          if plugin is not None:
            tick_mask = sensor_compute[:, adr:adr + dim].any(dim=1).contiguous()
            if int(self._sensors.descriptor.sensor_needstage[i]) not in tuple(int(s) for s in stages):
              continue
            if compute_worlds is not None:
              tick_mask = (tick_mask & compute_worlds).contiguous()
            state_before = plugin.device_snapshot()
            sensor_before = merged[:, adr:adr + dim].clone()
            try:
              output = plugin.run_device(plugin_state, sensordata=merged,
                                         sensor_adr=adr, sensor_dim=dim,
                                         compute_mask=tick_mask)
              if output is not None and (
                  not isinstance(output, torch.Tensor)
                  or output.device != merged.device
                  or output.dtype != torch.float32
                  or tuple(output.shape) != (self.batch_size, dim)):
                raise ValueError(f"sensor plugin {_plugin_state_key(plugin)!r} returned an invalid tensor")
              computed = (output if output is not None
                          else merged[:, adr:adr + dim])
              valid_rows = torch.isfinite(computed).all(dim=1)
              invalid = tick_mask & ~valid_rows
              self._sensor_plugin_status.copy_(torch.where(
                  invalid, torch.ones_like(self._sensor_plugin_status),
                  self._sensor_plugin_status))
              merged[:, adr:adr + dim] = torch.where(
                  tick_mask.unsqueeze(1), computed, sensor_before)
              # Enforce the state contract even for callbacks that update
              # eagerly: worlds outside this sensor's compiled tick mask
              # retain their previous plugin state.
              plugin.restore_masked(state_before, tick_mask)
            except Exception:
              _restore_plugins_device(sensor_plugin_states)
              raise
    if compute_worlds is not None:
      merged = torch.where(compute_worlds[:, None], merged, previous_raw)
    if self._raw_sensordata is None:
      self._raw_sensordata = merged.contiguous().clone()
    elif compute_worlds is None:
      self._raw_sensordata = merged.contiguous().clone()
    else:
      self._state.copy_masked_rows(
          self._raw_sensordata, merged.contiguous(), compute_worlds)
      merged = self._raw_sensordata
    sampled = self._history_program.samples(
        merged, self._state._history, self._state._time, kind=1)
    selected = sampled if self._history_program.sensor_enabled else previous
    if compute_worlds is None:
      self._sensordata.copy_(selected)
    else:
      self._state.copy_masked_rows(
          self._sensordata, selected.contiguous(), compute_worlds)

  def _advance_activations(self, state):
    """Advance activation state with pinned mj_nextActivation semantics.

    Under mjDSBL_ACTUATION the advance is skipped but the next buffer is
    still defined (live state carried over), so the end-of-step transaction
    preserves activation exactly. Failed worlds keep their previous
    activation via the final accepted-step mask, matching the qpos/qvel
    rollback contract.
    """
    import mujoco as _mj
    torch = state._torch
    if int(self._mjmodel.opt.disableflags) & int(_mj.mjtDisableBit.mjDSBL_ACTUATION):
      self._next_act.copy_(state._act)
      return
    new_act = self._actuators.advance(state._act, self._act_dot, self._act_vel)
    torch.where(
        self._success.unsqueeze(1),
        new_act,
        state._act,
        out=self._next_act,
    )

  @property
  def flex(self):
    """Underlying MetalFlex instance if model contains flex elements, else None."""
    return self._flex

  @property
  def flexvert_xpos(self):
    """World positions of flex vertices (B, nflexvert, 3) or None."""
    return self._flex.flexvert_xpos if self._flex is not None else None

  @property
  def flexedge_length(self):
    """Current lengths of flex edges (B, nflexedge) or None."""
    return self._flex.flexedge_length if self._flex is not None else None

  @property
  def flexedge_velocity(self):
    """Current deformation rates of flex edges (B, nflexedge) or None."""
    return self._flex.flexedge_velocity if self._flex is not None else None
