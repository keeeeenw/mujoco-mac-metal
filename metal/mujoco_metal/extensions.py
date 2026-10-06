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

"""Milestone 019: Native extension registration and lifecycle.

Provides:
- Native plugin architecture (SENSOR, FORCE, ACTUATOR).
- Registration and lifecycle management (init, run_device, reset, snapshot/restore).
- Executable native GPU compute plugins on MPS tensors.
- Strict isolation: Native mode does NOT silently invoke CPU callbacks;
  host callbacks require explicit `allow_host_callbacks=True`.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import copy
import threading
import weakref
import numbers
import mujoco
import numpy as np

try:
  import torch
except ImportError:
  torch = None


class PluginType(Enum):
  """Supported extension plugin roles in the physics pipeline."""
  SENSOR = "sensor"
  FORCE = "force"
  ACTUATOR = "actuator"


class NativePlugin:
  """Base class for device-native plugins."""

  spatial_force_workspace_kind = None
  device_workspace_bytes = 0

  def __init__(self, name: str, plugin_type: PluginType):
    self.name = str(name)
    self.plugin_type = plugin_type
    self.device = None
    self.batch_size = 1
    self.model = None

  def init(self, model: mujoco.MjModel, batch_size: int = 1, device=None):
    """Initialize plugin buffers and bind to model and batch size."""
    self.model = model
    self.batch_size = int(batch_size)
    requested = (device or (torch.device("mps")
                            if torch is not None and torch.backends.mps.is_available()
                            else "cpu"))
    if torch is not None:
      # Torch's unindexed MPS device compares unequal to the actual `mps:0`
      # device on tensors. Resolve once through an allocation-free empty
      # device view and keep that canonical owner for every plugin mask.
      self.device = torch.empty((0,), dtype=torch.uint8,
                                device=requested).device
    else:
      self.device = requested
    self._runtime_device = self.device

  def _owned_device(self):
    """Return the canonical device resolved when plugin storage is bound."""
    return torch.device(self._runtime_device)

  def run_device(self, state, **kwargs):
    """Execute plugin computation natively on device tensors.
    
    Must operate entirely on MPS tensors without CPU readback.
    ``state`` is a read-only view of this forward stage: its public qpos,
    qvel, qacc, act and time properties refer to the stage being evaluated,
    including RK4 sub-stages and post-sleep velocity refreshes. Do not call
    state mutation/reset/restore methods or mutate its private tensor fields;
    only the plugin's own registered device state may be changed.
    An optional ``compute_mask`` is a bool device tensor with shape [batch].
    It selects worlds requiring a conditional stage (for example the extra
    velocity forward on a sleep transition). Stateful plugins should gate
    updates by this mask; the simulation also restores device state for
    unselected worlds. Callbacks must not rely on untracked host side effects
    to represent simulation state.
    """
    raise NotImplementedError("NativePlugin subclasses must implement run_device")

  def create_instance(self):
    """Create an unbound instance from an unbound configuration prototype.

    Override this when configuration contains resources that cannot be copied,
    or register an explicit factory. Runtime buffers belong to ``init`` and
    must never be shared between simulations.
    """
    if self.model is not None or self.device is not None:
      raise ValueError("register an unbound plugin configuration or a factory")
    return copy.deepcopy(self)

  def run_host(self, data, **kwargs):
    """Optional host callback for legacy CPU execution.
    
    Only invoked when allow_host_callbacks is explicitly True.
    """
    raise NotImplementedError("run_host is not implemented for this plugin")

  def reset(self, env_ids=None):
    """Reset plugin internal states for selected or all worlds."""
    pass

  def reset_masked(self, reset_mask):
    """Reset selected runtime worlds from an on-device boolean mask.

    Automatic numerical recovery cannot transfer bad-world indices to the
    host. Stateful plugins must override this method and reset their owned
    device state where ``reset_mask`` is true. Stateless plugins need no
    payload and inherit the no-op implementation.
    """
    if torch is None or not isinstance(reset_mask, torch.Tensor):
      raise TypeError("reset_mask must be a device tensor")
    if (reset_mask.dtype != torch.bool
        or tuple(reset_mask.shape) != (self.batch_size,)
        or reset_mask.device != self._owned_device()):
      raise ValueError("reset_mask must be bool[batch_size] on the plugin device")
    if type(self).device_snapshot is not NativePlugin.device_snapshot:
      raise NotImplementedError(
          "stateful native plugins must implement device-native reset_masked")

  def snapshot(self):
    """Capture plugin internal state for replay."""
    return None

  def restore(self, snap, env_ids=None):
    """Restore plugin internal state from checkpoint."""
    pass

  def device_snapshot(self):
    """Return a device-resident copy of live state for step rollback.

    This is distinct from :meth:`snapshot`, which is the explicit host
    checkpoint boundary used by simulation snapshots. Stateful plugins that
    participate in stepping must override this method and the matching device
    restore hooks; copying a solver acceptance mask to the host is forbidden.
    Stateless plugins may keep the default ``None`` implementation.
    """
    return None

  def device_snapshot_bytes(self):
    """Return the exact byte bound for :meth:`device_snapshot` allocations.

    Stateless plugins inherit the zero-byte contract. A stateful plugin that
    overrides ``device_snapshot`` must also override this method so native
    derivative queries can admit their complete rollback checkpoint before
    allocating it.
    """
    if type(self).device_snapshot is not NativePlugin.device_snapshot:
      raise NotImplementedError(
          "stateful plugins must declare device_snapshot_bytes")
    return 0

  def restore_device(self, snapshot):
    """Restore a complete device snapshot without host transfers."""
    if snapshot is not None:
      raise NotImplementedError(
          "stateful native plugins must implement restore_device")

  def restore_masked(self, snapshot, accepted_mask):
    """Restore only failed worlds using a device boolean success mask.

    ``accepted_mask`` has shape ``[batch_size]`` and remains on ``device``.
    Implementations must preserve their current state where the mask is true
    and restore ``snapshot`` where it is false, without numerical-status
    readback. The base implementation accepts only stateless plugins.
    """
    if snapshot is None:
      return
    if torch is None or not isinstance(accepted_mask, torch.Tensor):
      raise TypeError("accepted_mask must be a device tensor")
    if (accepted_mask.dtype != torch.bool
        or tuple(accepted_mask.shape) != (self.batch_size,)
        or accepted_mask.device != self._owned_device()):
      raise ValueError("accepted_mask must be bool[batch_size] on the plugin device")
    raise NotImplementedError(
        "stateful native plugins must implement device-native restore_masked")


@dataclass(frozen=True)
class ActuatorPluginOutput:
  """Borrowed native actuator outputs for one force evaluation.

  ``force`` is actuator-space force ``[batch, nu]``. The simulation projects
  it through the current transmission moment matrix. ``activation_derivative``
  is an optional additive contribution in compiled activation order
  ``[batch, na]``. Both tensors remain owned by the plugin and are valid only
  until its next ``run_actuator_device`` call.
  """

  force: object
  activation_derivative: object | None = None


class NativeActuatorPlugin(NativePlugin):
  """Typed actuator-space extension with accepted-step state advancement.

  Implementations return :class:`ActuatorPluginOutput` from
  ``run_actuator_device``. That callback is an evaluation: it may run several
  times per physical timestep (RK4 stages included) and must not advance
  persistent plugin state. Stateful plugins update only in
  ``advance_actuator_device`` with the accepted-world mask. The simulation
  snapshots plugin-owned tensors around a step, restores failed worlds, and
  includes explicit snapshot/reset/restore state in its ordinary lifecycle.

  This typed protocol is separate from legacy ``PluginType.ACTUATOR``
  callbacks, whose ``run_device`` result is a generalized-force tensor. That
  older protocol remains supported for source compatibility.
  """

  def __init__(self, name: str):
    super().__init__(name, PluginType.ACTUATOR)

  def run_actuator_device(self, state, *, control, kinematics, dynamics,
                          act=None, act_dot=None, compute_mask=None):
    """Return actuator-space force and optional activation derivative."""
    raise NotImplementedError(
        "NativeActuatorPlugin subclasses must implement run_actuator_device")

  def advance_actuator_device(self, state, *, accepted_mask, timestep,
                              output):
    """Advance plugin state once after accepted simulation worlds integrate."""
    del state, accepted_mask, timestep, output


class NativeActuatorUserPlugin(NativePlugin):
  """Device-native implementation of MuJoCo's USER actuator callbacks.

  Bind explicit actuator IDs using ``dynamics``, ``gain`` and ``bias``.
  Subclasses implement :meth:`run_user_actuator_device` and write device
  output planes; arbitrary Python numerical callbacks are never run by the native
  simulation. Evaluation may occur at multiple RK substages and must be pure
  with respect to persistent plugin state. Advance state once in
  :meth:`advance_user_actuator_device` after accepted integration.

  An unbound USER model actuator retains MuJoCo's pinned no-callback defaults:
  zero activation derivative, unit gain and zero bias.

  The pinned implicit actuator Jacobian treats USER gain/bias velocity
  derivatives as zero. A callback may still read velocity when evaluating
  force, but this interface does not declare an implicit derivative for that
  custom force law.

  ``device_workspace_bytes`` must include every persistent plugin-owned device
  allocation and rollback/checkpoint copy. The output planes themselves are
  owned and accounted by ``MetalActuators``.
  """

  def __init__(self, name: str, *, dynamics=(), gain=(), bias=(),
               device_workspace_bytes=0):
    super().__init__(name, PluginType.ACTUATOR)
    if (isinstance(device_workspace_bytes, bool)
        or not isinstance(device_workspace_bytes, numbers.Integral)
        or device_workspace_bytes < 0):
      raise ValueError("device_workspace_bytes must be a nonnegative integer")
    self.device_workspace_bytes = int(device_workspace_bytes)
    self.actuator_user_bindings = _normalize_actuator_user_bindings(
        {"dynamics": dynamics, "gain": gain, "bias": bias})
    if not any(ids for _, ids in self.actuator_user_bindings):
      raise ValueError("NativeActuatorUserPlugin must bind at least one USER callback")

  def run_device(self, state, **kwargs):
    del state, kwargs
    raise NotImplementedError(
        "NativeActuatorUserPlugin is invoked by the actuator USER stage")

  def run_user_actuator_device(self, state, *, control, activation,
                               activation_derivative, kinematics, time,
                               compute_mask, out_activation_derivative,
                               out_gain, out_bias, dynamics_ids, gain_ids,
                               bias_ids):
    """Write registered USER callback values into borrowed device planes.

    Shapes are ``out_activation_derivative[B, max(na, 1)]`` and
    ``out_gain/out_bias[B, max(nu, 1)]``. Write only the activation slots and
    actuator IDs in ``dynamics_ids``, ``gain_ids`` and ``bias_ids``. Every
    declared value must be
    overwritten for each selected world on each evaluation; unregistered rows
    retain the pinned no-callback defaults.
    """
    raise NotImplementedError(
        "NativeActuatorUserPlugin subclasses must implement "
        "run_user_actuator_device")

  def advance_user_actuator_device(self, state, *, accepted_mask, timestep,
                                   output):
    """Advance plugin state once for accepted worlds.

    ``output`` is a step-local mapping of borrowed activation-derivative,
    gain and bias tensors plus the bound actuator IDs. It is valid only during
    this call; copy values into plugin-owned state before returning if needed.
    """
    del state, accepted_mask, timestep, output


def _normalize_actuator_user_bindings(bindings):
  if not isinstance(bindings, dict):
    raise TypeError("actuator USER bindings must be a role-to-ID mapping")
  roles = ("dynamics", "gain", "bias")
  if set(bindings) - set(roles):
    raise ValueError("actuator USER bindings contain an unknown role")
  normalized = []
  for role in roles:
    values = bindings.get(role, ())
    ids = []
    seen = set()
    for value in values:
      if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
          or value < 0 or value in seen):
        raise ValueError(
            f"actuator USER {role} IDs must be unique nonnegative integers")
      seen.add(value)
      ids.append(int(value))
    normalized.append((role, tuple(ids)))
  return tuple(normalized)


def validate_actuator_user_bindings(model, registrations):
  """CPU-only binding/type guard run before Simulation initializes MPS."""
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  import numpy as _np
  roles = {
      "dynamics": int(mujoco.mjtDyn.mjDYN_USER),
      "gain": int(mujoco.mjtGain.mjGAIN_USER),
      "bias": int(mujoco.mjtBias.mjBIAS_USER),
  }
  tags = {
      "dynamics": _np.asarray(model.actuator_dyntype),
      "gain": _np.asarray(model.actuator_gaintype),
      "bias": _np.asarray(model.actuator_biastype),
  }
  used = {role: set() for role in roles}
  for descriptor in registrations:
    bindings = tuple(descriptor.actuator_user_bindings)
    if not bindings:
      continue
    if descriptor.plugin_type != PluginType.ACTUATOR:
      raise ValueError("actuator USER bindings require an ACTUATOR plugin")
    # Validate factories before Simulation initializes MPS. Factories are
    # cheap, side-effect-free constructors; runtime buffers belong in init().
    probe = descriptor.factory()
    if (not isinstance(probe, NativeActuatorUserPlugin)
        or probe.name != descriptor.name
        or probe.plugin_type != descriptor.plugin_type
        or tuple(probe.actuator_user_bindings) != bindings
        or int(probe.device_workspace_bytes)
            != int(descriptor.device_workspace_bytes)
        or probe.model is not None or probe.device is not None):
      raise ValueError(
          f"actuator USER factory {descriptor.name!r} must create a matching "
          "fresh NativeActuatorUserPlugin")
    for role, ids in bindings:
      if role not in roles:
        raise ValueError("actuator USER binding has unknown role")
      for actuator_id in ids:
        if actuator_id >= int(model.nu):
          raise ValueError(
              f"actuator USER plugin {descriptor.name!r} binds out-of-range ID")
        if actuator_id in used[role]:
          raise ValueError(
              f"actuator {actuator_id} has multiple registered {role} callbacks")
        if int(tags[role][actuator_id]) != roles[role]:
          raise ValueError(
              f"actuator {actuator_id} callback binding requires {role} USER type")
        if role == "dynamics" and int(model.actuator_actnum[actuator_id]) == 0:
          raise ValueError(
              f"actuator {actuator_id} has no activation slots for dynamics callback")
        used[role].add(actuator_id)


@dataclass(frozen=True)
class PluginRegistration:
  """Immutable role/name/factory descriptor; never a bound runtime plugin."""

  name: str
  plugin_type: PluginType
  factory: object
  spatial_force_kind: str | None = None
  actuator_user_bindings: tuple = ()
  device_workspace_bytes: int = 0


class ExtensionRegistry:
  """Thread-safe registry for device-native physics plugins."""

  def __init__(self):
    self._plugins: dict[tuple[str, PluginType], NativePlugin] = {}
    self._registrations: dict[tuple[str, PluginType], PluginRegistration] = {}
    self._instances = weakref.WeakSet()
    self._lock = threading.RLock()
    self.allow_host_callbacks = False

  def register(self, plugin: NativePlugin):
    """Register a NativePlugin instance."""
    if not isinstance(plugin, NativePlugin):
      raise TypeError("plugin must be an instance of NativePlugin")
    if not isinstance(plugin.plugin_type, PluginType) or not plugin.name:
      raise ValueError("plugin requires a nonempty name and PluginType role")
    # Snapshot the configuration now. Later mutations to the supplied prototype
    # cannot reconfigure an existing registration or another simulation.
    blueprint = plugin.create_instance()
    if blueprint is plugin or not isinstance(blueprint, NativePlugin):
      raise ValueError("create_instance must return a distinct NativePlugin")
    if (getattr(blueprint, "actuator_user_bindings", ())
        and not isinstance(blueprint, NativeActuatorUserPlugin)):
      raise TypeError(
          "actuator USER bindings require NativeActuatorUserPlugin")
    with self._lock:
      spatial_kind = getattr(blueprint, "spatial_force_workspace_kind", None)
      user_bindings = getattr(blueprint, "actuator_user_bindings", ())
      workspace_bytes = getattr(blueprint, "device_workspace_bytes", 0)
      self._add_registration(
          plugin.name, plugin.plugin_type, blueprint.create_instance,
          spatial_force_kind=spatial_kind,
          actuator_user_bindings=user_bindings,
          device_workspace_bytes=workspace_bytes)
      self._plugins[(plugin.name, plugin.plugin_type)] = plugin

  def _add_registration(self, name, plugin_type, factory, *,
                        spatial_force_kind=None, actuator_user_bindings=(),
                        device_workspace_bytes=0):
    if spatial_force_kind not in (None, "magnetic", "site_feedback"):
      raise ValueError("unknown spatial force workspace kind")
    if isinstance(actuator_user_bindings, dict):
      actuator_user_bindings = _normalize_actuator_user_bindings(
          actuator_user_bindings)
    else:
      actuator_user_bindings = tuple(actuator_user_bindings)
    if (isinstance(device_workspace_bytes, bool)
        or not isinstance(device_workspace_bytes, numbers.Integral)
        or device_workspace_bytes < 0):
      raise ValueError("device_workspace_bytes must be a nonnegative integer")
    key = (name, plugin_type)
    if key in self._registrations:
      raise ValueError(f"Plugin {name!r} of type {plugin_type.value!r} already registered")
    self._registrations[key] = PluginRegistration(
        name, plugin_type, factory, spatial_force_kind,
        tuple(actuator_user_bindings), int(device_workspace_bytes))

  def register_factory(self, name, plugin_type, factory, *,
                       spatial_force_kind=None, actuator_user_bindings=(),
                       device_workspace_bytes=0):
    """Register a zero-argument factory returning a fresh unbound plugin."""
    if not isinstance(name, str) or not name or not isinstance(plugin_type, PluginType):
      raise ValueError("registration requires a nonempty name and PluginType role")
    if not callable(factory):
      raise TypeError("plugin factory must be callable")
    if spatial_force_kind not in (None, "magnetic", "site_feedback"):
      raise ValueError("unknown spatial force workspace kind")
    with self._lock:
      self._add_registration(
          name, plugin_type, factory, spatial_force_kind=spatial_force_kind,
          actuator_user_bindings=actuator_user_bindings,
          device_workspace_bytes=device_workspace_bytes)

  def registration_snapshot(self):
    """Return the immutable registrations a consumer can safely preflight."""
    with self._lock:
      return tuple(self._registrations.values())

  def spatial_force_workspace_counts(self, registrations=None):
    """Count per-instance built-in spatial FORCE workspaces.

    Supplying a snapshot ties capacity admission to the same descriptors that
    will later be instantiated, even if another thread edits the registry.
    """
    descriptors = (self.registration_snapshot() if registrations is None
                   else tuple(registrations))
    kinds = [entry.spatial_force_kind for entry in descriptors]
    return {kind: kinds.count(kind) for kind in ("magnetic", "site_feedback")}

  def instantiate(self, model, batch_size=1, device=None, *,
                  registrations=None):
    """Bind fresh instances to one simulation without changing registrations.

    The descriptor list is captured atomically. Registration/unregistration after
    construction affects future simulations only. Failed construction never
    rebinds an existing simulation's instance.
    """
    descriptors = (self.registration_snapshot() if registrations is None
                   else tuple(registrations))
    if any(not isinstance(desc, PluginRegistration) for desc in descriptors):
      raise TypeError("registrations must come from registration_snapshot()")
    instances = []
    for desc in descriptors:
      plugin = desc.factory()
      if (not isinstance(plugin, NativePlugin) or plugin.name != desc.name
          or plugin.plugin_type != desc.plugin_type
          or getattr(plugin, "spatial_force_workspace_kind", None)
              != desc.spatial_force_kind
          or tuple(getattr(plugin, "actuator_user_bindings", ()))
              != tuple(desc.actuator_user_bindings)
          or int(getattr(plugin, "device_workspace_bytes", 0))
              != int(desc.device_workspace_bytes)):
        raise ValueError(
            "plugin factory result must match its registered name and role, "
            "including its workspace kind")
      with self._lock:
        if plugin in self._instances or plugin.model is not None or plugin.device is not None:
          raise ValueError("plugin factory must return a fresh unbound instance")
        self._instances.add(plugin)
      plugin.init(model, batch_size, device)
      instances.append(plugin)
    return tuple(instances)

  def unregister(self, name: str, plugin_type: PluginType | None = None):
    """Remove a plugin from the registry."""
    with self._lock:
      keys = [(str(name), plugin_type)] if plugin_type is not None else [
          key for key in self._registrations if key[0] == str(name)]
      for key in keys:
        self._plugins.pop(key, None)
        self._registrations.pop(key, None)

  def get(self, name: str, plugin_type: PluginType) -> NativePlugin | None:
    """Retrieve a registered plugin by name and type."""
    with self._lock:
      return self._plugins.get((str(name), plugin_type), None)

  def has(self, name: str, plugin_type: PluginType) -> bool:
    """Check if a plugin is registered."""
    with self._lock:
      return (str(name), plugin_type) in self._registrations

  def list_plugins(self) -> list[tuple[str, str]]:
    """List all registered plugins as (name, type_str) tuples."""
    with self._lock:
      return [(k[0], k[1].value) for k in self._registrations]

  def clear(self):
    """Clear all registered plugins."""
    with self._lock:
      self._plugins.clear()
      self._registrations.clear()


# Global default registry instance
default_registry = ExtensionRegistry()


# -------------------------------------------------------------------------
# Executable Working Native Plugin Examples (milestone 019 qualification)
# -------------------------------------------------------------------------

class CustomMagneticForcePlugin(NativePlugin):
  """Apply ``charge * (COM velocity × world magnetic field)`` to bodies.

  Forces act at inertial centers of mass and are projected through each
  body's translational Jacobian. ``body=None`` applies the configured charge
  to every non-world body; an integer ID or name selects one body. Generalized
  joint velocities are never interpreted as Cartesian velocity.
  """

  spatial_force_workspace_kind = "magnetic"

  def __init__(self, name: str = "magnetic_force", charge: float = 1.0,
               b_field=(0.0, 0.0, 1.0), body=None):
    super().__init__(name, PluginType.FORCE)
    self.charge = float(charge)
    self.b_field_init = np.asarray(b_field, dtype=np.float32).copy()
    with np.errstate(over="ignore", invalid="ignore"):
      charge32 = np.float32(self.charge)
    if (not np.isfinite(self.charge) or not np.isfinite(charge32)
        or self.b_field_init.shape != (3,)
        or not np.all(np.isfinite(self.b_field_init))):
      raise ValueError(
          "charge and three-component magnetic field must be finite float32 values")
    if body is not None and (isinstance(body, bool)
        or not isinstance(body, (numbers.Integral, str))):
      raise TypeError("body must be an integer ID, name, or None")
    self.body = body
    self._b_field_dev = None
    self._queries = None

  def _bodies(self, model):
    if self.body is None:
      return tuple(range(1, int(model.nbody)))
    body = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body)
            if isinstance(self.body, str) else int(self.body))
    if not 0 <= body < int(model.nbody):
      raise ValueError("selected magnetic body is outside the compiled model")
    return (body,)

  def init(self, model: mujoco.MjModel, batch_size: int = 1, device=None):
    if torch is None:
      raise RuntimeError("native magnetic force requires the Torch runtime")
    bodies = self._bodies(model)
    super().init(model, batch_size, device)
    self.device = torch.device(self.device)
    from mujoco_metal.spatial_queries import DeviceSpatialQueries
    self._body_ids = bodies
    self._queries = DeviceSpatialQueries(model, self.batch_size, self.device)
    # Keep static body maps on the same device as stage inputs. Pad an empty
    # map with a sentinel because Metal requires a concrete buffer binding.
    body_map = np.asarray(bodies if bodies else (-1,), dtype=np.int32)
    self._body_ids_device = torch.tensor(
        body_map.copy(), dtype=torch.int32, device=self.device)
    self._b_field_dev = torch.tensor(
        self.b_field_init, dtype=torch.float32, device=self.device)

  def run_device(self, state, *, dynamics=None, compute_mask=None, **kwargs):
    """Return generalized force from the supplied native VEL-stage record.

    Initialization may prepare topology on the host; this method performs
    only device operations. A complete smooth stage is required, including
    at RK4 sub-stages and after a sleeping world's velocity is refreshed.
    """
    if self._queries is None:
      raise RuntimeError("magnetic plugin must be initialized before execution")
    if not isinstance(dynamics, dict):
      raise ValueError("magnetic force requires a smooth dynamics stage")
    if compute_mask is not None and self.device.type == "mps":
      return self._queries.masked_magnetic_force(
          dynamics, body_ids=self._body_ids_device, field=self._b_field_dev,
          charge=self.charge, compute_mask=compute_mask)
    qfrc = torch.zeros((self.batch_size, int(self.model.nv)),
                       dtype=torch.float32, device=self.device)
    field = self._b_field_dev.expand(self.batch_size, -1)
    for body in self._body_ids:
      point = dynamics["poses"]["inertial_pos"][:, body]
      velocity = self._queries.object_velocity(
          dynamics, mujoco.mjtObj.mjOBJ_BODY, body)[:, 3:]
      force = self.charge * torch.cross(velocity, field, dim=-1)
      jp, _ = self._queries.jac(dynamics, point, body)
      qfrc.add_(torch.bmm(jp.transpose(-1, -2), force[..., None]).squeeze(-1))
    if compute_mask is not None:
      qfrc = torch.where(compute_mask[:, None], qfrc, 0.0)
    return qfrc

  def run_host(self, data, **kwargs):
    if not default_registry.allow_host_callbacks:
      raise RuntimeError("Host callback execution requested but allow_host_callbacks is False")
    model = data.model
    for body in self._bodies(model):
      velocity = np.empty(6, dtype=np.float64)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                              body, velocity, 0)
      force = self.charge * np.cross(velocity[3:], self.b_field_init)
      mujoco.mj_applyFT(model, data, force, np.zeros(3), data.xipos[body],
                        body, data.qfrc_applied)


class CustomUserSensorPlugin(NativePlugin):
  """Native GPU plugin evaluating custom sensor metrics (e.g. kinetic energy or tracking)."""

  def __init__(self, name: str = "custom_user_sensor", dim: int = 2):
    super().__init__(name, PluginType.SENSOR)
    self.dim = int(dim)

  def run_device(self, state, sensordata=None, sensor_adr=0, **kwargs):
    """Compute sensor output directly on MPS device memory."""
    if torch is None or sensordata is None:
      return
    b = self.batch_size
    qpos = state.qpos
    qvel = state.qvel
    # Sensor channel 0: position norm; channel 1: velocity norm
    p_norm = torch.norm(qpos, dim=1)
    v_norm = torch.norm(qvel, dim=1)
    if self.dim >= 1:
      sensordata[:, sensor_adr] = p_norm
    if self.dim >= 2:
      sensordata[:, sensor_adr + 1] = v_norm
