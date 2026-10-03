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
    self.device = device or (torch.device("mps") if torch is not None and torch.backends.mps.is_available() else "cpu")

  def run_device(self, state, **kwargs):
    """Execute plugin computation natively on device tensors.
    
    Must operate entirely on MPS tensors without CPU readback.
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

  def snapshot(self):
    """Capture plugin internal state for replay."""
    return None

  def restore(self, snap, env_ids=None):
    """Restore plugin internal state from checkpoint."""
    pass


@dataclass(frozen=True)
class PluginRegistration:
  """Immutable role/name/factory descriptor; never a bound runtime plugin."""

  name: str
  plugin_type: PluginType
  factory: object


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
    with self._lock:
      self._add_registration(plugin.name, plugin.plugin_type, blueprint.create_instance)
      self._plugins[(plugin.name, plugin.plugin_type)] = plugin

  def _add_registration(self, name, plugin_type, factory):
    key = (name, plugin_type)
    if key in self._registrations:
      raise ValueError(f"Plugin {name!r} of type {plugin_type.value!r} already registered")
    self._registrations[key] = PluginRegistration(name, plugin_type, factory)

  def register_factory(self, name, plugin_type, factory):
    """Register a zero-argument factory returning a fresh unbound plugin."""
    if not isinstance(name, str) or not name or not isinstance(plugin_type, PluginType):
      raise ValueError("registration requires a nonempty name and PluginType role")
    if not callable(factory):
      raise TypeError("plugin factory must be callable")
    with self._lock:
      self._add_registration(name, plugin_type, factory)

  def instantiate(self, model, batch_size=1, device=None):
    """Bind fresh instances to one simulation without changing registrations.

    The descriptor list is captured atomically. Registration/unregistration after
    construction affects future simulations only. Failed construction never
    rebinds an existing simulation's instance.
    """
    with self._lock:
      descriptors = tuple(self._registrations.values())
    instances = []
    for desc in descriptors:
      plugin = desc.factory()
      if (not isinstance(plugin, NativePlugin) or plugin.name != desc.name
          or plugin.plugin_type != desc.plugin_type):
        raise ValueError("plugin factory result must match its registered name and role")
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
  """Native GPU plugin applying Lorentz force F = q (v x B) to bodies.
  
  Operates natively on device tensors without host roundtrips.
  """

  def __init__(self, name: str = "magnetic_force", charge: float = 1.0, b_field=(0.0, 0.0, 1.0)):
    super().__init__(name, PluginType.FORCE)
    self.charge = float(charge)
    self.b_field_init = np.asarray(b_field, dtype=np.float32)
    self._b_field_dev = None

  def init(self, model: mujoco.MjModel, batch_size: int = 1, device=None):
    super().init(model, batch_size, device)
    if torch is not None:
      self._b_field_dev = torch.as_tensor(self.b_field_init, dtype=torch.float32, device=self.device)

  def run_device(self, state, poses=None, **kwargs):
    """Compute Lorentz force natively on MPS.
    
    Returns qfrc tensor [batch, nv] to add to external forces.
    """
    if torch is None:
      return None
    b = self.batch_size
    nv = self.model.nv
    qvel = state.qvel
    # If free bodies or 3D joints exist, apply cross(qvel, B) * charge
    # For demonstration, applies to first 3 linear DOFs
    qfrc = torch.zeros((b, nv), dtype=torch.float32, device=self.device)
    if nv >= 3:
      v = qvel[:, :3]
      # v x B
      bx, by, bz = self._b_field_dev[0], self._b_field_dev[1], self._b_field_dev[2]
      fx = v[:, 1] * bz - v[:, 2] * by
      fy = v[:, 2] * bx - v[:, 0] * bz
      fz = v[:, 0] * by - v[:, 1] * bx
      qfrc[:, :3] = self.charge * torch.stack([fx, fy, fz], dim=1)
    return qfrc

  def run_host(self, data, **kwargs):
    if not default_registry.allow_host_callbacks:
      raise RuntimeError("Host callback execution requested but allow_host_callbacks is False")
    qv = np.asarray(data.qvel).reshape(-1)
    v = np.zeros(3, dtype=np.float64)
    v[:min(len(qv), 3)] = qv[:min(len(qv), 3)]
    f = self.charge * np.cross(v, self.b_field_init)
    data.qfrc_applied[:min(len(qv), 3)] += f[:min(len(qv), 3)]


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
