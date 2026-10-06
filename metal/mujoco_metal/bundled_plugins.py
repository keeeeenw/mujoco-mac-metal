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

"""Host-only lowering for the MuJoCo 3.10.0 bundled physics plugins.

The compiled ``MjModel`` stores a plugin *slot* and a packed sequence of
configuration strings.  It does not expose the registered plugin name or the
attribute names through Python.  The small ctypes prefix below reads the
documented ``mjpPlugin`` prefix from the already loaded MuJoCo 3.10 shared
library; no callback is invoked and no physics value is read from simulation
state.  Device programs consume the immutable descriptors returned here.

Only the eight bundled plugin classes in the 019 contract are admitted.  An
unknown or malformed plugin is rejected before device allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from types import MappingProxyType

import mujoco
import numpy as np


_SUPPORTED_DEFAULTS = {
    "mujoco.pid": {
        "kp": "0", "ki": "0", "kd": "0", "imax": "", "slewmax": "",
    },
    "mujoco.elasticity.cable": {
        "twist": "", "bend": "", "flat": "", "vmax": "0",
    },
    "mujoco.sensor.touch_grid": {
        "nchannel": "1", "size": "", "fov": "", "gamma": "0",
    },
    "mujoco.sdf.bolt": {"radius": "0.26"},
    "mujoco.sdf.bowl": {"height": "0.4", "radius": "1", "thickness": "0.02"},
    "mujoco.sdf.gear": {
        "alpha": "0", "diameter": "2.8", "teeth": "25",
        "thickness": "0.2", "innerdiameter": "-1",
    },
    "mujoco.sdf.nut": {"radius": "0.26"},
    "mujoco.sdf.torus": {"radius1": "0.35", "radius2": "0.15"},
}
_SDF_NAMES = frozenset(name for name in _SUPPORTED_DEFAULTS if name.startswith("mujoco.sdf."))
_SDF_KIND = {
    "mujoco.sdf.bolt": 1,
    "mujoco.sdf.bowl": 2,
    "mujoco.sdf.gear": 3,
    "mujoco.sdf.nut": 4,
    "mujoco.sdf.torus": 5,
}
_MJP_PREFIX = None
_MJP_LIB = None
_MJP_GET = None


@dataclass(frozen=True)
class BundledPluginInstance:
  """Immutable model-time view of one upstream plugin instance."""

  instance_id: int
  slot: int
  name: str
  attribute_names: tuple[str, ...]
  raw_attributes: tuple[str, ...]
  config: object

  def get(self, key: str, default=None):
    return self.config.get(key, default)


@dataclass(frozen=True)
class BundledPluginModel:
  """Immutable mapping from model plugin IDs to plugin descriptors."""

  instances: tuple[BundledPluginInstance, ...]
  body_plugin: np.ndarray
  geom_plugin: np.ndarray
  sensor_plugin: np.ndarray
  actuator_plugin: np.ndarray
  instance_kind: np.ndarray
  plugin_attributes: np.ndarray
  plugin_attributes_low: np.ndarray
  plugin_attributes_tail: np.ndarray
  geom_plugin_instance: np.ndarray
  geom_sdf_aabb: np.ndarray
  sdf_geom_id: np.ndarray
  sdf_geom_instance: np.ndarray
  sdf_geom_attributes: np.ndarray
  sdf_geom_attributes_low: np.ndarray
  sdf_geom_attributes_tail: np.ndarray
  sdf_geom_aabb: np.ndarray

  def __post_init__(self):
    for name, dtype in (
        ("body_plugin", np.int32), ("geom_plugin", np.int32),
        ("sensor_plugin", np.int32), ("actuator_plugin", np.int32),
        ("instance_kind", np.int32), ("plugin_attributes", np.float32),
        ("plugin_attributes_low", np.float32),
        ("plugin_attributes_tail", np.float32),
        ("geom_plugin_instance", np.int32), ("geom_sdf_aabb", np.float32),
        ("sdf_geom_id", np.int32), ("sdf_geom_instance", np.int32),
        ("sdf_geom_attributes", np.float32), ("sdf_geom_attributes_low", np.float32),
        ("sdf_geom_attributes_tail", np.float32),
        ("sdf_geom_aabb", np.float32),
    ):
      value = np.asarray(getattr(self, name), dtype=dtype, order="C")
      object.__setattr__(self, name,
                         np.frombuffer(value.tobytes(), dtype=dtype).reshape(value.shape))
    nplugin = len(self.instances)
    expected = {
        "body_plugin": (self.body_plugin.shape[0],),
        "geom_plugin": (self.geom_plugin.shape[0],),
        "sensor_plugin": (self.sensor_plugin.shape[0],),
        "actuator_plugin": (self.actuator_plugin.shape[0],),
        "instance_kind": (nplugin,),
        "plugin_attributes": (nplugin, 5),
        "plugin_attributes_low": (nplugin, 5),
        "plugin_attributes_tail": (nplugin, 5),
        "geom_plugin_instance": (self.geom_plugin.shape[0],),
        "geom_sdf_aabb": (self.geom_plugin.shape[0], 6),
        "sdf_geom_id": (self.sdf_geom_id.shape[0],),
        "sdf_geom_instance": (self.sdf_geom_id.shape[0],),
        "sdf_geom_attributes": (self.sdf_geom_id.shape[0], 5),
        "sdf_geom_attributes_low": (self.sdf_geom_id.shape[0], 5),
        "sdf_geom_attributes_tail": (self.sdf_geom_id.shape[0], 5),
        "sdf_geom_aabb": (self.sdf_geom_id.shape[0], 6),
    }
    for name, shape in expected.items():
      if getattr(self, name).shape != shape:
        raise BundledPluginError(f"lowered field {name} has shape {getattr(self, name).shape}, expected {shape}")
    for name in ("body_plugin", "geom_plugin", "sensor_plugin", "actuator_plugin",
                 "geom_plugin_instance", "sdf_geom_instance"):
      values = getattr(self, name)
      if np.any(values < -1) or np.any(values >= nplugin):
        raise BundledPluginError(f"lowered field {name} contains an invalid plugin id")
    if not all(np.all(np.isfinite(getattr(self, name))) for name in (
        "plugin_attributes", "plugin_attributes_low", "plugin_attributes_tail",
        "geom_sdf_aabb", "sdf_geom_attributes", "sdf_geom_attributes_low",
        "sdf_geom_attributes_tail", "sdf_geom_aabb")):
      raise BundledPluginError("lowered plugin floating-point metadata must be finite")

  def instance(self, instance_id: int) -> BundledPluginInstance:
    if instance_id < 0 or instance_id >= len(self.instances):
      raise ValueError(f"plugin instance id {instance_id} is out of range")
    return self.instances[instance_id]


class BundledPluginError(ValueError):
  """A compiled plugin model cannot be safely lowered to the native backend."""


def _mjp_registry():
  """Return MuJoCo's slot accessor using only the stable mjpPlugin prefix."""
  global _MJP_PREFIX, _MJP_LIB, _MJP_GET
  if _MJP_GET is not None:
    return _MJP_GET
  if getattr(mujoco, "__version__", None) != "3.10.0":
    raise BundledPluginError("bundled plugin lowering requires MuJoCo 3.10.0")
  try:
    import mujoco._structs as structs
    _MJP_LIB = ctypes.CDLL(structs.__file__)
    _MJP_PREFIX = type("_MjpPluginPrefix", (ctypes.Structure,), {
        "_fields_": [
            ("name", ctypes.c_char_p),
            ("nattribute", ctypes.c_int),
            ("attributes", ctypes.POINTER(ctypes.c_char_p)),
        ],
    })
    getter = _MJP_LIB.mjp_getPluginAtSlot
    getter.argtypes = [ctypes.c_int]
    getter.restype = ctypes.POINTER(_MJP_PREFIX)
    _MJP_GET = getter
  except (AttributeError, OSError) as exc:
    raise BundledPluginError(
        "MuJoCo's registered plugin descriptor API is unavailable") from exc
  return _MJP_GET


def _plugin_registration(slot: int) -> tuple[str, tuple[str, ...]]:
  getter = _mjp_registry()
  descriptor = getter(int(slot))
  if not descriptor:
    raise BundledPluginError(f"MuJoCo plugin slot {slot} is not registered")
  value = descriptor.contents
  name = value.name.decode("utf-8") if value.name else ""
  names = tuple(value.attributes[i].decode("utf-8")
                for i in range(value.nattribute))
  if name not in _SUPPORTED_DEFAULTS:
    raise BundledPluginError(f"unsupported MuJoCo plugin class {name!r}")
  if names != tuple(_SUPPORTED_DEFAULTS[name]):
    raise BundledPluginError(
        f"MuJoCo plugin {name!r} attribute ABI differs from pinned 3.10.0")
  return name, names


def _plugin_attr_bytes(model, instance: int, instance_count: int) -> bytes:
  attr = np.asarray(model.plugin_attr, dtype=np.int8)
  adr = np.asarray(model.plugin_attradr, dtype=np.int64)
  start = int(adr[instance])
  ends = [int(value) for value in adr[instance + 1:] if int(value) > start]
  end = min(ends) if ends else int(attr.size)
  if start < 0 or end < start or end > int(model.npluginattr):
    raise BundledPluginError("plugin attribute address is outside its backing")
  raw = attr[start:end].tobytes()
  # A zero-attribute plugin has a zero-length range.  Otherwise the final
  # terminator is part of the public packed-string ABI.
  if raw and not raw.endswith(b"\0"):
    raise BundledPluginError("plugin configuration is not NUL terminated")
  return raw


def _parse_config(name: str, attr_names: tuple[str, ...], raw: bytes):
  defaults = _SUPPORTED_DEFAULTS[name]
  if len(attr_names) != len(defaults):
    raise BundledPluginError(f"plugin {name!r} has an incomplete attribute table")
  fields = raw.split(b"\0") if raw else []
  if fields and fields[-1] == b"":
    fields.pop()
  if len(fields) != len(attr_names):
    raise BundledPluginError(
        f"plugin {name!r} has {len(fields)} packed attributes; expected {len(attr_names)}")
  result = {}
  raw_values = []
  for key, value in zip(attr_names, fields):
    text = value.decode("utf-8")
    raw_values.append(text)
    result[key] = text if text else defaults[key]
  def scalar(key, *, optional=False):
    text = result[key]
    if optional and not text:
      return None
    try:
      value = float(text)
    except (TypeError, ValueError) as exc:
      raise BundledPluginError(f"plugin {name!r} attribute {key!r} must be numeric") from exc
    if not np.isfinite(value):
      raise BundledPluginError(f"plugin {name!r} attribute {key!r} must be finite")
    return value
  if name == "mujoco.pid":
    for key in ("kp", "ki", "kd"):
      scalar(key)
    imax, slewmax = scalar("imax", optional=True), scalar("slewmax", optional=True)
    if imax is not None and imax < 0:
      raise BundledPluginError("PID imax must be nonnegative")
    if slewmax is not None and slewmax < 0:
      raise BundledPluginError("PID slewmax must be nonnegative")
  elif name == "mujoco.elasticity.cable":
    for key in ("twist", "bend", "vmax"):
      scalar(key)
    if result["flat"] not in ("true", "false", ""):
      raise BundledPluginError("cable flat must be 'true' or 'false'")
  elif name == "mujoco.sensor.touch_grid":
    nchannel, gamma = scalar("nchannel"), scalar("gamma")
    if not nchannel.is_integer() or not 1 <= nchannel <= 6:
      raise BundledPluginError("touch_grid nchannel must be an integer in [1, 6]")
    if not 0 <= gamma <= 1:
      raise BundledPluginError("touch_grid gamma must be in [0, 1]")
    for key in ("size", "fov"):
      try:
        parts = [float(v) for v in result[key].replace(",", " ").split()]
      except ValueError as exc:
        raise BundledPluginError(f"touch_grid {key} must contain two numbers") from exc
      if len(parts) != 2 or not np.all(np.isfinite(parts)):
        raise BundledPluginError(f"touch_grid {key} must contain two finite numbers")
      if key == "size" and any(int(v) != v or v <= 0 for v in parts):
        raise BundledPluginError("touch_grid size values must be positive integers")
      if key == "fov" and not (0 < parts[0] <= 180 and 0 < parts[1] <= 90):
        raise BundledPluginError("touch_grid fov must satisfy pinned angle limits")
  return tuple(raw_values), MappingProxyType(result)


def _sdf_attributes(name: str, config) -> np.ndarray:
  keys = tuple(_SUPPORTED_DEFAULTS[name])
  attrs = np.asarray([float(config[key]) for key in keys], dtype=np.float64)
  if attrs.size > 5:
    raise BundledPluginError("pinned SDF attribute ABI exceeds 5 values")
  if not np.all(np.isfinite(attrs)):
    raise BundledPluginError(f"SDF plugin {name!r} attributes must be finite")
  out = np.zeros((5,), dtype=np.float32)
  out[:attrs.size] = attrs.astype(np.float32)
  if not np.all(np.isfinite(out)):
    raise BundledPluginError("SDF attributes exceed float32 device range")
  return out


def _sdf_attribute_pair(name: str, config):
  """Split exact parsed attributes into three nonoverlapping float32 words."""
  keys = tuple(_SUPPORTED_DEFAULTS[name])
  exact = np.asarray([float(config[key]) for key in keys], dtype=np.float64)
  high = _sdf_attributes(name, config)
  low = np.zeros((5,), dtype=np.float32)
  low[:len(keys)] = (exact - high[:len(keys)].astype(np.float64)).astype(np.float32)
  tail = np.zeros((5,), dtype=np.float32)
  tail[:len(keys)] = (exact - high[:len(keys)].astype(np.float64)
                      - low[:len(keys)].astype(np.float64)).astype(np.float32)
  if not np.all(np.isfinite(low)) or not np.all(np.isfinite(tail)):
    raise BundledPluginError("SDF residual attributes exceed float32 device range")
  return high, low, tail


def _sdf_aabb(name: str, attr: np.ndarray) -> np.ndarray:
  # Pinned plugin callbacks return [center xyz, half extent xyz].
  if name in ("mujoco.sdf.bolt", "mujoco.sdf.nut"):
    return np.asarray([0, 0, 0, .6, .6, 1], dtype=np.float32)
  if name == "mujoco.sdf.bowl":
    extent = float(attr[1] + attr[2])
    return np.asarray([0, 0, 0, extent, extent, extent], dtype=np.float32)
  if name == "mujoco.sdf.gear":
    return np.asarray([0, 0, 0, float(attr[1]) * .625,
                       float(attr[1]) * .625, float(attr[3]) * .55], dtype=np.float32)
  return np.asarray([0, 0, 0, float(attr[0] + attr[1]),
                     float(attr[0] + attr[1]), float(attr[1])], dtype=np.float32)


def lower_bundled_plugins(model) -> BundledPluginModel:
  """Lower plugin class/config and attached-object maps from pinned MjModel.

  This is a host-only constructor path.  It does not query MjData, execute a
  plugin callback, make numerical decisions from device results, or allocate
  a tensor.  Unknown plugins fail closed before native workspaces are built.
  """
  if getattr(mujoco, "__version__", None) != "3.10.0":
    raise BundledPluginError("native bundled plugins target MuJoCo 3.10.0")
  count = int(model.nplugin)
  slots = np.asarray(model.plugin, dtype=np.int64).reshape(-1)
  if slots.shape != (count,):
    raise BundledPluginError("compiled plugin slot table has an invalid shape")
  attr_adr = np.asarray(model.plugin_attradr, dtype=np.int64).reshape(-1)
  if attr_adr.shape != (count,):
    raise BundledPluginError("compiled plugin attribute offsets have an invalid shape")
  instances = []
  sdf_by_instance = {}
  instance_kind = np.zeros((count,), dtype=np.int32)
  plugin_attributes = np.zeros((count, 5), dtype=np.float32)
  plugin_attributes_low = np.zeros((count, 5), dtype=np.float32)
  plugin_attributes_tail = np.zeros((count, 5), dtype=np.float32)
  for instance, slot in enumerate(slots.tolist()):
    name, keys = _plugin_registration(int(slot))
    raw = _plugin_attr_bytes(model, instance, count)
    values, config = _parse_config(name, keys, raw)
    descriptor = BundledPluginInstance(
        instance, int(slot), name, keys, values, config)
    instances.append(descriptor)
    if name in _SDF_NAMES:
      attributes, attributes_low, attributes_tail = _sdf_attribute_pair(name, config)
      sdf_by_instance[instance] = (name, attributes, attributes_low,
                                   attributes_tail)
      instance_kind[instance] = _SDF_KIND[name]
      plugin_attributes[instance] = attributes
      plugin_attributes_low[instance] = attributes_low
      plugin_attributes_tail[instance] = attributes_tail

  geom_plugin = np.asarray(getattr(model, "geom_plugin", np.full(int(model.ngeom), -1)),
                           dtype=np.int32).reshape(-1)
  if geom_plugin.shape != (int(model.ngeom),):
    raise BundledPluginError("geom_plugin table has an invalid shape")
  sdf_instance_ids = {i for i, instance in enumerate(instances)
                      if instance.name in _SDF_NAMES}
  if sdf_instance_ids:
    sdf_geom_type = int(mujoco.mjtGeom.mjGEOM_SDF)
    for geom, instance in enumerate(geom_plugin.tolist()):
      if instance in sdf_instance_ids and int(model.geom_type[geom]) != sdf_geom_type:
        raise BundledPluginError(
            f"SDF plugin instance {instance} is attached to non-SDF geom {geom}")
    # The native SDF evaluator is a pure geometry callback. It does not
    # implement body/actuator/sensor plugin state or their mutable callbacks.
    for label, values in (
        ("body", getattr(model, "body_plugin", ())),
        ("sensor", getattr(model, "sensor_plugin", ())),
        ("actuator", getattr(model, "actuator_plugin", ())),
    ):
      attached = set(int(v) for v in np.asarray(values).reshape(-1) if int(v) >= 0)
      if attached & sdf_instance_ids:
        raise BundledPluginError(
            f"SDF plugin instances may only attach to SDF geoms (found {label})")
  sdf_ids, sdf_instances, sdf_attrs, sdf_attrs_low, sdf_attrs_tail, sdf_aabbs = (
      [], [], [], [], [], [])
  geom_sdf_aabb = np.zeros((int(model.ngeom), 6), dtype=np.float32)
  for geom, instance in enumerate(geom_plugin.tolist()):
    if instance < 0:
      continue
    if instance not in sdf_by_instance:
      plugin_name = instances[instance].name if instance < len(instances) else "unknown"
      raise BundledPluginError(
          f"geom {geom} uses non-SDF plugin {plugin_name!r}")
    name, attrs, attrs_low, attrs_tail = sdf_by_instance[instance]
    sdf_ids.append(geom)
    sdf_instances.append(instance)
    sdf_attrs.append(attrs)
    sdf_attrs_low.append(attrs_low)
    sdf_attrs_tail.append(attrs_tail)
    aabb = _sdf_aabb(name, attrs)
    sdf_aabbs.append(aabb)
    geom_sdf_aabb[geom] = aabb
  return BundledPluginModel(
      tuple(instances),
      np.asarray(getattr(model, "body_plugin", np.full(int(model.nbody), -1)), dtype=np.int32),
      geom_plugin,
      np.asarray(getattr(model, "sensor_plugin", np.full(int(model.nsensor), -1)), dtype=np.int32),
      np.asarray(getattr(model, "actuator_plugin", np.full(int(model.nu), -1)), dtype=np.int32),
      instance_kind,
      plugin_attributes,
      plugin_attributes_low,
      plugin_attributes_tail,
      geom_plugin.copy(),
      geom_sdf_aabb,
      np.asarray(sdf_ids, dtype=np.int32),
      np.asarray(sdf_instances, dtype=np.int32),
      np.asarray(sdf_attrs, dtype=np.float32).reshape(-1, 5),
      np.asarray(sdf_attrs_low, dtype=np.float32).reshape(-1, 5),
      np.asarray(sdf_attrs_tail, dtype=np.float32).reshape(-1, 5),
      np.asarray(sdf_aabbs, dtype=np.float32).reshape(-1, 6),
  )
