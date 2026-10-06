# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native lowering/evaluation for MuJoCo 3.10's bundled touch-grid sensor.

The runtime consumes only the retained contact-force stage and current site
poses.  It never asks MuJoCo to evaluate a plugin callback and it performs no
host readback or contact-dependent host branching.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SHADER = Path(__file__).parent / "shaders" / "bundled_touch_grid.metal"


@dataclass(frozen=True)
class TouchGridSensor:
  sensor_id: int
  instance_id: int
  site_id: int
  address: int
  dimension: int
  nchannel: int
  size_x: int
  size_y: int
  fov_x: float
  fov_y: float
  gamma: float
  parent_body: int
  parent_weld: int


def lower_touch_grid_sensors(model, bundled_plugins):
  """Validate and lower every bundled touch-grid sensor on the model."""
  import mujoco

  plugin_sensor = int(mujoco.mjtSensor.mjSENS_PLUGIN)
  site_obj = int(mujoco.mjtObj.mjOBJ_SITE)
  types = np.asarray(model.sensor_type, dtype=np.int32).reshape(-1)
  plugin_ids = np.asarray(bundled_plugins.sensor_plugin, dtype=np.int32).reshape(-1)
  objtypes = np.asarray(model.sensor_objtype, dtype=np.int32).reshape(-1)
  objids = np.asarray(model.sensor_objid, dtype=np.int32).reshape(-1)
  dimensions = np.asarray(model.sensor_dim, dtype=np.int32).reshape(-1)
  addresses = np.asarray(model.sensor_adr, dtype=np.int32).reshape(-1)
  weld = np.asarray(model.body_weldid, dtype=np.int32).reshape(-1)
  site_body = np.asarray(model.site_bodyid, dtype=np.int32).reshape(-1)
  result = []
  for sensor, typ in enumerate(types.tolist()):
    if typ != plugin_sensor:
      continue
    instance_id = int(plugin_ids[sensor])
    if instance_id < 0:
      continue
    instance = bundled_plugins.instance(instance_id)
    if instance.name != "mujoco.sensor.touch_grid":
      continue
    if int(objtypes[sensor]) != site_obj:
      raise ValueError("bundled touch_grid sensor must be attached to a site")
    site = int(objids[sensor])
    config = instance.config
    channels = int(float(config["nchannel"]))
    size = tuple(int(float(v)) for v in config["size"].replace(",", " ").split())
    fov = tuple(float(v) for v in config["fov"].replace(",", " ").split())
    gamma = float(config["gamma"])
    if channels < 1 or channels > 6 or any(v <= 0 for v in size):
      raise ValueError("touch_grid requires 1..6 channels and positive image sizes")
    if any(not np.isfinite(v) or v <= 0 for v in fov):
      raise ValueError("touch_grid field of view must be finite and positive")
    if not np.isfinite(gamma) or gamma < 0 or gamma > 1:
      raise ValueError("touch_grid gamma must be finite and in [0,1]")
    expected = channels * size[0] * size[1]
    if int(dimensions[sensor]) != expected:
      raise ValueError(
          f"touch_grid sensor {sensor} dimension {dimensions[sensor]} != {expected}")
    parent = int(weld[int(site_body[site])])
    result.append(TouchGridSensor(
        sensor, instance_id, site, int(addresses[sensor]), expected,
        channels, size[0], size[1], fov[0], fov[1], gamma,
        parent, int(weld[parent])))
  return tuple(result)


def touch_grid_workspace_sizes(model, batch_size, contact_capacity,
                               bundled_plugins=None, pair_capacity=None):
  """Return exact element counts for the persistent touch-grid backings.

  Counts are scalar elements, with dtype-specific conversion left to the
  capacity inventory. This helper is host-only and is shared by construction
  preflight and the device program so the sensor metadata cannot silently
  outgrow its budget estimate.
  """
  if bundled_plugins is None:
    from mujoco_metal.bundled_plugins import lower_bundled_plugins
    bundled_plugins = lower_bundled_plugins(model)
  sensors = lower_touch_grid_sensors(model, bundled_plugins)
  if not sensors:
    return {}
  b = int(batch_size)
  c = max(int(contact_capacity), 1)
  npair = (int(model.npair) if pair_capacity is None else int(pair_capacity))
  if npair < 0:
    raise ValueError("touch_grid pair_capacity must be nonnegative")
  n = len(sensors)
  edge_stride = max(max(max(s.size_x, s.size_y) + 1 for s in sensors), 1)
  return {
      "body_weld": max(int(model.nbody), 1),
      "geom_body": max(int(model.ngeom), 1),
      "sensor_meta": n * 8,
      "sensor_params": n * 3,
      "x_edges": n * edge_stride,
      "y_edges": n * edge_stride,
      "dims": 10,
      "sensor_index": n,
      "all_worlds": b,
      "contact_active": b * c,
      "contact_mask_source": b * c,
      "contact_condim": c * 3,
      "contact_mu": c * 5,
      "empty_contact_mask": b * c,
      "empty_contact_frame": b * c * 12,
      "empty_contact_force": b * c * 11,
      "empty_contact_slot_pair": c,
      "empty_contact_pair_geoms": 2 * max(npair, 1),
      "status": b,
  }


def _lower_bound(edges, x):
  lo, hi = 0, len(edges)
  while lo < hi:
    mid = (lo + hi) // 2
    if x <= edges[mid]:
      hi = mid
    else:
      lo = mid + 1
  return lo


def _fovea(value, gamma):
  if gamma == 0:
    return value
  return gamma * value**5 + (1.0 - gamma) * value


def touch_grid_reference(sensor, *, contact_rows, site_pos, site_quat,
                         geom_bodies, body_weldid):
  """Independent NumPy implementation of pinned TouchGrid::Compute.

  Rows are `(valid, geom1, geom2, frame[9], position[3], wrench[6])` where
  frame axes are stored as three consecutive world-space vectors and wrench
  is in that contact frame.  The output excludes the private visualization
  distance matrix, exactly like the plugin's sensordata ABI.
  """
  import math

  frame_size = sensor.size_x * sensor.size_y
  output = np.zeros((sensor.dimension,), dtype=np.float64)
  edges_x = np.asarray([
      _fovea(-1.0 + 2.0 * i / sensor.size_x, sensor.gamma)
      * sensor.fov_x * math.pi / 180.0
      for i in range(sensor.size_x + 1)])
  edges_y = np.asarray([
      _fovea(-1.0 + 2.0 * i / sensor.size_y, sensor.gamma)
      * sensor.fov_y * math.pi / 180.0
      for i in range(sensor.size_y + 1)])
  # Match mj_contactForce's body sign, local->world contact transform, and
  # site-frame transform before spherical binning.
  w, x, y, z = np.asarray(site_quat, dtype=np.float64)
  rotation = np.asarray([
      [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
      [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
      [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
  ])
  accepted = []
  for row in contact_rows:
    valid, geom1, geom2 = bool(row[0]), int(row[1]), int(row[2])
    if not valid:
      continue
    body1, body2 = int(geom_bodies[geom1]), int(geom_bodies[geom2])
    if int(body_weldid[body1]) != sensor.parent_weld and int(body_weldid[body2]) != sensor.parent_weld:
      continue
    axes = np.asarray(row[3:12], dtype=np.float64).reshape(3, 3)
    position = np.asarray(row[12:15], dtype=np.float64)
    wrench = np.asarray(row[15:21], dtype=np.float64)
    world_force = axes.T @ wrench[:3]
    world_torque = axes.T @ wrench[3:]
    if sensor.parent_body < max(body1, body2):
      world_force = -world_force
      world_torque = -world_torque
    local_force = rotation.T @ world_force
    local_torque = rotation.T @ world_torque
    local_wrench = np.concatenate((local_force[[2, 0, 1]],
                                   local_torque[[2, 0, 1]]))
    xyz = rotation.T @ (position - np.asarray(site_pos, dtype=np.float64))
    radius = float(np.linalg.norm(xyz))
    azimuth = math.atan2(float(xyz[0]), -float(xyz[2]))
    elevation = math.atan2(float(xyz[1]), math.sqrt(float(xyz[0]**2 + xyz[2]**2)))
    ix, iy = _lower_bound(edges_x, azimuth), _lower_bound(edges_y, elevation)
    if 0 < ix < len(edges_x) and 0 < iy < len(edges_y):
      accepted.append((ix - 1, iy - 1, local_wrench, radius))
  for ix, iy, wrench, _ in accepted:
    pixel = iy * sensor.size_x + ix
    for channel in range(sensor.nchannel):
      output[channel * frame_size + pixel] += wrench[channel]
  return output


class MetalBundledTouchGrid:
  """Prepared fixed-shape MPS evaluator for bundled touch-grid sensors."""

  def __init__(self, model, bundled_plugins, *, batch_size, device,
               contact_capacity, pair_capacity=None):
    import torch
    from mujoco_metal.stateful_actuation import _INT32_MAX

    self.sensors = lower_touch_grid_sensors(model, bundled_plugins)
    self.batch_size = int(batch_size)
    self.contact_capacity = max(int(contact_capacity), 1)
    self.pair_capacity = (int(model.npair) if pair_capacity is None
                          else int(pair_capacity))
    if self.pair_capacity < 0:
      raise ValueError("touch_grid pair_capacity must be nonnegative")
    self.device = torch.device(device)
    self._torch = torch
    if self.device.type != "mps":
      raise ValueError("MetalBundledTouchGrid requires an MPS device")
    max_dim = max((s.dimension for s in self.sensors), default=1)
    size = self.batch_size * max(max_dim, 1) * max(self.contact_capacity, 1)
    if size > _INT32_MAX:
      raise ValueError("touch_grid indexing exceeds signed int32")
    self.ngeom = int(model.ngeom)
    self.nbody = int(model.nbody)
    self.nsite = int(model.nsite)
    self.nsensordata = int(model.nsensordata)
    self._body_weld = torch.as_tensor(
        np.asarray(model.body_weldid, dtype=np.int32).copy(),
        dtype=torch.int32, device=device)
    self._geom_body = torch.as_tensor(
        np.asarray(model.geom_bodyid, dtype=np.int32).copy(),
        dtype=torch.int32, device=device)
    self._sensor_meta = torch.as_tensor(np.asarray([
        (s.site_id, s.address, s.dimension, s.nchannel, s.size_x, s.size_y,
         s.parent_body, s.parent_weld)
        for s in self.sensors], dtype=np.int32).reshape(-1, 8).copy(),
        dtype=torch.int32, device=device)
    self._sensor_params = torch.as_tensor(np.asarray([
        (s.fov_x, s.fov_y, s.gamma) for s in self.sensors
    ], dtype=np.float32).reshape(-1, 3).copy(), dtype=torch.float32, device=device)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.bundled_touch_grid
    self._dims = torch.tensor([self.batch_size, self.contact_capacity,
                               self.ngeom, self.nbody, len(self.sensors),
                               max((max(s.size_x, s.size_y) + 1
                                    for s in self.sensors), default=1),
                               self.nsite, self.nsensordata,
                               int(model.opt.cone), self.pair_capacity],
                              dtype=torch.int32, device=device)
    edge_stride = int(self._dims[5])
    edges_x = np.zeros((max(len(self.sensors), 1), edge_stride), dtype=np.float32)
    edges_y = np.zeros_like(edges_x)
    for i, sensor in enumerate(self.sensors):
      edges_x[i, :sensor.size_x + 1] = [
          _fovea(-1.0 + 2.0 * k / sensor.size_x, sensor.gamma)
          * sensor.fov_x * np.pi / 180.0
          for k in range(sensor.size_x + 1)]
      edges_y[i, :sensor.size_y + 1] = [
          _fovea(-1.0 + 2.0 * k / sensor.size_y, sensor.gamma)
          * sensor.fov_y * np.pi / 180.0
          for k in range(sensor.size_y + 1)]
    self._edges_x = torch.as_tensor(edges_x, dtype=torch.float32, device=device)
    self._edges_y = torch.as_tensor(edges_y, dtype=torch.float32, device=device)
    self._sensor_index = torch.arange(len(self.sensors), dtype=torch.int32,
                                      device=device)
    self._all_worlds = torch.ones((self.batch_size,), dtype=torch.bool,
                                  device=device)
    self._status = torch.zeros((self.batch_size,), dtype=torch.int32,
                               device=device)
    self._contact_active = torch.zeros(
        (self.batch_size, self.contact_capacity), dtype=torch.int32,
        device=device)
    self._contact_mask_source = torch.zeros(
        (self.batch_size, self.contact_capacity), dtype=torch.float32,
        device=device)
    self._contact_frame = torch.zeros(
        (self.batch_size, self.contact_capacity, 12), dtype=torch.float32,
        device=device)
    self._contact_force = torch.zeros(
        (self.batch_size, self.contact_capacity, 11), dtype=torch.float32,
        device=device)
    self._contact_condim = torch.zeros(
        (self.contact_capacity, 3), dtype=torch.int32, device=device)
    self._contact_mu = torch.zeros(
        (self.contact_capacity, 5), dtype=torch.float32, device=device)
    self._slot_pair = torch.full(
        (self.contact_capacity,), -1, dtype=torch.int32, device=device)
    # Models with a touch-grid sensor may have no collidable pairs (for
    # example, every geom has contype=0). Keep a real one-slot zero input so
    # the output kernel still clears this sensor's stage sample without a
    # host-side contact-count branch.
    self._empty_contact = {
        "contact_mask": torch.zeros(
            (self.batch_size, self.contact_capacity), dtype=torch.float32,
            device=device),
        "frame": self._contact_frame,
        "force": self._contact_force,
        "contact_condim": self._contact_condim,
        "mu": self._contact_mu,
        "slot_pair": self._slot_pair,
        "pair_geoms": torch.zeros(
            (2 * max(self.pair_capacity, 1),), dtype=torch.int32,
            device=device),
    }

  @property
  def has_sensors(self):
    return bool(self.sensors)

  def run_device(self, contact, poses, sensordata, *, compute_mask=None,
                 compute_masks=None):
    """Write plugin sensor outputs into the borrowed sensordata tensor."""
    torch = self._torch
    if not self.sensors:
      return sensordata
    b, nc = self.batch_size, self.contact_capacity
    def on_device(value):
      return (isinstance(value, torch.Tensor)
              and value.device.type == self.device.type
              and (self.device.index is None
                   or value.device.index == self.device.index))
    if contact is None:
      contact = self._empty_contact
    # The solver owns the exact compiled slot count, which can be smaller
    # than the touch-grid program's conservative fixed contact capacity.
    # Validate each source shape, then zero-pad it into our owned input views.
    contact_mask = contact.get("contact_mask")
    frame = contact.get("frame")
    force = contact.get("force")
    slots = contact.get("slot_pair")
    condim = contact.get("contact_condim")
    mu = contact.get("mu")
    if (not isinstance(contact_mask, torch.Tensor) or contact_mask.ndim != 2
        or contact_mask.shape[0] != b or contact_mask.shape[1] > nc
        or contact_mask.dtype != torch.float32
        or not contact_mask.is_contiguous() or not on_device(contact_mask)):
      raise ValueError(
          "touch_grid contact_mask must be contiguous float32 MPS [B,C]")
    source_contacts = int(contact_mask.shape[1])
    for name, value, tail in (
        ("frame", frame, (12,)), ("force", force, (11,))):
      if (not isinstance(value, torch.Tensor)
          or tuple(value.shape) != (b, source_contacts) + tail
          or value.dtype != torch.float32 or not value.is_contiguous()
        or not on_device(value)):
        raise ValueError(
            f"touch_grid {name} must be contiguous float32 with its contact shape")
    if (not isinstance(condim, torch.Tensor)
        or tuple(condim.shape) != (source_contacts, 3)
        or condim.dtype != torch.int32 or not condim.is_contiguous()
        or not on_device(condim)):
      raise ValueError("touch_grid contact_condim must be contiguous int32 [C,3]")
    if (not isinstance(mu, torch.Tensor)
        or tuple(mu.shape) != (source_contacts, 5)
        or mu.dtype != torch.float32 or not mu.is_contiguous()
        or not on_device(mu)):
      raise ValueError("touch_grid friction must be contiguous float32 [C,5]")
    if (not isinstance(slots, torch.Tensor)
        or tuple(slots.shape) != (source_contacts,)
        or slots.dtype != torch.int32 or not slots.is_contiguous()
        or not on_device(slots)):
      raise ValueError("touch_grid slot_pair must be contiguous int32 [C]")
    # pair_geoms is passed separately by the current coupled descriptor.  Its
    # exact [npair,2] shape is model-static and checked before dispatch.
    pair_geoms = contact.get("pair_geoms")
    if (not isinstance(pair_geoms, torch.Tensor)
        or pair_geoms.dtype != torch.int32 or pair_geoms.ndim != 1
        or pair_geoms.numel() % 2 or not pair_geoms.is_contiguous()
        or pair_geoms.numel() < 2 * self.pair_capacity
        or not on_device(pair_geoms)):
      raise ValueError("touch_grid pair_geoms must be contiguous int32 [2*npair]")
    if (not isinstance(poses, dict)
        or not on_device(poses.get("site_pos"))
        or not on_device(poses.get("site_quat"))
        or tuple(poses["site_pos"].shape) != (b, self.nsite, 3)
        or tuple(poses["site_quat"].shape) != (b, self.nsite, 4)
        or poses["site_pos"].dtype != torch.float32
        or poses["site_quat"].dtype != torch.float32
        or not poses["site_pos"].is_contiguous()
        or not poses["site_quat"].is_contiguous()):
      raise ValueError("touch_grid site poses have invalid shapes")
    if (not isinstance(sensordata, torch.Tensor)
        or sensordata.dtype != torch.float32
        or tuple(sensordata.shape) != (b, self.nsensordata)
        or not sensordata.is_contiguous() or not on_device(sensordata)):
      raise ValueError("touch_grid sensordata must be contiguous MPS float32 [B,D]")
    if compute_mask is None:
      compute_mask = self._all_worlds
    if compute_masks is None:
      if (not isinstance(compute_mask, torch.Tensor)
          or tuple(compute_mask.shape) != (b,)
          or compute_mask.dtype != torch.bool
          or not on_device(compute_mask)
          or not compute_mask.is_contiguous()):
        raise ValueError("touch_grid compute_mask must be contiguous MPS bool [B]")
      compute_masks = (compute_mask,) * len(self.sensors)
    elif (compute_mask is not None and compute_mask is not self._all_worlds):
      raise ValueError("pass either compute_mask or compute_masks, not both")
    if len(compute_masks) != len(self.sensors):
      raise ValueError("touch_grid compute_masks must have one bool[B] per sensor")
    for mask in compute_masks:
      if (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (b,)
          or mask.dtype != torch.bool or not on_device(mask)
          or not mask.is_contiguous()):
        raise ValueError("touch_grid compute_masks must contain contiguous MPS bool[B]")
    self._contact_active.zero_()
    self._contact_frame.zero_()
    self._contact_force.zero_()
    self._contact_condim.zero_()
    self._contact_mu.zero_()
    self._slot_pair.fill_(-1)
    self._status.zero_()
    if source_contacts:
      self._contact_active[:, :source_contacts].copy_(contact_mask)
      self._contact_frame[:, :source_contacts].copy_(frame)
      self._contact_force[:, :source_contacts].copy_(force)
      self._contact_condim[:source_contacts].copy_(condim)
      self._contact_mu[:source_contacts].copy_(mu)
      self._slot_pair[:source_contacts].copy_(slots)
    for si, sensor in enumerate(self.sensors):
      self._kernel(
          self._contact_active.reshape(-1), self._contact_frame.reshape(-1),
          self._contact_force.reshape(-1), self._slot_pair.reshape(-1),
          pair_geoms.reshape(-1),
          self._geom_body, self._body_weld, poses["site_pos"].reshape(-1),
          poses["site_quat"].reshape(-1), self._sensor_meta.reshape(-1),
          self._sensor_params.reshape(-1), self._edges_x.reshape(-1),
          self._edges_y.reshape(-1), compute_masks[si],
          sensordata.reshape(-1), self._dims,
          self._sensor_index[si:si + 1],
          self._contact_condim.reshape(-1), self._contact_mu.reshape(-1),
          self._status,
          threads=(b * sensor.dimension,), group_size=(min(128, b * sensor.dimension),))
    return sensordata
