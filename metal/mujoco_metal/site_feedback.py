# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Stateless native Cartesian site-feedback force plugin (milestone 019).

The plugin follows a time-parameterized Lissajous target with a world-frame
Cartesian PD wrench, then projects that wrench through the current smooth
stage's site Jacobian. Model topology is resolved once in :meth:`init`; the
runtime callback consumes only supplied device-stage values.
"""

from __future__ import annotations

import numbers

import mujoco
import numpy as np

from mujoco_metal.extensions import NativePlugin, PluginType, default_registry


def _finite_vector(value, name):
  array = np.asarray(value, dtype=np.float64)
  if array.shape != (3,) or not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must be a finite three-vector")
  with np.errstate(over="ignore", invalid="ignore"):
    converted = array.astype(np.float32)
  if not np.all(np.isfinite(converted)):
    raise ValueError(f"{name} must be representable as finite float32 values")
  return converted


class SiteFeedbackPlugin(NativePlugin):
  """World-space PD feedback at one site with an analytic Lissajous target.

  ``evaluate`` is useful at explicit visualization/query boundaries and
  returns owned device tensors. ``run_device`` is the physics callback: it
  requires the current supplied VEL dynamics dictionary and returns only the
  generalized passive force. It never calls MuJoCo CPU physics or transfers
  runtime values to the host.
  """

  spatial_force_workspace_kind = "site_feedback"

  def __init__(self, name="site_feedback", site="pen_tip", *, kp=1.8, kd=0.65,
               center=(0.72, 0.0, 1.0), amplitude=(0.09, 0.0, 0.07),
               frequency=(1.0, 0.0, 1.5), phase=(0.0, 0.0, 1.1)):
    super().__init__(name, PluginType.FORCE)
    if not isinstance(site, str) or not site:
      raise TypeError("site must be a nonempty compiled site name")
    if (isinstance(kp, bool) or not isinstance(kp, numbers.Real)
        or not np.isfinite(kp) or kp < 0):
      raise ValueError("kp must be finite and nonnegative")
    if (isinstance(kd, bool) or not isinstance(kd, numbers.Real)
        or not np.isfinite(kd) or kd < 0):
      raise ValueError("kd must be finite and nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
      kp32, kd32 = np.float32(kp), np.float32(kd)
    if not np.isfinite(kp32) or not np.isfinite(kd32):
      raise ValueError("kp and kd must be representable as finite float32 values")
    self.site_name = site
    self.kp, self.kd = float(kp), float(kd)
    self.center_init = _finite_vector(center, "center")
    self.amplitude_init = _finite_vector(amplitude, "amplitude")
    self.frequency_init = _finite_vector(frequency, "frequency")
    self.phase_init = _finite_vector(phase, "phase")
    if np.any(self.frequency_init < 0):
      raise ValueError("frequency components must be nonnegative")
    self.site_id = None
    self.body_id = None
    self._queries = None
    self._center = self._amplitude = self._frequency = self._phase = None
    self._kp = self._kd = None

  def init(self, model: mujoco.MjModel, batch_size=1, device=None):
    import torch

    super().init(model, batch_size, device)
    self.device = torch.device(self.device)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                self.site_name)
    if site_id < 0:
      raise ValueError(f"unknown feedback site {self.site_name!r}")
    from mujoco_metal.spatial_queries import DeviceSpatialQueries
    self.site_id = int(site_id)
    self.body_id = int(model.site_bodyid[self.site_id])
    self._queries = DeviceSpatialQueries(model, self.batch_size, self.device)

    def tensor(value):
      return torch.tensor(np.asarray(value).copy(), dtype=torch.float32,
                          device=self.device)

    self._center = tensor(self.center_init)
    self._amplitude = tensor(self.amplitude_init)
    self._frequency = tensor(self.frequency_init)
    self._phase = tensor(self.phase_init)
    self._kp = tensor([self.kp])
    self._kd = tensor([self.kd])

  def _evaluate(self, state, dynamics, compute_mask=None):
    import torch

    if self._queries is None:
      raise RuntimeError("site feedback plugin is not initialized")
    if not isinstance(dynamics, dict) or not isinstance(dynamics.get("poses"), dict):
      raise ValueError("site feedback requires the current smooth dynamics stage")
    time = getattr(state, "_time", None)
    if (not isinstance(time, torch.Tensor)
        or time.dtype != torch.float32
        or tuple(time.shape) != (self.batch_size,)
        or time.device.type != self.device.type
        or (self.device.index is not None and time.device.index != self.device.index)):
      raise ValueError("feedback time must be float32[batch] on the plugin device")
    poses = dynamics["poses"]
    if "site_pos" not in poses:
      raise ValueError("smooth dynamics are missing site positions")
    point = poses["site_pos"][:, self.site_id]
    site_velocity = self._queries.object_velocity(
        dynamics, mujoco.mjtObj.mjOBJ_SITE, self.site_id)[:, 3:]
    angle = time[:, None] * self._frequency[None, :] + self._phase[None, :]
    target = self._center[None, :] + self._amplitude[None, :] * torch.sin(angle)
    target_velocity = (self._amplitude[None, :] * self._frequency[None, :]
                       * torch.cos(angle))
    force = self._kp * (target - point) + self._kd * (
        target_velocity - site_velocity)
    zero_torque = torch.zeros_like(force)
    qfrc = self._queries.apply_ft(
        dynamics, force, zero_torque, point, self.body_id)
    if compute_mask is not None:
      if (not isinstance(compute_mask, torch.Tensor)
          or tuple(compute_mask.shape) != (self.batch_size,)
          or compute_mask.dtype != torch.bool
          or compute_mask.device.type != self.device.type
          or (self.device.index is not None
              and compute_mask.device.index != self.device.index)):
        raise ValueError("compute_mask must be bool[batch] on the plugin device")
      qfrc = torch.where(compute_mask[:, None], qfrc,
                         torch.zeros_like(qfrc))
    return point, site_velocity, target, target_velocity, force, qfrc

  def evaluate(self, state, dynamics, *, compute_mask=None):
    """Return owned Cartesian measurements, target, wrench, and qfrc."""
    point, velocity, target, target_velocity, force, qfrc = self._evaluate(
        state, dynamics, compute_mask)
    return {
        "site_position": point.clone(),
        "site_velocity": velocity.clone(),
        "target_position": target.clone(),
        "target_velocity": target_velocity.clone(),
        "force": force.clone(),
        "qfrc": qfrc.clone(),
    }

  def run_device(self, state, *, dynamics=None, compute_mask=None, **kwargs):
    del kwargs
    if dynamics is None:
      raise ValueError("site feedback requires the current supplied VEL dynamics")
    if compute_mask is not None and self.device.type == "mps":
      return self._queries.masked_site_feedback_force(
          dynamics, time=getattr(state, "_time", None), site_id=self.site_id,
          body_id=self.body_id, center=self._center,
          amplitude=self._amplitude, frequency=self._frequency,
          phase=self._phase, kp=self.kp, kd=self.kd,
          compute_mask=compute_mask)
    return self._evaluate(state, dynamics, compute_mask)[-1]


def register_site_feedback(name="site_feedback", *, registry=None, **config):
  """Register a fresh FORCE plugin factory; caller must unregister afterward."""
  registry = default_registry if registry is None else registry
  registry.register_factory(
      name, PluginType.FORCE,
      lambda: SiteFeedbackPlugin(name=name, **config),
      spatial_force_kind="site_feedback")
  return name
