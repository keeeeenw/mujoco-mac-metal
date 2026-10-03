# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Geometry checks for demo trajectories, independent of collision masks."""

import mujoco


class ClearanceMonitor:
  """Check visible parts even when their collision pair is intentionally omitted.

  Uses a separate CPU MjData for geometry queries only; this never advances or
  changes the native simulation. Small compliant contact penetration is reported
  explicitly rather than described as exact non-penetration.
  """

  def __init__(self, model, pairs):
    self.model = model
    self.data = mujoco.MjData(model)
    self.pairs = []
    self.minimum = {}
    for first, second in pairs:
      ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
             for name in (first, second)]
      if min(ids) < 0:
        raise ValueError(f"Unknown clearance geometry: {first}, {second}")
      key = f"{first}/{second}"
      self.pairs.append((key, *ids))
      self.minimum[key] = 1.0

  def sample(self, qpos, mocap_pos=None, mocap_quat=None):
    self.data.qpos[:] = qpos
    if mocap_pos is not None:
      self.data.mocap_pos[:] = mocap_pos
    if mocap_quat is not None:
      self.data.mocap_quat[:] = mocap_quat
    mujoco.mj_kinematics(self.model, self.data)
    for key, first, second in self.pairs:
      distance = mujoco.mj_geomDistance(
          self.model, self.data, first, second, 1.0, None)
      self.minimum[key] = min(self.minimum[key], float(distance))

  def check(self, allowed_penetration=1e-5):
    for pair, distance in self.minimum.items():
      if distance < -allowed_penetration:
        raise AssertionError(f"Geometry overlap {pair}: {distance:.6f} m")
