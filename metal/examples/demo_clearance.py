# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Geometry checks for demo trajectories, independent of collision masks."""

import mujoco
import numpy as np


def _aabb(model, data, gid):
  """World AABB half-extents and center for box/sphere/capsule/ellipsoid.

  Returns (center, half) or None for unsupported types (plane handled
  separately, mesh/hfield/cylinder unsupported here).
  """
  gtype = int(model.geom_type[gid])
  center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
  rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
  size = np.asarray(model.geom_size[gid], dtype=np.float64)
  plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
  sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
  capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  ellipsoid = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
  box = int(mujoco.mjtGeom.mjGEOM_BOX)
  if gtype == box:
    half = np.abs(rot) @ size
  elif gtype == sphere:
    half = np.full(3, size[0])
  elif gtype == capsule:
    axis = rot[:, 2] * size[1]
    half = np.abs(axis) + size[0]
  elif gtype == ellipsoid:
    half = np.sqrt((rot * size[None, :]) ** 2 @ np.ones(3))
  elif gtype == plane:
    return None
  else:
    return None
  return center, half


def _slab_gap(model, data, first, second):
  """Sound lower bound on separation for AABB-able pairs, else None.

  AABBs contain their geoms, so AABB separation <= true separation. A
  positive value PROVES separation (immune to the GJK face-degeneracy in
  the class doc). Zero/negative is inconclusive (near or penetrating) and
  the caller falls back to the GJK ensemble. Plane pairs use the signed
  plane distance minus the other AABB extent.
  """
  plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
  types = (int(model.geom_type[first]), int(model.geom_type[second]))
  if plane in types:
    pg = first if types[0] == plane else second
    og = second if types[0] == plane else first
    other = _aabb(model, data, og)
    if other is None:
      return None
    center, half = other
    rot = np.asarray(data.geom_xmat[pg], dtype=np.float64).reshape(3, 3)
    normal = rot[:, 2]
    origin = np.asarray(data.geom_xpos[pg], dtype=np.float64)
    signed = float(normal @ (center - origin)) - float(np.abs(normal) @ half)
    return signed
  a = _aabb(model, data, first)
  b = _aabb(model, data, second)
  if a is None or b is None:
    return None
  (ca, ha), (cb, hb) = a, b
  gaps = np.maximum(cb - hb - (ca + ha), ca - ha - (cb + hb))
  if bool(np.all(gaps <= 0.0)):
    return 0.0 if bool(np.all(gaps == 0.0)) else None
  return float(np.linalg.norm(np.maximum(gaps, 0.0)))


class ClearanceMonitor:
  """Check visible parts even when their collision pair is intentionally omitted.

  Uses separate CPU MjData objects for geometry queries only; this never
  advances or changes the native simulation. Two pair classes:
  - clearance pairs must stay separated (min distance >= -allowed);
  - contact pairs are intentional mechanical joints or functional grasp
    contacts; their worst penetration is recorded for the report instead
    of failing (compliant contact legitimately penetrates ~mm with the
    demo solref settings).

  Robustness: MuJoCo box-box GJK collapses to exactly 0.0 on face-to-face
  parallel-overlap configurations (our axis-aligned furniture/gantry hits
  this across mm-scale basins; witness points come back outside both boxes
  while truly separated by ~0.2 m). Penetration queries still return honest
  negatives (verified: exact from +-0.1 mm gaps through -0.05 m on 20 mm
  overlaps, plus an in-scene -8.4 mm jaw catch during development), so the
  overlap gate tracks the BASE GJK reading only: real overlap reads
  negative and fails, and a false-0 can never mask it. Reported minima take
  max(slab lower bound, ensemble max - bound): the analytic AABB-slab gap
  PROVES far-field separation with no GJK involvement, while the nudge
  ensemble (translations breaking the parallel overlap) recovers near-field
  values; truly touching pairs still report ~0. Known edge: exactly
  coincident volumes report 0.0 instead of a negative depth; demo scenes
  never place distinct bodies exactly coincident.
  """

  # Reporting ensemble (nudge triplets on mocap xyz when present, else on
  # qpos[0]). Bound = largest shift; the ensemble term subtracts it, so it
  # stays conservative next to the slab term.
  NUDGES = ((0.0, 0.0, 0.0), (1e-4, 0.0, 0.0), (-1e-4, 0.0, 0.0),
            (5e-4, 0.0, 0.0), (0.0, 1e-4, 0.0), (0.0, 0.0, 1e-4),
            (0.0, 5e-4, 0.0))

  def __init__(self, model, clearance_pairs, contact_pairs=()):
    self.model = model
    self.data = [mujoco.MjData(model) for _ in self.NUDGES]
    self.clearance = []
    self.contacts = []
    self.minimum = {}
    self.reported = {}
    self.max_pen = {}
    for first, second in clearance_pairs:
      ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
             for name in (first, second)]
      if min(ids) < 0:
        raise ValueError(f"Unknown clearance geometry: {first}, {second}")
      key = f"{first}/{second}"
      self.clearance.append((key, *ids))
      self.minimum[key] = 1.0
      self.reported[key] = 1.0
    for first, second in contact_pairs:
      ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
             for name in (first, second)]
      if min(ids) < 0:
        raise ValueError(f"Unknown contact geometry: {first}, {second}")
      key = f"{first}/{second}"
      self.contacts.append((key, *ids))
      self.max_pen[key] = 0.0

  def _pose(self, slot, qpos, mocap_pos, mocap_quat, nudge):
    d = self.data[slot]
    d.qpos[:] = qpos
    if mocap_pos is not None:
      d.mocap_pos[:] = mocap_pos
    if mocap_quat is not None:
      d.mocap_quat[:] = mocap_quat
    if nudge != (0.0, 0.0, 0.0):
      if self.model.nmocap > 0:
        d.mocap_pos[0, 0] += nudge[0]
        d.mocap_pos[0, 1] += nudge[1]
        d.mocap_pos[0, 2] += nudge[2]
      elif self.model.nq > 0:
        d.qpos[0] += nudge[0]
    mujoco.mj_kinematics(self.model, d)
    return d

  def sample(self, qpos, mocap_pos=None, mocap_quat=None):
    states = [self._pose(k, qpos, mocap_pos, mocap_quat, n)
              for k, n in enumerate(self.NUDGES)]
    bound = max(abs(c) for n in self.NUDGES for c in n)
    for key, first, second in self.clearance:
      reads = [float(mujoco.mj_geomDistance(self.model, d, first, second,
                                            1.0, None))
               for d in states]
      # Gate: base reading only (see class doc).
      self.minimum[key] = min(self.minimum[key], reads[0])
      slab = _slab_gap(self.model, states[0], first, second)
      honest = max(reads) - bound
      if slab is not None and slab > 0.0:
        honest = max(honest, slab)
      self.reported[key] = min(self.reported[key], honest)
    for key, first, second in self.contacts:
      # Base reading: penetration depth comes from the EPA path, verified
      # honest from +-0.1 mm through -0.05 m; the GJK false-0 affects
      # separated pairs only.
      distance = float(mujoco.mj_geomDistance(
          self.model, states[0], first, second, 1.0, None))
      self.max_pen[key] = max(self.max_pen[key], -distance)

  def check(self, allowed_penetration=1e-5):
    for pair, distance in self.minimum.items():
      if distance < -allowed_penetration:
        raise AssertionError(f"Geometry overlap {pair}: {distance:.6f} m")
