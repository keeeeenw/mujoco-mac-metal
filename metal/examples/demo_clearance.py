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


def _obb_sat_penetration(model, data, first, second):
  """Separating-axis penetration depth for box-box pairs, else None.

  R09 coincidence control: MuJoCo GJK collapses to exactly 0.0 for exactly
  coincident volumes (and near-coincident basins), so a zero base reading
  with intersecting AABBs is ambiguous between touching and deep overlap.
  The 15-axis OBB SAT decides it analytically. Returns the minimum
  penetration depth (> 0 when volumes overlap, 0.0 when merely touching or
  separated) or None for non-box pairs.
  """
  box = int(mujoco.mjtGeom.mjGEOM_BOX)
  if int(model.geom_type[first]) != box or int(model.geom_type[second]) != box:
    return None
  c1 = np.asarray(data.geom_xpos[first], dtype=np.float64)
  c2 = np.asarray(data.geom_xpos[second], dtype=np.float64)
  r1 = np.asarray(data.geom_xmat[first], dtype=np.float64).reshape(3, 3)
  r2 = np.asarray(data.geom_xmat[second], dtype=np.float64).reshape(3, 3)
  h1 = np.asarray(model.geom_size[first], dtype=np.float64)
  h2 = np.asarray(model.geom_size[second], dtype=np.float64)
  t = c2 - c1
  axes = [r1[:, 0], r1[:, 1], r1[:, 2], r2[:, 0], r2[:, 1], r2[:, 2]]
  for i in range(3):
    for j in range(3):
      c = np.cross(r1[:, i], r2[:, j])
      if float(np.dot(c, c)) > 1e-24:
        axes.append(c / float(np.linalg.norm(c)))
  pen = np.inf
  for a in axes:
    ra = float(np.abs(a @ r1) @ h1)
    rb = float(np.abs(a @ r2) @ h2)
    gap = float(abs(a @ t)) - ra - rb
    if gap > 0.0:
      return 0.0
    pen = min(pen, -gap)
  return float(pen)


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
  values; truly touching pairs still report ~0. Exactly coincident volumes
  (GJK reads 0.0 instead of a negative depth) fall back to analytic
  box-box SAT (R09): real overlap records a negative depth and fails, while
  touching/near pairs are unaffected. Demo scenes never place distinct
  bodies exactly coincident, and the fallback is covered by negative
  controls in test_demo_fixes.py.
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
      base = reads[0]
      # R09 coincidence control: an exactly-zero base reading with
      # intersecting AABBs is ambiguous (touching vs collapsed-GJK overlap),
      # so box-box pairs fall back to analytic SAT; real overlap records a
      # negative depth and fails the gate below.
      if abs(base) < 1e-12:
        sat = _obb_sat_penetration(self.model, states[0], first, second)
        if sat is not None and sat > 0.0:
          base = -sat
      self.minimum[key] = min(self.minimum[key], base)
      slab = _slab_gap(self.model, states[0], first, second)
      honest = max(reads) - bound
      if slab is not None and slab > 0.0:
        honest = max(honest, slab)
      self.reported[key] = min(self.reported[key], honest)
    for key, first, second in self.contacts:
      # Base reading: penetration depth comes from the EPA path, verified
      # honest from +-0.1 mm through -0.05 m; the GJK false-0 affects
      # separated pairs only. Exact-zero readings on box-box pairs get the
      # same SAT fallback as clearance pairs (coincident contact volumes).
      distance = float(mujoco.mj_geomDistance(
          self.model, states[0], first, second, 1.0, None))
      if abs(distance) < 1e-12:
        sat = _obb_sat_penetration(self.model, states[0], first, second)
        if sat is not None and sat > 0.0:
          distance = -sat
      self.max_pen[key] = max(self.max_pen[key], -distance)

  def check(self, allowed_penetration=1e-5):
    for pair, distance in self.minimum.items():
      if distance < -allowed_penetration:
        raise AssertionError(f"Geometry overlap {pair}: {distance:.6f} m")
