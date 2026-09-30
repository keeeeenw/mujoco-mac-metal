# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Spatial tendon path lowering and CPU reference for MuJoCo 3.10.0 (milestone 008).

Ports the pinned spatial path from engine/engine_core_smooth.c mj_tendon and
engine/engine_util_misc.c (wrap_circle, wrap_inside, mju_wrap): site-led paths
with pulley divisors, site-to-site segments, and sphere/cylinder geom wrapping
with side-site selection. Fixed joint-led paths stay in tendons.py.

Pinned sources: engine/engine_core_smooth.c mj_tendon (spatial loop),
engine/engine_util_misc.c mju_wrap/wrap_circle/wrap_inside/length_circle.
"""

import math

import mujoco
import numpy as np

_MJMINVAL = 1e-15
_INT32_MAX = (1 << 31) - 1


def _frozen(values, dtype):
  array = np.array(values, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("tendon constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _is_intersect(p0, p1, p2, p3):
  """Port of pinned is_intersect: proper segment intersection test."""
  d = (p1[0] - p0[0]) * (p3[1] - p2[1]) - (p1[1] - p0[1]) * (p3[0] - p2[0])
  if abs(d) < _MJMINVAL:
    return False
  a = ((p3[0] - p2[0]) * (p0[1] - p2[1]) - (p3[1] - p2[1]) * (p0[0] - p2[0])) / d
  b = ((p1[0] - p0[0]) * (p0[1] - p2[1]) - (p1[1] - p0[1]) * (p0[0] - p2[0])) / d
  return 0 < a < 1 and 0 < b < 1


def _length_circle(p0, p1, ind, radius):
  """Port of pinned length_circle: arc length between unit tangent dirs."""
  p0n = [p0[0] / max(_MJMINVAL, math.hypot(*p0)),
         p0[1] / max(_MJMINVAL, math.hypot(*p0))]
  p1n = [p1[0] / max(_MJMINVAL, math.hypot(*p1)),
         p1[1] / max(_MJMINVAL, math.hypot(*p1))]
  dot = max(-1.0, min(1.0, p0n[0] * p1n[0] + p0n[1] * p1n[1]))
  angle = math.acos(dot)
  cross = p0[1] * p1[0] - p0[0] * p1[1]
  if (cross > 0 and ind) or (cross < 0 and not ind):
    angle = 2 * math.pi - angle
  return radius * angle


def wrap_circle(end, side, radius):
  """Port of pinned wrap_circle. Returns (wlen, pnt) or (-1, None)."""
  e = [float(end[0]), float(end[1]), float(end[2]), float(end[3])]
  sqlen0 = e[0] * e[0] + e[1] * e[1]
  sqlen1 = e[2] * e[2] + e[3] * e[3]
  sqrad = radius * radius
  if sqlen0 < sqrad or sqlen1 < sqrad or radius < _MJMINVAL:
    return -1.0, None
  dif = [e[2] - e[0], e[3] - e[1]]
  dd = dif[0] * dif[0] + dif[1] * dif[1]
  if dd < _MJMINVAL:
    return -1.0, None
  a = -(dif[0] * e[0] + dif[1] * e[1]) / dd
  a = max(0.0, min(1.0, a))
  tmp = [a * dif[0] + e[0], a * dif[1] + e[1]]
  if tmp[0] * tmp[0] + tmp[1] * tmp[1] > sqrad and (
      side is None or (side[0] * tmp[0] + side[1] * tmp[1]) >= 0):
    return -1.0, None
  sqrt0 = math.sqrt(max(0.0, sqlen0 - sqrad))
  sqrt1 = math.sqrt(max(0.0, sqlen1 - sqrad))
  sol = [[None, None], [None, None]]
  good = [0.0, 0.0]
  for i in range(2):
    sgn = 1 if i == 0 else -1
    s00 = [(e[0] * sqrad + sgn * radius * e[1] * sqrt0) / sqlen0,
           (e[1] * sqrad - sgn * radius * e[0] * sqrt0) / sqlen0]
    s01 = [(e[2] * sqrad - sgn * radius * e[3] * sqrt1) / sqlen1,
           (e[3] * sqrad + sgn * radius * e[2] * sqrt1) / sqlen1]
    sol[i][0], sol[i][1] = s00, s01
    if side is not None:
      t = [s00[0] + s01[0], s00[1] + s01[1]]
      n = math.hypot(*t)
      if n < _MJMINVAL:
        good[i] = -10000.0
      else:
        good[i] = (t[0] / n) * side[0] + (t[1] / n) * side[1]
    else:
      t = [s00[0] - s01[0], s00[1] - s01[1]]
      good[i] = -(t[0] * t[0] + t[1] * t[1])
    if _is_intersect(e[:2], s00, e[2:], s01):
      good[i] = -10000.0
  i = 0 if good[0] > good[1] else 1
  pnt = [sol[i][0][0], sol[i][0][1], sol[i][1][0], sol[i][1][1]]
  if _is_intersect(e[:2], pnt[:2], e[2:], pnt[2:]):
    return -1.0, None
  return _length_circle(pnt[:2], pnt[2:], i, radius), pnt


def wrap_inside(end, radius):
  """Port of pinned wrap_inside (Newton solve). Returns (wlen, pnt) or (r, None)."""
  e = [float(end[0]), float(end[1]), float(end[2]), float(end[3])]
  maxiter, zinit, tol = 20, 1 - 1e-7, 1e-6
  len0 = math.hypot(e[0], e[1])
  len1 = math.hypot(e[2], e[3])
  dif = [e[2] - e[0], e[3] - e[1]]
  dd = dif[0] * dif[0] + dif[1] * dif[1]
  if len0 <= radius or len1 <= radius or radius < _MJMINVAL or len0 < _MJMINVAL or len1 < _MJMINVAL:
    return -1.0, None
  if dd > _MJMINVAL:
    a = -(dif[0] * e[0] + dif[1] * e[1]) / dd
    if 0 < a < 1:
      tmp = [e[0] + a * dif[0], e[1] + a * dif[1]]
      if math.hypot(*tmp) <= radius:
        return -1.0, None
  pnt = [0.5 * (e[0] + e[2]), 0.5 * (e[1] + e[3])]
  n = math.hypot(*pnt)
  pnt = [pnt[0] / n * radius, pnt[1] / n * radius, pnt[0] / n * radius, pnt[1] / n * radius]
  A, B = radius / len0, radius / len1
  cosG = (len0 * len0 + len1 * len1 - dd) / (2 * len0 * len1)
  if cosG < -1 + _MJMINVAL:
    return -1.0, None
  if cosG > 1 - _MJMINVAL:
    return 0.0, pnt
  G = math.acos(max(-1.0, min(1.0, cosG)))
  z = zinit
  f = math.asin(A * z) + math.asin(B * z) - 2 * math.asin(z) + G
  if f > 0:
    return 0.0, pnt
  it = 0
  while it < maxiter and abs(f) > tol:
    df = (A / max(_MJMINVAL, math.sqrt(max(0.0, 1 - z * z * A * A)))
          + B / max(_MJMINVAL, math.sqrt(max(0.0, 1 - z * z * B * B)))
          - 2 / max(_MJMINVAL, math.sqrt(max(0.0, 1 - z * z))))
    if df > -_MJMINVAL:
      return 0.0, pnt
    z1 = z - f / df
    if z1 > z:
      return 0.0, pnt
    z = z1
    f = math.asin(A * z) + math.asin(B * z) - 2 * math.asin(z) + G
    if f > tol:
      return 0.0, pnt
    it += 1
  if it >= maxiter:
    return 0.0, pnt
  if e[0] * e[3] - e[1] * e[2] > 0:
    vec = [e[0], e[1]]
    ang = math.asin(z) - math.asin(A * z)
  else:
    vec = [e[2], e[3]]
    ang = math.asin(z) - math.asin(B * z)
  n = math.hypot(*vec)
  vec = [vec[0] / n, vec[1] / n]
  px = radius * (math.cos(ang) * vec[0] - math.sin(ang) * vec[1])
  py = radius * (math.sin(ang) * vec[0] + math.cos(ang) * vec[1])
  return 0.0, [px, py, px, py]


def tendon_wrap_point(x0, x1, xpos, xmat, radius, wraptype, side):
  """Port of pinned mju_wrap. Returns (wlen, wp0, wp1) or (-1, None, None)."""
  sphere = int(mujoco.mjtWrap.mjWRAP_SPHERE)
  cylinder = int(mujoco.mjtWrap.mjWRAP_CYLINDER)
  if wraptype not in (sphere, cylinder):
    raise ValueError("wrapping object must be a sphere or cylinder geom")
  x0 = np.asarray(x0, dtype=np.float64)
  x1 = np.asarray(x1, dtype=np.float64)
  xpos = np.asarray(xpos, dtype=np.float64)
  Rm = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
  p = [Rm.T @ (x0 - xpos), Rm.T @ (x1 - xpos)]
  if np.linalg.norm(p[0]) < _MJMINVAL or np.linalg.norm(p[1]) < _MJMINVAL:
    return -1.0, None, None
  if wraptype == sphere:
    axis0 = p[0] / np.linalg.norm(p[0])
    normal = np.cross(p[0], p[1])
    nrm = np.linalg.norm(normal)
    if nrm < _MJMINVAL:
      i = int(np.argmax(np.abs(axis0)))
      axis1 = np.array([1.0, 1.0, 1.0])
      axis1[i] = 0.0
      normal = np.cross(axis0, axis1)
      normal = normal / np.linalg.norm(normal)
    else:
      normal = normal / nrm
    axis1 = np.cross(normal, axis0)
    axis1 = axis1 / np.linalg.norm(axis1)
    axes = [axis0, axis1]
  else:
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
  d = [float(p[0] @ axes[0]), float(p[0] @ axes[1]),
       float(p[1] @ axes[0]), float(p[1] @ axes[1])]
  sd = None
  s = None
  if side is not None:
    s = Rm.T @ (np.asarray(side, dtype=np.float64) - xpos)
    sd = [float(s @ axes[0]), float(s @ axes[1])]
    n = math.hypot(*sd)
    sd = [sd[0] / n * radius, sd[1] / n * radius]
  if side is not None and np.linalg.norm(s) < radius:
    wlen, pnt = wrap_inside(d, radius)
  else:
    wlen, pnt = wrap_circle(d, sd, radius)
  if wlen < 0 or pnt is None:
    return -1.0, None, None
  res = []
  for i in range(2):
    res.extend(axes[0] * pnt[2 * i] + axes[1] * pnt[2 * i + 1])
  res = np.array(res, dtype=np.float64)
  if wraptype == cylinder:
    L0 = math.hypot(p[0][0] - res[0], p[0][1] - res[1])
    L1 = math.hypot(p[1][0] - res[3], p[1][1] - res[4])
    res[2] = p[0][2] + (p[1][2] - p[0][2]) * L0 / (L0 + wlen + L1)
    res[5] = p[0][2] + (p[1][2] - p[0][2]) * (L0 + wlen) / (L0 + wlen + L1)
    height = abs(res[5] - res[2])
    wlen = math.sqrt(wlen * wlen + height * height)
  wp0 = Rm @ res[:3] + xpos
  wp1 = Rm @ res[3:] + xpos
  return wlen, wp0, wp1


class SpatialTendonModel:
  """Immutable spatial tendon path lowering for MuJoCo 3.10.0.

  Admits site-led paths with PULLEY divisors, SITE segments and
  SPHERE/CYLINDER geom wraps with side-site selection. Joint-led (fixed)
  paths stay in tendons.FixedTendonModel. Tendon spring/damper/armature,
  limit, friction and equality parameters are carried for the dynamics and
  constraint stages.
  """

  def __init__(self, model):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"spatial tendon lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    ntendon = int(model.ntendon)
    if ntendon < 0 or ntendon > _INT32_MAX:
      raise ValueError("tendon dimensions exceed int32")
    wrap_joint = int(mujoco.mjtWrap.mjWRAP_JOINT)
    wrap_pulley = int(mujoco.mjtWrap.mjWRAP_PULLEY)
    wrap_site = int(mujoco.mjtWrap.mjWRAP_SITE)
    wrap_sphere = int(mujoco.mjtWrap.mjWRAP_SPHERE)
    wrap_cyl = int(mujoco.mjtWrap.mjWRAP_CYLINDER)
    nsite, ngeom = int(model.nsite), int(model.ngeom)
    self.nq, self.nv, self.nsite, self.ntendon = int(model.nq), int(model.nv), nsite, ntendon
    self.paths = []
    for tendon in range(ntendon):
      start, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
      if count <= 0:
        raise ValueError(f"tendon {tendon}: empty wrap path")
      types = [int(model.wrap_type[w]) for w in range(start, start + count)]
      objids = [int(model.wrap_objid[w]) for w in range(start, start + count)]
      prms = [float(model.wrap_prm[w]) for w in range(start, start + count)]
      if types[0] == wrap_joint:
        self.paths.append(None)  # fixed path: owned by FixedTendonModel
        continue
      if types[0] != wrap_site:
        raise ValueError(f"tendon {tendon}: spatial paths must start with a site")
      if any(t == wrap_joint for t in types):
        raise ValueError(f"tendon {tendon}: joint wraps are unsupported in spatial paths (owned by fixed model)")
      if any(t not in (wrap_site, wrap_pulley, wrap_sphere, wrap_cyl) for t in types):
        raise ValueError(f"tendon {tendon}: unknown wrap type")
      for k, (t, o) in enumerate(zip(types, objids)):
        if t == wrap_site and not (0 <= o < nsite):
          raise ValueError(f"tendon {tendon}: site id out of range")
        if t == wrap_pulley and not (np.isfinite(prms[k]) and prms[k] > 0):
          raise ValueError(f"tendon {tendon}: pulley divisor must be finite and positive")
        if t in (wrap_sphere, wrap_cyl):
          if not (0 <= o < ngeom):
            raise ValueError(f"tendon {tendon}: wrap geom out of range")
          gt = int(model.geom_type[o])
          if t == wrap_sphere and gt != int(mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError(f"tendon {tendon}: sphere wraps require sphere geoms")
          if t == wrap_cyl and gt != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise ValueError(f"tendon {tendon}: cylinder wraps require cylinder geoms")
          sideid = int(round(prms[k]))
          if sideid != -1 and not (0 <= sideid < nsite):
            raise ValueError(f"tendon {tendon}: side site out of range")
      self.paths.append({"start": start, "count": count, "types": types,
                         "objids": objids, "prms": prms})
    self.has_spatial = any(p is not None for p in self.paths)
    # Dynamics/constraint parameters (all tendons, fixed or spatial).
    self.stiffness = _frozen(model.tendon_stiffness, np.float32)
    self.stiffnesspoly = _frozen(np.asarray(model.tendon_stiffnesspoly).reshape(ntendon, 2)
                                 if ntendon else np.zeros((0, 2)), np.float32)
    self.damping = _frozen(model.tendon_damping, np.float32)
    self.dampingpoly = _frozen(np.asarray(model.tendon_dampingpoly).reshape(ntendon, 2)
                               if ntendon else np.zeros((0, 2)), np.float32)
    self.spring_range = _frozen(np.asarray(model.tendon_lengthspring).reshape(ntendon, 2)
                                if ntendon else np.zeros((0, 2)), np.float32)
    self.armature = _frozen(model.tendon_armature, np.float32)
    if np.any(np.asarray(self.armature) < 0) or np.any(np.asarray(self.damping) < 0):
      raise ValueError("tendon armature and damping must be nonnegative")
    self.limited = _frozen(model.tendon_limited, np.int32)
    self.range = _frozen(np.asarray(model.tendon_range).reshape(ntendon, 2)
                         if ntendon else np.zeros((0, 2)), np.float32)
    self.margin = _frozen(model.tendon_margin, np.float32)
    self.solref_limit = _frozen(np.asarray(model.tendon_solref_lim).reshape(ntendon, 2)
                                if ntendon else np.zeros((0, 2)), np.float32)
    self.solimp_limit = _frozen(np.asarray(model.tendon_solimp_lim).reshape(ntendon, 5)
                                if ntendon else np.zeros((0, 5)), np.float32)
    self.frictionloss = _frozen(model.tendon_frictionloss, np.float32)
    self.solref_friction = _frozen(np.asarray(model.tendon_solref_fri).reshape(ntendon, 2)
                                   if ntendon else np.zeros((0, 2)), np.float32)
    self.solimp_friction = _frozen(np.asarray(model.tendon_solimp_fri).reshape(ntendon, 5)
                                   if ntendon else np.zeros((0, 5)), np.float32)
    for name in ("tendon_stiffness", "tendon_damping", "tendon_armature",
                 "tendon_range", "tendon_margin", "tendon_frictionloss"):
      arr = np.asarray(getattr(model, name))
      if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    lim = np.asarray(self.limited)
    rng = np.asarray(self.range, dtype=np.float64)
    if np.any(lim & (rng[:, 0] > rng[:, 1])):
      raise ValueError("limited tendon ranges must be ordered")
    if np.any(np.asarray(self.frictionloss) < 0):
      raise ValueError("tendon friction loss must be nonnegative")


def spatial_length_reference(model, tendon, site_xpos, geom_xpos=None, geom_xmat=None):
  """Independent pinned mj_tendon port: length of one spatial tendon."""
  meta = SpatialTendonModel(model)
  path = meta.paths[tendon]
  if path is None:
    raise ValueError("tendon is a fixed joint path (owned by FixedTendonModel)")
  wrap_site = int(mujoco.mjtWrap.mjWRAP_SITE)
  wrap_pulley = int(mujoco.mjtWrap.mjWRAP_PULLEY)
  wrap_sphere = int(mujoco.mjtWrap.mjWRAP_SPHERE)
  wrap_cyl = int(mujoco.mjtWrap.mjWRAP_CYLINDER)
  types, objids, prms = path["types"], path["objids"], path["prms"]
  count = path["count"]
  length, divisor, j = 0.0, 1.0, 0
  while j < count - 1:
    t0, t1 = types[j], types[j + 1]
    id0, id1 = objids[j], objids[j + 1]
    if t0 == wrap_pulley or t1 == wrap_pulley:
      if t0 == wrap_pulley:
        divisor = prms[j]
      j += 1
      continue
    p0 = np.asarray(site_xpos[id0], dtype=np.float64)
    if t1 in (wrap_sphere, wrap_cyl):
      wraptype, wrapid = t1, id1
      t1 = types[j + 2]
      id1 = objids[j + 2]
      sideid = int(round(prms[j + 1]))
      side = np.asarray(site_xpos[sideid], dtype=np.float64) if sideid >= 0 else None
      gid = wrapid
      wlen, wp0, wp1 = tendon_wrap_point(
          p0, np.asarray(site_xpos[id1], dtype=np.float64),
          np.asarray(geom_xpos[gid], dtype=np.float64),
          np.asarray(geom_xmat[gid], dtype=np.float64).reshape(3, 3),
          float(np.asarray(model.geom_size)[gid, 0]), wraptype, side)
      if wlen < 0:
        length += float(np.linalg.norm(np.asarray(site_xpos[id1]) - p0)) / divisor
      else:
        length += (float(np.linalg.norm(wp0 - p0)) + wlen
                   + float(np.linalg.norm(np.asarray(site_xpos[id1]) - wp1))) / divisor
      j += 2
    else:
      length += float(np.linalg.norm(np.asarray(site_xpos[id1]) - p0)) / divisor
      j += 1
  return length
