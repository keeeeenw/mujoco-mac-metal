# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""MuJoCo 3.10 fluid forces, lowered to generalized force.

Covers the pinned inertia-box body model and the per-geom ellipsoid model
(added-mass, Magnus/Kutta lift, blunt/slender/angular viscous terms). Bodies
with any interacting ellipsoid geom use the geom model; other bodies use the
inertia-box model, matching pinned `mj_fluid` selection.
"""

from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "fluid.metal"
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


def _frozen(value):
  value = np.array(value, dtype=np.float32, order="C", copy=True)
  if not np.all(np.isfinite(value)):
    raise ValueError("fluid constants must be finite and float32-representable")
  return np.frombuffer(value.tobytes(), dtype=np.float32).reshape(value.shape)


def _skew(v):
  v = np.asarray(v, dtype=np.float64).reshape(3)
  return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _added_mass_jac(lvel, density, vmass, vinertia):
  """6x6 df/du of pinned mj_addedMassForces ([ang; lin] order)."""
  v = np.asarray(lvel[3:6], dtype=np.float64)
  w = np.asarray(lvel[0:3], dtype=np.float64)
  A = np.diag(density * np.asarray(vmass, dtype=np.float64))
  B = np.diag(density * np.asarray(vinertia, dtype=np.float64))
  Av, Bw = A @ v, B @ w
  J = np.zeros((6, 6))
  J[3:6, 3:6] = -_skew(w) @ A
  J[3:6, 0:3] = _skew(Av)
  J[0:3, 3:6] = -_skew(v) @ A + _skew(Av)
  J[0:3, 0:3] = -_skew(w) @ B + _skew(Bw)
  return J


def _viscous_jac(lvel, density, viscosity, size, magnus, kutta,
                 blunt, slender, ang_drag):
  """6x6 df/du of pinned mj_viscousForces ([ang; lin] order, R06d).

  Term-by-term analytic derivative of `_viscous_forces`; quotient branches
  (`num`, `denom`, norms) mirror the force guards exactly. Nonsmooth at
  `lin == 0` / `mom == 0` exactly like the force (FD validation stays at
  nonzero velocities).
  """
  v = np.asarray(lvel[3:6], dtype=np.float64)
  w = np.asarray(lvel[0:3], dtype=np.float64)
  s = np.asarray(size, dtype=np.float64).reshape(3)
  J = np.zeros((6, 6))
  volume = 4.0 / 3.0 * np.pi * s[0] * s[1] * s[2]
  # Magnus lift: m*V*(w x v).
  mV = magnus * density * volume
  J[3:6, 3:6] += mV * _skew(w)
  J[3:6, 0:3] += -mV * _skew(v)
  # Kutta lift scaffolding (shared with drag below).
  c2 = np.array([(s[1] * s[2]) ** 2, (s[2] * s[0]) ** 2, (s[0] * s[1]) ** 2])
  c4 = c2 ** 2
  spd = float(np.linalg.norm(v))
  num = float(c2[0] * v[0] ** 2 + c2[1] * v[1] ** 2 + c2[2] * v[2] ** 2)
  den = float(c4[0] * v[0] ** 2 + c4[1] * v[1] ** 2 + c4[2] * v[2] ** 2)
  ap = float(np.pi * np.sqrt(den / max(mujoco.mjMINVAL, num))) if num > 0 else 0.0
  nvec = c2 * v
  ca = (num / max(mujoco.mjMINVAL, spd * den) if den > 0 and spd > 0 else 0.0)
  dmax = float(np.max(s))
  dmin = float(np.min(s))
  dmid = float(s[0] + s[1] + s[2] - dmax - dmin)
  amax = np.pi * dmax * dmid
  # d(ap)/dv and d(ca)/dv (zero when their guards fail).
  dap = np.zeros(3)
  if num > mujoco.mjMINVAL:
    dnum = 2 * c2 * v
    dden = 2 * c4 * v
    dap = (np.pi * 0.5 / np.sqrt(den / num)
           * (dden * num - den * dnum) / num ** 2)
  dca = np.zeros(3)
  if den > 0 and spd > 0:
    dnum = 2 * c2 * v
    dden = 2 * c4 * v
    m0 = max(mujoco.mjMINVAL, spd * den)
    dca = (dnum * m0 - num * (v / spd * den + spd * dden)) / m0 ** 2
  dnvec = np.diag(c2)
  # circ = (n x v) * k*d*ca*ap; g = circ; f += g x v.
  kdc = kutta * density
  g = np.cross(nvec, v) * (kdc * ca * ap)
  dg = np.zeros((3, 3))
  if kdc != 0.0:
    # d[(n x v)]/dv = -skew(v) diag(c2) + skew(n).
    dnx = -_skew(v) @ dnvec + _skew(nvec)
    dg = (dnx * (kdc * ca * ap)
          + np.outer(np.cross(nvec, v), kdc * (dca * ap + ca * dap)))
  J[3:6, 3:6] += -_skew(v) @ dg + _skew(g)
  # Linear drag: Dl = mu*lc + rho*|v|*(ap*bl + sl*(am-ap)); f -= Dl*v.
  eq_d = 2.0 / 3.0 * float(s[0] + s[1] + s[2])
  Dl = viscosity * 3.0 * np.pi * eq_d + density * spd * (ap * blunt + slender * (amax - ap))
  dDl = np.zeros(3)
  if spd > 0:
    dDl = density * ((ap * blunt + slender * (amax - ap)) * v / spd
                     + spd * (blunt - slender) * dap)
  J[3:6, 3:6] += -(Dl * np.eye(3) + np.outer(v, dDl))
  # Angular drag: Da = mu*tc + rho*|mom|; f -= Da*w,
  # mom = w * (ad*ii + sl*(imax-ii)) elementwise.
  def _max_moment(d):
    d0, d1, d2 = s[d], s[(d + 1) % 3], s[(d + 2) % 3]
    return 8.0 / 15.0 * np.pi * d0 * max(d1, d2) ** 4
  ii = np.array([_max_moment(0), _max_moment(1), _max_moment(2)])
  imax = 8.0 / 15.0 * np.pi * dmid * dmax ** 4
  kk = ang_drag * ii + slender * (imax - ii)
  mom = w * kk
  msp = float(np.linalg.norm(mom))
  Da = viscosity * np.pi * eq_d ** 3 + density * msp
  dDa = np.zeros(3)
  if msp > 0:
    dDa = density * (mom / msp) * kk
  J[0:3, 0:3] += -(Da * np.eye(3) + np.outer(w, dDa))
  return J


def _body_jac6(model, data, body):
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  mujoco.mj_jacBody(model, data, jacp, jacr, int(body))
  return np.vstack([jacr, jacp])


def _geom_jac6(model, data, geom):
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  mujoco.mj_jacGeom(model, data, jacp, jacr, int(geom))
  return np.vstack([jacr, jacp])


def _rot6(rot):
  rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
  out = np.zeros((6, 6))
  out[:3, :3] = rot
  out[3:, 3:] = rot
  return out


def fluid_derivative_reference(model, qpos, qvel):
  """Assemble ``d(qfrc_fluid)/d(qvel)`` in host float64 (R06d).

  Mirrors ``InertiaBoxFluidModel.run`` term-by-term: per-geom ellipsoid
  model (added-mass + Magnus/Kutta + viscous Jacobians above) where a body
  has an interacting ellipsoid geom, otherwise the inertia-box model
  (linear Stokes + per-axis quadratic blunt drag). Wind is a constant
  offset (no Jacobian). Disable-flag and model gating match ``run``.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  nq, nv = int(model.nq), int(model.nv)
  qpos = np.asarray(qpos, dtype=np.float64)
  qvel = np.asarray(qvel, dtype=np.float64)
  if qpos.ndim != 2 or qpos.shape[1] != nq or qvel.shape != (qpos.shape[0], nv):
    raise ValueError(f"qpos/qvel must have shapes (B, {nq}) and (B, {nv})")
  if not qpos.shape[0]:
    raise ValueError("batch must be nonempty")
  if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
    raise ValueError("qpos and qvel must be finite")
  spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  if int(model.opt.disableflags) & spring and int(model.opt.disableflags) & damper:
    return np.zeros((qpos.shape[0], nv, nv))
  density, viscosity = float(model.opt.density), float(model.opt.viscosity)
  if not density and not viscosity:
    return np.zeros((qpos.shape[0], nv, nv))
  wind = np.r_[np.zeros(3), np.asarray(model.opt.wind, dtype=np.float64)]
  gf = np.asarray(model.geom_fluid, dtype=np.float64).reshape(-1, 12)
  out = np.zeros((qpos.shape[0], nv, nv))
  for row in range(qpos.shape[0]):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[row]
    data.qvel[:] = qvel[row]
    mujoco.mj_forward(model, data)
    for body in range(1, int(model.nbody)):
      mass = float(model.body_mass[body])
      if mass < mujoco.mjMINVAL:
        continue
      adr, num = int(model.body_geomadr[body]), int(model.body_geomnum[body])
      use_ellipsoid = bool(np.any(gf[adr:adr + num, 0] > 0)) if num else False
      if use_ellipsoid:
        for j in range(num):
          gid = adr + j
          if float(gf[gid, 0]) == 0.0:
            continue
          size = _geom_semiaxes(np.asarray(model.geom_size[gid]),
                                int(np.asarray(model.geom_type[gid])))
          lvel = np.zeros(6)
          mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, gid, lvel, 1)
          lw = np.zeros(6)
          root = int(model.body_rootid[body])
          mujoco.mju_transformSpatial(
              lw, wind, 0, np.asarray(data.geom_xpos[gid]),
              np.asarray(data.subtree_com[root]),
              np.asarray(data.geom_xmat[gid]).reshape(-1))
          u = lvel.copy()
          u[3:] -= lw[3:]
          rot = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
          Dl = (_added_mass_jac(u, density, gf[gid, 6:9], gf[gid, 9:12])
                + _viscous_jac(u, density, viscosity, size, gf[gid, 5],
                               gf[gid, 4], gf[gid, 1], gf[gid, 2], gf[gid, 3]))
          Dl *= float(gf[gid, 0])
          J = _geom_jac6(model, data, gid)
          R = _rot6(rot)
          out[row] += J.T @ R @ Dl @ R.T @ J
        continue
      inertia = np.asarray(model.body_inertia[body], dtype=np.float64)
      box = np.sqrt(np.maximum(mujoco.mjMINVAL, np.array(
          [inertia[1] + inertia[2] - inertia[0], inertia[0] + inertia[2] - inertia[1],
           inertia[0] + inertia[1] - inertia[2]]) / mass * 6))
      u = np.zeros(6)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, u, 1)
      lw = np.zeros(6)
      root = int(model.body_rootid[body])
      mujoco.mju_transformSpatial(lw, wind, 0, data.xipos[body],
                                  data.subtree_com[root], data.ximat[body].reshape(-1))
      u[3:] -= lw[3:]
      Dl = np.zeros((6, 6))
      if viscosity > 0:
        diameter = float(np.sum(box) / 3)
        Dl[:3, :3] = -np.pi * diameter ** 3 * viscosity * np.eye(3)
        Dl[3:, 3:] = -3 * np.pi * diameter * viscosity * np.eye(3)
      if density > 0:
        for axis in range(3):
          other = [i for i in range(3) if i != axis]
          j, k = other
          v = u[3 + axis]
          Dl[3 + axis, 3 + axis] += -density * box[j] * box[k] * abs(v)
          w = u[axis]
          Dl[axis, axis] += (-density * box[axis] * (box[j] ** 4 + box[k] ** 4)
                             * abs(w) / 32)
      J = _body_jac6(model, data, body)
      R = _rot6(data.ximat[body].reshape(3, 3))
      out[row] += J.T @ R @ Dl @ R.T @ J
  return out


def _supported_model(model):
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError(f"fluid lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  gf = np.asarray(model.geom_fluid, dtype=np.float64).reshape(int(model.ngeom), 12)
  if not np.all(np.isfinite(gf)):
    raise ValueError("geom-fluid parameters must be finite")
  with np.errstate(over="ignore", under="ignore", invalid="ignore"):
    gf32 = np.asarray(gf, dtype=np.float32)
  if not np.all(np.isfinite(gf32)):
    raise ValueError("geom-fluid parameters must be float32-representable")
  if np.any((gf[:, 0] != 0) & (gf[:, 0] != 1)):
    raise ValueError("geom interaction scale must be 0 or 1")
  if model.opt.density < 0 or model.opt.viscosity < 0:
    raise ValueError("fluid density and viscosity must be nonnegative")
  constants = np.r_[model.opt.density, model.opt.viscosity, model.opt.wind]
  if not np.all(np.isfinite(constants)) or not np.all(np.isfinite(constants.astype(np.float32))):
    raise ValueError("fluid coefficients and wind must be finite float32 values")
  dims = (int(model.nq), int(model.nv), int(model.nbody), int(model.njnt))
  if any(value < 0 or value > _INT32_MAX for value in dims):
    raise ValueError("fluid model dimensions exceed int32")
  return dims


def _geom_semiaxes(size, geom_type):
  # Pinned mju_geomSemiAxes (3.10.0).
  size = np.asarray(size, dtype=np.float64).reshape(3)
  if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
    return np.array([size[0], size[0], size[0]])
  if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
    return np.array([size[0], size[0], size[1] + size[0]])
  if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
    return np.array([size[0], size[0], size[1]])
  return np.array([size[0], size[1], size[2]])


def _added_mass_forces(lvel, density, vmass, vinertia):
  # Pinned mj_addedMassForces (local_accels disabled path).
  lin = lvel[3:6].astype(np.float64)
  ang = lvel[0:3].astype(np.float64)
  vlin = density * np.asarray(vmass, dtype=np.float64) * lin
  vang = density * np.asarray(vinertia, dtype=np.float64) * ang
  out = np.zeros(6, dtype=np.float64)
  out[3:6] += np.cross(vlin, ang)
  out[0:3] += np.cross(vlin, lin) + np.cross(vang, ang)
  return out


def _viscous_forces(lvel, density, viscosity, size, magnus, kutta,
                    blunt, slender, ang_drag):
  # Pinned mj_viscousForces (3.10.0).
  lin = lvel[3:6].astype(np.float64)
  ang = lvel[0:3].astype(np.float64)
  size = np.asarray(size, dtype=np.float64).reshape(3)
  out = np.zeros(6, dtype=np.float64)
  volume = 4.0 / 3.0 * np.pi * size[0] * size[1] * size[2]
  out[3:6] += magnus * density * volume * np.cross(ang, lin)
  dmax = float(np.max(size))
  dmin = float(np.min(size))
  dmid = float(size[0] + size[1] + size[2] - dmax - dmin)
  amax = np.pi * dmax * dmid
  denom = ((size[1] * size[2]) ** 4 * lin[0] ** 2
           + (size[2] * size[0]) ** 4 * lin[1] ** 2
           + (size[0] * size[1]) ** 4 * lin[2] ** 2)
  num = ((size[1] * size[2] * lin[0]) ** 2
         + (size[2] * size[0] * lin[1]) ** 2
         + (size[0] * size[1] * lin[2]) ** 2)
  aproj = np.pi * np.sqrt(denom / max(mujoco.mjMINVAL, num)) if num > 0 else 0.0
  norm = np.array([(size[1] * size[2]) ** 2 * lin[0],
                   (size[2] * size[0]) ** 2 * lin[1],
                   (size[0] * size[1]) ** 2 * lin[2]])
  cos_alpha = (num / max(mujoco.mjMINVAL, float(np.linalg.norm(lin)) * denom)
               if denom > 0 and float(np.linalg.norm(lin)) > 0 else 0.0)
  circ = np.cross(norm, lin) * kutta * density * cos_alpha * aproj
  out[3:6] += np.cross(circ, lin)
  eq_d = 2.0 / 3.0 * float(size[0] + size[1] + size[2])
  lin_coef = 3.0 * np.pi * eq_d
  torq_coef = np.pi * eq_d ** 3
  def max_moment(s, d):
    d0, d1, d2 = s[d], s[(d + 1) % 3], s[(d + 2) % 3]
    return 8.0 / 15.0 * np.pi * d0 * max(d1, d2) ** 4
  ii = np.array([max_moment(size, 0), max_moment(size, 1), max_moment(size, 2)])
  imax = 8.0 / 15.0 * np.pi * dmid * dmax ** 4
  mom = ang * (ang_drag * ii + slender * (imax - ii))
  drag_lin = viscosity * lin_coef + density * float(np.linalg.norm(lin)) * (
      aproj * blunt + slender * (amax - aproj))
  drag_ang = viscosity * torq_coef + density * float(np.linalg.norm(mom))
  out[0:3] -= drag_ang * ang
  out[3:6] -= drag_lin * lin
  return out


class InertiaBoxFluidModel:
  """CPU reference for the pinned fluid force models.

  Bodies with any interacting ellipsoid geom (`geom_fluid[..., 0] == 1`) use
  the per-geom ellipsoid model (added-mass + Magnus/Kutta lift + viscous
  blunt/slender/angular terms); other bodies use the inertia-box model,
  matching pinned `mj_fluid` selection.
  """

  def __init__(self, model):
    self.nq, self.nv, self.nbody, self.njnt = _supported_model(model)
    self.model = model
    self.mass = _frozen(model.body_mass)
    self.inertia = _frozen(model.body_inertia)
    self.wind = _frozen(model.opt.wind)
    self.density = float(model.opt.density)
    self.viscosity = float(model.opt.viscosity)
    self.disableflags = int(model.opt.disableflags)
    self.ngeom = int(model.ngeom)
    self.geom_bodyid = np.asarray(model.geom_bodyid, dtype=np.int32).copy()
    self.geom_type = np.asarray(model.geom_type, dtype=np.int32).copy()
    self.geom_size = _frozen(np.asarray(model.geom_size).reshape(-1, 3))
    self.geom_fluid = _frozen(
        np.asarray(model.geom_fluid, dtype=np.float32).reshape(-1, 12))

  def run(self, qpos, qvel):
    qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != self.nq or qvel.shape != (qpos.shape[0], self.nv) or not qpos.shape[0]:
      raise ValueError(f"qpos/qvel must have shapes (B, {self.nq}) and (B, {self.nv})")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
      raise ValueError("qpos and qvel must be finite")
    result = np.zeros((len(qpos), self.nv), dtype=np.float64)
    # mj_passive returns before fluid evaluation when both bits are disabled.
    spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    if self.disableflags & spring and self.disableflags & damper:
      return result
    if not self.density and not self.viscosity:
      return result
    for row in range(len(qpos)):
      data = mujoco.MjData(self.model)
      data.qpos[:] = qpos[row]
      data.qvel[:] = qvel[row]
      mujoco.mj_forward(self.model, data)
      gf = np.asarray(self.model.geom_fluid, dtype=np.float64).reshape(-1, 12)
      for body in range(1, self.nbody):
        mass = float(self.mass[body])
        if mass < mujoco.mjMINVAL:
          continue
        # Pinned selection: any interacting ellipsoid geom disables the
        # inertia-box model for the parent body.
        adr = int(self.model.body_geomadr[body])
        num = int(self.model.body_geomnum[body])
        use_ellipsoid = bool(np.any(gf[adr:adr + num, 0] > 0)) if num else False
        if use_ellipsoid:
          for j in range(num):
            gid = adr + j
            if float(gf[gid, 0]) == 0.0:
              continue
            size = _geom_semiaxes(np.asarray(self.model.geom_size[gid]),
                                  int(np.asarray(self.model.geom_type)[gid]))
            lvel = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                     gid, lvel, 1)
            wind = np.r_[np.zeros(3), self.wind.astype(np.float64)]
            lw = np.zeros(6, dtype=np.float64)
            root = int(self.model.body_rootid[body])
            mujoco.mju_transformSpatial(
                lw, wind, 0, np.asarray(data.geom_xpos[gid]),
                np.asarray(data.subtree_com[root]),
                np.asarray(data.geom_xmat[gid]).reshape(-1))
            lvel[3:] -= lw[3:]
            lfrc = _added_mass_forces(
                lvel, self.density, gf[gid, 6:9], gf[gid, 9:12])
            lfrc += _viscous_forces(
                lvel, self.density, self.viscosity, size, gf[gid, 5],
                gf[gid, 4], gf[gid, 1], gf[gid, 2], gf[gid, 3])
            lfrc *= float(gf[gid, 0])
            rot = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
            mujoco.mj_applyFT(
                self.model, data, rot @ lfrc[3:], rot @ lfrc[:3],
                np.asarray(data.geom_xpos[gid]), body, result[row])
          continue
        inertia = self.inertia[body].astype(np.float64)
        box = np.sqrt(np.maximum(mujoco.mjMINVAL,
            np.array([inertia[1]+inertia[2]-inertia[0], inertia[0]+inertia[2]-inertia[1], inertia[0]+inertia[1]-inertia[2]]) / mass * 6))
        local_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, local_velocity, 1)
        wind = np.r_[np.zeros(3), self.wind.astype(np.float64)]
        local_wind = np.zeros(6, dtype=np.float64)
        root = int(self.model.body_rootid[body])
        mujoco.mju_transformSpatial(local_wind, wind, 0, data.xipos[body], data.subtree_com[root], data.ximat[body].reshape(-1))
        local_velocity[3:] -= local_wind[3:]
        local_force = np.zeros(6, dtype=np.float64)
        if self.viscosity > 0:
          diameter = float(np.sum(box) / 3)
          local_force[:3] = -np.pi * diameter**3 * self.viscosity * local_velocity[:3]
          local_force[3:] = -3 * np.pi * diameter * self.viscosity * local_velocity[3:]
        if self.density > 0:
          for axis in range(3):
            other = [i for i in range(3) if i != axis]
            local_force[3+axis] -= .5 * self.density * box[other[0]] * box[other[1]] * abs(local_velocity[3+axis]) * local_velocity[3+axis]
            j, k = other
            local_force[axis] -= self.density * box[axis] * (box[j]**4 + box[k]**4) * abs(local_velocity[axis]) * local_velocity[axis] / 64
        rotation = data.ximat[body].reshape(3, 3)
        world_torque = rotation @ local_force[:3]
        world_force = rotation @ local_force[3:]
        mujoco.mj_applyFT(self.model, data, world_force, world_torque,
                          data.xipos[body], body, result[row])
    return result


class MetalInertiaBoxFluid:
  """Native MPS inertia-box fluid stage.

  `run_device` consumes qpos/qvel plus a native smooth-stage result containing
  `poses`, `cvel`, and `root_com`; this keeps kinematics owned by the caller's
  smooth stage and avoids a redundant mass/bias computation.
  """

  def __init__(self, model, batch_size=1):
    self._meta = InertiaBoxFluidModel(model)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0 or batch_size > _INT32_MAX:
      raise ValueError("batch_size must be a positive int32 integer")
    if max(batch_size * self._meta.nbody * 36,
           batch_size * self._meta.njnt * 3,
           batch_size * self._meta.nv,
           batch_size * max(self._meta.ngeom, 1) * 12) > _UINT32_MAX:
      raise ValueError("fluid workspace exceeds uint32 indexing")
    if self._meta.ngeom > _INT32_MAX:
      raise ValueError("fluid geom count exceeds int32")
    self.batch_size = batch_size
    import torch
    self._torch = torch
    self._device = torch.device("mps")
    meta = self._meta
    def tensor(value, dtype=torch.float32):
      arr = np.array(value, dtype=np.int32 if dtype == torch.int32 else np.float32, order="C", copy=True)
      if arr.dtype == np.float32 and not np.all(np.isfinite(arr)):
        raise ValueError("fluid constants must be finite float32")
      if not arr.size:
        arr = np.zeros(1, dtype=arr.dtype)
      return torch.as_tensor(arr, dtype=dtype, device=self._device)
    for name in ("body_parentid", "body_rootid", "body_jntadr", "body_jntnum", "jnt_type", "jnt_dofadr"):
      setattr(self, "_" + name, tensor(getattr(model, name), torch.int32))
    self._mass = tensor(meta.mass)
    self._inertia = tensor(meta.inertia.reshape(-1))
    self._fluid = tensor([meta.density, meta.viscosity, *meta.wind])
    spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    disabled = bool(meta.disableflags & spring and meta.disableflags & damper)
    self._dims = tensor([meta.nv, meta.nbody, meta.njnt, batch_size, int(disabled), meta.ngeom], torch.int32)
    # Bodies with an interacting ellipsoid geom skip inertia-box (pinned).
    skip = np.zeros(int(model.nbody), dtype=np.int32)
    gf0 = np.asarray(meta.geom_fluid).reshape(-1, 12)[:, 0]
    for b in range(int(model.nbody)):
      adr, num = int(model.body_geomadr[b]), int(model.body_geomnum[b])
      if num and bool(np.any(gf0[adr:adr + num] > 0)):
        skip[b] = 1
    self._body_skip = tensor(skip, torch.int32)
    self._geom_bodyid = tensor(meta.geom_bodyid, torch.int32)
    self._geom_type = tensor(meta.geom_type, torch.int32)
    self._geom_size = tensor(np.asarray(meta.geom_size).reshape(-1))
    self._geom_fluid = tensor(np.asarray(meta.geom_fluid).reshape(-1))
    self._output = torch.empty((batch_size, meta.nv), dtype=torch.float32, device=self._device)
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.inertia_box_fluid
    self._geom_kernel = self._library.ellipsoid_geom_fluid

  def run_device(self, qpos, qvel, dynamics=None):
    torch, meta = self._torch, self._meta
    for name, value, shape in (("qpos", qpos, (self.batch_size, meta.nq)), ("qvel", qvel, (self.batch_size, meta.nv))):
      if not isinstance(value, torch.Tensor) or value.device.type != "mps" or value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    if dynamics is None:
      raise ValueError("pass the native smooth-stage result with poses, cvel, and root_com")
    required = {"poses", "cvel", "root_com"}
    if not isinstance(dynamics, dict) or not required.issubset(dynamics):
      raise ValueError("dynamics must contain native poses, cvel, and root_com")
    poses = dynamics["poses"]
    names_shapes = [
        ("poses.body_quat", poses["body_quat"], (self.batch_size, meta.nbody, 4)),
        ("poses.inertial_pos", poses["inertial_pos"], (self.batch_size, meta.nbody, 3)),
        ("poses.inertial_quat", poses["inertial_quat"], (self.batch_size, meta.nbody, 4)),
        ("poses.joint_anchor", poses["joint_anchor"], (self.batch_size, meta.njnt, 3)),
        ("poses.joint_axis", poses["joint_axis"], (self.batch_size, meta.njnt, 3)),
        ("cvel", dynamics["cvel"], (self.batch_size, meta.nbody, 6)),
        ("root_com", dynamics["root_com"], (self.batch_size, meta.nbody, 3)),
    ]
    if meta.ngeom:
      names_shapes.extend([
          ("poses.geom_pos", poses["geom_pos"], (self.batch_size, meta.ngeom, 3)),
          ("poses.geom_quat", poses["geom_quat"], (self.batch_size, meta.ngeom, 4)),
      ])
    for name, value, shape in names_shapes:
      if not isinstance(value, torch.Tensor) or value.device.type != "mps" or value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    nv = meta.nv
    self._kernel(self._body_parentid, self._body_rootid, self._body_jntadr, self._body_jntnum,
        self._jnt_type, self._jnt_dofadr, self._mass, self._inertia,
        poses["body_quat"].reshape(-1), poses["inertial_pos"].reshape(-1),
        poses["inertial_quat"].reshape(-1), dynamics["cvel"].reshape(-1),
        dynamics["root_com"].reshape(-1), poses["joint_anchor"].reshape(-1),
        poses["joint_axis"].reshape(-1), self._fluid, self._dims,
        self._output.reshape(-1) if nv else self._dummy, self._body_skip,
        threads=(self.batch_size,), group_size=(1,))
    if nv and meta.ngeom:
      self._geom_kernel(self._body_parentid, self._body_rootid, self._body_jntadr, self._body_jntnum,
          self._jnt_type, self._jnt_dofadr,
          poses["body_quat"].reshape(-1), dynamics["cvel"].reshape(-1),
          dynamics["root_com"].reshape(-1), poses["joint_anchor"].reshape(-1),
          poses["joint_axis"].reshape(-1), self._geom_bodyid, self._geom_type,
          self._geom_size.reshape(-1), self._geom_fluid.reshape(-1),
          poses["geom_pos"].reshape(-1), poses["geom_quat"].reshape(-1),
          self._fluid, self._dims, self._output.reshape(-1),
          threads=(self.batch_size,), group_size=(1,))
    return self._output
