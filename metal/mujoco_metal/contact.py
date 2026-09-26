# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native Metal normal contact primitives for MuJoCo 3.10.0.

This stage lowers a deliberately bounded contact family (sphere/sphere and
plane/sphere, condim=1) to fixed geometry-pair work items. Detection,
Jacobian construction, reference generation and the regularized normal solve
run in MSL. Host code only validates/lower constants and allocates buffers.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "contact.metal"
_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_MINVAL = 1e-15


def _frozen(value, dtype):
  array = np.asarray(value, dtype=dtype, order="C")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ContactDescriptor:
  """Fixed model constants and candidate pairs for supported normal contacts."""

  geom1: np.ndarray
  geom2: np.ndarray
  radius1: np.ndarray
  radius2: np.ndarray
  margin: np.ndarray
  gap: np.ndarray
  solref: np.ndarray
  solimp: np.ndarray
  timestep: float
  refsafe: bool
  nv: int
  nbody: int
  njnt: int
  ngeom: int
  geom_bodyid: np.ndarray
  body_parentid: np.ndarray
  body_jntadr: np.ndarray
  body_jntnum: np.ndarray
  body_dofadr: np.ndarray
  body_dofnum: np.ndarray
  jnt_type: np.ndarray
  jnt_dofadr: np.ndarray
  body_invweight0: np.ndarray

  @property
  def pair_count(self):
    return len(self.geom1)


def lower_contacts(model):
  """Build eligible geom pairs and effective condim-1 contact parameters.

  Unsupported collision types, pair overrides, exclusions, runtime contact
  filters, override parameters, and adhesion fail explicitly. This avoids
  silently stepping models with a different collision/constraint law.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("contact lowering requires the source mujoco.MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("native contact requires MuJoCo 3.10.0")
  if model.npair or model.nexclude:
    raise ValueError("native contact does not support explicit pairs or exclusions")
  if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
    raise ValueError("native contact does not support global contact overrides")
  if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
    raise ValueError("native normal contact currently requires Euler integration")
  if mujoco.get_mjcb_contactfilter() is not None:
    raise ValueError("native contact does not support a global contact filter callback")
  geoms = np.asarray(model.geom_type)
  pairs = []
  for a in range(model.ngeom):
    for b in range(a + 1, model.ngeom):
      ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
      weld_a, weld_b = int(model.body_weldid[ba]), int(model.body_weldid[bb])
      parent_a = int(model.body_weldid[model.body_parentid[weld_a]])
      parent_b = int(model.body_weldid[model.body_parentid[weld_b]])
      filter_parent = not (
          int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
      )
      if weld_a == weld_b or (
          filter_parent and weld_a != 0 and weld_b != 0
          and (weld_a == parent_b or weld_b == parent_a)
      ):
        continue
      if not (
          (int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
          or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))
      ):
        continue
      ta, tb = int(geoms[a]), int(geoms[b])
      if (ta, tb) not in ((_SPHERE, _SPHERE), (_PLANE, _SPHERE), (_SPHERE, _PLANE)):
        raise ValueError(
            f"native contact does not support collidable geom pair ({a}, {b}) "
            f"with types ({ta}, {tb})"
        )
      p1, p2 = int(model.geom_priority[a]), int(model.geom_priority[b])
      condim = (
          int(model.geom_condim[a]) if p1 > p2 else
          int(model.geom_condim[b]) if p2 > p1 else
          max(int(model.geom_condim[a]), int(model.geom_condim[b]))
      )
      if condim != 1:
        raise ValueError(f"contact pair ({a}, {b}) has condim other than 1")
      # MuJoCo's regular collision path orders geom ids. Preserve that order.
      p1, p2 = a, b
      solref, solimp = _mix_contact_parameters(model, p1, p2)
      if bool(solref[0] > 0) != bool(solref[1] > 0):
        solref = np.asarray(model.opt.o_solref).copy()
      if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)) and solref[0] > 0:
        solref[0] = max(float(solref[0]), 2.0 * float(model.opt.timestep))
      pairs.append((p1, p2, solref, solimp))

  if pairs:
    g1 = np.array([p[0] for p in pairs], dtype=np.int32)
    g2 = np.array([p[1] for p in pairs], dtype=np.int32)
    sr = np.array([p[2] for p in pairs], dtype=np.float64)
    si = np.array([p[3] for p in pairs], dtype=np.float64)
  else:
    g1 = g2 = np.empty(0, dtype=np.int32)
    sr = np.empty((0, 2), dtype=np.float64)
    si = np.empty((0, 5), dtype=np.float64)
  return ContactDescriptor(
      geom1=_frozen(g1, np.int32),
      geom2=_frozen(g2, np.int32),
      radius1=_frozen(np.where(geoms[g1] == _PLANE, -1.0, model.geom_size[g1, 0]), np.float32),
      radius2=_frozen(np.where(geoms[g2] == _PLANE, -1.0, model.geom_size[g2, 0]), np.float32),
      margin=_frozen(model.geom_margin[g1] + model.geom_margin[g2], np.float32),
      gap=_frozen(model.geom_gap[g1] + model.geom_gap[g2], np.float32),
      solref=_frozen(sr, np.float32),
      solimp=_frozen(si, np.float32),
      nv=int(model.nv),
      nbody=int(model.nbody),
      njnt=int(model.njnt),
      ngeom=int(model.ngeom),
      timestep=float(model.opt.timestep),
      refsafe=not bool(
          int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
      ),
      geom_bodyid=_frozen(model.geom_bodyid, np.int32),
      body_parentid=_frozen(model.body_parentid, np.int32),
      body_jntadr=_frozen(model.body_jntadr, np.int32),
      body_jntnum=_frozen(model.body_jntnum, np.int32),
      body_dofadr=_frozen(model.body_dofadr, np.int32),
      body_dofnum=_frozen(model.body_dofnum, np.int32),
      jnt_type=_frozen(model.jnt_type, np.int32),
      jnt_dofadr=_frozen(model.jnt_dofadr, np.int32),
      body_invweight0=_frozen(model.body_invweight0.reshape(-1, 2), np.float32),
  )


def _mix_contact_parameters(model, g1, g2):
  """MuJoCo 3.10 same-priority solref/solimp mixing for geom contacts."""
  p1, p2 = int(model.geom_priority[g1]), int(model.geom_priority[g2])
  if p1 > p2:
    return model.geom_solref[g1].copy(), model.geom_solimp[g1].copy()
  if p2 > p1:
    return model.geom_solref[g2].copy(), model.geom_solimp[g2].copy()
  m1, m2 = float(model.geom_solmix[g1]), float(model.geom_solmix[g2])
  if m1 >= _MINVAL and m2 >= _MINVAL:
    mix = m1 / (m1 + m2)
  elif m1 < _MINVAL and m2 < _MINVAL:
    mix = 0.5
  else:
    mix = 0.0 if m1 < _MINVAL else 1.0
  r1, r2 = model.geom_solref[g1], model.geom_solref[g2]
  ref = mix * r1 + (1 - mix) * r2 if r1[0] > 0 and r2[0] > 0 else np.minimum(r1, r2)
  imp = mix * model.geom_solimp[g1] + (1 - mix) * model.geom_solimp[g2]
  return ref, imp


def _get_impedance(solimp, pos, margin):
  """MuJoCo 3.10 getimpedance, including its derivative."""
  d0, d1, width, midpoint, power = map(float, solimp)
  if d0 == d1 or width <= _MINVAL:
    return 0.5 * (d0 + d1), 0.0
  x = (pos - margin) / width
  sign = -1.0 if x < 0 else 1.0
  x = abs(x)
  if x >= 1 or x <= 0:
    return (d1 if x >= 1 else d0), 0.0
  if power == 1:
    y, yp = x, 1.0
  elif x <= midpoint:
    a = 1.0 / midpoint ** (power - 1)
    y, yp = a * x**power, power * a * x ** (power - 1)
  else:
    b = 1.0 / (1 - midpoint) ** (power - 1)
    y, yp = 1 - b * (1 - x) ** power, power * b * (1 - x) ** (power - 1)
  return d0 + y * (d1 - d0), yp * sign * (d1 - d0) / width


class MetalContact:
  """MPS contact kernels consuming device FK outputs and a generalized metric.

  This low-level stage is intentionally separate from ``MetalSimulation`` so
  the caller can compose it after native FK and before the generalized solve.
  ``run_device`` returns a fixed slot per eligible geometry pair plus a mask;
  contact creation never allocates or calls CPU MuJoCo during stepping.
  """

  def __init__(self, model, batch_size=1):
    self.descriptor = lower_contacts(model)
    if self.descriptor.nv > 32 or self.descriptor.pair_count > 16:
      raise ValueError("native contact currently bounds models to nv<=32 and 16 candidate pairs")
    self.batch_size = int(batch_size)
    if self.batch_size <= 0:
      raise ValueError("batch_size must be positive")
    import torch
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
      raise RuntimeError("native contact requires PyTorch MPS compile_shader")
    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.contact_normal
    self._solve_kernel = self._library.solve_normal_contacts
    self._constants = {
        "geom1": self._tensor(self.descriptor.geom1),
        "geom2": self._tensor(self.descriptor.geom2),
        "radius1": self._tensor(self.descriptor.radius1),
        "radius2": self._tensor(self.descriptor.radius2),
        "margin": self._tensor(self.descriptor.margin),
        "gap": self._tensor(self.descriptor.gap),
        "solref": self._tensor(self.descriptor.solref.reshape(-1)),
        "solimp": self._tensor(self.descriptor.solimp.reshape(-1)),
        "geom_bodyid": self._tensor(self.descriptor.geom_bodyid),
        "body_parentid": self._tensor(self.descriptor.body_parentid),
        "body_jntadr": self._tensor(self.descriptor.body_jntadr),
        "body_jntnum": self._tensor(self.descriptor.body_jntnum),
        "jnt_type": self._tensor(self.descriptor.jnt_type),
        "jnt_dofadr": self._tensor(self.descriptor.jnt_dofadr),
        "body_invweight0": self._tensor(self.descriptor.body_invweight0.reshape(-1)),
    }
    self._workspace = None
    self.prepare_workspace(self.batch_size)

  def _tensor(self, value):
    return self._torch.as_tensor(np.asarray(value).copy(), device=self._device)

  def prepare_workspace(self, batch_size):
    """Preallocate per-world contact and solver buffers."""
    d, torch = self.descriptor, self._torch
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ) or int(batch_size) <= 0:
      raise ValueError("batch_size must be positive")
    b, c, nv = int(batch_size), d.pair_count, d.nv
    if b > (1 << 31) - 1 or b * max(c, 1) * max(nv, 1) > (1 << 32):
      raise ValueError("contact workspace exceeds Metal index capacity")
    self.batch_size = b
    def empty(size):
      return torch.zeros(max(size, 1), dtype=torch.float32, device=self._device)
    self._workspace = {
        "row_data": empty(b * c * 6), "frame": empty(b * c * 6),
        "jacobian": empty(b * c * nv), "force": empty(b * c),
        "qacc": empty(b * nv),
        "status": torch.zeros(b, dtype=torch.int32, device=self._device),
        "dims": torch.tensor([nv, c, b, d.nbody, d.njnt, d.ngeom], dtype=torch.int32, device=self._device),
    }
    return self._workspace

  def run_device(self, fk, mass, free_acceleration, qvel):
    """Detect contacts and solve normal forces entirely on MPS.

    ``fk`` is the dict returned by ``MetalKinematics.run_device``. ``mass`` is
    batched dense generalized inertia [batch,nv,nv], and free acceleration and
    velocity have shape [batch,nv]. Returned tensors borrow this workspace.
    """
    w, torch, d = self._workspace, self._torch, self.descriptor
    def check(tensor, name, shape):
      if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
      if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
      if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
      if tensor.device.type != self._device.type:
        raise ValueError(f"{name} must be on {self._device}")
      if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    check(qvel, "qvel", (self.batch_size, d.nv))
    check(free_acceleration, "free_acceleration", (self.batch_size, d.nv))
    check(mass, "mass", (self.batch_size, d.nv, d.nv))
    for name, width, count in (
        ("geom_pos", 3, d.ngeom), ("geom_quat", 4, d.ngeom),
        ("body_pos", 3, d.nbody), ("body_quat", 4, d.nbody),
        ("joint_anchor", 3, d.njnt), ("joint_axis", 3, d.njnt),
    ):
      check(fk[name], f"fk[{name}]", (self.batch_size, count, width))
    self._kernel(
        fk["geom_pos"], fk["geom_quat"], fk["body_pos"], fk["body_quat"],
        fk["joint_anchor"], fk["joint_axis"], qvel.reshape(-1),
        *[self._constants[k] for k in ("geom1", "geom2", "radius1", "radius2", "margin", "gap", "solref", "solimp")],
        *[self._constants[k] for k in ("geom_bodyid", "body_parentid", "body_jntadr", "body_jntnum", "jnt_type", "jnt_dofadr", "body_invweight0")],
        w["row_data"], w["frame"], w["jacobian"], w["dims"],
        threads=(self.batch_size * max(d.pair_count, 1),), group_size=(1,),
    )
    self._solve_kernel(
        mass.reshape(-1), free_acceleration.reshape(-1), w["jacobian"],
        w["row_data"], w["force"], w["qacc"], w["status"], w["dims"],
        threads=(self.batch_size,), group_size=(1,),
    )
    rows = w["row_data"][: self.batch_size * d.pair_count * 6].reshape(
        self.batch_size, d.pair_count, 6
    )
    frames = w["frame"][: self.batch_size * d.pair_count * 6].reshape(
        self.batch_size, d.pair_count, 6
    )
    return {
        "mask": rows[:, :, 0], "dist": rows[:, :, 1],
        "velocity": rows[:, :, 2], "reference": rows[:, :, 3],
        "impedance": rows[:, :, 4], "normal": frames[:, :, :3],
        "position": frames[:, :, 3:],
        "jacobian": w["jacobian"][: self.batch_size * d.pair_count * d.nv].reshape(
            self.batch_size, d.pair_count, d.nv
        ),
        "force": w["force"][: self.batch_size * d.pair_count].reshape(
            self.batch_size, d.pair_count
        ),
        "qacc": w["qacc"][: self.batch_size * d.nv].reshape(self.batch_size, d.nv),
        "status": w["status"],
    }
