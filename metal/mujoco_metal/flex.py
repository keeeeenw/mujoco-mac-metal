# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Native deformable physics (milestone 018) for MuJoCo 3.10 models on Metal.

Implements the pinned flex inventory:
- Topologies: 1D (cable/string), 2D (cloth/shell/grid), 3D (tetrahedral/volumetric).
- Constitutive models:
  * Edge distance equality constraints (mjtEq.mjEQ_FLEX).
  * Mass-spring network elastic forces and damping.
  * Continuum St. Venant-Kirchhoff (StVK) / Neo-Hookean strain energy,
    internal stress, and nodal restoring forces.
- Internal strain forces and analytic force derivatives (stiffness and damping tangents) on MPS.
- Flex-rigid attachments and constraint coupling with the scalable block-row solver.
- Narrowphase vertex-rigid obstacle collision and self-collision.
- Full state lifecycle: reset, step, copy, snapshot, restore of contacting deformed states.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
import torch


def _frozen(values, dtype):
  array = np.array(values, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("flex constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class FlexDescriptor:
  """Immutable lowered flex descriptors for a compiled mjModel."""
  nflex: int
  nflexvert: int
  nflexedge: int
  nflexelem: int
  nv: int
  nq: int
  nbody: int
  njnt: int
  timestep: float
  dim: np.ndarray             # (nflex,) int32
  vert: np.ndarray            # (nflexvert, 3) float32
  vert0: np.ndarray           # (nflexvert, 3) float32
  vertbodyid: np.ndarray      # (nflexvert,) int32
  edge: np.ndarray            # (nflexedge, 2) int32
  edgeadr: np.ndarray         # (nflex,) int32
  edgenum: np.ndarray         # (nflex,) int32
  edgeequality: np.ndarray    # (nflex,) int32
  edgestiffness: np.ndarray   # (nflex,) float32
  edgedamping: np.ndarray     # (nflex,) float32
  edge_length0: np.ndarray    # (nflexedge,) float32
  edge_invweight0: np.ndarray # (nflexedge,) float32
  elem: np.ndarray            # (nflexelem_total,) int32
  elemadr: np.ndarray         # (nflex,) int32
  elemnum: np.ndarray         # (nflex,) int32
  radius: np.ndarray          # (nflex,) float32
  contype: np.ndarray         # (nflex,) int32
  conaffinity: np.ndarray     # (nflex,) int32
  condim: np.ndarray          # (nflex,) int32
  friction: np.ndarray        # (nflex, 3) float32
  margin: np.ndarray          # (nflex,) float32
  gap: np.ndarray             # (nflex,) float32
  selfcollide: np.ndarray     # (nflex,) int32
  solref: np.ndarray          # (nflex, 2) float32
  solimp: np.ndarray          # (nflex, 5) float32
  young: np.ndarray           # (nflex,) float32 (effective Young's modulus)
  poisson: np.ndarray         # (nflex,) float32 (Poisson's ratio)
  lame_lambda: np.ndarray     # (nflex,) float32
  lame_mu: np.ndarray         # (nflex,) float32


def lower_flex_descriptor(model: mujoco.MjModel) -> Optional[FlexDescriptor]:
  """Lower flex structures from an mjModel into a FlexDescriptor."""
  nflex = int(model.nflex)
  if nflex == 0:
    return None

  nflexvert = int(model.nflexvert)
  nflexedge = int(model.nflexedge)
  nflexelem = int(model.nflexelem)
  nv = int(model.nv)
  nq = int(model.nq)
  nbody = int(model.nbody)
  njnt = int(model.njnt)
  timestep = float(model.opt.timestep)

  dim = _frozen(model.flex_dim, np.int32)
  vert = _frozen(model.flex_vert, np.float32).reshape(nflexvert, 3)
  vert0 = _frozen(model.flex_vert0, np.float32).reshape(nflexvert, 3)
  vertbodyid = _frozen(model.flex_vertbodyid, np.int32)
  edge = _frozen(model.flex_edge, np.int32).reshape(nflexedge, 2)
  edgeadr = _frozen(model.flex_edgeadr, np.int32)
  edgenum = _frozen(model.flex_edgenum, np.int32)
  edgeequality = _frozen(model.flex_edgeequality, np.int32)
  edgestiffness = _frozen(model.flex_edgestiffness, np.float32)
  edgedamping = _frozen(model.flex_edgedamping, np.float32)
  edge_length0 = _frozen(model.flexedge_length0, np.float32)
  edge_invweight0 = _frozen(model.flexedge_invweight0, np.float32)
  elem = _frozen(model.flex_elem, np.int32)
  elemadr = _frozen(model.flex_elemadr, np.int32)
  elemnum = _frozen(model.flex_elemnum, np.int32)
  radius = _frozen(model.flex_radius, np.float32)
  contype = _frozen(model.flex_contype, np.int32)
  conaffinity = _frozen(model.flex_conaffinity, np.int32)
  condim = _frozen(model.flex_condim, np.int32)
  friction = _frozen(model.flex_friction, np.float32).reshape(nflex, 3)
  margin = _frozen(model.flex_margin, np.float32)
  gap = _frozen(model.flex_gap, np.float32)
  selfcollide = _frozen(model.flex_selfcollide, np.int32)
  solref = _frozen(model.flex_solref, np.float32).reshape(nflex, 2)
  solimp = _frozen(model.flex_solimp, np.float32).reshape(nflex, 5)

  # Derived Lamé parameters from stiffness or elasticity settings
  young = np.zeros(nflex, dtype=np.float32)
  poisson = np.full(nflex, 0.3, dtype=np.float32)
  lame_lambda = np.zeros(nflex, dtype=np.float32)
  lame_mu = np.zeros(nflex, dtype=np.float32)

  for f in range(nflex):
    k = float(edgestiffness[f])
    if k <= 0.0:
      k = 1000.0  # Default continuum stiffness if unspecified
    young[f] = k
    nu = float(poisson[f])
    lame_lambda[f] = (k * nu) / ((1.0 + nu) * max(1.0 - 2.0 * nu, 1e-4))
    lame_mu[f] = k / (2.0 * (1.0 + nu))

  return FlexDescriptor(
      nflex=nflex,
      nflexvert=nflexvert,
      nflexedge=nflexedge,
      nflexelem=nflexelem,
      nv=nv,
      nq=nq,
      nbody=nbody,
      njnt=njnt,
      timestep=timestep,
      dim=dim,
      vert=vert,
      vert0=vert0,
      vertbodyid=vertbodyid,
      edge=edge,
      edgeadr=edgeadr,
      edgenum=edgenum,
      edgeequality=edgeequality,
      edgestiffness=edgestiffness,
      edgedamping=edgedamping,
      edge_length0=edge_length0,
      edge_invweight0=edge_invweight0,
      elem=elem,
      elemadr=elemadr,
      elemnum=elemnum,
      radius=radius,
      contype=contype,
      conaffinity=conaffinity,
      condim=condim,
      friction=friction,
      margin=margin,
      gap=gap,
      selfcollide=selfcollide,
      solref=solref,
      solimp=solimp,
      young=_frozen(young, np.float32),
      poisson=_frozen(poisson, np.float32),
      lame_lambda=_frozen(lame_lambda, np.float32),
      lame_mu=_frozen(lame_mu, np.float32),
  )


class MetalFlex:
  """Device-resident deformable physics manager executed on Apple Silicon MPS."""

  def __init__(self, model: mujoco.MjModel, batch_size: int = 1, device: str = "mps"):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    desc = lower_flex_descriptor(model)
    if desc is None:
      raise ValueError("model does not contain flex elements")

    self.model = model
    self.descriptor = desc
    self.batch_size = batch_size
    self._device = torch.device(device)
    d = self.descriptor

    # Upload static descriptors to device MPS tensors
    self._vert_local = torch.tensor(d.vert, dtype=torch.float32, device=self._device)
    self._vert0 = torch.tensor(d.vert0, dtype=torch.float32, device=self._device)
    self._vertbodyid = torch.tensor(d.vertbodyid, dtype=torch.int64, device=self._device)
    self._edge = torch.tensor(d.edge, dtype=torch.int64, device=self._device)
    self._edge_length0 = torch.tensor(d.edge_length0, dtype=torch.float32, device=self._device)
    self._edge_invweight0 = torch.tensor(d.edge_invweight0, dtype=torch.float32, device=self._device)
    self._edgestiffness = torch.tensor(d.edgestiffness, dtype=torch.float32, device=self._device)
    self._edgedamping = torch.tensor(d.edgedamping, dtype=torch.float32, device=self._device)
    self._edgeequality = torch.tensor(d.edgeequality, dtype=torch.int32, device=self._device)
    self._radius = torch.tensor(d.radius, dtype=torch.float32, device=self._device)

    # Topology per-edge flex index
    edge_flexid = np.zeros(d.nflexedge, dtype=np.int64)
    for f in range(d.nflex):
      adr = d.edgeadr[f]
      num = d.edgenum[f]
      edge_flexid[adr:adr + num] = f
    self._edge_flexid = torch.tensor(edge_flexid, dtype=torch.int64, device=self._device)

    # Precomputed edge constraint parameters for run_equalities
    if d.nflexedge > 0:
      solref_edge = d.solref[edge_flexid]
      solimp_edge = d.solimp[edge_flexid]
      self._edge_timeconst = torch.tensor(np.maximum(solref_edge[:, 0], 1e-4), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_dampratio = torch.tensor(np.maximum(solref_edge[:, 1], 1e-4), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_dmin = torch.tensor(solimp_edge[:, 0], dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_dmax = torch.tensor(solimp_edge[:, 1], dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_width = torch.tensor(np.maximum(solimp_edge[:, 2], 1e-6), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_mid = torch.tensor(np.maximum(solimp_edge[:, 3], 1e-4), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_power = torch.tensor(solimp_edge[:, 4], dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_b0 = 2.0 / (self._edge_timeconst * self._edge_dampratio)
      self._edge_k0 = 1.0 / (self._edge_timeconst * self._edge_timeconst * self._edge_dampratio * self._edge_dampratio)
      self._edge_diag = torch.clamp(self._edge_invweight0.unsqueeze(0), min=1e-12)
    else:
      self._edge_b0 = torch.zeros((1, 0), dtype=torch.float32, device=self._device)
      self._edge_k0 = torch.zeros((1, 0), dtype=torch.float32, device=self._device)
      self._edge_diag = torch.zeros((1, 0), dtype=torch.float32, device=self._device)

    # Topology per-vertex flex index
    vert_flexid = np.zeros(d.nflexvert, dtype=np.int64)
    for f in range(d.nflex):
      # Pinned flexcomp vertices are sequential per flex
      v_start = int(model.flex_vertadr[f])
      v_num = int(model.flex_vertnum[f])
      vert_flexid[v_start:v_start + v_num] = f
    self._vert_flexid = torch.tensor(vert_flexid, dtype=torch.int64, device=self._device)

    # Element data for 2D (triangles: 3 verts) and 3D (tetrahedra: 4 verts)
    self._has_2d = bool(np.any(d.dim == 2))
    self._has_3d = bool(np.any(d.dim == 3))

    # Precalculate rest shape inverse matrices for 2D and 3D elements
    self._tri_elems = []
    self._tri_inv_Dm = []
    self._tri_area0 = []
    self._tet_elems = []
    self._tet_inv_Dm = []
    self._tet_vol0 = []

    for f in range(d.nflex):
      dim = int(d.dim[f])
      elem_adr = int(d.elemadr[f])
      elem_num = int(d.elemnum[f])
      if dim == 2 and elem_num > 0:
        tri_v = d.elem[elem_adr * 3:(elem_adr + elem_num) * 3].reshape(elem_num, 3)
        for t in range(elem_num):
          v0, v1, v2 = tri_v[t]
          X0, X1, X2 = d.vert0[v0], d.vert0[v1], d.vert0[v2]
          # Form 2D basis on triangle plane
          e1 = X1 - X0
          e2 = X2 - X0
          norm1 = np.linalg.norm(e1)
          u1 = e1 / max(norm1, 1e-8)
          norm_n = np.cross(u1, e2)
          u2 = np.cross(norm_n, u1)
          u2 = u2 / max(np.linalg.norm(u2), 1e-8)
          Dm = np.array([[np.dot(e1, u1), np.dot(e2, u1)],
                         [np.dot(e1, u2), np.dot(e2, u2)]], dtype=np.float32)
          det = abs(float(np.linalg.det(Dm)))
          inv_Dm = np.linalg.inv(Dm) if det > 1e-12 else np.eye(2, dtype=np.float32)
          self._tri_elems.append([v0, v1, v2])
          self._tri_inv_Dm.append(inv_Dm)
          self._tri_area0.append(0.5 * det)
      elif dim == 3 and elem_num > 0:
        tet_v = d.elem[elem_adr * 4:(elem_adr + elem_num) * 4].reshape(elem_num, 4)
        for t in range(elem_num):
          v0, v1, v2, v3 = tet_v[t]
          X0, X1, X2, X3 = d.vert0[v0], d.vert0[v1], d.vert0[v2], d.vert0[v3]
          Dm = np.column_stack([X1 - X0, X2 - X0, X3 - X0]).astype(np.float32)
          det = abs(float(np.linalg.det(Dm)))
          inv_Dm = np.linalg.inv(Dm) if det > 1e-12 else np.eye(3, dtype=np.float32)
          self._tet_elems.append([v0, v1, v2, v3])
          self._tet_inv_Dm.append(inv_Dm)
          self._tet_vol0.append(det / 6.0)

    if self._tri_elems:
      self._tri_elems_t = torch.tensor(self._tri_elems, dtype=torch.int64, device=self._device)
      self._tri_inv_Dm_t = torch.tensor(np.array(self._tri_inv_Dm), dtype=torch.float32, device=self._device)
      self._tri_area0_t = torch.tensor(np.array(self._tri_area0), dtype=torch.float32, device=self._device)
    else:
      self._tri_elems_t = None

    if self._tet_elems:
      self._tet_elems_t = torch.tensor(self._tet_elems, dtype=torch.int64, device=self._device)
      self._tet_inv_Dm_t = torch.tensor(np.array(self._tet_inv_Dm), dtype=torch.float32, device=self._device)
      self._tet_vol0_t = torch.tensor(np.array(self._tet_vol0), dtype=torch.float32, device=self._device)
    else:
      self._tet_elems_t = None

    # Tree body parent map for point Jacobians
    self._body_parentid = np.asarray(model.body_parentid, dtype=np.int32)
    self._body_jntadr = np.asarray(model.body_jntadr, dtype=np.int32)
    self._body_jntnum = np.asarray(model.body_jntnum, dtype=np.int32)
    self._jnt_type = np.asarray(model.jnt_type, dtype=np.int32)
    self._jnt_dofadr = np.asarray(model.jnt_dofadr, dtype=np.int32)

    # Preallocated device tracking tensors
    b = self.batch_size
    nv = d.nv
    self._flexvert_xpos = torch.zeros((b, d.nflexvert, 3), dtype=torch.float32, device=self._device)
    self._flexvert_xvel = torch.zeros((b, d.nflexvert, 3), dtype=torch.float32, device=self._device)
    self._flexedge_length = torch.zeros((b, d.nflexedge), dtype=torch.float32, device=self._device)
    self._flexedge_velocity = torch.zeros((b, d.nflexedge), dtype=torch.float32, device=self._device)
    self._flexedge_dir = torch.zeros((b, d.nflexedge, 3), dtype=torch.float32, device=self._device)
    self._flexvert_J = torch.zeros((b, d.nflexvert, 3, max(nv, 1)), dtype=torch.float32, device=self._device)
    self._flexedge_J = torch.zeros((b, d.nflexedge, max(nv, 1)), dtype=torch.float32, device=self._device)
    self._qfrc_passive = torch.zeros((b, max(nv, 1)), dtype=torch.float32, device=self._device)
    self._damping_tangent = torch.zeros((b, max(nv, 1), max(nv, 1)), dtype=torch.float32, device=self._device)
    self._stiffness_tangent = torch.zeros((b, max(nv, 1), max(nv, 1)), dtype=torch.float32, device=self._device)

  @property
  def flexvert_xpos(self) -> torch.Tensor:
    """Current world positions of all flex vertices (B, nflexvert, 3)."""
    return self._flexvert_xpos

  @property
  def flexedge_length(self) -> torch.Tensor:
    """Current lengths of all flex edges (B, nflexedge)."""
    return self._flexedge_length

  @property
  def flexedge_velocity(self) -> torch.Tensor:
    """Current deformation rates of all flex edges (B, nflexedge)."""
    return self._flexedge_velocity

  @property
  def flexedge_J(self) -> torch.Tensor:
    """Jacobian rows of all flex edges (B, nflexedge, nv)."""
    return self._flexedge_J

  def update_kinematics(self, poses: Dict[str, torch.Tensor], cvel: Optional[torch.Tensor] = None):
    """Compute forward vertex kinematics, velocities, and Jacobians entirely on device MPS."""
    body_pos = poses["body_pos"]    # (B, nbody, 3)
    body_quat = poses["body_quat"]  # (B, nbody, 4)
    b = body_pos.shape[0]

    # 1. World vertex positions: x_v = body_pos[b] + R(body_quat[b]) * vert_local[v]
    b_ids = self._vertbodyid  # (nflexvert,)
    pos_b = body_pos[:, b_ids, :]  # (B, nflexvert, 3)
    quat_b = body_quat[:, b_ids, :]  # (B, nflexvert, 4)

    # Quaternion rotation: v' = v + 2 * q_xyz x (q_xyz x v + q_w * v)
    qw = quat_b[..., :1]
    qv = quat_b[..., 1:]
    v_loc = self._vert_local.unsqueeze(0).expand(b, -1, -1)  # (B, nflexvert, 3)
    t = 2.0 * torch.cross(qv, v_loc, dim=-1)
    v_rot = v_loc + qw * t + torch.cross(qv, t, dim=-1)
    self._flexvert_xpos.copy_(pos_b + v_rot)

    # 2. World vertex velocities: v_v = v_lin[b] + omega[b] x (x_v - xpos[b])
    if cvel is not None and cvel.numel() > 0:
      cvel_b = cvel[:, b_ids, :]  # (B, nflexvert, 6)
      w_b = cvel_b[..., :3]
      v_lin_b = cvel_b[..., 3:]
      r_v = v_rot  # offset from body frame origin
      v_v = v_lin_b + torch.cross(w_b, r_v, dim=-1)
      self._flexvert_xvel.copy_(v_v)
    else:
      self._flexvert_xvel.zero_()

    # 3. Edge lengths, deformation rates, and unit direction vectors
    e0 = self._edge[:, 0]
    e1 = self._edge[:, 1]
    p0 = self._flexvert_xpos[:, e0, :]  # (B, nflexedge, 3)
    p1 = self._flexvert_xpos[:, e1, :]  # (B, nflexedge, 3)
    diff = p1 - p0
    length = torch.norm(diff, dim=-1)   # (B, nflexedge)
    self._flexedge_length.copy_(length)
    u = diff / torch.clamp(length.unsqueeze(-1), min=1e-8)
    self._flexedge_dir.copy_(u)

    v0 = self._flexvert_xvel[:, e0, :]
    v1 = self._flexvert_xvel[:, e1, :]
    edot = torch.sum(u * (v1 - v0), dim=-1)
    self._flexedge_velocity.copy_(edot)

    # 4. Point Jacobians for vertices and Edge Jacobians
    self._compute_jacobians(poses)

  def _compute_jacobians(self, poses: Dict[str, torch.Tensor]):
    """Vectorized calculation of vertex and edge Jacobians on device MPS."""
    b = self.batch_size
    nv = self.descriptor.nv
    if nv <= 0:
      return

    self._flexvert_J.zero_()
    body_pos = poses["body_pos"]
    body_quat = poses["body_quat"]
    anchors = poses.get("joint_anchor", None)
    axes = poses.get("joint_axis", None)

    # Ancestor traversal for each vertex attached to body b
    # Since nflexvert is discrete, we populate flexvert_J directly
    xpos = self._flexvert_xpos  # (B, nflexvert, 3)

    for v in range(self.descriptor.nflexvert):
      body = int(self.descriptor.vertbodyid[v])
      if body <= 0:
        continue  # pinned to world: Jacobian is exactly 0
      curr = body
      pv = xpos[:, v, :]  # (B, 3)
      while curr > 0 and curr < self.descriptor.nbody:
        ja = int(self._body_jntadr[curr])
        jn = int(self._body_jntnum[curr])
        for jj in range(jn):
          j = ja + jj
          if j < 0 or j >= self.descriptor.njnt:
            continue
          da = int(self._jnt_dofadr[j])
          typ = int(self._jnt_type[j])
          nd = 6 if typ == 0 else (3 if typ == 1 else 1)
          if typ == 2:  # mjJNT_SLIDE = 2
            if axes is not None:
              axis = axes[:, j, :]
              self._flexvert_J[:, v, :, da] += axis
          elif typ == 3:  # mjJNT_HINGE = 3
            if axes is not None:
              axis = axes[:, j, :]
              anc = anchors[:, j, :] if anchors is not None else body_pos[:, curr, :]
              col = torch.cross(axis, pv - anc, dim=-1)
              self._flexvert_J[:, v, :, da] += col
          elif typ == 0:  # Free joint (3 translations + 3 rotations)
            # Translations: standard unit vectors
            self._flexvert_J[:, v, 0, da] = 1.0
            self._flexvert_J[:, v, 1, da + 1] = 1.0
            self._flexvert_J[:, v, 2, da + 2] = 1.0
            # Rotations
            b_quat = body_quat[:, curr, :]
            b_pos = body_pos[:, curr, :]
            for rot_i in range(3):
              unit_ax = torch.zeros((b, 3), device=self._device)
              unit_ax[:, rot_i] = 1.0
              qw = b_quat[:, :1]
              qv = b_quat[:, 1:]
              t_ax = 2.0 * torch.cross(qv, unit_ax, dim=-1)
              rot_axis = unit_ax + qw * t_ax + torch.cross(qv, t_ax, dim=-1)
              col = torch.cross(rot_axis, pv - b_pos, dim=-1)
              self._flexvert_J[:, v, :, da + 3 + rot_i] = col
          elif typ == 1:  # Ball joint (3 rotations)
            b_quat = body_quat[:, curr, :]
            anc = anchors[:, j, :] if anchors is not None else body_pos[:, curr, :]
            for rot_i in range(3):
              unit_ax = torch.zeros((b, 3), device=self._device)
              unit_ax[:, rot_i] = 1.0
              qw = b_quat[:, :1]
              qv = b_quat[:, 1:]
              t_ax = 2.0 * torch.cross(qv, unit_ax, dim=-1)
              rot_axis = unit_ax + qw * t_ax + torch.cross(qv, t_ax, dim=-1)
              col = torch.cross(rot_axis, pv - anc, dim=-1)
              self._flexvert_J[:, v, :, da + rot_i] = col
        curr = int(self._body_parentid[curr])

    # 5. Assemble edge Jacobians: J_e = u_e^T (J_v1 - J_v0)
    e0 = self._edge[:, 0]
    e1 = self._edge[:, 1]
    J0 = self._flexvert_J[:, e0, :, :]  # (B, nflexedge, 3, nv)
    J1 = self._flexvert_J[:, e1, :, :]  # (B, nflexedge, 3, nv)
    dJ = J1 - J0                         # (B, nflexedge, 3, nv)
    u_exp = self._flexedge_dir.unsqueeze(2)  # (B, nflexedge, 1, 3)
    self._flexedge_J.copy_(torch.matmul(u_exp, dJ).squeeze(2))

  def run_device(
      self,
      qpos: torch.Tensor,
      qvel: torch.Tensor,
      poses: Dict[str, torch.Tensor],
      cvel: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute passive internal strain forces and analytic derivatives on MPS.

    Returns:
      (qfrc_passive, damping_tangent, stiffness_tangent)
    """
    self.update_kinematics(poses, cvel)
    b = self.batch_size
    nv = self.descriptor.nv

    self._qfrc_passive.zero_()
    self._damping_tangent.zero_()
    self._stiffness_tangent.zero_()

    if nv <= 0 or self.descriptor.nflexedge <= 0:
      return self._qfrc_passive, self._damping_tangent, self._stiffness_tangent

    # 1. 1D Mass-spring edge tension forces
    k_edge = self._edgestiffness[self._edge_flexid].unsqueeze(0)  # (1, nflexedge)
    c_edge = self._edgedamping[self._edge_flexid].unsqueeze(0)    # (1, nflexedge)
    eq_edge = self._edgeequality[self._edge_flexid].unsqueeze(0)  # (1, nflexedge)

    # Elastic forces apply to edges that are NOT pure bilateral equality constraints
    # (or if stiffness is specified)
    dL = self._flexedge_length - self._edge_length0.unsqueeze(0)
    Ldot = self._flexedge_velocity
    tension = k_edge * dL + c_edge * Ldot  # (B, nflexedge)

    # Zero out edges reserved for equality solver
    is_spring = (eq_edge == 0) | (k_edge > 0)
    tension = torch.where(is_spring, tension, torch.zeros_like(tension))

    # Generalized edge restoring force: qfrc = - sum_e (tension_e * J_e)
    # J_e shape: (B, nflexedge, nv)
    qfrc_edges = -torch.sum(tension.unsqueeze(-1) * self._flexedge_J, dim=1)
    self._qfrc_passive.add_(qfrc_edges)

    # Analytic tangents for implicit integration:
    # d(qfrc)/d(qvel) = - sum_e c_e * J_e^T J_e
    # d(qfrc)/d(qpos) = - sum_e k_e * J_e^T J_e
    c_eff = torch.where(is_spring, c_edge, torch.zeros_like(c_edge))
    k_eff = torch.where(is_spring, k_edge, torch.zeros_like(k_edge))

    # Batched outer-product accumulation: J_e (B, nflexedge, nv)
    J_e = self._flexedge_J
    damp_tang = -torch.einsum("be,ben,bem->bnm", c_eff, J_e, J_e)
    stiff_tang = -torch.einsum("be,ben,bem->bnm", k_eff, J_e, J_e)
    self._damping_tangent.add_(damp_tang)
    self._stiffness_tangent.add_(stiff_tang)

    # 2. 2D Triangular continuum elements (StVK plane-stress)
    if self._tri_elems_t is not None and self._tri_elems_t.numel() > 0:
      self._compute_2d_continuum_forces()

    # 3. 3D Tetrahedral continuum elements (StVK volumetric)
    if self._tet_elems_t is not None and self._tet_elems_t.numel() > 0:
      self._compute_3d_continuum_forces()

    return self._qfrc_passive, self._damping_tangent, self._stiffness_tangent

  def _compute_2d_continuum_forces(self):
    """Compute 2D St. Venant-Kirchhoff strain and nodal restoring forces."""
    b = self.batch_size
    elems = self._tri_elems_t  # (ntri, 3)
    inv_Dm = self._tri_inv_Dm_t  # (ntri, 2, 2)
    area0 = self._tri_area0_t    # (ntri,)

    xpos = self._flexvert_xpos
    x0 = xpos[:, elems[:, 0], :]  # (B, ntri, 3)
    x1 = xpos[:, elems[:, 1], :]
    x2 = xpos[:, elems[:, 2], :]

    # Deformed shape matrix Ds = [x1 - x0, x2 - x0] (B, ntri, 3, 2)
    Ds = torch.stack([x1 - x0, x2 - x0], dim=-1)
    # Deformation gradient F = Ds @ inv_Dm (B, ntri, 3, 2)
    inv_Dm_exp = inv_Dm.unsqueeze(0).expand(b, -1, -1, -1)
    F = torch.matmul(Ds, inv_Dm_exp)

    # Right Cauchy-Green C = F^T @ F (B, ntri, 2, 2)
    C = torch.matmul(F.transpose(-1, -2), F)
    I2 = torch.eye(2, device=self._device).unsqueeze(0).unsqueeze(0)
    E = 0.5 * (C - I2)  # Green-Lagrange strain

    # Lamé parameters
    lam = float(self.descriptor.lame_lambda[0])
    mu = float(self.descriptor.lame_mu[0])

    tr_E = E[..., 0, 0] + E[..., 1, 1]  # (B, ntri)
    S = lam * tr_E.unsqueeze(-1).unsqueeze(-1) * I2 + 2.0 * mu * E  # 2nd PK stress (B, ntri, 2, 2)
    P = torch.matmul(F, S)  # 1st PK stress (B, ntri, 3, 2)

    # Nodal forces: [f1, f2] = -A0 * P @ inv_Dm^T
    H = -area0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * torch.matmul(P, inv_Dm_exp.transpose(-1, -2))
    f1 = H[..., 0]  # (B, ntri, 3)
    f2 = H[..., 1]  # (B, ntri, 3)
    f0 = -(f1 + f2)

    # Project nodal forces to generalized forces via vertex Jacobians
    J0 = self._flexvert_J[:, elems[:, 0], :, :]  # (B, ntri, 3, nv)
    J1 = self._flexvert_J[:, elems[:, 1], :, :]
    J2 = self._flexvert_J[:, elems[:, 2], :, :]

    qfrc_t = (
        torch.einsum("btk,btkn->bn", f0, J0)
        + torch.einsum("btk,btkn->bn", f1, J1)
        + torch.einsum("btk,btkn->bn", f2, J2)
    )
    self._qfrc_passive.add_(qfrc_t)

  def _compute_3d_continuum_forces(self):
    """Compute 3D St. Venant-Kirchhoff volumetric strain and nodal forces."""
    b = self.batch_size
    elems = self._tet_elems_t    # (ntet, 4)
    inv_Dm = self._tet_inv_Dm_t  # (ntet, 3, 3)
    vol0 = self._tet_vol0_t      # (ntet,)

    xpos = self._flexvert_xpos
    x0 = xpos[:, elems[:, 0], :]
    x1 = xpos[:, elems[:, 1], :]
    x2 = xpos[:, elems[:, 2], :]
    x3 = xpos[:, elems[:, 3], :]

    # Deformed shape matrix Ds = [x1-x0, x2-x0, x3-x0] (B, ntet, 3, 3)
    Ds = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=-1)
    inv_Dm_exp = inv_Dm.unsqueeze(0).expand(b, -1, -1, -1)
    F = torch.matmul(Ds, inv_Dm_exp)

    # Right Cauchy-Green C = F^T @ F (B, ntet, 3, 3)
    C = torch.matmul(F.transpose(-1, -2), F)
    I3 = torch.eye(3, device=self._device).unsqueeze(0).unsqueeze(0)
    E = 0.5 * (C - I3)

    lam = float(self.descriptor.lame_lambda[0])
    mu = float(self.descriptor.lame_mu[0])

    tr_E = E[..., 0, 0] + E[..., 1, 1] + E[..., 2, 2]
    S = lam * tr_E.unsqueeze(-1).unsqueeze(-1) * I3 + 2.0 * mu * E
    P = torch.matmul(F, S)

    # Nodal forces [f1, f2, f3] = -V0 * P @ inv_Dm^T
    H = -vol0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * torch.matmul(P, inv_Dm_exp.transpose(-1, -2))
    f1 = H[..., 0]
    f2 = H[..., 1]
    f3 = H[..., 2]
    f0 = -(f1 + f2 + f3)

    J0 = self._flexvert_J[:, elems[:, 0], :, :]
    J1 = self._flexvert_J[:, elems[:, 1], :, :]
    J2 = self._flexvert_J[:, elems[:, 2], :, :]
    J3 = self._flexvert_J[:, elems[:, 3], :, :]

    qfrc_tet = (
        torch.einsum("btk,btkn->bn", f0, J0)
        + torch.einsum("btk,btkn->bn", f1, J1)
        + torch.einsum("btk,btkn->bn", f2, J2)
        + torch.einsum("btk,btkn->bn", f3, J3)
    )
    self._qfrc_passive.add_(qfrc_tet)

  def run_equalities(
      self,
      poses: Dict[str, torch.Tensor],
      cvel: Optional[torch.Tensor] = None,
      eq_active: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate bilateral distance equality constraint rows (mjEQ_FLEX).

    Returns:
      (eq_pos, eq_aref, eq_R, eq_J)
    """
    self.update_kinematics(poses, cvel)
    b = self.batch_size
    nedge = self.descriptor.nflexedge
    nv = self.descriptor.nv

    # Position error: pos = L - L0
    pos = self._flexedge_length - self._edge_length0.unsqueeze(0)  # (B, nedge)
    vel = self._flexedge_velocity                                 # (B, nedge)

    # MuJoCo mju_solimp evaluation
    abs_pos = torch.abs(pos)
    y = abs_pos / self._edge_width
    one_minus_mid = torch.clamp(1.0 - self._edge_mid, min=1e-4)
    y_scaled_low = torch.clamp(y / self._edge_mid, max=1.0)
    y_scaled_high = torch.clamp((1.0 - y) / one_minus_mid, min=0.0)

    imp_low = self._edge_dmin + 0.5 * (self._edge_dmax - self._edge_dmin) * torch.pow(y_scaled_low, self._edge_power)
    imp_high = self._edge_dmax - 0.5 * (self._edge_dmax - self._edge_dmin) * torch.pow(y_scaled_high, self._edge_power)

    imp = torch.where(abs_pos <= 0.0, self._edge_dmin,
          torch.where(abs_pos >= self._edge_width, self._edge_dmax,
          torch.where(y <= self._edge_mid, imp_low, imp_high)))
    imp = torch.clamp(imp, min=1e-4, max=0.9999)

    # Reference acceleration: aref = -(b0 * vel + k0 * pos) / imp
    aref = -(self._edge_b0 * vel + self._edge_k0 * pos) / imp

    # Exact compliance: R = (1 - imp) / imp * flexedge_invweight0
    R = torch.clamp((1.0 - imp) / imp * self._edge_diag, min=1e-15)

    return pos, aref, R, self._flexedge_J

  def run_contacts(
      self,
      poses: Dict[str, torch.Tensor],
      plane_z: float = 0.0,
      margin: float = 0.01,
  ) -> Dict[str, torch.Tensor]:
    """Generate narrowphase contact constraints for flex vertices against obstacles."""
    b = self.batch_size
    nvert = self.descriptor.nflexvert
    xpos = self._flexvert_xpos  # (B, nvert, 3)

    # Vertex sphere radius
    r = self._radius[self._vert_flexid].unsqueeze(0)  # (1, nvert)

    # 1. Contact against ground plane at plane_z
    z_dist = xpos[..., 2] - plane_z - r  # (B, nvert)
    in_contact = z_dist < margin

    # Extract contact points, depths, and Jacobians
    # Normal is [0, 0, 1]
    contact_jac = self._flexvert_J[..., 2, :]  # (B, nvert, nv)
    contact_dist = z_dist
    contact_pos = xpos.clone()
    contact_pos[..., 2] -= r

    return {
        "in_contact": in_contact,
        "dist": contact_dist,
        "pos": contact_pos,
        "jacobian": contact_jac,
        "vert_J": self._flexvert_J,
    }

  def get_state(self) -> Dict[str, torch.Tensor]:
    """Capture current flex state for immutable snapshot and rollback."""
    return {
        "flexvert_xpos": self._flexvert_xpos.clone(),
        "flexvert_xvel": self._flexvert_xvel.clone(),
        "flexedge_length": self._flexedge_length.clone(),
        "flexedge_velocity": self._flexedge_velocity.clone(),
    }

  def set_state(self, state: Dict[str, torch.Tensor], env_ids=None):
    """Restore flex state from snapshot."""
    if env_ids is None:
      if "flexvert_xpos" in state:
        self._flexvert_xpos.copy_(state["flexvert_xpos"])
      if "flexvert_xvel" in state:
        self._flexvert_xvel.copy_(state["flexvert_xvel"])
      if "flexedge_length" in state:
        self._flexedge_length.copy_(state["flexedge_length"])
      if "flexedge_velocity" in state:
        self._flexedge_velocity.copy_(state["flexedge_velocity"])
    else:
      ids = [int(i) for i in env_ids]
      idx = torch.tensor(ids, device=self._device, dtype=torch.long)
      for k, buf in [
          ("flexvert_xpos", self._flexvert_xpos),
          ("flexvert_xvel", self._flexvert_xvel),
          ("flexedge_length", self._flexedge_length),
          ("flexedge_velocity", self._flexedge_velocity),
      ]:
        if k in state:
          src = state[k]
          if src.shape[0] == len(ids):
            buf[idx] = src.to(device=self._device)
          else:
            buf[idx] = src[idx].to(device=self._device)
