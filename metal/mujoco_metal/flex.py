# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned MuJoCo 3.10 flex material primitives for the native backend.

This manager lowers compiled simplex stretch metrics and ordinary triangular
shell bending records. The signed interpolation modes, interpolated-shell
bending, full attachment Jacobian derivatives, and general flex contact are
not implemented here; callers must retain their existing admission guards for
those modes. Edge equalities, contact assembly and lifecycle are integrated by
the owning simulation stages.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
import torch

_MATERIAL_SHADER = Path(__file__).parent / "shaders" / "flex_material.metal"


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
  edgeflap: np.ndarray        # (nflexedge, 2) int32, local flap vertices
  edgeadr: np.ndarray         # (nflex,) int32
  edgenum: np.ndarray         # (nflex,) int32
  edgeequality: np.ndarray    # (nflex,) int32
  edgestiffness: np.ndarray   # (nflex,) float32
  edgedamping: np.ndarray     # (nflex,) float32
  edge_length0: np.ndarray    # (nflexedge,) float32
  edge_invweight0: np.ndarray # (nflexedge,) float32
  elem: np.ndarray            # (nflexelem_total,) int32
  elemadr: np.ndarray         # (nflex,) int32
  elemdataadr: np.ndarray     # (nflex,) int32, address into flex_elem
  elemnum: np.ndarray         # (nflex,) int32
  elemedge: np.ndarray        # (nflexelemedge,) int32, local edge ids per element
  elemedgeadr: np.ndarray     # (nflex,) int32
  stiffness: np.ndarray       # (nflexstiffness,) float32, pinned packed matrices
  stiffnessadr: np.ndarray    # (nflex,) int32
  interp: np.ndarray          # (nflex,) int32; signed interpolation order
  damping: np.ndarray         # (nflex,) float32, pinned Rayleigh coefficient
  bending: np.ndarray         # (nflexbending,) float32, pinned bending data
  bendingadr: np.ndarray      # (nflex,) int32
  cellnum: np.ndarray         # (nflex, 3) int32
  node: np.ndarray            # (nflexnode, 3) float32
  node0: np.ndarray           # (nflexnode, 3) float32
  nodebodyid: np.ndarray      # (nflexnode,) int32
  nodeadr: np.ndarray         # (nflex,) int32
  nodenum: np.ndarray         # (nflex,) int32
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
  young: np.ndarray           # (nflex,) float32; raw compiled metric is authoritative
  poisson: np.ndarray         # (nflex,) float32; raw compiled metric is authoritative
  lame_lambda: np.ndarray     # (nflex,) float32; raw compiled metric is authoritative
  lame_mu: np.ndarray         # (nflex,) float32; raw compiled metric is authoritative


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
  edge = np.array(model.flex_edge, dtype=np.int32, order="C", copy=True).reshape(nflexedge, 2)
  edgeadr = _frozen(model.flex_edgeadr, np.int32)
  edgenum = _frozen(model.flex_edgenum, np.int32)
  vertadr = _frozen(model.flex_vertadr, np.int32)
  for f in range(nflex):
    start, count = int(edgeadr[f]), int(edgenum[f])
    edge[start:start + count] += int(vertadr[f])
  edge = _frozen(edge, np.int32)
  edgeequality = _frozen(model.flex_edgeequality, np.int32)
  edgeflap = np.array(model.flex_edgeflap, dtype=np.int32, order="C", copy=True).reshape(nflexedge, 2)
  for f in range(nflex):
    start, count = int(edgeadr[f]), int(edgenum[f])
    mask = edgeflap[start:start + count] >= 0
    edgeflap[start:start + count][mask] += int(vertadr[f])
  edgeflap = _frozen(edgeflap, np.int32)
  edgestiffness = _frozen(model.flex_edgestiffness, np.float32)
  edgedamping = _frozen(model.flex_edgedamping, np.float32)
  edge_length0 = _frozen(model.flexedge_length0, np.float32)
  edge_invweight0 = _frozen(model.flexedge_invweight0, np.float32)
  elem = _frozen(model.flex_elem, np.int32)
  elemadr = _frozen(model.flex_elemadr, np.int32)
  elemdataadr = _frozen(model.flex_elemdataadr, np.int32)
  elemnum = _frozen(model.flex_elemnum, np.int32)
  elemedge = _frozen(model.flex_elemedge, np.int32)
  elemedgeadr = _frozen(model.flex_elemedgeadr, np.int32)
  stiffness = _frozen(model.flex_stiffness, np.float32)
  stiffnessadr = _frozen(model.flex_stiffnessadr, np.int32)
  interp = _frozen(model.flex_interp, np.int32)
  damping = _frozen(model.flex_damping, np.float32)
  bending = _frozen(model.flex_bending, np.float32)
  bendingadr = _frozen(model.flex_bendingadr, np.int32)
  cellnum = _frozen(model.flex_cellnum, np.int32).reshape(nflex, 3)
  node = _frozen(model.flex_node, np.float32).reshape(-1, 3)
  node0 = _frozen(model.flex_node0, np.float32).reshape(-1, 3)
  nodebodyid = _frozen(model.flex_nodebodyid, np.int32)
  nodeadr = _frozen(model.flex_nodeadr, np.int32)
  nodenum = _frozen(model.flex_nodenum, np.int32)
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
      edgeflap=edgeflap,
      edgeadr=edgeadr,
      edgenum=edgenum,
      edgeequality=edgeequality,
      edgestiffness=edgestiffness,
      edgedamping=edgedamping,
      edge_length0=edge_length0,
      edge_invweight0=edge_invweight0,
      elem=elem,
      elemadr=elemadr,
      elemdataadr=elemdataadr,
      elemnum=elemnum,
      elemedge=elemedge,
      elemedgeadr=elemedgeadr,
      stiffness=stiffness,
      stiffnessadr=stiffnessadr,
      interp=interp,
      damping=damping,
      bending=bending,
      bendingadr=bendingadr,
      cellnum=cellnum,
      node=node,
      node0=node0,
      nodebodyid=nodebodyid,
      nodeadr=nodeadr,
      nodenum=nodenum,
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
      young=_frozen(np.zeros(nflex, dtype=np.float32), np.float32),
      poisson=_frozen(np.zeros(nflex, dtype=np.float32), np.float32),
      lame_lambda=_frozen(np.zeros(nflex, dtype=np.float32), np.float32),
      lame_mu=_frozen(np.zeros(nflex, dtype=np.float32), np.float32),
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
    self._disableflags = int(model.opt.disableflags)

    # MuJoCo 3.10 compiles simplex elasticity into 21 upper-triangle values
    # per element. Lower that packed matrix and the corresponding element and
    # edge addresses verbatim; no coefficient is reinterpreted as Young's E.
    stretch_vertex, stretch_edge, stretch_metric = [], [], []
    stretch_edge_count, stretch_vertex_count, stretch_damping = [], [], []
    for f in range(d.nflex):
      dim = int(d.dim[f])
      kadr = int(d.stiffnessadr[f])
      if dim not in (2, 3) or int(d.interp[f]) != 0 or kadr < 0:
        continue
      nvert, nedge = dim + 1, (3 if dim == 2 else 6)
      for t in range(int(d.elemnum[f])):
        vstart = int(d.elemdataadr[f]) + t * nvert
        estart = int(d.elemedgeadr[f]) + t * nedge
        verts = int(model.flex_vertadr[f]) + np.asarray(
            d.elem[vstart:vstart + nvert], dtype=np.int32)
        edges = int(d.edgeadr[f]) + np.asarray(
            d.elemedge[estart:estart + nedge], dtype=np.int32)
        packed = np.asarray(d.stiffness[kadr + 21*t:kadr + 21*t + 21], dtype=np.float32)
        if len(packed) != 21:
          raise ValueError("compiled flex element stiffness is truncated")
        metric = np.zeros((6, 6), dtype=np.float32)
        index = 0
        for i in range(nedge):
          for j in range(i, nedge):
            metric[i, j] = metric[j, i] = packed[index]
            index += 1
        if not np.all(np.isfinite(metric)):
          raise ValueError("compiled flex stiffness must be finite and representable")
        stretch_vertex.append(np.pad(verts, (0, 4 - nvert), constant_values=-1))
        stretch_edge.append(np.pad(edges, (0, 6 - nedge), constant_values=-1))
        stretch_metric.append(metric)
        stretch_edge_count.append(nedge)
        stretch_vertex_count.append(nvert)
        stretch_damping.append(float(d.damping[f]))
    self._stretch_count = len(stretch_vertex)

    def static_tensor(values, shape, dtype):
      array = np.asarray(values, dtype=np.int32 if dtype == torch.int32 else np.float32)
      array = array.reshape(shape) if array.size else np.zeros(shape, dtype=array.dtype)
      return torch.as_tensor(array, dtype=dtype, device=self._device)

    self._stretch_vertex = static_tensor(stretch_vertex, (self._stretch_count, 4), torch.int32)
    self._stretch_edge = static_tensor(stretch_edge, (self._stretch_count, 6), torch.int32)
    self._stretch_metric = static_tensor(stretch_metric, (self._stretch_count, 6, 6), torch.float32)
    self._stretch_edge_count = static_tensor(stretch_edge_count, (self._stretch_count,), torch.int32)
    self._stretch_vertex_count = static_tensor(stretch_vertex_count, (self._stretch_count,), torch.int32)
    # Immutable host descriptors are used only to determine static loop bounds.
    # Never convert a device tensor scalar to Python in a native execution path.
    self._stretch_edge_count_host = tuple(stretch_edge_count)
    self._stretch_vertex_count_host = tuple(stretch_vertex_count)
    self._stretch_damping = static_tensor(stretch_damping, (self._stretch_count,), torch.float32)
    self._stretch_damping_host = tuple(stretch_damping)
    self._stretch_edge_pairs = [
        ([(1, 2), (2, 0), (0, 1)] if n == 3 else
         [(0, 1), (1, 2), (2, 0), (2, 3), (0, 3), (1, 3)])
        for n in stretch_edge_count]
    self._material_dummy_i = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._material_dummy_f = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._material_shader = (
        torch.mps.compile_shader(_MATERIAL_SHADER.read_text())
        if self._device.type == "mps" and self._stretch_count else None)

    bend_vertex, bend_data, bend_damping = [], [], []
    for f in range(d.nflex):
      if int(d.dim[f]) != 2 or int(d.bendingadr[f]) < 0:
        continue
      edge_start, edge_num = int(d.edgeadr[f]), int(d.edgenum[f])
      bend_start = int(d.bendingadr[f])
      vert_start = int(model.flex_vertadr[f])
      for e in range(edge_num):
        flap = d.edgeflap[edge_start + e]
        if int(flap[1]) < 0:
          continue
        pair = d.edge[edge_start + e]
        start = bend_start + 17 * e
        coeff = np.asarray(d.bending[start:start + 17], dtype=np.float32)
        if len(coeff) != 17:
          raise ValueError("compiled flex bending data is truncated")
        bend_vertex.append([int(pair[0]), int(pair[1]), int(flap[0]), int(flap[1])])
        bend_data.append(coeff)
        bend_damping.append(float(d.damping[f]))
    self._bend_count = len(bend_vertex)
    self._bend_vertex = static_tensor(bend_vertex, (self._bend_count, 4), torch.int32)
    self._bend_data = static_tensor(bend_data, (self._bend_count, 17), torch.float32)
    self._bend_damping = static_tensor(bend_damping, (self._bend_count,), torch.float32)
    self._bend_damping_host = tuple(bend_damping)
    self._bend_shader = (
        torch.mps.compile_shader(_MATERIAL_SHADER.read_text())
        if self._device.type == "mps" and self._bend_count else None)

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

    self._has_2d = bool(np.any(d.dim == 2))
    self._has_3d = bool(np.any(d.dim == 3))

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
    self._node_local = torch.tensor(np.array(d.node, copy=True), dtype=torch.float32, device=self._device)
    self._node0 = torch.tensor(np.array(d.node0, copy=True), dtype=torch.float32, device=self._device)
    self._nodebodyid = torch.tensor(np.array(d.nodebodyid, copy=True), dtype=torch.int64, device=self._device)
    self._node_xpos = torch.zeros((b, len(d.nodebodyid), 3), dtype=torch.float32, device=self._device)
    self._node_xvel = torch.zeros_like(self._node_xpos)
    self._node_J = torch.zeros((b, len(d.nodebodyid), 3, max(nv, 1)), dtype=torch.float32, device=self._device)
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

    if len(self.descriptor.nodebodyid):
      nbody_ids = self._nodebodyid
      nquat = body_quat[:, nbody_ids, :]
      nlocal = self._node_local.unsqueeze(0).expand(b, -1, -1)
      nqv, nqw = nquat[..., 1:], nquat[..., :1]
      nt = 2.0 * torch.cross(nqv, nlocal, dim=-1)
      nrot = nlocal + nqw * nt + torch.cross(nqv, nt, dim=-1)
      self._node_xpos.copy_(body_pos[:, nbody_ids, :] + nrot)
      if cvel is not None and cvel.numel() > 0:
        ncvel = cvel[:, nbody_ids, :]
        nvel = ncvel[..., 3:] + torch.cross(ncvel[..., :3], nrot, dim=-1)
        self._node_xvel.copy_(nvel)
      else:
        self._node_xvel.zero_()

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
    self._compute_point_jacobian(
        self._vertbodyid, self._flexvert_xpos, self._flexvert_J, poses)
    if len(self.descriptor.nodebodyid):
      self._compute_point_jacobian(
          self._nodebodyid, self._node_xpos, self._node_J, poses)

    # 5. Assemble edge Jacobians: J_e = u_e^T (J_v1 - J_v0)
    e0 = self._edge[:, 0]
    e1 = self._edge[:, 1]
    J0 = self._flexvert_J[:, e0, :, :]  # (B, nflexedge, 3, nv)
    J1 = self._flexvert_J[:, e1, :, :]  # (B, nflexedge, 3, nv)
    dJ = J1 - J0                         # (B, nflexedge, 3, nv)
    u_exp = self._flexedge_dir.unsqueeze(2)  # (B, nflexedge, 1, 3)
    self._flexedge_J.copy_(torch.matmul(u_exp, dJ).squeeze(2))

  def _compute_point_jacobian(self, body_ids, points, output, poses):
    b, nv = self.batch_size, self.descriptor.nv
    if nv <= 0:
      return
    output.zero_()
    body_pos = poses["body_pos"]
    body_quat = poses["body_quat"]
    anchors = poses.get("joint_anchor")
    axes = poses.get("joint_axis")
    for index, body_value in enumerate(self.descriptor.vertbodyid if output is self._flexvert_J else self.descriptor.nodebodyid):
      body = int(body_value)
      if body <= 0:
        continue
      curr = body
      point = points[:, index, :]
      while curr > 0 and curr < self.descriptor.nbody:
        ja, jn = int(self._body_jntadr[curr]), int(self._body_jntnum[curr])
        for j in range(ja, ja + jn):
          if j < 0 or j >= self.descriptor.njnt:
            continue
          da, typ = int(self._jnt_dofadr[j]), int(self._jnt_type[j])
          if typ == 2 and axes is not None:
            output[:, index, :, da] += axes[:, j, :]
          elif typ == 3 and axes is not None:
            anc = anchors[:, j, :] if anchors is not None else body_pos[:, curr, :]
            output[:, index, :, da] += torch.cross(axes[:, j, :], point - anc, dim=-1)
          elif typ in (0, 1):
            b_quat = body_quat[:, curr, :]
            anc = body_pos[:, curr, :] if typ == 0 else (
                anchors[:, j, :] if anchors is not None else body_pos[:, curr, :])
            for rot_i in range(3):
              unit_ax = torch.zeros((b, 3), device=self._device)
              unit_ax[:, rot_i] = 1.0
              qw, qv = b_quat[:, :1], b_quat[:, 1:]
              t_ax = 2.0 * torch.cross(qv, unit_ax, dim=-1)
              rot_axis = unit_ax + qw * t_ax + torch.cross(qv, t_ax, dim=-1)
              output[:, index, :, da + rot_i + (3 if typ == 0 else 0)] = torch.cross(
                  rot_axis, point - anc, dim=-1)
            if typ == 0:
              output[:, index, 0, da] = 1.0
              output[:, index, 1, da + 1] = 1.0
              output[:, index, 2, da + 2] = 1.0
        curr = int(self._body_parentid[curr])

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

    # 2. Pinned 2D membrane and 3D tetrahedral material forces.
    if self._stretch_count:
      self._compute_pinned_stretch()
    if self._bend_count:
      self._compute_pinned_bend()

    return self._qfrc_passive, self._damping_tangent, self._stiffness_tangent

  def _compute_pinned_stretch(self):
    """Evaluate MuJoCo 3.10's compiled simplex stiffness representation."""
    b, nv = self.batch_size, self.descriptor.nv
    d = self.descriptor
    if self._device.type == "mps":
      dims = torch.tensor(
          [b, nv, self._stretch_count, 4, d.nflexedge, d.nflexvert],
          dtype=torch.int32, device=self._device)
      elem_force = torch.empty(
          (b, self._stretch_count, nv), dtype=torch.float32, device=self._device)
      self._material_shader.flex_stretch_force(
          self._flexvert_xpos.reshape(-1), self._flexedge_length.reshape(-1),
          self._flexedge_velocity.reshape(-1), self._edge_length0,
          self._flexvert_J.reshape(-1), self._stretch_vertex.reshape(-1),
          self._stretch_edge.reshape(-1), self._stretch_metric.reshape(-1),
          self._stretch_damping, self._stretch_edge_count,
          self._stretch_vertex_count, dims,
          torch.as_tensor([d.timestep], dtype=torch.float32, device=self._device),
          elem_force.reshape(-1),
          threads=(b * self._stretch_count * nv,), group_size=(128,))
      self._qfrc_passive.add_(elem_force.sum(dim=1))
    else:
      # Identical arithmetic used by the CPU test backend. Production MPS
      # evaluates this element-local loop in flex_material.metal.
      qforce = torch.zeros((b, self._stretch_count, nv), dtype=torch.float32, device=self._device)
      for e in range(self._stretch_count):
        ne = self._stretch_edge_count_host[e]
        nve = self._stretch_vertex_count_host[e]
        verts = self._stretch_vertex[e, :nve].long()
        edge_ids = self._stretch_edge[e, :ne].long()
        metric = self._stretch_metric[e, :ne, :ne]
        pos = self._flexvert_xpos[:, verts, :]
        edgevec = torch.stack(
            [pos[:, a, :] - pos[:, c, :] for a, c in self._stretch_edge_pairs[e]], dim=1)
        lengths = self._flexedge_length[:, edge_ids]
        rates = self._flexedge_velocity[:, edge_ids]
        rest = self._edge_length0[edge_ids].unsqueeze(0)
        previous = lengths - rates * d.timestep
        kd = float(self._stretch_damping[e]) / d.timestep if d.timestep > 0 else 0.0
        elong = lengths.square() - rest.square() + (lengths.square() - previous.square()) * kd
        coeff = elong @ metric
        local = torch.zeros((b, nve, 3), dtype=torch.float32, device=self._device)
        for j, (a, c) in enumerate(self._stretch_edge_pairs[e]):
          local[:, a, :] -= coeff[:, j:j+1] * edgevec[:, j, :]
          local[:, c, :] += coeff[:, j:j+1] * edgevec[:, j, :]
        jac = self._flexvert_J[:, verts, :, :]
        qforce[:, e, :] = torch.einsum("bvk,bvkn->bn", local, jac)
      self._qfrc_passive.add_(qforce.sum(dim=1))

    # Exact local element derivatives of the pinned squared-length law. The
    # formula uses the point Jacobians already lowered for every vertex, so it
    # stays device resident and includes geometric terms from the moving edge
    # directions. Flexcomp's per-vertex slide-DOF Jacobians are constant; for
    # rotational attachments their Jacobian derivative is outside this local
    # element matrix and is accounted for by the enclosing rigid-body stage.
    for e in range(self._stretch_count):
      ne = self._stretch_edge_count_host[e]
      nve = self._stretch_vertex_count_host[e]
      verts = self._stretch_vertex[e, :nve].long()
      edge_ids = self._stretch_edge[e, :ne].long()
      metric = self._stretch_metric[e, :ne, :ne]
      pairs = self._stretch_edge_pairs[e]
      x = self._flexvert_xpos[:, verts, :]
      vel = self._flexvert_xvel[:, verts, :]
      jv = self._flexvert_J[:, verts, :, :]
      diffs = torch.stack([x[:, a] - x[:, c] for a, c in pairs], dim=1)
      vdiffs = torch.stack([vel[:, a] - vel[:, c] for a, c in pairs], dim=1)
      jrel = torch.stack([jv[:, a] - jv[:, c] for a, c in pairs], dim=1)
      lengths = self._flexedge_length[:, edge_ids]
      rates = self._flexedge_velocity[:, edge_ids]
      unit = diffs / torch.clamp(lengths.unsqueeze(-1), min=1e-10)
      je = torch.einsum("bei,bein->ben", unit, jrel)
      jrate = torch.einsum(
          "bei,bein->ben",
          (vdiffs - unit * rates.unsqueeze(-1)) /
          torch.clamp(lengths.unsqueeze(-1), min=1e-10), jrel)
      kd = self._stretch_damping_host[e] / d.timestep if d.timestep > 0 else 0.0
      previous = lengths - rates * d.timestep
      elong = lengths.square() - self._edge_length0[edge_ids].square().unsqueeze(0)
      elong = elong + (lengths.square() - previous.square()) * kd
      coeff = elong @ metric

      # Derivative of each projected edge gradient, Jrel' * Jrel.
      dgrad = torch.einsum("bein,beim->benm", jrel, jrel)
      geom = torch.einsum("be,benm->bnm", coeff, dgrad)
      if kd:
        delong = (
            2.0 * (lengths + kd * rates * d.timestep)[:, :, None] * je
            + 2.0 * kd * d.timestep *
            (lengths - rates * d.timestep)[:, :, None] * jrate)
      else:
        delong = 2.0 * lengths[:, :, None] * je
      generalized_edge_gradient = lengths[:, :, None] * je
      coupled = torch.einsum("ij,bim->bjm", metric, delong)
      geom = geom + torch.einsum(
          "ben,bem->bnm", generalized_edge_gradient, coupled)
      self._stiffness_tangent.add_(-geom)

      damping = self._stretch_damping_host[e]
      if damping:
        weight = 2.0 * damping * (lengths - rates * d.timestep)
        damp_tangent = -torch.einsum(
            "bjn,ij,bi,bim,bj->bnm", je, metric, weight, je, lengths)
        self._damping_tangent.add_(damp_tangent)

  def _compute_pinned_bend(self):
    """Evaluate the compiled 17-scalar per-edge 2D shell bending record."""
    b, nv = self.batch_size, self.descriptor.nv
    d = self.descriptor
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    damper = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    if self._device.type == "mps":
      dims = torch.tensor(
          [b, nv, self._bend_count, d.nflexvert],
          dtype=torch.int32, device=self._device)
      element_force = torch.empty(
          (b, self._bend_count, nv), dtype=torch.float32, device=self._device)
      # Keep pinned spring/damper enable bits in the pre-lowered coefficients.
      values = self._bend_data.clone()
      if not spring:
        values[:, :16] = 0
        values[:, 16] = 0
      damping = self._bend_damping if damper else self._material_dummy_f.expand_as(self._bend_damping)
      self._bend_shader.flex_bend_force(
          self._flexvert_xpos.reshape(-1), self._flexvert_xvel.reshape(-1),
          self._flexvert_J.reshape(-1), self._bend_vertex.reshape(-1),
          values.reshape(-1), damping, dims, element_force.reshape(-1),
          threads=(b * self._bend_count * nv,), group_size=(128,))
      self._qfrc_passive.add_(element_force.sum(dim=1))

    # The matrix contribution has a constant-Jacobian projected derivative.
    # The curved-rest correction and derivatives of attachment Jacobians are
    # not included, so this is not the complete configuration tangent for
    # general articulated/off-center attachments.
    if nv:
      for e in range(self._bend_count):
        if not spring and not damper:
          continue
        verts = self._bend_vertex[e].long()
        jac = self._flexvert_J[:, verts, :, :]
        matrix = self._bend_data[e, :16].reshape(4, 4)
        projected = torch.einsum("bikn,ij,bjkm->bnm", jac, matrix, jac)
        if spring:
          self._stiffness_tangent.sub_(projected)
        damping = self._bend_damping_host[e]
        if damper and damping != 0.0:
          self._damping_tangent.add_(-damping * projected)

    if self._device.type != "mps":
      for e in range(self._bend_count):
        verts = self._bend_vertex[e].long()
        x = self._flexvert_xpos[:, verts, :]
        v = self._flexvert_xvel[:, verts, :]
        ed0 = x[:, 1] - x[:, 0]
        ed1 = x[:, 2] - x[:, 0]
        ed2 = x[:, 3] - x[:, 0]
        ref = torch.stack([
            -torch.cross(ed1, ed2, dim=-1) - torch.cross(ed2, ed0, dim=-1) - torch.cross(ed0, ed1, dim=-1),
            torch.cross(ed1, ed2, dim=-1),
            torch.cross(ed2, ed0, dim=-1),
            torch.cross(ed0, ed1, dim=-1),
        ], dim=1)
        matrix = self._bend_data[e, :16].reshape(4, 4)
        total = torch.zeros_like(x)
        if spring:
          total = total + torch.einsum("ij,bjk->bik", matrix, x)
          total = total + self._bend_data[e, 16] * ref
        if damper:
          total = total + self._bend_damping_host[e] * torch.einsum(
              "ij,bjk->bik", matrix, v)
        jac = self._flexvert_J[:, verts, :, :]
        self._qfrc_passive.add_(-torch.einsum("bik,bikn->bn", total, jac))

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
