# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned MuJoCo 3.10 flex material primitives for the native backend.

Compiled simplex and interpolated volume forces and ordinary triangular shell
bending records are lowered here. Interpolated shell bending has a CPU tensor
oracle; full native interpolation tangents and general flex contact remain
separate qualification gates. Edge equalities, contact assembly and lifecycle
are integrated by the owning simulation stages.
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
  centered: np.ndarray        # (nflex,) bool
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
  centered = _frozen(model.flex_centered, np.bool_)
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
      centered=centered,
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
    stretch_vertex_host = []
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
        stretch_vertex_host.append(tuple(map(int, verts)))
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
    self._stretch_vertex_host = tuple(stretch_vertex_host)
    self._material_dummy_i = torch.zeros(1, dtype=torch.int32, device=self._device)
    self._material_dummy_f = torch.zeros(1, dtype=torch.float32, device=self._device)
    self._material_shader = (
        torch.mps.compile_shader(_MATERIAL_SHADER.read_text())
        if self._device.type == "mps" and (self._stretch_count or
                                           np.any(d.interp != 0)) else None)

    bend_vertex, bend_data, bend_damping = [], [], []
    bend_dofadr, bend_dofnum = [], []
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
        bodies = [int(d.vertbodyid[v]) for v in bend_vertex[-1]]
        bend_dofadr.append([int(model.body_dofadr[body]) for body in bodies])
        bend_dofnum.append([int(model.body_dofnum[body]) for body in bodies])
        bend_data.append(coeff)
        bend_damping.append(float(d.damping[f]))
    self._bend_count = len(bend_vertex)
    self._bend_vertex = static_tensor(bend_vertex, (self._bend_count, 4), torch.int32)
    self._bend_data = static_tensor(bend_data, (self._bend_count, 17), torch.float32)
    self._bend_damping = static_tensor(bend_damping, (self._bend_count,), torch.float32)
    self._bend_damping_host = tuple(bend_damping)
    self._bend_vertex_host = tuple(tuple(map(int, row)) for row in bend_vertex)
    self._bend_dofadr = static_tensor(bend_dofadr, (self._bend_count, 4), torch.int32)
    self._bend_dofnum = static_tensor(bend_dofnum, (self._bend_count, 4), torch.int32)
    self._bend_dofadr_host = tuple(tuple(map(int, row)) for row in bend_dofadr)
    self._bend_dofnum_host = tuple(tuple(map(int, row)) for row in bend_dofnum)
    self._bend_shader = (
        torch.mps.compile_shader(_MATERIAL_SHADER.read_text())
        if self._device.type == "mps" and self._bend_count else None)

    # Lower the full compiled interpolation element matrices and structured
    # node gather maps. These records are shared by order-1/order-2 volume
    # elements and signed shell faces; the runtime routine below currently
    # uses them on the CPU test backend while the matching MSL kernel lands.
    interp_nodes, interp_matrix, interp_grad, interp_npe, interp_axes = [], [], [], [], []
    interp_damping, interp_owner = [], []
    shell_face_nodes, shell_face_axes, shell_face_order = [], [], []
    shell_bend_records = []
    for f in range(d.nflex):
      signed_order = int(d.interp[f])
      kadr = int(d.stiffnessadr[f])
      if signed_order == 0 or kadr < 0:
        continue
      order = abs(signed_order)
      cx, cy, cz = map(int, d.cellnum[f])
      shell = signed_order < 0
      npe = (order + 1) ** (2 if shell else 3)
      has_stretch = (d.stiffness[kadr] != 0 and int(d.edgeequality[f]) != 3)
      if shell:
        nfe = 2 * (cy * cz + cx * cz + cx * cy)
        local_maps = []
        axes_maps = []
        sizes = [cy * cz, cy * cz, cx * cz, cx * cz, cx * cy, cx * cy]
        normals = [0, 0, 1, 1, 2, 2]
        counts = [cz, cz, cx, cx, cy, cy]
        fixed = [0, cx * order, 0, cy * order, 0, cz * order]
        offsets = np.cumsum([0] + sizes)
        ny, nz = cy * order + 1, cz * order + 1
        for fe in range(nfe):
          face = next(fi for fi in range(6) if fe < offsets[fi + 1])
          within = fe - int(offsets[face])
          na0, na1, normal = (normals[face] + 1) % 3, (normals[face] + 2) % 3, normals[face]
          q0, q1 = within // counts[face], within % counts[face]
          ids = []
          for l0 in range(order + 1):
            for l1 in range(order + 1):
              g = [0, 0, 0]
              g[normal] = fixed[face]
              g[na0], g[na1] = q0 * order + l0, q1 * order + l1
              ids.append(int(d.nodeadr[f]) + g[0] * ny * nz + g[1] * nz + g[2])
          local_maps.append(ids)
          axes_maps.append((na0, na1, normal))
        face_offset = len(shell_face_nodes)
        shell_face_nodes.extend(local_maps)
        shell_face_axes.extend(axes_maps)
        shell_face_order.extend([order] * nfe)
        badr = int(d.bendingadr[f])
        if badr >= 0:
          nbe = int(d.bending[badr])
          for e in range(nbe):
            vals = np.asarray(d.bending[badr + 1 + 10*e:badr + 1 + 10*(e+1)], dtype=np.float32)
            if vals.size != 10:
              raise ValueError("compiled interpolated shell bending record is truncated")
            shell_bend_records.append((
                face_offset + int(vals[0]), face_offset + int(vals[1]),
                vals[2:6].copy(), float(vals[6]), vals[7:10].copy(),
                float(d.damping[f])))
      if not has_stretch:
        continue
      else:
        nfe = cx * cy * cz
        local_maps = []
        axes_maps = []
        ny, nz = cy * order + 1, cz * order + 1
        for fe in range(nfe):
          ci, cj, ck = fe // (cy * cz), (fe // cz) % cy, fe % cz
          ids = []
          for li in range(order + 1):
            for lj in range(order + 1):
              for lk in range(order + 1):
                ids.append(int(d.nodeadr[f]) + (ci * order + li) * ny * nz
                            + (cj * order + lj) * nz + ck * order + lk)
          local_maps.append(ids)
          axes_maps.append((0, 1, 2))
      npe3 = 3 * npe
      stride = npe3 * npe3
      grad_coeff = self._interp_shape_gradient(order, shell=shell)
      for fe, ids in enumerate(local_maps):
        start = kadr + fe * stride
        matrix = np.asarray(d.stiffness[start:start + stride], dtype=np.float32)
        if matrix.size != stride:
          raise ValueError("compiled interpolated flex stiffness is truncated")
        # Exact pinned empty-element check: the leading matrix entry controls
        # whether mj_flexPassiveInterp processes this FE.
        if matrix[0] == 0:
          continue
        interp_nodes.append(ids)
        interp_matrix.append(matrix.reshape(npe3, npe3))
        interp_grad.append(grad_coeff)
        interp_npe.append(npe)
        interp_axes.append(axes_maps[fe])
        interp_damping.append(float(d.damping[f]))
        interp_owner.append(f)
    self._interp_count = len(interp_nodes)
    self._interp_nodes_host = tuple(tuple(row) for row in interp_nodes)
    self._interp_npe_host = tuple(interp_npe)
    self._interp_axes_host = tuple(interp_axes)
    self._interp_damping_host = tuple(interp_damping)
    self._interp_owner_host = tuple(interp_owner)
    self._interp_nodes = static_tensor(
        [np.pad(row, (0, 27 - len(row)), constant_values=-1) for row in interp_nodes],
        (self._interp_count, 27), torch.int32)
    self._interp_matrix = static_tensor(
        [np.pad(row, ((0, 81 - row.shape[0]), (0, 81 - row.shape[1])))
         for row in interp_matrix], (self._interp_count, 81, 81), torch.float32)
    self._interp_grad = static_tensor(
        [np.pad(row, ((0, 27 - row.shape[0]), (0, 3 - row.shape[1])))
         for row in interp_grad], (self._interp_count, 27, 3), torch.float32)
    self._interp_meta = static_tensor(
        [[npe, *axes, abs(int(d.interp[owner]))]
         for npe, axes, owner in zip(interp_npe, interp_axes, interp_owner)],
        (self._interp_count, 5), torch.int32)
    self._interp_damping = static_tensor(
        interp_damping, (self._interp_count,), torch.float32)
    self._shell_face_nodes_host = tuple(tuple(row) for row in shell_face_nodes)
    self._shell_face_axes_host = tuple(shell_face_axes)
    self._shell_face_order_host = tuple(shell_face_order)
    self._shell_bend_records_host = tuple(shell_bend_records)
    self._shell_bend_count = len(shell_bend_records)
    self._shell_face_nodes = static_tensor(
        [np.pad(row, (0, 9 - len(row)), constant_values=-1)
         for row in shell_face_nodes], (len(shell_face_nodes), 9), torch.int32)
    self._shell_face_axes = static_tensor(
        shell_face_axes, (len(shell_face_axes), 3), torch.int32)
    self._shell_face_order = static_tensor(
        shell_face_order, (len(shell_face_order),), torch.int32)
    self._shell_bend_records = static_tensor(
        [[face_a, face_b, *local, stiffness, *dn0]
         for face_a, face_b, local, stiffness, dn0, _ in shell_bend_records],
        (self._shell_bend_count, 10), torch.float32)

    # Upload static descriptors to device MPS tensors
    self._vert_local = torch.tensor(d.vert, dtype=torch.float32, device=self._device)
    self._vert0 = torch.tensor(d.vert0, dtype=torch.float32, device=self._device)
    self._vertbodyid = torch.tensor(d.vertbodyid, dtype=torch.int64, device=self._device)
    vert_centered = np.zeros(d.nflexvert, dtype=np.bool_)
    node_centered = np.zeros(len(d.nodebodyid), dtype=np.bool_)
    for f in range(d.nflex):
      va, vn = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
      na, nn = int(d.nodeadr[f]), int(d.nodenum[f])
      vert_centered[va:va + vn] = bool(d.centered[f])
      node_centered[na:na + nn] = bool(d.centered[f])
    self._vert_centered = torch.tensor(vert_centered, dtype=torch.bool, device=self._device)
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
    self._jnt_bodyid = np.asarray(model.jnt_bodyid, dtype=np.int32)
    body_joints = []
    for body in range(d.nbody):
      ancestors = []
      current = body
      while current > 0:
        adr, count = int(self._body_jntadr[current]), int(self._body_jntnum[current])
        ancestors.extend(range(adr, adr + count))
        current = int(self._body_parentid[current])
      body_joints.append(tuple(reversed(ancestors)))
    self._body_joint_chain = tuple(body_joints)

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
    self._node_centered = torch.tensor(node_centered, dtype=torch.bool, device=self._device)
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

  @staticmethod
  def _interp_shape_gradient(order, shell=False):
    """Tensor-product Q1/Q2 shape gradients at the element center."""
    if order == 1:
      phi, dphi = (0.5, 0.5), (-1.0, 1.0)
    elif order == 2:
      # Quadratic Lagrange basis values at the cell center. Only the midpoint
      # node is nonzero there; endpoint basis values are exactly zero.
      phi, dphi = (0.0, 1.0, 0.0), (-1.0, 0.0, 1.0)
    else:
      raise ValueError(f"unsupported pinned flex interpolation order {order}")
    grad = []
    if shell:
      for i in range(order + 1):
        for j in range(order + 1):
          grad.append((dphi[i] * phi[j], phi[i] * dphi[j], 0.0))
    else:
      for i in range(order + 1):
        for j in range(order + 1):
          for k in range(order + 1):
            grad.append((dphi[i] * phi[j] * phi[k],
                         phi[i] * dphi[j] * phi[k],
                         phi[i] * phi[j] * dphi[k]))
    return np.asarray(grad, dtype=np.float32)

  @staticmethod
  def _quat_rotate(quat, vector):
    qv = quat[:, None, 1:]
    qw = quat[:, None, :1]
    t = 2.0 * torch.cross(qv.expand_as(vector), vector, dim=-1)
    return vector + qw * t + torch.cross(qv.expand_as(vector), t, dim=-1)

  @staticmethod
  def _mat2rot_pinned(matrix):
    """Vectorized Müller et al. iteration used by pinned ``mju_mat2Rot``."""
    batch = matrix.shape[0]
    quat = torch.zeros((batch, 4), dtype=matrix.dtype, device=matrix.device)
    quat[:, 0] = 1.0
    for _ in range(48):
      w, x, y, z = quat.unbind(-1)
      rot = torch.stack((
          1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
          2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
          2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
      ), dim=-1).reshape(batch, 3, 3)
      omega = torch.cross(rot[:, :, 0], matrix[:, :, 0], dim=-1)
      omega += torch.cross(rot[:, :, 1], matrix[:, :, 1], dim=-1)
      omega += torch.cross(rot[:, :, 2], matrix[:, :, 2], dim=-1)
      den = torch.sum(rot * matrix, dim=(1, 2)).abs().clamp_min(1e-15)
      omega = omega / den[:, None]
      angle = torch.linalg.vector_norm(omega, dim=-1)
      half = 0.5 * angle
      axis_scale = torch.where(angle > 1e-15,
                               torch.sin(half) / angle.clamp_min(1e-15),
                               torch.zeros_like(angle))
      dq = torch.cat((torch.cos(half)[:, None], omega * axis_scale[:, None]), dim=-1)
      a, av = dq[:, :1], dq[:, 1:]
      b, bv = quat[:, :1], quat[:, 1:]
      quat = torch.cat((a*b - torch.sum(av*bv, dim=-1, keepdim=True),
                        a*bv + b*av + torch.cross(av, bv, dim=-1)), dim=-1)
      quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-15)
    return quat

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
    v_rot = torch.where(self._vert_centered[None, :, None], torch.zeros_like(v_rot), v_rot)
    self._flexvert_xpos.copy_(pos_b + v_rot)

    # 2. World vertex velocities: v_v = v_lin[b] + omega[b] x (x_v - xpos[b])
    if cvel is not None and cvel.numel() > 0:
      cvel_b = cvel[:, b_ids, :]  # (B, nflexvert, 6)
      w_b = cvel_b[..., :3]
      v_lin_b = cvel_b[..., 3:]
      root_com = poses.get("root_com")
      center = (root_com[:, b_ids, :] if root_com is not None
                else body_pos[:, b_ids, :])
      r_v = self._flexvert_xpos - center
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
      nrot = torch.where(self._node_centered[None, :, None], torch.zeros_like(nrot), nrot)
      self._node_xpos.copy_(body_pos[:, nbody_ids, :] + nrot)
      if cvel is not None and cvel.numel() > 0:
        ncvel = cvel[:, nbody_ids, :]
        root_com = poses.get("root_com")
        center = (root_com[:, nbody_ids, :] if root_com is not None
                  else body_pos[:, nbody_ids, :])
        nvel = ncvel[..., 3:] + torch.cross(
            ncvel[..., :3], self._node_xpos - center, dim=-1)
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

  def _add_attachment_tangent(self, body_ids, points, point_jacobian, point_force, poses):
    """Add d(J' f)/dq for rigidly attached force points.

    The material and edge element matrices differentiate Cartesian forces.
    This contracted kinematic Hessian supplies the missing articulated-body
    term for hinge, slide, ball, and free-joint ancestry without materializing
    a point-by-coordinate-by-coordinate Hessian tensor.
    """
    if self.descriptor.nv <= 0:
      return
    b = self.batch_size
    body_quat = poses["body_quat"]
    body_pos = poses["body_pos"]
    anchors = poses.get("joint_anchor")
    axes = poses.get("joint_axis")
    if axes is None:
      return
    eye = torch.eye(3, dtype=points.dtype, device=self._device)
    for pi, body_value in enumerate(body_ids):
      body = int(body_value)
      if body <= 0:
        continue
      specs = []
      for chain_index, joint in enumerate(self._body_joint_chain[body]):
        typ = int(self._jnt_type[joint])
        dofadr = int(self._jnt_dofadr[joint])
        joint_body = int(self._jnt_bodyid[joint])
        anchor = (anchors[:, joint, :] if anchors is not None else body_pos[:, joint_body, :])
        if typ in (0, 1):
          quat = body_quat[:, joint_body, :]
          for component in range(3 if typ == 0 else 0, 6 if typ == 0 else 3):
            local_component = component - (3 if typ == 0 else 0)
            basis = eye[local_component].expand(b, 3)
            qv, qw = quat[:, 1:], quat[:, :1]
            t = 2.0 * torch.cross(qv, basis, dim=-1)
            axis = basis + qw * t + torch.cross(qv, t, dim=-1)
            specs.append((dofadr + component, joint, chain_index, "rot", axis, anchor))
          if typ == 0:
            for component in range(3):
              specs.append((dofadr + component, joint, chain_index, "slide",
                            eye[component].expand(b, 3), anchor))
        else:
          mode = "slide" if typ == 2 else "rot"
          specs.append((dofadr, joint, chain_index, mode, axes[:, joint, :], anchor))
      point = points[:, pi, :]
      force = point_force[:, pi, :]
      jac = point_jacobian[:, pi, :, :]
      for dof_i, joint_i, order_i, mode_i, axis_i, anchor_i in specs:
        lever = point - anchor_i
        for dof_j, joint_j, order_j, mode_j, axis_j, anchor_j in specs:
          strict_ancestor = order_j < order_i
          same_joint = joint_i == joint_j
          d_axis = torch.zeros_like(axis_i)
          d_anchor = torch.zeros_like(anchor_i)
          if mode_j == "rot" and (strict_ancestor or (same_joint and mode_i == "rot")):
            d_axis = torch.cross(axis_j, axis_i, dim=-1)
          if strict_ancestor:
            if mode_j == "rot":
              d_anchor = torch.cross(axis_j, anchor_i - anchor_j, dim=-1)
            else:
              d_anchor = axis_j
          dpoint = jac[:, :, dof_j]
          if mode_i == "slide":
            d_jac = d_axis
          else:
            d_jac = (torch.cross(d_axis, lever, dim=-1)
                     + torch.cross(axis_i, dpoint - d_anchor, dim=-1))
          value = torch.sum(force * d_jac, dim=-1)
          self._stiffness_tangent[:, dof_i, dof_j].add_(value)

  def _point_velocity_qpos_jacobian(self, body_ids, points, jacobian, qvel, poses):
    """Return ``d(J*qvel)/dq`` for rigidly attached force points.

    This term matters for passive damping: changing a parent's pose rotates
    the child joint axes and therefore changes point velocity even when the
    generalized velocity vector is held fixed. The topology loops are over
    immutable compiled body/joint descriptors; all state arithmetic remains
    on the active tensor device.
    """
    b = self.batch_size
    nv = self.descriptor.nv
    result = torch.zeros((b, len(body_ids), 3, nv), dtype=points.dtype,
                         device=self._device)
    if nv <= 0:
      return result
    body_quat = poses["body_quat"]
    body_pos = poses["body_pos"]
    anchors = poses.get("joint_anchor")
    axes = poses.get("joint_axis")
    if axes is None:
      return result
    eye = torch.eye(3, dtype=points.dtype, device=self._device)
    for pi, body_value in enumerate(body_ids):
      body = int(body_value)
      if body <= 0:
        continue
      specs = []
      for chain_index, joint in enumerate(self._body_joint_chain[body]):
        typ = int(self._jnt_type[joint])
        dofadr = int(self._jnt_dofadr[joint])
        joint_body = int(self._jnt_bodyid[joint])
        anchor = (anchors[:, joint, :] if anchors is not None
                  else body_pos[:, joint_body, :])
        if typ in (0, 1):
          quat = body_quat[:, joint_body, :]
          for component in range(3 if typ == 0 else 0,
                                6 if typ == 0 else 3):
            local_component = component - (3 if typ == 0 else 0)
            basis = eye[local_component].expand(b, 3)
            qv, qw = quat[:, 1:], quat[:, :1]
            t = 2.0 * torch.cross(qv, basis, dim=-1)
            axis = basis + qw * t + torch.cross(qv, t, dim=-1)
            specs.append((dofadr + component, joint, chain_index,
                          "rot", axis, anchor))
          if typ == 0:
            for component in range(3):
              specs.append((dofadr + component, joint, chain_index,
                            "slide", eye[component].expand(b, 3), anchor))
        else:
          mode = "slide" if typ == 2 else "rot"
          specs.append((dofadr, joint, chain_index, mode,
                        axes[:, joint, :], anchor))
      point = points[:, pi, :]
      point_jac = jacobian[:, pi, :, :]
      for dof_j, joint_j, order_j, mode_j, axis_j, anchor_j in specs:
        dvelocity = torch.zeros_like(point)
        for dof_i, joint_i, order_i, mode_i, axis_i, anchor_i in specs:
          strict_ancestor = order_j < order_i
          same_joint = joint_i == joint_j
          d_axis = torch.zeros_like(axis_i)
          d_anchor = torch.zeros_like(anchor_i)
          if mode_j == "rot" and (strict_ancestor or
                                  (same_joint and mode_i == "rot")):
            d_axis = torch.cross(axis_j, axis_i, dim=-1)
          if strict_ancestor:
            if mode_j == "rot":
              d_anchor = torch.cross(axis_j, anchor_i - anchor_j, dim=-1)
            else:
              d_anchor = axis_j
          dpoint = point_jac[:, :, dof_j]
          if mode_i == "slide":
            d_jac = d_axis
          else:
            lever = point - anchor_i
            d_jac = (torch.cross(d_axis, lever, dim=-1)
                     + torch.cross(axis_i, dpoint - d_anchor, dim=-1))
          dvelocity += d_jac * qvel[:, dof_i:dof_i + 1]
        result[:, pi, :, dof_j] = dvelocity
    return result

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

    # Exact edge spring/damper tangents for fixed point Jacobians. The first
    # term is the familiar projected outer product. A stretched edge also
    # contributes geometric stiffness from the changing unit direction; the
    # Rayleigh-rate derivative contributes its nonsymmetric qpos term.
    c_eff = torch.where(is_spring, c_edge, torch.zeros_like(c_edge))
    k_eff = torch.where(is_spring, k_edge, torch.zeros_like(k_edge))

    # Batched outer-product accumulation: J_e (B, nflexedge, nv)
    J_e = self._flexedge_J
    damp_tang = -torch.einsum("be,ben,bem->bnm", c_eff, J_e, J_e)
    stiff_tang = -torch.einsum("be,ben,bem->bnm", k_eff, J_e, J_e)
    rel_jac = (
        self._flexvert_J[:, self._edge[:, 1], :, :]
        - self._flexvert_J[:, self._edge[:, 0], :, :])
    eye = torch.eye(3, dtype=rel_jac.dtype, device=self._device)
    projector = eye - self._flexedge_dir.unsqueeze(-1) * self._flexedge_dir.unsqueeze(-2)
    geom = torch.einsum("bein,beij,bejm->benm", rel_jac, projector, rel_jac)
    geom = geom / torch.clamp(self._flexedge_length[:, :, None, None], min=1e-10)
    stiff_tang -= torch.einsum("be,benm->bnm", tension, geom)
    if qvel is not None:
      rate_gradient = torch.einsum("bn,benm->bem", qvel[:, :nv], geom)
      point_vq = self._point_velocity_qpos_jacobian(
          self.descriptor.vertbodyid, self._flexvert_xpos,
          self._flexvert_J, qvel, poses)
      relative_vq = (point_vq[:, self._edge[:, 1], :, :]
                     - point_vq[:, self._edge[:, 0], :, :])
      rate_gradient += torch.einsum("bei,bein->ben", self._flexedge_dir,
                                    relative_vq)
      stiff_tang -= torch.einsum("be,ben,bem->bnm", c_eff, J_e, rate_gradient)
    self._damping_tangent.add_(damp_tang)
    self._stiffness_tangent.add_(stiff_tang)
    edge_force = torch.zeros_like(self._flexvert_xpos)
    for edge_idx in range(self.descriptor.nflexedge):
      v0, v1 = map(int, self.descriptor.edge[edge_idx])
      axial = tension[:, edge_idx, None] * self._flexedge_dir[:, edge_idx, :]
      edge_force[:, v0, :] += axial
      edge_force[:, v1, :] -= axial
    self._add_attachment_tangent(
        self.descriptor.vertbodyid, self._flexvert_xpos, self._flexvert_J,
        edge_force, poses)

    # 2. Pinned 2D membrane and 3D tetrahedral material forces.
    if self._stretch_count:
      self._compute_pinned_stretch(poses, qvel)
    if self._bend_count:
      self._compute_pinned_bend(qvel, poses)
    if self._interp_count or self._shell_bend_count:
      if self._interp_count and self._device.type == "mps":
        self._compute_interpolated_mps(poses, qvel)
      elif self._interp_count:
        self._compute_interpolated(poses, qvel)
      if self._shell_bend_count:
        if self._device.type == "mps":
          self._compute_interpolated_shell_bend_mps(qvel, poses)
        else:
          self._compute_interpolated_shell_bend(qvel, poses)

    return self._qfrc_passive, self._damping_tangent, self._stiffness_tangent

  def _compute_interpolated_mps(self, poses, qvel):
    """Dispatch the compiled Q1/Q2 volume element force kernel on MPS."""
    if not self._interp_count:
      return
    b, nv = self.batch_size, self.descriptor.nv
    node_count = len(self.descriptor.nodebodyid)
    elem_force = torch.empty(
        (b, self._interp_count, nv), dtype=torch.float32, device=self._device)
    elem_node_force = torch.empty(
        (b, self._interp_count, 81), dtype=torch.float32, device=self._device)
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    damper = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    dims = torch.tensor(
        [b, nv, self._interp_count, node_count, int(spring), int(damper)],
        dtype=torch.int32, device=self._device)
    self._material_shader.flex_interp_force(
        self._node_xpos.reshape(-1), self._node_xvel.reshape(-1),
        self._node0, self._node_J.reshape(-1), self._interp_nodes.reshape(-1),
        self._interp_matrix.reshape(-1), self._interp_grad.reshape(-1),
        self._interp_meta.reshape(-1), self._interp_damping, dims,
        elem_force.reshape(-1), elem_node_force.reshape(-1),
        threads=(b * self._interp_count,), group_size=(32,))
    self._qfrc_passive[:, :nv].add_(elem_force.sum(dim=1))
    # The material kernel emits Cartesian element forces as well as their
    # generalized projection. Contract the rigid attachment Jacobian Hessian
    # on-device so an articulated node's changing point Jacobian contributes
    # to the configuration tangent. The FE topology is immutable host data;
    # force values remain resident on the selected tensor device.
    if nv:
      for fe, node_ids in enumerate(self._interp_nodes_host):
        npe = self._interp_npe_host[fe]
        ids = np.asarray(node_ids, dtype=np.int32)
        node_force = elem_node_force[:, fe, :3*npe].reshape(b, npe, 3)
        self._add_attachment_tangent(
            ids, self._node_xpos[:, ids, :], self._node_J[:, ids, :, :],
            node_force, poses)
        self._compute_interpolated_tangent(fe, qvel, poses)

  def _compute_interpolated_shell_bend_mps(self, qvel, poses):
    """Dispatch compiled Crouzeix-Raviart shell-bend forces on MPS."""
    b, nv = self.batch_size, self.descriptor.nv
    elem_force = torch.empty(
        (b, self._shell_bend_count, nv), dtype=torch.float32,
        device=self._device)
    elem_node_force = torch.empty(
        (b, self._shell_bend_count, 54), dtype=torch.float32,
        device=self._device)
    dims = torch.tensor(
        [b, nv, self._shell_bend_count, len(self.descriptor.nodebodyid)],
        dtype=torch.int32, device=self._device)
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    records = self._shell_bend_records.clone()
    if not spring:
      records[:, 6:] = 0
    self._material_shader.flex_shell_bend_force(
        self._node_xpos.reshape(-1), self._node_J.reshape(-1),
        self._shell_face_nodes.reshape(-1), self._shell_face_axes.reshape(-1),
        self._shell_face_order, records.reshape(-1), dims,
        elem_force.reshape(-1), elem_node_force.reshape(-1),
        threads=(b * self._shell_bend_count,), group_size=(32,))
    self._qfrc_passive[:, :nv].add_(elem_force.sum(dim=1))
    if nv:
      for bend, record in enumerate(self._shell_bend_records_host):
        face_a, face_b = int(record[0]), int(record[1])
        for side, face in enumerate((face_a, face_b)):
          node_ids = self._shell_face_nodes_host[face]
          order = self._shell_face_order_host[face]
          npe = (order + 1) ** 2
          ids = np.asarray(node_ids[:npe], dtype=np.int32)
          start = side * 27
          node_force = elem_node_force[:, bend, start:start + 3*npe].reshape(
              b, npe, 3)
          self._add_attachment_tangent(
              ids, self._node_xpos[:, ids, :], self._node_J[:, ids, :, :],
              node_force, poses)
      if spring:
        for record in self._shell_bend_records_host:
          self._compute_shell_bend_material_tangent(record, qvel)

  def _compute_interpolated(self, poses, qvel):
    """Evaluate compiled Q1/Q2 corotational FE matrices (CPU test backend)."""
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    damper = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    if not spring and not damper:
      return
    for fe, node_ids in enumerate(self._interp_nodes_host):
      npe = self._interp_npe_host[fe]
      ids = torch.as_tensor(node_ids, dtype=torch.long, device=self._device)
      x = self._node_xpos[:, ids, :]
      v = self._node_xvel[:, ids, :]
      shape_grad = self._interp_grad[fe, :npe]
      axes = self._interp_axes_host[fe]
      Fparam = torch.einsum("bnd,nk->bdk", x, shape_grad)
      F = torch.zeros_like(Fparam)
      if len(node_ids) == (abs(int(self.descriptor.interp[self._interp_owner_host[fe]])) + 1) ** 3:
        F.copy_(Fparam)
      else:
        F[:, :, axes[0]] = Fparam[:, :, 0]
        F[:, :, axes[1]] = Fparam[:, :, 1]
        tangent0 = Fparam[:, :, 0]
        tangent1 = Fparam[:, :, 1]
        F[:, :, axes[2]] = torch.cross(tangent0, tangent1, dim=-1)
      rotation = self._mat2rot_pinned(F)
      # flexGather{Cell,Face}State returns the inverse quaternion so positions
      # and velocities enter the corotated material frame.
      qcorot = rotation.clone()
      qcorot[:, 1:] *= -1
      x_corot = self._quat_rotate(qcorot, x)
      v_corot = self._quat_rotate(qcorot, v)
      x0 = self._node0[ids].unsqueeze(0).expand_as(x)
      # Pinned mj_flexPassiveInterp rotates current positions and velocities,
      # while xpos0_e remains in the reference material frame.
      disp = (x_corot - x0).reshape(self.batch_size, 3 * npe)
      vel = v_corot.reshape(self.batch_size, 3 * npe)
      K = self._interp_matrix[fe, :3*npe, :3*npe]
      f_local = torch.zeros_like(disp)
      d_local = torch.zeros_like(vel)
      if spring:
        f_local = torch.einsum("ij,bj->bi", K, disp)
      if damper:
        d_local = self._interp_damping_host[fe] * torch.einsum("ij,bj->bi", K, vel)
      qworld = qcorot.clone()
      qworld[:, 1:] *= -1
      f_world = self._quat_rotate(qworld, f_local.reshape(self.batch_size, npe, 3))
      d_world = self._quat_rotate(qworld, d_local.reshape(self.batch_size, npe, 3))
      jac = self._node_J[:, ids, :, :]
      generalized = torch.zeros((self.batch_size, self.descriptor.nv),
                                dtype=x.dtype, device=self._device)
      if spring:
        generalized += torch.einsum("bnd,bndv->bv", f_world, jac)
      if damper:
        generalized += torch.einsum("bnd,bndv->bv", d_world, jac)
      self._qfrc_passive[:, :self.descriptor.nv].add_(generalized)
      self._add_attachment_tangent(
          np.asarray(node_ids, dtype=np.int32), x, jac, f_world, poses)
      self._compute_interpolated_tangent(fe, qvel, poses)

  @staticmethod
  def _matrix_from_quaternion(quat):
    """Return rotation matrices for scalar-first unit quaternions."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack((
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ), dim=-1).reshape(quat.shape[:-1] + (3, 3))

  def _polar_frame_tangent(self, F, dF):
    """Return pinned polar frame and its directional rotation derivatives."""
    b, _, _, nv = dF.shape
    quat = self._mat2rot_pinned(F)
    R = self._matrix_from_quaternion(quat)
    U = torch.matmul(R.transpose(-1, -2), F)
    A = torch.einsum("bda,bdjq->bajq", R, dF)
    rhs = torch.stack((A[:, 2, 1, :] - A[:, 1, 2, :],
                       A[:, 0, 2, :] - A[:, 2, 0, :],
                       A[:, 1, 0, :] - A[:, 0, 1, :]), dim=1)
    trace = torch.diagonal(U, dim1=-2, dim2=-1).sum(-1)
    sylvester = trace[:, None, None] * torch.eye(
        3, dtype=F.dtype, device=self._device)[None] - U
    c00, c01, c02 = sylvester[:, 0, 0], sylvester[:, 0, 1], sylvester[:, 0, 2]
    c11, c12, c22 = sylvester[:, 1, 1], sylvester[:, 1, 2], sylvester[:, 2, 2]
    i00 = c11*c22 - c12*c12
    i01 = c02*c12 - c01*c22
    i02 = c01*c12 - c02*c11
    i11 = c00*c22 - c02*c02
    i12 = c01*c02 - c00*c12
    i22 = c00*c11 - c01*c01
    det = c00*i00 + c01*i01 + c02*i02
    det_sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    invdet = 1.0 / (det_sign * det.abs().clamp_min(1e-12))
    rx, ry, rz = rhs[:, 0, :], rhs[:, 1, :], rhs[:, 2, :]
    omega_vec = torch.stack((
        (i00[:, None]*rx + i01[:, None]*ry + i02[:, None]*rz)*invdet[:, None],
        (i01[:, None]*rx + i11[:, None]*ry + i12[:, None]*rz)*invdet[:, None],
        (i02[:, None]*rx + i12[:, None]*ry + i22[:, None]*rz)*invdet[:, None],
    ), dim=1)
    ox, oy, oz = omega_vec.unbind(1)
    zero = torch.zeros_like(ox)
    Omega = torch.stack((zero, -oz, oy, oz, zero, -ox, -oy, ox, zero), dim=-1)
    return quat, R, Omega.reshape(b, nv, 3, 3)

  def _compute_interpolated_tangent(self, fe, qvel, poses):
    """Add the analytic corotational FE material and Rayleigh tangents.

    The polar-frame derivative uses the Sylvester equation for the skew part
    of ``R' dF``. All state-dependent arithmetic remains in torch tensors on
    the active device; the element node list, shape basis, and K matrix are
    immutable compiled descriptors.
    """
    if qvel is None or self.descriptor.nv <= 0:
      return
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    damper = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    if not spring and not damper:
      return
    b, nv = self.batch_size, self.descriptor.nv
    node_ids = np.asarray(self._interp_nodes_host[fe], dtype=np.int32)
    npe = self._interp_npe_host[fe]
    ids = torch.as_tensor(node_ids, dtype=torch.long, device=self._device)
    x = self._node_xpos[:, ids, :]
    v = self._node_xvel[:, ids, :]
    jac = self._node_J[:, ids, :, :]
    shape_grad = self._interp_grad[fe, :npe]
    axes = self._interp_axes_host[fe]
    is_volume = npe == (abs(int(self.descriptor.interp[
        self._interp_owner_host[fe]])) + 1) ** 3
    Fparam = torch.einsum("bnd,nk->bdk", x, shape_grad)
    dFparam = torch.einsum("bndq,nk->bdkq", jac, shape_grad)
    if is_volume:
      F = Fparam
      dF = dFparam
    else:
      F = torch.zeros_like(Fparam)
      F[:, :, axes[0]] = Fparam[:, :, 0]
      F[:, :, axes[1]] = Fparam[:, :, 1]
      t0, t1 = Fparam[:, :, 0], Fparam[:, :, 1]
      F[:, :, axes[2]] = torch.cross(t0, t1, dim=-1)
      dF = torch.zeros((b, 3, 3, nv), dtype=x.dtype, device=self._device)
      dF[:, :, axes[0], :] = dFparam[:, :, 0, :]
      dF[:, :, axes[1], :] = dFparam[:, :, 1, :]
      dt0, dt1 = dFparam[:, :, 0, :], dFparam[:, :, 1, :]
      dnormal = (torch.cross(dt0, t1[:, :, None, :].expand_as(dt0), dim=1)
                 + torch.cross(t0[:, :, None, :].expand_as(dt1), dt1, dim=1))
      dF[:, :, axes[2], :] = dnormal

    quat, R, Omega = self._polar_frame_tangent(F, dF)

    qc = quat.clone()
    qc[:, 1:] *= -1
    x_local = self._quat_rotate(qc, x)
    v_local = self._quat_rotate(qc, v)
    x0 = self._node0[ids].unsqueeze(0).expand_as(x)
    disp = (x_local - x0).reshape(b, 3*npe)
    vel = v_local.reshape(b, 3*npe)
    K = self._interp_matrix[fe, :3*npe, :3*npe]
    local_spring = torch.einsum("ij,bj->bi", K, disp).reshape(b, npe, 3)
    local_damper = torch.einsum("ij,bj->bi", K, vel).reshape(b, npe, 3)
    damping = self._interp_damping_host[fe] if damper else 0.0
    local_force = ((local_spring if spring else torch.zeros_like(local_spring))
                   + damping * local_damper)

    # Differentiate the corotated displacement, velocity, and world force for
    # all generalized-coordinate columns together.
    dx = jac.permute(0, 3, 1, 2)
    Rt = R.transpose(-1, -2)
    dx_local = torch.einsum("bji,bqnj->bqni", R, dx)
    y = x_local  # (B, node, xyz)
    yq = torch.einsum("bqij,bnj->bqni", Omega, y)
    ddisp = (dx_local - yq).reshape(b, nv, 3*npe)
    point_vq = self._point_velocity_qpos_jacobian(
        self.descriptor.nodebodyid[node_ids], x, jac, qvel, poses)
    dv = point_vq.permute(0, 3, 1, 2)
    dv_local = torch.einsum("bji,bqnj->bqni", R, dv)
    z = v_local
    zq = torch.einsum("bqij,bnj->bqni", Omega, z)
    dvel = (dv_local - zq).reshape(b, nv, 3*npe)
    dlocal = torch.zeros_like(ddisp)
    if spring:
      dlocal += torch.einsum("ij,bqj->bqi", K, ddisp)
    if damper and damping:
      dlocal += damping * torch.einsum("ij,bqj->bqi", K, dvel)
    dlocal = dlocal.reshape(b, nv, npe, 3)
    rotating_force = torch.einsum("bqij,bnj->bqni", Omega, local_force)
    df_world = torch.einsum(
        "bij,bqnj->bqni", R, dlocal + rotating_force)
    self._stiffness_tangent[:, :nv, :nv].add_(
        torch.einsum("bndv,bqnd->bvq", jac, df_world))

    if damper and damping:
      # d(J'F)/d qvel = J' R (cK) R' J; dJ is configuration-dependent only.
      K4 = K.reshape(npe, 3, npe, 3)
      Krot = torch.einsum("bda,namc,bec->bndme", R, K4, R)
      Krot = Krot.reshape(b, 3*npe, 3*npe)
      Jflat = jac.reshape(b, 3*npe, nv)
      damping_tangent = damping * torch.einsum(
          "bdn,bde,bem->bnm", Jflat, Krot, Jflat)
      self._damping_tangent[:, :nv, :nv].add_(damping_tangent)

  @staticmethod
  def _shape_phi(s, index, order):
    if order == 1:
      return 1.0 - s if index == 0 else s
    if index == 0:
      return 2*s*s - 3*s + 1
    if index == 1:
      return 4*(s - s*s)
    return 2*s*s - s

  @staticmethod
  def _shape_dphi(s, index, order):
    if order == 1:
      return -1.0 if index == 0 else 1.0
    if index == 0:
      return 4*s - 3
    if index == 1:
      return 4*(1 - 2*s)
    return 4*s - 1

  def _face_normal_frame(self, face_id, local):
    node_ids = self._shell_face_nodes_host[face_id]
    order = self._shell_face_order_host[face_id]
    x = self._node_xpos[:, node_ids, :]
    t1 = torch.zeros((self.batch_size, 3), dtype=x.dtype, device=self._device)
    t2 = torch.zeros_like(t1)
    grads = []
    idx = 0
    for l0 in range(order + 1):
      for l1 in range(order + 1):
        g0 = self._shape_dphi(local[0], l0, order) * self._shape_phi(local[1], l1, order)
        g1 = self._shape_phi(local[0], l0, order) * self._shape_dphi(local[1], l1, order)
        t1 += x[:, idx] * g0
        t2 += x[:, idx] * g1
        grads.append((g0, g1))
        idx += 1
    normal = torch.cross(t1, t2, dim=-1)
    return x, t1, t2, normal, grads

  def _face_corot_quat(self, face_id, x):
    order = self._shell_face_order_host[face_id]
    axes = self._shell_face_axes_host[face_id]
    grad = torch.as_tensor(
        self._interp_shape_gradient(order, shell=True), dtype=x.dtype, device=self._device)
    Fparam = torch.einsum("bnd,nk->bdk", x, grad)
    F = torch.zeros_like(Fparam)
    F[:, :, axes[0]] = Fparam[:, :, 0]
    F[:, :, axes[1]] = Fparam[:, :, 1]
    F[:, :, axes[2]] = torch.cross(Fparam[:, :, 0], Fparam[:, :, 1], dim=-1)
    q = self._mat2rot_pinned(F)
    q[:, 1:] *= -1
    return q

  @staticmethod
  def _quaternion_derivative(quat, Omega):
    """Directional quaternion derivative for ``dR = R * Omega``."""
    omega = torch.stack((Omega[:, :, 2, 1], Omega[:, :, 0, 2],
                         Omega[:, :, 1, 0]), dim=-1)
    q = quat[:, None, :].expand(-1, omega.shape[1], -1)
    qw, qv = q[..., :1], q[..., 1:]
    dqw = -0.5 * torch.sum(qv * omega, dim=-1, keepdim=True)
    dqv = 0.5 * (qw * omega + torch.cross(qv, omega, dim=-1))
    return torch.cat((dqw, dqv), dim=-1)

  def _shell_face_rotation_tangent(self, face_id, x, jac):
    order = self._shell_face_order_host[face_id]
    axes = self._shell_face_axes_host[face_id]
    grad = torch.as_tensor(
        self._interp_shape_gradient(order, shell=True),
        dtype=x.dtype, device=self._device)
    Fparam = torch.einsum("bnd,nk->bdk", x, grad)
    dFparam = torch.einsum("bndv,nk->bdkv", jac, grad)
    F = torch.zeros_like(Fparam)
    F[:, :, axes[0]] = Fparam[:, :, 0]
    F[:, :, axes[1]] = Fparam[:, :, 1]
    t0, t1 = Fparam[:, :, 0], Fparam[:, :, 1]
    F[:, :, axes[2]] = torch.cross(t0, t1, dim=-1)
    dF = torch.zeros((x.shape[0], 3, 3, self.descriptor.nv),
                     dtype=x.dtype, device=self._device)
    dF[:, :, axes[0], :] = dFparam[:, :, 0, :]
    dF[:, :, axes[1], :] = dFparam[:, :, 1, :]
    dt0, dt1 = dFparam[:, :, 0, :], dFparam[:, :, 1, :]
    dnormal = (torch.cross(dt0, t1[:, :, None].expand_as(dt0), dim=1)
               + torch.cross(t0[:, :, None].expand_as(dt1), dt1, dim=1))
    dF[:, :, axes[2], :] = dnormal
    quat, _R, Omega = self._polar_frame_tangent(F, dF)
    return quat, self._quaternion_derivative(quat, Omega)

  def _compute_shell_bend_material_tangent(self, record, qvel):
    """Differentiate the CR normal-jump force through normals and corotation."""
    if qvel is None or self.descriptor.nv <= 0:
      return
    face_a, face_b, local, stiffness, dn0, _damping = record
    if stiffness == 0:
      return
    xa, ta0, ta1, na_raw, grada = self._face_normal_frame(face_a, local[:2])
    xb, tb0, tb1, nb_raw, gradb = self._face_normal_frame(face_b, local[2:])
    ida = np.asarray(self._shell_face_nodes_host[face_a], dtype=np.int32)
    idb = np.asarray(self._shell_face_nodes_host[face_b], dtype=np.int32)
    ja = self._node_J[:, torch.as_tensor(ida, dtype=torch.long, device=self._device), :, :]
    jb = self._node_J[:, torch.as_tensor(idb, dtype=torch.long, device=self._device), :, :]
    grada_np, gradb_np = grada, gradb
    ga0 = torch.as_tensor([g[0] for g in grada_np], dtype=xa.dtype, device=self._device)
    ga1 = torch.as_tensor([g[1] for g in grada_np], dtype=xa.dtype, device=self._device)
    gb0 = torch.as_tensor([g[0] for g in gradb_np], dtype=xb.dtype, device=self._device)
    gb1 = torch.as_tensor([g[1] for g in gradb_np], dtype=xb.dtype, device=self._device)

    def normal_tangent(x, jac, t0, t1, raw, g0, g1):
      dt0 = torch.einsum("bndv,n->bdv", jac, g0).permute(0, 2, 1)
      dt1 = torch.einsum("bndv,n->bdv", jac, g1).permute(0, 2, 1)
      draw = (torch.cross(dt0, t1[:, None, :].expand_as(dt0), dim=-1)
              + torch.cross(t0[:, None, :].expand_as(dt1), dt1, dim=-1))
      length = torch.linalg.vector_norm(raw, dim=-1).clamp_min(1e-15)
      normal = raw / length[:, None]
      dlength = torch.sum(normal[:, None, :] * draw, dim=-1)
      dnormal = (draw - normal[:, None, :] * torch.sum(
          normal[:, None, :] * draw, dim=-1, keepdim=True)) / length[:, None, None]
      return length, normal, dt0, dt1, dlength, dnormal

    lena, na, dta0, dta1, dlena, dna = normal_tangent(
        xa, ja, ta0, ta1, na_raw, ga0, ga1)
    lenb, nb, dtb0, dtb1, dlenb, dnb = normal_tangent(
        xb, jb, tb0, tb1, nb_raw, gb0, gb1)
    qa, dqa = self._shell_face_rotation_tangent(face_a, xa, ja)
    qb, dqb = self._shell_face_rotation_tangent(face_b, xb, jb)
    qa = qa.clone()
    qb = qb.clone()
    qa[:, 1:] *= -1
    qb[:, 1:] *= -1
    dqa = dqa.clone()
    dqb = dqb.clone()
    dqa[:, :, 1:] *= -1
    dqb[:, :, 1:] *= -1
    same = (qa * qb).sum(dim=-1, keepdim=True) >= 0
    sign = torch.where(same, 1.0, -1.0)
    qb = qb * sign
    dqb = dqb * sign[:, None, :]
    qsum = qa + qb
    qnorm = torch.linalg.vector_norm(qsum, dim=-1, keepdim=True).clamp_min(1e-15)
    qavg = qsum / qnorm
    dqsum = dqa + dqb
    dqavg = (dqsum - qavg[:, None, :] * torch.sum(
        qavg[:, None, :] * dqsum, dim=-1, keepdim=True)) / qnorm[:, None, :]
    qworld = qavg.clone()
    qworld[:, 1:] *= -1
    dqworld = dqavg.clone()
    dqworld[:, :, 1:] *= -1
    dn0_tensor = torch.as_tensor(dn0, dtype=xa.dtype, device=self._device)
    dnrot = self._quat_rotate(qworld, dn0_tensor[None, None, :])[:, 0]
    qv, qw = qworld[:, None, 1:], qworld[:, None, :1]
    dqv, dqw = dqworld[..., 1:], dqworld[..., :1]
    inner = torch.cross(qv.expand(-1, self.descriptor.nv, -1),
                        dn0_tensor[None, None, :].expand_as(dqv), dim=-1)
    inner += qw * dn0_tensor[None, None, :]
    dinner = torch.cross(dqv, dn0_tensor[None, None, :].expand_as(dqv), dim=-1)
    dinner += dqw * dn0_tensor[None, None, :]
    d_dnrot = 2.0 * (torch.cross(dqv, inner, dim=-1)
                     + torch.cross(qv.expand_as(dqv), dinner, dim=-1))
    residual = na - nb - dnrot
    dresidual = dna - dnb - d_dnrot

    def differentiate_weight(normal, length, dnormal, dlength, residual, dresidual):
      dot = torch.sum(normal * residual, dim=-1)
      ddot = (torch.sum(dnormal * residual[:, None, :], dim=-1)
              + torch.sum(normal[:, None, :] * dresidual, dim=-1))
      numerator = residual - normal * dot[:, None]
      dnumerator = (dresidual - dnormal * dot[:, None, None]
                    - normal[:, None, :] * ddot[:, :, None])
      weight = numerator / length[:, None]
      dweight = (dnumerator / length[:, None, None]
                 - numerator[:, None, :] * dlength[:, :, None]
                 / length[:, None, None].square())
      return weight, dweight

    wa, dwa = differentiate_weight(na, lena, dna, dlena, residual, dresidual)
    wb, dwb = differentiate_weight(nb, lenb, dnb, dlenb, residual, dresidual)
    fda = []
    for n, (g0, g1) in enumerate(grada_np):
      df = stiffness * (
          g0 * (torch.cross(dwa, ta1[:, None, :], dim=-1)
                + torch.cross(wa[:, None, :], dta1, dim=-1))
          - g1 * (torch.cross(dwa, ta0[:, None, :], dim=-1)
                  + torch.cross(wa[:, None, :], dta0, dim=-1)))
      fda.append(df)
    fdb = []
    for n, (g0, g1) in enumerate(gradb_np):
      df = -stiffness * (
          g0 * (torch.cross(dwb, tb1[:, None, :], dim=-1)
                + torch.cross(wb[:, None, :], dtb1, dim=-1))
          - g1 * (torch.cross(dwb, tb0[:, None, :], dim=-1)
                  + torch.cross(wb[:, None, :], dtb0, dim=-1)))
      fdb.append(df)
    dfa = torch.stack(fda, dim=2)
    dfb = torch.stack(fdb, dim=2)
    self._stiffness_tangent[:, :self.descriptor.nv, :self.descriptor.nv].add_(
        torch.einsum("bndv,bqnd->bvq", ja, dfa)
        + torch.einsum("bndv,bqnd->bvq", jb, dfb))

  def _compute_interpolated_shell_bend(self, qvel, poses):
    """Pinned CR normal-jump force for interpolated shell edges."""
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    if not spring:
      return
    node_force = torch.zeros_like(self._node_xpos)
    for face_a, face_b, local, stiffness, dn0, _damping in self._shell_bend_records_host:
      x_a, t1_a, t2_a, n_a_raw, grad_a = self._face_normal_frame(
          face_a, local[:2])
      x_b, t1_b, t2_b, n_b_raw, grad_b = self._face_normal_frame(
          face_b, local[2:])
      len_a = torch.linalg.vector_norm(n_a_raw, dim=-1).clamp_min(1e-15)
      len_b = torch.linalg.vector_norm(n_b_raw, dim=-1).clamp_min(1e-15)
      n_a = n_a_raw / len_a[:, None]
      n_b = n_b_raw / len_b[:, None]
      quat_a = self._face_corot_quat(face_a, x_a)
      quat_b = self._face_corot_quat(face_b, x_b)
      same_hemi = (quat_a * quat_b).sum(dim=-1, keepdim=True) >= 0
      quat_b = torch.where(same_hemi, quat_b, -quat_b)
      quat_avg = quat_a + quat_b
      quat_avg = quat_avg / torch.linalg.vector_norm(
          quat_avg, dim=-1, keepdim=True).clamp_min(1e-15)
      quat_avg[:, 1:] *= -1
      dn = torch.as_tensor(dn0, dtype=x_a.dtype, device=self._device).expand(self.batch_size, 3)
      dn_rot = self._quat_rotate(quat_avg, dn[:, None, :])[:, 0]
      residual = n_a - n_b - dn_rot
      dot_a = torch.sum(n_a * residual, dim=-1, keepdim=True)
      dot_b = torch.sum(n_b * residual, dim=-1, keepdim=True)
      w_a = (residual - n_a * dot_a) / len_a[:, None]
      w_b = (residual - n_b * dot_b) / len_b[:, None]
      wa_t2 = torch.cross(w_a, t2_a, dim=-1)
      wa_t1 = torch.cross(w_a, t1_a, dim=-1)
      wb_t2 = torch.cross(w_b, t2_b, dim=-1)
      wb_t1 = torch.cross(w_b, t1_b, dim=-1)
      for idx, (g0, g1) in enumerate(grad_a):
        gi = self._shell_face_nodes_host[face_a][idx]
        node_force[:, gi] += stiffness * (g0 * wa_t2 - g1 * wa_t1)
      for idx, (g0, g1) in enumerate(grad_b):
        gi = self._shell_face_nodes_host[face_b][idx]
        node_force[:, gi] -= stiffness * (g0 * wb_t2 - g1 * wb_t1)
    self._qfrc_passive[:, :self.descriptor.nv].add_(
        torch.einsum("bnd,bndv->bv", node_force, self._node_J))
    self._add_attachment_tangent(
        self.descriptor.nodebodyid, self._node_xpos, self._node_J,
        node_force, poses)
    for record in self._shell_bend_records_host:
      self._compute_shell_bend_material_tangent(record, qvel)

  def _compute_pinned_stretch(self, poses, qvel):
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
      point_vq = self._point_velocity_qpos_jacobian(
          d.vertbodyid[np.asarray(self._stretch_vertex_host[e], dtype=np.int32)],
          x, jv, qvel, poses)
      rel_vq = torch.stack([
          point_vq[:, a, :, :] - point_vq[:, c, :, :]
          for a, c in pairs], dim=1)
      jrate += torch.einsum("bei,bein->ben", unit, rel_vq)
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

      local_force = torch.zeros((b, nve, 3), dtype=x.dtype, device=self._device)
      edgevec = diffs
      for edge_index, (a, c) in enumerate(pairs):
        local_force[:, a] -= coeff[:, edge_index, None] * edgevec[:, edge_index]
        local_force[:, c] += coeff[:, edge_index, None] * edgevec[:, edge_index]
      local_body_ids = d.vertbodyid[np.asarray(self._stretch_vertex_host[e], dtype=np.int32)]
      self._add_attachment_tangent(
          local_body_ids, x, jv, local_force, poses)

      damping = self._stretch_damping_host[e]
      if damping:
        weight = 2.0 * damping * (lengths - rates * d.timestep)
        damp_tangent = -torch.einsum(
            "bjn,ij,bi,bim,bj->bnm", je, metric, weight, je, lengths)
        self._damping_tangent.add_(damp_tangent)

  def _compute_pinned_bend(self, qvel, poses):
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
          qvel.reshape(-1), self._bend_dofadr.reshape(-1),
          self._bend_dofnum.reshape(-1),
          threads=(b * self._bend_count * nv,), group_size=(128,))
      self._qfrc_passive.add_(element_force.sum(dim=1))

    if self._device.type != "mps":
      for e in range(self._bend_count):
        verts = self._bend_vertex[e].long()
        x = self._flexvert_xpos[:, verts, :]
        v_components = []
        for adr, count in zip(self._bend_dofadr_host[e],
                              self._bend_dofnum_host[e]):
          local = torch.zeros((b, 3), dtype=torch.float32, device=self._device)
          if count:
            local[:, :count] = qvel[:, adr:adr + count]
          v_components.append(local)
        v = torch.stack(v_components, dim=1)
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
        for i, (adr, count) in enumerate(zip(
            self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
          if count:
            self._qfrc_passive[:, adr:adr + count].sub_(total[:, i, :count])
    if nv:
      self._compute_pinned_bend_tangents(spring, damper)

  def _compute_pinned_bend_tangents(self, spring, damper):
    """Differentiate the pinned direct-body-DOF ordinary bend implementation.

    MuJoCo's ordinary ``mj_flexPassiveBend`` writes its four vertex forces to
    ``body_dofadr`` directly (rather than projecting them through vertex J).
    Its damper likewise reads those raw qvel slots.  Preserve that source-level
    behavior here, including its lack of attachment Hessian terms.
    """
    b, nv = self.batch_size, self.descriptor.nv
    for e in range(self._bend_count):
      if not spring and not damper:
        continue
      verts_np = np.asarray(self._bend_vertex_host[e], dtype=np.int32)
      verts = torch.as_tensor(verts_np, dtype=torch.long, device=self._device)
      x = self._flexvert_xpos[:, verts, :]
      jac = self._flexvert_J[:, verts, :, :]
      matrix = self._bend_data[e, :16].reshape(4, 4)
      curvature = self._bend_data[e, 16]
      damping = self._bend_damping_host[e] if damper else 0.0
      edge0 = x[:, 1] - x[:, 0]
      edge1 = x[:, 2] - x[:, 0]
      edge2 = x[:, 3] - x[:, 0]
      ref = torch.stack((
          -torch.cross(edge1, edge2, dim=-1)
          - torch.cross(edge2, edge0, dim=-1)
          - torch.cross(edge0, edge1, dim=-1),
          torch.cross(edge1, edge2, dim=-1),
          torch.cross(edge2, edge0, dim=-1),
          torch.cross(edge0, edge1, dim=-1),
      ), dim=1)

      e0 = edge0
      e1 = edge1
      e2 = edge2
      de0 = (jac[:, 1] - jac[:, 0]).permute(0, 2, 1)
      de1 = (jac[:, 2] - jac[:, 0]).permute(0, 2, 1)
      de2 = (jac[:, 3] - jac[:, 0]).permute(0, 2, 1)
      dref1 = (torch.cross(de1, e2[:, None, :].expand_as(de1), dim=-1)
               + torch.cross(e1[:, None, :].expand_as(de2), de2, dim=-1))
      dref2 = (torch.cross(de2, e0[:, None, :].expand_as(de2), dim=-1)
               + torch.cross(e2[:, None, :].expand_as(de0), de0, dim=-1))
      dref3 = (torch.cross(de0, e1[:, None, :].expand_as(de0), dim=-1)
               + torch.cross(e0[:, None, :].expand_as(de1), de1, dim=-1))
      dref0 = -(dref1 + dref2 + dref3)
      dref = torch.stack((dref0, dref1, dref2, dref3), dim=2).permute(
          0, 2, 3, 1)
      dtotal = torch.einsum("ij,bjdq->bidq", matrix, jac)
      if spring and curvature != 0.0:
        dtotal = dtotal + curvature * dref
      if spring:
        for i, (adr, count) in enumerate(zip(
            self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
          if count:
            self._stiffness_tangent[:, adr:adr + count, :nv].sub_(
                dtotal[:, i, :count, :])
      if damper and damping:
        for i, (adr_i, count_i) in enumerate(zip(
            self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
          for j, (adr_j, count_j) in enumerate(zip(
              self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
            count = min(count_i, count_j)
            if count:
              self._damping_tangent[:, adr_i:adr_i + count,
                                    adr_j:adr_j + count].diagonal(
                                        dim1=-2, dim2=-1).sub_(
                                            damping * matrix[i, j])

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
