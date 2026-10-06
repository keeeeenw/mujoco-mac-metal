# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned MuJoCo 3.10 flex material primitives for the native backend.

Compiled simplex and interpolated volume forces and ordinary triangular shell
bending records are lowered here. Interpolated shell bending has a CPU tensor
oracle; full native interpolation tangents and general flex contact remain
separate qualification gates. Edge equalities, contact assembly and lifecycle
are integrated by the owning simulation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Optional, Tuple

import mujoco


class _FlexPositionContext:
  """Owner- and generation-checked borrowed frozen flex kinematics."""

  __slots__ = ("owner", "generation", "buffers")

  def __init__(self, owner, generation, buffers):
    self.owner = owner
    self.generation = generation
    self.buffers = buffers


class _FlexMaterialOperatorContext:
  """Frozen flex K/Kd geometry for matrix-free implicit correction."""

  __slots__ = ("owner", "generation", "rotations")

  def __init__(self, owner, generation, rotations):
    self.owner = owner
    self.generation = generation
    self.rotations = rotations
import numpy as np
try:
  import torch
except ImportError:  # Host-only model lowering/preflight does not need Torch.
  torch = None

_MATERIAL_SHADER = Path(__file__).parent / "shaders" / "flex_material.metal"


def _frozen(values, dtype):
  array = np.array(values, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("flex constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _same_torch_device(actual, expected):
  """Compare devices while allowing an unspecified accelerator index."""
  return (actual.type == expected.type
          and (expected.index is None or actual.index == expected.index))


def _lower_flexedge_jacobian_csr(model):
  """Return the pinned compiled flex-edge Jacobian CSR, with validation.

  The runtime edge geometry can be evaluated for every generalized DOF, but
  MuJoCo only projects forces through columns present in this compiled CSR.
  Keeping the support in CSR form avoids an additional ``nflexedge * nv``
  support matrix on large models.
  """
  nedge, nv = int(model.nflexedge), int(model.nv)
  try:
    rowadr = np.asarray(model.flexedge_J_rowadr, dtype=np.int64)
    rownnz = np.asarray(model.flexedge_J_rownnz, dtype=np.int64)
    colind = np.asarray(model.flexedge_J_colind, dtype=np.int64)
  except AttributeError as exc:
    raise ValueError("compiled flex edge Jacobian CSR is required") from exc
  if rowadr.shape != (nedge,) or rownnz.shape != (nedge,) or colind.ndim != 1:
    raise ValueError("compiled flex edge Jacobian CSR has invalid shapes")
  if np.any(rowadr < 0) or np.any(rownnz < 0) or np.any(rowadr + rownnz > len(colind)):
    raise ValueError("compiled flex edge Jacobian CSR row exceeds colind")
  for edge in range(nedge):
    cols = colind[int(rowadr[edge]):int(rowadr[edge] + rownnz[edge])]
    if np.any(cols < 0) or np.any(cols >= nv):
      raise ValueError("compiled flex edge Jacobian column is outside nv")
    if len(cols) > 1 and np.any(cols[1:] <= cols[:-1]):
      raise ValueError("compiled flex edge Jacobian columns must be sorted unique")
  return (_frozen(rowadr, np.int32), _frozen(rownnz, np.int32),
          _frozen(colind, np.int32))


def _interpolation_vertex_tables(descriptor):
  """Lower pinned cellLookup/evalBasis maps from vertices to flex nodes."""
  nvert = int(descriptor.nflexvert)
  nnode = len(descriptor.nodebodyid)
  indices = np.zeros((nvert, 27), dtype=np.int32)
  weights = np.zeros((nvert, 27), dtype=np.float64)
  enabled = np.zeros((nvert,), dtype=np.int32)
  for flex in range(int(descriptor.nflex)):
    interp = int(descriptor.interp[flex])
    if interp == 0:
      continue
    order = abs(interp)
    if order not in (1, 2):
      raise NotImplementedError(
          f"pinned flex interpolation order {order} is unsupported")
    cells = np.asarray(descriptor.cellnum[flex], dtype=np.int64)
    if np.any(cells <= 0):
      raise ValueError("interpolated flex has an empty cell grid")
    nxyz = cells * order + 1
    nodeadr = int(descriptor.nodeadr[flex])
    vertadr = int(descriptor.vertadr[flex])
    for local_vertex in range(int(descriptor.vertnum[flex])):
      vertex = vertadr + local_vertex
      coord = np.asarray(descriptor.vert0[vertex], dtype=np.float64)
      cell = np.floor(coord * cells).astype(np.int64)
      cell = np.minimum(np.maximum(cell, 0), cells - 1)
      local = np.clip(coord * cells - cell, 0.0, 1.0)
      cursor = 0
      for i in range(order + 1):
        gi = int(cell[0] * order + i)
        for j in range(order + 1):
          gj = int(cell[1] * order + j)
          for k in range(order + 1):
            gk = int(cell[2] * order + k)
            if order == 1:
              basis = ((1.0-local[0]) if i == 0 else local[0])
              basis *= ((1.0-local[1]) if j == 0 else local[1])
              basis *= ((1.0-local[2]) if k == 0 else local[2])
            else:
              def phi(x, index):
                return ((2*x*x - 3*x + 1) if index == 0 else
                        (4*(x-x*x) if index == 1 else 2*x*x-x))
              basis = phi(local[0], i) * phi(local[1], j) * phi(local[2], k)
            node = nodeadr + gi * int(nxyz[1]*nxyz[2]) + gj * int(nxyz[2]) + gk
            if (node < nodeadr
                or node >= nodeadr + int(descriptor.nodenum[flex])
                or node >= nnode):
              raise ValueError("compiled interpolation node is outside its flex")
            indices[vertex, cursor] = node
            weights[vertex, cursor] = basis
            cursor += 1
      enabled[vertex] = 1
  return indices, weights, enabled


def _shell_node_tfi_tables(descriptor):
  """Lower pinned mju_shellTrackInterior boundary weights per shell node."""
  nnode = len(descriptor.nodebodyid)
  indices = np.zeros((nnode, 26), dtype=np.int32)
  weights = np.zeros((nnode, 26), dtype=np.float64)
  enabled = np.zeros((nnode,), dtype=np.int32)
  for flex in range(int(descriptor.nflex)):
    if int(descriptor.interp[flex]) >= 0:
      continue
    order = -int(descriptor.interp[flex])
    cells = np.asarray(descriptor.cellnum[flex], dtype=np.int64)
    nx, ny, nz = (cells * order + 1).tolist()
    if min(nx, ny, nz) < 3:
      continue
    nodeadr = int(descriptor.nodeadr[flex])
    for i in range(1, nx-1):
      s = i / (nx - 1)
      for j in range(1, ny-1):
        t = j / (ny - 1)
        for k in range(1, nz-1):
          u = k / (nz - 1)
          target = nodeadr + i*ny*nz + j*nz + k
          terms = (
              ((0,j,k), 1-s), ((nx-1,j,k), s),
              ((i,0,k), 1-t), ((i,ny-1,k), t),
              ((i,j,0), 1-u), ((i,j,nz-1), u),
              ((i,0,0), -(1-t)*(1-u)),
              ((i,0,nz-1), -(1-t)*u),
              ((i,ny-1,0), -t*(1-u)),
              ((i,ny-1,nz-1), -t*u),
              ((0,j,0), -(1-s)*(1-u)),
              ((0,j,nz-1), -(1-s)*u),
              ((nx-1,j,0), -s*(1-u)),
              ((nx-1,j,nz-1), -s*u),
              ((0,0,k), -(1-s)*(1-t)),
              ((0,ny-1,k), -(1-s)*t),
              ((nx-1,0,k), -s*(1-t)),
              ((nx-1,ny-1,k), -s*t),
              ((0,0,0), (1-s)*(1-t)*(1-u)),
              ((0,0,nz-1), (1-s)*(1-t)*u),
              ((0,ny-1,0), (1-s)*t*(1-u)),
              ((0,ny-1,nz-1), (1-s)*t*u),
              ((nx-1,0,0), s*(1-t)*(1-u)),
              ((nx-1,0,nz-1), s*(1-t)*u),
              ((nx-1,ny-1,0), s*t*(1-u)),
              ((nx-1,ny-1,nz-1), s*t*u),
          )
          for index, ((ii, jj, kk), weight) in enumerate(terms):
            indices[target, index] = nodeadr + ii*ny*nz + jj*nz + kk
            weights[target, index] = weight
          enabled[target] = 1
  return indices, weights, enabled


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
  vertadr: np.ndarray         # (nflex,) int32 vertex starts
  vertnum: np.ndarray         # (nflex,) int32 vertex counts
  vert: np.ndarray            # (nflexvert, 3) float32
  vert0: np.ndarray           # (nflexvert, 3) float32
  vertbodyid: np.ndarray      # (nflexvert,) int32
  vertmetric: np.ndarray      # (nflexvert, 4) float32 reference inverse metrics
  vertedgeadr: np.ndarray     # (nflexvert,) int32 adjacency start
  vertedgenum: np.ndarray     # (nflexvert,) int32 adjacency count
  vertedge: np.ndarray        # (2*nflexedge,) int32 local edge IDs
  size: np.ndarray            # (nflex, 3) float32 half extents
  body_mass: np.ndarray       # (nbody,) float32 mass weights
  body_invweight0: np.ndarray # (2*nbody,) float32 compiled equality weights
  edge: np.ndarray            # (nflexedge, 2) int32
  edgeflap: np.ndarray        # (nflexedge, 2) int32, local flap vertices
  edgeadr: np.ndarray         # (nflex,) int32
  edgenum: np.ndarray         # (nflex,) int32
  edgeequality: np.ndarray    # (nflex,) int32
  edge_rigid: np.ndarray      # (nflexedge,) bool, omitted from mjEQ_FLEX rows
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
  equality_row_ids_by_eqid: MappingProxyType
  equality_row_counts: MappingProxyType


def _lower_equality_row_inventory(model, descriptor):
  """Return canonical flex-equality row identities without a device runtime.

  CC construction happens before a ``MetalFlex`` exists, so these counts and
  identities are derived solely from the compiled model.  Equality IDs are
  visited in source order and remain separate even when they target one flex.
  """
  row_ids = {}
  shell_offsets = {}
  shell_face_count = 0
  for f in range(descriptor.nflex):
    signed_order = int(descriptor.interp[f])
    kadr = int(descriptor.stiffnessadr[f])
    if (signed_order < 0 and kadr >= 0 and not bool(model.flex_rigid[f])
        and int(descriptor.dim[f]) != 1):
      cx, cy, cz = map(int, descriptor.cellnum[f])
      shell_offsets[f] = shell_face_count
      shell_face_count += 2 * (cy * cz + cx * cz + cx * cy)

  flex_type = int(mujoco.mjtEq.mjEQ_FLEX)
  flexvert_type = int(mujoco.mjtEq.mjEQ_FLEXVERT)
  flexstrain_type = int(mujoco.mjtEq.mjEQ_FLEXSTRAIN)
  for eqid in range(int(model.neq)):
    eq_type = int(model.eq_type[eqid])
    if eq_type not in (flex_type, flexvert_type, flexstrain_type):
      continue
    f = int(model.eq_obj1id[eqid])
    if f < 0 or f >= descriptor.nflex:
      raise ValueError(f"compiled flex equality {eqid} has invalid flex ID")
    if eq_type == flex_type:
      start, count = int(descriptor.edgeadr[f]), int(descriptor.edgenum[f])
      ids = [edge_id for edge_id in range(start, start + count)
             if not bool(descriptor.edge_rigid[edge_id])]
      row_ids[eqid] = _frozen(ids, np.int32)
    elif eq_type == flexvert_type:
      start = int(model.flex_vertadr[f])
      count = int(model.flex_vertnum[f])
      row_ids[eqid] = _frozen(np.arange(2 * start, 2 * (start + count)),
                              np.int32)
    else:
      signed_order = int(descriptor.interp[f])
      order = abs(signed_order)
      kadr = int(descriptor.stiffnessadr[f])
      nodenum = int(descriptor.nodenum[f])
      # MuJoCo can retain an auto-generated flexstrain equality on a flex
      # whose interpolation/material block was compiled away.  Such an
      # equality has no solver rows; it is different from malformed active
      # compiled data, which remains a fail-closed error.
      if order == 0 or nodenum == 0 or kadr < 0:
        row_ids[eqid] = _frozen([], np.int32)
        continue
      if order not in (1, 2):
        raise NotImplementedError(
            f"compiled flexstrain equality {eqid} has unsupported order {signed_order}")
      shell = signed_order < 0
      npe = (order + 1) ** (2 if shell else 3)
      data = np.asarray(model.eq_data[eqid], dtype=np.float64)
      if shell:
        elem = int(data[0])
        if f not in shell_offsets:
          raise ValueError(
              f"compiled flexstrain equality {eqid} has no shell element map")
        elem += shell_offsets[f]
      else:
        ci, cj, ck = map(int, data[:3])
        cx, cy, cz = map(int, descriptor.cellnum[f])
        if not (0 <= ci < cx and 0 <= cj < cy and 0 <= ck < cz):
          raise ValueError(
              f"compiled flexstrain equality {eqid} has an invalid cell")
        elem = ci * cy * cz + cj * cz + ck
      ndof = 3 * npe
      address = kadr + elem * ndof * ndof
      if address < 0 or address >= descriptor.stiffness.size:
        raise ValueError(
            f"compiled flexstrain equality {eqid} stiffness address is invalid")
      mode_count = int(descriptor.stiffness[address])
      if mode_count == 0:
        row_ids[eqid] = _frozen([], np.int32)
        continue
      if mode_count < 0 or mode_count > ndof:
        raise ValueError(
            f"compiled flexstrain equality {eqid} has invalid mode count")
      if address + 1 + mode_count * ndof > descriptor.stiffness.size:
        raise ValueError(
            f"compiled flexstrain equality {eqid} eigenmode block is truncated")
      row_ids[eqid] = _frozen(np.arange(mode_count), np.int32)
  counts = {eqid: int(ids.size) for eqid, ids in row_ids.items()}
  return MappingProxyType(row_ids), MappingProxyType(counts)


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
  vertmetric = _frozen(model.flex_vertmetric, np.float32).reshape(nflexvert, 4)
  vertedgeadr = _frozen(model.flex_vertedgeadr, np.int32)
  vertedgenum = _frozen(model.flex_vertedgenum, np.int32)
  vertedge = _frozen(model.flex_vertedge, np.int32).reshape(-1)
  vertadr = _frozen(model.flex_vertadr, np.int32)
  vertnum = _frozen(model.flex_vertnum, np.int32)
  size = _frozen(model.flex_size, np.float32).reshape(nflex, 3)
  body_mass = _frozen(model.body_mass, np.float32)
  body_invweight0 = _frozen(model.body_invweight0, np.float32).reshape(-1)
  edge = np.array(model.flex_edge, dtype=np.int32, order="C", copy=True).reshape(nflexedge, 2)
  edgeadr = _frozen(model.flex_edgeadr, np.int32)
  edgenum = _frozen(model.flex_edgenum, np.int32)
  vertadr = _frozen(model.flex_vertadr, np.int32)
  for f in range(nflex):
    start, count = int(edgeadr[f]), int(edgenum[f])
    edge[start:start + count] += int(vertadr[f])
  edge = _frozen(edge, np.int32)
  edgeequality = _frozen(model.flex_edgeequality, np.int32)
  edge_rigid = _frozen(model.flexedge_rigid, np.bool_)
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

  descriptor = FlexDescriptor(
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
      vertadr=vertadr,
      vertnum=vertnum,
      vert=vert,
      vert0=vert0,
      vertbodyid=vertbodyid,
      vertmetric=vertmetric,
      vertedgeadr=vertedgeadr,
      vertedgenum=vertedgenum,
      vertedge=vertedge,
      size=size,
      body_mass=body_mass,
      body_invweight0=body_invweight0,
      edge=edge,
      edgeflap=edgeflap,
      edgeadr=edgeadr,
      edgenum=edgenum,
      edgeequality=edgeequality,
      edge_rigid=edge_rigid,
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
      equality_row_ids_by_eqid=MappingProxyType({}),
      equality_row_counts=MappingProxyType({}),
  )
  ids, counts = _lower_equality_row_inventory(model, descriptor)
  # Frozen dataclass: construct the final immutable value with the inventory.
  from dataclasses import replace
  return replace(descriptor, equality_row_ids_by_eqid=ids,
                 equality_row_counts=counts)


class MetalFlex:
  """Device-resident deformable physics manager executed on Apple Silicon MPS."""

  def __init__(self, model: mujoco.MjModel, batch_size: int = 1, device: str = "mps"):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    if torch is None:
      raise RuntimeError("MetalFlex runtime requires PyTorch; model lowering is Torch-free")
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
                                           np.any(d.interp != 0) or
                                           d.nflexedge or d.nflexvert or
                                           len(d.nodebodyid)) else None)

    bend_vertex, bend_data, bend_damping = [], [], []
    bend_dofadr, bend_dofnum = [], []
    for f in range(d.nflex):
      if (int(d.dim[f]) != 2 or int(d.bendingadr[f]) < 0
          or bool(model.flex_rigid[f])):
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
    self._edge_operator_shader = self._material_shader
    if self._edge_operator_shader is None:
      self._edge_operator_shader = self._bend_shader
    if (self._edge_operator_shader is None and self._device.type == "mps"
        and d.nflexedge):
      self._edge_operator_shader = torch.mps.compile_shader(
          _MATERIAL_SHADER.read_text())

    # Lower the full compiled interpolation element matrices and structured
    # node gather maps. These records are shared by order-1/order-2 volume
    # elements and signed shell faces; the runtime routine below currently
    # uses them on the CPU test backend while the matching MSL kernel lands.
    interp_nodes, interp_matrix, interp_grad, interp_npe, interp_axes = [], [], [], [], []
    interp_damping, interp_owner = [], []
    shell_face_nodes, shell_face_axes, shell_face_order = [], [], []
    shell_face_offset_by_flex = {}
    shell_bend_records = []
    for f in range(d.nflex):
      signed_order = int(d.interp[f])
      kadr = int(d.stiffnessadr[f])
      if (signed_order == 0 or kadr < 0
          or bool(model.flex_rigid[f])
          or int(d.dim[f]) == 1):
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
        shell_face_offset_by_flex[f] = face_offset
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
    # Static int64 gather indices avoid repeating a host-to-device metadata
    # conversion inside each matrix-free CG matvec.
    self._interp_nodes_long = tuple(
        torch.as_tensor(np.array(row, dtype=np.int64, copy=True),
                        dtype=torch.long, device=self._device)
        for row in interp_nodes)
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
    vert_low = (np.asarray(model.flex_vert, dtype=np.float64).reshape(-1, 3)
                - np.asarray(d.vert, dtype=np.float32).astype(np.float64))
    vert_tail = (np.asarray(model.flex_vert, dtype=np.float64).reshape(-1, 3)
                 - np.asarray(d.vert, dtype=np.float32).astype(np.float64)
                 - np.asarray(vert_low, dtype=np.float32).astype(np.float64))
    self._vert_local_low = torch.as_tensor(
        np.array(vert_low, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device)
    self._vert_local_tail = torch.as_tensor(
        np.array(vert_tail, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device)
    self._vert0 = torch.tensor(d.vert0, dtype=torch.float32, device=self._device)
    self._vertbodyid = torch.tensor(d.vertbodyid, dtype=torch.int64, device=self._device)
    self._vertbodyid32 = torch.as_tensor(
        np.array(d.vertbodyid, dtype=np.int32, copy=True), dtype=torch.int32,
        device=self._device)
    vert_centered = np.zeros(d.nflexvert, dtype=np.bool_)
    node_centered = np.zeros(len(d.nodebodyid), dtype=np.bool_)
    for f in range(d.nflex):
      va, vn = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
      na, nn = int(d.nodeadr[f]), int(d.nodenum[f])
      vert_centered[va:va + vn] = bool(d.centered[f])
      node_centered[na:na + nn] = bool(d.centered[f])
    self._vert_centered = torch.tensor(vert_centered, dtype=torch.bool, device=self._device)
    self._vert_centered32 = torch.as_tensor(
        np.array(vert_centered, dtype=np.int32, copy=True), dtype=torch.int32,
        device=self._device)
    self._edge = torch.tensor(d.edge, dtype=torch.int64, device=self._device)
    self._recovery_edge = torch.as_tensor(
        np.asarray(d.edge, dtype=np.int32).copy(), dtype=torch.int32,
        device=self._device)
    self._edge_length0 = torch.tensor(d.edge_length0, dtype=torch.float32, device=self._device)
    self._edge_invweight0 = torch.tensor(d.edge_invweight0, dtype=torch.float32, device=self._device)
    self._edgestiffness = torch.tensor(d.edgestiffness, dtype=torch.float32, device=self._device)
    self._edgedamping = torch.tensor(d.edgedamping, dtype=torch.float32, device=self._device)
    self._edgeequality = torch.tensor(d.edgeequality, dtype=torch.int32, device=self._device)
    self._edge_rigid = torch.tensor(d.edge_rigid, dtype=torch.bool, device=self._device)
    self._radius = torch.tensor(d.radius, dtype=torch.float32, device=self._device)

    # Topology per-edge flex index
    edge_flexid = np.zeros(d.nflexedge, dtype=np.int64)
    for f in range(d.nflex):
      adr = d.edgeadr[f]
      num = d.edgenum[f]
      edge_flexid[adr:adr + num] = f
    self._edge_flexid = torch.tensor(edge_flexid, dtype=torch.int64, device=self._device)
    (edge_j_rowadr, edge_j_rownnz,
     edge_j_colind) = _lower_flexedge_jacobian_csr(model)
    self._edge_j_rowadr_host = edge_j_rowadr
    self._edge_j_rownnz_host = edge_j_rownnz
    self._edge_j_colind_host = edge_j_colind
    # These host-only index arrays feed Torch advanced indexing in the CPU
    # fallback. Keep the independently allocated arrays writable so Torch
    # does not wrap read-only NumPy storage at each kinematics update.
    self._edge_j_unsupported_host = tuple(
        np.setdiff1d(np.arange(d.nv, dtype=np.int32),
                     edge_j_colind[int(edge_j_rowadr[e]):
                                   int(edge_j_rowadr[e] + edge_j_rownnz[e])])
        for e in range(d.nflexedge))
    self._edge_j_rowadr = torch.tensor(
        np.array(edge_j_rowadr, copy=True), dtype=torch.int32, device=self._device)
    self._edge_j_rownnz = torch.tensor(
        np.array(edge_j_rownnz, copy=True), dtype=torch.int32, device=self._device)
    # Metal requires a physical binding even when the compiled matrix has no
    # entries.  The logical count in dims[3] remains zero in that case.
    edge_j_colind_binding = (np.array(edge_j_colind, copy=True)
                             if len(edge_j_colind) else np.zeros(1, dtype=np.int32))
    self._edge_j_colind = torch.tensor(
        edge_j_colind_binding, dtype=torch.int32, device=self._device)
    edge_has_jacobian = edge_j_rownnz > 0
    edge_rigid = np.asarray(d.edge_rigid, dtype=np.bool_).copy()
    if hasattr(model, "flexedge_rigid"):
      edge_rigid |= np.asarray(model.flexedge_rigid, dtype=bool)
    edge_rigid |= np.asarray(model.flex_rigid, dtype=bool)[edge_flexid]
    edge_spring_coeff = np.asarray(d.edgestiffness, dtype=np.float32)[edge_flexid].copy()
    edge_damping_coeff = np.asarray(d.edgedamping, dtype=np.float32)[edge_flexid].copy()
    edge_spring_coeff[edge_rigid | ~edge_has_jacobian] = 0.0
    edge_damping_coeff[edge_rigid | ~edge_has_jacobian] = 0.0
    self._edge_spring_coeff_host = _frozen(edge_spring_coeff, np.float32)
    self._edge_operator_coeff_host = _frozen(edge_damping_coeff, np.float32)
    self._edge_spring_coeff = torch.tensor(
        np.array(self._edge_spring_coeff_host, copy=True), dtype=torch.float32,
        device=self._device)
    self._edge_operator_coeff = torch.tensor(
        np.array(self._edge_operator_coeff_host, copy=True), dtype=torch.float32,
        device=self._device)
    self._edge_j_mask_dims = torch.tensor(
        [self.batch_size, d.nflexedge, d.nv, len(edge_j_colind)],
        dtype=torch.int32, device=self._device)
    self._edge_operator_coo_dims = torch.tensor(
        [d.nv, d.nflexedge, 0, self.batch_size],
        dtype=torch.int32, device=self._device)
    self._edge_operator_coo_edge_count = None

    # mjEQ_FLEX emits rows in edge order but omits rigid edges. Keep the
    # compiled row-to-edge map immutable so replay and the coupled allocator
    # never infer row identity from a per-step active count.
    flex_rigid = np.asarray(model.flex_rigid, dtype=np.bool_)
    edge_flexid_host = np.asarray(edge_flexid, dtype=np.int64)
    self._equality_edge_ids = tuple(
        _frozen(np.flatnonzero((edge_flexid_host == f) & ~d.edge_rigid
                               & ~flex_rigid[edge_flexid_host]), np.int32)
        for f in range(d.nflex))

    # Precomputed edge constraint parameters for run_equalities
    if d.nflexedge > 0:
      solref_edge = d.solref[edge_flexid]
      solimp_edge = d.solimp[edge_flexid]
      self._edge_timeconst = torch.tensor(np.maximum(solref_edge[:, 0], 1e-4), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_dampratio = torch.tensor(np.maximum(solref_edge[:, 1], 1e-4), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_dmin = torch.tensor(solimp_edge[:, 0], dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_dmax = torch.tensor(solimp_edge[:, 1], dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_width = torch.tensor(np.maximum(solimp_edge[:, 2], 0.0), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_mid = torch.tensor(np.maximum(solimp_edge[:, 3], 1e-4), dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_power = torch.tensor(solimp_edge[:, 4], dtype=torch.float32, device=self._device).unsqueeze(0)
      ref0 = np.asarray(solref_edge[:, 0], dtype=np.float32).copy()
      if not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)):
        ref0 = np.where(ref0 > 0.0,
                        np.maximum(ref0, 2.0 * float(model.opt.timestep)),
                        ref0)
      dwidth = np.maximum(np.asarray(solimp_edge[:, 1], dtype=np.float32), 0.0)
      ref1 = np.asarray(solref_edge[:, 1], dtype=np.float32)
      denom_b = np.maximum(dwidth * np.where(ref0 > 0.0, ref0, 1.0), 1e-15)
      denom_k = np.maximum(dwidth * dwidth * np.where(ref0 > 0.0, ref0 * ref0, 1.0)
                           * np.where(ref1 > 0.0, ref1 * ref1, 1.0), 1e-15)
      b0 = np.where(ref1 > 0.0, 2.0 / denom_b,
                    -ref1 / np.maximum(dwidth, 1e-15))
      k0 = np.where(ref0 > 0.0, 1.0 / denom_k,
                    -ref0 / np.maximum(dwidth * dwidth, 1e-15))
      self._edge_b0 = torch.tensor(b0, dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_k0 = torch.tensor(k0, dtype=torch.float32, device=self._device).unsqueeze(0)
      self._edge_diag = torch.clamp(self._edge_invweight0.unsqueeze(0), min=1e-12)
    else:
      self._edge_b0 = torch.zeros((1, 0), dtype=torch.float32, device=self._device)
      self._edge_k0 = torch.zeros((1, 0), dtype=torch.float32, device=self._device)
      self._edge_diag = torch.zeros((1, 0), dtype=torch.float32, device=self._device)

    # Explicit equality objects carry their own solref/solimp. Flexcomp's
    # edge parameters are the defaults only; retaining eq-indexed compiled
    # overrides is necessary when several mjEQ_FLEX rows target one flex.
    self._equality_parameter_sets = {}
    self._equality_flex_ids = {}
    for eqid in range(int(model.neq)):
      if int(model.eq_type[eqid]) != int(mujoco.mjtEq.mjEQ_FLEX):
        continue
      f = int(model.eq_obj1id[eqid])
      if f < 0 or f >= d.nflex:
        raise ValueError(f"compiled flex equality {eqid} has invalid flex ID")
      dmin = np.zeros(d.nflexedge, dtype=np.float32)
      dmax = np.zeros_like(dmin)
      width = np.zeros_like(dmin)
      mid = np.full_like(dmin, 0.5)
      power = np.full_like(dmin, 2.0)
      adr, count = int(d.edgeadr[f]), int(d.edgenum[f])
      ref = np.asarray(model.eq_solref[eqid], dtype=np.float32)
      imp = np.asarray(model.eq_solimp[eqid], dtype=np.float32)
      if not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)):
        ref = ref.copy()
        if ref[0] > 0.0:
          ref[0] = max(float(ref[0]), 2.0 * float(model.opt.timestep))
      dmin[adr:adr + count] = imp[0]
      dmax[adr:adr + count] = imp[1]
      width[adr:adr + count] = max(float(imp[2]), 0.0)
      mid[adr:adr + count] = max(float(imp[3]), 1e-4)
      power[adr:adr + count] = imp[4]
      diag = np.zeros(d.nflexedge, dtype=np.float32)
      diag[adr:adr + count] = d.edge_invweight0[adr:adr + count]
      as_tensor = lambda x: torch.as_tensor(
          x[None], dtype=torch.float32, device=self._device)
      dwidth = as_tensor(dmax)
      ref0 = np.zeros_like(dmin)
      ref1 = np.zeros_like(dmin)
      ref0[adr:adr + count] = float(ref[0])
      ref1[adr:adr + count] = float(ref[1])
      r0, r1 = as_tensor(ref0), as_tensor(ref1)
      denom_b = torch.clamp(dwidth * torch.where(r0 > 0.0, r0, 1.0), min=1e-15)
      denom_k = torch.clamp(dwidth.square()
          * torch.where(r0 > 0.0, r0.square(), 1.0)
          * torch.where(r1 > 0.0, r1.square(), 1.0), min=1e-15)
      self._equality_parameter_sets[eqid] = {
          "dmin": as_tensor(dmin), "dmax": as_tensor(dmax),
          "width": as_tensor(width), "mid": as_tensor(mid),
          "power": as_tensor(power),
          "solimp": torch.as_tensor(np.array(imp, copy=True),
                                    dtype=torch.float32, device=self._device),
          "solref": torch.as_tensor(np.array(ref, copy=True),
                                    dtype=torch.float32, device=self._device),
          "b0": torch.where(r1 > 0.0, 2.0 / denom_b,
                            -r1 / torch.clamp(dwidth, min=1e-15)),
          "k0": torch.where(r0 > 0.0, 1.0 / denom_k,
                            -r0 / torch.clamp(dwidth.square(), min=1e-15)),
          "diag": torch.clamp(as_tensor(diag), min=1e-12),
      }
      self._equality_flex_ids[eqid] = f

    # Topology per-vertex flex index
    vert_flexid = np.zeros(d.nflexvert, dtype=np.int64)
    for f in range(d.nflex):
      # Pinned flexcomp vertices are sequential per flex
      v_start = int(model.flex_vertadr[f])
      v_num = int(model.flex_vertnum[f])
      vert_flexid[v_start:v_start + v_num] = f
    self._vert_flexid = torch.tensor(vert_flexid, dtype=torch.int64, device=self._device)

    b, nv = self.batch_size, d.nv
    vertex_node_ids, vertex_node_weights, vertex_interp_enabled = (
        _interpolation_vertex_tables(d))
    shell_node_ids, shell_node_weights, shell_node_enabled = (
        _shell_node_tfi_tables(d))
    self._has_interpolated_vertices = bool(np.any(vertex_interp_enabled))
    self._interpolated_vertex_ids_host = tuple(
        int(value) for value in np.flatnonzero(vertex_interp_enabled))
    self._has_shell_tfi_nodes = bool(np.any(shell_node_enabled))
    self._vertex_interp_nodes = torch.as_tensor(
        np.array(vertex_node_ids, copy=True), dtype=torch.int32,
        device=self._device).contiguous()
    vertex_weight_high = np.asarray(vertex_node_weights, dtype=np.float32)
    vertex_weight_low = np.asarray(vertex_node_weights, dtype=np.float64) - \
        vertex_weight_high.astype(np.float64)
    vertex_weight_tail = vertex_weight_low - \
        vertex_weight_low.astype(np.float32).astype(np.float64)
    self._vertex_interp_weights_low = torch.as_tensor(
        np.array(vertex_weight_low, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device).contiguous()
    self._vertex_interp_weights_tail = torch.as_tensor(
        np.array(vertex_weight_tail, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device).contiguous()
    self._vertex_interp_weights = torch.as_tensor(
        np.array(vertex_weight_high, copy=True), dtype=torch.float32,
        device=self._device).contiguous()
    self._vertex_interp_enabled = torch.as_tensor(
        np.array(vertex_interp_enabled, copy=True), dtype=torch.int32,
        device=self._device).contiguous()
    self._vertex_interp_dims = torch.tensor(
        [b, d.nflexvert, len(d.nodebodyid), nv] + [1] * b, dtype=torch.int32,
        device=self._device)
    self._paired_interp_dims = torch.tensor(
        [b, 1, 1, 1] + [1] * b, dtype=torch.int32, device=self._device)
    self._paired_point_dims = torch.tensor(
        [b, 1, d.nbody] + [1] * b, dtype=torch.int32,
        device=self._device)
    self._shell_tfi_nodes = torch.as_tensor(
        np.array(shell_node_ids, copy=True), dtype=torch.int32,
        device=self._device).contiguous()
    shell_weight_high = np.asarray(shell_node_weights, dtype=np.float32)
    shell_weight_low = np.asarray(shell_node_weights, dtype=np.float64) - \
        shell_weight_high.astype(np.float64)
    shell_weight_tail = shell_weight_low - \
        shell_weight_low.astype(np.float32).astype(np.float64)
    self._shell_tfi_weights_low = torch.as_tensor(
        np.array(shell_weight_low, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device).contiguous()
    self._shell_tfi_weights_tail = torch.as_tensor(
        np.array(shell_weight_tail, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device).contiguous()
    self._shell_tfi_weights = torch.as_tensor(
        np.array(shell_weight_high, copy=True), dtype=torch.float32,
        device=self._device).contiguous()
    self._shell_tfi_enabled = torch.as_tensor(
        np.array(shell_node_enabled, copy=True), dtype=torch.int32,
        device=self._device).contiguous()
    self._shell_tfi_dims = torch.tensor(
        [b, len(d.nodebodyid), nv] + [1] * b,
        dtype=torch.int32, device=self._device)
    # Compiled adjacency is packed by vertex but each adjacency entry is a
    # local edge index within its owning flex. Freeze the exact endpoint,
    # reference metric, half-size and MuJoCo mass weights for flexvert rows.
    vertex_neighbors = []
    for v in range(d.nflexvert):
      f = int(vert_flexid[v])
      local_edges = d.vertedge[int(d.vertedgeadr[v]):
                               int(d.vertedgeadr[v] + d.vertedgenum[v])]
      records = []
      for local_edge in local_edges:
        ge = int(d.edgeadr[f]) + int(local_edge)
        v1, v2 = map(int, d.edge[ge])
        if v not in (v1, v2):
          raise ValueError("compiled flex vertex adjacency does not match its edge")
        neighbor = v2 if v == v1 else v1
        # MuJoCo's flexvert formulation is a 2D metric and passes the first
        # two scaled rest-coordinate components to its 3x2 products.
        dx = ((d.vert0[v2] - d.vert0[v1]) * (2.0 * d.size[f]))[:2]
        body = int(d.vertbodyid[neighbor])
        weight = max(float(d.body_mass[body]), 1e-15) if body >= 0 else 1.0
        dx_tensor = torch.as_tensor(np.asarray(dx, dtype=np.float32),
                                    dtype=torch.float32, device=self._device)
        records.append((v1, v2, dx_tensor, weight))
      vertex_neighbors.append(tuple(records))
    self._vertex_neighbors_host = tuple(vertex_neighbors)
    self._vertmetric = torch.as_tensor(
        np.array(d.vertmetric, copy=True), dtype=torch.float32, device=self._device)
    self._flexvert_eq_pos = torch.zeros(
        (b, d.nflexvert, 2), dtype=torch.float32, device=self._device)
    self._flexvert_eq_J = torch.zeros(
        (b, d.nflexvert, 2, max(nv, 1)), dtype=torch.float32, device=self._device)
    self._flexvert_eq_vel = torch.zeros(
        (b, d.nflexvert, 2), dtype=torch.float32, device=self._device)
    self._flexvert_eq_imp = torch.empty_like(self._flexvert_eq_pos)
    self._flexvert_eq_aref = torch.empty_like(self._flexvert_eq_pos)
    self._flexvert_eq_R = torch.empty_like(self._flexvert_eq_pos)
    self._flexvert_eq_flexids = tuple(
        int(model.eq_obj1id[eid]) for eid in range(model.neq)
        if int(model.eq_type[eid]) == int(mujoco.mjtEq.mjEQ_FLEXVERT))
    self._flexvert_eq_param_sets = {}
    for eqid in range(int(model.neq)):
      if int(model.eq_type[eqid]) != int(mujoco.mjtEq.mjEQ_FLEXVERT):
        continue
      f = int(model.eq_obj1id[eqid])
      if f < 0 or f >= d.nflex:
        raise ValueError(f"compiled flexvert equality {eqid} has invalid flex ID")
      va, vn = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
      bodies = np.asarray(d.vertbodyid[va:va + vn], dtype=np.int64)
      diag = np.repeat(np.asarray(d.body_invweight0[2 * bodies],
                                  dtype=np.float32), 2)
      ref = np.asarray(model.eq_solref[eqid], dtype=np.float32).copy()
      if not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)):
        if ref[0] > 0.0:
          ref[0] = max(float(ref[0]), 2.0 * float(model.opt.timestep))
      imp = np.asarray(model.eq_solimp[eqid], dtype=np.float32)
      dmax = max(float(imp[1]), 0.0)
      b0 = (2.0 / max(dmax * float(ref[0]), 1e-15) if ref[1] > 0.0
            else -float(ref[1]) / max(dmax, 1e-15))
      k0 = (1.0 / max(dmax * dmax * float(ref[0]) ** 2
                      * float(ref[1]) ** 2, 1e-15) if ref[0] > 0.0
            else -float(ref[0]) / max(dmax * dmax, 1e-15))
      self._flexvert_eq_param_sets[eqid] = {
          "flex_id": f,
          "solimp": torch.as_tensor(imp, dtype=torch.float32,
                                    device=self._device),
          "solref": torch.as_tensor(np.array(ref, copy=True),
                                    dtype=torch.float32, device=self._device),
          "b0": float(b0), "k0": float(k0),
          "diag": torch.as_tensor(diag[None], dtype=torch.float32,
                                  device=self._device),
      }

    # FLEXSTRAIN rows are compiled from one signed flex_interp element and
    # the per-element eigenmodes stored at flex_stiffnessadr.  Keep only
    # immutable identities and coefficients here; the execution routine
    # below evaluates the corotated residual/J from the live flex-node state.
    self._flexstrain_records = {}
    for eqid in range(int(model.neq)):
      if int(model.eq_type[eqid]) != int(mujoco.mjtEq.mjEQ_FLEXSTRAIN):
        continue
      f = int(model.eq_obj1id[eqid])
      if f < 0 or f >= d.nflex:
        raise ValueError(f"compiled flexstrain equality {eqid} has invalid flex ID")
      signed_order = int(d.interp[f])
      order = abs(signed_order)
      shell = signed_order < 0
      if d.equality_row_counts[eqid] == 0:
        # Compiled auto-generated equality with no eigenmodes has no CC rows.
        # Keep it out of the runtime record set, while its zero count remains
        # visible to the model-lowering allocator.
        continue
      if order not in (1, 2) or int(d.nodenum[f]) == 0:
        raise NotImplementedError(
            f"compiled flexstrain equality {eqid} uses unsupported flex order "
            f"{signed_order} or an empty node set")
      npe = (order + 1) ** (2 if shell else 3)
      data = np.asarray(model.eq_data[eqid], dtype=np.float64)
      if shell:
        elem = int(data[0])
        face_index = shell_face_offset_by_flex.get(f, -1) + elem
        if face_index < 0 or face_index >= len(self._shell_face_nodes_host):
          raise ValueError(
              f"compiled flexstrain equality {eqid} has invalid shell face {elem}")
        node_ids = np.asarray(
            self._shell_face_nodes_host[face_index], dtype=np.int32)
        axes = tuple(map(int, self._shell_face_axes_host[face_index]))
      else:
        ci, cj, ck = (int(data[0]), int(data[1]), int(data[2]))
        cx, cy, cz = map(int, d.cellnum[f])
        if not (0 <= ci < cx and 0 <= cj < cy and 0 <= ck < cz):
          raise ValueError(
              f"compiled flexstrain equality {eqid} has invalid cell "
              f"({ci}, {cj}, {ck}) for shape ({cx}, {cy}, {cz})")
        elem = ci * cy * cz + cj * cz + ck
        ny, nz = cy * order + 1, cz * order + 1
        node_ids = np.asarray([
            int(d.nodeadr[f]) + (ci * order + li) * ny * nz
            + (cj * order + lj) * nz + ck * order + lk
            for li in range(order + 1)
            for lj in range(order + 1)
            for lk in range(order + 1)], dtype=np.int32)
        axes = (0, 1, 2)
      if node_ids.size != npe:
        raise ValueError("compiled flexstrain element node map is truncated")
      kadr = int(d.stiffnessadr[f])
      if kadr < 0:
        raise ValueError(
            f"compiled flexstrain equality {eqid} has no stiffness block")
      ndof = 3 * npe
      start = kadr + elem * ndof * ndof
      neig = int(d.stiffness[start])
      if neig <= 0 or neig > ndof:
        raise ValueError(
            f"compiled flexstrain equality {eqid} has invalid mode count "
            f"{neig} for ndof={ndof}")
      vectors = np.asarray(d.stiffness[start + 1:start + 1 + neig * ndof],
                           dtype=np.float32)
      if vectors.size != neig * ndof:
        raise ValueError("compiled flexstrain eigenmode block is truncated")
      bodies = np.asarray(d.nodebodyid[node_ids], dtype=np.int64)
      diag = float(np.mean(d.body_invweight0[2 * bodies]))
      self._flexstrain_records[eqid] = {
          "flex_id": f, "order": order, "shell": shell,
          "axes": axes, "node_ids": tuple(map(int, node_ids)),
          "node_ids_t": torch.as_tensor(
              np.array(node_ids, copy=True), dtype=torch.long,
              device=self._device),
          "shape_grad": torch.as_tensor(
              self._interp_shape_gradient(order, shell=shell),
              dtype=torch.float32, device=self._device),
          "eigenvectors": torch.as_tensor(
              np.array(vectors.reshape(neig, npe, 3), copy=True),
              dtype=torch.float32,
              device=self._device),
          "diag": diag,
          "solimp": torch.as_tensor(model.eq_solimp[eqid],
                                    dtype=torch.float32, device=self._device),
          "solref": torch.as_tensor(model.eq_solref[eqid],
                                    dtype=torch.float32, device=self._device),
      }

    # Immutable logical row identities/counts for the coupled allocator.
    # The source instantiates model equality IDs in increasing order; multiple
    # equalities on one flex therefore remain distinct ordered blocks.
    self.equality_row_ids_by_eqid = d.equality_row_ids_by_eqid
    self.equality_row_counts = d.equality_row_counts

    self._has_2d = bool(np.any(d.dim == 2))
    self._has_3d = bool(np.any(d.dim == 3))

    # Tree body parent map for point Jacobians
    self._body_parentid = np.asarray(model.body_parentid, dtype=np.int32)
    self._body_jntadr = np.asarray(model.body_jntadr, dtype=np.int32)
    self._body_jntnum = np.asarray(model.body_jntnum, dtype=np.int32)
    self._jnt_type = np.asarray(model.jnt_type, dtype=np.int32)
    self._jnt_dofadr = np.asarray(model.jnt_dofadr, dtype=np.int32)
    self._jnt_bodyid = np.asarray(model.jnt_bodyid, dtype=np.int32)
    self._recovery_body_parentid = torch.as_tensor(
        self._body_parentid.copy(), dtype=torch.int32, device=self._device)
    self._recovery_body_jntadr = torch.as_tensor(
        self._body_jntadr.copy(), dtype=torch.int32, device=self._device)
    self._recovery_body_jntnum = torch.as_tensor(
        self._body_jntnum.copy(), dtype=torch.int32, device=self._device)
    self._recovery_jnt_type = torch.as_tensor(
        self._jnt_type.copy(), dtype=torch.int32, device=self._device)
    self._recovery_jnt_dofadr = torch.as_tensor(
        self._jnt_dofadr.copy(), dtype=torch.int32, device=self._device)
    self._recovery_vert_bodyid = torch.as_tensor(
        np.asarray(d.vertbodyid, dtype=np.int32).copy(),
        dtype=torch.int32, device=self._device)
    self._recovery_vert_local = torch.as_tensor(
        np.asarray(d.vert, dtype=np.float32).copy(),
        dtype=torch.float32, device=self._device)
    self._recovery_vert_centered = torch.as_tensor(
        vert_centered.astype(np.int32), dtype=torch.int32, device=self._device)
    self._recovery_node_bodyid = torch.as_tensor(
        np.asarray(d.nodebodyid, dtype=np.int32).copy(),
        dtype=torch.int32, device=self._device)
    self._recovery_node_local = torch.as_tensor(
        np.asarray(d.node, dtype=np.float32).copy(),
        dtype=torch.float32, device=self._device)
    self._recovery_node_centered = torch.as_tensor(
        node_centered.astype(np.int32), dtype=torch.int32, device=self._device)
    self._recovery_point_dims = torch.empty(
        (7 + self.batch_size,), dtype=torch.int32, device=self._device)
    self._recovery_edge_dims = torch.empty(
        (4 + self.batch_size,), dtype=torch.int32, device=self._device)
    self._recovery_edge_force_dims = torch.empty(
        (5,), dtype=torch.int32, device=self._device)
    self._recovery_copy_dims = torch.empty(
        (2 + self.batch_size,), dtype=torch.int32, device=self._device)
    self._recovery_matvec_dims = torch.empty(
        (3 + self.batch_size,), dtype=torch.int32, device=self._device)
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
    self._flexvert_xpos_low = torch.zeros_like(self._flexvert_xpos)
    self._flexvert_xpos_tail = torch.zeros_like(self._flexvert_xpos)
    self._zero_body_pos_low = torch.zeros(
        (b, d.nbody, 3), dtype=torch.float32, device=self._device)
    self._zero_body_pos_tail = torch.zeros_like(self._zero_body_pos_low)
    self._flexvert_xvel = torch.zeros((b, d.nflexvert, 3), dtype=torch.float32, device=self._device)
    self._flexedge_length = torch.zeros((b, d.nflexedge), dtype=torch.float32, device=self._device)
    self._flexedge_velocity = torch.zeros((b, d.nflexedge), dtype=torch.float32, device=self._device)
    self._flexedge_dir = torch.zeros((b, d.nflexedge, 3), dtype=torch.float32, device=self._device)
    self._flexvert_J = torch.zeros((b, d.nflexvert, 3, max(nv, 1)), dtype=torch.float32, device=self._device)
    # Contact rows derive their point Jacobians from the compiled source
    # body-weight/interpolation tables. Keep this legacy spatial-J API lazy so
    # sparse contact profiles do not retain a dense vertex-by-DOF workspace.
    self._flexvert_spatial_J = torch.empty(
        (1,), dtype=torch.float32, device=self._device)
    self._flexvert_spatial_J_allocated = False
    self._flexedge_J = torch.zeros((b, d.nflexedge, max(nv, 1)), dtype=torch.float32, device=self._device)
    self._node_local = torch.tensor(np.array(d.node, copy=True), dtype=torch.float32, device=self._device)
    node_low = (np.asarray(model.flex_node, dtype=np.float64).reshape(-1, 3)
                - np.asarray(d.node, dtype=np.float32).astype(np.float64))
    node_tail = (node_low
                 - node_low.astype(np.float32).astype(np.float64))
    self._node_local_low = torch.as_tensor(
        np.array(node_low, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device)
    self._node_local_tail = torch.as_tensor(
        np.array(node_tail, dtype=np.float32, copy=True), dtype=torch.float32,
        device=self._device)
    self._node0 = torch.tensor(np.array(d.node0, copy=True), dtype=torch.float32, device=self._device)
    self._nodebodyid = torch.tensor(np.array(d.nodebodyid, copy=True), dtype=torch.int64, device=self._device)
    self._nodebodyid32 = torch.as_tensor(
        np.array(d.nodebodyid, dtype=np.int32, copy=True), dtype=torch.int32,
        device=self._device)
    self._node_centered = torch.tensor(node_centered, dtype=torch.bool, device=self._device)
    self._node_centered32 = torch.as_tensor(
        np.array(node_centered, dtype=np.int32, copy=True), dtype=torch.int32,
        device=self._device)
    self._node_direct_dof = []
    for f in range(d.nflex):
      start, count = int(d.nodeadr[f]), int(d.nodenum[f])
      for local in range(count):
        node = start + local
        body = int(d.nodebodyid[node])
        direct = (int(model.body_dofnum[body]) > 0
                  and (bool(d.centered[f])
                       or not np.any(d.node[node] != 0.0)))
        adr = int(model.body_dofadr[body]) if direct else -1
        self._node_direct_dof.append((direct, adr))
    self._node_direct_dof = tuple(self._node_direct_dof)
    self._node_xpos = torch.zeros((b, len(d.nodebodyid), 3), dtype=torch.float32, device=self._device)
    self._node_xpos_low = torch.zeros_like(self._node_xpos)
    self._node_xpos_tail = torch.zeros_like(self._node_xpos)
    self._node_xvel = torch.zeros_like(self._node_xpos)
    self._node_J = torch.zeros((b, len(d.nodebodyid), 3, max(nv, 1)), dtype=torch.float32, device=self._device)
    self._qfrc_passive = torch.zeros((b, max(nv, 1)), dtype=torch.float32, device=self._device)
    self._damping_tangent = torch.zeros((b, max(nv, 1), max(nv, 1)), dtype=torch.float32, device=self._device)
    self._stiffness_tangent = torch.zeros((b, max(nv, 1), max(nv, 1)), dtype=torch.float32, device=self._device)
    # Recovery evaluates a selected subset without borrowing or clearing the
    # accepted ordinary result buffers. These fixed-capacity buffers are only
    # returned by the explicitly masked path.
    self._recovery_qfrc_passive = torch.zeros_like(self._qfrc_passive)
    self._recovery_damping_tangent = torch.zeros_like(self._damping_tangent)
    self._recovery_stiffness_tangent = torch.zeros_like(self._stiffness_tangent)
    self._operator_names = (
        "interp_stiffness", "interp_damped_stiffness",
        "bend_stiffness", "bend_damped_stiffness",
        "edge_velocity_derivative")
    self._operator_workspace = {
        name: torch.zeros((b, nv, nv), dtype=torch.float32, device=self._device)
        for name in self._operator_names}
    self._material_matvec_workspace = {
        name: torch.zeros((b, nv), dtype=torch.float32, device=self._device)
        for name in ("interp_stiffness", "interp_damped_stiffness",
                     "bend_stiffness", "bend_damped_stiffness")}
    # Borrowed output buffers for frozen-POS mjEQ_FLEX replay. The context
    # stores source coefficients/residuals, not just aref, so another qvel
    # stage can refresh the velocity-dependent part exactly.
    self._eq_pos = torch.empty((b, d.nflexedge), dtype=torch.float32, device=self._device)
    self._eq_imp = torch.empty_like(self._eq_pos)
    self._eq_aref = torch.empty_like(self._eq_pos)
    self._eq_R = torch.empty_like(self._eq_pos)
    self._eq_raw_context = torch.empty(
        (b, d.nflexedge, 5), dtype=torch.float32, device=self._device)
    self._equality_position_context = {
        "pos": self._eq_raw_context[..., 0],
        "impedance": self._eq_raw_context[..., 1],
        "b0": self._eq_raw_context[..., 2],
        "k0": self._eq_raw_context[..., 3],
        "diag": self._eq_raw_context[..., 4],
        "edge_J": self._flexedge_J,
        "edge_ids_by_flex": self._equality_edge_ids,
    }
    self._position_context_buffers = {
        name: torch.empty_like(getattr(self, name))
        for name in ("_flexvert_xpos", "_flexvert_xpos_low",
                     "_flexvert_xpos_tail", "_flexvert_J",
                     "_node_xpos", "_node_xpos_low", "_node_xpos_tail",
                     "_node_J", "_flexedge_length",
                     "_flexedge_dir", "_flexedge_J", "_flexvert_eq_pos",
                     "_flexvert_eq_J")}
    self._position_context_buffers["body_pos"] = torch.empty(
        (b, d.nbody, 3), dtype=torch.float32, device=self._device)
    self._position_context_buffers["body_pos_low"] = torch.empty(
        (b, d.nbody, 3), dtype=torch.float32, device=self._device)
    self._position_context_buffers["body_pos_tail"] = torch.empty(
        (b, d.nbody, 3), dtype=torch.float32, device=self._device)
    self._position_context_buffers["body_quat"] = torch.empty(
        (b, d.nbody, 4), dtype=torch.float32, device=self._device)
    self._position_context_buffers["joint_anchor"] = torch.zeros(
        (b, d.njnt, 3), dtype=torch.float32, device=self._device)
    self._position_context_buffers["joint_axis"] = torch.zeros(
        (b, d.njnt, 3), dtype=torch.float32, device=self._device)
    self._flexstrain_context_names = {}
    for eqid, record in self._flexstrain_records.items():
      name = f"flexstrain_{eqid}"
      count = int(record["eigenvectors"].shape[0])
      self._position_context_buffers[name + "_pos"] = torch.empty(
          (b, count), dtype=torch.float32, device=self._device)
      self._position_context_buffers[name + "_J"] = torch.empty(
          (b, count, max(nv, 1)), dtype=torch.float32, device=self._device)
      self._flexstrain_context_names[eqid] = name
    self._kinematics_generation = 0
    self._position_context_generation = 0
    self._latest_position_context = None

  def _add_node_generalized_force(self, node_ids, cartesian_force):
    """Scatter nodal force using pinned flex direct-DOF/applyFT branches."""
    for local, node_value in enumerate(node_ids):
      node = int(node_value)
      direct, adr = self._node_direct_dof[node]
      force = cartesian_force[:, local, :]
      if direct:
        self._qfrc_passive[:, adr:adr + 3].add_(force)
      else:
        jac = self._node_J[:, node, :, :]
        self._qfrc_passive.add_(torch.einsum("bd,bdi->bi", force, jac))

  def _add_node_material_tangent(self, node_ids, cartesian_force,
                                 cartesian_tangent, poses):
    """Scatter material dF/dq with the pinned direct-write/applyFT mapping."""
    body_ids = []
    slow_force = torch.zeros_like(cartesian_force)
    for local, node_value in enumerate(node_ids):
      node = int(node_value)
      direct, adr = self._node_direct_dof[node]
      if direct:
        self._stiffness_tangent[:, adr:adr + 3, :self.descriptor.nv].add_(
            cartesian_tangent[:, local, :, :])
      else:
        jac = self._node_J[:, node, :, :]
        self._stiffness_tangent[:, :self.descriptor.nv, :self.descriptor.nv].add_(
            torch.einsum("bdi,bdq->biq", jac, cartesian_tangent[:, local]))
        slow_force[:, local, :] = cartesian_force[:, local, :]
      body_ids.append(int(self.descriptor.nodebodyid[node]))
    self._add_attachment_tangent(
        np.asarray(body_ids, dtype=np.int32),
        self._node_xpos[:, node_ids, :], self._node_J[:, node_ids, :, :],
        slow_force, poses)

  def run_material_operators(self, qpos, qvel, poses, cvel=None):
    """Return source-faithful frozen flex operators for implicit stepping.

    ``interp_*`` are the compiled, corotated ``J' K J`` operators used by
    ``mjd_flexInterp_mul``; ``bend_*`` are the direct-body-DOF 2-D stencil
    operators used by ``mjd_flexBend_mul``.  They intentionally do not reuse
    the full configuration tangents returned by :meth:`run_device`.
    Buffers are borrowed and overwritten by the next call.
    """
    b, nv = self.batch_size, self.descriptor.nv
    if tuple(qpos.shape) != (b, self.descriptor.nq):
      raise ValueError("qpos must be [batch, nq]")
    if tuple(qvel.shape) != (b, nv):
      raise ValueError("qvel must be [batch, nv]")
    for name in self._operator_names:
      self._operator_workspace[name].zero_()
    self.update_kinematics(poses, cvel)
    out = self._operator_workspace
    damper = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    if nv:
      for fe, node_ids in enumerate(self._interp_nodes_host):
        owner = self._interp_owner_host[fe]
        if bool(self.model.flex_rigid[owner]) or int(self.model.flex_edgeequality[owner]) == 3:
          continue
        npe = self._interp_npe_host[fe]
        ids = self._interp_nodes_long[fe]
        x = self._node_xpos[:, ids, :]
        shape_grad = self._interp_grad[fe, :npe]
        axes = self._interp_axes_host[fe]
        Fparam = torch.einsum("bnd,nk->bdk", x, shape_grad)
        F = torch.zeros_like(Fparam)
        is_volume = len(node_ids) == (abs(int(self.descriptor.interp[owner])) + 1) ** 3
        if is_volume:
          F.copy_(Fparam)
        else:
          F[:, :, axes[0]] = Fparam[:, :, 0]
          F[:, :, axes[1]] = Fparam[:, :, 1]
          F[:, :, axes[2]] = torch.cross(Fparam[:, :, 0], Fparam[:, :, 1], dim=-1)
        # The gathered quaternion is global-to-local. The stiffness rotation
        # therefore follows source Krot = R^T K R with that frame.
        qlocal = self._mat2rot_pinned(F).clone()
        qlocal[:, 1:] *= -1
        R = self._matrix_from_quaternion(qlocal)
        K = self._interp_matrix[fe, :3*npe, :3*npe].reshape(npe, 3, npe, 3)
        Krot = torch.einsum("bad,iajc,bce->bidje", R, K, R)
        Krot = Krot.reshape(b, 3*npe, 3*npe)
        Jflat = self._node_J[:, ids, :, :].reshape(b, 3*npe, nv)
        Kgen = torch.bmm(Jflat.transpose(1, 2), torch.bmm(Krot, Jflat))
        # Pinned flexInterp_cgsolve builds the material operators independently
        # of the passive spring/damper disable bits.
        out["interp_stiffness"].add_(Kgen)
        out["interp_damped_stiffness"].add_(
            self._interp_damping_host[fe] * Kgen)

      for e in range(self._bend_count):
        matrix = self._bend_data[e, :16].reshape(4, 4)
        for i, (adr_i, count_i) in enumerate(zip(
              self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
          if not count_i:
            continue
          for j, (adr_j, count_j) in enumerate(zip(
                self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
            count = min(count_i, count_j)
            if not count:
              continue
            value = matrix[i, j]
            out["bend_stiffness"][:, adr_i:adr_i+count,
                                    adr_j:adr_j+count].diagonal(
                                        dim1=-2, dim2=-1).add_(value)
            out["bend_damped_stiffness"][:, adr_i:adr_i+count,
                                           adr_j:adr_j+count].diagonal(
                                               dim1=-2, dim2=-1).add_(
                                                   self._bend_damping_host[e] * value)

    if damper and nv and self.descriptor.nflexedge:
      out["edge_velocity_derivative"].add_(
          -torch.einsum("e,ben,bem->bnm", self._edge_operator_coeff, self._flexedge_J,
                        self._flexedge_J))
    return out

  def capture_material_operator_context(self, poses, cvel=None):
    """Freeze current kinematics and corotated frames for matrix-free K*x.

    The returned owner/generation token is valid only until the next call that
    updates flex kinematics. It stores one 3x3 frame per interpolation element,
    not a global stiffness matrix.
    """
    self.update_kinematics(poses, cvel)
    rotations = []
    for fe, node_ids in enumerate(self._interp_nodes_host):
      owner = self._interp_owner_host[fe]
      if (bool(self.model.flex_rigid[owner])
          or int(self.model.flex_edgeequality[owner]) == 3):
        rotations.append(None)
        continue
      ids = self._interp_nodes_long[fe]
      x = self._node_xpos[:, ids, :]
      shape_grad = self._interp_grad[fe, :self._interp_npe_host[fe]]
      axes = self._interp_axes_host[fe]
      Fparam = torch.einsum("bnd,nk->bdk", x, shape_grad)
      F = torch.zeros_like(Fparam)
      is_volume = (len(node_ids)
                   == (abs(int(self.descriptor.interp[owner])) + 1) ** 3)
      if is_volume:
        F.copy_(Fparam)
      else:
        F[:, :, axes[0]] = Fparam[:, :, 0]
        F[:, :, axes[1]] = Fparam[:, :, 1]
        F[:, :, axes[2]] = torch.cross(
            Fparam[:, :, 0], Fparam[:, :, 1], dim=-1)
      qlocal = self._mat2rot_pinned(F).clone()
      qlocal[:, 1:] *= -1
      rotations.append(self._matrix_from_quaternion(qlocal))
    return _FlexMaterialOperatorContext(
        self, self._kinematics_generation, tuple(rotations))

  def apply_material_operator_context_device(self, context, vector):
    """Apply all pinned frozen flex K/Kd operators to one vector.

    This matrix-free counterpart to :meth:`run_material_operators` is intended
    for the separate 50-iteration ``flexInterp_cgsolve`` PCG. Interpolated
    records use local ``J*x``, compiled element K, and ``J'`` products; ordinary
    2-D bend records apply their compiled direct-DOF stencil. No `[B,nv,nv]`
    backing is read or constructed. Returned `[B,nv]` vectors are borrowed and
    overwritten by the next product call.
    """
    if (not isinstance(context, _FlexMaterialOperatorContext)
        or context.owner is not self
        or context.generation != self._kinematics_generation):
      raise ValueError("flex material operator context is foreign or stale")
    self._validate_position_tensor(
        "material operator vector", vector,
        (self.batch_size, self.descriptor.nv))
    for output in self._material_matvec_workspace.values():
      output.zero_()
    out = self._material_matvec_workspace
    b, nv = self.batch_size, self.descriptor.nv
    if nv:
      for fe, node_ids in enumerate(self._interp_nodes_host):
        rotation = context.rotations[fe]
        if rotation is None:
          continue
        npe = self._interp_npe_host[fe]
        ids = self._interp_nodes_long[fe]
        Jflat = self._node_J[:, ids, :, :].reshape(b, 3*npe, nv)
        world_velocity = torch.bmm(
            Jflat, vector.unsqueeze(-1)).reshape(b, npe, 3)
        # Krot = R^T K R. Apply it without materializing Krot: first rotate
        # J*x to the compiled material basis, multiply by K, then rotate the
        # element force back before applying J'.
        local_velocity = torch.einsum(
            "bcd,bnd->bnc", rotation, world_velocity).reshape(b, 3*npe, 1)
        K = self._interp_matrix[fe, :3*npe, :3*npe]
        local_force = torch.bmm(
            K.unsqueeze(0).expand(b, -1, -1), local_velocity
        ).reshape(b, npe, 3)
        world_force = torch.einsum(
            "bcd,bnc->bnd", rotation, local_force).reshape(b, 3*npe, 1)
        contribution = torch.bmm(Jflat.transpose(1, 2), world_force).squeeze(-1)
        out["interp_stiffness"].add_(contribution)
        out["interp_damped_stiffness"].add_(
            self._interp_damping_host[fe] * contribution)

      for e in range(self._bend_count):
        matrix = self._bend_data[e, :16].reshape(4, 4)
        damping = self._bend_damping_host[e]
        for i, (adr_i, count_i) in enumerate(zip(
              self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
          if not count_i:
            continue
          for j, (adr_j, count_j) in enumerate(zip(
                self._bend_dofadr_host[e], self._bend_dofnum_host[e])):
            count = min(count_i, count_j)
            if not count:
              continue
            value = matrix[i, j]
            source = vector[:, adr_j:adr_j + count]
            out["bend_stiffness"][:, adr_i:adr_i + count].add_(
                value * source)
            out["bend_damped_stiffness"][:, adr_i:adr_i + count].add_(
                damping * value * source)
    return out

  def run_edge_velocity_derivative_coo_device(self, poses, edge_writer,
                                               cvel=None):
    """Add flex-edge damping qDeriv directly to the shared compiled COO.

    The pinned contribution is ``-Σ_e c[e] J[e]' J[e]``. One Metal thread
    owns each (world, COO slot), reducing over compiled flex edges without
    materializing a dense nv-by-nv derivative or reading device values back.
    """
    if self._device.type != "mps" or self._edge_operator_shader is None:
      raise RuntimeError("native flex edge derivative requires the MPS backend")
    if not isinstance(poses, dict):
      raise ValueError("poses must be the current smooth-pose mapping")
    nv = self.descriptor.nv
    edge_count = int(getattr(edge_writer, "edge_count", -1))
    expected_values = (self.batch_size, max(edge_count, 1))
    if (getattr(edge_writer, "nv", None) != nv or edge_count < 0
        or tuple(getattr(getattr(edge_writer, "values", None), "shape", ()))
        != expected_values):
      raise ValueError("edge_writer does not match flex compiled D COO layout")
    for name in ("_edge_rows", "_edge_cols", "values"):
      value = getattr(edge_writer, name, None)
      if (not isinstance(value, torch.Tensor)
          or value.device.type != "mps" or not value.is_contiguous()):
        raise ValueError(f"edge_writer.{name} must be contiguous MPS storage")
    for name, value in (("_edge_rows", edge_writer._edge_rows),
                        ("_edge_cols", edge_writer._edge_cols)):
      if value.dtype != torch.int32 or tuple(value.shape) != (max(edge_count, 1),):
        raise ValueError(f"edge_writer.{name} must be MPS int32 [max(E,1)]")
    if edge_writer.values.dtype != torch.float32:
      raise ValueError("edge_writer.values must be MPS float32")
    self.update_kinematics(poses, cvel)
    if (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
        or nv == 0 or edge_count == 0 or self.descriptor.nflexedge == 0):
      return edge_writer.values
    if self._edge_operator_coo_edge_count is None:
      self._edge_operator_coo_dims[2] = edge_count
      self._edge_operator_coo_edge_count = edge_count
    elif self._edge_operator_coo_edge_count != edge_count:
      raise ValueError("flex edge COO writer capacity changed after binding")
    self._edge_operator_shader.flex_edge_velocity_derivative_coo(
        self._flexedge_J.reshape(-1), self._edge_operator_coeff,
        edge_writer._edge_rows, edge_writer._edge_cols,
        edge_writer.values.reshape(-1), self._edge_operator_coo_dims,
        threads=(self.batch_size * edge_count,), group_size=(1,))
    return edge_writer.values

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
    # Let torch.cross apply standard leading-dimension broadcasting. Interpolated
    # shell records can have one shared reference vector while the quaternion
    # batch contains several environments; expanding qv to vector's batch
    # shape incorrectly rejects that valid B-by-1 input.
    t = 2.0 * torch.cross(qv, vector, dim=-1)
    return vector + qw * t + torch.cross(qv, t, dim=-1)

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

  def update_kinematics(self, poses: Dict[str, torch.Tensor],
                        cvel: Optional[torch.Tensor] = None,
                        world_mask=None):
    """Compute forward vertex kinematics, velocities, and Jacobians entirely on device MPS."""
    body_pos = poses["body_pos"]    # (B, nbody, 3)
    body_quat = poses["body_quat"]  # (B, nbody, 4)
    b = body_pos.shape[0]

    if world_mask is not None:
      self._update_kinematics_masked(poses, cvel, world_mask)
      return

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
    body_pos_low = poses.get("body_pos_low", self._zero_body_pos_low)
    body_pos_tail = poses.get("body_pos_tail", self._zero_body_pos_tail)
    self._update_paired_point_position_low(
        self._vert_local, self._vert_local_low, self._vert_local_tail,
        self._vertbodyid32, self._vert_centered32, body_pos, body_pos_low,
        body_pos_tail, body_quat, self._flexvert_xpos,
        self._flexvert_xpos_low, self._flexvert_xpos_tail)

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
      self._update_paired_point_position_low(
          self._node_local, self._node_local_low, self._node_local_tail,
          self._nodebodyid32, self._node_centered32, body_pos, body_pos_low,
          body_pos_tail, body_quat, self._node_xpos, self._node_xpos_low,
          self._node_xpos_tail)
      # update_kinematics may be called without velocities after a previous
      # velocity-bearing stage (for example a position-only restore). Never
      # let that leave stale nodal velocities for interpolated flex vertices.
      self._node_xvel.zero_()
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

    # MuJoCo first reconstructs shell interior nodes with TFI, then evaluates
    # every interpolated flex vertex from its compiled Q1/Q2 cell basis.
    if self._has_shell_tfi_nodes:
      if self._device.type == "mps":
        self._shell_tfi_dims[3:].fill_(1)
        self._material_shader.flex_shell_tfi_vectors(
            self._shell_tfi_enabled, self._shell_tfi_nodes,
            self._shell_tfi_weights, self._node_xpos.reshape(-1),
            self._node_xvel.reshape(-1), self._shell_tfi_dims,
            threads=(b * len(self.descriptor.nodebodyid),), group_size=(128,))
        self._update_paired_interpolated_position_low(
            self._shell_tfi_enabled, self._shell_tfi_nodes,
            self._shell_tfi_weights, self._shell_tfi_weights_low,
            self._shell_tfi_weights_tail, self._node_xpos, self._node_xpos_low,
            self._node_xpos_tail, self._shell_tfi_weights.shape[1],
            self._node_xpos, self._node_xpos_low, self._node_xpos_tail,
            world_mask)
      else:
        shell_x = (self._node_xpos[:, self._shell_tfi_nodes]
                   * self._shell_tfi_weights[None, :, :, None]).sum(dim=2)
        shell_v = (self._node_xvel[:, self._shell_tfi_nodes]
                   * self._shell_tfi_weights[None, :, :, None]).sum(dim=2)
        enabled = self._shell_tfi_enabled.bool()[None, :, None]
        self._node_xpos.copy_(torch.where(enabled, shell_x, self._node_xpos))
        self._node_xvel.copy_(torch.where(enabled, shell_v, self._node_xvel))
        self._update_paired_interpolated_position_low(
            self._shell_tfi_enabled, self._shell_tfi_nodes,
            self._shell_tfi_weights, self._shell_tfi_weights_low,
            self._shell_tfi_weights_tail, self._node_xpos, self._node_xpos_low,
            self._node_xpos_tail, self._shell_tfi_weights.shape[1],
            self._node_xpos, self._node_xpos_low, self._node_xpos_tail,
            world_mask)
    if self._has_interpolated_vertices:
      if self._device.type == "mps":
        self._vertex_interp_dims[4:].fill_(1)
        self._material_shader.flex_interpolate_vertices(
            self._vertex_interp_enabled, self._vertex_interp_nodes,
            self._vertex_interp_weights, self._node_xpos.reshape(-1),
            self._node_xvel.reshape(-1), self._vertex_interp_dims,
            self._flexvert_xpos.reshape(-1), self._flexvert_xvel.reshape(-1),
            threads=(b * self.descriptor.nflexvert,), group_size=(128,))
        self._update_paired_interpolated_position_low(
            self._vertex_interp_enabled, self._vertex_interp_nodes,
            self._vertex_interp_weights, self._vertex_interp_weights_low,
            self._vertex_interp_weights_tail, self._node_xpos,
            self._node_xpos_low, self._node_xpos_tail,
            self._vertex_interp_weights.shape[1],
            self._flexvert_xpos, self._flexvert_xpos_low,
            self._flexvert_xpos_tail, world_mask)
      else:
        interp_x = (self._node_xpos[:, self._vertex_interp_nodes]
                    * self._vertex_interp_weights[None, :, :, None]).sum(dim=2)
        interp_v = (self._node_xvel[:, self._vertex_interp_nodes]
                    * self._vertex_interp_weights[None, :, :, None]).sum(dim=2)
        enabled = self._vertex_interp_enabled.bool()[None, :, None]
        self._flexvert_xpos.copy_(torch.where(
            enabled, interp_x, self._flexvert_xpos))
        self._flexvert_xvel.copy_(torch.where(
            enabled, interp_v, self._flexvert_xvel))
        self._update_paired_interpolated_position_low(
            self._vertex_interp_enabled, self._vertex_interp_nodes,
            self._vertex_interp_weights, self._vertex_interp_weights_low,
            self._vertex_interp_weights_tail, self._node_xpos,
            self._node_xpos_low, self._node_xpos_tail,
            self._vertex_interp_weights.shape[1],
            self._flexvert_xpos, self._flexvert_xpos_low,
            self._flexvert_xpos_tail, world_mask)

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
    self._last_kinematics_poses = poses
    self._kinematics_generation += 1

  def _update_paired_point_position_low(
      self, local, local_low, local_tail, body_ids, centered, body_pos,
      body_pos_low, body_pos_tail, body_quat, position_high, position_low,
      position_tail, world_mask=None):
    """Refresh both residual words from source coordinates and FK inputs."""
    if position_high.shape[1] == 0:
      return
    if self._device.type == "mps":
      dims = self._paired_point_dims
      dims[0] = self.batch_size
      dims[1] = int(position_high.shape[1])
      dims[2] = self.descriptor.nbody
      dims[3:].fill_(1)
      if world_mask is not None:
        dims[3:].copy_(world_mask)
      self._material_shader.flex_paired_point_position_low(
          body_pos.reshape(-1), body_pos_low.reshape(-1),
          body_quat.reshape(-1), local.reshape(-1), local_low.reshape(-1),
          body_pos_tail.reshape(-1), local_tail.reshape(-1), body_ids,
          centered, dims, position_high.reshape(-1),
          position_low.reshape(-1), position_tail.reshape(-1),
          threads=(self.batch_size * int(position_high.shape[1]),),
          group_size=(128,))
      return
    # CPU tensor execution is the exact-input source oracle for this producer.
    pos = body_pos.detach().cpu().numpy().astype(np.float64)
    low = body_pos_low.detach().cpu().numpy().astype(np.float64)
    tail = body_pos_tail.detach().cpu().numpy().astype(np.float64)
    quat = body_quat.detach().cpu().numpy().astype(np.float64)
    loc = local.detach().cpu().numpy().astype(np.float64)
    loc += local_low.detach().cpu().numpy().astype(np.float64)
    loc += local_tail.detach().cpu().numpy().astype(np.float64)
    ids = body_ids.detach().cpu().numpy().astype(np.int64)
    center = centered.detach().cpu().numpy().astype(bool)
    q = quat[:, ids]
    v = np.broadcast_to(loc[None], (len(pos), *loc.shape)).copy()
    qv, qw = q[..., 1:], q[..., :1]
    t = 2.0 * np.cross(qv, v)
    rotated = v + qw * t + np.cross(qv, t)
    rotated[:, center] = 0.0
    exact = pos[:, ids] + low[:, ids] + tail[:, ids] + rotated
    high = position_high.detach().cpu().numpy().astype(np.float64)
    low_residual = np.asarray(exact-high, dtype=np.float32)
    tail_residual = np.asarray(exact-high-low_residual.astype(np.float64),
                               dtype=np.float32)
    position_low.copy_(torch.as_tensor(
        low_residual, dtype=position_low.dtype,
        device=position_low.device))
    position_tail.copy_(torch.as_tensor(
        tail_residual, dtype=position_tail.dtype,
        device=position_tail.device))

  def _update_paired_interpolated_position_low(
      self, enabled, indices, weights, weights_low, weights_tail,
      source_high, source_low, source_tail, nterm, target_high, target_low,
      target_tail, world_mask=None):
    """Apply one compiled TFI/Q1/Q2 position map to paired coordinates."""
    ntarget = int(target_high.shape[1])
    nsource = int(source_high.shape[1])
    if ntarget == 0 or nsource == 0:
      return
    if self._device.type == "mps":
      dims = self._paired_interp_dims
      dims[0] = self.batch_size
      dims[1] = ntarget
      dims[2] = nsource
      dims[3] = int(nterm)
      dims[4:].fill_(1)
      if world_mask is not None:
        dims[4:].copy_(world_mask)
      self._material_shader.flex_paired_interpolate_position_low(
          enabled, indices, weights.reshape(-1), weights_low.reshape(-1),
          weights_tail.reshape(-1), source_high.reshape(-1),
          source_low.reshape(-1), source_tail.reshape(-1), dims,
          target_high.reshape(-1), target_low.reshape(-1),
          target_tail.reshape(-1),
          threads=(self.batch_size * ntarget,), group_size=(128,))
      return
    x = (source_high.detach().cpu().numpy().astype(np.float64)
         + source_low.detach().cpu().numpy().astype(np.float64)
         + source_tail.detach().cpu().numpy().astype(np.float64))
    idx = indices.detach().cpu().numpy().astype(np.int64)
    w = (weights.detach().cpu().numpy().astype(np.float64)
         + weights_low.detach().cpu().numpy().astype(np.float64)
         + weights_tail.detach().cpu().numpy().astype(np.float64))
    mask = enabled.detach().cpu().numpy().astype(bool)
    exact = np.zeros((self.batch_size, ntarget, 3), dtype=np.float64)
    for env in range(self.batch_size):
      exact[env] = np.sum(x[env, idx] * w[:, :, None], axis=1)
    target = target_high.detach().cpu().numpy().astype(np.float64)
    delta = exact - target
    old_low = target_low.detach().cpu().numpy().astype(np.float64)
    old_tail = target_tail.detach().cpu().numpy().astype(np.float64)
    delta[:, ~mask] = old_low[:, ~mask] + old_tail[:, ~mask]
    low_residual = np.asarray(delta, dtype=np.float32)
    tail_residual = np.asarray(delta-low_residual.astype(np.float64),
                               dtype=np.float32)
    low_residual[:, ~mask] = old_low[:, ~mask].astype(np.float32)
    tail_residual[:, ~mask] = old_tail[:, ~mask].astype(np.float32)
    target_low.copy_(torch.as_tensor(
        low_residual, dtype=target_low.dtype,
        device=target_low.device))
    target_tail.copy_(torch.as_tensor(
        tail_residual, dtype=target_tail.dtype, device=target_tail.device))

  def _update_kinematics_masked(self, poses, cvel, world_mask):
    """Refresh only selected flex rows using per-world guarded MSL kernels."""
    if self._device.type != "mps":
      # CPU tensor evaluation remains a source-comparison backend only.
      self.update_kinematics(poses, cvel)
      return
    shader = self._material_shader or self._bend_shader
    if shader is None:
      raise RuntimeError("masked flex recovery requires the native material shader")
    self._validate_world_mask(world_mask)
    torch = __import__("torch")
    body_pos, body_quat = poses["body_pos"], poses["body_quat"]
    anchors = poses.get("joint_anchor")
    axes = poses.get("joint_axis")
    cvel_present = cvel is not None and cvel.numel() > 0
    cvel_arg = cvel if cvel_present else self._material_dummy_f
    root_com = poses.get("root_com")
    root_arg = root_com if root_com is not None else body_pos
    anchor_arg = anchors if anchors is not None else self._material_dummy_f
    axis_arg = axes if axes is not None else self._material_dummy_f
    spatial = self._flexvert_spatial_J_allocated

    def update_points(local, body_ids, centered, xpos, xvel, jac, npoint,
                      spatial_jac=None):
      if npoint <= 0:
        return
      dims = self._recovery_point_dims
      dims[0] = self.batch_size
      dims[1] = npoint
      dims[2] = self.descriptor.nv
      dims[3] = self.descriptor.nbody
      dims[4] = self.descriptor.njnt
      dims[5] = int(spatial)
      dims[6] = int(cvel_present)
      dims[7:].copy_(world_mask)
      spatial_arg = (spatial_jac if spatial_jac is not None
                     else self._material_dummy_f)
      count = self.batch_size * npoint
      if count > (1 << 31) - 1:
        raise ValueError("masked flex point dispatch exceeds signed int32 indexing")
      shader.flex_recovery_update_points(
          local.reshape(-1), body_ids, centered, body_pos.reshape(-1),
          body_quat.reshape(-1), cvel_arg.reshape(-1), root_arg.reshape(-1),
          anchor_arg.reshape(-1), axis_arg.reshape(-1),
          self._recovery_body_jntadr, self._recovery_body_jntnum,
          self._recovery_body_parentid,
          (self._recovery_jnt_dofadr if self.descriptor.njnt
           else self._material_dummy_i),
          (self._recovery_jnt_type if self.descriptor.njnt
           else self._material_dummy_i), world_mask, dims, xpos.reshape(-1),
          xvel.reshape(-1), jac.reshape(-1), spatial_arg.reshape(-1),
          threads=(count,), group_size=(128,))

    update_points(self._recovery_vert_local, self._recovery_vert_bodyid,
                  self._recovery_vert_centered, self._flexvert_xpos,
                  self._flexvert_xvel, self._flexvert_J,
                  self.descriptor.nflexvert, self._flexvert_spatial_J)
    self._update_paired_point_position_low(
        self._vert_local, self._vert_local_low, self._vert_local_tail,
        self._vertbodyid32, self._vert_centered32, body_pos,
        poses.get("body_pos_low", self._zero_body_pos_low),
        poses.get("body_pos_tail", self._zero_body_pos_tail), body_quat,
        self._flexvert_xpos, self._flexvert_xpos_low,
        self._flexvert_xpos_tail, world_mask)
    update_points(self._recovery_node_local, self._recovery_node_bodyid,
                  self._recovery_node_centered, self._node_xpos,
                  self._node_xvel, self._node_J,
                  len(self.descriptor.nodebodyid))
    self._update_paired_point_position_low(
        self._node_local, self._node_local_low, self._node_local_tail,
        self._nodebodyid32, self._node_centered32, body_pos,
        poses.get("body_pos_low", self._zero_body_pos_low),
        poses.get("body_pos_tail", self._zero_body_pos_tail), body_quat,
        self._node_xpos, self._node_xpos_low, self._node_xpos_tail,
        world_mask)

    if self._has_shell_tfi_nodes:
      self._shell_tfi_dims[3:].copy_(world_mask)
      self._material_shader.flex_shell_tfi_vectors(
          self._shell_tfi_enabled, self._shell_tfi_nodes,
          self._shell_tfi_weights, self._node_xpos.reshape(-1),
          self._node_xvel.reshape(-1), self._shell_tfi_dims,
          threads=(self.batch_size * len(self.descriptor.nodebodyid),),
          group_size=(128,))
      self._update_paired_interpolated_position_low(
          self._shell_tfi_enabled, self._shell_tfi_nodes,
          self._shell_tfi_weights, self._shell_tfi_weights_low,
          self._shell_tfi_weights_tail, self._node_xpos,
          self._node_xpos_low, self._node_xpos_tail,
          self._shell_tfi_weights.shape[1], self._node_xpos,
          self._node_xpos_low, self._node_xpos_tail, world_mask)
    if self._has_interpolated_vertices:
      self._vertex_interp_dims[4:].copy_(world_mask)
      self._material_shader.flex_interpolate_vertices(
          self._vertex_interp_enabled, self._vertex_interp_nodes,
          self._vertex_interp_weights, self._node_xpos.reshape(-1),
          self._node_xvel.reshape(-1), self._vertex_interp_dims,
          self._flexvert_xpos.reshape(-1), self._flexvert_xvel.reshape(-1),
          threads=(self.batch_size * self.descriptor.nflexvert,),
          group_size=(128,))
      self._update_paired_interpolated_position_low(
          self._vertex_interp_enabled, self._vertex_interp_nodes,
          self._vertex_interp_weights, self._vertex_interp_weights_low,
          self._vertex_interp_weights_tail, self._node_xpos,
          self._node_xpos_low, self._node_xpos_tail,
          self._vertex_interp_weights.shape[1], self._flexvert_xpos,
          self._flexvert_xpos_low, self._flexvert_xpos_tail, world_mask)
    if self._has_shell_tfi_nodes and self.descriptor.nv:
      self._material_shader.flex_shell_tfi_jacobian(
          self._shell_tfi_enabled, self._shell_tfi_nodes,
          self._shell_tfi_weights, self._node_J.reshape(-1),
          self._shell_tfi_dims,
          threads=(self.batch_size * len(self.descriptor.nodebodyid) * 3
                   * self.descriptor.nv,), group_size=(128,))
    if self._has_interpolated_vertices and self.descriptor.nv:
      self._material_shader.flex_interpolate_vertex_jacobian(
          self._vertex_interp_enabled, self._vertex_interp_nodes,
          self._vertex_interp_weights, self._node_J.reshape(-1),
          self._vertex_interp_dims, self._flexvert_J.reshape(-1),
          threads=(self.batch_size * self.descriptor.nflexvert * 3
                   * self.descriptor.nv,), group_size=(128,))
    if self._flexvert_spatial_J_allocated:
      dims = self._recovery_edge_dims
      dims[0] = self.batch_size
      dims[1] = self.descriptor.nflexvert
      dims[2] = self.descriptor.nv
      dims[3] = self.descriptor.nflexedge
      dims[4:].copy_(world_mask)
      self._material_shader.flex_recovery_copy_linear_spatial(
          world_mask, self._flexvert_J.reshape(-1),
          self._flexvert_spatial_J.reshape(-1),
          self._recovery_edge_dims, threads=(self.batch_size
                                             * self.descriptor.nflexvert
                                             * 3 * max(self.descriptor.nv, 1),),
          group_size=(128,))
    if self.descriptor.nflexedge:
      dims = self._recovery_edge_dims
      dims[0] = self.batch_size
      dims[1] = self.descriptor.nflexedge
      dims[2] = self.descriptor.nv
      dims[3] = self.descriptor.nflexvert
      dims[4:].copy_(world_mask)
      edge_count = self.batch_size * self.descriptor.nflexedge
      if edge_count > (1 << 31) - 1:
        raise ValueError("masked flex edge dispatch exceeds signed int32 indexing")
      shader.flex_recovery_update_edges(
          self._recovery_edge.reshape(-1),
          self._edge_j_rowadr, self._edge_j_rownnz, self._edge_j_colind,
          self._flexvert_xpos.reshape(-1), self._flexvert_xvel.reshape(-1),
          self._flexvert_J.reshape(-1), world_mask, dims,
          self._flexedge_length.reshape(-1),
          self._flexedge_velocity.reshape(-1), self._flexedge_dir.reshape(-1),
          self._flexedge_J.reshape(-1), threads=(edge_count,),
          group_size=(128,))

  def capture_position_context(self):
    """Capture frozen POS geometry/J buffers for later RK4 velocity stages.

    The returned context is borrowed from this manager and is invalidated by a
    newer capture.  It owns copies of every material position/J buffer plus
    the body/joint pose fields used by articulated force derivatives.
    """
    if self._kinematics_generation == 0:
      raise RuntimeError("flex kinematics must be updated before POS capture")
    buffers = self._position_context_buffers
    poses = self._last_kinematics_poses
    expected = {"body_pos": (self.batch_size, self.descriptor.nbody, 3),
                "body_quat": (self.batch_size, self.descriptor.nbody, 4)}
    if self.descriptor.njnt:
      expected.update({"joint_anchor":
                       (self.batch_size, self.descriptor.njnt, 3),
                       "joint_axis":
                       (self.batch_size, self.descriptor.njnt, 3)})
    for name, shape in expected.items():
      value = poses.get(name)
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != torch.float32
          or not _same_torch_device(value.device, self._device)
          or not value.is_contiguous()):
        raise ValueError(
            f"POS capture requires contiguous {name} on {self._device} with shape {shape}")
    # A failed copy after this point may have partially overwritten buffers,
    # so invalidate the previous token before beginning the first write.
    self._latest_position_context = None
    for name in ("_flexvert_xpos", "_flexvert_xpos_low",
                 "_flexvert_xpos_tail", "_flexvert_J",
                 "_node_xpos", "_node_xpos_low", "_node_xpos_tail",
                 "_node_J", "_flexedge_length",
                 "_flexedge_dir", "_flexedge_J"):
      buffers[name].copy_(getattr(self, name))
    self._compute_flexvert_equalities(
        buffers["_flexvert_xpos"], buffers["_flexvert_J"])
    buffers["_flexvert_eq_pos"].copy_(self._flexvert_eq_pos)
    buffers["_flexvert_eq_J"].copy_(self._flexvert_eq_J)
    for eqid, name in self._flexstrain_context_names.items():
      pos, jac = self._flexstrain_pos_jac(self._flexstrain_records[eqid])
      buffers[name + "_pos"].copy_(pos)
      buffers[name + "_J"].copy_(jac)
    # The last poses dictionary used by update_kinematics is retained only as
    # an alias. Copying happens here so a later integration stage may replace
    # its arrays without changing this position context.
    buffers["body_pos"].copy_(poses["body_pos"])
    buffers["body_pos_low"].copy_(
        poses.get("body_pos_low", self._zero_body_pos_low))
    buffers["body_pos_tail"].copy_(
        poses.get("body_pos_tail", self._zero_body_pos_tail))
    buffers["body_quat"].copy_(poses["body_quat"])
    for name in ("joint_anchor", "joint_axis"):
      value = poses.get(name)
      if value is None:
        if self.descriptor.njnt:
          raise ValueError(f"POS capture requires joint pose field {name}")
        buffers[name].zero_()
      else:
        buffers[name].copy_(value)
    self._position_context_generation += 1
    context = _FlexPositionContext(
        self, self._position_context_generation, buffers)
    self._latest_position_context = context
    return context

  def _masked_recovery_copy(self, source, target, world_mask):
    """Copy a captured per-world POS row only where recovery is selected."""
    b = self.batch_size
    if source.shape != target.shape or source.shape[0] != b:
      raise ValueError("flex recovery copy requires matching batched buffers")
    width = source.numel() // b
    if width == 0:
      return
    total = b * width
    if total > (1 << 31) - 1:
      raise ValueError("masked flex recovery copy exceeds signed int32 indexing")
    dims = self._recovery_copy_dims
    dims[0] = b
    dims[1] = width
    dims[2:].copy_(world_mask)
    self._material_shader.flex_recovery_masked_copy(
        source.reshape(-1), world_mask, dims, target.reshape(-1),
        threads=(total,), group_size=(128,))

  def _masked_recovery_matvec(self, jacobian, qvel, output, world_mask):
    """Evaluate J*qvel for selected flex worlds without touching others."""
    b, nv = self.batch_size, self.descriptor.nv
    width = output.numel() // b
    if width == 0:
      return
    total = b * width
    if total > (1 << 31) - 1:
      raise ValueError("masked flex recovery matvec exceeds signed int32 indexing")
    dims = self._recovery_matvec_dims
    dims[0] = b
    dims[1] = width
    dims[2] = nv
    dims[3:].copy_(world_mask)
    dummy = self._material_dummy_f
    self._material_shader.flex_recovery_masked_matvec(
        jacobian.reshape(-1) if nv else dummy,
        qvel.reshape(-1) if nv else dummy,
        world_mask, dims, output.reshape(-1), threads=(total,),
        group_size=(128,))

  def run_velocity_device(self, context, qpos, qvel, poses=None, cvel=None,
                          world_mask=None):
    """Re-evaluate passive flex forces at captured positions for new qvel.

    No forward kinematics or collision/strain position update is performed.
    Vertex/node velocities and edge rates are refreshed from the captured
    point/edge Jacobians times the supplied generalized velocity. ``qpos``,
    ``poses`` and ``cvel`` are accepted to match the integration hook; the
    captured body/joint pose and positions are authoritative for this frozen
    stage, and cvel is not used to reconstruct them.
    """
    if (not isinstance(context, _FlexPositionContext)
        or context.owner is not self
        or context is not self._latest_position_context
        or context.generation != self._position_context_generation):
      raise ValueError("flex POS context is foreign, stale, or invalidated")
    b, nv = self.batch_size, self.descriptor.nv
    self._validate_position_tensor("qpos", qpos, (b, self.descriptor.nq))
    self._validate_position_tensor("qvel", qvel, (b, nv))
    self._validate_world_mask(world_mask)
    del poses, cvel
    buffers = context.buffers
    if world_mask is not None and self._device.type == "mps":
      for name in ("_flexvert_xpos", "_flexvert_xpos_low", "_flexvert_J",
                   "_flexvert_xpos_tail", "_node_xpos", "_node_xpos_low",
                   "_node_xpos_tail", "_node_J", "_flexedge_length",
                   "_flexedge_dir", "_flexedge_J", "_flexvert_eq_pos",
                   "_flexvert_eq_J"):
        self._masked_recovery_copy(
            buffers[name], getattr(self, name), world_mask)
    else:
      for name in ("_flexvert_xpos", "_flexvert_xpos_low",
                   "_flexvert_xpos_tail", "_flexvert_J",
                   "_node_xpos", "_node_xpos_low", "_node_xpos_tail",
                   "_node_J", "_flexedge_length",
                   "_flexedge_dir", "_flexedge_J"):
        getattr(self, name).copy_(buffers[name])
      self._flexvert_eq_pos.copy_(buffers["_flexvert_eq_pos"])
      self._flexvert_eq_J.copy_(buffers["_flexvert_eq_J"])
      self._last_kinematics_poses = {
        "body_pos": buffers["body_pos"],
        "body_pos_low": buffers["body_pos_low"],
        "body_pos_tail": buffers["body_pos_tail"],
        "body_quat": buffers["body_quat"],
        "joint_anchor": buffers["joint_anchor"],
        "joint_axis": buffers["joint_axis"],
      }
    position_poses = {
        "body_pos": buffers["body_pos"],
        "body_pos_low": buffers["body_pos_low"],
        "body_pos_tail": buffers["body_pos_tail"],
        "body_quat": buffers["body_quat"],
        "joint_anchor": buffers["joint_anchor"],
        "joint_axis": buffers["joint_axis"],
    }
    if nv:
      if world_mask is not None and self._device.type == "mps":
        self._masked_recovery_matvec(
            self._flexvert_J, qvel, self._flexvert_xvel, world_mask)
        if len(self.descriptor.nodebodyid):
          self._masked_recovery_matvec(
              self._node_J, qvel, self._node_xvel, world_mask)
        if self.descriptor.nflexedge:
          self._masked_recovery_matvec(
              self._flexedge_J, qvel, self._flexedge_velocity, world_mask)
      else:
        self._flexvert_xvel.copy_(torch.einsum(
            "bvcn,bn->bvc", self._flexvert_J, qvel))
        if len(self.descriptor.nodebodyid):
          self._node_xvel.copy_(torch.einsum(
              "bvcn,bn->bvc", self._node_J, qvel))
        if self.descriptor.nflexedge:
          self._flexedge_velocity.copy_(torch.einsum(
              "ben,bn->be", self._flexedge_J, qvel))
    else:
      self._flexvert_xvel.zero_()
      self._node_xvel.zero_()
      self._flexedge_velocity.zero_()
    result_buffers = (self._qfrc_passive, self._damping_tangent,
                      self._stiffness_tangent)
    try:
      return self._compute_passive_from_current(
          qvel, position_poses, world_mask=world_mask)
    finally:
      (self._qfrc_passive, self._damping_tangent,
       self._stiffness_tangent) = result_buffers

  def _validate_position_tensor(self, name, value, shape):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
        or value.dtype != torch.float32
        or not _same_torch_device(value.device, self._device)
        or not value.is_contiguous()):
      raise ValueError(f"{name} must be contiguous float32 on {self._device} with shape {shape}")

  def _validate_world_mask(self, world_mask):
    """Validate the shared active-row ABI used by masked native recovery."""
    if world_mask is None:
      return
    if (not isinstance(world_mask, torch.Tensor)
        or world_mask.dtype != torch.int32
        or tuple(world_mask.shape) != (self.batch_size,)
        or not _same_torch_device(world_mask.device, self._device)
        or not world_mask.is_contiguous()):
      raise ValueError("world_mask must be contiguous int32[B] on the flex device")

  def _compute_flexvert_equalities(self, xpos=None, point_jacobian=None):
    """Compute pinned two-invariant cloth rows and their analytic Jacobian.

    This is the ``mjEQ_FLEXVERT`` law from MuJoCo 3.10: each vertex averages
    weighted neighbor-edge outer products, maps through its compiled 2x2
    reference metric, then constrains ``tr(F'F)-2`` and ``det(F'F)-1``.
    ``flexvert_J`` is formed by the pinned endpoint chain rule and mass scale.
    Topology and all Python loop bounds are immutable lowering data.
    """
    xpos = self._flexvert_xpos if xpos is None else xpos
    point_jacobian = self._flexvert_J if point_jacobian is None else point_jacobian
    out_pos = self._flexvert_eq_pos
    out_jac = self._flexvert_eq_J
    out_pos.zero_()
    out_jac.zero_()
    nv = self.descriptor.nv
    if nv == 0:
      return out_pos, out_jac[..., :0]
    for v, records in enumerate(self._vertex_neighbors_host):
      if not records:
        continue
      A = torch.zeros((self.batch_size, 3, 2),
                      dtype=torch.float32, device=self._device)
      for v1, v2, dx, weight in records:
        dy = xpos[:, v2, :] - xpos[:, v1, :]
        A.add_(weight * dy.unsqueeze(-1) * dx.view(1, 1, 2))
      metric = self._vertmetric[v].reshape(2, 2)
      F = torch.matmul(A, metric)
      f00, f01 = F[:, 0, 0], F[:, 0, 1]
      f10, f11 = F[:, 1, 0], F[:, 1, 1]
      f20, f21 = F[:, 2, 0], F[:, 2, 1]
      c00 = f00.square() + f10.square() + f20.square()
      c01 = f00 * f01 + f10 * f11 + f20 * f21
      c10 = c01
      c11 = f01.square() + f11.square() + f21.square()
      body = int(self.descriptor.vertbodyid[v])
      mass = float(self.descriptor.body_mass[body]) if body >= 0 else 0.0
      scale = math.sqrt(mass) if mass > 1e-15 else 1.0
      out_pos[:, v, 0].copy_((c00 + c11 - 2.0) * scale)
      out_pos[:, v, 1].copy_((c00 * c11 - c01 * c10 - 1.0) * scale)
      FB = torch.matmul(F, metric)
      adj = torch.stack((c11, -c01, -c10, c00), dim=-1).reshape(-1, 2, 2)
      FadjBinv = torch.matmul(torch.matmul(F, adj), metric)
      for v1, v2, dx, weight in records:
        dI1dy1 = -2.0 * weight * torch.matmul(FB, dx)
        dI2dy1 = -2.0 * weight * torch.matmul(FadjBinv, dx)
        J1 = point_jacobian[:, v1, :, :nv]
        J2 = point_jacobian[:, v2, :, :nv]
        out_jac[:, v, 0, :nv].add_(scale * torch.einsum(
            "bd,bdn->bn", dI1dy1, J1 - J2))
        out_jac[:, v, 1, :nv].add_(scale * torch.einsum(
            "bd,bdn->bn", dI2dy1, J1 - J2))
    return out_pos, out_jac[..., :nv]

  def run_flexvert_equalities(self, context=None, qpos=None, qvel=None):
    """Return all compiled ``mjEQ_FLEXVERT`` residual, velocity and J rows.

    Rows use pinned order ``(vertex0 invariant0, vertex0 invariant1, ...)``.
    Optional context selects frozen-POS replay; qvel refresh only evaluates
    ``J*qvel``. Each flex's rows are selected by ``flexvert_row_ids``.
    """
    if context is not None:
      if (not isinstance(context, _FlexPositionContext)
          or context.owner is not self
          or context is not self._latest_position_context
          or context.generation != self._position_context_generation):
        raise ValueError("flex POS context is foreign, stale, or invalidated")
      self._flexvert_eq_pos.copy_(context.buffers["_flexvert_eq_pos"])
      self._flexvert_eq_J.copy_(context.buffers["_flexvert_eq_J"])
    else:
      self._compute_flexvert_equalities()
    pos = self._flexvert_eq_pos.reshape(self.batch_size, -1)
    jac = self._flexvert_eq_J[..., :self.descriptor.nv].reshape(
        self.batch_size, 2 * self.descriptor.nflexvert, self.descriptor.nv)
    if qvel is None:
      vel = self._flexvert_eq_vel.reshape(self.batch_size, -1)
      vel.zero_()
    else:
      self._validate_position_tensor("qvel", qvel,
                                     (self.batch_size, self.descriptor.nv))
      self._flexvert_eq_vel.copy_(torch.einsum("brn,bn->br", jac, qvel).reshape(
          self.batch_size, self.descriptor.nflexvert, 2))
      vel = self._flexvert_eq_vel.reshape(self.batch_size, -1)
    return {"pos": pos, "vel": vel, "J": jac,
            "flex_ids": self._flexvert_eq_flexids}

  def run_flexvert_equality_rows(self, equality_id, qvel, context=None):
    """Produce one compiled ``mjEQ_FLEXVERT`` row block and solver refs.

    Rows are the canonical two invariant constraints per vertex for the
    equality's flex. The returned `pos`, `vel`, `J`, `R` and `aref` use the
    exact pinned row order; row IDs map directly to the global all-flex
    invariant arrays. The position context can refresh velocity without FK.
    """
    equality_id = int(equality_id)
    params = self._flexvert_eq_param_sets.get(equality_id)
    if params is None:
      raise ValueError("equality_id is not a compiled mjEQ_FLEXVERT")
    all_rows = self.run_flexvert_equalities(
        context=context, qvel=qvel)
    f = params["flex_id"]
    vert_start = int(self.model.flex_vertadr[f])
    vert_num = int(self.model.flex_vertnum[f])
    row_start, row_stop = 2 * vert_start, 2 * (vert_start + vert_num)
    pos = all_rows["pos"][:, row_start:row_stop]
    vel = all_rows["vel"][:, row_start:row_stop]
    jac = all_rows["J"][:, row_start:row_stop, :]
    solimp = params["solimp"]
    dmin, dmax = solimp[0], solimp[1]
    width, mid, power = solimp[2], solimp[3], solimp[4]
    abs_pos = torch.abs(pos)
    y = abs_pos / torch.clamp(width, min=1e-15)
    safe_mid = torch.clamp(mid, min=1e-15, max=1.0 - 1e-15)
    low_scale = torch.pow(safe_mid, power - 1.0).reciprocal()
    high_scale = torch.pow(torch.clamp(1.0 - mid, min=1e-15),
                           power - 1.0).reciprocal()
    curve = torch.where(y <= mid,
        low_scale * torch.pow(torch.clamp(y, min=0.0), power),
        1.0 - high_scale * torch.pow(torch.clamp(1.0 - y, min=0.0), power))
    curve_imp = dmin + curve * (dmax - dmin)
    imp = torch.where((width <= 1e-15) | (dmin == dmax),
        0.5 * (dmin + dmax),
        torch.where(y <= 0.0, dmin,
        torch.where(y >= 1.0, dmax, curve_imp)))
    imp = torch.clamp(imp, min=1e-4, max=0.9999)
    diag = params["diag"]
    target_shape = (self.batch_size, vert_num, 2)
    imp_out = self._flexvert_eq_imp[:, vert_start:vert_start + vert_num, :]
    aref_out = self._flexvert_eq_aref[:, vert_start:vert_start + vert_num, :]
    R_out = self._flexvert_eq_R[:, vert_start:vert_start + vert_num, :]
    imp_out.copy_(imp.reshape(target_shape))
    aref_out.copy_((-(params["b0"] * vel
                      + params["k0"] * imp * pos)).reshape(target_shape))
    R_out.copy_(torch.clamp(
        ((1.0 - imp) / imp * diag).reshape(target_shape), min=1e-15))
    return {"pos": pos, "vel": vel, "J": jac,
            "impedance": imp_out.reshape(self.batch_size, -1),
            "R": R_out.reshape(self.batch_size, -1),
            "aref": aref_out.reshape(self.batch_size, -1),
            "diag": diag,
            "solimp": solimp, "solref": params["solref"],
            "b0": params["b0"], "k0": params["k0"],
            "row_ids": self.flexvert_row_ids(f),
            "flex_id": f, "equality_id": equality_id,
            "jdot_included": False,
            "jdot_policy": "pinned_zero_for_flex_equality"}

  def flexvert_row_ids(self, flex_id: int) -> np.ndarray:
    """Return immutable canonical two-row IDs for a flexvert equality."""
    flex_id = int(flex_id)
    if flex_id < 0 or flex_id >= self.descriptor.nflex:
      raise IndexError("flex_id is outside the compiled model")
    if flex_id not in self._flexvert_eq_flexids:
      return _frozen([], np.int32)
    vert_start = int(self.model.flex_vertadr[flex_id])
    vert_num = int(self.model.flex_vertnum[flex_id])
    return _frozen(np.arange(2 * vert_start, 2 * (vert_start + vert_num)),
                   np.int32)

  def flex_equality_position_context(self, context):
    """Return a unified frozen-POS contract for compiled flex equalities.

    Entries are keyed by compiled equality ID and carry canonical source row
    identities, the captured residual/J, solver diagonal and source
    solref/solimp.  Consumers refresh velocity-dependent ``aref`` from
    ``J*qvel``; all three flex equality families have zero ``Jdot*v`` in the
    pinned 3.10 ``mj_Jdotv`` implementation (that routine handles only
    connect and weld rows).

    The arrays and tensors are borrowed and valid until the next POS capture.
    This is producer metadata only: it does not claim that an owning solver
    has allocated or admitted these rows.
    """
    self._validate_flex_position_context(context)
    out = {}
    for eqid, flex_id in self._equality_flex_ids.items():
      edge_ids = self.equality_edge_ids(flex_id)
      params = self._equality_parameter_sets[eqid]
      out[eqid] = {
          "kind": "flex", "flex_id": flex_id,
          "row_ids": edge_ids,
          "pos": context.buffers["_flexedge_length"][:, edge_ids]
                    - self._edge_length0[edge_ids].unsqueeze(0),
          "J": context.buffers["_flexedge_J"][:, edge_ids, :self.descriptor.nv],
          "diag": params["diag"][:, edge_ids],
          "solimp": params["solimp"],
          "solref": params["solref"],
          "jdot_included": False,
          "jdot_policy": "pinned_zero_for_flex_equality",
      }
    for eqid, params in self._flexvert_eq_param_sets.items():
      flex_id = params["flex_id"]
      start = int(self.model.flex_vertadr[flex_id])
      count = int(self.model.flex_vertnum[flex_id])
      row_ids = self.flexvert_row_ids(flex_id)
      out[eqid] = {
          "kind": "flexvert", "flex_id": flex_id, "row_ids": row_ids,
          "pos": context.buffers["_flexvert_eq_pos"][:, start:start + count]
                    .reshape(self.batch_size, -1),
          "J": context.buffers["_flexvert_eq_J"][:, start:start + count]
                    [..., :self.descriptor.nv].reshape(
                        self.batch_size, len(row_ids), self.descriptor.nv),
          "diag": params["diag"].expand(self.batch_size, -1).reshape(
              self.batch_size, -1),
          "solimp": params["solimp"],
          "solref": params["solref"],
          "jdot_included": False,
          "jdot_policy": "pinned_zero_for_flex_equality",
      }
    for eqid, name in self._flexstrain_context_names.items():
      rec = self._flexstrain_records[eqid]
      count = int(rec["eigenvectors"].shape[0])
      out[eqid] = {
          "kind": "flexstrain", "flex_id": rec["flex_id"],
          "row_ids": _frozen(np.arange(count, dtype=np.int32), np.int32),
          "pos": context.buffers[name + "_pos"],
          "J": context.buffers[name + "_J"][..., :self.descriptor.nv],
          "diag": float(rec["diag"]),
          "solimp": rec["solimp"], "solref": rec["solref"],
          "jdot_included": False,
          "jdot_policy": "pinned_zero_for_flex_equality",
      }
    return out

  def _validate_flex_position_context(self, context):
    if (not isinstance(context, _FlexPositionContext)
        or context.owner is not self
        or context is not self._latest_position_context
        or context.generation != self._position_context_generation):
      raise ValueError("flex POS context is foreign, stale, or invalidated")

  def run_flex_equality_rows(self, equality_id, context, qpos, qvel):
    """Dispatch one compiled flex equality through the common row contract.

    ``row_ids`` are stable producer identities, not current efc addresses.
    Solver allocation remains the coupled-constraint owner's responsibility.
    """
    equality_id = int(equality_id)
    if equality_id in self._equality_flex_ids:
      edge_ids = self.equality_edge_ids(self._equality_flex_ids[equality_id])
      pos, aref, R, J = self.run_equalities_velocity(
          context, qpos, qvel, equality_id=equality_id)
      raw = self.equality_position_context
      return {
          "kind": "flex", "equality_id": equality_id,
          "flex_id": self._equality_flex_ids[equality_id],
          "row_ids": edge_ids, "pos": pos[:, edge_ids],
          "vel": torch.einsum("ben,bn->be", J[:, edge_ids, :], qvel),
          "J": J[:, edge_ids, :], "aref": aref[:, edge_ids],
          "R": R[:, edge_ids],
          "diag": raw["diag"][:, edge_ids],
          "impedance": raw["impedance"][:, edge_ids],
          "b0": raw["b0"][:, edge_ids], "k0": raw["k0"][:, edge_ids],
          "jdot_included": False,
          "jdot_policy": "pinned_zero_for_flex_equality",
      }
    if equality_id in self._flexvert_eq_param_sets:
      out = self.run_flexvert_equality_rows(equality_id, qvel, context)
    elif equality_id in self._flexstrain_records:
      out = self.run_flexstrain_equality_rows(equality_id, qvel, context)
    else:
      raise ValueError("equality_id is not a compiled flex equality")
    out = dict(out)
    out["kind"] = "flexvert" if equality_id in self._flexvert_eq_param_sets else "flexstrain"
    if "row_ids" not in out:
      out["row_ids"] = out.get("row_ordinals")
    out["jdot_included"] = False
    out["jdot_policy"] = "pinned_zero_for_flex_equality"
    return out

  def flexstrain_position_context(self, context):
    """Borrowed raw POS row/J terms keyed by compiled flexstrain equality."""
    self._validate_flex_position_context(context)
    out = {}
    for eqid, name in self._flexstrain_context_names.items():
      rec = self._flexstrain_records[eqid]
      out[eqid] = {
          "pos": context.buffers[name + "_pos"],
          "J": context.buffers[name + "_J"][..., :self.descriptor.nv],
          "diag": float(rec["diag"]),
          "solimp": rec["solimp"], "solref": rec["solref"],
          "row_ordinals": np.arange(rec["eigenvectors"].shape[0],
                                     dtype=np.int32),
      }
    return out

  def run_flexstrain_equality_rows(self, equality_id, qvel, context=None):
    """Evaluate one compiled ``mjEQ_FLEXSTRAIN`` element's rows.

    Row identities and eigenvectors are compiled model data. The corotated
    residual and its source-defined frozen-frame Jacobian are evaluated from
    the current flex-node positions/Jacobians. Pinned ``mj_Jdotv`` only adds
    terms for connect/weld equalities and leaves flex rows unchanged, so no
    ``Jdot*qvel`` term is added for this equality family.
    """
    record = self._flexstrain_records.get(int(equality_id))
    if record is None:
      raise ValueError("equality_id is not a compiled flexstrain element")
    self._validate_position_tensor("qvel", qvel,
                                   (self.batch_size, self.descriptor.nv))
    if context is None:
      pos, jac = self._flexstrain_pos_jac(record)
    else:
      if (not isinstance(context, _FlexPositionContext)
          or context.owner is not self
          or context is not self._latest_position_context
          or context.generation != self._position_context_generation):
        raise ValueError("flex POS context is foreign, stale, or invalidated")
      name = self._flexstrain_context_names[int(equality_id)]
      pos = context.buffers[name + "_pos"]
      jac = context.buffers[name + "_J"][..., :self.descriptor.nv]
    vel = (torch.einsum("brv,bv->br", jac, qvel)
           if self.descriptor.nv else torch.zeros_like(pos))
    solimp, solref = record["solimp"], record["solref"]
    dmin, dmax = solimp[0], solimp[1]
    width, mid, power = solimp[2], solimp[3], solimp[4]
    safe_width = torch.clamp(width, min=1e-15)
    y = torch.abs(pos) / safe_width
    safe_mid = torch.clamp(mid, min=1e-15, max=1.0 - 1e-15)
    low = torch.pow(safe_mid, power - 1.0).reciprocal() * torch.pow(
        torch.clamp(y, min=0.0), power)
    high = 1.0 - torch.pow(torch.clamp(1.0 - y, min=0.0), power) / torch.pow(
        torch.clamp(1.0 - mid, min=1e-15), power - 1.0)
    imp = dmin + torch.where(y <= mid, low, high) * (dmax - dmin)
    imp = torch.where((width <= 1e-15) | (dmin == dmax),
                      0.5 * (dmin + dmax),
                      torch.where(y <= 0.0, dmin,
                                  torch.where(y >= 1.0, dmax, imp)))
    imp = imp.clamp(min=1e-4, max=0.9999)
    ref0 = solref[0]
    if not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)):
      ref0 = torch.where(ref0 > 0.0,
                         torch.clamp(ref0, min=2.0*self.descriptor.timestep),
                         ref0)
    b0 = torch.where(solref[1] > 0.0,
        2.0 / torch.clamp(dmax * torch.where(ref0 > 0.0, ref0, 1.0), min=1e-15),
        -solref[1] / torch.clamp(dmax, min=1e-15))
    k0 = torch.where(ref0 > 0.0,
        1.0 / torch.clamp(dmax*dmax*ref0*ref0
                          * torch.where(solref[1] > 0.0,
                                        solref[1]*solref[1], 1.0), min=1e-15),
        -ref0 / torch.clamp(dmax*dmax, min=1e-15))
    diag = torch.full_like(pos, float(record["diag"]))
    R = ((1.0 - imp) / imp * diag).clamp(min=1e-15)
    aref = -(b0 * vel + k0 * imp * pos)
    return {"pos": pos, "vel": vel, "J": jac, "impedance": imp,
            "R": R, "aref": aref, "diag": diag,
            "solimp": solimp, "solref": solref, "b0": b0, "k0": k0,
            "equality_id": int(equality_id),
            "row_ids": self.descriptor.equality_row_ids_by_eqid[
                int(equality_id)],
            "row_ordinals": self.descriptor.equality_row_ids_by_eqid[
                int(equality_id)],
            "jdot_included": False,
            "jdot_policy": "pinned_zero_for_flex_equality"}

  def _flexstrain_pos_jac(self, record):
    ids = record["node_ids_t"]
    x, Jnode = self._node_xpos[:, ids], self._node_J[:, ids]
    order = record["order"]
    if record["shell"]:
      grad = record["shape_grad"]
      tangents = torch.einsum("bnd,nk->bdk", x, grad[:, :2])
      axes = record["axes"]
      F = torch.zeros((self.batch_size, 3, 3), dtype=x.dtype,
                      device=self._device)
      F[:, :, axes[0]] = tangents[:, :, 0]
      F[:, :, axes[1]] = tangents[:, :, 1]
      F[:, :, axes[2]] = torch.cross(tangents[:, :, 0],
                                      tangents[:, :, 1], dim=-1)
    else:
      grad = record["shape_grad"]
      F = torch.einsum("bnd,nk->bdk", x, grad)
    qlocal = self._mat2rot_pinned(F).clone()
    qlocal[:, 1:] *= -1
    displacement = (self._quat_rotate(qlocal, x)
                    - self._node0[ids].unsqueeze(0))
    eig = record["eigenvectors"]
    pos = torch.einsum("enc,bnc->be", eig, displacement)
    qworld = qlocal.clone()
    qworld[:, 1:] *= -1
    world_eig = self._quat_rotate(
        qworld, eig.reshape(1, -1, 3).expand(self.batch_size, -1, -1))
    world_eig = world_eig.reshape(
        self.batch_size, eig.shape[0], eig.shape[1], 3)
    jac = torch.einsum("benc,bncq->beq", world_eig, Jnode)
    return pos, jac[..., :self.descriptor.nv]

  def _compute_jacobians(self, poses: Dict[str, torch.Tensor]):
    """Vectorized calculation of vertex and edge Jacobians on device MPS."""
    self._compute_point_jacobian(
        self._vertbodyid, self._flexvert_xpos, self._flexvert_J, poses)
    if self._flexvert_spatial_J_allocated:
      self._compute_point_angular_jacobian(self._flexvert_spatial_J, poses)
    if len(self.descriptor.nodebodyid):
      self._compute_point_jacobian(
          self._nodebodyid, self._node_xpos, self._node_J, poses)
    if self._has_shell_tfi_nodes and self.descriptor.nv:
      if self._device.type == "mps":
        self._material_shader.flex_shell_tfi_jacobian(
            self._shell_tfi_enabled, self._shell_tfi_nodes,
            self._shell_tfi_weights, self._node_J.reshape(-1),
            self._shell_tfi_dims,
            threads=(self.batch_size * len(self.descriptor.nodebodyid) * 3
                     * self.descriptor.nv,), group_size=(128,))
      else:
        shell_j = (self._node_J[:, self._shell_tfi_nodes]
                   * self._shell_tfi_weights[None, :, :, None, None]).sum(dim=2)
        enabled = self._shell_tfi_enabled.bool()[None, :, None, None]
        self._node_J.copy_(torch.where(enabled, shell_j, self._node_J))
    if self._has_interpolated_vertices and self.descriptor.nv:
      if self._device.type == "mps":
        self._material_shader.flex_interpolate_vertex_jacobian(
            self._vertex_interp_enabled, self._vertex_interp_nodes,
            self._vertex_interp_weights, self._node_J.reshape(-1),
            self._vertex_interp_dims, self._flexvert_J.reshape(-1),
            threads=(self.batch_size * self.descriptor.nflexvert * 3
                     * self.descriptor.nv,), group_size=(128,))
      else:
        interp_j = (self._node_J[:, self._vertex_interp_nodes]
                    * self._vertex_interp_weights[None, :, :, None, None]).sum(dim=2)
        enabled = self._vertex_interp_enabled.bool()[None, :, None, None]
        self._flexvert_J.copy_(torch.where(enabled, interp_j, self._flexvert_J))
      # Interpolated flex vertices are not rigid attachments with one angular
      # velocity. Source flex contact Jacobians use compiled node-body weights;
      # keep the linear generalized Jacobian here exact and avoid presenting
      # placeholder flex_vertbodyid angular rows as physical data.
      if (self._flexvert_spatial_J_allocated
          and self._interpolated_vertex_ids_host):
        self._flexvert_spatial_J[:, self._interpolated_vertex_ids_host, 3:, :] = 0
    if self._flexvert_spatial_J_allocated:
      self._flexvert_spatial_J[:, :, :3, :].copy_(self._flexvert_J)

    # 5. Assemble edge Jacobians: J_e = u_e^T (J_v1 - J_v0)
    e0 = self._edge[:, 0]
    e1 = self._edge[:, 1]
    J0 = self._flexvert_J[:, e0, :, :]  # (B, nflexedge, 3, nv)
    J1 = self._flexvert_J[:, e1, :, :]  # (B, nflexedge, 3, nv)
    dJ = J1 - J0                         # (B, nflexedge, 3, nv)
    u_exp = self._flexedge_dir.unsqueeze(2)  # (B, nflexedge, 1, 3)
    self._flexedge_J.copy_(torch.matmul(u_exp, dJ).squeeze(2))
    # Match engine_passive.c's compiled sparse projection exactly.  Geometric
    # edge motion may depend on more coordinates than the compiled flexedge_J
    # row exposes; those columns must not contribute force, velocity, or D.
    if self.descriptor.nv and self.descriptor.nflexedge:
      if self._device.type == "mps":
        self._edge_operator_shader.flex_edge_jacobian_compiled_support(
            self._edge_j_rowadr, self._edge_j_rownnz, self._edge_j_colind,
            self._edge_j_mask_dims, self._flexedge_J.reshape(-1),
            self._flexedge_velocity.reshape(-1),
            threads=(self.batch_size * self.descriptor.nflexedge,),
            group_size=(64,))
      else:
        for edge, unsupported in enumerate(self._edge_j_unsupported_host):
          if len(unsupported):
            self._flexedge_J[:, edge, unsupported] = 0.0
          if int(self._edge_j_rownnz_host[edge]) == 0:
            self._flexedge_velocity[:, edge] = 0.0

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

  def _compute_point_angular_jacobian(self, output, poses):
    """Fill rotational point-Jacobian rows from compiled body joints."""
    b, nv = self.batch_size, self.descriptor.nv
    output.zero_()
    if nv <= 0:
      return
    body_quat = poses["body_quat"]
    axes = poses.get("joint_axis")
    for index, body_value in enumerate(self.descriptor.vertbodyid):
      curr = int(body_value)
      while curr > 0 and curr < self.descriptor.nbody:
        ja = int(self._body_jntadr[curr])
        jn = int(self._body_jntnum[curr])
        for joint in range(ja, ja + jn):
          if joint < 0 or joint >= self.descriptor.njnt:
            continue
          dof = int(self._jnt_dofadr[joint])
          joint_type = int(self._jnt_type[joint])
          if joint_type == 3 and axes is not None:
            output[:, index, 3:, dof] += axes[:, joint, :]
          elif joint_type in (0, 1):
            quat = body_quat[:, curr, :]
            qw, qv = quat[:, :1], quat[:, 1:]
            for rot_i in range(3):
              unit = torch.zeros((b, 3), dtype=torch.float32,
                                 device=self._device)
              unit[:, rot_i] = 1.0
              t = 2.0 * torch.cross(qv, unit, dim=-1)
              axis = unit + qw * t + torch.cross(qv, t, dim=-1)
              offset = 3 + rot_i if joint_type == 0 else rot_i
              output[:, index, 3:, dof + offset] = axis
        curr = int(self._body_parentid[curr])

  def contact_spatial_jacobians(self):
    """Return the borrowed `[B,nflexvert,6,nv]` point spatial Jacobians.

    Call after `update_kinematics` with the current poses. The translational
    and angular halves use the same body/ancestor DOF mapping.
    """
    if self._kinematics_generation == 0:
      raise RuntimeError("flex kinematics must be updated before requesting contact Jacobians")
    if not self._flexvert_spatial_J_allocated:
      self._flexvert_spatial_J = torch.zeros(
          (self.batch_size, self.descriptor.nflexvert, 6,
           max(self.descriptor.nv, 1)), dtype=torch.float32,
          device=self._device)
      self._flexvert_spatial_J_allocated = True
    self._compute_point_angular_jacobian(
        self._flexvert_spatial_J, self._last_kinematics_poses)
    if self._interpolated_vertex_ids_host:
      self._flexvert_spatial_J[:, self._interpolated_vertex_ids_host, 3:, :] = 0
    self._flexvert_spatial_J[:, :, :3, :].copy_(self._flexvert_J)
    return self._flexvert_spatial_J

  def run_native_contact_rows(self, poses, qvel, diagA_local=None, *, cvel=None,
                              row_workspace=None, include_wake_links=True,
                              packed_jacobian=None, canonical_row_offset=0,
                              world_mask=None):
    """Produce fixed flex-contact candidates, full-frame Jacobians and rows.

    This is the owned flex-side seam for a coupled-constraint allocator.
    ``diagA_local`` may provide the pinned source-derived approximation for
    the immutable flex-contact row block, shaped ``[B, row_capacity]``. For
    admitted plane/vertex slots it is derived on device when omitted. The
    caller copies the returned local rows into its reserved global span; no
    candidate compaction or device-active readback is required. The raw
    replay context is ``[K, B, imp, pos, margin]`` with projected surface
    velocity returned separately. No Jdot contribution is folded into this
    context.

    The detector retains its construction-time fail-closed admission guard:
    this method does not make unsupported element/geometry routes public.
    """
    if self._device.type != "mps":
      raise RuntimeError("native flex contact rows require MPS")
    required = ("geom_pos", "geom_quat", "cdof", "root_com")
    missing = [name for name in required if name not in poses]
    if missing:
      raise ValueError("contact row production requires pose fields: "
                       + ", ".join(missing))
    self._validate_world_mask(world_mask)
    self.update_kinematics(poses, cvel, world_mask=world_mask)
    program = getattr(self, "_contact_program", None)
    sparse_rows = packed_jacobian is not None
    if program is None:
      from .flex_contact import FlexContactProgram
      program = FlexContactProgram(
          self.model, batch_size=self.batch_size, device=str(self._device),
          sparse_rows=sparse_rows)
      self._contact_program = program
    elif bool(getattr(program, "sparse_rows", False)) != sparse_rows:
      raise ValueError(
          "flex contact Jacobian storage mode changed after workspace creation")
    contact_result = program.run_device(
        self._flexvert_xpos, poses["geom_pos"], poses["geom_quat"],
        flexvert_xpos_low=self._flexvert_xpos_low,
        flexvert_xpos_tail=self._flexvert_xpos_tail,
        geom_pos_low=poses.get("geom_pos_low"),
        geom_pos_tail=poses.get("geom_pos_tail"),
        flexvert_spatial_jacobian=self._flexvert_spatial_J,
        cdof=poses["cdof"], root_com=poses["root_com"], qvel=qvel,
        diagA=diagA_local, include_wake_links=include_wake_links,
        row_workspace=row_workspace, packed_jacobian=packed_jacobian,
        canonical_row_offset=canonical_row_offset, world_mask=world_mask)
    row_names = ("R", "aref", "lo", "hi", "row_active",
                 "position_context", "surface_velocity", "row_owner",
                 "row_local", "row_cone", "row_friction", "row_start",
                 "row_span")
    if "workspace_J" in contact_result:
      row_names = ("workspace_J",) + row_names
    if "jacobian_packed" in contact_result:
      row_names += ("jacobian_packed", "canonical_row_offset")
    rows = {("active" if name == "row_active" else name): contact_result[name]
            for name in row_names}
    return {"contact_result": contact_result, "rows": rows}

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
      world_mask: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute passive internal strain forces and analytic derivatives on MPS.

    Returns:
      (qfrc_passive, damping_tangent, stiffness_tangent)
    """
    self._validate_world_mask(world_mask)
    self.update_kinematics(poses, cvel, world_mask=world_mask)
    result_buffers = (self._qfrc_passive, self._damping_tangent,
                      self._stiffness_tangent)
    try:
      return self._compute_passive_from_current(
          qvel, poses, world_mask=world_mask)
    finally:
      (self._qfrc_passive, self._damping_tangent,
       self._stiffness_tangent) = result_buffers

  def _compute_passive_from_current(self, qvel, poses, *, world_mask=None):
    """Evaluate passive materials from already populated POS/velocity buffers."""
    b = self.batch_size
    nv = self.descriptor.nv
    masked_native = world_mask is not None and self._device.type == "mps"

    # For recovery, use private result buffers so healthy rows of the ordinary
    # flex workspaces (which can be borrowed by the accepted forward stage)
    # are not cleared.  Element kernels below also receive the row predicate
    # and return before loading world data for inactive rows.
    if world_mask is None:
      qfrc_passive = self._qfrc_passive
      damping_tangent = self._damping_tangent
      stiffness_tangent = self._stiffness_tangent
      old_result_buffers = None
    else:
      qfrc_passive = self._recovery_qfrc_passive
      damping_tangent = self._recovery_damping_tangent
      stiffness_tangent = self._recovery_stiffness_tangent
      old_result_buffers = (self._qfrc_passive, self._damping_tangent,
                            self._stiffness_tangent)
      self._qfrc_passive = qfrc_passive
      self._damping_tangent = damping_tangent
      self._stiffness_tangent = stiffness_tangent
    qfrc_passive.zero_()
    damping_tangent.zero_()
    stiffness_tangent.zero_()

    spring_enabled = not (
        self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    damper_enabled = not (
        self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    # Pinned mj_passive returns before mj_springdamper when both flags are
    # disabled.  This matters for direct mj_flexPassiveStretch: that routine
    # receives the enable booleans but its compiled squared-length path does
    # not branch on either one, unlike the direct bend and interpolated paths.
    if not spring_enabled and not damper_enabled:
      if old_result_buffers is not None:
        (self._qfrc_passive, self._damping_tangent,
         self._stiffness_tangent) = old_result_buffers
      return qfrc_passive, damping_tangent, stiffness_tangent

    if nv <= 0 or self.descriptor.nflexedge <= 0:
      if old_result_buffers is not None:
        (self._qfrc_passive, self._damping_tangent,
         self._stiffness_tangent) = old_result_buffers
      return qfrc_passive, damping_tangent, stiffness_tangent

    # 1. 1D Mass-spring edge tension forces
    if masked_native:
      # The selected-row Metal force kernel computes tension after its early
      # world-mask guard. Do not run batched Torch formulas over healthy state.
      tension = None
    else:
      k_edge = self._edge_spring_coeff.unsqueeze(0)  # (1, nflexedge)
      c_edge = self._edge_operator_coeff.unsqueeze(0)  # (1, nflexedge)
      dL = self._flexedge_length - self._edge_length0.unsqueeze(0)
      Ldot = self._flexedge_velocity
      if spring_enabled and damper_enabled:
        tension = k_edge * dL + c_edge * Ldot
      elif spring_enabled:
        tension = k_edge * dL
      elif damper_enabled:
        tension = c_edge * Ldot
      else:
        tension = torch.zeros_like(dL)

    # Generalized edge restoring force: qfrc = - sum_e (tension_e * J_e)
    # J_e shape: (B, nflexedge, nv)
    if world_mask is not None and self._device.type == "mps":
      dims = self._recovery_edge_force_dims
      dims[0] = b
      dims[1] = self.descriptor.nflexedge
      dims[2] = nv
      dims[3] = int(spring_enabled)
      dims[4] = int(damper_enabled)
      self._material_shader.flex_recovery_edge_force(
          self._flexedge_length.reshape(-1),
          self._flexedge_velocity.reshape(-1), self._edge_length0,
          self._edge_spring_coeff, self._edge_operator_coeff,
          self._flexedge_J.reshape(-1), world_mask, dims,
          qfrc_passive.reshape(-1), threads=(b * max(nv, 1),),
          group_size=(128,))
    else:
      qfrc_edges = -torch.sum(tension.unsqueeze(-1) * self._flexedge_J, dim=1)
      if world_mask is None:
        qfrc_passive.add_(qfrc_edges)
      else:
        selected1 = world_mask.to(dtype=torch.bool).reshape(b, 1)
        qfrc_passive.copy_(torch.where(selected1, qfrc_edges,
                                       torch.zeros_like(qfrc_passive)))

    # Exact edge spring/damper tangents for fixed point Jacobians. The first
    # term is the familiar projected outer product. A stretched edge also
    # contributes geometric stiffness from the changing unit direction; the
    # Rayleigh-rate derivative contributes its nonsymmetric qpos term.
    if not masked_native:
      c_eff = c_edge if damper_enabled else torch.zeros_like(c_edge)
      k_eff = k_edge if spring_enabled else torch.zeros_like(k_edge)
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
      damping_tangent.add_(damp_tang)
      stiffness_tangent.add_(stiff_tang)
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
      self._compute_pinned_stretch(poses, qvel, world_mask=world_mask)
    if self._bend_count:
      self._compute_pinned_bend(qvel, poses, world_mask=world_mask)
    if self._interp_count or self._shell_bend_count:
      if self._interp_count and self._device.type == "mps":
        self._compute_interpolated_mps(poses, qvel, world_mask=world_mask)
      elif self._interp_count:
        self._compute_interpolated(poses, qvel)
      if self._shell_bend_count:
        if self._device.type == "mps":
          self._compute_interpolated_shell_bend_mps(
              qvel, poses, world_mask=world_mask)
        else:
          self._compute_interpolated_shell_bend(qvel, poses)

    if world_mask is not None:
      if not masked_native:
        selected_qfrc = world_mask.to(dtype=torch.bool).reshape(b, 1)
        selected_tangent = selected_qfrc.reshape(b, 1, 1)
        qfrc_passive.copy_(torch.where(
            selected_qfrc, qfrc_passive, torch.zeros_like(qfrc_passive)))
        damping_tangent.copy_(torch.where(
            selected_tangent, damping_tangent,
            torch.zeros_like(damping_tangent)))
        stiffness_tangent.copy_(torch.where(
            selected_tangent, stiffness_tangent,
            torch.zeros_like(stiffness_tangent)))
      (self._qfrc_passive, self._damping_tangent,
       self._stiffness_tangent) = old_result_buffers
    return qfrc_passive, damping_tangent, stiffness_tangent

  def _compute_interpolated_mps(self, poses, qvel, *, world_mask=None):
    """Dispatch the compiled Q1/Q2 volume element force kernel on MPS."""
    if not self._interp_count:
      return
    b, nv = self.batch_size, self.descriptor.nv
    node_count = len(self.descriptor.nodebodyid)
    elem_force = torch.zeros(
        (b, self._interp_count, nv), dtype=torch.float32, device=self._device)
    elem_node_force = torch.zeros(
        (b, self._interp_count, 81), dtype=torch.float32, device=self._device)
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    damper = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    dims = torch.empty((6 + b,), dtype=torch.int32, device=self._device)
    dims[:6].copy_(torch.tensor(
        [b, nv, self._interp_count, node_count, int(spring), int(damper)],
        dtype=torch.int32, device=self._device))
    dims[6:].fill_(1)
    if world_mask is not None:
      dims[6:].copy_(world_mask)
    self._material_shader.flex_interp_force(
        self._node_xpos.reshape(-1), self._node_xvel.reshape(-1),
        self._node0, self._node_J.reshape(-1), self._interp_nodes.reshape(-1),
        self._interp_matrix.reshape(-1), self._interp_grad.reshape(-1),
        self._interp_meta.reshape(-1), self._interp_damping, dims,
        elem_force.reshape(-1), elem_node_force.reshape(-1),
        threads=(b * self._interp_count,), group_size=(32,))
    # The compiled kernel emits a convenience J'F projection. Pinned MuJoCo
    # has a direct-DOF fast path for centered/origin nodes, however, so gather
    # the Cartesian nodal forces and apply the exact per-node scatter here.
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
        self._add_node_generalized_force(ids, node_force)
        if world_mask is None:
          self._compute_interpolated_tangent(fe, qvel, poses)

  def _compute_interpolated_shell_bend_mps(self, qvel, poses, *, world_mask=None):
    """Dispatch compiled Crouzeix-Raviart shell-bend forces on MPS."""
    b, nv = self.batch_size, self.descriptor.nv
    elem_force = torch.zeros(
        (b, self._shell_bend_count, nv), dtype=torch.float32,
        device=self._device)
    elem_node_force = torch.zeros(
        (b, self._shell_bend_count, 54), dtype=torch.float32,
        device=self._device)
    dims = torch.empty((4 + b,), dtype=torch.int32, device=self._device)
    dims[:4].copy_(torch.tensor(
        [b, nv, self._shell_bend_count, len(self.descriptor.nodebodyid)],
        dtype=torch.int32, device=self._device))
    dims[4:].fill_(1)
    if world_mask is not None:
      dims[4:].copy_(world_mask)
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
    # As with stretch, MuJoCo directly writes forces for eligible nodes at
    # their body origin; the generic element projection cannot encode that.
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
          self._add_node_generalized_force(ids, node_force)
      if spring and world_mask is None:
        for record in self._shell_bend_records_host:
          self._compute_shell_bend_material_tangent(record, qvel, poses)

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
      node_force = f_world + d_world
      self._add_node_generalized_force(node_ids, node_force)
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
    world_force = self._quat_rotate(quat, local_force)
    self._add_node_material_tangent(
        node_ids, world_force,
        df_world.permute(0, 2, 3, 1), poses)

    if damper and damping:
      # d(J'F)/d qvel = J' R (cK) R' J; dJ is configuration-dependent only.
      K4 = K.reshape(npe, 3, npe, 3)
      Krot = torch.einsum("bda,namc,bec->bndme", R, K4, R)
      Krot = Krot.reshape(b, 3*npe, 3*npe)
      Jflat = jac.reshape(b, 3*npe, nv)
      cart_damping_dq = damping * torch.einsum(
          "bidje,bjeq->bidq", Krot.reshape(b, npe, 3, npe, 3),
          jac.reshape(b, npe, 3, nv))
      for local, node_value in enumerate(node_ids):
        direct, adr = self._node_direct_dof[int(node_value)]
        value = cart_damping_dq[:, local].reshape(b, 3, nv)
        if direct:
          self._damping_tangent[:, adr:adr + 3, :nv].add_(value)
        else:
          self._damping_tangent[:, :nv, :nv].add_(
              torch.einsum("bdi,bdq->biq", jac[:, local], value))

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

  def _compute_shell_bend_material_tangent(self, record, qvel, poses):
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
    fda, force_a = [], []
    for n, (g0, g1) in enumerate(grada_np):
      force_a.append(stiffness * (
          g0 * torch.cross(wa, ta1, dim=-1)
          - g1 * torch.cross(wa, ta0, dim=-1)))
      df = stiffness * (
          g0 * (torch.cross(dwa, ta1[:, None, :], dim=-1)
                + torch.cross(wa[:, None, :], dta1, dim=-1))
          - g1 * (torch.cross(dwa, ta0[:, None, :], dim=-1)
                  + torch.cross(wa[:, None, :], dta0, dim=-1)))
      fda.append(df)
    fdb, force_b = [], []
    for n, (g0, g1) in enumerate(gradb_np):
      force_b.append(-stiffness * (
          g0 * torch.cross(wb, tb1, dim=-1)
          - g1 * torch.cross(wb, tb0, dim=-1)))
      df = -stiffness * (
          g0 * (torch.cross(dwb, tb1[:, None, :], dim=-1)
                + torch.cross(wb[:, None, :], dtb1, dim=-1))
          - g1 * (torch.cross(dwb, tb0[:, None, :], dim=-1)
                  + torch.cross(wb[:, None, :], dtb0, dim=-1)))
      fdb.append(df)
    dfa = torch.stack(fda, dim=2)
    dfb = torch.stack(fdb, dim=2)
    self._add_node_material_tangent(
        ida, torch.stack(force_a, dim=1), dfa.permute(0, 2, 3, 1), poses)
    self._add_node_material_tangent(
        idb, torch.stack(force_b, dim=1), dfb.permute(0, 2, 3, 1), poses)

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
    self._add_node_generalized_force(
        np.arange(len(self.descriptor.nodebodyid), dtype=np.int32), node_force)
    for record in self._shell_bend_records_host:
      self._compute_shell_bend_material_tangent(record, qvel, poses)

  def _compute_pinned_stretch(self, poses, qvel, *, world_mask=None):
    """Evaluate MuJoCo 3.10's compiled simplex stiffness representation."""
    b, nv = self.batch_size, self.descriptor.nv
    d = self.descriptor
    if self._device.type == "mps":
      dims = torch.empty((6 + b,), dtype=torch.int32, device=self._device)
      dims[:6].copy_(torch.tensor(
          [b, nv, self._stretch_count, 4, d.nflexedge, d.nflexvert],
          dtype=torch.int32, device=self._device))
      dims[6:].fill_(1)
      if world_mask is not None:
        dims[6:].copy_(world_mask)
      elem_force = torch.zeros(
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
    if world_mask is not None and self._device.type == "mps":
      # The selected-row recovery caller consumes only passive force; the
      # normal forward path owns the implicit tangent workspaces.
      return
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

  def _compute_pinned_bend(self, qvel, poses, *, world_mask=None):
    """Evaluate the compiled 17-scalar per-edge 2D shell bending record."""
    b, nv = self.batch_size, self.descriptor.nv
    d = self.descriptor
    spring = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    damper = not (self._disableflags & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    if self._device.type == "mps":
      dims = torch.empty((4 + b,), dtype=torch.int32, device=self._device)
      dims[:4].copy_(torch.tensor(
          [b, nv, self._bend_count, d.nflexvert],
          dtype=torch.int32, device=self._device))
      dims[4:].fill_(1)
      if world_mask is not None:
        dims[4:].copy_(world_mask)
      element_force = torch.zeros(
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
    if nv and not (world_mask is not None and self._device.type == "mps"):
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
      equality_id: Optional[int] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate bilateral distance equality constraint rows (mjEQ_FLEX).

    Returns:
      (eq_pos, eq_aref, eq_R, eq_J)
    """
    self.update_kinematics(poses, cvel)
    return self._form_equality_rows(
        self._flexedge_length, self._flexedge_velocity, self._flexedge_J,
        self._parameters_for_equality(equality_id))

  def equality_edge_ids(self, flex_id: int) -> np.ndarray:
    """Return immutable compiled edge IDs used by one ``mjEQ_FLEX``.

    MuJoCo omits rigid edges from both equality row count and row emission.
    The returned order is therefore the exact increasing compiled-edge order
    that maps the equality's canonical ``eq_rowadr`` range to edge identity.
    """
    flex_id = int(flex_id)
    if flex_id < 0 or flex_id >= self.descriptor.nflex:
      raise IndexError("flex_id is outside the compiled model")
    return self._equality_edge_ids[flex_id]

  @property
  def equality_position_context(self):
    """Borrowed raw POS terms for refreshing flex equality rows at new qvel.

    ``pos`` is ``length - length0``; ``impedance``, ``b0``, ``k0`` and
    ``diag`` are the pinned solimp/solref/invweight terms. ``edge_J`` is the
    captured full edge Jacobian. Rigid edges remain present in the arrays and
    are excluded by ``edge_ids_by_flex`` when mapping to canonical rows.
    """
    return self._equality_position_context

  def run_equalities_velocity(self, context, qpos, qvel, equality_id=None):
    """Refresh ``mjEQ_FLEX`` velocity terms using captured POS geometry/J.

    This implements the split ``mj_forwardSkip(mjSTAGE_POS)`` semantics:
    the latest owned POS context supplies edge lengths and Jacobians while the
    provided qvel supplies only ``J*qvel``. It does not run forward kinematics.
    Returned arrays and ``equality_position_context`` are borrowed workspaces.
    """
    if (not isinstance(context, _FlexPositionContext)
        or context.owner is not self
        or context is not self._latest_position_context
        or context.generation != self._position_context_generation):
      raise ValueError("flex POS context is foreign, stale, or invalidated")
    self._validate_position_tensor("qpos", qpos,
                                   (self.batch_size, self.descriptor.nq))
    self._validate_position_tensor("qvel", qvel,
                                   (self.batch_size, self.descriptor.nv))
    buffers = context.buffers
    self._flexedge_length.copy_(buffers["_flexedge_length"])
    self._flexedge_J.copy_(buffers["_flexedge_J"])
    if self.descriptor.nv:
      self._flexedge_velocity.copy_(torch.einsum(
          "ben,bn->be", self._flexedge_J, qvel))
    else:
      self._flexedge_velocity.zero_()
    return self._form_equality_rows(
        self._flexedge_length, self._flexedge_velocity, self._flexedge_J,
        self._parameters_for_equality(equality_id))

  def _parameters_for_equality(self, equality_id):
    if equality_id is None:
      return None
    equality_id = int(equality_id)
    if equality_id not in self._equality_parameter_sets:
      raise ValueError("equality_id is not a compiled mjEQ_FLEX for this model")
    return self._equality_parameter_sets[equality_id]

  def _form_equality_rows(self, lengths, velocity, jacobian, parameters=None):
    params = parameters
    dmin = self._edge_dmin if params is None else params["dmin"]
    dmax = self._edge_dmax if params is None else params["dmax"]
    width = self._edge_width if params is None else params["width"]
    mid = self._edge_mid if params is None else params["mid"]
    power = self._edge_power if params is None else params["power"]
    b0 = self._edge_b0 if params is None else params["b0"]
    k0 = self._edge_k0 if params is None else params["k0"]
    diag = self._edge_diag if params is None else params["diag"]
    pos = self._eq_pos
    torch.sub(lengths, self._edge_length0.unsqueeze(0), out=pos)
    abs_pos = torch.abs(pos)
    safe_width = torch.clamp(width, min=1e-15)
    y = abs_pos / safe_width
    safe_mid = torch.clamp(mid, min=1e-15, max=1.0 - 1e-15)
    one_minus_mid = torch.clamp(1.0 - mid, min=1e-15)
    low_scale = torch.pow(safe_mid, power - 1.0).reciprocal()
    high_scale = torch.pow(one_minus_mid, power - 1.0).reciprocal()
    low_curve = low_scale * torch.pow(torch.clamp(y, min=0.0), power)
    high_curve = 1.0 - high_scale * torch.pow(
        torch.clamp(1.0 - y, min=0.0), power)
    curve = torch.where(y <= mid, low_curve, high_curve)
    curve_imp = dmin + curve * (dmax - dmin)
    flat = (width <= 1e-15) | (dmin == dmax)
    self._eq_imp.copy_(torch.where(flat, 0.5 * (dmin + dmax),
        torch.where(y <= 0.0, dmin,
        torch.where(y >= 1.0, dmax, curve_imp))))
    self._eq_imp.clamp_(min=1e-4, max=0.9999)
    # Pinned mj_referenceConstraint uses KBIP directly, not coefficients
    # divided by impedance: aref = -B*vel - K*imp*(pos-margin). Flex
    # equality rows have zero margin.
    self._eq_aref.copy_(-(b0 * velocity + k0 * self._eq_imp * pos))
    torch.mul((1.0 - self._eq_imp) / self._eq_imp,
              diag, out=self._eq_R)
    self._eq_R.clamp_(min=1e-15)
    raw = self._eq_raw_context
    raw[..., 0].copy_(pos)
    raw[..., 1].copy_(self._eq_imp)
    raw[..., 2].copy_(b0)
    raw[..., 3].copy_(k0)
    raw[..., 4].copy_(diag)
    return pos, self._eq_aref, self._eq_R, jacobian

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
    self._latest_position_context = None
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
