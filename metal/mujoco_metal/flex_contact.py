# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Fixed-topology contact slots for MuJoCo 3.10 flex collision.

The descriptor is model-only: it enumerates eligible flex/rigid and flex/flex
feature pairs, and assigns each potential contact its exact solver row span.
State-dependent narrowphase belongs to the companion Metal program.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_ELLIPTIC = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
_DISABLE_CONTACT = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
_DISABLE_CONSTRAINT = int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
_ENABLE_OVERRIDE = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
_SHADER = Path(__file__).parent / "shaders" / "flex_contact.metal"

_KIND_PLANE_VERTEX = 0
_KIND_GEOM_ELEMENT = 2
_KIND_INTERNAL_VERTEX_ELEMENT = 3
_KIND_ELEMENT_PAIR = 6


def _frozen(value, dtype, shape=None):
  array = np.asarray(value, dtype=dtype, order="C")
  if shape is not None:
    array = array.reshape(shape)
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _contact_parameters(model, item1, item2):
  """Pinned mj_contactParam mixing for geom/flex feature pairs."""
  def get(item):
    kind, index = item
    if kind == "geom":
      return (int(model.geom_priority[index]), int(model.geom_condim[index]),
              float(model.geom_solmix[index]), model.geom_solref[index],
              model.geom_solimp[index], model.geom_friction[index])
    return (int(model.flex_priority[index]), int(model.flex_condim[index]),
            float(model.flex_solmix[index]), model.flex_solref[index],
            model.flex_solimp[index], model.flex_friction[index])

  p1, c1, m1, r1, i1, f1 = get(item1)
  p2, c2, m2, r2, i2, f2 = get(item2)
  if p1 > p2:
    condim, ref, imp, friction = c1, np.array(r1), np.array(i1), np.array(f1)
  elif p2 > p1:
    condim, ref, imp, friction = c2, np.array(r2), np.array(i2), np.array(f2)
  else:
    condim = max(c1, c2)
    if m1 >= 1e-15 and m2 >= 1e-15:
      mix = m1 / (m1 + m2)
    elif m1 < 1e-15 and m2 < 1e-15:
      mix = 0.5
    else:
      mix = 0.0 if m1 < 1e-15 else 1.0
    ref = (mix * np.asarray(r1) + (1.0 - mix) * np.asarray(r2)
           if r1[0] > 0 and r2[0] > 0
           else np.minimum(r1, r2))
    imp = mix * np.asarray(i1) + (1.0 - mix) * np.asarray(i2)
    friction = np.maximum(f1, f2)
  friction5 = np.asarray(
      [friction[0], friction[0], friction[1], friction[2], friction[2]],
      dtype=np.float32)
  return condim, np.asarray(ref, np.float32), np.asarray(imp, np.float32), friction5


def _row_span(condim, cone):
  if condim < 1 or condim > 6:
    raise ValueError(f"MuJoCo flex contact condim must be in [1, 6], got {condim}")
  if condim == 1 or cone == _ELLIPTIC:
    return condim
  return 2 * (condim - 1)


def _assigned_margin(model, source):
  """Match the pinned ``mj_assignMargin`` override behavior."""
  if int(model.opt.enableflags) & _ENABLE_OVERRIDE:
    return float(model.opt.o_margin)
  return float(source)


def _active_elements(model, flex_id, elements):
  """Return element IDs admitted by pinned ``mj_isElemActive`` semantics."""
  if int(model.flex_dim[flex_id]) < 3:
    return range(elements)
  start = int(model.flex_elemadr[flex_id])
  active_layers = int(model.flex_activelayers[flex_id])
  layers = np.asarray(model.flex_elemlayer, dtype=np.int32)
  return (e for e in range(elements)
          if int(layers[start + e]) < active_layers)


def _shares_body(model, vertices1, vertices2):
  """Pinned flex narrowphase excludes features attached to one body."""
  body = np.asarray(model.flex_vertbodyid, dtype=np.int32)
  bodies1 = {int(body[int(v)]) for v in vertices1}
  bodies1.discard(-1)
  if not bodies1:
    return False
  return any(int(body[int(v)]) in bodies1 for v in vertices2)


def _bitmasks_collide(contype1, conaffinity1, contype2, conaffinity2):
  """Opposite of pinned ``filterBitmask``: either directed bit matches."""
  return bool((int(contype1) & int(conaffinity2))
              or (int(contype2) & int(conaffinity1)))


@dataclass(frozen=True)
class FlexContactDescriptor:
  """Immutable candidate feature pairs and their fixed solver row spans."""

  kind: np.ndarray
  flex1: np.ndarray
  elem1: np.ndarray
  vert1: np.ndarray
  flex2: np.ndarray
  elem2: np.ndarray
  vert2: np.ndarray
  geom: np.ndarray
  nodes1: np.ndarray
  nodes2: np.ndarray
  feature1: np.ndarray
  feature2: np.ndarray
  row_start: np.ndarray
  row_span: np.ndarray
  condim: np.ndarray
  cone: np.ndarray
  friction: np.ndarray
  solref: np.ndarray
  solimp: np.ndarray
  radius1: np.ndarray
  radius2: np.ndarray
  geom_radius: np.ndarray
  margin: np.ndarray
  gap: np.ndarray
  feature_capacity: int
  row_capacity: int
  nv: int
  global_enabled: bool
  link_tree_pairs: np.ndarray
  candidate_link_ids: np.ndarray
  max_links_per_candidate: int
  link_capacity: int

  @property
  def slot_count(self):
    return int(self.kind.size)


def lower_flex_contacts(model):
  """Lower geometry filters, contact mixing, and feature-pair capacity.

  Element candidates retain the complete flex simplex and assign a fixed row
  span from mixed ``condim`` and cone. Plane contact retains its pinned
  vertex-based slot layout. No obstacle can overwrite another obstacle's slot.
  """
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("flex contact lowering requires a compiled mujoco.MjModel")
  if mujoco.__version__ != "3.10.0":
    raise RuntimeError("flex contacts require the pinned MuJoCo 3.10.0 model layout")
  cone = int(model.opt.cone)
  rows = []
  flex_elem = np.asarray(model.flex_elem, dtype=np.int32)
  for f in range(int(model.nflex)):
    fmask, famask = int(model.flex_contype[f]), int(model.flex_conaffinity[f])
    if not (fmask or famask):
      continue
    va, vn = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
    ea, en = int(model.flex_elemadr[f]), int(model.flex_elemnum[f])
    eda = int(model.flex_elemdataadr[f])
    nper = int(model.flex_dim[f]) + 1
    active_elements = tuple(_active_elements(model, f, en))
    vert_body = np.asarray(model.flex_vertbodyid, dtype=np.int32)
    elems = [flex_elem[eda + e*nper:eda + (e+1)*nper].astype(np.int32) + va
             for e in range(en)]

    for g in range(int(model.ngeom)):
      body = int(model.geom_bodyid[g])
      if not _bitmasks_collide(
          model.body_contype[body], model.body_conaffinity[body],
          fmask, famask):
        continue
      if not ((int(model.geom_contype[g]) & famask)
              or (fmask & int(model.geom_conaffinity[g]))):
        continue
      condim, solref, solimp, friction = _contact_parameters(
          model, ("geom", g), ("flex", f))
      if condim > 1 and int(model.opt.cone) not in (
          int(mujoco.mjtCone.mjCONE_ELLIPTIC),
          int(mujoco.mjtCone.mjCONE_PYRAMIDAL)):
        raise ValueError("flex friction contacts require a recognized cone")
      span = _row_span(condim, cone)
      margin = _assigned_margin(
          model, float(model.geom_margin[g]) + float(model.flex_margin[f]))
      gap = float(model.geom_gap[g]) + float(model.flex_gap[f])
      gtype = int(model.geom_type[g])
      if gtype == _PLANE:
        for v in range(va, va + vn):
          nodes = np.full(4, -1, np.int32)
          nodes[0] = v
          rows.append((_KIND_PLANE_VERTEX, f, -1, v, -1, -1, -1, g,
                       nodes, np.full(4, -1, np.int32), condim, span,
                       friction, solref, solimp, margin, gap, -1, -1))
      else:
        # The pinned element path sees the complete convex simplex. Its
        # support map chooses active vertices as the CCD direction changes;
        # lowering vertex/face samples here would duplicate physical contacts.
        if (gtype == int(mujoco.mjtGeom.mjGEOM_SDF)
            and int(model.flex_dim[f]) != 2):
          continue
        geom_body = int(model.geom_bodyid[g])
        for e in range(en):
          vertices = elems[e]
          if geom_body >= 0 and np.any(vert_body[vertices] == geom_body):
            continue
          nodes = np.full(4, -1, np.int32)
          nodes[:nper] = vertices
          rows.append((_KIND_GEOM_ELEMENT, f, e, -1, -1, -1, -1, g,
                       nodes, np.full(4, -1, np.int32), condim, span,
                       friction, solref, solimp, margin, gap, -1, -1))

    # Predefined internal contacts are vertex/element feature pairs.
    if (not bool(model.flex_rigid[f])
        and (fmask & famask)
        and bool(model.flex_internal[f])):
      pairadr = int(model.flex_evpairadr[f])
      pairnum = int(model.flex_evpairnum[f])
      evpair = np.asarray(model.flex_evpair, dtype=np.int32).reshape(-1, 2)
      condim, solref, solimp, friction = _contact_parameters(
          model, ("flex", f), ("flex", f))
      span = _row_span(condim, cone)
      for element, vertex in evpair[pairadr:pairadr + pairnum]:
        if 0 <= element < len(elems) and va <= vertex < va + vn:
          nodes = np.full(4, -1, np.int32)
          # mj_collideElemVert treats the element as a complete capsule,
          # triangle, or tetrahedron; the point is the separate feature.
          nodes[:nper] = elems[int(element)]
          rows.append((_KIND_INTERNAL_VERTEX_ELEMENT, f, int(element),
                       -1, f, int(element), int(vertex), -1,
                       nodes, np.full(4, -1, np.int32), condim, span,
                       friction, solref, solimp, 0.0, 0.0, -1, -1))
      if int(model.flex_dim[f]) == 3:
        # The pinned tetrahedral internal pass tests each opposite vertex
        # against the other three vertices of that tetrahedron.
        for element in range(en):
          vertices = elems[element]
          for opposite, vertex in enumerate(vertices):
            nodes = np.full(4, -1, np.int32)
            nodes[:3] = np.delete(vertices, opposite)
            rows.append((_KIND_INTERNAL_VERTEX_ELEMENT, f, element, -1,
                         f, element, int(vertex), -1, nodes,
                         np.full(4, -1, np.int32), 1, 1, friction, solref,
                         solimp, 0.0, 0.0, opposite, -1))

    # Self-collision candidates retain one complete convex pair per element
    # pair. Narrowphase feature selection belongs to the support algorithm,
    # never to a cartesian product of point/edge guesses.
    if (not bool(model.flex_rigid[f])
        and (fmask & famask)
        and int(model.flex_selfcollide[f]) != int(mujoco.mjtFlexSelf.mjFLEXSELF_NONE)):
      condim, solref, solimp, friction = _contact_parameters(
          model, ("flex", f), ("flex", f))
      span = _row_span(condim, cone)
      for e1 in active_elements:
        for e2 in active_elements:
          if e2 <= e1:
            continue
          if _shares_body(model, elems[e1], elems[e2]):
            continue
          nodes1 = np.full(4, -1, np.int32)
          nodes2 = np.full(4, -1, np.int32)
          nodes1[:nper], nodes2[:nper] = elems[e1], elems[e2]
          rows.append((_KIND_ELEMENT_PAIR, f, e1, -1,
                       f, e2, -1, -1, nodes1, nodes2,
                       condim, span, friction, solref, solimp,
                       0.0, 0.0, -1, -1))

  # Cross-flex element features. The opposite bitmask test is the same one
  # used by the pinned body:flex broadphase.
  for f1 in range(int(model.nflex)):
    for f2 in range(f1 + 1, int(model.nflex)):
      if not ((int(model.flex_contype[f1]) & int(model.flex_conaffinity[f2]))
              or (int(model.flex_contype[f2]) & int(model.flex_conaffinity[f1]))):
        continue
      e1a, e1n = int(model.flex_elemadr[f1]), int(model.flex_elemnum[f1])
      e2a, e2n = int(model.flex_elemadr[f2]), int(model.flex_elemnum[f2])
      n1, n2 = int(model.flex_dim[f1]) + 1, int(model.flex_dim[f2]) + 1
      d1, d2 = int(model.flex_elemdataadr[f1]), int(model.flex_elemdataadr[f2])
      va1, va2 = int(model.flex_vertadr[f1]), int(model.flex_vertadr[f2])
      elems1 = [flex_elem[d1 + e*n1:d1 + (e+1)*n1] + va1 for e in range(e1n)]
      elems2 = [flex_elem[d2 + e*n2:d2 + (e+1)*n2] + va2 for e in range(e2n)]
      condim, solref, solimp, friction = _contact_parameters(
          model, ("flex", f1), ("flex", f2))
      span = _row_span(condim, cone)
      for e1 in range(e1n):
        verts1 = elems1[e1]
        for e2 in range(e2n):
          verts2 = elems2[e2]
          if _shares_body(model, verts1, verts2):
            continue
          nodes1 = np.full(4, -1, np.int32)
          nodes2 = np.full(4, -1, np.int32)
          nodes1[:n1], nodes2[:n2] = verts1, verts2
          rows.append((_KIND_ELEMENT_PAIR, f1, e1, -1, f2, e2, -1, -1,
                       nodes1, nodes2, condim, span, friction, solref, solimp,
                       _assigned_margin(
                           model, float(model.flex_margin[f1]
                                        + model.flex_margin[f2])),
                       float(model.flex_gap[f1] + model.flex_gap[f2]), -1, -1))

  count = len(rows)
  kinds = np.asarray([r[0] for r in rows], np.int32)
  row_span = np.asarray([r[11] for r in rows], np.int32)
  row_start = np.cumsum(row_span, dtype=np.int32) - row_span
  def col(index, dtype=np.int32):
    return _frozen([r[index] for r in rows], dtype)
  # Wake-link expansion is static and duplicate-free. Every emitted edge is
  # between distinct moving kinematic trees that can contribute to the
  # candidate's barycentric relative Jacobian.
  vert_body = np.asarray(model.flex_vertbodyid, dtype=np.int32)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  geom_body = np.asarray(model.geom_bodyid, dtype=np.int32)
  link_sets = []
  all_links = set()
  for row in rows:
    _, f1, _e1, v1, f2, _e2, v2, geom, n1, n2, *_ = row
    if v1 >= 0 and v2 < 0 and geom >= 0:
      lhs = {int(body_tree[vert_body[v1]])}
      rhs = {int(body_tree[geom_body[geom]])}
    elif v1 < 0 and v2 >= 0 and geom >= 0:
      lhs = {int(body_tree[vert_body[v2]])}
      rhs = {int(body_tree[geom_body[geom]])}
    else:
      ids1 = [int(v) for v in n1 if int(v) >= 0]
      ids2 = [int(v) for v in n2 if int(v) >= 0]
      if v1 >= 0:
        ids1 = [v1]
      if v2 >= 0:
        ids2 = [v2]
      lhs = {int(body_tree[vert_body[v]]) for v in ids1}
      rhs = {int(body_tree[vert_body[v]]) for v in ids2}
      if geom >= 0:
        rhs = {int(body_tree[geom_body[geom]])}
    pairs = set()
    for a in lhs:
      for btree in rhs:
        if a >= 0 and btree >= 0 and a != btree:
          pairs.add((min(a, btree), max(a, btree)))
    link_sets.append(pairs)
    all_links.update(pairs)
  link_pairs = sorted(all_links)
  link_index = {pair: index for index, pair in enumerate(link_pairs)}
  max_links = max((len(pairs) for pairs in link_sets), default=0)
  candidate_link_ids = np.full((count, max_links), -1, dtype=np.int32)
  for slot, pairs in enumerate(link_sets):
    candidate_link_ids[slot, :len(pairs)] = [link_index[pair] for pair in sorted(pairs)]
  return FlexContactDescriptor(
      kind=_frozen(kinds, np.int32), flex1=col(1), elem1=col(2), vert1=col(3),
      flex2=col(4), elem2=col(5), vert2=col(6), geom=col(7),
      nodes1=_frozen([r[8] for r in rows], np.int32, (count, 4)),
      nodes2=_frozen([r[9] for r in rows], np.int32, (count, 4)),
      row_start=_frozen(row_start, np.int32), row_span=_frozen(row_span, np.int32),
      condim=col(10), cone=_frozen(np.full(count, cone), np.int32),
      friction=_frozen([r[12] for r in rows], np.float32, (count, 5)),
      solref=_frozen([r[13] for r in rows], np.float32, (count, 2)),
      solimp=_frozen([r[14] for r in rows], np.float32, (count, 5)),
      feature1=col(17), feature2=col(18),
      radius1=_frozen([float(model.flex_radius[r[1]]) for r in rows], np.float32),
      radius2=_frozen([float(model.flex_radius[r[4]]) if r[4] >= 0 else 0.0
                       for r in rows], np.float32),
      geom_radius=_frozen([float(model.geom_size[r[7], 0]) if r[7] >= 0 else 0.0
                           for r in rows], np.float32),
      margin=_frozen([r[15] for r in rows], np.float32),
      gap=_frozen([r[16] for r in rows], np.float32),
      feature_capacity=count, row_capacity=int(row_span.sum()), nv=int(model.nv),
      global_enabled=not bool(int(model.opt.disableflags)
                              & (_DISABLE_CONTACT | _DISABLE_CONSTRAINT)),
      link_tree_pairs=_frozen(link_pairs, np.int32).reshape(-1, 2),
      candidate_link_ids=_frozen(candidate_link_ids, np.int32),
      max_links_per_candidate=max_links,
      link_capacity=len(link_pairs))


class FlexContactProgram:
  """Fixed-slot flex contact workspace with a fail-closed narrowphase gate."""

  def __init__(self, model, batch_size=1, device="mps"):
    import torch

    self.model = model
    self.descriptor = lower_flex_contacts(model)
    self.batch_size = int(batch_size)
    self.device = torch.device(device)
    self._torch = torch
    d = self.descriptor
    self._narrowphase_admitted = (
        not d.global_enabled
        or bool(np.all(d.kind == _KIND_PLANE_VERTEX)))
    def tensor(value, dtype=None):
      return torch.as_tensor(np.array(value, copy=True), dtype=dtype,
                             device=self.device)
    self._kind = tensor(d.kind, torch.int32)
    self._flex1 = tensor(d.flex1, torch.int32)
    self._elem1 = tensor(d.elem1, torch.int32)
    self._vert1 = tensor(d.vert1, torch.int32)
    self._flex2 = tensor(d.flex2, torch.int32)
    self._elem2 = tensor(d.elem2, torch.int32)
    self._vert2 = tensor(d.vert2, torch.int32)
    self._geom = tensor(d.geom, torch.int32)
    self._nodes1 = tensor(d.nodes1, torch.int32)
    self._nodes2 = tensor(d.nodes2, torch.int32)
    self._radius1 = tensor(d.radius1, torch.float32)
    self._radius2 = tensor(d.radius2, torch.float32)
    self._margin_gap = tensor(np.stack((d.margin, d.gap), axis=1), torch.float32)
    self._geom_type = tensor(model.geom_type, torch.int32)
    self._geom_size = tensor(model.geom_size, torch.float32)
    self._row_start = tensor(d.row_start, torch.int32)
    self._row_span = tensor(d.row_span, torch.int32)
    self._condim = tensor(d.condim, torch.int32)
    self._cone = tensor(d.cone, torch.int32)
    self._friction = tensor(d.friction, torch.float32)
    self._solref = tensor(d.solref, torch.float32)
    self._solimp = tensor(d.solimp, torch.float32)
    self._feature1 = tensor(d.feature1, torch.int32)
    self._feature2 = tensor(d.feature2, torch.int32)
    self._candidate_link_ids = tensor(d.candidate_link_ids, torch.int32)
    self._link_tree_pairs = tensor(d.link_tree_pairs.reshape(-1), torch.int32)
    self.link_capacity = int(d.link_capacity)
    self._link_dims = torch.tensor(
        [self.batch_size, d.slot_count, d.max_links_per_candidate,
         d.link_capacity, d.link_capacity], dtype=torch.int32,
        device=self.device)
    self._active_links = torch.full(
        (self.batch_size, d.link_capacity, 2), -1, dtype=torch.int32,
        device=self.device)
    self._active_link_overflow = torch.zeros(
        (self.batch_size,), dtype=torch.int32, device=self.device)
    self._shader = (
        torch.mps.compile_shader(_SHADER.read_text())
        if self.device.type == "mps" and d.slot_count else None)

  def run_device(self, flexvert_xpos, geom_pos, geom_quat):
    """Return one immutable-identity result record per admitted candidate.

    Plane/vertex contact has the current source-matched narrowphase. Enabled
    geometry/element and self/cross-element contacts fail closed until their
    support/CCD/manifold implementation is complete. The fixed arrays are
    never compacted in Python.
    """
    torch = self._torch
    d = self.descriptor
    b, nslot = int(flexvert_xpos.shape[0]), d.slot_count
    if b != self.batch_size:
      raise ValueError("flex contact batch size differs from its lowering")
    if nslot == 0:
      empty = torch.empty((b, 0), dtype=torch.float32, device=self.device)
      return {"active": empty.to(torch.bool), "dist": empty,
              "pos": torch.empty((b, 0, 3), device=self.device),
              "normal": torch.empty((b, 0, 3), device=self.device),
              "barycentric1": torch.empty((b, 0, 4), device=self.device),
              "barycentric2": torch.empty((b, 0, 4), device=self.device),
              "row_start": self._row_start, "row_span": self._row_span,
              "condim": self._condim, "cone": self._cone,
              "friction": self._friction, "solref": self._solref,
              "solimp": self._solimp, "geom": self._geom,
              "flex1": self._flex1, "elem1": self._elem1,
              "flex2": self._flex2, "elem2": self._elem2,
              "vert1": self._vert1, "vert2": self._vert2,
              "nodes1": self._nodes1, "nodes2": self._nodes2,
              "feature1": self._feature1, "feature2": self._feature2}
    if d.global_enabled and not self._narrowphase_admitted:
      raise NotImplementedError(
          "flex element/geom and self/cross narrowphase is not admitted until "
          "the pinned support/CCD/manifold implementation is complete")
    active = torch.empty((b, nslot), dtype=torch.int32, device=self.device)
    dist = torch.empty((b, nslot), dtype=torch.float32, device=self.device)
    pos = torch.empty((b, nslot, 3), dtype=torch.float32, device=self.device)
    normal = torch.empty_like(pos)
    bary1 = torch.empty((b, nslot, 4), dtype=torch.float32, device=self.device)
    bary2 = torch.empty_like(bary1)
    if not d.global_enabled:
      active.zero_()
      dist.zero_()
      pos.zero_()
      normal.zero_()
      bary1.zero_()
      bary2.zero_()
    elif self.device.type == "mps":
      dims = torch.tensor(
          [b, nslot, int(flexvert_xpos.shape[1]), int(geom_pos.shape[1])],
          dtype=torch.int32, device=self.device)
      self._shader.flex_contact_detect(
          self._kind, self._flex1, self._elem1, self._vert1,
          self._flex2, self._elem2, self._vert2, self._geom,
          self._nodes1.reshape(-1), self._nodes2.reshape(-1),
          flexvert_xpos.reshape(-1), geom_pos.reshape(-1), geom_quat.reshape(-1),
          self._geom_size.reshape(-1), self._geom_type, self._radius1,
          self._radius2, self._margin_gap.reshape(-1), dims, active.reshape(-1),
          dist.reshape(-1), pos.reshape(-1), normal.reshape(-1),
          bary1.reshape(-1), bary2.reshape(-1),
          threads=(b*nslot,), group_size=(128,))
    else:
      raise RuntimeError("flex contact narrowphase requires native Metal execution")
    return {"active": active.to(torch.bool), "dist": dist, "pos": pos,
            "normal": normal, "barycentric1": bary1,
            "barycentric2": bary2, "row_start": self._row_start,
            "row_span": self._row_span, "condim": self._condim,
            "cone": self._cone, "friction": self._friction,
            "solref": self._solref, "solimp": self._solimp,
            "geom": self._geom, "flex1": self._flex1,
            "elem1": self._elem1, "vert1": self._vert1,
            "flex2": self._flex2, "elem2": self._elem2,
            "vert2": self._vert2, "nodes1": self._nodes1,
            "nodes2": self._nodes2, "feature1": self._feature1,
            "feature2": self._feature2}

  def run_active_tree_links(self, contact_result, capacity=None,
                            out_links=None, out_overflow=None):
    """Map active candidate slots to fixed cross-tree wake links on MPS.

    Returns ``(links, overflow)`` with ``links[B, capacity, 2]`` and
    ``overflow[B]``. Static tree pairs are deduplicated during model lowering;
    the native kernel only tests whether a candidate owning each pair is live.
    """
    torch = self._torch
    if capacity is None:
      capacity = self.link_capacity
    if isinstance(capacity, bool) or int(capacity) != capacity or int(capacity) < 0:
      raise ValueError("capacity must be a nonnegative integer")
    capacity = int(capacity)
    if capacity != self.link_capacity:
      raise ValueError("capacity must match the statically lowered flex link capacity")
    if self.device.type != "mps":
      raise RuntimeError("flex wake-link mapping requires native Metal execution")
    active = contact_result.get("active")
    if (not isinstance(active, torch.Tensor)
        or tuple(active.shape) != (self.batch_size, self.descriptor.slot_count)
        or active.dtype != torch.bool or active.device.type != "mps"
        or not active.is_contiguous()):
      raise ValueError(
          "contact_result active mask must be contiguous MPS bool [batch, slots]")
    links = self._active_links if out_links is None else out_links
    overflow = (self._active_link_overflow if out_overflow is None
                else out_overflow)
    if (not isinstance(links, torch.Tensor)
        or tuple(links.shape) != (self.batch_size, capacity, 2)
        or links.dtype != torch.int32 or links.device.type != "mps"
        or not links.is_contiguous()):
      raise ValueError("out_links must be contiguous MPS int32 [batch, capacity, 2]")
    if (not isinstance(overflow, torch.Tensor)
        or tuple(overflow.shape) != (self.batch_size,)
        or overflow.dtype != torch.int32 or overflow.device.type != "mps"
        or not overflow.is_contiguous()):
      raise ValueError("out_overflow must be contiguous MPS int32 [batch]")
    if self.descriptor.slot_count == 0 or self.link_capacity == 0:
      links.fill_(-1)
      overflow.zero_()
      return links, overflow
    self._shader.flex_contact_tree_links(
        active,
        self._candidate_link_ids.reshape(-1), self._link_tree_pairs,
        self._link_dims, links.reshape(-1), overflow,
        threads=(self.batch_size,), group_size=(1,))
    return links, overflow
