# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Fixed-topology contact slots for MuJoCo 3.10 flex collision.

The descriptor is model-only: it enumerates eligible flex/rigid and flex/flex
feature pairs, and assigns each potential contact its exact solver row span.
State-dependent narrowphase belongs to the companion Metal program.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_ELLIPTIC = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
_DISABLE_CONTACT = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
_DISABLE_CONSTRAINT = int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
_DISABLE_NATIVECCD = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
_DISABLE_MIDPHASE = int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
_ENABLE_OVERRIDE = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
_SHADER_DIR = Path(__file__).parent / "shaders"
_SHADER = _SHADER_DIR / "flex_contact.metal"
_COLLISION_SHADER = _SHADER_DIR / "collision_primitives.metal"
_CONVEX_SHADER = _SHADER_DIR / "convex_narrowphase.metal"
_SDF_SHADER = _SHADER_DIR / "sdf_narrowphase.metal"
_PLUGIN_SDF_SHADER = _SHADER_DIR / "plugin_sdf.metal"
_COMMON_CCD_SHADER = _SHADER_DIR / "common_native_ccd_support.metal"
_COMMON_FLEX_PAIR_SHADER = _SHADER_DIR / "flex_common_pair.metal"

_KIND_PLANE_VERTEX = 0
_KIND_GEOM_ELEMENT = 2
_KIND_INTERNAL_VERTEX_ELEMENT = 3
_KIND_ELEMENT_PAIR = 6
_MAX_FLEX_CONTACTS = 50  # pinned mjMAXCONPAIR in the installed 3.10 runtime
_FLEX_CCD_TRACE_SIZE = 252
_FLEX_CCD_ITER_TRACE_BASE = 32
_FLEX_CCD_ITER_TRACE_DISTANCE_WORDS = 8
_FLEX_CCD_ITER_TRACE_PROJECTION_WORDS = 18
_FLEX_CCD_ITER_TRACE_VERTEX_WORDS = 54
_FLEX_CCD_ITER_TRACE_HEADER = (
    _FLEX_CCD_ITER_TRACE_BASE + _FLEX_CCD_ITER_TRACE_DISTANCE_WORDS
    + _FLEX_CCD_ITER_TRACE_PROJECTION_WORDS)
_FLEX_CCD_ITER_TRACE_HORIZON = 48
_FLEX_GJK_TRACE_STAGE_COUNT = 4  # main and nested tetra query for point/full
_FLEX_GJK_TRACE_STRIDE = 352
_FLEX_GJK_TRACE_BYTES_LIMIT = 256 * 1024 * 1024
# Per-record float offsets are also asserted in test_flex_contact_rows.py.
# Three-word scalars/vectors store (hi, mid, tail); vertices store a,b,m.
_FLEX_GJK_TRACE_VALID_WORD = 351
_FLEX_DETECT_DIMS_FIXED_WORDS = 23


def _flex_pipeline_log(stage, started):
  """Opt-in timing around lazy MPS pipeline creation and dispatch stages."""
  if os.environ.get("MUJOCO_METAL_FLEX_PIPELINE_DIAGNOSTICS") == "1":
    elapsed = time.perf_counter() - started
    print(f"[flex-metal] {stage}: {elapsed:.6f}s", file=sys.stderr,
          flush=True)


def _flex_ccd_trace_capacity(iterations, batch_size=1, slot_count=1):
  """Return bounded per-iteration trace dimensions for the opt-in CCD audit."""
  iterations = max(int(iterations), 0)
  max_faces = 6 * iterations
  stride = (_FLEX_CCD_ITER_TRACE_HEADER + 3 * max_faces
            + _FLEX_CCD_ITER_TRACE_HORIZON
            + _FLEX_CCD_ITER_TRACE_VERTEX_WORDS)
  total = _FLEX_CCD_TRACE_SIZE + iterations * stride
  if int(batch_size) * int(slot_count) * total > _INT32_MAX:
    raise ValueError("flex CCD trace workspace exceeds signed 32-bit indexing")
  return iterations, max_faces, stride, total


def _flex_gjk_trace_capacity(iterations):
  """Return one-candidate GJK trace dimensions, including signed-index guard."""
  iterations = max(int(iterations), 0)
  total = (_FLEX_GJK_TRACE_STAGE_COUNT * iterations
           * _FLEX_GJK_TRACE_STRIDE)
  if total > _INT32_MAX:
    raise ValueError("flex GJK trace exceeds signed 32-bit indexing")
  if total * 4 > _FLEX_GJK_TRACE_BYTES_LIMIT:
    raise ValueError("flex GJK trace exceeds the 256 MiB diagnostic limit")
  return iterations, _FLEX_GJK_TRACE_STRIDE, total


def _flex_trace_slot(slot):
  """Validate a diagnostic target without accepting bools or lossy casts."""
  if (not isinstance(slot, (tuple, list)) or len(slot) != 2
      or any(isinstance(value, (bool, np.bool_))
             or not isinstance(value, (int, np.integer)) for value in slot)):
    raise TypeError("capture_gjk_trace_slot must contain integer (world, slot) ids")
  return int(slot[0]), int(slot[1])


def _flex_combined_trace_capacity(batch_size, slot_count, ccd_trace_size,
                                  gjk_trace_size):
  """Validate the combined diagnostic tail before any Torch allocation."""
  ccd_words = int(batch_size) * int(slot_count) * int(ccd_trace_size)
  total_words = ccd_words + int(gjk_trace_size)
  if (ccd_words < 0 or total_words < 0 or total_words > _INT32_MAX):
    raise ValueError("combined flex CCD/GJK trace exceeds int32 indexing")
  if total_words * 4 > _FLEX_GJK_TRACE_BYTES_LIMIT:
    raise ValueError("combined flex CCD/GJK trace exceeds the 256 MiB diagnostic limit")
  return ccd_words, total_words


def _flex_dims_with_world_mask(torch, values, batch_size, device):
  """Build fixed dims prefix followed by an active/recovery mask suffix."""
  prefix = torch.tensor(values, dtype=torch.int32, device=device)
  suffix = torch.ones((int(batch_size),), dtype=torch.int32, device=device)
  return torch.cat((prefix, suffix))


def _flex_ccd_workspace_capacity(iterations, batch_size=1, slot_count=1):
  """Return pinned mjc_ccd per-query work dimensions after int32 checks."""
  iterations = max(int(iterations), 0)
  max_faces = 6 * iterations  # pinned engine_collision_gjk.c:2378
  max_vertices = 5 + iterations  # pinned mjc_ccdSize allocation
  max_horizon = 24  # pinned Polytope horizon allocation
  max_stack = max(max_faces, 1)  # bounded iterative equivalent of horizonRec
  float_stride = (max(max_vertices, 1) * 39
                  + max(max_faces, 1) * 20 + 3) & ~3
  int_stride = (((max(max_faces, 1) + 1) & ~1)
                + max_horizon * 2 + max_stack * 6)
  products = (
      int(batch_size) * max(int(slot_count), 1) *
      float_stride,
      int(batch_size) * max(int(slot_count), 1) * int_stride,
  )
  if any(value > _INT32_MAX for value in products):
    raise ValueError("flex CCD workspace exceeds signed 32-bit indexing")
  return (max(max_vertices, 1), max(max_faces, 1),
          max_horizon, max_stack)
_CAPSULE_CAPSULE_MAX_CONTACTS = 4  # upper bound of mjraw_CapsuleCapsule
_FILTER_SDF_DIRECT = 3
_INT32_MAX = np.iinfo(np.int32).max


def _flex_position_workspace_elements(batch_size, nflexvert, ngeom):
  """Return exact independently addressed residual-position allocations.

  The flex dispatch slab owns two vertex residual planes and two geometry
  residual planes. The paired element kernel has a separate high/mid/tail
  vertex tensor; each optional-input fallback is an independent zero tensor.
  """
  b = int(batch_size)
  nvtx = int(nflexvert)
  ng = int(ngeom)
  vertex_words = b * nvtx * 3
  geom_words = b * ng * 3
  return {
      "flex_vertex_pair": 3 * vertex_words,
      "flex_vertex_low_tail": 2 * vertex_words,
      "geom_low_tail": 2 * geom_words,
      # These are distinct views into one flat allocation. Validate their
      # combined address range as well as the individual tensor extents.
      "vertex_geom_low_tail_dispatch": 2 * vertex_words + 2 * geom_words,
      "zero_vertex_low": vertex_words,
      "zero_vertex_tail": vertex_words,
      "zero_geom_low": geom_words,
      "zero_geom_tail": geom_words,
  }


def _validate_flex_position_workspace_capacity(batch_size, nflexvert, ngeom):
  """Reject residual-geometry tensors that overflow flattened int32 indexing."""
  sizes = _flex_position_workspace_elements(batch_size, nflexvert, ngeom)
  for name, elements in sizes.items():
    if elements > _INT32_MAX:
      raise ValueError(
          f"flex contact {name} workspace exceeds signed 32-bit indexing")
  return sizes


def _validate_contact_program_capacity(model, descriptor, batch_size):
  """Validate Metal dimensions and flattened offsets before Torch allocation."""
  if (isinstance(batch_size, (bool, np.bool_))
      or not isinstance(batch_size, (int, np.integer))
      or int(batch_size) < 1):
    raise ValueError("batch_size must be a positive integer")
  batch_size = int(batch_size)
  dims = (batch_size, descriptor.slot_count, descriptor.nv,
          int(model.nflex), int(model.nflexvert), int(model.ngeom), int(model.nbody),
          descriptor.feature_capacity, descriptor.row_capacity,
          descriptor.link_capacity, descriptor.max_links_per_candidate)
  if any(value < 0 or value > _INT32_MAX for value in dims):
    raise ValueError("flex contact dimensions must fit signed 32-bit indexing")
  # The Metal kernels flatten these dimensions into signed 32-bit addresses.
  # Check every largest per-slot buffer, plus row/link products, before a
  # device allocation can be attempted.
  products = (
      batch_size * descriptor.slot_count,
      batch_size * descriptor.slot_count * 3,
      batch_size * descriptor.slot_count * 9,
      batch_size * descriptor.slot_count * 6 * max(descriptor.nv, 1),
      batch_size * descriptor.slot_count * 4,
      batch_size * descriptor.row_capacity * max(descriptor.nv, 1),
      batch_size * descriptor.link_capacity * 2,
      descriptor.slot_count * descriptor.max_links_per_candidate,
      int(model.nflex),  # per-flex radius high and residual buffers
  )
  products += tuple(_validate_flex_position_workspace_capacity(
      batch_size, model.nflexvert, model.ngeom).values())
  if any(value > _INT32_MAX for value in products):
    raise ValueError("flex contact workspace exceeds signed 32-bit indexing")
  _flex_ccd_workspace_capacity(
      model.opt.ccd_iterations, batch_size, descriptor.slot_count)
  return batch_size


def _flex_vertices_body_weights(model, flex_id, vertices, vertex_weights):
  """Mirror pinned ``mj_vertBodyWeight`` for weighted compiled vertices."""
  interp = int(model.flex_interp[flex_id])
  vertices = np.asarray(vertices, dtype=np.int64).reshape(-1)
  vertex_weights = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
  if vertices.size != vertex_weights.size:
    raise ValueError("flex vertex and weight counts differ")
  if not vertices.size:
    return {}
  if np.any(vertices < 0) or np.any(vertices >= int(model.nflexvert)):
    raise ValueError("flex vertex id is outside the compiled vertex array")
  if interp == 0:
    body_ids = np.asarray(model.flex_vertbodyid, dtype=np.int64)[vertices]
    weights = {}
    for body, weight in zip(body_ids, vertex_weights):
      body = int(body)
      weights[body] = weights.get(body, 0.0) + float(weight)
    return weights
  order = abs(interp)
  if order not in (1, 2):
    raise NotImplementedError(
        f"pinned flex vertex weights do not support interpolation order {order}")
  node_body = np.asarray(model.flex_nodebodyid, dtype=np.int64)
  if not node_body.size:
    return {}
  vert0 = np.asarray(model.flex_vert0, dtype=np.float64).reshape(-1, 3)
  coord = np.sum(vert0[vertices] * np.abs(vertex_weights)[:, None], axis=0)
  cells = np.asarray(model.flex_cellnum, dtype=np.int64).reshape(-1)[
      3*flex_id:3*flex_id+3]
  if np.any(cells <= 0):
    raise ValueError("interpolated flex vertex has an empty cell grid")
  cell = np.clip(np.floor(coord*cells).astype(np.int64), 0, cells-1)
  local = np.clip(coord*cells-cell, 0.0, 1.0)
  npa = order + 1
  dims = cells*order+1
  ny, nz = int(dims[1]), int(dims[2])
  indices, basis = [], []
  for i in range(npa):
    gi = int(cell[0])*order+i
    for j in range(npa):
      gj = int(cell[1])*order+j
      for k in range(npa):
        gk = int(cell[2])*order+k
        indices.append(gi*ny*nz+gj*nz+gk)
        def phi(x, index):
          if order == 1:
            return 1.0-x if index == 0 else x
          return ((2*x*x-3*x+1) if index == 0 else
                  (4*(x-x*x)) if index == 1 else (2*x*x-x))
        basis.append(phi(local[0], i)*phi(local[1], j)*phi(local[2], k))
  nstart = int(model.flex_nodeadr[flex_id])
  nx = int(dims[0])
  shell = interp < 0
  weights = {}

  def add(index, weight):
    body = int(node_body[nstart+index])
    weights[body] = weights.get(body, 0.0) + float(weight)

  for index, weight in zip(indices, basis):
    if weight < 1e-5:
      continue
    i, rem = divmod(index, ny*nz)
    j, k = divmod(rem, nz)
    if not (shell and 0 < i < nx-1 and 0 < j < ny-1 and 0 < k < nz-1):
      add(index, weight)
      continue
    s, t, u = i/(nx-1), j/(ny-1), k/(nz-1)
    # Exact face, edge-correction, and corner TFI terms from mju_shellTFIWeights.
    terms = (
        ((0, j, k), 1-s), ((nx-1, j, k), s),
        ((i, 0, k), 1-t), ((i, ny-1, k), t),
        ((i, j, 0), 1-u), ((i, j, nz-1), u),
        ((i, 0, 0), -(1-t)*(1-u)),
        ((i, 0, nz-1), -(1-t)*u),
        ((i, ny-1, 0), -t*(1-u)),
        ((i, ny-1, nz-1), -t*u),
        ((0, j, 0), -(1-s)*(1-u)),
        ((0, j, nz-1), -(1-s)*u),
        ((nx-1, j, 0), -s*(1-u)),
        ((nx-1, j, nz-1), -s*u),
        ((0, 0, k), -(1-s)*(1-t)),
        ((0, ny-1, k), -(1-s)*t),
        ((nx-1, 0, k), -s*(1-t)),
        ((nx-1, ny-1, k), -s*t),
        ((0, 0, 0), (1-s)*(1-t)*(1-u)),
        ((0, 0, nz-1), (1-s)*(1-t)*u),
        ((0, ny-1, 0), (1-s)*t*(1-u)),
        ((0, ny-1, nz-1), (1-s)*t*u),
        ((nx-1, 0, 0), s*(1-t)*(1-u)),
        ((nx-1, 0, nz-1), s*(1-t)*u),
        ((nx-1, ny-1, 0), s*t*(1-u)),
        ((nx-1, ny-1, nz-1), s*t*u),
    )
    for (ii, jj, kk), factor in terms:
      add(ii*ny*nz+jj*nz+kk, weight*factor)
  return weights


def _flex_vertex_body_weights(model, flex_id, vertex):
  """Mirror pinned ``mj_vertBodyWeight`` for one compiled flex vertex."""
  return _flex_vertices_body_weights(model, flex_id, (vertex,), (1.0,))


def _flex_element_body_weights(model, flex_id, element, opposite_vertex,
                               contact_pos, flexvert_xpos):
  """Mirror ``mj_elemBodyWeight`` followed by ``mj_vertBodyWeight``.

  ``element`` and ``opposite_vertex`` use the local indices carried by an
  ``mjContact``.  Positions are the current world-space ``flexvert_xpos``;
  this helper is also the scalar oracle for a future device weight kernel.
  """
  dim = int(model.flex_dim[flex_id])
  if dim not in (1, 2, 3):
    raise NotImplementedError(f"unsupported compiled flex dimension {dim}")
  if element < 0 or element >= int(model.flex_elemnum[flex_id]):
    raise ValueError("flex element id is outside the compiled element array")
  adr = int(model.flex_elemdataadr[flex_id]) + int(element)*(dim+1)
  local_vertices = np.asarray(model.flex_elem, dtype=np.int64)[adr:adr+dim+1]
  global_vertices = int(model.flex_vertadr[flex_id]) + local_vertices
  point = np.asarray(contact_pos, dtype=np.float64).reshape(3)
  current = np.asarray(flexvert_xpos, dtype=np.float64).reshape(-1, 3)
  delta = current[global_vertices] - point
  inverse_distance = 1.0 / np.maximum(np.linalg.norm(delta, axis=1), 1e-15)
  if opposite_vertex >= 0:
    keep = local_vertices != int(opposite_vertex)
    global_vertices = global_vertices[keep]
    inverse_distance = inverse_distance[keep]
  total = float(np.sum(inverse_distance))
  if total < 1e-15:
    raise ValueError("flex element body weight sum is below mjMINVAL")
  inverse_distance /= total
  return _flex_vertices_body_weights(
      model, flex_id, global_vertices, inverse_distance)


def _candidate_filter_modes(model, descriptor):
  """Lower the distinct pinned SDF candidate routes for the native selector.

  The no-flex-BVH route calls ``mjc_FlexSDF``'s direct element loop, which
  emits unique penetrating Halton witnesses in traversal order and stops at
  50 contacts. The BVH callback instead emits one deepest witness per element
  before the driver-level contact filter. Keeping these routes distinct is
  necessary for candidate identity and manifold cardinality.
  """
  sdf_type = int(mujoco.mjtGeom.mjGEOM_SDF)
  return _frozen([
      (_FILTER_SDF_DIRECT if int(model.flex_bvhadr[int(flex)]) < 0 else 2)
      if int(kind) == _KIND_GEOM_ELEMENT and int(geom) >= 0
      and int(model.geom_type[int(geom)]) == sdf_type else 0
      for kind, geom, flex in zip(descriptor.kind, descriptor.geom,
                                  descriptor.flex1)], np.int32)


def _candidate_narrowphase_supported(model, descriptor):
  """Return whether every fixed candidate uses an admitted exact primitive.

  Admitted routes currently include flex vertex vs plane, direct (order-0)
  1D/2D flex elements vs sphere or capsule, interpolated 3D flex elements
  vs sphere, direct two-node capsule/capsule feature pairs, direct 1D
  internal vertex/edge capsule contacts, direct 2D internal vertex/triangle
  contacts, direct tetrahedral opposite-vertex/face candidates, and direct or
  Q1/Q2 interpolated flex-element pairs with one common source-order GJK/EPA
  witness per element pair. The two-edge route remains on the pinned raw
  capsule manifold path; unsupported geometry families remain closed.
  """
  sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
  capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  for slot, kind_value in enumerate(descriptor.kind):
    kind = int(kind_value)
    if kind == _KIND_PLANE_VERTEX:
      continue
    if kind == _KIND_GEOM_ELEMENT:
      flex = int(descriptor.flex1[slot])
      geom = int(descriptor.geom[slot])
      if 0 <= flex < int(model.nflex) and 0 <= geom < int(model.ngeom):
        dim = int(model.flex_dim[flex])
        interp = abs(int(model.flex_interp[flex]))
        geom_type = int(model.geom_type[geom])
        direct_simplex = (interp == 0 and dim in (1, 2)
                          and geom_type in (sphere, capsule))
        interpolated_sphere = (interp in (1, 2) and dim == 3
                               and geom_type == sphere)
        if direct_simplex or interpolated_sphere:
          continue
    if kind == _KIND_INTERNAL_VERTEX_ELEMENT:
      # The current internal detector's capsule branch implements the pinned
      # vertex-versus-edge route for direct 1D flexes.  The tetrahedral
      # opposite-vertex/face branch and triangle/volume internal routes stay
      # guarded until their own production force and trajectory gates pass.
      flex = int(descriptor.flex1[slot])
      nodes = np.asarray(descriptor.nodes2[slot], dtype=np.int32)
      if (0 <= flex < int(model.nflex)
          and int(model.flex_dim[flex]) == 1
          and int(model.flex_interp[flex]) == 0
          and bool(model.flex_internal[flex])
          and int(descriptor.vert1[slot]) >= 0
          and int(descriptor.elem2[slot]) >= 0
          and int(descriptor.vert2[slot]) < 0
          and np.count_nonzero(nodes >= 0) == 2):
        continue
      # A direct membrane internal pair has one point node on side 1 and a
      # triangle on side 2. Its detector uses the same pinned sphere/triangle
      # primitive and barycentric row weights as other direct triangles.
      point = np.asarray(descriptor.nodes1[slot], dtype=np.int32)
      triangle = np.asarray(descriptor.nodes2[slot], dtype=np.int32)
      if (0 <= flex < int(model.nflex)
          and int(model.flex_dim[flex]) == 2
          and int(model.flex_interp[flex]) == 0
          and bool(model.flex_internal[flex])
          and int(descriptor.vert1[slot]) >= 0
          and int(descriptor.elem2[slot]) >= 0
          and int(descriptor.elem1[slot]) < 0
          and int(descriptor.vert2[slot]) < 0
          and np.count_nonzero(point >= 0) == 1
          and np.count_nonzero(triangle >= 0) == 3):
        continue
      # For a direct tetrahedral flex, the second internal pass emits the
      # opposite vertex on side 2 and the remaining three tetra vertices as
      # side-1's face. The production detector computes that pinned face
      # plane contact and its barycentric weights directly.
      face = np.asarray(descriptor.nodes1[slot], dtype=np.int32)
      point = np.asarray(descriptor.nodes2[slot], dtype=np.int32)
      if (0 <= flex < int(model.nflex)
          and int(model.flex_dim[flex]) == 3
          and int(model.flex_interp[flex]) == 0
          and bool(model.flex_internal[flex])
          and int(descriptor.vert1[slot]) < 0
          and int(descriptor.elem1[slot]) >= 0
          and int(descriptor.elem2[slot]) < 0
          and int(descriptor.vert2[slot]) >= 0
          and np.count_nonzero(face >= 0) == 3
          and np.count_nonzero(point >= 0) == 1):
        continue
    if kind == _KIND_ELEMENT_PAIR:
      flex1, flex2 = int(descriptor.flex1[slot]), int(descriptor.flex2[slot])
      if (0 <= flex1 < int(model.nflex) and 0 <= flex2 < int(model.nflex)
          and int(model.flex_dim[flex1]) == 1
          and int(model.flex_dim[flex2]) == 1
          and abs(int(model.flex_interp[flex1])) == 0
          and abs(int(model.flex_interp[flex2])) == 0
          and int(np.count_nonzero(np.asarray(descriptor.nodes1[slot]) >= 0)) == 2
          and int(np.count_nonzero(np.asarray(descriptor.nodes2[slot]) >= 0)) == 2):
        continue
      # `mjc_ConvexElem` reads all current compiled element vertices from
      # d->flexvert_xpos. This includes Q1/Q2 interpolation; the row producer
      # maps those same vertices through the compiled interpolation weights.
      if 0 <= flex1 < int(model.nflex) and 0 <= flex2 < int(model.nflex):
        dim1, dim2 = int(model.flex_dim[flex1]), int(model.flex_dim[flex2])
        interp1, interp2 = (abs(int(model.flex_interp[flex1])),
                            abs(int(model.flex_interp[flex2])))
        count1 = int(np.count_nonzero(
            np.asarray(descriptor.nodes1[slot], dtype=np.int32) >= 0))
        count2 = int(np.count_nonzero(
            np.asarray(descriptor.nodes2[slot], dtype=np.int32) >= 0))
        if (dim1 in (1, 2, 3) and dim2 in (1, 2, 3)
            and max(dim1, dim2) >= 2
            and interp1 in (0, 1, 2) and interp2 in (0, 1, 2)
            and count1 == dim1 + 1 and count2 == dim2 + 1):
          continue
    return False
  return True


def _frozen(value, dtype, shape=None):
  array = np.asarray(value, dtype=dtype, order="C")
  if shape is not None:
    array = array.reshape(shape)
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _float32_expansion(value):
  """Split binary64 model operands into three float32 summands."""
  source = np.asarray(value, dtype=np.float64)
  high = source.astype(np.float32)
  residual = source - high.astype(np.float64)
  middle = residual.astype(np.float32)
  low = (residual - middle.astype(np.float64)).astype(np.float32)
  return high, middle, low


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
  override = bool(int(model.opt.enableflags) & _ENABLE_OVERRIDE)
  if override:
    ref = np.asarray(model.opt.o_solref, np.float64).copy()
    imp = np.asarray(model.opt.o_solimp, np.float64).copy()
    friction5 = np.asarray(model.opt.o_friction, dtype=np.float32).copy()
  else:
    friction5 = np.asarray(
        [friction[0], friction[0], friction[1], friction[2], friction[2]],
        dtype=np.float32)
  # mj_assignFriction clamps each material component after pair mixing or
  # global override assignment.
  friction5 = np.maximum(friction5, 1e-5).astype(np.float32)
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


def _flex_bvh_element_order(model, flex_id, elements):
  """Return compiled leaf callback order from pinned right-first traversal.

  A 3D flex BVH can omit elements above its compiled active-layer frontier.
  Runtime ``flex_activelayers`` edits do not rebuild those leaves, so the tree
  itself is the authority for this collision route.
  """
  root = int(model.flex_bvhadr[flex_id])
  if root < 0:
    return tuple(range(elements))
  node_count = int(model.flex_bvhnum[flex_id])
  node_ids = np.asarray(model.bvh_nodeid, dtype=np.int32)
  children = np.asarray(model.bvh_child, dtype=np.int32).reshape(-1, 2)
  order = []
  # MuJoCo stores each BVH's child indices relative to its root and passes
  # model arrays already sliced at the root to traverseBVH.
  stack = [0]
  while stack:
    node = stack.pop()
    if node < 0 or node >= node_count or root + node >= node_ids.size:
      raise ValueError(f"flex {flex_id} BVH child index {node} is out of range")
    global_node = root + node
    element = int(node_ids[global_node])
    if element >= 0:
      if element >= elements:
        raise ValueError(f"flex {flex_id} BVH leaf element {element} is out of range")
      order.append(element)
      continue
    # traverseBVH pushes children in 0,1 order onto a LIFO stack, so the
    # compiled child 1 subtree receives callbacks before child 0.
    child0, child1 = map(int, children[global_node])
    for child in (child0, child1):
      if child < 0:
        continue
      if len(stack) >= 64:
        raise ValueError(f"flex {flex_id} BVH traversal exceeds pinned stack depth")
      stack.append(child)
  if len(order) != len(set(order)):
    raise ValueError(f"flex {flex_id} BVH contains duplicate element leaves")
  return tuple(order)


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


def _flex_mesh_hull(model, descriptor):
  """Build convex, heightfield and plugin-free SDF payloads for flex pairs."""
  geoms = sorted({int(g) for g, kind in zip(descriptor.geom, descriptor.kind)
                  if g >= 0 and int(kind) == _KIND_GEOM_ELEMENT
                  and int(model.geom_type[int(g)]) in (
                      int(mujoco.mjtGeom.mjGEOM_MESH),
                      int(mujoco.mjtGeom.mjGEOM_HFIELD),
                      int(mujoco.mjtGeom.mjGEOM_SDF))})
  mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
  hfield_type = int(mujoco.mjtGeom.mjGEOM_HFIELD)
  sdf_type = int(mujoco.mjtGeom.mjGEOM_SDF)
  hull_data = []
  info = np.full((int(model.ngeom), 9), -1, np.int32)

  def append_hull(values):
    """Append one packed block while keeping every int32 offset representable."""
    block = np.asarray(values).reshape(-1)
    start = len(hull_data)
    if start > _INT32_MAX or block.size > _INT32_MAX - start:
      raise ValueError("flex geometry payload exceeds signed int32 offsets")
    hull_data.extend(block.tolist())
    return start

  plugin_sdf = None
  plugin_geom_ids = {geom for geom in geoms
                     if int(model.geom_plugin[geom]) >= 0}
  if plugin_geom_ids:
    from mujoco_metal.bundled_plugins import lower_bundled_plugins
    plugin_sdf = lower_bundled_plugins(model)
  plugin_geom_rows = {
      int(geom): row for row, geom in enumerate(
          () if plugin_sdf is None else plugin_sdf.sdf_geom_id.tolist())}
  for geom in geoms:
    gtype = int(model.geom_type[geom])
    if gtype == mesh_type:
      # Mesh support addresses vertices in units of float3; keep each mesh
      # block aligned even when scalar HField payloads precede it.
      while len(hull_data) % 3:
        append_hull([0.0])
      mesh = int(model.geom_dataid[geom])
      count = int(model.mesh_vertnum[mesh])
      if count < 1 or count > 64:
        raise ValueError(f"flex contact mesh geom {geom} has {count} hull vertices; "
                         "the pinned convex support kernel accepts 1..64")
      adr = int(model.mesh_vertadr[mesh])
      local = np.asarray(model.mesh_vert[adr:adr + count], np.float32).reshape(-1, 3)
      info[geom, 0] = len(hull_data) // 3
      info[geom, 1] = count
      append_hull(local.reshape(-1))
    elif gtype == hfield_type:
      hfield = int(model.geom_dataid[geom])
      nrow, ncol = int(model.hfield_nrow[hfield]), int(model.hfield_ncol[hfield])
      adr = int(model.hfield_adr[hfield])
      samples = np.asarray(
          model.hfield_data[adr:adr + nrow*ncol], np.float32).reshape(-1)
      # HField narrowphase indexes the packed data as scalar floats, while
      # mesh support above indexes float3 vertices.
      info[geom, 5] = len(hull_data)
      info[geom, 6] = nrow
      info[geom, 7] = ncol
      append_hull(samples)
      info[geom, 8] = len(hull_data)
      size = np.asarray(model.hfield_size[hfield], np.float32).reshape(4)
      append_hull(size)
    elif gtype == sdf_type:
      if int(model.geom_plugin[geom]) != -1:
        row = plugin_geom_rows.get(geom)
        if row is None:
          raise ValueError(f"flex SDF geom {geom} has no bundled plugin row")
        instance = int(plugin_sdf.sdf_geom_instance[row])
        info[geom, 0] = len(hull_data)
        info[geom, 1] = -2  # bundled plugin-SDF record
        info[geom, 2] = int(model.opt.sdf_iterations)
        info[geom, 4] = int(plugin_sdf.instance_kind[instance])
        append_hull(plugin_sdf.plugin_attributes[instance])
        continue
      # Match the SDF payload ABI used by coupled_constraints: child[8N],
      # local AABB[6N], coefficient[8N], then one geom-local AABB block.
      mesh = int(model.geom_dataid[geom])
      if mesh < 0:
        raise ValueError(f"flex SDF geom {geom} has no mesh octree asset")
      oct_adr = int(model.mesh_octadr[mesh])
      oct_num = int(model.mesh_octnum[mesh])
      if oct_adr < 0 or oct_num <= 0:
        raise ValueError(f"flex SDF geom {geom} mesh {mesh} has no compiled "
                         "octree")
      child = np.asarray(model.oct_child[oct_adr:oct_adr + oct_num],
                         np.int32).reshape(-1)
      aabb_source = np.asarray(
          model.oct_aabb[oct_adr:oct_adr + oct_num], np.float64).reshape(-1)
      coeff_source = np.asarray(
          model.oct_coeff[oct_adr:oct_adr + oct_num], np.float64).reshape(-1)
      aabb, aabb_low, aabb_tail = _split_source_mjt_num(aabb_source)
      coeff, coeff_low, coeff_tail = _split_source_mjt_num(coeff_source)
      if (child.size != 8 * oct_num or aabb.size != 6 * oct_num
          or coeff.size != 8 * oct_num):
        raise ValueError(f"flex SDF mesh {mesh} octree layout mismatch")
      # SDF child indices are stored as floats because the shared SDF kernels
      # use one packed float buffer. Validate exact integer representation.
      child_f = child.astype(np.float32)
      if not np.array_equal(child_f.astype(np.int32), child):
        raise ValueError(f"flex SDF mesh {mesh} child indices exceed float32 "
                         "integer precision")
      info[geom, 0] = len(hull_data)
      info[geom, 1] = oct_num
      info[geom, 2] = int(model.opt.sdf_iterations)
      append_hull(child_f)
      append_hull(aabb)
      append_hull(coeff)
      # Keep the existing single float payload and high-word offsets stable.
      # The next four blocks carry source mjtNum residuals for AABBs and
      # trilinear coefficients; geom_hull_info columns 4..7 point to them.
      info[geom, 4] = append_hull(aabb_low)
      info[geom, 5] = append_hull(aabb_tail)
      info[geom, 6] = append_hull(coeff_low)
      info[geom, 7] = append_hull(coeff_tail)
      # Reserve a fixed stride for all geom AABBs after the octree payload;
      # write their start once, after the last SDF asset has been appended.
      info[geom, 3] = -2
  sdf_geoms = [g for g in geoms if int(model.geom_type[g]) == sdf_type]
  if sdf_geoms:
    aabb_base = len(hull_data)
    geom_aabb = np.asarray(model.geom_aabb, np.float32).reshape(-1)
    if geom_aabb.size != 6 * int(model.ngeom):
      raise ValueError("compiled flex-SDF geom AABB layout mismatch")
    append_hull(geom_aabb)
    for geom in sdf_geoms:
      info[geom, 3] = aabb_base
    sdf_iterations = int(model.opt.sdf_iterations)
    if sdf_iterations <= 0:
      raise ValueError("opt.sdf_iterations must be positive with flex SDF pairs")
  flat = np.asarray(hull_data, dtype=np.float32).reshape(-1)
  if flat.size == 0:
    flat = np.zeros(1, np.float32)
  return _frozen(flat, np.float32), _frozen(info.reshape(-1), np.int32)


def _split_source_mjt_num(values):
  """Represent compiled binary64 SDF operands as float32 high/low/tail words."""
  source = np.asarray(values, dtype=np.float64)
  if not np.all(np.isfinite(source)):
    raise ValueError("compiled SDF octree values must be finite")
  high = source.astype(np.float32)
  remainder = source - high.astype(np.float64)
  low = remainder.astype(np.float32)
  tail = (remainder - low.astype(np.float64)).astype(np.float32)
  if (not np.all(np.isfinite(high)) or not np.all(np.isfinite(low))
      or not np.all(np.isfinite(tail))):
    raise ValueError("compiled SDF octree values exceed float32 word range")
  return high.reshape(-1), low.reshape(-1), tail.reshape(-1)


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
  flex_radius_hi: np.ndarray
  flex_radius_mid: np.ndarray
  flex_radius_low: np.ndarray
  geom_size_hi: np.ndarray
  geom_size_mid: np.ndarray
  geom_size_low: np.ndarray
  margin: np.ndarray
  margin_mid: np.ndarray
  margin_low: np.ndarray
  gap: np.ndarray
  gap_mid: np.ndarray
  gap_low: np.ndarray
  feature_capacity: int
  row_capacity: int
  nv: int
  global_enabled: bool
  link_tree_pairs: np.ndarray
  candidate_link_ids: np.ndarray
  max_links_per_candidate: int
  link_capacity: int
  filter_group: np.ndarray
  filter_group_count: int
  filterable: np.ndarray
  contact_ordinal: np.ndarray

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
      # The body:flex midphase traverses the compiled flex BVH, whose leaves
      # contain only the currently active elements.  The all-to-all fallback
      # in engine_collision_driver.c instead loops over every element ID and
      # lets mj_collideGeomElem's own filters reject candidates.  Preserve
      # that route distinction here: a CCD-overlapping inactive element must
      # not become a native contact slot when the CPU never dispatches it.
      use_midphase = (
          not (int(model.opt.disableflags) & _DISABLE_MIDPHASE)
          and int(model.body_bvhadr[body]) >= 0
          and int(model.flex_bvhadr[f]) >= 0)
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
        if (gtype == int(mujoco.mjtGeom.mjGEOM_SDF)
            and int(model.flex_bvhadr[f]) >= 0):
          element_order = _flex_bvh_element_order(model, f, en)
        elif use_midphase:
          element_order = _flex_bvh_element_order(model, f, en)
        else:
          # Without a flex BVH, mjc_FlexSDF iterates elements directly in
          # model order; other narrowphases use the same fixed model ordering.
          element_order = range(en)
        for e in element_order:
          vertices = elems[e]
          if geom_body >= 0 and np.any(vert_body[vertices] == geom_body):
            continue
          nodes = np.full(4, -1, np.int32)
          nodes[:nper] = vertices
          # Pinned raw capsule dispatch emits up to four capsule-capsule
          # witnesses or five capsule-triangle witnesses. Reserve each raw
          # ordinal so the later group cap sees the complete contact list.
          ncontact = 1
          ordinals = None
          if gtype == int(mujoco.mjtGeom.mjGEOM_SDF):
            if int(model.flex_bvhadr[f]) >= 0:
              # mjc_FlexSDF's BVH callback keeps one deepest witness per
              # intersecting element before its 50-contact FPS selection.
              ordinals = (-1,)
            else:
              # Its no-BVH path keeps every penetrating Halton seed, subject
              # to the pinned per-pair maximum.
              ncontact = min(max(1, int(model.opt.sdf_initpoints)),
                             _MAX_FLEX_CONTACTS)
          elif gtype == int(mujoco.mjtGeom.mjGEOM_HFIELD):
            # mjc_HFieldElem can return mjMAXCONPAIR independent witnesses
            # for a single flex element against the sampled terrain.
            ncontact = 50
          elif nper == 2:
            if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
              ncontact = 4
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
              ncontact = 2
          elif (nper == 3
                and gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            ncontact = 5
          elif (nper == 3
                and gtype == int(mujoco.mjtGeom.mjGEOM_BOX)):
            ncontact = 11
          if ordinals is None:
            ordinals = range(ncontact)
          for ordinal in ordinals:
            rows.append((_KIND_GEOM_ELEMENT, f, e, -1, -1, -1, -1, g,
                         nodes, np.full(4, -1, np.int32), condim, span,
                         friction, solref, solimp, margin, gap, -1, -1,
                         ordinal))

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
          point = np.full(4, -1, np.int32)
          simplex = np.full(4, -1, np.int32)
          # mj_collideElemVert passes the point as feature 1 and the whole
          # capsule/triangle/tetrahedron as feature 2. Keep that same ordering
          # in the fixed-slot descriptor: the weights then address nodes1 and
          # nodes2 without interpreting a vertex ID as an element vertex.
          point[0] = int(vertex)
          simplex[:nper] = elems[int(element)]
          rows.append((_KIND_INTERNAL_VERTEX_ELEMENT, f, -1, int(vertex),
                       f, int(element), -1, -1,
                       point, simplex, condim, span,
                       friction, solref, solimp, 0.0, 0.0, -1, -1))
      if int(model.flex_dim[f]) == 3:
        # Match mj_collideFlexInternal's four oriented face calls exactly.
        # The face winding determines planeVertex's signed distance, so an
        # unordered "all vertices except opposite" face can admit contacts
        # from the wrong side of the tetrahedron.
        tetra_faces = (
            (3, (0, 1, 2)),
            (1, (0, 2, 3)),
            (2, (0, 3, 1)),
            (0, (1, 3, 2)),
        )
        for element in range(en):
          vertices = elems[element]
          for opposite, face_local in tetra_faces:
            vertex = vertices[opposite]
            face = np.full(4, -1, np.int32)
            face[:3] = vertices[np.asarray(face_local, dtype=np.int64)]
            point = np.full(4, -1, np.int32)
            point[0] = int(vertex)
            # The tetrahedral special pass records the face as side 1 and the
            # opposite vertex as side 2, matching the pinned contact identity.
            rows.append((_KIND_INTERNAL_VERTEX_ELEMENT, f, element, -1,
                         f, -1, int(vertex), -1, face, point, 1, 1, friction,
                         solref, solimp, 0.0, 0.0, opposite, -1))

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
          ncontact = (_CAPSULE_CAPSULE_MAX_CONTACTS if nper == 2 else 1)
          for ordinal in range(ncontact):
            rows.append((_KIND_ELEMENT_PAIR, f, e1, -1,
                         f, e2, -1, -1, nodes1, nodes2,
                         condim, span, friction, solref, solimp,
                         0.0, 0.0, -1, -1, ordinal))

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
          ncontact = (_CAPSULE_CAPSULE_MAX_CONTACTS
                      if n1 == 2 and n2 == 2 else 1)
          for ordinal in range(ncontact):
            rows.append((_KIND_ELEMENT_PAIR, f1, e1, -1, f2, e2, -1, -1,
                         nodes1, nodes2, condim, span, friction, solref, solimp,
                         _assigned_margin(
                             model, float(model.flex_margin[f1]
                                          + model.flex_margin[f2])),
                         float(model.flex_gap[f1] + model.flex_gap[f2]),
                         -1, -1, ordinal))

  count = len(rows)
  int32_max = np.iinfo(np.int32).max
  if count > int32_max or int(model.nv) > int32_max:
    raise OverflowError("flex contact slots and nv must fit signed 32-bit indexing")
  kinds = np.asarray([r[0] for r in rows], np.int32)
  row_span64 = np.asarray([r[11] for r in rows], np.int64)
  if np.any(row_span64 <= 0):
    raise ValueError("every flex contact candidate must own a positive row span")
  row_capacity = int(row_span64.sum(dtype=np.int64))
  if row_capacity > int32_max:
    raise OverflowError("flex contact row capacity exceeds signed 32-bit indexing")
  row_start64 = np.cumsum(row_span64, dtype=np.int64) - row_span64
  row_span = row_span64.astype(np.int32)
  row_start = row_start64.astype(np.int32)
  def col(index, dtype=np.int32):
    return _frozen([r[index] if index < len(r) else -1 for r in rows], dtype)
  # Wake-link expansion is static and duplicate-free. Every emitted edge is
  # between distinct moving kinematic trees that can contribute to the
  # candidate's barycentric relative Jacobian.
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  geom_body = np.asarray(model.geom_bodyid, dtype=np.int32)

  def vertex_trees(flex_id, vertex_ids):
    """Static union of every rigid tree in pinned vertex body weights."""
    trees = set()
    for vertex in vertex_ids:
      vertex = int(vertex)
      if vertex < 0:
        continue
      for body in _flex_vertex_body_weights(model, flex_id, vertex):
        body = int(body)
        if 0 <= body < body_tree.size:
          tree = int(body_tree[body])
          if tree >= 0:
            trees.add(tree)
    return trees

  def element_trees(flex_id, vertex_ids):
    """Conservative tree support over the whole interpolated element domain.

    The contact point can lie between compiled flex vertices. Q1/Q2 basis
    nodes (and shell TFI boundary nodes) can contribute there even when none
    contributes at the simplex corners, so corner-weight unions are not a
    sound wake-link superset. Direct flexes have no interpolation nodes and
    retain their exact vertex-body support.
    """
    if int(model.flex_interp[flex_id]) == 0:
      return vertex_trees(flex_id, vertex_ids)
    node_adr = int(model.flex_nodeadr[flex_id])
    node_num = int(model.flex_nodenum[flex_id])
    node_bodies = np.asarray(model.flex_nodebodyid, dtype=np.int64)[
        node_adr:node_adr + node_num]
    return {
        int(body_tree[body]) for body in node_bodies
        if 0 <= int(body) < body_tree.size and int(body_tree[body]) >= 0
    }

  link_sets = []
  all_links = set()
  for row in rows:
    _, f1, e1, v1, f2, e2, v2, geom, n1, n2, *_ = row
    if v1 >= 0 and v2 < 0 and geom >= 0:
      lhs = vertex_trees(f1, (v1,))
      rhs = {int(body_tree[geom_body[geom]])}
    elif v1 < 0 and v2 >= 0 and geom >= 0:
      lhs = vertex_trees(f2, (v2,))
      rhs = {int(body_tree[geom_body[geom]])}
    else:
      ids1 = [int(v) for v in n1 if int(v) >= 0]
      ids2 = [int(v) for v in n2 if int(v) >= 0]
      if v1 >= 0:
        ids1 = [v1]
      if v2 >= 0:
        ids2 = [v2]
      lhs = (element_trees(f1, ids1) if e1 >= 0
             else vertex_trees(f1, ids1))
      rhs = ((element_trees(f2, ids2) if e2 >= 0
              else vertex_trees(f2, ids2)) if f2 >= 0 else set())
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
  group_keys = []
  for row in rows:
    kind, flex1, _elem1, _vert1, flex2, _elem2, _vert2, geom, *_ = row
    if kind in (_KIND_PLANE_VERTEX, _KIND_GEOM_ELEMENT):
      body = int(model.geom_bodyid[geom])
      use_midphase = (
          not (int(model.opt.disableflags) & _DISABLE_MIDPHASE)
          and int(model.body_bvhadr[body]) >= 0
          and int(model.flex_bvhadr[flex1]) >= 0)
      owner = body if use_midphase else int(geom)
      group = (0, int(use_midphase), owner, flex1)
    elif kind == _KIND_INTERNAL_VERTEX_ELEMENT:
      group = (1, flex1)
    elif flex1 == flex2:
      group = (2, flex1)
    else:
      group = (3, flex1, flex2)
    group_keys.append(group)
  group_index = {key: index for index, key in enumerate(sorted(set(group_keys)))}
  filter_group = np.asarray([group_index[key] for key in group_keys], np.int32)
  # The pinned driver calls filterFlexContacts for every flex-producing
  # route, including the plane special case and internal vertex/element pass.
  # Those routes therefore participate in the same per-driver-call 50-contact
  # cap/FPS selection as ordinary geom, self and cross-flex contacts.
  filterable = np.ones(count, dtype=np.int32)
  radius_hi, radius_mid, radius_low = _float32_expansion(model.flex_radius)
  geom_hi, geom_mid, geom_low = _float32_expansion(model.geom_size)
  margin_values = np.asarray([r[15] for r in rows], dtype=np.float64)
  gap_values = np.asarray([r[16] for r in rows], dtype=np.float64)
  margin_hi, margin_mid, margin_low = _float32_expansion(margin_values)
  gap_hi, gap_mid, gap_low = _float32_expansion(gap_values)
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
      flex_radius_hi=_frozen(radius_hi, np.float32),
      flex_radius_mid=_frozen(radius_mid, np.float32),
      flex_radius_low=_frozen(radius_low, np.float32),
      geom_size_hi=_frozen(geom_hi, np.float32,
                           (int(model.ngeom), 3)),
      geom_size_mid=_frozen(geom_mid, np.float32,
                           (int(model.ngeom), 3)),
      geom_size_low=_frozen(geom_low, np.float32, (int(model.ngeom), 3)),
      margin=_frozen(margin_hi, np.float32),
      margin_mid=_frozen(margin_mid, np.float32),
      margin_low=_frozen(margin_low, np.float32),
      gap=_frozen(gap_hi, np.float32),
      gap_mid=_frozen(gap_mid, np.float32),
      gap_low=_frozen(gap_low, np.float32),
      feature_capacity=count, row_capacity=row_capacity, nv=int(model.nv),
      global_enabled=not bool(int(model.opt.disableflags)
                              & (_DISABLE_CONTACT | _DISABLE_CONSTRAINT)),
      link_tree_pairs=_frozen(link_pairs, np.int32).reshape(-1, 2),
      candidate_link_ids=_frozen(candidate_link_ids, np.int32),
      max_links_per_candidate=max_links,
      link_capacity=len(link_pairs),
      filter_group=_frozen(filter_group, np.int32),
      filter_group_count=len(group_index),
      filterable=_frozen(filterable, np.int32), contact_ordinal=col(19))


class FlexContactProgram:
  """Fixed-slot flex contact workspace with a fail-closed narrowphase gate."""

  def __init__(self, model, batch_size=1, device="mps",
               capture_ccd_trace=False, sparse_rows=False,
               capture_gjk_trace_slot=None):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("flex contact construction requires a compiled mjModel")
    descriptor = lower_flex_contacts(model)
    batch_size = _validate_contact_program_capacity(
        model, descriptor, batch_size)
    self.model = model
    self.descriptor = descriptor
    self.batch_size = batch_size
    self.capture_gjk_trace_slot = None
    if capture_gjk_trace_slot is not None:
      trace_env, trace_slot = _flex_trace_slot(capture_gjk_trace_slot)
      if not (0 <= trace_env < batch_size
              and 0 <= trace_slot < descriptor.slot_count):
        raise ValueError("GJK trace target is outside the compiled workspace")
      trace_geom = int(descriptor.geom[trace_slot])
      if (int(descriptor.kind[trace_slot]) != _KIND_GEOM_ELEMENT
          or trace_geom < 0
          or int(model.geom_type[trace_geom])
             != int(mujoco.mjtGeom.mjGEOM_SPHERE)
          or np.count_nonzero(np.asarray(descriptor.nodes1[trace_slot]) >= 0)
             != 4):
        raise ValueError("GJK trace target must be a sphere-flex candidate")
      self.capture_gjk_trace_slot = (trace_env, trace_slot)
    self.capture_gjk_trace = self.capture_gjk_trace_slot is not None
    self.capture_ccd_trace = bool(capture_ccd_trace or self.capture_gjk_trace)
    (self._ccd_trace_iterations, self._ccd_trace_max_faces,
     self._ccd_trace_iter_stride, self._ccd_trace_size) = (
         _flex_ccd_trace_capacity(model.opt.ccd_iterations,
                                  batch_size, descriptor.slot_count)
         if self.capture_ccd_trace else (0, 0, 0, 1))
    (self._gjk_trace_iterations, self._gjk_trace_stride,
     self._gjk_trace_size) = (
         _flex_gjk_trace_capacity(model.opt.ccd_iterations)
         if self.capture_gjk_trace else (0, 0, 0))
    (self._ccd_trace_words, self._combined_trace_words) = (
        _flex_combined_trace_capacity(
            batch_size, descriptor.slot_count,
            self._ccd_trace_size if self.capture_ccd_trace else 0,
            self._gjk_trace_size)
        if self.capture_ccd_trace else (0, 1))
    import torch

    self.device = torch.device(device)
    self.sparse_rows = bool(sparse_rows)
    self._compiled_disableflags = int(model.opt.disableflags)
    self._native_ccd_enabled = not bool(
        self._compiled_disableflags & _DISABLE_NATIVECCD)
    if self.capture_gjk_trace and not self._native_ccd_enabled:
      raise ValueError("GJK trace requires the native CCD path")
    (self._epa_vertex_capacity, self._epa_face_capacity,
     self._epa_horizon_capacity, self._epa_stack_capacity) = (
         _flex_ccd_workspace_capacity(model.opt.ccd_iterations,
                                      batch_size, descriptor.slot_count))
    self._torch = torch
    d = self.descriptor
    supported_geom_types = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_HFIELD),
        int(mujoco.mjtGeom.mjGEOM_MESH),
    }
    geom_candidates_supported = all(
        int(kind) != _KIND_GEOM_ELEMENT
        or int(model.geom_type[int(geom)]) in supported_geom_types
        for kind, geom in zip(d.kind, d.geom))
    diag_components = np.zeros((d.slot_count, 2), dtype=np.float32)
    diag_supported = True
    body_invweight0 = np.asarray(model.body_invweight0, dtype=np.float64).reshape(
        int(model.nbody), 2)
    for slot, kind in enumerate(d.kind):
      if int(kind) != _KIND_PLANE_VERTEX:
        diag_supported = False
        continue
      geom, vertex = int(d.geom[slot]), int(d.vert1[slot])
      if vertex < 0:
        vertex = int(d.vert2[slot])
      flex_id = int(d.flex1[slot] if int(d.flex1[slot]) >= 0 else d.flex2[slot])
      if vertex < 0 or flex_id < 0:
        diag_supported = False
        continue
      geom_body = int(model.geom_bodyid[geom])
      try:
        body_weights = _flex_vertex_body_weights(model, flex_id, vertex)
      except NotImplementedError:
        diag_supported = False
        continue
      diag_components[slot, 0] = body_invweight0[geom_body, 0]
      diag_components[slot, 1] = body_invweight0[geom_body, 1]
      for body, weight in body_weights.items():
        diag_components[slot, 0] += body_invweight0[body, 0] * weight
        diag_components[slot, 1] += body_invweight0[body, 1] * weight
    self._diag_approx_supported = diag_supported
    self._diag_dynamic_supported = all(
        abs(int(model.flex_interp[int(flex_id)])) <= 2
        for flex_id in np.concatenate((d.flex1, d.flex2)) if int(flex_id) >= 0)
    self._narrowphase_admitted = (
        not d.global_enabled
        or (_candidate_narrowphase_supported(model, d)
            and self._diag_dynamic_supported and geom_candidates_supported))
    self._convex_source_supported = geom_candidates_supported
    self._has_native_ccd_candidates = any(
        int(kind) == _KIND_GEOM_ELEMENT
        and int(geom) >= 0
        and int(model.geom_type[int(geom)]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        and np.count_nonzero(np.asarray(d.nodes1[slot]) >= 0) == 4
        for slot, (kind, geom) in enumerate(zip(d.kind, d.geom)))
    self._has_native_flex_pair_candidates = any(
        int(kind) == _KIND_ELEMENT_PAIR
        and np.count_nonzero(np.asarray(d.nodes1[slot]) >= 0) >= 2
        and np.count_nonzero(np.asarray(d.nodes2[slot]) >= 0) >= 2
        and (np.count_nonzero(np.asarray(d.nodes1[slot]) >= 0) >= 3
             or np.count_nonzero(np.asarray(d.nodes2[slot]) >= 0) >= 3)
        for slot, kind in enumerate(d.kind))
    def tensor(value, dtype=None):
      return torch.as_tensor(np.array(value, copy=True), dtype=dtype,
                             device=self.device)
    def tensor_nonempty(value, dtype=None, fallback=0):
      array = np.array(value, copy=True)
      if array.size == 0:
        array = np.asarray([fallback], dtype=array.dtype if array.dtype != object
                           else np.int32)
      return torch.as_tensor(array, dtype=dtype, device=self.device).contiguous()
    self._kind = tensor(d.kind, torch.int32)
    self._slot_id = tensor(np.arange(d.slot_count, dtype=np.int32), torch.int32)
    self._flex1 = tensor(d.flex1, torch.int32)
    self._elem1 = tensor(d.elem1, torch.int32)
    self._vert1 = tensor(d.vert1, torch.int32)
    self._flex2 = tensor(d.flex2, torch.int32)
    self._elem2 = tensor(d.elem2, torch.int32)
    self._vert2 = tensor(d.vert2, torch.int32)
    self._geom = tensor(d.geom, torch.int32)
    self._nodes1 = tensor(d.nodes1, torch.int32)
    self._nodes2 = tensor(d.nodes2, torch.int32)
    # Collision kernels index these arrays by compiled flex id, not candidate
    # slot. Preserve the pinned double model operand as a float32 high/low pair
    # for the native CCD path; the ordinary path consumes the high word.
    self._radius1 = tensor(d.flex_radius_hi, torch.float32)
    self._radius1_midlow = tensor(
        np.stack((d.flex_radius_mid, d.flex_radius_low), axis=1), torch.float32)
    self._radius2 = tensor(d.radius2, torch.float32)
    self._margin_gap = tensor(np.stack((d.margin, d.gap), axis=1), torch.float32)
    self._margin_gap_midlow = tensor(
        np.stack((np.stack((d.margin_mid, d.margin_low), axis=1),
                  np.stack((d.gap_mid, d.gap_low), axis=1)), axis=1),
        torch.float32)
    self._margin = tensor(d.margin, torch.float32)
    self._gap = tensor(d.gap, torch.float32)
    self._geom_type = tensor(model.geom_type, torch.int32)
    self._geom_size = tensor(d.geom_size_hi, torch.float32)
    self._geom_size_midlow = tensor(
        np.stack((d.geom_size_mid, d.geom_size_low), axis=2), torch.float32)
    hull, hull_info = _flex_mesh_hull(model, d)
    self._geom_hull = tensor(hull, torch.float32)
    self._geom_hull_info = tensor(hull_info, torch.int32)
    self._row_start = tensor(d.row_start, torch.int32)
    self._row_span = tensor(d.row_span, torch.int32)
    self._condim = tensor(d.condim, torch.int32)
    self._cone = tensor(d.cone, torch.int32)
    self._friction = tensor(d.friction, torch.float32)
    self._solref = tensor(d.solref, torch.float32)
    self._solimp = tensor(d.solimp, torch.float32)
    self._feature1 = tensor(d.feature1, torch.int32)
    self._feature2 = tensor(d.feature2, torch.int32)
    self._contact_ordinal = tensor(d.contact_ordinal, torch.int32)
    detect_meta = np.stack((
        d.kind, d.flex1, d.elem1, d.vert1, d.flex2, d.elem2, d.vert2,
        d.geom, _candidate_filter_modes(model, d), d.contact_ordinal), axis=1)
    self._detect_meta = tensor(detect_meta, torch.int32)
    self._diag_components = tensor(diag_components, torch.float32)
    # Runtime element-contact mj_diagApprox depends on contact_pos and
    # flexvert_xpos.  Keep every compiled interpolation/body table immutable
    # on device so the producer never reads state back to the host.
    self._diag_geom_body = tensor(model.geom_bodyid, torch.int32)
    self._diag_flex_interp = tensor(model.flex_interp, torch.int32)
    self._diag_flex_vertadr = tensor(model.flex_vertadr, torch.int32)
    self._diag_flex_nodeadr = tensor(model.flex_nodeadr, torch.int32)
    self._diag_flex_cellnum = tensor_nonempty(
        np.asarray(model.flex_cellnum, dtype=np.int32).reshape(-1), torch.int32)
    self._diag_flex_dim = tensor(model.flex_dim, torch.int32)
    self._diag_flex_vert0 = tensor_nonempty(
        np.asarray(model.flex_vert0, dtype=np.float32).reshape(-1), torch.float32)
    self._diag_flex_vertbody = tensor_nonempty(model.flex_vertbodyid, torch.int32)
    self._diag_flex_nodebody = tensor_nonempty(model.flex_nodebodyid, torch.int32)
    self._diag_body_invweight = tensor(
        np.asarray(model.body_invweight0, dtype=np.float32).reshape(-1), torch.float32)
    self._diag_nflexnode = int(model.nflexnode)
    self._diag_current_flexvert_xpos = None
    self._zero_flexvert_xpos_low = torch.zeros(
        (self.batch_size, int(self.model.nflexvert), 3),
        dtype=torch.float32, device=self.device)
    self._zero_flexvert_xpos_tail = torch.zeros_like(
        self._zero_flexvert_xpos_low)
    # CCD entry points retain their fixed buffer ABI by passing low+tail as
    # two contiguous planes in one pointer. The common-pair path uses a
    # separate high+low+tail per-world slab.
    vertex_words = self.batch_size * int(self.model.nflexvert) * 3
    geom_words = self.batch_size * int(self.model.ngeom) * 3
    self._flexvert_xpos_low_tail = torch.empty(
        (2 * vertex_words + 2 * geom_words,),
        dtype=torch.float32, device=self.device)
    self._geom_pos_low_tail = self._flexvert_xpos_low_tail[
        2 * vertex_words:].view(
            2, self.batch_size, int(self.model.ngeom), 3)
    self._zero_geom_pos_low = torch.zeros(
        (self.batch_size, int(self.model.ngeom), 3),
        dtype=torch.float32, device=self.device)
    self._zero_geom_pos_tail = torch.zeros_like(self._zero_geom_pos_low)
    # The element-pair kernel must stay within Metal's 31-buffer ABI. It reads
    # high and residual coordinates from one preallocated per-world slab.
    self._flexvert_xpos_pair = torch.empty(
        (self.batch_size, 3 * int(self.model.nflexvert), 3),
        dtype=torch.float32, device=self.device)
    self._candidate_link_ids = tensor(d.candidate_link_ids, torch.int32)
    self._filter_group = tensor(d.filter_group, torch.int32)
    self._filterable = tensor(d.filterable, torch.int32)
    self._filter_mode = tensor(_candidate_filter_modes(model, d), torch.int32)
    self._filter_dims = _flex_dims_with_world_mask(torch,
        [self.batch_size, d.slot_count, d.filter_group_count,
         _MAX_FLEX_CONTACTS], self.batch_size, self.device)
    self._diag_dims = _flex_dims_with_world_mask(torch,
        [self.batch_size, d.slot_count, d.row_capacity,
         int(model.nflexvert), self._diag_nflexnode, int(model.nflex),
         int(model.ngeom)], self.batch_size, self.device)
    self._diag_static_dims = _flex_dims_with_world_mask(
        torch, [self.batch_size, d.slot_count, d.row_capacity],
        self.batch_size, self.device)
    self._diag_approx = torch.zeros(
        (self.batch_size, d.row_capacity), dtype=torch.float32,
        device=self.device)
    self._detect_dims = torch.tensor(
        [self.batch_size, d.slot_count, int(model.nflexvert), int(model.ngeom),
         int(self._native_ccd_enabled),
         int(model.opt.sdf_initpoints), int(model.opt.ccd_iterations),
         int(self.capture_ccd_trace), self._ccd_trace_iter_stride,
         self._ccd_trace_max_faces, self._ccd_trace_size,
         self._epa_vertex_capacity, self._epa_face_capacity,
         self._epa_horizon_capacity, self._epa_stack_capacity,
         (self._epa_face_capacity + 1) & ~1,
         (self._epa_vertex_capacity * 39 + self._epa_face_capacity * 20 + 3)
             & ~3,
         ((self._epa_face_capacity + 1) & ~1)
             + self._epa_horizon_capacity * 2
             + self._epa_stack_capacity * 6,
         int(self.capture_gjk_trace),
         (self.capture_gjk_trace_slot[0]
          if self.capture_gjk_trace else -1),
         (self.capture_gjk_trace_slot[1]
          if self.capture_gjk_trace else -1),
         self._gjk_trace_iterations, self._gjk_trace_stride],
      dtype=torch.int32, device=self.device)
    if int(self._detect_dims.numel()) != _FLEX_DETECT_DIMS_FIXED_WORDS:
      raise AssertionError("flex detector dims ABI changed without mask-tail review")
    # The suffix is a borrowed per-world recovery mask. Keeping it in the
    # existing dims binding preserves the native kernel argument ABI.
    self._detect_dims = torch.cat((
        self._detect_dims,
        torch.ones((self.batch_size,), dtype=torch.int32, device=self.device)))
    self._world_mask_current = self._detect_dims[
        _FLEX_DETECT_DIMS_FIXED_WORDS:]
    self._gjk_trace_dims = torch.tensor(
        [self.batch_size, int(model.nflexvert), int(model.ngeom),
         (self.capture_gjk_trace_slot[0]
          if self.capture_gjk_trace else -1),
         (self.capture_gjk_trace_slot[1]
          if self.capture_gjk_trace else -1),
         self._gjk_trace_iterations, self._gjk_trace_stride],
        dtype=torch.int32, device=self.device)
    self._ccd_tolerance = torch.tensor(
        [float(model.opt.ccd_tolerance)], dtype=torch.float32,
        device=self.device)
    tolerance_high, tolerance_mid, tolerance_low = _float32_expansion(
        [float(model.opt.ccd_tolerance)])
    self._ccd_tolerance = torch.tensor(
        tolerance_high, dtype=torch.float32, device=self.device)
    self._ccd_tolerance_midlow = torch.tensor(
        np.stack((tolerance_mid, tolerance_low), axis=1),
        dtype=torch.float32, device=self.device)
    self._raw_active = torch.zeros(
        (self.batch_size, d.slot_count), dtype=torch.int32, device=self.device)
    self._source_active = torch.zeros_like(self._raw_active)
    self._raw_active_mask = torch.zeros(
        (self.batch_size, d.slot_count), dtype=torch.bool, device=self.device)
    self._narrowphase_status = torch.zeros(
        (self.batch_size, d.slot_count), dtype=torch.int32, device=self.device)
    # Source-audit trace for native sphere/flex CCD. Fields 0:16 hold point
    # and full GJK distances, simplex count, EPA init status, terminal
    # iteration/face/bounds/residual/normal and workspace counts; 16:28 and
    # 28:40 hold GJK and post-initialization simplex points. Fields 40:44
    # identify candidate kind, geometry type, node count and primitive route;
    # 44:58 hold query/exit classification, iteration/horizon/map sizes,
    # bounds, query distances, shallow-test inputs and initializer fallback;
    # 58:64 hold the point-GJK witness pair. When enabled, 64:85 hold the last
    # EPA expansion's iteration/support/root-edge diagnostics and 85:133 hold
    # its emitted horizon face/edge sequence. Fields 136:174 capture the
    # first expansion's face-map order, overwritten by the first malformed
    # adjacency if one appears; 174:206 capture the selected face and its
    # three neighbors; 206 marks whether that captured adjacency was invalid.
    # The optional tail stores one bounded record per configured EPA iteration:
    # a 32-float header, pre/horizon/post face-map prefixes, then up to 24
    # ordered horizon face-edge pairs. Only `capture_ccd_trace=True` allocates
    # this larger workspace; production dispatch keeps the one-float dummy.
    # Physical separation remains inactive; an inconclusive EPA exit is
    # promoted to status 31.
    if self.capture_ccd_trace:
      ccd_trace_words = self._ccd_trace_words
      self._ccd_trace_storage = torch.zeros(
          (self._combined_trace_words,), dtype=torch.float32,
          device=self.device)
      self._ccd_trace = self._ccd_trace_storage[:ccd_trace_words].view(
          self.batch_size, d.slot_count, self._ccd_trace_size)
    else:
      self._ccd_trace = torch.zeros((1,), dtype=torch.float32,
                                    device=self.device)
      self._ccd_trace_storage = self._ccd_trace
    # A GJK event trace targets exactly one candidate. It records the point
    # and full-radius query, plus each query's nested tetrahedron test. Its
    # allocation is independent of total slot count and charged from the
    # configured CCD iteration count by _flex_gjk_trace_capacity.
    self._gjk_trace = (
        self._ccd_trace_storage[ccd_trace_words:].view(
            _FLEX_GJK_TRACE_STAGE_COUNT, self._gjk_trace_iterations,
            self._gjk_trace_stride)
        if self.capture_gjk_trace else torch.zeros(
            (1,), dtype=torch.float32, device=self.device))
    # Pinned mjc_ccd allocates these per query using ccd_iterations (see
    # engine_collision_gjk.c:mjc_ccdSize). They are persistent device scratch,
    # indexed by candidate slot, so large EPA state does not consume Metal's
    # per-thread private stack. The flat typed views are the shader ABI:
    # A per-slot float arena stores CxVertex=12 floats, FlexDDVertex=27
    # floats, FlexEpaFace=20 floats. The int arena stores map, horizon pairs,
    # and DFS frames (six int32 words each), with the map padded for int2.
    epa_slots = max(int(d.slot_count), 1)
    self._epa_float_workspace = torch.empty(
        (self.batch_size, epa_slots,
         (self._epa_vertex_capacity * 39 + self._epa_face_capacity * 20 + 3)
             & ~3),
        dtype=torch.float32, device=self.device)
    self._epa_int_workspace = torch.empty(
        (self.batch_size, epa_slots,
         ((self._epa_face_capacity + 1) & ~1)
         + self._epa_horizon_capacity * 2
         + self._epa_stack_capacity * 6),
        dtype=torch.int32, device=self.device)
    self._filtered_active = torch.zeros_like(self._raw_active_mask)
    # filterFlexContacts mutates a compact contact array while its selected
    # and min_dist scratch remains indexed by array position. Keep a slot
    # permutation plus the source-shaped position-indexed scratch in owned
    # device storage; the selector executes one serial thread per world.
    self._filter_permutation = torch.empty(
        (self.batch_size, d.slot_count), dtype=torch.int32, device=self.device)
    self._filter_selected_position = torch.empty_like(self._filter_permutation)
    self._filter_min_distance_words = torch.empty(
        (self.batch_size, d.slot_count, 3), dtype=torch.float32,
        device=self.device)
    self._contact_dist = torch.empty(
        (self.batch_size, d.slot_count), dtype=torch.float32, device=self.device)
    self._contact_pos = torch.empty(
        (self.batch_size, d.slot_count, 3), dtype=torch.float32, device=self.device)
    # SDF addPreContact duplicate checks use its raw Frank-Wolfe point. The
    # later group contact cap uses the emitted contact midpoint separately.
    self._selection_pos = torch.empty_like(self._contact_pos)
    # The first two planes retain the emitted contact midpoint residual; the
    # final two retain the raw SDF Frank-Wolfe selection point.
    self._contact_pos_low_tail = torch.zeros(
        (4, self.batch_size, d.slot_count, 3), dtype=torch.float32,
        device=self.device)
    self._contact_normal = torch.empty_like(self._contact_pos)
    self._contact_frame = torch.empty(
        (self.batch_size, d.slot_count, 9), dtype=torch.float32,
        device=self.device)
    self._contact_frame_view = self._contact_frame.reshape(
        self.batch_size, d.slot_count, 3, 3)
    self._contact_bary1 = torch.empty(
        (self.batch_size, d.slot_count, 4), dtype=torch.float32,
        device=self.device)
    self._contact_bary2 = torch.empty_like(self._contact_bary1)
    self._contact_jacobian_weights = torch.empty(
        (2, self.batch_size, d.slot_count, 4), dtype=torch.float32,
        device=self.device)
    self._contact_jacobian_weight1 = self._contact_jacobian_weights[0]
    self._contact_jacobian_weight2 = self._contact_jacobian_weights[1]
    self._link_tree_pairs = tensor(d.link_tree_pairs.reshape(-1), torch.int32)
    self.link_capacity = int(d.link_capacity)
    self._link_dims = _flex_dims_with_world_mask(torch,
        [self.batch_size, d.slot_count, d.max_links_per_candidate,
         d.link_capacity, d.link_capacity], self.batch_size, self.device)
    self._contact_jac_dims = _flex_dims_with_world_mask(
        torch, [d.nv, d.slot_count, int(model.nflexvert), self.batch_size],
        self.batch_size, self.device)
    jac_shape = (self.batch_size, d.slot_count, 3, d.nv)
    spatial_shape = (self.batch_size, d.slot_count, 6, d.nv)
    if self.sparse_rows:
      # Sparse coupled assembly writes source-derived contact J directly into
      # its canonical CSR record; do not retain candidate-by-DOF tensors.
      self._contact_side1_jacobian = torch.empty(
          (1,), dtype=torch.float32, device=self.device)
      self._contact_side2_jacobian = torch.empty_like(
          self._contact_side1_jacobian)
      self._contact_relative_jacobian = torch.empty_like(
          self._contact_side1_jacobian)
      self._contact_spatial_side1_jacobian = torch.empty_like(
          self._contact_side1_jacobian)
      self._contact_spatial_side2_jacobian = torch.empty_like(
          self._contact_side1_jacobian)
      self._contact_spatial_relative_jacobian = torch.empty_like(
          self._contact_side1_jacobian)
    else:
      self._contact_side1_jacobian = torch.empty(
          jac_shape, dtype=torch.float32, device=self.device)
      self._contact_side2_jacobian = torch.empty_like(
          self._contact_side1_jacobian)
      self._contact_relative_jacobian = torch.empty_like(
          self._contact_side1_jacobian)
      self._contact_spatial_side1_jacobian = torch.empty(
          spatial_shape, dtype=torch.float32, device=self.device)
      self._contact_spatial_side2_jacobian = torch.empty_like(
          self._contact_spatial_side1_jacobian)
      self._contact_spatial_relative_jacobian = torch.empty_like(
          self._contact_spatial_side1_jacobian)
    geom_roots = np.full(d.slot_count, -1, dtype=np.int32)
    geom_dof_mask = np.zeros((d.slot_count, d.nv), dtype=np.int32)
    body_rootid = np.asarray(model.body_rootid, dtype=np.int32)
    geom_bodyid = np.asarray(model.geom_bodyid, dtype=np.int32)
    body_parentid = np.asarray(model.body_parentid, dtype=np.int32)
    body_jntadr = np.asarray(model.body_jntadr, dtype=np.int32)
    body_jntnum = np.asarray(model.body_jntnum, dtype=np.int32)
    joint_type = np.asarray(model.jnt_type, dtype=np.int32)
    joint_dofadr = np.asarray(model.jnt_dofadr, dtype=np.int32)
    dof_count = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    for slot, geom in enumerate(d.geom):
      if int(geom) >= 0:
        body = int(geom_bodyid[int(geom)])
        geom_roots[slot] = body_rootid[body]
        current = body
        while current > 0:
          start = int(body_jntadr[current])
          stop = start + int(body_jntnum[current])
          for joint in range(start, stop):
            count = dof_count[int(joint_type[joint])]
            first = int(joint_dofadr[joint])
            geom_dof_mask[slot, first:first + count] = 1
          current = int(body_parentid[current])
    self._geom_root = tensor(geom_roots, torch.int32)
    self._geom_dof_mask = tensor(geom_dof_mask.reshape(-1), torch.int32)
    body_dof_mask = np.zeros((int(model.nbody), d.nv), dtype=np.int32)
    for body in range(1, int(model.nbody)):
      current = body
      while current > 0:
        start = int(body_jntadr[current])
        stop = start + int(body_jntnum[current])
        for joint in range(start, stop):
          count = dof_count[int(joint_type[joint])]
          first = int(joint_dofadr[joint])
          body_dof_mask[body, first:first + count] = 1
        current = int(body_parentid[current])
    self._body_root = tensor(np.asarray(model.body_rootid, dtype=np.int32),
                             torch.int32)
    self._body_dof_mask = tensor_nonempty(
        body_dof_mask.reshape(-1), torch.int32)
    self._source_spatial_dims = _flex_dims_with_world_mask(torch,
        [self.batch_size, d.slot_count, d.nv, int(model.nflexvert),
         int(model.nflex), int(model.nflexnode), int(model.nbody)],
        self.batch_size, self.device)
    self._source_packed_dims = _flex_dims_with_world_mask(torch,
        [self.batch_size, d.slot_count, d.nv, int(model.nflexvert),
         int(model.nflex), int(model.nflexnode), int(model.nbody), 0, 0],
        self.batch_size, self.device)
    self._geom_spatial_dims = _flex_dims_with_world_mask(
        torch, [self.batch_size, d.slot_count, d.nv, int(model.nbody)],
        self.batch_size, self.device)
    self._geom_contact_spatial_jacobian = (
        torch.empty((1,), dtype=torch.float32, device=self.device)
        if self.sparse_rows else torch.empty(
            spatial_shape, dtype=torch.float32, device=self.device))
    self._active_links = torch.full(
        (self.batch_size, d.link_capacity, 2), -1, dtype=torch.int32,
        device=self.device)
    self._active_link_overflow = torch.zeros(
        (self.batch_size,), dtype=torch.int32, device=self.device)
    if self.device.type == "mps" and d.slot_count:
      started = time.perf_counter()
      self._shader = torch.mps.compile_shader(
          _COLLISION_SHADER.read_text() + "\n" + _CONVEX_SHADER.read_text()
          + "\n" + _PLUGIN_SDF_SHADER.read_text()
          + "\n" + _SDF_SHADER.read_text() + "\n" + _SHADER.read_text()
          + "\n" + _COMMON_CCD_SHADER.read_text()
          + "\n" + _COMMON_FLEX_PAIR_SHADER.read_text())
      _flex_pipeline_log("compile_shader returned", started)
    else:
      self._shader = None

  def run_device(self, flexvert_xpos, geom_pos, geom_quat, *,
                 flexvert_xpos_low=None,
                 flexvert_xpos_tail=None,
                 geom_pos_low=None, geom_pos_tail=None,
                 flexvert_spatial_jacobian=None, cdof=None, root_com=None,
                 qvel=None, diagA=None, relative_surface_velocity=None,
                 include_wake_links=False, row_workspace=None,
                 packed_jacobian=None, canonical_row_offset=0,
                 world_mask=None):
    """Return one immutable-identity result record per admitted candidate.

    Fixed candidate identities and row spans are never compacted in Python.
    The constructor's fail-closed admission flag remains authoritative until
    every lowered geometry/element route has pinned narrowphase parity.
    """
    torch = self._torch
    d = self.descriptor
    b, nslot = int(flexvert_xpos.shape[0]), d.slot_count
    if flexvert_xpos_low is None:
      flexvert_xpos_low = self._zero_flexvert_xpos_low
    if flexvert_xpos_tail is None:
      flexvert_xpos_tail = self._zero_flexvert_xpos_tail
    if geom_pos_low is None:
      geom_pos_low = self._zero_geom_pos_low
    if geom_pos_tail is None:
      geom_pos_tail = self._zero_geom_pos_tail
    expected_low = (b, int(self.model.nflexvert), 3)
    if (not isinstance(flexvert_xpos_low, torch.Tensor)
        or tuple(flexvert_xpos_low.shape) != expected_low
        or flexvert_xpos_low.dtype != torch.float32
        or flexvert_xpos_low.device.type != self.device.type
        or not flexvert_xpos_low.is_contiguous()):
      raise ValueError(
          f"flexvert_xpos_low must be contiguous {self.device.type} float32 {expected_low}")
    if (not isinstance(flexvert_xpos_tail, torch.Tensor)
        or tuple(flexvert_xpos_tail.shape) != expected_low
        or flexvert_xpos_tail.dtype != torch.float32
        or flexvert_xpos_tail.device.type != self.device.type
        or not flexvert_xpos_tail.is_contiguous()):
      raise ValueError(
          f"flexvert_xpos_tail must be contiguous {self.device.type} float32 {expected_low}")
    expected_geom = (b, int(self.model.ngeom), 3)
    for name, value in (("geom_pos_low", geom_pos_low),
                        ("geom_pos_tail", geom_pos_tail)):
      if (not isinstance(value, torch.Tensor)
          or tuple(value.shape) != expected_geom
          or value.dtype != torch.float32
          or value.device.type != self.device.type
          or not value.is_contiguous()):
        raise ValueError(
            f"{name} must be contiguous {self.device.type} float32 {expected_geom}")
    if bool(int(self.model.opt.disableflags) & _DISABLE_NATIVECCD) == self._native_ccd_enabled:
      raise ValueError(
          "native CCD disableflags changed after flex contact lowering; "
          "reconstruct the flex contact program for the new option set")
    self._diag_current_flexvert_xpos = flexvert_xpos
    if b != self.batch_size:
      raise ValueError("flex contact batch size differs from its lowering")
    self._pack_paired_flexvert_positions(
        flexvert_xpos, flexvert_xpos_low, flexvert_xpos_tail)
    self._geom_pos_low_tail[0].copy_(geom_pos_low)
    self._geom_pos_low_tail[1].copy_(geom_pos_tail)
    if world_mask is not None:
      if (not isinstance(world_mask, torch.Tensor)
          or tuple(world_mask.shape) != (b,)
          or world_mask.dtype != torch.int32
          or world_mask.device.type != self.device.type
          or not world_mask.is_contiguous()):
        raise ValueError(
            "world_mask must be contiguous int32 [B] on the contact device")
      if not d.global_enabled:
        raise ValueError(
            "world-masked flex contact requires the compiled contact path")
    if world_mask is None:
      self._world_mask_current.fill_(1)
    else:
      self._world_mask_current.copy_(world_mask)
    if nslot == 0:
      # The constructor owns empty, correctly typed views too. Reuse them so
      # an all-static/zero-feature model does not allocate device outputs on
      # every simulation step.
      result = {"active": self._filtered_active,
              "dist": self._contact_dist,
              "raw_active": self._raw_active_mask,
              "narrowphase_status": self._narrowphase_status,
              "ccd_trace": self._ccd_trace,
              "gjk_trace": (self._gjk_trace
                            if self.capture_gjk_trace else None),
              "pos": self._contact_pos,
              "normal": self._contact_normal,
              "frame": self._contact_frame_view,
                "barycentric1": self._contact_bary1,
                "barycentric2": self._contact_bary2,
                "jacobian_weights1": self._contact_jacobian_weight1,
                "jacobian_weights2": self._contact_jacobian_weight2,
              "slot_id": self._slot_id, "kind": self._kind,
              "row_start": self._row_start, "row_span": self._row_span,
              "condim": self._condim, "cone": self._cone,
              "friction": self._friction, "solref": self._solref,
              "solimp": self._solimp, "geom": self._geom,
              "flex1": self._flex1, "elem1": self._elem1,
              "flex2": self._flex2, "elem2": self._elem2,
              "vert1": self._vert1, "vert2": self._vert2,
              "nodes1": self._nodes1, "nodes2": self._nodes2,
              "margin": self._margin, "gap": self._gap,
                "feature1": self._feature1, "feature2": self._feature2,
                "contact_ordinal": self._contact_ordinal}
      return self._finish_contact_pipeline(
          result, flexvert_spatial_jacobian, cdof, root_com, qvel, diagA,
          relative_surface_velocity, include_wake_links, row_workspace,
          packed_jacobian, canonical_row_offset,
          self._world_mask_current)
    if d.global_enabled and not self._narrowphase_admitted:
      raise NotImplementedError(
          "flex element/geom and self/cross narrowphase is not admitted until "
          "the pinned support/CCD/manifold implementation is complete")
    raw_active = self._raw_active
    active = self._filtered_active
    dist = self._contact_dist
    pos = self._contact_pos
    normal = self._contact_normal
    frame = self._contact_frame
    bary1 = self._contact_bary1
    bary2 = self._contact_bary2
    if not d.global_enabled:
      raw_active.zero_()
      active.zero_()
      self._raw_active_mask.zero_()
      self._narrowphase_status.zero_()
      self._ccd_trace.zero_()
      if self.capture_gjk_trace:
        self._gjk_trace.zero_()
      dist.zero_()
      pos.zero_()
      self._contact_pos_low_tail.zero_()
      normal.zero_()
      frame.zero_()
      bary1.zero_()
      bary2.zero_()
      self._contact_jacobian_weight1.zero_()
      self._contact_jacobian_weight2.zero_()
    elif self.device.type == "mps":
      expected_shapes = (
          ("flexvert_xpos", flexvert_xpos,
           (b, int(self.model.nflexvert), 3)),
          ("geom_pos", geom_pos, (b, int(self.model.ngeom), 3)),
          ("geom_quat", geom_quat, (b, int(self.model.ngeom), 4)))
      for name, value, shape in expected_shapes:
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or value.dtype != torch.float32 or value.device.type != "mps"
            or not value.is_contiguous()):
          raise ValueError(f"{name} must be contiguous MPS float32 with shape {shape}")
      diagnostics = (os.environ.get("MUJOCO_METAL_FLEX_PIPELINE_DIAGNOSTICS")
                     == "1")
      if diagnostics:
        started = time.perf_counter()
        print("[flex-metal] resolving flex_contact_detect pipeline",
              file=sys.stderr, flush=True)
      detect_kernel = self._shader.flex_contact_detect
      if diagnostics:
        _flex_pipeline_log("flex_contact_detect pipeline resolved", started)
        started = time.perf_counter()
      detect_kernel(
          self._detect_meta,
          self._nodes1.reshape(-1), self._nodes2.reshape(-1),
          flexvert_xpos.reshape(-1), geom_pos.reshape(-1), geom_quat.reshape(-1),
          self._geom_size.reshape(-1), self._geom_type, self._radius1,
          self._radius2, self._margin_gap.reshape(-1), self._geom_hull,
          self._geom_hull_info, self._detect_dims,
          raw_active.reshape(-1),
          dist.reshape(-1), pos.reshape(-1), normal.reshape(-1),
          bary1.reshape(-1), bary2.reshape(-1), frame.reshape(-1),
          self._selection_pos.reshape(-1), self._diag_flex_vertadr,
          self._contact_jacobian_weights.reshape(-1), self._ccd_tolerance,
          self._narrowphase_status.reshape(-1),
          self._ccd_trace.reshape(-1),
          self._radius1_midlow.reshape(-1),
          self._margin_gap_midlow.reshape(-1),
          self._flexvert_xpos_low_tail.reshape(-1),
          self._contact_pos_low_tail.reshape(-1),
          threads=(b*nslot,), group_size=(128,))
      if diagnostics:
        _flex_pipeline_log("flex_contact_detect dispatch returned", started)
        started = time.perf_counter()
        torch.mps.synchronize()
        _flex_pipeline_log("flex_contact_detect synchronize returned", started)
      if self._native_ccd_enabled and self._has_native_ccd_candidates:
        if diagnostics:
          started = time.perf_counter()
          print("[flex-metal] resolving flex_contact_detect_native_ccd pipeline",
                file=sys.stderr, flush=True)
        ccd_kernel = self._shader.flex_contact_detect_native_ccd
        if diagnostics:
          _flex_pipeline_log("native CCD pipeline resolved", started)
          started = time.perf_counter()
        if self.capture_gjk_trace:
          self._gjk_trace.zero_()
        ccd_kernel(
            self._detect_meta,
            self._nodes1.reshape(-1), flexvert_xpos.reshape(-1),
            geom_pos.reshape(-1), geom_quat.reshape(-1),
            self._geom_size.reshape(-1), self._geom_type, self._radius1,
            self._margin_gap.reshape(-1),
            self._flexvert_xpos_low_tail.reshape(-1),
            self._geom_hull_info, self._detect_dims,
            raw_active.reshape(-1), dist.reshape(-1), pos.reshape(-1),
            normal.reshape(-1), bary1.reshape(-1), bary2.reshape(-1),
            frame.reshape(-1), self._selection_pos.reshape(-1),
            self._diag_flex_vertadr, self._contact_jacobian_weights.reshape(-1),
            self._ccd_tolerance, self._narrowphase_status.reshape(-1),
            self._ccd_trace_storage.reshape(-1),
            self._epa_float_workspace.reshape(-1),
            self._epa_int_workspace.reshape(-1),
            self._radius1_midlow.reshape(-1),
            self._geom_size_midlow.reshape(-1),
            self._margin_gap_midlow.reshape(-1),
            self._ccd_tolerance_midlow.reshape(-1),
            threads=(b*nslot,), group_size=(128,))
        if diagnostics:
          _flex_pipeline_log("native CCD dispatch returned", started)
          started = time.perf_counter()
          torch.mps.synchronize()
          _flex_pipeline_log("native CCD synchronize returned", started)
      if self._native_ccd_enabled and self._has_native_flex_pair_candidates:
        pair_kernel = self._shader.flex_contact_detect_common_element_pair
        pair_kernel(
            self._detect_meta,
            self._nodes1.reshape(-1), self._flexvert_xpos_pair.reshape(-1),
            geom_pos.reshape(-1), geom_quat.reshape(-1),
            self._geom_size.reshape(-1), self._geom_type, self._radius1,
            self._margin_gap.reshape(-1), self._nodes2.reshape(-1),
            self._geom_hull_info, self._detect_dims,
            raw_active.reshape(-1), dist.reshape(-1), pos.reshape(-1),
            normal.reshape(-1), bary1.reshape(-1), bary2.reshape(-1),
            frame.reshape(-1), self._selection_pos.reshape(-1),
            self._diag_flex_vertadr, self._contact_jacobian_weights.reshape(-1),
            self._ccd_tolerance, self._narrowphase_status.reshape(-1),
            self._ccd_trace_storage.reshape(-1),
            self._epa_float_workspace.reshape(-1),
            self._epa_int_workspace.reshape(-1),
            self._radius1_midlow.reshape(-1),
            self._geom_size_midlow.reshape(-1),
            self._margin_gap_midlow.reshape(-1),
            self._ccd_tolerance_midlow.reshape(-1),
            threads=(b*nslot,), group_size=(128,))
      self._shader.flex_contact_select(
          raw_active.reshape(-1), dist.reshape(-1),
          self._selection_pos.reshape(-1), pos.reshape(-1),
          self._filter_group, self._filterable, self._filter_mode, self._elem1,
          self._geom, self._sync_world_mask_dims(self._filter_dims, 4),
          self._source_active.reshape(-1),
          self._raw_active_mask.reshape(-1),
          active.reshape(-1),
          self._filter_permutation.reshape(-1),
          self._filter_selected_position.reshape(-1),
          self._filter_min_distance_words.reshape(-1),
          self._contact_pos_low_tail.reshape(-1),
          threads=(b,), group_size=(1,))
    else:
      raise RuntimeError("flex contact narrowphase requires native Metal execution")
    result = {"active": active,
            "raw_active": self._raw_active_mask,
            "narrowphase_status": self._narrowphase_status,
            "ccd_trace": self._ccd_trace if self.capture_ccd_trace else None,
            "gjk_trace": self._gjk_trace if self.capture_gjk_trace else None,
            "dist": dist, "pos": pos,
            "normal": normal, "frame": self._contact_frame_view,
            "barycentric1": bary1,
            "barycentric2": bary2,
            "jacobian_weights1": self._contact_jacobian_weight1,
            "jacobian_weights2": self._contact_jacobian_weight2,
            "row_start": self._row_start,
            "row_span": self._row_span, "slot_id": self._slot_id,
            "kind": self._kind, "condim": self._condim,
            "cone": self._cone, "friction": self._friction,
            "solref": self._solref, "solimp": self._solimp,
            "geom": self._geom, "flex1": self._flex1,
            "elem1": self._elem1, "vert1": self._vert1,
            "flex2": self._flex2, "elem2": self._elem2,
            "vert2": self._vert2, "nodes1": self._nodes1,
            "nodes2": self._nodes2, "feature1": self._feature1,
            "feature2": self._feature2, "margin": self._margin,
            "gap": self._gap,
            "contact_ordinal": self._contact_ordinal}
    return self._finish_contact_pipeline(
        result, flexvert_spatial_jacobian, cdof, root_com, qvel, diagA,
        relative_surface_velocity, include_wake_links, row_workspace,
        packed_jacobian, canonical_row_offset, self._world_mask_current)

  def _finish_contact_pipeline(self, result, flexvert_spatial_jacobian,
                               cdof, root_com, qvel, diagA,
                               relative_surface_velocity, include_wake_links,
                               row_workspace, packed_jacobian=None,
                               canonical_row_offset=0, world_mask=None):
    # `active` is candidate-slot state; `row_active` is solver-row state.
    # Their axes differ whenever a candidate owns multiple cone rows.
    pipeline_args = (flexvert_spatial_jacobian, cdof, root_com, qvel)
    if any(value is not None for value in pipeline_args) or diagA is not None:
      if any(value is None for value in pipeline_args):
        raise ValueError(
            "contact row assembly requires flexvert_spatial_jacobian, cdof, "
            "root_com, and qvel together")
      row_data = self.run_spatial_rows_device(
          result, flexvert_spatial_jacobian, cdof, root_com, qvel, diagA,
          relative_surface_velocity=relative_surface_velocity,
          row_workspace=row_workspace, packed_jacobian=packed_jacobian,
          canonical_row_offset=canonical_row_offset,
          world_mask=world_mask)
      # Preserve the candidate-axis active mask for link generation and
      # contact identity consumers. Canonical solver-row liveness has a
      # different axis and must never overwrite it when slot spans expand.
      result["row_active"] = row_data.pop("active")
      result.update(row_data)
    elif (relative_surface_velocity is not None or row_workspace is not None
          or packed_jacobian is not None or canonical_row_offset != 0):
      raise ValueError(
          "relative_surface_velocity and row_workspace require row assembly inputs")
    if include_wake_links:
      result["active_tree_links"], result["active_tree_link_overflow"] = (
          self.run_active_tree_links(result, world_mask=world_mask))
    return result

  def _pack_paired_flexvert_positions(self, high, low, tail):
    """Pack high+two residual coordinate planes into CCD kernel ABIs."""
    pair = self._flexvert_xpos_pair.reshape(
        self.batch_size, 3, int(self.model.nflexvert), 3)
    pair[:, 0].copy_(high)
    pair[:, 1].copy_(low)
    pair[:, 2].copy_(tail)
    vertex_words = self.batch_size * int(self.model.nflexvert) * 3
    self._flexvert_xpos_low_tail[:vertex_words].copy_(low.reshape(-1))
    self._flexvert_xpos_low_tail[vertex_words:2 * vertex_words].copy_(
        tail.reshape(-1))
    return self._flexvert_xpos_pair

  def _sync_world_mask_dims(self, dims, fixed_words, world_mask=None):
    """Copy the current batch mask into an existing dimension suffix."""
    expected = int(fixed_words) + self.batch_size
    if int(dims.numel()) != expected:
      raise AssertionError(
          f"world-mask dims ABI expected {expected} words, got {dims.numel()}")
    if world_mask is not None:
      self._world_mask_current.copy_(world_mask)
    dims[int(fixed_words):].copy_(self._world_mask_current)
    return dims

  def run_spatial_rows_device(self, contact_result,
                              flexvert_spatial_jacobian, cdof, root_com,
                              qvel, diagA, *,
                              relative_surface_velocity=None,
                              row_workspace=None, packed_jacobian=None,
                              canonical_row_offset=0, world_mask=None):
    """Run both-side spatial-J and canonical row assembly for detected slots.

    ``diagA`` must be the local `[B,row_capacity]` source-derived diagonal
    approximation for this fixed slot block. The coupled-constraint owner
    places these borrowed rows into its global canonical slice. No live mask
    or row address is read back to the host.
    """
    if self.device.type != "mps":
      raise RuntimeError("flex candidate row pipeline requires native Metal")
    if row_workspace is None:
      rows = getattr(self, "_row_workspace", None)
      if (rows is None
          or bool(getattr(rows, "sparse_only", False))
              != (packed_jacobian is not None)):
        from .flex_contact_rows import FlexContactRows
        rows = FlexContactRows(
            self.model, self.descriptor, batch_size=self.batch_size,
            device=str(self.device),
            sparse_only=packed_jacobian is not None)
        self._row_workspace = rows
    else:
      rows = row_workspace
      if (getattr(rows, "_descriptor", None) is not self.descriptor
          or int(getattr(rows, "batch_size", -1)) != self.batch_size
          or int(getattr(rows, "nv", -1)) != self.descriptor.nv
          or int(getattr(rows, "nrow", -1)) != self.descriptor.row_capacity
          or bool(getattr(rows, "sparse_only", False))
              != (packed_jacobian is not None)):
        raise ValueError("row_workspace does not match this contact descriptor")
    if diagA is None:
      diagA = self.run_diag_approx_device(
          contact_result, world_mask=world_mask)
    if packed_jacobian is not None:
      if not self.sparse_rows:
        raise ValueError(
            "packed flex rows require a sparse_rows contact workspace")
      self._write_source_packed_jacobian(
          contact_result, cdof, root_com, rows, packed_jacobian,
          canonical_row_offset, world_mask)
      row_data = rows.run_device(
          contact_result, None, qvel, diagA,
          relative_surface_velocity=relative_surface_velocity,
          packed_jacobian=packed_jacobian,
          canonical_row_offset=canonical_row_offset,
          world_mask=world_mask)
      return {"packed_source_jacobian": packed_jacobian, **row_data}
    if self.sparse_rows:
      raise ValueError("sparse_rows workspaces require packed_jacobian output")
    spatial = self.run_native_contact_spatial_jacobians(
        contact_result, flexvert_spatial_jacobian, cdof, root_com)
    row_data = rows.run_device(
        contact_result, spatial["relative"], qvel, diagA,
        relative_surface_velocity=relative_surface_velocity,
        world_mask=world_mask)
    return {"side1_spatial_jacobian": spatial["side1"],
            "side2_spatial_jacobian": spatial["side2"],
            "relative_spatial_jacobian": spatial["relative"],
            "geom_point_spatial_jacobian": spatial["geom_point"],
            **row_data}

  def _write_source_packed_jacobian(self, contact_result, cdof, root_com,
                                    rows, packed_jacobian,
                                    canonical_row_offset, world_mask=None):
    """Project source point-J values directly into the canonical packed CSR."""
    torch = self._torch
    d = self.descriptor
    if self.device.type != "mps":
      raise RuntimeError("packed flex Jacobian production requires Metal")
    if (isinstance(canonical_row_offset, bool)
        or int(canonical_row_offset) != canonical_row_offset
        or int(canonical_row_offset) < 0
        or int(canonical_row_offset) > np.iinfo(np.int32).max):
      raise ValueError("canonical row offset must fit signed int32")
    canonical_row_offset = int(canonical_row_offset)
    if (not isinstance(packed_jacobian, torch.Tensor)
        or packed_jacobian.ndim != 1 or packed_jacobian.numel() < 12
        or packed_jacobian.numel() > np.iinfo(np.int32).max
        or packed_jacobian.dtype != torch.float32
        or packed_jacobian.device.type != "mps"
        or not packed_jacobian.is_contiguous()):
      raise ValueError(
          "packed_jacobian must be a flat contiguous MPS float32 CSR record")
    pos = contact_result.get("pos")
    frame = contact_result.get("frame")
    distance = contact_result.get("dist")
    active = contact_result.get("active")
    expected = (("contact_result.pos", pos,
                 (self.batch_size, d.slot_count, 3), torch.float32),
                ("contact_result.frame", frame,
                 (self.batch_size, d.slot_count, 3, 3), torch.float32),
                ("contact_result.dist", distance,
                 (self.batch_size, d.slot_count), torch.float32),
                ("contact_result.active", active,
                 (self.batch_size, d.slot_count), torch.bool),
                ("flexvert_xpos", self._diag_current_flexvert_xpos,
                 (self.batch_size, int(self.model.nflexvert), 3), torch.float32),
                ("cdof", cdof, (self.batch_size, d.nv, 6), torch.float32),
                ("root_com", root_com,
                 (self.batch_size, int(self.model.nbody), 3), torch.float32))
    for name, value, shape, dtype in expected:
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != dtype or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS {dtype} with shape {shape}")
    if (not bool(getattr(rows, "sparse_only", False))
        or getattr(rows, "_descriptor", None) is not d):
      raise ValueError("packed flex Jacobian requires matching sparse row workspace")
    dims = self._source_packed_dims
    dims[7] = canonical_row_offset
    dims[8] = int(packed_jacobian.numel())
    self._sync_world_mask_dims(dims, 9, world_mask)
    total = self.batch_size * d.slot_count * d.nv
    if total:
      self._shader.flex_contact_source_packed_rows(
          self._detect_meta.reshape(-1), self._nodes1.reshape(-1),
          self._nodes2.reshape(-1), pos.reshape(-1), frame.reshape(-1),
          active.reshape(-1), distance.reshape(-1), self._diag_current_flexvert_xpos.reshape(-1),
          self._diag_flex_vertadr, self._diag_flex_dim, self._diag_flex_interp,
          self._diag_flex_cellnum, self._diag_flex_vert0.reshape(-1),
          self._diag_flex_vertbody, self._diag_flex_nodeadr,
          self._diag_flex_nodebody, self._diag_geom_body, self._body_root,
          self._body_dof_mask, root_com.reshape(-1), cdof.reshape(-1),
          rows._row_start, rows._row_span, rows._condim, rows._cone,
          rows._friction, rows._margin, dims, packed_jacobian.reshape(-1),
          threads=(total,), group_size=(128,))

  def run_diag_approx_device(self, contact_result=None, world_mask=None):
    """Produce source-ordered ``mj_diagApprox`` values for contact candidates.

    The dynamic kernel uses each emitted contact point plus current flex-node
    positions to reproduce ``mj_elemBodyWeight`` and then ``mj_vertBodyWeight``
    for vertex, element, internal, self, and cross-flex descriptors. It is
    independent of the narrowphase admission gate; callers still must preserve
    that gate until each candidate generator/manifold is qualified.
    """
    if self.device.type != "mps":
      raise RuntimeError("flex diagApprox production requires native Metal")
    if self.descriptor.row_capacity == 0:
      return self._diag_approx
    use_dynamic = (contact_result is not None
                   and not self._diag_approx_supported)
    if (use_dynamic and not self._diag_dynamic_supported
        and self.descriptor.global_enabled):
      raise NotImplementedError(
          "dynamic diagApprox supports flex interpolation orders 0 through 2")
    if not use_dynamic:
      # Keep the qualified plane/vertex path on its compiled source weights.
      if not self._diag_approx_supported and self.descriptor.global_enabled:
        raise NotImplementedError(
            "pinned diagApprox is not lowered for this flex contact feature")
      self._shader.flex_contact_diag_approx(
          self._diag_components.reshape(-1), self._friction.reshape(-1),
          self._condim, self._cone, self._row_start, self._row_span,
          self._sync_world_mask_dims(self._diag_static_dims, 3, world_mask),
          self._diag_approx.reshape(-1),
          threads=(self.batch_size*self.descriptor.slot_count,), group_size=(128,))
    else:
      torch = self._torch
      pos = contact_result.get("pos")
      flexvert_xpos = self._diag_current_flexvert_xpos
      self._validate_diag_input("contact_result.pos", pos,
                                (self.batch_size, self.descriptor.slot_count, 3))
      self._validate_diag_input(
          "flexvert_xpos", flexvert_xpos,
          (self.batch_size, int(self.model.nflexvert), 3))
      self._shader.flex_contact_diag_approx_dynamic(
          self._kind, self._flex1, self._elem1, self._vert1,
          self._flex2, self._elem2, self._vert2, self._geom,
          self._nodes1.reshape(-1), self._nodes2.reshape(-1),
          pos.reshape(-1), flexvert_xpos.reshape(-1), self._friction.reshape(-1),
          self._condim, self._cone, self._row_start, self._row_span,
          self._diag_geom_body, self._diag_flex_interp,
          self._diag_flex_vertadr, self._diag_flex_nodeadr,
          self._diag_flex_cellnum, self._diag_flex_dim,
          self._diag_flex_vert0.reshape(-1), self._diag_flex_vertbody,
          self._diag_flex_nodebody, self._diag_body_invweight,
          self._sync_world_mask_dims(self._diag_dims, 7, world_mask),
          self._diag_approx.reshape(-1),
          threads=(self.batch_size*self.descriptor.slot_count,), group_size=(128,))
    return self._diag_approx

  def _validate_diag_input(self, name, value, shape):
    torch = self._torch
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
        or value.dtype != torch.float32 or value.device.type != "mps"
        or not value.is_contiguous()):
      raise ValueError(f"{name} must be contiguous MPS float32 with shape {shape}")

  def run_active_tree_links(self, contact_result, capacity=None,
                            out_links=None, out_overflow=None,
                            world_mask=None):
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
    self._shader.flex_contact_tree_links(
        active,
        self._candidate_link_ids.reshape(-1), self._link_tree_pairs,
        self._sync_world_mask_dims(self._link_dims, 5, world_mask),
        links.reshape(-1), overflow,
        threads=(self.batch_size,), group_size=(1,))
    return links, overflow

  def run_contact_jacobians(self, contact_result, flexvert_jacobian,
                            geom_point_jacobian, world_mask=None):
    """Interpolate descriptor-side point Jacobians at each raw witness.

    ``geom_point_jacobian`` is evaluated by the caller at each returned
    contact position, including the rigid angular lever arm. The result's
    canonical relative row is ``side1 - side2``; descriptor node identities
    and pinned inverse-distance Jacobian weights define the flex sides.
    Geometric barycentrics remain separate in the contact result. All returned arrays are
    borrowed, overwritten workspaces.
    """
    torch = self._torch
    d = self.descriptor
    if self.device.type != "mps":
      raise RuntimeError("flex contact Jacobians require native Metal execution")
    expected_flex = (self.batch_size, int(self.model.nflexvert), 3, d.nv)
    expected_geom = (self.batch_size, d.slot_count, 3, d.nv)
    for name, value, shape in (
        ("flexvert_jacobian", flexvert_jacobian, expected_flex),
        ("geom_point_jacobian", geom_point_jacobian, expected_geom)):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS float32 with shape {shape}")
    expected_weights = (self.batch_size, d.slot_count, 4)
    jac_weights1, jac_weights2 = (
        contact_result.get("jacobian_weights1"),
        contact_result.get("jacobian_weights2"))
    for name, value in (("jacobian_weights1", jac_weights1),
                        ("jacobian_weights2", jac_weights2)):
      if (not isinstance(value, torch.Tensor)
          or tuple(value.shape) != expected_weights
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"contact_result {name} must be contiguous MPS float32 "
                         f"with shape {expected_weights}")
    if d.slot_count == 0 or d.nv == 0:
      self._contact_side1_jacobian.zero_()
      self._contact_side2_jacobian.zero_()
      self._contact_relative_jacobian.zero_()
    else:
      total = self.batch_size * d.slot_count * 3 * d.nv
      self._shader.flex_contact_jacobians(
          self._geom, self._nodes1.reshape(-1),
          self._nodes2.reshape(-1), jac_weights1.reshape(-1),
          jac_weights2.reshape(-1), flexvert_jacobian.reshape(-1),
          geom_point_jacobian.reshape(-1),
          self._sync_world_mask_dims(
              self._contact_jac_dims, 4, world_mask),
          self._contact_side1_jacobian.reshape(-1),
          self._contact_side2_jacobian.reshape(-1),
          self._contact_relative_jacobian.reshape(-1),
          threads=(total,), group_size=(128,))
    return {"side1": self._contact_side1_jacobian,
            "side2": self._contact_side2_jacobian,
            "relative": self._contact_relative_jacobian}

  def run_contact_spatial_jacobians(self, contact_result,
                                    flexvert_spatial_jacobian,
                                    geom_point_spatial_jacobian,
                                    world_mask=None):
    """Interpolate full spatial point Jacobians for condim 1..6 contacts.

    Flex input is `[B,nflexvert,6,nv]` (linear then angular); geom input is
    `[B,slot,6,nv]` evaluated at each contact point so its angular half
    includes the moving rigid body's lever arm. Returns borrowed side1,
    side2, and side1-minus-side2 workspaces.
    """
    torch = self._torch
    d = self.descriptor
    if self.device.type != "mps":
      raise RuntimeError("flex contact spatial Jacobians require native Metal")
    expected_flex = (self.batch_size, int(self.model.nflexvert), 6, d.nv)
    expected_geom = (self.batch_size, d.slot_count, 6, d.nv)
    for name, value, shape in (
        ("flexvert_spatial_jacobian", flexvert_spatial_jacobian,
         expected_flex),
        ("geom_point_spatial_jacobian", geom_point_spatial_jacobian,
         expected_geom)):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS float32 with shape {shape}")
    jac_weights1, jac_weights2 = (
        contact_result.get("jacobian_weights1"),
        contact_result.get("jacobian_weights2"))
    expected_weights = (self.batch_size, d.slot_count, 4)
    for name, value in (("jacobian_weights1", jac_weights1),
                        ("jacobian_weights2", jac_weights2)):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_weights
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"contact_result {name} must be contiguous MPS float32 "
                         f"with shape {expected_weights}")
    total = self.batch_size * d.slot_count * 6 * d.nv
    if total:
      self._shader.flex_contact_spatial_jacobians(
          self._geom, self._nodes1.reshape(-1), self._nodes2.reshape(-1),
          jac_weights1.reshape(-1), jac_weights2.reshape(-1),
          flexvert_spatial_jacobian.reshape(-1),
          geom_point_spatial_jacobian.reshape(-1),
          self._sync_world_mask_dims(self._contact_jac_dims, 4, world_mask),
          self._contact_spatial_side1_jacobian.reshape(-1),
          self._contact_spatial_side2_jacobian.reshape(-1),
          self._contact_spatial_relative_jacobian.reshape(-1),
          threads=(total,), group_size=(128,))
    return {"side1": self._contact_spatial_side1_jacobian,
            "side2": self._contact_spatial_side2_jacobian,
            "relative": self._contact_spatial_relative_jacobian}

  def run_geom_contact_spatial_jacobians(self, contact_result, cdof,
                                         root_com):
    """Evaluate each moving geom-side point Jacobian from compiled ``cdof``.

    Inputs are the native forward-stage ``cdof[B,nv,6]`` and root-subtree
    COMs ``root_com[B,nbody,3]``. The contact position comes from the raw
    narrowphase result, before any row projection. The output is
    ``[B,slot,6,nv]`` with linear then angular components and is intended as
    side 2 in ``run_contact_spatial_jacobians``; flex/flex and world geometry
    slots evaluate to zero. This keeps the rigid lever-arm calculation on the
    device and avoids rebuilding or reading a Jacobian from CPU physics.
    """
    torch = self._torch
    d = self.descriptor
    if self.device.type != "mps":
      raise RuntimeError("rigid flex-contact Jacobians require native Metal")
    pos = contact_result.get("pos")
    if (not isinstance(pos, torch.Tensor)
        or tuple(pos.shape) != (self.batch_size, d.slot_count, 3)
        or pos.dtype != torch.float32 or pos.device.type != "mps"
        or not pos.is_contiguous()):
      raise ValueError("contact_result.pos must be contiguous MPS float32 [B,S,3]")
    if (not isinstance(cdof, torch.Tensor)
        or tuple(cdof.shape) != (self.batch_size, d.nv, 6)
        or cdof.dtype != torch.float32 or cdof.device.type != "mps"
        or not cdof.is_contiguous()):
      raise ValueError("cdof must be contiguous MPS float32 [B,nv,6]")
    if (not isinstance(root_com, torch.Tensor)
        or tuple(root_com.shape) != (self.batch_size, int(self.model.nbody), 3)
        or root_com.dtype != torch.float32 or root_com.device.type != "mps"
        or not root_com.is_contiguous()):
      raise ValueError("root_com must be contiguous MPS float32 [B,nbody,3]")
    if not d.slot_count or not d.nv:
      self._geom_contact_spatial_jacobian.zero_()
      return self._geom_contact_spatial_jacobian
      self._shader.flex_geom_contact_spatial_jacobians(
        self._geom_root, self._geom_dof_mask, pos.reshape(-1), root_com.reshape(-1),
        cdof.reshape(-1),
        self._sync_world_mask_dims(self._geom_spatial_dims, 4),
        self._geom_contact_spatial_jacobian.reshape(-1),
        threads=(self.batch_size*d.slot_count*d.nv,), group_size=(128,))
    return self._geom_contact_spatial_jacobian

  def run_source_contact_spatial_jacobians(self, contact_result, cdof,
                                            root_com):
    """Evaluate pinned body-weight contact Jacobians at each raw witness.

    This follows ``mj_elemBodyWeight`` and ``mj_vertBodyWeight``: current
    inverse-distance weights are mapped through the compiled interpolation
    basis (including shell TFI) to node bodies, and each body Jacobian is
    translated from root COM to the contact position. This is the source
    representation for contact J; geometric witness barycentrics and
    flex-vertex point Jacobians are not substitutes for it.
    """
    torch = self._torch
    d = self.descriptor
    if self.device.type != "mps":
      raise RuntimeError("source flex contact Jacobians require native Metal")
    pos = contact_result.get("pos")
    expected_pos = (self.batch_size, d.slot_count, 3)
    if (not isinstance(pos, torch.Tensor) or tuple(pos.shape) != expected_pos
        or pos.dtype != torch.float32 or pos.device.type != "mps"
        or not pos.is_contiguous()):
      raise ValueError(f"contact_result.pos must be contiguous MPS float32 {expected_pos}")
    flexvert_xpos = self._diag_current_flexvert_xpos
    expected_xpos = (self.batch_size, int(self.model.nflexvert), 3)
    if (not isinstance(flexvert_xpos, torch.Tensor)
        or tuple(flexvert_xpos.shape) != expected_xpos
        or flexvert_xpos.dtype != torch.float32
        or flexvert_xpos.device.type != "mps"
        or not flexvert_xpos.is_contiguous()):
      raise ValueError(
          f"current flexvert_xpos must be contiguous MPS float32 {expected_xpos}")
    for name, value, shape in (
        ("cdof", cdof, (self.batch_size, d.nv, 6)),
        ("root_com", root_com,
         (self.batch_size, int(self.model.nbody), 3))):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
          or value.dtype != torch.float32 or value.device.type != "mps"
          or not value.is_contiguous()):
        raise ValueError(f"{name} must be contiguous MPS float32 with shape {shape}")
    if not d.slot_count or not d.nv:
      self._contact_spatial_side1_jacobian.zero_()
      self._contact_spatial_side2_jacobian.zero_()
      self._contact_spatial_relative_jacobian.zero_()
    else:
      self._shader.flex_contact_source_spatial_jacobians(
          self._detect_meta.reshape(-1), self._nodes1.reshape(-1),
          self._nodes2.reshape(-1), pos.reshape(-1),
          self._diag_current_flexvert_xpos.reshape(-1),
          self._diag_flex_vertadr, self._diag_flex_dim, self._diag_flex_interp,
          self._diag_flex_cellnum, self._diag_flex_vert0.reshape(-1),
          self._diag_flex_vertbody, self._diag_flex_nodeadr,
          self._diag_flex_nodebody, self._diag_geom_body,
          self._body_root, self._body_dof_mask, root_com.reshape(-1),
          cdof.reshape(-1),
          self._sync_world_mask_dims(self._source_spatial_dims, 7),
          self._contact_spatial_side1_jacobian.reshape(-1),
          self._contact_spatial_side2_jacobian.reshape(-1),
          self._contact_spatial_relative_jacobian.reshape(-1),
          threads=(self.batch_size*d.slot_count*6*d.nv,), group_size=(128,))
    return {"side1": self._contact_spatial_side1_jacobian,
            "side2": self._contact_spatial_side2_jacobian,
            "relative": self._contact_spatial_relative_jacobian}

  def run_native_contact_spatial_jacobians(self, contact_result,
                                            flexvert_spatial_jacobian,
                                            cdof, root_com):
    """Produce full-frame contact J from compiled flex/rigid body weights."""
    del flexvert_spatial_jacobian  # retained only for compatibility at the seam
    geom_jacobian = self.run_geom_contact_spatial_jacobians(
        contact_result, cdof, root_com)
    result = self.run_source_contact_spatial_jacobians(
        contact_result, cdof, root_com)
    return {**result, "geom_point": geom_jacobian}
