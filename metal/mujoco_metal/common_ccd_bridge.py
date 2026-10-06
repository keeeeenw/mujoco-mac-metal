"""Bounded model-to-common-CCD descriptors for the private production bridge.

The descriptors deliberately use MuJoCo's compiled mesh vertex and adjacency
arrays, rather than the separate collision-hull upload used by the legacy
convex path.  Runtime contact generation is performed by the shared native
GJK/EPA shader; this module only lowers immutable model data and checks its
workspace contract.
"""
from __future__ import annotations

from operator import index as _index

_INT32_MAX = (1 << 31) - 1
_EPA_CAP = 1000
_HORIZON_CAP = 24
_DIAG_GJK_SUPPORTS = 64
_DIAG_SUPPORT_WORDS = 38
_DIAG_EPA_SUPPORT_OFFSET = 2600
_DIAG_EPA_SUPPORTS = 64
_DIAG_EPA_FACE_OFFSET = _DIAG_EPA_SUPPORT_OFFSET + _DIAG_EPA_SUPPORTS * _DIAG_SUPPORT_WORDS
_DIAG_EPA_INIT_FACE_WORDS = 20
_DIAG_EPA_ITER_OFFSET = _DIAG_EPA_FACE_OFFSET + 6 * _DIAG_EPA_INIT_FACE_WORDS
_DIAG_EPA_ITER_WORDS = 34
_DIAG_EPA_ITERATIONS = 64
_DIAG_EPA_COUNTS_OFFSET = (
    _DIAG_EPA_ITER_OFFSET + _DIAG_EPA_ITER_WORDS * _DIAG_EPA_ITERATIONS)
_DIAG_PRODUCTION_CONTEXT_OFFSET = _DIAG_EPA_COUNTS_OFFSET + 2
_DIAG_PRODUCTION_CONTEXT_WORDS = 232
_DIAGNOSTIC_WORDS = (
    _DIAG_PRODUCTION_CONTEXT_OFFSET + _DIAG_PRODUCTION_CONTEXT_WORDS)

# Symmetric upper triangle of MuJoCo 3.10's mjCOLLISIONFUNC entries whose
# callback is exactly mjc_Convex. Plane, HField, SDF, and the hand-specialized
# sphere/capsule/box pairs deliberately stay on their own source paths.
_MJC_CONVEX_TYPE_PAIRS = frozenset({
    (2, 4), (2, 7),
    (3, 4), (3, 5), (3, 7),
    (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 5), (5, 6), (5, 7),
    (6, 7), (7, 7),
})


def uses_pinned_mjc_convex(type_a, type_b):
  """Whether a geom-type pair dispatches to pinned ``mjc_Convex``.

  The numeric type table is pinned MuJoCo 3.10's public ``mjtGeom`` ordering.
  This classifier is intentionally narrow: it does not absorb the dedicated
  plane, HField, SDF, or analytic collision callbacks.
  """
  a = _checked(type_a, "type_a")
  b = _checked(type_b, "type_b")
  if a > 8 or b > 8:
    raise ValueError("geom type is outside the pinned MuJoCo 3.10 range")
  return (min(a, b), max(a, b)) in _MJC_CONVEX_TYPE_PAIRS


def _checked(value, name, *, minimum=0):
  if isinstance(value, bool):
    raise TypeError(f"{name} must be an integer, not bool")
  try:
    result = _index(value)
  except TypeError as exc:
    raise TypeError(f"{name} must be an integer") from exc
  if result < minimum or result > _INT32_MAX:
    raise ValueError(f"{name} is outside signed-32-bit range")
  return result


def production_ccd_layout(batch_size, pair_count, iterations):
  """Return checked per-query common GJK/EPA scratch capacities.

  One arena is reused while a pair walks HField prisms.  The returned byte
  count includes every simultaneously retained output and EPA array, with
  int32/address checks performed before callers allocate device storage.
  """
  batch = _checked(batch_size, "batch_size", minimum=1)
  pairs = _checked(pair_count, "pair_count", minimum=0)
  iters = _checked(iterations, "iterations", minimum=1)
  epa = min(iters, _EPA_CAP)
  vertices = 6 + epa
  faces = 6 + 6 * epa
  queries = batch * pairs
  if queries > _INT32_MAX:
    raise ValueError("common CCD query count exceeds signed-32-bit addressing")
  per_query = {
      "vertices": 29 * vertices,
      "faces": 20 * faces,
      "face_map": faces,
      "horizon": 2 * _HORIZON_CAP,
      "horizon_stack": 6 * faces,
      "contact_records_and_header": 2 + 50 * 16,
  }
  # epa_horizon is a typed int2 view into this float-word arena.  Metal
  # requires an 8-byte aligned base, so add one word after the face map when
  # the preceding support-vertex region has odd length.  The resulting
  # per-query stride is even, keeping every world's horizon aligned.
  horizon_padding = per_query["vertices"] & 1
  per_query["horizon_alignment"] = horizon_padding
  for name, words in per_query.items():
    if queries * words > _INT32_MAX:
      raise ValueError(f"common CCD {name} exceeds signed-32-bit addressing")
  unaligned_runtime_words = (
      queries * (sum(per_query.values()) + 18) + _DIAGNOSTIC_WORDS)
  if unaligned_runtime_words + 3 > _INT32_MAX:
    raise ValueError("common CCD aligned workspace exceeds signed-32-bit addressing")
  return {
      "queries": queries,
      "iterations": iters,
      "epa_iterations": epa,
      "vertex_capacity": vertices,
      "face_capacity": faces,
      "horizon_capacity": _HORIZON_CAP,
      "stack_capacity": faces,
      "words_per_query": per_query,
      "arena_words_per_query": sum(
          per_query[key] for key in ("vertices", "faces", "face_map",
                                      "horizon_alignment", "horizon",
                                      "horizon_stack")),
      "prism_vertex_words_per_query": 18,
      "diagnostic_words": _DIAGNOSTIC_WORDS,
      "diagnostic_gjk_support_offset": 32,
      "diagnostic_gjk_support_words": _DIAG_SUPPORT_WORDS,
      "diagnostic_epa_support_offset": _DIAG_EPA_SUPPORT_OFFSET,
      "diagnostic_epa_support_capacity": _DIAG_EPA_SUPPORTS,
      "diagnostic_epa_face_offset": _DIAG_EPA_FACE_OFFSET,
      "diagnostic_epa_init_face_words": _DIAG_EPA_INIT_FACE_WORDS,
      "diagnostic_epa_iteration_offset": _DIAG_EPA_ITER_OFFSET,
      "diagnostic_epa_iteration_words": _DIAG_EPA_ITER_WORDS,
      "diagnostic_epa_iteration_capacity": _DIAG_EPA_ITERATIONS,
      "diagnostic_epa_counts_offset": _DIAG_EPA_COUNTS_OFFSET,
      "diagnostic_production_context_offset": _DIAG_PRODUCTION_CONTEXT_OFFSET,
      "diagnostic_production_context_words": _DIAG_PRODUCTION_CONTEXT_WORDS,
      "output_words_per_query": per_query["contact_records_and_header"],
      "bytes_total_unaligned": 4 * unaligned_runtime_words,
      "alignment_padding_words_upper_bound": 3,
      "bytes_total": 4 * (unaligned_runtime_words + 3),
      "bytes_total_is_upper_bound": True,
      "bytes_static_model_excluded": True,
  }


def append_production_workspace(model, legacy_vertices, legacy_info,
                                batch_size, pair_count):
  """Build source-mesh descriptors and bounded runtime candidate/arena tail."""
  import numpy as np

  vertices, info, layout = build_common_ccd_model_upload(
      model, legacy_vertices, legacy_info)
  work = production_ccd_layout(batch_size, pair_count,
                               int(model.opt.ccd_iterations))
  queries = work["queries"]
  output_base = int(vertices.size)
  output_end = output_base + queries * work["output_words_per_query"]
  output_alignment = output_end & 1
  scratch_base = output_end + output_alignment
  scratch_end = scratch_base + queries * work["arena_words_per_query"]
  dynamic_vertex_base = (scratch_end + 2) // 3
  alignment_padding = 3 * dynamic_vertex_base - scratch_end
  vertex_words = queries * work["prism_vertex_words_per_query"]
  diagnostic_base = 3 * dynamic_vertex_base + vertex_words
  total_words = diagnostic_base + work["diagnostic_words"]
  for name, value in (("output_base", output_base), ("scratch_base", scratch_base),
                      ("dynamic_vertex_base", dynamic_vertex_base),
                      ("workspace_end", total_words)):
    if value > _INT32_MAX:
      raise ValueError(f"common CCD {name} exceeds signed-32-bit addressing")
  header = layout["header_base"]
  info[header + 3] = dynamic_vertex_base
  info[header + 5] = output_base
  info[header + 6] = work["output_words_per_query"]
  info[header + 7] = scratch_base
  info[header + 8] = work["arena_words_per_query"]
  info[header + 11] = work["vertex_capacity"]
  info[header + 12] = work["face_capacity"]
  info[header + 13] = work["horizon_capacity"]
  info[header + 14] = work["stack_capacity"]
  vertices = np.pad(vertices, (0, total_words - vertices.size))
  # Feature IDs are kept in an int2 sidecar because d8d8's shared flex EPA
  # arena is the compact three-field FlexDDVertex ABI. Rigid multicontact
  # needs the support pair for each EPA vertex, while flex leaves this region
  # unused and passes a null sidecar.
  vertex_id_base = (int(info.size) + 1) & ~1
  vertex_id_words = 2 * queries * work["vertex_capacity"]
  if vertex_id_base + vertex_id_words > _INT32_MAX:
    raise ValueError("common CCD feature sidecar exceeds signed-32-bit addressing")
  expanded_info = np.zeros(vertex_id_base + vertex_id_words, dtype=np.int32)
  expanded_info[:info.size] = info
  info = expanded_info
  info[header + 24] = vertex_id_base
  info[header + 25] = work["vertex_capacity"]
  info[header + 26] = queries
  info[header + 27] = diagnostic_base
  info[header + 28] = work["diagnostic_words"]
  info[header + 29] = -1  # selected flattened world/pair query; disabled
  info[header + 30] = -1  # source row
  info[header + 31] = -1  # source column
  info[header + 32] = -1  # source triangle
  layout.update(work)
  layout.update({"output_base": output_base, "scratch_base": scratch_base,
                 "vertex_id_base": vertex_id_base,
                 "vertex_id_words": vertex_id_words,
                 "vertex_id_bytes": 4 * vertex_id_words,
                 "vertex_id_padding_words": vertex_id_base - int(layout["base_info_words"]),
                 "dynamic_vertex_base": dynamic_vertex_base,
                 "diagnostic_base": diagnostic_base,
                 "diagnostic_words": work["diagnostic_words"],
                 "diagnostic_support_capacity": 64,
                 "output_alignment_words": output_alignment,
                 "alignment_padding_words": alignment_padding,
                 "bytes_total": (4 * (
                     queries * (work["output_words_per_query"]
                                + work["arena_words_per_query"]
                                + work["prism_vertex_words_per_query"])
                     + work["diagnostic_words"]
                     + output_alignment + alignment_padding)
                     + 4 * (expanded_info.size - int(layout.get("base_info_words", 0)))),
                 "bytes_total_is_upper_bound": False,
                 "total_words": total_words})
  return (np.ascontiguousarray(vertices), np.ascontiguousarray(info), layout)


def estimate_production_upload_bytes(model, legacy_vertices, legacy_info,
                                     batch_size, pair_count):
  """Compute final common-CCD upload bytes without allocating its buffers.

  This is used before ``append_production_workspace`` so the configured memory
  budget can reject the exact final model+runtime upload before constructing
  the potentially large NumPy vertex, output, arena, and feature-ID arrays.
  """
  import numpy as np

  ngeom = _checked(int(model.ngeom), "ngeom")
  old_vertex_words = int(np.asarray(legacy_vertices).size)
  old_info_words = int(np.asarray(legacy_info).size)
  if old_info_words != 9 * ngeom:
    raise ValueError("legacy mesh info must retain its 9-word geom prefix")
  padded_old_words = old_vertex_words + (-old_vertex_words) % 3
  source_vertex_words = int(np.asarray(model.mesh_vert).size)
  graph_words = int(np.asarray(model.mesh_graph).size)
  table_words = sum(int(np.asarray(getattr(model, name)).size) for name in (
      "mesh_polyvertadr", "mesh_polyvertnum", "mesh_polyvert",
      "mesh_polymapadr", "mesh_polymapnum", "mesh_polymap"))
  normal_words = int(np.asarray(model.mesh_polynormal).size)
  hfield_count = sum(int(model.geom_type[g]) == _hfield_type()
                     for g in range(ngeom))
  # Four source-double HField dimensions each occupy three float words.
  static_hull_words = (padded_old_words + source_vertex_words
                       + 12 * hfield_count + normal_words + 3)
  static_info_words = (old_info_words + _COMMON_HEADER_WORDS + 9 * ngeom
                       + graph_words + table_words)
  work = production_ccd_layout(batch_size, pair_count,
                               int(model.opt.ccd_iterations))
  queries = work["queries"]
  output_end = static_hull_words + queries * work["output_words_per_query"]
  output_alignment = output_end & 1
  scratch_end = (output_end + output_alignment
                 + queries * work["arena_words_per_query"])
  dynamic_vertex_base = (scratch_end + 2) // 3
  hull_words = (3 * dynamic_vertex_base
                + queries * work["prism_vertex_words_per_query"]
                + work["diagnostic_words"])
  vertex_id_base = (static_info_words + 1) & ~1
  info_words = (vertex_id_base
                + 2 * queries * work["vertex_capacity"])
  for name, value in (("common hull words", hull_words),
                      ("common info words", info_words)):
    if value > _INT32_MAX:
      raise ValueError(f"{name} exceeds signed-32-bit addressing")
  return 4 * (hull_words + info_words)


def build_common_ccd_model_upload(model, legacy_vertices, legacy_info):
  """Append pinned compiled mesh graph/vertices and HField source sizes.

  The first ``legacy_*`` regions remain byte-for-byte unchanged.  The new
  suffix has a 15-word header, one 9-word per-geom descriptor, followed by the
  original model.mesh_graph words.  Dynamic query output/arena offsets are
  filled into a private device copy before each workspace allocation.
  """
  import numpy as np

  ngeom = int(model.ngeom)
  old_info = np.asarray(legacy_info, dtype=np.int32).reshape(-1)
  if old_info.size != 9 * ngeom:
    raise ValueError("legacy mesh info must retain its 9-word geom prefix")
  old_vertices = np.asarray(legacy_vertices, dtype=np.float32).reshape(-1)
  padded = np.pad(old_vertices, (0, (-old_vertices.size) % 3))
  vertex_base_float = int(padded.size)
  source_vertices = np.asarray(model.mesh_vert, dtype=np.float32).reshape(-1)
  if source_vertices.size % 3:
    raise ValueError("compiled mesh vertex storage is not xyz aligned")
  mesh_hull = np.concatenate((padded, source_vertices)).astype(np.float32, copy=False)

  source_graph = np.asarray(model.mesh_graph, dtype=np.int32).reshape(-1)
  header_base = 9 * ngeom
  geom_base = header_base + _COMMON_HEADER_WORDS
  graph_base = geom_base + 9 * ngeom
  info = np.zeros(graph_base + source_graph.size, dtype=np.int32)
  info[:old_info.size] = old_info
  info[header_base:header_base + _COMMON_HEADER_WORDS] = np.asarray([
      _COMMON_MAGIC, geom_base, graph_base, vertex_base_float // 3,
      source_vertices.size // 3, -1, -1, -1, -1,
      int(model.opt.ccd_iterations), -1, -1, 0, 0, 0,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,
  ], dtype=np.int32)
  for geom in range(ngeom):
    rec = geom_base + 9 * geom
    mesh_id = int(model.geom_dataid[geom]) if int(model.geom_type[geom]) == _mesh_type() else -1
    hf_id = int(model.geom_dataid[geom]) if int(model.geom_type[geom]) == _hfield_type() else -1
    if mesh_id >= 0:
      voff = vertex_base_float // 3 + int(model.mesh_vertadr[mesh_id])
      vcount = int(model.mesh_vertnum[mesh_id])
      goff = int(model.mesh_graphadr[mesh_id])
      fcount = int(model.mesh_facenum[mesh_id])
      if _source_uses_mesh_graph(vcount, goff):
        graph_words = 2 + 3 * int(source_graph[goff]) + 6 * fcount
        info[rec:rec + 5] = [voff, vcount, graph_base + goff,
                             int(source_graph[goff]), graph_words]
      else:
        info[rec:rec + 5] = [voff, vcount, -1, 0, 0]
      polyadr = int(model.mesh_polyadr[mesh_id])
      polynum = int(model.mesh_polynum[mesh_id])
      info[rec + 5:rec + 9] = [polyadr, polynum,
                               int(model.mesh_vertadr[mesh_id]), vcount]
    else:
      info[rec:rec + 5] = [-1, 0, -1, 0, 0]
    if hf_id >= 0:
      # High/mid/tail for all four source-double HField dimensions.
      sizes = np.asarray(model.hfield_size[hf_id], dtype=np.float64)
      hi = sizes.astype(np.float32)
      rem = sizes - hi.astype(np.float64)
      mid = rem.astype(np.float32)
      low = (rem - mid.astype(np.float64)).astype(np.float32)
      size_base = int(mesh_hull.size)
      mesh_hull = np.concatenate((mesh_hull, np.stack((hi, mid, low), -1).reshape(-1)))
      nrow, ncol = int(model.hfield_nrow[hf_id]), int(model.hfield_ncol[hf_id])
      data_base = int(old_info[9 * geom + 5])
      info[rec + 5:rec + 9] = [size_base, data_base, nrow, ncol]
  info[graph_base:graph_base + source_graph.size] = source_graph
  # Native multicontact recovers feature polygons from EPA's selected support
  # vertex ids. Keep MuJoCo's compiled polygon, vertex-to-polygon and normal
  # tables as immutable descriptor suffixes, with model-global offsets intact.
  mesh_int_tables = (
      np.asarray(model.mesh_polyvertadr, dtype=np.int32).reshape(-1),
      np.asarray(model.mesh_polyvertnum, dtype=np.int32).reshape(-1),
      np.asarray(model.mesh_polyvert, dtype=np.int32).reshape(-1),
      np.asarray(model.mesh_polymapadr, dtype=np.int32).reshape(-1),
      np.asarray(model.mesh_polymapnum, dtype=np.int32).reshape(-1),
      np.asarray(model.mesh_polymap, dtype=np.int32).reshape(-1),
  )
  table_offsets = []
  cursor = int(info.size)
  for table in mesh_int_tables:
    table_offsets.append(cursor)
    cursor += int(table.size)
    if cursor > _INT32_MAX:
      raise ValueError("compiled mesh feature tables exceed signed-32-bit addressing")
  feature_info = np.empty(cursor, dtype=np.int32)
  feature_info[:info.size] = info
  for offset, table in zip(table_offsets, mesh_int_tables):
    feature_info[offset:offset + table.size] = table
  info = feature_info
  info[header_base + 15] = int(graph_base + source_graph.size)
  info[header_base + 16:header_base + 22] = table_offsets
  normals = np.asarray(model.mesh_polynormal, dtype=np.float32).reshape(-1)
  normal_base = int(mesh_hull.size)
  mesh_hull = np.concatenate((mesh_hull, normals))
  info[header_base + 22] = normal_base
  info[header_base + 23] = int(normals.size // 3)
  tolerance = float(model.opt.ccd_tolerance)
  high = np.float32(tolerance)
  remainder = tolerance - float(high)
  middle = np.float32(remainder)
  tail = np.float32(remainder - float(middle))
  tolerance_base = int(mesh_hull.size)
  mesh_hull = np.concatenate((mesh_hull, np.asarray(
      [high, middle, tail], dtype=np.float32)))
  info[header_base + 10] = tolerance_base
  return (np.ascontiguousarray(mesh_hull), np.ascontiguousarray(info),
          {"header_base": header_base, "geom_base": geom_base,
           "base_info_words": int(info.size),
           "graph_base": graph_base,
           "source_vertex_base": vertex_base_float // 3,
           "source_vertex_count": source_vertices.size // 3,
           "tolerance_base": tolerance_base})


def _mesh_type():
  import mujoco
  return int(mujoco.mjtGeom.mjGEOM_MESH)


def _hfield_type():
  import mujoco
  return int(mujoco.mjtGeom.mjGEOM_HFIELD)


_COMMON_MAGIC = 1128487732
_COMMON_HEADER_WORDS = 33
_MESH_HILLCLIMB_MIN = 10  # pinned mjMESH_HILLCLIMB_MIN, engine_collision_convex.h


def _source_uses_mesh_graph(vertex_count, graph_offset):
  """Match mjc_initCCDObj's exhaustive-vs-graph support selection."""
  return int(graph_offset) >= 0 and int(vertex_count) >= _MESH_HILLCLIMB_MIN


def compiled_mesh_support_record(model, geom_id, *, cached_vertex=-1):
  """Return the exact common-CCD kind-5 record for one compiled mesh geom."""
  import numpy as np

  geom = _checked(geom_id, "geom_id")
  if geom >= int(model.ngeom):
    raise ValueError("geom_id is outside the model")
  if int(model.geom_type[geom]) != int(__import__("mujoco").mjtGeom.mjGEOM_MESH):
    raise ValueError("common compiled-mesh support requires a mesh geom")
  mesh_id = int(model.geom_dataid[geom])
  if mesh_id < 0 or mesh_id >= int(model.nmesh):
    raise ValueError("mesh geom has no valid compiled mesh asset")
  vertex_offset = int(model.mesh_vertadr[mesh_id])
  vertex_count = int(model.mesh_vertnum[mesh_id])
  graph_offset = int(model.mesh_graphadr[mesh_id])
  face_count = int(model.mesh_facenum[mesh_id])
  if vertex_offset < 0 or vertex_count <= 0:
    raise ValueError("compiled mesh has no addressable vertices")
  if _source_uses_mesh_graph(vertex_count, graph_offset):
    graph_vertex_count = int(np.asarray(model.mesh_graph)[graph_offset])
    graph_face_count = int(np.asarray(model.mesh_graph)[graph_offset + 1])
    if graph_vertex_count <= 0 or graph_vertex_count > vertex_count:
      raise ValueError("compiled mesh graph has an invalid vertex count")
    if graph_face_count != face_count:
      raise ValueError("compiled mesh graph face count disagrees with model")
    graph_words = 2 + 3 * graph_vertex_count + 6 * graph_face_count
    if graph_offset + graph_words > len(model.mesh_graph):
      raise ValueError("compiled mesh graph range exceeds source storage")
  else:
    graph_offset = -1
    graph_vertex_count = 0
    graph_words = 0
  cache = _checked(cached_vertex, "cached_vertex", minimum=-1)
  if cache >= vertex_count:
    raise ValueError("cached mesh support vertex is out of range")
  record = np.asarray([
      5, geom, int(model.geom_type[geom]), vertex_offset, vertex_count,
      cache, graph_offset, graph_vertex_count, graph_words,
  ], dtype=np.int32)
  return np.ascontiguousarray(record)


def source_hfield_candidate_cells(*, nrow, ncol, size0, size1,
                                  xmin, xmax, ymin, ymax):
  """Pinned 3.10 source-order HField subgrid cells before prism tests.

  Mirrors `mjc_ConvexHField`'s floor/ceil, clamp and nested loop bounds.  It
  intentionally does not filter by height or penetration; those decisions
  belong to the production device support/witness bridge.
  """
  rows = _checked(nrow, "nrow", minimum=2)
  cols = _checked(ncol, "ncol", minimum=2)
  size0, size1 = float(size0), float(size1)
  xmin, xmax, ymin, ymax = map(float, (xmin, xmax, ymin, ymax))
  if not all(map(__import__("math").isfinite,
                 (size0, size1, xmin, xmax, ymin, ymax))):
    raise ValueError("HField bounds must be finite")
  if size0 <= 0 or size1 <= 0 or xmin > xmax or ymin > ymax:
    raise ValueError("HField bounds or sizes are invalid")
  import math
  cmin = max(0, int(math.floor((xmin + size0) / (2.0 * size0) * (cols - 1))))
  cmax = min(cols - 1, int(math.ceil((xmax + size0) / (2.0 * size0) * (cols - 1))))
  rmin = max(0, int(math.floor((ymin + size1) / (2.0 * size1) * (rows - 1))))
  rmax = min(rows - 1, int(math.ceil((ymax + size1) / (2.0 * size1) * (rows - 1))))
  return tuple((r, c, tri) for r in range(rmin, rmax)
               for c in range(cmin + 1, cmax + 1)
               for tri in (0, 1))


def hfield_prism_support_record(model, geom_id, row, col, triangle, margin,
                                *, vertex_offset):
  """Pack one pinned addPrismVert prism as a common-CCD kind-8 object.

  Returns ``(int_record[9], object_words[84], high_vertices[18])``; the
  residual prism vertices are stored in object-word slots 48:84.
  """
  import numpy as np
  import math

  geom = _checked(geom_id, "geom_id")
  if geom >= int(model.ngeom):
    raise ValueError("geom_id is outside the model")
  import mujoco
  if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
    raise ValueError("HField support requires a heightfield geom")
  hid = int(model.geom_dataid[geom])
  nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
  size = np.asarray(model.hfield_size[hid], dtype=np.float64)
  adr = int(model.hfield_adr[hid])
  data = np.asarray(model.hfield_data[adr:adr + nrow * ncol], dtype=np.float32)
  r, c, tri = (_checked(row, "row"), _checked(col, "col", minimum=1),
               _checked(triangle, "triangle"))
  if r >= nrow - 1 or c >= ncol or tri not in (0, 1):
    raise ValueError("prism coordinates lie outside the HField grid")
  margin = float(margin)
  if not math.isfinite(margin) or margin < 0:
    raise ValueError("margin must be finite and nonnegative")
  dx = (2.0 * float(size[0])) / (ncol - 1)
  dy = (2.0 * float(size[1])) / (nrow - 1)
  # Upstream addPrismVert's C expressions contract on the supported arm64
  # source build. Decimal computes the exact product-plus-add, then Python's
  # float conversion applies one correctly-rounded mjtNum result.
  from decimal import Decimal, localcontext
  def source_fma(a, b, c):
    with localcontext() as context:
      context.prec = 120
      return float(Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                   + Decimal.from_float(float(c)))
  grid = (((r + 1, c - 1), (r, c - 1), (r + 1, c)) if tri == 0 else
          ((r, c - 1), (r + 1, c), (r, c)))
  bottom, top = [], []
  for rr, cc in grid:
    x = source_fma(dx, cc, -float(size[0]))
    y = source_fma(dy, rr, -float(size[1]))
    z = float(data[rr * ncol + cc]) * float(size[2]) + margin
    bottom.append((x, y, -float(size[3])))
    top.append((x, y, z))
  vertices = np.asarray(bottom + top, dtype=np.float64)
  high = vertices.astype(np.float32)
  remainder = vertices - high.astype(np.float64)
  middle = remainder.astype(np.float32)
  tail = (remainder - middle.astype(np.float64)).astype(np.float32)
  words = np.zeros(84, dtype=np.float32)
  for values, start in ((np.zeros(3), 0), (np.eye(3).reshape(-1), 9),
                        (np.zeros(3), 36), (np.zeros(1), 45)):
    value = np.asarray(values, dtype=np.float64)
    hi = value.astype(np.float32)
    rem = value - hi.astype(np.float64)
    mid = rem.astype(np.float32)
    low = (rem - mid.astype(np.float64)).astype(np.float32)
    words[start:start + value.size * 3] = np.stack((hi, mid, low), axis=-1).reshape(-1)
  words[48:84] = np.stack((middle[:, 0], tail[:, 0],
                            middle[:, 1], tail[:, 1],
                            middle[:, 2], tail[:, 2]), axis=-1).reshape(-1)
  record = np.asarray([
      8, -1, int(model.geom_type[geom]), _checked(vertex_offset,
      "vertex_offset"), 6, -1, -1, 0, 0], dtype=np.int32)
  return (record, np.ascontiguousarray(words),
          np.ascontiguousarray(high.reshape(-1)))


def hfield_source_support_index(prism_vertices, direction):
  """Return pinned ``mjc_prism_support``'s strict first-maximum vertex.

  MuJoCo selects bottom vertices for a negative z direction and top vertices
  otherwise, then evaluates source-order ``mju_dot3`` scores in mjtNum
  precision.  The explicit fused steps model the arm64 binary64 evaluation
  order used by the pinned build (rounded y product, then x and z FMA).
  """
  import numpy as np
  from decimal import Decimal, localcontext

  vertices = np.asarray(prism_vertices, dtype=np.float64)
  direction = np.asarray(direction, dtype=np.float64)
  if vertices.shape != (6, 3) or direction.shape != (3,):
    raise ValueError("prism vertices and direction must have shapes (6,3) and (3,)")
  if not np.isfinite(vertices).all() or not np.isfinite(direction).all():
    raise ValueError("prism support inputs must be finite")

  def fused(a, b, c):
    with localcontext() as context:
      context.prec = 120
      return float(Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                   + Decimal.from_float(float(c)))

  start = 0 if direction[2] < 0.0 else 3
  best, best_score = start, None
  for vertex in range(start, start + 3):
    point = vertices[vertex]
    y_product = float(direction[1] * point[1])
    xy = fused(direction[0], point[0], y_product)
    score = fused(direction[2], point[2], xy)
    if best_score is None or score > best_score:
      best, best_score = vertex, score
  return best


def compiled_mesh_source_support_index(vertices, local_direction,
                                       cached_vertex=-1):
  """Return pinned exhaustive ``mjc_meshSupport``'s cached strict maximum."""
  import numpy as np
  from decimal import Decimal, localcontext

  points = np.asarray(vertices, dtype=np.float32)
  direction = np.asarray(local_direction, dtype=np.float64)
  if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] <= 0:
    raise ValueError("compiled vertices must have shape (n,3)")
  if direction.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(direction).all():
    raise ValueError("mesh support direction and vertices must be finite")
  try:
    cached = int(cached_vertex)
  except (TypeError, ValueError, OverflowError) as exc:
    raise TypeError("cached_vertex must be an integer") from exc
  if cached < -1 or cached >= len(points):
    raise ValueError("cached_vertex is outside the compiled vertex range")

  def fused(a, b, c):
    with localcontext() as context:
      context.prec = 120
      return float(Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                   + Decimal.from_float(float(c)))

  def score(index):
    point = points[index].astype(np.float64)
    y_product = float(direction[1] * point[1])
    xy = fused(direction[0], point[0], y_product)
    return fused(direction[2], point[2], xy)

  best = 0 if cached < 0 else cached
  best_score = score(best) if cached >= 0 else -3.402823466e38
  for index in range(len(points)):
    candidate = score(index)
    if candidate > best_score:
      best, best_score = index, candidate
  return best
