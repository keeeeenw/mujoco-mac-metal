"""CPU-side contract checks for model lowering used by the CCD bridge."""

from pathlib import Path

import numpy as np
import mujoco
import pytest

from mujoco_metal.common_ccd_bridge import (
    append_production_workspace,
    compiled_mesh_support_record,
    build_common_ccd_model_upload,
    compiled_mesh_source_support_index,
    hfield_prism_support_record,
    estimate_production_upload_bytes,
    production_ccd_layout,
    source_hfield_candidate_cells,
    uses_pinned_mjc_convex,
    _source_uses_mesh_graph,
)


def _mesh_hfield_model():
  return mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/>
      <asset>
        <hfield name="terrain" size="1 1 .4 .2" nrow="3" ncol="3"
                elevation="0 0 0 0 .5 0 0 0 0"/>
        <mesh name="tet" vertex="-.2 -.2 -.2 .2 -.2 -.2 -.2 .2 -.2
                                   -.2 -.2 .2 .2 .2 .2"
              face="0 2 1 0 1 3 0 3 2 1 2 4 1 4 3 2 3 4"/>
      </asset>
      <worldbody>
        <geom name="terrain_geom" type="hfield" hfield="terrain"/>
        <body name="mesh_body" pos="0 0 .2"><freejoint/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <geom name="mesh_geom" type="mesh" mesh="tet"/>
        </body>
      </worldbody>
    </mujoco>""")


def _convex_pair_model():
  return mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/>
      <asset><mesh name="tet" vertex="-.15 -.15 -.15 .15 -.15 -.15
        -.15 .15 -.15 -.15 -.15 .15"
        face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
      <worldbody>
        <body pos="0 0 .2"><freejoint/>
          <geom type="box" size=".15 .14 .12"/>
        </body>
        <body pos=".1 0 .2"><freejoint/>
          <geom type="mesh" mesh="tet"/>
        </body>
      </worldbody>
    </mujoco>""")


def test_common_ccd_scratch_layout_checks_aggregate_indices_before_allocation():
  layout = production_ccd_layout(3, 2, 35)
  assert layout["queries"] == 6
  assert layout["epa_iterations"] == 35
  assert layout["vertex_capacity"] == 41
  assert layout["face_capacity"] == 216
  assert layout["diagnostic_epa_counts_offset"] == 7328
  assert layout["diagnostic_production_context_offset"] == 7330
  assert layout["diagnostic_production_context_words"] == 232
  assert layout["diagnostic_words"] == 7562
  assert layout["bytes_total_unaligned"] == 4 * (
      6 * (sum(layout["words_per_query"].values()) +
           layout["prism_vertex_words_per_query"])
      + layout["diagnostic_words"])
  assert layout["bytes_total"] == layout["bytes_total_unaligned"] + 12
  assert layout["bytes_total_is_upper_bound"] is True
  assert layout["bytes_static_model_excluded"] is True
  with pytest.raises(ValueError, match="signed-32-bit"):
    production_ccd_layout((1 << 31) - 1, 2, 35)
  with pytest.raises(TypeError, match="not bool"):
    production_ccd_layout(True, 1, 35)


def test_compiled_mesh_adapter_uses_exact_model_graph_offsets_and_counts():
  model = _mesh_hfield_model()
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  record = compiled_mesh_support_record(model, geom)
  mid = int(model.geom_dataid[geom])
  assert record.tolist() == [
      5, geom, int(model.geom_type[geom]), int(model.mesh_vertadr[mid]),
      int(model.mesh_vertnum[mid]), -1, -1, 0, 0,
  ]
  vertices, info, layout = build_common_ccd_model_upload(
      model, np.zeros(15, dtype=np.float32), np.zeros(9 * model.ngeom,
                                                       dtype=np.int32))
  header, geom_info = layout["header_base"], layout["geom_base"] + 9 * geom
  assert int(info[header]) == 1128487732
  assert int(info[geom_info]) == layout["source_vertex_base"] + int(model.mesh_vertadr[mid])
  assert int(info[geom_info + 1]) == record[4]
  assert int(info[geom_info + 2]) == -1
  packed_vertex = int(info[geom_info])
  assert np.array_equal(vertices[3 * packed_vertex:3 * (packed_vertex + record[4])],
                        model.mesh_vert[int(model.mesh_vertadr[mid]):
                                        int(model.mesh_vertadr[mid] + model.mesh_vertnum[mid])].reshape(-1))


def test_mesh_support_strategy_matches_pinned_hillclimb_threshold():
  """Pinned 3.10 uses exhaustive support below ten compiled mesh vertices."""
  assert not _source_uses_mesh_graph(8, 0)
  assert not _source_uses_mesh_graph(9, 0)
  assert _source_uses_mesh_graph(10, 0)
  assert _source_uses_mesh_graph(12, 37)
  assert not _source_uses_mesh_graph(12, -1)

  # At this source tie, an exhaustive strict scan reaches the first maximum
  # from cached vertex 4. A local graph walk can
  # remain at vertex 5, which changes the P3 EPA seed on an eight-vertex mesh.
  box_vertices = np.asarray([
      [-.05, .12, .12], [-.05, -.12, .12], [-.05, .12, -.12],
      [-.05, -.12, -.12], [.05, .12, .12], [.05, -.12, .12],
      [.05, .12, -.12], [.05, -.12, -.12],
  ], dtype=np.float32)
  assert compiled_mesh_source_support_index(
      box_vertices, (0.0, -1.0, 1.0), cached_vertex=4) == 1

  phi = 0.61803398875
  vertices = []
  for a in (-1.0, 1.0):
    for b in (-phi, phi):
      vertices.extend(((0.0, a, b), (a, b, 0.0), (b, 0.0, a)))
  packed = " ".join(str(x) for vertex in vertices for x in vertex)
  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><asset><mesh name="m" vertex="{packed}"/></asset>'
      '<worldbody><body><freejoint/><geom name="m_geom" type="mesh" mesh="m"/>'
      '</body></worldbody></mujoco>')
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "m_geom")
  mesh_id = int(model.geom_dataid[geom])
  assert int(model.mesh_vertnum[mesh_id]) == 12
  record = compiled_mesh_support_record(model, geom)
  assert record[4] == 12 and record[6] == int(model.mesh_graphadr[mesh_id])


def test_production_workspace_appends_aligned_bounded_per_world_arenas():
  model = _mesh_hfield_model()
  old_hull = np.zeros(31, dtype=np.float32)
  old_info = np.zeros(9 * model.ngeom, dtype=np.int32)
  hull, info, layout = append_production_workspace(
      model, old_hull, old_info, batch_size=2, pair_count=3)
  header = layout["header_base"]
  assert len(hull) == layout["total_words"]
  assert layout["dynamic_vertex_base"] * 3 >= (
      layout["scratch_base"] + layout["queries"] * layout["arena_words_per_query"])
  assert layout["scratch_base"] % 2 == 0
  assert layout["arena_words_per_query"] % 2 == 0
  assert layout["output_alignment_words"] in (0, 1)
  assert int(info[header + 5]) == layout["output_base"]
  assert int(info[header + 6]) == layout["output_words_per_query"]
  assert int(info[header + 7]) == layout["scratch_base"]
  assert int(info[header + 8]) == layout["arena_words_per_query"]
  assert int(info[header + 9]) == int(model.opt.ccd_iterations)
  assert int(info[header + 10]) == layout["tolerance_base"]
  assert int(info[header + 11]) == layout["vertex_capacity"]
  assert int(info[header + 12]) == layout["face_capacity"]
  assert int(info[header + 13]) == layout["horizon_capacity"]
  assert int(info[header + 14]) == layout["stack_capacity"]
  assert int(info[header + 24]) == layout["vertex_id_base"]
  assert int(info[header + 25]) == layout["vertex_capacity"]
  assert int(info[header + 26]) == layout["queries"]
  assert int(info[header + 27]) == layout["diagnostic_base"]
  assert int(info[header + 28]) == layout["diagnostic_words"]
  assert int(info[header + 29]) == -1
  assert tuple(map(int, info[header + 30:header + 33])) == (-1, -1, -1)
  assert layout["diagnostic_base"] + layout["diagnostic_words"] == len(hull)
  assert layout["vertex_id_base"] % 2 == 0
  assert layout["vertex_id_words"] == 2 * layout["queries"] * layout["vertex_capacity"]
  assert layout["bytes_total"] <= hull.nbytes + layout["vertex_id_bytes"] + 4 * layout["vertex_id_padding_words"]
  expected_runtime_words = (
      layout["queries"] * (layout["output_words_per_query"]
                           + layout["arena_words_per_query"]
                           + layout["prism_vertex_words_per_query"])
      + layout["diagnostic_words"]
      + layout["output_alignment_words"]
      + layout["alignment_padding_words"])
  assert layout["bytes_total"] == (
      4 * expected_runtime_words + layout["vertex_id_bytes"]
      + 4 * layout["vertex_id_padding_words"])
  assert layout["bytes_total_is_upper_bound"] is False


def test_common_upload_preflight_matches_final_arrays_before_construction():
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  model = _convex_pair_model()
  desc = lower_coupled_constraints(model)
  predicted = estimate_production_upload_bytes(
      model, desc.mesh_hull, desc.mesh_hull_info, batch_size=2,
      pair_count=desc.npairs)
  actual_hull, actual_info, _ = append_production_workspace(
      model, desc.mesh_hull, desc.mesh_hull_info, batch_size=2,
      pair_count=desc.npairs)
  assert predicted == actual_hull.nbytes + actual_info.nbytes


def test_common_upload_budget_rejects_before_workspace_allocator(monkeypatch):
  from mujoco_metal.capacity import CapacityLimits, estimate_capacity
  from mujoco_metal.coupled_constraints import (
      MetalCoupledConstraints, lower_coupled_constraints)
  import mujoco_metal.common_ccd_bridge as bridge

  model = _convex_pair_model()
  desc = lower_coupled_constraints(model)
  base = estimate_capacity(
      model, 1, desc.npairs, desc.ncontacts_max, desc.nr,
      neq=desc.neq, nr_joint=desc.nr_joint, mass_storage="dense",
      jacobian_kind=str(getattr(desc, "jacobian_kind", "dense")),
      jacobian_nnz=(int(desc.jacobian_pattern.nnz)
                    if getattr(desc, "jacobian_pattern", None) is not None
                    else None))
  def forbidden_allocator(*args, **kwargs):
    pytest.fail("common workspace allocation ran before memory admission")

  monkeypatch.setattr(bridge, "append_production_workspace", forbidden_allocator)
  with pytest.raises(ValueError, match="exceeds budget"):
    MetalCoupledConstraints(
        model, batch_size=1,
        limits=CapacityLimits(memory_budget_bytes=base.memory_bytes))


@pytest.mark.parametrize(("iterations", "vertex_words_odd"), [(35, True),
                                                               (36, False)])
def test_epa_typed_horizon_ranges_are_aligned_and_disjoint(
    iterations, vertex_words_odd):
  from mujoco_metal.common_ccd_bridge import production_ccd_layout

  layout = production_ccd_layout(2, 2, iterations)
  words = layout["words_per_query"]
  assert bool(words["vertices"] & 1) is vertex_words_odd
  assert words["horizon_alignment"] == int(vertex_words_odd)
  assert layout["arena_words_per_query"] % 2 == 0
  output_end = 31 + layout["queries"] * layout["output_words_per_query"]
  scratch_base = output_end + (output_end & 1)
  for query in range(layout["queries"]):
    arena = scratch_base + query * layout["arena_words_per_query"]
    horizon = (arena + words["vertices"] + words["faces"] +
               words["face_map"] + words["horizon_alignment"])
    stack = horizon + words["horizon"]
    arena_end = arena + layout["arena_words_per_query"]
    assert arena % 2 == 0
    assert horizon % 2 == 0
    assert stack <= arena_end
    assert stack + words["horizon_stack"] == arena_end


def test_heightfield_adapter_matches_source_cell_order_and_retains_prism_residuals():
  model = _mesh_hfield_model()
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
  hid = int(model.geom_dataid[geom])
  cells = source_hfield_candidate_cells(
      nrow=int(model.hfield_nrow[hid]), ncol=int(model.hfield_ncol[hid]),
      size0=float(model.hfield_size[hid, 0]),
      size1=float(model.hfield_size[hid, 1]),
      xmin=-.9, xmax=.9, ymin=-.9, ymax=.9)
  assert cells == tuple((r, c, tri) for r in range(2)
                        for c in range(1, 3) for tri in (0, 1))
  record, words, high = hfield_prism_support_record(
      model, geom, row=0, col=1, triangle=0, margin=.013,
      vertex_offset=123)
  assert record.tolist() == [8, -1, int(model.geom_type[geom]), 123, 6, -1, -1, 0, 0]
  assert words.shape == (84,) and high.shape == (18,)
  assert np.isfinite(words).all() and np.isfinite(high).all()
  assert np.any(words[48:84] != 0), "double-sized HField prism lost its residual"
  assert high[2] == np.float32(-float(model.hfield_size[hid, 3]))
  assert high[17] > high[2]


def test_pinned_310_hfield_mesh_source_produces_multiple_ordered_witnesses():
  model = _mesh_hfield_model()
  data = mujoco.MjData(model)
  data.qpos[2] = .2
  mujoco.mj_forward(model, data)
  assert data.ncon >= 2
  distances = np.asarray([data.contact[i].dist for i in range(data.ncon)])
  positions = np.asarray([data.contact[i].pos for i in range(data.ncon)])
  assert np.all(np.isfinite(distances)) and np.all(np.isfinite(positions))
  assert np.any(distances < 0), "fixture must exercise actual penetration witnesses"
  assert np.unique(np.round(positions, decimals=9), axis=0).shape[0] >= 2


def test_pinned_310_convex_dispatch_classifier_matches_exact_callback_table():
  expected = {
      (2, 4), (2, 7),
      (3, 4), (3, 5), (3, 7),
      (4, 4), (4, 5), (4, 6), (4, 7),
      (5, 5), (5, 6), (5, 7),
      (6, 7), (7, 7),
  }
  actual = {(a, b) for a in range(9) for b in range(a, 9)
            if uses_pinned_mjc_convex(a, b)}
  assert actual == expected
  for dedicated in ((0, 2), (0, 7), (1, 7), (1, 8), (2, 3),
                    (2, 5), (2, 6), (3, 6), (6, 6), (7, 8), (8, 8)):
    assert not uses_pinned_mjc_convex(*dedicated)
  with pytest.raises(ValueError, match="range"):
    uses_pinned_mjc_convex(9, 7)
  with pytest.raises(TypeError, match="not bool"):
    uses_pinned_mjc_convex(True, 7)


def test_common_upload_retains_compiled_mesh_feature_tables_for_multicontact():
  model = _mesh_hfield_model()
  hull, info, layout = build_common_ccd_model_upload(
      model, np.zeros(15, dtype=np.float32),
      np.zeros(9 * model.ngeom, dtype=np.int32))
  h = int(layout["header_base"])
  mesh_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                "mesh_geom")
  mesh_id = int(model.geom_dataid[mesh_geom])
  rec = int(layout["geom_base"]) + 9 * mesh_geom
  assert info[rec + 5:rec + 9].tolist() == [
      int(model.mesh_polyadr[mesh_id]), int(model.mesh_polynum[mesh_id]),
      int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])]
  for header_index, source_name in zip(range(16, 22), (
      "mesh_polyvertadr", "mesh_polyvertnum", "mesh_polyvert",
      "mesh_polymapadr", "mesh_polymapnum", "mesh_polymap")):
    offset = int(info[h + header_index])
    source = np.asarray(getattr(model, source_name), dtype=np.int32).reshape(-1)
    assert np.array_equal(info[offset:offset + source.size], source)
  normal_offset = int(info[h + 22])
  normals = np.asarray(model.mesh_polynormal, dtype=np.float32).reshape(-1)
  assert int(info[h + 23]) == normals.size // 3
  assert np.array_equal(hull[normal_offset:normal_offset + normals.size], normals)


def test_mesh_feature_ids_resolve_source_ordered_faces_and_shared_edges():
  model = _mesh_hfield_model()
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mesh_geom")
  mesh = int(model.geom_dataid[geom])
  vertadr = int(model.mesh_vertadr[mesh])
  polyadr = int(model.mesh_polyadr[mesh])

  def incident_faces(local_vertex):
    global_vertex = vertadr + local_vertex
    begin = int(model.mesh_polymapadr[global_vertex])
    count = int(model.mesh_polymapnum[global_vertex])
    return list(np.asarray(model.mesh_polymap[begin:begin + count],
                           dtype=np.int32))

  face_sets = []
  for local_face in range(int(model.mesh_polynum[mesh])):
    global_face = polyadr + local_face
    begin = int(model.mesh_polyvertadr[global_face])
    count = int(model.mesh_polyvertnum[global_face])
    vertices = np.asarray(model.mesh_polyvert[begin:begin + count],
                          dtype=np.int32)
    common = set(incident_faces(int(vertices[0])))
    for vertex in vertices[1:]:
      common.intersection_update(incident_faces(int(vertex)))
    assert local_face in common, (local_face, vertices.tolist(), sorted(common))
    face_sets.append((vertices, common))

  # Every tested source face can be recovered by the exact address convention
  # used by mjc_initCCDObj: vertex maps stay globally addressed, while map
  # values and the polynormal/polyvert face indexing are mesh-local.
  assert face_sets
  assert np.asarray(model.mesh_polynormal).reshape(-1, 3).shape[0] >= (
      polyadr + int(model.mesh_polynum[mesh]))
  edge = tuple(sorted(set(map(int, face_sets[0][0][:2]))))
  shared = set(incident_faces(edge[0])).intersection(incident_faces(edge[1]))
  assert len(shared) >= 1
  assert all(0 <= face < int(model.mesh_polynum[mesh]) for face in shared)


def test_hfield_p3_centroid_uses_pinned_binary64_reciprocal():
  """P3 EPA's source centroid uses mjtNum 1/3, not a float32 reciprocal."""
  root = Path(__file__).resolve().parents[1]
  shader = (root / "mujoco_metal" / "shaders" /
            "common_native_ccd_support.metal").read_text()
  assert "0.3333333432674408f,-9.934107758624577e-9f,2.7755575615628914e-16f" in shader
  source = 1.0 / 3.0
  high = float(np.float32(source))
  low = float(np.float32(source - high))
  tail = float(np.float32(source - high - low))
  represented = float(np.float64(high) + np.float64(low) + np.float64(tail))
  assert represented == source
  assert float(np.float32(1.0 / 3.0)) != source


def test_common_ccd_failed_world_mask_is_folded_before_candidate_compaction():
  """Failure status gates pair/slot maps before contact-row consumers run."""
  root = Path(__file__).resolve().parents[1]
  shader = (root / "mujoco_metal" / "shaders" /
            "coupled_constraints.metal").read_text()
  host = (root / "mujoco_metal" / "coupled_constraints.py").read_text()
  assert "mask_failed_candidate_worlds" in shader
  assert "world_status[int(world)] == 0" in shader
  prepass = host.index("self._common_ccd_hfield_mesh_kernel(")
  fold = host.index("self._mask_failed_candidate_worlds_kernel(", prepass)
  remap = host.index("pair_maps = self._pair_compaction.run(", fold)
  contact = host.index("self._contact_kernel(", remap)
  assert prepass < fold < remap < contact
