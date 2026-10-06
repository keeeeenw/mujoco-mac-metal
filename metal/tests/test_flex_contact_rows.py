# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned MuJoCo contact-row source checks and native row assembly gates."""

import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    _FLEX_CCD_ITER_TRACE_BASE, _FLEX_CCD_ITER_TRACE_DISTANCE_WORDS,
    _FLEX_CCD_ITER_TRACE_HEADER, _FLEX_CCD_ITER_TRACE_HORIZON,
    _FLEX_CCD_ITER_TRACE_PROJECTION_WORDS, _FLEX_CCD_ITER_TRACE_VERTEX_WORDS,
    _FLEX_CCD_TRACE_SIZE, _FLEX_GJK_TRACE_STAGE_COUNT,
    _FLEX_GJK_TRACE_STRIDE, _FLEX_GJK_TRACE_VALID_WORD,
    _FLEX_DETECT_DIMS_FIXED_WORDS,
    _flex_ccd_trace_capacity,
    _flex_combined_trace_capacity,
    _flex_gjk_trace_capacity,
    _flex_trace_slot,
    _flex_ccd_workspace_capacity,
    _KIND_ELEMENT_PAIR, _KIND_GEOM_ELEMENT,
    _KIND_PLANE_VERTEX,
    _flex_bvh_element_order,
    _flex_element_body_weights, _flex_vertex_body_weights,
    FlexContactProgram, lower_flex_contacts)
from mujoco_metal.flex import MetalFlex


def test_optional_gjk_trace_record_layout_is_bounded_and_nonoverlapping():
  """Pin the opt-in per-iteration GJK record consumed by failure probes."""
  fields = (
      ("iteration/simplex-count/branch", 0, 3),
      ("x", 3, 12),
      ("support-direction", 12, 21),
      ("support-vertex-a-b-m", 21, 48),
      ("x-norm2", 48, 51),
      ("gap", 51, 54),
      ("separating", 54, 57),
      ("support-id/flags", 60, 64),
      ("pre-simplex", 64, 172),
      ("weights", 172, 184),
      ("vertex-projections", 184, 220),
      ("next-x", 220, 229),
      ("nested-result/iteration/post-count", 233, 236),
      ("post-simplex", 236, 344),
      ("reduction-counts", 344, 346),
      ("record-kind/valid", 350, 352),
  )
  occupied = []
  for _name, start, stop in fields:
    occupied.extend(range(start, stop))
  assert len(occupied) == len(set(occupied))
  assert min(occupied) == 0
  assert max(occupied) == _FLEX_GJK_TRACE_STRIDE - 1
  assert _FLEX_GJK_TRACE_STAGE_COUNT == 4
  iters, stride, total = _flex_gjk_trace_capacity(35)
  assert (iters, stride, total) == (35, 352, 4 * 35 * 352)
  with pytest.raises(ValueError, match="256 MiB"):
    _flex_gjk_trace_capacity(200_000)


def test_gjk_trace_target_and_combined_capacity_reject_lossy_values():
  assert _flex_trace_slot((np.int32(2), np.int64(7))) == (2, 7)
  for target in ((True, 1), (0, False), (0.5, 1), (0, 1.0), ("0", 1)):
    with pytest.raises(TypeError, match="integer"):
      _flex_trace_slot(target)
  with pytest.raises(ValueError, match="int32"):
    _flex_combined_trace_capacity(1, 1, np.iinfo(np.int32).max, 1)
  with pytest.raises(ValueError, match="256 MiB"):
    _flex_combined_trace_capacity(1, 1, 0, 67_108_865)


def test_flex_program_rejects_lossy_trace_targets_before_device_setup():
  model, _data, descriptor = _dynamic_diag_fixture(
      "trilinear", False, False, geom_margin_gap=0.0,
      disable_midphase=False)
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == 0))
  assert slots.size
  slot = int(slots[0])
  for target in ((False, slot), (0.0, slot), (0, float(slot))):
    with pytest.raises(TypeError, match="integer"):
      FlexContactProgram(model, batch_size=1, device="not-a-device",
                         capture_gjk_trace_slot=target)


def test_optional_gjk_trace_constructor_allocates_one_candidate_only():
  """The diagnostic trace is slot-targeted and opt-in at construction."""
  torch = pytest.importorskip("torch")
  model, _data, descriptor = _dynamic_diag_fixture(
      "trilinear", False, False, geom_margin_gap=0.0,
      disable_midphase=False)
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == 0))
  assert slots.size > 0
  target = (0, int(slots[0]))
  ordinary = FlexContactProgram(model, batch_size=2, device="cpu")
  traced = FlexContactProgram(model, batch_size=2, device="cpu",
                              capture_gjk_trace_slot=target)
  assert ordinary._gjk_trace.shape == (1,)
  assert traced._gjk_trace.shape == (
      _FLEX_GJK_TRACE_STAGE_COUNT, int(model.opt.ccd_iterations),
      _FLEX_GJK_TRACE_STRIDE)
  gjk_words = _flex_gjk_trace_capacity(model.opt.ccd_iterations)[2]
  ccd_words = (2 * descriptor.slot_count * traced._ccd_trace_size)
  assert traced._ccd_trace.numel() == ccd_words
  assert traced._ccd_trace_storage.numel() == ccd_words + gjk_words
  assert traced._gjk_trace.storage_offset() == ccd_words
  assert traced._gjk_trace_dims.tolist() == [
      2, int(model.nflexvert), int(model.ngeom), target[0], target[1],
      int(model.opt.ccd_iterations), _FLEX_GJK_TRACE_STRIDE]
  assert traced.capture_ccd_trace


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="GJK iteration trace requires native opt-in")
@pytest.mark.parametrize("elem_id", (39, 44), ids=("elem39", "elem44"))
def test_native_flex_gjk_iteration_trace_records_target_candidate_only(elem_id):
  """Capture actual point/full GJK iterations for one compiled tetra slot."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("GJK iteration trace requires MPS")
  model, first, descriptor = _dynamic_diag_fixture(
      "trilinear", False, False, geom_margin_gap=0.0,
      disable_midphase=False)
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == 0)
                         & (descriptor.elem1 == elem_id))
  assert slots.size == 1
  slot = int(slots[0])
  assert np.count_nonzero(np.asarray(descriptor.nodes1[slot]) >= 0) == 4
  program = FlexContactProgram(
      model, batch_size=1, device="mps",
      capture_gjk_trace_slot=(0, slot))
  assert not program._narrowphase_admitted
  # Isolated diagnostic-stage execution. Public admission remains unchanged.
  program._narrowphase_admitted = True
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  result = program.run_device(
      mps(first.flexvert_xpos[None]), mps(first.geom_xpos[None]),
      mps(_geom_quat_batch(model, [first])))
  records = result["gjk_trace"].cpu().numpy()
  assert records.shape == (
      4, int(model.opt.ccd_iterations), _FLEX_GJK_TRACE_STRIDE)
  assert np.any(records[0, :, _FLEX_GJK_TRACE_VALID_WORD] == 1.0)
  assert np.any(records[2, :, _FLEX_GJK_TRACE_VALID_WORD] == 1.0)
  for stage in (0, 2):
    valid = records[stage, :, _FLEX_GJK_TRACE_VALID_WORD] == 1.0
    support_ids = records[stage, valid, 60].astype(np.int32)
    assert np.all((support_ids >= 0) & (support_ids < 4))
  assert result["ccd_trace"].shape[:2] == (1, descriptor.slot_count)


def test_native_flex_row_adapter_forwards_packed_global_jacobian():
  """The flex manager preserves the coupled solver's packed-J row target."""
  sentinel = object()
  captured = {}
  names = (
      "R", "aref", "lo", "hi", "row_active", "position_context",
      "surface_velocity", "row_owner", "row_local", "row_cone",
      "row_friction", "row_start", "row_span", "jacobian_packed",
      "canonical_row_offset")
  result = {name: sentinel for name in names}
  result["canonical_row_offset"] = 37
  result["active_tree_links"] = sentinel
  result["active_tree_link_overflow"] = sentinel

  class Program:
    sparse_rows = True

    def run_device(self, *args, **kwargs):
      captured["args"] = args
      captured["kwargs"] = kwargs
      return result

  class Manager:
    _device = SimpleNamespace(type="mps")
    _flexvert_xpos = sentinel
    _flexvert_xpos_low = sentinel
    _flexvert_xpos_tail = sentinel
    _flexvert_spatial_J = sentinel
    _contact_program = Program()

    def _validate_world_mask(self, world_mask):
      captured["validated_mask"] = world_mask

    def update_kinematics(self, poses, cvel, *, world_mask=None):
      captured["updated"] = (poses, cvel)
      captured["update_mask"] = world_mask

  poses = {name: sentinel for name in ("geom_pos", "geom_quat", "cdof", "root_com")}
  packed = sentinel
  bundle = MetalFlex.run_native_contact_rows(
      Manager(), poses, sentinel, packed_jacobian=packed,
      canonical_row_offset=37, include_wake_links=True)

  assert captured["kwargs"]["packed_jacobian"] is packed
  assert captured["kwargs"]["flexvert_xpos_low"] is sentinel
  assert captured["kwargs"]["canonical_row_offset"] == 37
  assert captured["kwargs"]["include_wake_links"] is True
  assert "workspace_J" not in bundle["rows"]
  assert bundle["rows"]["jacobian_packed"] is packed
  assert bundle["rows"]["canonical_row_offset"] == 37
  for name in names:
    if name == "row_active":
      assert bundle["rows"]["active"] is sentinel
    elif name not in ("jacobian_packed", "canonical_row_offset"):
      assert bundle["rows"][name] is sentinel


def test_sparse_flex_contact_workspace_omits_dense_candidate_j_buffers():
  """Sparse row production keeps only the packed-J source workspace."""
  torch = pytest.importorskip("torch")
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom type="plane" size="0 0 .1"/>
      <flexcomp name="cloth" type="grid" count="4 4 1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  sparse = FlexContactProgram(model, batch_size=2, device="cpu",
                              sparse_rows=True)
  dense = FlexContactProgram(model, batch_size=2, device="cpu",
                             sparse_rows=False)
  for name in ("_contact_side1_jacobian", "_contact_side2_jacobian",
               "_contact_relative_jacobian", "_contact_spatial_side1_jacobian",
               "_contact_spatial_side2_jacobian",
               "_contact_spatial_relative_jacobian",
               "_geom_contact_spatial_jacobian"):
    assert getattr(sparse, name).numel() == 1
    assert getattr(dense, name).numel() > 1
  assert sparse.sparse_rows and not dense.sparse_rows
  assert _FLEX_DETECT_DIMS_FIXED_WORDS == 23
  for program in (sparse, dense):
    assert tuple(program._detect_dims.shape) == (23 + 2,)
    np.testing.assert_array_equal(
        program._detect_dims[23:].cpu().numpy(), np.ones(2, np.int32))


def test_flex_contact_recovery_mask_is_after_fixed_header_and_guards_reads():
  """The per-world mask cannot overlap existing detector dimensions."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("kernel void flex_contact_detect(")
  stop = shader.index("kernel void flex_contact_detect_native_ccd(", start)
  detect = shader[start:stop]
  assert "if (dims[23+env] == 0) return;" in detect
  assert detect.index("if (dims[23+env] == 0) return;") < detect.index(
      "candidate_meta[10*slot+0]")
  start = shader.index("kernel void flex_contact_detect_native_ccd(")
  stop = shader.index("kernel void flex_contact_select(", start)
  ccd = shader[start:stop]
  assert "if (dims[23+env] == 0) return;" in ccd
  assert ccd.index("if (dims[23+env] == 0) return;") < ccd.index(
      "candidate_meta[10*slot+0]")


def test_optional_ccd_trace_index_map_has_no_overlapping_fields():
  """Pin the opt-in EPA trace layout independently of a Metal dispatch."""
  fields = (
      ("legacy", 0, 64),
      ("iteration", 64, 65),
      ("best_face", 65, 66),
      ("support_minkowski", 66, 69),
      ("max_horizon_stack_depth", 69, 70),
      ("horizon_count", 70, 71),
      ("reserved", 71, 73),
      ("root_face_edge_visibility_delta", 73, 85),
      ("ordered_horizon_face_edge_pairs", 85, 133),
      ("reserved", 133, 136),
      ("captured_expansion_header", 136, 142),
      ("captured_expansion_face_map_prefix", 142, 174),
      ("captured_face_and_neighbors", 174, 206),
      ("captured_adjacency_invalid", 206, 207),
      ("p3_support_direction_index_and_vertices", 207, 222),
      ("p3_gjk_seed_minkowski_dd_parts", 222, 240),
      ("p3_gjk_seed_minkowski_tail_parts", 240, 249),
      ("p3_unit_normal_tail_parts", 249, 252),
  )
  indices = [index for _, start, stop in fields
             for index in range(start, stop)]
  assert _FLEX_CCD_TRACE_SIZE == 252
  assert indices == list(range(_FLEX_CCD_TRACE_SIZE))
  # The face map dump is bounded to 32 entries; the header reports how many
  # entries are populated. Four 8-word records follow for best+three neighbors.
  assert fields[11][2] - fields[11][1] == 32
  assert fields[12][2] - fields[12][1] == 4 * 8
  assert fields[-3][2] - fields[-3][1] == 18
  assert fields[-2][2] - fields[-2][1] == 9
  assert fields[-1][2] - fields[-1][1] == 3


def test_quantized_pinned_ccd_seed_retains_generic_low_part_for_p3():
  """Keep a tiny, source-derived P3 residual through float32 quantization.

  This seed was returned by pinned 3.10 ``mjc_ccd`` after flex vertices,
  geom position, and element AABB operands were explicitly rounded to the
  exact float32 values consumed by the Metal path.  Thus the nonzero residual
  is created by double-precision support/simplex arithmetic, not by extra
  precision in the model/data inputs.  A geometry-specific support tie rule
  would hide the arithmetic loss this regression records.
  """
  seed = np.asarray((
      (-0.04907477288111818, -0.04907477288111818, 0.049074772881118195),
      (0.14907477437123431, 0.049074772881118195, -0.14907477437123431),
      (-0.01986812463783179, 0.1802216646912189, 0.019868124637831747),
  ), dtype=np.float64)
  source_cross = np.cross(seed[1] - seed[0], seed[2] - seed[0])
  assert source_cross[1] != 0.0
  assert abs(source_cross[1]) == pytest.approx(1.1275702593849246e-17,
                                                rel=2e-2, abs=0.0)

  # Three float32 words reconstruct the source binary64 vector after the
  # same high conversion. Ordinary float32 subtraction/cross loses Y.
  high = seed.astype(np.float32)
  low = (seed - high.astype(np.float64)).astype(np.float32)
  split_seed = high.astype(np.float64) + low.astype(np.float64)
  split_cross = np.cross(split_seed[1] - split_seed[0],
                         split_seed[2] - split_seed[0])
  float_cross = np.cross(high[1] - high[0], high[2] - high[0])
  assert split_cross[1] != 0.0
  assert float_cross[1] == 0.0

  # The residual changes a generic strict support projection: no index or
  # contact-specific tie override is used in this comparison.
  direction = split_cross / np.linalg.norm(split_cross)
  direction_hi = direction.astype(np.float32)
  direction_lo = (direction - direction_hi.astype(np.float64)).astype(np.float32)
  ordinary_direction = float_cross / np.linalg.norm(float_cross)
  candidate0 = np.asarray((0.0, 0.0, 0.0), np.float32)
  candidate1 = np.asarray((0.0, 0.1, 0.0), np.float32)
  ordinary_delta = np.dot(candidate1, ordinary_direction) - np.dot(
      candidate0, ordinary_direction)
  split_delta = (np.dot(candidate1.astype(np.float64),
                        direction_hi.astype(np.float64)
                        + direction_lo.astype(np.float64))
                 - np.dot(candidate0.astype(np.float64),
                          direction_hi.astype(np.float64)
                          + direction_lo.astype(np.float64)))
  assert ordinary_delta == 0.0
  assert split_delta > 0.0


def test_ccd_quotient_refinement_preserves_unrounded_residual_digits():
  """Keep quotient correction terms until the final binary64 rounding.

  Flex CCD stores a source-rounded binary64 scalar as three float32 words.
  This example is a deterministic counterexample to refining q1/q2 from
  already-rounded ``b*q0`` and ``remainder-b*q1`` products: that loses the
  final quotient word even though each input word is exactly representable.
  The shader must use the exact product-error expansion before rounding the
  residual that generates each correction digit.
  """
  numerator = np.asarray((
      -0.28128743, -6.6804633e-9, -1.0551505e-15), dtype=np.float32)
  denominator = np.asarray((
      -0.39080098, 4.8194537e-9, -2.385536e-16), dtype=np.float32)
  with localcontext() as context:
    context.prec = 200
    decimal = lambda x: Decimal.from_float(float(np.float32(x)))
    a = sum((decimal(x) for x in numerator), Decimal(0))
    b = sum((decimal(x) for x in denominator), Decimal(0))
    q0 = np.float32(float(decimal(numerator[0]) / decimal(denominator[0])))

    def residual_words(value):
      # `flex_expansion_round53` returns the residual rounded to binary64 and
      # represents that value as three exact float32 words.
      rounded = float(value)
      high = np.float32(rounded)
      low = np.float32(rounded - float(high))
      tail = np.float32(rounded - float(high) - float(low))
      return high, low, tail

    residual = a - b * Decimal.from_float(float(q0))
    high, low, tail = residual_words(residual)
    q1_numerator = np.float32(np.float32(high + low) + tail)
    q1 = np.float32(q1_numerator / denominator[0])
    residual = (a - b * Decimal.from_float(float(q0))
                - b * Decimal.from_float(float(q1)))
    high, low, tail = residual_words(residual)
    q2_numerator = np.float32(np.float32(high + low) + tail)
    q2 = np.float32(q2_numerator / denominator[0])
    refined = float(sum((Decimal.from_float(float(x))
                         for x in (q0, q1, q2)), Decimal(0)))
    exact = float(a / b)
  assert refined == exact == 0.7197715827453792

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  division = shader.split(
      "static inline FlexDD flex_dd_source_div(FlexDD a, FlexDD b) {", 1)[1]
  division = division.split("static inline FlexDD flex_dd_source_sqrt", 1)[0]
  assert "flex_dd_division_residual(a,b,q0,0.0f)" in division
  assert "flex_dd_division_residual(a,b,q0,q1)" in division


def test_absolute_support_and_exact_aabb_midpoint_preserve_float_input_low_part():
  """The center/support boundary retains bits present in float32 inputs."""
  import numpy as np

  # This pair is representative of flex element coordinates in the strict
  # CCD witness: each endpoint is an exact float input, but their double
  # midpoint is not representable as float32. Computing support from
  # (vertex - float_center) and adding the center back loses that distinction.
  lo = np.float32(-0.1)
  hi = np.float32(0.4)
  midpoint_exact = (float(lo) + float(hi)) * 0.5
  midpoint_rounded = np.float32((lo + hi) * np.float32(0.5))
  assert midpoint_exact != float(midpoint_rounded)
  assert abs(midpoint_exact - float(midpoint_rounded)) > 0.0

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  support = shader[shader.index("static inline FlexDDVertex flex_sphere_flex_support_dd("):
                   shader.index("static inline FlexDD flex_dd_abs(FlexDD value) {")]
  assert "flex_mju_dot3_projection53(" in support
  assert "flex_projection53_compare(projection,best_projection)>0" in support
  assert "flex_dd3_sub(flex_world_vertices" not in support
  center = shader[shader.index("static inline FlexDD3 flex_center_precise("):
                  shader.index("static inline float3x3 flex_identity(")]
  assert "flex_dd3_add(flex_dd3(lo),flex_dd3(hi))" in center
  assert "flex_dd(0.5f)" in center


def test_sphere_flex_ccd_entry_uses_source_simplex_and_p3_path():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  source_gjk = shader[shader.index("static inline FlexDD flex_source_gjk_dd("):
                      shader.index("// Source-order polytope3 initialization", )]
  assert "FlexDDVertex simplex[4];" in source_gjk
  assert "flex_dd_subdistance(weights,n,simplex);" in source_gjk
  assert "flex_projection53_compare(projection,best_projection)>0" in shader
  # Support must project the absolute current world vertices. Subtracting the
  # rounded AABB center first erases low parts which affect pinned strict ties.
  assert "thread const float3* flex_world_vertices" in source_gjk
  assert "thread const FlexDD3* flex_world_vertices_dd" in source_gjk
  assert "flex_world_vertices_dd[i],flex_direction" in shader
  assert "FlexDD3 flex_center_exact" in source_gjk
  assert "flex_dd3_cross(diff1,diff2)" in shader
  core_start = shader.index("static inline int flex_sphere_tetra_epa_core(")
  core_stop = shader.index("static inline int flex_sphere_tetra_epa(", core_start)
  core = shader[core_start:core_stop]
  assert core.count("flex_source_gjk_dd(") == 2
  assert "precise_seed,trace);" in core


def test_sphere_flex_epa_initialization_uses_pinned_mjtnum_predicates():
  """The installed double build runs P3/P4 containment in mjtNum precision."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  init = shader[shader.index("static inline int flex_epa_init3("):
                shader.index("// Returns 0 on convergence",)]
  assert "flex_tri_point_intersect(seed[0].m,seed[1].m,seed[2].m,v4.m)" in init
  assert "flex_tri_point_intersect(seed[0].m,seed[1].m,seed[2].m,v5.m)" in init
  assert "flex_test_tetra(seed[0].m,seed[1].m,seed[2].m,v4.m)" in init
  assert "flex_test_tetra_dd(" not in init
  core = shader[shader.index("static inline int flex_sphere_tetra_epa_core("):
                shader.index("static inline int flex_sphere_tetra_epa(",
                             shader.index("static inline int flex_sphere_tetra_epa_core("))]
  assert "flex_test_tetra(vertices[0].m,vertices[1].m,vertices[2].m,vertices[3].m)" in core
  assert "flex_ray_triangle(seed[0].m,seed[1].m,vertices[2].m," in core
  assert "flex_test_tetra_dd(" not in core
  assert "flex_ray_triangle_dd(" not in core


def test_p2_epa_initialization_uses_precise_seed_and_pinned_rotation_route():
  """P2 axis selection/rotation must use the source simplex, not float mirrors.

  A near tie in |diff.y| and |diff.z| is enough to choose a different basis
  axis if the low word is discarded.  The exact polytope2 source then builds
  its 120-degree matrix and applies it twice through mju_mulMatVec3; replacing
  that route by a simplified cross-product expression changes support queries.
  """
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  core_start = shader.index("static inline int flex_sphere_tetra_epa_core(")
  core_stop = shader.index("static inline int flex_sphere_tetra_epa(", core_start)
  core = shader[core_start:core_stop]
  p2 = core[core.index("} else if (nseed==2) {"):]
  p2 = p2[:p2.index("  } else {")]
  assert "flex_epa_p2_directions_source(precise_seed[0].m,precise_seed[1].m,dirs)" in p2
  assert "flex_dd3(dirs[i])" not in p2

  helper_start = shader.index("static inline void flex_epa_p2_directions_source(")
  helper_stop = shader.index("static inline FlexDD3 flex_dd3_madd(", helper_start)
  helper = shader[helper_start:helper_stop]
  assert "FlexDD3 diff=flex_dd3_source_sub(b,a)" in helper
  assert "flex_dd_compare(ay,least)<0" in helper
  assert "flex_dd_compare(az,least)<0" in helper
  assert "FlexDD rotation[9]" in helper
  assert "0.8660253882408142f,1.553918593799608e-8f" in helper
  assert "-1.1102230246251565e-16f" in helper
  assert "rotation[8]" in helper
  assert helper.count("flex_dd_source_fma(") == 9
  assert "flex_dd_source_mul(u1,u1),one_minus_cos,flex_dd(-0.5f)" in helper
  assert "flex_dd_source_mul(u1,u2),one_minus_cos," in helper
  assert "flex_dd_source_mul(u3,u3),one_minus_cos,flex_dd(-0.5f)" in helper
  assert "FlexDD3 raw[3]={d1,d2,d3}" in helper
  assert "flex_dd3_div(raw[i],norm)" in helper
  matvec = shader[shader.index("static inline FlexDD flex_dd_source_matvec_row("):
                  shader.index("// Source-order counterpart of engine_collision_gjk.c:polytope2")]
  assert "flex_dd_source_mul(m1,x1)" in matvec
  assert "flex_dd_source_fma(m0,x0,value)" in matvec
  assert "flex_dd_source_fma(m2,x2,value)" in matvec

  # These source words would collapse to the same float32 pair.  In the
  # pinned mjtNum comparison, however, z is smaller than y, so strict '<'
  # selects coordinate 2 rather than retaining coordinate 1.
  edge_high = np.asarray([2.0, 1.0, 1.0], dtype=np.float32)
  edge_low = np.asarray([0.0, 2.0**-30, 0.0], dtype=np.float32)
  precise = edge_high.astype(np.float64) + edge_low.astype(np.float64)
  float_only = edge_high.copy()

  def least_axis(diff):
    value = float("inf")
    index = 0
    for axis in range(3):
      magnitude = abs(float(diff[axis]))
      if magnitude < value:
        value = magnitude
        index = axis
    return index

  assert least_axis(float_only) == 1
  assert least_axis(precise) == 2


def test_s2d_area_matches_pinned_arm64_fma_contraction():
  """The exact captured triangle needs the installed ARM64 FMA sequence."""
  from decimal import Decimal, localcontext

  # Captured exact-input 4085 GJK iter-2 simplex, projected on x/z. Pinned
  # engine_collision_gjk.c:S2D produces this C33 weight on the installed
  # ARM64 binary; separately rounded products lose one binary64 ULP.
  px, py = -3.46461688864063722e-35, -4.57054544407578593e-18
  ax, ay = -0.00288675134594812421, -0.00288675134594812421
  bx, by = 0.00288675134594812421, 0.102886752836064244
  det = -0.0113952870125042099

  def fma64(a, b, c):
    with localcontext() as ctx:
      ctx.prec = 200
      return float(Decimal.from_float(a) * Decimal.from_float(b)
                   + Decimal.from_float(c))

  def separately_rounded_area2():
    area = px * ay
    area = area + py * bx
    area = area + ax * by
    area = area - px * by
    area = area - py * ax
    area = area - bx * ay
    return area

  def arm64_area2():
    area = px * ay
    area = fma64(py, bx, area)
    area = fma64(ax, by, area)
    area = fma64(-px, by, area)
    area = fma64(-py, ax, area)
    area = fma64(-bx, ay, area)
    return area

  rounded_weight = separately_rounded_area2() / det
  pinned_weight = arm64_area2() / det
  assert rounded_weight == 0.02533285371220925
  assert pinned_weight == 0.025332853712209256
  assert pinned_weight != rounded_weight

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("static inline FlexDD flex_dd_area2(")
  stop = shader.index("static inline void flex_dd_s2d(", start)
  area = shader[start:stop]
  ordered = (
      "area=flex_dd_source_mul(ax,by)",
      "area=flex_dd_source_fma(ay,cx,area)",
      "area=flex_dd_source_fma(bx,cy,area)",
      "area=flex_dd_source_fma(flex_dd_neg(ax),cy,area)",
      "area=flex_dd_source_fma(flex_dd_neg(ay),bx,area)",
      "area=flex_dd_source_fma(flex_dd_neg(cx),by,area)",
  )
  assert tuple(sorted(ordered, key=area.index)) == ordered


def test_gjk_lincomb_preserves_installed_arm64_contraction_order():
  """GJK lincomb matches pinned arm64's reversed first FMA accumulation."""
  # Exact-input edcc line-simplex projection operands. This coordinate differs
  # by one binary64 result step when evaluated as source FMA versus a rounded
  # multiply followed by a rounded add.
  base = 0.04907477288111817
  direction = 0.09814954576223635
  scale = -0.7247617467462852
  rounded_product = scale * direction
  separate = base + rounded_product
  with localcontext() as context:
    context.prec = 100
    exact = (Decimal.from_float(base)
             + Decimal.from_float(scale) * Decimal.from_float(direction))
    fused = float(exact)
  assert separate == -0.022060263347874712
  assert fused == -0.02206026334787471
  assert separate != fused

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("static inline FlexDD3 flex_dd3_lincomb_source(")
  stop = shader.index("static inline FlexDD flex_dd3_component(", start)
  helper = shader[start:stop]
  assert "if (count==1) return flex_dd3_scale(values[0],weights[0]);" in helper
  assert "FlexDD3 result=flex_dd3_scale(values[1],weights[1]);" in helper
  assert "result=flex_dd3_madd(values[0],weights[0],result);" in helper
  assert "for (int i=2;i<count;i++)" in helper
  assert "result=flex_dd3_madd(values[i],weights[i],result);" in helper
  source_madd = shader[shader.index("static inline FlexDD3 flex_dd3_source_madd("):
                       shader.index("static inline FlexDD3 flex_dd3_source_sub(")]
  assert source_madd.count("flex_dd_source_fma(") == 3


def test_gjk_two_vertex_lincomb_rounds_term_one_before_fma_term_zero():
  """The pinned ARM64 two-point `lincomb` contraction is operand-ordered."""
  weight0, weight1 = 0.7247617467462851, 0.27523825325371465
  vertex0, vertex1 = -0.04907477288111818, 0.049074772881118195
  rounded_term0 = weight0 * vertex0
  rounded_term1 = weight1 * vertex1
  with localcontext() as context:
    context.prec = 100

    def fused(a, b, c):
      return float(Decimal.from_float(a) * Decimal.from_float(b)
                   + Decimal.from_float(c))

    arm64 = fused(weight0, vertex0, rounded_term1)
    opposite = fused(weight1, vertex1, rounded_term0)
  assert arm64 == -0.022060263347874698
  assert opposite == -0.0220602633478747
  assert arm64 != opposite


def test_sphere_support_keeps_geom_margin_as_a_separate_source_operation():
  """Sphere support + margin follows mjc_ccd's restored-support path."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  support = shader[shader.index("static inline FlexDDVertex flex_sphere_flex_support_dd("):
                   shader.index("static inline FlexDD flex_dd_abs(FlexDD value) {")]
  assert "FlexDD sphere_radius, FlexDD sphere_half_margin" in support
  assert "FlexDD3 a=flex_dd3_source_madd(direction,sphere_radius,sphere_origin);" in support
  assert "a=flex_dd3_source_madd(direction,sphere_half_margin,a);" in support
  assert "sphere_radius_margin_exact" not in support

  core = shader[shader.index("static inline int flex_sphere_tetra_epa_core("):
                shader.index("static inline int flex_sphere_tetra_epa(",
                             shader.index("static inline int flex_sphere_tetra_epa_core("))]
  assert "sphere_pos,sphere_rotation,flex_dd(0.0f),flex_dd(0.0f)," in core
  assert "sphere_pos,sphere_rotation,full_sphere_radius_exact,half_margin_exact," in core


def test_support_constructor_uses_source_fma_at_each_pinned_rounding_boundary():
  """The support path preserves the arm64 fused constructor operations."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  fma_start = shader.index(
      "static __attribute__((noinline)) FlexDD flex_dd_source_fma(FlexDD a,")
  fma_stop = shader.index("static inline FlexDD flex_dd_source_dot3(", fma_start)
  fma = shader[fma_start:fma_stop]
  # All nine products from the three-word operands are accumulated with their
  # exact float residuals before one source-matched binary64 rounding.
  assert fma.count("fma(a.") == 9
  assert "flex_expansion_round53(expansion,count)" in fma
  madd_start = shader.index("static inline FlexDD3 flex_dd3_source_madd(")
  madd_stop = shader.index("static inline FlexDD3 flex_dd3_source_sub(", madd_start)
  assert shader[madd_start:madd_stop].count("flex_dd_source_fma(") == 3
  support = shader[shader.index(
      "static inline FlexDDVertex flex_sphere_flex_support_dd("):
      shader.index("static inline FlexDD flex_dd_abs(FlexDD value) {")]
  assert "flex_dd3_source_madd(direction,sphere_radius,sphere_origin)" in support
  assert "flex_dd3_source_madd(direction,sphere_half_margin,a)" in support
  assert "flex_dd_source_add(flex_radius,flex_half_margin)" in support
  assert "flex_dd3_source_madd(\n      flex_direction,expansion," in support


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="support-constructor FMA helper requires GPU opt-in")
def test_mps_flex_support_fma_helper_matches_binary64_fma_cases():
  """Execute the production scalar FMA helper on captured-value-shaped words."""
  from fractions import Fraction

  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("support-constructor FMA helper qualification requires MPS")

  # Three float words encode source mjtNum operands.  Values include a
  # cancellation case and the same direction/radius/position shape used by
  # sphere and flex support constructors.
  values = [
      (0.5773502691896258, 0.005, 0.1),
      (-0.5773502691896258, 0.005, -0.2),
      (0.5773502691896258, 0.0025, 0.1),
      (1.0 + 2.0**-24, 1.0 - 2.0**-24, -1.0),
      (-1.0 + 2.0**-24, 1.0 + 2.0**-24, 1.0),
  ]

  def words(value):
    high = np.float32(value)
    rem = value - float(high)
    middle = np.float32(rem)
    low = np.float32(rem - float(middle))
    result = (high, middle, low)
    assert sum((Fraction.from_float(float(part)) for part in result),
               Fraction()) == Fraction.from_float(value)
    return result

  packed = np.zeros((len(values), 9), np.float32)
  expected = []
  for i, operands in enumerate(values):
    for j, value in enumerate(operands):
      packed[i, 3*j:3*j+3] = words(value)
    exact_a = sum((Fraction.from_float(float(x)) for x in packed[i, 0:3]),
                  Fraction())
    exact_b = sum((Fraction.from_float(float(x)) for x in packed[i, 3:6]),
                  Fraction())
    exact_c = sum((Fraction.from_float(float(x)) for x in packed[i, 6:9]),
                  Fraction())
    expected.append(float(exact_a * exact_b + exact_c))

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  helper_start = shader.index("struct FlexDD {")
  helper_stop = shader.index("static inline FlexDD3 flex_dd3_normalize(",
                             helper_start)
  helpers = shader[helper_start:helper_stop]
  probe = r"""
kernel void flex_support_fma_probe(
    device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int base=9*int(tid);
  FlexDD a=FlexDD{input[base],input[base+1],input[base+2]};
  FlexDD b=FlexDD{input[base+3],input[base+4],input[base+5]};
  FlexDD c=FlexDD{input[base+6],input[base+7],input[base+8]};
  FlexDD value=flex_dd_source_fma(a,b,c);
  output[3*tid]=value.hi;
  output[3*tid+1]=value.lo;
  output[3*tid+2]=value.tail;
}
"""
  library = torch.mps.compile_shader(
      "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + probe)
  input_device = torch.as_tensor(packed.copy(), dtype=torch.float32,
                                 device="mps").contiguous()
  output = torch.empty((len(values), 3), dtype=torch.float32,
                       device=input_device.device)
  library.flex_support_fma_probe(
      input_device.reshape(-1), output.reshape(-1),
      threads=(len(values),), group_size=(1,))
  torch.mps.synchronize()
  actual = output.cpu().numpy()
  for i in range(len(values)):
    exact_actual = sum((Fraction.from_float(float(x)) for x in actual[i]),
                       Fraction())
    assert exact_actual == Fraction.from_float(expected[i]), (
        i, actual[i], expected[i])


def test_absolute_source_dot_can_round_away_exact_projection_difference():
  """mju_flexSupport compares rounded absolute dots, not exact differences."""
  from decimal import Decimal, localcontext
  import numpy as np

  # The small y-direction contribution makes candidate-current's exact dot
  # positive, but it is below half an ulp of the large absolute projections.
  # Therefore source-order mju_dot3 comparisons tie and strict `>` retains the
  # first vertex. This prevents an algebraically equivalent, exact difference
  # comparison from being mistaken for pinned source arithmetic.
  lo = np.asarray([0.1, 0.0, 0.1], dtype=np.float32)
  candidate = np.asarray([0.1, 0.1, 0.1], dtype=np.float32)
  direction = np.asarray([0.70710677, 1.0e-17, 0.70710677], dtype=np.float64)
  delta = candidate.astype(np.float64) - lo.astype(np.float64)
  exact = sum(Decimal.from_float(float(delta[i]))
              * Decimal.from_float(float(direction[i])) for i in range(3))
  assert exact > 0
  source_dot_lo = float(lo[0] * direction[0] + lo[1] * direction[1]
                        + lo[2] * direction[2])
  source_dot_candidate = float(
      candidate[0] * direction[0] + candidate[1] * direction[1]
      + candidate[2] * direction[2])
  assert source_dot_candidate == source_dot_lo
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  assert "FlexProjection53 best_projection=flex_mju_dot3_projection53(" in shader
  assert "if (flex_projection53_compare(projection,best_projection)>0)" in shader
  assert "int quantum_exp=exponent-53;" in shader
  assert "product[i]=flex_expansion_round53(expansion,count);" in shader
  assert "FlexProjection53 first_fma=flex_expansion_round53(expansion,count);" in shader
  assert "int half_cmp=flex_expansion_compare_value(" in shader


def test_pinned_dot3_fma_order_resolves_float32_support_vertex():
  """Preserve the installed arm64 dot3 contraction at strict support ties."""
  from decimal import Decimal, localcontext
  import mujoco
  import numpy as np

  # These compiled float32 vertices tie in exact arithmetic. Pinned mju_dot3
  # contracts its source expression with two fused multiply-add boundaries and returns distinct
  # binary64 values; strict `>` must select vertex 1. Rounding the full exact
  # dot once ties and keeps vertex 0, changing P3's initial EPA faces.
  vertices = np.asarray([[0.1, 0.1, 0.1], [0.0, 0.1, 0.2]],
                        dtype=np.float32).astype(np.float64)
  direction = np.asarray([0.7071067811865475727,
                          1.8866277859186259e-16,
                          0.7071067811865475727], dtype=np.float64)
  projections = [float(mujoco.mju_dot3(vertex, direction))
                 for vertex in vertices]
  assert projections[0] < projections[1]
  assert int(np.argmax(projections)) == 1

  def fma64(a, b, c):
    with localcontext() as context:
      context.prec = 200
      return float(Decimal.from_float(float(a))
                   * Decimal.from_float(float(b))
                   + Decimal.from_float(float(c)))

  for vertex, projection in zip(vertices, projections):
    rounded_y = float(vertex[1] * direction[1])
    first_fma = fma64(vertex[0], direction[0], rounded_y)
    source_contract = fma64(vertex[2], direction[2], first_fma)
    assert source_contract == projection
  # The previous exact-sum implementation rounded the full dot only once,
  # preserving the mathematical tie that mju_dot3's two FMA boundaries break.
  assert sum(float(vertices[0, i] * direction[i]) for i in range(3)) == \
      sum(float(vertices[1, i] * direction[i]) for i in range(3))

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("FlexProjection53 flex_mju_dot3_projection53(")
  stop = shader.index("// Round the exact sum of float expansions", start)
  projection = shader[start:stop]
  assert "product[1].low" in projection
  assert "FlexProjection53 first_fma=flex_expansion_round53(expansion,count);" in projection
  assert "FlexDD scalar=coordinate[0];" in projection
  assert "scalar=coordinate[2]; d=axis[2];" in projection
  assert "return flex_expansion_round53(expansion,count);" in projection
  assert "FlexProjection53 last_two" not in projection


def test_epa_project_origin_plane_uses_pinned_source_rounding_boundaries():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("static inline bool flex_project_origin_plane_dd(")
  stop = shader.index("[[clang::noinline]] static inline bool flex_epa_make_face_dd(",
                      start)
  projection = shader[start:stop]
  assert "flex_dd3_source_sub(v2,v1)" in projection
  assert "flex_dd3_source_sub(v3,v1)" in projection
  assert "flex_dd3_source_sub(v3,v2)" in projection
  assert projection.count("FlexDD scale=flex_dd_source_div(nv,nn);") == 3
  assert projection.count("flex_dd_source_mul(n.x,scale)") == 3
  assert "flex_dd3_scale(n,flex_dd_div(nv,nn))" not in projection


def test_binary64_projection_rounding_reference_covers_gaps_and_ties():
  """Specify the pinned 53-bit projection boundary for generic expansions."""
  from decimal import Decimal, localcontext

  def exact(*parts):
    with localcontext() as context:
      context.prec = 120
      return sum(Decimal.from_float(float(x)) for x in parts)
  assert float(exact(1.0, 2.0**-60)) == 1.0
  assert float(exact(1.0, 2.0**-53)) == 1.0  # halfway, even lower significand
  assert float(exact(1.0, 3.0 * 2.0**-53)) == 1.0 + 2.0**-51
  assert float(exact(2.0, -(2.0**-53))) == 2.0  # binade boundary, ties even
  assert float(exact(1.0, -(1.0 - 2.0**-53))) == 2.0**-53
  # A float32-minimum tail sits far below the 53-bit quantum at this scale,
  # but it breaks an exact halfway case. The correct binary64 value is the
  # lower, odd neighbor because the exact input lies just below the midpoint.
  below_tie = (np.float32(2.0**100), np.float32(3.0 * 2.0**47),
               np.float32(-(2.0**-149)))
  assert np.float32(np.ldexp(below_tie[-1], -48)) == 0.0
  assert np.float32(np.ldexp(below_tie[1], -48)) == np.float32(1.5)
  with localcontext() as context:
    context.prec = 200
    below_tie_exact = sum(
        Decimal.from_float(float(value)) for value in below_tie)
  assert float(below_tie_exact) == 2.0**100 + 2.0**48

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("FlexProjection53 flex_expansion_round53(")
  stop = shader.index("static inline int flex_projection53_compare(", start)
  rounder = shader[start:stop]
  # Every component below the leading float contributes to the quantum count;
  # this is essential when the second expansion component is widely separated.
  assert "for (int i=0;i<count-1;i++)" in rounder
  assert "int quantum_exp=exponent-53;" in rounder
  assert "result.mid=ldexp(step_hi,quantum_exp);" in rounder
  assert "result.low=ldexp(step_lo,quantum_exp);" in rounder
  assert "residual,residual_count,ldexp(lower+0.5f,quantum_exp)" in rounder
  assert "((int(step_hi)+int(lower))&1)==0" in rounder
  add = shader.split(
      "int flex_expansion_add(", 1)[1].split(
          "int flex_expansion_sign(", 1)[0]
  sign = shader.split(
      "int flex_expansion_sign(", 1)[1].split(
          "int flex_expansion_compare_value(", 1)[0]
  # This exact halfway witness requires preserving the binary32 subnormal
  # word without arithmetic on it (MPS flushes such operations), then reading
  # its sign from the IEEE payload bits.
  assert "flex_float_is_subnormal(value)" in add
  assert "flex_float_is_subnormal(expansion[i])" in add
  assert "as_type<uint>(expansion[i])" in sign
  assert "subnormal_sum += " in sign


def test_epa_face_distance_comparison_uses_expansion_value_order():
  """Three-word source scalars compare by represented value, not word tuple."""
  from decimal import Decimal

  # Captured iteration-22 EPA face values from the original strict CCD fixture.
  # The first expansion has a lower high word but a larger represented value.
  best = (np.float32(0.006387272849678993),
          np.float32(3.2874220012857336e-10),
          np.float32(1.6479873021779667e-17))
  runner = (np.float32(0.0063872733153402805),
            np.float32(-1.369190733013781e-10),
            np.float32(8.673617379884035e-19))
  def exact(words):
    with localcontext() as context:
      context.prec = 120
      return sum(Decimal.from_float(float(word)) for word in words)
  assert best[0] < runner[0]  # Lexicographic comparison gets this backwards.
  assert exact(best) > exact(runner)

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  compare = shader[shader.index("static inline int flex_dd_compare("):
                   shader.index("static inline FlexDD3 flex_dd3(")]
  assert "return flex_dd_compare_exact(a,b);" in compare
  exact_compare = shader[
      shader.index("static __attribute__((noinline)) int flex_dd_compare_exact"):
      shader.index("static inline FlexProjection53 flex_mju_dot3_projection53")]
  assert "return flex_projection53_compare(" in exact_compare


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="EPA expansion comparator requires GPU opt-in")
def test_mps_epa_expansion_comparator_matches_captured_face_order():
  """Execute the production scalar comparator on the strict-face witness."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("EPA expansion comparator qualification requires MPS")

  first = np.asarray([[0.006387272849678993,
                       3.2874220012857336e-10,
                       1.6479873021779667e-17]], dtype=np.float32)
  second = np.asarray([[0.0063872733153402805,
                        -1.369190733013781e-10,
                        8.673617379884035e-19]], dtype=np.float32)
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  helper_start = shader.index("struct FlexDD {")
  helper_stop = shader.index("static inline FlexDD3 flex_dd3_normalize(",
                             helper_start)
  helpers = shader[helper_start:helper_stop]
  probe = r"""
kernel void flex_compare_expansion_probe(
    device const float* first [[buffer(0)]],
    device const float* second [[buffer(1)]],
    device int* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  output[tid]=flex_dd_compare_exact(
      FlexDD{first[3*tid],first[3*tid+1],first[3*tid+2]},
      FlexDD{second[3*tid],second[3*tid+1],second[3*tid+2]});
}
"""
  library = torch.mps.compile_shader(
      "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + probe)
  device = lambda value, dtype: torch.as_tensor(
      value, dtype=dtype, device="mps").contiguous()
  out = torch.empty((1,), dtype=torch.int32, device="mps")
  library.flex_compare_expansion_probe(
      device(first, torch.float32).reshape(-1),
      device(second, torch.float32).reshape(-1), out,
      threads=(1,), group_size=(1,))
  torch.mps.synchronize()
  assert int(out.cpu().item()) == 1


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="53-bit projection helper requires GPU opt-in")
def test_mps_flex_projection_round53_matches_binary64_reference_cases():
  """Execute the production expansion rounder on gap, tie, and binade inputs."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("53-bit projection helper qualification requires MPS")
  from decimal import Decimal, localcontext

  cases = [
      (1.0, 2.0**-60),
      (1.0, 2.0**-53),
      (1.0, 3.0 * 2.0**-53),
      (2.0, -(3.0 * 2.0**-54)),
      (-2.0, 3.0 * 2.0**-54),
      (2.0, -(2.0**-53)),
      (-2.0, 2.0**-53),
      (1.0e8, 1.0, -1.0e8),
      (2.0**-60, 1.0, -1.0),
      # The minimum float32 word breaks a halfway tie at a much larger scale.
      (2.0**100, 3.0 * 2.0**47, -(2.0**-149)),
      (-(2.0**100), -3.0 * 2.0**47, 2.0**-149),
  ]
  terms = np.zeros((len(cases), 8), dtype=np.float32)
  counts = np.zeros(len(cases), dtype=np.int32)
  for row, values in enumerate(cases):
    terms[row, :len(values)] = np.asarray(values, dtype=np.float32)
    counts[row] = len(values)

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  helper_start = shader.index("struct FlexDD {")
  helper_stop = shader.index("static inline FlexDD3 flex_dd3_normalize(",
                             helper_start)
  helpers = shader[helper_start:helper_stop]
  probe = r"""
kernel void flex_projection_round53_probe(
    device const float* terms [[buffer(0)]],
    device const int* counts [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  thread float expansion[16];
  thread float scratch[16];
  int count=0;
  int n=counts[tid];
  for (int i=0;i<n;i++) {
    count=flex_expansion_add(expansion,count,terms[8*tid+i],scratch);
  }
  FlexProjection53 value=flex_expansion_round53(expansion,count);
  output[3*tid]=value.hi;
  output[3*tid+1]=value.mid;
  output[3*tid+2]=value.low;
}
"""
  library = torch.mps.compile_shader(
      "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + probe)
  mps = lambda value, dtype: torch.as_tensor(
      value, dtype=dtype, device="mps").contiguous()
  terms_device = mps(terms, torch.float32)
  counts_device = mps(counts, torch.int32)
  output = torch.empty((len(cases), 3), dtype=torch.float32,
                       device=terms_device.device)
  library.flex_projection_round53_probe(
      terms_device.reshape(-1), counts_device, output.reshape(-1),
      threads=(len(cases),), group_size=(1,))
  torch.mps.synchronize()
  components = output.cpu().numpy()
  for row, values in enumerate(cases):
    with localcontext() as context:
      context.prec = 120
      exact_value = sum(
          Decimal.from_float(float(np.float32(value))) for value in values)
    expected = float(exact_value)
    actual = float(float(components[row, 0])
                   + float(components[row, 1])
                   + float(components[row, 2]))
    assert actual == expected, (row, values, components[row], expected, actual)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="CCD quotient helper requires GPU opt-in")
def test_mps_flex_source_division_matches_decimal_binary64_cases():
  """Execute production quotient refinement on full three-word operands."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("CCD quotient helper qualification requires MPS")
  from decimal import Decimal, localcontext

  f32 = np.float32
  cases = [
      ((-0.28128743, -6.6804633e-9, -1.0551505e-15),
       (-0.39080098, 4.8194537e-9, -2.385536e-16)),
      ((0.73123455, 1.0913936e-8, -1.4210855e-15),
       (0.9135792, -2.9802322e-8, 3.5527137e-15)),
      ((-1.0, 2.9802322e-8, -1.7763568e-15),
       (0.375, -1.4901161e-8, 8.8817842e-16)),
  ]
  operands = np.asarray([
      [*(f32(x) for x in numerator), *(f32(x) for x in denominator)]
      for numerator, denominator in cases], dtype=np.float32)
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  helper_start = shader.index("struct FlexDD {")
  helper_stop = shader.index("static inline FlexDD3 flex_dd3_normalize(",
                             helper_start)
  helpers = shader[helper_start:helper_stop]
  probe = r"""
kernel void flex_source_division_probe(
    device const float* operands [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  FlexDD numerator=FlexDD{operands[6*tid],operands[6*tid+1],
                          operands[6*tid+2]};
  FlexDD denominator=FlexDD{operands[6*tid+3],operands[6*tid+4],
                            operands[6*tid+5]};
  FlexDD quotient=flex_dd_source_div(numerator,denominator);
  output[3*tid]=quotient.hi;
  output[3*tid+1]=quotient.lo;
  output[3*tid+2]=quotient.tail;
}
"""
  library = torch.mps.compile_shader(
      "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + probe)
  operand_device = torch.as_tensor(
      operands, dtype=torch.float32, device="mps").contiguous()
  output = torch.empty((len(cases), 3), dtype=torch.float32,
                       device=operand_device.device)
  library.flex_source_division_probe(
      operand_device.reshape(-1), output.reshape(-1),
      threads=(len(cases),), group_size=(1,))
  torch.mps.synchronize()
  actual = output.cpu().numpy()
  with localcontext() as context:
    context.prec = 200
    for row, (numerator, denominator) in enumerate(cases):
      a = sum((Decimal.from_float(float(f32(value))) for value in numerator),
              Decimal(0))
      b = sum((Decimal.from_float(float(f32(value))) for value in denominator),
              Decimal(0))
      expected = float(a / b)
      reconstructed = sum((Decimal.from_float(float(value))
                           for value in actual[row]), Decimal(0))
      assert float(reconstructed) == expected, (row, actual[row], expected)


def test_flex_source_sqrt_uses_unrounded_product_residuals():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("static inline FlexDD flex_dd_source_sqrt(")
  stop = shader.index("static inline FlexDD3 flex_dd3_cross(", start)
  sqrt_source = shader[start:stop]
  assert "flex_dd_product_residual(value,q0_value,q0_value)" in sqrt_source
  assert "flex_dd_product_residual(value,approximation,approximation)" in sqrt_source

  # This input is a reproducible binary64-ulp counterexample: rounding q0*q0
  # before subtracting it from the three-word input loses the q2 correction.
  words = np.asarray((.6045916, -3.158632e-9, 1.110223e-16), np.float32)
  with localcontext() as context:
    context.prec = 200
    exact_value = sum((Decimal.from_float(float(word)) for word in words),
                      Decimal(0))
    expected = float(exact_value.sqrt())
  assert expected == 0.7775548886662944


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="source-rounded sqrt helper requires GPU opt-in")
def test_mps_flex_source_sqrt_matches_decimal_binary64_counterexample():
  """Execute production sqrt refinement on the captured three-word case."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("source-rounded sqrt qualification requires MPS")
  words = np.asarray((.6045916, -3.158632e-9, 1.110223e-16), np.float32)
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  helper_start = shader.index("struct FlexDD {")
  helper_stop = shader.index("static inline FlexDD3 flex_dd3_normalize(",
                             helper_start)
  helpers = shader[helper_start:helper_stop]
  probe = r"""
kernel void flex_source_sqrt_probe(
    device const float* words [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  FlexDD value=FlexDD{words[3*tid],words[3*tid+1],words[3*tid+2]};
  FlexDD root=flex_dd_source_sqrt(value);
  output[3*tid]=root.hi;
  output[3*tid+1]=root.lo;
  output[3*tid+2]=root.tail;
}
"""
  library = torch.mps.compile_shader(
      "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + probe)
  operand = torch.as_tensor(words[None, :], dtype=torch.float32,
                            device="mps").contiguous()
  output = torch.empty((1, 3), dtype=torch.float32, device=operand.device)
  library.flex_source_sqrt_probe(
      operand.reshape(-1), output.reshape(-1), threads=(1,), group_size=(1,))
  torch.mps.synchronize()
  actual_words = output.cpu().numpy()[0]
  actual = float(sum((Decimal.from_float(float(word))
                      for word in actual_words), Decimal(0)))
  with localcontext() as context:
    context.prec = 200
    value = sum((Decimal.from_float(float(word)) for word in words), Decimal(0))
    expected = float(value.sqrt())
  assert actual == expected == 0.7775548886662944, (actual_words, actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="source-rounded cross helper requires GPU opt-in")
def test_mps_flex_source_cross_matches_binary64_operands():
  """Execute source-rounded cross products, including near cancellation."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("source-rounded cross qualification requires MPS")
  f32 = np.float32
  cases = [
      ([f32(.2), f32(2**-26), f32(.2), f32(0), f32(.2), f32(0)],
       [f32(.3), f32(0), f32(.3), f32(2**-26), f32(.3), f32(0)]),
      ([f32(1.25), f32(2**-25), f32(-.75), f32(2**-27), f32(.5), f32(0)],
       [f32(-.5), f32(0), f32(.125), f32(2**-26), f32(2.0), f32(2**-24)]),
      ([f32(.01), f32(2**-31), f32(.02), f32(2**-30), f32(-.03), f32(0)],
       [f32(.04), f32(0), f32(-.05), f32(2**-29), f32(.06), f32(2**-28)]),
  ]
  packed = np.asarray([
      [word for values in pair for i in range(0, 6, 2)
       for word in (values[i], values[i+1], 0.0)]
      for pair in cases], dtype=np.float32)
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  helper_start = shader.index("struct FlexDD {")
  helper_stop = shader.index("static inline FlexDD3 flex_dd3_normalize(",
                             helper_start)
  helpers = shader[helper_start:helper_stop]
  probe = r"""
kernel void flex_source_cross_probe(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int base=int(tid)*18;
  FlexDD3 a=FlexDD3{FlexDD{input[base],input[base+1],input[base+2]},
                    FlexDD{input[base+3],input[base+4],input[base+5]},
                    FlexDD{input[base+6],input[base+7],input[base+8]}};
  FlexDD3 b=FlexDD3{FlexDD{input[base+9],input[base+10],input[base+11]},
                    FlexDD{input[base+12],input[base+13],input[base+14]},
                    FlexDD{input[base+15],input[base+16],input[base+17]}};
  FlexDD3 value=flex_dd3_cross(a,b);
  output[9*tid]=value.x.hi; output[9*tid+1]=value.x.lo;
  output[9*tid+2]=value.x.tail;
  output[9*tid+3]=value.y.hi; output[9*tid+4]=value.y.lo;
  output[9*tid+5]=value.y.tail;
  output[9*tid+6]=value.z.hi; output[9*tid+7]=value.z.lo;
  output[9*tid+8]=value.z.tail;
}
"""
  library = torch.mps.compile_shader(
      "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + probe)
  to_mps = lambda value, dtype: torch.as_tensor(
      value, dtype=dtype, device="mps").contiguous()
  source = to_mps(packed, torch.float32)
  output = torch.empty((len(cases), 9), dtype=torch.float32,
                      device=source.device)
  library.flex_source_cross_probe(
      source.reshape(-1), output.reshape(-1),
      threads=(len(cases),), group_size=(1,))
  torch.mps.synchronize()
  result = output.cpu().numpy()
  for row, (a, b) in enumerate(cases):
    av = [float(a[0])+float(a[1]), float(a[2])+float(a[3]),
          float(a[4])+float(a[5])]
    bv = [float(b[0])+float(b[1]), float(b[2])+float(b[3]),
          float(b[4])+float(b[5])]
    expected = [av[1]*bv[2]-av[2]*bv[1],
                av[2]*bv[0]-av[0]*bv[2],
                av[0]*bv[1]-av[1]*bv[0]]
    actual = [float(result[row, i])+float(result[row, i+1])
              +float(result[row, i+2]) for i in (0, 3, 6)]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-30)


def test_sphere_flex_deep_epa_uses_dynamic_device_workspace():
  """Keep EPA's large per-query arrays in configured-capacity device arenas."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  core_start = shader.index("static inline int flex_sphere_tetra_epa_core(")
  core_stop = shader.index("static inline int flex_sphere_tetra_epa(", core_start)
  core = shader[core_start:core_stop]
  assert "device FlexDDVertex* dd_vertices" in core
  assert "device FlexEpaFace* faces" in core
  assert "device int* face_map" in core
  assert "device int2* horizon" in core
  assert "device FlexHorizonFrame* horizon_stack" in core
  assert "FlexDDVertex dd_vertices[55];" not in core
  assert "static_assert(sizeof(CxVertex)==48" in shader
  assert "static_assert(sizeof(FlexDD)==12" in shader
  assert "static_assert(sizeof(FlexDDVertex)==108" in shader
  assert "static_assert(sizeof(FlexEpaFace)==80" in shader
  assert "static_assert(sizeof(FlexHorizonFrame)==24" in shader
  assert "device float* epa_float_workspace [[buffer(25)]]" in shader
  assert "device int* epa_int_workspace [[buffer(26)]]" in shader
  assert "FlexDD upper_dd" in core
  assert "face.dist2_dd" in core
  assert "flex_epa_horizon_visit_dd(" in core
  assert "flex_epa_make_face_dd(" in core
  assert "FlexEpaFace witness_face=faces[best_face];" in core
  assert "flex_epa_write_witness_dd(witness_face,dd_vertices,contact);" in core
  assert "flex_epa_write_witness(faces[best_face],vertices,contact);" not in core
  witness_start = shader.index("static inline void flex_epa_write_witness_dd(")
  witness_stop = shader.index("static inline int flex_epa_face_vertex", witness_start)
  witness = shader[witness_start:witness_stop]
  # These are the signed minors from pinned engine_collision_gjk.c's
  # triAffineCoord. Keep the -v1 cross terms negative in the DD lane.
  assert "flex_dd_sub(flex_dd_mul(s1.z,s3.y),flex_dd_mul(s1.y,s3.z))" in witness
  assert "flex_dd_sub(flex_dd_mul(s1.z,s3.x),flex_dd_mul(s1.x,s3.z))" in witness
  assert "flex_dd_sub(flex_dd_mul(s1.y,s3.x),flex_dd_mul(s1.x,s3.y))" in witness


def test_optional_ccd_iteration_trace_is_bounded_and_disjoint():
  """Size every native EPA event record from the configured iteration cap."""
  assert _flex_ccd_trace_capacity(35) == (35, 210, 790, 27_902)
  assert _flex_ccd_trace_capacity(1000) == (1000, 6000, 18_160,
                                             18_160_252)
  assert _flex_ccd_trace_capacity(-2) == (0, 0, 160, 252)
  assert _flex_ccd_workspace_capacity(35) == (40, 210, 24, 210)
  assert _flex_ccd_workspace_capacity(1000) == (1005, 6000, 24, 6000)
  vertex_capacity, face_capacity, horizon_capacity, stack_capacity = (
      _flex_ccd_workspace_capacity(35, batch_size=2, slot_count=7))
  float_stride = (vertex_capacity * 39 + face_capacity * 20 + 3) & ~3
  int_stride = ((face_capacity + 1) & ~1) + horizon_capacity * 2 + stack_capacity * 6
  assert (vertex_capacity, face_capacity, horizon_capacity, stack_capacity) == (
      40, 210, 24, 210)
  assert (float_stride, int_stride) == (5760, 1518)
  vcap, fcap, hcap, scap = _flex_ccd_workspace_capacity(34)
  stride = (vcap * 39 + fcap * 20 + 3) & ~3
  assert stride % 4 == 0  # next CxVertex pointer remains 16-byte aligned
  with pytest.raises(ValueError, match="signed 32-bit"):
    _flex_ccd_trace_capacity(50, batch_size=100_000, slot_count=100_000)
  with pytest.raises(ValueError, match="signed 32-bit"):
    _flex_ccd_workspace_capacity(1000, batch_size=100_000,
                                 slot_count=100_000)
  iterations, map_capacity, stride, total = _flex_ccd_trace_capacity(35)
  assert stride == (_FLEX_CCD_ITER_TRACE_HEADER + 3 * map_capacity
                    + _FLEX_CCD_ITER_TRACE_HORIZON
                    + _FLEX_CCD_ITER_TRACE_VERTEX_WORDS)
  assert total == _FLEX_CCD_TRACE_SIZE + iterations * stride
  assert _FLEX_CCD_ITER_TRACE_HEADER == 58
  assert (_FLEX_CCD_ITER_TRACE_BASE, _FLEX_CCD_ITER_TRACE_DISTANCE_WORDS,
          _FLEX_CCD_ITER_TRACE_PROJECTION_WORDS) == (32, 8, 18)
  assert _FLEX_CCD_ITER_TRACE_HEADER == (
      _FLEX_CCD_ITER_TRACE_BASE + _FLEX_CCD_ITER_TRACE_DISTANCE_WORDS
      + _FLEX_CCD_ITER_TRACE_PROJECTION_WORDS)
  # The last eight header words retain best and runner-up IDs plus the exact
  # three float words of each squared distance, without lossy hi+mid packing.
  assert list(range(_FLEX_CCD_ITER_TRACE_BASE,
                    _FLEX_CCD_ITER_TRACE_BASE
                    + _FLEX_CCD_ITER_TRACE_DISTANCE_WORDS)) == list(range(32, 40))
  # Candidate projections are stored as xyz high/mid/low triples after IDs
  # and distances, so source-order division/scaling can be compared directly.
  assert list(range(40, 58)) == list(range(40, _FLEX_CCD_ITER_TRACE_HEADER))
  # Header: 0:32; each of three map records has exactly map_capacity entries;
  # the final horizon block stores ordered face/edge pairs in 48 scalars.
  offsets = [0, _FLEX_CCD_ITER_TRACE_HEADER,
             _FLEX_CCD_ITER_TRACE_HEADER + map_capacity,
             _FLEX_CCD_ITER_TRACE_HEADER + 2 * map_capacity,
             _FLEX_CCD_ITER_TRACE_HEADER + 3 * map_capacity,
             _FLEX_CCD_ITER_TRACE_HEADER + 3 * map_capacity
             + _FLEX_CCD_ITER_TRACE_HORIZON]
  lengths = [_FLEX_CCD_ITER_TRACE_HEADER, map_capacity,
             map_capacity, map_capacity, _FLEX_CCD_ITER_TRACE_HORIZON,
             _FLEX_CCD_ITER_TRACE_VERTEX_WORDS]
  occupied = [i for start, length in zip(offsets, lengths)
              for i in range(start, start + length)]
  assert occupied == list(range(stride))


def test_ccd_trace_captures_both_candidate_face_projections_without_math_use():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  assert "static constant int FLEX_CCD_TRACE_ITER_HEADER = 58;" in shader
  start = shader.index("FlexDD3 best_trace_projection=faces[best_face].v_dd;")
  stop = shader.index("for (int i=0;i<iteration_trace_map_capacity;i++)", start)
  capture = shader[start:stop]
  assert "event[40]=best_trace_projection.x.hi" in capture
  assert "event[48]=best_trace_projection.z.tail" in capture
  assert "event[49]=runner_up_projection.x.hi" in capture
  assert "event[57]=runner_up_projection.z.tail" in capture
  # This block follows both physical face comparisons and only executes inside
  # capture_iteration_trace; diagnostic words never affect selected IDs.
  loop = shader.rindex("if (capture_iteration_trace) {", 0, start)
  assert loop < start
  face_vertices = capture[capture.index("int face_vertices[6]"):]
  assert "dd_vertices[vertex_id].m.x" in face_vertices
  assert "component.hi" in face_vertices and "component.tail" in face_vertices


def test_optional_ccd_trace_base_uses_dynamic_slot_stride():
  """Catch copies that alias base trace into the prior slot's EPA tail."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  assert "ccd_trace[dims[10]*tid+i]=ccd_trace_local[i]" in shader
  assert "ccd_trace[FLEX_CCD_TRACE_SIZE*tid+i]" not in shader


def test_epa_horizon_ring_uses_fixed_expansion_base():
  """Pin the source's fixed nfaces wrap while the live face count advances."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  assert "int expansion_base=nface;" in shader
  assert "(h==0) ? expansion_base+nhorizon-1 : new_id-1" in shader
  assert "(h==nhorizon-1) ? expansion_base : new_id+1" in shader
  for base in (6, 9, 18):
    for count in (3, 4, 7, 24):
      created = [base + h for h in range(count)]
      adjacency = []
      live_count = base
      for h in range(count):
        new_id = live_count
        prev = base + count - 1 if h == 0 else new_id - 1
        nxt = base if h == count - 1 else new_id + 1
        adjacency.append((prev, nxt))
        live_count += 1
      assert [item[0] for item in adjacency] == [created[-1], *created[:-1]]
      assert [item[1] for item in adjacency] == [*created[1:], created[0]]
      assert all(prev != base + h and nxt != base + h
                 for h, (prev, nxt) in enumerate(adjacency))


def test_native_sphere_tetra_ccd_has_a_separate_lazy_pipeline():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  general_start = shader.index("kernel void flex_contact_detect(")
  ccd_start = shader.index("kernel void flex_contact_detect_native_ccd(")
  general = shader[general_start:ccd_start]
  native_ccd = shader[ccd_start:shader.index(
      "// Pinned filterFlexContacts", ccd_start)]
  assert "flex_sphere_tetra_epa(" not in general
  assert "flex_sphere_tetra_epa(" in native_ccd
  assert "device float* epa_float_workspace [[buffer(25)]]" in native_ccd
  assert "device int* epa_int_workspace [[buffer(26)]]" in native_ccd
  assert "self._shader.flex_contact_detect_native_ccd" in (
      Path(__file__).parents[1] / "mujoco_metal" / "flex_contact.py").read_text()


def test_epa_dynamic_horizon_rejects_invalid_face_indices_before_deref():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("static inline int flex_epa_horizon_visit_dd(")
  stop = shader.index("static inline int flex_same_sign", start)
  visit = shader[start:stop]
  assert "int stack_capacity, int face_count" in visit
  assert "if (next_face<0 || next_face>=face_count) return -2;" in visit
  assert visit.index("next_face>=face_count") < visit.index("faces[next_face]")
  assert "++steps>ulong(face_count)*4ul+ulong(stack_capacity)" in visit
  assert "stack[top].face>=face_count" in visit
  assert "flex_epa_delete_face(faces,map,map_count,frame.face,face_count)" in visit
  assert "FLEX_CCD_EPA_BAD_ADJACENCY" in shader


def test_epa_delete_and_map_scan_validate_dynamic_indices():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("static inline bool flex_epa_delete_face(")
  stop = shader.index("struct FlexHorizonFrame", start)
  delete = shader[start:stop]
  assert "face_id>=face_count" in delete
  assert "index>=map_count" in delete
  assert "replacement>=face_count" in delete
  core_start = shader.index("static inline int flex_sphere_tetra_epa_core(")
  core_stop = shader.index("static inline int flex_sphere_tetra_epa(", core_start)
  core = shader[core_start:core_stop]
  assert "face_id>=nface" in core
  assert "adj_id>=nface" in core


def test_sphere_ccd_cutoff_and_inflation_follow_pinned_source():
  """Keep the shallow point query distinct from zero-cutoff deep GJK."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  assert "float cutoff2=distance_cutoff*distance_cutoff;" in shader
  assert "bool backup_gjk=distance_cutoff<=0.0f;" in shader
  assert "separating_dot*separating_dot/x_norm2)>=cutoff2" in shader
  assert "FlexDD sphere_radius_exact=FlexDD{geom_size[geom*3]," in shader
  assert "FlexDD threshold_exact=flex_dd_source_add(margin_exact,gap_exact);" in shader
  assert "FlexDD ccd_tolerance_exact=FlexDD{ccd_tolerance[0]," in shader
  assert "sphere_radius_margin_exact=flex_dd_source_add(" in shader
  assert "ccd_tolerance_exact,sphere_radius_margin_exact," in shader
  assert "FlexDD depth=flex_dd_sub(point_distance_dd," in shader
  assert "contact.dist=(depth.hi+depth.lo)+depth.tail;" in shader
  assert "gp,Rg,size,\n            CX_FLEX,center," in shader
  native = shader[shader.index("kernel void flex_contact_detect_native_ccd("):]
  assert "device const float* flexvert_xpos_low [[buffer(9)]]" in native
  assert "flex_center_paired_precise(flexvert_xpos,flexvert_xpos_low" in native
  native_compact = "".join(native.split())
  assert "support_world_dd[i]=FlexDD3{" in native_compact
  assert ("FlexDD{flexvert_xpos[index],flexvert_xpos_low[index],"
          "flexvert_xpos_low[batch*nvert*3+index]}" in native_compact)
  assert "geom_hull,geom_hull_info,supportV);" in shader
  assert "dist=raw_sphere_epa || raw_sphere_tetra ? selected.dist+threshold" in shader


def test_pinned_double_runtime_uses_double_build_epa_min_distances():
  """The installed 3.10 oracle is double precision, not mjUSESINGLE."""
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
  assert np.asarray(model.qpos0).dtype == np.float64
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  assert "mjMINDIST3 is\n  // mjMINVAL2 (1e-30)" in shader
  assert shader.count("flex_dd(1.0e-30f)") >= 8
  assert "face.dist2<1.0e-10f" not in shader
  assert "face.dist2<1.0e-17f" not in shader


def _dense_efc_jacobian(model, data):
  """Expand pinned efc_J using the compiled dense/CSR representation."""
  values = np.asarray(data.efc_J, dtype=np.float64)
  if not mujoco.mj_isSparse(model):
    return values.reshape(int(data.nefc), int(model.nv)).copy()
  result = np.zeros((int(data.nefc), int(model.nv)), dtype=np.float64)
  for row in range(int(data.nefc)):
    start = int(data.efc_J_rowadr[row])
    count = int(data.efc_J_rownnz[row])
    columns = np.asarray(data.efc_J_colind[start:start + count], dtype=np.int64)
    result[row, columns] = values[start:start + count]
  return result


def _fixture(condim, cone, gap=0.0, flex_z=0.0):
  xml = f"""
    <mujoco><option gravity="0 0 0" cone="{cone}" impratio="1.7"
                    timestep=".003"/>
      <worldbody>
        <geom name="ground" type="plane" size="0 0 .1"
              contype="0" conaffinity="1" condim="{condim}"
              friction=".7 .4 .03" gap="{gap}"/>
        <flexcomp name="cloth" type="grid" count="2 2 1"
                  pos="0 0 {flex_z}" spacing=".1 .1 .1" mass="1" dim="2">
          <contact contype="1" conaffinity="0" selfcollide="none"
                   condim="{condim}" friction=".5 .6 .02"/>
          <elasticity young="100" poisson=".2" thickness=".01"
                      elastic2d="stretch"/>
        </flexcomp>
      </worldbody>
    </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-0.03, 0.02, int(model.nv))
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)
  return model, data, descriptor


def _moving_geom_flex_fixture(joint_type="free"):
  if joint_type == "free":
    joint_xml = "<freejoint/>"
  elif joint_type == "ball":
    joint_xml = '<joint name="ball_joint" type="ball"/>'
  elif joint_type == "hinge":
    joint_xml = ('<joint name="ball_joint" type="hinge" '
                 'axis="0 1 0"/>' )
  elif joint_type == "slide":
    joint_xml = ('<joint name="ball_joint" type="slide" '
                 'axis="0 0 1"/>' )
  else:
    raise ValueError(joint_type)
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" jacobian="dense" cone="elliptic"/>
      <worldbody>
        <body name="ball">
          JOINT_XML
          <inertial pos=".013 -.007 .009" mass="1"
                    diaginertia=".001 .001 .001"/>
          <geom name="ball_geom" type="sphere" size=".05" mass="0"
                contype="0" conaffinity="1" condim="1"/>
        </body>
        <flexcomp name="cloth" type="grid" count="2 2 1"
                  pos="0 0 .04" spacing=".04 .04 .01"
                  mass=".1" radius=".002" dim="2">
          <contact contype="1" conaffinity="0" condim="1"
                   selfcollide="none"/>
          <elasticity young="100" poisson=".2" thickness=".01"
                      elastic2d="stretch"/>
        </flexcomp>
      </worldbody>
    </mujoco>
  """.replace("JOINT_XML", joint_xml))
  data = mujoco.MjData(model)
  if joint_type == "ball":
    mujoco.mju_axisAngle2Quat(data.qpos[:4], np.asarray([0., 1., 0.]), 0.2)
  elif joint_type == "hinge":
    data.qpos[0] = 0.2
  elif joint_type == "slide":
    data.qpos[0] = 0.005
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)
  assert data.ncon > 0 and data.nefc == data.ncon
  return model, data, descriptor


def _cdof_point_jacobian(model, data, body, point):
  """Independently translate compiled cdof columns to a rigid point."""
  body = int(body)
  root = int(model.body_rootid[body])
  offset = np.asarray(point, np.float64) - np.asarray(
      data.subtree_com[root], np.float64)
  cdof = np.asarray(data.cdof, np.float64).reshape(int(model.nv), 6)
  spatial = np.zeros((6, int(model.nv)), np.float64)
  spatial[3:] = cdof[:, :3].T
  spatial[:3] = cdof[:, 3:].T + np.cross(cdof[:, :3], offset).T
  # A cdof row is globally stored for every DOF, including unrelated trees;
  # only the geom's compiled body/joint ancestry contributes to mj_jac.
  mask = np.zeros(int(model.nv), dtype=bool)
  current = body
  dof_counts = {
      int(mujoco.mjtJoint.mjJNT_FREE): 6,
      int(mujoco.mjtJoint.mjJNT_BALL): 3,
      int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
      int(mujoco.mjtJoint.mjJNT_HINGE): 1,
  }
  while current > 0:
    start = int(model.body_jntadr[current])
    stop = start + int(model.body_jntnum[current])
    for joint in range(start, stop):
      first = int(model.jnt_dofadr[joint])
      mask[first:first + dof_counts[int(model.jnt_type[joint])]] = True
    current = int(model.body_parentid[current])
  spatial[:, ~mask] = 0.0
  return spatial


@pytest.mark.parametrize("joint_type", ["free", "ball", "hinge", "slide"])
def test_moving_geom_point_jacobian_cdof_matches_public_mj_jac(joint_type):
  model, data, _ = _moving_geom_flex_fixture(joint_type)
  contacts = [contact for contact in data.contact[:data.ncon]
              if int(contact.flex[1]) == 0]
  assert contacts
  for contact in contacts:
    body = int(model.geom_bodyid[int(contact.geom[0])])
    jp = np.zeros((3, int(model.nv)), np.float64)
    jr = np.zeros_like(jp)
    mujoco.mj_jac(model, data, jp, jr, contact.pos, body)
    np.testing.assert_allclose(
        _cdof_point_jacobian(model, data, body, contact.pos),
        np.concatenate((jp, jr), axis=0), rtol=0, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native rigid-flex Jacobian requires GPU opt-in")
def test_native_rigid_flex_spatial_jacobian_matches_pinned_contact_rows():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native rigid-flex Jacobian qualification requires MPS")
  from mujoco_metal.flex import MetalFlex
  from mujoco_metal.flex_contact import FlexContactProgram

  model, data, descriptor = _moving_geom_flex_fixture()
  mps = lambda x: torch.as_tensor(
      np.asarray(x, np.float32)[None].copy(), dtype=torch.float32,
      device="mps").contiguous()
  poses = {
      "body_pos": mps(data.xpos), "body_quat": mps(data.xquat),
      "joint_anchor": mps(data.xanchor), "joint_axis": mps(data.xaxis),
      "root_com": mps(data.subtree_com),
  }
  flex = MetalFlex(model, device="mps")
  flex.update_kinematics(poses, mps(data.cvel))
  np.testing.assert_allclose(flex.flexvert_xpos.cpu().numpy()[0],
                             data.flexvert_xpos, rtol=0, atol=2e-6)
  geom_quat = np.empty((int(model.ngeom), 4), np.float64)
  for geom in range(int(model.ngeom)):
    mujoco.mju_mat2Quat(
        geom_quat[geom], np.asarray(data.geom_xmat[geom], np.float64).reshape(9))
  program = FlexContactProgram(model, device="mps")
  # Isolated kernel exercise only. The public production contact route remains
  # guarded until the full candidate/solver pipeline passes admission.
  assert not program._narrowphase_admitted
  program._narrowphase_admitted = True
  result = program.run_device(
      flex.flexvert_xpos, mps(data.geom_xpos), mps(geom_quat))
  active = result["active"].cpu().numpy()[0]
  assert active.any()
  cpu_contacts = [c for c in data.contact[:data.ncon]
                  if int(c.geom[0]) == int(descriptor.geom[np.flatnonzero(
                      descriptor.geom >= 0)[0]]) and int(c.flex[1]) == 0]
  assert cpu_contacts
  sides = program.run_native_contact_spatial_jacobians(
      result, flex.contact_spatial_jacobians(),
      mps(data.cdof.reshape(model.nv, 6)), mps(data.subtree_com))
  dense_efc = _dense_efc_jacobian(model, data)
  # Exercise the complete two-sided native contact-J -> canonical row bridge,
  # not only its intermediate spatial-J buffers.  This fixture is condim=1,
  # so pinned post-impedance diagA is unchanged by the cone rewrite and is the
  # independent solver diagonal consumed by FlexContactRows.
  from mujoco_metal.flex_contact_rows import FlexContactRows
  nrow = int(descriptor.row_capacity)
  diagA = np.zeros((1, nrow), np.float32)
  expected_J = np.zeros((nrow, int(model.nv)), np.float32)
  expected_R = np.zeros(nrow, np.float32)
  expected_aref = np.zeros(nrow, np.float32)
  unmatched = list(cpu_contacts)
  for slot in np.flatnonzero(active):
    elem = int(descriptor.elem1[slot])
    candidates = [c for c in unmatched if int(c.elem[1]) == elem]
    assert candidates, (slot, elem)
    contact = candidates[0]
    unmatched.remove(contact)
    np.testing.assert_allclose(
        result["dist"].cpu().numpy()[0, slot], contact.dist,
        rtol=0, atol=3e-6)
    np.testing.assert_allclose(
        result["pos"].cpu().numpy()[0, slot], contact.pos,
        rtol=0, atol=3e-6)
    bary = result["barycentric1"].cpu().numpy()[0, slot, :3]
    np.testing.assert_allclose(bary.sum(), 1.0, rtol=0, atol=2e-5)
    node_ids = np.asarray(descriptor.nodes1[slot, :3], np.int32)
    surface_point = (np.asarray(contact.pos, np.float64)
                     + np.asarray(contact.frame[:3], np.float64)
                     * (float(model.flex_radius[0])
                        + 0.5*float(contact.dist)))
    tri = np.asarray(data.flexvert_xpos[node_ids], np.float64)
    basis = np.column_stack((tri[1]-tri[0], tri[2]-tri[0]))
    uv = np.linalg.lstsq(basis, surface_point-tri[0], rcond=None)[0]
    ref_bary = np.asarray([1.0-uv.sum(), uv[0], uv[1]])
    np.testing.assert_allclose(bary, ref_bary, rtol=0, atol=3e-5)

    ref_flex = np.zeros((6, int(model.nv)), np.float64)
    for weight, vertex in zip(ref_bary, node_ids):
      ref_flex += weight * _point_body_spatial_jacobian(
          model, data, data.flexvert_xpos[int(vertex)],
          model.flex_vertbodyid[int(vertex)])
    geom_body = int(model.geom_bodyid[int(contact.geom[0])])
    ref_geom = _point_body_spatial_jacobian(
        model, data, contact.pos, geom_body)
    ref_relative = ref_flex-ref_geom
    actual_relative = sides["relative"].cpu().numpy()[0, slot]
    np.testing.assert_allclose(actual_relative, ref_relative,
                               rtol=4e-5, atol=4e-6)
    efc_row = int(contact.efc_address)
    normal_row = np.asarray(contact.frame[:3], np.float64) @ ref_relative[:3]
    np.testing.assert_allclose(dense_efc[efc_row], normal_row,
                               rtol=4e-7, atol=4e-8)
    row = int(descriptor.row_start[slot])
    assert int(descriptor.row_span[slot]) == 1
    diagA[0, row] = float(data.efc_diagA[efc_row])
    expected_J[row] = dense_efc[efc_row].astype(np.float32)
    expected_R[row] = float(data.efc_R[efc_row])
    expected_aref[row] = float(data.efc_aref[efc_row])
  assert not unmatched

  row_program = FlexContactRows(model, descriptor, device="mps")
  rows = row_program.run_device(
      result, sides["relative"], mps(data.qvel),
      torch.as_tensor(diagA, dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(rows["workspace_J"].cpu().numpy()[0],
                             expected_J, rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(rows["R"].cpu().numpy()[0], expected_R,
                             rtol=8e-5, atol=2e-6)
  np.testing.assert_allclose(rows["aref"].cpu().numpy()[0], expected_aref,
                             rtol=8e-5, atol=2e-5)


def test_gap_only_flex_candidates_keep_zero_includemargin_and_are_excluded():
  model, data, descriptor = _fixture(3, "elliptic", gap=0.04, flex_z=0.025)
  assert data.ncon == int(model.flex_vertnum[0])
  assert descriptor.slot_count >= data.ncon
  np.testing.assert_allclose(descriptor.gap[descriptor.kind == 0], 0.04)
  for contact in data.contact[:data.ncon]:
    # mjc_FlexGeomVertex expands candidate search by margin+gap, while
    # mj_setContact stores margin alone and excludes contacts in the gap.
    assert contact.dist < float(model.geom_gap[0])
    assert contact.includemargin == pytest.approx(0.0)
    assert contact.exclude
    assert contact.efc_address < 0
  assert data.nefc == 0


def test_row_descriptor_validation_rejects_overlaps_and_wrong_cone_spans():
  _, _, descriptor = _fixture(3, "elliptic")
  from mujoco_metal.flex_contact_rows import FlexContactRows

  bad_start = np.array(descriptor.row_start, copy=True)
  bad_start[1] = bad_start[0]
  with pytest.raises(ValueError, match="disjoint|span"):
    FlexContactRows._validate_descriptor(
        replace(descriptor, row_start=bad_start), descriptor.slot_count,
        descriptor.row_capacity, descriptor.nv)

  bad_span = np.array(descriptor.row_span, copy=True)
  bad_span[0] += 1
  with pytest.raises(ValueError, match="span"):
      FlexContactRows._validate_descriptor(
          replace(descriptor, row_span=bad_span), descriptor.slot_count,
          descriptor.row_capacity, descriptor.nv)


def test_flex_radius_compiled_high_low_preserves_pinned_double_model_values():
  model, _data, descriptor = _dynamic_diag_fixture(
      "trilinear", False, False, geom_margin_gap=0.0,
      disable_midphase=False)
  high = np.asarray(descriptor.flex_radius_hi, dtype=np.float64)
  mid = np.asarray(descriptor.flex_radius_mid, dtype=np.float64)
  low = np.asarray(descriptor.flex_radius_low, dtype=np.float64)
  expected = np.asarray(model.flex_radius, dtype=np.float64)
  assert high.shape == mid.shape == low.shape == expected.shape
  assert np.any(mid != 0.0) or np.any(low != 0.0), (
      "fixture must exercise nonzero float32 residuals")
  np.testing.assert_array_equal(high + mid + low, expected)
  for slot, flex_id in enumerate(descriptor.flex1):
    if flex_id >= 0:
      assert descriptor.radius1[slot] == np.float32(expected[int(flex_id)])


def test_native_ccd_model_scalars_preserve_float32_residuals():
  model, _data, descriptor = _dynamic_diag_fixture(
      "trilinear", False, False, geom_margin_gap=0.0,
      disable_midphase=False)
  geom_high = np.asarray(descriptor.geom_size_hi, dtype=np.float64)
  geom_mid = np.asarray(descriptor.geom_size_mid, dtype=np.float64)
  geom_low = np.asarray(descriptor.geom_size_low, dtype=np.float64)
  geom_expected = np.asarray(model.geom_size, dtype=np.float64)
  assert np.any(geom_low != 0.0)
  np.testing.assert_array_equal(geom_high + geom_mid + geom_low, geom_expected)
  margin_high = np.asarray(descriptor.margin, dtype=np.float64)
  margin_mid = np.asarray(descriptor.margin_mid, dtype=np.float64)
  margin_low = np.asarray(descriptor.margin_low, dtype=np.float64)
  gap_high = np.asarray(descriptor.gap, dtype=np.float64)
  gap_mid = np.asarray(descriptor.gap_mid, dtype=np.float64)
  gap_low = np.asarray(descriptor.gap_low, dtype=np.float64)
  expected_margin = []
  expected_gap = []
  for geom, flex_id in zip(descriptor.geom, descriptor.flex1):
    if geom >= 0:
      source_margin = (float(model.geom_margin[int(geom)])
                       + float(model.flex_margin[int(flex_id)]))
      expected_margin.append(
          float(model.opt.o_margin)
          if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
          else source_margin)
      expected_gap.append(float(model.geom_gap[int(geom)])
                           + float(model.flex_gap[int(flex_id)]))
    else:
      expected_margin.append(float(model.flex_margin[int(flex_id)]))
      expected_gap.append(float(model.flex_gap[int(flex_id)]))
  np.testing.assert_array_equal(
      margin_high + margin_mid + margin_low, expected_margin)
  np.testing.assert_array_equal(gap_high + gap_mid + gap_low, expected_gap)


def test_ccd_cross_and_dot_follow_pinned_arm64_contraction_order():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  cross_start = shader.index("static inline FlexDD3 flex_dd3_cross(")
  cross_stop = shader.index("static inline FlexDD3 flex_dd3_normalize(",
                            cross_start)
  cross = shader[cross_start:cross_stop]
  assert cross.count("flex_dd_source_mulsub(") == 3
  mulsub_start = shader.index("static inline FlexDD flex_dd_source_mulsub(")
  mulsub_stop = shader.index("static inline FlexProjection53 flex_projection53_add(",
                             mulsub_start)
  mulsub = shader[mulsub_start:mulsub_stop]
  assert "flex_dd_neg(flex_dd_source_mul(c,d))" in mulsub
  assert "return flex_dd_source_fma(a,b,rounded_negative);" in mulsub
  dot_start = shader.rindex("static inline FlexDD flex_dd_source_dot3(",
                            0, cross_start)
  dot_stop = shader.index("static inline FlexDD3 flex_dd3_cross(", dot_start)
  dot = shader[dot_start:dot_stop]
  assert "FlexDD y=flex_dd_source_mul(a.y,b.y);" in dot
  assert "FlexDD xy=flex_dd_source_fma(a.x,b.x,y);" in dot
  assert "return flex_dd_source_fma(a.z,b.z,xy);" in dot
  support_start = shader.index(
      "static inline FlexDDVertex flex_sphere_flex_support_dd(")
  support_stop = shader.index("static inline FlexDD flex_dd_abs(",
                              support_start)
  support = shader[support_start:support_stop]
  assert "flex_dd3_source_madd(direction,sphere_radius,sphere_origin)" in support
  assert "a=flex_dd3_source_madd(direction,sphere_half_margin,a);" in support
  assert "flex_dd_source_add(flex_radius,flex_half_margin)" in support
  assert "flex_dd_source_sub(a.x,b.x)" in support
  assert "static inline FlexDD3 flex_dd3_source_sub(" in shader
  assert "FlexDD3 diff1=flex_dd3_source_sub(seed_dd[1].m,seed_dd[0].m);" in shader
  assert "FlexDD sphere_radius_margin_exact=flex_dd_source_add(" in shader


def test_arm64_fma_oracle_differs_from_two_rounded_products():
  """Exercise the source operation boundary for cancellation-sensitive cross."""
  def fma64(a, b, c):
    with localcontext() as ctx:
      ctx.prec = 200
      return float(Decimal.from_float(a) * Decimal.from_float(b)
                   + Decimal.from_float(c))

  # Values are exactly representable binary64 operands.  The pinned arm64
  # sequence rounds c*d, negates it, then fuses a*b with that rounded value.
  a = 1.0 + 2.0**-27
  b = 1.0 - 2.0**-27
  c = 1.0
  d = 1.0
  source = fma64(a, b, -float(c * d))
  separate = float(a * b) - float(c * d)
  assert source == -2.0**-54
  assert separate == 0.0


def test_gjk_direction_matches_pinned_reciprocal_then_scale_order():
  """GJK normalizes by one reciprocal; EPA support uses component division."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  gjk = shader[shader.index("static inline FlexDD flex_source_gjk_dd("):
               shader.index("static inline int flex_sphere_tetra_epa_core(")]
  assert "flex_dd_source_div(flex_dd(1.0f),x_norm)" in gjk
  assert "flex_dd3_scale(x,inverse_norm)" in gjk
  assert "flex_dd3_neg(dir_neg)" in gjk
  assert "flex_dd3_div(flex_dd3_neg(x),x_norm)" not in gjk
  # engine_collision_gjk.c:epaSupport divides each normal component by its
  # norm, so keep the EPA path's separately rounded operation intact.
  epa = shader[shader.index("int flex_sphere_tetra_epa_core("):
               shader.index("static inline int flex_sphere_tetra_epa(")]
  assert "flex_dd3_div(nearest.v_dd,lower_dd)" in epa


def test_flex_radius_high_low_device_buffers_match_descriptor():
  torch = pytest.importorskip("torch")
  model, _data, descriptor = _dynamic_diag_fixture(
      "trilinear", False, False, geom_margin_gap=0.0,
      disable_midphase=False)
  program = FlexContactProgram(model, device="cpu")
  np.testing.assert_array_equal(
      program._radius1.numpy(), descriptor.flex_radius_hi)
  np.testing.assert_array_equal(program._radius1_midlow.numpy(),
      np.stack((descriptor.flex_radius_mid, descriptor.flex_radius_low), axis=1))
  np.testing.assert_array_equal(program._geom_size_midlow.numpy(),
      np.stack((descriptor.geom_size_mid, descriptor.geom_size_low), axis=2))
  np.testing.assert_array_equal(program._margin_gap_midlow.numpy(),
      np.stack((np.stack((descriptor.margin_mid, descriptor.margin_low), axis=1),
                np.stack((descriptor.gap_mid, descriptor.gap_low), axis=1)),
               axis=1))
  np.testing.assert_array_equal(program._radius2.numpy(), descriptor.radius2)


@pytest.mark.parametrize("condim", [1, 3, 4, 6])
@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
def test_descriptor_spans_match_pinned_cpu_contacts(condim, cone):
  model, data, descriptor = _fixture(condim, cone)
  span = (condim if cone == "elliptic" or condim == 1
          else 2 * (condim - 1))
  plane_slots = np.flatnonzero(descriptor.kind == 0)
  assert plane_slots.size == int(model.flex_vertnum[0])
  np.testing.assert_array_equal(descriptor.row_span[plane_slots], span)
  assert descriptor.row_capacity == span * descriptor.slot_count
  assert data.ncon == plane_slots.size
  assert data.nefc == data.ncon * span
  contacts_by_vertex = {int(c.vert[1]): c for c in data.contact[:data.ncon]}
  for slot in plane_slots:
    contact = contacts_by_vertex[int(descriptor.vert1[slot])]
    assert int(contact.dim) == condim
    assert int(contact.efc_address) >= 0
    assert int(descriptor.row_start[slot]) == int(slot) * span
    assert int(data.efc_type[contact.efc_address]) == (
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        if condim == 1 else
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)
        if cone == "elliptic" else
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL))


@pytest.mark.parametrize(("condim", "cone"), [
    (1, "elliptic"), (3, "elliptic"), (3, "pyramidal"),
    (4, "elliptic"), (6, "pyramidal"),
])
def test_plane_flex_vertex_diag_approx_matches_pinned_cpu(condim, cone):
  """Check host-lowered body weights against mj_diagApprox's contact row."""
  model, data, descriptor = _fixture(condim, cone)
  invweight = np.asarray(model.body_invweight0).reshape(model.nbody, 2)
  for slot, kind in enumerate(descriptor.kind):
    if int(kind) != 0:
      continue
    contact = next(c for c in data.contact[:data.ncon]
                   if int(c.vert[1]) == int(descriptor.vert1[slot]))
    vertex = int(descriptor.vert1[slot])
    flex_id = int(descriptor.flex1[slot])
    weights = _flex_vertex_body_weights(model, flex_id, vertex)
    geom_body = int(model.geom_bodyid[int(descriptor.geom[slot])])
    expected = invweight[geom_body].copy()
    for body, weight in weights.items():
      expected += invweight[body] * weight
    addr = int(contact.efc_address)
    if cone == "pyramidal" and condim > 1:
      # mj_makeImpedance rewrites pyramid diagA after constructing Rpy.
      observed = float(data.efc_diagA[addr]) / (2.0*float(contact.mu)**2)
    else:
      observed = float(data.efc_diagA[addr])
    if cone == "pyramidal" and condim > 1:
      # mj_diagApprox includes each pyramid ray's mu^2 contribution before
      # mj_makeImpedance rewrites the row block to its common Rpy.
      expected[0] *= 1.0 + float(contact.friction[0])**2
    np.testing.assert_allclose(observed, expected[0], rtol=2e-7, atol=2e-9)


def test_interpolated_shell_vertex_diag_approx_matches_pinned_cpu():
  """Recover pre-impedance mj_diagApprox before checking shell body weights."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom type="plane" size="0 0 .1" contype="0" conaffinity="1"/>
      <flexcomp name="shell" type="grid" count="3 3 3"
                pos="0 0 -.02" spacing=".1 .1 .1" mass="1" dim="3"
                dof="trilinear">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <elasticity young="100" poisson=".2" damping=".1"
                    thickness=".01" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert int(model.flex_interp[0]) == -1
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  contact = next(c for c in data.contact[:data.ncon]
                 if int(c.flex[1]) == 0 and int(c.vert[1]) == 0)
  # The exact source host body weighting maps this Q1 corner to node 0. The
  # post-impedance pyramid efc_diagA has the common Rpy multiplier; undo it
  # before comparing with source mj_diagApprox's original contact row.
  weights = _flex_vertex_body_weights(model, 0, 0)
  tran = sum(float(model.body_invweight0[body, 0])*weight
             for body, weight in weights.items())
  source_diag = tran*(1.0+float(contact.friction[0])**2)
  mu_effective = float(contact.mu)
  observed_source_diag = (float(data.efc_diagA[int(contact.efc_address)])
                          /(2.0*mu_effective*mu_effective))
  np.testing.assert_allclose(observed_source_diag, source_diag,
                             rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_interpolated_flex_vertex_body_weights_match_frictionless_cpu(dof, shell):
  bend = ' thickness=".01" elastic2d="bend"' if shell else ""
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      <geom type="plane" size="0 0 .1" contype="0" conaffinity="1"
            condim="1"/>
      <flexcomp name="interp" type="grid" count="3 3 3"
                pos="0 0 -.02" spacing=".1 .1 .1" mass="1" dim="3"
                dof="{dof}">
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1"/>
        <elasticity young="100" poisson=".2" damping=".1"{bend}/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  order = 1 if dof == "trilinear" else 2
  assert int(model.flex_interp[0]) == (-order if shell else order)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  invweight = np.asarray(model.body_invweight0).reshape(model.nbody, 2)
  for contact in data.contact[:data.ncon]:
    vertex = int(contact.vert[1])
    weights = _flex_vertex_body_weights(model, 0, vertex)
    geom_body = int(model.geom_bodyid[int(contact.geom[0])])
    expected = float(invweight[geom_body, 0]) + sum(
        float(invweight[body, 0])*weight for body, weight in weights.items())
    np.testing.assert_allclose(
        float(data.efc_diagA[int(contact.efc_address)]), expected,
        rtol=2e-7, atol=2e-9)


def test_interpolated_plane_candidate_expands_all_node_trees_for_wake_links():
  """Q2 plane candidates depend on multiple compiled node-body trees."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" jacobian="dense"/><worldbody>
      <geom name="floor" type="plane" size="0 0 .1"
            contype="0" conaffinity="1"/>
      <flexcomp name="interpolated" type="grid" count="4 4 4"
                pos="0 0 -.02" spacing=".1 .1 .1" mass="1" dim="3"
                dof="quadratic">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)

  # Match a real pinned plane contact to its fixed candidate slot, then check
  # the complete mj_vertBodyWeight expansion.  `flex_vertbodyid` holds
  # placeholders for interpolated vertices and cannot establish these links.
  contact_and_vertex = next(
      (c, int(model.flex_vertadr[0]) + int(c.vert[1]))
      for c in data.contact[:data.ncon]
      if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
      and int(c.vert[1]) >= 0
      and len({int(model.body_treeid[body])
               for body in _flex_vertex_body_weights(
                   model, 0, int(model.flex_vertadr[0]) + int(c.vert[1]))
               if int(model.body_treeid[body]) >= 0}) > 2)
  _, vertex = contact_and_vertex
  slots = [slot for slot, kind in enumerate(descriptor.kind)
           if int(kind) == _KIND_PLANE_VERTEX
           and int(descriptor.geom[slot]) == 0
           and int(descriptor.vert1[slot]) == vertex]
  assert len(slots) == 1
  slot = slots[0]
  body_weights = _flex_vertex_body_weights(model, 0, vertex)
  expected_trees = {
      int(model.body_treeid[body]) for body in body_weights
      if int(model.body_treeid[body]) >= 0
  }
  assert len(expected_trees) > 2
  assert abs(int(model.flex_interp[0])) == 2
  assert len(body_weights) > 2

  # The plane is attached to the world, which is not a moving tree, so its
  # static side contributes no wake edge.  The multi-tree flex side must be
  # expanded before this omission; a -1 placeholder body would lose the
  # dependency set and fail the assertions above.
  assert int(model.body_treeid[int(model.geom_bodyid[0])]) < 0
  assert descriptor.link_capacity == 0
  assert descriptor.candidate_link_ids.shape == (descriptor.slot_count, 0)

  # The same source expansion must emit edges when the other side is a
  # moving tree.  This Q2 sphere/simplex fixture has real pinned contacts;
  # its candidate row's node weights span four flex trees in addition to
  # the sphere body tree.
  moving_model, moving_data, moving = _dynamic_diag_fixture(
      "quadratic", False, False)
  sphere_geom = 0
  contact_elements = {
      int(c.elem[1]) for c in moving_data.contact[:moving_data.ncon]
      if int(c.geom[0]) == sphere_geom and int(c.elem[1]) >= 0
  }
  assert contact_elements
  moving_slot = next(
      slot for slot, kind in enumerate(moving.kind)
      if int(kind) == _KIND_GEOM_ELEMENT
      and int(moving.geom[slot]) == sphere_geom
      and int(moving.elem1[slot]) in contact_elements)
  # Element contacts can be interior to the Q2 simplex. Include every basis
  # node body in the flex domain; a corner-only union would miss midside and
  # shell-TFI contributors that become active at an interior contact point.
  node_adr = int(moving_model.flex_nodeadr[0])
  node_num = int(moving_model.flex_nodenum[0])
  flex_trees = {
      int(moving_model.body_treeid[body])
      for body in np.asarray(moving_model.flex_nodebodyid)[
          node_adr:node_adr + node_num]
      if int(moving_model.body_treeid[body]) >= 0
  }
  sphere_body = int(moving_model.geom_bodyid[sphere_geom])
  sphere_tree = int(moving_model.body_treeid[sphere_body])
  assert len(flex_trees) > 2 and sphere_tree >= 0
  expected_pairs = {
      (min(tree, sphere_tree), max(tree, sphere_tree))
      for tree in flex_trees if tree >= 0 and tree != sphere_tree
  }
  emitted_ids = [int(i) for i in moving.candidate_link_ids[moving_slot]
                 if int(i) >= 0]
  emitted_pairs = {tuple(int(v) for v in moving.link_tree_pairs[i])
                   for i in emitted_ids}
  assert emitted_pairs == expected_pairs


@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_interpolated_element_ccd_center_is_compiled_aabb_center(dof, shell):
  """Pinned mjc_center uses flexelem_aabb center, not simplex centroid."""
  model, data, _ = _dynamic_diag_fixture(dof, shell, False)
  f = 0
  global_elem = int(model.flex_elemadr[f])
  nodes_per_elem = int(model.flex_dim[f]) + 1
  vertadr = int(model.flex_vertadr[f])
  elemadr = int(model.flex_elemdataadr[f]) + global_elem * nodes_per_elem
  vertices = (np.asarray(model.flex_elem[elemadr:elemadr + nodes_per_elem],
                         dtype=np.int32) + vertadr)
  positions = np.asarray(data.flexvert_xpos).reshape(-1, 3)[vertices]
  compiled_center = np.asarray(data.flexelem_aabb).reshape(-1, 6)[
      global_elem, :3]
  np.testing.assert_allclose(
      compiled_center, 0.5 * (positions.min(axis=0) + positions.max(axis=0)),
      rtol=0, atol=1e-12)
  assert np.linalg.norm(compiled_center - positions.mean(axis=0)) > 1e-3


@pytest.mark.parametrize(("dof", "shell"), [
    (None, False), ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_compiled_interpolated_element_candidates_match_pinned_contact_ids(
    dof, shell):
  """Keep fixed slot element/node identity tied to compiled MuJoCo contacts."""
  model, data, descriptor = _dynamic_diag_fixture(dof, shell, False)
  contacts = [c for c in data.contact[:data.ncon]
              if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
              and int(c.elem[1]) >= 0]
  assert contacts
  interp = int(model.flex_interp[0])
  expected_interp = (0 if dof is None else
                     (-1 if shell and dof == "trilinear" else
                      -2 if shell else 1 if dof == "trilinear" else 2))
  assert interp == expected_interp
  expected_elements = _flex_bvh_element_order(
      model, 0, int(model.flex_elemnum[0]))
  np.testing.assert_array_equal(descriptor.elem1, expected_elements)
  assert descriptor.slot_count == len(expected_elements)
  nodes_per_element = int(model.flex_dim[0]) + 1
  element_base = int(model.flex_elemdataadr[0])
  vertex_base = int(model.flex_vertadr[0])
  compiled = np.asarray(model.flex_elem, dtype=np.int32)
  seen = set()
  for contact in contacts:
    element = int(contact.elem[1])
    assert element not in seen
    seen.add(element)
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_GEOM_ELEMENT)
        & (descriptor.geom == int(contact.geom[0]))
        & (descriptor.flex1 == int(contact.flex[1]))
        & (descriptor.elem1 == element))
    assert slots.size == 1
    slot = int(slots[0])
    assert int(descriptor.contact_ordinal[slot]) == 0
    expected_nodes = (compiled[element_base + element*nodes_per_element:
                                element_base + (element+1)*nodes_per_element]
                      + vertex_base)
    np.testing.assert_array_equal(
        descriptor.nodes1[slot, :nodes_per_element], expected_nodes)
    assert np.all(descriptor.nodes1[slot, nodes_per_element:] == -1)
    # The same source mj_elemBodyWeight -> mj_vertBodyWeight mapping used by
    # dynamic diagApprox must be valid at each actual compiled contact point.
    weights = _flex_element_body_weights(
        model, 0, element, int(contact.vert[0]), contact.pos,
        data.flexvert_xpos)
    assert weights and np.isfinite(list(weights.values())).all()
    invweight = np.asarray(model.body_invweight0, np.float64).reshape(-1, 2)
    geom_body = int(model.geom_bodyid[int(contact.geom[0])])
    source_diag = float(invweight[geom_body, 0])
    source_diag += sum(float(invweight[body, 0])*weight
                       for body, weight in weights.items())
    np.testing.assert_allclose(
        float(data.efc_diagA[int(contact.efc_address)]), source_diag,
        rtol=2e-7, atol=2e-9)
  assert seen == {int(c.elem[1]) for c in contacts}


@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_interpolated_plane_candidate_admission_matches_pinned_cpu(dof, shell):
  pytest.importorskip("torch")
  model, descriptor, qpos, qvel = _interpolated_plane_fixture(dof, shell)
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  epa_v, epa_f, epa_h, epa_s = _flex_ccd_workspace_capacity(
      model.opt.ccd_iterations, batch_size=2, slot_count=descriptor.slot_count)
  assert tuple(program._epa_float_workspace.shape[:2]) == (
      2, max(descriptor.slot_count, 1))
  assert program._epa_float_workspace.shape[-1] == (
      epa_v * 39 + epa_f * 20 + 3) & ~3
  assert tuple(program._epa_int_workspace.shape[:2]) == (
      2, max(descriptor.slot_count, 1))
  assert program._epa_int_workspace.shape[-1] == (
      ((epa_f + 1) & ~1) + epa_h * 2 + epa_s * 6)
  assert int(model.flex_interp[0]) != 0
  assert program._narrowphase_admitted
  assert descriptor.slot_count == int(model.flex_vertnum[0])
  for env in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.flex[1]) == 0 and int(c.vert[1]) >= 0]
    assert contacts
    for contact in contacts:
      matches = np.flatnonzero(
          (descriptor.kind == _KIND_PLANE_VERTEX)
          & (descriptor.vert1 == int(contact.vert[1])))
      assert matches.size == 1


@pytest.mark.parametrize(("dof", "shell"), [
    (None, False), ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_element_contact_body_weights_match_pinned_diag_approx(dof, shell):
  dof_attr = "" if dof is None else f'dof="{dof}"'
  shell_attr = ' thickness=".01" elastic2d="bend"' if shell else ""
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      <body pos=".1 .1 .1"><freejoint/>
        <geom type="sphere" size=".08" condim="1"/>
      </body>
      <flexcomp name="interp" type="grid" count="3 3 3"
                pos="0 0 .1" spacing=".1 .1 .1" mass="1" dim="3"
                {dof_attr}>
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1"/>
        <elasticity young="100" poisson=".2"{shell_attr}/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  if dof is not None:
    order = 1 if dof == "trilinear" else 2
    assert int(model.flex_interp[0]) == (-order if shell else order)
  element_contacts = [c for c in data.contact[:data.ncon]
                      if int(c.flex[1]) == 0 and int(c.elem[1]) >= 0]
  assert element_contacts
  invweight = np.asarray(model.body_invweight0, dtype=np.float64).reshape(-1, 2)
  geom_body = int(model.geom_bodyid[0])
  for contact in element_contacts:
    weights = _flex_element_body_weights(
        model, 0, int(contact.elem[1]), int(contact.vert[0]), contact.pos,
        data.flexvert_xpos)
    expected = float(invweight[geom_body, 0]) + sum(
        float(invweight[body, 0])*weight for body, weight in weights.items())
    np.testing.assert_allclose(
        float(data.efc_diagA[int(contact.efc_address)]), expected,
        rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize(("dof", "shell"), [
    (None, False), ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_cross_flex_element_weights_match_pinned_diag_approx(dof, shell):
  dof_attr = "" if dof is None else f'dof="{dof}"'
  shell_attr = ' thickness=".01" elastic2d="bend"' if shell else ""
  flexes = []
  for name, pos, contype, conaffinity in (
      ("a", "0 0 .1", 1, 2), ("b", ".03 .02 .09", 2, 1)):
    flexes.append(f"""
      <flexcomp name="{name}" type="grid" count="2 2 2" pos="{pos}"
                spacing=".1 .1 .1" mass="1" dim="3" {dof_attr}>
        <contact contype="{contype}" conaffinity="{conaffinity}"
                 selfcollide="none" condim="1"/>
        <elasticity young="100" poisson=".2"{shell_attr}/>
      </flexcomp>
    """)
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      {''.join(flexes)}
    </worldbody></mujoco>
  """)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  if dof is not None:
    order = 1 if dof == "trilinear" else 2
    assert int(model.flex_interp[0]) == (-order if shell else order)
    assert int(model.flex_interp[1]) == (-order if shell else order)
  contacts = [c for c in data.contact[:data.ncon]
              if int(c.flex[0]) == 0 and int(c.flex[1]) == 1
              and int(c.elem[0]) >= 0 and int(c.elem[1]) >= 0]
  assert contacts
  invweight = np.asarray(model.body_invweight0, dtype=np.float64).reshape(-1, 2)
  for contact in contacts:
    expected = 0.0
    for side in (0, 1):
      flex_id = int(contact.flex[side])
      weights = _flex_element_body_weights(
          model, flex_id, int(contact.elem[side]),
          int(contact.vert[1-side]), contact.pos, data.flexvert_xpos)
      expected += sum(float(invweight[body, 0])*weight
                      for body, weight in weights.items())
    np.testing.assert_allclose(
        float(data.efc_diagA[int(contact.efc_address)]), expected,
        rtol=2e-7, atol=2e-9)


def _dynamic_diag_fixture(dof, shell, cross_flex, geom_margin_gap=0.0,
                          jacobian="dense", disable_midphase=False):
  dof_attr = "" if dof is None else f'dof="{dof}"'
  shell_attr = ' thickness=".01" elastic2d="bend"' if shell else ""
  if cross_flex:
    obstacles = "".join(f"""
      <flexcomp name="{name}" type="grid" count="2 2 2" pos="{pos}"
                spacing=".1 .1 .1" mass="1" dim="3" {dof_attr}>
        <contact contype="{contype}" conaffinity="{conaffinity}"
                 selfcollide="none" condim="1"/>
        <elasticity young="100" poisson=".2"{shell_attr}/>
      </flexcomp>
    """ for name, pos, contype, conaffinity in (
        ("a", "0 0 .1", 1, 2), ("b", ".03 .02 .09", 2, 1)))
    xml = f"""<mujoco><option gravity="0 0 0" cone="elliptic"
                         jacobian="{jacobian}"/>
      <worldbody>{obstacles}</worldbody></mujoco>"""
  else:
    xml = f"""<mujoco><option gravity="0 0 0" cone="elliptic"
                         jacobian="dense"/>
      <worldbody>
        <body pos=".1 .1 .1"><freejoint/>
          <geom type="sphere" size=".08" condim="1"
                margin="{geom_margin_gap}" gap="{geom_margin_gap}"/>
        </body>
        <flexcomp name="interp" type="grid" count="3 3 3"
                  pos="0 0 .1" spacing=".1 .1 .1" mass="1" dim="3"
                  {dof_attr}>
          <contact contype="1" conaffinity="0" selfcollide="none"
                   condim="1"/>
          <elasticity young="100" poisson=".2"{shell_attr}/>
        </flexcomp>
      </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  if disable_midphase:
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
  if jacobian == "sparse":
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)
  return model, data, descriptor


def _interpolated_sphere_public_fixture(dof, shell, batch_size=2,
                                        geom_type="sphere"):
  """Build a moving-frictional rigid obstacle against Q1/Q2 flex elements."""
  shell_attr = ' thickness=".01" elastic2d="bend"' if shell else ""
  if geom_type == "sphere":
    geom = ('<geom name="sphere" type="sphere" size=".08" condim="3" '
            'friction=".8 .01 .001" solref=".02 1" '
            'solimp=".9 .95 .001"/>')
  elif geom_type == "capsule":
    geom = ('<geom name="capsule" type="capsule" size=".04 .04" '
            'condim="3" friction=".8 .01 .001" solref=".02 1" '
            'solimp=".9 .95 .001"/>')
  else:
    raise ValueError(f"unsupported test geom type {geom_type!r}")
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep=".001" gravity="0 0 0" cone="elliptic"
                   solver="Newton" iterations="80" tolerance="1e-9"
                   jacobian="sparse"/>
      <worldbody>
        <body name="ball" pos=".1 .1 .1"><freejoint/>{geom}</body>
        <flexcomp name="interp" type="grid" count="3 3 3"
                  pos="0 0 .1" spacing=".1 .1 .1" mass="1" dim="3"
                  dof="{dof}">
          <contact contype="1" conaffinity="0" selfcollide="none"
                   condim="3" friction=".8 .01 .001"
                   solref=".02 1" solimp=".9 .95 .001"/>
          <elasticity young="100" poisson=".2"{shell_attr}/>
        </flexcomp>
      </worldbody>
    </mujoco>
  """)
  qpos = np.tile(np.asarray(model.qpos0, np.float64), (batch_size, 1))
  qvel = np.zeros((batch_size, int(model.nv)), np.float64)
  worlds = []
  for env in range(batch_size):
    qvel[env] = np.linspace(-.006, .008, int(model.nv))
    qvel[env, :6] *= .2
    if env:
      qpos[env, :3] += np.asarray([.004, -.003, .002])
      qvel[env] *= -.61
    mujoco.mj_integratePos(model, qpos[env], qvel[env], .0003*(env+1))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    worlds.append(data)
  return model, qpos.astype(np.float32), qvel.astype(np.float32), worlds


def test_interpolated_sphere_and_cross_pair_routes_are_lowered():
  pytest.importorskip("torch")
  from mujoco_metal.flex_contact import FlexContactProgram

  for dof, shell in (("trilinear", False), ("quadratic", False),
                     ("trilinear", True), ("quadratic", True)):
    model, qpos, qvel, worlds = _interpolated_sphere_public_fixture(dof, shell)
    program = FlexContactProgram(model, batch_size=2, device="cpu")
    assert program._narrowphase_admitted, (dof, shell)
    assert not np.array_equal(qpos[0], qpos[1])
    assert np.linalg.norm(qvel[0]) > 0.0 and np.linalg.norm(qvel[1]) > 0.0
    for data in worlds:
      contacts = [c for c in data.contact[:data.ncon]
                  if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
                  and int(c.elem[1]) >= 0]
      assert len(contacts) == 8
      assert all(int(contact.dim) == 3 for contact in contacts)
      assert np.linalg.norm(data.qfrc_constraint) > 1.0

  capsule_model, _, _, _ = _interpolated_sphere_public_fixture(
      "trilinear", False, geom_type="capsule")
  capsule_program = FlexContactProgram(
      capsule_model, batch_size=2, device="cpu")
  assert not capsule_program._narrowphase_admitted

  cross_model, _, _ = _dynamic_diag_fixture(
      "trilinear", False, True)
  cross_program = FlexContactProgram(cross_model, batch_size=1, device="cpu")
  # The source-order common pair kernel is wired for this descriptor. This
  # checks producer selection only; native row/force/lifecycle qualification
  # remains a separate gate.
  assert cross_program._narrowphase_admitted
  assert cross_program._has_native_flex_pair_candidates


def _cpu_collision_for_float32_detector_inputs(
    model, source_data, flexvert_xpos, geom_xpos, geom_quat):
  """Run pinned collision on the exact float32 geometry sent to Metal.

  The strict flex EPA support path can amplify a sub-ulp change in a vertex
  into a different polytope.  A native detector receives float32 positions,
  while the ordinary ``mj_forward`` reference keeps double positions.  For a
  meaningful geometry oracle, install the actual detector operands into a
  fresh CPU data object and rebuild every dynamic element/BVH bound from those
  operands before calling the pinned collision stages.
  """
  data = mujoco.MjData(model)
  data.qpos[:] = source_data.qpos
  data.qvel[:] = source_data.qvel
  if model.nmocap:
    data.mocap_pos[:] = source_data.mocap_pos
    data.mocap_quat[:] = source_data.mocap_quat
  mujoco.mj_forward(model, data)

  # Metal detector arguments are contiguous float32, so conversion back to
  # mjtNum here is exact for the geometry values passed into its kernels.
  data.flexvert_xpos[:] = np.asarray(flexvert_xpos, np.float32).astype(np.float64)
  data.geom_xpos[:] = np.asarray(geom_xpos, np.float32).astype(np.float64)
  geom_quat = np.asarray(geom_quat, np.float32)
  for geom in range(int(model.ngeom)):
    matrix = np.empty(9, np.float64)
    mujoco.mju_quat2Mat(matrix, geom_quat[geom].astype(np.float64))
    data.geom_xmat[geom] = matrix

  # Match engine_core_smooth.c:640-658 exactly: per-element min/max from the
  # compiled element vertices, center = (max + min)/2, and half extent plus
  # the flex radius. These centers are also consumed by mjc_center during CCD.
  for flex in range(int(model.nflex)):
    dim = int(model.flex_dim[flex])
    elem_adr = int(model.flex_elemdataadr[flex])
    vert_adr = int(model.flex_vertadr[flex])
    global_elem_adr = int(model.flex_elemadr[flex])
    radius = float(model.flex_radius[flex])
    for elem in range(int(model.flex_elemnum[flex])):
      nodes = np.asarray(model.flex_elem[
          elem_adr + elem*(dim+1):elem_adr + (elem+1)*(dim+1)], np.int32)
      points = data.flexvert_xpos[vert_adr + nodes]
      xmin = points[0].copy()
      xmax = points[0].copy()
      for point in points[1:]:
        xmin = np.minimum(xmin, point)
        xmax = np.maximum(xmax, point)
      data.flexelem_aabb[global_elem_adr+elem] = np.concatenate(
          (0.5*(xmax+xmin), 0.5*(xmax-xmin)+radius))

    # engine_core_smooth.c:660-676 copies element bounds into dynamic BVH
    # leaves, then merges parents in reverse order.  Do this before
    # mj_collision so the midphase route uses the same exact input geometry.
    bvh_adr = int(model.flex_bvhadr[flex])
    bvh_num = int(model.flex_bvhnum[flex])
    if bvh_adr < 0:
      continue
    for local in range(bvh_num):
      global_node = bvh_adr + local
      elem = int(model.bvh_nodeid[global_node])
      if elem >= 0:
        data.bvh_aabb_dyn[bvh_adr-int(model.nbvhstatic)+local] = (
            data.flexelem_aabb[global_elem_adr+elem])
    for local in range(bvh_num-1, -1, -1):
      global_node = bvh_adr + local
      if int(model.bvh_nodeid[global_node]) >= 0:
        continue
      child1, child2 = map(int, model.bvh_child[global_node])
      child_aabb1 = data.bvh_aabb_dyn[
          bvh_adr-int(model.nbvhstatic)+child1]
      child_aabb2 = data.bvh_aabb_dyn[
          bvh_adr-int(model.nbvhstatic)+child2]
      lower = np.minimum(child_aabb1[:3]-child_aabb1[3:],
                         child_aabb2[:3]-child_aabb2[3:])
      upper = np.maximum(child_aabb1[:3]+child_aabb1[3:],
                         child_aabb2[:3]+child_aabb2[3:])
      data.bvh_aabb_dyn[bvh_adr-int(model.nbvhstatic)+local] = np.concatenate(
          (0.5*(upper+lower), 0.5*(upper-lower)))

  mujoco.mj_collision(model, data)
  mujoco.mj_makeConstraint(model, data)
  mujoco.mj_projectConstraint(model, data)
  return data


def test_cpu_float32_detector_inputs_can_change_pinned_flex_epa_branch():
  """Demonstrate why an unrounded CPU contact is not a float32-input oracle."""
  xml = """<mujoco><option gravity="0 0 0" cone="elliptic"
                          jacobian="dense"/>
    <worldbody><body pos=".35 .35 .35"><freejoint/>
      <geom type="sphere" size=".08" condim="1"/>
    </body>
    <flexcomp name="interp" type="grid" count="3 3 3"
              pos=".25 .25 .35" spacing=".1 .1 .1" mass="1" dim="3"
              dof="trilinear">
      <contact contype="1" conaffinity="0" selfcollide="none" condim="1"/>
      <elasticity young="100" poisson=".2"/>
    </flexcomp></worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  base = mujoco.MjData(model)
  mujoco.mj_forward(model, base)
  qpos0 = np.asarray(base.qpos, np.float64).copy()
  qpos1 = qpos0.copy()
  qpos1[7:31] += np.random.default_rng(0).uniform(-1.4e-9, 1.4e-9, 24)
  data0 = mujoco.MjData(model)
  data0.qpos[:] = qpos0
  mujoco.mj_forward(model, data0)
  data1 = mujoco.MjData(model)
  data1.qpos[:] = qpos1
  mujoco.mj_forward(model, data1)

  input0 = (data0.flexvert_xpos.astype(np.float32),
            data0.geom_xpos.astype(np.float32),
            data0.geom_xmat.astype(np.float32))
  input1 = (data1.flexvert_xpos.astype(np.float32),
            data1.geom_xpos.astype(np.float32),
            data1.geom_xmat.astype(np.float32))
  assert all(np.array_equal(a, b) for a, b in zip(input0, input1))

  contacts = []
  for data in (data0, data1):
    contacts.append({
        int(contact.elem[1]): float(contact.dist)
        for contact in data.contact[:data.ncon]
        if int(contact.geom[0]) == 0 and int(contact.flex[1]) == 0
        and int(contact.elem[1]) >= 0})
  assert contacts[0].keys() == contacts[1].keys()
  assert max(abs(contacts[0][elem]-contacts[1][elem])
             for elem in contacts[0]) > 12e-6


@pytest.mark.parametrize("shell", [False, True])
def test_cpu_float32_geometry_oracle_rebuilds_pinned_contact_stages(shell):
  model, source_data, _ = _dynamic_diag_fixture(
      "trilinear", shell, False, geom_margin_gap=0.0,
      disable_midphase=False)
  exact = _cpu_collision_for_float32_detector_inputs(
      model, source_data, source_data.flexvert_xpos.astype(np.float32),
      source_data.geom_xpos.astype(np.float32),
      _geom_quat_batch(model, [source_data])[0].astype(np.float32))
  contact_ids = lambda data: [
      int(contact.elem[1]) for contact in data.contact[:data.ncon]
      if int(contact.geom[0]) == 0 and int(contact.flex[1]) == 0
      and int(contact.elem[1]) >= 0]
  assert contact_ids(exact) == contact_ids(source_data)
  exact_by_elem = {
      int(contact.elem[1]): float(contact.dist)
      for contact in exact.contact[:exact.ncon]
      if int(contact.geom[0]) == 0 and int(contact.flex[1]) == 0
      and int(contact.elem[1]) >= 0}
  source_by_elem = {
      int(contact.elem[1]): float(contact.dist)
      for contact in source_data.contact[:source_data.ncon]
      if int(contact.geom[0]) == 0 and int(contact.flex[1]) == 0
      and int(contact.elem[1]) >= 0}
  # This near-degenerate fixture deliberately retains a strict EPA branch
  # whose answer changes when the CPU double position is rounded to the
  # float32 position actually passed to the device.
  assert abs(exact_by_elem[45]-source_by_elem[45]) > 6e-6
  assert exact.nefc == len(contact_ids(exact))
  addresses = [int(contact.efc_address)
               for contact in exact.contact[:exact.ncon]
               if int(contact.geom[0]) == 0 and int(contact.flex[1]) == 0
               and int(contact.elem[1]) >= 0]
  assert np.all(np.isfinite(np.asarray(exact.efc_diagA)[addresses]))


@pytest.mark.parametrize("dof", ["trilinear", "quadratic"])
def test_interpolated_geom_slots_follow_pinned_midphase_element_route(dof):
  """Only flex-BVH leaves are candidates on the pinned body:flex midphase.

  The direct all-to-all route intentionally still visits every element.  This
  catches the distinction with inactive high-layer elements that can overlap
  the geom under a standalone CCD query but are absent from the compiled BVH.
  """
  model, data, _ = _dynamic_diag_fixture(dof, False, False)
  midphase_flag = int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
  body = int(model.geom_bodyid[0])
  assert int(model.body_bvhadr[body]) >= 0
  assert int(model.flex_bvhadr[0]) >= 0

  for use_midphase in (True, False):
    model.opt.disableflags &= ~midphase_flag
    if not use_midphase:
      model.opt.disableflags |= midphase_flag
    mujoco.mj_forward(model, data)
    descriptor = lower_flex_contacts(model)
    slots = ((descriptor.kind == _KIND_GEOM_ELEMENT)
             & (descriptor.geom == 0)
             & (descriptor.flex1 == 0))
    actual_order = descriptor.elem1[slots].astype(int).tolist()
    active = {e for e in range(int(model.flex_elemnum[0]))
              if int(model.flex_elemlayer[
                  int(model.flex_elemadr[0]) + e])
              < int(model.flex_activelayers[0])}
    if use_midphase:
      expected_order = [e for e in _flex_bvh_element_order(
          model, 0, int(model.flex_elemnum[0])) if e in active]
    else:
      expected_order = list(range(int(model.flex_elemnum[0])))
    assert actual_order == expected_order

    cpu_elements = [int(c.elem[1]) for c in data.contact[:data.ncon]
                    if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
                    and int(c.elem[1]) >= 0]
    assert len(cpu_elements) == (8 if use_midphase else 12)
    assert set(cpu_elements).issubset(set(actual_order))
    if use_midphase:
      assert set(cpu_elements) == {36, 38, 39, 41, 44, 45, 46, 47}
      assert not set(actual_order).intersection({37, 40, 42, 43})
    else:
      assert set(cpu_elements) == set(range(36, 48))


def _interpolated_plane_fixture(dof, shell, batch_size=2):
  shell_attr = ' thickness=".01" elastic2d="bend"' if shell else ""
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" timestep=".0002" integrator="Euler">
      <flag multiccd="disable"/>
    </option><worldbody>
      <geom name="floor" type="plane" pos="0 0 0" size="0 0 .1"
            contype="0" conaffinity="1" condim="1"/>
      <flexcomp name="interp" type="grid" count="3 3 3"
                pos="0 0 -.02" spacing=".1 .1 .1" mass="1" dim="3"
                dof="{dof}">
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1"/>
        <elasticity young="100" poisson=".2"{shell_attr}/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  descriptor = lower_flex_contacts(model)
  qpos = np.tile(np.asarray(model.qpos0, np.float64), (batch_size, 1))
  qvel = np.zeros((batch_size, int(model.nv)), np.float64)
  for env in range(batch_size):
    qvel[env] = np.linspace(-.003, .004, int(model.nv))
    if env:
      qvel[env] *= -.63
    mujoco.mj_integratePos(model, qpos[env], qvel[env],
                           float(model.opt.timestep)*(env+1))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    assert any(int(c.flex[1]) == 0 and int(c.vert[1]) >= 0
               for c in data.contact[:data.ncon])
  return model, descriptor, qpos.astype(np.float32), qvel.astype(np.float32)


def _self_flex_contact_fixture(batch_size=2):
  """Build two nonshared self-contacting line-element states."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      <flexcomp name="self" type="grid" count="5 1 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" dim="1" radius=".02">
        <contact contype="1" conaffinity="1" selfcollide="narrow"
                 condim="1"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  descriptor = lower_flex_contacts(model)
  worlds = []
  for world in range(batch_size):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rest_xpos = np.asarray(data.flexvert_xpos, np.float64).reshape(-1, 3).copy()
    # Fold the distal part of the line back across the first segment, with a
    # small out-of-line offset to avoid an ambiguous collinear normal.
    for vertex in range(2, 5):
      body = int(model.flex_vertbodyid[vertex])
      joint = int(model.body_jntadr[body])
      qadr = int(model.jnt_qposadr[joint])
      target = rest_xpos[vertex].copy()
      target[0] -= 0.70 - 0.015 * world
      target[1] += 0.030 + 0.002 * world
      target[2] += 0.010
      data.qpos[qadr:qadr + 3] = target - rest_xpos[vertex]
    data.qvel[:] = np.linspace(-0.025, 0.031, int(model.nv))
    if world:
      data.qvel[:] *= -0.73
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    assert np.linalg.norm(data.cvel) > 0
    worlds.append(data)
  return model, descriptor, worlds


def _cpu_self_flex_relative_jacobian(model, data, contact):
  relative = np.zeros((3, int(model.nv)), dtype=np.float64)
  for side in (0, 1):
    flex = int(contact.flex[side])
    elem = int(contact.elem[side])
    assert flex >= 0 and elem >= 0
    weights = _flex_element_body_weights(
        model, flex, elem, -1, contact.pos, data.flexvert_xpos)
    sign = -1.0 if side == 0 else 1.0
    for body, weight in weights.items():
      jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
      jacr = np.zeros_like(jacp)
      mujoco.mj_jac(model, data, jacp, jacr, contact.pos, int(body))
      relative += sign * float(weight) * jacp
  return relative


def _sphere_triangle_fixture(batch_size=2):
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      <body name="sphere_body" pos=".033 .041 .045">
        <freejoint/>
        <geom name="sphere" type="sphere" size=".07"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="sheet" type="grid" count="2 2 1" pos="0 0 .05"
                spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
        <contact contype="1" conaffinity="0" selfcollide="none" condim="1"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  descriptor = lower_flex_contacts(model)
  sphere = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "sphere"))
  joint = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "sphere_body_free"))
  qadr = int(model.jnt_qposadr[joint])
  vadr = int(model.jnt_dofadr[joint])
  worlds = []
  for world in range(batch_size):
    data = mujoco.MjData(model)
    qpos = np.asarray(model.qpos0, np.float64).copy()
    qvel = np.linspace(-0.02, 0.025, int(model.nv), dtype=np.float64)
    qvel *= 1.0 if world == 0 else -0.71
    mujoco.mj_integratePos(model, qpos, qvel, 0.0002 * (world + 1))
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == sphere and int(c.elem[1]) >= 0]
    assert contacts
    assert np.linalg.norm(data.cvel) > 0
    assert qadr >= 0 and vadr >= 0
    worlds.append(data)
  return model, descriptor, sphere, worlds


def _capsule_triangle_fixture(batch_size=2):
  """Pinned 2-D element versus moving capsule manifold fixture."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      <body name="capsule_body" pos=".033 .041 .045">
        <freejoint/>
        <geom name="capsule" type="capsule" size=".04 .06"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="sheet" type="grid" count="2 2 1" pos="0 0 .05"
                spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
        <contact contype="1" conaffinity="0" selfcollide="none" condim="1"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  descriptor = lower_flex_contacts(model)
  capsule = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "capsule"))
  worlds = []
  for world in range(batch_size):
    data = mujoco.MjData(model)
    qpos = np.asarray(model.qpos0, np.float64).copy()
    qvel = np.linspace(-0.02, 0.025, int(model.nv), dtype=np.float64)
    qvel *= 1.0 if world == 0 else -0.71
    mujoco.mj_integratePos(model, qpos, qvel, 0.0002 * (world + 1))
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == capsule and int(c.elem[1]) >= 0]
    assert contacts
    assert np.linalg.norm(data.cvel) > 0
    worlds.append(data)
  return model, descriptor, capsule, worlds


def _combined_plane_sphere_flex_fixture(batch_size=2):
  """Public-step fixture with concurrent plane and sphere flex contacts."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" timestep=".0002" integrator="Euler">
      <flag multiccd="disable"/>
    </option><worldbody>
      <geom name="floor" type="plane" pos="0 0 .04" size="0 0 .1"
            contype="0" conaffinity="1" condim="1" margin=".02"/>
      <body name="sphere_body" pos=".033 .041 .045">
        <freejoint/>
        <geom name="sphere" type="sphere" size=".07"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="sheet" type="grid" count="2 2 1" pos="0 0 .05"
                spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1" margin=".005"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.tile(np.asarray(model.qpos0, np.float64), (batch_size, 1))
  qvel = np.tile(np.linspace(-0.02, 0.025, int(model.nv), dtype=np.float64),
                 (batch_size, 1))
  for env in range(batch_size):
    qvel[env] *= 1.0 if env == 0 else -0.71
    mujoco.mj_integratePos(model, qpos[env], qvel[env], 0.0002 * (env + 1))
  worlds = []
  for env in range(batch_size):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    assert any(int(c.geom[0]) >= 0 and int(c.elem[1]) >= 0
               for c in data.contact[:data.ncon])
    assert any(int(c.geom[0]) == 0 and int(c.vert[1]) >= 0
               for c in data.contact[:data.ncon])
    assert any(int(c.geom[0]) == 1 and int(c.elem[1]) >= 0
               for c in data.contact[:data.ncon])
    worlds.append(data)
  return model, qpos.astype(np.float32), qvel.astype(np.float32), worlds


def _combined_plane_capsule_flex_fixture(batch_size=2):
  """Public-step fixture with plane and five-witness capsule flex contacts."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" timestep=".0002" integrator="Euler">
    </option><worldbody>
      <geom name="floor" type="plane" pos="0 0 .04" size="0 0 .1"
            contype="0" conaffinity="1" condim="1" margin=".02"/>
      <body name="capsule_body" pos=".033 .041 .045">
        <freejoint/>
        <geom name="capsule" type="capsule" size=".04 .06"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="sheet" type="grid" count="2 2 1" pos="0 0 .05"
                spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1" margin=".005"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.tile(np.asarray(model.qpos0, np.float64), (batch_size, 1))
  qvel = np.tile(np.linspace(-0.02, 0.025, int(model.nv), dtype=np.float64),
                 (batch_size, 1))
  for env in range(batch_size):
    qvel[env] *= 1.0 if env == 0 else -0.71
    mujoco.mj_integratePos(model, qpos[env], qvel[env], 0.0002 * (env + 1))
  worlds = []
  for env in range(batch_size):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    assert any(int(c.geom[0]) == 0 and int(c.vert[1]) >= 0
               for c in data.contact[:data.ncon])
    assert any(int(c.geom[0]) == 1 and int(c.elem[1]) >= 0
               for c in data.contact[:data.ncon])
    worlds.append(data)
  return model, qpos.astype(np.float32), qvel.astype(np.float32), worlds


def _initial_sleeping_sphere_flex_model():
  """Flex contact model whose initially isolated sphere tree is asleep."""
  return mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".0002" gravity="0 0 0">
      <flag sleep="enable"/>
    </option><worldbody>
      <geom name="floor" type="plane" pos="0 0 .04" size="0 0 .1"
            contype="0" conaffinity="1" condim="1" margin=".02"/>
      <body name="sphere_body" pos="1 1 1" sleep="init">
        <freejoint name="sphere_free"/>
        <geom name="sphere" type="sphere" size=".07"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="sheet" type="grid" count="2 2 1" pos="0 0 .05"
                spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
        <contact contype="1" conaffinity="0" selfcollide="none"
                 condim="1" margin=".005"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """)


def _wake_sphere_qpos(model):
  qpos = np.asarray(model.qpos0, np.float64).copy()
  joint = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "sphere_free"))
  adr = int(model.jnt_qposadr[joint])
  qpos[adr:adr+3] = [.033, .041, .045]
  return qpos


def _geom_quat_batch(model, worlds):
  """Convert pinned geom_xmat poses at the explicit CPU/native boundary."""
  quats = np.empty((len(worlds), int(model.ngeom), 4), dtype=np.float64)
  for env, data in enumerate(worlds):
    for geom in range(int(model.ngeom)):
      mujoco.mju_mat2Quat(
          quats[env, geom], np.asarray(data.geom_xmat[geom], np.float64).reshape(9))
  return quats


def _sphere_capsule_fixture(batch_size=2):
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      <body name="sphere_body" pos=".05 .02 0">
        <freejoint/>
        <geom name="sphere" type="sphere" size=".04"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="line" type="grid" count="2 1 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" dim="1" radius=".01">
        <contact contype="1" conaffinity="0" selfcollide="none" condim="1"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  descriptor = lower_flex_contacts(model)
  sphere = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "sphere"))
  worlds = []
  for world in range(batch_size):
    data = mujoco.MjData(model)
    qpos = np.asarray(model.qpos0, np.float64).copy()
    qvel = np.linspace(-0.02, 0.025, int(model.nv), dtype=np.float64)
    qvel *= 1.0 if world == 0 else -0.71
    mujoco.mj_integratePos(model, qpos, qvel, 0.0002 * (world + 1))
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    assert any(int(c.geom[0]) == sphere and int(c.elem[1]) == 0
               for c in data.contact[:data.ncon])
    assert np.linalg.norm(data.cvel) > 0
    worlds.append(data)
  return model, descriptor, sphere, worlds


def _capsule_flex_edge_fixture(batch_size=2):
  """Pinned raw capsule-capsule contacts for one direct flex edge."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" cone="elliptic"/><worldbody>
      <body name="capsule_body" pos=".05 .02 0">
        <freejoint/>
        <geom name="capsule" type="capsule" size=".04 .06"
              contype="0" conaffinity="1" condim="1"/>
      </body>
      <flexcomp name="line" type="grid" count="2 1 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" dim="1" radius=".01">
        <contact contype="1" conaffinity="0" selfcollide="none" condim="1"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  descriptor = lower_flex_contacts(model)
  capsule = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "capsule"))
  worlds = []
  for world in range(batch_size):
    data = mujoco.MjData(model)
    qpos = np.asarray(model.qpos0, np.float64).copy()
    qvel = np.linspace(-0.02, 0.025, int(model.nv), dtype=np.float64)
    qvel *= 1.0 if world == 0 else -0.71
    mujoco.mj_integratePos(model, qpos, qvel, 0.0002 * (world + 1))
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == capsule and int(c.elem[1]) == 0]
    assert contacts
    assert np.linalg.norm(data.cvel) > 0
    worlds.append(data)
  return model, descriptor, capsule, worlds


def test_self_flex_element_pair_contact_side_order_matches_pinned_efc():
  model, descriptor, worlds = _self_flex_contact_fixture()
  matched = 0
  for data in worlds:
    for contact in data.contact[:data.ncon]:
      if (int(contact.flex[0]) != 0 or int(contact.flex[1]) != 0
          or int(contact.elem[0]) < 0 or int(contact.elem[1]) < 0):
        continue
      slots = np.flatnonzero(
          (descriptor.kind == _KIND_ELEMENT_PAIR)
          & (descriptor.flex1 == 0) & (descriptor.elem1 == int(contact.elem[0]))
          & (descriptor.flex2 == 0) & (descriptor.elem2 == int(contact.elem[1])))
      assert slots.size, (contact.flex, contact.elem)
      assert int(contact.elem[0]) < int(contact.elem[1])
      relative = _cpu_self_flex_relative_jacobian(model, data, contact)
      expected = _dense_efc_jacobian(model, data)[int(contact.efc_address)]
      np.testing.assert_allclose(
          np.asarray(contact.frame[:3], np.float64) @ relative,
          expected, rtol=2e-10, atol=2e-11)
      matched += 1
      break
  assert matched == len(worlds)


def test_cpu_sphere_triangle_witnesses_map_to_fixed_element_slots():
  pytest.importorskip("torch")
  model, descriptor, sphere, worlds = _sphere_triangle_fixture()
  program = FlexContactProgram(model, batch_size=len(worlds), device="cpu")
  assert program._narrowphase_admitted
  assert descriptor.slot_count > 0
  geom_quats = _geom_quat_batch(model, worlds)
  assert geom_quats.shape == (len(worlds), int(model.ngeom), 4)
  np.testing.assert_allclose(
      np.linalg.norm(geom_quats, axis=-1), 1.0, rtol=0, atol=2e-15)
  for data in worlds:
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == sphere and int(c.elem[1]) >= 0]
    assert contacts
    for contact in contacts:
      slots = np.flatnonzero(
          (descriptor.kind == _KIND_GEOM_ELEMENT)
          & (descriptor.geom == sphere)
          & (descriptor.flex1 == int(contact.flex[1]))
          & (descriptor.elem1 == int(contact.elem[1])))
      assert slots.size == 1
      assert int(contact.vert[1]) == -1
      assert np.isfinite(contact.dist)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="sphere/triangle production detector requires GPU opt-in")
def test_native_sphere_triangle_detector_matches_pinned_contacts_B2():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("sphere/triangle production detector requires MPS")
  model, descriptor, sphere, worlds = _sphere_triangle_fixture()
  program = FlexContactProgram(model, batch_size=len(worlds), device="mps")
  assert program._narrowphase_admitted
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  batch = len(worlds)
  nv = int(model.nv)
  result = program.run_device(
      mps(np.stack([data.flexvert_xpos for data in worlds])),
      mps(np.stack([data.geom_xpos for data in worlds])),
      mps(_geom_quat_batch(model, worlds)),
      flexvert_spatial_jacobian=mps(np.zeros(
          (batch, int(model.nflexvert), 6, nv), np.float32)),
      cdof=mps(np.stack([np.asarray(data.cdof).reshape(nv, 6)
                         for data in worlds])),
      root_com=mps(np.stack([data.subtree_com for data in worlds])),
      qvel=mps(np.stack([data.qvel for data in worlds])),
      include_wake_links=False)
  active = result["active"].cpu().numpy()
  distances = result["dist"].cpu().numpy()
  positions = result["pos"].cpu().numpy()
  normals = result["normal"].cpu().numpy()
  barycentric = result["barycentric1"].cpu().numpy()
  row_J = result["workspace_J"].cpu().numpy()
  row_R = result["R"].cpu().numpy()
  row_aref = result["aref"].cpu().numpy()
  row_active = result["row_active"].cpu().numpy()
  ccd_trace = result["ccd_trace"].cpu().numpy()
  diagA = program._diag_approx.cpu().numpy()
  for env, data in enumerate(worlds):
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == sphere and int(c.elem[1]) >= 0]
    assert contacts
    expected_slots = set()
    for contact in contacts:
      slots = np.flatnonzero(
          (descriptor.kind == _KIND_GEOM_ELEMENT)
          & (descriptor.geom == sphere)
          & (descriptor.flex1 == int(contact.flex[1]))
          & (descriptor.elem1 == int(contact.elem[1])))
      assert slots.size == 1
      slot = int(slots[0])
      expected_slots.add(slot)
      assert active[env, slot]
      assert row_active[env, int(descriptor.row_start[slot])]
      np.testing.assert_allclose(
          distances[env, slot], float(contact.dist), rtol=0, atol=3e-6)
      np.testing.assert_allclose(
          positions[env, slot], contact.pos, rtol=0, atol=3e-6)
      np.testing.assert_allclose(
          normals[env, slot], contact.frame[:3], rtol=0, atol=3e-5)
      weights = barycentric[env, slot, :3]
      np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=2e-5)
      nodes = np.asarray(descriptor.nodes1[slot, :3], np.int32)
      expected_surface = (np.asarray(contact.pos, np.float64)
                          + np.asarray(contact.frame[:3], np.float64)
                          * (float(model.flex_radius[0])
                             + 0.5 * float(contact.dist)))
      actual_surface = weights @ np.asarray(data.flexvert_xpos[nodes], np.float64)
      np.testing.assert_allclose(
          actual_surface, expected_surface, rtol=0, atol=4e-5)
      row = int(descriptor.row_start[slot])
      pinned_J = _dense_efc_jacobian(model, data)[int(contact.efc_address)]
      np.testing.assert_allclose(
          row_J[env, row], pinned_J, rtol=5e-5, atol=5e-6)
      np.testing.assert_allclose(
          diagA[env, row], float(data.efc_diagA[int(contact.efc_address)]),
          rtol=3e-5, atol=3e-6)
      np.testing.assert_allclose(
          row_R[env, row], float(data.efc_R[int(contact.efc_address)]),
          rtol=8e-5, atol=2e-6)
    np.testing.assert_allclose(
        row_aref[env, row], float(data.efc_aref[int(contact.efc_address)]),
        rtol=8e-5, atol=2e-5)
    assert set(np.flatnonzero(active[env]).tolist()) == expected_slots


def test_cpu_sphere_flex_edge_contact_matches_fixed_slot():
  pytest.importorskip("torch")
  model, descriptor, sphere, worlds = _sphere_capsule_fixture()
  assert int(model.flex_dim[0]) == 1 and int(model.flex_interp[0]) == 0
  program = FlexContactProgram(model, batch_size=len(worlds), device="cpu")
  assert program._narrowphase_admitted
  for data in worlds:
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == sphere and int(c.elem[1]) == 0]
    assert len(contacts) == 1
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_GEOM_ELEMENT)
        & (descriptor.geom == sphere) & (descriptor.flex1 == 0)
        & (descriptor.elem1 == 0))
    assert slots.size == 1


def test_cpu_capsule_flex_edge_contacts_reserve_exact_raw_ordinals():
  pytest.importorskip("torch")
  model, descriptor, capsule, worlds = _capsule_flex_edge_fixture()
  assert int(model.flex_dim[0]) == 1 and int(model.flex_interp[0]) == 0
  program = FlexContactProgram(model, batch_size=len(worlds), device="cpu")
  assert program._narrowphase_admitted
  slots = np.flatnonzero(
      (descriptor.kind == _KIND_GEOM_ELEMENT)
      & (descriptor.geom == capsule) & (descriptor.flex1 == 0)
      & (descriptor.elem1 == 0))
  assert len(slots) == 4
  assert sorted(descriptor.contact_ordinal[slots].tolist()) == [0, 1, 2, 3]
  counts = []
  for data in worlds:
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == capsule and int(c.elem[1]) == 0]
    counts.append(len(contacts))
    assert 1 <= len(contacts) <= 4
    assert all(np.isfinite(float(c.dist)) and np.isfinite(c.pos).all()
               for c in contacts)
  assert counts[0] > 0 and counts[1] > 0


def test_cpu_capsule_triangle_raw_manifold_maps_to_fixed_ordinals():
  pytest.importorskip("torch")
  model, descriptor, capsule, worlds = _capsule_triangle_fixture()
  assert int(model.flex_dim[0]) == 2 and int(model.flex_interp[0]) == 0
  program = FlexContactProgram(model, batch_size=len(worlds), device="cpu")
  # Native composed candidate/row parity passed for this direct capsule
  # route. The public simulation trajectory remains a separate integration gate.
  assert program._narrowphase_admitted
  per_element_slots = {}
  for slot, (kind, geom, element, ordinal) in enumerate(zip(
      descriptor.kind, descriptor.geom, descriptor.elem1,
      descriptor.contact_ordinal)):
    if (int(kind) == _KIND_GEOM_ELEMENT and int(geom) == capsule
        and int(element) >= 0):
      per_element_slots.setdefault(int(element), []).append((int(ordinal), slot))
  assert per_element_slots
  assert all(sorted(ordinal for ordinal, _ in slots) == list(range(5))
             for slots in per_element_slots.values())
  counts = []
  for data in worlds:
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == capsule and int(c.elem[1]) >= 0]
    by_element = {}
    for contact in contacts:
      by_element.setdefault(int(contact.elem[1]), []).append(contact)
    assert set(by_element).issubset(per_element_slots)
    counts.append(sum(map(len, by_element.values())))
    for element, element_contacts in by_element.items():
      slots = sorted(per_element_slots[element])
      assert len(element_contacts) <= len(slots) == 5
      assert all(np.isfinite(float(contact.dist)) for contact in element_contacts)
      assert all(np.isfinite(np.asarray(contact.pos)).all()
                 for contact in element_contacts)
  assert counts[0] > 0 and counts[1] > 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="capsule/triangle detector requires GPU opt-in")
def test_native_capsule_triangle_manifold_matches_pinned_B2():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("capsule/triangle detector requires MPS")
  model, descriptor, capsule, worlds = _capsule_triangle_fixture()
  program = FlexContactProgram(model, batch_size=len(worlds), device="mps")
  assert not program._narrowphase_admitted
  # Stage qualification keeps the constructor gate closed; only this test
  # instance reaches the native source-composed detector and row builder.
  program._narrowphase_admitted = True
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  batch, nv = len(worlds), int(model.nv)
  result = program.run_device(
      mps(np.stack([data.flexvert_xpos for data in worlds])),
      mps(np.stack([data.geom_xpos for data in worlds])),
      mps(_geom_quat_batch(model, worlds)),
      flexvert_spatial_jacobian=mps(np.zeros(
          (batch, int(model.nflexvert), 6, nv), np.float32)),
      cdof=mps(np.stack([np.asarray(data.cdof).reshape(nv, 6)
                         for data in worlds])),
      root_com=mps(np.stack([data.subtree_com for data in worlds])),
      qvel=mps(np.stack([data.qvel for data in worlds])),
      include_wake_links=False)
  active = result["active"].cpu().numpy()
  pos = result["pos"].cpu().numpy()
  normal = result["normal"].cpu().numpy()
  dist = result["dist"].cpu().numpy()
  bary = result["barycentric1"].cpu().numpy()
  row_J = result["workspace_J"].cpu().numpy()
  row_R = result["R"].cpu().numpy()
  row_aref = result["aref"].cpu().numpy()
  row_active = result["row_active"].cpu().numpy()
  diagA = program._diag_approx.cpu().numpy()
  for env, data in enumerate(worlds):
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == capsule and int(c.elem[1]) >= 0]
    assert contacts
    ordinal_by_element = {}
    expected_slots = set()
    for contact in contacts:
      element = int(contact.elem[1])
      ordinal = ordinal_by_element.get(element, 0)
      ordinal_by_element[element] = ordinal + 1
      found = np.flatnonzero(
          (descriptor.kind == _KIND_GEOM_ELEMENT)
          & (descriptor.geom == capsule) & (descriptor.flex1 == 0)
          & (descriptor.elem1 == element)
          & (descriptor.contact_ordinal == ordinal))
      assert found.size == 1, (element, ordinal)
      slot = int(found[0])
      expected_slots.add(slot)
      assert active[env, slot]
      row = int(descriptor.row_start[slot])
      assert row_active[env, row]
      np.testing.assert_allclose(dist[env, slot], float(contact.dist),
                                 rtol=0, atol=4e-6)
      np.testing.assert_allclose(pos[env, slot], contact.pos,
                                 rtol=0, atol=4e-6)
      np.testing.assert_allclose(normal[env, slot], contact.frame[:3],
                                 rtol=0, atol=4e-5)
      weights = bary[env, slot, :3]
      np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=3e-5)
      nodes = np.asarray(descriptor.nodes1[slot, :3], np.int32)
      surface = weights @ np.asarray(data.flexvert_xpos[nodes], np.float64)
      expected_surface = (np.asarray(contact.pos, np.float64)
                          + np.asarray(contact.frame[:3], np.float64)
                          * (float(model.flex_radius[0])
                             + 0.5 * float(contact.dist)))
      np.testing.assert_allclose(surface, expected_surface,
                                 rtol=0, atol=6e-5)
      pinned_J = _dense_efc_jacobian(model, data)[int(contact.efc_address)]
      np.testing.assert_allclose(row_J[env, row], pinned_J,
                                 rtol=6e-5, atol=6e-6)
      np.testing.assert_allclose(
          diagA[env, row], float(data.efc_diagA[int(contact.efc_address)]),
          rtol=3e-5, atol=3e-6)
      np.testing.assert_allclose(
          row_R[env, row], float(data.efc_R[int(contact.efc_address)]),
          rtol=8e-5, atol=2e-6)
    np.testing.assert_allclose(
        row_aref[env, row], float(data.efc_aref[int(contact.efc_address)]),
        rtol=8e-5, atol=2e-5)
    assert set(np.flatnonzero(active[env]).tolist()) == expected_slots


def test_cpu_public_plane_sphere_flex_fixture_has_both_contact_families():
  model, qpos, qvel, worlds = _combined_plane_sphere_flex_fixture()
  from mujoco_metal.stepping import validate_stepping_profile
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert profile.execution_plan.is_stage_enabled("coupled_constraints")
  assert qpos.shape == (2, model.nq) and qvel.shape == (2, model.nv)
  assert all(any(int(c.geom[0]) == 0 and int(c.vert[1]) >= 0
                 for c in data.contact[:data.ncon]) for data in worlds)
  assert all(any(int(c.geom[0]) == 1 and int(c.elem[1]) >= 0
                 for c in data.contact[:data.ncon]) for data in worlds)


def test_coupled_flex_contact_allocation_uses_canonical_candidate_row_capacity():
  model, _, _, _ = _combined_plane_sphere_flex_fixture(batch_size=1)
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.flex_contact import lower_flex_contacts
  flex_rows = lower_flex_contacts(model)
  coupled = lower_coupled_constraints(model)
  assert coupled.n_flex_contact_rows == flex_rows.row_capacity
  assert coupled.flex_contact_base >= coupled.nr_joint
  assert coupled.flex_contact_base == coupled.nr - flex_rows.row_capacity
  assert coupled.nr == (coupled.flex_contact_base + flex_rows.row_capacity)


def test_cpu_public_plane_capsule_flex_fixture_has_both_contact_families():
  pytest.importorskip("torch")
  model, qpos, qvel, worlds = _combined_plane_capsule_flex_fixture()
  from mujoco_metal.stepping import validate_stepping_profile
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert profile.execution_plan.is_stage_enabled("coupled_constraints")
  from mujoco_metal.flex_contact import FlexContactProgram
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  assert program._narrowphase_admitted
  assert qpos.shape == (2, model.nq) and qvel.shape == (2, model.nv)
  assert all(any(int(c.geom[0]) == 0 and int(c.vert[1]) >= 0
                 for c in data.contact[:data.ncon]) for data in worlds)
  assert all(any(int(c.geom[0]) == 1 and int(c.elem[1]) >= 0
                 for c in data.contact[:data.ncon]) for data in worlds)


def test_cpu_flex_contact_link_connects_sleeping_obstacle_tree():
  model = _initial_sleeping_sphere_flex_model()
  sphere_body = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, "sphere_body"))
  sphere_tree = int(model.body_treeid[sphere_body])
  initial = mujoco.MjData(model)
  assert int(initial.tree_asleep[sphere_tree]) == 0
  descriptor = lower_flex_contacts(model)
  flex_trees = {int(model.body_treeid[int(body)])
                for body in np.asarray(model.flex_vertbodyid).reshape(-1)}
  assert sphere_tree >= 0 and flex_trees
  assert any(sphere_tree in (int(a), int(b))
             and (int(b) if int(a) == sphere_tree else int(a)) in flex_trees
             for a, b in descriptor.link_tree_pairs)

  target = _wake_sphere_qpos(model)
  initial.qpos[:] = target
  initial.qvel[:] = 0.0
  mujoco.mj_step(model, initial)
  assert any(int(c.geom[0]) == 1 and int(c.elem[1]) >= 0
             for c in initial.contact[:initial.ncon])
  assert int(initial.tree_asleep[sphere_tree]) == -10


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public flex contact trajectory requires GPU opt-in")
def test_native_public_plane_sphere_flex_cc_trajectory_B2():
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, _ = _combined_plane_sphere_flex_fixture()
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  # The public step must construct/use the unmodified admitted producer and
  # the canonical CC row slice; no test-only gate override is involved.
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  actual = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(actual.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public sparse flex Jacobian requires GPU opt-in")
def test_native_public_sparse_plane_sphere_flex_cc_rows_B2():
  """Exercise the fused flex→CSR writer, including gap-row zeroing."""
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, _ = _combined_plane_sphere_flex_fixture()
  model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  assert mujoco.mj_isSparse(model)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
  cc = sim._coupled_constraints
  assert cc is not None and cc._jacobian_layout.mode == 1
  assert cc._workspace["contact_jacobian"].numel() == 1
  assert cc._workspace["position_cache_contact_jacobian"].numel() == 1
  assert cc._jacobian_pattern.nnz < cc.descriptor.nr * cc.descriptor.nv

  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)

  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  assert program._geom_contact_spatial_jacobian.numel() == 1
  assert program._row_workspace.sparse_only
  assert program._row_workspace.workspace_J.numel() == 1
  actual = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(actual.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)

  canonical = cc._assembly_views(include_optimizer_outputs=False)
  result = canonical["flex_contact_result"]
  flex_rows = canonical["flex_contact_rows"]
  live = result["active"].detach().cpu().numpy()
  distance = result["dist"].detach().cpu().numpy()
  row_live = flex_rows["active"].detach().cpu().numpy() > .5
  desc = cc.descriptor.flex_contact_descriptor
  margin = np.asarray(desc.margin, dtype=np.float32)
  global_start = int(cc.descriptor.flex_contact_base)
  packed_dense = cc.materialize_jacobian().detach().cpu().numpy()
  for slot, (start, span) in enumerate(zip(desc.row_start, desc.row_span)):
    first = global_start + int(start)
    stop = first + int(span)
    for env in range(2):
      expected = bool(live[env, slot] and distance[env, slot] < margin[slot])
      assert np.all(row_live[env, int(start):int(start) + int(span)] == expected)
      if not expected:
        assert np.all(packed_dense[env, first:stop] == 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public flex capsule trajectory requires GPU opt-in")
def test_native_public_plane_capsule_flex_cc_trajectory_B2():
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, _ = _combined_plane_capsule_flex_fixture()
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  actual = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(actual.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public flex-edge/capsule trajectory requires GPU opt-in")
def test_native_public_capsule_flex_edge_cc_trajectory_B2():
  """Compare fixed four-witness edge/capsule slots through public CC stepping."""
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, _, capsule, worlds = _capsule_flex_edge_fixture()
  qpos = np.stack([np.asarray(data.qpos, np.float32) for data in worlds])
  qvel = np.stack([np.asarray(data.qvel, np.float32) for data in worlds])
  sim = MetalSimulation(model, batch_size=len(worlds), qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  refs = [mujoco.MjData(model) for _ in worlds]
  for ref, source in zip(refs, worlds):
    ref.qpos[:] = source.qpos
    ref.qvel[:] = source.qvel
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for ref in refs:
      mujoco.mj_step(model, ref)
  assert sim._flex._contact_program._narrowphase_admitted
  actual = sim.state.snapshot()
  for env, ref in enumerate(refs):
    np.testing.assert_allclose(actual.qpos[env], ref.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[env], ref.qvel,
                               rtol=2e-4, atol=2e-5)
  # Keep the regression tied to the four compiled raw ordinals for the
  # capsule-capsule manifold, even when a given state emits fewer contacts.
  descriptor = sim._flex._contact_program.descriptor
  slots = np.flatnonzero((descriptor.geom == capsule)
                         & (descriptor.kind == _KIND_GEOM_ELEMENT))
  assert slots.size == 4


@pytest.mark.gpu
@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="interpolated flex-plane trajectory requires GPU opt-in")
def test_native_public_interpolated_flex_plane_cc_trajectory_B2(dof, shell):
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, _, qpos, qvel = _interpolated_plane_fixture(dof, shell)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
  # Capture the retained canonical rows before stepping so a state/solver
  # discrepancy can be localized to producer rows versus the primal solve.
  system = sim.assembled_system(recompute=True)
  cc_desc = sim._coupled_constraints.descriptor
  contact_desc = cc_desc.flex_contact_descriptor
  row_base = int(cc_desc.flex_contact_base)
  flex_bundle = sim._coupled_constraints._flex_contact_current
  assert flex_bundle is not None
  flex_candidates = flex_bundle["contact_result"]
  flex_rows = flex_bundle["rows"]
  global_j = sim._coupled_constraints.materialize_jacobian()
  assert tuple(global_j.shape) == (2, sim._coupled_constraints.descriptor.nr, model.nv)
  sparse = sim._coupled_constraints._jacobian_layout.mode == 1
  if sparse:
    assert flex_rows.get("workspace_J") is None
    assert flex_candidates.get("relative_spatial_jacobian") is None
    assert sim._coupled_constraints._workspace["contact_jacobian"].numel() == 1
  # Verify the exact production detector inputs before diagnosing candidate
  # selection or contact rows. This distinguishes an FK/flex-vertex stage
  # mismatch from a narrowphase failure without changing production buffers.
  native_fk = sim._smooth._fk.run_device(sim._state._qpos)
  native_flex_xpos = (sim._flex._contact_program
                      ._diag_current_flexvert_xpos.detach().cpu().numpy())
  native_geom_pos = native_fk["geom_pos"].detach().cpu().numpy()
  native_geom_quat = native_fk["geom_quat"].detach().cpu().numpy()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(
        native_flex_xpos[env],
        np.asarray(data.flexvert_xpos, np.float32).reshape(-1, 3),
        rtol=2e-5, atol=2e-6,
        err_msg=f"flex detector vertex input differs from pinned FK, env={env}")
    np.testing.assert_allclose(
        native_geom_pos[env], np.asarray(data.geom_xpos, np.float32),
        rtol=2e-5, atol=2e-6,
        err_msg=f"flex detector geom position input differs from pinned FK, env={env}")
    pinned_geom_quat = np.empty((int(model.ngeom), 4), np.float64)
    for geom in range(int(model.ngeom)):
      mujoco.mju_mat2Quat(
          pinned_geom_quat[geom],
          np.asarray(data.geom_xmat[geom], np.float64).reshape(9))
    np.testing.assert_allclose(
        native_geom_quat[env], pinned_geom_quat.astype(np.float32),
        rtol=2e-5, atol=2e-6,
        err_msg=f"flex detector geom quaternion input differs from pinned FK, env={env}")
    pinned_J = _dense_efc_jacobian(model, data)
    for contact in data.contact[:data.ncon]:
      if int(contact.flex[1]) < 0 or int(contact.vert[1]) < 0:
        continue
      matches = np.flatnonzero(
          (contact_desc.kind == _KIND_PLANE_VERTEX)
          & (contact_desc.vert1 == int(contact.vert[1])))
      assert matches.size == 1
      slot = int(matches[0])
      local_row = int(contact_desc.row_start[slot])
      row = row_base + local_row
      pinned_contact = next(
          c for c in data.contact[:data.ncon]
          if int(c.flex[1]) == 0 and int(c.vert[1]) >= 0
          and int(c.vert[1]) == int(contact_desc.vert1[slot]))
      pinned_vertex = (int(model.flex_vertadr[int(pinned_contact.flex[1])])
                       + int(pinned_contact.vert[1]))
      pinned_weights = _flex_vertex_body_weights(
          model, int(pinned_contact.flex[1]), pinned_vertex)
      # Keep intermediate production stages visible in a failure: an all-zero
      # public row can originate in source body-weight J, normal projection,
      # local row assembly, or global CC scatter.  These checks distinguish
      # those stages without using the CPU oracle to repair device values.
      # Packed execution intentionally has no candidate dense spatial-J or
      # local dense row tensor. Compare its actual canonical row to the
      # independent CPU row below. Retain the original intermediate/scatter
      # checks when a legacy dense producer publishes those products.
      legacy_spatial = flex_candidates.get("relative_spatial_jacobian")
      legacy_local = flex_rows.get("workspace_J")
      if legacy_spatial is not None and legacy_local is not None:
        relative = legacy_spatial[env, slot]
        frame = flex_candidates["frame"][env, slot]
        expected_local = torch.einsum("i,ij->j", frame[0], relative[:3])
        actual_local = legacy_local[env, local_row]
        np.testing.assert_allclose(
            actual_local.detach().cpu().numpy(),
            expected_local.detach().cpu().numpy(), rtol=5e-5, atol=5e-6)
        np.testing.assert_array_equal(
            global_j[env, row].detach().cpu().numpy(),
            actual_local.detach().cpu().numpy())
      row_enabled = flex_rows["active"][env, local_row]
      scheduler = sim._sleep_schedule
      awake_context = ("none" if scheduler is None else
                       scheduler.tree_awake.detach().cpu().numpy())
      status_context = system.get("status")
      if status_context is not None:
        status_context = int(status_context[env].item())
      stage_context = (
          f"dof={dof}, shell={shell}, env={env}, slot={slot}, "
          f"kind={contact_desc.kind[slot]}, flex1={contact_desc.flex1[slot]}, "
          f"elem1={contact_desc.elem1[slot]}, vert1={contact_desc.vert1[slot]}, "
          f"flex2={contact_desc.flex2[slot]}, elem2={contact_desc.elem2[slot]}, "
          f"vert2={contact_desc.vert2[slot]}, geom={contact_desc.geom[slot]}, "
          f"nodes1={contact_desc.nodes1[slot].tolist()}, "
          f"nodes2={contact_desc.nodes2[slot].tolist()}, "
          f"pinned_body_weights={pinned_weights}, "
          f"active={flex_candidates['active'][env, slot].item()}, "
          f"raw_active={flex_candidates['raw_active'][env, slot].item()}, "
          f"dist={flex_candidates['dist'][env, slot].item()}, "
          f"pos={flex_candidates['pos'][env, slot].detach().cpu().numpy()}, "
          f"global_row_norm={torch.linalg.vector_norm(global_j[env, row]).item()}, "
          f"row_enabled={row_enabled.item()}, status={status_context}, "
          f"tree_awake={awake_context}, "
          f"pose_generation={sim._flex._kinematics_generation}, "
          f"component_mass={sim._component_mass_enabled}, "
          f"nv={model.nv}, row_capacity={contact_desc.row_capacity}")
      assert bool(flex_candidates["active"][env, slot].item()), stage_context
      address = int(contact.efc_address)
      np.testing.assert_allclose(
          global_j[env, row].detach().cpu().numpy(), pinned_J[address],
          rtol=5e-5, atol=5e-6,
          err_msg="public pinned J mismatch: " + stage_context)
      np.testing.assert_allclose(
          system["R"][env, row].detach().cpu().numpy(),
          float(data.efc_R[address]), rtol=8e-5, atol=2e-6)
      np.testing.assert_allclose(
          system["ar"][env, row].detach().cpu().numpy(),
          float(data.efc_aref[address]), rtol=8e-5, atol=2e-5)
  initial = sim.snapshot()
  middle = None
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
    if _ == 0:
      middle = sim.snapshot()
      accepted = sim.accepted_step["system"]
      for env, data in enumerate(refs):
        np.testing.assert_allclose(
            accepted["qfrc_constraint"][env].detach().cpu().numpy(),
            data.qfrc_constraint, rtol=5e-4, atol=2e-5)
  assert sim._flex._contact_program is not None
  assert sim._flex._contact_program._narrowphase_admitted
  # The scalable profile must exercise the component mass path for these
  # model-derived NV/row capacities; a legacy dense mass solve would not
  # qualify the large interpolated fixtures that exposed the profile ceiling.
  system = sim.assembled_system(recompute=True)
  assert system["mass_matrix"] is None
  assert system["mass_blocks"] is not None
  assert system["mass_block_layout"] is not None
  actual = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(actual.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)

  # Exercise the same full-size state through ordinary checkpoint replay and
  # reset. No smaller model, rows, contact counts or solver budgets are used.
  assert middle is not None
  sim.restore(middle)
  for _ in range(2):
    np.testing.assert_array_equal(sim.step().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, actual.qpos)
  np.testing.assert_array_equal(replay.qvel, actual.qvel)
  sim.restore(initial)
  sim.reset(qpos=qpos, qvel=qvel)
  reset_refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(reset_refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_step(model, data)
  np.testing.assert_array_equal(sim.step().cpu().numpy(), [0, 0])
  reset_state = sim.state.snapshot()
  for env, data in enumerate(reset_refs):
    np.testing.assert_allclose(reset_state.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(reset_state.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public interpolated flex-sphere trajectory requires GPU opt-in")
def test_native_public_interpolated_sphere_flex_cc_friction_trajectory_B2(
    dof, shell):
  """Exercise admitted Q1/Q2 sphere CCD through sparse CC and replay."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, _initial_worlds = (
      _interpolated_sphere_public_fixture(dof, shell))
  sphere = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "sphere"))
  refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(refs):
    # Match the exact float32 state passed into the native simulator.
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == sphere and int(c.flex[1]) == 0
                and int(c.elem[1]) >= 0]
    assert len(contacts) == 8
    assert all(int(contact.dim) == 3 for contact in contacts)
    assert np.linalg.norm(data.qfrc_constraint) > 1.0

  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  # The native detector is lazily prepared by the first public assembly.
  system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  assert sim._coupled_constraints._jacobian_layout.mode == 1
  assert sim._coupled_constraints._workspace["contact_jacobian"].numel() == 1
  assert program._geom_contact_spatial_jacobian.numel() == 1

  # Public assembly has to retain all candidate rows and both moving bodies.
  bundle = sim._coupled_constraints._flex_contact_current
  assert bundle is not None
  contact_result, rows = bundle["contact_result"], bundle["rows"]
  desc = sim._coupled_constraints.descriptor.flex_contact_descriptor
  row_base = int(sim._coupled_constraints.descriptor.flex_contact_base)
  active = contact_result["active"].detach().cpu().numpy()
  relative = contact_result.get("relative_spatial_jacobian")
  assert relative is None  # actual canonical CSR omits dense candidate J
  dense_contact_j = sim._coupled_constraints.materialize_jacobian()
  for env, data in enumerate(refs):
    pinned_j = _dense_efc_jacobian(model, data)
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == sphere and int(c.flex[1]) == 0
                and int(c.elem[1]) >= 0]
    expected_slots = set()
    for contact in contacts:
      matches = np.flatnonzero(
          (desc.kind == _KIND_GEOM_ELEMENT) & (desc.geom == sphere)
          & (desc.flex1 == 0) & (desc.elem1 == int(contact.elem[1])))
      assert matches.size == 1
      slot = int(matches[0])
      expected_slots.add(slot)
      start, span = int(desc.row_start[slot]), int(desc.row_span[slot])
      address = int(contact.efc_address)
      assert span == int(contact.dim) == 3
      np.testing.assert_allclose(
          dense_contact_j[env, row_base+start:row_base+start+span]
              .detach().cpu().numpy(),
          pinned_j[address:address+span], rtol=5e-5, atol=5e-6)
      np.testing.assert_allclose(
          system["R"][env, row_base+start:row_base+start+span]
              .detach().cpu().numpy(),
          data.efc_R[address:address+span], rtol=8e-5, atol=2e-6)
      np.testing.assert_allclose(
          system["ar"][env, row_base+start:row_base+start+span]
              .detach().cpu().numpy(),
          data.efc_aref[address:address+span], rtol=8e-5, atol=2e-5)
      assert np.all(rows["active"][env, start:start+span]
                    .detach().cpu().numpy() > .5)
      # Nonzero canonical rows are checked against independent CPU values;
      # they cannot pass by emitting an all-zero packed row.
      assert np.linalg.norm(pinned_j[address:address+span]) > 1e-4
      assert torch.linalg.vector_norm(
          dense_contact_j[env, row_base+start:row_base+start+span]).item() > 1e-4
    assert set(np.flatnonzero(active[env]).tolist()) == expected_slots

  initial_snapshot = sim.snapshot()
  middle_snapshot = None
  first_accepted_force = None
  for step in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
    if step == 0:
      first_accepted_force = (
          sim.accepted_step["system"]["qfrc_constraint"].detach().cpu().numpy())
      for env, data in enumerate(refs):
        np.testing.assert_allclose(first_accepted_force[env],
                                   data.qfrc_constraint,
                                   rtol=5e-4, atol=2e-5)
    if step == 1:
      middle_snapshot = sim.snapshot()
  assert middle_snapshot is not None
  final = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(final.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(final.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)

  sim.restore(middle_snapshot)
  for _ in range(2):
    np.testing.assert_array_equal(sim.step().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, final.qpos)
  np.testing.assert_array_equal(replay.qvel, final.qvel)

  sim.restore(initial_snapshot)
  sim.reset(qpos=qpos, qvel=qvel)
  reset_refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(reset_refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_step(model, data)
  np.testing.assert_array_equal(sim.step().cpu().numpy(), [0, 0])
  reset_step = sim.state.snapshot()
  for env, data in enumerate(reset_refs):
    np.testing.assert_allclose(reset_step.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(reset_step.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="flex contact sleep/wake requires GPU opt-in")
def test_native_flex_contact_wakes_initially_sleeping_tree():
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model = _initial_sleeping_sphere_flex_model()
  sphere_body = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, "sphere_body"))
  sphere_tree = int(model.body_treeid[sphere_body])
  target = _wake_sphere_qpos(model).astype(np.float32)
  sim = MetalSimulation(model, batch_size=1,
                        profile="integrated_scalable_v1")
  scheduler = sim._sleep_schedule
  assert int(scheduler.tree_state[0, sphere_tree].item()) == 0

  # Place the native state into the predicted collision after it has acquired
  # its compiled initial sleep state. The next public position prepass must
  # detect the real flex-sphere candidate and wake the independent sphere tree.
  sim._state._qpos[0].copy_(torch.as_tensor(target, device="mps"))
  sim._state._qvel.zero_()
  ref = mujoco.MjData(model)
  ref.qpos[:] = target
  ref.qvel[:] = 0.0
  mujoco.mj_step(model, ref)
  status = sim.step()
  np.testing.assert_array_equal(status.cpu().numpy(), [0])
  assert int(scheduler.tree_awake[0, sphere_tree].item()) == 1
  assert int(scheduler.tree_state[0, sphere_tree].item()) == -10

  result = sim._coupled_constraints._flex_contact_current
  assert result is not None
  contact_result = result["contact_result"]
  active = contact_result["active"].cpu().numpy()[0]
  sphere = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "sphere"))
  sphere_slots = np.flatnonzero(
      (sim._flex._contact_program.descriptor.geom == sphere)
      & (sim._flex._contact_program.descriptor.kind == _KIND_GEOM_ELEMENT))
  assert sphere_slots.size and active[sphere_slots].any()

  actual = sim.state.snapshot()
  np.testing.assert_allclose(actual.qpos[0], ref.qpos, rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(actual.qvel[0], ref.qvel, rtol=2e-4, atol=2e-5)


def test_contact_result_keeps_candidate_mask_separate_from_solver_row_mask():
  pytest.importorskip("torch")
  model, descriptor, _, _ = _sphere_triangle_fixture(batch_size=1)
  program = FlexContactProgram(model, batch_size=1, device="cpu")
  candidate_active = np.asarray([[True, False]], dtype=bool)
  solver_row_active = np.asarray([[True, True, False, False]], dtype=np.int32)
  workspace = object()
  program.run_spatial_rows_device = lambda *args, **kwargs: {
      "active": solver_row_active, "workspace_J": workspace}
  result = program._finish_contact_pipeline(
      {"active": candidate_active}, object(), object(), object(), object(),
      None, None, False, None)
  assert result["active"] is candidate_active
  assert result["row_active"] is solver_row_active
  assert result["workspace_J"] is workspace
  assert descriptor.slot_count != solver_row_active.shape[1]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="sphere/capsule detector requires GPU opt-in")
def test_native_sphere_capsule_detector_and_rows_match_pinned_B2():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("sphere/capsule detector requires MPS")
  model, descriptor, sphere, worlds = _sphere_capsule_fixture()
  program = FlexContactProgram(model, batch_size=len(worlds), device="mps",
                               capture_ccd_trace=True)
  assert program._narrowphase_admitted
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  batch, nv = len(worlds), int(model.nv)
  result = program.run_device(
      mps(np.stack([data.flexvert_xpos for data in worlds])),
      mps(np.stack([data.geom_xpos for data in worlds])),
      mps(_geom_quat_batch(model, worlds)),
      flexvert_spatial_jacobian=mps(np.zeros(
          (batch, int(model.nflexvert), 6, nv), np.float32)),
      cdof=mps(np.stack([np.asarray(data.cdof).reshape(nv, 6)
                         for data in worlds])),
      root_com=mps(np.stack([data.subtree_com for data in worlds])),
      qvel=mps(np.stack([data.qvel for data in worlds])),
      include_wake_links=False)
  active = result["active"].cpu().numpy()
  positions = result["pos"].cpu().numpy()
  normals = result["normal"].cpu().numpy()
  distances = result["dist"].cpu().numpy()
  barycentric = result["barycentric1"].cpu().numpy()
  row_J = result["workspace_J"].cpu().numpy()
  row_R = result["R"].cpu().numpy()
  row_aref = result["aref"].cpu().numpy()
  row_active = result["row_active"].cpu().numpy()
  ccd_trace = result["ccd_trace"].cpu().numpy()
  diagA = program._diag_approx.cpu().numpy()
  for env, data in enumerate(worlds):
    contact = next(c for c in data.contact[:data.ncon]
                   if int(c.geom[0]) == sphere and int(c.elem[1]) == 0)
    slot = int(np.flatnonzero(
        (descriptor.kind == _KIND_GEOM_ELEMENT)
        & (descriptor.geom == sphere) & (descriptor.flex1 == 0)
        & (descriptor.elem1 == 0))[0])
    assert active[env, slot]
    # The sphere-shrunk point precheck resolves these overlaps through the
    # shallow witness path before full GJK/EPA. Its full-simplex trace is
    # intentionally empty; the dedicated P2 fixture below exercises that
    # separate production branch.
    assert int(ccd_trace[env, slot, 2]) == 0
    assert int(ccd_trace[env, slot, 55]) == 1
    np.testing.assert_allclose(
        distances[env, slot], float(contact.dist), rtol=0, atol=3e-6)
    np.testing.assert_allclose(
        positions[env, slot], contact.pos, rtol=0, atol=3e-6)
    np.testing.assert_allclose(
        normals[env, slot], contact.frame[:3], rtol=0, atol=3e-5)
    weights = barycentric[env, slot, :2]
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=2e-5)
    nodes = np.asarray(descriptor.nodes1[slot, :2], np.int32)
    expected_surface = (np.asarray(contact.pos, np.float64)
                        + np.asarray(contact.frame[:3], np.float64)
                        * (float(model.flex_radius[0])
                           + 0.5 * float(contact.dist)))
    actual_surface = weights @ np.asarray(data.flexvert_xpos[nodes], np.float64)
    np.testing.assert_allclose(
        actual_surface, expected_surface, rtol=0, atol=4e-5)
    row = int(descriptor.row_start[slot])
    assert row_active[env, row]
    pinned_J = _dense_efc_jacobian(model, data)[int(contact.efc_address)]
    np.testing.assert_allclose(row_J[env, row], pinned_J,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(
        diagA[env, row], float(data.efc_diagA[int(contact.efc_address)]),
        rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(
        row_R[env, row], float(data.efc_R[int(contact.efc_address)]),
        rtol=8e-5, atol=2e-6)
    np.testing.assert_allclose(
        row_aref[env, row], float(data.efc_aref[int(contact.efc_address)]),
        rtol=8e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="P2 EPA seed witness requires GPU opt-in")
def test_native_flexedge_sphere_uses_direct_segment_route_not_gjk():
  """Record that sphere/flex-edge uses its direct native route."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("P2 EPA seed witness requires MPS")
  model, descriptor, sphere, _ = _sphere_capsule_fixture(batch_size=1)
  data = mujoco.MjData(model)
  qpos = np.asarray(model.qpos0, np.float64).copy()
  # Place the center on the line so the point precheck overlaps the flex
  # radius and full GJK reaches the pinned two-vertex simplex.
  qpos[1] = 0.0
  data.qpos[:] = qpos
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  assert any(int(c.geom[0]) == sphere and int(c.elem[1]) == 0
             for c in data.contact[:data.ncon])

  program = FlexContactProgram(model, batch_size=1, device="mps",
                               capture_ccd_trace=True)
  assert program._narrowphase_admitted
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  nv = int(model.nv)
  result = program.run_device(
      mps(np.asarray(data.flexvert_xpos)[None, ...]),
      mps(np.asarray(data.geom_xpos)[None, ...]),
      mps(_geom_quat_batch(model, [data])),
      flexvert_spatial_jacobian=mps(np.zeros(
          (1, int(model.nflexvert), 6, nv), np.float32)),
      cdof=mps(np.asarray(data.cdof).reshape(1, nv, 6)),
      root_com=mps(np.asarray(data.subtree_com)[None, ...]),
      qvel=mps(np.asarray(data.qvel)[None, ...]),
      include_wake_links=False)
  slot = int(np.flatnonzero(
      (descriptor.kind == _KIND_GEOM_ELEMENT)
      & (descriptor.geom == sphere) & (descriptor.flex1 == 0)
      & (descriptor.elem1 == 0))[0])
  assert bool(result["active"][0, slot].item())
  trace = result["ccd_trace"][0, slot].cpu().numpy()
  # Flex-edge candidates are handled by the direct segment/capsule producer;
  # the tetrahedron-specific GJK/EPA trace is intentionally untouched. A CPU
  # mjc_ccd oracle's simplex count does not imply native P2 execution.
  np.testing.assert_array_equal(trace, np.zeros_like(trace))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="capsule/capsule flex-edge detector needs GPU opt-in")
def test_native_capsule_flex_edge_manifold_matches_pinned_B2():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("capsule/capsule flex-edge detector requires MPS")
  model, descriptor, capsule, worlds = _capsule_flex_edge_fixture()
  program = FlexContactProgram(model, batch_size=len(worlds), device="mps")
  assert program._narrowphase_admitted
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  batch, nv = len(worlds), int(model.nv)
  result = program.run_device(
      mps(np.stack([data.flexvert_xpos for data in worlds])),
      mps(np.stack([data.geom_xpos for data in worlds])),
      mps(_geom_quat_batch(model, worlds)),
      flexvert_spatial_jacobian=mps(np.zeros(
          (batch, int(model.nflexvert), 6, nv), np.float32)),
      cdof=mps(np.stack([np.asarray(data.cdof).reshape(nv, 6)
                         for data in worlds])),
      root_com=mps(np.stack([data.subtree_com for data in worlds])),
      qvel=mps(np.stack([data.qvel for data in worlds])),
      include_wake_links=False)
  active = result["active"].cpu().numpy()
  pos = result["pos"].cpu().numpy()
  normal = result["normal"].cpu().numpy()
  dist = result["dist"].cpu().numpy()
  row_J = result["workspace_J"].cpu().numpy()
  row_R = result["R"].cpu().numpy()
  row_aref = result["aref"].cpu().numpy()
  row_active = result["row_active"].cpu().numpy()
  diagA = program._diag_approx.cpu().numpy()
  for env, data in enumerate(worlds):
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == capsule and int(c.elem[1]) == 0]
    assert contacts
    expected_slots = set()
    for ordinal, contact in enumerate(contacts):
      found = np.flatnonzero(
          (descriptor.kind == _KIND_GEOM_ELEMENT)
          & (descriptor.geom == capsule) & (descriptor.flex1 == 0)
          & (descriptor.elem1 == 0)
          & (descriptor.contact_ordinal == ordinal))
      assert found.size == 1
      slot = int(found[0])
      expected_slots.add(slot)
      assert active[env, slot]
      row = int(descriptor.row_start[slot])
      assert row_active[env, row]
      np.testing.assert_allclose(dist[env, slot], float(contact.dist),
                                 rtol=0, atol=4e-6)
      np.testing.assert_allclose(pos[env, slot], contact.pos,
                                 rtol=0, atol=4e-6)
      np.testing.assert_allclose(normal[env, slot], contact.frame[:3],
                                 rtol=0, atol=4e-5)
      pinned_J = _dense_efc_jacobian(model, data)[int(contact.efc_address)]
      np.testing.assert_allclose(row_J[env, row], pinned_J,
                                 rtol=6e-5, atol=6e-6)
      np.testing.assert_allclose(
          diagA[env, row], float(data.efc_diagA[int(contact.efc_address)]),
          rtol=3e-5, atol=3e-6)
      np.testing.assert_allclose(
          row_R[env, row], float(data.efc_R[int(contact.efc_address)]),
          rtol=8e-5, atol=2e-6)
      np.testing.assert_allclose(
          row_aref[env, row], float(data.efc_aref[int(contact.efc_address)]),
          rtol=8e-5, atol=2e-5)
    assert set(np.flatnonzero(active[env]).tolist()) == expected_slots


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public flex-edge/sphere trajectory requires GPU opt-in")
def test_native_public_sphere_flex_edge_cc_trajectory_B2():
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, _, sphere, worlds = _sphere_capsule_fixture()
  qpos = np.stack([data.qpos for data in worlds]).astype(np.float32)
  qvel = np.stack([data.qvel for data in worlds]).astype(np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  slots = np.flatnonzero(
      (program.descriptor.geom == sphere)
      & (program.descriptor.kind == _KIND_GEOM_ELEMENT))
  assert slots.size == 1
  actual = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(actual.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)


def test_dynamic_shell_diagonal_source_has_all_pinned_tfi_terms():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "flex_contact.metal").read_text()
  start = shader.index("inline float flex_diag_basis_node(")
  end = shader.index("inline float flex_diag_side_weight(", start)
  source = shader[start:end]
  # MuJoCo mju_shellTFIWeights has 6 face, 12 edge-correction, and
  # 8 corner contributions. The final two corners were omitted from the
  # native Q2 shell diagonal kernel and caused both native parity failures.
  assert "thread int3 index[26];" in source
  assert "thread float factor[26];" in source
  assert "index[24]=int3(nx-1,ny-1,0); factor[24]=s*t*(1-u);" in source
  assert "index[25]=int3(nx-1,ny-1,nz-1); factor[25]=s*t*u;" in source
  assert "for (int q=0;q<26;q++)" in source


@pytest.mark.parametrize(("dof", "shell"), [
    (None, False), ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
@pytest.mark.parametrize("cross_flex", [False, True])
def test_element_contact_weighted_point_jacobian_matches_pinned_efc(
    dof, shell, cross_flex):
  model, data, _ = _dynamic_diag_fixture(dof, shell, cross_flex)
  contacts = [c for c in data.contact[:data.ncon]
              if (int(c.elem[0]) >= 0 or int(c.elem[1]) >= 0)]
  assert contacts
  for contact in contacts:
    relative = np.zeros((3, int(model.nv)), np.float64)
    for side in (0, 1):
      sign = -1.0 if side == 0 else 1.0
      geom = int(contact.geom[side])
      flex_id = int(contact.flex[side])
      if geom >= 0:
        bodies = {int(model.geom_bodyid[geom]): 1.0}
      elif flex_id >= 0 and int(contact.elem[side]) >= 0:
        bodies = _flex_element_body_weights(
            model, flex_id, int(contact.elem[side]),
            int(contact.vert[1-side]), contact.pos, data.flexvert_xpos)
      elif flex_id >= 0 and int(contact.vert[side]) >= 0:
        vertex = (int(model.flex_vertadr[flex_id])
                  + int(contact.vert[side]))
        bodies = _flex_vertex_body_weights(model, flex_id, vertex)
      else:
        continue
      for body, weight in bodies.items():
        jacp = np.zeros((3, int(model.nv)), np.float64)
        jacr = np.zeros_like(jacp)
        mujoco.mj_jac(model, data, jacp, jacr, contact.pos, int(body))
        relative += sign*float(weight)*jacp
    expected = np.asarray(contact.frame[:3], np.float64) @ relative
    actual = _dense_efc_jacobian(model, data)[int(contact.efc_address)]
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.gpu
@pytest.mark.parametrize(("dof", "shell"), [
    (None, False), ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
@pytest.mark.parametrize("cross_flex", [False, True])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="source flex contact J requires GPU opt-in")
def test_native_source_contact_jacobians_match_pinned_efc_B2(
    dof, shell, cross_flex):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("source flex contact J requires MPS")
  model, _, descriptor = _dynamic_diag_fixture(dof, shell, cross_flex)
  assert model.nv > 0
  worlds = []
  qpos_worlds = []
  for world in range(2):
    data = mujoco.MjData(model)
    qvel = np.linspace(-0.025, 0.031, int(model.nv), dtype=np.float64)
    qvel *= (1.0 if world == 0 else -0.73)
    qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    mujoco.mj_integratePos(model, qpos, qvel, 0.0015 * (world + 1))
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.cvel) > 0
    assert data.ncon > 0
    worlds.append(data)
    qpos_worlds.append(qpos.copy())
  assert not np.allclose(qpos_worlds[0], qpos_worlds[1])
  assert not np.allclose(worlds[0].flexvert_xpos, worlds[1].flexvert_xpos)

  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  program = FlexContactProgram(model, batch_size=2, device="mps")
  flexvert_xpos = np.stack([data.flexvert_xpos for data in worlds])
  program._diag_current_flexvert_xpos = mps(flexvert_xpos)
  contact_pos = np.zeros((2, descriptor.slot_count, 3), np.float32)
  matched = []
  for env, data in enumerate(worlds):
    contacts = [c for c in data.contact[:data.ncon]
                if ((int(c.flex[0]) >= 0 and int(c.elem[0]) >= 0)
                    or (int(c.flex[1]) >= 0 and int(c.elem[1]) >= 0))]
    assert contacts
    for contact in contacts:
      if int(contact.geom[0]) >= 0 and int(contact.elem[1]) >= 0:
        slots = np.flatnonzero(
            (descriptor.kind == _KIND_GEOM_ELEMENT)
            & (descriptor.geom == int(contact.geom[0]))
            & (descriptor.flex1 == int(contact.flex[1]))
            & (descriptor.elem1 == int(contact.elem[1])))
      elif int(contact.flex[0]) >= 0 and int(contact.flex[1]) >= 0:
        slots = np.flatnonzero(
            (descriptor.kind == _KIND_ELEMENT_PAIR)
            & (descriptor.flex1 == int(contact.flex[0]))
            & (descriptor.elem1 == int(contact.elem[0]))
            & (descriptor.flex2 == int(contact.flex[1]))
            & (descriptor.elem2 == int(contact.elem[1])))
      else:
        continue
      assert slots.size, (contact.flex, contact.elem, contact.geom)
      slot = int(slots[0])
      contact_pos[env, slot] = contact.pos
      matched.append((env, slot, contact))
  assert matched
  result = program.run_source_contact_spatial_jacobians(
      {"pos": mps(contact_pos)}, mps(np.stack([d.cdof for d in worlds])),
      mps(np.stack([d.subtree_com for d in worlds])))
  relative = result["relative"].cpu().numpy()
  side2 = result["side2"].cpu().numpy()
  for env, slot, contact in matched:
    world_data = worlds[env]
    expected = _dense_efc_jacobian(model, world_data)[int(contact.efc_address)]
    # MuJoCo exposes `contact.frame` as a flat 9-vector. Its first three
    # values are the normal row, not a nested 3x3 array.
    normal = np.asarray(contact.frame[:3], np.float64)
    assert normal.shape == (3,)
    actual = normal @ relative[env, slot, :3]
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    # For flex/geom contacts, side 2 is the independently moving rigid body.
    # Check it separately so a correct-looking relative row cannot hide a
    # missing rigid reaction or a root-COM/contact-point translation error.
    if int(contact.geom[0]) >= 0 and int(contact.flex[1]) >= 0:
      body = int(model.geom_bodyid[int(contact.geom[0])])
      jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
      jacr = np.zeros_like(jacp)
      mujoco.mj_jac(model, world_data, jacp, jacr, contact.pos, body)
      rigid_expected = normal @ jacp
      rigid_actual = normal @ side2[env, slot, :3]
      if body != 0:
        assert np.linalg.norm(rigid_expected) > 1e-5
      np.testing.assert_allclose(
          rigid_actual, rigid_expected, rtol=3e-5, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="self-flex source contact J requires GPU opt-in")
def test_native_source_self_flex_jacobian_matches_pinned_efc_B2():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("self-flex source contact J requires MPS")
  model, descriptor, worlds = _self_flex_contact_fixture()
  program = FlexContactProgram(model, batch_size=len(worlds), device="mps")
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  positions = np.zeros((len(worlds), descriptor.slot_count, 3), np.float32)
  matches = []
  for env, data in enumerate(worlds):
    contacts = [c for c in data.contact[:data.ncon]
                if (int(c.flex[0]) == int(c.flex[1]) == 0
                    and int(c.elem[0]) >= 0 and int(c.elem[1]) >= 0)]
    assert contacts
    contact = contacts[0]
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == 0) & (descriptor.elem1 == int(contact.elem[0]))
        & (descriptor.flex2 == 0) & (descriptor.elem2 == int(contact.elem[1])))
    assert slots.size
    slot = int(slots[0])
    positions[env, slot] = contact.pos
    matches.append((env, slot, contact))
  assert not np.allclose(worlds[0].flexvert_xpos, worlds[1].flexvert_xpos)
  program._diag_current_flexvert_xpos = mps(
      np.stack([data.flexvert_xpos for data in worlds]))
  result = program.run_source_contact_spatial_jacobians(
      {"pos": mps(positions)}, mps(np.stack([data.cdof for data in worlds])),
      mps(np.stack([data.subtree_com for data in worlds])))
  relative = result["relative"].cpu().numpy()
  for env, slot, contact in matches:
    data = worlds[env]
    pinned = _dense_efc_jacobian(model, data)[int(contact.efc_address)]
    actual = np.asarray(contact.frame[:3], np.float64) @ relative[env, slot, :3]
    np.testing.assert_allclose(actual, pinned, rtol=3e-5, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(("dof", "shell"), [
    (None, False), ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="dynamic flex diagApprox requires GPU opt-in")
def test_native_dynamic_element_geom_diag_matches_pinned_cpu(dof, shell):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("dynamic flex diagApprox requires MPS")
  model, data, descriptor = _dynamic_diag_fixture(dof, shell, False)
  program = FlexContactProgram(model, device="mps")
  assert not program._narrowphase_admitted
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32)[None].copy(), dtype=torch.float32,
      device="mps").contiguous()
  program._diag_current_flexvert_xpos = mps(data.flexvert_xpos)
  contact_pos = np.zeros((int(descriptor.slot_count), 3), np.float32)
  contacts = [c for c in data.contact[:data.ncon]
              if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
              and int(c.elem[1]) >= 0]
  assert contacts
  slots = []
  for contact in contacts:
    matches = np.flatnonzero(
        (descriptor.kind == _KIND_GEOM_ELEMENT)
        & (descriptor.geom == int(contact.geom[0]))
        & (descriptor.flex1 == int(contact.flex[1]))
        & (descriptor.elem1 == int(contact.elem[1])))
    assert matches.size
    slot = int(matches[0])
    contact_pos[slot] = contact.pos
    slots.append((slot, int(contact.efc_address)))
  result = {"pos": mps(contact_pos)}
  diagonal = program.run_diag_approx_device(result).cpu().numpy()[0]
  for slot, efc in slots:
    row = int(descriptor.row_start[slot])
    np.testing.assert_allclose(
        diagonal[row], float(data.efc_diagA[efc]), rtol=3e-5, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
@pytest.mark.parametrize("geom_margin_gap", [0.0, 0.001])
@pytest.mark.parametrize("disable_midphase", [False, True])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="interpolated element detector requires GPU opt-in")
def test_native_interpolated_element_detector_and_diag_match_pinned_B2(
    dof, shell, geom_margin_gap, disable_midphase):
  """Stage-check the actual candidate detector on compiled Q1/Q2 simplices."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("interpolated element detector requires MPS")
  # Preserve the original zero-margin case and add the shallow-inflation
  # route with nonzero compiled margin and gap; compare physical dist/point.
  model, first, descriptor = _dynamic_diag_fixture(
      dof, shell, False, geom_margin_gap=geom_margin_gap,
      disable_midphase=disable_midphase)
  assert int(model.flex_interp[0]) != 0
  worlds = [first]
  sphere_body = int(model.geom_bodyid[0])
  joint = int(model.body_jntadr[sphere_body]) if model.njnt else -1
  assert joint >= 0
  qadr = int(model.jnt_qposadr[joint])
  vadr = int(model.jnt_dofadr[joint])
  for env in range(1, 2):
    data = mujoco.MjData(model)
    qpos = np.asarray(first.qpos, np.float64).copy()
    qvel = np.linspace(-0.015, 0.019, int(model.nv), dtype=np.float64)
    qvel *= -0.67 if env else 1.0
    qpos[qadr:qadr+3] += np.asarray([0.004, -0.003, 0.002])*env
    mujoco.mj_integratePos(model, qpos, qvel, 0.0003*(env+1))
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    worlds.append(data)
  for data in worlds:
    np.testing.assert_allclose(
        model.geom_margin[0] + model.geom_gap[0], 2*geom_margin_gap,
        rtol=0, atol=1e-9)
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
                and int(c.elem[1]) >= 0]
    expected_elem_ids = (
        [36, 38, 39, 41, 44, 45, 46, 47]
        if not disable_midphase else list(range(36, 48)))
    assert [int(contact.elem[1]) for contact in contacts] == expected_elem_ids
  # The native detector receives float32 positions/quaternions. Recompute the
  # independent pinned result with precisely those same geometry operands;
  # comparing its rounded positions to the original double-state contacts can
  # select a different strict EPA face even when the input change is sub-ulp.
  flexvert_inputs = np.stack([data.flexvert_xpos for data in worlds]).astype(
      np.float32)
  geom_pos_inputs = np.stack([data.geom_xpos for data in worlds]).astype(
      np.float32)
  geom_quat_inputs = _geom_quat_batch(model, worlds).astype(np.float32)
  cpu_oracle_worlds = [
      _cpu_collision_for_float32_detector_inputs(
          model, data, flexvert_inputs[env], geom_pos_inputs[env],
          geom_quat_inputs[env])
      for env, data in enumerate(worlds)]
  for source_data, oracle_data in zip(worlds, cpu_oracle_worlds):
    source_contacts = [
        int(contact.elem[1]) for contact in source_data.contact[:source_data.ncon]
        if int(contact.geom[0]) == 0 and int(contact.flex[1]) == 0
        and int(contact.elem[1]) >= 0]
    oracle_contacts = [
        int(contact.elem[1]) for contact in oracle_data.contact[:oracle_data.ncon]
        if int(contact.geom[0]) == 0 and int(contact.flex[1]) == 0
        and int(contact.elem[1]) >= 0]
    assert oracle_contacts == source_contacts
  program = FlexContactProgram(model, batch_size=2, device="mps",
                               capture_ccd_trace=True)
  assert program._narrowphase_admitted
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32).copy(), dtype=torch.float32,
      device="mps").contiguous()
  result = program.run_device(
      mps(np.stack([data.flexvert_xpos for data in worlds])),
      mps(np.stack([data.geom_xpos for data in worlds])),
      mps(_geom_quat_batch(model, worlds)))
  actual_active = result["active"].cpu().numpy()
  raw_active = result["raw_active"].cpu().numpy()
  actual_dist = result["dist"].cpu().numpy()
  actual_pos = result["pos"].cpu().numpy()
  actual_normal = result["normal"].cpu().numpy()
  actual_diag = program.run_diag_approx_device(result).cpu().numpy()
  narrowphase_status = result["narrowphase_status"].cpu().numpy()
  ccd_trace = result["ccd_trace"].cpu().numpy()
  trace_iterations, trace_map_capacity, trace_stride, trace_size = (
      _flex_ccd_trace_capacity(model.opt.ccd_iterations))
  assert ccd_trace.shape == (2, descriptor.slot_count, trace_size)
  expected_slots_by_world = []
  for data in worlds:
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
                and int(c.elem[1]) >= 0]
    expected_slots = set()
    for contact in contacts:
      matches = np.flatnonzero(
          (descriptor.kind == _KIND_GEOM_ELEMENT)
          & (descriptor.geom == 0)
          & (descriptor.flex1 == int(contact.flex[1]))
          & (descriptor.elem1 == int(contact.elem[1])))
      assert matches.size == 1
      expected_slots.add(int(matches[0]))
    expected_slots_by_world.append(expected_slots)
  # Check discrete routing before numeric distance/point tolerances so a CCD
  # mismatch cannot conceal a missing or extra compiled element candidate.
  for env, expected_slots in enumerate(expected_slots_by_world):
    actual_slots = set(map(int, np.flatnonzero(actual_active[env])))
    if actual_slots != expected_slots:
      affected = sorted(actual_slots ^ expected_slots)
      details = [{
          "slot": slot,
          "element": int(descriptor.elem1[slot]),
          "nodes": descriptor.nodes1[slot].tolist(),
          "expected_active": slot in expected_slots,
          "raw_active": bool(raw_active[env, slot]),
          "active": bool(actual_active[env, slot]),
          "status": int(narrowphase_status[env, slot]),
          "trace_header": ccd_trace[env, slot, :16].tolist(),
      } for slot in affected]
      pytest.fail(
          f"native candidate identities differ in world {env}; "
          f"expected slots {sorted(expected_slots)}, actual slots "
          f"{sorted(actual_slots)}, affected {details}")
  for env, data in enumerate(cpu_oracle_worlds):
    contacts = [c for c in data.contact[:data.ncon]
                if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
                and int(c.elem[1]) >= 0]
    expected_slots = set()
    for contact in contacts:
      matches = np.flatnonzero(
          (descriptor.kind == _KIND_GEOM_ELEMENT)
          & (descriptor.geom == 0)
          & (descriptor.flex1 == int(contact.flex[1]))
          & (descriptor.elem1 == int(contact.elem[1])))
      assert matches.size == 1
      slot = int(matches[0])
      expected_slots.add(slot)
      assert narrowphase_status[env, slot] == 0
      assert actual_active[env, slot]
      event_records = ccd_trace[env, slot]
      if int(event_records[2]) == 3 or event_records[57] >= 0.0:
        # P3-specific capture: exact unit-normal hi/lo and the support-vertex
        # choices for -normal (v5) then +normal (v4).  These checks validate
        # trace integrity; numerical source parity is checked by the pinned
        # contact assertions below and remains a separate gate.
        p3 = event_records[207:222]
        assert p3[0] == 1.0
        normal = np.asarray((p3[1]+p3[2], p3[3]+p3[4], p3[5]+p3[6]))
        assert np.all(np.isfinite(normal))
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0,
                                   rtol=0, atol=2e-6)
        assert all(float(index).is_integer() and 0 <= index < 4
                   for index in p3[7:9])
        assert np.all(np.isfinite(p3[9:15]))
        assert np.all(np.isfinite(event_records[222:240]))
        assert np.all(np.isfinite(event_records[240:252]))
      iteration_count = min(int(event_records[4]), trace_iterations)
      shallow_witness = event_records[55] == 1.0
      if shallow_witness:
        # Pinned sphere-special CCD inflates the point-GJK witness without
        # entering EPA; these valid active rows intentionally have no events.
        assert iteration_count == 0
      else:
        assert iteration_count > 0
      for iteration in range(iteration_count):
        start = (_FLEX_CCD_TRACE_SIZE + iteration*trace_stride)
        event = event_records[start:start+trace_stride]
        assert int(event[0]) == iteration
        assert event[31] in (1.0, 2.0, 3.0)
        map_count = int(event[8])
        assert 0 < map_count <= trace_map_capacity
        map_before = event[_FLEX_CCD_ITER_TRACE_HEADER:
                           _FLEX_CCD_ITER_TRACE_HEADER+trace_map_capacity]
        assert np.all(map_before[:map_count] >= 0)
        assert np.all(map_before[:map_count] < int(event[9]))
        if event[31] >= 2.0:
          horizon_count = int(event[30])
          assert 3 <= horizon_count <= 24
          horizon_start = (_FLEX_CCD_ITER_TRACE_HEADER
                           + 3*trace_map_capacity)
          horizon = event[horizon_start:horizon_start+2*horizon_count].reshape(
              horizon_count, 2)
          assert np.all((horizon[:, 0] >= 0)
                        & (horizon[:, 0] < int(event[9])))
          assert np.all((horizon[:, 1] >= 0) & (horizon[:, 1] < 3))
      np.testing.assert_allclose(
          actual_dist[env, slot], float(contact.dist), rtol=0, atol=6e-6)
      np.testing.assert_allclose(
          actual_pos[env, slot], contact.pos, rtol=0, atol=6e-6)
      np.testing.assert_allclose(
          actual_normal[env, slot], contact.frame[:3],
          rtol=0, atol=6e-5)
      row = int(descriptor.row_start[slot])
      np.testing.assert_allclose(
          actual_diag[env, row], float(data.efc_diagA[int(contact.efc_address)]),
          rtol=3e-5, atol=3e-6)
    assert expected_slots == expected_slots_by_world[env]


@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_interpolated_sphere_ccd_shallow_fixture_has_margin_and_contacts(
    dof, shell):
  """Keep stage inputs on pinned sphere shrink/inflate CCD branch."""
  model, data, descriptor = _dynamic_diag_fixture(
      dof, shell, False, geom_margin_gap=0.001)
  assert not (int(model.opt.disableflags)
              & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))
  np.testing.assert_allclose(
      model.geom_margin[0] + model.geom_gap[0], 0.002, rtol=0, atol=1e-9)
  contacts = [c for c in data.contact[:data.ncon]
              if int(c.geom[0]) == 0 and int(c.flex[1]) == 0
              and int(c.elem[1]) >= 0]
  assert len(contacts) == 8
  assert descriptor.slot_count == len(_flex_bvh_element_order(
      model, 0, int(model.flex_elemnum[0])))
  assert all(np.isfinite(np.asarray(c.pos)).all()
             and np.isfinite(float(c.dist)) for c in contacts)


def test_quadratic_flex_contact_efc_jacobian_expands_compiled_csr():
  model, data, _ = _dynamic_diag_fixture(
      "quadratic", False, False, jacobian="sparse")
  assert mujoco.mj_isSparse(model)
  dense = _dense_efc_jacobian(model, data)
  assert dense.shape == (int(data.nefc), int(model.nv))
  assert data.ncon > 0 and data.nefc > 0
  for row in range(int(data.nefc)):
    start = int(data.efc_J_rowadr[row])
    count = int(data.efc_J_rownnz[row])
    columns = np.asarray(
        data.efc_J_colind[start:start + count], dtype=np.int64)
    values = np.asarray(data.efc_J[start:start + count], dtype=np.float64)
    np.testing.assert_array_equal(dense[row, columns], values)


@pytest.mark.gpu
@pytest.mark.parametrize(("dof", "shell"), [
    (None, False), ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="dynamic flex diagApprox requires GPU opt-in")
def test_native_dynamic_cross_flex_diag_matches_pinned_cpu(dof, shell):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("dynamic flex diagApprox requires MPS")
  model, data, descriptor = _dynamic_diag_fixture(dof, shell, True)
  program = FlexContactProgram(model, device="mps")
  assert not program._narrowphase_admitted
  mps = lambda value: torch.as_tensor(
      np.asarray(value, np.float32)[None].copy(), dtype=torch.float32,
      device="mps").contiguous()
  program._diag_current_flexvert_xpos = mps(data.flexvert_xpos)
  contact_pos = np.zeros((int(descriptor.slot_count), 3), np.float32)
  contacts = [c for c in data.contact[:data.ncon]
              if int(c.flex[0]) == 0 and int(c.flex[1]) == 1
              and int(c.elem[0]) >= 0 and int(c.elem[1]) >= 0]
  assert contacts
  slots = []
  for contact in contacts:
    matches = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == int(contact.flex[0]))
        & (descriptor.flex2 == int(contact.flex[1]))
        & (descriptor.elem1 == int(contact.elem[0]))
        & (descriptor.elem2 == int(contact.elem[1])))
    assert matches.size
    slot = int(matches[0])
    contact_pos[slot] = contact.pos
    slots.append((slot, int(contact.efc_address)))
  diagonal = program.run_diag_approx_device(
      {"pos": mps(contact_pos)}).cpu().numpy()[0]
  for slot, efc in slots:
    row = int(descriptor.row_start[slot])
    np.testing.assert_allclose(
        diagonal[row], float(data.efc_diagA[efc]), rtol=3e-5, atol=3e-6)


def _cross_flex_capsule_fixture():
  """Small source fixture for the pinned 1D-flex/1D-flex collision route."""
  edge = lambda name, x: f"""
    <flexcomp name="{name}" type="grid" count="3 1 1"
              pos="{x} 0 0" spacing=".05 .05 .05" mass="1" dim="1">
      <contact contype="1" conaffinity="1" selfcollide="none"/>
      <edge stiffness="10" damping=".1"/>
      <elasticity young="100" poisson=".2"/>
    </flexcomp>
  """
  return mujoco.MjModel.from_xml_string(
      "<mujoco><option gravity='0 0 0'/><worldbody>"
      + edge("left", 0) + edge("right", .03)
      + "</worldbody></mujoco>")


def _cross_flex_contact_slots(model, data, descriptor):
  """Pair CPU contacts with their fixed candidate/ordinal identity."""
  grouped = {}
  for contact in data.contact[:data.ncon]:
    if int(contact.flex[0]) != 0 or int(contact.flex[1]) != 1:
      continue
    key = (int(contact.elem[0]), int(contact.elem[1]))
    grouped.setdefault(key, []).append(contact)
  expected = {}
  for (elem1, elem2), contacts in grouped.items():
    candidates = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == 0) & (descriptor.flex2 == 1)
        & (descriptor.elem1 == elem1) & (descriptor.elem2 == elem2))
    candidates = candidates[np.argsort(descriptor.contact_ordinal[candidates])]
    assert len(candidates) >= len(contacts)
    for ordinal, contact in enumerate(contacts):
      expected[int(candidates[ordinal])] = contact
  return expected


def test_cross_flex_capsule_fixture_has_exact_cpu_contact_and_slot_witness():
  """Guard the nonzero cross-flex raw capsule manifold used by the MPS gate."""
  model = _cross_flex_capsule_fixture()
  data = mujoco.MjData(model)
  # Use precisely the f32 state that the MPS candidate program receives.
  data.qpos[:] = data.qpos.astype(np.float32).astype(np.float64)
  data.qvel[:] = data.qvel.astype(np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)
  expected = _cross_flex_contact_slots(model, data, descriptor)
  assert len(expected) == 6
  assert int(np.count_nonzero(descriptor.kind == _KIND_ELEMENT_PAIR)) == 16
  assert all(int(descriptor.contact_ordinal[slot]) < 4 for slot in expected)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="cross-flex capsule detector requires GPU opt-in")
def test_native_cross_flex_capsule_manifold_matches_pinned_cpu():
  """Compare the native kind-6 capsule path without opening public admission."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("cross-flex capsule detector requires MPS")
  model = _cross_flex_capsule_fixture()
  data = mujoco.MjData(model)
  data.qpos[:] = data.qpos.astype(np.float32).astype(np.float64)
  data.qvel[:] = data.qvel.astype(np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)
  expected = _cross_flex_contact_slots(model, data, descriptor)
  assert len(expected) == 6
  program = FlexContactProgram(model, device="mps")
  # The public route remains fail-closed. This stage gate only executes the
  # already-implemented 1D/1D raw capsule pair branch for pinned comparison.
  assert not program._narrowphase_admitted
  program._narrowphase_admitted = True

  def mps(value):
    return torch.as_tensor(
        np.asarray(value, np.float32).copy(), dtype=torch.float32,
        device="mps").contiguous()

  result = program.run_device(
      mps(data.flexvert_xpos[None]),
      mps(np.zeros((1, model.ngeom, 3), np.float32)),
      mps(np.tile(np.array([1, 0, 0, 0], np.float32),
                  (1, model.ngeom, 1))))
  assert set(np.flatnonzero(result["active"].cpu().numpy()[0])) == set(expected)
  actual_dist = result["dist"].cpu().numpy()[0]
  actual_pos = result["pos"].cpu().numpy()[0]
  actual_normal = result["normal"].cpu().numpy()[0]
  for slot, contact in expected.items():
    assert int(result["narrowphase_status"].cpu().numpy()[0, slot]) == 0
    np.testing.assert_allclose(
        actual_dist[slot], float(contact.dist), rtol=0, atol=6e-6)
    np.testing.assert_allclose(
        actual_pos[slot], contact.pos, rtol=0, atol=6e-6)
    np.testing.assert_allclose(
        actual_normal[slot], contact.frame[:3], rtol=0, atol=6e-5)


def test_contact_row_materials_include_pinned_global_override_and_friction_clamp():
  model, data, descriptor = _fixture(6, "elliptic")
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
  model.opt.o_solref[:] = (-80.0, -7.0)
  model.opt.o_solimp[:] = (0.15, 0.85, 0.03, 0.4, 3.0)
  model.opt.o_friction[:] = (0.4, 0.6, 0.02, 0.0, 0.03)
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)
  np.testing.assert_allclose(
      descriptor.solref, np.tile([-80.0, -7.0], (descriptor.slot_count, 1)))
  np.testing.assert_allclose(
      descriptor.solimp,
      np.tile([0.15, 0.85, 0.03, 0.4, 3.0], (descriptor.slot_count, 1)))
  np.testing.assert_allclose(
      descriptor.friction[0], [0.4, 0.6, 0.02, 1e-5, 0.03])
  for slot in np.flatnonzero(descriptor.kind == 0):
    contact = next(c for c in data.contact[:data.ncon]
                   if int(c.vert[1]) == int(descriptor.vert1[slot]))
    np.testing.assert_allclose(descriptor.friction[slot], contact.friction,
                               rtol=0, atol=5e-8)
    np.testing.assert_allclose(descriptor.solref[slot], contact.solref)
    np.testing.assert_allclose(descriptor.solimp[slot], contact.solimp)


def _point_body_spatial_jacobian(model, data, point, body):
  """Independent point/angular Jacobian from the pinned public MuJoCo API."""
  jp = np.zeros((3, int(model.nv)), np.float64)
  jr = np.zeros_like(jp)
  mujoco.mj_jac(model, data, jp, jr, np.asarray(point, np.float64), int(body))
  return np.concatenate((jp, jr), axis=0)


def _project_spatial_jacobian(contact, spatial, condim, cone):
  """Project an independently computed 6D point Jacobian into contact rows."""
  frame = np.asarray(contact.frame, np.float64).reshape(3, 3)
  rows = np.zeros((condim if cone == "elliptic" or condim == 1
                   else 2 * (condim - 1), spatial.shape[1]), np.float64)
  if condim == 1:
    rows[0] = frame[0] @ spatial[:3]
    return rows
  if cone == "elliptic":
    for axis in range(condim):
      part = 0 if axis < 3 else 3
      frame_axis = axis if axis < 3 else axis - 3
      rows[axis] = frame[frame_axis] @ spatial[part:part+3]
  else:
    for pair in range(condim - 1):
      mu = float(contact.friction[pair])
      axis = pair + 1
      part = 0 if axis < 3 else 3
      frame_axis = axis if axis < 3 else axis - 3
      normal = frame[0] @ spatial[:3]
      tangent = frame[frame_axis] @ spatial[part:part+3]
      rows[2*pair] = normal + mu * tangent
      rows[2*pair+1] = normal - mu * tangent
  return rows


@pytest.mark.parametrize("condim", [1, 3, 4, 6])
@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
def test_public_node_point_jacobian_matches_actual_pinned_contact_rows(condim, cone):
  model, data, descriptor = _fixture(condim, cone)
  assert int(model.flex_interp[0]) == 0
  dense_J = _dense_efc_jacobian(model, data)
  for slot in np.flatnonzero(descriptor.kind == 0):
    contact = next(c for c in data.contact[:data.ncon]
                   if int(c.vert[1]) == int(descriptor.vert1[slot]))
    vertex = int(descriptor.vert1[slot])
    spatial = _point_body_spatial_jacobian(
        model, data, data.flexvert_xpos[vertex], model.flex_vertbodyid[vertex])
    rows = dense_J[int(contact.efc_address):
                   int(contact.efc_address) + int(descriptor.row_span[slot])]
    assert int(contact.flex[1]) == 0
    np.testing.assert_allclose(
        _project_spatial_jacobian(contact, spatial, condim, cone), rows,
        rtol=3e-7, atol=3e-8)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native row assembly requires explicit GPU opt-in")
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
def test_native_rows_match_actual_pinned_mujoco_efc(condim, cone):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native Metal row qualification requires MPS")
  from mujoco_metal.flex_contact_rows import FlexContactRows

  model, data, descriptor = _fixture(condim, cone)
  assert int(model.flex_interp[0]) == 0
  nslot, nrow, nv = (descriptor.slot_count, descriptor.row_capacity,
                     int(model.nv))
  dense_J = _dense_efc_jacobian(model, data)
  spatial = np.zeros((1, nslot, 6, nv), np.float32)
  dist = np.zeros((1, nslot), np.float32)
  active = np.zeros((1, nslot), np.bool_)
  frame = np.zeros((1, nslot, 3, 3), np.float32)
  diagA = np.zeros((1, nrow), np.float32)
  expected_J = np.zeros((nrow, nv), np.float32)
  expected_R = np.zeros(nrow, np.float32)
  expected_aref = np.zeros(nrow, np.float32)
  for slot in range(nslot):
    vert = int(descriptor.vert1[slot])
    contact = next(c for c in data.contact[:data.ncon]
                   if int(c.vert[1]) == vert)
    start = int(descriptor.row_start[slot])
    span = int(descriptor.row_span[slot])
    addr = int(contact.efc_address)
    frame[0, slot] = np.asarray(contact.frame, np.float32).reshape(3, 3)
    dist[0, slot] = float(contact.dist)
    active[0, slot] = not bool(contact.exclude)
    rows = dense_J[addr:addr+span]
    node_body = int(model.flex_vertbodyid[vert])
    point = np.asarray(data.flexvert_xpos[vert], np.float64)
    side1 = _point_body_spatial_jacobian(model, data, point, node_body)
    # This plane is attached to world body 0. Contact side 1 is the flex node,
    # so MuJoCo's point-velocity difference has the same positive sign.
    assert int(contact.geom[0]) >= 0 and int(contact.flex[1]) == 0
    spatial[0, slot] = side1.astype(np.float32)
    point_rows = _project_spatial_jacobian(contact, side1, condim, cone)
    np.testing.assert_allclose(point_rows, rows, rtol=2e-7, atol=2e-8)
    if cone == "pyramidal" and condim > 1:
      # mj_makeImpedance replaces all pyramid R rows with Rpy, then rewrites
      # efc_diagA. Undo Rpy to recover the original normal diagApprox.
      normal_diag = float(data.efc_diagA[addr]) / (2.0 * float(contact.mu) ** 2)
    else:
      normal_diag = float(data.efc_diagA[addr])
    diagA[0, start:start+span] = normal_diag
    expected_J[start:start+span] = rows.astype(np.float32)
    expected_R[start:start+span] = np.asarray(data.efc_R[addr:addr+span], np.float32)
    expected_aref[start:start+span] = np.asarray(
        data.efc_aref[addr:addr+span], np.float32)

  program = FlexContactRows(model, descriptor, device="mps")
  with pytest.raises(RuntimeError, match="assembled"):
    program.refresh_aref(
        torch.as_tensor(np.asarray(data.qvel, np.float32)[None].copy(),
                        device="mps"))
  result = program.run_device(
      {"active": torch.as_tensor(active, device="mps"),
       "dist": torch.as_tensor(dist, device="mps"),
       "frame": torch.as_tensor(frame, device="mps")},
      torch.as_tensor(spatial, device="mps"),
      torch.as_tensor(np.asarray(data.qvel, np.float32)[None].copy(),
                      device="mps"),
      torch.as_tensor(diagA, device="mps"))
  np.testing.assert_allclose(result["workspace_J"].cpu().numpy()[0],
                             expected_J, rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(result["R"].cpu().numpy()[0],
                             expected_R, rtol=5e-5, atol=2e-6)
  np.testing.assert_allclose(result["aref"].cpu().numpy()[0],
                             expected_aref, rtol=5e-5, atol=2e-5)
  assert result["active"].cpu().numpy().all()
  refreshed = program.refresh_aref(
      torch.as_tensor(np.asarray(data.qvel, np.float32)[None].copy(),
                      device="mps"))
  np.testing.assert_allclose(refreshed.cpu().numpy()[0],
                             expected_aref, rtol=5e-5, atol=2e-5)
