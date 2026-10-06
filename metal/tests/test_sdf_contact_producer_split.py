"""CPU source-contract tests for the staged rigid-contact producer split."""
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SHADERS = ROOT / "mujoco_metal" / "shaders"


def _function(source, signature):
  start = source.index(signature)
  brace = source.index("{", start)
  depth = 0
  for pos in range(brace, len(source)):
    if source[pos] == "{":
      depth += 1
    elif source[pos] == "}":
      depth -= 1
      if depth == 0:
        return source[start:pos + 1]
  raise AssertionError(f"unterminated function: {signature}")


def test_sdf_and_generic_dispatch_graphs_are_separate():
  collision = (SHADERS / "collision_primitives.metal").read_text()
  coupled = (SHADERS / "coupled_constraints.metal").read_text()
  generic_dispatch = _function(collision, "inline int collide_pair_without_sdf(")
  sdf_dispatch = _function(collision, "inline int collide_pair_sdf_cached(")
  mesh_dispatch = _function(collision, "inline int collide_pair_mesh_sdf(")
  generic_producer = _function(coupled, "kernel void contact_produce_generic(")
  mesh_producer = _function(coupled, "kernel void contact_produce_mesh_sdf(")
  sdf_seed_init = _function(coupled, "kernel void contact_sdf_seed_init(")
  sdf_descent = _function(coupled, "kernel void contact_sdf_descent_prepare(")
  sdf_line_search = _function(coupled, "kernel void contact_sdf_line_search(")
  sdf_publish_normal = _function(coupled, "kernel void contact_sdf_publish_normal(")
  sdf_publish_contact = _function(coupled, "kernel void contact_sdf_publish_contact(")
  sdf_finalize = _function(coupled, "kernel void contact_finalize_sdf(")
  row_assembler = _function(coupled, "kernel void contact_normal(")

  assert "collide_sdf(" not in generic_dispatch
  assert "collide_mesh_sdf(" not in generic_dispatch
  assert "collide_pair_without_sdf(" in generic_producer
  assert "collide_pair_sdf(" not in generic_producer
  # Analytic SDF pairs now run through per-seed kernels and a serial,
  # source-order finalizer instead of a one-shot pair producer.
  assert "sdf_stage_seed_init(" in sdf_seed_init
  assert "sdf_stage_prepare_step(" in sdf_descent
  assert "sdf_stage_line_search(" in sdf_line_search
  assert "sdf_vector_normalize(grad)" in sdf_publish_normal
  assert "sdf_stage_publish(" in sdf_publish_contact
  assert "for (int i=0;i<tile_count;i++)" in sdf_finalize
  assert "sdf_pair_distance(x,accepted[k])" in sdf_finalize
  assert "collide_pair_sdf_cached(" not in generic_producer
  assert "collide_mesh_sdf(" in mesh_dispatch
  assert "collide_sdf(" not in mesh_dispatch
  assert "collide_pair_mesh_sdf_cached(" in mesh_producer
  assert "ContactGeom con[50]" not in mesh_producer
  assert "collide_sdf_cached(" not in mesh_producer
  assert "collide_pair(" not in row_assembler
  assert "collide_sdf(" not in row_assembler
  assert "ContactGeom con[50]" not in row_assembler
  assert "contact_count[world*npairs+pair_idx]" in row_assembler
  assert "contact_records[contact_base+12]" in row_assembler


def test_pair_ownership_mask_count_and_order_contracts():
  coupled = (SHADERS / "coupled_constraints.metal").read_text()
  host = (ROOT / "mujoco_metal" / "coupled_constraints.py").read_text()
  generic = _function(coupled, "kernel void contact_produce_generic(")
  sdf_seed_init = _function(coupled, "kernel void contact_sdf_seed_init(")
  sdf_finalize = _function(coupled, "kernel void contact_finalize_sdf(")
  sdf_decode = _function(coupled, "inline bool sdf_stage_decode(")
  rows = _function(coupled, "kernel void contact_normal(")

  assert generic.index("if (ta==8 || tb==8) return;") < generic.index(
      "contact_count[count_index]=0;")
  # Finalization owns analytic-SDF counts. Eligibility, world masking, and
  # packed-pair pruning all precede the first-tile count reset.
  assert sdf_finalize.index(
      "if ((ta!=8 && tb!=8) || ta==7 || tb==7) return;") < sdf_finalize.index(
          "contact_count[count_index]=0;")
  assert sdf_finalize.index("if (!pair_contact_world_enabled(dims,world)) return;") < (
      sdf_finalize.index("contact_count[count_index]=0;"))
  assert sdf_finalize.index(
      "logical_to_packed[world*npairs+pair_idx]<0") < sdf_finalize.index(
          "contact_count[count_index]=0;")
  assert "if (!pair_contact_world_enabled(dims,world)) return;" in generic
  mesh = _function(coupled, "kernel void contact_produce_mesh_sdf(")
  assert "!pair_contact_world_enabled(dims, world)" in sdf_seed_init
  assert "if (!pair_contact_world_enabled(dims,world)) return;" in mesh
  assert "logical_to_packed[world*npairs+pair_idx]<0" in generic
  assert "logical_to_packed[world * npairs + pair_idx] < 0" in sdf_decode
  assert "logical_to_packed[world*npairs+pair_idx]<0" in sdf_finalize
  assert "if (seed_start==0) contact_count[count_index]=0;" in sdf_finalize
  assert "if (contact_count[count_index]<0) return;" in sdf_finalize
  assert "if (ncon>=max_con)" in sdf_finalize
  assert "contact_count[count_index]=-1;" in sdf_finalize
  assert "contact_count[count_index]=ncon;" in sdf_finalize
  assert "logical_to_packed[world*npairs+pair_idx]<0" in mesh
  assert "ncon=clamp(ncon,0,max_con);" in generic
  assert "contact_count[count_index]=clamp(ncon,0,max_con);" in mesh
  assert "for (int k=0;k<ncon;++k)" in generic
  assert "contact_pair_cache_store" not in mesh
  assert "contact_count[count_index]=clamp(ncon,0,max_con);" in mesh
  assert "world*ncontacts_max+offset+k" in generic
  assert "pair_types = [(int(model.geom_type[int(a)]), int(model.geom_type[int(b)]))" in host
  assert "if self._has_analytic_sdf_pairs else None" in host
  assert "self._library.contact_produce_generic" in host
  assert "self._library.contact_sdf_seed_init" in host
  assert "range(0, self._sdf_seed_count, self._sdf_seed_tile)" in host
  assert "self._contact_sdf_finalize_producer(" in host
  tile_loop = host[host.index("for seed_start in seed_starts:"):]
  tile_loop = tile_loop[:tile_loop.index(
      "if self._contact_mesh_sdf_producer is not None:")]
  assert "tile_count = min(self._sdf_seed_tile," in tile_loop
  assert "max(0, self._sdf_seed_count - seed_start)" in tile_loop
  assert "[tile_count, seed_start, seed_capacity, mode, step]" in tile_loop
  init_pos = tile_loop.index("self._contact_sdf_seed_init(")
  step_pos = tile_loop.index("for step in range(self._sdf_max_iterations)")
  normal_pos = tile_loop.index("self._contact_sdf_publish_normal(")
  contact_pos = tile_loop.index("self._contact_sdf_publish_contact(")
  final_pos = tile_loop.index("self._contact_sdf_finalize_producer(", contact_pos)
  assert init_pos < step_pos < normal_pos < contact_pos < final_pos
  assert host.index("self._contact_sdf_seed_init(") < host.index(
      "self._contact_kernel(")
  assert host.index("self._contact_generic_producer(") < host.index("self._contact_kernel(")
  assert '"contact_pair_count"' in host and '"contact_pair_records"' in host
  assert "_pair_contact_cache_bytes(\n        b, d.ncontacts_max, d.npairs, sdf_seed_tile)" in host
  assert "contact_produce_mesh_sdf" in host
  # The producer cache is transient per assembly; source row/cache restore copies
  # the assembled rows and frames rather than restoring stale producer output.
  restore = host[host.index("for destination_name, source_name in ("):]
  restore = restore[:restore.index("if self._jacobian_layout.mode == 0:")]
  assert "contact_pair_records" not in restore


def test_sdf_candidate_order_and_strict_normalization_remain_in_source():
  # This split leaves the exact pinned descent and mesh traversal source intact.
  # These assertions keep source-order candidate selection and finite-gradient
  # filtering visible in the base SDF implementation used for composition.
  source = (SHADERS / "sdf_narrowphase.metal").read_text()
  analytic = _function(source, "inline int collide_sdf(")
  analytic_cached = _function(source, "inline int collide_sdf_cached(")
  mesh = _function(source, "inline int collide_mesh_sdf(")
  descent = _function(source, "inline SdfWide sdf_descend(")
  assert "for (int j = 0; j < maxn; ++j)" in analytic
  assert "if (sw_sign(dint) > 0) continue;" in analytic
  assert "sdf_vector_normalize(g_pair)" in analytic
  assert "for (int k = 0; k < ncon; ++k)" in analytic
  assert "accepted_points[ncon]=x;" in analytic
  # The legacy cached helper handles one caller-selected seed; the host's
  # staged path iterates all source seeds in bounded tiles.
  assert "SdfPointPair accepted_points[50]" not in analytic_cached
  assert "int j = seed_index;" in analytic_cached
  assert "int candidate_base=record_base*25;" in analytic_cached
  assert "contact_records[candidate_base+8]=x.tail.z;" in analytic_cached
  assert "contact_records[output_base+12]=t2.z;" in analytic_cached
  assert "sdf_source_gradient_invalid(grad)" in descent
  assert "if (sdf_jet_gt(accepted_dist, dist0))" in descent
  mesh_cached = _function(source, "inline int collide_mesh_sdf_cached(")
  assert "SdfPointPair candidates[SDF_MESH_MAX_CANDIDATES]" not in mesh_cached
  assert "SdfPointPair accepted_points[SDF_MESH_MAX_CANDIDATES]" not in mesh_cached
  assert "int stack[64];" in mesh_cached
  assert "sdf_mesh_face_candidates_cached" in mesh_cached
  assert "if (ncon >= maxn) break;" in mesh_cached
  assert "int candidate_base=(scratch_base+i)*12;" in mesh_cached
  assert "records[accepted_base+0]=x.hi.x" in mesh_cached
  assert "for(int i=0;i<ncandidate;++i)" in mesh
  assert "if(ncandidate>=SDF_MESH_MAX_CANDIDATES)return;" in source
  assert "accepted_points[ncon]=x;" in mesh


def _tiny_sdf_model(mujoco):
  vertices = ("-0.05 -0.05 -0.04 0.05 -0.05 -0.04 "
              "0.05 0.05 -0.04 -0.05 0.05 -0.04 "
              "-0.05 -0.05 0.04 0.05 -0.05 0.04 "
              "0.05 0.05 0.04 -0.05 0.05 0.04")
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><asset><mesh name="box" vertex="{vertices}"/></asset>'
      '<option timestep="0.002" sdf_initpoints="4"/>'
      '<worldbody><geom type="sdf" mesh="box" contype="1" conaffinity="1"/>'
      '<body pos="0 0 0.2"><freejoint/>'
      '<geom type="sphere" size="0.04" contype="1" conaffinity="1"/>'
      '</body></worldbody></mujoco>')


def _tiny_mesh_sdf_model(mujoco):
  vertices = ("-0.05 -0.05 -0.04 0.05 -0.05 -0.04 "
              "0.05 0.05 -0.04 -0.05 0.05 -0.04 "
              "-0.05 -0.05 0.04 0.05 -0.05 0.04 "
              "0.05 0.05 0.04 -0.05 0.05 0.04")
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><asset><mesh name="box" vertex="{vertices}"/></asset>'
      '<option timestep="0.002" sdf_initpoints="4"/>'
      '<worldbody><geom type="sdf" mesh="box" contype="1" conaffinity="1"/>'
      '<body pos="0 0 0.2"><freejoint/>'
      '<geom type="mesh" mesh="box" contype="1" conaffinity="1"/>'
      '</body></worldbody></mujoco>')


def test_native_tiny_sdf_producer_pipeline_selector():
  import os
  import mujoco
  import pytest
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native producer PSO smoke is parent-dispatched")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  model = _tiny_sdf_model(mujoco)
  stage = MetalCoupledConstraints(model, batch_size=1)
  assert stage._contact_sdf_seed_init is not None
  assert stage._contact_sdf_finalize_producer is not None
  assert stage._contact_kernel is not None
  assert stage._contact_generic_producer is None


def test_native_tiny_generic_producer_pipeline_selector():
  import os
  import mujoco
  import pytest
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native producer PSO smoke is parent-dispatched")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
      '<body pos="0 0 .05"><freejoint/><geom type="sphere" size=".05"/>'
      '</body></worldbody></mujoco>')
  stage = MetalCoupledConstraints(model, batch_size=1)
  assert stage._contact_generic_producer is not None
  assert stage._contact_kernel is not None
  assert stage._contact_sdf_producer is None


def test_native_tiny_mesh_sdf_producer_pipeline_selector():
  import os
  import mujoco
  import pytest
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native producer PSO smoke is parent-dispatched")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  model = _tiny_mesh_sdf_model(mujoco)
  stage = MetalCoupledConstraints(model, batch_size=1)
  assert stage._contact_mesh_sdf_producer is not None
  assert stage._contact_kernel is not None
  assert stage._contact_sdf_seed_init is None


def test_contact_cache_capacity_matches_exact_allocations():
  from mujoco_metal.coupled_constraints import _pair_contact_cache_bytes
  def independently_summed_bytes(batch, ncontacts, npairs, tile):
    # Each plane is a separate float32 or int32 allocation. Empty planes keep
    # one-word sentinels; the reusable dispatch-parameter plane has five ints.
    count = max(batch * npairs, 1)
    records = max(batch * ncontacts * 25, 1)
    mesh_candidates = max(batch * npairs * 50 * 12, 1)
    seed_records = max(batch * npairs * tile * 25, 1)
    seed_valid = max(batch * npairs * tile, 1)
    seed_state = max(batch * npairs * tile * 159, 1)
    seed_control = max(batch * npairs * tile, 1)
    stage_dims = 5
    return 4 * (count + records + mesh_candidates + seed_records + seed_valid
                + seed_state + seed_control + stage_dims)

  assert _pair_contact_cache_bytes(1, 8, 2) == independently_summed_bytes(1, 8, 2, 0) == 5644
  assert _pair_contact_cache_bytes(3, 11, 5) == independently_summed_bytes(3, 11, 5, 0) == 39396
  assert _pair_contact_cache_bytes(1, 50, 1, 32) == independently_summed_bytes(1, 50, 1, 32) == 31232
  assert _pair_contact_cache_bytes(1, 0, 0) == independently_summed_bytes(1, 0, 0, 0) == 48
  host = (ROOT / "mujoco_metal" / "coupled_constraints.py").read_text()
  assert '"contact_pair_scratch": empty(max(b * d.npairs * 50 * 12, 1))' in host
  assert "self._sdf_stage_dims = torch.zeros(5, dtype=torch.int32" in host
  assert '"contact_sdf_seed_state": empty(' in host
  assert '"contact_sdf_seed_ctrl": torch.zeros(' in host
  assert '"contact_sdf_seed_records": empty(' in host
  assert '"contact_sdf_seed_valid": torch.zeros(' in host
