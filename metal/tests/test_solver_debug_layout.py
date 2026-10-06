"""CPU contract tests for per-world coupled solver workspace offsets."""

import mujoco
import pytest

from mujoco_metal.capacity import primal_scratch_floats, solver_debug_layout


def test_component_tail_does_not_move_per_world_scratch_offsets():
  for solver in (mujoco.mjtSolver.mjSOL_PGS,
                 mujoco.mjtSolver.mjSOL_CG,
                 mujoco.mjtSolver.mjSOL_NEWTON):
    dense = solver_debug_layout(7, 23, solver, ntree=4, nbody=9)
    sparse = solver_debug_layout(
        7, 23, solver, ntree=4, nbody=9, component_nnz=31,
        component_layout_words=56)
    assert sparse["debug_prefix"] == dense["debug_prefix"]
    assert sparse["base_stride"] == dense["base_stride"]
    assert sparse["qacc_warmstart_offset"] == dense["qacc_warmstart_offset"]
    assert sparse["equality_jacobian_offset"] == dense["equality_jacobian_offset"]
    assert sparse["component_operator_offset"] == sparse["base_stride"]
    assert sparse["component_armature_offset"] == sparse["base_stride"]
    assert sparse["component_layout_offset"] == sparse["base_stride"] + 31
    assert sparse["high_low_debug_offset"] == sparse["base_stride"] + 31 + 56
    assert sparse["high_low_workspace_offset"] == (
        sparse["high_low_debug_offset"] - sparse["debug_prefix"])
    assert sparse["high_low_workspace_length"] == 8 * 7 + 3 * 23
    assert sparse["final_stride"] == (
        sparse["high_low_debug_offset"]
        + sparse["high_low_workspace_length"])
    # For batch 2+, the shader's world stride lands the second world at the
    # true next debug record, rather than in world zero's sparse tail.
    assert sparse["final_stride"] * 2 > sparse["base_stride"] * 2


def test_qacc_tail_and_dynamic_equality_scratch_are_disjoint():
  layout = solver_debug_layout(
      33, 103, mujoco.mjtSolver.mjSOL_NEWTON, ntree=7, nbody=35,
      component_nnz=211, component_layout_words=317)
  qacc_start = layout["qacc_warmstart_offset"]
  eq_start = layout["equality_jacobian_offset"]
  assert layout["debug_prefix"] <= eq_start
  assert eq_start + 12 * 33 == qacc_start
  assert qacc_start + 33 == layout["base_stride"]
  assert layout["base_stride"] <= layout["component_armature_offset"]
  assert layout["component_layout_offset"] + 317 == layout["high_low_debug_offset"]
  assert layout["high_low_debug_offset"] + layout["high_low_workspace_length"] == layout["final_stride"]


def test_cached_smooth_acceleration_stage_flag_composes_and_restores():
  torch = pytest.importorskip("torch")

  from mujoco_metal.coupled_constraints import (
      _cached_position_stage,
      _provided_smooth_acceleration_stage,
  )

  dims = torch.zeros(32, dtype=torch.int32)
  with _cached_position_stage(dims, refsafe=True):
    assert int(dims[7]) == 3
    try:
      with _provided_smooth_acceleration_stage(dims):
        assert int(dims[7]) == 7
        raise RuntimeError("exercise flag restoration")
    except RuntimeError:
      pass
    assert int(dims[7]) == 3
  assert int(dims[7]) == 1


def test_paired_pgs_smooth_input_flag_composes_and_restores():
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import (
      _cached_position_stage,
      _paired_smooth_input_stage,
      _provided_smooth_acceleration_stage,
  )

  dims = torch.zeros(32, dtype=torch.int32)
  with _cached_position_stage(dims, refsafe=True):
    with _provided_smooth_acceleration_stage(dims):
      with _paired_smooth_input_stage(dims):
        assert int(dims[7]) == 23
      assert int(dims[7]) == 7
    assert int(dims[7]) == 3
  assert int(dims[7]) == 1


def test_newton_hessian_scratch_is_matrix_free_and_linear_in_model_size():
  nv, nr = 8192, 32768
  cg = primal_scratch_floats(
      nv, nr, mujoco.mjtSolver.mjSOL_CG, mass_storage="block_sparse")
  newton = primal_scratch_floats(
      nv, nr, mujoco.mjtSolver.mjSOL_NEWTON, mass_storage="block_sparse")
  # Newton adds three vectors over the CG layout. The retained J-transpose
  # block scales as nv*nr; all remaining scratch terms are independently
  # accounted below, including the shared row, island, and equality maps.
  assert newton - cg == 3 * nv
  assert newton == 30 * nv + 18 * nr + nv * nr + 6
  assert newton < nv * nv + nv * nr


def test_large_dynamic_solver_intervals_are_disjoint_and_addressable():
  nv, nr = 130, 390
  row_words = 12 * nr
  z_words = nv * nr
  for solver in (mujoco.mjtSolver.mjSOL_PGS,
                 mujoco.mjtSolver.mjSOL_CG,
                 mujoco.mjtSolver.mjSOL_NEWTON):
    layout = solver_debug_layout(nv, nr, solver, ntree=5, nbody=81)
    scratch_start = layout["debug_prefix"]
    # Dense Cholesky lives at the head. The primal solver starts immediately
    # after it; row metadata follows the complete configured solver vector
    # region, and J-transpose mass solves follow the typed metadata.
    factor_words = nv * nv + 2 * nv
    assert layout["primal_workspace_offset"] == factor_words
    assert layout["block_workspace_offset"] >= factor_words
    assert layout["z_workspace_offset"] == layout["block_workspace_offset"] + row_words
    z_end = layout["z_workspace_offset"] + z_words
    assert scratch_start + z_end <= layout["base_stride"]
    assert layout["equality_jacobian_offset"] + 12 * nv == (
        layout["qacc_warmstart_offset"])
    assert layout["qacc_warmstart_offset"] + nv == layout["base_stride"]


def test_high_low_tail_is_world_local_and_after_all_existing_regions():
  for nv, nr, nnz, layout_words in ((7, 23, 31, 56), (0, 0, 0, 0)):
    for solver in (mujoco.mjtSolver.mjSOL_PGS,
                   mujoco.mjtSolver.mjSOL_CG,
                   mujoco.mjtSolver.mjSOL_NEWTON):
      layout = solver_debug_layout(
          nv, nr, solver, ntree=3, nbody=5, component_nnz=nnz,
          component_layout_words=layout_words,
          mass_storage="block_sparse")
      tail_start = layout["high_low_debug_offset"]
      tail_words = layout["high_low_workspace_length"]
      assert tail_start == layout["component_layout_offset"] + layout_words
      assert tail_words == 8 * max(nv, 1) + 3 * max(nr, 1)
      assert tail_start + tail_words == layout["final_stride"]
      assert layout["high_low_workspace_offset"] == (
          tail_start - layout["debug_prefix"])
      # Each world owns a disjoint full record, including its scratch tail.
      assert layout["final_stride"] * 2 >= tail_start + tail_words + tail_words
      # The kernel base is world-local solver_scratch, not the debug record.
      for world in (0, 1):
        debug_address = world * layout["final_stride"] + tail_start
        scratch_address = (world * layout["final_stride"]
                           + layout["debug_prefix"]
                           + layout["high_low_workspace_offset"])
        assert debug_address == scratch_address
