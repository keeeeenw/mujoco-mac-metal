# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU preflight tests for the cached coupled-solver entry point."""

from types import SimpleNamespace
from pathlib import Path

import pytest

from mujoco_metal.coupled_constraints import (
    MetalCoupledConstraints,
    _component_primal_stage,
    _cached_position_stage,
    _solver_dimension_stage,
)


def test_cached_stage_flags_restore_after_success_and_failure():
  dims = [0] * 21
  dims[7] = 1
  dims[20] = 9
  with _cached_position_stage(dims, refsafe=True):
    assert dims[7] == 3
    assert dims[20] == 9
    with _solver_dimension_stage(dims, -2, 9):
      assert dims[20] == -2
    assert dims[20] == 9
  assert dims[7] == 1

  with pytest.raises(RuntimeError, match="probe"):
    with _cached_position_stage(dims, refsafe=False):
      assert dims[7] == 2
      raise RuntimeError("probe")
  assert dims[7] == 0


def test_component_primal_cached_dispatches_mark_precomputed_acceleration():
  dims = [0] * 27
  dims[7] = 1
  dims[20] = 9
  seen = []
  for stage in (-1, -3):
    with _component_primal_stage(
        dims, refsafe=True, stage=stage, line_search_iterations=9,
        provided_smooth_acceleration=True):
      seen.append((dims[7], dims[20]))
  assert seen == [(7, -1), (7, -3)]
  assert dims[7] == 1
  assert dims[20] == 9

  with pytest.raises(RuntimeError, match="dispatch"):
    with _component_primal_stage(
        dims, refsafe=True, stage=-3, line_search_iterations=9,
        provided_smooth_acceleration=True):
      assert dims[7] == 7
      assert dims[20] == -3
      raise RuntimeError("dispatch")
  assert dims[7] == 1
  assert dims[20] == 9


def test_cached_solver_rejects_foreign_or_stale_context_before_device_access():
  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  solver._assembly_generation = 4
  solver._position_context_valid = True
  solver._position_context_epoch = 4
  foreign = {"_owner": object(), "_epoch": 4}
  stale = {"_owner": solver, "_epoch": 3}
  for context in (foreign, stale):
    with pytest.raises(ValueError, match="stale or belongs"):
      solver.run_velocity_device(context, None, None, None, None, None, None)


def test_position_context_capture_requires_assembly_and_exact_nonempty_context():
  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  solver._assembly_generation = 0
  with pytest.raises(RuntimeError, match="completed row assembly"):
    solver.capture_position_context()


def test_failed_replacement_capture_invalidates_the_previous_epoch_before_writes():
  torch = pytest.importorskip("torch")
  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  solver._assembly_generation = 1
  solver._position_current_valid = True
  solver._position_current_eq_jdot_inputs_valid = False
  solver._position_context_valid = True
  solver._position_context_epoch = 7
  solver._torch = torch
  solver._device = torch.device("cpu")
  solver.descriptor = SimpleNamespace(nr=1, nv=2, neq=0)
  solver.batch_size = 1
  prior = {"_owner": solver, "_epoch": 7}
  with pytest.raises(ValueError, match="position_context"):
    solver.capture_position_context(position_context=torch.zeros((1, 2, 5)))
  assert solver._position_context_valid is False
  assert solver._position_context_epoch == 8
  assert prior["_epoch"] != solver._position_context_epoch


def test_cached_stage_requires_dense_dispatch_before_mutation():
  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  solver._assembly_generation = 1
  solver._position_context_epoch = 1
  solver._position_context_valid = True
  solver._torch = SimpleNamespace()
  solver.descriptor = SimpleNamespace(nv=1, nr=1, dense_path=False)
  solver.batch_size = 1
  context = {"_owner": solver, "_epoch": 1}
  with pytest.raises(NotImplementedError, match="dense mass or a component solver"):
    solver.run_velocity_device(context, None, None, None, None, None, None)


def test_cached_stage_rejects_missing_context_tensor_before_workspace_write():
  torch = pytest.importorskip("torch")
  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  solver._torch = torch
  solver._device = torch.device("cpu")
  solver.descriptor = SimpleNamespace(nv=1, nq=1, nr=1, nbody=1, neq=0,
                                     dense_path=True, refsafe=False)
  solver.batch_size = 1
  solver._position_context_valid = True
  solver._position_context_epoch = 2
  solver._solver_island_ntree = 1
  solver._workspace = {"workspace_debug": torch.full((1, 9), 17.0)}
  solver._debug_stride = 9
  solver._position_context_epoch = 2
  context = {
      "_owner": solver,
      "_epoch": 2,
      "position_context": torch.zeros((1, 1, 5)),
      "surface_velocity": None,
      "extra_aref": torch.zeros((1, 1)),
  }
  before = solver._workspace["workspace_debug"].clone()
  with pytest.raises(ValueError, match="malformed"):
    solver.run_velocity_device(
        context, None, torch.eye(1).reshape(1, 1, 1),
        torch.zeros((1, 1)), torch.zeros((1, 1)), torch.zeros((1, 1)))
  assert torch.equal(solver._workspace["workspace_debug"], before)


def test_cached_component_exact_rows_are_computed_before_borrowed_rhs_solve():
  """The helper's borrowed output is overwritten only after exact preparation."""
  import inspect
  from mujoco_metal.coupled_constraints import _run_component_solve_after_prepare

  source = inspect.getsource(MetalCoupledConstraints.run_velocity_device)
  prepare_start = source.index("      def prepare_component_rhs():")
  exact_call = source.index("self.recompute_exact_impedance_device(",
                            prepare_start)
  pack_rhs = source.index("self._pack_component_mass_rhs(", exact_call)
  sequence_call = source.index("_run_component_solve_after_prepare(", pack_rhs)
  solve_callback = source.index("lambda: component_solver.run_device(",
                               sequence_call)
  paired_dispatch = source.index(
      "dispatch_solver(component_solver.paired_output)", solve_callback)
  assert prepare_start < exact_call < pack_rhs < sequence_call < solve_callback < paired_dispatch
  assert "rhs_low=" in source[solve_callback:paired_dispatch]

  # Exercise the production ordering helper with a borrowed backing. The
  # exact mass-row preparation writes it first; the smooth-RHS solve is the
  # final writer and its paired output is what gets dispatched.
  shared = {"result": [None], "exact_status": [0]}
  events = []

  def prepare():
    events.append("exact")
    shared["result"][0] = "exact-mass-row-result"
    shared["exact_status"][0] = 7
    return {"status": shared["exact_status"]}

  def solve():
    events.append("smooth")
    shared["result"][0] = "smooth-rhs-paired-result"
    return shared["result"], "smooth-status"

  prepared, solved = _run_component_solve_after_prepare(prepare, solve)
  assert events == ["exact", "smooth"]
  assert prepared["status"] == [7]
  assert solved == (shared["result"], "smooth-status")
  assert solved[0][0] == "smooth-rhs-paired-result"

  helper_source = inspect.getsource(_run_component_solve_after_prepare)
  assert helper_source.index("prepared = prepare()") < helper_source.index("solved = solve()")


def test_masked_cached_refresh_copy_preserves_unselected_world_records():
  torch = pytest.importorskip("torch")

  class CpuKernelLibrary:
    @staticmethod
    def _copy(source, destination, dims, *, add=False, value=None, **_):
      batch, width, source_stride, destination_stride, source_offset, destination_offset = map(
          int, dims[:6].tolist())
      mask = dims[6:6 + batch]
      for world in range(batch):
        if mask[world] == 0:
          continue
        dst = destination[world * destination_stride + destination_offset:
                          world * destination_stride + destination_offset + width]
        if value is not None:
          dst.fill_(value)
        else:
          src = source[world * source_stride + source_offset:
                       world * source_stride + source_offset + width]
          if add:
            dst.add_(src)
          else:
            dst.copy_(src)

    def copy_selected_world_float(self, source, destination, dims, **kwargs):
      self._copy(source, destination, dims, **kwargs)

    def copy_selected_world_int(self, source, destination, dims, **kwargs):
      self._copy(source, destination, dims, **kwargs)

    def add_selected_world_float(self, source, destination, dims, **kwargs):
      self._copy(source, destination, dims, add=True)

    def clear_selected_world_int(self, destination, dims, **kwargs):
      self._copy(None, destination, dims, value=0)

    def fill_selected_world_int_one(self, destination, dims, **kwargs):
      self._copy(None, destination, dims, value=1)

  solver = MetalCoupledConstraints.__new__(MetalCoupledConstraints)
  solver._torch = torch
  solver._device = torch.device("cpu")
  solver.batch_size = 3
  solver._world_mask_offset = 0
  solver._constants = {"solver_dims": torch.tensor([1, 0, 1], dtype=torch.int32)}
  solver._position_refresh_copy_dims = torch.zeros((9,), dtype=torch.int32)
  solver._library = CpuKernelLibrary()
  mask = torch.tensor([1, 0, 1], dtype=torch.int32)

  source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
  destination = torch.full((3, 7), -5., dtype=torch.float32)
  solver._copy_world_masked(
      destination, source, mask, width=2, source_stride=4,
      destination_stride=7, source_offset=1, destination_offset=3)
  torch.testing.assert_close(destination[0, 3:5], source[0, 1:3])
  torch.testing.assert_close(destination[2, 3:5], source[2, 1:3])
  torch.testing.assert_close(destination[1], torch.full((7,), -5.))

  int_destination = torch.full((3, 7), -5, dtype=torch.int32)
  solver._fill_world_masked(
      int_destination, mask, width=2, stride=7, offset=0, value=1)
  torch.testing.assert_close(int_destination[0, :2], torch.ones(2, dtype=torch.int32))
  torch.testing.assert_close(int_destination[2, :2], torch.ones(2, dtype=torch.int32))
  torch.testing.assert_close(int_destination[1, :2], torch.full((2,), -5, dtype=torch.int32))
def test_equality_producer_publishes_cached_row_ownership_and_bilateral_bounds():
  """Equality rows must survive cache capture as active canonical rows."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "equality_assembly.metal").read_text()
  producer = shader.split("kernel void equality_assembly(", 1)[1]
  producer = producer.split("kernel void ", 1)[0]
  publish = producer.rsplit("Publish ownership/activity", 1)[1]
  assert "typ != 0 && typ != 1 && typ != 2" in publish
  assert "R > 0.0f || abs(ar) > 0.0f" in publish
  assert "dbg[nr * nr + 4 * nr + row] = row_enabled ? -INFINITY : 0.0f" in publish
  assert "dbg[nr * nr + 5 * nr + row] = row_enabled ? INFINITY : 0.0f" in publish
  assert "dbg[nr * nr + 6 * nr + row] = row_enabled ? 1.0f : 0.0f" in publish

  refresh_shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
                    / "coupled_constraints.metal").read_text()
  refresh = refresh_shader.split(
      "kernel void refresh_cached_constraint_aref(", 1)[1]
  refresh = refresh.split("kernel void ", 1)[0]
  assert "workspace_debug[debug_base + active_offset] <= 0.5f" in refresh
  assert "workspace_debug[debug_base + aref_offset] = 0.0f" in refresh


def test_tendon_reinitialization_clears_only_tendon_owned_row_metadata():
  """Tendon refresh may not erase equality/contact/flex cache ownership."""
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  tendon = shader.split("kernel void tendon_constraint_rows(", 1)[1]
  tendon = tendon.split("kernel void ", 1)[0]
  assert "int ten_stop = min(nr, ten_base + max(dims[17], 0));" in tendon
  assert "for (int r=max(0, ten_base);r<ten_stop;++r)" in tendon
  assert "for (int r=0;r<nr;++r)" not in tendon
  # Tendon equalities own just their compiled source row; all other equality
  # families keep the metadata published by equality_assembly.
  assert "if (eq_type[e] == 3)" in tendon
  assert "dbg[nr*nr+6*nr+row]=0.0f;" in tendon

  # Both coupled solver variants preserve the equality producer's zero bounds
  # for inactive rows instead of manufacturing bilateral bounds on replay.
  dense = shader.split("kernel void solve_coupled_constraints(", 1)[1]
  dense = dense.split("kernel void ", 1)[0]
  block = shader.split("kernel void solve_coupled_constraints_block(", 1)[1]
  block = block.split("kernel void ", 1)[0]
  bounds = "dbg[nr * nr + 4 * nr + r]"
  assert f"lo[r] = {bounds};" in dense
  assert f"hi[r] = dbg[nr * nr + 5 * nr + r];" in dense
  assert f"lo[r] = {bounds};" in block
  assert f"hi[r] = dbg[nr * nr + 5 * nr + r];" in block
