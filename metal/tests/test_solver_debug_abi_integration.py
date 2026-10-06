# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise the real host solver ABI packer and component-tail writer.

This catches multiworld addressing defects before dispatch. Numerical shader
parity remains a separate opt-in qualification gate.
"""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize('solver', [mujoco.mjtSolver.mjSOL_PGS,
                                    mujoco.mjtSolver.mjSOL_CG,
                                    mujoco.mjtSolver.mjSOL_NEWTON])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('nv,nr', [(7, 23), (33, 103)])
def test_real_solver_dims_and_component_writes_preserve_world_intervals(solver, sparse, nv, nr):
  torch = pytest.importorskip('torch')
  from mujoco_metal.capacity import solver_debug_layout
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  cc = object.__new__(MetalCoupledConstraints)
  b, ntree, nbody = 3, 4, nv + 2
  nnz, words = (nv * 2, nv * 3 + 7) if sparse else (0, 0)
  cc._torch, cc._device, cc.batch_size = torch, torch.device('cpu'), b
  cc._solver_island_ntree, cc._solver_island_nbody = ntree, nbody
  cc._solver_island_dof_tree = np.arange(nv, dtype=np.int32) % ntree
  cc._solver_island_row_metadata = np.zeros((nr, 10), dtype=np.int32)
  cc._component_operator_ncomponent = ntree if sparse else 0
  cc._component_operator_nnz = nnz
  cc._component_mass_storage = 'block_sparse' if sparse else 'dense'
  # Values larger than float32's exact integer range catch accidental numeric
  # conversion of the int32 metadata copied into the debug float backing.
  packed = torch.arange(words, dtype=torch.int32) + (1 << 24)
  cc._component_operator_layout_device = packed if sparse else None
  cc.descriptor = SimpleNamespace(nq=nv, nv=nv, njnt=nv, neq=0,
      ncontacts_max=0, disableflags=0, refsafe=True, iterations=13,
      nr=nr, cone_type=0, nbody=nbody, nsite=0, n_eq_rows=0,
      ntendon=0, ten_base=nr, ten_friction_rows=0, ten_limit_rows=0,
      noslip_iterations=2, solver_type=int(solver), line_search_iterations=11,
      flex_contact_descriptor=None, flex_contact_base=nr, nr_joint=nr,
      n_flex_contact_rows=0)
  dims = cc._solver_dimension_tensor()
  layout = solver_debug_layout(nv, nr, solver, ntree, nbody, nnz, words,
                               cc._component_mass_storage)
  assert int(dims[21]) == layout['final_stride']
  header = int(dims[24]) + nr * 10
  assert dims[header:header+8].tolist() == [
      cc._component_operator_ncomponent, nnz, layout['component_armature_offset'],
      layout['component_layout_offset'], layout['base_stride'],
      layout['qacc_warmstart_offset'], layout['equality_jacobian_offset'],
      layout['component_operator_offset']]
  assert dims[header+8:header+11].tolist() == [
      layout['block_workspace_offset'],
      layout['primal_workspace_offset'],
      layout['z_workspace_offset']]
  # The solver-mode sentinel is temporarily written into scalar dim 20 for
  # split sparse stages. Preserve the compiled line-search budget in immutable
  # metadata so sparse CG/Newton still honor model.opt.ls_iterations.
  assert int(dims[header+14]) == cc.descriptor.line_search_iterations
  assert int(dims[header+15]) == layout["high_low_workspace_offset"]
  assert int(dims[header+16]) == layout["high_low_workspace_length"]
  row_metadata_begin = layout['block_workspace_offset']
  row_metadata_end = row_metadata_begin + 12 * nr
  z_begin = layout['z_workspace_offset']
  z_end = z_begin + nv * nr
  assert layout['primal_workspace_offset'] <= row_metadata_begin
  assert row_metadata_end == z_begin
  assert z_end <= (layout['base_stride'] - layout['debug_prefix'])
  # Actual packed awake-map address must end before dynamic equality scratch.
  prefix = layout['debug_prefix']
  assert prefix + int(dims[27]) + max(ntree, 1) <= layout['equality_jacobian_offset']
  stride = int(dims[21])
  cc._debug_stride = stride
  storage = torch.full((b * stride,), -17., dtype=torch.float32)
  cc._workspace = {'workspace_debug': storage}
  worlds = storage.reshape(b, stride)
  eq_offset, warm_offset = int(dims[header+6]), int(dims[header+5])
  for world in range(b):
    worlds[world, eq_offset:eq_offset+12*nv] = 100 + world
    worlds[world, warm_offset:warm_offset+nv] = 200 + world
  prefix_before = worlds[:, :layout['base_stride']].clone()
  if sparse:
    cc._component_operator_armature_offset = int(dims[header+2])
    cc._component_operator_layout_offset = int(dims[header+3])
    armature = torch.arange(b * nnz, dtype=torch.float32).reshape(b, nnz)
    cc._stage_component_operator(armature)
    torch.testing.assert_close(worlds[:, :layout['base_stride']], prefix_before)
    start = int(dims[header+2])
    torch.testing.assert_close(worlds[:, start:start+nnz], armature)
    start = int(dims[header+3])
    assert torch.equal(worlds.view(torch.int32)[:, start:start+words], packed[None].expand(b, -1))
  # Decode exact shader-facing world addresses, rather than only comparing
  # two layouts generated by the same arithmetic helper.
  for world in range(b):
    shader_world = world * int(dims[21])
    torch.testing.assert_close(storage[shader_world+eq_offset:shader_world+eq_offset+12*nv],
                               torch.full((12*nv,), 100.+world))
    torch.testing.assert_close(storage[shader_world+warm_offset:shader_world+warm_offset+nv],
                               torch.full((nv,), 200.+world))


def test_solver_diagnostics_views_use_ten_word_per_world_stride():
  """The shader writes diagnostics as [B, 10], not a packed [B, 2] prefix."""
  torch = pytest.importorskip('torch')
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  cc = object.__new__(MetalCoupledConstraints)
  cc._torch = torch
  cc.batch_size = 3
  cc._workspace = {
      'out_diagnostics': torch.arange(30, dtype=torch.float32),
  }
  diagnostics, history = cc._diagnostic_views()
  assert diagnostics.shape == (3, 2)
  assert history.shape == (3, 8)
  assert diagnostics.tolist() == [[0., 1.], [10., 11.], [20., 21.]]
  assert history.tolist() == [list(map(float, range(2, 10))),
                              list(map(float, range(12, 20))),
                              list(map(float, range(22, 30)))]
