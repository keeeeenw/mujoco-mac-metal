# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracle and opt-in native gates for exact sparse component solves."""

import os

import numpy as np
import pytest

from mujoco_metal.component_solve import (
    build_euler_diagonal_cpu,
    component_solver_workspace_sizes,
    _component_layout_array,
    component_mass_matvec_cpu,
    solve_component_blocks_cpu,
)


def _layout():
  return {
      "nv": 3,
      "ncomponent": 2,
      "nnz": 5,
      "component_dof_offsets": np.asarray([0, 2, 3], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 2, 1], dtype=np.int32),
      "component_mass_offsets": np.asarray([0, 4], dtype=np.int32),
  }


def test_component_layout_packing_and_workspace_sizes_match_tensor_shapes():
  ncomponent, nnz, packed = _component_layout_array(_layout(), 3)
  assert (ncomponent, nnz) == (2, 5)
  # [offsets(3), DOF ids(3), mass offsets(2)] exactly matches the MSL ABI.
  np.testing.assert_array_equal(packed, [0, 2, 3, 0, 2, 1, 0, 4])
  sizes = component_solver_workspace_sizes(
      batch=2, nv=3, ncomponent=2, nnz=5, rhs_capacity=4,
      layout_size=packed.size)
  assert sizes == {
      "component_mass_dof_mask": 6,
      "component_mass_factor_dof_mask": 6,
      "component_mass_factor": 10,
      "component_mass_output": 24,
      "component_mass_single_output": 6,
      "component_mass_matvec_output": 6,
      "component_mass_status": 4,
      "component_mass_factor_status": 4,
      "component_mass_zero_blocks": 10,
      "component_mass_layout": 8,
      "component_mass_default_dof_ids": 6,
      "component_mass_default_counts": 6,
      "component_mass_diagonal_add": 6,
      "component_mass_damping_mask": 3,
      "component_mass_euler_dt": 1,
      "component_mass_allow_indefinite": 1,
      "component_mass_phase": 1,
      "component_mass_dims": 5,
      "component_mass_single_dims": 5,
      "component_mass_merge_dims": 2,
  }


@pytest.mark.parametrize("field,value", [
    ("batch", True), ("batch", 1.5), ("nv", 2.5),
    ("ncomponent", False), ("nnz", 1.25),
    ("rhs_capacity", True), ("layout_size", 4.5),
])
def test_component_workspace_rejects_boolean_and_fractional_dimensions(
    field, value):
  kwargs = dict(batch=1, nv=2, ncomponent=1, nnz=4,
                rhs_capacity=2, layout_size=4)
  kwargs[field] = value
  with pytest.raises(ValueError, match="integer"):
    component_solver_workspace_sizes(**kwargs)


def test_component_layout_rejects_fractional_and_overlapping_indices():
  malformed = _layout()
  malformed["component_dof_ids"] = np.asarray([0.0, 2.0, 1.0])
  with pytest.raises(ValueError, match="integer storage"):
    _component_layout_array(malformed, 3)
  malformed = _layout()
  malformed["component_mass_offsets"] = np.asarray([0, 3], dtype=np.int32)
  with pytest.raises(ValueError, match="densely partition"):
    _component_layout_array(malformed, 3)


def test_oversized_component_batch_rejected_before_torch_import(monkeypatch):
  import builtins
  from types import SimpleNamespace
  from mujoco_metal.component_solve import MetalComponentMassSolver

  original_import = builtins.__import__
  def guarded_import(name, *args, **kwargs):
    if name == "torch":
      raise AssertionError("torch imported before workspace admission")
    return original_import(name, *args, **kwargs)
  monkeypatch.setattr(builtins, "__import__", guarded_import)
  with pytest.raises(ValueError, match="batch_size exceeds.*int32"):
    MetalComponentMassSolver(
        SimpleNamespace(nv=3, nbody=4), _layout(),
        batch_size=1 << 31, rhs_capacity=1)


def test_invalid_euler_damping_mask_rejected_before_torch_import(monkeypatch):
  import builtins
  from types import SimpleNamespace
  from mujoco_metal.component_solve import MetalComponentMassSolver

  original_import = builtins.__import__
  def guarded_import(name, *args, **kwargs):
    if name == "torch":
      raise AssertionError("torch imported before Euler mask validation")
    return original_import(name, *args, **kwargs)
  monkeypatch.setattr(builtins, "__import__", guarded_import)
  model = SimpleNamespace(nv=3, nbody=4)
  with pytest.raises(ValueError, match="euler_damping_dofs must be boolean"):
    MetalComponentMassSolver(
        model, _layout(), batch_size=1, rhs_capacity=1,
        euler_damping_dofs=np.ones(3, dtype=np.float32))
  with pytest.raises(ValueError, match="euler_damping_dofs must be boolean"):
    MetalComponentMassSolver(
        model, _layout(), batch_size=1, rhs_capacity=1,
        euler_damping_dofs=np.ones(2, dtype=bool))


def test_component_solve_cpu_uses_exact_components_and_active_principal_system():
  layout = _layout()
  mass = np.asarray([[4, 1, 1, 3, 2]], dtype=np.float64)
  armature = np.asarray([[1, 0.5, 0.5, 2, 0]], dtype=np.float64)
  rhs = np.asarray([[[1, 4, 2], [3, 6, 5]]], dtype=np.float64)
  solution, status = solve_component_blocks_cpu(
      mass, layout, rhs, tendon_armature_blocks=armature)
  np.testing.assert_array_equal(status, [[0, 0]])
  matrix = np.zeros((3, 3), dtype=np.float64)
  ids = layout["component_dof_ids"]
  matrix[np.ix_(ids[:2], ids[:2])] = np.asarray([[5, 1.5], [1.5, 5]])
  matrix[1, 1] = 2
  np.testing.assert_allclose(solution[0], np.linalg.solve(matrix, rhs[0].T).T,
                             rtol=1e-13, atol=1e-13)

  active_solution, active_status = solve_component_blocks_cpu(
      mass, layout, rhs, active_dof=np.asarray([[True, True, False]]),
      tendon_armature_blocks=armature)
  np.testing.assert_array_equal(active_status, [[0, 0]])
  np.testing.assert_allclose(active_solution[0, :, 0], rhs[0, :, 0] / 5)
  np.testing.assert_allclose(active_solution[0, :, 1], rhs[0, :, 1] / 2)
  np.testing.assert_array_equal(active_solution[0, :, 2], 0)
  vector = np.asarray([[.5, -2., 3.]], dtype=np.float64)
  product = component_mass_matvec_cpu(
      mass, layout, vector, tendon_armature_blocks=armature)
  expected_matrix = np.asarray([[5., 0., 1.5],
                                [0., 2., 0.],
                                [1.5, 0., 5.]])
  np.testing.assert_allclose(product[0], expected_matrix @ vector[0])
  active_product = component_mass_matvec_cpu(
      mass, layout, vector, active_dof=np.asarray([[True, True, False]]),
      tendon_armature_blocks=armature)
  np.testing.assert_allclose(active_product[0], [2.5, -4., 0.])


def test_component_solve_cpu_reports_nonpositive_factor_per_component():
  layout = _layout()
  mass = np.asarray([[1, 0, 0, -1, 2]], dtype=np.float64)
  rhs = np.ones((1, 1, 3), dtype=np.float64)
  solution, status = solve_component_blocks_cpu(mass, layout, rhs)
  np.testing.assert_array_equal(status, [[1, 0]])
  np.testing.assert_allclose(solution[0, 0, 1], 0.5)
  np.testing.assert_array_equal(solution[0, 0, [0, 2]], [0, 0])


def test_component_solve_cpu_accepts_tiny_positive_mass_pivot():
  layout = {
      "nv": 2, "ncomponent": 1, "nnz": 4,
      "component_dof_offsets": np.asarray([0, 2], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 1], dtype=np.int32),
      "component_mass_offsets": np.asarray([0], dtype=np.int32),
  }
  mass = np.asarray([[1.0, 0.0, 0.0, 1.0e-12]])
  rhs = np.asarray([[[1.0, 1.0]]])
  solution, status = solve_component_blocks_cpu(mass, layout, rhs)
  np.testing.assert_array_equal(status, [[0]])
  np.testing.assert_allclose(solution, [[[1.0, 1.0e12]],], rtol=1e-14)


def test_component_ldlt_matches_pinned_descending_order_zero_leading_pivot():
  # In component order this is [[0,1],[1,1]]. Ascending LDL encounters a
  # zero first pivot; pinned mj_factorI works from the high DOF downward and
  # accepts the nonsingular block. The compiled component order is reversed
  # relative to the public/global RHS coordinates to exercise mapped scatters.
  layout = {
      "nv": 2, "ncomponent": 1, "nnz": 4,
      "component_dof_offsets": np.asarray([0, 2], dtype=np.int32),
      "component_dof_ids": np.asarray([1, 0], dtype=np.int32),
      "component_mass_offsets": np.asarray([0], dtype=np.int32),
  }
  mass = np.asarray([[0., 1., 1., 1.]])
  rhs = np.asarray([[[2., 1.], [0., 1.]]])
  solution, status = solve_component_blocks_cpu(
      mass, layout, rhs, allow_indefinite=True)
  np.testing.assert_array_equal(status, [[0]])
  global_mass = np.asarray([[1., 1.], [1., 0.]])
  np.testing.assert_allclose(
      solution[0], np.linalg.solve(global_mass, rhs[0].T).T,
      rtol=1e-14, atol=1e-14)
  # Sleep filtering removes the global DOF 1 principal row and leaves the
  # inactive coordinate at zero for every RHS.
  awake, awake_status = solve_component_blocks_cpu(
      mass, layout, rhs, active_dof=np.asarray([[True, False]]),
      allow_indefinite=True)
  np.testing.assert_array_equal(awake_status, [[0]])
  np.testing.assert_allclose(awake[0, :, 0], rhs[0, :, 0])
  np.testing.assert_array_equal(awake[0, :, 1], 0.0)


def test_component_solve_cpu_rejects_bad_active_blocks_but_ignores_inactive_nan():
  layout = {
      "nv": 4, "ncomponent": 3, "nnz": 6,
      "component_dof_offsets": np.asarray([0, 2, 3, 4], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 1, 2, 3], dtype=np.int32),
      "component_mass_offsets": np.asarray([0, 4, 5], dtype=np.int32),
  }
  mass = np.asarray([
      [1., 0., 0., 1., 0., 1.],
      [np.nan, 0., 0., 1., 1., 1.],
      [1., 0., 0., 1., 1., 1.],
  ])
  rhs = np.ones((3, 2, 4), dtype=np.float64)
  rhs[2, 1, 0] = np.inf
  awake = np.ones((3, 4), dtype=bool)
  # The final world's second DOF in component 0 is asleep. Its NaN mass/RHS
  # values are retained but must not affect the awake principal solve.
  mass[2, 0] = 2.0
  mass[2, 1] = np.nan
  mass[2, 3] = np.nan
  rhs[2, :, 1] = np.nan
  rhs[2, 1, 0] = 1.0
  awake[2, 1] = False
  solution, status = solve_component_blocks_cpu(
      mass, layout, rhs, active_dof=awake, allow_indefinite=True)
  np.testing.assert_array_equal(status, [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
  np.testing.assert_array_equal(solution[0, :, 2], 0.0)
  np.testing.assert_array_equal(solution[1, :, :2], 0.0)
  np.testing.assert_allclose(solution[2, :, 0], .5)
  np.testing.assert_array_equal(solution[2, :, 1], 0.0)


def test_component_solve_cpu_rejects_asymmetry_and_float32_overflow():
  layout = {
      "nv": 2, "ncomponent": 1, "nnz": 4,
      "component_dof_offsets": np.asarray([0, 2], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 1], dtype=np.int32),
      "component_mass_offsets": np.asarray([0], dtype=np.int32),
  }
  rhs = np.ones((1, 2, 2), dtype=np.float64)
  asymmetric = np.asarray([[1., 0.1, 0.1001, 1.]])
  _, asymmetry_status = solve_component_blocks_cpu(
      asymmetric, layout, rhs, allow_indefinite=True)
  np.testing.assert_array_equal(asymmetry_status, [[1]])

  max32 = np.finfo(np.float32).max
  overflowing_sum = np.asarray([[max32, 0., 0., 1.]])
  armature = np.asarray([[max32, 0., 0., 0.]])
  overflow_output, overflow_status = solve_component_blocks_cpu(
      overflowing_sum, layout, rhs, tendon_armature_blocks=armature,
      allow_indefinite=True)
  np.testing.assert_array_equal(overflow_status, [[1]])
  np.testing.assert_array_equal(overflow_output, 0.0)

  finite_inputs = np.asarray([[1., 0., 0., 1.]])
  too_large_rhs = np.full((1, 2, 2), float(max32) * 2., dtype=np.float64)
  output, status = solve_component_blocks_cpu(
      finite_inputs, layout, too_large_rhs, allow_indefinite=True)
  np.testing.assert_array_equal(status, [[1]])
  np.testing.assert_array_equal(output, 0.0)


def test_component_euler_diagonal_applies_signed_derivative_for_every_awake_dof():
  q_deriv = np.asarray([[2., -20.], [-.1, -.2]])
  eligible = np.asarray([True, False])
  awake = np.asarray([[True, True], [False, True]])
  diagonal = build_euler_diagonal_cpu(q_deriv, .1, eligible, awake)
  np.testing.assert_allclose(diagonal, [[.2, -2.], [0., 0.]])

  layout = {
      "nv": 2, "ncomponent": 1, "nnz": 4,
      "component_dof_offsets": np.asarray([0, 2], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 1], dtype=np.int32),
      "component_mass_offsets": np.asarray([0], dtype=np.int32),
  }
  mass = np.asarray([[1., .1, .1, 1.]])
  rhs = np.asarray([[[1., 2.]]])
  solution, status = solve_component_blocks_cpu(
      mass, layout, rhs, diagonal_add=diagonal[:1], allow_indefinite=True)
  assert status[0, 0] == 0
  expected = np.linalg.solve(
      mass[0].reshape(2, 2) + np.diag(diagonal[0]), rhs[0, 0])
  np.testing.assert_allclose(solution[0, 0], expected)
  product = component_mass_matvec_cpu(
      mass, layout, rhs[:, 0], diagonal_add=diagonal[:1])
  np.testing.assert_allclose(product[0],
      (mass[0].reshape(2, 2) + np.diag(diagonal[0])) @ rhs[0, 0])
  indefinite, indefinite_status = solve_component_blocks_cpu(
      np.asarray([[1., 0., 0., 1.]]), layout, rhs,
      diagonal_add=np.asarray([[.2, -2.]]), allow_indefinite=True)
  np.testing.assert_array_equal(indefinite_status, [[0]])
  np.testing.assert_allclose(indefinite[0, 0], [1. / 1.2, -2.])


def test_component_euler_diagonal_matches_pinned_mj_euler_qh_for_signed_damping():
  import mujoco

  from mujoco_metal.stepping import _euler_damping_dofs

  xml = '''<mujoco><option timestep=".1" gravity="0 0 0"
      integrator="Euler"/><worldbody>
    <body><joint name="positive" type="slide" damping="2"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    <body><joint name="negative" type="slide" damping="-20"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  data.qvel[:] = [.7, -.4]
  mujoco.mj_forward(model, data)
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  mujoco.mj_step(model, data)
  diag_slot = model.M_rowadr + model.M_rownnz - 1
  pinned_diagonal = np.asarray(data.qH)[diag_slot] - np.diag(mass)
  oracle = build_euler_diagonal_cpu(
      np.asarray(model.dof_damping)[None, :], model.opt.timestep,
      _euler_damping_dofs(model), np.ones((1, model.nv), dtype=bool))
  np.testing.assert_allclose(oracle[0], pinned_diagonal, atol=1e-12)
  assert pinned_diagonal[1] < 0
  assert np.all(np.isfinite(data.qvel))
  # A negative-only world does not enter mj_EulerSkip's implicit branch.
  no_trigger = build_euler_diagonal_cpu(
      np.asarray([[-20.]]), .1, np.asarray([False]),
      np.asarray([[True]]))
  np.testing.assert_array_equal(no_trigger, [[0.]])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_component_status_merge_is_device_side_and_preserves_existing_failure():
  import mujoco
  import torch
  from mujoco_metal.component_solve import MetalComponentMassSolver

  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint type="slide"/><geom type="sphere" size=".1"/></body>
    <body><joint type="slide"/><geom type="sphere" size=".1"/></body>
    <body><joint type="slide"/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
  solver = MetalComponentMassSolver(
      model, _layout(), batch_size=2, rhs_capacity=1)
  # Component zero fails in world zero; component one fails in world one.
  mass = torch.tensor([[-1., 0., 0., 1., 1.],
                       [1., 0., 0., 1., -1.]],
                      dtype=torch.float32, device="mps")
  rhs = torch.ones((2, 1, 3), dtype=torch.float32, device="mps")
  _, component_status = solver.run_device(mass, rhs)
  world_status = torch.tensor([5, 0], dtype=torch.int32, device="mps")
  returned = solver.merge_world_status(component_status, world_status)
  assert returned.data_ptr() == world_status.data_ptr()
  np.testing.assert_array_equal(component_status.cpu().numpy(),
                                [[1, 0], [0, 1]])
  np.testing.assert_array_equal(world_status.cpu().numpy(), [5, 1])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_component_solve_native_accepts_tiny_positive_mass_pivot():
  import torch
  from types import SimpleNamespace
  from mujoco_metal.component_solve import MetalComponentMassSolver

  layout = {
      "nv": 2, "ncomponent": 1, "nnz": 4,
      "component_dof_offsets": np.asarray([0, 2], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 1], dtype=np.int32),
      "component_mass_offsets": np.asarray([0], dtype=np.int32),
  }
  solver = MetalComponentMassSolver(
      SimpleNamespace(nv=2, nbody=3), layout, batch_size=1, rhs_capacity=1)
  mass = torch.tensor([[1., 0., 0., 1e-12]], dtype=torch.float32, device="mps")
  rhs = torch.ones((1, 1, 2), dtype=torch.float32, device="mps")
  solution, status = solver.run_device(mass, rhs)
  np.testing.assert_array_equal(status.cpu().numpy(), [[0]])
  np.testing.assert_allclose(
      solution.cpu().numpy(), [[[1., 1e12]]], rtol=2e-5, atol=1e3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_component_ldlt_native_descending_pivot_failure_isolation_and_recovery():
  import torch
  from types import SimpleNamespace
  from mujoco_metal.component_solve import MetalComponentMassSolver

  layout = {
      "nv": 4, "ncomponent": 3, "nnz": 6,
      "component_dof_offsets": np.asarray([0, 2, 3, 4], dtype=np.int32),
      "component_dof_ids": np.asarray([1, 0, 2, 3], dtype=np.int32),
      "component_mass_offsets": np.asarray([0, 4, 5], dtype=np.int32),
  }
  solver = MetalComponentMassSolver(
      SimpleNamespace(nv=4, nbody=5), layout, batch_size=2, rhs_capacity=2)
  rhs = torch.tensor([
      [[2., 1., 4., 8.], [0., 1., 6., 10.]],
      [[1., 2., 3., 4.], [5., 6., 7., 8.]],
  ], dtype=torch.float32, device="mps")
  mass = torch.tensor([
      # Component-order block 0 is [[0,1],[1,1]], a valid pinned reverse-LDL
      # case despite its zero leading pivot. The other two are scalar blocks.
      [0., 1., 1., 1., 2., 4.],
      # This block is nonsymmetric and must fail only component 0.
      [1., 0.1, 0.2, 1., 3., 5.],
  ], dtype=torch.float32, device="mps")
  solution, status = solver.run_device(mass, rhs, allow_indefinite=True)
  np.testing.assert_array_equal(status.cpu().numpy(),
                                [[0, 0, 0], [1, 0, 0]])
  rhs_np = rhs.cpu().numpy()[0]
  expected0 = np.linalg.solve(
      np.asarray([[0., 1.], [1., 1.]]), rhs_np[:, [1, 0]].T).T
  # Public DOF order [0,1] is reverse of component order [1,0].
  np.testing.assert_allclose(solution.cpu().numpy()[0][:, [1, 0]],
                             expected0, rtol=2e-5, atol=2e-5)
  np.testing.assert_allclose(solution.cpu().numpy()[0, :, 2],
                             rhs.cpu().numpy()[0, :, 2] / 2., atol=1e-6)
  np.testing.assert_allclose(solution.cpu().numpy()[0, :, 3],
                             rhs.cpu().numpy()[0, :, 3] / 4., atol=1e-6)
  np.testing.assert_array_equal(solution.cpu().numpy()[1, :, :2], 0.0)
  np.testing.assert_allclose(solution.cpu().numpy()[1, :, 2],
                             rhs.cpu().numpy()[1, :, 2] / 3., atol=1e-6)

  # Inactive NaNs are excluded from the active principal block. Here global
  # DOF 0 is inactive, leaving the one-by-one principal block on DOF 1.
  principal_mass = torch.tensor(
      [[1., float("nan"), float("nan"), 0., 2., 4.]],
      dtype=torch.float32, device="mps")
  principal_rhs = torch.tensor(
      [[[float("nan"), 1., 4., 8.], [float("nan"), 2., 6., 10.]]],
      dtype=torch.float32, device="mps")
  ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.int32, device="mps")
  counts = torch.tensor([[0, 0, 3]], dtype=torch.int32, device="mps")
  # This principal-block case intentionally exercises a separate one-world
  # active mask. Keep its solver batch ABI consistent with the fixture.
  principal_solver = MetalComponentMassSolver(
      SimpleNamespace(nv=4, nbody=5), layout, batch_size=1, rhs_capacity=2)
  principal, principal_status = principal_solver.run_device(
      principal_mass, principal_rhs, dof_ids=ids, counts=counts,
      allow_indefinite=True)
  np.testing.assert_array_equal(principal_status.cpu().numpy(), [[0, 0, 0]])
  np.testing.assert_array_equal(principal.cpu().numpy()[0, :, 0], 0.0)
  np.testing.assert_allclose(principal.cpu().numpy()[0, :, 1], [1., 2.])

  # Active nonfinite RHS and active mass+armature overflow each clear every
  # RHS coordinate in their failed component, while siblings still solve.
  bad_rhs = rhs.clone()
  bad_rhs[0, 1, 1] = float("inf")
  good_mass = torch.tensor(
      [[1., 0., 0., 1., 2., 4.], [1., 0., 0., 1., 3., 5.]],
      dtype=torch.float32, device="mps")
  output, bad_rhs_status = solver.run_device(
      good_mass, bad_rhs, allow_indefinite=True)
  np.testing.assert_array_equal(bad_rhs_status.cpu().numpy(),
                                [[1, 0, 0], [0, 0, 0]])
  np.testing.assert_array_equal(output.cpu().numpy()[0, :, :2], 0.0)

  bad_mass = good_mass.clone()
  bad_mass[0, 0] = float("nan")
  bad_mass_output, bad_mass_status = solver.run_device(
      bad_mass, rhs, allow_indefinite=True)
  np.testing.assert_array_equal(bad_mass_status.cpu().numpy(),
                                [[1, 0, 0], [0, 0, 0]])
  np.testing.assert_array_equal(bad_mass_output.cpu().numpy()[0, :, :2], 0.0)

  armature = torch.zeros_like(good_mass)
  armature[0, 0] = torch.finfo(torch.float32).max
  overflow_mass = good_mass.clone()
  overflow_mass[0, 0] = torch.finfo(torch.float32).max
  overflowing, overflow_status = solver.run_device(
      overflow_mass, rhs, tendon_armature_blocks=armature,
      allow_indefinite=True)
  np.testing.assert_array_equal(overflow_status.cpu().numpy(),
                                [[1, 0, 0], [0, 0, 0]])
  np.testing.assert_array_equal(overflowing.cpu().numpy()[0, :, :2], 0.0)
  np.testing.assert_allclose(overflowing.cpu().numpy()[0, :, 2],
                             rhs.cpu().numpy()[0, :, 2] / 2., atol=1e-6)

  # A finite but tiny pivot with a large finite RHS overflows during solve.
  # The whole component output is cleared, not just the coordinate that failed.
  tiny_mass = torch.tensor(
      [[1e-38, 0., 0., 1., 2., 4.], [1., 0., 0., 1., 3., 5.]],
      dtype=torch.float32, device="mps")
  huge_rhs = rhs.clone()
  huge_rhs[0, :, 0] = torch.finfo(torch.float32).max
  overflow_solve, overflow_solve_status = solver.run_device(
      tiny_mass, huge_rhs, allow_indefinite=True)
  np.testing.assert_array_equal(overflow_solve_status.cpu().numpy(),
                                [[1, 0, 0], [0, 0, 0]])
  np.testing.assert_array_equal(overflow_solve.cpu().numpy()[0, :, :2], 0.0)

  # Reusing the workspace after failures must not leak status or outputs.
  recovered, recovery_status = solver.run_device(
      good_mass, rhs, allow_indefinite=True)
  np.testing.assert_array_equal(recovery_status.cpu().numpy(), 0)
  assert np.all(np.isfinite(recovered.cpu().numpy()))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_component_euler_diagonal_native_masks_trigger_and_awake_lists():
  import mujoco
  import torch

  from mujoco_metal.component_solve import MetalComponentMassSolver

  mjmodel = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <worldbody>
      <body><joint type="slide" damping=".5"/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body><joint type="slide" damping="-.25"/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>''')
  layout = {
      "nv": 2, "ncomponent": 1, "nnz": 4,
      "component_dof_offsets": np.asarray([0, 2], dtype=np.int32),
      "component_dof_ids": np.asarray([0, 1], dtype=np.int32),
      "component_mass_offsets": np.asarray([0], dtype=np.int32),
  }
  solver = MetalComponentMassSolver(
      mjmodel, layout, batch_size=2, rhs_capacity=1)
  dof_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device="mps")
  counts = torch.tensor([[3, 3, 2], [2, 2, 1]], dtype=torch.int32, device="mps")
  q_deriv = torch.tensor([[2., -20.], [2., -20.]],
                         dtype=torch.float32, device="mps")
  diagonal = solver.build_euler_diagonal_device(
      q_deriv, .1, dof_ids=dof_ids, counts=counts)
  np.testing.assert_allclose(
      diagonal.cpu().numpy(), [[.2, -2.], [0., 0.]], atol=1e-8)

  mass = torch.tensor([[1., .1, .1, 1.], [1., .1, .1, 1.]],
                      dtype=torch.float32, device="mps")
  rhs = torch.ones((2, 1, 2), dtype=torch.float32, device="mps")
  solution, status = solver.run_device(
      mass, rhs, dof_ids=dof_ids, counts=counts, diagonal_add=diagonal,
      allow_indefinite=True)
  np.testing.assert_array_equal(status.cpu().numpy(), [[0], [0]])
  dense0 = np.asarray([[1.2, .1], [.1, -1.]])
  np.testing.assert_allclose(
      solution.cpu().numpy()[0, 0], np.linalg.solve(dense0, [1., 1.]),
      rtol=2e-5, atol=2e-5)
  # Only global DOF 1 is awake in world one. The active-principal solve keeps
  # its reciprocal mass and leaves the inactive coordinate exactly zero.
  np.testing.assert_array_equal(solution.cpu().numpy()[1, 0], [0., 1.])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_component_retained_factors_match_pinned_mass_in_distinct_worlds():
  """Qualify the primitive independently of the native mass producer."""
  import mujoco
  import torch
  from mujoco_metal.component_solve import MetalComponentMassSolver
  from mujoco_metal.mass_layout import compile_tree_mass_layout

  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body pos="-1 0 1"><freejoint/>
      <geom type="box" size=".1 .2 .3" mass="2"/>
      <body pos="0 0 .4"><joint type="hinge" axis="0 1 0" armature=".03"/>
        <geom type="sphere" pos="0 0 .2" size=".1" mass=".5"/></body></body>
    <body pos="1 0 1"><freejoint/>
      <geom type="sphere" size=".2" mass="1"/></body>
    </worldbody></mujoco>''')
  layout = compile_tree_mass_layout(model.body_treeid, model.dof_bodyid)
  blocks = np.zeros((2, layout["nnz"]), dtype=np.float32)
  dense_worlds = []
  for world, angle in enumerate((.1, .7)):
    data = mujoco.MjData(model)
    data.qpos[7] = angle
    mujoco.mj_forward(model, data)
    dense = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, dense)
    dense_worlds.append(dense)
    for component in range(layout["ncomponent"]):
      begin, end = layout["component_dof_offsets"][component:component + 2]
      ids = layout["component_dof_ids"][begin:end]
      offset = int(layout["component_mass_offsets"][component])
      width = int(end - begin)
      blocks[world, offset:offset + width * width] = dense[np.ix_(ids, ids)].ravel()
  assert not np.allclose(blocks[0], blocks[1])
  rng = np.random.default_rng(1729)
  rhs_np = rng.normal(size=(2, 3, model.nv)).astype(np.float32)
  rhs = torch.as_tensor(rhs_np, device="mps")
  mass = torch.as_tensor(blocks, device="mps")
  solver = MetalComponentMassSolver(model, layout, batch_size=2, rhs_capacity=3)
  fused, status = solver.run_device(mass, rhs)
  expected = np.stack([np.linalg.solve(dense, right.T).T
                       for dense, right in zip(dense_worlds, rhs_np)])
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  np.testing.assert_allclose(fused.cpu().numpy(), expected, atol=1e-4, rtol=1e-4)
  factor_status = solver.factorize_device(mass)
  np.testing.assert_array_equal(factor_status.cpu().numpy(), 0)
  for scale in (1., -.75):
    retained, retained_status = solver.solve_factored_device(rhs * scale)
    np.testing.assert_array_equal(retained_status.cpu().numpy(), 0)
    np.testing.assert_allclose(retained.cpu().numpy(), expected * scale,
                               atol=1e-4, rtol=1e-4)
  with pytest.raises(ValueError, match="rhs must be contiguous"):
    solver.solve_factored_device(rhs[:, :, :1])
  # A rejected call leaves the retained factors available.
  vector = rhs[:, 0, :].contiguous()
  single, single_status = solver.solve_factored_vector_device(vector)
  np.testing.assert_array_equal(single_status.cpu().numpy(), 0)
  np.testing.assert_allclose(single.cpu().numpy(), expected[:, 0, :],
                             atol=1e-4, rtol=1e-4)
  product = solver.run_mass_matvec_device(mass, vector)
  np.testing.assert_allclose(product.cpu().numpy(),
      np.stack([dense @ right for dense, right in zip(dense_worlds, rhs_np[:, 0])]),
      atol=3e-5, rtol=3e-5)
