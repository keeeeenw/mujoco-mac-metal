# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracle and opt-in native gates for exact sparse component solves."""

import os
from pathlib import Path

import numpy as np
import pytest

from mujoco_metal.component_solve import (
    build_euler_diagonal_cpu,
    component_solver_workspace_sizes,
    _component_layout_array,
    component_mass_matvec_cpu,
    component_mass_residual_pair_cpu,
    component_mass_residual_reference,
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


def test_component_mass_residual_reference_covers_cancellation_and_rhs_low():
  # A dense SPD matrix with mixed scales reproduces the component solve's
  # residual-refinement operation: the f32 RHS is nearly M @ x, so sequential
  # fma residual accumulation loses more than the residual itself.
  rng = np.random.default_rng(448)
  seed = rng.normal(size=(8, 8)).astype(np.float32)
  matrix = (seed.T @ seed + np.eye(8, dtype=np.float32) * .1).astype(np.float32)
  solution = (rng.normal(size=8) * 10).astype(np.float32)
  rhs = (matrix @ solution).astype(np.float32)
  rhs_low = np.zeros(8, dtype=np.float32)
  rhs_low[3] = np.float32(2.0e-7)
  exact = component_mass_residual_reference(
      matrix, solution, rhs, rhs_low=rhs_low)
  paired = component_mass_residual_pair_cpu(
      matrix, solution, rhs, rhs_low=rhs_low)

  # Emulate the old shader sequence, where each fma rounds the remaining
  # residual back to one float after every matrix entry.
  sequential = rhs.copy()
  for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
      sequential[row] = np.float32(
          float(sequential[row])
          - float(matrix[row, col]) * float(solution[col]))
    sequential[row] = np.float32(sequential[row] + rhs_low[row])
  exact64 = (rhs.astype(np.float64) + rhs_low.astype(np.float64)
             - matrix.astype(np.float64) @ solution.astype(np.float64))
  np.testing.assert_allclose(exact.astype(np.float64), exact64,
                             rtol=0, atol=1e-12)
  np.testing.assert_allclose(paired.astype(np.float64), exact.astype(np.float64),
                             rtol=0, atol=1e-12)
  assert np.max(np.abs(sequential.astype(np.float64) - exact64)) > 1e-6
  assert np.max(np.abs(exact.astype(np.float64) - exact64)) < 3e-7

  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "component_solve.metal").read_text()
  assert "struct ComponentFloatPair" in shader
  assert "component_pair_product(a.hi,output[out_row+dof_j])" in shader
  assert "ComponentFloatPair residual={rhs[rhs_row+dof_i]" in shader


def test_simulation_component_first_rhs_is_owned_contiguous_for_multiple_worlds():
  import types
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  sim = object.__new__(MetalSimulation)
  sim.batch_size = 2
  sim._mjmodel = types.SimpleNamespace(nv=3)
  sim._state = types.SimpleNamespace(_torch=torch)
  sim._component_solution_vector = torch.empty((2, 3), dtype=torch.float32)
  sim._component_solution_low_vector = torch.empty((2, 3), dtype=torch.float32)
  solver_output = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
  low_output = solver_output + 100
  sim._component_solver = types.SimpleNamespace(
      _workspace={"output_low": low_output})
  assert not solver_output[:, 0, :].is_contiguous()
  published = sim._copy_component_first_rhs(solver_output)
  assert published.is_contiguous()
  torch.testing.assert_close(published, solver_output[:, 0, :])
  expected = published.clone()
  expected_low = low_output[:, 0, :].clone()
  solver_output.zero_()
  low_output.zero_()
  torch.testing.assert_close(published, expected)
  torch.testing.assert_close(sim._component_solution_low_vector, expected_low)


def test_sensor_acceleration_low_cache_tracks_tensor_generation_and_clears():
  import types
  torch = pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  sim = object.__new__(MetalSimulation)
  sim._state = types.SimpleNamespace(_torch=torch, generation=4)
  high = torch.ones((2, 3), dtype=torch.float32)
  low = torch.full_like(high, 1e-6)
  sim._cache_sensor_acceleration(high, low)
  assert sim._qacc_low_for_sensor(high) is low

  # The no-constraint/early-return path republishes its own low plane, or
  # clears it when no paired producer was used; it cannot retain an older
  # component result for an unrelated acceleration tensor.
  next_high = torch.full_like(high, 2.0)
  sim._cache_sensor_acceleration(next_high)
  assert sim._qacc_low_for_sensor(next_high) is None
  sim._state.generation += 1
  assert sim._qacc_low_for_sensor(next_high) is None


def test_compiled_blocks_solve_cpu_match_pinned_sparse_tendon_armature():
  import mujoco
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import compile_tree_mass_layout

  xml = """<mujoco><option gravity="0 0 0"/><worldbody>
    <body pos="-1 0 0"><joint name="a" type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
    <body><joint name="b" type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
    <body pos="1 0 0"><joint name="c" type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody><tendon><fixed name="t" armature="1" limited="false">
      <joint joint="a" coef="1"/><joint joint="b" coef="-2"/>
      <joint joint="c" coef=".5"/></fixed></tendon></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = load_model(model)
  layout = compile_tree_mass_layout(
      descriptor.body_treeid, descriptor.dof_bodyid,
      tendon_treeid=descriptor.tendon_treeid,
      tendon_treenum=descriptor.tendon_treenum,
      tendon_armature=descriptor.tendon_armature,
      tendon_j_rowadr=descriptor.tendon_j_rowadr,
      tendon_j_rownnz=descriptor.tendon_j_rownnz,
      tendon_j_colind=descriptor.tendon_j_colind,
      mass_rowadr=descriptor.mass_rowadr,
      mass_rownnz=descriptor.mass_rownnz,
      mass_colind=descriptor.mass_colind)
  base_model = mujoco.MjModel.from_xml_string(
      xml.replace('armature="1"', 'armature="0"'))
  data, base_data = mujoco.MjData(model), mujoco.MjData(base_model)
  mujoco.mj_forward(model, data)
  mujoco.mj_forward(base_model, base_data)
  mass, base_mass = np.zeros((model.nv, model.nv)), np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  mujoco.mj_fullM(base_model, base_data, base_mass)
  armature = mass - base_mass
  mass_blocks = np.zeros((1, layout["nnz"]))
  armature_blocks = np.zeros_like(mass_blocks)
  for component in range(layout["ncomponent"]):
    begin = int(layout["component_dof_offsets"][component])
    end = int(layout["component_dof_offsets"][component + 1])
    ids = layout["component_dof_ids"][begin:end]
    width = end - begin
    offset = int(layout["component_mass_offsets"][component])
    mass_blocks[0, offset:offset + width * width] = base_mass[np.ix_(ids, ids)].reshape(-1)
    armature_blocks[0, offset:offset + width * width] = armature[np.ix_(ids, ids)].reshape(-1)
  rhs = np.asarray([[[1.0, -2.0, .25], [-.5, .75, 3.0]]])
  solution, status = solve_component_blocks_cpu(
      mass_blocks, layout, rhs, tendon_armature_blocks=armature_blocks)
  np.testing.assert_array_equal(status, 0)
  np.testing.assert_allclose(solution[0],
      np.linalg.solve(mass, rhs[0].T).T, rtol=1e-12, atol=1e-12)
  # Pinned compressed-M sparsity drops this tendon’s cross-tree outer terms.
  np.testing.assert_array_equal(armature, np.diag(np.diag(armature)))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_component_solve_native_matches_dense_multi_rhs_and_active_principal():
  import mujoco
  import torch

  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics, compile_tree_mass_layout
  from mujoco_metal.component_solve import MetalComponentMassSolver

  xml = """<mujoco><worldbody>
    <body pos="-1 0 1"><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/>
      <body pos="0 0 .4"><joint type="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".1" mass=".5"/></body></body>
    <body pos="1 0 1"><freejoint/><geom type="sphere" size=".2" mass="1"/></body>
  </worldbody></mujoco>"""
  mjmodel = mujoco.MjModel.from_xml_string(xml)
  descriptor = load_model(mjmodel)
  layout = compile_tree_mass_layout(
      descriptor.body_treeid, descriptor.dof_bodyid,
      tendon_treeid=descriptor.tendon_treeid,
      tendon_treenum=descriptor.tendon_treenum,
      tendon_armature=descriptor.tendon_armature,
      tendon_j_rowadr=descriptor.tendon_j_rowadr,
      tendon_j_rownnz=descriptor.tendon_j_rownnz,
      tendon_j_colind=descriptor.tendon_j_colind,
      mass_rowadr=descriptor.mass_rowadr,
      mass_rownnz=descriptor.mass_rownnz,
      mass_colind=descriptor.mass_colind)
  smooth = MetalSmoothDynamics(descriptor, batch_size=2, mass_storage="block_sparse")
  solver = MetalComponentMassSolver(descriptor, layout, batch_size=2,
                                    rhs_capacity=3)
  qpos = torch.as_tensor(np.tile(mjmodel.qpos0.astype(np.float32), (2, 1)),
                         device="mps")
  qvel = torch.zeros((2, mjmodel.nv), dtype=torch.float32, device="mps")
  dynamics = smooth.run_device(qpos, qvel)
  rng = np.random.default_rng(1729)
  rhs_np = rng.normal(size=(2, 3, mjmodel.nv)).astype(np.float32)
  rhs = torch.as_tensor(rhs_np, device="mps")
  solution, status = solver.run_device(dynamics["mass_blocks"], rhs)
  solution_np = solution.cpu().numpy()
  np.testing.assert_array_equal(status.cpu().numpy(), 0)
  solution_low_np = solver._workspace["output_low"].cpu().numpy()
  mass_blocks_np = dynamics["mass_blocks"].cpu().numpy()
  for world in range(2):
    device_mass = np.zeros((mjmodel.nv, mjmodel.nv), dtype=np.float32)
    for component in range(layout["ncomponent"]):
      begin = int(layout["component_dof_offsets"][component])
      end = int(layout["component_dof_offsets"][component + 1])
      ids = np.asarray(layout["component_dof_ids"][begin:end], dtype=np.int64)
      width = end - begin
      offset = int(layout["component_mass_offsets"][component])
      device_mass[np.ix_(ids, ids)] = mass_blocks_np[
          world, offset:offset + width * width].reshape(width, width)
    # The solver's second output is the preconditioned residual correction,
    # not a second RHS. Apply it to the high solution before certifying the
    # original equation in float64.
    pair_x = solution_np[world].astype(np.float64) + solution_low_np[world].astype(
        np.float64)
    pair_norm = np.linalg.norm(
        rhs_np[world].astype(np.float64)
        - pair_x @ device_mass.astype(np.float64).T, axis=1)
    high_norm = np.linalg.norm(
        rhs_np[world].astype(np.float64)
        - solution_np[world].astype(np.float64)
        @ device_mass.astype(np.float64).T, axis=1)
    assert np.all(pair_norm <= high_norm + 1e-7)
    assert np.any(pair_norm < high_norm)
  factor_status = solver.factorize_device(dynamics["mass_blocks"])
  np.testing.assert_array_equal(factor_status.cpu().numpy(), 0)
  reused_solution, reused_status = solver.solve_factored_device(rhs)
  np.testing.assert_array_equal(reused_status.cpu().numpy(), 0)
  np.testing.assert_allclose(reused_solution.cpu().numpy(), solution_np,
                             rtol=1e-5, atol=1e-5)
  single_rhs = rhs[:, 0].contiguous()
  single_solution, single_status = solver.solve_factored_vector_device(single_rhs)
  np.testing.assert_array_equal(single_status.cpu().numpy(), 0)
  np.testing.assert_allclose(single_solution.cpu().numpy(), solution_np[:, 0],
                             rtol=1e-5, atol=1e-5)
  with pytest.raises(ValueError, match="rhs must be contiguous"):
    solver.solve_factored_device(rhs[:, :, :1])
  # Rejected calls do not consume or replace a valid factor snapshot.
  reused_again, reused_status = solver.solve_factored_device(rhs)
  np.testing.assert_array_equal(reused_status.cpu().numpy(), 0)
  np.testing.assert_allclose(reused_again.cpu().numpy(), solution_np,
                             rtol=1e-5, atol=1e-5)
  for world in range(2):
    data = mujoco.MjData(mjmodel)
    data.qpos[:] = mjmodel.qpos0
    mujoco.mj_forward(mjmodel, data)
    dense = np.zeros((mjmodel.nv, mjmodel.nv))
    mujoco.mj_fullM(mjmodel, data, dense)
    np.testing.assert_allclose(
      solution_np[world], np.linalg.solve(dense, rhs_np[world].T).T,
      rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_component_solve_native_includes_nonzero_pinned_tendon_armature():
  import mujoco
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics, compile_tree_mass_layout
  from mujoco_metal.component_solve import MetalComponentMassSolver

  xml = """<mujoco><option gravity="0 0 0"/><worldbody>
    <body pos="-1 0 0"><joint name="a" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    <body><joint name="b" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    <body pos="1 0 0"><joint name="c" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody><tendon><fixed name="t" armature="1" limited="false">
      <joint joint="a" coef="1"/><joint joint="b" coef="-2"/>
      <joint joint="c" coef=".5"/></fixed></tendon></mujoco>"""
  mjmodel = mujoco.MjModel.from_xml_string(xml)
  descriptor = load_model(mjmodel)
  layout = compile_tree_mass_layout(
      descriptor.body_treeid, descriptor.dof_bodyid,
      tendon_treeid=descriptor.tendon_treeid,
      tendon_treenum=descriptor.tendon_treenum,
      tendon_armature=descriptor.tendon_armature,
      tendon_j_rowadr=descriptor.tendon_j_rowadr,
      tendon_j_rownnz=descriptor.tendon_j_rownnz,
      tendon_j_colind=descriptor.tendon_j_colind,
      mass_rowadr=descriptor.mass_rowadr,
      mass_rownnz=descriptor.mass_rownnz,
      mass_colind=descriptor.mass_colind)
  data = mujoco.MjData(mjmodel)
  mujoco.mj_forward(mjmodel, data)
  tendon_jacobian = np.zeros((1, mjmodel.ntendon, mjmodel.nv), dtype=np.float32)
  for tendon in range(mjmodel.ntendon):
    start = int(mjmodel.ten_J_rowadr[tendon])
    end = start + int(mjmodel.ten_J_rownnz[tendon])
    columns = np.asarray(mjmodel.ten_J_colind[start:end], dtype=np.int64)
    tendon_jacobian[0, tendon, columns] = data.ten_J[start:end]

  smooth = MetalSmoothDynamics(descriptor, batch_size=1,
                               mass_storage="block_sparse")
  solver = MetalComponentMassSolver(descriptor, layout, batch_size=1,
                                    rhs_capacity=2)
  qpos = torch.as_tensor(mjmodel.qpos0[None].astype(np.float32), device="mps")
  qvel = torch.zeros((1, mjmodel.nv), dtype=torch.float32, device="mps")
  dynamics = smooth.run_device(
      qpos, qvel,
      tendon_J=torch.as_tensor(tendon_jacobian, dtype=torch.float32,
                               device="mps"))
  armature_blocks = dynamics["tendon_armature_blocks"]
  assert bool(torch.any(torch.abs(armature_blocks) > 0))
  rhs_np = np.asarray([[[1., -2., .25], [-.5, .75, 3.]]], dtype=np.float32)
  solution, status = solver.run_device(
      dynamics["mass_blocks"], torch.as_tensor(rhs_np, device="mps"),
      tendon_armature_blocks=armature_blocks)
  np.testing.assert_array_equal(status.cpu().numpy(), [[0] * layout["ncomponent"]])
  dense = np.zeros((mjmodel.nv, mjmodel.nv), dtype=np.float64)
  mujoco.mj_fullM(mjmodel, data, dense)
  vector_np = np.asarray([[.5, -2., 3.]], dtype=np.float32)
  product = solver.run_mass_matvec_device(
      dynamics["mass_blocks"], torch.as_tensor(vector_np, device="mps"),
      tendon_armature_blocks=armature_blocks)
  np.testing.assert_allclose(
      product.cpu().numpy()[0], dense @ vector_np[0],
      rtol=3e-5, atol=3e-5)
  np.testing.assert_allclose(
      solution.cpu().numpy()[0], np.linalg.solve(dense, rhs_np[0].T).T,
      rtol=3e-4, atol=3e-4)
