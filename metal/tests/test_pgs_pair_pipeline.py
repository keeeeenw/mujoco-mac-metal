# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracle for the paired PGS row pipeline and its bounded scratch ABI."""

from pathlib import Path

import numpy as np

from mujoco_metal.capacity import solver_debug_layout


def _pair(value):
  value = np.float64(value)
  hi = np.float32(value)
  lo = np.float32(value - np.float64(hi))
  return np.float32(hi), np.float32(lo)


def _value(pair):
  return np.float64(pair[0]) + np.float64(pair[1])


def _pair_sum(values):
  return _pair(sum((np.float64(value) for value in values), np.float64(0)))


def test_pgs_rhs_delassus_multiplier_and_force_pairs_match_float64_oracle():
  # Large shared terms make each low component meaningful while remaining
  # representative of the small geometry/solver residual in the native rich
  # contact fixture.
  aref_hi = np.float32(4213.125)
  aref_low = np.float32(1.1205673e-5)
  J = np.asarray([0.125, -0.375, 0.5], dtype=np.float32)
  q0 = [_pair(33705.0), _pair(-11235.0), _pair(8426.25)]
  z = [_pair(0.00390625), _pair(-0.0026041667), _pair(0.001953125)]
  rhs = _pair(np.float64(aref_hi) + np.float64(aref_low)
              - sum(np.float64(J[k]) * _value(q0[k]) for k in range(3)))
  expected_rhs = np.float64(aref_hi) + np.float64(aref_low)
  expected_rhs -= sum(np.float64(J[k]) * _value(q0[k]) for k in range(3))
  assert abs(_value(rhs) - expected_rhs) <= 2e-7

  wij = _pair(sum(np.float64(J[k]) * _value(z[k]) for k in range(3)))
  expected_w = sum(np.float64(J[k]) * _value(z[k]) for k in range(3))
  assert abs(_value(wij) - expected_w) <= 2e-10

  old_lambda = _pair(0.125)
  diagonal = _pair(_value(wij) + 0.75)
  residual = _pair(-_value(rhs) + _value(wij) * _value(old_lambda))
  delta = _pair(-_value(residual) / _value(diagonal))
  next_lambda = _pair(_value(old_lambda) + _value(delta))
  expected_lambda = (np.float64(old_lambda[0]) + np.float64(old_lambda[1])
                     + (expected_rhs - expected_w * _value(old_lambda))
                     / (expected_w + 0.75))
  assert abs(_value(next_lambda) - expected_lambda) <= 2e-8

  projected_force = _pair_sum(
      np.float64(J[k]) * _value(next_lambda) for k in range(3))
  expected_force = sum(np.float64(J[k]) * expected_lambda for k in range(3))
  assert abs(_value(projected_force) - expected_force) <= 2e-9


def test_pgs_pair_rows_use_disjoint_accounted_solver_tail():
  nv, nr = 9, 37
  layout = solver_debug_layout(nv, nr, 0)
  expected_length = 8 * nv + 3 * nr
  assert layout["high_low_workspace_length"] == expected_length
  start = layout["high_low_workspace_offset"]
  qacc_smooth_low = (start + 7 * nv, nv)
  rhs_low = (qacc_smooth_low[0] + nv, nr)
  lambda_low = (rhs_low[0] + nr, nr)
  aref_low = (lambda_low[0] + nr, nr)
  assert qacc_smooth_low[0] + qacc_smooth_low[1] == rhs_low[0]
  assert rhs_low[0] + rhs_low[1] == lambda_low[0]
  assert lambda_low[0] + lambda_low[1] == aref_low[0]
  assert aref_low[0] + aref_low[1] == start + expected_length

  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "coupled_constraints.metal"
  source = shader.read_text()
  assert "{ar[row], aref_low[row]}" in source
  assert "device float* pgs_rhs_low = pgs_pair_tail + 8 * max(nv, 1);" in source
  assert "device float* pgs_lambda_low = pgs_rhs_low + max(nr, 1);" in source
  assert "pgs_delassus_pair(J, Z, mass, dims" in source
  assert "{lam[row], solver_type == 0 ? pgs_lambda_low[row] : 0.0f}" in source


def test_component_pgs_reconstruction_multiplies_both_words_of_lambda():
  # A component Z row is itself hi+low and the accepted multiplier is hi+low.
  # Omitting lambda_low loses a real cross term from the reconstructed qacc.
  z_hi, z_low = _pair(0.00390625), _pair(1.1e-8)
  lam_hi, lam_low = _pair(125.0), _pair(2.3e-5)
  reconstructed = _pair_sum((
      _value(z_hi) * np.float64(lam_hi[0]),
      _value(z_hi) * np.float64(lam_low[0]),
      _value(z_low) * np.float64(lam_hi[0]),
      _value(z_low) * np.float64(lam_low[0])))
  expected = (_value(z_hi) + _value(z_low)) * (_value(lam_hi) + _value(lam_low))
  assert abs(_value(reconstructed) - expected) < 1e-12
  without_lambda_low = _value(z_hi) * np.float64(lam_hi[0]) + _value(z_low) * np.float64(lam_hi[0])
  assert abs(expected - without_lambda_low) > 1e-9

  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "coupled_constraints.metal"
  source = shader.read_text()
  component_output = source[source.rfind("if (dims[20] == -2) {",
                                         0, source.rfind("// 11. Write joint forces")):]
  assert component_output
  assert component_output.count(
      "{lam[row], solver_type == 0 ? pgs_lambda_low[row] : 0.0f}") >= 2


def test_dense_pgs_pairs_smooth_force_or_cached_acceleration_inputs():
  module = Path(__file__).parents[1] / "mujoco_metal" / "coupled_constraints.py"
  host = module.read_text()
  assert "paired_dense_input = (d.dense_path" in host
  assert "pair[:b * nv].copy_(qfrc_smooth.reshape(-1))" in host
  assert "pair[b * nv:2 * b * nv].copy_(qfrc_smooth_low.reshape(-1))" in host
  assert "_paired_smooth_input_stage(solver_dims)" in host
  assert "self._coupled_constraints.descriptor.solver_type" in (
      (Path(__file__).parents[1] / "mujoco_metal" / "simulation.py").read_text())

  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "coupled_constraints.metal"
  source = shader.read_text()
  assert source.count("bool paired_qfrc_input = (dims[7] & 16) != 0;") == 2
  assert "primal_cholesky_solve_pair(L, nv, solver_awake_tree, dims," in source
  assert "? qfrc[batch * nv + qb + k] : 0.0f" in source


def test_non_pgs_warm_cost_never_reads_pgs_pair_tail():
  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "coupled_constraints.metal"
  source = shader.read_text()
  starts = []
  cursor = 0
  while True:
    start = source.find("// 7b. Warmstart cost check", cursor)
    if start < 0:
      break
    end = source.find("// 8. Projected Gauss-Seidel", start)
    if end < 0:
      end = source.find("// 8. Projected Gauss-Seidel solve", start)
    assert end > start
    starts.append(source[start:end])
    cursor = end
  assert len(starts) == 2
  for block in starts:
    paired_start = block.index("if (solver_type == 0 && warm_ok")
    ordinary_start = block.index("else if (warm_ok", paired_start)
    paired = block[paired_start:ordinary_start]
    ordinary = block[ordinary_start:]
    assert "pgs_rhs_low" in paired and "pgs_lambda_low" in paired
    assert "pgs_rhs_low" not in ordinary
    assert "pgs_lambda_low" not in ordinary
    assert "float wcost = 0.0f;" in ordinary
    assert "W[row * nr + col]" in ordinary


def test_no_slip_postpass_reconstructs_primal_acceleration_for_all_solvers():
  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "coupled_constraints.metal"
  source = shader.read_text()
  assert source.count("if (noslip_iters > 0 && total_nr > 0)") == 1
  assert source.count("if (noslip_iters > 0 && nr > 0)") == 1
  # Pinned no-slip follows CG/Newton too. Their retained primal iterate is
  # stale after the multiplier update, so restore qacc_smooth and dual-finish.
  assert source.count("if (primal_acceleration_output && noslip_iters > 0") == 2
  assert source.count("primal_acceleration_output = false;") >= 2
  assert source.count("solver_restore_smooth_acceleration(out_acc, qfrc") == 2
  # Component-factor storage has no dense L factor: use its compiled solve
  # scratch and the precomputed per-row M^-1 J^T blocks for reconstruction.
  assert source.count("if (dims[20] == -3) {") >= 4
  assert "solver_component_mass_solve(debug, dims, awake_tree, nv," in source
  assert "// The component-factor route has no dense L matrix." in source


def test_elliptic_pair_cost_and_projected_multiplier_words_stay_coherent():
  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "coupled_constraints.metal"
  source = shader.read_text()
  device_solver = source.split("inline int solve_pgs_device", 1)[1].split(
      "\nkernel void solve_coupled_constraints", 1)[0]
  assert "PrimalFloatPair A_pair[36], old_pair[6], res_pair_local[6];" in device_solver
  assert "change_pair = primal_pair_add(change_pair" in device_solver
  assert "lam_low[start + i] = old_low[i];" in device_solver
  assert "for (int i = 0; i < dim; ++i) lam_low[start + i] = 0.0f;" in device_solver
  # Later no-slip projection replaces cone coordinates in float, so it must
  # invalidate those coordinates' pre-projection low residual words too.
  no_slip = source.split("  float noslip_scale =", 1)[1].split("  // G2:", 1)[0]
  assert "pgs_lambda_low[row] = 0.0f;" in no_slip
  assert "pgs_lambda_low[start + 1 + i] = 0.0f;" in no_slip


def test_cached_dense_pgs_dispatch_binds_prepared_high_low_pair(monkeypatch):
  """The real cached entry must pass the pair backing, not a raw force view."""
  from types import SimpleNamespace

  import torch

  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  solver = object.__new__(MetalCoupledConstraints)
  batch, nv, nr = 1, 2, 0
  solver._torch = torch
  solver._has_sdf_contact_pairs = False
  solver._device = torch.device("cpu")
  solver.batch_size = batch
  solver.descriptor = SimpleNamespace(
      nv=nv, nq=nv, nr=nr, ncontacts_max=0, nbody=1, neq=0,
      nr_joint=0, ntrees=0, dense_path=True, refsafe=False,
      solver_type=0, cone_type=0, eq_type=np.zeros(0, np.int32),
      eq_objtype=np.zeros(0, np.int32))
  solver._position_context_valid = True
  solver._position_context_epoch = 9
  solver._position_context_eq_jdot_inputs_valid = True
  solver._debug_stride = 8
  solver._world_mask_offset = 22
  solver._debug_prefix = 0
  solver._solver_island_ntree = 0
  solver._solver_island_nbody = 0
  solver._solver_island_core_scratch = 0
  solver._solver_island_awake_offset = 3
  solver._primal_scratch_floats = nv
  solver._qacc_warmstart_offset = 0
  solver._aref_low_debug_offset = 0
  solver._jacobian_layout = SimpleNamespace(mode=1)
  solver._jacobian_pattern = None
  solver._line_search_iterations = 6
  solver._eq_active_default = torch.zeros((batch, 1), dtype=torch.int32)
  solver._empty_compaction_map = torch.empty(0, dtype=torch.int32)

  def f32(count):
    return torch.zeros(count, dtype=torch.float32)

  def i32(count):
    return torch.zeros(count, dtype=torch.int32)

  solver._workspace = {
      "component_smooth_acceleration_pair": f32(2 * batch * nv),
      "workspace_debug": f32(batch * solver._debug_stride),
      "workspace_J": f32(0), "position_cache_J": f32(0),
      "position_qvel": f32(batch * nv),
      "position_cache_qvel": f32(batch * nv),
      "position_cache_rows": f32(0), "position_cache_impedance": f32(0),
      "position_cache_aref_low": f32(0),
      "position_current_aref_low": f32(0),
      "position_context_zero": f32(0),
      "position_extra_aref": f32(0),
      "position_refresh_old_extra_aref": f32(0),
      "position_refresh_extra_aref": f32(0),
      "contact_row_data": f32(0), "position_cache_contact_data": f32(0),
      "contact_frame": f32(0), "position_cache_contact_frame": f32(0),
      "contact_jacobian": f32(0), "position_cache_contact_jacobian": f32(0),
      "out_force": f32(batch * nv), "out_acc": f32(2 * batch * nv),
      "out_status": i32(batch), "out_diagnostics": f32(2 * batch),
      "out_contact_force": f32(0),
  }
  solver._constants = {name: f32(0) for name in (
      "joint_qposadr", "qpos0", "joint_dofadr", "joint_limited",
      "joint_limit_params", "joint_sol_params", "dof_frictionloss",
      "dof_invweight0", "dof_sol_params", "eq_obj", "eq_data",
      "eq_sol_params", "contact_friction", "contact_condim")}
  solver._constants["solver_dims"] = i32(24)
  solver._constants["solver_dims"][21] = solver._debug_stride
  solver._constants["solver_dims"][20] = solver._line_search_iterations
  solver._constants["solver_params"] = f32(7)
  solver._constants["solver_params"][0] = 1e-3
  solver._constants["solver_params"][1] = 1.0
  solver._constants["solver_params"][4] = 1.0

  monkeypatch.setattr(solver, "refresh_position_references", lambda *a, **k: None)
  monkeypatch.setattr(solver, "_refresh_cached_aref_low", lambda *a, **k: None)
  monkeypatch.setattr(solver, "_diagnostic_views", lambda: (f32(2), f32(0)))
  captured = {}

  def solve(*args, **kwargs):
    captured["force_arg"] = args[1]
    captured["solver_dims"] = args[21].clone()

  solver._solve_kernel = solve
  qvel = torch.zeros((batch, nv), dtype=torch.float32)
  qfrc = torch.tensor([[100.0, -50.0]], dtype=torch.float32)
  qfrc_low = torch.tensor([[1.25e-4, -2.5e-4]], dtype=torch.float32)
  qpos = torch.zeros((batch, nv), dtype=torch.float32)
  context = {
      "_owner": solver, "_epoch": 9, "_eq_jdot_inputs_valid": True,
      "position_context": torch.zeros((batch, nr, 5), dtype=torch.float32),
      "surface_velocity": torch.zeros((batch, nr), dtype=torch.float32),
      "extra_aref": torch.zeros((batch, nr), dtype=torch.float32),
  }

  solver.run_velocity_device(context, None, torch.zeros((batch, nv, nv)),
                             qfrc, qpos, qvel, qfrc_smooth_low=qfrc_low)

  pair = solver._workspace["component_smooth_acceleration_pair"]
  assert captured["force_arg"].data_ptr() == pair.data_ptr()
  assert int(captured["solver_dims"][7]) == (2 | 16)
  assert int(solver._constants["solver_dims"][7]) == 0
  np.testing.assert_array_equal(pair[:batch * nv].numpy(), qfrc.numpy().reshape(-1))
  np.testing.assert_array_equal(pair[batch * nv:].numpy(), qfrc_low.numpy().reshape(-1))


def test_default_dense_pgs_dispatch_binds_prepared_high_low_pair(monkeypatch):
  """The non-cached run_device path binds high/low smooth-force planes."""
  from types import SimpleNamespace

  import torch

  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  solver = object.__new__(MetalCoupledConstraints)
  batch, nv, nr = 1, 2, 0
  solver._torch = torch
  solver._has_sdf_contact_pairs = False
  solver._device = torch.device("cpu")
  solver.batch_size = batch
  solver.descriptor = SimpleNamespace(
      nv=nv, nq=nv, njnt=0, neq=0, nc=0, npairs=0,
      ncontacts_max=0, nr_joint=0, nr=nr, nbody=1, nsite=0, ngeom=0,
      ntendon=0, ten_friction_rows=0, ten_limit_rows=0,
      n_eq_rows=0, n_flex_contact_rows=0, eq_type=np.zeros(0, np.int32),
      eq_objtype=np.zeros(0, np.int32), dense_path=True, refsafe=False,
      solver_type=0, cone_type=0)
  solver._candidate_generation_token = None
  solver._position_current_valid = False
  solver._assembly_generation = 0
  solver._position_current_eq_jdot_inputs_valid = False
  solver._debug_stride = 16
  solver._world_mask_offset = 22
  solver._debug_prefix = 0
  solver._solver_island_ntree = 0
  solver._solver_island_nbody = 0
  solver._solver_island_core_scratch = 0
  solver._solver_island_awake_offset = 3
  solver._primal_scratch_floats = nv
  solver._qacc_warmstart_offset = 0
  solver._aref_low_debug_offset = 0
  solver._jacobian_layout = SimpleNamespace(mode=1)
  solver._jacobian_pattern = None
  solver._line_search_iterations = 6
  solver._eq_active_default = torch.zeros((batch, 1), dtype=torch.int32)
  solver._empty_compaction_map = torch.empty(0, dtype=torch.int32)

  def f32(count):
    return torch.zeros(count, dtype=torch.float32)

  def i32(count):
    return torch.zeros(count, dtype=torch.int32)

  solver._workspace = {
      "component_smooth_acceleration_pair": f32(2 * batch * nv),
      "workspace_debug": f32(batch * solver._debug_stride),
      "workspace_J": f32(0), "contact_jacobian": f32(0),
      "contact_row_data": f32(0), "contact_frame": f32(0),
      "out_force": f32(batch * nv), "out_acc": f32(2 * batch * nv),
      "out_status": i32(batch), "out_diagnostics": f32(10 * batch),
      "out_contact_force": f32(0), "out_joint_force": f32(0),
      "position_current_qvel": f32(batch * nv),
      "position_current_cvel": f32(0), "position_current_cdof": f32(0),
      "position_current_cdof_dot": f32(0),
      "position_current_body_pos": f32(3),
      "position_current_body_quat": f32(4),
      "position_current_root_com": f32(3),
      "position_current_site_pos": f32(0),
      "position_current_site_quat": f32(0),
      "position_current_eq_active": i32(1),
      "position_assembly_context": f32(0),
      "position_surface_velocity": f32(0),
  }
  names = (
      "joint_qposadr", "qpos0", "joint_dofadr", "joint_limited",
      "joint_limit_params", "joint_sol_params", "dof_frictionloss",
      "dof_invweight0", "dof_sol_params", "eq_obj", "eq_data",
      "eq_sol_params", "eq_rowadr", "eq_rownum", "eq_type", "eq_objtype",
      "body_invweight0", "site_bodyid", "body_parentid", "body_jntadr",
      "body_jntnum", "jnt_type", "jnt_dofadr", "contact_friction",
      "contact_condim", "ten_length_map", "ten_moment_map", "ten_limited",
      "ten_range", "ten_margin", "ten_length0", "ten_invweight0",
      "ten_solref_lim", "ten_solimp_lim", "ten_frictionloss",
      "ten_solref_fri", "ten_solimp_fri")
  solver._constants = {name: f32(0) for name in names}
  solver._constants.update({
      "solver_dims": i32(24), "solver_params": f32(7),
  })
  solver._constants["solver_dims"][20] = solver._line_search_iterations
  solver._constants["solver_dims"][21] = solver._debug_stride
  solver._constants["solver_params"][0] = 1e-3
  solver._constants["solver_params"][1] = 1.0
  solver._constants["solver_params"][4] = 1.0

  monkeypatch.setattr(solver, "generate_candidates", lambda *a, **k: None)
  monkeypatch.setattr(solver, "_merge_flex_candidate_status", lambda: None)
  monkeypatch.setattr(solver, "_diagnostic_views", lambda: (
      solver._workspace["out_diagnostics"][:2].reshape(batch, 2),
      f32(0)))
  captured = {}

  def solve(*args, **kwargs):
    captured["force_arg"] = args[1]
    captured["solver_dims"] = args[21].clone()

  solver._solve_kernel = solve
  qfrc = torch.tensor([[100.0, -50.0]], dtype=torch.float32)
  qfrc_low = torch.tensor([[1.25e-4, -2.5e-4]], dtype=torch.float32)
  qpos = torch.zeros((batch, nv), dtype=torch.float32)
  qvel = torch.zeros((batch, nv), dtype=torch.float32)
  poses = {
      "body_pos": torch.zeros((batch, 1, 3), dtype=torch.float32),
      "body_quat": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
  }

  for name, width in (("geom_xmat", 9), ("geom_xmat_low", 9),
                      ("geom_xmat_tail", 9), ("geom_quat_low", 4),
                      ("geom_quat_tail", 4)):
    poses[name] = torch.zeros((batch, 0, width), dtype=torch.float32)

  solver.run_device(poses, torch.eye(nv)[None].to(torch.float32), qfrc,
                    qpos, qvel, qfrc_smooth_low=qfrc_low)

  pair = solver._workspace["component_smooth_acceleration_pair"]
  assert captured["force_arg"].data_ptr() == pair.data_ptr()
  assert int(captured["solver_dims"][7]) == 16
  assert int(solver._constants["solver_dims"][7]) == 0
  np.testing.assert_array_equal(pair[:batch * nv].numpy(), qfrc.numpy().reshape(-1))
  np.testing.assert_array_equal(pair[batch * nv:].numpy(), qfrc_low.numpy().reshape(-1))
