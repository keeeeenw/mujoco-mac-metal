# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Inverse row assembly must not require or invoke a mass solver."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


def _assembly_stub(torch):
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  cc = object.__new__(MetalCoupledConstraints)
  cc._torch = torch
  cc._device = torch.device("cpu")
  cc.batch_size = 1
  cc._world_mask_offset = 63
  cc._constants = {"solver_dims": torch.ones((64,), dtype=torch.int32)}
  cc.descriptor = SimpleNamespace(
      nq=2, nv=2, nbody=1, njnt=1, npairs=0, ngeom=0, nsite=0,
      neq=0, ntendon=0, eq_type=np.empty((0,), dtype=np.int32),
      eq_objtype=np.empty((0,), dtype=np.int32))
  return cc


def test_inverse_assembly_calls_assembly_only_without_mass_or_optimizer():
  torch = pytest.importorskip("torch")
  cc = _assembly_stub(torch)
  poses = {
      "body_pos": torch.zeros((1, 1, 3)),
      "body_quat": torch.tensor([[[1., 0., 0., 0.]]]),
      "joint_anchor": torch.zeros((1, 1, 3)),
      "joint_axis": torch.tensor([[[0., 0., 1.]]]),
  }
  qpos = torch.zeros((1, 2))
  qvel = torch.tensor([[.3, -.2]])
  marker = object()
  calls = []

  def run_device(*args, **kwargs):
    calls.append((args, kwargs))
    return marker

  cc.run_device = run_device
  assert cc.assemble_device(poses, qpos, qvel) is marker
  assert len(calls) == 1
  args, kwargs = calls[0]
  assert args[0] is poses and args[1] is None
  assert args[2] is qvel and args[3] is qpos and args[4] is qvel
  assert kwargs["assemble_only"] is True
  assert not any(name in kwargs for name in (
      "component_solver", "mass_blocks", "tendon_armature_blocks"))


def test_inverse_assembly_validates_inputs_before_touching_solver():
  torch = pytest.importorskip("torch")
  cc = _assembly_stub(torch)
  called = []
  cc.run_device = lambda *args, **kwargs: called.append(True)
  poses = {
      "body_pos": torch.zeros((1, 1, 3)),
      "body_quat": torch.tensor([[[1., 0., 0., 0.]]]),
      "joint_anchor": torch.zeros((1, 1, 3)),
      "joint_axis": torch.tensor([[[0., 0., 1.]]]),
  }
  with pytest.raises(ValueError, match="qvel must have shape"):
    cc.assemble_device(poses, torch.zeros((1, 2)), torch.zeros((1, 1)))
  assert called == []


def _run_device_cpu_standin(torch, kind, endpoint):
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  import mujoco

  cc = object.__new__(MetalCoupledConstraints)
  b, nv, nr = 1, 2, 1
  eq_type = int(mujoco.mjtEq.mjEQ_CONNECT if kind == "connect"
               else mujoco.mjtEq.mjEQ_WELD)
  objtype = 6 if endpoint == "site" else 1
  d = cc.descriptor = SimpleNamespace(
      nq=2, nv=nv, nr=nr, nr_joint=0, nbody=2, njnt=2, ngeom=0,
      nsite=2 if endpoint == "site" else 0, npairs=0, ncontacts_max=0,
      neq=1, n_eq_rows=1, ntendon=0, ten_friction_rows=0, ten_limit_rows=0,
      n_flex_contact_rows=0, eq_type=np.array([eq_type], np.int32),
      eq_objtype=np.array([objtype], np.int32), solver_type=int(mujoco.mjtSolver.mjSOL_PGS),
      dense_path=True)
  cc._torch = torch
  cc._device = torch.device("cpu")
  cc.batch_size = b
  cc._has_sdf_contact_pairs = False
  cc._line_search_iterations = 3
  cc._debug_stride = 40
  cc._aref_low_debug_offset = nr * nr + 7 * nr + 8 * max(nv, 1) + 2 * max(nr, 1)
  cc._debug_prefix = nr * nr + 7 * nr
  cc._primal_scratch_floats = 4
  cc._solver_island_awake_offset = 0
  cc._solver_island_ntree = 0
  cc._solver_island_nbody = 0
  cc._solver_island_core_scratch = 0
  cc._position_current_valid = True
  cc._assembly_generation = 9
  cc._position_current_eq_jdot_inputs_valid = False
  cc._component_operator_layout_device = None
  cc._component_operator_nnz = 0
  cc._empty_compaction_map = torch.full((1,), -1, dtype=torch.int32)
  cc._eq_active_default = torch.ones((b, 1), dtype=torch.int32)
  cc._constants = {name: torch.zeros((1,), dtype=torch.float32)
                   for name in (
                       "joint_qposadr", "qpos0", "joint_dofadr",
                       "joint_limited", "joint_limit_params", "joint_sol_params",
                       "dof_frictionloss", "dof_invweight0", "dof_sol_params",
                       "body_invweight0", "eq_obj", "eq_data", "eq_sol_params", "contact_friction",
                       "contact_condim", "solver_params")}
  cc._constants.update({
      "eq_rowadr": torch.tensor([0], dtype=torch.int32),
      "eq_rownum": torch.tensor([1], dtype=torch.int32),
      "eq_type": torch.tensor([eq_type], dtype=torch.int32),
      "eq_objtype": torch.tensor([objtype], dtype=torch.int32),
      "solver_dims": torch.zeros((32,), dtype=torch.int32),
      "row_velocity_scale_low": torch.zeros((1,), dtype=torch.float32),
      "site_bodyid": torch.zeros((2,), dtype=torch.int32),
      "body_parentid": torch.zeros((2,), dtype=torch.int32),
      "body_jntadr": torch.zeros((2,), dtype=torch.int32),
      "body_jntnum": torch.zeros((2,), dtype=torch.int32),
      "jnt_type": torch.zeros((2,), dtype=torch.int32),
      "jnt_dofadr": torch.zeros((2,), dtype=torch.int32),
  })
  cc._world_mask_offset = 31
  cc._constants["solver_dims"].fill_(1)
  cc._workspace = {
      "workspace_J": torch.zeros((b * nr * nv,), dtype=torch.float32),
      "position_cache_J": torch.full((b * nr * nv,), 7.0, dtype=torch.float32),
      "workspace_debug": torch.zeros((b * cc._debug_stride,), dtype=torch.float32),
      "position_current_qvel": torch.zeros((b * nv,), dtype=torch.float32),
      "position_current_cvel": torch.zeros((b * d.nbody * 6,), dtype=torch.float32),
      "position_current_cdof": torch.zeros((b * nv * 6,), dtype=torch.float32),
      "position_current_cdof_dot": torch.zeros((b * nv * 6,), dtype=torch.float32),
      "position_current_body_pos": torch.zeros((b * d.nbody * 3,), dtype=torch.float32),
      "position_current_body_quat": torch.zeros((b * d.nbody * 4,), dtype=torch.float32),
      "position_current_root_com": torch.zeros((b * d.nbody * 3,), dtype=torch.float32),
      "position_current_site_pos": torch.zeros((b * d.nsite * 3,), dtype=torch.float32),
      "position_current_site_quat": torch.zeros((b * d.nsite * 4,), dtype=torch.float32),
      "position_current_eq_active": torch.zeros((1,), dtype=torch.int32),
      "position_assembly_context": torch.zeros((b * nr * 5,), dtype=torch.float32),
      "position_surface_velocity": torch.zeros((b * nr,), dtype=torch.float32),
      "position_context_zero": torch.zeros((b * nr,), dtype=torch.float32),
      "contact_jacobian": torch.zeros((1,), dtype=torch.float32),
      "contact_row_data": torch.zeros((1,), dtype=torch.float32),
      "contact_frame": torch.zeros((1,), dtype=torch.float32),
      "out_force": torch.zeros((b * nv,), dtype=torch.float32),
      "out_acc": torch.zeros((b * nv,), dtype=torch.float32),
      "out_status": torch.zeros((b,), dtype=torch.int32),
      "out_diagnostics": torch.zeros((b * 10,), dtype=torch.float32),
      "out_contact_force": torch.zeros((1,), dtype=torch.float32),
  }
  cc._tendon_kernel = None
  cc._equality_kernel = None
  equality_calls = []

  def equality_kernel(*args, **kwargs):
    equality_calls.append(args)
    args[-3][:] = torch.tensor([1., -1.])
    dbg = args[-2]
    dbg[nr * nr] = 0.25
    dbg[nr * nr + nr] = -0.5
  cc._equality_kernel = equality_kernel
  cc.generate_candidates = lambda *args, **kwargs: None
  def refresh_jdot(qvel, cvel, cdof, cdof_dot, out, **kwargs):
    out.zero_()
    return out
  cc._refresh_equality_jdot = refresh_jdot
  dispatch = []
  def solver(*args, threads=None, group_size=None):
    dims = next(arg for arg in args if arg is cc._constants["solver_dims"])
    assert int(dims[20]) == -1
    dispatch.append((args, threads, group_size))
  cc._solve_kernel = solver

  poses = {
      "body_pos": torch.zeros((b, d.nbody, 3)),
      "body_quat": torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]]),
      "joint_anchor": torch.zeros((b, d.njnt, 3)),
      "joint_axis": torch.tensor([[[0., 0., 1.], [0., 1., 0.]]]),
      "root_com": torch.zeros((b, d.nbody, 3)),
  }
  # Even a geometry-free FK result retains the five typed, empty rotation
  # planes required by the public assembly ABI.
  for name, width in (("geom_xmat", 9), ("geom_xmat_low", 9),
                      ("geom_xmat_tail", 9), ("geom_quat_low", 4),
                      ("geom_quat_tail", 4)):
    poses[name] = torch.zeros((b, d.ngeom, width), dtype=torch.float32)
  if endpoint == "site":
    poses["site_pos"] = torch.zeros((b, d.nsite, 3))
    poses["site_quat"] = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
  qpos = torch.zeros((b, d.nq))
  qvel = torch.tensor([[.2, -.4]])
  dynamics = dict(cvel=torch.zeros((b, d.nbody, 6)),
                  cdof=torch.zeros((b, nv, 6)),
                  cdof_dot=torch.zeros((b, nv, 6)))
  result = cc.assemble_device(poses, qpos, qvel, **dynamics)
  assert len(equality_calls) == 1
  assert len(dispatch) == 1
  assert cc._position_current_valid and cc._assembly_generation == 10
  assert int(cc._constants["solver_dims"][20]) == 3
  assert result["J"].shape == (b, nr, nv)
  torch.testing.assert_close(result["J"], torch.tensor([[[1., -1.]]]))
  torch.testing.assert_close(result["R"], torch.tensor([[.25]]))
  torch.testing.assert_close(result["ar"], torch.tensor([[-.5]]))
  cache_J = cc._workspace["position_cache_J"]
  cache_snapshot = cache_J.clone()
  current_qvel = cc._workspace["position_current_qvel"].clone()
  with pytest.raises(ValueError, match="requires cvel, cdof, and cdof_dot"):
    cc.run_device(poses, None, qvel, qpos, qvel, cvel=dynamics["cvel"],
                  cdof=None, cdof_dot=dynamics["cdof_dot"], assemble_only=True)
  assert cc._assembly_generation == 10
  assert cc._workspace["position_cache_J"] is cache_J
  torch.testing.assert_close(cache_J, cache_snapshot, rtol=0, atol=0)
  torch.testing.assert_close(cc._workspace["position_current_qvel"], current_qvel,
                             rtol=0, atol=0)
  return cc


@pytest.mark.parametrize("kind", ["connect", "weld"])
@pytest.mark.parametrize("endpoint", ["body", "site"])
def test_real_run_device_inverse_assembly_uses_no_mass_solver(kind, endpoint):
  torch = pytest.importorskip("torch")
  _run_device_cpu_standin(torch, kind, endpoint)


def test_refresh_velocity_context_restores_captured_rows_without_a_solver():
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  cc = object.__new__(MetalCoupledConstraints)
  cc._torch = torch
  cc._device = torch.device("cpu")
  cc.batch_size = 1
  cc._world_mask_offset = 15
  cc._constants = {"solver_dims": torch.ones((16,), dtype=torch.int32)}
  cc.descriptor = SimpleNamespace(nq=1, nv=1, nr=1, neq=0, nbody=1,
                                  eq_type=np.empty((0,), np.int32))
  cc._position_context_valid = True
  cc._position_context_epoch = 4
  cc._position_context_eq_jdot_inputs_valid = False
  cc._debug_stride = 12
  cc._aref_low_debug_offset = 8
  cc._constants = {
      "row_velocity_scale_low": torch.zeros((1,)),
      "solver_dims": torch.zeros((32,), dtype=torch.int32),
  }
  cc._workspace = {
      "workspace_J": torch.full((1,), -1.0),
      "position_cache_J": torch.tensor([2.0]),
      "workspace_debug": torch.full((12,), -3.0),
      "position_cache_rows": torch.tensor([1., 2., 3., 4., 5., 6., 7.]),
      "position_cache_aref_low": torch.tensor([.125]),
      "position_current_aref_low": torch.tensor([0.]),
      "position_cache_qvel": torch.tensor([.1]),
      "position_qvel": torch.tensor([0.]),
      "position_refresh_old_extra_aref": torch.zeros((1,)),
      # Retained position-stage impedance owns this row independently of the
      # mutable velocity-stage row cache. Exercise its restoration too.
      "position_cache_impedance": torch.tensor([3.]),
      "position_cache_contact_data": torch.tensor([8.]),
      "contact_row_data": torch.tensor([0.]),
      "position_cache_contact_frame": torch.tensor([9.]),
      "contact_frame": torch.tensor([0.]),
      "position_cache_contact_jacobian": torch.tensor([10.]),
      "contact_jacobian": torch.tensor([0.]),
      "position_refresh_extra_aref": torch.zeros((1,)),
      "position_context_zero": torch.zeros((1,)),
  }
  position = torch.zeros((1, 1, 5))
  surface = torch.zeros((1, 1))
  extra = torch.tensor([[.25]])
  context = {"_owner": cc, "_epoch": 4, "position_context": position,
             "surface_velocity": surface, "extra_aref": extra}
  qpos, qvel = torch.tensor([[.2]]), torch.tensor([[.7]])
  seen = []

  def refresh(p, v, s, e, world_mask=None):
    seen.append((p, v, s, e, world_mask))
    return e

  cc.refresh_position_references = refresh
  low_calls = []
  cc._refresh_cached_aref_low = lambda *args, **kwargs: low_calls.append((args, kwargs))
  result = cc.refresh_velocity_context(context, qpos, qvel)
  assert result.data_ptr() == cc._workspace["position_refresh_extra_aref"].data_ptr()
  assert len(seen) == 1
  assert seen[0][0].shape == position.shape
  assert seen[0][0].data_ptr() == position.data_ptr()
  assert seen[0][1] is qvel and seen[0][2] is surface
  assert len(low_calls) == 1
  assert low_calls[0][0][0] is qvel and low_calls[0][0][1] is surface
  torch.testing.assert_close(low_calls[0][0][3], extra)
  assert low_calls[0][1] == {"world_mask": None}
  torch.testing.assert_close(cc._workspace["workspace_J"], torch.tensor([2.]))
  torch.testing.assert_close(cc._workspace["workspace_debug"][1:8],
                             torch.tensor([1., 2., 3., 4., 5., 6., 7.]))
  torch.testing.assert_close(cc._workspace["contact_row_data"], torch.tensor([8.]))
  torch.testing.assert_close(cc._workspace["contact_frame"], torch.tensor([9.]))
  torch.testing.assert_close(cc._workspace["contact_jacobian"], torch.tensor([10.]))


def test_cached_aref_low_refresh_uses_new_velocity_delta_and_retained_pos():
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  cc = object.__new__(MetalCoupledConstraints)
  cc._torch = torch
  cc._device = torch.device("cpu")
  cc.batch_size = 1
  cc.descriptor = SimpleNamespace(nv=2, nr=1)
  cc._debug_stride = 32
  cc._aref_low_debug_offset = 24
  debug = torch.zeros((32,), dtype=torch.float32)
  debug[1] = 1.0  # canonical row active
  debug[2] = .65  # refreshed high-word ar
  cc._workspace = {
      "workspace_debug": debug,
      "position_cache_J": torch.tensor([1., -2.]),
      "position_cache_qvel": torch.tensor([.125, -.249]),
      "position_context": torch.tensor([1., 1.5, .9, .2, 0.]),
      "position_cache_rows": torch.tensor([2., .75, 0., 0., -1., 1., 1.]),
      "position_cache_aref_low": torch.tensor([.125]),
      "position_current_aref_low": torch.zeros((1,)),
  }
  cc._constants = {
      "row_velocity_scale_low": torch.tensor([.0625]),
      "solver_dims": torch.zeros((32,), dtype=torch.int32),
  }

  def cpu_kernel(J, qvel, old_qvel, position_context, velocity_scale_low,
                 old_extra, new_extra, position_cache_rows,
                 position_cache_aref_low, workspace_debug, dims, *,
                 threads, group_size):
    assert threads == (1,) and group_size == (128,)
    velocity_delta = float(torch.dot(J.reshape(2),
                                     (qvel - old_qvel).reshape(2)))
    delta = -(float(position_context[1]) + float(velocity_scale_low[0])) * velocity_delta
    exact = (float(position_cache_rows[1]) + float(position_cache_aref_low[0])
             + delta + float(new_extra[0]) - float(old_extra[0]))
    tail_low = exact - float(workspace_debug[2])
    workspace_debug[cc._aref_low_debug_offset] = tail_low

  cc._refresh_aref_low_kernel = cpu_kernel
  qvel = torch.tensor([[.126, -.248]], dtype=torch.float32)
  surface = torch.zeros((1, 1), dtype=torch.float32)
  new_extra = torch.tensor([[.25]], dtype=torch.float32)
  old_extra = torch.tensor([[.2]], dtype=torch.float32)
  cc._refresh_cached_aref_low(qvel, surface, new_extra, old_extra)
  expected = (.75 + .125
              - (1.5 + .0625) * float(torch.dot(
                  torch.tensor([1., -2.]),
                  qvel.reshape(2) - torch.tensor([.125, -.249])))
              + .25 - .2)
  actual = float(debug[2] + cc._workspace["position_current_aref_low"][0])
  np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("batch", [2, 3])
def test_cached_aref_low_refresh_uses_static_row_scale_for_every_world(batch):
  """The velocity-scale low array is [nr], while residuals are [B,nr]."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  nv, nr, stride, low_offset = 2, 2, 64, 48
  cc = object.__new__(MetalCoupledConstraints)
  cc._torch = torch
  cc._device = torch.device("cpu")
  cc.batch_size = batch
  cc.descriptor = SimpleNamespace(nv=nv, nr=nr)
  cc._debug_stride = stride
  cc._aref_low_debug_offset = low_offset
  debug = torch.zeros((batch, stride), dtype=torch.float32)
  debug[:, nr * nr + 6 * nr:nr * nr + 7 * nr] = 1.0
  debug[:, nr * nr + nr:nr * nr + 2 * nr] = torch.tensor(
      [[.25, -.5]] * batch)
  cc._workspace = {
      "workspace_debug": debug.reshape(-1),
      "position_cache_J": torch.tensor(
          [[[1., -2.], [.5, 1.]]] * batch),
      "position_cache_qvel": torch.tensor([[.125, -.25]] * batch),
      "position_context": torch.tensor(
          [[[1., 1.5, .9, .2, 0.], [1., -.75, .8, -.1, 0.]]] * batch),
      "position_cache_rows": torch.tensor(
          [[0., 0., .75, -.25, 0., 0., 0.,
            0., 0., -.125, .375, 0., 0., 0.]] * batch),
      "position_cache_aref_low": torch.tensor(
          [[.125, -.0625]] * batch),
      "position_current_aref_low": torch.zeros((batch, nr)),
  }
  scale = torch.tensor([.0625, -.03125])
  cc._constants = {
      "row_velocity_scale_low": scale,
      "solver_dims": torch.zeros((32,), dtype=torch.int32),
  }

  def cpu_kernel(J, qvel, old_qvel, position_context, velocity_scale_low,
                 old_extra, new_extra, position_cache_rows,
                 position_cache_aref_low, workspace_debug, dims, *,
                 threads, group_size):
    assert threads == (batch * nr,) and group_size == (128,)
    J = J.reshape(batch, nr, nv)
    qvel = qvel.reshape(batch, nv)
    old_qvel = old_qvel.reshape(batch, nv)
    context = position_context.reshape(batch, nr, 5)
    cache_rows = position_cache_rows.reshape(batch, 7 * nr)
    old_low = position_cache_aref_low.reshape(batch, nr)
    debug_view = workspace_debug.reshape(batch, stride)
    for world in range(batch):
      for row in range(nr):
        velocity_delta = float(torch.dot(J[world, row],
                                         qvel[world] - old_qvel[world]))
        delta = -(float(context[world, row, 1])
                  + float(velocity_scale_low[row])) * velocity_delta
        exact = (float(cache_rows[world, nr + row]) + float(old_low[world, row])
                 + delta + float(new_extra[world * nr + row])
                 - float(old_extra[world * nr + row]))
        high = float(debug_view[world, nr * nr + nr + row])
        debug_view[world, low_offset + row] = exact - high

  cc._refresh_aref_low_kernel = cpu_kernel
  qvel = torch.tensor([[.126 + .001 * w, -.248 - .002 * w]
                       for w in range(batch)], dtype=torch.float32)
  surface = torch.zeros((batch, nr), dtype=torch.float32)
  old_extra = torch.tensor([[.2, -.3]] * batch, dtype=torch.float32)
  new_extra = torch.tensor([[.25, -.35]] * batch, dtype=torch.float32)
  cc._refresh_cached_aref_low(qvel, surface, new_extra, old_extra)

  for world in range(batch):
    for row in range(nr):
      J = cc._workspace["position_cache_J"].reshape(batch, nr, nv)[world, row]
      dv = qvel[world] - cc._workspace["position_cache_qvel"].reshape(batch, nv)[world]
      context = cc._workspace["position_context"].reshape(batch, nr, 5)[world, row]
      expected = (float(cc._workspace["position_cache_rows"].reshape(batch, 7 * nr)[world, nr + row])
                  + float(cc._workspace["position_cache_aref_low"][world, row])
                  - (float(context[1]) + float(scale[row])) * float(torch.dot(J, dv))
                  + float(new_extra[world, row] - old_extra[world, row]))
      actual = float(debug[world, nr * nr + nr + row]
                     + cc._workspace["position_current_aref_low"][world, row])
      np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)


def test_cached_aref_low_shader_indexes_static_velocity_scale_by_row():
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "coupled_constraints.metal").read_text()
  kernel = shader.split("kernel void refresh_cached_constraint_aref_low(", 1)[1]
  kernel = kernel.split("// Compute pinned dense mj_jacDot", 1)[0]
  assert "velocity_scale_low[row]" in kernel
  assert "velocity_scale_low[row_index]" not in kernel


def test_capture_position_context_retains_aref_low_and_current_qvel():
  """Execute capture itself; the cache must retain the POS residual word."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  cc = object.__new__(MetalCoupledConstraints)
  cc._torch = torch
  cc._device = torch.device("cpu")
  cc.batch_size = 1
  cc.descriptor = SimpleNamespace(nq=2, nv=2, nr=1, neq=0, nbody=1,
                                  nsite=0,
                                  eq_type=np.empty((0,), np.int32))
  cc._assembly_generation = 1
  cc._position_current_valid = True
  cc._position_context_valid = False
  cc._position_context_epoch = 3
  cc._position_context_eq_jdot_inputs_valid = False
  cc._position_current_eq_jdot_inputs_valid = False
  cc._debug_stride = 32
  cc._aref_low_debug_offset = 24
  debug = torch.zeros((32,), dtype=torch.float32)
  debug[2] = .625
  debug[24] = .125
  w = {
      "position_current_qvel": torch.tensor([.25, -.5]),
      "position_qvel": torch.zeros((2,)),
      "position_current_eq_active": torch.zeros((1,), dtype=torch.int32),
      "position_cache_eq_active": torch.zeros((1,), dtype=torch.int32),
      "position_current_cvel": torch.zeros((6,)),
      "position_cache_cvel": torch.zeros((6,)),
      "position_current_cdof": torch.zeros((12,)),
      "position_cache_cdof": torch.zeros((12,)),
      "position_current_cdof_dot": torch.zeros((12,)),
      "position_cache_cdof_dot": torch.zeros((12,)),
      "position_current_body_pos": torch.zeros((3,)),
      "position_cache_body_pos": torch.zeros((3,)),
      "position_current_body_quat": torch.zeros((4,)),
      "position_cache_body_quat": torch.zeros((4,)),
      "position_current_root_com": torch.zeros((3,)),
      "position_cache_root_com": torch.zeros((3,)),
      "position_current_site_pos": torch.empty((0,)),
      "position_cache_site_pos": torch.empty((0,)),
      "position_current_site_quat": torch.empty((0,)),
      "position_cache_site_quat": torch.empty((0,)),
      "position_context": torch.zeros((5,)),
      "position_surface_velocity": torch.zeros((1,)),
      "position_extra_aref": torch.zeros((1,)),
      "workspace_J": torch.tensor([1., -2.]),
      "position_cache_J": torch.zeros((2,)),
      "workspace_debug": debug,
      "position_cache_rows": torch.zeros((7,)),
      "position_cache_impedance": torch.zeros((1,)),
      "position_current_aref_low": torch.zeros((1,)),
      "position_cache_aref_low": torch.zeros((1,)),
      "position_cache_qvel": torch.zeros((2,)),
      "contact_row_data": torch.zeros((6,)),
      "position_cache_contact_data": torch.zeros((6,)),
      "contact_frame": torch.zeros((12,)),
      "position_cache_contact_frame": torch.zeros((12,)),
      "contact_jacobian": torch.zeros((12,)),
      "position_cache_contact_jacobian": torch.zeros((12,)),
      "position_cache_slot_packed": torch.full((1,), -1, dtype=torch.int32),
      "position_cache_slot_reverse": torch.full((1,), -1, dtype=torch.int32),
      "position_cache_slot_count": torch.zeros((1,), dtype=torch.int32),
      "position_cache_slot_overflow": torch.zeros((1,), dtype=torch.int32),
      "position_cache_pair_packed": torch.full((1,), -1, dtype=torch.int32),
      "position_cache_pair_reverse": torch.full((1,), -1, dtype=torch.int32),
      "position_cache_pair_count": torch.zeros((1,), dtype=torch.int32),
      "position_cache_pair_overflow": torch.zeros((1,), dtype=torch.int32),
  }
  cc._workspace = w
  position = torch.tensor([[[2., .75, .9, .2, 0.]]])
  surface = torch.tensor([[.125]])
  extra = torch.tensor([[.25]])
  context = cc.capture_position_context(
      position, surface_velocity=surface, extra_aref=extra)
  assert context["_epoch"] == 4
  assert cc._position_context_valid
  torch.testing.assert_close(w["position_cache_qvel"], torch.tensor([.25, -.5]))
  torch.testing.assert_close(w["position_cache_aref_low"], torch.tensor([.125]))
  torch.testing.assert_close(w["position_current_aref_low"], torch.tensor([.125]))
  torch.testing.assert_close(context["position_context"], position)
  torch.testing.assert_close(context["surface_velocity"], surface)
  torch.testing.assert_close(context["extra_aref"], extra)
  replacement = cc.capture_position_context(position)
  assert replacement["_epoch"] == 5
  torch.testing.assert_close(replacement["surface_velocity"], torch.zeros((1, 1)))
  torch.testing.assert_close(replacement["extra_aref"], torch.zeros((1, 1)))
