# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Execute original-operator mass residuals in fresh and retained CC solves."""
import os

import mujoco
import numpy as np
import pytest

_GPU = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                          reason="native coupled original mass qualification")


def test_mass_residual_flag_restores_nested_observer_bits_after_failure():
  import torch
  from mujoco_metal.coupled_constraints import _mass_residual_input_stage
  dims = torch.zeros(32, dtype=torch.int32)
  dims[7] = 1 | 2 | 4 | 16 | 32 | 64 | 128 | 256
  original = int(dims[7])
  with pytest.raises(RuntimeError):
    with _mass_residual_input_stage(dims, True):
      assert int(dims[7]) == original | 512
      with _mass_residual_input_stage(dims, True):
        raise RuntimeError("restore nested residual ownership")
  assert int(dims[7]) == original
  with _mass_residual_input_stage(dims, False):
    assert int(dims[7]) == original


def _fixture(solver, storage, friction=False, exact=False):
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" solver="{solver}"><flag contact="disable"/></option>
    <worldbody><body><joint type="free" frictionloss="{.125 if friction else 0}"/>
      <inertial pos="0 0 0" mass="1.3"
      diaginertia=".023125 .024125 .03125"/></body></worldbody></mujoco>''')
  if exact:
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  sim = MetalSimulation(model, 2, profile=("integrated_scalable_v1"
                                          if storage == "block_sparse"
                                          else "integrated_euler_v1"))
  dynamics = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)
  poses = dict(dynamics["poses"], root_com=dynamics["root_com"])
  lists = sim._smooth._workspace["all_awake_lists"]
  kwargs = dict(mass_low=dynamics.get("mass_matrix_low"))
  mass = dynamics.get("mass_matrix")
  if storage == "block_sparse":
    mass = None
    kwargs = dict(component_solver=sim._component_solver,
        mass_blocks=dynamics["mass_blocks"],
        mass_blocks_low=dynamics["mass_blocks_low"],
        awake_dof_ids=lists["dof_ids"], awake_counts=lists["counts"])
  reference = mujoco.MjData(model)
  mujoco.mj_forward(model, reference)
  full_mass = np.empty((model.nv, model.nv))
  mujoco.mj_fullM(model, reference, full_mass)
  return sim, dynamics, poses, mass, kwargs, full_mass


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("storage", ["dense", "block_sparse"])
@pytest.mark.parametrize("cached", [False, True])
def test_fresh_and_cached_nonbinary_source_mass(solver, storage, cached, monkeypatch):
  import torch
  sim, dynamics, poses, mass, kwargs, full_mass = _fixture(solver, storage)
  cc = sim._coupled_constraints
  hi = np.array([[100, -50, 25, .25, -.125, .5],
                 [-25, 50, 100, -.25, .5, .125]], dtype=np.float32)
  low = np.array([[1e-4, -2e-4, 3e-4, 1e-8, -2e-8, 3e-8],
                  [-3e-4, 2e-4, 1e-4, -3e-8, 2e-8, 1e-8]], dtype=np.float32)
  force, force_low = (torch.tensor(value, device="mps") for value in (hi, low))
  original = {key: value.cpu().numpy().copy() for key, value in kwargs.items()
              if isinstance(value, torch.Tensor)}
  args = (poses, mass, force, sim.state._qpos, sim.state._qvel)
  def forbidden(*args, **kwargs):
    raise AssertionError("CPU mass or forward computation in native CC")
  monkeypatch.setattr(mujoco, "mj_fullM", forbidden)
  monkeypatch.setattr(mujoco, "mj_forward", forbidden)
  result = cc.run_device(*args, qfrc_smooth_low=force_low, **kwargs)
  if cached:
    context = cc.capture_position_context()
    result = cc.run_velocity_device(context, *args, qfrc_smooth_low=force_low, **kwargs)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  actual = result["qacc"].cpu().numpy().astype(float) + result["qacc_low"].cpu().numpy()
  expected = np.linalg.solve(full_mass, (hi.astype(float) + low).T).T
  np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
  for key, value in original.items():
    np.testing.assert_array_equal(kwargs[key].cpu().numpy(), value)
  np.testing.assert_array_equal(force.cpu().numpy(), hi)
  np.testing.assert_array_equal(force_low.cpu().numpy(), low)
  assert int(cc._constants["solver_dims"][7].cpu()) == int(cc.descriptor.refsafe)
  from mujoco_metal.capacity import estimate_workspace
  d = cc.descriptor
  parts, _ = estimate_workspace(d.nv, d.npairs, d.ncontacts_max, d.nr,
      d.nr_joint, d.neq, 2, d.solver_type, ntree=cc._solver_island_ntree,
      nbody=cc._solver_island_nbody, nsite=d.nsite, mass_storage=storage,
      component_nnz=cc._component_operator_nnz,
      ncomponent=cc._component_operator_ncomponent,
      component_configured=storage == "block_sparse")
  sizes = dict(parts)
  assert sizes["workspace_debug"] == cc._workspace["workspace_debug"].numel() * 4
  assert sizes["solver_dims"] == cc._constants["solver_dims"].numel() * 4


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("storage", ["dense", "block_sparse"])
@pytest.mark.parametrize("cached", [False, True])
def test_mass_residual_failure_masking_recovery_and_metadata_atomicity(solver, storage, cached):
  import torch
  sim, dynamics, poses, mass, kwargs, full_mass = _fixture(solver, storage)
  cc = sim._coupled_constraints
  force = torch.ones((2, cc.descriptor.nv), device="mps")
  args = (poses, mass, force, sim.state._qpos, sim.state._qvel)
  result = cc.run_device(*args, **kwargs)
  context = cc.capture_position_context()
  dispatch = (lambda **kw: cc.run_velocity_device(context, *args, **kw)) if cached else (
      lambda **kw: cc.run_device(*args, **kw))
  low_key = "mass_low" if storage == "dense" else "mass_blocks_low"
  low = kwargs[low_key]
  saved = low.clone()
  before = {key: cc._workspace[key].cpu().numpy().copy()
            for key in ("workspace_debug", "out_acc", "out_status")}
  bad = dict(kwargs)
  bad[low_key] = torch.zeros((2, 1), device="mps")
  with pytest.raises(ValueError, match=low_key):
    dispatch(**bad)
  for key, value in before.items():
    np.testing.assert_array_equal(cc._workspace[key].cpu().numpy(), value)
  low[1].fill_(float("nan"))
  selected = torch.tensor([1, 0], dtype=torch.int32, device="mps")
  result = dispatch(world_mask=selected, **kwargs)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  stride = cc._debug_stride
  np.testing.assert_array_equal(cc._workspace["workspace_debug"].cpu().numpy()[stride:],
                               before["workspace_debug"][stride:])
  result = dispatch(**kwargs)
  assert int(result["status"][0].cpu()) == 0
  assert int(result["status"][1].cpu()) != 0
  low.copy_(saved)
  result = dispatch(**kwargs)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  pair = result["qacc"].cpu().numpy().astype(float) + result["qacc_low"].cpu().numpy()
  np.testing.assert_allclose(pair, np.tile(np.linalg.solve(full_mass, np.ones(6)), (2, 1)),
                             rtol=2e-12, atol=2e-12)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("storage", ["dense", "block_sparse"])
@pytest.mark.parametrize("cached", [False, True])
def test_actual_friction_constraints_use_original_mass(solver, storage, cached):
  import torch
  sim, dynamics, poses, mass, kwargs, full_mass = _fixture(solver, storage, friction=True)
  cc = sim._coupled_constraints
  force = torch.full((2, cc.descriptor.nv), 1., device="mps")
  args = (poses, mass, force, sim.state._qpos, sim.state._qvel)
  result = cc.run_device(*args, **kwargs)
  if cached:
    result = cc.run_velocity_device(cc.capture_position_context(), *args, **kwargs)
  # The compiled capacity includes two inactive joint-limit guard rows.
  assert cc.descriptor.nr == 8
  np.testing.assert_array_equal(np.count_nonzero(result["lambda"].cpu().numpy(), axis=1), 6)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  # All six friction rows saturate their source bounds. The independent
  # oracle therefore uses exactly 1 - .125 force with the pinned full mass.
  np.testing.assert_allclose(result["qfrc_constraint"].cpu().numpy(), -.125,
                             rtol=0, atol=2e-12)
  pair = result["qacc"].cpu().numpy().astype(float) + result["qacc_low"].cpu().numpy()
  np.testing.assert_allclose(pair, np.tile(np.linalg.solve(full_mass, np.full(6, .875)), (2, 1)),
                             rtol=2e-12, atol=2e-12)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("cached", [False, True])
def test_component_mass_and_armature_residuals_remain_borrowed_inputs(solver, cached):
  import torch
  sim, dynamics, poses, mass, kwargs, full_mass = _fixture(solver, "block_sparse")
  cc = sim._coupled_constraints
  arm_source = np.diag([.01, .02, .03, .04, .05, .06])
  arm_hi = arm_source.astype(np.float32)
  arm_low = (arm_source - arm_hi.astype(float)).astype(np.float32)
  kwargs["tendon_armature_blocks"] = torch.tensor(np.tile(arm_hi.reshape(1, -1), (2, 1)), device="mps")
  kwargs["tendon_armature_blocks_low"] = torch.tensor(np.tile(arm_low.reshape(1, -1), (2, 1)), device="mps")
  saved = {key: kwargs[key].cpu().numpy().copy() for key in
           ("mass_blocks", "mass_blocks_low", "tendon_armature_blocks", "tendon_armature_blocks_low")}
  args = (poses, mass, torch.ones((2, 6), device="mps"), sim.state._qpos, sim.state._qvel)
  result = cc.run_device(*args, **kwargs)
  if cached:
    result = cc.run_velocity_device(cc.capture_position_context(), *args, **kwargs)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  actual = result["qacc"].cpu().numpy().astype(float) + result["qacc_low"].cpu().numpy()
  expected = np.linalg.solve(full_mass + arm_source, np.ones(6))
  np.testing.assert_allclose(actual, np.tile(expected, (2, 1)), rtol=2e-12, atol=2e-12)
  for key, original in saved.items():
    np.testing.assert_array_equal(kwargs[key].cpu().numpy(), original)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_component_exact_impedance_row_solve_consumes_source_mass_residual(solver):
  import torch
  sim, dynamics, poses, mass, kwargs, full_mass = _fixture(
      solver, "block_sparse", friction=True, exact=True)
  cc = sim._coupled_constraints
  args = (poses, mass, torch.ones((2, 6), device="mps"), sim.state._qpos, sim.state._qvel)
  # Physical RHS storage includes a sentinel beyond the logical row count.
  # Poison it so incomplete dispatch coverage cannot pass by fresh allocation.
  sim._constraint_impedance._workspace["component_rhs"].fill_(123.0)
  result = cc.run_device(*args, exact_impedance=sim._constraint_impedance, **kwargs)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  # Recompute explicitly before the following acceleration solve can reuse
  # the component solver's borrowed multi-RHS result backing.
  sim._constraint_impedance._workspace["component_rhs"].fill_(-321.0)
  exact = cc.recompute_exact_impedance_device(sim._constraint_impedance,
      component_solver=sim._component_solver, mass_blocks=kwargs["mass_blocks"],
      mass_blocks_low=kwargs["mass_blocks_low"],
      dof_ids=kwargs["awake_dof_ids"], counts=kwargs["awake_counts"])
  np.testing.assert_array_equal(exact["status"].cpu().numpy(), 0)
  pair = sim._component_solver.paired_output.cpu().numpy().astype(float).sum(axis=0)
  jacobian = cc.materialize_jacobian().cpu().numpy().astype(float)
  for world in range(2):
    np.testing.assert_allclose(pair[world, :cc.descriptor.nr],
        np.linalg.solve(full_mass, jacobian[world].T).T, rtol=2e-12, atol=2e-12)
  np.testing.assert_array_equal(pair[:, cc.descriptor.nr], 0)
