# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native fresh/cached CG and Newton consume source smooth-force residuals."""
import os
import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native smooth-force qualification")
@pytest.mark.parametrize("solver", ["CG", "Newton"])
@pytest.mark.parametrize("cached", [False, True])
def test_component_smooth_force_pairs_match_pinned_mass_and_preserve_inputs(solver, cached):
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" solver="{solver}"><flag contact="disable"/></option>
    <worldbody><body><freejoint/><inertial pos="0 0 0" mass="2"
        diaginertia=".25 .375 .5"/>
      <geom type="box" size=".125 .25 .5"/></body></worldbody></mujoco>""")
  sim = MetalSimulation(model, 2, profile="integrated_scalable_v1")
  cc = sim._coupled_constraints
  dynamics = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)
  poses = dict(dynamics["poses"], root_com=dynamics["root_com"])
  hi = np.array([[100, -50, 25, .25, -.125, .5],
                 [-25, 50, 100, -.25, .5, .125]], dtype=np.float32)
  low = np.array([[1.25e-4, -2.5e-4, .75e-4, 1e-8, -2e-8, 3e-8],
                  [-3e-4, 2e-4, 1e-4, -3e-8, 2e-8, 1e-8]], dtype=np.float32)
  force = torch.tensor(hi, device="mps")
  residual = torch.tensor(low, device="mps")
  lists = sim._smooth._workspace["all_awake_lists"]
  args = (poses, None, force, sim.state._qpos, sim.state._qvel)
  kwargs = dict(component_solver=sim._component_solver,
                mass_blocks=dynamics["mass_blocks"],
                awake_dof_ids=lists["dof_ids"], awake_counts=lists["counts"],
                qfrc_smooth_low=residual)
  result = cc.run_device(*args, **kwargs)
  if cached:
    context = cc.capture_position_context()
    result = cc.run_velocity_device(context, *args, **kwargs)
  np.testing.assert_array_equal(result["status"].cpu().numpy(), 0)
  source = mujoco.MjData(model)
  mujoco.mj_forward(model, source)
  mass = np.empty((model.nv, model.nv))
  mujoco.mj_fullM(model, source, mass)
  expected = np.linalg.solve(mass, (hi.astype(float) + low).T).T
  actual = result["qacc"].cpu().numpy().astype(float) + result["qacc_low"].cpu().numpy()
  np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
  assert np.any(result["qacc_low"].cpu().numpy() != 0)
  np.testing.assert_array_equal(force.cpu().numpy(), hi)
  np.testing.assert_array_equal(residual.cpu().numpy(), low)
  assert int(cc._constants["solver_dims"][7].cpu()) == int(cc.descriptor.refsafe)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native large-row qualification")
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_large_row_block_route_is_exercised_and_restores(solver):
  """>96-row / nv>32 fixture reaches the block solve (dense_path False)."""
  from mujoco_metal.capacity import DENSE_ROW_THRESHOLD
  from mujoco_metal.simulation import MetalSimulation
  bodies = "".join(
      f'<body pos="{.3 * k} 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>'
      for k in range(6))
  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><option solver="{solver}"/><worldbody>'
      f'<geom type="plane" size="5 5 .1"/>{bodies}</worldbody></mujoco>')
  sim = MetalSimulation(model, 2, profile="integrated_scalable_v1")
  desc = sim._coupled_constraints.descriptor
  assert desc.nr > DENSE_ROW_THRESHOLD and not desc.dense_path
  snap = sim.snapshot()
  sim.step(5)
  first = sim.state.qpos.cpu().numpy().copy()
  source = mujoco.MjData(model)
  for _ in range(5):
    mujoco.mj_step(model, source)
  np.testing.assert_allclose(first[0], source.qpos, atol=1e-6)
  assert np.any(first != 0)
  sim.restore(snap)
  sim.step(5)
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(), first)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native force aggregation")
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
def test_cancelling_force_terms_keep_residual(solver):
  """passive(1) + applied(2**24) + actuator(-2**24) must not collapse to 0.

  In float32 source order 1 + 2**24 rounds back to 2**24, so a high-only sum
  loses the unit force; the compensated pair carries it to the solve.
  """
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" solver="{solver}" timestep="0.001"><flag contact="disable"/></option>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0" stiffness="1" ref="0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="j" ctrllimited="false" forcelimited="false"/></actuator>
  </mujoco>""")
  source = mujoco.MjData(model)
  source.qpos[0] = -1.0
  source.qfrc_applied[0] = 2.0**24
  source.ctrl[0] = -2.0**24
  mujoco.mj_step(model, source)
  assert source.qvel[0] == pytest.approx(model.opt.timestep * 1.0, rel=1e-9)
  sim = MetalSimulation(model, 1, profile="integrated_scalable_v1")
  sim.reset(qpos=np.array([[-1.0]], dtype=np.float32))
  sim.step(1, qfrc_applied=torch.tensor([[2.0**24]], device="mps"),
           ctrl=torch.tensor([[-2.0**24]], device="mps"))
  qvel = float(sim.state.qvel.cpu().numpy()[0, 0])
  assert qvel == pytest.approx(source.qvel[0], rel=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="zero-dof direct qualification")
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("with_residual", [False, True])
def test_zero_dof_direct_dispatch_and_cache(solver, with_residual):
  """Direct MetalCoupledConstraints dispatch and position caching on nv==0 models."""
  import torch
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option solver="{solver}"/>
    <worldbody><geom type="sphere" size=".1"/></worldbody>
  </mujoco>""")
  b = 2
  sim = MetalSimulation(model, b, profile="integrated_scalable_v1")
  cc = MetalCoupledConstraints(model, b)
  qpos = torch.zeros((b, 0), device="mps")
  qvel = torch.zeros((b, 0), device="mps")
  mass = torch.zeros((b, 0, 0), device="mps")
  force = torch.zeros((b, 0), device="mps")
  residual = torch.zeros((b, 0), device="mps") if with_residual else None
  poses = sim._smooth._fk.run_device(qpos)
  res_fresh = cc.run_device(poses, mass, force, qpos, qvel, qfrc_smooth_low=residual)
  np.testing.assert_array_equal(res_fresh["status"].cpu().numpy(), 0)
  ctx = cc.capture_position_context()
  res_cached = cc.run_velocity_device(ctx, poses, mass, force, qpos, qvel, qfrc_smooth_low=residual)
  np.testing.assert_array_equal(res_cached["status"].cpu().numpy(), 0)

