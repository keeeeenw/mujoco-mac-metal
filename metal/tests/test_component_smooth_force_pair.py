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
