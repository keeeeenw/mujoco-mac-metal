# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public RK4 coverage above the historical dense/body/row dimensions."""

import os

import mujoco
import numpy as np
import pytest


def _large_independent_slider_model(count=130):
  bodies = "".join(
      f'<body name="b{i}" pos="{i * .01} 0 0">'
      f'<joint name="j{i}" type="slide" axis="0 1 0" '
      'limited="true" range="0 1" frictionloss=".025" '
      'solreflimit=".02 1"/>'
      '<inertial pos="0 0 0" mass=".2" '
      'diaginertia=".001 .001 .001"/>'
      '</body>'
      for i in range(count))
  return mujoco.MjModel.from_xml_string(
      '<mujoco><option integrator="RK4" timestep=".0005" '
      'gravity="0 0 0" iterations="3">'
      '<flag contact="disable"/></option>'
      f'<worldbody>{bodies}</worldbody></mujoco>')


def test_rk4_large_model_host_lowering_exceeds_legacy_dimensions():
  """The model exercises more than 64 DOFs, 64 bodies, and 256 rows."""
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.simulation import _model_component_capacity_limits

  model = _large_independent_slider_model()
  limits = _model_component_capacity_limits(model, batch_size=2)
  rows = lower_coupled_constraints(model, limits=limits)
  assert model.nv == 130
  assert model.nbody > 64
  assert rows.nr > 256
  assert limits.max_nv >= model.nv
  assert limits.max_rows >= rows.nr


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_public_rk4_component_step_above_legacy_dimensions_matches_pinned():
  """Exercise the actual sparse mass and RK4 stage path at large dimensions."""
  import torch

  from mujoco_metal.simulation import MetalSimulation

  model = _large_independent_slider_model()
  batch = 2
  qpos = np.stack((np.full(model.nq, .002),
                   np.full(model.nq, .004))).astype(np.float32)
  qvel = np.stack((np.linspace(-.02, .02, model.nv),
                   np.linspace(.015, -.015, model.nv))).astype(np.float32)
  references = []
  for world in range(batch):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_step(model, data)
    references.append(data)

  sim = MetalSimulation(model, batch_size=batch, qpos=qpos, qvel=qvel,
                        profile="integrated_rk4_v1")
  assert sim._component_mass_enabled
  assert sim._component_solver is not None
  assert sim._smooth.mass_block_layout["ncomponent"] == model.nv
  status = sim.step()
  assert torch.all(status == 0)
  for world, reference in enumerate(references):
    np.testing.assert_allclose(
        sim.state.qpos[world].detach().cpu().numpy(), reference.qpos,
        rtol=5e-4, atol=3e-6)
    np.testing.assert_allclose(
        sim.state.qvel[world].detach().cpu().numpy(), reference.qvel,
        rtol=5e-4, atol=3e-5)
    np.testing.assert_allclose(
        sim.state.qacc[world].detach().cpu().numpy(), reference.qacc,
        rtol=5e-4, atol=3e-4)
