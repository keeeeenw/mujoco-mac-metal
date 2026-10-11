# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual warm selection must use the retained original objective operands."""
import os

import mujoco
import numpy as np
import pytest


def _boundary_model(solver):
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" solver="{solver}" iterations="0"/>
    <worldbody><body><joint name="j" type="slide"/>
      <geom type="sphere" size=".1" mass="1.3"/></body></worldbody>
    <equality><joint joint1="j" solimp=".83 .91 .001"/></equality>
  </mujoco>''')


def _boundary_oracle(model, accepted=False):
  qpos = np.asarray([0.10004884004592896], np.float32)
  seed = np.asarray([-500.24420166015625], np.float32)
  if accepted:
    seed[0] = np.nextafter(seed[0], np.float32(0.))
  data = mujoco.MjData(model)
  data.qpos[:], data.qacc_warmstart[:] = qpos, seed
  mujoco.mj_forward(model, data)
  assert data.nefc == 1
  assert data.qacc[0] == (float(seed[0]) if accepted else 0.0)
  mass = np.empty((1, 1))
  mujoco.mj_fullM(model, data, mass)
  cold_cost, warm_cost = np.zeros(1), np.zeros(1)
  target = float(data.efc_aref[0])
  mujoco.mj_constraintUpdate(model, data, np.asarray([-target]), cold_cost, 1)
  mujoco.mj_constraintUpdate(model, data, np.asarray([float(seed[0])-target]), warm_cost, 1)
  delta = float(warm_cost[0]+.5*mass[0, 0]*float(seed[0])**2-cold_cost[0])
  rounded_mass = float(np.float32(mass[0, 0]))
  rounded_R = float(np.float32(data.efc_R[0]))
  rounded_delta = .5*rounded_mass*float(seed[0])**2 + .5/rounded_R*(
      (float(seed[0])-target)**2-target**2)
  if accepted:
    assert delta < 0
  else:
    assert delta > .005 and rounded_delta < -.004
  return qpos, seed, data.qacc.copy()


@pytest.mark.parametrize('solver', ['CG', 'Newton'])
@pytest.mark.parametrize('accepted', [False, True])
def test_pinned_warm_boundary_selects_original_operator_choice(solver, accepted):
  _boundary_oracle(_boundary_model(solver), accepted)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('solver', ['CG', 'Newton'])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('accepted_seed', [False, True])
def test_native_zero_budget_warm_boundary_matches_source(solver, profile, accepted_seed, monkeypatch):
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.native_api import mj_setState

  model = _boundary_model(solver)
  qpos, seed, expected = _boundary_oracle(model, accepted_seed)
  sim = MetalSimulation(model, batch_size=1, qpos=qpos[None, :],
                        qvel=np.zeros((1, 1), np.float32), profile=profile)
  mj_setState(sim, {'qacc_warmstart': seed[None, :]})

  def forbidden(*args, **kwargs):
    raise AssertionError('CPU forward/mass physics during native warm selection')

  monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
  monkeypatch.setattr(mujoco, 'mj_fullM', forbidden)
  system = sim.assembled_system(recompute=True)
  host = lambda value: value.detach().cpu().numpy().astype(float)
  np.testing.assert_array_equal(host(system['status']), 0)
  accepted = host(system['qacc'])+host(system['qacc_low'])
  np.testing.assert_allclose(accepted[0], expected, rtol=0, atol=2e-12)
  np.testing.assert_array_equal(host(sim.state._qacc_warmstart)[0], seed)
