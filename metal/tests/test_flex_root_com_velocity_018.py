# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public flex point velocities use the articulated-root spatial origin."""
import os

import mujoco
import numpy as np
import pytest

from test_active_material_invariants_018 import _model, _state
from test_active_material_energy_018 import _oracle

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')]


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_descendant_point_velocity_com_and_cached_stage_restore(dim, profile, monkeypatch):
  from mujoco_metal import MetalSimulation, native_api
  model = _model(dim, 1000., damping=.1)
  qpos, root = _state(model)
  qvel = np.zeros((4, model.nv), np.float32)
  qvel[:, root:root+6] = [.08, -.04, .02, .3, -.2, .4]
  qvel[2:, root+6:] = np.linspace(-.04, .06, model.nv-root-6)
  refs = [_oracle(model, qpos[w], qvel[w]) for w in range(4)]
  expected = []
  bodies = np.asarray(model.flex_vertbodyid)
  roots = np.asarray(model.body_rootid)[bodies]
  for data in refs:
    cv = data.cvel[bodies]
    expected.append(cv[:, 3:] + np.cross(
        cv[:, :3], data.flexvert_xpos-data.subtree_com[roots]))
  expected = np.stack(expected)
  edge_rates = np.stack([data.flexedge_velocity.copy() for data in refs])
  assert np.linalg.norm(expected) > .1
  assert np.linalg.norm(edge_rates[2:]) > .01
  np.testing.assert_allclose(edge_rates[:2], 0., rtol=0., atol=1e-12)
  sim = MetalSimulation(model, batch_size=4, qpos=qpos, qvel=qvel, profile=profile)
  host = lambda tensor: tensor.detach().cpu().numpy().copy()

  def native_only(call):
    def forbidden(*args, **kwargs):
      raise AssertionError('CPU physics must not execute in native evaluation')
    with monkeypatch.context() as guard:
      for name in ('mj_forward', 'mj_forwardSkip', 'mj_step', 'mj_comVel', 'mj_flex'):
        guard.setattr(mujoco, name, forbidden)
      return call()

  def compare():
    np.testing.assert_allclose(host(sim._flex._flexvert_xvel), expected,
                               rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(host(sim._flex._flexedge_velocity), edge_rates,
                               rtol=3e-5, atol=3e-6)
    centers = host(sim._smooth._workspace['root_com']).reshape(4, model.nbody, 3)
    # The world body's stationary spatial velocity has no COM requirement.
    np.testing.assert_allclose(centers[:, 1:],
        [data.subtree_com[np.asarray(model.body_rootid)[1:]] for data in refs],
        rtol=3e-5, atol=3e-6)

  np.testing.assert_array_equal(host(native_only(
      lambda: sim.assembled_system(recompute=True))['status']), 0)
  compare()
  checkpoint = sim.snapshot()
  frozen = host(sim._flex._flexvert_xvel)
  record = native_only(lambda: native_api.mj_fwdPosition(sim, return_record=True))
  native_only(lambda: native_api.mj_fwdVelocity(sim, record=record))
  compare()
  sim.reset(env_ids=[0], qpos=qpos[:1], qvel=-qvel[:1])
  native_only(lambda: sim.assembled_system(recompute=True))
  np.testing.assert_allclose(host(sim._flex._flexvert_xvel)[0], -expected[0],
                             rtol=3e-5, atol=3e-6)
  np.testing.assert_array_equal(host(sim._flex._flexvert_xvel)[1:], frozen[1:])
  sim.restore(checkpoint)
  native_only(lambda: sim.assembled_system(recompute=True))
  compare()
  np.testing.assert_array_equal(host(sim._flex._flexvert_xvel), frozen)
