# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Nonzero material damping, work and velocity-derivative qualification.

The pinned CPU engine is an independent oracle only. No oracle state, force
or tangent is supplied to native physics. Force/state bounds match the active
material gates; derivative bounds account for the native float32 contraction.
"""
import os

import mujoco
import numpy as np
import pytest

from test_active_material_invariants_018 import _model, _state
from test_active_material_energy_018 import _METHODS, _oracle


def _fixture(dim, damping, integrator):
  # Damping must be enabled at compile time: the pinned compiler otherwise
  # elides its edge-velocity Jacobians. Mutating that field alone is invalid.
  model = _model(dim, 1000., damping=damping)
  model.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_'+integrator.upper())
  qpos, root = _state(model)
  qvel = np.zeros((4, model.nv), np.float32)
  # Both undeformed controls move rigidly. The deformed controls additionally
  # have relative vertex motion, so damping must do actual internal work.
  qvel[:, root:root+6] = [.08, -.04, .02, .3, -.2, .4]
  qvel[2:, root+6:] = np.linspace(-.04, .06, model.nv-root-6)
  return model, qpos, qvel, root


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('damping', [.1, .4])
@pytest.mark.parametrize('integrator,profile', _METHODS)
def test_material_damping_work_derivative_and_integrated_replay(
    dim, damping, integrator, profile, record_property, monkeypatch):
  from mujoco_metal import MetalSimulation
  model, qpos, qvel, root = _fixture(dim, damping, integrator)
  refs = [_oracle(model, qpos[w], qvel[w]) for w in range(4)]
  source = np.stack([d.qfrc_passive.copy() for d in refs])
  springs = np.stack([_oracle(model, qpos[w], np.zeros(model.nv)).qfrc_passive
                      for w in range(4)])
  dissipative = source-springs
  power = np.einsum('bi,bi->b', qvel, dissipative)
  np.testing.assert_allclose(source[:2], 0., rtol=0., atol=1e-8)
  assert np.all(power[2:] < -1e-5), power
  assert np.linalg.norm(dissipative[2]) > 1e-3
  # Independent central differences cover carrier translation/rotation and
  # moving material vertices, not just a zero or single uncoupled DOF.
  columns = list(range(root, root+6)) + [root+6, model.nv-1]
  oracle_derivatives = []
  for world in (2, 3):
    matrix = []
    for dof in columns:
      hi, lo = qvel[world].astype(float), qvel[world].astype(float)
      hi[dof] += .002
      lo[dof] -= .002
      matrix.append((_oracle(model, qpos[world], hi).qfrc_passive -
                     _oracle(model, qpos[world], lo).qfrc_passive)/.004)
    oracle_derivatives.append(np.stack(matrix, axis=1))
  sim = MetalSimulation(model, batch_size=4, qpos=qpos, qvel=qvel, profile=profile)
  host = lambda tensor: tensor.detach().cpu().numpy().copy()
  def native_only(call):
    def forbidden(*args, **kwargs):
      raise AssertionError('CPU physics must not execute in native evaluation')
    with monkeypatch.context() as guard:
      for name in ('mj_forward', 'mj_forwardSkip', 'mj_step', 'mj_fwdPosition',
                   'mj_fwdVelocity', 'mj_fwdAcceleration', 'mj_fwdConstraint',
                   'mj_flex', 'mj_passive'):
        guard.setattr(mujoco, name, forbidden)
      return call()
  np.testing.assert_array_equal(host(native_only(
      lambda: sim.assembled_system(recompute=True))['status']), 0)
  native_force = host(sim._flex._qfrc_passive)
  np.testing.assert_allclose(native_force, source, rtol=7e-4, atol=5e-6)
  native_power = np.einsum('bi,bi->b', qvel, native_force-springs)
  np.testing.assert_allclose(native_power, power, rtol=7e-4, atol=5e-6)
  tangent = host(sim._flex._damping_tangent)[2:]
  np.testing.assert_allclose(tangent[:, :, columns], oracle_derivatives,
                             rtol=3e-3, atol=2e-4)
  record_property('source_damping_power', power.tolist())
  record_property('native_damping_power', native_power.tolist())
  middle = None
  for step in range(4):
    np.testing.assert_array_equal(host(native_only(sim.step)), 0)
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_allclose(host(sim.state._qpos), [d.qpos for d in refs],
                               rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(host(sim.state._qvel), [d.qvel for d in refs],
                               rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(host(sim.state._qacc), [d.qacc for d in refs],
                               rtol=8e-4, atol=1e-5)
    if step == 1:
      middle = sim.snapshot()
  final = sim.snapshot()
  assert not np.any(final['device'].warning_number)
  sim.restore(middle)
  for _ in range(2):
    np.testing.assert_array_equal(host(native_only(sim.step)), 0)
  for name in ('qpos', 'qvel', 'qacc'):
    np.testing.assert_array_equal(host(getattr(sim.state, '_'+name)),
                                 getattr(final['device'], name))
