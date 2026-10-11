# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Independent conservative-material work and stability qualification.

The energy oracle integrates the pinned simplex spring law in
MuJoCo 3.10.0 engine_passive.c:mj_flexPassive (zero Rayleigh damping).
It uses original compiled binary64 constants, not native force/tangent buffers.
Energy evaluation and CPU stepping occur only in the test oracle. These checks
do not claim that upstream's potential-energy sensor includes flex elasticity.
"""
import os

import mujoco
import numpy as np
import pytest

from test_active_material_invariants_018 import _model, _state

_METHODS = [('Euler', 'integrated_scalable_v1'),
            ('RK4', 'integrated_rk4_v1'),
            ('implicit', 'integrated_implicit_v1'),
            ('implicitfast', 'integrated_implicitfast_v1')]


def _elastic_energy(model, data):
  """U = 1/4 delta^T K delta; -grad U is the pinned spring force."""
  total = 0.
  for flex in range(model.nflex):
    dim = int(model.flex_dim[flex])
    assert dim in (2, 3) and int(model.flex_interp[flex]) == 0
    nedge = 3 if dim == 2 else 6
    kadr = int(model.flex_stiffnessadr[flex])
    eadr = int(model.flex_edgeadr[flex])
    first = int(model.flex_elemedgeadr[flex])
    for element in range(int(model.flex_elemnum[flex])):
      ids = eadr + np.asarray(model.flex_elemedge[
          first+element*nedge:first+(element+1)*nedge], dtype=int)
      delta = data.flexedge_length[ids]**2 - model.flexedge_length0[ids]**2
      packed = model.flex_stiffness[kadr+21*element:kadr+21*(element+1)]
      metric = np.zeros((nedge, nedge))
      cursor = 0
      for i in range(nedge):
        for j in range(i, nedge):
          metric[i, j] = metric[j, i] = packed[cursor]
          cursor += 1
      total += .25 * float(delta @ metric @ delta)
  return total


def _total_energy(model, data):
  matrix = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, matrix)
  return _elastic_energy(model, data) + .5 * float(data.qvel @ matrix @ data.qvel)


def _oracle(model, qpos, qvel):
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  return data


@pytest.mark.parametrize('dim', [2, 3])
def test_simplex_energy_gradient_is_pinned_generalized_spring_force(dim):
  model = _model(dim, 4000.)
  qpos, _ = _state(model)
  velocity = np.zeros(model.nv)
  for world in (2, 3):
    data = _oracle(model, qpos[world], velocity)
    assert _elastic_energy(model, data) > 1e-6
    for dof in range(model.nv):
      direction = np.eye(1, model.nv, dof).ravel()
      samples = []
      for sign in (-1., 1.):
        displaced = data.qpos.copy()
        mujoco.mj_integratePos(model, displaced, direction, sign*1e-6)
        samples.append(_elastic_energy(model, _oracle(model, displaced, velocity)))
      gradient = (samples[1]-samples[0]) / 2e-6
      np.testing.assert_allclose(-gradient, data.qfrc_spring[dof],
                                 rtol=2e-7, atol=2e-7)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('young', [1000., 4000.])
@pytest.mark.parametrize('timestep', [.00002, .0002])
@pytest.mark.parametrize('integrator,profile', _METHODS)
def test_active_carrier_material_trajectory_energy_and_stability(
    dim, young, timestep, integrator, profile, record_property):
  from mujoco_metal import MetalSimulation
  model = _model(dim, young)
  model.opt.timestep = timestep
  model.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_'+integrator.upper())
  # Both worlds are deformed, one also has a rotated and translated carrier.
  qpos = _state(model)[0][2:].copy()
  qvel = np.zeros((2, model.nv), np.float32)
  refs = [_oracle(model, qpos[w], qvel[w]) for w in range(2)]
  initial = np.asarray([_total_energy(model, data) for data in refs])
  assert np.all(initial > 1e-6)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel, profile=profile)
  host = lambda tensor: tensor.detach().cpu().numpy().copy()
  system = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(host(system['status']), 0)
  np.testing.assert_allclose(host(sim._flex._qfrc_passive),
                             [d.qfrc_spring for d in refs], rtol=7e-4, atol=5e-6)
  source_drift, native_drift, energy_error = 0., 0., 0.
  middle = None
  for step in range(6):
    np.testing.assert_array_equal(host(sim.step()), 0)
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_allclose(host(sim.state._qpos), [d.qpos for d in refs],
                               rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(host(sim.state._qvel), [d.qvel for d in refs],
                               rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(host(sim.state._qacc), [d.qacc for d in refs],
                               rtol=8e-4, atol=1e-5)
    snapshot = sim.state.snapshot()
    nq, nv = model.nq, model.nv
    words = snapshot.precision_words.astype(float)
    full_q = snapshot.qpos.astype(float)+words[:, :nq]+words[:, nq:2*nq]
    full_v = snapshot.qvel.astype(float)+words[:, 2*nq:2*nq+nv]+words[:, 2*nq+nv:]
    source = np.asarray([_total_energy(model, _oracle(model, d.qpos, d.qvel))
                         for d in refs])
    native = np.asarray([_total_energy(model, _oracle(model, full_q[w], full_v[w]))
                         for w in range(2)])
    source_drift = max(source_drift, float(np.max(np.abs(source/initial-1))))
    native_drift = max(native_drift, float(np.max(np.abs(native/initial-1))))
    energy_error = max(energy_error, float(np.max(np.abs(native-source)/initial)))
    assert not np.any(snapshot.warning_number)
    if step == 3:
      middle = sim.snapshot()
  # A finite short conservative trajectory must not gain/drop significant
  # energy. Both engines are held to the same bound, with native/oracle error
  # separately tighter than integration drift.
  assert source_drift < .02
  assert native_drift < .02
  assert energy_error < 1e-3
  source_motion = np.linalg.norm(np.stack([d.qvel for d in refs]))
  native_motion = np.linalg.norm(host(sim.state._qvel))
  # The soft membrane at the smaller timestep has real motion below 1e-4.
  # Prove the fixture moves, then require essentially that same amount of
  # motion natively; a frozen material cannot pass an absolute error bound.
  assert source_motion > 1e-5
  assert native_motion > .99 * source_motion
  for name, value in [('max_source_relative_energy_drift', source_drift),
                      ('max_native_relative_energy_drift', native_drift),
                      ('max_relative_energy_disagreement', energy_error)]:
    record_property(name, value)
  final = {name: host(getattr(sim.state, '_'+name)) for name in ('qpos', 'qvel', 'qacc')}
  sim.restore(middle)
  for _ in range(2):
    np.testing.assert_array_equal(host(sim.step()), 0)
  for name, expected in final.items():
    np.testing.assert_array_equal(host(getattr(sim.state, '_'+name)), expected)
