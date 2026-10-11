# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compiled inverse-weight residuals reach actual native equality rows."""
import os

import mujoco
import numpy as np
import pytest


def _model(kind):
  joint = '<joint name="j" type="slide"/>' if kind == 'joint' else '<freejoint/>'
  target = 'joint1="j"' if kind == 'joint' else 'body1="b"'
  anchor = ' anchor=".02 0 0"' if kind == 'connect' else ''
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" solver="Newton" iterations="0"/>
    <worldbody><body name="b">{joint}
      <geom type="sphere" size=".1" mass="1.3"/></body></worldbody>
    <equality><{kind} {target}{anchor} solimp=".83 .91 .001"/></equality>
  </mujoco>''')


@pytest.mark.parametrize('kind', ['joint', 'connect', 'weld'])
def test_compiled_inverse_weight_residuals_retain_pinned_constants(kind):
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from mujoco_metal.capacity import _runtime_buffer_sizes
  model = _model(kind)
  desc = lower_coupled_constraints(model)
  for source, high, low in (
      (model.dof_invweight0, desc.dof_invweight0, desc.dof_invweight0_low),
      (model.body_invweight0, desc.body_invweight0, desc.body_invweight0_low)):
    assert low.shape == source.shape and not low.flags.writeable
    assert np.any(low != 0)
    np.testing.assert_allclose(high.astype(float) + low.astype(float), source,
                               rtol=2e-15, atol=2e-15)
  parts = dict(_runtime_buffer_sizes(model, 2))
  assert parts['model.coupled.dof_invweight0_low'] == max(model.nv, 1)
  assert parts['model.coupled.body_invweight0_low'] == model.nbody * 2


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('kind', ['joint', 'connect', 'weld'])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_native_equality_regularizer_matches_source_and_clears_disabled_world(kind, profile, monkeypatch):
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.native_api import mj_setState
  model = _model(kind)
  qpos = np.tile(model.qpos0.astype(np.float32), (2, 1))
  data = mujoco.MjData(model)
  data.qpos[:] = qpos[0]
  mujoco.mj_forward(model, data)
  expected = np.asarray(data.efc_R).copy()
  assert len(expected) == {'joint': 1, 'connect': 3, 'weld': 6}[kind]
  sim = MetalSimulation(model, 2, qpos=qpos,
      qvel=np.zeros((2, model.nv), np.float32), profile=profile)
  # Both words share existing bindings; high-only consumers retain their
  # original prefix. The estimator counts the added constant tails.
  assert sim._coupled_constraints._constants['dof_invweight0'].numel() == 2 * max(model.nv, 1)
  assert sim._coupled_constraints._constants['body_invweight0'].numel() == 4 * model.nbody
  mj_setState(sim, {'eq_active': np.asarray([[1], [0]], np.uint8)})

  def forbidden(*args, **kwargs):
    raise AssertionError('CPU physics during native equality assembly')

  monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
  monkeypatch.setattr(mujoco, 'mj_fullM', forbidden)
  for enabled in (False, True, False):
    mj_setState(sim, {'eq_active': np.asarray([[1], [enabled]], np.uint8)})
    system = sim.assembled_system(recompute=True)
    cc = sim._coupled_constraints
    host = lambda value: value.detach().cpu().numpy().astype(float)
    np.testing.assert_array_equal(host(system['status']), 0)
    debug = host(cc._workspace['workspace_debug']).reshape(2, -1)
    start = cc._regularizer_low_debug_offset
    low = debug[:, start:start + cc.descriptor.nr]
    actual = host(system['R']) + low
    np.testing.assert_allclose(actual[0, :len(expected)], expected,
                               rtol=2e-12, atol=2e-12)
    if enabled:
      np.testing.assert_allclose(actual[1, :len(expected)], expected,
                                 rtol=2e-12, atol=2e-12)
    else:
      np.testing.assert_array_equal(actual[1], 0)
      np.testing.assert_array_equal(low[1], 0)
