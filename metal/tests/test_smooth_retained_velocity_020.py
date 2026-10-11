# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public RNE bias must consume the velocity represented by retained state."""
import os

import mujoco
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')]


def _fixture():
  # Binary-exact inertia/identity pose isolate velocity-input loss from the
  # separately unqualified general articulated position/inertia precision.
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 0"/>
    <worldbody><body><freejoint/>
      <inertial mass="1" pos="0 0 0" diaginertia=".25 .5 .625"/>
      <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>''')
  velocities = np.asarray([[.03, -.01, .02, 1.+2**-27, -.75+2**-26, .5+2**-27],
                           [-.02, .04, .01, -.5+2**-27, 1.+2**-26, -.75+2**-27]])
  return model, velocities, _expected(model, velocities)


def _expected(model, velocities):
  expected = []
  for velocity in velocities:
    data = mujoco.MjData(model)
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    expected.append(data.qfrc_bias.copy())
  return np.asarray(expected)


def _bias(sim):
  sim.assembled_system(recompute=True)
  high = sim._smooth._workspace['bias'].detach().cpu().numpy().astype(float)
  low = sim._smooth._workspace['bias_low'].detach().cpu().numpy().astype(float)
  return (high+low).reshape(sim.batch_size, sim.model.nv)


def test_public_bias_retains_source_velocity_and_restore_reset(monkeypatch):
  from mujoco_metal import MetalSimulation
  model, velocities, expected = _fixture()
  changed = velocities.copy()
  changed[0] = changed[0].astype(np.float32)
  changed[0, 3] = .5
  mutation_expected = _expected(model, changed)
  replacement_expected = _expected(model, velocities.astype(np.float32))
  sim = MetalSimulation(model, batch_size=2, qvel=velocities,
                        profile='integrated_scalable_v1')
  def forbidden(*args, **kwargs):
    raise AssertionError('CPU physics called during native bias evaluation')
  monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
  monkeypatch.setattr(mujoco, 'mj_step', forbidden)
  np.testing.assert_allclose(_bias(sim), expected, rtol=0, atol=2e-13)
  checkpoint = sim.snapshot()
  # Actual tensor writes clear residuals of the changed row; replacement
  # clears every row even when its public high values are unchanged.
  sim.state._qvel[0, 3] = .5
  np.testing.assert_allclose(_bias(sim), mutation_expected, rtol=0, atol=2e-13)
  sim.restore(checkpoint)
  sim.state._qvel = sim.state._qvel.clone()
  np.testing.assert_allclose(_bias(sim), replacement_expected, rtol=0, atol=2e-13)
  sim.restore(checkpoint)
  # Row zero changes while the second world's retained source input survives.
  sim.reset(env_ids=[0], qvel=np.zeros((1, model.nv)))
  reset = _bias(sim)
  np.testing.assert_array_equal(reset[0], 0.)
  np.testing.assert_allclose(reset[1], expected[1], rtol=0, atol=2e-13)
  sim.restore(checkpoint)
  np.testing.assert_allclose(_bias(sim), expected, rtol=0, atol=2e-13)
  from mujoco_metal.capacity import _runtime_buffer_sizes
  charged = dict(_runtime_buffer_sizes(model, 2))['smooth.bias_velocity_words']
  assert sim._smooth._workspace['bias_velocity_words'].numel() == charged == 6*model.nv


def test_host_bias_entry_keeps_binary64_velocity_residuals():
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  model, velocities, expected = _fixture()
  smooth = MetalSmoothDynamics(load_model(model), batch_size=2)
  result = smooth.run(np.tile(model.qpos0, (2, 1)), velocities)
  high = result['qfrc_bias'].detach().cpu().numpy().astype(float)
  low = result['qfrc_bias_low'].detach().cpu().numpy().astype(float)
  np.testing.assert_allclose(high+low, expected, rtol=0, atol=2e-13)
