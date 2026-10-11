# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Energy lazy flags preserve source POS/VEL ordering across real updates."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal.native_api import mj_forward
from test_energy_lifecycle_006 import _METHODS, _native_only

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')]


def _host(value):
  return value.detach().cpu().numpy().copy()


@pytest.mark.parametrize('integrator,profile', _METHODS)
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('mode', ['direct', 'history', 'interval', 'delay', 'delay_interval'])
@pytest.mark.parametrize('sensors_disabled', [False, True])
def test_energy_sensor_forward_and_consecutive_step_order_matches_source(
    integrator, profile, enabled, mode, sensors_disabled, monkeypatch):
  spec = mujoco.MjSpec.from_string(f'''<mujoco>
    <option timestep=".01" integrator="{integrator}">
      <flag energy="{'enable' if enabled else 'disable'}"/></option>
    <worldbody><body pos="0 0 1"><joint name="slider" type="slide"
      axis="0 0 1"/><geom type="sphere" size=".05" mass="1"/></body></worldbody>
    <sensor><e_potential name="potential"/><e_kinetic name="kinetic"/></sensor>
    </mujoco>''')
  if mode != 'direct':
    for name in ('potential', 'kinetic'):
      sensor = spec.sensor(name)
      sensor.nsample, sensor.interp = 8, 1
      sensor.delay = .015 if mode in ('delay', 'delay_interval') else 0
      if mode in ('interval', 'delay_interval'):
        sensor.interval = [.02, -.003]
  model = spec.compile()
  if sensors_disabled:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  q = np.array([[.125], [.25]], np.float32)
  v = np.array([[.7], [-.4]], np.float32)
  refs = []
  for pos, vel in zip(q, v):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = pos, vel
    mujoco.mj_forward(model, data)
    refs.append(data)
  sim = MetalSimulation(model, 2, qpos=q, qvel=v, profile=profile)
  _native_only(monkeypatch, lambda: mj_forward(sim))

  def compare():
    np.testing.assert_allclose(sim.step_sensordata(), [d.sensordata for d in refs],
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(_host(sim.flg_energypos), [d.flg_energypos for d in refs])
    np.testing.assert_array_equal(_host(sim.flg_energyvel), [d.flg_energyvel for d in refs])
    np.testing.assert_allclose(_host(sim.state._history), [d.history for d in refs],
                               rtol=2e-5, atol=2e-6)
    for address in model.sensor_historyadr:
      if address >= 0:
        np.testing.assert_array_equal(_host(sim.state._history)[:, address+1],
                                      [d.history[address+1] for d in refs])
  compare()
  mid = sim.snapshot()
  for _ in range(5):
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_array_equal(_host(_native_only(monkeypatch, sim.step)), 0)
    compare()
  final = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(mid))
  for _ in range(5):
    _native_only(monkeypatch, sim.step)
  replay = sim.snapshot()
  for name in ('energy', 'energy_computed', 'sensordata'):
    np.testing.assert_array_equal(replay[name], final[name])
  np.testing.assert_array_equal(replay['native_state']['history'], final['native_state']['history'])
