# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Invalid device extension samples isolate the failed row and retain state.

This is the backend's documented adapter failure contract, not a claim about
the behavior of arbitrary upstream host callbacks producing NaN.
"""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api
from mujoco_metal.extensions import default_registry, PluginType
from test_energy_lifecycle_006 import _native_only
from test_sensor_stage_plugin_lifecycle import CounterSensor
from test_sensor_stage_publication import _host
from test_sensor_integrated_stages import _PROFILES

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native adapter invalid rows')]


class InvalidRowSensor(CounterSensor):
  def run_device(self, state, **kwargs):
    values = super().run_device(state, **kwargs)
    values[1] = float('nan')
    return values


@pytest.mark.parametrize('integrator,profile', _PROFILES)
@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('operation', ['publish_all', 'publish_healthy', 'step'])
def test_actual_nonfinite_adapter_retains_failed_sample_and_state_without_poisoning_checkpoint(
    integrator, profile, stage, operation, monkeypatch):
  name = 'invalid_'+stage.lower()
  peer_name = 'valid_'+stage.lower()
  default_registry.register(InvalidRowSensor(name))
  default_registry.register(CounterSensor(peer_name))
  try:
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0" integrator="{integrator}"/>
      <worldbody><body><joint name="j" type="slide"/><geom size=".1" mass="1"/>
      </body></worldbody><sensor><user name="{name}" dim="1"
      needstage="{stage.lower()}"/><user name="{peer_name}" dim="1"
      needstage="{stage.lower()}"/><jointpos joint="j"/></sensor></mujoco>''')
    qpos = np.asarray([[.1], [.2], [.3]], np.float32)
    qvel = np.asarray([[.2], [.3], [.4]], np.float32)
    sim = MetalSimulation(model, 3, qpos=qpos, qvel=qvel, profile=profile)
    sim._sensordata = sim.state._torch.full((3, model.nsensordata), 20., device=sim.state._device)
    sim._raw_sensordata = sim._sensordata.clone()
    before = sim.snapshot()
    record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
    if operation == 'step':
      _native_only(monkeypatch, sim.step)
      np.testing.assert_array_equal(_host(sim.state._qpos)[1], before['device'].qpos[1])
      np.testing.assert_array_equal(_host(sim.state._qvel)[1], before['device'].qvel[1])
      assert _host(sim.state._status)[1] != 0
      assert np.all(_host(sim.state._qpos)[[0, 2]] != qpos[[0, 2]])
    else:
      ids = [2, 0] if operation == 'publish_healthy' else None
      _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record, env_ids=ids))
      assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
      np.testing.assert_array_equal(_host(sim.state._status), 0)
    expected_status = [0, 0, 0] if operation == 'publish_healthy' else [0, 1, 0]
    np.testing.assert_array_equal(_host(sim._sensor_plugin_status), expected_status)
    expected = np.asarray([[1., 1., 20.], [20., 20., 20.], [1., 1., 20.]], np.float32)
    if stage == 'Pos' or operation == 'step':
      expected[[0, 2], 2] = qpos[[0, 2], 0]
    np.testing.assert_array_equal(sim.step_sensordata(), expected)
    np.testing.assert_array_equal(_host(sim._raw_sensordata), expected)
    adapter = next(plugin for plugin in sim._native_plugins if plugin.name == name)
    np.testing.assert_array_equal(_host(adapter.calls), [1, 0, 1])
    healthy_adapter = next(plugin for plugin in sim._native_plugins if plugin.name == peer_name)
    np.testing.assert_array_equal(_host(healthy_adapter.calls), [1, 0, 1])
    # Checkpointing must remain usable after an adapter reports a bad peer.
    checkpoint = sim.snapshot()
    _native_only(monkeypatch, lambda: sim.restore(checkpoint))
    np.testing.assert_array_equal(sim.snapshot()['raw_sensordata'], checkpoint['raw_sensordata'])
    np.testing.assert_array_equal(sim.snapshot()['sensor_plugin_status'], checkpoint['sensor_plugin_status'])
    np.testing.assert_array_equal(sim.snapshot()['plugins'][('sensor', name)], [1, 0, 1])
    np.testing.assert_array_equal(sim.snapshot()['plugins'][('sensor', peer_name)], [1, 0, 1])
  finally:
    default_registry.unregister(name, PluginType.SENSOR)
    default_registry.unregister(peer_name, PluginType.SENSOR)
