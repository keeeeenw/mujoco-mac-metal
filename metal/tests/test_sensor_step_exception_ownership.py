# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Exceptions after actual sensor evaluation retain lazy and live samples."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation
from mujoco_metal.extensions import default_registry, PluginType
from test_energy_lifecycle_006 import _native_only
from test_lifecycle_commit_rollback import _same
from test_sensor_stage_plugin_lifecycle import CounterSensor
from test_sensor_integrated_stages import _PROFILES

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native step sample rollback')]


@pytest.mark.parametrize('integrator,profile', _PROFILES)
@pytest.mark.parametrize('initialized', [False, True])
def test_exception_after_actual_sensor_dispatch_restores_old_bindings_samples_and_adapter_state(
    integrator, profile, initialized, monkeypatch):
  name = 'exception_counter'
  default_registry.register(CounterSensor(name))
  try:
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"
      integrator="{integrator}"/><worldbody><body><joint name="j" type="slide"/>
      <geom size=".1" mass="1"/></body></worldbody><sensor><jointpos joint="j"/>
      <jointvel joint="j"/><user name="{name}" dim="1" needstage="pos"/>
      </sensor></mujoco>''')
    sim = MetalSimulation(model, 3, qpos=np.asarray([[.1], [.2], [.3]], np.float32),
        qvel=np.asarray([[.2], [.3], [.4]], np.float32), profile=profile)
    if initialized:
      _native_only(monkeypatch, lambda: sim.step(3))
    before = sim.snapshot()
    old_raw, old_stored = sim._raw_sensordata, sim._sensordata
    original, calls = type(sim)._store_step_sensors, []
    with monkeypatch.context() as guard:
      def fail_after_actual_dispatch(owner, *args, **kwargs):
        original(owner, *args, **kwargs)
        mask = kwargs.get('compute_worlds')
        if mask is not None and not bool(mask.any().item()):
          # Device autorecovery can issue a fully masked no-op. Inject the
          # fault after actual sensor/adapter publication, not that no-op.
          return
        calls.append(True)
        assert owner._raw_sensordata is not None and owner._sensordata is not None
        assert np.all(owner.snapshot()['plugins'][('sensor', name)] > before['plugins'][('sensor', name)])
        raise RuntimeError('after actual step sensor evaluation')
      guard.setattr(type(sim), '_store_step_sensors', fail_after_actual_dispatch)
      with pytest.raises(RuntimeError, match='after actual step sensor evaluation'):
        _native_only(guard, sim.step)
    assert calls == [True]
    assert sim._raw_sensordata is old_raw and sim._sensordata is old_stored
    _same(sim.snapshot(), before)
    _native_only(monkeypatch, lambda: sim.step(2))
    expected = sim.snapshot()
    _native_only(monkeypatch, lambda: sim.restore(before))
    _native_only(monkeypatch, lambda: sim.step(2))
    _same(sim.snapshot(), expected)
  finally:
    default_registry.unregister(name, PluginType.SENSOR)
