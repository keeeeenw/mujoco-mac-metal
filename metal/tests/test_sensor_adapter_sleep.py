# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Object-bound USER adapters remain awake, as the pinned source requires."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api
from mujoco_metal.extensions import default_registry
from test_sensor_stage_plugin_lifecycle import CounterSensor
from test_energy_lifecycle_006 import _native_only
from test_sensor_stage_publication import _host
from test_sensor_integrated_stages import _PROFILES
from test_awake_simulation_017 import DAMPED_XML

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native adapter sleep composition')]


@pytest.mark.parametrize('integrator,profile', _PROFILES)
@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
def test_object_bound_adapter_remains_awake_for_sleeping_body_against_source(integrator, profile, stage, monkeypatch):
  name = 'sleep_'+stage.lower()
  default_registry.register(CounterSensor(name))
  callback = mujoco.get_mjcb_sensor()
  try:
    xml = DAMPED_XML.replace('<option', f'<option integrator="{integrator}"').replace('<option', '<size nuserdata="1"/><option').replace(
        '</mujoco>', f'<sensor><user name="{name}" objtype="joint" objname="slider" '
        f'dim="1" needstage="{stage.lower()}"/></sensor></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    # The second world stays awake; the first reaches real source sleep.
    velocity = np.asarray([[.04], [2.]], np.float32)
    if profile == 'integrated_euler_v1':
      # The legacy dense profile rejects sleep at admission. The scalable
      # Euler profile below exercises the actual source sleep behavior.
      with pytest.raises(ValueError, match='sleep mode is unsupported'):
        MetalSimulation(model, 2, qvel=velocity, profile=profile)
      return
    sim = MetalSimulation(model, 2, qvel=velocity, profile=profile)
    refs = [mujoco.MjData(model), mujoco.MjData(model)]
    expected_stage = getattr(mujoco.mjtStage, 'mjSTAGE_'+stage.upper())
    def source_counter(m, data, needstage):
      if int(needstage) == int(expected_stage):
        data.userdata[0] += 1
        data.sensordata[0] = data.userdata[0]
    mujoco.set_mjcb_sensor(source_counter)
    for data, value in zip(refs, velocity):
      data.qvel[:] = value
    # No host callback executes in native stepping. CounterSensor is a
    # separately registered device adapter with its declared stage contract.
    for _ in range(15):
      _native_only(monkeypatch, sim.step)
      for data in refs:
        mujoco.mj_step(model, data)
    if integrator == 'RK4':
      # Pinned RK4 does not transition this fixture into sleep. Its adapter
      # still must match the actual source callback state and samples.
      assert refs[0].tree_awake.tolist() == [1]
    else:
      assert refs[0].tree_awake.tolist() == [0]
    np.testing.assert_array_equal(_host(sim._sleep_schedule.tree_awake),
                                   [data.tree_awake for data in refs])
    runtime = next(p for p in sim._native_plugins if p.name == name)
    before = _host(runtime.calls)
    stored = sim.step_sensordata()
    record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
    for data in refs:
      mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
    baseline = np.asarray([d.sensordata.copy() for d in refs])
    np.testing.assert_array_equal(stored[0], baseline[0])
    for data in refs:
      getattr(mujoco, 'mj_sensor'+stage)(model, data)
    result = _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record))
    np.testing.assert_array_equal(_host(result), [d.sensordata for d in refs])
    # engine_sleep.c:735 explicitly treats USER/PLUGIN sensors as awake,
    # regardless of the sleep state of their referenced joint or body.
    np.testing.assert_array_equal(_host(runtime.calls)[0], before[0]+1)
    np.testing.assert_array_equal(_host(result)[0], stored[0]+1)
    checkpoint = sim.snapshot()
    _native_only(monkeypatch, sim.step)
    expected = sim.snapshot()
    _native_only(monkeypatch, lambda: sim.restore(checkpoint))
    _native_only(monkeypatch, sim.step)
    np.testing.assert_array_equal(sim.snapshot()['sensordata'], expected['sensordata'])
    np.testing.assert_array_equal(sim.snapshot()['plugins'][('sensor', name)],
                                   expected['plugins'][('sensor', name)])
  finally:
    mujoco.set_mjcb_sensor(callback)
    default_registry.unregister(name)
