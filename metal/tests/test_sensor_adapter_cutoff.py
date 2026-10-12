# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Device USER output cutoff follows the actual pinned stage callback rules."""
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
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native adapter cutoff composition')]


class VectorSensor(CounterSensor):
  def __init__(self, name, values):
    super().__init__(name)
    self.values = tuple(values)

  def run_device(self, state, **kwargs):
    self.calls.add_(1)
    values = state._torch.as_tensor(self.values, dtype=state._torch.float32, device=self.device)
    return (values[None, :] + .1*state._qpos[:, :1]).contiguous()


@pytest.mark.parametrize('integrator,profile', _PROFILES)
@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('datatype', ['real', 'positive', 'axis', 'quaternion'])
@pytest.mark.parametrize('cutoff', [0., 1.])
def test_actual_adapter_publication_applies_source_cutoff_after_callback(
    integrator, profile, stage, datatype, cutoff, monkeypatch):
  name = 'cutoff_'+stage.lower()
  values = [-3., 2., .25] if datatype != 'quaternion' else [-3., 2., .25, .5]
  default_registry.register(VectorSensor(name, values))
  callback = mujoco.get_mjcb_sensor()
  try:
    xml = f'''<mujoco><option gravity="0 0 0" integrator="{integrator}"/>
      <worldbody><body><joint name="j" type="slide"/><geom size=".1" mass="1"/>
      </body></worldbody><sensor><user name="{name}" dim="{len(values)}"
      needstage="{stage.lower()}" datatype="{datatype}" cutoff="{cutoff}"/>
      </sensor></mujoco>'''
    if datatype in ('axis', 'quaternion') and cutoff > 0:
      # The XML compiler rejects this combination. The C data API still
      # exposes the compiled cutoff array, and its runtime rule ignores
      # cutoff for these datatypes. Test both contracts separately.
      with pytest.raises(ValueError, match='cutoff applied to axis or quaternion'):
        mujoco.MjModel.from_xml_string(xml)
      model = mujoco.MjModel.from_xml_string(xml.replace(f'cutoff="{cutoff}"', 'cutoff="0"'))
      model.sensor_cutoff[0] = cutoff
    else:
      model = mujoco.MjModel.from_xml_string(xml)
    position = np.asarray([[-.2], [.1], [.4]], np.float32)
    sim = MetalSimulation(model, 3, qpos=position, profile=profile)
    refs = [mujoco.MjData(model) for _ in range(3)]
    expected_stage = int(getattr(mujoco.mjtStage, 'mjSTAGE_'+stage.upper()))
    def source_callback(m, data, needstage):
      if int(needstage) == expected_stage:
        data.sensordata[:] = np.asarray(values) + .1*data.qpos[0]
    mujoco.set_mjcb_sensor(source_callback)
    for data, qpos in zip(refs, position):
      data.qpos[:] = qpos
      mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
      data.sensordata[:] = 20.
    record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
    sim._sensordata = sim.state._torch.full((3, model.nsensordata), 20., device=sim.state._device)
    sim._raw_sensordata = sim._sensordata.clone()
    for world in (2, 0):
      getattr(mujoco, 'mj_sensor'+stage)(model, refs[world])
    result = _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(
        sim, record, env_ids=[2, 0]))
    np.testing.assert_allclose(_host(result), [data.sensordata for data in refs],
                               atol=2e-7, rtol=2e-7)
    adapter = next(plugin for plugin in sim._native_plugins if plugin.name == name)
    np.testing.assert_array_equal(_host(adapter.calls), [1, 0, 1])
    # Complete stepping invokes the same declared stage and post-callback
    # cutoff; the source callback itself does not preclip or normalize.
    _native_only(monkeypatch, sim.step)
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_allclose(sim.step_sensordata(), [data.sensordata for data in refs],
                               atol=2e-7, rtol=2e-7)
  finally:
    mujoco.set_mjcb_sensor(callback)
    default_registry.unregister(name, PluginType.SENSOR)
