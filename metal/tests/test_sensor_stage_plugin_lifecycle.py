# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Device extension state belongs to the selected standalone sensor stage."""
import os
import mujoco
import numpy as np
import pytest
from mujoco_metal import MetalSimulation, native_api as api
from mujoco_metal.extensions import NativePlugin, PluginType, default_registry
from test_energy_lifecycle_006 import _native_only
from test_sensor_stage_publication import _host

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native sensor extension lifecycle')]


class CounterSensor(NativePlugin):
  def __init__(self, name):
    super().__init__(name, PluginType.SENSOR)

  def init(self, model, batch_size=1, device=None):
    super().init(model, batch_size, device)
    import torch
    self.calls = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

  def run_device(self, state, **kwargs):
    # Deliberately eager: the evaluator must enforce its selected-world mask.
    self.calls.add_(1)
    return self.calls.to(dtype=state._torch.float32)[:, None].contiguous()

  def device_snapshot(self):
    return self.calls.clone()

  def device_snapshot_bytes(self):
    return self.calls.numel()*self.calls.element_size()

  def restore_device(self, value):
    self.calls.copy_(value)

  def restore_masked(self, value, mask):
    import torch
    self.calls.copy_(torch.where(mask, self.calls, value))

  def reset_masked(self, mask):
    self.calls.masked_fill_(mask, 0)

  def reset(self, env_ids=None):
    if env_ids is None:
      self.calls.zero_()
    else:
      self.calls[np.asarray(env_ids).tolist()] = 0

  def snapshot(self):
    return _host(self.calls)

  def restore(self, value, env_ids=None):
    import torch
    tensor = torch.as_tensor(value, dtype=self.calls.dtype, device=self.calls.device)
    if env_ids is None:
      self.calls.copy_(tensor)
    else:
      ids = np.asarray(env_ids).tolist()
      self.calls[ids] = tensor[ids]


@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('fault', [False, True])
def test_selected_stage_extension_publication_and_late_restore_failure_are_transactional(
    stage, fault, monkeypatch):
  names = ['stage_pos', 'stage_vel', 'stage_acc']
  for name in names:
    default_registry.register(CounterSensor(name))
  try:
    sensors = ''.join(f'<user name="{name}" dim="1" needstage="{need}"/>'
                      for name, need in zip(names, ('pos', 'vel', 'acc')))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body>
      <joint name="j" type="slide"/><geom size=".1" mass="1"/></body>
      </worldbody><sensor>{sensors}</sensor></mujoco>''')
    sim = MetalSimulation(model, 3, profile='integrated_scalable_v1')
    record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
    sim._sensordata = sim.state._torch.full((3, 3), 20., device=sim.state._device)
    sim._raw_sensordata = sim._sensordata.clone()
    runtimes = {p.name: p for p in sim._native_plugins}
    target = runtimes[names[('Pos', 'Vel', 'Acc').index(stage)]]
    before = sim.snapshot()
    if fault:
      original, calls = target.restore_device, []
      def fail_on_commit(value):
        original(value)
        calls.append(True)
        if len(calls) == 2:
          raise RuntimeError('late sensor extension commit fault')
      monkeypatch.setattr(target, 'restore_device', fail_on_commit)
      with pytest.raises(RuntimeError, match='late sensor extension commit fault'):
        _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record, env_ids=[2, 0]))
      assert len(calls) >= 3
      np.testing.assert_array_equal(sim.step_sensordata(), before['sensordata'])
      for runtime in runtimes.values():
        np.testing.assert_array_equal(_host(runtime.calls), 0)
    else:
      _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, record, env_ids=[2, 0]))
      for name, runtime in runtimes.items():
        expected = [1, 0, 1] if runtime is target else [0, 0, 0]
        np.testing.assert_array_equal(_host(runtime.calls), expected)
      expected = np.full((3, 3), 20., np.float32)
      expected[[0, 2], names.index(target.name)] = 1
      np.testing.assert_array_equal(sim.step_sensordata(), expected)
      saved = sim.snapshot()
      _native_only(monkeypatch, lambda: sim.reset(env_ids=[1]))
      _native_only(monkeypatch, lambda: sim.restore(saved, env_ids=[1]))
      with pytest.raises(ValueError, match='no forward-stage record'):
        _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, env_ids=[1]))
      _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))
      _native_only(monkeypatch, lambda: getattr(api, 'mj_sensor'+stage)(sim, env_ids=[1]))
      np.testing.assert_array_equal(_host(target.calls), [1, 1, 1])
    # A sensor transaction never runs plugins belonging to other stages.
    if fault:
      assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
  finally:
    for name in names:
      default_registry.unregister(name, PluginType.SENSOR)
