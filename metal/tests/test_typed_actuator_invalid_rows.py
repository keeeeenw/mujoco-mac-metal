# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Typed extension failures do not enter physical BADQACC recovery."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api
from mujoco_metal.extensions import default_registry, PluginType
from test_native_actuator_plugin_protocol_019 import _TransactionalStepPlugin, _transaction_model
from test_sensor_integrated_stages import _PROFILES
from test_sensor_stage_publication import _host
from test_energy_lifecycle_006 import _native_only

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native typed payload failure isolation')]


class FaultPayload(_TransactionalStepPlugin):
  def __init__(self, name, channel, value):
    super().__init__(name)
    self.channel, self.value, self.enabled = channel, value, False

  def run_actuator_device(self, state, **kwargs):
    output = super().run_actuator_device(state, **kwargs)
    if self.enabled:
      target = output.force if self.channel == 'force' else output.activation_derivative
      target[1].fill_(self.value)
    return output


@pytest.mark.parametrize('integrator,profile', _PROFILES)
@pytest.mark.parametrize('channel', ['force', 'derivative'])
@pytest.mark.parametrize('invalid', ['nan', 'inf'])
@pytest.mark.parametrize('route', ['step', 'split', 'forward'])
def test_nonfinite_typed_payload_preserves_failed_world_and_healthy_source_motion(
    integrator, profile, channel, invalid, route, monkeypatch):
  name = 'typed_payload_'+channel
  default_registry.register_factory(name, PluginType.ACTUATOR,
      lambda: FaultPayload(name, channel, float(invalid)))
  try:
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002" gravity="0 0 -9.81" integrator="{integrator}"/>
      <worldbody><body><joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size=".04" fromto="0 0 0 .3 0 0" mass="1"/>
      <site name="imu" pos=".2 0 0"/></body></worldbody>
      <actuator><general joint="j" dyntype="integrator" gainprm="1"
      biastype="none" actdim="1"/></actuator><sensor><jointpos joint="j"/>
      <accelerometer site="imu"/></sensor></mujoco>''')
    q = np.asarray([[.1], [-.2], [.3]], np.float32)
    v = np.asarray([[.3], [-.1], [.2]], np.float32)
    sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
    refs = [mujoco.MjData(model) for _ in range(3)]
    # Independent CPU equivalence of the healthy typed callback: gear=1
    # projects its .05 force into applied torque; .2 adds to act_dot.
    for data, position, velocity in zip(refs, q, v):
      data.qpos[:], data.qvel[:] = position, velocity
      data.ctrl[:] = .2
      data.qfrc_applied[:] = .05
    _native_only(monkeypatch, lambda: sim.step(2))
    for _ in range(2):
      for data in refs:
        mujoco.mj_step(model, data)
    before = sim.snapshot()
    plugin = sim._native_actuator_plugins[0]
    before_ticks = _host(plugin.ticks)
    plugin.enabled = True
    if route == 'forward':
      result = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))
      status = result['constraint']['status']
      assert np.isfinite(_host(result['constraint']['qacc'])).all()
      _native_only(monkeypatch, lambda: api.mj_sensorAcc(sim, result['record']))
      np.testing.assert_array_equal(sim.step_sensordata()[1], before['sensordata'][1])
      np.testing.assert_array_equal(_host(plugin.ticks), before_ticks)
      for data in (refs[0], refs[2]):
        mujoco.mj_forward(model, data)
    else:
      if route == 'split':
        _native_only(monkeypatch, lambda: api.mj_step1(sim))
        status = _native_only(monkeypatch, lambda: api.mj_step2(sim))
      else:
        status = _native_only(monkeypatch, sim.step)
      np.testing.assert_array_equal(_host(plugin.ticks), before_ticks+[1, 0, 1])
      for data in (refs[0], refs[2]):
        if route == 'split':
          mujoco.mj_step1(model, data)
          mujoco.mj_step2(model, data)
        else:
          mujoco.mj_step(model, data)
    np.testing.assert_array_equal(_host(status), [0, 1, 0])
    after = sim.snapshot()
    for field in ('qpos', 'qvel', 'qacc', 'act', 'time', 'precision_words',
                  'warning_number', 'warning_lastinfo'):
      np.testing.assert_array_equal(getattr(after['device'], field)[1],
                                     getattr(before['device'], field)[1])
    np.testing.assert_array_equal(after['native_state']['qacc_warmstart'][1],
                                   before['native_state']['qacc_warmstart'][1])
    for world in (0, 2):
      if route != 'forward':
        for field in ('qpos', 'qvel', 'qacc', 'act', 'time'):
          np.testing.assert_allclose(getattr(after['device'], field)[world],
              getattr(refs[world], field), rtol=2e-5, atol=2e-5)
      if route == 'forward':
        np.testing.assert_allclose(after['sensordata'][world, 1:], refs[world].sensordata[1:],
                                   rtol=2e-5, atol=2e-5)
    assert not after['device'].warning_number[1].any(), 'extension failure must not become BADQACC'
    plugin.enabled = False
    _native_only(monkeypatch, lambda: sim.reset(env_ids=[1]))
    expected_ticks = before_ticks.copy() if route == 'forward' else before_ticks+[1, 0, 1]
    expected_ticks[1] = 0
    np.testing.assert_array_equal(_host(plugin.ticks), expected_ticks)
    _native_only(monkeypatch, sim.step)
    np.testing.assert_array_equal(_host(plugin.ticks), expected_ticks+1)
  finally:
    default_registry.unregister(name, PluginType.ACTUATOR)


@pytest.mark.parametrize('autoreset', [False, True])
def test_finite_force_with_genuinely_bad_acceleration_keeps_source_check_behavior(autoreset, monkeypatch):
  name = 'typed_finite_bad_acceleration'
  default_registry.register_factory(name, PluginType.ACTUATOR,
      lambda: FaultPayload(name, 'force', 1e11))
  try:
    model = _transaction_model()
    if not autoreset:
      model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    sim = MetalSimulation(model, 2, qpos=np.asarray([[.1], [-.2]], np.float32),
        qvel=np.asarray([[.3], [-.1]], np.float32), profile='integrated_euler_v1')
    before = sim.snapshot()
    sim._native_actuator_plugins[0].enabled = True
    _native_only(monkeypatch, sim.step)
    after = sim.snapshot()
    warning = int(mujoco.mjtWarning.mjWARN_BADQACC)
    assert after['device'].warning_number[1, warning] == (1 if autoreset else 2)
    np.testing.assert_array_equal(after['device'].warning_number[0], 0)
    expected_epoch = before['device'].row_reset_epoch.copy()
    if autoreset:
      expected_epoch[1] += 1
    np.testing.assert_array_equal(after['device'].row_reset_epoch, expected_epoch)
  finally:
    default_registry.unregister(name, PluginType.ACTUATOR)
