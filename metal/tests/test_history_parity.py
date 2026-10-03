# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compiled history storage and pinned interpolation/recording semantics."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.history import lower_history


def _model(interp=1, delay=.003, interval=0, phase=0):
  spec = mujoco.MjSpec.from_string('''<mujoco><option gravity="0 0 0" timestep=".002"/>
    <worldbody><body><joint name="j" type="slide" axis="1 0 0"/>
    <geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
    </body></worldbody><actuator><motor name="motor" joint="j"/></actuator>
    <sensor><jointpos name="position" joint="j"/></sensor></mujoco>''')
  motor, sensor = spec.actuator("motor"), spec.sensor("position")
  motor.nsample, motor.interp, motor.delay = 8, interp, delay
  sensor.nsample, sensor.interp, sensor.delay = 8, interp, delay
  sensor.interval = [interval, phase]
  return spec.compile()


def test_history_layout_uses_compiled_addresses_and_exact_reset_values():
  model = _model()
  meta, parameters, reset = lower_history(model)
  assert model.nhistory == 36
  np.testing.assert_array_equal(meta[:, 1], [model.actuator_historyadr[0], model.sensor_historyadr[0]])
  np.testing.assert_array_equal(meta[:, 2], [8, 8])
  np.testing.assert_array_equal(reset, mujoco.MjData(model).history.astype(np.float32))
  np.testing.assert_allclose(parameters[:, 0], [.003, .003])
  assert reset[1] == reset[19] == 7


def test_history_rejects_uncompiled_storage_mutation_and_overlap():
  model = _model()
  model.sensor_history[0, 0] = 100
  with pytest.raises(ValueError, match="recompiling"):
    lower_history(model)
  model = _model()
  model.sensor_historyadr[0] = 1
  with pytest.raises(ValueError, match="overlap"):
    lower_history(model)


gpu = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("interp", [0, 1, 2])
def test_device_history_read_matches_pinned_nonuniform_interpolation(interp):
  import torch
  from mujoco_metal.history import DeviceHistory
  model = _model(interp)
  data = mujoco.MjData(model)
  times = np.array([-.016, -.014, -.011, -.007, -.004, -.001, .003, .008])
  values = np.array([.2, -.7, .4, 1.3, -.2, .9, -.4, .1])
  mujoco.mj_initCtrlHistory(model, data, 0, times, values[:, None])
  mujoco.mj_initSensorHistory(model, data, 0, times, values[:, None], -.002)
  native = DeviceHistory(model, batch_size=2)
  history = torch.as_tensor(np.tile(data.history, (2, 1)).astype(np.float32), device="mps")
  initial = history.clone()
  # Before oldest, exact sample, interior brackets and after newest, including
  # the endpoint-zero slopes of cubic interpolation.
  for query in [-.030, -.011, -.0085, -.003, .002, .007, .03]:
    time = torch.tensor([query, query+.0007], dtype=torch.float32, device="mps")
    raw = torch.full((2, 1), -9.0, device="mps")
    controls = native.samples(raw, history, time, kind=0).cpu().numpy()
    sensors = native.samples(raw, history, time, kind=1, force_read=True).cpu().numpy()
    for w, stamp in enumerate(time.cpu().numpy()):
      want = mujoco.mj_readCtrl(model, data, 0, float(stamp), interp)
      output = np.zeros(1)
      ptr = mujoco.mj_readSensor(model, data, 0, float(stamp), output, interp)
      np.testing.assert_allclose(controls[w], want, atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(sensors[w], ptr if ptr is not None else output,
                                 atol=5e-6, rtol=5e-6)
  torch.testing.assert_close(history, initial, rtol=0, atol=0)


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("interval,phase,delay", [(0, 0, 0), (0, 0, .003), (.006, 0, 0), (.006, -.002, .003)])
def test_device_history_record_interval_delay_out_of_order_and_failed_neighbors(interval, phase, delay):
  import torch
  from mujoco_metal.history import DeviceHistory
  model = _model(1, delay, interval, phase)
  native, data = DeviceHistory(model, batch_size=2), mujoco.MjData(model)
  history = torch.as_tensor(np.tile(data.history, (2, 1)).astype(np.float32), device="mps")
  untouched = history[1].clone()
  for step, stamp in enumerate([0, .002, .004, .006, .008, .010, .012, .0045, .014, .016, -.03]):
    stamp = float(np.float32(stamp))
    data.time, data.qpos[0], data.qvel[0], data.ctrl[0] = stamp, .03*step, 0, -.2+.1*step
    raw = torch.tensor([[data.qpos[0]], [1.0]], dtype=torch.float32, device="mps")
    controls = torch.tensor([[data.ctrl[0]], [1.0]], dtype=torch.float32, device="mps")
    time = torch.tensor([stamp, stamp], dtype=torch.float32, device="mps")
    sensor_sample = native.samples(raw, history, time, kind=1).cpu().numpy()[0]
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(sensor_sample, data.sensordata, atol=5e-6, rtol=5e-6)
    native.record(controls, raw, history, time, torch.tensor([True, False], device="mps"))
    np.testing.assert_allclose(history.cpu().numpy()[0], data.history, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(history[1], untouched, rtol=0, atol=0)

@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("interp,delay,interval,phase", [
    (0, .004, 0, 0), (1, .003, .006, -.002), (2, .003, .006, 0),
    (1, 0, .006, 0), (1, 0, 0, 0)])
def test_integrated_canonical_histories_and_lifecycle(interp, delay, interval, phase):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_setState, StateSpec
  model = _model(interp, delay, interval, phase)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  references = [mujoco.MjData(model), mujoco.MjData(model)]
  for step in range(18):
    controls = np.array([[.1*np.sin(step)], [-.15*np.cos(step)]], dtype=np.float32)
    # The native public time contract is float32. Use the identical timestamp
    # for the pinned double CPU stage so interval decisions test the kernel,
    # rather than comparing distinct rounded input clocks.
    times = sim.state._time.cpu().numpy()
    for w, data in enumerate(references):
      data.time = float(times[w])
      data.ctrl[:] = controls[w]
    sim.step(ctrl=controls)
    for w, data in enumerate(references):
      mujoco.mj_step(model, data)
      np.testing.assert_allclose(sim.step_sensordata()[w], data.sensordata,
                                 atol=3e-5, rtol=3e-5)
      np.testing.assert_allclose(sim.state._history.cpu().numpy()[w], data.history,
                                 atol=3e-5, rtol=3e-5)
      np.testing.assert_allclose(sim.state._qpos.cpu().numpy()[w], data.qpos, atol=3e-5, rtol=3e-5)
  checkpoint = sim.snapshot()
  sim.step(ctrl=np.ones((2, 1), np.float32))
  expected = sim.snapshot()
  sim.restore(checkpoint)
  sim.step(ctrl=np.ones((2, 1), np.float32))
  np.testing.assert_array_equal(sim.state._history.cpu().numpy(), expected["native_state"]["history"])
  np.testing.assert_array_equal(sim.step_sensordata(), expected["sensordata"])
  sim.reset(env_ids=[1])
  np.testing.assert_array_equal(sim.state._history.cpu().numpy()[1], mujoco.MjData(model).history.astype(np.float32))
  # Arbitrary public state replacement is authoritative, no shadow ring survives.
  history = torch.as_tensor(checkpoint["native_state"]["history"], device="mps")
  mj_setState(sim, {"history": history})
  torch.testing.assert_close(sim.state._history, history, rtol=0, atol=0)


def test_pinned_noise_metadata_does_not_add_stochastic_sensor_samples():
  from mujoco_metal.sensors import lower_sensors
  model = _model(delay=0)
  data = mujoco.MjData(model)
  data.qpos[0] = .2
  mujoco.mj_forward(model, data)
  expected = data.sensordata.copy()
  model.sensor_noise[0] = 100
  lower_sensors(model)
  for _ in range(5):
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.sensordata, expected)

@pytest.mark.gpu
@gpu
def test_history_public_validation_is_atomic_and_empty_kinds_are_safe():
  import torch
  from mujoco_metal.history import DeviceHistory
  model = _model()
  program = DeviceHistory(model)
  history = torch.as_tensor(program.reset_template[None].copy(), device="mps")
  before = history.clone()
  time = torch.zeros(1, device="mps")
  raw = torch.ones((1, 1), device="mps")
  for success in [torch.ones(1, device="mps"), torch.ones(2, dtype=torch.bool, device="mps")]:
    with pytest.raises(ValueError, match="success mask"):
      program.record(raw, raw, history, time, success)
    torch.testing.assert_close(history, before, rtol=0, atol=0)
  with pytest.raises(ValueError, match="raw data"):
    program.samples(np.ones((1, 1)), history, time, kind=0)
  with pytest.raises(ValueError, match="batch size"):
    DeviceHistory(model, batch_size=0)
  spec = mujoco.MjSpec.from_string('''<mujoco><worldbody><body><joint name="j"/>
    <geom type="sphere" size=".1"/></body></worldbody>
    <actuator><motor name="m" joint="j"/></actuator></mujoco>''')
  motor = spec.actuator("m")
  motor.nsample, motor.delay = 4, .002
  model = spec.compile()
  program = DeviceHistory(model)
  history = torch.as_tensor(program.reset_template[None].copy(), device="mps")
  empty = torch.zeros((1, 0), device="mps")
  torch.testing.assert_close(program.samples(empty, history, time, kind=1), empty)

@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("disabled,delay", [(False, .003), (True, .003), (True, 0)])
def test_sensor_raw_compute_mask_follows_interval_and_disabled_stage(disabled, delay):
  import torch
  from mujoco_metal.history import DeviceHistory
  model = _model(delay=delay, interval=.006, phase=-.002)
  if disabled:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  program = DeviceHistory(model, batch_size=2)
  history = torch.as_tensor(np.tile(program.reset_template, (2, 1)), device="mps")
  actual = program.sensor_compute_mask(history, torch.tensor([0, .0041], device="mps"))
  assert actual.cpu().numpy().tolist() == [[False], [not disabled or delay>0]]


def test_pinned_delayed_raw_sample_is_recorded_even_when_sensor_stage_disabled():
  model = _model(delay=.003)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  data = mujoco.MjData(model)
  data.qpos[0] = .2
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(data.sensordata, 0)
  adr = int(model.sensor_historyadr[0])
  cursor = int(data.history[adr+1])
  assert data.history[adr+2+8+cursor] == pytest.approx(.2)

@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("delay,interval", [(0, .006), (.003, 0), (.003, .006)])
def test_disabled_sensor_stage_retains_sample_but_records_pinned_delayed_raw(delay, interval):
  from mujoco_metal import MetalSimulation
  model = _model(delay=delay, interval=interval, phase=-.002 if interval else 0)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  data = mujoco.MjData(model)
  data.qpos[0], data.qvel[0] = .2, .1
  sim = MetalSimulation(model, qpos=data.qpos[None].astype(np.float32),
                        qvel=data.qvel[None].astype(np.float32), profile="integrated_euler_v1")
  for _ in range(9):
    data.time = float(sim.state._time.cpu().numpy()[0])
    mujoco.mj_step(model, data)
    sim.step()
    np.testing.assert_array_equal(sim.step_sensordata()[0], data.sensordata)
    np.testing.assert_allclose(sim.state._history.cpu().numpy()[0], data.history,
                               atol=5e-6, rtol=5e-6)
    before = sim.state._history.cpu().numpy().copy()
    # sensor_values is a caller-owned device tensor; the oracle comparison
    # is an explicit test-only readback, outside the native simulation path.
    queried = sim.sensor_values()
    assert queried.device.type == "mps"
    np.testing.assert_array_equal(queried[0].detach().cpu().numpy(), data.sensordata)
    np.testing.assert_array_equal(sim.state._history.cpu().numpy(), before)
