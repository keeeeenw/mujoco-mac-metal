# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned sleep/wake trajectories through the actual native stepping pipeline."""

import os

import mujoco
import numpy as np
import pytest


XML = '''<mujoco><option timestep=".002" gravity="0 0 0">
  <flag sleep="enable" contact="disable"/></option>
  <worldbody>
    <body name="a" pos="-1 0 0" sleep="allowed"><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body>
    <body name="b" pos="1 0 0" sleep="allowed"><joint type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".1" mass="2"/></body>
  </worldbody></mujoco>'''


def _references():
  model = mujoco.MjModel.from_xml_string(XML)
  references = [mujoco.MjData(model) for _ in range(2)]
  references[0].qvel[:] = [.3, 0]
  references[1].qvel[:] = [0, -.2]
  return model, references


def test_pinned_fixture_sleeps_quiet_trees_then_wakes_selected_force_rows():
  model, references = _references()
  for _ in range(15):
    for data in references:
      mujoco.mj_step(model, data)
  assert references[0].tree_asleep[1] >= 0
  assert references[1].tree_asleep[0] >= 0
  references[0].qfrc_applied[1] = .7
  references[1].qfrc_applied[0] = -.5
  for data in references:
    mujoco.mj_step(model, data)
    assert np.all(data.tree_asleep < 0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_stepping_sleep_force_wake_and_restore_matches_pinned_trajectory():
  from mujoco_metal import MetalSimulation

  model, references = _references()
  initial = np.stack([data.qvel for data in references]).astype(np.float32)
  sim = MetalSimulation(model, 2, qvel=initial, profile="integrated_scalable_v1")
  assert sim._sleep_schedule.dof_length.dtype == sim._state._torch.float32
  np.testing.assert_array_equal(
      sim._sleep_schedule.dof_length.cpu().numpy(),
      np.asarray(model.dof_length, dtype=np.float32))
  sleep_witness = False
  for step in range(30):
    applied = np.zeros((2, model.nv), dtype=np.float32)
    if step >= 15:
      applied[0, 1] = .7
      applied[1, 0] = -.5
    sim.step(qfrc_applied=applied)
    for world, data in enumerate(references):
      data.qfrc_applied[:] = applied[world]
      mujoco.mj_step(model, data)
      for field in ("qpos", "qvel", "qacc"):
        np.testing.assert_allclose(
            getattr(sim.state, "_" + field)[world].cpu().numpy(),
            np.asarray(getattr(data, field)), rtol=3e-5, atol=2e-7,
            err_msg=f"{field}, step {step}, world {world}")
      np.testing.assert_array_equal(
          sim._sleep_schedule.tree_state[world].cpu().numpy(), data.tree_asleep,
          err_msg=f"sleep state, step {step}, world {world}")
      np.testing.assert_array_equal(
          sim._sleep_schedule.awake_lists()["dof_count"][world].cpu().numpy(),
          data.nv_awake)
    if step == 14:
      sleep_witness = True
      assert references[0].tree_asleep[1] >= 0
      assert references[1].tree_asleep[0] >= 0
  assert sleep_witness
  checkpoint = sim.snapshot()
  sim.step(5)
  expected = {name: getattr(sim.state, "_" + name).cpu().numpy().copy()
              for name in ("qpos", "qvel", "qacc", "time", "status")}
  expected_sleep = sim._sleep_schedule.tree_state.cpu().numpy().copy()
  sim.restore(checkpoint)
  sim.step(5)
  for name, value in expected.items():
    np.testing.assert_array_equal(getattr(sim.state, "_" + name).cpu().numpy(), value)
  np.testing.assert_array_equal(sim._sleep_schedule.tree_state.cpu().numpy(), expected_sleep)


DAMPED_XML = '''<mujoco><option timestep=".1" gravity="0 0 0" sleep_tolerance=".05">
  <flag sleep="enable" contact="disable"/></option><worldbody>
  <body sleep="allowed"><joint name="slider" type="slide" axis="1 0 0" damping="2.5"/>
    <geom size=".1" mass="1"/></body></worldbody></mujoco>'''


def test_pinned_sleep_countdown_checks_preintegration_damped_velocity():
  model = mujoco.MjModel.from_xml_string(DAMPED_XML)
  data = mujoco.MjData(model)
  data.qvel[0] = .06
  mujoco.mj_step(model, data)
  np.testing.assert_allclose(data.qvel, [.048], atol=1e-14)
  np.testing.assert_array_equal(data.tree_asleep, [-11])
  mujoco.mj_step(model, data)
  np.testing.assert_array_equal(data.tree_asleep, [-10])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_sleep_cutoff_and_pose_freeze_match_damped_pinned_worlds():
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(DAMPED_XML)
  references = [mujoco.MjData(model) for _ in range(2)]
  references[0].qvel[:] = .06
  references[1].qvel[:] = .04
  initial = np.stack([data.qvel for data in references]).astype(np.float32)
  sim = MetalSimulation(model, 2, qvel=initial, profile="integrated_scalable_v1")
  for step in range(15):
    sim.step()
    for world, data in enumerate(references):
      mujoco.mj_step(model, data)
      for field in ("qpos", "qvel", "qacc"):
        np.testing.assert_allclose(getattr(sim.state, "_" + field)[world].cpu().numpy(),
            getattr(data, field), rtol=3e-5, atol=1e-7,
            err_msg=f"{field}, cutoff step {step}, world {world}")
      np.testing.assert_array_equal(sim._sleep_schedule.tree_state[world].cpu().numpy(),
          data.tree_asleep, err_msg=f"cutoff step {step}, world {world}")
  assert all(not data.tree_awake.any() for data in references)
  for field in ("qvel", "qacc"):
    np.testing.assert_array_equal(getattr(sim.state, "_" + field).cpu().numpy(), 0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_failed_native_world_preserves_sleep_counters_and_lists():
  import torch
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(DAMPED_XML)
  sim = MetalSimulation(model, 2, qvel=np.asarray([[.06], [.04]], np.float32),
                        profile="integrated_scalable_v1")
  saved = sim._snapshot_sleep_device()
  physical = {name: getattr(sim.state, "_" + name).clone()
              for name in ("qpos", "qvel", "qacc", "time")}
  force = torch.tensor([[0.], [float("nan")]], dtype=torch.float32, device="mps")
  sim.step(qfrc_applied=force)
  assert sim.state._status[0].item() == 0
  assert sim.state._status[1].item() != 0
  for name, before in saved.items():
    np.testing.assert_array_equal(getattr(sim._sleep_schedule, name)[1].cpu().numpy(),
                                  before[1].cpu().numpy(), err_msg=name)
  for name, before in physical.items():
    np.testing.assert_array_equal(getattr(sim.state, "_" + name)[1].cpu().numpy(),
                                  before[1].cpu().numpy(), err_msg=name)
  # A sticky failed world stays frozen even after its held force is cleared.
  sim.step(qfrc_applied=torch.zeros_like(force))
  for name, before in saved.items():
    np.testing.assert_array_equal(getattr(sim._sleep_schedule, name)[1].cpu().numpy(),
                                  before[1].cpu().numpy(), err_msg=name)


SENSOR_SLEEP_XML = DAMPED_XML.replace(
    '<joint type="slide"', '<joint name="slider" type="slide"').replace(
    '</body>', '<site name="probe"/></body>').replace(
    '</mujoco>', '<sensor><jointvel joint="slider"/>'
    '<framelinvel objtype="site" objname="probe"/>'
    '<accelerometer site="probe"/></sensor></mujoco>')


def test_pinned_sleep_transition_refreshes_velocity_and_acceleration_sensors():
  model = mujoco.MjModel.from_xml_string(SENSOR_SLEEP_XML)
  data = mujoco.MjData(model)
  data.qvel[:] = .04
  for _ in range(9):
    mujoco.mj_step(model, data)
  assert data.tree_asleep[0] < 0
  assert np.linalg.norm(data.sensordata) > 0
  mujoco.mj_step(model, data)
  assert data.tree_asleep[0] >= 0
  np.testing.assert_array_equal(data.sensordata, 0)
  np.testing.assert_array_equal(data.qfrc_damper, 0)


def test_callback_stage_view_preserves_live_state_and_uses_stage_inputs():
  from mujoco_metal.simulation import _plugin_stage_state
  class State:
    def __init__(self):
      self._qpos, self._qvel, self._qacc = [np.array([[n]], float) for n in (1, 2, 3)]
      self._act, self._time = np.array([[4.]]), np.array([5.])
      self.metadata = object()
    @property
    def qvel(self):
      return self._qvel.copy()
  state = State()
  original = state._qpos, state._qvel, state._qacc, state._act, state._time
  qpos, qvel, qacc, act, time = [np.array([[n]], float) for n in (6, 7, 8, 9, 10)]
  stage = _plugin_stage_state(state, qpos, qvel, qacc=qacc, act=act, time=time)
  assert stage.metadata is state.metadata
  for actual, expected in zip((stage._qpos, stage._qvel, stage._qacc, stage._act, stage._time),
                             (qpos, qvel, qacc, act, time)):
    assert actual is expected
  stage.qvel.fill(100)
  np.testing.assert_array_equal(stage._qvel, 7)
  for actual, expected in zip((state._qpos, state._qvel, state._qacc, state._act, state._time), original):
    assert actual is expected


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("already_sleeping_tree", [False, True])
def test_native_sleep_transition_refreshes_stored_sensor_stages(already_sleeping_tree):
  from mujoco_metal import MetalSimulation
  xml = SENSOR_SLEEP_XML
  if already_sleeping_tree:
    xml = xml.replace('</worldbody>', '<body sleep="init" pos="2 0 0">'
                      '<joint type="slide"/><geom size=".1" mass="1"/>'
                      '</body></worldbody>')
  model = mujoco.MjModel.from_xml_string(xml)
  references = [mujoco.MjData(model) for _ in range(2)]
  initial = np.zeros((2, model.nv), np.float32)
  initial[:, 0] = [.04, .08]
  for world, data in enumerate(references):
    data.qvel[:] = initial[world]
  sim = MetalSimulation(model, 2, qvel=initial, profile="integrated_scalable_v1")
  for step in range(15):
    sim.step()
    np.testing.assert_array_equal(sim.state._status.cpu().numpy(), 0,
                                  err_msg=f"step {step} failed unexpectedly")
    for data in references:
      mujoco.mj_step(model, data)
    for name in ("qpos", "qvel", "qacc"):
      np.testing.assert_allclose(
          getattr(sim.state, "_" + name).cpu().numpy(),
          np.stack([getattr(d, name) for d in references]),
          atol=2e-6, rtol=2e-5, err_msg=f"{name}, step {step}")
    np.testing.assert_array_equal(
        sim._sleep_schedule.tree_state.cpu().numpy(),
        np.stack([d.tree_asleep for d in references]), err_msg=f"step {step}")
    np.testing.assert_allclose(sim.step_sensordata(),
                               np.stack([d.sensordata for d in references]),
                               atol=2e-6, rtol=2e-5, err_msg=f"step {step}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_sleep_mocap_and_sensor_stages_share_internal_pose_abi():
  from mujoco_metal import MetalSimulation
  xml = SENSOR_SLEEP_XML.replace(
      '</worldbody>', '<body name="prescribed" mocap="true" pos="2 1 .5">'
      '<geom size=".1" contype="0" conaffinity="0"/></body></worldbody>').replace(
      '</sensor>', '<framepos objtype="body" objname="prescribed"/></sensor>')
  model = mujoco.MjModel.from_xml_string(xml)
  references = [mujoco.MjData(model) for _ in range(2)]
  initial = np.asarray([[.04], [.08]], np.float32)
  for world, data in enumerate(references):
    data.qvel[:] = initial[world]
  sim = MetalSimulation(model, 2, qvel=initial, profile="integrated_scalable_v1")
  for step in range(15):
    sim.step()
    np.testing.assert_array_equal(sim.state._status.cpu().numpy(), 0)
    for data in references:
      mujoco.mj_step(model, data)
    np.testing.assert_allclose(sim.step_sensordata(),
                               np.stack([d.sensordata for d in references]),
                               atol=2e-6, rtol=2e-5, err_msg=f"step {step}")
    np.testing.assert_array_equal(sim._sleep_schedule.tree_state.cpu().numpy(),
                                  np.stack([d.tree_asleep for d in references]))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_sleep_refresh_gates_stateful_plugins_by_world_and_sensor_stage():
  from mujoco_metal import MetalSimulation
  from mujoco_metal.extensions import NativePlugin, PluginType, default_registry
  class StatefulTestPlugin(NativePlugin):
    def init(self, model, batch_size=1, device=None):
      super().init(model, batch_size, device)
      import torch
      self.calls = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
      self.stage_error = torch.zeros_like(self.calls)
    def run_device(self, state, sensor_dim=1, **kwargs):
      import torch
      self.calls.add_(1)
      if self.plugin_type == PluginType.SENSOR:
        return self.calls.float()[:, None].expand(-1, sensor_dim).contiguous()
      self.stage_error += (state.qvel != kwargs['qvel']).any(dim=1).to(torch.int32)
      return torch.zeros((self.batch_size, self.model.nv), device=self.device)
    def snapshot(self):
      return {'calls': self.calls.cpu().numpy().copy(),
              'stage_error': self.stage_error.cpu().numpy().copy()}
    def device_snapshot(self):
      return {'calls': self.calls.clone(), 'stage_error': self.stage_error.clone()}
    def device_snapshot_bytes(self):
      return (self.calls.numel() * self.calls.element_size()
              + self.stage_error.numel() * self.stage_error.element_size())
    def restore_device(self, value):
      self.calls.copy_(value['calls'])
      self.stage_error.copy_(value['stage_error'])
    def restore_masked(self, value, mask):
      import torch
      self.calls.copy_(torch.where(mask, self.calls, value['calls']))
      self.stage_error.copy_(torch.where(mask, self.stage_error, value['stage_error']))
  xml = SENSOR_SLEEP_XML.replace('</sensor>',
      '<user name="sleep_pos" dim="1" needstage="pos"/>'
      '<user name="sleep_vel" dim="1" needstage="vel"/>'
      '<user name="sleep_acc" dim="1" needstage="acc"/></sensor>')
  model = mujoco.MjModel.from_xml_string(xml)
  plugins = [StatefulTestPlugin(name, PluginType.SENSOR)
             for name in ('sleep_pos', 'sleep_vel', 'sleep_acc')]
  plugins.append(StatefulTestPlugin('sleep_force', PluginType.FORCE))
  for plugin in plugins:
    default_registry.register(plugin)
  try:
    sim = MetalSimulation(model, 2, qvel=np.asarray([[.04], [.08]], np.float32),
                          profile="integrated_scalable_v1")
    transitions = np.zeros(2, dtype=np.int32)
    previous = np.asarray([[-11], [-11]], np.int32)
    for step in range(15):
      sim.step()
      current = sim._sleep_schedule.tree_state.cpu().numpy()
      transitions += ((previous < 0) & (current >= 0)).any(axis=1)
      previous = current.copy()
      runtime = {p.name: p for p in sim._native_plugins}
      for name in ('sleep_vel', 'sleep_acc', 'sleep_force'):
        np.testing.assert_array_equal(runtime[name].calls.cpu().numpy(),
                                      (step + 1) + transitions, err_msg=name)
      np.testing.assert_array_equal(runtime['sleep_pos'].calls.cpu().numpy(), step + 1)
      np.testing.assert_array_equal(runtime['sleep_force'].stage_error.cpu().numpy(), 0)
  finally:
    for plugin in plugins:
      default_registry.unregister(plugin.name, plugin.plugin_type)
