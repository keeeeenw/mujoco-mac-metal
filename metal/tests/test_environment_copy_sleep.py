# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Whole-row copies retain sleeping geometry and derived awake lists."""
import copy
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api
from test_awake_simulation_017 import DAMPED_XML
from test_energy_lifecycle_006 import _native_only
from test_sensor_stage_publication import _host

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native complete sleep row copy')]


@pytest.mark.parametrize('source,destination', [(0, 1), (1, 0), (0, 2)])
def test_real_sleep_copy_preserves_retained_pose_words_lists_peers_and_source_trajectory(
    source, destination, monkeypatch):
  model = mujoco.MjModel.from_xml_string(DAMPED_XML.replace(
      '</mujoco>', '<sensor><jointpos joint="slider"/><jointvel joint="slider"/></sensor></mujoco>'))
  pos = np.asarray([[.10000001], [.4], [.8]], np.float32)
  vel = np.asarray([[.04], [2.], [.03]], np.float32)
  sim = MetalSimulation(model, 3, qpos=pos, qvel=vel, profile='integrated_scalable_v1')
  refs = [mujoco.MjData(model) for _ in range(3)]
  for data, qpos, qvel in zip(refs, pos, vel):
    data.qpos[:], data.qvel[:] = qpos, qvel
  for _ in range(15):
    _native_only(monkeypatch, sim.step)
    for data in refs:
      mujoco.mj_step(model, data)
  assert not refs[0].tree_awake.any() and refs[1].tree_awake.any()
  scheduler = sim._sleep_schedule
  before_lists = {name: _host(getattr(scheduler, name)) for name in (
      'body_awake_ids', 'parent_awake_ids', 'dof_awake_ids', 'body_awake_mask', 'awake_counts')}
  outputs = sim._smooth._fk._workspace['outputs']
  before_poses = {name: _host(outputs[name]).reshape(3, model.nbody, width) for name, width in (
      ('body_pos', 3), ('body_pos_low', 3), ('body_pos_tail', 3),
      ('body_quat', 4), ('body_quat_low', 4), ('body_quat_tail', 4))}
  assert np.any(before_poses['body_pos_low'] != 0), 'fixture must exercise retained geometry words'
  _native_only(monkeypatch, lambda: sim.copy_environment(source, destination))
  refs[destination] = copy.copy(refs[source])
  # Immediate prepared consumers must see the copied awake lists. Waiting for
  # a subsequent forward call to repair them would hide incomplete row copying.
  for name, before in before_lists.items():
    expected = before.copy()
    expected[destination] = before[source]
    np.testing.assert_array_equal(_host(getattr(scheduler, name)), expected)
  for name, before in before_poses.items():
    expected = before.copy()
    expected[destination] = before[source]
    np.testing.assert_array_equal(_host(outputs[name]).reshape(expected.shape), expected)
  checkpoint = sim.snapshot()
  for step in range(4):
    force = np.zeros((3, model.nv), np.float32)
    if step >= 2:
      force[destination, 0] = .4
    _native_only(monkeypatch, lambda: sim.step(qfrc_applied=force))
    for world, data in enumerate(refs):
      data.qfrc_applied[:] = force[world]
      mujoco.mj_step(model, data)
      for name in ('qpos', 'qvel', 'qacc'):
        np.testing.assert_allclose(_host(getattr(sim.state, '_'+name))[world], getattr(data, name),
                                   rtol=3e-5, atol=2e-7)
      np.testing.assert_array_equal(_host(scheduler.tree_state)[world], data.tree_asleep)
      np.testing.assert_allclose(sim.step_sensordata()[world], data.sensordata,
                                rtol=3e-5, atol=2e-7)
  expected = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(checkpoint))
  for step in range(4):
    force = np.zeros((3, model.nv), np.float32)
    if step >= 2:
      force[destination, 0] = .4
    _native_only(monkeypatch, lambda: sim.step(qfrc_applied=force))
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status', 'precision_words'):
    np.testing.assert_array_equal(getattr(sim.snapshot()['device'], name), getattr(expected['device'], name))
  np.testing.assert_array_equal(sim.snapshot()['sleep_schedule']['tree_state'],
                                expected['sleep_schedule']['tree_state'])
  np.testing.assert_array_equal(sim.snapshot()['sensordata'], expected['sensordata'])
