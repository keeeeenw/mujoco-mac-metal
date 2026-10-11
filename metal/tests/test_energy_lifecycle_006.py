# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Stored stage energy follows full simulation state transactions."""
import os
import mujoco
import numpy as np
import pytest

_METHODS = [('Euler', 'integrated_euler_v1'),
            ('Euler', 'integrated_scalable_v1'),
            ('RK4', 'integrated_rk4_v1'),
            ('implicit', 'integrated_implicit_v1'),
            ('implicitfast', 'integrated_implicitfast_v1')]


def _fixture(integrator, profile):
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".001" gravity="0 0 -9.81" integrator="{integrator}">
      <flag energy="enable"/></option>
    <worldbody><geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1"><joint name="slider" type="slide" axis="0 0 1"
        stiffness="2"/><geom type="sphere" size=".05" mass="1"/></body>
    </worldbody><sensor><jointpos joint="slider"/>
      <jointvel joint="slider"/></sensor></mujoco>''')
  qpos = np.array([[.125], [.25], [.375]], np.float32)
  qvel = np.array([[.25], [-.5], [.75]], np.float32)
  refs = []
  for q, v in zip(qpos, qvel):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mujoco.mj_forward(model, data)
    refs.append(data)
  sim = MetalSimulation(model, 3, qpos=qpos, qvel=qvel, profile=profile)
  return model, sim, refs


def _native_only(monkeypatch, call):
  def forbidden(*args, **kwargs):
    raise AssertionError('CPU physics must not execute in native transaction')
  with monkeypatch.context() as guard:
    for name in ('mj_forward', 'mj_forwardSkip', 'mj_step', 'mj_step1',
                 'mj_step2', 'mj_energyPos', 'mj_energyVel',
                 'mj_fwdKinematics', 'mj_fwdPosition', 'mj_fwdVelocity',
                 'mj_fwdActuation', 'mj_fwdAcceleration', 'mj_fwdConstraint',
                 'mj_kinematics', 'mj_comPos', 'mj_comVel', 'mj_collision',
                 'mj_passive', 'mj_rne', 'mj_rnePostConstraint',
                 'mj_sensorPos', 'mj_sensorVel', 'mj_sensorAcc'):
      guard.setattr(mujoco, name, forbidden)
    return call()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('integrator,profile', _METHODS)
def test_copy_stored_energy_selected_rows_and_owned_output(
    integrator, profile, monkeypatch):
  from mujoco_metal.native_api import mj_forward
  _, sim, refs = _fixture(integrator, profile)
  _native_only(monkeypatch, lambda: mj_forward(sim))
  host = lambda x: x.detach().cpu().numpy().copy()
  expected = np.stack([r.energy.copy() for r in refs])
  np.testing.assert_allclose(host(sim.energy), expected, rtol=2e-5, atol=2e-6)
  assert len(np.unique(expected[:, 0])) == 3
  assert len(np.unique(expected[:, 1])) == 3
  before = host(sim.energy)
  exposed = sim.energy
  storage = sim._energy.data_ptr()
  _native_only(monkeypatch, lambda: sim.copy_environment(0, 1))
  after = host(sim.energy)
  np.testing.assert_array_equal(after[1], before[0])
  np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
  np.testing.assert_array_equal(host(exposed), before)
  assert sim._energy.data_ptr() == storage


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('integrator,profile', _METHODS)
def test_snapshot_stored_energy_full_and_selected_restore(
    integrator, profile, monkeypatch):
  from mujoco_metal.native_api import mj_forward
  _, sim, refs = _fixture(integrator, profile)
  _native_only(monkeypatch, lambda: mj_forward(sim))
  host = lambda x: x.detach().cpu().numpy().copy()
  before = host(sim.energy)
  snap = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.reset())
  np.testing.assert_array_equal(host(sim.energy), 0)
  _native_only(monkeypatch, lambda: sim.restore(snap, env_ids=[1]))
  np.testing.assert_array_equal(host(sim.energy)[1], before[1])
  np.testing.assert_array_equal(host(sim.energy)[[0, 2]], 0)
  _native_only(monkeypatch, lambda: sim.restore(snap))
  np.testing.assert_array_equal(host(sim.energy), before)
  # Regenerate and compare against independent source after restoration.
  _native_only(monkeypatch, lambda: mj_forward(sim))
  np.testing.assert_allclose(host(sim.energy), [r.energy for r in refs],
                             rtol=2e-5, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
@pytest.mark.parametrize('bad', [None, [[1., 2.]], [[1., 2., 3.]]*3,
                               [[np.nan, 2.]]*3, [[1., np.inf]]*3,
                               [[1e100, 2.]]*3])
def test_invalid_energy_snapshot_rejected_before_transaction(bad, monkeypatch):
  from mujoco_metal.native_api import mj_forward
  _, sim, _ = _fixture('Euler', 'integrated_scalable_v1')
  record = _native_only(monkeypatch, lambda: mj_forward(sim))['record']
  host = lambda x: x.detach().cpu().numpy().copy()
  before = sim.snapshot()
  energy = host(sim.energy)
  sensors = sim.step_sensordata().copy()
  generation, epoch = sim.state.generation, sim._forward_stages.epoch
  words = host(sim.state._precision.words)
  bad_snapshot = dict(before, energy=bad)
  with pytest.raises(ValueError, match='energy'):
    _native_only(monkeypatch, lambda: sim.restore(bad_snapshot, env_ids=[1]))
  np.testing.assert_array_equal(host(sim.energy), energy)
  np.testing.assert_array_equal(sim.step_sensordata(), sensors)
  np.testing.assert_array_equal(host(sim.state._precision.words), words)
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status'):
    np.testing.assert_array_equal(host(getattr(sim.state, '_'+name)),
                                 getattr(before['device'], name))
  assert (sim.state.generation, sim._forward_stages.epoch) == (generation, epoch)
  assert sim._forward_stages._record is record


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
def test_legacy_energy_snapshot_resets_selected_storage_and_recomputes(monkeypatch):
  from mujoco_metal.native_api import mj_forward
  _, sim, refs = _fixture('Euler', 'integrated_scalable_v1')
  _native_only(monkeypatch, lambda: mj_forward(sim))
  host = lambda x: x.detach().cpu().numpy().copy()
  snap = sim.snapshot()
  expected = host(sim.energy)
  # Before this optional extension, schema1 snapshots carried no energy key.
  old = dict(snap)
  old.pop('energy', None)
  _native_only(monkeypatch, lambda: sim.restore(old, env_ids=[1]))
  np.testing.assert_array_equal(host(sim.energy)[1], 0)
  np.testing.assert_array_equal(host(sim.energy)[[0, 2]], expected[[0, 2]])
  _native_only(monkeypatch, lambda: mj_forward(sim))
  np.testing.assert_allclose(host(sim.energy), [r.energy for r in refs],
                             rtol=2e-5, atol=2e-6)
  _native_only(monkeypatch, lambda: sim.restore(old))
  np.testing.assert_array_equal(host(sim.energy), 0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
def test_disabled_energy_cannot_be_injected_by_snapshot(monkeypatch):
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
  sim = MetalSimulation(model, 3, profile='integrated_scalable_v1')
  assert sim.energy is None
  before = sim.snapshot()
  assert before['energy'] is None
  with pytest.raises(ValueError, match='energy'):
    _native_only(monkeypatch, lambda: sim.restore(
        dict(before, energy=np.zeros((3, 2), np.float32))))
  np.testing.assert_array_equal(sim.state.snapshot().qpos, before['device'].qpos)
  assert sim.energy is None


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')
def test_energy_checkpoint_is_owned_empty_ids_rejected_and_replay(monkeypatch):
  from mujoco_metal.native_api import mj_forward
  _, sim, _ = _fixture('Euler', 'integrated_scalable_v1')
  _native_only(monkeypatch, lambda: mj_forward(sim))
  host = lambda x: x.detach().cpu().numpy().copy()
  start = sim.snapshot()
  saved_energy = start['energy'].copy()
  start['energy'][0] += 1
  np.testing.assert_array_equal(host(sim.energy), saved_energy)
  start['energy'][:] = saved_energy
  generation = sim.state.generation
  # Full simulation restore already rejects empty selections; energy must
  # remain unchanged on that rejected transaction as well.
  with pytest.raises(ValueError, match='at least one world'):
    _native_only(monkeypatch, lambda: sim.restore(start, env_ids=[]))
  assert sim.state.generation == generation
  np.testing.assert_array_equal(host(sim.energy), saved_energy)
  _native_only(monkeypatch, lambda: sim.copy_environment(1, 1))
  np.testing.assert_array_equal(host(sim.energy), saved_energy)
  # Capture real stored preintegration energy and sensors through four steps;
  # a restored prefix must reproduce them without a fresh forward query.
  start = sim.snapshot()
  trajectory = []
  for _ in range(4):
    _native_only(monkeypatch, sim.step)
    trajectory.append((host(sim.state._qpos), host(sim.state._qvel),
                       host(sim.energy), sim.step_sensordata().copy()))
  _native_only(monkeypatch, lambda: sim.restore(start))
  np.testing.assert_array_equal(host(sim.energy), start['energy'])
  for q, v, energy, sensors in trajectory:
    _native_only(monkeypatch, sim.step)
    np.testing.assert_array_equal(host(sim.state._qpos), q)
    np.testing.assert_array_equal(host(sim.state._qvel), v)
    np.testing.assert_array_equal(host(sim.energy), energy)
    np.testing.assert_array_equal(sim.step_sensordata(), sensors)
