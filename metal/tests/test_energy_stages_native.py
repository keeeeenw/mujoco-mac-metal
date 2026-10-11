# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Direct prepared energy APIs: native physics, ownership and transactions."""
import os
import mujoco
import numpy as np
import pytest

from test_energy_lifecycle_006 import _METHODS, _native_only
from mujoco_metal.forward_stages import ForwardStage
from mujoco_metal.native_api import (
    mj_energyPos, mj_energyVel, mj_fwdPosition, mj_fwdVelocity)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native opt-in')]


def _fixture(integrator='Euler', profile='integrated_scalable_v1', enabled=False):
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".001" integrator="{integrator}">
      <flag energy="{'enable' if enabled else 'disable'}"/></option>
    <worldbody><body pos="0 0 .4"><joint name="slider" type="slide"
      axis="0 0 1" stiffness="2" armature=".25"/>
      <geom type="sphere" size=".05" mass="1"/></body></worldbody>
    <tendon><fixed stiffness="3" springlength=".1 .2">
      <joint joint="slider" coef="2"/></fixed></tendon></mujoco>''')
  qpos = np.array([[.05], [.125], [.375]], np.float32)
  qvel = np.array([[.25], [-.5], [.75]], np.float32)
  refs = []
  for q, v in zip(qpos, qvel):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = q, v
    mujoco.mj_forward(model, d)
    # Explicit APIs are independent of automatic energy sampling.
    mujoco.mj_energyPos(model, d)
    mujoco.mj_energyVel(model, d)
    refs.append(d)
  return model, MetalSimulation(model, 3, qpos=qpos, qvel=qvel,
                                 profile=profile), refs


def _host(x):
  return x.detach().cpu().numpy().copy()


def _prepare(sim, monkeypatch):
  record = _native_only(monkeypatch, lambda: mj_fwdPosition(
      sim, skipsensor=True, return_record=True))
  _native_only(monkeypatch, lambda: mj_fwdVelocity(
      sim, record=record, skipsensor=True))
  return record


def _forbid_producers(sim, guard):
  def no_dispatch(*args, **kwargs):
    raise AssertionError('prepared energy must not recompute physics stages')
  for owner, names in ((sim, ('prepare_forward_position', 'prepare_forward_velocity',
                              'assembled_system', 'step')),
                       (sim._smooth, ('run_device',))):
    for name in names:
      guard.setattr(owner, name, no_dispatch)


@pytest.mark.parametrize('integrator,profile', _METHODS)
@pytest.mark.parametrize('enabled', [False, True])
def test_direct_energy_matches_source_without_recomputing_prepared_stages(
    integrator, profile, enabled, monkeypatch):
  _, sim, refs = _fixture(integrator, profile, enabled)
  record = _prepare(sim, monkeypatch)
  before_q, before_v = _host(sim.state._qpos), _host(sim.state._qvel)
  epoch, generation = sim._forward_stages.epoch, sim.state.generation
  borrowed = record.values[ForwardStage.POS].get('energy_pos')
  borrowed_before = None if borrowed is None else _host(borrowed)
  with monkeypatch.context() as guard:
    _forbid_producers(sim, guard)
    pos = _native_only(guard, lambda: mj_energyPos(sim, record))
    old_kinetic = _host(sim.energy)[:, 1]
    vel = _native_only(guard, lambda: mj_energyVel(sim, record))
  expected = np.stack([d.energy for d in refs])
  np.testing.assert_allclose(_host(pos), expected[:, 0], rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(_host(vel), expected[:, 1], rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(_host(sim.energy), expected, rtol=2e-5, atol=2e-6)
  assert np.ptp(expected[:, 0]) > .1 and np.ptp(expected[:, 1]) > .1
  if borrowed is not None:
    np.testing.assert_array_equal(_host(borrowed), borrowed_before)
  pos.fill_(-8)
  vel.fill_(-9)
  np.testing.assert_allclose(_host(sim.energy), expected, rtol=2e-5, atol=2e-6)
  np.testing.assert_array_equal(_host(sim.state._qpos), before_q)
  np.testing.assert_array_equal(_host(sim.state._qvel), before_v)
  assert sim.validate_forward_stage_record(record, 'VEL') is record
  assert (sim._forward_stages.epoch, sim.state.generation) == (epoch, generation)


def test_selected_energy_checkpoint_reset_copy_and_exact_replay(monkeypatch):
  _, sim, refs = _fixture()
  record = _prepare(sim, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, record, env_ids=[2, 0]))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, record, env_ids=[1]))
  before = _host(sim.energy)
  np.testing.assert_allclose(before[[0, 2], 0], [refs[i].energy[0] for i in (0, 2)],
                             rtol=2e-5, atol=2e-6)
  np.testing.assert_array_equal(before[1, 0], 0)
  np.testing.assert_array_equal(before[[0, 2], 1], 0)
  snap = sim.snapshot()
  assert snap['energy_manual'] is True
  _native_only(monkeypatch, lambda: sim.copy_environment(0, 1))
  np.testing.assert_array_equal(_host(sim.energy)[1], before[0])
  _native_only(monkeypatch, lambda: sim.restore(snap))
  np.testing.assert_array_equal(_host(sim.energy), before)
  _native_only(monkeypatch, lambda: sim.reset(env_ids=[1]))
  np.testing.assert_array_equal(_host(sim.energy)[1], 0)
  _native_only(monkeypatch, lambda: sim.restore(snap, env_ids=[1]))
  np.testing.assert_array_equal(_host(sim.energy), before)
  for _ in range(3):
    np.testing.assert_array_equal(_host(_native_only(monkeypatch, sim.step)), 0)
  final = sim.snapshot()
  # The pinned full forward clears both samples when ENERGY is disabled and
  # no potential-energy sensor has evaluated them in the position sweep.
  for d in refs:
    for _ in range(3):
      mujoco.mj_step(sim._mjmodel, d)
  np.testing.assert_array_equal(final['energy'], [d.energy for d in refs])
  _native_only(monkeypatch, lambda: sim.restore(snap))
  for _ in range(3):
    _native_only(monkeypatch, sim.step)
  again = sim.snapshot()
  for name in ('qpos', 'qvel', 'qacc', 'time', 'status', 'precision_words'):
    np.testing.assert_array_equal(getattr(final['device'], name),
                                   getattr(again['device'], name))
  np.testing.assert_array_equal(final['energy'], again['energy'])


@pytest.mark.parametrize('existing', [False, True])
def test_native_post_publication_failure_is_atomic(existing, monkeypatch):
  import mujoco_metal.energy as implementation
  _, sim, _ = _fixture()
  record = _prepare(sim, monkeypatch)
  if existing:
    _native_only(monkeypatch, lambda: mj_energyPos(sim, record))
  old = sim._energy
  before = None if old is None else _host(old)
  scratch = {k: _host(v) for k, v in sim._smooth._workspace.items()
             if k in ('mass_sparse_product', 'mass_matvec_dims', 'energy_velocity_dims')}
  publish = implementation._publish_energy
  calls = []
  def fail(*args):
    publish(*args)
    calls.append(True)
    raise RuntimeError('native energy publication fault')
  monkeypatch.setattr(implementation, '_publish_energy', fail)
  with pytest.raises(RuntimeError, match='publication fault'):
    _native_only(monkeypatch, lambda: mj_energyVel(sim, record))
  assert calls == [True] and sim._energy is old
  if old is not None:
    np.testing.assert_array_equal(_host(old), before)
  for k, value in scratch.items():
    np.testing.assert_array_equal(_host(sim._smooth._workspace[k]), value)
  assert sim.validate_forward_stage_record(record, 'VEL') is record


@pytest.mark.parametrize('enabled', [False, True])
def test_lifecycle_rebuild_preserves_samples_and_uses_new_constants(enabled, monkeypatch):
  model, sim, _ = _fixture(enabled=enabled)
  record = _prepare(sim, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, record))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, record))
  before = _host(sim.energy)
  q, v = _host(sim.state._qpos), _host(sim.state._qvel)
  import copy
  replacement = copy.copy(model)
  replacement.opt.gravity[:] = [0, 0, -3.5]
  replacement.jnt_stiffness[:] = 7.
  replacement.tendon_stiffness[:] = 5.
  refs = []
  for pos, vel in zip(q, v):
    d = mujoco.MjData(replacement)
    d.qpos[:], d.qvel[:] = pos, vel
    mujoco.mj_forward(replacement, d)
    mujoco.mj_energyPos(replacement, d)
    mujoco.mj_energyVel(replacement, d)
    refs.append(d)
  _native_only(monkeypatch, lambda: sim.apply_lifecycle(replacement))
  np.testing.assert_array_equal(_host(sim.energy), before)
  np.testing.assert_array_equal(_host(sim.state._qpos), q)
  np.testing.assert_array_equal(_host(sim.state._qvel), v)
  with pytest.raises(ValueError):
    mj_energyPos(sim, record)
  updated = _prepare(sim, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, updated))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, updated))
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)
  assert np.max(np.abs(_host(sim.energy)[:, 0] - before[:, 0])) > 1.


@pytest.mark.parametrize('case', ['free', 'ball', 'spatial', 'flex', 'static'])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('enabled', [False, True])
def test_direct_energy_real_model_families(case, profile, enabled, monkeypatch):
  from mujoco_metal import MetalSimulation
  bodies = {
    'free': '<body pos="0 0 1"><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/></body>',
    'ball': '<body pos="0 0 1"><joint type="ball" stiffness="3" armature=".4"/><geom type="box" size=".1 .2 .3" mass="2"/></body>',
    'spatial': '<site name="anchor" pos="0 0 2"/><body pos="0 0 1"><joint type="slide" axis="0 0 1"/><geom type="sphere" size=".1" mass="2"/><site name="load"/></body>',
    'flex': '<flexcomp name="rope" type="grid" count="3 1 1" spacing=".2 .2 .2" dim="1" pos="0 0 1" radius=".01" mass="1"><edge stiffness="20" damping="0"/><contact contype="0" conaffinity="0"/></flexcomp>',
    'static': '<geom type="sphere" size=".1" pos="0 0 1" mass="2"/>',
  }
  tendon = ('<tendon><spatial stiffness="4" springlength=".6 .8"><site site="anchor"/><site site="load"/></spatial></tendon>'
            if case == 'spatial' else '')
  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+bodies[case]+
                                       '</worldbody>'+tendon+'</mujoco>')
  if enabled:
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
  if case == 'free':
    model.jnt_stiffness[:] = 3.
    model.jnt_stiffnesspoly[:] = [1., .5]
  elif case == 'ball':
    model.jnt_stiffnesspoly[:] = [1., .5]
  q = np.tile(model.qpos0, (3, 1))
  v = np.tile(np.linspace(.15, .45, model.nv), (3, 1))
  for row in range(3):
    if case == 'free':
      q[row, :3] += [row*.05, -.04, .12]
      q[row, 3:7] = [np.cos(.1*(row+1)), np.sin(.1*(row+1)), 0, 0]
    elif case == 'ball':
      q[row] = [np.cos(.1*(row+1)), np.sin(.1*(row+1)), 0, 0]
    elif case == 'spatial':
      q[row, 0] = [.1, .3, .45][row]
    elif case == 'flex':
      q[row, 3] += .01*(row+1)
    v[row] *= row+1
  refs = []
  for pos, vel in zip(q, v):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = pos, vel
    mujoco.mj_forward(model, d)
    mujoco.mj_energyPos(model, d)
    mujoco.mj_energyVel(model, d)
    refs.append(d)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
  record = _prepare(sim, monkeypatch)
  with monkeypatch.context() as guard:
    _forbid_producers(sim, guard)
    _native_only(guard, lambda: mj_energyPos(sim, record))
    _native_only(guard, lambda: mj_energyVel(sim, record))
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)
  if case != 'static':
    assert min(d.energy[1] for d in refs) > 1e-4


@pytest.mark.parametrize('sensors', ['potential', 'kinetic', 'both'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('disable_sensors', [False, True])
def test_energy_sensors_share_source_storage_and_current_velocity(
    sensors, enabled, profile, disable_sensors, monkeypatch):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_forward
  entries = ('<e_potential name="potential"/>' if sensors in ('potential', 'both') else '')
  entries += ('<e_kinetic name="kinetic"/>' if sensors in ('kinetic', 'both') else '')
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option><flag energy="{'enable' if enabled else 'disable'}"/></option>
    <worldbody><body pos="0 0 .4"><joint name="j" type="slide"
      axis="0 0 1" stiffness="2" armature=".25"/>
      <geom type="sphere" size=".05" mass="1"/></body></worldbody>
    <sensor>{entries}</sensor></mujoco>''')
  if disable_sensors:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
  q = np.array([[.05], [.125], [.375]], np.float32)
  v = np.array([[.25], [-.5], [.75]], np.float32)
  refs = []
  for pos, vel in zip(q, v):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = pos, vel
    mujoco.mj_forward(model, d)
    refs.append(d)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
  _native_only(monkeypatch, lambda: mj_forward(sim))
  np.testing.assert_allclose(sim.step_sensordata(), [d.sensordata for d in refs],
                             rtol=2e-5, atol=2e-6)
  assert sim.energy is not None
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('integrator,profile', _METHODS)
@pytest.mark.parametrize('enabled', [False, True])
def test_timed_energy_sensors_samples_storage_and_checkpoint_replay(
    integrator, profile, enabled, monkeypatch):
  from mujoco_metal import MetalSimulation
  spec = mujoco.MjSpec.from_string(f'''<mujoco>
    <option timestep=".001" integrator="{integrator}">
      <flag energy="{'enable' if enabled else 'disable'}"/></option>
    <worldbody><body pos="0 0 .4"><joint name="j" type="slide"
      axis="0 0 1" stiffness="2" armature=".25"/>
      <geom type="sphere" size=".05" mass="1"/></body></worldbody>
    <sensor><e_potential name="potential" cutoff="3"/>
      <e_kinetic name="kinetic" cutoff=".2"/></sensor></mujoco>''')
  for name in ('potential', 'kinetic'):
    sensor = spec.sensor(name)
    sensor.nsample, sensor.interp, sensor.delay = 8, 1, .002
    sensor.interval = [.002, 0]
  model = spec.compile()
  q = np.array([[.05], [.125], [.375]], np.float32)
  v = np.array([[.25], [-.5], [.75]], np.float32)
  refs = []
  for pos, vel in zip(q, v):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = pos, vel
    refs.append(d)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
  mid = None
  for step in range(8):
    for d in refs:
      mujoco.mj_step(model, d)
    np.testing.assert_array_equal(_host(_native_only(monkeypatch, sim.step)), 0)
    np.testing.assert_allclose(sim.step_sensordata(), [d.sensordata for d in refs],
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                               rtol=2e-5, atol=2e-6)
    if step == 3:
      mid = sim.snapshot()
  final = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(mid))
  for _ in range(4):
    _native_only(monkeypatch, sim.step)
  replay = sim.snapshot()
  np.testing.assert_array_equal(replay['energy'], final['energy'])
  for name in ('qpos', 'qvel', 'qacc', 'time', 'precision_words'):
    np.testing.assert_array_equal(getattr(replay['device'], name),
                                   getattr(final['device'], name))
  np.testing.assert_array_equal(replay['native_state']['history'],
                                 final['native_state']['history'])
  np.testing.assert_array_equal(replay['sensordata'], final['sensordata'])


@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('failure', [False, True])
def test_current_energy_sensor_query_preserves_stored_step_samples(profile, failure, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string('''<mujoco><option><flag energy="enable"/></option>
    <worldbody><body pos="0 0 .4"><joint type="slide" axis="0 0 1"
      stiffness="2" armature=".25"/><geom type="sphere" size=".05" mass="1"/></body>
    </worldbody><sensor><e_potential/><e_kinetic/></sensor></mujoco>''')
  sim = MetalSimulation(model, 3, qpos=np.array([[.05], [.125], [.375]]),
      qvel=np.array([[.25], [-.5], [.75]]), profile=profile)
  _native_only(monkeypatch, lambda: sim.step(3))
  before = sim.snapshot()
  q, v = _host(sim.state._qpos), _host(sim.state._qvel)
  refs = []
  for pos, vel in zip(q, v):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = pos, vel
    mujoco.mj_forward(model, d)
    refs.append(d)
  if failure:
    original = sim._sensor_values_impl
    calls = []
    def fail_after_query():
      original()
      calls.append(True)
      raise RuntimeError('completed current energy query fault')
    monkeypatch.setattr(sim, '_sensor_values_impl', fail_after_query)
    with pytest.raises(RuntimeError, match='energy query fault'):
      _native_only(monkeypatch, sim.sensor_values)
    assert calls == [True]
  else:
    result = _native_only(monkeypatch, sim.sensor_values)
    np.testing.assert_allclose(_host(result), [d.sensordata for d in refs], rtol=2e-5, atol=2e-6)
  after = sim.snapshot()
  np.testing.assert_array_equal(after['energy'], before['energy'])
  np.testing.assert_array_equal(after['sensordata'], before['sensordata'])
  np.testing.assert_array_equal(after['native_state']['history'], before['native_state']['history'])
