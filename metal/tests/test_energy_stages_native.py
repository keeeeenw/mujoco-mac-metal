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


def _fixture(integrator='Euler', profile='integrated_scalable_v1', enabled=False,
             disableflags=0):
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".001" integrator="{integrator}">
      <flag energy="{'enable' if enabled else 'disable'}"/></option>
    <worldbody><body pos="0 0 .4"><joint name="slider" type="slide"
      axis="0 0 1" stiffness="2" armature=".25"/>
      <geom type="sphere" size=".05" mass="1"/></body></worldbody>
    <tendon><fixed stiffness="3" springlength=".1 .2">
      <joint joint="slider" coef="2"/></fixed></tendon></mujoco>''')
  model.opt.disableflags |= int(disableflags)
  model.tendon_armature[:] = .4
  model.tendon_stiffnesspoly[:] = [-.25, .4]
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


def _computed(sim):
  return np.column_stack((_host(sim.flg_energypos), _host(sim.flg_energyvel)))


def test_prepared_energy_computed_flags_follow_selected_publication_and_restore(monkeypatch):
  _, sim, _ = _fixture()
  np.testing.assert_array_equal(_computed(sim), 0)
  record = _prepare(sim, monkeypatch)
  np.testing.assert_array_equal(_computed(sim), 0)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, record, env_ids=[2, 0]))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, record, env_ids=[1]))
  expected = np.array([[1, 0], [0, 1], [1, 0]], np.int32)
  np.testing.assert_array_equal(_computed(sim), expected)
  owned = sim.flg_energypos
  owned.zero_()
  np.testing.assert_array_equal(_computed(sim), expected)
  snap = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.reset(env_ids=[1]))
  expected[1] = 0
  np.testing.assert_array_equal(_computed(sim), expected)
  _native_only(monkeypatch, lambda: sim.restore(snap, env_ids=[1]))
  np.testing.assert_array_equal(_computed(sim), snap['energy_computed'])
  _native_only(monkeypatch, lambda: sim.copy_environment(1, 2))
  np.testing.assert_array_equal(_computed(sim)[2], snap['energy_computed'][1])
  _native_only(monkeypatch, lambda: sim.restore(snap))
  np.testing.assert_array_equal(_computed(sim), snap['energy_computed'])


@pytest.mark.parametrize('bad', [None, [[1, 0]], [[1, 0, 0]]*3,
                               [[-1, 0]]*3, [[2, 0]]*3, [[.5, 0]]*3,
                               [[np.nan, 0]]*3, [['yes', 'no']]*3])
def test_invalid_computed_flags_restore_is_atomic(bad, monkeypatch):
  _, sim, _ = _fixture()
  record = _prepare(sim, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, record))
  before = sim.snapshot()
  identity = (sim.state.generation, sim._forward_stages.epoch)
  with pytest.raises(ValueError, match='energy_computed'):
    _native_only(monkeypatch, lambda: sim.restore(dict(before, energy_computed=bad),
                                                  env_ids=[1]))
  np.testing.assert_array_equal(_host(sim.energy), before['energy'])
  np.testing.assert_array_equal(_computed(sim), before['energy_computed'])
  np.testing.assert_array_equal(_host(sim.state._qpos), before['device'].qpos)
  assert sim.validate_forward_stage_record(record, 'VEL') is record
  assert (sim.state.generation, sim._forward_stages.epoch) == identity


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
  flags_before = _computed(sim)
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
  np.testing.assert_array_equal(_computed(sim), flags_before)
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
  np.testing.assert_array_equal(_computed(sim), 0)
  np.testing.assert_array_equal(_host(sim.state._qpos), q)
  np.testing.assert_array_equal(_host(sim.state._qvel), v)
  with pytest.raises(ValueError):
    mj_energyPos(sim, record)
  updated = _prepare(sim, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, updated))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, updated))
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)
  np.testing.assert_array_equal(_computed(sim),
      [[d.flg_energypos, d.flg_energyvel] for d in refs])
  assert np.max(np.abs(_host(sim.energy)[:, 0] - before[:, 0])) > 1.


@pytest.mark.parametrize('case', ['free', 'ball', 'hinge', 'spatial', 'flex', 'static'])
@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('enabled', [False, True])
def test_direct_energy_real_model_families(case, profile, enabled, monkeypatch):
  from mujoco_metal import MetalSimulation
  bodies = {
    'free': '<body pos="0 0 1"><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/></body>',
    'ball': '<body pos="0 0 1"><joint type="ball" stiffness="3" armature=".4"/><geom type="box" size=".1 .2 .3" mass="2"/></body>',
    'hinge': '<body pos="0 0 1"><joint type="hinge" axis="0 1 0" stiffness="3" armature=".4"/><geom type="box" pos=".1 0 .05" size=".1 .2 .3" mass="2"/></body>',
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
  elif case in ('ball', 'hinge'):
    model.jnt_stiffnesspoly[:] = [1., .5]
  q = np.tile(model.qpos0, (3, 1))
  v = np.tile(np.linspace(.15, .45, model.nv), (3, 1))
  for row in range(3):
    if case == 'free':
      q[row, :3] += [row*.05, -.04, .12]
      q[row, 3:7] = [np.cos(.1*(row+1)), np.sin(.1*(row+1)), 0, 0]
    elif case == 'ball':
      q[row] = [np.cos(.1*(row+1)), np.sin(.1*(row+1)), 0, 0]
    elif case == 'hinge':
      q[row, 0] = [-.3, .1, .4][row]
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
  np.testing.assert_array_equal(_computed(sim),
      [[d.flg_energypos, d.flg_energyvel] for d in refs])


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
    np.testing.assert_array_equal(_computed(sim),
        [[d.flg_energypos, d.flg_energyvel] for d in refs])
    if step == 3:
      mid = sim.snapshot()
  final = sim.snapshot()
  _native_only(monkeypatch, lambda: sim.restore(mid))
  for _ in range(4):
    _native_only(monkeypatch, sim.step)
  replay = sim.snapshot()
  np.testing.assert_array_equal(replay['energy'], final['energy'])
  np.testing.assert_array_equal(replay['energy_computed'], final['energy_computed'])
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


@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
@pytest.mark.parametrize('flags', [int(mujoco.mjtDisableBit.mjDSBL_GRAVITY),
    int(mujoco.mjtDisableBit.mjDSBL_SPRING),
    int(mujoco.mjtDisableBit.mjDSBL_GRAVITY) | int(mujoco.mjtDisableBit.mjDSBL_SPRING)])
def test_explicit_energy_obeys_gravity_and_spring_disable(flags, profile, monkeypatch):
  _, sim, refs = _fixture(profile=profile, disableflags=flags)
  record = _prepare(sim, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, record))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, record))
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)
  if flags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY) and flags & int(mujoco.mjtDisableBit.mjDSBL_SPRING):
    np.testing.assert_array_equal(_host(sim.energy)[:, 0], 0)


@pytest.mark.parametrize('profile', ['integrated_euler_v1', 'integrated_scalable_v1'])
def test_spatial_pulley_armature_is_in_complete_prepared_energy(profile, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><site name="anchor" pos="0 0 2"/><body pos="0 0 1">
      <joint type="slide" axis="0 0 1" armature=".25"/>
      <geom type="sphere" size=".1" mass="2"/><site name="load"/>
    </body></worldbody><tendon><spatial armature=".4" stiffness="4"
      springlength=".6 .8"><site site="anchor"/><site site="load"/>
      <pulley divisor="2"/><site site="load"/><site site="anchor"/>
    </spatial></tendon></mujoco>''')
  q = np.array([[.1], [.3], [.45]])
  v = np.array([[.25], [-.5], [.75]])
  refs = []
  for pos, vel in zip(q, v):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = pos, vel
    mujoco.mj_forward(model, d)
    mujoco.mj_energyPos(model, d)
    mujoco.mj_energyVel(model, d)
    refs.append(d)
    assert abs(d.ten_velocity[0]) > .3
    assert d.energy[1] > .5*(2+.25)*float(vel[0])**2 + .02
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
  record = _prepare(sim, monkeypatch)
  with monkeypatch.context() as guard:
    _forbid_producers(sim, guard)
    _native_only(guard, lambda: mj_energyPos(sim, record))
    _native_only(guard, lambda: mj_energyVel(sim, record))
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)


def test_explicit_energy_filters_real_failed_and_stale_peer_rows(monkeypatch):
  _, sim, refs = _fixture()
  record = _prepare(sim, monkeypatch)
  sim.state._status[1] = 1
  record.row_valid[2] = False
  _native_only(monkeypatch, lambda: mj_energyPos(sim, record))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, record))
  np.testing.assert_allclose(_host(sim.energy)[0], refs[0].energy, rtol=2e-5, atol=2e-6)
  np.testing.assert_array_equal(_host(sim.energy)[1:], 0)


@pytest.mark.parametrize('mutation', ['qpos', 'qvel', 'replacement', 'reset', 'restore'])
def test_stale_energy_record_rejected_before_lazy_publication(mutation, monkeypatch):
  import torch
  _, sim, _ = _fixture()
  snapshot = sim.snapshot()
  record = _prepare(sim, monkeypatch)
  if mutation == 'qpos':
    sim.state._qpos.add_(.01)
  elif mutation == 'qvel':
    sim.state._qvel.add_(.01)
  elif mutation == 'replacement':
    sim.state._qpos = sim.state._qpos.clone()
  elif mutation == 'reset':
    _native_only(monkeypatch, sim.reset)
  else:
    _native_only(monkeypatch, lambda: sim.restore(snapshot))
  with pytest.raises(ValueError):
    _native_only(monkeypatch, lambda: mj_energyVel(sim, record))
  assert sim._energy is None


def test_manual_energy_restore_into_unallocated_simulation_is_owned(monkeypatch):
  import torch
  _, src, refs = _fixture()
  _, dst, _ = _fixture()
  record = _prepare(src, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(src, record))
  _native_only(monkeypatch, lambda: mj_energyVel(src, record))
  snap = src.snapshot()
  assert dst.energy is None and dst._energy is None
  with torch.inference_mode():
    _native_only(monkeypatch, lambda: dst.restore(snap, env_ids=[2, 0]))
  expected = snap['energy'].copy()
  expected[1] = 0
  np.testing.assert_array_equal(_host(dst.energy), expected)
  assert not torch.is_inference(dst._energy)
  snap['energy'][:] = -9
  np.testing.assert_array_equal(_host(dst.energy), expected)
  _native_only(monkeypatch, lambda: dst.reset(env_ids=[1]))
  updated = _prepare(dst, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyVel(dst, updated))
  np.testing.assert_allclose(_host(dst.energy)[:, 1], [refs[0].energy[1], 0, refs[2].energy[1]],
                             rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('integrator,profile', [
    ('Euler', 'integrated_scalable_v1'), ('RK4', 'integrated_rk4_v1')])
def test_explicit_energy_retains_static_gravity_and_filters_real_sleeping_springs(
    integrator, profile, monkeypatch):
  import copy
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{integrator}"><flag sleep="enable" contact="disable"/></option>
    <worldbody><body name="sleeping" pos="-1 0 1" sleep="init">
      <joint type="slide" axis="0 0 1" stiffness="8" springref=".2"/>
      <geom type="sphere" size=".1" mass="2"/></body>
      <body name="awake" pos="1 0 .5" sleep="allowed">
        <joint type="slide" axis="0 0 1" stiffness="3" armature=".25"/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body pos="0 0 2"><geom type="box" size=".1 .1 .1" mass="3"/></body>
    </worldbody></mujoco>''')
  q = np.array([[0., .1], [0., .2], [0., .3]])
  v = np.array([[0., .2], [0., .3], [0., .4]])
  refs = []
  counterfactual = copy.copy(model)
  counterfactual.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  for pos, vel in zip(q, v):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = pos, vel
    mujoco.mj_forward(model, d)
    mujoco.mj_energyPos(model, d)
    mujoco.mj_energyVel(model, d)
    assert d.tree_asleep[0] >= 0 and d.tree_asleep[1] < 0
    assert d.nbody_awake < model.nbody
    assert d.energy[0] > 70  # sleeping and static masses retain gravity energy
    full = mujoco.MjData(counterfactual)
    full.qpos[:], full.qvel[:] = pos, vel
    mujoco.mj_forward(counterfactual, full)
    mujoco.mj_energyPos(counterfactual, full)
    np.testing.assert_allclose(full.energy[0]-d.energy[0], .16, rtol=0, atol=1e-12)
    refs.append(d)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
  record = _prepare(sim, monkeypatch)
  np.testing.assert_array_equal(_host(sim._sleep_schedule.tree_state),
                                 [d.tree_asleep for d in refs])
  with monkeypatch.context() as guard:
    _forbid_producers(sim, guard)
    _native_only(guard, lambda: mj_energyPos(sim, record))
    _native_only(guard, lambda: mj_energyVel(sim, record))
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('integrator,profile', _METHODS)
@pytest.mark.parametrize('enabled', [False, True])
def test_static_energy_sensors_and_storage_match_zero_dof_source(
    integrator, profile, enabled, monkeypatch):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_forward
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{integrator}"><flag energy="{'enable' if enabled else 'disable'}"/></option>
    <worldbody><body pos="0 0 1"><geom type="sphere" size=".1" mass="2"/></body>
    </worldbody><sensor><e_potential/><e_kinetic/></sensor></mujoco>''')
  ref = mujoco.MjData(model)
  mujoco.mj_forward(model, ref)
  assert model.nv == 0 and ref.energy[0] > 10 and ref.energy[1] == 0
  sim = MetalSimulation(model, 3, profile=profile)
  _native_only(monkeypatch, lambda: mj_forward(sim))
  np.testing.assert_allclose(sim.step_sensordata(), np.tile(ref.sensordata, (3, 1)),
                             rtol=2e-5, atol=2e-6)
  np.testing.assert_allclose(_host(sim.energy), np.tile(ref.energy, (3, 1)),
                             rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('input_name', ['qpos', 'qvel'])
def test_native_energy_reuse_without_mutation_counter_rejects_before_allocation(
    input_name, monkeypatch):
  import torch
  _, sim, _ = _fixture()
  with torch.inference_mode():
    value = torch.tensor([[.1], [.2], [.3]], dtype=torch.float32,
                         device=sim.state._qpos.device)
  if input_name == 'qpos':
    record = _native_only(monkeypatch, lambda: mj_fwdPosition(
        sim, value, skipsensor=True, return_record=True))
    call = lambda: mj_energyPos(sim, record)
  else:
    record = _native_only(monkeypatch, lambda: mj_fwdPosition(
        sim, skipsensor=True, return_record=True))
    _native_only(monkeypatch, lambda: mj_fwdVelocity(
        sim, qvel=value, record=record, skipsensor=True))
    call = lambda: mj_energyVel(sim, record)
  with pytest.raises(ValueError, match='mutation counter'):
    _native_only(monkeypatch, call)
  assert sim._energy is None


@pytest.mark.parametrize('integrator,profile', [
    ('Euler', 'integrated_scalable_v1'), ('RK4', 'integrated_rk4_v1')])
def test_sleeping_and_static_tendon_springs_match_source_filter(integrator, profile, monkeypatch):
  import copy
  from mujoco_metal import MetalSimulation
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="{integrator}"><flag sleep="enable" contact="disable"/></option><worldbody>
      <site name="anchor" pos="0 0 2"/>
      <site name="static_a" pos="0 1 0"/><site name="static_b" pos="0 1 1"/>
      <body pos="-1 0 1" sleep="init"><joint name="sleep" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="2"/><site name="load" pos="1 0 0"/></body>
      <body pos="1 0 .5" sleep="allowed"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody><tendon>
      <fixed stiffness="4" springlength=".3 .4"><joint joint="sleep" coef="1"/></fixed>
      <spatial stiffness="4" springlength=".6 .8"><site site="anchor"/><site site="load"/></spatial>
      <spatial stiffness="2" springlength=".4 .5"><site site="static_a"/><site site="static_b"/></spatial>
    </tendon></mujoco>''')
  q = np.array([[0., .1], [0., .2], [0., .3]])
  v = np.array([[0., .2], [0., .3], [0., .4]])
  refs = []
  counter = copy.copy(model)
  counter.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  for pos, vel in zip(q, v):
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = pos, vel
    mujoco.mj_forward(model, d)
    mujoco.mj_energyPos(model, d)
    mujoco.mj_energyVel(model, d)
    assert d.tree_asleep[0] >= 0 and d.nbody_awake < model.nbody
    other = mujoco.MjData(counter)
    other.qpos[:], other.qvel[:] = pos, vel
    mujoco.mj_forward(counter, other)
    mujoco.mj_energyPos(counter, other)
    np.testing.assert_allclose(other.energy[0]-d.energy[0], .51, rtol=0, atol=1e-12)
    refs.append(d)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile)
  record = _prepare(sim, monkeypatch)
  _native_only(monkeypatch, lambda: mj_energyPos(sim, record))
  _native_only(monkeypatch, lambda: mj_energyVel(sim, record))
  np.testing.assert_allclose(_host(sim.energy), [d.energy for d in refs],
                             rtol=2e-5, atol=2e-6)
