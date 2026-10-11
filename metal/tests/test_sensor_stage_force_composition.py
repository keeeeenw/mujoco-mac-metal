# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Standalone stage publication with real equality/contact reaction forces."""
import os
import mujoco
import numpy as np
import pytest

from mujoco_metal import MetalSimulation, native_api as api
from mujoco_metal.capacity import model_component_capacity_limits
from test_energy_lifecycle_006 import _native_only
from test_sensor_force_016 import _XML
from test_sensor_integrated_stages import _PROFILES
from test_sensor_stage_publication import _host

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native sensor force composition')]


@pytest.mark.parametrize('integrator,profile', _PROFILES)
@pytest.mark.parametrize('stage', ['Pos', 'Vel', 'Acc'])
@pytest.mark.parametrize('islands_disabled', [False, True])
@pytest.mark.parametrize('solver', ['CG', 'Newton'])
@pytest.mark.parametrize('jacobian', ['dense', 'sparse'])
def test_standalone_stage_reuses_nonzero_equality_contact_forces_and_motion(
    integrator, profile, stage, islands_disabled, solver, jacobian, monkeypatch):
  xml = _XML.replace('integrator="Euler"', f'integrator="{integrator}"')
  xml = xml.replace('<actuator>', '<equality><connect body1="swing" '
      'body2="ball" anchor="0.1 0 0.4"/></equality><actuator>')
  xml = xml.replace('</sensor>', '<jointvel joint="h"/><gyro site="gauge"/>'
      '<subtreelinvel body="swing"/><subtreeangmom body="swing"/></sensor>')
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.solver = getattr(mujoco.mjtSolver, 'mjSOL_'+solver.upper())
  model.opt.jacobian = getattr(mujoco.mjtJacobian, 'mjJAC_'+jacobian.upper())
  if islands_disabled:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
  q = np.tile(model.qpos0, (3, 1)).astype(np.float32)
  q[:, 0], q[:, 3] = [.4, -.25, .1], [.08, .4, .085]
  v = np.asarray([np.linspace(-.5, .7, model.nv)*(w+1) for w in range(3)], np.float32)
  ctrl = np.asarray([[.5], [-.2], [.7]], np.float32)
  active = np.asarray([[1], [0], [1]], np.int32)
  force = np.zeros((3, model.nv), np.float32)
  force[:, 0] = [.3, -.2, .4]
  wrench = np.zeros((3, model.nbody, 6), np.float32)
  wrench[:, -1] = [.4, .1, -.2, .03, .1, -.07]
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile=profile,
                        limits=model_component_capacity_limits(model, 3))
  api.mj_setState(sim, dict(ctrl=ctrl, eq_active=active,
                           qfrc_applied=force, xfrc_applied=wrench))
  refs = [mujoco.MjData(model) for _ in range(3)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:], data.ctrl[:] = q[world], v[world], ctrl[world]
    data.eq_active[:], data.qfrc_applied[:], data.xfrc_applied[:] = active[world], force[world], wrench[world]
    mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_NONE, 1)
  assert refs[0].ncon > 0 and refs[1].ncon == 0
  assert refs[0].ne > 0 and refs[1].ne == 0
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1
  record = _native_only(monkeypatch, lambda: api.mj_forward(sim, skipsensor=True))['record']
  if integrator == 'implicitfast':
    assert sim._component_mass_enabled and sim._midpoint is not None
    # Eligibility consumes the step's actual constraint assembly, not an
    # independently published inspection record before stepping.
    midpoint_run = sim._midpoint.run_device
    def check_midpoint(*args, **kwargs):
      np.testing.assert_array_equal(_host(kwargs['eligible_mask']), [[False], [True], [False]])
      return midpoint_run(*args, **kwargs)
    monkeypatch.setattr(sim._midpoint, 'run_device', check_midpoint)
  np.testing.assert_array_equal(_host(record.values[5]['status']), 0)
  np.testing.assert_allclose(_host(record.values[5]['qacc']), [d.qacc for d in refs],
                             rtol=2e-4, atol=2e-4)
  sentinel = np.arange(3*model.nsensordata, dtype=np.float32).reshape(3, -1)+20
  sim._sensordata = sim.state._torch.as_tensor(sentinel, device=sim.state._device).clone()
  sim._raw_sensordata = sim._sensordata.clone()
  for data, values in zip(refs, sentinel):
    data.sensordata[:] = values
  for world in (2, 0):
    getattr(mujoco, 'mj_sensor'+stage)(model, refs[world])
  expected = np.stack([d.sensordata.copy() for d in refs])
  if stage == 'Acc':
    for kind in (mujoco.mjtSensor.mjSENS_FORCE, mujoco.mjtSensor.mjSENS_TORQUE):
      index = np.flatnonzero(model.sensor_type == int(kind))[0]
      first, width = model.sensor_adr[index], model.sensor_dim[index]
      assert np.linalg.norm(expected[0, first:first+width]) > .01
  before = sim.snapshot()
  def forbidden(*args, **kwargs):
    raise AssertionError('standalone sensor redispatched forward physics')
  with monkeypatch.context() as guard:
    for name in ('prepare_forward_position', 'prepare_forward_velocity',
                 'prepare_forward_actuation', 'prepare_forward_acceleration',
                 'prepare_forward_constraint', '_acceleration'):
      guard.setattr(sim, name, forbidden)
    result = _native_only(guard, lambda: getattr(api, 'mj_sensor'+stage)(sim, record, env_ids=[2, 0]))
  np.testing.assert_allclose(_host(result), expected, rtol=2e-4, atol=2e-4)
  np.testing.assert_array_equal(sim.snapshot()['native_state']['history'], before['native_state']['history'])
  assert sim.validate_forward_stage_record(record, 'CONSTRAINT') is record
  saved = sim.snapshot()
  trajectory = []
  for _ in range(3):
    _native_only(monkeypatch, sim.step)
    for data in refs:
      mujoco.mj_step(model, data)
    state = sim.snapshot()
    np.testing.assert_allclose(state['device'].qpos, [d.qpos for d in refs], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(state['device'].qvel, [d.qvel for d in refs], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(state['sensordata'], [d.sensordata for d in refs], rtol=2e-3, atol=2e-3)
    trajectory.append(state)
  _native_only(monkeypatch, lambda: sim.restore(saved))
  for expected_state in trajectory:
    _native_only(monkeypatch, sim.step)
    actual = sim.snapshot()
    for name in ('qpos', 'qvel', 'qacc', 'precision_words'):
      np.testing.assert_array_equal(getattr(actual['device'], name), getattr(expected_state['device'], name))
    np.testing.assert_array_equal(actual['sensordata'], expected_state['sensordata'])
