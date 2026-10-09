# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Full forward sensors retain source stage ordering across integrated profiles."""
import os
import mujoco
import numpy as np
import pytest
from test_sensor_spatial_016 import _XML as SPATIAL, _states
from test_sensor_remaining_016 import _XML as LIMITS

_PROFILES = [('Euler', 'integrated_euler_v1'),
             ('Euler', 'integrated_scalable_v1'),
             ('RK4', 'integrated_rk4_v1'),
             ('implicitfast', 'integrated_implicitfast_v1'),
             ('implicit', 'integrated_implicit_v1')]

def _model(family, integrator):
  xml = SPATIAL if family == 'spatial' else LIMITS
  return mujoco.MjModel.from_xml_string(xml.replace('integrator="Euler"', f'integrator="{integrator}"'))

@pytest.mark.parametrize('family', ['spatial', 'limits'])
@pytest.mark.parametrize('integrator,profile', _PROFILES)
def test_integrated_spatial_and_force_sensor_admission(family, integrator, profile):
  from mujoco_metal.stepping import validate_stepping_profile
  from mujoco_metal.capacity import model_component_capacity_limits
  m = _model(family, integrator)
  validate_stepping_profile(m, profile=profile,
                   limits=model_component_capacity_limits(m, 2))

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native sensor stage composition')
@pytest.mark.parametrize('family', ['spatial', 'limits'])
@pytest.mark.parametrize('integrator,profile', _PROFILES)
def test_sensor_current_stored_trajectory_and_restore_across_integrators(family, integrator, profile):
  import torch
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.capacity import model_component_capacity_limits
  m = _model(family, integrator)
  if family == 'spatial':
    q, v, _ = _states(m, batch=2)
    q[0, 2] = .08  # actual touch/contact response from the first forward
    q[1, 2] = .4   # inactive touch world, different ray/distance result
    ctrl = np.empty((2, 0), np.float32)
  else:
    q = np.array([[1.1, .1], [0, 1.2], [0, -.1]])
    v = np.array([[-.5, .7], [.3, -.4], [.2, .1]])
    ctrl = np.array([[.5, .3], [-.2, .8], [.1, -.2]], np.float32)
  q, v = q.astype(np.float32), v.astype(np.float32)
  sim = MetalSimulation(m, len(q), qpos=q, qvel=v, profile=profile,
                        limits=model_component_capacity_limits(m, len(q)))
  from mujoco_metal.native_api import mj_setState
  mj_setState(sim, {"ctrl": ctrl})
  refs = [mujoco.MjData(m) for _ in range(len(q))]
  for w, ref in enumerate(refs):
    ref.qpos[:] = q[w]; ref.qvel[:] = v[w]; ref.ctrl[:] = ctrl[w]
    mujoco.mj_forward(m, ref)
  host = lambda value: value.cpu().numpy().copy()
  initial = sim.snapshot()
  got = host(sim.sensor_values())
  np.testing.assert_allclose(got, [r.sensordata for r in refs], rtol=2e-4, atol=2e-4)
  if family == 'spatial':
    assert got[0, m.sensor_adr[0]] > 1
    assert got[1, m.sensor_adr[0]] == 0
    assert np.all(got[:, m.sensor_adr[2]] >= 0)
  else:
    assert abs(got[1, m.sensor_adr[1]]) > 1
    assert abs(got[0, m.sensor_adr[2]]) > 1
    assert got[0, m.sensor_adr[1]] == got[1, m.sensor_adr[2]] == 0
    assert got[2, m.sensor_adr[1]] == got[2, m.sensor_adr[2]] == 0
    assert np.any(abs(got[:, 0]) > .1)
  # Queries may use full ACC/CCD workspaces but cannot overwrite caller-owned
  # results, represented physical state, or the stored pre-integration sample.
  words = sim.state._precision.words.clone()
  stored = sim.step_sensordata() if sim._sensordata is not None else None
  again = sim.sensor_values()
  torch.testing.assert_close(again, torch.tensor(got, device='mps'), rtol=0, atol=0)
  torch.testing.assert_close(sim.state._precision.words, words, rtol=0, atol=0)
  if stored is None:
    assert sim._sensordata is None
  else:
    np.testing.assert_array_equal(sim.step_sensordata(), stored)
  trajectory = []
  for _ in range(8):
    np.testing.assert_array_equal(host(sim.step()), 0)
    for ref in refs: mujoco.mj_step(m, ref)
    for field in ('qpos', 'qvel'):
      np.testing.assert_allclose(host(getattr(sim.state, field)),
                                 [getattr(ref, field) for ref in refs], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(sim.step_sensordata(), [r.sensordata for r in refs],
                               rtol=2e-3, atol=2e-3)
    trajectory.append((host(sim.state.qpos), host(sim.state.qvel), sim.step_sensordata()))
  sim.restore(initial)
  # Full replay without queries compares against the trajectory with queries;
  # this detects retained solver/history side effects of current-state reads.
  for qp, qv, sensors in trajectory:
    np.testing.assert_array_equal(host(sim.step()), 0)
    np.testing.assert_array_equal(host(sim.state.qpos), qp)
    np.testing.assert_array_equal(host(sim.state.qvel), qv)
    np.testing.assert_array_equal(sim.step_sensordata(), sensors)
  sim.reset(qpos=q, qvel=v)
  np.testing.assert_array_equal(sim.step_sensordata(), 0)

_VERTICES = '-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 -.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1'

def _stock_model(shape, integrator, libccd):
  flag = '<flag nativeccd="disable"/>' if libccd else ''
  asset = (f'<mesh name="solid" vertex="{_VERTICES}"/>' if shape != 'hfield' else
           '<hfield name="solid" nrow="5" ncol="5" size=".3 .3 .06 .02" elevation="' + ' '.join(['0']*25) + '"/>')
  geom = (f'<geom name="solid" type="{shape}" mesh="solid" mass="1"/>' if shape != 'hfield' else
          '<geom name="solid" type="hfield" hfield="solid" mass="1"/>')
  # Heightfields are static in pinned models. Give the analytic distance
  # counterpart motion instead; the scanner always sees the stock asset.
  if shape == 'hfield':
    bodies = geom + '<body pos=".02 .01 .08"><freejoint/><geom name="other" type="sphere" size=".1" mass="1"/></body>'
  else:
    bodies = '<geom name="other" type="plane" size="1 1 .1"/><body pos=".02 .01 .08"><freejoint/>' + geom + '</body>'
  sensors = '<rangefinder site="eye" data="dist dir origin point normal depth"/>'
  sensors += ''.join(f'<{kind} geom1="{a}" geom2="{b}" cutoff=".5"/>'
                    for a,b in [('solid','other'),('other','solid')]
                    for kind in ([] if shape == 'hfield' else ['distance','normal','fromto']))
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep=".001" gravity="0 0 0" integrator="{integrator}">{flag}</option>'
      '<default><geom contype="0" conaffinity="0"/></default><asset>' + asset + '</asset><worldbody>' + bodies +
      '<body name="scanner" pos=".06 .03125 .5"><site name="eye" quat="0 1 0 0"/></body>'
      '</worldbody><sensor>' + sensors + '</sensor></mujoco>')

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native stock geometry sensor integration')
@pytest.mark.parametrize('shape', ['mesh','sdf','hfield'])
@pytest.mark.parametrize('libccd', [False, True])
@pytest.mark.parametrize('integrator,profile', _PROFILES)
def test_stock_geometry_sensor_queries_stored_stages_and_replay(shape, libccd, integrator, profile):
  from mujoco_metal.simulation import MetalSimulation
  from mujoco_metal.capacity import model_component_capacity_limits
  m = _stock_model(shape, integrator, libccd)
  q = np.tile(m.qpos0.astype(np.float32), (2,1)); q[1,0] = 2
  v = np.tile(np.array([.03,-.02,.01,.01,-.02,.03], np.float32), (2,1))
  sim = MetalSimulation(m, 2, qpos=q, qvel=v, profile=profile,
                        limits=model_component_capacity_limits(m,2))
  refs = [mujoco.MjData(m) for _ in range(2)]
  for w, ref in enumerate(refs):
    ref.qpos[:] = q[w]; ref.qvel[:] = v[w]; mujoco.mj_forward(m,ref)
  host = lambda value: value.cpu().numpy().copy()
  got = host(sim.sensor_values())
  np.testing.assert_allclose(got, [r.sensordata for r in refs], rtol=3e-5, atol=3e-5)
  assert got[0, 0] > 0
  assert np.any(got[0] != got[1])
  snapshot = sim.snapshot(); records=[]
  for _ in range(5):
    np.testing.assert_array_equal(host(sim.step()),0)
    for ref in refs: mujoco.mj_step(m,ref)
    np.testing.assert_allclose(sim.step_sensordata(), [r.sensordata for r in refs], rtol=3e-5, atol=3e-5)
    records.append((host(sim.state.qpos),sim.step_sensordata()))
  sim.restore(snapshot)
  for qp, sensors in records:
    np.testing.assert_array_equal(host(sim.step()),0)
    np.testing.assert_array_equal(host(sim.state.qpos),qp)
    np.testing.assert_array_equal(sim.step_sensordata(),sensors)
