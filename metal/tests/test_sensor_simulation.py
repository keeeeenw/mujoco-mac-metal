# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Sensor queries across stepping, reset and checkpoint restore."""

import os
import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
def test_sensor_queries_follow_live_state_and_restore():
  from mujoco_metal.simulation import MetalSimulation

  m = mujoco.MjModel.from_xml_string(
      """<mujoco><option gravity='0 0 0'><flag contact='disable'/></option>
  <worldbody><body><joint name='j' damping='.1'/><geom type='box' size='.1 .2 .3'/><site name='tip' pos='.2 .1 .3'/></body></worldbody>
  <actuator><motor joint='j'/></actuator><sensor><jointpos joint='j'/><jointvel joint='j'/><framepos objtype='site' objname='tip'/><clock/></sensor></mujoco>"""
  )
  sim = MetalSimulation(m, 2, profile='contact_free_sensor_euler_v1')

  def check():
    snap = sim.state.snapshot()
    actual = sim.sensor_values().cpu().numpy()
    for i in range(2):
      d = mujoco.MjData(m)
      d.qpos[:] = snap.qpos[i]
      d.qvel[:] = snap.qvel[i]
      d.time = float(snap.time[i])
      mujoco.mj_forward(m, d)
      np.testing.assert_allclose(actual[i], d.sensordata, atol=2e-6, rtol=2e-5)
    return actual

  start = check()
  sim.step(20, ctrl=np.array([[1.0], [-2.0]], dtype='float32'))
  saved = sim.state.snapshot()
  values = check()
  sim.step(20)
  check()
  sim.state.restore(saved)
  np.testing.assert_array_equal(check(), values)
  sim.state.reset()
  np.testing.assert_array_equal(check(), start)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='opt-in GPU'
)
def test_live_velocity_sensors_on_rotated_articulated_free_body():
  from mujoco_metal.simulation import MetalSimulation

  m = mujoco.MjModel.from_xml_string(
      """<mujoco><option gravity="0 0 0"><flag contact="disable"/></option>
  <worldbody><body name="root" pos=".3 -.2 .5" quat=".8 .1 .3 -.2"><freejoint/><geom type="box" size=".1 .2 .3" pos=".07 -.1 .04" mass="2"/>
    <site name="ref" pos=".1 -.2 .3" quat=".9 .2 .1 .3"/>
    <body name="child" pos=".4 .2 .1" quat=".9 .1 -.2 .1"><joint name="j" type="ball"/><geom name="geom" type="box" size=".1 .2 .15" pos="-.1 .1 .2" mass="1"/><site name="tip" pos=".3 .2 .1" quat=".9 -.2 .1 .3"/></body>
  </body></worldbody><sensor>
  <framelinvel objtype="body" objname="child"/>
  <frameangvel objtype="xbody" objname="child" reftype="site" refname="ref"/>
  <framelinvel objtype="geom" objname="geom" reftype="body" refname="root"/>
  <framelinvel objtype="site" objname="tip" reftype="site" refname="ref"/>
  <gyro site="tip"/><velocimeter site="tip"/>
  <framepos objtype="body" objname="child" reftype="site" refname="ref"/>
  </sensor></mujoco>"""
  )
  q = np.tile(m.qpos0, (3, 1))
  v = np.tile(np.linspace(-0.8, 0.9, m.nv), (3, 1))
  for row in range(3):
    mujoco.mj_integratePos(m, q[row], v[row], 0.2 * row)
    v[row] *= row + 1
  sim = MetalSimulation(
      m,
      3,
      q.astype('float32'),
      v.astype('float32'),
      profile='contact_free_sensor_euler_v1',
  )
  for _ in range(4):
    snap = sim.state.snapshot()
    actual = sim.sensor_values().cpu().numpy()
    for row in range(3):
      d = mujoco.MjData(m)
      d.qpos[:] = snap.qpos[row]
      d.qvel[:] = snap.qvel[row]
      mujoco.mj_forward(m, d)
      np.testing.assert_allclose(
          actual[row], d.sensordata, atol=4e-6, rtol=4e-5
      )
    sim.step(10)
