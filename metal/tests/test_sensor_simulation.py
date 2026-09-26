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
