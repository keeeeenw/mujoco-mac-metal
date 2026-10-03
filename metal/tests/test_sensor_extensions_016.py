# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 016 & 019: Tactile, USER, and PLUGIN sensors, and multislot reductions."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.extensions import default_registry, CustomUserSensorPlugin

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
    ),
]


def test_tactile_sensor_multislot_reduction_gpu():
  """Tactile sensor reports taxel penetration with a multi-point manifold."""
  from mujoco_metal.simulation import MetalSimulation
  # Floor with multiple spheres contacting a box geom
  xml = """
  <mujoco>
    <option timestep="0.002" cone="pyramidal"/>
    <asset>
      <mesh name="m_box" vertex="-0.1 -0.1 -0.1  0.1 -0.1 -0.1  0.1 0.1 -0.1  -0.1 0.1 -0.1
                                -0.1 -0.1 0.1   0.1 -0.1 0.1   0.1 0.1 0.1   -0.1 0.1 0.1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 0.1"/>
      <body name="sensor_body" pos="0 0 0.05">
        <joint type="slide" axis="0 0 1"/>
        <geom name="sensor_geom" type="box" size="0.1 0.1 0.05" condim="3"/>
      </body>
    </worldbody>
    <sensor>
      <tactile name="tactile_box" geom="sensor_geom" mesh="m_box"/>
    </sensor>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset()

  # Step simulation
  for _ in range(5):
    sim.step(1)

  sens = sim.step_sensordata()
  assert sens.shape == (1, m.nsensordata)
  # Tactile depth is geometric; it is not the solved normal contact force.
  assert np.all(np.isfinite(sens))
  assert np.sum(sens[0]) > 0.0, f"Expected positive tactile depth, got {sens[0]}"


def test_user_sensor_native_plugin_execution_gpu():
  """User sensor evaluates custom plugin natively on GPU device memory."""
  from mujoco_metal.simulation import MetalSimulation
  xml = """
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body name="body1" pos="0 0 0">
        <joint name="j1" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <user name="u_sensor" dim="2"/>
    </sensor>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)

  # Register custom user sensor plugin
  plugin = CustomUserSensorPlugin(name="u_sensor", dim=2)
  default_registry.register(plugin)

  try:
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    sim.reset(qpos=np.array([[0.3]], dtype=np.float32),
              qvel=np.array([[0.4]], dtype=np.float32))

    # Step simulation
    sim.step(1)

    sens = sim.step_sensordata()[0]
    # Plugin sets [norm(qpos), norm(qvel)]
    # qpos=0.3, qvel=0.4
    assert sens.shape == (2,)
    assert np.all(np.isfinite(sens))
    assert sens[0] > 0.0
    assert sens[1] > 0.0
  finally:
    default_registry.unregister("u_sensor")


def test_multislot_contact_sensor_reduction_gpu():
  """Contact sensor reduces multiple active contact slots correctly."""
  from mujoco_metal.simulation import MetalSimulation
  xml = """
  <mujoco>
    <option timestep="0.002" cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 0.1"/>
      <body name="b1" pos="-0.05 0 0.05">
        <joint type="slide" axis="0 0 1"/>
        <geom name="g1" type="sphere" size="0.05" condim="3"/>
      </body>
      <body name="b2" pos="0.05 0 0.05">
        <joint type="slide" axis="0 0 1"/>
        <geom name="g2" type="sphere" size="0.05" condim="3"/>
      </body>
    </worldbody>
    <sensor>
      <touch name="t1" site="s_floor"/>
    </sensor>
    <worldbody>
      <site name="s_floor" pos="0 0 0" size="0.5 0.5 0.2" type="box"/>
    </worldbody>
  </mujoco>
  """
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset()

  # Step simulation
  for _ in range(5):
    sim.step(1)

  sens = sim.step_sensordata()[0]
  # Both spheres contact floor in site volume -> touch sensor sums both
  assert sens[0] > 0.0
