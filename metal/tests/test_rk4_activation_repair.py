# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R06/D3: stateful RK4 activation-stage composition (failing-first)."""

import mujoco
import numpy as np
import pytest


def _filter_model():
  return mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='RK4'><flag contact='disable'/></option>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><general name='g' joint='h' dyntype='filter' gainprm='3' biasprm='0 -2 0'/></actuator>"
      "</mujoco>")


def test_rk4_actuator_admission_cpu():
  from mujoco_metal.stepping import validate_stepping_profile
  m = _filter_model()
  assert m.na == 1
  prof = validate_stepping_profile(m, 0.002, "integrated_rk4_v1")
  assert prof is not None


def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
def test_rk4_filter_activation_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = _filter_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_rk4_v1")
  sim.reset(qpos=np.array([[0.2]], dtype=np.float32),
            qvel=np.zeros((1, m.nv), dtype=np.float32),
            act=np.array([[0.1]], dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = [0.2]
  cpu.act[:] = [0.1]
  for _ in range(60):
    sim.step(1, ctrl=np.array([[0.8]], dtype=np.float32))
    cpu.ctrl[:] = [0.8]
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], np.asarray(cpu.qpos),
                             rtol=1e-4, atol=1e-5)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[0], np.asarray(cpu.qvel),
                             rtol=1e-4, atol=1e-5)
  np.testing.assert_allclose(sim.state._act.cpu().numpy()[0], np.asarray(cpu.act),
                             rtol=1e-4, atol=1e-5)


@_needs_gpu()
def test_rk4_muscle_activation_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='RK4'><flag contact='disable'/></option>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1' limited='true' range='-1 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><muscle name='mu' joint='h' force='5'/></actuator>"
      "</mujoco>")
  assert m.na == 1
  sim = MetalSimulation(m, batch_size=1, profile="integrated_rk4_v1")
  sim.reset(qpos=np.array([[0.1]], dtype=np.float32),
            qvel=np.zeros((1, m.nv), dtype=np.float32),
            act=np.array([[0.2]], dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = [0.1]
  cpu.act[:] = [0.2]
  for _ in range(40):
    sim.step(1, ctrl=np.array([[0.6]], dtype=np.float32))
    cpu.ctrl[:] = [0.6]
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], np.asarray(cpu.qpos),
                             rtol=1e-3, atol=1e-4)
  np.testing.assert_allclose(sim.state._act.cpu().numpy()[0], np.asarray(cpu.act),
                             rtol=1e-3, atol=1e-4)
