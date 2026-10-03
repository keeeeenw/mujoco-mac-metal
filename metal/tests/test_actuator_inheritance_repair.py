# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R06-1 repair: actuator armature/damping inheritance to joints and tendons.

Motors with armature/damping on joints and fixed tendons must match the
CPU oracle (mass diagonal, damping force, trajectory); tendon-targeted
armature/damping is supported via tendon stage folding; delay and history
stay rejected.
"""

import mujoco
import numpy as np
import pytest


def _arm_model():
  return mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1' damping='0.3'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><motor name='m' joint='h' gear='2' armature='0.4' damping='0.7'/></actuator>"
      "</mujoco>")


def test_actuator_armature_damping_folds_cpu():
  # Host-side folds are CPU-testable: gear^2 inheritance lands on the
  # driven dof (armature 0.4*4=1.6, damping 0.7*4=2.8).
  from mujoco_metal.model import actuator_joint_inheritance, actuator_tendon_inheritance
  m = _arm_model()
  arm, damp, damppoly = actuator_joint_inheritance(m)
  np.testing.assert_allclose(arm, [1.6])
  np.testing.assert_allclose(damp, [2.8])
  assert damppoly.shape == (1, 2)

  # Tendon-targeted raises on joint inheritance when tendon_ok=False.
  mt = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<tendon><fixed name='t'><joint joint='h' coef='1'/></fixed></tendon>"
      "<actuator><motor name='m' tendon='t' gear='2' armature='0.2' damping='0.3'/></actuator>"
      "</mujoco>")
  with pytest.raises(ValueError, match="tendon-targeted"):
    actuator_joint_inheritance(mt, tendon_ok=False)

  # Tendon inheritance folds gear^2 into tendon armature and damping (0.2*4=0.8, 0.3*4=1.2)
  t_arm, t_damp, _ = actuator_tendon_inheritance(mt)
  np.testing.assert_allclose(t_arm, [0.8])
  np.testing.assert_allclose(t_damp, [1.2])

  # Site-targeted raises
  ms = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "<site name='s'/></body></worldbody>"
      "<actuator><motor name='m' site='s' gear='1'/></actuator>"
      "</mujoco>")
  ms.actuator_armature[0] = 0.2
  with pytest.raises(ValueError, match="site/body-targeted"):
    actuator_joint_inheritance(ms, tendon_ok=True)


def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
def test_actuator_armature_damping_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = _arm_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  qv = np.array([[1.5]], dtype=np.float32)
  sim.reset(qpos=qp, qvel=qv)
  # Mass diagonal carries gear^2 * armature on the driven dof.
  asm = sim.assembled_system(recompute=True)
  M = np.asarray(asm["mass_matrix"].cpu().numpy())[0]
  d = mujoco.MjData(m)
  d.qpos[:] = qp[0]
  d.qvel[:] = qv[0]
  mujoco.mj_forward(m, d)
  np.testing.assert_allclose(np.diag(M), np.diag(np.asarray(d.qM).reshape(m.nv, m.nv)),
                             rtol=1e-5, atol=1e-6)
  # Trajectory parity under drive (armature slows response, damping bleeds).
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  cpu.qvel[:] = qv[0]
  for step in range(100):
    u = np.array([[0.6]], dtype=np.float32)
    sim.step(1, ctrl=u)
    cpu.ctrl[:] = [0.6]
    mujoco.mj_step(m, cpu)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], np.asarray(cpu.qpos),
                             rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[0], np.asarray(cpu.qvel),
                             rtol=1e-4, atol=1e-4)


@_needs_gpu()
def test_actuator_dampingpoly_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><motor name='m' joint='h' gear='1'/></actuator>"
      "</mujoco>")
  m.dof_dampingpoly[0] = [0.1, 0.02]
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  qv = np.array([[2.0]], dtype=np.float32)
  sim.reset(qpos=qp, qvel=qv)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  cpu.qvel[:] = qv[0]
  # Assert oracle passive forces and damping derivative before trajectory
  mujoco.mj_forward(m, cpu)
  np.testing.assert_allclose(cpu.qfrc_passive, [-0.56], rtol=1e-5, atol=1e-6)
  for _ in range(60):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[0], np.asarray(cpu.qvel),
                             rtol=1e-4, atol=1e-4)


@_needs_gpu()
def test_tendon_armature_damping_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  mt = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<tendon><fixed name='t'><joint joint='h' coef='1'/></fixed></tendon>"
      "<actuator><motor name='m' tendon='t' gear='2' armature='0.2' damping='0.3'/></actuator>"
      "</mujoco>")
  sim = MetalSimulation(mt, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(mt.qpos0, dtype=np.float32).reshape(1, -1)
  qv = np.array([[1.5]], dtype=np.float32)
  sim.reset(qpos=qp, qvel=qv)
  asm = sim.assembled_system(recompute=True)
  d = mujoco.MjData(mt)
  d.qpos[:] = qp[0]
  d.qvel[:] = qv[0]
  mujoco.mj_forward(mt, d)
  np.testing.assert_allclose(np.asarray(asm["mass_matrix"].cpu().numpy())[0],
                             np.asarray(d.qM).reshape(mt.nv, mt.nv), rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(d.qfrc_passive, [-1.8], rtol=1e-5, atol=1e-6)
  for step in range(50):
    u = np.array([[0.5]], dtype=np.float32)
    sim.step(1, ctrl=u)
    d.ctrl[:] = [0.5]
    mujoco.mj_step(mt, d)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], np.asarray(d.qpos),
                             rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[0], np.asarray(d.qvel),
                             rtol=1e-4, atol=1e-4)


def test_actuator_delay_history_rejected_cpu():
  from mujoco_metal.stepping import validate_stepping_profile
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><motor name='m' joint='h' gear='1'/></actuator>"
      "</mujoco>")
  m.actuator_delay[0] = 0.01
  with pytest.raises(ValueError, match="delay"):
    validate_stepping_profile(m, 0.002, profile="integrated_euler_v1")

  m2 = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><motor name='m' joint='h' gear='1'/></actuator>"
      "</mujoco>")
  m2.actuator_history[0] = 1
  with pytest.raises(ValueError, match="history"):
    validate_stepping_profile(m2, 0.002, profile="integrated_euler_v1")


@_needs_gpu()
def test_multiple_actuators_and_gear_signs_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator>"
      "<motor name='m1' joint='h' gear='1.5' armature='0.1' damping='0.2'/>"
      "<motor name='m2' joint='h' gear='-2.0' armature='0.05' damping='0.1'/>"
      "</actuator>"
      "</mujoco>")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  qv = np.array([[1.0]], dtype=np.float32)
  sim.reset(qpos=qp, qvel=qv)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  cpu.qvel[:] = qv[0]
  for _ in range(50):
    u = np.array([[0.3, -0.4]], dtype=np.float32)
    sim.step(1, ctrl=u)
    cpu.ctrl[:] = [0.3, -0.4]
    mujoco.mj_step(m, cpu)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], np.asarray(cpu.qpos),
                             rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[0], np.asarray(cpu.qvel),
                             rtol=1e-4, atol=1e-4)


def test_ball_and_free_joint_actuator_inheritance_cpu():
  from mujoco_metal.model import actuator_joint_inheritance
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='b' type='ball'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><motor name='m' joint='b' gear='2' armature='0.1' damping='0.3'/></actuator>"
      "</mujoco>")
  arm, damp, _ = actuator_joint_inheritance(m)
  assert arm.shape == (3,)
  assert damp.shape == (3,)
  # gear^2 = 4 -> arm = 0.4 on all 3 dofs, damp = 1.2 on all 3 dofs
  np.testing.assert_allclose(arm, [0.4, 0.4, 0.4])
  np.testing.assert_allclose(damp, [1.2, 1.2, 1.2])

  # Free joint (6 dofs)
  mf = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='f' type='free'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      "<actuator><motor name='m' joint='f' gear='1' armature='0.5' damping='0.25'/></actuator>"
      "</mujoco>")
  arm_f, damp_f, _ = actuator_joint_inheritance(mf)
  assert arm_f.shape == (6,)
  assert damp_f.shape == (6,)
  np.testing.assert_allclose(arm_f, [0.5] * 6)
  np.testing.assert_allclose(damp_f, [0.25] * 6)


@_needs_gpu()
def test_disabled_actuation_semantics_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = _arm_model()
  m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  qv = np.array([[1.5]], dtype=np.float32)
  sim.reset(qpos=qp, qvel=qv)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  cpu.qvel[:] = qv[0]
  for _ in range(50):
    u = np.array([[1.0]], dtype=np.float32)
    sim.step(1, ctrl=u)
    cpu.ctrl[:] = [1.0]
    mujoco.mj_step(m, cpu)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], np.asarray(cpu.qpos),
                             rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[0], np.asarray(cpu.qvel),
                             rtol=1e-4, atol=1e-4)


def test_model_rebuilding_with_actuator_inheritance():
  from mujoco_metal.model import load_model, snapshot_descriptor
  m = _arm_model()
  desc = load_model(m)
  assert desc.actuator_armature.shape == (1,)
  assert desc.actuator_trntype.shape == (1,)
  assert desc.actuator_gear.shape == (1, 6)
  assert desc.actuator_damping.shape == (1,)
  assert desc.actuator_dampingpoly.shape == (1, 2)
  snap = snapshot_descriptor(desc)
  assert snap.actuator_armature[0] == desc.actuator_armature[0]
  assert snap.actuator_damping[0] == desc.actuator_damping[0]
