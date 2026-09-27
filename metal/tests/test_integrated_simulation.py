# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Integration and numerical qualification tests for integrated_euler_v1 pipeline."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.stepping import validate_stepping_profile

INTEGRATED_XML = """<mujoco model="integrated_all_families">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler" iterations="1000" tolerance="1e-6" density="1.2" viscosity="0.00001">
    <flag contact="enable" equality="enable" limit="enable" frictionloss="enable"/>
  </option>
  <default>
    <geom friction="0.4 0.1 0.1" solref="0.02 1.0" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="lever1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" range="-0.8 0.8" limited="true" frictionloss="0.05" damping="0.1"/>
      <geom name="g_lever1" type="sphere" size="0.2" pos="0.5 0 0" mass="1"/>
    </body>
    <body name="lever2" pos="0 1 1">
      <joint name="j2" type="hinge" axis="0 1 0" range="-0.8 0.8" limited="true" frictionloss="0.02" damping="0.05"/>
      <geom name="g_lever2" type="sphere" size="0.2" pos="0.5 0 0" mass="1"/>
    </body>
    <body name="ball" pos="0.5 0 1.5">
      <joint name="ball_z" type="slide" axis="0 0 1" damping="0.02"/>
      <geom name="g_ball" type="sphere" size="0.1" mass="0.5"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j2" joint2="j1" polycoef="0 1.2 0 0 0"/>
  </equality>
  <tendon>
    <fixed name="t1" stiffness="5.0" damping="0.1" armature="0.02">
      <joint joint="j1" coef="1.0"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <motor joint="j1" ctrlrange="-5 5"/>
  </actuator>
  <sensor>
    <jointpos joint="j1"/>
    <jointvel joint="j2"/>
  </sensor>
</mujoco>"""


def test_integrated_profile_validation():
  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  assert profile.name == "integrated_euler_v1"
  assert profile.nv == 3
  assert profile.nq == 3
  assert any("coupled constraint" in s for s in profile.supported)
  assert any("tendon" in s for s in profile.supported)
  assert any("actuator" in s for s in profile.supported)


def test_integrated_profile_rejections():
  # Non-Euler integrator
  xml = INTEGRATED_XML.replace('integrator="Euler"', 'integrator="RK4"')
  with pytest.raises(ValueError, match="Euler integrator"):
    validate_stepping_profile(mujoco.MjModel.from_xml_string(xml), profile="integrated_euler_v1")

  # Excessive degrees of freedom
  huge_xml = """<mujoco><compiler angle="radian"/><option integrator="Euler"/><worldbody>"""
  for i in range(35):
    huge_xml += f'<body pos="{i} 0 0"><joint type="hinge"/><geom type="sphere" size="0.1"/></body>'
  huge_xml += "</worldbody></mujoco>"
  with pytest.raises(ValueError, match="bounds nv to 32"):
    validate_stepping_profile(mujoco.MjModel.from_xml_string(huge_xml), profile="integrated_euler_v1")


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_trajectory_matches_cpu():
  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  batch = 2
  q0 = np.array([[0.0, 0.0, -0.25], [0.05, 0.06, -0.2]], dtype=np.float32)
  v0 = np.zeros((batch, model.nv), dtype=np.float32)

  sim = MetalSimulation(model, batch_size=batch, qpos=q0, qvel=v0, profile="integrated_euler_v1")

  # CPU references
  cpu_datas = [mujoco.MjData(model) for _ in range(batch)]
  for b in range(batch):
    cpu_datas[b].qpos[:] = q0[b]
    cpu_datas[b].qvel[:] = v0[b]

  max_pos_err = 0.0
  max_vel_err = 0.0
  max_sensor_err = 0.0

  for step in range(100):
    ctrl = np.array([[0.5 * np.sin(step * 0.1)], [-0.3 * np.cos(step * 0.08)]], dtype=np.float32)
    sim.step(1, ctrl=ctrl)

    for b in range(batch):
      cpu_datas[b].ctrl[0] = ctrl[b, 0]
      mujoco.mj_step(model, cpu_datas[b])

    # Check errors
    gpu_qpos = sim.state.qpos.cpu().numpy()
    gpu_qvel = sim.state.qvel.cpu().numpy()
    for b in range(batch):
      pos_err = np.max(np.abs(gpu_qpos[b] - cpu_datas[b].qpos))
      vel_err = np.max(np.abs(gpu_qvel[b] - cpu_datas[b].qvel))
      max_pos_err = max(max_pos_err, pos_err)
      max_vel_err = max(max_vel_err, vel_err)

    # Sensor queries
    gpu_sensors = sim.sensor_values().cpu().numpy()
    for b in range(batch):
      d_check = mujoco.MjData(model)
      d_check.qpos[:] = cpu_datas[b].qpos
      d_check.qvel[:] = cpu_datas[b].qvel
      d_check.time = cpu_datas[b].time
      mujoco.mj_fwdPosition(model, d_check)
      mujoco.mj_fwdVelocity(model, d_check)
      mujoco.mj_sensorPos(model, d_check)
      mujoco.mj_sensorVel(model, d_check)
      s_err = np.max(np.abs(gpu_sensors[b] - d_check.sensordata))
      max_sensor_err = max(max_sensor_err, s_err)

  assert max_pos_err < 1e-5, f"Position error {max_pos_err} exceeded 1e-5"
  assert max_vel_err < 1e-4, f"Velocity error {max_vel_err} exceeded 1e-4"
  assert max_sensor_err < 1e-4, f"Sensor error {max_sensor_err} exceeded 1e-4"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_snapshot_and_reset():
  import torch

  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  batch = 2
  q0 = np.array([[0.0, 0.0, -0.25], [0.05, 0.06, -0.2]], dtype=np.float32)
  v0 = np.zeros((batch, model.nv), dtype=np.float32)

  sim = MetalSimulation(model, batch_size=batch, qpos=q0, qvel=v0, profile="integrated_euler_v1")
  ctrl = np.array([[0.5], [-0.5]], dtype=np.float32)

  sim.step(15, ctrl=ctrl)
  snapshot = sim.state.snapshot()

  sim.step(15, ctrl=ctrl)
  qpos_30 = sim.state.qpos.clone()

  sim.state.restore(snapshot)
  sim.step(15, ctrl=ctrl)
  qpos_replayed_30 = sim.state.qpos.clone()

  diff = torch.max(torch.abs(qpos_30 - qpos_replayed_30)).item()
  assert diff == 0.0, f"Replay divergence: {diff}"

  # Selective reset
  sim.state.reset(env_ids=[1])
  assert torch.allclose(sim.state.qpos[1], torch.tensor(model.qpos0, device="mps", dtype=torch.float32))
  assert not torch.allclose(sim.state.qpos[0], torch.tensor(model.qpos0, device="mps", dtype=torch.float32))


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_sticky_failure_isolation():
  import torch

  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  batch = 2
  sim = MetalSimulation(model, batch_size=batch, profile="integrated_euler_v1")

  # Feed NaN control only to world 1
  ctrl = torch.tensor([[0.5], [float("nan")]], dtype=torch.float32, device="mps")
  sim.step(1, ctrl=ctrl)

  status = sim.state.status
  assert status[0] == 0
  assert status[1] != 0

  # World 0 continues advancing while World 1 remains stuck
  qpos0_before = sim.state.qpos[0].clone()
  qpos1_before = sim.state.qpos[1].clone()

  valid_ctrl = torch.tensor([[0.5], [0.5]], dtype=torch.float32, device="mps")
  sim.step(1, ctrl=valid_ctrl)

  assert not torch.allclose(sim.state.qpos[0], qpos0_before)
  assert torch.allclose(sim.state.qpos[1], qpos1_before)
  assert sim.state.status[1] != 0


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_zero_equality_model_gpu():
  xml = """<mujoco model="zero_equality">
    <compiler angle="radian"/>
    <option timestep="0.002" integrator="Euler"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <body name="ball" pos="0 0 1">
        <joint name="j" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.neq == 0
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  d_cpu = [mujoco.MjData(model) for _ in range(2)]

  for _ in range(20):
    sim.step(1)
    for b in range(2):
      mujoco.mj_step(model, d_cpu[b])

  for b in range(2):
    np.testing.assert_allclose(sim.state.qpos[b].cpu().numpy(), d_cpu[b].qpos, atol=1e-5)
    np.testing.assert_allclose(sim.state.qvel[b].cpu().numpy(), d_cpu[b].qvel, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_zero_contact_model_gpu():
  xml = """<mujoco model="zero_contact">
    <compiler angle="radian"/>
    <option timestep="0.002" integrator="Euler"/>
    <worldbody>
      <body name="b1" pos="0 0 1">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.01" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body name="b2" pos="0 1 1">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.01" mass="1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <equality>
      <joint joint1="j2" joint2="j1" polycoef="0 1 0 0 0"/>
    </equality>
  </mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  d_cpu = [mujoco.MjData(model) for _ in range(2)]

  for _ in range(20):
    sim.step(1)
    for b in range(2):
      mujoco.mj_step(model, d_cpu[b])

  for b in range(2):
    np.testing.assert_allclose(sim.state.qpos[b].cpu().numpy(), d_cpu[b].qpos, atol=1e-5)
    np.testing.assert_allclose(sim.state.qvel[b].cpu().numpy(), d_cpu[b].qvel, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_empty_fixed_world_gpu():
  xml = """<mujoco model="empty_world">
    <compiler angle="radian"/>
    <option timestep="0.002" integrator="Euler"/>
    <worldbody>
      <geom type="sphere" size="0.1"/>
    </worldbody>
  </mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.nv == 0
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  sim.step(5)
  assert sim.state.qpos.shape == (2, 0)
  assert sim.state.qvel.shape == (2, 0)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_minimal_actuated_model_gpu():
  xml = """<mujoco model="minimal_actuated">
    <compiler angle="radian"/>
    <option timestep="0.002" integrator="Euler"/>
    <worldbody>
      <body name="arm" pos="0 0 1">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="j"/>
    </actuator>
    <sensor>
      <jointpos joint="j"/>
    </sensor>
  </mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  d_cpu = [mujoco.MjData(model) for _ in range(2)]

  ctrl = np.array([[1.0], [-0.5]], dtype=np.float32)
  for _ in range(20):
    sim.step(1, ctrl=ctrl)
    for b in range(2):
      d_cpu[b].ctrl[0] = ctrl[b, 0]
      mujoco.mj_step(model, d_cpu[b])

  for b in range(2):
    np.testing.assert_allclose(sim.state.qpos[b].cpu().numpy(), d_cpu[b].qpos, atol=1e-5)
    np.testing.assert_allclose(sim.state.qvel[b].cpu().numpy(), d_cpu[b].qvel, atol=1e-4)


def test_integrated_simulation_execution_plan_and_buffer_audit():
  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  profile = validate_stepping_profile(model, profile="integrated_euler_v1")
  plan = profile.execution_plan
  assert plan is not None
  assert plan.profile_name == "integrated_euler_v1"
  assert len(plan.stages) == 9
  assert any(s.name == "coupled_constraints" and s.enabled for s in plan.stages)
  assert any(s.name == "fixed_tendons" and s.enabled for s in plan.stages)
  assert any(s.name == "sensors" and s.enabled for s in plan.stages)

  audit = plan.buffer_audit
  assert len(audit) >= 14
  names = [b["name"] for b in audit]
  assert "state.qpos" in names
  assert "workspace_J" in names
  assert "_eq_active_default" in names
  assert "sensordata" in names


def test_integrated_simulation_capacity_boundary_and_overflow_rejection():
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  # nv > 32
  huge_xml = """<mujoco><compiler angle="radian"/><option integrator="Euler"/><worldbody>"""
  for i in range(35):
    huge_xml += f'<body pos="{i} 0 0"><joint type="hinge"/><geom type="sphere" size="0.1"/></body>'
  huge_xml += "</worldbody></mujoco>"
  with pytest.raises(ValueError, match="bounds nv to 32"):
    validate_stepping_profile(mujoco.MjModel.from_xml_string(huge_xml), profile="integrated_euler_v1")

  # nc > 16 (7 dynamic bodies give 21 candidate pairs > 16)
  pairs_xml = """<mujoco><compiler angle="radian"/><option integrator="Euler"/><worldbody>"""
  for i in range(7):
    pairs_xml += f'<body pos="{i} 0 0"><joint type="slide" axis="0 0 1"/><geom type="sphere" size="0.1"/></body>'
  pairs_xml += "</worldbody></mujoco>"
  with pytest.raises(ValueError, match="contact pairs"):
    lower_coupled_constraints(mujoco.MjModel.from_xml_string(pairs_xml))


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_multiple_timesteps_and_nonzero_initial_states():
  import torch

  for dt in [0.001, 0.002, 0.004]:
    xml = INTEGRATED_XML.replace('timestep="0.002"', f'timestep="{dt}"')
    model = mujoco.MjModel.from_xml_string(xml)
    q0 = np.array([[0.1, -0.15, -0.22]], dtype=np.float32)
    v0 = np.array([[0.2, 0.1, -0.05]], dtype=np.float32)

    sim = MetalSimulation(model, batch_size=1, qpos=q0, qvel=v0, profile="integrated_euler_v1")
    d_cpu = mujoco.MjData(model)
    d_cpu.qpos[:] = q0[0]
    d_cpu.qvel[:] = v0[0]

    for step in range(25):
      ctrl = np.array([[0.3 * np.sin(step * 0.1)]], dtype=np.float32)
      sim.step(1, ctrl=ctrl)
      d_cpu.ctrl[0] = ctrl[0, 0]
      mujoco.mj_step(model, d_cpu)

    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), d_cpu.qpos, atol=1e-5)
    np.testing.assert_allclose(sim.state.qvel[0].cpu().numpy(), d_cpu.qvel, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_varying_forces_and_body_wrenches():
  import torch

  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(model)

  for step in range(30):
    qfrc = np.array([[0.2 * np.sin(step * 0.2), -0.1 * np.cos(step * 0.15), 0.05 * np.sin(step * 0.1)]], dtype=np.float32)
    xfrc = np.zeros((1, model.nbody, 6), dtype=np.float32)
    xfrc[0, 1, :3] = [0.1 * np.cos(step * 0.1), 0.0, 0.2 * np.sin(step * 0.1)]

    sim.step(1, qfrc_applied=qfrc, xfrc_applied=xfrc)

    d_cpu.qfrc_applied[:] = qfrc[0]
    d_cpu.xfrc_applied[:] = xfrc[0]
    mujoco.mj_step(model, d_cpu)

    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), d_cpu.qpos, atol=1e-5)
    np.testing.assert_allclose(sim.state.qvel[0].cpu().numpy(), d_cpu.qvel, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_disable_flags_physical_effect():
  import torch

  for bit in [
      mujoco.mjtDisableBit.mjDSBL_ACTUATION,
      mujoco.mjtDisableBit.mjDSBL_DAMPER,
      mujoco.mjtDisableBit.mjDSBL_EQUALITY,
      mujoco.mjtDisableBit.mjDSBL_LIMIT,
      mujoco.mjtDisableBit.mjDSBL_CONTACT,
  ]:
    model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
    model.opt.disableflags |= int(bit)
    sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
    d_cpu = mujoco.MjData(model)
    ctrl = np.array([[0.5]], dtype=np.float32)

    for _ in range(15):
      sim.step(1, ctrl=ctrl)
      d_cpu.ctrl[0] = ctrl[0, 0]
      mujoco.mj_step(model, d_cpu)

    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), d_cpu.qpos, atol=1e-5)
    np.testing.assert_allclose(sim.state.qvel[0].cpu().numpy(), d_cpu.qvel, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_convergence_failure_and_rollback():
  import torch

  # Coupled problem with tight tolerance and low iteration cap causing iteration exhaustion
  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="2" tolerance="1e-6"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.05">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" friction="1 0.1 0.1"/>
    </body>
    <body pos="0 0 0.15">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" friction="1 0.1 0.1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0.05 1 0 0 0"/>
  </equality>
</mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")

  qpos_initial = sim.state.qpos.clone()
  sim.step(1)

  # Status reports 3 (iteration exhaustion / non-convergence)
  status = sim.state.status.cpu().numpy()
  assert np.all(status == 3), f"Expected status 3, got {status}"

  # Verify world state did not advance on failure (rollback to previous valid state)
  assert torch.allclose(sim.state.qpos, qpos_initial)

  # Verify subsequent step remains sticky failure
  sim.step(1)
  assert np.all(sim.state.status.cpu().numpy() == 3)
  assert torch.allclose(sim.state.qpos, qpos_initial)

  # Verify selective reset recovers cleanly per environment
  sim.state.reset(env_ids=[0])
  assert sim.state.status[0].item() == 0
  assert sim.state.status[1].item() == 3

  sim.state.reset(env_ids=[1])
  assert torch.all(sim.state.status == 0)

