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
  assert len(plan.stages) == 10
  expected_order = (
      "smooth_dynamics",
      "passive_forces",
      "fluid_forces",
      "fixed_tendons",
      "actuation",
      "smooth_assembly",
      "unconstrained_solve",
      "coupled_constraints",
      "euler_damping",
      "euler_integration",
  )
  assert tuple(s.name for s in plan.stages) == expected_order
  assert any(s.name == "coupled_constraints" and s.enabled for s in plan.stages)
  assert any(s.name == "fixed_tendons" and s.enabled for s in plan.stages)
  assert len(plan.on_demand_queries) == 1
  assert plan.on_demand_queries[0].name == "sensor_query"
  assert plan.is_stage_enabled("sensor_query")
  plan.validate_dependencies()

  audit = plan.buffer_audit
  assert len(audit) >= 12
  names = [b["name"] for b in audit]
  assert "state.qpos" in names
  assert "workspace_J" in names
  assert "_eq_active_default" in names
  assert "sensordata" in names
  status_entry = next(b for b in audit if b["name"] == "state.status")
  assert status_entry["dtype"] == "int32"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_plan_runtime_agreement():
  # 1. Enabled baseline: plan and runtime agree on all stages and subsystems
  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  plan = sim.execution_plan
  assert plan.is_stage_enabled("smooth_dynamics")
  assert sim._smooth is not None
  assert plan.is_stage_enabled("passive_forces")
  assert sim._passive is not None
  assert plan.is_stage_enabled("fluid_forces")
  assert sim._fluid is not None
  assert plan.is_stage_enabled("fixed_tendons")
  assert sim._tendons is not None
  assert plan.is_stage_enabled("actuation")
  assert sim._transmissions is not None
  assert plan.is_stage_enabled("coupled_constraints")
  assert sim._coupled_constraints is not None
  assert plan.is_stage_enabled("euler_damping")
  assert sim._euler_solver is not None
  assert plan.is_stage_enabled("euler_integration")
  assert sim._integrator is not None
  assert plan.is_stage_enabled("sensor_query")
  assert sim._sensors is not None

  audit_names = [e["name"] for e in sim.buffer_audit()]
  assert "workspace_J" in audit_names
  assert "_eq_active_default" in audit_names
  assert "contact_jacobian" in audit_names
  assert "sensordata" in audit_names

  # 2. Disabled constraints via mjDSBL_CONSTRAINT:
  # Plan reports coupled_constraints disabled; simulation omits coupled subsystem and audit omits workspaces
  m_nocnstr = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_nocnstr.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
  sim_nocnstr = MetalSimulation(m_nocnstr, batch_size=2, profile="integrated_euler_v1")
  plan_nocnstr = sim_nocnstr.execution_plan
  assert not plan_nocnstr.is_stage_enabled("coupled_constraints")
  assert sim_nocnstr._coupled_constraints is None
  audit_nocnstr_names = [e["name"] for e in sim_nocnstr.buffer_audit()]
  assert "workspace_J" not in audit_nocnstr_names
  assert "_eq_active_default" not in audit_nocnstr_names
  assert "contact_jacobian" not in audit_nocnstr_names

  # 3. Disabled actuation via mjDSBL_ACTUATION:
  m_noact = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_noact.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  sim_noact = MetalSimulation(m_noact, batch_size=2, profile="integrated_euler_v1")
  assert not sim_noact.execution_plan.is_stage_enabled("actuation")
  assert sim_noact._transmissions is None

  # 4. Minimal empty model (no sensors, no tendons, no actuators):
  empty_xml = """<mujoco>
    <option integrator="Euler" timestep="0.002"/>
    <worldbody>
      <body>
        <joint type="hinge"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>"""
  m_empty = mujoco.MjModel.from_xml_string(empty_xml)
  sim_empty = MetalSimulation(m_empty, batch_size=1, profile="integrated_euler_v1")
  plan_empty = sim_empty.execution_plan
  assert not plan_empty.is_stage_enabled("actuation")
  assert sim_empty._transmissions is None
  assert not plan_empty.is_stage_enabled("fixed_tendons")
  assert sim_empty._tendons is None
  assert not plan_empty.is_stage_enabled("sensor_query")
  assert sim_empty._sensors is None
  audit_empty_names = [e["name"] for e in sim_empty.buffer_audit()]
  assert "sensordata" not in audit_empty_names


def test_integrated_simulation_capacity_boundary_and_overflow_rejection():
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  # 1. nv boundary: nv=32 accepted, nv=33 rejected
  xml_nv32 = """<mujoco><compiler angle="radian"/><option integrator="Euler"/><worldbody>"""
  for i in range(32):
    xml_nv32 += f'<body pos="{i} 0 0"><joint type="hinge"/><geom type="sphere" size="0.1" conaffinity="0"/></body>'
  xml_nv32 += "</worldbody></mujoco>"
  p32 = validate_stepping_profile(mujoco.MjModel.from_xml_string(xml_nv32), profile="integrated_euler_v1")
  assert p32.nv == 32

  xml_nv33 = """<mujoco><compiler angle="radian"/><option integrator="Euler"/><worldbody>"""
  for i in range(33):
    xml_nv33 += f'<body pos="{i} 0 0"><joint type="hinge"/><geom type="sphere" size="0.1" conaffinity="0"/></body>'
  xml_nv33 += "</worldbody></mujoco>"
  with pytest.raises(ValueError, match="bounds nv to 32"):
    validate_stepping_profile(mujoco.MjModel.from_xml_string(xml_nv33), profile="integrated_euler_v1")

  # 2. nc boundary: nc=16 accepted, nc=17 rejected
  # 8 bodies, each with 2 geoms (16 geoms) colliding against plane
  xml_nc16 = """<mujoco><compiler angle="radian"/><option integrator="Euler"/><worldbody>
  <geom type="plane" size="5 5 0.1"/>"""
  for i in range(8):
    xml_nc16 += f"""<body pos="{i} 0 1"><joint type="slide" axis="0 0 1"/>
    <geom type="sphere" size="0.1" pos="0 0 0" conaffinity="0"/>
    <geom type="sphere" size="0.1" pos="0 0 0.5" conaffinity="0"/>
    </body>"""
  xml_nc16 += "</worldbody></mujoco>"
  d16 = lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_nc16))
  assert d16.nc == 16

  xml_nc17 = xml_nc16.replace("</worldbody></mujoco>", """<body pos="10 0 1"><joint type="slide" axis="0 0 1"/>
  <geom type="sphere" size="0.1" pos="0 0 0" conaffinity="0"/>
  </body></worldbody></mujoco>""")
  with pytest.raises(ValueError, match="candidate contact pairs \\(17\\) exceeds capacity 16"):
    lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_nc17))

  # 3. nr boundary: nr=96 accepted, nr=97 rejected
  xml_nr96 = xml_nc16.replace("</worldbody></mujoco>", "</worldbody><equality>")
  for i in range(8):
    xml_nr96 = xml_nr96.replace(f'<body pos="{i} 0 1"><joint type="slide"', f'<body pos="{i} 0 1"><joint name="j{i}" type="slide"')
    xml_nr96 += f'<joint joint1="j{i}" polycoef="0 1 0 0 0"/>'
  xml_nr96 += "</equality></mujoco>"
  d96 = lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_nr96))
  assert d96.nr == 96

  xml_nr97 = xml_nr96.replace("</equality></mujoco>", '<joint joint1="j0" joint2="j1" polycoef="0 1 0 0 0"/></equality></mujoco>')
  with pytest.raises(ValueError, match="total candidate constraint rows \\(97\\) exceeds capacity 96"):
    lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_nr97))


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

  # Case 1: CONTACT disable flag with active contact
  # Ball penetrating floor at z=0 (center at 0.09, radius 0.1, ball_z = -1.41)
  m_base = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  q0_contact = np.array([[0.0, 0.0, -1.41]], dtype=np.float32)
  sim_base = MetalSimulation(m_base, batch_size=1, qpos=q0_contact, profile="integrated_euler_v1")

  m_dis = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_dis.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  sim_dis = MetalSimulation(m_dis, batch_size=1, qpos=q0_contact, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m_dis)
  d_cpu.qpos[:] = q0_contact[0]

  for _ in range(50):
    sim_base.step(1)
    sim_dis.step(1)
    mujoco.mj_step(m_dis, d_cpu)

  assert sim_base.state.status.item() == 0
  assert sim_dis.state.status.item() == 0
  q_base = sim_base.state.qpos[0].cpu().numpy()
  q_dis = sim_dis.state.qpos[0].cpu().numpy()
  assert abs(q_base[2] - q_dis[2]) > 0.05, f"Contact disable had no physical effect: {abs(q_base[2] - q_dis[2])}"
  np.testing.assert_allclose(q_dis, d_cpu.qpos, atol=1e-5)

  # Case 2: LIMIT disable flag with active limit
  # Both levers at limit 0.8 with positive velocity and torque
  q0_limit = np.array([[0.70, 0.84, 0.0]], dtype=np.float32)
  v0_limit = np.array([[1.0, 1.2, 0.0]], dtype=np.float32)
  ctrl_limit = np.array([[5.0]], dtype=np.float32)
  sim_base = MetalSimulation(m_base, batch_size=1, qpos=q0_limit, qvel=v0_limit, profile="integrated_euler_v1")

  m_dis = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_dis.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
  sim_dis = MetalSimulation(m_dis, batch_size=1, qpos=q0_limit, qvel=v0_limit, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m_dis)
  d_cpu.qpos[:] = q0_limit[0]
  d_cpu.qvel[:] = v0_limit[0]

  for _ in range(40):
    sim_base.step(1, ctrl=ctrl_limit)
    sim_dis.step(1, ctrl=ctrl_limit)
    d_cpu.ctrl[0] = ctrl_limit[0, 0]
    mujoco.mj_step(m_dis, d_cpu)

  assert sim_base.state.status.item() == 0
  assert sim_dis.state.status.item() == 0
  q_base = sim_base.state.qpos[0].cpu().numpy()
  q_dis = sim_dis.state.qpos[0].cpu().numpy()
  assert abs(q_base[0] - q_dis[0]) > 0.05, f"Limit disable had no physical effect: {abs(q_base[0] - q_dis[0])}"
  np.testing.assert_allclose(q_dis, d_cpu.qpos, atol=1e-5)

  # Case 3: EQUALITY disable flag with active equality violation
  q0_eq = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
  sim_base = MetalSimulation(m_base, batch_size=1, qpos=q0_eq, profile="integrated_euler_v1")

  m_dis = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_dis.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
  sim_dis = MetalSimulation(m_dis, batch_size=1, qpos=q0_eq, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m_dis)
  d_cpu.qpos[:] = q0_eq[0]

  for _ in range(30):
    sim_base.step(1)
    sim_dis.step(1)
    mujoco.mj_step(m_dis, d_cpu)

  assert sim_base.state.status.item() == 0
  assert sim_dis.state.status.item() == 0
  q_base = sim_base.state.qpos[0].cpu().numpy()
  q_dis = sim_dis.state.qpos[0].cpu().numpy()
  assert abs(q_base[1] - q_dis[1]) > 0.05, f"Equality disable had no physical effect: {abs(q_base[1] - q_dis[1])}"
  np.testing.assert_allclose(q_dis, d_cpu.qpos, atol=1e-5)

  # Case 4: ACTUATION disable flag
  sim_base = MetalSimulation(m_base, batch_size=1, profile="integrated_euler_v1")
  m_dis = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_dis.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  sim_dis = MetalSimulation(m_dis, batch_size=1, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m_dis)
  ctrl = np.array([[5.0]], dtype=np.float32)

  for _ in range(80):
    sim_base.step(1, ctrl=ctrl)
    sim_dis.step(1, ctrl=ctrl)
    d_cpu.ctrl[0] = ctrl[0, 0]
    mujoco.mj_step(m_dis, d_cpu)

  assert sim_base.state.status.item() == 0
  assert sim_dis.state.status.item() == 0
  q_base = sim_base.state.qpos[0].cpu().numpy()
  q_dis = sim_dis.state.qpos[0].cpu().numpy()
  assert abs(q_base[0] - q_dis[0]) > 0.05, f"Actuation disable had no physical effect: {abs(q_base[0] - q_dis[0])}"
  np.testing.assert_allclose(q_dis, d_cpu.qpos, atol=1e-5)

  # Case 5: DAMPER disable flag with active velocities
  v0_damp = np.array([[3.0, 3.0, 0.0]], dtype=np.float32)
  sim_base = MetalSimulation(m_base, batch_size=1, qvel=v0_damp, profile="integrated_euler_v1")
  m_dis = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_dis.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  sim_dis = MetalSimulation(m_dis, batch_size=1, qvel=v0_damp, profile="integrated_euler_v1")
  d_cpu = mujoco.MjData(m_dis)
  d_cpu.qvel[:] = v0_damp[0]

  for _ in range(40):
    sim_base.step(1)
    sim_dis.step(1)
    mujoco.mj_step(m_dis, d_cpu)

  assert sim_base.state.status.item() == 0
  assert sim_dis.state.status.item() == 0
  v_base = sim_base.state.qvel[0].cpu().numpy()
  v_dis = sim_dis.state.qvel[0].cpu().numpy()
  assert abs(v_base[0] - v_dis[0]) > 0.05, f"Damper disable had no physical effect: {abs(v_base[0] - v_dis[0])}"
  np.testing.assert_allclose(v_dis, d_cpu.qvel, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_control_clipping():
  import torch

  # Actuator in INTEGRATED_XML has ctrlrange="-5 5"
  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)

  # 1. Exceeding positive bound ctrl=10.0 is clipped to 5.0
  sim10 = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim5 = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  d_cpu10 = mujoco.MjData(model)
  ctrl10 = np.array([[10.0]], dtype=np.float32)
  ctrl5 = np.array([[5.0]], dtype=np.float32)

  for _ in range(30):
    sim10.step(1, ctrl=ctrl10)
    sim5.step(1, ctrl=ctrl5)
    d_cpu10.ctrl[0] = 10.0
    mujoco.mj_step(model, d_cpu10)

  q10 = sim10.state.qpos[0].cpu().numpy()
  q5 = sim5.state.qpos[0].cpu().numpy()
  np.testing.assert_allclose(q10, q5, atol=1e-6)
  np.testing.assert_allclose(q10, d_cpu10.qpos, atol=1e-5)

  # 2. Exceeding negative bound ctrl=-12.0 is clipped to -5.0
  sim_neg12 = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim_neg5 = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  d_cpu_neg = mujoco.MjData(model)
  ctrl_neg12 = np.array([[-12.0]], dtype=np.float32)
  ctrl_neg5 = np.array([[-5.0]], dtype=np.float32)

  for _ in range(30):
    sim_neg12.step(1, ctrl=ctrl_neg12)
    sim_neg5.step(1, ctrl=ctrl_neg5)
    d_cpu_neg.ctrl[0] = -12.0
    mujoco.mj_step(model, d_cpu_neg)

  q_neg12 = sim_neg12.state.qpos[0].cpu().numpy()
  q_neg5 = sim_neg5.state.qpos[0].cpu().numpy()
  np.testing.assert_allclose(q_neg12, q_neg5, atol=1e-6)
  np.testing.assert_allclose(q_neg12, d_cpu_neg.qpos, atol=1e-5)

  # 3. mjDSBL_CLAMPCTRL unclips control, producing distinct motion matching CPU
  m_noclamp = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_noclamp.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
  sim_noclamp = MetalSimulation(m_noclamp, batch_size=1, profile="integrated_euler_v1")
  d_cpu_noclamp = mujoco.MjData(m_noclamp)

  for _ in range(50):
    sim_noclamp.step(1, ctrl=ctrl10)
    d_cpu_noclamp.ctrl[0] = 10.0
    mujoco.mj_step(m_noclamp, d_cpu_noclamp)

  q_noclamp = sim_noclamp.state.qpos[0].cpu().numpy()
  np.testing.assert_allclose(q_noclamp, d_cpu_noclamp.qpos, atol=1e-5)
  assert abs(q_noclamp[0] - q10[0]) > 0.02, "Unclamped control should differ from clamped control"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_assembled_system_matches_cpu():
  import torch

  model = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  q0 = np.array([[0.5, 0.0, -1.41]], dtype=np.float32)
  v0 = np.array([[0.2, -0.1, 0.05]], dtype=np.float32)

  sim = MetalSimulation(model, batch_size=1, qpos=q0, qvel=v0, profile="integrated_euler_v1")
  sim.step(1)

  d_cpu = mujoco.MjData(model)
  d_cpu.qpos[:] = q0[0]
  d_cpu.qvel[:] = v0[0]
  mujoco.mj_forward(model, d_cpu)

  # Assembled system comparisons
  w = sim._coupled_constraints._workspace
  metal_qacc = w["out_acc"][:model.nv].cpu().numpy()
  metal_qfrc = w["out_force"][:model.nv].cpu().numpy()

  np.testing.assert_allclose(metal_qacc, d_cpu.qacc, rtol=1e-3, atol=1e-2)
  np.testing.assert_allclose(metal_qfrc, d_cpu.qfrc_constraint, rtol=1e-3, atol=1e-2)

  # Verify assembled contact Jacobian non-zero rows match contact normals
  c_mask = (d_cpu.efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
  J_contact_cpu = d_cpu.efc_J.reshape(d_cpu.nefc, model.nv)[c_mask]
  nc = sim._coupled_constraints.descriptor.nc
  J_contact_metal = w["contact_jacobian"][:nc * 5 * model.nv].cpu().numpy().reshape(-1, model.nv)
  active_metal_normals = J_contact_metal[np.any(J_contact_metal != 0, axis=1)]
  assert len(active_metal_normals) > 0
  np.testing.assert_allclose(active_metal_normals[0], J_contact_cpu[0], atol=1e-5)


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


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_independent_row_recovery_and_replay():
  import torch

  xml = """<mujoco>
  <option timestep="0.002" integrator="Euler" iterations="2" tolerance="1e-6"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.05">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" friction="1 0.1 0.1"/>
    </body>
    <body pos="0 1 0.15">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" friction="1 0.1 0.1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
  </equality>
</mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)

  q0 = np.array([
      [5.0, 5.0],    # World 0: benign state (succeeds)
      [0.0, 0.0],    # World 1: colliding state (exhausts iterations)
  ], dtype=np.float32)

  sim = MetalSimulation(m, batch_size=2, qpos=q0, profile="integrated_euler_v1")

  # Step 1: world 0 succeeds, world 1 fails with status 3
  sim.step(1)
  status1 = sim.state.status.cpu().numpy()
  assert status1[0] == 0, f"World 0 should succeed, got {status1[0]}"
  assert status1[1] == 3, f"World 1 should exhaust iterations (status 3), got {status1[1]}"

  # Verify world 0 advanced while world 1 rolled back
  qpos1 = sim.state.qpos.cpu().numpy()
  assert abs(qpos1[0, 0] - 5.0) > 1e-6, "World 0 should advance"
  np.testing.assert_allclose(qpos1[1], q0[1], atol=1e-6)

  # Step 2: failure remains sticky on world 1, world 0 continues advancing
  sim.step(1)
  status2 = sim.state.status.cpu().numpy()
  assert status2[0] == 0
  assert status2[1] == 3

  # Reset World 1 to replacement valid state [6.0, 6.0]
  sim.state.reset(env_ids=[1], qpos=[[6.0, 6.0]], qvel=[[0.0, 0.0]])

  status_reset = sim.state.status.cpu().numpy()
  assert np.all(status_reset == 0)

  # Take snapshot after reset
  chk = sim.state.snapshot()

  # Step 5 more steps: World 1 now succeeds and advances cleanly with status 0, while World 0 continues cleanly!
  for step in range(5):
    sim.step(1)
    st = sim.state.status.cpu().numpy()
    assert np.all(st == 0), f"Step {step} failed: {st}"

  qpos_stepped = sim.state.qpos.clone()
  assert not np.allclose(sim.state.qpos[1].cpu().numpy(), [6.0, 6.0])

  # Replay: restore checkpoint and replay exactly 5 steps
  sim.state.restore(chk)
  for _ in range(5):
    sim.step(1)

  assert torch.equal(sim.state.qpos, qpos_stepped), "Replay must reproduce exact state trajectory"


