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

  # If running in GPU mode, execute the accepted capacity boundaries on MPS
  if os.getenv("MUJOCO_METAL_RUN_GPU") == "1":
    _run_gpu_capacity_boundary_qualification()


def _run_gpu_capacity_boundary_qualification():
  """Execute accepted capacity boundaries nv=32, nc=16, nr=96 on GPU."""
  # 1. nv=32 boundary execution on GPU
  xml_nv32 = """<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="Euler"/><worldbody>"""
  for i in range(32):
    xml_nv32 += f"""<body pos="{i} 0 0"><joint type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1" conaffinity="0"/></body>"""
  xml_nv32 += "</worldbody></mujoco>"
  m32 = mujoco.MjModel.from_xml_string(xml_nv32)
  q0_32 = np.zeros((2, 32), dtype=np.float32)
  q0_32[0] = 0.02 * np.arange(32, dtype=np.float32)
  q0_32[1] = -0.01 * np.arange(32, dtype=np.float32)
  sim32 = MetalSimulation(m32, batch_size=2, qpos=q0_32, profile="integrated_euler_v1")
  sim32.step(3)
  st32 = sim32.state.status.cpu().numpy()
  qp32 = sim32.state.qpos.cpu().numpy()
  assert np.all(st32 == 0), f"nv=32 GPU step failed: {st32}"
  assert np.all(np.isfinite(qp32)), "nv=32 outputs must be finite float32"
  assert np.max(np.abs(qp32[0] - qp32[1])) > 0.01, "nv=32 worlds must be isolated"
  d_cpu32 = mujoco.MjData(m32)
  d_cpu32.qpos[:] = q0_32[0]
  for _ in range(3):
    mujoco.mj_step(m32, d_cpu32)
  np.testing.assert_allclose(qp32[0], d_cpu32.qpos, atol=1e-5)

  # 2. nc=16 boundary execution on GPU (16 candidate contact pairs, 8 active contacts)
  xml_nc16 = """<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="Euler" iterations="100" tolerance="1e-5"/><worldbody>
  <geom type="plane" size="10 10 0.1"/>"""
  for i in range(8):
    xml_nc16 += f"""<body pos="{i} 0 0.08"><joint type="slide" axis="0 0 1"/>
    <geom type="sphere" size="0.1" pos="0 0 0" friction="0.8 0.1 0.1" conaffinity="0"/>
    <geom type="sphere" size="0.1" pos="0 0 0.5" friction="0.8 0.1 0.1" conaffinity="0"/>
    </body>"""
  xml_nc16 += "</worldbody></mujoco>"
  m16 = mujoco.MjModel.from_xml_string(xml_nc16)
  q0_16 = np.zeros((2, 8), dtype=np.float32)
  q0_16[0] = -0.005 * np.ones(8, dtype=np.float32)
  q0_16[1] = -0.010 * np.ones(8, dtype=np.float32)
  sim16 = MetalSimulation(m16, batch_size=2, qpos=q0_16, profile="integrated_euler_v1")
  sim16.step(1)
  st16 = sim16.state.status.cpu().numpy()
  qp16 = sim16.state.qpos.cpu().numpy()
  qa16 = sim16.state.qacc.cpu().numpy()
  assert np.all(st16 == 0), f"nc=16 GPU step failed: {st16}"
  assert np.all(np.isfinite(qp16)) and np.all(np.isfinite(qa16)), "nc=16 outputs must be finite float32"
  assert np.max(np.abs(qp16[0] - qp16[1])) > 0.001, "nc=16 worlds must be isolated"
  d_cpu16 = mujoco.MjData(m16)
  d_cpu16.qpos[:] = q0_16[0]
  mujoco.mj_step(m16, d_cpu16)
  np.testing.assert_allclose(qa16[0], d_cpu16.qacc, rtol=1e-3, atol=1e-2)

  # 3. nr=96 boundary execution on GPU exercising row slot 95 and maximum workspace extents
  # Note: not all 96 candidate rows can be physically active simultaneously because
  # joints cannot simultaneously violate mutually exclusive upper and lower limits.
  # We explicitly exercise row slot 95 (the 96th row) via active contact pair 15.
  xml_nr96_base = """<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="Euler" iterations="400" tolerance="1e-4"/><worldbody>
  <geom type="plane" size="10 10 0.1"/>"""
  for i in range(8):
    xml_nr96_base += f"""<body pos="{i} 0 0.08"><joint name="j{i}" type="slide" axis="0 0 1" range="-0.6 0.5" limited="true" frictionloss="0.05"/>
    <geom type="sphere" size="0.1" pos="0 0 0" friction="0.8 0.1 0.1" conaffinity="0"/>
    <geom type="sphere" size="0.1" pos="0 0 0.5" friction="0.8 0.1 0.1" conaffinity="0"/>
    </body>"""
  xml_nr96_base += "</worldbody><equality>"
  xml_nr96 = xml_nr96_base
  for i in range(8):
    xml_nr96 += f"""<joint joint1="j{i}" polycoef="0 1 0 0 0"/>"""
  xml_nr96 += "</equality></mujoco>"
  m96 = mujoco.MjModel.from_xml_string(xml_nr96)
  q0_96 = np.zeros((2, 8), dtype=np.float32)
  q0_96[0, :7] = 0.0
  q0_96[1, :7] = 0.005
  q0_96[0, 7] = -0.50
  q0_96[1, 7] = -0.49
  sim96 = MetalSimulation(m96, batch_size=2, qpos=q0_96, profile="integrated_euler_v1")
  assembled96 = sim96.assembled_system()
  sim96.step(1)
  st96 = sim96.state.status.cpu().numpy()
  qp96 = sim96.state.qpos.cpu().numpy()
  qa96 = sim96.state.qacc.cpu().numpy()
  assert np.all(st96 == 0), f"nr=96 GPU step failed: {st96}"
  assert np.all(np.isfinite(qp96)) and np.all(np.isfinite(qa96)), "nr=96 outputs must be finite float32"
  assert np.max(np.abs(qp96[0] - qp96[1])) > 0.001, "nr=96 worlds must be isolated"
  # Exercise row slot 95 (the 96th row)
  J96 = assembled96["J"].cpu().numpy()
  W96 = assembled96["W"].cpu().numpy()
  assert np.linalg.norm(J96[0, 95]) > 0, "Row slot 95 must be active in World 0"
  assert np.linalg.norm(J96[1, 95]) > 0, "Row slot 95 must be active in World 1"
  assert W96[0, 95, 95] > 0, "Row slot 95 diagonal Delassus must be positive"
  # Verify against CPU reference
  d_cpu96 = mujoco.MjData(m96)
  d_cpu96.qpos[:] = q0_96[0]
  mujoco.mj_step(m96, d_cpu96)
  np.testing.assert_allclose(qa96[0], d_cpu96.qacc, rtol=1e-3, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_gpu_capacity_boundaries_execution():
  _run_gpu_capacity_boundary_qualification()


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

  # 3. Same-time physical effect of mjDSBL_CLAMPCTRL:
  # Run both clamped and unclamped simulations from identical initial states for the
  # SAME step count (50 steps) and control sequence (ctrl=10.0, where ctrlrange is [-5, 5]).
  m_noclamp = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  m_noclamp.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
  sim_noclamp = MetalSimulation(m_noclamp, batch_size=1, profile="integrated_euler_v1")
  d_cpu_noclamp = mujoco.MjData(m_noclamp)

  sim_clamp = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  d_cpu_clamp = mujoco.MjData(model)

  for _ in range(50):
    sim_noclamp.step(1, ctrl=ctrl10)
    sim_clamp.step(1, ctrl=ctrl10)
    d_cpu_noclamp.ctrl[0] = 10.0
    d_cpu_clamp.ctrl[0] = 10.0
    mujoco.mj_step(m_noclamp, d_cpu_noclamp)
    mujoco.mj_step(model, d_cpu_clamp)

  assert sim_noclamp.state.status[0].item() == 0, "Unclamped simulation should succeed with status 0"
  assert sim_clamp.state.status[0].item() == 0, "Clamped simulation should succeed with status 0"

  q_noclamp = sim_noclamp.state.qpos[0].cpu().numpy()
  q_clamp = sim_clamp.state.qpos[0].cpu().numpy()

  np.testing.assert_allclose(q_noclamp, d_cpu_noclamp.qpos, atol=1e-5)
  np.testing.assert_allclose(q_clamp, d_cpu_clamp.qpos, atol=1e-5)

  measured_diff = abs(float(q_noclamp[0] - q_clamp[0]))
  # Measured physical difference at identical elapsed time t=0.1s is ~0.03632 > 0.02
  assert measured_diff > 0.02, f"Expected same-time clipping effect > 0.02, got {measured_diff}"
  np.testing.assert_allclose(measured_diff, 0.0363218, rtol=1e-3, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_integrated_simulation_assembled_system_matches_cpu():
  import torch

  # -------------------------------------------------------------------------
  # Part A: Deterministic mixed-contact/joint fixture with nontrivial 3D
  # rotational and tangential Jacobians (not purely axis-aligned)
  # -------------------------------------------------------------------------
  xml_mixed = """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler" iterations="1000" tolerance="1e-6"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="body1" pos="0 0 0.5">
      <joint name="j1" type="hinge" pos="0 0 0" axis="0.6 0.8 0" range="-0.2 0.5" margin="0.05" limited="true" frictionloss="0.1"/>
      <geom name="geom1" type="sphere" size="0.1" pos="0 0 0.2"/>
      <body name="body2" pos="0.2 0.1 -0.2">
        <joint name="j2" type="hinge" pos="0 0 0" axis="0.8 -0.6 0" range="-0.3 0.5" margin="0.05" limited="true" frictionloss="0.05"/>
        <geom name="geom2" type="sphere" size="0.18" pos="0 -0.1 -0.17" friction="0.8 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1.5"/>
      <joint joint="j2" coef="-0.8"/>
    </fixed>
  </tendon>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0.02 0.5 0 0 0"/>
  </equality>
</mujoco>"""

  m_mixed = mujoco.MjModel.from_xml_string(xml_mixed)
  q0_mixed = np.array([[-0.18, 0.05]], dtype=np.float32)
  v0_mixed = np.array([[0.1, -0.2]], dtype=np.float32)

  sim_mixed = MetalSimulation(m_mixed, batch_size=1, qpos=q0_mixed, qvel=v0_mixed, profile="integrated_euler_v1")
  assembled_mixed = sim_mixed.assembled_system()

  d_cpu_mixed = mujoco.MjData(m_mixed)
  d_cpu_mixed.qpos[:] = q0_mixed[0]
  d_cpu_mixed.qvel[:] = v0_mixed[0]
  mujoco.mj_forward(m_mixed, d_cpu_mixed)

  J_gpu = assembled_mixed["J"][0].cpu().numpy()
  W_gpu = assembled_mixed["W"][0].cpu().numpy()
  W_reg_gpu = assembled_mixed["W_regularized"][0].cpu().numpy()
  R_gpu = assembled_mixed["R"][0].cpu().numpy()
  ar_gpu = assembled_mixed["ar"][0].cpu().numpy()
  rhs_gpu = assembled_mixed["rhs"][0].cpu().numpy()
  lam_gpu = assembled_mixed["lambda"][0].cpu().numpy()
  M_gpu = assembled_mixed["mass_matrix"][0].cpu().numpy()

  # 1. Effective mass comparison (including tendon armature assembled separately as J^T armature J)
  M_cpu = np.zeros((m_mixed.nv, m_mixed.nv), dtype=np.float64)
  mujoco.mj_fullM(m_mixed, d_cpu_mixed, M_cpu)
  max_M_err = np.max(np.abs(M_gpu - M_cpu))
  assert max_M_err < 1e-6, f"Effective mass mismatch: {max_M_err}"

  # 2. Map active CPU rows to GPU rows:
  # CPU row 0: equality (eq 0) -> GPU row 0
  # CPU row 1: frictionloss DOF 0 -> GPU row 1
  # CPU row 2: frictionloss DOF 1 -> GPU row 2
  # CPU row 3: joint limit j1 lower -> GPU row 3 (neq + nv + 2*0 = 1 + 2 + 0 = 3)
  # CPU rows 4..7: pyramidal contact edges between floor and geom2 (candidate pair 1) -> GPU rows 11, 12, 13, 14
  mapped_rows = [0, 1, 2, 3, 11, 12, 13, 14]
  J_cpu = d_cpu_mixed.efc_J.reshape(d_cpu_mixed.nefc, m_mixed.nv)
  R_cpu = d_cpu_mixed.efc_R
  ar_cpu = d_cpu_mixed.efc_aref
  rhs_cpu = -d_cpu_mixed.efc_b

  # Compare active Jacobian rows (nontrivial 3D rotational and tangential Jacobians)
  max_J_err = np.max(np.abs(J_gpu[mapped_rows] - J_cpu))
  assert max_J_err < 1e-6, f"Jacobian mismatch: {max_J_err}"
  # Ensure contact rows have non-zero components in both DOFs (tangential & rotational coupling)
  contact_J = J_gpu[11:15]
  assert np.all(np.abs(contact_J[:, 0]) > 0.05), "DOF 0 contact coupling must be non-zero"
  assert np.all(np.abs(contact_J[:, 1]) > 0.05), "DOF 1 contact coupling must be non-zero"

  # Compare regularizer R
  max_R_err = np.max(np.abs(R_gpu[mapped_rows] - R_cpu))
  assert max_R_err < 1e-6, f"Regularizer mismatch: {max_R_err}"

  # Compare acceleration reference ar and RHS
  max_ar_err = np.max(np.abs(ar_gpu[mapped_rows] - ar_cpu))
  max_rhs_err = np.max(np.abs(rhs_gpu[mapped_rows] - rhs_cpu))
  assert max_ar_err < 1e-3, f"aref mismatch: {max_ar_err}"
  assert max_rhs_err < 1e-3, f"rhs mismatch: {max_rhs_err}"

  # Compare Delassus matrix W = J M^-1 J^T + R
  W_sub = W_reg_gpu[np.ix_(mapped_rows, mapped_rows)]
  W_expected = J_cpu @ np.linalg.inv(M_cpu) @ J_cpu.T + np.diag(R_cpu)
  max_W_err = np.max(np.abs(W_sub - W_expected))
  assert max_W_err < 1e-6, f"Assembled Delassus mismatch: {max_W_err}"

  # Verify nonzero off-diagonal cross-coupling block W_cj between joint rows (0..3) and contact rows (4..7)
  cross_block = W_sub[4:8, 0:4]
  cross_norm = float(np.linalg.norm(cross_block))
  assert cross_norm > 0.1, f"Cross-coupling block must be non-trivial, got norm {cross_norm}"

  # Verify inactive rows in GPU matrices and vectors are all exactly 0
  inactive_rows = [r for r in range(sim_mixed._coupled_constraints.descriptor.nr) if r not in mapped_rows]
  assert len(inactive_rows) > 0
  for r in inactive_rows:
    assert np.all(J_gpu[r] == 0), f"Inactive J[{r}] must be 0"
    assert R_gpu[r] == 0, f"Inactive R[{r}] must be 0"
    assert ar_gpu[r] == 0, f"Inactive ar[{r}] must be 0"
    assert rhs_gpu[r] == 0, f"Inactive rhs[{r}] must be 0"
    assert np.all(W_gpu[r] == 0), f"Inactive W[{r}, :] must be 0"
    assert np.all(W_gpu[:, r] == 0), f"Inactive W[:, {r}] must be 0"

  # Step on GPU and verify step completion, status 0, and KKT complementarity
  sim_mixed.step(1)
  assert sim_mixed.state.status[0].item() == 0, "GPU stepping must succeed with status 0"
  np.testing.assert_allclose(sim_mixed.state.qacc[0].cpu().numpy(), d_cpu_mixed.qacc, rtol=1e-3, atol=1e-2)
  np.testing.assert_allclose(sim_mixed._last_coupled["qfrc_constraint"][0].cpu().numpy(), d_cpu_mixed.qfrc_constraint, rtol=1e-3, atol=1e-2)

  # Independent KKT complementarity and projected gradient residual check evaluated on host CPU from device outputs:
  lam_sub = lam_gpu[mapped_rows]
  grad = W_sub @ lam_sub - rhs_cpu
  lo = np.array([-np.inf, -0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
  hi = np.array([np.inf, 0.1, 0.05, np.inf, np.inf, np.inf, np.inf, np.inf])
  diag_W = np.diag(W_sub)
  proj = np.clip(lam_sub - grad / diag_W, lo, hi)
  row_scale = np.maximum(1.0, np.abs(rhs_cpu) + np.abs(W_sub @ lam_sub))
  kkt_res = np.max(np.abs(proj - lam_sub) * diag_W / row_scale)
  assert kkt_res <= 1e-4, f"KKT projected residual exceeds tolerance: {kkt_res}"

  # -------------------------------------------------------------------------
  # Part B: Axis-aligned fixture (INTEGRATED_XML)
  # -------------------------------------------------------------------------
  model_aa = mujoco.MjModel.from_xml_string(INTEGRATED_XML)
  q0_aa = np.array([[0.5, 0.0, -1.41]], dtype=np.float32)
  v0_aa = np.array([[0.2, -0.1, 0.05]], dtype=np.float32)

  sim_aa = MetalSimulation(model_aa, batch_size=1, qpos=q0_aa, qvel=v0_aa, profile="integrated_euler_v1")
  assembled_aa = sim_aa.assembled_system()

  d_cpu_aa = mujoco.MjData(model_aa)
  d_cpu_aa.qpos[:] = q0_aa[0]
  d_cpu_aa.qvel[:] = v0_aa[0]
  mujoco.mj_forward(model_aa, d_cpu_aa)

  J_aa_gpu = assembled_aa["J"][0].cpu().numpy()
  W_aa_gpu = assembled_aa["W"][0].cpu().numpy()
  W_aa_reg = assembled_aa["W_regularized"][0].cpu().numpy()
  R_aa_gpu = assembled_aa["R"][0].cpu().numpy()
  ar_aa_gpu = assembled_aa["ar"][0].cpu().numpy()
  rhs_aa_gpu = assembled_aa["rhs"][0].cpu().numpy()
  M_aa_gpu = assembled_aa["mass_matrix"][0].cpu().numpy()

  M_aa_cpu = np.zeros((model_aa.nv, model_aa.nv), dtype=np.float64)
  mujoco.mj_fullM(model_aa, d_cpu_aa, M_aa_cpu)
  np.testing.assert_allclose(M_aa_gpu, M_aa_cpu, atol=1e-6)

  # Map active CPU rows for axis-aligned model:
  # CPU row 0: equality -> GPU row 0
  # CPU rows 1, 2: frictionloss -> GPU rows 1, 2
  # CPU rows 3..6: contact pair 2 (floor vs sphere, base_contact=10 + 4*2 = 18) -> GPU rows 18..21
  mapped_aa = [0, 1, 2, 18, 19, 20, 21]
  J_aa_cpu = d_cpu_aa.efc_J.reshape(d_cpu_aa.nefc, model_aa.nv)
  R_aa_cpu = d_cpu_aa.efc_R
  ar_aa_cpu = d_cpu_aa.efc_aref
  rhs_aa_cpu = -d_cpu_aa.efc_b

  np.testing.assert_allclose(J_aa_gpu[mapped_aa], J_aa_cpu, atol=1e-6)
  np.testing.assert_allclose(R_aa_gpu[mapped_aa], R_aa_cpu, atol=1e-6)
  np.testing.assert_allclose(ar_aa_gpu[mapped_aa], ar_aa_cpu, atol=1e-3)
  np.testing.assert_allclose(rhs_aa_gpu[mapped_aa], rhs_aa_cpu, atol=1e-3)

  W_aa_sub = W_aa_reg[np.ix_(mapped_aa, mapped_aa)]
  W_aa_expected = J_aa_cpu @ np.linalg.inv(M_aa_cpu) @ J_aa_cpu.T + np.diag(R_aa_cpu)
  np.testing.assert_allclose(W_aa_sub, W_aa_expected, atol=1e-6)

  inactive_aa = [r for r in range(sim_aa._coupled_constraints.descriptor.nr) if r not in mapped_aa]
  for r in inactive_aa:
    assert np.all(J_aa_gpu[r] == 0)
    assert R_aa_gpu[r] == 0
    assert ar_aa_gpu[r] == 0
    assert rhs_aa_gpu[r] == 0
    assert np.all(W_aa_gpu[r] == 0)
    assert np.all(W_aa_gpu[:, r] == 0)

  sim_aa.step(1)
  assert sim_aa.state.status[0].item() == 0
  np.testing.assert_allclose(sim_aa.state.qacc[0].cpu().numpy(), d_cpu_aa.qacc, rtol=1e-3, atol=1e-2)
  np.testing.assert_allclose(sim_aa._last_coupled["qfrc_constraint"][0].cpu().numpy(), d_cpu_aa.qfrc_constraint, rtol=1e-3, atol=1e-2)


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


