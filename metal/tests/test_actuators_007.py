# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actuator dynamics/transmissions qualification (CPU admission + GPU parity)."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.stateful_actuation import ActuatorModel
from mujoco_metal.stateful_actuation import act_dot_reference
from mujoco_metal.stateful_actuation import advance_activation_reference
from mujoco_metal.stateful_actuation import force_reference

HINGE_BODY = '<body pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/><geom type="sphere" size="0.1" mass="1"/></body>'


def _model(body=HINGE_BODY, actuator="", option='timestep="0.002"'):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option {option}/><worldbody>{body}</worldbody><actuator>{actuator}</actuator></mujoco>')


def test_admission_accepts_full_families_cpu():
  m = _model(actuator='<general joint="j" dyntype="filter" dynprm="0.05 0 0"/>')
  meta = ActuatorModel(m)
  assert meta.na == 1 and meta.needs_general_path
  m = _model(actuator='<general joint="j" dyntype="integrator"/><general joint="j" dyntype="filterexact" dynprm="0.05 0 0"/>')
  assert ActuatorModel(m).na == 2
  m = _model(actuator='<general joint="j" dyntype="muscle" dynprm="0.01 0.04 0.01" gaintype="muscle" gainprm="0.5 1.5 100 100 0.5 1.5 1 1 1.2 0" biastype="muscle" biasprm="0.5 1.5 100 100 0.5 1.5 1 1 1.2 0" lengthrange="0.5 1.5"/>')
  assert ActuatorModel(m).na == 1
  # DC motor with current + integral slots.
  m = _model(actuator='<general joint="j" dyntype="dcmotor" dynprm="0.01 0 0 0 0 0 0 0 1 0" gaintype="dcmotor" gainprm="1 1 0 0 0 1 0 0 1" biastype="dcmotor" actearly="true" actdim="2"/>')
  assert ActuatorModel(m).na == 2
  # Ball/free gears, slider-crank, site, body, tendon-fixed.
  ball = '<body pos="0 0 1"><joint name="b" type="ball"/><geom type="sphere" size="0.1" mass="1"/></body>'
  m = _model(body=ball, actuator='<general joint="b" gear="0 0 1"/>')
  ActuatorModel(m)
  free = '<body pos="0 0 1"><freejoint name="f"/><geom type="sphere" size="0.1" mass="1"/></body>'
  m = _model(body=free, actuator='<general joint="f" gear="1 0 0 0 0 1"/>')
  ActuatorModel(m)
  crank = ('<site name="s1" pos="0.1 0 1"/><site name="s2" pos="0 0 1"/>' + HINGE_BODY)
  m = _model(body=crank, actuator='<general cranksite="s1" slidersite="s2" gear="2" cranklength="0.05"/>')
  assert ActuatorModel(m).needs_general_path
  site = '<site name="s1" pos="0 0 1"/>' + HINGE_BODY
  m = _model(body=site, actuator='<general site="s1" gear="1 0 0 0 0 0"/>')
  ActuatorModel(m)
  m = _model(body='<body name="b" pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/><geom type="sphere" size="0.1" mass="1"/></body>',
             actuator='<general body="b"/>')
  assert ActuatorModel(m).has_body_transmission


def test_admission_rejects_with_owners_cpu():
  # MuJoCo's no-callback USER defaults are a supported source behavior;
  # registered device callbacks are separately validated by milestone 019.
  user_dyn = ActuatorModel(_model(
      actuator='<general joint="j" dyntype="user" actdim="2"/>'))
  assert user_dyn.na == 2
  assert user_dyn.needs_general_path
  user_gain = ActuatorModel(_model(
      actuator='<general joint="j" gaintype="user"/>'))
  assert user_gain.has_user_gain
  user_bias = ActuatorModel(_model(
      actuator='<general joint="j" biastype="user"/>'))
  assert user_bias.has_user_bias
  m = _model(actuator='<general joint="j"/>')
  m.actuator_delay[0] = 0.01
  # R06/D1: delay admitted with validated config (empty history here means
  # live reads, exactly like pinned nsample==0); malformed config raises.
  ActuatorModel(m)
  with pytest.raises(ValueError, match="015"):
    ActuatorModel(_model(actuator='<general joint="j" armature="0.1"/>'))
  m = _model(actuator='<general joint="j"/>')
  m.actuator_plugin[0] = 0
  with pytest.raises(ValueError, match="019"):
    ActuatorModel(m)
  with pytest.raises(ValueError, match="range"):
    ActuatorModel(_model(actuator='<general joint="j" ctrllimited="true" ctrlrange="1 -1"/>'))
  with pytest.raises(ValueError, match="group"):
    ActuatorModel(_model(actuator='<general joint="j" group="31"/>'))
  with pytest.raises(ValueError, match="crank"):
    ActuatorModel(_model(body='<site name="s1" pos="0 0 1"/><site name="s2" pos="0 0 1"/>' + HINGE_BODY,
                         actuator='<general cranksite="s1" slidersite="s2" cranklength="0"/>'))
  # Spatial tendon actuators are owned by 008 (R1): admitted via the
  # general path, not rejected.
  spat_world = ('<site name="s1" pos="0 0 1"/><site name="s2" pos="0 0 2"/>' + HINGE_BODY)
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep="0.002"/><worldbody>{spat_world}</worldbody>'
      '<tendon><spatial name="st"><site site="s1"/><site site="s2"/></spatial></tendon>'
      '<actuator><general tendon="st"/></actuator></mujoco>')
  meta = ActuatorModel(m)
  assert meta.needs_general_path


def test_reference_matches_mjstep_intermediates_cpu():
  m = _model(actuator='<general joint="j" dyntype="filter" dynprm="0.05 0 0" gainprm="2 0 0" biasprm="0.5 0 0"/>')
  meta = ActuatorModel(m)
  d = mujoco.MjData(m)
  d.ctrl[0] = 0.7
  mujoco.mj_forward(m, d)
  length = np.asarray(d.actuator_length)
  vel = np.asarray(d.actuator_velocity)
  ad = act_dot_reference(meta, np.array([0.7]), np.asarray(d.act), length, vel)
  np.testing.assert_allclose(ad, np.asarray(d.act_dot), rtol=1e-5, atol=1e-7)
  moment = np.asarray(d.actuator_moment).reshape(1, -1)
  fr = force_reference(meta, np.array([0.7]), np.asarray(d.act), length, vel, moment,
                       None, int(m.opt.disableflags), int(m.opt.disableactuator))
  np.testing.assert_allclose(fr["force"], np.asarray(d.actuator_force), rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(fr["qfrc"], np.asarray(d.qfrc_actuator), rtol=1e-5, atol=1e-7)
  adv = advance_activation_reference(meta, np.asarray(d.act), np.asarray(d.act_dot), vel)
  mujoco.mj_step(m, d)
  np.testing.assert_allclose(adv, np.asarray(d.act), rtol=1e-5, atol=1e-7)


def test_finite_difference_moment_matches_oracle_cpu():
  # Independent math: moment rows are d(length)/dq (no production code).
  body = ('<site name="s2" pos="0 0 1"/>'
          '<body pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/>'
          '<geom type="sphere" size="0.1" mass="1"/>'
          '<site name="s1" pos="0.1 0 0"/></body>')
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep="0.002"/><worldbody>{body}</worldbody>'
      '<actuator><general site="s1" refsite="s2" gear="0 0 1 0 0 0"/></actuator></mujoco>')
  d = mujoco.MjData(m)
  d.qpos[0] = 0.3
  mujoco.mj_forward(m, d)
  l0 = float(np.asarray(d.actuator_length)[0])
  assert abs(l0) > 1e-4  # non-degenerate fixture
  mom = np.zeros(m.nv)
  rownnz = np.asarray(d.moment_rownnz)
  rowadr = np.asarray(d.moment_rowadr)
  colind = np.asarray(d.moment_colind)
  vals = np.asarray(d.actuator_moment)
  for j in range(rownnz[0]):
    mom[colind[rowadr[0] + j]] = vals[rowadr[0] + j]
  assert np.any(mom != 0)
  h = 1e-6
  fd = np.zeros(m.nv)
  for k in range(m.nv):
    d.qpos[0] = 0.3
    d.qpos[k] += h
    mujoco.mj_forward(m, d)
    fd[k] = (float(np.asarray(d.actuator_length)[0]) - l0) / h
  np.testing.assert_allclose(mom, fd, rtol=1e-3, atol=1e-5)


def _dense_moment(m, d):
  out = np.zeros((m.nu, m.nv))
  rownnz = np.asarray(d.moment_rownnz)
  rowadr = np.asarray(d.moment_rowadr)
  colind = np.asarray(d.moment_colind)
  vals = np.asarray(d.actuator_moment)
  for i in range(m.nu):
    for j in range(rownnz[i]):
      out[i, colind[rowadr[i] + j]] = vals[rowadr[i] + j]
  return out


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_kinematics_native_parity_gpu():
  import torch
  from mujoco_metal import MetalSimulation
  # Note: pure scalar hinge/slide models take the pre-qualified fast path
  # (covered by test_transmissions.py); this matrix covers general-path types.
  hinge_site = ('<site name="s2" pos="0 0 1"/>' + HINGE_BODY.replace(
      '</body>', '<site name="s1" pos="0.1 0 0"/></body>', 1))
  cases = [
      ('<general joint="b" gear="0 0 1"/>',
       '<body pos="0 0 1"><joint name="b" type="ball"/><geom type="sphere" size="0.1" mass="1"/></body>'),
      ('<general joint="f" gear="1 0 0 0 0 1"/>',
       '<body pos="0 0 1"><freejoint name="f"/><geom type="sphere" size="0.1" mass="1"/></body>'),
      ('<general site="s1" gear="1 0 0 0 0 0"/>', hinge_site),
      ('<general site="s1" refsite="s2" gear="0 0 1 0 1 0"/>', hinge_site),
      ('<general cranksite="s1" slidersite="s2" gear="2" cranklength="0.05"/>', hinge_site),
  ]
  for act_xml, body in cases:
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><option timestep="0.002"/><worldbody>{body}</worldbody><actuator>{act_xml}</actuator></mujoco>')
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(m.qpos0) + np.linspace(0, 0.05, m.nq)
    mujoco.mj_forward(m, d)
    sim.state._qpos.copy_(torch.as_tensor(
        np.asarray(d.qpos, dtype=np.float32).reshape(1, -1), device=sim.state._device))
    kin = sim._actuators.run_kinematics(
        sim.state._qpos, sim.state._qvel, sim._smooth.run_device(
            sim.state._qpos, sim.state._qvel,
            getattr(sim.state, "_mpos", None), getattr(sim.state, "_mquat", None))["poses"])
    np.testing.assert_allclose(
        kin["length"].cpu().numpy()[0], np.asarray(d.actuator_length), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        kin["velocity"].cpu().numpy()[0], np.asarray(d.actuator_velocity), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        kin["moment"].cpu().numpy()[0], _dense_moment(m, d),
        rtol=2e-5, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_activation_trajectories_native_parity_gpu():
  from mujoco_metal import MetalSimulation
  actuators = [
      ('<general joint="j" dyntype="integrator"/>', 5e-4, 5e-4),
      ('<general joint="j" dyntype="filter" dynprm="0.05 0 0" gainprm="2 0 0" actlimited="true" actrange="-2 2"/>', 5e-4, 5e-4),
      ('<general joint="j" dyntype="filterexact" dynprm="0.05 0 0" gainprm="2 0 0"/>', 5e-4, 5e-4),
      ('<general joint="j" dyntype="muscle" dynprm="0.01 0.04 0.01" gaintype="muscle" gainprm="0.6 1.4 100 100 0.5 1.5 1 1 1.2 0" biastype="muscle" biasprm="0.6 1.4 100 100 0.5 1.5 1 1 1.2 0" lengthrange="-0.6 0.6"/>', 2e-3, 2e-3),
      ('<general joint="j" dyntype="dcmotor" dynprm="0.01 0 0 0 0 0 0 0 1 0" gaintype="dcmotor" gainprm="1 1 0 0 0 1 0 0 1" biastype="dcmotor" actearly="true" actdim="2"/>', 2e-3, 2e-3),
  ]
  for act_xml, tol_q, tol_a in actuators:
    m = _model(actuator=act_xml)
    sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
    a0 = np.zeros((2, m.na), dtype=np.float32)
    if m.na:
      a0[0, :] = 0.3
    sim.reset(act=a0)
    cpu = [mujoco.MjData(m), mujoco.MjData(m)]
    for k in range(2):
      cpu[k].act[:] = a0[k]
    max_q = 0.0
    max_a = 0.0
    for step in range(60):
      u = [0.8 if step < 30 else 0.1, -0.4]
      st = sim.step(1, ctrl=np.array(u, dtype=np.float32).reshape(2, 1))
      assert int(st.cpu().numpy()[0]) == 0 and int(st.cpu().numpy()[1]) == 0, (act_xml, step)
      for k in range(2):
        cpu[k].ctrl[0] = u[k]
        mujoco.mj_step(m, cpu[k])
      gq = sim.state.qpos.cpu().numpy()
      ga = sim.state.act.cpu().numpy()
      for k in range(2):
        max_q = max(max_q, float(np.max(np.abs(gq[k] - cpu[k].qpos))))
        max_a = max(max_a, float(np.max(np.abs(ga[k] - cpu[k].act))))
    assert max_q < tol_q, (act_xml, max_q)
    assert max_a < tol_a, (act_xml, max_a)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_device_buffer_dtypes_match_kernel_declarations_gpu():
  import torch
  from mujoco_metal import MetalSimulation
  # Regression: a uint8 device buffer read as Metal int overreads 3 bytes of
  # adjacent memory (milestone 007 actearly flake). All integer kernel buffers
  # must be int32.
  m = _model(actuator='<general joint="j" dyntype="filter" dynprm="0.05 0 0" actearly="true"/>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  assert sim._actuators is not None
  for name in ("_trntype", "_trnid", "_jnt_type", "_jnt_qposadr", "_jnt_dofadr",
               "_body_parentid", "_body_jntadr", "_body_jntnum", "_body_weldid",
               "_body_dofadr", "_body_dofnum", "_dof_parentid", "_site_bodyid",
               "_dyntype", "_gaintype", "_biastype", "_actadr", "_actnum",
               "_actearly", "_ctrllimited", "_forcelimited", "_tendon_limited",
               "_jnt_limited", "_jnt_gravcomp", "_group", "_actlimited",
               "_kin_dims", "_dot_dims", "_force_dims",
               "_qfrc_dims", "_adv_dims"):
    buf = getattr(sim._actuators, name, None)
    if buf is None:
      continue
    assert isinstance(buf, torch.Tensor) and buf.dtype == torch.int32, name
    assert buf.is_contiguous() and buf.device.type == "mps", name


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_body_adhesion_native_parity_gpu():
  from mujoco_metal import MetalSimulation
  xml = '''<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="box" pos="0 0 0.0501"><freejoint/><geom name="bg" type="box" size="0.05 0.05 0.05" mass="0.5"/></body>
    </worldbody>
    <contact><pair geom1="bg" geom2="floor"/></contact>
    <actuator><general body="box" gear="0 0 1 0 0 0"/></actuator></mujoco>'''
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  # Resting fixture (no impact transient): the pre-existing native contact
  # solver fails the dropped-box impact for all actuator variants alike, so
  # start settled and compare adhesion-loaded equilibrium.
  sim.reset(qpos=np.array([0, 0, 0.0501, 1, 0, 0, 0], dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = [0, 0, 0.0501, 1, 0, 0, 0]
  mujoco.mj_forward(m, cpu)
  for step in range(30):
    sim.step(1, ctrl=np.array([[2.0]], dtype=np.float32))
    cpu.ctrl[0] = 2.0
    mujoco.mj_step(m, cpu)
  assert sim.state.status.cpu().numpy()[0] == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq, np.asarray(cpu.qpos), rtol=1e-5, atol=5e-5)
  assert cpu.ncon > 0  # adhesion fixture genuinely in contact


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_limits_groups_actearly_native_parity_gpu():
  from mujoco_metal import MetalSimulation
  xml = f'''<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>{HINGE_BODY}
    <body pos="0 0 2"><joint name="j2" type="slide" axis="1 0 0"/><geom type="sphere" size="0.1" mass="1"/></body>
    </worldbody><actuator>
    <general joint="j" gear="3" ctrllimited="true" ctrlrange="-0.5 0.5" forcelimited="true" forcerange="-1 1" actearly="true"/>
    <general joint="j2" gear="2" group="2"/>
    </actuator></mujoco>'''
  m = mujoco.MjModel.from_xml_string(xml)
  m.opt.disableactuator = 1 << 2  # disable group 2
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  for step in range(40):
    u = 5.0 if step < 20 else -5.0  # saturates both clip stages
    sim.step(1, ctrl=np.array([[u, 1.0]], dtype=np.float32))
    cpu.ctrl[:] = [u, 1.0]
    mujoco.mj_step(m, cpu)
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq, np.asarray(cpu.qpos), rtol=1e-5, atol=5e-5)
  # Group-2 actuator contributes nothing: j2 stays at free fall from gravity only.
  asm = sim.assembled_system(recompute=True)
  assert asm is not None


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_act_lifecycle_native_gpu():
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002"/><worldbody>' + HINGE_BODY + '</worldbody>'
      '<actuator><general joint="j" dyntype="filter" dynprm="0.05 0 0"/></actuator>'
      '<keyframe><key name="k0" qpos="0.5" act="0.25"/></keyframe></mujoco>')
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  assert sim.state.na == 1
  snap = sim.state.snapshot()
  assert snap.schema_version == 6 and snap.nact == 1
  sim.step(5, ctrl=np.ones((2, 1), dtype=np.float32))
  assert float(sim.state.act.cpu().numpy()[0, 0]) > 0.05
  sim.state.restore(snap)
  np.testing.assert_allclose(sim.state.act.cpu().numpy(), np.zeros((2, 1)), atol=1e-7)
  sim.reset_to_keyframe(0, env_ids=[0])
  np.testing.assert_allclose(sim.state.act.cpu().numpy()[0], [0.25], atol=1e-6)
  np.testing.assert_allclose(sim.state.act.cpu().numpy()[1], [0.0], atol=1e-7)
  sim.copy_environment(0, 1)
  np.testing.assert_allclose(sim.state.act.cpu().numpy()[1], [0.25], atol=1e-6)
  # Old schema into activation model is rejected, never silently dropped.
  from mujoco_metal.device_state import StateSnapshot
  v3 = StateSnapshot(
      model_fingerprint=snap.model_fingerprint, profile_fingerprint=snap.profile_fingerprint,
      timestep=snap.timestep, nq=snap.nq, nv=snap.nv, batch_size=2,
      qpos=snap.qpos, qvel=snap.qvel, qacc=snap.qacc, time=snap.time,
      status=snap.status, schema_version=3, neq=snap.neq, eq_active=snap.eq_active,
      nmocap=snap.nmocap, mpos=snap.mpos, mquat=snap.mquat)
  with pytest.raises(ValueError, match="activation"):
    sim.state.restore(v3)
  with pytest.raises(ValueError):
    sim.reset(act=np.array([[0.0, 1.0]], dtype=np.float32))


def test_muscle_dynamics_transition_band_cpu():
  # R5 regression: pinned mju_sigmoid is a quintic smootherstep over [0,1],
  # not a logistic. The transition band must match the CPU oracle tightly.
  import mujoco
  import numpy as np
  from mujoco_metal.stateful_actuation import ActuatorModel, act_dot_reference
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002"/><worldbody>'
      '<body pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/>'
      '<geom type="sphere" size="0.1" mass="1"/></body>'
      '</worldbody><actuator>'
      '<general joint="j" dyntype="muscle" dynprm="0.01 0.04 0.01"/>'
      '</actuator></mujoco>')
  meta = ActuatorModel(m)
  d = mujoco.MjData(m)
  d.act[0] = 0.2
  mujoco.mj_forward(m, d)
  length = np.asarray(d.actuator_length)
  vel = np.asarray(d.actuator_velocity)
  for dc in [-0.005, -0.002, -0.001, -0.0005, 0.0005, 0.001, 0.002, 0.005]:
    ctrl = np.array([0.2 + dc])
    d.ctrl[0] = 0.2 + dc
    mujoco.mj_forward(m, d)
    ref = act_dot_reference(meta, ctrl, np.asarray(d.act), length, vel)
    np.testing.assert_allclose(ref[0], float(d.act_dot[0]), rtol=1e-6, atol=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_muscle_dynamics_transition_band_gpu():
  # R5 regression (native): one native step from matched states must match
  # CPU act_dot in the transition band, not just in saturated regimes.
  import mujoco
  import numpy as np
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002"/><worldbody>'
      '<body pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/>'
      '<geom type="sphere" size="0.1" mass="1"/></body>'
      '</worldbody><actuator>'
      '<general joint="j" dyntype="muscle" dynprm="0.01 0.04 0.01"/>'
      '</actuator></mujoco>')
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  cpu.act[0] = 0.2
  mujoco.mj_forward(m, cpu)
  worst = 0.0
  for dc in [-0.002, -0.001, 0.001, 0.002]:
    cpu.ctrl[0] = 0.2 + dc
    mujoco.mj_forward(m, cpu)
    cpu_dot = float(np.asarray(cpu.act_dot)[0])
    sim.reset(act=np.array([[0.2]], dtype=np.float32))
    sim.step(1, ctrl=np.array([[0.2 + dc]], dtype=np.float32))
    nat_dot = float(sim._act_dot.cpu().numpy()[0, 0])
    worst = max(worst, abs(nat_dot - cpu_dot))
  assert worst < 1e-5, worst
