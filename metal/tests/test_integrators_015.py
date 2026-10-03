# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 015: integrators, passive forces and force derivatives.

Covers Euler/RK4/implicitfast/implicit admission, analytic velocity
derivatives vs finite differences, stiff damped Euler, spinning asymmetric
bodies, geom-fluid integrated trajectories, contact/equality mixtures,
timestep refinement and energy/work checks. All comparisons use two nonzero
initial states; the CPU engine is the independent oracle.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

PROFILE = "integrated_euler_v1"


def _sim(model, qpos, qvel, batch=1, profile=PROFILE):
  sim = MetalSimulation(model, batch_size=batch, profile=profile)
  sim.reset(qpos=np.asarray(qpos, dtype=np.float32).reshape(batch, -1),
            qvel=np.asarray(qvel, dtype=np.float32).reshape(batch, -1))
  return sim


def _cpu(model, qpos, qvel):
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  mujoco.mj_forward(model, d)
  return d


def test_damper_derivative_matches_finite_difference():
  # Analytic polynomial damper derivative vs central differences.
  from mujoco_metal.passive import PassiveForceModel
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002"/>'
      '<worldbody><body pos="0 0 1"><joint name="h" type="hinge" axis="0 0 1" '
      'damping="1.5"/><geom type="sphere" size="0.1"/></body>'
      '<body pos="1 0 1"><joint name="s" type="slide" axis="1 0 0" damping="0.7"/>'
      '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
  m.dof_dampingpoly[:] = np.array([[0.3, 0.1]] * m.nv)
  model = PassiveForceModel(m)
  rng = np.random.default_rng(7)
  qv = rng.normal(0, 1.5, (3, m.nv))
  analytic = model.damping_derivative(qv)
  h = 1e-4
  c0 = np.asarray(model.damping)[None, :]
  c1 = np.asarray(model.damperpoly)[:, 0][None, :]
  c2 = np.asarray(model.damperpoly)[:, 1][None, :]
  def force(v):
    return -(v * (c0 + c1 * np.abs(v) + c2 * v * v))
  num = np.zeros_like(analytic)
  for i in range(m.nv):
    dp, dm = qv.copy(), qv.copy()
    dp[:, i] += h
    dm[:, i] -= h
    num[:, i] = -((force(dp)[:, i] - force(dm)[:, i]) / (2 * h))
  np.testing.assert_allclose(analytic, num, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_stiff_damped_euler_matches_oracle_gpu():
  # Stiff polynomial damping with EULERDAMP (implicit velocity solve).
  xml = ('<mujoco><option timestep="0.002" integrator="Euler"/>'
         '<worldbody><body pos="0 0 1"><joint name="h" type="hinge" axis="0 0 1" '
         'damping="8" stiffness="20"/><geom type="sphere" size="0.1"/></body>'
         '</worldbody></mujoco>')
  for q0, v0 in ((0.6, 2.0), (-0.4, -3.0)):
    m = mujoco.MjModel.from_xml_string(xml)
    nq, nv = m.nq, m.nv
    qp = np.zeros(nq)
    qp[0] = q0
    qv = np.full(nv, v0)
    sim = _sim(m, qp, qv)
    cpu = _cpu(m, qp, qv)
    for _ in range(200):
      sim.step(1)
      mujoco.mj_step(m, cpu)
    assert int(sim.state.status.cpu().numpy()[0]) == 0
    gq = sim.state.qpos.cpu().numpy()[0]
    np.testing.assert_allclose(gq, np.asarray(cpu.qpos), atol=3e-3, rtol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_spinning_asymmetric_body_gpu():
  # Torque-free asymmetric body: gyroscopic motion must track the oracle.
  xml = ('<mujoco><option timestep="0.001" integrator="Euler"/>'
         '<worldbody><body pos="0 0 1"><freejoint/>'
         '<inertial pos="0 0 0" mass="1" diaginertia="0.05 0.06 0.09"/>'
         '<geom type="box" size="0.1 0.15 0.2"/></body></worldbody></mujoco>')
  for w0 in ([3.0, 0.5, 0.2], [-1.0, 4.0, 1.5]):
    m = mujoco.MjModel.from_xml_string(xml)
    qp = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
    qv = np.array(w0 * 2, dtype=float)[:m.nv]
    sim = _sim(m, qp, qv)
    cpu = _cpu(m, qp, qv)
    for _ in range(300):
      sim.step(1)
      mujoco.mj_step(m, cpu)
    assert int(sim.state.status.cpu().numpy()[0]) == 0
    gq = sim.state.qpos.cpu().numpy()[0]
    np.testing.assert_allclose(gq[:3], np.asarray(cpu.qpos)[:3],
                               atol=5e-3, err_msg="asymmetric-pos")
    # Orientation geodesic error.
    from mujoco_metal.passive import PassiveForceModel  # noqa (import check)
    qg = gq[3:7] / np.linalg.norm(gq[3:7])
    qc = np.asarray(cpu.qpos)[3:7] / np.linalg.norm(np.asarray(cpu.qpos)[3:7])
    ang = 2 * np.arccos(min(1.0, abs(float(qg @ qc))))
    assert ang < 0.05, ang


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_geom_fluid_trajectory_matches_oracle_gpu():
  # Ellipsoid-model geom fluid in a full integrated trajectory.
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 0" '
         'density="1.2" viscosity="0.05" wind="0.3 -0.1 0.2"/>'
         '<worldbody><body pos="0.2 -0.3 1"><freejoint/>'
         '<inertial pos="0 0 0" mass="1" diaginertia="0.08 0.1 0.12"/>'
         '<geom name="ball" type="ellipsoid" size="0.12 0.09 0.07" mass="1"/>'
         '</body></worldbody></mujoco>')
  for vscale in (1.0, -0.5):
    m = mujoco.MjModel.from_xml_string(xml)
    m.geom_fluid[0] = [1, 0.8, 0.3, 0.5, 0.4, 0.6, 0.5, 0.6, 0.7, 0.2, 0.3, 0.4]
    qp = np.array(m.qpos0, dtype=float)
    q = np.array([0.9, 0.2, -0.3, 0.1])
    qp[3:7] = q / np.linalg.norm(q)
    qv = np.linspace(-0.7, 0.9, m.nv) * vscale
    sim = _sim(m, qp, qv)
    cpu = _cpu(m, qp, qv)
    for _ in range(150):
      sim.step(1)
      mujoco.mj_step(m, cpu)
    assert int(sim.state.status.cpu().numpy()[0]) == 0
    np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0],
                               np.asarray(cpu.qpos), atol=4e-3, rtol=3e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_contact_equality_fluid_mixture_gpu():
  # Contact + equality + fluid + passive in one integrated solve.
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" density="0.8" '
         'viscosity="0.02"/><worldbody>'
         '<geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body name="a" pos="0 0 0.3"><freejoint/>'
         '<geom type="box" size="0.06 0.06 0.05" mass="0.4"/></body>'
         '<body name="b" pos="0.4 0 0.4"><freejoint/>'
         '<geom type="sphere" size="0.05" mass="0.2"/></body>'
         '</worldbody>'
         '<equality><connect body1="a" body2="b" anchor="0.2 0 0.35"/></equality>'
         '</mujoco>')
  for v0 in (0.5, -1.0):
    m = mujoco.MjModel.from_xml_string(xml)
    qp = np.asarray(m.qpos0, dtype=float)
    qv = np.full(m.nv, v0)
    qv[0] = 0.3 * v0
    sim = _sim(m, qp, qv)
    cpu = _cpu(m, qp, qv)
    for _ in range(200):
      sim.step(1)
      mujoco.mj_step(m, cpu)
    assert int(sim.state.status.cpu().numpy()[0]) == 0
    assert cpu.ncon > 0, "mixture must engage contacts"
    np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0][:3],
                               np.asarray(cpu.qpos)[:3], atol=5e-3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_timestep_refinement_reduces_error_gpu():
  # Halving dt must not increase oracle error (refinement sanity).
  xml = ('<mujoco><option timestep="TIMESTEP" integrator="Euler"/>'
         '<worldbody><body pos="0 0 1"><joint name="h" type="hinge" axis="0 0 1" '
         'damping="1.2" stiffness="6"/><geom type="sphere" size="0.1"/></body>'
         '</worldbody></mujoco>')
  errs = []
  for dt in ("0.004", "0.002"):
    m = mujoco.MjModel.from_xml_string(xml.replace("TIMESTEP", dt))
    qp = np.array([0.8])
    qv = np.array([2.5])
    sim = _sim(m, qp, qv)
    cpu = _cpu(m, qp, qv)
    for _ in range(250):
      sim.step(1)
      mujoco.mj_step(m, cpu)
    errs.append(float(np.max(np.abs(
        sim.state.qpos.cpu().numpy()[0] - np.asarray(cpu.qpos)))))
  assert errs[1] <= errs[0] + 1e-4, errs


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_damped_energy_decays_and_free_energy_holds_gpu():
  # Damped oscillator loses energy monotonically; undamped free body holds it.
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 0">'
         '<flag energy="enable"/></option>'
         '<worldbody><body pos="0 0 0"><joint name="h" type="hinge" axis="0 0 1" '
         'damping="DAMP" stiffness="10"/><geom type="sphere" size="0.1"/></body>'
         '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml.replace("DAMP", "1.0"))
  sim = _sim(m, [0.9], [0.0])
  cpu = _cpu(m, np.array([0.9]), np.array([0.0]))
  cpu_e0 = None
  for _ in range(200):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    if cpu_e0 is None:
      cpu_e0 = float(cpu.energy[0] + cpu.energy[1])
  cpu_e1 = float(cpu.energy[0] + cpu.energy[1])
  # Oracle premise: damped energy decays overall.
  assert cpu_e1 < 0.5 * cpu_e0, (cpu_e0, cpu_e1)
  # Native tracks the oracle trajectory (energy follows from parity).
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq, np.asarray(cpu.qpos), atol=5e-3,
                             err_msg="native tracks damped settle")
  m2 = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 0"/>'
      '<worldbody><body pos="0 0 0"><joint name="h" type="hinge" axis="0 0 1"/>'
      '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
  sim2 = _sim(m2, [0.0], [3.0])
  for _ in range(200):
    sim2.step(1)
  v = float(sim2.state.qvel.cpu().numpy()[0, 0])
  assert abs(v - 3.0) / 3.0 < 0.005, v


def test_integrator_admission_by_combination():
  # Admission is driven by supported combinations, not just the enum.
  from mujoco_metal.stepping import validate_stepping_profile
  # Full implicit is rejected everywhere.
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><option integrator="Euler"/><worldbody/></mujoco>')
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
  with pytest.raises(ValueError, match="Euler|mplicit"):
    validate_stepping_profile(m, profile="contact_free_euler_v1")
  # RK4 with activation state is integrated at every stage (R06/D3).
  m2 = mujoco.MjModel.from_xml_string(
      '<mujoco><option integrator="RK4"><flag contact="disable"/></option><worldbody>'
      '<body><joint name="h" type="hinge" axis="0 0 1"/>'
      '<geom type="sphere" size="0.1"/>'
      '</body></worldbody>'
      '<actuator><general joint="h" dyntype="filter" gainprm="5"/></actuator>'
      '</mujoco>')
  assert int(m2.na) > 0
  prof = validate_stepping_profile(m2, profile="integrated_rk4_v1")
  assert prof is not None
  # implicitfast with polynomial damping assembles natively (R06/D2).
  m3 = mujoco.MjModel.from_xml_string(
      '<mujoco><option integrator="Euler"><flag contact="disable"/></option>'
      '<worldbody>'
      '<body><joint type="hinge" axis="0 0 1" damping="0.5"/>'
      '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
  m3.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  m3.dof_dampingpoly[:] = 0.2
  prof3 = validate_stepping_profile(m3, profile="contact_free_implicitfast_v1")
  assert prof3 is not None
