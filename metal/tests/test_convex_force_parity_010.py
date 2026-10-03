# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 010: solved force/torque parity for convex contact pairs.

Compares native solved generalized constraint force (`qfrc_constraint`) and
accelerations (`qacc`) against the CPU oracle stepping from IDENTICAL states
(no drift ambiguity), plus single-step kinematic sanity and a sustained
settle trajectory. Covers sustained press, impacts, sliding with friction,
spin, off-center (torque) contacts, scaled models and the admitted CCD
option combinations. Geometric manifold checks live in
test_analytic_collision_010.py; this file gates physics.

Two deliberate scope choices (both documented, neither hides error):
- Friction branch flips: static/kinetic Coulomb decisions flip on
  micrometer-per-second velocity noise, changing forces by O(mu*N).
  Multi-step force comparison therefore cannot distinguish solver error
  from chaotic branch flips. Solved forces are gated at identical
  pre-step states (step 0); multi-step runs gate kinematics only, with an
  envelope derived below, plus equilibrium (settle) trajectories where
  branches reconverge.
- Redundant manifolds admit non-unique friction/torque splits (both
  engines' PGS distributions are valid). Translation (total-force)
  components are determinate and gated tightly; rotation components are
  gated at force scale (tight) while accelerations are compared
  mass-weighted (M*qacc at force tolerance) so small-inertia DOFs are not
  false-alarmed by sub-newton torque residuals (a 7e-4 N m residual is
  invisible at force scale but reads 0.1 rad/s^2 through a 6e-3 kg m^2
  inertia).
- Rest equilibria with friction are non-unique (any pose within the
  friction cone holds); settle trajectories gate the rest pose within a
  few millimeters, not micrometers.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

_PROFILE = "integrated_euler_v1"
_IQ = [1.0, 0.0, 0.0, 0.0]


def _free_qpos(x, y, z, quat=_IQ):
  return [x, y, z] + list(quat)


def _load(xml, qpos0, qvel0):
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
  qpos0 = np.asarray(qpos0, dtype=np.float32).reshape(1, -1)
  qvel0 = np.asarray(qvel0, dtype=np.float32).reshape(1, -1)
  sim.reset(qpos=qpos0, qvel=qvel0)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qpos0[0]
  cpu.qvel[:] = qvel0[0]
  mujoco.mj_forward(m, cpu)
  return m, sim, cpu


def _mass_diag(asm, nv):
  mass = np.asarray(asm["mass_matrix"].cpu().numpy()[0]).reshape(nv, nv)
  return np.abs(np.diag(mass))


def _parity_step0(xml, qpos0, qvel0, tag,
                  f_rtol=0.03, f_atol=0.02, a_rtol=0.03):
  """Solved-force parity at identical pre-step states (both engines)."""
  m, sim, cpu = _load(xml, qpos0, qvel0)
  asm = sim.assembled_system(recompute=True)
  mujoco.mj_step(m, cpu)
  f_nat = np.asarray(asm["qfrc_constraint"].cpu().numpy()[0])
  f_cpu = np.asarray(cpu.qfrc_constraint)
  np.testing.assert_allclose(f_nat, f_cpu, rtol=f_rtol, atol=f_atol,
                             err_msg=f"{tag} qfrc_constraint")
  a_nat = np.asarray(asm["qacc"].cpu().numpy()[0])
  a_cpu = np.asarray(cpu.qacc)
  mdiag = _mass_diag(asm, m.nv)
  # Mass-weighted acceleration comparison: a torque residual at force scale
  # f_atol reads as f_atol/inertia in acceleration, false-alarming on
  # small-inertia DOFs. Comparing M*qacc (force scale) with the force
  # tolerance keeps every DOF at its natural sensitivity (array atol is
  # unsupported here, hence the explicit weighting).
  np.testing.assert_allclose(mdiag * a_nat, mdiag * a_cpu,
                             rtol=a_rtol, atol=f_atol,
                             err_msg=f"{tag} qacc")
  assert int(sim.state.status.cpu().numpy()[0]) == 0, tag
  return m, sim, cpu


def _parity_1step(xml, qpos0, qvel0, tag, **kw):
  """Step-0 solved forces plus one integrated step (kinematic sanity).

  Single-step velocity envelopes: translation uses rtol 2e-3 plus atol
  5e-4 (worst-case static/kinetic branch flip: mu*N ~ 10 N on ~1.5 kg over
  2 ms, plus relative precision for large sliding velocities); rotation
  uses the same rtol with atol 5e-3 to absorb torque-split residuals (see
  module docstring). Position drift from the same events is ~5e-7, hence
  the tight qpos gate (5e-6 admits quaternion integration noise).
  """
  m, sim, cpu = _parity_step0(xml, qpos0, qvel0, tag, **kw)
  sim.step(1)
  np.testing.assert_allclose(
      sim.state.qpos.cpu().numpy()[0], np.asarray(cpu.qpos),
      atol=5e-6, err_msg=f"{tag} qpos")
  # Split velocity gates (fixtures here are dual free bodies, so DOFs
  # 0:3/6:9 translate and 3:6/9:12 rotate): translation is determinate
  # and tight; rotation accumulates torque-split residuals (PGS friction
  # distributions are non-unique on redundant manifolds, plus CPU facet
  # torque noise), so rotation gets a loose absolute bound that still
  # catches gross spin errors.
  qv_nat = sim.state.qvel.cpu().numpy()[0]
  qv_cpu = np.asarray(cpu.qvel)
  trans_idx = [0, 1, 2, 6, 7, 8]
  rot_idx = [3, 4, 5, 9, 10, 11]
  np.testing.assert_allclose(
      qv_nat[trans_idx], qv_cpu[trans_idx],
      rtol=2e-3, atol=5e-4, err_msg=f"{tag} qvel-linear")
  np.testing.assert_allclose(
      qv_nat[rot_idx], qv_cpu[rot_idx],
      rtol=2e-3, atol=5e-3, err_msg=f"{tag} qvel-angular")


_CAPCYL_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 0.5"><freejoint/>
<geom name="c" type="cylinder" size="0.05 0.1" friction="0.8 0.1 0.1"/></body>
<body pos="0.089 0 0.5"><freejoint/>
<geom name="k" type="capsule" size="0.04 0.08" friction="0.8 0.1 0.1"/></body>
</worldbody></mujoco>"""

_CYLCYL_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 0.5"><freejoint/>
<geom name="c1" type="cylinder" size="0.05 0.1" friction="0.8 0.1 0.1"/></body>
<body pos="0.089 0 0.5"><freejoint/>
<geom name="c2" type="cylinder" size="0.04 0.12" friction="0.8 0.1 0.1"/></body>
</worldbody></mujoco>"""

_CYLBOX_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 0.5"><freejoint/>
<geom name="c" type="cylinder" size="0.05 0.1" friction="0.8 0.1 0.1"/></body>
<body pos="0.099 0 0.5"><freejoint/>
<geom name="b" type="box" size="0.05 0.05 0.05" friction="0.8 0.1 0.1"/></body>
</worldbody></mujoco>"""

_CAPELL_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 0.5"><freejoint/>
<geom name="e" type="ellipsoid" size="0.06 0.04 0.05" friction="0.8 0.1 0.1"/></body>
<body pos="0.099 0 0.5"><freejoint/>
<geom name="k" type="capsule" size="0.04 0.08" friction="0.8 0.1 0.1"/></body>
</worldbody></mujoco>"""

_CYLELL_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 0.5"><freejoint/>
<geom name="c" type="cylinder" size="0.05 0.1" friction="0.8 0.1 0.1"/></body>
<body pos="0.109 0 0.5"><freejoint/>
<geom name="e" type="ellipsoid" size="0.06 0.04 0.05" friction="0.8 0.1 0.1"/></body>
</worldbody></mujoco>"""

_ELLELL_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 0.5"><freejoint/>
<geom name="ea" type="ellipsoid" size="0.06 0.04 0.05" friction="0.8 0.1 0.1"/></body>
<body pos="0.109 0 0.5"><freejoint/>
<geom name="eb" type="ellipsoid" size="0.05 0.05 0.03" friction="0.8 0.1 0.1"/></body>
</worldbody></mujoco>"""

_SPHCYL_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body pos="0 0 0.5"><freejoint/>
<geom name="c" type="cylinder" size="0.05 0.1" friction="0.8 0.1 0.1"/></body>
<body pos="0.089 0 0.5"><freejoint/>
<geom name="s" type="sphere" size="0.04" friction="0.8 0.1 0.1"/></body>
</worldbody></mujoco>"""


def _q(qa, qb, wa=None, wb=None):
  wa = [0.0] * 6 if wa is None else list(wa)
  wb = [0.0] * 6 if wb is None else list(wb)
  return qa + qb, wa + wb


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_press_sustained_gpu():
  # Sustained 1 mm press under gravity, zero velocity: normal force parity.
  for tag, xml, dx in [("capcyl", _CAPCYL_XML, 0.089),
                       ("cylcyl", _CYLCYL_XML, 0.089),
                       ("cylbox", _CYLBOX_XML, 0.099),
                       ("sphcyl", _SPHCYL_XML, 0.089),
                       ("capell", _CAPELL_XML, 0.099),
                       ("cylell", _CYLELL_XML, 0.109),
                       ("ellell", _ELLELL_XML, 0.109)]:
    qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(dx, 0, 0.5))
    _parity_1step(xml, qp, qv, f"press/{tag}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_impact_gpu():
  # Normal approach velocity 1 m/s from 2 cm: impact transient parity.
  for tag, xml, dx in [("capcyl", _CAPCYL_XML, 0.11),
                       ("cylcyl", _CYLCYL_XML, 0.11),
                       ("cylbox", _CYLBOX_XML, 0.12)]:
    qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(dx, 0, 0.5),
                [0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0])
    _parity_1step(xml, qp, qv, f"impact/{tag}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_slide_friction_gpu():
  # Tangential sliding 0.5 m/s with friction: friction force parity.
  for tag, xml, dx in [("capcyl", _CAPCYL_XML, 0.089),
                       ("cylbox", _CYLBOX_XML, 0.099)]:
    qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(dx, 0, 0.5),
                [0, 0, 0, 0, 0, 0], [0, 0.5, 0, 0, 0, 0])
    _parity_1step(xml, qp, qv, f"slide/{tag}")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_spin_offcenter_gpu():
  # Spin about the contact normal plus an axial offset (torque arms).
  qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(0.089, 0, 0.56),
              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 3.0])
  _parity_1step(_CAPCYL_XML, qp, qv, "spin-offcenter/capcyl")
  qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(0.099, 0.02, 0.5),
              [0, 0, 0, 0, 0, 0], [0, 0, 0, 2.0, 0, 0])
  _parity_1step(_CYLBOX_XML, qp, qv, "spin-offcenter/cylbox")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_scaled_press_gpu():
  # Same press at 0.2x and 5x scale: relative multiCCD tolerance holds.
  for scale in (0.2, 5.0):
    xml = _CAPCYL_XML.replace("0.05 0.1", f"{0.05*scale} {0.1*scale}").replace(
        "0.04 0.08", f"{0.04*scale} {0.08*scale}")
    dx = 0.089 * scale
    z = 0.5 * scale
    qp, qv = _q(_free_qpos(0, 0, z), _free_qpos(dx, 0, z))
    _parity_1step(xml, qp, qv, f"scaled-press/{scale}",
                  f_atol=0.02 * scale ** 3)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_ccd_options_gpu():
  # Admitted CCD option combinations use the same oracle flags on both sides.
  for name, bits in [
      ("default", 0),
      ("no-multiccd", int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),
      ("no-nativeccd", int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))]:
    m = mujoco.MjModel.from_xml_string(_CAPCYL_XML)
    m.opt.disableflags = bits
    sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
    qp = np.array(_free_qpos(0, 0, 0.5) + _free_qpos(0.089, 0, 0.5),
                  dtype=np.float32).reshape(1, -1)
    qv = np.zeros((1, 12), dtype=np.float32)
    sim.reset(qpos=qp, qvel=qv)
    cpu = mujoco.MjData(m)
    cpu.qpos[:] = qp[0]
    cpu.qvel[:] = qv[0]
    mujoco.mj_forward(m, cpu)
    asm = sim.assembled_system(recompute=True)
    mujoco.mj_step(m, cpu)
    sim.step(1)
    np.testing.assert_allclose(
        np.asarray(asm["qfrc_constraint"].cpu().numpy()[0]),
        np.asarray(cpu.qfrc_constraint), rtol=0.03, atol=0.02,
        err_msg=f"options/{name} qfrc")
    np.testing.assert_allclose(
        sim.state.qpos.cpu().numpy()[0], np.asarray(cpu.qpos),
        atol=5e-6, err_msg=f"options/{name} qpos")
    if name == "no-multiccd":
      # Single witness on both sides without multiCCD restarts.
      assert int(asm["contact_mask"].cpu().numpy()[0].sum()) == 1, name
      assert cpu.ncon == 1, name
    else:
      assert int(asm["contact_mask"].cpu().numpy()[0].sum()) >= 1, name


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_rotated_press_gpu():
  # Rotated press (roll tilt about X keeps the side-contact regime while
  # genuinely tilting patch frames): frame-sensitive torque parity.
  q30 = [np.cos(np.pi / 12.0), np.sin(np.pi / 12.0), 0.0, 0.0]
  qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(0.089, 0, 0.5, q30))
  _parity_1step(_CAPCYL_XML, qp, qv, "rotated-press/capcyl")
  qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(0.099, 0, 0.5, q30))
  _parity_1step(_CYLBOX_XML, qp, qv, "rotated-press/cylbox")
  qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(0.109, 0, 0.5, q30))
  _parity_1step(_CYLELL_XML, qp, qv, "rotated-press/cylell")
  qp, qv = _q(_free_qpos(0, 0, 0.5), _free_qpos(0.109, 0, 0.5, q30))
  _parity_1step(_ELLELL_XML, qp, qv, "rotated-press/ellell")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_deep_settle_trajectory_gpu():
  # Deep overlap recovers and settles; final rest pose matches. Exercises
  # the penetration-lens regime dynamically instead of asserting static
  # deep witness placement (implementation-defined on both sides).
  for tag, xml, dx in [("capcyl", _CAPCYL_XML, 0.054),
                       ("capell", _CAPELL_XML, 0.06),
                       ("cylell", _CYLELL_XML, 0.066)]:
    m = mujoco.MjModel.from_xml_string(xml)
    sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
    qp = np.array(_free_qpos(0, 0, 0.5) + _free_qpos(dx, 0, 0.5),
                  dtype=np.float32).reshape(1, -1)
    sim.reset(qpos=qp, qvel=np.zeros((1, 12), dtype=np.float32))
    cpu = mujoco.MjData(m)
    cpu.qpos[:] = qp[0]
    cpu.qvel[:] = 0
    mujoco.mj_forward(m, cpu)
    for _ in range(200):
      sim.step(1)
      mujoco.mj_step(m, cpu)
    # Static friction makes rest poses non-unique and facet/lens torques
    # integrate into attitude; gate positions (3 mm) and quaternions
    # (1e-2 rad) separately, plus clean solver status.
    qn = sim.state.qpos.cpu().numpy()[0]
    qc = np.asarray(cpu.qpos)
    np.testing.assert_allclose(qn[[0, 1, 2, 7, 8, 9]], qc[[0, 1, 2, 7, 8, 9]],
                               atol=3e-3, err_msg=f"deep-settle/{tag} pos")
    np.testing.assert_allclose(
        qn[[3, 4, 5, 6, 10, 11, 12, 13]], qc[[3, 4, 5, 6, 10, 11, 12, 13]],
        atol=1e-2, err_msg=f"deep-settle/{tag} quat")
    assert int(sim.state.status.cpu().numpy()[0]) == 0, tag


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_force_press_settle_trajectory_gpu():
  # Sustained press settles; final rest pose matches (force convergence).
  m = mujoco.MjModel.from_xml_string(_CAPCYL_XML)
  sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
  qp = np.array(_free_qpos(0, 0, 0.5) + _free_qpos(0.089, 0, 0.5),
                dtype=np.float32).reshape(1, -1)
  sim.reset(qpos=qp, qvel=np.zeros((1, 12), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(200):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0],
                             np.asarray(cpu.qpos), atol=1e-3)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
