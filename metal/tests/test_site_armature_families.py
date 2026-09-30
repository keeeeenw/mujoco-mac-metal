# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""G1: site-only spatial armature across joint families (CPU + GPU).

Covers free, ball, slide and multi-joint bodies with offset inertias/sites,
randomized rotations and nonzero angular/translational velocities. Compares
native dots/bias/mass/qacc against the pinned oracle plus trajectories.
"""

import os

import mujoco
import numpy as np
import pytest


def _pinned_dot_generic(m, d, site_names):
  # Exact pinned mj_tendonDot for a site-only tendon (dense path).
  nv = m.nv
  qvel = np.asarray(d.qvel)
  sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in site_names]
  res = 0.0
  for k in range(len(sids) - 1):
    id0, id1 = sids[k], sids[k + 1]
    b0, b1 = int(m.site_bodyid[id0]), int(m.site_bodyid[id1])
    if b0 == b1:
      continue
    p0 = np.asarray(d.site_xpos[id0])
    p1 = np.asarray(d.site_xpos[id1])
    dpnt = p1 - p0
    norm = float(np.linalg.norm(dpnt))
    if norm < 1e-12:
      continue
    dpnt = dpnt / norm
    J0 = np.zeros((3, nv)); J1 = np.zeros((3, nv)); Jr = np.zeros((3, nv))
    Jd0 = np.zeros((3, nv)); Jd1 = np.zeros((3, nv))
    mujoco.mj_jac(m, d, J0, Jr, p0.reshape(3, 1), b0)
    mujoco.mj_jac(m, d, J1, Jr, p1.reshape(3, 1), b1)
    mujoco.mj_jacDot(m, d, Jd0, Jr, p0.reshape(3, 1), b0)
    mujoco.mj_jacDot(m, d, Jd1, Jr, p1.reshape(3, 1), b1)
    v0, v1 = J0 @ qvel, J1 @ qvel
    dv = v1 - v0
    dvel = (dv - dpnt * float(dpnt @ dv)) / norm
    res += float(dpnt @ ((Jd1 - Jd0) @ qvel)) + float(dvel @ (v1 - v0))
  return res


def _native_state(sim, m, qp, qv):
  sim.reset(qpos=np.asarray(qp, dtype=np.float32).reshape(1, -1),
            qvel=np.asarray(qv, dtype=np.float32).reshape(1, -1))
  qpos_t, qvel_t = sim.state._qpos, sim.state._qvel
  dynamics = sim._smooth.run_device(qpos_t, qvel_t, None, None)
  kin = sim._spatial_tendons.run_kinematics(qvel_t, dynamics["poses"])
  bias, dots = sim._spatial_tendons.run_armature_bias(
      kin, qvel_t, dynamics["poses"], dynamics.get("cvel", None),
      dynamics.get("root_com", None), dynamics.get("cdof", None),
      dynamics.get("cdof_dot", None))
  asm = sim.assembled_system(recompute=True)
  return dots.cpu().numpy()[0], bias.cpu().numpy()[0], asm["qacc"].cpu().numpy()[0]


FREE_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<site name="anchor" pos="-0.3 0.2 1.5"/>
<body name="body" pos="0.5 0 1"><freejoint/>
<geom type="sphere" size="0.1" pos="0.1 0 0" mass="1"/>
<site name="tip" pos="0.2 0.1 0"/></body>
</worldbody>
<tendon><spatial name="t" armature="0.5"><site site="anchor"/><site site="tip"/></spatial></tendon>
</mujoco>"""

BALL_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<site name="anchor" pos="-0.3 0.2 1.5"/>
<body name="body" pos="0.5 0 1"><joint name="b" type="ball"/>
<geom type="sphere" size="0.1" pos="0.1 0.05 0" mass="1"/>
<site name="tip" pos="0.2 0.1 0"/></body>
</worldbody>
<tendon><spatial name="t" armature="0.5"><site site="anchor"/><site site="tip"/></spatial></tendon>
</mujoco>"""

SLIDE_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<site name="anchor" pos="-0.3 0.2 1.5"/>
<body name="body" pos="0.5 0 1"><joint name="s" type="slide" axis="1 0 1"/>
<geom type="box" size="0.08 0.06 0.1" pos="0.05 0 0.02" mass="1"/>
<site name="tip" pos="0.2 0.1 0"/></body>
</worldbody>
<tendon><spatial name="t" armature="0.5"><site site="anchor"/><site site="tip"/></spatial></tendon>
</mujoco>"""

MULTI_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<site name="anchor" pos="-0.3 0.2 1.5"/>
<body name="car" pos="0 0 1"><joint name="x" type="slide" axis="1 0 0"/>
<joint name="z" type="slide" axis="0 0 1"/>
<geom type="box" size="0.1 0.08 0.06" pos="0.03 0 0" mass="1"/>
<site name="tip" pos="0.15 0.05 0.02"/>
<body name="arm" pos="0.2 0 0"><joint name="e" type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05" pos="0 0.04 0" mass="0.4"/>
<site name="tip2" pos="0.08 -0.03 0.02"/></body></body>
</worldbody>
<tendon><spatial name="t" armature="0.5"><site site="anchor"/><site site="tip"/><site site="tip2"/></spatial></tendon>
</mujoco>"""

TWIN_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<site name="anchor" pos="-0.3 0.2 1.5"/>
<body name="b1" pos="0.4 0 1"><freejoint/>
<geom type="sphere" size="0.08" pos="0.06 0 0" mass="0.7"/>
<site name="tip1" pos="0.12 0.05 0"/></body>
<body name="b2" pos="-0.1 0 0.8"><freejoint/>
<geom type="box" size="0.07 0.06 0.09" pos="0 0.03 0" mass="0.5"/>
<site name="tip2" pos="-0.06 -0.04 0.02"/></body>
</worldbody>
<tendon><spatial name="t" armature="0.5"><site site="anchor"/><site site="tip1"/><site site="tip2"/></spatial></tendon>
</mujoco>"""

# Blocking-review fixtures: two joints serially on ONE body. The second
# joint's frame velocity must include the first joint's motion.
HH_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<site name="anchor" pos="-0.3 0.2 1.5"/>
<body name="car" pos="0 0 1"><joint name="h1" type="hinge" axis="0 1 0"/>
<joint name="h2" type="hinge" axis="1 0 0"/>
<geom type="box" size="0.1 0.08 0.06" pos="0.03 0 0" mass="1"/>
<site name="tip" pos="0.15 0.05 0.02"/></body>
</worldbody>
<tendon><spatial name="t" armature="0.5"><site site="anchor"/><site site="tip"/></spatial></tendon>
</mujoco>"""

HS_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<site name="anchor" pos="-0.3 0.2 1.5"/>
<body name="car" pos="0 0 1"><joint name="h" type="hinge" axis="0 1 0"/>
<joint name="s" type="slide" axis="1 0 0"/>
<geom type="box" size="0.1 0.08 0.06" pos="0.03 0 0" mass="1"/>
<site name="tip" pos="0.15 0.05 0.02"/></body>
</worldbody>
<tendon><spatial name="t" armature="0.5"><site site="anchor"/><site site="tip"/></spatial></tendon>
</mujoco>"""


def _finite_diff_dot(m, qp, qv):
  # Independent oracle: Jdot.qvel via central differences of the dense
  # tendon Jacobian along the velocity direction (no mj_jacDot involved).
  # Valid for hinge/slide models (no quaternion renormalization needed).
  h = 1e-7
  nv = m.nv
  rows = []
  for sgn in (1.0, -1.0):
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(qp, dtype=float) + sgn * h * np.asarray(qv, dtype=float)
    mujoco.mj_forward(m, d)
    r = np.zeros(nv)
    for k in range(int(m.ten_J_rownnz[0])):
      r[int(m.ten_J_colind[int(m.ten_J_rowadr[0]) + k])] = float(d.ten_J[int(m.ten_J_rowadr[0]) + k])
    rows.append(r)
  Jdot_qvel = (rows[0] - rows[1]) / (2 * h)
  return float(Jdot_qvel @ np.asarray(qv, dtype=float))


def _random_state(m, seed, vscale=3.0):
  rng = np.random.default_rng(seed)
  qp = (rng.random(m.nq) - 0.5) * 0.8
  qv = (rng.random(m.nv) - 0.5) * 2 * vscale
  # Normalize quaternions for ball/free joints (assume unit quats at compile).
  q = qp.copy()
  for j in range(m.njnt):
    typ = int(m.jnt_type[j])
    if typ in (int(mujoco.mjtJoint.mjJNT_BALL),):
      a = int(m.jnt_qposadr[j])
      n = np.linalg.norm(q[a:a + 4])
      if n > 1e-12:
        q[a:a + 4] /= n
    elif typ == int(mujoco.mjtJoint.mjJNT_FREE):
      a = int(m.jnt_qposadr[j])
      n = np.linalg.norm(q[a + 3:a + 7])
      if n > 1e-12:
        q[a + 3:a + 7] /= n
  return q, qv


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("xml,sites,seeds", [
    (FREE_XML, ("anchor", "tip"), (11, 23)),
    (BALL_XML, ("anchor", "tip"), (11, 23)),
    (SLIDE_XML, ("anchor", "tip"), (0, 1)),
    (MULTI_XML, ("anchor", "tip", "tip2"), (11, 23)),
    (TWIN_XML, ("anchor", "tip1", "tip2"), (11, 23)),
])
def test_armature_family_dots_and_qacc_gpu(xml, sites, seeds):
  # G1: dots/bias/mass/qacc vs pinned oracle per joint family, with offset
  # coms, randomized rotations and nonzero velocities.
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  worst_dot, worst_q = 0.0, 0.0
  for seed in seeds:
    qp, qv = _random_state(m, seed)
    dots, _, nqacc = _native_state(sim, m, qp, qv)
    d = mujoco.MjData(m)
    d.qpos[:] = qp
    d.qvel[:] = qv
    mujoco.mj_forward(m, d)
    ref = _pinned_dot_generic(m, d, sites)
    assert abs(ref) > 1e-3  # nonzero bias velocity guard
    worst_dot = max(worst_dot, abs(float(dots[0]) - ref))
    np.testing.assert_allclose(nqacc[:m.nv], np.asarray(d.qacc)[:m.nv],
                               rtol=1e-5, atol=2e-4)
    worst_q = max(worst_q, float(np.max(np.abs(nqacc[:m.nv] - np.asarray(d.qacc)[:m.nv]))))
  assert worst_dot < 2e-5, worst_dot


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("xml,sites", [
    (HH_XML, ("anchor", "tip")),
    (HS_XML, ("anchor", "tip")),
])
def test_armature_serial_same_body_joints_gpu(xml, sites):
  # Blocking review: two joints serially on ONE body. The second joint's
  # frame velocity must include the first joint's motion. Triple-checked:
  # native dots vs pinned mj_tendonDot vs finite differences of ten_J.
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  for seed in (3, 17):
    qp = (np.random.default_rng(seed).random(m.nq) - 0.5) * 0.8
    qv = (np.random.default_rng(seed + 100).random(m.nv) - 0.5) * 6.0
    dots, _, nqacc = _native_state(sim, m, qp, qv)
    d = mujoco.MjData(m)
    d.qpos[:] = qp
    d.qvel[:] = qv
    mujoco.mj_forward(m, d)
    ref = _pinned_dot_generic(m, d, sites)
    fd = _finite_diff_dot(m, qp, qv)
    assert abs(ref) > 1e-3, ref  # nonzero guard
    np.testing.assert_allclose(ref, fd, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(dots[0]), ref, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(nqacc[:m.nv], np.asarray(d.qacc)[:m.nv],
                               rtol=1e-5, atol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_armature_family_bias_and_replay_gpu():
  # G1: native bias force matches the two-model CPU bias oracle; snapshot +
  # rerun reproduces the trajectory.
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(FREE_XML)
  m0 = mujoco.MjModel.from_xml_string(FREE_XML.replace('armature="0.5"', 'armature="0.0"'))
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp, qv = _random_state(m, 5)
  dots, nbias, _ = _native_state(sim, m, qp, qv)
  d = mujoco.MjData(m)
  d.qpos[:] = qp
  d.qvel[:] = qv
  mujoco.mj_forward(m, d)
  d0 = mujoco.MjData(m0)
  d0.qpos[:] = qp
  d0.qvel[:] = qv
  mujoco.mj_forward(m0, d0)
  cpu_bias = np.asarray(d.qfrc_bias) - np.asarray(d0.qfrc_bias)
  # Native bias uses the pinned qfrc_bias-side convention (callers subtract
  # it from rhs), so it equals the two-model CPU bias difference directly.
  np.testing.assert_allclose(nbias[:m.nv], cpu_bias[:m.nv], atol=2e-4)
  # Trajectory + replay.
  sim.reset(qpos=np.asarray(qp, dtype=np.float32).reshape(1, -1),
            qvel=np.asarray(qv, dtype=np.float32).reshape(1, -1))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = qv
  mujoco.mj_forward(m, cpu)
  for _ in range(20):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  snap = sim.state.snapshot()
  traj = []
  for _ in range(10):
    sim.step(1)
    traj.append(sim.state.qpos.cpu().numpy()[0].copy())
  sim.state.restore(snap)
  for k in range(10):
    sim.step(1)
    np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], traj[k], atol=1e-7)
    mujoco.mj_step(m, cpu)
  # Rerun trajectory matches the oracle advanced from the snapshot state.
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq[:m.nq], np.asarray(cpu.qpos)[:m.nq], atol=5e-4)
