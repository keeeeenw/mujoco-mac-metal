# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Armature closeout regressions: exact probe states, ordered chains,
welded sites, and batched states (CPU + GPU)."""

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


# Exact review-probe geometry/states (hinge-z first, then hinge-y/slide-x).
PROBE_HH_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>
<site name="anchor" pos="-.3 .2 1.5"/>
<body pos=".5 0 1"><joint type="hinge" axis="0 0 1"/><joint type="hinge" axis="0 1 0"/>
<geom type="box" size=".1 .08 .06" pos=".1 .04 .03" mass="1"/>
<site name="tip" pos=".2 .1 .05"/></body></worldbody>
<tendon><spatial armature=".5"><site site="anchor"/><site site="tip"/></spatial></tendon></mujoco>"""

PROBE_HS_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>
<site name="anchor" pos="-.3 .2 1.5"/>
<body pos=".5 0 1"><joint type="hinge" axis="0 0 1"/><joint type="slide" axis="1 0 0"/>
<geom type="box" size=".1 .08 .06" pos=".1 .04 .03" mass="1"/>
<site name="tip" pos=".2 .1 .05"/></body></worldbody>
<tendon><spatial armature=".5"><site site="anchor"/><site site="tip"/></spatial></tendon></mujoco>"""

# Moving ancestor carrying an ordered same-body chain (slide then hinge).
ANCESTOR_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>
<site name="anchor" pos="-.3 .2 1.5"/>
<body pos="0 0 1"><joint name="base" type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.06" pos="0 0 0" mass="0.5"/>
<body pos="0.25 0.02 0"><joint name="s" type="slide" axis="1 0 0"/>
<joint name="h" type="hinge" axis="0 0 1"/>
<geom type="box" size="0.08 0.06 0.05" pos="0.04 0 0.02" mass="0.6"/>
<site name="tip" pos="0.12 0.06 0.03"/></body></body></worldbody>
<tendon><spatial armature=".5"><site site="anchor"/><site site="tip"/></spatial></tendon></mujoco>"""

# Welded (jointless, fixed to parent) site-bearing child in the path.
WELD_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>
<site name="anchor" pos="-.3 .2 1.5"/>
<body pos="0 0 1"><joint name="sh" type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.06" pos="0.02 0 0" mass="0.5"/>
<body pos="0.15 0.03 0.01">
<geom type="box" size="0.05 0.04 0.03" mass="0.3"/>
<site name="tip" pos="0.06 0.02 0.01"/></body></body></worldbody>
<tendon><spatial armature=".5"><site site="anchor"/><site site="tip"/></spatial></tendon></mujoco>"""

# Slide-before-hinge ordering on one body with nonzero offsets.
SH_XML = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>
<site name="anchor" pos="-.3 .2 1.5"/>
<body pos=".4 -.1 1.1"><joint name="s" type="slide" axis="0 0 1"/>
<joint name="h" type="hinge" axis="1 0 0"/>
<geom type="box" size=".09 .07 .05" pos="-.04 .02 0" mass=".8"/>
<site name="tip" pos=".1 -.05 .04"/></body></worldbody>
<tendon><spatial armature=".5"><site site="anchor"/><site site="tip"/></spatial></tendon></mujoco>"""


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("xml,sites", [
    (ANCESTOR_XML, ("anchor", "tip")),
    (WELD_XML, ("anchor", "tip")),
    (SH_XML, ("anchor", "tip")),
])
def test_closeout_chain_weld_order_gpu(xml, sites):
  # Moving ancestor + ordered chain / welded site child / slide-before-hinge:
  # dots, bias-vs-oracle, qacc, short trajectory parity, restore/replay.
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(xml)
  m0 = mujoco.MjModel.from_xml_string(xml.replace('armature=".5"', 'armature="0.0"'))
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  rng = np.random.default_rng(41)
  for trial in range(3):
    qp = (rng.random(m.nq) - 0.5) * 0.8
    qv = (rng.random(m.nv) - 0.5) * 6.0
    dots, nbias, nqacc = _native_state(sim, m, qp, qv)
    d = mujoco.MjData(m)
    d.qpos[:] = qp
    d.qvel[:] = qv
    mujoco.mj_forward(m, d)
    ref = _pinned_dot_generic(m, d, sites)
    assert abs(ref) > 1e-3, (sites, ref)
    np.testing.assert_allclose(float(dots[0]), ref, rtol=1e-5, atol=2e-5)
    d0 = mujoco.MjData(m0)
    d0.qpos[:] = qp
    d0.qvel[:] = qv
    mujoco.mj_forward(m0, d0)
    cpu_bias = np.asarray(d.qfrc_bias) - np.asarray(d0.qfrc_bias)
    np.testing.assert_allclose(nbias[:m.nv], cpu_bias[:m.nv], atol=2e-4)
    np.testing.assert_allclose(nqacc[:m.nv], np.asarray(d.qacc)[:m.nv],
                               rtol=1e-5, atol=2e-4)
  # Trajectory + restore/replay from the last state.
  sim.reset(qpos=np.asarray(qp, dtype=np.float32).reshape(1, -1),
            qvel=np.asarray(qv, dtype=np.float32).reshape(1, -1))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = qv
  mujoco.mj_forward(m, cpu)
  max_err = 0.0
  for _ in range(30):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    max_err = max(max_err, float(
        np.max(np.abs(sim.state.qpos.cpu().numpy()[0] - cpu.qpos))))
  assert max_err < 5e-4, max_err
  snap = sim.state.snapshot()
  traj = []
  for _ in range(8):
    sim.step(1)
    traj.append(sim.state.qpos.cpu().numpy()[0].copy())
  sim.state.restore(snap)
  for k in range(8):
    sim.step(1)
    np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0], traj[k], atol=1e-7)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_closeout_batched_states_gpu():
  # Batch with different states exercises per-world offsets through dots.
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(ANCESTOR_XML)
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
  qp = np.array([[0.3, 0.1, -0.2], [-0.2, 0.5, 0.1]])
  qv = np.array([[1.0, -1.5, 2.0], [0.5, 1.0, -2.0]])
  sim.reset(qpos=qp.astype(np.float32), qvel=qv.astype(np.float32))
  qpos_t, qvel_t = sim.state._qpos, sim.state._qvel
  dynamics = sim._smooth.run_device(qpos_t, qvel_t, None, None)
  kin = sim._spatial_tendons.run_kinematics(qvel_t, dynamics["poses"])
  bias, dots = sim._spatial_tendons.run_armature_bias(
      kin, qvel_t, dynamics["poses"], dynamics.get("cvel", None),
      dynamics.get("root_com", None), dynamics.get("cdof", None),
      dynamics.get("cdof_dot", None))
  dots = dots.cpu().numpy()
  for w in range(2):
    d = mujoco.MjData(m)
    d.qpos[:] = qp[w]
    d.qvel[:] = qv[w]
    mujoco.mj_forward(m, d)
    ref = _pinned_dot_generic(m, d, ("anchor", "tip"))
    assert abs(ref) > 1e-3, (w, ref)
    np.testing.assert_allclose(float(dots[w, 0]), ref, rtol=1e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("xml,sites", [
    (PROBE_HH_XML, ("anchor", "tip")),
    (PROBE_HS_XML, ("anchor", "tip")),
])
def test_closeout_probe_states_gpu(xml, sites):
  # The review probe's exact failing states: qpos [.4, -.7], qvel [1.2, -2].
  from mujoco_metal import MetalSimulation
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.array([0.4, -0.7])
  qv = np.array([1.2, -2.0])
  dots, _, nqacc = _native_state(sim, m, qp, qv)
  d = mujoco.MjData(m)
  d.qpos[:] = qp
  d.qvel[:] = qv
  mujoco.mj_forward(m, d)
  ref = _pinned_dot_generic(m, d, sites)
  assert abs(ref) > 1e-2, ref
  np.testing.assert_allclose(float(dots[0]), ref, rtol=1e-5, atol=2e-5)
  np.testing.assert_allclose(nqacc[:m.nv], np.asarray(d.qacc)[:m.nv],
                             rtol=1e-5, atol=2e-4)
