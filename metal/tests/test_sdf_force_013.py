# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 013: solved force/torque parity for SDF contact pairs.

Step-0 solved forces at identical states plus slide/settle trajectories.
SDF presses land several per-seed witnesses; translation resultants gate
tighter than rotation components (redundant-manifold torque split, same
treatment as the 011 face-manifold restriction). Multi-seed minima may
differ in membership across engines; trajectories use envelopes.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

_PROFILE = "integrated_euler_v1"
QID = [1.0, 0.0, 0.0, 0.0]
VBOX = ("-0.05 -0.05 -0.04  0.05 -0.05 -0.04  0.05 0.05 -0.04  -0.05 0.05 -0.04  "
        "-0.05 -0.05 0.04  0.05 -0.05 0.04  0.05 0.05 0.04  -0.05 0.05 0.04")
ASSETS = f'<asset><mesh name="b" vertex="{VBOX}"/></asset>'


def _model(body, gravity="0 0 0", body_pos="0 0 1", sdf_pos="0 0 0", ip=4,
           floor=False):
  opt = (f'<option timestep="0.002" integrator="Euler" iterations="100" '
         f'tolerance="1e-8" gravity="{gravity}" sdf_initpoints="{ip}"/>')
  floor_xml = ('<geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.1" '
               'contype="1" conaffinity="1"/>' if floor else "")
  return (f"<mujoco>{ASSETS}{opt}<worldbody>"
          f"{floor_xml}"
          f'<geom name="sdf" type="sdf" mesh="b" pos="{sdf_pos}" contype="1" '
          'conaffinity="1"/>'
          f'<body pos="{body_pos}"><freejoint/>{body}</body>'
          "</worldbody></mujoco>")


def _load(xml):
  return mujoco.MjModel.from_xml_string(xml)


_SPH = ('<geom name="ball" type="sphere" size="0.05" contype="1" '
        'conaffinity="1"/>')


def _touch(m, x, y):
  zt = None
  z = 0.5
  while z > -0.2:
    d = mujoco.MjData(m)
    d.qpos[:] = [x, y, z, *QID]
    mujoco.mj_forward(m, d)
    if d.ncon > 0:
      zt = z
      break
    z -= 0.005
  assert zt is not None
  return zt


def _parity_step0(xml, qpos0, qvel0, tag, a_atol=0.2):
  m = _load(xml)
  sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
  qpos0 = np.asarray(qpos0, dtype=np.float32).reshape(1, -1)
  qvel0 = np.asarray(qvel0, dtype=np.float32).reshape(1, -1)
  sim.reset(qpos=qpos0, qvel=qvel0)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qpos0[0]
  cpu.qvel[:] = qvel0[0]
  mujoco.mj_forward(m, cpu)
  asm = sim.assembled_system(recompute=True)
  mujoco.mj_step(m, cpu)
  f_nat = np.asarray(asm["qfrc_constraint"].cpu().numpy()[0])
  f_cpu = np.asarray(cpu.qfrc_constraint)
  # Translation resultants gate at 15% + 0.1 N (single-seed SDF depths
  # inherit voxel-interpolation spread across precisions, measured 9%);
  # rotation components carry the redundant-manifold torque split,
  # gated at force scale.
  np.testing.assert_allclose(f_nat[:3], f_cpu[:3], rtol=0.15, atol=0.1,
                             err_msg=f"{tag} qfrc lin")
  np.testing.assert_allclose(f_nat[3:], f_cpu[3:], rtol=0.2, atol=0.5,
                             err_msg=f"{tag} qfrc rot")
  a_nat = np.asarray(asm["qacc"].cpu().numpy()[0])
  a_cpu = np.asarray(cpu.qacc)
  mass = np.abs(np.asarray(
      asm["mass_matrix"].cpu().numpy()[0]).reshape(m.nv, m.nv).diagonal())
  np.testing.assert_allclose(mass * a_nat, mass * a_cpu,
                             rtol=0.05, atol=a_atol, err_msg=f"{tag} qacc")
  assert int(sim.state.status.cpu().numpy()[0]) == 0, tag
  return m, sim, cpu


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_sdf_force_press_gpu():
  # Single-seed top-face press (initpoints=1, fixed 2 mm over the top):
  # both engines descend from the same Halton seed to the same face
  # minimum, isolating the force path from multi-seed membership chaos
  # (documented in the matrix test). Touch calibration is avoided: with
  # one seed the first-touch height is a side graze, not the top face.
  # Gravity preloads the contact (weight-dominated like real scenes);
  # the penetration-driven remainder agrees to ~9% (voxel interp spread).
  xml = _model(_SPH, body_pos="0 0 0.088", ip=1, gravity="0 0 -9.81")
  qp = list(_load(xml).qpos0)
  _parity_step0(xml, qp, [0] * 6, "press/sphere", a_atol=0.6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_sdf_slide_trajectory_gpu():
  # Sphere rolls off the SDF box top, lands on the floor and settles.
  # Edge roll-off is descent-chaotic on both engines, so positions use a
  # 12 mm envelope while spin accumulates contact-phase differences.
  xml = _model(_SPH, gravity="0 0 -9.81", body_pos="-0.02 0 0.12", floor=True)
  m = _load(xml)
  sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(200):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  cq = np.asarray(cpu.qpos)
  np.testing.assert_allclose(gq[:3], cq[:3], atol=12e-3, err_msg="slide/pos")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_sdf_settle_gpu():
  # Sphere drops onto the SDF box top and rests; rest height agrees to
  # millimeters (static rest poses are friction-non-unique laterally).
  xml = _model(_SPH, gravity="0 0 -9.81", body_pos="0 0 0.2")
  m = _load(xml)
  sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(300):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  cq = np.asarray(cpu.qpos)
  np.testing.assert_allclose(gq[2], cq[2], atol=5e-3, err_msg="settle/z")
  np.testing.assert_allclose(gq[:2], cq[:2], atol=2e-2, err_msg="settle/xy")
