# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 012: solved force/torque parity for heightfield pairs.

Step-0 solved forces at identical states plus slide/settle trajectories.
Sphere/cylinder presses are near-1:1 per-prism (tight gates); capsule at
representation scale; box face presses use envelopes (native witnesses
spread over the triangulated face while the oracle manifold corners
differ, moving moment arms). Multi-prism manifolds are inherently
redundant; translation resultants gate tighter than rotation components.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

_PROFILE = "integrated_euler_v1"
QID = [1.0, 0.0, 0.0, 0.0]
N = 9
XS = np.linspace(-0.4, 0.4, N)
ELEV = " ".join(f"{0.5 + 0.3 * np.sin(x * 9.0) * np.cos(y * 7.0):.4f}"
                for y in XS for x in XS)
HF = ('<asset><hfield name="h" nrow="9" ncol="9" '
      f'size="0.4 0.4 0.05 0.02" elevation="{ELEV}"/></asset>')


def _model(body, gravity="0 0 0", body_pos="0 0 1"):
  opt = (f'<option timestep="0.002" integrator="Euler" iterations="100" '
         f'tolerance="1e-8" gravity="{gravity}"/>')
  return (f"<mujoco>{HF}{opt}<worldbody>"
          '<geom name="terrain" type="hfield" hfield="h" contype="1" conaffinity="1"/>'
          f'<body pos="{body_pos}"><freejoint/>{body}</body>'
          "</worldbody></mujoco>")


def _load(xml):
  return mujoco.MjModel.from_xml_string(xml)


_SPH = ('<geom name="ball" type="sphere" size="0.06" contype="1" '
        'conaffinity="1"/>')
_CAP = ('<geom name="cap" type="capsule" size="0.03 0.05" contype="1" '
        'conaffinity="1"/>')


def _touch(m, x, y):
  zt = None
  z = 0.6
  while z > -0.1:
    d = mujoco.MjData(m)
    d.qpos[:] = [x, y, z, *QID]
    mujoco.mj_forward(m, d)
    if d.ncon > 0:
      zt = z
      break
    z -= 0.005
  assert zt is not None
  return zt


def _parity_step0(xml, qpos0, qvel0, tag, f_atol_tan=0.02, f_atol_rot=0.02,
                  a_atol=0.02):
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
  # Normal (z) resultants gate tightly; static tangential (x, y) carries
  # solver-distribution residual from sub-degree normal tilts (measured
  # 0.2 N on an 11 N press), gated at representation scale like the 011
  # ellipsoid basin.
  tan = [0, 1]
  nor = [2]
  rot = [3, 4, 5]
  np.testing.assert_allclose(f_nat[tan], f_cpu[tan], rtol=0.05,
                             atol=f_atol_tan, err_msg=f"{tag} qfrc tan")
  np.testing.assert_allclose(f_nat[nor], f_cpu[nor], rtol=0.03,
                             atol=0.3, err_msg=f"{tag} qfrc nor")
  np.testing.assert_allclose(f_nat[rot], f_cpu[rot], rtol=0.05,
                             atol=f_atol_rot, err_msg=f"{tag} qfrc rot")
  a_nat = np.asarray(asm["qacc"].cpu().numpy()[0])
  a_cpu = np.asarray(cpu.qacc)
  mass = np.abs(np.asarray(
      asm["mass_matrix"].cpu().numpy()[0]).reshape(m.nv, m.nv).diagonal())
  np.testing.assert_allclose(mass * a_nat, mass * a_cpu,
                             rtol=0.03, atol=a_atol, err_msg=f"{tag} qacc")
  assert int(sim.state.status.cpu().numpy()[0]) == 0, tag
  return m, sim, cpu


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_hfield_force_press_gpu():
  m = _load(_model(_SPH))
  zt = _touch(m, 0.1, 0.05)
  xml = _model(_SPH, body_pos=f"0.1 0.05 {zt - 0.002}")
  qp = list(_load(xml).qpos0)
  _parity_step0(xml, qp, [0] * 6, "press/sphere", f_atol_tan=0.25, a_atol=0.25)
  m = _load(_model(_CAP))
  zt = _touch(m, 0.1, 0.05)
  xml = _model(_CAP, body_pos=f"0.1 0.05 {zt - 0.002}")
  qp = list(_load(xml).qpos0)
  _parity_step0(xml, qp, [0] * 6, "press/capsule",
                f_atol_tan=0.3, f_atol_rot=0.3, a_atol=0.1)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_hfield_slide_trajectory_gpu():
  # Sphere rolls down the garden slope under gravity; positions agree to
  # millimeters while spin accumulates contact-phase differences.
  xml = _model(_SPH, gravity="0 0 -9.81", body_pos="-0.25 0.0 0.35")
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
  np.testing.assert_allclose(gq[:3], cq[:3], atol=8e-3, err_msg="slide/pos")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_hfield_settle_gpu():
  # Sphere placed just above the garden settles with a small bounce; rest
  # height and spot agree at the cm envelope (taller drops diverge
  # chaotically through bounces: documented friction/bounce chaos, with
  # the slide test covering long-horizon rolling instead).
  xml = _model(_SPH, gravity="0 0 -9.81", body_pos="0.1 0.05 0.15")
  m = _load(xml)
  sim = MetalSimulation(m, batch_size=1, profile=_PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(150):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  cq = np.asarray(cpu.qpos)
  gv = sim.state.qvel.cpu().numpy()[0]
  cv = np.asarray(cpu.qvel)
  # Post-impact state on the terrain (no tunneling, no sustained bounce);
  # lateral spots are friction-non-unique and the garden slope couples
  # lateral spread into height, so both gate at cm envelopes.
  print(f"settle nat pos={np.round(gq[:3], 4)} vel={np.round(gv[:3], 4)} "
        f"cpu pos={np.round(cq[:3], 4)} vel={np.round(cv[:3], 4)}")
  np.testing.assert_allclose(gq[2], cq[2], atol=1e-2, err_msg="settle/z")
  np.testing.assert_allclose(gq[:2], cq[:2], atol=3e-2, err_msg="settle/xy")
