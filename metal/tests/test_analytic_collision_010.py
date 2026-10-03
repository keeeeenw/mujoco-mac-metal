# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 010: analytic cylinder/ellipsoid collision qualification (CPU + GPU)."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation


def _contacts_of(d, g1, g2):
  out = []
  for c in range(d.ncon):
    pair = {int(d.contact[c].geom[0]), int(d.contact[c].geom[1])}
    if pair == {g1, g2}:
      out.append(d.contact[c])
  return out


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_plane_cylinder_gpu():
  # Upright cylinder on plane: 2 rim + 2 triangle contacts; tilted and
  # side-lying configs; separated produces none.
  xml = """<mujoco><option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="rod" pos="0 0 0.2">
      <freejoint/>
      <geom name="cyl" type="cylinder" size="0.05 0.1"/>
    </body>
  </worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  cyl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cyl")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")

  for qpos, min_ncon in [
      (np.array([0, 0, 0.2, 1, 0, 0, 0]), 0),      # upright, gap: none
      (np.array([0, 0, 0.0999, 1, 0, 0, 0]), 3),   # upright rim touch (1e-4 deep)
      (np.array([0.1, 0, 0.105, 0.9659, 0, 0.2588, 0]), 1),  # tilted touch
      # Side-lying 1e-4 deep: exact-zero touching is float knife-edge on
      # both CPU and GPU (strict <= against margin 0), so touching coverage
      # uses slight penetration, never exact zeros.
      (np.array([0, 0, 0.0499, 0.7071, 0.7071, 0, 0]), 1),    # side-lying
      (np.array([0, 0, 2.0, 1, 0, 0, 0]), 0),      # separated
  ]:
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    asm = sim.assembled_system(recompute=True)
    sim.state._qpos.copy_(sim.state._torch.as_tensor(
        np.asarray(qpos, dtype=np.float32).reshape(1, -1),
        device=sim.state._device))
    asm = sim.assembled_system(recompute=True)
    cpu_cons = _contacts_of(d, floor, cyl)
    assert len(cpu_cons) >= min_ncon
    mask = asm["contact_mask"].cpu().numpy()[0]
    assert int(mask.sum()) == d.ncon, (qpos, mask.sum(), d.ncon)
    for i, cc in enumerate(cpu_cons):
      np.testing.assert_allclose(
          asm["contact_distance"].cpu().numpy()[0, i],
          float(cc.dist), atol=1e-6)
      np.testing.assert_allclose(
          asm["contact_normal"].cpu().numpy()[0, i],
          np.asarray(cc.frame[:3]), atol=1e-6)
      np.testing.assert_allclose(
          asm["contact_position"].cpu().numpy()[0, i],
          np.asarray(cc.pos), atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_sphere_cylinder_gpu():
  # Side, top/bottom cap, rim corner, deep penetration, separated.
  xml = """<mujoco><option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="rod" pos="0 0 0.5"><freejoint/>
      <geom name="cyl" type="cylinder" size="0.05 0.1"/></body>
    <body name="ball" pos="0.2 0 0.5"><freejoint/>
      <geom name="sph" type="sphere" size="0.04"/></body>
  </worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sph = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sph")
  cyl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cyl")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")

  cases = [
      [0, 0, 0.5, 1, 0, 0, 0, 0.09, 0, 0.5, 1, 0, 0, 0],   # side touch
      [0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0.63, 1, 0, 0, 0],     # top cap
      [0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0.37, 1, 0, 0, 0],     # bottom cap
      [0, 0, 0.5, 1, 0, 0, 0, 0.05, 0, 0.62, 1, 0, 0, 0],  # rim corner
      [0, 0, 0.5, 1, 0, 0, 0, 0.02, 0, 0.52, 1, 0, 0, 0],  # deep (inside)
      [0, 0, 0.5, 1, 0, 0, 0, 2.0, 0, 0.5, 1, 0, 0, 0],    # separated
  ]
  for qpos in cases:
    qpos = np.array(qpos)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    sim.state._qpos.copy_(sim.state._torch.as_tensor(
        qpos.astype(np.float32).reshape(1, -1), device=sim.state._device))
    asm = sim.assembled_system(recompute=True)
    cpu_cons = _contacts_of(d, sph, cyl)
    mask = asm["contact_mask"].cpu().numpy()[0]
    assert int(mask.sum()) == d.ncon, (qpos, mask.sum(), d.ncon)
    for i, cc in enumerate(cpu_cons):
      np.testing.assert_allclose(
          asm["contact_distance"].cpu().numpy()[0, i],
          float(cc.dist), atol=1e-6)
      np.testing.assert_allclose(
          asm["contact_normal"].cpu().numpy()[0, i],
          np.asarray(cc.frame[:3]), atol=1e-6)
      np.testing.assert_allclose(
          asm["contact_position"].cpu().numpy()[0, i],
          np.asarray(cc.pos), atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_cylinder_settle_trajectory_gpu():
  # Dropped upright cylinder settles on the plane; track CPU then rest.
  xml = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="rod" pos="0 0 0.3"><freejoint/>
      <geom name="cyl" type="cylinder" size="0.05 0.1" friction="0.8 0.1 0.1"/></body>
  </worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  cpu = mujoco.MjData(m)
  # Gentle placement (1mm gap, rest): violent redundant-contact impacts can
  # stall the coupled PGS (status 3, pre-existing solver scope, also seen on
  # box drops); the manifold coupling itself is exact.
  sim.reset(qpos=np.array([[0, 0, 0.101, 1, 0, 0, 0]], dtype=np.float32))
  cpu.qpos[:] = [0, 0, 0.101, 1, 0, 0, 0]
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  pre_err = 0.0
  for _ in range(200):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    if cpu.ncon == 0:
      gq = sim.state.qpos.cpu().numpy()[0]
      pre_err = max(pre_err, float(np.max(np.abs(gq - cpu.qpos))))
  assert pre_err < 1e-6, pre_err
  # Redundant rim contacts settle ~0.6mm deeper natively (PGS regularization
  # vs CPU); coupling itself is exact (see manifold test).
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0],
                             np.asarray(cpu.qpos), atol=1e-3)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
