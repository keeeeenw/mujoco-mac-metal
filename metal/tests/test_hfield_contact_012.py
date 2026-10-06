# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 012: heightfield contact qualification (CPU + GPU).

Native per-prism terrain collision (pinned mjc_ConvexHField) against the
admitted pair matrix (hfield vs sphere/capsule/box/cylinder/ellipsoid).
Each overlapped terrain prism yields one witness on both engines in the
same traversal order, so comparison pairs BY INDEX (same prism), unlike
mesh singles-vs-manifolds. Mesh-hfield and hfield-hfield pairs are
rejected at lowering (documented restrictions); plane-hfield reserves
zero slots. Margin qualified at 0 only.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.simulation import (
    MetalSimulation,
    validate_stepping_profile,
)

N = 9
XS = np.linspace(-0.4, 0.4, N)
ELEV = " ".join(f"{0.5 + 0.3 * np.sin(x * 9.0) * np.cos(y * 7.0):.4f}"
                for y in XS for x in XS)
HF = ('<asset><hfield name="h" nrow="9" ncol="9" '
      f'size="0.4 0.4 0.05 0.02" elevation="{ELEV}"/></asset>')
OPT = ('<option timestep="0.002" integrator="Euler" iterations="100" '
       'tolerance="1e-8" gravity="0 0 0"/>')
QID = (1.0, 0.0, 0.0, 0.0)

KINDS = {
    "sphere": '<geom name="o" type="sphere" size="0.06" contype="1" conaffinity="1"/>',
    "capsule": '<geom name="o" type="capsule" size="0.03 0.05" contype="1" conaffinity="1"/>',
    "box": '<geom name="o" type="box" size="0.05 0.05 0.03" contype="1" conaffinity="1"/>',
    "cylinder": '<geom name="o" type="cylinder" size="0.05 0.04" contype="1" conaffinity="1"/>',
    "ellipsoid": '<geom name="o" type="ellipsoid" size="0.06 0.05 0.04" contype="1" conaffinity="1"/>',
}
SLOTS = {"sphere": 8, "capsule": 8, "box": 16, "cylinder": 8, "ellipsoid": 8}

SPOTS = [(0.1, 0.05), (-0.15, 0.1), (0.0, -0.2), (0.36, 0.3)]


def _model(kind, hfx="", size=None):
  assets = HF
  if size is not None:
    assets = assets.replace('size="0.4 0.4 0.05 0.02"', f'size="{size}"')
  return mujoco.MjModel.from_xml_string(
      f'<mujoco>{assets}{OPT}<worldbody>'
      f'<geom name="terrain" type="hfield" hfield="h" contype="1" conaffinity="1"{hfx}/>'
      f'<body pos="0 0 1"><freejoint/>{KINDS[kind]}</body>'
      '</worldbody></mujoco>')


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
  assert zt is not None, (x, y)
  lo, hi = zt - 0.005, zt
  for _ in range(18):
    mid = 0.5 * (lo + hi)
    d = mujoco.MjData(m)
    d.qpos[:] = [x, y, mid, *QID]
    mujoco.mj_forward(m, d)
    if d.ncon > 0:
      lo = mid
    else:
      hi = mid
  return lo


def _check(kind, cfg, cpu, nat):
  # Same prisms in the same order: pair penetrating contacts by index.
  # Knife-edge rule (010 precedent): exact-touching witnesses (dist ~ 0)
  # flip across float32/64 and execution environments while carrying no
  # force, so only penetrating contacts (dist < -2e-4, safely above float
  # noise and below the 1 mm shallow presses) gate strictly. Grazing CPU
  # entries may be absent natively; grazing natives must lie near some CPU
  # witness (no invented contacts). Box witnesses tier by face
  # representation (flat-face-vs-triangulation placement is underdetermined;
  # depths still gate). Deep presses admit EPA-basin spread on rough cells.
  cpu_pen = [c for c in cpu if c[0] < -2e-4]
  nat_pen = [n for n in nat if n[0] < -2e-4]
  assert len(nat_pen) == len(cpu_pen) and nat_pen, \
      (kind, cfg, len(nat_pen), len(cpu_pen), len(nat), len(cpu))
  for k, (n, c) in enumerate(zip(nat_pen, cpu_pen)):
    if kind == "box":
      dtol = 2.5e-3 if cfg == "shallow" else 8e-3
    else:
      dtol = 5e-4 if cfg == "shallow" else 1e-3
    assert abs(n[0] - c[0]) < dtol, (kind, cfg, k, n[0], c[0])
  if cfg == "shallow":
    ntol = {"sphere": 0.1, "cylinder": 0.1, "capsule": 0.15,
            "ellipsoid": 0.15, "box": 0.5}[kind]
  else:
    # Press on rough cells admits EPA-basin spread; border-edge prisms add
    # triangulation-edge ambiguity (measured 0.16 ellipsoid, 0.45 capsule).
    ntol = {"sphere": 0.4, "cylinder": 0.15, "capsule": 0.5,
            "ellipsoid": 0.2, "box": 1.0}[kind]
  for k, (n, c) in enumerate(zip(nat_pen, cpu_pen)):
    np.testing.assert_allclose(n[1], c[1], atol=ntol,
                               err_msg=f"{kind}/{cfg}[{k}] nrm")
  if kind == "box":
    # Face-representation tier: same-plane witnesses, tangential spread
    # bounded by the triangulation cell.
    for k, (n, c) in enumerate(zip(nat_pen, cpu_pen)):
      gap = float(np.dot(n[2] - c[2], c[1]))
      assert abs(gap) < 3e-3, (kind, cfg, k, gap)
      tang = float(np.linalg.norm((n[2] - c[2]) - gap * c[1]))
      assert tang < 0.06, (kind, cfg, k, tang)
    for n in nat:
      best = min(float(np.linalg.norm(n[2] - c[2])) for c in cpu)
      assert best < 0.08, (kind, cfg, best)
  else:
    for k, (n, c) in enumerate(zip(nat_pen, cpu_pen)):
      # 12 mm bounds edge-basin witness spread on rough/border cells while
      # still catching wrong-prism bugs (cells are 100 mm).
      np.testing.assert_allclose(n[2], c[2], atol=12e-3,
                                 err_msg=f"{kind}/{cfg}[{k}] pos")
    for n in nat:
      best = min(float(np.linalg.norm(n[2] - c[2])) for c in cpu)
      assert best < 0.02, (kind, cfg, best)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_matrix_hfield_gpu():
  failures = []
  for kind in KINDS:
    m = _model(kind)
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    for (x, y) in SPOTS:
      try:
        zt = _touch(m, x, y)
      except AssertionError as exc:
        failures.append(f"{kind}@({x},{y}): no CPU touch: {exc}")
        continue
      for depth, cfg in [(0.001, "shallow"), (0.005, "press")]:
        z = zt - depth
        try:
          qp = np.array([x, y, z, *QID], dtype=np.float32)
          d = mujoco.MjData(m)
          d.qpos[:] = qp
          d.qvel[:] = 0
          mujoco.mj_forward(m, d)
          sim.reset(qpos=qp.reshape(1, -1),
                    qvel=np.zeros((1, m.nv), dtype=np.float32))
          assert int(sim.state.status.cpu().numpy()[0]) == 0, (kind, cfg)
          asm = sim.assembled_system()
          cpu = [(float(d.contact[c].dist),
                  np.asarray(d.contact[c].frame[:3]),
                  np.asarray(d.contact[c].pos)) for c in range(d.ncon)]
          mask = asm["contact_mask"].cpu().numpy()[0]
          nat = [(float(asm["contact_distance"].cpu().numpy()[0, i]),
                  np.asarray(asm["contact_normal"].cpu().numpy()[0, i]),
                  np.asarray(asm["contact_position"].cpu().numpy()[0, i]))
                 for i in range(len(mask)) if mask[i] > 0.5]
          if not cpu and not nat:
            continue
          _check(kind, cfg, cpu, nat)
        except AssertionError as exc:
          failures.append(f"{kind}@({x},{y})/{cfg}: {exc}")
  assert not failures, "\n".join(failures)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_hfield_frames_gpu():
  # Rotated (yaw+pitch) and scaled terrain: world-frame prism handling.
  failures = []
  for hfx, size, spots in [
      (' quat="0.9659258 0 0 0.2588190"', None, [(0.1, 0.05)]),
      (' quat="0.9914449 0.1305262 0 0"', None, [(0.1, 0.05)]),
      ("", "0.8 0.8 0.1 0.04", [(0.2, 0.1)]),
  ]:
    m = _model("sphere", hfx=hfx, size=size)
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    for (x, y) in spots:
      try:
        zt = _touch(m, x, y)
      except AssertionError as exc:
        failures.append(f"frames{hfx}{size}@({x},{y}): {exc}")
        continue
      qp = np.array([x, y, zt - 0.002, *QID], dtype=np.float32)
      try:
        d = mujoco.MjData(m)
        d.qpos[:] = qp
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        sim.reset(qpos=qp.reshape(1, -1),
                  qvel=np.zeros((1, m.nv), dtype=np.float32))
        assert int(sim.state.status.cpu().numpy()[0]) == 0
        asm = sim.assembled_system()
        cpu = [(float(d.contact[c].dist),
                np.asarray(d.contact[c].frame[:3]),
                np.asarray(d.contact[c].pos)) for c in range(d.ncon)]
        mask = asm["contact_mask"].cpu().numpy()[0]
        nat = [(float(asm["contact_distance"].cpu().numpy()[0, i]),
                np.asarray(asm["contact_normal"].cpu().numpy()[0, i]),
                np.asarray(asm["contact_position"].cpu().numpy()[0, i]))
               for i in range(len(mask)) if mask[i] > 0.5]
        _check("sphere", "shallow", cpu, nat)
      except AssertionError as exc:
        failures.append(f"frames{hfx}{size}@({x},{y}): {exc}")
  assert not failures, "\n".join(failures)


def test_hfield_admit_cpu():
  for kind, slots in SLOTS.items():
    m = _model(kind)
    validate_stepping_profile(m, profile="integrated_euler_v1")
    desc = lower_coupled_constraints(m)
    assert desc.npairs == 1 and desc.ncontacts_max == slots, (kind, desc.ncontacts_max)


def test_hfield_rejections_cpu():
  # Oversize dims.
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><asset><hfield name="big" nrow="80" ncol="80" '
      'size="1 1 0.2 0.05"/></asset>'
      f'{OPT}<worldbody>'
      '<geom name="terrain" type="hfield" hfield="big" contype="1" conaffinity="1"/>'
      '<body pos="0 0 1"><freejoint/>'
      '<geom name="o" type="sphere" size="0.06" contype="1" conaffinity="1"/></body>'
      '</worldbody></mujoco>')
  with pytest.raises(ValueError, match="bounds"):
    lower_coupled_constraints(m)
  # hfield-hfield.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{HF}{OPT}<worldbody>'
      '<body pos="0 0 0.5"><freejoint/>'
      '<geom name="t1" type="hfield" hfield="h" contype="1" conaffinity="1"/></body>'
      '<body pos="0 0 1.5"><freejoint/>'
      '<geom name="t2" type="hfield" hfield="h" contype="1" conaffinity="1"/></body>'
      '</worldbody></mujoco>')
  with pytest.raises(ValueError, match="heightfield-heightfield"):
    lower_coupled_constraints(m)
  # mesh-hfield.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{HF}<asset><mesh name="bbox" vertex="-0.04 -0.04 -0.03  0.04 -0.04 -0.03  '
      '0.04 0.04 -0.03  -0.04 0.04 -0.03  -0.04 -0.04 0.03  0.04 -0.04 0.03  '
      '0.04 0.04 0.03  -0.04 0.04 0.03"/></asset>{OPT}<worldbody>'
      '<geom name="terrain" type="hfield" hfield="h" contype="1" conaffinity="1"/>'
      '<body pos="0 0 0.3"><freejoint/>'
      '<geom name="o" type="mesh" mesh="bbox" contype="1" conaffinity="1"/></body>'
      '</worldbody></mujoco>')
  # Mesh-heightfield now uses the pinned common-CCD terrain pipeline. Its
  # source manifold capacity is50 and must survive ordinary host lowering.
  desc = lower_coupled_constraints(m)
  assert desc.npairs == 1 and desc.ncontacts_max == 50
  assert desc.pair_max_contacts.tolist() == [50]
  # plane-hfield reserves zero slots (mobile terrain overlapping a plane).
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{HF}{OPT}<worldbody>'
      '<geom name="pp" type="plane" size="1 1 0.1" pos="0 0 0.3" contype="1" conaffinity="1"/>'
      '<body pos="0 0 0"><freejoint/>'
      '<geom name="terrain" type="hfield" hfield="h" contype="1" conaffinity="1"/></body>'
      '</worldbody></mujoco>')
  desc = lower_coupled_constraints(m)
  assert desc.npairs == 1 and desc.ncontacts_max == 0, (desc.npairs, desc.ncontacts_max)
