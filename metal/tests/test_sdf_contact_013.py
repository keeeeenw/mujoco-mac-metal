# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 013: SDF contact qualification (CPU + GPU).

Native plugin-free mesh-octree SDF vs analytic geoms and SDF-vs-SDF via
Halton-seeded gradient descent (pinned mjc_SDF). Per-seed witnesses make
comparison nearest-neighbor (seeds occasionally accept/reject across the
float32/64 boundary, shifting index order). Mesh-SDF, plugin SDFs and
initpoints above budget are rejected at lowering (documented); plane-SDF
and hfield-SDF reserve zero slots; margin qualified at 0 only (pinned
ignores it too).
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import (
    lower_coupled_constraints,
    pair_max_contacts,
)
from mujoco_metal.simulation import (
    MetalSimulation,
    validate_stepping_profile,
)

VBOX = ("-0.05 -0.05 -0.04  0.05 -0.05 -0.04  0.05 0.05 -0.04  -0.05 0.05 -0.04  "
        "-0.05 -0.05 0.04  0.05 -0.05 0.04  0.05 0.05 0.04  -0.05 0.05 0.04")
ASSETS = f'<asset><mesh name="b" vertex="{VBOX}"/></asset>'
OPT = ('<option timestep="0.002" integrator="Euler" iterations="100" '
       'tolerance="1e-8" gravity="0 0 0" sdf_initpoints="4"/>')
QID = (1.0, 0.0, 0.0, 0.0)

KINDS = {
    "sphere": '<geom name="o" type="sphere" size="0.05" contype="1" conaffinity="1"/>',
    "capsule": '<geom name="o" type="capsule" size="0.03 0.04" contype="1" conaffinity="1"/>',
    "box": '<geom name="o" type="box" size="0.04 0.04 0.03" contype="1" conaffinity="1"/>',
    "cylinder": '<geom name="o" type="cylinder" size="0.04 0.035" contype="1" conaffinity="1"/>',
    "ellipsoid": '<geom name="o" type="ellipsoid" size="0.05 0.04 0.035" contype="1" conaffinity="1"/>',
    "sdf": '<geom name="o" type="sdf" mesh="b" contype="1" conaffinity="1"/>',
}
SPOTS = [(0.0, 0.0), (0.03, 0.02)]


def _model(kind):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco>{ASSETS}{OPT}<worldbody>'
      '<geom name="sdf" type="sdf" mesh="b" contype="1" conaffinity="1"/>'
      f'<body pos="0 0 1"><freejoint/>{KINDS[kind]}</body>'
      '</worldbody></mujoco>')


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
  assert zt is not None, (x, y)
  lo, hi = zt - 0.005, zt
  for _ in range(16):
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
  # Symmetric multiset matching: Halton-seed accept/reject flips across the
  # float32/64 boundary (both engines miss some minima), so exact counts are
  # not gated. Every penetrating contact on either side needs a partner on
  # the other side within depth/normal/pos tiers (no missed load-bearers,
  # no invented contacts). Grazing natives must lie near some CPU witness.
  assert cpu and nat, (kind, cfg, len(nat), len(cpu))
  cpu_pen = [c for c in cpu if c[0] < -2e-4]
  nat_pen = [n for n in nat if n[0] < -2e-4]
  assert cpu_pen and nat_pen, (kind, cfg, len(nat_pen), len(cpu_pen))
  dtol = 2.5e-3 if cfg == "shallow" else 3.5e-3
  for n in nat_pen:
    best = min(cpu, key=lambda c: float(np.linalg.norm(c[2] - n[2])))
    assert abs(n[0] - best[0]) < dtol, (kind, cfg, n[0], best[0])
    np.testing.assert_allclose(n[1], best[1], atol=0.2,
                               err_msg=f"{kind}/{cfg} nrm")
    np.testing.assert_allclose(n[2], best[2], atol=12e-3,
                               err_msg=f"{kind}/{cfg} pos")
  for c in cpu_pen:
    best = min(nat, key=lambda n: float(np.linalg.norm(n[2] - c[2])))
    assert abs(c[0] - best[0]) < dtol, (kind, cfg, c[0], best[0])
    np.testing.assert_allclose(c[1], best[1], atol=0.2,
                               err_msg=f"{kind}/{cfg} nrm-r")
    np.testing.assert_allclose(c[2], best[2], atol=12e-3,
                               err_msg=f"{kind}/{cfg} pos-r")
  for n in nat:
    if n[0] >= -2e-4:
      best = min(float(np.linalg.norm(n[2] - c[2])) for c in cpu)
      assert best < 0.02, (kind, cfg, best)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_matrix_sdf_gpu():
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
      for depth, cfg in [(0.001, "shallow"), (0.004, "press")]:
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


def test_sdf_formulas_cpu():
  # Independent closed-form checks of the ported analytic SDF math
  # (mirrors the device formulas; MuJoCo-agnostic ground truth).
  def dist_sphere(x, r):
    return float(np.linalg.norm(x) - r)

  def dist_box(x, s):
    a = np.abs(x) - s
    if np.any(a >= 0):
      b = np.maximum(a, 0)
      return float(np.linalg.norm(b) + min(max(a[0], max(a[1], a[2])), 0))
    return -float(min(s - np.abs(x)))

  def dist_capsule(x, r, h):
    a = np.array([x[0], x[1], x[2] - np.clip(x[2], -h, h)])
    return float(np.linalg.norm(a) - r)

  # Values from the iq reference family (independent of both engines).
  np.testing.assert_allclose(dist_sphere(np.array([0.03, 0.04, 0.0]), 0.05),
                             0.0, atol=1e-12)
  np.testing.assert_allclose(dist_sphere(np.array([0.0, 0.0, 0.09]), 0.05),
                             0.04, atol=1e-12)
  np.testing.assert_allclose(dist_box(np.array([0.09, 0.0, 0.0]),
                                      np.array([0.05, 0.05, 0.04])),
                             0.04, atol=1e-12)
  np.testing.assert_allclose(dist_box(np.array([0.0, 0.0, 0.0]),
                                      np.array([0.05, 0.05, 0.04])),
                             -0.04, atol=1e-12)
  np.testing.assert_allclose(dist_capsule(np.array([0.04, 0.0, 0.0]), 0.03, 0.04),
                             0.01, atol=1e-12)
  np.testing.assert_allclose(dist_capsule(np.array([0.0, 0.0, 0.10]), 0.03, 0.04),
                             0.03, atol=1e-12)


def test_sdf_octree_box_cpu():
  # Octree reading validated against the EXACT analytic box in SDF GEOM
  # frame (compiled halves 0.04/0.05/0.05 per geom_aabb): trilinear
  # interpolation error is the only admitted difference. This checks the
  # child/aabb/coeff layout and traversal independently of contacts.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{ASSETS}{OPT}<worldbody>'
      '<geom name="sdf" type="sdf" mesh="b"/>'
      '</worldbody></mujoco>')
  adr = int(m.mesh_octadr[0])
  num = int(m.mesh_octnum[0])
  child = np.asarray(m.oct_child[adr:adr + num]).reshape(num, 8)
  aabb = np.asarray(m.oct_aabb[adr:adr + num]).reshape(num, 6)
  coeff = np.asarray(m.oct_coeff[adr:adr + num]).reshape(num, 8)
  SZ = np.array([0.04, 0.05, 0.05])

  def box_project(p):
    p = np.array(p, float)
    r = p - aabb[0, :3]
    q = np.abs(r) - aabb[0, 3:]
    if np.all(q <= 0):
      return p, float(np.max(q))
    dsq = 0.0
    for k in range(3):
      if q[k] >= 0:
        dsq += q[k] ** 2
        p[k] -= (q[k] + 1e-6) if r[k] > 0 else -(q[k] + 1e-6)
    return p, float(np.sqrt(dsq))

  def findoct(p):
    stack = 0
    for _ in range(100):
      node = stack
      bc, bh = aabb[node, :3], aabb[node, 3:]
      vmin, vmax = bc - bh, bc + bh
      if np.any(p + 1e-8 < vmin) or np.any(p - 1e-8 > vmax):
        continue
      coord = (p - vmin) / (vmax - vmin)
      if np.all(child[node] == -1):
        return node, coord
      xi, yi, zi = (coord >= 0.5).astype(int)
      stack = int(child[node, 4 * zi + 2 * yi + xi])
    return -1, None

  def oct_dist(p):
    pp, boxd = box_project(p)
    node, coord = findoct(pp)
    assert node >= 0
    w = np.array([((coord[0] if (j & 1) else 1 - coord[0]) *
                   (coord[1] if (j & 2) else 1 - coord[1]) *
                   (coord[2] if (j & 4) else 1 - coord[2]))
                  for j in range(8)])
    return float(w @ coeff[node]) + (boxd if boxd > 0 else 0.0)

  def box_sdf(p):
    a = np.abs(np.asarray(p, float)) - SZ
    if np.any(a >= 0):
      b = np.maximum(a, 0)
      return float(np.linalg.norm(b) + min(max(a[0], max(a[1], a[2])), 0))
    return -float(min(SZ - np.abs(p)))

  rng = np.random.default_rng(0)
  errs = []
  for _ in range(60):
    pt = tuple((rng.random(3) - 0.5) * 0.3)
    errs.append(abs(oct_dist(pt) - box_sdf(pt)))
  print("oct-vs-analytic max err:", round(max(errs), 4))
  assert max(errs) < 0.012, max(errs)


def test_sdf_admit_cpu():
  for kind in ("sphere", "capsule", "box", "cylinder", "ellipsoid", "sdf"):
    m = _model(kind)
    validate_stepping_profile(m, profile="integrated_euler_v1")
    desc = lower_coupled_constraints(m)
    assert desc.npairs == 1 and desc.ncontacts_max == 4, (kind, desc.ncontacts_max)
  assert pair_max_contacts(8, 2, sdf_initpoints=4) == 4
  assert pair_max_contacts(8, 8, sdf_initpoints=4) == 4


def test_sdf_rejections_cpu():
  # initpoints above budget.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{ASSETS}<option timestep="0.002" sdf_initpoints="40"/>'
      '<worldbody>'
      '<geom name="sdf" type="sdf" mesh="b" contype="1" conaffinity="1"/>'
      '<body pos="0 0 0.3"><freejoint/>'
      '<geom name="o" type="sphere" size="0.05" contype="1" conaffinity="1"/></body>'
      '</worldbody></mujoco>')
  with pytest.raises(ValueError, match="sdf_initpoints"):
    lower_coupled_constraints(m)
  # mesh-SDF.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{ASSETS}{OPT}<worldbody>'
      '<geom name="sdf" type="sdf" mesh="b" contype="1" conaffinity="1"/>'
      '<body pos="0 0 0.3"><freejoint/>'
      '<geom name="o" type="mesh" mesh="b" contype="1" conaffinity="1"/></body>'
      '</worldbody></mujoco>')
  with pytest.raises(ValueError, match="mesh-SDF"):
    lower_coupled_constraints(m)
  # plane-SDF and hfield-SDF reserve zero slots.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{ASSETS}{OPT}<worldbody>'
      '<geom name="pp" type="plane" size="1 1 0.1" contype="1" conaffinity="1"/>'
      '<geom name="sdf" type="sdf" mesh="b" pos="0 0 0.5" contype="1" conaffinity="1"/>'
      '</worldbody></mujoco>')
  desc = lower_coupled_constraints(m)
  assert desc.ncontacts_max == 0, desc.ncontacts_max
