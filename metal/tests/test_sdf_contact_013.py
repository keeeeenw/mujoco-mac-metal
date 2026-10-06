# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 013: SDF contact qualification (CPU + GPU).

Native plugin-free mesh-octree SDF vs analytic geoms and SDF-vs-SDF via
Halton-seeded gradient descent (pinned mjc_SDF). Per-seed witnesses make
comparison nearest-neighbor (seeds occasionally accept/reject across the
float32/64 boundary, shifting index order). Mesh-SDF uses the pinned compiled
mesh BVH and is exercised by a native witness test. Bundled plugin SDFs use
their registered analytic distance/gradient callback with compact high/low
attribute records. Initpoint budgets follow the pinned source loop; output
capacity is separately capped at mjMAXCONPAIR (50). Plane-SDF and hfield-SDF
reserve zero slots. Margin is qualified at 0 only (pinned ignores it too).
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
from mujoco_metal.bundled_plugins import lower_bundled_plugins

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


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in mesh-SDF native BVH integration")
def test_mesh_sdf_bvh_emits_multiple_witnesses_and_replays_gpu():
  """Compare the native BVH/FPS path against the pinned CPU collision oracle."""
  import torch

  sdf_vertices = VBOX
  mesh_vertices = " ".join(str(float(v) * 0.5)
                            for v in np.fromstring(VBOX, sep=" "))
  xml = f'''<mujoco>
    <asset><mesh name="sdfmesh" vertex="{sdf_vertices}"/>
          <mesh name="collider" vertex="{mesh_vertices}"/></asset>
    <option timestep=".0005" integrator="Euler" gravity="0 0 0"
            sdf_initpoints="50" sdf_iterations="10"/>
    <worldbody>
      <geom name="sdf" type="sdf" mesh="sdfmesh"
            contype="1" conaffinity="1"/>
      <body><freejoint/><geom name="mesh" type="mesh" mesh="collider"
            contype="1" conaffinity="1"/></body>
    </worldbody>
  </mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  validate_stepping_profile(model, profile="integrated_euler_v1")
  desc = lower_coupled_constraints(model)
  assert desc.npairs == 1 and desc.ncontacts_max == 50
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")

  # mj_maxContact reserves sdf_initpoints entries, whereas mjc_MeshSDF can
  # return up to mjMAXCONPAIR (50). Use the source-safe full bound here so the
  # pinned 3.10.0 CPU oracle's pair buffer is large enough for its own output.
  # The smaller default reproducer is captured in the qualification notes.
  qpos = np.array([0, 0, 0, *QID], dtype=np.float64)
  cpu_data = mujoco.MjData(model)
  cpu_data.qpos[:] = qpos
  cpu_data.qvel[:] = 0
  mujoco.mj_forward(model, cpu_data)
  cpu = [(float(cpu_data.contact[i].dist),
          np.asarray(cpu_data.contact[i].frame[:3]).copy(),
          np.asarray(cpu_data.contact[i].pos).copy())
         for i in range(cpu_data.ncon)]
  assert len(cpu) >= 2 and any(c[0] < 0 for c in cpu), cpu
  assert len({tuple(np.round(c[2], 5)) for c in cpu}) >= 2, cpu
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))

  def witnesses():
    system = sim.assembled_system(recompute=True)
    mask = system["contact_mask"][0] > 0.5
    positions = system["contact_position"][0][mask].clone()
    distances = system["contact_distance"][0][mask].clone()
    normals = system["contact_normal"][0][mask].clone()
    assert positions.shape[0] >= 2, positions
    assert torch.isfinite(positions).all() and torch.isfinite(distances).all()
    assert torch.isfinite(normals).all()
    assert bool(torch.all(distances < 0)), distances
    # MPS does not implement torch.unique(dim=...), and this is only a
    # qualification assertion over the owned result.  Keep all collision
    # work on device; copy the final points for this host-side set check.
    unique = np.unique(
        (torch.round(positions * 1.0e5) / 1.0e5).detach().cpu().numpy(),
        axis=0)
    assert unique.shape[0] >= 2, positions
    nat = [(float(distances[i].item()),
            normals[i].detach().cpu().numpy(),
            positions[i].detach().cpu().numpy())
           for i in range(positions.shape[0])]
    assert len(nat) == len(cpu), (len(nat), len(cpu))
    _check("mesh-sdf", "press", cpu, nat)
    assert system["J"].shape[0] == 1
    return {key: system[key].clone() for key in (
        "contact_mask", "contact_position", "contact_distance",
        "contact_normal", "J", "R", "ar")}

  initial = witnesses()
  checkpoint = sim.snapshot()
  sim.step()
  sim.reset()
  sim.restore(checkpoint)
  replay = witnesses()
  for key in initial:
    torch.testing.assert_close(replay[key], initial[key], rtol=0, atol=0)


def test_mesh_sdf_source_oracle_safe_capacity_cpu():
  """Run the pinned CPU mesh-SDF oracle with its full 50-contact bound."""
  mesh_vertices = " ".join(str(float(v) * 0.5)
                            for v in np.fromstring(VBOX, sep=" "))
  xml = f'''<mujoco>
    <asset><mesh name="sdfmesh" vertex="{VBOX}"/>
          <mesh name="collider" vertex="{mesh_vertices}"/></asset>
    <option timestep=".0005" integrator="Euler" gravity="0 0 0"
            sdf_initpoints="50" sdf_iterations="10"/>
    <worldbody>
      <geom name="sdf" type="sdf" mesh="sdfmesh"
            contype="1" conaffinity="1"/>
      <body><freejoint/><geom name="mesh" type="mesh" mesh="collider"
            contype="1" conaffinity="1"/></body>
    </worldbody>
  </mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  data.qpos[:] = [0, 0, 0, *QID]
  mujoco.mj_forward(model, data)
  contacts = [(float(data.contact[i].dist),
               np.asarray(data.contact[i].frame[:3]).copy(),
               np.asarray(data.contact[i].pos).copy())
              for i in range(data.ncon)]
  assert len(contacts) >= 2, contacts
  assert all(np.isfinite(c[0]) and np.isfinite(c[1]).all() and
             np.isfinite(c[2]).all() for c in contacts)
  assert any(c[0] < 0 for c in contacts), contacts
  assert len({tuple(np.round(c[2], 5)) for c in contacts}) >= 2, contacts


def test_pinned_step_gradient_accepts_improvement_and_exits_on_increase_cpu():
  """Lock down the source Wolfe retry and post-line-search exit directions."""
  # These are the two source predicates in engine_collision_sdf.c: the line
  # search retries while dist - dist0 > wolfe, then stepGradient returns only
  # if the accepted distance increased (dist0 < dist).
  def retry(dist0, dist, wolfe):
    return dist - dist0 > wolfe

  def source_exit(dist0, dist):
    return dist0 < dist

  # A proper decrease satisfies the Armijo condition and does not take the
  # source's failure exit.
  assert not retry(1.0, 0.8, -0.1)
  assert not source_exit(1.0, 0.8)
  # A rejected, worse trial retries and, once line search bottoms out, follows
  # the pinned early-return branch.
  assert retry(1.0, 1.2, -0.1)
  assert source_exit(1.0, 1.2)


def test_pinned_step_gradient_source_bounds_are_strict_cpu():
  """Pinned mjMAXVAL equality is admitted; only values outside are rejected."""
  maxval = 1.0e10  # pinned include/mujoco/mjtype.h, exactly representable

  def source_rejects(value):
    return np.isnan(value) or value > maxval or value < -maxval

  assert not source_rejects(maxval)
  assert not source_rejects(-maxval)
  assert source_rejects(np.nextafter(maxval, np.inf))
  assert source_rejects(np.nextafter(-maxval, -np.inf))
  assert source_rejects(np.nan)


def test_sdf_stage_step_gradient_preserves_source_checks_and_trial_value():
  from pathlib import Path

  shader = (Path(__file__).resolve().parents[1] / "mujoco_metal" / "shaders"
            / "sdf_narrowphase.metal").read_text()
  assert "inline bool sdf_source_gradient_invalid(SdfVectorPair grad)" in shader
  assert "sw(1.0e10f)" in shader
  assert "sw_sign(sw_sub(component, limit)) > 0" in shader
  assert "sw_sign(sw_sub(component, negative_limit)) < 0" in shader
  assert "fabs(grad.hi) >= SDF_MAXVAL" not in shader
  assert "Pinned stepGradient evaluates the accepted trial one more time" not in shader
  scalar_step = shader.split("inline SdfWide sdf_descend(", 1)[1].split(
      "// SDF-vs-(analytic|SDF) narrow phase", 1)[0]
  assert scalar_step.count("sdf_obj_pair_jet_dist(") == 2  # dist0 + trial
  assert "x = x0;\n      return sjwide_value(dist0);" not in scalar_step
  assert "return sjwide_value(accepted_dist);" in scalar_step
  staged_step = shader.split("inline void sdf_stage_line_search(", 1)[1].split(
      "inline void sdf_stage_publish(", 1)[0]
  assert staged_step.count("sdf_obj_pair_jet_dist(") == 1  # trial only
  assert "if (increased) x = x0;" not in staged_step
  assert "SdfWide dist = sjwide_value(accepted_dist);" in staged_step


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


def test_sdf_admission_and_rejections_cpu():
  from mujoco_metal.capacity import CapacityLimits

  # Seed budget follows the admitted source loop; output capacity caps at 50.
  for initpoints, expected in ((40, 40), (51, 50), (64, 50)):
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco>{ASSETS}<option timestep="0.002" '
        f'sdf_initpoints="{initpoints}"/><worldbody>'
        '<geom name="sdf" type="sdf" mesh="b" contype="1" conaffinity="1"/>'
        '<body pos="0 0 0.3"><freejoint/>'
        '<geom name="o" type="sphere" size="0.05" conaffinity="1" contype="1"/></body>'
        '</worldbody></mujoco>')
    descriptor = lower_coupled_constraints(
        m, limits=CapacityLimits(max_slots=50, max_rows=256))
    assert descriptor.ncontacts_max == expected
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    assert data.ncon == 0
  # Mesh-SDF is admitted after native mesh/BVH upload validation. Avoid
  # calling the pinned CPU forward path here: its overlapping mesh-SDF case
  # currently crashes the private 3.10.0 wheel before returning contacts.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{ASSETS}{OPT}<worldbody>'
      '<geom name="sdf" type="sdf" mesh="b" contype="1" conaffinity="1"/>'
      '<body pos="0 0 0.3"><freejoint/>'
      '<geom name="o" type="mesh" mesh="b" contype="1" conaffinity="1"/></body>'
      '</worldbody></mujoco>')
  desc = lower_coupled_constraints(
      m, limits=CapacityLimits(max_slots=50, max_rows=256))
  assert desc.ncontacts_max > 0
  info = np.asarray(desc.mesh_hull_info, dtype=np.int32).reshape(-1, 9)
  sdf_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sdf"))
  mesh_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "o"))
  assert int(info[sdf_id, 1]) == int(m.mesh_octnum[int(m.geom_dataid[sdf_id])])
  assert int(info[mesh_id, 4]) == int(m.mesh_facenum[int(m.geom_dataid[mesh_id])])
  assert int(info[mesh_id, 8]) == int(m.mesh_bvhnum[int(m.geom_dataid[mesh_id])])
  vertex_base, face_base, bvh_base = (int(info[mesh_id, i]) for i in (5, 6, 7))
  mesh_asset = int(m.geom_dataid[mesh_id])
  nvert, nface = int(m.mesh_vertnum[mesh_asset]), int(m.mesh_facenum[mesh_asset])
  vertices = np.asarray(desc.mesh_hull[
      vertex_base:vertex_base + 3 * nvert]).reshape(-1, 3)
  faces = np.asarray(desc.mesh_hull[
      face_base:face_base + 3 * nface]).reshape(-1, 3)
  nodes = np.asarray(desc.mesh_hull[bvh_base:bvh_base + 9 * info[mesh_id, 8]])
  va, fa = int(m.mesh_vertadr[mesh_asset]), int(m.mesh_faceadr[mesh_asset])
  np.testing.assert_array_equal(
      vertices, m.mesh_vert[va:va + nvert].reshape(-1, 3))
  np.testing.assert_array_equal(
      faces, m.mesh_face[fa:fa + nface].reshape(-1, 3))
  np.testing.assert_array_equal(nodes.reshape(-1, 9)[:, 0],
      m.bvh_nodeid[int(m.mesh_bvhadr[mesh_asset]):
                   int(m.mesh_bvhadr[mesh_asset]) +
                   int(m.mesh_bvhnum[mesh_asset])])
  # plane-SDF and hfield-SDF reserve zero slots.
  m = mujoco.MjModel.from_xml_string(
      f'<mujoco>{ASSETS}{OPT}<worldbody>'
      '<geom name="pp" type="plane" size="1 1 0.1" contype="1" conaffinity="1"/>'
      '<geom name="sdf" type="sdf" mesh="b" pos="0 0 0.5" contype="1" conaffinity="1"/>'
      '</worldbody></mujoco>')
  desc = lower_coupled_constraints(m)
  assert desc.ncontacts_max == 0, desc.ncontacts_max


_PLUGIN_POSES = {
    "bolt": (-0.1, -0.1, -0.3),
    "bowl": (0.8, -0.6, 0.0),
    "gear": (0.9, 0.0, 0.0),
    "nut": (0.1, -0.4, -0.3),
    "torus": (0.3, -0.2, 0.0),
}


def _plugin_sdf_model(name):
  x, y, z = _PLUGIN_POSES[name]
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              sdf_initpoints="8" iterations="100"/>
      <extension><plugin plugin="mujoco.sdf.{name}">
        <instance name="shape"/>
      </plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="probe" pos="{x} {y} {z}">
          <freejoint/><geom name="probe_sphere" type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <contact><pair geom1="plugin_sdf" geom2="probe_sphere"/></contact>
    </mujoco>
  """)


@pytest.mark.parametrize("name", tuple(_PLUGIN_POSES))
def test_bundled_plugin_sdf_rigid_descriptor_and_cpu_contacts(name):
  """Use the pinned plugin callback plus its compact native descriptor."""
  model = _plugin_sdf_model(name)
  cpu = mujoco.MjData(model)
  mujoco.mj_forward(model, cpu)
  assert cpu.ncon == 8 and all(float(c.dist) < 0 for c in cpu.contact[:cpu.ncon])

  bundled = lower_bundled_plugins(model)
  desc = lower_coupled_constraints(model)
  plugin_geom = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "plugin_sdf"))
  info = np.asarray(desc.mesh_hull_info, dtype=np.int32).reshape(-1, 9)
  row = int(np.flatnonzero(bundled.sdf_geom_id == plugin_geom)[0])
  instance = int(bundled.sdf_geom_instance[row])
  base = int(info[plugin_geom, 5])
  assert info[plugin_geom, 0] == -2
  assert info[plugin_geom, 1] == int(bundled.instance_kind[instance])
  assert info[plugin_geom, 2] == int(model.opt.sdf_iterations)
  assert info[plugin_geom, 4] == int(model.opt.sdf_initpoints) == desc.pair_max_contacts[0]
  np.testing.assert_allclose(
      desc.mesh_hull[base:base + 5], bundled.sdf_geom_attributes[row],
      rtol=0, atol=0)
  np.testing.assert_allclose(
      desc.mesh_hull[base + 5:base + 10], bundled.sdf_geom_attributes_low[row],
      rtol=0, atol=0)
  aabb_base = int(info[plugin_geom, 3])
  np.testing.assert_allclose(
      desc.mesh_hull[aabb_base + 12 * plugin_geom:aabb_base + 12 * plugin_geom + 6],
      np.asarray(model.geom_aabb[plugin_geom], dtype=np.float32), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in bundled plugin SDF rigid contact")
@pytest.mark.parametrize("name", tuple(_PLUGIN_POSES))
def test_bundled_plugin_sdf_rigid_contacts_match_pinned_cpu(name):
  model = _plugin_sdf_model(name)
  source = mujoco.MjData(model)
  mujoco.mj_forward(model, source)
  cpu = [(float(source.contact[i].dist),
          np.asarray(source.contact[i].frame[:3]),
          np.asarray(source.contact[i].pos)) for i in range(source.ncon)]
  assert len(cpu) == 8 and min(row[0] for row in cpu) < -0.01

  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  native = sim.assembled_system()
  mask = native["contact_mask"].cpu().numpy()[0]
  dist = native["contact_distance"].cpu().numpy()[0]
  normal = native["contact_normal"].cpu().numpy()[0]
  position = native["contact_position"].cpu().numpy()[0]
  contacts = [(float(dist[i]), normal[i], position[i])
              for i in range(len(mask)) if mask[i] > 0.5]
  _check(f"plugin-{name}", "press", cpu, contacts)

  # Exercise the same admitted rigid-force route and prove the complete
  # contact/plugin query state remains replayable across a checkpoint.
  checkpoint = sim.snapshot()
  status = sim.step(1).cpu().numpy().copy()
  qpos = sim.state._qpos.cpu().numpy().copy()
  qvel = sim.state._qvel.cpu().numpy().copy()
  assert status.tolist() == [0]
  assert np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
  sim.restore(checkpoint)
  np.testing.assert_array_equal(sim.step(1).cpu().numpy(), status)
  np.testing.assert_array_equal(sim.state._qpos.cpu().numpy(), qpos)
  np.testing.assert_array_equal(sim.state._qvel.cpu().numpy(), qvel)


@pytest.mark.parametrize("name", ["bowl", "torus"])
def test_plugin_sdf_seed_boxes_use_compiled_mesh_aabb_not_callback_bounds(name):
  """Execute lowering against pinned mjc_SDF's actual geom_aabb inputs."""
  model = _plugin_sdf_model(name)
  bundled = lower_bundled_plugins(model)
  plugin_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plugin_sdf")
  compiled = np.asarray(model.geom_aabb[plugin_geom], dtype=np.float32)
  callback_bounds = np.asarray(bundled.geom_sdf_aabb[plugin_geom], dtype=np.float32)
  # These source fixtures expose the distinction rather than accidentally
  # passing because both boxes happen to be equal.
  assert not np.array_equal(compiled, callback_bounds)
  desc = lower_coupled_constraints(model)
  base = int(np.asarray(desc.mesh_hull_info).reshape(-1, 9)[plugin_geom, 3])
  for geom in range(model.ngeom):
    actual = desc.mesh_hull[base + 18 * geom:base + 18 * geom + 6]
    np.testing.assert_array_equal(
        actual, np.asarray(model.geom_aabb[geom], dtype=np.float32))


def test_compiled_sdf_seed_boxes_preserve_mjtNum_residuals():
  model = _plugin_sdf_model("bowl")
  descriptor = lower_coupled_constraints(model)
  geom = int(np.flatnonzero(np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_SDF))[0])
  base = int(descriptor.mesh_hull_info.reshape(-1, 9)[geom, 3])
  packed = descriptor.mesh_hull[base:base + 18*model.ngeom].reshape(-1, 18)
  actual = packed[:, :6].astype(np.float64) + packed[:, 6:12].astype(np.float64)
  np.testing.assert_allclose(actual, model.geom_aabb, rtol=2e-15, atol=2e-15)
  assert np.any(packed[:, 6:12] != 0)


def test_compiled_sdf_analytic_sizes_preserve_mjtNum_residuals():
  model = _model("box")
  descriptor = lower_coupled_constraints(model)
  geom = int(np.flatnonzero(np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_SDF))[0])
  base = int(descriptor.mesh_hull_info.reshape(-1, 9)[geom, 3])
  packed = descriptor.mesh_hull[base:base + 18*model.ngeom].reshape(-1, 18)
  actual = packed[:, 12:15].astype(np.float64) + packed[:, 15:18].astype(np.float64)
  np.testing.assert_allclose(actual, model.geom_size, rtol=2e-15, atol=2e-15)
  assert np.any(packed[:, 15:18] != 0)
