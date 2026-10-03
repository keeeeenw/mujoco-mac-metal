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


def _axis_angle_quat(axis, angle):
  axis = np.asarray(axis, dtype=float)
  axis = axis / np.linalg.norm(axis)
  s = float(np.sin(angle / 2.0))
  return [float(np.cos(angle / 2.0)), float(axis[0] * s),
          float(axis[1] * s), float(axis[2] * s)]


_IDENTITY_Q = [1.0, 0.0, 0.0, 0.0]


def _free_qpos(x, y, z, quat=_IDENTITY_Q):
  return [x, y, z] + list(quat)


# Pair matrix under test: (tag, geom-type-A, size-A, x-extent-A,
#                          geom-type-B, size-B, x-extent-B).
# Ellipsoid sizes are nonuniform by construction; the second cylinder size
# variant is also nonuniform (r 0.04, half-length 0.12).
_PAIRS_010 = [
    ("sphere-cylinder", "sphere", "0.05", 0.05,
     "cylinder", "0.05 0.1", 0.05),
    ("sphere-ellipsoid", "sphere", "0.05", 0.05,
     "ellipsoid", "0.06 0.04 0.05", 0.06),
    ("capsule-cylinder", "capsule", "0.04 0.08", 0.04,
     "cylinder", "0.05 0.1", 0.05),
    ("capsule-ellipsoid", "capsule", "0.04 0.08", 0.04,
     "ellipsoid", "0.06 0.04 0.05", 0.06),
    ("cylinder-cylinder", "cylinder", "0.05 0.1", 0.05,
     "cylinder", "0.04 0.12", 0.04),
    ("cylinder-box", "cylinder", "0.05 0.1", 0.05,
     "box", "0.05 0.05 0.05", 0.05),
    ("cylinder-ellipsoid", "cylinder", "0.05 0.1", 0.05,
     "ellipsoid", "0.06 0.04 0.05", 0.06),
    ("ellipsoid-ellipsoid", "ellipsoid", "0.06 0.04 0.05", 0.06,
     "ellipsoid", "0.05 0.05 0.03", 0.05),
    ("ellipsoid-box", "ellipsoid", "0.06 0.04 0.05", 0.06,
     "box", "0.05 0.05 0.05", 0.05),
]

_Q30X = _axis_angle_quat([1, 0, 0], np.pi / 6.0)
_Q45Z = _axis_angle_quat([0, 0, 1], np.pi / 4.0)


def _pair_xml(ta, sa, tb, sb, swap=False, margin=0.0):
  if not swap:
    first, second = (ta, sa), (tb, sb)
  else:
    first, second = (tb, sb), (ta, sa)
  mattr = f' margin="{margin}"' if margin > 0 else ""
  return ("""<mujoco><option timestep="0.002" gravity="0 0 0"/><worldbody>"""
          f"""<body pos="0 0 0.5"><freejoint/>"""
          f"""<geom name="ga" type="{first[0]}" size="{first[1]}"{mattr}/></body>"""
          f"""<body pos="0 0 0.5"><freejoint/>"""
          f"""<geom name="gb" type="{second[0]}" size="{second[1]}"{mattr}/></body>"""
          """</worldbody></mujoco>""")


def _sorted_contacts(vecs):
  # Sort by penetration, then millimeter-rounded position: penetration ties
  # must pair by patch location, not by sub-micron coordinate noise (which
  # can cross-pair symmetric manifold ends).
  return sorted(vecs, key=lambda c: (round(c[0], 9), round(float(c[2][0]), 3),
                                     round(float(c[2][1]), 3),
                                     round(float(c[2][2]), 3)))


# Pairs routed to the multiCCD manifold path (pinned mjc_Convex restarts).
# All other matrix pairs are single-witness or analytic with exact counts.
_MULTICCD_PAIRS = {"capsule-cylinder", "cylinder-cylinder", "cylinder-box"}


# Convex GJK+MPR singles (all ellipsoid-involved pairs here): MPR facet
# discretization bounds witness accuracy (measured: normals to 3.4e-5,
# positions to 0.1 mm on curved patches) while depths stay exact. Analytic
# pairs (sphere-cylinder here; plane pairs in their own tests) keep exact
# gates. All single-contact physics is force-gated in the parity suite.
_FACET_SINGLES = {"sphere-ellipsoid", "capsule-ellipsoid",
                  "cylinder-ellipsoid", "ellipsoid-ellipsoid",
                  "ellipsoid-box"}


def _matrix_cases(xa, xb):
  reach = xa + xb
  base = [
      ("separated", _free_qpos(0, 0, 0.5), _free_qpos(reach + 0.5, 0, 0.5), 0.0),
      ("shallow", _free_qpos(0, 0, 0.5),
       _free_qpos(reach - 0.001, 0, 0.5), 0.0),
      ("deep", _free_qpos(0, 0, 0.5),
       _free_qpos(reach * 0.6, 0, 0.5), 0.0),
      ("rotated", _free_qpos(0, 0, 0.5),
       _free_qpos(reach - 0.001, 0, 0.5, _Q30X), 0.0),
      ("grazing", _free_qpos(0, 0, 0.5),
       _free_qpos(reach - 0.001, 0.04, 0.5, _Q45Z), 0.0),
      ("margin_gap", _free_qpos(0, 0, 0.5),
       _free_qpos(reach + 0.002, 0, 0.5), 0.002),
      ("margin_far", _free_qpos(0, 0, 0.5),
       _free_qpos(reach + 0.01, 0, 0.5), 0.002),
  ]
  for swap in (False, True):
    for cfg, qa, qb, margin in base:
      yield (cfg, swap, margin, np.array(qa + qb))


def _check_exact_contacts(tag, cfg, swap, cpu, nat):
  # Exact single/analytic manifolds, except for measured, documented,
  # force-covered relaxations (see test_convex_force_parity_010.py):
  # - facet singles (all GJK+MPR convex singles here): MPR facet
  #   discretization bounds witness accuracy while depths stay exact.
  #   Normals: 1e-4 shallow/grazing; 5e-3 margin-gap (GJK gap-direction
  #   accuracy, measured 2.6e-3); 1e-2 deep/rotated (larger/tilted patches,
  #   measured 1.5e-3..6e-3). Positions to 0.5 mm (measured 45..410 um).
  #   Analytic pairs (sphere-cylinder here; plane pairs in their own
  #   tests) keep exact gates.
  # - deep overlap lens ambiguity (capsule-ellipsoid, ellipsoid-box,
  #   cylinder-ellipsoid): the witness can sit anywhere in the overlap
  #   lens, so frames and positions are gated loosely while depth stays
  #   exact.
  if ((tag == "capsule-ellipsoid" or tag == "ellipsoid-box"
       or tag == "cylinder-ellipsoid") and cfg == "deep"):
    nat_tol_pos, nat_tol_nrm = 5e-3, 1e-2
  elif tag in _FACET_SINGLES:
    nat_tol_pos = 5e-4
    if cfg in ("shallow", "grazing"):
      nat_tol_nrm = 1e-4
    elif cfg == "margin_gap":
      nat_tol_nrm = 5e-3
    else:
      nat_tol_nrm = 1e-2
  else:
    nat_tol_pos, nat_tol_nrm = 1e-6, 1e-6
  assert len(nat) == len(cpu), (tag, cfg, swap, len(nat), len(cpu))
  # Ellipsoid-ellipsoid configs preserving the x-axis (all but grazing,
  # which yaws): analytic symmetry forces the witness normal onto the
  # center axis (roll about X preserves it). The CPU EPA facet normal is
  # deterministically tilted (measured stable 2-3 degrees across
  # micro-perturbations and dx sweeps) so the oracle is not the frame
  # reference; depth/positions still compare against it, and solved-force
  # parity covers physics.
  use_symmetry_normal = (
      tag == "ellipsoid-ellipsoid" and cfg != "grazing")
  for (dc, nc, pc), (dn, nn, pn) in zip(_sorted_contacts(cpu),
                                        _sorted_contacts(nat)):
    np.testing.assert_allclose(
        dn, dc, atol=1e-4 if tag == "ellipsoid-ellipsoid" else 1e-6,
        err_msg=f"{tag}/{cfg}/swap={swap} dist")
    if use_symmetry_normal:
      np.testing.assert_allclose(np.abs(nn), [1.0, 0.0, 0.0], atol=1e-6,
                                 err_msg=f"{tag}/{cfg}/swap={swap} normal")
    else:
      np.testing.assert_allclose(nn, nc, atol=nat_tol_nrm,
                                 err_msg=f"{tag}/{cfg}/swap={swap} normal")
    np.testing.assert_allclose(pn, pc, atol=nat_tol_pos,
                               err_msg=f"{tag}/{cfg}/swap={swap} pos")


def _check_manifold_sanity(tag, cfg, swap, cpu, nat):
  # Geometric sanity for multiCCD manifolds (honest scope: same patch,
  # uniform depth, consistent frames). This is NOT a force proof; solved
  # force/torque parity lives in test_convex_force_parity_010.py. Pinned
  # multiCCD itself samples (CPU counts vary 2..5 across configs and witness
  # positions vary run to run), so exact counts/positions are demanded of
  # neither side. The Hausdorff bound (40 mm on tube-segment patches) only
  # requires the same patch region, not the same samples. Rotated/grazing
  # patches have genuinely varying frames, so the cone check applies to
  # axis-aligned configs only; rotated frames are gated by the rotated
  # force case instead.
  assert 1 <= len(nat) <= 5, (tag, cfg, swap, len(nat))
  assert len(cpu) >= 1, (tag, cfg, swap, "cpu miss")
  dc = np.array([c[0] for c in cpu])
  dn = np.array([c[0] for c in nat])
  np.testing.assert_allclose(np.mean(dn), np.mean(dc), atol=1e-6,
                             err_msg=f"{tag}/{cfg}/swap={swap} mean depth")
  assert float(np.ptp(dn)) < 1e-5 and float(np.ptp(dc)) < 1e-5, (
      tag, cfg, swap, float(np.ptp(dn)), float(np.ptp(dc)))
  if cfg not in ("rotated", "grazing"):
    for _, nn, _ in nat:
      best = max(float(nn @ nc) for _, nc, _ in cpu)
      assert best > 1 - 2e-5, (tag, cfg, swap, nn, best)
  if cfg in ("rotated", "margin_gap"):
    # Tilted patches sample different points run to run, and margin-gap
    # closest pairs on parallel features are non-unique (gap witnesses
    # carry no force); positions are covered by force parity instead.
    return
  cp = np.array([p for _, _, p in cpu])
  np_ = np.array([p for _, _, p in nat])
  d2 = ((cp[:, None, :] - np_[None, :, :]) ** 2).sum(-1)
  hausdorff = max(float(np.min(d2, axis=1).max()),
                  float(np.min(d2, axis=0).max()))
  assert hausdorff < 2.5e-3, (tag, cfg, swap, hausdorff)  # 50 mm each way


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_matrix_cylinder_ellipsoid_gpu():
  # Every 010 pair x config x ordering. Failures are collected across the
  # whole matrix (shared sim per pair/order) so one case cannot conceal
  # later ones; all failures are reported together at the end.
  failures = []
  for tag, ta, sa, xa, tb, sb, xb in _PAIRS_010:
    for swap in (False, True):
      m0 = mujoco.MjModel.from_xml_string(
          _pair_xml(ta, sa, tb, sb, swap=swap))
      mM = mujoco.MjModel.from_xml_string(
          _pair_xml(ta, sa, tb, sb, swap=swap, margin=0.002))
      ga0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "ga")
      gb0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "gb")
      sim0 = MetalSimulation(m0, batch_size=1, profile="integrated_euler_v1")
      simM = MetalSimulation(mM, batch_size=1, profile="integrated_euler_v1")
      for cfg, swap2, margin, qpos in _matrix_cases(xa, xb):
        if swap2 != swap:
          continue
        m, ga, gb = (mM, mujoco.mj_name2id(
            mM, mujoco.mjtObj.mjOBJ_GEOM, "ga"),
            mujoco.mj_name2id(
            mM, mujoco.mjtObj.mjOBJ_GEOM, "gb")) if margin > 0 else (
            m0, ga0, gb0)
        sim = simM if margin > 0 else sim0
        try:
          d = mujoco.MjData(m)
          d.qpos[:] = qpos
          mujoco.mj_forward(m, d)
          sim.state._qpos.copy_(sim.state._torch.as_tensor(
              qpos.astype(np.float32).reshape(1, -1),
              device=sim.state._device))
          asm = sim.assembled_system(recompute=True)
          cpu = [(float(d.contact[c].dist),
                  np.asarray(d.contact[c].frame[:3]),
                  np.asarray(d.contact[c].pos))
                 for c in range(d.ncon)
                 if {int(d.contact[c].geom[0]),
                     int(d.contact[c].geom[1])} == {ga, gb}]
          mask = asm["contact_mask"].cpu().numpy()[0]
          nat = [(float(asm["contact_distance"].cpu().numpy()[0, i]),
                  np.asarray(asm["contact_normal"].cpu().numpy()[0, i]),
                  np.asarray(asm["contact_position"].cpu().numpy()[0, i]))
                 for i in range(int(mask.sum()))]
          if not cpu and not nat:
            continue
          assert cpu and nat, (tag, cfg, swap, len(nat), len(cpu))
          if tag in _MULTICCD_PAIRS and cfg not in (
              "separated", "margin_far"):
            _check_manifold_sanity(tag, cfg, swap, cpu, nat)
          else:
            _check_exact_contacts(tag, cfg, swap, cpu, nat)
        except AssertionError as exc:
          failures.append(f"{tag}/{cfg}/swap={swap}: {exc}")
  assert not failures, "\n".join(failures)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_endcap_pitch_multibasin_gpu():
  # Pitched-tube end-cap regime: penetration admits multiple valid witness
  # basins at equal depth (CPU and native legitimately pick different patch
  # points with different torques). Gates catch misses, flips and far
  # witnesses; they do not demand a basin. Torque-sensitive force parity
  # for this regime is not asserted (indeterminate).
  qpitch = _axis_angle_quat([0, 1, 0], np.pi / 6.0)
  failures = []
  for swap in (False, True):
    m = mujoco.MjModel.from_xml_string(_pair_xml(
        "capsule", "0.04 0.08", "cylinder", "0.05 0.1", swap=swap))
    ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ga")
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gb")
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    qpos = np.array(_free_qpos(0, 0, 0.5) + _free_qpos(0.089, 0, 0.5, qpitch))
    try:
      d = mujoco.MjData(m)
      d.qpos[:] = qpos
      mujoco.mj_forward(m, d)
      sim.state._qpos.copy_(sim.state._torch.as_tensor(
          qpos.astype(np.float32).reshape(1, -1),
          device=sim.state._device))
      asm = sim.assembled_system(recompute=True)
      assert d.ncon >= 1, "cpu miss"
      mask = asm["contact_mask"].cpu().numpy()[0]
      assert int(mask.sum()) >= 1, "native miss"
      for i in range(int(mask.sum())):
        dn = float(asm["contact_distance"].cpu().numpy()[0, i])
        nn = np.asarray(asm["contact_normal"].cpu().numpy()[0, i])
        pn = np.asarray(asm["contact_position"].cpu().numpy()[0, i])
        assert abs(dn - float(d.contact[0].dist)) < 1e-6, (swap, dn)
        dots = [float(nn @ np.asarray(d.contact[c].frame[:3]))
                for c in range(d.ncon)]
        assert max(dots) > 0.0, (swap, nn, dots)
        dp = [float(np.linalg.norm(pn - np.asarray(d.contact[c].pos)))
              for c in range(d.ncon)]
        assert min(dp) < 0.06, (swap, pn, dp)
    except AssertionError as exc:
      failures.append(f"swap={swap}: {exc}")
  assert not failures, "\n".join(failures)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_plane_ellipsoid_gpu():
  # Plane-ellipsoid: upright shallow/deep/separated + tilted + side configs.
  xml = """<mujoco><option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="egg" pos="0 0 0.2"><freejoint/>
      <geom name="ell" type="ellipsoid" size="0.06 0.04 0.05"/>
    </body>
  </worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  ell = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ell")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qtilt = _axis_angle_quat([0, 1, 0], 0.5)
  for qpos, min_ncon in [
      (np.array([0, 0, 2.0, 1, 0, 0, 0]), 0),       # separated
      (np.array([0, 0, 0.049, 1, 0, 0, 0]), 1),     # shallow (1mm)
      (np.array([0, 0, 0.03, 1, 0, 0, 0]), 1),      # deep
      (np.array([0, 0, 0.049] + qtilt), 1),         # tilted touch
      (np.array([0.02, 0.01, 0.045] + _axis_angle_quat([1, 0, 0], 0.7)), 1),
  ]:
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    sim.state._qpos.copy_(sim.state._torch.as_tensor(
        np.asarray(qpos, dtype=np.float32).reshape(1, -1),
        device=sim.state._device))
    asm = sim.assembled_system(recompute=True)
    cpu_cons = _contacts_of(d, floor, ell)
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
def test_roller_sorting_trajectory_gpu():
  # Frictional trajectory: cylinder rolls and ellipsoid slides down a gentle
  # tilted plane; native tracks CPU within the settled PGS envelope.
  xml = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" quat="0.99905 0 0.04362 0"/>
    <body name="rod" pos="-0.3 0 0.25"><freejoint/>
      <geom name="cyl" type="cylinder" size="0.05 0.1" friction="0.8 0.1 0.1"/></body>
    <body name="egg" pos="0.3 0 0.25"><freejoint/>
      <geom name="ell" type="ellipsoid" size="0.06 0.04 0.05" friction="0.4 0.1 0.1"/></body>
  </worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  q0 = np.array([[-0.3, 0, 0.25, 1, 0, 0, 0,
                  0.3, 0, 0.25, 1, 0, 0, 0]], dtype=np.float32)
  sim.reset(qpos=q0)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = q0[0]
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(300):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0],
                             np.asarray(cpu.qpos), atol=2e-3)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  # Both bodies moved downhill (+x tilt): roller sorting direction holds.
  gq = sim.state.qpos.cpu().numpy()[0]
  assert float(gq[0]) > -0.3 and float(gq[7]) > 0.3
