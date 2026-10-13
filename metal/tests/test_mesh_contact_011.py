# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 011 baseline convex mesh contact checks.

Compiled convex hulls use pinned native GJK/EPA and alternate libCCD support.
These original fixtures retain their strict source geometry assertions. Complete
manifold, imported asset, degeneracy, full-mass and lifecycle coverage is in the
additional mesh asset/material/boundary/graph qualification files.
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

TETRA = "0 0 0  1 0 0  0 1 0  0 0 1"
BOX8 = ("-1 -0.5 -0.25  1 -0.5 -0.25  1 0.5 -0.25  -1 0.5 -0.25  "
        "-1 -0.5 0.25  1 -0.5 0.25  1 0.5 0.25  -1 0.5 0.25")
OCTA = "1 0 0  -1 0 0  0 1 0  0 -1 0  0 0 0.5  0 0 -0.5"

OPT = ('<option timestep="0.002" integrator="Euler" iterations="100" '
       'tolerance="1e-8" gravity="0 0 0"/>')
QID = (1.0, 0.0, 0.0, 0.0)
Q15X = (0.9914449, 0.1305262, 0.0, 0.0)
# Yaw 30 degrees about z: spins the bottom triangle in-plane (new witness
# basin) while keeping bottom-face height exact, so overlap stays certain
# (tilt drops corners outside small bodies laterally: placement lottery).
Q30Z = (0.9659258, 0.0, 0.0, 0.2588190)

# Other-body top offset above its origin (identity orientation).
_TOP = {"sphere": 0.09, "capsule": 0.15, "box": 0.10, "ellipsoid": 0.06,
        "cylinder": 0.08}


def _other_xml(kind):
  sizes = {"sphere": 'size="0.09"', "capsule": 'size="0.05 0.1"',
           "box": 'size="0.1 0.1 0.1"',
           "ellipsoid": 'size="0.09 0.07 0.06"',
           "cylinder": 'size="0.07 0.08"'}
  if kind == "mesh":
    return ('<geom name="mb" type="mesh" mesh="hull" contype="1" '
            'conaffinity="1"/>')
  return (f'<geom name="mb" type="{kind}" {sizes[kind]} contype="1" '
          f'conaffinity="1"/>')


def _model(tag, mesh_verts=TETRA):
  other = tag.split("-")[1]
  second = (f"<body pos=\"0 0 0.25\"><freejoint/>{_other_xml(other)}</body>"
            if other != "plane" else "")
  # Floor excluded from contact (contype 0) except in the mesh-plane case,
  # where it is the pair under test: pair isolation, so native slot counts
  # compare against the CPU pair exactly.
  fmask = ('contype="1" conaffinity="1"' if other == "plane"
           else 'contype="0" conaffinity="0"')
  return mujoco.MjModel.from_xml_string(
      f"<mujoco><asset><mesh name=\"hull\" vertex=\"{mesh_verts}\"/></asset>"
      f"{OPT}<worldbody>"
      # Floor excluded from contact (contype 0) except in the mesh-plane
      # case, where it is the pair under test: pair isolation, so native
      # slot counts compare against the CPU pair exactly.
      f"<geom name=\"floor\" type=\"plane\" size=\"5 5 0.1\" {fmask}/>"
      "<body pos=\"0 0 0.3\"><freejoint/>"
      "<geom name=\"ma\" type=\"mesh\" mesh=\"hull\" contype=\"1\" "
      "conaffinity=\"1\"/></body>"
      f"{second}</worldbody></mujoco>")


def _configs(tag):
  """Full qpos vectors (mesh free x quat + other free x quat).

  Placements target real penetration (1 mm shallow, 30 mm deep) computed
  from identity-orientation top offsets; non-separated configs assert the
  CPU sees contact so missed placements fail loudly instead of skipping.
  """
  other = tag.split("-")[1]
  top = _TOP.get(other, 0.0)
  out = []
  # Tetra bottom sits at body z (identity).
  out.append(("separated", [0, 0, 0.5, *QID, 0, 0, 0.25, *QID]))
  out.append(("shallow", [0, 0, 0.0, *QID, 0, 0, 0.001 - top, *QID]))
  out.append(("deep", [0, 0, 0.0, *QID, 0, 0, 0.03 - top, *QID]))
  out.append(("rotated", [0, 0, 0.0, *Q30Z, 0, 0, 0.03 - top, *QID]))
  out.append(("vertex", [0, 0, 0.0, *QID, 0, 0, 0.001 - top, *QID]))
  out.append(("edge", [0.5, 0, 0.0, *QID, 0.5, 0, 0.001 - top, *QID]))
  if tag == "mesh-mesh":
    out = [("separated", [0, 0, 0.3, *QID, 2.5, 0, 0.3, *QID]),
           ("shallow", [0, 0, 0.3, *QID, 0.99, 0, 0.3, *QID]),
           ("deep", [0, 0, 0.3, *QID, 0.1, 0, 0.3, *QID]),
           ("rotated", [0, 0, 0.3, *Q15X, 0.3, 0, 0.3, *QID])]
  if tag == "mesh-plane":
    out = [("separated", [0, 0, 0.5, *QID]),
           ("shallow", [0, 0, -0.001, *QID]),
           ("deep", [0, 0, -0.05, *QID]),
           ("rotated", [0, 0, -0.1, *Q15X]),
           ("flipped", [0, 0, -0.5, 0.0, 1.0, 0.0, 0.0])]
  return out


def _one_to_one(cpu, nat):
  """Return the permutation ``perm`` pairing nat[i] with cpu[perm[i]].

  Witnesses are paired by minimum total position distance, one native
  witness per source witness, so a duplicated native witness cannot hide a
  lost source witness. Manifolds hold at most the pinned pair maximum (5)
  entries, so exhaustive search is exact and cheap.
  """
  import itertools
  assert len(cpu) == len(nat), ("witness count", len(nat), len(cpu))
  best, best_cost = None, np.inf
  for perm in itertools.permutations(range(len(cpu))):
    cost = sum(float(np.linalg.norm(nat[i][2] - cpu[j][2]))
               for i, j in enumerate(perm))
    if cost < best_cost:
      best, best_cost = perm, cost
  return best


def _check_single(tag, cfg, cpu, nat, max_contacts=None):
  # Manifold contract: the native count may not exceed the pinned
  # per-pair producer maximum (``pair_max_contacts``; mesh-cylinder emits five
  # valid witnesses) and, since every fixture has a unique source manifold,
  # must equal the CPU cardinality. Witnesses are matched one-to-one and each
  # matched witness is compared on depth, normal and position.
  assert cpu and nat, (tag, cfg, len(nat), len(cpu))
  if max_contacts is not None:
    assert len(nat) <= max_contacts, (tag, cfg, len(nat), max_contacts)
    assert len(cpu) <= max_contacts, (tag, cfg, len(cpu), max_contacts)
  dtol = 2e-4
  ntol = 5e-2 if tag == "mesh-ellipsoid" else 5e-3
  perm = _one_to_one(cpu, nat)
  for i, j in enumerate(perm):
    nd, nn, np_ = nat[i]
    cd, cn, cp = cpu[j]
    assert abs(nd - cd) < dtol, (tag, cfg, "depth", i, nd, cd)
    np.testing.assert_allclose(nn, cn, atol=ntol, err_msg=f"{tag}/{cfg} nrm")
    np.testing.assert_allclose(np_, cp, atol=2e-3, err_msg=f"{tag}/{cfg} pos")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_pair_matrix_mesh_gpu():
  tags = ["mesh-plane", "mesh-sphere", "mesh-capsule", "mesh-box",
          "mesh-ellipsoid", "mesh-cylinder", "mesh-mesh"]
  failures = []
  for tag in tags:
    m = _model(tag)
    ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ma")
    name_b = "floor" if tag == "mesh-plane" else "mb"
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name_b)
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    pair_max = int(lower_coupled_constraints(m).pair_max_contacts[0])
    assert int(sim._coupled_constraints.descriptor.ncontacts_max) == pair_max
    for cfg, qpos in _configs(tag):
      try:
        qp = np.array(qpos, dtype=np.float32)
        d = mujoco.MjData(m)
        d.qpos[:] = qp
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        sim.reset(qpos=qp.reshape(1, -1),
                  qvel=np.zeros((1, m.nv), dtype=np.float32))
        assert int(sim.state.status.cpu().numpy()[0]) == 0, (tag, cfg)
        asm = sim.assembled_system()
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
               for i in range(len(mask)) if mask[i] > 0.5]
        if cfg == "separated":
          assert not cpu and not nat, (tag, cfg, len(nat), len(cpu))
          continue
        # Non-separated placements must truly overlap (or the config is a
        # mis-designed skip, not coverage).
        assert cpu, (tag, cfg, "placement misses; fix config")
        _check_single(tag, cfg, cpu, nat, max_contacts=pair_max)
      except AssertionError as exc:
        failures.append(f"{tag}/{cfg}: {exc}")
  assert not failures, "\n".join(failures)


def _notch_model():
  # Preserved-concave L extrusion (MuJoCo keeps all 24 faces; the CPU
  # collides it by the compiled convex hull, which native lowering preserves).
  v = [(0, 0, 0), (2, 0, 0), (2, 0, 1), (1.5, 0, 1), (1.5, 0, 0.5),
       (0.5, 0, 0.5), (0.5, 0, 1), (0, 0, 1)]
  v += [(x, 1, z) for (x, y, z) in v]
  quads = [(0, 1, 9, 8), (1, 2, 10, 9), (2, 3, 11, 10), (3, 4, 12, 11),
           (4, 5, 13, 12), (5, 6, 14, 13), (6, 7, 15, 14), (7, 0, 8, 15)]
  faces = []
  for a, b, c, dd in quads:
    faces += [(a, b, c), (a, c, dd)]
  faces += [(0, 5, 4), (0, 4, 3), (0, 3, 2), (0, 2, 1),
            (8, 9, 10), (8, 10, 11), (8, 11, 12), (8, 12, 13)]
  vert = " ".join(f"{x} {y} {z}" for (x, y, z) in v)
  face = " ".join(f"{a} {b} {c}" for (a, b, c) in faces)
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><asset><mesh name="L" vertex="{vert}" face="{face}"/>'
      f"</asset>{OPT}"
      '<worldbody><body><freejoint/>'
      '<geom name="ma" type="mesh" mesh="L" contype="1" conaffinity="1"/>'
      '</body><body pos="5 0 0"><freejoint/>'
      '<geom name="mb" type="sphere" size="0.1" contype="1" conaffinity="1"/>'
      '</body></worldbody></mujoco>')


def test_mesh_rejections_cpu():
  model = _notch_model()
  desc = lower_coupled_constraints(model)
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ma")
  header = int(model.mesh_graphadr[0])
  hull_count = int(model.mesh_graph[header])
  hull_ids = model.mesh_graph[header + 2 + hull_count:header + 2 + 2 * hull_count]
  base, count = desc.mesh_hull_info[9 * geom:9 * geom + 2]
  assert count == hull_count
  np.testing.assert_array_equal(
      desc.mesh_hull[3 * base:3 * (base + count)].reshape(count, 3),
      model.mesh_vert[int(model.mesh_vertadr[0]) + hull_ids])
  # A missing collision graph must not use concave surface faces as a hull.
  model.mesh_graphadr[0] = -1
  with pytest.raises(ValueError, match="non-convex"):
    lower_coupled_constraints(model)
  # Every Fibonacci-sphere vertex belongs to the hull, so the compiled hull
  # actually exceeds64; random interior points would not exercise this cap.
  points = []
  for i in range(70):
    z = 1 - 2 * (i + .5) / 70
    phi = i * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z*z)
    points.append((radius*np.cos(phi), radius*np.sin(phi), z))
  verts = " ".join(str(value) for point in points for value in point)
  m2 = mujoco.MjModel.from_xml_string(
      f'<mujoco><asset><mesh name="big" vertex="{verts}"/></asset>{OPT}'
      '<worldbody><body><freejoint/>'
      '<geom name="ma" type="mesh" mesh="big" contype="1" conaffinity="1"/>'
      '</body><body pos="2 0 0"><freejoint/>'
      '<geom name="mb" type="sphere" size="0.1" contype="1" conaffinity="1"/>'
      '</body></worldbody></mujoco>')
  assert int(m2.mesh_graph[int(m2.mesh_graphadr[0])]) == 70
  large = lower_coupled_constraints(m2)
  assert int(large.mesh_hull_info[1]) == 70
  assert large.mesh_hull.size > 3 * 70


def test_mesh_assets_admit_cpu():
  for verts in (TETRA, BOX8, OCTA):
    m = mujoco.MjModel.from_xml_string(
        f"<mujoco><asset><mesh name=\"hull\" vertex=\"{verts}\"/></asset>"
        f"{OPT}<worldbody>"
        "<body pos=\"0 0 0.3\"><freejoint/>"
        "<geom name=\"ma\" type=\"mesh\" mesh=\"hull\" contype=\"1\" "
        "conaffinity=\"1\"/></body>"
        "<body pos=\"0 0 0.25\"><freejoint/>"
        '<geom name=\"mb\" type=\"sphere\" size=\"0.09\" contype=\"1\" '
        'conaffinity=\"1\"/></body></worldbody></mujoco>')
    validate_stepping_profile(m, profile="integrated_euler_v1")
    desc = lower_coupled_constraints(m)
    assert desc.npairs == 1 and desc.ncontacts_max == 1


def _cpu_witnesses(tag, cfg):
  m = _model(tag)
  ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ma")
  gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                         "floor" if tag == "mesh-plane" else "mb")
  qpos = dict(_configs(tag))[cfg]
  d = mujoco.MjData(m)
  d.qpos[:] = np.array(qpos, dtype=np.float64)
  mujoco.mj_forward(m, d)
  out = [(float(d.contact[c].dist), np.asarray(d.contact[c].frame[:3]).copy(),
          np.asarray(d.contact[c].pos).copy())
         for c in range(d.ncon)
         if {int(d.contact[c].geom[0]), int(d.contact[c].geom[1])} == {ga, gb}]
  return out, int(lower_coupled_constraints(m).pair_max_contacts[0])


_SELF_ORACLE = [("mesh-plane", "flipped", 3), ("mesh-cylinder", "shallow", 5),
                ("mesh-cylinder", "deep", 5)]


@pytest.mark.parametrize("tag,cfg,count", _SELF_ORACLE)
def test_cpu_self_oracle_manifold_contract(tag, cfg, count):
  cpu, pair_max = _cpu_witnesses(tag, cfg)
  assert len(cpu) == count <= pair_max
  # Reordering the same witnesses must still match one-to-one.
  _check_single(tag, cfg, cpu, list(reversed(cpu)), max_contacts=pair_max)


@pytest.mark.parametrize("tag,cfg,count", _SELF_ORACLE)
def test_manifold_contract_negative_controls(tag, cfg, count):
  cpu, pair_max = _cpu_witnesses(tag, cfg)

  def bad(nat, **kw):
    with pytest.raises(AssertionError):
      _check_single(tag, cfg, cpu, nat, max_contacts=kw.get("cap", pair_max))

  base = [(d, n.copy(), p.copy()) for d, n, p in cpu]
  bad(base[:-1])                                   # missing witness
  bad(base + [base[0]])                            # extra/duplicate witness
  bad(base[:-1] + [base[0]])                       # duplicate hides lost one
  bad(base, cap=count - 1)                         # overflow of producer cap
  wrong = list(base)
  wrong[0] = (wrong[0][0] + 1e-3, wrong[0][1], wrong[0][2])
  bad(wrong)                                       # wrong depth
  wrong[0] = (base[0][0], np.array([1.0, 0.0, 0.0]), base[0][2])
  bad(wrong)                                       # wrong normal
  wrong[0] = (base[0][0], base[0][1], base[0][2] + np.array([0.01, 0, 0]))
  bad(wrong)                                       # wrong position
