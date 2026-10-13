# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Admitted thin features, near-parallel frames and near-coincident controls."""
import os
import mujoco
import numpy as np
import pytest

from test_analytic_boundaries_010 import PAIRS
from test_analytic_material_lifecycle_010 import _reference, _native, _check_physics


CASES = ("thin", "near_parallel", "near_coincident", "small", "large")


def _rotation(quat):
  matrix = np.empty(9)
  mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
  return matrix.reshape(3, 3)


def _reach(kind, size, matrix, direction):
  local = matrix.T @ direction
  if kind == "sphere":
    return size[0]
  if kind == "capsule":
    return size[0] + size[1] * abs(local[2])
  if kind == "cylinder":
    return size[0] * np.linalg.norm(local[:2]) + size[1] * abs(local[2])
  if kind == "ellipsoid":
    return np.linalg.norm(np.asarray(size) * local)
  assert kind == "box"
  return np.dot(np.asarray(size), np.abs(local))


def _fixture(first, second, reverse, cone, case):
  scale = {"small": .125, "large": 4.}.get(case, 1.)
  half = 2.**-20 if case == "thin" else .125
  sizes = {"plane": [2., 2., .125], "sphere": [.0625],
           "capsule": [.0625, .125], "box": [.0625, .03125, .125],
           "cylinder": [.0625, half], "ellipsoid": [.0625, .03125, half]}
  sizes = {name: np.asarray(values) * scale for name, values in sizes.items()}
  order = (second, first) if reverse and first != "plane" else (first, second)
  text = lambda kind: " ".join(format(value, ".17g") for value in sizes[kind])
  if first == "plane":
    geoms = (f'<geom name="a" type="plane" size="{text(first)}"/>'
             f'<body pos="0 0 .5"><freejoint/><geom name="b" '
             f'type="{second}" size="{text(second)}"/></body>')
  else:
    geoms = "".join(f'<body pos="0 0 .5"><freejoint/><geom name="{name}" '
                    f'type="{kind}" size="{text(kind)}"/></body>'
                    for name, kind in zip(("a", "b"), order))
  pair = ("b", "a") if reverse else ("a", "b")
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".002" gravity="0 0 0" cone="{cone}"
      iterations="1000" tolerance="1e-8" ccd_iterations="1000" ccd_tolerance="1e-10"/>
    <default><geom mass=".7" condim="3" friction=".7 .03 .01"/></default>
    <worldbody>{geoms}</worldbody>
    <contact><pair geom1="{pair[0]}" geom2="{pair[1]}" condim="3"
      friction=".7 .7 .03 .01 .01"/></contact></mujoco>''')
  quat = np.asarray([1., 0., 0., 0.])
  if case == "near_parallel":
    angle = 2.**-18
    axis = np.asarray([1., .3, .7]); axis /= np.linalg.norm(axis)
    quat = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
    if first == "plane":
      # Cylinder axes / ellipsoid features almost parallel to the plane.
      quarter = np.asarray([np.sqrt(.5), np.sqrt(.5), 0., 0.])
      product = np.empty(4)
      mujoco.mju_mulQuat(product, quat, quarter)
      quat = product
  q = model.qpos0.astype(np.float32).copy()
  v = np.zeros(model.nv, np.float32)
  if first == "plane":
    q[3:7] = quat
    q[2] = _reach(second, sizes[second], _rotation(quat), np.array([0., 0., 1.])) - 2.**-10 * scale
    v[:3] = [0., .01 * scale, -.02 * scale]
  else:
    q[:3] = [0, 0, .5 * scale]
    q[7:10] = [0, 0, .5 * scale]
    q[10:14] = quat
    if case == "near_coincident":
      q[7] = 2.**-18 * scale
      q[8] = 2.**-19 * scale
    else:
      direction = np.array([1., 0., 0.])
      q[7] = (_reach(order[0], sizes[order[0]], np.eye(3), direction) +
              _reach(order[1], sizes[order[1]], _rotation(quat), direction) - 2.**-10 * scale)
    v[6:9] = [-.02 * scale, .01 * scale, 0.]
  return model, q, v


@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_near_degenerate_reference_is_admitted_and_loaded(first, second, reverse, case):
  model, q, v = _fixture(first, second, reverse, "pyramidal", case)
  cpu = mujoco.MjData(model)
  cpu.qpos[:], cpu.qvel[:] = q, v
  mujoco.mj_forward(model, cpu)
  assert model.nv > 0 and np.all(model.body_mass[1:] > 0)
  assert np.all(np.isfinite(cpu.qacc)) and cpu.ncon > 0
  assert np.linalg.norm(cpu.qfrc_constraint) > .001
  if case == "thin":
    ids = [g for g in range(model.ngeom) if model.geom_type[g] in (4, 5)]
    assert ids and all(model.geom_size[g, 1 if model.geom_type[g] == 5 else 2] /
                       model.geom_size[g, 0] < 2.**-15 for g in ids)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual degenerate analytic physics")
@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("case", CASES)
def test_near_degenerate_native_geometry_full_mass_force_and_retained_certificate(
    first, second, reverse, cone, case, monkeypatch):
  from mujoco_metal import MetalSimulation
  model, q, v = _fixture(first, second, reverse, cone, case)
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  assert cpu.ncon > 0 and np.linalg.norm(cpu.qfrc_constraint) > .001
  assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
  arrays, _ = _check_physics(model, sim, cpu, assembly)
  records = sim._coupled_constraints._workspace["contact_pair_records"].cpu().numpy().reshape(-1, 37)
  count = int(sim._coupled_constraints._workspace["contact_pair_count"][0].cpu())
  assert count == cpu.ncon and arrays["contact_mask"].any()
  records = records[:count].astype(float)
  depth = records[:, 12] + records[:, 9] + records[:, 10]
  expected = np.asarray([contact.dist for contact in cpu.contact])
  errors = np.abs(depth[:, None] - expected[None])
  assert np.max(errors.min(axis=1)) < 1e-7
  assert np.max(errors.min(axis=0)) < 1e-7
  position = arrays["contact_position"][0, np.flatnonzero(arrays["contact_mask"][0])]
  normal = arrays["contact_normal"][0, np.flatnonzero(arrays["contact_mask"][0])]
  points = np.asarray([contact.pos for contact in cpu.contact])
  normals = np.asarray([contact.frame[:3] for contact in cpu.contact])
  matched = ((np.linalg.norm(position[:, None] - points[None], axis=2) < 6e-6) &
             (np.linalg.norm(normal[:, None] - normals[None], axis=2) < 6e-5))
  assert matched.any(axis=0).all() and matched.any(axis=1).all()
