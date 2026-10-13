# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Imported hull frames and actual mesh contact forces, without CPU fallback."""
import itertools
import os
from pathlib import Path
import struct

import mujoco
import numpy as np
import pytest

from test_analytic_material_lifecycle_010 import _reference, _native, _check_physics
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.mesh_hull import collision_mesh_hull

KINDS = ("plane", "sphere", "capsule", "box", "ellipsoid", "cylinder", "mesh")
SHAPES = ("asymmetric", "face", "thin", "long", "near_coplanar")
SIZES = {"plane": "2 2 .125", "sphere": ".0625", "capsule": ".0625 .09375",
         "box": ".0625 .03125 .09375", "ellipsoid": ".0625 .03125 .09375",
         "cylinder": ".0625 .09375"}


def _asset(shape):
  if shape == "face":
    vertices = np.asarray(list(itertools.product((-.0625, .0625),
                                                (-.03125, .03125),
                                                (-.015625, .015625))))
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads]
  elif shape == "near_coplanar":
    vertices = np.asarray([[-.0625, -.03125, -.015625], [.0625, -.03125, -.015625],
                           [.0625, .03125, -.015625], [-.0625, .03125, -.015625 + 2.**-20],
                           [-.015625, .0078125, .09375]])
    faces = [(0, 2, 1), (0, 3, 2), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
  else:
    vertices = np.asarray([[-.0625, -.03125, -.015625], [.0625, -.03125, -.015625],
                           [-.0625, .0625, -.015625], [-.03125, -.015625, .09375]])
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    if shape == "thin":
      vertices[:, 2] *= 2.**-10
    elif shape == "long":
      vertices[:, 2] *= 8.
    else:
      assert shape == "asymmetric"
  # Orient all authored triangles away from the hull interior.
  center = vertices.mean(axis=0)
  oriented = []
  for a, b, c in faces:
    normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    oriented.append((a, c, b) if np.dot(normal, vertices[a] - center) < 0 else (a, b, c))
  return vertices, oriented


def _write_asset(path, vertices, faces, encoding):
  if encoding == "obj":
    text = "".join("v " + " ".join(format(x, ".17g") for x in v) + "\n" for v in vertices)
    text += "".join("f " + " ".join(str(i + 1) for i in face) + "\n" for face in faces)
    path.write_text(text)
  else:
    assert encoding == "stl"
    payload = bytearray(b"Authored Apache-2.0 mesh qualification".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(faces)))
    for face in faces:
      points = vertices[list(face)]
      normal = np.cross(points[1] - points[0], points[2] - points[0])
      normal /= np.linalg.norm(normal)
      payload.extend(struct.pack("<12fH", *normal, *points.reshape(-1), 0))
    path.write_bytes(payload)


def _model(tmp_path, kind, reverse, cone, shape="asymmetric", encoding="obj"):
  vertices, faces = _asset(shape)
  asset_path = tmp_path / ("authored_hull." + encoding)
  _write_asset(asset_path, vertices, faces, encoding)
  # Nontrivial import, asset reference, geom and free-body frames all compose.
  mesh_asset = (f'<mesh name="hull" file="{asset_path.name}" scale="1.25 .75 1.5" '
                'refpos=".015625 -.0078125 .00390625" '
                'refquat=".991444861 .130526192 0 0"/>')
  mesh_geom = '<geom type="mesh" mesh="hull" pos=".03125 -.015625 .0078125" euler=".12 -.08 .16"/>'
  other_geom = mesh_geom if kind == "mesh" else f'<geom type="{kind}" size="{SIZES[kind]}"/>'
  if kind == "plane":
    geoms = (other_geom.replace('/>', ' name="a"/>') +
             '<body name="body0" pos="0 0 .5" euler=".07 -.05 .09"><freejoint/>' +
             mesh_geom.replace('/>', ' name="b"/>') + '</body>')
  else:
    order = (other_geom, mesh_geom) if reverse else (mesh_geom, other_geom)
    geoms = "".join(f'<body name="body{i}" pos="0 0 .5" euler=".07 -.05 .09">'
                    '<freejoint/><inertial mass=".7" pos=".015625 -.0078125 .00390625" '
                    'quat=".965925826 0 .258819045 0" diaginertia=".03 .04 .05"/>' +
                    geom.replace('/>', f' name="{name}"/>') + '</body>'
                    for i, (name, geom) in enumerate(zip(("a", "b"), order)))
  pair = ("b", "a") if reverse else ("a", "b")
  xml = f'''<mujoco><compiler angle="radian"/><option gravity="0 0 0"
    timestep=".002" cone="{cone}" iterations="1000" tolerance="1e-8"
    ccd_iterations="1000" ccd_tolerance="1e-10"/>
    <default><geom condim="3" mass=".7" friction=".7 .03 .01"/></default>
    <asset>{mesh_asset}</asset><worldbody>{geoms}</worldbody>
    <contact><pair geom1="{pair[0]}" geom2="{pair[1]}" condim="3"
      friction=".7 .7 .03 .01 .01"/></contact></mujoco>'''
  xml_path = tmp_path / "imported_scene.xml"
  xml_path.write_text(xml)
  return mujoco.MjModel.from_xml_path(str(xml_path))


def _extent(model, data, geom, direction):
  center = data.geom_xpos[geom]
  matrix = data.geom_xmat[geom].reshape(3, 3)
  local = matrix.T @ direction
  kind = int(model.geom_type[geom])
  size = model.geom_size[geom]
  if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
    vertices, _ = collision_mesh_hull(model, int(model.geom_dataid[geom]))
    return float(direction @ center + np.max(vertices @ local))
  if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
    reach = size[0]
  elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
    reach = size[0] + size[1] * abs(local[2])
  elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
    reach = size[0] * np.linalg.norm(local[:2]) + size[1] * abs(local[2])
  elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
    reach = np.linalg.norm(size * local)
  else:
    assert kind == int(mujoco.mjtGeom.mjGEOM_BOX)
    reach = np.abs(local) @ size
  return float(direction @ center + reach)


def _state(model, kind, separation=-.00390625, moving=True):
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  q = model.qpos0.copy()
  v = np.zeros(model.nv)
  if kind == "plane":
    q[2] += _extent(model, data, 1, np.array([0., 0., -1.])) + separation
    if moving:
      v[:] = [.03, -.02, -.1, .1, -.07, .03]
  else:
    axis = np.array([1., 0., 0.])
    if kind == "mesh":
      vertices, _ = collision_mesh_hull(model, int(model.geom_dataid[0]))
      _, _, vectors = np.linalg.svd(vertices.astype(float) - vertices.mean(axis=0))
      axis = data.geom_xmat[0].reshape(3, 3) @ vectors[-1]
      axis /= np.linalg.norm(axis)
      if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    # Prepare aligned interior centers before locating a real source
    # crossing. AABB overlap alone is not contact for an asymmetric hull.
    # CPU geometry is used only once to place this independent fixture;
    # it supplies no native contact/force result and never runs per step.
    centers = []
    for g in (0, 1):
      center = data.geom_xpos[g].copy()
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        vertices, _ = collision_mesh_hull(model, int(model.geom_dataid[g]))
        center += data.geom_xmat[g].reshape(3, 3) @ vertices.astype(float).mean(axis=0)
      centers.append(center)
    q[7:10] += centers[0] - centers[1]
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    reach = _extent(model, data, 0, axis) + _extent(model, data, 1, -axis)
    width = sum(_extent(model, data, g, axis) + _extent(model, data, g, -axis) for g in (0, 1))
    outer = reach + .001
    base = q[7:10].copy()
    previous = outer
    for shift in np.linspace(outer, 0., 257)[1:]:
      data.qpos[7:10] = base + shift * axis
      mujoco.mj_forward(model, data)
      if data.ncon and any(c.dist < 0 for c in data.contact):
        inner, outer = shift, previous
        break
      previous = shift
    else:
      raise AssertionError("Source crossing must be loaded")
    for _ in range(35):
      middle = .5 * (outer + inner)
      data.qpos[7:10] = base + middle * axis
      mujoco.mj_forward(model, data)
      if data.ncon and any(c.dist < 0 for c in data.contact):
        inner = middle
      else:
        outer = middle
    # Positive tiny hulls need a depth smaller than their thickness;
    # pushing millimeters through a micrometer slab would miss contact.
    depth = np.clip(separation, -width / 16, np.inf)
    q[7:10] = base + (.5 * (outer + inner) + depth) * axis
    if moving:
      v[6:9] = -.1 * axis + np.cross(axis, [.03, -.02, .01])
      v[9:] = [.1, -.07, .03]
  return q, v


def _check_geometry_and_world_wrenches(model, data, arrays):
  slots = np.flatnonzero(arrays["contact_mask"][0])
  assert len(slots) == data.ncon > 0
  actual_points = arrays["contact_position"][0, slots]
  expected_points = np.asarray([c.pos for c in data.contact])
  permutation = min(itertools.permutations(range(data.ncon)), key=lambda perm:
                    sum(np.linalg.norm(actual_points[i] - expected_points[j])
                        for i, j in enumerate(perm)))
  for slot, index in zip(slots, permutation):
    contact = data.contact[index]
    np.testing.assert_allclose(arrays["contact_position"][0, slot], contact.pos, rtol=0, atol=6e-6)
    np.testing.assert_allclose(arrays["contact_distance"][0, slot], contact.dist, rtol=0, atol=1e-7)
    frame = np.asarray([arrays[name][0, slot] for name in
                        ("contact_normal", "contact_tangent1", "contact_tangent2")])
    np.testing.assert_allclose(frame, contact.frame.reshape(3, 3), rtol=0, atol=6e-5)
  wrenches = np.zeros((model.nbody, 6))
  for index, contact in enumerate(data.contact):
    local = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, local)
    frame = contact.frame.reshape(3, 3)
    force, moment = frame.T @ local[:3], frame.T @ local[3:]
    for sign, geom in ((-1, contact.geom[0]), (1, contact.geom[1])):
      wrenches[model.geom_bodyid[geom]] += sign * np.r_[force, moment + np.cross(contact.pos, force)]
  native = arrays["qfrc_constraint"][0]
  for body in range(1, model.nbody):
    adr = int(model.jnt_dofadr[model.body_jntadr[body]])
    force = native[adr:adr + 3]
    moment = data.xmat[body].reshape(3, 3) @ native[adr + 3:adr + 6]
    actual = np.r_[force, moment + np.cross(data.xpos[body], force)]
    np.testing.assert_allclose(actual, wrenches[body], rtol=.03, atol=.02)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
def test_imported_mesh_frame_hull_and_source_loaded_controls(tmp_path, kind, reverse, shape):
  model = _model(tmp_path, kind, reverse, "pyramidal", shape)
  desc = lower_coupled_constraints(model)
  assert model.nmesh == 1
  assert np.linalg.norm(model.mesh_pos[0]) > .001
  assert np.linalg.norm(model.mesh_quat[0, 1:]) > .001
  for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
    vertices, _ = collision_mesh_hull(model, int(model.geom_dataid[geom]))
    offset, count = desc.mesh_hull_info[9 * geom:9 * geom + 2]
    np.testing.assert_array_equal(desc.mesh_hull[3 * offset:3 * (offset + count)].reshape(-1, 3), vertices)
  q, v = _state(model, kind)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = q, v
  mujoco.mj_forward(model, data)
  assert data.ncon > 0 and np.linalg.norm(data.qfrc_constraint) > .001


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual imported mesh physics")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("shape", SHAPES)
def test_imported_mesh_frames_full_mass_wrenches_and_retained_certificate(
    tmp_path, kind, reverse, cone, shape, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = _model(tmp_path, kind, reverse, cone, shape)
  q, v = _state(model, kind)
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  assert cpu.ncon > 0 and np.linalg.norm(cpu.qfrc_constraint) > .001
  assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
  arrays, _ = _check_physics(model, sim, cpu, assembly)
  _check_geometry_and_world_wrenches(model, cpu, arrays)


@pytest.mark.parametrize("encoding", ["obj", "stl"])
def test_real_imported_asset_formats_preserve_compiled_hull(tmp_path, encoding):
  model = _model(tmp_path, "box", False, "pyramidal", encoding=encoding)
  vertices, faces = collision_mesh_hull(model, 0)
  assert len(vertices) == 4 and len(faces) == 4
  assert (tmp_path / ("authored_hull." + encoding)).is_file()
  q, v = _state(model, "box")
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = q, v
  mujoco.mj_forward(model, data)
  assert data.ncon > 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual imported asset trajectory")
@pytest.mark.parametrize("encoding", ["obj", "stl"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_imported_asset_friction_trajectory_and_full_held_force_replay(
    tmp_path, encoding, kind, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model = _model(tmp_path, kind, False, cone, encoding=encoding)
  q, v = _state(model, kind)
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
  arrays, _ = _check_physics(model, sim, cpu, assembly)
  _check_geometry_and_world_wrenches(model, cpu, arrays)
  assert np.linalg.norm(arrays["contact_wrench"][0, :, 1:]) > .0001
  for _ in range(8):
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
  assert sim._last_coupled["contact_mask"].any().item(), "Replay must start during loaded motion"
  before = sim.snapshot()
  commands = [.0001 * np.cos(np.arange(model.nv) + i).astype(np.float32)[None]
              for i in range(20)]
  for command in commands:
    np.testing.assert_array_equal(_native(monkeypatch,
        lambda: sim.step(qfrc_applied=command)).cpu().numpy(), [0])
  expected = sim.snapshot()
  sim.restore(before)
  for command in commands:
    np.testing.assert_array_equal(_native(monkeypatch,
        lambda: sim.step(qfrc_applied=command)).cpu().numpy(), [0])
  _same_checkpoint(sim.snapshot(), expected)
