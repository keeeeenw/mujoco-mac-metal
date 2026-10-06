# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned collision-hull semantics for meshes with concave source surfaces."""

import os
from pathlib import Path
import importlib.util

import mujoco
import numpy as np
import pytest

from mujoco_metal.mesh_hull import collision_mesh_hull
from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _model():
  source = Path(__file__).with_name("test_mesh_contact_011.py")
  spec = importlib.util.spec_from_file_location("mesh_011_fixture", source)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module._notch_model()


def test_compiler_collision_hull_covers_concave_surface_and_real_lowering():
  model = _model()
  vertices, faces = collision_mesh_hull(model, 0)
  assert model.mesh_vertnum[0] == 16 and model.mesh_facenum[0] == 24
  assert vertices.shape == (8, 3) and faces.shape == (12, 3)
  # Independent compiler record, rather than a second implementation of the
  # lowering algorithm: exact support extrema and outward half-spaces.
  raw = np.asarray(model.mesh_vert, dtype=np.float64)
  for direction in np.random.default_rng(74).normal(size=(100, 3)):
    assert np.max(vertices @ direction) == pytest.approx(
        np.max(raw @ direction), abs=1e-12)
  center = vertices.astype(np.float64).mean(axis=0)
  for face in faces:
    tri = vertices[face].astype(np.float64)
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if np.dot(normal, tri.mean(axis=0) - center) < 0:
      normal *= -1
    assert np.max((vertices - tri[0]) @ normal) <= 1e-6
  descriptor = lower_coupled_constraints(model)
  geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ma")
  info = descriptor.mesh_hull_info.reshape(-1, 9)[geom]
  assert int(info[1]) == 8 and int(info[4]) == 12
  packed = descriptor.mesh_hull[int(info[0])*3:(int(info[0])+8)*3].reshape(8, 3)
  np.testing.assert_array_equal(packed, vertices)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native collision-hull lifecycle requires GPU opt-in")
def test_concave_source_mesh_native_hull_contact_force_and_lifecycle():
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _model()
  model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
  model.opt.iterations = 100
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None], 2, axis=0)
  qpos[:, :3] = 0
  qpos[:, 7:10] = [[1., .5, 1.09], [1., .5, 1.11]]
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:] = qpos[world]
    mujoco.mj_forward(model, data)
  assert refs[0].ncon == 1 and refs[1].ncon == 0
  # The sphere is above the concave notch's original recessed face: the
  # contact exists specifically because the CPU uses the collision hull.
  assert float(refs[0].contact[0].dist) == pytest.approx(-.01, abs=1e-6)
  sim = MetalSimulation(model, batch_size=2, profile="integrated_scalable_v1")
  sim.reset(qpos=qpos, qvel=np.zeros((2, model.nv), dtype=np.float32))
  system = sim.assembled_system()
  host = lambda value: value.detach().cpu().numpy().copy()
  mask = host(system["contact_mask"])
  assert int((mask[0] > .5).sum()) == 1 and not np.any(mask[1] > .5)
  slot = np.flatnonzero(mask[0] > .5)[0]
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1
  np.testing.assert_allclose(host(system["qfrc_constraint"]),
                             [d.qfrc_constraint for d in refs],
                             rtol=8e-3, atol=8e-3)
  np.testing.assert_allclose(host(system["contact_distance"])[0, slot],
                             refs[0].contact[0].dist, atol=2e-5, rtol=0)
  np.testing.assert_allclose(host(system["contact_position"])[0, slot],
                             refs[0].contact[0].pos, atol=2e-4, rtol=0)
  np.testing.assert_allclose(host(system["contact_normal"])[0, slot],
                             refs[0].contact[0].frame[:3], atol=2e-4, rtol=0)
  for _ in range(4):
    sim.step()
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_array_equal(host(sim.state.status), np.zeros(2, dtype=np.int32))
    np.testing.assert_allclose(host(sim.state._qacc),
                               [d.qacc for d in refs], rtol=8e-3, atol=8e-3)
    np.testing.assert_allclose(host(sim.state._qpos),
                               [d.qpos for d in refs], rtol=8e-3, atol=8e-3)
    np.testing.assert_allclose(host(sim.state._qvel),
                               [d.qvel for d in refs], rtol=8e-3, atol=8e-3)
  checkpoint = sim.snapshot()
  sim.step()
  expected = [host(sim.state._qpos), host(sim.state._qvel), host(sim.state._qacc)]
  sim.restore(checkpoint)
  sim.step()
  for actual, wanted in zip((sim.state._qpos, sim.state._qvel, sim.state._qacc), expected):
    np.testing.assert_array_equal(host(actual), wanted)
  sim.reset(qpos=qpos, qvel=np.zeros((2, model.nv), dtype=np.float32))
  sim.step()
  np.testing.assert_array_equal(host(sim.state.status), np.zeros(2, dtype=np.int32))


def _simplified_model(bound):
  vertices = np.random.default_rng(63).normal(size=(100, 3))
  vertices *= .25 / np.linalg.norm(vertices, axis=1)[:, None]
  packed = " ".join(map(str, vertices.reshape(-1)))
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="100" timestep=".002"/>
    <asset><mesh name="h" vertex="{packed}" maxhullvert="{bound}"/></asset>
    <worldbody><geom type="mesh" mesh="h"/>
      <body pos=".30 0 0"><freejoint/><geom type="sphere" size=".09"/></body>
    </worldbody></mujoco>''')


def test_compiled_simplification_controls_support_and_actual_capacity():
  model = _simplified_model(16)
  full = _simplified_model(-1)
  vertices, faces = collision_mesh_hull(model, 0)
  assert int(model.mesh_vertnum[0]) == 100
  assert len(vertices) == 16 and len(faces) == 28
  descriptor = lower_coupled_constraints(model)
  assert int(descriptor.mesh_hull_info.reshape(-1, 9)[0, 1]) == 16
  a, b = mujoco.MjData(model), mujoco.MjData(full)
  mujoco.mj_forward(model, a)
  mujoco.mj_forward(full, b)
  assert a.ncon == b.ncon == 1
  # The same original surface has a different simplified collision witness.
  # Comparing only a raw-vertex support set would incorrectly pass admission
  # while simulating the unsimplified hull.
  assert abs(float(a.contact[0].dist) - float(b.contact[0].dist)) > 1e-4
  assert np.linalg.norm(a.contact[0].frame[:3] - b.contact[0].frame[:3]) > .05
  with pytest.raises(ValueError, match="64"):
    lower_coupled_constraints(full)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native simplified-hull trajectory requires GPU opt-in")
@pytest.mark.parametrize("bound", [16, 32])
def test_native_compiled_simplification_contact_force_trajectory(bound):
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _simplified_model(bound)
  qpos = np.repeat(model.qpos0.astype(np.float32)[None], 2, axis=0)
  qpos[1, 0] = .6
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:] = qpos[world]
    mujoco.mj_forward(model, data)
  assert refs[0].ncon == 1 and refs[1].ncon == 0
  assert np.linalg.norm(refs[0].qfrc_constraint) > 100
  sim = MetalSimulation(model, batch_size=2, profile="integrated_scalable_v1")
  sim.reset(qpos=qpos, qvel=np.zeros((2, model.nv), np.float32))
  host = lambda value: value.detach().cpu().numpy().copy()
  system = sim.assembled_system()
  mask = host(system["contact_mask"])
  assert int((mask[0] > .5).sum()) == 1 and not np.any(mask[1] > .5)
  slot = np.flatnonzero(mask[0] > .5)[0]
  for field, expected, tolerance in (
      ("contact_distance", refs[0].contact[0].dist, 2e-5),
      ("contact_position", refs[0].contact[0].pos, 2e-4),
      ("contact_normal", refs[0].contact[0].frame[:3], 2e-4)):
    np.testing.assert_allclose(host(system[field])[0, slot], expected,
                               rtol=0, atol=tolerance)
  np.testing.assert_allclose(host(system["qfrc_constraint"]),
                             [data.qfrc_constraint for data in refs],
                             rtol=8e-3, atol=8e-3)
  for _ in range(8):
    sim.step()
    for data in refs:
      mujoco.mj_step(model, data)
    np.testing.assert_array_equal(host(sim.state.status), np.zeros(2, np.int32))
    for field in ("qpos", "qvel", "qacc"):
      np.testing.assert_allclose(host(getattr(sim.state, "_" + field)),
                                 [getattr(data, field) for data in refs],
                                 rtol=8e-3, atol=8e-3)
  snapshot = sim.snapshot()
  sim.step()
  saved = [host(getattr(sim.state, "_" + name)) for name in ("qpos", "qvel", "qacc")]
  sim.restore(snapshot)
  sim.step()
  for name, expected in zip(("qpos", "qvel", "qacc"), saved):
    np.testing.assert_array_equal(host(getattr(sim.state, "_" + name)), expected)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="sphere-hull feature matrix requires GPU opt-in")
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_native_sphere_hull_interior_exterior_features_and_friction(condim):
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _simplified_model(16)
  model.geom_condim[:] = condim
  model.geom_friction[:] = [.7, .04, .02]
  centers = np.asarray([
      [.30, 0, 0], [.15, .02, .03], [0, .01, .02],
      [.20, .20, .02], [-.30, .03, .02], [1, 0, 0]], np.float32)
  batch = len(centers)
  qpos = np.repeat(model.qpos0.astype(np.float32)[None], batch, axis=0)
  qpos[:, :3] = centers
  qvel = np.repeat(np.asarray([[.02, -.03, .01, .2, -.1, .3]], np.float32),
                   batch, axis=0)
  refs = [mujoco.MjData(model) for _ in range(batch)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
  assert all(data.ncon == 1 for data in refs[:-1])
  assert refs[-1].ncon == 0
  sim = MetalSimulation(model, batch_size=batch, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1")
  system = sim.assembled_system()
  host = lambda value: value.detach().cpu().numpy().copy()
  masks = host(system["contact_mask"])
  for env, data in enumerate(refs):
    slots = np.flatnonzero(masks[env] > .5)
    assert len(slots) == data.ncon
    if not data.ncon:
      continue
    slot = slots[0]
    for field, expected, tolerance in (
        ("contact_distance", data.contact[0].dist, 2e-5),
        ("contact_position", data.contact[0].pos, 2e-4),
        ("contact_normal", data.contact[0].frame[:3], 2e-4)):
      np.testing.assert_allclose(host(system[field])[env, slot], expected,
                                 rtol=0, atol=tolerance,
                                 err_msg=f"condim={condim}, env={env}, {field}")
  np.testing.assert_allclose(host(system["qfrc_constraint"]),
                             [data.qfrc_constraint for data in refs],
                             rtol=8e-3, atol=8e-3)
  sim.step()
  for data in refs:
    mujoco.mj_step(model, data)
  np.testing.assert_array_equal(host(sim.state.status), np.zeros(batch, np.int32))
  for field in ("qpos", "qvel", "qacc"):
    np.testing.assert_allclose(host(getattr(sim.state, "_" + field)),
                               [getattr(data, field) for data in refs],
                               rtol=8e-3, atol=8e-3)
