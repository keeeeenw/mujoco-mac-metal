# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Binary mesh touching and actual feature/candidate history transitions."""
import itertools
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from test_mesh_asset_physics_011 import KINDS, _check_geometry_and_world_wrenches
from test_analytic_material_lifecycle_010 import _reference, _native, _check_physics

CUBE = " ".join(" ".join(str(x) for x in v) for v in
                itertools.product((-.0625, .0625), (-.03125, .03125), (-.09375, .09375)))
SIZES = {"plane": "2 2 .125", "sphere": ".0625", "capsule": ".0625 .09375",
         "box": ".0625 .03125 .09375", "ellipsoid": ".0625 .03125 .09375",
         "cylinder": ".0625 .09375"}
CASES = ("separated", "touch", "shallow", "deep", "grazing", "rotated")


def _fixture(kind, reverse, cone):
  mesh = '<geom type="mesh" mesh="cube"/>'
  other = mesh if kind == "mesh" else f'<geom type="{kind}" size="{SIZES[kind]}"/>'
  if kind == "plane":
    world = other.replace('/>', ' name="a"/>') + '<body><freejoint/>' + mesh.replace('/>', ' name="b"/>') + '</body>'
  else:
    order = (other, mesh) if reverse else (mesh, other)
    world = "".join('<body><freejoint/><inertial mass=".7" pos=".015625 -.0078125 .00390625" '
                    'quat=".965925826 0 .258819045 0" diaginertia=".03 .04 .05"/>' +
                    geom.replace('/>', f' name="{name}"/>') + '</body>'
                    for name, geom in zip(("a", "b"), order))
  order = ("b", "a") if reverse else ("a", "b")
  return mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
    <option timestep=".002" gravity="0 0 0" cone="{cone}" iterations="1000"
      tolerance="1e-8" ccd_iterations="1000" ccd_tolerance="1e-10"/>
    <default><geom mass=".7" condim="3" friction=".7 .03 .01"/></default>
    <asset><mesh name="cube" vertex="{CUBE}"/></asset><worldbody>{world}</worldbody>
    <contact><pair geom1="{order[0]}" geom2="{order[1]}" condim="3"
      friction=".7 .7 .03 .01 .01"/></contact></mujoco>''')


def _state(model, kind, case):
  q = model.qpos0.copy()
  v = np.zeros(model.nv)
  delta = {"separated": .03125, "touch": 0., "shallow": -2.**-10,
           "deep": -.03125, "grazing": -.0625, "rotated": -.0625}[case]
  if kind == "plane":
    q[2] = .09375 + delta
    if case in ("rotated", "grazing"):
      axis = np.array([1., .5, -.25]); axis /= np.linalg.norm(axis)
      q[3:7] = np.r_[np.cos(.1), axis * np.sin(.1)]
    v[:3] = [.03, -.02, -.05]
  else:
    q[:3], q[7:10] = [0, 0, .5], [.125 + delta, 0, .5]
    if case == "grazing":
      # Off-center side loading reaches the cube's edge/vertex features.
      other = next(g for g in range(2) if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH) if kind != "mesh" else 1
      half_y = .0625 if kind in ("sphere", "capsule", "cylinder") else .03125
      assert 0 <= other < 2
      q[8] = .75 * (.03125 + half_y)
    if case == "rotated":
      axis = np.array([1., .5, -.25]); axis /= np.linalg.norm(axis)
      q[10:14] = np.r_[np.cos(.1), axis * np.sin(.1)]
    v[6:9] = [-.05, .03, -.02]
  return q, v


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_binary_mesh_source_touch_and_loaded_feature_controls(kind, reverse):
  model = _fixture(kind, reverse, "pyramidal")
  for case in CASES:
    q, v = _state(model, kind, case)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mujoco.mj_forward(model, data)
    if case == "separated":
      assert data.ncon == 0
    elif case == "touch":
      assert all(abs(c.dist) < 1e-8 for c in data.contact)
    else:
      assert data.ncon and np.linalg.norm(data.qfrc_constraint) > .001, (kind, reverse, case)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual mesh features and history")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_binary_mesh_touch_features_mutation_history_and_full_restore(kind, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model = _fixture(kind, reverse, cone)
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  for case in CASES:
    q, v = _state(model, kind, case)
    sim.reset(qpos=q[None], qvel=v[None])
    cpu = _reference(model, sim)
    arrays, _ = _check_physics(model, sim, cpu, _native(monkeypatch,
        lambda: sim.assembled_system(recompute=True)))
    workspace = sim._coupled_constraints._workspace
    count = int(workspace["contact_pair_count"][0].cpu())
    assert count == cpu.ncon, (kind, reverse, cone, case)
    assert bool(arrays["contact_mask"].any()) == any(c.efc_address >= 0 for c in cpu.contact)
    if count:
      records = workspace["contact_pair_records"].cpu().numpy().reshape(-1, 37)[:count].astype(float)
      positions = records[:, 13:16] + records[:, 25:28] + records[:, 31:34]
      normals = records[:, 16:19] + records[:, 28:31] + records[:, 34:37]
      distances = records[:, 12] + records[:, 9] + records[:, 10]
      expected = np.asarray([c.pos for c in cpu.contact])
      perm = min(itertools.permutations(range(count)), key=lambda indices:
                 sum(np.linalg.norm(positions[i] - expected[j]) for i, j in enumerate(indices)))
      for i, j in enumerate(perm):
        np.testing.assert_allclose(positions[i], cpu.contact[j].pos, rtol=0, atol=6e-6)
        np.testing.assert_allclose(normals[i], cpu.contact[j].frame[:3], rtol=0, atol=6e-5)
        assert abs(distances[i] - cpu.contact[j].dist) < 1e-7
    if case not in ("touch", "separated"):
      assert np.linalg.norm(arrays["qfrc_constraint"]) > .001
      _check_geometry_and_world_wrenches(model, cpu, arrays)
  # Keep loaded history, then mutate actual MPS pose/velocity tensors. The
  # candidate and warm-slot state must become empty without a reset shortcut.
  q, v = _state(model, kind, "shallow")
  sim.reset(qpos=q[None], qvel=v[None])
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
  before = sim.snapshot()
  assert sim._last_coupled["contact_mask"].any().item()
  far, velocity = _state(model, kind, "separated")
  sim.state._qpos[0].copy_(sim.state._torch.as_tensor(
      far, dtype=sim.state._torch.float32, device=sim.state._device))
  sim.state._qvel[0].copy_(sim.state._torch.as_tensor(
      velocity, dtype=sim.state._torch.float32, device=sim.state._device))
  cpu = _reference(model, sim)
  assert cpu.ncon == 0
  arrays, _ = _check_physics(model, sim, cpu, _native(monkeypatch,
      lambda: sim.assembled_system(recompute=True)))
  assert not arrays["contact_mask"].any()
  np.testing.assert_array_equal(arrays["lambda"], 0)
  sim.restore(before)
  _same_checkpoint(sim.snapshot(), before)
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
