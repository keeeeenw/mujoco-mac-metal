# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Terrain callback detection margins must remain separate from solver inclusion."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from test_analytic_material_lifecycle_010 import _check_physics, _native, _reference
from test_terrain_geometry_physics_012 import KINDS, LIMITS, _model, _state


def _fixture(kind, reverse, cone, policy):
  model = _model(kind, "ridge", reverse, cone)
  model.pair_margin[0], model.pair_gap[0] = ((0., .003) if policy == "gap_only"
                                           else (.002, .001))
  return model, *_state(model)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("policy", ["gap_only", "margin_gap"])
def test_source_terrain_margin_gap_has_actual_loaded_contacts(kind, reverse, policy):
  model, q, v = _fixture(kind, reverse, "elliptic", policy)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = q, v
  mujoco.mj_forward(model, data)
  assert data.ncon > 0 and data.nefc > 0
  assert np.linalg.norm(data.qfrc_constraint) > .1
  desc = lower_coupled_constraints(model, limits=LIMITS)
  words = desc.pair_margin_source_words.reshape(3, -1).astype(float).sum(0)
  np.testing.assert_array_equal(words, model.pair_margin + model.pair_gap)
  assert any(contact.dist < 0 for contact in data.contact)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native terrain margin/gap")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("policy", ["gap_only", "margin_gap"])
def test_terrain_margin_gap_candidates_geometry_and_solved_force(
    kind, reverse, cone, policy, monkeypatch):
  from mujoco_metal import MetalSimulation
  model, q, v = _fixture(kind, reverse, cone, policy)
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None],
                        profile="integrated_euler_v1", limits=LIMITS)
  cpu = _reference(model, sim)
  assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
  workspace = sim._coupled_constraints._workspace
  count = int(workspace["contact_pair_count"][0].cpu())
  assert count == cpu.ncon > 0
  arrays, _ = _check_physics(model, sim, cpu, assembly)
  records = workspace["contact_pair_records"].detach().cpu().double().numpy().reshape(1, 50, 37)[0, :count]
  points = records[:, 13:16] + records[:, 25:28] + records[:, 31:34]
  normals = records[:, 16:19] + records[:, 28:31] + records[:, 34:37]
  distances = records[:, 12] + records[:, 9] + records[:, 10]
  np.testing.assert_allclose(points, [c.pos for c in cpu.contact], rtol=0, atol=6e-6)
  first = int(sim._coupled_constraints.descriptor.geom1[0])
  expected_normals = [c.frame[:3] * (1 if int(c.geom[0]) == first else -1)
                      for c in cpu.contact]
  np.testing.assert_allclose(normals, expected_normals, rtol=0, atol=6e-5)
  np.testing.assert_allclose(distances, [c.dist for c in cpu.contact], rtol=0, atol=1e-7)
  assert int(arrays["contact_mask"][0].sum()) == sum(c.efc_address >= 0 for c in cpu.contact)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native terrain queries")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("libccd", [False, True])
def test_public_terrain_distance_queries_preserve_physics_and_full_restore(kind, libccd, monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_geomDistance
  model, q, v = _fixture(kind, False, "elliptic", "margin_gap")
  if libccd:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None],
                        profile="integrated_euler_v1", limits=LIMITS)
  cpu = _reference(model, sim)
  snapshot = sim.snapshot()
  precision = sim.state._precision.words.clone()
  assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
  prior = {name: value.clone() for name, value in assembly.items() if isinstance(value, torch.Tensor)}
  saved = []
  terrain = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"))
  obj = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object"))
  for bound in (.03125, .00390625, -.00390625):
    for a, b in ((terrain, obj), (obj, terrain)):
      expected = np.zeros(6)
      distance = mujoco.mj_geomDistance(model, cpu, a, b, bound, expected)
      actual = _native(monkeypatch, lambda: mj_geomDistance(sim, a, b, bound))
      np.testing.assert_array_equal(actual["status"].cpu().numpy(), [0])
      np.testing.assert_allclose(actual["distance"].cpu().numpy(), [distance], rtol=3e-5, atol=3e-5)
      np.testing.assert_allclose(actual["fromto"][0].cpu().numpy(), expected, rtol=3e-5, atol=3e-5)
      program = sim._geometry_distance_query[1]
      backings, seen = {}, set()
      def visit(value):
        if id(value) in seen:
          return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
          storage = value.untyped_storage()
          backings[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, dict):
          for child in value.values():
            visit(child)
        elif isinstance(value, (tuple, list)):
          for child in value:
            visit(child)
        elif type(value).__module__.startswith("mujoco_metal.") and hasattr(value, "__dict__"):
          for child in vars(value).values():
            visit(child)
      visit(program)
      visit(actual)
      assert sum(backings.values()) == 4 * sum(program.workspace_sizes.values())
      saved.append((actual, actual["fromto"].clone()))
  for actual, expected in saved:
    torch.testing.assert_close(actual["fromto"], expected, rtol=0, atol=0)
  for name, expected in prior.items():
    torch.testing.assert_close(assembly[name], expected, rtol=0, atol=0)
  torch.testing.assert_close(sim.state._precision.words, precision, rtol=0, atol=0)
  sim.restore(snapshot)
  repeated = _native(monkeypatch, lambda: mj_geomDistance(sim, terrain, obj, .03125))
  torch.testing.assert_close(repeated["fromto"], saved[0][0]["fromto"], rtol=0, atol=0)
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
  sim.restore(snapshot)
  torch.testing.assert_close(_native(monkeypatch, lambda: mj_geomDistance(sim, terrain, obj, .03125))["fromto"],
                             repeated["fromto"], rtol=0, atol=0)
