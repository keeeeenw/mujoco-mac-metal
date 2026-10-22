# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual transformed terrain witnesses, full mass and physical wrenches."""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import CapacityLimits
from mujoco_metal.coupled_constraints import lower_coupled_constraints
from test_analytic_material_lifecycle_010 import _check_physics, _native, _reference


KINDS = ("sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh")
TERRAINS = ("slope", "ridge", "pit")
LIMITS = CapacityLimits(max_slots=50, max_rows=600)


def _model(kind, terrain, reverse, cone):
  rows, cols = 7, 11
  y, x = np.meshgrid(np.linspace(-1, 1, rows),
                     np.linspace(-1, 1, cols), indexing="ij")
  if terrain == "slope":
    height = .4 + .2 * x + .1 * y
  elif terrain == "ridge":
    height = .1 + .7 * np.exp(-18 * (x - .15) ** 2)
  else:
    assert terrain == "pit"
    height = .8 - .6 * np.exp(-9 * (x * x + y * y))
  elevation = " ".join(format(v, ".17g") for v in height.reshape(-1))
  sizes = {"sphere": ".035", "capsule": ".025 .035",
           "ellipsoid": ".035 .027 .04", "cylinder": ".03 .035",
           "box": ".035 .025 .02"}
  geom = ('type="mesh" mesh="hull"' if kind == "mesh" else
          f'type="{kind}" size="{sizes[kind]}"')
  pair = ("object", "terrain") if reverse else ("terrain", "object")
  xml = f'''<mujoco><compiler angle="radian"/>
    <option timestep=".001" cone="{cone}" iterations="1000"
      tolerance="1e-8" ccd_iterations="1000" ccd_tolerance="1e-10"
      gravity="0 0 -9.81"/>
    <asset><hfield name="grid" nrow="{rows}" ncol="{cols}"
      size=".4375 .3125 .0625 .03125" elevation="{elevation}"/>
      <mesh name="hull" vertex="-.035 -.025 -.02 .04 -.025 -.02
        -.03 .035 -.02 -.01 -.005 .055"/></asset>
    <worldbody><geom name="terrain" type="hfield" hfield="grid"
      pos=".03125 -.015625 .0078125" euler=".08 -.06 .13"/>
      <body pos="0 0 .3"><freejoint/>
        <inertial mass=".7" pos=".015625 -.0078125 .00390625"
          quat=".965925826 0 .258819045 0" diaginertia=".03 .04 .05"/>
        <geom name="object" {geom} pos=".0078125 -.00390625 .001953125"
          euler=".12 -.08 .16"/></body></worldbody>
    <contact><pair geom1="{pair[0]}" geom2="{pair[1]}" condim="3"
      friction=".7 .7 .03 .01 .01"/></contact></mujoco>'''
  return mujoco.MjModel.from_xml_string(xml)


def _state(model):
  # Source collision prepares the fixture once. No source result is supplied
  # to native contact generation or used to move a body during native advance.
  data = mujoco.MjData(model)
  q = model.qpos0.copy()
  q[:2] = [.043, .009]
  upper = .3
  data.qpos[:] = q
  data.qpos[2] = upper
  mujoco.mj_forward(model, data)
  assert data.ncon == 0
  # Terrain prisms have finite base thickness. A point far below their
  # bottom is separated, so find the first surface crossing from above.
  for lower in np.linspace(upper, -.1, 161)[1:]:
    data.qpos[2] = lower
    mujoco.mj_forward(model, data)
    if data.ncon:
      break
    upper = lower
  else:
    raise AssertionError("The source must detect the actual top-surface crossing")
  for _ in range(35):
    middle = (upper + lower) * .5
    data.qpos[2] = middle
    mujoco.mj_forward(model, data)
    if data.ncon:
      lower = middle
    else:
      upper = middle
  q[2] = (upper + lower) * .5 - .001953125
  return q.astype(np.float32), np.array([.03, -.02, -.04, .07, -.05, .03], np.float32)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("terrain", TERRAINS)
@pytest.mark.parametrize("reverse", [False, True])
def test_compiled_nonsquare_transformed_terrain_loaded_controls(kind, terrain, reverse):
  model = _model(kind, terrain, reverse, "elliptic")
  desc = lower_coupled_constraints(model, limits=LIMITS)
  assert desc.npairs == 1 and tuple(desc.pair_max_contacts) == (50,)
  assert int(model.hfield_nrow[0]) == 7 and int(model.hfield_ncol[0]) == 11
  np.testing.assert_array_equal(model.hfield_size[0], [.4375, .3125, .0625, .03125])
  assert np.ptp(model.hfield_data) > .9
  q, v = _state(model)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = q, v
  mujoco.mj_forward(model, data)
  assert data.ncon > 0 and data.nefc > 0
  assert np.linalg.norm(data.qfrc_constraint) > .1
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  assert np.max(np.abs(mass - np.diag(mass.diagonal()))) > .001


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native terrain qualification")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("terrain", TERRAINS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_transformed_terrain_candidates_full_mass_and_world_wrenches(
    kind, terrain, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = _model(kind, terrain, reverse, cone)
  q, v = _state(model)
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None],
                        profile="integrated_euler_v1", limits=LIMITS)
  cpu = _reference(model, sim)
  assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
  arrays, _ = _check_physics(model, sim, cpu, assembly)
  candidate_count = int(sim._coupled_constraints._workspace["contact_pair_count"][0].cpu())
  assert candidate_count == cpu.ncon > 0
  slots = np.flatnonzero(arrays["contact_mask"][0])
  assert len(slots) == cpu.ncon
  for slot, contact in zip(slots, cpu.contact):
    np.testing.assert_allclose(arrays["contact_position"][0, slot], contact.pos,
                               rtol=0, atol=6e-6)
    np.testing.assert_allclose(arrays["contact_distance"][0, slot], contact.dist,
                               rtol=0, atol=1e-7)
    frame = np.asarray([arrays[name][0, slot] for name in
                        ("contact_normal", "contact_tangent1", "contact_tangent2")])
    np.testing.assert_allclose(frame, contact.frame.reshape(3, 3), rtol=0, atol=6e-5)
  expected = np.zeros(6)
  for index, contact in enumerate(cpu.contact):
    local = np.zeros(6)
    mujoco.mj_contactForce(model, cpu, index, local)
    frame = contact.frame.reshape(3, 3)
    force, moment = frame.T @ local[:3], frame.T @ local[3:]
    sign = 1 if model.geom_bodyid[contact.geom[1]] == 1 else -1
    expected += sign * np.r_[force, moment + np.cross(contact.pos, force)]
  force = arrays["qfrc_constraint"][0, :3]
  moment = cpu.xmat[1].reshape(3, 3) @ arrays["qfrc_constraint"][0, 3:]
  actual = np.r_[force, moment + np.cross(cpu.xpos[1], force)]
  np.testing.assert_allclose(actual, expected, rtol=.03, atol=.02)
