# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual compiled-hull and hill-climb routes, beyond small inline assets."""
import os
import mujoco
import numpy as np
import pytest

from test_mesh_asset_physics_011 import _state, _check_geometry_and_world_wrenches
from test_mesh_contact_011 import _notch_model
from test_analytic_material_lifecycle_010 import _reference, _native, _check_physics


def _large_hull(count, cone, reverse):
  index = np.arange(count)
  z = 1. - 2. * (index + .5) / count
  angle = index * np.pi * (3. - np.sqrt(5.))
  points = np.column_stack((np.sqrt(1-z*z)*np.cos(angle),
                           np.sqrt(1-z*z)*np.sin(angle), z)) * [.08, .06, .1]
  vertices = " ".join(format(value, ".17g") for value in points.reshape(-1))
  mesh = '<geom type="mesh" mesh="hull"/>'
  box = '<geom type="box" size=".0625 .03125 .09375"/>'
  order = (box, mesh) if reverse else (mesh, box)
  bodies = "".join('<body pos="0 0 .5"><freejoint/><inertial mass=".7" '
                   'pos=".015625 -.0078125 .00390625" quat=".965925826 0 .258819045 0" '
                   'diaginertia=".03 .04 .05"/>' + geom + '</body>' for geom in order)
  return mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".002"
    gravity="0 0 0" cone="{cone}" iterations="1000" tolerance="1e-8"
    ccd_iterations="1000" ccd_tolerance="1e-10"/>
    <default><geom condim="3" friction=".7 .03 .01"/></default>
    <asset><mesh name="hull" vertex="{vertices}"/></asset>
    <worldbody>{bodies}</worldbody></mujoco>''')


@pytest.mark.parametrize("count", [10, 70])
@pytest.mark.parametrize("reverse", [False, True])
def test_large_compiled_hull_graph_source_controls(count, reverse):
  model = _large_hull(count, "pyramidal", reverse)
  assert model.mesh_vertnum[0] == count
  assert model.mesh_graph[int(model.mesh_graphadr[0])] == count
  q, v = _state(model, "box")
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = q, v
  mujoco.mj_forward(model, data)
  assert data.ncon and np.linalg.norm(data.qfrc_constraint) > .001


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native hill-climb physics")
@pytest.mark.parametrize("count", [10, 70])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_actual_hillclimb_large_hull_geometry_force_and_trajectory(count, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = _large_hull(count, cone, reverse)
  q, v = _state(model, "box")
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  arrays, _ = _check_physics(model, sim, cpu, _native(monkeypatch,
      lambda: sim.assembled_system(recompute=True)))
  _check_geometry_and_world_wrenches(model, cpu, arrays)
  assert np.linalg.norm(arrays["qfrc_constraint"]) > .001
  for _ in range(8):
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native compiled convex representation")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_nonconvex_visual_surface_uses_actual_compiled_collision_hull(cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = _notch_model()
  model.opt.cone = getattr(mujoco.mjtCone, "mjCONE_" + cone.upper())
  q = model.qpos0.copy()
  q[7:10] = [1., .5, .95]
  v = np.zeros(model.nv)
  v[6:9] = [.01, -.02, -.03]
  # The sphere lies in the authored surface's notch, above its z=.5 floor
  # and between its x=.5/1.5 sides. Pinned ordinary mesh collision still
  # sees the compiled convex hull, not a concave triangle narrow phase.
  assert q[7] - .1 > .5 and q[7] + .1 < 1.5 and q[9] - .1 > .5
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  assert cpu.ncon and np.linalg.norm(cpu.qfrc_constraint) > .001
  arrays, _ = _check_physics(model, sim, cpu, _native(monkeypatch,
      lambda: sim.assembled_system(recompute=True)))
  _check_geometry_and_world_wrenches(model, cpu, arrays)
