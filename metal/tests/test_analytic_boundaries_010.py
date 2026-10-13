# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Exactly represented contact boundaries and analytic aspect-ratio controls."""
import os
import mujoco
import numpy as np
import pytest

from test_analytic_collision_010 import _PAIRS_010

PAIRS = [("plane", "cylinder"), ("plane", "ellipsoid")] + [
    (entry[1], entry[4]) for entry in _PAIRS_010]


def _fixture(first, second, aspect, reverse):
  sizes = {"sphere": ".0625", "capsule": ".0625 .125",
           "box": ".0625 .03125 .125", "plane": "2 2 .125"}
  half = {"flat": .00390625, "ordinary": .125, "long": .5}[aspect]
  sizes["cylinder"] = f".0625 {half}"
  sizes["ellipsoid"] = f".0625 .03125 {half}"
  if first == "plane":
    geoms = (f'<geom name="a" type="plane" size="{sizes[first]}"/>'
             f'<body><freejoint/><geom name="b" type="{second}" '
             f'size="{sizes[second]}"/></body>')
  else:
    ordered = (second, first) if reverse else (first, second)
    geoms = ''.join(f'<body><freejoint/><geom name="{name}" type="{kind}" '
                    f'size="{sizes[kind]}"/></body>'
                    for name, kind in zip(("a", "b"), ordered))
  # Explicit pairs reach the pinned callback at exact touch. The CPU's
  # body broadphase may omit zero-volume implicit overlaps; that is a
  # distinct filter contract, not a reason to replace touch by penetration.
  order = ("b", "a") if reverse else ("a", "b")
  pair = f'<contact><pair geom1="{order[0]}" geom2="{order[1]}"/></contact>'
  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><option gravity="0 0 0" ccd_tolerance="1e-10" ccd_iterations="1000"/>'
      f'<worldbody>{geoms}</worldbody>{pair}</mujoco>')
  return model, half


@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("aspect", ["flat", "ordinary", "long"])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_boundary_reference_controls(first, second, aspect, reverse):
  model, half = _fixture(first, second, aspect, reverse)
  data = mujoco.MjData(model)
  # All sizes, support reaches and offsets are binary fractions. Neither
  # backend receives a rounded displacement disguised as exact touching.
  for delta in (.015625, 0., -.0009765625, -.03125):
    q = model.qpos0.astype(np.float32)
    if first == "plane":
      q[2] = half + delta
    else:
      q[0:3] = [0, 0, .5]
      q[7:10] = [.125 + delta, 0, .5]
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    if delta > 0:
      assert data.ncon == 0
    if delta < 0:
      assert data.ncon > 0
    if delta == 0:
      assert all(abs(float(c.dist)) < 1e-8 for c in data.contact)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native boundaries")
@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("aspect", ["flat", "ordinary", "long"])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_exact_touch_separation_and_aspect_ratios(first, second, aspect,
                                                       reverse, monkeypatch):
  from mujoco_metal import MetalSimulation
  model, half = _fixture(first, second, aspect, reverse)
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  data = mujoco.MjData(model)
  for delta in (.015625, 0., -.0009765625, -.03125):
    q = model.qpos0.astype(np.float32)
    if first == "plane":
      q[2] = half + delta
    else:
      q[0:3] = [0, 0, .5]
      q[7:10] = [.125 + delta, 0, .5]
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    sim.reset(qpos=q[None], qvel=np.zeros((1, model.nv), np.float32))
    with monkeypatch.context() as context:
      def forbidden(*args, **kwargs):
        raise AssertionError("native analytic boundaries used CPU physics")
      for name in ("mj_forward", "mj_step", "mj_collision", "mj_kinematics"):
        context.setattr(mujoco, name, forbidden)
      assembly = sim.assembled_system(recompute=True)
    np.testing.assert_array_equal(assembly["status"].cpu().numpy(), [0])
    live = assembly["contact_mask"][0].cpu().numpy().astype(bool)
    workspace = sim._coupled_constraints._workspace
    count = int(workspace["contact_pair_count"][0].cpu())
    assert bool(count) == bool(data.ncon), (first, second, aspect, delta)
    # Solver activity is a separate contract: exact touching contacts may
    # be generated with efc_address=-1 and must not produce active rows.
    assert bool(live.any()) == any(c.efc_address >= 0 for c in data.contact)
    if not count:
      continue
    records = workspace["contact_pair_records"].cpu().numpy().reshape(-1, 37)[:count].astype(float)
    distances = records[:, 12] + records[:, 9] + records[:, 10]
    expected = np.asarray([c.dist for c in data.contact])
    # A deeply buried flat cylinder has witnesses on both endcaps with
    # different depths. Compare both sets of actual depths; averaging them
    # would invent an oracle distance that neither backend emitted.
    errors = np.abs(distances[:, None] - expected[None, :])
    assert np.max(np.min(errors, axis=1)) < 1e-7
    assert np.max(np.min(errors, axis=0)) < 1e-7
    if delta == 0:
      assert np.max(np.abs(distances)) < 1e-8
    normals = records[:, 16:19] + records[:, 28:31] + records[:, 34:37]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    cpu_normals = np.asarray([c.frame[:3] for c in data.contact])
    assert np.min(np.max(normals @ cpu_normals.T, axis=1)) > 1 - 1e-5
