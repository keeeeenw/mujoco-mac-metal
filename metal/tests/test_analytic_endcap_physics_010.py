# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Aligned endcap surface, force, full-mass and loaded replay qualification."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest

from test_analytic_collision_010 import _pair_xml, _free_qpos, _axis_angle_quat


def _surface_distance(model, data, geom, point):
  local = data.geom_xmat[geom].reshape(3, 3).T @ (point - data.geom_xpos[geom])
  radius, half = model.geom_size[geom, :2]
  if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
    local[2] -= np.clip(local[2], -half, half)
    return np.linalg.norm(local) - radius
  assert int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
  gap = np.array([np.linalg.norm(local[:2]) - radius, abs(local[2]) - half])
  return np.linalg.norm(np.maximum(gap, 0)) + min(float(gap.max()), 0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native endcap physics")
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("motion", ["press", "slide", "spin"])
def test_endcap_actual_surfaces_world_wrenches_full_mass_and_loaded_replay(
    reverse, cone, motion, monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  root = ET.fromstring(_pair_xml("capsule", ".04 .08", "cylinder", ".05 .1",
                                swap=reverse))
  root.find("option").set("cone", cone)
  model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
  q = [_free_qpos(0, 0, .5),
       _free_qpos(.089, 0, .5, _axis_angle_quat([0, 1, 0], np.pi / 6))]
  v = [np.zeros(6), np.zeros(6)]
  if motion == "slide":
    v[1][:3] = [-.02, .1, .05]
  if motion == "spin":
    v[1][3:] = [.2, -.1, .3]
  if reverse:
    q.reverse()
    v.reverse()
  sim = MetalSimulation(model, 1, qpos=np.asarray(q, np.float32).reshape(1, -1),
                        qvel=np.asarray(v, np.float32).reshape(1, -1),
                        profile="integrated_euler_v1")
  state = sim.state.snapshot()
  words = state.precision_words[0].astype(float)
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = state.qpos[0].astype(float) + words[:model.nq] + words[model.nq:2*model.nq]
  cpu.qvel[:] = state.qvel[0].astype(float) + words[2*model.nq:2*model.nq+model.nv] + words[2*model.nq+model.nv:]
  mujoco.mj_forward(model, cpu)
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, cpu, mass)
  with monkeypatch.context() as context:
    def forbidden(*args, **kwargs):
      raise AssertionError("native endcap physics used CPU stepping or collision")
    for name in ("mj_step", "mj_forward", "mj_collision", "mj_kinematics"):
      context.setattr(mujoco, name, forbidden)
    assembly = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(assembly["status"].cpu().numpy(), [0])
  live = assembly["contact_mask"][0].cpu().numpy().astype(bool)
  assert live.any() and cpu.ncon > 0
  desc = sim._coupled_constraints.descriptor
  first, second = int(desc.geom1[0]), int(desc.geom2[0])
  for slot in np.flatnonzero(live):
    point = assembly["contact_position"][0, slot].cpu().numpy().astype(float)
    normal = assembly["contact_normal"][0, slot].cpu().numpy().astype(float)
    distance = float(assembly["contact_distance"][0, slot].cpu())
    assert abs(_surface_distance(model, cpu, first, point - .5 * distance * normal)) < 6e-6
    assert abs(_surface_distance(model, cpu, second, point + .5 * distance * normal)) < 6e-6
    assert min(np.linalg.norm(point - c.pos) for c in cpu.contact) < 6e-6
    assert min(np.linalg.norm(normal - c.frame[:3]) for c in cpu.contact) < 6e-5
  force = assembly["qfrc_constraint"][0].cpu().numpy().astype(float)
  acceleration = assembly["qacc"][0].cpu().numpy().astype(float)
  assert np.linalg.norm(force) > .1
  np.testing.assert_allclose(force, cpu.qfrc_constraint, rtol=3e-5, atol=3e-4)
  np.testing.assert_allclose(mass @ acceleration, mass @ cpu.qacc, rtol=3e-5, atol=3e-4)
  np.testing.assert_allclose(assembly["J"][0].cpu().numpy().T @
                            assembly["lambda"][0].cpu().numpy(), force,
                            rtol=3e-6, atol=2e-5)
  wrenches = np.zeros((model.nbody, 6))
  for contact_index, contact in enumerate(cpu.contact):
    local = np.zeros(6)
    mujoco.mj_contactForce(model, cpu, contact_index, local)
    frame = contact.frame.reshape(3, 3)
    linear, angular = frame.T @ local[:3], frame.T @ local[3:]
    for sign, geom in ((-1, contact.geom[0]), (1, contact.geom[1])):
      wrenches[model.geom_bodyid[geom]] += sign * np.r_[linear, angular + np.cross(contact.pos, linear)]
  for body in (1, 2):
    offset = int(model.jnt_dofadr[model.body_jntadr[body]])
    linear = force[offset:offset + 3]
    moment = cpu.xmat[body].reshape(3, 3) @ force[offset + 3:offset + 6]
    actual = np.r_[linear, moment + np.cross(cpu.xpos[body], linear)]
    np.testing.assert_allclose(actual, wrenches[body], rtol=3e-5, atol=3e-4)
  # Preserve full source trajectory gates, then replay every public and
  # retained checkpoint field with changing held forces and warm contacts.
  for _ in range(8):
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(sim.step().cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
  before = sim.snapshot()
  commands = [.001 * np.cos(np.arange(model.nv) + i).astype(np.float32)[None]
              for i in range(20)]
  for command in commands:
    np.testing.assert_array_equal(sim.step(qfrc_applied=command).cpu().numpy(), [0])
  expected = sim.snapshot()
  sim.restore(before)
  for command in commands:
    np.testing.assert_array_equal(sim.step(qfrc_applied=command).cpu().numpy(), [0])
  _same_checkpoint(sim.snapshot(), expected)
