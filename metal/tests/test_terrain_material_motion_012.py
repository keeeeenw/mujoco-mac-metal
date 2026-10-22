# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Executed terrain materials, cross-cell motion, impacts and loaded history."""
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from test_terrain_geometry_physics_012 import KINDS, LIMITS, _model as _terrain_model, _state
from test_analytic_material_lifecycle_010 import _check_physics, _native, _reference

MOTION_STEPS = 250
DRIVE = np.asarray([10., -.5, 0., .001, -.002, .001], np.float32)


def _model(kind, reverse, cone, policy="mixed"):
  original = _terrain_model(kind, "slope", reverse, cone)
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "terrain.xml"
    mujoco.mj_saveLastXML(str(path), original)
    root = ET.fromstring(path.read_text())
  contact = root.find("contact")
  root.remove(contact)
  geoms = list(root.find("worldbody").iter("geom"))
  assert len(geoms) == 2
  for i, geom in enumerate(geoms):
    geom.set("friction", (".3 .02 .01", ".7 .04 .03")[i])
    geom.set("solref", (".03 1", ".02 .8")[i])
    geom.set("solimp", (".8 .9 .01 .5 2", ".7 .95 .02 .4 3")[i])
    geom.set("solmix", (".3", ".7")[i])
    geom.set("condim", "3")
  if policy == "priority":
    geoms[1].set("priority", "2")
    geoms[1].set("condim", "6")
  elif policy == "direct":
    geoms[1].set("solref", "-1000 -20")
    geoms[1].set("condim", "4")
  elif policy == "override":
    option = root.find("option")
    flag = option.find("flag")
    if flag is None:
      flag = ET.SubElement(option, "flag")
    flag.set("override", "enable")
    option.set("o_solref", ".015 .9")
    option.set("o_solimp", ".75 .95 .02 .5 2")
    option.set("o_friction", ".6 .4 .03 .01 .02")
    option.set("o_margin", ".006")
  elif policy == "explicit":
    for geom in geoms:
      geom.set("contype", "0")
      geom.set("conaffinity", "0")
    names = ("object", "terrain") if reverse else ("terrain", "object")
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "pair", geom1=names[0], geom2=names[1],
                  condim="6", friction=".8 .4 .03 .01 .02",
                  solref=".025 .8", solreffriction=".04 .7",
                  solimp=".7 .9 .02 .5 2")
  elif policy == "mask":
    for geom in geoms:
      geom.set("contype", "0")
      geom.set("conaffinity", "0")
  elif policy == "exclude":
    body = root.find("worldbody/body")
    body.set("name", "object_body")
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="world", body2="object_body")
  else:
    assert policy == "mixed"
  return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_source_terrain_material_controls_have_real_load_and_full_mass(kind, reverse):
  for policy in ("mixed", "priority", "direct", "override", "explicit"):
    model = _model(kind, reverse, "elliptic", policy)
    q, v = _state(model)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mujoco.mj_forward(model, data)
    assert data.ncon > 0 and data.nefc > 0
    assert np.linalg.norm(data.qfrc_constraint) > .1
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    assert np.max(np.abs(mass - np.diag(mass.diagonal()))) > .001


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_source_terrain_motion_really_impacts_and_crosses_two_cells(kind, cone):
  model = _model(kind, False, cone)
  q, v = _state(model)
  q[2] += .025
  v[:] = [1.25, -.15, -.4, .07, -.05, .03]
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = q, v
  mujoco.mj_forward(model, data)
  assert data.ncon == 0
  axis = data.geom_xmat[model.geom("terrain").id].reshape(3, 3)[:, 0].copy()
  start = data.qpos[:3].copy()
  contact_steps = 0
  for _ in range(MOTION_STEPS):
    data.qfrc_applied[:] = DRIVE
    mujoco.mj_step(model, data)
    contact_steps += int(data.ncon > 0)
  assert contact_steps > 10
  assert float((data.qpos[:3] - start) @ axis) > 2 * (2 * model.hfield_size[0, 0] / (model.hfield_ncol[0] - 1))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native terrain materials")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_all_terrain_material_policies_filters_and_full_frames(kind, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  for policy in ("mixed", "priority", "direct", "override", "explicit", "mask", "exclude"):
    model = _model(kind, reverse, cone, policy)
    # Filtered controls keep the same loaded geometry, prepared with the
    # source collision-enabled fixture rather than inventing an empty pose.
    q, v = _state(_model(kind, reverse, cone, "mixed") if policy in ("mask", "exclude") else model)
    sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None],
                          profile="integrated_euler_v1", limits=LIMITS)
    cpu = _reference(model, sim)
    assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
    if policy in ("mask", "exclude"):
      assert cpu.ncon == 0 and cpu.nefc == 0
      assert not sim._coupled_constraints.descriptor.npairs
      np.testing.assert_array_equal(assembly["qfrc_constraint"].cpu().numpy(), 0)
      np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
      mujoco.mj_step(model, cpu)
      np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
      continue
    arrays, desc = _check_physics(model, sim, cpu, assembly)
    assert np.linalg.norm(arrays["qfrc_constraint"]) > .1
    assert int(sim._coupled_constraints._workspace["contact_pair_count"][0].cpu()) == cpu.ncon
    slots = np.flatnonzero(arrays["contact_mask"][0])
    assert len(slots) == cpu.ncon
    for slot, contact in zip(slots, cpu.contact):
      np.testing.assert_allclose(arrays["contact_position"][0, slot], contact.pos, rtol=0, atol=6e-6)
      np.testing.assert_allclose(arrays["contact_distance"][0, slot], contact.dist, rtol=0, atol=1e-7)
      frame = np.array([arrays[name][0, slot] for name in
                        ("contact_normal", "contact_tangent1", "contact_tangent2")])
      np.testing.assert_allclose(frame, contact.frame.reshape(3, 3), rtol=0, atol=6e-5)
      np.testing.assert_allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-6)
      assert np.linalg.det(frame) > .999999
      np.testing.assert_allclose(arrays["contact_friction"][slot], contact.friction, rtol=3e-7, atol=1e-8)
      assert int(desc.contact_condim_packed.reshape(-1, 3)[slot, 0]) == contact.dim
      for field in ("solref", "solreffriction", "solimp"):
        np.testing.assert_allclose(getattr(desc, field)[0], getattr(contact, field), rtol=3e-7, atol=1e-8)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native terrain cross-cell motion")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_terrain_cross_cell_friction_entering_impact_and_full_loaded_replay(kind, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model = _model(kind, reverse, cone)
  q, v = _state(model)
  q[2] += .025
  v[:] = [1.25, -.15, -.4, .07, -.05, .03]
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None],
                        profile="integrated_euler_v1", limits=LIMITS)
  cpu = _reference(model, sim)
  assert cpu.ncon == 0
  contact_steps = 0
  friction_steps = 0
  candidate_counts = set()
  positions = []
  for step in range(MOTION_STEPS):
    applied = DRIVE
    # The uninterrupted trajectory oracle gates position and velocity. The
    # instantaneous acceleration/force operator must use the actual retained
    # pre-step pose: pinned prism traversal can change a witness for nearby
    # poses even while the complete trajectories remain within their bounds.
    # This source reference is diagnostic input to assertions only, never an
    # input to native execution or a replacement for the continuous oracle.
    aligned = _reference(model, sim)
    aligned.qfrc_applied[:] = applied
    mujoco.mj_forward(model, aligned)
    cpu.qfrc_applied[:] = applied
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch,
        lambda: sim.step(qfrc_applied=applied[None])).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
    np.testing.assert_allclose(sim.state.qvel[0].cpu().numpy(), cpu.qvel, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(sim.state.qacc[0].cpu().numpy(), aligned.qacc, rtol=2e-3, atol=2e-3)
    actual = sim._last_coupled
    counts = int(sim._coupled_constraints._workspace["contact_pair_count"][0].cpu())
    assert counts == aligned.ncon
    np.testing.assert_allclose(actual["qfrc_constraint"][0].cpu().numpy(),
                               aligned.qfrc_constraint, rtol=2e-3, atol=2e-3)
    candidate_counts.add(counts)
    if actual["contact_mask"].any().item():
      contact_steps += 1
      assert np.linalg.norm(actual["qfrc_constraint"].cpu().numpy()) > .01
      friction_steps += int(np.linalg.norm(actual["contact_wrench"][0, :, 1:3].cpu().numpy()) > .001)
    positions.append(sim.state.qpos[0, :3].cpu().numpy().copy())
  assert contact_steps > 10 and friction_steps > 10 and len(candidate_counts) > 1
  # Two grid columns, not an eight-step stationary press. The transform
  # preserves column direction while the body crosses the pitched terrain.
  axis = _reference(model, sim).geom_xmat[model.geom("terrain").id].reshape(3, 3)[:, 0]
  travel = float((positions[-1] - positions[0]) @ axis)
  assert travel > 2 * (2 * model.hfield_size[0, 0] / (model.hfield_ncol[0] - 1))
  before = sim.snapshot()
  commands = [(.01 * np.cos(np.arange(model.nv) + i)).astype(np.float32)[None] for i in range(20)]
  for command in commands:
    np.testing.assert_array_equal(_native(monkeypatch, lambda: sim.step(qfrc_applied=command)).cpu().numpy(), [0])
  expected = sim.snapshot()
  sim.restore(before)
  for command in commands:
    np.testing.assert_array_equal(_native(monkeypatch, lambda: sim.step(qfrc_applied=command)).cpu().numpy(), [0])
  _same_checkpoint(sim.snapshot(), expected)
  sim.restore(before)
  _same_checkpoint(sim.snapshot(), before)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native terrain lifecycle")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_terrain_failed_healthy_empty_mutation_selected_reset_and_full_restore(kind, cone, monkeypatch):
  from dataclasses import fields
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model = _model(kind, False, cone)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
  near, velocity = _state(model)
  far = near.copy()
  far[2] = .8
  q = np.stack([near, near, far])
  v = np.repeat(velocity[None], 3, axis=0)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v,
                        profile="integrated_euler_v1", limits=LIMITS)
  before = sim.snapshot()
  forces = sim.state._torch.zeros((3, model.nv), device=sim.state._device)
  forces[0, 0] = float("nan")
  np.testing.assert_array_equal(_native(monkeypatch,
      lambda: sim.step(qfrc_applied=forces)).cpu().numpy(), [1, 0, 0])
  failed = sim.state.snapshot()
  for field in ("qpos", "qvel", "time"):
    np.testing.assert_array_equal(getattr(failed, field)[0], getattr(before["device"], field)[0])
  for world in (1, 2):
    cpu = mujoco.MjData(model)
    cpu.qpos[:], cpu.qvel[:] = q[world], v[world]
    mujoco.mj_step(model, cpu)
    np.testing.assert_allclose(failed.qpos[world], cpu.qpos, rtol=0, atol=5e-6)
  assert sim._last_coupled["contact_mask"][1].any().item()
  assert not sim._last_coupled["contact_mask"][2].any().item()
  np.testing.assert_array_equal(_native(monkeypatch,
      lambda: sim.step(qfrc_applied=forces.new_zeros(forces.shape))).cpu().numpy(), [1, 0, 0])
  sticky = sim.state.snapshot()
  for field in ("qpos", "qvel", "time"):
    np.testing.assert_array_equal(getattr(sticky, field)[0], getattr(failed, field)[0])
  sim.restore(before)
  commands = [(.001 * np.cos(np.arange(model.nv) + i)).astype(np.float32) for i in range(20)]
  for command in commands:
    applied = np.repeat(command[None], 3, axis=0)
    np.testing.assert_array_equal(_native(monkeypatch,
        lambda: sim.step(qfrc_applied=applied)).cpu().numpy(), [0, 0, 0])
  expected = sim.snapshot()
  sim.restore(before)
  for command in commands:
    applied = np.repeat(command[None], 3, axis=0)
    np.testing.assert_array_equal(_native(monkeypatch,
        lambda: sim.step(qfrc_applied=applied)).cpu().numpy(), [0, 0, 0])
  _same_checkpoint(sim.snapshot(), expected)
  sim.restore(before)
  # Real MPS tensor mutation must invalidate the pose/manifold cache without
  # resetting loaded peers or supplying a source-generated contact.
  sim.state._qpos[1].copy_(sim.state._torch.as_tensor(far, device=sim.state._device))
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0, 0, 0])
  assert sim._last_coupled["contact_mask"][0].any().item()
  assert not sim._last_coupled["contact_mask"][1:].any().item()
  np.testing.assert_array_equal(sim._last_coupled["lambda"][1:].cpu().numpy(), 0)
  sim.restore(before)
  _same_checkpoint(sim.snapshot(), before)
  peer_before = sim.snapshot()["device"]
  sim.reset(env_ids=[0], qpos=far[None], qvel=velocity[None])
  peer_after = sim.snapshot()["device"]
  for field in fields(peer_before):
    value = getattr(peer_before, field.name)
    if isinstance(value, np.ndarray) and value.shape[0] == 3:
      np.testing.assert_array_equal(getattr(peer_after, field.name)[1:], value[1:])
