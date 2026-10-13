# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Mesh material/CCD options, actual impacts and transactional world isolation."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from test_mesh_asset_physics_011 import (
    KINDS, _model, _state, _check_geometry_and_world_wrenches,
)
from test_analytic_material_lifecycle_010 import _reference, _native, _check_physics


def _variant(tmp_path, kind, reverse, cone, policy="mixed"):
  original = _model(tmp_path, kind, reverse, cone)
  path = tmp_path / "source_variant.xml"
  mujoco.mj_saveLastXML(str(path), original)
  root = ET.fromstring(path.read_text())
  root.remove(root.find("contact"))
  geoms = list(root.find("worldbody").iter("geom"))
  for index, geom in enumerate(geoms):
    geom.set("friction", (".3 .02 .01", ".7 .04 .03")[index])
    geom.set("solref", (".03 1", ".02 .8")[index])
    geom.set("solimp", (".8 .9 .01 .5 2", ".7 .95 .02 .4 3")[index])
    geom.set("solmix", (".3", ".7")[index])
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
  elif policy in ("explicit", "mask"):
    for geom in geoms:
      geom.set("contype", "0")
      geom.set("conaffinity", "0")
    if policy == "explicit":
      contact = ET.SubElement(root, "contact")
      names = ("b", "a") if reverse else ("a", "b")
      ET.SubElement(contact, "pair", geom1=names[0], geom2=names[1], condim="6",
                    friction=".8 .4 .03 .01 .02", solref=".025 .8",
                    solreffriction=".04 .7", solimp=".7 .9 .02 .5 2")
  elif policy == "exclude":
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="world" if kind == "plane" else "body0",
                  body2="body0" if kind == "plane" else "body1")
  else:
    assert policy == "mixed"
  path.write_text(ET.tostring(root, encoding="unicode"))
  return mujoco.MjModel.from_xml_path(str(path))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_mesh_source_material_controls_are_genuine(tmp_path, kind, reverse):
  # Masked/excluded source models cannot locate a contact crossing themselves;
  # use a separately admitted geometric control to retain identical placement.
  original = _model(tmp_path, kind, reverse, "pyramidal")
  q, v = _state(original, kind)
  for policy in ("mixed", "priority", "direct", "override", "explicit", "mask", "exclude"):
    model = _variant(tmp_path, kind, reverse, "pyramidal", policy)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q, v
    mujoco.mj_forward(model, data)
    if policy in ("mask", "exclude"):
      assert data.ncon == data.nefc == 0
    else:
      assert data.ncon and np.linalg.norm(data.qfrc_constraint) > .001


def _boundary_states(model, kind):
  q, v = _state(model, kind)
  far, _ = _state(model, kind, separation=1.)
  moving = (slice(2, 3) if kind == "plane" else slice(7, 10))
  axis = far[moving] - q[moving]
  axis /= np.linalg.norm(axis)
  data = mujoco.MjData(model)
  result = []
  for target in (-.0009765625, .00390625, .01953125):
    lower, upper = -.05, 1.
    for _ in range(40):
      shift = .5 * (lower + upper)
      candidate = q.copy()
      candidate[moving] += shift * axis
      data.qpos[:], data.qvel[:] = candidate, v
      mujoco.mj_forward(model, data)
      # Use the actual callback's contact distances to prepare detection
      # and inclusion controls. A separate query uses a different cutoff
      # and witness mode and does not establish callback candidate presence.
      distance = min((c.dist for c in data.contact), default=2.)
      if distance < target:
        lower = shift
      else:
        upper = shift
    result.append(candidate)
  result.append(far)
  return result, v


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_mesh_source_detection_and_inclusion_controls_are_distinct(tmp_path, kind, reverse):
  model = _model(tmp_path, kind, reverse, "pyramidal")
  model.pair_margin[0], model.pair_gap[0] = .015625, .0078125
  poses, velocity = _boundary_states(model, kind)
  observed = []
  for pose in poses:
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = pose, velocity
    mujoco.mj_forward(model, data)
    observed.append((bool(data.ncon), any(c.efc_address >= 0 for c in data.contact)))
  assert observed == [(True, True), (True, True), (True, False), (False, False)]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual mesh inclusion boundaries")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_every_mesh_family_detection_margin_gap_and_active_solver_boundaries(
    tmp_path, kind, reverse, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = _model(tmp_path, kind, reverse, "pyramidal")
  model.pair_margin[0], model.pair_gap[0] = .015625, .0078125
  poses, velocity = _boundary_states(model, kind)
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  observed = []
  for pose in poses:
    sim.reset(qpos=pose[None], qvel=velocity[None])
    cpu = _reference(model, sim)
    arrays, _ = _check_physics(model, sim, cpu, _native(monkeypatch,
        lambda: sim.assembled_system(recompute=True)))
    count = int(sim._coupled_constraints._workspace["contact_pair_count"][0].cpu())
    active = bool(arrays["contact_mask"].any())
    assert bool(count) == bool(cpu.ncon)
    assert active == any(c.efc_address >= 0 for c in cpu.contact)
    observed.append((bool(count), active))
    if not active:
      np.testing.assert_array_equal(arrays["lambda"], 0)
      np.testing.assert_array_equal(arrays["qfrc_constraint"], 0)
  assert observed == [(True, True), (True, True), (True, False), (False, False)]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual mesh materials")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_every_mesh_family_material_policies_and_complete_frames(
    tmp_path, kind, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  original = _model(tmp_path, kind, reverse, cone)
  q, v = _state(original, kind)
  for policy in ("mixed", "priority", "direct", "override", "explicit", "mask", "exclude"):
    model = _variant(tmp_path, kind, reverse, cone, policy)
    sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
    cpu = _reference(model, sim)
    assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
    arrays, desc = _check_physics(model, sim, cpu, assembly)
    if policy in ("mask", "exclude"):
      # A lowered empty pair universe has no contact views to publish.
      # Qualify its actual metadata, zero rows/forces and real native step,
      # rather than assuming a diagnostic key exists for zero capacity.
      assert cpu.ncon == cpu.nefc == desc.npairs == desc.ncontacts_max == 0
      if "contact_mask" in arrays:
        assert arrays["contact_mask"].size == 0
      np.testing.assert_array_equal(arrays["lambda"], 0)
      np.testing.assert_array_equal(arrays["qfrc_constraint"], 0)
      mujoco.mj_step(model, cpu)
      np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
      np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
      continue
    assert cpu.ncon and np.linalg.norm(cpu.qfrc_constraint) > .001
    _check_geometry_and_world_wrenches(model, cpu, arrays)
    for slot in np.flatnonzero(arrays["contact_mask"][0]):
      point = arrays["contact_position"][0, slot]
      contact = min(cpu.contact, key=lambda c: np.linalg.norm(point - c.pos))
      np.testing.assert_allclose(arrays["contact_friction"][slot], contact.friction,
                                rtol=3e-7, atol=1e-8)
      assert int(desc.contact_condim_packed.reshape(-1, 3)[slot, 0]) == contact.dim
      for field in ("solref", "solreffriction", "solimp"):
        np.testing.assert_allclose(getattr(desc, field)[0], getattr(contact, field),
                                  rtol=3e-7, atol=1e-8)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual mesh CCD settings")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_every_mesh_family_default_and_disabled_ccd_options(tmp_path, kind, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  for flag in (0, int(mujoco.mjtDisableBit.mjDSBL_MULTICCD),
               int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)):
    model = _model(tmp_path, kind, reverse, cone)
    model.opt.disableflags |= flag
    q, v = _state(model, kind)
    sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
    cpu = _reference(model, sim)
    arrays, _ = _check_physics(model, sim, cpu, _native(monkeypatch,
        lambda: sim.assembled_system(recompute=True)))
    _check_geometry_and_world_wrenches(model, cpu, arrays)
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual mesh entering impacts")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_every_mesh_family_genuine_entering_impact(tmp_path, kind, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  model = _model(tmp_path, kind, reverse, cone)
  q, v = _state(model, kind, separation=.00390625)
  if kind == "plane":
    v[2] = -1.
  else:
    # The generic fixture chooses a true source crossing direction. Recover
    # its normal translation axis from its two separated/loaded preparations.
    loaded, _ = _state(model, kind)
    axis = q[7:10] - loaded[7:10]
    axis /= np.linalg.norm(axis)
    v[6:9] = -axis + np.cross(axis, [.03, -.02, .01])
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  assert cpu.ncon == 0
  impacts = 0
  for _ in range(12):
    aligned = _reference(model, sim)
    arrays, _ = _check_physics(model, sim, aligned, _native(monkeypatch,
        lambda: sim.assembled_system(recompute=True)))
    if aligned.ncon:
      _check_geometry_and_world_wrenches(model, aligned, arrays)
      impacts += int(np.linalg.norm(arrays["qfrc_constraint"]) > .001)
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
  assert impacts > 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="actual mesh world isolation")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_mesh_failed_healthy_empty_sticky_restore_and_selected_reset(tmp_path, kind, cone, monkeypatch):
  from dataclasses import fields
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model = _model(tmp_path, kind, False, cone)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
  near, velocity = _state(model, kind)
  far, _ = _state(model, kind, separation=1.)
  q, v = np.stack([near, near, far]), np.repeat(velocity[None], 3, axis=0)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile="integrated_euler_v1")
  before = sim.snapshot()
  force = sim.state._torch.zeros((3, model.nv), device=sim.state._device)
  force[0, 0] = float("nan")
  np.testing.assert_array_equal(_native(monkeypatch,
      lambda: sim.step(qfrc_applied=force)).cpu().numpy(), [1, 0, 0])
  failed = sim.state.snapshot()
  for name in ("qpos", "qvel", "time"):
    np.testing.assert_array_equal(getattr(failed, name)[0], getattr(before["device"], name)[0])
  for world in (1, 2):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q[world], v[world]
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(failed.qpos[world], data.qpos, rtol=0, atol=5e-6)
  assert sim._last_coupled["contact_mask"][1].any().item()
  assert not sim._last_coupled["contact_mask"][2].any().item()
  np.testing.assert_array_equal(_native(monkeypatch,
      lambda: sim.step(qfrc_applied=force.new_zeros(force.shape))).cpu().numpy(), [1, 0, 0])
  sticky = sim.state.snapshot()
  for name in ("qpos", "qvel", "time"):
    np.testing.assert_array_equal(getattr(sticky, name)[0], getattr(failed, name)[0])
  sim.restore(before)
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0, 0, 0])
  expected = sim.snapshot()
  sim.restore(before)
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0, 0, 0])
  _same_checkpoint(sim.snapshot(), expected)
  peer_before = sim.snapshot()["device"]
  sim.reset(env_ids=[0], qpos=far[None], qvel=velocity[None])
  peer_after = sim.snapshot()["device"]
  for field in fields(peer_before):
    original, current = getattr(peer_before, field.name), getattr(peer_after, field.name)
    if isinstance(original, np.ndarray) and original.shape[0] == 3:
      np.testing.assert_array_equal(current[1:], original[1:])
