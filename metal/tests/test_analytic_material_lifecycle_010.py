# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Executed analytic material/frame semantics and frictional lifecycle gates.

The source oracle consumes the public state's represented initial pose. All
forces and certificates refer to the emitted native iterate, never a re-solve.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from test_analytic_boundaries_010 import PAIRS, _fixture
from test_rigid_composition_complete_009 import _retained_certificate


def _model(first, second, reverse, cone, policy="mixed", margin=0., gap=0.):
  # Use the exact binary-fraction dimensions of the boundary qualification.
  original, half = _fixture(first, second, "ordinary", reverse)
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "fixture.xml"
    mujoco.mj_saveLastXML(str(path), original)
    root = ET.fromstring(path.read_text())
  option = root.find("option")
  option.set("timestep", ".002")
  option.set("cone", cone)
  option.set("iterations", "1000")
  option.set("tolerance", "1e-8")
  contact = root.find("contact")
  root.remove(contact)
  geoms = list(root.find("worldbody").iter("geom"))
  for i, geom in enumerate(geoms):
    geom.set("friction", (".3 .02 .01", ".7 .04 .03")[i])
    geom.set("solref", (".03 1", ".02 .8")[i])
    geom.set("solimp", (".8 .9 .01 .5 2", ".7 .95 .02 .4 3")[i])
    geom.set("solmix", (".3", ".7")[i])
    geom.set("condim", "3")
    geom.set("margin", str(margin))
    geom.set("gap", str(gap))
  if policy == "priority":
    geoms[1].set("priority", "2")
    geoms[1].set("condim", "6")
  elif policy == "direct":
    geoms[1].set("solref", "-1000 -20")
    geoms[1].set("condim", "4")
  elif policy == "override":
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
    contact = ET.SubElement(root, "contact")
    names = ("b", "a") if reverse else ("a", "b")
    ET.SubElement(contact, "pair", geom1=names[0], geom2=names[1],
                  condim="6", friction=".8 .4 .03 .01 .02",
                  solref=".025 .8", solreffriction=".04 .7",
                  solimp=".7 .9 .02 .5 2", margin=str(margin), gap=str(gap))
  elif policy == "mask":
    for geom in geoms:
      geom.set("contype", "0")
      geom.set("conaffinity", "0")
  elif policy == "exclude":
    world = root.find("worldbody")
    bodies = list(world.findall("body"))
    for i, body in enumerate(bodies):
      body.set("name", f"body{i}")
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="world" if first == "plane" else "body0",
                  body2="body0" if first == "plane" else "body1")
  else:
    assert policy == "mixed"
  return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), half


def _state(model, first, half, separation=-.0009765625, moving=False):
  q = model.qpos0.astype(np.float32).copy()
  v = np.zeros(model.nv, np.float32)
  if first == "plane":
    q[:3] = [0, 0, half + separation]
    if moving:
      v[:] = [.08, .03, -.02, .1, -.05, .07]
  else:
    q[:3] = [0, 0, .5]
    q[7:10] = [.125 + separation, 0, .5]
    if moving:
      v[6:] = [-.02, .08, .03, .1, -.05, .07]
  return q, v


def _reference(model, sim):
  snap = sim.state.snapshot()
  words = snap.precision_words[0].astype(float)
  data = mujoco.MjData(model)
  data.qpos[:] = snap.qpos[0].astype(float) + words[:model.nq] + words[model.nq:2 * model.nq]
  data.qvel[:] = snap.qvel[0].astype(float) + words[2 * model.nq:2 * model.nq + model.nv] + words[2 * model.nq + model.nv:]
  mujoco.mj_forward(model, data)
  return data


def _native(monkeypatch, operation):
  with monkeypatch.context() as context:
    def forbidden(*args, **kwargs):
      raise AssertionError("analytic native execution used CPU physics")
    for name in ("mj_step", "mj_forward", "mj_collision", "mj_kinematics"):
      context.setattr(mujoco, name, forbidden)
    return operation()


def _check_physics(model, sim, data, assembly):
  desc = sim._coupled_constraints.descriptor
  arrays = {name: value.detach().cpu().numpy().copy()
            for name, value in assembly.items() if hasattr(value, "cpu")}
  np.testing.assert_array_equal(arrays["status"], [0])
  live = arrays["R"][0] > 0
  np.testing.assert_array_equal(live, arrays["row_island"][0] >= 0)
  assert np.count_nonzero(live) == data.nefc
  J, force, lam = (arrays[name][0] for name in ("J", "qfrc_constraint", "lambda"))
  np.testing.assert_allclose(J.T @ lam, force, rtol=5e-6, atol=5e-4)
  mass = np.zeros((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  np.testing.assert_allclose(arrays["mass_matrix"][0].reshape(model.nv, model.nv),
                             mass, rtol=3e-6, atol=1e-7)
  np.testing.assert_allclose(force, data.qfrc_constraint, rtol=.03, atol=.02)
  np.testing.assert_allclose(mass @ arrays["qacc"][0], mass @ data.qacc,
                             rtol=.03, atol=.02)
  assert _retained_certificate(arrays, desc) < 2e-6
  np.testing.assert_allclose(arrays["W"][0], J @ np.linalg.solve(mass, J.T),
                             rtol=3e-6, atol=5e-4)
  return arrays, desc


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native analytic semantics")
@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_all_analytic_material_policies_filters_and_full_frames(first, second, reverse,
                                                             cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  for policy in ("mixed", "priority", "direct", "override", "explicit", "mask", "exclude"):
    model, half = _model(first, second, reverse, cone, policy)
    q, v = _state(model, first, half, moving=True)
    sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
    cpu = _reference(model, sim)
    if policy in ("mask", "exclude"):
      assert cpu.ncon == 0 and cpu.nefc == 0
      np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
      mujoco.mj_step(model, cpu)
      np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
      continue
    assert cpu.ncon > 0 and cpu.nefc > 0, (first, second, policy)
    assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
    arrays, desc = _check_physics(model, sim, cpu, assembly)
    assert np.linalg.norm(arrays["qfrc_constraint"]) > .1
    slots = np.flatnonzero(arrays["contact_mask"][0])
    assert slots.size == cpu.ncon
    used = set()
    for slot in slots:
      point = arrays["contact_position"][0, slot]
      candidates = [(np.linalg.norm(point - contact.pos), index)
                    for index, contact in enumerate(cpu.contact) if index not in used]
      error, index = min(candidates)
      assert error < 6e-6, (first, second, policy, error)
      used.add(index)
      contact = cpu.contact[index]
      frame = np.asarray([arrays[name][0, slot] for name in
                          ("contact_normal", "contact_tangent1", "contact_tangent2")])
      np.testing.assert_allclose(frame, contact.frame.reshape(3, 3), rtol=0, atol=6e-5)
      np.testing.assert_allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-6)
      assert np.linalg.det(frame) > .999999
      np.testing.assert_allclose(arrays["contact_friction"][slot], contact.friction,
                                rtol=3e-7, atol=1e-8)
      assert int(desc.contact_condim_packed.reshape(-1, 3)[slot, 0]) == contact.dim
      for field in ("solref", "solreffriction", "solimp"):
        # These fixtures have one candidate pair. Material solref/solimp
        # belong to that pair, while friction dimensions have slot spans.
        actual = getattr(desc, field)[0]
        np.testing.assert_allclose(actual, getattr(contact, field), rtol=3e-7, atol=1e-8)
      np.testing.assert_allclose(arrays["contact_distance"][0, slot], contact.dist,
                                rtol=0, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native analytic inclusion")
@pytest.mark.parametrize("first,second", PAIRS + [("plane", "box"), ("capsule", "box")])
@pytest.mark.parametrize("reverse", [False, True])
def test_all_analytic_margin_gap_candidate_and_solver_boundaries(first, second, reverse,
                                                              monkeypatch):
  from mujoco_metal import MetalSimulation
  model, half = _model(first, second, reverse, "pyramidal", "explicit", margin=.015625, gap=.0078125)
  sim = MetalSimulation(model, 1, profile="integrated_euler_v1")
  observations = []
  # Pinned 3.10 detects through margin+gap; solver inclusion stays at
  # margin. The third distance must lie BETWEEN those two boundaries.
  for distance in (-.0009765625, .00390625, .01953125, .03125):
    q, v = _state(model, first, half, distance)
    sim.reset(qpos=q[None], qvel=v[None])
    cpu = _reference(model, sim)
    assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
    arrays, desc = _check_physics(model, sim, cpu, assembly)
    workspace = sim._coupled_constraints._workspace
    count = int(workspace["contact_pair_count"][0].cpu())
    assert bool(count) == bool(cpu.ncon), (first, second, distance)
    active = bool(arrays["contact_mask"][0].any())
    assert active == any(contact.efc_address >= 0 for contact in cpu.contact)
    observations.append((bool(count), active))
    if not active:
      np.testing.assert_array_equal(arrays["lambda"], 0)
      np.testing.assert_array_equal(arrays["qfrc_constraint"], 0)
    for slot in np.flatnonzero(arrays["contact_mask"][0]):
      assert min(abs(float(arrays["contact_distance"][0, slot]) - contact.dist)
                 for contact in cpu.contact) < 1e-7
  assert observations == [(True, True), (True, True), (True, False), (False, False)]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native analytic CCD options")
@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_every_new_analytic_pair_admitted_ccd_options(first, second, reverse, cone,
                                                    monkeypatch):
  from mujoco_metal import MetalSimulation
  for flag in (0, int(mujoco.mjtDisableBit.mjDSBL_MULTICCD),
               int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)):
    model, half = _model(first, second, reverse, cone)
    model.opt.disableflags |= flag
    q, v = _state(model, first, half, moving=True)
    sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
    cpu = _reference(model, sim)
    assert cpu.ncon > 0 and cpu.nefc > 0
    assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
    arrays, _ = _check_physics(model, sim, cpu, assembly)
    assert np.linalg.norm(arrays["qfrc_constraint"]) > .1
    for slot in np.flatnonzero(arrays["contact_mask"][0]):
      assert min(abs(float(arrays["contact_distance"][0, slot]) - contact.dist)
                 for contact in cpu.contact) < 1e-6
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native analytic lifecycle")
@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_every_new_analytic_pair_frictional_trajectory_history_and_full_replay(
    first, second, reverse, cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model, half = _model(first, second, reverse, cone)
  q, v = _state(model, first, half, moving=True)
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
  arrays, desc = _check_physics(model, sim, cpu, assembly)
  assert np.linalg.norm(arrays["qfrc_constraint"]) > .1 and cpu.ncon > 0
  assert np.linalg.norm(arrays["contact_wrench"][0, :, 1:]) > .001
  for _ in range(8):
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
  before = sim.snapshot()
  commands = [.001 * np.cos(np.arange(model.nv) + i).astype(np.float32)[None]
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
  # A real public tensor write changes candidates without resetting warm
  # history. Restore brings back the entire loaded history, not just qpos.
  far, _ = _state(model, first, half, separation=1.)
  sim.state._qpos[0].copy_(sim.state._torch.as_tensor(far, device=sim.state._device))
  sim.state._qvel.zero_()
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
  assert not sim._last_coupled["contact_mask"].any().item()
  np.testing.assert_array_equal(sim._last_coupled["lambda"].cpu().numpy(), 0)
  sim.restore(before)
  _same_checkpoint(sim.snapshot(), before)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native analytic impacts")
@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_every_new_analytic_pair_enters_actual_frictional_impact(first, second, reverse,
                                                             cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  model, half = _model(first, second, reverse, cone)
  q, v = _state(model, first, half, separation=.00390625, moving=True)
  v[2 if first == "plane" else 6] = -1.
  sim = MetalSimulation(model, 1, qpos=q[None], qvel=v[None], profile="integrated_euler_v1")
  cpu = _reference(model, sim)
  assert cpu.ncon == 0
  loaded = 0
  for _ in range(12):
    # Compare operators at the actual native pre-step state in addition to
    # the uninterrupted source trajectory. This separates accumulated
    # path differences from the contact operator's own force correctness.
    aligned = _reference(model, sim)
    assembly = _native(monkeypatch, lambda: sim.assembled_system(recompute=True))
    arrays, _ = _check_physics(model, sim, aligned, assembly)
    if aligned.ncon:
      assert arrays["contact_mask"].any()
      loaded += int(np.linalg.norm(arrays["qfrc_constraint"]) > .1)
    mujoco.mj_step(model, cpu)
    np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0])
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, rtol=0, atol=5e-6)
  assert loaded > 0, "A separated one-step control does not qualify an impact"


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native analytic failure isolation")
@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_analytic_failed_healthy_empty_worlds_full_restore_and_mask(first, second,
                                                                cone, monkeypatch):
  from mujoco_metal import MetalSimulation
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  from magnetic_crane import _same_checkpoint
  model, half = _model(first, second, False, cone)
  # Pinned automatic BADQACC reset is a distinct recovery mode. This gate
  # checks rollback/sticky failure, so explicitly select its documented
  # AUTORESET-disabled contract instead of assuming default reset is rollback.
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
  near, velocity = _state(model, first, half, moving=True)
  far, _ = _state(model, first, half, separation=1.)
  q = np.stack([near, near, far])
  v = np.repeat(velocity[None], 3, axis=0)
  sim = MetalSimulation(model, 3, qpos=q, qvel=v, profile="integrated_euler_v1")
  before = sim.snapshot()
  # Exercise the documented device-force failure contract on each family.
  # Malformed CCD support/metadata has separate real producer regressions;
  # changing an unused quaternion is not a valid fault for a matrix reader.
  forces = sim.state._torch.zeros((3, model.nv), device=sim.state._device)
  forces[0, 0] = float("nan")
  np.testing.assert_array_equal(_native(monkeypatch,
      lambda: sim.step(qfrc_applied=forces)).cpu().numpy(), [1, 0, 0])
  failed = sim.state.snapshot()
  for field in ("qpos", "qvel", "time"):
    np.testing.assert_array_equal(getattr(failed, field)[0], getattr(before["device"], field)[0])
  for world in (1, 2):
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = q[world]
    cpu.qvel[:] = v[world]
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
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0, 0, 0])
  expected = sim.snapshot()
  sim.restore(before)
  np.testing.assert_array_equal(_native(monkeypatch, sim.step).cpu().numpy(), [0, 0, 0])
  _same_checkpoint(sim.snapshot(), expected)
  # Execute the public selected-world reset mask. It must preserve every
  # peer device field, including residual words and retained warm seeds.
  from dataclasses import fields
  peer_before = sim.snapshot()["device"]
  sim.reset(env_ids=[0], qpos=far[None], qvel=velocity[None])
  peer_after = sim.snapshot()["device"]
  for field in fields(peer_before):
    original = getattr(peer_before, field.name)
    current = getattr(peer_after, field.name)
    if isinstance(original, np.ndarray) and original.shape[0] == 3:
      np.testing.assert_array_equal(current[1:], original[1:])
