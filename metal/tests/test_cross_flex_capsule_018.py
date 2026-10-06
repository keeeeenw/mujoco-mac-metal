# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Pinned 1D flex/flex capsule contacts for internal and kind-6 routes.

The internal test covers the admitted direct two-node vertex/edge route end
to end. Cross and self tests exercise the already admitted order-0 kind-6
capsule route when available; the local detector override is reserved for any
future fixture that is outside the public narrowphase set.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    FlexContactProgram,
    _KIND_ELEMENT_PAIR,
    _KIND_INTERNAL_VERTEX_ELEMENT,
    lower_flex_contacts,
)


def _flexcomp(name, x, count, selfcollide):
  return f"""
    <flexcomp name="{name}" type="grid" count="{count} 1 1"
              pos="{x} 0 0" spacing=".05 .05 .05" mass="1" dim="1">
      <contact contype="1" conaffinity="1" selfcollide="{selfcollide}"/>
      <edge stiffness="10" damping=".1"/>
      <elasticity young="100" poisson=".2"/>
    </flexcomp>
  """


def _fixture(which, jacobian=None):
  if which == "cross":
    flexes = _flexcomp("left", 0, 3, "none")
    flexes += _flexcomp("right", .03, 3, "none")
  elif which == "self":
    flexes = _flexcomp("strand", 0, 4, "narrow")
  elif which == "internal":
    flexes = _flexcomp("strand", 0, 4, "none").replace(
        'selfcollide="none"', 'selfcollide="none" internal="true"')
  else:
    raise ValueError(f"unknown capsule fixture {which!r}")
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option gravity='0 0 0'/><worldbody>"
      + flexes + "</worldbody></mujoco>")
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  data = mujoco.MjData(model)
  if which in ("self", "internal"):
    # Fold element 2 back over element 0 with a nonzero transverse offset.
    # These four slide coordinates are the exact compiled 4-node grid layout.
    assert model.nq == 12 and model.nv == 12
    data.qpos[6] = -.1
    data.qpos[7] = .001
    data.qpos[9] = -.1
    data.qpos[10] = .001
  # The detector consumes float32 MPS pose inputs. Seed the CPU oracle from
  # that same state representation before it computes flex vertex positions.
  data.qpos[:] = data.qpos.astype(np.float32).astype(np.float64)
  data.qvel[:] = data.qvel.astype(np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  return model, data


def _expected_contacts(model, data, descriptor, which):
  grouped = {}
  for contact in data.contact[:data.ncon]:
    f1, f2 = map(int, contact.flex)
    if which == "cross" and (f1, f2) != (0, 1):
      continue
    if which in ("self", "internal") and (f1, f2) != (0, 0):
      continue
    if which == "internal":
      key = (int(contact.vert[0]), int(contact.elem[1]))
    else:
      key = (f1, int(contact.elem[0]), f2, int(contact.elem[1]))
    grouped.setdefault(key, []).append(contact)
  expected = {}
  for key, contacts in grouped.items():
    if which == "internal":
      vertex, element = key
      candidates = np.flatnonzero(
          (descriptor.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
          & (descriptor.flex1 == 0) & (descriptor.vert1 == vertex)
          & (descriptor.flex2 == 0) & (descriptor.elem2 == element))
    else:
      f1, e1, f2, e2 = key
      candidates = np.flatnonzero(
          (descriptor.kind == _KIND_ELEMENT_PAIR)
          & (descriptor.flex1 == f1) & (descriptor.elem1 == e1)
          & (descriptor.flex2 == f2) & (descriptor.elem2 == e2))
    candidates = candidates[np.argsort(descriptor.contact_ordinal[candidates])]
    assert len(candidates) >= len(contacts)
    for slot, contact in zip(candidates, contacts):
      expected[int(slot)] = contact
  return expected


def _dense_efc_jacobian(model, data):
  """Expand pinned efc_J using the model's compiled dense/CSR layout."""
  values = np.asarray(data.efc_J, dtype=np.float64)
  if not mujoco.mj_isSparse(model):
    return values.reshape(int(data.nefc), int(model.nv)).copy()
  result = np.zeros((int(data.nefc), int(model.nv)), dtype=np.float64)
  for row in range(int(data.nefc)):
    start = int(data.efc_J_rowadr[row])
    count = int(data.efc_J_rownnz[row])
    columns = np.asarray(data.efc_J_colind[start:start + count], dtype=np.int64)
    result[row, columns] = values[start:start + count]
  return result


@pytest.mark.parametrize("which", ["cross", "self", "internal"])
def test_flex_capsule_fixture_matches_pinned_contact_identity(which):
  model, data = _fixture(which)
  descriptor = lower_flex_contacts(model)
  expected = _expected_contacts(model, data, descriptor, which)
  if which == "cross":
    assert descriptor.slot_count == 16
    assert len(expected) == 6
  elif which == "self":
    assert descriptor.slot_count == 4
    assert len(expected) == 2
    assert set((int(descriptor.elem1[s]), int(descriptor.elem2[s]))
               for s in expected) == {(0, 2)}
  else:
    assert descriptor.slot_count == 4
    assert len(expected) == 4
  assert all(int(descriptor.contact_ordinal[s]) < 4 for s in expected)


@pytest.mark.gpu
@pytest.mark.parametrize("which", ["cross", "self", "internal"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="flex/flex capsule stage requires GPU opt-in")
def test_native_flex_capsule_pair_manifold_matches_pinned_cpu(which):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("flex/flex capsule stage requires MPS")
  model, data = _fixture(which)
  descriptor = lower_flex_contacts(model)
  expected = _expected_contacts(model, data, descriptor, which)
  assert expected
  program = FlexContactProgram(model, device="mps")
  # These fixtures are direct order-0 1D kind-6 pairs, or the admitted
  # direct internal vertex/edge route. Keep the raw toggle only as a detector
  # diagnostic fallback if a later fixture adds an unsupported descriptor.
  if which == "internal":
    assert program._narrowphase_admitted
  elif not program._narrowphase_admitted:
    program._narrowphase_admitted = True

  def mps(value):
    return torch.as_tensor(np.asarray(value, np.float32).copy(),
                           dtype=torch.float32, device="mps").contiguous()

  result = program.run_device(
      mps(data.flexvert_xpos[None]),
      mps(np.zeros((1, model.ngeom, 3), np.float32)),
      mps(np.tile(np.array([1, 0, 0, 0], np.float32),
                  (1, model.ngeom, 1))))
  active = result["active"].cpu().numpy()[0]
  assert set(map(int, np.flatnonzero(active))) == set(expected)
  distance = result["dist"].cpu().numpy()[0]
  position = result["pos"].cpu().numpy()[0]
  normal = result["normal"].cpu().numpy()[0]
  status = result["narrowphase_status"].cpu().numpy()[0]
  for slot, contact in expected.items():
    assert status[slot] == 0
    np.testing.assert_allclose(
        distance[slot], float(contact.dist), rtol=0, atol=6e-6)
    np.testing.assert_allclose(
        position[slot], contact.pos, rtol=0, atol=6e-6)
    np.testing.assert_allclose(
        normal[slot], contact.frame[:3], rtol=0, atol=6e-5)


@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_cpu_internal_capsule_route_has_public_admission_and_rows(jacobian):
  torch = pytest.importorskip("torch")
  model, data = _fixture("internal", jacobian=jacobian)
  from mujoco_metal.flex_contact import FlexContactProgram
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  assert program._narrowphase_admitted
  assert bool(mujoco.mj_isSparse(model)) == (jacobian == "sparse")
  desc = lower_flex_contacts(model)
  assert desc.slot_count == 4
  assert np.all(desc.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
  assert np.all(desc.row_span == 4)
  expected = _expected_contacts(model, data, desc, "internal")
  assert len(expected) == 4
  # The pinned CPU fixture has real penetration and nonzero constraint force;
  # public admission is useful only if the canonical row span can represent it.
  assert data.ncon == 4 and np.linalg.norm(data.qfrc_constraint) > 1.0
  jacobian = _dense_efc_jacobian(model, data)
  for slot, contact in expected.items():
    assert int(desc.vert1[slot]) == int(contact.vert[0])
    assert int(desc.elem2[slot]) == int(contact.elem[1])
    assert int(desc.row_start[slot]) >= 0
    assert int(desc.row_start[slot] + desc.row_span[slot]) <= desc.row_capacity
    vertex = int(contact.vert[0])
    element = int(contact.elem[1])
    flex_adr = int(model.flex_elemadr[0]) + 2 * element
    edge_nodes = np.asarray(model.flex_elem[flex_adr:flex_adr + 2], np.int64)
    vertex_body = int(model.flex_vertbodyid[vertex])
    vertex_start = int(model.body_dofadr[vertex_body])
    vertex_dofs = np.arange(vertex_start, vertex_start + 3)
    edge_dofs = []
    for node in edge_nodes:
      body = int(model.flex_vertbodyid[int(node)])
      start = int(model.body_dofadr[body])
      edge_dofs.extend(range(start, start + 3))
    normal_row = int(contact.efc_address)
    point_reaction = jacobian[normal_row, vertex_dofs]
    edge_reaction = jacobian[normal_row, edge_dofs].reshape(2, 3).sum(axis=0)
    assert np.linalg.norm(point_reaction) > 1.0
    assert np.linalg.norm(edge_reaction) > 1.0
    np.testing.assert_allclose(point_reaction + edge_reaction, 0,
                               rtol=0, atol=6e-6)


@pytest.mark.parametrize("which", ["cross", "self"])
def test_cpu_direct_kind6_capsule_candidates_remain_publicly_admitted(which):
  pytest.importorskip("torch")
  model, _ = _fixture(which)
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  assert program._narrowphase_admitted


@pytest.mark.gpu
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public internal flex contact requires GPU opt-in")
def test_native_public_internal_capsule_flex_trajectory_B2(jacobian):
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, source = _fixture("internal", jacobian=jacobian)
  qpos = np.stack([np.asarray(source.qpos, np.float32)] * 2)
  qvel = np.stack([np.asarray(source.qvel, np.float32)] * 2)
  # Separate the worlds by a small, source-representable generalized velocity
  # while keeping the same penetrating feature pair in both environments.
  qvel[1, 0] = np.float32(.01)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for env, data in enumerate(refs):
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  # The contact program is lazy. Force the ordinary public assembly route,
  # then verify production canonical rows before taking a step.
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  assert cc is not None
  assert int(cc._jacobian_layout.mode) == (1 if jacobian == "sparse" else 0)
  descriptor = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  bundle = cc._flex_contact_current
  assert bundle is not None
  contact_result = bundle["contact_result"]
  expected_by_world = [
      _expected_contacts(model, data, descriptor, "internal")
      for data in refs]
  assert all(len(expected) == 4 for expected in expected_by_world)
  active = contact_result["active"].detach().cpu().numpy()
  expected_active = np.zeros_like(active)
  for env, expected in enumerate(expected_by_world):
    expected_active[env, list(expected)] = 1
  np.testing.assert_array_equal(active, expected_active)
  native_dist = contact_result["dist"].detach().cpu().numpy()
  native_pos = contact_result["pos"].detach().cpu().numpy()
  native_normal = contact_result["normal"].detach().cpu().numpy()
  native_frame = contact_result["frame"].detach().cpu().numpy()
  dense_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = initial_system["R"].detach().cpu().numpy().copy()
  native_ar = initial_system["ar"].detach().cpu().numpy().copy()
  errors = []
  for env, data in enumerate(refs):
    expected = expected_by_world[env]
    cpu_J = _dense_efc_jacobian(model, data)
    for slot, contact in expected.items():
      row = flex_base + int(descriptor.row_start[slot])
      span = int(descriptor.row_span[slot])
      # MuJoCo contact.dim is the contact cone dimension (3 here), while the
      # pyramid solver has a fixed four-row canonical slot.
      assert span == 4 and int(contact.dim) == 3
      address = int(contact.efc_address)
      np.testing.assert_allclose(
          native_dist[env, slot], float(contact.dist), rtol=0, atol=6e-6)
      np.testing.assert_allclose(
          native_pos[env, slot], contact.pos, rtol=0, atol=6e-6)
      np.testing.assert_allclose(
          native_normal[env, slot], contact.frame[:3], rtol=0, atol=6e-5)
      # The cone-pyramid Jacobian uses all three frame axes. Matching only the
      # normal can hide a tangent-axis mismatch that permutes/signs canonical
      # rows, so check the whole frame before diagnosing the row coefficients.
      np.testing.assert_allclose(
          native_frame[env, slot].reshape(3, 3),
          np.asarray(contact.frame).reshape(3, 3), rtol=0, atol=6e-5)
      errors.append((
          dense_j[env, row:row + span] - cpu_J[address:address + span],
          native_R[env, row:row + span]
          - np.asarray(data.efc_R[address:address + span], np.float32),
          native_ar[env, row:row + span]
          - np.asarray(data.efc_aref[address:address + span], np.float32),
      ))
  for dJ, dR, dar in errors:
    np.testing.assert_allclose(dJ, 0, rtol=0, atol=6e-6)
    np.testing.assert_allclose(dR, 0, rtol=0, atol=6e-6)
    np.testing.assert_allclose(dar, 0, rtol=0, atol=6e-5)

  initial_snapshot = sim.snapshot()
  first_status = sim.step()
  np.testing.assert_array_equal(first_status.detach().cpu().numpy(), [0, 0])
  first_system = sim.accepted_step["system"]
  first_force = first_system["qfrc_constraint"].detach().cpu().numpy().copy()
  cpu_force = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32).copy())
  cpu_force = np.stack(cpu_force)
  np.testing.assert_allclose(first_force, cpu_force, rtol=1e-4, atol=2e-5)
  assert np.linalg.norm(cpu_force) > 1.0
  checkpoint = sim.snapshot()

  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  final = sim.state.snapshot()
  expected_final = [(d.qpos.copy(), d.qvel.copy()) for d in refs]
  for env, (expected_qpos, expected_qvel) in enumerate(expected_final):
    np.testing.assert_allclose(final.qpos[env], expected_qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(final.qvel[env], expected_qvel,
                               rtol=2e-4, atol=2e-5)

  # Full simulation snapshots include solver warmstarts and held state. Replay
  # from the one-step checkpoint must reproduce the same three-step endpoint.
  sim.restore(checkpoint)
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, final.qpos)
  np.testing.assert_array_equal(replay.qvel, final.qvel)

  # A cold reset invalidates the replay cache and reproduces the full path.
  sim.restore(initial_snapshot)
  sim.reset(qpos=qpos, qvel=qvel)
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  reset_final = sim.state.snapshot()
  np.testing.assert_array_equal(reset_final.qpos, final.qpos)
  np.testing.assert_array_equal(reset_final.qvel, final.qvel)
