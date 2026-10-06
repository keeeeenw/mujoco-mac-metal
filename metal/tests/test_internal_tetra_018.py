# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Direct tetrahedral internal opposite-vertex/face contact oracle."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    FlexContactProgram,
    _KIND_INTERNAL_VERTEX_ELEMENT,
    lower_flex_contacts,
)


def _fixture(jacobian="dense"):
  xml = """<mujoco><option gravity="0 0 0" timestep=".001"/>
    <worldbody>
      <flexcomp name="volume" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3">
        <contact contype="1" conaffinity="1" selfcollide="none"
                 internal="true" condim="1" friction="0 0 0"/>
        <elasticity young="100" poisson=".2"/>
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.jacobian = {
      "dense": mujoco.mjtJacobian.mjJAC_DENSE,
      "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
  }[jacobian]
  source = mujoco.MjData(model)
  mujoco.mj_forward(model, source)
  rest = np.asarray(source.flexvert_xpos, dtype=np.float64).reshape(-1, 3).copy()
  target = rest.copy()
  target[:, 2] *= .05
  source.qpos[:] = (target - rest).reshape(-1)
  mujoco.mj_forward(model, source)
  assert source.ncon == 12
  qpos = np.asarray(source.qpos, np.float32).copy()
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  qvel[1, 0] = np.float32(.001)
  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    refs.append(data)
  return model, qpos, qvel, refs


def _surface_fixture(jacobian="dense"):
  xml = """<mujoco><option gravity="0 0 0" timestep=".001"/>
    <worldbody>
      <flexcomp name="surface" type="grid" count="2 2 1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"
                 internal="true" condim="1" friction="0 0 0"/>
        <elasticity young="100" poisson=".2"/>
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.jacobian = {
      "dense": mujoco.mjtJacobian.mjJAC_DENSE,
      "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
  }[jacobian]
  qpos = np.asarray([
      .017846379410022012, .11465716052849383, .007321614253856654,
      .046462167626383735, -.004542655551415323, -.013632606493136355,
      -.06235859173803357, .03442410871660426, -.06812297513486143,
      .053246818905074336, .08682296035567282, .029322512578978395,
  ], dtype=np.float32)
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  qvel[1, 0] = np.float32(.001)
  refs = []
  for env in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos.astype(np.float64)
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    refs.append(data)
  assert all(data.ncon == 1 for data in refs)
  return model, qpos, qvel, refs


def _expected(model, data, desc):
  slots = {}
  for contact in data.contact[:data.ncon]:
    elem, vertex = int(contact.elem[0]), int(contact.vert[1])
    match = np.flatnonzero(
        (desc.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
        & (desc.flex1 == 0) & (desc.flex2 == 0)
        & (desc.elem1 == elem) & (desc.vert1 < 0)
        & (desc.elem2 < 0) & (desc.vert2 == vertex)
        & (np.count_nonzero(desc.nodes1 >= 0, axis=1) == 3)
        & (np.count_nonzero(desc.nodes2 >= 0, axis=1) == 1))
    assert match.size == 1
    slots[int(match[0])] = contact
  return slots


def _expected_surface(model, data, desc):
  slots = {}
  for contact in data.contact[:data.ncon]:
    vertex, element = int(contact.vert[0]), int(contact.elem[1])
    match = np.flatnonzero(
        (desc.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
        & (desc.flex1 == 0) & (desc.flex2 == 0)
        & (desc.vert1 == vertex) & (desc.elem2 == element)
        & (desc.elem1 < 0) & (desc.vert2 < 0)
        & (np.count_nonzero(desc.nodes1 >= 0, axis=1) == 1)
        & (np.count_nonzero(desc.nodes2 >= 0, axis=1) == 3))
    assert match.size == 1
    slots[int(match[0])] = contact
  return slots


def _dense_efc_jacobian(model, data):
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


@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_cpu_tetra_internal_face_descriptor_rows_and_two_sided_J(jacobian):
  pytest.importorskip("torch")
  model, _, _, refs = _fixture(jacobian)
  desc = lower_flex_contacts(model)
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  assert program._narrowphase_admitted
  assert desc.slot_count == 24
  assert np.all(desc.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
  assert np.all(desc.row_span == 1)
  assert mujoco.mj_isSparse(model) == (jacobian == "sparse")

  # Preserve the exact pinned engine_collision_driver.c face winding for
  # each opposite-vertex test.  planeVertex uses signed distance, so merely
  # retaining the same three face vertices in another order changes which
  # side of a tetrahedron is eligible for contact.
  pinned_face_local = {
      3: (0, 1, 2),
      1: (0, 2, 3),
      2: (0, 3, 1),
      0: (1, 3, 2),
  }
  element_data_adr = int(model.flex_elemdataadr[0])
  element_data = np.asarray(model.flex_elem, dtype=np.int64)
  for slot in range(desc.slot_count):
    element = int(desc.elem1[slot])
    opposite_vertex = int(desc.vert2[slot])
    vertices = element_data[element_data_adr + 4 * element:
                            element_data_adr + 4 * element + 4]
    opposite_local = int(np.flatnonzero(vertices == opposite_vertex)[0])
    expected_face = vertices[np.asarray(
        pinned_face_local[opposite_local], dtype=np.int64)]
    np.testing.assert_array_equal(desc.nodes1[slot, :3], expected_face)
    assert int(desc.nodes1[slot, 3]) == -1

  for data in refs:
    contacts = _expected(model, data, desc)
    assert len(contacts) == 12
    assert np.linalg.norm(data.qfrc_constraint) > 1.0
    dense = _dense_efc_jacobian(model, data)
    for slot, contact in contacts.items():
      assert int(desc.elem1[slot]) == int(contact.elem[0])
      assert int(desc.vert2[slot]) == int(contact.vert[1])
      assert np.all(desc.nodes1[slot][desc.nodes1[slot] >= 0] != int(contact.vert[1]))
      row = int(contact.efc_address)
      face_nodes = np.asarray(desc.nodes1[slot], dtype=np.int64)
      face_nodes = face_nodes[face_nodes >= 0]
      vertex_body = int(model.flex_vertbodyid[int(contact.vert[1])])
      vertex_start = int(model.body_dofadr[vertex_body])
      point_dofs = np.arange(vertex_start, vertex_start + 3)
      face_dofs = []
      for node in face_nodes:
        body = int(model.flex_vertbodyid[int(node)])
        start = int(model.body_dofadr[body])
        face_dofs.extend(range(start, start + 3))
      point_J = dense[row, point_dofs]
      face_J = dense[row, face_dofs].reshape(3, 3).sum(axis=0)
      assert np.linalg.norm(point_J) > .9
      assert np.linalg.norm(face_J) > .9
      np.testing.assert_allclose(point_J + face_J, 0, rtol=0, atol=6e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="tetrahedral internal contact requires GPU opt-in")
def test_native_tetra_internal_face_public_rows_force_trajectory_replay(jacobian):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("tetrahedral internal contact requires MPS")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, refs = _fixture(jacobian)
  sim = MetalSimulation(model, batch_size=2, qpos=np.stack([qpos] * 2),
                        qvel=qvel, profile="integrated_euler_v1")
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  assert cc is not None
  assert int(cc._jacobian_layout.mode) == (1 if jacobian == "sparse" else 0)
  desc = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  bundle = cc._flex_contact_current
  assert bundle is not None
  result = bundle["contact_result"]
  active = result["active"].detach().cpu().numpy()
  distance = result["dist"].detach().cpu().numpy()
  position = result["pos"].detach().cpu().numpy()
  normal = result["normal"].detach().cpu().numpy()
  frame = result["frame"].detach().cpu().numpy()
  dense_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = initial_system["R"].detach().cpu().numpy().copy()
  native_ar = initial_system["ar"].detach().cpu().numpy().copy()
  for env, data in enumerate(refs):
    expected = _expected(model, data, desc)
    wanted = np.zeros(desc.slot_count, dtype=active.dtype)
    wanted[list(expected)] = 1
    np.testing.assert_array_equal(active[env], wanted)
    cpu_J = _dense_efc_jacobian(model, data)
    for slot, contact in expected.items():
      row = flex_base + int(desc.row_start[slot])
      span = int(desc.row_span[slot])
      assert span == int(contact.dim) == 1
      source_row = int(contact.efc_address)
      np.testing.assert_allclose(distance[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(position[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(normal[env, slot], contact.frame[:3],
                                 rtol=0, atol=6e-5)
      # Pyramid/elliptic row parity depends on both tangent axes as well as
      # the normal; retain the complete frame check for the internal routes.
      np.testing.assert_allclose(
          frame[env, slot].reshape(3, 3),
          np.asarray(contact.frame).reshape(3, 3), rtol=0, atol=6e-5)
      np.testing.assert_allclose(dense_j[env, row:row + span],
                                 cpu_J[source_row:source_row + span],
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_R[env, row:row + span],
                                 data.efc_R[source_row:source_row + span],
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_ar[env, row:row + span],
                                 data.efc_aref[source_row:source_row + span],
                                 rtol=0, atol=6e-5)

  initial_snapshot = sim.snapshot()
  status = sim.step()
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  first_force = sim.accepted_step["system"]["qfrc_constraint"].detach().cpu().numpy()
  cpu_first_force = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_first_force.append(np.asarray(data.qfrc_constraint, np.float32))
  cpu_first_force = np.stack(cpu_first_force)
  np.testing.assert_allclose(first_force, cpu_first_force,
                             rtol=1e-4, atol=2e-5)
  assert np.linalg.norm(cpu_first_force) > 1.0
  checkpoint = sim.snapshot()
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  final = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(final.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(final.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)
  sim.restore(checkpoint)
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, final.qpos)
  np.testing.assert_array_equal(replay.qvel, final.qvel)
  sim.restore(initial_snapshot)
  sim.reset(qpos=np.stack([qpos] * 2), qvel=qvel)
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  reset_final = sim.state.snapshot()
  np.testing.assert_array_equal(reset_final.qpos, final.qpos)
  np.testing.assert_array_equal(reset_final.qvel, final.qvel)


@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_cpu_internal_triangle_descriptor_rows_and_two_sided_J(jacobian):
  pytest.importorskip("torch")
  model, _, _, refs = _surface_fixture(jacobian)
  desc = lower_flex_contacts(model)
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  assert program._narrowphase_admitted
  assert desc.slot_count == 2
  assert np.all(desc.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
  assert np.all(desc.row_span == 1)
  assert mujoco.mj_isSparse(model) == (jacobian == "sparse")
  for data in refs:
    expected = _expected_surface(model, data, desc)
    assert len(expected) == 1
    assert np.linalg.norm(data.qfrc_constraint) > 1.0
    dense = _dense_efc_jacobian(model, data)
    for slot, contact in expected.items():
      assert int(desc.vert1[slot]) == int(contact.vert[0])
      assert int(desc.elem2[slot]) == int(contact.elem[1])
      row = int(contact.efc_address)
      vertex_body = int(model.flex_vertbodyid[int(contact.vert[0])])
      vertex_start = int(model.body_dofadr[vertex_body])
      point_dofs = np.arange(vertex_start, vertex_start + 3)
      element_adr = int(model.flex_elemadr[0]) + 3 * int(contact.elem[1])
      triangle_nodes = np.asarray(model.flex_elem[element_adr:element_adr + 3],
                                  dtype=np.int64)
      triangle_dofs = []
      for node in triangle_nodes:
        body = int(model.flex_vertbodyid[int(node)])
        start = int(model.body_dofadr[body])
        triangle_dofs.extend(range(start, start + 3))
      point_J = dense[row, point_dofs]
      triangle_J = dense[row, triangle_dofs].reshape(3, 3).sum(axis=0)
      assert np.linalg.norm(point_J) > .9
      assert np.linalg.norm(triangle_J) > .9
      np.testing.assert_allclose(point_J + triangle_J, 0, rtol=0, atol=6e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="2D internal flex contact requires GPU opt-in")
def test_native_internal_triangle_public_rows_force_trajectory_replay_reset(jacobian):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("2D internal flex contact requires MPS")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, refs = _surface_fixture(jacobian)
  sim = MetalSimulation(model, batch_size=2, qpos=np.stack([qpos] * 2),
                        qvel=qvel, profile="integrated_euler_v1")
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  assert cc is not None
  assert int(cc._jacobian_layout.mode) == (1 if jacobian == "sparse" else 0)
  desc = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  bundle = cc._flex_contact_current
  assert bundle is not None
  result = bundle["contact_result"]
  active = result["active"].detach().cpu().numpy()
  distance = result["dist"].detach().cpu().numpy()
  position = result["pos"].detach().cpu().numpy()
  normal = result["normal"].detach().cpu().numpy()
  frame = result["frame"].detach().cpu().numpy()
  dense_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = initial_system["R"].detach().cpu().numpy().copy()
  native_ar = initial_system["ar"].detach().cpu().numpy().copy()
  for env, data in enumerate(refs):
    expected = _expected_surface(model, data, desc)
    wanted = np.zeros(desc.slot_count, dtype=active.dtype)
    wanted[list(expected)] = 1
    np.testing.assert_array_equal(active[env], wanted)
    cpu_J = _dense_efc_jacobian(model, data)
    for slot, contact in expected.items():
      row = flex_base + int(desc.row_start[slot])
      span = int(desc.row_span[slot])
      assert span == int(contact.dim) == 1
      source_row = int(contact.efc_address)
      np.testing.assert_allclose(distance[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(position[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(normal[env, slot], contact.frame[:3],
                                 rtol=0, atol=6e-5)
      # Pyramid/elliptic row parity depends on both tangent axes as well as
      # the normal; retain the complete frame check for the internal routes.
      np.testing.assert_allclose(
          frame[env, slot].reshape(3, 3),
          np.asarray(contact.frame).reshape(3, 3), rtol=0, atol=6e-5)
      np.testing.assert_allclose(dense_j[env, row:row + span],
                                 cpu_J[source_row:source_row + span],
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_R[env, row:row + span],
                                 data.efc_R[source_row:source_row + span],
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_ar[env, row:row + span],
                                 data.efc_aref[source_row:source_row + span],
                                 rtol=0, atol=6e-5)

  initial_snapshot = sim.snapshot()
  status = sim.step()
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  first_force = sim.accepted_step["system"]["qfrc_constraint"].detach().cpu().numpy()
  cpu_first_force = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_first_force.append(np.asarray(data.qfrc_constraint, np.float32))
  cpu_first_force = np.stack(cpu_first_force)
  np.testing.assert_allclose(first_force, cpu_first_force,
                             rtol=1e-4, atol=2e-5)
  assert np.linalg.norm(cpu_first_force) > 1.0
  checkpoint = sim.snapshot()
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  final = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(final.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(final.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)
  sim.restore(checkpoint)
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, final.qpos)
  np.testing.assert_array_equal(replay.qvel, final.qvel)
  sim.restore(initial_snapshot)
  sim.reset(qpos=np.stack([qpos] * 2), qvel=qvel)
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  reset_final = sim.state.snapshot()
  np.testing.assert_array_equal(reset_final.qpos, final.qpos)
  np.testing.assert_array_equal(reset_final.qvel, final.qvel)
