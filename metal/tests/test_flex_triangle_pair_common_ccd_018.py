# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Source-faithful common GJK/EPA path for direct flex element pairs.

The private candidate covers direct order-0 edge, triangle, and tetrahedron
pair combinations. Opt-in MPS selectors check pinned CPU contact witnesses;
public dense/CSR selectors check canonical rows, force, trajectory, replay and
reset before any production merge.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    FlexContactProgram,
    _KIND_ELEMENT_PAIR,
    lower_flex_contacts,
)


def _model_scaled_simulation(model, qpos, qvel):
  """Build the same Euler route with host capacity derived from this model."""
  from mujoco_metal.simulation import (
      MetalSimulation, _model_component_capacity_limits,
      validate_stepping_profile)

  limits = _model_component_capacity_limits(model, batch_size=2)
  profile = validate_stepping_profile(
      model, profile="integrated_scalable_v1", limits=limits)
  return MetalSimulation(
      model, batch_size=2, qpos=qpos, qvel=qvel,
      profile=profile.name, limits=limits)


def _triangle_pair_fixture(jacobian=None, which="cross"):
  def flex(name, z, count="2 2 1", selfcollide="none", radius=".01"):
    return f"""
      <flexcomp name="{name}" type="grid" count="{count}"
                pos="0 0 {z}" spacing=".1 .1 .1" mass="1"
                dim="2" radius="{radius}">
        <contact contype="1" conaffinity="1" selfcollide="{selfcollide}"/>
      </flexcomp>
    """
  if which == "cross":
    flexes = flex("lower", "0") + flex("upper", ".015")
  elif which == "self":
    flexes = flex("sheet", "0", count="4 2 1", selfcollide="narrow",
                  radius=".005")
  else:
    raise ValueError(f"unknown triangle-pair fixture {which!r}")
  xml = "<mujoco><option gravity='0 0 0'/><worldbody>" + flexes + "</worldbody></mujoco>"
  model = mujoco.MjModel.from_xml_string(xml)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  data = mujoco.MjData(model)
  if which == "self":
    # Fold the rightmost, nonadjacent triangle column over the left column.
    # The two elements share no nodes; all candidate identities remain fixed.
    data.qpos[18] = np.float32(-.3)
    data.qpos[21] = np.float32(-.3)
  # The device receives float32 generalized coordinates and velocities.
  data.qpos[:] = np.asarray(data.qpos, np.float32).astype(np.float64)
  data.qvel[:] = np.asarray(data.qvel, np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  return model, data


def _mixed_edge_triangle_fixture(reverse=False):
  def flex(name, dim, count, z):
    return f"""
      <flexcomp name="{name}" type="grid" count="{count}"
                pos="0 0 {z}" spacing=".1 .1 .1" mass="1"
                dim="{dim}" radius=".01">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
      </flexcomp>
    """
  edge = flex("edge", 1, "2 1 1", "0")
  triangle = flex("triangle", 2, "2 2 1", ".015")
  flexes = triangle + edge if reverse else edge + triangle
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option gravity='0 0 0'/><worldbody>" + flexes
      + "</worldbody></mujoco>")
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(data.qpos, np.float32).astype(np.float64)
  data.qvel[:] = np.asarray(data.qvel, np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  return model, data


def _direct_simplex_pair_fixture(dim1, dim2, jacobian=None):
  counts = {1: "2 1 1", 2: "2 2 1", 3: "2 2 2"}
  def flex(name, dim, z):
    return f"""
      <flexcomp name="{name}" type="grid" count="{counts[dim]}"
                pos="0 0 {z}" spacing=".1 .1 .1" mass="1"
                dim="{dim}" radius=".01">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
      </flexcomp>
    """
  xml = ("<mujoco><option gravity='0 0 0'/><worldbody>"
         + flex("lower", dim1, "0") + flex("upper", dim2, ".015")
         + "</worldbody></mujoco>")
  model = mujoco.MjModel.from_xml_string(xml)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(data.qpos, np.float32).astype(np.float64)
  data.qvel[:] = np.asarray(data.qvel, np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  return model, data


def _tetra_pair_fixture(jacobian=None):
  return _direct_simplex_pair_fixture(3, 3, jacobian=jacobian)


def _dense_efc_jacobian(model, data):
  """Expand pinned dense/CSR efc_J using the model's compiled layout."""
  values = np.asarray(data.efc_J, dtype=np.float64)
  if not mujoco.mj_isSparse(model):
    return values.reshape(int(data.nefc), int(model.nv)).copy()
  result = np.zeros((int(data.nefc), int(model.nv)), dtype=np.float64)
  for row in range(int(data.nefc)):
    start = int(data.efc_J_rowadr[row])
    count = int(data.efc_J_rownnz[row])
    columns = np.asarray(data.efc_J_colind[start:start + count],
                         dtype=np.int64)
    result[row, columns] = values[start:start + count]
  return result


def _tetra_pair_fixture(jacobian=None):
  def flex(name, z):
    return f"""
      <flexcomp name="{name}" type="grid" count="2 2 2"
                pos="0 0 {z}" spacing=".1 .1 .1" mass="1"
                dim="3" radius=".01">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
      </flexcomp>
    """
  xml = ("<mujoco><option gravity='0 0 0'/><worldbody>"
         + flex("lower", "0") + flex("upper", ".015")
         + "</worldbody></mujoco>")
  model = mujoco.MjModel.from_xml_string(xml)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(data.qpos, np.float32).astype(np.float64)
  data.qvel[:] = np.asarray(data.qvel, np.float32).astype(np.float64)
  mujoco.mj_forward(model, data)
  return model, data


def _dense_efc_jacobian(model, data):
  """Expand the pinned dense or compiled-CSR efc_J representation."""
  values = np.asarray(data.efc_J, dtype=np.float64)
  if not mujoco.mj_isSparse(model):
    return values.reshape(int(data.nefc), int(model.nv)).copy()
  result = np.zeros((int(data.nefc), int(model.nv)), dtype=np.float64)
  for row in range(int(data.nefc)):
    start = int(data.efc_J_rowadr[row])
    count = int(data.efc_J_rownnz[row])
    columns = np.asarray(data.efc_J_colind[start:start + count],
                         dtype=np.int64)
    result[row, columns] = values[start:start + count]
  return result


def _expected_pair_contacts(model, data, descriptor):
  expected = {}
  for contact in data.contact[:data.ncon]:
    expected_flex = (0, 1) if int(model.nflex) == 2 else (0, 0)
    if tuple(map(int, contact.flex)) != expected_flex:
      continue
    pair = tuple(map(int, contact.elem))
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == expected_flex[0]) & (descriptor.elem1 == pair[0])
        & (descriptor.flex2 == expected_flex[1]) & (descriptor.elem2 == pair[1]))
    slots = slots[np.argsort(descriptor.contact_ordinal[slots])]
    assert len(slots) >= 1
    # Pinned mjc_ConvexElem is called with max_contacts=1 for each element
    # pair, so its single witness maps to the fixed canonical slot ordinal 0.
    expected[int(slots[0])] = contact
  return expected


@pytest.mark.parametrize("which", ["cross", "self"])
def test_cpu_triangle_pair_fixture_uses_pinned_one_contact_per_pair_contract(which):
  model, data = _triangle_pair_fixture(which=which)
  descriptor = lower_flex_contacts(model)
  expected = _expected_pair_contacts(model, data, descriptor)
  assert tuple(map(int, model.flex_dim)) == ((2, 2) if which == "cross" else (2,))
  assert np.all(descriptor.kind == _KIND_ELEMENT_PAIR)
  assert len(expected) == data.ncon == (4 if which == "cross" else 5)
  for slot, contact in expected.items():
    assert int(descriptor.flex1[slot]) == 0
    assert int(descriptor.flex2[slot]) == (1 if which == "cross" else 0)
    assert int(descriptor.contact_ordinal[slot]) == 0
    assert int(descriptor.row_start[slot]) >= 0
    assert int(descriptor.row_start[slot] + descriptor.row_span[slot]) <= descriptor.row_capacity
    assert float(contact.dist) < -4.9e-3
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


def test_cpu_direct_tetra_pair_fixture_has_fixed_pair_slots_and_force():
  model, data = _tetra_pair_fixture()
  descriptor = lower_flex_contacts(model)
  expected = _expected_pair_contacts(model, data, descriptor)
  assert tuple(map(int, model.flex_dim)) == (3, 3)
  assert len(expected) == data.ncon == 36
  assert descriptor.slot_count == 36
  assert np.all(descriptor.kind == _KIND_ELEMENT_PAIR)
  assert np.all(descriptor.contact_ordinal == 0)
  assert np.linalg.norm(data.qfrc_constraint) > 1.0
  lifted_qpos = np.asarray(data.qpos, np.float32).copy()
  start = int(model.flex_vertadr[1])
  count = int(model.flex_vertnum[1])
  lifted_qpos[3 * start + 2:3 * (start + count):3] += np.float32(.3)
  lifted = mujoco.MjData(model)
  lifted.qpos[:] = lifted_qpos.astype(np.float64)
  mujoco.mj_forward(model, lifted)
  assert lifted.ncon == 0
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


@pytest.mark.parametrize("dim1,dim2,count", [
    (1, 3, 6), (3, 1, 6), (2, 3, 12), (3, 2, 12)])
def test_cpu_direct_mixed_tetra_pair_fixtures_preserve_each_pair_slot(
    dim1, dim2, count):
  model, data = _direct_simplex_pair_fixture(dim1, dim2)
  descriptor = lower_flex_contacts(model)
  expected = _expected_pair_contacts(model, data, descriptor)
  assert tuple(map(int, model.flex_dim)) == (dim1, dim2)
  assert len(expected) == data.ncon == count
  assert descriptor.slot_count == count
  assert np.all(descriptor.contact_ordinal == 0)
  assert np.all(descriptor.kind == _KIND_ELEMENT_PAIR)
  assert all(np.count_nonzero(descriptor.nodes1[slot] >= 0) == dim1 + 1
             for slot in expected)
  assert all(np.count_nonzero(descriptor.nodes2[slot] >= 0) == dim2 + 1
             for slot in expected)
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="direct tetra pair CCD requires GPU opt-in")
def test_native_direct_tetra_pair_common_ccd_matches_pinned_B2():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("direct tetra pair CCD requires MPS")
  model, data = _tetra_pair_fixture()
  descriptor = lower_flex_contacts(model)
  expected = _expected_pair_contacts(model, data, descriptor)
  assert len(expected) == 36
  program = FlexContactProgram(model, batch_size=2, device="mps")
  assert program._has_native_flex_pair_candidates
  assert program._narrowphase_admitted
  vertices = np.asarray(data.flexvert_xpos, np.float32).copy()
  vertex_batches = np.stack([vertices, vertices.copy()])
  start = int(model.flex_vertadr[1])
  count = int(model.flex_vertnum[1])
  vertex_batches[1, start:start + count, 2] += np.float32(.3)
  def mps(value):
    return torch.as_tensor(np.asarray(value, np.float32).copy(),
                           dtype=torch.float32, device="mps").contiguous()
  result = program.run_device(
      mps(vertex_batches), mps(np.zeros((2, model.ngeom, 3), np.float32)),
      mps(np.tile(np.asarray([1, 0, 0, 0], np.float32),
                  (2, model.ngeom, 1))))
  active = result["active"].cpu().numpy()
  status = result["narrowphase_status"].cpu().numpy()
  distance = result["dist"].cpu().numpy()
  position = result["pos"].cpu().numpy()
  normal = result["normal"].cpu().numpy()
  np.testing.assert_array_equal(np.flatnonzero(active[0]),
                                np.asarray(sorted(expected)))
  assert not np.any(active[1])
  for slot, contact in expected.items():
    assert status[0, slot] == 0
    np.testing.assert_allclose(distance[0, slot], contact.dist,
                               rtol=0, atol=6e-6)
    np.testing.assert_allclose(position[0, slot], contact.pos,
                               rtol=0, atol=6e-6)
    np.testing.assert_allclose(normal[0, slot], contact.frame[:3],
                               rtol=0, atol=6e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("dim1,dim2,count", [
    (1, 3, 6), (3, 1, 6), (2, 3, 12), (3, 2, 12)])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="mixed-order direct flex CCD requires GPU opt-in")
def test_native_direct_mixed_simplex_pair_common_ccd_matches_pinned_B2(
    dim1, dim2, count):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("mixed-order direct flex CCD requires MPS")
  model, data = _direct_simplex_pair_fixture(dim1, dim2)
  descriptor = lower_flex_contacts(model)
  expected = _expected_pair_contacts(model, data, descriptor)
  assert len(expected) == count
  program = FlexContactProgram(model, batch_size=2, device="mps")
  assert program._has_native_flex_pair_candidates
  assert program._narrowphase_admitted
  vertices = np.asarray(data.flexvert_xpos, np.float32).copy()
  vertex_batches = np.stack([vertices, vertices.copy()])
  start = int(model.flex_vertadr[1])
  nvert = int(model.flex_vertnum[1])
  vertex_batches[1, start:start + nvert, 2] += np.float32(.3)
  def mps(value):
    return torch.as_tensor(np.asarray(value, np.float32).copy(),
                           dtype=torch.float32, device="mps").contiguous()
  result = program.run_device(
      mps(vertex_batches), mps(np.zeros((2, model.ngeom, 3), np.float32)),
      mps(np.tile(np.asarray([1, 0, 0, 0], np.float32),
                  (2, model.ngeom, 1))))
  active = result["active"].cpu().numpy()
  status = result["narrowphase_status"].cpu().numpy()
  distance = result["dist"].cpu().numpy()
  position = result["pos"].cpu().numpy()
  normal = result["normal"].cpu().numpy()
  np.testing.assert_array_equal(np.flatnonzero(active[0]),
                                np.asarray(sorted(expected)))
  assert not np.any(active[1])
  for slot, contact in expected.items():
    assert status[0, slot] == 0
    np.testing.assert_allclose(distance[0, slot], contact.dist,
                               rtol=0, atol=6e-6)
    np.testing.assert_allclose(position[0, slot], contact.pos,
                               rtol=0, atol=6e-6)
    np.testing.assert_allclose(normal[0, slot], contact.frame[:3],
                               rtol=0, atol=6e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("dim1,dim2,expected_count", [
    (1, 3, 6), (3, 1, 6), (2, 3, 12), (3, 2, 12), (3, 3, 36)])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public mixed simplex pair requires GPU opt-in")
def test_native_public_direct_simplex_pair_B2_rows_force_replay_reset(
    dim1, dim2, expected_count, jacobian):
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation
  model, source = _direct_simplex_pair_fixture(dim1, dim2,
                                                jacobian=jacobian)
  qpos = np.stack([np.asarray(source.qpos, np.float32)] * 2)
  qvel = np.stack([np.asarray(source.qvel, np.float32)] * 2)
  qvel[1, 0] = np.float32(.01)
  start = int(model.flex_vertadr[1])
  count = int(model.flex_vertnum[1])
  qpos[1, 3 * start + 2:3 * (start + count):3] += np.float32(.3)
  refs = []
  for env in range(2):
    current = mujoco.MjData(model)
    current.qpos[:] = qpos[env].astype(np.float64)
    current.qvel[:] = qvel[env].astype(np.float64)
    mujoco.mj_forward(model, current)
    refs.append(current)
  assert refs[0].ncon == expected_count and refs[1].ncon == 0

  sim = _model_scaled_simulation(model, qpos, qvel)
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  result = cc._flex_contact_current["contact_result"]
  descriptor = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  expected = [_expected_pair_contacts(model, data, descriptor)
              for data in refs]
  assert [len(x) for x in expected] == [expected_count, 0]
  active = result["active"].detach().cpu().numpy()
  expected_active = np.zeros_like(active)
  expected_active[0, list(expected[0])] = 1
  np.testing.assert_array_equal(active, expected_active)
  native_dist = result["dist"].detach().cpu().numpy()
  native_pos = result["pos"].detach().cpu().numpy()
  native_frame = result["frame"].detach().cpu().numpy()
  native_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = initial_system["R"].detach().cpu().numpy().copy()
  native_ar = initial_system["ar"].detach().cpu().numpy().copy()
  for env, data in enumerate(refs):
    cpu_j = _dense_efc_jacobian(model, data)
    for slot, contact in expected[env].items():
      row = flex_base + int(descriptor.row_start[slot])
      span = int(descriptor.row_span[slot])
      address = int(contact.efc_address)
      np.testing.assert_allclose(native_dist[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_pos[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_frame[env, slot].reshape(3, 3),
                                 np.asarray(contact.frame).reshape(3, 3),
                                 rtol=0, atol=6e-5)
      np.testing.assert_allclose(native_j[env, row:row + span],
                                 cpu_j[address:address + span],
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_R[env, row:row + span],
                                 np.asarray(data.efc_R[address:address + span],
                                            np.float32), rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_ar[env, row:row + span],
                                 np.asarray(data.efc_aref[address:address + span],
                                            np.float32), rtol=0, atol=6e-5)
  initial = sim.snapshot()
  np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  native_force = sim.accepted_step["system"]["qfrc_constraint"].detach().cpu().numpy()
  cpu_force = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
  cpu_force = np.stack(cpu_force)
  assert np.linalg.norm(cpu_force[0]) > 1.0
  np.testing.assert_allclose(native_force, cpu_force, rtol=1e-4, atol=2e-5)
  checkpoint = sim.snapshot()
  for _ in range(3):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  endpoint = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(endpoint.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(endpoint.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)
  sim.restore(checkpoint)
  for _ in range(3):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, endpoint.qpos)
  np.testing.assert_array_equal(replay.qvel, endpoint.qvel)
  sim.restore(initial)
  sim.reset(qpos=qpos, qvel=qvel)
  for _ in range(4):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  reset = sim.state.snapshot()
  np.testing.assert_array_equal(reset.qpos, endpoint.qpos)
  np.testing.assert_array_equal(reset.qvel, endpoint.qvel)
  assert program._has_native_flex_pair_candidates


def test_scaled_simulation_passes_profile_name_at_constructor_boundary(monkeypatch):
  """The validated profile object is lowered to its public string API."""
  from types import SimpleNamespace
  import mujoco_metal.simulation as simulation

  seen = {}

  class StubSimulation:
    def __init__(self, model, **kwargs):
      seen.update(kwargs)

  monkeypatch.setattr(simulation, "MetalSimulation", StubSimulation)
  monkeypatch.setattr(
      simulation, "_model_component_capacity_limits",
      lambda model, batch_size: {"batch_size": batch_size})
  monkeypatch.setattr(
      simulation, "validate_stepping_profile",
      lambda model, profile, limits: SimpleNamespace(name=profile))
  model, _ = _triangle_pair_fixture()
  _model_scaled_simulation(model, np.zeros(model.nq), np.zeros(model.nv))
  assert seen["profile"] == "integrated_scalable_v1"
  assert isinstance(seen["profile"], str)


@pytest.mark.parametrize("reverse", [False, True])
def test_cpu_mixed_edge_triangle_fixture_uses_one_source_contact_per_pair(reverse):
  model, data = _mixed_edge_triangle_fixture(reverse=reverse)
  descriptor = lower_flex_contacts(model)
  expected = [c for c in data.contact[:data.ncon]
              if int(c.flex[0]) != int(c.flex[1])]
  assert tuple(map(int, model.flex_dim)) == ((2, 1) if reverse else (1, 2))
  assert len(expected) == data.ncon == 2
  assert np.all(descriptor.kind == _KIND_ELEMENT_PAIR)
  assert all(int(np.count_nonzero(descriptor.nodes1[i] >= 0)) ==
             (3 if reverse else 2) for i in range(len(descriptor.kind)))
  assert all(int(np.count_nonzero(descriptor.nodes2[i] >= 0)) ==
             (2 if reverse else 3) for i in range(len(descriptor.kind)))
  for contact in expected:
    assert int(contact.flex[0]) != int(contact.flex[1])
    assert float(contact.dist) < -4.9e-3
  assert np.linalg.norm(data.qfrc_constraint) > 1.0
  lifted_qpos = np.asarray(data.qpos, np.float32).copy()
  triangle = 0 if reverse else 1
  first = int(model.flex_vertadr[triangle])
  count = int(model.flex_vertnum[triangle])
  lifted_qpos[3 * first + 2:3 * (first + count):3] += np.float32(.1)
  lifted = mujoco.MjData(model)
  lifted.qpos[:] = lifted_qpos.astype(np.float64)
  mujoco.mj_forward(model, lifted)
  assert lifted.ncon == 0
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted


@pytest.mark.gpu
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="mixed edge/triangle CCD requires GPU opt-in")
def test_native_mixed_edge_triangle_common_ccd_matches_pinned_B2(reverse):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("mixed edge/triangle CCD requires MPS")
  model, data = _mixed_edge_triangle_fixture(reverse=reverse)
  descriptor = lower_flex_contacts(model)
  program = FlexContactProgram(model, batch_size=2, device="mps")
  assert program._has_native_flex_pair_candidates
  assert program._narrowphase_admitted

  vertices = np.asarray(data.flexvert_xpos, np.float32).copy()
  vertex_batches = np.stack([vertices, vertices.copy()])
  triangle_flex = 0 if reverse else 1
  start = int(model.flex_vertadr[triangle_flex])
  count = int(model.flex_vertnum[triangle_flex])
  vertex_batches[1, start:start + count, 2] += np.float32(.1)
  def mps(value):
    return torch.as_tensor(np.asarray(value, np.float32).copy(),
                           dtype=torch.float32, device="mps").contiguous()

  result = program.run_device(
      mps(vertex_batches),
      mps(np.zeros((2, model.ngeom, 3), np.float32)),
      mps(np.tile(np.asarray([1, 0, 0, 0], np.float32),
                  (2, model.ngeom, 1))))
  actual = result["active"].cpu().numpy()
  status = result["narrowphase_status"].cpu().numpy()
  distance = result["dist"].cpu().numpy()
  position = result["pos"].cpu().numpy()
  normal = result["normal"].cpu().numpy()
  expected_slots = set(range(len(descriptor.kind)))
  np.testing.assert_array_equal(np.flatnonzero(actual[0]),
                                np.asarray(sorted(expected_slots)))
  assert not np.any(actual[1])
  for contact_slot in expected_slots:
    assert status[0, contact_slot] == 0
    contact = next(c for c in data.contact[:data.ncon]
                   if tuple(map(int, c.elem)) ==
                   (int(descriptor.elem1[contact_slot]),
                    int(descriptor.elem2[contact_slot])))
    np.testing.assert_allclose(distance[0, contact_slot], contact.dist,
                               rtol=0, atol=6e-6)
    np.testing.assert_allclose(position[0, contact_slot], contact.pos,
                               rtol=0, atol=6e-6)
    np.testing.assert_allclose(normal[0, contact_slot], contact.frame[:3],
                               rtol=0, atol=6e-5)


def test_common_pair_shader_keeps_inline_flex_support_and_source_core():
  shader_dir = Path(__file__).parents[1] / "mujoco_metal" / "shaders"
  common = (shader_dir / "common_native_ccd_support.metal").read_text()
  pair = (shader_dir / "flex_common_pair.metal").read_text()
  host = (Path(__file__).parents[1] / "mujoco_metal" / "flex_contact.py").read_text()
  assert "FlexDD3 flex_vertices[4]" in common
  assert "object.flex_vertices[i]" in common
  assert "common_ccd_source_gjk(" in pair
  assert "common_ccd_source_epa(" in pair
  assert "flex_contact_detect_common_element_pair" in host
  assert "_narrowphase_admitted" in host


@pytest.mark.gpu
@pytest.mark.parametrize("which", ["cross", "self"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="triangle-pair common CCD requires GPU opt-in")
def test_native_triangle_pair_common_ccd_detector_matches_pinned_B2(which):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("triangle-pair common CCD requires MPS")
  model, data = _triangle_pair_fixture(which=which)
  descriptor = lower_flex_contacts(model)
  expected = _expected_pair_contacts(model, data, descriptor)
  assert len(expected) == (4 if which == "cross" else 5)
  program = FlexContactProgram(model, batch_size=2, device="mps")
  assert program._has_native_flex_pair_candidates
  assert program._narrowphase_admitted

  def mps(value):
    return torch.as_tensor(np.asarray(value, np.float32).copy(),
                           dtype=torch.float32, device="mps").contiguous()

  qpos = np.asarray(data.qpos, np.float32)
  qvel = np.asarray(data.qvel, np.float32)
  reference = []
  for env in range(2):
    current = mujoco.MjData(model)
    current.qpos[:] = qpos.astype(np.float64)
    current.qvel[:] = qvel.astype(np.float64)
    if env and which == "cross":
      # Move the upper flex beyond contact in one world; its fixed pair slots
      # must remain inactive while world zero retains the four contacts.
      for qadr in range(14, 24, 3):
        current.qpos[qadr] = np.float32(current.qpos[qadr] + .1)
    elif env and which == "self":
      # The undeformed sheet has no self-collision; only world zero has the
      # folded nonadjacent triangle pair.
      current.qpos[:] = np.asarray(data.qpos, np.float64)
      current.qpos[18:24] = 0.0
    mujoco.mj_forward(model, current)
    reference.append(current)
  result = program.run_device(
      mps(np.stack([np.asarray(x.flexvert_xpos, np.float32)
                    for x in reference])),
      mps(np.zeros((2, model.ngeom, 3), np.float32)),
      mps(np.tile(np.asarray([1, 0, 0, 0], np.float32),
                  (2, model.ngeom, 1))))
  actual = result["active"].cpu().numpy()
  distance = result["dist"].cpu().numpy()
  position = result["pos"].cpu().numpy()
  normal = result["normal"].cpu().numpy()
  status = result["narrowphase_status"].cpu().numpy()
  for env, current in enumerate(reference):
    env_expected = _expected_pair_contacts(model, current, descriptor)
    assert set(map(int, np.flatnonzero(actual[env]))) == set(env_expected)
    for slot, contact in env_expected.items():
      assert status[env, slot] == 0
      np.testing.assert_allclose(distance[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(position[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(normal[env, slot], contact.frame[:3],
                                 rtol=0, atol=6e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("which", ["cross", "self"])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public flex triangle pair requires GPU opt-in")
def test_native_public_triangle_pair_B2_rows_force_replay_reset(which, jacobian):
  """Full public path required before direct 2D pair admission is accepted."""
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, source = _triangle_pair_fixture(jacobian=jacobian, which=which)
  qpos = np.stack([np.asarray(source.qpos, np.float32)] * 2)
  qvel = np.stack([np.asarray(source.qvel, np.float32)] * 2)
  qvel[1, 0] = np.float32(.01)
  refs = []
  for env in range(2):
    current = mujoco.MjData(model)
    current.qpos[:] = qpos[env].astype(np.float64)
    current.qvel[:] = qvel[env].astype(np.float64)
    mujoco.mj_forward(model, current)
    refs.append(current)

  sim = _model_scaled_simulation(model, qpos, qvel)
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates
  cc = sim._coupled_constraints
  descriptor = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  bundle = cc._flex_contact_current
  contact_result = bundle["contact_result"]
  expected_by_world = [
      _expected_pair_contacts(model, data, descriptor) for data in refs]
  expected_count = 4 if which == "cross" else 5
  assert all(len(expected) == expected_count for expected in expected_by_world)
  active = contact_result["active"].detach().cpu().numpy()
  expected_active = np.zeros_like(active)
  for env, expected in enumerate(expected_by_world):
    expected_active[env, list(expected)] = 1
  np.testing.assert_array_equal(active, expected_active)

  native_dist = contact_result["dist"].detach().cpu().numpy()
  native_pos = contact_result["pos"].detach().cpu().numpy()
  native_frame = contact_result["frame"].detach().cpu().numpy()
  dense_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = initial_system["R"].detach().cpu().numpy().copy()
  native_ar = initial_system["ar"].detach().cpu().numpy().copy()
  for env, data in enumerate(refs):
    cpu_j = _dense_efc_jacobian(model, data)
    for slot, contact in expected_by_world[env].items():
      row = flex_base + int(descriptor.row_start[slot])
      span = int(descriptor.row_span[slot])
      address = int(contact.efc_address)
      assert span == 4 and int(contact.dim) == 3
      np.testing.assert_allclose(native_dist[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_pos[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_frame[env, slot].reshape(3, 3),
                                 np.asarray(contact.frame).reshape(3, 3),
                                 rtol=0, atol=6e-5)
      np.testing.assert_allclose(dense_j[env, row:row + span],
                                 cpu_j[address:address + span],
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_R[env, row:row + span],
                                 np.asarray(data.efc_R[address:address + span],
                                            np.float32),
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_ar[env, row:row + span],
                                 np.asarray(data.efc_aref[address:address + span],
                                            np.float32),
                                 rtol=0, atol=6e-5)

  initial_snapshot = sim.snapshot()
  status = sim.step()
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  native_force = sim.accepted_step["system"]["qfrc_constraint"].detach().cpu().numpy()
  cpu_force = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
  cpu_force = np.stack(cpu_force)
  assert np.linalg.norm(cpu_force) > 1.0
  np.testing.assert_allclose(native_force, cpu_force, rtol=1e-4, atol=2e-5)
  checkpoint = sim.snapshot()
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  endpoint = sim.state.snapshot()
  expected_endpoint = [(data.qpos.copy(), data.qvel.copy()) for data in refs]
  for env, (qpos_ref, qvel_ref) in enumerate(expected_endpoint):
    np.testing.assert_allclose(endpoint.qpos[env], qpos_ref, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(endpoint.qvel[env], qvel_ref, rtol=2e-4, atol=2e-5)
  sim.restore(checkpoint)
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, endpoint.qpos)
  np.testing.assert_array_equal(replay.qvel, endpoint.qvel)
  sim.restore(initial_snapshot)
  sim.reset(qpos=qpos, qvel=qvel)
  for _ in range(4):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  reset = sim.state.snapshot()
  np.testing.assert_array_equal(reset.qpos, endpoint.qpos)
  np.testing.assert_array_equal(reset.qvel, endpoint.qvel)


def _expected_mixed_pair_contacts(model, data, descriptor):
  expected = {}
  for contact in data.contact[:data.ncon]:
    if int(contact.flex[0]) == int(contact.flex[1]):
      continue
    pair = tuple(map(int, contact.elem))
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == int(contact.flex[0]))
        & (descriptor.elem1 == pair[0])
        & (descriptor.flex2 == int(contact.flex[1]))
        & (descriptor.elem2 == pair[1]))
    assert len(slots) == 1
    expected[int(slots[0])] = contact
  return expected


@pytest.mark.gpu
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="public mixed flex pair requires GPU opt-in")
def test_native_public_mixed_edge_triangle_B2_rows_force_replay_reset(
    reverse, jacobian):
  """Qualify mixed simplex contact through canonical rows and stepping."""
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, source = _mixed_edge_triangle_fixture(reverse=reverse)
  model.opt.jacobian = {
      "dense": mujoco.mjtJacobian.mjJAC_DENSE,
      "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
  }[jacobian]
  descriptor = lower_flex_contacts(model)
  qpos = np.stack([np.asarray(source.qpos, np.float32)] * 2)
  qvel = np.stack([np.asarray(source.qvel, np.float32)] * 2)
  qvel[1, 0] = np.float32(.01)
  triangle = 0 if reverse else 1
  first_node = int(model.flex_vertadr[triangle])
  node_count = int(model.flex_vertnum[triangle])
  # World one lifts only the triangle nodes, so fixed slots remain present but
  # inactive while world zero has both pinned edge/triangle contacts.
  qpos[1, 3 * first_node + 2:3 * (first_node + node_count):3] += np.float32(.1)
  refs = []
  for env in range(2):
    current = mujoco.MjData(model)
    current.qpos[:] = qpos[env].astype(np.float64)
    current.qvel[:] = qvel[env].astype(np.float64)
    mujoco.mj_forward(model, current)
    refs.append(current)
  assert refs[0].ncon == 2 and refs[1].ncon == 0

  sim = _model_scaled_simulation(model, qpos, qvel)
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  bundle = cc._flex_contact_current
  result = bundle["contact_result"]
  desc = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  expected = [_expected_mixed_pair_contacts(model, data, desc) for data in refs]
  assert [len(x) for x in expected] == [2, 0]
  active = result["active"].detach().cpu().numpy()
  expected_active = np.zeros_like(active)
  expected_active[0, list(expected[0])] = 1
  np.testing.assert_array_equal(active, expected_active)

  native_dist = result["dist"].detach().cpu().numpy()
  native_pos = result["pos"].detach().cpu().numpy()
  native_frame = result["frame"].detach().cpu().numpy()
  native_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = initial_system["R"].detach().cpu().numpy().copy()
  native_ar = initial_system["ar"].detach().cpu().numpy().copy()
  for env, data in enumerate(refs):
    cpu_j = _dense_efc_jacobian(model, data)
    for slot, contact in expected[env].items():
      row = flex_base + int(desc.row_start[slot])
      span = int(desc.row_span[slot])
      address = int(contact.efc_address)
      assert span == 4 and int(contact.dim) == 3
      np.testing.assert_allclose(native_dist[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_pos[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_frame[env, slot].reshape(3, 3),
                                 np.asarray(contact.frame).reshape(3, 3),
                                 rtol=0, atol=6e-5)
      np.testing.assert_allclose(native_j[env, row:row + span],
                                 cpu_j[address:address + span],
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_R[env, row:row + span],
                                 np.asarray(data.efc_R[address:address + span],
                                            np.float32), rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_ar[env, row:row + span],
                                 np.asarray(data.efc_aref[address:address + span],
                                            np.float32), rtol=0, atol=6e-5)

  initial_snapshot = sim.snapshot()
  status = sim.step()
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
  native_force = sim.accepted_step["system"]["qfrc_constraint"].detach().cpu().numpy()
  cpu_force = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
  cpu_force = np.stack(cpu_force)
  assert np.linalg.norm(cpu_force[0]) > 1.0
  np.testing.assert_allclose(native_force, cpu_force, rtol=1e-4, atol=2e-5)
  checkpoint = sim.snapshot()
  for _ in range(3):
    status = sim.step()
    np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
    for data in refs:
      mujoco.mj_step(model, data)
  endpoint = sim.state.snapshot()
  for env, data in enumerate(refs):
    np.testing.assert_allclose(endpoint.qpos[env], data.qpos,
                               rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(endpoint.qvel[env], data.qvel,
                               rtol=2e-4, atol=2e-5)
  sim.restore(checkpoint)
  for _ in range(3):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, endpoint.qpos)
  np.testing.assert_array_equal(replay.qvel, endpoint.qvel)
  sim.restore(initial_snapshot)
  sim.reset(qpos=qpos, qvel=qvel)
  for _ in range(4):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  reset = sim.state.snapshot()
  np.testing.assert_array_equal(reset.qpos, endpoint.qpos)
  np.testing.assert_array_equal(reset.qvel, endpoint.qvel)
