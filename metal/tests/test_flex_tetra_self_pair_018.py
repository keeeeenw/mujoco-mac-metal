# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Self-contacting direct tetrahedra through canonical flex rows and stepping.

The CPU fixture is the oracle for the unmodified pinned 3.10 self-collision
route. Native selectors are intentionally opt-in and compare candidate IDs,
contact witnesses, canonical dense/CSR rows, force, trajectory, replay, and
reset at the public Simulation boundary.
"""

import os

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


def _self_tetra_fixture(jacobian=None):
  xml = """<mujoco>
    <option timestep=".001" gravity="0 0 0" jacobian="dense"/>
    <worldbody>
      <flexcomp name="tetra" type="grid" count="3 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" radius=".02">
        <elasticity young="1000" poisson=".2" damping=".1"/>
        <contact contype="1" conaffinity="1" selfcollide="narrow"/>
      </flexcomp>
    </worldbody>
  </mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  source = mujoco.MjData(model)
  mujoco.mj_forward(model, source)
  rest = np.asarray(source.flexvert_xpos, dtype=np.float32).reshape(-1, 3)

  # Express a deterministic folded configuration in generalized coordinates.
  # The distal two x layers collapse onto the proximal layer while y/z remain
  # separated. This makes nonadjacent tetrahedra overlap without changing the
  # compiled element topology or contact configuration.
  qpos_contact = rest.copy()
  qpos_contact[rest[:, 0] >= np.float32(.1), 0] = np.float32(-.1)
  qpos_contact = qpos_contact.reshape(-1)
  qpos = np.zeros((2, int(model.nq)), dtype=np.float32)
  qpos[0] = qpos_contact
  qvel = np.zeros((2, int(model.nv)), dtype=np.float32)
  # Give the healthy, no-contact world a distinct but small velocity so the
  # B2 isolation test also catches accidental cross-world row/status writes.
  qvel[1, 0] = np.float32(.01)

  refs = []
  for env in range(2):
    current = mujoco.MjData(model)
    current.qpos[:] = qpos[env].astype(np.float64)
    current.qvel[:] = qvel[env].astype(np.float64)
    mujoco.mj_forward(model, current)
    refs.append(current)
  return model, qpos, qvel, refs


def _expected_self_contacts(model, data, descriptor):
  expected = {}
  for contact in data.contact[:data.ncon]:
    if tuple(map(int, contact.flex)) != (0, 0):
      continue
    e1, e2 = map(int, contact.elem)
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == 0) & (descriptor.flex2 == 0)
        & (descriptor.elem1 == e1) & (descriptor.elem2 == e2))
    slots = slots[np.argsort(descriptor.contact_ordinal[slots])]
    assert len(slots) == 1
    expected[int(slots[0])] = contact
  return expected


def _dense_efc_jacobian(model, data):
  values = np.asarray(data.efc_J, dtype=np.float64)
  if not mujoco.mj_isSparse(model):
    return values.reshape(int(data.nefc), int(model.nv)).copy()
  result = np.zeros((int(data.nefc), int(model.nv)), dtype=np.float64)
  for row in range(int(data.nefc)):
    start = int(data.efc_J_rowadr[row])
    count = int(data.efc_J_rownnz[row])
    cols = np.asarray(data.efc_J_colind[start:start + count], dtype=np.int64)
    result[row, cols] = values[start:start + count]
  return result


def test_cpu_self_tetra_fixture_uses_nonadjacent_source_element_pairs():
  model, qpos, qvel, refs = _self_tetra_fixture()
  desc = lower_flex_contacts(model)
  assert int(model.flex_dim[0]) == 3
  assert int(model.flex_selfcollide[0]) == int(mujoco.mjtFlexSelf.mjFLEXSELF_NARROW)
  assert not bool(model.flex_internal[0])
  expected = [_expected_self_contacts(model, ref, desc) for ref in refs]
  assert len(expected[0]) == refs[0].ncon == 14
  assert len(expected[1]) == refs[1].ncon == 0
  assert desc.slot_count == 14
  assert np.all(desc.kind == _KIND_ELEMENT_PAIR)
  for slot, contact in expected[0].items():
    assert int(desc.flex1[slot]) == int(desc.flex2[slot]) == 0
    assert int(desc.contact_ordinal[slot]) == 0
    assert int(desc.elem2[slot]) > int(desc.elem1[slot])
    assert float(contact.dist) < -.02
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1.0
  assert np.linalg.norm(refs[0].qfrc_passive) > 1.0
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


@pytest.mark.gpu
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="self tetra pair lifecycle requires GPU opt-in")
def test_native_public_self_tetra_pair_B2_rows_force_replay_reset(jacobian):
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, refs = _self_tetra_fixture(jacobian=jacobian)
  desc = lower_flex_contacts(model)
  expected = [_expected_self_contacts(model, ref, desc) for ref in refs]
  assert [len(x) for x in expected] == [14, 0]
  assert refs[0].ncon == 14 and refs[1].ncon == 0

  sim = _model_scaled_simulation(model, qpos, qvel)
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates
  cc = sim._coupled_constraints
  desc = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  bundle = cc._flex_contact_current
  result = bundle["contact_result"]
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
      assert span == int(contact.dim) + 1
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
  system = sim.accepted_step["system"]
  native_force = system["qfrc_constraint"].detach().cpu().numpy()
  native_passive = sim._flex._qfrc_passive.detach().cpu().numpy()
  cpu_force = []
  cpu_passive = []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
    cpu_passive.append(np.asarray(data.qfrc_passive, np.float32))
  cpu_force = np.stack(cpu_force)
  cpu_passive = np.stack(cpu_passive)
  assert np.linalg.norm(cpu_force[0]) > 1.0
  assert np.linalg.norm(cpu_passive[0]) > 1.0
  np.testing.assert_allclose(native_force, cpu_force, rtol=1e-4, atol=2e-5)
  np.testing.assert_allclose(native_passive, cpu_passive, rtol=1e-4, atol=2e-5)

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
