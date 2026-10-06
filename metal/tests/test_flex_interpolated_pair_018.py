# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Source-derived Q1/Q2 tetra flex-pair candidate and lifecycle fixtures."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    FlexContactProgram,
    _KIND_ELEMENT_PAIR,
    lower_flex_contacts,
)


_INTERP_CASES = (("trilinear", "2 2 2", 22),
                 ("quadratic", "3 3 3", 50))
_INTERP_COUNTS = {dof: count for dof, count, _ in _INTERP_CASES}


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


def _interpolated_pair_fixture(dof="trilinear", jacobian=None,
                              heterogeneous=False):
  count = _INTERP_COUNTS[dof]
  if heterogeneous:
    lower_contact = ('<contact contype="1" conaffinity="1" '
                     'selfcollide="none" condim="4" '
                     'friction=".6 .4 .03" priority="2" '
                     'solref=".02 1" solimp=".8 .9 .01 .5 2"/>')
    upper_contact = ('<contact contype="1" conaffinity="1" '
                     'selfcollide="none" condim="6" '
                     'friction=".8 .5 .04" priority="1" '
                     'solref=".03 2" solimp=".7 .85 .02 .4 3"/>')
    upper_elasticity = ('<elasticity young="500" poisson=".25" '
                        'damping=".2"/>')
  else:
    lower_contact = upper_contact = (
        '<contact contype="1" conaffinity="1" selfcollide="none"/>')
    upper_elasticity = '<elasticity young="1000" poisson=".2" damping=".1"/>'
  xml = f"""<mujoco><option timestep=".001" gravity="0 0 0" jacobian="dense"/>
    <worldbody>
      <flexcomp name="lower" type="grid" count="{count}"
                spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
        {lower_contact}
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
      <flexcomp name="upper" type="grid" count="{count}"
                pos="0 0 .04" spacing=".1 .1 .1" mass="1" dim="3"
                dof="{dof}">
        {upper_contact}
        {upper_elasticity}
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  source = mujoco.MjData(model)
  source.qvel[0] = .01
  mujoco.mj_forward(model, source)
  qpos = np.stack([np.asarray(source.qpos, np.float32)] * 2)
  qvel = np.stack([np.asarray(source.qvel, np.float32)] * 2)
  # The second world lifts only the upper flex by more than the source
  # candidate cutoff, retaining the same compiled element/row capacity.
  second = int(model.flex_vertadr[1])
  number = int(model.flex_vertnum[1])
  qpos[1, 3 * second + 2:3 * (second + number):3] += np.float32(.3)
  qvel[1, 0] = np.float32(0)
  refs = []
  for env in range(2):
    current = mujoco.MjData(model)
    current.qpos[:] = qpos[env].astype(np.float64)
    current.qvel[:] = qvel[env].astype(np.float64)
    mujoco.mj_forward(model, current)
    refs.append(current)
  return model, qpos, qvel, refs


def test_cpu_interpolated_pair_mixes_heterogeneous_contact_materials():
  """Priority, condim, friction, and impedance follow pinned mixing."""
  model, _, _, refs = _interpolated_pair_fixture(heterogeneous=True)
  descriptor = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, ref, descriptor) for ref in refs]
  assert [len(x) for x in expected] == [22, 0]
  assert int(model.flex_priority[0]) > int(model.flex_priority[1])
  assert int(model.flex_condim[0]) == 4
  assert int(model.flex_condim[1]) == 6
  for slot, contact in expected[0].items():
    assert int(contact.dim) == int(descriptor.condim[slot]) == 4
    np.testing.assert_allclose(contact.friction, descriptor.friction[slot],
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(contact.solref, descriptor.solref[slot],
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(contact.solimp, descriptor.solimp[slot],
                               rtol=0, atol=1e-7)
    assert int(descriptor.row_span[slot]) == 2 * (int(contact.dim) - 1)
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


@pytest.mark.parametrize("dof", ["trilinear", "quadratic"])
def test_cpu_pinned_source_rejects_interpolated_self_collision(dof):
  """Do not mistake the general pair kernel for a legal self route."""
  count = _INTERP_COUNTS[dof]
  xml = f"""<mujoco><worldbody>
    <flexcomp name="volume" type="grid" count="{count}"
              spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
      <contact contype="1" conaffinity="1" selfcollide="narrow"/>
      <elasticity young="1000" poisson=".2"/>
    </flexcomp>
  </worldbody></mujoco>"""
  with pytest.raises(ValueError, match="interpolation cannot do self-collision"):
    mujoco.MjModel.from_xml_string(xml)


@pytest.mark.gpu
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="heterogeneous interpolated rows require GPU opt-in")
def test_native_public_interpolated_pair_heterogeneous_material_lifecycle(
    jacobian):
  """Exercise pinned priority/material mixing through the public row path."""
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, refs = _interpolated_pair_fixture(
      jacobian=jacobian, heterogeneous=True)
  descriptor = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, ref, descriptor) for ref in refs]
  assert [len(x) for x in expected] == [22, 0]
  sim = _model_scaled_simulation(model, qpos, qvel)
  system0 = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  result = cc._flex_contact_current["contact_result"]
  native_active = result["active"].detach().cpu().numpy()
  expected_active = np.zeros_like(native_active)
  expected_active[0, list(expected[0])] = 1
  np.testing.assert_array_equal(native_active, expected_active)
  native_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = system0["R"].detach().cpu().numpy().copy()
  native_ar = system0["ar"].detach().cpu().numpy().copy()
  native_dist = result["dist"].detach().cpu().numpy()
  native_pos = result["pos"].detach().cpu().numpy()
  native_frame = result["frame"].detach().cpu().numpy()
  native_condim = result["condim"].detach().cpu().numpy()
  native_friction = result["friction"].detach().cpu().numpy()
  native_solref = result["solref"].detach().cpu().numpy()
  native_solimp = result["solimp"].detach().cpu().numpy()
  desc = cc.descriptor.flex_contact_descriptor
  row_base = int(cc.descriptor.flex_contact_base)
  for env, data in enumerate(refs):
    cpu_j = _dense_efc_jacobian(model, data)
    for slot, contact in expected[env].items():
      assert int(native_condim[slot]) == int(contact.dim) == 4
      np.testing.assert_allclose(native_friction[slot], contact.friction,
                                 rtol=0, atol=1e-7)
      np.testing.assert_allclose(native_solref[slot], contact.solref,
                                 rtol=0, atol=1e-7)
      np.testing.assert_allclose(native_solimp[slot], contact.solimp,
                                 rtol=0, atol=1e-7)
      np.testing.assert_allclose(native_dist[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_pos[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_frame[env, slot].reshape(3, 3),
                                 np.asarray(contact.frame).reshape(3, 3),
                                 rtol=0, atol=6e-5)
      row = row_base + int(desc.row_start[slot])
      span = int(desc.row_span[slot])
      address = int(contact.efc_address)
      assert span == 2 * (int(contact.dim) - 1)
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
  native_passive = sim._flex._qfrc_passive.detach().cpu().numpy()
  cpu_force, cpu_passive = [], []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
    cpu_passive.append(np.asarray(data.qfrc_passive, np.float32))
  cpu_force, cpu_passive = np.stack(cpu_force), np.stack(cpu_passive)
  assert np.linalg.norm(cpu_force[0]) > 1.0
  assert np.linalg.norm(cpu_passive[0]) > 1e-4
  np.testing.assert_allclose(native_force, cpu_force, rtol=1e-4, atol=2e-5)
  np.testing.assert_allclose(native_passive, cpu_passive, rtol=1e-4, atol=2e-5)
  checkpoint = sim.snapshot()
  for _ in range(2):
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
  for _ in range(2):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, endpoint.qpos)
  np.testing.assert_array_equal(replay.qvel, endpoint.qvel)
  sim.restore(initial)
  sim.reset(qpos=qpos, qvel=qvel)
  for _ in range(3):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  reset = sim.state.snapshot()
  np.testing.assert_array_equal(reset.qpos, endpoint.qpos)
  np.testing.assert_array_equal(reset.qvel, endpoint.qvel)


def _expected_pair_contacts(model, data, descriptor):
  expected = {}
  for contact in data.contact[:data.ncon]:
    if tuple(map(int, contact.flex)) != (0, 1):
      continue
    e1, e2 = map(int, contact.elem)
    slots = np.flatnonzero(
        (descriptor.kind == _KIND_ELEMENT_PAIR)
        & (descriptor.flex1 == 0) & (descriptor.flex2 == 1)
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


@pytest.mark.parametrize("dof,count,expected_count", [
    ("trilinear", "2 2 2", 22), ("quadratic", "3 3 3", 50)])
def test_cpu_interpolated_tetra_pair_uses_current_compiled_vertices(
    dof, count, expected_count):
  model, _, _, refs = _interpolated_pair_fixture(dof)
  desc = lower_flex_contacts(model)
  assert tuple(map(int, model.flex_dim)) == (3, 3)
  assert tuple(map(int, np.abs(model.flex_interp))) == (
      (1, 1) if dof == "trilinear" else (2, 2))
  assert tuple(map(int, model.flex_vertnum)) == (
      (8, 8) if dof == "trilinear" else (27, 27))
  expected = [_expected_pair_contacts(model, ref, desc) for ref in refs]
  assert [len(x) for x in expected] == [expected_count, 0]
  assert [ref.ncon for ref in refs] == [expected_count, 0]
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1.0
  assert np.linalg.norm(refs[0].qfrc_passive) > 1e-4
  assert np.all(desc.kind == _KIND_ELEMENT_PAIR)
  assert all(np.count_nonzero(desc.nodes1[slot] >= 0) == 4
             and np.count_nonzero(desc.nodes2[slot] >= 0) == 4
             for slot in expected[0])
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


@pytest.mark.gpu
@pytest.mark.parametrize("dof,count,expected_count", [
    ("trilinear", "2 2 2", 22), ("quadratic", "3 3 3", 50)])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="interpolated flex pair requires GPU opt-in")
def test_native_public_interpolated_tetra_pair_B2_rows_force_replay_reset(
    dof, count, expected_count, jacobian):
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, refs = _interpolated_pair_fixture(
      dof, jacobian=jacobian)
  desc = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, ref, desc) for ref in refs]
  assert [len(x) for x in expected] == [expected_count, 0]
  sim = _model_scaled_simulation(model, qpos, qvel)
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  bundle = cc._flex_contact_current
  result = bundle["contact_result"]
  descriptor = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
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
  system = sim.accepted_step["system"]
  native_force = system["qfrc_constraint"].detach().cpu().numpy()
  native_passive = sim._flex._qfrc_passive.detach().cpu().numpy()
  cpu_force, cpu_passive = [], []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
    cpu_passive.append(np.asarray(data.qfrc_passive, np.float32))
  cpu_force, cpu_passive = np.stack(cpu_force), np.stack(cpu_passive)
  assert np.linalg.norm(cpu_force[0]) > 1.0
  assert np.linalg.norm(cpu_passive[0]) > 1e-4
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


def _articulated_interpolated_pair_fixture(jacobian=None,
                                          heterogeneous=False):
  if heterogeneous:
    lower_contact = ('<contact contype="1" conaffinity="1" '
                     'selfcollide="none" condim="4" '
                     'friction=".6 .4 .03" priority="2" '
                     'solref=".02 1" solimp=".8 .9 .01 .5 2"/>')
    upper_contact = ('<contact contype="1" conaffinity="1" '
                     'selfcollide="none" condim="6" '
                     'friction=".8 .5 .04" priority="1" '
                     'solref=".03 2" solimp=".7 .85 .02 .4 3"/>')
    upper_elasticity = ('<elasticity young="500" poisson=".25" '
                        'damping=".2"/>')
  else:
    lower_contact = upper_contact = (
        '<contact contype="1" conaffinity="1" selfcollide="none"/>')
    upper_elasticity = '<elasticity young="1000" poisson=".2" damping=".1"/>'
  xml = """<mujoco><option timestep=".001" gravity="0 0 0"/>
    <worldbody>
      <body name="lower_body">
        <joint name="lower_hinge" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="lower" type="grid" count="2 2 2"
                  spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
          {lower_contact}
          <elasticity young="1000" poisson=".2" damping=".1"/>
        </flexcomp>
      </body>
      <body name="upper_body" pos="0 0 .04">
        <joint name="upper_slide" type="slide" axis="0 0 1"/>
        <joint name="upper_hinge" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="upper" type="grid" count="2 2 2"
                  spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
          {upper_contact}
          {upper_elasticity}
        </flexcomp>
      </body>
    </worldbody></mujoco>"""
  xml = xml.format(lower_contact=lower_contact,
                   upper_contact=upper_contact,
                   upper_elasticity=upper_elasticity)
  model = mujoco.MjModel.from_xml_string(xml)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  lower_hinge = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "lower_hinge")
  upper_slide = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "upper_slide")
  upper_hinge = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "upper_hinge")
  assert min(lower_hinge, upper_slide, upper_hinge) >= 0
  qpos = np.stack([np.asarray(model.qpos0, np.float32)] * 2)
  qpos[0, int(model.jnt_qposadr[lower_hinge])] = np.float32(.1)
  qpos[0, int(model.jnt_qposadr[upper_hinge])] = np.float32(-.08)
  qpos[1] = qpos[0]
  qpos[1, int(model.jnt_qposadr[upper_slide])] = np.float32(.2)
  qvel = np.zeros((2, int(model.nv)), dtype=np.float32)
  # Distinct node velocities exercise the elastic rate response in addition
  # to the parent joint contributions in the contact Jacobian.
  qvel[0] = np.linspace(-.02, .02, int(model.nv), dtype=np.float32)
  qvel[0, int(model.jnt_dofadr[lower_hinge])] = np.float32(.02)
  qvel[0, int(model.jnt_dofadr[upper_hinge])] = np.float32(-.015)
  qvel[0, int(model.jnt_dofadr[upper_slide])] = np.float32(.01)
  qvel[1] = np.linspace(.005, -.005, int(model.nv), dtype=np.float32)
  qvel[1, int(model.jnt_dofadr[lower_hinge])] = np.float32(.005)
  refs = []
  for env in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env].astype(np.float64)
    data.qvel[:] = qvel[env].astype(np.float64)
    mujoco.mj_forward(model, data)
    refs.append(data)
  return model, qpos, qvel, refs, (lower_hinge, upper_slide, upper_hinge)


def _mixed_direct_interpolated_pair_fixture(dof="trilinear", jacobian=None):
  count = _INTERP_COUNTS[dof]
  xml = f"""<mujoco><option timestep=".001" gravity="0 0 0" jacobian="dense"/>
    <worldbody>
      <flexcomp name="edge" type="grid" count="2 1 1"
                spacing=".1 .1 .1" mass="1" dim="1" radius=".01">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="10" damping=".1"/>
      </flexcomp>
      <flexcomp name="volume" type="grid" count="{count}"
                pos="0 0 .015" spacing=".1 .1 .1" mass="1" dim="3"
                dof="{dof}" radius=".01">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  if jacobian is not None:
    model.opt.jacobian = {
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    }[jacobian]
  source = mujoco.MjData(model)
  source.qvel[0] = .01
  mujoco.mj_forward(model, source)
  qpos = np.stack([np.asarray(source.qpos, np.float32)] * 2)
  qvel = np.stack([np.asarray(source.qvel, np.float32)] * 2)
  # Keep the compiled candidate and row slots fixed while making the second
  # world's edge/volume pair physically inactive.
  start = int(model.flex_vertadr[1])
  number = int(model.flex_vertnum[1])
  qpos[1, 3 * start + 2:3 * (start + number):3] += np.float32(.3)
  qvel[1, 0] = 0
  refs = []
  for env in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env].astype(np.float64)
    data.qvel[:] = qvel[env].astype(np.float64)
    mujoco.mj_forward(model, data)
    refs.append(data)
  return model, qpos, qvel, refs


@pytest.mark.parametrize("dof,count", [
    ("trilinear", 6), ("quadratic", 12)])
def test_cpu_direct_edge_interpolated_volume_pair_matches_source(
    dof, count):
  model, _, _, refs = _mixed_direct_interpolated_pair_fixture(dof)
  descriptor = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, data, descriptor)
              for data in refs]
  assert tuple(map(int, model.flex_dim)) == (1, 3)
  assert tuple(map(int, np.abs(model.flex_interp))) == (
      0, 1 if dof == "trilinear" else 2)
  assert [len(x) for x in expected] == [count, 0]
  assert [data.ncon for data in refs] == [count, 0]
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1.0
  assert np.linalg.norm(refs[0].qfrc_passive) > 1e-4
  assert np.all(descriptor.kind == _KIND_ELEMENT_PAIR)
  for slot, contact in expected[0].items():
    assert int(descriptor.flex1[slot]) == 0
    assert int(descriptor.flex2[slot]) == 1
    assert np.count_nonzero(descriptor.nodes1[slot] >= 0) == 2
    assert np.count_nonzero(descriptor.nodes2[slot] >= 0) == 4
    assert int(descriptor.row_span[slot]) == int(contact.dim) + 1
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


@pytest.mark.gpu
@pytest.mark.parametrize("dof,contact_count", [("trilinear", 6),
                                               ("quadratic", 12)])
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="mixed interpolated pair requires GPU opt-in")
def test_native_public_direct_edge_interpolated_volume_B2_lifecycle(
    dof, contact_count, jacobian):
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, refs = _mixed_direct_interpolated_pair_fixture(
      dof, jacobian=jacobian)
  descriptor = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, data, descriptor)
              for data in refs]
  assert [len(x) for x in expected] == [contact_count, 0]
  sim = _model_scaled_simulation(model, qpos, qvel)
  system0 = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  result = cc._flex_contact_current["contact_result"]
  active = result["active"].detach().cpu().numpy()
  expected_active = np.zeros_like(active)
  expected_active[0, list(expected[0])] = 1
  np.testing.assert_array_equal(active, expected_active)
  native_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = system0["R"].detach().cpu().numpy().copy()
  native_ar = system0["ar"].detach().cpu().numpy().copy()
  native_dist = result["dist"].detach().cpu().numpy()
  native_pos = result["pos"].detach().cpu().numpy()
  native_frame = result["frame"].detach().cpu().numpy()
  desc = cc.descriptor.flex_contact_descriptor
  row_base = int(cc.descriptor.flex_contact_base)
  for env, data in enumerate(refs):
    cpu_j = _dense_efc_jacobian(model, data)
    for slot, contact in expected[env].items():
      np.testing.assert_allclose(native_dist[env, slot], contact.dist,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_pos[env, slot], contact.pos,
                                 rtol=0, atol=6e-6)
      np.testing.assert_allclose(native_frame[env, slot].reshape(3, 3),
                                 np.asarray(contact.frame).reshape(3, 3),
                                 rtol=0, atol=6e-5)
      row = row_base + int(desc.row_start[slot])
      span = int(desc.row_span[slot])
      address = int(contact.efc_address)
      assert span == int(contact.dim) + 1
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
  cpu_force, cpu_passive = [], []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
    cpu_passive.append(np.asarray(data.qfrc_passive, np.float32))
  cpu_force, cpu_passive = np.stack(cpu_force), np.stack(cpu_passive)
  assert np.linalg.norm(cpu_force[0]) > 1.0
  assert np.linalg.norm(cpu_passive[0]) > 1e-4
  np.testing.assert_allclose(native_force, cpu_force, rtol=1e-4, atol=2e-5)
  np.testing.assert_allclose(sim._flex._qfrc_passive.detach().cpu().numpy(),
                             cpu_passive, rtol=1e-4, atol=2e-5)
  checkpoint = sim.snapshot()
  for _ in range(2):
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
  for _ in range(2):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, endpoint.qpos)
  np.testing.assert_array_equal(replay.qvel, endpoint.qvel)
  sim.restore(initial)
  sim.reset(qpos=qpos, qvel=qvel)
  for _ in range(3):
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
  reset = sim.state.snapshot()
  np.testing.assert_array_equal(reset.qpos, endpoint.qpos)
  np.testing.assert_array_equal(reset.qvel, endpoint.qvel)


def test_cpu_interpolated_attached_pair_has_source_joint_jacobian_and_force():
  model, _, _, refs, joints = _articulated_interpolated_pair_fixture()
  descriptor = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, data, descriptor) for data in refs]
  assert [len(x) for x in expected] == [22, 0]
  assert [data.ncon for data in refs] == [22, 0]
  assert np.linalg.norm(refs[0].qfrc_constraint) > 1.0
  assert np.linalg.norm(refs[0].qfrc_passive) > 1e-4
  jacobian = _dense_efc_jacobian(model, refs[0])
  lower_dof = int(model.jnt_dofadr[joints[0]])
  slide_dof = int(model.jnt_dofadr[joints[1]])
  upper_dof = int(model.jnt_dofadr[joints[2]])
  for contact in expected[0].values():
    address = int(contact.efc_address)
    dim = int(contact.dim)
    # The relative contact rows retain both articulated trees, while flex
    # node rows carry the compiled Q1 support DOFs independently.
    assert np.linalg.norm(jacobian[address:address + dim, lower_dof]) > 1e-4
    assert np.linalg.norm(jacobian[address:address + dim, slide_dof]) > 1e-4
    assert np.linalg.norm(jacobian[address:address + dim, upper_dof]) > 1e-4
  program = FlexContactProgram(model, device="cpu")
  assert program._narrowphase_admitted
  assert program._has_native_flex_pair_candidates


def test_cpu_attached_interpolated_pair_mixes_distinct_materials():
  model, _, _, refs, joints = _articulated_interpolated_pair_fixture(
      heterogeneous=True)
  descriptor = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, data, descriptor) for data in refs]
  assert [len(x) for x in expected] == [22, 0]
  jacobian = _dense_efc_jacobian(model, refs[0])
  axes = [int(model.jnt_dofadr[joint]) for joint in joints]
  for slot, contact in expected[0].items():
    assert int(contact.dim) == int(descriptor.condim[slot]) == 4
    np.testing.assert_allclose(contact.friction, descriptor.friction[slot],
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(contact.solref, descriptor.solref[slot],
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(contact.solimp, descriptor.solimp[slot],
                               rtol=0, atol=1e-7)
    address = int(contact.efc_address)
    for dof in axes:
      assert np.linalg.norm(jacobian[address:address + contact.dim, dof]) > 1e-4


@pytest.mark.gpu
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="articulated interpolated pair requires GPU opt-in")
def test_native_public_articulated_interpolated_pair_B2_lifecycle(
    jacobian, heterogeneous):
  pytest.importorskip("torch")
  from mujoco_metal.simulation import MetalSimulation

  model, qpos, qvel, refs, _ = _articulated_interpolated_pair_fixture(
      jacobian, heterogeneous=heterogeneous)
  descriptor = lower_flex_contacts(model)
  expected = [_expected_pair_contacts(model, data, descriptor) for data in refs]
  assert [len(x) for x in expected] == [22, 0]
  sim = _model_scaled_simulation(model, qpos, qvel)
  initial_system = sim.assembled_system(recompute=True)
  program = sim._flex._contact_program
  assert program is not None and program._narrowphase_admitted
  cc = sim._coupled_constraints
  bundle = cc._flex_contact_current
  result = bundle["contact_result"]
  descriptor = cc.descriptor.flex_contact_descriptor
  flex_base = int(cc.descriptor.flex_contact_base)
  active = result["active"].detach().cpu().numpy()
  expected_active = np.zeros_like(active)
  expected_active[0, list(expected[0])] = 1
  np.testing.assert_array_equal(active, expected_active)
  native_j = cc.materialize_jacobian().detach().cpu().numpy().copy()
  native_R = initial_system["R"].detach().cpu().numpy().copy()
  native_ar = initial_system["ar"].detach().cpu().numpy().copy()
  contact_result = bundle["contact_result"]
  native_dist = contact_result["dist"].detach().cpu().numpy()
  native_pos = contact_result["pos"].detach().cpu().numpy()
  native_frame = contact_result["frame"].detach().cpu().numpy()
  for env, data in enumerate(refs):
    cpu_j = _dense_efc_jacobian(model, data)
    for slot, contact in expected[env].items():
      row = flex_base + int(descriptor.row_start[slot])
      span = int(descriptor.row_span[slot])
      address = int(contact.efc_address)
      if heterogeneous:
        assert int(contact.dim) == int(result["condim"][slot]) == 4
        np.testing.assert_allclose(
            result["friction"][slot].detach().cpu().numpy(), contact.friction,
            rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            result["solref"][slot].detach().cpu().numpy(), contact.solref,
            rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            result["solimp"][slot].detach().cpu().numpy(), contact.solimp,
            rtol=0, atol=1e-7)
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
  cpu_force, cpu_passive = [], []
  for data in refs:
    mujoco.mj_step(model, data)
    cpu_force.append(np.asarray(data.qfrc_constraint, np.float32))
    cpu_passive.append(np.asarray(data.qfrc_passive, np.float32))
  cpu_force, cpu_passive = np.stack(cpu_force), np.stack(cpu_passive)
  assert np.linalg.norm(cpu_force[0]) > 1.0
  assert np.linalg.norm(cpu_passive[0]) > 1e-4
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
