# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Opt-in end-to-end DIAGEXACT gates through public prepared stages."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.constraint_impedance import (
    ConeGroup,
    compile_contact_cone_groups,
    exact_constraint_diagonal_cpu,
    recompute_impedance_cpu,
)


_GPU_ONLY = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit GPU qualification slot")


def _diag_model(cone, solver):
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option timestep='.001' solver='{solver}' iterations='80' cone='{cone}'
      impratio='1.7'>
      <flag energy='enable'/>
    </option>
    <worldbody>
      <geom name='floor' type='plane' size='2 2 .1'/>
      <body name='floating' pos='0 0 .075'>
        <freejoint/>
        <geom type='box' size='.1 .08 .08' mass='1' condim='4'
          friction='.8 .4 .2'/>
        <body name='upper' pos='.25 0 .1'>
          <joint name='hinge_a' type='hinge' axis='0 1 0'/>
          <geom type='capsule' fromto='0 0 0 .25 0 0' size='.035' mass='.3'/>
          <body name='tip' pos='.25 0 0'>
            <joint name='hinge_b' type='hinge' axis='0 1 0'/>
            <geom type='capsule' fromto='0 0 0 .2 0 0' size='.025' mass='.2'/>
          </body>
        </body>
      </body>
    </worldbody>
    <equality>
      <joint name='nonlocal' joint1='hinge_a' joint2='hinge_b'
        polycoef='0 1 0 0 0' solimp='.82 .91 .01'/>
    </equality>
  </mujoco>""")


def _cpu_reference(model, qpos, qvel):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  result = {
      "mass": mass,
      "J": np.asarray(data.efc_J, dtype=np.float64).reshape(
          int(data.nefc), int(model.nv)).copy(),
      "R": np.asarray(data.efc_R, dtype=np.float64).copy(),
      "diagA": np.asarray(data.efc_diagA, dtype=np.float64).copy(),
      "qacc": np.asarray(data.qacc, dtype=np.float64).copy(),
      "qfrc_constraint": np.asarray(data.qfrc_constraint,
                                    dtype=np.float64).copy(),
      "efc_type": np.asarray(data.efc_type, dtype=np.int32).copy(),
      "efc_id": np.asarray(data.efc_id, dtype=np.int32).copy(),
      "contacts": tuple({
          "geom1": int(data.contact[i].geom1),
          "geom2": int(data.contact[i].geom2),
          "dim": int(data.contact[i].dim),
          "efc_address": int(data.contact[i].efc_address),
          "mu": float(data.contact[i].mu),
      } for i in range(int(data.ncon))),
      "nefc": int(data.nefc),
      "ncon": int(data.ncon),
  }
  mujoco.mj_inverse(model, data)
  result["qfrc_inverse"] = np.asarray(data.qfrc_inverse, dtype=np.float64).copy()
  return result


def _canonical_cpu_row_map(model, descriptor, cpu, cone_groups):
  """Map exact logical equality/contact rows to pinned CPU efc rows."""
  equality_kind = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  mapping = {}
  for eqid in range(int(descriptor.neq)):
    cpu_rows = np.flatnonzero((cpu["efc_type"] == equality_kind)
                              & (cpu["efc_id"] == eqid))
    count = int(descriptor.eq_rownum[eqid])
    if cpu_rows.size != count:
      raise AssertionError(
          f"equality {eqid}: pinned rows {cpu_rows.size}, expected {count}")
    start = int(descriptor.eq_rowadr[eqid])
    for offset, cpu_row in enumerate(cpu_rows):
      mapping[start + offset] = int(cpu_row)

  packed = np.asarray(descriptor.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  offsets = np.asarray(descriptor.pair_contact_offset, dtype=np.int32)
  used_per_pair = np.zeros(int(descriptor.npairs), dtype=np.int32)
  cone_by_start = {int(group[0]): index
                   for index, group in enumerate(cone_groups)}
  expected_mu = np.zeros(len(cone_groups), dtype=np.float64)
  for contact in cpu["contacts"]:
    pair_matches = np.flatnonzero(
        ((descriptor.geom1 == contact["geom1"])
         & (descriptor.geom2 == contact["geom2"]))
        | ((descriptor.geom1 == contact["geom2"])
           & (descriptor.geom2 == contact["geom1"])))
    if pair_matches.size != 1:
      raise AssertionError("pinned contact does not map to one lowered pair")
    pair = int(pair_matches[0])
    slot = int(offsets[pair] + used_per_pair[pair])
    used_per_pair[pair] += 1
    condim, row_offset, kind = (int(x) for x in packed[slot])
    if condim != int(contact["dim"]):
      raise AssertionError("pinned contact dim differs from compiled candidate")
    start = int(descriptor.nr_joint) + row_offset
    row_count = (condim if kind == 1 else 2 * (condim - 1))
    if start not in cone_by_start and condim != 1:
      raise AssertionError("frictional contact has no compiled cone group")
    cpu_start = int(contact["efc_address"])
    if cpu_start < 0:
      continue
    for offset in range(row_count):
      mapping[start + offset] = cpu_start + offset
    if condim != 1:
      expected_mu[cone_by_start[start]] = float(contact["mu"])
  return mapping, expected_mu


def test_cpu_diagexact_oracle_is_nontrivial_for_articulated_contact_rows():
  """Pin the fixture's exact-vs-approximate gap and nonlocal row structure."""
  model = _diag_model("elliptic", "PGS")
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[int(model.jnt_qposadr[1])] = .24
  qpos[int(model.jnt_qposadr[2])] = -.11
  qvel = np.linspace(-.13, .16, model.nv, dtype=np.float64)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  exact = _cpu_reference(model, qpos, qvel)
  model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  approximate = _cpu_reference(model, qpos, qvel)
  assert exact["ncon"] > 0 and exact["nefc"] > 0
  assert np.max(np.abs(exact["R"] - approximate["R"])) > .05
  # The compiled joint equality is genuinely articulated/nonlocal, rather
  # than an identity or one-DOF diagonal case.
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  desc = __import__("mujoco_metal.coupled_constraints", fromlist=[
      "lower_coupled_constraints"]).lower_coupled_constraints(model)
  eq_start = int(desc.eq_rowadr[0])
  eq_jac = exact["J"]
  # Locate the pinned CPU equality row by type/id, independent of its ordering
  # relative to contact rows.
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  row = next(i for i in range(data.nefc)
             if int(data.efc_type[i]) == equality_type
             and int(data.efc_id[i]) == 0)
  assert np.count_nonzero(np.abs(eq_jac[row]) > 1e-8) >= 2
  assert eq_start >= 0


@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
def test_cpu_diagexact_canonical_rows_map_to_pinned_contact_and_equality_ids(cone):
  from mujoco_metal.coupled_constraints import lower_coupled_constraints

  model = _diag_model(cone, "PGS")
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  data.qpos[int(model.jnt_qposadr[1])] = .24
  data.qpos[int(model.jnt_qposadr[2])] = -.11
  mujoco.mj_forward(model, data)
  cpu = _cpu_reference(model, data.qpos.copy(), data.qvel.copy())
  descriptor = lower_coupled_constraints(model)
  packed, _ = compile_contact_cone_groups(descriptor)
  mapping, expected_mu = _canonical_cpu_row_map(
      model, descriptor, cpu, packed)
  assert descriptor.eq_rownum[0] == 1
  assert mapping[int(descriptor.eq_rowadr[0])] in np.flatnonzero(
      (cpu["efc_type"] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
      & (cpu["efc_id"] == 0))
  assert len(mapping) >= 1 + int(data.ncon) * int(data.contact[0].dim)
  assert np.count_nonzero(expected_mu) == int(data.ncon)


@pytest.mark.gpu
@_GPU_ONLY
@pytest.mark.parametrize("cone", ["elliptic", "pyramidal"])
@pytest.mark.parametrize("solver", ["PGS", "CG", "Newton"])
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                      "integrated_scalable_v1"])
def test_public_diagexact_prepared_and_inverse_stages(cone, solver, profile):
  """Exercise exact R through POS/VEL/ACC/constraint and inverse APIs."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("MPS is unavailable")
  from mujoco_metal.forward_stages import ForwardStage
  from mujoco_metal.native_api import mj_inverseSkip
  from mujoco_metal.simulation import MetalSimulation

  model = _diag_model(cone, solver)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None, :], 2, axis=0)
  qpos[0, int(model.jnt_qposadr[1])] = .24
  qpos[0, int(model.jnt_qposadr[2])] = -.11
  qpos[0, 2] = .075
  qpos[1, 2] = .65
  qvel = np.stack((np.linspace(-.13, .16, model.nv),
                   np.linspace(.09, -.12, model.nv))).astype(np.float32)

  cpu = [_cpu_reference(model, qpos[w].astype(np.float64),
                        qvel[w].astype(np.float64)) for w in range(2)]
  assert cpu[0]["ncon"] > 0 and cpu[1]["ncon"] == 0
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  assert bool(sim._component_mass_enabled) == (
      profile == "integrated_scalable_v1")
  record = sim.prepare_forward_position()
  cc = sim._coupled_constraints
  rows = cc._assembly_views(include_optimizer_outputs=False)
  J = rows["J"].detach().cpu().numpy().astype(np.float64)
  active = rows["active"].detach().cpu().numpy() != 0
  source_impedance = rows["row_impedance"].detach().cpu().numpy().astype(
      np.float64)
  actual_R = rows["R"].detach().cpu().numpy().astype(np.float64)
  mass = np.stack([item["mass"] for item in cpu])
  exact_diag = exact_constraint_diagonal_cpu(
      J, mass, active_rows=active)
  safe_impedance = np.where(active, source_impedance, .5)
  desc = cc.descriptor
  packed_groups, packed_friction = compile_contact_cone_groups(desc)
  groups = []
  for (start, condim, kind), friction in zip(packed_groups, packed_friction):
    if not np.any(friction):
      continue
    groups.append(ConeGroup(
        int(start), int(condim), "elliptic" if int(kind) == 1 else "pyramidal",
        tuple(np.asarray(friction, dtype=np.float64))))
  expected_R, expected_diagA, expected_mu = recompute_impedance_cpu(
      exact_diag, safe_impedance, cone_groups=groups,
      impratio=float(model.opt.impratio))
  np.testing.assert_allclose(actual_R[active], expected_R[active],
                             rtol=5e-4, atol=5e-6)
  helper = sim._constraint_impedance._workspace
  actual_diagA = helper["diagA"].detach().cpu().numpy().astype(np.float64)
  np.testing.assert_allclose(actual_diagA[active], expected_diagA[active],
                             rtol=5e-4, atol=5e-6)
  actual_mu = helper["cone_mu"].detach().cpu().numpy().astype(np.float64)
  assert not np.any(active[1, int(desc.nr_joint):])
  assert record.values[ForwardStage.POS]["exact_impedance_status"].tolist() == [0, 0]

  # Validate canonical rows before calling any constraint optimizer. This
  # isolates row-production/impedance parity from solver convergence.
  row_map, expected_contact_mu = [], []
  for world in range(2):
    world_map, world_mu = _canonical_cpu_row_map(
        model, desc, cpu[world], packed_groups)
    row_map.append(world_map)
    expected_contact_mu.append(world_mu)
    for canonical_row, cpu_row in world_map.items():
      np.testing.assert_allclose(
          J[world, canonical_row], cpu[world]["J"][cpu_row],
          rtol=5e-4, atol=5e-6)
      np.testing.assert_allclose(
          actual_R[world, canonical_row], cpu[world]["R"][cpu_row],
          rtol=5e-4, atol=5e-6)
      np.testing.assert_allclose(
          actual_diagA[world, canonical_row], cpu[world]["diagA"][cpu_row],
          rtol=5e-4, atol=5e-6)
  if groups:
    np.testing.assert_allclose(actual_mu, np.asarray(expected_contact_mu),
                               rtol=5e-4, atol=5e-6)

  velocity = sim.prepare_forward_velocity(record)
  sim.prepare_forward_actuation(record)
  acceleration = sim.prepare_forward_acceleration(record)
  constrained = sim.prepare_forward_constraint(record)
  assert acceleration["status"].tolist() == [0, 0]
  assert constrained["status"].tolist() == [0, 0]
  # The solve must consume the exact R captured by POS and preserve it through
  # cached VEL/constraint stages.
  rows_after = cc._assembly_views(include_optimizer_outputs=False)
  np.testing.assert_array_equal(
      rows_after["R"].detach().cpu().numpy(), actual_R.astype(np.float32))

  qacc = torch.as_tensor(np.stack([item["qacc"] for item in cpu]).astype(
      np.float32), device=sim.state._device)
  expected_inverse = np.stack([item["qfrc_inverse"] for item in cpu])
  for skipstage in (mujoco.mjtStage.mjSTAGE_VEL,
                    mujoco.mjtStage.mjSTAGE_POS,
                    mujoco.mjtStage.mjSTAGE_NONE):
    result = mj_inverseSkip(
        sim, skipstage=skipstage, record=record, qacc=qacc,
        return_details=True)
    np.testing.assert_allclose(
        result["qfrc_inverse"].detach().cpu().numpy(), expected_inverse,
        rtol=2e-3, atol=2e-4)
    # Public inverse queries run transactionally and must retain the same
    # exact-R stage snapshot for the next skip-stage call.
    np.testing.assert_array_equal(
        cc._assembly_views(include_optimizer_outputs=False)["R"].detach().cpu().numpy(),
        actual_R.astype(np.float32))

  # Compare the constrained solve to the independently computed pinned state.
  np.testing.assert_allclose(
      constrained["qacc"].detach().cpu().numpy(),
      np.stack([item["qacc"] for item in cpu]).astype(np.float32),
      rtol=2e-3, atol=2e-4)
  np.testing.assert_allclose(
      constrained["qfrc_constraint"].detach().cpu().numpy(),
      np.stack([item["qfrc_constraint"] for item in cpu]).astype(np.float32),
      rtol=2e-3, atol=2e-4)


@pytest.mark.gpu
@_GPU_ONLY
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                      "integrated_scalable_v1"])
def test_public_diagexact_restore_reset_mutation_and_failed_peer(profile):
  """Check exact-R lifecycle in addition to a one-shot row/solve comparison."""
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("MPS is unavailable")
  from mujoco_metal.simulation import MetalSimulation

  model = _diag_model("pyramidal", "PGS")
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_DIAGEXACT)
  qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float32)[None, :], 2, axis=0)
  qpos[0, int(model.jnt_qposadr[1])] = .24
  qpos[0, int(model.jnt_qposadr[2])] = -.11
  qpos[0, 2] = .075
  qpos[1, 2] = .65
  qvel = np.stack((np.linspace(-.13, .16, model.nv),
                   np.linspace(.09, -.12, model.nv))).astype(np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  checkpoint = sim.snapshot()

  # In-place mutation of an input owner must invalidate the prepared POS
  # record before VEL writes anything. Restore the exact checkpoint afterward.
  record = sim.prepare_forward_position()
  sim.state._qpos[0, 0].add_(.001)
  with pytest.raises((ValueError, RuntimeError)):
    sim.prepare_forward_velocity(record)
  sim.restore(checkpoint)

  reference = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    reference.append(data)
  for step in range(3):
    status = sim.step().detach().cpu().numpy().copy()
    assert status.tolist() == [0, 0], f"{profile} step {step} status {status}"
    for world, data in enumerate(reference):
      mujoco.mj_step(model, data)
      np.testing.assert_allclose(
          sim.state.qpos[world].detach().cpu().numpy(), data.qpos,
          rtol=4e-4, atol=5e-6,
          err_msg=f"{profile} step {step} world {world} qpos")
      np.testing.assert_allclose(
          sim.state.qvel[world].detach().cpu().numpy(), data.qvel,
          rtol=4e-4, atol=2e-5,
          err_msg=f"{profile} step {step} world {world} qvel")
  expected_qpos = sim.state.qpos.detach().clone()
  expected_qvel = sim.state.qvel.detach().clone()
  sim.restore(checkpoint)
  for step in range(3):
    assert sim.step().tolist() == [0, 0], f"{profile} replay step {step}"
  assert torch.equal(sim.state.qpos, expected_qpos)
  assert torch.equal(sim.state.qvel, expected_qvel)

  # Inject a sticky failed peer after replay: the healthy world advances while
  # the failed world's state is preserved. Reset only that peer, leaving the
  # healthy world untouched.
  healthy_before = sim.state.qpos[0].detach().clone()
  failed_before = sim.state.qpos[1].detach().clone()
  sim.state._status[1] = 2
  status = sim.step().detach().cpu().tolist()
  assert status == [0, 2]
  assert not torch.equal(sim.state.qpos[0], healthy_before)
  assert torch.equal(sim.state.qpos[1], failed_before)
  healthy_after = sim.state.qpos[0].detach().clone()
  sim.reset(env_ids=[1])
  assert sim.state.status.tolist() == [0, 0]
  assert torch.equal(sim.state.qpos[0], healthy_after)
  np.testing.assert_array_equal(
      sim.state.qpos[1].detach().cpu().numpy(), model.qpos0.astype(np.float32))
