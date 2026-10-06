"""Pinned CPU and opt-in Metal witnesses for attached kind-6 capsules.

This fixture is intentionally CPU-only source/oracle evidence.  It exercises
an off-center, articulated flex pair and is not a claim that the current Metal
row/solve path is admitted or qualified.
"""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    FlexContactProgram, _KIND_ELEMENT_PAIR, _candidate_narrowphase_supported,
    lower_flex_contacts,
)


def _model(jacobian=None):
    jacobian_attribute = "" if jacobian is None else f' jacobian="{jacobian}"'
    return mujoco.MjModel.from_xml_string(f"""
      <mujoco><option gravity="0 0 0" cone="elliptic"{jacobian_attribute}/>
        <worldbody>
          <body name="carrier_a" pos="0 0 0">
            <freejoint/>
            <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
            <flexcomp name="a" type="grid" count="3 1 1"
                      pos="0 0 0" spacing=".05 .05 .05"
                      mass="1" dim="1" radius=".00500000012345">
              <contact contype="1" conaffinity="1" selfcollide="none" condim="1"
                       margin=".000100000012345" gap=".000200000023456"/>
              <edge stiffness="10" damping=".1"/>
            </flexcomp>
          </body>
          <body name="carrier_b" pos=".03 .004 0">
            <joint name="carrier_hinge" type="hinge" axis="0 0 1"/>
            <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
            <flexcomp name="b" type="grid" count="3 1 1"
                      pos="0 0 0" spacing=".05 .05 .05"
                      mass="1" dim="1" radius=".00700000023456">
              <contact contype="1" conaffinity="1" selfcollide="none" condim="1"
                       margin=".000300000034567" gap=".000400000045678"/>
              <edge stiffness="10" damping=".1"/>
            </flexcomp>
          </body>
        </worldbody>
      </mujoco>""")


def test_attached_capsule_contacts_have_pinned_rigid_and_node_jacobian_terms():
    model = _model()
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    qvel = np.linspace(-.02, .025, model.nv)
    mujoco.mj_integratePos(model, qpos, qvel, .0002)
    qpos = qpos.astype(np.float32).astype(np.float64)
    qvel = qvel.astype(np.float32).astype(np.float64)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    assert model.nflex == 2
    assert model.nv == 25
    assert data.ncon == 6
    assert np.linalg.norm(data.qfrc_constraint) > 1.0
    descriptor = lower_flex_contacts(model)
    grouped = {}
    for contact in data.contact[:data.ncon]:
        pair = (tuple(map(int, contact.flex)), tuple(map(int, contact.elem)))
        grouped.setdefault(pair, []).append(contact)
    expected_slots = set()
    for (flexes, elems), contacts in grouped.items():
        f1, f2 = flexes
        e1, e2 = elems
        candidates = np.flatnonzero(
            (descriptor.kind == _KIND_ELEMENT_PAIR)
            & (descriptor.flex1 == f1) & (descriptor.elem1 == e1)
            & (descriptor.flex2 == f2) & (descriptor.elem2 == e2))
        candidates = candidates[np.argsort(descriptor.contact_ordinal[candidates])]
        assert len(candidates) >= len(contacts)
        for slot, contact in zip(candidates, contacts):
            expected_slots.add(int(slot))
            row = int(descriptor.row_start[slot])
            assert int(descriptor.row_span[slot]) == 1
            assert 0 <= row < descriptor.row_capacity
    assert expected_slots == {0, 1, 8, 9, 12, 13}
    local_rows = np.asarray([descriptor.row_start[s] for s in sorted(expected_slots)])
    assert np.all(np.diff(local_rows) > 0)
    efc_rows = [int(c.efc_address) for c in data.contact[:data.ncon]]
    assert sorted(efc_rows) == list(range(data.nefc))
    # Each flexcomp creates child node bodies under a moving carrier.  The
    # contact rows therefore combine moving rigid ancestor and flex-node terms.
    body_ids = np.array(model.flex_vertbodyid, dtype=np.int32)
    assert body_ids.shape == (6,)
    assert body_ids[0] != 1 and body_ids[3] != 5
    assert int(model.body_parentid[body_ids[0]]) == 1
    assert int(model.body_parentid[body_ids[3]]) == 5

    jac = np.asarray(data.efc_J).reshape(data.nefc, model.nv)
    carrier_a_dofs = np.arange(0, 6)
    carrier_b_dof = int(model.jnt_dofadr[10])
    node_dofs = []
    for vertex_body in body_ids:
        joints = np.flatnonzero(model.jnt_bodyid == vertex_body)
        assert joints.size == 3
        node_dofs.extend(dof
                         for j in joints
                         for dof in range(int(model.jnt_dofadr[j]),
                                          int(model.jnt_dofadr[j] + (6 if model.jnt_type[j] == 0 else 3 if model.jnt_type[j] == 1 else 1))))
    node_dofs = np.asarray(node_dofs, dtype=np.int32)

    hinge_moments = []
    for contact in data.contact[:data.ncon]:
        assert tuple(map(int, contact.flex)) == (0, 1)
        assert int(contact.dim) == 1
        row = jac[int(contact.efc_address)]
        # The point lies off the parent origins, so articulated transforms
        # contribute nontrivial generalized terms in addition to node motion.
        assert np.linalg.norm(row[carrier_a_dofs]) > .1
        hinge_moments.append(abs(row[carrier_b_dof]))
        assert np.linalg.norm(row[node_dofs]) > .1
        np.testing.assert_allclose(np.linalg.norm(contact.frame[:3]), 1.,
                                   rtol=0, atol=2e-15)

    assert max(hinge_moments) > .01

    mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="attached kind-6 detector requires GPU opt-in")
def test_native_attached_capsule_detector_matches_articulated_pinned_contacts():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("attached kind-6 detector requires MPS")
    model = _model()
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    qvel = np.linspace(-.02, .025, model.nv)
    mujoco.mj_integratePos(model, qpos, qvel, .0002)
    data.qpos[:] = qpos.astype(np.float32).astype(np.float64)
    data.qvel[:] = qvel.astype(np.float32).astype(np.float64)
    mujoco.mj_forward(model, data)
    descriptor = lower_flex_contacts(model)
    expected = {}
    for contact in data.contact[:data.ncon]:
        f1, f2 = map(int, contact.flex)
        assert (f1, f2) == (0, 1)
        e1, e2 = map(int, contact.elem)
        slots = np.flatnonzero(
            (descriptor.kind == _KIND_ELEMENT_PAIR)
            & (descriptor.flex1 == f1) & (descriptor.elem1 == e1)
            & (descriptor.flex2 == f2) & (descriptor.elem2 == e2))
        slots = slots[np.argsort(descriptor.contact_ordinal[slots])]
        ordinal = sum(1 for c in expected.values()
                      if tuple(map(int, c.flex)) == (f1, f2)
                      and tuple(map(int, c.elem)) == (e1, e2))
        expected[int(slots[ordinal])] = contact
    assert set(expected) == {0, 1, 8, 9, 12, 13}

    def mps(value):
        return torch.as_tensor(np.asarray(value, np.float32).copy(),
                               dtype=torch.float32, device="mps").contiguous()

    program = FlexContactProgram(model, device="mps")
    assert program._narrowphase_admitted
    result = program.run_device(
        mps(data.flexvert_xpos[None]),
        mps(np.zeros((1, model.ngeom, 3), np.float32)),
        mps(np.tile(np.asarray([1, 0, 0, 0], np.float32),
                    (1, model.ngeom, 1))))
    active = result["active"].cpu().numpy()[0]
    assert set(np.flatnonzero(active).tolist()) == set(expected)
    distance = result["dist"].cpu().numpy()[0]
    pos = result["pos"].cpu().numpy()[0]
    normal = result["normal"].cpu().numpy()[0]
    status = result["narrowphase_status"].cpu().numpy()[0]
    for slot, contact in expected.items():
        assert status[slot] == 0
        np.testing.assert_allclose(distance[slot], contact.dist, rtol=0, atol=6e-6)
        np.testing.assert_allclose(pos[slot], contact.pos, rtol=0, atol=6e-6)
        np.testing.assert_allclose(normal[slot], contact.frame[:3],
                                   rtol=0, atol=6e-5)


def _dense_efc_jacobian(model, data):
    raw = np.asarray(data.efc_J, dtype=np.float64)
    if raw.size == int(data.nefc) * int(model.nv):
        return raw.reshape(int(data.nefc), int(model.nv)).copy()
    out = np.zeros((int(data.nefc), int(model.nv)), dtype=np.float64)
    rowadr = np.asarray(data.efc_J_rowadr, dtype=np.int32)
    rownnz = np.asarray(data.efc_J_rownnz, dtype=np.int32)
    colind = np.asarray(data.efc_J_colind, dtype=np.int32)
    for row in range(int(data.nefc)):
        start, count = int(rowadr[row]), int(rownnz[row])
        out[row, colind[start:start+count]] = raw[start:start+count]
    return out


def _public_attached_seed(model):
    qpos = np.repeat(model.qpos0[None, :], 2, axis=0).astype(np.float32)
    qvel = np.stack((np.linspace(-.02, .025, model.nv),
                     np.linspace(.017, -.023, model.nv))).astype(np.float32)
    refs = []
    hinge_dof = int(model.jnt_dofadr[10])
    body_ids = np.asarray(model.flex_vertbodyid, dtype=np.int32)
    node_dofs = []
    for vertex_body in body_ids:
        joints = np.flatnonzero(model.jnt_bodyid == vertex_body)
        for joint in joints:
            width = (6 if model.jnt_type[joint] == 0 else
                     3 if model.jnt_type[joint] == 1 else 1)
            node_dofs.extend(range(int(model.jnt_dofadr[joint]),
                                   int(model.jnt_dofadr[joint] + width)))
    node_dofs = np.asarray(node_dofs, dtype=np.int32)
    for env in range(2):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[env].astype(np.float64)
        data.qvel[:] = qvel[env].astype(np.float64)
        mujoco.mj_forward(model, data)
        assert data.ncon == 6 and data.nefc == 6
        assert np.linalg.norm(data.qfrc_constraint) > 1.0
        jac = _dense_efc_jacobian(model, data)
        hinge_terms = []
        for contact in data.contact[:data.ncon]:
            assert tuple(map(int, contact.flex)) == (0, 1)
            assert int(contact.dim) == 1
            row = jac[int(contact.efc_address)]
            assert np.linalg.norm(row[:6]) > .1
            assert np.linalg.norm(row[node_dofs]) > .1
            hinge_terms.append(abs(row[hinge_dof]))
        # The carrier hinge is represented across the whole six-contact
        # manifold. Individual rows can have zero hinge projection.
        assert max(hinge_terms) > .01
        refs.append(data)
    return qpos, qvel, refs


def test_public_attached_capsule_exact_native_seed_has_manifold_motion_terms():
    model = _model()
    qpos, qvel, refs = _public_attached_seed(model)
    assert qpos.shape == (2, model.nq) and qvel.shape == (2, model.nv)
    assert len(refs) == 2


def test_public_capsule_pair_admission_is_limited_to_direct_two_node_features():
    model = _model()
    descriptor = lower_flex_contacts(model)
    assert np.all(descriptor.kind == _KIND_ELEMENT_PAIR)
    assert _candidate_narrowphase_supported(model, descriptor)
    from mujoco_metal.stepping import validate_stepping_profile
    validate_stepping_profile(model, profile="integrated_scalable_v1")
    assert all(int(model.flex_dim[int(flex)]) == 1
               and int(model.flex_interp[int(flex)]) == 0
               for flex in np.concatenate((descriptor.flex1, descriptor.flex2)))
    assert np.all(np.count_nonzero(descriptor.nodes1 >= 0, axis=1) == 2)
    assert np.all(np.count_nonzero(descriptor.nodes2 >= 0, axis=1) == 2)
    # This gate is the precise source boundary of the candidate admission.
    # Interpolated/triangle/tetra capsule pairs are not covered by these rows.


@pytest.mark.gpu
@pytest.mark.parametrize(("jacobian", "expected_mode"),
                         [("dense", 0), ("sparse", 1)])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="articulated capsule row/solve path requires GPU opt-in")
def test_native_public_articulated_capsule_rows_force_step_replay_reset_B2(
        jacobian, expected_mode):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("articulated capsule integration requires MPS")
    from mujoco_metal.simulation import MetalSimulation

    model = _model(jacobian)
    qpos, qvel, refs = _public_attached_seed(model)

    sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                          profile="integrated_scalable_v1")
    system = sim.assembled_system(recompute=True)
    program = sim._flex._contact_program
    assert program is not None and program._narrowphase_admitted
    assert sim._coupled_constraints._jacobian_layout.mode == expected_mode
    desc = sim._coupled_constraints.descriptor.flex_contact_descriptor
    row_base = int(sim._coupled_constraints.descriptor.flex_contact_base)
    bundle = sim._coupled_constraints._flex_contact_current
    assert bundle is not None
    result, rows = bundle["contact_result"], bundle["rows"]
    actual_active = result["active"].detach().cpu().numpy()
    actual_j = sim._coupled_constraints.materialize_jacobian()
    if expected_mode == 1:
        assert result.get("relative_spatial_jacobian") is None

    for env, data in enumerate(refs):
        pinned_j = _dense_efc_jacobian(model, data)
        by_pair = {}
        for contact in data.contact[:data.ncon]:
            pair = (tuple(map(int, contact.flex)), tuple(map(int, contact.elem)))
            by_pair.setdefault(pair, []).append(contact)
        expected_slots = set()
        parent_hinge_terms = []
        for (flexes, elems), contacts in by_pair.items():
            f1, f2 = flexes
            e1, e2 = elems
            slots = np.flatnonzero(
                (desc.kind == _KIND_ELEMENT_PAIR)
                & (desc.flex1 == f1) & (desc.elem1 == e1)
                & (desc.flex2 == f2) & (desc.elem2 == e2))
            slots = slots[np.argsort(desc.contact_ordinal[slots])]
            assert len(slots) >= len(contacts)
            for ordinal, contact in enumerate(contacts):
                slot = int(slots[ordinal])
                expected_slots.add(slot)
                start = row_base + int(desc.row_start[slot])
                row = int(contact.efc_address)
                np.testing.assert_allclose(
                    actual_j[env, start].detach().cpu().numpy(),
                    pinned_j[row], rtol=5e-5, atol=5e-6)
                np.testing.assert_allclose(
                    system["R"][env, start].detach().cpu().numpy(),
                    data.efc_R[row], rtol=8e-5, atol=2e-6)
                np.testing.assert_allclose(
                    system["ar"][env, start].detach().cpu().numpy(),
                    data.efc_aref[row], rtol=8e-5, atol=2e-5)
                assert rows["active"][env, int(desc.row_start[slot])].item() > .5
                # The free carrier participates in every row. The hinge
                # participates across the manifold; individual source rows
                # legitimately have zero hinge moment at this exact pose.
                assert np.linalg.norm(pinned_j[row, :6]) > .1
                parent_hinge_terms.append(abs(pinned_j[row, int(model.jnt_dofadr[10])]))
        assert max(parent_hinge_terms) > .01
        assert set(np.flatnonzero(actual_active[env]).tolist()) == expected_slots

    initial = sim.snapshot()
    middle = None
    for step in range(2):
        status = sim.step()
        np.testing.assert_array_equal(status.detach().cpu().numpy(), [0, 0])
        for data in refs:
            mujoco.mj_step(model, data)
        if step == 0:
            force = sim.accepted_step["system"]["qfrc_constraint"].detach().cpu().numpy()
            for env, data in enumerate(refs):
                np.testing.assert_allclose(force[env], data.qfrc_constraint,
                                           rtol=5e-4, atol=2e-5)
            middle = sim.snapshot()
    final = sim.state.snapshot()
    for env, data in enumerate(refs):
        np.testing.assert_allclose(final.qpos[env], data.qpos,
                                   rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(final.qvel[env], data.qvel,
                                   rtol=2e-4, atol=2e-5)
    assert middle is not None
    sim.restore(middle)
    for _ in range(1):
        np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
    replay = sim.state.snapshot()
    np.testing.assert_array_equal(replay.qpos, final.qpos)
    np.testing.assert_array_equal(replay.qvel, final.qvel)
    sim.restore(initial)
    sim.reset(qpos=qpos, qvel=qvel)
    reset_refs = [mujoco.MjData(model) for _ in range(2)]
    for env, data in enumerate(reset_refs):
        data.qpos[:] = qpos[env].astype(np.float64)
        data.qvel[:] = qvel[env].astype(np.float64)
        mujoco.mj_step(model, data)
    np.testing.assert_array_equal(sim.step().detach().cpu().numpy(), [0, 0])
    reset_state = sim.state.snapshot()
    for env, data in enumerate(reset_refs):
        np.testing.assert_allclose(reset_state.qpos[env], data.qpos,
                                   rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(reset_state.qvel[env], data.qvel,
                                   rtol=2e-4, atol=2e-5)
