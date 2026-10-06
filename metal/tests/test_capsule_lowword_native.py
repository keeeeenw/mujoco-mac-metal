"""Private opt-in raw capsule primitive observer; not a production test yet."""
import os
from pathlib import Path

import numpy as np
import pytest

import mujoco
from mujoco_metal.flex_contact import (
    FlexContactProgram, lower_flex_contacts, _KIND_ELEMENT_PAIR,
    _KIND_INTERNAL_VERTEX_ELEMENT,
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


def _fixture(which):
    if which == "cross":
        flexes = _flexcomp("left", 0, 3, "none")
        flexes += _flexcomp("right", .03, 3, "none")
    elif which == "self":
        flexes = _flexcomp("strand", 0, 4, "narrow")
    elif which == "internal":
        flexes = _flexcomp("strand", 0, 4, "none").replace(
            'selfcollide="none"', 'selfcollide="none" internal="true"')
    else:
        raise ValueError(f"unknown fixture {which!r}")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><option gravity='0 0 0'/><worldbody>" + flexes + "</worldbody></mujoco>")
    data = mujoco.MjData(model)
    if which in ("self", "internal"):
        data.qpos[6] = -.1
        data.qpos[7] = .001
        data.qpos[9] = -.1
        data.qpos[10] = .001
    data.qpos[:] = data.qpos.astype(np.float32).astype(np.float64)
    data.qvel[:] = data.qvel.astype(np.float32).astype(np.float64)
    mujoco.mj_forward(model, data)
    return model, data


def _lowword_fixture():
    flex0 = """
      <flexcomp name="left" type="grid" count="3 1 1" spacing=".05 .05 .05"
                mass="1" dim="1" radius=".00500000012345">
        <contact contype="1" conaffinity="1" selfcollide="none"
                 margin=".000100000012345" gap=".000200000023456"/>
      </flexcomp>
    """
    flex1 = """
      <flexcomp name="right" type="grid" count="3 1 1" pos=".03 0 0"
                spacing=".05 .05 .05" mass="1" dim="1"
                radius=".00700000023456">
        <contact contype="1" conaffinity="1" selfcollide="none"
                 margin=".000300000034567" gap=".000400000045678"/>
      </flexcomp>
    """
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><option gravity='0 0 0'/><worldbody>" + flex0 + flex1
        + "</worldbody></mujoco>")
    data = mujoco.MjData(model)
    data.qpos[:] = data.qpos.astype(np.float32).astype(np.float64)
    data.qvel[:] = data.qvel.astype(np.float32).astype(np.float64)
    mujoco.mj_forward(model, data)
    return model, data


def _expected_contacts(model, data, descriptor, which):
    grouped = {}
    for contact in data.contact[:data.ncon]:
        f1, f2 = map(int, contact.flex)
        if which in ("cross", "lowword-cross") and (f1, f2) != (0, 1):
            continue
        if which in ("self", "internal") and (f1, f2) != (0, 0):
            continue
        key = ((int(contact.vert[0]), int(contact.elem[1])) if which == "internal"
               else (f1, int(contact.elem[0]), f2, int(contact.elem[1])))
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


def test_sync_world_mask_dims_returns_the_mutated_dimension_tensor():
    torch = pytest.importorskip("torch")
    program = FlexContactProgram.__new__(FlexContactProgram)
    program.batch_size = 2
    program._world_mask_current = torch.tensor([1, 1], dtype=torch.int32)
    dims = torch.tensor([4, 9, 0, 0], dtype=torch.int32)
    mask = torch.tensor([0, 1], dtype=torch.int32)

    result = program._sync_world_mask_dims(dims, 2, mask)

    assert result is dims
    torch.testing.assert_close(dims, torch.tensor([4, 9, 0, 1], dtype=torch.int32))


@pytest.mark.gpu
@pytest.mark.parametrize("which", ["cross", "self", "internal", "lowword-cross"])
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="private capsule observer requires parent GPU opt-in")
def test_native_flex_capsule_raw_primitive_observer(which):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("capsule observer requires MPS")
    model, data = (_lowword_fixture() if which == "lowword-cross"
                   else _fixture(which))
    descriptor = lower_flex_contacts(model)
    if which == "lowword-cross":
        assert np.any(descriptor.flex_radius_mid != 0)
        assert np.any(descriptor.flex_radius_low != 0)
        assert np.any(descriptor.margin_mid != 0)
        assert np.any(descriptor.gap_mid != 0)
    expected = _expected_contacts(model, data, descriptor, which)
    program = FlexContactProgram(model, device="mps", capture_ccd_trace=True)
    # These direct 1D capsule and internal point/edge fixtures have public
    # admission. The separate dense/CSR lifecycle gate qualifies internal
    # rows, force, trajectory, replay and reset; this observer keeps the raw
    # geometry and low-word checks without overriding production admission.
    assert program._narrowphase_admitted

    def mps(value):
        return torch.as_tensor(np.asarray(value, np.float32).copy(),
                               dtype=torch.float32, device="mps").contiguous()

    result = program.run_device(
        mps(data.flexvert_xpos[None]),
        mps(np.zeros((1, model.ngeom, 3), np.float32)),
        mps(np.tile(np.array([1, 0, 0, 0], np.float32),
                    (1, model.ngeom, 1))))
    arrays = {
        "active": result["active"].cpu().numpy(),
        "dist": result["dist"].cpu().numpy(),
        "pos": result["pos"].cpu().numpy(),
        "normal": result["normal"].cpu().numpy(),
        "status": result["narrowphase_status"].cpu().numpy(),
        "trace": result["ccd_trace"].cpu().numpy(),
        "vertices": np.asarray(data.flexvert_xpos, np.float32),
        "candidate_kind": np.asarray(descriptor.kind, np.int32),
        "candidate_flex1": np.asarray(descriptor.flex1, np.int32),
        "candidate_elem1": np.asarray(descriptor.elem1, np.int32),
        "candidate_vert1": np.asarray(descriptor.vert1, np.int32),
        "candidate_flex2": np.asarray(descriptor.flex2, np.int32),
        "candidate_elem2": np.asarray(descriptor.elem2, np.int32),
        "candidate_vert2": np.asarray(descriptor.vert2, np.int32),
        "candidate_geom": np.asarray(descriptor.geom, np.int32),
        "candidate_ordinal": np.asarray(descriptor.contact_ordinal, np.int32),
        "expected_slots": np.asarray(sorted(expected), np.int32),
    }
    trace = arrays["trace"][0]
    arrays["capsule_operands"] = trace[:, 16:34].copy()
    arrays["capsule_raw_count"] = trace[:, 34].copy()
    arrays["capsule_raw_contacts"] = trace[:, 222:250].reshape(-1, 4, 7).copy()
    outdir = os.getenv("MUJOCO_METAL_FLEX_CAPSULE_TRACE_DIR")
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(outdir) / f"capsule-{which}.npz", **arrays)

    # Keep the strict existing gate. The observer never accepts missing or
    # numerically mismatched raw contacts.
    active = arrays["active"][0]
    assert set(map(int, np.flatnonzero(active))) == set(expected)
    for slot, contact in expected.items():
        assert arrays["status"][0, slot] == 0
        np.testing.assert_allclose(arrays["dist"][0, slot], contact.dist,
                                   rtol=0, atol=6e-6)
        np.testing.assert_allclose(arrays["pos"][0, slot], contact.pos,
                                   rtol=0, atol=6e-6)
        np.testing.assert_allclose(arrays["normal"][0, slot], contact.frame[:3],
                                   rtol=0, atol=6e-5)
