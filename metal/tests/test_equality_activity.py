# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public equality activation and complete state ownership (CPU + GPU)."""

import os

import mujoco
import numpy as np
import pytest

XML_CONNECT = """<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
<body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
<body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
</worldbody><equality><connect body1="b1" body2="b2" anchor="0.5 0 0"/></equality></mujoco>"""

XML_WELD = """<mujoco><option timestep="0.002" gravity="0 0 0"/>
<worldbody>
<body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
<body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
</worldbody><equality><weld body1="b1" body2="b2"/></equality></mujoco>"""


def test_device_state_eq_defaults_and_reset_override_cpu():
    pytest.importorskip("torch")
    from mujoco_metal.device_state import DeviceState
    from mujoco_metal.stepping import validate_stepping_profile

    m = mujoco.MjModel.from_xml_string(XML_CONNECT)
    profile = validate_stepping_profile(m, profile="integrated_euler_v1")
    st = DeviceState(m, profile, 2, device="cpu")
    assert st.neq == 1
    np.testing.assert_array_equal(st.eq_active.numpy(), [[1], [1]])
    st.set_equality_active(np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(st.eq_active.numpy(), [[0], [0]])
    st.reset(env_ids=[0])
    # Selected world back to defaults, unselected unchanged.
    np.testing.assert_array_equal(st.eq_active.numpy(), [[1], [0]])
    st.reset(env_ids=[1], eq_active=np.array([[0]], dtype=np.int32))
    np.testing.assert_array_equal(st.eq_active.numpy(), [[1], [0]])
    snap = st.snapshot()
    assert snap.schema_version == 6 and snap.neq == 1
    st.set_equality_active(np.array([[0], [0]], dtype=np.int32))
    st.restore(snap)
    np.testing.assert_array_equal(st.eq_active.numpy(), [[1], [0]])


def test_device_state_eq_validation_atomic_cpu():
    pytest.importorskip("torch")
    from mujoco_metal.device_state import DeviceState
    from mujoco_metal.stepping import validate_stepping_profile

    m = mujoco.MjModel.from_xml_string(XML_CONNECT)
    profile = validate_stepping_profile(m, profile="integrated_euler_v1")
    st = DeviceState(m, profile, 2, device="cpu")
    before = st.snapshot()
    with pytest.raises(ValueError):
        st.set_equality_active(np.array([2], dtype=np.int32))
    with pytest.raises(ValueError):
        st.set_equality_active(np.array([[1, 0]], dtype=np.int32))
    with pytest.raises(ValueError):
        st.reset(env_ids=[0], eq_active=np.array([5], dtype=np.int32))
    after = st.snapshot()
    np.testing.assert_array_equal(after.eq_active, before.eq_active)
    np.testing.assert_array_equal(after.qpos, before.qpos)


def test_snapshot_old_schema_rejected_for_equalities_cpu():
    pytest.importorskip("torch")
    from mujoco_metal.device_state import DeviceState, StateSnapshot
    from mujoco_metal.stepping import validate_stepping_profile

    m = mujoco.MjModel.from_xml_string(XML_CONNECT)
    profile = validate_stepping_profile(m, profile="integrated_euler_v1")
    st = DeviceState(m, profile, 1, device="cpu")
    # Manually craft a v1 snapshot (no activity) with matching fingerprints.
    v1 = st.snapshot()
    assert v1.schema_version == 6  # current state has epochs plus warnings
    # Forge a v1 with same physical fields but schema 1.
    forged = StateSnapshot(
        model_fingerprint=v1.model_fingerprint,
        profile_fingerprint=v1.profile_fingerprint,
        timestep=v1.timestep, nq=v1.nq, nv=v1.nv, batch_size=v1.batch_size,
        qpos=v1.qpos, qvel=v1.qvel, qacc=v1.qacc, time=v1.time, status=v1.status,
        schema_version=1,
    )
    with pytest.raises(ValueError, match="no equality activity"):
        st.restore(forged)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_public_activity_release_reattach_and_cache_invalidation():
    from mujoco_metal.simulation import MetalSimulation

    m = mujoco.MjModel.from_xml_string(XML_CONNECT)
    sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
    # Initially active; release world 1 via public API (host broadcast + per-world).
    sim.set_equality_active(np.array([0], dtype=np.int32), env_ids=[1])
    np.testing.assert_array_equal(sim.state.eq_active.cpu().numpy(), [[1], [0]])
    # Cached assembly must invalidate: first call caches, second after change recomputes.
    a1 = sim.assembled_system()
    # Touch cache (second call without recompute returns same object).
    a_cached = sim.assembled_system()
    assert a_cached is a1
    sim.set_equality_active(np.array([[1], [1]], dtype=np.int32))
    a2 = sim.assembled_system()
    assert a2 is not a1
    # Zero stale disabled-row forces: release world 0, recompute, check joint force zero.
    sim.set_equality_active(np.array([[0], [1]], dtype=np.int32))
    a_rel = sim.assembled_system(recompute=True)
    # Equality rows are 0..2 (connect span 3); world 0 disabled => J zero, lambda zero.
    J0 = a_rel["J"][0].cpu().numpy()
    lam0 = a_rel["lambda"][0].cpu().numpy()
    assert np.all(J0[:3] == 0)
    assert np.all(lam0[:3] == 0)
    # Reattach uses compiled reference (anchor still constrained after stepping).
    sim.set_equality_active(np.array([[1], [1]], dtype=np.int32))
    sim.step(5)
    assert sim.state.status[0].item() == 0
    # Device input path (contiguous int32 MPS) works and is copied (borrow check).
    import torch

    dev = torch.tensor([[0], [1]], dtype=torch.int32, device="mps")
    sim.set_equality_active(dev, env_ids=[0, 1])
    dev[0, 0] = 1  # mutating input after call must not affect persistent mask.
    np.testing.assert_array_equal(sim.state.eq_active.cpu().numpy(), [[0], [1]])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_state_lifecycle_snapshot_restore_replay_with_masks():
    from mujoco_metal.simulation import MetalSimulation

    m = mujoco.MjModel.from_xml_string(XML_WELD)
    sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
    sim.set_equality_active(np.array([[1], [0]], dtype=np.int32))
    sim.step(3)
    snap = sim.state.snapshot()
    assert snap.schema_version == 2
    sim.step(3)
    q_after = sim.state.qpos.clone()
    sim.state.restore(snap)
    # Masks survive restore and replay reproduces attach/release trajectory.
    np.testing.assert_array_equal(sim.state.eq_active.cpu().numpy(), [[1], [0]])
    sim.step(3)
    assert (sim.state.qpos == q_after).all()
    # Selected reset restores defaults for selected worlds only.
    sim.state.reset(env_ids=[0])
    np.testing.assert_array_equal(sim.state.eq_active.cpu().numpy(), [[1], [0]])
    sim.state.reset(env_ids=[1])
    np.testing.assert_array_equal(sim.state.eq_active.cpu().numpy(), [[1], [1]])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_failure_rollback_sticky_and_activity_persistence():
    import torch
    from mujoco_metal.simulation import MetalSimulation

    m = mujoco.MjModel.from_xml_string(XML_CONNECT)
    sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")
    # Force failure in world 1 via NaN control? Connect model has no actuators;
    # use NaN qfrc_applied to fail world 1 only (device path, per-world failure).
    good = torch.zeros((2, m.nv), dtype=torch.float32, device="mps")
    bad = good.clone()
    bad[1, 0] = float("nan")
    sim.step(1, qfrc_applied=bad)
    status = sim.state.status.cpu().numpy()
    assert status[0] == 0 and status[1] != 0
    qpos_fail = sim.state.qpos.clone()
    time_fail = sim.state.time.clone()
    # Explicit activity change is persistent input, must not clear sticky failure.
    sim.set_equality_active(np.array([[0], [0]], dtype=np.int32))
    assert sim.state.status.cpu().numpy()[1] != 0
    np.testing.assert_array_equal(sim.state.eq_active.cpu().numpy(), [[0], [0]])
    # Failed world does not advance on next step; healthy world does.
    sim.step(1, qfrc_applied=good)
    assert sim.state.status.cpu().numpy()[1] != 0
    assert (sim.state.qpos[1] == qpos_fail[1]).all()
    assert (sim.state.time[1] == time_fail[1]).all()
    # Recovery via selective reset clears failure and restores defaults.
    sim.state.reset(env_ids=[1])
    assert sim.state.status.cpu().numpy()[1] == 0
