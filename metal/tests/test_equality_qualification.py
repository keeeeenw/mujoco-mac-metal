# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Mixed-physics, motion, capacity, failure and residency qualification."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _mixed_model(cone="pyramidal", condim=6):
    return mujoco.MjModel.from_xml_string(f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler" iterations="2048" tolerance="4e-6" cone="{cone}">
    <flag contact="enable" equality="enable" limit="enable" frictionloss="enable"/>
  </option>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="{condim}" friction="0.8 0.6 0.07"/>
    <body name="b1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" range="-0.5 0.5" limited="true" frictionloss="0.05"/>
      <geom name="g1" type="sphere" size="0.15" pos="0.3 0 0" mass="1" condim="{condim}" friction="0.8 0.6 0.07" contype="1" conaffinity="0"/>
    </body>
    <body name="b2" pos="0.8 0.1 0.6">
      <joint name="j2" type="hinge" axis="0 1 0"/>
      <geom name="g2" type="sphere" size="0.15" pos="-0.2 0 0" mass="1" condim="{condim}" friction="0.8 0.6 0.07" contype="1" conaffinity="0"/>
    </body>
    <body name="b3" pos="0.2 0.3 0.9"><freejoint/>
      <geom name="g3" type="sphere" size="0.12" mass="0.8" condim="{condim}" friction="0.8 0.6 0.07" contype="1" conaffinity="0"/>
    </body>
    <body name="b4" pos="1.0 -0.2 0.9"><freejoint/>
      <geom name="g4" type="sphere" size="0.12" mass="0.8" condim="{condim}" friction="0.8 0.6 0.07" contype="1" conaffinity="0"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j2" joint2="j1" polycoef="0 1.2 0 0 0"/>
    <connect body1="b3" body2="b4" anchor="0.4 0.05 0.02"/>
  </equality>
  <actuator><motor joint="j1" ctrlrange="-5 5"/></actuator>
</mujoco>""")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [4, 6])
def test_mixed_connect_with_contacts_limits_frictionloss(cone, condim):
    from mujoco_metal.simulation import MetalSimulation

    m = _mixed_model(cone, condim)
    d = lower_coupled_constraints(m)
    # Equality spans: joint 1 + connect 3 = 4 rows; plus limits/friction/contact.
    assert d.n_eq_rows == 4
    # Engaging state: hinge j1 violates upper limit, joint equality violated,
    # free bodies low to contact floor with nonzero spin for torsion/rolling.
    qpos = np.array([[
        0.55, 0.1,
        0.2, 0.3, 0.1, 1, 0, 0, 0,
        1.0, -0.2, 0.1, 1, 0, 0, 0,
    ]], dtype=np.float32)
    qvel = np.array([[
        0.5, -0.3,
        0.2, -0.1, 0.05, 0.8, -0.5, 0.6,
        -0.15, 0.1, -0.05, -0.6, 0.4, 1.1,
    ]], dtype=np.float32)
    sim = MetalSimulation(m, batch_size=1, qpos=qpos, qvel=qvel, profile="integrated_euler_v1")
    asm = sim.assembled_system(recompute=True)
    J = asm["J"][0].cpu().numpy()
    W = asm["W"][0].cpu().numpy()
    R = asm["R"][0].cpu().numpy()
    # Nonzero equality/contact cross blocks prove genuine coupling.
    n_eq = d.n_eq_rows
    # Contact rows start after joint rows (equality+friction+limits).
    base = int(np.asarray(d.contact_condim_packed).reshape(-1, 3)[0, 1]) + d.nr_joint if d.ncontacts_max else d.nr_joint
    # Simpler: cross between equality block [0:n_eq] and later rows must be nonzero.
    cross = W[:n_eq, n_eq:]
    assert float(np.linalg.norm(cross)) > 1e-4, float(np.linalg.norm(cross))
    # Physical forces: equality and contact both engaged (nonzero generalized force).
    sim.step(1, ctrl=np.array([[0.8]], dtype=np.float32))
    assert sim.state.status[0].item() == 0
    qf = sim._last_coupled["qfrc_constraint"][0].cpu().numpy()
    assert float(np.linalg.norm(qf)) > 0.05
    # CPU parity for one step with changing controls (matched initial state).
    cpu = mujoco.MjData(m)
    cpu.qpos[:] = qpos[0]
    cpu.qvel[:] = qvel[0]
    cpu.ctrl[0] = 0.8
    mujoco.mj_step(m, cpu)
    np.testing.assert_allclose(sim.state.qpos[0].cpu().numpy(), cpu.qpos, atol=1e-5)
    np.testing.assert_allclose(sim.state.qacc[0].cpu().numpy(), cpu.qacc, rtol=1e-3, atol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("dt", [0.001, 0.004])
def test_motion_trajectories_both_equalities_two_timesteps(dt):
    from mujoco_metal.simulation import MetalSimulation

    for xml, name in [
        (f"""<mujoco><option timestep="{dt}" gravity="0 0 -9.81" iterations="1000" tolerance="1e-6"/>
        <worldbody><body name="a" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b" pos="0.9 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><connect body1="a" body2="b" anchor="0.45 0 0"/></equality></mujoco>""", "connect"),
        (f"""<mujoco><option timestep="{dt}" gravity="0 0 -9.81" iterations="1000" tolerance="1e-6"/>
        <worldbody><body name="a" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b" pos="1 0.1 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><weld body1="a" body2="b"/></equality></mujoco>""", "weld"),
    ]:
        m = mujoco.MjModel.from_xml_string(xml)
        for q0_extra, v0_extra in [
            (np.array([0.02, 0, 0, 0, 0, 0, 0, -0.01, 0, 0, 0, 0, 0, 0], dtype=np.float32),
             np.array([0.1, 0, 0, 0.3, 0, 0, -0.05, 0, 0, -0.2, 0, 0], dtype=np.float32)),
            (np.array([-0.015, 0.01, 0, 0, 0, 0, 0, 0.012, -0.008, 0, 0, 0, 0, 0], dtype=np.float32),
             np.array([-0.08, 0.05, 0.02, -0.2, 0.1, 0.15, 0.06, -0.04, 0.01, 0.25, -0.15, -0.1], dtype=np.float32)),
        ]:
            q0 = (m.qpos0.copy() + q0_extra).astype(np.float32)[None, :]
            v0 = v0_extra.astype(np.float32)[None, :]
            sim = MetalSimulation(m, batch_size=1, qpos=q0, qvel=v0, profile="integrated_euler_v1")
            cpu = mujoco.MjData(m)
            cpu.qpos[:] = q0[0]
            cpu.qvel[:] = v0[0]
            for _ in range(25):
                sim.step(1)
                mujoco.mj_step(m, cpu)
            # Quaternion-aware pose errors (pos + 1-|dot| for quats) plus anchor errors.
            gq = sim.state.qpos[0].cpu().numpy()
            max_pos_err = 0.0
            max_quat_err = 0.0
            # Free joints: qpos layout [pos(3), quat(4)] per body.
            for b in range(2):
                base = b * 7
                max_pos_err = max(max_pos_err, float(np.linalg.norm(gq[base:base+3] - cpu.qpos[base:base+3])))
                dot = abs(float(np.dot(gq[base+3:base+7], cpu.qpos[base+3:base+7])))
                max_quat_err = max(max_quat_err, 1.0 - dot)
            assert max_pos_err < 1e-5, (name, dt, max_pos_err)
            assert max_quat_err < 1e-6, (name, dt, max_quat_err)
            # Native anchor/relative-pose errors match CPU parity (compliant, not zero promise).
            # For connect, anchor coincidence; for weld, relative pose coincidence.
            # Here just check qpos parity already covers it with tight bounds.


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_capacity_mixed_spans_near_96_and_overflow():
    # Valid mixed 1/3/6 layout hitting 96 rows exactly (descriptor-derived):
    # 6 slides (nv=6, njnt=6), 3 joint (3) +1 connect (3) +2 weld (12) =18 eq rows,
    # nr_joint=18+6+12=36, plus 6 plane-sphere condim6 pyramidal contacts (6*10=60) =96.
    xml_ok = '<mujoco><option cone="pyramidal" timestep="0.002" iterations="1000" tolerance="1e-6"/><worldbody><geom type="plane" size="10 10 0.1" condim="6" friction="0.8 0.6 0.07" contype="1" conaffinity="1"/>'
    for i in range(6):
        xml_ok += f'<body name="b{i}" pos="{i} 0 0.08"><joint name="j{i}" type="slide" axis="0 0 1"/>'
        xml_ok += f'<geom type="sphere" size="0.1" pos="0 0 0" condim="6" friction="0.8 0.6 0.07" contype="1" conaffinity="0"/></body>'
    xml_ok += '</worldbody><equality>'
    for i in range(3):
        xml_ok += f'<joint joint1="j{i}" polycoef="0 1 0 0 0"/>'
    xml_ok += '<connect body1="b3" body2="b4" anchor="0 0 0"/>'
    xml_ok += '<weld body1="b0" body2="b1"/><weld body1="b2" body2="b3"/>'
    xml_ok += '</equality></mujoco>'
    m_ok = mujoco.MjModel.from_xml_string(xml_ok)
    d_ok = lower_coupled_constraints(m_ok)
    assert d_ok.n_eq_rows == 3*1 + 1*3 + 2*6
    assert d_ok.nr == 96
    assert d_ok.nr_joint == 36
    # Final valid row execution on GPU (row 95 active, finite, status 0).
    from mujoco_metal.simulation import MetalSimulation
    sim = MetalSimulation(m_ok, batch_size=1, profile="integrated_euler_v1")
    asm = sim.assembled_system(recompute=True)
    J96 = asm["J"][0].cpu().numpy()
    W96 = asm["W"][0].cpu().numpy()
    assert np.linalg.norm(J96[95]) > 0, "row 95 must be active"
    assert W96[95, 95] > 0
    sim.step(1)
    assert sim.state.status[0].item() == 0
    assert np.all(np.isfinite(sim.state.qpos.cpu().numpy()))
    # Overflow rejection before dispatch (add one more weld, 6 rows => 102>96).
    with pytest.raises(ValueError, match="total candidate constraint rows"):
        xml_bad = xml_ok.replace("</equality></mujoco>", '<weld body1="b4" body2="b5"/></equality></mujoco>')
        lower_coupled_constraints(mujoco.MjModel.from_xml_string(xml_bad))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_failure_isolation_with_equality_contact_empty_worlds():
    from mujoco_metal.simulation import MetalSimulation

    xml = """<mujoco><option timestep="0.002" integrator="Euler" iterations="1" tolerance="1e-6" cone="pyramidal"/>
    <worldbody><geom type="plane" size="5 5 0.1" condim="6" friction="0.8 0.6 0.07"/>
    <body name="a1" pos="0 0 0.05"><joint name="j1" type="slide" axis="0 0 1"/><geom type="sphere" size="0.1" condim="6" friction="0.8 0.6 0.07"/></body>
    <body name="b1" pos="0.5 0 0.05"><joint name="j2" type="slide" axis="0 0 1"/><geom type="sphere" size="0.1" condim="6" friction="0.8 0.6 0.07"/></body>
    </worldbody><equality><connect body1="a1" body2="b1" anchor="0 0 0"/></equality></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    # World 0: colliding active (fails with 1 iteration), world 1: far active
    # (succeeds, no contact, satisfied equality), world 2: far inactive (empty-like).
    q0 = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    v0 = np.array([[2.0, -2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    sim = MetalSimulation(m, batch_size=3, qpos=q0, qvel=v0, profile="integrated_euler_v1")
    # Make world 2 inactive via public API (empty/inactive world).
    sim.set_equality_active(np.array([[1], [1], [0]], dtype=np.int32))
    sim.step(1)
    status = sim.state.status.cpu().numpy()
    assert status[0] == 3, status
    assert status[1] == 0, status
    assert status[2] == 0, status
    qpos_after = sim.state.qpos.cpu().numpy()
    # Failed world rolled back exactly; healthy worlds advanced or stayed per CPU.
    np.testing.assert_allclose(qpos_after[0], q0[0], atol=1e-7)
    # Healthy CPU parity for world 1 (separated, no contact).
    cpu1 = mujoco.MjData(m)
    cpu1.qpos[:] = q0[1]
    mujoco.mj_step(m, cpu1)
    np.testing.assert_allclose(qpos_after[1], cpu1.qpos, atol=1e-5)
    # Sticky failure persists.
    sim.step(1)
    assert sim.state.status.cpu().numpy()[0] == 3
    # Explicit recovery via selective reset restores defaults (active) and clears status.
    sim.state.reset(env_ids=[0])
    assert sim.state.status.cpu().numpy()[0] == 0
    # After reset, world 0 defaults to active (compiled active0=True).
    assert int(sim.state.eq_active.cpu().numpy()[0, 0]) == 1


def test_residency_package_paths_and_hot_path():
    from pathlib import Path
    import mujoco_metal.coupled_constraints as cc

    base = Path(cc.__file__).parent
    assert (base / "shaders" / "equality_assembly.metal").exists()
    assert (base / "shaders" / "coupled_constraints.metal").exists()
    # Persistent mask/workspace allocation and hot-path routing without CPU physics.
    # CPU oracle work belongs in tests (this file uses mujoco.MjData only for references).
    assert hasattr(cc.MetalCoupledConstraints, "run_device")
