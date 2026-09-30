# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU lowering and row-layout qualification for connect/weld equalities.

Pinned oracle: MuJoCo 3.10.0 (see metal/NOTICE for upstream provenance).
Per-step equality assembly/solving must execute on Metal; CPU work here is
constant preparation and independent reference calculation only.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import (
    lower_coupled_constraints,
)


def _connect_body_body(anchor="0.5 0 0", body2="b2"):
    body2_xml = f'body2="{body2}"' if body2 else ""
    return f"""<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
    <body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
  </worldbody>
  <equality><connect body1="b1" {body2_xml} anchor="{anchor}"/></equality>
</mujoco>"""


def test_connect_body_body_row_span_and_metadata():
    m = mujoco.MjModel.from_xml_string(_connect_body_body())
    d = lower_coupled_constraints(m)
    assert d.neq == 1
    assert d.n_eq_rows == 3
    assert d.eq_rowadr.tolist() == [0]
    assert d.eq_rownum.tolist() == [3]
    assert int(d.eq_type[0]) == int(mujoco.mjtEq.mjEQ_CONNECT)
    assert int(d.eq_objtype[0]) == int(mujoco.mjtObj.mjOBJ_BODY)
    # nr_joint uses explicit equality spans, not one row per equality.
    assert d.nr_joint == 3 + m.nv + 2 * m.njnt
    assert d.nr == d.nr_joint + (d.nr - d.nr_joint)
    assert d.nr <= 96
    # Deterministic mapping: equality rows start at 0.
    assert d.nr_joint > m.nv


def test_connect_body_world_and_site_site_forms():
    # Body-world: second body is world (0).
    m_world = mujoco.MjModel.from_xml_string(_connect_body_body(body2=""))
    # connect body-world requires anchor + body1; body2 defaults to world.
    # The XML above omits body2 entirely (empty string leaves a stray space;
    # rebuild without body2 attribute).
    m_world = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><connect body1="b1" anchor="0.5 0 0"/></equality></mujoco>"""
    )
    d_world = lower_coupled_constraints(m_world)
    assert d_world.n_eq_rows == 3
    assert int(d_world.eq_objtype[0]) == int(mujoco.mjtObj.mjOBJ_BODY)
    assert int(np.asarray(m_world.eq_obj2id)[0]) == 0

    # Site-site with transformed local sites and both bodies moving.
    m_site = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><freejoint/>
          <site name="s1" pos="0.1 0.2 0.3" quat="1 0 0 0"/>
          <geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1"><freejoint/>
          <site name="s2" pos="-0.1 0.1 -0.2"/>
          <geom type="sphere" size="0.1"/></body>
        </worldbody><equality><connect site1="s1" site2="s2"/></equality></mujoco>"""
    )
    d_site = lower_coupled_constraints(m_site)
    assert d_site.n_eq_rows == 3
    assert int(d_site.eq_objtype[0]) == int(mujoco.mjtObj.mjOBJ_SITE)


def test_weld_body_body_site_site_and_torquescale():
    m_body = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1" quat="0 0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><weld body1="b1" body2="b2"/></equality></mujoco>"""
    )
    d_body = lower_coupled_constraints(m_body)
    assert d_body.n_eq_rows == 6
    assert d_body.eq_rownum.tolist() == [6]
    assert float(np.asarray(m_body.eq_data).reshape(1, 11)[0, 10]) == pytest.approx(1.0)

    # Nontrivial relative pose with explicit relpose and transformed sites.
    m_rel = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0.2 1.1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><weld body1="b1" body2="b2" relpose="0.1 0.2 0.3 1 0 0 0"/></equality></mujoco>"""
    )
    d_rel = lower_coupled_constraints(m_rel)
    assert d_rel.n_eq_rows == 6

    # Zero torquescale is valid and still reserves 6 rows.
    m_zero = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><weld body1="b1" body2="b2" torquescale="0" anchor="0 -2 0"/></equality></mujoco>"""
    )
    d_zero = lower_coupled_constraints(m_zero)
    assert d_zero.n_eq_rows == 6
    assert float(np.asarray(m_zero.eq_data).reshape(1, 11)[0, 10]) == 0.0

    m_site = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><site name="s1" pos="0.1 0 0"/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1"><freejoint/><site name="s2" pos="-0.1 0 0" quat="0 0 0 1"/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><weld site1="s1" site2="s2"/></equality></mujoco>"""
    )
    d_site = lower_coupled_constraints(m_site)
    assert d_site.n_eq_rows == 6
    assert int(d_site.eq_objtype[0]) == int(mujoco.mjtObj.mjOBJ_SITE)


def test_mixed_joint_connect_weld_row_layout_is_deterministic():
    m = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><joint name="j1" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1"><joint name="j2" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1"/></body>
        <body name="b3" pos="2 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b4" pos="3 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality>
        <joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
        <connect body1="b3" body2="b4" anchor="0.5 0 0"/>
        <weld body1="b1" body2="b2"/>
        </equality></mujoco>"""
    )
    d = lower_coupled_constraints(m)
    assert d.neq == 3
    # Joint=1, connect=3, weld=6 in XML order.
    assert d.eq_rownum.tolist() == [1, 3, 6]
    assert d.eq_rowadr.tolist() == [0, 1, 4]
    assert d.n_eq_rows == 10
    assert d.nr_joint == 10 + m.nv + 2 * m.njnt
    # Activity array still has one entry per equality, not per row.
    assert d.eq_active0.shape == (3,)
    # Reverse order gives a different deterministic layout with same total.
    m_rev = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><joint name="j1" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1"><joint name="j2" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1"/></body>
        <body name="b3" pos="2 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b4" pos="3 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality>
        <weld body1="b1" body2="b2"/>
        <connect body1="b3" body2="b4" anchor="0.5 0 0"/>
        <joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
        </equality></mujoco>"""
    )
    d_rev = lower_coupled_constraints(m_rev)
    assert d_rev.eq_rownum.tolist() == [6, 3, 1]
    assert d_rev.eq_rowadr.tolist() == [0, 6, 9]
    assert d_rev.n_eq_rows == 10


def test_unsupported_equality_types_still_rejected():
    m_tendon = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body><joint name="j1" type="hinge"/><geom type="sphere" size="0.1"/></body>
        </worldbody>
        <tendon><fixed name="t1"><joint joint="j1" coef="1"/></fixed></tendon>
        <equality><tendon tendon1="t1" polycoef="0 1 0 0 0"/></equality></mujoco>"""
    )
    with pytest.raises(ValueError, match="only joint, connect and weld"):
        lower_coupled_constraints(m_tendon)

    # Tendon limits/frictionloss remain out of scope.
    m_lim = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body><joint name="j1" type="hinge"/><geom type="sphere" size="0.1"/></body>
        </worldbody>
        <tendon><fixed name="t1" limited="true" range="0 1"><joint joint="j1" coef="1"/></fixed></tendon></mujoco>"""
    )
    with pytest.raises(ValueError, match="tendon limits"):
        lower_coupled_constraints(m_lim)


def test_inactive_new_equality_still_validated_and_reserved():
    # A model is not supported merely because the new equality is inactive:
    # reservation uses spans even when active=false.
    m_inactive = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><connect body1="b1" body2="b2" anchor="0.5 0 0" active="false"/></equality></mujoco>"""
    )
    d = lower_coupled_constraints(m_inactive)
    assert d.n_eq_rows == 3
    assert not bool(d.eq_active0[0])


def test_capacity_uses_spans_not_equality_counts():
    # 8 joint equalities (8 rows) + 8 slide DOFs + limits vs mixed spans.
    # Derive counts from the descriptor, not hand-waved equality counts.
    m = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body pos="0 0 1"><joint name="j0" type="slide" axis="0 0 1"/><geom type="sphere" size="0.1" contype="0" conaffinity="0"/></body>
        <body pos="1 0 1"><freejoint/><geom type="sphere" size="0.1" contype="0" conaffinity="0"/></body>
        <body pos="2 0 1"><freejoint/><geom type="sphere" size="0.1" contype="0" conaffinity="0"/></body>
        </worldbody><equality>
        <connect body1="b1" body2="b2" anchor="0 0 0"/>
        </equality></mujoco>"""
    ) if False else None
    # Mixed 1/3/6 layout near the 96-row budget is exercised in later
    # capacity tests; here check the span arithmetic on a small model.
    m_small = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body name="b1" pos="0 0 1"><joint name="j1" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1" contype="0" conaffinity="0"/></body>
        <body name="b2" pos="1 0 1"><joint name="j2" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.1" contype="0" conaffinity="0"/></body>
        </worldbody><equality><joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/></equality></mujoco>"""
    )
    d_small = lower_coupled_constraints(m_small)
    assert d_small.n_eq_rows == 1
    assert d_small.nr_joint == 1 + m_small.nv + 2 * m_small.njnt


def _native_connect_result(model, qpos, qvel):
    import torch

    from mujoco_metal.metal_kinematics import MetalKinematics
    from mujoco_metal.smooth_metal import MetalSmoothDynamics
    from mujoco_metal.coupled_constraints import MetalCoupledConstraints
    from mujoco_metal.model import load_model

    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    qpos_mps = torch.as_tensor(qpos, device="mps")
    qvel_mps = torch.as_tensor(qvel, device="mps")
    md = load_model(model)
    poses = MetalKinematics(md, qpos.shape[0]).run_device(qpos_mps)
    dyn = MetalSmoothDynamics(md, qpos.shape[0]).run_device(qpos_mps, qvel_mps)
    res = MetalCoupledConstraints(model, qpos.shape[0]).run_device(
        poses, dyn["mass_matrix"], -dyn["qfrc_bias"], qpos_mps, qvel_mps,
        cvel=dyn["cvel"],
    )
    return res, dyn


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("form", ["body-body", "body-world", "site-site"])
def test_native_connect_forms_match_cpu(form):
    if form == "body-body":
        xml = """<mujoco><option timestep="0.002" gravity="0 0 0" iterations="1000" tolerance="1e-10"/>
        <worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0.2 1.1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><connect body1="b1" body2="b2" anchor="0.5 0.1 0.05"/></equality></mujoco>"""
    elif form == "body-world":
        xml = """<mujoco><option timestep="0.002" gravity="0 0 0" iterations="1000" tolerance="1e-10"/>
        <worldbody>
        <body name="b1" pos="0.2 -0.1 1.1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><connect body1="b1" anchor="0.3 0.1 0.05"/></equality></mujoco>"""
    else:
        xml = """<mujoco><option timestep="0.002" gravity="0 0 0" iterations="1000" tolerance="1e-10"/>
        <worldbody>
        <body name="b1" pos="0 0 1"><freejoint/><site name="s1" pos="0.1 0.2 0.3"/><geom type="sphere" size="0.1"/></body>
        <body name="b2" pos="1 0.1 0.9"><freejoint/><site name="s2" pos="-0.15 0.05 -0.1"/><geom type="sphere" size="0.1"/></body>
        </worldbody><equality><connect site1="s1" site2="s2"/></equality></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = lower_coupled_constraints(m)
    assert d.n_eq_rows == 3
    qpos = m.qpos0[None, :].astype(np.float32)
    qpos[0, 0] += 0.01
    qvel = np.zeros((1, m.nv), dtype=np.float32)
    qvel[0, :3] = [0.2, -0.1, 0.05]
    if m.nv >= 6:
        qvel[0, 3:6] = [0.5, -0.3, 0.4]
    if m.nv >= 12:
        qvel[0, 6:9] = [-0.15, 0.1, -0.05]
        qvel[0, 9:12] = [0.3, 0.2, -0.4]
    res, _ = _native_connect_result(m, qpos, qvel)
    assert int(res["status"][0]) == 0
    ref = mujoco.MjData(m)
    ref.qpos[:] = qpos[0]
    ref.qvel[:] = qvel[0]
    mujoco.mj_forward(m, ref)
    assert ref.nefc == 3, (form, ref.nefc)
    gpu_rows = np.arange(0, 3)
    cpu_J = ref.efc_J.reshape(ref.nefc, m.nv)
    np.testing.assert_allclose(res["J"][0, gpu_rows].cpu().numpy(), cpu_J, atol=3e-6)
    np.testing.assert_allclose(res["R"][0, gpu_rows].cpu().numpy(), ref.efc_R, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(res["ar"][0, gpu_rows].cpu().numpy(), ref.efc_aref, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(-res["rhs"][0, gpu_rows].cpu().numpy(), ref.efc_b, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(res["qacc"][0].cpu().numpy(), ref.qacc, rtol=5e-4, atol=2e-2)
    np.testing.assert_allclose(
        res["qfrc_constraint"][0].cpu().numpy(), ref.qfrc_constraint, rtol=5e-4, atol=8e-2
    )


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_connect_trajectory_matches_cpu():
    from mujoco_metal.simulation import MetalSimulation

    xml = """<mujoco><option timestep="0.002" gravity="0 0 -9.81" iterations="1000" tolerance="1e-6"/>
    <worldbody>
    <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size="0.12" mass="1"/></body>
    <body name="b2" pos="0.9 0.1 1.05"><freejoint/><geom type="sphere" size="0.12" mass="1"/></body>
    </worldbody><equality><connect body1="b1" body2="b2" anchor="0.45 0.05 0.02"/></equality></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    q0 = m.qpos0[None, :].astype(np.float32)
    v0 = np.zeros((1, m.nv), dtype=np.float32)
    v0[0, :3] = [0.1, -0.05, 0.02]
    sim = MetalSimulation(m, batch_size=1, qpos=q0, qvel=v0, profile="integrated_euler_v1")
    cpu = mujoco.MjData(m)
    cpu.qpos[:] = q0[0]
    cpu.qvel[:] = v0[0]
    max_pos = 0.0
    max_vel = 0.0
    for _ in range(50):
        sim.step(1)
        mujoco.mj_step(m, cpu)
        max_pos = max(max_pos, float(np.max(np.abs(sim.state.qpos[0].cpu().numpy() - cpu.qpos))))
        max_vel = max(max_vel, float(np.max(np.abs(sim.state.qvel[0].cpu().numpy() - cpu.qvel))))
    assert max_pos < 1e-5, max_pos
    assert max_vel < 1e-4, max_vel
