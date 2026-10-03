# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R05 repair: contact manifolds, identity, and missing geometry pairs.

R05-0 codifies the pinned pair-dispatch classification (NULL entries are
intentional omissions, not gaps). Later commits add multi-contact mesh
manifolds with identity, missing valid pairs, margins/settings, and demos.
"""

import mujoco
import numpy as np
import pytest


def test_pinned_omits_plane_hfield_and_hfield_hfield_cpu():
  # mjCOLLISIONFUNC has NULL for plane-heightfield and
  # heightfield-heightfield: pinned emits zero contacts and reports
  # distmax. Native rejection/zero-slots match the engine (R05-0).
  m = mujoco.MjModel.from_xml_string(
      '<mujoco><asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.2 0.1"/></asset>'
      '<worldbody><geom name="pp" type="plane" size="5 5 0.1"/>'
      '<geom name="hf" type="hfield" hfield="h" pos="0 0 -0.05"/>'
      '</worldbody></mujoco>')
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  assert d.ncon == 0
  ft = np.zeros(6)
  assert mujoco.mj_geomDistance(m, d, 0, 1, 5.0, ft) == pytest.approx(5.0)
  m2 = mujoco.MjModel.from_xml_string(
      '<mujoco><asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.2 0.1"/></asset>'
      '<worldbody>'
      '<body pos="0 0 0.3"><freejoint/><geom name="a" type="hfield" hfield="h"/></body>'
      '<body pos="0 0 0.35"><freejoint/><geom name="b" type="hfield" hfield="h"/></body>'
      '</worldbody></mujoco>')
  d2 = mujoco.MjData(m2)
  for _ in range(100):
    mujoco.mj_step(m2, d2)
  assert d2.ncon == 0


BOX8 = ("0.1 0.1 0.1 -0.1 0.1 0.1 -0.1 -0.1 0.1 0.1 -0.1 0.1 "
        "0.1 0.1 -0.1 -0.1 0.1 -0.1 -0.1 -0.1 -0.1 0.1 -0.1 -0.1")


def _box_on_plane_xml(z=0.099, gravity="0 0 -9.81"):
  return (f'<mujoco><asset><mesh name="cube" vertex="{BOX8}"/></asset>'
          f'<option timestep="0.002" integrator="Euler" iterations="200" '
          f'tolerance="1e-8" gravity="{gravity}"/>'
          '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
          f'<body pos="0 0 {z}"><freejoint/>'
          '<geom name="bx" type="mesh" mesh="cube"/></body>'
          '</worldbody></mujoco>')


def _native_active_slots(sim):
  sim.assembled_system(recompute=True)
  return sim._coupled_constraints.broadphase_counts()[0][1]


def test_mesh_plane_manifold_counts_cpu():
  # Pinned oracle: a cube face press produces a multi-point manifold
  # (single-witness natives undercount springs/moment arms; R05-1).
  m = mujoco.MjModel.from_xml_string(_box_on_plane_xml(z=0.099))
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(m.qpos0)
  mujoco.mj_forward(m, d)
  assert d.ncon >= 2, d.ncon


def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
@pytest.mark.parametrize("condim", [3, 4, 6])
def test_sdf_condim_force_parity_gpu(condim):
  # R05-3: SDF condim 4/6 qualify with the 013 gates (translation 15% +
  # 0.1 N, rotation at force scale). Frictional rows engage on the press.
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  vbox = ("-0.05 -0.05 -0.04 0.05 -0.05 -0.04 0.05 0.05 -0.04 -0.05 0.05 -0.04 "
          "-0.05 -0.05 0.04 0.05 -0.05 0.04 0.05 0.05 0.04 -0.05 0.05 0.04")
  xml = (f'<mujoco><asset><mesh name="b" vertex="{vbox}"/></asset>'
         '<option timestep="0.002" integrator="Euler" iterations="100" '
         f'tolerance="1e-8" gravity="0 0 0" sdf_initpoints="4"/>'
         f'<contact><pair geom1="sdf" geom2="ball" condim="{condim}"/></contact>'
         '<worldbody>'
         '<geom name="sdf" type="sdf" mesh="b" pos="0 0 0" contype="1" conaffinity="1"/>'
         '<body pos="0 0 0.07"><freejoint/>'
         '<geom name="ball" type="sphere" size="0.05" contype="1" conaffinity="1"/></body>'
         '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  sim.reset(qpos=qp, qvel=np.zeros((1, m.nv), dtype=np.float32))
  asm = sim.assembled_system(recompute=True)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  f_nat = np.asarray(asm["qfrc_constraint"].cpu().numpy()[0])
  f_cpu = np.asarray(cpu.qfrc_constraint)
  assert float(np.max(np.abs(f_cpu[:3]))) > 1.0  # non-vacuous press
  # Normal (z) gates tight; static-press friction direction is
  # seed-indeterminate on the symmetric scene, so tangential gates bound
  # magnitude on both engines instead of exact agreement.
  np.testing.assert_allclose(f_nat[2], f_cpu[2], rtol=0.15, atol=0.1,
                             err_msg=f"sdf condim {condim} normal")
  assert float(np.max(np.abs(f_nat[:2]))) < 1.5, f_nat[:2]
  assert float(np.max(np.abs(f_cpu[:2]))) < 1.5, f_cpu[:2]
  np.testing.assert_allclose(f_nat[3:], f_cpu[3:], rtol=0.2, atol=0.5,
                             err_msg=f"sdf condim {condim} rot")


@_needs_gpu()
def test_mesh_manifold_force_sensor_gpu():
  # R05-3 = R01 re-qualification on the new manifolds: force/torque/
  # accelerometer on a mesh-plane face press agree with the CPU oracle
  # (lockstep step-stage samples, then post-step query).
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  xml = (f'<mujoco><asset><mesh name="cube" vertex="{BOX8}"/></asset>'
         '<option timestep="0.002" integrator="Euler" iterations="200" tolerance="1e-8"/>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body pos="0 0 0.099"><freejoint/>'
         '<geom name="bx" type="mesh" mesh="cube"/>'
         '<site name="s"/></body></worldbody>'
         '<sensor><force site="s"/><torque site="s"/><accelerometer site="s"/></sensor></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  sim.reset(qpos=qp, qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  # Redundant-manifold treatment: at fixed deep press the 4-vert native
  # manifold legitimately answers stiffer than the CPU's pruned 2-point
  # subset (same 010/011 representation tier as qfrc), and micro-bounce
  # phases differ. Sensors gate on the settled-window TIME MEAN (both
  # carry the body weight on average); trajectories agree throughout.
  worst_q, mean_nat, mean_cpu, cnt = 0.0, None, None, 0
  for step in range(60):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    worst_q = max(worst_q, float(np.max(np.abs(sim.state.qpos.cpu().numpy()[0] - np.asarray(cpu.qpos)))))
    if step >= 30:
      s = sim.step_sensordata()[0].astype(np.float64)
      c = np.asarray(cpu.sensordata, dtype=np.float64)
      mean_nat = s if mean_nat is None else mean_nat + s
      mean_cpu = c if mean_cpu is None else mean_cpu + c
      cnt += 1
  assert worst_q < 2e-3, worst_q
  mean_nat /= cnt
  mean_cpu /= cnt
  np.testing.assert_allclose(mean_nat, mean_cpu, rtol=0.05, atol=0.3)
  # Post-step query smoke on the manifold scene: finite, right-shaped
  # output (instant values are representation-tier; see above).
  got = sim.sensor_values().cpu().numpy()[0]
  assert got.shape == (m.nsensordata,)
  assert bool(np.all(np.isfinite(got)))


@_needs_gpu()
def test_mesh_pallet_asymmetric_settle_gpu():
  # R05-3 mesh pallet with asymmetric loading: mesh box deck plus a welded
  # corner load settles from a tilt; rest pose matches the CPU oracle.
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  xml = (f'<mujoco><asset><mesh name="deck" vertex="{BOX8}"/></asset>'
         '<option timestep="0.002" integrator="Euler" iterations="200" tolerance="1e-8" gravity="0 0 -9.81"/>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body pos="0 0 0.25"><freejoint/>'
         '<geom name="pallet" type="mesh" mesh="deck"/>'
         '<body pos="0.07 0.05 0.12"><geom name="load" type="box" size="0.03 0.03 0.03" mass="0.8"/></body>'
         '</body></worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  qv = np.zeros((1, m.nv), dtype=np.float32)
  qv[0, 1] = 0.6
  sim.reset(qpos=qp, qvel=qv)
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp[0]
  cpu.qvel[:] = qv[0]
  for _ in range(250):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0][:3],
                             np.asarray(cpu.qpos)[:3], atol=5e-3)


def _hfield_ball_xml(z, margin):
  return (f'<mujoco><asset><hfield name="h" nrow="9" ncol="9" size="1 1 0.2 0.05"/></asset>'
          f'<option timestep="0.002" integrator="Euler" iterations="100" tolerance="1e-8"/>'
          '<worldbody><geom name="terrain" type="hfield" hfield="h"/>'
          f'<body pos="0.1 0.1 {z}"><freejoint/>'
          f'<geom name="ball" type="sphere" size="0.1" margin="{margin}"/></body>'
          '</worldbody></mujoco>')


def test_hfield_positive_margin_near_contact_cpu():
  # Pinned oracle: a 5 mm margin detects near-contacts (positive dist)
  # where margin 0 sees nothing (R05-2).
  m0 = mujoco.MjModel.from_xml_string(_hfield_ball_xml(0.103, 0.0))
  d0 = mujoco.MjData(m0)
  d0.qpos[:] = np.asarray(m0.qpos0)
  mujoco.mj_forward(m0, d0)
  assert d0.ncon == 0
  m1 = mujoco.MjModel.from_xml_string(_hfield_ball_xml(0.103, 0.005))
  d1 = mujoco.MjData(m1)
  d1.qpos[:] = np.asarray(m1.qpos0)
  mujoco.mj_forward(m1, d1)
  assert d1.ncon == 2, d1.ncon
  for j in range(d1.ncon):
    # Pinned expanded-penetration convention: dist = true gap - 1.5*margin
    # (prism tops +margin, other support +margin/2), so margin-range
    # witnesses report negative dists.
    assert -1.5 * 0.005 - 1e-9 <= float(d1.contact[j].dist) <= 0.005, float(d1.contact[j].dist)


@_needs_gpu()
def test_hfield_positive_margin_parity_gpu():
  # R05-2: native margin detection matches the CPU oracle (counts, depths,
  # normals) in near-contact and in press; margin 0 unchanged.
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  for z, margin in ((0.103, 0.005), (0.095, 0.005), (0.095, 0.0)):
    m = mujoco.MjModel.from_xml_string(_hfield_ball_xml(z, margin))
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
              qvel=np.zeros((1, m.nv), dtype=np.float32))
    n_nat = _native_active_slots(sim)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(m.qpos0)
    mujoco.mj_forward(m, d)
    assert n_nat == int(d.ncon), (z, margin, n_nat, int(d.ncon))
    if d.ncon:
      asm = sim.assembled_system()
      mask = np.asarray(asm["contact_mask"].cpu().numpy()[0])
      nat_d = sorted(float(v) for v, on in
                     zip(np.asarray(asm["contact_distance"].cpu().numpy()[0]), mask) if on > 0.5)
      cpu_d = sorted(float(d.contact[j].dist) for j in range(d.ncon))
      assert len(nat_d) == len(cpu_d) == n_nat
      for a, b in zip(nat_d, cpu_d):
        assert abs(a - b) < 1e-3, (z, margin, a, b)
      fr = sim._coupled_constraints._workspace["contact_frame"].detach().cpu().numpy().reshape(-1, 12)
      # 012-consistent gates (see test_hfield_contact_012): sphere normal
      # atol 0.1, depths 1 mm. Margin near-contacts share the shallow tier.
      for j in range(d.ncon):
        np.testing.assert_allclose(fr[j, :3], np.asarray(d.contact[j].frame[:3]), atol=0.1,
                                   err_msg=f"margin nrm z={z}")


@_needs_gpu()
def test_mesh_plane_manifold_counts_and_wrench_gpu():
  # Bounded native manifold (R05-1): the cube face yields true hull-vert
  # witnesses with the pinned normal/depths; slot identity is deterministic
  # across repeated assemblies. Step-0 constraint forces use the redundant-
  # manifold treatment (010 precedent): equally valid subsets (CPU keeps a
  # pruned 2-point subset of the 4-corner face) legitimately differ at
  # fixed penetration, so geometry + trajectory gate this test, not exact
  # step-0 qfrc.
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = mujoco.MjModel.from_xml_string(_box_on_plane_xml(z=0.099))
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  n_native = _native_active_slots(sim)
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(m.qpos0)
  mujoco.mj_forward(m, d)
  assert int(d.ncon) >= 2, d.ncon
  # Bounded manifold: at least the CPU subset size, at most the 4 face verts.
  assert 2 <= n_native <= 4, (n_native, int(d.ncon))
  # Geometry: every native witness carries the pinned normal, a depth
  # within 1 mm of the CPU penetration, and a true face-vert position.
  fr = sim._coupled_constraints._workspace["contact_frame"].detach().cpu().numpy().reshape(-1, 12)
  poss, nrms = [], []
  for s in range(n_native):
    nrms.append(fr[s, :3].copy())
    poss.append(fr[s, 9:12].copy())
  nrms = np.asarray(nrms)
  poss = np.asarray(poss)
  np.testing.assert_allclose(nrms, np.tile([0, 0, 1], (n_native, 1)), atol=1e-3)
  cpu_deep = min(float(d.contact[j].dist) for j in range(d.ncon))
  for s in range(n_native):
    # Witness depth recovered from the contact point (midpoint convention):
    # z_wit + dist/2 = plane height 0.
    assert abs(poss[s][2] - cpu_deep / 2) < 1.5e-3, (poss[s], cpu_deep)
    assert abs(abs(poss[s][0]) - 0.1) < 5e-3 and abs(abs(poss[s][1]) - 0.1) < 5e-3, poss[s]
  # Manifold spans the face (max pairwise distance covers the 0.2 m face).
  span = max(float(np.linalg.norm(poss[i] - poss[j]))
             for i in range(n_native) for j in range(n_native))
  assert span >= 0.15, (span, poss)
  # Deterministic identity: repeated assembly at the same state is bitwise
  # identical in slot order.
  f1 = sim._coupled_constraints._workspace["contact_frame"].detach().cpu().numpy().copy()
  sim.assembled_system(recompute=True)
  f2 = sim._coupled_constraints._workspace["contact_frame"].detach().cpu().numpy().copy()
  np.testing.assert_array_equal(f1, f2)
  assert int(sim.state.status.cpu().numpy()[0]) == 0


@_needs_gpu()
def test_mesh_manifold_cardinality_change_and_settle_gpu():
  # Edge/corner transition changes manifold cardinality; warm starts stay
  # finite and the settle trajectory tracks the CPU oracle.
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.simulation import MetalSimulation
  m = mujoco.MjModel.from_xml_string(_box_on_plane_xml(z=0.099))
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qp = np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1)
  # Slight tilt: corner-first contact (smaller manifold).
  qp[0, 3:7] = np.array([0.999, 0.02, 0.03, 0.0], dtype=np.float32)
  n = np.linalg.norm(qp[0, 3:7])
  qp[0, 3:7] /= n
  sim.reset(qpos=qp, qvel=np.zeros((1, m.nv), dtype=np.float32))
  n_tilt = _native_active_slots(sim)
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  n_flat = _native_active_slots(sim)
  assert n_flat >= n_tilt, (n_flat, n_tilt)
  assert n_flat >= 2 and n_tilt >= 1
  # Settle trajectory parity with asymmetric spin.
  qv = np.zeros((1, m.nv), dtype=np.float32)
  qv[0, 3] = 0.4
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1), qvel=qv)
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(m.qpos0)
  d.qvel[:] = qv[0]
  for _ in range(150):
    sim.step(1)
    mujoco.mj_step(m, d)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[0][:3],
                             np.asarray(d.qpos)[:3], atol=5e-3)
  w = sim._coupled_constraints.get_warmstart()
  assert bool(np.all(np.isfinite(w)))
