# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 017: explicit broadphase mask + compaction (GPU).

Conservative sphere-overlap mask matches a CPU oracle, every slot-active
pair is mask-active (superset), ordering is deterministic across runs and
batch worlds, and mask-inactive pairs do not affect the solve.
"""

import os

import mujoco
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")]


def _scene():
  return mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' cone='pyramidal'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='-0.4 0 0.12'><freejoint/><geom name='a' type='sphere' size='0.09'/></body>"
      "<body pos='0.4 0 0.5'><freejoint/><geom name='b' type='sphere' size='0.09'/></body>"
      "<body pos='3 0 0.12'><freejoint/><geom name='far' type='sphere' size='0.09'/></body>"
      "</worldbody></mujoco>")


def _cpu_mask(model, data):
  geoms = int(model.ngeom)
  gp = np.asarray(model.geom_bodyid)
  rb = np.asarray(model.geom_rbound, dtype=np.float64)
  mg = []
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints as _M
  xp = np.asarray(data.geom_xpos).reshape(geoms, 3)
  out = []
  # Pair order follows the lowering (sorted mask-passing pairs); recompute
  # the expected mask per pair from the descriptor order instead.
  return xp, rb


def test_mask_matches_cpu_sphere_superset_and_deterministic_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _scene()
  sim = MetalSimulation(model, batch_size=2, profile="integrated_euler_v1")
  qp = np.asarray(model.qpos0, dtype=np.float32)
  qv = np.zeros((2, model.nv), dtype=np.float32)
  sim.reset(qpos=np.stack([qp, qp]), qvel=qv)
  sim.step(5)
  cc = sim._coupled_constraints
  d = cc.descriptor
  pairs = np.column_stack([np.asarray(d.geom1), np.asarray(d.geom2)])
  assert pairs.shape[0] >= 3
  counts1 = cc.broadphase_counts()
  counts2 = cc.broadphase_counts()
  assert counts1 == counts2  # deterministic across reads
  assert counts1[0] == counts1[1]  # batch independence (identical states)
  mask = cc._workspace["pair_mask"][: 2 * d.npairs].cpu().numpy().reshape(2, d.npairs)
  # CPU sphere oracle on the stepped state.
  gq = sim.state.qpos.cpu().numpy()[0].astype(float)
  dd = mujoco.MjData(model)
  dd.qpos[:] = gq
  mujoco.mj_forward(model, dd)
  xp = np.asarray(dd.geom_xpos).reshape(-1, 3)
  rb = np.asarray(model.geom_rbound, dtype=float)
  margin = np.asarray(d.margin, dtype=float)
  gap = np.asarray(d.gap, dtype=float)
  gtype = np.asarray(model.geom_type, dtype=int)
  for w in range(2):
    for p, (a, b) in enumerate(pairs.tolist()):
      if gtype[a] == 0 or gtype[b] == 0:
        expect = 1.0  # planes are unbounded: mask always overlaps
      else:
        R = float(rb[a] + rb[b] + margin[p] + gap[p]) + 1e-6
        expect = float(np.linalg.norm(xp[a] - xp[b]) <= R)
      assert float(mask[w, p] > 0.5) == expect, (w, p)
  # Superset: every slot-active pair is mask-active.
  rows = cc._workspace["contact_row_data"][: 2 * d.ncontacts_max * 36].cpu().numpy().reshape(2, d.ncontacts_max, 6, 6)
  off = np.asarray(d.pair_contact_offset, dtype=int)
  for w in range(2):
    for p in range(d.npairs):
      active = bool(np.any(rows[w, off[p]:off[p + 1], 0, 0] > 0.5)) if off[p + 1] > off[p] else False
      if active:
        assert mask[w, p] > 0.5, (w, p)


def test_inactive_pairs_do_not_affect_solve_gpu():
  # Same close pair, with and without an extra far pair: qacc must match.
  from mujoco_metal.simulation import MetalSimulation
  close = (
      "<mujoco><option timestep='0.002' cone='pyramidal'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='-0.4 0 0.12'><freejoint/><geom name='a' type='sphere' size='0.09'/></body>"
      "</worldbody></mujoco>")
  far = (
      "<mujoco><option timestep='0.002' cone='pyramidal'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='-0.4 0 0.12'><freejoint/><geom name='a' type='sphere' size='0.09'/></body>"
      "<body pos='3 0 0.12'><freejoint/><geom name='b' type='sphere' size='0.09'/></body>"
      "</worldbody></mujoco>")
  m1 = mujoco.MjModel.from_xml_string(close)
  m2 = mujoco.MjModel.from_xml_string(far)
  s1 = MetalSimulation(m1, batch_size=1, profile="integrated_euler_v1")
  s2 = MetalSimulation(m2, batch_size=1, profile="integrated_euler_v1")
  q1 = np.asarray(m1.qpos0, dtype=np.float32).reshape(1, -1)
  q2 = np.asarray(m2.qpos0, dtype=np.float32)
  # Align shared DoFs: m2 has extra free joint appended; copy the common prefix.
  n = min(q1.shape[1], q2.shape[0])
  qq2 = np.asarray(m2.qpos0, dtype=np.float32).reshape(1, -1).copy()
  qq2[0, :n] = q1[0, :n]
  s1.reset(qpos=q1, qvel=np.zeros((1, m1.nv), dtype=np.float32))
  s2.reset(qpos=qq2, qvel=np.zeros((1, m2.nv), dtype=np.float32))
  s1.step(80)
  s2.step(80)
  a1 = s1._acceleration(s1.state._qpos, s1.state._qvel)[0].cpu().numpy()[0]
  a2 = s2._acceleration(s2.state._qpos, s2.state._qvel)[0].cpu().numpy()[0]
  np.testing.assert_allclose(a1, a2[: len(a1)], rtol=2e-5, atol=2e-5)
  c2 = s2._coupled_constraints.broadphase_counts()[0]
  assert c2[0] >= 1  # mask sees pairs, slot-active subset compacts the solve


def _press_scene():
  # One pressed pair (active contact) plus far prunable pairs.
  return mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' cone='pyramidal'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='-0.4 0 0.085'><freejoint/><geom name='a' type='sphere' size='0.09'/></body>"
      "<body pos='3 0 0.5'><freejoint/><geom name='b' type='sphere' size='0.09'/></body>"
      "<body pos='-3 0 0.5'><freejoint/><geom name='c' type='sphere' size='0.09'/></body>"
      "</worldbody></mujoco>")


def test_pruning_on_off_physics_equivalence_gpu():
  # R08/F1: pruning toggle must not change physics (same qacc/qfrc/status).
  from mujoco_metal.simulation import MetalSimulation
  m = _press_scene()
  results = {}
  for enabled in (True, False):
    sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
    sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
              qvel=np.zeros((1, m.nv), dtype=np.float32))
    sim._coupled_constraints.set_broadphase_pruning(enabled)
    for _ in range(30):
      sim.step(1)
    asm = sim.assembled_system()
    results[enabled] = (
        np.asarray(asm["qacc"].cpu().numpy()).copy(),
        np.asarray(asm["qfrc_constraint"].cpu().numpy()).copy(),
        np.asarray(asm["status"].cpu().numpy()).copy(),
    )
  np.testing.assert_array_equal(results[True][0], results[False][0])
  np.testing.assert_array_equal(results[True][1], results[False][1])
  np.testing.assert_array_equal(results[True][2], results[False][2])


def test_pruned_slots_carry_sentinel_gpu():
  # R08/F1: pruned pairs stamp the finite sentinel (skip path executed);
  # active contact slots never carry it. Non-vacuous both ways.
  from mujoco_metal.simulation import MetalSimulation
  m = _press_scene()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray(m.qpos0, dtype=np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  sim._coupled_constraints.set_broadphase_pruning(True)
  for _ in range(10):
    sim.step(1)
  cc = sim._coupled_constraints
  nc = cc.descriptor.ncontacts_max
  frame = cc._workspace["contact_frame"][:nc * 12].cpu().numpy().reshape(nc, 12)
  rows = cc._workspace["contact_row_data"][:nc * 36].cpu().numpy().reshape(nc, 6, 6)
  active = rows[:, 0, 0] > 0.5
  assert int(active.sum()) >= 1  # the pressed pair is active
  sentinel = frame[:, 0] == np.float32(1234.0)
  assert int((~active & sentinel).sum()) >= 1  # pruned slots stamped
  assert int((active & sentinel).sum()) == 0  # active slots never stamped
  # Every inactive unstamped slot belongs to a plane-involved pair (planes
  # always overlap: never pruned, correctly left to the narrowphase).
  import mujoco as _mj
  plane = int(_mj.mjtGeom.mjGEOM_PLANE)
  gtype = np.asarray(m.geom_type)
  pair_geoms = cc._constants["pair_geoms"].cpu().numpy().reshape(-1, 2)
  offsets = cc._constants["pair_contact_offset"].cpu().numpy()
  for s in np.flatnonzero(~active & ~sentinel):
    pair = int(np.searchsorted(offsets[1:], s, side="right"))
    assert plane in (int(gtype[pair_geoms[pair, 0]]), int(gtype[pair_geoms[pair, 1]])), s
  # Mask agreement: slot-active pairs are mask-active (superset), and the
  # mask sees strictly fewer than all pairs (pruning is live).
  mask_active, slot_active = cc.broadphase_counts()[0]
  assert 0 < slot_active <= mask_active < cc.descriptor.npairs
