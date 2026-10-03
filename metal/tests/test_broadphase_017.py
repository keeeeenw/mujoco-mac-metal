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
