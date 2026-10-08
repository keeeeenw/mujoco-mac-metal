# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Forced eight-step torus reproducer with separated error accounts.

Bounded reproducer for the ORIGINAL forced gate
(`test_bundled_rigid_plugin_sdf_force_trajectory_checkpoint_matches_cpu[torus]`,
qfrc_applied[0] = 0.01, 8 steps) plus structured same-input stage packets:

- per-step trajectory divergence curve (earliest mismatch: step 1
  carries qacc noise, the gate tier first breaks later);
- same-state forward error at the post-step-1 native state, separated
  from accumulated trajectory error (contacts exact, force at the
  float32 floor, source-row-mapped multipliers — no sorted vectors);
- reproducible oracle sensitivity: uniform +1e-9 perturbation on all 7
  qpos entries (meters on positions, unnormalized units on the quaternion
  components) rolled out on the CPU for the same 8 forced steps, with
  per-step values;
- trajectory warmstart norms on both sides.

Magnitudes are REPORTED and structurally asserted (growth, separation,
floors). They pin the current precision-amplification regime; a repair
that reaches the gate tiers must drive the trajectory curve below them,
at which point these windows update. No fixture, tolerance, seed-count
or iteration change.
"""

import os

import numpy as np
import pytest


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


def _forced_rollout(nsteps=8):
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
  from mujoco_metal.simulation import MetalSimulation
  import mujoco
  m = _plugin_sphere_model("torus")
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qpos = sim.state._qpos[0].cpu().numpy().astype(np.float64, copy=True)
  qvel = sim.state._qvel[0].cpu().numpy().astype(np.float64, copy=True)
  from test_bundled_sdf_rigid_oracle_019 import _cpu_contacts
  cpu_data, _, _ = _cpu_contacts(m, qpos, qvel)
  qfrc = np.zeros((1, m.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  curve = []
  for _ in range(nsteps):
    cpu_data.qfrc_applied[:] = qfrc[0]
    mujoco.mj_step(m, cpu_data)
    sim.step(1, qfrc_applied=qfrc)
    curve.append((
        float(np.max(np.abs(
            np.asarray(sim.state._qpos[0].cpu().numpy()) - cpu_data.qpos))),
        float(np.max(np.abs(
            np.asarray(sim.state._qvel[0].cpu().numpy()) - cpu_data.qvel))),
        float(np.max(np.abs(
            np.asarray(sim.state._qacc[0].cpu().numpy()) - cpu_data.qacc)))))
  return sim, m, qfrc, curve


@_needs_gpu()
def test_forced_rollout_divergence_curve_gpu():
  _, _, _, curve = _forced_rollout()
  assert len(curve) == 8
  dq, dv, da = curve[0]
  # Earliest mismatch (step 1) is small-acceleration noise, far below the
  # gate tiers: seed-level, not a contact miss.
  assert 1e-9 <= dq <= 1e-8, dq
  assert 1e-6 <= dv <= 1e-5, dv
  assert 5e-4 <= da <= 5e-3, da
  # Accumulated trajectory error exceeds the seed noise by orders of
  # magnitude: sensitive dynamics amplify per-step differences.
  assert curve[-1][1] / max(dv, 1e-30) > 100.0
  print("ROLLOUT " + " ".join(
      f"step{k + 1}=({q:.2e},{v:.2e},{a:.2e})"
      for k, (q, v, a) in enumerate(curve)), flush=True)


@_needs_gpu()
def test_same_state_forward_separated_from_trajectory_gpu():
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _unique_native_slots
  import mujoco
  sim, m, qfrc, _ = _forced_rollout(nsteps=1)
  nq = np.asarray(sim.state._qpos[0].cpu().numpy(), dtype=np.float64)
  nqv = np.asarray(sim.state._qvel[0].cpu().numpy(), dtype=np.float64)
  d2 = mujoco.MjData(m)
  d2.qpos[:] = nq
  d2.qvel[:] = nqv
  d2.qfrc_applied[:] = qfrc[0].astype(np.float64)
  mujoco.mj_forward(m, d2)
  s = sim.assembled_system(qfrc_applied=qfrc, recompute=True)
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  ndist = np.asarray(s["contact_distance"][0].cpu().numpy())
  nnrm = np.asarray(s["contact_normal"][0].cpu().numpy())
  cpu = [(float(d2.contact[c].dist), np.asarray(d2.contact[c].pos),
          np.asarray(d2.contact[c].frame[:3])) for c in range(d2.ncon)]
  assert d2.ncon == int(np.sum(mask > 0.5)) == 8
  slots = [x for x in range(len(mask)) if mask[x] > 0.5]
  used, worst = set(), [0.0, 0.0, 0.0]
  for i in slots:
    c = min((x for x in range(d2.ncon) if x not in used),
            key=lambda x: float(np.linalg.norm(cpu[x][1] - npos[i])))
    used.add(c)
    worst[0] = max(worst[0], abs(cpu[c][0] - ndist[i]))
    worst[1] = max(worst[1], float(np.linalg.norm(cpu[c][1] - npos[i])))
    worst[2] = max(worst[2], float(np.linalg.norm(cpu[c][2] - nnrm[i])))
  # Same-state contacts are exact: no positional seed error drives the
  # first divergence.
  assert worst[0] < 1e-12, worst
  assert worst[1] < 1e-7, worst
  assert worst[2] < 1e-7, worst
  natf = np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64)
  cpuf = np.asarray(d2.qfrc_constraint, dtype=np.float64)
  scale = max(1.0, float(np.max(np.abs(cpuf))))
  assert float(np.max(np.abs(natf - cpuf))) / scale < 1e-7
  nata = np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64)
  cpua = np.asarray(d2.qacc, dtype=np.float64)
  # Acceleration carries the small-inertia-amplified floor (inertia
  # 0.0168, ~60x gain on rotational rows); trajectory error at later
  # steps exceeds this by orders of magnitude (see curve test).
  assert float(np.max(np.abs(nata - cpua))) < 2e-3
  # Source-row-mapped multipliers (no sorted vectors): per-contact row
  # blocks from the descriptor packing against CPU efc rows.
  desc = sim._coupled_constraints.descriptor
  packed = np.asarray(desc.contact_condim_packed, dtype=np.int32).reshape(-1, 3)
  lam = np.asarray(s["lambda"][0].cpu().numpy(), dtype=np.float64)
  lc = np.asarray(d2.efc_force, dtype=np.float64)
  rows = np.asarray(d2.contact[0].efc_address, dtype=np.int64)
  base0 = int(d2.contact[0].efc_address)
  assert base0 == 0
  for ci in range(d2.ncon):
    base = int(d2.contact[ci].efc_address)
    count = (int(d2.contact[ci + 1].efc_address) - base
             if ci + 1 < d2.ncon else int(d2.nefc) - base)
    slot = min(slots, key=lambda i: float(np.linalg.norm(
        np.asarray(d2.contact[ci].pos) - npos[i])))
    gpu_base = int(desc.nr_joint) + int(packed[slot, 1])
    gpu_rows = np.arange(gpu_base, gpu_base + count)
    np.testing.assert_allclose(
        lam[gpu_rows], lc[base:base + count], rtol=1e-4, atol=1e-4)
  print(f"SEPARATION dist={worst[0]:.2e} pos={worst[1]:.2e} "
        f"nrm={worst[2]:.2e} qfrc_rel="
        f"{float(np.max(np.abs(natf - cpuf))) / scale:.2e} "
        f"qacc={float(np.max(np.abs(nata - cpua))):.2e}", flush=True)


def test_oracle_sensitivity_reproducible_cpu():
  # Explicit perturbation experiment on the installed oracle: uniform
  # +1e-9 on all 7 qpos entries (meters on positions; unnormalized
  # quaternion components), same 8 forced steps (qfrc[0] = 0.01).
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
  import mujoco
  m = _plugin_sphere_model("torus")
  qpos = np.asarray(m.qpos0, dtype=np.float64).copy()
  qfrc = np.zeros(m.nv)
  qfrc[0] = 0.01
  d1 = mujoco.MjData(m)
  d1.qpos[:] = qpos
  d1.qvel[:] = 0
  d2 = mujoco.MjData(m)
  d2.qpos[:] = qpos + 1e-9
  d2.qvel[:] = 0
  curve = []
  for _ in range(8):
    d1.qfrc_applied[:] = qfrc
    mujoco.mj_step(m, d1)
    d2.qfrc_applied[:] = qfrc
    mujoco.mj_step(m, d2)
    curve.append((
        float(np.max(np.abs(d1.qpos - d2.qpos))),
        float(np.max(np.abs(d1.qvel - d2.qvel)))))
  assert 1e-6 <= curve[0][1] <= 1e-5, curve[0]
  assert 1e-3 <= curve[-1][1] <= 1e-1, curve[-1]
  assert curve[-1][1] / max(curve[0][1], 1e-30) > 100.0
  print("SENSITIVITY " + " ".join(
      f"step{k + 1}=({q:.2e},{v:.2e})" for k, (q, v) in enumerate(curve)),
      flush=True)
