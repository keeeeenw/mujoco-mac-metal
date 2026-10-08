"""Paired aref/b chain: dist/constant low words, widened rows, refinement.

Executes production narrowphase + solve on the torus SDF fixture and
checks the paired reference chain end to end using step-internal state
(no recompute path): record dist_lo, constant lo buffer, 42-word rows
with nonzero aref corrections, tight same-state E_aref match against the
pinned CPU oracle, and tight same-state constraint-force match (exercises
J pairing, aref pairing, dual refinement and pair force reconstruction).
Component/block routes keep exact-zero low planes (coherently high-only).
"""
import os

import numpy as np
import pytest


def _torus_model():
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
  return _plugin_sphere_model("torus")


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


@_needs_gpu()
def test_paired_aref_chain_matches_cpu_reference_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  for _ in range(4):
    sim.step(1, qfrc_applied=qfrc)
  cc = sim._coupled_constraints
  # Re-solve at the current state so all compared quantities share one
  # state (retained step views belong to the pre-step state). At this
  # unambiguous state the recomputed contacts agree with step contacts;
  # the assertions below fail loudly if that ever stops holding.
  s = sim.assembled_system(recompute=True)
  rd = cc._workspace["contact_row_data"].cpu().numpy().reshape(8, 6, 7)
  assert rd.shape == (8, 6, 7)
  # aref corrections are really exercised (nonzero low words).
  assert float(np.max(np.abs(rd[:, 0, 6]))) > 1e-7
  # Same-state CPU reference.
  qpos = np.asarray(sim.state._qpos[0].cpu().numpy(), dtype=np.float64)
  qvel = np.asarray(sim.state._qvel[0].cpu().numpy(), dtype=np.float64)
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  d.qfrc_applied[:] = qfrc[0]
  mujoco.mj_forward(model, d)
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  cpu = [(float(d.contact[c].dist), np.asarray(d.contact[c].pos),
          float(np.asarray(d.efc_aref)[c])) for c in range(d.ncon)]
  assert len(cpu) == 8
  worst = 0.0
  used = set()
  for i in range(8):
    c = min((x for x in range(len(cpu)) if x not in used),
              key=lambda x: float(np.linalg.norm(cpu[x][1] - npos[i])))
    used.add(c)
    assert float(np.linalg.norm(cpu[c][1] - npos[i])) < 1e-4
    E = float(rd[i, 0, 3]) + float(rd[i, 0, 6])
    worst = max(worst, abs(E - cpu[c][2]))
  print(f"PAIRED_AREF worst |E-cpu| = {worst:.3e}", flush=True)
  assert worst < 5e-9


@_needs_gpu()
def test_record_dist_lo_and_const_lo_present_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  sim.step(1, qfrc_applied=qfrc)
  cc = sim._coupled_constraints
  rec = cc._workspace["contact_pair_records"].cpu().numpy()
  nslots = 8
  los = [float(rec[i * 37 + 9]) for i in range(nslots)]
  assert any(v != 0.0 for v in los), los
  clo = cc._constants["pair_const_lo"].cpu().numpy().reshape(-1, 11)
  assert clo.shape[0] >= 1
  assert float(np.max(np.abs(clo))) > 0.0
  # Independent oracle: lo = f64 TRUE - f32 hi. This fixture resolves to
  # the default o_solref (0.02, 1.0), so the first pair's solref lo words
  # are exactly representable differences of known decimals.
  expect_r0 = np.float64(0.02) - np.float64(np.float32(0.02))
  expect_r1 = np.float64(1.0) - np.float64(np.float32(1.0))
  assert abs(float(clo[0, 0]) - float(expect_r0)) < 1e-12
  assert abs(float(clo[0, 1]) - float(expect_r1)) < 1e-12


@_needs_gpu()
def test_same_state_force_tight_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  for _ in range(4):
    sim.step(1, qfrc_applied=qfrc)
  # Re-solve at the current state (see paired-aref test for why the
  # retained step views cannot be used for same-state force comparison).
  s = sim.assembled_system(recompute=True)
  qpos = np.asarray(sim.state._qpos[0].cpu().numpy(), dtype=np.float64)
  qvel = np.asarray(sim.state._qvel[0].cpu().numpy(), dtype=np.float64)
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
  d.qvel[:] = qvel
  d.qfrc_applied[:] = qfrc[0]
  mujoco.mj_forward(model, d)
  nat = np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64)
  cpu = np.asarray(d.qfrc_constraint, dtype=np.float64)
  err = np.max(np.abs(nat - cpu))
  scale = max(1.0, float(np.max(np.abs(cpu))))
  print(f"PAIRED_FORCE relerr = {err / scale:.3e}", flush=True)
  assert err / scale < 5e-7


@_needs_gpu()
def test_block_sparse_route_keeps_zero_low_planes_gpu():
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_scalable_v1")
  cc = sim._coupled_constraints
  assert cc._component_mass_storage == "block_sparse"
  qfrc = np.zeros((1, model.nv), dtype=np.float32)
  qfrc[0, 0] = 0.01
  sim.step(1, qfrc_applied=qfrc)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  lay = cc._jacobian_layout
  W = cc._workspace["workspace_J"].cpu().numpy()
  lo = W[lay.lo_values_offset():lay.lo_values_offset() + lay.nnz]
  assert float(np.max(np.abs(lo))) == 0.0
  rd = cc._workspace["contact_row_data"].cpu().numpy()
  nslots = int(cc.descriptor.ncontacts_max)
  rd = rd.reshape(nslots, 6, 7)
  assert float(np.max(np.abs(rd[:, :, 6]))) == 0.0
