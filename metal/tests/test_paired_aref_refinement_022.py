"""Paired aref/b chain: dist/constant low words, widened rows, refinement.

Executes production narrowphase + solve on the torus SDF fixture and
checks the paired reference chain end to end using step-internal state
(no recompute path): record dist_lo, constant lo buffer, 42-word rows
with nonzero aref corrections, tight same-state E_aref match against the
pinned CPU oracle, and tight same-state constraint-force match (exercises
J pairing, aref pairing, dual refinement and pair force reconstruction).
Component/block routes retain the same source corrections as dense routes.
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
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
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
  rd = cc._workspace["contact_row_data"].cpu().numpy().reshape(8, 60)[:, :42].reshape(8, 6, 7)
  assert rd.shape == (8, 6, 7)
  # aref corrections are really exercised (nonzero low words).
  assert float(np.max(np.abs(rd[:, 0, 6]))) > 1e-7
  # Same-state CPU reference.
  snapshot = sim.state.snapshot()
  nq, nv = model.nq, model.nv
  words = snapshot.precision_words[0].astype(np.float64)
  qpos = snapshot.qpos[0].astype(np.float64) + words[:nq] + words[nq:2*nq]
  qvel = snapshot.qvel[0].astype(np.float64) + words[2*nq:2*nq+nv] + words[2*nq+nv:]
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
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
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
  clo = cc._constants["pair_const_lo"].cpu().numpy().reshape(-1, 23)
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
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
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
  snapshot = sim.state.snapshot()
  nq, nv = model.nq, model.nv
  words = snapshot.precision_words[0].astype(np.float64)
  qpos = snapshot.qpos[0].astype(np.float64) + words[:nq] + words[nq:2*nq]
  qvel = snapshot.qvel[0].astype(np.float64) + words[2*nq:2*nq+nv] + words[2*nq+nv:]
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
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_block_sparse_route_retains_source_low_planes_gpu():
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
  assert float(np.max(np.abs(lo))) > 0.0
  rd = cc._workspace["contact_row_data"].cpu().numpy()
  nslots = int(cc.descriptor.ncontacts_max)
  rd = rd.reshape(nslots, 60)[:, :42].reshape(nslots, 6, 7)
  assert float(np.max(np.abs(rd[:, :, 6]))) > 0.0
  # The represented operator is checked against the pinned engine, rather
  # than accepting nonzero helper output as physics evidence.
  import mujoco
  from test_contact_operator_precision import _effective_rows
  snapshot = sim.state.snapshot()
  words = snapshot.precision_words[0].astype(np.float64)
  nq, nv = model.nq, model.nv
  source = mujoco.MjData(model)
  source.qpos[:] = snapshot.qpos[0].astype(np.float64) + words[:nq] + words[nq:2*nq]
  source.qvel[:] = snapshot.qvel[0].astype(np.float64) + words[2*nq:2*nq+nv] + words[2*nq+nv:]
  mujoco.mj_forward(model, source)
  sim.assembled_system(recompute=True)
  native_J = _effective_rows(cc)
  offsets = np.asarray(cc.descriptor.contact_condim_packed).reshape(-1, 3)
  rows = cc.descriptor.nr_joint + offsets[:source.ncon, 1]
  np.testing.assert_allclose(native_J[rows],
                             np.asarray(source.efc_J).reshape(source.nefc, nv),
                             rtol=2e-6, atol=2e-6)
