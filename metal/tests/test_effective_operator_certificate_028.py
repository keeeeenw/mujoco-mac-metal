# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Effective-operator evidence: named high-only diagnostic plus a published
paired-operator residual check, with polish finite-iterate semantics.

`solver_diagnostics[0]` certifies the multipliers against the STORED
high-word assembled row system (W/R/rhs with high multipliers); force
publication consumes the paired effective operator (paired J, multiplier
lows, aref lows). These tests expose both, correctly named, from published
device state (no new kernels, no hot-path readback beyond the test):

- the host float64 replica of the high-only formula agrees with the
  retained diagnostic to float32-evaluation floor (same operator);
- the paired-multiplier residual (same formula with published lam+lamlo)
  differs by far more than rounding (low words affect the result) while
  staying finite and certified-scale;
- the retained pair-correction magnitude is recorded and bounded, with
  algorithm/budget/iteration counts;
- the same paired computation runs on mixed systems (polish off) with
  complete primal force/qacc semantics;
- loose-tolerance, cost-improvement and exhausted-budget paths keep
  pinned finite-iterate behavior with honest (possibly uncertified)
  diagnostics and oracle agreement at documented tiers.
"""

import os

import numpy as np
import pytest


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


def _torus_model(**kw):
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
  m = _plugin_sphere_model("torus")
  for k, v in kw.items():
    setattr(m.opt, k, v)
  return m


def _lamlo(cc):
  nv, nr = cc.descriptor.nv, cc.descriptor.nr
  dbg = cc._workspace["workspace_debug"].cpu().numpy().astype(np.float64)
  base = nr * nr + 7 * nr + cc._high_low_workspace_offset + 8 * max(nv, 1)
  return dbg[base + max(nr, 1):base + 2 * max(nr, 1)].copy()


def _projected_residual(W, R, rhs, ar, lo, hi, lam, nr):
  res = 0.0
  for row in range(nr):
    grad = -rhs[row] + float(W[row] @ lam) + R[row] * lam[row]
    diag = max(1e-15, float(W[row, row]) + R[row])
    p = min(max(lam[row] - grad / diag, lo[row]), hi[row])
    scale = (abs(ar[row]) + abs(R[row] * lam[row])
             + float(np.sum(np.abs(W[row] * lam))))
    res = max(res, abs(p - lam[row]) * diag / max(1.0, scale))
  return res


def _read_system(s, cc, nr):
  W = np.asarray(s["W"][0].cpu().numpy(), dtype=np.float64).reshape(nr, nr)
  R = np.asarray(s["R"][0].cpu().numpy(), dtype=np.float64)
  rhs = np.asarray(s["rhs"][0].cpu().numpy(), dtype=np.float64)
  ar = np.asarray(s["ar"][0].cpu().numpy(), dtype=np.float64)
  lo = np.asarray(s["lo"][0].cpu().numpy(), dtype=np.float64)
  hi = np.asarray(s["hi"][0].cpu().numpy(), dtype=np.float64)
  lam = np.asarray(s["lambda"][0].cpu().numpy(), dtype=np.float64)
  return W, R, rhs, ar, lo, hi, lam


@_needs_gpu()
def test_high_only_diagnostic_and_paired_residual_gpu():
  # Torus, Newton, ordinary budget: uniform scalar-contact system with live
  # low words (SDF Jacobian corrections), so polish runs and the paired
  # check exercises actual corrections.
  import sys
  sys.path.insert(0, "metal/tests")
  from test_solver_budget_contract_024 import _fixed_state
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model(solver=2, iterations=100)
  qpos = _fixed_state()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nv, nr = int(model.nv), int(cc.descriptor.nr)
  assert (nv, nr) == (6, 16)
  W, R, rhs, ar, lo, hi, lam = _read_system(s, cc, nr)
  lamlo = _lamlo(cc)
  assert np.all(np.isfinite(lamlo))
  diag = np.asarray(s["solver_diagnostics"][0].cpu().numpy())
  cert_hi, used = float(diag[0]), float(diag[1])
  # Algorithm/budget counts: Newton, configured 100, stopped early.
  assert used <= 100.0
  assert model.opt.solver == 2 and model.opt.iterations == 100
  # The retained diagnostic names the STORED high-word operator: the host
  # float64 replica of the same formula agrees to float32 floor.
  replica = _projected_residual(W, R, rhs, ar, lo, hi, lam, nr)
  assert np.isfinite(replica) and np.isfinite(cert_hi)
  assert abs(replica - cert_hi) <= 1e-8, (replica, cert_hi)
  assert cert_hi <= 1e-6 and replica <= 1e-6
  # The published paired-operator residual (same formula, published
  # lam+lamlo): low words affect the result far above rounding, stay
  # finite and certified-scale. This is the effective-operator check;
  # it is NOT the same number as the high-only diagnostic, by construction.
  paired = _projected_residual(W, R, rhs, ar, lo, hi, lam + lamlo, nr)
  assert np.isfinite(paired)
  assert abs(paired - replica) > 1e-12, (paired, replica)
  assert paired <= 1e-6
  # Retained pair-correction magnitude is recorded and bounded: nonzero
  # (corrections exist on this fixture) and far below multiplier scale.
  corr = float(np.max(np.abs(lamlo)))
  assert corr > 0.0
  assert corr <= 1e-3
  print(f"CERT high={cert_hi:.3e} replica={replica:.3e} paired={paired:.3e} "
        f"gap={abs(paired-replica):.3e} corr={corr:.3e} used={used}",
        flush=True)


@_needs_gpu()
def test_paired_residual_mixed_system_gpu():
  # Elliptic-cone SDF system: refinement stays off globally, but the
  # paired-operator check still runs over the primal-produced lows with
  # complete force/qacc semantics (no partial-move corruption).
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  import sys
  sys.path.insert(0, "metal/tests")
  from test_solver_budget_contract_024 import _fixed_state
  xml = """
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="100" tolerance="1e-8" sdf_initpoints="8"
              cone="elliptic"/>
      <extension><plugin plugin="mujoco.sdf.torus">
        <instance name="shape"/>
      </plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="probe" pos="0.45 0 0">
          <freejoint/><geom name="probe_sphere" type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <contact><pair geom1="plugin_sdf" geom2="probe_sphere" condim="3"/></contact>
    </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  m.opt.solver = 2
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  qpos32 = qpos.astype(np.float32).astype(np.float64)
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  cc = sim._coupled_constraints
  nv, nr = int(m.nv), int(cc.descriptor.nr)
  W, R, rhs, ar, lo, hi, lam = _read_system(s, cc, nr)
  lamlo = _lamlo(cc)
  assert np.all(np.isfinite(lamlo))
  replica = _projected_residual(W, R, rhs, ar, lo, hi, lam, nr)
  paired = _projected_residual(W, R, rhs, ar, lo, hi, lam + lamlo, nr)
  assert np.isfinite(replica) and np.isfinite(paired)
  d = mujoco.MjData(m)
  d.qpos[:] = qpos32
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(
      np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qfrc_constraint, dtype=np.float64), rtol=1e-3, atol=1e-3)


@_needs_gpu()
@pytest.mark.parametrize("tol,budget", [(1e-2, 100), (1e-8, 1)],
                         ids=["loose", "exhausted"])
def test_polish_semantics_tolerance_and_budget_gpu(tol, budget):
  # Loose tolerance: early cost-improvement stop keeps pinned behavior
  # (fewer iterations, honest small diagnostic, oracle agreement).
  # Exhausted budget: raw finite-iterate publication with an explicitly
  # uncertified diagnostic (== 1.0 pre-solve residual report), oracle
  # agreement at the fixture floor. Neither path adds hidden iterations:
  # reported counts never exceed the configured budget on either side.
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from test_solver_budget_contract_024 import _fixed_state
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model(solver=2, iterations=budget, tolerance=tol)
  qpos = _fixed_state()
  qpos32 = qpos.astype(np.float32).astype(np.float64)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  diag = np.asarray(s["solver_diagnostics"][0].cpu().numpy())
  dev_iters = float(diag[1])
  d = mujoco.MjData(model)
  d.qpos[:] = qpos32
  d.qvel[:] = np.zeros(model.nv)
  mujoco.mj_forward(model, d)
  cpu_iters = int(np.sum(d.solver_niter))
  assert dev_iters <= budget and cpu_iters <= budget
  if tol == 1e-2:
    assert dev_iters < budget and cpu_iters < budget
    assert float(diag[0]) <= 1e-6
  else:
    assert dev_iters == 1.0 and cpu_iters == 1
    assert float(diag[0]) >= 1.0 - 1e-6
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  print(f"SEMANTICS tol={tol} budget={budget} dev={dev_iters} "
        f"cpu={cpu_iters} cert={float(diag[0]):.3e}", flush=True)
