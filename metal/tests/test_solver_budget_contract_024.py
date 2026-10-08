"""Solver budget contract: cold/warm 0/1/2/ordinary budgets vs oracle.

Polish (dual refinement, qacc reconstruction) runs only on early stops
(used < maxiter); budget-exhausted iterates publish raw primal values.
At zero budget the island solver's own first-update forces publish
natively (matching the pinned initial-point publication); nothing extra
runs. Tests compare retained state, force, qacc, iteration counts and
restore against the pinned oracle for CG/Newton, cold/warm, with and
without no-slip, with nonzero-low-word observability (refinement fired
at ordinary budgets, skipped at exhausted budgets) and zero-low
controls (analytic models keep exact-zero low planes).
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


_QPOS0 = None


def _fixed_state():
  global _QPOS0
  if _QPOS0 is None:
    import sys
    sys.path.insert(0, "metal/tests")
    from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
    from mujoco_metal.simulation import MetalSimulation
    base = _plugin_sphere_model("torus")
    ref = MetalSimulation(base, batch_size=1,
                          profile="integrated_euler_v1")
    _QPOS0 = np.asarray(ref.state._qpos[0].cpu().numpy(),
                        dtype=np.float64).copy()
  return _QPOS0


def _cpu_forward(model, qpos):
  import mujoco
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
  d.qvel[:] = np.zeros(model.nv)
  mujoco.mj_forward(model, d)
  return d


def _row_map(s, d, npos):
  # Slot order is an internal packing detail; budget value comparison uses
  # sorted contact-row vectors (order-independent) while exact iteration
  # counts, status, bounds and restore pin the contract. Index-aware row
  # ownership is covered separately by the island tests via row_island.
  return None


@_needs_gpu()
@pytest.mark.parametrize("solver", [2, 1], ids=["newton", "cg"])
@pytest.mark.parametrize("budget", [0, 1, 2, 100])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_budget_contract_matches_oracle_gpu(solver, budget, warm):
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model()
  model.opt.solver = solver
  model.opt.iterations = budget
  if not warm:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
  qpos = _fixed_state()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  d = _cpu_forward(model, qpos)
  nv = model.nv
  # Iteration counts and status are exact contract terms.
  dev_iters = float(np.asarray(s["solver_diagnostics"][0].cpu().numpy())[1])
  cpu_iters = int(np.sum(d.solver_niter))
  assert cpu_iters <= budget
  assert dev_iters <= budget
  if budget <= 2:
    assert dev_iters == float(cpu_iters)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  lam = np.asarray(s["lambda"][0].cpu().numpy(), dtype=np.float64)
  lc = np.asarray(d.efc_force, dtype=np.float64)
  # Contact-row count must agree exactly (no invented/dropped rows).
  assert int(np.count_nonzero(lam)) == int(np.count_nonzero(lc))
  # Device contact rows are the trailing block for this pinned fixture
  # (nr=16, 8 joint-pad zeros then 8 contact rows); CPU efc is contacts.
  assert lam.shape == (16,) and lc.shape == (8,)
  dev_lam = np.sort(lam[8:16])
  cpu_lam = np.sort(lc)
  scale = max(1.0, float(np.max(np.abs(cpu_lam))))
  print(f"BUDGET solver={solver} budget={budget} warm={warm} "
        f"lam_rel={float(np.max(np.abs(dev_lam - cpu_lam))) / scale:.2e} "
        f"dev_iters={dev_iters} cpu_niter={int(np.sum(d.solver_niter))}",
        flush=True)
  # Tiers tighten as fewer solver steps separate the implementations:
  # zero budget publishes formulas (near-exact); 1-2 budgets share one
  # trajectory step; ordinary budgets accumulate rounding over many steps.
  rtol = 1e-5 if budget == 0 else (1e-4 if budget <= 2 else 1e-3)
  atol = 1e-9 if budget == 0 else (1e-4 if budget <= 2 else 1e-3)
  np.testing.assert_allclose(dev_lam, cpu_lam, rtol=rtol, atol=atol * scale)
  nat = np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64)
  cpu = np.asarray(d.qacc, dtype=np.float64)
  np.testing.assert_allclose(nat, cpu, rtol=rtol, atol=atol)
  natf = np.asarray(s["qfrc_constraint"][0].cpu().numpy(), dtype=np.float64)
  cpuf = np.asarray(d.qfrc_constraint, dtype=np.float64)
  np.testing.assert_allclose(natf, cpuf, rtol=rtol, atol=atol)
  # Checkpoint restore: save, step, restore, re-solve reproduces the
  # pre-step assembly bit-identically (retained solver/warmstart state).
  lam0 = np.asarray(s["lambda"][0].cpu().numpy()).copy()
  snap = sim.snapshot()
  sim.step(1)
  sim.restore(snap)
  sr = sim.assembled_system(recompute=True)
  np.testing.assert_array_equal(
      np.asarray(sr["lambda"][0].cpu().numpy()), lam0)


@_needs_gpu()
@pytest.mark.parametrize("noslip", [0, 2], ids=["noslip0", "noslip2"])
def test_noslip_budget_contract_gpu(noslip):
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  model = _torus_model()
  model.opt.noslip_iterations = noslip
  qpos = _fixed_state()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  d = _cpu_forward(model, qpos)
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-3, atol=1e-3)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0


@_needs_gpu()
def test_refinement_fires_only_on_early_stop_gpu():
  from mujoco_metal.simulation import MetalSimulation
  import mujoco
  # Ordinary budget (early stop expected): refinement writes pair lows.
  model = _torus_model()
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=_fixed_state().astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  sim.assembled_system(recompute=True)
  cc = sim._coupled_constraints
  nv, nr = model.nv, cc.descriptor.nr
  dbg = cc._workspace["workspace_debug"].cpu().numpy()
  base = nr * nr + 7 * nr + cc._high_low_workspace_offset + 8 * max(nv, 1)
  lamlo = dbg[base + max(nr, 1):base + 2 * max(nr, 1)]
  assert float(np.max(np.abs(lamlo))) > 0.0
  # Zero budget: no polish. The lam_low plane must equal the solver's own
  # first-update force lows exactly (host mirror below); any hidden sweep
  # would move values far beyond pair rounding. All words finite.
  model0 = _torus_model()
  model0.opt.iterations = 0
  model0.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
  sim0 = MetalSimulation(model0, batch_size=1, profile="integrated_euler_v1")
  sim0.reset(qpos=_fixed_state().astype(np.float32).reshape(1, -1),
             qvel=np.zeros((1, model0.nv), dtype=np.float32))
  s0 = sim0.assembled_system(recompute=True)
  cc0 = sim0._coupled_constraints
  lay0 = cc0._jacobian_layout
  W0 = cc0._workspace["workspace_J"].cpu().numpy()
  Jhi = W0[lay0.values_offset:lay0.values_offset + lay0.nnz].reshape(nr, nv)
  Jlo = W0[lay0.lo_values_offset():lay0.lo_values_offset() + lay0.nnz].reshape(
      nr, nv)
  dbg0 = cc0._workspace["workspace_debug"].cpu().numpy()
  lamlo0 = dbg0[base:base + max(nr, 1) + max(nr, 1)][max(nr, 1):]
  assert np.all(np.isfinite(lamlo0))
  lam0 = np.asarray(s0["lambda"][0].cpu().numpy(), dtype=np.float64)
  R0 = np.asarray(s0["R"][0].cpu().numpy(), dtype=np.float64)
  ar0 = np.asarray(s0["ar"][0].cpu().numpy(), dtype=np.float64)
  acc = np.asarray(
      sim0.state._qacc[0].cpu().numpy(), dtype=np.float64)
  for row in range(8, 16):
    assert R0[row] > 0
    jaref = float((Jhi[row] + Jlo[row]) @ acc
                  - (ar0[row] + float(
                      dbg0[base + 2 * max(nr, 1) + row])))
    f0 = -jaref / R0[row]
    if jaref >= 0:
      assert lam0[row] == 0.0
      assert float(lamlo0[row]) == 0.0
    else:
      assert abs(lam0[row] - f0) <= 1e-6 * max(1.0, abs(f0))
      # Pair low completes the device value self-consistently (working
      # precision from float32 inputs, not bit-exact float64). A hidden
      # sweep would leave far larger traces (see budget zero-budget lam).
      assert abs(float(lamlo0[row]) - (f0 - lam0[row])) <= 1e-6 * abs(f0)


@_needs_gpu()
def test_zero_low_control_analytic_gpu():
  import mujoco
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><geom name='floor' type='plane' size='5 5 0.1'/>"
      "<body pos='0 0 0.3'><freejoint/>"
      "<geom name='ball' type='sphere' size='0.05'/></body></worldbody></mujoco>")
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  qpos = np.asarray(sim.state._qpos[0].cpu().numpy(), dtype=np.float64).copy()
  sim.reset(qpos=qpos.astype(np.float32).reshape(1, -1),
            qvel=np.zeros((1, model.nv), dtype=np.float32))
  s = sim.assembled_system(recompute=True)
  cc = sim._coupled_constraints
  lay = cc._jacobian_layout
  W = cc._workspace["workspace_J"].cpu().numpy()
  assert float(np.max(np.abs(
      W[lay.lo_values_offset():lay.lo_values_offset() + lay.nnz]))) == 0.0
  d = mujoco.MjData(model)
  d.qpos[:] = qpos
  d.qvel[:] = np.zeros(model.nv)
  mujoco.mj_forward(model, d)
  np.testing.assert_allclose(
      np.asarray(s["qacc"][0].cpu().numpy(), dtype=np.float64),
      np.asarray(d.qacc, dtype=np.float64), rtol=1e-4, atol=1e-4)
