# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 014: solver options, warm starts and completion evidence.

Covers the PGS/Newton/CG selector mapping (all run the native projected
solver), cold/warm starts with the cost-gated retained multipliers,
convergence histories, iteration-contract behavior (adaptive extension),
KKT residual evidence on the retained solution, low budgets, converged
limits, contact transitions, redundant/ill-conditioned systems, no-slip
behavior (stick/slip + noslip-stage subsumption probe) across both cones.
"""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

PROFILE = "integrated_euler_v1"
QID = (1.0, 0.0, 0.0, 0.0)


def _press_model(cone="pyramidal", condim=3, solver="PGS", iterations=100,
                 tol="1e-8", extra=""):
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option timestep="0.002" integrator="Euler" cone="{cone}" '
      f'solver="{solver}" iterations="{iterations}" tolerance="{tol}"/>{extra}'
      '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
      '<body pos="0 0 0.04"><freejoint/>'
      f'<geom name="ball" type="sphere" size="0.05" condim="{condim}"/></body>'
      '</worldbody></mujoco>')


def kkt_violation(asm):
  """Host float64 KKT check on the retained solution (independent of the
  device certification path): normalized projected-gradient residual with
  the same row scaling the kernel certifies."""
  g = lambda k: np.asarray(asm[k].cpu().numpy(), dtype=np.float64)
  W, R, ar, rhs, lam, lo, hi = (g("W"), g("R"), g("ar"), g("rhs"),
                                g("lambda"), g("lo"), g("hi"))
  b = W.shape[0]
  worst = 0.0
  for world in range(b):
    grad = (W[world] + np.diag(R[world])) @ lam[world] - rhs[world]
    for r in range(W.shape[1]):
      diag = max(float(W[world][r, r] + R[world][r]), 1e-15)
      if lam[world][r] <= lo[world][r] + 1e-9:
        proj = min(lam[world][r] - grad[r] / diag, hi[world][r])
      elif lam[world][r] >= hi[world][r] - 1e-9:
        proj = max(lam[world][r] - grad[r] / diag, lo[world][r])
      else:
        proj = lam[world][r] - grad[r] / diag
      scale = abs(ar[world][r]) + abs(R[world][r] * lam[world][r])
      scale += float(np.sum(np.abs(W[world][r] * lam[world])))
      scale = max(scale, 1.0)
      worst = max(worst, abs(proj - lam[world][r]) * diag / scale)
  return worst


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_solver_selection_parity_gpu():
  # PGS/Newton/CG selections run identical native code (mapped); all match
  # the CPU PGS oracle on a sustained press.
  refs = {}
  for solver in ("PGS", "Newton", "CG"):
    m = _press_model(solver=solver)
    sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
    qp = np.asarray(m.qpos0, dtype=np.float32)
    sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
    for _ in range(50):
      sim.step(1)
    refs[solver] = (sim.state.qpos.cpu().numpy().copy(),
                    sim.state.qacc.cpu().numpy().copy())
  np.testing.assert_array_equal(refs["PGS"][0], refs["CG"][0])
  np.testing.assert_array_equal(refs["PGS"][1], refs["CG"][1])
  np.testing.assert_array_equal(refs["PGS"][0], refs["Newton"][0])
  m = _press_model(solver="PGS")
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = m.qpos0
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(50):
    mujoco.mj_step(m, cpu)
  np.testing.assert_allclose(refs["PGS"][0][0, :3], np.asarray(cpu.qpos)[:3],
                             atol=2e-3, err_msg="native-vs-cpu press")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_warm_beats_cold_gpu(cone):
  # Sustained press: warm starts certify in fewer total iterations than
  # clearing every step, reaching the same physics. Iteration counts are
  # read directly from the step workspace (assembled_system would recompute
  # at the post-step state and mask the step-save).
  def run(warm):
    m = _press_model(cone=cone)
    sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
    qp = np.asarray(m.qpos0, dtype=np.float32)
    sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
    total = 0
    for _ in range(40):
      if not warm:
        sim.clear_warmstart()
      sim.step(1)
      w = sim._coupled_constraints._workspace["out_diagnostics"][:10].reshape(10)
      total += int(w[1].detach().cpu().numpy())
    return total, sim.state.qpos.cpu().numpy().copy()
  cold_iters, cold_q = run(False)
  warm_iters, warm_q = run(True)
  assert warm_iters < cold_iters, (warm_iters, cold_iters)
  np.testing.assert_allclose(warm_q, cold_q, atol=1e-4, err_msg="warm-vs-cold")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_low_iteration_budget_gpu():
  # iterations=1 stays finite and reports honestly; physics degrades
  # gracefully without crashing or claiming false certification.
  m = _press_model(iterations=1)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(20):
    sim.step(1)
  asm = sim.assembled_system()
  diag = np.asarray(asm["solver_diagnostics"].cpu().numpy()[0])
  hist = np.asarray(asm["solver_history"].cpu().numpy()[0])
  assert np.all(np.isfinite(diag)) and np.all(np.isfinite(hist))
  assert int(diag[1]) <= 2, diag
  q = sim.state.qpos.cpu().numpy()[0]
  assert np.all(np.isfinite(q)) and q[2] > -0.05


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_converged_limit_contact_gpu():
  # Joint limit + contact converge together and match the CPU oracle.
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" iterations="100" '
         'tolerance="1e-8" gravity="0 0 -9.81"/>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body pos="0 0 0.3"><joint name="slide" type="slide" axis="0 0 1" '
         'limited="true" range="-0.25 0.0"/>'
         '<geom name="ball" type="sphere" size="0.06"/></body>'
         '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(200):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq[0], float(np.asarray(cpu.qpos)[0]),
                             atol=5e-3, err_msg="limit+contact")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_contact_transition_envelope_gpu():
  # Bouncing ball (contact on/off transitions) tracks the CPU oracle
  # within an envelope and never reports false convergence.
  m = _press_model()
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.array([0, 0, 0.4, *QID], dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  worst = 0.0
  for _ in range(250):
    sim.step(1)
    mujoco.mj_step(m, cpu)
    assert int(sim.state.status.cpu().numpy()[0]) == 0
    gq = sim.state.qpos.cpu().numpy()[0]
    worst = max(worst, float(np.max(np.abs(gq[:3] - np.asarray(cpu.qpos)[:3]))))
  assert worst < 2e-2, worst


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_redundant_face_rows_gpu(cone):
  # Box face plant (redundant manifold) certifies and matches CPU height.
  xml = (f'<mujoco><option timestep="0.002" integrator="Euler" cone="{cone}" '
         'iterations="200" tolerance="1e-8" gravity="0 0 -9.81"/>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body pos="0 0 0.2"><freejoint/>'
         '<geom name="bx" type="box" size="0.08 0.08 0.05"/></body>'
         '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(300):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq[2], float(np.asarray(cpu.qpos)[2]),
                             atol=5e-3, err_msg=f"redundant/{cone}")
  assert kkt_violation(sim.assembled_system(recompute=True)) < 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_soft_stiff_mix_gpu():
  # Soft spring joint + hard contact: mixed conditioning converges and the
  # spring equilibrium matches the oracle.
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" iterations="200" '
         'tolerance="1e-8" gravity="0 0 -9.81"/>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body pos="0 0 0.3"><joint name="soft" type="slide" axis="0 0 1" '
         'stiffness="20" damping="2"/>'
         '<geom name="ball" type="sphere" size="0.06"/></body>'
         '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(400):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq[0], float(np.asarray(cpu.qpos)[0]),
                             atol=5e-3, err_msg="soft-stiff")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_kkt_retained_solution_gpu(condim, cone):
  # Independent float64 KKT residual on the retained complete solution and
  # the actual assembled system (never a helper-only certificate).
  if cone == "elliptic" and condim in (4, 6):
    pytest.skip("elliptic condim 4/6 covered by dedicated suites")
  m = _press_model(cone=cone, condim=condim)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(30):
    sim.step(1)
  asm = sim.assembled_system(recompute=True)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  assert kkt_violation(asm) < 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("tilt_deg,stuck", [(10, True), (50, False)])
def test_noslip_stick_slip_gpu(tilt_deg, stuck):
  # No-slip behavior: below the friction angle the block sticks (no slip);
  # above it slides. Both engines agree on the regime.
  th = float(np.deg2rad(tilt_deg))
  c, s = float(np.cos(th)), float(np.sin(th))
  xml = (f'<mujoco><option timestep="0.002" integrator="Euler" iterations="100" '
         f'tolerance="1e-8" gravity="0 0 -9.81"/>'
         f'<worldbody><geom name="slope" type="plane" size="5 5 0.1" '
         f'quat="0 {s / 2:.6f} 0 {c / 2:.6f}" friction="0.8 0.05 0.02"/>'
         '<body pos="0 0 0.3"><freejoint/>'
         '<geom name="bx" type="box" size="0.06 0.06 0.04"/></body>'
         '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(300):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  cq = np.asarray(cpu.qpos)
  moved_nat = float(np.linalg.norm(gq[:2] - qp[:2]))
  moved_cpu = float(np.linalg.norm(cq[:2] - qp[:2]))
  if stuck:
    # Bounded drop-settle transient, identical on both engines (stick =
    # no runaway slide, not zero micro-motion).
    assert abs(moved_nat - moved_cpu) < 2e-3, (moved_nat, moved_cpu)
    assert moved_nat < 3e-2 and moved_cpu < 3e-2, (moved_nat, moved_cpu)
  else:
    assert moved_nat > 0.1 and moved_cpu > 0.1, (moved_nat, moved_cpu)
    np.testing.assert_allclose(gq[:2], cq[:2], atol=3e-2, err_msg="slide")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_noslip_stage_subsumption_gpu():
  # The native every-step block refinement subsumes the pinned separate
  # no-slip stage: native (plain model) matches the CPU oracle WITH noslip
  # iterations enabled at least as well as CPU-without.
  base = ('<option timestep="0.002" integrator="Euler" iterations="100" '
          'tolerance="1e-8" gravity="0 0 -9.81" NOSLIP/>'
          '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
          '<body pos="0.05 0 0.2"><freejoint/>'
          '<geom name="bx" type="box" size="0.08 0.08 0.05" friction="0.6 0.05 0.02"/>'
          '</body></worldbody></mujoco>')
  m_plain = mujoco.MjModel.from_xml_string(
      "<mujoco>" + base.replace("NOSLIP", "") + "</mujoco>")
  m_noslip = mujoco.MjModel.from_xml_string(
      "<mujoco>" + base.replace("NOSLIP",
                                'noslip_iterations="5" noslip_tolerance="1e-6"') + "</mujoco>")
  sim = MetalSimulation(m_plain, batch_size=1, profile=PROFILE)
  qp = np.asarray(m_plain.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m_plain.nv), dtype=np.float32))
  cpu_p = mujoco.MjData(m_plain)
  cpu_p.qpos[:] = qp
  cpu_p.qvel[:] = 0
  mujoco.mj_forward(m_plain, cpu_p)
  cpu_n = mujoco.MjData(m_noslip)
  cpu_n.qpos[:] = qp
  cpu_n.qvel[:] = 0
  mujoco.mj_forward(m_noslip, cpu_n)
  for _ in range(200):
    sim.step(1)
    mujoco.mj_step(m_plain, cpu_p)
    mujoco.mj_step(m_noslip, cpu_n)
  gq = sim.state.qpos.cpu().numpy()[0]
  dp = float(np.max(np.abs(gq[:3] - np.asarray(cpu_p.qpos)[:3])))
  dn = float(np.max(np.abs(gq[:3] - np.asarray(cpu_n.qpos)[:3])))
  assert dn <= dp + 2e-3, (dp, dn)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_iteration_contract_extension_gpu():
  # The explicit G3 contract: certification may use adaptive extension
  # past the requested budget (bounded by 1024) while improving; the
  # diagnostics report actual counts separately from configured budgets.
  m = _press_model(iterations=100)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(10):
    sim.step(1)
  asm = sim.assembled_system()
  it = int(asm["solver_diagnostics"].cpu().numpy()[0, 1])
  assert it <= 1024, it
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  assert sim._coupled_constraints.solver_settings.requested_iterations == 100
  assert sim._coupled_constraints.solver_settings.adaptive_max_iterations == 1024
  # Contact-free problems certify within a single sweep.
  m2 = mujoco.MjModel.from_xml_string(
      '<mujoco><option timestep="0.002" integrator="Euler" iterations="100" '
      'tolerance="1e-8"/><worldbody>'
      '<body pos="0 0 1"><freejoint/><geom name="b" type="sphere" size="0.05"/>'
      '</body></worldbody></mujoco>')
  sim2 = MetalSimulation(m2, batch_size=1, profile=PROFILE)
  q2 = np.asarray(m2.qpos0, dtype=np.float32)
  sim2.reset(qpos=q2.reshape(1, -1), qvel=np.zeros((1, m2.nv), dtype=np.float32))
  sim2.step(1)
  it2 = int(sim2.assembled_system()["solver_diagnostics"].cpu().numpy()[0, 1])
  assert it2 <= 100, it2


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_warmstart_cost_guard_gpu():
  # Garbage warm vectors are rejected by the cost check: physics matches a
  # cold run even when seeded with huge forces.
  m = _press_model()
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  nr = sim._coupled_constraints.descriptor.nr
  sim.set_warmstart(np.full(nr, 1e6, dtype=np.float32))
  for _ in range(20):
    sim.step(1)
  poisoned = sim.state.qpos.cpu().numpy().copy()
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(20):
    sim.step(1)
  cold = sim.state.qpos.cpu().numpy().copy()
  np.testing.assert_allclose(poisoned, cold, atol=1e-4, err_msg="cost-guard")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_warmstart_disable_flag_gpu():
  # With the WARMSTART disable flag, retained vectors are ignored: seeding
  # garbage behaves bit-identically to clearing.
  xml = ('<mujoco><option timestep="0.002" integrator="Euler" iterations="100" '
         'tolerance="1e-8"><flag warmstart="disable"/></option>'
         '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
         '<body pos="0 0 0.04"><freejoint/>'
         '<geom name="ball" type="sphere" size="0.05"/></body>'
         '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  nr = sim._coupled_constraints.descriptor.nr
  sim.set_warmstart(np.full(nr, 3.0, dtype=np.float32))
  for _ in range(10):
    sim.step(1)
  seeded = (sim.state.qpos.cpu().numpy().copy(),
            sim.assembled_system()["solver_diagnostics"].cpu().numpy().copy())
  sim.clear_warmstart()
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(10):
    sim.step(1)
  cleared = (sim.state.qpos.cpu().numpy().copy(),
             sim.assembled_system()["solver_diagnostics"].cpu().numpy().copy())
  np.testing.assert_array_equal(seeded[0], cleared[0])
  np.testing.assert_array_equal(seeded[1], cleared[1])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_warmstart_api_gpu():
  m = _press_model()
  sim = MetalSimulation(m, batch_size=2, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=np.stack([qp, qp]), qvel=np.zeros((2, m.nv), dtype=np.float32))
  nr = sim._coupled_constraints.descriptor.nr
  w0 = sim.get_warmstart()
  assert w0.shape == (2, nr) and np.all(w0 == 0)
  # Broadcast + per-env set.
  sim.set_warmstart(np.full(nr, 0.5, dtype=np.float32))
  np.testing.assert_allclose(sim.get_warmstart(), 0.5)
  sim.set_warmstart(np.zeros(nr, dtype=np.float32), env_ids=[1])
  got = sim.get_warmstart()
  assert float(np.max(got[0])) == 0.5 and float(np.max(got[1])) == 0.0
  # Invalid: shape, nonfinite, bad/duplicate ids.
  with pytest.raises(ValueError, match="shape"):
    sim.set_warmstart(np.zeros(nr + 1, dtype=np.float32))
  with pytest.raises(ValueError, match="finite"):
    sim.set_warmstart(np.full(nr, np.inf))
  with pytest.raises(ValueError, match="out of range"):
    sim.set_warmstart(np.zeros(nr), env_ids=[7])
  with pytest.raises(ValueError, match="duplicate"):
    sim.set_warmstart(np.zeros((2, nr)), env_ids=[0, 0])
  with pytest.raises(ValueError, match="at least one"):
    sim.clear_warmstart(env_ids=[])
  # Reset clears.
  sim.set_warmstart(np.full((2, nr), 2.0, dtype=np.float32))
  sim.reset()
  np.testing.assert_allclose(sim.get_warmstart(), 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_history_contract_gpu():
  # Histories are finite, nonnegative and overall decreasing across the
  # main sweep; a converged step certifies within tolerance.
  m = _press_model()
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(10):
    sim.step(1)
  asm = sim.assembled_system(recompute=True)
  hist = np.asarray(asm["solver_history"].cpu().numpy()[0])
  diag = np.asarray(asm["solver_diagnostics"].cpu().numpy()[0])
  assert hist.shape == (8,)
  assert np.all(np.isfinite(hist)) and np.all(hist >= 0)
  assert hist[0] >= hist[7], hist
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  assert diag[0] <= float(np.asarray(m.opt.tolerance)) * 10 + 1e-6

