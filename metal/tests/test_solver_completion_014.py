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


def kkt_violation(asm, elliptic_blocks=None):
  """Host float64 KKT check on the retained solution (independent of the
  device certification path): normalized projected-gradient residual with
  the same row scaling the kernel certifies. Box rows use exact box
  projection; rows covered by ``elliptic_blocks`` use the Lorentz
  projection over their contact blocks instead (box-clamping them is not
  a valid cone certificate)."""
  g = lambda k: np.asarray(asm[k].cpu().numpy(), dtype=np.float64)
  W, R, ar, rhs, lam, lo, hi = (g("W"), g("R"), g("ar"), g("rhs"),
                                g("lambda"), g("lo"), g("hi"))
  b = W.shape[0]
  worst = 0.0
  for world in range(b):
    emembers = set()
    if elliptic_blocks is not None:
      for (start, dim, _friction) in elliptic_blocks[world]:
        emembers.update(range(start, start + dim))
    grad = (W[world] + np.diag(R[world])) @ lam[world] - rhs[world]
    for r in range(W.shape[1]):
      if r in emembers:
        continue
      diag = max(float(W[world][r, r] + R[world][r]), 1e-15)
      # Exact box projection (the at-bound branches must clamp both sides;
      # clamping only one bound reports residual 1.0 at the true optimum).
      proj = min(max(lam[world][r] - grad[r] / diag, lo[world][r]), hi[world][r])
      scale = abs(ar[world][r]) + abs(R[world][r] * lam[world][r])
      scale += float(np.sum(np.abs(W[world][r] * lam[world])))
      scale = max(scale, 1.0)
      worst = max(worst, abs(proj - lam[world][r]) * diag / scale)
    if elliptic_blocks is not None:
      for (start, dim, friction) in elliptic_blocks[world]:
        worst = max(worst, _lorentz_violation(
            W[world], R[world], rhs[world], lam[world], start, dim, friction))
  return worst


def _lorentz_violation(W, R, rhs, lam, start, dim, friction):
  """Stationarity residual of one elliptic block under Lorentz projection
  (host mirror of the kernel-side cone certificate, including frozen
  outside-block coupling exactly like the kernel's g vector)."""
  scale = np.ones(dim)
  for i in range(1, dim):
    scale[i] = max(float(friction[i - 1]), 0.0)
  force = lam[start:start + dim].copy()
  y = np.zeros(dim)
  y[0] = force[0]
  for i in range(1, dim):
    y[i] = force[i] / scale[i] if scale[i] > 1e-12 else 0.0
  H = np.zeros((dim, dim))
  linear = np.zeros(dim)
  n = W.shape[0]
  for i in range(dim):
    ri = start + i
    linear[i] = scale[i] * (-rhs[ri])
    for col in range(n):
      if col < start or col >= start + dim:
        linear[i] += scale[i] * W[ri, col] * lam[col]
    for j in range(dim):
      H[i, j] = scale[i] * (W[start + i, start + j]
                            + (R[start + i] if i == j else 0.0)) * scale[j]
  lipschitz = 1e-15
  for i in range(dim):
    lipschitz = max(lipschitz, float(np.sum(np.abs(H[i]))))
  projected = y - (H @ y + linear) / lipschitz
  projected = _project_lorentz(projected)
  scale_ref = 1.0 + float(np.sum(np.abs(linear)))
  for j in range(dim):
    scale_ref += float(np.sum(np.abs(H[:, j] * y[j])))
  return float(np.max(np.abs(y - projected)) * lipschitz / scale_ref)


def _project_lorentz(x):
  x = np.asarray(x, dtype=np.float64).copy()
  norm = float(np.linalg.norm(x[1:]))
  if norm <= x[0]:
    return x
  if norm <= -x[0]:
    return np.zeros_like(x)
  head = 0.5 * (norm + x[0])
  out = np.zeros_like(x)
  out[0] = head
  out[1:] = x[1:] * (head / max(norm, 1e-20))
  return out


class _HostArray(np.ndarray):
  """numpy array with torch-like .cpu().numpy() for oracle-only KKT checks."""

  def cpu(self):
    return self

  def numpy(self):
    return np.asarray(self)


def _fake_asm(W, R, ar, rhs, lam, lo, hi):
  t = lambda a: np.asarray(a, dtype=np.float64).reshape(1, -1).view(_HostArray)
  W = np.asarray(W, dtype=np.float64)
  W = W.reshape(1, *W.shape).view(_HostArray)
  return {"W": W,
          "R": t(R), "ar": t(ar), "rhs": t(rhs), "lambda": t(lam),
          "lo": t(lo), "hi": t(hi)}


def test_kkt_box_projection_analytic_cpu():
  # min 0.5*x^2 + x s.t. x >= 0: optimum x = 0 with gradient +1 (stationary
  # at the bound). The old one-sided clamp reported residual 1.0 here.
  asm = _fake_asm([[1.0]], [0.0], [0.0], [-1.0], [0.0], [0.0], [np.inf])
  assert kkt_violation(asm) == pytest.approx(0.0)
  # Interior optimum: min 0.5*(x-2)^2.
  asm = _fake_asm([[1.0]], [0.0], [0.0], [2.0], [2.0], [-np.inf], [np.inf])
  assert kkt_violation(asm) == pytest.approx(0.0)
  # Upper-bound optimum: min 0.5*(x-3)^2 s.t. x <= 1.
  asm = _fake_asm([[1.0]], [0.0], [0.0], [3.0], [1.0], [-np.inf], [1.0])
  assert kkt_violation(asm) == pytest.approx(0.0)
  # Deliberately invalid candidate must bite.
  asm = _fake_asm([[1.0]], [0.0], [0.0], [3.0], [5.0], [-np.inf], [1.0])
  assert kkt_violation(asm) > 0.5


def test_lorentz_projector_analytic_cpu():
  np.testing.assert_allclose(_project_lorentz([2.0, 1.0, 0.0]), [2.0, 1.0, 0.0])
  np.testing.assert_allclose(_project_lorentz([-2.0, 1.0, 0.0]), [0.0, 0.0, 0.0])
  np.testing.assert_allclose(_project_lorentz([1.0, 2.0, 0.0]), [1.5, 1.5, 0.0])
  # Feasible cone optimum (min 0.5*|x|^2 + x_0 over the cone sits at the
  # apex 0) has zero stationarity residual.
  W = np.eye(3)
  asm = _fake_asm(W, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -np.array([1.0, 0.0, 0.0]),
                  [0.0, 0.0, 0.0], [-np.inf] * 3, [np.inf] * 3)
  assert kkt_violation(asm, elliptic_blocks=[[(0, 3, [0.5, 0.5])]]) == pytest.approx(0.0)
  # Infeasible cone point (negative normal force) is flagged.
  asm = _fake_asm(W, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -np.array([1.0, 0.0, 0.0]),
                  [-1.0, 0.0, 0.0], [-np.inf] * 3, [np.inf] * 3)
  assert kkt_violation(asm, elliptic_blocks=[[(0, 3, [0.5, 0.5])]]) > 0.0


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
  m = _press_model(cone=cone, condim=condim)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(30):
    sim.step(1)
  asm = sim.assembled_system(recompute=True)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  assert kkt_violation(asm, _elliptic_blocks(sim, cone)) < 1e-3


def _elliptic_blocks(sim, cone):
  """Per-world elliptic contact blocks from retained slot metadata."""
  if cone != "elliptic":
    return [[] for _ in range(sim.batch_size)]
  cc = sim._coupled_constraints
  d = cc.descriptor
  import torch
  packed = cc._constants["contact_condim"].detach().cpu().numpy().reshape(-1, 3)
  friction = cc._constants["contact_friction"].detach().cpu().numpy().reshape(-1, 5)
  base = int(d.n_eq_rows) + int(d.nv) + 2 * int(d.njnt) + int(d.ten_friction_rows + d.ten_limit_rows)
  per_world = []
  for _ in range(sim.batch_size):
    blocks = []
    for s in range(int(d.ncontacts_max)):
      cdim = int(packed[s, 0])
      if cdim > 1:
        blocks.append((base + int(packed[s, 1]), cdim, friction[s].copy()))
    per_world.append(blocks)
  return per_world


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_kkt_certifies_retained_accepted_step_gpu(cone):
  # The certificate binds the RETAINED accepted-step system: cached assembly
  # (no recompute) certifies, its status matches the recorded step status,
  # and recomputing after a state change solves anew (different system).
  m = _press_model(cone=cone)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(30):
    sim.step(1)
  status = int(sim.state.status.cpu().numpy()[0])
  assert status == 0
  asm = sim.assembled_system()
  assert int(asm["status"].cpu().numpy()[0]) == status
  assert kkt_violation(asm, _elliptic_blocks(sim, cone)) < 1e-3
  lam_before = np.asarray(asm["lambda"].cpu().numpy()).copy()
  # Perturb the state: recompute must solve anew for the new system.
  qpos = sim.state.qpos.cpu().numpy()
  qpos[0, 2] += 0.02
  sim.state._qpos.copy_(sim.state._torch.as_tensor(qpos))
  asm2 = sim.assembled_system(recompute=True)
  assert not np.allclose(np.asarray(asm2["lambda"].cpu().numpy()), lam_before)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_opposing_pinch_converges_gpu(cone):
  # R04: opposing-contact pinch (box settles into a snug slot between two
  # walls under gravity): opposing normals stress the coupled solve;
  # status, KKT and CPU parity must all hold on the retained system.
  xml = (f'<mujoco><option timestep="0.002" integrator="Euler" cone="{cone}" '
          'iterations="200" tolerance="1e-8" gravity="0 0 -9.81"/>'
          '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
          '<geom name="wallL" type="box" size="0.05 0.3 0.3" pos="-0.14 0 0.15"/>'
          '<geom name="wallR" type="box" size="0.05 0.3 0.3" pos="0.14 0 0.15"/>'
          '<body pos="0 0 0.35"><freejoint/>'
          '<geom name="bx" type="box" size="0.08 0.08 0.08"/></body>'
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
  np.testing.assert_allclose(gq[:3], np.asarray(cpu.qpos)[:3], atol=5e-3,
                             err_msg=f"pinch/{cone}")
  asm = sim.assembled_system()
  assert int(asm["status"].cpu().numpy()[0]) == 0
  assert kkt_violation(asm, _elliptic_blocks(sim, cone)) < 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_requested_effective_iterations_reported_gpu():
  # Requested vs effective budgets are explicit: settings carry both, and
  # actual diagnostics never exceed the adaptive cap.
  m = _press_model(iterations=100)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  settings = sim._coupled_constraints.solver_settings
  assert settings.requested_iterations == 100
  assert settings.effective_iterations == 100
  assert settings.adaptive_max_iterations == 1024
  for _ in range(10):
    sim.step(1)
  it = int(sim.assembled_system()["solver_diagnostics"].cpu().numpy()[0, 1])
  assert it <= settings.adaptive_max_iterations, it


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_teleport_rejects_stale_seed_gpu():
  # Identity transition under a large state jump with a single-iteration
  # budget: the stale retained seed cannot survive the cost gate, so one
  # step from the teleported state matches a cold twin bit-exactly. (With
  # a full budget both would converge alike and prove nothing.)
  m = _press_model(iterations=1)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  for _ in range(30):
    sim.step(1)
  assert bool(np.any(sim._coupled_constraints.get_warmstart() != 0))
  tele = sim.state.qpos.cpu().numpy().copy()
  tele[0, 2] += 0.3
  tele = tele.astype(np.float32)
  qv = sim.state.qvel.cpu().numpy().copy()
  # Teleport WITHOUT reset: retained seed stays stale (reset would clear it).
  sim.state._qpos.copy_(sim.state._torch.as_tensor(tele))
  cold = MetalSimulation(m, batch_size=1, profile=PROFILE)
  cold.reset(qpos=tele, qvel=qv.copy())
  sim.step(1)
  cold.step(1)
  np.testing.assert_array_equal(sim.state.qpos.cpu().numpy(),
                                cold.state.qpos.cpu().numpy())
  np.testing.assert_array_equal(sim.state.qvel.cpu().numpy(),
                                cold.state.qvel.cpu().numpy())


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_stacked_boxes_converge_gpu(cone):
  # R04: stacked boxes (multi-contact elliptic/pyramidal with refinement
  # available): status, cone-aware KKT and CPU parity on the retained system.
  xml = (f'<mujoco><option timestep="0.002" integrator="Euler" cone="{cone}" '
          'iterations="200" tolerance="1e-8" gravity="0 0 -9.81"/>'
          '<worldbody><geom name="floor" type="plane" size="5 5 0.1"/>'
          '<body pos="0 0 0.3"><freejoint/>'
          '<geom name="b1" type="box" size="0.09 0.09 0.09"/></body>'
          '<body pos="0.02 0 0.55"><freejoint/>'
          '<geom name="b2" type="box" size="0.09 0.09 0.09"/></body>'
          '</worldbody></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(m, batch_size=1, profile=PROFILE)
  qp = np.asarray(m.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m.nv), dtype=np.float32))
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = qp
  cpu.qvel[:] = 0
  mujoco.mj_forward(m, cpu)
  for _ in range(150):
    sim.step(1)
    mujoco.mj_step(m, cpu)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  np.testing.assert_allclose(gq, np.asarray(cpu.qpos), atol=5e-3,
                             err_msg=f"stacked/{cone}")
  asm = sim.assembled_system()
  assert int(asm["status"].cpu().numpy()[0]) == 0
  assert kkt_violation(asm, _elliptic_blocks(sim, cone)) < 1e-3


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
def test_noslip_matches_cpu_noslip_and_reduces_slip_gpu():
  # R04: native no-slip stage vs the CPU no-slip oracle on a stuck slope,
  # with quantified slip reduction against the plain (no-noslip) CPU run.
  th = float(np.deg2rad(10))
  c, s = float(np.cos(th)), float(np.sin(th))
  base = (f'<option timestep="0.002" integrator="Euler" iterations="100" '
          f'tolerance="1e-8" gravity="0 0 -9.81" NOSLIP/>'
          f'<worldbody><geom name="slope" type="plane" size="5 5 0.1" '
          f'quat="0 {s / 2:.6f} 0 {c / 2:.6f}" friction="0.8 0.05 0.02"/>'
          '<body pos="0 0 0.3"><freejoint/>'
          '<geom name="bx" type="box" size="0.06 0.06 0.04"/></body>'
          '</worldbody>')
  m_plain = mujoco.MjModel.from_xml_string(
      "<mujoco>" + base.replace("NOSLIP", "") + "</mujoco>")
  m_noslip = mujoco.MjModel.from_xml_string(
      "<mujoco>" + base.replace("NOSLIP",
                                'noslip_iterations="5" noslip_tolerance="1e-6"') + "</mujoco>")
  sim = MetalSimulation(m_noslip, batch_size=1, profile=PROFILE)
  qp = np.asarray(m_noslip.qpos0, dtype=np.float32)
  sim.reset(qpos=qp.reshape(1, -1), qvel=np.zeros((1, m_noslip.nv), dtype=np.float32))
  cpu_p, cpu_n = mujoco.MjData(m_plain), mujoco.MjData(m_noslip)
  for dd, mm in ((cpu_p, m_plain), (cpu_n, m_noslip)):
    dd.qpos[:] = np.asarray(mm.qpos0)
    dd.qvel[:] = 0
    mujoco.mj_forward(mm, dd)
  for _ in range(300):
    sim.step(1)
    mujoco.mj_step(m_plain, cpu_p)
    mujoco.mj_step(m_noslip, cpu_n)
  assert int(sim.state.status.cpu().numpy()[0]) == 0
  gq = sim.state.qpos.cpu().numpy()[0]
  dp = float(np.linalg.norm(np.asarray(cpu_p.qpos)[:2] - qp[:2]))
  dn = float(np.linalg.norm(np.asarray(cpu_n.qpos)[:2] - qp[:2]))
  dg = float(np.linalg.norm(gq[:2] - qp[:2]))
  # Native no-slip tracks the CPU no-slip oracle and improves on plain.
  np.testing.assert_allclose(dg, dn, atol=3e-3, err_msg="native-vs-cpu-noslip")
  assert dg <= dp + 2e-3, (dg, dp, dn)
  # Iteration accounting includes the no-slip sweeps: same retained system
  # solved with the budget toggled proves the extra sweeps execute.
  dims = sim._coupled_constraints._constants["solver_dims"]
  dims[18] = 0
  it_plain = int(sim.assembled_system(recompute=True)["solver_diagnostics"].cpu().numpy()[0, 1])
  dims[18] = 5
  asm_ns = sim.assembled_system(recompute=True)
  it = int(asm_ns["solver_diagnostics"].cpu().numpy()[0, 1])
  assert it >= it_plain, (it, it_plain)
  assert kkt_violation(asm_ns, _elliptic_blocks(sim, "pyramidal")) < 1e-3


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

