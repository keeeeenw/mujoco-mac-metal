# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Independent scalar oracle for pinned elliptic PGS block bookkeeping."""

import os

import numpy as np
import mujoco
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints


def _pinned_first_elliptic_sweep(W, R, rhs, old, mu):
  """Closed-form special case of MuJoCo 3.10's first cold elliptic sweep.

  The matrix is diagonal in the tangent subspace and has no normal/tangent
  coupling, so the normal step and the constrained tangential QCQP have exact
  analytic solutions. This deliberately starts with zero normal force: the
  source still runs its friction update after the normal projected step.
  """
  A = np.asarray(W, dtype=np.float64) + np.diag(R)
  lam = np.asarray(old, dtype=np.float64).copy()
  residual = A @ lam - rhs
  old_lam = lam.copy()
  lam[0] = max(0.0, lam[0] - residual[0] / A[0, 0])
  tangent_rhs = residual[1:] + A[1:, 0] * (lam[0] - old_lam[0])
  tangent_A = A[1:, 1:]
  unconstrained = -np.linalg.solve(tangent_A, tangent_rhs)
  radius = float(mu) * lam[0]
  norm = np.linalg.norm(unconstrained)
  if norm > radius and norm > 0:
    unconstrained *= radius / norm
  lam[1:] = unconstrained
  delta = lam - old_lam
  change = 0.5 * delta @ A @ delta + delta @ residual
  if change > 1e-10:
    lam[:] = old_lam
    change = 0.0
  return lam, change


def test_cold_elliptic_block_runs_friction_update_after_normal_projection():
  # Residual is (-.25, 2, 0); the normal projection makes lambda_n=.1.
  # The pinned follow-up QCQP must then put the first tangent at the friction
  # boundary .8*.1. A branch that runs QCQP only when old lambda_n>0 returns
  # zero tangential force on this first sweep.
  W = np.diag([2.0, 1.0, 1.0])
  R = np.array([.5, .2, .2])
  rhs = np.array([.25, -2.0, 0.0])
  actual, change = _pinned_first_elliptic_sweep(
      W, R, rhs, np.zeros(3), mu=.8)
  np.testing.assert_allclose(actual, [.1, -.08, 0.0], rtol=0, atol=1e-14)
  assert change < 0.0


def test_regularized_delassus_residual_and_cost_use_same_operator():
  W = np.array([[2.0, .25, 0.0], [.25, 1.2, .1], [0.0, .1, .8]])
  R = np.array([.4, .3, .2])
  rhs = np.array([.1, -.2, .05])
  lam = np.array([.35, -.12, .09])
  full_residual = (W + np.diag(R)) @ lam - rhs
  unregularized_residual = W @ lam - rhs
  # Omitting R*lambda produces a materially different ray/QCQP RHS for an
  # accepted nonzero warm-start, while scalar diagonals already contain R.
  assert np.linalg.norm(full_residual - unregularized_residual) > .1
  delta = np.array([-.03, .08, -.04])
  source_cost_change = .5 * delta @ (W + np.diag(R)) @ delta + delta @ full_residual
  omitted_R_cost_change = .5 * delta @ (W + np.diag(R)) @ delta + delta @ unregularized_residual
  assert abs(source_cost_change - omitted_R_cost_change) > .005


def test_compiled_small_positive_mean_inertia_is_preserved():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body>
          <freejoint/>
          <inertial pos="0 0 0" mass="1e-8"
                    diaginertia="1e-10 2e-10 3e-10"/>
        </body>
      </worldbody>
    </mujoco>
  """)
  source_scale = float(model.stat.meaninertia)
  assert 0.0 < source_scale < 1e-6
  descriptor = lower_coupled_constraints(model)
  assert descriptor.mean_inertia == source_scale


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native gate")
@pytest.mark.parametrize("profile", ["integrated_euler_v1", "integrated_scalable_v1"])
def test_small_positive_meaninertia_pgs_matches_actual_pinned_engine(profile):
  from mujoco_metal import MetalSimulation
  from mujoco_metal.native_api import mj_forward

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" solver="PGS" iterations="80" tolerance="1e-8">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="1e-8" diaginertia="1e-10 2e-10 3e-10"/></body>
      <body pos="1 0 0"><joint name="b" type="slide" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="2e-8" diaginertia="2e-10 3e-10 4e-10"/></body>
      <body pos="2 0 0"><joint name="c" type="slide" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="3e-8" diaginertia="3e-10 4e-10 5e-10"/></body>
    </worldbody>
    <equality>
      <joint joint1="a" joint2="b" polycoef="0 1 0 0 0"/>
      <joint joint1="b" joint2="c" polycoef="0 .5 0 0 0"/>
    </equality>
  </mujoco>""")
  assert 0 < model.stat.meaninertia < 1e-6
  qpos = np.array([[.03, -.01, .02], [-.02, .015, -.03]], np.float32)
  qvel = np.array([[.2, -.1, .05], [-.15, .08, -.12]], np.float32)
  refs = []
  for env in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[env]
    data.qvel[:] = qvel[env]
    mujoco.mj_forward(model, data)
    refs.append(data)
  sim = MetalSimulation(model, 2, qpos=qpos, qvel=qvel, profile=profile)
  result = mj_forward(sim)
  actual = result["qacc"].detach().cpu().numpy()
  expected = np.stack([data.qacc for data in refs])
  np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=3e-5)
  force = result["constraint"]["qfrc_constraint"].detach().cpu().numpy()
  expected_force = np.stack([data.qfrc_constraint for data in refs])
  np.testing.assert_allclose(force, expected_force, rtol=5e-4, atol=3e-12)
  actual_counts = result["constraint"]["solver_diagnostics"][:, 1].cpu().numpy()
  expected_counts = [np.sum(data.solver_niter[:data.nisland]) for data in refs]
  np.testing.assert_array_equal(actual_counts, expected_counts)
