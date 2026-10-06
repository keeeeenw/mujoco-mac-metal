"""Original spatial-limit first-solve gate for the bounded PCG recovery work."""
import os

import mujoco
import numpy as np
import pytest


def _original_spatial_limit_model():
  xml = ('<mujoco><option timestep="0.002"/><worldbody>'
         '<site name="s0" pos="0 0 1"/><site name="s1" pos="0.2 0 1"/>'
         '<body pos="0 0 1"><joint name="j" type="slide" axis="1 0 0"/>'
         '<geom type="sphere" size="0.05" mass="1"/>'
         '<site name="s2" pos="0.1 0 0"/></body></worldbody>'
         '<tendon><spatial name="t"><site site="s0"/><site site="s2"/>'
         '<site site="s1"/></spatial></tendon></mujoco>')
  model = mujoco.MjModel.from_xml_string(xml)
  initial = mujoco.MjData(model)
  mujoco.mj_forward(model, initial)
  length = float(initial.ten_length[0])
  model.tendon_limited[0] = True
  model.tendon_range[0] = [length - 0.5, length + 0.05]
  model.tendon_margin[0] = 0.02
  return model


def _cpu_first_solve(model, qpos=0.3, qvel=1.0):
  data = mujoco.MjData(model)
  data.qpos[0] = qpos
  data.qvel[0] = qvel
  mujoco.mj_forward(model, data)
  assert data.nefc == 1
  mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, mass)
  jacobian = np.asarray(data.efc_J, dtype=np.float64).reshape(data.nefc,
                                                               model.nv).copy()
  regularizer = np.asarray(data.efc_R, dtype=np.float64).copy()
  rhs = -np.asarray(data.efc_b, dtype=np.float64).copy()
  lam = np.asarray(data.efc_force, dtype=np.float64).copy()
  delassus = jacobian @ np.linalg.solve(mass, jacobian.T)
  residual = (delassus + np.diag(regularizer)) @ lam - rhs
  if qpos == 0.3 and qvel == 1.0:
    np.testing.assert_allclose(lam, [296.0526315789474], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(data.qacc, [-592.1052631578948], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(data.qfrc_constraint, [-592.1052631578948],
                               rtol=2e-12, atol=2e-12)
  assert np.max(np.abs(residual)) < 1e-9
  return data, mass, jacobian, regularizer, rhs, lam, residual


def test_cpu_original_spatial_limit_first_solve_is_source_consistent():
  model = _original_spatial_limit_model()
  data, mass, jacobian, regularizer, rhs, lam, residual = _cpu_first_solve(model)
  # The source-reference values above use the original Python double inputs.
  # Also record the coefficients for the exact float32 state uploaded by the
  # public native fixture, promoted back to float64 for the CPU oracle.
  represented_qpos = float(np.float32(0.3))
  represented_qvel = float(np.float32(1.0))
  matched = _cpu_first_solve(model, represented_qpos, represented_qvel)
  np.testing.assert_allclose(matched[2], jacobian, atol=0.0, rtol=0.0)
  matched_data = matched[0]
  assert abs(float(matched[5][0]) - float(matched_data.qfrc_constraint[0] / matched[2][0, 0])) < 1e-10
  assert np.isfinite(mass).all() and np.isfinite(matched[3]).all()


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in original first-solve spatial-limit gate")
def test_native_original_spatial_limit_first_solve_matches_cpu_force_and_residual():
  import torch
  from mujoco_metal import MetalSimulation

  model = _original_spatial_limit_model()
  # Preserve and prove the original double-input CPU reference separately.
  double_cpu, double_mass, double_J, double_R, double_rhs, double_lambda, double_residual = _cpu_first_solve(model)
  represented_qpos = float(np.float32(0.3))
  represented_qvel = float(np.float32(1.0))
  cpu, mass, cpu_J, cpu_R, cpu_rhs, cpu_lambda, cpu_residual = _cpu_first_solve(
      model, represented_qpos, represented_qvel)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.asarray([[0.3]], dtype=np.float32),
            qvel=np.asarray([[1.0]], dtype=np.float32))
  cc = sim._coupled_constraints
  captures = []
  run_device = cc.run_device

  def capture_first_solver_result(*args, **kwargs):
    result = run_device(*args, **kwargs)
    rows = cc._assembly_views(include_optimizer_outputs=True,
                              materialize_jacobian=True)
    names = ("J", "R", "ar", "ar_low", "rhs", "lambda", "active",
             "qacc", "qacc_low", "qfrc_constraint", "status")
    captures.append({name: rows[name].detach().cpu().numpy().copy()
                     for name in names if name in rows})
    return result

  cc.run_device = capture_first_solver_result
  status = sim.step(1)
  np.testing.assert_array_equal(status.detach().cpu().numpy(), [0])
  assert len(captures) >= 1
  solved = captures[0]
  assert int(solved["status"].reshape(-1)[0]) == 0
  native_J = solved["J"].reshape(-1, model.nv)
  active = solved["active"].reshape(-1) > 0.5
  row = np.flatnonzero(active & (np.linalg.norm(native_J, axis=1) > 0.0))
  assert row.size == 1
  row = int(row[0])
  native_lambda = solved["lambda"].reshape(-1)[row]
  native_regularizer = solved["R"].reshape(-1)[row]
  native_row = native_J[row:row + 1].astype(np.float64)
  native_delassus = native_row @ np.linalg.solve(mass, native_row.T)
  native_residual = ((native_delassus[0, 0] + native_regularizer)
                     * native_lambda - cpu_rhs[0])

  # This is the original CPU source problem at the exact represented initial
  # qpos/qvel. Compare accepted multiplier/force and acceleration, then certify
  # the retained scalar row equation independently using the captured native J,
  # R and lambda with CPU M and b. The 40-step qpos gate remains unchanged.
  np.testing.assert_allclose(native_lambda, cpu_lambda[0], rtol=2e-5, atol=2e-3)
  np.testing.assert_allclose(solved["qfrc_constraint"].reshape(-1),
                             cpu.qfrc_constraint, rtol=1e-3, atol=1e-3)
  qacc_pair = (solved["qacc"].reshape(-1).astype(np.float64)
               + solved["qacc_low"].reshape(-1).astype(np.float64))
  np.testing.assert_allclose(solved["qacc"].reshape(-1), cpu.qacc,
                             rtol=1e-3, atol=1e-3)
  np.testing.assert_allclose(qacc_pair, cpu.qacc, rtol=1e-3, atol=1e-3)
  assert abs(float(native_residual)) < 1e-2, native_residual
  np.testing.assert_allclose(solved["ar"].reshape(-1)[row], cpu.efc_aref[0],
                             rtol=2e-5, atol=2e-3)
  # The double-input baseline is retained as a separate source proof; the
  # native parity check above uses the exact represented float32 upload.
  assert abs(float(double_residual[0])) < 1e-9
  assert np.isfinite(double_lambda).all() and np.isfinite(double_J).all()
  np.testing.assert_allclose(native_residual, cpu_residual[0], atol=1e-2,
                             rtol=0.0)
