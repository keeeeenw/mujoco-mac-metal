# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 015: Full implicit integrator (mjINT_IMPLICIT) qualification.

Verifies:
1. Pinned model lowering (nv <= 32, contact-free, mjINT_IMPLICIT).
2. MetalGeneralDenseSolve LU with partial pivoting on MPS.
3. Automatic nonsymmetric derivative assembly (passive, tendon, fluid, Coriolis).
4. Native MPS stepping parity against MuJoCo 3.10.0 CPU mj_step.
5. Exact checkpoint snapshot and restore reproducibility.
"""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.implicit import (
    lower_implicit,
    implicit_oracle,
    ImplicitProgram,
)
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.stepping import validate_stepping_profile

pytestmark_gpu = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
]


def _make_implicit_model(
    density=0.0, viscosity=0.0, damping=0.4, stiffness=0.0, nu=0, ntendon=False
):
  act_xml = "<actuator><motor joint='j1' gear='1.2'/></actuator>" if nu > 0 else ""
  ten_xml = (
      "<tendon><fixed name='t1'><joint joint='j1' coef='1.5'/><joint"
      " joint='j2' coef='-0.8'/></fixed></tendon>"
      if ntendon
      else ""
  )
  xml = f"""<mujoco>
  <option integrator="implicit" timestep="0.005" density="{density}" viscosity="{viscosity}">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <body name="b1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" damping="{damping}" stiffness="{stiffness}"/>
      <geom type="box" size="0.1 0.2 0.3" mass="1.0"/>
      <body name="b2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="{damping * 0.7}"/>
        <geom type="capsule" size="0.05 0.1" mass="0.6"/>
      </body>
    </body>
  </worldbody>
  {act_xml}
  {ten_xml}
</mujoco>"""
  return mujoco.MjModel.from_xml_string(xml)


def test_implicit_lowering_and_profile_validation_cpu():
  m = _make_implicit_model()
  desc = lower_implicit(m)
  assert desc.nv == 2
  assert desc.timestep == 0.005
  assert desc.auto_derivative is True

  # Profile validation
  prof = validate_stepping_profile(m, profile="contact_free_implicit_v1")
  assert prof.name == "contact_free_implicit_v1"
  assert "nonsymmetric implicit velocity solve" in str(prof.supported)

  # Rejection of wrong integrator
  m_euler = mujoco.MjModel.from_xml_string(
      """<mujoco><option integrator="Euler"><flag contact="disable"/></option>
      <worldbody><body pos="0 0 0"><joint type="hinge"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
  )
  with pytest.raises(ValueError, match="implicit integrator"):
    lower_implicit(m_euler)

  # Rejection of enabled contact
  m_contact = mujoco.MjModel.from_xml_string(
      """<mujoco><option integrator="implicit"/>
      <worldbody><body pos="0 0 0"><joint type="hinge"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
  )
  with pytest.raises(ValueError, match="contact explicitly disabled"):
    lower_implicit(m_contact)


def test_implicit_oracle_matches_dense_lu_cpu():
  m = _make_implicit_model(density=1.2, viscosity=0.01)
  d = mujoco.MjData(m)
  d.qpos[:] = [0.3, -0.4]
  d.qvel[:] = [1.2, -1.8]
  mujoco.mj_forward(m, d)

  M = np.empty((m.nv, m.nv), dtype=np.float64)
  mujoco.mj_fullM(m, d, M)

  # Nonsymmetric derivative
  deriv = np.array([[-0.52, 0.03], [-0.01, -0.31]], dtype=np.float64)
  oracle = implicit_oracle(m, M, d.qfrc_smooth, deriv)
  assert np.all(oracle["status"] == 0)

  H = M - m.opt.timestep * deriv
  expected_acc = np.linalg.solve(H, d.qfrc_smooth)
  np.testing.assert_allclose(oracle["qacc"][0], expected_acc, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_implicit_program_device_execution_gpu():
  import torch

  m = _make_implicit_model(density=1.2, viscosity=0.01)
  batch = 2
  prog = ImplicitProgram(m, batch, external_derivative=True)

  M = torch.eye(2, dtype=torch.float32, device="mps").unsqueeze(0).repeat(batch, 1, 1)
  M[:, 0, 1] = 0.2
  M[:, 1, 0] = 0.2
  force = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32, device="mps")
  deriv = torch.tensor(
      [[[-0.4, 0.1], [-0.05, -0.3]], [[-0.5, -0.02], [0.08, -0.25]]],
      dtype=torch.float32,
      device="mps",
  )

  res = prog.run_device(M, force, deriv)
  assert torch.all(res["status"] == 0)

  # Validate against CPU np.linalg.solve
  H_cpu = (M - m.opt.timestep * deriv).cpu().numpy()
  f_cpu = force.cpu().numpy()
  acc_cpu = np.stack([np.linalg.solve(H_cpu[b], f_cpu[b]) for b in range(batch)])
  np.testing.assert_allclose(res["qacc"].cpu().numpy(), acc_cpu, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_implicit_step_rollout_matches_mujoco_cpu():
  m = _make_implicit_model(density=1.2, viscosity=0.01, nu=1)
  batch = 2
  q0 = np.array([[0.2, -0.3], [-0.1, 0.4]], dtype=np.float32)
  v0 = np.array([[1.5, -2.0], [-0.8, 1.2]], dtype=np.float32)

  sim = MetalSimulation(m, batch, qpos=q0, qvel=v0, profile="contact_free_implicit_v1")

  d0 = mujoco.MjData(m)
  d0.qpos[:] = q0[0]
  d0.qvel[:] = v0[0]

  d1 = mujoco.MjData(m)
  d1.qpos[:] = q0[1]
  d1.qvel[:] = v0[1]

  ctrl = np.array([[0.4], [-0.25]], dtype=np.float32)

  for _ in range(80):
    sim.step(ctrl=ctrl)
    d0.ctrl[:] = ctrl[0]
    d1.ctrl[:] = ctrl[1]
    mujoco.mj_step(m, d0)
    mujoco.mj_step(m, d1)

  snap = sim.state.snapshot()
  assert np.all(snap.status == 0)

  np.testing.assert_allclose(snap.qpos[0], d0.qpos, atol=2e-5, rtol=1e-4)
  np.testing.assert_allclose(snap.qvel[0], d0.qvel, atol=2e-5, rtol=1e-4)
  np.testing.assert_allclose(snap.qpos[1], d1.qpos, atol=2e-5, rtol=1e-4)
  np.testing.assert_allclose(snap.qvel[1], d1.qvel, atol=2e-5, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_implicit_snapshot_restore_reproducibility():
  m = _make_implicit_model(density=1.0, viscosity=0.02, nu=1)
  sim = MetalSimulation(m, 1, profile="contact_free_implicit_v1")
  ctrl = np.array([[0.35]], dtype=np.float32)

  sim.step(15, ctrl=ctrl)
  checkpoint = sim.state.snapshot()

  sim.step(20, ctrl=ctrl)
  ref_qpos = sim.state.snapshot().qpos.copy()
  ref_qvel = sim.state.snapshot().qvel.copy()

  # Restore and step identical count
  sim.state.restore(checkpoint)
  sim.step(20, ctrl=ctrl)
  restored_snap = sim.state.snapshot()

  np.testing.assert_array_equal(restored_snap.qpos, ref_qpos)
  np.testing.assert_array_equal(restored_snap.qvel, ref_qvel)
