# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""R06b repair: automatic implicitfast velocity derivatives.

Failing-first: the native implicitfast path must assemble passive-damper
(linear + polynomial) and tendon-damping velocity derivatives automatically
instead of rejecting those models. Guards remain for fluid Jacobians and
velocity-dependent actuator forces.
"""

import mujoco
import numpy as np
import pytest


def _hinge_model(damping="0.3", dampingpoly=None, tendon=True):
  # NOTE: MuJoCo XML has no joint dampingpoly attribute; polynomial
  # damping is configured on the compiled field (see REVIEW_UPDATE §8).
  tendon_xml = ("<tendon><fixed name='t'><joint joint='h' coef='2'/></fixed></tendon>"
                if tendon else "")
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler'/>"
      "<worldbody><body pos='0 0 0.5'>"
      f"<joint name='h' type='hinge' axis='0 0 1' damping='{damping}'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody>"
      f"{tendon_xml}</mujoco>")
  if dampingpoly is not None:
    m.dof_dampingpoly[0] = dampingpoly
  return m


def test_automatic_derivative_matches_finite_differences_cpu():
  from mujoco_metal.implicit import implicit_derivative_reference
  m = _hinge_model(damping="0.3", dampingpoly=[0.1, 0.02])
  m.tendon_damping[0] = 0.5  # nonzero tendon tangent: J' diag J block
  qpos = np.array([[0.4]])
  qvel = np.array([[1.7]])
  got = implicit_derivative_reference(m, qpos, qvel)
  assert got.shape == (1, 1, 1)
  # Non-vacuous: passive (0.8134) + tendon (0.5*4=2.0) tangents.
  np.testing.assert_allclose(got[0, 0, 0], -(0.8134 + 2.0), rtol=1e-4)
  # Independent central finite differences of the smooth force sum.
  from mujoco_metal.passive import PassiveForceModel
  from mujoco_metal.tendons import FixedTendonModel
  passive = PassiveForceModel(m)
  tendon = FixedTendonModel(m)
  h = 1e-6
  fp = lambda v: passive.force(qpos, v) + tendon.run(qpos, v)[0]
  fd = (fp(qvel + h) - fp(qvel - h)) / (2 * h)
  np.testing.assert_allclose(got[0], fd, rtol=1e-4, atol=1e-6)


def test_automatic_derivative_admission_cpu():
  from mujoco_metal.implicit import implicitfast_supports_automatic
  assert implicitfast_supports_automatic(_hinge_model()) is True
  # Fluid Jacobians stay rejected.
  m = mujoco.MjModel.from_xml_string(
      "<mujoco><option timestep='0.002' integrator='Euler' density='1000'/>"
      "<worldbody><body pos='0 0 0.5'>"
      "<joint name='h' type='hinge' axis='0 0 1'/>"
      "<geom type='sphere' size='0.1' mass='0.5'/>"
      "</body></worldbody></mujoco>")
  assert implicitfast_supports_automatic(m) is False
  with pytest.raises(ValueError, match="fluid"):
    from mujoco_metal.implicit import implicit_derivative_reference
    implicit_derivative_reference(m, np.array([[0.0]]), np.array([[0.0]]))


def _needs_gpu():
  import os
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")


@_needs_gpu()
def test_automatic_derivative_native_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.implicit import (
      ImplicitFastProgram, implicit_derivative_reference, implicitfast_oracle,
  )
  from mujoco_metal.model import load_model
  from mujoco_metal.passive import PassiveForceModel
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  import torch
  m = _hinge_model(damping="0.3", dampingpoly=[0.1, 0.02])
  m.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
  m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  desc = load_model(m)
  b = 2
  qpf = np.array([[0.4], [-0.2]])
  qvf = np.array([[1.7], [-0.9]])
  qp = torch.as_tensor(qpf.astype(np.float32), device="mps")
  qv = torch.as_tensor(qvf.astype(np.float32), device="mps")
  smooth = MetalSmoothDynamics(desc, batch_size=b)
  dyn = smooth.run_device(qp, qv, None, None)
  # Full smooth force (bias + passive); the tendon here is undamped and
  # stiffness-free, contributing nothing.
  pf = PassiveForceModel(m).force(qpf, qvf)
  rhs = dyn["qfrc_bias"].cpu().numpy() + pf
  prog = ImplicitFastProgram(m, batch_size=b, external_derivative=True)
  auto = implicit_derivative_reference(m, qpf, qvf)
  got = prog.run_device(dyn["mass_matrix"],
                        torch.as_tensor(rhs.astype(np.float32), device="mps"),
                        torch.as_tensor(auto.astype(np.float32), device="mps"))
  want = implicitfast_oracle(m, dyn["mass_matrix"].cpu().numpy(), rhs,
                             force_velocity_derivative=auto)
  np.testing.assert_allclose(got["qacc"].cpu().numpy(), want["qacc"],
                             rtol=1e-4, atol=1e-5)
  assert int(got["status"].cpu().numpy()[0]) == 0
  assert float(np.max(np.abs(want["qacc"]))) > 1.0  # non-vacuous solve


@_needs_gpu()
def test_automatic_derivative_pinned_step_parity_gpu():
  import os
  assert os.getenv("MUJOCO_METAL_RUN_GPU") == "1"
  from mujoco_metal.implicit import ImplicitFastProgram, implicit_derivative_reference
  from mujoco_metal.model import load_model
  from mujoco_metal.passive import PassiveForceModel
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  import torch
  # Pinned oracle: CPU mj_step with the implicitfast integrator. NOTE:
  # pinned keeps the Euler forward qacc in d.qacc (mj_implicitSkip solves
  # into a stack buffer); the implicit solution only flows into qvel/qpos
  # via mj_advance. The true oracle for our solve is therefore the pinned
  # assembled matrix H = M - h*qDeriv: expected qacc = H^-1 f. A short
  # manual loop (native solve + pinned advance math) must also reproduce
  # the CPU trajectory.
  m = _hinge_model(damping="0.3", dampingpoly=[0.1, 0.02])
  m.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
  m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  h = float(m.opt.timestep)
  desc = load_model(m)
  passive = PassiveForceModel(m)
  qp = torch.as_tensor(np.array([[0.4]], dtype=np.float32), device="mps")
  qv = torch.as_tensor(np.array([[1.7]], dtype=np.float32), device="mps")
  smooth = MetalSmoothDynamics(desc, batch_size=1)
  dyn = smooth.run_device(qp, qv, None, None)
  prog = ImplicitFastProgram(m, batch_size=1, external_derivative=True)
  auto = implicit_derivative_reference(m, np.array([[0.4]]), np.array([[1.7]]))
  rhs = dyn["qfrc_bias"].cpu().numpy() + passive.force(np.array([[0.4]]), np.array([[1.7]]))
  got = prog.run_device(dyn["mass_matrix"],
                        torch.as_tensor(rhs.astype(np.float32), device="mps"),
                        torch.as_tensor(auto.astype(np.float32), device="mps"))
  d = mujoco.MjData(m)
  d.qpos[:] = [0.4]
  d.qvel[:] = [1.7]
  mujoco.mj_step(m, d)
  M = float(np.asarray(d.qM).reshape(1, 1)[0, 0])
  R = float(np.asarray(d.qDeriv)[0])
  f = float(np.asarray(d.qfrc_smooth)[0])
  want = f / (M - h * R)
  np.testing.assert_allclose(got["qacc"].cpu().numpy()[0], [want],
                             rtol=1e-5, atol=1e-6)
  # Trajectory parity: manual native-solve loop vs CPU steps.
  nq, nv = np.array([[0.4]]), np.array([[1.7]])
  d2 = mujoco.MjData(m)
  d2.qpos[:] = [0.4]
  d2.qvel[:] = [1.7]
  for _ in range(20):
    dyn = smooth.run_device(torch.as_tensor(nq.astype(np.float32), device="mps"),
                            torch.as_tensor(nv.astype(np.float32), device="mps"),
                            None, None)
    auto = implicit_derivative_reference(m, nq, nv)
    rhs = dyn["qfrc_bias"].cpu().numpy() + passive.force(nq, nv)
    sol = prog.run_device(dyn["mass_matrix"],
                          torch.as_tensor(rhs.astype(np.float32), device="mps"),
                          torch.as_tensor(auto.astype(np.float32), device="mps"))
    a = sol["qacc"].cpu().numpy()
    nv = nv + h * a  # pinned mj_advance Euler velocity update
    nq = nq + h * nv  # pinned position update with new velocity
    mujoco.mj_step(m, d2)
  np.testing.assert_allclose(nq[0], np.asarray(d2.qpos), rtol=1e-4, atol=1e-5)
  np.testing.assert_allclose(nv[0], np.asarray(d2.qvel), rtol=1e-4, atol=1e-5)
