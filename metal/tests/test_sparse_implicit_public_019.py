# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public sparse implicit integration against pinned B2 CPU trajectories."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.fluid import fluid_derivative_reference
from mujoco_metal.simulation import MetalSimulation
from mujoco_metal.stepping import validate_stepping_profile
from mujoco_metal.implicit_effective import select_gmres_krylov_dimension


_XML = """<mujoco>
  <option integrator="implicit" timestep=".001" gravity="0 0 0"
          density="850" viscosity=".12" solver="PGS" iterations="40">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <body name="root" pos="0 0 .4">
      <joint name="hinge0" type="hinge" axis="0 0 1" damping=".08"/>
      <geom type="ellipsoid" size=".12 .08 .06" mass=".8"/>
      <body name="child" pos=".25 0 0">
        <joint name="hinge1" type="hinge" axis="0 1 0" damping=".04"/>
        <geom type="capsule" size=".045 .12" mass=".35"/>
      </body>
    </body>
  </worldbody>
  <tendon><fixed name="damped_armature" damping=".16" armature=".025">
    <joint joint="hinge0" coef="1"/><joint joint="hinge1" coef="-.7"/>
  </fixed></tendon>
  <actuator><general name="velocity_bias" joint="hinge1"
      gaintype="fixed" gainprm="1" biastype="affine"
      biasprm="0 0 -.22"/></actuator>
</mujoco>"""

_GPU_ONLY = pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires serialized Apple GPU qualification slot")


def test_gmres_profile_preflight_uses_keyword_only_capacity_selector():
  """Exercise the same selector ABI used before Simulation allocates MPS."""
  dimension = select_gmres_krylov_dimension(
      batch=2, nv=12, edge_count=30, available_bytes=1 << 20,
      max_iterations=12, krylov_dimension=8)
  assert dimension == 8
  with pytest.raises(ValueError, match="GMRES workspace needs"):
    select_gmres_krylov_dimension(
        batch=2, nv=12, edge_count=30, available_bytes=1,
        max_iterations=12, krylov_dimension=8)


def _fixture(profile):
  model = mujoco.MjModel.from_xml_string(_XML)
  if profile == "integrated_implicitfast_v1":
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  else:
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
  return model


def _bias_velocity_derivative(model, qpos, qvel):
  """Small pinned finite-difference oracle for the held-position RNE term."""
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  base = data.qfrc_bias.copy()
  derivative = np.zeros((model.nv, model.nv), dtype=np.float64)
  eps = 1e-5
  for column in range(model.nv):
    data.qvel[:] = qvel
    data.qvel[column] += eps
    mujoco.mj_forward(model, data)
    derivative[:, column] = -(data.qfrc_bias - base) / eps
  return derivative


@pytest.mark.parametrize("profile", [
    "integrated_implicit_v1", "integrated_implicitfast_v1"])
def test_sparse_implicit_public_profile_has_nontrivial_pinned_b2_oracle(profile):
  """The composed fixture exercises passive, actuator, tendon, and fluid D."""
  model = _fixture(profile)
  validate_stepping_profile(model, profile=profile)
  qpos = np.asarray([[.12, -.21], [-.17, .24]], dtype=np.float32)
  qvel = np.asarray([[.7, -.45], [-.3, .62]], dtype=np.float32)
  ctrl = np.asarray([[.18], [-.11]], dtype=np.float32)
  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.ctrl[:] = ctrl[world]
    refs.append(data)
  fluid_D = fluid_derivative_reference(model, qpos, qvel)
  assert np.linalg.norm(fluid_D) > 1e-7
  bias_derivatives = []
  for data in refs:
    assert np.linalg.norm(data.qvel) > 0
    bias_derivatives.append(
        _bias_velocity_derivative(model, data.qpos.copy(), data.qvel.copy()))
  # Full implicit includes the RNE bias derivative; implicitfast does not.
  # Keep a nonzero pinned witness so the sparse full-implicit composition
  # cannot pass by silently omitting that source.
  if profile == "integrated_implicit_v1":
    assert max(np.linalg.norm(value) for value in bias_derivatives) > 1e-4
  for _ in range(8):
    for data in refs:
      mujoco.mj_step(model, data)
  oracle_qpos = np.stack([data.qpos.copy() for data in refs])
  oracle_qvel = np.stack([data.qvel.copy() for data in refs])
  oracle_qacc = np.stack([data.qacc.copy() for data in refs])
  assert np.all(np.isfinite(oracle_qacc))
  assert np.max(np.abs(oracle_qpos - qpos)) > 1e-6


@pytest.mark.gpu
@_GPU_ONLY
@pytest.mark.parametrize("profile", [
    "integrated_implicit_v1", "integrated_implicitfast_v1"])
def test_native_sparse_implicit_public_b2_matches_pinned_trajectory_and_replay(profile):
  """Run actual public component-sparse stepping, then checkpoint replay."""
  model = _fixture(profile)
  qpos = np.asarray([[.12, -.21], [-.17, .24]], dtype=np.float32)
  qvel = np.asarray([[.7, -.45], [-.3, .62]], dtype=np.float32)
  ctrl = np.asarray([[.18], [-.11]], dtype=np.float32)
  refs = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.ctrl[:] = ctrl[world]
    refs.append(data)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  assert sim._component_mass_enabled
  assert sim._smooth._mass_storage == "block_sparse"
  assert sim._component_solver is not None
  assert sim._effective_implicit is not None
  assert sim._smooth._workspace.get("mass") is None
  assert "mass_matrix" not in sim._smooth._workspace
  assert sim._implicit is None and sim._implicitfast is None
  assert sim._velocity_derivative_values.values.shape == (
      2, max(int(model.nD), 1))
  for _ in range(8):
    status = sim.step(ctrl=ctrl)
    for data in refs:
      mujoco.mj_step(model, data)
    assert status.detach().cpu().numpy().tolist() == [0, 0]
    snap_step = sim.state.snapshot()
    np.testing.assert_allclose(
        snap_step.qpos, np.stack([data.qpos for data in refs]),
        rtol=4e-4, atol=5e-6)
    np.testing.assert_allclose(
        snap_step.qvel, np.stack([data.qvel for data in refs]),
        rtol=4e-4, atol=2e-5)
    np.testing.assert_allclose(
        snap_step.qacc, np.stack([data.qacc for data in refs]),
        rtol=7e-4, atol=3e-5)
  snap = sim.state.snapshot()
  np.testing.assert_allclose(snap.qpos,
                             np.stack([data.qpos for data in refs]),
                             rtol=4e-4, atol=5e-6)
  np.testing.assert_allclose(snap.qvel,
                             np.stack([data.qvel for data in refs]),
                             rtol=4e-4, atol=2e-5)
  np.testing.assert_allclose(snap.qacc,
                             np.stack([data.qacc for data in refs]),
                             rtol=7e-4, atol=3e-5)
  checkpoint = sim.state.snapshot()
  for _ in range(4):
    sim.step(ctrl=ctrl)
  final = sim.state.snapshot()
  sim.state.restore(checkpoint)
  for _ in range(4):
    sim.step(ctrl=ctrl)
  replay = sim.state.snapshot()
  np.testing.assert_array_equal(replay.qpos, final.qpos)
  np.testing.assert_array_equal(replay.qvel, final.qvel)
  np.testing.assert_array_equal(replay.qacc, final.qacc)


@pytest.mark.gpu
@_GPU_ONLY
@pytest.mark.parametrize("profile", [
    "integrated_implicit_v1", "integrated_implicitfast_v1"])
def test_native_sparse_inverse_discrete_applies_compiled_velocity_operator(profile):
  """INVDISCRETE applies M-hD, then solves the original sparse M operator."""
  import torch

  model = _fixture(profile)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
  qpos = np.asarray([[.12, -.21], [-.17, .24]], dtype=np.float32)
  qvel = np.asarray([[.7, -.45], [-.3, .62]], dtype=np.float32)
  requested_qacc = np.asarray([[.9, -.35], [-.6, .42]], dtype=np.float32)
  expected = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    data.qacc[:] = requested_qacc[world]
    mujoco.mj_inverse(model, data)
    expected.append(data.qfrc_inverse.copy())
    np.testing.assert_array_equal(data.qacc, requested_qacc[world])

  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  assert sim._component_mass_enabled and sim._effective_implicit is not None
  qacc_tensor = torch.tensor(requested_qacc, dtype=torch.float32,
                             device=sim.state._device)
  result = sim.inverse_skip(qacc=qacc_tensor)
  assert result["status"].detach().cpu().numpy().tolist() == [0, 0]
  np.testing.assert_array_equal(qacc_tensor.cpu().numpy(), requested_qacc)
  np.testing.assert_allclose(result["qfrc_inverse"].cpu().numpy(),
                             np.asarray(expected, dtype=np.float32),
                             rtol=8e-4, atol=4e-5)
  # The coupled assembler borrows/rewrites this output backing. INVDISCRETE
  # must convert the owned input snapshot, and the public query transaction
  # must restore the caller's borrowed workspace contents on return.
  cc_output = sim._coupled_constraints._workspace["out_acc"]
  borrowed_qacc = cc_output[:2 * model.nv].reshape(2, model.nv)
  borrowed_qacc.copy_(torch.tensor(requested_qacc, dtype=torch.float32,
                                   device=sim.state._device))
  from mujoco_metal.native_api import mj_inverseSkip
  alias_result = mj_inverseSkip(sim, qacc=borrowed_qacc,
                                return_details=True)
  assert alias_result["status"].detach().cpu().numpy().tolist() == [0, 0]
  np.testing.assert_allclose(alias_result["qfrc_inverse"].cpu().numpy(),
                             np.asarray(expected, dtype=np.float32),
                             rtol=8e-4, atol=4e-5)
  np.testing.assert_array_equal(borrowed_qacc.cpu().numpy(), requested_qacc)
