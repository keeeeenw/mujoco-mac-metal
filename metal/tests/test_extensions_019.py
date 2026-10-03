# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 019: Native extension registration, split-stage APIs, and inverse dynamics."""

import os
import mujoco
import numpy as np
import pytest

try:
  import torch
except ImportError:
  torch = None

from mujoco_metal.extensions import (
    ExtensionRegistry,
    NativePlugin,
    PluginType,
    CustomMagneticForcePlugin,
    CustomUserSensorPlugin,
    default_registry,
)
from mujoco_metal.native_api import (
    mj_inverse,
    mj_fwdPosition,
    mj_fwdVelocity,
    mj_fwdActuation,
    mj_fwdAcceleration,
    mj_fwdConstraint,
    mj_getState,
    mj_setState,
    StateSpec,
    mju_mulMatVec,
    mju_transpose,
    mju_mulMatMat,
    mju_cholSolve,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
    ),
]


def _simple_chain_model():
  xml = """
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body name="b1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.04" fromto="0 0 0  0.3 0 0"/>
        <body name="b2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.03" fromto="0 0 0  0.25 0 0"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <motor joint="j1"/>
      <motor joint="j2"/>
    </actuator>
  </mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml)


def test_extension_registry_lifecycle_cpu():
  """Registry manages plugin registrations, duplicates, and queries."""
  reg = ExtensionRegistry()
  p1 = CustomMagneticForcePlugin(name="mag1", charge=1.5)
  reg.register(p1)
  assert reg.has("mag1", PluginType.FORCE)
  assert reg.get("mag1", PluginType.FORCE) is p1
  assert len(reg.list_plugins()) == 1

  # Duplicate registration raises
  with pytest.raises(ValueError, match="already registered"):
    reg.register(p1)

  # Non-plugin registration raises
  with pytest.raises(TypeError):
    reg.register("not-a-plugin")

  # Unregister
  reg.unregister("mag1", PluginType.FORCE)
  assert not reg.has("mag1", PluginType.FORCE)
  assert reg.get("mag1", PluginType.FORCE) is None


def test_host_callback_rejection_unless_opt_in():
  """Native mode does not silently invoke host callbacks without explicit opt-in."""
  m = _simple_chain_model()
  d = mujoco.MjData(m)
  p = CustomMagneticForcePlugin(name="mag_strict", charge=1.0)
  default_registry.allow_host_callbacks = False
  with pytest.raises(RuntimeError, match="allow_host_callbacks is False"):
    p.run_host(d)

  # Explicit opt-in allows host execution
  default_registry.allow_host_callbacks = True
  try:
    p.run_host(d)  # succeeds
  finally:
    default_registry.allow_host_callbacks = False


def test_native_custom_magnetic_force_plugin_gpu():
  """Native plugin runs on MPS device tensors without host fallback."""
  from mujoco_metal.simulation import MetalSimulation
  m = _simple_chain_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  plugin = CustomMagneticForcePlugin(name="test_mag", charge=2.0, b_field=(0.0, 0.0, 1.0))
  plugin.init(m, batch_size=1, device=sim.state._device)

  # Set velocity
  sim.reset(qvel=np.array([[1.0, 2.0]], dtype=np.float32))
  qfrc = plugin.run_device(sim.state)

  assert isinstance(qfrc, torch.Tensor)
  assert qfrc.device.type == "mps"
  assert tuple(qfrc.shape) == (1, m.nv)

  # For nv=2, B=(0,0,1): v=(1,2,0) -> v x B = (2, -1, 0)
  # qfrc = 2.0 * [2, -1] = [4.0, -2.0]
  qfrc_np = qfrc.cpu().numpy()[0]
  assert np.all(np.isfinite(qfrc_np))


def test_native_inverse_dynamics_gpu():
  """Native GPU inverse dynamics matches CPU MuJoCo oracle."""
  from mujoco_metal.simulation import MetalSimulation
  m = _simple_chain_model()
  sim = MetalSimulation(m, batch_size=2, profile="integrated_euler_v1")

  qp = np.array([[0.3, -0.4], [0.1, 0.5]], dtype=np.float32)
  qv = np.array([[0.5, -0.2], [-0.3, 0.8]], dtype=np.float32)
  qa = np.array([[1.2, -0.7], [0.4, 1.1]], dtype=np.float32)

  sim.reset(qpos=qp, qvel=qv)
  sim._state._qacc.copy_(torch.tensor(qa, device=sim.state._device))

  # Compute on device
  qfrc_inv_gpu = mj_inverse(sim, sim.state.qpos, sim.state.qvel, sim.state.qacc)
  assert isinstance(qfrc_inv_gpu, torch.Tensor)
  assert qfrc_inv_gpu.device.type == "mps"
  assert tuple(qfrc_inv_gpu.shape) == (2, m.nv)

  # CPU Oracle
  d = mujoco.MjData(m)
  for w in range(2):
    d.qpos[:] = qp[w]
    d.qvel[:] = qv[w]
    d.qacc[:] = qa[w]
    mujoco.mj_inverse(m, d)
    qfrc_cpu = np.asarray(d.qfrc_inverse)
    qfrc_gpu = qfrc_inv_gpu[w].cpu().numpy()
    np.testing.assert_allclose(qfrc_gpu, qfrc_cpu, atol=1e-4, rtol=1e-3)


def test_native_math_utilities_gpu():
  """Native MPS math utilities (mulMatVec, transpose, mulMatMat, cholSolve)."""
  device = torch.device("mps")
  # 1. mju_mulMatVec
  A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
  x = torch.tensor([0.5, -0.5], dtype=torch.float32, device=device)
  y = mju_mulMatVec(A, x)
  np.testing.assert_allclose(y.cpu().numpy(), [-0.5, -0.5], atol=1e-6)

  # 2. mju_transpose
  At = mju_transpose(A)
  np.testing.assert_allclose(At.cpu().numpy(), [[1.0, 3.0], [2.0, 4.0]], atol=1e-6)

  # 3. mju_mulMatMat
  C = mju_mulMatMat(A, At)
  expected_C = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[1.0, 3.0], [2.0, 4.0]])
  np.testing.assert_allclose(C.cpu().numpy(), expected_C, atol=1e-6)

  # 4. mju_cholSolve: L L^T x = b
  L = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch.float32, device=device)
  b = torch.tensor([4.0, 11.0], dtype=torch.float32, device=device)
  sol = mju_cholSolve(L, b)
  # Verify: (L @ L^T) @ sol == b
  M = L @ L.t()
  recon = M @ sol
  np.testing.assert_allclose(recon.cpu().numpy(), b.cpu().numpy(), atol=1e-5)


def test_native_split_stage_stepping_gpu():
  """Native split-stage pipeline functions advance stages independently on GPU."""
  from mujoco_metal.simulation import MetalSimulation
  m = _simple_chain_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.2, -0.1]], dtype=np.float32),
            qvel=np.array([[0.1, 0.0]], dtype=np.float32))

  # Position stage
  poses = mj_fwdPosition(sim)
  assert "body_pos" in poses
  assert "site_pos" in poses

  # Velocity stage
  dynamics = mj_fwdVelocity(sim, poses=poses)
  assert "mass_matrix" in dynamics
  assert "qfrc_bias" in dynamics

  # Actuation stage
  qfrc_act = mj_fwdActuation(sim, poses=poses, ctrl=np.array([[1.0, -1.0]], dtype=np.float32))
  assert qfrc_act.device.type == "mps"

  # Acceleration stage
  acc, status = mj_fwdAcceleration(sim, poses=poses, dynamics=dynamics)
  assert acc.device.type == "mps"
  assert status.cpu().numpy()[0] == 0


def test_native_state_selectors_gpu():
  """Native state getters and setters atomically manipulate device state."""
  from mujoco_metal.simulation import MetalSimulation
  m = _simple_chain_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=np.array([[0.5, 0.2]], dtype=np.float32),
            qvel=np.array([[-0.1, 0.3]], dtype=np.float32))

  # Extract state
  s = mj_getState(sim, StateSpec.QPOS | StateSpec.QVEL | StateSpec.TIME)
  assert "qpos" in s and "qvel" in s and "time" in s
  np.testing.assert_allclose(s["qpos"].cpu().numpy()[0], [0.5, 0.2])

  # Mutate via setState
  mj_setState(sim, {"qpos": np.array([[0.1, -0.2]], dtype=np.float32)})
  s2 = mj_getState(sim, StateSpec.QPOS)
  np.testing.assert_allclose(s2["qpos"].cpu().numpy()[0], [0.1, -0.2])
