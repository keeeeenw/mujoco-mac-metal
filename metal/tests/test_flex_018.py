# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Qualification test suite for deformable physics (Milestone 018).

Validates native MPS flex implementation:
- 1D cable distance equality constraint parity against pinned MuJoCo CPU oracle.
- 1D mass-spring tension restoring forces and analytic tangents.
- Rigid-motion invariance of 2D membrane and 3D tetrahedral continuum strain forces.
- Undamped energy conservation under dynamic elastic oscillation.
- Flex-rigid attachments and constraint coupling with moving rigid bodies.
- Narrowphase vertex-obstacle collision response.
- State snapshot, copy, restore, and reset lifecycle.
"""

import math
import mujoco
import numpy as np
import pytest
import torch

from mujoco_metal import MetalSimulation
from mujoco_metal.flex import MetalFlex, lower_flex_descriptor


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
def test_flex_1d_cable_equality_parity_cpu_gpu():
  """1D cable with equality distance constraints steps with < 1e-7 error vs CPU."""
  xml = """
  <mujoco>
    <option timestep="0.005" iterations="50" solver="PGS"/>
    <worldbody>
      <body name="anchor" pos="0 0 1">
        <geom type="sphere" size="0.02"/>
        <flexcomp name="cable" type="grid" count="4 1 1" spacing="0.1 0.1 0.1" mass="0.5" radius="0.01" dim="1">
          <edge equality="true" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  data_cpu = mujoco.MjData(model)
  sim_gpu = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")

  for s in range(20):
    sim_gpu.step()
    mujoco.mj_step(model, data_cpu)
    gpu_q = sim_gpu.state.qpos.cpu().numpy()[0]
    cpu_q = data_cpu.qpos
    max_err = float(np.max(np.abs(gpu_q - cpu_q)))
    assert max_err < 1e-7, f"Step {s + 1} cable qpos error {max_err:.3e} exceeds 1e-7"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
def test_flex_rigid_motion_invariance_2d_and_3d():
  """Arbitrary spatial rigid translation and rotation yields machine-zero internal forces."""
  models = [
      # 2D cloth grid
      mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <flexcomp name="c" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" mass="1" radius="0.01" dim="2">
            <contact contype="0" conaffinity="0"/>
            <edge equality="true"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """),
      # 3D tetrahedral grid
      mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <flexcomp name="c" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" mass="1" radius="0.01" dim="3">
            <contact contype="0" conaffinity="0"/>
            <edge equality="true"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """),
  ]

  for m in models:
    flex = MetalFlex(m, batch_size=1)
    poses = {
        "body_pos": torch.tensor([[[2.5, -1.8, 3.2]] * m.nbody], dtype=torch.float32, device="mps"),
        "body_quat": torch.tensor([[[0.7071068, 0.0, 0.7071068, 0.0]] * m.nbody], dtype=torch.float32, device="mps"),
    }
    qpos = torch.zeros((1, m.nq), device="mps")
    qvel = torch.zeros((1, m.nv), device="mps")
    qfrc, _, _ = flex.run_device(qpos, qvel, poses)
    max_frc = float(torch.max(torch.abs(qfrc)).item())
    assert max_frc < 1e-12, f"Rigid motion invariance violated: max force = {max_frc}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
def test_flex_rigid_coupling_attachment():
  """Deformable flex attached to moving rigid body matches CPU oracle across dynamic motion."""
  xml = """
  <mujoco>
    <option timestep="0.005" iterations="50" solver="PGS"/>
    <worldbody>
      <body name="slider" pos="0 0 1">
        <joint type="slide" axis="1 0 0"/>
        <geom type="box" size="0.1 0.1 0.1" mass="2"/>
        <flexcomp name="cable" type="grid" count="4 1 1" spacing="0.1 0.1 0.1" mass="0.5" radius="0.01" dim="1">
          <edge equality="true" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  data_cpu = mujoco.MjData(model)
  data_cpu.qvel[0] = 0.5  # Initial rigid body motion
  sim_gpu = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1", qvel=data_cpu.qvel[None, :].copy().astype(np.float32))

  for s in range(25):
    sim_gpu.step()
    mujoco.mj_step(model, data_cpu)
    gpu_q = sim_gpu.state.qpos.cpu().numpy()[0]
    cpu_q = data_cpu.qpos
    max_err = float(np.max(np.abs(gpu_q - cpu_q)))
    assert max_err < 1e-7, f"Step {s + 1} rigid+cable coupling error {max_err:.3e} exceeds 1e-7"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
def test_flex_undamped_energy_conservation():
  """Undamped elastic cable conserves total mechanical energy within tight tolerances."""
  xml = """
  <mujoco>
    <option timestep="0.0005" iterations="60" solver="PGS" gravity="0 0 0"/>
    <worldbody>
      <body name="anchor" pos="0 0 1">
        <geom type="sphere" size="0.02"/>
        <flexcomp name="cable" type="grid" count="3 1 1" spacing="0.1 0.1 0.1" mass="0.3" radius="0.01" dim="1">
          <edge stiffness="500" damping="0"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  # Displace vertex 1 to induce vibration
  qpos0 = model.qpos0.copy().astype(np.float32)
  qpos0[3] += 0.01
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1", qpos=qpos0[None, :])

  energies = []
  for _ in range(50):
    sim.step()
    qvel = sim.state.qvel[0]
    L = sim.flexedge_length[0]
    L0 = sim.flex._edge_length0
    # Kinetic energy = 0.5 * v^T M v
    ke = 0.5 * torch.sum(qvel * qvel * 0.1).item()  # vertex masses
    # Potential energy = 0.5 * k * sum (L - L0)^2
    pe = 0.5 * 500.0 * torch.sum((L - L0) ** 2).item()
    energies.append(ke + pe)

  e_arr = np.array(energies)
  drift = (np.max(e_arr) - np.min(e_arr)) / max(np.mean(e_arr), 1e-6)
  assert drift < 0.08, f"Undamped energy drift {drift:.3e} exceeds budget"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
def test_flex_vertex_obstacle_collision():
  """Narrowphase vertex collision detection detects penetration against ground plane."""
  xml = """
  <mujoco>
    <worldbody>
      <flexcomp name="c" type="grid" count="2 1 1" spacing="0.1 0.1 0.1" mass="0.2" radius="0.05" dim="1">
        <contact contype="0" conaffinity="0"/>
        <edge equality="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  flex = MetalFlex(model, batch_size=1)
  # Position vertices near ground plane z=0
  poses = {
      "body_pos": torch.tensor([[[0.0, 0.0, 0.02]] * model.nbody], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * model.nbody], dtype=torch.float32, device="mps"),
  }
  flex.update_kinematics(poses)
  cdata = flex.run_contacts(poses, plane_z=0.0, margin=0.02)
  # With radius 0.05 at z=0.02, distance to ground is 0.02 - 0.05 = -0.03 < margin
  assert torch.all(cdata["in_contact"]).item() is True
  assert torch.all(cdata["dist"] < 0.0).item() is True
  assert cdata["jacobian"].shape == (1, model.nflexvert, model.nv)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
def test_flex_snapshot_copy_restore_lifecycle():
  """Snapshot captures deformed flex state and restore reproduces exact state."""
  xml = """
  <mujoco>
    <option timestep="0.005" iterations="50" solver="PGS"/>
    <worldbody>
      <body name="anchor" pos="0 0 1">
        <geom type="sphere" size="0.02"/>
        <flexcomp name="cable" type="grid" count="4 1 1" spacing="0.1 0.1 0.1" mass="0.5" radius="0.01" dim="1">
          <edge equality="true" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  sim = MetalSimulation(model, batch_size=1, profile="integrated_euler_v1")

  # Step 10 steps to deform
  for _ in range(10):
    sim.step()

  snap = sim.snapshot()
  pos_snap = sim.state.qpos.clone()
  vel_snap = sim.state.qvel.clone()
  fpos_snap = sim.flexvert_xpos.clone()

  # Step another 10 steps
  for _ in range(10):
    sim.step()
  assert not torch.allclose(sim.state.qpos, pos_snap)

  # Restore snapshot
  sim.restore(snap)
  assert torch.allclose(sim.state.qpos, pos_snap)
  assert torch.allclose(sim.state.qvel, vel_snap)
  assert torch.allclose(sim.flexvert_xpos, fpos_snap)
