# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Unit and numerical qualification tests for coupled constraints."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import (
    CoupledConstraintDescriptor,
    coupled_constraint_oracle,
    lower_coupled_constraints,
    MetalCoupledConstraints,
)
from mujoco_metal.model import load_model
from mujoco_metal.smooth_metal import MetalSmoothDynamics

COUPLED_XML = """<mujoco model="coupled_test">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler" iterations="1000" tolerance="1e-6">
    <flag contact="enable" equality="enable" limit="enable" frictionloss="enable"/>
  </option>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <body name="lever1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" range="-0.5 0.5" limited="true" frictionloss="0.1" damping="0.05"/>
      <geom name="g_lever1" type="sphere" size="0.2" pos="0.5 0 0" mass="1"/>
    </body>
    <body name="lever2" pos="0 1 1">
      <joint name="j2" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="g_lever2" type="sphere" size="0.2" pos="0.5 0 0" mass="1"/>
    </body>
    <body name="ball" pos="0.5 0 1.5">
      <joint name="ball_z" type="slide" axis="0 0 1"/>
      <geom name="g_ball" type="sphere" size="0.1" mass="0.5"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j2" joint2="j1" polycoef="0 1.5 0 0 0"/>
  </equality>
  <actuator>
    <motor joint="j1" ctrlrange="-10 10"/>
  </actuator>
  <sensor>
    <jointpos joint="j1"/>
    <jointvel joint="j1"/>
  </sensor>
</mujoco>"""


def test_coupled_lowering_and_immutability():
  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  desc = lower_coupled_constraints(model)
  assert isinstance(desc, CoupledConstraintDescriptor)
  assert desc.nv == 3
  assert desc.neq == 1
  assert desc.nc >= 1
  assert desc.nr <= 96
  assert desc.qpos0.flags.writeable is False
  assert desc.joint_sol_params.flags.writeable is False
  with pytest.raises(ValueError):
    desc.qpos0[0] = 1.0


def test_coupled_lowering_unsupported_geoms_rejected():
  xml = COUPLED_XML.replace('type="sphere" size="0.2"', 'type="box" size="0.1 0.1 0.1"')
  m = mujoco.MjModel.from_xml_string(xml)
  with pytest.raises(ValueError, match="only sphere-plane and sphere-sphere"):
    lower_coupled_constraints(m)


def test_coupled_lowering_unsupported_condim_rejected():
  xml = COUPLED_XML.replace('<compiler angle="radian"/>', '<compiler angle="radian"/><default><geom condim="4"/></default>')
  m = mujoco.MjModel.from_xml_string(xml)
  with pytest.raises(ValueError, match="condim"):
    lower_coupled_constraints(m)


def test_coupled_lowering_elliptic_cone_rejected_for_condim3():
  xml = COUPLED_XML.replace('tolerance="1e-6">', 'tolerance="1e-6" cone="elliptic">')
  m = mujoco.MjModel.from_xml_string(xml)
  with pytest.raises(ValueError, match="pyramidal"):
    lower_coupled_constraints(m)


def test_coupled_oracle_matches_mujoco():
  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  qpos = np.array([[0.0, 0.0, -0.25], [0.1, 0.15, -0.2]], dtype=np.float64)
  qvel = np.array([[0.1, -0.1, 0.05], [-0.05, 0.1, -0.1]], dtype=np.float64)
  res = coupled_constraint_oracle(model, qpos, qvel)
  assert np.all(res["status"] == 0)
  for b in range(2):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos[b]
    d.qvel[:] = qvel[b]
    mujoco.mj_forward(model, d)
    np.testing.assert_allclose(res["qacc"][b], d.qacc, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res["qfrc_constraint"][b], d.qfrc_constraint, rtol=1e-5, atol=1e-5)


def test_decoupled_vs_coupled_divergence():
  """Demonstrates that solving decoupled produces substantial error vs coupled solve."""
  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  d_cpu = mujoco.MjData(model)
  d_cpu.qpos[model.jnt_qposadr[model.joint("ball_z").id]] = -0.25
  mujoco.mj_forward(model, d_cpu)

  # Coupled reference acceleration
  coupled_acc = d_cpu.qacc.copy()

  # Decoupled sequence:
  # Step 1: Solve contacts independently (disable equality, limits, frictionloss)
  m_contact = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m_contact.opt.disableflags |= (
      mujoco.mjtDisableBit.mjDSBL_EQUALITY
      | mujoco.mjtDisableBit.mjDSBL_LIMIT
      | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
  )
  d_contact = mujoco.MjData(m_contact)
  d_contact.qpos[:] = d_cpu.qpos
  d_contact.qvel[:] = d_cpu.qvel
  mujoco.mj_forward(m_contact, d_contact)
  f_contact = d_contact.qfrc_constraint.copy()

  # Step 2: Joint constraints solve with f_contact injected as external force
  m_joint = mujoco.MjModel.from_xml_string(COUPLED_XML)
  m_joint.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  d_joint = mujoco.MjData(m_joint)
  d_joint.qpos[:] = d_cpu.qpos
  d_joint.qvel[:] = d_cpu.qvel
  d_joint.qfrc_applied[:] = f_contact
  mujoco.mj_forward(m_joint, d_joint)
  decoupled_acc = d_joint.qacc.copy()

  # Verify Delassus cross-coupling block norm is substantial
  M = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, d_cpu, M)
  Minv = np.linalg.inv(M)
  J = d_cpu.efc_J.reshape(d_cpu.nefc, model.nv)
  c_mask = (d_cpu.efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL) | (
      d_cpu.efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
  )
  J_c = J[c_mask]
  J_j = J[~c_mask]
  W_cj = J_c @ Minv @ J_j.T
  cross_norm = float(np.linalg.norm(W_cj))
  assert cross_norm > 5.0, f"Expected strong Delassus cross-term coupling, got {cross_norm}"

  # Calculate relative acceleration difference
  rel_error = float(np.linalg.norm(decoupled_acc - coupled_acc) / np.linalg.norm(coupled_acc))
  assert rel_error > 0.10, f"Expected decoupled relative error > 10%, got {rel_error * 100:.2f}%"
  assert np.isclose(rel_error, 0.15688, atol=1e-3), f"Expected ~15.69% relative error, got {rel_error}"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_metal_coupled_constraints_matches_cpu():
  import torch

  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  batch = 2
  qpos = np.array([[0.0, 0.0, -0.25], [0.05, 0.075, -0.23]], dtype=np.float32)
  qvel = np.array([[0.0, 0.0, 0.0], [0.1, -0.05, 0.02]], dtype=np.float32)

  coupled_stage = MetalCoupledConstraints(model, batch_size=batch)
  desc = load_model(model)
  smooth = MetalSmoothDynamics(desc, batch_size=batch)

  qpos_dev = torch.tensor(qpos, device="mps")
  qvel_dev = torch.tensor(qvel, device="mps")

  dyn = smooth.run_device(qpos_dev, qvel_dev)
  res = coupled_stage.run_device(
      dyn["poses"], dyn["mass_matrix"], -dyn["qfrc_bias"], qpos_dev, qvel_dev
  )

  assert torch.all(res["status"] == 0)

  # Compare with CPU oracle
  oracle = coupled_constraint_oracle(model, qpos, qvel)
  np.testing.assert_allclose(res["qacc"].cpu().numpy(), oracle["qacc"], rtol=2e-3, atol=1e-2)
  np.testing.assert_allclose(res["qfrc_constraint"].cpu().numpy(), oracle["qfrc_constraint"], rtol=2e-3, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_metal_coupled_constraints_zero_contact():
  import torch

  xml = COUPLED_XML.replace('type="sphere" size="0.1"', 'type="sphere" size="0.01"').replace('pos="0.5 0 1.5"', 'pos="10 10 10"')
  model = mujoco.MjModel.from_xml_string(xml)
  coupled = MetalCoupledConstraints(model, batch_size=1)
  desc = load_model(model)
  smooth = MetalSmoothDynamics(desc, batch_size=1)
  qpos = torch.zeros((1, model.nq), device="mps")
  qvel = torch.zeros((1, model.nv), device="mps")
  dyn = smooth.run_device(qpos, qvel)
  res = coupled.run_device(dyn["poses"], dyn["mass_matrix"], -dyn["qfrc_bias"], qpos, qvel)
  assert res["status"].item() == 0


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_hot_loop_device_resident_invariants():
  """Verifies that running coupled constraints in a hot loop preserves device resident state."""
  import torch

  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  batch = 2
  qpos = torch.zeros((batch, model.nq), device="mps", dtype=torch.float32)
  qvel = torch.zeros((batch, model.nv), device="mps", dtype=torch.float32)

  stage = MetalCoupledConstraints(model, batch_size=batch)
  desc = load_model(model)
  smooth = MetalSmoothDynamics(desc, batch_size=batch)

  # Check preallocated default equality state on device
  assert hasattr(stage, "_eq_active_default")
  assert stage._eq_active_default.device.type == "mps"
  assert stage._eq_active_default.dtype == torch.int32
  assert stage._eq_active_default.shape == (batch, max(1, stage.descriptor.neq))

  dyn = smooth.run_device(qpos, qvel)

  for _ in range(5):
    res = stage.run_device(
        dyn["poses"], dyn["mass_matrix"], -dyn["qfrc_bias"], qpos, qvel
    )
    assert res["status"].device.type == "mps"
    assert res["qacc"].device.type == "mps"
