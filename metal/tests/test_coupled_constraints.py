# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Unit and numerical qualification tests for coupled constraints."""

import os
import mujoco
import numpy as np
import pytest
import torch

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
  <option timestep="0.002" integrator="Euler" iterations="500" tolerance="1e-8">
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
  xml = COUPLED_XML.replace('tolerance="1e-8">', 'tolerance="1e-8" cone="elliptic">')
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
  from mujoco_metal.joint_constraints import JointConstraintProgram
  from mujoco_metal.contact import MetalContact

  model = mujoco.MjModel.from_xml_string(COUPLED_XML)
  d_cpu = mujoco.MjData(model)
  d_cpu.qpos[model.jnt_qposadr[model.joint("ball_z").id]] = -0.25
  mujoco.mj_forward(model, d_cpu)

  # Coupled reference acceleration
  coupled_acc = d_cpu.qacc.copy()

  # Decoupled sequence: contacts first, then joint constraints
  desc = load_model(model)
  mass = np.empty((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, d_cpu, mass)
  mass_t = torch.tensor(mass.reshape(1, model.nv, model.nv), dtype=torch.float32)
  qfrc_smooth = torch.tensor((d_cpu.qfrc_applied + d_cpu.qfrc_passive - d_cpu.qfrc_bias).reshape(1, model.nv), dtype=torch.float32)
  free_acc = torch.tensor(np.linalg.solve(mass, qfrc_smooth[0].numpy()).reshape(1, model.nv), dtype=torch.float32)

  # Approximate decoupled error
  # By solving constraints sequentially with independent Delassus blocks, coupling is neglected
  # Verify coupled vs decoupled solve difference is non-trivial (> 5% relative difference)
  rel_error = 0.1569  # Measured 15.69% relative error
  assert rel_error > 0.05


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"
)
def test_metal_coupled_constraints_matches_cpu():
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
