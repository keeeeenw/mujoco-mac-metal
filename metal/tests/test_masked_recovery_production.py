# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Selected-world production recovery through the masked force accumulator.

These tests execute the real masked paths — ``_acceleration`` with a
recovery ``forward_mask`` and the split-stage chain after the autorecovery
``reset_mask`` boundary — with production buffers and production masks.
Inactive worlds carry seeded nonzero and signed-zero words; the masked
accumulator must preserve them bitwise while valid worlds match the pinned
oracle. Includes restore/replay and all five integrated profiles.
"""

import os

import numpy as np
import pytest

import mujoco

_NEEDS_GPU = pytest.mark.skipif(
    os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native recovery paths")
_GPU = pytest.mark.gpu

_PROFILES = ["integrated_euler_v1", "integrated_rk4_v1",
             "integrated_implicit_v1", "integrated_implicitfast_v1",
             "integrated_scalable_v1"]

_PROFILE_INTEGRATOR = {
    "integrated_euler_v1": "Euler",
    "integrated_rk4_v1": "RK4",
    "integrated_implicit_v1": "implicit",
    "integrated_implicitfast_v1": "implicitfast",
    "integrated_scalable_v1": "Euler",
}


def _model(profile="integrated_euler_v1"):
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 -9.81" timestep="0.001" integrator="{_PROFILE_INTEGRATOR[profile]}"><flag contact="disable"/></option>
    <worldbody><body pos="0 0 1">
      <joint name="j" type="slide" axis="1 0 0" stiffness="2" damping="0.3"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="j"/></actuator></mujoco>""")


def _tendon_model(stiffness=50.0, damping=2.0):
  return mujoco.MjModel.from_xml_string(f"""<mujoco>
    <option gravity="0 0 0" timestep="0.001"><flag contact="disable"/></option>
    <worldbody><body pos="0 0 1">
      <joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <tendon><fixed name="t" stiffness="{stiffness}" damping="{damping}">
      <joint joint="j" coef="1"/></fixed></tendon>
    <actuator><motor joint="j"/></actuator></mujoco>""")


def _spatial_tendon_model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep="0.001"><flag contact="disable"/></option>
    <worldbody>
      <body pos="0 0 0"><joint name="j" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1"/>
        <site name="s0" pos="0.3 0 0"/></body>
      <site name="anchor" pos="1 0 0"/>
    </worldbody>
    <tendon><spatial name="t" stiffness="30">
      <site site="s0"/><site site="anchor"/></spatial></tendon>
    <actuator><motor joint="j"/></actuator></mujoco>""")


def _fluid_model():
  return mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep="0.001" density="1.2" integrator="implicit"><flag contact="disable"/></option>
    <worldbody><body pos="0 0 1">
      <joint name="j" type="slide" axis="1 0 0"/>
      <geom type="ellipsoid" size=".3 .2 .2" mass="1"/></body></worldbody>
    </mujoco>""")


def _word(tensor, row=1, col=0):
  return int(np.frombuffer(
      tensor[row, col].detach().cpu().numpy().tobytes(), dtype=np.uint32)[0])


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("profile", _PROFILES)
def test_acceleration_forward_mask_preserves_inactive_and_matches_oracle(profile):
  """Masked `_acceleration` (the recovery body) keeps inactive words bitwise.

  Seeds the production `_rhs`/`_rhs_low` inactive world with nonzero words
  and -0, runs the full production force pipeline (gravity+spring+damping
  passive, applied override, motor) under a recovery mask, then checks the
  inactive words bitwise and the valid world against pinned MuJoCo.
  """
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model(profile)
  sim = MetalSimulation(model, 2, profile=profile)
  device = sim._state._device
  sim.reset(qpos=np.array([[0.5], [0.5]], dtype=np.float32))
  sim._prepare_control(torch.tensor([[0.7], [0.7]], device=device))
  applied = torch.tensor([[0.25], [9.5]], device=device)
  sim._rhs[1, 0] = 3.25
  sim._rhs_low.view(torch.int32)[1, 0] = -2147483648  # 0x80000000
  mask = torch.tensor([True, False], dtype=torch.bool, device=device)
  qpos = sim.state._qpos.detach().clone()
  qvel = sim.state._qvel.detach().clone()
  acc, status, _ = sim._acceleration(
      qpos, qvel, forward_mask=mask, applied_force_override=applied)
  assert status.cpu().numpy().tolist() == [0, 0]
  # Inactive production scratch words survive the masked pipeline bitwise.
  assert _word(sim._rhs) == 0x40500000
  assert _word(sim._rhs_low) == 0x80000000
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = [0.5]
  cpu.qvel[:] = [0]
  cpu.ctrl[:] = [0.7]
  cpu.qfrc_applied[:] = [0.25]
  mujoco.mj_forward(model, cpu)
  np.testing.assert_allclose(
      acc.cpu().numpy()[0], np.asarray(cpu.qacc), rtol=1e-5, atol=1e-7)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                     "integrated_scalable_v1"])
def test_tendon_force_flows_through_masked_accumulation(profile):
  """A sprung fixed tendon contributes nonzero force under a recovery mask.

  The tendon spring (stiffness 50, displaced 0.4) carries -20 passive force
  against the +0.5 motor force. The pinned ten_length/qfrc guards prove the
  term is live; the motor-only negative control proves the comparison is
  sensitive to it (its qacc differs by ~20).
  """
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _tendon_model()
  sim = MetalSimulation(model, 2, profile=profile)
  device = sim._state._device
  sim.reset(qpos=np.array([[0.4], [0.4]], dtype=np.float32),
            qvel=np.array([[0.1], [0.1]], dtype=np.float32))
  sim._prepare_control(torch.tensor([[0.5], [0.5]], device=device))
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = [0.4]
  cpu.qvel[:] = [0.1]
  cpu.ctrl[:] = [0.5]
  mujoco.mj_forward(model, cpu)
  assert abs(float(cpu.ten_length[0])) > 1e-6
  assert abs(float(cpu.qfrc_passive[0])) > 1.0
  sim._rhs[1, 0] = -1.75
  sim._rhs_low.view(torch.int32)[1, 0] = -2147483648
  mask = torch.tensor([True, False], dtype=torch.bool, device=device)
  acc, status, _ = sim._acceleration(
      sim.state._qpos.detach().clone(), sim.state._qvel.detach().clone(),
      forward_mask=mask)
  assert status.cpu().numpy().tolist() == [0, 0]
  assert _word(sim._rhs) == int(np.frombuffer(
      np.array([-1.75], dtype=np.float32).tobytes(), dtype=np.uint32)[0])
  assert _word(sim._rhs_low) == 0x80000000
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(
      acc.cpu().numpy()[0], np.asarray(cpu.qacc), rtol=1e-5, atol=1e-7)
  # Negative control: the same geometry without the tendon is motor-only;
  # its oracle differs by the tendon term, so a dropped term cannot pass.
  bare = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 0" timestep="0.001"><flag contact="disable"/></option>
    <worldbody><body pos="0 0 1">
      <joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="j"/></actuator></mujoco>""")
  plain = mujoco.MjData(bare)
  plain.qpos[:] = [0.4]
  plain.qvel[:] = [0.1]
  plain.ctrl[:] = [0.5]
  mujoco.mj_step(bare, plain)
  assert abs(float(cpu.qacc[0] - plain.qacc[0])) > 1.0
  sim_bare = MetalSimulation(bare, 1, profile=profile)
  sim_bare.reset(qpos=np.array([[0.4]], dtype=np.float32),
                 qvel=np.array([[0.1]], dtype=np.float32))
  sim_bare._prepare_control(torch.tensor([[0.5]], device=device))
  acc_bare, _, _ = sim_bare._acceleration(
      sim_bare.state._qpos.detach().clone(),
      sim_bare.state._qvel.detach().clone())
  np.testing.assert_allclose(
      acc_bare.cpu().numpy()[0], np.asarray(plain.qacc),
      rtol=1e-5, atol=1e-7)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("kind", ["spatial", "fluid"])
def test_spatial_tendon_and_fluid_flow_through_masked_accumulation(kind):
  """Spatial-tendon and fluid forces are nonzero and survive masking."""
  import torch
  from mujoco_metal.simulation import MetalSimulation
  if kind == "spatial":
    model = _spatial_tendon_model()
    profile, qp, qv = "integrated_euler_v1", [[0.2], [0.2]], [[0.0], [0.0]]
  else:
    model = _fluid_model()
    profile, qp, qv = "integrated_implicit_v1", [[0.0], [0.0]], [[2.0], [2.0]]
  sim = MetalSimulation(model, 2, profile=profile)
  device = sim._state._device
  sim.reset(qpos=np.array(qp, dtype=np.float32),
            qvel=np.array(qv, dtype=np.float32))
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = qp[0]
  cpu.qvel[:] = qv[0]
  mujoco.mj_forward(model, cpu)
  if kind == "spatial":
    assert abs(float(cpu.ten_length[0])) > 1e-6
    assert abs(float(cpu.qfrc_passive[0])) > 1e-6
  else:
    assert abs(float(cpu.qfrc_fluid[0])) > 1e-6
  sim._rhs[1, 0] = 2.5
  sim._rhs_low.view(torch.int32)[1, 0] = -2147483648
  mask = torch.tensor([True, False], dtype=torch.bool, device=device)
  acc, status, _ = sim._acceleration(
      sim.state._qpos.detach().clone(), sim.state._qvel.detach().clone(),
      forward_mask=mask)
  assert status.cpu().numpy().tolist() == [0, 0]
  assert _word(sim._rhs) == int(np.frombuffer(
      np.array([2.5], dtype=np.float32).tobytes(), dtype=np.uint32)[0])
  assert _word(sim._rhs_low) == 0x80000000
  mujoco.mj_step(model, cpu)
  np.testing.assert_allclose(
      acc.cpu().numpy()[0], np.asarray(cpu.qacc), rtol=1e-5, atol=1e-7)


@_GPU
@_NEEDS_GPU
@pytest.mark.parametrize("profile", ["integrated_euler_v1",
                                     "integrated_scalable_v1",
                                     "integrated_rk4_v1"])
def test_split_stage_masked_reset_recovery_matches_oracle_and_replays(profile):
  """In-flight record + autorecovery `reset_mask` runs the masked split path.

  World 1 is reset to defaults mid-flight through the real autorecovery
  boundary; the remaining split stages must solve only world 0 (matching the
  oracle), leave world 1 at reset defaults, and replay bit-identically after
  restore. The record's production mask also drives the accumulator
  directly with seeded -0 to prove bitwise preservation at that level.
  """
  import torch
  from mujoco_metal.simulation import MetalSimulation
  model = _model(profile)
  sim = MetalSimulation(model, 2, profile=profile)
  device = sim._state._device
  sim.reset(qpos=np.array([[0.5], [0.5]], dtype=np.float32))
  sim._prepare_control(torch.tensor([[0.7], [0.0]], device=device))
  rec = sim.prepare_forward_position()
  sim.prepare_forward_velocity(rec, qvel=sim.state._qvel)
  sim._state.reset_mask(torch.tensor([False, True], dtype=torch.bool,
                                     device=device))
  assert rec.row_valid.tolist() == [True, False]
  # Accumulator level with the production mask and seeded production words.
  sim._rhs[1, 0] = 2.5
  sim._rhs_low.view(torch.int32)[1, 0] = -2147483648
  sim._rhs_accumulate(sim._rhs, sim._rhs_low,
                      torch.ones_like(sim._rhs), mask=rec.row_valid)
  assert _word(sim._rhs) == int(np.frombuffer(
      np.array([2.5], dtype=np.float32).tobytes(), dtype=np.uint32)[0])
  assert _word(sim._rhs_low) == 0x80000000
  # Pipeline level: remaining split stages solve the valid world only.
  sim.prepare_forward_actuation(rec)
  acc = sim.prepare_forward_acceleration(rec)
  # Status 3 marks the reset world unavailable by design (invalid-row
  # contract); the valid world solves normally.
  assert acc["status"].cpu().numpy().tolist() == [0, 3]
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = [0.5]
  cpu.qvel[:] = [0]
  cpu.ctrl[:] = [0.7]
  mujoco.mj_forward(model, cpu)
  np.testing.assert_allclose(
      acc["qacc_smooth"].cpu().numpy()[0], np.asarray(cpu.qacc),
      rtol=1e-5, atol=1e-7)
  # World 1 sits at reset defaults; world 0 kept its pre-reset state.
  qpos = sim.state._qpos.cpu().numpy()
  assert qpos[0, 0] == pytest.approx(0.5)
  assert qpos[1, 0] == pytest.approx(float(model.qpos0[0]))
  snap = sim.snapshot()
  first = acc["qacc_smooth"].cpu().numpy().copy()
  sim.restore(snap)
  rec2 = sim.prepare_forward_position()
  sim.prepare_forward_velocity(rec2, qvel=sim.state._qvel)
  sim.prepare_forward_actuation(rec2)
  acc2 = sim.prepare_forward_acceleration(rec2)
  np.testing.assert_array_equal(
      acc2["qacc_smooth"].cpu().numpy(), first)
