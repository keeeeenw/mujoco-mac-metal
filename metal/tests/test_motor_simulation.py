# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Opt-in end-to-end scalar motor stepping checks."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

pytestmark = pytest.mark.gpu
_requires_gpu = pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)

_XML = """<mujoco><option timestep='.001' gravity='0 0 -9.81'>
  <flag contact='disable'/></option><worldbody>
  <body><joint name='hinge' type='hinge' damping='.1'/>
    <geom type='capsule' size='.08 .2'/></body>
  <body pos='1 0 0'><joint name='slide' type='slide' damping='.03'/>
    <geom type='box' size='.1 .1 .1'/></body>
</worldbody><actuator>
  <general joint='hinge' gear='2' dyntype='none' gaintype='fixed'
    biastype='none' gainprm='2' ctrllimited='true' ctrlrange='-1 1'/>
  <general joint='slide' gear='-.75' dyntype='none' gaintype='fixed'
    biastype='none' gainprm='1.5' forcelimited='true' forcerange='-1 1'/>
</actuator></mujoco>"""


def _data(model, qpos, qvel):
  result = mujoco.MjData(model)
  result.qpos[:] = qpos
  result.qvel[:] = qvel
  return result


@_requires_gpu
def test_control_and_applied_force_match_mujoco_over_1000_steps():
  import torch

  model = mujoco.MjModel.from_xml_string(_XML)
  batch = 2
  qpos = np.tile(model.qpos0.astype(np.float32), (batch, 1))
  qvel = np.stack(
      [np.linspace(-0.2, 0.25, model.nv), np.linspace(0.3, -0.1, model.nv)]
  ).astype(np.float32)
  simulation = MetalSimulation(
      model,
      batch_size=batch,
      qpos=qpos,
      qvel=qvel,
      profile="contact_free_motor_euler_v1",
  )
  references = [_data(model, qpos[i], qvel[i]) for i in range(batch)]
  rng = np.random.default_rng(4701)
  for _ in range(1000):
    controls = rng.normal(0, 2, size=(batch, model.nu)).astype(np.float32)
    applied = rng.normal(0, 0.1, size=(batch, model.nv)).astype(np.float32)
    device_ctrl = torch.tensor(controls, device="mps")
    device_force = torch.tensor(applied, device="mps")
    ctrl_copy, force_copy = device_ctrl.clone(), device_force.clone()
    for i, data in enumerate(references):
      data.ctrl[:] = controls[i]
      data.qfrc_applied[:] = applied[i]
      mujoco.mj_step(model, data)
    status = simulation.step(ctrl=device_ctrl, qfrc_applied=device_force)
    assert torch.equal(status, torch.zeros_like(status))
    assert torch.equal(device_ctrl, ctrl_copy)
    assert torch.equal(device_force, force_copy)
  actual = simulation.state.snapshot()
  for i, data in enumerate(references):
    np.testing.assert_allclose(actual.qpos[i], data.qpos, rtol=3e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[i], data.qvel, rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(actual.qacc[i], data.qacc, rtol=3e-4, atol=5e-4)

  # A missing control resets to zero for the next call.
  for data in references:
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_step(model, data)
  simulation.step()
  actual = simulation.state.snapshot()
  for i, data in enumerate(references):
    np.testing.assert_allclose(actual.qpos[i], data.qpos, rtol=3e-5, atol=5e-6)
    np.testing.assert_allclose(actual.qvel[i], data.qvel, rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(actual.qacc[i], data.qacc, rtol=3e-4, atol=5e-4)


@_requires_gpu
def test_nonfinite_control_is_sticky_per_world():
  import torch

  model = mujoco.MjModel.from_xml_string(_XML)
  simulation = MetalSimulation(
      model, batch_size=2, profile="contact_free_motor_euler_v1"
  )
  controls = torch.zeros((2, model.nu), dtype=torch.float32, device="mps")
  controls[0, 0] = float("nan")
  before = simulation.state.snapshot()
  status = simulation.step(ctrl=controls).clone()
  assert status[0].item() != 0 and status[1].item() == 0
  failed = simulation.state.snapshot()
  np.testing.assert_array_equal(failed.qpos[0], before.qpos[0])
  np.testing.assert_array_equal(failed.qvel[0], before.qvel[0])
  controls.zero_()
  status = simulation.step(ctrl=controls).clone()
  assert status[0].item() != 0 and status[1].item() == 0
  recovered = simulation.state.snapshot()
  np.testing.assert_array_equal(recovered.qpos[0], failed.qpos[0])
  assert not np.array_equal(recovered.qpos[1], failed.qpos[1])


@_requires_gpu
def test_invalid_host_control_fails_before_generation_changes():
  model = mujoco.MjModel.from_xml_string(_XML)
  simulation = MetalSimulation(
      model, batch_size=2, profile="contact_free_motor_euler_v1"
  )
  for controls in (
      np.zeros((1, model.nu), dtype=np.float32),
      np.full((2, model.nu), np.inf, dtype=np.float64),
      np.full((2, model.nu), np.finfo(np.float64).max),
  ):
    with pytest.raises(ValueError, match="ctrl"):
      simulation.step(ctrl=controls)
    assert simulation.state.generation == 0
