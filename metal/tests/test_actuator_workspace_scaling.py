# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native actuator workspace parity beyond former fixed local-array caps."""
import os
import mujoco
import numpy as np
import pytest

gpu = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                         reason="native GPU opt-in")

def _many_actuators(count):
  actuators = ''.join(f'<general joint="j" dyntype="filterexact" dynprm=".05" '
                     f'actearly="true" gaintype="affine" gainprm="{1+.01*i} 0 .03" '
                     'biastype="affine" biasprm="0 0 -.04"/>' for i in range(count))
  return mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><body><joint name="j"/><geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator>{actuators}</actuator></mujoco>''')


def test_many_stateful_actuator_fixture_exceeds_prior_thread_array_cap():
  from mujoco_metal.stateful_actuation import ActuatorModel
  model = _many_actuators(65)
  assert ActuatorModel(model).nu == 65
  data = mujoco.MjData(model)
  data.act[:], data.ctrl[:] = .2, .4
  mujoco.mj_forward(model, data)
  assert np.count_nonzero(data.actuator_force) == 65
  assert data.qfrc_actuator[0] > 10


def test_actuator_workspace_overflow_rejected_before_shader_compilation():
  pytest.importorskip("torch")
  from mujoco_metal.stateful_actuation import MetalActuators
  with pytest.raises(ValueError, match="indexing capacity"):
    MetalActuators(_many_actuators(65), 2**30)


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("count", [40, 65])
def test_native_actuator_workspace_exceeds_prior_cap(count):
  import torch
  from mujoco_metal.stateful_actuation import MetalActuators
  model = _many_actuators(count)
  data = mujoco.MjData(model)
  data.qvel[:] = .13
  data.act[:], data.ctrl[:] = np.linspace(.1,.3,count), np.linspace(.2,.5,count)
  mujoco.mj_forward(model,data)
  tensor = lambda x: torch.as_tensor(np.asarray(x,dtype=np.float32)[None].copy(), device="mps")
  kin = {"length": tensor(data.actuator_length), "velocity": tensor(data.actuator_velocity),
         "moment": torch.ones((1,count,1), device="mps")}
  program = MetalActuators(model,1)
  result = program.run_forces(tensor(data.ctrl),tensor(data.act),kin)
  np.testing.assert_allclose(result["force"].cpu().numpy()[0],data.actuator_force,atol=1e-6,rtol=2e-6)
  np.testing.assert_allclose(result["qfrc"].cpu().numpy()[0],data.qfrc_actuator,atol=1e-5,rtol=2e-6)
  np.testing.assert_allclose(result["act_dot"].cpu().numpy()[0],data.act_dot,
                             atol=1e-6,rtol=2e-6)
  np.testing.assert_allclose(result["ctrl"].cpu().numpy()[0],data.ctrl,
                             atol=1e-7,rtol=2e-6)
  # Rerun with different values to catch stale per-world scratch beyond slot 32.
  force_ptr = result["force"].data_ptr()
  data.ctrl[:] = np.linspace(-.1,.2,count)
  data.act[:] = np.linspace(.25,.05,count)
  mujoco.mj_forward(model,data)
  result = program.run_forces(tensor(data.ctrl),tensor(data.act),kin)
  assert result["force"].data_ptr() == force_ptr
  np.testing.assert_allclose(result["force"].cpu().numpy()[0],data.actuator_force,
                             atol=1e-6,rtol=2e-6)
  np.testing.assert_allclose(result["act_dot"].cpu().numpy()[0],data.act_dot,
                             atol=1e-6,rtol=2e-6)
