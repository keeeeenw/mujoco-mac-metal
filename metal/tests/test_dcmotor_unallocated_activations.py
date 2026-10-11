# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Zero allocated DC dynamics mixed with real neighboring activation states.

Pinned mj_fwdActuation skips actadr < 0 before inspecting dyntype. A generic
DC dynamics tag alone does not allocate states in the compiled model. All
twenty non-DC-gain / bias combinations below execute actual CPU forward and
native integration; stateful neighboring actuators catch negative addressing.
"""
import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.stateful_actuation import ActuatorModel


def _model(integrator="Euler"):
  gains = {"fixed": "2", "affine": "2 .1 .2",
           "muscle": ".5 1.5 10 10 .5 1.5 1 1 1.2", "user": "0"}
  biases = {"none": "0", "affine": ".1 .2 -.3",
            "muscle": ".5 1.5 10 10 .5 1.5 1 1 1.2", "user": "0",
            "dcmotor": ".125 3 .25"}
  actuators = [f'<general name="{gain}_{bias}" joint="j" dyntype="dcmotor" '
               'dynprm=".01 0 0 0 0 0 0 0 1" actearly="true" '
               f'gaintype="{gain}" gainprm="{gp}" '
               f'biastype="{bias}" biasprm="{bp}" lengthrange=".5 1.5"/>'
               for gain, gp in gains.items() for bias, bp in biases.items()]
  actuators += ['<general joint="j" dyntype="filterexact" dynprm=".05" gainprm=".25"/>',
                '<general joint="j" dyntype="dcmotor" dynprm=".01 0 0 0 0 0 0 0 1" '
                'actearly="true" actdim="2" gaintype="dcmotor" '
                'gainprm="1 1 0 0 0 1 0 0 1" biastype="dcmotor"/>']
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option integrator="{integrator}" timestep=".00025" gravity="0 0 0"/>'
      '<worldbody><body><joint name="j" axis="0 1 0"/>'
      '<geom type="sphere" size=".2" mass="10"/></body></worldbody>'
      f'<actuator>{"".join(actuators)}</actuator></mujoco>')


def test_unallocated_dc_dynamics_execute_cpu_forward_before_admission():
  model = _model()
  assert model.actuator_actadr[:20].tolist() == [-1] * 20
  assert model.actuator_actnum[:20].tolist() == [0] * 20
  assert model.na == 3
  data = mujoco.MjData(model)
  data.qpos[:] = .625
  data.qvel[:] = -.03125
  data.ctrl[:] = .0625
  data.act[:] = [.125, .0625, .03125]
  mujoco.mj_forward(model, data)
  assert np.all(np.isfinite(data.qacc))
  assert np.all(np.abs(data.actuator_force[:20]) > 1e-4)
  assert ActuatorModel(model).na == 3


def test_dc_gain_and_positive_activation_shapes_remain_validated():
  model = _model()
  model.actuator_actadr[0] = 0
  with pytest.raises(mujoco.FatalError, match="inconsistent state array dimension"):
    mujoco.mj_forward(model, mujoco.MjData(model))
  with pytest.raises(ValueError, match="DC-motor slots"):
    ActuatorModel(model)
  model = _model()
  model.actuator_actnum[-1] = 1
  with pytest.raises(ValueError, match="DC-motor slots"):
    ActuatorModel(model)
  model = _model()
  model.actuator_gaintype[0] = mujoco.mjtGain.mjGAIN_DCMOTOR
  with pytest.raises(mujoco.FatalError, match="inconsistent state array dimension"):
    mujoco.mj_forward(model, mujoco.MjData(model))
  with pytest.raises(ValueError, match="DC-motor gain slots"):
    ActuatorModel(model)
  model = _model()
  model.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_DCMOTOR
  model.actuator_dynprm[0, 5] = 1.
  with pytest.raises(ValueError, match="bristle state is unallocated"):
    ActuatorModel(model)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native GPU opt-in")
@pytest.mark.parametrize("integrator,profile", [
    ("Euler", "integrated_euler_v1"), ("RK4", "integrated_rk4_v1"),
    ("implicit", "integrated_implicit_v1"),
    ("implicitfast", "integrated_implicitfast_v1")])
def test_unallocated_dc_dynamics_mixed_forces_integration_and_replay(integrator, profile, monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  model = _model(integrator)
  qpos = np.array([[.625], [.5625]], dtype=np.float32)
  qvel = np.array([[-.03125], [.0625]], dtype=np.float32)
  act = np.array([[.125, .0625, .03125], [.0625, .03125, .015625]], dtype=np.float32)
  ctrl = np.full((2, model.nu), .0625, dtype=np.float32)
  ctrl[1] *= .5
  sim = MetalSimulation(model, 2, profile=profile)
  sim.reset(qpos=qpos, qvel=qvel, act=act)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for row, data in enumerate(refs):
    data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:] = qpos[row], qvel[row], act[row], ctrl[row]
    mujoco.mj_forward(model, data)
  with monkeypatch.context() as context:
    def forbidden(*args, **kwargs):
      raise AssertionError("actuator evaluation fell back to CPU physics")
    for name in ("mj_forward", "mj_step", "mj_fwdPosition", "mj_fwdVelocity",
                 "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint"):
      context.setattr(mujoco, name, forbidden)
    dynamics = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)
    kin = sim._actuators.run_kinematics(sim.state._qpos, sim.state._qvel,
                                       dynamics["poses"])
    result = sim._actuators.run_forces(torch.as_tensor(ctrl, device="mps"),
                                      sim.state._act, kin)
  # Checking every force prevents cancellation in the aggregated trajectory
  # from hiding a wrong gain/bias family or a neighboring state overread.
  for key, field in (("force", "actuator_force"), ("qfrc", "qfrc_actuator"),
                     ("act_dot", "act_dot")):
    np.testing.assert_allclose(result[key].cpu().numpy(),
                               np.stack([getattr(data, field) for data in refs]),
                               rtol=4e-5, atol=5e-6)
  for _ in range(8):
    assert sim.step(ctrl=ctrl).cpu().numpy().tolist() == [0, 0]
    for data in refs:
      mujoco.mj_step(model, data)
    for name in ("qpos", "qvel", "qacc", "act"):
      got = getattr(sim.state, name).cpu().numpy()
      expected = np.stack([getattr(data, name) for data in refs])
      np.testing.assert_allclose(got, expected, rtol=4e-5, atol=5e-6)
  assert np.all(np.abs(sim.state.qpos.cpu().numpy() - qpos) > 1e-6)
  snapshot = sim.snapshot()
  sim.step(3, ctrl=ctrl)
  expected = sim.state.snapshot()
  sim.restore(snapshot)
  sim.step(3, ctrl=ctrl)
  got = sim.state.snapshot()
  for name in ("qpos", "qvel", "qacc", "act", "time", "precision_words"):
    assert getattr(got, name).tobytes() == getattr(expected, name).tobytes(), name
