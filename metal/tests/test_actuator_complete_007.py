# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""All DC optional-state/controller branches against independent pinned physics."""

import os
from itertools import product

import mujoco
import numpy as np
import pytest


def _dc_model(bits, mode):
  # Pinned slot order: slew, integral, temperature, bristle, current.
  enabled = [(bits >> index) & 1 for index in range(5)]
  slew, integral, temperature, bristle, current = enabled
  dyn = [.025*current, 0, .8*temperature, 2, 20,
         3*bristle, .04, 2*slew, .4, 0]
  gain = [2, .6, .004, 20, .9, .4*integral, .08, .75, mode, 0]
  bias = [.03, 2, .125, .12, .2, .3, 0, 0, 0, 0]
  text = lambda values: " ".join(str(x) for x in values)
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".0005" gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><body><joint name="j" axis="0 1 0"/>
      <geom size=".1" mass="3"/></body></worldbody>
    <actuator><general joint="j" gear="1.3" dyntype="dcmotor" actearly="true"
      actdim="{sum(enabled)}" gaintype="dcmotor" biastype="dcmotor"
      dynprm="{text(dyn)}" gainprm="{text(gain)}" biasprm="{text(bias)}"
      forcelimited="true" forcerange="-.28 .23"/>
      <general joint="j" gear="-.7" dyntype="filterexact" dynprm=".03"
      gainprm="1.1" biastype="affine" biasprm=".01 .02 -.04"/>
    </actuator></mujoco>''')


def _initial(model, bits):
  qpos = np.asarray([[.17], [-.21]], np.float32)
  qvel = np.asarray([[.12], [-.18]], np.float32)
  enabled = [(bits >> index) & 1 for index in range(5)]
  values = [.05, .45, .3, .005, .08]
  first = [value for value, include in zip(values, enabled) if include]
  second = [-value if index != 2 else .4
            for index, value in enumerate(values) if enabled[index]]
  act = np.asarray([first + [.13], second + [-.09]], np.float32)
  assert model.na == sum(enabled)+1
  return qpos, qvel, act


CASES = list(product(range(32), range(3)))


@pytest.mark.parametrize("bits,mode", CASES)
def test_pinned_dc_optional_slot_fixture_executes_nonzero_physics(bits, mode):
  from mujoco_metal.stateful_actuation import ActuatorModel, act_dot_reference
  model = _dc_model(bits, mode)
  meta = ActuatorModel(model)
  # The normal lowering quantizes shader constants to float32. This host-only
  # source-order comparison uses the original double constants instead.
  for name in ("dynprm", "gainprm", "biasprm"):
    setattr(meta, name, getattr(model, "actuator_"+name).copy())
  qpos, qvel, act = _initial(model, bits)
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:] = qpos[world], qvel[world], act[world]
    data.ctrl[:] = [.4 if world == 0 else -.35, -.2 if world == 0 else .3]
    mujoco.mj_forward(model, data)
    assert np.all(np.isfinite(data.act_dot))
    assert np.linalg.norm(data.actuator_force) > .01
    assert abs(data.qfrc_actuator[0]) > .001
    assert np.linalg.norm(data.act_dot) > .01
    np.testing.assert_allclose(act_dot_reference(meta, data.ctrl,
        data.act, data.actuator_length, data.actuator_velocity), data.act_dot,
        rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native DC matrix")
@pytest.mark.parametrize("bits,mode", CASES)
def test_all_dc_optional_slots_force_derivative_trajectory_and_replay(bits, mode, monkeypatch):
  import torch
  from mujoco_metal import MetalSimulation
  model = _dc_model(bits, mode)
  qpos, qvel, act = _initial(model, bits)
  ctrl = np.asarray([[.4, -.2], [-.35, .3]], np.float32)
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  sim.reset(qpos=qpos, qvel=qvel, act=act)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:] = qpos[world], qvel[world], act[world], ctrl[world]
    mujoco.mj_forward(model, data)
  with monkeypatch.context() as context:
    def forbidden(*_args, **_kwargs):
      raise AssertionError("native actuator path used CPU physics")
    for name in ("mj_forward", "mj_step", "mj_fwdPosition", "mj_fwdVelocity",
                 "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint"):
      context.setattr(mujoco, name, forbidden)
    dynamics = sim._smooth.run_device(sim.state._qpos, sim.state._qvel)
    kin = sim._actuators.run_kinematics(sim.state._qpos, sim.state._qvel, dynamics["poses"])
    result = sim._actuators.run_forces(torch.as_tensor(ctrl, device="mps"), sim.state._act, kin)
  for key, field in (("force", "actuator_force"), ("qfrc", "qfrc_actuator"), ("act_dot", "act_dot")):
    np.testing.assert_allclose(result[key].cpu().numpy(),
        np.stack([getattr(data, field) for data in refs]), rtol=3e-5, atol=3e-6)
  for step in range(8):
    inputs = ctrl if step < 4 else -.7*ctrl
    assert sim.step(ctrl=inputs).cpu().numpy().tolist() == [0, 0]
    for world, data in enumerate(refs):
      data.ctrl[:] = inputs[world]
      mujoco.mj_step(model, data)
    for field in ("qpos", "qvel", "qacc", "act"):
      np.testing.assert_allclose(getattr(sim.state, field).cpu().numpy(),
          np.stack([getattr(data, field) for data in refs]), rtol=3e-5, atol=3e-6)
  checkpoint = sim.snapshot()
  sim.step(3, ctrl=ctrl)
  expected = sim.state.snapshot()
  sim.restore(checkpoint)
  sim.step(3, ctrl=ctrl)
  for field in ("qpos", "qvel", "qacc", "act", "time", "precision_words"):
    assert getattr(sim.state.snapshot(), field).tobytes() == getattr(expected, field).tobytes(), field
