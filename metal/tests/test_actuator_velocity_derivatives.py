# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Analytical native actuator derivatives against pinned engine qDeriv.

The oracle runs the CPU engine's implicit integrator at the same forward
state. Its analytical derivative intentionally differs from finite
differences at clamped controls and for the DC LuGre approximation.
"""
import os

import mujoco
import numpy as np
import pytest

gpu = pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                         reason="native GPU opt-in")


def _model(family, *, clamp=False, disabled=False):
  attrs = {
      "affine": 'gaintype="affine" gainprm="2 .1 .3" biastype="affine" biasprm=".2 -.1 -.4"',
      "filterearly": 'dyntype="filterexact" dynprm=".05" actearly="true" gaintype="affine" gainprm="2 .1 .3" biastype="affine" biasprm=".2 -.1 -.4" actlimited="true" actrange="0 .6"',
      "muscle": 'dyntype="muscle" dynprm=".01 .04 .01" gaintype="muscle" gainprm=".5 1.5 100 100 .5 1.5 1 1 1.2" biastype="muscle" biasprm=".5 1.5 100 100 .5 1.5 1 1 1.2" lengthrange=".5 1.5"',
      "dcstateless": 'dyntype="dcmotor" dynprm="0" actearly="true" gaintype="dcmotor" gainprm="1 .4 0 0 .8 0 .2 0 1" biastype="dcmotor"',
      "dccurrent": 'dyntype="dcmotor" dynprm=".01 0 0 0 0 2 .1 0 1" actdim="2" actearly="true" gaintype="dcmotor" gainprm="1 .4 0 0 .8 0 .2 0 1" biastype="dcmotor" biasprm="0 0 0 .2 .3 .4"',
  }[family]
  limits = 'forcelimited="true" forcerange="-.01 .01"' if clamp else ''
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option integrator="implicitfast" timestep=".002" gravity="0 0 0"><flag contact="disable"/></option>
    <worldbody><body><joint name="a" axis="0 0 1"/><geom type="sphere" size=".1" mass="1"/>
      <body pos="1 0 0"><joint name="b" axis="0 1 0"/><geom type="sphere" size=".1" mass="1"/></body></body></worldbody>
    <tendon><fixed name="t"><joint joint="a" coef="2"/><joint joint="b" coef="-.7"/></fixed></tendon>
    <actuator><motor joint="a" gear=".1"/><general tendon="t" group="1" {attrs} {limits}/></actuator>
    </mujoco>''')
  if disabled:
    model.opt.disableactuator = 1 << 1
  return model


def _stages(model, velocity):
  data = mujoco.MjData(model)
  data.qpos[:] = [.3, .1]
  data.qvel[:] = [velocity, -.23]
  data.ctrl[:] = [.2, .7]
  data.act[:] = np.linspace(.03, .3, model.na)
  mujoco.mj_forward(model, data)
  moment = np.zeros((model.nu, model.nv))
  for a in range(model.nu):
    start, count = int(data.moment_rowadr[a]), int(data.moment_rownnz[a])
    moment[a, data.moment_colind[start:start+count]] = data.actuator_moment[start:start+count]
  stage = dict(ctrl=data.ctrl.copy(), act=data.act.copy(),
               length=data.actuator_length.copy(), velocity=data.actuator_velocity.copy(),
               moment=moment, force=data.actuator_force.copy(), qfrc=data.qfrc_actuator.copy())
  mujoco.mj_step(model, data)
  deriv = np.zeros((model.nv, model.nv))
  for row in range(model.nv):
    start, count = int(model.D_rowadr[row]), int(model.D_rownnz[row])
    deriv[row, model.D_colind[start:start+count]] = data.qDeriv[start:start+count]
  return stage, deriv


@pytest.mark.parametrize("family", ["affine", "filterearly", "muscle", "dcstateless", "dccurrent"])
def test_pinned_derivative_fixture_has_nonzero_mixed_moments(family):
  model = _model(family)
  stage, derivative = _stages(model, -.2 if family == "muscle" else .17)
  assert np.all(stage["moment"][1] != 0)
  assert np.linalg.norm(stage["force"]) > .01
  assert abs(derivative[0, 1]) > .01
  assert np.all(np.isfinite(derivative))


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("family,velocity,clamp,disabled", [
    ("affine", .17, False, False), ("affine", .17, True, False),
    ("affine", .17, False, True), ("filterearly", .17, False, False),
    ("muscle", -.65, False, False), ("muscle", -.2, False, False),
    ("muscle", 0, False, False), ("muscle", .5, False, False),
    ("dcstateless", .17, False, False), ("dccurrent", .17, False, False),
])
def test_native_actuator_velocity_derivative_matches_pinned_engine(family, velocity, clamp, disabled):
  import torch
  from mujoco_metal.stateful_actuation import MetalActuators
  model = _model(family, clamp=clamp, disabled=disabled)
  stages = [_stages(model, velocity), _stages(model, velocity + .03)]
  tensor = lambda key: torch.as_tensor(np.stack([s[0][key] for s in stages]).astype(np.float32), device="mps")
  kin = {key: tensor(key) for key in ("length", "velocity", "moment")}
  program = MetalActuators(model, 2)
  result = program.run_forces(tensor("ctrl"), tensor("act"), kin)
  np.testing.assert_allclose(result["force"].cpu().numpy(),
                             np.stack([s[0]["force"] for s in stages]), rtol=3e-5, atol=3e-6)
  got = program.run_velocity_derivative(tensor("ctrl"), tensor("act"), kin)
  np.testing.assert_allclose(got.cpu().numpy(), np.stack([s[1] for s in stages]),
                             rtol=4e-5, atol=5e-6)
