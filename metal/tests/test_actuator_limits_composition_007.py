# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Antagonistic forces, tendon/joint limits and gravity routing under contact."""

import os
from itertools import product

import mujoco
import numpy as np
import pytest


def _model(routing, clamp_disabled, group_disabled):
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".0005" gravity="0 0 -9.81" iterations="120"/>
    <worldbody><geom type="plane" size="2 2 .1"/>
      <body gravcomp=".7" pos="0 0 .5">
        <joint name="h" axis="0 1 0" actuatorgravcomp="{str(routing).lower()}"
          actuatorfrclimited="true" actuatorfrcrange="-.7 .9"/>
        <geom size=".05" pos=".2 0 0" mass="2"/></body>
      <body pos=".5 0 .085"><joint name="s" type="slide" axis="0 0 1"
        actuatorfrclimited="true" actuatorfrcrange="-.8 .8"/>
        <geom size=".1" mass=".4"/></body></worldbody>
    <tendon><fixed name="t" actuatorfrclimited="true" actuatorfrcrange="-.4 .5">
      <joint joint="h" coef=".4"/><joint joint="s" coef="1.2"/>
    </fixed></tendon>
    <actuator><general joint="h" gear="2" ctrllimited="true" ctrlrange="-.4 .4"
      forcelimited="true" forcerange="-.9 .7"/>
      <general joint="h" gear="-1.5" gainprm="2"/>
      <general tendon="t" gear="1" gainprm=".8"/>
      <general tendon="t" gear="-.5" gainprm="1.3" group="1"/>
    </actuator></mujoco>''')
  if clamp_disabled:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
  if group_disabled:
    model.opt.disableactuator = 1 << 1
  return model


CASES = list(product((False, True), repeat=3))


@pytest.mark.parametrize("routing,clamp_disabled,group_disabled", CASES)
def test_source_force_limits_and_gravity_routing_have_active_independent_oracles(
    routing, clamp_disabled, group_disabled):
  model = _model(routing, clamp_disabled, group_disabled)
  for sign in (-1, 1):
    data = mujoco.MjData(model)
    data.qpos[:] = [.17, 0]
    data.qvel[:] = [.08, -.03]
    data.ctrl[:] = sign*np.asarray([3, -2, 4, 3])
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    assert np.linalg.norm(data.qfrc_constraint) > 1
    assert np.linalg.norm(data.qfrc_gravcomp) > 1
    assert np.all(data.qfrc_actuator >= model.jnt_actfrcrange[:, 0]-1e-12)
    assert np.all(data.qfrc_actuator <= model.jnt_actfrcrange[:, 1]+1e-12)
    tendon_total = data.actuator_force[2:].sum()
    assert -.4-1e-12 <= tendon_total <= .5+1e-12
    assert np.any(np.abs(data.actuator_force[:2]) > .1)
    if group_disabled:
      assert data.actuator_force[3] == 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native actuator limits")
@pytest.mark.parametrize("routing,clamp_disabled,group_disabled", CASES)
def test_native_aggregate_limits_gravity_routing_contact_and_sign_changing_trajectory(
    routing, clamp_disabled, group_disabled, monkeypatch):
  from mujoco_metal import MetalSimulation, native_api as api
  from mujoco_metal.forward_stages import ForwardStage
  model = _model(routing, clamp_disabled, group_disabled)
  sim = MetalSimulation(model, 2, profile="integrated_euler_v1")
  qpos = np.asarray([[.17, 0], [-.14, .002]], np.float32)
  qvel = np.asarray([[.08, -.03], [-.09, .04]], np.float32)
  sim.reset(qpos=qpos, qvel=qvel)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
  for step in range(12):
    sign = 1 if step < 6 else -1
    ctrl = sign*np.asarray([[3, -2, 4, 3], [-3, 2, -4, -3]], np.float32)
    for world, data in enumerate(refs):
      data.ctrl[:] = ctrl[world]
      mujoco.mj_forward(model, data)
    # Publish controls through the public state API, then compare a native
    # forward record before integration to the independent CPU stage.
    api.mj_setState(sim, {"ctrl": ctrl})
    with monkeypatch.context() as context:
      def forbidden(*_args, **_kwargs):
        raise AssertionError("native actuation fell back to CPU physics")
      for name in ("mj_forward", "mj_step", "mj_fwdPosition", "mj_fwdVelocity",
                   "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint"):
        context.setattr(mujoco, name, forbidden)
      result = api.mj_forward(sim, skipsensor=True)
    record = result["record"]
    actual = record.values[ForwardStage.ACT]["qfrc_actuator"].cpu().numpy()
    np.testing.assert_allclose(actual, np.stack([data.qfrc_actuator for data in refs]),
                               rtol=3e-5, atol=3e-6)
    forces = sim._actuators._ws["force"].cpu().numpy().reshape(2, model.nu)
    np.testing.assert_allclose(forces, np.stack([data.actuator_force for data in refs]),
                               rtol=3e-5, atol=3e-6)
    assert sim.step(ctrl=ctrl).cpu().numpy().tolist() == [0, 0]
    for data in refs:
      mujoco.mj_step(model, data)
    for name in ("qpos", "qvel", "qacc"):
      np.testing.assert_allclose(getattr(sim.state, name).cpu().numpy(),
          np.stack([getattr(data, name) for data in refs]), rtol=3e-5, atol=5e-6)
