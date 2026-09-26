# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Independent CPU mj_step trajectories for free-body implicitfast semantics."""

import os
import mujoco
import numpy as np
import pytest

from mujoco_metal.simulation import MetalSimulation

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in native GPU"
    ),
]


def _model(offset, armature, child=False):
  child_xml = (
      """<body pos=".3 .1 0"><joint type="hinge" axis="1 1 0" damping=".02"/>
      <geom type="box" size=".08 .06 .12" mass=".4"/></body>"""
      if child
      else ""
  )
  return mujoco.MjModel.from_xml_string(f"""<mujoco><compiler angle="radian"/>
    <option timestep=".003" integrator="implicitfast" gravity=".3 -.2 -2">
    <flag contact="disable"/></option><worldbody>
    <body pos=".2 -.1 .8" quat=".8 .1 .3 -.2">
      <joint type="free" damping=".08" armature="{armature}"/>
      <inertial pos="{offset}" quat=".7 -.2 .3 .1" mass="1.2" diaginertia=".07 .11 .13"/>
      <geom type="box" size=".12 .2 .25"/>{child_xml}
    </body></worldbody></mujoco>""")


@pytest.mark.parametrize(
    "offset,armature,child,disable_gravity",
    [
        ("0 0 0", 0, False, False),
        (".12 -.08 .04", 0, False, False),
        ("0 0 0", 0.02, False, True),
        (".12 -.08 .04", 0.02, False, False),
        (".12 -.08 .04", 0.02, True, False),
    ],
)
def test_free_implicitfast_matches_cpu_trajectory_and_qacc(
    offset, armature, child, disable_gravity
):
  model = _model(offset, armature, child)
  if disable_gravity:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  qpos = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  qvel = np.tile(np.linspace(-0.4, 0.7, model.nv), (2, 1)).astype(np.float32)
  qvel[:, 3:6] = [[1.6, 2.3, -0.7], [-2.1, 0.6, 1.2]]
  force = np.tile(np.linspace(-0.1, 0.2, model.nv), (2, 1)).astype(np.float32)
  force[1] *= -0.7
  native = MetalSimulation(
      model, 2, qpos, qvel, profile="contact_free_implicitfast_v1"
  )
  refs = [mujoco.MjData(model) for _ in range(2)]
  for row, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[row], qvel[row]
    data.qfrc_applied[:] = force[row]
  for step in range(300):
    native.step(qfrc_applied=force)
    for data in refs:
      mujoco.mj_step(model, data)
  state = native.state.snapshot()
  assert not np.any(state.status)
  for row, data in enumerate(refs):
    np.testing.assert_allclose(state.qpos[row], data.qpos, atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(state.qvel[row], data.qvel, atol=1e-4, rtol=2e-4)
    np.testing.assert_allclose(state.qacc[row], data.qacc, atol=3e-4, rtol=1e-3)
  native.step(5, qfrc_applied=force)
  expected = native.state.snapshot()
  native.state.restore(state)
  native.step(5, qfrc_applied=force)
  actual = native.state.snapshot()
  np.testing.assert_array_equal(actual.qpos, expected.qpos)
  np.testing.assert_array_equal(actual.qvel, expected.qvel)
  np.testing.assert_array_equal(actual.qacc, expected.qacc)


def test_mixed_free_trees_and_failed_world_reset():
  import torch

  model = mujoco.MjModel.from_xml_string(
      """<mujoco><option timestep=".002" integrator="implicitfast" gravity="0 0 -.5"><flag contact="disable"/></option>
  <worldbody>
  <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .2 .15" mass="1"/></body>
  <body pos="1 0 1"><freejoint/><geom type="box" size=".1 .2 .15" mass="1"/>
    <body pos=".3 0 .1"><joint type="hinge" damping=".1"/><geom type="sphere" size=".1" mass=".3"/></body></body>
  <body pos="2 0 1"><freejoint/><inertial pos=".05 .1 -.07" mass="1" diaginertia=".02 .03 .04" quat=".8 .2 -.1 .3"/><geom type="box" size=".1 .2 .15"/></body>
  </worldbody></mujoco>"""
  )
  rng = np.random.default_rng(904)
  q = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  v = rng.normal(0, 0.4, (2, model.nv)).astype(np.float32)
  sim = MetalSimulation(model, 2, q, v, profile="contact_free_implicitfast_v1")
  assert sim._midpoint.descriptor.nfree == 2
  refs = [mujoco.MjData(model) for _ in range(2)]
  for row, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = q[row], v[row]
  sim.step(200)
  for data in refs:
    for _ in range(200):
      mujoco.mj_step(model, data)
  prior = sim.state.snapshot()
  assert not np.any(prior.status)
  for row, data in enumerate(refs):
    np.testing.assert_allclose(prior.qpos[row], data.qpos, atol=5e-5, rtol=1e-4)
    np.testing.assert_allclose(prior.qvel[row], data.qvel, atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(prior.qacc[row], data.qacc, atol=3e-4, rtol=1e-3)
  force = torch.zeros((2, model.nv), device="mps", dtype=torch.float32)
  force[1, 0] = float("nan")
  sim.step(qfrc_applied=force)
  failed = sim.state.snapshot()
  assert failed.status[0] == 0 and failed.status[1] != 0
  for name in ("qpos", "qvel", "qacc", "time"):
    np.testing.assert_array_equal(
        getattr(failed, name)[1], getattr(prior, name)[1]
    )
  sim.step(2)
  sticky = sim.state.snapshot()
  assert sticky.status[1] == failed.status[1]
  np.testing.assert_array_equal(sticky.qpos[1], prior.qpos[1])
  sim.state.reset(env_ids=[1], qpos=q[1:2], qvel=v[1:2])
  sim.step()
  reset = sim.state.snapshot()
  assert reset.status.tolist() == [0, 0]


def test_midpoint_stage_no_eligible_bodies_rejects_nonfinite_and_overflow():
  import torch
  from mujoco_metal.implicit_midpoint import FreeBodyMidpointProgram

  model = mujoco.MjModel.from_xml_string(
      """<mujoco><option integrator="implicitfast" timestep="2"><flag contact="disable"/></option>
  <worldbody><body><joint type="ball"/><geom type="box" size=".1 .2 .3"/></body></worldbody></mujoco>"""
  )
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  program = FreeBodyMidpointProgram(model, 3)
  assert program.descriptor.nfree == 0
  qvel = torch.zeros((3, model.nv), dtype=torch.float32, device="mps")
  effective = torch.zeros_like(qvel)
  effective[2, 0] = 3e38
  force = torch.zeros_like(qvel)
  force[1, 0] = float("nan")
  quat = torch.tensor(
      np.tile(data.xquat, (3, 1, 1)), dtype=torch.float32, device="mps"
  )
  out = program.run_device(qvel, effective, torch.zeros_like(qvel), force, quat)
  assert out["status"].cpu().tolist() == [0, 21, 21]
  np.testing.assert_array_equal(
      out["qvel_next"].cpu().numpy(), np.zeros((3, model.nv))
  )
