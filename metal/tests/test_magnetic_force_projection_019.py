# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Physical COM force projection, independent of generalized DOF ordering."""

import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_metal.extensions import CustomMagneticForcePlugin, default_registry


XML = '''<mujoco><compiler angle="radian"/>
  <option timestep=".001" gravity="0 0 0"><flag contact="disable"/></option>
  <worldbody>
    <body name="swing" pos="-.7 0 .4"><joint type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .4 .1 0" size=".04" mass="1"/>
      <body name="tip" pos=".4 .1 0"><joint type="hinge" axis="0 1 0"/>
        <geom type="box" pos=".12 0 .03" size=".12 .05 .04" mass=".5"/>
      </body>
    </body>
    <body name="slider" pos=".5 0 .5"><joint type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".1" mass="1"/>
    </body>
    <body name="free" pos="1.2 .2 .7" quat=".9238795 .3826834 0 0">
      <freejoint/><geom type="box" pos=".05 .03 -.01" size=".1 .08 .06" mass="2"/>
    </body>
  </worldbody>
</mujoco>'''


def _inputs(model):
  qp = np.tile(model.qpos0, (2, 1)).astype(np.float32)
  qp[:, :3] = [[.4, -.3, .05], [-.2, .25, -.03]]
  qv = np.asarray([[.6, -.2, .3, .4, -.3, .2, -.4, .5, .7],
                   [-.3, .7, -.4, -.2, .6, -.1, .3, -.5, .2]], dtype=np.float32)
  assert qv.shape == (2, model.nv)
  return qp, qv


def _oracle(model, qp, qv, bodies, charge, field):
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qp, qv
  mujoco.mj_forward(model, data)
  output = np.zeros(model.nv)
  for body in bodies:
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                            body, velocity, 0)
    force = charge * np.cross(velocity[3:], field)
    mujoco.mj_applyFT(model, data, force, np.zeros(3), data.xipos[body], body, output)
  return output, data


def _cpu_stage(data, torch):
  def tensor(array):
    return torch.tensor(np.asarray(array)[None].copy(), dtype=torch.float32)
  inertial_quat = np.empty((data.model.nbody, 4))
  for body in range(data.model.nbody):
    mujoco.mju_mat2Quat(inertial_quat[body], data.ximat[body])
  return {"poses": {"inertial_pos": tensor(data.xipos),
                    "inertial_quat": tensor(inertial_quat)},
          "cdof": tensor(data.cdof), "cvel": tensor(data.cvel),
          "root_com": tensor(data.subtree_com)}


@pytest.mark.parametrize("body", [None, "swing", "tip", "slider", "free", 0])
def test_magnetic_force_cpu_tensor_matches_com_wrench_oracle(body):
  torch = pytest.importorskip("torch")
  model = mujoco.MjModel.from_xml_string(XML)
  qp, qv = _inputs(model)
  plugin = CustomMagneticForcePlugin(charge=-1.7, b_field=(.3, -.2, .8), body=body)
  plugin.init(model, 1, "cpu")
  for world in range(2):
    expected, data = _oracle(model, qp[world], qv[world], plugin._body_ids,
                             plugin.charge, plugin.b_field_init)
    stage = _cpu_stage(data, torch)
    before = {key: value.clone() for key, value in stage.items() if key != "poses"}
    actual = plugin.run_device(SimpleNamespace(), dynamics=stage)
    np.testing.assert_allclose(actual.numpy()[0], expected, rtol=2e-5, atol=2e-6)
    # A magnetic force is perpendicular to COM velocity and does no work.
    assert abs(np.dot(qv[world], actual.numpy()[0])) < 3e-6
    if body in (None, "tip", "free"):
      assert np.linalg.norm(expected) > .01
    else:
      # For one DOF, the skew-symmetric Lorentz projection is identically
      # zero: a perpendicular Cartesian force cannot drive that same DOF.
      np.testing.assert_allclose(expected, 0, rtol=0, atol=1e-12)
    for key in before:
      torch.testing.assert_close(stage[key], before[key], rtol=0, atol=0)
    assert torch.count_nonzero(plugin.run_device(
        SimpleNamespace(), dynamics=stage, compute_mask=torch.tensor([False]))) == 0


@pytest.mark.parametrize("kwargs", [
    {"charge": float("nan")}, {"charge": 1e100}, {"b_field": (0, 1)},
    {"b_field": (0, float("inf"), 0)}, {"body": True}, {"body": []},
])
def test_magnetic_force_invalid_configuration_is_rejected(kwargs):
  with pytest.raises((ValueError, TypeError)):
    CustomMagneticForcePlugin(**kwargs)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_magnetic_force_has_physical_projection_and_batch_isolation():
  import torch
  from mujoco_metal.native_api import mj_fwdPosition, mj_fwdVelocity
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(XML)
  qp, qv = _inputs(model)
  sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv,
                        profile="integrated_euler_v1")
  record = mj_fwdPosition(sim, return_record=True)
  stage = mj_fwdVelocity(sim, record=record)
  plugin = CustomMagneticForcePlugin(charge=1.7, b_field=(.3, -.2, .8))
  plugin.init(model, 2, sim.state._device)
  actual = plugin.run_device(sim.state, dynamics=stage).cpu().numpy()
  for world in range(2):
    expected, _ = _oracle(model, qp[world], qv[world], plugin._body_ids,
                          plugin.charge, plugin.b_field_init)
    np.testing.assert_allclose(actual[world], expected, rtol=3e-5, atol=3e-6)
    assert np.linalg.norm(expected) > .1
    assert abs(np.dot(qv[world], actual[world])) < 5e-6
  # The recovery callback must use the guarded per-world Metal kernel. Its
  # inactive output row is retained exactly, while the selected row matches
  # the independent pinned mj_applyFT oracle.
  plugin._queries._plugin_force[1].fill_(1234.5)
  masked = plugin.run_device(
      sim.state, dynamics=stage,
      compute_mask=torch.tensor([True, False], dtype=torch.bool,
                                device="mps"))
  np.testing.assert_allclose(masked[0].cpu().numpy(),
                             _oracle(model, qp[0], qv[0], plugin._body_ids,
                                     plugin.charge, plugin.b_field_init)[0],
                             rtol=3e-5, atol=3e-6)
  np.testing.assert_array_equal(masked[1].cpu().numpy(),
                                np.full(model.nv, 1234.5, dtype=np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_registered_magnetic_force_trajectory_and_checkpoint_replay():
  from mujoco_metal.simulation import MetalSimulation
  model = mujoco.MjModel.from_xml_string(XML)
  qp, qv = _inputs(model)
  config = CustomMagneticForcePlugin(name="com_projection_gate", charge=.7,
                                    b_field=(.3, -.2, .8), body="tip")
  default_registry.register(config)
  try:
    sim = MetalSimulation(model, batch_size=2, qpos=qp, qvel=qv,
                          profile="integrated_euler_v1")
    refs = [mujoco.MjData(model) for _ in range(2)]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
    for world, data in enumerate(refs):
      data.qpos[:], data.qvel[:] = qp[world], qv[world]
    for _ in range(20):
      for data in refs:
        force, _ = _oracle(model, data.qpos, data.qvel, (body,),
                            config.charge, config.b_field_init)
        data.qfrc_applied[:] = force
        mujoco.mj_step(model, data)
      np.testing.assert_array_equal(sim.step().cpu().numpy(), [0, 0])
      for world, data in enumerate(refs):
        np.testing.assert_allclose(sim.state.qpos.cpu().numpy()[world], data.qpos,
                                   rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(sim.state.qvel.cpu().numpy()[world], data.qvel,
                                   rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(sim.state.qacc.cpu().numpy()[world], data.qacc,
                                   rtol=3e-4, atol=5e-5)
    checkpoint = sim.snapshot()
    sim.step(steps=3)
    first = sim.state.snapshot()
    sim.restore(checkpoint)
    sim.step(steps=3)
    replay = sim.state.snapshot()
    for key in ("qpos", "qvel", "qacc", "time", "status"):
      np.testing.assert_array_equal(getattr(first, key), getattr(replay, key))
  finally:
    default_registry.unregister(config.name)
