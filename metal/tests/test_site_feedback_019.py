# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""CPU oracles and opt-in public trajectories for the site feedback plugin."""

import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_metal.extensions import PluginType, default_registry
from mujoco_metal.site_feedback import SiteFeedbackPlugin


def _model():
  return mujoco.MjModel.from_xml_path(
      str(Path(__file__).parents[1] / "examples" / "haptic_calligraphy.xml"))


def _dynamics(model, data, torch):
  def tensor(value):
    return torch.tensor(np.asarray(value).copy()[None], dtype=torch.float32,
                        device="cpu")
  poses = {name: tensor(value) for name, value in (
      ("body_pos", data.xpos), ("body_quat", data.xquat),
      ("inertial_pos", data.xipos), ("geom_pos", data.geom_xpos),
      ("site_pos", data.site_xpos), ("joint_anchor", data.xanchor),
      ("joint_axis", data.xaxis))}
  for prefix, matrices in (("site", data.site_xmat),):
    quats = np.empty((len(matrices), 4), dtype=np.float64)
    for i, matrix in enumerate(matrices):
      mujoco.mju_mat2Quat(quats[i], matrix)
    poses[prefix + "_quat"] = tensor(quats)
  return {
      "poses": poses,
      "cdof": tensor(data.cdof),
      "cdof_dot": tensor(data.cdof_dot),
      "cvel": tensor(data.cvel),
      "root_com": tensor(data.subtree_com),
  }


def _cpu_wrench(model, data, site_id):
  point = data.site_xpos[site_id].copy()
  velocity = np.empty(6, dtype=np.float64)
  mujoco.mj_objectVelocity(
      model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity, 0)
  phase = (np.array([.9, 0., 1.3]) * float(data.time)
           + np.array([0., 0., 1.1]))
  target = (np.array([1., 0., .9])
            + np.array([.06, 0., .055]) * np.sin(phase))
  target_velocity = (np.array([.06, 0., .055])
                     * np.array([.9, 0., 1.3]) * np.cos(phase))
  force = 1.8 * (target - point) + .65 * (target_velocity - velocity[3:])
  qfrc = np.zeros(model.nv, dtype=np.float64)
  mujoco.mj_applyFT(model, data, force, np.zeros(3), point,
                    int(model.site_bodyid[site_id]), qfrc)
  return point, velocity[3:], target, target_velocity, force, qfrc


def test_site_feedback_config_rejects_invalid_coefficients_and_vectors():
  with pytest.raises(ValueError, match="kp"):
    SiteFeedbackPlugin(kp=-1)
  with pytest.raises(ValueError, match="kd"):
    SiteFeedbackPlugin(kd=float("nan"))
  with pytest.raises(ValueError, match="center"):
    SiteFeedbackPlugin(center=(0, 1))
  with pytest.raises(ValueError, match="frequency"):
    SiteFeedbackPlugin(frequency=(1, -1, 2))


def test_site_feedback_cpu_torch_matches_independent_pinned_wrench_oracle(monkeypatch):
  torch = pytest.importorskip("torch")
  model = _model()
  data = mujoco.MjData(model)
  data.qpos[:] = (.31, -.55)
  data.qvel[:] = (.25, -.4)
  data.time = .37
  mujoco.mj_forward(model, data)
  dynamics = _dynamics(model, data, torch)
  plugin = SiteFeedbackPlugin(
      kp=1.8, kd=.65, center=(1., 0., .9), amplitude=(.06, 0., .055),
      frequency=(.9, 0., 1.3), phase=(0., 0., 1.1))
  plugin.init(model, batch_size=1, device="cpu")
  state = SimpleNamespace(_time=torch.tensor([.37], dtype=torch.float32))
  site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip"))
  expected = _cpu_wrench(model, data, site)

  def forbidden(*_args, **_kwargs):
    raise AssertionError("native plugin runtime entered a host physics/readback path")

  with monkeypatch.context() as guard:
    guard.setattr(mujoco, "mj_step", forbidden)
    guard.setattr(torch.Tensor, "cpu", forbidden)
    guard.setattr(torch.Tensor, "numpy", forbidden)
    result = plugin.evaluate(state, dynamics)
    output = plugin.run_device(state, dynamics=dynamics)
  np.testing.assert_allclose(result["site_position"].numpy()[0], expected[0],
                             atol=3e-6, rtol=3e-5)
  np.testing.assert_allclose(result["site_velocity"].numpy()[0], expected[1],
                             atol=4e-6, rtol=4e-5)
  np.testing.assert_allclose(result["target_position"].numpy()[0], expected[2],
                             atol=2e-7, rtol=2e-7)
  np.testing.assert_allclose(result["target_velocity"].numpy()[0], expected[3],
                             atol=2e-7, rtol=2e-7)
  np.testing.assert_allclose(result["force"].numpy()[0], expected[4],
                             atol=4e-6, rtol=4e-5)
  np.testing.assert_allclose(result["qfrc"].numpy()[0], expected[5],
                             atol=5e-6, rtol=5e-5)
  np.testing.assert_allclose(output.numpy()[0], expected[5], atol=5e-6, rtol=5e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native Metal trajectory")
def test_native_site_feedback_euler_trajectory_batch_reset_and_replay():
  import torch
  from mujoco_metal import MetalSimulation
  from mujoco_metal.site_feedback import register_site_feedback

  model = _model()
  qpos = np.array([[.2, -.4], [-.25, .48]], dtype=np.float32)
  qvel = np.array([[.05, -.03], [-.02, .06]], dtype=np.float32)
  references = [mujoco.MjData(model) for _ in range(2)]
  for i, data in enumerate(references):
    data.qpos[:] = qpos[i]
    data.qvel[:] = qvel[i]
    mujoco.mj_forward(model, data)
  register_site_feedback(
      "site_feedback_trajectory_test", site="pen_tip", kp=1.8, kd=.65,
      center=(1., 0., .9), amplitude=(.06, 0., .055),
      frequency=(.9, 0., 1.3), phase=(0., 0., 1.1))
  try:
    sim = MetalSimulation(
        model, batch_size=2, qpos=qpos, qvel=qvel,
        profile="integrated_euler_v1")
    site_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip"))
    # Verify the extension is routed through MuJoCo's FORCE/passive bucket,
    # not the ACT actuator bucket. This checks the real prepared simulation
    # stage composition before trajectory stepping, with an independent
    # mj_forward passive-force baseline plus the CPU wrench oracle.
    position_record = sim.prepare_forward_position()
    velocity_stage = sim.prepare_forward_velocity(position_record)
    actuation_stage = sim.prepare_forward_actuation(position_record)
    plugin = next(p for p in sim._force_plugins
                  if p.name == "site_feedback_trajectory_test")
    plugin_result = plugin.evaluate(
        sim.state, velocity_stage["dynamics"])
    for world, data in enumerate(references):
      expected_passive = (data.qfrc_passive
                          + _cpu_wrench(model, data, site_id)[5])
      np.testing.assert_allclose(
          velocity_stage["qfrc_passive"][world].detach().cpu().numpy(),
          expected_passive, atol=8e-6, rtol=8e-5,
          err_msg=f"FORCE plugin passive bucket, world {world}")
    np.testing.assert_allclose(
        plugin_result["qfrc"].detach().cpu().numpy(),
        np.stack([_cpu_wrench(model, data, site_id)[5] for data in references]),
        atol=7e-6, rtol=7e-5)
    np.testing.assert_array_equal(
        actuation_stage["qfrc_actuator"].detach().cpu().numpy(),
        np.zeros((2, model.nv), dtype=np.float32))
    max_qpos = max_qvel = max_qacc = max_sensor = 0.0
    checkpoint = None
    final_at_checkpoint = None
    references_at_checkpoint = None
    for step in range(120):
      for data in references:
        data.qfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = _cpu_wrench(model, data, site_id)[5]
        mujoco.mj_step(model, data)
      def forbidden_hotpath(*_args, **_kwargs):
        raise AssertionError("native haptic step entered host MuJoCo physics")

      with pytest.MonkeyPatch.context() as guard:
        guard.setattr(mujoco, "mj_step", forbidden_hotpath)
        guard.setattr(mujoco, "mj_forward", forbidden_hotpath)
        guard.setattr(torch.Tensor, "cpu", forbidden_hotpath)
        guard.setattr(torch.Tensor, "numpy", forbidden_hotpath)
        status = sim.step()
      assert np.all(status.detach().cpu().numpy() == 0)
      snap = sim.state.snapshot()
      if step == 24:
        checkpoint = sim.snapshot()
      if step == 39:
        final_at_checkpoint = sim.state.snapshot()
        references_at_checkpoint = [mujoco.MjData(model) for _ in references]
        for dest, source in zip(references_at_checkpoint, references):
          mujoco.mj_copyData(dest, model, source)
      for world, data in enumerate(references):
        max_qpos = max(max_qpos, float(np.max(np.abs(snap.qpos[world]-data.qpos))))
        max_qvel = max(max_qvel, float(np.max(np.abs(snap.qvel[world]-data.qvel))))
        max_qacc = max(max_qacc, float(np.max(np.abs(snap.qacc[world]-data.qacc))))
      if step % 10 == 0 or step == 119:
        sensor = sim.sensor_values().detach().cpu().numpy()
        for world, data in enumerate(references):
          mujoco.mj_forward(model, data)
          max_sensor = max(max_sensor, float(np.max(np.abs(
              sensor[world] - data.sensordata))))
    assert max_qpos < 8e-4, max_qpos
    assert max_qvel < 5e-3, max_qvel
    assert max_qacc < 2e-2, max_qacc
    assert max_sensor < 5e-3, max_sensor
    assert checkpoint is not None and final_at_checkpoint is not None
    sim.restore(checkpoint)
    for _ in range(15):
      sim.step()
    replay = sim.state.snapshot()
    for key in ("qpos", "qvel", "qacc", "time", "status"):
      np.testing.assert_array_equal(getattr(final_at_checkpoint, key),
                                    getattr(replay, key))

    assert references_at_checkpoint is not None
    references = references_at_checkpoint
    before_world0 = {
        "qpos": sim.state.qpos[0].detach().cpu().numpy().copy(),
        "qvel": sim.state.qvel[0].detach().cpu().numpy().copy(),
        "time": float(sim.state.time[0].detach().cpu().item()),
    }
    reset = sim.state.qpos[1].detach().cpu().numpy().copy()
    reset[0] = .12
    reset_velocity = np.array([.03, -.02], dtype=np.float32)
    sim.reset(env_ids=[1], qpos=reset, qvel=reset_velocity)
    references[1] = mujoco.MjData(model)
    references[1].qpos[:] = reset
    references[1].qvel[:] = reset_velocity
    mujoco.mj_forward(model, references[1])
    np.testing.assert_array_equal(sim.state.qpos[0].detach().cpu().numpy(),
                                  before_world0["qpos"])
    np.testing.assert_array_equal(sim.state.qvel[0].detach().cpu().numpy(),
                                  before_world0["qvel"])
    assert float(sim.state.time[0].detach().cpu().item()) == before_world0["time"]
    np.testing.assert_array_equal(sim.state.qpos[1].detach().cpu().numpy(), reset)
    np.testing.assert_array_equal(sim.state.qvel[1].detach().cpu().numpy(),
                                  reset_velocity)
    assert float(sim.state.time[1].detach().cpu().item()) == 0.0
    for reset_step in range(4):
      for data in references:
        data.qfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = _cpu_wrench(model, data, site_id)[5]
        mujoco.mj_step(model, data)
      with pytest.MonkeyPatch.context() as guard:
        guard.setattr(mujoco, "mj_step", forbidden_hotpath)
        guard.setattr(mujoco, "mj_forward", forbidden_hotpath)
        guard.setattr(torch.Tensor, "cpu", forbidden_hotpath)
        guard.setattr(torch.Tensor, "numpy", forbidden_hotpath)
        status = sim.step()
      assert np.all(status.detach().cpu().numpy() == 0)
      snap = sim.state.snapshot()
      for world, data in enumerate(references):
        np.testing.assert_allclose(
            snap.qpos[world], data.qpos, atol=8e-4, rtol=5e-3,
            err_msg=f"post-reset qpos, step {reset_step}, world {world}")
        np.testing.assert_allclose(
            snap.qvel[world], data.qvel, atol=5e-3, rtol=5e-3,
            err_msg=f"post-reset qvel, step {reset_step}, world {world}")
  finally:
    default_registry.unregister("site_feedback_trajectory_test", PluginType.FORCE)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in native Metal guarded FORCE kernel")
def test_native_site_feedback_masked_force_preserves_unselected_output_row():
  import torch
  from mujoco_metal.native_api import mj_fwdPosition, mj_fwdVelocity
  from mujoco_metal.simulation import MetalSimulation

  model = _model()
  qpos = np.array([[.2, -.4], [-.25, .48]], dtype=np.float32)
  qvel = np.array([[.05, -.03], [-.02, .06]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  position = mj_fwdPosition(sim, return_record=True)
  stage = mj_fwdVelocity(sim, record=position)
  plugin = SiteFeedbackPlugin()
  plugin.init(model, batch_size=2, device=sim.state._device)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for world, data in enumerate(refs):
    data.qpos[:], data.qvel[:] = qpos[world], qvel[world]
    mujoco.mj_forward(model, data)
  plugin._queries._plugin_force[1].fill_(-432.25)
  actual = plugin.run_device(
      sim.state, dynamics=stage["dynamics"],
      compute_mask=torch.tensor([True, False], dtype=torch.bool, device="mps"))
  site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip"))
  np.testing.assert_allclose(
      actual[0].cpu().numpy(), _cpu_wrench(model, refs[0], site_id)[5],
      rtol=5e-5, atol=5e-6)
  np.testing.assert_array_equal(
      actual[1].cpu().numpy(), np.full(model.nv, -432.25, dtype=np.float32))
