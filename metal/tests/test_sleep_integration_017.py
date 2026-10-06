# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pre-integration pinned sleep transition and awake integration gates."""

import os

import mujoco
import numpy as np
import pytest


XML = """<mujoco><option timestep=".01" gravity="0 0 0">
  <flag sleep="enable" contact="disable"/></option><worldbody>
  <body name="free" pos="-1 0 1" sleep="allowed"><freejoint name="freej"/>
    <geom type="box" size=".1 .2 .3" mass="2"/>
    <body name="hinged" pos="0 0 .4"><joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size=".1" mass=".5"/></body>
  </body>
  <body name="slider" pos="1 0 1" sleep="allowed"><joint name="slide" type="slide" axis="1 0 0"/>
    <geom type="sphere" size=".2" mass="1"/></body>
</worldbody></mujoco>"""


def _cpu_sleep_filtered_euler(model, qpos, qvel, qacc, time, dt,
                               body_ids, dof_ids):
  """Use MuJoCo's joint-position integrator plus pinned sleep list ownership."""
  qpos_out = np.asarray(qpos, dtype=np.float64).copy()
  qvel_out = np.zeros_like(qvel, dtype=np.float64)
  dof_ids = np.asarray(dof_ids, dtype=np.int32)
  if dof_ids.size:
    qvel_out[dof_ids] = qvel[dof_ids] + dt * qacc[dof_ids]
  integrated = qpos_out.copy()
  mujoco.mj_integratePos(model, integrated, qvel_out, dt)
  awake_bodies = set(map(int, body_ids))
  qpos_size = (7, 4, 1, 1)
  for joint in range(model.njnt):
    body = int(model.jnt_bodyid[joint])
    if body in awake_bodies:
      continue
    address = int(model.jnt_qposadr[joint])
    width = qpos_size[int(model.jnt_type[joint])]
    integrated[address:address + width] = qpos_out[address:address + width]
  return integrated, qvel_out, time + dt


def test_cpu_awake_euler_retains_sleeping_pose_and_zeros_velocity():
  model = mujoco.MjModel.from_xml_string(XML)
  batch = 2
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  assert model.ntree == 2
  awake = np.array([[1, 0], [0, 0]], dtype=np.int32)
  lists = []
  for world in range(batch):
    bodies = [b for b, tree in enumerate(body_tree)
              if tree < 0 or awake[world, tree]]
    dofs = [d for d, tree in enumerate(model.dof_treeid)
            if tree >= 0 and awake[world, tree]]
    lists.append((bodies, dofs))
  qpos = np.tile(np.asarray(model.qpos0), (batch, 1))
  qvel = np.linspace(-.4, .7, batch * model.nv).reshape(batch, model.nv)
  qacc = np.linspace(.8, -.6, batch * model.nv).reshape(batch, model.nv)
  qpos[0, 0:3] += [.2, -.1, .3]
  qpos[0, 3:7] = [np.cos(.2), 0, np.sin(.2), 0]
  dt = float(model.opt.timestep)
  for world, (bodies, dofs) in enumerate(lists):
    expected = _cpu_sleep_filtered_euler(
        model, qpos[world], qvel[world], qacc[world], .3,
        dt, bodies, dofs)
    assert expected[2] == pytest.approx(.3 + dt)
    np.testing.assert_array_equal(expected[1][
        np.setdiff1d(np.arange(model.nv), dofs)], 0.0)
    inactive_bodies = [b for b, tree in enumerate(body_tree)
                       if tree >= 0 and not awake[world, tree]]
    for body in inactive_bodies:
      for joint in range(model.body_jntadr[body],
                         model.body_jntadr[body] + model.body_jntnum[body]):
        adr = int(model.jnt_qposadr[joint])
        width = (7, 4, 1, 1)[int(model.jnt_type[joint])]
        np.testing.assert_array_equal(expected[0][adr:adr + width],
                                      qpos[world, adr:adr + width])


def test_cpu_sleep_cutoff_uses_preintegration_velocity():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option timestep=".01" gravity="0 0 0" sleep_tolerance=".05">
      <flag sleep="enable" contact="disable"/></option><worldbody>
      <body sleep="allowed"><joint type="slide" axis="1 0 0" damping="20"/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
  data = mujoco.MjData(model)
  tree = int(model.dof_treeid[0])
  data.qvel[0] = .06
  before = float(data.qvel[0])
  mujoco.mj_step(model, data)
  # Damping makes post-integration velocity cross below tolerance, but pinned
  # mj_sleep already observed .06 and must leave the awake countdown untouched.
  assert before > model.opt.sleep_tolerance
  assert abs(data.qvel[0]) < model.opt.sleep_tolerance
  assert int(data.tree_asleep[tree]) == -11
  preintegration_velocity = float(data.qvel[0])
  mujoco.mj_step(model, data)
  assert preintegration_velocity < model.opt.sleep_tolerance
  assert int(data.tree_asleep[tree]) == -10


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_awake_euler_integrates_only_pre_sleep_lists():
  import torch
  from mujoco_metal.integration import MetalEulerIntegration
  from mujoco_metal.model import load_model
  from mujoco_metal.sleep_schedule import DeviceSleepScheduler

  model = mujoco.MjModel.from_xml_string(XML)
  model.opt.timestep = .01
  descriptor = load_model(model)
  batch, dt = 2, float(model.opt.timestep)
  scheduler = DeviceSleepScheduler(model, batch_size=batch)
  awake = np.array([[1, 0], [0, 0]], dtype=np.int32)
  tree_state = np.full((batch, model.ntree), -11, dtype=np.int32)
  tree_state[awake == 0] = np.indices(awake.shape)[1][awake == 0]
  scheduler.tree_state.copy_(torch.as_tensor(tree_state, device="mps"))
  scheduler.tree_awake.copy_(torch.as_tensor(awake, device="mps"))
  scheduler._build_awake_lists()
  lists = scheduler.awake_lists()

  qpos_np = np.tile(np.asarray(model.qpos0, dtype=np.float32), (batch, 1))
  qpos_np[0, 0:3] += [.2, -.1, .3]
  qpos_np[0, 3:7] = [np.cos(.2), 0, np.sin(.2), 0]
  qvel_np = np.linspace(-.4, .7, batch * model.nv, dtype=np.float32).reshape(batch, model.nv)
  qacc_np = np.linspace(.8, -.6, batch * model.nv, dtype=np.float32).reshape(batch, model.nv)
  time_np = np.array([.3, .6], dtype=np.float32)
  qpos = torch.as_tensor(qpos_np, device="mps")
  qvel = torch.as_tensor(qvel_np, device="mps")
  qacc = torch.as_tensor(qacc_np, device="mps")
  scheduler.zero_asleep_acceleration(qacc)
  stage = MetalEulerIntegration(descriptor, batch, dt)
  out_qpos, out_qvel, out_time, status = stage.run_device(
      qpos, qvel, qacc, torch.as_tensor(time_np, device="mps"),
      torch.zeros(batch, dtype=torch.int32, device="mps"), awake_lists=lists)
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
  np.testing.assert_allclose(out_time.cpu().numpy(), time_np + dt, rtol=0, atol=1e-7)
  masked_qacc = qacc.cpu().numpy()
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  for world in range(batch):
    inactive = np.flatnonzero((dof_tree >= 0) & (awake[world, dof_tree] == 0))
    np.testing.assert_array_equal(masked_qacc[world, inactive], 0.0)
  for world in range(batch):
    body_count = int(lists["counts"][world, 0].item())
    dof_count = int(lists["counts"][world, 2].item())
    bodies = lists["body_ids"][world, :body_count].cpu().numpy()
    dofs = lists["dof_ids"][world, :dof_count].cpu().numpy()
    expected = _cpu_sleep_filtered_euler(
        model, qpos_np[world], qvel_np[world], qacc.cpu().numpy()[world],
        float(time_np[world]), dt, bodies, dofs)
    np.testing.assert_allclose(out_qpos[world].cpu().numpy(), expected[0],
                               rtol=0, atol=3e-6)
    np.testing.assert_allclose(out_qvel[world].cpu().numpy(), expected[1],
                               rtol=0, atol=3e-6)
