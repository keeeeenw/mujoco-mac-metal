# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Opt-in checks for the supplementary compensated smooth-bias word."""

import os

import mujoco
import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU"),
]


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(),
                    reason="Apple MPS is unavailable")
def test_bias_low_is_supplementary_and_cleared_for_inactive_dofs_gpu():
  import torch

  from mujoco_metal.model import load_model
  from mujoco_metal.sleep_schedule import DeviceSleepScheduler
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option gravity="0 0 -9.81"><flag sleep="enable"/></option>
    <worldbody>
      <body name="left" pos="-1 0 1" sleep="allowed">
        <freejoint/><geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
        <body name="left_child" pos="0 0 .4">
          <joint name="child_hinge" axis="0 1 0"/>
          <geom type="capsule" size=".05 .2" mass=".3" contype="0" conaffinity="0"/>
        </body>
      </body>
      <body name="right" pos="1 0 1" sleep="allowed">
        <freejoint/><geom type="box" size=".1 .2 .3" mass="2" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
  </mujoco>""")
  batch, nv = 2, int(model.nv)
  smooth = MetalSmoothDynamics(load_model(model), batch_size=batch)
  scheduler = DeviceSleepScheduler(model, batch_size=batch)
  qpos = torch.as_tensor(
      np.tile(np.asarray(model.qpos0, dtype=np.float32), (batch, 1)),
      dtype=torch.float32, device="mps")
  qvel = torch.as_tensor(
      np.tile(np.linspace(-.2, .3, nv, dtype=np.float32), (batch, 1)),
      dtype=torch.float32, device="mps")

  full = smooth.run_device(qpos, qvel)
  assert full["qfrc_bias_low"].shape == (batch, nv)
  assert torch.isfinite(full["qfrc_bias_low"]).all()
  cpu = mujoco.MjData(model)
  cpu.qpos[:] = np.asarray(model.qpos0)
  cpu.qvel[:] = np.linspace(-.2, .3, nv)
  mujoco.mj_forward(model, cpu)
  np.testing.assert_allclose(full["qfrc_bias"].detach().cpu().numpy(),
                             np.tile(np.asarray(cpu.qfrc_bias, dtype=np.float32),
                                     (batch, 1)), rtol=3e-5, atol=3e-5)

  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left")
  right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right")
  active_tree = int(body_tree[left])
  asleep_tree = int(body_tree[right])
  sleep_state = torch.zeros((batch, model.ntree), dtype=torch.int32,
                           device="mps")
  sleep_state[:, active_tree] = -11
  scheduler.tree_state.copy_(sleep_state)
  scheduler.tree_awake.copy_((sleep_state < 0).to(dtype=torch.int32))
  scheduler._build_awake_lists()
  awake = scheduler.awake_lists()
  asleep_dof = torch.as_tensor(
      np.flatnonzero(np.asarray(model.dof_treeid) == asleep_tree),
      dtype=torch.long, device="mps")
  awake_dof = torch.as_tensor(
      np.flatnonzero(np.asarray(model.dof_treeid) == active_tree),
      dtype=torch.long, device="mps")
  assert asleep_dof.numel() > 0
  assert awake_dof.numel() > 0
  active_count = int(awake["counts"][0, 2].item())
  active_ids = awake["dof_ids"][0, :active_count]
  assert set(active_ids.tolist()) == set(awake_dof.tolist())

  partial = smooth.run_device(qpos, qvel, awake_lists=awake)
  assert partial["qfrc_bias"] is not None
  assert torch.count_nonzero(partial["qfrc_bias_low"][:, asleep_dof]) == 0
  assert torch.isfinite(partial["qfrc_bias"][:, asleep_dof]).all()

  # Pin a sleeping child in the parent-awake reverse list. Its retained
  # spatial low word must survive VEL and contribute to the awake parent's
  # generalized force, while the child's inactive DOF low row is cleared.
  child = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_child")
  child_joint = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "child_hinge")
  child_dof = int(model.jnt_dofadr[child_joint])
  lists = smooth._all_awake_lists(batch)
  lists.update(body_count=lists["counts"][:, 0],
               parent_count=lists["counts"][:, 1],
               dof_count=lists["counts"][:, 2])
  active_bodies = [body for body in range(model.nbody) if body != child]
  active_dofs = [dof for dof in range(nv) if dof != child_dof]
  for world in range(batch):
    lists["body_ids"][world].fill_(-1)
    lists["body_ids"][world, :len(active_bodies)] = torch.tensor(
        active_bodies, dtype=torch.int32, device="mps")
    lists["dof_ids"][world].fill_(-1)
    lists["dof_ids"][world, :len(active_dofs)] = torch.tensor(
        active_dofs, dtype=torch.int32, device="mps")
    lists["counts"][world, 0] = len(active_bodies)
    lists["counts"][world, 2] = len(active_dofs)
  workspace = smooth._workspace
  force_low = workspace["body_force_low"].reshape(batch, model.nbody, 6)
  force_high = workspace["body_force"].reshape(batch, model.nbody, 6)
  force_low[:, child].zero_()
  force_high[:, child].zero_()
  base_partial = smooth.run_device(qpos, qvel, awake_lists=lists)
  base_bias_low = base_partial["qfrc_bias_low"].clone()
  retained = torch.full((batch, 6), .125, dtype=torch.float32, device="mps")
  force_low[:, child].copy_(retained)
  with_child_low = smooth.run_device(qpos, qvel, awake_lists=lists)
  torch.testing.assert_close(force_low[:, child], retained, rtol=0, atol=0)
  parent_dofs = torch.as_tensor(
      np.flatnonzero(np.asarray(model.dof_bodyid) == left),
      dtype=torch.long, device="mps")
  parent_axes = smooth._workspace["cdof"].reshape(batch, nv, 6)[0,
                                                                parent_dofs]
  expected = .125 * parent_axes.sum(dim=1)
  torch.testing.assert_close(
      with_child_low["qfrc_bias_low"][0, parent_dofs] -
      base_bias_low[0, parent_dofs], expected, rtol=0, atol=2e-7)
  assert torch.count_nonzero(with_child_low["qfrc_bias_low"][:, child_dof]) == 0

  # Public position-only execution is the full low-word reset boundary.
  for name in ("cvel_low", "cdof_dot_low", "cacc_low", "body_force_low"):
    smooth._workspace[name].fill_(.125)
  smooth.run_device(qpos, qvel, _position_only=True)
  for name in ("bias_low", "cvel_low", "cdof_dot_low", "cacc_low",
               "body_force_low"):
    assert torch.count_nonzero(smooth._workspace[name]) == 0, name
