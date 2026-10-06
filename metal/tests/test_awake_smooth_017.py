# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Sleep-filtered smooth, RNE, and mass-factor execution gates."""

import os

import mujoco
import numpy as np
import pytest


XML = """<mujoco>
  <option gravity="0 0 -9.81"><flag sleep="enable"/></option>
  <worldbody>
    <body name="left" pos="-1 0 1" sleep="allowed">
      <freejoint name="left_free"/><geom type="sphere" size=".1" mass="1" contype="0" conaffinity="0"/>
      <body name="left_child" pos="0 0 .4">
        <joint name="left_hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".05 .2" mass=".3" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="right" pos="1 0 1" sleep="allowed">
      <freejoint name="right_free"/><geom type="box" size=".1 .2 .3" mass="2" contype="0" conaffinity="0"/>
    </body>
    <body name="anchor" pos="3 0 0">
      <geom type="box" size=".08 .08 .08" mass=".5" contype="0" conaffinity="0"/>
      <body name="anchored" pos="0 0 .5" sleep="allowed">
        <joint name="anchored_hinge" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".07" mass=".8" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def _set_pinned_sleep_filter(model, data, active_tree):
  """Install MuJoCo 3.10's already-lowered sleep lists without stepping."""
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  parent = np.asarray(model.body_parentid, dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  body_ids = [b for b, tree in enumerate(body_tree)
              if tree < 0 or tree == active_tree]
  parent_ids = [b for b in range(1, model.nbody)
                if body_tree[parent[b]] < 0 or body_tree[parent[b]] == active_tree]
  dof_ids = [d for d, tree in enumerate(dof_tree) if tree == active_tree]
  data.body_awake_ind[:len(body_ids)] = body_ids
  data.parent_awake_ind[:len(parent_ids)] = parent_ids
  data.dof_awake_ind[:len(dof_ids)] = dof_ids
  data.body_awake[:] = [
      int(mujoco.mjtSleepState.mjS_STATIC) if tree < 0 else
      (int(mujoco.mjtSleepState.mjS_AWAKE) if tree == active_tree else
       int(mujoco.mjtSleepState.mjS_ASLEEP))
      for tree in body_tree]
  asleep = np.zeros(model.ntree, dtype=np.int32)
  asleep[active_tree] = -5
  data.tree_asleep[:] = asleep
  data.nbody_awake = len(body_ids)
  data.nparent_awake = len(parent_ids)
  data.nv_awake = len(dof_ids)
  return np.asarray(body_ids, dtype=np.int32), np.asarray(dof_ids, dtype=np.int32)


def _pinned_smooth_reference(model, qpos, qvel, active_tree):
  """Run MuJoCo's CRB/RNE with a single active tree and retain old outputs."""
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  # First compute all position-dependent data and establish retained values.
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  mujoco.mj_crb(model, data)
  full_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, full_mass)
  mujoco.mj_comVel(model, data)
  data.qfrc_bias[:] = np.arange(model.nv, dtype=np.float64) + 19.0
  mujoco.mj_rne(model, data, 0, data.qfrc_bias)
  retained_bias = data.qfrc_bias.copy()

  bodies, dofs = _set_pinned_sleep_filter(model, data, active_tree)
  parents = np.asarray(data.parent_awake_ind[:data.nparent_awake]).copy()
  # Pinned mj_comPos, mj_crb, mj_comVel, and mj_rne consume these lists and
  # leave inactive matrix rows and bias entries untouched.
  mujoco.mj_comPos(model, data)
  mujoco.mj_crb(model, data)
  awake_mass = np.zeros((model.nv, model.nv), dtype=np.float64)
  mujoco.mj_fullM(model, data, awake_mass)
  mujoco.mj_comVel(model, data)
  mujoco.mj_rne(model, data, 0, data.qfrc_bias)
  return {
      "body_ids": bodies,
      "parent_ids": parents,
      "dof_ids": dofs,
      "mass": awake_mass,
      "full_mass": full_mass,
      "bias": data.qfrc_bias.copy(),
      "retained_bias": retained_bias,
      "cvel": np.asarray(data.cvel).copy().reshape(model.nbody, 6),
      "cdof_dot": np.asarray(data.cdof_dot).copy().reshape(model.nv, 6),
  }


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_awake_smooth_rne_and_dense_factor_skip_inactive_dofs_gpu():
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.sleep_schedule import DeviceSleepScheduler
  from mujoco_metal.smooth_metal import MetalSmoothDynamics
  from mujoco_metal.smooth_solve import MetalDenseSolve

  model = mujoco.MjModel.from_xml_string(XML)
  descriptor = load_model(model)
  batch, nv = 2, int(model.nv)
  assert model.ntree == 3
  scheduler = DeviceSleepScheduler(model, batch_size=batch)
  smooth = MetalSmoothDynamics(descriptor, batch_size=batch)
  solver = MetalDenseSolve(nv, batch)
  qpos_host = np.tile(np.asarray(model.qpos0, dtype=np.float64), (batch, 1))
  free_addresses = {}
  for name in ("left_free", "right_free"):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    free_addresses[name] = int(model.jnt_qposadr[joint])
  hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hinge")
  qpos_host[:, int(model.jnt_qposadr[hinge])] = 0.43
  anchored_hinge = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "anchored_hinge")
  qpos_host[:, int(model.jnt_qposadr[anchored_hinge])] = -0.29
  for world, angle in enumerate((0.61, -0.37)):
    left_adr = free_addresses["left_free"]
    right_adr = free_addresses["right_free"]
    qpos_host[world, left_adr + 3:left_adr + 7] = (
        np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))
    qpos_host[world, right_adr + 3:right_adr + 7] = (
        np.cos(-angle / 3), 0.0, np.sin(-angle / 3), 0.0)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left")
  right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right")
  active_trees = [int(body_tree[left_body]), int(body_tree[right_body])]
  assert len(set(active_trees)) == 2
  anchored_body = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, "anchored")
  anchor_body = int(model.body_parentid[anchored_body])
  qvel_host = np.zeros((batch, nv), dtype=np.float32)
  for world, active_tree in enumerate(active_trees):
    active = np.flatnonzero(dof_tree == active_tree)
    qvel_host[world, active] = np.linspace(-0.27, 0.31, active.size)
  qpos = torch.as_tensor(qpos_host.astype(np.float32), device="mps")
  qvel = torch.as_tensor(qvel_host, dtype=torch.float32, device="mps")

  # Capture initialized full-workspace entries so we can check pinned retention
  # after one independent tree sleeps in each world.
  full = smooth.run_device(qpos, qvel)
  full_mass = full["mass_matrix"].clone()
  full_bias = full["qfrc_bias"].clone()
  sleep_state = np.zeros((batch, model.ntree), dtype=np.int32)
  for world, active_tree in enumerate(active_trees):
    sleep_state[world, active_tree] = -11
  scheduler.tree_state.copy_(torch.as_tensor(sleep_state, device="mps"))
  scheduler.tree_awake.copy_((scheduler.tree_state < 0).to(dtype=torch.int32))
  scheduler._build_awake_lists()
  awake = scheduler.awake_lists()
  actual = smooth.run_device(qpos, qvel, awake_lists=awake)
  mass = actual["mass_matrix"]
  bias = actual["qfrc_bias"]
  for world, active_tree in enumerate(active_trees):
    active = np.flatnonzero(dof_tree == active_tree)
    inactive = np.flatnonzero(dof_tree != active_tree)
    oracle = _pinned_smooth_reference(
        model, qpos_host[world], qvel_host[world], active_tree)
    np.testing.assert_array_equal(
        awake["body_ids"][world, :len(oracle["body_ids"])].cpu().numpy(),
        oracle["body_ids"])
    np.testing.assert_array_equal(
        awake["parent_ids"][world, :len(oracle["parent_ids"])].cpu().numpy(),
        oracle["parent_ids"])
    np.testing.assert_array_equal(
        awake["dof_ids"][world, :len(oracle["dof_ids"])].cpu().numpy(),
        oracle["dof_ids"])
    mass_host = mass[world].cpu().numpy()
    full_host = full_mass[world].cpu().numpy()
    np.testing.assert_allclose(
        mass_host[np.ix_(active, active)],
        oracle["mass"][np.ix_(active, active)], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        mass_host[np.ix_(active, active)],
        full_host[np.ix_(active, active)], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(
        mass_host[inactive], full_host[inactive])
    np.testing.assert_array_equal(
        bias[world, inactive].cpu().numpy(), full_bias[world, inactive].cpu().numpy())
    np.testing.assert_allclose(
        bias[world, inactive].cpu().numpy(),
        oracle["retained_bias"][inactive], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        bias[world, active].cpu().numpy(),
        oracle["bias"][active], rtol=3e-5, atol=3e-5)
    bodies = oracle["body_ids"]
    assert anchored_body not in bodies
    assert anchored_body in oracle["parent_ids"]
    assert anchor_body in bodies
    np.testing.assert_allclose(
        actual["cvel"][world, bodies].cpu().numpy(),
        oracle["cvel"][bodies], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        actual["cdof_dot"][world, active].cpu().numpy(),
        oracle["cdof_dot"][active], rtol=3e-5, atol=3e-5)

  # The solver gathers only active principal entries. NaNs in retained asleep
  # RHS slots do not poison the world, and inactive qacc remains caller-owned.
  rhs = torch.ones((batch, nv), dtype=torch.float32, device="mps")
  retained = torch.full((batch, nv), 7.0, dtype=torch.float32, device="mps")
  for world, active_tree in enumerate(active_trees):
    inactive = np.flatnonzero(dof_tree != active_tree)
    if inactive.size:
      rhs[world, inactive] = float("nan")
  qacc, status = solver.run_device(
      mass, rhs, awake_lists=awake, retained=retained)
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
  qacc_host = qacc.cpu().numpy()
  rhs_host = rhs.cpu().numpy()
  mass_host = mass.cpu().numpy()
  for world, active_tree in enumerate((0, 1)):
    active = np.flatnonzero(dof_tree == active_tree)
    inactive = np.flatnonzero(dof_tree != active_tree)
    reference = np.linalg.solve(
        mass_host[world][np.ix_(active, active)], rhs_host[world, active])
    np.testing.assert_allclose(qacc_host[world, active], reference,
                               rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(qacc_host[world, inactive], 7.0)

  # Empty active systems perform no factorization, tolerate arbitrary retained
  # asleep inputs and preserve every output coordinate.
  empty_cycles = np.tile(np.arange(model.ntree, dtype=np.int32),
                         (scheduler.batch_size, 1))
  scheduler.tree_state.copy_(torch.as_tensor(
      empty_cycles, dtype=torch.int32, device="mps"))
  scheduler.tree_awake.zero_()
  scheduler._build_awake_lists()
  empty = scheduler.awake_lists()
  rhs.fill_(float("nan"))
  qacc, status = solver.run_device(
      mass, rhs, awake_lists=empty, retained=retained)
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
  np.testing.assert_array_equal(qacc.cpu().numpy(), 7.0)
