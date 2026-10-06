"""MuJoCo 3.10 CPU oracles and opt-in native checks for transmissions."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.transmissions import MetalTransmissions
from mujoco_metal.transmissions import TransmissionModel
from mujoco_metal.transmissions import actuator_awake_cpu


_XML = """<mujoco><option gravity="0 0 0"/><worldbody>
  <body><joint name="hinge" type="hinge" axis="0 1 0"/>
  <geom type="capsule" size=".08 .2"/></body>
  <body pos="1 0 0"><joint name="slide" type="slide" axis="1 0 0"/>
  <geom type="box" size=".1 .1 .1"/></body>
</worldbody><actuator>
  <general name="position" joint="hinge" dyntype="none" gaintype="fixed"
    biastype="affine" gainprm="2" biasprm="0 -2 -.3"/>
  <general name="affine" joint="slide" dyntype="none" gaintype="affine"
    gainprm="1.2 .4 -.1" biastype="affine" biasprm=".2 -.2 -.3"
    ctrllimited="true" ctrlrange="-1 1" forcelimited="true"
    forcerange="-2 2" group="1"/>
  <general name="velocity" joint="hinge" dyntype="none" gaintype="fixed"
    gainprm=".7" biastype="affine" biasprm="0 0 -.7" group="2"/>
</actuator></mujoco>"""


def _oracle(model, qpos, qvel, ctrl):
  rows = []
  for q, v, u in zip(qpos, qvel, ctrl):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = v
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)
    rows.append(data.qfrc_actuator.copy())
  return np.asarray(rows)


def _pinned_sleep_actuation(model, qpos, qvel, ctrl, tree_awake):
  """Run pinned transmission/actuation with MuJoCo's real awake lists."""
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  data.ctrl[:] = ctrl
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  mujoco.mj_comVel(model, data)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  parent = np.asarray(model.body_parentid, dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  body_ids = [b for b, tree in enumerate(body_tree)
              if tree < 0 or tree_awake[tree]]
  parent_ids = [b for b in range(1, model.nbody)
                if body_tree[parent[b]] < 0 or tree_awake[body_tree[parent[b]]]]
  dof_ids = [d for d, tree in enumerate(dof_tree)
             if tree >= 0 and tree_awake[tree]]
  data.body_awake_ind[:len(body_ids)] = body_ids
  data.parent_awake_ind[:len(parent_ids)] = parent_ids
  data.dof_awake_ind[:len(dof_ids)] = dof_ids
  data.body_awake[:] = [
      int(mujoco.mjtSleepState.mjS_STATIC) if tree < 0 else
      (int(mujoco.mjtSleepState.mjS_AWAKE) if tree_awake[tree] else
       int(mujoco.mjtSleepState.mjS_ASLEEP)) for tree in body_tree]
  data.tree_asleep[:] = [-11 if awake else tree
                         for tree, awake in enumerate(tree_awake)]
  data.nbody_awake = len(body_ids)
  data.nparent_awake = len(parent_ids)
  data.nv_awake = len(dof_ids)
  data.ntree_awake = int(np.count_nonzero(tree_awake))
  mujoco.mj_transmission(model, data)
  mujoco.mj_fwdVelocity(model, data)
  mujoco.mj_fwdActuation(model, data)
  return data.qfrc_actuator.copy(), data.actuator_force.copy()


def test_fixed_and_affine_gains_biases_match_mujoco_joint_oracle():
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos = np.array([[.35, -.4], [-.6, .7], [.1, .2]])
  qvel = np.array([[.8, -.5], [-.3, 1.1], [0., -.2]])
  ctrl = np.array([[.2, 2.0, .8], [-.5, -.4, -.7], [.6, .25, 1.2]])
  stage = TransmissionModel.from_model(model)
  np.testing.assert_allclose(stage.generalized_force(qpos, qvel, ctrl), _oracle(model, qpos, qvel, ctrl), rtol=0, atol=2e-7)
  length, moment, velocity = stage.transmission_state(qpos, qvel)
  np.testing.assert_allclose(length, np.stack([qpos[:, 0], qpos[:, 1], qpos[:, 0]], axis=1), rtol=0, atol=1e-7)
  np.testing.assert_allclose(velocity, np.stack([qvel[:, 0], qvel[:, 1], qvel[:, 0]], axis=1), rtol=0, atol=1e-7)
  assert moment.shape == (len(qpos), model.nu, model.nv)


def test_sleep_awake_actuator_classification_matches_joint_and_tendon_rules():
  model = mujoco.MjModel.from_xml_string(_XML)
  awake = np.array([[1, 0], [0, 1]], dtype=np.int32)
  np.testing.assert_array_equal(
      actuator_awake_cpu(model, awake),
      [[True, False, True], [False, True, False]])

  tendon_xml = """<mujoco><option><flag sleep="enable"/></option><worldbody>
    <body><joint name="a" type="hinge"/><geom type="sphere" size=".1"/></body>
    <body pos="1 0 0"><joint name="b" type="hinge"/><geom type="sphere" size=".1"/></body>
    <body pos="2 0 0"><joint name="c" type="hinge"/><geom type="sphere" size=".1"/></body>
  </worldbody><tendon><fixed name="two"><joint joint="a" coef="1"/><joint joint="b" coef="1"/></fixed>
    <fixed name="three"><joint joint="a" coef="1"/><joint joint="b" coef="1"/><joint joint="c" coef="1"/></fixed>
  </tendon><actuator><general tendon="two" dyntype="none" gaintype="fixed" gainprm="1"/>
    <general tendon="three" dyntype="none" gaintype="fixed" gainprm="1"/>
  </actuator></mujoco>"""
  model = mujoco.MjModel.from_xml_string(tendon_xml)
  assert model.ntree == 3
  states = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32)
  np.testing.assert_array_equal(
      actuator_awake_cpu(model, states), [[True, True], [False, True]])


def test_pinned_actuator_sleep_reference_uses_current_control_and_clears_force():
  model = mujoco.MjModel.from_xml_string(_XML)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  qpos = np.array([.2, -.65])
  qvel = np.array([-.4, .9])
  ctrl = np.array([.7, -.8, .35])
  tree_awake = np.array([1, 0], dtype=np.int32)
  qfrc, force = _pinned_sleep_actuation(model, qpos, qvel, ctrl, tree_awake)
  full_qfrc = _oracle(model, qpos[None, :], qvel[None, :], ctrl[None, :])[0]
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  inactive = np.flatnonzero((dof_tree >= 0) & (tree_awake[dof_tree] == 0))
  expected = full_qfrc.copy()
  expected[inactive] = 0
  np.testing.assert_allclose(qfrc, expected, rtol=0, atol=2e-7)
  meta = TransmissionModel.from_model(model)
  for actuator, count in enumerate(meta.actuator_treenum):
    if count in (1, 2) and not any(
        tree_awake[t] for t in meta.actuator_treeids[actuator, :count]
        if t >= 0):
      assert force[actuator] == 0


def test_fixed_joint_tendon_transmission_uses_raw_qpos_reference_semantics():
  xml = """<mujoco><compiler angle="radian"/><worldbody>
    <body><joint name="h" type="hinge" ref=".25"/>
      <geom type="capsule" size=".05 .1"/></body>
    <body pos="1 0 0"><joint name="s" type="slide" ref="-.15"/>
      <geom type="sphere" size=".1"/></body>
  </worldbody>
  <tendon><fixed name="cable"><joint joint="h" coef="2"/>
    <joint joint="s" coef="-.5"/></fixed></tendon>
  <actuator><general name="pull" tendon="cable" gear="1.4"
    dyntype="none" gaintype="affine" gainprm="1.1 .2 -.3"
    biastype="affine" biasprm=".1 -.7 -.2" forcelimited="true"
    forcerange="-1.5 1.5"/></actuator></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.qpos0.tolist() == pytest.approx([.25, -.15])
  qpos = np.array([[.4, -.7], [-.1, .3]])
  qvel = np.array([[.8, -.4], [-.5, .2]])
  ctrl = np.array([[.3], [-.8]])
  stage = TransmissionModel(model)
  np.testing.assert_allclose(stage.generalized_force(qpos, qvel, ctrl), _oracle(model, qpos, qvel, ctrl), rtol=0, atol=2e-7)
  length, moment, velocity = stage.transmission_state(qpos, qvel)
  expected_length = 1.4 * (2*qpos[:, 0] - .5*qpos[:, 1])
  expected_velocity = 1.4 * (2*qvel[:, 0] - .5*qvel[:, 1])
  np.testing.assert_allclose(length[:, 0], expected_length, rtol=0, atol=1e-7)
  np.testing.assert_allclose(velocity[:, 0], expected_velocity, rtol=0, atol=1e-7)
  np.testing.assert_allclose(moment[0, 0], [2.8, -.7], rtol=0, atol=1e-7)


def test_control_force_limits_and_all_disable_modes_match_mujoco():
  qpos = np.array([[.35, -.4], [-.6, .7]])
  qvel = np.array([[.8, -.5], [-.3, 1.1]])
  ctrl = np.array([[2., 4., 3.], [-.5, -4., -.7]])
  for mutate in (
      lambda m: setattr(m.opt, "disableflags", int(m.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)),
      lambda m: setattr(m.opt, "disableactuator", 1 << 1),
      lambda m: setattr(m.opt, "disableflags", int(m.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)),
  ):
    model = mujoco.MjModel.from_xml_string(_XML)
    mutate(model)
    np.testing.assert_allclose(TransmissionModel(model).generalized_force(qpos, qvel, ctrl), _oracle(model, qpos, qvel, ctrl), rtol=0, atol=2e-7)


def test_empty_and_malformed_actuator_inputs():
  model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="hinge"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
  stage = TransmissionModel(model)
  result = stage.generalized_force(np.zeros((2, model.nq)), np.zeros((2, model.nv)), np.empty((2, 0)))
  np.testing.assert_array_equal(result, np.zeros((2, model.nv)))
  with pytest.raises(ValueError, match="ctrl"):
    stage.generalized_force(np.zeros((2, model.nq)), np.zeros((2, model.nv)), np.zeros((2, 1)))


@pytest.mark.parametrize(
    "xml, message",
    [
        (_XML.replace('joint="hinge" dyntype="none"', 'joint="hinge" dyntype="filter"'), "stateful"),
        (_XML.replace('joint="hinge" dyntype="none"', 'joint="hinge" dyntype="none" armature=".1"', 1), "armature"),
        (_XML.replace('joint="hinge" dyntype="none" gaintype="fixed"', 'joint="hinge" dyntype="none" gaintype="user"', 1), "gain"),
    ],
)
def test_unsupported_actuator_models_are_rejected(xml, message):
  with pytest.raises(ValueError, match=message):
    TransmissionModel(mujoco.MjModel.from_xml_string(xml))


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_joint_transmissions_match_mujoco():
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos = np.array([[.35, -.4], [-.6, .7]], dtype=np.float32)
  qvel = np.array([[.8, -.5], [-.3, 1.1]], dtype=np.float32)
  ctrl = np.array([[.2, 2., .8], [-.5, -.4, -.7]], dtype=np.float32)
  import torch
  stage = MetalTransmissions(model)
  actual = stage.run_device(
      torch.tensor(qpos, dtype=torch.float32, device="mps"),
      torch.tensor(qvel, dtype=torch.float32, device="mps"),
      torch.tensor(ctrl, dtype=torch.float32, device="mps"),
  )
  np.testing.assert_allclose(actual.cpu().numpy(), _oracle(model, qpos, qvel, ctrl), rtol=0, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_scalar_transmission_reuses_cached_rk_position_length():
  import torch

  model = mujoco.MjModel.from_xml_string(_XML)
  qpos_x3 = np.array([[.35, -.4], [-.6, .7]], dtype=np.float32)
  qpos_x0 = np.array([[-.2, .15], [.25, -.1]], dtype=np.float32)
  qvel_x0 = np.array([[.8, -.5], [-.3, 1.1]], dtype=np.float32)
  ctrl_x0 = np.array([[.2, 2., .8], [-.5, -.4, -.7]], dtype=np.float32)
  stage = MetalTransmissions(model, batch_size=2)
  context = stage.capture_position_context(
      torch.as_tensor(qpos_x3, dtype=torch.float32, device="mps"))
  actual = stage.run_device(
      torch.as_tensor(qpos_x0, dtype=torch.float32, device="mps"),
      torch.as_tensor(qvel_x0, dtype=torch.float32, device="mps"),
      torch.as_tensor(ctrl_x0, dtype=torch.float32, device="mps"),
      position_context=context)
  expected = _oracle(model, qpos_x3, qvel_x0, ctrl_x0)
  np.testing.assert_allclose(actual.cpu().numpy(), expected,
                             rtol=0, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_transmission_sleep_lists_use_per_world_count_columns():
  model = mujoco.MjModel.from_xml_string(_XML)
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  import torch
  tree_awake = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  dof_ids = np.full((3, model.nv), -1, dtype=np.int32)
  counts = np.zeros((3, 3), dtype=np.int32)
  for world in range(3):
    dofs = [d for d, tree in enumerate(dof_tree)
            if tree >= 0 and tree_awake[world, tree] != 0]
    dof_ids[world, :len(dofs)] = dofs
    counts[world, 2] = len(dofs)
  awake = {
      "dof_ids": torch.as_tensor(dof_ids, dtype=torch.int32, device="mps"),
      "counts": torch.as_tensor(counts, dtype=torch.int32, device="mps"),
      "tree_awake": torch.as_tensor(tree_awake, dtype=torch.int32, device="mps"),
  }
  seed_qpos = np.array([[.3, -.5], [-.4, .7], [.6, -.2]], dtype=np.float32)
  seed_qvel = np.array([[.8, -.3], [-.5, 1.1], [.4, .9]], dtype=np.float32)
  seed_ctrl = np.array([[.2, .6, -.4], [-.5, -.4, -.7], [.1, -.2, .8]], dtype=np.float32)
  qpos = seed_qpos + np.array([[.1, .2], [-.2, -.15], [.3, .25]], np.float32)
  qvel = seed_qvel + np.array([[-.4, .5], [.2, -.7], [-.3, .1]], np.float32)
  ctrl = seed_ctrl + np.array([[.4, -.3, .2], [.1, .2, -.5], [-.2, .6, .4]], np.float32)
  stage = MetalTransmissions(model)
  # Seed the persistent output buffers with a different full-awake input so
  # the second call must honor the current sleep policy and current controls.
  stage.run_device(
      torch.as_tensor(seed_qpos, dtype=torch.float32, device="mps"),
      torch.as_tensor(seed_qvel, dtype=torch.float32, device="mps"),
      torch.as_tensor(seed_ctrl, dtype=torch.float32, device="mps"))
  actual = stage.run_device(
      torch.as_tensor(qpos, dtype=torch.float32, device="mps"),
      torch.as_tensor(qvel, dtype=torch.float32, device="mps"),
      torch.as_tensor(ctrl, dtype=torch.float32, device="mps"), awake_lists=awake)
  expected, expected_force = zip(*[
      _pinned_sleep_actuation(model, qpos[w], qvel[w], ctrl[w], tree_awake[w])
      for w in range(3)])
  expected = np.asarray(expected, dtype=np.float32)
  expected_force = np.asarray(expected_force, dtype=np.float32)
  np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=2e-6)
  np.testing.assert_allclose(stage._last_force.cpu().numpy(), expected_force,
                             rtol=0, atol=2e-6)
  first_force_ptr = stage._last_force.data_ptr()
  second = stage.run_device(
      torch.as_tensor(qpos, dtype=torch.float32, device="mps"),
      torch.as_tensor(qvel, dtype=torch.float32, device="mps"),
      torch.as_tensor(ctrl, dtype=torch.float32, device="mps"), awake_lists=awake)
  assert stage._last_force.data_ptr() == first_force_ptr
  np.testing.assert_allclose(second.cpu().numpy(), expected, rtol=0, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_fixed_tendon_transmission_matches_mujoco():
  xml = """<mujoco><compiler angle="radian"/><worldbody>
    <body><joint name="h" type="hinge" ref=".25"/>
      <geom type="capsule" size=".05 .1"/></body>
    <body pos="1 0 0"><joint name="s" type="slide" ref="-.15"/>
      <geom type="sphere" size=".1"/></body>
  </worldbody><tendon><fixed name="cable"><joint joint="h" coef="2"/>
    <joint joint="s" coef="-.5"/></fixed></tendon>
  <actuator><general tendon="cable" gear="1.4" dyntype="none"
    gaintype="affine" gainprm="1.1 .2 -.3" biastype="affine"
    biasprm=".1 -.7 -.2" forcelimited="true" forcerange="-1.5 1.5"/>
  </actuator></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.array([[.4, -.7], [-.1, .3]], dtype=np.float32)
  qvel = np.array([[.8, -.4], [-.5, .2]], dtype=np.float32)
  ctrl = np.array([[.3], [-.8]], dtype=np.float32)
  import torch
  stage = MetalTransmissions(model)
  actual = stage.run_device(
      torch.tensor(qpos, dtype=torch.float32, device="mps"),
      torch.tensor(qvel, dtype=torch.float32, device="mps"),
      torch.tensor(ctrl, dtype=torch.float32, device="mps"),
  )
  np.testing.assert_allclose(actual.cpu().numpy(), _oracle(model, qpos, qvel, ctrl), rtol=0, atol=2e-6)
