"""CPU oracle checks and opt-in GPU checks for passive joint forces."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.passive import PassiveForceModel
from mujoco_metal.passive import MetalPassiveForces


_XML = """<mujoco><option gravity="0 0 0"/><worldbody>
<body><joint type="hinge" stiffness="2" springref=".2" damping=".4"/>
<geom type="sphere" size=".1"/></body>
<body pos="1 0 0"><joint type="slide" axis="1 0 0" stiffness="1.5"
springref="-.1" damping=".2"/><geom type="sphere" size=".1"/></body>
</worldbody></mujoco>"""


def _oracle(model, qpos, qvel):
  rows = []
  for q, v in zip(qpos, qvel):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = v
    mujoco.mj_forward(model, data)
    rows.append(data.qfrc_spring + data.qfrc_damper)
  return np.asarray(rows)


def test_joint_springs_and_dampers_match_mujoco_310():
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos = np.array([[.7, -.5], [-.3, .2], [.2, -.1]])
  qvel = np.array([[1.2, -.8], [-.4, .5], [0., 0.]])
  stage = PassiveForceModel(model)
  np.testing.assert_allclose(stage.force(qpos, qvel), _oracle(model, qpos, qvel), rtol=0, atol=1e-7)


def test_polynomial_spring_and_damper_match_mujoco():
  model = mujoco.MjModel.from_xml_string(_XML)
  model.jnt_stiffnesspoly[0] = [.3, .1]
  model.dof_dampingpoly[0] = [.2, .05]
  qpos = np.array([[.8, -.5], [-.3, .2]])
  qvel = np.array([[1.2, -.8], [-.4, .5]])
  stage = PassiveForceModel(model)
  np.testing.assert_allclose(stage.force(qpos, qvel), _oracle(model, qpos, qvel), rtol=0, atol=1e-7)
  expected_derivative = model.dof_damping + 2*model.dof_dampingpoly[:, 0]*np.abs(qvel) + 3*model.dof_dampingpoly[:, 1]*qvel*qvel
  np.testing.assert_allclose(stage.damping_derivative(qvel), expected_derivative, rtol=0, atol=3e-8)


def test_disable_flags_and_input_validation():
  model = mujoco.MjModel.from_xml_string(_XML)
  model.opt.disableflags = int(mujoco.mjtDisableBit.mjDSBL_SPRING) | int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  stage = PassiveForceModel(model)
  result = stage.force(np.zeros((2, 2)), np.ones((2, 2)))
  np.testing.assert_array_equal(result, np.zeros((2, 2)))
  np.testing.assert_array_equal(stage.damping_derivative(np.ones((2, 2))), np.zeros((2, 2)))
  with pytest.raises(ValueError, match="shapes"):
    stage.force(np.zeros((2, 3)), np.zeros((2, 2)))


def test_ball_and_free_rigid_springs_match_mujoco_quaternion_convention():
  xml = """<mujoco><option gravity="0 0 0"/><worldbody>
  <body><joint type="free" stiffness="2"/><geom type="sphere" size=".1"/></body>
  <body pos="1 0 0"><joint type="ball" stiffness="1.5"/>
  <geom type="sphere" size=".1"/></body></worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.array([[.1, -.2, .3, .98, .1, -.05, .02, .96, .1, .2, -.1],
                   [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float64)
  qvel = np.zeros((2, model.nv))
  np.testing.assert_allclose(PassiveForceModel(model).force(qpos, qvel), _oracle(model, qpos, qvel), rtol=0, atol=2e-7)


def test_body_gravity_compensation_and_applied_wrenches_match_mujoco():
  xml = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
  <body gravcomp=".75"><joint type="hinge" axis="0 1 0"/>
  <geom type="capsule" size=".08 .2"/></body>
  <body pos="1 0 0" gravcomp="1"><joint type="slide" axis="1 0 0"/>
  <geom type="box" size=".1 .1 .1"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.array([[.3, -.2], [-.4, .5]])
  qvel = np.zeros((2, model.nv))
  wrench = np.zeros((2, model.nbody, 6))
  wrench[:, 1, :3] = [1.2, -.7, .5]
  wrench[:, 1, 3:] = [.3, .1, -.2]
  wrench[:, 2, :3] = [-.4, .8, 1.1]
  wrench[:, 2, 3:] = [0., .2, .1]
  stage = PassiveForceModel(model)
  actual = stage.force(qpos, qvel, wrench)
  expected = []
  for q, v, applied in zip(qpos, qvel, wrench):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    row = data.qfrc_passive.copy()
    for body in range(1, model.nbody):
      mujoco.mj_applyFT(model, data, applied[body, :3], applied[body, 3:], data.xipos[body].copy(), body, row)
    expected.append(row)
  np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


def _wrench_oracle(model, qpos, xfrc):
  rows = []
  for q, applied in zip(qpos, xfrc):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    row = data.qfrc_passive.copy()
    for body in range(1, model.nbody):
      mujoco.mj_applyFT(model, data, applied[body, :3], applied[body, 3:], data.xipos[body].copy(), body, row)
    rows.append(row)
  return np.asarray(rows)


def _pinned_sleep_passive(model, seed_qpos, seed_qvel, qpos, qvel, tree_awake):
  """Seed MuJoCo passive outputs, then run its real awake-index path."""
  data = mujoco.MjData(model)
  data.qpos[:] = seed_qpos
  data.qvel[:] = seed_qvel
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  mujoco.mj_comVel(model, data)
  mujoco.mj_passive(model, data)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  parent = np.asarray(model.body_parentid, dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  # Update positions and velocities first. The explicit sleep mask below then
  # tests the pinned passive kernel itself while avoiding a host state-write
  # wake, which normal Simulation setters already perform.
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  mujoco.mj_comVel(model, data)
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
  data.tree_asleep[:] = [(-11 if awake else tree)
                         for tree, awake in enumerate(tree_awake)]
  data.nbody_awake = len(body_ids)
  data.nparent_awake = len(parent_ids)
  data.nv_awake = len(dof_ids)
  data.ntree_awake = int(np.count_nonzero(tree_awake))
  mujoco.mj_passive(model, data)
  return data.qfrc_passive.copy()


def test_pinned_passive_sleep_reference_retains_only_inactive_dof_values():
  xml = """<mujoco><option gravity="0 0 0"><flag sleep="enable"/></option><worldbody>
    <body sleep="allowed"><joint type="hinge" stiffness="2" damping=".4"/>
      <geom type="sphere" size=".1"/></body>
    <body pos="1 0 0" sleep="allowed"><joint type="slide" axis="1 0 0"
      stiffness="1.5" damping=".2"/><geom type="sphere" size=".1"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  seed_qpos = np.array([.4, -.2])
  seed_qvel = np.array([.7, -.5])
  qpos = np.array([-.3, .6])
  qvel = np.array([-.8, 1.1])
  tree_awake = np.array([1, 0], dtype=np.int32)
  actual = _pinned_sleep_passive(
      model, seed_qpos, seed_qvel, qpos, qvel, tree_awake)
  old = _oracle(model, seed_qpos[None, :], seed_qvel[None, :])[0]
  current = _oracle(model, qpos[None, :], qvel[None, :])[0]
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  active = np.flatnonzero((dof_tree < 0) | (tree_awake[dof_tree] != 0))
  asleep = np.flatnonzero((dof_tree >= 0) & (tree_awake[dof_tree] == 0))
  np.testing.assert_allclose(actual[active], current[active], rtol=0, atol=1e-8)
  np.testing.assert_array_equal(actual[asleep], old[asleep])


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
@pytest.mark.parametrize("disable_gravity", [False, True])
def test_native_body_wrench_projection_and_gravcomp_match_mujoco(disable_gravity):
  xml = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
  <body pos=".2 -.1 .3" gravcomp=".6"><joint type="free"/>
    <geom pos=".1 0 .05" type="sphere" size=".1"/>
    <body pos=".4 .1 0" gravcomp="1"><joint type="ball" pos=".05 0 0"/>
      <geom pos="0 .08 0" type="sphere" size=".1"/>
      <body pos=".2 0 .1" gravcomp=".3"><joint type="hinge" axis="0 1 0" pos=".02 0 0"/>
        <geom pos=".03 0 .04" type="capsule" size=".03 .1"/>
      </body>
    </body>
  </body></worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  if disable_gravity:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
  qpos = np.array([
      [.3, -.2, .1, .9238795, 0, 0, .3826834, .9659258, .2588190, 0, 0, .4],
      [-.1, .25, .2, .9659258, .2588190, 0, 0, .9238795, 0, .3826834, 0, -.35],
  ], dtype=np.float32)
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  xfrc = np.zeros((2, model.nbody, 6), dtype=np.float32)
  xfrc[:, 0] = [91, 92, 93, 94, 95, 96]  # world-body entries must be ignored
  xfrc[:, 1] = [1.1, -.7, .3, .2, .4, -.5]
  xfrc[:, 2] = [-.2, .8, .5, .1, -.3, .6]
  xfrc[:, 3] = [.9, .1, -.4, -.2, .5, .3]
  import torch
  stage = MetalPassiveForces(model)
  actual = stage.run_device(
      torch.tensor(qpos, dtype=torch.float32, device="mps"),
      torch.tensor(qvel, dtype=torch.float32, device="mps"),
      torch.tensor(xfrc, dtype=torch.float32, device="mps"),
  )
  np.testing.assert_allclose(actual.cpu().numpy(), _wrench_oracle(model, qpos, xfrc), rtol=0, atol=3e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_gravcomp_uses_cached_rk_position_without_fk():
  import torch
  from mujoco_metal.passive import MetalPassiveForces

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option gravity="0 0 -9.81"/><worldbody>
      <body gravcomp="1"><joint type="hinge" axis="0 1 0"/>
        <inertial pos=".5 0 0" mass="1" diaginertia=".1 .1 .1"/>
        <geom type="sphere" pos=".5 0 0" size=".1" mass="1"/>
      </body></worldbody></mujoco>''')
  stage = MetalPassiveForces(model, batch_size=1)
  qpos_x3 = torch.tensor([[np.pi / 2]], dtype=torch.float32, device="mps")
  qpos_x0 = torch.zeros((1, 1), dtype=torch.float32, device="mps")
  cached_poses = stage._fk.run_device(qpos_x3)
  stage._fk.run_device = lambda *args, **kwargs: (_ for _ in ()).throw(
      AssertionError("cached gravcomp path recomputed FK"))
  actual = stage.gravcomp_device(qpos_x0, poses=cached_poses)

  cpu = mujoco.MjData(model)
  cpu.qpos[0] = np.pi / 2
  mujoco.mj_forward(model, cpu)
  cached = cpu.qfrc_gravcomp.copy()
  cpu.qpos[0] = 0.0
  mujoco.mj_forwardSkip(model, cpu, mujoco.mjtStage.mjSTAGE_POS, 0)
  np.testing.assert_allclose(actual.cpu().numpy()[0], cached,
                             rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native MPS joint gravcomp routing qualification")
def test_native_standalone_gravcomp_ignores_joint_routing_flags_and_retains_sleeping_world():
  import torch
  xml = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
    <body pos="0 0 1" gravcomp=".7"><joint name="h0" type="hinge"
      axis="0 1 0" stiffness="1.2" damping=".15"/>
      <geom type="sphere" size=".1" mass="1"/>
      <body pos=".2 0 0" gravcomp=".4"><joint name="h1" type="hinge"
        axis="1 0 0" stiffness=".8" damping=".12"/>
        <geom type="sphere" pos=".3 0 0" size=".1" mass=".5"/>
        <body pos=".3 0 0" gravcomp="1"><joint name="s2" type="slide"
          axis="0 0 1" stiffness=".6" damping=".08"/>
          <geom type="sphere" pos=".2 0 0" size=".1" mass=".3"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  model.jnt_actgravcomp[:] = [1, 0, 1]
  qpos = np.array([[.3, -.4, .2], [-.1, .2, .5]], dtype=np.float32)
  qvel = np.array([[.2, -.3, .1], [-.4, .1, .2]], dtype=np.float32)
  xfrc = np.zeros((2, model.nbody, 6), dtype=np.float32)
  xfrc[:, 1, :] = [.3, -.2, .5, .1, -.4, .2]
  xfrc[:, 2, :] = [-.1, .6, .2, -.3, .2, .4]
  xfrc[:, 3, :] = [.5, .1, -.2, .2, .3, -.1]
  stage = MetalPassiveForces(model, batch_size=2)
  qpos_mps = torch.tensor(qpos, dtype=torch.float32, device="mps")
  qvel_mps = torch.tensor(qvel, dtype=torch.float32, device="mps")
  xfrc_mps = torch.tensor(xfrc, dtype=torch.float32, device="mps")

  # Standalone output is the complete qfrc_gravcomp source, independent of
  # which joint DOFs later route it through qfrc_actuator.
  standalone = stage.gravcomp_device(qpos_mps)
  cpu_data = []
  for q, v in zip(qpos, qvel):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = v
    mujoco.mj_forward(model, data)
    cpu_data.append(data)
  expected_gravcomp = np.stack([data.qfrc_gravcomp for data in cpu_data])
  np.testing.assert_allclose(standalone.cpu().numpy(), expected_gravcomp,
                             rtol=0, atol=3e-5)

  # The normal passive accumulation applies actuator routing to gravity only;
  # applied body wrenches and spring/damper terms remain present on every DOF.
  passive = stage.run_device(qpos_mps, qvel_mps, xfrc_mps)
  expected_passive = []
  for data, applied in zip(cpu_data, xfrc):
    row = data.qfrc_passive.copy()
    for body in range(1, model.nbody):
      mujoco.mj_applyFT(model, data, applied[body, :3], applied[body, 3:],
                        data.xipos[body].copy(), body, row)
    expected_passive.append(row)
  np.testing.assert_allclose(passive.cpu().numpy(), expected_passive,
                             rtol=0, atol=3e-5)

  # world 1 has no awake trees: the standalone clear and projection must
  # preserve its prior gravcomp values, while world 0 refreshes all DOFs.
  stage._gravcomp_output[1].fill_(17.0)
  tree_count = int(model.ntree)
  tree_awake = np.ones((2, max(tree_count, 1)), dtype=np.int32)
  tree_awake[1].fill(0)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  body_ids = np.full((2, model.nbody), -1, dtype=np.int32)
  dof_ids = np.full((2, model.nv), -1, dtype=np.int32)
  counts = np.zeros((2, 3), dtype=np.int32)
  for world in range(2):
    awake = tree_awake[world]
    bodies = [body for body, tree in enumerate(body_tree)
              if tree < 0 or awake[tree] != 0]
    dofs = [dof for dof, tree in enumerate(dof_tree)
            if tree >= 0 and awake[tree] != 0]
    body_ids[world, :len(bodies)] = bodies
    dof_ids[world, :len(dofs)] = dofs
    counts[world] = [len(bodies), 0, len(dofs)]
  awake_lists = {
      "body_ids": torch.tensor(body_ids, dtype=torch.int32, device="mps"),
      "dof_ids": torch.tensor(dof_ids, dtype=torch.int32, device="mps"),
      "counts": torch.tensor(counts, dtype=torch.int32, device="mps"),
      "tree_awake": torch.tensor(tree_awake, dtype=torch.int32, device="mps"),
  }
  retained = stage.gravcomp_device(qpos_mps, awake_lists=awake_lists)
  np.testing.assert_allclose(retained[0].cpu().numpy(), expected_gravcomp[0],
                             rtol=0, atol=3e-5)
  np.testing.assert_array_equal(retained[1].cpu().numpy(),
                                np.full(model.nv, 17.0, dtype=np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_passive_kernel_matches_cpu_oracle():
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos = np.array([[.7, -.5], [-.3, .2]], dtype=np.float32)
  qvel = np.array([[1.2, -.8], [-.4, .5]], dtype=np.float32)
  import torch
  stage = MetalPassiveForces(model)
  actual = stage.run_device(torch.tensor(qpos, device="mps"), torch.tensor(qvel, device="mps"))
  np.testing.assert_allclose(actual.cpu().numpy(), _oracle(model, qpos, qvel), rtol=0, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1", reason="set MUJOCO_METAL_RUN_GPU=1 on an Apple GPU")
def test_native_passive_sleep_lists_use_per_world_count_columns():
  xml = """<mujoco><option><flag sleep="enable"/></option><worldbody>
    <body sleep="allowed"><joint type="hinge" stiffness="2" damping=".4"/>
      <geom type="sphere" size=".1"/></body>
    <body pos="1 0 0" sleep="allowed"><joint type="slide" axis="1 0 0" stiffness="1.5" damping=".2"/>
      <geom type="sphere" size=".1"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  import torch
  # The three rows have different active body and DOF counts, so indexing
  # strided [B,3] count views as if each column were contiguous is incorrect.
  tree_awake = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.int32)
  body_tree = np.asarray(model.body_treeid, dtype=np.int32)
  dof_tree = np.asarray(model.dof_treeid, dtype=np.int32)
  body_ids = np.full((3, model.nbody), -1, dtype=np.int32)
  dof_ids = np.full((3, model.nv), -1, dtype=np.int32)
  counts = np.zeros((3, 3), dtype=np.int32)
  for world in range(3):
    bodies = [b for b, tree in enumerate(body_tree)
              if tree < 0 or tree_awake[world, tree] != 0]
    dofs = [d for d, tree in enumerate(dof_tree)
            if tree >= 0 and tree_awake[world, tree] != 0]
    body_ids[world, :len(bodies)] = bodies
    dof_ids[world, :len(dofs)] = dofs
    counts[world] = [len(bodies), 0, len(dofs)]
  awake = {
      "body_ids": torch.as_tensor(body_ids, dtype=torch.int32, device="mps"),
      "dof_ids": torch.as_tensor(dof_ids, dtype=torch.int32, device="mps"),
      "counts": torch.as_tensor(counts, dtype=torch.int32, device="mps"),
      "tree_awake": torch.as_tensor(tree_awake, dtype=torch.int32, device="mps"),
  }
  seed_qpos = np.array([[.5, -.4], [-.2, .8], [.7, -.3]], dtype=np.float32)
  seed_qvel = np.array([[1.2, -.8], [-.4, .5], [.9, .6]], dtype=np.float32)
  qpos = seed_qpos + np.array([[.2, -.1], [-.3, .25], [.15, .4]], np.float32)
  qvel = seed_qvel + np.array([[-.3, .2], [.45, -.35], [-.6, .1]], np.float32)
  stage = MetalPassiveForces(model)
  # Seed persistent results while every tree is awake; asleep force entries
  # are retained by the pinned filtered CRB/RNE path.
  stage.run_device(torch.as_tensor(seed_qpos, device="mps"),
                   torch.as_tensor(seed_qvel, device="mps"))
  actual = stage.run_device(
      torch.as_tensor(qpos, device="mps"),
      torch.as_tensor(qvel, device="mps"), awake_lists=awake)
  expected = np.asarray([
      _pinned_sleep_passive(model, seed_qpos[w], seed_qvel[w], qpos[w],
                            qvel[w], tree_awake[w])
      for w in range(3)], dtype=np.float32)
  np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=2e-6)


def test_passive_gravcomp_is_filtered_by_joint_but_keeps_applied_wrench():
  xml = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
    <body name="flagged" gravcomp=".35"><joint name="z0" type="slide"
      axis="0 0 1" actuatorgravcomp="true"/>
      <geom type="sphere" size=".1" mass="2"/></body>
    <body name="ordinary" pos="1 0 0" gravcomp=".8">
      <joint name="z1" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".1" mass="1.5"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.array([[.2, -.1], [-.3, .4]], dtype=np.float64)
  qvel = np.zeros_like(qpos)
  wrench = np.zeros((2, model.nbody, 6), dtype=np.float64)
  wrench[:, 1, :3] = [.4, -.2, .7]
  wrench[:, 2, :3] = [-.3, .5, .2]
  wrench[:, 1, 3:] = [.2, .1, -.4]
  wrench[:, 2, 3:] = [-.1, .3, .25]
  actual = PassiveForceModel(model).force(qpos, qvel, wrench)
  expected = []
  for q, applied in zip(qpos, wrench):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    row = data.qfrc_passive.copy()
    for body in range(1, model.nbody):
      mujoco.mj_applyFT(model, data, applied[body, :3], applied[body, 3:],
                         data.xipos[body].copy(), body, row)
    expected.append(row)
  np.testing.assert_allclose(actual, np.asarray(expected), rtol=0, atol=2e-8)
  # Gravity compensation is omitted only from the flagged joint's passive
  # row; it remains in the model's independently published source vector.
  reference = mujoco.MjData(model)
  reference.qpos[:] = qpos[0]
  mujoco.mj_forward(model, reference)
  np.testing.assert_allclose(reference.qfrc_gravcomp, [2 * .35 * 9.81,
                                                       1.5 * .8 * 9.81],
                             rtol=0, atol=2e-8)
  assert abs(actual[0, 0]) > 0.1


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native MPS actuator gravcomp qualification")
@pytest.mark.parametrize("disableflags", [
    0,
    int(mujoco.mjtDisableBit.mjDSBL_GRAVITY),
    int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    | int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
])
def test_native_joint_actuator_gravcomp_routing_matches_forward(disableflags):
  from mujoco_metal.simulation import MetalSimulation

  xml = """<mujoco><option timestep=".002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="flagged" gravcomp=".35">
      <joint name="flagged_joint" type="slide" axis="0 0 1"
             actuatorgravcomp="true"/>
      <inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/>
    </body>
    <body name="ordinary" pos="1 0 0" gravcomp=".8">
      <joint name="ordinary_joint" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.5" diaginertia=".08 .08 .08"/>
    </body>
  </worldbody>
  <actuator><motor joint="flagged_joint" gear=".7"/>
    <motor joint="ordinary_joint" gear="-.4"/></actuator></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.disableflags = disableflags
  qpos = np.array([[.1, -.2], [-.3, .4]], dtype=np.float32)
  qvel = np.array([[.3, -.5], [-.2, .6]], dtype=np.float32)
  ctrl = np.array([[.8, -.7], [-.1, .4]], dtype=np.float32)
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_euler_v1")
  result = sim.forward_skip(ctrl=ctrl)
  actual_passive = result["velocity"]["qfrc_passive"].cpu().numpy()
  actual_actuator = result["actuation"]["qfrc_actuator"].cpu().numpy()
  expected_passive, expected_actuator = [], []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    data.ctrl[:] = ctrl[world]
    mujoco.mj_forward(model, data)
    expected_passive.append(data.qfrc_passive.copy())
    expected_actuator.append(data.qfrc_actuator.copy())
  np.testing.assert_allclose(actual_passive, expected_passive,
                             rtol=2e-5, atol=3e-5)
  np.testing.assert_allclose(actual_actuator, expected_actuator,
                             rtol=2e-5, atol=3e-5)


def _public_gravcomp_trajectory_model(profile, stateful, disableflags):
  integrator = {
      "integrated_euler_v1": "Euler",
      "integrated_rk4_v1": "RK4",
      "integrated_implicit_v1": "implicit",
  }[profile]
  if stateful:
    actuators = (
        '<general name="filtered" joint="flagged_joint" dyntype="filter" '
        'actdim="1" dynprm=".025" gaintype="fixed" gainprm=".7" '
        'biastype="affine" biasprm="0 .04 0" ctrllimited="true" '
        'ctrlrange="-1 1"/>'
        '<motor name="ordinary" joint="ordinary_joint" gear="-.4" '
        'ctrllimited="true" ctrlrange="-1 1"/>')
  else:
    actuators = (
        '<motor name="flagged" joint="flagged_joint" gear=".7" '
        'ctrllimited="true" ctrlrange="-1 1"/>'
        '<motor name="ordinary" joint="ordinary_joint" gear="-.4" '
        'ctrllimited="true" ctrlrange="-1 1"/>')
  xml = f'''<mujoco><option timestep=".002" gravity="0 0 -9.81"
      integrator="{integrator}"/><worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="2 2 .1"/>
    <body name="flagged" pos="0 0 .12" gravcomp=".35">
      <joint name="flagged_joint" type="slide" axis="0 0 1"
             actuatorgravcomp="true" stiffness=".3" damping=".08"/>
      <geom type="sphere" size=".1" mass="2"/>
    </body>
    <body name="ordinary" pos=".6 0 .12" gravcomp=".8">
      <joint name="ordinary_joint" type="slide" axis="0 0 1"
             stiffness=".2" damping=".05"/>
      <geom type="sphere" size=".1" mass="1.5"/>
    </body>
  </worldbody><equality><joint joint1="flagged_joint"
    joint2="ordinary_joint" polycoef="0 1 0 0 0"/></equality>
  <actuator>{actuators}</actuator></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  model.opt.disableflags = disableflags
  return model


@pytest.mark.parametrize("profile", [
    "integrated_euler_v1", "integrated_rk4_v1", "integrated_implicit_v1",
])
@pytest.mark.parametrize("stateful", [False, True], ids=["stateless", "filter"])
@pytest.mark.parametrize("disableflags", [
    0,
    int(mujoco.mjtDisableBit.mjDSBL_GRAVITY),
    int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    | int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
])
def test_public_gravcomp_trajectory_fixture_is_a_valid_pinned_cpu_oracle(
    profile, stateful, disableflags):
  model = _public_gravcomp_trajectory_model(profile, stateful, disableflags)
  assert model.jnt_actgravcomp.tolist() == [1, 0]
  np.testing.assert_allclose(model.body_gravcomp[1:], [.35, .8], rtol=0, atol=0)
  qpos = np.array([0., 0.])
  qvel = np.array([.08, .08])
  ctrl = np.array([.65, -.3])
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  if stateful:
    data.act[:] = [.15]
  data.ctrl[:] = ctrl
  mujoco.mj_forward(model, data)
  assert np.all(np.isfinite(data.qfrc_gravcomp))
  passive_pair = (int(mujoco.mjtDisableBit.mjDSBL_SPRING)
                  | int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
  gravity_or_passive_disabled = (
      bool(disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
      or (disableflags & passive_pair) == passive_pair)
  if gravity_or_passive_disabled:
    np.testing.assert_array_equal(data.qfrc_gravcomp,
                                  np.zeros(model.nv, dtype=data.qfrc_gravcomp.dtype))
  else:
    assert abs(float(data.qfrc_gravcomp[0])) > 1.0
  if not gravity_or_passive_disabled:
    assert abs(float(data.qfrc_actuator[0])) > .1
  mujoco.mj_step(model, data)
  assert np.all(np.isfinite(data.qpos))
  assert np.all(np.isfinite(data.qvel))


@pytest.mark.gpu
@pytest.mark.parametrize("profile", [
    "integrated_euler_v1", "integrated_rk4_v1", "integrated_implicit_v1",
])
@pytest.mark.parametrize("stateful", [False, True], ids=["stateless", "filter"])
@pytest.mark.parametrize("disableflags", [
    0,
    int(mujoco.mjtDisableBit.mjDSBL_GRAVITY),
    int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    | int(mujoco.mjtDisableBit.mjDSBL_DAMPER),
], ids=["normal", "gravity-disabled", "passive-disabled"])
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native public gravcomp trajectory gate")
def test_native_actuatorgravcomp_public_trajectory_cpu_reset_replay(
    profile, stateful, disableflags):
  """Exercise routing through contact/equality and actual integrator steps."""
  import torch
  from mujoco_metal.simulation import MetalSimulation

  model = _public_gravcomp_trajectory_model(profile, stateful, disableflags)
  qpos = np.array([[0., 0.], [.004, .004]], dtype=np.float32)
  qvel = np.array([[.08, .08], [-.035, -.035]], dtype=np.float32)
  act = np.array([[.15], [-.2]], dtype=np.float32) if stateful else None
  controls = [
      np.array([[.65, -.3], [-.4, .55]], dtype=np.float32),
      np.array([[-.2, .45], [.7, -.15]], dtype=np.float32),
      np.array([[.3, .1], [-.5, .25]], dtype=np.float32),
  ]
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile=profile)
  sim.reset(qpos=qpos, qvel=qvel, act=act)
  cpu = []
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    if stateful:
      data.act[:] = act[world]
    cpu.append(data)
  for control in controls:
    status = sim.step(ctrl=control)
    np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
    for world, data in enumerate(cpu):
      data.ctrl[:] = control[world]
      mujoco.mj_step(model, data)
    for field, cpu_field in (("_qpos", "qpos"), ("_qvel", "qvel"),
                             ("_qacc", "qacc")):
      actual = getattr(sim._state, field).detach().cpu().numpy()
      expected = np.stack([getattr(data, cpu_field) for data in cpu])
      np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-5,
                                 err_msg=f"{profile}/{stateful}/{cpu_field}")
    if stateful:
      actual_act = sim._state._act.detach().cpu().numpy()
      expected_act = np.stack([data.act for data in cpu])
      np.testing.assert_allclose(actual_act, expected_act, rtol=3e-5,
                                 atol=4e-5)

  checkpoint = sim.snapshot()
  replay_control = controls[-1]
  sim.step(ctrl=replay_control)
  target = {}
  for name in ("_qpos", "_qvel", "_qacc", "_act"):
    value = getattr(sim._state, name, None)
    if value is not None:
      target[name] = value.clone()
  sim.restore(checkpoint)
  status = sim.step(ctrl=replay_control)
  np.testing.assert_array_equal(status.cpu().numpy(), [0, 0])
  for name, expected in target.items():
    torch.testing.assert_close(getattr(sim._state, name), expected,
                               rtol=0, atol=0, msg=name)

  # Reset one world and prove the other world's accepted trajectory is kept.
  retained = sim._state._qpos[0].clone()
  sim.reset(env_ids=[1], qpos=qpos[1:2], qvel=qvel[1:2],
            act=None if act is None else act[1:2])
  torch.testing.assert_close(sim._state._qpos[0], retained, rtol=0, atol=0)
  torch.testing.assert_close(sim._state._qpos[1], torch.as_tensor(
      qpos[1], dtype=torch.float32, device="mps"), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native MPS flagged-wrench qualification")
def test_native_flagged_joint_keeps_applied_body_wrench_in_passive_force():
  import torch
  xml = """<mujoco><option gravity="0 0 -9.81"/><worldbody>
    <body gravcomp=".6"><joint name="flagged" type="slide" axis="0 0 1"
       actuatorgravcomp="true"/>
      <inertial pos=".2 0 0" mass="2" diaginertia=".1 .1 .1"/></body>
    <body pos="1 0 0" gravcomp=".4"><joint name="ordinary" type="slide"
       axis="0 0 1"/><inertial pos="0 0 0" mass="1.5"
       diaginertia=".1 .1 .1"/></body>
  </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.zeros((2, model.nq), dtype=np.float32)
  qvel = np.zeros((2, model.nv), dtype=np.float32)
  wrench = np.zeros((2, model.nbody, 6), dtype=np.float32)
  wrench[:, 1, :] = [.4, -.2, .1, .3, .5, -.4]
  wrench[:, 2, :] = [-.3, .5, .2, -.1, .3, .25]
  stage = MetalPassiveForces(model, batch_size=2)
  actual = stage.run_device(
      torch.as_tensor(qpos, device="mps"), torch.as_tensor(qvel, device="mps"),
      torch.as_tensor(wrench, device="mps"))
  expected = []
  for q, applied in zip(qpos, wrench):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    row = data.qfrc_passive.copy()
    for body in range(1, model.nbody):
      mujoco.mj_applyFT(model, data, applied[body, :3], applied[body, 3:],
                         data.xipos[body].copy(), body, row)
    expected.append(row)
  np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=3e-5)
