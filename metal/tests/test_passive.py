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
def test_native_passive_kernel_matches_cpu_oracle():
  model = mujoco.MjModel.from_xml_string(_XML)
  qpos = np.array([[.7, -.5], [-.3, .2]], dtype=np.float32)
  qvel = np.array([[1.2, -.8], [-.4, .5]], dtype=np.float32)
  import torch
  stage = MetalPassiveForces(model)
  actual = stage.run_device(torch.tensor(qpos, device="mps"), torch.tensor(qvel, device="mps"))
  np.testing.assert_allclose(actual.cpu().numpy(), _oracle(model, qpos, qvel), rtol=0, atol=2e-6)
