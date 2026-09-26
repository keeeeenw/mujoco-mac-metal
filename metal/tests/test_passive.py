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
  np.testing.assert_allclose(PassiveForceModel(model).force(qpos, qvel), _oracle(model, qpos, qvel), rtol=0, atol=1e-7)


def test_disable_flags_and_input_validation():
  model = mujoco.MjModel.from_xml_string(_XML)
  model.opt.disableflags = int(mujoco.mjtDisableBit.mjDSBL_SPRING) | int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  stage = PassiveForceModel(model)
  result = stage.force(np.zeros((2, 2)), np.ones((2, 2)))
  np.testing.assert_array_equal(result, np.zeros((2, 2)))
  with pytest.raises(ValueError, match="shapes"):
    stage.force(np.zeros((2, 3)), np.zeros((2, 2)))


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
