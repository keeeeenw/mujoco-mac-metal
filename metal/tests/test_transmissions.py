"""MuJoCo 3.10 CPU oracles and opt-in native checks for transmissions."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.transmissions import MetalTransmissions
from mujoco_metal.transmissions import TransmissionModel


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
