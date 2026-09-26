# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU contract and opt-in MPS checks for the isolated scalar motor stage."""

import os
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from mujoco_metal.actuation import MetalScalarMotorForce
from mujoco_metal.actuation import ScalarMotorModel

_XML = """<mujoco>
  <option timestep=".002"/>
  <worldbody>
    <body><joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size=".08 .2"/></body>
    <body pos="1 0 0"><joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .1"/></body>
  </worldbody>
  <actuator>
    <general name="positive" joint="hinge" gear="2" dyntype="none"
      gaintype="fixed" biastype="none" gainprm="2" ctrllimited="true"
      ctrlrange="-1 1" forcelimited="true" forcerange="-3 3"
      group="0"/>
    <general name="negative" joint="hinge" gear="-0.5" dyntype="none"
      gaintype="fixed" biastype="none" gainprm="1.5" group="1"/>
    <general name="slide" joint="slide" gear="1.25" dyntype="none"
      gaintype="fixed" biastype="none" gainprm=".75"
      ctrllimited="true" ctrlrange="-2 2" group="2"/>
  </actuator>
</mujoco>"""


def _compiled(xml=_XML):
  return mujoco.MjModel.from_xml_string(xml)


def _cpu_oracle(model, ctrl):
  result = []
  for controls in ctrl:
    data = mujoco.MjData(model)
    data.ctrl[:] = controls
    mujoco.mj_forward(model, data)
    result.append(data.qfrc_actuator.copy())
  return np.asarray(result)


def test_scalar_motor_lowering_matches_mujoco_clipping_and_summed_moments():
  model = _compiled()
  controls = np.array(
      [
          [4.0, 2.0, -3.0],
          [-4.0, -6.0, 1.0],
          [0.5, -0.25, 2.5],
      ]
  )
  lowered = ScalarMotorModel.from_model(model)
  actual = lowered.generalized_force(controls)
  expected = _cpu_oracle(model, controls)
  np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
  assert lowered.dof.tolist() == [0, 0, 1]
  assert not lowered.dof.flags.writeable
  assert not lowered.gear.flags.writeable


def test_control_clamp_disable_group_disable_and_global_disable_match_mujoco():
  controls = np.array([[4.0, 2.0, -3.0], [-4.0, -6.0, 1.0]])
  for mutation in (
      lambda model: setattr(
          model.opt,
          "disableflags",
          int(model.opt.disableflags)
          | int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL),
      ),
      lambda model: setattr(model.opt, "disableactuator", 1 << 1),
      lambda model: setattr(
          model.opt,
          "disableflags",
          int(model.opt.disableflags)
          | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION),
      ),
  ):
    model = _compiled()
    mutation(model)
    lowered = ScalarMotorModel.from_model(model)
    np.testing.assert_allclose(
        lowered.generalized_force(controls),
        _cpu_oracle(model, controls),
        rtol=0,
        atol=1e-7,
    )


def test_empty_actuator_set_has_explicit_empty_shapes():
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><body><joint type="hinge"/>'
      '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
  )
  lowered = ScalarMotorModel.from_model(model)
  result = lowered.generalized_force(np.empty((2, 0)))
  assert result.shape == (2, model.nv)
  np.testing.assert_array_equal(result, np.zeros((2, model.nv)))


def test_public_metadata_constructor_detaches_arrays_and_checks_bounds():
  lowered = ScalarMotorModel.from_model(_compiled())
  source = np.array(lowered.gear, copy=True)
  copied = replace(lowered, gear=source)
  source[0] = 17
  assert copied.gear[0] != 17
  assert not copied.gear.flags.writeable
  with pytest.raises(ValueError, match="dof indices"):
    replace(lowered, dof=np.array([-1, 0, 1], dtype=np.int32))
  with pytest.raises(ValueError, match="integer storage range"):
    replace(lowered, dof=np.array([2**32, 0, 1], dtype=np.int64))
  with pytest.raises(ValueError, match="int32 dimension"):
    replace(lowered, nv=True)


@pytest.mark.parametrize(
    "xml, pattern",
    [
        (
            _XML.replace(
                'joint="hinge" gear="2"',
                'joint="hinge" gear="2" ' 'armature=".1"',
            ),
            "armature",
        ),
        (
            _XML.replace(
                'joint="hinge" gear="2"',
                'joint="hinge" gear="2" ' 'damping=".1"',
            ),
            "damping",
        ),
        (
            """<mujoco><worldbody><body><joint name="h" type="hinge"/>
              <geom type="capsule" size=".05 .1"/></body></worldbody>
              <actuator><general joint="h" dyntype="none"
                gaintype="affine" biastype="none" gainprm="1 0 0"/>
              </actuator></mujoco>""",
            "fixed-gain",
        ),
    ],
)
def test_unsupported_models_fail_during_lowering(xml, pattern):
  with pytest.raises(ValueError, match=pattern):
    ScalarMotorModel.from_model(_compiled(xml))


def test_tendon_and_non_scalar_joint_transmissions_are_rejected():
  tendon = """<mujoco><worldbody><body><joint name="h" type="hinge"/>
    <geom type="capsule" size=".05 .2"/><site name="a"/>
    <site name="b" pos="0 0 .2"/></body></worldbody>
    <tendon><spatial name="t"><site site="a"/><site site="b"/></spatial></tendon>
    <actuator><motor tendon="t"/></actuator></mujoco>"""
  with pytest.raises(ValueError, match="fixed-gain"):
    ScalarMotorModel.from_model(_compiled(tendon))

  ball = """<mujoco><worldbody><body><joint type="ball"/>
    <geom type="sphere" size=".1"/></body></worldbody>
    <actuator><motor joint="0"/></actuator></mujoco>"""
  # A named joint keeps the fixture robust against compiler-assigned names.
  ball = ball.replace('<joint type="ball"/>', '<joint name="b" type="ball"/>')
  ball = ball.replace('joint="0"', 'joint="b"')
  with pytest.raises(ValueError, match="hinge or slide"):
    ScalarMotorModel.from_model(_compiled(ball))


@pytest.mark.parametrize(
    "controls, message",
    [
        (np.zeros(3), "shape"),
        (np.zeros((0, 3)), "batch > 0"),
        (np.array([[np.nan, 0, 0]]), "finite"),
        (np.array([[np.inf, 0, 0]]), "finite"),
        (np.array([[1e100, 0, 0]]), "finite float32"),
    ],
)
def test_cpu_primitive_rejects_malformed_or_unrepresentable_controls(
    controls, message
):
  with pytest.raises(ValueError, match=message):
    ScalarMotorModel.from_model(_compiled()).generalized_force(controls)


def test_lowering_rejects_actuator_level_and_joint_level_force_modifiers():
  actuator_armature = _compiled()
  actuator_armature.actuator_armature[0] = 0.1
  with pytest.raises(ValueError, match="armature"):
    ScalarMotorModel.from_model(actuator_armature)

  actuator_damping = _compiled()
  actuator_damping.actuator_damping[0] = 0.1
  with pytest.raises(ValueError, match="damping"):
    ScalarMotorModel.from_model(actuator_damping)

  joint_force_limit = _compiled()
  joint_force_limit.jnt_actfrclimited[0] = 1
  with pytest.raises(ValueError, match="joint-level"):
    ScalarMotorModel.from_model(joint_force_limit)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_native_scalar_motor_force_matches_mujoco_and_tracks_controls():
  import torch

  model = _compiled()
  lowered = ScalarMotorModel.from_model(model)
  controls = np.array([[4.0, 2.0, -3.0], [-4.0, -6.0, 1.0]])
  stage = MetalScalarMotorForce(lowered, batch_size=2)
  device_ctrl = torch.tensor(controls, dtype=torch.float32, device="mps")
  output = stage.run_device(device_ctrl)
  np.testing.assert_allclose(
      output.cpu().numpy(), _cpu_oracle(model, controls), rtol=2e-6, atol=2e-6
  )

  device_ctrl[0, 0] = -0.5
  updated = stage.run_device(device_ctrl)
  np.testing.assert_allclose(
      updated.cpu().numpy(),
      _cpu_oracle(model, device_ctrl.cpu().numpy()),
      rtol=2e-6,
      atol=2e-6,
  )
  with pytest.raises(ValueError, match="batch"):
    stage.run_device(
        torch.zeros((1, model.nu), dtype=torch.float32, device="mps")
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
@pytest.mark.parametrize("disable", ["clampctrl", "group", "actuation"])
def test_native_motor_disable_flags_match_mujoco(disable):
  import torch

  model = _compiled()
  if disable == "clampctrl":
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
  elif disable == "group":
    model.opt.disableactuator |= 1 << 1
  else:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  controls = np.array([[4.0, 2.0, -3.0], [-4.0, -6.0, 1.0]])
  stage = MetalScalarMotorForce(
      ScalarMotorModel.from_model(model), batch_size=2
  )
  actual = (
      stage.run_device(
          torch.tensor(controls, dtype=torch.float32, device="mps")
      )
      .cpu()
      .numpy()
  )
  np.testing.assert_allclose(
      actual, _cpu_oracle(model, controls), rtol=2e-6, atol=2e-6
  )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
    reason="requires explicit MUJOCO_METAL_RUN_GPU=1 and idle GPU",
)
def test_native_motor_empty_actuators_and_nonfinite_controls_are_explicit():
  import torch

  no_actuator = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><body><joint type="hinge"/>'
      '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
  )
  empty_stage = MetalScalarMotorForce(
      ScalarMotorModel.from_model(no_actuator), batch_size=2
  )
  empty = empty_stage.run_device(
      torch.empty((2, 0), dtype=torch.float32, device="mps")
  )
  assert tuple(empty.shape) == (2, no_actuator.nv)
  np.testing.assert_array_equal(
      empty.cpu().numpy(), np.zeros((2, no_actuator.nv))
  )

  model = _compiled()
  stage = MetalScalarMotorForce(
      ScalarMotorModel.from_model(model), batch_size=2
  )
  controls = torch.zeros((2, model.nu), dtype=torch.float32, device="mps")
  controls[1, 1] = float("nan")
  result = stage.run_device(controls).cpu().numpy()
  assert np.all(np.isfinite(result[0]))
  assert np.all(np.isnan(result[1]))

  controls.zero_()
  controls[0, 1] = 3e38
  overflow = stage.run_device(controls).cpu().numpy()
  assert not np.all(np.isfinite(overflow[0]))
  assert np.all(np.isfinite(overflow[1]))
