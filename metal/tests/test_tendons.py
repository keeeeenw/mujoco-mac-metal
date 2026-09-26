# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

import mujoco
import numpy as np
import pytest

from mujoco_metal.tendons import FixedTendonModel


def _xml(extra_tendon=''):
  return f'''<mujoco model="fixed_tendon_test">
    <compiler angle="radian"/><option gravity="0 0 0"/>
    <default><joint limited="false" damping="0"/></default>
    <worldbody><body pos="0 0 1"><joint name="x" type="slide" axis="1 0 0" ref=".1"/>
      <geom type="sphere" size=".1" mass="1"/>
      <body><joint name="y" type="hinge" axis="0 0 1" ref="-.2"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body></body></worldbody>
    <tendon><fixed name="cable" stiffness="3" damping=".7" springlength="-.1 .2" armature=".4">
      <joint joint="x" coef="1.5"/><joint joint="y" coef="-.75"/>
    </fixed>{extra_tendon}</tendon>
  </mujoco>'''


def _oracle(model, qpos, qvel):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  return data


def test_fixed_tendon_force_tangent_and_armature_match_mujoco():
  model = mujoco.MjModel.from_xml_string(_xml())
  stage = FixedTendonModel(model)
  qpos = np.array([[.27, .31], [-.36, -.42]])
  qvel = np.array([[.5, -.2], [-.17, .41]])
  force, tangent, armature = stage.run(qpos, qvel)

  for batch in range(len(qpos)):
    data = _oracle(model, qpos[batch], qvel[batch])
    np.testing.assert_allclose(force[batch], data.qfrc_passive, rtol=3e-8, atol=3e-8)
    eps = 1e-6
    columns = []
    for axis in range(model.nv):
      plus, minus = qvel[batch].copy(), qvel[batch].copy()
      plus[axis] += eps
      minus[axis] -= eps
      columns.append(-(_oracle(model, qpos[batch], plus).qfrc_passive -
                       _oracle(model, qpos[batch], minus).qfrc_passive) / (2 * eps))
    np.testing.assert_allclose(tangent[batch], np.column_stack(columns), rtol=3e-8, atol=3e-8)

  arm_model = mujoco.MjModel.from_xml_string(_xml().replace(' armature=".4"', ' armature="0"'))
  data_with = _oracle(model, qpos[0], qvel[0])
  data_without = _oracle(arm_model, qpos[0], qvel[0])
  mass_with, mass_without = np.zeros((2, 2)), np.zeros((2, 2))
  mujoco.mj_fullM(model, data_with, mass_with)
  mujoco.mj_fullM(arm_model, data_without, mass_without)
  np.testing.assert_allclose(armature, mass_with - mass_without, rtol=2e-8, atol=2e-8)


def test_disable_bits_and_raw_qpos_reference_are_respected():
  xml = _xml()
  model = mujoco.MjModel.from_xml_string(xml)
  data = _oracle(model, [.1, -.2], [0, 0])
  # A nonzero joint ref does not shift fixed-tendon length: source uses raw qpos.
  assert abs(data.ten_length[0] - (.1 * 1.5 + -.2 * -.75)) < 1e-12
  spring = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  damper = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  model.opt.disableflags = spring | damper
  force, tangent, _ = FixedTendonModel(model).run([[.3, -.4]], [[.7, -.5]])
  np.testing.assert_array_equal(force, np.zeros((1, model.nv)))
  np.testing.assert_array_equal(tangent, np.zeros((1, model.nv, model.nv)))


def test_spatial_tendons_are_rejected():
  xml = _xml().replace('</worldbody>',
      '<site name="a" pos="0 0 0"/><site name="b" pos=".1 0 0"/></worldbody>')
  xml = xml.replace('</tendon>', '<spatial name="s"><site site="a"/><site site="b"/></spatial></tendon>')
  with pytest.raises(ValueError, match='joint wraps'):
    FixedTendonModel(mujoco.MjModel.from_xml_string(xml))


def test_tendon_limits_are_rejected():
  xml = _xml().replace('springlength="-.1 .2"', 'springlength="-.1 .2" limited="true" range="-.5 .5"')
  with pytest.raises(ValueError, match='limits'):
    FixedTendonModel(mujoco.MjModel.from_xml_string(xml))


def test_state_shape_and_finiteness_are_checked():
  model = mujoco.MjModel.from_xml_string(_xml())
  stage = FixedTendonModel(model)
  with pytest.raises(ValueError, match='shapes'):
    stage.run(np.zeros(2), np.zeros((1, 2)))
  with pytest.raises(ValueError, match='finite'):
    stage.run([[np.nan, 0]], [[0, 0]])
