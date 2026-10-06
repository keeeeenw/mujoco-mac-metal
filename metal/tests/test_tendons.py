# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

import os

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


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_fixed_tendon_can_leave_armature_to_component_projection():
  import torch
  from mujoco_metal.tendons import MetalFixedTendonDynamics

  model = mujoco.MjModel.from_xml_string(_xml())
  qpos_np = np.asarray([[.27, .31]], dtype=np.float32)
  qvel_np = np.asarray([[.5, -.2]], dtype=np.float32)
  args = (torch.as_tensor(qpos_np, device="mps"),
          torch.as_tensor(qvel_np, device="mps"))
  dense = MetalFixedTendonDynamics(model, batch_size=1)
  external = MetalFixedTendonDynamics(model, batch_size=1,
                                      armature_storage="external")
  dense_values = dense.run_device(*args)
  external_values = external.run_device(*args)
  assert external_values[2] is None
  assert external._armature_matrix.numel() == 1
  torch.testing.assert_close(external_values[0], dense_values[0],
                             rtol=0, atol=0)
  torch.testing.assert_close(external_values[1], dense_values[1],
                             rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_fixed_tendon_damping_writes_compiled_coo_directly():
  import torch
  from mujoco_metal.tendons import MetalFixedTendonDynamics
  from mujoco_metal.velocity_derivative import (
      MetalVelocityDerivativeValues, compile_velocity_derivative_layout)

  model = mujoco.MjModel.from_xml_string(_xml())
  model.tendon_dampingpoly[0] = [.1, .15]
  layout = compile_velocity_derivative_layout(model)
  program = MetalFixedTendonDynamics(
      model, batch_size=2, armature_storage="external",
      velocity_derivative_layout=layout)
  qpos_np = np.asarray([[.27, .31], [-.13, .43]], dtype=np.float32)
  qvel_np = np.asarray([[.5, -.2], [-.3, .7]], dtype=np.float32)
  qpos = torch.as_tensor(qpos_np, device="mps")
  qvel = torch.as_tensor(qvel_np, device="mps")
  _, dense_tangent, _ = program.run_device(qpos, qvel)
  values = torch.zeros((2, max(layout.edge_count, 1)),
                       dtype=torch.float32, device="mps")
  writer = MetalVelocityDerivativeValues(layout, 2, values)
  program.run_damping_derivative_coo_device(qvel, writer)
  torch.mps.synchronize()
  expected = np.zeros((2, max(layout.edge_count, 1)), dtype=np.float32)
  expected[:, :layout.edge_count] = -dense_tangent.cpu().numpy()[
      :, layout.edge_rows, layout.edge_cols]
  np.testing.assert_allclose(values.cpu().numpy(), expected,
                             rtol=3e-6, atol=3e-6)


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


def test_polynomial_spring_and_damper_match_mujoco_tangent():
  model = mujoco.MjModel.from_xml_string(_xml())
  model.tendon_stiffnesspoly[0] = [.3, .2]
  model.tendon_dampingpoly[0] = [.1, .15]
  stage = FixedTendonModel(model)
  qpos = np.array([[.27, .31]])
  qvel = np.array([[.5, -.2]])
  force, tangent, _ = stage.run(qpos, qvel)
  data = _oracle(model, qpos[0], qvel[0])
  np.testing.assert_allclose(force[0], data.qfrc_passive, rtol=3e-8, atol=3e-8)
  eps = 1e-6
  columns = []
  for axis in range(model.nv):
    plus, minus = qvel[0].copy(), qvel[0].copy()
    plus[axis] += eps
    minus[axis] -= eps
    columns.append(-(_oracle(model, qpos[0], plus).qfrc_passive -
                     _oracle(model, qpos[0], minus).qfrc_passive) / (2 * eps))
  np.testing.assert_allclose(tangent[0], np.column_stack(columns), rtol=3e-8, atol=3e-8)


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


def test_tendon_armature_crba_sparsity_matches_mujoco_fullM():
  """Independently verifies MuJoCo CRBA mass matrix sparsity against FixedTendonModel."""
  xml = """<mujoco model="tree_tendon">
    <worldbody>
      <!-- Branch A -->
      <body name="a1" pos="0 0 1">
        <joint name="ja1" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="a2" pos="0.5 0 0">
          <joint name="ja2" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </body>
      <!-- Branch B -->
      <body name="b1" pos="0 1 1">
        <joint name="jb1" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="b2" pos="0.5 0 0">
          <joint name="jb2" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </body>
    </worldbody>
    <tendon>
      <!-- Cross-branch tendon between disjoint branches -->
      <fixed name="t_cross" armature="0.2">
        <joint joint="ja1" coef="1.0"/>
        <joint joint="jb1" coef="0.5"/>
      </fixed>
      <!-- Ancestor-chain tendon on Branch A -->
      <fixed name="t_chain" armature="0.3">
        <joint joint="ja1" coef="0.4"/>
        <joint joint="ja2" coef="0.8"/>
      </fixed>
    </tendon>
  </mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  M_with = np.zeros((m.nv, m.nv))
  mujoco.mj_fullM(m, d, M_with)

  xml_no_arm = xml.replace('armature="0.2"', '').replace('armature="0.3"', '')
  m_no = mujoco.MjModel.from_xml_string(xml_no_arm)
  d_no = mujoco.MjData(m_no)
  mujoco.mj_forward(m_no, d_no)
  M_no = np.zeros((m.nv, m.nv))
  mujoco.mj_fullM(m_no, d_no, M_no)

  # Direct MuJoCo ground truth difference
  delta_M_mujoco = M_with - M_no

  # Independent assert 1: MuJoCo CRBA tree representation explicitly zeros disjoint cross-branch coupling
  # ja1 is DOF 0, jb1 is DOF 2. Naive outer product would have 0.2 * 1.0 * 0.5 = 0.10.
  assert delta_M_mujoco[0, 2] == 0.0, "MuJoCo must drop cross-branch armature coupling"
  assert delta_M_mujoco[2, 0] == 0.0, "MuJoCo must drop cross-branch armature coupling"

  # Independent assert 2: MuJoCo preserves ancestor-descendant chain coupling
  # ja1 is DOF 0, ja2 is DOF 1. Expected armature: 0.3 * 0.4 * 0.8 = 0.096.
  assert np.isclose(delta_M_mujoco[0, 1], 0.096, atol=1e-6)
  assert np.isclose(delta_M_mujoco[1, 0], 0.096, atol=1e-6)

  # Independent assert 3: FixedTendonModel matches MuJoCo delta_M across the entire matrix
  tendon_model = FixedTendonModel(m)
  _, _, armature_matrix = tendon_model.run(np.zeros((1, m.nq)), np.zeros((1, m.nv)))
  np.testing.assert_allclose(armature_matrix, delta_M_mujoco, atol=1e-7)
