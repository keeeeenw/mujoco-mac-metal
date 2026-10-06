# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""High-DOF tendon row regressions for scalable coupled assembly."""

import os

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.capacity import CapacityLimits
from mujoco_metal.spatial_tendons import SpatialTendonModel


def _model_xml(nv=70):
  chain = ""
  for i in range(nv):
    chain += (
        f'<body name="link{i}" pos="0 .01 0">'
        f'<joint name="j{i}" type="slide" axis="1 0 0"/>'
        '<inertial pos="0 0 0" mass=".2" '
        'diaginertia=".001 .001 .001"/>'
        '<geom type="sphere" size=".02" mass=".01" '
        'contype="0" conaffinity="0"/>'
        + ('<site name="end" pos=".3 0 0"/>' if i == nv - 1 else ''))
  chain += '</body>' * nv
  fixed_joints = ''.join(
      f'<joint joint="j{i}" coef="{1.0 / nv:.9g}"/>' for i in range(nv))
  return f'''<mujoco>
    <option timestep=".0002" gravity="0 0 0" solver="PGS" iterations="2">
      <flag contact="disable"/>
    </option>
    <worldbody><site name="anchor" pos="0 0 0"/>{chain}</worldbody>
    <tendon>
      <spatial name="spatial" frictionloss=".1">
        <site site="anchor"/><site site="end"/>
      </spatial>
      <fixed name="fixed" frictionloss=".2">{fixed_joints}</fixed>
    </tendon>
    <equality><tendon name="couple" tendon1="spatial" tendon2="fixed"
      polycoef="0 1 0 0 0"/></equality>
  </mujoco>'''


def _finite_difference_tendon_jac(model, qpos, tendon_id, eps=2e-4):
  result = np.empty(model.nv, dtype=np.float64)
  for dof in range(model.nv):
    plus = mujoco.MjData(model)
    minus = mujoco.MjData(model)
    plus.qpos[:] = qpos
    minus.qpos[:] = qpos
    plus.qpos[dof] += eps
    minus.qpos[dof] -= eps
    mujoco.mj_forward(model, plus)
    mujoco.mj_forward(model, minus)
    result[dof] = (plus.ten_length[tendon_id] - minus.ten_length[tendon_id]) / (2 * eps)
  return result


def test_high_dof_fixed_and_spatial_tendon_rows_lower_without_thread_caps():
  model = mujoco.MjModel.from_xml_string(_model_xml())
  assert model.nv == 70
  assert model.ntendon == 2
  desc = lower_coupled_constraints(
      model, limits=CapacityLimits(max_nv=70, max_rows=512,
                                   max_pairs=64, max_slots=64))
  spatial = SpatialTendonModel(model)
  assert desc.nv == model.nv
  assert desc.ten_friction_rows == 2
  assert spatial.nv == model.nv
  assert spatial.ntendon == model.ntendon
  assert spatial.paths[0] is not None
  assert np.count_nonzero(np.abs(_finite_difference_tendon_jac(
      model, np.zeros(model.nq), 0)[32:]) > 1e-5) > 20
  fixed_jac = _finite_difference_tendon_jac(model, np.zeros(model.nq), 1)
  assert np.count_nonzero(np.abs(fixed_jac[32:]) > 1e-5) == 38
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert np.all(np.isfinite(data.qacc))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="requires an explicitly serialized native GPU slot")
def test_high_dof_fixed_and_spatial_tendon_rows_match_pinned_jacobians():
  from mujoco_metal.simulation import MetalSimulation

  model = mujoco.MjModel.from_xml_string(_model_xml())
  assert model.nv == 70
  qpos = np.zeros((2, model.nq), dtype=np.float32)
  qpos[0, :] = .002
  qpos[1, :] = -.001
  qvel = np.zeros_like(qpos)
  qvel[0, :] = .03
  qvel[1, :] = -.02
  sim = MetalSimulation(model, batch_size=2, qpos=qpos, qvel=qvel,
                        profile="integrated_scalable_v1",
                        limits=CapacityLimits(max_nv=70, max_rows=512,
                                              max_pairs=64, max_slots=64))
  assembled = sim.assembled_system(recompute=True)
  rows = assembled["J"].detach().cpu().numpy()
  desc = sim._coupled_constraints.descriptor
  assert rows.shape[2] == 70
  assert desc.ten_friction_rows == 2
  for world in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    for tendon_id in range(2):
      expected = _finite_difference_tendon_jac(model, qpos[world], tendon_id)
      row = int(desc.ten_base + tendon_id)
      np.testing.assert_allclose(rows[world, row], expected,
                                 rtol=3e-3, atol=3e-3)
      assert np.any(np.abs(rows[world, row, 32:]) > 1e-5)
    eqrow = int(desc.eq_rowadr[0])
    np.testing.assert_allclose(rows[world, eqrow],
                               _finite_difference_tendon_jac(
                                   model, qpos[world], 0)
                               - _finite_difference_tendon_jac(
                                   model, qpos[world], 1),
                               rtol=4e-3, atol=4e-3)
