# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Regression checks for the pinned MuJoCo 3.10 compiled flex material law."""

import mujoco
import numpy as np
import pytest
import torch

from mujoco_metal.flex import MetalFlex, lower_flex_descriptor


def _model(dim):
  if dim == 2:
    count = "2 2 1"
    elasticity = '<elasticity young="1000" poisson="0.2" damping="0.4" thickness="0.01" elastic2d="stretch"/>'
  else:
    count = "2 2 2"
    elasticity = '<elasticity young="1000" poisson="0.2" damping="0.4"/>'
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" timestep="0.002"/><worldbody>
      <flexcomp name="material" type="grid" count="{count}"
                spacing="0.1 0.1 0.1" mass="1" radius="0.01" dim="{dim}">
        <contact contype="0" conaffinity="0"/>
        <edge stiffness="0" damping="0"/>
        {elasticity}
      </flexcomp>
    </worldbody></mujoco>
  """)


def _evaluate(model, qpos, qvel):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, device="cpu")
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
  }
  force, damping, stiffness = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(qvel[None], dtype=torch.float32), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32))
  return data, flex, force.detach().numpy()[0], damping.detach().numpy()[0], stiffness


@pytest.mark.parametrize("dim", [2, 3])
def test_compiled_simplex_metric_matches_pinned_cpu(dim):
  model = _model(dim)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qvel = np.zeros(model.nv, dtype=np.float64)
  qpos[3] += 0.02
  qpos[model.nq - 3] -= 0.013
  qvel[0] = 0.17
  qvel[-1] = -0.11

  data, flex, force, _, _ = _evaluate(model, qpos, qvel)
  assert np.max(np.abs(data.qfrc_passive)) > 1e-5
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=3e-4, atol=4e-5)
  assert flex.descriptor.stiffness.size == model.flex_stiffness.size
  np.testing.assert_array_equal(flex.descriptor.stiffness, model.flex_stiffness.astype(np.float32))
  assert not np.any(flex.descriptor.young)


def test_material_rayleigh_velocity_tangent_matches_finite_difference():
  model = _model(3)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.025
  qvel = np.zeros(model.nv, dtype=np.float64)
  qvel[0] = 0.2
  data, flex, force, damping_tangent, _ = _evaluate(model, qpos, qvel)
  assert np.any(model.flex_damping != 0)

  dof = 0
  eps = 2e-4
  qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
  qvel_hi[dof] += eps
  qvel_lo[dof] -= eps
  _, _, force_hi, _, _ = _evaluate(model, qpos, qvel_hi)
  _, _, force_lo, _, _ = _evaluate(model, qpos, qvel_lo)
  finite_difference = (force_hi - force_lo) / (2 * eps)
  np.testing.assert_allclose(
      damping_tangent[:, dof], finite_difference, rtol=3e-2, atol=2e-2)

  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[dof] += eps
  qpos_lo[dof] -= eps
  _, _, force_hi, _, _ = _evaluate(model, qpos_hi, qvel)
  _, _, force_lo, _, _ = _evaluate(model, qpos_lo, qvel)
  position_difference = (force_hi - force_lo) / (2 * eps)
  analytic = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(qvel[None], dtype=torch.float32),
      {"body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
       "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
       "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
       "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32)},
      torch.tensor(data.cvel[None], dtype=torch.float32))[2]
  np.testing.assert_allclose(
      analytic.detach().numpy()[0, :, dof], position_difference,
      rtol=3e-2, atol=2e-2)


def test_compiled_triangle_shell_bending_matches_pinned_cpu():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="shell" type="grid" count="3 3 1" spacing=".1 .1 .1"
                mass="1" dim="2">
        <contact contype="0" conaffinity="0"/>
        <elasticity young="3000" poisson="0.2" damping="0.15"
                    thickness="0.02" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert model.flex_bending.size > 0
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.04
  qpos[-1] -= 0.03
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data, flex, force, _, _ = _evaluate(model, qpos, qvel)
  assert flex._bend_count > 0
  assert np.max(np.abs(data.qfrc_passive)) > 1e-5
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=5e-4, atol=5e-5)


def test_material_force_uses_articulated_off_center_flex_vertices():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="arm" pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="attached" type="grid" count="2 2 1"
                  spacing=".1 .1 .1" mass="1" dim="2">
          <contact contype="0" conaffinity="0"/>
          <edge stiffness="0" damping="0"/>
          <elasticity young="1000" poisson=".2" damping=".4"
                      thickness=".01" elastic2d="stretch"/>
        </flexcomp>
      </body>
    </worldbody></mujoco>
  """)
  assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_HINGE
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[0] = 0.31
  qpos[1] += 0.035
  qvel = np.linspace(-0.1, 0.1, model.nv)
  data, flex, force, _, _ = _evaluate(model, qpos, qvel)
  assert np.ptp(data.flexvert_xpos.reshape(-1, 3), axis=0).max() > 0.05
  assert np.max(np.abs(data.qfrc_passive)) > 1e-3
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=1e-3, atol=8e-5)
  np.testing.assert_allclose(
      flex.flexvert_xpos.detach().numpy()[0], data.flexvert_xpos.reshape(-1, 3),
      rtol=1e-5, atol=2e-6)


def test_mixed_flexes_keep_compiled_material_offsets_and_zero_material():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1"
                mass="1" dim="2">
        <contact contype="0" conaffinity="0"/><edge stiffness="0" damping="0"/>
        <elasticity young="500" poisson="0.25" damping="0.1" thickness="0.02" elastic2d="stretch"/>
      </flexcomp>
      <flexcomp name="tet" type="grid" count="2 2 2" spacing=".1 .1 .1"
                pos="0 0 1" mass="1" dim="3">
        <contact contype="0" conaffinity="0"/><edge stiffness="0" damping="0"/>
        <elasticity young="2000" poisson="0.15" damping="0.2"/>
      </flexcomp>
      <flexcomp name="no_material" type="grid" count="2 2 2" spacing=".1 .1 .1"
                pos="0 0 2" mass="1" dim="3">
        <contact contype="0" conaffinity="0"/><edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  desc = lower_flex_descriptor(model)
  assert desc is not None
  assert len(np.unique(desc.stiffnessadr[:2])) == 2
  assert desc.stiffnessadr[2] == -1
  assert np.any(desc.stiffness[desc.stiffnessadr[0]:desc.stiffnessadr[1]] != 0)
  assert np.any(desc.stiffness[desc.stiffnessadr[1]:] != 0)

  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qvel = np.linspace(-0.08, 0.09, model.nv)
  for f in range(3):
    vertex = int(model.flex_vertadr[f])
    body = int(model.flex_vertbodyid[vertex])
    joint = int(model.body_jntadr[body])
    qpos[int(model.jnt_qposadr[joint])] += 0.017 * (f + 1)
  data, _, force, _, _ = _evaluate(model, qpos, qvel)
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=5e-4, atol=5e-5)
  no_material = int(model.flex_vertadr[2])
  no_material_bodies = model.flex_vertbodyid[
      no_material:no_material + int(model.flex_vertnum[2])]
  no_material_dofs = np.concatenate([
      np.arange(model.body_dofadr[int(body)],
                model.body_dofadr[int(body)] + model.body_dofnum[int(body)])
      for body in no_material_bodies])
  np.testing.assert_allclose(force[no_material_dofs], 0.0, atol=1e-7)
