# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Regression checks for the pinned MuJoCo 3.10 compiled flex material law."""

import mujoco
import numpy as np
import pytest
try:
  import torch
except ImportError:  # --cpu test environment intentionally omits Torch.
  torch = None


def _flex_api():
  if torch is None:
    pytest.skip("flex tensor parity requires the prepared Torch runtime")
  from mujoco_metal.flex import MetalFlex, lower_flex_descriptor
  return MetalFlex, lower_flex_descriptor


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


def _root_com(model, data):
  """Repeat each kinematic root's subtree COM for its descendant bodies."""
  out = np.zeros((model.nbody, 3), dtype=np.float64)
  for body in range(1, model.nbody):
    root = body
    while int(model.body_parentid[root]) > 0:
      root = int(model.body_parentid[root])
    out[body] = data.subtree_com[root]
  return out


def _evaluate(model, qpos, qvel):
  MetalFlex, _ = _flex_api()
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
      "root_com": torch.tensor(
          _root_com(model, data)[None], dtype=torch.float32),
  }
  force, damping, stiffness = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(qvel[None], dtype=torch.float32), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32))
  return data, flex, force.detach().numpy()[0], damping.detach().numpy()[0], stiffness


def _oracle_passive_force(model, qpos, qvel):
  """Independent pinned-engine force evaluation at a supplied state."""
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  return np.asarray(data.qfrc_passive).copy()


@pytest.mark.parametrize("dim", [2, 3])
def test_compiled_simplex_metric_matches_pinned_cpu(dim):
  model = _model(dim)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qvel = np.zeros(model.nv, dtype=np.float64)
  qpos[3] += 0.02
  qpos[model.nq - 3] -= 0.013
  qvel[0] = 0.17
  qvel[-1] = -0.11

  data, flex, force, damping_tangent, stiffness_tangent = _evaluate(model, qpos, qvel)
  assert np.max(np.abs(data.qfrc_passive)) > 1e-5
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=3e-4, atol=4e-5)
  assert flex.descriptor.stiffness.size == model.flex_stiffness.size
  np.testing.assert_array_equal(flex.descriptor.stiffness, model.flex_stiffness.astype(np.float32))
  assert not np.any(flex.descriptor.young)
  eps = 2e-3
  for dof in (0, model.nv - 1):
    qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
    qpos_hi[dof] += eps
    qpos_lo[dof] -= eps
    numerical_q = (_oracle_passive_force(model, qpos_hi, qvel)
                   - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
    np.testing.assert_allclose(
        stiffness_tangent.detach().numpy()[0, :, dof], numerical_q,
        rtol=3e-2, atol=8e-2)
    qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
    qvel_hi[dof] += eps
    qvel_lo[dof] -= eps
    numerical_v = (_oracle_passive_force(model, qpos, qvel_hi)
                   - _oracle_passive_force(model, qpos, qvel_lo)) / (2*eps)
    np.testing.assert_allclose(
        damping_tangent[:, dof], numerical_v,
        rtol=3e-2, atol=8e-2)


@pytest.mark.parametrize("dim", [2, 3])
def test_mps_simplex_kernel_matches_pinned_cpu(dim):
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  model = _model(dim)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[-3] -= 0.013
  qvel = np.linspace(-0.04, 0.06, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, device="mps")
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32, device="mps"),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32, device="mps"),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32, device="mps"),
      "root_com": torch.tensor(
          _root_com(model, data)[None],
          dtype=torch.float32, device="mps"),
  }
  force, damping_tangent, stiffness_tangent = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(
      force.detach().cpu().numpy()[0], data.qfrc_passive,
      rtol=8e-4, atol=6e-5)
  eps = 2e-3
  for dof in (0, model.nv - 1):
    qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
    qpos_hi[dof] += eps
    qpos_lo[dof] -= eps
    numerical_q = (_oracle_passive_force(model, qpos_hi, qvel)
                   - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
    np.testing.assert_allclose(
        stiffness_tangent.detach().cpu().numpy()[0, :, dof], numerical_q,
        rtol=3e-2, atol=8e-2)
    qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
    qvel_hi[dof] += eps
    qvel_lo[dof] -= eps
    numerical_v = (_oracle_passive_force(model, qpos, qvel_hi)
                   - _oracle_passive_force(model, qpos, qvel_lo)) / (2*eps)
    np.testing.assert_allclose(
        damping_tangent.detach().cpu().numpy()[0, :, dof], numerical_v,
        rtol=3e-2, atol=8e-2)


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


def test_1d_edge_spring_tangent_includes_directional_geometry():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" timestep=".002"/><worldbody>
      <flexcomp name="cable" type="grid" count="3 1 1"
                spacing=".1 .1 .1" mass="1" dim="1">
        <contact contype="0" conaffinity="0"/>
        <edge stiffness="100" damping=".4"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[7] -= 0.01
  qvel = np.linspace(-0.08, 0.11, model.nv)
  data, flex, force, damping, stiffness = _evaluate(model, qpos, qvel)
  assert np.max(np.abs(data.qfrc_passive)) > 1e-5
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=1e-4, atol=2e-6)
  eps = 2e-4
  for dof in (0, 4, 8):
    qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
    qpos_hi[dof] += eps
    qpos_lo[dof] -= eps
    _, _, hi, _, _ = _evaluate(model, qpos_hi, qvel)
    _, _, lo, _, _ = _evaluate(model, qpos_lo, qvel)
    numerical = (hi - lo) / (2 * eps)
    np.testing.assert_allclose(
        stiffness.detach().numpy()[0, :, dof], numerical,
        rtol=3e-2, atol=1e-2)

    qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
    qvel_hi[dof] += eps
    qvel_lo[dof] -= eps
    _, _, hi, _, _ = _evaluate(model, qpos, qvel_hi)
    _, _, lo, _, _ = _evaluate(model, qpos, qvel_lo)
    numerical_damping = (hi - lo) / (2 * eps)
    np.testing.assert_allclose(
        damping[:, dof], numerical_damping,
        rtol=3e-2, atol=1e-2)


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
  data, flex, force, damping_tangent, stiffness_tangent = _evaluate(model, qpos, qvel)
  assert flex._bend_count > 0
  assert np.max(np.abs(data.qfrc_passive)) > 1e-5
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=5e-4, atol=5e-5)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  numerical_q = (_oracle_passive_force(model, qpos_hi, qvel)
                 - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      stiffness_tangent.detach().numpy()[0, :, 3], numerical_q,
      rtol=3e-2, atol=5e-3)
  qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
  qvel_hi[3] += eps
  qvel_lo[3] -= eps
  numerical_v = (_oracle_passive_force(model, qpos, qvel_hi)
                 - _oracle_passive_force(model, qpos, qvel_lo)) / (2*eps)
  np.testing.assert_allclose(
      damping_tangent[:, 3], numerical_v,
      rtol=3e-2, atol=5e-3)


def test_mps_triangle_shell_bending_force_and_tangent():
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="shell" type="grid" count="3 3 1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="0" conaffinity="0"/>
        <elasticity young="3000" poisson=".2" damping=".15"
                    thickness=".02" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.04
  qpos[-1] -= 0.03
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, device="mps")
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32, device="mps"),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32, device="mps"),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32, device="mps"),
      "root_com": torch.tensor(_root_com(model, data)[None],
                                dtype=torch.float32, device="mps"),
  }
  force, damping_tangent, stiffness_tangent = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(
      force.detach().cpu().numpy()[0], data.qfrc_passive,
      rtol=5e-4, atol=5e-5)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  numerical_q = (_oracle_passive_force(model, qpos_hi, qvel)
                 - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      stiffness_tangent.detach().cpu().numpy()[0, :, 3], numerical_q,
      rtol=3e-2, atol=5e-3)
  qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
  qvel_hi[3] += eps
  qvel_lo[3] -= eps
  numerical_v = (_oracle_passive_force(model, qpos, qvel_hi)
                 - _oracle_passive_force(model, qpos, qvel_lo)) / (2*eps)
  np.testing.assert_allclose(
      damping_tangent.detach().cpu().numpy()[0, :, 3], numerical_v,
      rtol=3e-2, atol=5e-3)


def test_articulated_triangle_bend_tangent_matches_pinned_qpos_difference():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="arm" pos=".1 -.2 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="shell" type="grid" count="2 2 1"
                  pos=".13 .21 -.07" spacing=".1 .1 .1" mass="1" dim="2">
          <contact contype="0" conaffinity="0"/>
          <elasticity young="3000" poisson=".2" damping=".15"
                      thickness=".02" elastic2d="bend"/>
        </flexcomp>
      </body>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[0] = 0.27
  qpos[1] += 0.035
  qvel = np.linspace(-0.1, 0.1, model.nv)
  data, flex, force, _, tangent = _evaluate(model, qpos, qvel)
  assert flex._bend_count > 0
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=1e-3, atol=8e-5)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[0] += eps
  qpos_lo[0] -= eps
  numerical = (_oracle_passive_force(model, qpos_hi, qvel)
               - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      tangent.detach().numpy()[0, :, 0], numerical,
      rtol=5e-2, atol=5e-3)


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
  data, flex, force, _, stiffness = _evaluate(model, qpos, qvel)
  assert np.ptp(data.flexvert_xpos.reshape(-1, 3), axis=0).max() > 0.05
  assert np.max(np.abs(data.qfrc_passive)) > 1e-3
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=1e-3, atol=8e-5)
  np.testing.assert_allclose(
      flex.flexvert_xpos.detach().numpy()[0], data.flexvert_xpos.reshape(-1, 3),
      rtol=1e-5, atol=2e-6)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[0] += eps
  qpos_lo[0] -= eps
  _, _, force_hi, _, _ = _evaluate(model, qpos_hi, qvel)
  _, _, force_lo, _, _ = _evaluate(model, qpos_lo, qvel)
  numerical = (force_hi - force_lo) / (2 * eps)
  np.testing.assert_allclose(
      stiffness.detach().numpy()[0, :, 0], numerical,
      rtol=5e-2, atol=2e-3)


@pytest.mark.parametrize(("joint_type", "rot_dof"), [("ball", 1), ("free", 4)])
def test_material_force_tangent_tracks_articulated_rotation(joint_type, rot_dof):
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="arm" pos=".2 -.1 1">
        <joint name="root" type="{joint_type}" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="attached" type="grid" count="2 2 1"
                  pos=".13 .21 -.07" spacing=".1 .1 .1" mass="1" dim="2">
          <contact contype="0" conaffinity="0"/>
          <edge stiffness="0" damping="0"/>
          <elasticity young="1000" poisson=".2" damping=".4"
                      thickness=".01" elastic2d="stretch"/>
        </flexcomp>
      </body>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[-1] += 0.035
  qvel = np.linspace(-0.1, 0.1, model.nv)
  data, _, _, _, stiffness = _evaluate(model, qpos, qvel)
  rot_start = 0 if joint_type == "ball" else 3
  assert np.max(np.abs(data.qfrc_passive[rot_start:rot_start + 3])) < 2e-5
  eps = 2e-2
  perturb = np.zeros(model.nv, dtype=np.float64)
  perturb[rot_dof] = eps
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  mujoco.mj_integratePos(model, qpos_hi, perturb, 1.0)
  mujoco.mj_integratePos(model, qpos_lo, perturb, -1.0)
  _, _, force_hi, _, _ = _evaluate(model, qpos_hi, qvel)
  _, _, force_lo, _, _ = _evaluate(model, qpos_lo, qvel)
  numerical = (force_hi - force_lo) / (2 * eps)
  np.testing.assert_allclose(
      stiffness.detach().numpy()[0, :, rot_dof], numerical,
      rtol=8e-2, atol=2e-4)


def test_interpolated_q1_volume_uses_compiled_element_matrix():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="q1" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert model.flex_interp[0] == 1
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data, flex, force, _, _ = _evaluate(model, qpos, qvel)
  assert flex._interp_count == 1
  assert np.max(np.abs(data.qfrc_passive)) > 1e-4
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=2e-4, atol=8e-6)


def test_mps_interpolated_q1_volume_kernel_matches_pinned_cpu():
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="q1" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, device="mps")
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32, device="mps"),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32, device="mps"),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32, device="mps"),
      "root_com": torch.tensor(
          _root_com(model, data)[None],
          dtype=torch.float32, device="mps"),
  }
  force, damping_tangent, stiffness_tangent = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(
      force.detach().cpu().numpy()[0], data.qfrc_passive,
      rtol=3e-4, atol=1e-5)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  numerical_q = (_oracle_passive_force(model, qpos_hi, qvel)
                 - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      stiffness_tangent.detach().cpu().numpy()[0, :, 3], numerical_q,
      rtol=3e-2, atol=4e-3)
  qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
  qvel_hi[3] += eps
  qvel_lo[3] -= eps
  numerical_v = (_oracle_passive_force(model, qpos, qvel_hi)
                 - _oracle_passive_force(model, qpos, qvel_lo)) / (2*eps)
  np.testing.assert_allclose(
      damping_tangent.detach().cpu().numpy()[0, :, 3], numerical_v,
      rtol=3e-2, atol=4e-3)
  spring_bit = int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  damper_bit = int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  for disable in (spring_bit, damper_bit, spring_bit | damper_bit):
    model.opt.disableflags = disable
    disabled_data = mujoco.MjData(model)
    disabled_data.qpos[:], disabled_data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, disabled_data)
    flex._disableflags = disable
    disabled_force = flex.run_device(
        torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
        torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
        torch.tensor(disabled_data.cvel[None], dtype=torch.float32, device="mps"))[0]
    np.testing.assert_allclose(
        disabled_force.detach().cpu().numpy()[0], disabled_data.qfrc_passive,
        rtol=3e-4, atol=1e-5)
  model.opt.disableflags = 0


def test_mps_interpolated_q2_volume_kernel_matches_pinned_cpu():
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="q2" type="grid" count="3 3 3"
                spacing=".1 .1 .1" mass="1" dim="3" dof="quadratic">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert model.flex_interp[0] == 2
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[-4] -= 0.01
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, device="mps")
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32, device="mps"),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32, device="mps"),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32, device="mps"),
      "root_com": torch.tensor(
          _root_com(model, data)[None],
          dtype=torch.float32, device="mps"),
  }
  force, damping_tangent, stiffness_tangent = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(
      force.detach().cpu().numpy()[0], data.qfrc_passive,
      rtol=3e-4, atol=1e-5)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  numerical_q = (_oracle_passive_force(model, qpos_hi, qvel)
                 - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      stiffness_tangent.detach().cpu().numpy()[0, :, 3], numerical_q,
      rtol=3e-2, atol=4e-3)
  qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
  qvel_hi[3] += eps
  qvel_lo[3] -= eps
  numerical_v = (_oracle_passive_force(model, qpos, qvel_hi)
                 - _oracle_passive_force(model, qpos, qvel_lo)) / (2*eps)
  np.testing.assert_allclose(
      damping_tangent.detach().cpu().numpy()[0, :, 3], numerical_v,
      rtol=3e-2, atol=4e-3)


def test_interpolated_q2_volume_uses_compiled_element_matrix():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="q2" type="grid" count="3 3 3"
                spacing=".1 .1 .1" mass="1" dim="3" dof="quadratic">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert model.flex_interp[0] == 2
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[-4] -= 0.01
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data, flex, force, _, _ = _evaluate(model, qpos, qvel)
  assert flex._interp_npe_host == (27,)
  assert np.max(np.abs(data.qfrc_passive)) > 1e-4
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=3e-4, atol=1e-5)


@pytest.mark.parametrize(("count", "dof"), [("2 2 2", "trilinear"),
                                              ("3 3 3", "quadratic")])
def test_interpolated_volume_analytic_tangent_matches_qpos_difference(count, dof):
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="volume" type="grid" count="{count}"
                spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qvel = np.linspace(-0.03, 0.04, model.nv)
  _, _, _, damping_tangent, tangent = _evaluate(model, qpos, qvel)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  _, _, force_hi, _, _ = _evaluate(model, qpos_hi, qvel)
  _, _, force_lo, _, _ = _evaluate(model, qpos_lo, qvel)
  numerical = (force_hi - force_lo) / (2 * eps)
  np.testing.assert_allclose(
      tangent.detach().numpy()[0, :, 3], numerical,
      rtol=2e-2, atol=4e-3)
  qvel_hi, qvel_lo = qvel.copy(), qvel.copy()
  qvel_hi[3] += eps
  qvel_lo[3] -= eps
  _, _, force_vhi, _, _ = _evaluate(model, qpos, qvel_hi)
  _, _, force_vlo, _, _ = _evaluate(model, qpos, qvel_lo)
  numerical_damping = (force_vhi - force_vlo) / (2 * eps)
  np.testing.assert_allclose(
      damping_tangent[:, 3], numerical_damping,
      rtol=2e-2, atol=4e-3)


def test_mps_interpolated_q1_shell_bending_matches_pinned_cpu():
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="shell" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"
                    thickness=".01" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[-1] -= 0.015
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, device="mps")
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32, device="mps"),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32, device="mps"),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32, device="mps"),
      "root_com": torch.tensor(
          _root_com(model, data)[None],
          dtype=torch.float32, device="mps"),
  }
  force, _, stiffness_tangent = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(
      force.detach().cpu().numpy()[0], data.qfrc_passive,
      rtol=3e-4, atol=2e-7)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  numerical = (_oracle_passive_force(model, qpos_hi, qvel)
               - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      stiffness_tangent.detach().cpu().numpy()[0, :, 3], numerical,
      rtol=3e-2, atol=5e-3)
def test_interpolated_q1_shell_bending_matches_pinned_cpu():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="shell" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"
                    thickness=".01" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert model.flex_interp[0] == -1
  assert model.flex_bending.size > 0
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[-1] -= 0.015
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data, flex, force, _, tangent = _evaluate(model, qpos, qvel)
  assert flex._shell_bend_count > 0
  assert np.max(np.abs(data.qfrc_passive)) > 1e-5
  # Pinned 3.10 warns and leaves interpolated shell bending damping disabled.
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=2e-4, atol=2e-7)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  _, _, force_hi, _, _ = _evaluate(model, qpos_hi, qvel)
  _, _, force_lo, _, _ = _evaluate(model, qpos_lo, qvel)
  numerical = (force_hi - force_lo) / (2 * eps)
  np.testing.assert_allclose(
      tangent.detach().numpy()[0, :, 3], numerical,
      rtol=3e-2, atol=5e-3)


def test_interpolated_q2_shell_bending_tangent_matches_pinned_cpu():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="shell" type="grid" count="3 3 3"
                spacing=".1 .1 .1" mass="1" dim="3" dof="quadratic">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"
                    thickness=".01" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert model.flex_interp[0] == -2
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[-1] -= 0.015
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data, flex, force, _, tangent = _evaluate(model, qpos, qvel)
  assert flex._shell_bend_count > 0
  assert 2 in set(flex._shell_face_order_host)
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=2e-4, atol=2e-7)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  numerical = (_oracle_passive_force(model, qpos_hi, qvel)
               - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      tangent.detach().numpy()[0, :, 3], numerical,
      rtol=3e-2, atol=2e-4)


@pytest.mark.parametrize("profile", ["integrated_euler_v1", "integrated_scalable_v1"])
@pytest.mark.parametrize(("count", "dof", "elastic2d"), [
    ("2 2 2", "trilinear", ""),
    ("3 3 3", "quadratic", ""),
    ("2 2 2", "trilinear", "bend"),
    ("3 3 3", "quadratic", "bend"),
])
def test_public_integrators_step_compiled_flex_material(profile, count, dof, elastic2d):
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  from mujoco_metal import MetalSimulation
  elastic2d_attr = f' elastic2d="{elastic2d}"' if elastic2d else ""
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep=".001" gravity="0 0 0"/><worldbody>
      <flexcomp name="volume" type="grid" count="{count}"
                spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"
                    thickness=".01"{elastic2d_attr}/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.01
  qvel = np.linspace(-0.01, 0.01, model.nv)
  cpu = mujoco.MjData(model)
  cpu.qpos[:], cpu.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, cpu)
  sim = MetalSimulation(
      model, batch_size=1, qpos=qpos[None].astype(np.float32),
      qvel=qvel[None].astype(np.float32), profile=profile)
  sim.step()
  mujoco.mj_step(model, cpu)
  native_qpos = sim.state.qpos.detach().cpu().numpy()[0]
  native_qvel = sim.state.qvel.detach().cpu().numpy()[0]
  assert np.all(np.isfinite(native_qpos))
  assert np.all(np.isfinite(native_qvel))
  np.testing.assert_allclose(native_qpos, cpu.qpos, rtol=3e-3, atol=3e-5)
  np.testing.assert_allclose(native_qvel, cpu.qvel, rtol=5e-3, atol=5e-4)


def test_mps_interpolated_q2_shell_bending_force_and_tangent():
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="shell" type="grid" count="3 3 3"
                spacing=".1 .1 .1" mass="1" dim="3" dof="quadratic">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"
                    thickness=".01" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += 0.02
  qpos[-1] -= 0.015
  qvel = np.linspace(-0.03, 0.04, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, device="mps")
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32, device="mps"),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32, device="mps"),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32, device="mps"),
      "root_com": torch.tensor(_root_com(model, data)[None],
                                dtype=torch.float32, device="mps"),
  }
  force, _, tangent = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(
      force.detach().cpu().numpy()[0], data.qfrc_passive,
      rtol=3e-4, atol=2e-7)
  eps = 2e-3
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  qpos_hi[3] += eps
  qpos_lo[3] -= eps
  numerical = (_oracle_passive_force(model, qpos_hi, qvel)
               - _oracle_passive_force(model, qpos_lo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      tangent.detach().cpu().numpy()[0, :, 3], numerical,
      rtol=3e-2, atol=2e-4)


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
  _, lower_flex_descriptor = _flex_api()
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
