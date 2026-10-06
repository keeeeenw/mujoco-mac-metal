# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Regression checks for the pinned MuJoCo 3.10 compiled flex material law."""

import os

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


def test_quaternion_rotate_broadcasts_shared_shell_reference_across_batch():
  if torch is None:
    pytest.skip("quaternion tensor check requires the prepared Torch runtime")
  MetalFlex, _ = _flex_api()
  half = np.sqrt(0.5)
  quat = torch.tensor([[1., 0., 0., 0.],
                       [half, 0., 0., half]], dtype=torch.float32)
  vector = torch.tensor([[[1., 0., 0.]]], dtype=torch.float32)
  rotated = MetalFlex._quat_rotate(quat, vector)
  assert tuple(rotated.shape) == (2, 1, 3)
  torch.testing.assert_close(
      rotated,
      torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]]], dtype=torch.float32),
      rtol=1e-6, atol=1e-6)


def test_vertex_spatial_jacobian_is_lazy_and_rebuilds_after_position_restore():
  """Sparse flex contact does not reserve the legacy dense vertex-by-DOF J."""
  if torch is None:
    pytest.skip("flex tensor check requires the prepared Torch runtime")
  MetalFlex, _ = _flex_api()
  model = _model(2)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  root_com = _root_com(model, data)
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(root_com[None], dtype=torch.float32),
  }
  cvel = torch.tensor(data.cvel[None], dtype=torch.float32)
  flex = MetalFlex(model, device="cpu")
  assert not flex._flexvert_spatial_J_allocated
  assert flex._flexvert_spatial_J.numel() == 1
  flex.update_kinematics(poses, cvel)
  assert not flex._flexvert_spatial_J_allocated
  context = flex.capture_position_context()

  spatial = flex.contact_spatial_jacobians()
  assert flex._flexvert_spatial_J_allocated
  assert tuple(spatial.shape) == (1, model.nflexvert, 6, model.nv)
  torch.testing.assert_close(spatial[:, :, :3], flex._flexvert_J)

  flex.update_kinematics(poses, cvel)
  flex.run_velocity_device(
      context, torch.tensor(data.qpos[None], dtype=torch.float32),
      torch.tensor(data.qvel[None], dtype=torch.float32))
  restored = flex.contact_spatial_jacobians()
  torch.testing.assert_close(restored[:, :, :3], flex._flexvert_J)


def test_cached_pos_velocity_batches_flex_jacobian_contraction():
  """The generalized-velocity axis must remain distinct from point count."""
  if torch is None:
    pytest.skip("flex tensor check requires the prepared Torch runtime")
  MetalFlex, _ = _flex_api()
  model = _model(2)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  root_com = _root_com(model, data)
  repeat = lambda value: np.repeat(np.asarray(value)[None], 2, axis=0)
  poses = {
      "body_pos": torch.tensor(repeat(data.xpos), dtype=torch.float32),
      "body_quat": torch.tensor(repeat(data.xquat), dtype=torch.float32),
      "joint_anchor": torch.tensor(repeat(data.xanchor), dtype=torch.float32),
      "joint_axis": torch.tensor(repeat(data.xaxis), dtype=torch.float32),
      "root_com": torch.tensor(repeat(root_com), dtype=torch.float32),
  }
  cvel = torch.tensor(repeat(data.cvel), dtype=torch.float32)
  flex = MetalFlex(model, batch_size=2, device="cpu")
  flex.update_kinematics(poses, cvel)
  context = flex.capture_position_context()
  qpos = torch.tensor(repeat(data.qpos), dtype=torch.float32)
  qvel = torch.arange(2 * model.nv, dtype=torch.float32).reshape(2, model.nv)
  flex.run_velocity_device(context, qpos, qvel, poses, cvel)
  expected = torch.einsum("bvcn,bn->bvc", flex._flexvert_J, qvel)
  torch.testing.assert_close(flex._flexvert_xvel, expected)


def test_flex_device_validation_accepts_unindexed_device_alias():
  if torch is None:
    pytest.skip("device alias check requires Torch")
  from mujoco_metal.flex import _same_torch_device
  assert _same_torch_device(torch.device("cpu"), torch.device("cpu"))
  assert _same_torch_device(torch.device("mps:0"), torch.device("mps"))
  assert not _same_torch_device(torch.device("mps:1"), torch.device("mps:0"))


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


def test_masked_recovery_flex_force_uses_separate_result_owner():
  """Selected recovery forces do not clear the accepted ordinary result."""
  if torch is None:
    pytest.skip("flex tensor parity requires the prepared Torch runtime")
  MetalFlex, _ = _flex_api()
  model = _model(2)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  flex = MetalFlex(model, batch_size=2, device="cpu")
  qpos = torch.tensor(np.repeat(data.qpos[None], 2, axis=0), dtype=torch.float32)
  qvel = torch.tensor(np.repeat(data.qvel[None], 2, axis=0), dtype=torch.float32)
  poses = {
      "body_pos": torch.tensor(np.repeat(data.xpos[None], 2, axis=0),
                                dtype=torch.float32),
      "body_quat": torch.tensor(np.repeat(data.xquat[None], 2, axis=0),
                                 dtype=torch.float32),
      "joint_anchor": torch.tensor(np.repeat(data.xanchor[None], 2, axis=0),
                                    dtype=torch.float32),
      "joint_axis": torch.tensor(np.repeat(data.xaxis[None], 2, axis=0),
                                  dtype=torch.float32),
      "root_com": torch.tensor(np.repeat(_root_com(model, data)[None], 2,
                                         axis=0), dtype=torch.float32),
  }
  full_force, full_damping, full_stiffness = flex.run_device(qpos, qvel, poses)
  ordinary = tuple(value.clone() for value in (
      flex._qfrc_passive, flex._damping_tangent, flex._stiffness_tangent))
  selected_force, selected_damping, selected_stiffness = flex.run_device(
      qpos, qvel, poses, world_mask=torch.tensor([1, 0], dtype=torch.int32))
  torch.testing.assert_close(selected_force[0], full_force[0])
  torch.testing.assert_close(selected_damping[0], full_damping[0])
  torch.testing.assert_close(selected_stiffness[0], full_stiffness[0])
  assert torch.count_nonzero(selected_force[1]) == 0
  assert torch.count_nonzero(selected_damping[1]) == 0
  assert torch.count_nonzero(selected_stiffness[1]) == 0
  for actual, before in zip((flex._qfrc_passive, flex._damping_tangent,
                             flex._stiffness_tangent), ordinary):
    assert torch.equal(actual, before)


def _cached_pos_velocity_witness(device):
  MetalFlex, _ = _flex_api()
  to_device = lambda value: torch.as_tensor(
      value, dtype=torch.float32, device=device).contiguous()
  # The flexcomp is off-center on a hinge body and its generated vertices are
  # attached through articulated slide coordinates.
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" timestep=".002"/><worldbody>
      <body name="arm" pos=".1 -.2 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
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
  qpos0 = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos0[0] = 0.27
  qpos0[1] += 0.035
  qvel0 = np.linspace(-0.08, 0.06, model.nv)
  data0 = mujoco.MjData(model)
  data0.qpos[:], data0.qvel[:] = qpos0, qvel0
  mujoco.mj_forward(model, data0)
  poses0 = {
      "body_pos": to_device(data0.xpos[None]),
      "body_quat": to_device(data0.xquat[None]),
      "joint_anchor": to_device(data0.xanchor[None]),
      "joint_axis": to_device(data0.xaxis[None]),
      "root_com": to_device(_root_com(model, data0)[None]),
  }
  flex = MetalFlex(model, device=device)
  flex.update_kinematics(poses0, to_device(data0.cvel[None]))
  context = flex.capture_position_context()
  frozen_positions = flex._flexvert_xpos.clone()
  frozen_generation = flex._kinematics_generation

  qpos3 = qpos0.copy()
  qpos3[0] += 0.12
  qpos3[2] -= 0.02
  qvel3 = np.linspace(0.12, -0.07, model.nv)
  data3 = mujoco.MjData(model)
  data3.qpos[:], data3.qvel[:] = qpos3, qvel3
  mujoco.mj_forward(model, data3)
  poses3 = {
      "body_pos": to_device(data3.xpos[None]),
      "body_quat": to_device(data3.xquat[None]),
      "joint_anchor": to_device(data3.xanchor[None]),
      "joint_axis": to_device(data3.xaxis[None]),
      "root_com": to_device(_root_com(model, data3)[None]),
  }
  cached = flex.run_velocity_device(
      context, to_device(qpos3[None]), to_device(qvel3[None]), poses3,
      to_device(data3.cvel[None]))
  assert flex._kinematics_generation == frozen_generation
  torch.testing.assert_close(flex._flexvert_xpos, frozen_positions)

  # The independent ordinary evaluation uses X0 and the new generalized
  # velocity, so its physical force/tangents must match the frozen-POS replay.
  reference_data = mujoco.MjData(model)
  reference_data.qpos[:], reference_data.qvel[:] = qpos0, qvel3
  mujoco.mj_forward(model, reference_data)
  pinned_pos = np.asarray(reference_data.flexvert_xpos).copy()
  pinned_length = np.asarray(reference_data.flexedge_length).copy()
  pinned_velocity = np.asarray(reference_data.flexedge_velocity).copy()
  reference_data.qpos[:] = qpos3
  mujoco.mj_forwardSkip(
      model, reference_data, mujoco.mjtStage.mjSTAGE_POS, 0)
  # This is the exact X3-qpos/X0-POS behavior used by the split RK4 path.
  np.testing.assert_array_equal(reference_data.qpos, qpos3)
  np.testing.assert_allclose(reference_data.flexvert_xpos, pinned_pos, atol=1e-12)
  np.testing.assert_allclose(reference_data.flexedge_length, pinned_length, atol=1e-12)
  np.testing.assert_allclose(reference_data.flexedge_velocity, pinned_velocity,
                             rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(
      cached[0].detach().numpy()[0], reference_data.qfrc_passive,
      rtol=4e-4, atol=5e-5)
  np.testing.assert_allclose(flex._flexvert_xpos.detach().numpy()[0],
                             reference_data.flexvert_xpos,
                             rtol=0, atol=3e-7)
  np.testing.assert_allclose(flex._flexedge_length.detach().numpy()[0],
                             reference_data.flexedge_length,
                             rtol=2e-6, atol=2e-7)
  np.testing.assert_allclose(flex.flexedge_velocity.detach().numpy()[0],
                             reference_data.flexedge_velocity,
                             rtol=3e-5, atol=3e-6)
  root_com = _root_com(model, reference_data)[model.flex_vertbodyid]
  point_velocity = (reference_data.cvel[model.flex_vertbodyid, 3:]
                    + np.cross(reference_data.cvel[model.flex_vertbodyid, :3],
                               reference_data.flexvert_xpos - root_com))
  np.testing.assert_allclose(flex._flexvert_xvel.detach().numpy()[0],
                             point_velocity, rtol=3e-5, atol=3e-6)
  reference_poses = {
      "body_pos": to_device(reference_data.xpos[None]),
      "body_quat": to_device(reference_data.xquat[None]),
      "joint_anchor": to_device(reference_data.xanchor[None]),
      "joint_axis": to_device(reference_data.xaxis[None]),
      "root_com": to_device(_root_com(model, reference_data)[None]),
  }
  reference = MetalFlex(model, device=device).run_device(
      to_device(qpos0[None]), to_device(qvel3[None]), reference_poses,
      to_device(reference_data.cvel[None]))
  for actual, expected in zip(cached, reference):
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=3e-5)

  old_context = context
  flex.capture_position_context()
  with pytest.raises(ValueError, match="stale"):
    flex.run_velocity_device(old_context,
        to_device(qpos0[None]), to_device(qvel3[None]))


def test_cached_pos_velocity_passive_evaluation_skips_kinematics_and_guards_context():
  _cached_pos_velocity_witness("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native frozen-POS flex replay requires explicit GPU opt-in")
def test_native_cached_pos_velocity_matches_pinned_forward_skip():
  torch_runtime = pytest.importorskip("torch")
  if not torch_runtime.backends.mps.is_available():
    pytest.skip("native Metal flex replay requires MPS")
  _cached_pos_velocity_witness("mps")


def _cached_flex_equality_witness(device):
  MetalFlex, _ = _flex_api()
  to_device = lambda value: torch.as_tensor(
      value, dtype=torch.float32, device=device).contiguous()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 -9.81" timestep=".002" jacobian="dense"/>
      <worldbody><body name="arm" pos=".1 -.2 1">
        <joint name="slide" type="slide" axis="1 .2 0"/>
        <geom type="sphere" size=".02" mass="1"/>
        <flexcomp name="cable" type="grid" count="4 1 1"
                  pos=".03 .02 .01" spacing=".1 .1 .1" mass=".5"
                  radius=".01" dim="1">
          <edge equality="true" solref=".03 1.2"
                solimp=".8 .95 .01 .5 2"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </body></worldbody>
      <equality><flex flex="cable"/></equality>
    </mujoco>
  """)
  qpos0 = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos0[0] = .17
  qpos0[1] += .013
  qvel0 = np.linspace(-.09, .07, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos0, qvel0
  mujoco.mj_forward(model, data)
  root_com = _root_com(model, data)
  poses = {
      "body_pos": to_device(data.xpos[None]),
      "body_quat": to_device(data.xquat[None]),
      "joint_anchor": to_device(data.xanchor[None]),
      "joint_axis": to_device(data.xaxis[None]),
      "root_com": to_device(root_com[None]),
  }
  flex = MetalFlex(model, device=device)
  flex.update_kinematics(poses, to_device(data.cvel[None]))
  context = flex.capture_position_context()
  frozen_length = data.flexedge_length.copy()
  def dense_rows(values, rowadr, rownnz, colind, nrow):
    values = np.asarray(values)
    if values.size == nrow * model.nv:
      return values.reshape(nrow, model.nv).astype(np.float64, copy=True)
    dense = np.zeros((nrow, model.nv), dtype=np.float64)
    for row in range(nrow):
      start, count = int(rowadr[row]), int(rownnz[row])
      dense[row, np.asarray(colind[start:start + count], dtype=np.int64)] = \
          np.asarray(values[start:start + count], dtype=np.float64)
    return dense
  frozen_J = dense_rows(data.flexedge_J, model.flexedge_J_rowadr,
                        model.flexedge_J_rownnz, model.flexedge_J_colind,
                        model.nflexedge)

  # MuJoCo's split-stage contract keeps the original qpos-dependent POS
  # state, restores another qpos, and updates velocity-dependent rows only.
  qpos3 = qpos0.copy()
  qpos3[0] -= .11
  qpos3[1] -= .006
  qvel3 = np.linspace(.12, -.08, model.nv)
  data.qpos[:] = qpos3
  data.qvel[:] = qvel3
  mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_POS, 0)
  rows = np.flatnonzero((data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
                        & (data.efc_id == 0))
  assert rows.size > 0
  np.testing.assert_allclose(data.flexedge_length, frozen_length, atol=1e-12)
  current_flex_J = dense_rows(data.flexedge_J, model.flexedge_J_rowadr,
                              model.flexedge_J_rownnz,
                              model.flexedge_J_colind, model.nflexedge)
  np.testing.assert_allclose(current_flex_J, frozen_J, atol=1e-12)

  actual = flex.run_equalities_velocity(
      context, to_device(qpos3[None]), to_device(qvel3[None]), equality_id=0)
  edge_ids = flex.equality_edge_ids(int(model.eq_obj1id[0]))
  # Canonical row order is increasing non-rigid edge ID; the compiled
  # equality can share this flex with a second flexcomp-generated equality.
  assert edge_ids.size == rows.size
  np.testing.assert_allclose(actual[0].detach().numpy()[0, edge_ids],
                             data.efc_pos[rows], rtol=2e-6, atol=2e-7)
  np.testing.assert_allclose(actual[3].detach().numpy()[0, edge_ids],
                             dense_rows(data.efc_J, data.efc_J_rowadr,
                                        data.efc_J_rownnz, data.efc_J_colind,
                                        data.nefc)[rows],
                             rtol=3e-5, atol=3e-6)
  np.testing.assert_allclose(
      flex.flexedge_velocity.detach().numpy()[0, edge_ids],
      data.efc_vel[rows], rtol=3e-5, atol=3e-6)
  np.testing.assert_allclose(actual[1].detach().numpy()[0, edge_ids],
                             data.efc_aref[rows], rtol=5e-4, atol=5e-5)
  np.testing.assert_allclose(actual[2].detach().numpy()[0, edge_ids],
                             data.efc_R[rows], rtol=5e-4, atol=5e-5)
  raw = flex.equality_position_context
  assert raw["edge_J"] is flex.flexedge_J
  np.testing.assert_allclose(raw["pos"].detach().numpy()[0, edge_ids],
                             data.efc_pos[rows], rtol=2e-6, atol=2e-7)
  assert len(raw["edge_ids_by_flex"]) == model.nflex


def test_cached_pos_flex_equalities_match_pinned_forward_skip():
  _cached_flex_equality_witness("cpu")


def test_pinned_flex_equality_rows_follow_compact_nonrigid_edge_order():
  """CPU-only source oracle for `mjEQ_FLEX` sparse row identity."""
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option jacobian="dense"/><worldbody>
      <body name="arm"><joint type="slide" axis="1 .2 0"/>
        <geom type="sphere" size=".02" mass="1"/>
        <flexcomp name="cable" type="grid" count="4 1 1"
                  spacing=".1 .1 .1" mass=".5" radius=".01" dim="1">
          <edge equality="true"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </body></worldbody><equality><flex flex="cable"/></equality>
    </mujoco>
  """)
  data = mujoco.MjData(model)
  data.qpos[0] = .08
  data.qvel[:] = np.linspace(-.06, .09, model.nv)
  mujoco.mj_forward(model, data)
  eqid = int(np.flatnonzero(
      np.asarray(model.eq_type) == int(mujoco.mjtEq.mjEQ_FLEX))[0])
  flexid = int(model.eq_obj1id[eqid])
  rows = np.flatnonzero(
      (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
      & (data.efc_id == eqid))
  edges = np.arange(int(model.flex_edgeadr[flexid]),
                    int(model.flex_edgeadr[flexid] + model.flex_edgenum[flexid]))
  expected_edges = edges[~np.asarray(model.flexedge_rigid[edges], dtype=bool)]
  assert rows.size == expected_edges.size > 0
  dense_edge_J = np.zeros((model.nflexedge, model.nv), dtype=np.float64)
  for edge in range(model.nflexedge):
    start = int(model.flexedge_J_rowadr[edge])
    count = int(model.flexedge_J_rownnz[edge])
    cols = np.asarray(model.flexedge_J_colind[start:start + count], dtype=np.int64)
    dense_edge_J[edge, cols] = data.flexedge_J[start:start + count]
  dense_efc_J = data.efc_J.reshape(data.nefc, model.nv)
  np.testing.assert_allclose(dense_edge_J[expected_edges], dense_efc_J[rows],
                             rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(data.efc_pos[rows],
      data.flexedge_length[expected_edges] - model.flexedge_length0[expected_edges],
      rtol=1e-12, atol=1e-12)


def _flexvert_equalities_witness(device):
  MetalFlex, _ = _flex_api()
  to_device = lambda value: torch.as_tensor(
      value, dtype=torch.float32, device=device).contiguous()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option jacobian="dense" gravity="0 0 0" timestep=".002"/>
      <worldbody><flexcomp name="cloth" type="grid" count="2 2 1"
          spacing=".1 .1 .1" mass="1" radius=".01" dim="2">
        <edge equality="strain"/>
        <contact contype="0" conaffinity="0"/>
      </flexcomp></worldbody>
      <equality><flexvert flex="cloth" solref=".03 1.2"
          solimp=".8 .95 .01 .5 2"/></equality>
    </mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[-1] += .02
  qvel = np.linspace(-.07, .09, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  poses = {
      "body_pos": to_device(data.xpos[None]),
      "body_quat": to_device(data.xquat[None]),
      "joint_anchor": to_device(data.xanchor[None]),
      "joint_axis": to_device(data.xaxis[None]),
      "root_com": to_device(data.subtree_com[None]),
  }
  flex = MetalFlex(model, device=device)
  flex.update_kinematics(poses, to_device(data.cvel[None]))
  context = flex.capture_position_context()
  eqid = int(np.flatnonzero(
      np.asarray(model.eq_type) == int(mujoco.mjtEq.mjEQ_FLEXVERT))[0])
  actual = flex.run_flexvert_equality_rows(
      eqid, to_device(qvel[None]), context=context)
  rows = np.flatnonzero(
      (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
      & (data.efc_id == eqid))
  f = int(model.eq_obj1id[eqid])
  row_ids = flex.flexvert_row_ids(f)
  assert rows.size == row_ids.size == 2 * int(model.flex_vertnum[f])
  np.testing.assert_array_equal(actual["row_ids"], row_ids)
  np.testing.assert_allclose(actual["pos"].detach().numpy()[0],
                             data.efc_pos[rows], rtol=3e-5, atol=3e-6)
  np.testing.assert_allclose(actual["vel"].detach().numpy()[0],
                             data.efc_vel[rows], rtol=3e-5, atol=3e-6)
  # flexvert_J is a paired sparse value matrix; rowadr counts scalar entries
  # in its flattened storage, not rows in the 2-column Python view.
  rowadr = np.asarray(model.flexvert_J_rowadr).reshape(-1)
  rownnz = np.asarray(model.flexvert_J_rownnz).reshape(-1)
  colind = np.asarray(model.flexvert_J_colind).reshape(-1)
  values = np.asarray(data.flexvert_J).reshape(-1)
  dense_J = np.zeros((2 * model.nflexvert, model.nv), dtype=np.float64)
  for row in range(2 * model.nflexvert):
    start, count = int(rowadr[row]), int(rownnz[row])
    dense_J[row, colind[start:start + count]] = values[start:start + count]
  np.testing.assert_allclose(actual["J"].detach().numpy()[0], dense_J,
                             rtol=5e-5, atol=5e-6)
  np.testing.assert_allclose(actual["aref"].detach().numpy()[0],
                             data.efc_aref[rows], rtol=5e-4, atol=5e-5)
  np.testing.assert_allclose(actual["R"].detach().numpy()[0],
                             data.efc_R[rows], rtol=5e-4, atol=5e-5)
  unified = flex.flex_equality_position_context(context)[eqid]
  assert unified["kind"] == "flexvert"
  assert unified["jdot_included"] is False
  assert unified["jdot_policy"] == "pinned_zero_for_flex_equality"
  np.testing.assert_array_equal(unified["row_ids"], row_ids)
  np.testing.assert_allclose(unified["pos"].detach().numpy()[0],
                             data.efc_pos[rows], rtol=3e-5, atol=3e-6)
  np.testing.assert_allclose(unified["J"].detach().numpy()[0], dense_J,
                             rtol=5e-5, atol=5e-6)
  qvel_replay = qvel[::-1].copy()
  replay = flex.run_flex_equality_rows(
      eqid, context, to_device(qpos[None]), to_device(qvel_replay[None]))
  np.testing.assert_allclose(replay["pos"].detach().numpy()[0], data.efc_pos[rows],
                             rtol=3e-5, atol=3e-6)
  np.testing.assert_allclose(replay["vel"].detach().numpy()[0],
                             dense_J @ qvel_replay, rtol=5e-5, atol=5e-6)
  assert replay["jdot_included"] is False


def test_flexvert_equality_rows_match_pinned_cpu():
  _flexvert_equalities_witness("cpu")


def test_compiled_flex_equality_row_inventory_is_lowered_before_manager():
  from mujoco_metal.flex import lower_flex_descriptor
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option jacobian="dense" gravity="0 0 0"/>
      <worldbody><flexcomp name="cloth" type="grid" count="2 2 1"
          spacing=".1 .1 .1" mass="1" radius=".01" dim="2">
        <edge equality="strain"/>
        <contact contype="0" conaffinity="0"/>
      </flexcomp></worldbody>
      <equality><flexvert flex="cloth"/></equality>
    </mujoco>
  """)
  desc = lower_flex_descriptor(model)
  equality_types = np.asarray(model.eq_type)
  vert_id = int(np.flatnonzero(
      equality_types == int(mujoco.mjtEq.mjEQ_FLEXVERT))[0])
  strain_id = int(np.flatnonzero(
      equality_types == int(mujoco.mjtEq.mjEQ_FLEXSTRAIN))[0])
  assert tuple(desc.equality_row_ids_by_eqid) == tuple(
      sorted(desc.equality_row_ids_by_eqid))
  assert desc.equality_row_counts[vert_id] == 2 * int(model.flex_vertnum[0])
  np.testing.assert_array_equal(
      desc.equality_row_ids_by_eqid[vert_id],
      np.arange(2 * int(model.flex_vertadr[0]),
                2 * (int(model.flex_vertadr[0]) + int(model.flex_vertnum[0])),
                dtype=np.int32))
  # The compiler keeps an auto-generated FLEXSTRAIN equality for this
  # non-interpolated shell, but emits no stiffness eigenmodes/solver rows.
  assert desc.equality_row_counts[strain_id] == 0
  assert desc.equality_row_ids_by_eqid[strain_id].size == 0
  with pytest.raises(TypeError):
    desc.equality_row_counts[strain_id] = 1
  with pytest.raises(ValueError):
    desc.equality_row_ids_by_eqid[vert_id][0] = 42


def test_compiled_edge_equality_inventory_keeps_eqids_separate():
  from mujoco_metal.flex import lower_flex_descriptor
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="arm">
      <joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".02" mass="1"/>
      <flexcomp name="cable" type="grid" count="4 1 1"
          spacing=".1 .1 .1" mass=".5" radius=".01" dim="1">
        <edge equality="true"/>
        <contact contype="0" conaffinity="0"/>
      </flexcomp>
    </body></worldbody>
    <equality><flex flex="cable"/><flex flex="cable"/></equality>
    </mujoco>
  """)
  desc = lower_flex_descriptor(model)
  ids = np.flatnonzero(
      np.asarray(model.eq_type) == int(mujoco.mjtEq.mjEQ_FLEX))
  assert ids.size >= 2
  flex_id = int(model.eq_obj1id[int(ids[0])])
  edge_start = int(model.flex_edgeadr[flex_id])
  edge_stop = edge_start + int(model.flex_edgenum[flex_id])
  expected = np.asarray([
      edge for edge in range(edge_start, edge_stop)
      if not bool(model.flexedge_rigid[edge])], dtype=np.int32)
  for eqid in ids:
    assert desc.equality_row_counts[int(eqid)] == expected.size
    np.testing.assert_array_equal(
        desc.equality_row_ids_by_eqid[int(eqid)], expected)
  assert tuple(int(eqid) for eqid in ids) == tuple(sorted(
      desc.equality_row_ids_by_eqid.keys()))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native flexvert parity requires explicit GPU opt-in")
def test_native_flexvert_equality_rows_match_pinned_cpu():
  torch_runtime = pytest.importorskip("torch")
  if not torch_runtime.backends.mps.is_available():
    pytest.skip("native Metal flexvert equality requires MPS")
  _flexvert_equalities_witness("mps")


def _flexstrain_equality_witness(device, order, jacobian="dense", shell=False):
  MetalFlex, _ = _flex_api()
  dof = "trilinear" if order == 1 else "quadratic"
  if shell:
    count, cellcount = "2 2 2", "1 1 1"
    edge, elasticity = "", '<elasticity young="1000" poisson=".2" thickness=".01" elastic2d="bend"/>'
    explicit_eq = '<equality><flexstrain name="strain" flex="volume" cell="0 0 0"/></equality>'
  else:
    # Q1 spans two cells so the per-equality cell identity/offset is exercised;
    # Q2 is one cell because its node count grows cubically.
    count = "3 2 2" if order == 1 else "3 3 3"
    cellcount = "2 1 1" if order == 1 else "1 1 1"
    edge = '<edge equality="strain" solref=".03 1.2" solimp=".8 .95 .01 .5 2"/>'
    elasticity, explicit_eq = "", ""
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" jacobian="{jacobian}"/>
      <worldbody><body name="root"><freejoint/>
        <geom type="sphere" size=".02" mass="1"/>
        <flexcomp name="volume" type="grid" count="{count}"
                  cellcount="{cellcount}"
                  spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          {edge}{elasticity}
        </flexcomp>
      </body></worldbody>{explicit_eq}
    </mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[7] += .008
  qpos[10] -= .004
  if order == 2:
    qpos[13] += .003
  qvel = np.linspace(-.08, .07, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  eqids = np.flatnonzero(
      np.asarray(model.eq_type) == int(mujoco.mjtEq.mjEQ_FLEXSTRAIN))
  assert eqids.size == (1 if shell or order == 2 else 2)
  to_device = lambda value: torch.as_tensor(
      value[None], dtype=torch.float32, device=device).contiguous()
  poses = {
      "body_pos": to_device(data.xpos), "body_quat": to_device(data.xquat),
      "joint_anchor": to_device(data.xanchor),
      "joint_axis": to_device(data.xaxis),
  }
  flex = MetalFlex(model, device=device)
  for compiled_eqid in eqids:
    expected_rows = np.flatnonzero(
        (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        & (data.efc_id == compiled_eqid))
    assert (flex.descriptor.equality_row_counts[int(compiled_eqid)]
            == expected_rows.size)
    np.testing.assert_array_equal(
        flex.descriptor.equality_row_ids_by_eqid[int(compiled_eqid)],
        np.arange(expected_rows.size, dtype=np.int32))
  flex.update_kinematics(poses, to_device(data.cvel))
  context = flex.capture_position_context()
  efc_values = np.asarray(data.efc_J)
  if efc_values.size == data.nefc * model.nv:
    efc_J = efc_values.reshape(data.nefc, model.nv)
  else:
    efc_J = np.zeros((data.nefc, model.nv), dtype=np.float64)
    for row in range(data.nefc):
      start, nnz = int(data.efc_J_rowadr[row]), int(data.efc_J_rownnz[row])
      cols = np.asarray(data.efc_J_colind[start:start + nnz], dtype=np.int64)
      efc_J[row, cols] = efc_values[start:start + nnz]
  raw_context = flex.flexstrain_position_context(context)
  assert set(raw_context) == set(map(int, eqids))
  unified_context = flex.flex_equality_position_context(context)
  assert set(unified_context) == set(map(int, eqids))
  for eqid in eqids:
    rows = np.flatnonzero(
        (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        & (data.efc_id == eqid))
    assert rows.size > 0
    actual = flex.run_flex_equality_rows(
        int(eqid), context, to_device(qpos), to_device(qvel))
    unified = unified_context[int(eqid)]
    assert unified["kind"] == "flexstrain"
    assert unified["jdot_included"] is False
    np.testing.assert_array_equal(actual["row_ids"],
                                  np.arange(rows.size, dtype=np.int32))
    np.testing.assert_allclose(unified["pos"].detach().cpu().numpy()[0],
                               data.efc_pos[rows], rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(unified["J"].detach().cpu().numpy()[0],
                               efc_J[rows], rtol=5e-5, atol=5e-7)
    ci, cj, ck = map(int, model.eq_data[int(eqid), :3])
    cy, cz = map(int, model.flex_cellnum[int(model.eq_obj1id[eqid]), 1:])
    order = abs(int(model.flex_interp[int(model.eq_obj1id[eqid])]))
    if shell:
      node_local = [li * (cz * order + 1) + lj
                    for li in range(order + 1)
                    for lj in range(order + 1)]
    else:
      node_local = [
          (ci * order + li) * (cy * order + 1) * (cz * order + 1)
          + (cj * order + lj) * (cz * order + 1) + ck * order + lk
          for li in range(order + 1)
          for lj in range(order + 1)
          for lk in range(order + 1)]
    node_start = int(model.flex_nodeadr[int(model.eq_obj1id[eqid])])
    node_bodies = np.asarray(model.flex_nodebodyid)[
        node_start + np.asarray(node_local, dtype=np.int64)]
    expected_diag = float(np.mean(
        np.asarray(model.body_invweight0).reshape(-1)[2 * node_bodies]))
    np.testing.assert_allclose(actual["diag"].detach().cpu().numpy()[0],
                               expected_diag, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(actual["pos"].detach().cpu().numpy()[0],
                               data.efc_pos[rows], rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(actual["J"].detach().cpu().numpy()[0],
                               efc_J[rows],
                               rtol=5e-5, atol=5e-7)
    np.testing.assert_allclose(actual["vel"].detach().cpu().numpy()[0],
                               data.efc_vel[rows], rtol=5e-5, atol=5e-7)
    np.testing.assert_allclose(actual["impedance"].detach().cpu().numpy()[0],
                               data.efc_KBIP[rows, 2], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual["R"].detach().cpu().numpy()[0],
                               data.efc_R[rows], rtol=4e-5, atol=1e-9)
    np.testing.assert_allclose(actual["aref"].detach().cpu().numpy()[0],
                               data.efc_aref[rows], rtol=5e-5, atol=5e-7)
    qvel2 = -0.4 * qvel
    refreshed = flex.run_flexstrain_equality_rows(
        int(eqid), to_device(qvel2), context)
    np.testing.assert_allclose(refreshed["pos"].detach().cpu().numpy()[0],
                               actual["pos"].detach().cpu().numpy()[0],
                               rtol=0, atol=0)
    np.testing.assert_allclose(refreshed["J"].detach().cpu().numpy()[0],
                               actual["J"].detach().cpu().numpy()[0],
                               rtol=0, atol=0)
    np.testing.assert_allclose(refreshed["vel"].detach().cpu().numpy()[0],
                               -0.4 * data.efc_vel[rows],
                               rtol=5e-5, atol=5e-7)
    np.testing.assert_allclose(
        raw_context[int(eqid)]["pos"].detach().cpu().numpy()[0],
        data.efc_pos[rows], rtol=3e-5, atol=3e-7)


@pytest.mark.parametrize("order", [1, 2])
def test_flexstrain_equality_rows_match_pinned_cpu(order):
  _flexstrain_equality_witness("cpu", order)


def test_flexstrain_equality_rows_match_pinned_sparse_jacobian():
  _flexstrain_equality_witness("cpu", 1, jacobian="sparse")


@pytest.mark.parametrize("order", [1, 2])
def test_flexstrain_shell_equality_rows_match_pinned_cpu(order):
  _flexstrain_equality_witness("cpu", order, shell=True)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native shell strain parity requires explicit GPU opt-in")
@pytest.mark.parametrize("order", [1, 2])
def test_native_flexstrain_shell_equality_rows_match_pinned_cpu(order):
  torch_runtime = pytest.importorskip("torch")
  if not torch_runtime.backends.mps.is_available():
    pytest.skip("native Metal shell strain equality requires MPS")
  _flexstrain_equality_witness("mps", order, shell=True)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native flexstrain parity requires explicit GPU opt-in")
@pytest.mark.parametrize("order", [1, 2])
def test_native_flexstrain_equality_rows_match_pinned_cpu(order):
  torch_runtime = pytest.importorskip("torch")
  if not torch_runtime.backends.mps.is_available():
    pytest.skip("native Metal flexstrain equality requires MPS")
  _flexstrain_equality_witness("mps", order)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native flex equality replay requires explicit GPU opt-in")
def test_native_cached_pos_flex_equalities_match_pinned_forward_skip():
  torch_runtime = pytest.importorskip("torch")
  if not torch_runtime.backends.mps.is_available():
    pytest.skip("native Metal flex equality replay requires MPS")
  _cached_flex_equality_witness("mps")


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
          <contact contype="0" conaffinity="0" selfcollide="none"/>
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


@pytest.mark.parametrize(("count", "dof", "elasticity"), [
    ("2 2 2", "trilinear", '<elasticity young="1000" poisson=".2" damping=".1"/>'),
    ("3 3 3", "quadratic", '<elasticity young="1000" poisson=".2" damping=".1"/>'),
    ("3 3 3", "trilinear", '<elasticity young="1000" poisson=".2" damping=".1" thickness=".01" elastic2d="bend"/>'),
    ("3 3 3", "quadratic", '<elasticity young="1000" poisson=".2" damping=".1" thickness=".01" elastic2d="bend"/>'),
])
def test_interpolated_and_shell_kinematics_match_pinned_cpu(
    count, dof, elasticity):
  """The contact/material vertex buffers use source mju_flex positions."""
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="interp" type="grid" count="{count}"
                spacing=".1 .1 .1" mass="1" dim="3" dof="{dof}">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        {elasticity}
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert abs(int(model.flex_interp[0])) == (1 if dof == "trilinear" else 2)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += .013
  qpos[-2] -= .009
  qvel = np.linspace(-.04, .05, model.nv)
  data, flex, _, _, _ = _evaluate(model, qpos, qvel)
  np.testing.assert_allclose(
      flex._flexvert_xpos.detach().cpu().numpy()[0], data.flexvert_xpos,
      rtol=2e-6, atol=2e-7)
  eps = 2e-4
  qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
  mujoco.mj_integratePos(model, qpos_hi, qvel, eps)
  mujoco.mj_integratePos(model, qpos_lo, qvel, -eps)
  edge = np.asarray(model.flex_edge, dtype=np.int64).reshape(-1, 2)
  def edge_lengths(position):
    return np.linalg.norm(position[edge[:, 1]] - position[edge[:, 0]], axis=1)
  data_hi, data_lo = mujoco.MjData(model), mujoco.MjData(model)
  data_hi.qpos[:], data_lo.qpos[:] = qpos_hi, qpos_lo
  mujoco.mj_forward(model, data_hi)
  mujoco.mj_forward(model, data_lo)
  numerical_edot = (edge_lengths(data_hi.flexvert_xpos)
                    - edge_lengths(data_lo.flexvert_xpos)) / (2 * eps)
  # MuJoCo deliberately leaves the public compiled flexedge_velocity zero
  # for interpolated and rigid flexes. The geometric length rate still exists
  # for interpolation/material damping, so validate it from vertex velocities
  # separately from that pinned public field.
  np.testing.assert_allclose(
      flex.flexedge_velocity.detach().cpu().numpy()[0],
      data.flexedge_velocity, rtol=3e-5, atol=3e-6)
  vertex_velocity = flex._flexvert_xvel.detach().cpu().numpy()[0]
  edge_direction = flex._flexedge_dir.detach().cpu().numpy()[0]
  kinematic_edot = np.einsum(
      "ij,ij->i", edge_direction,
      vertex_velocity[edge[:, 1]] - vertex_velocity[edge[:, 0]])
  assert np.max(np.abs(numerical_edot)) > 1e-5
  np.testing.assert_allclose(kinematic_edot, numerical_edot,
                             rtol=3e-3, atol=2e-5)
  # Check the exact generalized Jacobian consumed by flex force/contact
  # stages, independently perturbing the configuration through MuJoCo's
  # manifold-aware integration routine.
  vertex_jac = flex._flexvert_J.detach().cpu().numpy()[0]
  for dof in sorted(set((0, model.nv // 2, model.nv - 1))):
    delta = np.zeros(model.nv, dtype=np.float64)
    delta[dof] = eps
    qpos_hi, qpos_lo = qpos.copy(), qpos.copy()
    mujoco.mj_integratePos(model, qpos_hi, delta, 1.0)
    mujoco.mj_integratePos(model, qpos_lo, delta, -1.0)
    data_hi, data_lo = mujoco.MjData(model), mujoco.MjData(model)
    data_hi.qpos[:], data_lo.qpos[:] = qpos_hi, qpos_lo
    mujoco.mj_forward(model, data_hi)
    mujoco.mj_forward(model, data_lo)
    numerical = (data_hi.flexvert_xpos - data_lo.flexvert_xpos) / (2 * eps)
    np.testing.assert_allclose(
        vertex_jac[:, :, dof], numerical, rtol=4e-3, atol=3e-5)


def test_interpolated_node_velocity_clears_on_position_only_update():
  if torch is None:
    pytest.skip("flex tensor lifecycle check requires prepared Torch runtime")
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="q2" type="grid" count="3 3 3"
                spacing=".1 .1 .1" mass="1" dim="3" dof="quadratic">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-.03, .04, model.nv)
  mujoco.mj_forward(model, data)
  _, flex, _, _, _ = _evaluate(model, data.qpos, data.qvel)
  assert torch.count_nonzero(flex._node_xvel).item() > 0
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "root_com": torch.tensor(_root_com(model, data)[None], dtype=torch.float32),
  }
  flex.update_kinematics(poses, cvel=None)
  assert torch.count_nonzero(flex._node_xvel).item() == 0
  assert torch.count_nonzero(flex._flexvert_xvel).item() == 0


def test_interpolated_material_operators_are_separate_and_reused():
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
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  root_com = _root_com(model, data)
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(root_com[None], dtype=torch.float32),
  }
  flex = MetalFlex(model, device="cpu")
  operators = flex.run_material_operators(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(data.qvel[None], dtype=torch.float32), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32))
  assert set(operators) == {
      "interp_stiffness", "interp_damped_stiffness", "bend_stiffness",
      "bend_damped_stiffness", "edge_velocity_derivative"}
  interp = operators["interp_stiffness"]
  damped = operators["interp_damped_stiffness"]
  assert interp.shape == (1, model.nv, model.nv)
  assert interp.dtype == torch.float32 and interp.is_contiguous()
  assert float(torch.linalg.vector_norm(interp)) > 1.0
  torch.testing.assert_close(interp, interp.transpose(1, 2), rtol=1e-5, atol=2e-5)
  torch.testing.assert_close(damped, 0.1 * interp, rtol=2e-5, atol=2e-6)
  ptr = interp.data_ptr()
  second = flex.run_material_operators(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(data.qvel[None], dtype=torch.float32), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32))
  assert second["interp_stiffness"].data_ptr() == ptr
  vector = torch.linspace(-0.4, 0.7, model.nv, dtype=torch.float32)[None]
  context = flex.capture_material_operator_context(
      poses, torch.tensor(data.cvel[None], dtype=torch.float32))
  matrix_free = flex.apply_material_operator_context_device(context, vector)
  for name in ("interp_stiffness", "interp_damped_stiffness",
               "bend_stiffness", "bend_damped_stiffness"):
    expected_product = torch.bmm(second[name], vector.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(matrix_free[name], expected_product,
                               rtol=4e-5, atol=4e-5)
  flex.update_kinematics(poses, torch.tensor(data.cvel[None], dtype=torch.float32))
  with pytest.raises(ValueError, match="stale"):
    flex.apply_material_operator_context_device(context, vector)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  disabled_ops = MetalFlex(model, device="cpu").run_material_operators(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(data.qvel[None], dtype=torch.float32), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32))
  # MuJoCo's implicit flex correction does not apply passive spring/damper
  # disable bits to the frozen interpolation K/Kd operators. The separate
  # flex-edge qDeriv path does honor the damper disable bit.
  for name in ("interp_stiffness", "interp_damped_stiffness",
               "bend_stiffness", "bend_damped_stiffness"):
    torch.testing.assert_close(disabled_ops[name], second[name])
  assert not torch.any(disabled_ops["edge_velocity_derivative"])


def test_matrix_free_standard_shell_bend_matches_dense_stencil_product():
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" jacobian="dense"/><worldbody>
      <flexcomp name="shell" type="grid" count="3 3 1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="0" conaffinity="0"/>
        <elasticity young="3000" poisson=".2" damping=".15"
                    thickness=".02" elastic2d="bend"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert model.flex_bending.size > 0
  data = mujoco.MjData(model)
  data.qpos[3] += .02
  mujoco.mj_forward(model, data)
  root_com = _root_com(model, data)
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(root_com[None], dtype=torch.float32),
  }
  cvel = torch.tensor(data.cvel[None], dtype=torch.float32)
  flex = MetalFlex(model, device="cpu")
  dense = flex.run_material_operators(
      torch.tensor(data.qpos[None], dtype=torch.float32),
      torch.tensor(data.qvel[None], dtype=torch.float32), poses, cvel)
  assert flex._bend_count > 0
  assert float(torch.linalg.vector_norm(dense["bend_stiffness"])) > 0.0
  index = torch.arange(model.nv, dtype=torch.float32)
  vector = (torch.sin(index * 1.17) + .03 * (index.remainder(7) ** 2))[None]
  context = flex.capture_material_operator_context(poses, cvel)
  products = flex.apply_material_operator_context_device(context, vector)
  assert float(torch.linalg.vector_norm(products["bend_stiffness"])) > 1e-4
  for name in ("interp_stiffness", "interp_damped_stiffness",
               "bend_stiffness", "bend_damped_stiffness"):
    expected = torch.bmm(dense[name], vector.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(products[name], expected,
                               rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(("count", "dim", "dof", "elasticity"), [
    ("3 3 3", 3, "quadratic",
     '<elasticity young="1000" poisson=".2" damping=".1"/>'),
])
def test_matrix_free_quadratic_material_matches_dense_product(
    count, dim, dof, elasticity):
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" jacobian="dense"/><worldbody>
      <flexcomp name="q2" type="grid" count="{count}"
                spacing=".1 .1 .1" mass="1" dim="{dim}" dof="{dof}">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        {elasticity}
      </flexcomp>
    </worldbody></mujoco>
  """)
  assert abs(int(model.flex_interp[0])) == 2
  data = mujoco.MjData(model)
  data.qpos[3] += .01
  mujoco.mj_forward(model, data)
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(_root_com(model, data)[None],
                               dtype=torch.float32),
  }
  cvel = torch.tensor(data.cvel[None], dtype=torch.float32)
  flex = MetalFlex(model, device="cpu")
  dense = flex.run_material_operators(
      torch.tensor(data.qpos[None], dtype=torch.float32),
      torch.tensor(data.qvel[None], dtype=torch.float32), poses, cvel)
  assert flex._interp_count > 0
  index = torch.arange(model.nv, dtype=torch.float32)
  vector = (torch.sin(index * 1.17) + .03 * (index.remainder(7) ** 2))[None]
  context = flex.capture_material_operator_context(poses, cvel)
  products = flex.apply_material_operator_context_device(context, vector)
  for name in ("interp_stiffness", "interp_damped_stiffness",
               "bend_stiffness", "bend_damped_stiffness"):
    expected = torch.bmm(dense[name], vector.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(products[name], expected,
                               rtol=4e-5, atol=5e-5)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="flex edge COO kernel requires explicit GPU opt-in")
def test_native_flex_edge_velocity_derivative_coo_matches_dense_operator():
  torch_runtime = pytest.importorskip("torch")
  if not torch_runtime.backends.mps.is_available():
    pytest.skip("native flex edge COO qualification requires MPS")
  from mujoco_metal.flex import MetalFlex
  from mujoco_metal.velocity_derivative import (
      MetalVelocityDerivativeValues, compile_velocity_derivative_layout)

  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" timestep=".002" jacobian="dense"/>
      <worldbody><flexcomp name="cable" type="grid" count="4 1 1"
                spacing=".1 .1 .1" mass=".5" radius=".01" dim="1">
        <edge equality="false" stiffness="0" damping=".4"/>
        <elasticity young="0" poisson=".2" damping=".4"/>
      </flexcomp></worldbody></mujoco>
  """)
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-.2, .3, model.nv)
  mujoco.mj_forward(model, data)
  poses = {
      "body_pos": torch_runtime.tensor(data.xpos[None].repeat(2, axis=0),
                                       dtype=torch_runtime.float32,
                                       device="mps").contiguous(),
      "body_quat": torch_runtime.tensor(data.xquat[None].repeat(2, axis=0),
                                        dtype=torch_runtime.float32,
                                        device="mps").contiguous(),
      "joint_anchor": torch_runtime.tensor(data.xanchor[None].repeat(2, axis=0),
                                           dtype=torch_runtime.float32,
                                           device="mps").contiguous(),
      "joint_axis": torch_runtime.tensor(data.xaxis[None].repeat(2, axis=0),
                                         dtype=torch_runtime.float32,
                                         device="mps").contiguous(),
      "root_com": torch_runtime.tensor(
          _root_com(model, data)[None].repeat(2, axis=0),
          dtype=torch_runtime.float32, device="mps").contiguous(),
  }
  cvel = torch_runtime.tensor(data.cvel[None].repeat(2, axis=0),
                              dtype=torch_runtime.float32,
                              device="mps").contiguous()
  qpos = torch_runtime.tensor(model.qpos0[None].repeat(2, axis=0),
                              dtype=torch_runtime.float32,
                              device="mps").contiguous()
  qvel = torch_runtime.tensor(data.qvel[None].repeat(2, axis=0),
                              dtype=torch_runtime.float32,
                              device="mps").contiguous()
  flex = MetalFlex(model, device="mps", batch_size=2)
  dense = flex.run_material_operators(qpos, qvel, poses, cvel)[
      "edge_velocity_derivative"]
  layout = compile_velocity_derivative_layout(model)
  values = torch_runtime.zeros(
      (2, max(layout.edge_count, 1)), dtype=torch_runtime.float32,
      device="mps")
  writer = MetalVelocityDerivativeValues(layout, 2, values)
  writer.clear_device()
  result = flex.run_edge_velocity_derivative_coo_device(
      poses, writer, cvel=cvel)
  edge_rows = torch_runtime.as_tensor(
      np.array(layout.edge_rows, dtype=np.int64, copy=True),
      dtype=torch_runtime.long, device="mps")
  edge_cols = torch_runtime.as_tensor(
      np.array(layout.edge_cols, dtype=np.int64, copy=True),
      dtype=torch_runtime.long, device="mps")
  expected = dense[:, edge_rows, edge_cols]
  torch_runtime.testing.assert_close(
      result[:, :layout.edge_count], expected,
      rtol=3e-5, atol=3e-6)


def test_pinned_implicit_flex_correction_survives_passive_disable_flags():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0" integrator="implicit"
                    timestep=".002"/><worldbody>
      <flexcomp name="q1" type="grid" count="2 2 2"
                spacing=".1 .1 .1" mass="1" dim="3" dof="trilinear">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody></mujoco>
  """)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-0.2, 0.3, model.nv)
  mujoco.mj_forward(model, data)
  np.testing.assert_array_equal(data.qfrc_passive, np.zeros(model.nv))
  assert not np.any(data.qacc)
  qvel0 = np.asarray(data.qvel).copy()
  mujoco.mj_step(model, data)
  # The passive force is disabled, but pinned flexInterp_cgsolve still applies
  # compiled K/Kd in its velocity correction.
  assert np.max(np.abs(data.qvel - qvel0)) > 1e-5


def test_attached_trilinear_q1_material_uses_pinned_direct_dof_path():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="arm" pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="attached" type="grid" count="2 2 2"
                  pos=".13 .21 -.07" spacing=".1 .1 .1" mass="1"
                  dim="3" dof="trilinear">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <edge stiffness="0" damping="0"/>
          <elasticity young="1000" poisson=".2" damping=".4"/>
        </flexcomp>
      </body>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[0] = 0.24
  qpos[1] += 0.01
  qvel = np.zeros(model.nv)
  data, flex, force, _, tangent = _evaluate(model, qpos, qvel)
  assert model.flex_interp[0] == 1
  np.testing.assert_allclose(force, data.qfrc_passive, rtol=5e-4, atol=3e-5)
  eps = 1e-3
  qhi, qlo = qpos.copy(), qpos.copy()
  qhi[1] += eps
  qlo[1] -= eps
  numerical = (_oracle_passive_force(model, qhi, qvel)
               - _oracle_passive_force(model, qlo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      tangent.detach().numpy()[0, :, 1], numerical,
      rtol=3e-2, atol=3e-3)


def test_mps_attached_trilinear_q1_material_uses_pinned_direct_dof_path():
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="arm" pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <flexcomp name="attached" type="grid" count="2 2 2"
                  pos=".13 .21 -.07" spacing=".1 .1 .1" mass="1"
                  dim="3" dof="trilinear">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <edge stiffness="0" damping="0"/>
          <elasticity young="1000" poisson=".2" damping=".4"/>
        </flexcomp>
      </body>
    </worldbody></mujoco>
  """)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[0] = 0.24
  qpos[1] += 0.01
  qvel = np.zeros(model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32, device="mps"),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32, device="mps"),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32, device="mps"),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32, device="mps"),
      "root_com": torch.tensor(_root_com(model, data)[None],
                                dtype=torch.float32, device="mps"),
  }
  flex = MetalFlex(model, device="mps")
  force, _, tangent = flex.run_device(
      torch.tensor(qpos[None], dtype=torch.float32, device="mps"),
      torch.tensor(qvel[None], dtype=torch.float32, device="mps"), poses,
      torch.tensor(data.cvel[None], dtype=torch.float32, device="mps"))
  np.testing.assert_allclose(
      force.detach().cpu().numpy()[0], data.qfrc_passive,
      rtol=5e-4, atol=3e-5)
  eps = 1e-3
  qhi, qlo = qpos.copy(), qpos.copy()
  qhi[1] += eps
  qlo[1] -= eps
  numerical = (_oracle_passive_force(model, qhi, qvel)
               - _oracle_passive_force(model, qlo, qvel)) / (2*eps)
  np.testing.assert_allclose(
      tangent.detach().cpu().numpy()[0, :, 1], numerical,
      rtol=3e-2, atol=3e-3)


def test_mps_interpolated_material_operator_workspace_matches_cpu():
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
  qvel = np.zeros(model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  root_com = _root_com(model, data)
  poses_cpu = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(root_com[None], dtype=torch.float32),
  }
  poses_mps = {name: value.to("mps") for name, value in poses_cpu.items()}
  cvel_cpu = torch.tensor(data.cvel[None], dtype=torch.float32)
  cvel_mps = cvel_cpu.to("mps")
  qpos_cpu = torch.tensor(qpos[None], dtype=torch.float32)
  qvel_cpu = torch.tensor(qvel[None], dtype=torch.float32)
  cpu_ops = MetalFlex(model, device="cpu").run_material_operators(
      qpos_cpu, qvel_cpu, poses_cpu, cvel_cpu)
  native_ops = MetalFlex(model, device="mps").run_material_operators(
      qpos_cpu.to("mps"), qvel_cpu.to("mps"), poses_mps, cvel_mps)
  for name, expected in cpu_ops.items():
    actual = native_ops[name]
    assert actual.device.type == "mps" and actual.is_contiguous()
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected.detach().numpy(),
        rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(("name", "count", "dim", "dof", "elasticity",
                         "nonzero_operator"), [
    ("q1-volume", "2 2 2", 3, "trilinear",
     '<elasticity young="1000" poisson=".2" damping=".1"/>',
     "interp_stiffness"),
    ("q2-volume", "3 3 3", 3, "quadratic",
     '<elasticity young="1000" poisson=".2" damping=".1"/>',
     "interp_stiffness"),
    ("q1-shell", "3 3 3", 3, "trilinear",
     '<elasticity young="1000" poisson=".2" damping=".1" thickness=".01" elastic2d="stretch"/>',
     None),
    ("q2-shell", "3 3 3", 3, "quadratic",
     '<elasticity young="1000" poisson=".2" damping=".1" thickness=".01" elastic2d="stretch"/>',
     None),
    ("ordinary-bend", "3 3 1", 2, None,
     '<elasticity young="3000" poisson=".2" damping=".15" thickness=".02" elastic2d="bend"/>',
     "bend_stiffness"),
])
def test_cpu_matrix_free_flex_operator_families_match_dense_oracle(
    name, count, dim, dof, elasticity, nonzero_operator):
  MetalFlex, _ = _flex_api()
  dof_xml = f' dof="{dof}"' if dof is not None else ""
  xml = f'''<mujoco><option gravity="0 0 0"/><worldbody>
    <flexcomp name="material" type="grid" count="{count}"
      spacing=".1 .1 .1" mass="1" dim="{dim}"{dof_xml}>
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      {elasticity}
    </flexcomp></worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  qpos[3] += .012
  qpos[-1] -= .007
  qvel = np.linspace(-.03, .04, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  poses = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(_root_com(model, data)[None], dtype=torch.float32),
  }
  cvel = torch.tensor(data.cvel[None], dtype=torch.float32)
  flex = MetalFlex(model, device="cpu")
  oracle = flex.run_material_operators(
      torch.tensor(qpos[None], dtype=torch.float32),
      torch.tensor(qvel[None], dtype=torch.float32), poses, cvel)
  context = flex.capture_material_operator_context(poses, cvel)
  index = torch.arange(model.nv, dtype=torch.float32)
  vector = (torch.sin(index * 1.17) + .03 * (index.remainder(7) ** 2))[None]
  product = flex.apply_material_operator_context_device(context, vector)
  names = ("interp_stiffness", "interp_damped_stiffness",
           "bend_stiffness", "bend_damped_stiffness")
  for key in names:
    expected = torch.bmm(oracle[key], vector.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(product[key], expected, rtol=4e-5, atol=5e-5)
  if nonzero_operator is None:
    assert all(torch.count_nonzero(oracle[key]).item() == 0 for key in names), name
    assert all(torch.count_nonzero(product[key]).item() == 0 for key in names), name
    adr = int(model.flex_stiffnessadr[0])
    assert adr >= 0 and model.flex_stiffness[adr] == 0.0
  else:
    assert float(torch.linalg.vector_norm(product[nonzero_operator])) > 1e-4, name
  flex.update_kinematics(poses, cvel)
  with pytest.raises(ValueError, match="stale"):
    flex.apply_material_operator_context_device(context, vector)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="matrix-free flex operator qualification is opt-in")
@pytest.mark.parametrize(("name", "count", "dim", "dof", "elasticity",
                         "nonzero_operator"), [
    ("q1-volume", "2 2 2", 3, "trilinear",
     '<elasticity young="1000" poisson=".2" damping=".1"/>',
     "interp_stiffness"),
    ("q2-volume", "3 3 3", 3, "quadratic",
     '<elasticity young="1000" poisson=".2" damping=".1"/>',
     "interp_stiffness"),
    ("q1-shell", "3 3 3", 3, "trilinear",
     '<elasticity young="1000" poisson=".2" damping=".1" thickness=".01" elastic2d="stretch"/>',
     None),
    ("q2-shell", "3 3 3", 3, "quadratic",
     '<elasticity young="1000" poisson=".2" damping=".1" thickness=".01" elastic2d="stretch"/>',
     None),
    ("ordinary-bend", "3 3 1", 2, None,
     '<elasticity young="3000" poisson=".2" damping=".15" thickness=".02" elastic2d="bend"/>',
     "bend_stiffness"),
])
def test_mps_matrix_free_flex_material_products_match_dense_oracle(
    name, count, dim, dof, elasticity, nonzero_operator):
  if torch is None or not torch.backends.mps.is_available():
    pytest.skip("Apple MPS is unavailable in this test process")
  MetalFlex, _ = _flex_api()
  flexcomp = f'''<flexcomp name="material" type="grid" count="{count}"
      spacing=".1 .1 .1" mass="1" dim="{dim}"'''
  if dof is not None:
    flexcomp += f' dof="{dof}"'
  flexcomp += f'''>
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      {elasticity}
    </flexcomp>'''
  model = mujoco.MjModel.from_xml_string(
      f'<mujoco><option gravity="0 0 0"/><worldbody>{flexcomp}</worldbody></mujoco>')
  qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
  if model.nv:
    qpos[3] += .012
    qpos[-1] -= .007
  qvel = np.linspace(-.03, .04, model.nv)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  poses_cpu = {
      "body_pos": torch.tensor(data.xpos[None], dtype=torch.float32),
      "body_quat": torch.tensor(data.xquat[None], dtype=torch.float32),
      "joint_anchor": torch.tensor(data.xanchor[None], dtype=torch.float32),
      "joint_axis": torch.tensor(data.xaxis[None], dtype=torch.float32),
      "root_com": torch.tensor(_root_com(model, data)[None], dtype=torch.float32),
  }
  cvel_cpu = torch.tensor(data.cvel[None], dtype=torch.float32)
  qpos_cpu = torch.tensor(qpos[None], dtype=torch.float32)
  qvel_cpu = torch.tensor(qvel[None], dtype=torch.float32)
  oracle = MetalFlex(model, device="cpu").run_material_operators(
      qpos_cpu, qvel_cpu, poses_cpu, cvel_cpu)
  flex = MetalFlex(model, device="mps")
  poses_mps = {key: value.to("mps") for key, value in poses_cpu.items()}
  cvel_mps = cvel_cpu.to("mps")
  context = flex.capture_material_operator_context(poses_mps, cvel_mps)
  index = torch.arange(model.nv, dtype=torch.float32)
  vector_cpu = (torch.sin(index * 1.17) + .03 * (index.remainder(7) ** 2))[None]
  vector_mps = vector_cpu.to("mps")
  result = flex.apply_material_operator_context_device(context, vector_mps)
  names = ("interp_stiffness", "interp_damped_stiffness",
           "bend_stiffness", "bend_damped_stiffness")
  total_norm = 0.0
  for key in names:
    expected = torch.bmm(oracle[key], vector_cpu.unsqueeze(-1)).squeeze(-1)
    actual = result[key]
    assert tuple(actual.shape) == (1, model.nv)
    assert actual.device.type == "mps" and actual.is_contiguous()
    np.testing.assert_allclose(actual.detach().cpu().numpy(),
                               expected.numpy(), rtol=5e-4, atol=5e-4)
    total_norm += float(torch.linalg.vector_norm(actual).detach().cpu())
  if nonzero_operator is None:
    assert total_norm == 0.0, name
    for key in names:
      assert torch.count_nonzero(oracle[key]).item() == 0, (name, key)
    # Pinned mjd_flexInterp_mul skips signed shell flexes with a zero compiled
    # element matrix; mjd_flexBend_mul separately skips every interp != 0.
    owner = int(model.flex_stiffnessadr[0])
    assert owner >= 0 and model.flex_stiffness[owner] == 0.0
  else:
    assert float(torch.linalg.vector_norm(result[nonzero_operator]).detach().cpu()) > 1e-4, name
  pointers = {key: result[key].data_ptr() for key in names}
  second = flex.apply_material_operator_context_device(context, vector_mps)
  assert {key: second[key].data_ptr() for key in names} == pointers
  flex.update_kinematics(poses_mps, cvel_mps)
  with pytest.raises(ValueError, match="stale"):
    flex.apply_material_operator_context_device(context, vector_mps)


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
