# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Multiworld bounds regression for coupled joint-force outputs."""

import os

import mujoco
import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                       reason="opt-in GPU"),
]


def _host(value):
  return value.detach().cpu().numpy().copy()


def _model(nv):
  bodies = []
  for index in range(nv):
    joint = f'<joint name="j{index}" type="slide" axis="1 0 0"/>'
    if index == 0:
      joint += '<geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>'
    else:
      # Keep every independent slider dynamically valid without adding more
      # collision rows to the force-output regression.
      joint += '<inertial pos="0 0 0" mass="1" diaginertia=".001 .001 .001"/>'
    bodies.append(
        f'<body pos="{2 * index} 0 0">{joint}</body>')
  return mujoco.MjModel.from_xml_string(
      '<mujoco><option gravity="0 0 0" solver="PGS" iterations="100" '
      'tolerance="1e-8"/><worldbody>' + ''.join(bodies) +
      '</worldbody><equality><joint joint1="j0" joint2="j1" '
      'polycoef="0 1 0 0 0"/></equality></mujoco>')


@pytest.mark.parametrize("batch", [2, 3])
@pytest.mark.parametrize("nv", [2, 40])
def test_joint_force_world_slices_preserve_neighbors_and_pinned_rows(batch, nv):
  """Dense and block kernels keep all worlds in their own output intervals."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model = _model(nv)
  limits = CapacityLimits(max_nv=40, max_rows=256, max_batch=3)
  qpos = np.tile(model.qpos0, (batch, 1)).astype(np.float32)
  qvel = np.zeros((batch, nv), dtype=np.float32)
  qpos[:, 0] = np.asarray([.15, -.3, .45][:batch], dtype=np.float32)
  qpos[:, 1] = np.asarray([-.1, .2, .05][:batch], dtype=np.float32)
  qvel[:, 0] = np.asarray([.1, -.3, .25][:batch], dtype=np.float32)
  qvel[:, 1] = np.asarray([-.2, .15, -.1][:batch], dtype=np.float32)
  qpos_device = torch.as_tensor(qpos, dtype=torch.float32, device="mps")
  qvel_device = torch.as_tensor(qvel, dtype=torch.float32, device="mps")
  descriptor = load_model(model)
  poses = MetalKinematics(descriptor, batch).run_device(qpos_device)
  dynamics = MetalSmoothDynamics(descriptor, batch).run_device(
      qpos_device, qvel_device)
  solver = MetalCoupledConstraints(model, batch, limits=limits)
  assert solver.descriptor.dense_path == (nv == 2)

  # Put guard words immediately after both output views. The previous bug
  # applied the world offset twice after the per-world joint-force pointer had
  # already been advanced, allowing later worlds to overwrite this tail.
  workspace = solver._workspace
  backing = workspace["force_outputs"]
  joint_force = workspace["out_joint_force"]
  guard = torch.full((32,), 12345.0, dtype=torch.float32, device="mps")
  guarded = torch.cat((backing, guard))
  workspace["force_outputs"] = guarded
  workspace["out_contact_force"] = guarded[:workspace["out_contact_force"].numel()]
  joint_start = workspace["out_contact_force"].numel()
  workspace["out_joint_force"] = guarded[
      joint_start:joint_start + joint_force.numel()]
  guard_view = guarded[backing.numel():]

  result = solver.run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"],
      qpos_device, qvel_device)
  assert torch.all(result["status"] == 0)
  torch.testing.assert_close(
      guard_view, torch.full_like(guard_view, 12345.0), rtol=0, atol=0)

  references = []
  for world in range(batch):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    references.append(data)
  expected_qacc = np.stack([data.qacc for data in references])
  np.testing.assert_allclose(_host(result["qacc"]), expected_qacc,
                             rtol=5e-4, atol=3e-5)

  row = int(solver.descriptor.eq_rowadr[0])
  native_force = _host(result["joint_force"])
  expected_force = np.stack([
      data.efc_force[
          data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)]
      for data in references])
  assert expected_force.shape == (batch, 1)
  np.testing.assert_allclose(native_force[:, row:row + 1], expected_force,
                             rtol=5e-4, atol=3e-5)
  assert np.ptp(native_force[:, row]) > 1e-3
  expected_j = np.stack([
      data.efc_J.reshape(data.nefc, nv)[
          data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)][0]
      for data in references])
  np.testing.assert_allclose(_host(result["J"][:, row]), expected_j,
                             rtol=2e-5, atol=2e-6)
