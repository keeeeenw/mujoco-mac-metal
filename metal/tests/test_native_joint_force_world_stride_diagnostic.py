# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Observational companion for the guarded block joint-force regression."""

import json
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


@pytest.mark.parametrize("batch", [2, 3])
def test_capture_block_joint_force_world_stride_diagnostics(batch):
  """Capture assembly, optimizer and final acceleration for a block solve."""
  torch = pytest.importorskip("torch")
  from mujoco_metal.capacity import CapacityLimits
  from mujoco_metal.coupled_constraints import MetalCoupledConstraints
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  nv = 40
  bodies = []
  for index in range(nv):
    joint = f'<joint name="j{index}" type="slide" axis="1 0 0"/>'
    if index == 0:
      joint += '<geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>'
    else:
      joint += '<inertial pos="0 0 0" mass="1" diaginertia=".001 .001 .001"/>'
    bodies.append(f'<body pos="{2 * index} 0 0">{joint}</body>')
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><option gravity="0 0 0" solver="PGS" iterations="100" '
      'tolerance="1e-8"/><worldbody>' + ''.join(bodies) +
      '</worldbody><equality><joint joint1="j0" joint2="j1" '
      'polycoef="0 1 0 0 0"/></equality></mujoco>')
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
  solver = MetalCoupledConstraints(
      model, batch,
      limits=CapacityLimits(max_nv=40, max_rows=256, max_batch=3))
  result = solver.run_device(
      poses, dynamics["mass_matrix"], -dynamics["qfrc_bias"],
      qpos_device, qvel_device)
  nr = int(solver.descriptor.nr)
  debug = solver._workspace["workspace_debug"].reshape(
      batch, int(solver._debug_stride))
  R = debug[:, nr * nr:nr * nr + nr]
  lambdas = debug[:, nr * nr + 3 * nr:nr * nr + 4 * nr]
  active = debug[:, nr * nr + 6 * nr:nr * nr + 7 * nr]
  eq_row = int(solver.descriptor.eq_rowadr[0])
  cpu = []
  for world in range(batch):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[world]
    data.qvel[:] = qvel[world]
    mujoco.mj_forward(model, data)
    cpu.append({
        "qacc": np.array(data.qacc).tolist(),
        "qfrc_constraint": np.array(data.qfrc_constraint).tolist(),
        "eq_force": float(data.efc_force[
            data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)][0]),
        "eq_J": np.array(data.efc_J.reshape(data.nefc, nv)[
            data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)][0]).tolist(),
    })
  payload = {
      "batch": batch,
      "status": _host(result["status"]).tolist(),
      "qacc": _host(result["qacc"]).tolist(),
      "qfrc_constraint": _host(result["qfrc_constraint"]).tolist(),
      "joint_force_eq": _host(result["joint_force"][:, eq_row:eq_row + 1]).tolist(),
      "J_eq": _host(result["J"][:, eq_row]).tolist(),
      "diagnostics": _host(result["solver_diagnostics"]).tolist(),
      "lambda_eq": _host(lambdas[:, eq_row:eq_row + 1]).tolist(),
      "solver_workspace_eq": _host(R[:, eq_row:eq_row + 1]).tolist(),
      "enabled_eq": _host(active[:, eq_row:eq_row + 1]).tolist(),
      "cpu": cpu,
  }
  print("BLOCK_JOINT_FORCE_DIAGNOSTIC=" + json.dumps(payload, sort_keys=True))
  assert payload["status"] == [0] * batch
