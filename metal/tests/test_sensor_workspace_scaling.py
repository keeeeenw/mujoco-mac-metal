# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compiled-size sensor workspaces, with independent pinned-engine inputs."""

import os
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from mujoco_metal.sensors import lower_sensors, sensor_workspace_sizes


def _large_model():
  bodies = "".join(f'''<body name="b{i}" pos="{i*.1} 0 1">
    <joint type="hinge" axis="0 1 0" damping=".1"/>
    <geom type="box" size=".02 .03 .04" pos=".1 .02 0" mass="{.1+i*.001}"
          contype="0" conaffinity="0"/>
    <site name="s{i}" pos=".1 .01 0"/></body>''' for i in range(72))
  return mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity=".1 .2 -9.81"/>
    <worldbody>{bodies}</worldbody><sensor>
    <subtreecom body="b71"/><subtreelinvel body="b71"/>
    <subtreeangmom body="b71"/><subtreecom body="world"/>
    <subtreelinvel body="world"/><subtreeangmom body="world"/>
    <accelerometer site="s71"/><force site="s71"/><torque site="s71"/>
    <framelinacc objtype="body" objname="b71"/>
    <frameangacc objtype="body" objname="b71"/>
    </sensor></mujoco>''')


def test_lowering_above_old_body_and_dof_caps_and_workspace_overflow():
  model = _large_model()
  descriptor = lower_sensors(model)
  assert descriptor.nbody == 73 and descriptor.nv == 72
  sizes = sensor_workspace_sizes(descriptor, 2)
  assert sizes["subtree_runtime"] == 4+2*73*32
  assert sizes["com_scratch"] == 2*73*12
  with pytest.raises(ValueError, match="workspace.*int32"):
    sensor_workspace_sizes(replace(descriptor, nbody=1 << 30), 2)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_native_sensor_workspaces_above_old_caps_match_pinned_and_reuse():
  import torch
  from mujoco_metal.sensors import SensorProgram
  model = _large_model()
  program = SensorProgram(model, 2)
  scratch_ptr = program._s_subtree_runtime.data_ptr()
  com_ptr = program._rne_com_scratch.data_ptr()
  for replay in range(2):
    q, v, acc, expected = [], [], [], []
    pose_rows = {key: [] for key in ("body_pos", "body_quat", "inertial_pos",
        "inertial_quat", "site_pos", "site_quat", "geom_pos", "geom_quat",
        "joint_anchor", "joint_axis", "cvel", "root_com")}
    for world in range(2):
      data = mujoco.MjData(model)
      data.qpos[:] = np.linspace(-.4, .7, model.nq)+world*.03+replay*.02
      data.qvel[:] = np.linspace(.2, -.6, model.nv)+world*.04
      mujoco.mj_forward(model, data)
      q.append(data.qpos.copy()); v.append(data.qvel.copy())
      acc.append(data.qacc.copy()); expected.append(data.sensordata.copy())
      arrays = dict(body_pos=data.xpos, body_quat=data.xquat,
          inertial_pos=data.xipos, inertial_quat=data.xiquat,
          site_pos=data.site_xpos, site_quat=np.array([
              _mat_quat(mat) for mat in data.site_xmat]),
          geom_pos=data.geom_xpos, geom_quat=np.array([
              _mat_quat(mat) for mat in data.geom_xmat]),
          joint_anchor=data.xanchor, joint_axis=data.xaxis, cvel=data.cvel,
          root_com=data.subtree_com)
      for key in pose_rows:
        pose_rows[key].append(np.array(arrays[key], copy=True))
    def tensor(value):
      return torch.tensor(np.asarray(value), dtype=torch.float32, device="mps")
    poses = {key: tensor(value) for key, value in pose_rows.items()}
    qp, qv = tensor(q), tensor(v)
    got = program.run_state_device(qp, qv, poses)
    got = program.run_acc_device(qp, qv, tensor(acc), poses, out=got)
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=5e-4, atol=5e-4)
    assert scratch_ptr == program._s_subtree_runtime.data_ptr()
    assert com_ptr == program._rne_com_scratch.data_ptr()


def _mat_quat(mat):
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, mat)
  return quat
