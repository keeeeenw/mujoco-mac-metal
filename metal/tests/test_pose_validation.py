# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Boundary and trusted-internal pose validation contracts."""

import os

import mujoco
import numpy as np
import pytest


def _pose_fixture():
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import validate_pose_dict

  model = load_model(mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><joint type="slide"/><geom type="sphere" size=".1" mass="1"/>
      <site name="s"/></body></worldbody></mujoco>"""))
  shapes = {
      "body_pos": (1, model.nbody, 3),
      "body_quat": (1, model.nbody, 4),
      "geom_pos": (1, model.ngeom, 3),
      "geom_quat": (1, model.ngeom, 4),
      "site_pos": (1, model.nsite, 3),
      "site_quat": (1, model.nsite, 4),
      "inertial_pos": (1, model.nbody, 3),
      "inertial_quat": (1, model.nbody, 4),
      "joint_anchor": (1, model.njnt, 3),
      "joint_axis": (1, model.njnt, 3),
  }
  class Device:
    type = "cpu"
    index = None

  class Tensor:
    dtype = "float32"
    device = Device()

    def __init__(self, array):
      self.array = np.asarray(array, dtype=np.float32)
      self.shape = self.array.shape

    def is_contiguous(self):
      return self.array.flags.c_contiguous

  class Torch:
    float32 = "float32"

    @staticmethod
    def isfinite(tensor):
      return np.isfinite(tensor.array)

  Torch.Tensor = Tensor

  poses = {name: Tensor(np.zeros(shape, dtype=np.float32))
           for name, shape in shapes.items()}
  return model, poses, validate_pose_dict, Torch, Device()


def test_external_pose_validation_rejects_nonfinite_values():
  model, poses, validate, torch, device = _pose_fixture()
  poses["body_pos"].array[0, 0, 0] = float("nan")
  with pytest.raises(ValueError, match="nonfinite"):
    validate(model, poses, 1, device, torch)


def test_trusted_internal_pose_validation_is_metadata_only(monkeypatch):
  model, poses, validate, torch, device = _pose_fixture()

  def unexpected_device_read(_):
    raise AssertionError("trusted pose validation inspected device values")

  monkeypatch.setattr(torch, "isfinite", unexpected_device_read)
  validate(model, poses, 1, device, torch, check_finite=False)
  malformed = dict(poses)
  malformed["body_pos"] = torch.Tensor(np.zeros((1, model.nbody + 1, 3)))
  with pytest.raises(ValueError, match="invalid shape"):
    validate(model, malformed, 1, device, torch, check_finite=False)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_internal_pose_finite_status_is_per_world_and_handles_zero_dof():
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><joint type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
  smooth = MetalSmoothDynamics(load_model(model), batch_size=2)
  qpos = torch.as_tensor([[0.0], [float("nan")]], dtype=torch.float32,
                         device="mps")
  qvel = torch.zeros((2, model.nv), dtype=torch.float32, device="mps")
  result = smooth.run_device(qpos, qvel)
  np.testing.assert_array_equal(result["pose_status"].cpu().numpy(), [0, 1])

  fixed_model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
  fixed = MetalSmoothDynamics(load_model(fixed_model), batch_size=1)
  empty_qpos = torch.zeros((1, 0), dtype=torch.float32, device="mps")
  empty_qvel = torch.zeros((1, 0), dtype=torch.float32, device="mps")
  poses = fixed._fk.run_device(empty_qpos)
  poses = {name: value.clone() for name, value in poses.items()}
  poses["body_pos"].fill_(float("nan"))
  fixed_result = fixed.run_device(
      empty_qpos, empty_qvel, poses=poses, _trusted_internal_poses=True)
  np.testing.assert_array_equal(fixed_result["pose_status"].cpu().numpy(), [1])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="opt-in GPU")
def test_internal_pose_status_covers_every_consumed_field_and_recovers():
  import torch
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  raw = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><joint type="slide"/><geom type="sphere" size=".1" mass="1"/>
      <site name="s" pos=".01 .02 .03"/></body></worldbody></mujoco>""")
  smooth = MetalSmoothDynamics(load_model(raw), batch_size=2)
  q = torch.zeros((2, raw.nq), dtype=torch.float32, device="mps")
  v = torch.zeros((2, raw.nv), dtype=torch.float32, device="mps")
  poses = {name: value.clone() for name, value in smooth._fk.run_device(q).items()}
  for name in ("body_pos", "body_quat", "geom_pos", "geom_quat",
               "site_pos", "site_quat", "inertial_pos", "inertial_quat",
               "joint_anchor", "joint_axis"):
    bad = dict(poses)
    bad[name] = poses[name].clone()
    bad[name][1].reshape(-1)[0] = float("inf")
    result = smooth.run_device(q, v, poses=bad, _trusted_internal_poses=True)
    np.testing.assert_array_equal(result["pose_status"].cpu().numpy(), [0, 1],
                                  err_msg=name)
    # Status borrows workspace; clone it before the subsequent valid dispatch.
    saved = result["pose_status"].clone()
    valid = smooth.run_device(q, v, poses=poses, _trusted_internal_poses=True)
    np.testing.assert_array_equal(valid["pose_status"].cpu().numpy(), [0, 0])
    np.testing.assert_array_equal(saved.cpu().numpy(), [0, 1])
    # External supplied packets keep synchronous validation at the API boundary.
    with pytest.raises(ValueError, match="nonfinite"):
      smooth.run_device(q, v, poses=bad)
