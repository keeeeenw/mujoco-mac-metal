"""CPU contract checks for the retained geometry-pose residual ABI.

The candidate is a delta-only tree. Load its smooth module explicitly while
using the paired source package for its ordinary model/kinematics dependencies.
"""

import importlib.util
import os
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

import pytest
import torch
import numpy as np


METAL = Path(__file__).resolve().parents[1]
SMOOTH_PATH = METAL / "mujoco_metal" / "smooth_metal.py"
_SPEC = importlib.util.spec_from_file_location(
    "mujoco_metal._position_context_candidate", SMOOTH_PATH)
_SMOOTH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMOOTH)


class _Model:
  nbody = 3
  ngeom = 2
  nsite = 1
  njnt = 2
  nv = 3


def _pose_tensors(batch=2, device="cpu"):
  model = _Model()
  poses = {}
  for kind, count in (("body", model.nbody), ("inertial", model.nbody),
                      ("geom", model.ngeom), ("site", model.nsite)):
    poses[f"{kind}_pos"] = torch.zeros((batch, count, 3), device=device)
    poses[f"{kind}_pos_low"] = torch.zeros((batch, count, 3), device=device)
    poses[f"{kind}_pos_tail"] = torch.zeros((batch, count, 3), device=device)
    poses[f"{kind}_quat"] = torch.zeros((batch, count, 4), device=device)
    poses[f"{kind}_quat_low"] = torch.zeros((batch, count, 4), device=device)
    poses[f"{kind}_quat_tail"] = torch.zeros((batch, count, 4), device=device)
  for name in ("geom_xmat", "geom_xmat_low", "geom_xmat_tail"):
    poses[name] = torch.zeros((batch, model.ngeom, 9), device=device)
  poses["joint_anchor"] = torch.zeros((batch, model.njnt, 3), device=device)
  poses["joint_axis"] = torch.zeros((batch, model.njnt, 3), device=device)
  return model, poses


def _smooth_stub(batch=2):
  model, poses = _pose_tensors(batch)
  shapes = _SMOOTH.position_context_shapes(model, batch)
  views = {}
  for name, shape in shapes.items():
    dtype = torch.int32 if name == "pose_status" else torch.float32
    views[name] = torch.full(shape, 17, dtype=dtype)
  obj = SimpleNamespace(
      _workspace={"batch_size": batch,
                  "local_inertia": torch.full((batch, model.nbody, 36), 17.)},
      _position_shapes=shapes,
      _position_views=views,
      _torch=torch,
      _fk=SimpleNamespace(_device=torch.device("cpu")),
      model=model,
      _position_capture=0,
      _position_context_epoch=0,
  )
  return obj, poses


def _complete_dynamics(obj, poses):
  batch, model = obj._workspace["batch_size"], obj.model
  result = {
      "poses": poses,
      "root_com": torch.zeros((batch, model.nbody, 3)),
      "cdof": torch.zeros((batch, model.nv, 6)),
      "pose_status": torch.zeros((batch,), dtype=torch.int32),
      "mass_matrix": torch.zeros((batch, model.nv, model.nv)),
  }
  return result


def test_position_context_shapes_retain_all_fk_geom_rotation_words():
  shapes = _SMOOTH.position_context_shapes(_Model(), 2)
  assert shapes["poses.geom_xmat"] == (2, 2, 9)
  assert shapes["poses.geom_xmat_low"] == (2, 2, 9)
  assert shapes["poses.geom_xmat_tail"] == (2, 2, 9)
  assert shapes["poses.geom_quat_low"] == (2, 2, 4)
  assert shapes["poses.geom_quat_tail"] == (2, 2, 4)
  sizes = _SMOOTH.position_context_workspace_sizes(_Model(), 2)
  assert sizes["position_cache.poses.geom_xmat"] == 2 * 2 * 9
  assert sizes["position_cache.poses.geom_xmat_low"] == 2 * 2 * 9
  assert sizes["position_cache.poses.geom_xmat_tail"] == 2 * 2 * 9


def test_pose_and_cached_pose_validation_rejects_each_added_field():
  model, poses = _pose_tensors()
  for name in ("geom_xmat", "geom_xmat_low", "geom_xmat_tail",
               "geom_quat_low", "geom_quat_tail"):
    _SMOOTH.validate_pose_dict(model, poses, 2, torch.device("cpu"), torch)
    bad = dict(poses)
    bad.pop(name)
    with pytest.raises(ValueError, match="complete position-stage ABI"):
      _SMOOTH.validate_pose_dict(model, bad, 2, torch.device("cpu"), torch)
    bad = dict(poses)
    bad[name] = poses[name].to(torch.float64)
    with pytest.raises(ValueError, match=name):
      _SMOOTH.validate_pose_dict(model, bad, 2, torch.device("cpu"), torch)
    bad = dict(poses)
    bad[name] = torch.empty(poses[name].shape, device="meta")
    with pytest.raises(ValueError, match=name):
      _SMOOTH.validate_pose_dict(model, bad, 2, torch.device("cpu"), torch)
    bad = dict(poses)
    bad[name] = torch.zeros((2, 2, 18))[:, :, ::2]
    with pytest.raises(ValueError, match=name):
      _SMOOTH.validate_pose_dict(model, bad, 2, torch.device("cpu"), torch)

    stub, _ = _smooth_stub()
    context = {"poses": dict(poses)}
    context["poses"].pop(name)
    with pytest.raises(ValueError, match="incomplete cached pose ABI"):
      _SMOOTH.MetalSmoothDynamics._validate_cached_pose_fields(
          stub, context, 2)


def test_position_context_rejects_bad_residuals_before_cache_writes():
  for name in ("geom_xmat", "geom_xmat_low", "geom_xmat_tail",
               "geom_quat_low", "geom_quat_tail"):
    stub, poses = _smooth_stub()
    before = {key: value.clone() for key, value in stub._position_views.items()}
    dynamics = _complete_dynamics(stub, poses)
    dynamics["poses"] = dict(poses)
    dynamics["poses"][name] = torch.zeros_like(poses[name], dtype=torch.float64)
    with pytest.raises(ValueError, match=f"position stage poses\\.{name}"):
      _SMOOTH.MetalSmoothDynamics.position_context(stub, dynamics)
    for key, value in before.items():
      torch.testing.assert_close(stub._position_views[key], value, rtol=0, atol=0)


def test_velocity_dispatch_validates_cached_residual_and_mask_before_bias():
  stub, poses = _smooth_stub()
  calls = []
  stub._position_capture = 3
  context = {"_owner": stub, "_epoch": 0, "_capture": 3,
             "poses": dict(poses), "cdof": torch.zeros((2, 3, 6)),
             "_local_inertia": torch.zeros((2, 3, 36))}
  stub._run_velocity_bias = lambda *args, **kwargs: calls.append(1)
  stub._workspace["all_awake_lists"] = {}
  stub._validate_cached_pose_fields = MethodType(
      _SMOOTH.MetalSmoothDynamics._validate_cached_pose_fields, stub)
  stub._validate_world_mask = MethodType(
      _SMOOTH.MetalSmoothDynamics._validate_world_mask, stub)
  stub._validate_awake_lists = lambda lists, batch: None
  stub._fk._check_device_tensor = lambda *args: None
  bad = dict(context)
  bad["poses"] = dict(poses)
  bad["poses"]["geom_xmat_tail"] = torch.zeros((2, 2, 9), dtype=torch.float64)
  with pytest.raises(ValueError, match="geom_xmat_tail"):
    _SMOOTH.MetalSmoothDynamics.run_velocity_device(
        stub, bad, torch.zeros((2, 3)))
  assert not calls
  with pytest.raises(ValueError, match="world_mask"):
    _SMOOTH.MetalSmoothDynamics.run_velocity_device(
        stub, context, torch.zeros((2, 3)), world_mask=torch.zeros((2,), dtype=torch.int64))
  assert not calls


@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="opt-in GPU")
def test_native_public_smooth_pose_cache_retains_rotation_residuals_atomically():
  import mujoco
  from mujoco_metal.model import load_model
  from mujoco_metal.smooth_metal import MetalSmoothDynamics

  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><joint type="hinge"/><geom type="sphere" size=".1"/></body>
  </worldbody></mujoco>""")
  smooth = MetalSmoothDynamics(load_model(model), batch_size=2)
  qpos = torch.as_tensor(np.tile(model.qpos0, (2, 1)),
                         dtype=torch.float32, device="mps")
  qvel = torch.zeros((2, model.nv), dtype=torch.float32, device="mps")
  dynamics = smooth.run_device(qpos, qvel)
  context = smooth.position_context(dynamics)
  fields = ("geom_xmat", "geom_xmat_low", "geom_xmat_tail",
            "geom_quat_low", "geom_quat_tail")
  expected = {name: context["poses"][name].clone() for name in fields}

  # Subsequent borrowed FK outputs may be overwritten, but the retained POS
  # cache has independent backing and must remain exact for the VEL stage.
  for name in fields:
    dynamics["poses"][name].zero_()
  result = smooth.run_velocity_device(context, qvel)
  for name in fields:
    torch.testing.assert_close(result["poses"][name], expected[name],
                               rtol=0, atol=0)

  status_before = smooth._workspace["pose_status"].clone()
  malformed = dict(result["poses"])
  malformed["geom_xmat_tail"] = torch.empty(
      (2, model.ngeom, 9), dtype=torch.float16, device="mps")
  with pytest.raises(ValueError, match="geom_xmat_tail"):
    smooth.run_device(qpos, qvel, poses=malformed)
  torch.testing.assert_close(smooth._workspace["pose_status"], status_before,
                             rtol=0, atol=0)

  bias_before = smooth._workspace["bias"].clone()
  corrupt_context = dict(context)
  corrupt_context["poses"] = dict(context["poses"])
  corrupt_context["poses"]["geom_xmat_low"] = torch.empty(
      (2, model.ngeom, 9), dtype=torch.float16, device="mps")
  with pytest.raises(ValueError, match="geom_xmat_low"):
    smooth.run_velocity_device(corrupt_context, qvel)
  torch.testing.assert_close(smooth._workspace["bias"], bias_before,
                             rtol=0, atol=0)

  newer = smooth.position_context(dynamics)
  with pytest.raises(ValueError, match="another or replaced workspace"):
    smooth.run_velocity_device(context, qvel)
  assert newer["_capture"] == smooth._position_capture
