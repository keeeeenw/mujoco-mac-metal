# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Dynamic compiled-mesh frames need source-rounded quaternion operations."""
import math
import os

import mujoco
import numpy as np
import pytest

from test_mesh_hfield_contact_012 import _model


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_actual_dynamic_mesh_frames_equal_pinned_binary64_values():
  import torch
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model

  model = _model()
  # The compiler rotates this box mesh by a quaternion whose source low
  # words matter. Small nonzero free-joint rotations reproduce the dynamic
  # frame defect; the former identity-only fixture did not exercise it.
  rng = np.random.default_rng(118)
  qpos = np.repeat(model.qpos0.astype(np.float32)[None], 16, axis=0)
  qpos[:, 3:] = np.column_stack((np.ones(16), rng.normal(size=(16, 3)) * .002))
  qpos[:, :3] += rng.normal(size=(16, 3)).astype(np.float32) * .02
  references = [mujoco.MjData(model) for _ in range(len(qpos))]
  for world, data in enumerate(references):
    data.qpos[:] = qpos[world]
    mujoco.mj_forward(model, data)
  fk = MetalKinematics(load_model(model), batch_size=len(qpos))
  poses = fk.run_device(torch.as_tensor(qpos, device="mps"))
  for name, expected in (
      ("body_quat", [d.xquat for d in references]),
      ("geom_pos", [d.geom_xpos for d in references]),
      ("geom_xmat", [d.geom_xmat for d in references])):
    planes = [poses[name + suffix].detach().cpu().numpy().reshape(-1)
              for suffix in ("", "_low", "_tail")]
    actual = np.asarray([math.fsum(words) for words in zip(*planes)]).reshape(np.shape(expected))
    np.testing.assert_array_equal(actual, expected, err_msg=name)


_CHAIN_XML = """
<mujoco><option gravity="0 0 0"/>
<worldbody><body pos=".1234567890123 -.0712345678901 .2056789012345">
  <freejoint/><inertial pos=".0234567890123 -.0312345678901 .0476543210987"
      quat=".91 .21 -.13 .29" mass="1" diaginertia=".1 .12 .15"/>
  <geom type="box" pos="-.0276543210987 .0376543210987 .0165432109876"
      quat=".83 -.19 .31 .27" size=".1 .08 .07"/>
  <site pos=".0434567890123 -.0134567890123 .0245678901234"/>
  <body pos=".1376543210987 -.1123456789012 .0712345678901"
      quat=".79 .32 -.17 .41">
    <joint type="ball" pos=".0123456789012 -.0276543210987 .0312345678901"/>
    <geom type="sphere" size=".04" pos="-.0212345678901 .0345678901234 .0423456789012"/>
    <site pos=".0543210987654 -.0254321098765 .0154321098765"/>
    <body pos=".1234567890123 .0876543210987 -.0534567890123"
        quat=".87 -.21 .23 -.31">
      <joint type="slide" axis=".23 -.41 .87"/>
      <geom type="capsule" size=".025 .07" pos=".0432109876543 -.0321098765432 .0210987654321"/>
      <site pos=".0243210987654 .0343210987654 -.0143210987654"/>
    </body>
  </body>
</body></worldbody></mujoco>
"""


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_actual_composed_frames_equal_pinned_binary64_values():
  import torch
  from mujoco_metal.metal_kinematics import MetalKinematics
  from mujoco_metal.model import load_model

  model = mujoco.MjModel.from_xml_string(_CHAIN_XML)
  rng = np.random.default_rng(912)
  qpos = np.repeat(model.qpos0.astype(np.float32)[None], 12, axis=0)
  qpos[:, :3] += rng.normal(size=(12, 3)).astype(np.float32) * .2
  qpos[:, 3:7] = rng.normal(size=(12, 4)).astype(np.float32)
  qpos[:, 7:11] = rng.normal(size=(12, 4)).astype(np.float32)
  qpos[:, 11] = rng.normal(size=12).astype(np.float32) * .13
  references = [mujoco.MjData(model) for _ in qpos]
  for values, data in zip(qpos, references):
    data.qpos[:] = values
    mujoco.mj_forward(model, data)
  fk = MetalKinematics(load_model(model), batch_size=len(qpos))
  poses = fk.run_device(torch.as_tensor(qpos, device="mps"))
  for name, field in (("body_quat", "xquat"), ("body_pos", "xpos"),
                      ("geom_pos", "geom_xpos"), ("site_pos", "site_xpos"),
                      ("inertial_pos", "xipos"), ("geom_xmat", "geom_xmat")):
    expected = np.asarray([getattr(d, field) for d in references])
    planes = [poses[name + suffix].detach().cpu().numpy().reshape(-1)
              for suffix in ("", "_low", "_tail")]
    actual = np.asarray([math.fsum(words) for words in zip(*planes)]).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected, err_msg=name)
