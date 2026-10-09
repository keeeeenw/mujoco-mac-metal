# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Real rotating depth images, including executable installed-demo discovery."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from mujoco_metal.demo_cli import available_demos


def _scene(mode):
  path = Path(__file__).resolve().parents[1]/"examples/optical_inspection.py"
  spec = importlib.util.spec_from_file_location("inspection_demo", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.InspectionCarousel(mode)


@pytest.mark.parametrize("mode", ["cpu", "metal"])
def test_real_carousel_depth_frames_and_independent_rollout(mode):
  if mode == "metal" and os.getenv("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("explicit native qualification required")
  assert "optical_inspection" in available_demos()
  scene = _scene(mode)
  for _ in range(80):
    scene.step()
  scene.check()
  assert scene.images.shape == (2, 24, 32)
  assert scene.model.nv == 1 and scene.model.nu == 1
  assert np.max(np.abs(scene.images[0]-scene.images[1])) > .05
  assert scene.report()["max_image_change"] > .1
  if scene.native is not None:
    # The pixels must come from the native stored sample, without a fresh
    # scene query or substituting CPU/rendered depth.
    np.testing.assert_array_equal(scene.native.step_sensordata()[0].reshape(2, 24, 32),
                                  scene.images)
