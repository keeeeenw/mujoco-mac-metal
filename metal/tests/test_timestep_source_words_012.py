# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual integration constants and updates retain the source timestep."""
import os
import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native timestep precision")
@pytest.mark.parametrize("timestep", [.0001, .0005, .001, .002, .017,
                                     np.nextafter(.001, 1.)])
def test_native_timestep_words_and_velocity_match_installed_source(timestep):
  import torch
  from mujoco_metal.integration import MetalEulerIntegration
  from mujoco_metal.model import load_model
  descriptor = load_model('<mujoco><worldbody><body><freejoint/>'
                          '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
  stage = MetalEulerIntegration(descriptor, 1, float(timestep))
  represented = stage._dt.cpu().double().numpy().sum()
  assert represented == float(timestep)
  # Arbitrary native operator inputs; only the expected output uses source C.
  velocity = np.array([.02, -.01, .03, .04, -.02, .01], np.float32)
  acceleration = np.array([-12.9, -35.1, 140.2, 428.8, -54.7, 3.35], np.float32)
  acceleration_low = np.array([1e-7, -3e-7, 2e-6, -3e-6, 4e-7, -2e-8], np.float32)
  expected = velocity.astype(float)
  mujoco.mju_addToScl(expected, acceleration.astype(float) + acceleration_low.astype(float),
                      float(timestep))
  q = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], device="mps")
  v = torch.tensor(velocity[None], device="mps")
  a = torch.tensor(acceleration[None], device="mps")
  low = torch.tensor(acceleration_low[None], device="mps")
  _, output_v, _, status = stage.run_device(q, v, a,
      torch.zeros(1, device="mps"), torch.zeros(1, device="mps", dtype=torch.int32),
      qacc_low=low)
  np.testing.assert_array_equal(status.cpu().numpy(), [0])
  words = stage._output_words.cpu().double().numpy()[0]
  actual = output_v.cpu().double().numpy()[0] + words[2*descriptor.nq:2*descriptor.nq+descriptor.nv] + words[2*descriptor.nq+descriptor.nv:]
  np.testing.assert_array_equal(actual, expected)
