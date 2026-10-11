# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Persistent state projections include the word that decides a halfway tie."""
import os

import numpy as np
import pytest

from mujoco_metal.retained_state import pack_residuals, validate_residuals


def _cases():
  # This noncanonical high/low/tail came from an ordinary native Euler step.
  values = [[float(np.float32(0.10089728236198425)),
             float(np.float32(3.725290298461914e-9)),
             float(np.float32(1.3877787807814457e-17))]]
  # Both signs, both neighboring directions, both parities, binade edges,
  # normal/subnormal transport and a tail on either side of a halfway tie.
  for exponent in (-120, -96, -64, -16, -1, 0, 16, 64, 96, 120):
    base = np.float32(np.ldexp(1., exponent))
    for hi in (base, np.nextafter(base, np.float32(np.inf))):
      for sign in (-1., 1.):
        high = np.float32(sign * hi)
        for toward in (-np.inf, np.inf):
          neighbor = np.nextafter(high, np.float32(toward))
          midpoint = (float(high) + float(neighbor)) / 2
          for source in (np.nextafter(midpoint, -np.inf), midpoint,
                         np.nextafter(midpoint, np.inf)):
            residual = source - float(high)
            low = np.float32(residual)
            tail = np.float32(residual - float(low))
            if float(high) + float(low) + float(tail) == source:
              values.append([high, low, tail])
  # Exact subnormal values cannot contain a binary32 half-ulp word, but their
  # stored bits must survive Metal's arithmetic flush-to-zero behavior.
  for bits in (1, 2, 0x007fffff, 0x00800000, 0x00800001,
               0x80000001, 0x807fffff, 0x80800000):
    values.append([np.array(bits, np.uint32).view(np.float32), 0., 0.])
  values.extend([[0., 0., 0.], [-0., 0., 0.]])
  return np.asarray(values, np.float32)


def test_checkpoint_validator_rejects_dropped_halfway_tail():
  words = _cases()[:1]
  qpos = words[:, :1].copy()
  qvel = np.zeros_like(qpos)
  residuals = np.array([[words[0, 1], words[0, 2], 0., 0.]], np.float32)
  with pytest.raises(ValueError, match="high projection"):
    validate_residuals(qpos, qvel, residuals)
  represented = words.astype(np.float64).sum(axis=1, keepdims=True)
  canonical = pack_residuals(represented, qvel)
  accepted, restored, _ = validate_residuals(
      represented.astype(np.float32), qvel, canonical)
  np.testing.assert_array_equal(accepted, canonical)
  np.testing.assert_array_equal(restored, represented)
  assert represented.astype(np.float32).view(np.uint32)[0, 0] == 0x3dcea33d


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_state_writer_publishes_canonical_projection_and_residuals():
  import torch
  from mujoco_metal.integration import _SHADER
  from mujoco_metal.metal_kinematics import _SOURCE_MATH_SHADER

  source = ("#define MUJOCO_METAL_SOURCE_MATH_ONLY\n"
            + _SOURCE_MATH_SHADER.read_text()
            + "\n#undef MUJOCO_METAL_SOURCE_MATH_ONLY\n"
            + _SHADER.with_name("position_math.metal").read_text() + r'''
  kernel void test_state_projection(device const float* input [[buffer(0)]],
      device float* high [[buffer(1)]], device float* residual [[buffer(2)]],
      uint i [[thread_position_in_grid]]) {
    uint at=3*i;
    kin_state_store(high,residual,i,2*i,2*i+1,
                    KinDD{input[at],input[at+1],input[at+2]});
  }
  ''')
  cases = _cases()
  represented = cases.astype(np.float64).sum(axis=1, keepdims=True)
  # Preserve the high word's sign for an exactly zero represented state.
  represented[represented[:, 0] == 0, 0] = cases[represented[:, 0] == 0, 0]
  expected_high = represented.astype(np.float32)
  expected_words = pack_residuals(represented, np.zeros_like(represented))[:, :2]
  operands = torch.tensor(cases, device="mps")
  before = operands.clone()
  high = torch.full((len(cases),), -432.25, device="mps")
  words = torch.full((len(cases), 2), -432.25, device="mps")
  library = torch.mps.compile_shader(source)
  library.test_state_projection(operands, high, words, threads=len(cases))
  actual_high = high.cpu().numpy()[:, None]
  actual_words = words.cpu().numpy()
  np.testing.assert_array_equal(actual_high.view(np.uint32),
                                expected_high.view(np.uint32))
  np.testing.assert_array_equal(actual_words.view(np.uint32),
                                expected_words.view(np.uint32))
  np.testing.assert_array_equal(operands.cpu().numpy().view(np.uint32),
                                before.cpu().numpy().view(np.uint32))
  full_words = np.concatenate((actual_words, np.zeros_like(actual_words)), axis=1)
  _, restored, _ = validate_residuals(actual_high, np.zeros_like(actual_high), full_words)
  np.testing.assert_array_equal(restored, represented)
