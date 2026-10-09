# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native SDF roots preserve the pinned binary64 rounding boundary."""
import math
import os

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_sdf_sqrt_midpoints_and_search_scales_match_cpu():
  import torch
  from mujoco_metal.plugin_sdf import _SHADER
  rng = np.random.default_rng(198733)
  values = list(np.exp2(rng.uniform(-64, 64, 256)))
  for exponent in range(-64, 65, 8):
    boundary = math.ldexp(1.0, exponent)
    values.extend((boundary, math.nextafter(boundary, math.inf),
                   math.nextafter(boundary, 0.0),
                   math.nextafter(math.nextafter(boundary, math.inf), math.inf)))
  values.extend((0.0, 1.0000000000000002, .6045915968413681))
  values = np.asarray(values, dtype=np.float64)
  high = values.astype(np.float32)
  residual = values - high.astype(np.float64)
  low = residual.astype(np.float32)
  tail = (residual - low.astype(np.float64)).astype(np.float32)
  operands = torch.tensor(np.stack((high, low, tail), axis=-1), device="mps")
  before = operands.clone()
  output = torch.zeros_like(operands)
  source = _SHADER.read_text() + r'''
  kernel void test_sdf_roots(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    int base=3*int(i);
    SdfWide root=sw_sqrt({input[base],input[base+1],input[base+2]});
    output[base]=root.hi;output[base+1]=root.mid;output[base+2]=root.lo;
  }
  '''
  library = torch.mps.compile_shader(source)
  library.test_sdf_roots(operands, output, threads=len(values))
  actual = output.cpu().double().numpy().sum(axis=-1)
  np.testing.assert_array_equal(actual, [math.sqrt(x) for x in values])
  torch.testing.assert_close(operands.view(torch.int32), before.view(torch.int32),
                             rtol=0, atol=0)
