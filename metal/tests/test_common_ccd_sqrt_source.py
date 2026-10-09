# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native source-rounded roots, including EPA support midpoint cases."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from test_common_ccd_epa_witness_source import _function
from test_common_ccd_matrix_source import _words


def test_sqrt_refinement_is_shared_between_rigid_and_flex_sources():
  root = Path(__file__).parents[1] / "mujoco_metal/shaders"
  bodies = [_function((root / name).read_text(), r"static inline FlexDD flex_dd_source_sqrt\(")
            for name in ("common_ccd_core.metal", "flex_contact.metal")]
  assert bodies[0] == bodies[1]


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_native_sqrt_matches_binary64_at_midpoints_and_support_scales():
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  rng = np.random.default_rng(198733)
  values = list(np.exp2(rng.uniform(-64, 64, 256)))
  for exponent in range(-64, 65, 8):
    boundary = math.ldexp(1., exponent)
    values.extend([boundary, math.nextafter(boundary, math.inf),
                   math.nextafter(boundary, 0.),
                   math.nextafter(math.nextafter(boundary, math.inf), math.inf)])
  values.extend([0., 1.0000000000000002, .6045915968413681])
  values = np.asarray(values, np.float64)
  expected = np.array([math.sqrt(value) for value in values])
  source = _coupled_shader_source() + r"""
  kernel void test_source_roots(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    int base=3*int(tid);
    FlexDD value={input[base],input[base+1],input[base+2]};
    FlexDD root=flex_dd_source_sqrt(value);
    output[base]=root.hi;output[base+1]=root.lo;output[base+2]=root.tail;
  }
  """
  library = torch.mps.compile_shader(source)
  operand = torch.tensor(np.stack([_words([value]) for value in values]),
                         dtype=torch.float32, device="mps")
  before = operand.clone()
  output = torch.zeros_like(operand)
  library.test_source_roots(operand, output, threads=len(values))
  represented = output.cpu().double().numpy().sum(axis=-1)
  np.testing.assert_array_equal(represented, expected)
  torch.testing.assert_close(operand.view(torch.int32), before.view(torch.int32),
                             rtol=0, atol=0)
