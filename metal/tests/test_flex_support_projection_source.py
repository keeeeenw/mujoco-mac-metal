# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual support projections must retain every coordinate storage word."""
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from test_common_ccd_matrix_source import _words


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_flex_support_projection_matches_pinned_full_coordinate_operands():
  import torch
  source = (Path(__file__).parents[1] / "mujoco_metal/shaders/flex_contact.metal").read_text()
  helpers = source[source.index("struct FlexDD {"):
                   source.index("static inline FlexDD3 flex_dd3_normalize(")]
  rng = np.random.default_rng(39119)
  inputs = rng.uniform(-.2, .2, (256, 2, 3))
  inputs[:, 1] /= np.linalg.norm(inputs[:, 1], axis=1)[:, None]
  expected = np.array([mujoco.mju_dot3(row[0], row[1]) for row in inputs])
  shader = "#include <metal_stdlib>\nusing namespace metal;\n" + helpers + r"""
  kernel void test_support_projection(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int p=18*int(tid);
    FlexDD3 a=FlexDD3{FlexDD{input[p],input[p+1],input[p+2]},
      FlexDD{input[p+3],input[p+4],input[p+5]},
      FlexDD{input[p+6],input[p+7],input[p+8]}};
    p+=9;
    FlexDD3 b=FlexDD3{FlexDD{input[p],input[p+1],input[p+2]},
      FlexDD{input[p+3],input[p+4],input[p+5]},
      FlexDD{input[p+6],input[p+7],input[p+8]}};
    FlexProjection53 value=flex_mju_dot3_projection53(a,b);
    output[3*tid]=value.hi; output[3*tid+1]=value.mid; output[3*tid+2]=value.low;
  }
  """
  lib = torch.mps.compile_shader(shader)
  data = torch.tensor(np.stack([_words(row) for row in inputs]), device="mps")
  output = torch.zeros((len(inputs), 3), device="mps")
  lib.test_support_projection(data, output, threads=len(inputs))
  actual = np.array([math.fsum(map(float, row)) for row in output.cpu().numpy()])
  np.testing.assert_array_equal(actual, expected)
