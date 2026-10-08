# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compare actual native CCD matrix operations with pinned compiled queries."""
import os
import mujoco
import numpy as np
import pytest


def _words(values):
  values = np.asarray(values, np.float64)
  high = values.astype(np.float32)
  remainder = values - high.astype(np.float64)
  low = remainder.astype(np.float32)
  tail = (remainder - low.astype(np.float64)).astype(np.float32)
  return np.stack((high, low, tail), axis=-1).reshape(-1)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_common_ccd_matrix_actions_match_pinned_compiled_binary64_queries():
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  rng = np.random.default_rng(9817)
  matrices = rng.normal(size=(32, 3, 3))
  vectors = rng.normal(size=(32, 3))
  packed = np.stack([_words(np.concatenate((M.reshape(-1), v)))
                     for M, v in zip(matrices, vectors)])
  source = _coupled_shader_source() + r"""
  kernel void test_common_matrix_actions(device const float* inputs [[buffer(0)]],
      device float* outputs [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    int base=36*int(tid);FlexDD matrix[9];
    for (int i=0;i<9;i++) matrix[i]=FlexDD{inputs[base+3*i],inputs[base+3*i+1],inputs[base+3*i+2]};
    FlexDD3 vector=common_ccd_load_dd3(inputs,base+27);
    common_ccd_store_dd3(outputs,18*int(tid),common_ccd_matrix_vector(matrix,vector));
    common_ccd_store_dd3(outputs,18*int(tid)+9,common_ccd_matrix_transpose_vector(matrix,vector));
  }
  """
  library = torch.mps.compile_shader(source)
  inputs = torch.tensor(packed, dtype=torch.float32, device="mps")
  outputs = torch.zeros((32, 18), dtype=torch.float32, device="mps")
  before = inputs.clone()
  library.test_common_matrix_actions(inputs, outputs, threads=32)
  represented = outputs.cpu().double().numpy().reshape(32, 2, 3, 3).sum(axis=-1)
  expected = np.zeros((32, 2, 3), dtype=np.float64)
  for i, (M, v) in enumerate(zip(matrices, vectors)):
    mujoco.mju_mulMatVec3(expected[i, 0], M.reshape(-1), v)
    mujoco.mju_mulMatTVec3(expected[i, 1], M.reshape(-1), v)
  np.testing.assert_array_equal(represented, expected)
  torch.testing.assert_close(inputs.view(torch.int32), before.view(torch.int32), rtol=0, atol=0)
