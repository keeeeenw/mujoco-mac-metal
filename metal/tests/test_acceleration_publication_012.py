# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public acceleration splitting preserves exact values across Metal FTZ."""
from fractions import Fraction
import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native publication')
def test_final_acceleration_publication_preserves_exact_pair_and_subnormal_words():
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  root = Path(__file__).resolve().parents[1] / 'mujoco_metal' / 'shaders'
  shader = _coupled_shader_source(expected_root=root)
  rows = []
  for exponent in (-149, -148, -140, -127, -126, -125, -120, -100, -1, 0, 24, 100, 126, 127):
    high = np.float32(np.ldexp(1., exponent))
    for sign in (-1, 1):
      for low in (np.float32(0), high, np.nextafter(high, np.float32(np.inf)),
                  -high, -np.nextafter(high, np.float32(np.inf)),
                  np.float32(float(high) * 2**-23)):
        rows.append((np.float32(sign * high), np.float32(sign * low)))
  for bits in (1, 2, 3, 0x3fffff, 0x7fffff, 0x800000, 0x800001, 0x7f7fffff):
    high = np.array([bits], np.uint32).view(np.float32)[0]
    for low in (high, -high, np.nextafter(high, np.float32(0))):
      rows.append((high, low))
  rng = np.random.default_rng(120145)
  bits = rng.integers(0, 2**32, (512, 2), dtype=np.uint32)
  # Include nonfinite payloads explicitly; publication must preserve the
  # existing fault-producing words rather than manufacture a different fault.
  rows.extend(bits.view(np.float32))
  rows.extend(np.array([[0x7f800000, 1], [0xff800000, 0x7f800000],
                        [0x7fc00001, 0], [0x80000000, 0]], np.uint32).view(np.float32))
  packed = np.asarray(rows, np.float32)
  shader += r'''
  kernel void final_acceleration_split_control(device const float* input,
      device float* output, device const int* mask,
      uint tid [[thread_position_in_grid]]) {
    if (mask[tid] == 0) return;
    PrimalFloatPair result = primal_acceleration_publication_pair(
        input[2*tid], input[2*tid+1]);
    output[2*tid] = result.hi; output[2*tid+1] = result.lo;
  }
  '''
  library = torch.mps.compile_shader(shader)
  inputs = torch.tensor(packed, device='mps'); before = inputs.clone()
  outputs = torch.full_like(inputs, 123.25)
  mask = np.ones(len(packed), np.int32); mask[7::19] = 0
  library.final_acceleration_split_control(inputs, outputs,
      torch.tensor(mask, device='mps'), threads=len(packed))
  actual = outputs.cpu().numpy()
  for index, (original, result) in enumerate(zip(packed, actual)):
    if mask[index] == 0:
      assert result.tobytes() == np.full(2, 123.25, np.float32).tobytes()
      continue
    if not np.isfinite(original).all():
      assert result.tobytes() == original.tobytes()
      continue
    exact = Fraction(float(original[0])) + Fraction(float(original[1]))
    with np.errstate(over='ignore'):
      expected_high = np.float32(float(exact))
    if not np.isfinite(expected_high):
      assert result.tobytes() == original.tobytes()
      continue
    assert np.isfinite(result).all()
    assert Fraction(float(result[0])) + Fraction(float(result[1])) == exact, (index, original.view(np.uint32), result.view(np.uint32))
    assert result[0] == expected_high
  torch.testing.assert_close(inputs.view(torch.int32), before.view(torch.int32), rtol=0, atol=0)
