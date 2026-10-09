# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source scalar rounding and subnormal word transport on actual Metal."""
import ctypes
import math
import os
import subprocess

import numpy as np
import pytest

from test_common_ccd_matrix_source import _words


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_source_power_of_two_transport_preserves_all_float_word_classes():
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  bits = np.array([
      0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
      0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
      0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
      0x7fc00000,
  ], dtype=np.uint32)
  pairs = np.array([(int(value), shift) for value in bits
                    for shift in (-160, -100, -24, -1, 0, 1, 24, 100, 160)],
                   dtype=np.int64)
  values = pairs[:, 0].astype(np.uint32).view(np.float32)
  with np.errstate(over="ignore", invalid="ignore", under="ignore"):
    expected = np.ldexp(values.astype(np.float64), pairs[:, 1]).astype(np.float32)
  shader = _coupled_shader_source() + r"""
  kernel void test_scalbn_words(device const uint* input [[buffer(0)]],
      device const int* shifts [[buffer(1)]], device uint* output [[buffer(2)]],
      uint tid [[thread_position_in_grid]]) {
    output[tid]=as_type<uint>(flex_float_scalbn(as_type<float>(input[tid]),shifts[tid]));
  }
  """
  lib = torch.mps.compile_shader(shader)
  source = torch.tensor(pairs[:, 0].astype(np.uint32).view(np.int32).copy(),
                        dtype=torch.int32, device="mps")
  shifts = torch.tensor(pairs[:, 1], dtype=torch.int32, device="mps")
  output = torch.empty_like(source)
  lib.test_scalbn_words(source, shifts, output, threads=len(pairs))
  np.testing.assert_array_equal(output.cpu().numpy().view(np.uint32),
                                expected.view(np.uint32))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_source_scalar_products_divisions_and_fma_match_compiled_binary64(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  src = tmp_path / "scalar.c"
  src.write_text("""
#include <math.h>
void scalar(const double *i, double *o) {
  o[0]=i[0]*i[1];
  o[1]=i[0]/i[1];
  o[2]=fma(i[0],i[1],i[2]);
}
""")
  path = tmp_path / "scalar.dylib"
  subprocess.run(["/usr/bin/clang", "-O3", "-std=c11", "-shared",
                  str(src), "-o", str(path)], check=True, capture_output=True,
                 text=True, timeout=30)
  oracle = ctypes.CDLL(str(path))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  oracle.scalar.argtypes = [pointer, pointer]
  oracle.scalar.restype = None
  rng = np.random.default_rng(8243)
  inputs = np.ldexp(rng.uniform(-1, 1, size=(512, 3)),
                    rng.integers(-30, 31, size=(512, 3)))
  cancellation = inputs[:256].copy()
  cancellation[:, 2] = -(cancellation[:, 0] * cancellation[:, 1])
  inputs = np.concatenate((inputs, cancellation,
      np.array([[0, 1e20, 1e-30], [1e20, 0, 1e-30],
                [0, 1e-20, 1e20], [1e-20, 0, 1e20]], dtype=np.float64)))
  # The quotient is only requested where the source operation is finite.
  expected = np.zeros((len(inputs), 3), np.float64)
  for i in range(len(inputs)):
    oracle.scalar(inputs[i], expected[i])
  shader = _coupled_shader_source() + r"""
  kernel void test_scalar_source(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int base=9*int(tid);
    FlexDD a=FlexDD{input[base],input[base+1],input[base+2]};
    FlexDD b=FlexDD{input[base+3],input[base+4],input[base+5]};
    FlexDD c=FlexDD{input[base+6],input[base+7],input[base+8]};
    FlexDD product=flex_dd_mul(a,b);
    FlexDD quotient=(b.hi==0.0f) ? flex_dd(0.0f) : flex_dd_div(a,b);
    FlexDD fused=flex_dd_source_fma(a,b,c);
    common_ccd_store_dd3(output,base,FlexDD3{product,quotient,fused});
  }
  """
  lib = torch.mps.compile_shader(shader)
  source = torch.tensor(np.stack([_words(row) for row in inputs]),
                        dtype=torch.float32, device="mps")
  output = torch.zeros_like(source)
  lib.test_scalar_source(source, output, threads=len(inputs))
  words = output.cpu().numpy().reshape(-1, 3)
  actual = np.array([math.fsum(map(float, row)) for row in words]).reshape(-1, 3)
  products_and_fma = expected[:, (0, 2)]
  stored_words = _words(products_and_fma).reshape(-1, 3)
  stored = np.array([math.fsum(map(float, row)) for row in stored_words]).reshape(-1, 2)
  representable = stored == products_and_fma
  np.testing.assert_array_equal(actual[:, (0, 2)][representable],
                                products_and_fma[representable])
  # Three binary32 words cannot encode every binary64 subnormal or tiny
  # cancellation result. Keep those cases: their error must be bounded by
  # half the smallest storage-word quantum, rather than claiming exactness.
  unrepresentable = ~representable
  assert unrepresentable.any()
  error = np.abs(actual[:, (0, 2)] - products_and_fma)
  assert np.all(error[unrepresentable] <= np.ldexp(1.0, -150))
  finite_quotient = inputs[:, 1] != 0
  np.testing.assert_array_equal(actual[finite_quotient, 1],
                                expected[finite_quotient, 1])
