# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native GJK triangle steps against unmodified compiled pinned C."""
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_common_ccd_epa_witness_source import _function
from test_common_ccd_matrix_source import _words


def _pinned_triangle_oracle(tmp_path):
  root = os.getenv("MUJOCO_METAL_TRACE_ROOT")
  if not root:
    pytest.skip("pinned source root is required for the compiled C oracle")
  pinned = Path(root) / ".upstream/mujoco-3.10.0"
  path = pinned / "src/engine/engine_collision_gjk.c"
  assert path.is_file(), "pinned GJK source is missing"
  source = path.read_text()
  code = source[:source.index("#include")] + """
#include <math.h>
#include <stddef.h>
#include <float.h>
#include <mujoco/mjtype.h>
#define mjMAX_LIMIT DBL_MAX
#define mju_abs fabs
#define mju_sqrt sqrt
"""
  for name in ("sub3", "dot3", "norm3", "scl3", "cross3", "projectOriginPlane",
               "projectOriginLine", "sameSign2", "lincomb", "S1D", "S2D"):
    code += _function(source, r"static(?: inline)? (?:void|int|mjtNum) " + name + r"\(") + "\n"
  code += """
void triangle(const double *input, double *output) {
  S2D(output,input,input+3,input+6);
  lincomb(output+3,output,3,input,input+3,input+6,NULL);
}
"""
  src = tmp_path / "pinned_triangle.c"
  src.write_text(code)
  libpath = tmp_path / "pinned_triangle.dylib"
  subprocess.run(["/usr/bin/clang", "-O3", "-ffp-contract=fast", "-shared",
                  "-I", str(pinned / "include"), str(src), "-o", str(libpath)],
                 check=True, capture_output=True, text=True, timeout=30)
  lib = ctypes.CDLL(str(libpath))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  lib.triangle.argtypes = [pointer, pointer]
  lib.triangle.restype = None
  return lib


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_gjk_triangle_matches_pinned_subdistance_and_closest_point(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  oracle = _pinned_triangle_oracle(tmp_path)
  rng = np.random.default_rng(22414)
  inputs = rng.normal(size=(32, 9))
  # Exact degeneracies exercise S2D's source line and vertex fallbacks.
  degenerate = np.array([
      [1,0,0, 2,0,0, 3,0,0],
      [-1,0,0, 1,0,0, 2,0,0],
      [0,1,0, 0,1,0, 0,1,0],
      [1,1,0, -1,1,0, 0,1,0],
  ], dtype=np.float64)
  inputs = np.concatenate((inputs, degenerate))
  count = len(inputs)
  expected = np.zeros((count, 6), np.float64)
  for i in range(count):
    oracle.triangle(inputs[i], expected[i])
  shader = _coupled_shader_source() + r"""
  kernel void test_pinned_triangle(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int base=27*int(tid), out=18*int(tid);
    FlexDD3 a=common_ccd_load_dd3(input,base);
    FlexDD3 b=common_ccd_load_dd3(input,base+9);
    FlexDD3 c=common_ccd_load_dd3(input,base+18);
    FlexDD lambda[3]; flex_dd_s2d(lambda,a,b,c);
    FlexDD3 values[3]={a,b,c};
    common_ccd_store_dd3(output,out,FlexDD3{lambda[0],lambda[1],lambda[2]});
    common_ccd_store_dd3(output,out+9,flex_dd3_lincomb_source(values,lambda,3));
  }
  """
  library = torch.mps.compile_shader(shader)
  device_input = torch.tensor(np.stack([_words(row) for row in inputs]),
                              dtype=torch.float32, device="mps")
  before = device_input.clone()
  output = torch.zeros((count, 18), dtype=torch.float32, device="mps")
  library.test_pinned_triangle(device_input, output, threads=count)
  represented = output.cpu().double().numpy().reshape(count, 6, 3).sum(axis=-1)
  np.testing.assert_array_equal(represented, expected)
  torch.testing.assert_close(device_input.view(torch.int32), before.view(torch.int32),
                             rtol=0, atol=0)
