# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native tetrahedral GJK reduction against the pinned C compiler mode."""
import ctypes
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_common_ccd_epa_witness_source import _function
from test_common_ccd_matrix_source import _words


def _pinned_tetra_oracle(tmp_path):
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
  for name in ("sub3", "dot3", "det3", "norm3", "scl3", "cross3",
               "projectOriginPlane", "projectOriginLine", "sameSign2",
               "lincomb", "S1D", "S2D", "S3D"):
    code += _function(
        source, r"static(?: inline)? (?:void|int|mjtNum) " + name + r"\(") + "\n"
  code += """
void tetra(const double *input, double *output) {
  S3D(output,input,input+3,input+6,input+9);
  lincomb(output+4,output,4,input,input+3,input+6,input+9);
}
"""
  src = tmp_path / "pinned_tetra.c"
  src.write_text(code)
  libpath = tmp_path / "pinned_tetra.dylib"
  # Matches the pinned engine build's default C expression contractions.
  subprocess.run(["/usr/bin/clang", "-O3", "-std=c11", "-shared",
                  "-I", str(pinned / "include"), str(src), "-o", str(libpath)],
                 check=True, capture_output=True, text=True, timeout=30)
  lib = ctypes.CDLL(str(libpath))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  lib.tetra.argtypes = [pointer, pointer]
  lib.tetra.restype = None
  return lib


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_gjk_tetra_matches_pinned_subdistance_and_closest_point(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  oracle = _pinned_tetra_oracle(tmp_path)
  rng = np.random.default_rng(22414)
  random = rng.normal(size=(64, 12))
  degenerate = np.array([
      [1,0,0, 2,0,0, 3,0,0, 4,0,0],
      [-1,0,0, 1,0,0, 2,0,0, 3,0,0],
      [0,1,0, 0,1,0, 0,1,0, 0,1,0],
      [1,1,0, -1,1,0, 0,-1,0, 0,0,0],
      [1,0,0, 0,1,0, 0,0,1, -1,-1,-1],
  ], dtype=np.float64)
  thin = random.copy().reshape(-1, 4, 3)
  thin[:, :, 2] *= 1e-10
  inputs = np.concatenate((random, random * 1e-8, random * 1e8,
                           thin.reshape(-1, 12), degenerate))
  count = len(inputs)
  expected = np.zeros((count, 7), np.float64)
  for i in range(count):
    oracle.tetra(inputs[i], expected[i])
  assert np.isfinite(expected).all()
  shader = _coupled_shader_source() + r"""
  kernel void test_pinned_tetra(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int base=36*int(tid), out=21*int(tid);
    FlexDD3 a=common_ccd_load_dd3(input,base);
    FlexDD3 b=common_ccd_load_dd3(input,base+9);
    FlexDD3 c=common_ccd_load_dd3(input,base+18);
    FlexDD3 d=common_ccd_load_dd3(input,base+27);
    FlexDD lambda[4]; flex_dd_s3d(lambda,a,b,c,d);
    FlexDD3 values[4]={a,b,c,d};
    common_ccd_store_dd3(output,out,FlexDD3{lambda[0],lambda[1],lambda[2]});
    output[out+9]=lambda[3].hi;
    output[out+10]=lambda[3].lo;
    output[out+11]=lambda[3].tail;
    common_ccd_store_dd3(output,out+12,flex_dd3_lincomb_source(values,lambda,4));
  }
  """
  library = torch.mps.compile_shader(shader)
  device_input = torch.tensor(np.stack([_words(row) for row in inputs]),
                              dtype=torch.float32, device="mps")
  before = device_input.clone()
  output = torch.zeros((count, 21), dtype=torch.float32, device="mps")
  library.test_pinned_tetra(device_input, output, threads=count)
  words = output.cpu().numpy().reshape(count, 7, 3)
  represented = np.array([math.fsum(map(float, row))
                          for row in words.reshape(-1, 3)]).reshape(count, 7)
  np.testing.assert_array_equal(represented, expected)
  torch.testing.assert_close(device_input.view(torch.int32), before.view(torch.int32),
                             rtol=0, atol=0)
