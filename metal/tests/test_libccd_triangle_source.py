# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Actual native MPR closest points against unmodified pinned libccd C."""
import ctypes
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_common_ccd_matrix_source import _words


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_mpr_triangle_closest_matches_pinned_libccd(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source

  root = os.getenv("MUJOCO_METAL_TRACE_ROOT")
  assert root, "MUJOCO_METAL_TRACE_ROOT must identify the pinned source workspace"
  deps = Path(root) / ".upstream/mujoco-trace/build-fixed/_deps"
  source = deps / "ccd-src/src/vec3.c"
  assert source.is_file(), "pinned libccd source is missing"
  library_path = tmp_path / "pinned_libccd_vec3.dylib"
  subprocess.run([
      # Match libccd's pinned build: default C contractions, not fast
      # contraction across its separate scale/add witness statements.
      "/usr/bin/clang", "-O3", "-std=c11", "-shared",
      "-I", str(deps / "ccd-src/src"),
      "-I", str(deps / "ccd-build/src"), str(source),
      "-o", str(library_path)], check=True, capture_output=True, text=True,
      timeout=30)
  oracle = ctypes.CDLL(str(library_path))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  oracle.ccdVec3PointTriDist2.argtypes = [pointer] * 5
  oracle.ccdVec3PointTriDist2.restype = ctypes.c_double

  rng = np.random.default_rng(22134)
  triangles = rng.normal(size=(48, 3, 3))
  # Collinear triangles and thin portals exercise the original fallback and
  # rounded determinant branches, not just easy interior projections.
  triangles = np.concatenate((triangles, np.array([
      [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
      [[-1, 0, 0], [1, 0, 0], [2, 0, 0]],
      [[1, 1, 0], [-1, 1, 0], [0, 1, 0]],
      [[.2, .3, 1e-8], [.4, .6, 2e-8], [-.2, -.3, -1e-8]],
  ], np.float64)))
  base = rng.normal(size=(12, 3))
  line = rng.normal(size=(12, 3))
  thin = np.stack((base, base + line,
                   base - line + rng.normal(size=(12, 3)) * 1e-7), axis=1)
  triangles = np.concatenate((triangles, thin))
  count = len(triangles)
  expected = np.zeros((count, 3), np.float64)
  origin = np.zeros(3, np.float64)
  for i, triangle in enumerate(triangles):
    oracle.ccdVec3PointTriDist2(origin, *triangle, expected[i])
  assert np.isfinite(expected).all(), "the source fixture must have valid witnesses"

  shader = _coupled_shader_source(libccd=True) + r"""
  kernel void test_mpr_triangle_source(device const float* input [[buffer(0)]],
      device float* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int base=27*int(tid);
    FlexDD3 a=common_ccd_load_dd3(input,base);
    FlexDD3 b=common_ccd_load_dd3(input,base+9);
    FlexDD3 c=common_ccd_load_dd3(input,base+18);
    common_ccd_store_dd3(output,9*int(tid),mpr_triangle_closest(a,b,c));
  }
  """
  library = torch.mps.compile_shader(shader)
  inputs = torch.tensor(np.stack([_words(t.reshape(-1)) for t in triangles]),
                        dtype=torch.float32, device="mps")
  before = inputs.clone()
  output = torch.zeros((count, 9), dtype=torch.float32, device="mps")
  library.test_mpr_triangle_source(inputs, output, threads=count)
  words = output.detach().cpu().numpy().reshape(count, 3, 3)
  actual = np.asarray([[math.fsum(v) for v in row] for row in words])
  np.testing.assert_array_equal(actual, expected)
  torch.testing.assert_close(inputs.view(torch.int32), before.view(torch.int32),
                             rtol=0, atol=0)
