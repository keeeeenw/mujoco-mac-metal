# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Native EPA witnesses against compiled, unmodified pinned C arithmetic."""
import ctypes
import os
from pathlib import Path
import re
import subprocess

import mujoco
import numpy as np
import pytest

from test_common_ccd_matrix_source import _words


def _function(source, signature):
  match = re.search(signature + r"[^;{}]+\{", source)
  assert match is not None
  begin = match.start()
  end = match.end()
  depth = 1
  while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
  return source[begin:end]


def _pinned_oracle(tmp_path):
  root = os.getenv("MUJOCO_METAL_TRACE_ROOT")
  if not root:
    pytest.skip("pinned source root is required for the compiled C oracle")
  pinned = Path(root) / ".upstream/mujoco-3.10.0/src/engine/engine_collision_gjk.c"
  assert pinned.is_file(), "pinned GJK source is missing"
  source = pinned.read_text()
  license_header = source[:source.index("#include")]
  code = license_header + "#include <math.h>\n#include <stddef.h>\ntypedef double mjtNum;\n#define mju_abs fabs\n"
  code += _function(source, r"static inline void lincomb\(") + "\n"
  code += _function(source, r"static void triAffineCoord\(") + "\n"
  code += """
  void witness(const double *input, double *output) {
    // Three vertices: each has Minkowski, geom1 and geom2 coordinates.
    triAffineCoord(output,input,input+9,input+18,input+27);
    lincomb(output+3,output,3,input+3,input+12,input+21,NULL);
    lincomb(output+6,output,3,input+6,input+15,input+24,NULL);
  }
  """
  path = tmp_path / "pinned_epa.c"
  path.write_text(code)
  library = tmp_path / "pinned_epa.dylib"
  subprocess.run(["/usr/bin/clang", "-O3", "-std=c11", "-shared",
                  str(path), "-o", str(library)], check=True,
                 capture_output=True, text=True, timeout=30)
  lib = ctypes.CDLL(str(library))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  lib.witness.argtypes = [pointer, pointer]
  lib.witness.restype = None
  return lib


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_epa_witness_matches_unmodified_pinned_affine_and_lincomb(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  oracle = _pinned_oracle(tmp_path)
  rng = np.random.default_rng(4421)
  count = 32
  inputs = np.zeros((count, 30), np.float64)
  expected = np.zeros((count, 6), np.float64)
  for i in range(count):
    minkowski = rng.normal(size=(3, 3))
    # Wide absolute positions expose cancellation in witness reconstruction.
    second = rng.normal(size=(3, 3)) + (1e5 if i % 2 else 0)
    first = second + minkowski
    # Both native and C consume the exact same represented vertices.
    inputs[i, :27] = np.stack((minkowski, first, second), axis=1).reshape(-1)
    inputs[i, 27:] = minkowski.mean(axis=0)
    result = np.zeros(9, np.float64)
    oracle.witness(inputs[i], result)
    expected[i, :3] = (result[3:6] + result[6:9]) * .5
    expected[i, 3:] = result[3:6] - result[6:9]
    mujoco.mju_normalize3(expected[i, 3:])
  packed = np.stack([_words(row) for row in inputs])
  source = _coupled_shader_source() + r"""
  kernel void test_pinned_epa_witness(device const float* input [[buffer(0)]],
      device float* scratch [[buffer(1)]], device float* output [[buffer(2)]],
      uint tid [[thread_position_in_grid]]) {
    int base=90*int(tid);
    device FlexDDVertex* vertices=reinterpret_cast<device FlexDDVertex*>(scratch)+3*int(tid);
    for(int i=0;i<3;i++) {
      int start=base+27*i;
      vertices[i].m=common_ccd_load_dd3(input,start);
      vertices[i].a=common_ccd_load_dd3(input,start+9);
      vertices[i].b=common_ccd_load_dd3(input,start+18);
    }
    FlexEpaFace face;face.a=0;face.b=1;face.c=2;
    face.v_dd=common_ccd_load_dd3(input,base+81);
    face.dist2_dd=flex_dd3_dot(face.v_dd,face.v_dd);
    ContactGeom contact;FlexDD3 position,normal;
    flex_epa_write_witness_dd(face,vertices,contact,&position,&normal);
    common_ccd_store_dd3(output,18*int(tid),position);
    common_ccd_store_dd3(output,18*int(tid)+9,normal);
  }
  """
  library = torch.mps.compile_shader(source)
  device_input = torch.tensor(packed, dtype=torch.float32, device="mps")
  before = device_input.clone()
  scratch = torch.zeros((count * 3 * 27,), dtype=torch.float32, device="mps")
  output = torch.zeros((count, 18), dtype=torch.float32, device="mps")
  library.test_pinned_epa_witness(device_input, scratch, output, threads=count)
  represented = output.cpu().double().numpy().reshape(count, 6, 3).sum(axis=-1)
  np.testing.assert_array_equal(represented, expected)
  torch.testing.assert_close(device_input.view(torch.int32), before.view(torch.int32),
                             rtol=0, atol=0)
