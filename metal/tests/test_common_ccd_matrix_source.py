# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Compare actual native CCD matrix operations with pinned compiled queries."""
import ctypes
import os
from pathlib import Path
import subprocess
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


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_terrain_relative_matrix_product_matches_pinned_compiled_source(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  from test_common_ccd_epa_witness_source import _function

  root = os.environ.get("MUJOCO_METAL_TRACE_ROOT")
  assert root, "Pinned source headers are required for this opt-in native gate"
  inline = (Path(root) / ".upstream/mujoco-3.10.0/src/engine/engine_inline.h").read_text()
  function = _function(inline, r"void mji_mulMatTMat3")
  source_path, library_path = tmp_path / "terrain_matrix.c", tmp_path / "terrain_matrix.dylib"
  source_path.write_text('#include <assert.h>\n#include <mujoco/mujoco.h>\n' + function)
  package = Path(mujoco.__file__).parent
  subprocess.run(["/usr/bin/clang", "-O3", "-shared", "-I", str(package / "include"),
                  str(source_path), "-o", str(library_path)], check=True,
                 capture_output=True, text=True, timeout=30)
  oracle = ctypes.CDLL(str(library_path))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  oracle.mji_mulMatTMat3.argtypes = [pointer, pointer, pointer]

  rng = np.random.default_rng(12014)
  matrices = rng.normal(size=(64, 2, 3, 3))
  # Include the actual compiled terrain/object frames from the failing
  # public box/slope fixture, whose relative matrix changes an EPA witness.
  from test_terrain_geometry_physics_012 import _model, _state
  model = _model("box", "slope", False, "pyramidal")
  qpos, qvel = _state(model)
  data = mujoco.MjData(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  matrices[0] = data.geom_xmat.reshape(2, 3, 3)
  matrices[1] = [np.eye(3), matrices[0, 1]]
  matrices[2] = [matrices[0, 0], np.eye(3)]
  expected = np.empty((len(matrices), 9), dtype=np.float64)
  for i, (a, b) in enumerate(matrices):
    oracle.mji_mulMatTMat3(expected[i], np.ascontiguousarray(a.reshape(-1)),
                          np.ascontiguousarray(b.reshape(-1)))
  packed = np.stack([_words(pair.reshape(-1)) for pair in matrices])
  shader = _coupled_shader_source() + r"""
  kernel void test_terrain_relative_matrix(device const float* input,
      device float* output,uint tid [[thread_position_in_grid]]) {
    FlexDD a[9],b[9],result[9];int base=54*int(tid);
    for(int i=0;i<9;i++) {
      a[i]=FlexDD{input[base+3*i],input[base+3*i+1],input[base+3*i+2]};
      b[i]=FlexDD{input[base+27+3*i],input[base+28+3*i],input[base+29+3*i]};
    }
    common_ccd_matrix_transpose_product(a,b,result);
    for(int i=0;i<9;i++) {
      output[27*int(tid)+3*i]=result[i].hi;
      output[27*int(tid)+3*i+1]=result[i].lo;
      output[27*int(tid)+3*i+2]=result[i].tail;
    }
  }
  """
  library = torch.mps.compile_shader(shader)
  inputs = torch.tensor(packed, device="mps", dtype=torch.float32)
  outputs = torch.zeros((len(matrices), 27), device="mps", dtype=torch.float32)
  before = inputs.clone()
  library.test_terrain_relative_matrix(inputs, outputs, threads=len(matrices))
  actual = outputs.cpu().double().numpy().reshape(-1, 9, 3).sum(-1)
  np.testing.assert_array_equal(actual, expected)
  torch.testing.assert_close(inputs.view(torch.int32), before.view(torch.int32), rtol=0, atol=0)
