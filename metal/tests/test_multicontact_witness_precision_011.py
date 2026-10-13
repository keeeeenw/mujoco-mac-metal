# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source witness arithmetic supplements actual thin-hull physics regressions."""
import os
import mujoco
import numpy as np
import pytest

from test_common_ccd_matrix_source import _words
from test_common_ccd_bridge_cpu import _mesh_hfield_model
from mujoco_metal.common_ccd_bridge import build_common_ccd_model_upload


def test_compiled_polygon_normal_nonfinite_input_is_rejected():
  model = _mesh_hfield_model()
  model.mesh_polynormal[0, 0] = np.nan
  with pytest.raises(ValueError, match="polygon normals are nonfinite"):
    build_common_ccd_model_upload(model, np.zeros(15, np.float32),
                                 np.zeros(9 * model.ngeom, np.int32))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native witness arithmetic")
def test_multicontact_source_witness_subtraction_and_normalization():
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  direction = np.array([-.2229495635918975, .2793395896604891, .9339501516368537])
  inputs, expected = [], []
  for origin in (.5, 1e5, -1e5, -.001):
    for distance in (.0075, .000075, .000001, 1e-9, 0., 1e-16):
      point = np.array([origin, origin + .125, origin - .25])
      displacement = direction * distance
      for flip in (False, True):
        x1, x2 = point - displacement, point.copy()
        if flip:
          x1, x2 = x2, x1
        normal = x1 - x2
        mujoco.mju_normalize3(normal)
        expected.append(np.r_[.5 * (x1 + x2), normal])
        inputs.append(np.r_[point, displacement, float(flip)])
  packed = np.stack([_words(row) for row in inputs])
  shader = _coupled_shader_source() + r"""
  kernel void check_multicontact_source_witness(
      device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
      uint tid [[thread_position_in_grid]]) {
    int base=21*int(tid);
    ContactGeom contact; FlexDD3 position,normal;
    common_ccd_multicontact_witness(common_ccd_load_dd3(input,base),
        common_ccd_load_dd3(input,base+9),input[base+18]!=0,
        contact,&position,&normal);
    common_ccd_store_dd3(output,18*int(tid),position);
    common_ccd_store_dd3(output,18*int(tid)+9,normal);
  }
  """
  library = torch.mps.compile_shader(shader)
  device_input = torch.tensor(packed, dtype=torch.float32, device="mps")
  output = torch.zeros((len(inputs), 18), dtype=torch.float32, device="mps")
  library.check_multicontact_source_witness(device_input, output, threads=len(inputs))
  actual = output.cpu().double().numpy().reshape(len(inputs), 6, 3).sum(axis=-1)
  np.testing.assert_array_equal(actual, np.asarray(expected))
