# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Shrunk capsule endpoints against the unmodified pinned line callback."""
import ctypes
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pytest

from test_common_ccd_epa_witness_source import _function
from test_common_ccd_matrix_source import _words


def _line_oracle(tmp_path):
  root = os.getenv("MUJOCO_METAL_TRACE_ROOT")
  assert root, "Pinned source is required for the explicit native qualification"
  engine = Path(root) / ".upstream/mujoco-3.10.0/src/engine"
  header = (engine / "engine_collision_convex.h").read_text()
  struct = _function(header, r"struct _mjCCDObj")
  callback = _function((engine / "engine_collision_convex.c").read_text(),
                       r"void mjc_lineSupport\(")
  code = '#include <mujoco/mujoco.h>\n' + struct + ';\ntypedef struct _mjCCDObj mjCCDObj;\n'
  code += callback + "\n" + r"""
extern void mjc_initCCDObj(mjCCDObj*,const mjModel*,const mjData*,int,mjtNum);
void query(const mjModel* m,const mjData* d,int geom,
           const double* dirs,int count,double* out) {
  mjCCDObj object; mjc_initCCDObj(&object,m,d,geom,0);
  for(int i=0;i<count;i++) mjc_lineSupport(out+3*i,&object,dirs+3*i);
}
"""
  source, binary = tmp_path / "pinned_line.c", tmp_path / "pinned_line.dylib"
  source.write_text(code)
  package = Path(mujoco.__file__).parent
  subprocess.run(["/usr/bin/clang", "-O3", "-shared", "-I", str(package / "include"),
                  str(source), str(package / "libmujoco.3.10.0.dylib"),
                  "-Wl,-rpath," + str(package), "-o", str(binary)],
                 check=True, capture_output=True, text=True, timeout=30)
  lib = ctypes.CDLL(str(binary))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  lib.query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                       pointer, ctypes.c_int, pointer]
  lib.query.restype = None
  return lib


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native shrunk capsule support")
@pytest.mark.parametrize("rotation", ["0 0 0", "0 0 .002", ".3 -.2 .1"])
def test_shrunk_capsule_line_support_world_endpoints_match_pinned_source(rotation, tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
    <worldbody><geom type="capsule" size=".0625 .09375"
      pos=".06249999990625 .0001249999166667 .5470625" euler="{rotation}"/>
    </worldbody></mujoco>''')
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  rng = np.random.default_rng(11278)
  directions = np.concatenate((rng.normal(size=(32, 3)), np.eye(3), -np.eye(3),
                               np.zeros((1, 3))))
  expected = np.zeros((len(directions), 3), dtype=np.float64)
  _line_oracle(tmp_path).query(model._address, data._address, 0,
                              directions, len(directions), expected)
  packed = np.r_[_words(np.r_[data.geom_xpos[0], data.geom_xmat[0],
                              model.geom_size[0], 0]), np.zeros(36)].astype(np.float32)
  source = _coupled_shader_source() + r"""
  kernel void pinned_line_support(device const int* meta [[buffer(0)]],
      device const float* object [[buffer(1)]],device const float* dirs [[buffer(2)]],
      device float* output [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    CommonCCDSupportObject o=common_ccd_load_object(meta,object);
    CommonCCDSupportResult r=common_ccd_object_support(o,
        common_ccd_load_dd3(dirs,9*int(tid)),object,meta);
    common_ccd_store_dd3(output,9*int(tid),r.point);
  }
  """
  library = torch.mps.compile_shader(source)
  meta = torch.tensor([2, -1, 3, -1, 0, -1, -1, 0, 0], dtype=torch.int32, device="mps")
  geometry = torch.tensor(packed, dtype=torch.float32, device="mps")
  operands = torch.tensor(np.stack([_words(row) for row in directions]),
                          dtype=torch.float32, device="mps")
  saved = [value.clone() for value in (meta, geometry, operands)]
  output = torch.zeros((len(directions), 9), dtype=torch.float32, device="mps")
  library.pinned_line_support(meta, geometry, operands, output, threads=len(directions))
  actual = output.cpu().double().numpy().reshape(-1, 3, 3).sum(-1)
  np.testing.assert_array_equal(actual, expected)
  for value, before in zip((meta, geometry, operands), saved):
    torch.testing.assert_close(value.view(torch.int32), before.view(torch.int32),
                               rtol=0, atol=0)
