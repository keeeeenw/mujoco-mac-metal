# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Execute installed MuJoCo support callbacks against the native support kernel."""
import ctypes
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pytest

from test_common_ccd_epa_witness_source import _function
from test_common_ccd_matrix_source import _words
from test_common_ccd_rigid_native import _pair_xml


def _oracle(tmp_path):
  root = os.getenv("MUJOCO_METAL_TRACE_ROOT")
  if not root:
    pytest.skip("pinned source root required for the callback ABI")
  header = (Path(root) / ".upstream/mujoco-3.10.0/src/engine/engine_collision_convex.h").read_text()
  struct = _function(header, r"struct _mjCCDObj")
  code = header[:header.index("#ifndef")] + "#include <mujoco/mujoco.h>\n"
  code += struct + ";\ntypedef struct _mjCCDObj mjCCDObj;\n"
  code += """
  extern void mjc_initCCDObj(mjCCDObj*,const mjModel*,const mjData*,int,mjtNum);
  void query(const mjModel* m,const mjData* d,int geom,
             const double* dirs,int count,double* output) {
    mjCCDObj object; mjc_initCCDObj(&object,m,d,geom,0);
    for(int i=0;i<count;i++) object.support(output+3*i,&object,dirs+3*i);
  }
  """
  source = tmp_path / "installed_support.c"
  source.write_text(code)
  package = Path(mujoco.__file__).parent
  binary = package / "libmujoco.3.10.0.dylib"
  assert binary.is_file(), "this callback ABI is pinned to MuJoCo 3.10.0"
  library = tmp_path / "installed_support.dylib"
  subprocess.run(["/usr/bin/clang", "-O3", "-shared", "-I", str(package / "include"),
                  str(source), str(binary), "-Wl,-rpath," + str(package),
                  "-o", str(library)], check=True, capture_output=True,
                 text=True, timeout=30)
  result = ctypes.CDLL(str(library))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  result.query.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,
                          pointer,ctypes.c_int,pointer]
  result.query.restype = None
  return result


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize("rotation", [None, "17 31 -23"])
def test_native_cylinder_support_matches_installed_callback(tmp_path, rotation):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  model = mujoco.MjModel.from_xml_string(_pair_xml(5,5,rotation=rotation))
  data = mujoco.MjData(model)
  data.qpos[:] = data.qpos.astype(np.float32)
  mujoco.mj_forward(model,data)
  rng = np.random.default_rng(33721)
  directions = rng.normal(size=(32,3))
  directions /= np.linalg.norm(directions,axis=-1,keepdims=True)
  directions = np.concatenate((directions, np.array([
      [.9999952938095764,.0030679567629247314,0],
      [1,0,0],[-1,0,0],[0,1,0],[0,0,1],[0,0,-1]],np.float64)))
  expected = np.zeros_like(directions)
  _oracle(tmp_path).query(model._address,data._address,1,directions,
                         len(directions),expected)
  packed = np.concatenate((_words(np.concatenate((data.geom_xpos[1],
      data.geom_xmat[1],model.geom_size[1],[0]))),np.zeros(36,np.float32)))
  source = _coupled_shader_source() + r"""
  kernel void installed_support_query(device const int* meta [[buffer(0)]],
      device const float* object [[buffer(1)]],device const float* dirs [[buffer(2)]],
      device float* output [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    CommonCCDSupportObject o=common_ccd_load_object(meta,object);
    CommonCCDSupportResult r=common_ccd_object_support(o,
        common_ccd_load_dd3(dirs,9*int(tid)),object,meta);
    common_ccd_store_dd3(output,9*int(tid),r.point);
  }
  """
  library = torch.mps.compile_shader(source)
  meta = torch.tensor([7,1,5,-1,0,-1,-1,0,0],dtype=torch.int32,device="mps")
  geometry = torch.tensor(packed,dtype=torch.float32,device="mps")
  operands = torch.tensor(np.stack([_words(row) for row in directions]),
                          dtype=torch.float32,device="mps")
  before = [tensor.clone() for tensor in (meta,geometry,operands)]
  output = torch.zeros((len(directions),9),dtype=torch.float32,device="mps")
  library.installed_support_query(meta,geometry,operands,output,
                                  threads=len(directions))
  actual = output.cpu().double().numpy().reshape(-1,3,3).sum(axis=-1)
  np.testing.assert_array_equal(actual,expected)
  for tensor,saved in zip((meta,geometry,operands),before):
    torch.testing.assert_close(tensor.view(torch.int32),saved.view(torch.int32),
                               rtol=0,atol=0)
