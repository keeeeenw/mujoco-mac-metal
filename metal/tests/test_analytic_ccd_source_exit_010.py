# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Real GJK stopping values and EPA boundary outcomes against installed C."""
import ctypes
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pytest

from test_analytic_boundaries_010 import _fixture
from test_common_ccd_epa_witness_source import _function
from test_common_ccd_matrix_source import _words


def _installed_ccd(tmp_path):
  assert mujoco.__version__ == "3.10.0"
  root = os.getenv("MUJOCO_METAL_TRACE_ROOT")
  assert root, "Pinned internal headers are required for the opt-in native gate"
  engine = Path(root) / ".upstream/mujoco-3.10.0/src/engine"
  obj = _function((engine / "engine_collision_convex.h").read_text(),
                  r"struct _mjCCDObj")
  header = (engine / "engine_collision_gjk.h").read_text()
  types = header[header.index("// Status of an EPA run"):
                 header.index("// return size in bytes")]
  code = '#include <stdlib.h>\n#include <string.h>\n#include <mujoco/mujoco.h>\n'
  code += obj + ';\ntypedef struct _mjCCDObj mjCCDObj;\n' + types
  code += """
extern void mjc_initCCDObj(mjCCDObj*,const mjModel*,const mjData*,int,mjtNum);
extern size_t mjc_ccdSize(int);
extern mjtNum mjc_ccd(const mjCCDConfig*,mjCCDStatus*,mjCCDObj*,mjCCDObj*);
void query(const mjModel* m,const mjData* d,int a,int b,int contacts,double* out) {
  mjCCDObj x,y; mjc_initCCDObj(&x,m,d,a,0); mjc_initCCDObj(&y,m,d,b,0);
  mjCCDConfig c={m->opt.ccd_iterations,m->opt.ccd_tolerance,contacts,0,NULL};
  c.buffer=malloc(mjc_ccdSize(c.max_iterations));
  mjCCDStatus s; memset(&s,0,sizeof(s));
  out[0]=mjc_ccd(&c,&s,&x,&y); out[1]=s.gjk_iterations;
  out[2]=s.nsimplex; out[3]=s.epa_status;
  free(c.buffer);
}
"""
  package = Path(mujoco.__file__).parent
  src, binary = tmp_path / "installed_ccd.c", tmp_path / "installed_ccd.dylib"
  src.write_text(code)
  subprocess.run(["/usr/bin/clang", "-O3", "-shared", "-I",
                  str(package / "include"), str(src),
                  str(package / "libmujoco.3.10.0.dylib"),
                  "-Wl,-rpath," + str(package), "-o", str(binary)],
                 check=True, capture_output=True, text=True, timeout=30)
  lib = ctypes.CDLL(str(binary))
  lib.query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int,
                       np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")]
  return lib


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native C oracle")
def test_actual_gjk_stopping_value_and_public_epa_boundary_match_installed_c(tmp_path):
  import torch
  from mujoco_metal import MetalSimulation
  oracle = _installed_ccd(tmp_path)
  cases = [("cylinder", "box", "flat", False),
           ("cylinder", "box", "long", False),
           ("ellipsoid", "box", "long", False),
           ("cylinder", "box", "flat", True),
           ("ellipsoid", "box", "flat", True),
           ("ellipsoid", "box", "ordinary", True),
           ("ellipsoid", "box", "long", True),
           ("ellipsoid", "box", "flat", False)]
  for first, second, aspect, reverse in cases:
    model, _ = _fixture(first, second, aspect, reverse)
    q = model.qpos0.astype(np.float32)
    q[:3], q[7:10] = [0, 0, .5], [.125, 0, .5]
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    sim = MetalSimulation(model, 1, qpos=q[None], profile="integrated_euler_v1")
    assembly = sim.assembled_system(recompute=True)
    np.testing.assert_array_equal(assembly["status"].cpu().numpy(), [0])
    cc = sim._coupled_constraints
    count = int(cc._workspace["contact_pair_count"][0].cpu())
    assert count == data.ncon
    a, b = int(cc.descriptor.geom1[0]), int(cc.descriptor.geom2[0])
    expected = np.zeros(4)
    oracle.query(model._address, data._address, a, b, 0, expected)
    assert 0 <= expected[0] < 1e-20 and expected[2] == 3
    # Exact represented object poses, sizes and model CCD options. Probe the
    # same shared GJK used by public candidate generation, with owned outputs.
    def f(value):
      return torch.tensor(value, dtype=torch.float32, device="mps")
    def i(value):
      return torch.tensor(value, dtype=torch.int32, device="mps")
    kind = {4: 6, 5: 7, 6: 4}
    metadata = i(np.asarray([
        [kind[int(model.geom_type[g])], g, int(model.geom_type[g]), -1, 0, -1, -1, 0, 0]
        for g in (a, b)]).reshape(-1))
    objects = f(np.concatenate([
        np.r_[_words(np.r_[data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], 0]),
              np.zeros(36)] for g in (a, b)]))
    result, status = f(np.zeros(21)), i(np.zeros(6))
    cc._library.common_ccd_source_gjk_probe(
        i([1, model.opt.ccd_iterations]), metadata, objects,
        f(_words(np.r_[data.geom_xpos[a], data.geom_xpos[b]])),
        f(_words([model.opt.ccd_tolerance, 0])), f([0]), i([0]),
        result, status, f(np.zeros(108)), i(np.zeros(8)),
        threads=(1,), group_size=(1,))
    actual = result.cpu().double().numpy().reshape(-1, 3).sum(-1)
    if expected[0] >= 1e-30:
      np.testing.assert_array_equal(actual[:1], expected[:1])
    else:
      # At ~1e-33 a binary64 residual may need a word below Metal float's
      # smallest subnormal. Keep an explicit one-word underflow bound;
      # this is not blanket relative tolerance or binary64 emulation.
      assert abs(actual[0] - expected[0]) <= float(np.finfo(np.float32).smallest_subnormal)
    assert status[0].item() == 0
    assert status[1].item() == expected[2] and status[2].item() == expected[1]
    oracle.query(model._address, data._address, a, b, 1, expected)
    if (first, second, aspect, reverse) == ("ellipsoid", "box", "flat", False):
      # Pinned P3_ORIGIN_ON_FACE is a valid no-contact outcome, not a failed
      # world. The separate invalid-support/capacity gates must still fail.
      assert expected[0] == 0 and expected[3] == 8 and count == 0
    else:
      assert expected[0] < 0 and expected[3] == 0 and count > 0
