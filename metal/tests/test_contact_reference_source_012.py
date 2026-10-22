# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Complete retained source constants and actual native reference operations."""
import ctypes
import os
from pathlib import Path
import subprocess
import mujoco
import numpy as np
import pytest


def test_contact_constants_retain_third_source_words_without_extra_kernel_binding():
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  from test_terrain_geometry_physics_012 import _model
  model = _model("box", "slope", False, "elliptic")
  desc = lower_coupled_constraints(model)
  # Source geometry mixing/clamping stays independently owned by lowering.
  for name, high, indices in (("solref", desc.solref, (0, 1)),
                              ("solimp", desc.solimp, tuple(range(2, 7))),
                              ("friction", desc.friction, tuple(range(11, 16)))):
    assert model.npair == 1
    source = np.asarray(getattr(model, "pair_" + name), np.float64)[0]
    actual = (high[0].astype(float) + desc.pair_const_lo[0, list(indices)].astype(float)
              + desc.pair_const_tail[0, list(indices)].astype(float))
    np.testing.assert_array_equal(actual, source)
  weight = np.asarray(model.body_invweight0)[np.asarray(model.geom_bodyid)[[0, 1]]].sum(axis=0)
  actual = (desc.pair_const_lo[0, [17, 19]].astype(float)
            + desc.pair_const_lo[0, [18, 20]].astype(float)
            + desc.pair_const_tail[0, [17, 19]].astype(float))
  np.testing.assert_array_equal(actual, weight)
  assert not desc.pair_const_tail.flags.writeable


@pytest.mark.parametrize("refsafe", [True, False])
@pytest.mark.parametrize("solref", [".00001 1", ".02 .7", "-500 -20"])
def test_source_constant_tails_preserve_clamping_refsafety_and_pair_offsets(refsafe, solref):
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".002"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/>
    <body pos="0 0 .1"><freejoint/><geom name="a" type="sphere" size=".15"/></body>
    <body pos=".5 0 .1"><freejoint/><geom name="b" type="sphere" size=".15"/></body>
    </worldbody><contact><pair geom1="floor" geom2="a" solref="{solref}"
       solimp="-1 2 -1 2 0"/><pair geom1="floor" geom2="b"
       solref=".02 1" solimp=".9 .95 .001 .5 2"/></contact></mujoco>''')
  if not refsafe:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
  desc = lower_coupled_constraints(model)
  assert desc.npairs == 3  # two explicit pairs plus the admitted sphere/sphere pair
  for i in range(2):
    sr = model.pair_solref[i].copy()
    if refsafe and sr[0] > 0:
      sr[0] = max(sr[0], 2 * model.opt.timestep)
    si = model.pair_solimp[i].copy()
    si[0:2] = np.clip(si[0:2], .0001, .9999)
    si[2] = max(si[2], 0)
    si[3] = min(.9999, max(.0001, si[3]))
    si[4] = max(si[4], 1)
    for expected, hi, indices in ((sr, desc.solref[i], [0, 1]),
                                  (si, desc.solimp[i], list(range(2, 7)))):
      actual = (hi.astype(float) + desc.pair_const_lo[i, indices].astype(float)
                + desc.pair_const_tail[i, indices].astype(float))
      np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native source reference")
def test_native_source_contact_reference_matches_literal_pinned_C(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  from test_common_ccd_epa_witness_source import _function
  root = os.environ.get("MUJOCO_METAL_TRACE_ROOT")
  assert root, "Pinned C source required for this actual native gate"
  text = (Path(root)/".upstream/mujoco-3.10.0/src/engine/engine_core_constraint.c").read_text()
  code = '#include <mujoco/mujoco.h>\n#include <math.h>\n'
  code += _function(text, r"mjtNum power") + '\n'
  code += _function(text, r"void getimpedance") + '\n'
  code += r'''
  void reference_probe(const double* in,double* out) {
    double K,B,I,IP;const double* sr=in;const double* si=in+2;
    K=sr[0]>0?1/mju_max(mjMINVAL,si[1]*si[1]*sr[0]*sr[0]*sr[1]*sr[1]):
      -sr[0]/mju_max(mjMINVAL,si[1]*si[1]);
    B=sr[1]>0?2/mju_max(mjMINVAL,si[1]*sr[0]):-sr[1]/si[1];
    getimpedance(si,in[7],in[8],&I,&IP);
    out[0]=K;out[1]=B;out[2]=I;out[3]=(1-I)*in[10]/I;
    out[4]=-B*in[9]-K*I*(in[7]-in[8]);
  }
  '''
  file, library = tmp_path/"reference.c", tmp_path/"reference.dylib"
  file.write_text(code)
  package = Path(mujoco.__file__).parent
  subprocess.run(["/usr/bin/clang","-O3","-shared","-I",str(package/"include"),
                  str(file),str(package/"libmujoco.3.10.0.dylib"),"-Wl,-rpath,"+str(package),
                  "-o",str(library)],check=True,capture_output=True,text=True,timeout=30)
  oracle = ctypes.CDLL(str(library))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  oracle.reference_probe.argtypes=[pointer,pointer]
  cases=[]
  for sr in ((.02,1),(.004,.7),(-500,-20)):
    for si in ((.95,.95,.001,.5,2),(.7,.93,.003,.31,1),(.7,.93,.003,.31,2),(.8,.95,0,.5,2)):
      for position in (0,.0002,-.0004,.0012,.004):
        cases.append([*sr,*si,position,.0001,.173,.17361111628512563])
  cases=np.asarray(cases,np.float64)
  expected=np.empty((len(cases),5))
  for i,case in enumerate(cases):oracle.reference_probe(case,expected[i])
  high=cases.astype(np.float32);low=(cases-high.astype(float)).astype(np.float32)
  tail=(cases-high.astype(float)-low.astype(float)).astype(np.float32)
  packed=np.stack((high,low,tail),axis=-1).reshape(len(cases),-1)
  shader=_coupled_shader_source()+r'''
  kernel void test_source_reference(device const float* in [[buffer(0)]],
      device float* out [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    FlexDD x[11];for(int i=0;i<11;i++)x[i]=FlexDD{in[33*tid+3*i],in[33*tid+3*i+1],in[33*tid+3*i+2]};
    FlexDD K=primal_source_contact_K(x[0],x[1],x[3]);
    FlexDD B=primal_source_contact_B(x[0],x[1],x[3]);
    bool supported;FlexDD I=primal_source_contact_impedance(x+2,x[7],x[8],FlexDD{0,0,0},supported);
    FlexDD result[5]={K,B,I,primal_source_contact_R(I,x[10]),
      primal_source_contact_aref(B,K,I,x[9],x[7],x[8])};
    for(int i=0;i<5;i++){out[15*tid+3*i]=result[i].hi;out[15*tid+3*i+1]=result[i].lo;out[15*tid+3*i+2]=result[i].tail;}
  }
  '''
  native=torch.mps.compile_shader(shader)
  inputs=torch.tensor(packed,device="mps");before=inputs.clone()
  outputs=torch.zeros((len(cases),15),device="mps")
  native.test_source_reference(inputs,outputs,threads=len(cases))
  represented=outputs.cpu().double().numpy().reshape(len(cases),5,3).sum(axis=-1)
  np.testing.assert_array_equal(represented,expected)
  torch.testing.assert_close(inputs.view(torch.int32),before.view(torch.int32),rtol=0,atol=0)
