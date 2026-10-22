# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Scalar force and transpose accumulation at actual paired input boundaries."""
import ctypes
import os
from pathlib import Path
import subprocess
import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native scalar force precision')
def test_scalar_force_and_gradient_match_installed_source_at_represented_words(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  code = r'''
  #include <mujoco/mujoco.h>
  void scalar_gradient(const double* x,double* out) {
    double force[8],jt[3];
    for(int r=0;r<8;r++) {
      double residual=x[32+r],D=1/x[24+r];
      force[r]=residual<0 ? -D*residual : 0;
      out[r]=force[r];
    }
    mju_mulMatTVec(jt,x,force,8,3);
    for(int i=0;i<3;i++)out[8+i]=x[40+i]-jt[i];
  }
  '''
  package = Path(mujoco.__file__).parent
  source, binary = tmp_path/'scalar_gradient.c', tmp_path/'scalar_gradient.dylib'
  source.write_text(code)
  subprocess.run(['/usr/bin/clang','-O3','-shared','-I',str(package/'include'),
                  str(source),str(package/'libmujoco.3.10.0.dylib'),
                  '-Wl,-rpath,'+str(package),'-o',str(binary)],check=True,
                 capture_output=True,text=True,timeout=30)
  oracle = ctypes.CDLL(str(binary))
  ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
  oracle.scalar_gradient.argtypes = [ptr,ptr]
  rng = np.random.default_rng(13579)
  packed, expected = [], []
  for case in range(12):
    J = rng.uniform(-1,1,(8,3))
    R = np.array([.007,.032,.12,.15,.42,.9,1.1,1.2])*(.125 if case%3==0 else 1)
    residual = rng.uniform(-300,50,8)
    residual[case%8] = 0
    if case%2: residual[::3] = np.abs(residual[::3])
    gauss = np.array([564.123456789,.00000123456789,-42.87654321])
    values = np.r_[J.ravel(),R,residual,gauss]
    high = values.astype(np.float32)
    # The producer can preserve an earlier high word and its correction.
    # Keep those exact words, including a noncanonical Jacobian decomposition.
    high[:24:2] = np.nextafter(high[:24:2],np.float32(np.inf))
    low = (values-high.astype(float)).astype(np.float32)
    represented = high.astype(float)+low.astype(float)
    out = np.zeros(11)
    oracle.scalar_gradient(np.ascontiguousarray(represented),out)
    packed.append(np.stack((high,low),axis=-1).ravel())
    expected.append(out)
  expected = np.asarray(expected)
  shader = _coupled_shader_source()+r'''
  kernel void scalar_gradient_control(device const float* in,device float* out,
      uint tid [[thread_position_in_grid]]) {
    device const float* x=in+86*tid;device float* y=out+30*tid;
    FlexDD forces[8];
    for(int r=0;r<8;r++) {
      PrimalFloatPair residual={x[2*(32+r)],x[2*(32+r)+1]};
      PrimalFloatPair R={x[2*(24+r)],x[2*(24+r)+1]};
      forces[r]=primal_pair_negative(residual)
        ? primal_scalar_source_force(residual,R) : FlexDD{0,0,0};
      y[3*r]=forces[r].hi;y[3*r+1]=forces[r].lo;y[3*r+2]=forces[r].tail;
    }
    for(int i=0;i<3;i++) {
      FlexDD jt={0,0,0};
      for(int r=0;r<8;r++) {
        PrimalFloatPair J={x[2*(3*r+i)],x[2*(3*r+i)+1]};
        jt=primal_source_transpose_force_add(jt,J,forces[r]);
      }
      PrimalFloatPair g=primal_source_gradient_pair(
          PrimalFloatPair{x[2*(40+i)],x[2*(40+i)+1]},jt);
      y[24+2*i]=g.hi;y[25+2*i]=g.lo;
    }
  }
  '''
  library = torch.mps.compile_shader(shader)
  inputs = torch.tensor(np.asarray(packed),device='mps')
  before = inputs.clone()
  outputs = torch.zeros((12,30),dtype=torch.float32,device='mps')
  library.scalar_gradient_control(inputs,outputs,threads=12)
  actual = outputs.cpu().double().numpy()
  np.testing.assert_array_equal(actual[:,:24].reshape(12,8,3).sum(-1),expected[:,:8])
  high = expected[:,8:].astype(np.float32).astype(float)
  published = high+(expected[:,8:]-high).astype(np.float32).astype(float)
  np.testing.assert_array_equal(actual[:,24:].reshape(12,3,2).sum(-1),published)
  torch.testing.assert_close(inputs.view(torch.int32),before.view(torch.int32),rtol=0,atol=0)
