# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Nearest source quotient rounding uses the operands, including midpoint ties."""
import ctypes
import os
from pathlib import Path
import subprocess
import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native source division')
def test_source_division_matches_C_near_midpoints_binades_and_signed_scales(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  package = Path(mujoco.__file__).parent
  source, binary = tmp_path/'quotient.c', tmp_path/'quotient.dylib'
  source.write_text('void quotient(const double*x,double*y){y[0]=x[0]/x[1];}\n')
  subprocess.run(['/usr/bin/clang','-O3','-shared',str(source),'-o',str(binary)],
                 check=True,capture_output=True,text=True,timeout=30)
  oracle = ctypes.CDLL(str(binary))
  pointer = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
  oracle.quotient.argtypes = [pointer,pointer]
  cases = []
  # These legitimate paired denominators place 1/R almost exactly at a
  # binary64 midpoint. A three-digit approximation selects the wrong side.
  for value in (.12,.015):
    high = np.float32(value)
    low = np.float32(value-float(high))
    denominator = float(high)+float(low)
    for a_sign in (-1,1):
      for b_sign in (-1,1):cases.append([float(a_sign),b_sign*denominator])
  for exponent in (-25,-1,0,1,25):
    for value in (np.ldexp(1.,exponent),np.nextafter(np.ldexp(1.,exponent),0.),
                  np.nextafter(np.ldexp(1.,exponent),np.inf)):
      for sign in (-1,1):cases.extend([[sign*value,1.],[sign,value]])
  rng = np.random.default_rng(7821)
  for _ in range(256):
    magnitudes = np.ldexp(rng.uniform(.5,1,2),rng.integers(-25,26,2))
    signs = rng.choice([-1.,1.],2)
    cases.append(magnitudes*signs)
  cases = np.asarray(cases,np.float64)
  expected = np.empty((len(cases),1))
  for i,case in enumerate(cases):oracle.quotient(np.ascontiguousarray(case),expected[i])
  high = cases.astype(np.float32)
  low = (cases-high.astype(float)).astype(np.float32)
  tail = (cases-high.astype(float)-low.astype(float)).astype(np.float32)
  packed = np.stack((high,low,tail),axis=-1).reshape(len(cases),6)
  np.testing.assert_array_equal(packed.astype(float).reshape(-1,2,3).sum(-1),cases)
  shader = _coupled_shader_source()+r'''
  kernel void source_division_control(device const float* in,device float* out,
      uint tid [[thread_position_in_grid]]) {
    device const float*x=in+6*tid;
    FlexDD result=flex_dd_source_div(FlexDD{x[0],x[1],x[2]},FlexDD{x[3],x[4],x[5]});
    out[3*tid]=result.hi;out[3*tid+1]=result.lo;out[3*tid+2]=result.tail;
  }
  '''
  library = torch.mps.compile_shader(shader)
  inputs = torch.tensor(packed,device='mps')
  before = inputs.clone()
  outputs = torch.zeros((len(cases),3),device='mps')
  library.source_division_control(inputs,outputs,threads=len(cases))
  np.testing.assert_array_equal(outputs.cpu().double().numpy().sum(-1),expected[:,0])
  torch.testing.assert_close(inputs.view(torch.int32),before.view(torch.int32),rtol=0,atol=0)
