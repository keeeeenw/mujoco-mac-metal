# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Canonical solver storage must round the complete source value at both words."""
import ctypes
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native source pair publication')
def test_source_pair_publication_matches_C_at_midpoints_binades_and_subnormal_words(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source, binary = tmp_path/'pair.c', tmp_path/'pair.dylib'
  source.write_text('void publish(const double*x,float*y){y[0]=(float)x[0];y[1]=(float)(x[0]-(double)y[0]);}\n')
  subprocess.run(['/usr/bin/clang','-O3','-shared',str(source),'-o',str(binary)],
                 check=True,capture_output=True,text=True,timeout=30)
  oracle = ctypes.CDLL(str(binary))
  oracle.publish.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')]
  values = [-1137.108712366278, 0.]
  for exponent in (-100,-50,-1,0,1,50,100):
    high = np.float32(np.ldexp(1.,exponent))
    next_high = np.nextafter(high,np.float32(np.inf))
    midpoint = (float(high)+float(next_high))/2
    for value in (float(high),midpoint,np.nextafter(midpoint,-np.inf),
                  np.nextafter(midpoint,np.inf),float(next_high)):
      values.extend((value,-value))
  # Storage words can be subnormal even when the complete value is normal.
  for mantissa in (1,2,3,0x3fffff,0x7fffff,0x800001):
    value = float(np.array([mantissa],np.uint32).view(np.float32)[0])
    values.extend((value,-value))
  rng = np.random.default_rng(60421)
  values.extend(np.ldexp(rng.uniform(-1.,1.,256),rng.integers(-80,81,256)))
  values = np.asarray(values,np.float64)
  high = values.astype(np.float32)
  # Preserve noncanonical producer high words; publication owns normalization.
  high[2::3] = np.nextafter(high[2::3],np.float32(np.inf))
  low = (values-high.astype(float)).astype(np.float32)
  tail = (values-high.astype(float)-low.astype(float)).astype(np.float32)
  packed = np.stack((high,low,tail),axis=-1)
  represented = packed.astype(float).sum(-1)
  expected = np.empty((len(values),2),np.float32)
  for i,value in enumerate(represented):oracle.publish(np.array([value]),expected[i])
  shader = _coupled_shader_source()+r'''
  kernel void source_pair_control(device const float* in,device float* out,
      uint tid [[thread_position_in_grid]]) {
    device const float*x=in+3*tid;
    PrimalFloatPair result=primal_source_pair(FlexDD{x[0],x[1],x[2]});
    out[2*tid]=result.hi;out[2*tid+1]=result.lo;
  }
  '''
  library = torch.mps.compile_shader(shader)
  inputs = torch.tensor(packed,device='mps')
  before = inputs.clone()
  outputs = torch.zeros((len(values),2),device='mps')
  library.source_pair_control(inputs,outputs,threads=len(values))
  # Zero signs are immaterial; nonzero storage words must match literally,
  # including subnormal words without performing flushed device arithmetic.
  actual = outputs.cpu().numpy()
  np.testing.assert_array_equal(actual,expected)
  torch.testing.assert_close(inputs.view(torch.int32),before.view(torch.int32),rtol=0,atol=0)
