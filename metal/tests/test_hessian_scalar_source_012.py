# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Matrix-free scalar Hessian intermediates retain source precision until output."""
import ctypes
import os
import subprocess
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native scalar Hessian precision')
def test_scalar_hessian_source_intermediates_and_pair_boundary(tmp_path):
  import torch
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  c=r'''#include <math.h>
  void h(const double*x,double*y){const double*M=x,*J=M+36,*R=J+180,*p=R+30;
    for(int i=0;i<6;i++){double value=0;
      for(int k=0;k<6;k++)value=fma(M[i*6+k],p[k],value);
      for(int r=0;r<30;r++){double jd=0;
        for(int k=0;k<6;k++)jd=fma(J[r*6+k],p[k],jd);
        double D=1/R[r],coef=D*J[r*6+i];value=fma(coef,jd,value);}
      y[i]=value;}}
  '''
  source,binary=tmp_path/'h.c',tmp_path/'h.dylib';source.write_text(c)
  subprocess.run(['/usr/bin/clang','-O3','-shared',str(source),'-o',str(binary)],
                 check=True,capture_output=True,text=True,timeout=30)
  oracle=ctypes.CDLL(str(binary));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
  oracle.h.argtypes=[ptr,ptr]
  rng=np.random.default_rng(101203);packed=[];expected=[]
  for case in range(24):
    A=rng.uniform(-1,1,(6,6));M=A@A.T+np.eye(6)*.01
    J=rng.uniform(-1,1,(30,6));R=np.ldexp(rng.uniform(.5,1,30),rng.integers(-18,2,30))
    p=rng.uniform(-1,1,6)*np.ldexp(1.,(case%5)-2)
    values=np.r_[M.ravel(),J.ravel(),R,p];high=values.astype(np.float32)
    # Producers may retain an earlier, noncanonical high word.
    high[::3]=np.nextafter(high[::3],np.float32(np.inf))
    low=(values-high.astype(float)).astype(np.float32)
    words=np.stack((high,low),axis=-1);represented=words.astype(float).sum(-1)
    y=np.empty(6);oracle.h(np.ascontiguousarray(represented),y)
    expected.append(y);packed.append(words)
  expected=np.asarray(expected)
  shader=_coupled_shader_source()+r'''
  kernel void h_source_control(device const float*in,device float*out,
      uint tid [[thread_position_in_grid]]){
    device const float*M=in+504*tid,*J=M+72,*R=J+360,*p=R+60;
    for(int i=0;i<6;i++){
      FlexDD value={0,0,0};
      for(int k=0;k<6;k++)value=primal_source_transpose_force_add(value,
        PrimalFloatPair{M[2*(i*6+k)],M[2*(i*6+k)+1]},FlexDD{p[2*k],p[2*k+1],0});
      for(int r=0;r<30;r++){
        FlexDD jd={0,0,0};
        for(int k=0;k<6;k++)jd=primal_source_transpose_force_add(jd,
          PrimalFloatPair{J[2*(r*6+k)],J[2*(r*6+k)+1]},FlexDD{p[2*k],p[2*k+1],0});
        value=primal_hessian_scalar_source_add(value,PrimalFloatPair{R[2*r],R[2*r+1]},
          PrimalFloatPair{J[2*(r*6+i)],J[2*(r*6+i)+1]},jd);
      }
      PrimalFloatPair pair=primal_source_pair(value);device float*y=out+5*(6*tid+i);
      y[0]=value.hi;y[1]=value.lo;y[2]=value.tail;y[3]=pair.hi;y[4]=pair.lo;
    }
  }
  '''
  library=torch.mps.compile_shader(shader);inputs=torch.tensor(np.asarray(packed),device='mps')
  protected=inputs.clone();outputs=torch.zeros((24,6,5),device='mps')
  library.h_source_control(inputs,outputs,threads=24)
  actual=outputs.cpu().double().numpy()
  np.testing.assert_array_equal(actual[:,:,:3].sum(-1),expected)
  high=expected.astype(np.float32).astype(float);pair=high+(expected-high).astype(np.float32).astype(float)
  np.testing.assert_array_equal(actual[:,:,3:].sum(-1),pair)
  torch.testing.assert_close(inputs.view(torch.int32),protected.view(torch.int32),rtol=0,atol=0)
