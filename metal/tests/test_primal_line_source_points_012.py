# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Scalar line evaluation preserves source operations through the pair boundary."""
import ctypes
import os
import subprocess
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv('MUJOCO_METAL_RUN_GPU') != '1', reason='native line source precision')
def test_scalar_line_points_match_source_with_protected_noncanonical_inputs(tmp_path):
  import mujoco
  import torch
  from pathlib import Path
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  from mujoco_metal.constraint_jacobian import PackedJacobianLayout, initialize_packed_jacobian_records

  # Scalar/equality restriction of pinned MuJoCo 3.10 PrimalPrepare/PrimalEval
  # (28009f9105cd92784b7b0b30c0605a5e29107a77). Installed mju functions supply
  # the independent matrix/vector and dot operations; no native physics input
  # or emitted iterate is replaced by this control.
  code = r'''#include <mujoco/mujoco.h>
  void reference(const double*M,const double*J,const double*R,const double*v,
      const double*g,const double*res,int nv,int n,int ne,double a,double*out){
    double Mv[16],Jv[32];mju_mulMatVec(Mv,M,v,nv,nv);mju_mulMatVec(Jv,J,v,n,nv);
    double q0=0,q1=mju_dot(v,g,nv),q2=.5*mju_dot(v,Mv,nv),cost=0;
    for(int i=0;i<n;i++){
      double D=1/R[i],DJ0=D*res[i],c0=.5*(res[i]*DJ0),c1=Jv[i]*DJ0;
      double c2=.5*(Jv[i]*D*Jv[i]);
      if(i<ne){q1+=c1;q2+=c2;continue;}
      double x=res[i]+a*Jv[i],cost0=res[i]<0?c0:0;
      if(x<0){q0+=c0-cost0;q1+=c1;q2+=c2;}else cost-=cost0;
    }
    out[0]=cost+(a*a*q2+a*q1+q0);out[1]=2*a*q2+q1;out[2]=2*q2;
  }'''
  package=Path(mujoco.__file__).parent
  c,binary=tmp_path/'line.c',tmp_path/'line.dylib';c.write_text(code)
  subprocess.run(['/usr/bin/clang','-O3','-shared','-I',str(package/'include'),
                  str(c),str(package/'libmujoco.3.10.0.dylib'),
                  '-Wl,-rpath,'+str(package),'-o',str(binary)],
                 check=True,capture_output=True,text=True,timeout=30)
  reference=ctypes.CDLL(str(binary)).reference
  pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
  reference.argtypes=[pointer]*6+[ctypes.c_int]*3+[ctypes.c_double,pointer]
  shader=_coupled_shader_source()+r'''
  kernel void line_point_control(device const float*in,device const float*J,
      constant int*dims,device const float*debug,device const int*flags,
      device float*out,device float*scratch,uint tid [[thread_position_in_grid]]){
    if(tid!=0)return;int nv=dims[0],n=dims[9];
    device const float*M=in,*R=M+nv*nv,*Rl=R+n,*v=Rl+n,*vl=v+nv,
      *g=vl+nv,*gl=g+nv,*res=gl+nv,*rl=res+n,*zero=rl+n,
      *lo=zero+nv,*hi=lo+n,*ar=hi+n;
    PrimalFloatPair der,sec;
    PrimalFloatPair cost=primal_accel_cost_delta_elliptic({ar[n],ar[n+1]},
      M,false,debug,dims,J,R,ar,lo,hi,zero,zero,zero,v,vl,flags,flags+n,
      flags+2*n,flags+2*n,ar,0,scratch,scratch+nv,nv,n,n,dims[1],
      der,sec,g,gl,res,rl,Rl,nullptr);
    out[0]=cost.hi;out[1]=cost.lo;out[2]=der.hi;out[3]=der.lo;
    out[4]=sec.hi;out[5]=sec.lo;
  }'''
  library=torch.mps.compile_shader(shader);rng=np.random.default_rng(120016)
  for case,nv in enumerate([1,2,3,4,5,6,7,8,9,12,15,16]):
    n=1+(case*7)%32;ne=case%(n+1)
    A=rng.uniform(-1,1,(nv,nv));M=A@A.T+.125*np.eye(nv)
    J=rng.uniform(-1,1,(n,nv));R=np.ldexp(rng.uniform(.5,1,n),rng.integers(-12,5,n))
    v=rng.uniform(-2,2,nv);g=rng.uniform(-50,50,nv);res=rng.uniform(-2,2,n)
    pairs=[]
    for value in [M,J,R,v,g,res]:
      high=value.astype(np.float32)
      high.flat[::3]=np.nextafter(high.flat[::3],np.float32(np.inf))
      low=(value-high.astype(float)).astype(np.float32)
      pairs.append(np.stack((high,low),-1))
    M,J,R,v,g,res=pairs
    dims=np.zeros(64+10*n,np.int32)
    dims[0]=nv;dims[1]=ne;dims[7]=512;dims[9]=n;dims[24]=32
    debug=np.zeros(n*n+10*n+10*nv+nv*n+256,np.float32)
    debug[:nv*nv]=M[:,:,1].ravel()
    layout=PackedJacobianLayout.create(n,nv,mode=0,paired=True)
    packed=np.zeros(layout.stride_words,np.float32)
    initialize_packed_jacobian_records(packed.view(np.int32),1,layout)
    packed[layout.values_offset:layout.values_offset+layout.nnz]=J[:,:,0].ravel()
    packed[layout.lo_values_offset():layout.lo_values_offset()+layout.nnz]=J[:,:,1].ravel()
    flags=np.r_[np.ones(n),np.zeros(n),np.full(n,-1)].astype(np.int32)
    for alpha in [0.,2**-24,.375,1.,1.75,4.,-.5]:
      high=np.nextafter(np.float32(alpha),np.float32(np.inf))
      alpha_words=np.asarray([high,np.float32(alpha-float(high))])
      values=np.concatenate([M[:,:,0].ravel(),R[:,0],R[:,1],v[:,0],v[:,1],
          g[:,0],g[:,1],res[:,0],res[:,1],np.zeros(nv),np.zeros(n),
          np.full(n,np.inf),np.zeros(n),alpha_words]).astype(np.float32)
      buffers=[torch.tensor(a,device='mps') for a in [values,packed,dims,debug,flags]]
      protected=[a.clone() for a in buffers];outputs=torch.zeros(6,device='mps')
      library.line_point_control(*buffers,outputs,torch.zeros(2*nv,device='mps'),threads=1)
      expected=np.zeros(3)
      reference(*[np.ascontiguousarray(a.astype(float).sum(-1)) for a in pairs],
                nv,n,ne,float(alpha_words.astype(float).sum()),expected)
      high=expected.astype(np.float32)
      low=(expected-high.astype(float)).astype(np.float32)
      np.testing.assert_array_equal(outputs.cpu().numpy().view(np.uint32),
                                    np.stack((high,low),-1).ravel().view(np.uint32))
      for actual,before in zip(buffers,protected):
        torch.testing.assert_close(actual.view(torch.int32),before.view(torch.int32),rtol=0,atol=0)
