# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Scalar refresh uses installed source dot/subtract order at raw native words."""
import ctypes
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native source refresh")
def test_scalar_refresh_dynamic_dot_residual_force_gradient_and_pair_boundary(tmp_path):
  import torch
  from mujoco_metal.constraint_jacobian import (
      PackedJacobianLayout, initialize_packed_jacobian_records)
  from mujoco_metal.coupled_constraints import _coupled_shader_source

  source = tmp_path / "refresh.c"
  binary = tmp_path / "refresh.dylib"
  source.write_text(r'''#include <mujoco/mujoco.h>
  void oracle(const double*M,const double*rhs,const double*J,const double*R,
              const double*ar,const double*acc,int nv,double*out) {
    double Ma[16],jar[5],force[5],jt[16];
    mju_mulMatVec(Ma,M,acc,nv,nv);mju_mulMatVec(jar,J,acc,5,nv);
    for(int r=0;r<5;r++){jar[r]-=ar[r];double D=1/R[r];
      force[r]=jar[r]<0?-D*jar[r]:0;out[16+r]=jar[r];out[21+r]=force[r];}
    mju_mulMatTVec(jt,J,force,5,nv);
    for(int i=0;i<nv;i++)out[i]=Ma[i]-rhs[i]-jt[i];
  }''')
  package = Path(mujoco.__file__).parent
  subprocess.run([
      "/usr/bin/clang", "-O3", "-shared", "-I", str(package / "include"),
      str(source), str(package / "libmujoco.3.10.0.dylib"),
      "-Wl,-rpath," + str(package), "-o", str(binary)],
      check=True, capture_output=True, text=True, timeout=30)
  oracle = ctypes.CDLL(str(binary))
  ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
  oracle.oracle.argtypes = [ptr] * 6 + [ctypes.c_int, ptr]

  def words(values):
    values = np.asarray(values, dtype=float)
    high = values.astype(np.float32)
    high.reshape(-1)[::3] = np.nextafter(
        high.reshape(-1)[::3], np.float32(np.inf))
    low = (values - high.astype(float)).astype(np.float32)
    return high, low

  rng = np.random.default_rng(120031)
  widths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16] * 2
  layouts = [PackedJacobianLayout.create(5, n, mode=0, paired=True) for n in widths]
  stride = max(layout.stride_words for layout in layouts)
  count = len(widths)
  mass = np.zeros((count, 256), np.float32)
  debug = np.zeros((count, 512), np.float32)
  jacobian = np.zeros((count, stride), np.float32)
  acc = np.zeros((count, 32), np.float32)
  regularizer = np.zeros((count, 10), np.float32)
  reference = np.zeros((count, 10), np.float32)
  dims = np.zeros((count, 192), np.int32)
  expected = np.zeros((count, 26))
  for case, (n, layout) in enumerate(zip(widths, layouts)):
    a = rng.normal(size=(n, n))
    M = a @ a.T + np.eye(n) * .125
    J = rng.normal(size=(5, n))
    rhs = rng.normal(size=n) * 8
    acceleration = rng.normal(size=n) * (1 if case < 12 else 1e-4)
    R = np.exp2(rng.uniform(-12, 3, 5))
    ar = rng.normal(size=5) * .25
    mh, ml = words(M)
    jh, jl = words(J)
    rh, rl = words(rhs)
    ah, al = words(acceleration)
    Rh, Rl = words(R)
    bh, bl = words(ar)
    mass[case, :n*n] = mh.ravel()
    debug[case, 256:256+n*n] = ml.ravel()
    debug[case, :n] = rh
    debug[case, n:2*n] = rl
    initialize_packed_jacobian_records(jacobian[case].view(np.int32), 1, layout)
    jacobian[case, layout.values_offset:layout.values_offset+layout.nnz] = jh.ravel()
    start = layout.lo_values_offset()
    jacobian[case, start:start+layout.nnz] = jl.ravel()
    acc[case, :n], acc[case, n:2*n] = ah, al
    regularizer[case, :5], regularizer[case, 5:] = Rh, Rl
    reference[case, :5], reference[case, 5:] = bh, bl
    dims[case, 1], dims[case, 7], dims[case, 9] = n, 512, 5
    dims[case, 24] = 64
    dims[case, 64+50+21] = 256
    inputs = [mh.astype(float)+ml, rh.astype(float)+rl,
              jh.astype(float)+jl, Rh.astype(float)+Rl,
              bh.astype(float)+bl, ah.astype(float)+al]
    oracle.oracle(*[np.ascontiguousarray(x) for x in inputs], n, expected[case])

  shader = _coupled_shader_source() + r'''
  kernel void source_refresh_control(device const float*M,device const float*dbg,
      device const float*J,device const float*a,device const float*R,
      device const float*ar,constant int*dims,constant int*stride,
      device float*out,uint tid [[thread_position_in_grid]]) {
    M+=tid*256;dbg+=tid*512;J+=tid*stride[0];a+=tid*32;
    R+=tid*10;ar+=tid*10;dims+=tid*192;out+=tid*110;
    int n=dims[1];FlexDD force[5];
    for(int r=0;r<5;r++) {
      FlexDD residual=primal_refresh_source_residual(dbg,dims,J,r,n,a,a+n,ar,ar+5);
      force[r]=residual.hi+residual.lo+residual.tail<0?
        primal_refresh_source_force(residual,PrimalFloatPair{R[r],R[5+r]}):FlexDD{0,0,0};
      out[80+3*r]=residual.hi;out[81+3*r]=residual.lo;out[82+3*r]=residual.tail;
      out[95+3*r]=force[r].hi;out[96+3*r]=force[r].lo;out[97+3*r]=force[r].tail;
    }
    for(int i=0;i<n;i++) {
      FlexDD gauss=primal_refresh_source_gauss(M,false,dbg,dims,i,n,a,a+n,dbg);
      FlexDD jt={0,0,0};
      for(int r=0;r<5;r++)jt=flex_dd_source_fma(
        primal_refresh_scalar_input53(FlexDD{ccj_get(J,r,i,n),ccj_get_lo(J,r,i,n),0}),force[r],jt);
      FlexDD g=flex_dd_source_sub(gauss,jt);PrimalFloatPair pair=primal_source_pair(g);
      out[5*i]=g.hi;out[5*i+1]=g.lo;out[5*i+2]=g.tail;
      out[5*i+3]=pair.hi;out[5*i+4]=pair.lo;
    }
  }'''
  library = torch.mps.compile_shader(shader)
  tensors = [torch.tensor(x, device="mps") for x in
             [mass, debug, jacobian, acc, regularizer, reference, dims]]
  protected = [x.clone() for x in tensors]
  outputs = torch.zeros((count, 110), device="mps")
  library.source_refresh_control(*tensors,
      torch.tensor([stride], dtype=torch.int32, device="mps"), outputs, threads=count)
  actual = outputs.cpu().double().numpy()
  for case, n in enumerate(widths):
    g = actual[case, :80].reshape(16, 5)[:n]
    np.testing.assert_array_equal(g[:, :3].sum(-1), expected[case, :n])
    high = expected[case, :n].astype(np.float32).astype(float)
    pair = high + (expected[case, :n]-high).astype(np.float32).astype(float)
    np.testing.assert_array_equal(g[:, 3:].sum(-1), pair)
    np.testing.assert_array_equal(actual[case, 80:95].reshape(5, 3).sum(-1),
                                  expected[case, 16:21])
    np.testing.assert_array_equal(actual[case, 95:].reshape(5, 3).sum(-1),
                                  expected[case, 21:])
  for before, after in zip(protected, tensors):
    torch.testing.assert_close(before.view(torch.int32), after.view(torch.int32), rtol=0, atol=0)
