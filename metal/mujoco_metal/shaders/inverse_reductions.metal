// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline float2 inverse_two_sum(float a,float b) {
  float h=a+b, bv=h-a;
  return float2(h,(a-(h-bv))+(b-bv));
}
inline float2 inverse_pair_add(float2 a,float2 b) {
  float2 sum=inverse_two_sum(a.x,b.x);
  float tail=a.y+b.y;
  return inverse_two_sum(sum.x,tail+sum.y);
}
inline float2 inverse_product(float a,float b) {
  float h=a*b;
  return float2(h,fma(a,b,-h));
}

// The immutable host-compiled CSR map is copied once per query. Numerical
// operands remain borrowed device views; no row/entry loop launches Torch
// operations and no sparse matrix is materialized as dense scratch.
// Header: batch,nr,nv,nnz,stride,values_offset,low_offset,paired, then rowptr
// and sorted columns. Both native actions accumulate in pinned row/DOF order.
kernel void inverse_csr_pair_matvec(
    device const float* packed [[buffer(0)]],
    device const float* acceleration [[buffer(1)]],
    device const int* map [[buffer(2)]],
    device float* high [[buffer(3)]],
    device float* low [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  int nr=map[1], nv=map[2];
  if (int(tid)>=map[0]*nr) return;
  int row=int(tid)%nr, world=int(tid)/nr;
  int begin=map[8+row], end=map[9+row], colbase=9+nr;
  int base=world*map[4];
  float2 acc=float2(0.0f);
  for (int k=begin;k<end;k++) {
    int col=map[colbase+k];
    float a=acceleration[world*nv+col];
    acc=inverse_pair_add(acc,inverse_product(packed[base+map[5]+k],a));
    if (map[7]) acc=inverse_pair_add(acc,inverse_product(packed[base+map[6]+k],a));
  }
  high[tid]=acc.x; low[tid]=acc.y;
}

kernel void inverse_csr_transpose_pair_matvec(
    device const float* packed [[buffer(0)]],
    device const float* force_high [[buffer(1)]],
    device const float* force_low [[buffer(2)]],
    device const bool* live [[buffer(3)]],
    device const int* map [[buffer(4)]],
    device float* high [[buffer(5)]],
    device float* low [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int nr=map[1], nv=map[2];
  if (int(tid)>=map[0]*nv) return;
  int col=int(tid)%nv, world=int(tid)/nv;
  int colbase=9+nr, base=world*map[4];
  float2 acc=float2(0.0f);
  for (int row=0;row<nr;row++) {
    // Mask before loading invalid inactive numerical words: zero*NaN is NaN.
    if (!live[world*nr+row]) continue;
    int begin=map[8+row], end=map[9+row];
    int left=begin,right=end;
    while (left<right) {
      int middle=left+(right-left)/2;
      if (map[colbase+middle]<col) left=middle+1;
      else right=middle;
    }
    if (left==end || map[colbase+left]!=col) continue;
    float j=packed[base+map[5]+left];
    float f=force_high[world*nr+row], fl=force_low[world*nr+row];
    acc=inverse_pair_add(acc,inverse_product(j,f));
    acc=inverse_pair_add(acc,inverse_product(j,fl));
    if (map[7]) {
      float jl=packed[base+map[6]+left];
      acc=inverse_pair_add(acc,inverse_product(jl,f));
      acc=inverse_pair_add(acc,inverse_product(jl,fl));
    }
  }
  high[tid]=acc.x; low[tid]=acc.y;
}
