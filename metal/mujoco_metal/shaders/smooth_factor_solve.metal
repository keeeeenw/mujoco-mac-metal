// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline bool factored_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

// Factor once, solve multiple changing RHS without refactoring. Each world
// owns a compiled-sized device slice; no model-sized thread-local arrays.
kernel void dense_factor(
    device const float* matrix [[buffer(0)]],
    device float* factor [[buffer(1)]],
    device int* pivots [[buffer(2)]],
    device int* status [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint world [[thread_position_in_grid]]) {
  uint n = uint(dims[0]), batch = uint(dims[1]);
  if (world >= batch) return;
  uint base = world*n*n;
  bool general = dims[3] != 0;
  int failed = 0;
  float scale = 0;
  for (uint i=0; i<n*n; ++i) {
    float x = matrix[base+i];
    if (!factored_finite(x)) failed=1;
    else scale=max(scale,abs(x));
  }
  if (failed) { status[world]=failed; return; }
  if (!general) {
    float tol = scale*0.000003814697265625f;
    for (uint row=0; row<n; ++row)
      for (uint col=row+1; col<n; ++col)
        if (abs(matrix[base+row*n+col]-matrix[base+col*n+row]) > tol) failed=2;
    if (failed) { status[world]=failed; return; }
  }
  for (uint row=0; row<n; ++row) {
    pivots[world*n+row]=int(row);
    for (uint col=0; col<n; ++col) {
      float x=matrix[base+row*n+col];
      factor[base+row*n+col] = general || row==col ? x
          : 0.5f*x+0.5f*matrix[base+col*n+row];
    }
  }
  if (general) {
    for (uint k=0; k<n; ++k) {
      uint pivot=k;
      float magnitude=abs(factor[base+k*n+k]);
      for (uint row=k+1; row<n; ++row) {
        float x=abs(factor[base+row*n+k]);
        if (x>magnitude) { magnitude=x; pivot=row; }
      }
      if (!factored_finite(magnitude)) { failed=4; break; }
      if (!(magnitude>1e-15f)) { failed=3; break; }
      pivots[world*n+k]=int(pivot);
      if (pivot!=k) {
        for (uint col=0; col<n; ++col) {
          float x=factor[base+k*n+col];
          factor[base+k*n+col]=factor[base+pivot*n+col];
          factor[base+pivot*n+col]=x;
        }
      }
      for (uint row=k+1; row<n; ++row) {
        float multiplier=factor[base+row*n+k]/factor[base+k*n+k];
        if (!factored_finite(multiplier)) { failed=4; break; }
        factor[base+row*n+k]=multiplier;
        for (uint col=k+1; col<n; ++col) {
          float x=factor[base+row*n+col]-multiplier*factor[base+k*n+col];
          if (!factored_finite(x)) { failed=4; break; }
          factor[base+row*n+col]=x;
        }
        if (failed) break;
      }
      if (failed) break;
    }
  } else {
    for (uint k=0; k<n; ++k) {
      float diagonal=factor[base+k*n+k];
      for (uint j=0; j<k; ++j) {
        float x=factor[base+k*n+j]; diagonal-=x*x;
      }
      if (!factored_finite(diagonal)) { failed=4; break; }
      if (!(diagonal>0)) { failed=3; break; }
      float root=sqrt(diagonal);
      if (!factored_finite(root) || !(root>0)) { failed=4; break; }
      factor[base+k*n+k]=root;
      for (uint row=k+1; row<n; ++row) {
        float x=factor[base+row*n+k];
        for (uint j=0; j<k; ++j)
          x-=factor[base+row*n+j]*factor[base+k*n+j];
        x/=root;
        if (!factored_finite(x)) { failed=4; break; }
        factor[base+row*n+k]=x;
      }
      if (failed) break;
    }
  }
  status[world]=failed;
}

kernel void dense_solve_factored(
    device const float* factor [[buffer(0)]],
    device const int* pivots [[buffer(1)]],
    device const int* factor_status [[buffer(2)]],
    device const float* rhs [[buffer(3)]],
    device float* solution [[buffer(4)]],
    device int* status [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    uint world [[thread_position_in_grid]]) {
  uint n=uint(dims[0]), batch=uint(dims[1]), nrhs=uint(dims[2]);
  if (world>=batch) return;
  uint base=world*n*n, rb=world*n*nrhs;
  bool general=dims[3]!=0;
  int failed=factor_status[world];
  for (uint i=0; i<n*nrhs; ++i) {
    solution[rb+i]=0;
    if (!failed && !factored_finite(rhs[rb+i])) failed=1;
  }
  if (failed) { status[world]=failed; return; }
  for (uint i=0; i<n*nrhs; ++i) solution[rb+i]=rhs[rb+i];
  if (general) {
    for (uint k=0; k<n; ++k) {
      uint pivot=uint(pivots[world*n+k]);
      for (uint col=0; col<nrhs; ++col) {
        float x=solution[rb+k*nrhs+col];
        solution[rb+k*nrhs+col]=solution[rb+pivot*nrhs+col];
        solution[rb+pivot*nrhs+col]=x;
      }
    }
  }
  for (uint col=0; col<nrhs; ++col) {
    for (uint row=0; row<n; ++row) {
      float x=solution[rb+row*nrhs+col];
      for (uint j=0; j<row; ++j)
        x-=factor[base+row*n+j]*solution[rb+j*nrhs+col];
      if (!general) x/=factor[base+row*n+row];
      if (!factored_finite(x)) { failed=4; break; }
      solution[rb+row*nrhs+col]=x;
    }
    if (failed) break;
    for (int row=int(n)-1; row>=0; --row) {
      float x=solution[rb+uint(row)*nrhs+col];
      for (uint j=uint(row)+1; j<n; ++j)
        x-=factor[base+(general ? uint(row)*n+j : j*n+uint(row))]
            *solution[rb+j*nrhs+col];
      x/=factor[base+uint(row)*n+uint(row)];
      if (!factored_finite(x)) { failed=4; break; }
      solution[rb+uint(row)*nrhs+col]=x;
    }
    if (failed) break;
  }
  if (failed) for (uint i=0; i<n*nrhs; ++i) solution[rb+i]=0;
  status[world]=failed;
}
