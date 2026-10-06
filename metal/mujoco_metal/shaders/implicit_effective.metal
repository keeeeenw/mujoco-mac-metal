// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline bool ie_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

struct IEFloatPair { float hi; float lo; };
inline IEFloatPair ie_pair_add(IEFloatPair a, IEFloatPair b) {
  float sum=a.hi+b.hi;
  float virtual_b=sum-a.hi;
  float err=(a.hi-(sum-virtual_b))+(b.hi-virtual_b);
  float tail=a.lo+b.lo+err;
  float total=sum+tail;
  float vtotal=total-sum;
  float low=(sum-(total-vtotal))+(tail-vtotal);
  return {total,low};
}
inline IEFloatPair ie_pair_product(float a, float b) {
  float product=a*b;
  return {product,fma(a,b,-product)};
}
inline IEFloatPair ie_pair_scale(IEFloatPair a, float b) {
  IEFloatPair high=ie_pair_product(a.hi,b);
  IEFloatPair low=ie_pair_product(a.lo,b);
  return ie_pair_add(high,low);
}

kernel void effective_validate_edge_values(
    device const float* edge_values [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    device int* status [[buffer(2)]],
    uint world [[thread_position_in_grid]]) {
  int edge_count=dims[1], batch=dims[2];
  if (int(world)>=batch || status[world]!=0) return;
  int base=int(world)*max(edge_count,1);
  for (int e=0; e<edge_count; ++e)
    if (!ie_finite(edge_values[base+e])) { status[world]=1; return; }
}

kernel void effective_coo_matvec(
    device const float* basis [[buffer(0)]],
    device const float* edge_values [[buffer(1)]],
    device const int* edge_rows [[buffer(2)]],
    device const int* edge_cols [[buffer(3)]],
    device const int* active_dof [[buffer(4)]],
    device const int* cycle_iterations [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device const int* status [[buffer(7)]],
    device const int* done [[buffer(8)]],
    device const int* cycle_stop [[buffer(9)]],
    device float* output [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0], edge_count=dims[1], batch=dims[2], m=dims[3];
  if (nv<=0 || int(tid)>=batch*nv) return;
  int w=int(tid)/nv, row=int(tid)%nv, j=cycle_iterations[w];
  int basis_base=w*(m+1)*nv+j*nv, vector_base=w*nv;
  int edge_base=w*max(edge_count,1);
  float sum=0.0f;
  if (status[w]==0 && done[w]==0 && cycle_stop[w]==0
      && active_dof[vector_base+row]!=0) {
    for (int e=0; e<edge_count; ++e) {
      int r=edge_rows[e], c=edge_cols[e];
      if (r==row && r>=0 && r<nv && c>=0 && c<nv
          && active_dof[vector_base+c]!=0) {
        float value=edge_values[edge_base+e];
        float product=value*basis[basis_base+c];
        if (!ie_finite(value) || !ie_finite(product)) {
          sum=as_type<float>(0x7fc00000u);
          break;
        }
        sum+=product;
      }
    }
  }
  output[vector_base+row]=sum;
}

kernel void effective_coo_vector_matvec(
    device const float* vector [[buffer(0)]],
    device const float* edge_values [[buffer(1)]],
    device const int* edge_rows [[buffer(2)]],
    device const int* edge_cols [[buffer(3)]],
    device const int* active_dof [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0], edge_count=dims[1], batch=dims[2];
  if (nv<=0 || int(tid)>=batch*nv) return;
  int world=int(tid)/nv, row=int(tid)%nv;
  int base=world*nv, edge_base=world*max(edge_count,1);
  float sum=0.0f;
  if (active_dof[base+row]!=0) {
    for (int e=0; e<edge_count; ++e) {
      int r=edge_rows[e], c=edge_cols[e];
      if (r==row && r>=0 && r<nv && c>=0 && c<nv
          && active_dof[base+c]!=0) {
        float value=edge_values[edge_base+e]*vector[base+c];
        if (!ie_finite(value)) { sum=as_type<float>(0x7fc00000u); break; }
        sum+=value;
      }
    }
  }
  output[base+row]=sum;
}

kernel void effective_combine_operator(
    device const float* mass_product [[buffer(0)]],
    device const float* derivative_product [[buffer(1)]],
    device const int* active_dof [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    constant float* params [[buffer(4)]],
    device int* status [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2];
  if (int(world)>=batch) return;
  int w=int(world), base=w*nv;
  if (status[w]!=0) {
    for (int d=0; d<nv; ++d) output[base+d]=0.0f;
    return;
  }
  for (int d=0; d<nv; ++d) {
    if (active_dof[base+d]==0) { output[base+d]=0.0f; continue; }
    float m=mass_product[base+d], derivative=derivative_product[base+d];
    float value=m-params[0]*derivative;
    if (!ie_finite(m) || !ie_finite(derivative) || !ie_finite(value)) {
      status[w]=1;
      for (int k=0; k<nv; ++k) output[base+k]=0.0f;
      return;
    }
    output[base+d]=value;
  }
}

kernel void effective_coo_vector_matvec_pair(
    device const float* vector_hi [[buffer(0)]],
    device const float* vector_low [[buffer(1)]],
    device const float* edge_values [[buffer(2)]],
    device const int* edge_rows [[buffer(3)]],
    device const int* edge_cols [[buffer(4)]],
    device const int* active_dof [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device float* output_hi [[buffer(7)]],
    device float* output_low [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0], edge_count=dims[1], batch=dims[2];
  if (nv<=0 || int(tid)>=batch*nv) return;
  int world=int(tid)/nv, row=int(tid)%nv;
  int base=world*nv, edge_base=world*max(edge_count,1);
  IEFloatPair sum={0.0f,0.0f};
  if (active_dof[base+row]!=0) {
    for (int e=0; e<edge_count; ++e) {
      int r=edge_rows[e], c=edge_cols[e];
      if (r==row && r>=0 && r<nv && c>=0 && c<nv
          && active_dof[base+c]!=0) {
        float coefficient=edge_values[edge_base+e];
        IEFloatPair term=ie_pair_product(coefficient,vector_hi[base+c]);
        sum=ie_pair_add(sum,term);
        term=ie_pair_product(coefficient,vector_low[base+c]);
        sum=ie_pair_add(sum,term);
      }
    }
  }
  output_hi[base+row]=sum.hi;
  output_low[base+row]=sum.lo;
}

kernel void effective_combine_operator_pair(
    device const float* mass_hi [[buffer(0)]],
    device const float* mass_low [[buffer(1)]],
    device const float* derivative_hi [[buffer(2)]],
    device const float* derivative_low [[buffer(3)]],
    device const int* active_dof [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    constant float* params [[buffer(6)]],
    device int* status [[buffer(7)]],
    device float* output_hi [[buffer(8)]],
    device float* output_low [[buffer(9)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2];
  if (int(world)>=batch) return;
  int w=int(world), base=w*nv;
  if (status[w]!=0) {
    for (int d=0; d<nv; ++d) {
      output_hi[base+d]=0.0f;
      output_low[base+d]=0.0f;
    }
    return;
  }
  for (int d=0; d<nv; ++d) {
    if (active_dof[base+d]==0) {
      output_hi[base+d]=0.0f;
      output_low[base+d]=0.0f;
      continue;
    }
    IEFloatPair derivative={derivative_hi[base+d],derivative_low[base+d]};
    IEFloatPair product=ie_pair_scale(derivative,params[0]);
    IEFloatPair value=ie_pair_add(
        {mass_hi[base+d],mass_low[base+d]}, {-product.hi,-product.lo});
    if (!ie_finite(value.hi) || !ie_finite(value.lo)) {
      status[w]=1;
      for (int k=0; k<nv; ++k) {
        output_hi[base+k]=0.0f;
        output_low[base+k]=0.0f;
      }
      return;
    }
    output_hi[base+d]=value.hi;
    output_low[base+d]=value.lo;
  }
}

kernel void effective_gmres_prepare_cycle(
    device const float* preconditioned_residual [[buffer(0)]],
    device const int* active_dof [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    device int* status [[buffer(3)]],
    device int* done [[buffer(4)]],
    device int* cycle_iterations [[buffer(5)]],
    device int* cycle_stop [[buffer(6)]],
    device float* basis [[buffer(7)]],
    device float* hessenberg [[buffer(8)]],
    device float* givens_cos [[buffer(9)]],
    device float* givens_sin [[buffer(10)]],
    device float* residual_rhs [[buffer(11)]],
    device float* rhs_norms [[buffer(12)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2], m=dims[3];
  if (int(world)>=batch) return;
  int w=int(world), base=w*nv, basis_base=w*(m+1)*nv;
  int hbase=w*(m+1)*m, gbase=w*(m+1), cbase=w*m;
  if (status[w]!=0 || done[w]!=0 || cycle_stop[w]!=2) return;
  float square=0.0f;
  for (int i=0; i<nv; ++i) {
    float value=active_dof[base+i]!=0?preconditioned_residual[base+i]:0.0f;
    if (!ie_finite(value)) status[w]=1;
    square+=value*value;
  }
  float beta=sqrt(max(square,0.0f));
  if (!ie_finite(beta)) status[w]=1;
  if (status[w]!=0) { cycle_stop[w]=0; return; }
  cycle_iterations[w]=0;
  // `2` means a preconditioned residual is ready to seed a new cycle. A
  // numerically empty Krylov vector still requires a physical-residual check.
  if (beta<=1e-30f) { cycle_stop[w]=1; return; }
  cycle_stop[w]=0;
  rhs_norms[w]=beta;
  for (int i=0; i<nv; ++i)
    basis[basis_base+i]=(!cycle_stop[w] && active_dof[base+i]!=0)
        ? preconditioned_residual[base+i]/beta : 0.0f;
  for (int i=0; i<m+1; ++i) residual_rhs[gbase+i]=0.0f;
  residual_rhs[gbase]=beta;
  for (int i=0; i<(m+1)*m; ++i) hessenberg[hbase+i]=0.0f;
  for (int i=0; i<m; ++i) {
    givens_cos[cbase+i]=0.0f;
    givens_sin[cbase+i]=0.0f;
  }
}

kernel void effective_gmres_arnoldi(
    device float* basis [[buffer(0)]],
    device const float* preconditioned_dv [[buffer(1)]],
    device const int* active_dof [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    constant float* params [[buffer(4)]],
    device int* status [[buffer(5)]],
    device const int* done [[buffer(6)]],
    device int* cycle_iterations [[buffer(7)]],
    device int* cycle_stop [[buffer(8)]],
    device int* total_iterations [[buffer(9)]],
    device float* hessenberg [[buffer(10)]],
    device float* givens_cos [[buffer(11)]],
    device float* givens_sin [[buffer(12)]],
    device float* residual_rhs [[buffer(13)]],
    device float* work [[buffer(14)]],
    device float* rhs_norms [[buffer(15)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2], m=dims[3], max_iterations=dims[4];
  if (int(world)>=batch) return;
  int w=int(world), j=cycle_iterations[world];
  if (status[w]!=0 || done[w]!=0 || cycle_stop[w]!=0 || j<0 || j>=m
      || total_iterations[w]>=max_iterations) return;
  int base=w*nv, basis_base=w*(m+1)*nv;
  int hbase=w*(m+1)*m, gbase=w*(m+1), cbase=w*m;
  float dt=params[0];
  for (int d=0; d<nv; ++d) {
    float value=active_dof[base+d]!=0
        ? basis[basis_base+j*nv+d]-dt*preconditioned_dv[base+d] : 0.0f;
    if (!ie_finite(value)) status[w]=1;
    work[base+d]=value;
  }
  if (status[w]!=0) { cycle_stop[w]=1; return; }
  float operator_square=0.0f;
  for (int d=0; d<nv; ++d) operator_square+=work[base+d]*work[base+d];
  float operator_norm=sqrt(max(operator_square,0.0f));
  if (!ie_finite(operator_norm)) { status[w]=1; cycle_stop[w]=1; return; }
  // Reorthogonalize once to limit loss of orthogonality in float32. Near an
  // invariant Krylov subspace the remaining vector is roundoff, not a useful
  // new basis vector; detect that relative to ||Ahat*v|| instead of using an
  // absolute 1e-30 cutoff. The physical residual below remains authoritative.
  for (int pass=0; pass<2; ++pass) {
    for (int i=0; i<=j; ++i) {
      float projection=0.0f;
      for (int d=0; d<nv; ++d)
        projection+=basis[basis_base+i*nv+d]*work[base+d];
      hessenberg[hbase+i*m+j]+=projection;
      for (int d=0; d<nv; ++d)
        work[base+d]-=projection*basis[basis_base+i*nv+d];
    }
  }
  float square=0.0f;
  for (int d=0; d<nv; ++d) square+=work[base+d]*work[base+d];
  float beta=sqrt(max(square,0.0f));
  if (!ie_finite(beta)) { status[w]=1; cycle_stop[w]=1; return; }
  hessenberg[hbase+(j+1)*m+j]=beta;
  float breakdown_threshold=8.0f*1.1920928955078125e-7f
      *max(operator_norm,1e-30f);
  bool breakdown=beta<=breakdown_threshold;
  if (!breakdown) {
    for (int d=0; d<nv; ++d)
      basis[basis_base+(j+1)*nv+d]=work[base+d]/beta;
  } else {
    for (int d=0; d<nv; ++d) basis[basis_base+(j+1)*nv+d]=0.0f;
  }
  for (int i=0; i<j; ++i) {
    int at_i=hbase+i*m+j, at_next=hbase+(i+1)*m+j;
    float a=hessenberg[at_i], b=hessenberg[at_next];
    hessenberg[at_i]=givens_cos[cbase+i]*a+givens_sin[cbase+i]*b;
    hessenberg[at_next]=-givens_sin[cbase+i]*a+givens_cos[cbase+i]*b;
  }
  int at_j=hbase+j*m+j, at_next=hbase+(j+1)*m+j;
  float a=hessenberg[at_j], b=hessenberg[at_next];
  float rho=sqrt(a*a+b*b);
  if (!ie_finite(rho)) { status[w]=1; cycle_stop[w]=1; return; }
  float cosine=rho>1e-30f?a/rho:1.0f;
  float sine=rho>1e-30f?b/rho:0.0f;
  givens_cos[cbase+j]=cosine; givens_sin[cbase+j]=sine;
  hessenberg[at_j]=cosine*a+sine*b;
  hessenberg[at_next]=0.0f;
  float gj=residual_rhs[gbase+j], gj1=residual_rhs[gbase+j+1];
  residual_rhs[gbase+j]=cosine*gj+sine*gj1;
  residual_rhs[gbase+j+1]=-sine*gj+cosine*gj1;
  cycle_iterations[w]=j+1;
  total_iterations[w]++;
  float estimate=abs(residual_rhs[gbase+j+1]);
  if (!ie_finite(estimate)) { status[w]=1; cycle_stop[w]=1; return; }
  // Keep a cycle full unless the Arnoldi basis has broken down. The true
  // unpreconditioned residual, not this scaled estimate, decides convergence.
  if (breakdown || total_iterations[w]>=max_iterations || j+1>=m)
    cycle_stop[w]=1;
}

kernel void effective_gmres_finish_cycle(
    device const float* basis [[buffer(0)]],
    device const float* hessenberg [[buffer(1)]],
    device const float* residual_rhs [[buffer(2)]],
    device const int* cycle_iterations [[buffer(3)]],
    device const int* cycle_stop [[buffer(4)]],
    device int* done [[buffer(5)]],
    device int* status [[buffer(6)]],
    constant int* dims [[buffer(7)]],
    device float* coefficients [[buffer(8)]],
    device float* solution [[buffer(9)]],
    device float* residual [[buffer(10)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2], m=dims[3];
  if (int(world)>=batch || status[world]!=0 || done[world]!=0
      || cycle_stop[world]!=1) return;
  int w=int(world), used=clamp(cycle_iterations[w],0,m);
  int basis_base=w*(m+1)*nv, hbase=w*(m+1)*m, gbase=w*(m+1);
  int cbase=w*m, xbase=w*nv;
  for (int i=0; i<m; ++i) coefficients[cbase+i]=0.0f;
  if (used==0) return;
  for (int ii=used-1; ii>=0; --ii) {
    float value=residual_rhs[gbase+ii];
    for (int j=ii+1; j<used; ++j)
      value-=hessenberg[hbase+ii*m+j]*coefficients[cbase+j];
    float diagonal=hessenberg[hbase+ii*m+ii];
    if (!ie_finite(value) || !ie_finite(diagonal) || abs(diagonal)<=1e-30f) {
      status[w]=1;
      done[w]=1;
      residual[w]=as_type<float>(0x7fc00000u);
      for (int d=0; d<nv; ++d) solution[xbase+d]=0.0f;
      return;
    }
    coefficients[cbase+ii]=value/diagonal;
    if (!ie_finite(coefficients[cbase+ii])) {
      status[w]=1;
      done[w]=1;
      residual[w]=as_type<float>(0x7fc00000u);
      for (int d=0; d<nv; ++d) solution[xbase+d]=0.0f;
      return;
    }
  }
  for (int i=0; i<used; ++i)
    for (int d=0; d<nv; ++d)
      solution[xbase+d]+=basis[basis_base+i*nv+d]*coefficients[cbase+i];
  for (int d=0; d<nv; ++d) {
    if (!ie_finite(solution[xbase+d])) {
      status[w]=1;
      done[w]=1;
      residual[w]=as_type<float>(0x7fc00000u);
      for (int k=0; k<nv; ++k) solution[xbase+k]=0.0f;
      return;
    }
  }
}

kernel void effective_gmres_true_residual(
    device const float* rhs [[buffer(0)]],
    device const float* mass_product [[buffer(1)]],
    device const float* derivative_product [[buffer(2)]],
    device const int* active_dof [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    constant float* params [[buffer(5)]],
    device int* status [[buffer(6)]],
    device int* done [[buffer(7)]],
    device const int* iterations [[buffer(8)]],
    device float* residual_vector [[buffer(9)]],
    device float* residual [[buffer(10)]],
    device float* rhs_norms [[buffer(11)]],
    device float* solution [[buffer(12)]],
    device const int* cycle_iterations [[buffer(13)]],
    device int* cycle_stop [[buffer(14)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2], max_iterations=dims[4];
  if (int(world)>=batch) return;
  int w=int(world), base=w*nv;
  if (cycle_stop[w]!=1 && status[w]==0) return;
  if (status[w]!=0) {
    done[w]=1;
    cycle_stop[w]=0;
    for (int d=0; d<nv; ++d) solution[base+d]=0.0f;
    return;
  }
  float residual_square=0.0f, rhs_square=0.0f;
  for (int d=0; d<nv; ++d) {
    if (active_dof[base+d]==0) {
      residual_vector[base+d]=0.0f;
      solution[base+d]=0.0f;
      continue;
    }
    float b=rhs[base+d], mx=mass_product[base+d];
    float dx=derivative_product[base+d];
    float value=b-(mx-params[0]*dx);
    if (!ie_finite(b) || !ie_finite(mx) || !ie_finite(dx) || !ie_finite(value)) {
      status[w]=1;
      residual[w]=as_type<float>(0x7fc00000u);
      done[w]=1;
      for (int k=0; k<nv; ++k) solution[base+k]=0.0f;
      return;
    }
    residual_vector[base+d]=value;
    residual_square+=value*value;
    rhs_square+=b*b;
  }
  float norm=sqrt(max(residual_square,0.0f));
  float rhs_norm=sqrt(max(rhs_square,0.0f));
  residual[w]=norm;
  rhs_norms[w]=rhs_norm;
  if (!ie_finite(norm) || !ie_finite(rhs_norm)) {
    status[w]=1;
    done[w]=1;
    cycle_stop[w]=0;
  } else if (norm<=params[2]+params[1]*rhs_norm) {
    done[w]=1;
    cycle_stop[w]=0;
  } else if (iterations[w]>=max_iterations || cycle_iterations[w]==0) {
    status[w]=2;
    done[w]=1;
    cycle_stop[w]=0;
  } else {
    done[w]=0;
    cycle_stop[w]=2;
  }
  if (status[w]!=0)
    for (int d=0; d<nv; ++d) solution[base+d]=0.0f;
}
