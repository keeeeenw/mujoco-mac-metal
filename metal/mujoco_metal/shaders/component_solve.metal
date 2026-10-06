// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.

#include <metal_stdlib>
using namespace metal;

kernel void build_dof_awake_mask(
    device const int* dof_ids [[buffer(0)]],
    device const int* awake_counts [[buffer(1)]],
    device int* dof_mask [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    device const int* world_mask [[buffer(4)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2];
  if (int(world) >= batch || world_mask[world] == 0) return;
  int stride=max(nv,1), base=int(world)*stride;
  for (int i=0; i<stride; ++i) dof_mask[base+i]=0;
  int count=clamp(awake_counts[world*3+2],0,nv);
  for (int k=0; k<count; ++k) {
    int dof=dof_ids[base+k];
    if (dof >= 0 && dof < nv) dof_mask[base+dof]=1;
  }
}

kernel void stage_component_rhs(
    device const float* source [[buffer(0)]],
    device float* target [[buffer(1)]],
    device const int* world_mask [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    constant int* rhs_index [[buffer(4)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], capacity=dims[1], batch=dims[2], rhs=rhs_index[0];
  if (int(world) >= batch || world_mask[world] == 0) return;
  int rhs_base=int(world)*capacity*nv;
  int source_base=int(world)*nv;
  for (int r=0; r<capacity; ++r) {
    for (int i=0; i<nv; ++i)
      target[rhs_base+r*nv+i] = r == rhs ? source[source_base+i] : 0.0f;
  }
}

kernel void publish_component_solution(
    device const float* source [[buffer(0)]],
    device const float* source_low [[buffer(1)]],
    device float* target [[buffer(2)]],
    device float* target_low [[buffer(3)]],
    device const int* world_mask [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    constant int* rhs_index [[buffer(6)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], capacity=dims[1], batch=dims[2], rhs=rhs_index[0];
  if (int(world) >= batch || world_mask[world] == 0) return;
  int source_base=int(world)*capacity*nv+rhs*nv;
  int target_base=int(world)*nv;
  for (int i=0; i<nv; ++i) {
    target[target_base+i]=source[source_base+i];
    target_low[target_base+i]=source_low[source_base+i];
  }
}

inline bool component_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

// A compact two-word expansion is used only while forming the residual of
// the rounded component solve.  In particular, fma(a, x, -a*x) preserves
// the multiplication error before a long row sum can cancel it.
struct ComponentFloatPair {
  float hi;
  float lo;
};

inline ComponentFloatPair component_pair_renormalize(float hi, float lo) {
  float sum=hi+lo;
  float virtual_lo=sum-hi;
  float error=(hi-(sum-virtual_lo))+(lo-virtual_lo);
  return {sum,error};
}

inline ComponentFloatPair component_pair_add(ComponentFloatPair a,
                                               ComponentFloatPair b) {
  float sum=a.hi+b.hi;
  float virtual_b=sum-a.hi;
  float error=(a.hi-(sum-virtual_b))+(b.hi-virtual_b);
  error += a.lo+b.lo;
  return component_pair_renormalize(sum,error);
}

inline ComponentFloatPair component_pair_product(float a, float b) {
  float product=a*b;
  float error=fma(a,b,-product);
  return {product,error};
}

inline ComponentFloatPair component_pair_scale(ComponentFloatPair a,
                                                float scale) {
  ComponentFloatPair high=component_pair_product(a.hi,scale);
  ComponentFloatPair low=component_pair_product(a.lo,scale);
  return component_pair_add(high,low);
}

inline void clear_component_outputs(
    device float* output, device float* output_low, int output_base,
    int nv, int rhs_capacity,
    device const int* component_layout, int dof_base, int begin, int end) {
  for (int r=0; r<rhs_capacity; ++r) {
    for (int i=begin; i<end; ++i) {
      int dof=component_layout[dof_base+i];
      output[output_base+r*nv+dof]=0.0f;
      output_low[output_base+r*nv+dof]=0.0f;
    }
  }
}

kernel void factor_apply_components(
    device const float* mass_blocks [[buffer(0)]],
    device const float* tendon_armature_blocks [[buffer(1)]],
    device const float* rhs [[buffer(2)]],
    device const int* dof_mask [[buffer(3)]],
    device const int* component_layout [[buffer(4)]],
    device float* factor [[buffer(5)]],
    device float* output [[buffer(6)]],
    device int* status [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    device const float* diagonal_add [[buffer(9)]],
    constant int* allow_indefinite [[buffer(10)]],
    constant int* phase [[buffer(11)]],
    device const int* factor_status [[buffer(12)]],
    device float* output_low [[buffer(13)]],
    device const float* rhs_low [[buffer(14)]],
    device const int* world_mask [[buffer(15)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0], ncomponent=dims[1], batch=dims[2];
  int nnz=dims[3], rhs_capacity=dims[4], nrhs=rhs_capacity;
  if (int(tid) >= batch*ncomponent) return;
  int world=int(tid)/ncomponent;
  if (world_mask[world] == 0) return;
  int component=int(tid)%ncomponent;
  // The packed ABI is [component offsets][DOF ids][mass block offsets].
  int dof_base=ncomponent+1;
  int mass_offset_base=dof_base+max(nv,1);
  int begin=component_layout[component];
  int end=component_layout[component+1];
  int width=end-begin;
  int block_offset=component_layout[mass_offset_base+component];
  int dof_stride=max(nv,1);
  int dof_mask_base=world*dof_stride;
  int mass_base=world*nnz+block_offset;
  int rhs_base=world*rhs_capacity*nv;
  int output_base=world*rhs_capacity*nv;
  int factor_base=world*nnz+block_offset;
  bool any_active=false;
  for (int i=0; i<width; ++i) {
    int dof=component_layout[dof_base+begin+i];
    if (dof_mask[dof_mask_base+dof] != 0) any_active=true;
    for (int r=0; r<nrhs; ++r) {
      output[output_base+r*nv+dof]=0.0f;
      output_low[output_base+r*nv+dof]=0.0f;
    }
  }
  int mode=phase[0];  // 0=factor, 1=solve retained factor, 2=factor+solve.
  int component_index=world*ncomponent+component;
  if (mode == 1) {
    status[component_index]=factor_status[component_index];
    if (factor_status[component_index] != 0) {
      clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                             component_layout,dof_base,begin,end);
      return;
    }
  } else {
    status[component_index]=0;
  }
  if (!any_active || width <= 0) return;

  if (mode != 1) {
  // Form the active principal matrix in persistent scratch. Inactive DOFs
  // receive an isolated unit diagonal; their matrix and RHS storage is never
  // read, so NaNs in a retained sleeping row do not poison this solve.
  float scale=0.0f;
  for (int i=0; i<width; ++i) {
    int dof_i=component_layout[dof_base+begin+i];
    bool active_i=dof_mask[dof_mask_base+dof_i] != 0;
    for (int j=0; j<width; ++j) {
      int dof_j=component_layout[dof_base+begin+j];
      bool active_j=dof_mask[dof_mask_base+dof_j] != 0;
      int idx=mass_base+i*width+j;
      float value=0.0f;
      if (active_i && active_j) {
        float mass=mass_blocks[idx];
        float armature=tendon_armature_blocks[idx];
        if (!component_finite(mass) || !component_finite(armature)) {
          status[world*ncomponent+component]=1;
          clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                                 component_layout,dof_base,begin,end);
          return;
        }
        value=mass+armature;
        if (!component_finite(value)) {
          status[world*ncomponent+component]=1;
          clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                                 component_layout,dof_base,begin,end);
          return;
        }
        if (i == j) {
          float diagonal=diagonal_add[world*max(nv,1)+dof_i];
          if (!component_finite(diagonal)) {
            status[world*ncomponent+component]=1;
            clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                                   component_layout,dof_base,begin,end);
            return;
          }
          value+=diagonal;
          if (!component_finite(value)) {
            status[world*ncomponent+component]=1;
            clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                                   component_layout,dof_base,begin,end);
            return;
          }
        }
        scale=max(scale,abs(value));
      } else if (!active_i && i == j) {
        value=1.0f;
      }
      factor[factor_base+i*width+j]=value;
    }
  }

  // Pinned mass blocks are symmetric. Permit only roundoff-sized drift and
  // factor their symmetric mean. The bit-level finite checks above remain
  // effective under Metal fast-math compilation.
  float symmetry_tolerance=scale*0.000003814697265625f;
  for (int i=0; i<width; ++i) {
    int dof_i=component_layout[dof_base+begin+i];
    if (dof_mask[dof_mask_base+dof_i] == 0) continue;
    for (int j=i+1; j<width; ++j) {
      int dof_j=component_layout[dof_base+begin+j];
      if (dof_mask[dof_mask_base+dof_j] == 0) continue;
      int upper_idx=factor_base+i*width+j;
      int lower_idx=factor_base+j*width+i;
      float upper=factor[upper_idx], lower=factor[lower_idx];
      float difference=upper-lower;
      if (!component_finite(difference)
          || abs(difference)>symmetry_tolerance) {
        status[world*ncomponent+component]=1;
        clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                               component_layout,dof_base,begin,end);
        return;
      }
      float mean=0.5f*upper+0.5f*lower;
      if (!component_finite(mean)) {
        status[world*ncomponent+component]=1;
        clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                               component_layout,dof_base,begin,end);
        return;
      }
      factor[upper_idx]=mean;
      factor[lower_idx]=mean;
    }
  }

  }

  // Reject nonfinite active right-hand sides before solve or writes.
  if (mode != 0) for (int r=0; r<nrhs; ++r) {
    int rhs_row=rhs_base+r*nv;
    for (int i=0; i<width; ++i) {
      int dof=component_layout[dof_base+begin+i];
      if (dof_mask[dof_mask_base+dof] != 0
          && !component_finite(rhs[rhs_row+dof])) {
        status[world*ncomponent+component]=1;
        clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                               component_layout,dof_base,begin,end);
        return;
      }
      if (r == nrhs-1 && dof_mask[dof_mask_base+dof] != 0
          && !component_finite(rhs_low[world*max(nv,1)+dof])) {
        status[world*ncomponent+component]=1;
        clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                               component_layout,dof_base,begin,end);
        return;
      }
    }
  }

  bool use_ldlt=allow_indefinite[0] != 0;
  bool failed=false;
  if (mode != 1) {
  if (use_ldlt) {
    // Pinned mj_factorI performs descending L' D L elimination. This accepts
    // valid indefinite blocks such as [[0,1],[1,1]], whose first ascending
    // LDL pivot is zero but whose trailing pivot is one.
    for (int k=width-1; k>=0 && !failed; --k) {
      float diagonal=factor[factor_base+k*width+k];
      if (!component_finite(diagonal) || diagonal==0.0f) {
        failed=true;
        break;
      }
      for (int i=k-1; i>=0 && !failed; --i) {
        float multiplier=factor[factor_base+k*width+i]/diagonal;
        if (!component_finite(multiplier)) { failed=true; break; }
        for (int j=0; j<=i; ++j) {
          float product=multiplier*factor[factor_base+k*width+j];
          float value=factor[factor_base+i*width+j]-product;
          if (!component_finite(product) || !component_finite(value)) {
            failed=true;
            break;
          }
          factor[factor_base+i*width+j]=value;
        }
      }
      for (int j=0; j<k && !failed; ++j) {
        float value=factor[factor_base+k*width+j]/diagonal;
        if (!component_finite(value)) { failed=true; break; }
        factor[factor_base+k*width+j]=value;
      }
    }
  } else {
    // The physical M-only/positive-definite path retains its lower Cholesky.
    for (int k=0; k<width && !failed; ++k) {
      float diagonal=factor[factor_base+k*width+k];
      for (int p=0; p<k; ++p) {
        float value=factor[factor_base+k*width+p];
        float square=value*value;
        diagonal-=square;
        if (!component_finite(square) || !component_finite(diagonal)) {
          failed=true;
          break;
        }
      }
      if (failed) break;
      if (!component_finite(diagonal) || !(diagonal>0.0f)) {
        failed=true;
        break;
      }
      float pivot=sqrt(diagonal);
      if (!component_finite(pivot) || !(pivot>0.0f)) {
        failed=true;
        break;
      }
      factor[factor_base+k*width+k]=pivot;
      for (int i=k+1; i<width && !failed; ++i) {
        float value=factor[factor_base+i*width+k];
        for (int p=0; p<k; ++p) {
          float product=factor[factor_base+i*width+p]
              *factor[factor_base+k*width+p];
          value-=product;
          if (!component_finite(product) || !component_finite(value)) {
            failed=true;
            break;
          }
        }
        if (failed) break;
        value/=pivot;
        if (!component_finite(value)) { failed=true; break; }
        factor[factor_base+i*width+k]=value;
      }
    }
  }
  }
  if (mode == 0) {
    if (failed) {
      status[world*ncomponent+component]=1;
      clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                             component_layout,dof_base,begin,end);
    }
    return;
  }

  for (int r=0; r<nrhs && !failed; ++r) {
    int rhs_row=rhs_base+r*nv;
    int out_row=output_base+r*nv;
    if (use_ldlt) {
      for (int i=0; i<width; ++i) {
        int dof=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof] != 0) {
          float value=rhs[rhs_row+dof];
          if (!component_finite(value)) { failed=true; break; }
          output[out_row+dof]=value;
        }
      }
      if (failed) break;
      // First apply inv(L') in descending order by scattering into earlier
      // global-DOF RHS entries, then D^-1, then L^-1 in ascending order.
      for (int i=width-1; i>=0 && !failed; --i) {
        int dof_i=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=output[out_row+dof_i];
        for (int j=0; j<i; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          float product=factor[factor_base+i*width+j]*value;
          float updated=output[out_row+dof_j]-product;
          if (!component_finite(product) || !component_finite(updated)) {
            failed=true;
            break;
          }
          output[out_row+dof_j]=updated;
        }
      }
      for (int i=0; i<width && !failed; ++i) {
        int dof=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof] == 0) continue;
        float value=output[out_row+dof]/factor[factor_base+i*width+i];
        if (!component_finite(value)) { failed=true; break; }
        output[out_row+dof]=value;
      }
      for (int i=0; i<width && !failed; ++i) {
        int dof_i=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=output[out_row+dof_i];
        for (int j=0; j<i; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          float product=factor[factor_base+i*width+j]*output[out_row+dof_j];
          value-=product;
          if (!component_finite(product) || !component_finite(value)) {
            failed=true;
            break;
          }
        }
        if (!failed) output[out_row+dof_i]=value;
      }
    } else {
      // Cholesky: forward solve L y=b, then backward solve L' x=y.
      for (int i=0; i<width && !failed; ++i) {
        int dof_i=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=rhs[rhs_row+dof_i];
        for (int j=0; j<i; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          float product=factor[factor_base+i*width+j]*output[out_row+dof_j];
          value-=product;
          if (!component_finite(product) || !component_finite(value)) {
            failed=true;
            break;
          }
        }
        if (failed) break;
        value/=factor[factor_base+i*width+i];
        if (!component_finite(value)) { failed=true; break; }
        output[out_row+dof_i]=value;
      }
      for (int ii=width-1; ii>=0 && !failed; --ii) {
        int dof_i=component_layout[dof_base+begin+ii];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=output[out_row+dof_i];
        for (int j=ii+1; j<width; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          float product=factor[factor_base+j*width+ii]*output[out_row+dof_j];
          value-=product;
          if (!component_finite(product) || !component_finite(value)) {
            failed=true;
            break;
          }
        }
        if (failed) break;
        value/=factor[factor_base+ii*width+ii];
        if (!component_finite(value)) { failed=true; break; }
        output[out_row+dof_i]=value;
      }
    }
    // Validate every written active coordinate as a final guard against
    // overflow or NaN propagation from any substitution path.
    for (int i=0; i<width && !failed; ++i) {
      int dof=component_layout[dof_base+begin+i];
      if (dof_mask[dof_mask_base+dof] != 0
          && !component_finite(output[out_row+dof])) failed=true;
    }
  }
  // Refine each rounded high solution once against the original component
  // equation. The correction is retained in a separate plane so callers can
  // propagate the pair without changing the established high-result ABI.
  for (int r=0; r<nrhs && !failed && mode != 0; ++r) {
    int rhs_row=rhs_base+r*nv;
    int out_row=output_base+r*nv;
    for (int i=0; i<width; ++i) {
      int dof_i=component_layout[dof_base+begin+i];
      if (dof_mask[dof_mask_base+dof_i] == 0) {
        output_low[out_row+dof_i]=0.0f;
        continue;
      }
      ComponentFloatPair residual={rhs[rhs_row+dof_i],
          r == nrhs-1 ? rhs_low[world*max(nv,1)+dof_i] : 0.0f};
      for (int j=0; j<width; ++j) {
        int dof_j=component_layout[dof_base+begin+j];
        if (dof_mask[dof_mask_base+dof_j] == 0) continue;
        int ij=mass_base+i*width+j;
        int ji=mass_base+j*width+i;
        ComponentFloatPair a={mass_blocks[ij],0.0f};
        a=component_pair_add(a,{tendon_armature_blocks[ij],0.0f});
        a=component_pair_add(a,{mass_blocks[ji],0.0f});
        a=component_pair_add(a,{tendon_armature_blocks[ji],0.0f});
        a=component_pair_scale(a,0.5f);
        if (i == j)
          a=component_pair_add(a,
              {diagonal_add[world*max(nv,1)+dof_i],0.0f});
        ComponentFloatPair product=component_pair_add(
            component_pair_product(a.hi,output[out_row+dof_j]),
            component_pair_product(a.lo,output[out_row+dof_j]));
        residual=component_pair_add(residual,{-product.hi,-product.lo});
      }
      float residual_value=residual.hi+residual.lo;
      if (!component_finite(residual_value)) { failed=true; break; }
      output_low[out_row+dof_i]=residual_value;
    }
    if (failed) break;
    if (use_ldlt) {
      // Apply inverse(L') by descending scatter, then D^-1 and L^-1.
      for (int i=width-1; i>=0 && !failed; --i) {
        int dof_i=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=output_low[out_row+dof_i];
        for (int j=0; j<i; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          float product=factor[factor_base+i*width+j]*value;
          float updated=output_low[out_row+dof_j]-product;
          if (!component_finite(product) || !component_finite(updated)) {
            failed=true;
            break;
          }
          output_low[out_row+dof_j]=updated;
        }
      }
      for (int i=0; i<width && !failed; ++i) {
        int dof=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof] == 0) continue;
        output_low[out_row+dof] /= factor[factor_base+i*width+i];
        if (!component_finite(output_low[out_row+dof])) failed=true;
      }
      for (int i=0; i<width && !failed; ++i) {
        int dof_i=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=output_low[out_row+dof_i];
        for (int j=0; j<i; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          value -= factor[factor_base+i*width+j]*output_low[out_row+dof_j];
        }
        if (!component_finite(value)) failed=true;
        else output_low[out_row+dof_i]=value;
      }
    } else {
      for (int i=0; i<width && !failed; ++i) {
        int dof_i=component_layout[dof_base+begin+i];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=output_low[out_row+dof_i];
        for (int j=0; j<i; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          value -= factor[factor_base+i*width+j]*output_low[out_row+dof_j];
        }
        value /= factor[factor_base+i*width+i];
        if (!component_finite(value)) failed=true;
        else output_low[out_row+dof_i]=value;
      }
      for (int ii=width-1; ii>=0 && !failed; --ii) {
        int dof_i=component_layout[dof_base+begin+ii];
        if (dof_mask[dof_mask_base+dof_i] == 0) continue;
        float value=output_low[out_row+dof_i];
        for (int j=ii+1; j<width; ++j) {
          int dof_j=component_layout[dof_base+begin+j];
          if (dof_mask[dof_mask_base+dof_j] == 0) continue;
          value -= factor[factor_base+j*width+ii]*output_low[out_row+dof_j];
        }
        value /= factor[factor_base+ii*width+ii];
        if (!component_finite(value)) failed=true;
        else output_low[out_row+dof_i]=value;
      }
    }
  }
  if (failed) {
    status[world*ncomponent+component]=1;
    clear_component_outputs(output,output_low,output_base,nv,rhs_capacity,
                           component_layout,dof_base,begin,end);
  }
}

kernel void merge_component_world_status(
    device const int* component_status [[buffer(0)]],
    device int* world_status [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    device const int* world_mask [[buffer(3)]],
    uint world [[thread_position_in_grid]]) {
  int ncomponent=dims[0], batch=dims[1];
  if (int(world) >= batch || world_mask[world] == 0) return;
  int status=world_status[world];
  for (int component=0; component<ncomponent; ++component)
    status=max(status, component_status[world*ncomponent+component]);
  world_status[world]=status;
}

kernel void apply_component_mass(
    device const float* mass_blocks [[buffer(0)]],
    device const float* tendon_armature_blocks [[buffer(1)]],
    device const float* vector [[buffer(2)]],
    device const int* dof_mask [[buffer(3)]],
    device const int* component_layout [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device const float* diagonal_add [[buffer(7)]],
    device const int* world_mask [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0], ncomponent=dims[1], batch=dims[2], nnz=dims[3];
  if (int(tid) >= batch*ncomponent) return;
  int world=int(tid)/ncomponent;
  if (world_mask[world] == 0) return;
  int component=int(tid)%ncomponent;
  int dof_base=ncomponent+1;
  int mass_offset_base=dof_base+max(nv,1);
  int begin=component_layout[component];
  int end=component_layout[component+1];
  int width=end-begin;
  int block_offset=component_layout[mass_offset_base+component];
  int dof_stride=max(nv,1);
  int vector_base=world*nv;
  int mask_base=world*dof_stride;
  int output_base=world*dof_stride;
  int mass_base=world*nnz+block_offset;
  for (int i=0; i<width; ++i) {
    int dof_i=component_layout[dof_base+begin+i];
    if (dof_mask[mask_base+dof_i] == 0) {
      output[output_base+dof_i]=0.0f;
      continue;
    }
    float sum=0.0f;
    for (int j=0; j<width; ++j) {
      int dof_j=component_layout[dof_base+begin+j];
      if (dof_mask[mask_base+dof_j] == 0) continue;
      int idx=mass_base+i*width+j;
      sum+=(mass_blocks[idx]+tendon_armature_blocks[idx])
          * vector[vector_base+dof_j];
    }
    sum+=diagonal_add[world*dof_stride+dof_i]*vector[vector_base+dof_i];
    output[output_base+dof_i]=sum;
  }
}

kernel void build_euler_damping_diagonal(
    device const int* dof_ids [[buffer(0)]],
    device const int* awake_counts [[buffer(1)]],
    device const int* eligible_dof [[buffer(2)]],
    device const float* q_deriv [[buffer(3)]],
    device float* diagonal_add [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    constant float* timestep [[buffer(6)]],
    device const int* world_mask [[buffer(7)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], batch=dims[2];
  if (int(world) >= batch || world_mask[world] == 0) return;
  int stride=max(nv,1), base=int(world)*stride;
  for (int i=0; i<stride; ++i) diagonal_add[base+i]=0.0f;
  int count=clamp(awake_counts[world*3+2],0,nv);
  bool damping_trigger=false;
  for (int k=0; k<count; ++k) {
    int dof=dof_ids[base+k];
    if (dof >= 0 && dof < nv && eligible_dof[dof] != 0)
      damping_trigger=true;
  }
  if (damping_trigger) {
    for (int k=0; k<count; ++k) {
      int dof=dof_ids[base+k];
      if (dof >= 0 && dof < nv)
        diagonal_add[base+dof]=timestep[0]*q_deriv[world*nv+dof];
    }
  }
}

kernel void apply_component_mass_pair(
    device const float* mass_blocks [[buffer(0)]],
    device const float* tendon_armature_blocks [[buffer(1)]],
    device const float* vector_hi [[buffer(2)]],
    device const float* vector_low [[buffer(3)]],
    device const int* dof_mask [[buffer(4)]],
    device const int* component_layout [[buffer(5)]],
    device float* output_hi [[buffer(6)]],
    device float* output_low [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    device const float* diagonal_add [[buffer(9)]],
    device const int* world_mask [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0], ncomponent=dims[1], batch=dims[2], nnz=dims[3];
  if (int(tid) >= batch*ncomponent) return;
  int world=int(tid)/ncomponent;
  if (world_mask[world] == 0) return;
  int component=int(tid)%ncomponent;
  int dof_base=ncomponent+1;
  int mass_offset_base=dof_base+max(nv,1);
  int begin=component_layout[component];
  int end=component_layout[component+1];
  int width=end-begin;
  int block_offset=component_layout[mass_offset_base+component];
  int stride=max(nv,1), base=world*stride;
  int mass_base=world*nnz+block_offset;
  for (int i=0; i<width; ++i) {
    int dof_i=component_layout[dof_base+begin+i];
    if (dof_mask[base+dof_i] == 0) {
      output_hi[base+dof_i]=0.0f;
      output_low[base+dof_i]=0.0f;
      continue;
    }
    ComponentFloatPair sum={0.0f,0.0f};
    for (int j=0; j<width; ++j) {
      int dof_j=component_layout[dof_base+begin+j];
      if (dof_mask[base+dof_j] == 0) continue;
      int idx=mass_base+i*width+j;
      ComponentFloatPair coefficient=component_pair_add(
          {mass_blocks[idx],0.0f},
          {tendon_armature_blocks[idx],0.0f});
      ComponentFloatPair term=component_pair_scale(
          coefficient,vector_hi[base+dof_j]);
      sum=component_pair_add(sum,term);
      term=component_pair_scale(coefficient,vector_low[base+dof_j]);
      sum=component_pair_add(sum,term);
    }
    ComponentFloatPair term=component_pair_product(
        diagonal_add[base+dof_i],vector_hi[base+dof_i]);
    sum=component_pair_add(sum,term);
    term=component_pair_product(
        diagonal_add[base+dof_i],vector_low[base+dof_i]);
    sum=component_pair_add(sum,term);
    if (!component_finite(sum.hi) || !component_finite(sum.lo)) {
      output_hi[base+dof_i]=as_type<float>(0x7fc00000u);
      output_low[base+dof_i]=as_type<float>(0x7fc00000u);
    } else {
      output_hi[base+dof_i]=sum.hi;
      output_low[base+dof_i]=sum.lo;
    }
  }
}
