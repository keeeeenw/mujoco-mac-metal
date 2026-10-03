// Copyright 2026 The MuJoCo Metal contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

// Inspect the exponent bits because shader compilation may enable fast-math.
inline bool finite_float(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

kernel void dense_spd_solve(
    device const float* mass [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* factor [[buffer(2)]],
    device float* solution [[buffer(3)]],
    device int* status [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    uint world [[thread_position_in_grid]]) {
  uint nv = uint(dims[0]);
  uint batch = uint(dims[1]);
  uint nrhs = uint(dims[2]);
  if (world >= batch) return;

  uint mass_base = world * nv * nv;
  uint rhs_base = world * nv * nrhs;
  uint failed = 0;

  // Zero first so every validation or factorization failure is safe to consume.
  for (uint i = 0; i < nv * nrhs; ++i) solution[rhs_base + i] = 0.0f;

  // Validate both complete inputs before starting factorization.
  float scale = 0.0f;
  for (uint i = 0; i < nv * nv; ++i) {
    float value = mass[mass_base + i];
    if (!finite_float(value)) failed = 1;
    else scale = max(scale, abs(value));
  }
  for (uint i = 0; i < nv * nrhs; ++i) {
    if (!finite_float(rhs[rhs_base + i])) failed = 1;
  }
  if (failed != 0) {
    status[world] = 1;
    return;
  }

  // Accept only roundoff-sized asymmetry, then factor the symmetric mean.
  float symmetry_tolerance = scale * 0.000003814697265625f;
  for (uint row = 0; row < nv; ++row) {
    for (uint col = row + 1; col < nv; ++col) {
      float upper = mass[mass_base + row * nv + col];
      float lower = mass[mass_base + col * nv + row];
      if (abs(upper - lower) > symmetry_tolerance) {
        failed = 2;
      }
    }
  }
  if (failed != 0) {
    status[world] = int(failed);
    return;
  }

  for (uint row = 0; row < nv; ++row) {
    for (uint col = 0; col < nv; ++col) {
      float value = mass[mass_base + row * nv + col];
      if (row != col) {
        value = 0.5f * value + 0.5f * mass[mass_base + col * nv + row];
      }
      factor[mass_base + row * nv + col] = value;
    }
  }

  // In-place lower-triangular Cholesky. A non-positive pivot is reported;
  // no diagonal shift, damping, or retry is applied.
  for (uint pivot = 0; pivot < nv; ++pivot) {
    float diagonal = factor[mass_base + pivot * nv + pivot];
    for (uint k = 0; k < pivot; ++k) {
      float value = factor[mass_base + pivot * nv + k];
      diagonal -= value * value;
    }
    if (!finite_float(diagonal)) {
      failed = 4;
      break;
    }
    if (!(diagonal > 0.0f)) {
      failed = 3;
      break;
    }
    float root = sqrt(diagonal);
    if (!finite_float(root) || !(root > 0.0f)) {
      failed = 4;
      break;
    }
    factor[mass_base + pivot * nv + pivot] = root;

    for (uint row = pivot + 1; row < nv; ++row) {
      float value = factor[mass_base + row * nv + pivot];
      for (uint k = 0; k < pivot; ++k) {
        value -= factor[mass_base + row * nv + k] *
                 factor[mass_base + pivot * nv + k];
      }
      value /= root;
      if (!finite_float(value)) {
        failed = 4;
        break;
      }
      factor[mass_base + row * nv + pivot] = value;
    }
    if (failed != 0) break;
  }

  // Forward and backward substitution for each RHS column.
  if (failed == 0) {
    for (uint col = 0; col < nrhs; ++col) {
      for (uint row = 0; row < nv; ++row) {
        float value = rhs[rhs_base + row * nrhs + col];
        for (uint k = 0; k < row; ++k) {
          value -= factor[mass_base + row * nv + k] *
                   solution[rhs_base + k * nrhs + col];
        }
        value /= factor[mass_base + row * nv + row];
        if (!finite_float(value)) {
          failed = 4;
          break;
        }
        solution[rhs_base + row * nrhs + col] = value;
      }
      if (failed != 0) break;

      for (int row = int(nv) - 1; row >= 0; --row) {
        float value = solution[rhs_base + uint(row) * nrhs + col];
        for (uint k = uint(row) + 1; k < nv; ++k) {
          value -= factor[mass_base + k * nv + uint(row)] *
                   solution[rhs_base + k * nrhs + col];
        }
        value /= factor[mass_base + uint(row) * nv + uint(row)];
        if (!finite_float(value)) {
          failed = 4;
          break;
        }
        solution[rhs_base + uint(row) * nrhs + col] = value;
      }
      if (failed != 0) break;
    }
  }

  if (failed != 0) {
    for (uint i = 0; i < nv * nrhs; ++i) solution[rhs_base + i] = 0.0f;
  }
  status[world] = int(failed);
}

kernel void dense_general_solve(
    device const float* matrix [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* factor [[buffer(2)]],
    device float* solution [[buffer(3)]],
    device int* status [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    uint world [[thread_position_in_grid]]) {
  uint nv = uint(dims[0]);
  uint batch = uint(dims[1]);
  uint nrhs = uint(dims[2]);
  if (world >= batch) return;

  uint mat_base = world * nv * nv;
  uint rhs_base = world * nv * nrhs;
  uint failed = 0;

  for (uint i = 0; i < nv * nrhs; ++i) solution[rhs_base + i] = 0.0f;

  for (uint i = 0; i < nv * nv; ++i) {
    if (!finite_float(matrix[mat_base + i])) { failed = 1; break; }
  }
  for (uint i = 0; i < nv * nrhs; ++i) {
    if (!finite_float(rhs[rhs_base + i])) { failed = 1; break; }
  }
  if (failed != 0) {
    status[world] = 1;
    return;
  }

  for (uint i = 0; i < nv * nv; ++i) {
    factor[mat_base + i] = matrix[mat_base + i];
  }
  for (uint i = 0; i < nv * nrhs; ++i) {
    solution[rhs_base + i] = rhs[rhs_base + i];
  }

  // LU decomposition with partial row pivoting
  for (uint k = 0; k < nv; ++k) {
    uint max_row = k;
    float max_val = abs(factor[mat_base + k * nv + k]);
    for (uint i = k + 1; i < nv; ++i) {
      float val = abs(factor[mat_base + i * nv + k]);
      if (val > max_val) {
        max_val = val;
        max_row = i;
      }
    }
    if (!(max_val > 1e-15f) || !finite_float(max_val)) {
      failed = 3;
      break;
    }

    if (max_row != k) {
      for (uint j = 0; j < nv; ++j) {
        float tmp = factor[mat_base + k * nv + j];
        factor[mat_base + k * nv + j] = factor[mat_base + max_row * nv + j];
        factor[mat_base + max_row * nv + j] = tmp;
      }
      for (uint c = 0; c < nrhs; ++c) {
        float tmp = solution[rhs_base + k * nrhs + c];
        solution[rhs_base + k * nrhs + c] = solution[rhs_base + max_row * nrhs + c];
        solution[rhs_base + max_row * nrhs + c] = tmp;
      }
    }

    float pivot = factor[mat_base + k * nv + k];
    for (uint i = k + 1; i < nv; ++i) {
      float factor_ik = factor[mat_base + i * nv + k] / pivot;
      if (!finite_float(factor_ik)) { failed = 4; break; }
      factor[mat_base + i * nv + k] = factor_ik;
      for (uint j = k + 1; j < nv; ++j) {
        factor[mat_base + i * nv + j] -= factor_ik * factor[mat_base + k * nv + j];
      }
      for (uint c = 0; c < nrhs; ++c) {
        solution[rhs_base + i * nrhs + c] -= factor_ik * solution[rhs_base + k * nrhs + c];
      }
    }
    if (failed != 0) break;
  }

  if (failed == 0) {
    for (uint c = 0; c < nrhs; ++c) {
      for (int i = int(nv) - 1; i >= 0; --i) {
        float sum = solution[rhs_base + uint(i) * nrhs + c];
        for (uint j = uint(i) + 1; j < nv; ++j) {
          sum -= factor[mat_base + uint(i) * nv + j] * solution[rhs_base + j * nrhs + c];
        }
        float diag = factor[mat_base + uint(i) * nv + uint(i)];
        if (abs(diag) < 1e-15f || !finite_float(diag)) {
          failed = 3;
          break;
        }
        float val = sum / diag;
        if (!finite_float(val)) {
          failed = 4;
          break;
        }
        solution[rhs_base + uint(i) * nrhs + c] = val;
      }
      if (failed != 0) break;
    }
  }

  if (failed != 0) {
    for (uint i = 0; i < nv * nrhs; ++i) solution[rhs_base + i] = 0.0f;
  }
  status[world] = int(failed);
}

// Sleep-filtered dense adapter. Matrix factorization runs only on the packed
// awake principal DOFs; the public result retains full [B,nv,nrhs] shape.
// Inactive single-RHS outputs may preserve a caller-owned acceleration vector.
// Pinned mj_factorI forms L' D L in descending DOF order. In particular,
// Euler's signed damping may yield finite negative D entries even though
// the physical inertia is SPD. No positive-pivot cutoff belongs in this path.
inline uint reverse_ldl_apply(device float* mat, uint stride, uint count,
                              device float* rhs, uint nrhs) {
  for (int k=int(count)-1; k>=0; --k) {
    float diagonal=mat[uint(k)*stride+uint(k)];
    if (!finite_float(diagonal)) return 4;
    if (diagonal==0.0f) return 3;
    for (int i=k-1; i>=0; --i) {
      float multiplier=mat[uint(k)*stride+uint(i)]/diagonal;
      if (!finite_float(multiplier)) return 4;
      for (int j=0; j<=i; ++j) {
        float value=mat[uint(i)*stride+uint(j)]
            -multiplier*mat[uint(k)*stride+uint(j)];
        if (!finite_float(value)) return 4;
        mat[uint(i)*stride+uint(j)]=value;
      }
    }
    for (int j=0; j<k; ++j) {
      float value=mat[uint(k)*stride+uint(j)]/diagonal;
      if (!finite_float(value)) return 4;
      mat[uint(k)*stride+uint(j)]=value;
    }
  }
  for (uint col=0; col<nrhs; ++col) {
    // inv(L'): descending scatter preserves pinned solve ordering.
    for (int i=int(count)-1; i>=0; --i) {
      float value=rhs[uint(i)*nrhs+col];
      for (int j=0; j<i; ++j) {
        float updated=rhs[uint(j)*nrhs+col]-mat[uint(i)*stride+uint(j)]*value;
        if (!finite_float(updated)) return 4;
        rhs[uint(j)*nrhs+col]=updated;
      }
    }
    for (uint i=0; i<count; ++i) {
      float value=rhs[i*nrhs+col]/mat[i*stride+i];
      if (!finite_float(value)) return 4;
      rhs[i*nrhs+col]=value;
    }
    for (uint i=0; i<count; ++i) {
      float value=rhs[i*nrhs+col];
      for (uint j=0; j<i; ++j) value-=mat[i*stride+j]*rhs[j*nrhs+col];
      if (!finite_float(value)) return 4;
      rhs[i*nrhs+col]=value;
    }
  }
  return 0;
}

kernel void dense_awake_solve(
    device const float* mass [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device const int* dof_ids [[buffer(2)]],
    device const int* awake_counts [[buffer(3)]],
    device const float* retained [[buffer(4)]],
    device float* factor [[buffer(5)]],
    device float* work [[buffer(6)]],
    device float* solution [[buffer(7)]],
    device int* status [[buffer(8)]],
    constant int* dims [[buffer(9)]],
    constant int* flags [[buffer(10)]],
    uint world [[thread_position_in_grid]]) {
  uint nv=uint(dims[0]), batch=uint(dims[1]), nrhs=uint(dims[2]);
  if (world>=batch) return;
  uint count=uint(awake_counts[world*3+2]);
  uint mat_base=world*nv*nv, vec_base=world*nv*nrhs;
  bool general=flags[0]==1, use_ldl=flags[0]==2, use_retained=flags[1]!=0;
  uint dof_stride=max(nv,1u), dof_base=world*dof_stride;
  uint failed=0;
  for (uint i=0;i<nv*nrhs;++i) {
    solution[vec_base+i]=0.0f;
    if (use_retained && nrhs==1) solution[vec_base+i]=retained[world*nv+i];
    work[vec_base+i]=0.0f;
  }
  status[world]=0;
  if (count>nv) { status[world]=1; return; }
  // Validate only entries in the active principal system. A quiet sleeping
  // tree may retain old nonfinite scratch without poisoning an awake solve.
  for (uint row=0;row<count;++row) {
    int i=dof_ids[dof_base+row];
    if (i<0 || i>=int(nv)) { failed=1; break; }
    for (uint col=0;col<count;++col) {
      int j=dof_ids[dof_base+col];
      if (j<0 || j>=int(nv)) { failed=1; break; }
      float value=mass[mat_base+uint(i)*nv+uint(j)];
      if (!finite_float(value)) { failed=1; break; }
      factor[mat_base+row*nv+col]=value;
    }
    if (failed) break;
    for (uint col=0;col<nrhs;++col) {
      float value=rhs[vec_base+uint(i)*nrhs+col];
      if (!finite_float(value)) { failed=1; break; }
      work[vec_base+row*nrhs+col]=value;
    }
    if (failed) break;
  }
  if (failed) {
    for (uint i=0;i<nv*nrhs;++i) solution[vec_base+i]=0.0f;
    status[world]=1;
    return;
  }

  if (!general) {
    float scale=0.0f;
    for (uint row=0;row<count;++row)
      for (uint col=0;col<count;++col)
        scale=max(scale,abs(factor[mat_base+row*nv+col]));
    float tol=scale*0.000003814697265625f;
    for (uint row=0;row<count;++row) {
      for (uint col=row+1;col<count;++col) {
        float upper=factor[mat_base+row*nv+col];
        float lower=factor[mat_base+col*nv+row];
        if (abs(upper-lower)>tol) failed=2;
        float mean=0.5f*upper+0.5f*lower;
        factor[mat_base+row*nv+col]=mean;
        factor[mat_base+col*nv+row]=mean;
      }
    }
    if (failed) {
      for (uint i=0;i<nv*nrhs;++i) solution[vec_base+i]=0.0f;
      status[world]=int(failed);
      return;
    }
    if (use_ldl) {
      failed=reverse_ldl_apply(factor+mat_base,nv,count,work+vec_base,nrhs);
    } else {
      for (uint pivot=0;pivot<count;++pivot) {
        float diagonal=factor[mat_base+pivot*nv+pivot];
        for (uint k=0;k<pivot;++k) {
          float value=factor[mat_base+pivot*nv+k];
          diagonal-=value*value;
        }
        if (!finite_float(diagonal)) { failed=4; break; }
        if (!(diagonal>0.0f)) { failed=3; break; }
        float root=sqrt(diagonal);
        if (!finite_float(root) || !(root>0.0f)) { failed=4; break; }
        factor[mat_base+pivot*nv+pivot]=root;
        for (uint row=pivot+1;row<count;++row) {
          float value=factor[mat_base+row*nv+pivot];
          for (uint k=0;k<pivot;++k)
            value-=factor[mat_base+row*nv+k]*factor[mat_base+pivot*nv+k];
          value/=root;
          if (!finite_float(value)) { failed=4; break; }
          factor[mat_base+row*nv+pivot]=value;
        }
        if (failed) break;
      }
      if (!failed) {
        for (uint col=0;col<nrhs;++col) {
          for (uint row=0;row<count;++row) {
            float value=work[vec_base+row*nrhs+col];
            for (uint k=0;k<row;++k)
              value-=factor[mat_base+row*nv+k]*work[vec_base+k*nrhs+col];
            value/=factor[mat_base+row*nv+row];
            if (!finite_float(value)) { failed=4; break; }
            work[vec_base+row*nrhs+col]=value;
          }
          if (failed) break;
          for (int row=int(count)-1;row>=0;--row) {
            float value=work[vec_base+uint(row)*nrhs+col];
            for (uint k=uint(row)+1;k<count;++k)
              value-=factor[mat_base+k*nv+uint(row)]*work[vec_base+k*nrhs+col];
            value/=factor[mat_base+uint(row)*nv+uint(row)];
            if (!finite_float(value)) { failed=4; break; }
            work[vec_base+uint(row)*nrhs+col]=value;
          }
          if (failed) break;
        }
      }
    }
  } else {
    for (uint k=0;k<count;++k) {
      uint pivot=k;
      float magnitude=abs(factor[mat_base+k*nv+k]);
      for (uint row=k+1;row<count;++row) {
        float value=abs(factor[mat_base+row*nv+k]);
        if (value>magnitude) { magnitude=value; pivot=row; }
      }
      if (!finite_float(magnitude)) { failed=4; break; }
      if (!(magnitude>1e-15f)) { failed=3; break; }
      if (pivot!=k) {
        for (uint col=0;col<count;++col) {
          float value=factor[mat_base+k*nv+col];
          factor[mat_base+k*nv+col]=factor[mat_base+pivot*nv+col];
          factor[mat_base+pivot*nv+col]=value;
        }
        for (uint col=0;col<nrhs;++col) {
          float value=work[vec_base+k*nrhs+col];
          work[vec_base+k*nrhs+col]=work[vec_base+pivot*nrhs+col];
          work[vec_base+pivot*nrhs+col]=value;
        }
      }
      float diagonal=factor[mat_base+k*nv+k];
      for (uint row=k+1;row<count;++row) {
        float multiplier=factor[mat_base+row*nv+k]/diagonal;
        if (!finite_float(multiplier)) { failed=4; break; }
        factor[mat_base+row*nv+k]=multiplier;
        for (uint col=k+1;col<count;++col) {
          float value=factor[mat_base+row*nv+col]
              -multiplier*factor[mat_base+k*nv+col];
          if (!finite_float(value)) { failed=4; break; }
          factor[mat_base+row*nv+col]=value;
        }
        for (uint col=0;col<nrhs;++col) {
          float value=work[vec_base+row*nrhs+col]
              -multiplier*work[vec_base+k*nrhs+col];
          if (!finite_float(value)) { failed=4; break; }
          work[vec_base+row*nrhs+col]=value;
        }
        if (failed) break;
      }
      if (failed) break;
    }
    if (!failed) {
      for (uint col=0;col<nrhs;++col) {
        for (int row=int(count)-1;row>=0;--row) {
          float value=work[vec_base+uint(row)*nrhs+col];
          for (uint k=uint(row)+1;k<count;++k)
            value-=factor[mat_base+uint(row)*nv+k]*work[vec_base+k*nrhs+col];
          float diagonal=factor[mat_base+uint(row)*nv+uint(row)];
          if (!finite_float(diagonal) || abs(diagonal)<1e-15f) {
            failed=3; break;
          }
          value/=diagonal;
          if (!finite_float(value)) { failed=4; break; }
          work[vec_base+uint(row)*nrhs+col]=value;
        }
        if (failed) break;
      }
    }
  }
  if (failed) {
    for (uint i=0;i<nv*nrhs;++i) solution[vec_base+i]=0.0f;
    status[world]=int(failed);
    return;
  }
  for (uint row=0;row<count;++row) {
    uint dof=uint(dof_ids[dof_base+row]);
    for (uint col=0;col<nrhs;++col)
      solution[vec_base+dof*nrhs+col]=work[vec_base+row*nrhs+col];
  }
  status[world]=0;
}
