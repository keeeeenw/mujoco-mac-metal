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

// Native equality assembly for joint/connect/weld equalities.
// Pinned semantics: MuJoCo 3.10.0 engine/engine_setconst.c (eq_data body/site
// semantics), engine/engine_core_constraint.c (mj_equalityAnchors,
// mj_instantiateEquality, mj_diagApprox, getposdim/getimpedance,
// mj_referenceConstraint, mj_Jdotv translational part plus weld rotational
// term1/term3 with torquescale), doc/XMLreference.rst
// (connect anchor in body1 local, weld anchor in body2 local/relpose).
// Per-step assembly executes on Metal; CPU work is constant preparation only.

#include <metal_stdlib>
using namespace metal;

// Packed canonical-J record helpers. Integer header/CSR metadata is read only
// through an int32 reinterpretation of the existing float workspace binding.
// Header words: magic, version, mode, nr, nv, nnz, rowptr, columns, values,
// world stride in 32-bit words, unsupported-write status, reserved.
constant int CCJ_MAGIC = 0x4d4a4353;
constant int CCJ_VERSION = 1;
constant int CCJ_DENSE = 0;
constant int CCJ_CSR = 1;
constant int CCJ_HEADER_WORDS = 12;

inline device int* ccj_header(device float* record) {
  return reinterpret_cast<device int*>(record);
}
inline device const int* ccj_header(device const float* record) {
  return reinterpret_cast<device const int*>(record);
}
inline bool ccj_header_valid(device const int* h, int nr, int nv) {
  if (h[0] != CCJ_MAGIC || h[1] != CCJ_VERSION || h[3] != nr
      || h[4] != nv || h[5] < 0 || h[9] < CCJ_HEADER_WORDS
      || h[8] < CCJ_HEADER_WORDS || h[8] > h[9]) return false;
  if (h[2] == CCJ_DENSE)
    return h[5] == nr * nv && h[8] + h[5] <= h[9];
  if (h[2] != CCJ_CSR || h[6] < CCJ_HEADER_WORDS
      || h[7] < h[6] + nr + 1 || h[8] < h[7] + h[5]
      || h[8] + h[5] > h[9]) return false;
  return true;
}
inline int ccj_stride_from_dims(constant int* dims) {
  int tail = dims[24] + dims[9] * 10;
  return dims[tail + 17];
}
inline device float* ccj_world(device float* storage, int world, int nr, int nv,
                               constant int* dims) {
  int stride = ccj_stride_from_dims(dims);
  device float* record = storage + world * stride;
  device int* h = ccj_header(record);
  if (stride < CCJ_HEADER_WORDS || !ccj_header_valid(h, nr, nv)
      || h[9] != stride) {
    atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                          memory_order_relaxed);
    h[9] = -1;
  }
  return record;
}
inline device const float* ccj_world(device const float* storage, int world,
                                     int nr, int nv, constant int* dims) {
  int stride = ccj_stride_from_dims(dims);
  return storage + world * stride;
}
inline float ccj_get(device const float* record, int row, int dof, int nv) {
  device const int* h = ccj_header(record);
  if (!ccj_header_valid(h, h[3], nv)) return 0.0f;
  if (row < 0 || row >= h[3] || dof < 0 || dof >= h[4]) return 0.0f;
  if (h[2] == CCJ_DENSE)
    return record[h[8] + row * h[4] + dof];
  if (h[2] != CCJ_CSR) return 0.0f;
  device const int* rowptr = reinterpret_cast<device const int*>(record) + h[6];
  device const int* columns = reinterpret_cast<device const int*>(record) + h[7];
  int lo = rowptr[row], hi = rowptr[row + 1];
  if (lo < 0 || hi < lo || hi > h[5]) return 0.0f;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (columns[mid] < 0 || columns[mid] >= h[4]) return 0.0f;
    if (columns[mid] < dof) lo = mid + 1;
    else hi = mid;
  }
  if (lo >= rowptr[row + 1] || columns[lo] != dof) return 0.0f;
  return record[h[8] + lo];
}
inline void ccj_set(device float* record, int row, int dof, float value, int nv) {
  device int* h = ccj_header(record);
  if (!ccj_header_valid(h, h[3], nv)) {
    atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                          memory_order_relaxed);
    return;
  }
  if (row < 0 || row >= h[3] || dof < 0 || dof >= h[4]) {
    if (value != 0.0f && h[0] == CCJ_MAGIC)
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
    return;
  }
  if (h[2] == CCJ_DENSE) {
    record[h[8] + row * h[4] + dof] = value;
    return;
  }
  if (h[2] != CCJ_CSR) {
    if (value != 0.0f)
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
    return;
  }
  device const int* rowptr = reinterpret_cast<device const int*>(record) + h[6];
  device const int* columns = reinterpret_cast<device const int*>(record) + h[7];
  int lo = rowptr[row], hi = rowptr[row + 1];
  if (lo < 0 || hi < lo || hi > h[5]) {
    atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                          memory_order_relaxed);
    return;
  }
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (columns[mid] < 0 || columns[mid] >= h[4]) {
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
      return;
    }
    if (columns[mid] < dof) lo = mid + 1;
    else hi = mid;
  }
  if (lo >= rowptr[row + 1] || columns[lo] != dof) {
    if (value != 0.0f)
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
    return;
  }
  record[h[8] + lo] = value;
}
inline void ccj_add(device float* record, int row, int dof, float value, int nv) {
  ccj_set(record, row, dof, ccj_get(record, row, dof, nv) + value, nv);
}
inline void ccj_zero_row(device float* record, int row, int nv) {
  device int* h = ccj_header(record);
  if (!ccj_header_valid(h, h[3], nv)) {
    atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                          memory_order_relaxed);
    return;
  }
  if (row < 0 || row >= h[3]) return;
  if (h[2] == CCJ_DENSE) {
    for (int dof = 0; dof < h[4]; ++dof) record[h[8] + row * h[4] + dof] = 0.0f;
  } else if (h[2] == CCJ_CSR) {
    device const int* rowptr = reinterpret_cast<device const int*>(record) + h[6];
    int begin = rowptr[row], end = rowptr[row + 1];
    if (begin < 0 || end < begin || end > h[5]) {
      atomic_store_explicit(reinterpret_cast<device atomic_int*>(&h[10]), 1,
                            memory_order_relaxed);
      return;
    }
    for (int slot = begin; slot < end; ++slot)
      record[h[8] + slot] = 0.0f;
  }
}

inline float3 eq_qrot(float4 q, float3 v) {
  return v + 2.0f * cross(q.yzw, cross(q.yzw, v) + q.x * v);
}
inline float4 eq_qmul(float4 a, float4 b) {
  return float4(a.x * b.x - dot(a.yzw, b.yzw),
      a.x * b.yzw + b.x * a.yzw + cross(a.yzw, b.yzw));
}
inline float4 eq_qneg(float4 q) {
  return float4(q.x, -q.y, -q.z, -q.w);
}
inline float4 eq_qmul_axis(float4 q, float3 axis) {
  return float4(-dot(q.yzw, axis),
      q.x * axis + cross(q.yzw, axis));
}
inline float4 eq_qderiv(float4 q, float3 vel) {
  return float4(
    0.5f * (-vel.x * q.y - vel.y * q.z - vel.z * q.w),
    0.5f * ( vel.x * q.x + vel.y * q.w - vel.z * q.z),
    0.5f * (-vel.x * q.w + vel.y * q.x + vel.z * q.y),
    0.5f * ( vel.x * q.z - vel.y * q.y + vel.z * q.x));
}

struct EqFloatPair {
  float hi;
  float lo;
};

struct EqVec3Pair {
  float3 hi;
  float3 lo;
};

inline EqFloatPair eq_pair_renormalize(float hi, float lo) {
  float sum = hi + lo;
  float virtual_lo = sum - hi;
  float error = (hi - (sum - virtual_lo)) + (lo - virtual_lo);
  return {sum, error};
}

inline EqFloatPair eq_pair_add(EqFloatPair a, EqFloatPair b) {
  float sum = a.hi + b.hi;
  float virtual_b = sum - a.hi;
  float error = (a.hi - (sum - virtual_b)) + (b.hi - virtual_b);
  error += a.lo + b.lo;
  return eq_pair_renormalize(sum, error);
}

inline EqFloatPair eq_pair_neg(EqFloatPair a) {
  return {-a.hi, -a.lo};
}

inline EqFloatPair eq_pair_mul(EqFloatPair a, EqFloatPair b) {
  float product = a.hi * b.hi;
  float error = fma(a.hi, b.hi, -product);
  error = fma(a.hi, b.lo, error);
  error = fma(a.lo, b.hi, error);
  float low = a.lo * b.lo;
  error += low;
  return eq_pair_renormalize(product, error);
}

inline EqFloatPair eq_pair_div(EqFloatPair a, EqFloatPair b) {
  float q0 = a.hi / b.hi;
  EqFloatPair rem = eq_pair_add(a, eq_pair_neg(eq_pair_mul(b, {q0, 0.0f})));
  float q1 = (rem.hi + rem.lo) / b.hi;
  return eq_pair_renormalize(q0, q1);
}

inline EqVec3Pair eq_vec_pair_add(EqVec3Pair a, EqVec3Pair b) {
  EqFloatPair x = eq_pair_add({a.hi.x, a.lo.x}, {b.hi.x, b.lo.x});
  EqFloatPair y = eq_pair_add({a.hi.y, a.lo.y}, {b.hi.y, b.lo.y});
  EqFloatPair z = eq_pair_add({a.hi.z, a.lo.z}, {b.hi.z, b.lo.z});
  return {float3(x.hi, y.hi, z.hi), float3(x.lo, y.lo, z.lo)};
}

inline EqVec3Pair eq_vec_pair_sub(EqVec3Pair a, EqVec3Pair b) {
  return eq_vec_pair_add(a, {-b.hi, -b.lo});
}

inline EqVec3Pair eq_vec_pair_scale(EqVec3Pair a, float scale) {
  EqFloatPair x = eq_pair_mul({a.hi.x, a.lo.x}, {scale, 0.0f});
  EqFloatPair y = eq_pair_mul({a.hi.y, a.lo.y}, {scale, 0.0f});
  EqFloatPair z = eq_pair_mul({a.hi.z, a.lo.z}, {scale, 0.0f});
  return {float3(x.hi, y.hi, z.hi), float3(x.lo, y.lo, z.lo)};
}

inline EqVec3Pair eq_cross_vec_pair(float3 a, EqVec3Pair b) {
  EqFloatPair x = eq_pair_add(
      eq_pair_mul({a.y, 0.0f}, {b.hi.z, b.lo.z}),
      eq_pair_neg(eq_pair_mul({a.z, 0.0f}, {b.hi.y, b.lo.y})));
  EqFloatPair y = eq_pair_add(
      eq_pair_mul({a.z, 0.0f}, {b.hi.x, b.lo.x}),
      eq_pair_neg(eq_pair_mul({a.x, 0.0f}, {b.hi.z, b.lo.z})));
  EqFloatPair z = eq_pair_add(
      eq_pair_mul({a.x, 0.0f}, {b.hi.y, b.lo.y}),
      eq_pair_neg(eq_pair_mul({a.y, 0.0f}, {b.hi.x, b.lo.x})));
  return {float3(x.hi, y.hi, z.hi), float3(x.lo, y.lo, z.lo)};
}

// Match the source body-anchor transform while retaining float32 product and
// subtraction residuals.  These low words feed equality aref directly; they
// are not reconstructed from an already rounded cpos.
inline EqVec3Pair eq_qrot_pair(float4 q, float3 v) {
  EqVec3Pair input = {v, float3(0.0f)};
  EqVec3Pair inner = eq_vec_pair_add(
      eq_cross_vec_pair(q.yzw, input),
      eq_vec_pair_scale(input, q.x));
  EqVec3Pair outer = eq_cross_vec_pair(q.yzw, inner);
  return eq_vec_pair_add(input, eq_vec_pair_scale(outer, 2.0f));
}

inline EqVec3Pair eq_vec_pair_from_float(float3 value) {
  return {value, float3(0.0f)};
}

inline EqVec3Pair eq_qrot_translate_pair(float4 q, float3 local,
                                         float3 translation) {
  return eq_vec_pair_add(eq_qrot_pair(q, local),
                         eq_vec_pair_from_float(translation));
}

inline EqFloatPair eq_vec_pair_component(EqVec3Pair value, int axis) {
  return axis == 0 ? EqFloatPair{value.hi.x, value.lo.x}
      : (axis == 1 ? EqFloatPair{value.hi.y, value.lo.y}
                   : EqFloatPair{value.hi.z, value.lo.z});
}

inline EqFloatPair eq_sol_param(device const float* eq_sol, int e, int k) {
  return {eq_sol[e * 14 + k], eq_sol[e * 14 + 7 + k]};
}

inline EqFloatPair eq_impedance_pair(device const float* eq_sol, int e,
                                     float pos, float margin) {
  EqFloatPair d0 = eq_sol_param(eq_sol, e, 2);
  EqFloatPair d1 = eq_sol_param(eq_sol, e, 3);
  float width = eq_sol[e * 14 + 4];
  float mid = eq_sol[e * 14 + 5];
  float power = eq_sol[e * 14 + 6];
  if (d0.hi == d1.hi || width <= 1e-15f)
    return eq_pair_mul(eq_pair_add(d0, d1), {0.5f, 0.0f});
  float x = abs((pos - margin) / width);
  if (x >= 1.0f) return d1;
  if (x <= 0.0f) return d0;
  float y = power == 1.0f ? x : (x <= mid
      ? pow(x, power) / pow(mid, power - 1.0f)
      : 1.0f - pow(1.0f - x, power)
          / pow(1.0f - mid, power - 1.0f));
  return eq_pair_add(d0, eq_pair_mul({y, 0.0f}, eq_pair_add(d1, eq_pair_neg(d0))));
}

inline float eq_impedance(device const float* eq_sol, int e,
                          float pos, float margin) {
  float d0 = eq_sol[e * 14 + 2];
  float d1 = eq_sol[e * 14 + 3];
  float width = eq_sol[e * 14 + 4];
  float mid = eq_sol[e * 14 + 5];
  float power = eq_sol[e * 14 + 6];
  if (d0 == d1 || width <= 1e-15f) return 0.5f * (d0 + d1);
  float x = abs((pos - margin) / width);
  if (x >= 1.0f) return d1;
  if (x <= 0.0f) return d0;
  float y = power == 1.0f ? x : (x <= mid ? pow(x, power) / pow(mid, power - 1.0f)
      : 1.0f - pow(1.0f - x, power) / pow(1.0f - mid, power - 1.0f));
  return d0 + y * (d1 - d0);
}

// Re-evaluate the scalar equality reference with source-double solref/solimp
// constants represented as high/low float words. `pos` and `vel` are also
// expansions, so cancellation in -B*vel - K*imp*pos is retained for the
// acceleration-space primal solver.
inline EqFloatPair eq_aref_pair(device const float* eq_sol, int e,
    EqFloatPair pos, EqFloatPair vel, EqFloatPair impedance,
    float jd, bool refsafe, float timestep) {
  EqFloatPair width = eq_sol_param(eq_sol, e, 3);
  EqFloatPair r0 = eq_sol_param(eq_sol, e, 0);
  EqFloatPair r1 = eq_sol_param(eq_sol, e, 1);
  if (refsafe && r0.hi > 0.0f && r0.hi < 2.0f * timestep)
    r0 = {2.0f * timestep, 0.0f};
  EqFloatPair K, B;
  if (r0.hi > 0.0f) {
    EqFloatPair den = eq_pair_mul(eq_pair_mul(width, width),
        eq_pair_mul(eq_pair_mul(r0, r0), eq_pair_mul(r1, r1)));
    if (den.hi < 1e-15f) den = {1e-15f, 0.0f};
    K = eq_pair_div({1.0f, 0.0f}, den);
  } else {
    EqFloatPair width_sq = eq_pair_mul(width, width);
    if (width_sq.hi < 1e-15f) width_sq = {1e-15f, 0.0f};
    K = eq_pair_div(eq_pair_neg(r0), width_sq);
  }
  if (r1.hi > 0.0f && r0.hi > 0.0f) {
    EqFloatPair den = eq_pair_mul(width, r0);
    if (den.hi < 1e-15f) den = {1e-15f, 0.0f};
    B = eq_pair_div({2.0f, 0.0f}, den);
  } else {
    B = eq_pair_div(eq_pair_neg(r1), width);
  }
  EqFloatPair bvel = eq_pair_mul(B, vel);
  EqFloatPair kip = eq_pair_mul(eq_pair_mul(K, impedance), pos);
  return eq_pair_add(eq_pair_add(eq_pair_neg(bvel), eq_pair_neg(kip)),
                     {-jd, 0.0f});
}

// Point Jacobian via parent traversal. Output blocks are device-owned
// 3*nv slices in the workspace tail so equality assembly has no fixed local
// DOF ceiling.
inline void eq_point_jac(
    float3 point, int body,
    device const float* body_pos, device const float* body_quat,
    device const float* anchors, device const float* axes,
    device const int* body_parentid, device const int* body_jntadr,
    device const int* body_jntnum, device const int* jnt_type,
    device const int* jnt_dofadr,
    int nbody, int njnt, int nv, int bo, int jo,
    device float* Jp, device float* Jr) {
  for (int i = 0; i < 3 * nv; ++i) { Jp[i] = 0.0f; Jr[i] = 0.0f; }
  if (nv == 0) return;
  // Traverse ancestors, accumulating contributions (opposite sign handled by caller).
  // This fills Jp for the single body point (caller differences two bodies).
  int b = body;
  while (b > 0 && b < nbody) {
    int ja = body_jntadr[b];
    int jn = body_jntnum[b];
    for (int jj = 0; jj < jn; ++jj) {
      int j = ja + jj;
      if (j < 0 || j >= njnt) continue;
      int da = jnt_dofadr[j];
      int typ = jnt_type[j];
      int nd = typ == 0 ? 6 : typ == 1 ? 3 : 1;
      for (int q = 0; q < nd; ++q) {
        int dof = da + q;
        if (dof < 0 || dof >= nv) continue;
        float3 col = float3(0.0f);
        float3 ang = float3(0.0f);
        if (typ == 2) {
          col = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
        } else if (typ == 3) {
          float3 axis = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
          float3 anchor = float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]);
          col = cross(axis, point - anchor);
          ang = axis;
        } else if (typ == 0 || typ == 1) {
          if (typ == 0 && q < 3) {
            col = float3(q == 0, q == 1, q == 2);
          } else {
            int qr = typ == 0 ? q - 3 : q;
            float4 bq = float4(body_quat[(bo + b) * 4], body_quat[(bo + b) * 4 + 1],
                               body_quat[(bo + b) * 4 + 2], body_quat[(bo + b) * 4 + 3]);
            // Rotate unit axis by body quat (free/ball angular).
            float3 unit = float3(qr == 0, qr == 1, qr == 2);
            float3 axis = eq_qrot(bq, unit);
            ang = axis;
            float3 pivot;
            if (typ == 0) {
              pivot = float3(body_pos[(bo + b) * 3], body_pos[(bo + b) * 3 + 1], body_pos[(bo + b) * 3 + 2]);
            } else {
              pivot = float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]);
            }
            col = cross(axis, point - pivot);
          }
        }
        Jp[0 * nv + dof] += col.x;
        Jp[1 * nv + dof] += col.y;
        Jp[2 * nv + dof] += col.z;
        Jr[0 * nv + dof] += ang.x;
        Jr[1 * nv + dof] += ang.y;
        Jr[2 * nv + dof] += ang.z;
      }
    }
    b = body_parentid[b];
  }
}

// The coupled stage also uses the canonical Jacobian and equality helpers.
// It excludes only this entry point when building its separate library.
#ifndef MUJOCO_METAL_EQUALITY_HELPERS_ONLY
kernel void equality_assembly(
    device const int* eq_obj [[buffer(0)]],
    device const float* eq_data [[buffer(1)]],
    device const float* eq_sol [[buffer(2)]],
    device const int* eq_active [[buffer(3)]],
    device const int* eq_rowadr [[buffer(4)]],
    device const int* eq_type [[buffer(5)]],
    device const int* eq_objtype [[buffer(6)]],
    device const int* joint_qadr [[buffer(7)]],
    device const float* qpos0 [[buffer(8)]],
    device const int* joint_dadr [[buffer(9)]],
    device const float* dof_invweight [[buffer(10)]],
    device const float* body_invweight [[buffer(11)]],
    device const int* site_bodyid [[buffer(12)]],
    device const int* body_parentid [[buffer(13)]],
    device const int* body_jntadr [[buffer(14)]],
    device const int* body_jntnum [[buffer(15)]],
    device const int* jnt_type [[buffer(16)]],
    device const int* jnt_dofadr [[buffer(17)]],
    constant int* dims [[buffer(18)]],
    constant float* params [[buffer(19)]],
    device const float* qpos [[buffer(20)]],
    device const float* qvel [[buffer(21)]],
    device const float* body_pos [[buffer(22)]],
    device const float* body_quat [[buffer(23)]],
    device const float* joint_anchor [[buffer(24)]],
    device const float* joint_axis [[buffer(25)]],
    device const float* site_pos [[buffer(26)]],
    device const int* eq_rownum [[buffer(27)]],
    device float* workspace_J [[buffer(28)]],
    device float* workspace_debug [[buffer(29)]],
    device const float* site_quat [[buffer(30)]],
    uint world [[thread_position_in_grid]]) {
  int nq = dims[0];
  int nv = dims[1];
  int nj = dims[2];
  int neq = dims[3];
  int batch = dims[5];
  int flags = dims[6];
  bool refsafe = (dims[7] & 1) != 0;
  int nr = dims[9];
  int nbody = dims[11];
  int njnt = dims[12];
  int nsite = dims[13];
  int n_eq_rows = dims[14];
  if (world >= uint(batch)) return;
  device float* dbg = workspace_debug + int(world) * max(dims[21], nr * nr + 7 * nr);
  int solver_metadata = dims[24] + dims[9] * 10;
  device float* solver_scratch = dbg + nr * nr + 7 * nr;
  device float* pair_tail = solver_scratch + dims[solver_metadata + 15];
  // The third row-vector in the paired tail is the immutable low word of
  // aref; the first two are residual-low and force-low solver state.
  device float* aref_low = pair_tail + 8 * max(nv, 1) + 2 * max(nr, 1);
  for (int row = 0; row < nr; ++row) aref_low[row] = 0.0f;
  if (neq == 0 || n_eq_rows == 0) return;

  float timestep = params[0];
  int qb = int(world) * nv;
  int pb = int(world) * nq;
  int bo = int(world) * nbody;
  int jo = int(world) * njnt;
  int so = int(world) * max(nsite, 1);

  device float* J_world = ccj_world(workspace_J, int(world), nr, nv, dims);
  // Equality assembly owns only the equality interval. Candidate contacts
  // may already have written their disjoint CSR rows for this position epoch;
  // preserve those packed values on sparse profiles. Dense compatibility
  // keeps its historical full-matrix clear.
  device const int* J_header = ccj_header(J_world);
  if (J_header[2] == CCJ_DENSE) {
    for (int row = 0; row < nr; ++row) ccj_zero_row(J_world, row, nv);
  } else {
    for (int row = 0; row < n_eq_rows; ++row) ccj_zero_row(J_world, row, nv);
  }
  device float* eq_scratch = dbg + dims[solver_metadata + 6];
  device float* Jp1 = eq_scratch;
  device float* Jr1 = Jp1 + 3 * max(nv, 1);
  device float* Jp2 = Jr1 + 3 * max(nv, 1);
  device float* Jr2 = Jp2 + 3 * max(nv, 1);

  for (int e = 0; e < neq; ++e) {
    int typ = eq_type[e];
    int objtype = eq_objtype[e]; // 1=body, 6=site, 0=unknown(joint)
    int rowadr = eq_rowadr[e];
    int span = eq_rownum[e];
    if (span <= 0) continue;
    // Flex families are emitted by their dedicated compiled row producers.
    // In particular, zero-row FLEXSTRAIN records must not reserve six weld
    // rows and overwrite the following equality.
    if (typ == 4 || typ == 5 || typ == 6) continue;
    bool active = !(flags & 1) && !(flags & 2) && (eq_active[int(world) * neq + e] != 0);
    if (typ == 3) {
      // Tendon equalities are assembled by the tendon-constraint stage
      // (milestone 008): it owns ten_length/ten_J and the cubic coupling.
      // Reserve zeros here so the row mapping stays dense.
      for (int k = 0; k < span; ++k) {
        int row = rowadr + k;
        if (row < 0 || row >= nr) continue;
        ccj_zero_row(J_world, row, nv);
        dbg[nr * nr + row] = 0.0f;
        dbg[nr * nr + nr + row] = 0.0f;
        dbg[nr * nr + 2 * nr + row] = 0.0f;
      }
      continue;
    }
    if (!active) {
      for (int k = 0; k < span; ++k) {
        int row = rowadr + k;
        if (row < 0 || row >= nr) continue;
        ccj_zero_row(J_world, row, nv);
        dbg[nr * nr + row] = 0.0f;
        dbg[nr * nr + nr + row] = 0.0f;
        dbg[nr * nr + 2 * nr + row] = 0.0f;
      }
      continue;
    }

    if (typ == 2) {
      // Joint equality (existing semantics, no Jdot).
      int j1 = eq_obj[e * 2];
      int j2 = eq_obj[e * 2 + 1];
      if (j1 < 0 || j1 >= nj) {
        for (int k = 0; k < 1; ++k) {
          int row = rowadr + k;
          ccj_zero_row(J_world, row, nv);
          dbg[nr * nr + row] = 0.0f;
          dbg[nr * nr + nr + row] = 0.0f;
          dbg[nr * nr + 2 * nr + row] = 0.0f;
        }
        continue;
      }
      int q1 = joint_qadr[j1];
      int d1 = joint_dadr[j1];
      float value1 = qpos[pb + q1];
      float ref1 = qpos0[q1];
      float data0 = eq_data[e * 11];
      float pos = value1 - ref1 - data0;
      float vel = qvel[qb + d1];
      EqFloatPair pos_pair = eq_pair_add({value1, 0.0f},
          eq_pair_neg({ref1, 0.0f}));
      pos_pair = eq_pair_add(pos_pair, {-data0, 0.0f});
      EqFloatPair vel_pair = {vel, 0.0f};
      float diag = dof_invweight[d1];
      int row = rowadr;
      ccj_zero_row(J_world, row, nv);
      if (j2 >= 0) {
        int q2 = joint_qadr[j2];
        int d2 = joint_dadr[j2];
        float dif = qpos[pb + q2] - qpos0[q2];
        EqFloatPair dif_pair = eq_pair_add({qpos[pb + q2], 0.0f},
            eq_pair_neg({qpos0[q2], 0.0f}));
        float poly = 0.0f, deriv = 0.0f, power = dif;
        EqFloatPair poly_pair = {0.0f, 0.0f};
        EqFloatPair power_pair = dif_pair;
        for (int k = 0; k < 4; ++k) {
          float coeff = eq_data[e * 11 + k + 1];
          poly += coeff * power;
          poly_pair = eq_pair_add(poly_pair,
              eq_pair_mul({coeff, 0.0f}, power_pair));
          deriv += (float(k + 1)) * coeff * pow(dif, float(k));
          power *= dif;
          power_pair = eq_pair_mul(power_pair, dif_pair);
        }
        pos -= poly;
        pos_pair = eq_pair_add(pos_pair, eq_pair_neg(poly_pair));
        ccj_set(J_world, row, d1, 1.0f, nv);
        ccj_set(J_world, row, d2, -deriv, nv);
        vel -= deriv * qvel[qb + d2];
        vel_pair = eq_pair_add(vel_pair, eq_pair_neg(eq_pair_mul(
            {deriv, 0.0f}, {qvel[qb + d2], 0.0f})));
        diag += dof_invweight[d2];
      } else {
        ccj_set(J_world, row, d1, 1.0f, nv);
      }
      // Reference params (same as solver's reference_params).
      float imp = eq_impedance(eq_sol, e, pos, 0.0f);
      EqFloatPair imp_pair = eq_impedance_pair(eq_sol, e, pos, 0.0f);
      float R = max(1e-15f, (1.0f - imp) * diag / imp);
      EqFloatPair ar_pair = eq_aref_pair(eq_sol, e, pos_pair,
          vel_pair, imp_pair, 0.0f, refsafe, timestep);
      float ar = ar_pair.hi;
      dbg[nr * nr + row] = R;
      dbg[nr * nr + nr + row] = ar;
      dbg[nr * nr + 2 * nr + row] = imp;
      aref_low[row] = ar_pair.lo;
    } else if (typ == 0) {
      // Connect (3 rows): body-body, body-world, site-site.
      int o1 = eq_obj[e * 2];
      int o2 = eq_obj[e * 2 + 1];
      float3 pos1 = float3(0.0f), pos2 = float3(0.0f);
      EqVec3Pair pos1_pair = eq_vec_pair_from_float(pos1);
      EqVec3Pair pos2_pair = eq_vec_pair_from_float(pos2);
      int b1 = 0, b2 = 0;
      if (objtype == 1) {
        b1 = o1; b2 = o2;
        float3 l1 = float3(eq_data[e * 11], eq_data[e * 11 + 1], eq_data[e * 11 + 2]);
        float3 l2 = float3(eq_data[e * 11 + 3], eq_data[e * 11 + 4], eq_data[e * 11 + 5]);
        float4 q1 = b1 >= 0 && b1 < nbody ? float4(body_quat[(bo + b1) * 4], body_quat[(bo + b1) * 4 + 1], body_quat[(bo + b1) * 4 + 2], body_quat[(bo + b1) * 4 + 3]) : float4(1,0,0,0);
        float4 q2 = b2 >= 0 && b2 < nbody ? float4(body_quat[(bo + b2) * 4], body_quat[(bo + b2) * 4 + 1], body_quat[(bo + b2) * 4 + 2], body_quat[(bo + b2) * 4 + 3]) : float4(1,0,0,0);
        float3 p1 = b1 >= 0 && b1 < nbody ? float3(body_pos[(bo + b1) * 3], body_pos[(bo + b1) * 3 + 1], body_pos[(bo + b1) * 3 + 2]) : float3(0.0f);
        float3 p2 = b2 >= 0 && b2 < nbody ? float3(body_pos[(bo + b2) * 3], body_pos[(bo + b2) * 3 + 1], body_pos[(bo + b2) * 3 + 2]) : float3(0.0f);
        pos1_pair = eq_qrot_translate_pair(q1, l1, p1);
        pos2_pair = eq_qrot_translate_pair(q2, l2, p2);
        pos1 = pos1_pair.hi;
        pos2 = pos2_pair.hi;
      } else {
        // Site-site: eq_data unused, use site_xpos directly.
        float3 s1 = o1 >= 0 && o1 < nsite ? float3(site_pos[(so + o1) * 3], site_pos[(so + o1) * 3 + 1], site_pos[(so + o1) * 3 + 2]) : float3(0.0f);
        float3 s2 = o2 >= 0 && o2 < nsite ? float3(site_pos[(so + o2) * 3], site_pos[(so + o2) * 3 + 1], site_pos[(so + o2) * 3 + 2]) : float3(0.0f);
        pos1 = s1; pos2 = s2;
        pos1_pair = eq_vec_pair_from_float(s1);
        pos2_pair = eq_vec_pair_from_float(s2);
        b1 = o1 >= 0 && o1 < nsite ? site_bodyid[o1] : 0;
        b2 = o2 >= 0 && o2 < nsite ? site_bodyid[o2] : 0;
      }
      EqVec3Pair cpos_pair = eq_vec_pair_sub(pos1_pair, pos2_pair);
      float3 cpos = cpos_pair.hi;

      // Jacobians J1, J2 (3 x nv) via parent traversal (plus unused Jr for weld reuse).
      eq_point_jac(pos1, b1, body_pos, body_quat, joint_anchor, joint_axis,
                   body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_dofadr,
                   nbody, njnt, nv, bo, jo, Jp1, Jr1);
      eq_point_jac(pos2, b2, body_pos, body_quat, joint_anchor, joint_axis,
                   body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_dofadr,
                   nbody, njnt, nv, bo, jo, Jp2, Jr2);

      // Velocities v1, v2 and difference vel.
      float3 v1 = float3(0.0f), v2 = float3(0.0f);
      for (int i = 0; i < nv; ++i) {
        float qv = qvel[qb + i];
        v1.x += Jp1[0 * nv + i] * qv; v1.y += Jp1[1 * nv + i] * qv; v1.z += Jp1[2 * nv + i] * qv;
        v2.x += Jp2[0 * nv + i] * qv; v2.y += Jp2[1 * nv + i] * qv; v2.z += Jp2[2 * nv + i] * qv;
      }
      float3 vel = v1 - v2;

      // Exact dense mj_jacDot is applied in a follow-up pass from cdof and
      // cdof_dot; do not use an articulated-motion surrogate here.
      float3 jdv = float3(0.0f);

      // diagA (translational invweights).
      float w1 = 0.0f, w2 = 0.0f;
      if (b1 >= 0 && b1 < nbody) w1 = body_invweight[b1 * 2];
      if (b2 >= 0 && b2 < nbody) w2 = body_invweight[b2 * 2];
      float diag = max(w1 + w2, 1e-15f);

      // Impedance from norm(cpos), margin 0.
      float pos_norm = length(cpos);
      float imp = eq_impedance(eq_sol, e, pos_norm, 0.0f);
      EqFloatPair imp_pair = eq_impedance_pair(eq_sol, e, pos_norm, 0.0f);
      imp = clamp(imp, 1e-6f, 0.999999f);
      float R = max(1e-15f, (1.0f - imp) * diag / imp);

      // Degenerate empty Jacobian with zero inverse weight (e.g. both sides
      // mocap/world-fixed) matches MuJoCo skipping the constraint: force all
      // rows to exact zero so multipliers stay zero.
      float jmax = 0.0f;
      for (int i = 0; i < nv; ++i) {
        jmax = max(jmax, abs(Jp1[0 * nv + i] - Jp2[0 * nv + i]));
        jmax = max(jmax, abs(Jp1[1 * nv + i] - Jp2[1 * nv + i]));
        jmax = max(jmax, abs(Jp1[2 * nv + i] - Jp2[2 * nv + i]));
      }
      bool degenerate = (jmax == 0.0f && (w1 + w2) == 0.0f);
      if (degenerate) R = 0.0f;

      for (int k = 0; k < 3; ++k) {
        int row = rowadr + k;
        if (row < 0 || row >= nr) continue;
        float v = k == 0 ? vel.x : (k == 1 ? vel.y : vel.z);
        float jd = k == 0 ? jdv.x : (k == 1 ? jdv.y : jdv.z);
        EqFloatPair ar_pair = degenerate ? EqFloatPair{0.0f, 0.0f}
            : eq_aref_pair(eq_sol, e, eq_vec_pair_component(cpos_pair, k), {v, 0.0f},
                           imp_pair, jd, refsafe, timestep);
        float ar = ar_pair.hi;
        for (int i = 0; i < nv; ++i) {
          float j1v = k == 0 ? Jp1[0 * nv + i] : (k == 1 ? Jp1[1 * nv + i] : Jp1[2 * nv + i]);
          float j2v = k == 0 ? Jp2[0 * nv + i] : (k == 1 ? Jp2[1 * nv + i] : Jp2[2 * nv + i]);
          ccj_set(J_world, row, i, j1v - j2v, nv);
        }
        dbg[nr * nr + row] = R;
        dbg[nr * nr + nr + row] = ar;
        dbg[nr * nr + 2 * nr + row] = imp;
        aref_low[row] = ar_pair.lo;
      }
    } else {
      // Weld (6 rows): translational (3) + rotational (3) with pinned
      // MuJoCo 3.10 quaternion error, torquescale, and reference semantics.
      int o1 = eq_obj[e * 2];
      int o2 = eq_obj[e * 2 + 1];
      float torquescale = eq_data[e * 11 + 10];
      float3 pos1 = float3(0.0f), pos2 = float3(0.0f);
      EqVec3Pair pos1_pair = eq_vec_pair_from_float(pos1);
      EqVec3Pair pos2_pair = eq_vec_pair_from_float(pos2);
      int b1 = 0, b2 = 0;
      float4 q0_full = float4(1,0,0,0), q1_full = float4(1,0,0,0);
      float4 relpose = float4(1,0,0,0);
      bool is_site = (objtype == 6);
      if (!is_site) {
        b1 = o1; b2 = o2;
        float3 l1 = float3(eq_data[e * 11 + 3], eq_data[e * 11 + 4], eq_data[e * 11 + 5]);
        float3 l2 = float3(eq_data[e * 11], eq_data[e * 11 + 1], eq_data[e * 11 + 2]);
        float4 qb1 = b1 >= 0 && b1 < nbody ? float4(body_quat[(bo + b1) * 4], body_quat[(bo + b1) * 4 + 1], body_quat[(bo + b1) * 4 + 2], body_quat[(bo + b1) * 4 + 3]) : float4(1,0,0,0);
        float4 qb2 = b2 >= 0 && b2 < nbody ? float4(body_quat[(bo + b2) * 4], body_quat[(bo + b2) * 4 + 1], body_quat[(bo + b2) * 4 + 2], body_quat[(bo + b2) * 4 + 3]) : float4(1,0,0,0);
        float3 p1 = b1 >= 0 && b1 < nbody ? float3(body_pos[(bo + b1) * 3], body_pos[(bo + b1) * 3 + 1], body_pos[(bo + b1) * 3 + 2]) : float3(0.0f);
        float3 p2 = b2 >= 0 && b2 < nbody ? float3(body_pos[(bo + b2) * 3], body_pos[(bo + b2) * 3 + 1], body_pos[(bo + b2) * 3 + 2]) : float3(0.0f);
        pos1_pair = eq_qrot_translate_pair(qb1, l1, p1);
        pos2_pair = eq_qrot_translate_pair(qb2, l2, p2);
        pos1 = pos1_pair.hi;
        pos2 = pos2_pair.hi;
        q0_full = qb1;
        q1_full = qb2;
        relpose = float4(eq_data[e * 11 + 6], eq_data[e * 11 + 7], eq_data[e * 11 + 8], eq_data[e * 11 + 9]);
      } else {
        float3 s1 = o1 >= 0 && o1 < nsite ? float3(site_pos[(so + o1) * 3], site_pos[(so + o1) * 3 + 1], site_pos[(so + o1) * 3 + 2]) : float3(0.0f);
        float3 s2 = o2 >= 0 && o2 < nsite ? float3(site_pos[(so + o2) * 3], site_pos[(so + o2) * 3 + 1], site_pos[(so + o2) * 3 + 2]) : float3(0.0f);
        pos1 = s1; pos2 = s2;
        pos1_pair = eq_vec_pair_from_float(s1);
        pos2_pair = eq_vec_pair_from_float(s2);
        b1 = o1 >= 0 && o1 < nsite ? site_bodyid[o1] : 0;
        b2 = o2 >= 0 && o2 < nsite ? site_bodyid[o2] : 0;
        float4 sq1 = o1 >= 0 && o1 < nsite ? float4(site_quat[(so + o1) * 4], site_quat[(so + o1) * 4 + 1], site_quat[(so + o1) * 4 + 2], site_quat[(so + o1) * 4 + 3]) : float4(1,0,0,0);
        float4 sq2 = o2 >= 0 && o2 < nsite ? float4(site_quat[(so + o2) * 4], site_quat[(so + o2) * 4 + 1], site_quat[(so + o2) * 4 + 2], site_quat[(so + o2) * 4 + 3]) : float4(1,0,0,0);
        q0_full = sq1;
        q1_full = sq2;
        relpose = float4(1,0,0,0);
      }
      EqVec3Pair cpos_t_pair = eq_vec_pair_sub(pos1_pair, pos2_pair);
      float3 cpos_t = cpos_t_pair.hi;
      // Orientation error: quat = q0*rel (body) or sq1 (site); quat1 = neg(q1 or sq2).
      float4 quat = is_site ? q0_full : eq_qmul(q0_full, relpose);
      float4 quat1 = eq_qneg(q1_full);
      float4 quat2 = eq_qmul(quat1, quat);
      float3 cpos_r = torquescale * quat2.yzw;

      // Translational + rotational Jacobians via parent traversal.
      eq_point_jac(pos1, b1, body_pos, body_quat, joint_anchor, joint_axis,
                   body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_dofadr,
                   nbody, njnt, nv, bo, jo, Jp1, Jr1);
      eq_point_jac(pos2, b2, body_pos, body_quat, joint_anchor, joint_axis,
                   body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_dofadr,
                   nbody, njnt, nv, bo, jo, Jp2, Jr2);

      float3 v1 = float3(0.0f), v2 = float3(0.0f);
      for (int i = 0; i < nv; ++i) {
        float qv = qvel[qb + i];
        v1.x += Jp1[0 * nv + i] * qv; v1.y += Jp1[1 * nv + i] * qv; v1.z += Jp1[2 * nv + i] * qv;
        v2.x += Jp2[0 * nv + i] * qv; v2.y += Jp2[1 * nv + i] * qv; v2.z += Jp2[2 * nv + i] * qv;
      }
      float3 vel_t = v1 - v2;
      // Exact dense mj_jacDot is applied in the follow-up pass.
      float3 jdv_t = float3(0.0f);

      // Rotational Jacobian: 0.5 * neg(q1) * (Jr1-Jr2) * quat * torquescale.
      float3 vel_r = float3(0.0f);
      for (int i = 0; i < nv; ++i) {
        float3 axis = float3(Jr1[0 * nv + i] - Jr2[0 * nv + i],
                             Jr1[1 * nv + i] - Jr2[1 * nv + i],
                             Jr1[2 * nv + i] - Jr2[2 * nv + i]);
        float4 t1 = eq_qmul_axis(quat1, axis);
        float4 t2 = eq_qmul(t1, quat);
        float3 jrot = 0.5f * float3(t2.y, t2.z, t2.w) * torquescale;
        float qv = qvel[qb + i];
        vel_r += jrot * qv;
      }

      // diagA: translational uses body trans invweights, rotational uses rot invweights.
      float w1t = 0.0f, w2t = 0.0f, w1r = 0.0f, w2r = 0.0f;
      if (b1 >= 0 && b1 < nbody) { w1t = body_invweight[b1 * 2]; w1r = body_invweight[b1 * 2 + 1]; }
      if (b2 >= 0 && b2 < nbody) { w2t = body_invweight[b2 * 2]; w2r = body_invweight[b2 * 2 + 1]; }
      float diag_t = max(w1t + w2t, 1e-15f);
      float diag_r = max(w1r + w2r, 1e-15f);

      float pos_norm = sqrt(dot(cpos_t, cpos_t) + dot(cpos_r, cpos_r));
      float imp = eq_impedance(eq_sol, e, pos_norm, 0.0f);
      EqFloatPair imp_pair = eq_impedance_pair(eq_sol, e, pos_norm, 0.0f);
      imp = clamp(imp, 1e-6f, 0.999999f);
      float R_t = max(1e-15f, (1.0f - imp) * diag_t / imp);
      float R_r = max(1e-15f, (1.0f - imp) * diag_r / imp);

      // Degenerate empty Jacobian with zero inverse weight on both sides
      // (e.g. both-mocap weld) matches MuJoCo skipping the constraint.
      float jmax = 0.0f;
      for (int i = 0; i < nv; ++i) {
        jmax = max(jmax, abs(Jp1[0 * nv + i] - Jp2[0 * nv + i]));
        jmax = max(jmax, abs(Jp1[1 * nv + i] - Jp2[1 * nv + i]));
        jmax = max(jmax, abs(Jp1[2 * nv + i] - Jp2[2 * nv + i]));
        float3 axis = float3(Jr1[0 * nv + i] - Jr2[0 * nv + i],
                             Jr1[1 * nv + i] - Jr2[1 * nv + i],
                             Jr1[2 * nv + i] - Jr2[2 * nv + i]);
        float4 t1 = eq_qmul_axis(quat1, axis);
        float4 t2 = eq_qmul(t1, quat);
        float3 jrot = 0.5f * float3(t2.y, t2.z, t2.w) * torquescale;
        jmax = max(jmax, max(abs(jrot.x), max(abs(jrot.y), abs(jrot.z))));
      }
      bool degenerate = (jmax == 0.0f && (w1t + w2t) == 0.0f && (w1r + w2r) == 0.0f);
      if (degenerate) { R_t = 0.0f; R_r = 0.0f; }

      for (int k = 0; k < 3; ++k) {
        int row = rowadr + k;
        if (row < 0 || row >= nr) continue;
        float v = k == 0 ? vel_t.x : (k == 1 ? vel_t.y : vel_t.z);
        float jd = k == 0 ? jdv_t.x : (k == 1 ? jdv_t.y : jdv_t.z);
        EqFloatPair ar_pair = degenerate ? EqFloatPair{0.0f, 0.0f}
            : eq_aref_pair(eq_sol, e, eq_vec_pair_component(cpos_t_pair, k), {v, 0.0f},
                           imp_pair, jd, refsafe, timestep);
        float ar = ar_pair.hi;
        for (int i = 0; i < nv; ++i) {
          float a = k == 0 ? Jp1[0 * nv + i] : (k == 1 ? Jp1[1 * nv + i] : Jp1[2 * nv + i]);
          float b_ = k == 0 ? Jp2[0 * nv + i] : (k == 1 ? Jp2[1 * nv + i] : Jp2[2 * nv + i]);
          ccj_set(J_world, row, i, a - b_, nv);
        }
        dbg[nr * nr + row] = R_t;
        dbg[nr * nr + nr + row] = ar;
        dbg[nr * nr + 2 * nr + row] = degenerate ? 0.0f : imp;
        aref_low[row] = ar_pair.lo;
      }
      // Exact three-term rotational Jdot is applied in the follow-up pass.
      float3 rot_corr = float3(0.0f);
      for (int k = 0; k < 3; ++k) {
        int row = rowadr + 3 + k;
        if (row < 0 || row >= nr) continue;
        float c = k == 0 ? cpos_r.x : (k == 1 ? cpos_r.y : cpos_r.z);
        float v = k == 0 ? vel_r.x : (k == 1 ? vel_r.y : vel_r.z);
        float jd = k == 0 ? rot_corr.x : (k == 1 ? rot_corr.y : rot_corr.z);
        EqFloatPair ar_pair = degenerate ? EqFloatPair{0.0f, 0.0f}
            : eq_aref_pair(eq_sol, e, {c, 0.0f}, {v, 0.0f},
                           imp_pair, jd, refsafe, timestep);
        float ar = ar_pair.hi;
        for (int i = 0; i < nv; ++i) {
          float3 axis = float3(Jr1[0 * nv + i] - Jr2[0 * nv + i],
                               Jr1[1 * nv + i] - Jr2[1 * nv + i],
                               Jr1[2 * nv + i] - Jr2[2 * nv + i]);
          float4 t1 = eq_qmul_axis(quat1, axis);
          float4 t2 = eq_qmul(t1, quat);
          float3 jrot = 0.5f * float3(t2.y, t2.z, t2.w) * torquescale;
          ccj_set(J_world, row, i,
              k == 0 ? jrot.x : (k == 1 ? jrot.y : jrot.z), nv);
        }
        dbg[nr * nr + row] = R_r;
        dbg[nr * nr + nr + row] = ar;
        dbg[nr * nr + 2 * nr + row] = degenerate ? 0.0f : imp;
        aref_low[row] = ar_pair.lo;
      }
    }
  }

  // Publish ownership/activity and bilateral bounds for rows produced here.
  // The active slice is also the cached-position validity mask; without this
  // write a later VEL-only refresh treats a valid equality row as inactive
  // and zeros its refreshed aref. Tendon and flex equalities are owned by
  // their later dedicated producers and must not be marked here.
  for (int e = 0; e < neq; ++e) {
    int typ = eq_type[e];
    if (typ != 0 && typ != 1 && typ != 2) continue;
    int rowadr = eq_rowadr[e];
    int span = eq_rownum[e];
    if (span <= 0) continue;
    bool eq_enabled = !(flags & 1) && !(flags & 2)
        && (eq_active[int(world) * neq + e] != 0);
    for (int k = 0; k < span; ++k) {
      int row = rowadr + k;
      if (row < 0 || row >= nr) continue;
      float R = dbg[nr * nr + row];
      float ar = dbg[nr * nr + nr + row];
      bool row_enabled = eq_enabled && (R > 0.0f || abs(ar) > 0.0f);
      dbg[nr * nr + 4 * nr + row] = row_enabled ? -INFINITY : 0.0f;
      dbg[nr * nr + 5 * nr + row] = row_enabled ? INFINITY : 0.0f;
      dbg[nr * nr + 6 * nr + row] = row_enabled ? 1.0f : 0.0f;
    }
  }
}

#endif  // MUJOCO_METAL_EQUALITY_HELPERS_ONLY
