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

inline float3 rotate_q(float4 q, float3 v) {
  float3 u = q.yzw;
  return v + 2.0f * cross(u, cross(u, v) + q.x * v);
}
inline float3 cross3(float3 a, float3 b) { return cross(a, b); }

inline float impedance_at(device const float* imp, int index, float pos, float margin) {
  float d0 = imp[index * 5];
  float d1 = imp[index * 5 + 1];
  float width = imp[index * 5 + 2];
  float mid = imp[index * 5 + 3];
  float power = imp[index * 5 + 4];
  if (d0 == d1 || width <= 1e-15f) return 0.5f * (d0 + d1);
  float x = abs((pos - margin) / width);
  if (x >= 1.0f) return d1;
  if (x <= 0.0f) return d0;
  float y = power == 1.0f ? x : (x <= mid ? pow(x, power) / pow(mid, power - 1.0f)
      : 1.0f - pow(1.0f - x, power) / pow(1.0f - mid, power - 1.0f));
  return d0 + y * (d1 - d0);
}

inline void reference_params(device const float* solref, device const float* solimp,
    int index, float pos, float margin, float vel, float diag, bool friction,
    float timestep, bool refsafe, thread float& compliance, thread float& aref) {
  float width = max(1e-15f, solimp[index * 5 + 1]);
  float impedance = impedance_at(solimp, index, pos, margin);
  float r0 = solref[index * 2];
  float r1 = solref[index * 2 + 1];
  if (refsafe && r0 > 0.0f) r0 = max(r0, 2.0f * timestep);
  float K = r0 > 0.0f ? 1.0f / max(1e-15f, width * width * r0 * r0 * r1 * r1)
                      : -r0 / max(1e-15f, width * width);
  float B = r1 > 0.0f ? 2.0f / max(1e-15f, width * r0) : -r1 / width;
  if (friction) K = 0.0f;
  compliance = max(1e-15f, (1.0f - impedance) * diag / impedance);
  aref = -B * vel - K * impedance * (pos - margin);
}

inline float solve_contact_block(
    thread const float* A, thread const float* b, int n,
    thread float* solution) {
  float best_error = 1e30f;
  float best_objective = 1e30f;
  for (int i = 0; i < 4; ++i) solution[i] = 0.0f;
  for (int set = 0; set < (1 << n); ++set) {
    thread int ids[4];
    int nactive = 0;
    for (int i = 0; i < n; ++i) if (set & (1 << i)) ids[nactive++] = i;
    thread float mat[16], rhs[4], x[4], candidate[4];
    for (int i = 0; i < 16; ++i) mat[i] = 0.0f;
    for (int i = 0; i < 4; ++i) { rhs[i] = 0.0f; x[i] = 0.0f; candidate[i] = 0.0f; }
    for (int i = 0; i < nactive; ++i) {
      rhs[i] = -b[ids[i]];
      for (int j = 0; j < nactive; ++j)
        mat[i * 4 + j] = A[ids[i] * 4 + ids[j]];
    }
    bool valid = true;
    for (int p = 0; p < nactive; ++p) {
      int pivot_row = p;
      float pivot_abs = abs(mat[p * 4 + p]);
      for (int r = p + 1; r < nactive; ++r) {
        float value = abs(mat[r * 4 + p]);
        if (value > pivot_abs) { pivot_abs = value; pivot_row = r; }
      }
      if (!(pivot_abs > 1e-15f) || !isfinite(pivot_abs)) { valid = false; break; }
      if (pivot_row != p) {
        for (int j = 0; j < nactive; ++j) {
          float tmp = mat[p * 4 + j];
          mat[p * 4 + j] = mat[pivot_row * 4 + j];
          mat[pivot_row * 4 + j] = tmp;
        }
        float tmp = rhs[p];
        rhs[p] = rhs[pivot_row];
        rhs[pivot_row] = tmp;
      }
      float pivot = mat[p * 4 + p];
      for (int r = p + 1; r < nactive; ++r) {
        float factor = mat[r * 4 + p] / pivot;
        for (int j = p; j < nactive; ++j) mat[r * 4 + j] -= factor * mat[p * 4 + j];
        rhs[r] -= factor * rhs[p];
      }
    }
    if (!valid) continue;
    for (int r = nactive - 1; r >= 0; --r) {
      float value = rhs[r];
      for (int j = r + 1; j < nactive; ++j) value -= mat[r * 4 + j] * x[j];
      x[r] = value / mat[r * 4 + r];
      if (!isfinite(x[r])) valid = false;
    }
    if (!valid) continue;
    for (int i = 0; i < nactive; ++i) candidate[ids[i]] = x[i];
    float error = 0.0f;
    float scale = 1.0f;
    float objective = 0.0f;
    for (int i = 0; i < n; ++i) {
      float gradient = b[i];
      for (int j = 0; j < n; ++j) gradient += A[i * 4 + j] * candidate[j];
      scale += abs(b[i]);
      for (int j = 0; j < n; ++j) scale += abs(A[i * 4 + j] * candidate[j]);
      bool active = (set & (1 << i)) != 0;
      error = max(error, active ? max(0.0f, -candidate[i]) : max(0.0f, -gradient));
      objective += 0.5f * candidate[i] * (gradient + b[i]);
    }
    float scaled_error = error / scale;
    if (scaled_error < best_error ||
        (scaled_error == best_error && objective < best_objective)) {
      best_error = scaled_error;
      best_objective = objective;
      for (int i = 0; i < 4; ++i) solution[i] = candidate[i];
    }
  }
  return best_error;
}

// Geometric contact detection and Jacobian generation
kernel void contact_normal(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* body_pos [[buffer(2)]],
    device const float* body_quat [[buffer(3)]],
    device const float* anchors [[buffer(4)]],
    device const float* axes [[buffer(5)]],
    device const float* qvel [[buffer(6)]],
    device const int* geom1 [[buffer(7)]],
    device const int* geom2 [[buffer(8)]],
    device const float* radius1 [[buffer(9)]],
    device const float* radius2 [[buffer(10)]],
    device const float* margin [[buffer(11)]],
    device const float* gap [[buffer(12)]],
    device const float* solref [[buffer(13)]],
    device const float* solimp [[buffer(14)]],
    device const int* condim [[buffer(15)]],
    device const float* friction [[buffer(16)]],
    device const int* geom_bodyid [[buffer(17)]],
    device const int* body_parentid [[buffer(18)]],
    device const int* body_jntadr [[buffer(19)]],
    device const int* body_jntnum [[buffer(20)]],
    device const int* jnt_type [[buffer(21)]],
    device const int* jnt_dofadr [[buffer(22)]],
    device const float* body_invweight0 [[buffer(23)]],
    device float* row_data [[buffer(24)]],
    device float* frame [[buffer(25)]],
    device float* jacobian [[buffer(26)]],
    constant int* dims [[buffer(27)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0];
  int nc = dims[1];
  int batch = dims[2];
  int nbody = dims[3];
  int njnt = dims[4];
  int ngeom = dims[5];
  int world = int(tid) / max(nc, 1);
  int slot = int(tid) % max(nc, 1);
  if (uint(world) >= uint(batch) || slot >= nc) return;

  int out = world * nc + slot;
  int jbase = out * 5 * nv;
  int rb = out * 5 * 6;
  int fb = out * 12;
  for (int k = 0; k < 5 * 6; ++k) row_data[rb + k] = 0.0f;
  for (int k = 0; k < 5 * nv; ++k) jacobian[jbase + k] = 0.0f;
  for (int k = 0; k < 12; ++k) frame[fb + k] = 0.0f;

  int a = geom1[slot];
  int b = geom2[slot];
  int go = world * ngeom;
  float3 pa = float3(geom_pos[(go + a) * 3], geom_pos[(go + a) * 3 + 1], geom_pos[(go + a) * 3 + 2]);
  float3 pb = float3(geom_pos[(go + b) * 3], geom_pos[(go + b) * 3 + 1], geom_pos[(go + b) * 3 + 2]);
  float4 qa = float4(geom_quat[(go + a) * 4], geom_quat[(go + a) * 4 + 1], geom_quat[(go + a) * 4 + 2], geom_quat[(go + a) * 4 + 3]);
  float4 qb = float4(geom_quat[(go + b) * 4], geom_quat[(go + b) * 4 + 1], geom_quat[(go + b) * 4 + 2], geom_quat[(go + b) * 4 + 3]);
  float d = 0.0f;
  float3 n = float3(0.0f);
  float3 point = float3(0.0f);
  if (radius1[slot] < 0.0f) {
    float3 pn = rotate_q(qa, float3(0, 0, 1));
    d = dot(pb - pa, pn) - radius2[slot];
    n = pn;
    point = pb - n * (d * 0.5f + radius2[slot]);
  } else if (radius2[slot] < 0.0f) {
    float3 pn = rotate_q(qb, float3(0, 0, 1));
    d = dot(pa - pb, pn) - radius1[slot];
    n = -pn;
    point = pa - pn * (d * 0.5f + radius1[slot]);
  } else {
    float3 delta = pb - pa;
    float distance = length(delta);
    if (!(distance > 1e-12f)) return;
    n = delta / distance;
    d = distance - radius1[slot] - radius2[slot];
    point = pa + n * (radius1[slot] + 0.5f * d);
  }
  float m = margin[slot];
  float g = gap[slot];
  if (d > m + g) return;

  float3 t1 = (n.y < 0.5f && n.y > -0.5f) ? float3(0, 1, 0) : float3(0, 0, 1);
  t1 = normalize(t1 - n * dot(n, t1));
  float3 t2 = cross(n, t1);

  int ba = geom_bodyid[a];
  int bb = geom_bodyid[b];
  int bo = world * nbody;
  int jo = world * njnt;
  float3 rel = point;
  for (int side = 0; side < 2; ++side) {
    int body = side == 0 ? ba : bb;
    float sign = side == 0 ? -1.0f : 1.0f;
    while (body > 0 && body < nbody) {
      int ja = body_jntadr[body];
      int jn = body_jntnum[body];
      for (int jj = 0; jj < jn; ++jj) {
        int j = ja + jj;
        int da = jnt_dofadr[j];
        int typ = jnt_type[j];
        int nd = typ == 0 ? 6 : typ == 1 ? 3 : 1;
        for (int q = 0; q < nd; ++q) {
          int dof = da + q;
          if (dof < 0 || dof >= nv) continue;
          float3 col = float3(0.0f);
          if (typ == 2) {
            col = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
          } else if (typ == 3) {
            float3 axis = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
            col = cross3(axis, rel - float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]));
          } else if (typ == 0 || typ == 1) {
            if (typ == 0 && q < 3) {
              col = float3(q == 0, q == 1, q == 2);
            } else {
              int qrot = typ == 0 ? q - 3 : q;
              float4 bq = float4(body_quat[(bo + body) * 4], body_quat[(bo + body) * 4 + 1], body_quat[(bo + body) * 4 + 2], body_quat[(bo + body) * 4 + 3]);
              float3 axis = rotate_q(bq, float3(qrot == 0, qrot == 1, qrot == 2));
              float3 pivot = typ == 0 ? float3(body_pos[(bo + body) * 3], body_pos[(bo + body) * 3 + 1], body_pos[(bo + body) * 3 + 2])
                                      : float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]);
              col = cross3(axis, rel - pivot);
            }
          }
          jacobian[jbase + dof] += sign * dot(n, col);
          jacobian[jbase + nv + dof] += sign * dot(t1, col);
          jacobian[jbase + 2 * nv + dof] += sign * dot(t2, col);
        }
      }
      body = body_parentid[body];
    }
  }

  int dim = condim[slot];
  float mu0 = friction[slot * 2];
  float mu1 = friction[slot * 2 + 1];
  if (dim == 3) {
    for (int i = 0; i < nv; ++i) {
      float jn = jacobian[jbase + i];
      float jt1 = jacobian[jbase + nv + i];
      float jt2 = jacobian[jbase + 2 * nv + i];
      jacobian[jbase + nv + i] = jn + mu0 * jt1;
      jacobian[jbase + 2 * nv + i] = jn - mu0 * jt1;
      jacobian[jbase + 3 * nv + i] = jn + mu1 * jt2;
      jacobian[jbase + 4 * nv + i] = jn - mu1 * jt2;
    }
  }

  float d0 = solimp[slot * 5];
  float d1 = solimp[slot * 5 + 1];
  float width = solimp[slot * 5 + 2];
  float mid = solimp[slot * 5 + 3];
  float power = solimp[slot * 5 + 4];
  float impedance;
  float x = width > 1e-15f ? abs((d - m) / width) : 1.0f;
  if (d0 == d1 || width <= 1e-15f) impedance = 0.5f * (d0 + d1);
  else if (x <= 0.0f) impedance = d0;
  else if (x >= 1.0f) impedance = d1;
  else {
    float y;
    if (power == 1.0f) y = x;
    else if (x <= mid) y = pow(x, power) / pow(mid, power - 1.0f);
    else y = 1.0f - pow(1.0f - x, power) / pow(1.0f - mid, power - 1.0f);
    impedance = d0 + y * (d1 - d0);
  }

  float r0 = solref[slot * 2];
  float r1 = solref[slot * 2 + 1];
  float K, B;
  if (r0 > 0.0f) K = 1.0f / max(1e-15f, d1 * d1 * r0 * r0 * r1 * r1);
  else K = -r0 / max(1e-15f, d1 * d1);
  if (r1 > 0.0f) B = 2.0f / max(1e-15f, d1 * r0);
  else B = -r1 / max(1e-15f, d1);

  float diag_approx = body_invweight0[ba * 2] + body_invweight0[bb * 2];
  for (int row = 0; row < (dim == 3 ? 5 : 1); ++row) {
    float vel = 0.0f;
    for (int i = 0; i < nv; ++i) vel += jacobian[jbase + row * nv + i] * qvel[world * nv + i];
    int r = rb + row * 6;
    row_data[r] = (d < m && (dim == 1 || row > 0)) ? 1.0f : 0.0f;
    row_data[r + 1] = d;
    row_data[r + 2] = vel;
    row_data[r + 3] = -B * vel - K * impedance * (d - m);
    row_data[r + 4] = impedance;
    row_data[r + 5] = diag_approx;
  }
  for (int k = 0; k < 3; ++k) {
    frame[fb + k] = n[k];
    frame[fb + 3 + k] = t1[k];
    frame[fb + 6 + k] = t2[k];
    frame[fb + 9 + k] = point[k];
  }
}

// Coupled projected constraint solve
kernel void solve_coupled_constraints(
    device const float* mass [[buffer(0)]],
    device const float* qfrc [[buffer(1)]],
    device const float* qpos [[buffer(2)]],
    device const float* qvel [[buffer(3)]],
    device const int* eq_active [[buffer(4)]],
    device const int* joint_qadr [[buffer(5)]],
    device const float* qpos0 [[buffer(6)]],
    device const int* joint_dadr [[buffer(7)]],
    device const uchar* joint_limited [[buffer(8)]],
    device const float* joint_limit_params [[buffer(9)]],
    device const float* joint_sol_params [[buffer(10)]],
    device const float* frictionloss [[buffer(11)]],
    device const float* invweight [[buffer(12)]],
    device const float* dof_sol_params [[buffer(13)]],
    device const int* eq_obj [[buffer(14)]],
    device const float* eq_data [[buffer(15)]],
    device const float* eq_sol_params [[buffer(16)]],
    device const float* contact_jacobian [[buffer(17)]],
    device const float* contact_row_data [[buffer(18)]],
    device const float* contact_friction [[buffer(19)]],
    device const int* contact_condim [[buffer(20)]],
    constant int* dims [[buffer(21)]],
    constant float* params [[buffer(22)]],
    device float* out_force [[buffer(23)]],
    device float* out_acc [[buffer(24)]],
    device int* out_status [[buffer(25)]],
    device float* out_diagnostics [[buffer(26)]],
    device float* out_contact_force [[buffer(27)]],
    device float* out_joint_force [[buffer(28)]],
    device float* workspace_J [[buffer(29)]],
    device float* workspace_debug [[buffer(30)]],
    uint world [[thread_position_in_grid]]) {
  int nq = dims[0];
  int nv = dims[1];
  int nj = dims[2];
  int neq = dims[3];
  int nc = dims[4];
  int batch = dims[5];
  int flags = dims[6];
  bool refsafe = dims[7] != 0;
  int maxiter = dims[8];
  int nr = dims[9];
  if (world >= uint(batch)) return;

  int mb = world * nv * nv;
  int qb = world * nv;
  int pb = world * nq;
  out_status[world] = 0;
  out_diagnostics[world * 2] = 0.0f;
  out_diagnostics[world * 2 + 1] = 0.0f;
  for (int i = 0; i < nv; ++i) { out_force[qb + i] = 0.0f; out_acc[qb + i] = 0.0f; }
  if (workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * (nr * nr + 4 * nr);
    for (int i = 0; i < nr * nr + 4 * nr; ++i) dbg[i] = 0.0f;
  }
  if (nv == 0) return;
  if (nv > 32 || nr > 96) { out_status[world] = 2; return; }

  device float* J_world = workspace_J + world * nr * nv;
  for (int i = 0; i < nr * nv; ++i) J_world[i] = 0.0f;

  thread float L[32 * 32];
  thread float Z[96 * 32];
  thread float W[96 * 96];
  thread float R[96], ar[96], lo[96], hi[96], lam[96], rhs[96];
  thread bool enabled[96];
  thread float y[32], x[32];
  for (int i = 0; i < nr * nr; ++i) W[i] = 0.0f;
  for (int i = 0; i < nr; ++i) {
    R[i] = 0.0f; ar[i] = 0.0f; lo[i] = 0.0f; hi[i] = 0.0f; lam[i] = 0.0f; enabled[i] = false;
  }

  int base_contact = neq + nv + 2 * nj;

  // 1. Joint constraints (if constraint disable flag not set: bit 0 mjDSBL_CONSTRAINT)
  if ((flags & 1) == 0) {
    // Equalities (bit 1 mjDSBL_EQUALITY)
    for (int e = 0; e < neq; ++e) {
      if ((flags & 2) != 0 || eq_active[world * neq + e] == 0) continue;
      int j1 = eq_obj[e * 2];
      int j2 = eq_obj[e * 2 + 1];
      int q1 = joint_qadr[j1];
      int d1 = joint_dadr[j1];
      float value1 = qpos[pb + q1];
      float ref1 = qpos0[q1];
      float data0 = eq_data[e * 11];
      float pos = value1 - ref1 - data0;
      float vel = qvel[qb + d1];
      float diag = invweight[d1];
      if (j2 >= 0) {
        int q2 = joint_qadr[j2];
        int d2 = joint_dadr[j2];
        float dif = qpos[pb + q2] - qpos0[q2];
        float poly = 0.0f, deriv = 0.0f, power = dif;
        for (int k = 0; k < 4; ++k) {
          poly += eq_data[e * 11 + k + 1] * power;
          deriv += (k + 1) * eq_data[e * 11 + k + 1] * pow(dif, float(k));
          power *= dif;
        }
        pos -= poly;
        J_world[e * nv + d1] = 1.0f;
        J_world[e * nv + d2] = -deriv;
        vel -= deriv * qvel[qb + d2];
        diag += invweight[d2];
      } else {
        J_world[e * nv + d1] = 1.0f;
      }
      reference_params(eq_sol_params + e * 7, eq_sol_params + e * 7 + 2, 0, pos, 0.0f, vel, diag, false, params[0], refsafe, R[e], ar[e]);
      lo[e] = -INFINITY;
      hi[e] = INFINITY;
      enabled[e] = true;
    }

    // Frictionloss (bit 2 mjDSBL_FRICTIONLOSS)
    for (int d = 0; d < nv; ++d) {
      int row = neq + d;
      float loss = frictionloss[d];
      if ((flags & 4) != 0 || loss <= 0.0f) continue;
      J_world[row * nv + d] = 1.0f;
      reference_params(dof_sol_params + d * 7, dof_sol_params + d * 7 + 2, 0, 0.0f, 0.0f, qvel[qb + d], invweight[d], true, params[0], refsafe, R[row], ar[row]);
      lo[row] = -loss;
      hi[row] = loss;
      enabled[row] = true;
    }

    // Joint Limits (bit 3 mjDSBL_LIMIT)
    for (int j = 0; j < nj; ++j) {
      if (joint_limited[j] == 0) continue;
      int d = joint_dadr[j];
      int q = joint_qadr[j];
      float margin = joint_limit_params[j * 3 + 2];
      // Lower limit
      int row0 = neq + nv + 2 * j;
      float dist0 = qpos[pb + q] - joint_limit_params[j * 3 + 0];
      if ((flags & 8) == 0 && dist0 < margin) {
        J_world[row0 * nv + d] = 1.0f;
        reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist0, margin, qvel[qb + d], invweight[d], false, params[0], refsafe, R[row0], ar[row0]);
        lo[row0] = 0.0f;
        hi[row0] = INFINITY;
        enabled[row0] = true;
      }
      // Upper limit
      int row1 = neq + nv + 2 * j + 1;
      float dist1 = joint_limit_params[j * 3 + 1] - qpos[pb + q];
      if ((flags & 8) == 0 && dist1 < margin) {
        J_world[row1 * nv + d] = -1.0f;
        reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist1, margin, -qvel[qb + d], invweight[d], false, params[0], refsafe, R[row1], ar[row1]);
        lo[row1] = 0.0f;
        hi[row1] = INFINITY;
        enabled[row1] = true;
      }
    }
  }

  // 2. Contacts (if not disabled by mjDSBL_CONSTRAINT bit 0 or mjDSBL_CONTACT bit 4)
  if ((flags & 1) == 0 && (flags & 16) == 0) {
    for (int c = 0; c < nc; ++c) {
      int cdim = contact_condim[c];
      int cb = world * nc + c;
      int cjbase = cb * 5 * nv;
      int crbase = cb * 5 * 6;
      float mu0 = contact_friction[c * 2];
      if (cdim == 1) {
        int row = base_contact + c * 4;
        int r = crbase;
        if (contact_row_data[r] > 0.5f) {
          for (int i = 0; i < nv; ++i) J_world[row * nv + i] = contact_jacobian[cjbase + i];
          float imp = clamp(contact_row_data[r + 4], 1e-6f, 0.999999f);
          float diag_approx = max(contact_row_data[r + 5], 1e-15f);
          R[row] = max(1e-15f, (1.0f - imp) * diag_approx / imp);
          ar[row] = contact_row_data[r + 3];
          lo[row] = 0.0f;
          hi[row] = INFINITY;
          enabled[row] = true;
        }
      } else if (cdim == 3) {
        for (int k = 0; k < 4; ++k) {
          int row = base_contact + c * 4 + k;
          int subrow = k + 1;
          int r = crbase + subrow * 6;
          if (contact_row_data[r] > 0.5f) {
            for (int i = 0; i < nv; ++i) J_world[row * nv + i] = contact_jacobian[cjbase + subrow * nv + i];
            float imp = clamp(contact_row_data[r + 4], 1e-6f, 0.999999f);
            float diag_approx = max(contact_row_data[r + 5], 1e-15f);
            float normal_R = max(1e-15f, (1.0f - imp) * diag_approx / imp);
            float edge_R = normal_R * (1.0f + mu0 * mu0);
            R[row] = 2.0f * mu0 * mu0 / max(params[1], 1e-15f) * edge_R;
            ar[row] = contact_row_data[r + 3];
            lo[row] = 0.0f;
            hi[row] = INFINITY;
            enabled[row] = true;
          }
        }
      }
    }
  }

  // 3. Dense Cholesky factorization of M
  for (int i = 0; i < nv; ++i) for (int j = 0; j < nv; ++j) L[i * nv + j] = 0.0f;
  for (int i = 0; i < nv; ++i) for (int j = 0; j <= i; ++j) {
    float v = mass[mb + i * nv + j];
    for (int k = 0; k < j; ++k) v -= L[i * nv + k] * L[j * nv + k];
    if (i == j) {
      if (!(v > 1e-12f) || !isfinite(v)) { out_status[world] = 2; return; }
      L[i * nv + j] = sqrt(v);
    } else {
      L[i * nv + j] = v / L[j * nv + j];
    }
  }

  // 4. Solve M q0 = qfrc (unconstrained acceleration)
  for (int i = 0; i < nv; ++i) {
    float v = qfrc[qb + i];
    for (int k = 0; k < i; ++k) v -= L[i * nv + k] * y[k];
    y[i] = v / L[i * nv + i];
  }
  for (int i = nv - 1; i >= 0; --i) {
    float v = y[i];
    for (int k = i + 1; k < nv; ++k) v -= L[k * nv + i] * x[k];
    x[i] = v / L[i * nv + i];
  }
  for (int i = 0; i < nv; ++i) out_acc[qb + i] = x[i];

  // 5. Solve M Z_r = J_r^T for all active rows
  for (int row = 0; row < nr; ++row) if (enabled[row]) {
    for (int i = 0; i < nv; ++i) {
      float v = J_world[row * nv + i];
      for (int k = 0; k < i; ++k) v -= L[i * nv + k] * y[k];
      y[i] = v / L[i * nv + i];
    }
    for (int i = nv - 1; i >= 0; --i) {
      float v = y[i];
      for (int k = i + 1; k < nv; ++k) v -= L[k * nv + i] * x[k];
      x[i] = v / L[i * nv + i];
    }
    for (int i = 0; i < nv; ++i) Z[row * nv + i] = x[i];
  }

  // 6. Form Delassus matrix W = J M^-1 J^T
  for (int a = 0; a < nr; ++a) if (enabled[a]) {
    for (int b = 0; b < nr; ++b) if (enabled[b]) {
      float v = 0.0f;
      for (int k = 0; k < nv; ++k) v += J_world[a * nv + k] * Z[b * nv + k];
      W[a * nr + b] = v;
    }
  }

  // 7. Form RHS: ar - J q0
  for (int row = 0; row < nr; ++row) {
    if (enabled[row]) {
      float ja = 0.0f;
      for (int k = 0; k < nv; ++k) ja += J_world[row * nv + k] * out_acc[qb + k];
      rhs[row] = ar[row] - ja;
    } else {
      rhs[row] = 0.0f;
    }
  }

  // 8. Projected Gauss-Seidel on coupled W + diag(R)
  float tol = params[2];
  bool converged = false;
  float max_res = 0.0f;
  for (int it = 0; it < maxiter; ++it) {
    for (int row = 0; row < nr; ++row) if (enabled[row]) {
      float diag = max(1e-15f, W[row * nr + row] + R[row]);
      float v = rhs[row];
      for (int col = 0; col < nr; ++col) if (enabled[col] && col != row) {
        v -= W[row * nr + col] * lam[col];
      }
      lam[row] = clamp(v / diag, lo[row], hi[row]);
    }
    max_res = 0.0f;
    for (int row = 0; row < nr; ++row) if (enabled[row]) {
      float grad = -rhs[row];
      for (int col = 0; col < nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
      grad += R[row] * lam[row];
      float diag = max(1e-15f, W[row * nr + row] + R[row]);
      float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
      float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
      for (int col = 0; col < nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
      row_scale = max(1.0f, row_scale);
      max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
    }
    out_diagnostics[world * 2] = max_res;
    out_diagnostics[world * 2 + 1] = float(it + 1);
    if (max_res <= tol) { converged = true; break; }
  }

  // 9. Contact Block Refinement (if not converged and contacts exist)
  if (!converged && nc > 0) {
    for (int ref = 0; ref < 64; ++ref) {
      for (int c = 0; c < nc; ++c) {
        if (contact_condim[c] != 3) continue;
        int local_rows[4];
        int nlocal = 0;
        for (int k = 0; k < 4; ++k) {
          int row = base_contact + c * 4 + k;
          if (enabled[row]) local_rows[nlocal++] = row;
        }
        if (nlocal == 0) continue;
        thread float local_A[16], local_b[4], local_sol[4];
        for (int i = 0; i < 16; ++i) local_A[i] = 0.0f;
        for (int i = 0; i < 4; ++i) { local_b[i] = 0.0f; local_sol[i] = 0.0f; }
        for (int i = 0; i < nlocal; ++i) {
          int row = local_rows[i];
          float v = -rhs[row];
          for (int col = 0; col < nr; ++col) if (enabled[col] && (col < base_contact + c * 4 || col >= base_contact + (c + 1) * 4)) {
            v += W[row * nr + col] * lam[col];
          }
          local_b[i] = v;
          for (int j = 0; j < nlocal; ++j) {
            int other = local_rows[j];
            local_A[i * 4 + j] = W[row * nr + other] + (i == j ? R[row] : 0.0f);
          }
        }
        float b_err = solve_contact_block(local_A, local_b, nlocal, local_sol);
        if (b_err < 1e-5f) {
          for (int i = 0; i < nlocal; ++i) lam[local_rows[i]] = max(0.0f, local_sol[i]);
        }
      }
      max_res = 0.0f;
      for (int row = 0; row < nr; ++row) if (enabled[row]) {
        float grad = -rhs[row];
        for (int col = 0; col < nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      out_diagnostics[world * 2] = max_res;
      out_diagnostics[world * 2 + 1] = float(int(out_diagnostics[world * 2 + 1]) + 1);
      if (max_res <= tol) { converged = true; break; }
    }
  }

  if (!converged && max_res > tol) out_status[world] = 3;

  // 10. Reconstruct forces and acceleration
  for (int row = 0; row < nr; ++row) if (enabled[row]) {
    for (int i = 0; i < nv; ++i) out_force[qb + i] += J_world[row * nv + i] * lam[row];
  }
  for (int row = 0; row < nr; ++row) if (enabled[row]) {
    for (int i = 0; i < nv; ++i) out_acc[qb + i] += Z[row * nv + i] * lam[row];
  }

  // 11. Write joint forces
  for (int row = 0; row < base_contact; ++row) {
    out_joint_force[world * max(base_contact, 1) + row] = enabled[row] ? lam[row] : 0.0f;
  }

  // 12. Write contact forces
  for (int c = 0; c < nc; ++c) {
    int cdim = contact_condim[c];
    int ofb = (world * nc + c) * 5;
    if (cdim == 1) {
      int row = base_contact + c * 4;
      out_contact_force[ofb] = enabled[row] ? lam[row] : 0.0f;
    } else if (cdim == 3) {
      float fn = 0.0f;
      for (int k = 0; k < 4; ++k) {
        int row = base_contact + c * 4 + k;
        float v = enabled[row] ? lam[row] : 0.0f;
        out_contact_force[ofb + 1 + k] = v;
        fn += v;
      }
      out_contact_force[ofb] = fn;
    }
  }

  // 13. Write debug matrices and vectors if workspace_debug is provided
  if (workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * (nr * nr + 4 * nr);
    for (int a = 0; a < nr; ++a) {
      for (int b = 0; b < nr; ++b) {
        dbg[a * nr + b] = W[a * nr + b];
      }
      dbg[nr * nr + a] = R[a];
      dbg[nr * nr + nr + a] = ar[a];
      dbg[nr * nr + 2 * nr + a] = rhs[a];
      dbg[nr * nr + 3 * nr + a] = enabled[a] ? lam[a] : 0.0f;
    }
  }
}
