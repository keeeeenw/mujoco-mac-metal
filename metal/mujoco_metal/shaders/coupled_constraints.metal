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

inline float solve_contact_block_iterative(
    thread const float* A, thread const float* b, int n,
    thread float* solution) {
  thread float x[10], z[10], next[10];
  float lipschitz = 1e-15f;
  for (int i = 0; i < 10; ++i) x[i] = z[i] = next[i] = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = z[i] = max(0.0f, solution[i]);
    float row_sum = 0.0f;
    for (int j = 0; j < n; ++j) row_sum += abs(A[i * 10 + j]);
    lipschitz = max(lipschitz, row_sum);
  }
  float momentum = 1.0f;
  float residual = INFINITY;
  for (int iteration = 0; iteration < 1024; ++iteration) {
    for (int i = 0; i < n; ++i) {
      float gradient = b[i];
      for (int j = 0; j < n; ++j) gradient += A[i * 10 + j] * z[j];
      next[i] = max(0.0f, z[i] - gradient / lipschitz);
    }
    float next_momentum = 0.5f * (1.0f + sqrt(1.0f + 4.0f * momentum * momentum));
    float beta = (momentum - 1.0f) / next_momentum;
    for (int i = 0; i < n; ++i) {
      float previous = x[i];
      x[i] = next[i];
      z[i] = next[i] + beta * (next[i] - previous);
    }
    momentum = next_momentum;

    if ((iteration & 15) == 15 || iteration == 1023) {
      residual = 0.0f;
      for (int i = 0; i < n; ++i) {
        float gradient = b[i];
        float row_scale = abs(b[i]);
        for (int j = 0; j < n; ++j) {
          float value = A[i * 10 + j] * x[j];
          gradient += value;
          row_scale += abs(value);
        }
        float projected = max(0.0f, x[i] - gradient / max(A[i * 10 + i], 1e-15f));
        residual = max(residual, abs(projected - x[i]) * A[i * 10 + i] / max(1.0f, row_scale));
      }
      if (residual <= 5e-7f) break;
    }
  }
  for (int i = 0; i < n; ++i) solution[i] = x[i];
  return residual;
}

// Geometric contact detection and Jacobian generation for all 9 primitive pairs
kernel void contact_normal(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_size [[buffer(2)]],
    device const int* geom_type [[buffer(3)]],
    device const int* geom_bodyid [[buffer(4)]],
    device const float* body_pos [[buffer(5)]],
    device const float* body_quat [[buffer(6)]],
    device const float* anchors [[buffer(7)]],
    device const float* axes [[buffer(8)]],
    device const float* qvel [[buffer(9)]],
    device const int* body_parentid [[buffer(10)]],
    device const int* body_jntadr [[buffer(11)]],
    device const int* body_jntnum [[buffer(12)]],
    device const int* jnt_type [[buffer(13)]],
    device const int* jnt_dofadr [[buffer(14)]],
    device const float* body_invweight0 [[buffer(15)]],
    device const int* pair_geoms [[buffer(16)]],
    device const float* pair_margin_gap [[buffer(17)]],
    device const float* pair_solref [[buffer(18)]],
    device const float* pair_solimp [[buffer(19)]],
    device const int* pair_condim [[buffer(20)]],
    device const float* pair_friction [[buffer(21)]],
    device const float* pair_solreffriction [[buffer(22)]],
    device const int* pair_contact_offset [[buffer(23)]],
    device float* row_data [[buffer(24)]],
    device float* frame [[buffer(25)]],
    device float* jacobian [[buffer(26)]],
    constant int* dims [[buffer(27)]],
    device const float* geom_rbound [[buffer(28)]],
    device const float* mesh_hull [[buffer(29)]],
    device const int* mesh_hull_info [[buffer(30)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0];
  int npairs = dims[1];
  int ncontacts_max = dims[2];
  int batch = dims[3];
  int nbody = dims[4];
  int njnt = dims[5];
  int ngeom = dims[6];

  int world = int(tid) / max(npairs, 1);
  int pair_idx = int(tid) % max(npairs, 1);
  if (uint(world) >= uint(batch) || pair_idx >= npairs) return;

  int a = pair_geoms[pair_idx * 2 + 0];
  int b = pair_geoms[pair_idx * 2 + 1];
  int ta = geom_type[a];
  int tb = geom_type[b];
  float3 sza = float3(geom_size[a * 3], geom_size[a * 3 + 1], geom_size[a * 3 + 2]);
  float3 szb = float3(geom_size[b * 3], geom_size[b * 3 + 1], geom_size[b * 3 + 2]);
  float rba = geom_rbound[a];
  float rbb = geom_rbound[b];
  int disable_multiccd = dims[8];

  int go = world * ngeom;
  float3 pa = float3(geom_pos[(go + a) * 3], geom_pos[(go + a) * 3 + 1], geom_pos[(go + a) * 3 + 2]);
  float3 pb = float3(geom_pos[(go + b) * 3], geom_pos[(go + b) * 3 + 1], geom_pos[(go + b) * 3 + 2]);
  float4 qa = float4(geom_quat[(go + a) * 4], geom_quat[(go + a) * 4 + 1], geom_quat[(go + a) * 4 + 2], geom_quat[(go + a) * 4 + 3]);
  float4 qb = float4(geom_quat[(go + b) * 4], geom_quat[(go + b) * 4 + 1], geom_quat[(go + b) * 4 + 2], geom_quat[(go + b) * 4 + 3]);

  float m = pair_margin_gap[pair_idx * 2 + 0];
  float g = pair_margin_gap[pair_idx * 2 + 1];
  int dim = pair_condim[pair_idx];
  float cone = float(dims[7]);

  int offset = pair_contact_offset[pair_idx];
  int max_con = pair_contact_offset[pair_idx + 1] - offset;

  // R08/F1 broadphase-fed pruning: the conservative sphere test mirrors
  // broadphase_mask exactly (planes always overlap; otherwise skip when
  // center distance exceeds rbound sum + margin + gap + 1e-6). Skipped
  // pairs stamp a finite sentinel into their slots (active flag stays 0)
  // so tests can prove the skip path executed instead of a silent zero.
  // dims[9] toggles pruning for on/off physics-equivalence qualification.
  if (dims[9] != 0 && ta != 0 && tb != 0) {
    float3 delta_pb = pb - pa;
    float prune_R = rba + rbb + m + g;
    if (length(delta_pb) > prune_R + 1e-6f) {
      for (int k = 0; k < max_con; ++k) {
        int slot = offset + k;
        if (slot >= ncontacts_max) break;
        int fb = (world * ncontacts_max + slot) * 12;
        frame[fb + 0] = 1234.0f;
      }
      return;
    }
  }

  // Run collision algorithm
  ContactGeom con[16];
  for (int k = 0; k < 16; ++k) {
    con[k].dist = 1e30f;
    con[k].pos = float3(0.0f);
    con[k].normal = float3(0.0f);
    con[k].t1 = float3(0.0f);
    con[k].t2 = float3(0.0f);
  }
  int ncon = collide_pair(ta, pa, qa, sza, rba, tb, pb, qb, szb, rbb,
                            m, g, disable_multiccd, con, a, b,
                            mesh_hull, mesh_hull_info, max_con);
  ncon = min(ncon, max_con);

  int ba = geom_bodyid[a];
  int bb = geom_bodyid[b];
  int bo = world * nbody;
  int jo = world * njnt;
  float diag_approx = body_invweight0[ba * 2] + body_invweight0[bb * 2];

  float d0 = pair_solimp[pair_idx * 5 + 0];
  float d1 = pair_solimp[pair_idx * 5 + 1];
  float width = pair_solimp[pair_idx * 5 + 2];
  float mid = pair_solimp[pair_idx * 5 + 3];
  float power = pair_solimp[pair_idx * 5 + 4];

  float r0 = pair_solref[pair_idx * 2 + 0];
  float r1 = pair_solref[pair_idx * 2 + 1];
  float K, B;
  if (r0 > 0.0f) K = 1.0f / max(1e-15f, d1 * d1 * r0 * r0 * r1 * r1);
  else K = -r0 / max(1e-15f, d1 * d1);
  if (r1 > 0.0f) B = 2.0f / max(1e-15f, d1 * r0);
  else B = -r1 / max(1e-15f, d1);

  // For each detected contact:
  for (int k = 0; k < ncon; ++k) {
    int slot = offset + k;
    if (slot >= ncontacts_max) break;
    int out = world * ncontacts_max + slot;
    int jbase = out * 6 * nv;
    int rb = out * 6 * 6;
    int fb = out * 12;

    float d = con[k].dist;
    float3 n = con[k].normal;
    float3 t1 = con[k].t1;
    float3 t2 = con[k].t2;
    float3 point = con[k].pos;

    // Zero out buffers
    for (int i = 0; i < 6 * 6; ++i) row_data[rb + i] = 0.0f;
    for (int i = 0; i < 6 * nv; ++i) jacobian[jbase + i] = 0.0f;
    for (int i = 0; i < 12; ++i) frame[fb + i] = 0.0f;

    // Compute Kinematics Jacobian
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
              float3 angular = float3(0.0f);
              if (typ == 2) {
                col = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
              } else if (typ == 3) {
                float3 axis = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
                angular = axis;
                col = cross(axis, rel - float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]));
              } else if (typ == 0 || typ == 1) {
              if (typ == 0 && q < 3) {
                col = float3(q == 0, q == 1, q == 2);
              } else {
                int qrot = typ == 0 ? q - 3 : q;
                float4 bq = float4(body_quat[(bo + body) * 4], body_quat[(bo + body) * 4 + 1], body_quat[(bo + body) * 4 + 2], body_quat[(bo + body) * 4 + 3]);
                float3 axis = rotate_q(bq, float3(qrot == 0, qrot == 1, qrot == 2));
                angular = axis;
                float3 pivot = typ == 0 ? float3(body_pos[(bo + body) * 3], body_pos[(bo + body) * 3 + 1], body_pos[(bo + body) * 3 + 2])
                                        : float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]);
                col = cross(axis, rel - pivot);
              }
            }
              jacobian[jbase + dof] += sign * dot(n, col);
              jacobian[jbase + nv + dof] += sign * dot(t1, col);
              jacobian[jbase + 2 * nv + dof] += sign * dot(t2, col);
              jacobian[jbase + 3 * nv + dof] += sign * dot(n, angular);
              jacobian[jbase + 4 * nv + dof] += sign * dot(t1, angular);
              jacobian[jbase + 5 * nv + dof] += sign * dot(t2, angular);
          }
        }
        body = body_parentid[body];
      }
    }

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

    float friction_B = B;
    if (cone > 0.5f &&
        (pair_solreffriction[pair_idx * 2] != 0.0f || pair_solreffriction[pair_idx * 2 + 1] != 0.0f)) {
      float fr0 = pair_solreffriction[pair_idx * 2];
      float fr1 = pair_solreffriction[pair_idx * 2 + 1];
      friction_B = fr1 > 0.0f ? 2.0f / max(1e-15f, d1 * fr0) : -fr1 / max(1e-15f, d1);
    }

    for (int row = 0; row < 6; ++row) {
      float vel = 0.0f;
      if (dim == 3 && cone < 0.5f && row > 0 && row < 5) {
        int axis = (row - 1) / 2 + 1;
        float sign = ((row - 1) & 1) == 0 ? 1.0f : -1.0f;
        float mu = pair_friction[pair_idx * 5 + axis - 1];
        for (int i = 0; i < nv; ++i) {
          float edge_jac = jacobian[jbase + i] + sign * mu * jacobian[jbase + axis * nv + i];
          vel += edge_jac * qvel[world * nv + i];
        }
      } else {
        for (int i = 0; i < nv; ++i) vel += jacobian[jbase + row * nv + i] * qvel[world * nv + i];
      }
      int r = rb + row * 6;
      row_data[r] = d < m ? 1.0f : 0.0f;
      row_data[r + 1] = d;
      row_data[r + 2] = vel;
      row_data[r + 3] = (row == 0 || (dim == 3 && cone < 0.5f))
          ? -B * vel - K * impedance * (d - m)
          : -((cone > 0.5f) ? friction_B : B) * vel;
      row_data[r + 4] = impedance;
      row_data[r + 5] = diag_approx;
    }

    frame[fb + 0] = n.x; frame[fb + 1] = n.y; frame[fb + 2] = n.z;
    frame[fb + 3] = t1.x; frame[fb + 4] = t1.y; frame[fb + 5] = t1.z;
    frame[fb + 6] = t2.x; frame[fb + 7] = t2.y; frame[fb + 8] = t2.z;
    frame[fb + 9] = point.x; frame[fb + 10] = point.y; frame[fb + 11] = point.z;
  }

  // Clear unpopulated slots for this pair
  for (int k = ncon; k < max_con; ++k) {
    int slot = offset + k;
    if (slot >= ncontacts_max) break;
    int out = world * ncontacts_max + slot;
    int jbase = out * 6 * nv;
    int rb = out * 6 * 6;
    int fb = out * 12;
    for (int i = 0; i < 6 * 6; ++i) row_data[rb + i] = 0.0f;
    for (int i = 0; i < 6 * nv; ++i) jacobian[jbase + i] = 0.0f;
    for (int i = 0; i < 12; ++i) frame[fb + i] = 0.0f;
  }
}

// Coupled projected constraint solve
inline void project_lorentz(thread float* x, int dim) {
  float norm = 0.0f;
  for (int i = 1; i < dim; ++i) norm += x[i] * x[i];
  norm = sqrt(norm);
  if (norm <= x[0]) return;
  if (norm <= -x[0]) {
    for (int i = 0; i < dim; ++i) x[i] = 0.0f;
    return;
  }
  float head = 0.5f * (norm + x[0]);
  float scale = head / max(norm, 1e-20f);
  x[0] = head;
  for (int i = 1; i < dim; ++i) x[i] *= scale;
}

inline float elliptic_projected_residual(thread const float* A,
    thread const float* g, thread const float* friction, int dim,
    thread const float* force) {
  thread float scale[6], H[36], linear[6], y[6], projected[6];
  for (int i = 0; i < 6; ++i) {
    scale[i] = i == 0 ? 1.0f : max(friction[i - 1], 0.0f);
    y[i] = projected[i] = 0.0f;
    linear[i] = 0.0f;
  }
  for (int i = 0; i < 36; ++i) H[i] = 0.0f;
  y[0] = force[0];
  for (int i = 1; i < dim; ++i) {
    y[i] = scale[i] > 1e-12f ? force[i] / scale[i] : 0.0f;
  }
  for (int i = 0; i < dim; ++i) {
    linear[i] = scale[i] * g[i];
    for (int j = 0; j < dim; ++j) H[i * 6 + j] = scale[i] * A[i * 6 + j] * scale[j];
  }
  float lipschitz = 1e-15f;
  for (int i = 0; i < dim; ++i) {
    float row_sum = 0.0f;
    for (int j = 0; j < dim; ++j) row_sum += abs(H[i * 6 + j]);
    lipschitz = max(lipschitz, row_sum);
  }
  // Measure stationarity at the retained global force. A convergence check
  // must not run a second local optimizer whose result is thrown away, since
  // coupled contacts can change this block's optimum.
  float residual = 0.0f, scale_ref = 1.0f;
  for (int i = 0; i < dim; ++i) {
    float gradient = linear[i];
    for (int j = 0; j < dim; ++j) gradient += H[i * 6 + j] * y[j];
    projected[i] = y[i] - gradient / lipschitz;
    scale_ref += abs(linear[i]);
    for (int j = 0; j < dim; ++j) scale_ref += abs(H[i * 6 + j] * y[j]);
  }
  project_lorentz(projected, dim);
  for (int i = 0; i < dim; ++i)
    residual = max(residual, abs(y[i] - projected[i]) * lipschitz / scale_ref);
  return residual;
}

// Cone-aware row residual for re-certification loops (R04): elliptic
// member rows use the Lorentz certificate over their block; all other
// rows use the box-projected row residual with identical scaling.
inline float cert_row_residual(int row,
    thread const float* W, thread const float* R, thread const float* rhs,
    thread const float* ar, thread const float* lam,
    thread const float* lo, thread const float* hi,
    thread const bool* enabled, int nr, int total_nr,
    int eblock, int edim, thread const float* emu) {
  if (eblock >= 0) {
    return 0.0f;  // aggregated per block by the caller (see below)
  }
  float grad = -rhs[row];
  for (int col = 0; col < total_nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
  grad += R[row] * lam[row];
  float diag = max(1e-15f, W[row * nr + row] + R[row]);
  float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
  float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
  for (int col = 0; col < total_nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
  row_scale = max(1.0f, row_scale);
  return abs(proj - lam[row]) * diag / row_scale;
}

// Per-block cone residual (caller aggregates the max over blocks).
inline float cert_block_residual(int start, int dim, thread const float* mu,
    thread const float* W, thread const float* R, thread const float* rhs,
    thread const float* lam, thread const bool* enabled,
    int nr, int total_nr) {
  thread float A[36], g[6], frc[6], mulo[5];
  for (int i = 0; i < 36; ++i) A[i] = 0.0f;
  for (int i = 0; i < 6; ++i) { g[i] = 0.0f; frc[i] = 0.0f; }
  for (int i = 0; i < 5; ++i) mulo[i] = mu[i];
  for (int i = 0; i < dim; ++i) {
    int ri = start + i;
    frc[i] = lam[ri];
    g[i] = -rhs[ri];
    for (int col = 0; col < total_nr; ++col)
      if (enabled[col] && (col < start || col >= start + dim))
        g[i] += W[ri * nr + col] * lam[col];
    for (int j = 0; j < dim; ++j) {
      int cj = start + j;
      A[i * 6 + j] = W[ri * nr + cj] + (i == j ? R[ri] : 0.0f);
    }
  }
  return elliptic_projected_residual(A, g, mulo, dim, frc);
}

// ---- No-slip helpers (pinned solNoSlip/mju_QCQP family, R04) ----
// QCQP in 2 dimensions: min 0.5*x'*A*x + x'*b s.t. sum (xi/di)^2 <= r^2.
// Returns 0 if unconstrained, 1 if constrained.
inline int nsl_qcqp2(thread float* res, thread const float* Ain, thread const float* bin,
    thread const float* d, float r) {
  float b1 = bin[0]*d[0];
  float b2 = bin[1]*d[1];
  float A11 = Ain[0]*d[0]*d[0];
  float A22 = Ain[3]*d[1]*d[1];
  float A12 = Ain[1]*d[0]*d[1];
  float la = 0.0f;
  float v1 = 0.0f, v2 = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    float det = (A11+la)*(A22+la) - A12*A12;
    if (det < 1e-10f) {
      res[0] = 0.0f;
      res[1] = 0.0f;
      return 0;
    }
    float detinv = 1.0f/det;
    float P11 = (A22+la)*detinv;
    float P22 = (A11+la)*detinv;
    float P12 = -A12*detinv;
    v1 = -P11*b1 - P12*b2;
    v2 = -P12*b1 - P22*b2;
    float val = v1*v1 + v2*v2 - r*r;
    if (val < 1e-10f) break;
    float deriv = -2.0f*(P11*v1*v1 + 2.0f*P12*v1*v2 + P22*v2*v2);
    float delta = -val/deriv;
    if (delta < 1e-10f) break;
    la += delta;
  }
  res[0] = v1*d[0];
  res[1] = v2*d[1];
  return (la != 0.0f);
}

// QCQP in 3 dimensions (same contract).
inline int nsl_qcqp3(thread float* res, thread const float* Ain, thread const float* bin,
    thread const float* d, float r) {
  float b1 = bin[0]*d[0];
  float b2 = bin[1]*d[1];
  float b3 = bin[2]*d[2];
  float A11 = Ain[0]*d[0]*d[0];
  float A22 = Ain[4]*d[1]*d[1];
  float A33 = Ain[8]*d[2]*d[2];
  float A12 = Ain[1]*d[0]*d[1];
  float A13 = Ain[2]*d[0]*d[2];
  float A23 = Ain[5]*d[1]*d[2];
  float la = 0.0f;
  float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    float P11 = (A22+la)*(A33+la) - A23*A23;
    float P22 = (A11+la)*(A33+la) - A13*A13;
    float P33 = (A11+la)*(A22+la) - A12*A12;
    float P12 = A13*A23 - A12*(A33+la);
    float P13 = A12*A23 - A13*(A22+la);
    float P23 = A12*A13 - A23*(A11+la);
    float det = (A11+la)*P11 + A12*P12 + A13*P13;
    if (det < 1e-10f) {
      res[0] = 0.0f;
      res[1] = 0.0f;
      res[2] = 0.0f;
      return 0;
    }
    float detinv = 1.0f/det;
    P11 *= detinv;
    P22 *= detinv;
    P33 *= detinv;
    P12 *= detinv;
    P13 *= detinv;
    P23 *= detinv;
    v1 = -P11*b1 - P12*b2 - P13*b3;
    v2 = -P12*b1 - P22*b2 - P23*b3;
    v3 = -P13*b1 - P23*b2 - P33*b3;
    float val = v1*v1 + v2*v2 + v3*v3 - r*r;
    if (val < 1e-10f) break;
    float deriv = -2.0f*(P11*v1*v1 + P22*v2*v2 + P33*v3*v3)
            -4.0f*(P12*v1*v2 + P13*v1*v3 + P23*v2*v3);
    float delta = -val/deriv;
    if (delta < 1e-10f) break;
    la += delta;
  }
  res[0] = v1*d[0];
  res[1] = v2*d[1];
  res[2] = v3*d[2];
  return (la != 0.0f);
}

// QCQP in n<=5 dimensions via Cholesky Newton (same contract). The pinned
// mju_cholFactor/mju_cholSolve steps are inlined (rank threshold 1e-10).
inline int nsl_qcqpn(thread float* res, thread const float* Ain, thread const float* bin,
    thread const float* d, float r, int n) {
  thread float A[25], Ala[25], b[5], tmp[5];
  for (int i = 0; i < 25; ++i) A[i] = 0.0f;
  for (int i = 0; i < 5; ++i) { b[i] = 0.0f; tmp[i] = 0.0f; res[i] = 0.0f; }
  if (n > 5) return 0;
  for (int i = 0; i < n; ++i) {
    b[i] = bin[i] * d[i];
    for (int j = 0; j < n; ++j) A[j+i*n] = Ain[j+i*n] * d[i] * d[j];
  }
  float la = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    for (int i = 0; i < n*n; ++i) Ala[i] = A[i];
    for (int i = 0; i < n; ++i) Ala[i*(n+1)] += la;
    int rank = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        float v = Ala[i*n+j];
        for (int k = 0; k < j; ++k) v -= Ala[i*n+k] * Ala[j*n+k];
        if (i == j) {
          if (!(v > 1e-10f)) break;
          Ala[i*n+j] = sqrt(v);
          rank++;
        } else {
          Ala[i*n+j] = v / Ala[j*n+j];
        }
      }
      if (rank <= i) break;
    }
    if (rank < n) {
      for (int i = 0; i < n; ++i) res[i] = 0.0f;
      return 0;
    }
    // res = -Ala \ b.
    for (int i = 0; i < n; ++i) {
      float v = b[i];
      for (int k = 0; k < i; ++k) v -= Ala[i*n+k] * tmp[k];
      tmp[i] = v / Ala[i*n+i];
    }
    for (int i = n - 1; i >= 0; --i) {
      float v = tmp[i];
      for (int k = i + 1; k < n; ++k) v -= Ala[k*n+i] * res[k];
      res[i] = v / Ala[i*n+i];
    }
    for (int i = 0; i < n; ++i) res[i] = -res[i];
    float val = 0.0f;
    for (int i = 0; i < n; ++i) val += res[i]*res[i];
    val -= r*r;
    if (val < 1e-10f) break;
    // deriv = -2 * res' * Ala^-1 * res.
    for (int i = 0; i < n; ++i) {
      float v = res[i];
      for (int k = 0; k < i; ++k) v -= Ala[i*n+k] * tmp[k];
      tmp[i] = v / Ala[i*n+i];
    }
    for (int i = n - 1; i >= 0; --i) {
      float v = tmp[i];
      for (int k = i + 1; k < n; ++k) v -= Ala[k*n+i] * tmp[k];
      tmp[i] = v / Ala[i*n+i];
    }
    float deriv = 0.0f;
    for (int i = 0; i < n; ++i) deriv += res[i] * tmp[i];
    deriv *= -2.0f;
    float delta = -val/deriv;
    if (delta < 1e-10f) break;
    la += delta;
  }
  for (int i = 0; i < n; ++i) res[i] = res[i] * d[i];
  return (la != 0.0f);
}

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
  int ncontacts_max = dims[4];
  int batch = dims[5];
  int flags = dims[6];
  bool refsafe = dims[7] != 0;
  int maxiter = dims[8];
  int nr = dims[9];
  int cone_type = dims[10];
  int n_eq_rows = nr > 0 ? dims[14] : 0;
  if (n_eq_rows < 0) n_eq_rows = 0;
  if (n_eq_rows > nr) n_eq_rows = nr;
  int ten_base = nr > 0 ? dims[16] : 0;
  if (ten_base < 0) ten_base = 0;
  if (ten_base > nr) ten_base = nr;
  if (world >= uint(batch)) return;

  int mb = world * nv * nv;  int qb = world * nv;
  int pb = world * nq;
  out_status[world] = 0;
  out_diagnostics[world * 10] = 0.0f;
  out_diagnostics[world * 10 + 1] = 0.0f;
  for (int k = 0; k < 8; ++k) out_diagnostics[world * 10 + 2 + k] = 0.0f;
  for (int i = 0; i < nv; ++i) { out_force[qb + i] = 0.0f; out_acc[qb + i] = 0.0f; }
  if (workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * (nr * nr + 7 * nr);
    for (int i = 0; i < nr * nr; ++i) dbg[i] = 0.0f;
    for (int r = n_eq_rows; r < nr; ++r) {
      // Tendon-owned rows are preassembled by tendon_constraint_rows; keep them.
      if (dbg[nr * nr + 6 * nr + r] > 0.5f) continue;
      dbg[nr * nr + r] = 0.0f;
      dbg[nr * nr + nr + r] = 0.0f;
    }
    // Zero the residual vector region only; the retained-multiplier region
    // [3*nr, 4*nr) persists across steps for warm starts (cleared by reset,
    // restore-clear and clear_warmstart on the host).
    for (int i = 0; i < nr; ++i) dbg[nr * nr + 2 * nr + i] = 0.0f;
  }
  if (nv == 0) return;
  if (nv > 32 || nr > 96) { out_status[world] = 2; return; }

  device float* J_world = workspace_J + world * nr * nv;
  device float* dbg_pre = workspace_debug + world * (nr * nr + 7 * nr);
  for (int r = n_eq_rows; r < nr; ++r) {
    // Tendon-owned rows are preassembled by tendon_constraint_rows; keep them.
    if (nr > 0 && dbg_pre[nr * nr + 6 * nr + r] > 0.5f) continue;
    for (int i = 0; i < nv; ++i) J_world[r * nv + i] = 0.0f;
  }

  thread float L[32 * 32];
  thread float Z[96 * 32];
  thread float W[96 * 96];
  thread float R[96], ar[96], lo[96], hi[96], lam[96], rhs[96];
  thread bool enabled[96];
  thread float y[32], x[32];
  for (int i = 0; i < nr * nr; ++i) W[i] = 0.0f;
  // Warm start: seed multipliers from the retained solution unless the
  // WARMSTART disable bit (512) is set. Bad/foreign warm vectors are
  // rejected by the cost check after Delassus assembly below.
  bool warm_ok = (nr > 0) && ((flags & 512) == 0);
  for (int i = 0; i < nr; ++i) {
    R[i] = 0.0f; ar[i] = 0.0f; lo[i] = 0.0f; hi[i] = 0.0f; lam[i] = 0.0f; enabled[i] = false;
  }
  if (warm_ok && workspace_debug) {
    device float* dbg0 = workspace_debug + world * (nr * nr + 7 * nr);
    for (int i = 0; i < nr; ++i) lam[i] = dbg0[nr * nr + 3 * nr + i];
  }
  // Equality rows [0, n_eq_rows) are preassembled on Metal by equality_assembly
  // (joint/connect/weld with version-pinned residuals, Jacobians, Jdot, impedance).
  // Treat them as bilateral; inactive rows were written as zero J/R/ar and
  // naturally yield zero multipliers without affecting coupled rows.
  if (workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * (nr * nr + 7 * nr);
    for (int r = 0; r < n_eq_rows && r < nr; ++r) {
      if (dbg[nr * nr + 6 * nr + r] > 0.5f) {
        // Tendon-owned equality row (milestone 008): fully preassembled by
        // tendon_constraint_rows, including bilateral bounds.
        R[r] = dbg[nr * nr + r];
        ar[r] = dbg[nr * nr + nr + r];
        lo[r] = dbg[nr * nr + 4 * nr + r];
        hi[r] = dbg[nr * nr + 5 * nr + r];
        enabled[r] = true;
        continue;
      }
      R[r] = dbg[nr * nr + r];
      ar[r] = dbg[nr * nr + nr + r];
      lo[r] = -INFINITY;
      hi[r] = INFINITY;
      enabled[r] = true;
    }
    // Tendon limit/friction rows in the reserved ten region.
    for (int r = ten_base; r < nr; ++r) {
      if (dbg[nr * nr + 6 * nr + r] > 0.5f) {
        R[r] = dbg[nr * nr + r];
        ar[r] = dbg[nr * nr + nr + r];
        lo[r] = dbg[nr * nr + 4 * nr + r];
        hi[r] = dbg[nr * nr + 5 * nr + r];
        enabled[r] = true;
      }
    }
  }

  int base_contact = n_eq_rows + nv + 2 * nj;
  if (nr > 0) {
    int ten_rows = dims[17];
    if (ten_rows < 0) ten_rows = 0;
    if (ten_rows > nr) ten_rows = nr;
    base_contact += ten_rows;
  }

  // 1. Joint constraints (if constraint disable flag not set: bit 0 mjDSBL_CONSTRAINT)
  if ((flags & 1) == 0) {
    // Frictionloss (bit 2 mjDSBL_FRICTIONLOSS)
    for (int d = 0; d < nv; ++d) {
      int row = n_eq_rows + d;
      float loss = frictionloss[d];
      if ((flags & 4) != 0 || loss <= 0.0f) continue;
      J_world[row * nv + d] = 1.0f;
      reference_params(dof_sol_params + d * 7, dof_sol_params + d * 7 + 2, 0, 0.0f, 0.0f, qvel[qb + d], invweight[d], true, params[0], refsafe, R[row], ar[row]);
      lo[row] = -loss;
      hi[row] = loss;
      enabled[row] = true;
    }

    // Joint Limits (bit 3 mjDSBL_LIMIT). joint_limited packs bit0=limited,
    // bits[2:1]=joint type (milestone 009 ball support).
    for (int j = 0; j < nj; ++j) {
      int limpack = joint_limited[j];
      if ((limpack & 1) == 0) continue;
      int d = joint_dadr[j];
      int q = joint_qadr[j];
      float margin = joint_limit_params[j * 3 + 2];
      if (((limpack >> 1) & 3) == 1) {
        // Ball limit (pinned mj_instantiateLimit): single row on the rotation
        // angle with Jacobian -axis over the 3 DOFs. Uses the first reserved
        // row of the pair; the second stays disabled.
        float4 quat = float4(qpos[pb+q], qpos[pb+q+1], qpos[pb+q+2], qpos[pb+q+3]);
        float nq4 = length(quat);
        quat = nq4 > 1e-30f ? quat / nq4 : float4(1, 0, 0, 0);
        float3 vv = quat.yzw;
        float s = length(vv);
        float speed = 2.0f * atan2(s, quat.x);
        if (speed > 3.14159265358979f) speed -= 2.0f * 3.14159265358979f;
        float3 aa = s > 1e-30f ? vv * (speed / s) : float3(0.0f);
        float value = length(aa);
        // Pinned mju_normalize3 normalizes angleAxis in place: the Jacobian
        // uses the UNIT axis, value is the pre-normalized angle.
        float3 naxis = value > 1e-30f ? aa / value : float3(0.0f);
        float rmax = max(joint_limit_params[j * 3], joint_limit_params[j * 3 + 1]);
        float dist = rmax - value;
        int rowb = n_eq_rows + nv + 2 * j;
        if ((flags & 8) == 0 && dist < margin) {
          for (int k = 0; k < 3; ++k) {
            int dof = d + k;
            if (dof >= 0 && dof < nv) J_world[rowb * nv + dof] = -naxis[k];
          }
          float vel = 0.0f;
          for (int k = 0; k < 3; ++k) {
            int dof = d + k;
            if (dof >= 0 && dof < nv) vel += -naxis[k] * qvel[qb + dof];
          }
          reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist, margin, vel, invweight[d], false, params[0], refsafe, R[rowb], ar[rowb]);
          lo[rowb] = 0.0f;
          hi[rowb] = INFINITY;
          enabled[rowb] = true;
        }
        continue;
      }
      // Lower limit
      int row0 = n_eq_rows + nv + 2 * j;
      float dist0 = qpos[pb + q] - joint_limit_params[j * 3 + 0];
      if ((flags & 8) == 0 && dist0 < margin) {
        J_world[row0 * nv + d] = 1.0f;
        reference_params(joint_sol_params + j * 7, joint_sol_params + j * 7 + 2, 0, dist0, margin, qvel[qb + d], invweight[d], false, params[0], refsafe, R[row0], ar[row0]);
        lo[row0] = 0.0f;
        hi[row0] = INFINITY;
        enabled[row0] = true;
      }
      // Upper limit
      int row1 = n_eq_rows + nv + 2 * j + 1;
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
  thread int contact_block_start[24];
  thread int contact_block_size[24];
  thread int contact_block_count = 0;
  thread int elliptic_start[24];
  thread int elliptic_dim[24];
  thread float elliptic_friction[24 * 5];
  thread int elliptic_count = 0;
  thread bool elliptic_member[96];
  for (int i = 0; i < 96; ++i) elliptic_member[i] = false;

  if ((flags & 1) == 0 && (flags & 16) == 0) {
    for (int s = 0; s < ncontacts_max; ++s) {
      int cdim = contact_condim[s * 3 + 0];
      int row_offset = contact_condim[s * 3 + 1];
      int row_start = base_contact + row_offset;
      int cb = world * ncontacts_max + s;
      int cjbase = cb * 6 * nv;
      int crbase = cb * 6 * 6;

      if (cdim == 1) {
        int r = crbase;
        if (contact_row_data[r] > 0.5f) {
          if (row_start + 1 > nr) { out_status[world] = 2; return; }
          int row = row_start;
          for (int i = 0; i < nv; ++i) J_world[row * nv + i] = contact_jacobian[cjbase + i];
          float imp = clamp(contact_row_data[r + 4], 1e-6f, 0.999999f);
          float diag_approx = max(contact_row_data[r + 5], 1e-15f);
          R[row] = max(1e-15f, (1.0f - imp) * diag_approx / imp);
          ar[row] = contact_row_data[r + 3];
          lo[row] = 0.0f;
          hi[row] = INFINITY;
          enabled[row] = true;
        }
      } else if (contact_row_data[crbase] > 0.5f) {
        int edge_count = 2 * (cdim - 1);
        int block_rows = cone_type == 0 ? edge_count : cdim;
        if (row_start + block_rows > nr) { out_status[world] = 2; return; }
        float imp = clamp(contact_row_data[crbase + 4], 1e-6f, 0.999999f);
        float diag_approx = max(contact_row_data[crbase + 5], 1e-15f);
        float normal_R = max(1e-15f, (1.0f - imp) * diag_approx / imp);
        if (cone_type == 1) {
          if (elliptic_count < 24) {
            elliptic_start[elliptic_count] = row_start;
            elliptic_dim[elliptic_count] = cdim;
            for (int k = 0; k < 5; ++k) elliptic_friction[elliptic_count * 5 + k] = contact_friction[s * 5 + k];
            ++elliptic_count;
          }
          for (int k = 0; k < cdim; ++k) {
            int row = row_start + k;
            elliptic_member[row] = true;
            int r = crbase + k * 6;
            for (int i = 0; i < nv; ++i) J_world[row * nv + i] = contact_jacobian[cjbase + k * nv + i];
            if (k == 0) {
              R[row] = normal_R;
              lo[row] = 0.0f;
              hi[row] = INFINITY;
            } else {
              float mu = contact_friction[s * 5 + k - 1];
              float mu0 = contact_friction[s * 5];
              float tangent_R = normal_R / max(params[1], 1e-15f);
              R[row] = tangent_R * mu0 * mu0 / max(mu * mu, 1e-12f);
              lo[row] = -INFINITY;
              hi[row] = INFINITY;
            }
            ar[row] = contact_row_data[r + 3];
            enabled[row] = true;
          }
        } else if (cdim == 3) {
          if (contact_block_count < 24) {
            contact_block_start[contact_block_count] = row_start;
            contact_block_size[contact_block_count++] = 4;
          }
          float mu0 = contact_friction[s * 5];
          for (int k = 0; k < 4; ++k) {
            int row = row_start + k;
            int r = crbase + (k + 1) * 6;
            int axis = k / 2 + 1;
            float sign = (k & 1) == 0 ? 1.0f : -1.0f;
            float mu = contact_friction[s * 5 + axis - 1];
            for (int i = 0; i < nv; ++i) {
              J_world[row * nv + i] = contact_jacobian[cjbase + i]
                  + sign * mu * contact_jacobian[cjbase + axis * nv + i];
            }
            float edge_imp = clamp(contact_row_data[r + 4], 1e-6f, 0.999999f);
            float edge_diag = max(contact_row_data[r + 5], 1e-15f);
            float edge_R = max(1e-15f, (1.0f - edge_imp) * edge_diag / edge_imp) * (1.0f + mu0 * mu0);
            R[row] = 2.0f * mu0 * mu0 / max(params[1], 1e-15f) * edge_R;
            ar[row] = contact_row_data[r + 3];
            lo[row] = 0.0f;
            hi[row] = INFINITY;
            enabled[row] = true;
          }
        } else {
          if (cdim > 1 && contact_block_count < 24) {
            contact_block_start[contact_block_count] = row_start;
            contact_block_size[contact_block_count++] = 2 * (cdim - 1);
          }
          float mu0 = contact_friction[s * 5];
          float mu_master = mu0 / sqrt(max(params[1], 1e-15f));
          // MuJoCo 3.10 first builds the first pyramid edge diagonal as
          // Rnormal * (1 + mu0^2), then derives its common Rpy from that row.
          // Every edge shares this regularizer; each edge Jacobian carries its
          // own friction coefficient.
          float pyramid_R = max(1e-15f, 2.0f * mu_master * mu_master * normal_R * (1.0f + mu0 * mu0));
          for (int axis = 0; axis < cdim - 1; ++axis) {
            float mu = contact_friction[s * 5 + axis];
            for (int side = 0; side < 2; ++side) {
              int k = axis * 2 + side;
              int row = row_start + k;
              float sign = side == 0 ? 1.0f : -1.0f;
              int r0 = crbase;
              int rt = crbase + (axis + 1) * 6;
              for (int i = 0; i < nv; ++i) {
                J_world[row * nv + i] = contact_jacobian[cjbase + i] + sign * mu * contact_jacobian[cjbase + (axis + 1) * nv + i];
              }
              R[row] = pyramid_R;
              ar[row] = contact_row_data[r0 + 3] + sign * mu * contact_row_data[rt + 3];
              lo[row] = 0.0f;
              hi[row] = INFINITY;
              enabled[row] = true;
            }
          }
        }
      }
    }
  }

  int total_nr = nr;


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
  for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
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
  for (int a = 0; a < total_nr; ++a) if (enabled[a]) {
    for (int b = 0; b < total_nr; ++b) if (enabled[b]) {
      float v = 0.0f;
      for (int k = 0; k < nv; ++k) v += J_world[a * nv + k] * Z[b * nv + k];
      W[a * nr + b] = v;
    }
  }

  // 7. Form RHS: ar - J q0
  for (int row = 0; row < total_nr; ++row) {
    if (enabled[row]) {
      float ja = 0.0f;
      for (int k = 0; k < nv; ++k) ja += J_world[row * nv + k] * out_acc[qb + k];
      rhs[row] = ar[row] - ja;
    } else {
      rhs[row] = 0.0f;
    }
  }

  // 7b. Warmstart cost check (pinned PGS rule): start from the retained
  // multipliers only when they improve on zero for the CURRENT Delassus
  // system; otherwise (released contacts, new manifolds, foreign vectors)
  // fall back to a cold start. Disabled rows never contribute.
  // Pinned: efc_b = J*qacc_smooth - aref = -rhs, cost = f.b + 0.5 f.A.f.
  if (warm_ok && total_nr > 0) {
    float wcost = 0.0f;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      wcost -= lam[row] * rhs[row];
      for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
        float a = W[row * nr + col] + (row == col ? R[row] : 0.0f);
        wcost += 0.5f * lam[row] * a * lam[col];
      }
    }
    if (wcost > 0.0f || !isfinite(wcost)) {
      for (int row = 0; row < total_nr; ++row) lam[row] = 0.0f;
    }
  }

  // 8. Projected Gauss-Seidel on coupled W + diag(R)
  float tol = params[2];
  bool converged = false;
  float max_res = 0.0f;
  // Convergence history: 7 coarse main-sweep samples + final main-sweep
  // residual (refinement afterwards is a separate exact phase; the retained
  // diagnostics[0] is post-refinement). Unfilled slots repeat the final.
  thread float hist[8];
  int hsamp = 0;
  int hstep = max(1, maxiter / 7);
  for (int k = 0; k < 8; ++k) hist[k] = 0.0f;
  if (elliptic_count > 0) {
    // Elliptic contacts are cone blocks in scaled coordinates. Solve the full
    // coupled quadratic with projected FISTA so interactions between multiple
    // manifold points are handled globally, rather than by slowly converging
    // local contact updates. Scalar joint/pyramid rows keep their own bounds.
    thread float dscale[96], z[96], extrapolated[96], candidate[96], gradient_z[96];
    for (int row = 0; row < 96; ++row) {
      dscale[row] = 1.0f;
      z[row] = extrapolated[row] = candidate[row] = 0.0f;
      gradient_z[row] = 0.0f;
    }
    for (int block = 0; block < elliptic_count; ++block) {
      int start = elliptic_start[block];
      int dim = elliptic_dim[block];
      for (int k = 1; k < dim; ++k)
        dscale[start + k] = max(elliptic_friction[block * 5 + k - 1], 0.0f);
    }
    float lipschitz = 1e-15f;
    int enabled_count = 0;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      ++enabled_count;
      float row_sum = 0.0f;
      for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
        float value = dscale[row] * W[row * nr + col] * dscale[col];
        if (row == col) value += dscale[row] * R[row] * dscale[col];
        row_sum += abs(value);
      }
      lipschitz = max(lipschitz, row_sum);
      float scale = dscale[row];
      z[row] = scale > 1e-12f ? lam[row] / scale : 0.0f;
      extrapolated[row] = z[row];
    }
    float gershgorin_bound = lipschitz;
    // The row-sum bound is safe but can be overly conservative for manifold
    // blocks. Estimate the dominant eigenvalue of the symmetric PSD scaled
    // Delassus matrix and retain a 10% safety margin for the global step.
    thread float power_vector[96], power_product[96];
    float inv_norm = rsqrt(float(max(enabled_count, 1)));
    for (int row = 0; row < 96; ++row)
      power_vector[row] = row < total_nr && enabled[row] ? inv_norm : 0.0f;
    for (int iteration = 0; iteration < 24; ++iteration) {
      float norm_sq = 0.0f;
      for (int row = 0; row < total_nr; ++row) {
        float value = 0.0f;
        if (enabled[row]) {
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
            float entry = dscale[row] * W[row * nr + col] * dscale[col];
            if (row == col) entry += dscale[row] * R[row] * dscale[col];
            value += entry * power_vector[col];
          }
        }
        power_product[row] = value;
        norm_sq += value * value;
      }
      float inv_power_norm = rsqrt(max(norm_sq, 1e-30f));
      for (int row = 0; row < total_nr; ++row)
        power_vector[row] = power_product[row] * inv_power_norm;
    }
    float rayleigh = 0.0f;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      float value = 0.0f;
      for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
        float entry = dscale[row] * W[row * nr + col] * dscale[col];
        if (row == col) entry += dscale[row] * R[row] * dscale[col];
        value += entry * power_vector[col];
      }
      rayleigh += power_vector[row] * value;
    }
    lipschitz = min(gershgorin_bound, max(1e-15f, rayleigh * 1.1f));
    float momentum = 1.0f;
    for (int it = 0; it < maxiter; ++it) {
      for (int row = 0; row < total_nr; ++row)
        lam[row] = enabled[row] ? dscale[row] * extrapolated[row] : 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float gradient = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col])
          gradient += W[row * nr + col] * lam[col];
        gradient += R[row] * lam[row];
        gradient_z[row] = dscale[row] * gradient;
      } else {
        gradient_z[row] = 0.0f;
      }
      // Backtracking makes the power-iteration step estimate safe even when
      // the transformed Delassus spectrum is clustered. The quadratic model
      // check guarantees that the accepted projected-gradient step majorizes
      // the actual coupled objective at this extrapolated point.
      bool step_accepted = false;
      for (int backtrack = 0; backtrack < 12; ++backtrack) {
        for (int row = 0; row < total_nr; ++row)
          candidate[row] = enabled[row]
              ? extrapolated[row] - gradient_z[row] / lipschitz : 0.0f;
        for (int block = 0; block < elliptic_count; ++block) {
          int start = elliptic_start[block];
          int dim = elliptic_dim[block];
          thread float cone_value[6];
          for (int k = 0; k < 6; ++k) cone_value[k] = k < dim ? candidate[start + k] : 0.0f;
          project_lorentz(cone_value, dim);
          for (int k = 0; k < dim; ++k) candidate[start + k] = cone_value[k];
        }
        for (int row = 0; row < total_nr; ++row)
          if (enabled[row] && !elliptic_member[row])
            candidate[row] = clamp(candidate[row], lo[row], hi[row]);

        float objective_base = 0.0f, objective_candidate = 0.0f;
        float linear_delta = 0.0f, delta_norm_sq = 0.0f;
        for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
          float base_lambda = dscale[row] * extrapolated[row];
          float candidate_lambda = dscale[row] * candidate[row];
          float base_product = 0.0f;
          float candidate_product = 0.0f;
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
            float base_col = dscale[col] * extrapolated[col];
            float candidate_col = dscale[col] * candidate[col];
            base_product += W[row * nr + col] * base_col;
            candidate_product += W[row * nr + col] * candidate_col;
          }
          base_product += R[row] * base_lambda;
          candidate_product += R[row] * candidate_lambda;
          objective_base += 0.5f * base_lambda * base_product - rhs[row] * base_lambda;
          objective_candidate += 0.5f * candidate_lambda * candidate_product - rhs[row] * candidate_lambda;
          float delta = candidate[row] - extrapolated[row];
          linear_delta += gradient_z[row] * delta;
          delta_norm_sq += delta * delta;
        }
        float majorizer = objective_base + linear_delta + 0.5f * lipschitz * delta_norm_sq;
        if (objective_candidate <= majorizer + 1e-6f * max(1.0f, abs(objective_candidate))) {
          step_accepted = true;
          break;
        }
        lipschitz *= 2.0f;
      }
      if (!step_accepted) { out_status[world] = 2; return; }

      float next_momentum = 0.5f * (1.0f + sqrt(1.0f + 4.0f * momentum * momentum));
      float beta = (momentum - 1.0f) / next_momentum;
      float restart_dot = 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row])
        restart_dot += (candidate[row] - z[row]) * (extrapolated[row] - candidate[row]);
      if (restart_dot > 0.0f) {
        next_momentum = 1.0f;
        beta = 0.0f;
      }
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float previous = z[row];
        z[row] = candidate[row];
        extrapolated[row] = candidate[row] + beta * (candidate[row] - previous);
        lam[row] = dscale[row] * z[row];
      }
      momentum = next_momentum;

      max_res = 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        if (elliptic_member[row]) {
          int block = -1;
          for (int b = 0; b < elliptic_count; ++b) if (elliptic_start[b] == row) block = b;
          if (block < 0) continue;
          int dim = elliptic_dim[block];
          thread float A[36], g[6], mu[5], force[6];
          for (int i = 0; i < 36; ++i) A[i] = 0.0f;
          for (int i = 0; i < 6; ++i) { g[i] = 0.0f; force[i] = 0.0f; }
          for (int i = 0; i < 5; ++i) mu[i] = elliptic_friction[block * 5 + i];
          for (int i = 0; i < dim; ++i) {
            int ri = row + i;
            force[i] = lam[ri];
            g[i] = -rhs[ri];
            for (int col = 0; col < total_nr; ++col)
              if (enabled[col] && (col < row || col >= row + dim))
                g[i] += W[ri * nr + col] * lam[col];
            for (int j = 0; j < dim; ++j) {
              int rj = row + j;
              A[i * 6 + j] = W[ri * nr + rj] + (i == j ? R[ri] : 0.0f);
            }
          }
          max_res = max(max_res, elliptic_projected_residual(A, g, mu, dim, force));
          continue;
        }
        float grad = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      out_diagnostics[world * 10] = max_res;
      out_diagnostics[world * 10 + 1] = float(it + 1);
      if (hsamp < 7 && ((it + 1) % hstep == 0 || it + 1 == maxiter)) {
        hist[hsamp++] = max_res;
      }
      if (max_res <= tol) { converged = true; break; }
    }
  } else {
    // F4: redundant-contact systems certify slowly in float32, so a
    // budgeted PGS sweep can end close (<=1e-3) but uncertified while still
    // improving (box face plants freeze without this): grant another budget
    // window up to a hard cap. Acceptance still requires max_res <= tol;
    // true divergence keeps status 3 via the unchanged failure path below.
    // Loose-but-certified solutions are tightened by block refinement in
    // section 9; already-tight ones skip it bit-identically.
    int pgs_cap = maxiter;
    // G3: seed the progress comparison from the actual starting residual
    // (lam holds the retained warm vector, or zero when cold), not
    // from +inf, so the first extension must prove improvement over the
    // actual starting point.
    float win_start = 0.0f;
    for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
      float grad = -rhs[row];
      for (int col = 0; col < total_nr; ++col) if (enabled[col])
        grad += W[row * nr + col] * lam[col];
      grad += R[row] * lam[row];
      float diag = max(1e-15f, W[row * nr + row] + R[row]);
      float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
      float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
      for (int col = 0; col < total_nr; ++col) if (enabled[col])
        row_scale += abs(W[row * nr + col] * lam[col]);
      row_scale = max(1.0f, row_scale);
      win_start = max(win_start, abs(proj - lam[row]) * diag / row_scale);
    }
    for (int it = 0; it < pgs_cap; ++it) {
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float v = rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col] && col != row)
          v -= W[row * nr + col] * lam[col];
        lam[row] = clamp(v / diag, lo[row], hi[row]);
      }
      max_res = 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float grad = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      out_diagnostics[world * 10] = max_res;
      out_diagnostics[world * 10 + 1] = float(it + 1);
      if (hsamp < 7 && (it + 1) % hstep == 0) hist[hsamp++] = max_res;
      if (max_res <= tol) {
        converged = true;
        break;
      } else if (maxiter > 2 && it + 1 == pgs_cap && pgs_cap < 1024 && max_res <= 1e-3f
          && max_res < win_start) {
        // Close and improved over the window: grant another window.
        win_start = max_res;
        pgs_cap = min(1024, pgs_cap * 2);
      }
    }
  }

  // Complete the convergence history: unfilled sample slots repeat the
  // final main-sweep residual; slot 7 always holds it.
  while (hsamp < 7) hist[hsamp++] = max_res;
  hist[7] = max_res;
  for (int k = 0; k < 8; ++k) out_diagnostics[world * 10 + 2 + k] = hist[k];

  // 9. Contact Block Refinement (exact per-block solves tighten the PGS
  // solution). Runs when the sweep did not certify, or when it certified
  // only loosely (res > 1e-7): loose PGS solutions drift on rolling
  // friction while exact block solves match the CPU oracle. Already-tight
  // solutions skip it, bit-identical to before. Elliptic blocks join via
  // alternating normal/QCQP exact block rounds (R04 pinch repair).
  // G2: refinement changes retained multipliers, so the pre-refinement
  // certificate must not survive it. Snapshot the certified candidate;
  // after refinement, re-certify from the FINAL multipliers: revert to the
  // snapshot if refinement lost certification, fail on nonfinite residuals.
  // Tiny budgets (maxiter <= 2) skip refinement: the 256-round exact phase
  // would dominate the reported iteration count and misreport exhaustion.
  if (maxiter > 2 && (!converged || max_res > 1e-7f) && (contact_block_count > 0 || elliptic_count > 0)) {
    thread float snap_lam[96];
    for (int r = 0; r < 96; ++r) snap_lam[r] = lam[r];
    float snap_res = max_res;
    bool snap_certified = converged;
    converged = false;
    for (int ref = 0; ref < 256; ++ref) {
      for (int b = 0; b < contact_block_count; ++b) {
        int row_start = contact_block_start[b];
        int block_size = contact_block_size[b];
        // Preserve the original four-edge pyramid refinement budget. The
        // expanded condim-4/6 blocks need the larger bound qualified below.
        if (block_size <= 4 && ref >= 64) continue;
        int local_rows[10];
        int nlocal = 0;
        for (int k = 0; k < block_size; ++k) {
          int row = row_start + k;
          if (enabled[row]) local_rows[nlocal++] = row;
        }
        if (nlocal == 0) continue;
        thread float local_A[100], local_b[10], local_sol[10];
        for (int i = 0; i < 100; ++i) local_A[i] = 0.0f;
        for (int i = 0; i < 10; ++i) { local_b[i] = 0.0f; local_sol[i] = 0.0f; }
        for (int i = 0; i < nlocal; ++i) {
          int row = local_rows[i];
          float v = -rhs[row];
          for (int col = 0; col < total_nr; ++col) if (enabled[col] && (col < row_start || col >= row_start + block_size)) {
            v += W[row * nr + col] * lam[col];
          }
          local_b[i] = v;
          local_sol[i] = lam[row];
          for (int j = 0; j < nlocal; ++j) {
            int other = local_rows[j];
            local_A[i * 10 + j] = W[row * nr + other] + (i == j ? R[row] : 0.0f);
          }
        }
        float b_err;
        if (nlocal <= 4) {
          thread float small_A[16], small_b[4], small_sol[4];
          for (int i = 0; i < 16; ++i) small_A[i] = 0.0f;
          for (int i = 0; i < 4; ++i) { small_b[i] = 0.0f; small_sol[i] = 0.0f; }
          for (int i = 0; i < nlocal; ++i) {
            small_b[i] = local_b[i];
            for (int j = 0; j < nlocal; ++j) small_A[i * 4 + j] = local_A[i * 10 + j];
          }
          b_err = solve_contact_block(small_A, small_b, nlocal, small_sol);
          for (int i = 0; i < nlocal; ++i) local_sol[i] = small_sol[i];
        } else {
          b_err = solve_contact_block_iterative(local_A, local_b, nlocal, local_sol);
        }
        if (b_err < 1e-5f) {
          for (int i = 0; i < nlocal; ++i) lam[local_rows[i]] = max(0.0f, local_sol[i]);
        }
      }
      // Elliptic blocks: alternating exact normal 1D Newton and QCQP
      // friction solves (4 rounds), with per-block revert when the block
      // cone residual does not improve.
      for (int b = 0; b < elliptic_count; ++b) {
        int start = elliptic_start[b];
        int dim = elliptic_dim[b];
        thread float mu[5];
        for (int k = 0; k < 5; ++k) mu[k] = elliptic_friction[b * 5 + k];
        float before = cert_block_residual(start, dim, mu, W, R, rhs, lam,
                                           enabled, nr, total_nr);
        thread float keep[6];
        for (int k = 0; k < 6; ++k) keep[k] = 0.0f;
        for (int k = 0; k < dim; ++k) keep[k] = lam[start + k];
        for (int round = 0; round < 4; ++round) {
          // Friction QCQP given the current normal.
          {
            int nf = dim - 1;
            thread float Ac[36], bc[6], oldf[6];
            for (int i = 0; i < 36; ++i) Ac[i] = 0.0f;
            for (int i = 0; i < 6; ++i) { bc[i] = 0.0f; oldf[i] = 0.0f; }
            float fn = lam[start];
            if (fn < 1e-15f) {
              for (int k = 1; k < dim; ++k) lam[start + k] = 0.0f;
            } else {
              for (int i = 0; i < nf; ++i) {
                int ri = start + 1 + i;
                oldf[i] = lam[ri];
                float v = -rhs[ri];
                for (int col = 0; col < total_nr; ++col)
                  if (enabled[col] && (col < start + 1 || col >= start + dim))
                    v += W[ri * nr + col] * lam[col];
                bc[i] = v;
                for (int j = 0; j < nf; ++j)
                  Ac[i * nf + j] = W[ri * nr + start + 1 + j] + (i == j ? R[ri] : 0.0f);
              }
              thread float qv[6];
              for (int i = 0; i < 6; ++i) qv[i] = 0.0f;
              int act = 0;
              if (nf == 2) act = nsl_qcqp2(qv, Ac, bc, mu, fn);
              else if (nf == 3) act = nsl_qcqp3(qv, Ac, bc, mu, fn);
              else act = nsl_qcqpn(qv, Ac, bc, mu, fn, nf);
              if (act) {
                float s = 0.0f;
                for (int j = 0; j < nf; ++j) s += qv[j] * qv[j] / max(mu[j] * mu[j], 1e-30f);
                s = sqrt(fn * fn / max(1e-15f, s));
                for (int j = 0; j < nf; ++j) qv[j] *= s;
              }
              for (int i = 0; i < nf; ++i) lam[start + 1 + i] = qv[i];
            }
          }
          // Normal 1D Newton given friction.
          {
            float grad = -rhs[start];
            for (int col = 0; col < total_nr; ++col)
              if (enabled[col]) grad += W[start * nr + col] * lam[col];
            grad += R[start] * lam[start];
            float diag = max(1e-15f, W[start * nr + start] + R[start]);
            lam[start] = max(0.0f, lam[start] - grad / diag);
          }
        }
        float after = cert_block_residual(start, dim, mu, W, R, rhs, lam,
                                          enabled, nr, total_nr);
        if (!(after <= before) || !isfinite(after)) {
          for (int k = 0; k < dim; ++k) lam[start + k] = keep[k];
        }
      }
      max_res = 0.0f;
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        if (elliptic_member[row]) continue;
        float grad = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) grad += W[row * nr + col] * lam[col];
        grad += R[row] * lam[row];
        float diag = max(1e-15f, W[row * nr + row] + R[row]);
        float proj = clamp(lam[row] - grad / diag, lo[row], hi[row]);
        float row_scale = abs(ar[row]) + abs(R[row] * lam[row]);
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) row_scale += abs(W[row * nr + col] * lam[col]);
        row_scale = max(1.0f, row_scale);
        max_res = max(max_res, abs(proj - lam[row]) * diag / row_scale);
      }
      for (int b = 0; b < elliptic_count; ++b) {
        int start = elliptic_start[b];
        int dim = elliptic_dim[b];
        thread float mu[5];
        for (int k = 0; k < 5; ++k) mu[k] = elliptic_friction[b * 5 + k];
        max_res = max(max_res, cert_block_residual(start, dim, mu, W, R, rhs,
                                                   lam, enabled, nr, total_nr));
      }
      out_diagnostics[world * 10] = max_res;
      out_diagnostics[world * 10 + 1] = float(int(out_diagnostics[world * 10 + 1]) + 1);
      if (max_res <= tol) { converged = true; break; }
    }
    // G2: certify the FINAL retained multipliers from scratch. Refinement
    // may have replaced a certified candidate with an uncertified (or
    // nonfinite) one while other coupled rows stayed fixed: revert to the
    // snapshot when it was certified, otherwise fail loudly. The status
    // below therefore always describes the retained solution.
    if (!isfinite(max_res)) {
      if (snap_certified) {
        for (int r = 0; r < 96; ++r) lam[r] = snap_lam[r];
        max_res = snap_res;
        converged = true;
      } else {
        converged = false;
      }
    } else if (max_res > tol) {
      if (snap_certified) {
        for (int r = 0; r < 96; ++r) lam[r] = snap_lam[r];
        max_res = snap_res;
        converged = true;
      } else {
        converged = false;
      }
    } else {
      converged = true;
    }
    out_diagnostics[world * 10] = max_res;
  }

  // 9b. No-slip post-pass (pinned solNoSlip): exact friction subproblem
  // solves over dry-friction rows and contact friction blocks. Gated on
  // noslip_iterations (dims[18]); zero iterations skip bit-identically.
  // Improvement accounting and the noslip_tolerance stop rule (params[3])
  // mirror the pinned stage; iteration counts accumulate into diagnostics.
  int noslip_iters = dims[18];
  float noslip_tol = params[3];
  float mean_inertia = params[4];
  if (!isfinite(mean_inertia) || mean_inertia <= 1e-6f) mean_inertia = 1.0f;
  float noslip_scale = 1.0f / (mean_inertia * float(max(1, nv)));
  if (noslip_iters > 0 && total_nr > 0) {
    int ns_done = 0;
    for (int nsit = 0; nsit < noslip_iters; ++nsit) {
      float improvement = 0.0f;
      // At iteration 0, account for regularizer removal (pinned engine_solver.c:674-678)
      if (nsit == 0) {
        for (int row = 0; row < total_nr; ++row) {
          if (enabled[row]) {
            improvement += 0.5f * lam[row] * lam[row] * R[row];
          }
        }
      }

      // Dry friction rows: symmetric finite bounds identify joint and
      // tendon frictionloss rows (bilateral equalities use infinite
      // bounds; limits use [0, inf)). Regularizer R is excluded.
      for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
        float lorb = lo[row], hib = hi[row];
        if (!(lorb < 0.0f && hib == -lorb && hib < INFINITY)) continue;
        float diag = max(1e-10f, W[row * nr + row]);
        float res = -rhs[row];
        for (int col = 0; col < total_nr; ++col) if (enabled[col]) res += W[row * nr + col] * lam[col];
        float old = lam[row];
        float v = clamp(old - res / diag, lorb, hib);
        float delta = v - old;
        float change = 0.5f * delta * delta * diag + delta * res;
        if (change > 1e-10f) {
          v = old;
          change = 0.0f;
        }
        lam[row] = v;
        improvement -= change;
      }
      // Contact friction blocks.
      for (int b = 0; b < contact_block_count; ++b) {
        int row_start = contact_block_start[b];
        int block_size = contact_block_size[b];
        // Pyramidal edge pairs: exact 2D solve preserving the pair sum (R excluded).
        for (int k = 0; k + 1 < block_size; k += 2) {
          int j0 = row_start + k, j1 = row_start + k + 1;
          if (!(enabled[j0] && enabled[j1])) continue;
          float r0 = -rhs[j0], r1 = -rhs[j1];
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) {
            r0 += W[j0 * nr + col] * lam[col];
            r1 += W[j1 * nr + col] * lam[col];
          }
          float old0 = lam[j0], old1 = lam[j1];
          float A00 = W[j0 * nr + j0], A11 = W[j1 * nr + j1];
          float A01 = W[j0 * nr + j1];
          float bc0 = r0 - (A00 * old0 + A01 * old1);
          float bc1 = r1 - (A01 * old0 + A11 * old1);
          float mid = 0.5f * (old0 + old1);
          float y = 0.5f * (old0 - old1);
          float K1 = A00 + A11 - 2.0f * A01;
          float K0 = mid * (A00 - A11) + bc0 - bc1;
          float ny = y;
          if (K1 < 1e-15f) {
            ny = y;
          } else {
            ny = -K0 / K1;
            ny = ny < -mid ? -mid : (ny > mid ? mid : ny);
          }
          float f0 = mid + ny, f1 = mid - ny;
          // costChange with unregularized residual r0, r1.
          float d0 = f0 - old0, d1 = f1 - old1;
          float change = 0.5f * (d0 * (A00 * d0 + A01 * d1) + d1 * (A01 * d0 + A11 * d1))
              + d0 * r0 + d1 * r1;
          if (change > 1e-10f) {
            f0 = old0;
            f1 = old1;
            change = 0.0f;
          }
          lam[j0] = f0;
          lam[j1] = f1;
          improvement -= change;
        }
      }
      // Elliptic friction blocks: QCQP over friction rows given normal (R excluded).
      for (int b = 0; b < elliptic_count; ++b) {
        int start = elliptic_start[b];
        int dim = elliptic_dim[b];
        if (!enabled[start]) continue;
        float fn = lam[start];
        if (fn < 1e-15f) {
          for (int k = 1; k < dim; ++k) lam[start + k] = 0.0f;
          continue;
        }
        int nf = dim - 1;
        thread float Ac[36], bc[6], oldf[6], muf[5], res_arr[6];
        for (int i = 0; i < 36; ++i) Ac[i] = 0.0f;
        for (int i = 0; i < 6; ++i) { bc[i] = 0.0f; oldf[i] = 0.0f; res_arr[i] = 0.0f; }
        for (int i = 0; i < 5; ++i) muf[i] = elliptic_friction[b * 5 + i];
        for (int i = 0; i < nf; ++i) {
          int ri = start + 1 + i;
          oldf[i] = lam[ri];
          float v = -rhs[ri];
          for (int col = 0; col < total_nr; ++col) if (enabled[col]) v += W[ri * nr + col] * lam[col];
          res_arr[i] = v;
          bc[i] = v;
          for (int j = 0; j < nf; ++j) {
            Ac[i * nf + j] = W[ri * nr + start + 1 + j];
            bc[i] -= Ac[i * nf + j] * lam[start + 1 + j];
          }
        }
        thread float qv[6];
        for (int i = 0; i < 6; ++i) qv[i] = 0.0f;
        int active = 0;
        if (nf == 2) active = nsl_qcqp2(qv, Ac, bc, muf, fn);
        else if (nf == 3) active = nsl_qcqp3(qv, Ac, bc, muf, fn);
        else active = nsl_qcqpn(qv, Ac, bc, muf, fn, nf);
        if (active) {
          float s = 0.0f;
          for (int j = 0; j < nf; ++j) s += qv[j] * qv[j] / max(muf[j] * muf[j], 1e-30f);
          s = sqrt(fn * fn / max(1e-15f, s));
          for (int j = 0; j < nf; ++j) qv[j] *= s;
        }
        for (int i = 0; i < nf; ++i) lam[start + 1 + i] = qv[i];
        // costChange uses unregularized residual res_arr, matching pinned engine_solver.c.
        float change = 0.0f;
        for (int i = 0; i < nf; ++i) {
          float delta = lam[start + 1 + i] - oldf[i];
          float ad = 0.0f;
          for (int j = 0; j < nf; ++j) ad += Ac[i * nf + j] * delta;
          change += 0.5f * delta * ad + delta * res_arr[i];
        }
        if (change > 1e-10f) {
          for (int i = 0; i < nf; ++i) lam[start + 1 + i] = oldf[i];
          change = 0.0f;
        }
        improvement -= change;
      }
      ns_done++;
      improvement *= noslip_scale;
      if (improvement < noslip_tol) break;
    }
    out_diagnostics[world * 10 + 1] += float(ns_done);
    // Note: No-slip solves the unregularized friction subproblem and deliberately
    // deviates from the regularized QP optimum. Do not overwrite `converged` with
    // a regularized KKT certificate here (pinned semantics, R04).
    bool lam_finite = true;
    for (int r = 0; r < total_nr; ++r) {
      if (enabled[r] && !isfinite(lam[r])) { lam_finite = false; break; }
    }
    if (!lam_finite) converged = false;
  }

  // G2: nonfinite residuals fail explicitly (NaN never satisfies `> tol`,
  // so without this guard a nonfinite solve would report success).
  if (!isfinite(max_res) || !converged) out_status[world] = 3;

  // 10. Reconstruct forces and acceleration
  for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
    for (int i = 0; i < nv; ++i) out_force[qb + i] += J_world[row * nv + i] * lam[row];
  }
  for (int row = 0; row < total_nr; ++row) if (enabled[row]) {
    for (int i = 0; i < nv; ++i) out_acc[qb + i] += Z[row * nv + i] * lam[row];
  }

  // 11. Write joint forces
  for (int row = 0; row < base_contact; ++row) {
    out_joint_force[world * max(base_contact, 1) + row] = enabled[row] ? lam[row] : 0.0f;
  }

  // 12. Write contact forces
  for (int s = 0; s < ncontacts_max; ++s) {
    int cdim = contact_condim[s * 3 + 0];
    int row_offset = contact_condim[s * 3 + 1];
    int row_start = base_contact + row_offset;
    int ofb = (world * ncontacts_max + s) * 11;
    for (int k = 0; k < 11; ++k) out_contact_force[ofb + k] = 0.0f;
    if (cdim == 0 || !enabled[row_start]) continue;
    if (cdim == 1) {
      out_contact_force[ofb] = lam[row_start];
    } else if (cone_type == 1) {
      for (int k = 0; k < cdim; ++k) out_contact_force[ofb + k] = enabled[row_start + k] ? lam[row_start + k] : 0.0f;
    } else {
      int edge_count = 2 * (cdim - 1);
      float normal = 0.0f;
      for (int k = 0; k < edge_count; ++k) {
        float value = lam[row_start + k];
        normal += value;
        out_contact_force[ofb + 1 + k] = value;
      }
      out_contact_force[ofb] = normal;
    }
  }


  // 13. Write debug matrices and vectors if workspace_debug is provided
  if (workspace_debug && nr > 0) {
    device float* dbg = workspace_debug + world * (nr * nr + 7 * nr);
    for (int a = 0; a < nr; ++a) {
      for (int b = 0; b < nr; ++b) {
        dbg[a * nr + b] = (a < total_nr && b < total_nr) ? W[a * nr + b] : 0.0f;
      }
      dbg[nr * nr + a] = a < total_nr ? R[a] : 0.0f;
      dbg[nr * nr + nr + a] = a < total_nr ? ar[a] : 0.0f;
      dbg[nr * nr + 2 * nr + a] = a < total_nr ? rhs[a] : 0.0f;
      dbg[nr * nr + 3 * nr + a] = (a < total_nr && enabled[a]) ? lam[a] : 0.0f;
      // Retain joint/contact row bounds for host-side KKT residual checks.
      // Equality and tendon-owned rows keep their preassembled bounds.
      if (a < total_nr && a >= n_eq_rows && dbg[nr * nr + 6 * nr + a] <= 0.5f) {
        dbg[nr * nr + 4 * nr + a] = lo[a];
        dbg[nr * nr + 5 * nr + a] = hi[a];
      }
    }
  }
}

// Tendon constraint rows for MuJoCo 3.10.0 (milestone 008).
// Pinned sources: engine/engine_core_constraint.c (tendon-limit loop with
// side-scaled Jacobian, friction-tendon rows, JOINT/TENDON cubic equality with
// tendon_length0 reference and tendon_invweight0 diagonal).
// Assembles tendon equality rows (eq region, cubic coupling), tendon
// frictionloss rows and tendon limit rows (reserved ten region) into
// workspace_J, and writes R/ar/lo/hi/enabled into the extended debug regions
// [4*nr, 7*nr). Runs after equality_assembly (which reserves zeros for tendon
// equalities) and before solve_coupled_constraints (which reads rows flagged
// in ten_en instead of computing them).
kernel void tendon_constraint_rows(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* ten_J_spatial [[buffer(2)]],
    device const float* ten_length_spatial [[buffer(3)]],
    device const float* ten_length_map [[buffer(4)]],
    device const float* ten_moment_map [[buffer(5)]],
    device const int* ten_limited [[buffer(6)]],
    device const float* ten_range [[buffer(7)]],
    device const float* ten_margin [[buffer(8)]],
    device const float* ten_length0 [[buffer(9)]],
    device const float* ten_invweight0 [[buffer(10)]],
    device const float* ten_solref_lim [[buffer(11)]],
    device const float* ten_solimp_lim [[buffer(12)]],
    device const float* ten_frictionloss [[buffer(13)]],
    device const float* ten_solref_fri [[buffer(14)]],
    device const float* ten_solimp_fri [[buffer(15)]],
    device const int* eq_type [[buffer(16)]],
    device const int* eq_obj [[buffer(17)]],
    device const float* eq_data [[buffer(18)]],
    device const float* eq_sol_params [[buffer(19)]],
    device const int* eq_rowadr [[buffer(20)]],
    device const int* eq_active [[buffer(21)]],
    constant int* dims [[buffer(22)]],
    constant float* sparams [[buffer(23)]],
    device float* workspace_J [[buffer(24)]],
    device float* workspace_debug [[buffer(25)]],
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nv=dims[1], nt=dims[15], neq=dims[3], batch=dims[5];
  int flags=dims[6];
  bool refsafe=dims[7]!=0;
  int nr=dims[9];
  int ten_base=dims[16];
  float timestep=sparams[0];
  if (uint(world)>=uint(batch)) return;
  if (nr<=0||nv<=0) return;
  device float* Jw = workspace_J + uint(world)*uint(nr)*uint(max(nv,1));
  device float* dbg = workspace_debug + uint(world)*(uint(nr*nr)+uint(7*nr));
  uint vbase=uint(world)*uint(max(nv,1)), qbase=uint(world)*uint(max(nq,1));
  uint tbase=uint(world)*uint(max(nt,1));
  for (int r=0;r<nr;++r) {
    dbg[nr*nr+4*nr+r]=0.0f;
    dbg[nr*nr+5*nr+r]=0.0f;
    dbg[nr*nr+6*nr+r]=0.0f;
  }
  if (nt<=0) return;
  if ((flags&1)!=0) return;
  // Per-tendon combined length and dense Jacobian row.
  // nt is small (tendon counts stay bounded); thread-local row buffer.
  for (int t=0;t<nt;++t) {
    float L=0.0f;
    for (int q=0;q<nq;++q) L+=ten_length_map[t*max(nq,1)+q]*qpos[qbase+uint(q)];
    L+=ten_length_spatial[tbase+uint(t)];
    float Jrow[32];
    for (int d=0;d<32;++d) Jrow[d]=0.0f;
    for (int d=0;d<nv;++d)
      Jrow[d]=ten_moment_map[t*max(nv,1)+d]+ten_J_spatial[(tbase+uint(t))*uint(max(nv,1))+uint(d)];
    float vel=0.0f;
    for (int d=0;d<nv;++d) vel+=Jrow[d]*qvel[vbase+uint(d)];
    // Friction loss row (pinned: J=ten_J, margin 0, bound loss).
    if ((flags&4)==0 && ten_frictionloss[t]>0.0f) {
      int slot=0;
      for (int u=0;u<t;++u) if (ten_frictionloss[u]>0.0f) slot++;
      int row=ten_base+slot;
      if (row>=0&&row<nr) {
        for (int d=0;d<nv;++d) Jw[row*nv+d]=Jrow[d];
        float R, ar;
        reference_params(ten_solref_fri+t*2, ten_solimp_fri+t*5, 0, 0.0f, 0.0f,
          vel, ten_invweight0[t], true, timestep, refsafe, R, ar);
        dbg[nr*nr+row]=R; dbg[nr*nr+nr+row]=ar;
        dbg[nr*nr+4*nr+row]=-ten_frictionloss[t];
        dbg[nr*nr+5*nr+row]=ten_frictionloss[t];
        dbg[nr*nr+6*nr+row]=1.0f;
      }
    }
    // Limit rows (pinned: J scaled by -side, dist=side*(range-value)).
    if ((flags&8)==0 && ten_limited[t]!=0) {
      int nfric=0;
      for (int u=0;u<nt;++u) if (ten_frictionloss[u]>0.0f) nfric++;
      int pre=0;
      for (int u=0;u<t;++u) if (ten_limited[u]!=0) pre++;
      for (int s=0;s<2;++s) {
        float side=s==0?-1.0f:1.0f;
        float rangev=s==0?ten_range[2*t]:ten_range[2*t+1];
        float dist=side*(rangev-L);
        if (dist<ten_margin[t]) {
          int row=ten_base+nfric+2*pre+s;
          if (row>=0&&row<nr) {
            for (int d=0;d<nv;++d) Jw[row*nv+d]=-side*Jrow[d];
            float R, ar;
            reference_params(ten_solref_lim+t*2, ten_solimp_lim+t*5, 0, dist,
              ten_margin[t], -side*vel, ten_invweight0[t], false, timestep, refsafe, R, ar);
            dbg[nr*nr+row]=R; dbg[nr*nr+nr+row]=ar;
            dbg[nr*nr+4*nr+row]=0.0f;
            dbg[nr*nr+5*nr+row]=INFINITY;
            dbg[nr*nr+6*nr+row]=1.0f;
          }
        }
      }
    }
  }
  // Tendon equality rows (pinned cubic coupling, margin 0).
  if ((flags&2)!=0) return;
  for (int e=0;e<neq;++e) {
    if (eq_type[e]!=3) continue;
    if (eq_active[uint(world)*uint(max(neq,1))+uint(e)]==0) continue;
    int t1=eq_obj[2*e], t2=eq_obj[2*e+1];
    if (t1<0||t1>=nt) continue;
    int row=eq_rowadr[e];
    if (row<0||row>=nr) continue;
    // combined length/J for t1
    float L1=0.0f;
    for (int q=0;q<nq;++q) L1+=ten_length_map[t1*max(nq,1)+q]*qpos[qbase+uint(q)];
    L1+=ten_length_spatial[tbase+uint(t1)];
    float J1[32];
    for (int d=0;d<32;++d) J1[d]=0.0f;
    for (int d=0;d<nv;++d)
      J1[d]=ten_moment_map[t1*max(nv,1)+d]+ten_J_spatial[(tbase+uint(t1))*uint(max(nv,1))+uint(d)];
    float pos=L1-ten_length0[t1]-eq_data[e*11];
    float vel=0.0f;
    for (int d=0;d<nv;++d) vel+=J1[d]*qvel[vbase+uint(d)];
    float diag=ten_invweight0[t1];
    for (int d=0;d<nv;++d) Jw[row*nv+d]=J1[d];
    if (t2>=0&&t2<nt) {
      float L2=0.0f;
      for (int q=0;q<nq;++q) L2+=ten_length_map[t2*max(nq,1)+q]*qpos[qbase+uint(q)];
      L2+=ten_length_spatial[tbase+uint(t2)];
      float J2[32];
      for (int d=0;d<32;++d) J2[d]=0.0f;
      for (int d=0;d<nv;++d)
        J2[d]=ten_moment_map[t2*max(nv,1)+d]+ten_J_spatial[(tbase+uint(t2))*uint(max(nv,1))+uint(d)];
      float dif=L2-ten_length0[t2];
      float poly=0.0f, deriv=0.0f, power=dif;
      for (int k=0;k<4;++k) {
        poly+=eq_data[e*11+k+1]*power;
        deriv+=float(k+1)*eq_data[e*11+k+1]*pow(dif,float(k));
        power*=dif;
      }
      pos-=poly;
      float vel2=0.0f;
      for (int d=0;d<nv;++d) { vel2+=J2[d]*qvel[vbase+uint(d)]; Jw[row*nv+d]=J1[d]-deriv*J2[d]; }
      vel-=deriv*vel2;
      diag+=ten_invweight0[t2];
    }
    float d_width=max(1e-15f, eq_sol_params[e*7+3]);
    float imp=eq_impedance(eq_sol_params+e*7+2, pos, 0.0f);
    float r0=eq_sol_params[e*7], r1=eq_sol_params[e*7+1];
    if (refsafe&&r0>0.0f) r0=max(r0,2.0f*timestep);
    float K=r0>0.0f?1.0f/max(1e-15f,d_width*d_width*r0*r0*r1*r1):-r0/max(1e-15f,d_width*d_width);
    float B=r1>0.0f?2.0f/max(1e-15f,d_width*r0):-r1/d_width;
    dbg[nr*nr+row]=max(1e-15f,(1.0f-imp)*diag/imp);
    dbg[nr*nr+nr+row]=-B*vel-K*imp*pos;
    dbg[nr*nr+4*nr+row]=-INFINITY;
    dbg[nr*nr+5*nr+row]=INFINITY;
    dbg[nr*nr+6*nr+row]=1.0f;
  }
}
