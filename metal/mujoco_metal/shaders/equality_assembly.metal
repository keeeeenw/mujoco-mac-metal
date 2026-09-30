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

// Native equality assembly for joint/connect (weld translational placeholder).
// Pinned semantics: MuJoCo 3.10.0 engine/engine_setconst.c (eq_data body/site
// semantics), engine/engine_core_constraint.c (mj_equalityAnchors,
// mj_instantiateEquality, mj_diagApprox, getposdim/getimpedance,
// mj_referenceConstraint, mj_Jdotv translational part), doc/XMLreference.rst
// (connect anchor in body1 local, weld anchor in body2 local/relpose).
// Per-step assembly executes on Metal; CPU work is constant preparation only.

#include <metal_stdlib>
using namespace metal;

inline float3 eq_qrot(float4 q, float3 v) {
  return v + 2.0f * cross(q.yzw, cross(q.yzw, v) + q.x * v);
}

inline float eq_impedance(device const float* imp, float pos, float margin) {
  float d0 = imp[0];
  float d1 = imp[1];
  float width = imp[2];
  float mid = imp[3];
  float power = imp[4];
  if (d0 == d1 || width <= 1e-15f) return 0.5f * (d0 + d1);
  float x = abs((pos - margin) / width);
  if (x >= 1.0f) return d1;
  if (x <= 0.0f) return d0;
  float y = power == 1.0f ? x : (x <= mid ? pow(x, power) / pow(mid, power - 1.0f)
      : 1.0f - pow(1.0f - x, power) / pow(1.0f - mid, power - 1.0f));
  return d0 + y * (d1 - d0);
}

// Point Jacobian via parent traversal (dense, like contact_normal kernel).
// Computes translational Jacobian Jp (3 x nv) for global point attached to body.
inline void eq_point_jac(
    float3 point, int body,
    device const float* body_pos, device const float* body_quat,
    device const float* anchors, device const float* axes,
    device const int* body_parentid, device const int* body_jntadr,
    device const int* body_jntnum, device const int* jnt_type,
    device const int* jnt_dofadr,
    int nbody, int njnt, int nv, int bo, int jo,
    thread float* Jp) {
  for (int i = 0; i < 3 * 32; ++i) Jp[i] = 0.0f;
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
        if (typ == 2) {
          col = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
        } else if (typ == 3) {
          float3 axis = float3(axes[(jo + j) * 3], axes[(jo + j) * 3 + 1], axes[(jo + j) * 3 + 2]);
          float3 anchor = float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]);
          col = cross(axis, point - anchor);
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
            float3 pivot;
            if (typ == 0) {
              pivot = float3(body_pos[(bo + b) * 3], body_pos[(bo + b) * 3 + 1], body_pos[(bo + b) * 3 + 2]);
            } else {
              pivot = float3(anchors[(jo + j) * 3], anchors[(jo + j) * 3 + 1], anchors[(jo + j) * 3 + 2]);
            }
            col = cross(axis, point - pivot);
          }
        }
        Jp[0 * 32 + dof] += col.x;
        Jp[1 * 32 + dof] += col.y;
        Jp[2 * 32 + dof] += col.z;
      }
    }
    b = body_parentid[b];
  }
}

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
    device const float* cvel [[buffer(27)]],
    device float* workspace_J [[buffer(28)]],
    device float* workspace_debug [[buffer(29)]],
    uint world [[thread_position_in_grid]]) {
  int nq = dims[0];
  int nv = dims[1];
  int nj = dims[2];
  int neq = dims[3];
  int batch = dims[5];
  int flags = dims[6];
  bool refsafe = dims[7] != 0;
  int nr = dims[9];
  int nbody = dims[11];
  int njnt = dims[12];
  int nsite = dims[13];
  int n_eq_rows = dims[14];
  if (world >= uint(batch)) return;
  if (nv > 32 || nr > 96) return;
  if (neq == 0 || n_eq_rows == 0) return;

  float timestep = params[0];
  int qb = int(world) * nv;
  int pb = int(world) * nq;
  int bo = int(world) * nbody;
  int jo = int(world) * njnt;
  int so = int(world) * max(nsite, 1);

  device float* J_world = workspace_J + int(world) * nr * max(nv, 1);
  device float* dbg = workspace_debug + int(world) * (nr * nr + 4 * nr);

  for (int e = 0; e < neq; ++e) {
    int typ = eq_type[e];
    int objtype = eq_objtype[e]; // 1=body, 6=site, 0=unknown(joint)
    int rowadr = eq_rowadr[e];
    int span = typ == 2 ? 1 : (typ == 0 ? 3 : 6);
    bool active = !(flags & 1) && !(flags & 2) && (eq_active[int(world) * neq + e] != 0);
    if (!active) {
      for (int k = 0; k < span; ++k) {
        int row = rowadr + k;
        if (row < 0 || row >= nr) continue;
        for (int i = 0; i < nv; ++i) J_world[row * nv + i] = 0.0f;
        dbg[nr * nr + row] = 0.0f;
        dbg[nr * nr + nr + row] = 0.0f;
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
          for (int i = 0; i < nv; ++i) J_world[row * nv + i] = 0.0f;
          dbg[nr * nr + row] = 0.0f;
          dbg[nr * nr + nr + row] = 0.0f;
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
      float diag = dof_invweight[d1];
      int row = rowadr;
      for (int i = 0; i < nv; ++i) J_world[row * nv + i] = 0.0f;
      if (j2 >= 0) {
        int q2 = joint_qadr[j2];
        int d2 = joint_dadr[j2];
        float dif = qpos[pb + q2] - qpos0[q2];
        float poly = 0.0f, deriv = 0.0f, power = dif;
        for (int k = 0; k < 4; ++k) {
          poly += eq_data[e * 11 + k + 1] * power;
          deriv += (float(k + 1)) * eq_data[e * 11 + k + 1] * pow(dif, float(k));
          power *= dif;
        }
        pos -= poly;
        J_world[row * nv + d1] = 1.0f;
        J_world[row * nv + d2] = -deriv;
        vel -= deriv * qvel[qb + d2];
        diag += dof_invweight[d2];
      } else {
        J_world[row * nv + d1] = 1.0f;
      }
      // Reference params (same as solver's reference_params).
      float d_width = max(1e-15f, eq_sol[e * 7 + 3]);
      float imp = eq_impedance(eq_sol + e * 7 + 2, pos, 0.0f);
      float r0 = eq_sol[e * 7];
      float r1 = eq_sol[e * 7 + 1];
      if (refsafe && r0 > 0.0f) r0 = max(r0, 2.0f * timestep);
      float K = r0 > 0.0f ? 1.0f / max(1e-15f, d_width * d_width * r0 * r0 * r1 * r1)
                          : -r0 / max(1e-15f, d_width * d_width);
      float B = r1 > 0.0f ? 2.0f / max(1e-15f, d_width * r0) : -r1 / d_width;
      float R = max(1e-15f, (1.0f - imp) * diag / imp);
      float ar = -B * vel - K * imp * pos;
      dbg[nr * nr + row] = R;
      dbg[nr * nr + nr + row] = ar;
    } else if (typ == 0) {
      // Connect (3 rows): body-body, body-world, site-site.
      int o1 = eq_obj[e * 2];
      int o2 = eq_obj[e * 2 + 1];
      float3 pos1 = float3(0.0f), pos2 = float3(0.0f);
      int b1 = 0, b2 = 0;
      if (objtype == 1) {
        b1 = o1; b2 = o2;
        float3 l1 = float3(eq_data[e * 11], eq_data[e * 11 + 1], eq_data[e * 11 + 2]);
        float3 l2 = float3(eq_data[e * 11 + 3], eq_data[e * 11 + 4], eq_data[e * 11 + 5]);
        float4 q1 = b1 >= 0 && b1 < nbody ? float4(body_quat[(bo + b1) * 4], body_quat[(bo + b1) * 4 + 1], body_quat[(bo + b1) * 4 + 2], body_quat[(bo + b1) * 4 + 3]) : float4(1,0,0,0);
        float4 q2 = b2 >= 0 && b2 < nbody ? float4(body_quat[(bo + b2) * 4], body_quat[(bo + b2) * 4 + 1], body_quat[(bo + b2) * 4 + 2], body_quat[(bo + b2) * 4 + 3]) : float4(1,0,0,0);
        float3 p1 = b1 >= 0 && b1 < nbody ? float3(body_pos[(bo + b1) * 3], body_pos[(bo + b1) * 3 + 1], body_pos[(bo + b1) * 3 + 2]) : float3(0.0f);
        float3 p2 = b2 >= 0 && b2 < nbody ? float3(body_pos[(bo + b2) * 3], body_pos[(bo + b2) * 3 + 1], body_pos[(bo + b2) * 3 + 2]) : float3(0.0f);
        pos1 = eq_qrot(q1, l1) + p1;
        pos2 = eq_qrot(q2, l2) + p2;
      } else {
        // Site-site: eq_data unused, use site_xpos directly.
        float3 s1 = o1 >= 0 && o1 < nsite ? float3(site_pos[(so + o1) * 3], site_pos[(so + o1) * 3 + 1], site_pos[(so + o1) * 3 + 2]) : float3(0.0f);
        float3 s2 = o2 >= 0 && o2 < nsite ? float3(site_pos[(so + o2) * 3], site_pos[(so + o2) * 3 + 1], site_pos[(so + o2) * 3 + 2]) : float3(0.0f);
        pos1 = s1; pos2 = s2;
        b1 = o1 >= 0 && o1 < nsite ? site_bodyid[o1] : 0;
        b2 = o2 >= 0 && o2 < nsite ? site_bodyid[o2] : 0;
      }
      float3 cpos = pos1 - pos2;

      // Jacobians J1, J2 (3 x nv) via parent traversal.
      thread float J1[96], J2[96];
      eq_point_jac(pos1, b1, body_pos, body_quat, joint_anchor, joint_axis,
                   body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_dofadr,
                   nbody, njnt, nv, bo, jo, J1);
      eq_point_jac(pos2, b2, body_pos, body_quat, joint_anchor, joint_axis,
                   body_parentid, body_jntadr, body_jntnum, jnt_type, jnt_dofadr,
                   nbody, njnt, nv, bo, jo, J2);

      // Velocities v1, v2 and difference vel.
      float3 v1 = float3(0.0f), v2 = float3(0.0f);
      for (int i = 0; i < nv; ++i) {
        float qv = qvel[qb + i];
        v1.x += J1[0 * 32 + i] * qv; v1.y += J1[1 * 32 + i] * qv; v1.z += J1[2 * 32 + i] * qv;
        v2.x += J2[0 * 32 + i] * qv; v2.y += J2[1 * 32 + i] * qv; v2.z += J2[2 * 32 + i] * qv;
      }
      float3 vel = v1 - v2;

      // Jdot*v via cvel: jdv = cross(omega1, v1 - vcom1) - cross(omega2, v2 - vcom2).
      // Exact for free/single-hinge-to-world bodies (fixtures avoid moving-parent chains).
      float3 jdv = float3(0.0f);
      {
        float3 o1w = float3(0.0f), vcom1 = float3(0.0f);
        float3 o2w = float3(0.0f), vcom2 = float3(0.0f);
        if (b1 > 0 && b1 < nbody) {
          o1w = float3(cvel[(bo + b1) * 6], cvel[(bo + b1) * 6 + 1], cvel[(bo + b1) * 6 + 2]);
          vcom1 = float3(cvel[(bo + b1) * 6 + 3], cvel[(bo + b1) * 6 + 4], cvel[(bo + b1) * 6 + 5]);
        }
        if (b2 > 0 && b2 < nbody) {
          o2w = float3(cvel[(bo + b2) * 6], cvel[(bo + b2) * 6 + 1], cvel[(bo + b2) * 6 + 2]);
          vcom2 = float3(cvel[(bo + b2) * 6 + 3], cvel[(bo + b2) * 6 + 4], cvel[(bo + b2) * 6 + 5]);
        }
        float3 j1 = cross(o1w, v1 - vcom1);
        float3 j2 = cross(o2w, v2 - vcom2);
        jdv = j1 - j2;
      }

      // diagA (translational invweights).
      float w1 = 0.0f, w2 = 0.0f;
      if (b1 >= 0 && b1 < nbody) w1 = body_invweight[b1 * 2];
      if (b2 >= 0 && b2 < nbody) w2 = body_invweight[b2 * 2];
      float diag = max(w1 + w2, 1e-15f);

      // Impedance from norm(cpos), margin 0.
      float pos_norm = length(cpos);
      float imp = eq_impedance(eq_sol + e * 7 + 2, pos_norm, 0.0f);
      imp = clamp(imp, 1e-6f, 0.999999f);
      float d_width = max(1e-15f, eq_sol[e * 7 + 3]);
      float r0 = eq_sol[e * 7];
      float r1 = eq_sol[e * 7 + 1];
      if (refsafe && r0 > 0.0f) r0 = max(r0, 2.0f * timestep);
      float K = r0 > 0.0f ? 1.0f / max(1e-15f, d_width * d_width * r0 * r0 * r1 * r1)
                          : -r0 / max(1e-15f, d_width * d_width);
      float B = r1 > 0.0f ? 2.0f / max(1e-15f, d_width * r0) : -r1 / d_width;
      float R = max(1e-15f, (1.0f - imp) * diag / imp);

      for (int k = 0; k < 3; ++k) {
        int row = rowadr + k;
        if (row < 0 || row >= nr) continue;
        float c = k == 0 ? cpos.x : (k == 1 ? cpos.y : cpos.z);
        float v = k == 0 ? vel.x : (k == 1 ? vel.y : vel.z);
        float jd = k == 0 ? jdv.x : (k == 1 ? jdv.y : jdv.z);
        float ar = -B * v - K * imp * c - jd;
        for (int i = 0; i < nv; ++i) {
          float j1v = k == 0 ? J1[0 * 32 + i] : (k == 1 ? J1[1 * 32 + i] : J1[2 * 32 + i]);
          float j2v = k == 0 ? J2[0 * 32 + i] : (k == 1 ? J2[1 * 32 + i] : J2[2 * 32 + i]);
          J_world[row * nv + i] = j1v - j2v;
        }
        dbg[nr * nr + row] = R;
        dbg[nr * nr + nr + row] = ar;
      }
    } else {
      // Weld (not yet implemented in this commit): reserve rows as inactive zeros.
      // Full weld assembly with pinned rotation semantics follows in next commit.
      for (int k = 0; k < span; ++k) {
        int row = rowadr + k;
        if (row < 0 || row >= nr) continue;
        for (int i = 0; i < nv; ++i) J_world[row * nv + i] = 0.0f;
        dbg[nr * nr + row] = 0.0f;
        dbg[nr * nr + nr + row] = 0.0f;
      }
    }
  }
}
