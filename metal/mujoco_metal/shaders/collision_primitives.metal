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

#pragma once
#include <metal_stdlib>
using namespace metal;

#define MJ_MINVAL 1e-15f
#define MJBOXBOX_MAXVERT 12
#define MJBOXBOX_SEPEPS 1e-6f
#define MJBOXBOX_PAREPS 1e-7f
#define MJBOXBOX_SGNEPS 1e-5f
#define MJBOXBOX_DUPEPS 1e-10f
#define MJBOXBOX_EDGEBIAS 1e-6f

struct ContactGeom {
  float dist;
  float3 pos;
  float3 normal; // points from geom1 to geom2
  float3 t1;
  float3 t2;
};

inline float3 rotate_q(float4 q, float3 v) {
  float3 u = q.yzw;
  return v + 2.0f * cross(u, cross(u, v) + q.x * v);
}

inline float3x3 quat_to_mat(float4 q) {
  return float3x3(
      rotate_q(q, float3(1.0f, 0.0f, 0.0f)),
      rotate_q(q, float3(0.0f, 1.0f, 0.0f)),
      rotate_q(q, float3(0.0f, 0.0f, 1.0f))
  );
}

inline float mju_clip(float x, float min_val, float max_val) {
  return clamp(x, min_val, max_val);
}

inline void make_frame(float3 n, thread float3& t1, thread float3& t2) {
  if (length_squared(t1) < 0.25f) {
    if (n.y < 0.5f && n.y > -0.5f) {
      t1 = float3(0.0f, 1.0f, 0.0f);
    } else {
      t1 = float3(0.0f, 0.0f, 1.0f);
    }
  }
  t1 = t1 - n * dot(n, t1);
  float len = length(t1);
  if (len > 1e-12f) {
    t1 = t1 / len;
  } else {
    t1 = (n.y < 0.5f && n.y > -0.5f) ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 1.0f);
    t1 = normalize(t1 - n * dot(n, t1));
  }
  t2 = cross(n, t1);
}

// 1. Plane - Sphere
inline int collide_plane_sphere(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float3 n = rotate_q(q1, float3(0.0f, 0.0f, 1.0f));
  float cdist = dot(p2 - p1, n);
  float r2 = sz2.x;
  if (cdist > margin + r2) return 0;

  con[0].dist = cdist - r2;
  con[0].normal = n;
  con[0].pos = p2 - n * (con[0].dist * 0.5f + r2);
  con[0].t1 = float3(0.0f);
  make_frame(con[0].normal, con[0].t1, con[0].t2);
  return 1;
}

// 2. Sphere - Sphere
inline int collide_sphere_sphere(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float r1 = sz1.x;
  float r2 = sz2.x;
  float3 delta = p2 - p1;
  float dsq = dot(delta, delta);
  float min_dist = margin + r1 + r2;
  if (dsq > min_dist * min_dist) return 0;

  float dist_val = sqrt(dsq);
  float3 n;
  if (dist_val > 1e-12f) {
    n = delta / dist_val;
  } else {
    float3 z1 = rotate_q(q1, float3(0.0f, 0.0f, 1.0f));
    float3 z2 = rotate_q(q2, float3(0.0f, 0.0f, 1.0f));
    n = cross(z1, z2);
    float nlen = length(n);
    if (nlen > 1e-12f) n = n / nlen;
    else n = float3(1.0f, 0.0f, 0.0f);
  }
  con[0].dist = dist_val - r1 - r2;
  con[0].normal = n;
  con[0].pos = p1 + n * (r1 + con[0].dist * 0.5f);
  con[0].t1 = float3(0.0f);
  make_frame(con[0].normal, con[0].t1, con[0].t2);
  return 1;
}

// 3. Plane - Capsule
inline int collide_plane_capsule(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float r2 = sz2.x;
  float h2 = sz2.y;
  float3 axis = rotate_q(q2, float3(0.0f, 0.0f, 1.0f));
  float3 seg = axis * h2;

  int ncon = 0;
  // Endpoint 1
  float3 ep1 = p2 + seg;
  int n1 = collide_plane_sphere(p1, q1, sz1, ep1, q2, float3(r2, 0, 0), margin, con + ncon);
  if (n1 > 0) {
    con[ncon].t1 = axis;
    make_frame(con[ncon].normal, con[ncon].t1, con[ncon].t2);
    ncon += n1;
  }
  // Endpoint 2
  float3 ep2 = p2 - seg;
  int n2 = collide_plane_sphere(p1, q1, sz1, ep2, q2, float3(r2, 0, 0), margin, con + ncon);
  if (n2 > 0) {
    con[ncon].t1 = axis;
    make_frame(con[ncon].normal, con[ncon].t1, con[ncon].t2);
    ncon += n2;
  }
  return ncon;
}

// 4. Sphere - Capsule
inline int collide_sphere_capsule(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float r2 = sz2.x;
  float h2 = sz2.y;
  float3 axis = rotate_q(q2, float3(0.0f, 0.0f, 1.0f));
  float3 vec = p1 - p2;
  float x = mju_clip(dot(axis, vec), -h2, h2);
  float3 nearest_p2 = p2 + axis * x;
  return collide_sphere_sphere(p1, q1, sz1, nearest_p2, q2, float3(r2, 0, 0), margin, con);
}

// 5. Capsule - Capsule
inline int collide_capsule_capsule(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float3 axis1 = rotate_q(q1, float3(0.0f, 0.0f, 1.0f)) * sz1.y;
  float3 axis2 = rotate_q(q2, float3(0.0f, 0.0f, 1.0f)) * sz2.y;
  float3 dif = p1 - p2;

  float ma = dot(axis1, axis1);
  float mb = -dot(axis1, axis2);
  float mc = dot(axis2, axis2);
  float u = -dot(axis1, dif);
  float v = dot(axis2, dif);
  float det = ma * mc - mb * mb;
  float scale = ma * mc;
  if (scale > 1e-15f && abs(det) >= 1e-6f * scale) {
    float x1 = (mc * u - mb * v) / det;
    float x2 = (ma * v - mb * u) / det;
    if (x1 > 1.0f) {
      x1 = 1.0f;
      x2 = (v - mb) / mc;
    } else if (x1 < -1.0f) {
      x1 = -1.0f;
      x2 = (v + mb) / mc;
    }
    if (x2 > 1.0f) {
      x2 = 1.0f;
      x1 = mju_clip((u - mb) / ma, -1.0f, 1.0f);
    } else if (x2 < -1.0f) {
      x2 = -1.0f;
      x1 = mju_clip((u + mb) / ma, -1.0f, 1.0f);
    }
    float3 vec1 = p1 + axis1 * x1;
    float3 vec2 = p2 + axis2 * x2;
    return collide_sphere_sphere(vec1, q1, float3(sz1.x, 0, 0), vec2, q2, float3(sz2.x, 0, 0), margin, con);
  } else {
    int n1 = collide_sphere_sphere(p1 + axis1, q1, float3(sz1.x, 0, 0),
                                  p2 + axis2 * mju_clip((v - mb) / mc, -1.0f, 1.0f), q2, float3(sz2.x, 0, 0), margin, con);
    int n2 = collide_sphere_sphere(p1 - axis1, q1, float3(sz1.x, 0, 0),
                                  p2 + axis2 * mju_clip((v + mb) / mc, -1.0f, 1.0f), q2, float3(sz2.x, 0, 0), margin, con + n1);
    if (n1 + n2 >= 2) return n1 + n2;

    int n3 = collide_sphere_sphere(p1 + axis1 * mju_clip((u - mb) / ma, -1.0f, 1.0f), q1, float3(sz1.x, 0, 0),
                                  p2 + axis2, q2, float3(sz2.x, 0, 0), margin, con + n1 + n2);
    if (n1 + n2 + n3 >= 2) return n1 + n2 + n3;

    int n4 = collide_sphere_sphere(p1 + axis1 * mju_clip((u + mb) / ma, -1.0f, 1.0f), q1, float3(sz1.x, 0, 0),
                                  p2 - axis2, q2, float3(sz2.x, 0, 0), margin, con + n1 + n2 + n3);
    return n1 + n2 + n3 + n4;
  }
}

// 6. Plane - Box
inline int collide_plane_box(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float3 norm = rotate_q(q1, float3(0.0f, 0.0f, 1.0f));
  float3 dif = p2 - p1;
  float dist = dot(dif, norm);

  int cnt = 0;
  for (int i = 0; i < 8; ++i) {
    float3 vec = float3(
        (i & 1) ? sz2.x : -sz2.x,
        (i & 2) ? sz2.y : -sz2.y,
        (i & 4) ? sz2.z : -sz2.z
    );
    float3 corner = rotate_q(q2, vec);
    float ldist = dot(norm, corner);
    if (dist + ldist > margin || ldist > 0.0f) continue;

    con[cnt].dist = dist + ldist;
    con[cnt].normal = norm;
    con[cnt].pos = p2 + corner - norm * (con[cnt].dist * 0.5f);
    con[cnt].t1 = float3(0.0f);
    make_frame(con[cnt].normal, con[cnt].t1, con[cnt].t2);
    if (++cnt >= 4) return 4;
  }
  return cnt;
}

// 7. Sphere - Box
inline int collide_sphere_box(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float r1 = sz1.x;
  float3x3 mat2 = quat_to_mat(q2);
  float3 center = transpose(mat2) * (p1 - p2);
  float3 clamped = clamp(center, -sz2, sz2);
  float3 diff = clamped - center;
  float dist = length(diff);

  if (dist - r1 > margin) return 0;

  float3 pos, normal;
  if (dist <= MJ_MINVAL) {
    float closest = (sz2.x + sz2.y + sz2.z) * 2.0f;
    int k = 0;
    for (int i = 0; i < 6; ++i) {
      float sign_val = (i % 2) ? 1.0f : -1.0f;
      int axis = i / 2;
      float d_face = abs(sign_val * sz2[axis] - center[axis]);
      if (d_face < closest) {
        closest = d_face;
        k = i;
      }
    }
    float3 nearest = float3(0.0f);
    nearest[k / 2] = (k % 2) ? -1.0f : 1.0f;
    pos = center + nearest * ((r1 - closest) * 0.5f);
    normal = mat2 * nearest;
    dist = -closest;
  } else {
    float3 unit = diff / dist;
    float3 deepest = center + unit * r1;
    pos = 0.5f * (clamped + deepest);
    normal = mat2 * unit;
  }

  con[0].pos = p2 + mat2 * pos;
  con[0].dist = dist - r1;
  con[0].normal = normal;
  con[0].t1 = float3(0.0f);
  make_frame(con[0].normal, con[0].t1, con[0].t2);
  return 1;
}

// 8. Capsule - Box
inline int collide_capsule_box(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float3x3 mat2 = quat_to_mat(q2);
  float3x3 mat2T = transpose(mat2);
  float3 pos = mat2T * (p1 - p2);

  float3 a1 = rotate_q(q1, float3(0.0f, 0.0f, 1.0f));
  float3 axis = mat2T * a1;
  float halflength = sz1.y;
  float3 halfaxis = axis * halflength;

  int axisdir = 0;
  if (halfaxis.x > 0.0f) axisdir += 1;
  if (halfaxis.y > 0.0f) axisdir += 2;
  if (halfaxis.z > 0.0f) axisdir += 4;

  float bestdistmax = margin + 2.0f * (sz1.x + halflength + sz2.x + sz2.y + sz2.z);
  float bestdist = bestdistmax;
  float bestsegmentpos = 0.0f;
  float secondpos = -4.0f;
  float bestboxpos = 0.0f;
  int cltype = -4;
  int clface = -1;
  int clcorner = 0;
  int cledge = 0;

  // 1. Face tests
  for (int i = -1; i <= 1; i += 2) {
    float3 tmp1 = pos + halfaxis * float(i);
    float3 tmp2 = tmp1;
    int c1 = 0, c2 = -1;
    for (int j = 0; j < 3; ++j) {
      if (tmp1[j] < -sz2[j]) { c1++; c2 = j; tmp1[j] = -sz2[j]; }
      else if (tmp1[j] > sz2[j]) { c1++; c2 = j; tmp1[j] = sz2[j]; }
    }
    if (c1 > 1) continue;
    float3 diff = tmp1 - tmp2;
    float dist = dot(diff, diff);
    if (dist < bestdist) {
      bestdist = dist;
      bestsegmentpos = float(i);
      cltype = -2 + i;
      clface = c2;
    }
  }

  // 2. Edge tests
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 8; ++i) {
      if ((i & (1 << j)) != 0) continue;
      float3 tmp3 = float3(
          ((i & 1) ? 1.0f : -1.0f) * sz2.x,
          ((i & 2) ? 1.0f : -1.0f) * sz2.y,
          ((i & 4) ? 1.0f : -1.0f) * sz2.z
      );
      tmp3[j] = 0.0f;

      float3 dif = tmp3 - pos;
      float ma = sz2[j] * sz2[j];
      float mb = -sz2[j] * halfaxis[j];
      float mc = halflength * halflength;
      float u = -sz2[j] * dif[j];
      float v = dot(halfaxis, dif);
      float det = ma * mc - mb * mb;
      if (abs(det) < MJ_MINVAL) continue;
      float idet = 1.0f / det;

      float x1 = (mc * u - mb * v) * idet;
      float x2 = (ma * v - mb * u) * idet;
      int s1 = 1, s2 = 1;
      if (x1 > 1.0f) { x1 = 1.0f; s1 = 2; x2 = (v - mb) / mc; }
      else if (x1 < -1.0f) { x1 = -1.0f; s1 = 0; x2 = (v + mb) / mc; }

      if (x2 > 1.0f) {
        x2 = 1.0f; s2 = 2; x1 = (u - mb) / ma;
        if (x1 > 1.0f) { x1 = 1.0f; s1 = 2; }
        else if (x1 < -1.0f) { x1 = -1.0f; s1 = 0; }
      } else if (x2 < -1.0f) {
        x2 = -1.0f; s2 = 0; x1 = (u + mb) / ma;
        if (x1 > 1.0f) { x1 = 1.0f; s1 = 2; }
        else if (x1 < -1.0f) { x1 = -1.0f; s1 = 0; }
      }

      dif = tmp3 - pos - halfaxis * x2;
      dif[j] += sz2[j] * x1;
      float dist_sq = dot(dif, dif);
      int c1 = s1 * 3 + s2;
      if (dist_sq < bestdist - MJ_MINVAL) {
        bestdist = dist_sq;
        bestsegmentpos = x2;
        bestboxpos = x1;
        int c2 = c1 / 6;
        clcorner = i + (1 << j) * c2;
        cledge = j;
        cltype = c1;
      }
    }
  }

  if (cltype == -4) return 0;

  // 3. Second contact point determination
  if (cltype >= 0 && cltype / 3 != 1) {
    int c1 = axisdir ^ clcorner;
    if (c1 != 0 && c1 != 7) {
      int mul = (c1 == 1 || c1 == 2 || c1 == 4) ? 1 : -1;
      if (mul == -1) c1 = 7 - c1;
      int ax = (c1 == 1) ? 0 : (c1 == 2) ? 1 : 2;
      int ax1 = (c1 == 1) ? 1 : (c1 == 2) ? 2 : 0;
      int ax2 = (c1 == 1) ? 2 : (c1 == 2) ? 0 : 1;
      if (axis[ax] * axis[ax] > 0.5f) {
        float m = 2.0f * sz2[ax] / max(abs(halfaxis[ax]), 1e-15f);
        secondpos = min(1.0f - float(mul) * bestsegmentpos, m);
      } else {
        float m = 2.0f * min(sz2[ax1] / max(abs(halfaxis[ax1]), 1e-15f), sz2[ax2] / max(abs(halfaxis[ax2]), 1e-15f));
        secondpos = -min(1.0f + float(mul) * bestsegmentpos, m);
      }
      secondpos *= float(mul);
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {
    int c1 = (axisdir ^ clcorner) & (7 - (1 << cledge));
    if (c1 == 1 || c1 == 2 || c1 == 4) {
      int ax = cledge;
      int ax1 = (cledge == 0) ? 1 : (cledge == 1) ? 2 : 0;
      int ax2 = (cledge == 0) ? 2 : (cledge == 1) ? 0 : 1;
      if (abs(axis[ax1]) > abs(axis[ax2])) { int tmp = ax1; ax1 = ax2; ax2 = tmp; }
      int mul = (c1 & (1 << ax2)) ? 1 : -1;
      secondpos = (mul == 1) ? (1.0f - bestsegmentpos) : (1.0f + bestsegmentpos);
      float e1 = 2.0f * sz2[ax2] / max(abs(halfaxis[ax2]), 1e-15f);
      secondpos = min(e1, secondpos);
      float e2 = ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0) ? (1.0f - bestboxpos) : (1.0f + bestboxpos);
      e1 = sz2[ax] * e2 / max(abs(halfaxis[ax]), 1e-15f);
      secondpos = min(e1, secondpos);
      secondpos *= float(mul);
    }
  } else if (cltype < 0) {
    if (clface != -1) {
      int mul = (cltype == -3) ? 1 : -1;
      secondpos = 2.0f;
      float3 tmp1 = pos - halfaxis * float(mul);
      for (int i = 0; i < 3; ++i) {
        if (i != clface) {
          float ha_r = float(mul) / (abs(halfaxis[i]) > 1e-15f ? halfaxis[i] : (halfaxis[i] >= 0 ? 1e-15f : -1e-15f));
          float e1 = (sz2[i] - tmp1[i]) * ha_r;
          if (e1 > 0.0f && e1 < secondpos) secondpos = e1;
          e1 = (-sz2[i] - tmp1[i]) * ha_r;
          if (e1 > 0.0f && e1 < secondpos) secondpos = e1;
        }
      }
      secondpos *= float(mul);
    }
  }

  // 4. Contact 1
  float3 pt1 = p1 + a1 * (halflength * bestsegmentpos);
  int n = collide_sphere_box(pt1, q1, float3(sz1.x, 0, 0), p2, q2, sz2, margin, con);

  // 5. Contact 2
  if (secondpos > -3.0f) {
    float3 pt2 = p1 + a1 * (halflength * (secondpos + bestsegmentpos));
    n += collide_sphere_box(pt2, q1, float3(sz1.x, 0, 0), p2, q2, sz2, margin, con + n);
  }
  return n;
}

// 9. Box - Box
inline int clipHalfPlane(int nin, thread float in[MJBOXBOX_MAXVERT][3], thread float out[MJBOXBOX_MAXVERT][3], int coord, float sign_val, float limit) {
  float d[MJBOXBOX_MAXVERT];
  bool all_inside = true;
  for (int k = 0; k < nin; ++k) {
    d[k] = sign_val * in[k][coord] - limit;
    if (d[k] > 0.0f) all_inside = false;
  }
  if (all_inside) {
    for (int k = 0; k < nin; ++k) {
      out[k][0] = in[k][0]; out[k][1] = in[k][1]; out[k][2] = in[k][2];
    }
    return nin;
  }
  int nout = 0;
  for (int k = 0; k < nin; ++k) {
    int k1 = (k + 1 == nin) ? 0 : k + 1;
    float dp = d[k], dq = d[k1];
    if (dp <= 0.0f && nout < MJBOXBOX_MAXVERT) {
      out[nout][0] = in[k][0]; out[nout][1] = in[k][1]; out[nout][2] = in[k][2];
      nout++;
    }
    if (((dp < 0.0f && dq > 0.0f) || (dp > 0.0f && dq < 0.0f)) && nout < MJBOXBOX_MAXVERT) {
      float t = dp / (dp - dq);
      out[nout][0] = in[k][0] + t * (in[k1][0] - in[k][0]);
      out[nout][1] = in[k][1] + t * (in[k1][1] - in[k][1]);
      out[nout][2] = in[k][2] + t * (in[k1][2] - in[k][2]);
      nout++;
    }
  }
  return nout;
}

inline int collide_box_box(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float3x3 mat1 = quat_to_mat(q1);
  float3x3 mat2 = quat_to_mat(q2);

  float3 pos21 = transpose(mat1) * (p2 - p1);
  float3 pos12 = transpose(mat2) * (p1 - p2);

  float rot[9], rotabs[9];
  float3x3 R = transpose(mat1) * mat2;
  // Store rot in row-major: rot[3*r + c] = R[c][r] (since R[col][row])
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      rot[3 * r + c] = R[c][r];
      rotabs[3 * r + c] = abs(rot[3 * r + c]);
    }
  }

  float septol = margin + MJBOXBOX_SEPEPS * (sz1.x + sz1.y + sz1.z + sz2.x + sz2.y + sz2.z);
  float sep_best = -1e30f;
  float sep_face = -1e30f;
  int code = -1;

  // Face axes of box 1
  for (int i = 0; i < 3; ++i) {
    float radius2 = rotabs[3 * i + 0] * sz2.x + rotabs[3 * i + 1] * sz2.y + rotabs[3 * i + 2] * sz2.z;
    float sep = abs(pos21[i]) - sz1[i] - radius2;
    if (sep > septol) return 0;
    if (sep > sep_best) { sep_best = sep; code = i; }
  }

  // Face axes of box 2
  for (int j = 0; j < 3; ++j) {
    float radius1 = rotabs[0 + j] * sz1.x + rotabs[3 + j] * sz1.y + rotabs[6 + j] * sz1.z;
    float sep = abs(pos12[j]) - sz2[j] - radius1;
    if (sep > septol) return 0;
    if (sep > sep_best) { sep_best = sep; code = 3 + j; }
  }
  sep_face = sep_best;
  int code_face = code;

  // Edge-cross axes
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      float ax1 = -rot[3 * i2 + j];
      float ax2 = rot[3 * i1 + j];
      float norm2 = ax1 * ax1 + ax2 * ax2;
      if (norm2 < MJBOXBOX_PAREPS) continue;
      float inv = rsqrt(norm2);
      ax1 *= inv; ax2 *= inv;

      float radius1 = sz1[i1] * abs(ax1) + sz1[i2] * abs(ax2);
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      float a2_1 = ax1 * rot[3 * i1 + j1] + ax2 * rot[3 * i2 + j1];
      float a2_2 = ax1 * rot[3 * i1 + j2] + ax2 * rot[3 * i2 + j2];
      float radius2 = sz2[j1] * abs(a2_1) + sz2[j2] * abs(a2_2);

      float sep = abs(ax1 * pos21[i1] + ax2 * pos21[i2]) - radius1 - radius2;
      if (sep > septol) return 0;
      if (sep - MJBOXBOX_EDGEBIAS * abs(sep) > sep_best && sep > sep_face) {
        sep_best = sep;
        code = 6 + 3 * i + j;
      }
    }
  }

  if (code < 0) return 0;

  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    float3 ax = float3(0.0f);
    ax[i1] = -rot[3 * i2 + j];
    ax[i2] = rot[3 * i1 + j];
    ax = normalize(ax);
    float face_dot;
    if (code_face < 3) {
      face_dot = abs(ax[code_face]);
    } else {
      int f = code_face - 3;
      face_dot = abs(ax[0] * rot[0 + f] + ax[1] * rot[3 + f] + ax[2] * rot[6 + f]);
    }
    if (face_dot > 0.99f && sep_best < sep_face + 0.05f * abs(sep_face) + MJ_MINVAL) {
      code = code_face;
      sep_best = sep_face;
    }
  }

  // Stage 2a: Edge-edge contact
  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    int j1 = (j + 1) % 3, j2 = (j + 2) % 3;

    float3 axis = float3(0.0f);
    axis[i1] = -rot[3 * i2 + j];
    axis[i2] = rot[3 * i1 + j];
    axis = normalize(axis);
    if (dot(axis, pos21) < 0.0f) axis = -axis;

    float a2[3] = {
      axis[0] * rot[0 + 0] + axis[1] * rot[3 + 0] + axis[2] * rot[6 + 0],
      axis[0] * rot[0 + 1] + axis[1] * rot[3 + 1] + axis[2] * rot[6 + 1],
      axis[0] * rot[0 + 2] + axis[1] * rot[3 + 2] + axis[2] * rot[6 + 2],
    };
    int amb1 = -1, amb2 = -1;
    if (abs(axis[i1]) < MJBOXBOX_SGNEPS) amb1 = i1;
    else if (abs(axis[i2]) < MJBOXBOX_SGNEPS) amb1 = i2;
    if (abs(a2[j1]) < MJBOXBOX_SGNEPS) amb2 = j1;
    else if (abs(a2[j2]) < MJBOXBOX_SGNEPS) amb2 = j2;

    float d2[3] = {rot[0 + j], rot[3 + j], rot[6 + j]};
    float b = d2[i];
    float denom = 1.0f - b * b;

    float3 w1 = float3(0.0f), w2 = float3(0.0f);
    float best_d2 = 1e30f;
    for (int v1 = 0; v1 < (amb1 >= 0 ? 2 : 1); ++v1) {
      for (int v2 = 0; v2 < (amb2 >= 0 ? 2 : 1); ++v2) {
        float3 c1 = float3(0.0f);
        c1[i1] = axis[i1] >= 0.0f ? sz1[i1] : -sz1[i1];
        c1[i2] = axis[i2] >= 0.0f ? sz1[i2] : -sz1[i2];
        if (amb1 >= 0 && v1) c1[amb1] = -c1[amb1];

        float3 cc = float3(0.0f);
        cc[j1] = a2[j1] >= 0.0f ? -sz2[j1] : sz2[j1];
        cc[j2] = a2[j2] >= 0.0f ? -sz2[j2] : sz2[j2];
        if (amb2 >= 0 && v2) cc[amb2] = -cc[amb2];

        float3 c2 = float3(
            rot[0] * cc[0] + rot[1] * cc[1] + rot[2] * cc[2] + pos21[0],
            rot[3] * cc[0] + rot[4] * cc[1] + rot[5] * cc[2] + pos21[1],
            rot[6] * cc[0] + rot[7] * cc[1] + rot[8] * cc[2] + pos21[2]
        );

        float3 e = c2 - c1;
        float d1e = e[i];
        float d2e = d2[0] * e[0] + d2[1] * e[1] + d2[2] * e[2];
        float s = denom < MJ_MINVAL ? 0.0f : (d1e - b * d2e) / denom;
        s = mju_clip(s, -sz1[i], sz1[i]);
        float t = mju_clip(b * s - d2e, -sz2[j], sz2[j]);
        s = mju_clip(d1e + b * t, -sz1[i], sz1[i]);

        float3 p1_pt = c1; p1_pt[i] += s;
        float3 p2_pt = c2 + float3(d2[0], d2[1], d2[2]) * t;
        float3 gap = p2_pt - p1_pt;
        float gap2 = dot(gap, gap);
        if (gap2 < best_d2) {
          best_d2 = gap2;
          w1 = p1_pt;
          w2 = p2_pt;
        }
      }
    }

    float3 gap = w2 - w1;
    float dist = dot(gap, axis);
    if (dist > septol) return 0;

    float3 mid = 0.5f * (w1 + w2);
    con[0].dist = dist;
    con[0].pos = p1 + mat1 * mid;
    con[0].normal = mat1 * axis;
    con[0].t1 = float3(0.0f);
    make_frame(con[0].normal, con[0].t1, con[0].t2);
    return 1;
  }

  // Stage 2b: Face contact
  bool ref1 = code < 3;
  int a = ref1 ? code : code - 3;
  float3 sizeref = ref1 ? sz1 : sz2;
  float3 sizeinc = ref1 ? sz2 : sz1;
  float3 posref = ref1 ? p1 : p2;
  float3x3 matref = ref1 ? mat1 : mat2;
  float3 posoi = ref1 ? pos21 : pos12;

  float rinc[9];
  if (ref1) {
    for (int k = 0; k < 9; ++k) rinc[k] = rot[k];
  } else {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) rinc[3 * r + c] = rot[3 * c + r];
    }
  }

  float sgn = posoi[a] >= 0.0f ? 1.0f : -1.0f;
  int binc = 0;
  for (int k = 1; k < 3; ++k) {
    if (abs(rinc[3 * a + k]) > abs(rinc[3 * a + binc])) binc = k;
  }
  float tinc = sgn * rinc[3 * a + binc] > 0.0f ? -1.0f : 1.0f;

  int ax = (a + 1) % 3, ay = (a + 2) % 3;
  int bu = (binc + 1) % 3, bv = (binc + 2) % 3;

  float poly0[MJBOXBOX_MAXVERT][3];
  float poly1[MJBOXBOX_MAXVERT][3];

  float cx[3], du[3], dv[3];
  for (int r = 0; r < 3; ++r) {
    int c = (r == 0) ? ax : ((r == 1) ? ay : a);
    cx[r] = posoi[c] + tinc * sizeinc[binc] * rinc[3 * c + binc];
    du[r] = sizeinc[bu] * rinc[3 * c + bu];
    dv[r] = sizeinc[bv] * rinc[3 * c + bv];
  }
  cx[2] = sgn * cx[2] - sizeref[a];
  du[2] *= sgn;
  dv[2] *= sgn;

  const float corner_sign[4][2] = {{1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, -1.0f}};
  for (int k = 0; k < 4; ++k) {
    float su = corner_sign[k][0], sv = corner_sign[k][1];
    poly0[k][0] = cx[0] + su * du[0] + sv * dv[0];
    poly0[k][1] = cx[1] + su * du[1] + sv * dv[1];
    poly0[k][2] = cx[2] + su * du[2] + sv * dv[2];
  }

  int nvert = 4;
  nvert = clipHalfPlane(nvert, poly0, poly1, 0, 1.0f, sizeref[ax]);
  nvert = clipHalfPlane(nvert, poly1, poly0, 0, -1.0f, sizeref[ax]);
  nvert = clipHalfPlane(nvert, poly0, poly1, 1, 1.0f, sizeref[ay]);
  nvert = clipHalfPlane(nvert, poly1, poly0, 1, -1.0f, sizeref[ay]);

  float accepted[MJBOXBOX_MAXVERT][3];
  int naccept = 0;
  float dupe2 = MJBOXBOX_DUPEPS * (sizeref[ax] * sizeref[ax] + sizeref[ay] * sizeref[ay]);
  for (int k = 0; k < nvert; ++k) {
    if (poly0[k][2] > margin) continue;
    bool dupe = false;
    for (int q = 0; q < naccept; ++q) {
      float dx = accepted[q][0] - poly0[k][0];
      float dy = accepted[q][1] - poly0[k][1];
      if (dx * dx + dy * dy < dupe2) { dupe = true; break; }
    }
    if (!dupe && naccept < 8) {
      accepted[naccept][0] = poly0[k][0];
      accepted[naccept][1] = poly0[k][1];
      accepted[naccept][2] = poly0[k][2];
      naccept++;
    }
  }

  if (naccept == 0) return 0;

  float nsign = ref1 ? sgn : -sgn;
  float3 normal = float3(
      nsign * matref[a][0],
      nsign * matref[a][1],
      nsign * matref[a][2]
  );

  for (int k = 0; k < naccept; ++k) {
    float posc[3];
    posc[ax] = accepted[k][0];
    posc[ay] = accepted[k][1];
    posc[a] = sgn * (sizeref[a] + 0.5f * accepted[k][2]);

    con[k].dist = accepted[k][2];
    con[k].pos = posref + matref * float3(posc[0], posc[1], posc[2]);
    con[k].normal = normal;
    con[k].t1 = float3(0.0f);
    make_frame(con[k].normal, con[k].t1, con[k].t2);
  }
  return naccept;
}

// Unified Pair Dispatcher
inline int collide_pair(
    int type1, float3 p1, float4 q1, float3 sz1,
    int type2, float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  // Types: 0 = plane, 2 = sphere, 3 = capsule, 6 = box
  bool swapped = (type1 > type2);
  int t1 = swapped ? type2 : type1;
  int t2 = swapped ? type1 : type2;
  float3 pos1 = swapped ? p2 : p1;
  float4 quat1 = swapped ? q2 : q1;
  float3 size1 = swapped ? sz2 : sz1;
  float3 pos2 = swapped ? p1 : p2;
  float4 quat2 = swapped ? q1 : q2;
  float3 size2 = swapped ? sz1 : sz2;

  int n = 0;
  if (t1 == 0 && t2 == 2) {
    n = collide_plane_sphere(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 2 && t2 == 2) {
    n = collide_sphere_sphere(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 0 && t2 == 3) {
    n = collide_plane_capsule(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 2 && t2 == 3) {
    n = collide_sphere_capsule(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 3 && t2 == 3) {
    n = collide_capsule_capsule(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 0 && t2 == 6) {
    n = collide_plane_box(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 2 && t2 == 6) {
    n = collide_sphere_box(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 3 && t2 == 6) {
    n = collide_capsule_box(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 6 && t2 == 6) {
    n = collide_box_box(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  }

  // If order was swapped, normal points from pos1 to pos2, which is from original geom2 to geom1.
  // Negate normal to point from original geom1 to geom2!
  if (swapped) {
    for (int k = 0; k < n; ++k) {
      con[k].normal = -con[k].normal;
      con[k].t1 = -con[k].t1;
      con[k].t2 = cross(con[k].normal, con[k].t1);
    }
  }
  return n;
}
