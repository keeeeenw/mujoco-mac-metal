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

// 9. Box - Box (MuJoCo 3.10.0 algorithm matching mjc_BoxBox)
struct RawPreContact {
  float dist;
  float pos[3];
  float normal[3];
  float tangent[3];
};

inline void bb_zero(thread float* v, int n) {
  for (int i = 0; i < n; ++i) v[i] = 0.0f;
}

inline void bb_zero3(thread float* v) { v[0] = 0.0f; v[1] = 0.0f; v[2] = 0.0f; }
inline void bb_copy3(thread float* d, thread const float* s) { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; }
inline void bb_add3(thread float* d, thread const float* a, thread const float* b) {
  d[0] = a[0] + b[0]; d[1] = a[1] + b[1]; d[2] = a[2] + b[2];
}
inline void bb_sub3(thread float* d, thread const float* a, thread const float* b) {
  d[0] = a[0] - b[0]; d[1] = a[1] - b[1]; d[2] = a[2] - b[2];
}
inline void bb_addTo3(thread float* d, thread const float* a) {
  d[0] += a[0]; d[1] += a[1]; d[2] += a[2];
}
inline void bb_scl3(thread float* d, thread const float* a, float s) {
  d[0] = a[0] * s; d[1] = a[1] * s; d[2] = a[2] * s;
}
inline void bb_addToScl3(thread float* d, thread const float* a, float s) {
  d[0] += a[0] * s; d[1] += a[1] * s; d[2] += a[2] * s;
}
inline float bb_dot3(thread const float* a, thread const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline float bb_normalize3(thread float* v) {
  float n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (n > 1e-15f) {
    float inv = 1.0f / n;
    v[0] *= inv; v[1] *= inv; v[2] *= inv;
  }
  return n;
}
inline void bb_mulMatVec3(thread float* res, thread const float* mat, thread const float* vec) {
  float tmp[3] = {
    mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
    mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
    mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2]
  };
  res[0] = tmp[0]; res[1] = tmp[1]; res[2] = tmp[2];
}
inline void bb_mulMatTVec3(thread float* res, thread const float* mat, thread const float* vec) {
  float tmp[3] = {
    mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2],
    mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2],
    mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2]
  };
  res[0] = tmp[0]; res[1] = tmp[1]; res[2] = tmp[2];
}
inline void bb_mulMatTMat3(thread float* res, thread const float* mat1, thread const float* mat2) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      res[3 * i + j] = mat1[0 + i] * mat2[0 + j] + mat1[3 + i] * mat2[3 + j] + mat1[6 + i] * mat2[6 + j];
    }
  }
}
inline void bb_mulMatMatT3(thread float* res, thread const float* mat1, thread const float* mat2) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      res[3 * i + j] = mat1[3 * i + 0] * mat2[3 * j + 0] + mat1[3 * i + 1] * mat2[3 * j + 1] + mat1[3 * i + 2] * mat2[3 * j + 2];
    }
  }
}
inline void bb_transpose3(thread float* res, thread const float* mat) {
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      res[c * 3 + r] = mat[r * 3 + c];
    }
  }
}
inline int bb_outsideBox(thread const float point[3], thread const float pos[3], thread const float mat[9],
                         thread const float size[3], float inflate) {
  float vec[3] = {point[0] - pos[0], point[1] - pos[1], point[2] - pos[2]};
  bb_mulMatTVec3(vec, mat, vec);
  float big[3] = {size[0] * inflate, size[1] * inflate, size[2] * inflate};
  if (vec[0] > big[0] || vec[0] < -big[0] ||
      vec[1] > big[1] || vec[1] < -big[1] ||
      vec[2] > big[2] || vec[2] < -big[2]) {
    return 1;
  }
  if (inflate == 1.0f) return -1;
  float small[3] = {size[0] / inflate, size[1] / inflate, size[2] / inflate};
  if (vec[0] < small[0] && vec[0] > -small[0] &&
      vec[1] < small[1] && vec[1] > -small[1] &&
      vec[2] < small[2] && vec[2] > -small[2]) {
    return -1;
  }
  return 0;
}

inline void quat_to_mat_row_major(float4 q, thread float mat[9]) {
  float w = q.x, x = q.y, y = q.z, z = q.w;
  mat[0] = 1.0f - 2.0f * (y * y + z * z);
  mat[1] = 2.0f * (x * y - w * z);
  mat[2] = 2.0f * (x * z + w * y);

  mat[3] = 2.0f * (x * y + w * z);
  mat[4] = 1.0f - 2.0f * (x * x + z * z);
  mat[5] = 2.0f * (y * z - w * x);

  mat[6] = 2.0f * (x * z - w * y);
  mat[7] = 2.0f * (y * z + w * x);
  mat[8] = 1.0f - 2.0f * (x * x + y * y);
}

inline int _boxbox_metal(
    thread RawPreContact* con,
    thread const float pos1[3], thread const float mat1[9], thread const float size1[3],
    thread const float pos2[3], thread const float mat2[9], thread const float size2[3],
    float margin) {
  float pos12[3], pos21[3], rot[9], rott[9], rotabs[9], rottabs[9], tmp1[3], tmp2[3], plen1[3], plen2[3];
  float rotmore[9], p[3], r[9], s[3], ss[3], lp[3], rt[9], points[8][3];
  float depth[8], pts[6][3], ppts2[4][2], pu[4][3], axi[3][3];
  float linesu[4][6], lines[4][6], clnorm[3], rnorm[3];
  float penetration, c1, c2, c3, a, b, c, d, lx, ly, hz, l, x, y, u, v, llx, lly, innorm, margin2;

  int i0 = 0, i1 = 1, i2 = 2;
  float f0 = 1.0f, f1 = 1.0f, f2 = 1.0f;
  int i, j, q, code = -1, q1, q2, clcorner, n = 0, m, k;
  int cle1 = 0, cle2 = 0, in = 0, ax1 = 0, ax2 = 0, pax1 = 0, pax2 = 0, clface = 0, nl, nf;

  margin2 = margin * margin;

  bb_sub3(tmp1, pos2, pos1);
  bb_mulMatTVec3(pos21, mat1, tmp1);

  bb_sub3(tmp1, pos1, pos2);
  bb_mulMatTVec3(pos12, mat2, tmp1);

  bb_mulMatTMat3(rot, mat1, mat2);
  bb_transpose3(rott, rot);

  for (i = 0; i < 9; i++) rotabs[i] = abs(rot[i]);
  for (i = 0; i < 9; i++) rottabs[i] = abs(rott[i]);

  bb_mulMatVec3(plen2, rotabs, size2);
  bb_mulMatTVec3(plen1, rotabs, size1);

  for (i = 0, penetration = margin; i < 3; i++)
    penetration += size1[i] * 3.0f + size2[i] * 3.0f;

  for (i = 0; i < 3; i++) {
    c1 = -abs(pos21[i]) + size1[i] + plen2[i];
    c2 = -abs(pos12[i]) + size2[i] + plen1[i];

    if (c1 < -margin || c2 < -margin)
      return 0;

    if (c1 < penetration) {
      penetration = c1;
      code = i + 3 * (pos21[i] < 0.0f) + 0;
    }
    if (c2 < penetration) {
      penetration = c2;
      code = i + 3 * (pos12[i] < 0.0f) + 6;
    }
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      bb_zero3(tmp2);
      if (i == 0) {
        tmp2[1] = -rott[3 * j + 2];
        tmp2[2] = +rott[3 * j + 1];
      } else if (i == 1) {
        tmp2[0] = +rott[3 * j + 2];
        tmp2[2] = -rott[3 * j + 0];
      } else if (i == 2) {
        tmp2[0] = -rott[3 * j + 1];
        tmp2[1] = +rott[3 * j + 0];
      }

      c1 = bb_normalize3(tmp2);
      if (c1 < 1e-15f) continue;

      c2 = bb_dot3(pos21, tmp2);
      c3 = 0.0f;

      for (k = 0; k < 3; k++)
        if (k != i)
          c3 += size1[k] * abs(tmp2[k]);
      for (k = 0; k < 3; k++)
        if (k != j)
          c3 += size2[k] * rotabs[3 * i + 3 - k - j] / c1;

      c3 -= abs(c2);
      if (c3 < -margin) return 0;

      if (c3 < penetration * (1.0f - 1e-6f)) {
        penetration = c3;
        for (k = cle1 = 0; k < 3; k++)
          if (k != i)
            if ((tmp2[k] > 0.0f) ^ (c2 < 0.0f))
              cle1 += 1 << k;
        for (k = cle2 = 0; k < 3; k++)
          if (k != j)
            if ((rot[3 * i + 3 - k - j] > 0.0f) ^ (c2 < 0.0f) ^ ((k - j + 3) % 3 == 1))
              cle2 += 1 << k;

        code = 12 + i * 3 + j;
        bb_copy3(clnorm, tmp2);
        in = c2 < 0.0f;
      }
    }
  }

  if (code == -1) return 0;

  if (code < 12) {
    q1 = code % 6;
    q2 = code / 6;

    bb_zero(rotmore, 9);
    if (q1 == 0)      { rotmore[2] = -1.0f; rotmore[4] = +1.0f; rotmore[6] = +1.0f; }
    else if (q1 == 1) { rotmore[0] = +1.0f; rotmore[5] = -1.0f; rotmore[7] = +1.0f; }
    else if (q1 == 2) { rotmore[0] = +1.0f; rotmore[4] = +1.0f; rotmore[8] = +1.0f; }
    else if (q1 == 3) { rotmore[2] = +1.0f; rotmore[4] = +1.0f; rotmore[6] = -1.0f; }
    else if (q1 == 4) { rotmore[0] = +1.0f; rotmore[5] = +1.0f; rotmore[7] = -1.0f; }
    else if (q1 == 5) { rotmore[0] = -1.0f; rotmore[4] = +1.0f; rotmore[8] = -1.0f; }

    i0 = 0; i1 = 1; i2 = 2;
    f0 = 1.0f; f1 = 1.0f; f2 = 1.0f;
    if (q1 == 0)      { i0 = 2; f0 = -1.0f; i2 = 0; }
    else if (q1 == 1) { i1 = 2; f1 = -1.0f; i2 = 1; }
    else if (q1 == 2) {}
    else if (q1 == 3) { i0 = 2; i2 = 0; f2 = -1.0f; }
    else if (q1 == 4) { i1 = 2; i2 = 1; f2 = -1.0f; }
    else if (q1 == 5) { f0 = -1.0f; f2 = -1.0f; }

    if (q2) {
      bb_mulMatMatT3(r, rotmore, rot);
      p[0] = pos12[i0] * f0; p[1] = pos12[i1] * f1; p[2] = pos12[i2] * f2;
      tmp1[0] = size2[i0] * f0; tmp1[1] = size2[i1] * f1; tmp1[2] = size2[i2] * f2;
      bb_copy3(s, size1);
    } else {
      bb_scl3(r + 0, rot + i0 * 3, f0);
      bb_scl3(r + 3, rot + i1 * 3, f1);
      bb_scl3(r + 6, rot + i2 * 3, f2);
      p[0] = pos21[i0] * f0; p[1] = pos21[i1] * f1; p[2] = pos21[i2] * f2;
      tmp1[0] = size1[i0] * f0; tmp1[1] = size1[i1] * f1; tmp1[2] = size1[i2] * f2;
      bb_copy3(s, size2);
    }

    bb_transpose3(rt, r);
    for (i = 0; i < 3; i++) ss[i] = abs(tmp1[i]);
    lx = ss[0]; ly = ss[1]; hz = ss[2];
    p[2] -= hz;

    bb_copy3(lp, p);
    clcorner = 0;
    for (i = 0; i < 3; i++)
      if (r[6 + i] < 0.0f) clcorner += 1 << i;

    bb_addToScl3(lp, rt + 0, s[0] * ((clcorner & 1) ? 1.0f : -1.0f));
    bb_addToScl3(lp, rt + 3, s[1] * ((clcorner & 2) ? 1.0f : -1.0f));
    bb_addToScl3(lp, rt + 6, s[2] * ((clcorner & 4) ? 1.0f : -1.0f));

    m = 0; k = 0;
    bb_copy3(pts[m++], lp);
    for (i = 0; i < 3; i++)
      if (abs(r[6 + i]) < 0.5f)
        bb_scl3(pts[m++], rt + 3 * i, s[i] * ((clcorner & (1 << i)) ? -2.0f : 2.0f));

    bb_add3(pts[3], pts[0], pts[1]);
    bb_add3(pts[4], pts[0], pts[2]);
    bb_add3(pts[5], pts[3], pts[2]);

    if (m > 1) {
      bb_copy3(lines[k] + 0, pts[0]);
      bb_copy3(lines[k++] + 3, pts[1]);
    }
    if (m > 2) {
      bb_copy3(lines[k] + 0, pts[0]);
      bb_copy3(lines[k++] + 3, pts[2]);
      bb_copy3(lines[k] + 0, pts[3]);
      bb_copy3(lines[k++] + 3, pts[2]);
      bb_copy3(lines[k] + 0, pts[4]);
      bb_copy3(lines[k++] + 3, pts[1]);
    }

    for (i = 0; i < k; i++) {
      for (q = 0; q < 2; q++) {
        a = lines[i][0 + q];
        b = lines[i][3 + q];
        c = lines[i][1 - q];
        d = lines[i][4 - q];
        if (abs(b) > 1e-15f) {
          for (j = -1; j <= 1; j += 2) {
            l = ss[q] * float(j);
            c1 = (l - a) * (1.0f / b);
            if (c1 < 0.0f || c1 > 1.0f) continue;
            c2 = c + d * c1;
            if (abs(c2) > ss[1 - q]) continue;
            if (n < 8) {
              bb_copy3(points[n], lines[i]);
              bb_addToScl3(points[n++], lines[i] + 3, c1);
            }
          }
        }
      }
    }

    a = pts[1][0]; b = pts[2][0];
    c = pts[1][1]; d = pts[2][1];
    c1 = a * d - b * c;

    if (m > 2) {
      for (i = 0; i < 4; i++) {
        llx = (i / 2) ? lx : -lx;
        lly = (i % 2) ? ly : -ly;
        x = llx - pts[0][0];
        y = lly - pts[0][1];
        u = (x * d - y * b) * (1.0f / c1);
        v = (y * a - x * c) * (1.0f / c1);
        if (u <= 0.0f || v <= 0.0f || u >= 1.0f || v >= 1.0f) continue;
        if (n < 8) {
          points[n][0] = llx;
          points[n][1] = lly;
          points[n][2] = (pts[0][2] + u * pts[1][2] + v * pts[2][2]);
          n++;
        }
      }
    }

    for (i = 0; i < (1 << (m - 1)); i++) {
      bb_copy3(tmp1, pts[i == 0 ? 0 : i + 2]);
      if (i) {
        if (tmp1[0] <= -lx || tmp1[0] >= lx) continue;
        if (tmp1[1] <= -ly || tmp1[1] >= ly) continue;
      }
      if (n < 8) {
        bb_copy3(points[n++], tmp1);
      }
    }

    m = n;
    n = 0;
    for (i = 0; i < m; i++) {
      if (points[i][2] > margin) continue;
      if (n != i) bb_copy3(points[n], points[i]);
      depth[n] = points[n][2];
      points[n][2] *= 0.5f;
      n++;
    }

    bb_mulMatMatT3(r, q2 ? mat2 : mat1, rotmore);
    bb_copy3(p, q2 ? pos2 : pos1);

    tmp2[0] = (q2 ? -1.0f : 1.0f) * r[2];
    tmp2[1] = (q2 ? -1.0f : 1.0f) * r[5];
    tmp2[2] = (q2 ? -1.0f : 1.0f) * r[8];

    bb_copy3(con[0].normal, tmp2);
    bb_zero3(con[0].tangent);

    for (i = 0; i < n; i++) {
      con[i].dist = 2.0f * points[i][2];
      points[i][2] += hz;
      bb_mulMatVec3(tmp2, r, points[i]);
      bb_add3(con[i].pos, tmp2, p);
      if (i) {
        bb_copy3(con[i].normal, con[0].normal);
        bb_zero3(con[i].tangent);
      }
    }
    return n;
  } else {
    // Edge - Edge
    code -= 12;
    q1 = code / 3;
    q2 = code % 3;

    if (q2 == 0) { ax1 = 1; ax2 = 2; }
    if (q2 == 1) { ax1 = 0; ax2 = 2; }
    if (q2 == 2) { ax1 = 1; ax2 = 0; }
    if (q1 == 0) { pax1 = 1; pax2 = 2; }
    if (q1 == 1) { pax1 = 0; pax2 = 2; }
    if (q1 == 2) { pax1 = 1; pax2 = 0; }

    if (rotabs[3 * q1 + ax1] < rotabs[3 * q1 + ax2]) {
      ax1 = ax2;
      ax2 = 3 - q2 - ax1;
    }
    if (rottabs[3 * q2 + pax1] < rottabs[3 * q2 + pax2]) {
      pax1 = pax2;
      pax2 = 3 - q1 - pax1;
    }

    if (cle1 & (1 << pax2)) clface = pax2;
    else clface = pax2 + 3;

    bb_zero(rotmore, 9);
    if (clface == 0)      { rotmore[2] = -1.0f; rotmore[4] = +1.0f; rotmore[6] = +1.0f; }
    else if (clface == 1) { rotmore[0] = +1.0f; rotmore[5] = -1.0f; rotmore[7] = +1.0f; }
    else if (clface == 2) { rotmore[0] = +1.0f; rotmore[4] = +1.0f; rotmore[8] = +1.0f; }
    else if (clface == 3) { rotmore[2] = +1.0f; rotmore[4] = +1.0f; rotmore[6] = -1.0f; }
    else if (clface == 4) { rotmore[0] = +1.0f; rotmore[5] = +1.0f; rotmore[7] = -1.0f; }
    else if (clface == 5) { rotmore[0] = -1.0f; rotmore[4] = +1.0f; rotmore[8] = -1.0f; }

    i0 = 0; i1 = 1; i2 = 2;
    f0 = 1.0f; f1 = 1.0f; f2 = 1.0f;
    if (clface == 0)      { i0 = 2; f0 = -1.0f; i2 = 0; }
    else if (clface == 1) { i1 = 2; f1 = -1.0f; i2 = 1; }
    else if (clface == 2) {}
    else if (clface == 3) { i0 = 2; i2 = 0; f2 = -1.0f; }
    else if (clface == 4) { i1 = 2; i2 = 1; f2 = -1.0f; }
    else if (clface == 5) { f0 = -1.0f; f2 = -1.0f; }

    p[0] = pos21[i0] * f0; p[1] = pos21[i1] * f1; p[2] = pos21[i2] * f2;
    rnorm[0] = clnorm[i0] * f0; rnorm[1] = clnorm[i1] * f1; rnorm[2] = clnorm[i2] * f2;

    bb_scl3(r + 0, rot + i0 * 3, f0);
    bb_scl3(r + 3, rot + i1 * 3, f1);
    bb_scl3(r + 6, rot + i2 * 3, f2);

    bb_mulMatTVec3(tmp1, rotmore, size1);
    for (i = 0; i < 3; i++) s[i] = abs(tmp1[i]);
    bb_transpose3(rt, r);

    lx = s[0]; ly = s[1]; hz = s[2];
    p[2] -= hz;

    n = 0;
    bb_copy3(points[n], p);
    bb_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? 1.0f : -1.0f));
    bb_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1.0f : -1.0f));
    bb_copy3(points[n + 1], points[n]);
    bb_addToScl3(points[n], rt + 3 * q2, size2[q2]);
    n = 1;
    bb_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
    n = 2;

    bb_copy3(points[n], p);
    bb_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? -1.0f : 1.0f));
    bb_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1.0f : -1.0f));
    bb_copy3(points[n + 1], points[n]);
    bb_addToScl3(points[n], rt + 3 * q2, size2[q2]);
    n = 3;
    bb_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
    n = 4;

    bb_copy3(axi[0], points[0]);
    bb_sub3(axi[1], points[1], points[0]);
    bb_sub3(axi[2], points[2], points[0]);

    if (abs(rnorm[2]) < 1e-15f) return 0;
    innorm = (1.0f / rnorm[2]) * (in ? -1.0f : 1.0f);

    for (i = 0; i < 4; i++) {
      c1 = -points[i][2] * (1.0f / rnorm[2]);
      bb_copy3(pu[i], points[i]);
      bb_addToScl3(points[i], rnorm, c1);
      ppts2[i][0] = points[i][0];
      ppts2[i][1] = points[i][1];
    }

    bb_copy3(pts[0], points[0]);
    bb_sub3(pts[1], points[1], points[0]);
    bb_sub3(pts[2], points[2], points[0]);

    m = 3; k = 0; n = 0;
    if (m > 1) {
      bb_copy3(lines[k] + 0, pts[0]);
      bb_copy3(lines[k] + 3, pts[1]);
      bb_copy3(linesu[k] + 0, axi[0]);
      bb_copy3(linesu[k++] + 3, axi[1]);
    }
    if (m > 2) {
      bb_copy3(lines[k] + 0, pts[0]);
      bb_copy3(lines[k] + 3, pts[2]);
      bb_copy3(linesu[k] + 0, axi[0]);
      bb_copy3(linesu[k++] + 3, axi[2]);

      bb_add3(lines[k] + 0, pts[0], pts[1]);
      bb_copy3(lines[k] + 3, pts[2]);
      bb_add3(linesu[k] + 0, axi[0], axi[1]);
      bb_copy3(linesu[k++] + 3, axi[2]);

      bb_add3(lines[k] + 0, pts[0], pts[2]);
      bb_copy3(lines[k] + 3, pts[1]);
      bb_add3(linesu[k] + 0, axi[0], axi[2]);
      bb_copy3(linesu[k++] + 3, axi[1]);
    }

    for (i = 0; i < k; i++) {
      for (q = 0; q < 2; q++) {
        a = lines[i][0 + q];
        b = lines[i][3 + q];
        c = lines[i][1 - q];
        d = lines[i][4 - q];
        if (abs(b) > 1e-15f) {
          for (j = -1; j <= 1; j += 2) {
            if (n < 8) {
              l = s[q] * float(j);
              c1 = (l - a) * (1.0f / b);
              if (c1 < 0.0f || c1 > 1.0f) continue;
              c2 = c + d * c1;
              if (abs(c2) > s[1 - q]) continue;
              if ((linesu[i][2] + linesu[i][5] * c1) * innorm > margin) continue;

              bb_scl3(points[n], linesu[i], 0.5f);
              bb_addToScl3(points[n], linesu[i] + 3, 0.5f * c1);
              points[n][0 + q] += 0.5f * l;
              points[n][1 - q] += 0.5f * c2;
              depth[n] = points[n][2] * innorm * 2.0f;
              n++;
            }
          }
        }
      }
    }

    nl = n;
    a = pts[1][0]; b = pts[2][0];
    c = pts[1][1]; d = pts[2][1];
    c1 = a * d - b * c;

    for (i = 0; i < 4; i++) {
      if (n < 8) {
        llx = (i / 2) ? lx : -lx;
        lly = (i % 2) ? ly : -ly;
        x = llx - pts[0][0];
        y = lly - pts[0][1];
        u = (x * d - y * b) * (1.0f / c1);
        v = (y * a - x * c) * (1.0f / c1);
        if (nl == 0) {
          if ((u < 0.0f || u > 1.0f) && (v < 0.0f || v > 1.0f)) continue;
        } else {
          if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) continue;
        }
        if (u < 0.0f) u = 0.0f;
        if (u > 1.0f) u = 1.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;

        bb_scl3(tmp1, pu[0], 1.0f - u - v);
        bb_addToScl3(tmp1, pu[1], u);
        bb_addToScl3(tmp1, pu[2], v);

        points[n][0] = llx;
        points[n][1] = lly;
        points[n][2] = 0.0f;

        bb_sub3(tmp2, points[n], tmp1);
        c1 = bb_dot3(tmp2, tmp2);
        if (tmp1[2] > 0.0f)
          if (c1 > margin2) continue;

        bb_addTo3(points[n], tmp1);
        bb_scl3(points[n], points[n], 0.5f);
        depth[n] = sqrt(c1) * (tmp1[2] < 0.0f ? -1.0f : 1.0f);
        n++;
      }
    }

    nf = n;
    for (i = 0; i < 4; i++) {
      if (n < 8) {
        x = ppts2[i][0];
        y = ppts2[i][1];
        if (nl == 0) {
          if (nf != 0) {
            if (x < -lx || x > lx)
              if (y < -ly || y > ly) continue;
          }
        } else {
          if (x < -lx || x > lx || y < -ly || y > ly) continue;
        }

        c1 = 0.0f;
        for (j = 0; j < 2; j++) {
          if (ppts2[i][j] < -s[j])
            c1 += (ppts2[i][j] + s[j]) * (ppts2[i][j] + s[j]);
          else if (ppts2[i][j] > s[j])
            c1 += (ppts2[i][j] - s[j]) * (ppts2[i][j] - s[j]);
        }
        c1 += pu[i][2] * innorm * pu[i][2] * innorm;
        if (pu[i][2] > 0.0f)
          if (c1 > margin2) continue;

        tmp1[0] = ppts2[i][0] * 0.5f;
        tmp1[1] = ppts2[i][1] * 0.5f;
        tmp1[2] = 0.0f;
        for (j = 0; j < 2; j++) {
          if (ppts2[i][j] < -s[j]) tmp1[j] = -s[j] * 0.5f;
          else if (ppts2[i][j] > s[j]) tmp1[j] = +s[j] * 0.5f;
        }
        bb_addToScl3(tmp1, pu[i], 0.5f);
        bb_copy3(points[n], tmp1);
        depth[n] = sqrt(c1) * (pu[i][2] < 0.0f ? -1.0f : 1.0f);
        n++;
      }
    }

    bb_mulMatMatT3(r, mat1, rotmore);
    bb_mulMatVec3(tmp1, r, rnorm);
    bb_scl3(con[0].normal, tmp1, in ? -1.0f : 1.0f);
    bb_zero3(con[0].tangent);

    for (i = 0; i < n; i++) {
      con[i].dist = depth[i];
      points[i][2] += hz;
      bb_mulMatVec3(tmp2, r, points[i]);
      bb_add3(con[i].pos, tmp2, pos1);
      bb_copy3(con[i].normal, con[0].normal);
      bb_zero3(con[i].tangent);
    }
    return n;
  }
}

inline int collide_box_box(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float mat1[9], mat2[9];
  quat_to_mat_row_major(q1, mat1);
  quat_to_mat_row_major(q2, mat2);

  float pos1[3] = {p1.x, p1.y, p1.z};
  float pos2[3] = {p2.x, p2.y, p2.z};
  float size1[3] = {sz1.x, sz1.y, sz1.z};
  float size2[3] = {sz2.x, sz2.y, sz2.z};

  RawPreContact tmp_con[8];
  int num = _boxbox_metal(tmp_con, pos1, mat1, size1, pos2, mat2, size2, margin);
  if (num <= 0) return 0;

  int dupe[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float sz1_m[3] = {size1[0] + margin, size1[1] + margin, size1[2] + margin};
  float sz2_m[3] = {size2[0] + margin, size2[1] + margin, size2[2] + margin};
  float kRemoveRatio = 1.01f;

  for (int i = 0; i < num; ++i) {
    int out1 = bb_outsideBox(tmp_con[i].pos, pos1, mat1, sz1_m, kRemoveRatio);
    int out2 = bb_outsideBox(tmp_con[i].pos, pos2, mat2, sz2_m, kRemoveRatio);
    if ((out1 == 1 && out2 != -1) || (out2 == 1 && out1 != -1)) {
      dupe[i] = -1;
    }
  }

  for (int i = 0; i < num - 1; ++i) {
    if (dupe[i] == -1) continue;
    for (int j = i + 1; j < num; ++j) {
      if (dupe[j] == -1) continue;
      if (tmp_con[i].pos[0] == tmp_con[j].pos[0] &&
          tmp_con[i].pos[1] == tmp_con[j].pos[1] &&
          tmp_con[i].pos[2] == tmp_con[j].pos[2]) {
        dupe[i] = -1;
        break;
      }
    }
  }

  int ncon = 0;
  for (int j = 0; j < num; ++j) {
    if (dupe[j] == 0) {
      con[ncon].dist = tmp_con[j].dist;
      con[ncon].pos = float3(tmp_con[j].pos[0], tmp_con[j].pos[1], tmp_con[j].pos[2]);
      con[ncon].normal = float3(tmp_con[j].normal[0], tmp_con[j].normal[1], tmp_con[j].normal[2]);
      con[ncon].t1 = float3(0.0f);
      make_frame(con[ncon].normal, con[ncon].t1, con[ncon].t2);
      ncon++;
      if (ncon >= 8) break;
    }
  }
  return ncon;
}

// 10. Plane - Cylinder (pinned mjc_PlaneCylinder port; g1 = plane, g2 = cylinder)
inline int collide_plane_cylinder(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float3 normal = rotate_q(q1, float3(0.0f, 0.0f, 1.0f));
  float3 axis = rotate_q(q2, float3(0.0f, 0.0f, 1.0f));
  float radius = sz2.x;
  float half_len = sz2.y;

  // Project, make sure axis points towards plane.
  float prjaxis = dot(normal, axis);
  // Sub-float32-noise tilts are rounding, not physics: snap them so the two
  // rim checks stay symmetric (pinned float64 sees exactly 0.0 here and
  // reports both rims; an asymmetric 1e-9 would drop one rim).
  if (fabs(prjaxis) < 1e-7f) prjaxis = 0.0f;
  if (prjaxis > 0.0f) {
    axis = -axis;
    prjaxis = -prjaxis;
  }

  // Normal distance to cylinder center.
  float3 to_c = p2 - p1;
  float dist0 = dot(to_c, normal);

  // Remove component of -normal along axis, compute length.
  float3 vec = prjaxis * axis - normal;
  float len_sqr = dot(vec, vec);
  if (len_sqr >= MJ_MINVAL * MJ_MINVAL) {
    vec *= radius / sqrt(len_sqr);
  } else {
    // Disk parallel to plane: pick x-axis of cylinder, scale by radius.
    vec = rotate_q(q2, float3(1.0f, 0.0f, 0.0f)) * radius;
  }
  float prjvec = dot(vec, normal);

  // Scale axis by half-length.
  axis *= half_len;
  prjaxis *= half_len;

  int cnt = 0;
  // First rim point.
  if (dist0 + prjaxis + prjvec <= margin) {
    con[cnt].dist = dist0 + prjaxis + prjvec;
    con[cnt].pos = p2 + vec + axis - normal * (con[cnt].dist * 0.5f);
    con[cnt].normal = normal;
    con[cnt].t1 = float3(0.0f);
    make_frame(con[cnt].normal, con[cnt].t1, con[cnt].t2);
    cnt++;
  } else {
    return 0;  // nearest point above margin: no contacts
  }
  // Second rim point.
  if (dist0 - prjaxis + prjvec <= margin) {
    con[cnt].dist = dist0 - prjaxis + prjvec;
    con[cnt].pos = p2 + vec - axis - normal * (con[cnt].dist * 0.5f);
    con[cnt].normal = normal;
    con[cnt].t1 = float3(0.0f);
    make_frame(con[cnt].normal, con[cnt].t1, con[cnt].t2);
    cnt++;
  }
  // Triangle points on the side closer to plane.
  float prjvec1 = -prjvec * 0.5f;
  if (dist0 + prjaxis + prjvec1 <= margin) {
    float3 vec1 = cross(vec, axis);
    float vl = length(vec1);
    if (vl > 1e-12f) vec1 = vec1 / vl;
    vec1 *= radius * sqrt(3.0f) * 0.5f;
    con[cnt].dist = dist0 + prjaxis + prjvec1;
    con[cnt].pos = p2 + vec1 + axis - vec * 0.5f - normal * (con[cnt].dist * 0.5f);
    con[cnt].normal = normal;
    con[cnt].t1 = float3(0.0f);
    make_frame(con[cnt].normal, con[cnt].t1, con[cnt].t2);
    cnt++;
    con[cnt].dist = dist0 + prjaxis + prjvec1;
    con[cnt].pos = p2 - vec1 + axis - vec * 0.5f - normal * (con[cnt].dist * 0.5f);
    con[cnt].normal = normal;
    con[cnt].t1 = float3(0.0f);
    make_frame(con[cnt].normal, con[cnt].t1, con[cnt].t2);
    cnt++;
  }
  return cnt;
}

// Raw plane-point/sphere contact with an explicit plane normal (for caps).
inline int raw_plane_normal_sphere(
    float3 plane_pos, float3 n,
    float3 p1, float radius,
    float margin, thread ContactGeom* con) {
  float cdist = dot(p1 - plane_pos, n);
  if (cdist > margin + radius) return 0;
  con[0].dist = cdist - radius;
  con[0].normal = n;
  con[0].pos = p1 - n * (radius + con[0].dist * 0.5f);
  con[0].t1 = float3(0.0f);
  make_frame(con[0].normal, con[0].t1, con[0].t2);
  return 1;
}

// 11. Sphere - Cylinder (pinned mjc_SphereCylinder port; g1 = sphere, g2 = cylinder)
inline int collide_sphere_cylinder(
    float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  float radius = sz2.x;
  float height = sz2.y;  // half-height, matching pinned size convention
  float3 axis = rotate_q(q2, float3(0.0f, 0.0f, 1.0f));

  // Sphere center relative to cylinder center, split into axial/radial parts.
  float3 vec = p1 - p2;
  float x = dot(axis, vec);
  float3 a_proj = axis * x;
  float3 p_proj = vec - a_proj;
  float p_proj_sqr = dot(p_proj, p_proj);

  bool collide_side = fabs(x) < height;
  bool collide_cap = p_proj_sqr < radius * radius;
  if (collide_side && collide_cap) {  // deep penetration: keep nearer exit
    float dist_cap = height - fabs(x);
    float dist_radius = radius - sqrt(p_proj_sqr);
    if (dist_cap < dist_radius) {
      collide_side = false;
    } else {
      collide_cap = false;
    }
  }

  // Side collision: sphere vs closest axis point (radius = cylinder radius).
  if (collide_side) {
    float3 a_point = p2 + a_proj;
    return collide_sphere_sphere(p1, q1, sz1, a_point, q2,
                                 float3(radius, 0.0f, 0.0f), margin, con);
  }

  // Cap collision: plane-sphere against the nearer cap, normal flipped to
  // point sphere -> cylinder (pinned flips: PLANE < SPHERE < CYLINDER).
  if (collide_cap) {
    float3 n_cap;
    float3 pos_cap;
    if (x > 0.0f) {
      n_cap = axis;
      pos_cap = p2 + axis * height;
    } else {
      n_cap = -axis;
      pos_cap = p2 - axis * height;
    }
    int ncon = raw_plane_normal_sphere(pos_cap, n_cap, p1, sz1.x, margin, con);
    if (ncon) {
      con[0].normal = -con[0].normal;
      con[0].t1 = -con[0].t1;
      con[0].t2 = cross(con[0].normal, con[0].t1);
    }
    return ncon;
  }

  // Corner collision: sphere vs rim corner point (zero-radius point sphere).
  float3 corner = p_proj * (radius / sqrt(p_proj_sqr)) + axis * (x > 0.0f ? height : -height) + p2;
  return collide_sphere_sphere(p1, q1, sz1, corner, q2,
                               float3(0.0f, 0.0f, 0.0f), margin, con);
}

// Forward declarations for milestone-010 convex pairs (defined in
// convex_narrowphase.metal, compiled after this file).
inline int collide_plane_convex(float3 p1, float4 q1,
                                int t2, float3 p2, float4 q2, float3 sz2,
                                float margin, thread ContactGeom* con);
inline int collide_convex_single(int ta, float3 pa, float4 qa, float3 sza,
                                 int tb, float3 pb, float4 qb, float3 szb,
                                 float margin, thread ContactGeom* con);
inline int collide_convex_multi(int ta, float3 pa, float4 qa, float3 sza,
                                int tb, float3 pb, float4 qb, float3 szb,
                                float margin, int maxn, thread ContactGeom* con);

// Unified Pair Dispatcher
inline int collide_pair(
    int type1, float3 p1, float4 q1, float3 sz1,
    int type2, float3 p2, float4 q2, float3 sz2,
    float margin, thread ContactGeom* con) {
  // Types: 0 = plane, 2 = sphere, 3 = capsule, 4 = ellipsoid, 5 = cylinder, 6 = box
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
  } else if (t1 == 0 && t2 == 5) {
    n = collide_plane_cylinder(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 2 && t2 == 5) {
    n = collide_sphere_cylinder(pos1, quat1, size1, pos2, quat2, size2, margin, con);
  } else if (t1 == 0 && t2 == 4) {
    n = collide_plane_convex(pos1, quat1, t2, pos2, quat2, size2, margin, con);
  } else if ((t1 == 2 && t2 == 4) || (t1 == 3 && t2 == 4) || (t1 == 4 && t2 == 4)
             || (t1 == 4 && t2 == 5) || (t1 == 4 && t2 == 6)) {
    n = collide_convex_single(t1, pos1, quat1, size1, t2, pos2, quat2, size2, margin, con);
  } else if ((t1 == 3 && t2 == 5) || (t1 == 5 && t2 == 5) || (t1 == 5 && t2 == 6)) {
    n = collide_convex_multi(t1, pos1, quat1, size1, t2, pos2, quat2, size2,
                             margin, 5, con);
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
