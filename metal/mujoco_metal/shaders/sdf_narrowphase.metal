// Copyright 2026 The MuJoCo Metal contributors
//
// Licensed under the Apache License, Version 2.0.
//
// Milestone 013 signed-distance collision: plugin-free mesh-octree SDFs
// against analytic geoms and SDF-vs-SDF via Halton-seeded gradient descent
// (pinned mjc_SDF, 3.10.0). Per-seed witnesses make native-vs-CPU
// comparison nearly 1:1 (same seeds, same traversal). Third-party plugin
// SDFs have no native path (rejected at lowering; 019 extension contract).
// Mesh-SDF (BVH face processing + FPS) is a documented follow-up gap.

#pragma once
#include <metal_stdlib>
using namespace metal;

// Geom type codes must match MuJoCo mjtGeom.
#define SDF_SDF 8
#define SDF_SPHERE 2
#define SDF_CAPSULE 3
#define SDF_ELLIPSOID 4
#define SDF_CYLINDER 5
#define SDF_BOX 6

#define SDF_MAXVAL 3.4028235e+38f
#define SDF_FINDOCT_ITERS 100

// Radical-inverse Halton value (pinned mju_Halton, 0-based).
inline float sdf_halton(int index, int base) {
  int n0 = index;
  float b = (float)base;
  float f = 1.0f / b;
  float hn = 0.0f;
  while (n0 > 0) {
    int n1 = n0 / base;
    int r = n0 - n1 * base;
    hn += f * (float)r;
    f /= b;
    n0 = n1;
  }
  return hn;
}

// Analytic SDF distance in geom-local frame (pinned geomDistance cases).
inline float sdf_analytic_dist(int type, float3 sz, float3 x) {
  if (type == SDF_SPHERE) {
    return length(x) - sz.x;
  } else if (type == SDF_BOX) {
    float3 a = float3(fabs(x.x) - sz.x, fabs(x.y) - sz.y, fabs(x.z) - sz.z);
    if (a.x >= 0.0f || a.y >= 0.0f || a.z >= 0.0f) {
      float3 b = float3(max(a.x, 0.0f), max(a.y, 0.0f), max(a.z, 0.0f));
      return length(b) + min(max(a.x, max(a.y, a.z)), 0.0f);
    }
    // Interior: intersect with unit gradient rotating radial to faces.
    float3 rr = float3(-sz.x / a.x, -sz.y / a.y, -sz.z / a.z);
    float rl = length(rr);
    rr = rl > 1e-30f ? rr / rl : float3(1.0f, 0.0f, 0.0f);
    if (x.x < 0.0f) rr.x = -rr.x;
    if (x.y < 0.0f) rr.y = -rr.y;
    if (x.z < 0.0f) rr.z = -rr.z;
    float3 t = float3(-a.x / fabs(rr.x), -a.y / fabs(rr.y), -a.z / fabs(rr.z));
    return -min(t.x, min(t.y, t.z)) * length(rr);
  } else if (type == SDF_CAPSULE) {
    float3 a = float3(x.x, x.y, x.z - clamp(x.z, -sz.y, sz.y));
    return length(a) - sz.x;
  } else if (type == SDF_ELLIPSOID) {
    float3 a = float3(x.x / sz.x, x.y / sz.y, x.z / sz.z);
    float3 b = float3(a.x / sz.x, a.y / sz.y, a.z / sz.z);
    float k0 = length(a);
    float k1 = length(b);
    return k0 * (k0 - 1.0f) / k1;
  } else {  // SDF_CYLINDER
    float ar = sqrt(x.x * x.x + x.y * x.y) - sz.x;
    float az = fabs(x.z) - sz.y;
    float2 bb = float2(max(ar, 0.0f), max(az, 0.0f));
    return min(max(ar, az), 0.0f) + length(bb);
  }
}

// Analytic SDF gradient in geom-local frame (pinned geomGradient cases).
inline float3 sdf_analytic_grad(int type, float3 sz, float3 x) {
  if (type == SDF_SPHERE) {
    float c = length(x);
    return x / c;
  } else if (type == SDF_BOX) {
    float3 a = float3(fabs(x.x) - sz.x, fabs(x.y) - sz.y, fabs(x.z) - sz.z);
    int k = a.x > a.y ? 0 : 1;
    int l = a.z > (k == 0 ? a.x : a.y) ? 2 : k;
    float al = l == 0 ? a.x : (l == 1 ? a.y : a.z);
    if (al < 0.0f) {
      float3 rr = float3(-sz.x / a.x, -sz.y / a.y, -sz.z / a.z);
      float rl = length(rr);
      rr = rl > 1e-30f ? rr / rl : float3(1.0f, 0.0f, 0.0f);
      if (x.x < 0.0f) rr.x = -rr.x;
      if (x.y < 0.0f) rr.y = -rr.y;
      if (x.z < 0.0f) rr.z = -rr.z;
      return rr;
    }
    float3 b = float3(max(a.x, 0.0f), max(a.y, 0.0f), max(a.z, 0.0f));
    float c = length(b);
    float sx = x.x > 0.0f ? 1.0f : (x.x < 0.0f ? -1.0f : 0.0f);
    float sy = x.y > 0.0f ? 1.0f : (x.y < 0.0f ? -1.0f : 0.0f);
    float szz = x.z > 0.0f ? 1.0f : (x.z < 0.0f ? -1.0f : 0.0f);
    return float3(a.x > 0.0f ? b.x / c * sx : 0.0f,
                  a.y > 0.0f ? b.y / c * sy : 0.0f,
                  a.z > 0.0f ? b.z / c * szz : 0.0f);
  } else if (type == SDF_CAPSULE) {
    float3 a = float3(x.x, x.y, x.z - clamp(x.z, -sz.y, sz.y));
    float c = length(a);
    return a / c;
  } else if (type == SDF_ELLIPSOID) {
    float3 a = float3(x.x / sz.x, x.y / sz.y, x.z / sz.z);
    float3 b = float3(a.x / sz.x, a.y / sz.y, a.z / sz.z);
    float k0 = length(a);
    float k1 = length(b);
    float invK0 = 1.0f / k0;
    float invK1 = 1.0f / k1;
    float3 gk0 = float3(b.x * invK0, b.y * invK0, b.z * invK0);
    float3 gk1 = float3(b.x * invK1 / (sz.x * sz.x),
                        b.y * invK1 / (sz.y * sz.y),
                        b.z * invK1 / (sz.z * sz.z));
    float df0 = (2.0f * k0 - 1.0f) * invK1;
    float df1 = k0 * (k0 - 1.0f) * invK1 * invK1;
    float3 g = float3(gk0.x * df0 - gk1.x * df1,
                      gk0.y * df0 - gk1.y * df1,
                      gk0.z * df0 - gk1.z * df1);
    float l = length(g);
    return l > 1e-24f ? g / l : g;
  } else {  // SDF_CYLINDER
    float c = sqrt(x.x * x.x + x.y * x.y);
    float e = fabs(x.z);
    float a0 = c - sz.x;
    float a1 = e - sz.y;
    float tiny = 1e-30f;
    float3 gr = float3(x.x / max(c, tiny), x.y / max(c, tiny),
                       x.z / max(e, tiny));
    int j = a0 > a1 ? 0 : 1;
    if ((j == 0 ? a0 : a1) < 0.0f) {
      return float3(j == 0 ? gr.x : 0.0f,
                    j == 0 ? gr.y : 0.0f,
                    j == 1 ? gr.z : 0.0f);
    }
    float2 b = float2(max(a0, 0.0f), max(a1, 0.0f));
    float bn = max(length(b), tiny);
    return float3(gr.x * b.x / bn, gr.y * b.x / bn, gr.z * b.y / bn);
  }
}

// Project onto the octree root box (pinned boxProjection): mutates point,
// returns signed distance (negative inside).
inline float sdf_box_project(thread float3& point, float3 bc, float3 bh) {
  float3 r = point - bc;
  float3 q = float3(fabs(r.x) - bh.x, fabs(r.y) - bh.y, fabs(r.z) - bh.z);
  float eps = 1e-6f;
  if (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f) {
    return max(q.x, max(q.y, q.z));
  }
  float dsq = 0.0f;
  if (q.x >= 0.0f) {
    dsq += q.x * q.x;
    point.x -= r.x > 0.0f ? (q.x + eps) : -(q.x + eps);
  }
  if (q.y >= 0.0f) {
    dsq += q.y * q.y;
    point.y -= r.y > 0.0f ? (q.y + eps) : -(q.y + eps);
  }
  if (q.z >= 0.0f) {
    dsq += q.z * q.z;
    point.z -= r.z > 0.0f ? (q.z + eps) : -(q.z + eps);
  }
  return sqrt(dsq);
}

// Octree leaf search (pinned findOct): trilinear weights and/or gradient
// weights for the containing leaf; returns local node index or -1.
inline int sdf_findoct(thread float* w, bool wantW, thread float* dw, bool wantDW,
                       device const float* hull, int co, int nn, float3 p) {
  int ab = co + 8 * nn;
  int cb = co;
  int stack = 0;
  float eps = 1e-8f;
  for (int niter = 0; niter < SDF_FINDOCT_ITERS; ++niter) {
    int node = stack;
    if (node < 0 || node >= nn) return -1;
    float3 bc = float3(hull[ab + 6 * node], hull[ab + 6 * node + 1],
                       hull[ab + 6 * node + 2]);
    float3 bh = float3(hull[ab + 6 * node + 3], hull[ab + 6 * node + 4],
                       hull[ab + 6 * node + 5]);
    float3 vmin = bc - bh;
    float3 vmax = bc + bh;
    if (p.x + eps < vmin.x || p.x - eps > vmax.x ||
        p.y + eps < vmin.y || p.y - eps > vmax.y ||
        p.z + eps < vmin.z || p.z - eps > vmax.z) {
      continue;
    }
    float3 span = vmax - vmin;
    float3 coord = float3((p.x - vmin.x) / span.x,
                          (p.y - vmin.y) / span.y,
                          (p.z - vmin.z) / span.z);
    bool leaf = true;
    for (int k = 0; k < 8; ++k) {
      if (int(hull[cb + 8 * node + k]) != -1) { leaf = false; break; }
    }
    if (leaf) {
      for (int j = 0; j < 8; ++j) {
        float wx = (j & 1) ? coord.x : 1.0f - coord.x;
        float wy = (j & 2) ? coord.y : 1.0f - coord.y;
        float wz = (j & 4) ? coord.z : 1.0f - coord.z;
        if (wantW) w[j] = wx * wy * wz;
        if (wantDW) {
          dw[j * 3] = ((j & 1) ? 1.0f : -1.0f) * wy * wz;
          dw[j * 3 + 1] = wx * ((j & 2) ? 1.0f : -1.0f) * wz;
          dw[j * 3 + 2] = wx * wy * ((j & 4) ? 1.0f : -1.0f);
        }
      }
      return node;
    }
    int xi = coord.x < 0.5f ? 0 : 1;
    int yi = coord.y < 0.5f ? 0 : 1;
    int zi = coord.z < 0.5f ? 0 : 1;
    stack = int(hull[cb + 8 * node + 4 * zi + 2 * yi + xi]);
  }
  return -1;
}

// Interpolated octree distance (pinned oct_distance).
inline float sdf_oct_dist(device const float* hull, int co, int nn, float3 p) {
  int ab = co + 8 * nn;
  float3 bc = float3(hull[ab], hull[ab + 1], hull[ab + 2]);
  float3 bh = float3(hull[ab + 3], hull[ab + 4], hull[ab + 5]);
  float3 point = p;
  float boxDist = sdf_box_project(point, bc, bh);
  float w[8];
  int node = sdf_findoct(w, true, nullptr, false, hull, co, nn, point);
  if (node < 0) return 0.0f;
  int fb = co + 14 * nn;
  float sdf = 0.0f;
  for (int i = 0; i < 8; ++i) sdf += w[i] * hull[fb + 8 * node + i];
  return boxDist > 0.0f ? sdf + boxDist : sdf;
}

// Interpolated octree gradient (pinned oct_gradient).
inline float3 sdf_oct_grad(device const float* hull, int co, int nn, float3 p) {
  int ab = co + 8 * nn;
  float3 bc = float3(hull[ab], hull[ab + 1], hull[ab + 2]);
  float3 bh = float3(hull[ab + 3], hull[ab + 4], hull[ab + 5]);
  float3 q = p;
  if (sdf_box_project(q, bc, bh) <= 0.0f) {
    float dw[24];
    int node = sdf_findoct(nullptr, false, dw, true, hull, co, nn, q);
    if (node < 0) return float3(0.0f);
    int fb = co + 14 * nn;
    float3 g = float3(0.0f);
    for (int i = 0; i < 8; ++i) {
      float c = hull[fb + 8 * node + i];
      g.x += dw[i * 3] * c;
      g.y += dw[i * 3 + 1] * c;
      g.z += dw[i * 3 + 2] * c;
    }
    return g;
  }
  float eps = 1e-8f;
  float d0 = sdf_oct_dist(hull, co, nn, p);
  float dx = sdf_oct_dist(hull, co, nn, p + float3(eps, 0.0f, 0.0f));
  float dy = sdf_oct_dist(hull, co, nn, p + float3(0.0f, eps, 0.0f));
  float dz = sdf_oct_dist(hull, co, nn, p + float3(0.0f, 0.0f, eps));
  return float3((dx - d0) / eps, (dy - d0) / eps, (dz - d0) / eps);
}

// SDF side evaluation: oct (built-in) at x in SDF-geom frame.
inline float sdf_side_dist(bool isOct, int type, float3 sz, int co, int nn,
                           device const float* hull, float3 x) {
  if (isOct) return sdf_oct_dist(hull, co, nn, x);
  return sdf_analytic_dist(type, sz, x);
}

inline float3 sdf_side_grad(bool isOct, int type, float3 sz, int co, int nn,
                            device const float* hull, float3 x) {
  if (isOct) return sdf_oct_grad(hull, co, nn, x);
  return sdf_analytic_grad(type, sz, x);
}

// Combined objectives (pinned mjc_distance/mjc_gradient for the
// INTERSECTION, COLLISION and MIDSURFACE types). Side A is the SDF geom
// (x in A frame); side B an analytic or second SDF (y = Rab*x + tab).
inline float sdf_obj_dist(int mode, bool octA, int typeA, float3 szA,
                          int coA, int nnA, bool octB, int typeB, float3 szB,
                          int coB, int nnB, float3x3 Rab, float3 tab,
                          device const float* hull, float3 x) {
  float3 y = Rab * x + tab;
  if (mode == 0) {  // INTERSECTION: max
    float a = sdf_side_dist(octA, typeA, szA, coA, nnA, hull, x);
    float b = sdf_side_dist(octB, typeB, szB, coB, nnB, hull, y);
    return max(a, b);
  } else if (mode == 1) {  // MIDSURFACE: difference
    float a = sdf_side_dist(octA, typeA, szA, coA, nnA, hull, x);
    float b = sdf_side_dist(octB, typeB, szB, coB, nnB, hull, y);
    return a - b;
  }
  // COLLISION
  float a = sdf_side_dist(octA, typeA, szA, coA, nnA, hull, x);
  float b = sdf_side_dist(octB, typeB, szB, coB, nnB, hull, y);
  return a + b + fabs(max(a, b));
}

inline float3 sdf_obj_grad(int mode, bool octA, int typeA, float3 szA,
                           int coA, int nnA, bool octB, int typeB, float3 szB,
                           int coB, int nnB, float3x3 Rab, float3 tab,
                           device const float* hull, float3 x) {
  float3 y = Rab * x + tab;
  float3x3 Rt = transpose(Rab);
  if (mode == 0) {  // INTERSECTION: gradient of max side
    float a = sdf_side_dist(octA, typeA, szA, coA, nnA, hull, x);
    float b = sdf_side_dist(octB, typeB, szB, coB, nnB, hull, y);
    if (a > b) return sdf_side_grad(octA, typeA, szA, coA, nnA, hull, x);
    return Rt * sdf_side_grad(octB, typeB, szB, coB, nnB, hull, y);
  } else if (mode == 1) {  // MIDSURFACE: normalized difference
    float3 g1 = sdf_side_grad(octA, typeA, szA, coA, nnA, hull, x);
    float l1 = length(g1);
    g1 = l1 > 1e-24f ? g1 / l1 : g1;
    float3 g2 = sdf_side_grad(octB, typeB, szB, coB, nnB, hull, y);
    g2 = Rt * g2;
    float l2 = length(g2);
    g2 = l2 > 1e-24f ? g2 / l2 : g2;
    float3 g = g1 - g2;
    float l = length(g);
    return l > 1e-24f ? g / l : g;
  }
  // COLLISION
  float a = sdf_side_dist(octA, typeA, szA, coA, nnA, hull, x);
  float b = sdf_side_dist(octB, typeB, szB, coB, nnB, hull, y);
  float3 g1 = sdf_side_grad(octA, typeA, szA, coA, nnA, hull, x);
  float3 g2 = Rt * sdf_side_grad(octB, typeB, szB, coB, nnB, hull, y);
  float3 g = g1 + g2;
  float3 pick = a > b ? g1 : g2;
  float s = max(a, b) > 0.0f ? 1.0f : -1.0f;
  return g + pick * s;
}

// Gradient descent with backtracking (pinned stepGradient).
inline float sdf_descend(thread float3& x, int mode, bool octA, int typeA,
                         float3 szA, int coA, int nnA, bool octB, int typeB,
                         float3 szB, int coB, int nnB, float3x3 Rab, float3 tab,
                         device const float* hull, int niter) {
  const float c = 0.1f;
  const float rho = 0.5f;
  const float amin = 1e-4f;
  float dist = SDF_MAXVAL;
  for (int step = 0; step < niter; ++step) {
    float3 grad = sdf_obj_grad(mode, octA, typeA, szA, coA, nnA,
                               octB, typeB, szB, coB, nnB, Rab, tab, hull, x);
    if (!isfinite(grad.x) || !isfinite(grad.y) || !isfinite(grad.z) ||
        fabs(grad.x) >= SDF_MAXVAL || fabs(grad.y) >= SDF_MAXVAL ||
        fabs(grad.z) >= SDF_MAXVAL) {
      return SDF_MAXVAL;
    }
    float3 x0 = x;
    float alpha = 2.0f;
    float dist0 = sdf_obj_dist(mode, octA, typeA, szA, coA, nnA,
                               octB, typeB, szB, coB, nnB, Rab, tab, hull, x0);
    float wolfe = -c * alpha * dot(grad, grad);
    do {
      alpha *= rho;
      wolfe *= rho;
      x = x0 - grad * alpha;
      dist = sdf_obj_dist(mode, octA, typeA, szA, coA, nnA,
                          octB, typeB, szB, coB, nnB, Rab, tab, hull, x);
    } while (alpha > amin && dist - dist0 > wolfe);
    if (dist0 < dist) return dist;
  }
  return dist;
}

// SDF-vs-(analytic|SDF) narrow phase (pinned mjc_SDF): Halton seeds in the
// AABB intersection (side-1 frame), descent in SDF frame, midsurface
// contacts. Margin ignored (pinned ignores it too). Per-seed witnesses make
// native-vs-CPU comparison nearly 1:1.
inline int collide_sdf(
    int t1, float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2,
    int gi1, int gi2, int maxn, thread ContactGeom* con,
    device const float* hull, device const int* hull_info) {
  int abase = hull_info[gi2 * 9 + 3];
  if (abase < 0 || maxn <= 0) return 0;
  // Side descriptors: A is the SDF geom (oct), B analytic or second SDF.
  int coA = hull_info[gi2 * 9];
  int nnA = hull_info[gi2 * 9 + 1];
  int iters = hull_info[gi2 * 9 + 2];
  if (coA < 0 || nnA <= 0 || iters <= 0) return 0;
  bool octB = (t1 == SDF_SDF);
  int coB = octB ? hull_info[gi1 * 9] : 0;
  int nnB = octB ? hull_info[gi1 * 9 + 1] : 0;
  if (octB && (coB < 0 || nnB <= 0)) return 0;
  // Local boxes (compiled center+half).
  float3 c1 = float3(hull[abase + 6 * gi1], hull[abase + 6 * gi1 + 1],
                     hull[abase + 6 * gi1 + 2]);
  float3 h1 = float3(hull[abase + 6 * gi1 + 3], hull[abase + 6 * gi1 + 4],
                     hull[abase + 6 * gi1 + 5]);
  float3 c2 = float3(hull[abase + 6 * gi2], hull[abase + 6 * gi2 + 1],
                     hull[abase + 6 * gi2 + 2]);
  float3 h2 = float3(hull[abase + 6 * gi2 + 3], hull[abase + 6 * gi2 + 4],
                     hull[abase + 6 * gi2 + 5]);
  float3x3 R1 = float3x3(rotate_q(q1, float3(1,0,0)),
                         rotate_q(q1, float3(0,1,0)),
                         rotate_q(q1, float3(0,0,1)));
  float3x3 R2 = float3x3(rotate_q(q2, float3(1,0,0)),
                         rotate_q(q2, float3(0,1,0)),
                         rotate_q(q2, float3(0,0,1)));
  float3x3 R12 = transpose(R2) * R1;  // side-1 frame -> SDF frame
  float3 t12 = transpose(R2) * (p1 - p2);
  float3x3 R21 = transpose(R12);      // SDF frame -> side-1 frame
  float3 t21 = transpose(R1) * (p2 - p1);
  // AABB intersection in side-1 frame (pinned per-side boxes, then
  // per-axis max-of-mins / min-of-maxes).
  float3 lo1 = float3(SDF_MAXVAL), hi1 = float3(-SDF_MAXVAL);
  float3 lo2 = float3(SDF_MAXVAL), hi2 = float3(-SDF_MAXVAL);
  for (int k = 0; k < 8; ++k) {
    float3 v1 = c1 + float3((k & 1) ? h1.x : -h1.x,
                            (k & 2) ? h1.y : -h1.y,
                            (k & 4) ? h1.z : -h1.z);
    lo1 = min(lo1, v1);
    hi1 = max(hi1, v1);
    float3 v2 = c2 + float3((k & 1) ? h2.x : -h2.x,
                            (k & 2) ? h2.y : -h2.y,
                            (k & 4) ? h2.z : -h2.z);
    float3 w2 = R21 * v2 + t21;
    lo2 = min(lo2, w2);
    hi2 = max(hi2, w2);
  }
  float3 lo = max(lo1, lo2);
  float3 hi = min(hi1, hi2);
  if (hi.x < lo.x || hi.y < lo.y || hi.z < lo.z) return 0;
  float3 ext = hi - lo;
  int ncon = 0;
  float pts[24];
  for (int j = 0; j < maxn; ++j) {
    float3 s1 = lo + ext * float3(sdf_halton(j, 2), sdf_halton(j, 3),
                                 sdf_halton(j, 5));
    float3 x = R12 * s1 + t12;
    float dcol = sdf_descend(x, 2, true, SDF_SDF, float3(0.0f), coA, nnA,
                             octB, t1, sz1, coB, nnB, R21, t21, hull, iters);
    (void)dcol;
    float dint = sdf_descend(x, 0, true, SDF_SDF, float3(0.0f), coA, nnA,
                             octB, t1, sz1, coB, nnB, R21, t21, hull, 1);
    // Gate and witness depth use the INTERSECTION result (pinned: dist2
    // from the 1-iteration pass); the normal comes from the MIDSURFACE
    // gradient at the same point.
    if (dint > 0.0f) continue;
    float3 g = sdf_obj_grad(1, true, SDF_SDF, float3(0.0f), coA, nnA,
                            octB, t1, sz1, coB, nnB, R21, t21, hull, x);
    float gl = length(g);
    if (gl < 1e-15f) continue;
    float3 nrm = -g / gl;  // INTO the SDF (pinned flipNormal=0)
    bool known = false;
    for (int k = 0; k < ncon; ++k) {
      float3 dd = x - float3(pts[k * 3], pts[k * 3 + 1], pts[k * 3 + 2]);
      if (dot(dd, dd) < 1e-30f) { known = true; break; }
    }
    if (known) continue;
    pts[ncon * 3] = x.x; pts[ncon * 3 + 1] = x.y; pts[ncon * 3 + 2] = x.z;
    con[ncon].dist = dint;
    con[ncon].normal = R2 * nrm;
    con[ncon].pos = R2 * x + p2 - (R2 * nrm) * (0.5f * dint);
    con[ncon].t1 = float3(0.0f);
    make_frame(con[ncon].normal, con[ncon].t1, con[ncon].t2);
    ++ncon;
  }
  return ncon;
}
