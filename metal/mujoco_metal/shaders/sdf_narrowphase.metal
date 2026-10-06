// Copyright 2026 The MuJoCo Metal contributors
//
// Licensed under the Apache License, Version 2.0.
//
// Milestone 013 signed-distance collision: plugin-free mesh-octree SDFs
// against analytic geoms and SDF-vs-SDF via Halton-seeded gradient descent
// (pinned mjc_SDF, 3.10.0). Per-seed witnesses make native-vs-CPU
// comparison nearly 1:1 (same seeds, same traversal). Third-party plugin
// Mesh-SDF follows the pinned BVH face-processing and farthest-point
// subsampling path. Plugin-defined SDFs remain outside this native route.

#pragma once
#include <metal_stdlib>
using namespace metal;

// Geom type codes must match MuJoCo mjtGeom.
#define SDF_SDF 8
#define SDF_PLANE 1
#define SDF_SPHERE 2
#define SDF_CAPSULE 3
#define SDF_ELLIPSOID 4
#define SDF_CYLINDER 5
#define SDF_BOX 6
#define SDF_MESH 7

#define SDF_MAXVAL 3.4028235e+38f
#define SDF_FINDOCT_ITERS 100
#define SDF_MESH_MAX_CANDIDATES 50

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
  if (type == SDF_PLANE) {
    return x.z;
  } else if (type == SDF_SPHERE) {
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
    rr = rl >= 1e-15f ? rr / rl : float3(1.0f, 0.0f, 0.0f);
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
  if (type == SDF_PLANE) {
    return float3(0.0f, 0.0f, 1.0f);
  } else if (type == SDF_SPHERE) {
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
      rr = rl >= 1e-15f ? rr / rl : float3(1.0f, 0.0f, 0.0f);
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
    return l >= 1e-15f ? g / l : float3(1.0f, 0.0f, 0.0f);
  } else {  // SDF_CYLINDER
    float c = sqrt(x.x * x.x + x.y * x.y);
    float e = fabs(x.z);
    float a0 = c - sz.x;
    float a1 = e - sz.y;
    // Pinned source uses 1/mjMAXVAL with mjMAXVAL=1e10.
    float tiny = 1e-10f;
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

struct SdfPointPair { float3 hi; float3 lo; float3 tail; };
struct SdfVectorPair { float3 hi; float3 lo; float3 tail; };
struct SdfMatrixWide { SdfWide values[9]; };
struct SdfTransformWide { SdfMatrixWide rotation; SdfWide translation[3]; };
inline SdfWide sdf_point_component(SdfPointPair p, int i) {
  return sw_add(sw_add(sw(p.hi[i]), sw(p.lo[i])), sw(p.tail[i]));
}
inline SdfWide sdf_vector_component(SdfVectorPair v, int i) {
  return sw_add(sw_add(sw(v.hi[i]), sw(v.lo[i])), sw(v.tail[i]));
}
inline void sdf_point_store(thread SdfPointPair& p, int i, SdfWide v) {
  p.hi[i] = v.hi; p.lo[i] = v.mid; p.tail[i] = v.lo;
}
inline void sdf_vector_store(thread SdfVectorPair& v, int i, SdfWide x) {
  v.hi[i] = x.hi; v.lo[i] = x.mid; v.tail[i] = x.lo;
}
inline float3 sdf_pair_value(SdfPointPair p) {
  return float3(sw_float32(sdf_point_component(p,0)),
                sw_float32(sdf_point_component(p,1)),
                sw_float32(sdf_point_component(p,2)));
}
inline float3 sdf_vector_value(SdfVectorPair v) {
  return float3(sw_float32(sdf_vector_component(v,0)),
                sw_float32(sdf_vector_component(v,1)),
                sw_float32(sdf_vector_component(v,2)));
}
inline SdfVectorPair sdf_vector_add(SdfVectorPair a, SdfVectorPair b) {
  SdfVectorPair r;
  for (int i = 0; i < 3; ++i)
    sdf_vector_store(r, i, sw_add(sdf_vector_component(a, i),
                                  sdf_vector_component(b, i)));
  return r;
}
inline SdfVectorPair sdf_vector_scale(SdfVectorPair a, float scale) {
  SdfVectorPair r;
  for (int i = 0; i < 3; ++i)
    sdf_vector_store(r, i, sw_mul(sdf_vector_component(a, i), sw(scale)));
  return r;
}
inline SdfVectorPair sdf_vector_transform(float3x3 R, SdfVectorPair a) {
  SdfVectorPair r;
  for (int column = 0; column < 3; ++column) {
    SdfWide x = sw_mul(sw(R[0][column]), sdf_vector_component(a, 0));
    SdfWide y = sw_mul(sw(R[1][column]), sdf_vector_component(a, 1));
    SdfWide z = sw_mul(sw(R[2][column]), sdf_vector_component(a, 2));
    sdf_vector_store(r, column, sw_add(sw_add(x, y), z));
  }
  return r;
}
inline SdfWide sdf_vector_dot_wide(SdfVectorPair a, SdfVectorPair b) {
  return sw_source_dot3(sdf_vector_component(a,0),
                        sdf_vector_component(a,1),
                        sdf_vector_component(a,2),
                        sdf_vector_component(b,0),
                        sdf_vector_component(b,1),
                        sdf_vector_component(b,2));
}
inline float2 sdf_vector_dot(SdfVectorPair a, SdfVectorPair b) {
  SdfWide total = sdf_vector_dot_wide(a, b);
  return float2(total.hi, total.mid + total.lo);
}
inline SdfVectorPair sdf_vector_normalize(SdfVectorPair a) {
  SdfWide norm = sw_sqrt(sdf_vector_dot_wide(a, a));
  // Match mju_normalize3: below mjMINVAL the output is the x unit vector.
  if (sw_sign(sw_sub(norm, sw_source_minval())) < 0)
    return {float3(1.0f, 0.0f, 0.0f), float3(0.0f), float3(0.0f)};
  SdfVectorPair result;
  for (int i = 0; i < 3; ++i)
    sdf_vector_store(result, i,
                     sw_div(sdf_vector_component(a, i), norm));
  return result;
}
inline SdfPointPair sdf_pair_add(SdfPointPair p, float3 v) {
  SdfPointPair result;
  for (int i = 0; i < 3; ++i)
    sdf_point_store(result, i,
                    sw_add(sdf_point_component(p, i), sw(v[i])));
  return result;
}

inline int sdf_findoct_pair(thread SdfJet* weights,
                            thread SdfJet* coord_output,
                            thread SdfJet* span_output,
                            device const float* hull, int co, int nn,
                            SdfPointPair p) {
  int ab = co + 8 * nn, cb = co, stack = 0;
  SdfWide eps = sw_source_eps();
  for (int niter = 0; niter < SDF_FINDOCT_ITERS; ++niter) {
    int node = stack;
    if (node < 0 || node >= nn) return -1;
    SdfJet coord[3];
    bool outside = false, leaf = true;
    for (int axis = 0; axis < 3; ++axis) {
      int al = co + 22 * nn;
      SdfWide center=sw_add(sw(hull[ab+6*node+axis]),
                            sw(hull[al+6*node+axis]));
      SdfWide half_size=sw_add(sw(hull[ab+6*node+3+axis]),
                               sw(hull[al+6*node+3+axis]));
      SdfWide vmin=sw_sub(center,half_size);
      SdfWide vmax=sw_add(center,half_size);
      SdfWide value=sdf_point_component(p,axis);
      if (sw_sign(sw_sub(sw_add(value,eps),vmin))<0 ||
          sw_sign(sw_sub(sw_sub(value,eps),vmax))>0)
        outside = true;
      SdfWide span=sw_sub(vmax,vmin);
      coord[axis]=sjwide(sw_div(sw_sub(value,vmin),span));
      if (coord_output) coord_output[axis] = coord[axis];
      if (span_output) span_output[axis] = sjwide(span);
      for (int k = 0; k < 8; ++k)
        if (int(hull[cb + 8 * node + k]) != -1) leaf = false;
    }
    if (outside) continue;
    if (leaf) {
      for (int j = 0; j < 8; ++j) {
        SdfJet wx = (j & 1) ? coord[0] : sjsub(sj(1.0f), coord[0]);
        SdfJet wy = (j & 2) ? coord[1] : sjsub(sj(1.0f), coord[1]);
        SdfJet wz = (j & 4) ? coord[2] : sjsub(sj(1.0f), coord[2]);
        weights[j] = sjmul(sjmul(wx, wy), wz);
      }
      return node;
    }
    int xi = sjcompare(coord[0], sj(0.5f)) < 0 ? 0 : 1;
    int yi = sjcompare(coord[1], sj(0.5f)) < 0 ? 0 : 1;
    int zi = sjcompare(coord[2], sj(0.5f)) < 0 ? 0 : 1;
    stack = int(hull[cb + 8 * node + 4 * zi + 2 * yi + xi]);
  }
  return -1;
}

inline SdfJet sdf_oct_pair_box_project(thread SdfPointPair& point,
                                       SdfPointPair bc, SdfPointPair bh) {
  SdfJet q[3];
  for (int axis = 0; axis < 3; ++axis) {
    SdfJet p = sjwide(sdf_point_component(point,axis));
    SdfJet c = sjwide(sw_add(sw(bc.hi[axis]),sw(bc.lo[axis])));
    SdfJet h = sjwide(sw_add(sw(bh.hi[axis]),sw(bh.lo[axis])));
    q[axis] = sjsub(sjabs(sjsub(p,c)),h);
  }
  if (sjcompare(q[0], sj(0.0f)) <= 0 &&
      sjcompare(q[1], sj(0.0f)) <= 0 &&
      sjcompare(q[2], sj(0.0f)) <= 0)
    return sjmax(q[0], sjmax(q[1], q[2]));
  SdfJet distance2 = sj(0.0f);
  for (int axis = 0; axis < 3; ++axis) {
    if (sjcompare(q[axis], sj(0.0f)) >= 0) {
      distance2 = sjadd(distance2, sjmul(q[axis], q[axis]));
      SdfJet p = sjwide(sdf_point_component(point,axis));
      SdfJet c = sjwide(sw_add(sw(bc.hi[axis]),sw(bc.lo[axis])));
      float sign = sjcompare(sjsub(p,c),sj(0.0f)) > 0 ? 1.0f : -1.0f;
      SdfJet moved = sjsub(p,
          sjscale(sjadd(q[axis],sjwide(sw_source_box_eps())),sign));
      sdf_point_store(point,axis,sjwide_value(moved));
    }
  }
  return sjsqrt(distance2);
}

inline SdfJet sdf_oct_pair_dist(device const float* hull, int co, int nn,
                                SdfPointPair point) {
  int ab = co + 8 * nn;
  int al = co + 22 * nn;
  SdfPointPair bc = {float3(hull[ab], hull[ab + 1], hull[ab + 2]),
                     float3(hull[al], hull[al + 1], hull[al + 2])};
  SdfPointPair bh = {float3(hull[ab + 3], hull[ab + 4], hull[ab + 5]),
                     float3(hull[al + 3], hull[al + 4], hull[al + 5])};
  SdfJet box_distance = sdf_oct_pair_box_project(point, bc, bh);
  SdfJet weights[8];
  int node = sdf_findoct_pair(weights, nullptr, nullptr, hull, co, nn, point);
  if (node < 0) return sj(0.0f);
  int fb = co + 14 * nn;
  SdfJet value = sj(0.0f);
  int fl = co + 28 * nn;
  for (int i = 0; i < 8; ++i)
    value = sjadd(value, sjmul(weights[i],
        sjpair(hull[fb + 8 * node + i], hull[fl + 8 * node + i], float3(0.0f))));
  return sjcompare(box_distance, sj(0.0f)) > 0
      ? sjadd(value, box_distance) : value;
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

// An SDF operand is either a compiled octree, bundled analytic plugin, or
// ordinary analytic primitive. The plugin descriptor is stored in hull_info.
struct SdfSide {
  int kind;  // 0 analytic, 1 octree, 2 bundled plugin
  int type;
  int co;
  int nn;
  int plugin_kind;
  int attribute_base;
  float3 size;
  float3 size_low;
  float3 size_tail;
  int attribute_tail_base;
};

inline float sdf_side_dist(SdfSide side, device const float* hull, float3 x) {
  if (side.kind == 1) return sdf_oct_dist(hull, side.co, side.nn, x);
  if (side.kind == 2)
    return plugin_sdf_device_distance(side.plugin_kind, x, hull,
                                     side.attribute_base);
  return sdf_analytic_dist(side.type, side.size, x);
}

inline SdfJet sdf_side_jet(SdfSide side, device const float* hull, float3 x) {
  if (side.kind == 2)
    return plugin_sdf_device_jet(side.plugin_kind, x, hull,
                                 side.attribute_base);
  return sj(sdf_side_dist(side, hull, x));
}

inline float3 sdf_side_grad(SdfSide side, device const float* hull, float3 x) {
  if (side.kind == 1) return sdf_oct_grad(hull, side.co, side.nn, x);
  if (side.kind == 2)
    return plugin_sdf_device_gradient(side.plugin_kind, x, hull,
                                      side.attribute_base);
  return sdf_analytic_grad(side.type, side.size, x);
}

// Mesh-SDF helpers transcribed from pinned engine_collision_sdf.c. The
// compiled BVH node records store leaf face IDs, child indices, and local
// AABBs in that order; candidates retain upstream traversal order.
inline bool sdf_mesh_box_intersect(SdfSide sdf, device const float* hull,
                                   float3x3 R, float3 t, float3 bc, float3 bh) {
  return sdf_side_dist(sdf, hull, R * bc + t) < length(bh);
}

inline bool sdf_mesh_triangle_intersect(SdfSide sdf,
                                         device const float* hull,
                                         float3 a, float3 b, float3 c) {
  float3 e1 = b - a, e2 = c - a;
  float3 normal = cross(e1, e2);
  float nl = length(normal);
  if (nl > 1e-30f) normal /= nl;
  float3 p = (a + b + c) * (1.0f / 3.0f);
  float dcent = sdf_side_dist(sdf, hull, p);
  float h = -dcent * 0.1f;
  if (fabs(h) < 0.1f) {
    float la = length(b - a), lb = length(c - b), lc = length(a - c);
    float s = (la + lb + lc) * 0.5f;
    float area = sqrt(max(s * (s-la) * (s-lb) * (s-lc), 0.0f));
    float radius = (la * lb * lc) / (4.0f * max(area, 1e-30f));
    return dcent < radius;
  }
  p -= normal * h;
  float3 v0 = a - p, v1 = b - p, v2 = c - p;
  float3 c0 = cross(v1, v2), c1 = cross(v2, v0), c2 = cross(v0, v1);
  float denom = 2.0f * dot(v0, c0);
  if (fabs(denom) < 1e-30f) return dcent < max(length(e1), length(e2));
  float3 center = (c0 * dot(v0, v0) + c1 * dot(v1, v1) +
                   c2 * dot(v2, v2)) / denom;
  float radius = length(center);
  center += p;
  return sdf_side_dist(sdf, hull, center) < radius;
}

// Mesh candidates must retain point residuals through the actual source
// Frank-Wolfe updates and isknown() predicate. Publishing a float contact is
// separate from comparing two candidate points at mjMINVAL precision.
inline SdfPointPair sdf_pair_transform(float3x3 R, SdfPointPair p, float3 t);
inline SdfPointPair sdf_pair_transform(SdfTransformWide T, SdfPointPair p);
inline SdfTransformWide sdf_transform_transpose(SdfTransformWide T);
inline SdfVectorPair sdf_vector_transform(SdfMatrixWide R, SdfVectorPair a);
inline SdfJet sdf_side_pair_jet(SdfSide side, device const float* hull,
                                SdfPointPair x);
inline SdfVectorPair sdf_side_pair_grad(SdfSide side,
                                        device const float* hull,
                                        SdfPointPair x);
inline SdfPointPair sdf_mesh_pair_difference(SdfPointPair a, SdfPointPair b) {
  SdfPointPair result;
  for (int i = 0; i < 3; ++i)
    sdf_point_store(result, i, sw_sub(sdf_point_component(a, i),
                                     sdf_point_component(b, i)));
  return result;
}
inline SdfPointPair sdf_mesh_pair_add_scaled(SdfPointPair p, SdfPointPair v,
                                             float2 scale) {
  SdfWide factor = sw_add(sw(scale.x), sw(scale.y));
  SdfPointPair result;
  for (int i = 0; i < 3; ++i)
    sdf_point_store(result, i, sw_add(sdf_point_component(p, i),
        sw_mul(sdf_point_component(v, i), factor)));
  return result;
}
inline SdfPointPair sdf_pair_add_scaled(SdfPointPair p,SdfVectorPair v,
                                        SdfWide scale) {
  SdfPointPair result;
  for (int i=0;i<3;i++)
    sdf_point_store(result,i,sw_add(sdf_point_component(p,i),
        sw_mul(sdf_vector_component(v,i),scale)));
  return result;
}
inline SdfWide sdf_pair_distance(SdfPointPair a,SdfPointPair b) {
  SdfVectorPair delta;
  for (int i=0;i<3;i++)
    sdf_vector_store(delta,i,sw_sub(sdf_point_component(a,i),
                                    sdf_point_component(b,i)));
  return sw_sqrt(sdf_vector_dot_wide(delta,delta));
}
inline SdfWide sdf_mesh_pair_dot(SdfPointPair p, SdfVectorPair v) {
  return sdf_vector_dot_wide(SdfVectorPair{p.hi,p.lo,p.tail},v);
}
inline SdfWide sdf_halton_pair(int index,int base) {
  SdfWide b=sw(float(base));
  SdfWide factor=sw_div(sw(1.0f),b), result=sw(0.0f);
  while(index>0) {
    int next=index/base, digit=index-next*base;
    result=sw_add(result,sw_mul(factor,sw(float(digit))));
    factor=sw_div(factor,b);
    index=next;
  }
  return result;
}
inline SdfPointPair sdf_mesh_pair_add_scaled(SdfPointPair p,
                                             SdfPointPair v,
                                             SdfWide scale) {
  SdfPointPair result;
  for (int i=0;i<3;i++)
    sdf_point_store(result,i,sw_add(sdf_point_component(p,i),
        sw_mul(sdf_point_component(v,i),scale)));
  return result;
}
inline SdfJet sdf_mesh_frank_wolfe(thread SdfPointPair& x,
                                   SdfPointPair a,SdfPointPair b,SdfPointPair c,
                                   SdfSide sdf,device const float* hull,
                                   int iterations) {
  for(int step=0;step<iterations;++step) {
    SdfVectorPair grad=sdf_side_pair_grad(sdf,hull,x);
    SdfWide best=sdf_mesh_pair_dot(a,grad);
    SdfPointPair selected=a;
    SdfWide fb=sdf_mesh_pair_dot(b,grad);
    if(sw_sign(sw_sub(fb,best))<0){best=fb;selected=b;}
    if(sw_sign(sw_sub(sdf_mesh_pair_dot(c,grad),best))<0)selected=c;
    // Pinned mju_subFrom3(s,x), then mju_addToScl3(x,s,2/(step+2)).
    SdfPointPair delta=sdf_mesh_pair_difference(selected,x);
    SdfWide alpha=sw_div(sw(2.0f),sw(float(step+2)));
    x=sdf_mesh_pair_add_scaled(x,delta,alpha);
  }
  return sdf_side_pair_jet(sdf,hull,x);
}

inline void sdf_mesh_face_candidates(int face_id,float3x3 R,float3 t,
                                     int nstart,int iterations,
                                     SdfSide sdf,device const float* hull,
                                     int vertbase,int facebase,
                                     thread SdfPointPair* candidates,
                                     thread SdfJet* distances,
                                     thread int& ncandidate) {
  if(ncandidate>=SDF_MESH_MAX_CANDIDATES)return;
  int ib=facebase+3*face_id;
  int ia=int(hull[ib]),im=int(hull[ib+1]),ic=int(hull[ib+2]);
  SdfPointPair a=sdf_pair_transform(R,SdfPointPair{
      float3(hull[vertbase+3*ia],hull[vertbase+3*ia+1],hull[vertbase+3*ia+2]),
      float3(0)},t);
  SdfPointPair b=sdf_pair_transform(R,SdfPointPair{
      float3(hull[vertbase+3*im],hull[vertbase+3*im+1],hull[vertbase+3*im+2]),
      float3(0)},t);
  SdfPointPair c=sdf_pair_transform(R,SdfPointPair{
      float3(hull[vertbase+3*ic],hull[vertbase+3*ic+1],hull[vertbase+3*ic+2]),
      float3(0)},t);
  if(!sdf_mesh_triangle_intersect(sdf,hull,sdf_pair_value(a),
                                  sdf_pair_value(b),sdf_pair_value(c)))return;
  for(int sp=0;sp<nstart && ncandidate<SDF_MESH_MAX_CANDIDATES;++sp) {
    SdfWide u=sdf_halton_pair(sp+1,2),v=sdf_halton_pair(sp+1,3);
    if(sw_sign(sw_sub(sw_add(u,v),sw(1.0f)))>0){
      u=sw_sub(sw(1.0f),u);v=sw_sub(sw(1.0f),v);
    }
    SdfWide b0=sw_sub(sw_sub(sw(1.0f),u),v);
    SdfPointPair x=sdf_mesh_pair_add_scaled(
        SdfPointPair{float3(0),float3(0),float3(0)},a,b0);
    x=sdf_mesh_pair_add_scaled(x,b,u);
    x=sdf_mesh_pair_add_scaled(x,c,v);
    SdfJet depth=sdf_mesh_frank_wolfe(x,a,b,c,sdf,hull,iterations);
    if(sjcompare(depth,sj(0))<0){
      candidates[ncandidate]=x;distances[ncandidate]=depth;++ncandidate;
    }
  }
}


inline void sdf_mesh_face_candidates_cached(int face_id,float3x3 R,float3 t,
                                     int nstart,int iterations,
                                     SdfSide sdf,device const float* hull,
                                     int vertbase,int facebase,
                                     int scratch_base, device float* scratch,
                                     thread int& ncandidate) {
  if(ncandidate>=SDF_MESH_MAX_CANDIDATES)return;
  int ib=facebase+3*face_id;
  int ia=int(hull[ib]),im=int(hull[ib+1]),ic=int(hull[ib+2]);
  SdfPointPair a=sdf_pair_transform(R,SdfPointPair{
      float3(hull[vertbase+3*ia],hull[vertbase+3*ia+1],hull[vertbase+3*ia+2]),
      float3(0)},t);
  SdfPointPair b=sdf_pair_transform(R,SdfPointPair{
      float3(hull[vertbase+3*im],hull[vertbase+3*im+1],hull[vertbase+3*im+2]),
      float3(0)},t);
  SdfPointPair c=sdf_pair_transform(R,SdfPointPair{
      float3(hull[vertbase+3*ic],hull[vertbase+3*ic+1],hull[vertbase+3*ic+2]),
      float3(0)},t);
  if(!sdf_mesh_triangle_intersect(sdf,hull,sdf_pair_value(a),
                                  sdf_pair_value(b),sdf_pair_value(c)))return;
  for(int sp=0;sp<nstart && ncandidate<SDF_MESH_MAX_CANDIDATES;++sp) {
    SdfWide u=sdf_halton_pair(sp+1,2),v=sdf_halton_pair(sp+1,3);
    if(sw_sign(sw_sub(sw_add(u,v),sw(1.0f)))>0){
      u=sw_sub(sw(1.0f),u);v=sw_sub(sw(1.0f),v);
    }
    SdfWide b0=sw_sub(sw_sub(sw(1.0f),u),v);
    SdfPointPair x=sdf_mesh_pair_add_scaled(
        SdfPointPair{float3(0),float3(0),float3(0)},a,b0);
    x=sdf_mesh_pair_add_scaled(x,b,u);
    x=sdf_mesh_pair_add_scaled(x,c,v);
    SdfJet depth=sdf_mesh_frank_wolfe(x,a,b,c,sdf,hull,iterations);
    if(sjcompare(depth,sj(0))<0){
      int base=(scratch_base+ncandidate)*12;
      SdfWide depth_value=sjwide_value(depth);
      scratch[base+0]=x.hi.x; scratch[base+1]=x.hi.y; scratch[base+2]=x.hi.z;
      scratch[base+3]=x.lo.x; scratch[base+4]=x.lo.y; scratch[base+5]=x.lo.z;
      scratch[base+6]=x.tail.x; scratch[base+7]=x.tail.y; scratch[base+8]=x.tail.z;
      scratch[base+9]=depth_value.hi; scratch[base+10]=depth_value.mid;
      scratch[base+11]=depth_value.lo;
      ++ncandidate;
    }
  }
}
inline SdfSide sdf_geom_side(int geom, int type, float3 size,
                            device const int* hull_info) {
  int base = geom * 9;
  int descriptor = hull_info[base];
  if (descriptor == -2) {
    return {2, SDF_SDF, -1, 0, hull_info[base + 1],
            hull_info[base + 5], size, float3(0.0f), float3(0.0f),
            hull_info[base + 7]};
  }
  return {1, SDF_SDF, descriptor, hull_info[base + 1], 0, 0, size,
          float3(0.0f), float3(0.0f), -1};
}

inline SdfVectorPair sdf_pair_vector_from_jets(SdfJet x, SdfJet y,
                                                SdfJet z) {
  return {float3(x.v, y.v, z.v), float3(x.e, y.e, z.e),
          float3(x.tail, y.tail, z.tail)};
}

// Source-faithful paired version of pinned geomGradient() in
// engine_collision_sdf.c. Some shapes deliberately use a prescribed gradient
// that is not the derivative of geomDistance() (notably box interiors,
// ellipsoids and cylinders); do not replace these formulas with the distance
// jet's automatic derivative.
inline SdfVectorPair sdf_analytic_pair_source_grad(int type, float3 size,
                                                    SdfPointPair point,
                                                    float3 size_low = float3(0.0f),
                                                    float3 size_tail = float3(0.0f)) {
  SdfJet x = sjwide(sdf_point_component(point, 0));
  SdfJet y = sjwide(sdf_point_component(point, 1));
  SdfJet z = sjwide(sdf_point_component(point, 2));
  SdfJet sx = sjwide(sw_add(sw_add(sw(size.x),sw(size_low.x)),sw(size_tail.x)));
  SdfJet sy = sjwide(sw_add(sw_add(sw(size.y),sw(size_low.y)),sw(size_tail.y)));
  SdfJet sz = sjwide(sw_add(sw_add(sw(size.z),sw(size_low.z)),sw(size_tail.z)));
  SdfJet zero = sj(0.0f);
  SdfJet gx, gy, gz;

  if (type == SDF_PLANE) {
    return {float3(0.0f, 0.0f, 1.0f), float3(0.0f)};
  } else if (type == SDF_SPHERE) {
    SdfJet norm = sjlength3(x, y, z);
    gx = sjdiv(x, norm); gy = sjdiv(y, norm); gz = sjdiv(z, norm);
  } else if (type == SDF_BOX) {
    SdfJet a[3] = {sjsub(sjabs(x), sx),
                   sjsub(sjabs(y), sy),
                   sjsub(sjabs(z), sz)};
    // The strict comparisons and tie order mirror geomGradient exactly.
    int k = sjcompare(a[0], a[1]) > 0 ? 0 : 1;
    int l = sjcompare(a[2], a[k]) > 0 ? 2 : k;
    if (sjcompare(a[l], zero) < 0) {
      SdfJet radial_x = sjdiv(sjneg(sx), a[0]);
      SdfJet radial_y = sjdiv(sjneg(sy), a[1]);
      SdfJet radial_z = sjdiv(sjneg(sz), a[2]);
      SdfVectorPair radial = sdf_vector_normalize(
          sdf_pair_vector_from_jets(radial_x, radial_y, radial_z));
      if (sjcompare(x, zero) < 0) {
        radial.hi.x = -radial.hi.x; radial.lo.x = -radial.lo.x;
        radial.tail.x = -radial.tail.x;
      }
      if (sjcompare(y, zero) < 0) {
        radial.hi.y = -radial.hi.y; radial.lo.y = -radial.lo.y;
        radial.tail.y = -radial.tail.y;
      }
      if (sjcompare(z, zero) < 0) {
        radial.hi.z = -radial.hi.z; radial.lo.z = -radial.lo.z;
        radial.tail.z = -radial.tail.z;
      }
      return radial;
    }
    SdfJet b[3] = {sjmax(a[0], zero), sjmax(a[1], zero),
                   sjmax(a[2], zero)};
    SdfJet norm = sjlength3(b[0], b[1], b[2]);
    gx = zero; gy = zero; gz = zero;
    if (sjcompare(a[0], zero) > 0)
      gx = sjmul(sjdiv(b[0], norm), sjdiv(x, sjabs(x)));
    if (sjcompare(a[1], zero) > 0)
      gy = sjmul(sjdiv(b[1], norm), sjdiv(y, sjabs(y)));
    if (sjcompare(a[2], zero) > 0)
      gz = sjmul(sjdiv(b[2], norm), sjdiv(z, sjabs(z)));
  } else if (type == SDF_CAPSULE) {
    SdfJet clamped = sjmax(sjneg(sy), sjmin(z, sy));
    SdfJet az = sjsub(z, clamped);
    SdfJet norm = sjlength3(x, y, az);
    gx = sjdiv(x, norm); gy = sjdiv(y, norm); gz = sjdiv(az, norm);
  } else if (type == SDF_ELLIPSOID) {
    SdfJet a[3] = {sjdiv(x, sx), sjdiv(y, sy),
                   sjdiv(z, sz)};
    SdfJet b[3] = {sjdiv(a[0], sx), sjdiv(a[1], sy),
                   sjdiv(a[2], sz)};
    SdfJet k0 = sjlength3(a[0], a[1], a[2]);
    SdfJet k1 = sjlength3(b[0], b[1], b[2]);
    SdfJet inv_k0 = sjdiv(sj(1.0f), k0);
    SdfJet inv_k1 = sjdiv(sj(1.0f), k1);
    SdfJet gk0[3], gk1[3], g[3];
    for (int i = 0; i < 3; ++i) {
      SdfJet size_i = (i == 0 ? sx : (i == 1 ? sy : sz));
      SdfJet size_sq = sjmul(size_i, size_i);
      gk0[i] = sjmul(b[i], inv_k0);
      gk1[i] = sjdiv(sjmul(b[i], inv_k1), size_sq);
    }
    SdfJet df0 = sjmul(sjsub(sjmul(sj(2.0f), k0), sj(1.0f)), inv_k1);
    SdfJet df1 = sjmul(
        sjmul(sjmul(k0, sjsub(k0, sj(1.0f))), inv_k1), inv_k1);
    for (int i = 0; i < 3; ++i)
      g[i] = sjsub(sjmul(gk0[i], df0), sjmul(gk1[i], df1));
    return sdf_vector_normalize(sdf_pair_vector_from_jets(g[0], g[1], g[2]));
  } else {  // SDF_CYLINDER
    SdfJet c = sjlength2(x, y);
    SdfJet e = sjabs(z);
    SdfJet a0 = sjsub(c, sx);
    SdfJet a1 = sjsub(e, sy);
    // Pinned source clamps by 1/mjMAXVAL (mjMAXVAL=1e10).
    SdfJet gr[3] = {sjdiv(x, sjmax(c, sj(1e-10f))),
                    sjdiv(y, sjmax(c, sj(1e-10f))),
                    sjdiv(z, sjmax(e, sj(1e-10f)))};
    int j = sjcompare(a0, a1) > 0 ? 0 : 1;
    if (sjcompare(j == 0 ? a0 : a1, zero) < 0) {
      gx = j == 0 ? gr[0] : zero;
      gy = j == 0 ? gr[1] : zero;
      gz = j == 1 ? gr[2] : zero;
    } else {
      SdfJet b0 = sjmax(a0, zero), b1 = sjmax(a1, zero);
      SdfJet norm = sjmax(sjlength2(b0, b1), sj(1e-10f));
      gx = sjmul(gr[0], sjdiv(b0, norm));
      gy = sjmul(gr[1], sjdiv(b0, norm));
      gz = sjmul(gr[2], sjdiv(b1, norm));
    }
  }
  return sdf_pair_vector_from_jets(gx, gy, gz);
}

inline int collide_mesh_sdf(int gi_mesh, int gi_sdf, float3 pmesh,
                            float4 qmesh, float3 psdf, float4 qsdf,
                            thread ContactGeom* con,
                            device const float* hull,
                            device const int* hull_info) {
  int mb = gi_mesh * 9, sb = gi_sdf * 9;
  SdfSide sdf = sdf_geom_side(gi_sdf, SDF_SDF, float3(0.0f), hull_info);
  int iterations = hull_info[sb + 2], nstart = max(1, hull_info[sb + 4]);
  int nfaces = hull_info[mb + 4], vertbase = hull_info[mb + 5];
  int facebase = hull_info[mb + 6], bvhbase = hull_info[mb + 7];
  int nbvh = hull_info[mb + 8];
  if ((sdf.kind == 1 && sdf.nn <= 0) || (sdf.kind == 2 &&
      (sdf.plugin_kind <= 0 || sdf.attribute_base < 0)) ||
      iterations <= 0 || nfaces <= 0 || vertbase < 0 || facebase < 0 ||
      bvhbase < 0 || nbvh <= 0) return 0;
  float3x3 Rm = float3x3(rotate_q(qmesh, float3(1,0,0)),
                         rotate_q(qmesh, float3(0,1,0)),
                         rotate_q(qmesh, float3(0,0,1)));
  float3x3 Rs = float3x3(rotate_q(qsdf, float3(1,0,0)),
                         rotate_q(qsdf, float3(0,1,0)),
                         rotate_q(qsdf, float3(0,0,1)));
  float3x3 R = transpose(Rs) * Rm;
  float3 t = transpose(Rs) * (pmesh - psdf);
  SdfPointPair candidates[SDF_MESH_MAX_CANDIDATES];
  SdfJet distances[SDF_MESH_MAX_CANDIDATES];
  int ncandidate = 0;
  int stack[64];
  int nstack = 1;
  stack[0] = 0;
  while (nstack > 0 && ncandidate < SDF_MESH_MAX_CANDIDATES) {
    int node = stack[--nstack];
    if (node < 0 || node >= nbvh) continue;
    int rb = bvhbase + 9 * node;
    int face_id = int(hull[rb]);
    float3 bc = float3(hull[rb + 3], hull[rb + 4], hull[rb + 5]);
    float3 bh = float3(hull[rb + 6], hull[rb + 7], hull[rb + 8]);
    if (!sdf_mesh_box_intersect(sdf, hull, R, t, bc, bh)) continue;
    if (face_id >= 0) {
      if (face_id < nfaces)
        sdf_mesh_face_candidates(face_id, R, t, nstart, iterations, sdf,
                                 hull, vertbase, facebase, candidates,
                                 distances, ncandidate);
    } else {
      int c0 = int(hull[rb + 1]), c1 = int(hull[rb + 2]);
      if (c0 >= 0 && nstack < 64) stack[nstack++] = c0;
      if (c1 >= 0 && nstack < 64) stack[nstack++] = c1;
    }
  }
  // processSdfCorners stops after mjMAXCONPAIR candidates, making its later
  // FPS reduction unreachable; preserve all source candidates in BVH order.
  int ncon = 0;
  SdfPointPair accepted_points[SDF_MESH_MAX_CANDIDATES];
  for(int i=0;i<ncandidate;++i) {
    SdfPointPair x=candidates[i];
    bool known=false;
    for(int j=0;j<ncon;++j) {
      SdfWide distance=sdf_pair_distance(x,accepted_points[j]);
      if(sw_sign(sw_sub(distance,sw_source_minval()))<0){known=true;break;}
    }
    if(known)continue;
    SdfVectorPair gradient=sdf_side_pair_grad(sdf,hull,x);
    SdfWide gradient_length=sw_sqrt(sdf_vector_dot_wide(gradient,gradient));
    if(sw_sign(sw_sub(gradient_length,sw_source_minval()))<0)continue;
    SdfVectorPair local_normal=sdf_vector_normalize(gradient);
    local_normal.hi=-local_normal.hi;local_normal.lo=-local_normal.lo;
    local_normal.tail=-local_normal.tail;
    SdfVectorPair world_normal=sdf_vector_transform(Rs,local_normal);
    SdfPointPair world_point=sdf_pair_transform(Rs,x,psdf);
    world_point=sdf_pair_add_scaled(world_point,world_normal,
        sw_neg(sw_mul(sw(0.5f),sjwide_value(distances[i]))));
    accepted_points[ncon]=x;
    con[ncon].dist=sw_float32(sjwide_value(distances[i]));
    con[ncon].normal=sdf_vector_value(world_normal);
    con[ncon].pos=sdf_pair_value(world_point);
    con[ncon].t1=float3(0);
    make_frame(con[ncon].normal,con[ncon].t1,con[ncon].t2);
    ++ncon;
  }
  return ncon;
}


inline int collide_mesh_sdf_cached(int gi_mesh, int gi_sdf, float3 pmesh,
                            float4 qmesh, float3 psdf, float4 qsdf,
                            int maxn, int record_base, device float* records,
                            int scratch_base, device float* scratch,
                            device const float* hull,
                            device const int* hull_info) {
  int mb = gi_mesh * 9, sb = gi_sdf * 9;
  SdfSide sdf = sdf_geom_side(gi_sdf, SDF_SDF, float3(0.0f), hull_info);
  int iterations = hull_info[sb + 2], nstart = max(1, hull_info[sb + 4]);
  int nfaces = hull_info[mb + 4], vertbase = hull_info[mb + 5];
  int facebase = hull_info[mb + 6], bvhbase = hull_info[mb + 7];
  int nbvh = hull_info[mb + 8];
  if ((sdf.kind == 1 && sdf.nn <= 0) || (sdf.kind == 2 &&
      (sdf.plugin_kind <= 0 || sdf.attribute_base < 0)) ||
      iterations <= 0 || nfaces <= 0 || vertbase < 0 || facebase < 0 ||
      bvhbase < 0 || nbvh <= 0) return 0;
  float3x3 Rm = float3x3(rotate_q(qmesh, float3(1,0,0)),
                         rotate_q(qmesh, float3(0,1,0)),
                         rotate_q(qmesh, float3(0,0,1)));
  float3x3 Rs = float3x3(rotate_q(qsdf, float3(1,0,0)),
                         rotate_q(qsdf, float3(0,1,0)),
                         rotate_q(qsdf, float3(0,0,1)));
  float3x3 R = transpose(Rs) * Rm;
  float3 t = transpose(Rs) * (pmesh - psdf);
  int ncandidate = 0;
  int stack[64];
  int nstack = 1;
  stack[0] = 0;
  while (nstack > 0 && ncandidate < SDF_MESH_MAX_CANDIDATES) {
    int node = stack[--nstack];
    if (node < 0 || node >= nbvh) continue;
    int rb = bvhbase + 9 * node;
    int face_id = int(hull[rb]);
    float3 bc = float3(hull[rb + 3], hull[rb + 4], hull[rb + 5]);
    float3 bh = float3(hull[rb + 6], hull[rb + 7], hull[rb + 8]);
    if (!sdf_mesh_box_intersect(sdf, hull, R, t, bc, bh)) continue;
    if (face_id >= 0) {
      if (face_id < nfaces)
        sdf_mesh_face_candidates_cached(face_id, R, t, nstart, iterations, sdf,
                                        hull, vertbase, facebase, scratch_base,
                                        scratch, ncandidate);
    } else {
      int c0 = int(hull[rb + 1]), c1 = int(hull[rb + 2]);
      if (c0 >= 0 && nstack < 64) stack[nstack++] = c0;
      if (c1 >= 0 && nstack < 64) stack[nstack++] = c1;
    }
  }
  // processSdfCorners stops after mjMAXCONPAIR candidates, making its later
  // FPS reduction unreachable; preserve all source candidates in BVH order.
  int ncon = 0;
  for(int i=0;i<ncandidate;++i) {
    if (ncon >= maxn) break;
    int candidate_base=(scratch_base+i)*12;
    SdfPointPair x={
        float3(scratch[candidate_base+0],scratch[candidate_base+1],scratch[candidate_base+2]),
        float3(scratch[candidate_base+3],scratch[candidate_base+4],scratch[candidate_base+5]),
        float3(scratch[candidate_base+6],scratch[candidate_base+7],scratch[candidate_base+8])};
    bool known=false;
    for(int j=0;j<ncon;++j) {
      int accepted_base=(record_base+j)*25;
      SdfPointPair accepted={
          float3(records[accepted_base+0],records[accepted_base+1],records[accepted_base+2]),
          float3(records[accepted_base+3],records[accepted_base+4],records[accepted_base+5]),
          float3(records[accepted_base+6],records[accepted_base+7],records[accepted_base+8])};
      SdfWide distance=sdf_pair_distance(x,accepted);
      if(sw_sign(sw_sub(distance,sw_source_minval()))<0){known=true;break;}
    }
    if(known)continue;
    SdfVectorPair gradient=sdf_side_pair_grad(sdf,hull,x);
    SdfWide gradient_length=sw_sqrt(sdf_vector_dot_wide(gradient,gradient));
    if(sw_sign(sw_sub(gradient_length,sw_source_minval()))<0)continue;
    SdfVectorPair local_normal=sdf_vector_normalize(gradient);
    local_normal.hi=-local_normal.hi;local_normal.lo=-local_normal.lo;
    local_normal.tail=-local_normal.tail;
    SdfVectorPair world_normal=sdf_vector_transform(Rs,local_normal);
    SdfPointPair world_point=sdf_pair_transform(Rs,x,psdf);
    world_point=sdf_pair_add_scaled(world_point,world_normal,
        sw_neg(sw_mul(sw(0.5f),SdfWide{scratch[candidate_base+9],
                                           scratch[candidate_base+10],
                                           scratch[candidate_base+11]})));
    int accepted_base=(record_base+ncon)*25;
    records[accepted_base+0]=x.hi.x; records[accepted_base+1]=x.hi.y;
    records[accepted_base+2]=x.hi.z; records[accepted_base+3]=x.lo.x;
    records[accepted_base+4]=x.lo.y; records[accepted_base+5]=x.lo.z;
    records[accepted_base+6]=x.tail.x; records[accepted_base+7]=x.tail.y;
    records[accepted_base+8]=x.tail.z;
    float3 normal=sdf_vector_value(world_normal);
    float3 point=sdf_pair_value(world_point);
    float3 t1=float3(0),t2=float3(0);
    make_frame(normal,t1,t2);
    int output_base=accepted_base+12;
    records[output_base+0]=sw_float32(SdfWide{scratch[candidate_base+9],
        scratch[candidate_base+10],scratch[candidate_base+11]});
    records[output_base+1]=point.x; records[output_base+2]=point.y;
    records[output_base+3]=point.z; records[output_base+4]=normal.x;
    records[output_base+5]=normal.y; records[output_base+6]=normal.z;
    records[output_base+7]=t1.x; records[output_base+8]=t1.y;
    records[output_base+9]=t1.z; records[output_base+10]=t2.x;
    records[output_base+11]=t2.y; records[output_base+12]=t2.z;
    ++ncon;
  }
  return ncon;
}
// Combined objectives (pinned mjc_distance/mjc_gradient for the
// INTERSECTION, COLLISION and MIDSURFACE types). Side A is the SDF geom
// (x in A frame); side B an analytic or second SDF (y = Rab*x + tab).
inline float sdf_obj_dist(int mode, SdfSide A, SdfSide B, float3x3 Rab,
                          float3 tab, device const float* hull, float3 x) {
  float3 y = Rab * x + tab;
  float a = sdf_side_dist(A, hull, x);
  float b = sdf_side_dist(B, hull, y);
  if (mode == 0) return max(a, b);  // INTERSECTION
  if (mode == 1) return a - b;      // MIDSURFACE
  return a + b + fabs(max(a, b));   // COLLISION
}

inline SdfJet sdf_obj_jet_dist(int mode, SdfSide A, SdfSide B, float3x3 Rab,
                              float3 tab, device const float* hull, float3 x) {
  float3 y = Rab * x + tab;
  SdfJet a = sdf_side_jet(A, hull, x);
  SdfJet b = sdf_side_jet(B, hull, y);
  if (mode == 0) return sjmax(a, b);
  if (mode == 1) return sjsub(a, b);
  return sjadd(sjadd(a, b), sjabs(sjmax(a, b)));
}

// Keep candidate points and rigid transforms in a three-word expansion
// through SDF descent. The high word remains the historical float path; low
// and tail carry residuals through source mjtNum objectives and callback
// stencils.
inline bool sdf_jet_gt(SdfJet a, SdfJet b);
inline SdfPointPair sdf_pair_from(float3 p) {
  return {p, float3(0.0f), float3(0.0f)};
}
inline SdfPointPair sdf_pair_transform(float3x3 R, SdfPointPair p, float3 t) {
  SdfPointPair result;
  for (int column = 0; column < 3; ++column) {
    SdfWide x = sw_mul(sw(R[0][column]), sdf_point_component(p, 0));
    SdfWide y = sw_mul(sw(R[1][column]), sdf_point_component(p, 1));
    SdfWide z = sw_mul(sw(R[2][column]), sdf_point_component(p, 2));
    SdfWide value = sw_add(sw_add(sw_add(x, y), z), sw(t[column]));
    sdf_point_store(result, column, value);
  }
  return result;
}
inline SdfWide sdf_matrix_get(SdfMatrixWide A, int column, int row) {
  return A.values[3 * column + row];
}
inline void sdf_matrix_set(thread SdfMatrixWide& A, int column, int row,
                           SdfWide value) {
  A.values[3 * column + row] = value;
}
inline SdfMatrixWide sdf_matrix_transpose(SdfMatrixWide A) {
  SdfMatrixWide result;
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r)
      sdf_matrix_set(result, c, r, sdf_matrix_get(A, r, c));
  return result;
}
inline SdfMatrixWide sdf_matrix_multiply(SdfMatrixWide A, SdfMatrixWide B) {
  SdfMatrixWide result;
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) {
      SdfWide value = sw_mul(sdf_matrix_get(A, 0, r),
                             sdf_matrix_get(B, c, 0));
      value = sw_add(value, sw_mul(sdf_matrix_get(A, 1, r),
                                   sdf_matrix_get(B, c, 1)));
      value = sw_add(value, sw_mul(sdf_matrix_get(A, 2, r),
                                   sdf_matrix_get(B, c, 2)));
      sdf_matrix_set(result, c, r, value);
    }
  }
  return result;
}
inline SdfMatrixWide sdf_quat_matrix_wide(float4 q) {
  SdfWide u[3] = {sw(q.y), sw(q.z), sw(q.w)};
  SdfWide w = sw(q.x);
  SdfMatrixWide result;
  for (int column = 0; column < 3; ++column) {
    SdfWide v[3] = {sw(column == 0 ? 1.0f : 0.0f),
                    sw(column == 1 ? 1.0f : 0.0f),
                    sw(column == 2 ? 1.0f : 0.0f)};
    SdfWide inner[3];
    inner[0] = sw_add(sw_sub(sw_mul(u[1], v[2]), sw_mul(u[2], v[1])),
                      sw_mul(w, v[0]));
    inner[1] = sw_add(sw_sub(sw_mul(u[2], v[0]), sw_mul(u[0], v[2])),
                      sw_mul(w, v[1]));
    inner[2] = sw_add(sw_sub(sw_mul(u[0], v[1]), sw_mul(u[1], v[0])),
                      sw_mul(w, v[2]));
    SdfWide rotated[3];
    rotated[0] = sw_add(v[0], sw_mul(sw(2.0f),
        sw_sub(sw_mul(u[1], inner[2]), sw_mul(u[2], inner[1]))));
    rotated[1] = sw_add(v[1], sw_mul(sw(2.0f),
        sw_sub(sw_mul(u[2], inner[0]), sw_mul(u[0], inner[2]))));
    rotated[2] = sw_add(v[2], sw_mul(sw(2.0f),
        sw_sub(sw_mul(u[0], inner[1]), sw_mul(u[1], inner[0]))));
    for (int row = 0; row < 3; ++row)
      sdf_matrix_set(result, column, row, rotated[row]);
  }
  return result;
}
inline SdfTransformWide sdf_relative_transform_wide(float4 q_from,
    float4 q_to, float3 p_from, float3 p_to) {
  SdfMatrixWide Rfrom = sdf_quat_matrix_wide(q_from);
  SdfMatrixWide Rto = sdf_quat_matrix_wide(q_to);
  SdfTransformWide result;
  result.rotation = sdf_matrix_multiply(sdf_matrix_transpose(Rto), Rfrom);
  SdfWide delta[3] = {sw_sub(sw(p_from.x), sw(p_to.x)),
                      sw_sub(sw(p_from.y), sw(p_to.y)),
                      sw_sub(sw(p_from.z), sw(p_to.z))};
  SdfMatrixWide inverse = sdf_matrix_transpose(Rto);
  for (int row = 0; row < 3; ++row) {
    SdfWide value = sw_mul(sdf_matrix_get(inverse, 0, row), delta[0]);
    value = sw_add(value, sw_mul(sdf_matrix_get(inverse, 1, row), delta[1]));
    value = sw_add(value, sw_mul(sdf_matrix_get(inverse, 2, row), delta[2]));
    result.translation[row] = value;
  }
  return result;
}
inline SdfTransformWide sdf_reverse_transform_wide(
    SdfTransformWide forward, float4 q_from, float3 p_from, float3 p_to) {
  SdfTransformWide result;
  result.rotation = sdf_matrix_transpose(forward.rotation);
  SdfMatrixWide inverse = sdf_matrix_transpose(sdf_quat_matrix_wide(q_from));
  SdfWide delta[3] = {sw_sub(sw(p_to.x), sw(p_from.x)),
                      sw_sub(sw(p_to.y), sw(p_from.y)),
                      sw_sub(sw(p_to.z), sw(p_from.z))};
  for (int row = 0; row < 3; ++row) {
    SdfWide value = sw_mul(sdf_matrix_get(inverse, 0, row), delta[0]);
    value = sw_add(value, sw_mul(sdf_matrix_get(inverse, 1, row), delta[1]));
    value = sw_add(value, sw_mul(sdf_matrix_get(inverse, 2, row), delta[2]));
    result.translation[row] = value;
  }
  return result;
}
inline SdfTransformWide sdf_transform_transpose(SdfTransformWide T) {
  SdfTransformWide result;
  result.rotation = sdf_matrix_transpose(T.rotation);
  for (int r = 0; r < 3; ++r) result.translation[r] = sw(0.0f);
  return result;
}
inline SdfVectorPair sdf_vector_transform(SdfMatrixWide R,
                                         SdfVectorPair a) {
  SdfVectorPair result;
  for (int row = 0; row < 3; ++row) {
    SdfWide value = sw_mul(sdf_matrix_get(R, 0, row),
                           sdf_vector_component(a, 0));
    value = sw_add(value, sw_mul(sdf_matrix_get(R, 1, row),
                                 sdf_vector_component(a, 1)));
    value = sw_add(value, sw_mul(sdf_matrix_get(R, 2, row),
                                 sdf_vector_component(a, 2)));
    sdf_vector_store(result, row, value);
  }
  return result;
}
inline SdfPointPair sdf_pair_transform(SdfTransformWide T, SdfPointPair p) {
  SdfPointPair result;
  for (int row = 0; row < 3; ++row) {
    SdfWide value = sw_mul(sdf_matrix_get(T.rotation, 0, row),
                           sdf_point_component(p, 0));
    value = sw_add(value, sw_mul(sdf_matrix_get(T.rotation, 1, row),
                                 sdf_point_component(p, 1)));
    value = sw_add(value, sw_mul(sdf_matrix_get(T.rotation, 2, row),
                                 sdf_point_component(p, 2)));
    value = sw_add(value, T.translation[row]);
    sdf_point_store(result, row, value);
  }
  return result;
}
inline SdfPointPair sdf_pair_step(SdfPointPair p, float3 grad, float alpha) {
  SdfPointPair result;
  for (int i = 0; i < 3; ++i)
    sdf_point_store(result, i, sw_sub(sdf_point_component(p, i),
        sw_mul(sw(grad[i]), sw(alpha))));
  return result;
}
inline SdfPointPair sdf_pair_step(SdfPointPair p, SdfVectorPair grad,
                                  SdfWide alpha) {
  SdfPointPair result;
  for (int i = 0; i < 3; ++i)
    sdf_point_store(result, i, sw_sub(sdf_point_component(p, i),
        sw_mul(sdf_vector_component(grad, i), alpha)));
  return result;
}
inline SdfJet sdf_analytic_pair_jet_values(int type, float3 size,
                                           SdfJet x, SdfJet y, SdfJet z,
                                           float3 size_low = float3(0.0f),
                                           float3 size_tail = float3(0.0f)) {
  SdfJet sx = sjwide(sw_add(sw_add(sw(size.x),sw(size_low.x)),sw(size_tail.x)));
  SdfJet sy = sjwide(sw_add(sw_add(sw(size.y),sw(size_low.y)),sw(size_tail.y)));
  SdfJet sz = sjwide(sw_add(sw_add(sw(size.z),sw(size_low.z)),sw(size_tail.z)));
  if (type == SDF_PLANE) {
    return z;
  } else if (type == SDF_SPHERE) {
    return sjsub(sjlength3(x, y, z), sx);
  } else if (type == SDF_CAPSULE) {
    SdfJet clamped = sjmax(sjneg(sy), sjmin(z, sy));
    return sjsub(sjlength3(x, y, sjsub(z, clamped)), sx);
  } else if (type == SDF_ELLIPSOID) {
    SdfJet ax = sjdiv(x, sx), ay = sjdiv(y, sy), az = sjdiv(z, sz);
    SdfJet bx = sjdiv(ax, sx), by = sjdiv(ay, sy), bz = sjdiv(az, sz);
    SdfJet k0 = sjlength3(ax, ay, az);
    SdfJet k1 = sjlength3(bx, by, bz);
    return sjdiv(sjmul(k0, sjsub(k0, sj(1.0f))), k1);
  } else if (type == SDF_BOX) {
    SdfJet ax = sjsub(sjabs(x), sx);
    SdfJet ay = sjsub(sjabs(y), sy);
    SdfJet az = sjsub(sjabs(z), sz);
    bool outside = sjcompare(ax, sj(0.0f)) >= 0 ||
                   sjcompare(ay, sj(0.0f)) >= 0 ||
                   sjcompare(az, sj(0.0f)) >= 0;
    if (outside) {
      SdfJet bx = sjmax(ax, sj(0.0f));
      SdfJet by = sjmax(ay, sj(0.0f));
      SdfJet bz = sjmax(az, sj(0.0f));
      SdfJet radial = sjlength3(bx, by, bz);
      SdfJet axial = sjmin(sjmax(ax, sjmax(ay, az)), sj(0.0f));
      return sjadd(radial, axial);
    }
    SdfJet rx = sjdiv(sjneg(sx), ax);
    SdfJet ry = sjdiv(sjneg(sy), ay);
    SdfJet rz = sjdiv(sjneg(sz), az);
    SdfJet rlen = sjlength3(rx, ry, rz);
    rx = sjdiv(rx, rlen); ry = sjdiv(ry, rlen); rz = sjdiv(rz, rlen);
    if (sjcompare(x, sj(0.0f)) < 0) rx = sjneg(rx);
    if (sjcompare(y, sj(0.0f)) < 0) ry = sjneg(ry);
    if (sjcompare(z, sj(0.0f)) < 0) rz = sjneg(rz);
    SdfJet tx = sjdiv(sjneg(ax), sjabs(rx));
    SdfJet ty = sjdiv(sjneg(ay), sjabs(ry));
    SdfJet tz = sjdiv(sjneg(az), sjabs(rz));
    // Pinned geomDistance measures the already normalized radial field.
    // Its pre-normalization length must not scale penetration a second time.
    return sjneg(sjmul(sjmin(tx, sjmin(ty, tz)), sjlength3(rx, ry, rz)));
  } else {
    SdfJet radial = sjsub(sjlength2(x, y), sx);
    SdfJet axial = sjsub(sjabs(z), sy);
    SdfJet bx = sjmax(radial, sj(0.0f));
    SdfJet by = sjmax(axial, sj(0.0f));
    return sjadd(sjmin(sjmax(radial, axial), sj(0.0f)),
                 sjlength2(bx, by));
  }
}
inline SdfJet sdf_analytic_pair_jet(int type, float3 size,
                                    SdfPointPair point,
                                    float3 size_low = float3(0.0f),
                                    float3 size_tail = float3(0.0f)) {
  return sdf_analytic_pair_jet_values(type, size,
      sjwide(sdf_point_component(point, 0)),
      sjwide(sdf_point_component(point, 1)),
      sjwide(sdf_point_component(point, 2)), size_low, size_tail);
}
inline SdfJet sdf_side_pair_jet(SdfSide side, device const float* hull,
                                SdfPointPair x) {
  if (side.kind == 1) return sdf_oct_pair_dist(hull, side.co, side.nn, x);
  if (side.kind == 2)
    return plugin_sdf_device_jet_pair(side.plugin_kind, x.hi, x.lo, x.tail,
        hull, side.attribute_base, side.attribute_tail_base);
  if (side.kind == 0) return sdf_analytic_pair_jet(side.type, side.size, x,
      side.size_low, side.size_tail);
  return sj(sdf_side_dist(side, hull, sdf_pair_value(x)));
}
inline SdfJet sdf_obj_pair_jet_dist(int mode, SdfSide A, SdfSide B,
                                    SdfTransformWide T,
                                    device const float* hull, SdfPointPair x) {
  SdfPointPair y = sdf_pair_transform(T, x);
  SdfJet a = sdf_side_pair_jet(A, hull, x);
  SdfJet b = sdf_side_pair_jet(B, hull, y);
  if (mode == 0) return sjmax(a, b);
  if (mode == 1) return sjsub(a, b);
  return sjadd(sjadd(a, b), sjabs(sjmax(a, b)));
}
inline SdfVectorPair sdf_side_pair_grad(SdfSide side,
                                        device const float* hull,
                                        SdfPointPair x) {
  if (side.kind == 1) {
    int ab = side.co + 8 * side.nn;
    int al = side.co + 22 * side.nn;
    SdfPointPair bc = {float3(hull[ab], hull[ab + 1], hull[ab + 2]),
                       float3(hull[al], hull[al + 1], hull[al + 2])};
    SdfPointPair bh = {float3(hull[ab + 3], hull[ab + 4], hull[ab + 5]),
                       float3(hull[al + 3], hull[al + 4], hull[al + 5])};
    SdfPointPair inside_point = x;
    SdfJet box_distance = sdf_oct_pair_box_project(inside_point, bc, bh);
    if (sjcompare(box_distance, sj(0.0f)) <= 0) {
      SdfJet weights[8], coord[3], span[3];
      int node = sdf_findoct_pair(weights, coord, span, hull,
                                  side.co, side.nn, inside_point);
      if (node < 0) return {float3(0.0f), float3(0.0f)};
      SdfJet gradient[3] = {sj(0.0f), sj(0.0f), sj(0.0f)};
      int fb = side.co + 14 * side.nn;
      for (int i = 0; i < 8; ++i) {
        SdfJet wx = (i & 1) ? coord[0] : sjsub(sj(1.0f), coord[0]);
        SdfJet wy = (i & 2) ? coord[1] : sjsub(sj(1.0f), coord[1]);
        SdfJet wz = (i & 4) ? coord[2] : sjsub(sj(1.0f), coord[2]);
        SdfJet sign_x = sj((i & 1) ? 1.0f : -1.0f);
        SdfJet sign_y = sj((i & 2) ? 1.0f : -1.0f);
        SdfJet sign_z = sj((i & 4) ? 1.0f : -1.0f);
        // Pinned findOct/oct_gradient uses derivatives with respect to
        // normalized leaf coordinates; it does not divide these weights by
        // the physical leaf span. Preserve that source convention here too.
        SdfJet dx = sjmul(sjmul(sign_x, wy), wz);
        SdfJet dy = sjmul(sjmul(wx, sign_y), wz);
        SdfJet dz = sjmul(sjmul(wx, wy), sign_z);
        SdfJet coeff = sjpair(hull[fb + 8 * node + i],
            hull[side.co + 28 * side.nn + 8 * node + i], float3(0.0f));
        gradient[0] = sjadd(gradient[0], sjmul(dx, coeff));
        gradient[1] = sjadd(gradient[1], sjmul(dy, coeff));
        gradient[2] = sjadd(gradient[2], sjmul(dz, coeff));
      }
      return {float3(gradient[0].v, gradient[1].v, gradient[2].v),
              float3(gradient[0].e, gradient[1].e, gradient[2].e),
              float3(gradient[0].tail,gradient[1].tail,gradient[2].tail)};
    }
    SdfWide eps = sw_source_eps();
    SdfJet base = sdf_oct_pair_dist(hull, side.co, side.nn, x);
    SdfVectorPair result;
    for (int axis = 0; axis < 3; ++axis) {
      SdfPointPair shifted=x;
      sdf_point_store(shifted,axis,
          sw_add(sdf_point_component(shifted,axis),eps));
      SdfJet value = sdf_oct_pair_dist(hull, side.co, side.nn,shifted);
      SdfWide difference = sw_sub(sjwide_value(value),sjwide_value(base));
      sdf_vector_store(result,axis,sw_div(difference,eps));
    }
    return result;
  }
  if (side.kind == 2) {
    float3 result[3];
    plugin_sdf_device_gradient_pair(side.plugin_kind, x.hi, x.lo, x.tail,
        hull, side.attribute_base, side.attribute_tail_base, result);
    return {float3(result[0].x, result[1].x, result[2].x),
            float3(result[0].y, result[1].y, result[2].y),
            float3(result[0].z, result[1].z, result[2].z)};
  }
  if (side.kind == 0) {
    return sdf_analytic_pair_source_grad(side.type, side.size, x,
                                          side.size_low, side.size_tail);
  }
  return {sdf_side_grad(side, hull, sdf_pair_value(x)), float3(0.0f)};
}
inline SdfVectorPair sdf_obj_pair_grad(int mode, SdfSide A, SdfSide B,
                                       SdfTransformWide T,
                                       device const float* hull, SdfPointPair x) {
  SdfPointPair y = sdf_pair_transform(T, x);
  SdfTransformWide Tt = sdf_transform_transpose(T);
  SdfJet a = sdf_side_pair_jet(A, hull, x);
  SdfJet b = sdf_side_pair_jet(B, hull, y);
  SdfVectorPair g1 = sdf_side_pair_grad(A, hull, x);
  SdfVectorPair g2 = sdf_vector_transform(Tt.rotation,
                                           sdf_side_pair_grad(B, hull, y));
  if (mode == 0) return sdf_jet_gt(a, b) ? g1 : g2;
  if (mode == 1) {
    SdfVectorPair n1 = sdf_vector_normalize(g1);
    SdfVectorPair n2 = sdf_vector_normalize(g2);
    SdfVectorPair g = sdf_vector_add(n1, sdf_vector_scale(n2, -1.0f));
    return sdf_vector_normalize(g);
  }
  SdfVectorPair g = sdf_vector_add(g1, g2);
  SdfVectorPair pick = sdf_jet_gt(a, b) ? g1 : g2;
  float sign = sdf_jet_gt(sjmax(a, b), sj(0.0f)) ? 1.0f : -1.0f;
  return sdf_vector_add(g, sdf_vector_scale(pick, sign));
}

// Compare compensated values without first rounding hi+lo back to float32.
inline bool sdf_jet_gt(SdfJet a, SdfJet b) {
  SdfJet delta = sjsub(a, b);
  return sw_sign(sw_add(sw_add(sw(delta.v), sw(delta.e)), sw(delta.tail))) > 0;
}

// engine_collision_sdf.c:stepGradient rejects NaN or values strictly outside
// +/- mjMAXVAL.  In pinned MuJoCo 3.10.0 mjMAXVAL is exactly 1e10.  Compare
// the full three-word source value and preserve equality at the bound.
inline bool sdf_source_gradient_invalid(SdfVectorPair grad) {
  SdfWide limit = sw(1.0e10f);
  SdfWide negative_limit = sw(-1.0e10f);
  for (int i = 0; i < 3; ++i) {
    float hi = grad.hi[i], mid = grad.lo[i], low = grad.tail[i];
    if (!isfinite(hi) || !isfinite(mid) || !isfinite(low)) return true;
    SdfWide component = {hi, mid, low};
    if (sw_sign(sw_sub(component, limit)) > 0 ||
        sw_sign(sw_sub(component, negative_limit)) < 0) return true;
  }
  return false;
}

inline float3 sdf_obj_grad(int mode, SdfSide A, SdfSide B, float3x3 Rab,
                           float3 tab, device const float* hull, float3 x) {
  float3 y = Rab * x + tab;
  float3x3 Rt = transpose(Rab);
  SdfJet a = sdf_side_jet(A, hull, x);
  SdfJet b = sdf_side_jet(B, hull, y);
  float3 g1 = sdf_side_grad(A, hull, x);
  float3 g2 = Rt * sdf_side_grad(B, hull, y);
  if (mode == 0) return sdf_jet_gt(a, b) ? g1 : g2;
  if (mode == 1) {
    float l1 = length(g1), l2 = length(g2);
    g1 = l1 > 1e-24f ? g1 / l1 : g1;
    g2 = l2 > 1e-24f ? g2 / l2 : g2;
    float3 g = g1 - g2;
    float l = length(g);
    return l > 1e-24f ? g / l : g;
  }
  float3 g = g1 + g2;
  float3 pick = sdf_jet_gt(a, b) ? g1 : g2;
  float sign = sdf_jet_gt(sjmax(a, b), sj(0.0f)) ? 1.0f : -1.0f;
  return g + pick * sign;
}

// Gradient descent with backtracking (pinned stepGradient).
inline SdfWide sdf_descend(thread SdfPointPair& x, int mode, SdfSide A, SdfSide B,
                         SdfTransformWide T,
                         device const float* hull, int niter) {
  const SdfWide c = sw_source_point_one();
  const SdfWide rho = sw(0.5f);
  const SdfWide amin = sw_source_amin();
  SdfWide dist = sw(SDF_MAXVAL);
  for (int step = 0; step < niter; ++step) {
    SdfVectorPair grad = sdf_obj_pair_grad(mode, A, B, T, hull, x);
    if (sdf_source_gradient_invalid(grad)) return sw(1.0e10f);
    SdfPointPair x0 = x;
    SdfWide alpha = sw(2.0f);
    SdfJet dist0 = sdf_obj_pair_jet_dist(mode, A, B, T, hull, x0);
    SdfWide grad_squared = sdf_vector_dot_wide(grad, grad);
    SdfWide c_alpha = sw_mul(sw_neg(c), alpha);
    SdfWide wolfe = sw_mul(grad_squared, c_alpha);
    SdfJet accepted_dist;
    do {
      alpha = sw_mul(alpha, rho);
      wolfe = sw_mul(wolfe, rho);
      x = sdf_pair_step(x0, grad, alpha);
      accepted_dist = sdf_obj_pair_jet_dist(mode, A, B, T, hull, x);
      dist = sjwide_value(accepted_dist);
      SdfWide decrease = sw_sub(sjwide_value(accepted_dist),
                                 sjwide_value(dist0));
      if (!(sw_sign(sw_sub(alpha, amin)) > 0 &&
            sw_sign(sw_sub(decrease, wolfe)) > 0)) break;
    } while (true);
    if (sdf_jet_gt(accepted_dist, dist0)) {
      // Pinned stepGradient returns the last line-search trial distance and
      // leaves x at that trial point, even when it is worse than dist0.
      return sjwide_value(accepted_dist);
    }
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
  int aabb_tail_base = hull_info[gi2 * 9 + 6];
  if (abase < 0 || maxn <= 0) return 0;
  int iters = hull_info[gi2 * 9 + 2];
  if (iters <= 0) return 0;
  SdfSide A = sdf_geom_side(gi2, SDF_SDF, float3(0.0f), hull_info);
  if ((A.kind == 1 && A.nn <= 0) || (A.kind == 2 &&
      (A.plugin_kind <= 0 || A.attribute_base < 0))) return 0;
  bool sdfB = (t1 == SDF_SDF);
  SdfSide B = sdfB
      ? sdf_geom_side(gi1, SDF_SDF, float3(0.0f), hull_info)
      : SdfSide{0, t1, 0, 0, 0, 0, sz1,
          float3(hull[abase+18*gi1+15],hull[abase+18*gi1+16],hull[abase+18*gi1+17]),
          aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi1+12],
              hull[aabb_tail_base+15*gi1+13],hull[aabb_tail_base+15*gi1+14])
              : float3(0.0f), -1};
  if (sdfB && ((B.kind == 1 && B.nn <= 0) || (B.kind == 2 &&
      (B.plugin_kind <= 0 || B.attribute_base < 0)))) return 0;
  // Retain the compiler mjtNum AABB residuals and source Halton terms
  // through the seed-box intersection. Public contacts still use float32.
  SdfPointPair c1 = {float3(hull[abase+18*gi1],hull[abase+18*gi1+1],hull[abase+18*gi1+2]),
                     float3(hull[abase+18*gi1+6],hull[abase+18*gi1+7],hull[abase+18*gi1+8]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi1],
                         hull[aabb_tail_base+15*gi1+1],hull[aabb_tail_base+15*gi1+2])
                         : float3(0.0f)};
  SdfPointPair h1 = {float3(hull[abase+18*gi1+3],hull[abase+18*gi1+4],hull[abase+18*gi1+5]),
                     float3(hull[abase+18*gi1+9],hull[abase+18*gi1+10],hull[abase+18*gi1+11]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi1+6],
                         hull[aabb_tail_base+15*gi1+7],hull[aabb_tail_base+15*gi1+8])
                         : float3(0.0f)};
  SdfPointPair c2 = {float3(hull[abase+18*gi2],hull[abase+18*gi2+1],hull[abase+18*gi2+2]),
                     float3(hull[abase+18*gi2+6],hull[abase+18*gi2+7],hull[abase+18*gi2+8]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi2],
                         hull[aabb_tail_base+15*gi2+1],hull[aabb_tail_base+15*gi2+2])
                         : float3(0.0f)};
  SdfPointPair h2 = {float3(hull[abase+18*gi2+3],hull[abase+18*gi2+4],hull[abase+18*gi2+5]),
                     float3(hull[abase+18*gi2+9],hull[abase+18*gi2+10],hull[abase+18*gi2+11]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi2+6],
                         hull[aabb_tail_base+15*gi2+7],hull[aabb_tail_base+15*gi2+8])
                         : float3(0.0f)};
  SdfTransformWide T12 = sdf_relative_transform_wide(q1, q2, p1, p2);
  SdfTransformWide T21 = sdf_reverse_transform_wide(T12, q1, p1, p2);
  SdfPointPair lo = {float3(SDF_MAXVAL),float3(0.0f),float3(0.0f)};
  SdfPointPair hi = {float3(-SDF_MAXVAL),float3(0.0f),float3(0.0f)};
  SdfWide lower1[3],upper1[3],lower2[3],upper2[3];
  for(int a=0;a<3;a++) {
    lower1[a]=lower2[a]=sw(SDF_MAXVAL);
    upper1[a]=upper2[a]=sw(-SDF_MAXVAL);
  }
  for(int k=0;k<8;k++) {
    SdfPointPair v1,v2;
    for(int a=0;a<3;a++) {
      float sign=(k&(1<<a)) ? 1.0f : -1.0f;
      SdfWide one=sw_add(sdf_point_component(c1,a),
                          sw_mul(sdf_point_component(h1,a),sw(sign)));
      SdfWide two=sw_add(sdf_point_component(c2,a),
                          sw_mul(sdf_point_component(h2,a),sw(sign)));
      sdf_point_store(v1,a,one);sdf_point_store(v2,a,two);
    }
    SdfPointPair w2=sdf_pair_transform(T21,v2);
    for(int a=0;a<3;a++) {
      SdfWide one=sdf_point_component(v1,a),two=sdf_point_component(w2,a);
      if(sw_sign(sw_sub(one,lower1[a]))<0) lower1[a]=one;
      if(sw_sign(sw_sub(one,upper1[a]))>0) upper1[a]=one;
      if(sw_sign(sw_sub(two,lower2[a]))<0) lower2[a]=two;
      if(sw_sign(sw_sub(two,upper2[a]))>0) upper2[a]=two;
    }
  }
  for(int a=0;a<3;a++) {
    SdfWide l=sw_sign(sw_sub(lower1[a],lower2[a]))>0 ? lower1[a] : lower2[a];
    SdfWide h=sw_sign(sw_sub(upper1[a],upper2[a]))<0 ? upper1[a] : upper2[a];
    if(sw_sign(sw_sub(h,l))<0) return 0;
    sdf_point_store(lo,a,l);sdf_point_store(hi,a,h);
  }
  int ncon = 0;
  SdfPointPair accepted_points[50];
  for (int j = 0; j < maxn; ++j) {
    SdfPointPair s1;
    for(int a=0;a<3;a++) {
      SdfWide l=sdf_point_component(lo,a),h=sdf_point_component(hi,a);
      SdfWide halton=sdf_halton_pair(j,a==0 ? 2 : a==1 ? 3 : 5);
      SdfWide v=sw_add(l,sw_mul(sw_sub(h,l),halton));
      sdf_point_store(s1,a,v);
    }
    SdfPointPair x = sdf_pair_transform(T12, s1);
    SdfWide dcol = sdf_descend(x, 2, A, B, T21, hull, iters);
    (void)dcol;
    SdfWide dint = sdf_descend(x, 0, A, B, T21, hull, 1);
    // Gate and witness depth use the INTERSECTION result (pinned: dist2
    // from the 1-iteration pass); the normal comes from the MIDSURFACE
    // gradient at the same point.
    if (sw_sign(dint) > 0) continue;
    SdfVectorPair g_pair = sdf_obj_pair_grad(1, A, B, T21, hull, x);
    SdfWide gl_pair = sw_sqrt(sdf_vector_dot_wide(g_pair,g_pair));
    if (sw_sign(sw_sub(gl_pair,sw_source_minval())) < 0) continue;
    // Keep the gradient's correction word through norm/division. Collapsing
    // g_pair before normalization can erase represented transform/SDF terms
    // precisely when the two normalized side gradients nearly cancel.
    SdfVectorPair n_pair = sdf_vector_normalize(g_pair);
    bool known = false;
    for (int k = 0; k < ncon; ++k) {
      SdfWide separation=sdf_pair_distance(x,accepted_points[k]);
      if (sw_sign(sw_sub(separation,sw_source_minval())) < 0) {
        known=true; break;
      }
    }
    if (known) continue;
    accepted_points[ncon]=x;
    con[ncon].dist = sw_float32(dint);
    SdfVectorPair local_normal = {-n_pair.hi, -n_pair.lo, -n_pair.tail};
    SdfMatrixWide world_rotation = sdf_quat_matrix_wide(q2);
    SdfVectorPair world_normal = sdf_vector_transform(world_rotation, local_normal);
    SdfTransformWide world_transform;
    world_transform.rotation = world_rotation;
    world_transform.translation[0] = sw(p2.x);
    world_transform.translation[1] = sw(p2.y);
    world_transform.translation[2] = sw(p2.z);
    SdfPointPair world_point = sdf_pair_transform(world_transform, x);
    world_point = sdf_pair_add_scaled(world_point,world_normal,
        sw_neg(sw_mul(sw(0.5f),dint)));
    con[ncon].normal = sdf_vector_value(world_normal);
    con[ncon].pos = sdf_pair_value(world_point);
    con[ncon].t1 = float3(0.0f);
    make_frame(con[ncon].normal, con[ncon].t1, con[ncon].t2);
    ++ncon;
  }
  return ncon;
}

// Persistent source-order staging state for one analytic SDF seed. Every
// numerical component is stored as the same three float words as SdfWide.
// This keeps source-rounded values intact across kernel boundaries without
// keeping descent, line search, intersection, and publication in one PSO.
#define SDF_STAGE_STATE_WORDS 159
#define SDF_STAGE_X 0
#define SDF_STAGE_X0 9
#define SDF_STAGE_GRAD 18
#define SDF_STAGE_DIST0 27
#define SDF_STAGE_ALPHA 33
#define SDF_STAGE_WOLFE 36
#define SDF_STAGE_DIST 39
#define SDF_STAGE_T21 42
#define SDF_STAGE_NORMAL 150

inline SdfWide sdf_stage_load_wide(device const float* state, int base) {
  return {state[base], state[base + 1], state[base + 2]};
}
inline void sdf_stage_store_wide(device float* state, int base, SdfWide value) {
  state[base] = value.hi;
  state[base + 1] = value.mid;
  state[base + 2] = value.lo;
}
inline SdfPointPair sdf_stage_load_point(device const float* state, int base) {
  return {float3(state[base], state[base + 1], state[base + 2]),
          float3(state[base + 3], state[base + 4], state[base + 5]),
          float3(state[base + 6], state[base + 7], state[base + 8])};
}
inline void sdf_stage_store_point(device float* state, int base,
                                  SdfPointPair point) {
  state[base] = point.hi.x; state[base + 1] = point.hi.y;
  state[base + 2] = point.hi.z; state[base + 3] = point.lo.x;
  state[base + 4] = point.lo.y; state[base + 5] = point.lo.z;
  state[base + 6] = point.tail.x; state[base + 7] = point.tail.y;
  state[base + 8] = point.tail.z;
}
inline SdfVectorPair sdf_stage_load_vector(device const float* state, int base) {
  return {float3(state[base], state[base + 1], state[base + 2]),
          float3(state[base + 3], state[base + 4], state[base + 5]),
          float3(state[base + 6], state[base + 7], state[base + 8])};
}
inline void sdf_stage_store_vector(device float* state, int base,
                                   SdfVectorPair value) {
  state[base] = value.hi.x; state[base + 1] = value.hi.y;
  state[base + 2] = value.hi.z; state[base + 3] = value.lo.x;
  state[base + 4] = value.lo.y; state[base + 5] = value.lo.z;
  state[base + 6] = value.tail.x; state[base + 7] = value.tail.y;
  state[base + 8] = value.tail.z;
}
inline SdfJet sdf_stage_load_jet(device const float* state, int base) {
  return {state[base], state[base + 1],
          float3(state[base + 2], state[base + 3], state[base + 4]),
          state[base + 5]};
}
inline void sdf_stage_store_jet(device float* state, int base, SdfJet value) {
  state[base] = value.v; state[base + 1] = value.e;
  state[base + 2] = value.d.x; state[base + 3] = value.d.y;
  state[base + 4] = value.d.z; state[base + 5] = value.tail;
}
inline void sdf_stage_store_transform(device float* state, int base,
                                      SdfTransformWide T) {
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) {
      SdfWide value = sdf_matrix_get(T.rotation, c, r);
      int at = base + 9 * c + 3 * r;
      sdf_stage_store_wide(state, at, value);
    }
  }
  for (int r = 0; r < 3; ++r)
    sdf_stage_store_wide(state, base + 27 + 3 * r, T.translation[r]);
}
inline SdfTransformWide sdf_stage_load_transform(device const float* state,
                                                  int base) {
  SdfTransformWide T;
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r)
      sdf_matrix_set(T.rotation, c, r,
          sdf_stage_load_wide(state, base + 9 * c + 3 * r));
  for (int r = 0; r < 3; ++r)
    T.translation[r] = sdf_stage_load_wide(state, base + 27 + 3 * r);
  return T;
}

inline bool sdf_stage_sides(int t1, float3 sz1, int gi1, int gi2,
                            device const float* hull,
                            device const int* hull_info,
                            thread SdfSide& A, thread SdfSide& B) {
  int abase = hull_info[gi2 * 9 + 3];
  if (abase < 0) return false;
  A = sdf_geom_side(gi2, SDF_SDF, float3(0.0f), hull_info);
  B = t1 == SDF_SDF
      ? sdf_geom_side(gi1, SDF_SDF, float3(0.0f), hull_info)
      : SdfSide{0, t1, 0, 0, 0, 0, sz1,
          float3(hull[abase + 18 * gi1 + 15],
                 hull[abase + 18 * gi1 + 16],
                 hull[abase + 18 * gi1 + 17]),
          hull_info[gi2 * 9 + 6] >= 0
              ? float3(hull[hull_info[gi2 * 9 + 6] + 15 * gi1 + 12],
                       hull[hull_info[gi2 * 9 + 6] + 15 * gi1 + 13],
                       hull[hull_info[gi2 * 9 + 6] + 15 * gi1 + 14])
              : float3(0.0f), -1};
  if ((A.kind == 1 && A.nn <= 0) ||
      (A.kind == 2 && (A.plugin_kind <= 0 || A.attribute_base < 0)))
    return false;
  if (t1 == SDF_SDF &&
      ((B.kind == 1 && B.nn <= 0) ||
       (B.kind == 2 && (B.plugin_kind <= 0 || B.attribute_base < 0))))
    return false;
  return true;
}

inline bool sdf_stage_seed_init(int t1, float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2, int gi1, int gi2, int seed_index,
    device const float* hull, device const int* hull_info,
    device float* state, int state_base) {
  int abase = hull_info[gi2 * 9 + 3];
  if (abase < 0 || seed_index < 0) return false;
  int iters = hull_info[gi2 * 9 + 2];
  if (iters <= 0) return false;
  SdfSide A, B;
  if (!sdf_stage_sides(t1, sz1, gi1, gi2, hull, hull_info, A, B))
    return false;
  int aabb_tail_base = hull_info[gi2 * 9 + 6];
  SdfPointPair c1 = {float3(hull[abase + 18 * gi1],
      hull[abase + 18 * gi1 + 1], hull[abase + 18 * gi1 + 2]),
      float3(hull[abase + 18 * gi1 + 6], hull[abase + 18 * gi1 + 7],
             hull[abase + 18 * gi1 + 8]),
      aabb_tail_base >= 0 ? float3(hull[aabb_tail_base + 15 * gi1],
          hull[aabb_tail_base + 15 * gi1 + 1],
          hull[aabb_tail_base + 15 * gi1 + 2]) : float3(0.0f)};
  SdfPointPair h1 = {float3(hull[abase + 18 * gi1 + 3],
      hull[abase + 18 * gi1 + 4], hull[abase + 18 * gi1 + 5]),
      float3(hull[abase + 18 * gi1 + 9], hull[abase + 18 * gi1 + 10],
             hull[abase + 18 * gi1 + 11]),
      aabb_tail_base >= 0 ? float3(hull[aabb_tail_base + 15 * gi1 + 6],
          hull[aabb_tail_base + 15 * gi1 + 7],
          hull[aabb_tail_base + 15 * gi1 + 8]) : float3(0.0f)};
  SdfPointPair c2 = {float3(hull[abase + 18 * gi2],
      hull[abase + 18 * gi2 + 1], hull[abase + 18 * gi2 + 2]),
      float3(hull[abase + 18 * gi2 + 6], hull[abase + 18 * gi2 + 7],
             hull[abase + 18 * gi2 + 8]),
      aabb_tail_base >= 0 ? float3(hull[aabb_tail_base + 15 * gi2],
          hull[aabb_tail_base + 15 * gi2 + 1],
          hull[aabb_tail_base + 15 * gi2 + 2]) : float3(0.0f)};
  SdfPointPair h2 = {float3(hull[abase + 18 * gi2 + 3],
      hull[abase + 18 * gi2 + 4], hull[abase + 18 * gi2 + 5]),
      float3(hull[abase + 18 * gi2 + 9], hull[abase + 18 * gi2 + 10],
             hull[abase + 18 * gi2 + 11]),
      aabb_tail_base >= 0 ? float3(hull[aabb_tail_base + 15 * gi2 + 6],
          hull[aabb_tail_base + 15 * gi2 + 7],
          hull[aabb_tail_base + 15 * gi2 + 8]) : float3(0.0f)};
  SdfTransformWide T12 = sdf_relative_transform_wide(q1, q2, p1, p2);
  SdfTransformWide T21 = sdf_reverse_transform_wide(T12, q1, p1, p2);
  SdfPointPair lo = {float3(SDF_MAXVAL), float3(0.0f), float3(0.0f)};
  SdfPointPair hi = {float3(-SDF_MAXVAL), float3(0.0f), float3(0.0f)};
  SdfWide lower1[3], upper1[3], lower2[3], upper2[3];
  for (int a = 0; a < 3; ++a) {
    lower1[a] = lower2[a] = sw(SDF_MAXVAL);
    upper1[a] = upper2[a] = sw(-SDF_MAXVAL);
  }
  for (int k = 0; k < 8; ++k) {
    SdfPointPair v1, v2;
    for (int a = 0; a < 3; ++a) {
      float sign = (k & (1 << a)) ? 1.0f : -1.0f;
      SdfWide one = sw_add(sdf_point_component(c1, a),
                           sw_mul(sdf_point_component(h1, a), sw(sign)));
      SdfWide two = sw_add(sdf_point_component(c2, a),
                           sw_mul(sdf_point_component(h2, a), sw(sign)));
      sdf_point_store(v1, a, one); sdf_point_store(v2, a, two);
    }
    SdfPointPair w2 = sdf_pair_transform(T21, v2);
    for (int a = 0; a < 3; ++a) {
      SdfWide one = sdf_point_component(v1, a), two = sdf_point_component(w2, a);
      if (sw_sign(sw_sub(one, lower1[a])) < 0) lower1[a] = one;
      if (sw_sign(sw_sub(one, upper1[a])) > 0) upper1[a] = one;
      if (sw_sign(sw_sub(two, lower2[a])) < 0) lower2[a] = two;
      if (sw_sign(sw_sub(two, upper2[a])) > 0) upper2[a] = two;
    }
  }
  for (int a = 0; a < 3; ++a) {
    SdfWide l = sw_sign(sw_sub(lower1[a], lower2[a])) > 0
        ? lower1[a] : lower2[a];
    SdfWide h = sw_sign(sw_sub(upper1[a], upper2[a])) < 0
        ? upper1[a] : upper2[a];
    if (sw_sign(sw_sub(h, l)) < 0) return false;
    sdf_point_store(lo, a, l); sdf_point_store(hi, a, h);
  }
  SdfPointPair s1;
  for (int a = 0; a < 3; ++a) {
    SdfWide l = sdf_point_component(lo, a), h = sdf_point_component(hi, a);
    SdfWide halton = sdf_halton_pair(seed_index,
                                      a == 0 ? 2 : a == 1 ? 3 : 5);
    SdfWide value = sw_add(l, sw_mul(sw_sub(h, l), halton));
    sdf_point_store(s1, a, value);
  }
  SdfPointPair x = sdf_pair_transform(T12, s1);
  sdf_stage_store_point(state, state_base + SDF_STAGE_X, x);
  sdf_stage_store_transform(state, state_base + SDF_STAGE_T21, T21);
  sdf_stage_store_wide(state, state_base + SDF_STAGE_DIST,
                       sw(SDF_MAXVAL));
  return true;
}

inline bool sdf_stage_load_sides_transform(int t1, float3 sz1,
    int gi1, int gi2, device const float* hull,
    device const int* hull_info, device const float* state, int state_base,
    thread SdfSide& A, thread SdfSide& B,
    thread SdfTransformWide& T21) {
  if (!sdf_stage_sides(t1, sz1, gi1, gi2, hull, hull_info, A, B)) return false;
  T21 = sdf_stage_load_transform(state, state_base + SDF_STAGE_T21);
  return true;
}

inline void sdf_stage_prepare_step(int mode, SdfSide A, SdfSide B,
    SdfTransformWide T21, device const float* hull,
    device float* state, int state_base, device int* ctrl, int ctrl_index) {
  if (ctrl[ctrl_index] == 0) return;
  SdfPointPair x = sdf_stage_load_point(state, state_base + SDF_STAGE_X);
  SdfVectorPair grad = sdf_obj_pair_grad(mode, A, B, T21, hull, x);
  if (sdf_source_gradient_invalid(grad)) {
    sdf_stage_store_wide(state, state_base + SDF_STAGE_DIST,
                         sw(1.0e10f));
    ctrl[ctrl_index] = 0;
    return;
  }
  SdfJet dist0 = sdf_obj_pair_jet_dist(mode, A, B, T21, hull, x);
  SdfWide grad_squared = sdf_vector_dot_wide(grad, grad);
  SdfWide alpha = sw(2.0f);
  SdfWide c_alpha = sw_mul(sw_neg(sw_source_point_one()), alpha);
  SdfWide wolfe = sw_mul(grad_squared, c_alpha);
  sdf_stage_store_point(state, state_base + SDF_STAGE_X0, x);
  sdf_stage_store_vector(state, state_base + SDF_STAGE_GRAD, grad);
  sdf_stage_store_jet(state, state_base + SDF_STAGE_DIST0, dist0);
  sdf_stage_store_wide(state, state_base + SDF_STAGE_ALPHA, alpha);
  sdf_stage_store_wide(state, state_base + SDF_STAGE_WOLFE, wolfe);
}

inline void sdf_stage_line_search(int mode, SdfSide A, SdfSide B,
    SdfTransformWide T21, device const float* hull,
    device float* state, int state_base, device int* ctrl, int ctrl_index) {
  if (ctrl[ctrl_index] == 0) return;
  const SdfWide rho = sw(0.5f), amin = sw_source_amin();
  SdfPointPair x0 = sdf_stage_load_point(state, state_base + SDF_STAGE_X0);
  SdfVectorPair grad = sdf_stage_load_vector(state, state_base + SDF_STAGE_GRAD);
  SdfJet dist0 = sdf_stage_load_jet(state, state_base + SDF_STAGE_DIST0);
  SdfWide alpha = sdf_stage_load_wide(state, state_base + SDF_STAGE_ALPHA);
  SdfWide wolfe = sdf_stage_load_wide(state, state_base + SDF_STAGE_WOLFE);
  SdfPointPair x;
  SdfJet accepted_dist;
  do {
    alpha = sw_mul(alpha, rho);
    wolfe = sw_mul(wolfe, rho);
    x = sdf_pair_step(x0, grad, alpha);
    accepted_dist = sdf_obj_pair_jet_dist(mode, A, B, T21, hull, x);
    SdfWide decrease = sw_sub(sjwide_value(accepted_dist),
                              sjwide_value(dist0));
    if (!(sw_sign(sw_sub(alpha, amin)) > 0 &&
          sw_sign(sw_sub(decrease, wolfe)) > 0)) break;
  } while (true);
  // The in-loop trial value is the source's `dist`; stepGradient does not call
  // mjc_distance again after backtracking terminates.  On strict increase,
  // source returns this distance and leaves x at the rejected trial point.
  bool increased = sdf_jet_gt(accepted_dist, dist0);
  SdfWide dist = sjwide_value(accepted_dist);
  sdf_stage_store_point(state, state_base + SDF_STAGE_X, x);
  sdf_stage_store_wide(state, state_base + SDF_STAGE_DIST, dist);
  sdf_stage_store_wide(state, state_base + SDF_STAGE_ALPHA, alpha);
  sdf_stage_store_wide(state, state_base + SDF_STAGE_WOLFE, wolfe);
  ctrl[ctrl_index] = increased ? 0 : 1;
}

inline void sdf_stage_publish(int swapped, float3 p2, float4 q2,
    SdfVectorPair normal, SdfWide dist, device float* state, int state_base,
    device float* records, int record_base) {
  SdfPointPair x = sdf_stage_load_point(state, state_base + SDF_STAGE_X);
  SdfVectorPair oriented = {-normal.hi, -normal.lo, -normal.tail};
  SdfMatrixWide world_rotation = sdf_quat_matrix_wide(q2);
  SdfVectorPair world_normal = sdf_vector_transform(world_rotation, oriented);
  SdfTransformWide world_transform;
  world_transform.rotation = world_rotation;
  world_transform.translation[0] = sw(p2.x);
  world_transform.translation[1] = sw(p2.y);
  world_transform.translation[2] = sw(p2.z);
  SdfPointPair world_point = sdf_pair_transform(world_transform, x);
  world_point = sdf_pair_add_scaled(world_point, world_normal,
      sw_neg(sw_mul(sw(0.5f), dist)));
  float3 contact_normal = sdf_vector_value(world_normal);
  float3 point = sdf_pair_value(world_point);
  float3 t1 = float3(0.0f), t2 = float3(0.0f);
  make_frame(contact_normal, t1, t2);
  if (swapped) {
    contact_normal = -contact_normal;
    t1 = -t1;
    t2 = cross(contact_normal, t1);
  }
  sdf_stage_store_point(state, state_base + SDF_STAGE_X, x);
  int out = record_base * 25;
  records[out + 0] = x.hi.x; records[out + 1] = x.hi.y;
  records[out + 2] = x.hi.z; records[out + 3] = x.lo.x;
  records[out + 4] = x.lo.y; records[out + 5] = x.lo.z;
  records[out + 6] = x.tail.x; records[out + 7] = x.tail.y;
  records[out + 8] = x.tail.z;
  records[out + 12] = sw_float32(dist);
  records[out + 13] = point.x; records[out + 14] = point.y;
  records[out + 15] = point.z; records[out + 16] = contact_normal.x;
  records[out + 17] = contact_normal.y; records[out + 18] = contact_normal.z;
  records[out + 19] = t1.x; records[out + 20] = t1.y; records[out + 21] = t1.z;
  records[out + 22] = t2.x; records[out + 23] = t2.y; records[out + 24] = t2.z;
}

inline int collide_sdf_cached(
    int t1, float3 p1, float4 q1, float3 sz1,
    float3 p2, float4 q2,
    int gi1, int gi2, int seed_index, int record_base,
    device float* contact_records, device const float* hull,
    device const int* hull_info) {
  int abase = hull_info[gi2 * 9 + 3];
  int aabb_tail_base = hull_info[gi2 * 9 + 6];
  if (abase < 0 || seed_index < 0) return 0;
  int iters = hull_info[gi2 * 9 + 2];
  if (iters <= 0) return 0;
  SdfSide A = sdf_geom_side(gi2, SDF_SDF, float3(0.0f), hull_info);
  if ((A.kind == 1 && A.nn <= 0) || (A.kind == 2 &&
      (A.plugin_kind <= 0 || A.attribute_base < 0))) return 0;
  bool sdfB = (t1 == SDF_SDF);
  SdfSide B = sdfB
      ? sdf_geom_side(gi1, SDF_SDF, float3(0.0f), hull_info)
      : SdfSide{0, t1, 0, 0, 0, 0, sz1,
          float3(hull[abase+18*gi1+15],hull[abase+18*gi1+16],hull[abase+18*gi1+17]),
          aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi1+12],
              hull[aabb_tail_base+15*gi1+13],hull[aabb_tail_base+15*gi1+14])
              : float3(0.0f), -1};
  if (sdfB && ((B.kind == 1 && B.nn <= 0) || (B.kind == 2 &&
      (B.plugin_kind <= 0 || B.attribute_base < 0)))) return 0;
  // Retain the compiler mjtNum AABB residuals and source Halton terms
  // through the seed-box intersection. Public contacts still use float32.
  SdfPointPair c1 = {float3(hull[abase+18*gi1],hull[abase+18*gi1+1],hull[abase+18*gi1+2]),
                     float3(hull[abase+18*gi1+6],hull[abase+18*gi1+7],hull[abase+18*gi1+8]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi1],
                         hull[aabb_tail_base+15*gi1+1],hull[aabb_tail_base+15*gi1+2])
                         : float3(0.0f)};
  SdfPointPair h1 = {float3(hull[abase+18*gi1+3],hull[abase+18*gi1+4],hull[abase+18*gi1+5]),
                     float3(hull[abase+18*gi1+9],hull[abase+18*gi1+10],hull[abase+18*gi1+11]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi1+6],
                         hull[aabb_tail_base+15*gi1+7],hull[aabb_tail_base+15*gi1+8])
                         : float3(0.0f)};
  SdfPointPair c2 = {float3(hull[abase+18*gi2],hull[abase+18*gi2+1],hull[abase+18*gi2+2]),
                     float3(hull[abase+18*gi2+6],hull[abase+18*gi2+7],hull[abase+18*gi2+8]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi2],
                         hull[aabb_tail_base+15*gi2+1],hull[aabb_tail_base+15*gi2+2])
                         : float3(0.0f)};
  SdfPointPair h2 = {float3(hull[abase+18*gi2+3],hull[abase+18*gi2+4],hull[abase+18*gi2+5]),
                     float3(hull[abase+18*gi2+9],hull[abase+18*gi2+10],hull[abase+18*gi2+11]),
                     aabb_tail_base >= 0 ? float3(hull[aabb_tail_base+15*gi2+6],
                         hull[aabb_tail_base+15*gi2+7],hull[aabb_tail_base+15*gi2+8])
                         : float3(0.0f)};
  SdfTransformWide T12 = sdf_relative_transform_wide(q1, q2, p1, p2);
  SdfTransformWide T21 = sdf_reverse_transform_wide(T12, q1, p1, p2);
  SdfPointPair lo = {float3(SDF_MAXVAL),float3(0.0f),float3(0.0f)};
  SdfPointPair hi = {float3(-SDF_MAXVAL),float3(0.0f),float3(0.0f)};
  SdfWide lower1[3],upper1[3],lower2[3],upper2[3];
  for(int a=0;a<3;a++) {
    lower1[a]=lower2[a]=sw(SDF_MAXVAL);
    upper1[a]=upper2[a]=sw(-SDF_MAXVAL);
  }
  for(int k=0;k<8;k++) {
    SdfPointPair v1,v2;
    for(int a=0;a<3;a++) {
      float sign=(k&(1<<a)) ? 1.0f : -1.0f;
      SdfWide one=sw_add(sdf_point_component(c1,a),
                          sw_mul(sdf_point_component(h1,a),sw(sign)));
      SdfWide two=sw_add(sdf_point_component(c2,a),
                          sw_mul(sdf_point_component(h2,a),sw(sign)));
      sdf_point_store(v1,a,one);sdf_point_store(v2,a,two);
    }
    SdfPointPair w2=sdf_pair_transform(T21,v2);
    for(int a=0;a<3;a++) {
      SdfWide one=sdf_point_component(v1,a),two=sdf_point_component(w2,a);
      if(sw_sign(sw_sub(one,lower1[a]))<0) lower1[a]=one;
      if(sw_sign(sw_sub(one,upper1[a]))>0) upper1[a]=one;
      if(sw_sign(sw_sub(two,lower2[a]))<0) lower2[a]=two;
      if(sw_sign(sw_sub(two,upper2[a]))>0) upper2[a]=two;
    }
  }
  for(int a=0;a<3;a++) {
    SdfWide l=sw_sign(sw_sub(lower1[a],lower2[a]))>0 ? lower1[a] : lower2[a];
    SdfWide h=sw_sign(sw_sub(upper1[a],upper2[a]))<0 ? upper1[a] : upper2[a];
    if(sw_sign(sw_sub(h,l))<0) return 0;
    sdf_point_store(lo,a,l);sdf_point_store(hi,a,h);
  }
  int ncon = 0;
  int j = seed_index;
  {
    SdfPointPair s1;
    for(int a=0;a<3;a++) {
      SdfWide l=sdf_point_component(lo,a),h=sdf_point_component(hi,a);
      SdfWide halton=sdf_halton_pair(j,a==0 ? 2 : a==1 ? 3 : 5);
      SdfWide v=sw_add(l,sw_mul(sw_sub(h,l),halton));
      sdf_point_store(s1,a,v);
    }
    SdfPointPair x = sdf_pair_transform(T12, s1);
    SdfWide dcol = sdf_descend(x, 2, A, B, T21, hull, iters);
    (void)dcol;
    SdfWide dint = sdf_descend(x, 0, A, B, T21, hull, 1);
    // Gate and witness depth use the INTERSECTION result (pinned: dist2
    // from the 1-iteration pass); the normal comes from the MIDSURFACE
    // gradient at the same point.
    if (sw_sign(dint) > 0) return 0;
    SdfVectorPair g_pair = sdf_obj_pair_grad(1, A, B, T21, hull, x);
    SdfWide gl_pair = sw_sqrt(sdf_vector_dot_wide(g_pair,g_pair));
    if (sw_sign(sw_sub(gl_pair,sw_source_minval())) < 0) return 0;
    // Keep the gradient's correction word through norm/division. Collapsing
    // g_pair before normalization can erase represented transform/SDF terms
    // precisely when the two normalized side gradients nearly cancel.
    SdfVectorPair n_pair = sdf_vector_normalize(g_pair);
    int candidate_base=record_base*25;
    contact_records[candidate_base+0]=x.hi.x;
    contact_records[candidate_base+1]=x.hi.y;
    contact_records[candidate_base+2]=x.hi.z;
    contact_records[candidate_base+3]=x.lo.x;
    contact_records[candidate_base+4]=x.lo.y;
    contact_records[candidate_base+5]=x.lo.z;
    contact_records[candidate_base+6]=x.tail.x;
    contact_records[candidate_base+7]=x.tail.y;
    contact_records[candidate_base+8]=x.tail.z;
    int output_base=candidate_base+12;
    float dist=sw_float32(dint);
    SdfVectorPair local_normal = {-n_pair.hi, -n_pair.lo, -n_pair.tail};
    SdfMatrixWide world_rotation = sdf_quat_matrix_wide(q2);
    SdfVectorPair world_normal = sdf_vector_transform(world_rotation, local_normal);
    SdfTransformWide world_transform;
    world_transform.rotation = world_rotation;
    world_transform.translation[0] = sw(p2.x);
    world_transform.translation[1] = sw(p2.y);
    world_transform.translation[2] = sw(p2.z);
    SdfPointPair world_point = sdf_pair_transform(world_transform, x);
    world_point = sdf_pair_add_scaled(world_point,world_normal,
        sw_neg(sw_mul(sw(0.5f),dint)));
    float3 normal=sdf_vector_value(world_normal);
    float3 point=sdf_pair_value(world_point);
    float3 t1=float3(0.0f),t2=float3(0.0f);
    make_frame(normal,t1,t2);
    contact_records[output_base+0]=dist;
    contact_records[output_base+1]=point.x;
    contact_records[output_base+2]=point.y;
    contact_records[output_base+3]=point.z;
    contact_records[output_base+4]=normal.x;
    contact_records[output_base+5]=normal.y;
    contact_records[output_base+6]=normal.z;
    contact_records[output_base+7]=t1.x;
    contact_records[output_base+8]=t1.y;
    contact_records[output_base+9]=t1.z;
    contact_records[output_base+10]=t2.x;
    contact_records[output_base+11]=t2.y;
    contact_records[output_base+12]=t2.z;
    ++ncon;
  }
  return ncon;
}
