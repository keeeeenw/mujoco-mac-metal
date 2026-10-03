// Copyright 2026 The MuJoCo Metal contributors
//
// Licensed under the Apache License, Version 2.0.
//
// Milestone 010 convex narrow-phase: GJK distance + EPA penetration with
// analytic support functions for sphere/capsule/ellipsoid/cylinder/box.
// Validated against the pinned CPU oracle per pair (manifold + trajectories),
// not a line port of engine_collision_gjk.c. Single-witness exactness is the
// goal; multi-witness uses perturbed restarts (see collide_convex_multi).

#pragma once
#include <metal_stdlib>
using namespace metal;

// Geom type codes must match MuJoCo mjtGeom.
#define CX_HFIELD 1
#define CX_SPHERE 2
#define CX_CAPSULE 3
#define CX_ELLIPSOID 4
#define CX_CYLINDER 5
#define CX_BOX 6
#define CX_MESH 7

// Milestone 011: convex-hull vertex support. Hull verts are geom-local
// (world = geom frame composed with mesh_vert, calibrated against CPU
// contacts); hull_info holds per-geom (hull offset, hull count, face
// normal offset, face index offset, face count, hfield data offset,
// hfield nrow, hfield ncol, hfield size offset) — 9 ints (hfield fields
// are -1 for non-heightfield geoms).
#define CX_MESH_MAXSCAN 64

#define CX_MINVAL 1e-15f
#define CX_EPA_MAXFACES 64
#define CX_EPA_MAXITER 50
#define CX_GJK_MAXITER 64

// Analytic support point of a convex geom in world direction d.
inline float3 cx_support(int type, float3 p, float3x3 R, float3 sz, float3 d,
                         int gi, device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  float dl = length(d);
  float3 dn = dl > 1e-12f ? d / dl : float3(1.0f, 0.0f, 0.0f);
  // Local direction.
  float3 l = float3(dot(R[0], dn), dot(R[1], dn), dot(R[2], dn));
  float3 s = float3(0.0f);
  if (type == CX_SPHERE) {
    s = l * sz.x;
  } else if (type == CX_CAPSULE) {
    s = l * sz.x;
    s.z += (l.z >= 0.0f ? sz.y : -sz.y);
  } else if (type == CX_ELLIPSOID) {
    float3 e = l * sz;
    float n2 = dot(e, e);
    if (n2 < CX_MINVAL * CX_MINVAL) {
      s = float3(sz.x, 0.0f, 0.0f);
    } else {
      s = e * (1.0f / sqrt(n2)) * sz;
    }
  } else if (type == CX_CYLINDER) {
    float n2 = l.x * l.x + l.y * l.y;
    float scl = n2 >= CX_MINVAL * CX_MINVAL ? sz.x / sqrt(n2) : 0.0f;
    s = float3(scl * l.x, scl * l.y, l.z >= 0.0f ? sz.y : -sz.y);
  } else if (type == CX_MESH) {
    // Vertex-support over hull verts. Ties (face/edge contact) average to
    // the tied feature's centroid so portal seeding stays symmetric: a
    // bottom-face tie seeds the face basin, not an arbitrary tied vertex's
    // edge basin. Tie eps 1e-6 m never merges distinct mm-scale features.
    int off = hull_info[gi * 9];
    int cnt = hull_info[gi * 9 + 1];
    if (cnt < 0) cnt = 0;
    if (cnt > CX_MESH_MAXSCAN) cnt = CX_MESH_MAXSCAN;
    float best = -3.4028235e+38f;
    for (int k = 0; k < cnt; ++k) {
      float3 v = float3(hull[(off + k) * 3], hull[(off + k) * 3 + 1],
                        hull[(off + k) * 3 + 2]);
      float dp = dot(v, l);
      if (dp > best) best = dp;
    }
    float3 bs = float3(0.0f);
    float nw = 0.0f;
    for (int k = 0; k < cnt; ++k) {
      float3 v = float3(hull[(off + k) * 3], hull[(off + k) * 3 + 1],
                        hull[(off + k) * 3 + 2]);
      if (dot(v, l) >= best - 1e-6f) { bs += v; nw += 1.0f; }
    }
    s = nw > 0.0f ? bs / nw : float3(0.0f);
  } else if (type == CX_HFIELD) {
    // Milestone 012 terrain prism: 6 verts in the working frame
    // (collide_hfield runs GJK in the hfield frame, so these are local).
    // Pinned halfspace rule (mjc_prism_support): bottom triangle for -z,
    // top triangle otherwise; strict > keeps the first best vertex.
    int istart = (l.z < 0.0f) ? 0 : 3;
    float best = dot(prismV[istart], l);
    float3 bs = prismV[istart];
    for (int k = 1; k < 3; ++k) {
      float q = dot(prismV[istart + k], l);
      if (q > best) { best = q; bs = prismV[istart + k]; }
    }
    return bs;
  } else {  // CX_BOX
    s = float3(l.x >= 0.0f ? sz.x : -sz.x,
               l.y >= 0.0f ? sz.y : -sz.y,
               l.z >= 0.0f ? sz.z : -sz.z);
  }
  return p + R * s;
}

// Minkowski support: point on A-B boundary, plus witnesses on A and B.
struct CxVertex {
  float3 m;  // a - b
  float3 a;  // on A
  float3 b;  // on B
};

inline CxVertex cx_minkowski(int ta, float3 pa, float3x3 Ra, float3 sza,
                             int tb, float3 pb, float3x3 Rb, float3 szb,
                             float3 d, int ia, int ib,
                             device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  CxVertex v;
  v.a = cx_support(ta, pa, Ra, sza, d, ia, hull, hull_info, prismV);
  v.b = cx_support(tb, pb, Rb, szb, -d, ib, hull, hull_info, prismV);
  v.m = v.a - v.b;
  return v;
}

// Closest point to origin on segment ab; returns barycentric (u for a, v for b).
inline float3 cx_seg_closest(float3 a, float3 b, thread float* u, thread float* v) {
  float3 ab = b - a;
  float denom = dot(ab, ab);
  float t = denom > 1e-24f ? -dot(a, ab) / denom : 0.0f;
  t = clamp(t, 0.0f, 1.0f);
  *u = 1.0f - t;
  *v = t;
  return a + ab * t;
}

// Closest point to origin on triangle abc (Ericson 5.1); barycentric weights.
inline float3 cx_tri_closest(float3 a, float3 b, float3 c,
                             thread float* u, thread float* v, thread float* w) {
  float3 ab = b - a, ac = c - a, ap = -a;
  float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) { *u = 1; *v = 0; *w = 0; return a; }
  float3 bp = -b;
  float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) { *u = 0; *v = 1; *w = 0; return b; }
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float t = d1 / (d1 - d3);
    *u = 1 - t; *v = t; *w = 0; return a + ab * t;
  }
  float3 cp = -c;
  float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) { *u = 0; *v = 0; *w = 1; return c; }
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float t = d2 / (d2 - d6);
    *u = 1 - t; *v = 0; *w = t; return a + ac * t;
  }
  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    *u = 0; *v = 1 - t; *w = t; return b + (c - b) * t;
  }
  float denom = 1.0f / (va + vb + vc);
  *v = vb * denom; *w = vc * denom; *u = 1.0f - *v - *w;
  return a + ab * (*v) + ac * (*w);
}

// Closest point to origin on tetrahedron abcd (boundary enumeration with
// inside test via signed sub-volumes; also yields barycentric weights).
// Returns the origin itself when enclosed.
inline float cx_svol(float3 p, float3 q, float3 r, float3 s) {
  return dot(q - p, cross(r - p, s - p));
}
inline float3 cx_tet_closest(float3 a, float3 b, float3 c, float3 d,
                             thread float* u, thread float* v,
                             thread float* w, thread float* x) {
  float V = cx_svol(a, b, c, d);
  if (fabs(V) > 1e-30f) {
    float la = cx_svol(float3(0.0f), b, c, d) / V;
    float lb = cx_svol(a, float3(0.0f), c, d) / V;
    float lc = cx_svol(a, b, float3(0.0f), d) / V;
    float ld = cx_svol(a, b, c, float3(0.0f)) / V;
    if (la >= 0.0f && lb >= 0.0f && lc >= 0.0f && ld >= 0.0f) {
      *u = la; *v = lb; *w = lc; *x = ld;
      return float3(0.0f);
    }
  }
  // Outside: nearest of the 4 faces.
  float bu, bv, bw, best = 1e30f;
  float3 bp = a;
  float wu = 0.0f, wv = 0.0f, ww = 0.0f, wx = 0.0f;
  {
    float3 p = cx_tri_closest(b, c, d, &bu, &bv, &bw);
    float l = dot(p, p);
    if (l < best) { best = l; bp = p; wu = 0.0f; wv = bu; ww = bv; wx = bw; }
  }
  {
    float3 p = cx_tri_closest(a, c, d, &bu, &bv, &bw);
    float l = dot(p, p);
    if (l < best) { best = l; bp = p; wu = bu; wv = 0.0f; ww = bv; wx = bw; }
  }
  {
    float3 p = cx_tri_closest(a, b, d, &bu, &bv, &bw);
    float l = dot(p, p);
    if (l < best) { best = l; bp = p; wu = bu; wv = bv; ww = 0.0f; wx = bw; }
  }
  {
    float3 p = cx_tri_closest(a, b, c, &bu, &bv, &bw);
    float l = dot(p, p);
    if (l < best) { best = l; bp = p; wu = bu; wv = bv; ww = bw; wx = 0.0f; }
  }
  *u = wu; *v = wv; *w = ww; *x = wx;
  return bp;
}

// GJK distance query with Johnson reduction. Returns separation (>=0 when
// disjoint). overlap=true with a seed simplex when enclosed (use EPA).
// wpa/wpb are the witness points on A/B.
inline float cx_gjk(int ta, float3 pa, float3x3 Ra, float3 sza,
                    int tb, float3 pb, float3x3 Rb, float3 szb,
                    float3 d0, thread float3& wpa, thread float3& wpb,
                    thread bool& overlap, thread CxVertex* seed, thread int* nseed,
                    int ia, int ib,
                    device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  CxVertex S[4];
  int n = 0;
  float3 d = d0;
  if (dot(d, d) < 1e-24f) d = pa - pb;
  if (dot(d, d) < 1e-24f) d = float3(1.0f, 0.0f, 0.0f);
  float wts[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float3 closest = float3(0.0f);
  float span = 0.0f;
  overlap = false;
  *nseed = 0;
  for (int iter = 0; iter < CX_GJK_MAXITER; ++iter) {
    CxVertex v = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, d, ia, ib, hull, hull_info, prismV);
    if (n == 0) {
      S[0] = v; n = 1;
      closest = v.m; wts[0] = 1.0f;
      span = length(v.m);
    } else {
      // No progress past the current closest point: disjoint.
      if (dot(v.m, d) <= dot(closest, d) + 1e-9f * max(1.0f, length(closest))) {
        break;
      }
      // Duplicate vertex: cannot advance (degenerate curved patch).
      bool dup = false;
      for (int k = 0; k < n; ++k) {
        float3 dd = v.m - S[k].m;
        if (dot(dd, dd) < 1e-24f) { dup = true; break; }
      }
      if (dup || n >= 4) break;
      for (int k = 0; k < n; ++k) {
        float3 dd = v.m - S[k].m;
        span = max(span, length(dd));
      }
      S[n] = v; n++;
      // Johnson reduction to the nonzero-weight feature.
      float uu = 0.0f, vv = 0.0f, ww = 0.0f, xx = 0.0f;
      if (n == 2) {
        closest = cx_seg_closest(S[0].m, S[1].m, &uu, &vv);
      } else if (n == 3) {
        closest = cx_tri_closest(S[0].m, S[1].m, S[2].m, &uu, &vv, &ww);
      } else {
        closest = cx_tet_closest(S[0].m, S[1].m, S[2].m, S[3].m, &uu, &vv, &ww, &xx);
      }
      float nw[4] = {uu, vv, ww, xx};
      CxVertex T[4];
      int m = 0;
      for (int k = 0; k < n; ++k) {
        if (nw[k] > 1e-12f) { T[m] = S[k]; wts[m] = nw[k]; m++; }
      }
      for (int k = 0; k < m; ++k) S[k] = T[k];
      n = m;
      if (n == 0) break;
    }
    // Enclosure is scale-relative: an exactly-touching simplex in float32
    // sits ~1e-8 off the origin at shape scale ~0.1; absolute 1e-9 misses
    // genuine overlap while true separations stay far above this band.
    if (length(closest) <= 1e-6f * max(span, 1e-9f)) {
      overlap = true;
      break;
    }
    d = -closest;
  }
  // Witness interpolation + EPA seed.
  wpa = float3(0.0f); wpb = float3(0.0f);
  for (int k = 0; k < n; ++k) {
    wpa += S[k].a * wts[k];
    wpb += S[k].b * wts[k];
    seed[k] = S[k];
  }
  *nseed = n;
  if (overlap) return 0.0f;
  return length(closest);
}

// MPR penetration query (Minkowski Portal Refinement; pinned MuJoCo uses
// the same algorithm family via mjc_Convex/libccd for these pairs).
// Portal = tetrahedron (v0 = Minkowski center + face v1,v2,v3 facing
// outside). Returns depth (>0), normal, and witness points. Returns 0
// (touching, center-delta normal) in degenerate configurations.
#define CX_MPR_TOL 1e-6f
#define CX_MPR_MAXITER 64

// Outward face normal + signed dist (origin strictly inside iff dist > 0).
// Outward is defined against the opposite tetra vertex `o`.
inline float cx_face_out(float3 a, float3 b, float3 c, float3 o, thread float3& n) {
  n = cross(b - a, c - a);
  float l = length(n);
  if (l < 1e-24f) { n = float3(1.0f, 0.0f, 0.0f); return 0.0f; }
  n = n / l;
  if (dot(n, o - a) > 0.0f) n = -n;
  return dot(n, a);
}

// Portal face direction (v1,v2,v3): outward by construction (legacy helper
// for the portal-only MPR path; winding fixed by the caller).
inline float3 cx_portal_dir(float3 v1, float3 v2, float3 v3) {
  float3 n = cross(v2 - v1, v3 - v1);
  float l = length(n);
  return l > 1e-24f ? n / l : float3(1.0f, 0.0f, 0.0f);
}

// Expand portal with v4, keeping v0 and outward facing (libccd expandPortal).
inline void cx_expand_portal(thread CxVertex* P, thread CxVertex* v4) {
  float3 t = cross(v4->m, P[0].m);
  if (dot(P[1].m, t) > 0.0f) {
    if (dot(P[2].m, t) > 0.0f) {
      P[1] = *v4;
    } else {
      P[3] = *v4;
    }
  } else {
    if (dot(P[3].m, t) > 0.0f) {
      P[2] = *v4;
    } else {
      P[1] = *v4;
    }
  }
}

// Minimum over the four outward face distances (containment margin).
inline float cx_min_face_dist(thread CxVertex* P) {
  float3 n;
  float m = cx_face_out(P[1].m, P[2].m, P[3].m, P[0].m, n);
  m = min(m, cx_face_out(P[0].m, P[3].m, P[2].m, P[1].m, n));
  m = min(m, cx_face_out(P[0].m, P[1].m, P[3].m, P[2].m, n));
  m = min(m, cx_face_out(P[0].m, P[2].m, P[1].m, P[3].m, n));
  return m;
}

// Milestone 011 mesh face-snap: given a mesh-side portal witness (world)
// and the current portal normal u (A->B convention), snap to the
// most-facing hull face through the nearest hull vertex. side = +1 when
// the mesh is geom B (its outward faces oppose u), -1 when geom A (they
// align with u). The gate (within ~45 degrees of u) keeps genuinely
// side-on basins on the MPR result; face-like basins snap exactly.
inline bool cx_mesh_snap(int gi, float3 p, float3x3 R,
                         float3 rayDir, float side,
                         device const float* hull, device const int* hull_info,
                         thread float3& n_out,
                         thread float3& fw0, thread float3& fw1, thread float3& fw2) {
  // Contact face by center-ray-cast: ray from the hull centroid (interior
  // for convex hulls) toward the other body exits exactly through the
  // facing face. rayDir points from this geom toward the other
  // (body-center delta). Robust for deep/oblique penetration where portal
  // witnesses sit far from the contact region. Moller-Trumbore, nearest
  // hit wins; the 0.5 gate rejects grazing hits.
  int hoff = hull_info[gi * 9];
  int hcnt = hull_info[gi * 9 + 1];
  if (hcnt <= 0 || hcnt > CX_MESH_MAXSCAN) return false;
  float rl = length(rayDir);
  if (rl < 1e-24f) return false;
  float3 rd = rayDir / rl;
  float3 c = float3(0.0f);
  for (int k = 0; k < hcnt; ++k) {
    c += float3(hull[(hoff + k) * 3], hull[(hoff + k) * 3 + 1],
                hull[(hoff + k) * 3 + 2]);
  }
  c /= float(hcnt);
  float3 cw = p + R * c;
  // Backstep inside along -rd: the centroid can sit exactly on a hull
  // face plane (mesh bodies often rest at face height), which would make
  // the ray start on the boundary and miss. 1 mm ≪ hull scale.
  float3 org = cw - rd * 1e-3f;
  int foff = hull_info[gi * 9 + 3];
  int fcnt = hull_info[gi * 9 + 4];
  int noff = hull_info[gi * 9 + 2];
  if (fcnt < 0) return false;
  float bestT = 3.4028235e+38f;
  float3 bestN = float3(0.0f);
  float3 b0 = float3(0.0f), b1 = float3(0.0f), b2 = float3(0.0f);
  bool found = false;
  for (int f = 0; f < fcnt; ++f) {
    int l0 = int(hull[(foff + f) * 3]);
    int l1 = int(hull[(foff + f) * 3 + 1]);
    int l2 = int(hull[(foff + f) * 3 + 2]);
    if (l0 < 0 || l0 >= hcnt || l1 < 0 || l1 >= hcnt || l2 < 0 || l2 >= hcnt)
      continue;
    float3 nl = float3(hull[(noff + f) * 3], hull[(noff + f) * 3 + 1],
                       hull[(noff + f) * 3 + 2]);
    if (dot(nl, nl) < 0.25f) continue;
    float3 v0 = p + R * float3(hull[(hoff + l0) * 3], hull[(hoff + l0) * 3 + 1],
                               hull[(hoff + l0) * 3 + 2]);
    float3 v1 = p + R * float3(hull[(hoff + l1) * 3], hull[(hoff + l1) * 3 + 1],
                               hull[(hoff + l1) * 3 + 2]);
    float3 v2 = p + R * float3(hull[(hoff + l2) * 3], hull[(hoff + l2) * 3 + 1],
                               hull[(hoff + l2) * 3 + 2]);
    float3 e1 = v1 - v0, e2 = v2 - v0;
    float3 pv = cross(rd, e2);
    float det = dot(e1, pv);
    if (fabs(det) < 1e-24f) continue;
    float inv = 1.0f / det;
    float3 tv = org - v0;
    float uu2 = dot(tv, pv) * inv;
    if (uu2 < 0.0f || uu2 > 1.0f) continue;
    float3 qv = cross(tv, e1);
    float vv2 = dot(rd, qv) * inv;
    if (vv2 < 0.0f || uu2 + vv2 > 1.0f) continue;
    float t = dot(e2, qv) * inv;
    if (t < 1e-9f || t >= bestT) continue;
    float3 nw = R * (nl / sqrt(dot(nl, nl)));
    if (dot(nw, rd) <= 0.5f) continue;
    bestT = t;
    bestN = (side > 0.0f ? -nw : nw);
    b0 = v0; b1 = v1; b2 = v2;
    found = true;
  }
  if (!found) return false;
  n_out = bestN;
  fw0 = b0; fw1 = b1; fw2 = b2;
  return true;
}

// Terminal-scored face-snap variant (mesh-mesh path): nearest hull vertex
// to the refined portal witness, best face through it vs the terminal
// normal (45-degree gate). Used when both sides are meshes, where
// refinement converges the basin and only tilt correction is needed.
inline bool cx_mesh_snap_u(int gi, float3 p, float3x3 R,
                         float3 eaW, float3 u, float side,
                         device const float* hull, device const int* hull_info,
                         thread float3& n_out,
                         thread float3& fw0, thread float3& fw1, thread float3& fw2) {
  float3 d = eaW - p;
  float3 el = float3(dot(R[0], d), dot(R[1], d), dot(R[2], d));
  int hoff = hull_info[gi * 9];
  int hcnt = hull_info[gi * 9 + 1];
  if (hcnt <= 0 || hcnt > CX_MESH_MAXSCAN) return false;
  int bestV = -1;
  float bestD2 = 3.4028235e+38f;
  for (int k = 0; k < hcnt; ++k) {
    float3 v = float3(hull[(hoff + k) * 3], hull[(hoff + k) * 3 + 1],
                      hull[(hoff + k) * 3 + 2]);
    float3 dd = v - el;
    float q = dot(dd, dd);
    if (q < bestD2) { bestD2 = q; bestV = k; }
  }
  if (bestV < 0) return false;
  int foff = hull_info[gi * 9 + 3];
  int fcnt = hull_info[gi * 9 + 4];
  int noff = hull_info[gi * 9 + 2];
  if (fcnt < 0) return false;
  float bestS = 0.7f;
  float3 bestN = float3(0.0f);
  bool found = false;
  for (int f = 0; f < fcnt; ++f) {
    int l0 = int(hull[(foff + f) * 3]);
    int l1 = int(hull[(foff + f) * 3 + 1]);
    int l2 = int(hull[(foff + f) * 3 + 2]);
    if (l0 != bestV && l1 != bestV && l2 != bestV) continue;
    float3 nl = float3(hull[(noff + f) * 3], hull[(noff + f) * 3 + 1],
                       hull[(noff + f) * 3 + 2]);
    if (dot(nl, nl) < 0.25f) continue;
    float3 nw = R * (nl / sqrt(dot(nl, nl)));
    float s = dot(nw, u) * (side > 0.0f ? -1.0f : 1.0f);
    if (s > bestS) {
      bestS = s;
      bestN = (side > 0.0f ? -nw : nw);
      float3 g0 = float3(hull[(hoff + l0) * 3], hull[(hoff + l0) * 3 + 1],
                         hull[(hoff + l0) * 3 + 2]);
      float3 g1 = float3(hull[(hoff + l1) * 3], hull[(hoff + l1) * 3 + 1],
                         hull[(hoff + l1) * 3 + 2]);
      float3 g2 = float3(hull[(hoff + l2) * 3], hull[(hoff + l2) * 3 + 1],
                         hull[(hoff + l2) * 3 + 2]);
      fw0 = p + R * g0; fw1 = p + R * g1; fw2 = p + R * g2;
      found = true;
    }
  }
  if (!found) return false;
  n_out = bestN;
  return true;
}

// MPR penetration query (Minkowski Portal Refinement; pinned MuJoCo uses
// the same algorithm family via mjc_Convex/libccd for these pairs).
// Portal = tetrahedron (v0 = Minkowski center + face v1,v2,v3 facing
// outside). Returns depth (>0), normal, and witness points. Returns 0
// (touching, center-delta normal) in degenerate configurations.
// Milestone 011 box-face clamp: analytic box supports tie across whole
// faces but return a single sign-picked corner, which can sit far from a
// localized contact (e.g. tray corner 40 cm from a tetra). Given a query
// point near the contact and the contact normal, return the closest point
// on the most-facing box face (empty = keep query). Used for witness
// placement only; overlap depths use min/max projections (tie-invariant).
inline float3 cx_box_clamp(float3 p, float3x3 R, float3 sz, float3 q,
                           float3 fdir, thread bool& ok) {
  ok = false;
  float3 best = q;
  float bestS = 0.5f;
  for (int k = 0; k < 3; ++k) {
    float3 ck = float3(R[0][k], R[1][k], R[2][k]);
    for (int s = -1; s <= 1; s += 2) {
      float3 m = ck * float(s);
      float sc = dot(m, fdir);
      if (sc <= bestS) continue;
      float3 c = p + ck * (float(s) * ((k == 0) ? sz.x : ((k == 1) ? sz.y : sz.z)));
      float3 d = q - c;
      float3 dl = float3(dot(R[0], d), dot(R[1], d), dot(R[2], d));
      float3 h = sz;
      float3 dc = float3(min(max(dl.x, -h.x), h.x),
                         min(max(dl.y, -h.y), h.y),
                         min(max(dl.z, -h.z), h.z));
      // Pin to the face plane through c (c already carries the half-size
      // offset, so the normal coordinate is 0 here, not +/-sz).
      if (k == 0) dc.x = 0.0f;
      if (k == 1) dc.y = 0.0f;
      if (k == 2) dc.z = 0.0f;
      bestS = sc;
      best = c + R * dc;
      ok = true;
    }
  }
  return best;
}

// Milestone 011 mesh readout helper: given the final normal and optional
// snapped faces per side, compute the exact overlap interval along it with
// face-projected interface witnesses (analytic sides contribute their
// toward-supports). Returns false for touching/degenerate (caller falls
// back to the barycentric restore).
inline bool cx_mesh_readout(int ta, float3 pa, float3x3 Ra, float3 sza,
                            int tb, float3 pb, float3x3 Rb, float3 szb,
                            int ia, int ib,
                            device const float* hull, device const int* hull_info,
                            thread float3* prismV,
                            float3 nn, bool sA, float3 fA0, float3 fA1, float3 fA2,
                            bool sB, float3 fB0, float3 fB1, float3 fB2,
                            thread float& depth, thread float3& wmid) {
  float3 seedA = cx_support(ta, pa, Ra, sza, nn, ia, hull, hull_info, prismV);
  float3 seedB = cx_support(tb, pb, Rb, szb, -nn, ib, hull, hull_info, prismV);
  // Box sides clamp to the facing face (corner ties sit far from
  // localized contacts); other analytic supports are already unique.
  // Canonical order puts box first in mesh-box pairs, but both sides
  // are handled for robustness.
  if (ta == 6) {
    bool ok = false;
    float3 c = cx_box_clamp(pa, Ra, sza, seedB, nn, ok);
    if (ok) seedA = c;
  }
  if (tb == 6) {
    bool ok = false;
    float3 c = cx_box_clamp(pb, Rb, szb, seedA, -nn, ok);
    if (ok) seedB = c;
  }
  float3 ifA = seedA, ifB = seedB;
  if (tb == 7 && sB) {
    float u, v, w;
    float3 q = cx_tri_closest(fB0 - seedA, fB1 - seedA, fB2 - seedA,
                              &u, &v, &w);
    ifB = q + seedA;
    if (ta == 7 && sA) {
      float u2, v2, w2;
      float3 q2 = cx_tri_closest(fA0 - ifB, fA1 - ifB, fA2 - ifB,
                                  &u2, &v2, &w2);
      ifA = q2 + ifB;
    }
  } else if (ta == 7 && sA) {
    float u, v, w;
    float3 q = cx_tri_closest(fA0 - seedB, fA1 - seedB, fA2 - seedB,
                              &u, &v, &w);
    ifA = q + seedB;
  }
  float3 saP = cx_support(ta, pa, Ra, sza, nn, ia, hull, hull_info, prismV);
  float3 saM = cx_support(ta, pa, Ra, sza, -nn, ia, hull, hull_info, prismV);
  float3 sbP = cx_support(tb, pb, Rb, szb, nn, ib, hull, hull_info, prismV);
  float3 sbM = cx_support(tb, pb, Rb, szb, -nn, ib, hull, hull_info, prismV);
  float ha = dot(saP, nn), la = dot(saM, nn);
  float hb = dot(sbP, nn), lb = dot(sbM, nn);
  depth = min(ha, hb) - max(la, lb);
  if (depth <= 1e-9f) return false;
  wmid = (ifA + ifB) * 0.5f;
  return true;
}
inline float cx_mpr(int ta, float3 pa, float3x3 Ra, float3 sza,
                    int tb, float3 pb, float3x3 Rb, float3 szb,
                    bool raw, thread float3& normal,
                    thread float3& wpa, thread float3& wpb,
                    int ia, int ib,
                    device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  CxVertex P[4];
  P[0].a = pa; P[0].b = pb; P[0].m = pa - pb;
  float3 c0 = P[0].m;
  if (dot(c0, c0) < 1e-30f) {
    c0 = float3(CX_MPR_TOL * 10.0f, 0.0f, 0.0f);
    P[0].m = c0;
  }
  // v1 = support toward origin (libccd discoverPortal; the multiCCD path
  // varies geometry frames instead of this direction).
  float3 dir = -c0;
  float dl = length(dir);
  dir = dl > 1e-24f ? dir / dl : float3(1.0f, 0.0f, 0.0f);
  P[1] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dir, ia, ib, hull, hull_info, prismV);
  if (dot(P[1].m, dir) <= 0.0f) return 0.0f;
  // v2 perpendicular to plane (origin, v0, v1). The cross-product sign
  // is ambiguous: try both, keep the portal with the best containment
  // margin, and let the encapsulation loop below repair or reject it
  // (libccd refinePortal semantics).
  CxVertex v1v = P[1];
  CxVertex bestD[4];
  float bestMargin = -1e30f;
  bool established = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    P[1] = v1v;
    dir = cross(P[0].m, P[1].m);
    if (attempt == 1) dir = -dir;
    if (dot(dir, dir) < 1e-30f) {
      // Origin on segment v0-v1 (libccd findPenetrSegment): depth = |v1|.
      float l1 = length(P[1].m);
      normal = l1 > 1e-24f ? P[1].m / l1 : c0 / max(length(c0), 1e-24f);
      wpa = (P[1].a + P[1].b) * 0.5f;
      wpb = wpa;
      return l1;
    }
    dl = length(dir); dir = dir / dl;
    P[2] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dir, ia, ib, hull, hull_info, prismV);
    if (dot(P[2].m, dir) <= 0.0f) {
      if (attempt == 1 && !established) return 0.0f;
      continue;
    }
    // v3: perpendicular to plane (v0, v1, v2). libccd orients the portal
    // face "outside" by swapping v1/v2 when needed (keeps the v1-v2-v3
    // winding consistent for expandPortal); negating dir alone is wrong.
    {
      float3 e1 = P[1].m - P[0].m, e2 = P[2].m - P[0].m;
      dir = cross(e1, e2);
      dl = length(dir);
      if (dl < 1e-24f) {
        normal = c0 / max(length(c0), 1e-24f);
        wpa = (P[0].a + P[1].a + P[2].a) / 3.0f;
        wpb = (P[0].b + P[1].b + P[2].b) / 3.0f;
        return 0.0f;
      }
      dir = dir / dl;
      if (dot(dir, P[0].m) > 0.0f) {
        CxVertex tmp = P[1]; P[1] = P[2]; P[2] = tmp;
        dir = -dir;
      }
    }
    bool failed = false;
    for (int guard = 0; guard < 8; ++guard) {
      P[3] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dir, ia, ib, hull, hull_info, prismV);
      if (dot(P[3].m, dir) <= 0.0f) { failed = true; break; }
      // Origin outside face (v1,v0,v3)? -> v2 = v3 (libccd absolute test).
      float3 va = cross(P[1].m, P[3].m);
      if (dot(va, P[0].m) < 0.0f && dot(va, va) > 1e-30f) {
        P[2] = P[3];
      } else {
        // Origin outside face (v3,v0,v2)? -> v1 = v3.
        va = cross(P[3].m, P[2].m);
        if (dot(va, P[0].m) < 0.0f && dot(va, va) > 1e-30f) {
          P[1] = P[3];
        } else {
          break;  // portal established
        }
      }
      float3 e1 = P[1].m - P[0].m, e2 = P[2].m - P[0].m;
      dir = cross(e1, e2);
      dl = length(dir);
      if (dl < 1e-24f) { failed = true; break; }
      // libccd: no negate/swap here; the initial orientation is preserved.
      dir = dir / dl;
    }
    if (failed) {
      if (attempt == 1 && !established) return 0.0f;
      continue;
    }
    float margin = cx_min_face_dist(P);
    if (!established || margin > bestMargin) {
      bestMargin = margin;
      bestD[0] = P[0]; bestD[1] = P[1]; bestD[2] = P[2]; bestD[3] = P[3];
      established = true;
    }
    // Pinned discoverPortal takes the first established portal. Raw mode
    // (multiCCD restarts) keeps perturbation-driven basin hops that
    // spanning relies on; refined singles use least-bad selection for
    // float32 portal robustness, since accuracy (not diversity) governs.
    // The second winding is a fallback for invalid portals either way.
    if (raw && margin > -1e-7f) break;
  }
  if (!established) return 0.0f;
  P[0] = bestD[0]; P[1] = bestD[1]; P[2] = bestD[2]; P[3] = bestD[3];
  // Portal winding is already outward-consistent from discovery (libccd
  // orients via the v1/v2 swap above); do not flip here — a flipped
  // normal would mask a non-encapsulating portal from the loop below.
  // Encapsulation phase (libccd refinePortal): expand until the portal
  // face strictly separates the origin (dist > 0) or prove impossibility.
  for (int guard = 0; guard < 64; ++guard) {
    float3 fdir = cx_portal_dir(P[1].m, P[2].m, P[3].m);
    if (dot(fdir, P[1].m) > 0.0f) break;
    CxVertex v4 = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, fdir, ia, ib, hull, hull_info, prismV);
    if (dot(v4.m, fdir) <= 0.0f) return 0.0f;
    float adv0 = dot(v4.m, fdir) - dot(P[1].m, fdir);
    adv0 = min(adv0, dot(v4.m, fdir) - dot(P[2].m, fdir));
    adv0 = min(adv0, dot(v4.m, fdir) - dot(P[3].m, fdir));
    if (adv0 < CX_MPR_TOL) return 0.0f;
    cx_expand_portal(P, &v4);
  }
  // Refine portal face toward origin, then read depth. Tracks the best
  // (deepest) certified state so post-convergence wander cannot regress.
  // Mesh pairs additionally score face-snap against the refined terminal
  // normal afterwards (see below); analytic pairs use deepest-tracking
  // directly (010-qualified).
  float3 nrm = cx_portal_dir(P[1].m, P[2].m, P[3].m);
  float best = dot(nrm, P[1].m);
  CxVertex bestP[4];
  float bestVal = best;
  float3 bestNrm = nrm;
  bestP[0] = P[0]; bestP[1] = P[1]; bestP[2] = P[2]; bestP[3] = P[3];
  for (int iter = 0; iter < CX_MPR_MAXITER; ++iter) {
    nrm = cx_portal_dir(P[1].m, P[2].m, P[3].m);
    best = dot(nrm, P[1].m);
    if (best > bestVal) {
      bestVal = best; bestNrm = nrm;
      bestP[0] = P[0]; bestP[1] = P[1]; bestP[2] = P[2]; bestP[3] = P[3];
    }
    CxVertex v4 = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, nrm, ia, ib, hull, hull_info, prismV);
    float adv = dot(v4.m, nrm) - best;
    if (adv < CX_MPR_TOL * max(1.0f, best)) break;
    cx_expand_portal(P, &v4);
  }
  // Milestone 011 dual-candidate mesh readout. Candidate A scores
  // face-snap against the refined terminal normal (refinement basin);
  // candidate B scores center-ray snap against the body-delta direction
  // (body-delta basin, robust when the terminal portal wanders). Each
  // candidate reads depth as the exact overlap interval along its normal
  // with face-projected witnesses. Select: agree (10 deg) keeps terminal
  // for continuity, else min penetration depth (CPU EPA is the global
  // minimum). Analytic pairs use the barycentric restore below
  // (010-qualified).
  if (ta == 7 || tb == 7) {
    float bl = length(bestNrm);
    float3 uu = bl > 1e-24f ? bestNrm / bl : float3(1.0f, 0.0f, 0.0f);
    float3 d0 = pa - pb;
    float dl0 = length(d0);
    float3 rayA = dl0 > 1e-12f ? -d0 / dl0 : uu;
    float3 rayB = dl0 > 1e-12f ? d0 / dl0 : uu;
    // Candidate A: terminal-scored snap.
    float3 eaA = (bestP[1].a + bestP[2].a + bestP[3].a) / 3.0f;
    float3 eaB = (bestP[1].b + bestP[2].b + bestP[3].b) / 3.0f;
    float3 nA = uu, nB = uu;
    float3 fA0 = pa, fA1 = pa, fA2 = pa;
    float3 fB0 = pb, fB1 = pb, fB2 = pb;
    bool sA = (ta == 7) ? cx_mesh_snap_u(ia, pa, Ra, eaA, uu, -1.0f,
                                         hull, hull_info, nA, fA0, fA1, fA2)
                        : true;
    bool sB = (tb == 7) ? cx_mesh_snap_u(ib, pb, Rb, eaB, uu, 1.0f,
                                         hull, hull_info, nB, fB0, fB1, fB2)
                        : true;
    float3 nn = uu;
    if (sA && sB) {
      if (ta == 7 && tb == 7) {
        nn = nA + nB;
        float lnn = length(nn);
        nn = lnn > 1e-24f ? nn / lnn : uu;
      } else if (tb == 7) {
        nn = nB;
      } else {
        nn = nA;
      }
    }
    float dA = 0.0f;
    float3 wA = float3(0.0f);
    bool okA = cx_mesh_readout(ta, pa, Ra, sza, tb, pb, Rb, szb, ia, ib,
                               hull, hull_info, prismV, nn, sA, fA0, fA1, fA2,
                               sB, fB0, fB1, fB2, dA, wA);
    // Candidate B: center-ray snap.
    float3 mA = uu, mB = uu;
    float3 hA0 = pa, hA1 = pa, hA2 = pa;
    float3 hB0 = pb, hB1 = pb, hB2 = pb;
    bool qA = (ta == 7 && dl0 > 1e-12f)
                  ? cx_mesh_snap(ia, pa, Ra, rayA, -1.0f,
                                 hull, hull_info, mA, hA0, hA1, hA2)
                  : false;
    bool qB = (tb == 7 && dl0 > 1e-12f)
                  ? cx_mesh_snap(ib, pb, Rb, rayB, 1.0f,
                                 hull, hull_info, mB, hB0, hB1, hB2)
                  : false;
    float dB = 0.0f;
    float3 wB = float3(0.0f);
    float3 nnB = uu;
    bool okB = false;
    if (qA || qB) {
      if (qA && qB) {
        nnB = mA + mB;
        float lmB = length(nnB);
        nnB = lmB > 1e-24f ? nnB / lmB : uu;
      } else if (qB) {
        nnB = (tb == 7) ? mB : uu;
      } else {
        nnB = (ta == 7) ? mA : uu;
      }
      okB = cx_mesh_readout(ta, pa, Ra, sza, tb, pb, Rb, szb, ia, ib,
                            hull, hull_info, prismV, nnB, qA, hA0, hA1, hA2,
                            qB, hB0, hB1, hB2, dB, wB);
    }
    // Candidate C: axis-min over world axes (lateral-offset safety net).
    // Center-delta discovery follows lateral offsets into side basins
    // while the true minimum exit is vertical (measured fall-through
    // with lateral garbage witnesses); the minimum axis overlap is a
    // valid extent that can only approach the oracle global minimum from
    // above, never below it. Witness is the binding-supports midpoint.
    float dC = 0.0f;
    float3 wC = float3(0.0f);
    float3 nnC = uu;
    bool okC = false;
    {
      float3 ab = pb - pa;
      for (int ax = 0; ax < 3; ++ax) {
        float3 e = (ax == 0) ? float3(1.0f, 0.0f, 0.0f)
                   : ((ax == 1) ? float3(0.0f, 1.0f, 0.0f)
                                : float3(0.0f, 0.0f, 1.0f));
        float sg = dot(e, ab);
        float3 n = (sg >= 0.0f) ? e : -e;
        if (fabs(sg) < 1e-9f) {
          n = (dot(e, uu) >= 0.0f) ? e : -e;
        }
        float dd = 0.0f;
        float3 ww = float3(0.0f);
        if (cx_mesh_readout(ta, pa, Ra, sza, tb, pb, Rb, szb, ia, ib,
                            hull, hull_info, prismV, n, false, pa, pa, pa,
                            false, pb, pb, pb, dd, ww)) {
          if (!okC || dd < dC) {
            okC = true;
            dC = dd;
            wC = ww;
            nnC = n;
          }
        }
      }
    }
    if (okA && okB && dot(nn, nnB) > 0.985f) {
      // Agreeing basins keep terminal for continuity, unless the axis
      // candidate is substantially shallower (lateral-garbage agreement
      // with a true vertical exit falls here): axis overlap is a valid
      // extent, so a 2x-shallower axis minimum is the truer contact.
      float dab = min(dA, dB);
      if (okC && dC < 0.5f * dab) {
        normal = nnC;
        wpa = wC;
        wpb = wpa;
        return dC;
      }
      normal = nn;
      wpa = wA;
      wpb = wpa;
      return dA;
    }
    if (okC && (!okA || dC < dA) && (!okB || dC < dB)) {
      normal = nnC;
      wpa = wC;
      wpb = wpa;
      return dC;
    }
    if (okB && (!okA || dB < dA)) {
      normal = nnB;
      wpa = wB;
      wpb = wpa;
      return dB;
    }
    if (okA) {
      normal = nn;
      wpa = wA;
      wpb = wpa;
      return dA;
    }
    // All readouts failed (touching or side basin): fall through to the
    // barycentric restore shared with analytic pairs.
  }
  // Restore the best certified portal for witness/depth readout.
  P[0] = bestP[0]; P[1] = bestP[1]; P[2] = bestP[2]; P[3] = bestP[3];
  nrm = bestNrm;
  best = bestVal;
  // Witnesses: tetrahedron barycentric of origin (libccd findPos).
  {
    float3 t0 = P[0].m, t1 = P[1].m, t2 = P[2].m, t3 = P[3].m;
    float b0 = dot(cross(t1, t2), t3);
    float b1 = dot(cross(t3, t2), t0);
    float b2 = dot(cross(t0, t1), t3);
    float b3 = dot(cross(t2, t1), t0);
    float sum = b0 + b1 + b2 + b3;
    if (fabs(sum) < 1e-30f || sum < 0.0f) {
      float u, w, x;
      cx_tri_closest(t1, t2, t3, &u, &w, &x);
      wpa = P[1].a * u + P[2].a * w + P[3].a * x;
      wpb = P[1].b * u + P[2].b * w + P[3].b * x;
    } else {
      float inv = 1.0f / sum;
      wpa = (P[0].a * b0 + P[1].a * b1 + P[2].a * b2 + P[3].a * b3) * inv;
      wpb = (P[0].b * b0 + P[1].b * b1 + P[2].b * b2 + P[3].b * b3) * inv;
    }
    best = dot(nrm, P[1].m);
  }
  normal = nrm;
  return max(best, 0.0f);
}

// High-level convex entry points used by collide_pair (milestone 010).
// Normal convention: output normals point from geom1 to geom2, matching
// ContactGeom. Pinned sources: mjc_Convex / mjc_PlaneConvex.

// Ellipsoid-inside normal solve (pinned mjc_ellipsoidInside): ray-march
// from an interior point along the current normal to the surface.
inline bool cx_ellipsoid_inside(thread float3& nrm, float3 pos, float3 sz) {
  float3 S2inv = float3(1.0f / max(sz.x * sz.x, 1e-30f),
                        1.0f / max(sz.y * sz.y, 1e-30f),
                        1.0f / max(sz.z * sz.z, 1e-30f));
  float C = pos.x * pos.x * S2inv.x + pos.y * pos.y * S2inv.y
          + pos.z * pos.z * S2inv.z - 1.0f;
  if (C > 0.0f) return false;
  float nl = length(nrm);
  if (nl > 1e-24f) nrm = nrm / nl;
  for (int iter = 0; iter < 30; ++iter) {
    float A = nrm.x * nrm.x * S2inv.x + nrm.y * nrm.y * S2inv.y
            + nrm.z * nrm.z * S2inv.z;
    float B = pos.x * nrm.x * S2inv.x + pos.y * nrm.y * S2inv.y
            + pos.z * nrm.z * S2inv.z;
    float det = B * B - A * C;
    if (det < 1e-15f || A < 1e-15f) return iter > 0;
    float x = (-B + sqrt(det)) / A;
    if (x < 0.0f) return iter > 0;
    float3 pnt = pos + nrm * x;
    float3 nn = float3(pnt.x * S2inv.x, pnt.y * S2inv.y, pnt.z * S2inv.z);
    float ll = length(nn);
    if (ll > 1e-24f) nn = nn / ll;
    float change = length(nrm - nn);
    nrm = nn;
    if (change < 1e-6f) break;
  }
  return true;
}

// Ellipsoid-outside normal solve (pinned mjc_ellipsoidOutside): Newton on
// the Lagrange multiplier of the diagonal QCQP.
inline bool cx_ellipsoid_outside(thread float3& nrm, float3 pos, float3 sz) {
  float3 S2 = float3(sz.x * sz.x, sz.y * sz.y, sz.z * sz.z);
  float3 PS2 = float3(pos.x * pos.x * S2.x, pos.y * pos.y * S2.y,
                      pos.z * pos.z * S2.z);
  float la = 0.0f;
  for (int iter = 0; iter < 30; ++iter) {
    float3 R = float3(1.0f / (S2.x + la), 1.0f / (S2.y + la),
                      1.0f / (S2.z + la));
    float val = PS2.x * R.x * R.x + PS2.y * R.y * R.y + PS2.z * R.z * R.z - 1.0f;
    if (val < 1e-6f) break;
    float deriv = -2.0f * (PS2.x * R.x * R.x * R.x + PS2.y * R.y * R.y * R.y
                           + PS2.z * R.z * R.z * R.z);
    if (deriv > -1e-15f) break;
    float delta = -val / deriv;
    if (delta < 1e-6f) break;
    la += delta;
  }
  nrm = float3(pos.x / (S2.x + la), pos.y / (S2.y + la), pos.z / (S2.z + la));
  float ll = length(nrm);
  if (ll > 1e-24f) nrm = nrm / ll;
  return true;
}

// Analytic normal for one side (pinned mjc_fixNormal branches): maps the
// witness and input normal to the geom frame, replaces the normal for
// sphere/capsule/ellipsoid/cylinder surface points, maps back. Box and
// unknown types are left unprocessed.
inline void cx_fix_normal_side(int t, float3 p, float3x3 R, float3 sz,
                               float3 pos, float3 n_in,
                               thread float3& n_out, thread bool& done) {
  done = false;
  n_out = n_in;
  float3 d = pos - p;
  float3 pos1 = float3(dot(R[0], d), dot(R[1], d), dot(R[2], d));
  float3 nrm = float3(dot(R[0], n_in), dot(R[1], n_in), dot(R[2], n_in));
  if (t == 2) {
    nrm = pos1;
    done = true;
  } else if (t == 3) {
    if (pos1.z < -sz.y) nrm.z = pos1.z + sz.y;
    else if (pos1.z > sz.y) nrm.z = pos1.z - sz.y;
    else nrm.z = 0.0f;
    nrm.x = pos1.x;
    nrm.y = pos1.y;
    done = true;
  } else if (t == 4) {
    if (sz.x < 1e-15f || sz.y < 1e-15f || sz.z < 1e-15f) return;
    float dst1 = (pos1.x * pos1.x) / (sz.x * sz.x)
               + (pos1.y * pos1.y) / (sz.y * sz.y)
               + (pos1.z * pos1.z) / (sz.z * sz.z);
    if (dst1 <= 1.0f) done = cx_ellipsoid_inside(nrm, pos1, sz);
    else done = cx_ellipsoid_outside(nrm, pos1, sz);
  } else if (t == 5) {
    if (fabs(pos1.z) > 0.95f * sz.y) return;
    float dst1 = fabs(sz.y - fabs(pos1.z));
    float dst2 = fabs(sz.x - length(pos1.xy));
    if (dst1 < 0.25f * dst2) return;
    nrm = float3(pos1.x, pos1.y, 0.0f);
    done = true;
  } else {
    return;
  }
  if (done) {
    float l = length(nrm);
    n_out = l > 1e-24f ? R * (nrm / l) : n_in;
  }
}

// Single contact via GJK distance / EPA penetration.
inline int cx_single_contact(int ta, float3 pa, float3x3 Ra, float3 sza,
                             int tb, float3 pb, float3x3 Rb, float3 szb,
                             float margin, float3 d0, bool raw,
                             thread ContactGeom* out,
                             int ia, int ib,
                             device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  float3 wpa, wpb;
  CxVertex seed[4];
  int nseed = 0;
  bool overlap = false;
  float gap = cx_gjk(ta, pa, Ra, sza, tb, pb, Rb, szb, d0, wpa, wpb,
                     overlap, seed, &nseed, ia, ib, hull, hull_info, prismV);
  if (!overlap) {
    if (gap > margin) return 0;
    float3 delta = wpa - wpb;
    float dsq = dot(delta, delta);
    float3 n;
    if (dsq > 1e-24f) {
      n = delta / sqrt(dsq);
    } else {
      n = pa - pb;
      float nl = length(n);
      n = nl > 1e-12f ? n / nl : float3(1.0f, 0.0f, 0.0f);
    }
    // g1 -> g2 points from A toward B = -delta direction.
    out[0].normal = -n;
    out[0].dist = gap;
    out[0].pos = (wpa + wpb) * 0.5f;
    out[0].t1 = float3(0.0f);
    make_frame(out[0].normal, out[0].t1, out[0].t2);
    return 1;
  }
  float3 nrm, ea, eb;
  float depth = cx_mpr(ta, pa, Ra, sza, tb, pb, Rb, szb, raw, nrm, ea, eb,
                       ia, ib, hull, hull_info, prismV);
  // Guard against relative-enclosure false positives at ~1e-8 gaps.
  if (depth <= 1e-9f) return 0;
  // Enforce the A->B normal convention (matches the GJK gap path and the
  // dispatcher swap rule): portal winding can come back flipped after
  // refinement expansion replaces vertices, which used to emit B->A
  // witnesses that push bodies through each other in dynamics. Only the
  // normal is flipped (portal witnesses stay side-tagged: P.a is always
  // on A, P.b on B, by Minkowski construction). Degenerate centers keep
  // the portal orientation.
  {
    float3 ab = pb - pa;
    if (dot(ab, ab) > 1e-18f && dot(nrm, ab) < 0.0f) {
      nrm = -nrm;
    }
  }
  out[0].normal = nrm;
  out[0].dist = -depth;
  out[0].pos = (ea + eb) * 0.5f;
  // Analytic normal refinement for capsule-involved pairs (pinned
  // mjc_fixNormal branches): the libccd-family MPR facet normal carries
  // discretization error on curved patches; the analytic correction
  // matches pinned default (EPA-quality) normals. Wider application was
  // measured to regress currently-exact pairs: where the CPU oracle is
  // itself facet-tilted (e.g. cylinder-ellipsoid, whose MPR facet tilt
  // matches), analytic "correction" moves away from the oracle. Box
  // sides are never processed.
  // Pinned native path (NATIVECCD, the default) runs no fixNormal on
  // heightfield pairs; the libccd path does. Gate capsules accordingly.
  if ((ta == 3 || tb == 3) && ta != 1 && tb != 1) {
    float3 n0, n1;
    bool p0 = false, p1 = false;
    cx_fix_normal_side(ta, pa, Ra, sza, out[0].pos, nrm, n0, p0);
    cx_fix_normal_side(tb, pb, Rb, szb, out[0].pos, -nrm, n1, p1);    if (p0 && p1) {
      float3 c = n0 - n1;
      float l = length(c);
      if (l > 1e-24f) out[0].normal = c / l;
    } else if (p0) {
      out[0].normal = n0;
    } else if (p1) {
      out[0].normal = -n1;
    }
  }
  out[0].t1 = float3(0.0f);
  make_frame(out[0].normal, out[0].t1, out[0].t2);
  return 1;
}

// Plane-convex (pinned mjc_PlaneConvex): support point in -normal.
inline int collide_plane_convex(float3 p1, float4 q1,
                                int t2, float3 p2, float4 q2, float3 sz2,
                                float margin, thread ContactGeom* con,
                                int ia, int ib,
                                device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  float3 n = rotate_q(q1, float3(0.0f, 0.0f, 1.0f));
  float3x3 R2 = float3x3(rotate_q(q2, float3(1,0,0)),
                         rotate_q(q2, float3(0,1,0)),
                         rotate_q(q2, float3(0,0,1)));
  float3 s = cx_support(t2, p2, R2, sz2, -n, ib, hull, hull_info, prismV);
  float dist = dot(s - p1, n);
  if (dist > margin) return 0;
  con[0].dist = dist;
  con[0].normal = n;
  con[0].pos = s - n * (dist * 0.5f);
  con[0].t1 = float3(0.0f);
  make_frame(con[0].normal, con[0].t1, con[0].t2);
  return 1;
}

// Quat-based single convex contact (matrix conversion wrapper).
inline int collide_convex_single(int ta, float3 pa, float4 qa, float3 sza,
                                 int tb, float3 pb, float4 qb, float3 szb,
                                 float margin, thread ContactGeom* con,
                                 int ia, int ib,
                                 device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  float3x3 Ra = float3x3(rotate_q(qa, float3(1,0,0)),
                         rotate_q(qa, float3(0,1,0)),
                         rotate_q(qa, float3(0,0,1)));
  float3x3 Rb = float3x3(rotate_q(qb, float3(1,0,0)),
                         rotate_q(qb, float3(0,1,0)),
                         rotate_q(qb, float3(0,0,1)));
  float3 d0 = pa - pb;
  if (dot(d0, d0) < 1e-24f) d0 = float3(1.0f, 0.0f, 0.0f);
  return cx_single_contact(ta, pa, Ra, sza, tb, pb, Rb, szb, margin, d0, false, con,
                           ia, ib, hull, hull_info, prismV);
}

// In-place rigid rotation of a geom frame about origin (pinned
// mju_rotateFrame: xmat = rot*xmat; xpos -= (rot*rel - rel),
// rel = origin - xpos). Copies preserve the caller's frames, which is the
// pinned per-try restore.
inline void cx_rotate_frame(float3 origin, float3x3 rot,
                            thread float3x3* mat, thread float3* pos) {
  *mat = rot * (*mat);
  float3 rel = origin - (*pos);
  float3 vec = rot * rel - rel;
  *pos = (*pos) - vec;
}

// Rotation matrix for unit axis + angle (Rodrigues; pinned multiCCD builds
// the same rotation from an axis-angle quat).
inline float3x3 cx_axis_angle_mat(float3 a, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  float C = 1.0f - c;
  float x = a.x, y = a.y, z = a.z;
  return float3x3(
    c + C*x*x, s*z + C*x*y, -s*y + C*x*z,
    -s*z + C*x*y, c + C*y*y, s*x + C*y*z,
    s*y + C*x*z, -s*x + C*y*z, c + C*z*z);
}

// Tangent frame from contact normal (pinned mju_makeFrame: yaxis defaults
// to (0,1,0) unless |n.y| > 0.5, then (0,0,1); zaxis = x cross y).
inline void cx_contact_tangents(float3 n, thread float3& t1, thread float3& t2) {
  float nl = length(n);
  float3 nn = nl > 1e-24f ? n / nl : float3(1.0f, 0.0f, 0.0f);
  t1 = (nn.y < 0.5f && nn.y > -0.5f) ? float3(0.0f, 1.0f, 0.0f)
                                     : float3(0.0f, 0.0f, 1.0f);
  t1 = t1 - nn * dot(nn, t1);
  float l = length(t1);
  t1 = l > 1e-12f ? t1 / l : float3(1.0f, 0.0f, 0.0f);
  t2 = cross(nn, t1);
}

// Deterministic slot identity (R05-1): insertion sort of witnesses by
// world position (x, then y, then z). Manifold cardinality changes across
// steps keep stable row mapping instead of restart-order churn.
inline void cx_sort_contacts(thread ContactGeom* con, int n) {
  for (int i = 1; i < n; ++i) {
    ContactGeom key = con[i];
    int j = i - 1;
    while (j >= 0 && (con[j].pos.x > key.pos.x + 1e-9f ||
           (abs(con[j].pos.x - key.pos.x) <= 1e-9f && con[j].pos.y > key.pos.y + 1e-9f) ||
           (abs(con[j].pos.x - key.pos.x) <= 1e-9f && abs(con[j].pos.y - key.pos.y) <= 1e-9f &&
            con[j].pos.z > key.pos.z + 1e-9f))) {
      con[j + 1] = con[j];
      --j;
    }
    con[j + 1] = key;
  }
}

// R05-1 mesh-plane face manifold: the single deepest support seeds the
// witness, then hull verts within a 2 mm depth band of it join as face
// contacts (same normal, own depths/positions, tangent frames rebuilt).
// Bounded by maxn (host mesh budget 4) and sorted for stable identity.
inline int collide_mesh_plane_manifold(float3 p1, float4 q1,
                                int t2, float3 p2, float4 q2, float3 sz2,
                                float margin, int maxn, thread ContactGeom* con,
                                int ia, int ib,
                                device const float* hull, device const int* hull_info) {
  int n = collide_plane_convex(p1, q1, t2, p2, q2, sz2, margin, con,
                               ia, ib, hull, hull_info, nullptr);
  if (n == 0 || maxn <= 1 || t2 != CX_MESH) return n;
  float3 nrm = con[0].normal;
  float3x3 R2 = float3x3(rotate_q(q2, float3(1,0,0)),
                         rotate_q(q2, float3(0,1,0)),
                         rotate_q(q2, float3(0,0,1)));
  int off = hull_info[ib * 9];
  int cnt = hull_info[ib * 9 + 1];
  if (cnt < 0) cnt = 0;
  if (cnt > CX_MESH_MAXSCAN) cnt = CX_MESH_MAXSCAN;
  // Deepest hull vert in world (recompute to map ties to verts, not the
  // averaged support centroid).
  float deepest = 3.4028235e+38f;
  for (int k = 0; k < cnt; ++k) {
    float3 v = float3(hull[(off + k) * 3], hull[(off + k) * 3 + 1],
                      hull[(off + k) * 3 + 2]);
    float3 w = p2 + R2 * v;
    float dv = dot(w - p1, nrm);
    if (dv < deepest) deepest = dv;
  }
  // Collect face-band hull verts into a local manifold, then replace the
  // centroid-seeded primary: the averaged support point is not a hull
  // feature (face centers / edge midpoints), so pure vert witnesses give
  // true moment arms. The band always contains the deepest vert itself,
  // so vertex/edge/face contacts yield 1/2/4 witnesses respectively.
  ContactGeom verts[8];
  int nvert = 0;
  for (int k = 0; k < cnt && nvert < maxn && nvert < 8; ++k) {
    float3 v = float3(hull[(off + k) * 3], hull[(off + k) * 3 + 1],
                      hull[(off + k) * 3 + 2]);
    float3 w = p2 + R2 * v;
    float dv = dot(w - p1, nrm);
    if (dv > margin) continue;
    if (dv > deepest + 2e-3f) continue;
    bool dup = false;
    for (int j = 0; j < nvert; ++j) {
      float3 dq = (w - nrm * (dv * 0.5f)) - verts[j].pos;
      if (dot(dq, dq) <= 1e-6f) { dup = true; break; }
    }
    if (dup) continue;
    verts[nvert].dist = dv;
    verts[nvert].normal = nrm;
    verts[nvert].pos = w - nrm * (dv * 0.5f);
    cx_contact_tangents(nrm, verts[nvert].t1, verts[nvert].t2);
    ++nvert;
  }
  if (nvert == 0) return n;  // keep the single primary (no band verts)
  n = min(nvert, maxn);
  for (int k = 0; k < n; ++k) con[k] = verts[k];
  cx_sort_contacts(con, n);
  // Face-vert own-depths differ by < 2 mm; keep own depths (matches
  // per-point CPU depths) — depths already stored per witness.
  return n;
}
// Pinned multiCCD manifold port (MuJoCo mjc_Convex): after one primary
// witness, rotate both geometry frames in opposite directions about the
// primary contact point (tangent axes, +/-1e-3 rad), re-run the isolated
// single-contact solver, and keep witnesses distinct by the relative
// tolerance 1e-3*min(rbound). Frames are per-try copies, which is the
// pinned restore. New witnesses inherit the primary penetration.
inline int collide_convex_multi(int ta, float3 pa, float4 qa, float3 sza,
                                int tb, float3 pb, float4 qb, float3 szb,
                                float margin, int maxn,
                                float rb1, float rb2, int disable_multiccd,
                                thread ContactGeom* con,
                                int ia, int ib,
                                device const float* hull, device const int* hull_info,
                     thread float3* prismV) {
  float3x3 Ra = float3x3(rotate_q(qa, float3(1,0,0)),
                         rotate_q(qa, float3(0,1,0)),
                         rotate_q(qa, float3(0,0,1)));
  float3x3 Rb = float3x3(rotate_q(qb, float3(1,0,0)),
                         rotate_q(qb, float3(0,1,0)),
                         rotate_q(qb, float3(0,0,1)));
  float3 d0 = pa - pb;
  if (dot(d0, d0) < 1e-24f) d0 = float3(1.0f, 0.0f, 0.0f);
  int n = cx_single_contact(ta, pa, Ra, sza, tb, pb, Rb, szb, margin, d0, false, con,
                            ia, ib, hull, hull_info, prismV);
  if (n == 0 || maxn <= 1) return n;
  // Pinned gates: no restarts with MULTICCD disabled, without exactly one
  // primary witness, or when either geom is spherical/ellipsoidal.
  if (disable_multiccd) return n;
  if (n != 1) return n;
  if (ta == 2 || ta == 4 || tb == 2 || tb == 4) return n;
  float3 n0 = con[0].normal;
  float dd0 = con[0].dist;
  float3 P0 = con[0].pos;
  float3 t1, t2;
  cx_contact_tangents(n0, t1, t2);
  float tol = 1e-3f * min(rb1, rb2);
  float3 axes[2] = {t1, t2};
  float angles[2] = {-1e-3f, 1e-3f};
  for (int ax = 0; ax < 2 && n < maxn && n < 8; ++ax) {
    for (int an = 0; an < 2 && n < maxn && n < 8; ++an) {
      float3x3 rot = cx_axis_angle_mat(axes[ax], angles[an]);
      float3x3 inv = transpose(rot);
      float3x3 Ra1 = Ra, Rb1 = Rb;
      float3 pa1 = pa, pb1 = pb;
      cx_rotate_frame(P0, rot, &Ra1, &pa1);
      cx_rotate_frame(P0, inv, &Rb1, &pb1);
      ContactGeom cand;
      cand.dist = 1e30f; cand.pos = float3(0.0f);
      cand.normal = float3(0.0f); cand.t1 = float3(0.0f); cand.t2 = float3(0.0f);
      float3 d1 = pa1 - pb1;
      if (dot(d1, d1) < 1e-24f) d1 = float3(1.0f, 0.0f, 0.0f);
      if (!cx_single_contact(ta, pa1, Ra1, sza, tb, pb1, Rb1, szb,
                             margin, d1, true, &cand, ia, ib, hull, hull_info, prismV))
        continue;
      // Pinned distinctness: farther than tolerance from every previous.
      bool dup = false;
      for (int j = 0; j < n; ++j) {
        float3 dd = cand.pos - con[j].pos;
        if (dot(dd, dd) <= tol * tol) { dup = true; break; }
      }
      if (dup) continue;
      // R05-1 mesh-manifold coherence: restart witnesses must share the
      // primary face normal (2 deg). Tilted EPA basins on sharp hull
      // features are multi-basin regime (excluded from tight parity);
      // admitting them corrupts the manifold frame gates.
      if ((ta == CX_MESH || tb == CX_MESH) && dot(cand.normal, n0) < 0.9994f) continue;
      cand.dist = dd0;
      con[n] = cand;
      n++;
    }
  }
  // Interior compaction: drop restart witnesses strictly inside the span of
  // two accepted same-patch witnesses (uniform depth/normal). Pinned
  // multiCCD hops to patch ends; our basin-stable MPR additionally keeps
  // interior samples that inflate stiffness without information, so the
  // interior points are removed here to preserve force parity. con[0]
  // (primary) is never dropped.
  if (n >= 3) {
    bool drop[8];
    for (int i = 0; i < 8; ++i) drop[i] = false;
    for (int i = 1; i < n; ++i) {
      for (int a = 0; a < n && !drop[i]; ++a) {
        for (int b = a + 1; b < n && !drop[i]; ++b) {
          if (a == i || b == i) continue;
          if (fabs(con[i].dist - dd0) > 1e-6f) continue;
          if (fabs(con[a].dist - dd0) > 1e-6f
              || fabs(con[b].dist - dd0) > 1e-6f) continue;
          if (dot(con[i].normal, n0) < 0.9995f) continue;
          if (dot(con[a].normal, n0) < 0.9995f
              || dot(con[b].normal, n0) < 0.9995f) continue;
          float3 ab = con[b].pos - con[a].pos;
          float L2 = dot(ab, ab);
          if (L2 < 1e-12f) continue;
          float t = clamp(dot(con[i].pos - con[a].pos, ab) / L2, 0.0f, 1.0f);
          if (t < 0.05f || t > 0.95f) continue;
          float3 q = con[a].pos + ab * t;
          float3 dq = con[i].pos - q;
          if (dot(dq, dq) < 4e-6f) drop[i] = true;
        }
      }
    }
    for (int i = n - 1; i >= 1; --i) {
      if (drop[i] && n > 1) {
        con[i] = con[n - 1];
        n--;
      }
    }
  }
  cx_sort_contacts(con, n);
  return n;
}

// Milestone 012 heightfield collision (pinned mjc_ConvexHField, 3.10.0).
// The terrain is a grid of triangular prisms: surface triangle tops at
// data*size2 over (x, y), flat bottoms at -size3. Each overlapped prism is
// tested independently with one GJK/MPR witness (penetration margin 0);
// results transform back to world. Margin handling: qualified at margin 0
// (prism tops unexpanded, cutoff at 0, exactly matching the pinned path);
// margin > 0 runs the same cutoff and stays a documented restriction.
// Per-prism witnesses make native-vs-CPU comparison nearly 1:1 (same loop
// order, same support rule), unlike mesh singles-vs-manifolds.
inline int collide_hfield(
    float3 ph, float4 qh,
    int t2, float3 po, float4 qo, float3 szo, float rbo,
    float margin, int maxn, thread ContactGeom* con,
    int gih, int gio,
    device const float* hull, device const int* hull_info) {
  int doff = hull_info[gih * 9 + 5];
  int nrow = hull_info[gih * 9 + 6];
  int ncol = hull_info[gih * 9 + 7];
  int soff = hull_info[gih * 9 + 8];
  if (doff < 0 || nrow < 2 || ncol < 2 || soff < 0) return 0;
  if (maxn <= 0) return 0;
  float size0 = hull[soff];
  float size1 = hull[soff + 1];
  float size2 = hull[soff + 2];
  float size3 = hull[soff + 3];
  if (size0 <= 0.0f || size1 <= 0.0f) return 0;

  float3x3 Rh = float3x3(rotate_q(qh, float3(1,0,0)),
                         rotate_q(qh, float3(0,1,0)),
                         rotate_q(qh, float3(0,0,1)));
  float3x3 RhT = transpose(Rh);
  float3x3 Ro = float3x3(rotate_q(qo, float3(1,0,0)),
                         rotate_q(qo, float3(0,1,0)),
                         rotate_q(qo, float3(0,0,1)));
  // Other geom center in the hfield frame.
  float3 lp = RhT * (po - ph);

  // Early return: box-sphere test with rbound + margin.
  float radius = rbo + margin;
  if ((size0 < lp.x - radius) || (-size0 > lp.x + radius) ||
      (size1 < lp.y - radius) || (-size1 > lp.y + radius) ||
      (size2 < lp.z - radius) || (-size3 > lp.z + radius)) {
    return 0;
  }

  // Other geom frame in hfield coordinates; GJK runs in this frame.
  float3x3 Roh = RhT * Ro;
  float3 poh = lp;
  float3 I0 = float3(1,0,0), I1 = float3(0,1,0), I2 = float3(0,0,1);
  float3x3 Im = float3x3(I0, I1, I2);

  // Other geom AABB in the hfield frame via supports.
  float3 sx = cx_support(t2, poh, Roh, szo, I0, gio, hull, hull_info, nullptr);
  float3 nx = cx_support(t2, poh, Roh, szo, -I0, gio, hull, hull_info, nullptr);
  float3 sy = cx_support(t2, poh, Roh, szo, I1, gio, hull, hull_info, nullptr);
  float3 ny = cx_support(t2, poh, Roh, szo, -I1, gio, hull, hull_info, nullptr);
  float3 sz = cx_support(t2, poh, Roh, szo, I2, gio, hull, hull_info, nullptr);
  float3 nz = cx_support(t2, poh, Roh, szo, -I2, gio, hull, hull_info, nullptr);
  float xmax = sx.x, xmin = nx.x;
  float ymax = sy.y, ymin = ny.y;
  float zmax = sz.z, zmin = nz.z;

  // AABB box-box test.
  if ((xmin - margin > size0) || (xmax + margin < -size0) ||
      (ymin - margin > size1) || (ymax + margin < -size1) ||
      (zmin - margin > size2) || (zmax + margin < -size3)) {
    return 0;
  }

  // Sub-grid bounds (pinned floor/ceil + clamp).
  float fx = (float)(ncol - 1) / max(2.0f * size0, 1e-30f);
  float fy = (float)(nrow - 1) / max(2.0f * size1, 1e-30f);
  int cmin = (int)floor((xmin + size0) * fx);
  int cmax = (int)ceil((xmax + size0) * fx);
  int rmin = (int)floor((ymin + size1) * fy);
  int rmax = (int)ceil((ymax + size1) * fy);
  cmin = max(0, cmin);
  cmax = min(ncol - 1, cmax);
  rmin = max(0, rmin);
  rmax = min(nrow - 1, rmax);

  float dx = (2.0f * size0) / (float)(ncol - 1);
  float dy = (2.0f * size1) / (float)(nrow - 1);

  int ncon = 0;
  for (int r = rmin; r < rmax; ++r) {
    for (int c = cmin + 1; c <= cmax; ++c) {
      for (int i = 0; i < 2; ++i) {
        // Surface-triangle nodes in pinned traversal order: the last
        // three pushed verts; i selects the row offset within the strip.
        int n0r, n0c, n1r, n1c, n2r, n2c;
        if (i == 0) {
          n0r = r + 1; n0c = c - 1;
          n1r = r;     n1c = c - 1;
          n2r = r + 1; n2c = c;
        } else {
          n0r = r;     n0c = c - 1;
          n1r = r + 1; n1c = c;
          n2r = r;     n2c = c;
        }
        float h0 = hull[doff + n0r * ncol + n0c] * size2;
        float h1 = hull[doff + n1r * ncol + n1c] * size2;
        float h2 = hull[doff + n2r * ncol + n2c] * size2;
        // Prism height test.
        if (h0 < zmin && h1 < zmin && h2 < zmin) continue;
        float3 v0 = float3(dx * (float)n0c - size0, dy * (float)n0r - size1, 0.0f);
        float3 v1 = float3(dx * (float)n1c - size0, dy * (float)n1r - size1, 0.0f);
        float3 v2 = float3(dx * (float)n2c - size0, dy * (float)n2r - size1, 0.0f);
        float3 bot = float3(0.0f, 0.0f, -size3);
        thread float3 prismV[6];
        prismV[0] = v0 + bot; prismV[1] = v1 + bot; prismV[2] = v2 + bot;
        prismV[3] = v0 + float3(0.0f, 0.0f, h0);
        prismV[4] = v1 + float3(0.0f, 0.0f, h1);
        prismV[5] = v2 + float3(0.0f, 0.0f, h2);
        float3 centroid = (v0 + v1 + v2) / 3.0f
                        + float3(0.0f, 0.0f, (h0 + h1 + h2) / 3.0f - size3) * 0.5f;
        float3 d0 = centroid - poh;
        if (dot(d0, d0) < 1e-24f) d0 = float3(1.0f, 0.0f, 0.0f);
        ContactGeom cand;
        cand.dist = 1e30f; cand.pos = float3(0.0f);
        cand.normal = float3(0.0f); cand.t1 = float3(0.0f); cand.t2 = float3(0.0f);
        float3 zsz = float3(0.0f);
        int got = cx_single_contact(1, centroid, Im, zsz,
                                    t2, poh, Roh, szo,
                                    margin, d0, false, &cand,
                                    gih, gio, hull, hull_info, prismV);
        if (got) {
          con[ncon].dist = cand.dist;
          con[ncon].normal = Rh * cand.normal;
          con[ncon].pos = Rh * cand.pos + ph;
          con[ncon].t1 = Rh * cand.t1;
          con[ncon].t2 = Rh * cand.t2;
          ++ncon;
          if (ncon >= maxn) return ncon;
          if (ncon >= 50) return ncon;  // pinned mjMAXCONPAIR cap
        }
      }
    }
  }
  return ncon;
}
