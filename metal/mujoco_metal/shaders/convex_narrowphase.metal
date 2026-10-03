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
#define CX_SPHERE 2
#define CX_CAPSULE 3
#define CX_ELLIPSOID 4
#define CX_CYLINDER 5
#define CX_BOX 6

#define CX_MINVAL 1e-15f
#define CX_EPA_MAXFACES 64
#define CX_EPA_MAXITER 50
#define CX_GJK_MAXITER 64

// Analytic support point of a convex geom in world direction d.
inline float3 cx_support(int type, float3 p, float3x3 R, float3 sz, float3 d) {
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
                             float3 d) {
  CxVertex v;
  v.a = cx_support(ta, pa, Ra, sza, d);
  v.b = cx_support(tb, pb, Rb, szb, -d);
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
                    thread bool& overlap, thread CxVertex* seed, thread int* nseed) {
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
    CxVertex v = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, d);
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

// MPR penetration query (Minkowski Portal Refinement; pinned MuJoCo uses
// the same algorithm family via mjc_Convex/libccd for these pairs).
// Portal = tetrahedron (v0 = Minkowski center + face v1,v2,v3 facing
// outside). Returns depth (>0), normal, and witness points. Returns 0
// (touching, center-delta normal) in degenerate configurations.
inline float cx_mpr(int ta, float3 pa, float3x3 Ra, float3 sza,
                    int tb, float3 pb, float3x3 Rb, float3 szb,
                    bool raw, thread float3& normal,
                    thread float3& wpa, thread float3& wpb) {
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
  P[1] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dir);
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
    P[2] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dir);
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
      P[3] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dir);
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
    CxVertex v4 = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, fdir);
    if (dot(v4.m, fdir) <= 0.0f) return 0.0f;
    float adv0 = dot(v4.m, fdir) - dot(P[1].m, fdir);
    adv0 = min(adv0, dot(v4.m, fdir) - dot(P[2].m, fdir));
    adv0 = min(adv0, dot(v4.m, fdir) - dot(P[3].m, fdir));
    if (adv0 < CX_MPR_TOL) return 0.0f;
    cx_expand_portal(P, &v4);
  }
  // Refine portal face toward origin, then read depth. Tracks the best
  // (deepest) certified state so post-convergence wander cannot regress.
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
    CxVertex v4 = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, nrm);
    float adv = dot(v4.m, nrm) - best;
    if (adv < CX_MPR_TOL * max(1.0f, best)) break;
    cx_expand_portal(P, &v4);
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
                             thread ContactGeom* out) {
  float3 wpa, wpb;
  CxVertex seed[4];
  int nseed = 0;
  bool overlap = false;
  float gap = cx_gjk(ta, pa, Ra, sza, tb, pb, Rb, szb, d0, wpa, wpb,
                     overlap, seed, &nseed);
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
  float depth = cx_mpr(ta, pa, Ra, sza, tb, pb, Rb, szb, raw, nrm, ea, eb);
  // Guard against relative-enclosure false positives at ~1e-8 gaps.
  if (depth <= 1e-9f) return 0;
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
  if (ta == 3 || tb == 3) {
    float3 n0, n1;
    bool p0 = false, p1 = false;
    cx_fix_normal_side(ta, pa, Ra, sza, out[0].pos, nrm, n0, p0);
    cx_fix_normal_side(tb, pb, Rb, szb, out[0].pos, -nrm, n1, p1);
    if (p0 && p1) {
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
                                float margin, thread ContactGeom* con) {
  float3 n = rotate_q(q1, float3(0.0f, 0.0f, 1.0f));
  float3x3 R2 = float3x3(rotate_q(q2, float3(1,0,0)),
                         rotate_q(q2, float3(0,1,0)),
                         rotate_q(q2, float3(0,0,1)));
  float3 s = cx_support(t2, p2, R2, sz2, -n);
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
                                 float margin, thread ContactGeom* con) {
  float3x3 Ra = float3x3(rotate_q(qa, float3(1,0,0)),
                         rotate_q(qa, float3(0,1,0)),
                         rotate_q(qa, float3(0,0,1)));
  float3x3 Rb = float3x3(rotate_q(qb, float3(1,0,0)),
                         rotate_q(qb, float3(0,1,0)),
                         rotate_q(qb, float3(0,0,1)));
  float3 d0 = pa - pb;
  if (dot(d0, d0) < 1e-24f) d0 = float3(1.0f, 0.0f, 0.0f);
  return cx_single_contact(ta, pa, Ra, sza, tb, pb, Rb, szb, margin, d0, false, con);
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
                                thread ContactGeom* con) {
  float3x3 Ra = float3x3(rotate_q(qa, float3(1,0,0)),
                         rotate_q(qa, float3(0,1,0)),
                         rotate_q(qa, float3(0,0,1)));
  float3x3 Rb = float3x3(rotate_q(qb, float3(1,0,0)),
                         rotate_q(qb, float3(0,1,0)),
                         rotate_q(qb, float3(0,0,1)));
  float3 d0 = pa - pb;
  if (dot(d0, d0) < 1e-24f) d0 = float3(1.0f, 0.0f, 0.0f);
  int n = cx_single_contact(ta, pa, Ra, sza, tb, pb, Rb, szb, margin, d0, false, con);
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
                             margin, d1, true, &cand))
        continue;
      // Pinned distinctness: farther than tolerance from every previous.
      bool dup = false;
      for (int j = 0; j < n; ++j) {
        float3 dd = cand.pos - con[j].pos;
        if (dot(dd, dd) <= tol * tol) { dup = true; break; }
      }
      if (dup) continue;
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
  return n;
}
