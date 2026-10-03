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

// Ensure strict origin containment, repairing with support points.
// Returns false when unfixable (caller reports no contact).
inline bool cx_ensure_contained(thread CxVertex* P, int ta, float3 pa, float3x3 Ra, float3 sza,
                                int tb, float3 pb, float3x3 Rb, float3 szb) {
  for (int guard = 0; guard < 16; ++guard) {
    float3 n0, n1, n2, n3;
    float d0 = cx_face_out(P[1].m, P[2].m, P[3].m, P[0].m, n0);
    float d1 = cx_face_out(P[0].m, P[3].m, P[2].m, P[1].m, n1);
    float d2 = cx_face_out(P[0].m, P[1].m, P[3].m, P[2].m, n2);
    float d3 = cx_face_out(P[0].m, P[2].m, P[1].m, P[3].m, n3);
    float worst = d0;
    int wi = 0;
    float3 wn = n0;
    if (d1 < worst) { worst = d1; wi = 1; wn = n1; }
    if (d2 < worst) { worst = d2; wi = 2; wn = n2; }
    if (d3 < worst) { worst = d3; wi = 3; wn = n3; }
    if (worst > 0.0f) return true;
    CxVertex v = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, wn);
    if (dot(v.m, wn) <= worst) return false;
    // Replace the vertex opposite the worst face (faces omit vertex wi).
    int drop = wi;
    P[drop] = v;
  }
  return false;
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

// Strict origin-in-tetrahedron test (all faces separate origin).
inline bool cx_tet_contains(thread CxVertex* P) {
  float3 n;
  if (cx_face_out(P[1].m, P[2].m, P[3].m, P[0].m, n) <= 0.0f) return false;
  if (cx_face_out(P[0].m, P[3].m, P[2].m, P[1].m, n) <= 0.0f) return false;
  if (cx_face_out(P[0].m, P[1].m, P[3].m, P[2].m, n) <= 0.0f) return false;
  if (cx_face_out(P[0].m, P[2].m, P[1].m, P[3].m, n) <= 0.0f) return false;
  return true;
}

// MPR penetration query (Minkowski Portal Refinement; pinned MuJoCo uses
// the same algorithm family via mjc_Convex/libccd for these pairs).
// Portal = tetrahedron (v0 = Minkowski center + face v1,v2,v3 facing
// outside). Returns depth (>0), normal, and witness points. Returns 0
// (touching, center-delta normal) in degenerate configurations.
inline float cx_mpr(int ta, float3 pa, float3x3 Ra, float3 sza,
                    int tb, float3 pb, float3x3 Rb, float3 szb,
                    thread float3& normal, thread float3& wpa, thread float3& wpb) {
  CxVertex P[4];
  P[0].a = pa; P[0].b = pb; P[0].m = pa - pb;
  float3 c0 = P[0].m;
  if (dot(c0, c0) < 1e-30f) {
    c0 = float3(CX_MPR_TOL * 10.0f, 0.0f, 0.0f);
    P[0].m = c0;
  }
  // v1 = support toward origin.
  float3 dir = -c0;
  float dl = length(dir);
  dir = dl > 1e-24f ? dir / dl : float3(1.0f, 0.0f, 0.0f);
  P[1] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dir);
  if (dot(P[1].m, dir) <= 0.0f) return 0.0f;
  // v2 perpendicular to plane (origin, v0, v1). The cross-product sign
  // is ambiguous: try both and keep a strictly containing portal.
  CxVertex v1v = P[1];
  bool have_portal = false;
  for (int attempt = 0; attempt < 2 && !have_portal; ++attempt) {
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
      if (attempt == 1) return 0.0f;
      continue;
    }
    // v3: perpendicular to plane (v0, v1, v2), oriented outside origin.
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
      if (dot(dir, P[0].m) > 0.0f) dir = -dir;
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
      dir = dir / dl;
      if (dot(dir, P[0].m) > 0.0f) dir = -dir;
    }
    if (failed) {
      if (attempt == 1) return 0.0f;
      continue;
    }
    if (cx_tet_contains(P)) {
      have_portal = true;
    } else if (attempt == 1) {
      // Neither winding contains the origin: report no contact rather
      // than refining an invalid portal (prevents wander like capsule-cyl
      // converging to a tilted face with overestimated depth).
      normal = c0 / max(length(c0), 1e-24f);
      wpa = pa; wpb = pb;
      return 0.0f;
    }
  }
  // Force the portal face outward (origin strictly on the inner side).
  // All subsequent faces are maintained outward by expandPortal's rule.
  {
    float3 n0 = cx_portal_dir(P[1].m, P[2].m, P[3].m);
    if (dot(n0, P[1].m) < 0.0f) {
      CxVertex tmp = P[2]; P[2] = P[3]; P[3] = tmp;
    }
  }
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

// Single contact via GJK distance / EPA penetration.
inline int cx_single_contact(int ta, float3 pa, float3x3 Ra, float3 sza,
                             int tb, float3 pb, float3x3 Rb, float3 szb,
                             float margin, float3 d0,
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
  float depth = cx_mpr(ta, pa, Ra, sza, tb, pb, Rb, szb, nrm, ea, eb);
  // Guard against relative-enclosure false positives at ~1e-8 gaps.
  if (depth <= 1e-9f) return 0;
  out[0].normal = nrm;
  out[0].dist = -depth;
  out[0].pos = (ea + eb) * 0.5f;
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
  return cx_single_contact(ta, pa, Ra, sza, tb, pb, Rb, szb, margin, d0, con);
}

// General convex-convex with perturbed-restart witnesses (up to maxn).
inline int collide_convex_multi(int ta, float3 pa, float4 qa, float3 sza,
                                int tb, float3 pb, float4 qb, float3 szb,
                                float margin, int maxn, thread ContactGeom* con) {
  float3x3 Ra = float3x3(rotate_q(qa, float3(1,0,0)),
                         rotate_q(qa, float3(0,1,0)),
                         rotate_q(qa, float3(0,0,1)));
  float3x3 Rb = float3x3(rotate_q(qb, float3(1,0,0)),
                         rotate_q(qb, float3(0,1,0)),
                         rotate_q(qb, float3(0,0,1)));
  float3 d0 = pa - pb;
  if (dot(d0, d0) < 1e-24f) d0 = float3(1.0f, 0.0f, 0.0f);
  int n = cx_single_contact(ta, pa, Ra, sza, tb, pb, Rb, szb, margin, d0, con);
  if (n == 0 || maxn <= 1) return n;
  float3 n0 = con[0].normal;
  float3 t1 = abs(n0.y) < 0.9f ? normalize(cross(n0, float3(0,1,0)))
                               : normalize(cross(n0, float3(1,0,0)));
  float3 t2 = cross(n0, t1);
  float3 dirs[4] = {t1 * 0.5f + n0, -t1 * 0.5f + n0,
                    t2 * 0.5f + n0, -t2 * 0.5f + n0};
  for (int k = 0; k < 4 && n < maxn && n < 8; ++k) {
    ContactGeom cand;
    cand.dist = 1e30f; cand.pos = float3(0.0f);
    cand.normal = float3(0.0f); cand.t1 = float3(0.0f); cand.t2 = float3(0.0f);
    if (!cx_single_contact(ta, pa, Ra, sza, tb, pb, Rb, szb, margin, dirs[k], &cand))
      continue;
    bool dup = false;
    for (int j = 0; j < n; ++j) {
      float3 dd = cand.pos - con[j].pos;
      if (dot(dd, dd) < 1e-8f) { dup = true; break; }
    }
    if (!dup) {
      con[n] = cand;
      n++;
    }
  }
  return n;
}
