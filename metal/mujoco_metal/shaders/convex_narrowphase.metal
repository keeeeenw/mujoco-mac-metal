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

// GJK distance query. Returns separation (>=0 when disjoint; ~0 on touch).
// pa/pb are the witness points. ok=false means overlap (use EPA).
inline float cx_gjk(int ta, float3 pa, float3x3 Ra, float3 sza,
                    int tb, float3 pb, float3x3 Rb, float3 szb,
                    float3 d0, thread float3& wpa, thread float3& wpb,
                    thread bool& ok) {
  CxVertex S[4];
  int n = 0;
  float3 d = d0;
  if (dot(d, d) < 1e-24f) d = pa - pb;
  if (dot(d, d) < 1e-24f) d = float3(1.0f, 0.0f, 0.0f);
  S[0] = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, d);
  n = 1;
  d = -S[0].m;
  float prev_dist = 1e30f;
  ok = true;
  for (int iter = 0; iter < CX_GJK_MAXITER; ++iter) {
    if (dot(d, d) < 1e-24f) { ok = false; break; }
    CxVertex v = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, d);
    // No progress past the current closest point: disjoint or touching.
    float3 closest;
    if (n == 1) {
      closest = S[0].m;
      wpa = S[0].a; wpb = S[0].b;
    } else if (n == 2) {
      float u, w;
      closest = cx_seg_closest(S[0].m, S[1].m, &u, &w);
      wpa = S[0].a * u + S[1].a * w;
      wpb = S[0].b * u + S[1].b * w;
    } else {
      float u, w, x;
      closest = cx_tri_closest(S[0].m, S[1].m, S[2].m, &u, &w, &x);
      wpa = S[0].a * u + S[1].a * w + S[2].a * x;
      wpb = S[0].b * u + S[1].b * w + S[2].b * x;
    }
    float dist = length(closest);
    if (dot(v.m, d) < dot(closest, d) + 1e-9f * max(1.0f, dist)) {
      return dist;  // converged: separating distance
    }
    // Grow simplex toward origin.
    if (n == 1) {
      S[1] = v; n = 2;
    } else if (n == 2) {
      S[2] = v; n = 3;
      // Check enclosure: origin inside triangle => overlap.
      float uu, vv, ww;
      float3 cc = cx_tri_closest(S[0].m, S[1].m, S[2].m, &uu, &vv, &ww);
      if (length(cc) < 1e-9f) { ok = false; break; }
    } else {
      S[0] = S[1]; S[1] = S[2]; S[2] = v;
      float uu, vv, ww;
      float3 cc = cx_tri_closest(S[0].m, S[1].m, S[2].m, &uu, &vv, &ww);
      if (length(cc) < 1e-9f) { ok = false; break; }
    }
    // New search direction = -closest.
    if (n == 1) d = -S[0].m;
    else if (n == 2) {
      float u, w;
      float3 c = cx_seg_closest(S[0].m, S[1].m, &u, &w);
      d = -c;
    } else {
      float u, w, x;
      float3 c = cx_tri_closest(S[0].m, S[1].m, S[2].m, &u, &w, &x);
      d = -c;
      if (dot(d, d) < 1e-24f) { ok = false; break; }
    }
    if (fabs(prev_dist - length(d)) < 1e-9f * max(1.0f, prev_dist)) {
      // Stalled: report current distance.
      float uu, vv, ww;
      if (n == 3) {
        float3 c = cx_tri_closest(S[0].m, S[1].m, S[2].m, &uu, &vv, &ww);
        wpa = S[0].a * uu + S[1].a * vv + S[2].a * ww;
        wpb = S[0].b * uu + S[1].b * vv + S[2].b * ww;
        return length(c);
      }
      return length(d);
    }
    prev_dist = length(d);
  }
  if (ok) {
    float uu, vv, ww;
    if (n == 3) {
      float3 c = cx_tri_closest(S[0].m, S[1].m, S[2].m, &uu, &vv, &ww);
      wpa = S[0].a * uu + S[1].a * vv + S[2].a * ww;
      wpb = S[0].b * uu + S[1].b * vv + S[2].b * ww;
      return length(c);
    } else if (n == 2) {
      float u, w;
      float3 c = cx_seg_closest(S[0].m, S[1].m, &u, &w);
      wpa = S[0].a * u + S[1].a * w;
      wpb = S[0].b * u + S[1].b * w;
      return length(c);
    }
    wpa = S[0].a; wpb = S[0].b;
    return length(S[0].m);
  }
  return 0.0f;
}

// EPA penetration query from an overlap state. Returns depth (>0), normal
// (points A -> B... note: B - A direction), and witness points.
inline float cx_epa(int ta, float3 pa, float3x3 Ra, float3 sza,
                    int tb, float3 pb, float3x3 Rb, float3 szb,
                    thread float3& normal, thread float3& wpa, thread float3& wpb) {
  // Seed tetrahedron: coordinate-axis supports around the centroid delta.
  float3 d0 = pa - pb;
  CxVertex V[8];
  int nv = 0;
  float3 dirs[6] = {float3(1,0,0), float3(-1,0,0), float3(0,1,0),
                    float3(0,-1,0), float3(0,0,1), float3(0,0,-1)};
  for (int i = 0; i < 6 && nv < 4; ++i) {
    CxVertex v = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, dirs[i]);
    bool dup = false;
    for (int j = 0; j < nv; ++j) {
      if (dot(V[j].m - v.m, V[j].m - v.m) < 1e-18f) { dup = true; break; }
    }
    if (!dup) { V[nv] = v; nv++; }
  }
  if (nv < 4) {
    // Degenerate (coincident shapes): use center delta.
    normal = length(d0) > 1e-12f ? d0 / length(d0) : float3(1,0,0);
    wpa = pa; wpb = pb;
    return length(d0) > 1e-12f ? 0.0f : 0.0f;
  }
  // Polytope faces as index triples (outward orientation fixed below).
  int F[CX_EPA_MAXFACES * 3];
  int nf = 0;
  int seed[4][3] = {{0,1,2},{0,3,1},{0,2,3},{1,3,2}};
  for (int i = 0; i < 4; ++i) {
    F[nf*3+0] = seed[i][0]; F[nf*3+1] = seed[i][1]; F[nf*3+2] = seed[i][2];
    // Flip to outward: origin must be on the negative side.
    float3 a = V[F[nf*3]].m, b = V[F[nf*3+1]].m, c = V[F[nf*3+2]].m;
    float3 nrm = cross(b - a, c - a);
    if (dot(nrm, a) > 0.0f) {
      int t = F[nf*3+1]; F[nf*3+1] = F[nf*3+2]; F[nf*3+2] = t;
    }
    nf++;
  }
  float best = 0.0f;
  int bi = 0;
  float3 bn = float3(1,0,0);
  for (int iter = 0; iter < CX_EPA_MAXITER; ++iter) {
    // Find face closest to origin.
    best = 1e30f; bi = -1;
    for (int i = 0; i < nf; ++i) {
      float3 a = V[F[i*3]].m, b = V[F[i*3+1]].m, c = V[F[i*3+2]].m;
      float3 nrm = cross(b - a, c - a);
      float l = length(nrm);
      if (l < 1e-24f) continue;
      nrm = nrm / l;
      float dist = dot(nrm, a);
      if (dist < best) { best = dist; bi = i; bn = nrm; }
    }
    if (bi < 0) break;
    CxVertex v = cx_minkowski(ta, pa, Ra, sza, tb, pb, Rb, szb, bn);
    float adv = dot(v.m, bn) - best;
    if (adv < 1e-6f * max(1.0f, best) || nv >= 8 - 1) {
      // Converged: barycentric witness on best face.
      float u, w, x;
      float3 a = V[F[bi*3]].m, b = V[F[bi*3+1]].m, c = V[F[bi*3+2]].m;
      cx_tri_closest(a, b, c, &u, &w, &x);
      wpa = V[F[bi*3]].a * u + V[F[bi*3+1]].a * w + V[F[bi*3+2]].a * x;
      wpb = V[F[bi*3]].b * u + V[F[bi*3+1]].b * w + V[F[bi*3+2]].b * x;
      normal = bn;
      return best;
    }
    // Expand: remove visible faces, stitch horizon.
    V[nv] = v;
    int vnew = nv; nv++;
    int kept[CX_EPA_MAXFACES * 3];
    int nk = 0;
    int horizon[64];
    int nh = 0;
    for (int i = 0; i < nf; ++i) {
      float3 a = V[F[i*3]].m;
      // Face visible if new vertex is strictly outside it.
      float3 b = V[F[i*3+1]].m, c = V[F[i*3+2]].m;
      float3 nrm = cross(b - a, c - a);
      float l = length(nrm);
      bool vis = false;
      if (l > 1e-24f) {
        nrm = nrm / l;
        vis = dot(v.m - a, nrm) > 1e-9f;
      }
      if (!vis) {
        kept[nk*3] = F[i*3]; kept[nk*3+1] = F[i*3+1]; kept[nk*3+2] = F[i*3+2];
        nk++;
      } else {
        // Boundary edges: add all three, duplicates cancel below.
        int e[3][2] = {{F[i*3],F[i*3+1]},{F[i*3+1],F[i*3+2]},{F[i*3+2],F[i*3]}};
        for (int e2 = 0; e2 < 3 && nh < 62; ++e2) {
          horizon[nh++] = e[e2][0]; horizon[nh++] = e[e2][1];
        }
      }
    }
    // Cancel shared (opposite) edges; keep boundary loop edges.
    bool used[64];
    for (int i = 0; i < nh; ++i) used[i] = false;
    for (int i = 0; i < nf && nk * 3 + 3 <= CX_EPA_MAXFACES * 3; ++i) {
      (void)i;
      break;
    }
    for (int i = 0; i < nh; i += 2) {
      if (used[i]) continue;
      bool shared = false;
      for (int j = 0; j < nh; j += 2) {
        if (i == j || used[j]) continue;
        if (horizon[i] == horizon[j+1] && horizon[i+1] == horizon[j]) {
          used[j] = true; shared = true; break;
        }
      }
      if (!shared && nk * 3 + 3 <= CX_EPA_MAXFACES * 3) {
        kept[nk*3] = horizon[i]; kept[nk*3+1] = horizon[i+1]; kept[nk*3+2] = vnew;
        nk++;
      }
    }
    nf = 0;
    for (int i = 0; i < nk && i < CX_EPA_MAXFACES; ++i) {
      // Orient outward: new vertex must be on positive side.
      float3 a = V[kept[i*3]].m, b = V[kept[i*3+1]].m, c = V[kept[i*3+2]].m;
      float3 nrm = cross(b - a, c - a);
      if (dot(nrm, V[vnew].m - a) < 0.0f) {
        int t = kept[i*3+1]; kept[i*3+1] = kept[i*3+2]; kept[i*3+2] = t;
      }
      F[nf*3] = kept[i*3]; F[nf*3+1] = kept[i*3+1]; F[nf*3+2] = kept[i*3+2];
      nf++;
    }
    if (nf == 0) break;
  }
  // Fallback: best face found.
  if (bi >= 0 && bi < nf) {
    float u, w, x;
    float3 a = V[F[bi*3]].m, b = V[F[bi*3+1]].m, c = V[F[bi*3+2]].m;
    cx_tri_closest(a, b, c, &u, &w, &x);
    wpa = V[F[bi*3]].a * u + V[F[bi*3+1]].a * w + V[F[bi*3+2]].a * x;
    wpb = V[F[bi*3]].b * u + V[F[bi*3+1]].b * w + V[F[bi*3+2]].b * x;
    normal = bn;
    return max(best, 0.0f);
  }
  normal = length(d0) > 1e-12f ? d0 / length(d0) : float3(1,0,0);
  wpa = pa; wpb = pb;
  return 0.0f;
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
  bool ok = true;
  float gap = cx_gjk(ta, pa, Ra, sza, tb, pb, Rb, szb, d0, wpa, wpb, ok);
  if (ok) {
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
  float depth = cx_epa(ta, pa, Ra, sza, tb, pb, Rb, szb, nrm, ea, eb);
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
