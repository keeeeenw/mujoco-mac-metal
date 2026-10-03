// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Small unique helpers for spatial sensor kernels.

inline float3 sp_r3(device const float* a, uint i) {
  return float3(a[i],a[i+1],a[i+2]);
}
inline float4 sp_r4(device const float* a, uint i) {
  return float4(a[i],a[i+1],a[i+2],a[i+3]);
}
inline float4 sp_qconj(float4 q) { return float4(q.x,-q.yzw); }
inline float4 sp_qunit(float4 q) {
  float n=length(q);
  return n>1e-30f ? q/n : float4(1,0,0,0);
}
inline float3 sp_qrot(float4 q, float3 v) {
  return v+2.0f*cross(q.yzw,cross(q.yzw,v)+q.x*v);
}
// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Milestone 016 spatial queries: CONTACT sensor matching, site rangefinder
// rays and geom-distance witnesses. Built concatenated after the collision
// narrow-phase sources so geomdist reuses collide_pair directly. Ray
// primitives mirror engine_ray.c; the hfield march mirrors mj_rayHfield.

inline void rg_map(float3 pos, float3x3 mat, float3 pnt, float3 vec,
    thread float3& lpnt, thread float3& lvec) {
  lpnt = transpose(mat) * (pnt - pos);
  lvec = transpose(mat) * vec;
}

inline float rg_quad(float a, float b, float c, thread float2& xx) {
  float disc = b*b - a*c;
  if (disc < 0.0f || abs(a) < 1e-30f) return -1.0f;
  float root = sqrt(disc);
  xx = float2((-b - root)/a, (-b + root)/a);
  if (xx.x >= 0.0f) return xx.x;
  if (xx.y >= 0.0f) return xx.y;
  return -1.0f;
}

// Ray vs analytic geom; returns distance or -1. Normal kept global-frame.
inline float rg_analytic(int type, float3 pos, float3x3 mat, float3 size,
    float3 pnt, float3 vec, thread float3& normal) {
  normal = float3(0.0f);
  float3 lpnt, lvec;
  rg_map(pos, mat, pnt, vec, lpnt, lvec);
  if (type == 0) {
    // Plane: front face only, rendered rectangle.
    if (lvec.z > -1e-12f) return -1.0f;
    float x = -lpnt.z/lvec.z;
    if (x < 0.0f) return -1.0f;
    float p0 = lpnt.x + x*lvec.x, p1 = lpnt.y + x*lvec.y;
    if ((size.x <= 0.0f || abs(p0) <= size.x) &&
        (size.y <= 0.0f || abs(p1) <= size.y)) {
      normal = mat[2];
      return x;
    }
    return -1.0f;
  }
  if (type == 2) {
    float3 dif = pnt - pos;
    float a = dot(vec, vec), b = dot(vec, dif);
    float c = dot(dif, dif) - size.x*size.x;
    float2 xx;
    float x = rg_quad(a, b, c, xx);
    if (x >= 0.0f) {
      normal = normalize(pnt + vec*x - pos);
    }
    return x;
  }
  if (type == 3) {
    float ssz = size.x + size.y;
    float3 dif = pnt - pos;
    float2 xx0;
    if (rg_quad(dot(vec,vec), dot(vec,dif),
                dot(dif,dif)-ssz*ssz, xx0) < 0.0f) return -1.0f;
    float x = -1.0f;
    int tp = 0;
    float a = lvec.x*lvec.x + lvec.y*lvec.y;
    float b = lvec.x*lpnt.x + lvec.y*lpnt.y;
    float c = lpnt.x*lpnt.x + lpnt.y*lpnt.y - size.x*size.x;
    float2 xx;
    float sol = rg_quad(a, b, c, xx);
    if (sol >= 0.0f && abs(lpnt.z+sol*lvec.z) <= size.y) { x = sol; tp = 0; }
    float avec = dot(lvec, lvec);
    for (int side = -1; side <= 1; side += 2) {
      float3 ld = lpnt - float3(0.0f, 0.0f, float(side)*size.y);
      float bb = dot(lvec, ld);
      float cc = dot(ld, ld) - size.x*size.x;
      float2 rts = float2(-1.0f);
      rg_quad(avec, bb, cc, rts);
      for (int k = 0; k < 2; ++k) {
        float cand = k == 0 ? rts.x : rts.y;
        float z = lpnt.z + cand*lvec.z;
        if (cand >= 0.0f && ((side > 0 && z >= size.y) || (side < 0 && z <= -size.y))) {
          if (x < 0.0f || cand < x) { x = cand; tp = side; }
        }
      }
    }
    if (x >= 0.0f) {
      float3 ln;
      if (tp == 0) ln = float3(lpnt.x+lvec.x*x, lpnt.y+lvec.y*x, 0.0f);
      else ln = float3(lpnt.x+lvec.x*x, lpnt.y+lvec.y*x, lpnt.z+lvec.z*x-float(tp)*size.y);
      normal = normalize(mat * normalize(ln));
    }
    return x;
  }
  if (type == 4) {
    float3 s = float3(1.0f/(size.x*size.x), 1.0f/(size.y*size.y), 1.0f/(size.z*size.z));
    float a = dot(s*lvec, lvec);
    float b = dot(s*lvec, lpnt);
    float c = dot(s*lpnt, lpnt) - 1.0f;
    float2 xx;
    float x = rg_quad(a, b, c, xx);
    if (x >= 0.0f) {
      float3 l = lpnt + lvec*x;
      normal = normalize(mat * normalize(s*l));
    }
    return x;
  }
  if (type == 5) {
    float ssz = size.x*size.x + size.y*size.y;
    float3 dif = pnt - pos;
    float2 xx0;
    if (rg_quad(dot(vec,vec), dot(vec,dif), dot(dif,dif)-ssz, xx0) < 0.0f) return -1.0f;
    float x = -1.0f;
    int tp = 0;
    if (abs(lvec.z) > 1e-12f) {
      for (int side = -1; side <= 1; side += 2) {
        float sol = (float(side)*size.y-lpnt.z)/lvec.z;
        if (sol >= 0.0f) {
          float p0 = lpnt.x+sol*lvec.x, p1 = lpnt.y+sol*lvec.y;
          if (p0*p0+p1*p1 <= size.x*size.x && (x < 0.0f || sol < x)) {
            x = sol; tp = side;
          }
        }
      }
    }
    float a = lvec.x*lvec.x + lvec.y*lvec.y;
    float b = lvec.x*lpnt.x + lvec.y*lpnt.y;
    float c = lpnt.x*lpnt.x + lpnt.y*lpnt.y - size.x*size.x;
    float2 xx;
    float sol = rg_quad(a, b, c, xx);
    if (sol >= 0.0f && abs(lpnt.z+sol*lvec.z) <= size.y && (x < 0.0f || sol < x)) {
      x = sol; tp = 0;
    }
    if (x >= 0.0f) {
      float3 ln;
      if (tp == 0) ln = normalize(float3(lpnt.x+lvec.x*x, lpnt.y+lvec.y*x, 0.0f));
      else ln = float3(0.0f, 0.0f, float(tp));
      normal = normalize(mat * ln);
    }
    return x;
  }
  if (type == 6) {
    float ssz = dot(size, size);
    float3 dif = pnt - pos;
    float2 xx0;
    if (rg_quad(dot(vec,vec), dot(vec,dif), dot(dif,dif)-ssz, xx0) < 0.0f) return -1.0f;
    float x = -1.0f;
    int ax = -1, sd = 0;
    for (int k = 0; k < 3; ++k) {
      float lk = k==0 ? lvec.x : (k==1 ? lvec.y : lvec.z);
      float pk = k==0 ? lpnt.x : (k==1 ? lpnt.y : lpnt.z);
      float sk = k==0 ? size.x : (k==1 ? size.y : size.z);
      if (abs(lk) <= 1e-12f) continue;
      for (int side = -1; side <= 1; side += 2) {
        float sol = (float(side)*sk-pk)/lk;
        if (sol < 0.0f) continue;
        float q0 = (k==0 ? lpnt.y : lpnt.x) + sol*(k==0 ? lvec.y : lvec.x);
        float q1 = (k==2 ? lpnt.y : lpnt.z) + sol*(k==2 ? lvec.y : lvec.z);
        float b0 = k==0 ? size.y : size.x;
        float b1 = k==2 ? size.y : size.z;
        if (abs(q0) <= b0 && abs(q1) <= b1 && (x < 0.0f || sol < x)) {
          x = sol; ax = k; sd = side;
        }
      }
    }
    if (x >= 0.0f) {
      float3 ln = float3(0.0f);
      if (ax==0) ln.x = float(sd); else if (ax==1) ln.y = float(sd); else ln.z = float(sd);
      normal = normalize(mat * ln);
    }
    return x;
  }
  return -1.0f;
}

// Ray vs triangle (pinned ray_triangle, hit distance; normal optional).
inline float rg_triangle(float3 v0, float3 v1, float3 v2,
    float3 lpnt, float3 lvec, float3 b0, float3 b1, thread float3& nrm) {
  nrm = float3(0.0f);
  float3 d0 = v0-lpnt, d1 = v1-lpnt, d2 = v2-lpnt;
  float2 p0 = float2(dot(b0,d0),dot(b1,d0));
  float2 p1 = float2(dot(b0,d1),dot(b1,d1));
  float2 p2 = float2(dot(b0,d2),dot(b1,d2));
  if ((p0.x>0.0f && p1.x>0.0f && p2.x>0.0f) ||
      (p0.x<0.0f && p1.x<0.0f && p2.x<0.0f) ||
      (p0.y>0.0f && p1.y>0.0f && p2.y>0.0f) ||
      (p0.y<0.0f && p1.y<0.0f && p2.y<0.0f)) return -1.0f;
  float2 A0 = float2(p0.x-p2.x, p1.x-p2.x);
  float2 A1 = float2(p0.y-p2.y, p1.y-p2.y);
  float det = A0.x*A1.y - A0.y*A1.x;
  if (abs(det) < 1e-12f) return -1.0f;
  float t0 = (A1.y*(-p2.x)-A0.y*(-p2.y))/det;
  float t1 = (-A1.x*(-p2.x)+A0.x*(-p2.y))/det;
  if (t0<0.0f || t1<0.0f || t0+t1>1.0f) return -1.0f;
  float3 e0=v0-v2, e1=v1-v2, dd=lpnt-v2;
  float3 n=cross(e0,e1);
  float denom=dot(lvec,n);
  if (abs(denom) < 1e-12f) return -1.0f;
  float x=-dot(dd,n)/denom;
  if (x>=0.0f) nrm=normalize(n);
  return x>=0.0f ? x : -1.0f;
}

// Ray vs mesh hull via face triangles (pinned mj_rayMesh triangle set).
inline float rg_mesh(float3 gpos, float3x3 gmat, float3 gsize,
    float3 pnt, float3 vec, device const float* hull, device const int* info,
    int g, thread float3& normal) {
  normal = float3(0.0f);
  float3 dummy_n;
  if (rg_analytic(6, gpos, gmat, gsize, pnt, vec, dummy_n) < 0.0f) return -1.0f;
  int voff = info[9*g+0], vnum = info[9*g+1];
  int ioff = info[9*g+3]*3, fnum = info[9*g+4];
  float x = -1.0f;
  for (int f = 0; f < fnum; ++f) {
    int i0 = int(hull[ioff+f*3+0]), i1 = int(hull[ioff+f*3+1]), i2 = int(hull[ioff+f*3+2]);
    if (i0<0||i1<0||i2<0||i0>=vnum||i1>=vnum||i2>=vnum) continue;
    float3 v0 = gpos + gmat*float3(hull[3*(voff+i0)],hull[3*(voff+i0)+1],hull[3*(voff+i0)+2]);
    float3 v1 = gpos + gmat*float3(hull[3*(voff+i1)],hull[3*(voff+i1)+1],hull[3*(voff+i1)+2]);
    float3 v2 = gpos + gmat*float3(hull[3*(voff+i2)],hull[3*(voff+i2)+1],hull[3*(voff+i2)+2]);
    // Pinned triangle basis (b0,b1) for the inside test.
    float3 e0 = v1-v0, e1 = v2-v0;
    float3 b1v = normalize(cross(e0,e1));
    float3 b0v = normalize(cross(b1v,e0));
    float3 nrm;
    float sol = rg_triangle(v0,v1,v2,pnt,vec,b0v,b1v,nrm);
    if (sol >= 0.0f && (x < 0.0f || sol < x)) { x = sol; normal = nrm; }
  }
  return x;
}

// Box face solutions (pinned ray_box `all` outputs).
inline void rg_box_all(float3 pos, float3x3 mat, float3 size,
    float3 pnt, float3 vec, thread float all_[6]) {
  for (int k=0;k<6;++k) all_[k]=-1.0f;
  float3 lpnt = transpose(mat)*(pnt-pos);
  float3 lvec = transpose(mat)*vec;
  for (int k=0;k<3;++k) {
    float lk = k==0?lvec.x:(k==1?lvec.y:lvec.z);
    if (abs(lk) <= 1e-12f) continue;
    for (int side=-1;side<=1;side+=2) {
      float pk = k==0?lpnt.x:(k==1?lpnt.y:lpnt.z);
      float sk = k==0?size.x:(k==1?size.y:size.z);
      float sol=(float(side)*sk-pk)/lk;
      if (sol<0.0f) continue;
      float qa = (k==0?lpnt.y:lpnt.x)+sol*(k==0?lvec.y:lvec.x);
      float qb2 = (k==2?lpnt.y:lpnt.z)+sol*(k==2?lvec.y:lvec.z);
      float ba = k==0?size.y:size.x;
      float bb2 = k==2?size.y:size.z;
      if (abs(qa)<=ba && abs(qb2)<=bb2) {
        float cur = all_[2*k+(side+1)/2];
        if (cur<0.0f || sol<cur) all_[2*k+(side+1)/2]=sol;
      }
    }
  }
}

// Heightfield ray march (pinned mj_rayHfield).
inline float rg_hfield(float3 gpos, float3x3 gmat,
    device const float* hull, device const int* info, int g,
    float3 pnt, float3 vec, thread float3& normal) {
  normal = float3(0.0f);
  int dcursor = info[9*g+5], nrow = info[9*g+6], ncol = info[9*g+7];
  int scursor = info[9*g+8];
  if (dcursor<0 || nrow<2 || ncol<2) return -1.0f;
  float sx = hull[scursor], sy = hull[scursor+1];
  float sz2 = hull[scursor+2], sz3 = hull[scursor+3];
  float3 xcol0 = gmat[0], xcol1 = gmat[1], xcol2 = gmat[2];
  float3 base_pos = gpos - xcol2*(sz3*0.5f);
  float3 bsz = float3(sx, sy, sz3*0.5f);
  float3 nbase(0.0f);
  float x = rg_analytic(6, base_pos, gmat, bsz, pnt, vec, nbase);
  float all_[6];
  // Top-box call needs `all`; recompute via rg_box_all in world frame is
  // wrong (boxes are oriented); do it in local frame instead.
  float3 lpnt = transpose(gmat)*(pnt-gpos);
  float3 lvec = transpose(gmat)*vec;
  // Local base/top boxes (axis aligned).
  float lb_all[6], lt_all[6];
  // base box center (0,0,-sz3/2), half (sx,sy,sz3/2); recompute inline:
  {
    float3 bp = float3(0.0f,0.0f,-sz3*0.5f);
    float3 bs = float3(sx,sy,sz3*0.5f);
    for (int k=0;k<6;++k) lb_all[k]=-1.0f;
    for (int k=0;k<3;++k) {
      float lk = k==0?lvec.x:(k==1?lvec.y:lvec.z);
      if (abs(lk) <= 1e-12f) continue;
      for (int side=-1;side<=1;side+=2) {
        float pk = k==0?lpnt.x:(k==1?lpnt.y:lpnt.z);
        float ck = k==0?bp.x:(k==1?bp.y:bp.z);
        float sk = k==0?bs.x:(k==1?bs.y:bs.z);
        float sol=(float(side)*sk+ck-pk)/lk;
        if (sol<0.0f) continue;
        float qa = (k==0?lpnt.y:lpnt.x)+sol*(k==0?lvec.y:lvec.x);
        float qb2 = (k==2?lpnt.y:lpnt.z)+sol*(k==2?lvec.y:lvec.z);
        float ca = k==0?bp.y:bp.x, cb2 = k==2?bp.y:bp.z;
        float ba = k==0?bs.y:bs.x, bb2 = k==2?bs.y:bs.z;
        if (abs(qa-ca)<=ba && abs(qb2-cb2)<=bb2) {
          float cur = lb_all[2*k+(side+1)/2];
          if (cur<0.0f || sol<cur) lb_all[2*k+(side+1)/2]=sol;
        }
      }
    }
  }
  float top_hit = -1.0f;
  {
    float3 bp = float3(0.0f,0.0f,0.0f);
    float3 bs = float3(sx,sy,sz2*0.5f);
    for (int k=0;k<6;++k) lt_all[k]=-1.0f;
    for (int k=0;k<3;++k) {
      float lk = k==0?lvec.x:(k==1?lvec.y:lvec.z);
      if (abs(lk) <= 1e-12f) continue;
      for (int side=-1;side<=1;side+=2) {
        float pk = k==0?lpnt.x:(k==1?lpnt.y:lpnt.z);
        float ck = k==0?bp.x:(k==1?bp.y:bp.z);
        float sk = k==0?bs.x:(k==1?bs.y:bs.z);
        float sol=(float(side)*sk+ck-pk)/lk;
        if (sol<0.0f) continue;
        float qa = (k==0?lpnt.y:lpnt.x)+sol*(k==0?lvec.y:lvec.x);
        float qb2 = (k==2?lpnt.y:lpnt.z)+sol*(k==2?lvec.y:lvec.z);
        float ca = k==0?bp.y:bp.x, cb2 = k==2?bp.y:bp.z;
        float ba = k==0?bs.y:bs.x, bb2 = k==2?bs.y:bs.z;
        if (abs(qa-ca)<=ba && abs(qb2-cb2)<=bb2) {
          float cur = lt_all[2*k+(side+1)/2];
          if (cur<0.0f || sol<cur) { lt_all[2*k+(side+1)/2]=sol; }
          if (top_hit<0.0f || sol<top_hit) top_hit=sol;
        }
      }
    }
  }
  if (top_hit < 0.0f) {
    if (x >= 0.0f) normal = nbase;
    return x;
  }
  // Normal-plane basis.
  float3 b0 = float3(1.0f,1.0f,1.0f), b1;
  if (abs(lvec.x)>=abs(lvec.y) && abs(lvec.x)>=abs(lvec.z)) b0.x=0.0f;
  else if (abs(lvec.y)>=abs(lvec.z)) b0.y=0.0f;
  else b0.z=0.0f;
  b1 = normalize(b0-dot(b0,lvec)/max(dot(lvec,lvec),1e-30f)*lvec);
  b0 = normalize(cross(b1,lvec));
  float seg0 = 0.0f, seg1 = top_hit;
  for (int k=0;k<6;++k) {
    if (lt_all[k] > seg1) { seg0 = top_hit; seg1 = lt_all[k]; }
  }
  float dx = (2.0f*sx)/float(ncol-1);
  float dy = (2.0f*sy)/float(nrow-1);
  float SX0=(lpnt.x+seg0*lvec.x+sx)/dx, SX1=(lpnt.x+seg1*lvec.x+sx)/dx;
  float SY0=(lpnt.y+seg0*lvec.y+sy)/dy, SY1=(lpnt.y+seg1*lvec.y+sy)/dy;
  int cmin=max(0,int(floor(min(SX0,SX1)))-1);
  int cmax=min(ncol-1,int(ceil(max(SX0,SX1)))+1);
  int rmin=max(0,int(floor(min(SY0,SY1)))-1);
  int rmax=min(nrow-1,int(ceil(max(SY0,SY1)))+1);
  float3 nlocal=float3(0.0f);
  if (x>=0.0f) nlocal=transpose(gmat)*nbase;
  for (int r=rmin;r<rmax;++r) {
    for (int c=cmin;c<cmax;++c) {
      float3 va[3] = {
        float3(dx*float(c)-sx, dy*float(r)-sy, hull[dcursor+r*ncol+c]*sz2),
        float3(dx*float(c+1)-sx, dy*float(r)-sy, hull[dcursor+r*ncol+c+1]*sz2),
        float3(dx*float(c+1)-sx, dy*float(r+1)-sy, hull[dcursor+(r+1)*ncol+c+1]*sz2)};
      float3 ntri;
      float sol=rg_triangle(va[0],va[1],va[2],lpnt,lvec,b0,b1,ntri);
      if (sol>=0.0f && (x<0.0f || sol<x)) { x=sol; nlocal=ntri; }
      float3 vb[3] = {
        float3(dx*float(c)-sx, dy*float(r)-sy, hull[dcursor+r*ncol+c]*sz2),
        float3(dx*float(c+1)-sx, dy*float(r+1)-sy, hull[dcursor+(r+1)*ncol+c+1]*sz2),
        float3(dx*float(c)-sx, dy*float(r+1)-sy, hull[dcursor+(r+1)*ncol+c]*sz2)};
      sol=rg_triangle(vb[0],vb[1],vb[2],lpnt,lvec,b0,b1,ntri);
      if (sol>=0.0f && (x<0.0f || sol<x)) { x=sol; nlocal=ntri; }
    }
  }
  // Viable top-box sides.
  for (int k=0;k<4;++k) {
    float ai=lt_all[k];
    if (ai>=0.0f && (ai<x || x<0.0f)) {
      float z=(lpnt.z+ai*lvec.z)/sz2;
      float y, y0, z0, z1;
      if (k<2) {
        y=(lpnt.y+ai*lvec.y+sy)/dy;
        y0=max(0.0f,min(float(nrow-2),floor(y)));
        int yy=int(y0+0.5f);
        z0=hull[dcursor+yy*ncol+(k==1?ncol-1:0)];
        z1=hull[dcursor+(yy+1)*ncol+(k==1?ncol-1:0)];
      } else {
        y=(lpnt.x+ai*lvec.x+sx)/dx;
        y0=max(0.0f,min(float(ncol-2),floor(y)));
        int xx=int(y0+0.5f);
        z0=hull[dcursor+(k==3?(nrow-1)*ncol:0)+xx];
        z1=hull[dcursor+(k==3?(nrow-1)*ncol:0)+xx+1];
      }
      if (z < z0*(y0+1.0f-y)+z1*(y-y0)) {
        x=ai;
        nlocal=float3(0.0f);
        if (k==0) nlocal.x=-1.0f;
        else if (k==1) nlocal.x=1.0f;
        else if (k==2) nlocal.y=-1.0f;
        else nlocal.y=1.0f;
      }
    }
  }
  if (x>=0.0f) normal=gmat*nlocal;
  return x;
}

// Contact-frame 6D wrench from native slot storage (pinned mj_contactForce).
inline void sp_contact_wrench(int cdim, int cone,
    device const float* force, uint foff, device const float* mu, uint muoff,
    thread float* out6) {
  for (int k=0;k<6;++k) out6[k]=0.0f;
  if (cone==0) {
    float fn=0.0f;
    int ne2=2*(cdim-1);
    for (int k=0;k<ne2;++k) fn+=force[foff+1+k];
    out6[0]=fn;
    for (int k=0;k<cdim-1;++k)
      out6[1+k]=(force[foff+1+2*k]-force[foff+2+2*k])*mu[muoff+k];
  } else {
    for (int k=0;k<cdim && k<6;++k) out6[k]=force[foff+k];
  }
}

// CONTACT sensor matching (pinned matchContact/checkMatch + copySensorData,
// single compiler slot, all four reductions).
inline bool sp_check_match(int body, int geom, int type, int id,
    device const int* body_tree) {
  if (type==0) return true;
  if (type==6) return true;
  if (type==5) return id==geom;
  if (type==1) return id==body;
  if (type==2) {
    int b=body;
    while (b>id) b=body_tree[b*2+0];
    return b==id;
  }
  return false;
}

kernel void evaluate_contact_sensors(
    device const float* contact_frame [[buffer(0)]],
    device const float* contact_force [[buffer(1)]],
    device const float* contact_row_data [[buffer(2)]],
    device const int* contact_packed [[buffer(3)]],
    device const float* contact_mu [[buffer(4)]],
    device const int* slot_pair [[buffer(5)]],
    device const int* pair_live [[buffer(6)]],
    device const int* pair_geoms [[buffer(7)]],
    device const int* geom_bodyid [[buffer(8)]],
    device const int* site_bodyid [[buffer(9)]],
    device const int* body_tree [[buffer(10)]],
    device const float* site_pos [[buffer(11)]],
    device const float* site_quat [[buffer(12)]],
    device const float* site_geom [[buffer(13)]],
    device const int* meta [[buffer(14)]],
    device const int* intprm [[buffer(15)]],
    device float* output [[buffer(16)]],
    constant int* stage_mask [[buffer(17)]],
    constant int* dims [[buffer(18)]],
    uint index [[thread_position_in_grid]]) {
  // meta (10 ints): type, datatype, needstage, objtype, objid, dim, adr,
  //   cutoff_bits, reftype, refid. intprm (3 ints): dataspec, reduce, x.
  // dims (7 ints): batch, nsensor, ndata, nc, nsite, nbody, disable.
  uint batch=uint(dims[0]), nsensor=uint(dims[1]), ndata=uint(dims[2]);
  uint nc=uint(dims[3]), nsite=uint(dims[4]), nbody=uint(dims[5]);
  uint ACC=3;
  if (index>=batch*nsensor) return;
  if ((uint(stage_mask[0])&(1u<<ACC))==0) return;
  if (dims[6] & 8192) return;
  uint world=index/nsensor, i=index-world*nsensor;
  if (uint(meta[i*10+2])!=ACC) return;
  int styp=meta[i*10+0];
  if (styp!=42 && styp!=0) return;
  uint dim=uint(meta[i*10+5]), adr=uint(meta[i*10+6]);
  uint base=world*ndata+adr;
  for (uint j=0;j<dim;j++) output[base+j]=0.0f;
  if (styp==0) {
    // TOUCH: sum of normal forces in the site zone (pinned ray test).
    uint s=uint(meta[i*10+4]);
    int sb=site_bodyid[s];
    float3 sp=sp_r3(site_pos,(world*nsite+s)*3);
    float4 sq=sp_qunit(sp_r4(site_quat,(world*nsite+s)*4));
    float3 c0=sp_qrot(sq,float3(1,0,0)), c1=sp_qrot(sq,float3(0,1,0)), c2=sp_qrot(sq,float3(0,0,1));
    float3x3 sm=float3x3(c0,c1,c2);
    float3 ssize=sp_r3(site_geom,s*4);
    int stype=int(site_geom[s*4+3]);
    float total=0.0f;
    for (uint c=0;c<nc;++c) {
      if (contact_row_data[(world*nc+c)*36]<=0.5f) continue;
      if (pair_live[slot_pair[c]]==0) continue;
      uint pair=uint(slot_pair[c]);
      int g1=pair_geoms[pair*2+0], g2=pair_geoms[pair*2+1];
      int b1=g1>=0 ? geom_bodyid[g1] : -1;
      int b2=g2>=0 ? geom_bodyid[g2] : -1;
      if (sb!=b1 && sb!=b2) continue;
      float fn=contact_force[(world*nc+c)*11];
      if (fn<=0.0f) continue;
      float3 n=sp_r3(contact_frame,(world*nc+c)*12);
      float3 cp=sp_r3(contact_frame,(world*nc+c)*12+9);
      float3 ray=n*fn;
      float rl=length(ray);
      if (rl<=0.0f) continue;
      ray/=rl;
      if (sb==b2) ray=-ray;
      float3 dn=float3(0.0f);
      if (rg_analytic(stype,sp,sm,ssize,cp,ray,dn)>=0.0f) total+=fn;
    }
    output[base]=total;
    return;
  }
  int t1=meta[i*10+3], id1=meta[i*10+4];
  int t2=meta[i*10+8], id2=meta[i*10+9];
  int dataspec=intprm[i*3+0], reduce=intprm[i*3+1];
  int mid[24];
  int mflip[24];
  float mcrit[24];
  int nmatch=0;
  for (uint s=0;s<nc && nmatch<24;++s) {
    if (contact_row_data[(world*nc+s)*36]<=0.5f) continue;
    if (pair_live[slot_pair[s]]==0) continue;
    uint pair=uint(slot_pair[s]);
    int g1=pair_geoms[pair*2+0], g2=pair_geoms[pair*2+1];
    int b1=g1>=0 ? geom_bodyid[g1] : -1;
    int b2=g2>=0 ? geom_bodyid[g2] : -1;
    int m=0;
    if (t1==0 && t2==0) m=1;
    else {
      if (t1==6) {
        float3 sp=sp_r3(site_pos,(world*nsite+uint(id1))*3);
        float4 sq=sp_qunit(sp_r4(site_quat,(world*nsite+uint(id1))*4));
        float3 cp=sp_r3(contact_frame,(world*nc+s)*12+9);
        float st=site_geom[uint(id1)*4+3];
        float3 vec=cp-sp;
        float3 pl=sp_qrot(sp_qconj(sq),vec);
        float sx=site_geom[uint(id1)*4+0], sy=site_geom[uint(id1)*4+1], sz=site_geom[uint(id1)*4+2];
        bool inside=false;
        if (st==2.0f) inside=dot(vec,vec)<sx*sx;
        else if (st==3.0f) {
          float zc=clamp(pl.z,-sy,sy);
          inside=pl.x*pl.x+pl.y*pl.y+(pl.z-zc)*(pl.z-zc)<sx*sx;
        }
        else if (st==4.0f) inside=(pl.x*pl.x/(sx*sx)+pl.y*pl.y/(sy*sy)+pl.z*pl.z/(sz*sz))<1.0f;
        else if (st==5.0f) inside=(abs(pl.z)<sy && pl.x*pl.x+pl.y*pl.y<sx*sx);
        else if (st==6.0f) inside=(abs(pl.x)<sx && abs(pl.y)<sy && abs(pl.z)<sz);
        else if (st==0.0f) inside=(pl.z<0.0f);
        if (!inside) continue;
      }
      bool m11=sp_check_match(b1,g1,t1,id1,body_tree);
      bool m12=sp_check_match(b2,g2,t1,id1,body_tree);
      bool m21=sp_check_match(b1,g1,t2,id2,body_tree);
      bool m22=sp_check_match(b2,g2,t2,id2,body_tree);
      if (!(m11||m12) || !(m21||m22)) continue;
      if (t1!=0 && t2!=0) {
        bool reg=m11&&m22, rev=m12&&m21;
        if (reg&&!rev) m=1;
        else if (rev&&!reg) m=-1;
        else if (reg&&rev) m=1;
        else continue;
      } else if (t1!=0) m=m11 ? 1 : -1;
      else if (t2!=0) m=m22 ? 1 : -1;
      else continue;
    }
    if (!m) continue;
    mid[nmatch]=int(s);
    mflip[nmatch]=m;
    mcrit[nmatch]=contact_row_data[(world*nc+s)*36+1];
    nmatch++;
  }
  for (int k=0;k<nmatch;++k) {
    uint s=uint(mid[k]);
    if (reduce==2) {
      int cdim=contact_packed[s*3+0], cone=contact_packed[s*3+2];
      float w6[6];
      sp_contact_wrench(cdim,cone,contact_force,(world*nc+s)*11,contact_mu,s*5,w6);
      mcrit[k]=-(w6[0]*w6[0]+w6[1]*w6[1]+w6[2]*w6[2]);
    }
  }
  // NOTE: dist criterion for MINDIST filled in the selection below.
  int pick=-1;
  if (reduce==3) {
    // Net-force aggregation over all matches.
    float3 F=float3(0.0f), T=float3(0.0f), P=float3(0.0f);
    float tot=0.0f;
    for (int k=0;k<nmatch;++k) {
      uint s=uint(mid[k]);
      int cdim=contact_packed[s*3+0], cone=contact_packed[s*3+2];
      float w6[6];
      sp_contact_wrench(cdim,cone,contact_force,(world*nc+s)*11,contact_mu,s*5,w6);
      float3 n=sp_r3(contact_frame,(world*nc+s)*12);
      float3 t1v=sp_r3(contact_frame,(world*nc+s)*12+3);
      float3 t2v=sp_r3(contact_frame,(world*nc+s)*12+6);
      float3 ppos=sp_r3(contact_frame,(world*nc+s)*12+9);
      float3 fj=n*w6[0]+t1v*w6[1]+t2v*w6[2];
      float3 tj=n*w6[3]+t1v*w6[4]+t2v*w6[5];
      if (mflip[k]<0) { fj=-fj; tj=-tj; }
      F+=fj; T+=tj;
      float wgt=length(fj);
      P+=wgt*ppos; tot+=wgt;
    }
    if (tot>0.0f) P/=tot;
    // Induced torques about the centroid (second pass for positions).
    for (int k=0;k<nmatch;++k) {
      uint s=uint(mid[k]);
      int cdim=contact_packed[s*3+0], cone=contact_packed[s*3+2];
      float w6[6];
      sp_contact_wrench(cdim,cone,contact_force,(world*nc+s)*11,contact_mu,s*5,w6);
      float3 n=sp_r3(contact_frame,(world*nc+s)*12);
      float3 t1v=sp_r3(contact_frame,(world*nc+s)*12+3);
      float3 t2v=sp_r3(contact_frame,(world*nc+s)*12+6);
      float3 ppos=sp_r3(contact_frame,(world*nc+s)*12+9);
      float3 fj=n*w6[0]+t1v*w6[1]+t2v*w6[2];
      if (mflip[k]<0) fj=-fj;
      T+=cross(ppos-P,fj);
    }
    uint o=0;
    if (dataspec&1) output[base+o++]=float(nmatch);
    if (dataspec&2) { output[base+o]=F.x; output[base+o+1]=F.y; output[base+o+2]=F.z; o+=3; }
    if (dataspec&4) { output[base+o]=T.x; output[base+o+1]=T.y; output[base+o+2]=T.z; o+=3; }
    if (dataspec&8) { output[base+o]=0.0f; o+=1; }
    if (dataspec&16) { output[base+o]=0.0f; output[base+o+1]=0.0f; output[base+o+2]=0.0f; o+=3; }
    if (dataspec&32) { output[base+o]=1.0f; output[base+o+1]=0.0f; output[base+o+2]=0.0f; o+=3; }
    if (dataspec&64) { output[base+o]=0.0f; output[base+o+1]=1.0f; output[base+o+2]=0.0f; o+=3; }
    return;
  }
  if (nmatch>0 && reduce==0) pick=0;
  else if (nmatch>0) {
    pick=0;
    float best=mcrit[0];
    for (int k=1;k<nmatch;++k) {
      if (mcrit[k]<best) { best=mcrit[k]; pick=k; }
    }
  }
  if (pick>=0) {
    uint s=uint(mid[pick]);
    int flip=mflip[pick];
    int cdim=contact_packed[s*3+0], cone=contact_packed[s*3+2];
    float w6[6];
    sp_contact_wrench(cdim,cone,contact_force,(world*nc+s)*11,contact_mu,s*5,w6);
    float3 n=sp_r3(contact_frame,(world*nc+s)*12);
    float3 t1v=sp_r3(contact_frame,(world*nc+s)*12+3);
    float3 pp=sp_r3(contact_frame,(world*nc+s)*12+9);
    float dd=contact_row_data[(world*nc+s)*36+1];
    uint o=0;
    if (dataspec&1) output[base+o++]=float(nmatch);
    if (dataspec&2) {
      float3 f=float3(w6[0],w6[1],w6[2]);
      // Contact-frame force (no world rotation, like pinned).
      if (flip<0) f.z*=-1.0f;
      output[base+o]=f.x; output[base+o+1]=f.y; output[base+o+2]=f.z; o+=3;
    }
    if (dataspec&4) {
      float3 t=float3(w6[3],w6[4],w6[5]);
      if (flip<0) t.z*=-1.0f;
      output[base+o]=t.x; output[base+o+1]=t.y; output[base+o+2]=t.z; o+=3;
    }
    if (dataspec&8) { output[base+o]=dd; o+=1; }
    if (dataspec&16) {
      float3 nn = flip<0 ? -n : n;
      output[base+o]=nn.x; output[base+o+1]=nn.y; output[base+o+2]=nn.z; o+=3;
    }
    if (dataspec&32) {
      float3 tt = flip<0 ? -t1v : t1v;
      output[base+o]=tt.x; output[base+o+1]=tt.y; output[base+o+2]=tt.z; o+=3;
    }
  }
}

// Site rangefinder rays (pinned site path: origin = site pos, dir = site
// z-axis, bodyexclude = site body, static included, geomgroup NULL).
kernel void evaluate_rays(
    device const float* site_pos [[buffer(0)]],
    device const float* site_quat [[buffer(1)]],
    device const float* geom_pos [[buffer(2)]],
    device const float* geom_quat [[buffer(3)]],
    device const int* geom_type [[buffer(4)]],
    device const float* geom_size [[buffer(5)]],
    device const int* geom_bodyid [[buffer(6)]],
    device const int* geom_matid [[buffer(7)]],
    device const float* geom_rgba [[buffer(8)]],
    device const float* mat_rgba [[buffer(9)]],
    device const float* hull [[buffer(10)]],
    device const int* hull_info [[buffer(11)]],
    device const int* site_bodyid [[buffer(12)]],
    device const int* meta [[buffer(13)]],
    device const int* intprm [[buffer(14)]],
    device float* output [[buffer(15)]],
    constant int* stage_mask [[buffer(16)]],
    constant int* dims [[buffer(17)]],
    uint index [[thread_position_in_grid]]) {
  // meta (10 ints): type, datatype, needstage, objtype=SITE, objid=site,
  //   dim, adr, cutoff_bits, reftype, refid. intprm (3 ints): dataspec...
  // dims (8 ints): batch, nsensor, ndata, ngeom, nsite, nmat, disable, x.
  uint batch=uint(dims[0]), nsensor=uint(dims[1]), ndata=uint(dims[2]);
  uint ngeom=uint(dims[3]), nsite=uint(dims[4]), nmat=uint(dims[5]);
  uint POS=1;
  if (index>=batch*nsensor) return;
  if ((uint(stage_mask[0])&(1u<<POS))==0) return;
  if (dims[6] & 8192) return;
  uint world=index/nsensor, i=index-world*nsensor;
  if (uint(meta[i*10+2])!=POS || meta[i*10+0]!=7) return;
  uint s=uint(meta[i*10+4]);
  uint dim=uint(meta[i*10+5]), adr=uint(meta[i*10+6]);
  int dataspec=intprm[i*3+0];
  uint base=world*ndata+adr;
  float3 origin=sp_r3(site_pos,(world*nsite+s)*3);
  float4 sq=sp_qunit(sp_r4(site_quat,(world*nsite+s)*4));
  float3 dir=sp_qrot(sq,float3(0,0,1));
  int bexcl=site_bodyid[s];
  float best=-1.0f;
  float3 bn=float3(0.0f);
  for (uint g=0;g<ngeom;++g) {
    if (geom_bodyid[g]==bexcl) continue;
    int mid=geom_matid[g];
    if (mid<0) {
      if (geom_rgba[g*4+3]==0.0f) continue;
    } else if (uint(mid)<nmat) {
      if (mat_rgba[mid*4+3]==0.0f) continue;
    }
    int t=geom_type[g];
    if (t==8) continue; // SDF rays are plugin-defined (019 gap).
    float3 gp=sp_r3(geom_pos,(world*ngeom+g)*3);
    float4 gq=sp_qunit(float4(geom_quat[(world*ngeom+g)*4],geom_quat[(world*ngeom+g)*4+1],geom_quat[(world*ngeom+g)*4+2],geom_quat[(world*ngeom+g)*4+3]));
    float3 c0=sp_qrot(gq,float3(1,0,0)), c1=sp_qrot(gq,float3(0,1,0)), c2=sp_qrot(gq,float3(0,0,1));
    float3x3 gm = float3x3(c0,c1,c2);
    float3 sz=sp_r3(geom_size,g*3);
    float3 hn=float3(0.0f);
    float dd=-1.0f;
    if (t==7) {
      dd=rg_mesh(gp,gm,sz,origin,dir,hull,hull_info,int(g),hn);
    } else if (t==1) {
      dd=rg_hfield(gp,gm,hull,hull_info,int(g),origin,dir,hn);
    } else {
      dd=rg_analytic(t,gp,gm,sz,origin,dir,hn);
    }
    if (dd>=0.0f && (best<0.0f || dd<best)) { best=dd; bn=hn; }
  }
  // Pinned fill_raydata (site sensor: cam_z NULL so depth = dist).
  bool hit=best>=0.0f;
  float3 point = hit ? origin+dir*best : float3(0.0f);
  uint o=0;
  if (dataspec&1) output[base+o++]=best;
  if (dataspec&2) {
    float3 v = hit ? dir : float3(0.0f);
    output[base+o]=v.x; output[base+o+1]=v.y; output[base+o+2]=v.z; o+=3;
  }
  if (dataspec&4) {
    output[base+o]=origin.x; output[base+o+1]=origin.y; output[base+o+2]=origin.z; o+=3;
  }
  if (dataspec&8) {
    output[base+o]=point.x; output[base+o+1]=point.y; output[base+o+2]=point.z; o+=3;
  }
  if (dataspec&16) {
    float3 v = hit ? bn : float3(0.0f);
    output[base+o]=v.x; output[base+o+1]=v.y; output[base+o+2]=v.z; o+=3;
  }
  if (dataspec&32) output[base+o++]= hit ? best : -1.0f;
  (void)dim;
}

// Geom-distance witnesses (pinned mj_geomDistance over the same narrow
// phase: canonical type-order flip, margin=cutoff, min over witnesses,
// fromto split about the witness point).
kernel void evaluate_geomdist(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const int* geom_type [[buffer(2)]],
    device const float* geom_size [[buffer(3)]],
    device const float* geom_rbound [[buffer(4)]],
    device const int* body_geoms [[buffer(5)]],
    device const float* hull [[buffer(6)]],
    device const int* hull_info [[buffer(7)]],
    device const int* meta [[buffer(8)]],
    device float* output [[buffer(9)]],
    constant int* stage_mask [[buffer(10)]],
    constant int* dims [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
  // meta (10 ints) as usual; cutoff from meta bits.
  // dims (9 ints): batch, nsensor, ndata, ngeom, nbody, disable,
  //   multiccd_off, sdf_maxn, x.
  uint batch=uint(dims[0]), nsensor=uint(dims[1]), ndata=uint(dims[2]);
  uint ngeom=uint(dims[3]);
  uint POS=1;
  if (index>=batch*nsensor) return;
  if ((uint(stage_mask[0])&(1u<<POS))==0) return;
  if (dims[5] & 8192) return;
  uint world=index/nsensor, i=index-world*nsensor;
  uint stage=uint(meta[i*10+2]);
  if (stage!=POS) return;
  int typ=meta[i*10+0];
  if (typ!=39 && typ!=40 && typ!=41) return;
  int otype=meta[i*10+3], oid=meta[i*10+4];
  int rtype=meta[i*10+8], rid=meta[i*10+9];
  uint dim=uint(meta[i*10+5]), adr=uint(meta[i*10+6]);
  float cutoff=as_type<float>(meta[i*10+7]);
  uint base=world*ndata+adr;
  float best=cutoff;
  float3 bpos=float3(0.0f), bnorm=float3(0.0f);
  float bsign=1.0f;
  bool have=false;
  int multiccd_off = dims[6];
  // Expand both sides to geom lists (bounded: 32 each).
  int ga[32], gb[32];
  int na=0, nb=0;
  if (otype==5) { ga[0]=oid; na=1; }
  else {
    int a0=body_geoms[oid*2+0], an=body_geoms[oid*2+1];
    for (int k=0;k<an && na<32;++k) { ga[na]=a0+k; na++; }
  }
  if (rtype==5) { gb[0]=rid; nb=1; }
  else {
    int b0=body_geoms[rid*2+0], bn=body_geoms[rid*2+1];
    for (int k=0;k<bn && nb<32;++k) { gb[nb]=b0+k; nb++; }
  }
  for (int a=0;a<na;++a) {
    for (int b=0;b<nb;++b) {
      int g1=ga[a], g2=gb[b];
      int t1=geom_type[g1], t2=geom_type[g2];
      bool flip = t1>t2;
      int s1 = flip?g2:g1, s2 = flip?g1:g2;
      float sign = flip?-1.0f:1.0f;
      float3 p1=sp_r3(geom_pos,(world*ngeom+uint(s1))*3);
      float3 p2=sp_r3(geom_pos,(world*ngeom+uint(s2))*3);
      float4 q1=sp_qunit(sp_r4(geom_quat,(world*ngeom+uint(s1))*4));
      float4 q2=sp_qunit(sp_r4(geom_quat,(world*ngeom+uint(s2))*4));
      int tt1=flip?t2:t1, tt2=flip?t1:t2;
      float3 sz1=sp_r3(geom_size,uint(s1)*3), sz2=sp_r3(geom_size,uint(s2)*3);
      float rb1=geom_rbound[s1], rb2=geom_rbound[s2];
      ContactGeom con[16];
      int n=collide_pair(tt1,p1,q1,sz1,rb1,tt2,p2,q2,sz2,rb2,
                         cutoff,multiccd_off,con,s1,s2,hull,hull_info,16);
      for (int k=0;k<n;++k) {
        if (con[k].dist<best) {
          best=con[k].dist;
          bpos=con[k].pos;
          bnorm=con[k].normal;
          bsign=sign;
          have=true;
        }
      }
      (void)sign;
    }
  }
  float3 f0 = have ? bpos-0.5f*bsign*best*bnorm : float3(0.0f);
  float3 f1 = have ? bpos+0.5f*bsign*best*bnorm : float3(0.0f);
  if (typ==39) {
    output[base]=best;
  } else if (typ==40) {
    float3 seg=f1-f0;
    float3 n=(have && dot(seg,seg)>0.0f) ? normalize(seg) : float3(0.0f);
    output[base]=n.x; output[base+1]=n.y; output[base+2]=n.z;
  } else {
    output[base]=f0.x; output[base+1]=f0.y; output[base+2]=f0.z;
    output[base+3]=f1.x; output[base+4]=f1.y; output[base+5]=f1.z;
  }
  (void)dim;
}
