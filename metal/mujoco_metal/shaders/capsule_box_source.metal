// Copyright 2016 Svetoslav Kolev
// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Source-rounded mjraw_{Sphere,Capsule}Box from pinned engine_collision_box.c.
// Capsule endpoint/edge ties must see the retained FK matrices, not matrices
// reconstructed from rounded float quaternions. Losing a tiny capsule tilt
// changes the manifold cardinality and the coupled trajectory.

struct CBNum {
  FlexDD words;
  CBNum() : words(flex_dd(0.0f)) {}
  CBNum(float value) : words(flex_dd(value)) {}
  CBNum(int value) : words(flex_dd(float(value))) {}
  CBNum(FlexDD value) : words(value) {}
};
inline CBNum operator+(CBNum a, CBNum b) { return CBNum(flex_dd_source_add(a.words,b.words)); }
inline CBNum operator-(CBNum a, CBNum b) { return CBNum(flex_dd_sub(a.words,b.words)); }
inline CBNum operator-(CBNum a) { return CBNum(flex_dd_neg(a.words)); }
inline CBNum operator*(CBNum a, CBNum b) { return CBNum(flex_dd_source_mul(a.words,b.words)); }
inline CBNum operator/(CBNum a, CBNum b) {
  if (flex_dd_compare(b.words,flex_dd(0.0f))==0) {
    if (flex_dd_compare(a.words,flex_dd(0.0f))==0)
      return CBNum(as_type<float>(0x7fc00000u));
    uint sign=(as_type<uint>(a.words.hi)^as_type<uint>(b.words.hi))&0x80000000u;
    return CBNum(as_type<float>(sign|0x7f800000u));
  }
  return CBNum(flex_dd_source_div(a.words,b.words));
}
inline bool operator<(CBNum a, CBNum b) {
  uint aa=as_type<uint>(a.words.hi),bb=as_type<uint>(b.words.hi);
  uint am=aa&0x7fffffffu,bm=bb&0x7fffffffu;
  // Metal fast-math can fold isinf/isnan under its finite-value assumption.
  // Classify encoded words before the finite expansion arithmetic instead.
  if(am>0x7f800000u || bm>0x7f800000u)return false;
  if(am==0x7f800000u || bm==0x7f800000u) {
    bool an=(aa>>31)!=0,bn=(bb>>31)!=0;
    if(am==0x7f800000u && bm==0x7f800000u)return an&&!bn;
    return am==0x7f800000u?an:!bn;
  }
  return flex_dd_compare(a.words,b.words)<0;
}
inline bool operator>(CBNum a, CBNum b) { return b<a; }
inline bool operator<=(CBNum a, CBNum b) {
  if((as_type<uint>(a.words.hi)&0x7fffffffu)>0x7f800000u ||
     (as_type<uint>(b.words.hi)&0x7fffffffu)>0x7f800000u)return false;
  return !(b<a);
}
inline CBNum cb_abs(CBNum a) {
  // C fabs(-0) is +0. Keeping the sign would turn an unbounded positive
  // second-point distance into -infinity and incorrectly select that bound.
  if(flex_dd_compare(a.words,flex_dd(0.0f))==0)return CBNum(0);
  return CBNum(flex_dd_abs(a.words));
}
inline CBNum cb_sqrt(CBNum a) { return CBNum(flex_dd_source_sqrt(a.words)); }
inline void cb_copy3(thread CBNum* d, thread const CBNum* s) {
  for(int i=0;i<3;i++) d[i]=s[i];
}
inline void cb_zero3(thread CBNum* d) { for(int i=0;i<3;i++) d[i]=0; }
inline void cb_sub3(thread CBNum* d,thread const CBNum* a,thread const CBNum* b) {
  for(int i=0;i<3;i++) d[i]=a[i]-b[i];
}
inline void cb_add3(thread CBNum* d,thread const CBNum* a,thread const CBNum* b) {
  for(int i=0;i<3;i++) d[i]=a[i]+b[i];
}
inline void cb_addTo3(thread CBNum* d,thread const CBNum* a) {
  for(int i=0;i<3;i++) d[i]=d[i]+a[i];
}
inline void cb_subFrom3(thread CBNum* d,thread const CBNum* a) {
  for(int i=0;i<3;i++) d[i]=d[i]-a[i];
}
inline void cb_scl3(thread CBNum* d,thread const CBNum* a,CBNum scale) {
  for(int i=0;i<3;i++) d[i]=a[i]*scale;
}
inline void cb_addToScl3(thread CBNum* d,thread const CBNum* a,CBNum scale) {
  for(int i=0;i<3;i++) d[i]=CBNum(flex_dd_source_fma(a[i].words,scale.words,d[i].words));
}
inline CBNum cb_dot3(thread const CBNum* a,thread const CBNum* b) {
  return CBNum(flex_dd_source_dot3(FlexDD3{a[0].words,a[1].words,a[2].words},
                                 FlexDD3{b[0].words,b[1].words,b[2].words}));
}
inline void cb_mulMatVec3(thread CBNum* d,thread const CBNum* m,thread const CBNum* v) {
  for(int i=0;i<3;i++) d[i]=CBNum(flex_dd_source_matvec_row(m[3*i].words,m[3*i+1].words,
      m[3*i+2].words,v[0].words,v[1].words,v[2].words));
}
inline void cb_mulMatTVec3(thread CBNum* d,thread const CBNum* m,thread const CBNum* v) {
  for(int i=0;i<3;i++) d[i]=CBNum(flex_dd_source_matvec_row(m[i].words,m[i+3].words,
      m[i+6].words,v[0].words,v[1].words,v[2].words));
}
inline CBNum cb_normalize3(thread CBNum* v) {
  CBNum length=cb_sqrt(cb_dot3(v,v));
  CBNum minval(FlexDD{1e-15f,-3.627493647833322e-24f,0.0f});
  if(length<minval) { cb_zero3(v);v[0]=1;return 0; }
  CBNum inverse=CBNum(1)/length;
  cb_scl3(v,v,inverse);return length;
}
inline void cb_clampVec(thread CBNum* v,thread const CBNum* size,int n) {
  for(int i=0;i<n;i++) if(size[i]>0) {
    if(v[i]<-size[i])v[i]=-size[i];else if(v[i]>size[i])v[i]=size[i];
  }
}
struct CBPreContact { CBNum dist,pos[3],normal[3],tangent[3]; };

// The two raw source routines follow below, with scalar operations mapped to
// the shared native 53-bit arithmetic and no changes to branch thresholds.
int cb_raw_sphere_box(thread CBPreContact* con, CBNum margin,
                    thread const CBNum* pos1, thread const CBNum* mat1, thread const CBNum* size1,
                    thread const CBNum* pos2, thread const CBNum* mat2, thread const CBNum* size2) {
  int i, k;
  CBNum tmp[3], center[3], clamped[3], deepest[3];
  CBNum pos[3];
  CBNum dist, closest;

  cb_sub3(tmp, pos1, pos2);
  cb_mulMatTVec3(center, mat2, tmp);

  cb_copy3(clamped, center);
  cb_clampVec(clamped, size2, 3);

  cb_copy3(deepest, center);
  cb_sub3(tmp, clamped, center);
  dist = cb_normalize3(tmp);

  if (dist - size1[0] > margin)
    return 0;

  if (dist <= CBNum(FlexDD{1e-15f,-3.627493647833322e-24f,0.0f})) {
    closest = (size2[0] + size2[1] + size2[2]) * 2;

    for (i = 0; i < 6; i++) {
      if (closest > cb_abs((i % 2 ? 1 : -1)*size2[i / 2] - center[i / 2])) {
        closest = cb_abs((i % 2 ? 1 : -1) * size2[i / 2] - center[i / 2]);
        k = i;
      }
    }

    CBNum nearest[3] = {CBNum(0),CBNum(0),CBNum(0)};
    nearest[k / 2] = (k % 2 ? -1 : 1);

    cb_copy3(pos, center);
    cb_addToScl3(pos, nearest, (size1[0] - closest) / 2);
    cb_mulMatVec3(con[0].normal, mat2, nearest);
    dist = -closest;
  } else {
    cb_addToScl3(deepest, tmp, size1[0]);
    cb_zero3(pos);
    cb_addToScl3(pos, clamped, 0.5);
    cb_addToScl3(pos, deepest, 0.5);
    cb_mulMatVec3(con[0].normal, mat2, tmp);
  }

  cb_mulMatVec3(tmp, mat2, pos);
  cb_add3(con[0].pos, tmp, pos2);
  con[0].dist = dist - size1[0];
  cb_zero3(con[0].tangent);
  return 1;
}



int cb_raw_capsule_box(thread CBPreContact* con, CBNum margin,
                     thread const CBNum* pos1, thread const CBNum* mat1, thread const CBNum* size1,
                     thread const CBNum* pos2, thread const CBNum* mat2,
                     thread const CBNum* size2) {
  CBNum tmp1[3], tmp2[3], tmp3[3], halfaxis[3], axis[3], dif[3];
  CBNum pos[3];

  CBNum halflength;
  CBNum bestdist;
  CBNum bestdistmax;
  CBNum bestsegmentpos;
  CBNum secondpos;
  CBNum dist;
  CBNum bestboxpos;
  CBNum mul, e1, e2, dp, de;

  CBNum ma, mb, mc, u, v, det, x1, x2, idet;

  int s1, s2;
  int i, j, c1, c2;
  int cltype = -4;
  int clface;
  int clcorner = 0;
  int cledge;
  int axisdir;
  int n;
  int ax1, ax2, ax;

  halflength = size1[1];
  secondpos = -4;

  cb_sub3(tmp1, pos1, pos2);
  cb_mulMatTVec3(pos, mat2, tmp1);

  tmp1[0] = mat1[2];
  tmp1[1] = mat1[5];
  tmp1[2] = mat1[8];

  cb_mulMatTVec3(axis, mat2, tmp1);
  cb_scl3(halfaxis, axis, halflength);

  axisdir = 0;
  if (halfaxis[0] > 0)
    axisdir += 1;
  if (halfaxis[1] > 0)
    axisdir += 2;
  if (halfaxis[2] > 0)
    axisdir += 4;

  bestdistmax = margin + 2 * (size1[0] + halflength + size2[0] + size2[1] +
                              size2[2]);
  bestdist = bestdistmax;
  bestsegmentpos = 0;

  cb_zero3(tmp2);

  for (i = -1; i <= 1; i += 2) {
    cb_copy3(tmp1, pos);
    cb_addToScl3(tmp1, halfaxis, i);
    cb_copy3(tmp2, tmp1);

    for (c1 = 0, j = 0, c2 = -1; j < 3; j++) {
      if (tmp1[j] < -size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = -size2[j];
      } else if (tmp1[j] > size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = size2[j];
      }
    }

    if (c1 > 1)
      continue;

    cb_subFrom3(tmp1, tmp2);
    dist = cb_dot3(tmp1, tmp1);

    if (dist < bestdist) {
      bestdist = dist;
      bestsegmentpos = i;
      cltype = -2 + i;
      clface = c2;
    }
  }

  cb_zero3(tmp2);

  for (j = 0; j < 3; j++) {
    for (i = 0; i < 8; i++) {
      if ((i & (1 << j)) == 0) {

        tmp3[0] = ((i & 1) ? 1 : -1) * size2[0];
        tmp3[1] = ((i & 2) ? 1 : -1) * size2[1];
        tmp3[2] = ((i & 4) ? 1 : -1) * size2[2];
        tmp3[j] = 0;

        cb_sub3(dif, tmp3, pos);

        ma = size2[j] * size2[j];
        mb = -size2[j] * halfaxis[j];
        mc = size1[1] * size1[1];

        u = -size2[j] * dif[j];
        v = cb_dot3(halfaxis, dif);

        det = ma * mc - mb * mb;
        if (cb_abs(det) < CBNum(FlexDD{1e-15f,-3.627493647833322e-24f,0.0f}))
          continue;
        idet = 1 / det;

        x1 = (mc * u - mb * v) * idet;
        x2 = (ma * v - mb * u) * idet;

        s1 = s2 = 1;

        if (x1 > 1) {
          x1 = 1;
          s1 = 2;
          x2 = (v - mb) * (1 / mc);
        } else if (x1 < -1) {
          x1 = -1;
          s1 = 0;
          x2 = (v + mb) * (1 / mc);
        }

        if (x2 > 1) {
          x2 = 1;
          s2 = 2;
          x1 = (u - mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        } else if (x2 < -1) {
          x2 = -1;
          s2 = 0;
          x1 = (u + mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        }

        cb_sub3(dif, tmp3, pos);

        cb_addToScl3(dif, halfaxis, -x2);
        dif[j] = CBNum(flex_dd_source_fma(size2[j].words,x1.words,dif[j].words));

        tmp1[2] = cb_dot3(dif, dif);

        c1 = s1 * 3 + s2;

        if (tmp1[2] < bestdist - CBNum(FlexDD{1e-15f,-3.627493647833322e-24f,0.0f})) {
          bestdist = tmp1[2];
          bestsegmentpos = x2;
          bestboxpos = x1;

          c2 = c1 / 6;

          clcorner = i + (1 << j) * c2;
          cledge = j;
          cltype = c1;
        }
      }
    }
  }

  if (cltype == -4)
    return 0;

  if (cltype >= 0 && cltype / 3 != 1) {
    c1 = axisdir ^ clcorner;

    if (c1 != 0 && c1 != 7) {

    if (c1 == 1 || c1 == 2 || c1 == 4) {
      mul = 1;
      de = 1 - bestsegmentpos;
      dp = 1 + bestsegmentpos;
    }

    if (c1 == 3 || c1 == 5 || c1 == 6) {
      mul = -1;
      c1 = 7 - c1;
      dp = 1 - bestsegmentpos;
      de = 1 + bestsegmentpos;
    }

    if (c1 == 1)
      ax = 0, ax1 = 1, ax2 = 2;
    if (c1 == 2)
      ax = 1, ax1 = 2, ax2 = 0;
    if (c1 == 4)
      ax = 2, ax1 = 0, ax2 = 1;

    if (axis[ax]*axis[ax] > 0.5) {
      secondpos = de;
      e1 = 2 * size2[ax] / cb_abs(halfaxis[ax]);

      if (e1 < secondpos) {
        secondpos = e1;
      }
      secondpos = secondpos * mul;
    } else {
      secondpos = dp;

      e1 = 2 * size2[ax1] / cb_abs(halfaxis[ax1]);
      if (e1 < secondpos)
        secondpos = e1;

      e1 = 2 * size2[ax2] / cb_abs(halfaxis[ax2]);
      if (e1 < secondpos)
        secondpos = e1;

      secondpos = secondpos * (-mul);
    }
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {

    c1 = axisdir ^ clcorner;

    c1 &= 7 - (1 << cledge);

    if (c1 == 1 || c1 == 2 || c1 == 4) {

    if (cledge == 0)
      ax1 = 1, ax2 = 2;
    if (cledge == 1)
      ax1 = 2, ax2 = 0;
    if (cledge == 2)
      ax1 = 0, ax2 = 1;
    ax = cledge;

    if (cb_abs(axis[ax1]) > cb_abs(axis[ax2]))
      ax1 = ax2;
    ax2 = 3 - ax - ax1;

    if (c1 & (1 << ax2)) {
      mul = 1;
      secondpos = 1 - bestsegmentpos;
    } else {
      mul = -1;
      secondpos = 1 + bestsegmentpos;
    }

    e1 = 2 * size2[ax2] / cb_abs(halfaxis[ax2]);
    if (e1 < secondpos)
      secondpos = e1;

    if (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0))
      e2 = 1 - bestboxpos;
    else
      e2 = 1 + bestboxpos;

    e1 = size2[ax] * e2 / cb_abs(halfaxis[ax]);

    if (e1 < secondpos)
      secondpos = e1;

    secondpos = secondpos * mul;
    }
  } else if (cltype < 0) {

    if (clface != -1) {
    if (cltype == -3)
      mul = 1;
    else
      mul = -1;

    secondpos = 2;

    cb_copy3(tmp1, pos);
    cb_addToScl3(tmp1, halfaxis, -mul);

    for (i = 0; i < 3; i++) {
      if (i != clface) {
        e1 = (size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;

        e1 = (-size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;
      }
    }
    secondpos = secondpos * mul;
    }
  }

  cb_copy3(tmp1, pos);
  cb_addToScl3(tmp1, halfaxis, bestsegmentpos);
  cb_mulMatVec3(tmp2, mat2, tmp1);
  cb_addTo3(tmp2, pos2);

  n = cb_raw_sphere_box(con, margin, tmp2, mat1, size1, pos2, mat2, size2);

  if (secondpos > -3) {
    cb_copy3(tmp1, pos);
    cb_addToScl3(tmp1, halfaxis, secondpos + bestsegmentpos);
    cb_mulMatVec3(tmp2, mat2, tmp1);
    cb_addTo3(tmp2, pos2);
    n += cb_raw_sphere_box(con + n, margin, tmp2, mat1, size1, pos2, mat2, size2);
  }

  return n;
}

// Declared here; the canonical record helper is defined by the coupled unit.
inline void contact_cache_store(device float* records,int record,
                               thread const ContactGeom& contact);

kernel void contact_produce_capsule_box_source(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_pos_low [[buffer(1)]],
    device const float* geom_pos_tail [[buffer(2)]],
    device const float* geom_xmat [[buffer(3)]],
    device const float* geom_xmat_low [[buffer(4)]],
    device const float* geom_xmat_tail [[buffer(5)]],
    device const float* size_words [[buffer(6)]],
    device const int* geom_type [[buffer(7)]],
    device const int* pair_geoms [[buffer(8)]],
    device const float* margin_words [[buffer(9)]],
    device const int* dims [[buffer(10)]],
    device const int* logical_to_packed [[buffer(11)]],
    device int* contact_count [[buffer(12)]],
    device float* records [[buffer(13)]],
    uint tid [[thread_position_in_grid]]) {
  int npairs=dims[1],nc=dims[2],batch=dims[3],ng=dims[6];
  int world=int(tid)/max(npairs,1),pair=int(tid)%max(npairs,1);
  if(npairs<=0 || world>=batch || pair>=npairs || !pair_contact_world_enabled(dims,world))return;
  int a=pair_geoms[2*pair],b=pair_geoms[2*pair+1];
  int ta=geom_type[a],tb=geom_type[b];
  if(!((ta==3 && tb==6)||(ta==6 && tb==3)))return;
  contact_count[world*npairs+pair]=0;
  if((dims[9]&1)!=0 && logical_to_packed[world*npairs+pair]<0)return;
  bool swapped=ta==6;if(swapped){int old=a;a=b;b=old;}
  int go=world*ng,sw=3*ng;
  thread CBNum pa[3],pb[3],ma[9],mb[9],sa[3],sb[3];
  for(int i=0;i<3;i++) {
    pa[i]=CBNum(FlexDD{geom_pos[(go+a)*3+i],geom_pos_low[(go+a)*3+i],geom_pos_tail[(go+a)*3+i]});
    pb[i]=CBNum(FlexDD{geom_pos[(go+b)*3+i],geom_pos_low[(go+b)*3+i],geom_pos_tail[(go+b)*3+i]});
    sa[i]=CBNum(FlexDD{size_words[a*3+i],size_words[sw+a*3+i],size_words[2*sw+a*3+i]});
    sb[i]=CBNum(FlexDD{size_words[b*3+i],size_words[sw+b*3+i],size_words[2*sw+b*3+i]});
  }
  for(int i=0;i<9;i++) {
    ma[i]=CBNum(FlexDD{geom_xmat[(go+a)*9+i],geom_xmat_low[(go+a)*9+i],geom_xmat_tail[(go+a)*9+i]});
    mb[i]=CBNum(FlexDD{geom_xmat[(go+b)*9+i],geom_xmat_low[(go+b)*9+i],geom_xmat_tail[(go+b)*9+i]});
  }
  CBNum margin(FlexDD{margin_words[pair],margin_words[npairs+pair],margin_words[2*npairs+pair]});
  thread CBPreContact con[2];
  int count=cb_raw_capsule_box(con,margin,pa,ma,sa,pb,mb,sb);
  int offset=dims[10+pair],capacity=dims[11+pair]-offset;
  count=min(count,capacity);contact_count[world*npairs+pair]=count;
  for(int k=0;k<count;k++) {
    ContactGeom c;
    c.dist=con[k].dist.words.hi;
    c.pos=float3(con[k].pos[0].words.hi,con[k].pos[1].words.hi,con[k].pos[2].words.hi);
    float sign=swapped?-1.0f:1.0f;
    c.normal=sign*float3(con[k].normal[0].words.hi,con[k].normal[1].words.hi,con[k].normal[2].words.hi);
    c.t1=float3(0.0f);make_frame(c.normal,c.t1,c.t2);
    int record=world*nc+offset+k;contact_cache_store(records,record,c);
    int base=record*37;
    records[base+9]=con[k].dist.words.lo;records[base+10]=con[k].dist.words.tail;
    for(int i=0;i<3;i++) {
      records[base+25+i]=con[k].pos[i].words.lo;
      records[base+31+i]=con[k].pos[i].words.tail;
      records[base+28+i]=sign*con[k].normal[i].words.lo;
      records[base+34+i]=sign*con[k].normal[i].words.tail;
    }
  }
}

