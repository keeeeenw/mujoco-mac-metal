// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Shared small helpers for milestone-016 RNE/sensor kernels (pinned
// spatial-algebra ports).

#include <metal_stdlib>
using namespace metal;

inline float3 r3(device const float* a, uint i) {
  return float3(a[i],a[i+1],a[i+2]);
}
inline float4 r4(device const float* a, uint i) {
  return float4(a[i],a[i+1],a[i+2],a[i+3]);
}
inline float4 rqmul(float4 a, float4 b) {
  return float4(a.x*b.x-dot(a.yzw,b.yzw),
      a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
inline float4 rqconj(float4 q) { return float4(q.x,-q.yzw); }
inline float4 rqunit(float4 q) {
  float n=length(q);
  return n>1e-30f ? q/n : float4(1.0f,0.0f,0.0f,0.0f);
}
inline float3 rqrot(float4 q, float3 v) {
  return v+2.0f*cross(q.yzw,cross(q.yzw,v)+q.x*v);
}
inline float3 rmat_col(float4 q, int col) {
  if (col==0) return rqrot(q,float3(1,0,0));
  if (col==1) return rqrot(q,float3(0,1,0));
  return rqrot(q,float3(0,0,1));
}
// 6D motion cross product: res = vel x s (pinned mju_crossMotion).
inline void cross_motion(float3 wa, float3 wl, float3 sa, float3 sl,
    thread float3& ra, thread float3& rl) {
  ra=cross(wa,sa);
  rl=cross(wa,sl)+cross(wl,sa);
}
// 6D force cross product: res = vel x f (pinned mju_crossForce).
inline void cross_force(float3 wa, float3 wl, float3 fa, float3 fl,
    thread float3& ra, thread float3& rl) {
  ra=cross(wa,fa)+cross(wl,fl);
  rl=cross(wa,fl);
}
// Com-based spatial inertia multiply (pinned mju_mulInertVec). Layout (10):
// [Ixx,Iyy,Izz,Ixy,Ixz,Iyz,hx,hy,hz,m].
inline void mul_inert(thread float* I, float3 va, float3 vl,
    thread float3& ra, thread float3& rl) {
  ra=float3(I[0]*va.x+I[3]*va.y+I[4]*va.z - I[8]*vl.y+I[7]*vl.z,
            I[3]*va.x+I[1]*va.y+I[5]*va.z + I[8]*vl.x-I[6]*vl.z,
            I[4]*va.x+I[5]*va.y+I[2]*va.z - I[7]*vl.x+I[6]*vl.y);
  rl=float3(I[8]*va.y-I[7]*va.z+I[9]*vl.x,
            I[6]*va.z-I[8]*va.x+I[9]*vl.y,
            I[7]*va.x-I[6]*va.y+I[9]*vl.z);
}
// Shift torque:force from oldpos to newpos (pinned mju_transformSpatial
// with identity rotation, flg_force=1).
inline float3 shift_torque(float3 torque, float3 force, float3 np_, float3 op) {
  return torque-cross(np_-op,force);
}
// Two-word arithmetic used only for the derivative side-channel.  Existing
// sensor values keep their original single-word operation order; the low word
// records the residual of a more accurate evaluation of the same transform.
inline float2 sensor_dd_add(float2 a, float2 b) {
  float s=a.x+b.x;
  float v=s-a.x;
  float e=(a.x-(s-v))+(b.x-v)+a.y+b.y;
  float h=s+e;
  return float2(h,e-(h-s));
}
inline float2 sensor_dd_neg(float2 a) { return float2(-a.x,-a.y); }
inline float2 sensor_dd_sub(float2 a, float2 b) {
  return sensor_dd_add(a,sensor_dd_neg(b));
}
inline float2 sensor_dd_mul(float2 a, float2 b) {
  float p=a.x*b.x;
  float e=fma(a.x,b.x,-p)+a.x*b.y+a.y*b.x+a.y*b.y;
  float h=p+e;
  return float2(h,e-(h-p));
}
inline void sensor_dd_cross(thread const float2* a,
                            thread const float2* b,
                            thread float2* out) {
  out[0]=sensor_dd_sub(sensor_dd_mul(a[1],b[2]),
                       sensor_dd_mul(a[2],b[1]));
  out[1]=sensor_dd_sub(sensor_dd_mul(a[2],b[0]),
                       sensor_dd_mul(a[0],b[2]));
  out[2]=sensor_dd_sub(sensor_dd_mul(a[0],b[1]),
                       sensor_dd_mul(a[1],b[0]));
}
inline void sensor_dd_cross_motion(thread const float2* wa,
                                   thread const float2* wl,
                                   thread const float2* sa,
                                   thread const float2* sl,
                                   thread float2* ra,
                                   thread float2* rl) {
  float2 first[3], second[3];
  sensor_dd_cross(wa, sa, ra);
  sensor_dd_cross(wa, sl, first);
  sensor_dd_cross(wl, sa, second);
  for (uint k=0;k<3;++k) rl[k]=sensor_dd_add(first[k],second[k]);
}
inline void sensor_dd_axpy3(thread float2* dst,
                            thread const float2* src, float scale) {
  float2 s=float2(scale,0.0f);
  for (uint k=0;k<3;++k)
    dst[k]=sensor_dd_add(dst[k],sensor_dd_mul(src[k],s));
}
inline void sensor_dd_rotate(float4 q, thread const float2* v,
                             thread float2* out) {
  float2 qv[3] = {float2(q.y,0.0f),float2(q.z,0.0f),
                  float2(q.w,0.0f)};
  float2 cross1[3], cross2[3];
  sensor_dd_cross(qv,v,cross1);
  for (uint k=0;k<3;++k)
    cross1[k]=sensor_dd_add(cross1[k],sensor_dd_mul(float2(q.x,0.0f),v[k]));
  sensor_dd_cross(qv,cross1,cross2);
  for (uint k=0;k<3;++k)
    out[k]=sensor_dd_add(v[k],sensor_dd_mul(float2(2.0f,0.0f),cross2[k]));
}
inline float sensor_dd_residual(float2 accurate, float legacy) {
  float2 diff=sensor_dd_sub(accurate,float2(legacy,0.0f));
  return diff.x+diff.y;
}
struct SensorComScratch { float3 scom; float smass; float3 smom; };
static_assert(sizeof(SensorComScratch)==48, "sensor COM workspace ABI");
// cfrc_ext assembly: xfrc_applied + solver-included contacts +
// connect/weld equalities, all as subtree-com-based torque:force.
kernel void assemble_cfrc_ext(
    device const float* xfrc [[buffer(0)]],
    device const float* contact_frame [[buffer(1)]],
    device const float* contact_force [[buffer(2)]],
    device const float* contact_row [[buffer(3)]],
    device const int* contact_packed [[buffer(4)]],
    device const float* contact_mu [[buffer(5)]],
    device const int* pair_geoms [[buffer(6)]],
    device const int* pair_offset [[buffer(7)]],
    device const int* geom_bodyid [[buffer(8)]],
    device const int* body_jntnum [[buffer(9)]],
    device const int* eq_meta [[buffer(10)]],
    device const int* eq_rowadr [[buffer(11)]],
    device const float* eq_data [[buffer(12)]],
    device const float* site_lpos [[buffer(13)]],
    device const int* site_body [[buffer(14)]],
    device const float* lam_all [[buffer(15)]],
    device const float* body_pos [[buffer(16)]],
    device const float* body_quat [[buffer(17)]],
    device const float* inertial_pos [[buffer(18)]],
    device const float* mass [[buffer(19)]],
    device const int* body_tree [[buffer(20)]],
    device float* cfrc_ext [[buffer(21)]],
    constant int* dims [[buffer(22)]],
    device SensorComScratch* scratch [[buffer(23)]],
    uint world [[thread_position_in_grid]]) {
  // eq_meta per equality (4 ints): type, objtype, obj1, obj2. (objtype: 1
  // body, else site; world body 0 skipped like pinned.)
  // eq_data per equality (11 floats, pinned layout: anchors at [0:3]/[3:6]).
  // lam_all: raw workspace_debug; lam region at nr*nr+3*nr, stride S.
  // contact_packed per slot (3 ints): cdim, row_offset, cone.
  // contact_row per slot (1 float): row_data active flag.
  // dims prefix (8 ints): batch, nbody, neq, nc, npairs, has_xfrc, nr, S;
  // the appended batch selector uses 1=compute and 0=preserve.
  uint batch=uint(dims[0]), nbody=uint(dims[1]);
  uint neq=uint(dims[2]), nc=uint(dims[3]);
  uint has_xfrc=uint(dims[5]), nr=uint(dims[6]), S=uint(dims[7]);
  if (world>=batch) return;
  if (dims[8+int(world)] == 0) return;
  uint b3=world*nbody*3, b6=world*nbody*6;
  for (uint b=0;b<nbody;++b) {
    cfrc_ext[b6+b*6+0]=0.0f; cfrc_ext[b6+b*6+1]=0.0f; cfrc_ext[b6+b*6+2]=0.0f;
    cfrc_ext[b6+b*6+3]=0.0f; cfrc_ext[b6+b*6+4]=0.0f; cfrc_ext[b6+b*6+5]=0.0f;
  }
  // Subtree com (pinned mj_comPos: mass-weighted moments accumulated
  // backward, normalized by subtree mass; xipos fallback below MINVAL).
  device SensorComScratch* sub=scratch+world*nbody;
  for (uint b=0;b<nbody;++b) {
    sub[b].scom=r3(inertial_pos,b3+b*3);
    sub[b].smass=mass[b];
    sub[b].smom=mass[b]*r3(inertial_pos,b3+b*3);
  }
  for (int k=int(nbody)-1;k>0;--k) {
    uint kk=uint(k), p=uint(body_tree[kk*2+0]);
    sub[p].smass+=sub[kk].smass;
    sub[p].smom+=sub[kk].smom;
  }
  for (uint b=0;b<nbody;++b) {
    sub[b].scom=sub[b].smass>1e-15f ? sub[b].smom/sub[b].smass : r3(inertial_pos,b3+b*3);
  }
  // xfrc_applied: torque:force rearranged, mapped to com.
  if (has_xfrc!=0) {
    for (uint b=1;b<nbody;++b) {
      float3 F=r3(xfrc,b6+b*6), T=r3(xfrc,b6+b*6+3);
      if (dot(F,F)+dot(T,T)==0.0f) continue;
      uint root=uint(body_tree[b*2+1]);
      T=shift_torque(T,F,sub[root].scom,r3(inertial_pos,b3+b*3));
      cfrc_ext[b6+b*6+0]+=T.x; cfrc_ext[b6+b*6+1]+=T.y; cfrc_ext[b6+b*6+2]+=T.z;
      cfrc_ext[b6+b*6+3]+=F.x; cfrc_ext[b6+b*6+4]+=F.y; cfrc_ext[b6+b*6+5]+=F.z;
    }
  }
  // Contacts with solver rows: skip undetected slots and static-static
  // pairs (pinned NV==0 exclusion: neither ancestor chain has joints).
  // contact_row is per-world flattened (world*nc+s); packed/mu are per-slot.
  for (uint s=0;s<nc;++s) {
    if (contact_row[world*nc+s]<=0.5f) continue;
    int cdim=contact_packed[s*3+0], cone=contact_packed[s*3+2];
    uint pair=0;
    for (uint pp=0;pp<uint(dims[4]);++pp) {
      if (s>=uint(pair_offset[pp]) && s<uint(pair_offset[pp+1])) { pair=pp; break; }
    }
    int g1=pair_geoms[pair*2+0], g2=pair_geoms[pair*2+1];
    int b1 = g1>=0 ? geom_bodyid[g1] : -1;
    int b2 = g2>=0 ? geom_bodyid[g2] : -1;
    bool live=false;
    for (int side=0;side<2 && !live;++side) {
      int bb = side==0 ? b1 : b2;
      while (bb>0) {
        if (body_jntnum[bb]>0) { live=true; break; }
        // parent walk needs parent ids: body_tree[bb*2+0].
        bb=body_tree[uint(bb)*2+0];
      }
    }
    if (!live) continue;
    float3 n=r3(contact_frame,(world*nc+s)*12);
    float3 t1=r3(contact_frame,(world*nc+s)*12+3);
    float3 t2=r3(contact_frame,(world*nc+s)*12+6);
    float3 ppos=r3(contact_frame,(world*nc+s)*12+9);
    // Six-component contact-local wrench (pinned mj_contactForce +
    // mju_decodePyramid: frictionless condim-1 reads the single pyramid
    // value; pyramidal torque is zero for point contacts; elliptic
    // condim>3 carries torsional/rolling torque in entries 3-5).
    float fl[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (cone==0) {
      if (cdim==1) {
        fl[0]=contact_force[(world*nc+s)*11];
      } else {
        float fn=0.0f;
        int ne2=2*(cdim-1);
        for (int k=0;k<ne2;++k) fn+=contact_force[(world*nc+s)*11+1+k];
        fl[0]=fn;
        for (int k=0;k<cdim-1;++k) {
          float mu=contact_mu[s*5+k];
          fl[1+k]=(contact_force[(world*nc+s)*11+1+2*k]-contact_force[(world*nc+s)*11+2+2*k])*mu;
        }
      }
    } else {
      for (int k=0;k<cdim && k<6;++k) fl[k]=contact_force[(world*nc+s)*11+k];
    }
    float3 fw=n*fl[0]+t1*fl[1]+t2*fl[2];
    float3 tw=float3(0.0f);
    if (cdim>3) {
      float3 tl=float3(fl[3],fl[4],fl[5]);
      tw=n*tl.x+t1*tl.y+t2*tl.z;
    }
    if (b1>0) {
      uint root=uint(body_tree[uint(b1)*2+1]);
      float3 T=shift_torque(tw,fw,sub[root].scom,ppos);
      cfrc_ext[b6+uint(b1)*6+0]-=T.x; cfrc_ext[b6+uint(b1)*6+1]-=T.y; cfrc_ext[b6+uint(b1)*6+2]-=T.z;
      cfrc_ext[b6+uint(b1)*6+3]-=fw.x; cfrc_ext[b6+uint(b1)*6+4]-=fw.y; cfrc_ext[b6+uint(b1)*6+5]-=fw.z;
    }
    if (b2>0) {
      uint root=uint(body_tree[uint(b2)*2+1]);
      float3 T=shift_torque(tw,fw,sub[root].scom,ppos);
      cfrc_ext[b6+uint(b2)*6+0]+=T.x; cfrc_ext[b6+uint(b2)*6+1]+=T.y; cfrc_ext[b6+uint(b2)*6+2]+=T.z;
      cfrc_ext[b6+uint(b2)*6+3]+=fw.x; cfrc_ext[b6+uint(b2)*6+4]+=fw.y; cfrc_ext[b6+uint(b2)*6+5]+=fw.z;
    }
  }
  // Connect/weld equalities (types 0/1); joint/tendon rows skipped.
  for (uint e=0;e<neq;++e) {
    int et=eq_meta[e*4+0];
    if (et!=0 && et!=1) continue;
    int row0=eq_rowadr[e];
    float3 F=float3(lam_all[world*S+nr*nr+3*nr+uint(row0)],
                    lam_all[world*S+nr*nr+3*nr+uint(row0)+1],
                    lam_all[world*S+nr*nr+3*nr+uint(row0)+2]);
    float3 T=float3(0.0f);
    if (et==1) {
      T=float3(lam_all[world*S+nr*nr+3*nr+uint(row0)+3],
               lam_all[world*S+nr*nr+3*nr+uint(row0)+4],
               lam_all[world*S+nr*nr+3*nr+uint(row0)+5]);
    }
    int ot=eq_meta[e*4+1], o1=eq_meta[e*4+2], o2=eq_meta[e*4+3];
    // Body 1 (adds).
    {
      int k = -1;
      float3 pos;
      if (ot==1) {
        k=o1;
        float3 off = et==1 ? float3(eq_data[e*11+3],eq_data[e*11+4],eq_data[e*11+5])
                           : float3(eq_data[e*11+0],eq_data[e*11+1],eq_data[e*11+2]);
        float4 q=rqunit(r4(body_quat,(world*nbody+uint(k))*4));
        pos=r3(body_pos,(world*nbody+uint(k))*3)+rqrot(q,off);
      } else {
        k=site_body[o1];
        float4 q=rqunit(r4(body_quat,(world*nbody+uint(k))*4));
        pos=r3(body_pos,(world*nbody+uint(k))*3)+rqrot(q,r3(site_lpos,uint(o1)*3));
      }
      if (k>0) {
        uint root=uint(body_tree[uint(k)*2+1]);
        float3 Tc=shift_torque(T,F,sub[root].scom,pos);
        cfrc_ext[b6+uint(k)*6+0]+=Tc.x; cfrc_ext[b6+uint(k)*6+1]+=Tc.y; cfrc_ext[b6+uint(k)*6+2]+=Tc.z;
        cfrc_ext[b6+uint(k)*6+3]+=F.x; cfrc_ext[b6+uint(k)*6+4]+=F.y; cfrc_ext[b6+uint(k)*6+5]+=F.z;
      }
    }
    // Body 2 (subtracts).
    {
      int k = -1;
      float3 pos;
      if (ot==1) {
        k=o2;
        float3 off = et==0 ? float3(eq_data[e*11+3],eq_data[e*11+4],eq_data[e*11+5])
                           : float3(eq_data[e*11+0],eq_data[e*11+1],eq_data[e*11+2]);
        float4 q=rqunit(r4(body_quat,(world*nbody+uint(k))*4));
        pos=r3(body_pos,(world*nbody+uint(k))*3)+rqrot(q,off);
      } else {
        k=site_body[o2];
        float4 q=rqunit(r4(body_quat,(world*nbody+uint(k))*4));
        pos=r3(body_pos,(world*nbody+uint(k))*3)+rqrot(q,r3(site_lpos,uint(o2)*3));
      }
      if (k>0) {
        uint root=uint(body_tree[uint(k)*2+1]);
        float3 Tc=shift_torque(T,F,sub[root].scom,pos);
        cfrc_ext[b6+uint(k)*6+0]-=Tc.x; cfrc_ext[b6+uint(k)*6+1]-=Tc.y; cfrc_ext[b6+uint(k)*6+2]-=Tc.z;
        cfrc_ext[b6+uint(k)*6+3]-=F.x; cfrc_ext[b6+uint(k)*6+4]-=F.y; cfrc_ext[b6+uint(k)*6+5]-=F.z;
      }
    }
  }
}

// Forward cacc + cfrc_int (pinned mj_rnePostConstraint tail): cacc from
// cdof/cdof_dot (pinned mj_comPos/mj_comVel), cfrc backward accumulation.
kernel void rne_post(
    device const float* qvel [[buffer(0)]],
    device const float* qacc [[buffer(1)]],
    device const float* cvel [[buffer(2)]],
    device const float* mass [[buffer(3)]],
    device const float* inertia [[buffer(4)]],
    device const int* body_tree [[buffer(5)]],
    device const int* body_jnt [[buffer(6)]],
    device const int* jnt_type [[buffer(7)]],
    device const int* jnt_dofadr [[buffer(8)]],
    device const float* joint_anchor [[buffer(9)]],
    device const float* joint_axis [[buffer(10)]],
    device const float* inertial_pos [[buffer(11)]],
    device const float* inertial_quat [[buffer(12)]],
    device const float* body_quat [[buffer(13)]],
    device const float* cfrc_ext [[buffer(14)]],
    device float* cacc [[buffer(15)]],
    device float* cfrc_int [[buffer(16)]],
    device float* scom_out [[buffer(17)]],
    constant int* dims [[buffer(18)]],
    device const float* gravity [[buffer(19)]],
    device SensorComScratch* scratch [[buffer(20)]],
    device const float* qacc_low [[buffer(21)]],
    uint world [[thread_position_in_grid]]) {
  // body_jnt per body (4 ints): jntadr, jntnum, dofadr, dofnum.
  // dims prefix (5 ints): batch, nbody, nv, njnt, grav_off; the appended
  // batch selector uses 1=compute and 0=preserve.
  uint batch=uint(dims[0]), nbody=uint(dims[1]), nv=uint(dims[2]);
  uint njnt=uint(dims[3]);
  if (world>=batch) return;
  if (dims[5+int(world)] == 0) return;
  uint vb=world*nv, b3=world*nbody*3, b6=world*nbody*6;
  device float* cacc_low=cacc+batch*nbody*6;
  // Subtree com (pinned mj_comPos: moment accumulation + MINVAL fallback).
  device SensorComScratch* sub=scratch+world*nbody;
  for (uint b=0;b<nbody;++b) {
    sub[b].scom=r3(inertial_pos,b3+b*3);
    sub[b].smass=mass[b];
    sub[b].smom=mass[b]*r3(inertial_pos,b3+b*3);
  }
  for (int k=int(nbody)-1;k>0;--k) {
    uint kk=uint(k), p=uint(body_tree[kk*2+0]);
    sub[p].smass+=sub[kk].smass;
    sub[p].smom+=sub[kk].smom;
  }
  for (uint b=0;b<nbody;++b) {
    sub[b].scom=sub[b].smass>1e-15f ? sub[b].smom/sub[b].smass : r3(inertial_pos,b3+b*3);
  }
  // Root-com-per-body for downstream kernels.
  for (uint b=0;b<nbody;++b) {
    uint r=uint(body_tree[b*2+1]);
    scom_out[(world*nbody+b)*3+0]=sub[r].scom.x;
    scom_out[(world*nbody+b)*3+1]=sub[r].scom.y;
    scom_out[(world*nbody+b)*3+2]=sub[r].scom.z;
  }
  // World cacc = -gravity.
  float3 wg=float3(0.0f);
  if (dims[4]==0) wg=-float3(gravity[0],gravity[1],gravity[2]);
  for (uint k=0;k<6;++k) {
    cacc[b6+k]=0.0f;
    cacc_low[b6+k]=0.0f;
  }
  cacc[b6+3]=wg.x; cacc[b6+4]=wg.y; cacc[b6+5]=wg.z;
  // Per-body forward pass (bodies are topologically ordered).
  for (uint b=1;b<nbody;++b) {
    uint p=uint(body_tree[b*2+0]);
    uint root=uint(body_tree[b*2+1]);
    float3 cw=r3(cvel,b6+p*6), cv=r3(cvel,b6+p*6+3);
    float3 ca=r3(cacc,b6+p*6), cl=r3(cacc,b6+p*6+3);
    float3 ca_low=r3(cacc_low,b6+p*6), cl_low=r3(cacc_low,b6+p*6+3);
    // Keep an independent two-word spatial-acceleration recurrence.  The
    // legacy high-word recurrence below preserves its established operation
    // order; this pair carries product residuals (not just qacc_low) through
    // cdof*qacc and cdof_dot*qvel before the ACC side-channel is formed.
    float2 cw_pair[3], cv_pair[3], ca_pair[3], cl_pair[3];
    for (uint k=0;k<3;++k) {
      cw_pair[k]=float2(cw[k],0.0f);
      cv_pair[k]=float2(cv[k],0.0f);
      ca_pair[k]=float2(ca[k],ca_low[k]);
      cl_pair[k]=float2(cl[k],cl_low[k]);
    }
    float3 com=sub[root].scom;
    int ja=body_jnt[b*4+0], jn=body_jnt[b*4+1];
    for (int j=0;j<jn;++j) {
      int jj=ja+j;
      int jt=jnt_type[jj];
      int dd=jnt_dofadr[jj];
      float3 anchor=r3(joint_anchor,(world*njnt+uint(jj))*3);
      float3 axis=r3(joint_axis,(world*njnt+uint(jj))*3);
      float3 off=com-anchor;
      if (jt==2 || jt==3) {
        float3 Sa, Sl;
        if (jt==3) { Sa=axis; Sl=cross(axis,off); }
        else { Sa=float3(0.0f); Sl=axis; }
        float2 Sa_pair[3], Sl_pair[3];
        for (uint k=0;k<3;++k) {
          Sa_pair[k]=float2(Sa[k],0.0f);
          Sl_pair[k]=float2(Sl[k],0.0f);
        }
        float qv=qvel[vb+uint(dd)], qa=qacc[vb+uint(dd)];
        float qa_low=qacc_low[vb+uint(dd)];
        float3 cda, cdl;
        cross_motion(cw,cv,Sa,Sl,cda,cdl);
        ca+=cda*qv; cl+=cdl*qv;
        ca+=Sa*qa; cl+=Sl*qa;
        ca_low+=Sa*qa_low; cl_low+=Sl*qa_low;
        float2 cda_pair[3], cdl_pair[3];
        sensor_dd_cross_motion(cw_pair,cv_pair,Sa_pair,Sl_pair,
                               cda_pair,cdl_pair);
        sensor_dd_axpy3(ca_pair,cda_pair,qv);
        sensor_dd_axpy3(cl_pair,cdl_pair,qv);
        sensor_dd_axpy3(ca_pair,Sa_pair,qa);
        sensor_dd_axpy3(cl_pair,Sl_pair,qa);
        sensor_dd_axpy3(ca_pair,Sa_pair,qa_low);
        sensor_dd_axpy3(cl_pair,Sl_pair,qa_low);
        cw+=Sa*qv; cv+=Sl*qv;
        sensor_dd_axpy3(cw_pair,Sa_pair,qv);
        sensor_dd_axpy3(cv_pair,Sl_pair,qv);
      } else if (jt==1 || jt==0) {
        // Ball (3 rotary dofs) and free (3 slide + 3 rotary): pinned
        // mj_comVel computes all rotary cdofdots from the same cvel.
        float4 q=rqunit(r4(body_quat,(world*nbody+b)*4));
        int r0 = jt==0 ? dd+3 : dd;
        float3 cw0=cw, cv0=cv;
        float2 cw0_pair[3], cv0_pair[3];
        for (uint k=0;k<3;++k) {
          cw0_pair[k]=cw_pair[k];
          cv0_pair[k]=cv_pair[k];
        }
        if (jt==0) {
          for (int k=0;k<3;++k) {
            float qv=qvel[vb+uint(dd+k)], qa=qacc[vb+uint(dd+k)];
            float qa_low=qacc_low[vb+uint(dd+k)];
            cv0[k]+=qv; cl[k]+=qa;
            cl_low[k]+=qa_low;
            cv0_pair[k]=sensor_dd_add(cv0_pair[k],float2(qv,0.0f));
            cl_pair[k]=sensor_dd_add(
                cl_pair[k],sensor_dd_mul(float2(qa,qa_low),float2(1.0f,0.0f)));
          }
          cw=cw0;
          // NOTE: translation part of cvel for cdofdot below uses updated cv0.
          float3 cda, cdl;
          for (int k=0;k<3;++k) {
            float3 ax=rmat_col(q,k);
            float2 ax_pair[3], off_pair[3], cross_axis[3];
            for (uint c=0;c<3;++c) {
              ax_pair[c]=float2(ax[c],0.0f);
              off_pair[c]=float2(off[c],0.0f);
            }
            sensor_dd_cross(ax_pair,off_pair,cross_axis);
            float qv=qvel[vb+uint(r0+k)], qa=qacc[vb+uint(r0+k)];
            float qa_low=qacc_low[vb+uint(r0+k)];
            cross_motion(cw0,cv0,ax,cross(ax,off),cda,cdl);
            ca+=cda*qv; cl+=cdl*qv;
            ca+=ax*qa; cl+=cross(ax,off)*qa;
            ca_low+=ax*qa_low; cl_low+=cross(ax,off)*qa_low;
            float2 cda_pair[3], cdl_pair[3];
            sensor_dd_cross_motion(cw0_pair,cv0_pair,ax_pair,cross_axis,
                                   cda_pair,cdl_pair);
            sensor_dd_axpy3(ca_pair,cda_pair,qv);
            sensor_dd_axpy3(cl_pair,cdl_pair,qv);
            sensor_dd_axpy3(ca_pair,ax_pair,qa);
            sensor_dd_axpy3(cl_pair,cross_axis,qa);
            sensor_dd_axpy3(ca_pair,ax_pair,qa_low);
            sensor_dd_axpy3(cl_pair,cross_axis,qa_low);
            cw+=ax*qv; cv+=cross(ax,off)*qv;
            sensor_dd_axpy3(cw_pair,ax_pair,qv);
            sensor_dd_axpy3(cv_pair,cross_axis,qv);
          }
        } else {
          for (int k=0;k<3;++k) {
            float3 ax=rmat_col(q,k);
            float2 ax_pair[3], off_pair[3], cross_axis[3];
            for (uint c=0;c<3;++c) {
              ax_pair[c]=float2(ax[c],0.0f);
              off_pair[c]=float2(off[c],0.0f);
            }
            sensor_dd_cross(ax_pair,off_pair,cross_axis);
            float qv=qvel[vb+uint(r0+k)], qa=qacc[vb+uint(r0+k)];
            float qa_low=qacc_low[vb+uint(r0+k)];
            float3 cda, cdl;
            cross_motion(cw0,cv0,ax,cross(ax,off),cda,cdl);
            ca+=cda*qv; cl+=cdl*qv;
            ca+=ax*qa; cl+=cross(ax,off)*qa;
            ca_low+=ax*qa_low; cl_low+=cross(ax,off)*qa_low;
            float2 cda_pair[3], cdl_pair[3];
            sensor_dd_cross_motion(cw0_pair,cv0_pair,ax_pair,cross_axis,
                                   cda_pair,cdl_pair);
            sensor_dd_axpy3(ca_pair,cda_pair,qv);
            sensor_dd_axpy3(cl_pair,cdl_pair,qv);
            sensor_dd_axpy3(ca_pair,ax_pair,qa);
            sensor_dd_axpy3(cl_pair,cross_axis,qa);
            sensor_dd_axpy3(ca_pair,ax_pair,qa_low);
            sensor_dd_axpy3(cl_pair,cross_axis,qa_low);
          }
          cw+=float3(0.0f);
          for (int k=0;k<3;++k) {
            float3 ax=rmat_col(q,k);
            float qv=qvel[vb+uint(r0+k)];
            cw+=ax*qv; cv+=cross(ax,off)*qv;
            float2 ax_pair[3], off_pair[3], cross_axis[3];
            for (uint c=0;c<3;++c) {
              ax_pair[c]=float2(ax[c],0.0f);
              off_pair[c]=float2(off[c],0.0f);
            }
            sensor_dd_cross(ax_pair,off_pair,cross_axis);
            sensor_dd_axpy3(cw_pair,ax_pair,qv);
            sensor_dd_axpy3(cv_pair,cross_axis,qv);
          }
        }
      }
    }
    cacc[b6+b*6+0]=ca.x; cacc[b6+b*6+1]=ca.y; cacc[b6+b*6+2]=ca.z;
    cacc[b6+b*6+3]=cl.x; cacc[b6+b*6+4]=cl.y; cacc[b6+b*6+5]=cl.z;
    cacc_low[b6+b*6+0]=ca_low.x; cacc_low[b6+b*6+1]=ca_low.y;
    cacc_low[b6+b*6+2]=ca_low.z; cacc_low[b6+b*6+3]=cl_low.x;
    cacc_low[b6+b*6+4]=cl_low.y; cacc_low[b6+b*6+5]=cl_low.z;
    cacc_low[b6+b*6+0]=sensor_dd_residual(ca_pair[0],ca.x);
    cacc_low[b6+b*6+1]=sensor_dd_residual(ca_pair[1],ca.y);
    cacc_low[b6+b*6+2]=sensor_dd_residual(ca_pair[2],ca.z);
    cacc_low[b6+b*6+3]=sensor_dd_residual(cl_pair[0],cl.x);
    cacc_low[b6+b*6+4]=sensor_dd_residual(cl_pair[1],cl.y);
    cacc_low[b6+b*6+5]=sensor_dd_residual(cl_pair[2],cl.z);
    // cfrc_body = cinert*cacc + cvel x (cinert*cvel), com-based.
    float3 xip=r3(inertial_pos,b3+b*3);
    float4 xiq=rqunit(r4(inertial_quat,(world*nbody+b)*4));
    float3 ib=float3(inertia[b*3],inertia[b*3+1],inertia[b*3+2]);
    float m=mass[b];
    float3 c0=rmat_col(xiq,0), c1=rmat_col(xiq,1), c2=rmat_col(xiq,2);
    float3 Rw0=c0*ib.x, Rw1=c1*ib.y, Rw2=c2*ib.z;
    float3 dif=xip-com;
    float I_[10];
    I_[0]=dot(c0,Rw0)+m*(dif.y*dif.y+dif.z*dif.z);
    I_[1]=dot(c1,Rw1)+m*(dif.x*dif.x+dif.z*dif.z);
    I_[2]=dot(c2,Rw2)+m*(dif.x*dif.x+dif.y*dif.y);
    I_[3]=dot(c0,Rw1)-m*dif.x*dif.y;
    I_[4]=dot(c0,Rw2)-m*dif.x*dif.z;
    I_[5]=dot(c1,Rw2)-m*dif.y*dif.z;
    I_[6]=m*dif.x; I_[7]=m*dif.y; I_[8]=m*dif.z; I_[9]=m;
    float3 bw=r3(cvel,b6+b*6);
    float3 bv=r3(cvel,b6+b*6+3);
    float3 Ia, Il, Ma, Ml, ga, gl;
    mul_inert(I_,ca,cl,Ia,Il);
    mul_inert(I_,bw,bv,Ma,Ml);
    cross_force(bw,bv,Ma,Ml,ga,gl);
    Ia+=ga; Il+=gl;
    float3 exa=r3(cfrc_ext,b6+b*6), exl=r3(cfrc_ext,b6+b*6+3);
    cfrc_int[b6+b*6+0]=Ia.x-exa.x; cfrc_int[b6+b*6+1]=Ia.y-exa.y; cfrc_int[b6+b*6+2]=Ia.z-exa.z;
    cfrc_int[b6+b*6+3]=Il.x-exl.x; cfrc_int[b6+b*6+4]=Il.y-exl.y; cfrc_int[b6+b*6+5]=Il.z-exl.z;
  }
  for (uint k=0;k<6;++k) cfrc_int[b6+k]=0.0f;
  // Backward accumulation to parents.
  for (int k=int(nbody)-1;k>0;--k) {
    uint pp=uint(body_tree[uint(k)*2+0]);
    for (uint j=0;j<6;++j) cfrc_int[b6+pp*6+j]+=cfrc_int[b6+uint(k)*6+j];
  }
}

// Milestone 016 ACC families: touch, accelerometer, force, torque,
// actuator/joint/tendon/limit forces, frame accelerations. Reads RNE-post
// cacc/cfrc_int plus retained solve/actuator rows. TOUCH zone tests reuse
// analytic ray shapes (hit-only); CONTACT arrives with the spatial commit.
kernel void evaluate_acc_sensors(
    device const float* cacc [[buffer(0)]],
    device const float* cfrc_int [[buffer(1)]],
    device const float* scom [[buffer(2)]],
    device const float* act_force [[buffer(3)]],
    device const float* qfrc_act [[buffer(4)]],
    device const int* act_trn [[buffer(5)]],
    device const int* jnt_map [[buffer(6)]],
    device const int* ten_lim_row [[buffer(7)]],
    device const float* lam_all [[buffer(8)]],
    device const float* site_pos [[buffer(9)]],
    device const float* site_quat [[buffer(10)]],
    device const float* body_pos [[buffer(11)]],
    device const float* inertial_pos [[buffer(12)]],
    device const float* geom_pos [[buffer(13)]],
    device const float* cvel [[buffer(14)]],
    device const float* root_com [[buffer(15)]],
    device const float* contact_frame [[buffer(16)]],
    device const float* contact_force [[buffer(17)]],
    device const float* contact_row [[buffer(18)]],
    device const int* contact_packed [[buffer(19)]],
    device const float* contact_mu [[buffer(20)]],
    device const int* slot_pair [[buffer(21)]],
    device const int* pair_live [[buffer(22)]],
    device const float* site_geom [[buffer(23)]],
    device const int* meta [[buffer(24)]],
    device const int* geom_bodyid [[buffer(25)]],
    device const int* site_bodyid [[buffer(26)]],
    device const int* body_weld [[buffer(27)]],
    device float* output [[buffer(28)]],
    constant int* stage_mask [[buffer(29)]],
    constant int* dims [[buffer(30)]],
    uint index [[thread_position_in_grid]]) {
  // scom (buffer 2) carries each body's ROOT subtree com: sub[b].scom gives the
  // com frame of body b's own dynamics (written by rne_post).
  // meta (10 ints): type, datatype, needstage, objtype, objid, dim, adr,
  //   cutoff_bits, reftype, refid. jnt_map per joint (3 ints): row_lo,
  //   row_hi, dofadr (rows -1 when absent). act_trn per actuator (2 ints):
  //   trntype, trnid0. contact_packed per slot (3 ints): cdim, row_offset,
  //   cone. site_geom per site (4 floats): size + type.
  // dims (14 ints): batch, nsensor, ndata, nr, S, nc, nu, nt, nv, njnt,
  //   nbody, has_act, nsite, ngeom, followed by a batch selector.
  uint batch=uint(dims[0]), nsensor=uint(dims[1]), ndata=uint(dims[2]);
  device const float* cacc_low=cacc+batch*uint(dims[10])*6;
  device float* output_low=output+batch*ndata;
  uint nr=uint(dims[3]), S=uint(dims[4]);
  uint nu=uint(dims[6]), nv=uint(dims[8]);
  uint nbody=uint(dims[10]), nsite=uint(dims[12]);
  uint ACC=3;
  if (index>=batch*nsensor) return;
  if ((uint(stage_mask[0])&(1u<<ACC))==0) return;
  uint world=index/nsensor, i=index-world*nsensor;
  if (dims[14+int(world)] == 0) return;
  if (stage_mask[1+batch*max(nsensor,1u)+world] == 0) return;
  if (stage_mask[1+index] == 0) return;
  if (uint(meta[i*10+2])!=ACC) return;
  int typ=meta[i*10+0], objid=meta[i*10+4];
  uint dim=uint(meta[i*10+5]), adr=uint(meta[i*10+6]);
  uint base=world*ndata+adr;
  // Force families here; touch(0)/contact(42) have the spatial kernel.
  bool handled = typ==1||typ==4||typ==5||typ==15||typ==16||typ==17
      ||typ==22||typ==25||typ==33||typ==34;
  if (!handled) return;
  float value[6] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
  uint vb=world*nv;
  if (typ==15) {
    if (objid>=0 && uint(objid)<nu && dims[11]!=0)
      value[0]=act_force[world*nu+uint(objid)];
  } else if (typ==16) {
    if (objid>=0) {
      int da=jnt_map[objid*3+2];
      if (da>=0 && uint(da)<nv) value[0]=qfrc_act[vb+uint(da)];
    }
  } else if (typ==17) {
    float frc=0.0f;
    if (dims[11]!=0) {
      for (uint j=0;j<nu;++j) {
        if (act_trn[j*2+0]==3 && act_trn[j*2+1]==objid)
          frc+=act_force[world*nu+j];
      }
    }
    value[0]=frc;
  } else if (typ==22 || typ==25) {
    int row = typ==22 ? jnt_map[objid*3+0] : ten_lim_row[objid*2+0];
    int row2 = typ==22 ? jnt_map[objid*3+1] : ten_lim_row[objid*2+1];
    // Pinned reads the first active row; retained lam is 0 for inactive.
    float f0 = row>=0 ? lam_all[world*S+nr*nr+3*nr+uint(row)] : 0.0f;
    float f1 = row2>=0 ? lam_all[world*S+nr*nr+3*nr+uint(row2)] : 0.0f;
    value[0] = f0!=0.0f ? f0 : f1;
  } else if (typ==1) {
    // Accelerometer: site spatial acceleration in site frame (pinned
    // mj_objectAcceleration local + Coriolis correction). scom carries
    // each body's own root com, so no root lookup is needed.
    uint s=uint(objid);
    int b=site_bodyid[s];
    // Pinned mj_objectAcceleration returns zero for bodies welded to world,
    // including mocap frames. Do not expose RNE's -gravity cacc in this case.
    if (body_weld[b] == 0) {
      for (uint k=0;k<dim;++k) output_low[base+k]=0.0f;
    } else {
      float3 sp=r3(site_pos,(world*nsite+s)*3);
      float3 com=r3(scom,(world*nbody+uint(b))*3);
      float3 aa=r3(cacc,(world*nbody+uint(b))*6);
      float3 al0=r3(cacc,(world*nbody+uint(b))*6+3);
      float3 al=al0;
      al=al-cross(sp-com,aa);
      float3 wa=r3(cvel,(world*nbody+uint(b))*6);
      float3 wl=r3(cvel,(world*nbody+uint(b))*6+3);
      float3 vw=wl+cross(wa,sp-com);
      float4 sq=rqunit(r4(site_quat,(world*nsite+s)*4));
      float3 ll=rqrot(rqconj(sq),al);
      float3 lv_ang=rqrot(rqconj(sq),wa);
      float3 lv_lin=rqrot(rqconj(sq),vw);
      ll+=cross(lv_ang,lv_lin);
      value[0]=ll.x; value[1]=ll.y; value[2]=ll.z;
      // Evaluate the same angular-acceleration, point-shift, Coriolis and site
      // rotation with two-word intermediates.  This is a side-channel for
      // finite differences: the ordinary output above remains bit-for-bit on
      // its established float32 path.
      float2 off[3], aa_dd[3], al_dd[3], wa_dd[3], wl_dd[3];
      float2 cross_a[3], cross_v[3], vw_dd[3], lv_ang_dd[3], lv_lin_dd[3];
      for (uint k=0;k<3;++k) {
        off[k]=sensor_dd_sub(float2(sp[k],0.0f),float2(com[k],0.0f));
        aa_dd[k]=float2(aa[k],cacc_low[(world*nbody+uint(b))*6+k]);
        wa_dd[k]=float2(wa[k],0.0f);
        wl_dd[k]=float2(wl[k],0.0f);
      }
      sensor_dd_cross(off,aa_dd,cross_a);
      for (uint k=0;k<3;++k)
        al_dd[k]=sensor_dd_sub(
            float2(al0[k],cacc_low[(world*nbody+uint(b))*6+3+k]),cross_a[k]);
      sensor_dd_cross(wa_dd,off,cross_v);
      for (uint k=0;k<3;++k)
        vw_dd[k]=sensor_dd_add(wl_dd[k],cross_v[k]);
      float4 sqc=rqconj(sq);
      sensor_dd_rotate(sqc,al_dd,lv_ang_dd); // angular acceleration
      sensor_dd_rotate(sqc,wa_dd,cross_a);   // angular velocity
      sensor_dd_rotate(sqc,vw_dd,lv_lin_dd); // point linear velocity
      sensor_dd_cross(cross_a,lv_lin_dd,cross_v);
      for (uint k=0;k<3;++k) {
        float2 acc_dd=sensor_dd_add(lv_ang_dd[k],cross_v[k]);
        output_low[base+k]=sensor_dd_residual(acc_dd,value[k]);
      }
    }
  } else if (typ==4 || typ==5) {
    // Force/torque: body interaction wrench in site frame (pinned
    // transformSpatial flg_force=1, then site rotation).
    uint s=uint(objid);
    int b=site_bodyid[s];
    float3 sp=r3(site_pos,(world*nsite+s)*3);
    float3 com=r3(scom,(world*nbody+uint(b))*3);
    float3 t=r3(cfrc_int,(world*nbody+uint(b))*6);
    float3 f=r3(cfrc_int,(world*nbody+uint(b))*6+3);
    t=t-cross(sp-com,f);
    float4 sq=rqunit(r4(site_quat,(world*nsite+s)*4));
    t=rqrot(rqconj(sq),t);
    f=rqrot(rqconj(sq),f);
    float3 v = typ==4 ? f : t;
    value[0]=v.x; value[1]=v.y; value[2]=v.z;
  } else if (typ==33 || typ==34) {
    // Frame accelerations in global frame (pinned mj_objectAcceleration
    // flg_local=0 + Coriolis correction). Static bodies read zero.
    int otype=meta[i*10+3];
    int bb=objid;
    float3 pos;
    if (otype==1) pos=r3(inertial_pos,(world*nbody+uint(objid))*3);
    else if (otype==2) pos=r3(body_pos,(world*nbody+uint(objid))*3);
    else if (otype==5) {
      bb=geom_bodyid[objid];
      pos=r3(geom_pos,(world*uint(dims[13])+uint(objid))*3);
    } else {
      bb=site_bodyid[objid];
      pos=r3(site_pos,(world*nsite+uint(objid))*3);
    }
    if (body_weld[bb]==0) {
      value[0]=0.0f; value[1]=0.0f; value[2]=0.0f;
    } else {
      float3 com=r3(scom,(world*nbody+uint(bb))*3);
      float3 aa=r3(cacc,(world*nbody+uint(bb))*6);
      float3 al=r3(cacc,(world*nbody+uint(bb))*6+3);
      al=al-cross(pos-com,aa);
      float3 wa=r3(cvel,(world*nbody+uint(bb))*6);
      float3 wl=r3(cvel,(world*nbody+uint(bb))*6+3);
      float3 vw=wl+cross(wa,pos-com);
      al+=cross(wa,vw);
      float3 v = typ==33 ? al : aa;
      value[0]=v.x; value[1]=v.y; value[2]=v.z;
    }
  }
  float cutoff=as_type<float>(meta[i*10+7]);
  int datatype=meta[i*10+1];
  if (cutoff>0.0f && (datatype==0 || datatype==1)) {
    for (uint j=0;j<dim && j<6;++j) {
      float unclamped=value[j];
      float clipped=datatype==0 ? clamp(unclamped,-cutoff,cutoff)
                                : min(unclamped,cutoff);
      if (typ==1) {
        float2 accurate=sensor_dd_add(float2(unclamped,0.0f),
                                      float2(output_low[base+j],0.0f));
        if (datatype==0) {
          if (accurate.x>cutoff ||
              (accurate.x==cutoff && accurate.y>0.0f)) {
            clipped=cutoff; output_low[base+j]=0.0f;
          } else if (accurate.x< -cutoff ||
                     (accurate.x== -cutoff && accurate.y<0.0f)) {
            clipped= -cutoff; output_low[base+j]=0.0f;
          } else {
            output_low[base+j]=sensor_dd_residual(accurate,clipped);
          }
        } else if (accurate.x>cutoff ||
                   (accurate.x==cutoff && accurate.y>0.0f)) {
          clipped=cutoff; output_low[base+j]=0.0f;
        } else {
          output_low[base+j]=sensor_dd_residual(accurate,clipped);
        }
      } else {
        output_low[base+j]=0.0f;
      }
      value[j]=clipped;
    }
  } else if (typ!=1) {
    for (uint j=0;j<dim && j<6;++j) output_low[base+j]=0.0f;
  }
  for (uint j=0;j<dim && j<6;j++) output[base+j]=value[j];
}
