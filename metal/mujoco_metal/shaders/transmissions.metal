// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

kernel void scalar_transmission_force(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* ctrl [[buffer(2)]],
    device const float* length_map [[buffer(3)]],
    device const float* moment_map [[buffer(4)]],
    device const int* gain_affine [[buffer(5)]],
    device const float* gainprm [[buffer(6)]],
    device const int* bias_enabled [[buffer(7)]],
    device const float* biasprm [[buffer(8)]],
    device const int* ctrl_limited [[buffer(9)]],
    device const float* ctrl_range [[buffer(10)]],
    device const int* force_limited [[buffer(11)]],
    device const float* force_range [[buffer(12)]],
    device const int* actuator_group [[buffer(13)]],
    constant int* dims [[buffer(14)]],
    device float* qfrc [[buffer(15)]],
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nv=dims[1], nu=dims[2];
  int actuation_disabled=dims[3], clampctrl_disabled=dims[4], disableactuator=dims[5];
  uint qbase=world*uint(nq), vbase=world*uint(nv), ubase=world*uint(nu);
  for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=0.0f;
  for (int i=0;i<nq;++i) {
    if ((as_type<uint>(qpos[qbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      return;
    }
  }
  for (int i=0;i<nv;++i) {
    if ((as_type<uint>(qvel[vbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      return;
    }
  }
  for (int i=0;i<nu;++i) {
    if ((as_type<uint>(ctrl[ubase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      return;
    }
  }
  if (actuation_disabled) return;
  for (int a=0;a<nu;++a) {
    int group=actuator_group[a];
    if ((disableactuator & (1 << group)) != 0) continue;
    float length=0.0f, velocity=0.0f;
    for (int q=0;q<nq;++q) length+=length_map[a*nq+q]*qpos[qbase+uint(q)];
    for (int d=0;d<nv;++d) velocity+=moment_map[a*nv+d]*qvel[vbase+uint(d)];
    float input=ctrl[ubase+uint(a)];
    if (!clampctrl_disabled && ctrl_limited[a]) {
      input=clamp(input,ctrl_range[2*a],ctrl_range[2*a+1]);
    }
    float gain=gainprm[3*a];
    if (gain_affine[a]) gain+=gainprm[3*a+1]*length+gainprm[3*a+2]*velocity;
    float force=gain*input;
    if (bias_enabled[a]) force+=biasprm[3*a]+biasprm[3*a+1]*length+biasprm[3*a+2]*velocity;
    if (force_limited[a]) force=clamp(force,force_range[2*a],force_range[2*a+1]);
    for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]+=moment_map[a*nv+d]*force;
  }
  for (int d=0;d<nv;++d) {
    if ((as_type<uint>(qfrc[vbase+uint(d)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int k=0;k<nv;++k) qfrc[vbase+uint(k)]=bad;
      return;
    }
  }
}

// General actuator transmission kinematics for MuJoCo 3.10.0 (milestone 007).
// Pinned source: engine/engine_core_smooth.c mj_transmission. Computes
// per-actuator length, dense moment row and velocity from qpos, FK poses and
// qvel. Scalar hinge/slide joint and fixed-tendon rows use constant maps
// (identical math to the scalar fast path); ball/free gears, slider-crank and
// site transmissions are assembled state-dependently. BODY adhesion uses a
// separate kernel reading candidate-contact buffers (same-step contacts).

inline float4 ak_qmul(float4 a, float4 b) {
  return float4(a.x*b.x-dot(a.yzw,b.yzw), a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
inline float3 ak_qrot(float4 q, float3 v) {
  return v + 2.0f*cross(q.yzw, cross(q.yzw, v) + q.x*v);
}
inline float4 ak_qconj(float4 q) { return float4(q.x, -q.y, -q.z, -q.w); }
inline float4 ak_qnorm(float4 q) {
  float n = sqrt(dot(q,q));
  return n > 1e-30f ? q/n : float4(1.0f,0.0f,0.0f,0.0f);
}
inline float3 ak_quat2vel(float4 q) {
  float3 axis = q.yzw;
  float s = length(axis);
  float speed = 2.0f * atan2(s, q.x);
  if (speed > 3.14159265358979f) speed -= 2.0f*3.14159265358979f;
  if (s < 1e-30f) return float3(0.0f);
  return axis * (speed / s);
}
inline float3x3 ak_quat2mat(float4 q) {
  float w=q.x, x=q.y, y=q.z, z=q.w;
  return float3x3(
    1-2*(y*y+z*z), 2*(x*y+z*w),   2*(x*z-y*w),
    2*(x*y-z*w),   1-2*(x*x+z*z), 2*(y*z+x*w),
    2*(x*z+y*w),   2*(y*z-x*w),   1-2*(x*x+y*y));
}

// Translational point Jacobian Jp (3x32) + angular Jacobian Jr (3x32).
inline void ak_point_jac(float3 point, int body,
    device const float* body_pos, device const float* body_quat,
    device const float* anchors, device const float* axes,
    device const int* body_parentid, device const int* body_jntadr,
    device const int* body_jntnum, device const int* jnt_type,
    device const int* jnt_dofadr,
    int nbody, int njnt, int nv, int bo, int jo,
    thread float* Jp, thread float* Jr) {
  for (int i=0;i<3*32;++i) { Jp[i]=0.0f; Jr[i]=0.0f; }
  if (nv==0) return;
  int b=body;
  while (b>0 && b<nbody) {
    int ja=body_jntadr[b], jn=body_jntnum[b];
    for (int jj=0;jj<jn;++jj) {
      int j=ja+jj;
      if (j<0||j>=njnt) continue;
      int da=jnt_dofadr[j], typ=jnt_type[j];
      int nd = typ==0?6:(typ==1?3:1);
      for (int q=0;q<nd;++q) {
        int dof=da+q;
        if (dof<0||dof>=nv) continue;
        float3 col=float3(0.0f), ang=float3(0.0f);
        if (typ==2) {
          col=float3(axes[(jo+j)*3],axes[(jo+j)*3+1],axes[(jo+j)*3+2]);
        } else if (typ==3) {
          float3 ax=float3(axes[(jo+j)*3],axes[(jo+j)*3+1],axes[(jo+j)*3+2]);
          float3 an=float3(anchors[(jo+j)*3],anchors[(jo+j)*3+1],anchors[(jo+j)*3+2]);
          col=cross(ax,point-an); ang=ax;
        } else {
          if (typ==0 && q<3) { col=float3(q==0,q==1,q==2); }
          else {
            int qr = typ==0?q-3:q;
            float4 bq=float4(body_quat[(bo+b)*4],body_quat[(bo+b)*4+1],body_quat[(bo+b)*4+2],body_quat[(bo+b)*4+3]);
            float3 ax=ak_qrot(bq,float3(qr==0,qr==1,qr==2));
            ang=ax;
            float3 piv = typ==0
              ? float3(body_pos[(bo+b)*3],body_pos[(bo+b)*3+1],body_pos[(bo+b)*3+2])
              : float3(anchors[(jo+j)*3],anchors[(jo+j)*3+1],anchors[(jo+j)*3+2]);
            col=cross(ax,point-piv);
          }
        }
        Jp[0*32+dof]+=col.x; Jp[1*32+dof]+=col.y; Jp[2*32+dof]+=col.z;
        Jr[0*32+dof]+=ang.x; Jr[1*32+dof]+=ang.y; Jr[2*32+dof]+=ang.z;
      }
    }
    b=body_parentid[b];
  }
}

kernel void general_actuator_kinematics(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* body_pos [[buffer(2)]],
    device const float* body_quat [[buffer(3)]],
    device const float* site_pos [[buffer(4)]],
    device const float* site_quat [[buffer(5)]],
    device const float* joint_anchor [[buffer(6)]],
    device const float* joint_axis [[buffer(7)]],
    device const int* trntype [[buffer(8)]],
    device const int* trnid [[buffer(9)]],
    device const float* gear [[buffer(10)]],
    device const float* cranklength [[buffer(11)]],
    device const float* fixed_length_map [[buffer(12)]],
    device const float* fixed_moment_map [[buffer(13)]],
    device const int* jnt_type [[buffer(14)]],
    device const int* jnt_qposadr [[buffer(15)]],
    device const int* jnt_dofadr [[buffer(16)]],
    device const int* body_parentid [[buffer(17)]],
    device const int* body_jntadr [[buffer(18)]],
    device const int* body_jntnum [[buffer(19)]],
    device const int* body_weldid [[buffer(20)]],
    device const int* body_dofadr [[buffer(21)]],
    device const int* body_dofnum [[buffer(22)]],
    device const int* dof_parentid [[buffer(23)]],
    device const int* site_bodyid [[buffer(24)]],
    constant int* dims [[buffer(25)]],
    device float* out_length [[buffer(26)]],
    device float* out_velocity [[buffer(27)]],
    device float* out_moment [[buffer(28)]],
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nv=dims[1], nu=dims[2];
  int nbody=dims[3], njnt=dims[4], nsite=dims[5], batch=dims[7];
  if (uint(world)>=uint(batch)) return;
  uint qbase=uint(world)*uint(max(nq,1)), vbase=uint(world)*uint(max(nv,1));
  uint ubase=uint(world)*uint(max(nu,1));
  int bo=world*nbody, jo=world*max(njnt,1), so=world*max(nsite,1);
  for (int a=0;a<nu;++a) {
    float len=0.0f;
    float mom[32];
    for (int d=0;d<32;++d) mom[d]=0.0f;
    int t=trntype[a];
    int id0=trnid[2*a], id1=trnid[2*a+1];
    float g0=gear[6*a], g1=gear[6*a+1], g2=gear[6*a+2];
    if (t==0 || t==1) {
      int j=id0;
      int jt=(j>=0&&j<njnt)?jnt_type[j]:-1;
      if (jt==2 || jt==3) {
        int qa=jnt_qposadr[j], da=jnt_dofadr[j];
        len=qpos[qbase+uint(qa)]*g0;
        if (da>=0&&da<nv) mom[da]=g0;
      } else if (jt==1) {
        int qa=jnt_qposadr[j], da=jnt_dofadr[j];
        float4 q=ak_qnorm(float4(qpos[qbase+uint(qa)],qpos[qbase+uint(qa)+1],qpos[qbase+uint(qa)+2],qpos[qbase+uint(qa)+3]));
        float3 axis=ak_quat2vel(q);
        float3 gax=float3(g0,g1,g2);
        if (t==1) gax=ak_qrot(ak_qconj(q),gax);
        len=dot(axis,gax);
        for (int k=0;k<3;++k) { int dof=da+k; if (dof>=0&&dof<nv) mom[dof]=k==0?gax.x:(k==1?gax.y:gax.z); }
      } else if (jt==0) {
        int qa=jnt_qposadr[j], da=jnt_dofadr[j];
        float4 q=float4(qpos[qbase+uint(qa)+3],qpos[qbase+uint(qa)+4],qpos[qbase+uint(qa)+5],qpos[qbase+uint(qa)+6]);
        q=ak_qnorm(q);
        float3 gax=ak_qrot(t==0?float4(1,0,0,0):ak_qconj(q),float3(gear[6*a+3],gear[6*a+4],gear[6*a+5]));
        len=0.0f;
        for (int k=0;k<3;++k) { int dof=da+k; if (dof>=0&&dof<nv) mom[dof]=k==0?g0:(k==1?g1:g2); }
        for (int k=0;k<3;++k) { int dof=da+3+k; if (dof>=0&&dof<nv) mom[dof]=k==0?gax.x:(k==1?gax.y:gax.z); }
      }
    } else if (t==3) {
      for (int q=0;q<nq;++q) len+=fixed_length_map[a*max(nq,1)+q]*qpos[qbase+uint(q)];
      for (int d=0;d<nv;++d) mom[d]=fixed_moment_map[a*max(nv,1)+d];
    } else if (t==2) {
      int crank=id0, slider=id1;
      float3 sp=float3(site_pos[(so+slider)*3],site_pos[(so+slider)*3+1],site_pos[(so+slider)*3+2]);
      float4 sq=float4(site_quat[(so+slider)*4],site_quat[(so+slider)*4+1],site_quat[(so+slider)*4+2],site_quat[(so+slider)*4+3]);
      float3x3 sm=ak_quat2mat(sq);
      float3 axis=float3(sm[0][2],sm[1][2],sm[2][2]);
      float3 cp=float3(site_pos[(so+crank)*3],site_pos[(so+crank)*3+1],site_pos[(so+crank)*3+2]);
      float3 vec=cp-sp;
      float rod=cranklength[a];
      float av=dot(vec,axis);
      float det=av*av+rod*rod-dot(vec,vec);
      float sdet=0.0f;
      bool ok=det>0.0f;
      if (ok) { sdet=sqrt(det); len=av-sdet; } else { len=av; }
      thread float jacS[96], jacStmp[96], jacP[96], jacPt[96];
      ak_point_jac(sp, (slider>=0&&slider<nsite)?site_bodyid[slider]:0,
        body_pos,body_quat,joint_anchor,joint_axis,body_parentid,body_jntadr,body_jntnum,
        jnt_type,jnt_dofadr,nbody,njnt,nv,bo,jo,jacS,jacStmp);
      ak_point_jac(cp, (crank>=0&&crank<nsite)?site_bodyid[crank]:0,
        body_pos,body_quat,joint_anchor,joint_axis,body_parentid,body_jntadr,body_jntnum,
        jnt_type,jnt_dofadr,nbody,njnt,nv,bo,jo,jacP,jacPt);
      float dlda[3], dldv[3];
      if (ok) {
        float k1=1.0f-av/sdet;
        dldv[0]=axis.x*k1+vec.x/sdet; dldv[1]=axis.y*k1+vec.y/sdet; dldv[2]=axis.z*k1+vec.z/sdet;
        float k2=1.0f/sdet;
        dlda[0]=vec.x*k2; dlda[1]=vec.y*k2; dlda[2]=vec.z*k2;
      } else {
        dlda[0]=vec.x; dlda[1]=vec.y; dlda[2]=vec.z;
        dldv[0]=axis.x; dldv[1]=axis.y; dldv[2]=axis.z;
      }
      for (int d=0;d<nv;++d) {
        float row=0.0f;
        float3 jr=float3(jacStmp[0*32+d],jacStmp[1*32+d],jacStmp[2*32+d]);
        float3 cx=cross(jr,axis);
        for (int k=0;k<3;++k) {
          float jp = (k==0?jacP[0*32+d]:(k==1?jacP[1*32+d]:jacP[2*32+d]))
                   - (k==0?jacS[0*32+d]:(k==1?jacS[1*32+d]:jacS[2*32+d]));
          float axj = k==0?cx.x:(k==1?cx.y:cx.z);
          float dd0 = k==0?dlda[0]:(k==1?dlda[1]:dlda[2]);
          float dd1 = k==0?dldv[0]:(k==1?dldv[1]:dldv[2]);
          row += dd0*axj + dd1*jp;
        }
        mom[d]=row*g0;
      }
      len*=g0;
    } else if (t==4) {
      thread float Jt[96], Jr[96];
      int sb=(id0>=0&&id0<nsite)?site_bodyid[id0]:0;
      float3 sp=float3(site_pos[(so+id0)*3],site_pos[(so+id0)*3+1],site_pos[(so+id0)*3+2]);
      float4 sq=float4(site_quat[(so+id0)*4],site_quat[(so+id0)*4+1],site_quat[(so+id0)*4+2],site_quat[(so+id0)*4+3]);
      ak_point_jac(sp,sb,body_pos,body_quat,joint_anchor,joint_axis,body_parentid,
        body_jntadr,body_jntnum,jnt_type,jnt_dofadr,nbody,njnt,nv,bo,jo,Jt,Jr);
      if (id1==-1) {
        float3x3 sm=ak_quat2mat(sq);
        float3 wt=sm*float3(g0,g1,g2);
        float3 wr=sm*float3(gear[6*a+3],gear[6*a+4],gear[6*a+5]);
        len=0.0f;
        for (int d=0;d<nv;++d)
          mom[d]=Jt[0*32+d]*wt.x+Jt[1*32+d]*wt.y+Jt[2*32+d]*wt.z
                +Jr[0*32+d]*wr.x+Jr[1*32+d]*wr.y+Jr[2*32+d]*wr.z;
      } else {
        int rsb=(id1>=0&&id1<nsite)?site_bodyid[id1]:0;
        float3 rp=float3(site_pos[(so+id1)*3],site_pos[(so+id1)*3+1],site_pos[(so+id1)*3+2]);
        float4 rq=float4(site_quat[(so+id1)*4],site_quat[(so+id1)*4+1],site_quat[(so+id1)*4+2],site_quat[(so+id1)*4+3]);
        int b0=(sb>=0&&sb<nbody)?body_weldid[sb]:0;
        int b1=(rsb>=0&&rsb<nbody)?body_weldid[rsb]:0;
        int da0=(b0>=0&&b0<nbody)?(body_dofadr[b0]+body_dofnum[b0]-1):-1;
        int da1=(b1>=0&&b1<nbody)?(body_dofadr[b1]+body_dofnum[b1]-1):-1;
        int common=-1;
        if (da0>=0&&da1>=0) {
          int x0=da0,x1=da1,guard=0;
          while (x0!=x1&&guard<64) {
            if (x0<x1) x1=(x1>=0&&x1<1024)?dof_parentid[x1]:-1; else x0=(x0>=0&&x0<1024)?dof_parentid[x0]:-1;
            if (x0==-1||x1==-1) break;
            guard++;
          }
          if (x0==x1) common=x0;
        }
        len=0.0f;
        for (int d=0;d<nv;++d) mom[d]=0.0f;
        bool has_t = (g0!=0.0f||g1!=0.0f||g2!=0.0f);
        bool has_r = (gear[6*a+3]!=0.0f||gear[6*a+4]!=0.0f||gear[6*a+5]!=0.0f);
        if (has_t) {
          thread float Jr0[96], Jt0[96];
          ak_point_jac(rp,rsb,body_pos,body_quat,joint_anchor,joint_axis,body_parentid,
            body_jntadr,body_jntnum,jnt_type,jnt_dofadr,nbody,njnt,nv,bo,jo,Jt0,Jr0);
          float3 vec=sp-rp;
          float3x3 rm=ak_quat2mat(rq);
          float3 vloc=transpose(rm)*vec;
          len+=dot(vloc,float3(g0,g1,g2));
          for (int d=0;d<nv;++d) {
            Jt[0*32+d]-=Jt0[0*32+d]; Jt[1*32+d]-=Jt0[1*32+d]; Jt[2*32+d]-=Jt0[2*32+d];
          }
          int da=common;
          int g2_=0;
          while (da>=0&&g2_<64) {
            Jt[0*32+da]=0.0f; Jt[1*32+da]=0.0f; Jt[2*32+da]=0.0f;
            da=(da>=0&&da<1024)?dof_parentid[da]:-1;
            g2_++;
          }
          float3 wt=rm*float3(g0,g1,g2);
          for (int d=0;d<nv;++d) mom[d]+=Jt[0*32+d]*wt.x+Jt[1*32+d]*wt.y+Jt[2*32+d]*wt.z;
        }
        if (has_r) {
          thread float Jr0[96], Jt0[96];
          ak_point_jac(rp,rsb,body_pos,body_quat,joint_anchor,joint_axis,body_parentid,
            body_jntadr,body_jntnum,jnt_type,jnt_dofadr,nbody,njnt,nv,bo,jo,Jt0,Jr0);
          float4 q0=float4(site_quat[(so+id0)*4],site_quat[(so+id0)*4+1],site_quat[(so+id0)*4+2],site_quat[(so+id0)*4+3]);
          float3 vec=ak_quat2vel(ak_qmul(ak_qconj(rq),q0));
          len+=dot(vec,float3(gear[6*a+3],gear[6*a+4],gear[6*a+5]));
          for (int d=0;d<nv;++d) {
            Jr[0*32+d]-=Jr0[0*32+d]; Jr[1*32+d]-=Jr0[1*32+d]; Jr[2*32+d]-=Jr0[2*32+d];
          }
          int da=common;
          int g3_=0;
          while (da>=0&&g3_<64) {
            Jr[0*32+da]=0.0f; Jr[1*32+da]=0.0f; Jr[2*32+da]=0.0f;
            da=(da>=0&&da<1024)?dof_parentid[da]:-1;
            g3_++;
          }
          float3x3 rm=ak_quat2mat(rq);
          float3 wr=rm*float3(gear[6*a+3],gear[6*a+4],gear[6*a+5]);
          for (int d=0;d<nv;++d) mom[d]+=Jr[0*32+d]*wr.x+Jr[1*32+d]*wr.y+Jr[2*32+d]*wr.z;
        }
      }
    }
    float vel=0.0f;
    for (int d=0;d<nv;++d) vel+=mom[d]*qvel[vbase+uint(d)];
    out_length[ubase+uint(a)]=len;
    out_velocity[ubase+uint(a)]=vel;
    for (int d=0;d<nv;++d) out_moment[(ubase+uint(a))*uint(max(nv,1))+uint(d)]=mom[d];
  }
}

// BODY-transmission adhesion moments from same-step candidate contacts.
// Pinned source: engine/engine_core_smooth.c mj_transmission (mjTRN_BODY).
// Averages contact normal Jacobian rows over candidate contacts involving the
// body and negates. The normal row is the stored translational normal
// projection (moment arms included); stored angular rows serve torsional
// friction, not force balance. Pinned pyramid-edge averaging reduces to the
// normal row by +/- symmetry. Gap candidates are included like the pinned
// exclude==1 path. Candidate buffers must be zeroed before generation so
// unwritten slots read exact zero (their frame normal is degenerate).
kernel void body_adhesion_moment(
    device float* moment [[buffer(0)]],
    device const float* contact_frame [[buffer(1)]],
    device const float* contact_jacobian [[buffer(2)]],
    device const int* pair_geoms [[buffer(3)]],
    device const int* geom_bodyid [[buffer(4)]],
    device const int* pair_offset [[buffer(5)]],
    device const int* trntype [[buffer(6)]],
    device const int* trnid [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint world [[thread_position_in_grid]]) {
  int nc=dims[0], npairs=dims[1], nv=dims[2], nu=dims[3], batch=dims[4];
  if (uint(world)>=uint(batch)) return;
  uint ubase=uint(world)*uint(max(nu,1));
  for (int a=0;a<nu;++a) {
    if (trntype[a]!=5) continue;
    int id=trnid[2*a];
    float row[32];
    for (int d=0;d<32;++d) row[d]=0.0f;
    int counter=0;
    for (int s=0;s<nc;++s) {
      int p=-1;
      for (int q=0;q<npairs;++q) {
        if (s>=pair_offset[q]&&s<pair_offset[q+1]) { p=q; break; }
      }
      if (p<0) continue;
      int ba=geom_bodyid[pair_geoms[2*p]], bb=geom_bodyid[pair_geoms[2*p+1]];
      if (ba!=id&&bb!=id) continue;
      uint fb=(uint(world)*uint(max(nc,1))+uint(s))*12u;
      float3 n=float3(contact_frame[fb],contact_frame[fb+1],contact_frame[fb+2]);
      if (dot(n,n)<1e-20f) continue;
      uint jb=(uint(world)*uint(max(nc,1))+uint(s))*uint(6*max(nv,1));
      for (int d=0;d<nv;++d)
        row[d]+=contact_jacobian[jb+uint(d)];
      counter++;
    }
    if (counter>0) {
      for (int d=0;d<nv;++d)
        moment[(ubase+uint(a))*uint(max(nv,1))+uint(d)]+= -row[d]/float(counter);
    }
  }
}
