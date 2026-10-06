// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

kernel void fixed_tendon_dynamics(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const float* length_map [[buffer(2)]],
    device const float* moment_map [[buffer(3)]],
    device const float* stiffness [[buffer(4)]],
    device const float* stiffnesspoly [[buffer(5)]],
    device const float* damping [[buffer(6)]],
    device const float* dampingpoly [[buffer(7)]],
    device const float* spring_range [[buffer(8)]],
    device const float* armature [[buffer(9)]],
    constant int* dims [[buffer(10)]],
    device float* qfrc [[buffer(11)]],
    device float* damping_matrix [[buffer(12)]],
    device float* armature_matrix [[buffer(13)]],
    device const float* ancestor_mask [[buffer(14)]],
    device const float* cached_length [[buffer(15)]],
    device float* out_length [[buffer(16)]],
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nv=dims[1], ntendon=dims[2], batch=dims[3];
  int spring_disabled=dims[4], damper_disabled=dims[5];
  bool write_armature=dims[6] != 0;
  if (world>=uint(batch)) return;
  if (dims[8 + int(world)] == 0) return;
  uint qbase=world*uint(nq), vbase=world*uint(nv);
  uint mbase=world*uint(nv*nv);
  for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=0.0f;
  for (int i=0;i<nv*nv;++i) {
    damping_matrix[mbase+uint(i)]=0.0f;
    if (write_armature) armature_matrix[mbase+uint(i)]=0.0f;
  }
  for (int i=0;i<nq;++i) {
    if ((as_type<uint>(qpos[qbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) {
        damping_matrix[mbase+uint(k)]=bad;
        if (write_armature) armature_matrix[mbase+uint(k)]=bad;
      }
      return;
    }
  }
  for (int i=0;i<nv;++i) {
    if ((as_type<uint>(qvel[vbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) {
        damping_matrix[mbase+uint(k)]=bad;
        if (write_armature) armature_matrix[mbase+uint(k)]=bad;
      }
      return;
    }
  }
  for (int t=0;t<ntendon;++t) {
    float length=0.0f, velocity=0.0f;
    if (dims[7]) length=cached_length[world*uint(max(ntendon,1))+uint(t)];
    else for (int q=0;q<nq;++q) length+=length_map[t*nq+q]*qpos[qbase+uint(q)];
    out_length[world*uint(max(ntendon,1))+uint(t)]=length;
    for (int d=0;d<nv;++d) velocity+=moment_map[t*nv+d]*qvel[vbase+uint(d)];
    float spring_force=0.0f;
    if (!spring_disabled) {
      float lower=spring_range[2*t], upper=spring_range[2*t+1];
      float x=length>upper ? length-upper : (length<lower ? length-lower : 0.0f);
      float k=stiffness[t]+stiffnesspoly[2*t]*x+stiffnesspoly[2*t+1]*x*x;
      spring_force=-x*k;
    }
    float damper_force=0.0f, damper_tangent=0.0f;
    if (!damper_disabled) {
      float av=abs(velocity);
      float c=damping[t]+dampingpoly[2*t]*av+dampingpoly[2*t+1]*velocity*velocity;
      damper_force=-velocity*c;
      damper_tangent=damping[t]+2.0f*dampingpoly[2*t]*av+3.0f*dampingpoly[2*t+1]*velocity*velocity;
    }
    float total=spring_force+damper_force;
    for (int i=0;i<nv;++i) {
      float ji=moment_map[t*nv+i];
      qfrc[vbase+uint(i)]+=ji*total;
      for (int j=0;j<nv;++j) {
        float jj=moment_map[t*nv+j];
        uint index=mbase+uint(i*nv+j);
        damping_matrix[index]+=damper_tangent*ji*jj;
        if (write_armature)
          armature_matrix[index]+=armature[t]*ji*jj*ancestor_mask[i*nv+j];
      }
    }
  }
  for (int i=0;i<nv;++i) {
    if ((as_type<uint>(qfrc[vbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) {
        damping_matrix[mbase+uint(k)]=bad;
        if (write_armature) armature_matrix[mbase+uint(k)]=bad;
      }
      return;
    }
  }
  for (int i=0;i<nv*nv;++i) {
    if ((as_type<uint>(damping_matrix[mbase+uint(i)]) & 0x7f800000u)==0x7f800000u ||
        (write_armature &&
         (as_type<uint>(armature_matrix[mbase+uint(i)]) & 0x7f800000u)==0x7f800000u)) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) {
        damping_matrix[mbase+uint(k)]=bad;
        if (write_armature) armature_matrix[mbase+uint(k)]=bad;
      }
      return;
    }
  }
}

// Direct sparse qDeriv contribution from fixed-tendon damping. The compiled
// D-CSR edge list already restricts the source to MuJoCo's retained pattern.
kernel void fixed_tendon_damping_coo(
    device const float* qvel [[buffer(0)]],
    device const float* moment_map [[buffer(1)]],
    device const float* damping [[buffer(2)]],
    device const float* dampingpoly [[buffer(3)]],
    device const int* edge_rows [[buffer(4)]],
    device const int* edge_cols [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device float* edge_values [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0], ntendon=dims[1], batch=dims[2];
  int edge_count=dims[3], damper_disabled=dims[4];
  int stride=max(edge_count,1);
  if (damper_disabled || edge_count<=0 || int(tid)>=batch*stride) return;
  int world=int(tid)/stride, edge=int(tid)-world*stride;
  int row=edge_rows[edge], col=edge_cols[edge];
  if (row<0 || row>=nv || col<0 || col>=nv) return;
  float value=0.0f;
  for (int t=0;t<ntendon;++t) {
    float velocity=0.0f;
    for (int d=0;d<nv;++d)
      velocity+=moment_map[t*nv+d]*qvel[world*nv+d];
    float speed=abs(velocity);
    float tangent=damping[t]+2.0f*dampingpoly[2*t]*speed+
                  3.0f*dampingpoly[2*t+1]*velocity*velocity;
    value-=tangent*moment_map[t*nv+row]*moment_map[t*nv+col];
  }
  edge_values[tid]+=value;
}

// Spatial tendon kinematics for MuJoCo 3.10.0 (milestone 008).
// Pinned source: engine/engine_core_smooth.c mj_tendon (spatial loop) with
// engine/engine_util_misc.c mju_wrap/wrap_circle/wrap_inside. One thread per
// world; loops tendons and path wraps. Point Jacobians via parent traversal
// (same convention as the actuator kinematics kernel). Segment Jacobian is
// (J(pb)-J(pa)) projected on the unit segment direction, divided by the
// running pulley divisor; wrap middle segments share the geom body and are
// skipped like the pinned code. Fixed (joint-led) tendons output zero rows
// here; the fixed stage owns them.

inline float3 st8_segdir(float3 a, float3 b) {
  float3 d = b - a;
  float n = length(d);
  return n > 1e-30f ? d / n : float3(0.0f);
}
inline float3 st8_point_jac_dof(float3 point, int body, int target_dof,
    device const float* body_pos, device const float* body_quat,
    device const float* anchors, device const float* axes,
    device const int* body_parentid, device const int* body_jntadr,
    device const int* body_jntnum, device const int* jnt_type,
    device const int* jnt_dofadr,
    int nbody, int njnt, int nv, int bo, int jo) {
  if (nv==0 || target_dof<0 || target_dof>=nv) return float3(0.0f);
  float3 result=float3(0.0f);
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
        float3 col=float3(0.0f);
        if (typ==2) {
          col=float3(axes[(jo+j)*3],axes[(jo+j)*3+1],axes[(jo+j)*3+2]);
        } else if (typ==3) {
          float3 ax=float3(axes[(jo+j)*3],axes[(jo+j)*3+1],axes[(jo+j)*3+2]);
          float3 an=float3(anchors[(jo+j)*3],anchors[(jo+j)*3+1],anchors[(jo+j)*3+2]);
          col=cross(ax,point-an);
        } else {
          if (typ==0 && q<3) { col=float3(q==0,q==1,q==2); }
          else {
            int qr = typ==0?q-3:q;
            float4 bq=float4(body_quat[(bo+b)*4],body_quat[(bo+b)*4+1],body_quat[(bo+b)*4+2],body_quat[(bo+b)*4+3]);
            float nq=length(bq); bq=nq>1e-30f?bq/nq:float4(1,0,0,0);
            float3 unit=float3(qr==0,qr==1,qr==2);
            float3 ax=unit+2.0f*cross(bq.yzw,cross(bq.yzw,unit)+bq.x*unit);
            float3 piv = typ==0
              ? float3(body_pos[(bo+b)*3],body_pos[(bo+b)*3+1],body_pos[(bo+b)*3+2])
              : float3(anchors[(jo+j)*3],anchors[(jo+j)*3+1],anchors[(jo+j)*3+2]);
            col=cross(ax,point-piv);
          }
        }
        if (dof==target_dof) result+=col;
      }
    }
    b=body_parentid[b];
  }
  return result;
}

// 2D circle wrap port (pinned wrap_circle). Returns arc length or -1.
inline float st8_wrap_circle(thread float* pnt, float e0, float e1, float e2, float e3,
    bool has_side, float s0, float s1, float radius) {
  float sqlen0=e0*e0+e1*e1, sqlen1=e2*e2+e3*e3, sqrad=radius*radius;
  if (sqlen0<sqrad||sqlen1<sqrad||radius<1e-15f) return -1.0f;
  float dx=e2-e0, dy=e3-e1, dd=dx*dx+dy*dy;
  if (dd<1e-15f) return -1.0f;
  float a=-(dx*e0+dy*e1)/dd;
  a=a<0.0f?0.0f:(a>1.0f?1.0f:a);
  float tx=a*dx+e0, ty=a*dy+e1;
  if (tx*tx+ty*ty>sqrad && (!has_side || s0*tx+s1*ty>=0.0f)) return -1.0f;
  float sqrt0=sqrt(max(0.0f,sqlen0-sqrad)), sqrt1=sqrt(max(0.0f,sqlen1-sqrad));
  float s00x,s00y,s01x,s01y,s10x,s10y,s11x,s11y,g0,g1;
  s00x=(e0*sqrad+radius*e1*sqrt0)/sqlen0; s00y=(e1*sqrad-radius*e0*sqrt0)/sqlen0;
  s01x=(e2*sqrad-radius*e3*sqrt1)/sqlen1; s01y=(e3*sqrad+radius*e2*sqrt1)/sqlen1;
  if (has_side) {
    float nx=s00x+s01x, ny=s00y+s01y, n=sqrt(nx*nx+ny*ny);
    g0 = n>1e-15f ? (nx/n)*s0+(ny/n)*s1 : -10000.0f;
  } else { float qx=s00x-s01x, qy=s00y-s01y; g0=-(qx*qx+qy*qy); }
  {
    float ix0=e0,iy0=e1,ix1=s00x,iy1=s00y,jx0=e2,jy0=e3,jx1=s01x,jy1=s01y;
    float dd2=(ix1-ix0)*(jy1-jy0)-(iy1-iy0)*(jx1-jx0);
    if (abs(dd2)>1e-15f) {
      float aa=((jx1-jx0)*(iy0-jy0)-(jy1-jy0)*(ix0-jx0))/dd2;
      float bb=((ix1-ix0)*(iy0-jy0)-(iy1-iy0)*(ix0-jx0))/dd2;
      if (aa>0.0f&&aa<1.0f&&bb>0.0f&&bb<1.0f) g0=-10000.0f;
    }
  }
  s10x=(e0*sqrad-radius*e1*sqrt0)/sqlen0; s10y=(e1*sqrad+radius*e0*sqrt0)/sqlen0;
  s11x=(e2*sqrad+radius*e3*sqrt1)/sqlen1; s11y=(e3*sqrad-radius*e2*sqrt1)/sqlen1;
  if (has_side) {
    float nx=s10x+s11x, ny=s10y+s11y, n=sqrt(nx*nx+ny*ny);
    g1 = n>1e-15f ? (nx/n)*s0+(ny/n)*s1 : -10000.0f;
  } else { float qx=s10x-s11x, qy=s10y-s11y; g1=-(qx*qx+qy*qy); }
  {
    float ix0=e0,iy0=e1,ix1=s10x,iy1=s10y,jx0=e2,jy0=e3,jx1=s11x,jy1=s11y;
    float dd2=(ix1-ix0)*(jy1-jy0)-(iy1-iy0)*(jx1-jx0);
    if (abs(dd2)>1e-15f) {
      float aa=((jx1-jx0)*(iy0-jy0)-(jy1-jy0)*(ix0-jx0))/dd2;
      float bb=((ix1-ix0)*(iy0-jy0)-(iy1-iy0)*(ix0-jx0))/dd2;
      if (aa>0.0f&&aa<1.0f&&bb>0.0f&&bb<1.0f) g1=-10000.0f;
    }
  }
  int i = g0>g1?0:1;
  float q0x=i==0?s00x:s10x, q0y=i==0?s00y:s10y, q1x=i==0?s01x:s11x, q1y=i==0?s01y:s11y;
  {
    float ix0=e0,iy0=e1,ix1=q0x,iy1=q0y,jx0=e2,jy0=e3,jx1=q1x,jy1=q1y;
    float dd2=(ix1-ix0)*(jy1-jy0)-(iy1-iy0)*(jx1-jx0);
    if (abs(dd2)>1e-15f) {
      float aa=((jx1-jx0)*(iy0-jy0)-(jy1-jy0)*(ix0-jx0))/dd2;
      float bb=((ix1-ix0)*(iy0-jy0)-(iy1-iy0)*(ix0-jx0))/dd2;
      if (aa>0.0f&&aa<1.0f&&bb>0.0f&&bb<1.0f) return -1.0f;
    }
  }
  pnt[0]=q0x; pnt[1]=q0y; pnt[2]=q1x; pnt[3]=q1y;
  float l0=sqrt(q0x*q0x+q0y*q0y), l1=sqrt(q1x*q1x+q1y*q1y);
  float a0x=q0x/max(1e-15f,l0), a0y=q0y/max(1e-15f,l0);
  float a1x=q1x/max(1e-15f,l1), a1y=q1y/max(1e-15f,l1);
  float dot=max(-1.0f,min(1.0f,a0x*a1x+a0y*a1y));
  float ang=acos(dot);
  float cross=q0y*q1x-q0x*q1y;
  if ((cross>0.0f&&i==1)||(cross<0.0f&&i==0)) ang=2.0f*3.14159265358979f-ang;
  return radius*ang;
}

// Newton inside-wrap port (pinned wrap_inside). Returns arc length (0 on
// success path) or -1; pnt holds the contact point (doubled).
inline float st8_wrap_inside(thread float* pnt, float e0, float e1, float e2, float e3,
    float radius) {
  float len0=sqrt(e0*e0+e1*e1), len1=sqrt(e2*e2+e3*e3);
  float dx=e2-e0, dy=e3-e1, dd=dx*dx+dy*dy;
  if (len0<=radius||len1<=radius||radius<1e-15f||len0<1e-15f||len1<1e-15f) return -1.0f;
  if (dd>1e-15f) {
    float a=-(dx*e0+dy*e1)/dd;
    if (a>0.0f&&a<1.0f) {
      float tx=e0+a*dx, ty=e1+a*dy;
      if (sqrt(tx*tx+ty*ty)<=radius) return -1.0f;
    }
  }
  float mx=0.5f*(e0+e2), my=0.5f*(e1+e3);
  float n=sqrt(mx*mx+my*my);
  pnt[0]=mx/n*radius; pnt[1]=my/n*radius; pnt[2]=pnt[0]; pnt[3]=pnt[1];
  float A=radius/len0, B=radius/len1;
  float cosG=(len0*len0+len1*len1-dd)/(2.0f*len0*len1);
  if (cosG<-1.0f+1e-15f) return -1.0f;
  if (cosG>1.0f-1e-15f) return 0.0f;
  float G=acos(max(-1.0f,min(1.0f,cosG)));
  float z=1.0f-1e-7f;
  float f=asin(A*z)+asin(B*z)-2.0f*asin(z)+G;
  if (f>0.0f) return 0.0f;
  for (int it=0;it<20;++it) {
    if (abs(f)<=1e-6f) break;
    float df=A/max(1e-15f,sqrt(max(0.0f,1.0f-z*z*A*A)))
            +B/max(1e-15f,sqrt(max(0.0f,1.0f-z*z*B*B)))
            -2.0f/max(1e-15f,sqrt(max(0.0f,1.0f-z*z)));
    if (df>-1e-15f) return 0.0f;
    float z1=z-f/df;
    if (z1>z) return 0.0f;
    z=z1;
    f=asin(A*z)+asin(B*z)-2.0f*asin(z)+G;
    if (f>1e-6f) return 0.0f;
    if (it==19) return 0.0f;
  }
  float vx, vy, ang;
  if (e0*e3-e1*e2>0.0f) { vx=e0; vy=e1; ang=asin(z)-asin(A*z); }
  else { vx=e2; vy=e3; ang=asin(z)-asin(B*z); }
  float vn=sqrt(vx*vx+vy*vy);
  vx/=vn; vy/=vn;
  pnt[0]=radius*(cos(ang)*vx-sin(ang)*vy);
  pnt[1]=radius*(sin(ang)*vx+cos(ang)*vy);
  pnt[2]=pnt[0]; pnt[3]=pnt[1];
  return 0.0f;
}

inline float3x3 st8_quat2mat(float4 q) {
  float w=q.x, x=q.y, y=q.z, z=q.w;
  return float3x3(
    1-2*(y*y+z*z), 2*(x*y+z*w),   2*(x*z-y*w),
    2*(x*y-z*w),   1-2*(x*x+z*z), 2*(y*z+x*w),
    2*(x*z+y*w),   2*(y*z-x*w),   1-2*(x*x+y*y));
}

// Full wrap port (pinned mju_wrap). is_sphere: 1 sphere, 0 cylinder.
// Returns wrap length or -1; wp0/wp1 hold global wrap points.
inline float st8_wrap(thread float* wp, float3 x0, float3 x1,
    float3 xpos, float3x3 xmat, float radius, int is_sphere,
    bool has_side, float3 side) {
  float3x3 xt=transpose(xmat);
  float3 p0=xt*(x0-xpos), p1=xt*(x1-xpos);
  if (length(p0)<1e-15f||length(p1)<1e-15f) return -1.0f;
  float3 ax0, ax1;
  if (is_sphere) {
    ax0=p0/length(p0);
    float3 normal=cross(p0,p1);
    float nrm=length(normal);
    if (nrm<1e-15f) {
      float ax=abs(ax0.x), ay=abs(ax0.y), az=abs(ax0.z);
      int ii = (ay>ax&&ay>az)?1:((az>ax&&az>ay)?2:0);
      float3 b=float3(1.0f,1.0f,1.0f);
      if (ii==0) b.x=0.0f; else if (ii==1) b.y=0.0f; else b.z=0.0f;
      normal=cross(ax0,b); normal/=length(normal);
    } else normal/=nrm;
    ax1=cross(normal,ax0); ax1/=length(ax1);
  } else { ax0=float3(1,0,0); ax1=float3(0,1,0); }
  float e0=dot(p0,ax0), e1=dot(p0,ax1), e2=dot(p1,ax0), e3=dot(p1,ax1);
  float sd0=0.0f, sd1=0.0f;
  float3 sv=float3(0.0f);
  if (has_side) {
    sv=xt*(side-xpos);
    float q0=dot(sv,ax0), q1=dot(sv,ax1);
    float n=sqrt(q0*q0+q1*q1);
    sd0=q0/max(1e-30f,n)*radius; sd1=q1/max(1e-30f,n)*radius;
  }
  float pnt[4];
  float wlen;
  if (has_side && length(sv)<radius) {
    wlen=st8_wrap_inside(pnt,e0,e1,e2,e3,radius);
  } else {
    wlen=st8_wrap_circle(pnt,e0,e1,e2,e3,has_side,sd0,sd1,radius);
  }
  if (wlen<0.0f) return -1.0f;
  float3 r0=ax0*pnt[0]+ax1*pnt[1], r1=ax0*pnt[2]+ax1*pnt[3];
  if (!is_sphere) {
    float L0=sqrt((p0.x-r0.x)*(p0.x-r0.x)+(p0.y-r0.y)*(p0.y-r0.y));
    float L1=sqrt((p1.x-r1.x)*(p1.x-r1.x)+(p1.y-r1.y)*(p1.y-r1.y));
    r0.z=p0.z+(p1.z-p0.z)*L0/(L0+wlen+L1);
    r1.z=p0.z+(p1.z-p0.z)*(L0+wlen)/(L0+wlen+L1);
    float h=abs(r1.z-r0.z);
    wlen=sqrt(wlen*wlen+h*h);
  }
  float3 wp0=xmat*r0+xpos, wp1=xmat*r1+xpos;
  wp[0]=wp0.x; wp[1]=wp0.y; wp[2]=wp0.z; wp[3]=wp1.x; wp[4]=wp1.y; wp[5]=wp1.z;
  return wlen;
}

kernel void spatial_tendon_kinematics(
    device const float* qvel [[buffer(0)]],
    device const float* site_pos [[buffer(1)]],
    device const float* geom_pos [[buffer(2)]],
    device const float* geom_quat [[buffer(3)]],
    device const float* geom_size [[buffer(4)]],
    device const float* body_pos [[buffer(5)]],
    device const float* body_quat [[buffer(6)]],
    device const float* joint_anchor [[buffer(7)]],
    device const float* joint_axis [[buffer(8)]],
    device const int* path_types [[buffer(9)]],
    device const int* path_objids [[buffer(10)]],
    device const float* path_prms [[buffer(11)]],
    device const int* path_offset [[buffer(12)]],
    device const int* path_count [[buffer(13)]],
    device const int* geom_type [[buffer(14)]],
    device const int* geom_bodyid [[buffer(15)]],
    device const int* site_bodyid [[buffer(16)]],
    device const int* jnt_type [[buffer(17)]],
    device const int* jnt_dofadr [[buffer(18)]],
    device const int* body_parentid [[buffer(19)]],
    device const int* body_jntadr [[buffer(20)]],
    device const int* body_jntnum [[buffer(21)]],
    constant int* dims [[buffer(22)]],
    device float* out_length [[buffer(23)]],
    device float* out_velocity [[buffer(24)]],
    device float* out_jacobian [[buffer(25)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nt=dims[1], nsite=dims[2], ngeom=dims[3];
  int nbody=dims[4], njnt=dims[5], batch=dims[7];
  if (uint(world)>=uint(batch)) return;
  if (dims[8 + int(world)] == 0) return;
  uint vbase=uint(world)*uint(max(nv,1)), tbase=uint(world)*uint(max(nt,1));
  int bo=world*nbody, jo=world*max(njnt,1), so=world*max(nsite,1), go=world*max(ngeom,1);
  for (int t=0;t<nt;++t) {
    int off=path_offset[t], count=path_count[t];
    float len=0.0f;
    device float* row=out_jacobian+(tbase+uint(t))*uint(max(nv,1));
    for (int d=0;d<nv;++d) row[d]=0.0f;
    if (count>0) {
      float divisor=1.0f;
      int j=0;
      while (j<count-1) {
        int t0=path_types[off+j], t1=path_types[off+j+1];
        int id0=path_objids[off+j], id1=path_objids[off+j+1];
        if (t0==2||t1==2) {  // mjWRAP_PULLEY
          if (t0==2) divisor=path_prms[off+j];
          j++;
          continue;
        }
        float3 p0=float3(site_pos[(so+id0)*3],site_pos[(so+id0)*3+1],site_pos[(so+id0)*3+2]);
        int b0=(id0>=0&&id0<nsite)?site_bodyid[id0]:0;
        // site-geom-site wrap: t1 is SPHERE(4)/CYLINDER(5), next is a site
        if ((t1==4||t1==5)&&j+2<count) {
          int id2=path_objids[off+j+2];
          float side_prm=path_prms[off+j+1];
          int sideid=int(round(side_prm));
          bool has_side=(sideid>=0&&sideid<nsite);
          float3 sp=has_side?float3(site_pos[(so+sideid)*3],site_pos[(so+sideid)*3+1],site_pos[(so+sideid)*3+2]):float3(0.0f);
          int gid=id1;
          float3 gp=float3(geom_pos[(go+gid)*3],geom_pos[(go+gid)*3+1],geom_pos[(go+gid)*3+2]);
          float4 gq=float4(geom_quat[(go+gid)*4],geom_quat[(go+gid)*4+1],geom_quat[(go+gid)*4+2],geom_quat[(go+gid)*4+3]);
          float ng=length(gq); gq=ng>1e-30f?gq/ng:float4(1,0,0,0);
          float3x3 gm=st8_quat2mat(gq);
          float radius=geom_size[gid*3];
          float3 p2=float3(site_pos[(so+id2)*3],site_pos[(so+id2)*3+1],site_pos[(so+id2)*3+2]);
          int b2=(id2>=0&&id2<nsite)?site_bodyid[id2]:0;
          int gb=(gid>=0&&gid<ngeom)?geom_bodyid[gid]:0;
          thread float wp[6];
          float wlen=st8_wrap(wp,p0,p2,gp,gm,radius,t1==4?1:0,has_side,sp);
          if (wlen<0.0f) {
            // no-wrap fallback: straight site-site segment (pinned)
            len+=distance(p0,p2)/divisor;
            if (b0!=b2) {
              float3 dir=st8_segdir(p0,p2);
              for (int d=0;d<nv;++d) {
                float3 J0=st8_point_jac_dof(p0,b0,d,body_pos,body_quat,joint_anchor,joint_axis,
                  body_parentid,body_jntadr,body_jntnum,jnt_type,jnt_dofadr,
                  nbody,njnt,nv,bo,jo);
                float3 J1=st8_point_jac_dof(p2,b2,d,body_pos,body_quat,joint_anchor,joint_axis,
                  body_parentid,body_jntadr,body_jntnum,jnt_type,jnt_dofadr,
                  nbody,njnt,nv,bo,jo);
                row[d]+=dot(J1-J0,dir)/divisor;
              }
            }
          } else {
            float3 wp0=float3(wp[0],wp[1],wp[2]), wp1=float3(wp[3],wp[4],wp[5]);
            len+=(distance(p0,wp0)+wlen+distance(wp1,p2))/divisor;
            // three sub-segments; middle shares the geom body (skipped)
            float3 segA[2]={p0,wp0}; int segAB[2]={b0,gb};
            float3 segC[2]={wp1,p2}; int segCB[2]={gb,b2};
            for (int s=0;s<2;++s) {
              float3 qa=s==0?segA[0]:segC[0], qb=s==0?segA[1]:segC[1];
              int ba=s==0?segAB[0]:segCB[0], bb=s==0?segAB[1]:segCB[1];
              if (ba==bb) continue;
              float3 dir=st8_segdir(qa,qb);
              for (int d=0;d<nv;++d) {
                float3 J0=st8_point_jac_dof(qa,ba,d,body_pos,body_quat,joint_anchor,joint_axis,
                  body_parentid,body_jntadr,body_jntnum,jnt_type,jnt_dofadr,
                  nbody,njnt,nv,bo,jo);
                float3 J1=st8_point_jac_dof(qb,bb,d,body_pos,body_quat,joint_anchor,joint_axis,
                  body_parentid,body_jntadr,body_jntnum,jnt_type,jnt_dofadr,
                  nbody,njnt,nv,bo,jo);
                row[d]+=dot(J1-J0,dir)/divisor;
              }
            }
          }
          j+=2;
          continue;
        }
        // straight site-site segment
        float3 p1=float3(site_pos[(so+id1)*3],site_pos[(so+id1)*3+1],site_pos[(so+id1)*3+2]);
        int b1=(id1>=0&&id1<nsite)?site_bodyid[id1]:0;
        len+=distance(p0,p1)/divisor;
        if (b0!=b1) {
          float3 dir=st8_segdir(p0,p1);
          for (int d=0;d<nv;++d) {
            float3 J0=st8_point_jac_dof(p0,b0,d,body_pos,body_quat,joint_anchor,joint_axis,
              body_parentid,body_jntadr,body_jntnum,jnt_type,jnt_dofadr,
              nbody,njnt,nv,bo,jo);
            float3 J1=st8_point_jac_dof(p1,b1,d,body_pos,body_quat,joint_anchor,joint_axis,
              body_parentid,body_jntadr,body_jntnum,jnt_type,jnt_dofadr,
              nbody,njnt,nv,bo,jo);
            row[d]+=dot(J1-J0,dir)/divisor;
          }
        }
        j+=1;
      }
    }
    out_length[tbase+uint(t)]=len;
    float vel=0.0f;
    for (int d=0;d<nv;++d) vel+=row[d]*qvel[vbase+uint(d)];
    out_velocity[tbase+uint(t)]=vel;
  }
}

// Refresh speeds from a retained POS-stage tendon Jacobian. The per-world
// mask is checked before reading qvel/J or touching the cached output, so a
// state-check recovery leaves every healthy world's sampled VEL workspace
// byte-for-byte unchanged.
kernel void spatial_tendon_velocity(
    device const float* qvel [[buffer(0)]],
    device const float* jacobian [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    device float* out_velocity [[buffer(3)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nt=dims[1], batch=dims[7];
  if (world>=uint(batch) || dims[8 + int(world)]==0) return;
  uint vbase=world*uint(max(nv,1));
  uint tbase=world*uint(max(nt,1));
  for (int t=0;t<nt;++t) {
    float value=0.0f;
    for (int d=0;d<nv;++d)
      value+=jacobian[(tbase+uint(t))*uint(max(nv,1))+uint(d)]*
              qvel[vbase+uint(d)];
    out_velocity[tbase+uint(t)]=value;
  }
}

// Recovery-only actuator state overlay. Each selected world is updated by a
// single thread; a rejected world returns before reading spatial state or
// touching the retained actuator kinematics. `row_map` packs (actuator,
// tendon) pairs and `gear` carries the corresponding tendon gear.
kernel void spatial_tendon_actuator_state(
    device const float* tendon_length [[buffer(0)]],
    device const float* tendon_velocity [[buffer(1)]],
    device const float* tendon_jacobian [[buffer(2)]],
    device const int* row_map [[buffer(3)]],
    device const float* gear [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    device float* actuator_length [[buffer(6)]],
    device float* actuator_velocity [[buffer(7)]],
    device float* actuator_moment [[buffer(8)]],
    uint world [[thread_position_in_grid]]) {
  int batch=dims[0], nu=dims[1], nv=dims[2], rows=dims[3];
  int ntendon=dims[4];
  if (int(world)>=batch || dims[5+int(world)]==0) return;
  int tbase=int(world)*max(nu,1);
  int jbase=int(world)*max(nu,1)*nv;
  int Jbase=int(world)*max(ntendon,1)*nv;
  for (int r=0; r<rows; ++r) {
    int actuator=row_map[2*r];
    int tendon=row_map[2*r+1];
    float scale=gear[r];
    actuator_length[tbase+actuator]=scale*tendon_length[
        int(world)*max(ntendon,1)+tendon];
    actuator_velocity[tbase+actuator]=scale*tendon_velocity[
        int(world)*max(ntendon,1)+tendon];
    int moment_base=jbase+actuator*nv;
    int jac_base=Jbase+tendon*nv;
    for (int d=0; d<nv; ++d)
      actuator_moment[moment_base+d] += scale*tendon_jacobian[jac_base+d];
  }
}

kernel void spatial_tendon_forces(
    device const float* length [[buffer(0)]],
    device const float* velocity [[buffer(1)]],
    device const float* jacobian [[buffer(2)]],
    device const float* stiffness [[buffer(3)]],
    device const float* stiffnesspoly [[buffer(4)]],
    device const float* damping [[buffer(5)]],
    device const float* dampingpoly [[buffer(6)]],
    device const float* spring_range [[buffer(7)]],
    device const float* armature [[buffer(8)]],
    device const float* ancestor [[buffer(9)]],
    constant int* dims [[buffer(10)]],
    device float* qfrc [[buffer(11)]],
    device float* damping_matrix [[buffer(12)]],
    device float* armature_matrix [[buffer(13)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nt=dims[1], batch=dims[7];
  if (world>=uint(batch) || dims[8 + int(world)]==0) return;
  bool write_armature=dims[8+batch]!=0;
  bool spring_off=dims[9+batch]!=0;
  bool damper_off=dims[10+batch]!=0;
  uint vb=world*uint(max(nv,1)), tb=world*uint(max(nt,1));
  uint mb=world*uint(max(nv*nv,1));
  for (int d=0;d<nv;++d) qfrc[vb+uint(d)]=0.0f;
  for (int k=0;k<nv*nv;++k) {
    damping_matrix[mb+uint(k)]=0.0f;
    if (write_armature) armature_matrix[mb+uint(k)]=0.0f;
  }
  for (int t=0;t<nt;++t) {
    float L=length[tb+uint(t)], V=velocity[tb+uint(t)];
    float lo=spring_range[2*t], hi=spring_range[2*t+1];
    float disp=L>hi?L-hi:(L<lo?L-lo:0.0f);
    float k=stiffness[t]+stiffnesspoly[2*t]*disp+
            stiffnesspoly[2*t+1]*disp*disp;
    float sf=spring_off?0.0f:-disp*k;
    float av=abs(V);
    float c=damping[t]+dampingpoly[2*t]*av+dampingpoly[2*t+1]*V*V;
    float df=damper_off?0.0f:-V*c;
    float tangent=damper_off?0.0f:
        damping[t]+2.0f*dampingpoly[2*t]*av+
        3.0f*dampingpoly[2*t+1]*V*V;
    float total=sf+df;
    for (int i=0;i<nv;++i) {
      float ji=jacobian[(tb+uint(t))*uint(max(nv,1))+uint(i)];
      qfrc[vb+uint(i)]+=total*ji;
      for (int j=0;j<nv;++j) {
        uint ij=mb+uint(i*nv+j);
        float jj=jacobian[(tb+uint(t))*uint(max(nv,1))+uint(j)];
        damping_matrix[ij]+=tangent*ji*jj;
        if (write_armature)
          armature_matrix[ij]+=armature[t]*ji*jj*
              ancestor[uint(i*max(nv,1)+j)];
      }
    }
  }
}

// Site velocity from COM-based body spatial velocity (cvel is [ang, lin]
// measured at the subtree COM, like pinned; see sensors.metal).
inline float3 st8_site_vel(float3 p, int body,
    device const float* cvel, device const float* root_com,
    device const int* body_rootid,
    int nbody, int bo) {
  int b = (body >= 0 && body < nbody) ? body : 0;
  int rc = body_rootid[b];
  rc = (rc >= 0 && rc < nbody) ? rc : 0;
  float3 w = float3(cvel[(bo+b)*6], cvel[(bo+b)*6+1], cvel[(bo+b)*6+2]);
  float3 v = float3(cvel[(bo+b)*6+3], cvel[(bo+b)*6+4], cvel[(bo+b)*6+5]);
  float3 com = float3(root_com[(bo+rc)*3], root_com[(bo+rc)*3+1], root_com[(bo+rc)*3+2]);
  return v + cross(w, p - com);
}

// Point-Jacobian-dot times qvel: exact port of pinned mj_jacDot (dense,
// translational part only; engine_core_util.c). Uses the smooth stage's
// cdof/cdof_dot/cvel plus subtree COMs, so serial same-body joints,
// ball/free quaternion joints and multi-joint bodies match by construction.
inline float3 st8_point_jacdot(float3 point, int body,
    device const float* cvel, device const float* cdof, device const float* cdof_dot,
    device const float* root_com, device const int* body_rootid,
    device const int* body_weldid, device const int* body_dofadr, device const int* body_dofnum,
    device const int* dof_parentid, device const int* dof_bodyid, device const int* dof_is_quat,
    device const float* qvel,
    int nbody, int nv, int bo, int vbase) {
  float3 A = float3(0.0f);
  if (nv == 0) return A;
  int b = (body >= 0 && body < nbody) ? body : 0;
  int rc = body_rootid[b];
  rc = (rc >= 0 && rc < nbody) ? rc : 0;
  float3 com = float3(root_com[(bo+rc)*3], root_com[(bo+rc)*3+1], root_com[(bo+rc)*3+2]);
  float3 off = point - com;
  // pvel = transformSpatial(cvel[b], point, com).
  float3 bw = float3(cvel[(bo+b)*6], cvel[(bo+b)*6+1], cvel[(bo+b)*6+2]);
  float3 bv = float3(cvel[(bo+b)*6+3], cvel[(bo+b)*6+4], cvel[(bo+b)*6+5]);
  float3 pv = bv + cross(bw, off);
  // Skip fixed bodies (weld target 0).
  int wb = body_weldid[b];
  if (wb <= 0 || wb >= nbody) return A;
  // Backward pass over the dof ancestor chain.
  int i = body_dofadr[wb] + body_dofnum[wb] - 1;
  int guard = nv + 1;
  while (i >= 0 && i < nv && guard-- > 0) {
    float3 cd = float3(cdof[(vbase+i)*6], cdof[(vbase+i)*6+1], cdof[(vbase+i)*6+2]);
    float3 cdv = float3(cdof[(vbase+i)*6+3], cdof[(vbase+i)*6+4], cdof[(vbase+i)*6+5]);
    float3 cdd = float3(cdof_dot[(vbase+i)*6], cdof_dot[(vbase+i)*6+1], cdof_dot[(vbase+i)*6+2]);
    float3 cddv = float3(cdof_dot[(vbase+i)*6+3], cdof_dot[(vbase+i)*6+4], cdof_dot[(vbase+i)*6+5]);
    if (dof_is_quat[i]) {
      // Quaternion joints use the joint body's current velocity.
      int db = dof_bodyid[i];
      db = (db >= 0 && db < nbody) ? db : 0;
      float3 jw = float3(cvel[(bo+db)*6], cvel[(bo+db)*6+1], cvel[(bo+db)*6+2]);
      float3 jv = float3(cvel[(bo+db)*6+3], cvel[(bo+db)*6+4], cvel[(bo+db)*6+5]);
      cdd = cross(jw, cd);
      cddv = cross(jw, cdv) + cross(jv, cd);
    }
    A += (cddv + cross(cdd, off) + cross(cd, pv)) * qvel[vbase+i];
    i = dof_parentid[i];
  }
  return A;
}

// Per-tendon Jdot(qvel) (tendon-space bias velocity) for site-only paths.
// Pinned source: engine_core_smooth.c mj_tendonDot, dense site-only path.
// Wrapped paths carry no armature (rejected at admission); their dots stay 0.
kernel void spatial_armature_dots(
    device const float* qvel [[buffer(0)]],
    device const float* site_pos [[buffer(1)]],
    device const float* cvel [[buffer(2)]],
    device const float* cdof [[buffer(3)]],
    device const float* cdof_dot [[buffer(4)]],
    device const float* root_com [[buffer(5)]],
    device const int* path_types [[buffer(6)]],
    device const int* path_objids [[buffer(7)]],
    device const float* path_prms [[buffer(8)]],
    device const int* path_offset [[buffer(9)]],
    device const int* path_count [[buffer(10)]],
    device const int* site_bodyid [[buffer(11)]],
    device const int* body_rootid [[buffer(12)]],
    device const int* body_weldid [[buffer(13)]],
    device const int* body_dofadr [[buffer(14)]],
    device const int* body_dofnum [[buffer(15)]],
    device const int* dof_parentid [[buffer(16)]],
    device const int* dof_bodyid [[buffer(17)]],
    device const int* dof_is_quat [[buffer(18)]],
    device const float* armature [[buffer(19)]],
    constant int* dims [[buffer(20)]],
    device float* out_dots [[buffer(21)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nt=dims[1], nsite=dims[2];
  int nbody=dims[4], batch=dims[7];
  if (uint(world)>=uint(batch) || dims[8 + int(world)]==0) return;
  uint vbase=uint(world)*uint(max(nv,1)), tbase=uint(world)*uint(max(nt,1));
  int bo=world*nbody, so=world*max(nsite,1);
  for (int t=0;t<nt;++t) {
    float tdot=0.0f;
    if (armature[t] != 0.0f) {
      int off=path_offset[t], count=path_count[t];
      bool wrapped=false;
      for (int k=0;k<count;++k) {
        int wt=path_types[off+k];
        if (wt==4||wt==5) { wrapped=true; break; }
      }
      if (!wrapped && count>0) {
        float divisor=1.0f;
        int j=0;
        while (j<count-1) {
          int t0=path_types[off+j], t1=path_types[off+j+1];
          int id0=path_objids[off+j], id1=path_objids[off+j+1];
          if (t0==2||t1==2) {  // mjWRAP_PULLEY
            if (t0==2) divisor=path_prms[off+j];
            j++;
            continue;
          }
          // site-site segment (site-only paths admitted with armature)
          if (t0==3&&t1==3) {
            float3 p0=float3(site_pos[(so+id0)*3],site_pos[(so+id0)*3+1],site_pos[(so+id0)*3+2]);
            float3 p1=float3(site_pos[(so+id1)*3],site_pos[(so+id1)*3+1],site_pos[(so+id1)*3+2]);
            int b0=(id0>=0&&id0<nsite)?site_bodyid[id0]:0;
            int b1=(id1>=0&&id1<nsite)?site_bodyid[id1]:0;
            if (b0!=b1) {
              float3 d=p1-p0;
              float n=length(d);
              if (n>1e-9f) {
                float3 dpnt=d/n;
                float3 v0=st8_site_vel(p0,b0,cvel,root_com,body_rootid,nbody,bo);
                float3 v1=st8_site_vel(p1,b1,cvel,root_com,body_rootid,nbody,bo);
                float3 dv=v1-v0;
                float s=dot(dpnt,dv);
                float3 dvel=(dv-dpnt*s)/n;
                float3 A0=st8_point_jacdot(p0,b0,cvel,cdof,cdof_dot,
                  root_com,body_rootid,body_weldid,body_dofadr,body_dofnum,
                  dof_parentid,dof_bodyid,dof_is_quat,qvel,nbody,nv,bo,vbase);
                float3 A1=st8_point_jacdot(p1,b1,cvel,cdof,cdof_dot,
                  root_com,body_rootid,body_weldid,body_dofadr,body_dofnum,
                  dof_parentid,dof_bodyid,dof_is_quat,qvel,nbody,nv,bo,vbase);
                tdot+=(dot(dpnt,A1-A0)+dot(dvel,v1-v0))/divisor;
              }
            }
          }
          j+=1;
        }
      }
    }
    out_dots[tbase+uint(t)]=tdot;
  }
}

kernel void spatial_armature_bias(
    device const float* jacobian [[buffer(0)]],
    device const float* dots [[buffer(1)]],
    device const float* armature [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    device float* out_bias [[buffer(4)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nt=dims[1], batch=dims[7];
  if (world>=uint(batch) || dims[8 + int(world)]==0) return;
  uint vb=world*uint(max(nv,1)), tb=world*uint(max(nt,1));
  for (int d=0;d<nv;++d) {
    float value=0.0f;
    for (int t=0;t<nt;++t)
      value+=armature[t]*dots[tb+uint(t)]*
          jacobian[(tb+uint(t))*uint(max(nv,1))+uint(d)];
    out_bias[vb+uint(d)]=value;
  }
}
