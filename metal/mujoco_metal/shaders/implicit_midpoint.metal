// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline bool finite_bits(float x) {
  return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u;
}

inline float3 qrotate(float4 q, float3 v) {
  float3 u=q.yzw;
  return v+2.0f*cross(u,cross(u,v)+q.x*v);
}
inline float4 qmultiply(float4 a,float4 b) {
  float3 av=a.yzw,bv=b.yzw;
  return float4(a.x*b.x-dot(av,bv),a.x*bv+b.x*av+cross(av,bv));
}
inline float3 midpoint_residual(float3 inertia,float3 wmid,float3 w,float3 tau,float i2h) {
  return i2h*inertia*(wmid-w)+cross(wmid,inertia*wmid)-tau;
}

inline bool solve3(thread float* a, float3 rhs, thread float3& out) {
  a[3]=rhs.x; a[7]=rhs.y; a[11]=rhs.z;
  for(int col=0;col<3;++col) {
    int pivot=col;
    float best=abs(a[col*4+col]);
    for(int row=col+1;row<3;++row) {
      float candidate=abs(a[row*4+col]);
      if(candidate>best) { best=candidate; pivot=row; }
    }
    if(!(best>1e-20f) || !finite_bits(best)) return false;
    if(pivot!=col) for(int k=col;k<4;++k) {
      float temp=a[col*4+k]; a[col*4+k]=a[pivot*4+k]; a[pivot*4+k]=temp;
    }
    float diagonal=a[col*4+col];
    for(int k=col;k<4;++k) a[col*4+k]/=diagonal;
    for(int row=0;row<3;++row) if(row!=col) {
      float factor=a[row*4+col];
      for(int k=col;k<4;++k) a[row*4+k]-=factor*a[col*4+k];
    }
  }
  out=float3(a[3],a[7],a[11]);
  return finite_bits(out.x) && finite_bits(out.y) && finite_bits(out.z);
}

inline bool midpoint_newton(float3 inertia,float3 w,float3 tau,float h,
                            thread float3& wmid) {
  float i2h=2.0f/h;
  float3 dI=float3(inertia.z-inertia.y,inertia.x-inertia.z,inertia.y-inertia.x);
  wmid=w;
  for(int iteration=0;iteration<100;++iteration) {
    float3 Iw=inertia*wmid;
    float3 f=midpoint_residual(inertia,wmid,w,tau,i2h);
    float fnorm=length(f);
    if(!finite_bits(fnorm)) return false;
    if(fnorm<1e-6f*(1.0f+i2h*length(Iw))) return true;
    thread float jac[12];
    jac[0]=i2h*inertia.x; jac[1]=wmid.z*dI.x; jac[2]=wmid.y*dI.x;
    jac[4]=wmid.z*dI.y; jac[5]=i2h*inertia.y; jac[6]=wmid.x*dI.y;
    jac[8]=wmid.y*dI.z; jac[9]=wmid.x*dI.z; jac[10]=i2h*inertia.z;
    float3 delta;
    if(!solve3(jac,-f,delta)) return false;
    float step=1.0f;
    for(int line=0;line<20;++line) {
      float3 candidate=wmid+step*delta;
      float candidate_norm=length(midpoint_residual(inertia,candidate,w,tau,i2h));
      if(!finite_bits(candidate_norm)) return false;
      if(candidate_norm<fnorm) { wmid=candidate; break; }
      step*=0.5f;
    }
  }
  return false;
}

// Per world: fallback is ordinary semi-implicit implicitfast. Eligible
// standalone free bodies overwrite selected components using MuJoCo 3.10's
// inertial-frame Newton midpoint rotation and analytic COM translation.
kernel void free_body_midpoint(
    device const float* qvel [[buffer(0)]],
    device const float* effective_acceleration [[buffer(1)]],
    device const float* physical_qacc [[buffer(2)]],
    device const float* qfrc_total [[buffer(3)]],
    device const float* body_quat [[buffer(4)]],
    device const int* dofadr [[buffer(5)]], device const int* bodyid [[buffer(6)]],
    device const float* body_mass [[buffer(7)]], device const float* body_inertia [[buffer(8)]],
    device const float* body_ipos [[buffer(9)]], device const float* body_iquat [[buffer(10)]],
    device const uchar* aligned [[buffer(11)]], device const float* gravity [[buffer(12)]],
    device float* qvel_next [[buffer(13)]], device float* position_velocity [[buffer(14)]],
    device float* reported_qacc [[buffer(15)]], device int* status [[buffer(16)]],
    constant int* dims [[buffer(17)]], constant float* timestep [[buffer(18)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0],nbody=dims[1],nfree=dims[2],batch=dims[3];
  bool gravity_enabled=dims[4]!=0;
  if(world>=uint(batch)) return;
  int vb=int(world)*nv, qb=int(world)*nbody*4;
  status[world]=0;
  for(int dof=0;dof<nv;++dof) {
    float old=qvel[vb+dof];
    if(!finite_bits(old) || !finite_bits(effective_acceleration[vb+dof]) ||
       !finite_bits(physical_qacc[vb+dof]) || !finite_bits(qfrc_total[vb+dof])) status[world]=21;
    qvel_next[vb+dof]=old+timestep[0]*effective_acceleration[vb+dof];
    position_velocity[vb+dof]=qvel_next[vb+dof];
    reported_qacc[vb+dof]=physical_qacc[vb+dof];
    if(!finite_bits(qvel_next[vb+dof])) status[world]=21;
  }
  for(int k=0;k<nbody*4;++k) if(!finite_bits(body_quat[qb+k])) status[world]=21;
  for(int slot=0;slot<nfree;++slot) {
    int dof=dofadr[slot],body=bodyid[slot],offset=vb+dof;
    int body_offset=qb+body*4;
    float4 iquat=float4(body_iquat[slot*4],body_iquat[slot*4+1],body_iquat[slot*4+2],body_iquat[slot*4+3]);
    float4 iquat_neg=float4(iquat.x,-iquat.yzw);
    float4 xquat=float4(body_quat[body_offset],body_quat[body_offset+1],body_quat[body_offset+2],body_quat[body_offset+3]);
    float3 v_old=float3(qvel[offset],qvel[offset+1],qvel[offset+2]);
    float3 w_body=float3(qvel[offset+3],qvel[offset+4],qvel[offset+5]);
    float3 f_world=float3(qfrc_total[offset],qfrc_total[offset+1],qfrc_total[offset+2]);
    float3 tau_body=float3(qfrc_total[offset+3],qfrc_total[offset+4],qfrc_total[offset+5]);
    float3 w=qrotate(iquat_neg,w_body),tau=qrotate(iquat_neg,tau_body);
    bool is_aligned=aligned[slot]!=0;
    float4 rot_x2i=float4(1,0,0,0);
    float3 force=float3(0),r_com=float3(0),tau_com=tau;
    if(!is_aligned) {
      float4 xquat_neg=float4(xquat.x,-xquat.yzw);
      rot_x2i=qmultiply(iquat_neg,xquat_neg);
      force=qrotate(rot_x2i,f_world);
      r_com=qrotate(iquat_neg,float3(body_ipos[slot*3],body_ipos[slot*3+1],body_ipos[slot*3+2]));
      tau_com=tau-cross(r_com,force);
    }
    float3 inertia=float3(body_inertia[slot*3],body_inertia[slot*3+1],body_inertia[slot*3+2]);
    float3 wmid;
    if(!midpoint_newton(inertia,w,tau_com,timestep[0],wmid)) {
      status[world]=20;
      break;
    }
    float3 wnew=2.0f*wmid-w;
    float3 wnew_body=qrotate(iquat,wnew),wmid_body=qrotate(iquat,wmid);
    if(!is_aligned) {
      float3 v=qrotate(rot_x2i,v_old);
      float3 vcom=v+cross(w,r_com);
      float i2h=2.0f/timestep[0];
      float3 b=force/body_mass[slot]+i2h*vcom;
      if(gravity_enabled) b+=qrotate(rot_x2i,float3(gravity[0],gravity[1],gravity[2]));
      float norm2=dot(wmid,wmid);
      float denom=i2h*i2h+norm2;
      float3 vcommid=(i2h*b+(dot(wmid,b)/i2h)*wmid-cross(wmid,b))/denom;
      float3 vmid=vcommid-cross(wmid,r_com);
      float3 vnew=2.0f*vmid-v;
      float wnorm=length(wmid_body);
      float3 axis=wnorm>0 ? wmid_body/wnorm : float3(0);
      float angle=timestep[0]*wnorm;
      float4 qrot=float4(cos(0.5f*angle),axis*sin(0.5f*angle));
      float4 xquat_new=qmultiply(xquat,qrot);
      float3 vbody=qrotate(iquat,vnew);
      float3 vnew_world=qrotate(xquat_new,vbody);
      qvel_next[offset]=vnew_world.x;qvel_next[offset+1]=vnew_world.y;qvel_next[offset+2]=vnew_world.z;
      position_velocity[offset]=0.5f*(v_old.x+vnew_world.x);
      position_velocity[offset+1]=0.5f*(v_old.y+vnew_world.y);
      position_velocity[offset+2]=0.5f*(v_old.z+vnew_world.z);
    }
    float3 mid_angular=0.5f*(w_body+wnew_body);
    float3 angular_acc=(wnew_body-w_body)/timestep[0];
    qvel_next[offset+3]=wnew_body.x;qvel_next[offset+4]=wnew_body.y;qvel_next[offset+5]=wnew_body.z;
    position_velocity[offset+3]=mid_angular.x;position_velocity[offset+4]=mid_angular.y;position_velocity[offset+5]=mid_angular.z;
    reported_qacc[offset+3]=angular_acc.x;reported_qacc[offset+4]=angular_acc.y;reported_qacc[offset+5]=angular_acc.z;
    if(!finite_bits(wnew_body.x) || !finite_bits(wnew_body.y) || !finite_bits(wnew_body.z) ||
       !finite_bits(mid_angular.x) || !finite_bits(mid_angular.y) || !finite_bits(mid_angular.z) ||
       !finite_bits(angular_acc.x) || !finite_bits(angular_acc.y) || !finite_bits(angular_acc.z)) status[world]=21;
    if(!is_aligned) {
      float3 old_v=v_old;
      float3 new_v=float3(qvel_next[offset],qvel_next[offset+1],qvel_next[offset+2]);
      float3 acc=(new_v-old_v)/timestep[0];
      if(!finite_bits(new_v.x) || !finite_bits(new_v.y) || !finite_bits(new_v.z) ||
         !finite_bits(acc.x) || !finite_bits(acc.y) || !finite_bits(acc.z)) status[world]=21;
      reported_qacc[offset]=acc.x;reported_qacc[offset+1]=acc.y;reported_qacc[offset+2]=acc.z;
    }
  }
  if(status[world]!=0) {
    for(int dof=0;dof<nv;++dof) {
      qvel_next[vb+dof]=qvel[vb+dof];
      position_velocity[vb+dof]=qvel[vb+dof];
      reported_qacc[vb+dof]=physical_qacc[vb+dof];
    }
  }
}
