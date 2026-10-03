// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

static float3 rot(float4 q, float3 v) {
  float3 u=q.yzw;
  return v+2.0f*cross(u,cross(u,v)+q.x*v);
}
static float3 invrot(float4 q, float3 v) { return rot(float4(q.x,-q.yzw),v); }
static bool bad(float x) { return (as_type<uint>(x)&0x7f800000u)==0x7f800000u; }
static float3 get3(device const float* a, uint i) { return float3(a[i],a[i+1],a[i+2]); }
static float4 get4(device const float* a, uint i) { return float4(a[i],a[i+1],a[i+2],a[i+3]); }

kernel void inertia_box_fluid(
    device const int* parent [[buffer(0)]],
    device const int* body_rootid [[buffer(1)]],
    device const int* body_jntadr [[buffer(2)]],
    device const int* body_jntnum [[buffer(3)]],
    device const int* jnt_type [[buffer(4)]],
    device const int* jnt_dofadr [[buffer(5)]],
    device const float* body_mass [[buffer(6)]],
    device const float* body_inertia [[buffer(7)]],
    device const float* body_quat [[buffer(8)]],
    device const float* inertial_pos [[buffer(9)]],
    device const float* inertial_quat [[buffer(10)]],
    device const float* cvel [[buffer(11)]],
    device const float* root_com [[buffer(12)]],
    device const float* joint_anchor [[buffer(13)]],
    device const float* joint_axis [[buffer(14)]],
    device const float* fluid [[buffer(15)]],
    constant int* dims [[buffer(16)]],
    device float* qfrc [[buffer(17)]],
    device const int* body_skip [[buffer(18)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nbody=dims[1], njnt=dims[2], batch=dims[3], disabled=dims[4];
  if (world>=uint(batch)) return;
  uint forcebase=world*uint(nv), jointbase=world*uint(njnt);
  for (int d=0;d<nv;++d) qfrc[forcebase+uint(d)]=0.0f;
  if (disabled || (fluid[0]<=0.0f && fluid[1]<=0.0f)) return;
  // Reject non-finite native state before any projection.
  uint body6=world*uint(nbody*6), body3=world*uint(nbody*3), body4=world*uint(nbody*4);
  bool invalid=false;
  for (int i=0;i<nbody*6;++i) invalid|=bad(cvel[body6+uint(i)]);
  for (int i=0;i<nbody*3;++i) invalid|=bad(root_com[body3+uint(i)]) || bad(inertial_pos[body3+uint(i)]);
  for (int i=0;i<nbody*4;++i) invalid|=bad(body_quat[body4+uint(i)]) || bad(inertial_quat[body4+uint(i)]);
  for (int i=0;i<5;++i) invalid|=bad(fluid[uint(i)]);
  if (invalid) {
    for (int d=0;d<nv;++d) qfrc[forcebase+uint(d)]=as_type<float>(0x7fc00000u);
    return;
  }

  for (int b=1;b<nbody;++b) {
    if (body_skip && body_skip[b]) continue;
    float mass=body_mass[b];
    if (mass<1e-15f) continue;
    float3 inertia=get3(body_inertia,uint(b*3));
    float3 box=sqrt(max(float3(1e-15f),float3(inertia.y+inertia.z-inertia.x,
        inertia.x+inertia.z-inertia.y,inertia.x+inertia.y-inertia.z))/mass*6.0f);
    float3 omega=get3(cvel,body6+uint(b*6));
    float3 linear=get3(cvel,body6+uint(b*6+3));
    float3 pos=get3(inertial_pos,body3+uint(b*3));
    float3 center=get3(root_com,body3+uint(body_rootid[b]*3));
    linear-=cross(pos-center,omega);
    float4 inertial_rotation=get4(inertial_quat,body4+uint(b*4));
    float3 local_omega=invrot(inertial_rotation,omega);
    float3 local_linear=invrot(inertial_rotation,linear);
    // Wind is a world-space translational velocity transformed into the body frame.
    local_linear-=invrot(inertial_rotation,float3(fluid[2],fluid[3],fluid[4]));
    float3 torque=float3(0.0f), force=float3(0.0f);
    float viscosity=fluid[1], density=fluid[0];
    if (viscosity>0.0f) {
      float diameter=(box.x+box.y+box.z)/3.0f;
      torque-=M_PI_F*diameter*diameter*diameter*viscosity*local_omega;
      force-=3.0f*M_PI_F*diameter*viscosity*local_linear;
    }
    if (density>0.0f) {
      force.x-=0.5f*density*box.y*box.z*abs(local_linear.x)*local_linear.x;
      force.y-=0.5f*density*box.x*box.z*abs(local_linear.y)*local_linear.y;
      force.z-=0.5f*density*box.x*box.y*abs(local_linear.z)*local_linear.z;
      torque.x-=density*box.x*(pow(box.y,4.0f)+pow(box.z,4.0f))*abs(local_omega.x)*local_omega.x/64.0f;
      torque.y-=density*box.y*(pow(box.x,4.0f)+pow(box.z,4.0f))*abs(local_omega.y)*local_omega.y/64.0f;
      torque.z-=density*box.z*(pow(box.x,4.0f)+pow(box.y,4.0f))*abs(local_omega.z)*local_omega.z/64.0f;
    }
    float3 world_torque=rot(inertial_rotation,torque);
    float3 world_force=rot(inertial_rotation,force);
    int ancestor=b;
    while (ancestor>0) {
      int first=body_jntadr[ancestor], count=body_jntnum[ancestor];
      for (int k=0;k<count;++k) {
        int j=first+k, type=jnt_type[j], da=jnt_dofadr[j];
        uint ji=(jointbase+uint(j))*3;
        float3 anchor=get3(joint_anchor,ji), axis=get3(joint_axis,ji);
        float3 moment=cross(pos-anchor,world_force)+world_torque;
        if (type==2) qfrc[forcebase+uint(da)]+=dot(axis,world_force);
        else if (type==3) qfrc[forcebase+uint(da)]+=dot(axis,moment);
        else if (type==1) {
          float4 q=get4(body_quat,body4+uint(ancestor*4));
          for (int a=0;a<3;++a) qfrc[forcebase+uint(da+a)]+=dot(rot(q,float3(float(a==0),float(a==1),float(a==2))),moment);
        } else if (type==0) {
          for (int a=0;a<3;++a) qfrc[forcebase+uint(da+a)]+=world_force[a];
          float4 q=get4(body_quat,body4+uint(ancestor*4));
          for (int a=0;a<3;++a) qfrc[forcebase+uint(da+3+a)]+=dot(rot(q,float3(float(a==0),float(a==1),float(a==2))),moment);
        }
      }
      ancestor=parent[ancestor];
    }
  }
  for (int d=0;d<nv;++d) {
    if (bad(qfrc[forcebase+uint(d)])) {
      for (int k=0;k<nv;++k) qfrc[forcebase+uint(k)]=as_type<float>(0x7fc00000u);
      return;
    }
  }
}

// Per-geom ellipsoid fluid model (pinned mj_ellipsoidFluidModel +
// mj_addedMassForces + mj_viscousForces). Bodies using it skip the
// inertia-box kernel above (body_skip). Geom world velocity is derived from
// body cvel: v_geom = v_root + omega x (x_geom - com_root), then rotated
// into the geom frame; wind is rotation-only. Forces apply at geom_xpos.
kernel void ellipsoid_geom_fluid(
    device const int* parent [[buffer(0)]],
    device const int* body_rootid [[buffer(1)]],
    device const int* body_jntadr [[buffer(2)]],
    device const int* body_jntnum [[buffer(3)]],
    device const int* jnt_type [[buffer(4)]],
    device const int* jnt_dofadr [[buffer(5)]],
    device const float* body_quat [[buffer(6)]],
    device const float* cvel [[buffer(7)]],
    device const float* root_com [[buffer(8)]],
    device const float* joint_anchor [[buffer(9)]],
    device const float* joint_axis [[buffer(10)]],
    device const int* geom_bodyid [[buffer(11)]],
    device const int* geom_type [[buffer(12)]],
    device const float* geom_size [[buffer(13)]],
    device const float* geom_fluid [[buffer(14)]],
    device const float* geom_pos [[buffer(15)]],
    device const float* geom_quat [[buffer(16)]],
    device const float* fluid [[buffer(17)]],
    constant int* dims [[buffer(18)]],
    device float* qfrc [[buffer(19)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nbody=dims[1], njnt=dims[2], batch=dims[3], ngeom=dims[5];
  int disabled=dims[4];
  if (world>=uint(batch)) return;
  uint forcebase=world*uint(nv), jointbase=world*uint(njnt);
  if (disabled || (fluid[0]<=0.0f && fluid[1]<=0.0f)) return;
  uint body6=world*uint(nbody*6), body3=world*uint(nbody*3), body4=world*uint(nbody*4);
  uint geom3=world*uint(max(ngeom,1)*3), geom4=world*uint(max(ngeom,1)*4);
  float density=fluid[0], viscosity=fluid[1];
  float3 wind_w=float3(fluid[2],fluid[3],fluid[4]);
  for (int g=0; g<ngeom; ++g) {
    float interact=geom_fluid[g*12];
    if (interact==0.0f) continue;
    int b=geom_bodyid[g];
    if (b<=0 || b>=nbody) continue;
    float3 omega=get3(cvel,body6+uint(b*6));
    float3 vroot=get3(cvel,body6+uint(b*6+3));
    float3 xgeom=get3(geom_pos,geom3+uint(g*3));
    float3 com=get3(root_com,body3+uint(body_rootid[b]*3));
    float3 vworld=vroot+cross(omega,xgeom-com);
    float4 gq=get4(geom_quat,geom4+uint(g*4));
    float3 lang=invrot(gq,omega);
    float3 llin=invrot(gq,vworld-wind_w);
    if (bad(lang.x+lang.y+lang.z+llin.x+llin.y+llin.z)) continue;
    int gt=geom_type[g];
    float3 sz=get3(geom_size,uint(g*3));
    float3 size;
    if (gt==2) size=float3(sz.x,sz.x,sz.x);
    else if (gt==3) size=float3(sz.x,sz.x,sz.x+sz.y);
    else if (gt==5) size=float3(sz.x,sz.x,sz.y);
    else size=sz;
    float blunt=geom_fluid[g*12+1], slender=geom_fluid[g*12+2], angd=geom_fluid[g*12+3];
    float kutta=geom_fluid[g*12+4], magnus=geom_fluid[g*12+5];
    float3 vmass=get3(geom_fluid,uint(g*12+6)), vinert=get3(geom_fluid,uint(g*12+9));
    // Added-mass forces.
    float3 vlin=density*vmass*llin, vang=density*vinert*lang;
    float3 lf=cross(vlin,lang);
    float3 lt=cross(vlin,llin)+cross(vang,lang);
    // Viscous lift/drag.
    float vol=4.0f/3.0f*M_PI_F*size.x*size.y*size.z;
    lf+=magnus*density*vol*cross(lang,llin);
    float l2=dot(llin,llin);
    float denom=pow(size.y*size.z,4.0f)*llin.x*llin.x+pow(size.z*size.x,4.0f)*llin.y*llin.y+pow(size.x*size.y,4.0f)*llin.z*llin.z;
    float num=pow(size.y*size.z*llin.x,2.0f)+pow(size.z*size.x*llin.y,2.0f)+pow(size.x*size.y*llin.z,2.0f);
    float aproj = num>0.0f ? M_PI_F*sqrt(max(denom,0.0f)/max(num,1e-30f)) : 0.0f;
    float3 nrm=float3(pow(size.y*size.z,2.0f)*llin.x,pow(size.z*size.x,2.0f)*llin.y,pow(size.x*size.y,2.0f)*llin.z);
    float lnorm=length(llin);
    float cos_alpha=(denom>0.0f && lnorm>0.0f) ? num/max(1e-30f,lnorm*denom) : 0.0f;
    float3 circ=cross(nrm,llin)*kutta*density*cos_alpha*aproj;
    lf+=cross(circ,llin);
    float eq_d=2.0f/3.0f*(size.x+size.y+size.z);
    float lin_c=3.0f*M_PI_F*eq_d, tq_c=M_PI_F*eq_d*eq_d*eq_d;
    float dmax=max(size.x,max(size.y,size.z)), dmin=min(size.x,min(size.y,size.z));
    float dmid=size.x+size.y+size.z-dmax-dmin;
    float amax=M_PI_F*dmax*dmid;
    float3 ii=float3(8.0f/15.0f*M_PI_F*size.x*pow(max(size.y,size.z),4.0f),
                     8.0f/15.0f*M_PI_F*size.y*pow(max(size.z,size.x),4.0f),
                     8.0f/15.0f*M_PI_F*size.z*pow(max(size.x,size.y),4.0f));
    float imax=8.0f/15.0f*M_PI_F*dmid*pow(dmax,4.0f);
    float3 mom=lang*(angd*ii+slender*(imax-ii));
    float drag_lin=viscosity*lin_c+density*sqrt(max(l2,0.0f))*(aproj*blunt+slender*(amax-aproj));
    float drag_ang=viscosity*tq_c+density*length(mom);
    lt-=drag_ang*lang;
    lf-=drag_lin*llin;
    lf*=interact; lt*=interact;
    float3 world_torque=rot(gq,lt);
    float3 world_force=rot(gq,lf);
    int ancestor=b;
    while (ancestor>0) {
      int first=body_jntadr[ancestor], count=body_jntnum[ancestor];
      for (int k=0;k<count;++k) {
        int j=first+k, type=jnt_type[j], da=jnt_dofadr[j];
        uint ji=(jointbase+uint(j))*3;
        float3 anchor=get3(joint_anchor,ji), axis=get3(joint_axis,ji);
        float3 moment=cross(xgeom-anchor,world_force)+world_torque;
        if (type==2) qfrc[forcebase+uint(da)]+=dot(axis,world_force);
        else if (type==3) qfrc[forcebase+uint(da)]+=dot(axis,moment);
        else if (type==1) {
          float4 q=get4(body_quat,body4+uint(ancestor*4));
          for (int a=0;a<3;++a) qfrc[forcebase+uint(da+a)]+=dot(rot(q,float3(float(a==0),float(a==1),float(a==2))),moment);
        } else if (type==0) {
          for (int a=0;a<3;++a) qfrc[forcebase+uint(da+a)]+=world_force[a];
          float4 q=get4(body_quat,body4+uint(ancestor*4));
          for (int a=0;a<3;++a) qfrc[forcebase+uint(da+3+a)]+=dot(rot(q,float3(float(a==0),float(a==1),float(a==2))),moment);
        }
      }
      ancestor=parent[ancestor];
    }
  }
}
