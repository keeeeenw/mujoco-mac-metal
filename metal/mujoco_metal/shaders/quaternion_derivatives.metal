// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline float4 qd_read(device const float* p) { return float4(p[0],p[1],p[2],p[3]); }
inline float4 qd_mul(float4 a,float4 b) {
  return float4(a.x*b.x-a.y*b.y-a.z*b.z-a.w*b.w,
      a.x*b.y+a.y*b.x+a.z*b.w-a.w*b.z,
      a.x*b.z-a.y*b.w+a.z*b.x+a.w*b.y,
      a.x*b.w+a.y*b.z-a.z*b.y+a.w*b.x);
}
inline float qd_normalize(thread float3& a) {
  float norm=sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
  if (norm<1e-15f) a=float3(1,0,0);
  else a/=norm;
  return norm;
}

kernel void quaternion_derivatives(
    device const float* first [[buffer(0)]],
    device const float* second [[buffer(1)]],
    constant int* batch [[buffer(2)]],
    constant float* scale [[buffer(3)]],
    constant int* operation [[buffer(4)]],
    device float* output [[buffer(5)]],
    device int* status [[buffer(6)]],
    uint world [[thread_position_in_grid]]) {
  if (world>=uint(batch[0])) return;
  uint base=21*world;
  for (int i=0;i<21;++i) output[base+i]=0;
  status[world]=0;
  int mode=operation[0];
  int width=mode==0?4:3;
  bool ok=true;
  for (int i=0;i<width;++i) {
    ok=ok && isfinite(first[world*width+i]);
    if (mode==0) ok=ok && isfinite(second[world*width+i]);
  }
  if (!ok) { status[world]=1; return; }
  float A[9], B[9];
  if (mode==0) {
    float4 qb=qd_read(second+4*world);
    qb.yzw=-qb.yzw;
    float4 delta=qd_mul(qb,qd_read(first+4*world));
    float3 axis=delta.yzw;
    float sine=qd_normalize(axis);
    float angle=2*atan2(sine,delta.x);
    if (angle>M_PI_F) angle-=2*M_PI_F;
    axis*=angle;
    float half_angle=.5f*qd_normalize(axis);
    float K[9]={0,-axis.z,axis.y,axis.z,0,-axis.x,-axis.y,axis.x,0};
    float coef=0;
    if (half_angle>=6e-8f) {
      if (half_angle<1.0f/32) {
        float hh=half_angle*half_angle;
        coef=hh*(1.0f/3+hh*(1.0f/45+hh*(2.0f/945)));
      } else coef=1-half_angle/tan(half_angle);
    }
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      float kk=0;
      for (int k=0;k<3;++k) kk+=K[3*i+k]*K[3*k+j];
      A[3*i+j]=(i==j?1.0f:0.0f)+half_angle*K[3*i+j]+coef*kk;
    }
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) B[3*i+j]=-A[3*j+i];
  } else {
    float3 vel=float3(first[3*world],first[3*world+1],first[3*world+2]);
    float3 s=scale[0]*vel;
    float xx=dot(s,s),x=sqrt(xx),a=cos(x),b,c,d;
    if (fabs(x)>1.0f/32) {
      b=sin(x)/x; c=(1-a)/xx; d=(1-b)/xx;
    } else {
      b=1+xx/6*(xx/20*(1-xx/42)-1);
      c=(1+xx/12*(xx/30*(1-xx/56)-1))/2;
      d=(1+xx/20*(xx/42*(1-xx/72)-1))/6;
    }
    float K[9]={0,s.z,-s.y,-s.z,0,s.x,s.y,-s.x,0};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      float eye=i==j?1.0f:0.0f,outer=s[i]*s[j];
      A[3*i+j]=a*eye+b*K[3*i+j]+c*outer;
      B[3*i+j]=b*eye+c*K[3*i+j]+d*outer;
    }
    for (int i=0;i<3;++i)
      output[base+18+i]=B[3*i]*vel.x+B[3*i+1]*vel.y+B[3*i+2]*vel.z;
  }
  for (int i=0;i<9;++i) {
    output[base+i]=A[i]; output[base+9+i]=B[i];
  }
  for (int i=0;i<21;++i) ok=ok && isfinite(output[base+i]);
  if (!ok) {
    status[world]=2;
    for (int i=0;i<21;++i) output[base+i]=0;
  }
}
