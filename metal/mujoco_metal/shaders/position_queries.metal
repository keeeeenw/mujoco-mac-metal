// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Pinned engine_support.c / engine_util_spatial.c position manifold utilities.
#include <metal_stdlib>
using namespace metal;

inline float4 pos_mul(float4 a, float4 b) {
  return float4(a.x*b.x-a.y*b.y-a.z*b.z-a.w*b.w,
      a.x*b.y+a.y*b.x+a.z*b.w-a.w*b.z,
      a.x*b.z-a.y*b.w+a.z*b.x+a.w*b.y,
      a.x*b.w+a.y*b.z-a.z*b.y+a.w*b.x);
}

inline float4 pos_read4(device const float* p) {
  return float4(p[0],p[1],p[2],p[3]);
}

inline float4 pos_normalize(float4 q) {
  // Scaling avoids overflow/underflow in the source's norm calculation while
  // retaining its mjMINVAL identity rule and quaternion sign convention.
  float scale = max(max(abs(q.x),abs(q.y)),max(abs(q.z),abs(q.w)));
  if (scale == 0.0f) return float4(1,0,0,0);
  float4 scaled = q/scale;
  float norm = sqrt(dot(scaled,scaled));
  if (scale < 1e-15f/norm) return float4(1,0,0,0);
  return scaled/norm;
}

inline float pos_axis(thread float3& axis) {
  float scale = max(max(abs(axis.x),abs(axis.y)),abs(axis.z));
  if (scale == 0.0f) { axis=float3(1,0,0); return 0.0f; }
  float3 scaled = axis/scale;
  float norm = sqrt(dot(scaled,scaled));
  float magnitude = scale*norm;
  if (magnitude < 1e-15f) axis=float3(1,0,0);
  else axis=scaled/norm;
  return magnitude;
}

kernel void position_queries(
    device const float* first [[buffer(0)]],
    device const float* second [[buffer(1)]],
    device const int* joints [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    constant float* timestep [[buffer(4)]],
    constant int* operation [[buffer(5)]],
    device float* output [[buffer(6)]],
    device int* status [[buffer(7)]],
    uint world [[thread_position_in_grid]]) {
  int batch=dims[0], nq=dims[1], nv=dims[2], nj=dims[3], mode=operation[0];
  if (world >= uint(batch)) return;
  uint qb=world*uint(nq), vb=world*uint(nv);
  int width=mode==1?nv:nq;
  uint ob=world*uint(width);
  for (int i=0;i<width;++i) output[ob+i]=mode==1?0.0f:first[qb+i];
  status[world]=0;
  bool valid=true;
  for (int i=0;i<nq;++i) valid=valid && isfinite(first[qb+i]);
  if (mode<2) {
    int count=mode==0?nv:nq;
    uint base=mode==0?vb:qb;
    for (int i=0;i<count;++i) valid=valid && isfinite(second[base+i]);
  }
  if (!valid) { status[world]=1; return; }
  float dt=timestep[0];
  for (int j=0;j<nj;++j) {
    int kind=joints[3*j], q=joints[3*j+1], v=joints[3*j+2];
    if (kind==0 || kind>=2) {
      int count=kind==0?3:1;
      for (int i=0;i<count;++i) {
        if (mode==0) output[qb+q+i]=first[qb+q+i]+dt*second[vb+v+i];
        else if (mode==1) output[vb+v+i]=(second[qb+q+i]-first[qb+q+i])/dt;
      }
      if (kind==0) { q+=3; v+=3; }
    }
    if (kind==0 || kind==1) {
      float4 quat=pos_read4(first+qb+q);
      if (mode==1) {
        quat.yzw=-quat.yzw;
        float4 delta=pos_mul(quat,pos_read4(second+qb+q));
        float3 axis=delta.yzw;
        float sine=pos_axis(axis);
        float angle=2.0f*atan2(sine,delta.x);
        if (angle>M_PI_F) angle-=2.0f*M_PI_F;
        float3 velocity=axis*(angle/dt);
        for (int i=0;i<3;++i) output[vb+v+i]=velocity[i];
      } else {
        quat=pos_normalize(quat);
        if (mode==0) {
          float3 axis=float3(second[vb+v],second[vb+v+1],second[vb+v+2]);
          float angle=dt*pos_axis(axis);
          float4 rotation=angle==0.0f?float4(1,0,0,0):
              float4(cos(angle*.5f),axis*sin(angle*.5f));
          quat=pos_mul(quat,rotation);
        }
        for (int i=0;i<4;++i) output[qb+q+i]=quat[i];
      }
    }
  }
  for (int i=0;i<width;++i) valid=valid && isfinite(output[ob+i]);
  if (!valid) {
    status[world]=2;
    for (int i=0;i<width;++i) output[ob+i]=mode==1?0.0f:first[qb+i];
  }
}
