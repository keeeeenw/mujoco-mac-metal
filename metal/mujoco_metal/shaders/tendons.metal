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
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nv=dims[1], ntendon=dims[2], batch=dims[3];
  int spring_disabled=dims[4], damper_disabled=dims[5];
  if (world>=uint(batch)) return;
  uint qbase=world*uint(nq), vbase=world*uint(nv);
  uint mbase=world*uint(nv*nv);
  for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=0.0f;
  for (int i=0;i<nv*nv;++i) { damping_matrix[mbase+uint(i)]=0.0f; armature_matrix[mbase+uint(i)]=0.0f; }
  for (int i=0;i<nq;++i) {
    if ((as_type<uint>(qpos[qbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) { damping_matrix[mbase+uint(k)]=bad; armature_matrix[mbase+uint(k)]=bad; }
      return;
    }
  }
  for (int i=0;i<nv;++i) {
    if ((as_type<uint>(qvel[vbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) { damping_matrix[mbase+uint(k)]=bad; armature_matrix[mbase+uint(k)]=bad; }
      return;
    }
  }
  for (int t=0;t<ntendon;++t) {
    float length=0.0f, velocity=0.0f;
    for (int q=0;q<nq;++q) length+=length_map[t*nq+q]*qpos[qbase+uint(q)];
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
        armature_matrix[index]+=armature[t]*ji*jj;
      }
    }
  }
  for (int i=0;i<nv;++i) {
    if ((as_type<uint>(qfrc[vbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) { damping_matrix[mbase+uint(k)]=bad; armature_matrix[mbase+uint(k)]=bad; }
      return;
    }
  }
  for (int i=0;i<nv*nv;++i) {
    if ((as_type<uint>(damping_matrix[mbase+uint(i)]) & 0x7f800000u)==0x7f800000u ||
        (as_type<uint>(armature_matrix[mbase+uint(i)]) & 0x7f800000u)==0x7f800000u) {
      float bad=as_type<float>(0x7fc00000u);
      for (int d=0;d<nv;++d) qfrc[vbase+uint(d)]=bad;
      for (int k=0;k<nv*nv;++k) { damping_matrix[mbase+uint(k)]=bad; armature_matrix[mbase+uint(k)]=bad; }
      return;
    }
  }
}
