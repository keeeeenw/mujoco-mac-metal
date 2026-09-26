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
}
