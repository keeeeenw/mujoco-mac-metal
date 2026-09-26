// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0
#include <metal_stdlib>
using namespace metal;

kernel void passive_joint_force(
    device const float* qpos [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const int* qadr [[buffer(2)]],
    device const int* dadr [[buffer(3)]],
    device const int* type [[buffer(4)]],
    device const float* springref [[buffer(5)]],
    device const float* stiffness [[buffer(6)]],
    device const float* springpoly [[buffer(7)]],
    device const float* damping [[buffer(8)]],
    device const float* damperpoly [[buffer(9)]],
    constant int* dims [[buffer(10)]],
    device float* force [[buffer(11)]],
    uint world [[thread_position_in_grid]]) {
  int nq = dims[0], nv = dims[1], njnt = dims[2], n_poly = dims[3];
  int disableflags = dims[4];
  if (n_poly != 2) return;
  uint qo = world * uint(nq), vo = world * uint(nv), fo = world * uint(nv);
  for (int d = 0; d < nv; ++d) force[fo + uint(d)] = 0.0f;
  const int disable_spring = 1 << 5;
  const int disable_damper = 1 << 6;
  if ((disableflags & disable_spring) == 0) {
    for (int j = 0; j < njnt; ++j) {
      int qa = qadr[j], da = dadr[j];
      if (type[j] == 2 || type[j] == 3) {
        float x = qpos[qo + uint(qa)] - springref[qa];
        float k = stiffness[j] + springpoly[2*j] * x + springpoly[2*j+1] * x*x;
        force[fo + uint(da)] -= x*k;
      } else if (type[j] == 0) {
        float3 diff = float3(qpos[qo+uint(qa)]-springref[qa], qpos[qo+uint(qa+1)]-springref[qa+1], qpos[qo+uint(qa+2)]-springref[qa+2]);
        float r = length(diff);
        float k = stiffness[j] + springpoly[2*j]*r + springpoly[2*j+1]*r*r;
        for (int a=0; a<3; ++a) force[fo+uint(da+a)] -= diff[a]*k;
      }
    }
  }
  if ((disableflags & disable_damper) == 0) {
    for (int d=0; d<nv; ++d) {
      float v = qvel[vo+uint(d)];
      float av = abs(v);
      float c = damping[d] + damperpoly[2*d]*av + damperpoly[2*d+1]*v*v;
      force[fo+uint(d)] -= v*c;
    }
  }
}
