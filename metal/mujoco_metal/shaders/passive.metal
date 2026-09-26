// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0
#include <metal_stdlib>
using namespace metal;

static float4 qmul_passive(float4 a, float4 b) {
  return float4(a.x*b.x-dot(a.yzw,b.yzw),
    a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}

static float3 quat_diff(float4 qa, float4 qb) {
  qa = normalize(qa); qb = normalize(qb);
  float4 d = qmul_passive(float4(qb.x, -qb.yzw), qa);
  float s = length(d.yzw);
  if (s < 1e-12f) return float3(0.0f);
  float speed = 2.0f * atan2(s, d.x);
  if (speed > M_PI_F) speed -= 2.0f * M_PI_F;
  return d.yzw * (speed / s);
}

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
    device float* damping_derivative [[buffer(12)]],
    uint world [[thread_position_in_grid]]) {
  int nq = dims[0], nv = dims[1], njnt = dims[2], n_poly = dims[3];
  int spring_disabled = dims[4];
  int damper_disabled = dims[5];
  if (n_poly != 2) return;
  uint qo = world * uint(nq), vo = world * uint(nv), fo = world * uint(nv);
  for (int d = 0; d < nv; ++d) { force[fo + uint(d)] = 0.0f; damping_derivative[fo + uint(d)] = 0.0f; }
  for (int i=0; i<nq; ++i) {
    if ((as_type<uint>(qpos[qo+uint(i)]) & 0x7f800000u) == 0x7f800000u) {
      float bad = as_type<float>(0x7fc00000u);
      for (int d=0; d<nv; ++d) { force[fo+uint(d)] = bad; damping_derivative[fo+uint(d)] = bad; }
      return;
    }
  }
  for (int i=0; i<nv; ++i) {
    if ((as_type<uint>(qvel[vo+uint(i)]) & 0x7f800000u) == 0x7f800000u) {
      float bad = as_type<float>(0x7fc00000u);
      for (int d=0; d<nv; ++d) { force[fo+uint(d)] = bad; damping_derivative[fo+uint(d)] = bad; }
      return;
    }
  }
  if (!spring_disabled) {
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
        float4 current = normalize(float4(qpos[qo+uint(qa+3)], qpos[qo+uint(qa+4)], qpos[qo+uint(qa+5)], qpos[qo+uint(qa+6)]));
        float4 spring = float4(springref[qa+3], springref[qa+4], springref[qa+5], springref[qa+6]);
        float3 angular = quat_diff(current, spring);
        float ar = length(angular);
        float ak = stiffness[j] + springpoly[2*j]*ar + springpoly[2*j+1]*ar*ar;
        for (int a=0; a<3; ++a) force[fo+uint(da+3+a)] -= angular[a]*ak;
      } else if (type[j] == 1) {
        float4 current = normalize(float4(qpos[qo+uint(qa)], qpos[qo+uint(qa+1)], qpos[qo+uint(qa+2)], qpos[qo+uint(qa+3)]));
        float4 spring = float4(springref[qa], springref[qa+1], springref[qa+2], springref[qa+3]);
        float3 angular = quat_diff(current, spring);
        float r = length(angular);
        float k = stiffness[j] + springpoly[2*j]*r + springpoly[2*j+1]*r*r;
        for (int a=0; a<3; ++a) force[fo+uint(da+a)] -= angular[a]*k;
      }
    }
  }
  if (!damper_disabled) {
    for (int d=0; d<nv; ++d) {
      float v = qvel[vo+uint(d)];
      float av = abs(v);
      float c = damping[d] + damperpoly[2*d]*av + damperpoly[2*d+1]*v*v;
      force[fo+uint(d)] -= v*c;
      damping_derivative[fo+uint(d)] = damping[d] + 2.0f*damperpoly[2*d]*av + 3.0f*damperpoly[2*d+1]*v*v;
    }
  }
}
