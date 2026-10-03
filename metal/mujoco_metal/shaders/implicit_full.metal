// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

// Assemble full nonsymmetric H = M - h*qDeriv for the general implicit integrator.
kernel void assemble_nonsymmetric_implicit_mass(
    device const float* mass [[buffer(0)]],
    device const float* force_velocity_derivative [[buffer(1)]],
    device float* effective_mass [[buffer(2)]],
    constant int* dims [[buffer(3)]], constant float* timestep [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nv=dims[1];
  uint count=uint(batch*nv*nv);
  if (tid>=count) return;
  effective_mass[tid]=mass[tid]-timestep[0]*force_velocity_derivative[tid];
}
