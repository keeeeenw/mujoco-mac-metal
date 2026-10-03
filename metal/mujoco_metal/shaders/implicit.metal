// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

// Assemble H=M-h*qDeriv. qDeriv is the total velocity derivative of
// qfrc_passive + qfrc_actuator; MuJoCo implicitfast intentionally omits the
// RNE/bias derivative. One thread handles one matrix element.
kernel void assemble_implicit_mass(
    device const float* mass [[buffer(0)]],
    device const float* force_velocity_derivative [[buffer(1)]],
    device float* effective_mass [[buffer(2)]],
    constant int* dims [[buffer(3)]], constant float* timestep [[buffer(4)]],
    device float* full_effective_mass [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nv=dims[1];
  uint count=uint(batch*nv*nv);
  if (tid>=count) return;
  int row=int(tid)%nv;
  int col=(int(tid)/nv)%nv;
  int base=int(tid)/(nv*nv)*(nv*nv);
  int derivative_index=base+(row>=col ? row*nv+col : col*nv+row);
  // qH gathers qDeriv into MuJoCo's lower-triangular mass pattern. Mirror
  // that authoritative lower triangle so the dense LDL/Cholesky input is SPD.
  effective_mass[tid]=mass[tid]-timestep[0]*force_velocity_derivative[derivative_index];
  // The stiffness correction's matrix-vector product uses full qDeriv,
  // unlike its symmetric preconditioner. Do not mirror this output.
  full_effective_mass[tid]=mass[tid]-timestep[0]*force_velocity_derivative[tid];
}

