// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

// Add the spatial-tendon damping tangent directly to MuJoCo's compiled D-CSR
// COO slots. The spatial tendon Jacobian is already assembled by the current
// kinematics stage; fixed tendons are excluded by their zero path-count.
kernel void spatial_tendon_damping_coo(
    device const float* jacobian [[buffer(0)]],
    device const float* velocity [[buffer(1)]],
    device const float* damping [[buffer(2)]],
    device const float* dampingpoly [[buffer(3)]],
    device const int* path_count [[buffer(4)]],
    device const int* edge_rows [[buffer(5)]],
    device const int* edge_cols [[buffer(6)]],
    constant int* dims [[buffer(7)]],
    device float* edge_values [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0];
  int ntendon = dims[1];
  int batch = dims[2];
  int edge_count = dims[3];
  int damper_disabled = dims[4];
  int stride = max(edge_count, 1);
  if (damper_disabled || edge_count <= 0 || int(tid) >= batch * stride) return;
  int world = int(tid) / stride;
  int edge = int(tid) - world * stride;
  int row = edge_rows[edge];
  int col = edge_cols[edge];
  if (row < 0 || row >= nv || col < 0 || col >= nv) return;

  float value = 0.0f;
  for (int tendon = 0; tendon < ntendon; ++tendon) {
    if (path_count[tendon] <= 0) continue;
    int vbase = world * max(ntendon, 1) + tendon;
    float speed = velocity[vbase];
    float tangent = damping[tendon]
        + 2.0f * dampingpoly[2 * tendon] * abs(speed)
        + 3.0f * dampingpoly[2 * tendon + 1] * speed * speed;
    int jbase = (world * max(ntendon, 1) + tendon) * max(nv, 1);
    value -= tangent * jacobian[jbase + row] * jacobian[jbase + col];
  }
  edge_values[tid] += value;
}
