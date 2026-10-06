// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

kernel void velocity_derivative_clear(
    device float* values [[buffer(0)]],
    constant int* dims [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[2];
  int stride = max(dims[1], 1);
  if (int(tid) >= batch * stride) return;
  values[tid] = 0.0f;
}

kernel void velocity_derivative_mirror_lower(
    device const float* values [[buffer(0)]],
    device const int* source_slots [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int edge_count = dims[1], batch = dims[2];
  int stride = max(edge_count, 1);
  if (int(tid) >= batch * stride) return;
  int edge = int(tid) - (int(tid) / stride) * stride;
  int source = source_slots[edge];
  output[tid] = (edge < edge_count && source >= 0 && source < edge_count)
      ? values[(int(tid) / stride) * stride + source] : 0.0f;
}

kernel void velocity_derivative_add_diagonal(
    device float* values [[buffer(0)]],
    device const float* damping_magnitude [[buffer(1)]],
    device const int* diagonal_slots [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0], edge_count = dims[1], batch = dims[2];
  if (nv <= 0 || int(tid) >= batch * nv) return;
  int world = int(tid) / nv;
  int dof = int(tid) - world * nv;
  int slot = diagonal_slots[dof];
  if (slot < 0 || slot >= edge_count) return;
  // MuJoCo qDeriv is d(qfrc)/d(qvel); passive damper contributes -D.
  values[world * max(edge_count, 1) + slot] -= damping_magnitude[tid];
}

kernel void velocity_derivative_add_column(
    device float* values [[buffer(0)]],
    device const float* derivative_column [[buffer(1)]],
    device const int* edge_rows [[buffer(2)]],
    device const int* edge_cols [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0], edge_count = dims[1], batch = dims[2];
  int column = dims[3], stride = max(edge_count, 1);
  if (edge_count <= 0 || int(tid) >= batch * stride) return;
  int world = int(tid) / stride;
  int edge = int(tid) - world * stride;
  if (edge_cols[edge] != column) return;
  int row = edge_rows[edge];
  if (row < 0 || row >= nv) return;
  values[tid] += derivative_column[world * nv + row];
}

inline void add_dense_source(
    device float* values, device const float* source,
    device const int* edge_rows, device const int* edge_cols,
    constant int* dims, uint tid, float sign) {
  int nv = dims[0], edge_count = dims[1], batch = dims[2];
  int stride = max(edge_count, 1);
  if (edge_count <= 0 || int(tid) >= batch * stride) return;
  int world = int(tid) / stride;
  int edge = int(tid) - world * stride;
  int row = edge_rows[edge], col = edge_cols[edge];
  if (row < 0 || row >= nv || col < 0 || col >= nv) return;
  values[tid] += sign * source[world * nv * nv + row * nv + col];
}

kernel void velocity_derivative_add_dense_positive(
    device float* values [[buffer(0)]],
    device const float* source [[buffer(1)]],
    device const int* edge_rows [[buffer(2)]],
    device const int* edge_cols [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  add_dense_source(values, source, edge_rows, edge_cols, dims, tid, 1.0f);
}

kernel void velocity_derivative_add_dense_negative(
    device float* values [[buffer(0)]],
    device const float* source [[buffer(1)]],
    device const int* edge_rows [[buffer(2)]],
    device const int* edge_cols [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  add_dense_source(values, source, edge_rows, edge_cols, dims, tid, -1.0f);
}
