// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

// Pinned MuJoCo 3.10 mj_flexPassiveStretch for simplex elements. The host
// lowers the compiler-generated 21-value upper-triangle into a symmetric 6x6
// metric. One thread emits one (environment, element, generalized-DOF) force;
// summing element contributions happens on-device after this kernel.
kernel void flex_stretch_force(
    device const float* vertex_xpos [[buffer(0)]],
    device const float* edge_length [[buffer(1)]],
    device const float* edge_velocity [[buffer(2)]],
    device const float* edge_length0 [[buffer(3)]],
    device const float* vertex_jacobian [[buffer(4)]],
    device const int* element_vertex [[buffer(5)]],
    device const int* element_edge [[buffer(6)]],
    device const float* metric [[buffer(7)]],
    device const float* damping [[buffer(8)]],
    device const int* edge_count [[buffer(9)]],
    device const int* vertex_count [[buffer(10)]],
    constant int* dims [[buffer(11)]],
    constant float* timestep [[buffer(12)]],
    device float* element_force [[buffer(13)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0];
  int nv = dims[1];
  int nelem = dims[2];
  int nvert = dims[3];
  int maxelem = batch * nelem * nv;
  if (tid >= uint(maxelem)) return;

  int dof = int(tid) % nv;
  int elem = (int(tid) / nv) % nelem;
  int env = int(tid) / (nv * nelem);
  int ne = edge_count[elem];
  int nve = vertex_count[elem];
  float h = timestep[0];
  float elong[6] = {0, 0, 0, 0, 0, 0};
  float grad[6][2][3];
  int eadr[6];
  int v[4];
  for (int i = 0; i < nve; ++i) {
    v[i] = element_vertex[4 * elem + i];
  }
  for (int e = 0; e < ne; ++e) {
    int ge = element_edge[6 * elem + e];
    eadr[e] = env * dims[4] + ge;
    int a;
    int b;
    if (nve == 3) {
      const int edge_a[3] = {1, 2, 0};
      const int edge_b[3] = {2, 0, 1};
      a = edge_a[e]; b = edge_b[e];
    } else {
      const int edge_a[6] = {0, 1, 2, 2, 0, 1};
      const int edge_b[6] = {1, 2, 0, 3, 3, 3};
      a = edge_a[e]; b = edge_b[e];
    }
    for (int d = 0; d < 3; ++d) {
      float xa = vertex_xpos[(env * dims[5] + v[a]) * 3 + d];
      float xb = vertex_xpos[(env * dims[5] + v[b]) * 3 + d];
      grad[e][0][d] = xa - xb;
      grad[e][1][d] = xb - xa;
    }
    float len = edge_length[eadr[e]];
    float rate = edge_velocity[eadr[e]];
    float prev = len - rate * h;
    float rest = edge_length0[ge];
    float kd = h > 0.0f ? damping[elem] / h : 0.0f;
    elong[e] = len * len - rest * rest +
               (len * len - prev * prev) * kd;
  }

  float local[4][3] = {{0}};
  for (int e1 = 0; e1 < ne; ++e1) {
    for (int e2 = 0; e2 < ne; ++e2) {
      float kij = metric[(elem * 6 + e1) * 6 + e2];
      for (int endpoint = 0; endpoint < 2; ++endpoint) {
        int local_vertex = (nve == 3 ?
            (e2 == 0 ? (endpoint == 0 ? 1 : 2) :
             e2 == 1 ? (endpoint == 0 ? 2 : 0) : (endpoint == 0 ? 0 : 1)) :
            (e2 == 0 ? (endpoint == 0 ? 0 : 1) :
             e2 == 1 ? (endpoint == 0 ? 1 : 2) :
             e2 == 2 ? (endpoint == 0 ? 2 : 0) :
             e2 == 3 ? (endpoint == 0 ? 2 : 3) :
             e2 == 4 ? (endpoint == 0 ? 0 : 3) : (endpoint == 0 ? 1 : 3)));
        for (int d = 0; d < 3; ++d) {
          local[local_vertex][d] -= elong[e1] * kij * grad[e2][endpoint][d];
        }
      }
    }
  }
  float qf = 0.0f;
  for (int i = 0; i < nve; ++i) {
    int vi = env * dims[5] + v[i];
    for (int d = 0; d < 3; ++d) {
      int jidx = ((env * dims[5] + v[i]) * 3 + d) * nv + dof;
      qf += local[i][d] * vertex_jacobian[jidx];
    }
  }
  element_force[tid] = qf;
}

// Pinned mj_flexPassiveBend for ordinary 2D triangular flexes. The 4x4
// bending matrix and curved-reference coefficient are compiler generated.
kernel void flex_bend_force(
    device const float* vertex_xpos [[buffer(0)]],
    device const float* vertex_xvel [[buffer(1)]],
    device const float* vertex_jacobian [[buffer(2)]],
    device const int* bend_vertex [[buffer(3)]],
    device const float* bend_data [[buffer(4)]],
    device const float* flex_damping [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device float* bend_force [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0];
  int nv = dims[1];
  int nedge = dims[2];
  int nvert = dims[3];
  if (tid >= uint(batch * nedge * nv)) return;
  int dof = int(tid) % nv;
  int e = (int(tid) / nv) % nedge;
  int env = int(tid) / (nv * nedge);
  int v[4];
  for (int i = 0; i < 4; ++i) v[i] = bend_vertex[4 * e + i];
  float ed0[3], ed1[3], ed2[3];
  for (int d = 0; d < 3; ++d) {
    float x0 = vertex_xpos[(env * nvert + v[0]) * 3 + d];
    ed0[d] = vertex_xpos[(env * nvert + v[1]) * 3 + d] - x0;
    ed1[d] = vertex_xpos[(env * nvert + v[2]) * 3 + d] - x0;
    ed2[d] = vertex_xpos[(env * nvert + v[3]) * 3 + d] - x0;
  }
  float ref[4][3];
  float3 c1 = cross(float3(ed1[0], ed1[1], ed1[2]), float3(ed2[0], ed2[1], ed2[2]));
  float3 c2 = cross(float3(ed2[0], ed2[1], ed2[2]), float3(ed0[0], ed0[1], ed0[2]));
  float3 c3 = cross(float3(ed0[0], ed0[1], ed0[2]), float3(ed1[0], ed1[1], ed1[2]));
  for (int d = 0; d < 3; ++d) {
    ref[1][d] = c1[d]; ref[2][d] = c2[d]; ref[3][d] = c3[d];
    ref[0][d] = -c1[d] - c2[d] - c3[d];
  }
  float f[4][3] = {{0}};
  float h[4][3] = {{0}};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float kij = bend_data[17 * e + 4 * i + j];
      for (int d = 0; d < 3; ++d) {
        f[i][d] += kij * vertex_xpos[(env * nvert + v[j]) * 3 + d];
        h[i][d] += kij * vertex_xvel[(env * nvert + v[j]) * 3 + d];
      }
    }
    for (int d = 0; d < 3; ++d) f[i][d] += bend_data[17 * e + 16] * ref[i][d];
  }
  float qf = 0.0f;
  for (int i = 0; i < 4; ++i) {
    for (int d = 0; d < 3; ++d) {
      float total = f[i][d] + flex_damping[e] * h[i][d];
      int ji = ((env * nvert + v[i]) * 3 + d) * nv + dof;
      qf -= total * vertex_jacobian[ji];
    }
  }
  bend_force[tid] = qf;
}
