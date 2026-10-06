// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

// Three-word coordinate arithmetic retains the residual lost by public
// float32 pose/material buffers. The first word remains the established ABI;
// two auxiliary residual planes reach support without host readback.
struct FPDD { float hi; float lo; float tail; };
struct FPDD3 { FPDD x; FPDD y; FPDD z; };
inline FPDD fpdd(float x) { return FPDD{x,0.0f,0.0f}; }
inline FPDD fpadd(FPDD a, FPDD b) {
  float s=a.hi+b.hi;
  float bv=s-a.hi;
  float e=(a.hi-(s-bv))+(b.hi-bv);
  float t=a.lo+b.lo;
  float tv=t-a.lo;
  float te=(a.lo-(t-tv))+(b.lo-tv);
  float u=e+t;
  float uv=u-e;
  float ue=(e-(u-uv))+(t-uv);
  float h=s+u;
  float hv=h-s;
  float he=(s-(h-hv))+(u-hv);
  float tail=ue+te+a.tail+b.tail;
  return FPDD{h,he,tail};
}
inline FPDD fpneg(FPDD a) { return FPDD{-a.hi,-a.lo,-a.tail}; }
inline FPDD fpsub(FPDD a, FPDD b) { return fpadd(a,fpneg(b)); }
inline FPDD fpmul(FPDD a, FPDD b) {
  FPDD out=fpdd(0.0f);
  float av[3]={a.hi,a.lo,a.tail};
  float bv[3]={b.hi,b.lo,b.tail};
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
    float p=av[i]*bv[j];
    float e=fma(av[i],bv[j],-p);
    out=fpadd(out,FPDD{p,e,0.0f});
  }
  return out;
}
inline FPDD3 fp3(float3 hi, float3 lo, float3 tail) {
  return FPDD3{FPDD{hi.x,lo.x,tail.x},FPDD{hi.y,lo.y,tail.y},
               FPDD{hi.z,lo.z,tail.z}};
}
inline FPDD3 fp3add(FPDD3 a, FPDD3 b) {
  return FPDD3{fpadd(a.x,b.x),fpadd(a.y,b.y),fpadd(a.z,b.z)};
}
inline FPDD3 fp3sub(FPDD3 a, FPDD3 b) {
  return FPDD3{fpsub(a.x,b.x),fpsub(a.y,b.y),fpsub(a.z,b.z)};
}
inline FPDD3 fp3cross(float3 a, FPDD3 b) {
  return FPDD3{
    fpsub(fpmul(fpdd(a.y),b.z),fpmul(fpdd(a.z),b.y)),
    fpsub(fpmul(fpdd(a.z),b.x),fpmul(fpdd(a.x),b.z)),
    fpsub(fpmul(fpdd(a.x),b.y),fpmul(fpdd(a.y),b.x))};
}
inline FPDD3 fp3scale(FPDD3 a, float s) {
  return FPDD3{fpmul(a.x,fpdd(s)),fpmul(a.y,fpdd(s)),fpmul(a.z,fpdd(s))};
}
inline FPDD3 fpqrot(float4 q, FPDD3 v) {
  FPDD3 c=fp3cross(q.yzw,v);
  FPDD3 inner=FPDD3{
      fpadd(c.x,fpmul(v.x,fpdd(q.x))),
      fpadd(c.y,fpmul(v.y,fpdd(q.x))),
      fpadd(c.z,fpmul(v.z,fpdd(q.x)))};
  return fp3add(v,fp3scale(fp3cross(q.yzw,inner),2.0f));
}
inline float2 fp_residual_pair(FPDD a, float high) {
  FPDD d=fpsub(a,fpdd(high));
  return float2(d.hi,d.lo+d.tail);
}

kernel void flex_paired_point_position_low(
    device const float* body_pos [[buffer(0)]],
    device const float* body_pos_low [[buffer(1)]],
    device const float* body_quat [[buffer(2)]],
    device const float* local [[buffer(3)]],
    device const float* local_low [[buffer(4)]],
    device const float* body_pos_tail [[buffer(5)]],
    device const float* local_tail [[buffer(6)]],
    device const int* body_ids [[buffer(7)]],
    device const int* centered [[buffer(8)]],
    constant int* dims [[buffer(9)]],
    device const float* position_high [[buffer(10)]],
    device float* position_low [[buffer(11)]],
    device float* position_tail [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], npoint=dims[1], nbody=dims[2];
  if (int(tid)>=batch*npoint) return;
  int point=int(tid)%npoint, world=int(tid)/npoint;
  if (dims[3+world]==0) return;
  int body=body_ids[point];
  int bp=(world*nbody+body)*3, q=(world*nbody+body)*4, p=int(tid)*3;
  float3 bh=float3(body_pos[bp],body_pos[bp+1],body_pos[bp+2]);
  float3 bl=float3(body_pos_low[bp],body_pos_low[bp+1],body_pos_low[bp+2]);
  float3 bt=float3(body_pos_tail[bp],body_pos_tail[bp+1],body_pos_tail[bp+2]);
  float4 quat=float4(body_quat[q],body_quat[q+1],body_quat[q+2],body_quat[q+3]);
  float3 lh=float3(local[point*3],local[point*3+1],local[point*3+2]);
  float3 ll=float3(local_low[point*3],local_low[point*3+1],local_low[point*3+2]);
  float3 lt=float3(local_tail[point*3],local_tail[point*3+1],local_tail[point*3+2]);
  FPDD3 value=fp3(bh,bl,bt);
  if (!centered[point]) value=fp3add(value,fpqrot(quat,fp3(lh,ll,lt)));
  float2 rx=fp_residual_pair(value.x,position_high[p]);
  float2 ry=fp_residual_pair(value.y,position_high[p+1]);
  float2 rz=fp_residual_pair(value.z,position_high[p+2]);
  position_low[p]=rx.x; position_tail[p]=rx.y;
  position_low[p+1]=ry.x; position_tail[p+1]=ry.y;
  position_low[p+2]=rz.x; position_tail[p+2]=rz.y;
}

kernel void flex_paired_interpolate_position_low(
    device const int* enabled [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* weights_low [[buffer(3)]],
    device const float* weights_tail [[buffer(4)]],
    device const float* source_high [[buffer(5)]],
    device const float* source_low [[buffer(6)]],
    device const float* source_tail [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    device const float* target_high [[buffer(9)]],
    device float* target_low [[buffer(10)]],
    device float* target_tail [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], ntarget=dims[1], nsource=dims[2], nterm=dims[3];
  if (int(tid)>=batch*ntarget) return;
  int target=int(tid)%ntarget, world=int(tid)/ntarget;
  if (dims[4+world]==0 || !enabled[target]) return;
  FPDD3 value=FPDD3{fpdd(0.0f),fpdd(0.0f),fpdd(0.0f)};
  for (int term=0;term<nterm;term++) {
    int source=indices[target*nterm+term];
    int s=(world*nsource+source)*3;
    FPDD w=FPDD{weights[target*nterm+term],weights_low[target*nterm+term],
                weights_tail[target*nterm+term]};
    FPDD3 x=fp3(float3(source_high[s],source_high[s+1],source_high[s+2]),
                float3(source_low[s],source_low[s+1],source_low[s+2]),
                float3(source_tail[s],source_tail[s+1],source_tail[s+2]));
    value=fp3add(value,FPDD3{fpmul(w,x.x),fpmul(w,x.y),fpmul(w,x.z)});
  }
  int out=(world*ntarget+target)*3;
  float2 rx=fp_residual_pair(value.x,target_high[out]);
  float2 ry=fp_residual_pair(value.y,target_high[out+1]);
  float2 rz=fp_residual_pair(value.z,target_high[out+2]);
  target_low[out]=rx.x; target_tail[out]=rx.y;
  target_low[out+1]=ry.x; target_tail[out+1]=ry.y;
  target_low[out+2]=rz.x; target_tail[out+2]=rz.y;
}

// Source-faithful mju_shellTrackInterior for compiled shell node grids.
// In-place writes are race-free: every interior target reads only boundary
// nodes, which are never targets in this dispatch.
kernel void flex_shell_tfi_vectors(
    device const int* shell_enabled [[buffer(0)]],
    device const int* node_indices [[buffer(1)]],
    device const float* node_weights [[buffer(2)]],
    device float* node_position [[buffer(3)]],
    device float* node_velocity [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nnode=dims[1];
  if (int(tid)>=batch*nnode) return;
  int env=int(tid)/nnode, node=int(tid)%nnode;
  if(dims[3+env]==0) return;
  if (!shell_enabled[node]) return;
  float3 pos=float3(0.0f), vel=float3(0.0f);
  for (int term=0;term<26;term++) {
    float weight=node_weights[node*26+term];
    int source=node_indices[node*26+term];
    int base=(env*nnode+source)*3;
    pos+=weight*float3(node_position[base],node_position[base+1],
                       node_position[base+2]);
    vel+=weight*float3(node_velocity[base],node_velocity[base+1],
                       node_velocity[base+2]);
  }
  int out=(env*nnode+node)*3;
  node_position[out]=pos.x; node_position[out+1]=pos.y;
  node_position[out+2]=pos.z;
  node_velocity[out]=vel.x; node_velocity[out+1]=vel.y;
  node_velocity[out+2]=vel.z;
}

// Apply the same pinned shell TFI map to each Cartesian generalized-J entry.
kernel void flex_shell_tfi_jacobian(
    device const int* shell_enabled [[buffer(0)]],
    device const int* node_indices [[buffer(1)]],
    device const float* node_weights [[buffer(2)]],
    device float* node_jacobian [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nnode=dims[1], nv=dims[2], stride=max(nv,1);
  int total=batch*nnode*3*nv;
  if (int(tid)>=total || nv<=0) return;
  int dof=int(tid)%nv;
  int component=(int(tid)/nv)%3;
  int node=(int(tid)/(3*nv))%nnode;
  int env=int(tid)/(nnode*3*nv);
  if(dims[3+env]==0) return;
  if (!shell_enabled[node]) return;
  float value=0.0f;
  for (int term=0;term<26;term++) {
    int source=node_indices[node*26+term];
    value+=node_weights[node*26+term]
        *node_jacobian[((env*nnode+source)*3+component)*stride+dof];
  }
  node_jacobian[((env*nnode+node)*3+component)*stride+dof]=value;
}

// Interpolate compiled Q1/Q2 vertices from their source nodal state. Direct
// flex vertices are left untouched; the fixed map is built from flex_vert0,
// flex_cellnum, and the exact mju_cellLookup/evalBasis ordering.
kernel void flex_interpolate_vertices(
    device const int* vertex_enabled [[buffer(0)]],
    device const int* node_indices [[buffer(1)]],
    device const float* node_weights [[buffer(2)]],
    device const float* node_position [[buffer(3)]],
    device const float* node_velocity [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    device float* vertex_position [[buffer(6)]],
    device float* vertex_velocity [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nvert=dims[1], nnode=dims[2];
  if (int(tid)>=batch*nvert) return;
  int env=int(tid)/nvert, vert=int(tid)%nvert;
  if(dims[4+env]==0) return;
  if (!vertex_enabled[vert]) return;
  float3 pos=float3(0.0f), vel=float3(0.0f);
  for (int term=0;term<27;term++) {
    float weight=node_weights[vert*27+term];
    int source=node_indices[vert*27+term];
    int base=(env*nnode+source)*3;
    pos+=weight*float3(node_position[base],node_position[base+1],
                       node_position[base+2]);
    vel+=weight*float3(node_velocity[base],node_velocity[base+1],
                       node_velocity[base+2]);
  }
  int out=(env*nvert+vert)*3;
  vertex_position[out]=pos.x; vertex_position[out+1]=pos.y;
  vertex_position[out+2]=pos.z;
  vertex_velocity[out]=vel.x; vertex_velocity[out+1]=vel.y;
  vertex_velocity[out+2]=vel.z;
}

kernel void flex_interpolate_vertex_jacobian(
    device const int* vertex_enabled [[buffer(0)]],
    device const int* node_indices [[buffer(1)]],
    device const float* node_weights [[buffer(2)]],
    device const float* node_jacobian [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    device float* vertex_jacobian [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nvert=dims[1], nnode=dims[2], nv=dims[3];
  int stride=max(nv,1), total=batch*nvert*3*nv;
  if (int(tid)>=total || nv<=0) return;
  int dof=int(tid)%nv;
  int component=(int(tid)/nv)%3;
  int vert=(int(tid)/(3*nv))%nvert;
  int env=int(tid)/(nvert*3*nv);
  if(dims[4+env]==0) return;
  if (!vertex_enabled[vert]) return;
  float value=0.0f;
  for (int term=0;term<27;term++) {
    int source=node_indices[vert*27+term];
    value+=node_weights[vert*27+term]
        *node_jacobian[((env*nnode+source)*3+component)*stride+dof];
  }
  vertex_jacobian[((env*nvert+vert)*3+component)*stride+dof]=value;
}

// Add the compiled flex-edge damper contribution directly to the shared
// MuJoCo D-CSR COO values. Each thread owns one (world, COO slot), so it can
// reduce over all flex edges without atomics or a dense nv-by-nv temporary.
kernel void flex_edge_velocity_derivative_coo(
    device const float* edge_jacobian [[buffer(0)]],
    device const float* edge_coefficient [[buffer(1)]],
    device const int* edge_rows [[buffer(2)]],
    device const int* edge_cols [[buffer(3)]],
    device float* values [[buffer(4)]],
    constant int* dims [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0];
  int nflexedge = dims[1];
  int d_edges = dims[2];
  int batch = dims[3];
  int stride = max(d_edges, 1);
  if (d_edges <= 0 || int(tid) >= batch * stride) return;
  int world = int(tid) / stride;
  int slot = int(tid) - world * stride;
  int row = edge_rows[slot];
  int col = edge_cols[slot];
  if (row < 0 || row >= nv || col < 0 || col >= nv) return;
  float contribution = 0.0f;
  for (int edge = 0; edge < nflexedge; ++edge) {
    float coefficient = edge_coefficient[edge];
    if (coefficient == 0.0f) continue;
    int base = (world * nflexedge + edge) * nv;
    contribution -= coefficient * edge_jacobian[base + row]
                                  * edge_jacobian[base + col];
  }
  values[tid] += contribution;
}

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
  // Selected-world recovery may recompute this material row while healthy
  // worlds retain their previous force/cache state.  Test the row predicate
  // before reading any world-dependent geometry.
  if (dims[6 + env] == 0) return;
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
    device const float* qvel [[buffer(8)]],
    device const int* vertex_dofadr [[buffer(9)]],
    device const int* vertex_dofnum [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0];
  int nv = dims[1];
  int nedge = dims[2];
  int nvert = dims[3];
  if (tid >= uint(batch * nedge * nv)) return;
  int dof = int(tid) % nv;
  int e = (int(tid) / nv) % nedge;
  int env = int(tid) / (nv * nedge);
  if (dims[4 + env] == 0) return;
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
        int vadr=vertex_dofadr[4*e+j];
        if (d<vertex_dofnum[4*e+j]) h[i][d] += kij * qvel[env*nv+vadr+d];
      }
    }
    for (int d = 0; d < 3; ++d) f[i][d] += bend_data[17 * e + 16] * ref[i][d];
  }
  float qf = 0.0f;
  for (int i = 0; i < 4; ++i) {
    int vadr=vertex_dofadr[4*e+i], vnum=vertex_dofnum[4*e+i];
    if (dof>=vadr && dof<vadr+vnum) {
      int d=dof-vadr;
      qf -= f[i][d] + flex_damping[e] * h[i][d];
    }
  }
  bend_force[tid] = qf;
}

static inline float3 quat_rotate(float4 q, float3 v) {
  float3 qv = q.yzw;
  float3 t = 2.0f * cross(qv, v);
  return v + q.x * t + cross(qv, t);
}

// Selected-world flex point kinematics and rigid attachment Jacobians for
// checkAcc recovery. One thread owns a world/point row and returns on healthy
// rows before reading any state-dependent input.
kernel void flex_recovery_update_points(
    device const float* local [[buffer(0)]],
    device const int* body_ids [[buffer(1)]],
    device const int* centered [[buffer(2)]],
    device const float* body_pos [[buffer(3)]],
    device const float* body_quat [[buffer(4)]],
    device const float* cvel [[buffer(5)]],
    device const float* root_com [[buffer(6)]],
    device const float* joint_anchor [[buffer(7)]],
    device const float* joint_axis [[buffer(8)]],
    device const int* body_jntadr [[buffer(9)]],
    device const int* body_jntnum [[buffer(10)]],
    device const int* body_parentid [[buffer(11)]],
    device const int* jnt_dofadr [[buffer(12)]],
    device const int* jnt_type [[buffer(13)]],
    device const int* selected [[buffer(14)]],
    constant int* dims [[buffer(15)]],
    device float* xpos [[buffer(16)]],
    device float* xvel [[buffer(17)]],
    device float* jacobian [[buffer(18)]],
    device float* spatial_jacobian [[buffer(19)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], npoint=dims[1], nv=dims[2], nbody=dims[3];
  int njnt=dims[4], has_spatial=dims[5];
  if (tid>=uint(batch*npoint)) return;
  int point=int(tid)%npoint, world=int(tid)/npoint;
  if (selected[world]==0) return;
  int body=body_ids[point];
  int pose=(world*nbody+body)*3;
  float3 bp=float3(body_pos[pose],body_pos[pose+1],body_pos[pose+2]);
  int quat=(world*nbody+body)*4;
  float4 bq=float4(body_quat[quat],body_quat[quat+1],
                   body_quat[quat+2],body_quat[quat+3]);
  int lp=point*3;
  float3 offset=float3(local[lp],local[lp+1],local[lp+2]);
  float3 rotated=centered[point] ? float3(0.0f) : quat_rotate(bq,offset);
  float3 world_point=bp+rotated;
  int outp=(world*npoint+point)*3;
  xpos[outp]=world_point.x; xpos[outp+1]=world_point.y; xpos[outp+2]=world_point.z;
  float3 velocity=float3(0.0f);
  if (dims[6]!=0) {
    int cv=(world*nbody+body)*6;
    float3 omega=float3(cvel[cv],cvel[cv+1],cvel[cv+2]);
    float3 linear=float3(cvel[cv+3],cvel[cv+4],cvel[cv+5]);
    int center=(world*nbody+body)*3;
    float3 com=float3(root_com[center],root_com[center+1],root_com[center+2]);
    velocity=linear+cross(omega,world_point-com);
  }
  xvel[outp]=velocity.x; xvel[outp+1]=velocity.y; xvel[outp+2]=velocity.z;
  int jbase=(world*npoint+point)*3*max(nv,1);
  for(int d=0;d<3*max(nv,1);++d) jacobian[jbase+d]=0.0f;
  if(has_spatial!=0) {
    int sbase=(world*npoint+point)*6*max(nv,1);
    for(int d=0;d<6*max(nv,1);++d) spatial_jacobian[sbase+d]=0.0f;
  }
  int curr=body;
  while(curr>0 && curr<nbody) {
    int ja=body_jntadr[curr], jn=body_jntnum[curr];
    int curr_pose=(world*nbody+curr)*3;
    float3 curr_pos=float3(body_pos[curr_pose],body_pos[curr_pose+1],
                           body_pos[curr_pose+2]);
    int curr_quat=(world*nbody+curr)*4;
    float4 curr_q=float4(body_quat[curr_quat],body_quat[curr_quat+1],
                         body_quat[curr_quat+2],body_quat[curr_quat+3]);
    for(int j=ja;j<ja+jn;++j) {
      if(j<0 || j>=njnt) continue;
      int da=jnt_dofadr[j], typ=jnt_type[j];
      float3 axis=float3(0.0f), anchor=curr_pos;
      int axis_base=(world*njnt+j)*3;
      int anchor_base=(world*njnt+j)*3;
      if(typ==0) {
        for(int r=0;r<3;++r) {
          float3 unit=r==0?float3(1,0,0):(r==1?float3(0,1,0):float3(0,0,1));
          axis=quat_rotate(curr_q,unit);
          int col=da+3+r;
          if(col<nv) {
            int colbase=jbase+col;
            float3 lin=cross(axis,world_point-curr_pos);
            jacobian[colbase]=lin.x;
            jacobian[colbase+max(nv,1)]=lin.y;
            jacobian[colbase+2*max(nv,1)]=lin.z;
            if(has_spatial!=0) {
              int scol=(world*npoint+point)*6*max(nv,1)+col;
              spatial_jacobian[scol+3*max(nv,1)]=axis.x;
              spatial_jacobian[scol+4*max(nv,1)]=axis.y;
              spatial_jacobian[scol+5*max(nv,1)]=axis.z;
            }
          }
        }
        for(int r=0;r<3;++r) if(da+r<nv) {
          int col=jbase+da+r;
          jacobian[col+r*max(nv,1)]=1.0f;
        }
      } else {
        axis=float3(joint_axis[axis_base],joint_axis[axis_base+1],joint_axis[axis_base+2]);
        anchor=float3(joint_anchor[anchor_base],joint_anchor[anchor_base+1],joint_anchor[anchor_base+2]);
        if(typ==2) {
          if(da<nv) {
            int col=jbase+da;
            jacobian[col]=axis.x; jacobian[col+max(nv,1)]=axis.y;
            jacobian[col+2*max(nv,1)]=axis.z;
          }
        } else if(typ==1) {
          for(int r=0;r<3;++r) if(da+r<nv) {
            float3 unit=r==0?float3(1,0,0):(r==1?float3(0,1,0):float3(0,0,1));
            axis=quat_rotate(curr_q,unit);
            int col=jbase+da+r;
            float3 lin=cross(axis,world_point-anchor);
            jacobian[col]=lin.x; jacobian[col+max(nv,1)]=lin.y;
            jacobian[col+2*max(nv,1)]=lin.z;
            if(has_spatial!=0) {
              int scol=(world*npoint+point)*6*max(nv,1)+col;
              spatial_jacobian[scol+3*max(nv,1)]=axis.x;
              spatial_jacobian[scol+4*max(nv,1)]=axis.y;
              spatial_jacobian[scol+5*max(nv,1)]=axis.z;
            }
          }
        } else if(typ==3 && da<nv) {
          int col=jbase+da;
          float3 lin=cross(axis,world_point-anchor);
          jacobian[col]=lin.x; jacobian[col+max(nv,1)]=lin.y;
          jacobian[col+2*max(nv,1)]=lin.z;
          if(has_spatial!=0) {
            int scol=(world*npoint+point)*6*max(nv,1)+da;
            spatial_jacobian[scol+3*max(nv,1)]=axis.x;
            spatial_jacobian[scol+4*max(nv,1)]=axis.y;
            spatial_jacobian[scol+5*max(nv,1)]=axis.z;
          }
        }
      }
    }
    curr=body_parentid[curr];
  }
}

kernel void flex_recovery_update_edges(
    device const int* edge [[buffer(0)]],
    device const int* rowadr [[buffer(1)]],
    device const int* rownnz [[buffer(2)]],
    device const int* colind [[buffer(3)]],
    device const float* vertex_xpos [[buffer(4)]],
    device const float* vertex_xvel [[buffer(5)]],
    device const float* vertex_jac [[buffer(6)]],
    device const int* selected [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    device float* edge_length [[buffer(9)]],
    device float* edge_velocity [[buffer(10)]],
    device float* edge_dir [[buffer(11)]],
    device float* edge_jac [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nedge=dims[1], nv=dims[2], nvert=dims[3];
  if(tid>=uint(batch*nedge)) return;
  int e=int(tid)%nedge, world=int(tid)/nedge;
  if(selected[world]==0) return;
  int v0=edge[2*e],v1=edge[2*e+1];
  int p0=(world*nvert+v0)*3,p1=(world*nvert+v1)*3;
  float3 delta=float3(vertex_xpos[p1]-vertex_xpos[p0],
                      vertex_xpos[p1+1]-vertex_xpos[p0+1],
                      vertex_xpos[p1+2]-vertex_xpos[p0+2]);
  float len=length(delta), invlen=1.0f/max(len,1e-8f);
  float3 u=delta*invlen;
  int out=world*nedge+e;
  edge_length[out]=len;
  edge_dir[3*out]=u.x; edge_dir[3*out+1]=u.y; edge_dir[3*out+2]=u.z;
  float3 dv=float3(vertex_xvel[p1]-vertex_xvel[p0],
                   vertex_xvel[p1+1]-vertex_xvel[p0+1],
                   vertex_xvel[p1+2]-vertex_xvel[p0+2]);
  edge_velocity[out]=dot(u,dv);
  int support=rowadr[e], count=rownnz[e];
  for(int dof=0;dof<max(nv,1);++dof) {
    float value=0.0f;
    bool present=false;
    for(int k=0;k<count;++k) if(colind[support+k]==dof) present=true;
    if(present && dof<nv) {
      float3 j0=float3(vertex_jac[((world*nvert+v0)*3)*nv+dof],
                       vertex_jac[((world*nvert+v0)*3+1)*nv+dof],
                       vertex_jac[((world*nvert+v0)*3+2)*nv+dof]);
      float3 j1=float3(vertex_jac[((world*nvert+v1)*3)*nv+dof],
                       vertex_jac[((world*nvert+v1)*3+1)*nv+dof],
                       vertex_jac[((world*nvert+v1)*3+2)*nv+dof]);
      value=dot(u,j1-j0);
    }
    edge_jac[(world*nedge+e)*max(nv,1)+dof]=value;
  }
}

kernel void flex_recovery_copy_linear_spatial(
    device const int* selected [[buffer(0)]],
    device const float* linear [[buffer(1)]],
    device float* spatial [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], npoint=dims[1], nv=dims[2], stride=max(nv,1);
  int total=batch*npoint*3*stride;
  if(tid>=uint(total)) return;
  int dof=int(tid)%stride;
  int component=(int(tid)/stride)%3;
  int point=(int(tid)/(3*stride))%npoint;
  int world=int(tid)/(npoint*3*stride);
  if(selected[world]==0 || dof>=nv) return;
  int src=((world*npoint+point)*3+component)*stride+dof;
  int dst=((world*npoint+point)*6+component)*stride+dof;
  spatial[dst]=linear[src];
}

// Copy selected captured POS rows without touching healthy workspaces.
// dims = [batch, per_world_width, selected[B]].
kernel void flex_recovery_masked_copy(
    device const float* source [[buffer(0)]],
    device const int* selected [[buffer(1)]],
    constant int* dims [[buffer(2)]],
    device float* target [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1];
  if(tid>=uint(batch*width)) return;
  int world=int(tid)/width;
  if(selected[world]==0) return;
  target[tid]=source[tid];
}

// Selected-row generalized velocity from one packed per-world Jacobian.
// dims = [batch, output_lanes_per_world, nv, selected[B]].
kernel void flex_recovery_masked_matvec(
    device const float* jacobian [[buffer(0)]],
    device const float* qvel [[buffer(1)]],
    device const int* selected [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], width=dims[1], nv=dims[2], stride=max(nv,1);
  if(tid>=uint(batch*width)) return;
  int lane=int(tid)%width, world=int(tid)/width;
  if(selected[world]==0) return;
  float value=0.0f;
  for(int dof=0;dof<nv;++dof)
    value=fma(jacobian[(world*width+lane)*stride+dof],
              qvel[world*stride+dof],value);
  output[world*width+lane]=value;
}

kernel void flex_recovery_edge_force(
    device const float* length [[buffer(0)]],
    device const float* rate [[buffer(1)]],
    device const float* rest [[buffer(2)]],
    device const float* spring_coeff [[buffer(3)]],
    device const float* damper_coeff [[buffer(4)]],
    device const float* edge_jac [[buffer(5)]],
    device const int* selected [[buffer(6)]],
    constant int* dims [[buffer(7)]],
    device float* qfrc [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nedge=dims[1], nv=dims[2];
  bool spring=dims[3]!=0, damper=dims[4]!=0;
  if(tid>=uint(batch*max(nv,1))) return;
  int dof=int(tid)%max(nv,1), world=int(tid)/max(nv,1);
  if(selected[world]==0) return;
  float force=0.0f;
  if(dof<nv) for(int e=0;e<nedge;++e) {
    int ge=world*nedge+e;
    float tension=0.0f;
    if(spring) tension+=spring_coeff[e]*(length[ge]-rest[e]);
    if(damper) tension+=damper_coeff[e]*rate[ge];
    force-=tension*edge_jac[ge*max(nv,1)+dof];
  }
  qfrc[world*max(nv,1)+dof]=force;
}

static inline float4 quat_product(float4 a, float4 b) {
  return float4(a.x*b.x - dot(a.yzw, b.yzw),
                a.x*b.yzw + b.x*a.yzw + cross(a.yzw, b.yzw));
}

static inline float3x3 quat_matrix(float4 q) {
  float w=q.x, x=q.y, y=q.z, z=q.w;
  return float3x3(float3(1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)),
                  float3(2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)),
                  float3(2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)));
}

// Port of pinned mju_mat2Rot (Muller et al. quaternion polar iteration).
static inline float4 mat2quat_pinned(float3x3 F) {
  float4 q=float4(1,0,0,0);
  for (int it=0; it<48; ++it) {
    float3x3 R=quat_matrix(q);
    float3 omega=cross(R[0],F[0])+cross(R[1],F[1])+cross(R[2],F[2]);
    float den=abs(dot(R[0],F[0])+dot(R[1],F[1])+dot(R[2],F[2]))+1e-15f;
    omega/=den;
    float angle=length(omega);
    if (angle<1e-7f) break;
    float4 dq=float4(cos(0.5f*angle), sin(0.5f*angle)*omega/angle);
    q=normalize(quat_product(dq,q));
  }
  return q;
}

// Interpolated Q1/Q2 volume-material force from compiled flex_stiffness. One
// thread owns one FE, computes its corotational frame and nodal forces, then
// projects that element to all generalized coordinates without atomics.
kernel void flex_interp_force(
    device const float* node_xpos [[buffer(0)]],
    device const float* node_xvel [[buffer(1)]],
    device const float* node_rest [[buffer(2)]],
    device const float* node_jac [[buffer(3)]],
    device const int* elem_nodes [[buffer(4)]],
    device const float* elem_K [[buffer(5)]],
    device const float* shape_grad [[buffer(6)]],
    device const int* elem_meta [[buffer(7)]],
    device const float* flex_damping [[buffer(8)]],
    constant int* dims [[buffer(9)]],
    device float* elem_force [[buffer(10)]],
    device float* elem_node_force [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nv=dims[1], nelem=dims[2], nnode=dims[3];
  bool spring=dims[4]!=0, damper=dims[5]!=0;
  if (tid>=uint(batch*nelem)) return;
  int e=int(tid)%nelem, b=int(tid)/nelem;
  if (dims[6+b] == 0) return;
  int meta=5*e, npe=elem_meta[meta], ax0=elem_meta[meta+1];
  int ax1=elem_meta[meta+2], ax2=elem_meta[meta+3];
  int nd=3*npe;
  float3 X[27], V[27], R0[27];
  float Fraw[3][3]={{0}};
  for (int n=0;n<npe;++n) {
    int gi=elem_nodes[27*e+n];
    X[n]=float3(node_xpos[(b*nnode+gi)*3+0],node_xpos[(b*nnode+gi)*3+1],node_xpos[(b*nnode+gi)*3+2]);
    V[n]=float3(node_xvel[(b*nnode+gi)*3+0],node_xvel[(b*nnode+gi)*3+1],node_xvel[(b*nnode+gi)*3+2]);
    R0[n]=float3(node_rest[gi*3+0],node_rest[gi*3+1],node_rest[gi*3+2]);
    for (int d=0;d<3;++d) for (int k=0;k<3;++k)
      Fraw[d][k]+=X[n][d]*shape_grad[(e*27+n)*3+k];
  }
  float3x3 F;
  if (npe==8 || npe==27) {
    F=float3x3(float3(Fraw[0][0],Fraw[1][0],Fraw[2][0]),
               float3(Fraw[0][1],Fraw[1][1],Fraw[2][1]),
               float3(Fraw[0][2],Fraw[1][2],Fraw[2][2]));
  } else {
    float3 t0=float3(Fraw[0][0],Fraw[1][0],Fraw[2][0]);
    float3 t1=float3(Fraw[0][1],Fraw[1][1],Fraw[2][1]);
    float3 normal=cross(t0,t1);
    float3 cols[3]; cols[ax0]=t0; cols[ax1]=t1; cols[ax2]=normal;
    F=float3x3(cols[0],cols[1],cols[2]);
  }
  float4 q=mat2quat_pinned(F);
  float4 qc=float4(q.x,-q.yzw);
  float3 disp[27], vel[27], fnode[27];
  for (int n=0;n<npe;++n) {
    disp[n]=quat_rotate(qc,X[n])-R0[n];
    vel[n]=quat_rotate(qc,V[n]);
  }
  float localf[81]={0}, locald[81]={0};
  for (int r=0;r<nd;++r) for (int c=0;c<nd;++c) {
    float k=elem_K[(e*81+r)*81+c];
    if (spring) localf[r]+=k*disp[c/3][c%3];
    if (damper) locald[r]+=k*vel[c/3][c%3];
  }
  float dmp=flex_damping[e];
  for (int n=0;n<npe;++n) {
    float3 local= float3(localf[3*n],localf[3*n+1],localf[3*n+2])
                 +dmp*float3(locald[3*n],locald[3*n+1],locald[3*n+2]);
    fnode[n]=quat_rotate(q,local);
    int gi=elem_nodes[27*e+n];
    for (int d=0;d<3;++d) elem_node_force[(b*nelem+e)*81+3*n+d]=fnode[n][d];
  }
  for (int dof=0;dof<nv;++dof) {
    float qf=0;
    for (int n=0;n<npe;++n) {
      int gi=elem_nodes[27*e+n];
      for (int d=0;d<3;++d)
        qf+=fnode[n][d]*node_jac[((b*nnode+gi)*3+d)*nv+dof];
    }
    elem_force[(b*nelem+e)*nv+dof]=qf;
  }
}

static inline float flex_phi(float s, int i, int order) {
  if (order==1) return i==0 ? 1.0f-s : s;
  if (i==0) return 2*s*s-3*s+1;
  if (i==1) return 4*(s-s*s);
  return 2*s*s-s;
}

static inline float flex_dphi(float s, int i, int order) {
  if (order==1) return i==0 ? -1.0f : 1.0f;
  if (i==0) return 4*s-3;
  if (i==1) return 4*(1-2*s);
  return 4*s-1;
}

struct ShellFace {
  float3 x[9];
  float3 t0;
  float3 t1;
  float3 nraw;
  float3 normal;
  float area_scale;
  float3x3 F;
  float4 q_inverse;
};

static inline ShellFace shell_face(
    device const float* node_xpos, device const int* face_nodes,
    device const int* face_axes, device const int* face_order,
    int batch, int nnode, int face) {
  ShellFace out;
  int order=face_order[face], npe=(order+1)*(order+1);
  int ax0=face_axes[3*face], ax1=face_axes[3*face+1], axn=face_axes[3*face+2];
  float3 Fcol[3]={float3(0),float3(0),float3(0)};
  float3 center_t0=float3(0),center_t1=float3(0);
  for (int i=0;i<=order;++i) for (int j=0;j<=order;++j) {
    int n=i*(order+1)+j;
    int gi=face_nodes[9*face+n];
    float3 x=float3(node_xpos[(batch*nnode+gi)*3],
                    node_xpos[(batch*nnode+gi)*3+1],
                    node_xpos[(batch*nnode+gi)*3+2]);
    out.x[n]=x;
    float g0=flex_dphi(0.5f,i,order)*flex_phi(0.5f,j,order);
    float g1=flex_phi(0.5f,i,order)*flex_dphi(0.5f,j,order);
    center_t0+=g0*x; center_t1+=g1*x;
  }
  Fcol[ax0]=center_t0; Fcol[ax1]=center_t1;
  Fcol[axn]=cross(center_t0,center_t1);
  out.F=float3x3(Fcol[0],Fcol[1],Fcol[2]);
  float4 q=mat2quat_pinned(out.F);
  out.q_inverse=float4(q.x,-q.yzw);
  return out;
}

static inline float3 face_normal_at(
    thread const ShellFace& face, float s0, float s1, int order,
    int axis0, int axis1, thread float3& tangent0,
    thread float3& tangent1, thread float grad[9][2]) {
  tangent0=float3(0); tangent1=float3(0);
  int idx=0;
  for (int i=0;i<=order;++i) for (int j=0;j<=order;++j) {
    float g0=flex_dphi(s0,i,order)*flex_phi(s1,j,order);
    float g1=flex_phi(s0,i,order)*flex_dphi(s1,j,order);
    grad[idx][0]=g0; grad[idx][1]=g1;
    tangent0+=g0*face.x[idx]; tangent1+=g1*face.x[idx]; idx++;
  }
  return cross(tangent0,tangent1);
}

// Pinned mj_flexPassiveBendInterp CR normal-jump law. One thread owns one
// compiled shell edge and emits its generalized-force contribution.
kernel void flex_shell_bend_force(
    device const float* node_xpos [[buffer(0)]],
    device const float* node_jac [[buffer(1)]],
    device const int* face_nodes [[buffer(2)]],
    device const int* face_axes [[buffer(3)]],
    device const int* face_order [[buffer(4)]],
    device const float* records [[buffer(5)]],
    constant int* dims [[buffer(6)]],
    device float* elem_force [[buffer(7)]],
    device float* elem_node_force [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0],nv=dims[1],nedge=dims[2],nnode=dims[3];
  if (tid>=uint(batch*nedge)) return;
  int e=int(tid)%nedge,b=int(tid)/nedge;
  if (dims[4+b] == 0) return;
  int rec=10*e;
  int fa=int(records[rec]),fb=int(records[rec+1]);
  float2 local_a=float2(records[rec+2],records[rec+3]);
  float2 local_b=float2(records[rec+4],records[rec+5]);
  float stiffness=records[rec+6];
  float3 dn0=float3(records[rec+7],records[rec+8],records[rec+9]);
  int order_a=face_order[fa],order_b=face_order[fb];
  int ax0a=face_axes[3*fa],ax1a=face_axes[3*fa+1];
  int ax0b=face_axes[3*fb],ax1b=face_axes[3*fb+1];
  ShellFace A=shell_face(node_xpos,face_nodes,face_axes,face_order,b,nnode,fa);
  ShellFace B=shell_face(node_xpos,face_nodes,face_axes,face_order,b,nnode,fb);
  float ga[9][2],gb[9][2];
  float3 ta0,ta1,tb0,tb1;
  float3 na_raw=face_normal_at(A,local_a.x,local_a.y,order_a,ax0a,ax1a,ta0,ta1,ga);
  float3 nb_raw=face_normal_at(B,local_b.x,local_b.y,order_b,ax0b,ax1b,tb0,tb1,gb);
  float lena=max(length(na_raw),1e-15f),lenb=max(length(nb_raw),1e-15f);
  float3 na=na_raw/lena,nb=nb_raw/lenb;
  float4 qa=A.q_inverse,qb=B.q_inverse;
  if (dot(qa,qb)<0) qb=-qb;
  float4 qavg=normalize(qa+qb);
  qavg=float4(qavg.x,-qavg.yzw);
  float3 residual=na-nb-quat_rotate(qavg,dn0);
  float3 wa=(residual-na*dot(na,residual))/lena;
  float3 wb=(residual-nb*dot(nb,residual))/lenb;
  float3 wa_t2=cross(wa,ta1),wa_t1=cross(wa,ta0);
  float3 wb_t2=cross(wb,tb1),wb_t1=cross(wb,tb0);
  for (int n=0;n<9;++n) {
    float3 fna=float3(0),fnb=float3(0);
    if (n<(order_a+1)*(order_a+1))
      fna=stiffness*(ga[n][0]*wa_t2-ga[n][1]*wa_t1);
    if (n<(order_b+1)*(order_b+1))
      fnb=-stiffness*(gb[n][0]*wb_t2-gb[n][1]*wb_t1);
    for (int d=0;d<3;++d) {
      elem_node_force[(b*nedge+e)*54+3*n+d]=fna[d];
      elem_node_force[(b*nedge+e)*54+27+3*n+d]=fnb[d];
    }
  }
  for (int dof=0;dof<nv;++dof) {
    float qf=0;
    for (int n=0;n<(order_a+1)*(order_a+1);++n) {
      int gi=face_nodes[9*fa+n];
      float3 fn=stiffness*(ga[n][0]*wa_t2-ga[n][1]*wa_t1);
      for (int d=0;d<3;++d)
        qf+=fn[d]*node_jac[((b*nnode+gi)*3+d)*nv+dof];
    }
    for (int n=0;n<(order_b+1)*(order_b+1);++n) {
      int gi=face_nodes[9*fb+n];
      float3 fn=-stiffness*(gb[n][0]*wb_t2-gb[n][1]*wb_t1);
      for (int d=0;d<3;++d)
        qf+=fn[d]*node_jac[((b*nnode+gi)*3+d)*nv+dof];
    }
    elem_force[(b*nedge+e)*nv+dof]=qf;
  }
}

// Retain only the columns in MuJoCo's compiled flexedge_J CSR.  One thread
// owns an entire edge row and walks its sorted sparse column list, so runtime
// metadata stays O(nflexedge + nnz), never O(nflexedge * nv).
kernel void flex_edge_jacobian_compiled_support(
    device const int* rowadr [[buffer(0)]],
    device const int* rownnz [[buffer(1)]],
    device const int* colind [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    device float* edge_jac [[buffer(4)]],
    device float* edge_velocity [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0];
  int nedge = dims[1];
  int nv = dims[2];
  int nnz = dims[3];
  if (tid >= uint(batch * nedge)) return;
  if (nv <= 0) return;
  int edge = int(tid) % nedge;
  int world = int(tid) / nedge;
  int adr = rowadr[edge];
  int count = rownnz[edge];
  if (adr < 0 || count < 0 || adr + count > nnz) return;
  if (count == 0) edge_velocity[world * nedge + edge] = 0.0f;
  int cursor = 0;
  int base = (world * nedge + edge) * nv;
  for (int dof = 0; dof < nv; ++dof) {
    if (cursor < count && colind[adr + cursor] == dof) {
      ++cursor;
    } else {
      edge_jac[base + dof] = 0.0f;
    }
  }
}
