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

static inline float3 quat_rotate(float4 q, float3 v) {
  float3 qv = q.yzw;
  float3 t = 2.0f * cross(qv, v);
  return v + q.x * t + cross(qv, t);
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
  if (tid>=uint(batch*nelem)) return;
  int e=int(tid)%nelem, b=int(tid)/nelem;
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
    localf[r]+=k*disp[c/3][c%3];
    locald[r]+=k*vel[c/3][c%3];
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
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0],nv=dims[1],nedge=dims[2],nnode=dims[3];
  if (tid>=uint(batch*nedge)) return;
  int e=int(tid)%nedge,b=int(tid)/nedge;
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
