// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline float3 rotate_q(float4 q, float3 v) {
  float3 u = q.yzw;
  return v + 2.0f * cross(u, cross(u, v) + q.x * v);
}
inline float3 cross3(float3 a, float3 b) { return cross(a, b); }

// One fixed work item per eligible geometry pair. All outputs are initialized,
// including inactive pairs, so masks are safe for downstream kernels.
kernel void contact_normal(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* body_pos [[buffer(2)]],
    device const float* body_quat [[buffer(3)]],
    device const float* anchors [[buffer(4)]],
    device const float* axes [[buffer(5)]],
    device const float* qvel [[buffer(6)]],
    device const int* geom1 [[buffer(7)]], device const int* geom2 [[buffer(8)]],
    device const float* radius1 [[buffer(9)]], device const float* radius2 [[buffer(10)]],
    device const float* margin [[buffer(11)]], device const float* gap [[buffer(12)]],
    device const float* solref [[buffer(13)]], device const float* solimp [[buffer(14)]],
    device const int* geom_bodyid [[buffer(15)]], device const int* body_parentid [[buffer(16)]],
    device const int* body_jntadr [[buffer(17)]], device const int* body_jntnum [[buffer(18)]],
    device const int* jnt_type [[buffer(19)]], device const int* jnt_dofadr [[buffer(20)]],
    device const float* body_invweight0 [[buffer(21)]],
    device float* row_data [[buffer(22)]], device float* frame [[buffer(23)]],
    device float* jacobian [[buffer(24)]],
    constant int* dims [[buffer(25)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0], nc = dims[1], batch = dims[2], nbody = dims[3], njnt = dims[4], ngeom=dims[5];
  int world = int(tid) / max(nc, 1), slot = int(tid) % max(nc, 1);
  if (uint(world) >= uint(batch) || slot >= nc) return;
  int out = world * nc + slot;
  int jbase = out * nv;
  int rb=out*6, fb=out*6;
  for (int k=0;k<6;++k) row_data[rb+k]=0.0f;
  for (int k=0; k<nv; ++k) jacobian[jbase+k] = 0.0f;
  for (int k=0; k<6; ++k) frame[fb+k]=0.0f;

  int a=geom1[slot], b=geom2[slot];
  int go=world*ngeom;
  float3 pa=float3(geom_pos[(go+a)*3],geom_pos[(go+a)*3+1],geom_pos[(go+a)*3+2]);
  float3 pb=float3(geom_pos[(go+b)*3],geom_pos[(go+b)*3+1],geom_pos[(go+b)*3+2]);
  float4 qa=float4(geom_quat[(go+a)*4],geom_quat[(go+a)*4+1],geom_quat[(go+a)*4+2],geom_quat[(go+a)*4+3]);
  float4 qb=float4(geom_quat[(go+b)*4],geom_quat[(go+b)*4+1],geom_quat[(go+b)*4+2],geom_quat[(go+b)*4+3]);
  float d=0.0f; float3 n=float3(0.0f), point=float3(0.0f);
  // MuJoCo plane:sphere normal points from geom1 toward geom2.
  if (radius1[slot] < 0.0f) {
    float3 pn=rotate_q(qa,float3(0,0,1));
    d=dot(pb-pa,pn)-radius2[slot]; n=pn;
    point=pb-n*(d*0.5f+radius2[slot]);
  } else if (radius2[slot] < 0.0f) {
    float3 pn=rotate_q(qb,float3(0,0,1));
    d=dot(pa-pb,pn)-radius1[slot]; n=-pn;
    point=pa-pn*(d*0.5f+radius1[slot]);
  } else {
    float3 delta=pb-pa; float distance=length(delta);
    if (!(distance > 1e-12f)) return;
    n=delta/distance; d=distance-radius1[slot]-radius2[slot];
    point=pa+n*(radius1[slot]+0.5f*d);
  }
  float m=margin[slot], g=gap[slot];
  if (d > m+g) return;

  int ba=geom_bodyid[a], bb=geom_bodyid[b];
  int bo=world*nbody, jo=world*njnt;
  float3 rel=point;
  // Build the relative point-velocity Jacobian by walking each body's joint
  // ancestry. Hinge/slide and ball/free blocks use generalized-coordinate axes.
  for (int side=0; side<2; ++side) {
    int body=side==0 ? ba : bb; float sign=side==0 ? -1.0f : 1.0f;
    while (body>0 && body<nbody) {
      int ja=body_jntadr[body], jn=body_jntnum[body];
      for (int jj=0; jj<jn; ++jj) {
        int j=ja+jj, da=jnt_dofadr[j], typ=jnt_type[j];
        int nd=typ==0 ? 6 : typ==1 ? 3 : 1;
        for (int q=0; q<nd; ++q) {
          int dof=da+q; if (dof<0 || dof>=nv) continue;
          float3 col=float3(0.0f);
          if (typ==2) { // slide
            col=float3(axes[(jo+j)*3],axes[(jo+j)*3+1],axes[(jo+j)*3+2]);
          } else if (typ==3) { // hinge
            float3 axis=float3(axes[(jo+j)*3],axes[(jo+j)*3+1],axes[(jo+j)*3+2]);
            col=cross3(axis,rel-float3(anchors[(jo+j)*3],anchors[(jo+j)*3+1],anchors[(jo+j)*3+2]));
          } else if (typ==0 || typ==1) { // free/ball rotational axes are body-local
            if (typ==0 && q<3) col=float3(q==0, q==1, q==2);
            else {
              int qrot=typ==0 ? q-3 : q;
              float4 bq=float4(body_quat[(bo+body)*4],body_quat[(bo+body)*4+1],body_quat[(bo+body)*4+2],body_quat[(bo+body)*4+3]);
              float3 axis=rotate_q(bq,float3(qrot==0,qrot==1,qrot==2));
              float3 pivot=typ==0 ? float3(body_pos[(bo+body)*3],body_pos[(bo+body)*3+1],body_pos[(bo+body)*3+2]) : float3(anchors[(jo+j)*3],anchors[(jo+j)*3+1],anchors[(jo+j)*3+2]);
              col=cross3(axis,rel-pivot);
            }
          }
          jacobian[jbase+dof] += sign*dot(n,col);
        }
      }
      body=body_parentid[body];
    }
  }
  float vel=0.0f;
  for (int i=0;i<nv;++i) vel+=jacobian[jbase+i]*qvel[world*nv+i];
  float d0=solimp[slot*5], d1=solimp[slot*5+1], width=solimp[slot*5+2];
  float mid=solimp[slot*5+3], power=solimp[slot*5+4], impedance;
  float x=width>1e-15f ? abs((d-m)/width) : 1.0f;
  if (d0==d1 || width<=1e-15f) impedance=0.5f*(d0+d1);
  else if (x<=0.0f) impedance=d0;
  else if (x>=1.0f) impedance=d1;
  else {
    float y;
    if (power==1.0f) y=x;
    else if (x<=mid) y=pow(x,power)/pow(mid,power-1.0f);
    else y=1.0f-pow(1.0f-x,power)/pow(1.0f-mid,power-1.0f);
    impedance=d0+y*(d1-d0);
  }
  float r0=solref[slot*2], r1=solref[slot*2+1];
  float K, B;
  if (r0>0.0f) K=1.0f/max(1e-15f,d1*d1*r0*r0*r1*r1);
  else K=-r0/max(1e-15f,d1*d1);
  if (r1>0.0f) B=2.0f/max(1e-15f,d1*r0);
  else B=-r1/max(1e-15f,d1);
  // Euler's classical constraint reference is explicit in position; timestep
  // safety is applied while lowering standard solref, as in getsolparam.
  float aref=-B*vel-K*impedance*(d-m);
  row_data[rb]=d<=m ? 1.0f : 0.0f;
  row_data[rb+1]=d; row_data[rb+2]=vel;
  float diag_approx=body_invweight0[ba*2]+body_invweight0[bb*2];
  row_data[rb+3]=aref; row_data[rb+4]=impedance;
  row_data[rb+5]=diag_approx;
  for (int k=0;k<3;++k) { frame[fb+k]=n[k]; frame[fb+3+k]=point[k]; }
}

// Dense coupled projected solve for normal-only contacts. Each world factors
// M once, forms W=J M^-1 J', then performs deterministic PGS on the unilateral
// rows. This uses the same regularized complementarity form as MuJoCo's normal
// contact rows; the fixed iteration cap is part of this stage's profile.
kernel void solve_normal_contacts(
    device const float* mass [[buffer(0)]], device const float* free_acc [[buffer(1)]],
    device const float* J [[buffer(2)]], device const float* row_data [[buffer(3)]],
    device float* force [[buffer(4)]], device float* qacc [[buffer(5)]],
    device int* status [[buffer(6)]], constant int* dims [[buffer(7)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nc=dims[1], batch=dims[2]; if (world>=uint(batch)) return;
  status[world]=0;
  int mb=world*nv*nv, jb=world*nc*nv;
  for (int i=0;i<nc;++i) force[world*nc+i]=0.0f;
  for (int i=0;i<nv;++i) qacc[world*nv+i]=free_acc[world*nv+i];
  if (nc==0 || nv==0) return;
  // Cholesky of the dense symmetric positive-definite generalized mass.
  thread float L[32*32]; if (nv>32) { status[world]=1; return; }
  for (int i=0;i<nv*nv;++i) L[i]=mass[mb+i];
  for (int p=0;p<nv;++p) {
    float diag=L[p*nv+p]; for (int k=0;k<p;++k) diag-=L[p*nv+k]*L[p*nv+k];
    if (!(diag>0.0f) || !isfinite(diag)) { status[world]=2; return; }
    L[p*nv+p]=sqrt(diag);
    for (int r=p+1;r<nv;++r) {
      float v=L[r*nv+p]; for (int k=0;k<p;++k) v-=L[r*nv+k]*L[p*nv+k];
      L[r*nv+p]=v/L[p*nv+p];
    }
  }
  thread float response[16*32]; thread float rhs[32];
  if (nc>16) { status[world]=1; return; }
  for (int c=0;c<nc;++c) {
    for (int r=0;r<nv;++r) rhs[r]=J[jb+c*nv+r];
    // Solve M x = J' using L L'.
    for (int r=0;r<nv;++r) { for (int k=0;k<r;++k) rhs[r]-=L[r*nv+k]*rhs[k]; rhs[r]/=L[r*nv+r]; }
    for (int r=nv-1;r>=0;--r) { for (int k=r+1;k<nv;++k) rhs[r]-=L[k*nv+r]*rhs[k]; rhs[r]/=L[r*nv+r]; }
    for (int r=0;r<nv;++r) response[c*nv+r]=rhs[r];
  }
  thread float W[16*16];
  for (int i=0;i<nc*nc;++i) W[i]=0.0f;
  for (int a=0;a<nc;++a) for (int b=0;b<nc;++b) {
    float v=0.0f; for (int k=0;k<nv;++k) v+=J[jb+a*nv+k]*response[b*nv+k];
    W[a*nc+b]=v;
  }
  // MuJoCo's classic Euler contact diagonal approximation uses the sum of
  // translational inverse weights of the two contacting bodies.
  thread float R[16];
  for (int c=0;c<nc;++c) {
    int rb=(world*nc+c)*6;
    float imp=clamp(row_data[rb+4],1e-6f,0.999999f);
    float diag_approx=max(row_data[rb+5],1e-15f);
    R[c]=max(1e-15f,(1.0f-imp)*diag_approx/imp);
  }
  // Recompute current J*qacc then do projected Gauss-Seidel on lambda.
  bool converged=false;
  for (int sweep=0;sweep<48;++sweep) {
    float max_delta=0.0f;
    for (int c=0;c<nc;++c) {
      if (row_data[(world*nc+c)*6]<0.5f) continue;
      float ja=0.0f; for (int k=0;k<nv;++k) ja+=J[jb+c*nv+k]*qacc[world*nv+k];
      float old=force[world*nc+c];
      float ref=row_data[(world*nc+c)*6+3];
      float next=max(0.0f,old+(ref-ja-R[c]*old)/max(1e-15f,W[c*nc+c]+R[c]));
      float delta=next-old; max_delta=max(max_delta,abs(delta)); force[world*nc+c]=next;
      for (int k=0;k<nv;++k) qacc[world*nv+k]+=response[c*nv+k]*delta;
    }
    if (max_delta<1e-5f) { converged=true; break; }
  }
  if (!converged) status[world]=3;
}
