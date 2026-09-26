// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline float impedance_at(device const float* imp, int index, float pos, float margin) {
  float d0=imp[index*5], d1=imp[index*5+1], width=imp[index*5+2];
  float mid=imp[index*5+3], power=imp[index*5+4];
  if (d0==d1 || width<=1e-15f) return 0.5f*(d0+d1);
  float x=abs((pos-margin)/width);
  if (x>=1.0f) return d1;
  if (x<=0.0f) return d0;
  float y=power==1.0f ? x : (x<=mid ? pow(x,power)/pow(mid,power-1.0f)
      : 1.0f-pow(1.0f-x,power)/pow(1.0f-mid,power-1.0f));
  return d0+y*(d1-d0);
}
inline void reference_params(device const float* solref, device const float* solimp,
    int index, float pos, float margin, float vel, float diag, bool friction,
    float timestep, bool refsafe, thread float& compliance, thread float& aref) {
  float width=max(1e-15f,solimp[index*5+1]);
  float impedance=impedance_at(solimp,index,pos,margin);
  float r0=solref[index*2], r1=solref[index*2+1];
  if (refsafe && r0>0.0f) r0=max(r0,2.0f*timestep);
  float K=r0>0.0f ? 1.0f/max(1e-15f,width*width*r0*r0*r1*r1)
                    : -r0/max(1e-15f,width*width);
  float B=r1>0.0f ? 2.0f/max(1e-15f,width*r0) : -r1/width;
  if (friction) K=0.0f;
  compliance=max(1e-15f,(1.0f-impedance)*diag/impedance);
  aref=-B*vel-K*impedance*(pos-margin);
}

// Each thread solves one independent world. Fixed row layout: polynomial joint
// equalities, DOF frictionloss, then lower/upper limit slots for each joint.
// Dense Cholesky solves M^{-1}J' and M^{-1}qfrc; PGS projects the joint rows
// onto equality, friction-box, and limit unilateral sets respectively.
kernel void solve_joint_constraints(
    device const float* mass [[buffer(0)]], device const float* qfrc [[buffer(1)]],
    device const float* qpos [[buffer(2)]], device const float* qvel [[buffer(3)]],
    device const int* eq_active [[buffer(4)]],
    device const int* joint_qadr [[buffer(5)]], device const float* qpos0 [[buffer(6)]],
    device const int* joint_dadr [[buffer(7)]], device const uchar* joint_limited [[buffer(8)]],
    device const float* joint_range [[buffer(9)]], device const float* joint_margin [[buffer(10)]],
    device const float* joint_solref [[buffer(11)]], device const float* joint_solimp [[buffer(12)]],
    device const float* frictionloss [[buffer(13)]], device const float* invweight [[buffer(14)]],
    device const float* dof_solref [[buffer(15)]], device const float* dof_solimp [[buffer(16)]],
    device const int* eq_obj1 [[buffer(17)]], device const int* eq_obj2 [[buffer(18)]],
    device const float* eq_data [[buffer(19)]], device const float* eq_solref [[buffer(20)]],
    device const float* eq_solimp [[buffer(21)]], constant int* dims [[buffer(22)]],
    constant float* timestep [[buffer(23)]], device float* out_force [[buffer(24)]],
    device float* out_acc [[buffer(25)]], device int* out_status [[buffer(26)]],
    device float* out_residual [[buffer(27)]], device int* out_iterations [[buffer(28)]],
    uint world [[thread_position_in_grid]]) {
  int nq=dims[0], nv=dims[1], nj=dims[2], neq=dims[3], nr=dims[4], batch=dims[5];
  int flags=dims[6]; bool refsafe=dims[7]!=0; int maxiter=dims[8];
  if (world>=uint(batch)) return;
  int mb=world*nv*nv, qb=world*nv, pb=world*nq;
  out_status[world]=0; out_residual[world]=0.0f; out_iterations[world]=0;
  for (int i=0;i<nv;++i) { out_force[qb+i]=0.0f; out_acc[qb+i]=0.0f; }
  if (nv==0) return;
  thread float L[24*24]; thread float J[32*24]; thread float Z[32*24];
  thread float R[32]; thread float ar[32]; thread float lo[32]; thread float hi[32];
  thread float lam[32]; thread float rhs[32]; thread float y[24]; thread float x[24];
  thread bool enabled[32];
  if (nv>24 || nr>32) { out_status[world]=2; return; }
  for (int i=0;i<nr*nv;++i) { J[i]=0.0f; Z[i]=0.0f; }
  for (int i=0;i<nr;++i) { R[i]=1.0f; ar[i]=0.0f; lo[i]=0.0f; hi[i]=0.0f; lam[i]=0.0f; enabled[i]=false; }
  if ((flags & 1)==0) {
    for (int e=0;e<neq;++e) {
      if ((flags&2)!=0 || eq_active[world*neq+e]==0) continue;
      int j1=eq_obj1[e], j2=eq_obj2[e], q1=joint_qadr[j1], d1=joint_dadr[j1];
      float value1=qpos[pb+q1], ref1=qpos0[q1];
      float data0=eq_data[e*11];
      float pos=value1-ref1-data0, vel=qvel[qb+d1], diag=invweight[d1];
      if (j2>=0) {
        int q2=joint_qadr[j2], d2=joint_dadr[j2];
        float dif=qpos[pb+q2]-qpos0[q2]; float poly=0.0f, deriv=0.0f, power=dif;
        for (int k=0;k<4;++k) { poly+=eq_data[e*11+k+1]*power; deriv+=(k+1)*eq_data[e*11+k+1]*pow(dif,float(k)); power*=dif; }
        pos-=poly; J[e*nv+d1]=1.0f; J[e*nv+d2]=-deriv;
        vel-=deriv*qvel[qb+d2]; diag+=invweight[d2];
      } else J[e*nv+d1]=1.0f;
      reference_params(eq_solref,eq_solimp,e,pos,0.0f,vel,diag,false,timestep[0],refsafe,R[e],ar[e]);
      lo[e]=-INFINITY; hi[e]=INFINITY; enabled[e]=true;
    }
    for (int d=0;d<nv;++d) {
      int row=neq+d; float loss=frictionloss[d];
      if ((flags&4)!=0 || loss<=0.0f) continue;
      J[row*nv+d]=1.0f;
      reference_params(dof_solref,dof_solimp,d,0.0f,0.0f,qvel[qb+d],invweight[d],true,timestep[0],refsafe,R[row],ar[row]);
      lo[row]=-loss; hi[row]=loss; enabled[row]=true;
    }
    for (int j=0;j<nj;++j) {
      if (joint_limited[j]==0) continue;
      int d=joint_dadr[j], q=joint_qadr[j];
      for (int s=0;s<2;++s) {
        int row=neq+nv+2*j+s; float side=s==0 ? -1.0f : 1.0f;
        float value=qpos[pb+q], range=joint_range[j*2+s], dist=side*(range-value), margin=joint_margin[j];
        if ((flags&8)!=0 || dist>=margin) continue;
        J[row*nv+d]=-side;
        reference_params(joint_solref,joint_solimp,j,dist,margin,J[row*nv+d]*qvel[qb+d],invweight[d],false,timestep[0],refsafe,R[row],ar[row]);
        lo[row]=0.0f; hi[row]=INFINITY; enabled[row]=true;
      }
    }
  }
  // Cholesky factorization of the dense generalized mass matrix.
  for (int i=0;i<nv;++i) for (int j=0;j<nv;++j) L[i*nv+j]=0.0f;
  for (int i=0;i<nv;++i) for (int j=0;j<=i;++j) {
    float v=mass[mb+i*nv+j]; for (int k=0;k<j;++k) v-=L[i*nv+k]*L[j*nv+k];
    if (i==j) { if (!(v>1e-12f)) { out_status[world]=2; return; } L[i*nv+j]=sqrt(v); }
    else L[i*nv+j]=v/L[j*nv+j];
  }
  // Solve M q0 = qfrc, then M z_i = J_i' for each constraint.
  for (int i=0;i<nv;++i) {
    float v=qfrc[qb+i]; for (int k=0;k<i;++k) v-=L[i*nv+k]*y[k]; y[i]=v/L[i*nv+i];
  }
  for (int i=nv-1;i>=0;--i) { float v=y[i]; for(int k=i+1;k<nv;++k)v-=L[k*nv+i]*x[k]; x[i]=v/L[i*nv+i]; }
  for (int i=0;i<nv;++i) out_acc[qb+i]=x[i];
  for (int row=0;row<nr;++row) if (enabled[row]) {
    for (int i=0;i<nv;++i) { float v=J[row*nv+i]; for(int k=0;k<i;++k)v-=L[i*nv+k]*y[k]; y[i]=v/L[i*nv+i]; }
    for (int i=nv-1;i>=0;--i) { float v=y[i]; for(int k=i+1;k<nv;++k)v-=L[k*nv+i]*x[k]; x[i]=v/L[i*nv+i]; }
    for(int i=0;i<nv;++i) Z[row*nv+i]=x[i];
  }
  for (int row=0;row<nr;++row) {
    float v=ar[row]; for(int i=0;i<nv;++i)v-=J[row*nv+i]*out_acc[qb+i]; rhs[row]=enabled[row]?v:0.0f;
  }
  // Projected Gauss-Seidel on W = J M^-1 J' + diag(R).
  float res=INFINITY;
  for (int it=0;it<maxiter;++it) {
    for (int row=0;row<nr;++row) {
      if (!enabled[row]) { lam[row]=0.0f; continue; }
      float diag=R[row]; for(int i=0;i<nv;++i)diag+=J[row*nv+i]*Z[row*nv+i];
      if (!(diag>1e-12f)) { out_status[world]=2; return; }
      float v=rhs[row];
      for(int col=0;col<nr;++col) if(enabled[col] && col!=row)
        for(int i=0;i<nv;++i)v-=J[row*nv+i]*Z[col*nv+i]*lam[col];
      lam[row]=clamp(v/diag,lo[row],hi[row]);
    }
    res=0.0f;
    for(int row=0;row<nr;++row) if(enabled[row]) {
      float grad=-rhs[row]; for(int col=0;col<nr;++col) if(enabled[col])
        for(int i=0;i<nv;++i) grad+=J[row*nv+i]*Z[col*nv+i]*lam[col];
      grad+=R[row]*lam[row];
      float diag=R[row]; for(int i=0;i<nv;++i)diag+=J[row*nv+i]*Z[row*nv+i];
      float proj=clamp(lam[row]-grad/diag,lo[row],hi[row]); res=max(res,abs(proj-lam[row]));
    }
    out_iterations[world]=it+1; if(res<=2e-6f)break;
  }
  out_residual[world]=res; if(res>2e-6f)out_status[world]=1;
  for(int row=0;row<nr;++row) if(enabled[row]) for(int i=0;i<nv;++i)out_force[qb+i]+=J[row*nv+i]*lam[row];
  for(int i=0;i<nv;++i) {
    float v=out_force[qb+i]; for(int k=0;k<i;++k)v-=L[i*nv+k]*y[k]; y[i]=v/L[i*nv+i];
  }
  for(int i=nv-1;i>=0;--i) { float v=y[i]; for(int k=i+1;k<nv;++k)v-=L[k*nv+i]*x[k]; x[i]=v/L[i*nv+i]; }
  for(int i=0;i<nv;++i)out_acc[qb+i]+=x[i];
}
