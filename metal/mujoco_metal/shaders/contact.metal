// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
#include <metal_stdlib>
using namespace metal;

inline float3 rotate_q(float4 q, float3 v) {
  float3 u = q.yzw;
  return v + 2.0f * cross(u, cross(u, v) + q.x * v);
}
inline float3 cross3(float3 a, float3 b) { return cross(a, b); }

// Solve the local nonnegative convex QP for one contact's (at most four)
// pyramidal edge forces by enumerating its 16 possible active sets. Returns
// the best scaled feasibility error and writes that candidate to solution.
inline float solve_contact_block(
    thread const float* A, thread const float* b, int n,
    thread float* solution) {
  float best_error=1e30f, best_objective=1e30f;
  for (int i=0;i<4;++i) solution[i]=0.0f;
  for (int set=0;set<(1<<n);++set) {
    thread int ids[4]; int nactive=0;
    for (int i=0;i<n;++i) if (set & (1<<i)) ids[nactive++]=i;
    thread float mat[16], rhs[4], x[4], candidate[4];
    for (int i=0;i<16;++i) mat[i]=0.0f;
    for (int i=0;i<4;++i) { rhs[i]=0.0f; x[i]=0.0f; candidate[i]=0.0f; }
    for (int i=0;i<nactive;++i) {
      rhs[i]=-b[ids[i]];
      for (int j=0;j<nactive;++j)
        mat[i*4+j]=A[ids[i]*4+ids[j]];
    }
    bool valid=true;
    // Partial-pivoted Gaussian elimination on the active principal block.
    for (int p=0;p<nactive;++p) {
      int pivot_row=p; float pivot_abs=abs(mat[p*4+p]);
      for (int r=p+1;r<nactive;++r) {
        float value=abs(mat[r*4+p]);
        if (value>pivot_abs) { pivot_abs=value; pivot_row=r; }
      }
      if (!(pivot_abs>1e-15f) || !isfinite(pivot_abs)) { valid=false; break; }
      if (pivot_row!=p) {
        for (int j=0;j<nactive;++j) {
          float tmp=mat[p*4+j]; mat[p*4+j]=mat[pivot_row*4+j]; mat[pivot_row*4+j]=tmp;
        }
        float tmp=rhs[p]; rhs[p]=rhs[pivot_row]; rhs[pivot_row]=tmp;
      }
      float pivot=mat[p*4+p];
      for (int r=p+1;r<nactive;++r) {
        float factor=mat[r*4+p]/pivot;
        for (int j=p;j<nactive;++j) mat[r*4+j]-=factor*mat[p*4+j];
        rhs[r]-=factor*rhs[p];
      }
    }
    if (!valid) continue;
    for (int r=nactive-1;r>=0;--r) {
      float value=rhs[r];
      for (int j=r+1;j<nactive;++j) value-=mat[r*4+j]*x[j];
      x[r]=value/mat[r*4+r];
      if (!isfinite(x[r])) valid=false;
    }
    if (!valid) continue;
    for (int i=0;i<nactive;++i) candidate[ids[i]]=x[i];
    float error=0.0f, scale=1.0f, objective=0.0f;
    for (int i=0;i<n;++i) {
      float gradient=b[i];
      for (int j=0;j<n;++j) gradient+=A[i*4+j]*candidate[j];
      scale+=abs(b[i]);
      for (int j=0;j<n;++j) scale+=abs(A[i*4+j]*candidate[j]);
      bool active=(set & (1<<i))!=0;
      error=max(error,active ? max(0.0f,-candidate[i]) : max(0.0f,-gradient));
      objective+=0.5f*candidate[i]*(gradient+b[i]);
    }
    float scaled_error=error/scale;
    if (scaled_error<best_error ||
        (scaled_error==best_error && objective<best_objective)) {
      best_error=scaled_error; best_objective=objective;
      for (int i=0;i<4;++i) solution[i]=candidate[i];
    }
  }
  return best_error;
}

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
    device const int* condim [[buffer(15)]], device const float* friction [[buffer(16)]],
    device const int* geom_bodyid [[buffer(17)]], device const int* body_parentid [[buffer(18)]],
    device const int* body_jntadr [[buffer(19)]], device const int* body_jntnum [[buffer(20)]],
    device const int* jnt_type [[buffer(21)]], device const int* jnt_dofadr [[buffer(22)]],
    device const float* body_invweight0 [[buffer(23)]],
    device float* row_data [[buffer(24)]], device float* frame [[buffer(25)]],
    device float* jacobian [[buffer(26)]],
    constant int* dims [[buffer(27)]],
    device float* canonical_rows [[buffer(28)]],
    constant float* solver_params [[buffer(29)]],
    uint tid [[thread_position_in_grid]]) {
  int nv = dims[0], nc = dims[1], batch = dims[2], nbody = dims[3], njnt = dims[4], ngeom=dims[5];
  int world = int(tid) / max(nc, 1), slot = int(tid) % max(nc, 1);
  if (uint(world) >= uint(batch) || slot >= nc) return;
  if (dims[6 + world] == 0) return;
  int out = world * nc + slot;
  int jbase = out * 5 * nv;
  int rb=out*5*6, fb=out*12;
  for (int k=0;k<5*6;++k) row_data[rb+k]=0.0f;
  for (int k=0; k<5*nv; ++k) jacobian[jbase+k] = 0.0f;
  for (int k=0; k<12; ++k) frame[fb+k]=0.0f;
  int canonical_base=out*5*(nv+5);
  for (int row=0;row<5;++row) {
    int base=canonical_base+row*(nv+5);
    for (int k=0;k<nv;++k) canonical_rows[base+k]=0.0f;
    canonical_rows[base+nv]=1.0f;
    canonical_rows[base+nv+1]=0.0f;
    canonical_rows[base+nv+2]=0.0f;
    canonical_rows[base+nv+3]=0.0f;
    canonical_rows[base+nv+4]=0.0f;
  }

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

  float3 t1=(n.y<0.5f && n.y>-0.5f) ? float3(0,1,0) : float3(0,0,1);
  t1=normalize(t1-n*dot(n,t1));
  float3 t2=cross(n,t1);

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
          jacobian[jbase+nv+dof] += sign*dot(t1,col);
          jacobian[jbase+2*nv+dof] += sign*dot(t2,col);
        }
      }
      body=body_parentid[body];
    }
  }
  int dim=condim[slot];
  float mu0=friction[slot*2], mu1=friction[slot*2+1];
  if (dim==3) for (int i=0;i<nv;++i) {
    float jn=jacobian[jbase+i], jt1=jacobian[jbase+nv+i], jt2=jacobian[jbase+2*nv+i];
    jacobian[jbase+nv+i]=jn+mu0*jt1;
    jacobian[jbase+2*nv+i]=jn-mu0*jt1;
    jacobian[jbase+3*nv+i]=jn+mu1*jt2;
    jacobian[jbase+4*nv+i]=jn-mu1*jt2;
  }
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
  float diag_approx=body_invweight0[ba*2]+body_invweight0[bb*2];
  for (int row=0;row<(dim==3 ? 5 : 1);++row) {
    float vel=0.0f;
    for (int i=0;i<nv;++i) vel+=jacobian[jbase+row*nv+i]*qvel[world*nv+i];
    int r=rb+row*6;
    // Condim-3 uses four pyramidal edge rows. The normal Jacobian remains
    // available in slot zero for inspection, but is not a fifth constraint.
    row_data[r]=(d<m && (dim==1 || row>0)) ? 1.0f : 0.0f;
    row_data[r+1]=d;
    row_data[r+2]=vel; row_data[r+3]=-B*vel-K*impedance*(d-m);
    row_data[r+4]=impedance; row_data[r+5]=diag_approx;
  }
  for (int k=0;k<3;++k) {
    frame[fb+k]=n[k]; frame[fb+3+k]=t1[k];
    frame[fb+6+k]=t2[k]; frame[fb+9+k]=point[k];
  }
  for (int row=0;row<5;++row) {
    int data_base=rb+row*6;
    int base=canonical_base+row*(nv+5);
    float impedance=clamp(row_data[data_base+4],1e-6f,0.999999f);
    float diag_approx=max(row_data[data_base+5],1e-15f);
    float normal_R=max(1e-15f,(1.0f-impedance)*diag_approx/impedance);
    float mu0=friction[slot*2];
    float edge_R=normal_R*(1.0f+mu0*mu0);
    canonical_rows[base+nv]=(row%5)==0 ? normal_R :
        2.0f*mu0*mu0/max(solver_params[0],1e-15f)*edge_R;
    canonical_rows[base+nv+1]=row_data[data_base+3];
    canonical_rows[base+nv+2]=0.0f;
    canonical_rows[base+nv+3]=INFINITY;
    canonical_rows[base+nv+4]=row_data[data_base]>=0.5f ? 1.0f : 0.0f;
    for (int k=0;k<nv;++k)
      canonical_rows[base+k]=jacobian[jbase+row*nv+k];
  }
}

// Dense coupled projected solve for normal-only contacts. Each world factors
// M once, forms W=J M^-1 J', then performs deterministic PGS on the unilateral
// rows. This uses the same regularized complementarity form as MuJoCo's normal
// contact rows; the fixed iteration cap is part of this stage's profile.
kernel void solve_normal_contacts(
    device const float* mass [[buffer(0)]], device const float* free_acc [[buffer(1)]],
    device const float* J [[buffer(2)]], device const float* row_data [[buffer(3)]],
    device const float* friction [[buffer(4)]],
    device float* force [[buffer(5)]], device float* qacc [[buffer(6)]],
    device float* qfrc_contact [[buffer(7)]], device int* status [[buffer(8)]],
    device float* diagnostics [[buffer(9)]],
    constant int* dims [[buffer(10)]], constant float* solver_params [[buffer(11)]],
    uint world [[thread_position_in_grid]]) {
  int nv=dims[0], nc=dims[1], batch=dims[2]; if (world>=uint(batch)) return;
  if (dims[6 + int(world)] == 0) return;
  status[world]=0;
  diagnostics[world*2]=0.0f; diagnostics[world*2+1]=0.0f;
  int nr=nc*5, mb=world*nv*nv, jb=world*nr*nv;
  for (int i=0;i<nr;++i) force[world*nr+i]=0.0f;
  for (int i=0;i<nv;++i) {
    qacc[world*nv+i]=free_acc[world*nv+i];
    qfrc_contact[world*nv+i]=0.0f;
  }
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
  thread float response[80*32]; thread float rhs[32];
  if (nr>80) { status[world]=1; return; }
  for (int c=0;c<nr;++c) {
    for (int r=0;r<nv;++r) rhs[r]=J[jb+c*nv+r];
    // Solve M x = J' using L L'.
    for (int r=0;r<nv;++r) { for (int k=0;k<r;++k) rhs[r]-=L[r*nv+k]*rhs[k]; rhs[r]/=L[r*nv+r]; }
    for (int r=nv-1;r>=0;--r) { for (int k=r+1;k<nv;++k) rhs[r]-=L[k*nv+r]*rhs[k]; rhs[r]/=L[r*nv+r]; }
    for (int r=0;r<nv;++r) response[c*nv+r]=rhs[r];
  }
  thread float W[80*80];
  for (int i=0;i<nr*nr;++i) W[i]=0.0f;
  for (int a=0;a<nr;++a) for (int b=0;b<nr;++b) {
    float v=0.0f; for (int k=0;k<nv;++k) v+=J[jb+a*nv+k]*response[b*nv+k];
    W[a*nr+b]=v;
  }
  // MuJoCo's classic Euler contact diagonal approximation uses the sum of
  // translational inverse weights of the two contacting bodies.
  thread float R[80];
  for (int row=0;row<nr;++row) {
    int contact=row/5, subrow=row%5;
    int rb=(world*nr+row)*6;
    float imp=clamp(row_data[rb+4],1e-6f,0.999999f);
    float diag_approx=max(row_data[rb+5],1e-15f);
    float normal_R=max(1e-15f,(1.0f-imp)*diag_approx/imp);
    // MuJoCo forms the first edge's base R with diagApprox = tran + mu0^2*tran
    // before applying its shared pyramidal regularizer.
    float mu0=friction[contact*2];
    float edge_R=normal_R*(1.0f+mu0*mu0);
    R[row]=subrow==0 ? normal_R :
        2.0f*mu0*mu0/max(solver_params[0],1e-15f)*edge_R;
  }
  // Recompute current J*qacc then do projected Gauss-Seidel on lambda.
  bool converged=false;
  // Four pyramid facets can be strongly coupled near the cone edges. Keep a
  // bounded but generous sweep budget and report status 3 if not converged.
  for (int sweep=0;sweep<256;++sweep) {
    for (int row=0;row<nr;++row) {
      int rb=(world*nr+row)*6;
      if (row_data[rb]<0.5f) continue;
      float ja=0.0f; for (int k=0;k<nv;++k) ja+=J[jb+row*nv+k]*qacc[world*nv+k];
      float old=force[world*nr+row];
      float ref=row_data[rb+3];
      float diagonal=max(1e-15f,W[row*nr+row]+R[row]);
      float next=max(0.0f,old+(ref-ja-R[row]*old)/diagonal);
      float delta=next-old;
      force[world*nr+row]=next;
      for (int k=0;k<nv;++k) qacc[world*nv+k]+=response[row*nv+k]*delta;
    }

    // Reconstruct qacc from the current multipliers to prevent roundoff drift
    // from accumulating over many coordinate updates.
    for (int k=0;k<nv;++k) qacc[world*nv+k]=free_acc[world*nv+k];
    for (int row=0;row<nr;++row) for (int k=0;k<nv;++k)
      qacc[world*nv+k]+=response[row*nv+k]*force[world*nr+row];

    // Evaluate the normalized projected KKT residual from the reconstructed
    // state, including the exact unilateral projection at lambda >= 0.
    float max_residual=0.0f;
    for (int row=0;row<nr;++row) {
      int rb=(world*nr+row)*6;
      if (row_data[rb]<0.5f) continue;
      float ja=0.0f; for (int k=0;k<nv;++k) ja+=J[jb+row*nv+k]*qacc[world*nv+k];
      float old=force[world*nr+row], ref=row_data[rb+3];
      float diagonal=max(1e-15f,W[row*nr+row]+R[row]);
      float projected=max(0.0f,old+(ref-ja-R[row]*old)/diagonal);
      // Normwise backward-error scale: retain the magnitudes of terms that
      // may cancel in J*qacc, rather than scaling only by the small remainder.
      float jfree=0.0f;
      for (int k=0;k<nv;++k) jfree+=J[jb+row*nv+k]*free_acc[world*nv+k];
      float row_scale=abs(ref)+abs(jfree)+abs(R[row]*old);
      for (int c=0;c<nr;++c)
        row_scale+=abs(W[row*nr+c]*force[world*nr+c]);
      row_scale=max(1.0f,row_scale);
      max_residual=max(max_residual,abs(projected-old)*diagonal/row_scale);
    }
    diagnostics[world*2]=max_residual;
    diagnostics[world*2+1]=float(sweep+1);
    if (max_residual<1e-6f) { converged=true; break; }
  }
  // Scalar PGS can stagnate at low slip speeds when all four pyramidal edges
  // are nearly dependent. Refine unresolved contacts with an exact local
  // active-set solve (16 subsets per contact), cycling bounded contact blocks
  // while holding other contact forces fixed. The global KKT test below still
  // decides success for coupled contacts.
  if (!converged) {
    int pgs_sweeps=int(diagnostics[world*2+1]);
    for (int refinement=0;refinement<64;++refinement) {
      for (int contact=0;contact<nc;++contact) {
        int local_rows[4]; int nlocal=0;
        for (int slot=0;slot<5;++slot) {
          int row=contact*5+slot, rb=(world*nr+row)*6;
          if (row_data[rb]>=0.5f && nlocal<4) local_rows[nlocal++]=row;
        }
        if (nlocal==0) continue;
        thread float local_A[16], local_b[4], local_solution[4];
        for (int i=0;i<16;++i) local_A[i]=0.0f;
        for (int i=0;i<4;++i) { local_b[i]=0.0f; local_solution[i]=0.0f; }
        for (int i=0;i<nlocal;++i) {
          int row=local_rows[i], rb=(world*nr+row)*6;
          float value=-row_data[rb+3];
          for (int k=0;k<nv;++k)
            value+=J[jb+row*nv+k]*free_acc[world*nv+k];
          for (int c=0;c<nr;++c) if (c/5!=contact)
            value+=W[row*nr+c]*force[world*nr+c];
          local_b[i]=value;
          for (int j=0;j<nlocal;++j) {
            int other=local_rows[j];
            local_A[i*4+j]=W[row*nr+other]+(i==j ? R[row] : 0.0f);
          }
        }
        float block_error=solve_contact_block(local_A,local_b,nlocal,local_solution);
        if (block_error<1e-5f) for (int i=0;i<nlocal;++i)
          force[world*nr+local_rows[i]]=max(0.0f,local_solution[i]);
      }

      // Rebuild the generalized acceleration after each contact-block sweep.
      for (int k=0;k<nv;++k) qacc[world*nv+k]=free_acc[world*nv+k];
      for (int row=0;row<nr;++row) for (int k=0;k<nv;++k)
        qacc[world*nv+k]+=response[row*nv+k]*force[world*nr+row];

      float max_residual=0.0f;
      for (int row=0;row<nr;++row) {
        int rb=(world*nr+row)*6;
        if (row_data[rb]<0.5f) continue;
        float ja=0.0f; for (int k=0;k<nv;++k) ja+=J[jb+row*nv+k]*qacc[world*nv+k];
        float old=force[world*nr+row], ref=row_data[rb+3];
        float diagonal=max(1e-15f,W[row*nr+row]+R[row]);
        float projected=max(0.0f,old+(ref-ja-R[row]*old)/diagonal);
        float jfree=0.0f;
        for (int k=0;k<nv;++k) jfree+=J[jb+row*nv+k]*free_acc[world*nv+k];
        float row_scale=abs(ref)+abs(jfree)+abs(R[row]*old);
        for (int c=0;c<nr;++c)
          row_scale+=abs(W[row*nr+c]*force[world*nr+c]);
        row_scale=max(1.0f,row_scale);
        max_residual=max(max_residual,abs(projected-old)*diagonal/row_scale);
      }
      diagnostics[world*2]=max_residual;
      diagnostics[world*2+1]=float(pgs_sweeps+refinement+1);
      if (max_residual<1e-6f) { converged=true; break; }
    }
  }
  if (!converged) status[world]=3;
  for (int row=0;row<nr;++row) for (int k=0;k<nv;++k)
    qfrc_contact[world*nv+k]+=J[jb+row*nv+k]*force[world*nr+row];
}
