// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
//
// Compensated site-only mj_tendon geometry. Endpoint/anchor cancellation must
// consume retained FK words before rounding the public length/Jacobian outputs.
#include <metal_stdlib>
using namespace metal;
struct STPair {
  float hi, lo;
  STPair() : hi(0), lo(0) {}
  STPair(float x) : hi(x), lo(0) {}
  STPair(float h,float l) : hi(h),lo(l) {}
};
inline STPair stp_norm(float h,float l) {
  float s=h+l; return STPair(s,l-(s-h));
}
inline STPair operator+(STPair a,STPair b) {
  float s=a.hi+b.hi, v=s-a.hi;
  float e=(a.hi-(s-v))+(b.hi-v);
  return stp_norm(s,e+a.lo+b.lo);
}
inline STPair operator-(STPair a) {return STPair(-a.hi,-a.lo);}
inline STPair operator-(STPair a,STPair b) {return a+(-b);}
inline STPair operator*(STPair a,STPair b) {
  float p=a.hi*b.hi;
  return stp_norm(p,fma(a.hi,b.hi,-p)+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo);
}
inline STPair operator/(STPair a,STPair b) {
  float q=a.hi/b.hi; STPair r=a-b*STPair(q);
  return stp_norm(q,(r.hi+r.lo)/b.hi);
}
inline STPair stp_sqrt(STPair a) {
  float q=sqrt(max(a.hi,0.0f));
  if(q==0)return STPair(0);
  STPair r=a-STPair(q)*STPair(q);
  return stp_norm(q,(r.hi+r.lo)/(2*q));
}
struct STVec {STPair x,y,z;};
inline STVec stv_zero(){return {STPair(0),STPair(0),STPair(0)};}
inline STVec stv_add(STVec a,STVec b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline STVec stv_sub(STVec a,STVec b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline STVec stv_scale(STVec a,STPair b){return {a.x*b,a.y*b,a.z*b};}
inline STPair stv_dot(STVec a,STVec b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline STVec stv_cross(STVec a,STVec b){
  return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline STPair stp_load(device const float* p,int index,int width){
  return STPair(p[index],p[width+index]);
}
inline STVec stv_load(device const float* p,int index,int width){
  return {stp_load(p,index,width),stp_load(p,index+1,width),stp_load(p,index+2,width)};
}
inline STVec stv_rotate(device const float* poses,int index,int width,STVec v){
  STPair w=stp_load(poses,index,width);
  STVec q=stv_load(poses,index+1,width);
  return stv_add(v,stv_scale(stv_cross(q,stv_add(stv_cross(q,v),stv_scale(v,w))),STPair(2)));
}
// Dynamic DOF traversal: no fixed 32/64-column local array or hidden cutoff.
inline STVec st_source_point_jac(
    STVec point,int body,int dof,device const float* poses,
    device const int* parent,device const int* jadr,device const int* jnum,
    device const int* jtype,device const int* dadr,
    int ns,int nb,int nj,int width) {
  STVec result=stv_zero();
  for(int b=body;b>0 && b<nb;b=parent[b]){
    int quat=3*ns+3*nb+4*b;
    for(int jj=0;jj<jnum[b];jj++){
      int j=jadr[b]+jj, type=jtype[j], local=dof-dadr[j];
      int count=type==0?6:(type==1?3:1);
      if(local<0 || local>=count)continue;
      if(type==0 && local<3){
        STVec basis={STPair(local==0),STPair(local==1),STPair(local==2)};
        result=stv_add(result,basis);continue;
      }
      STVec local_axis;
      if(type==2 || type==3)local_axis=stv_zero();
      else {
        int k=type==0?local-3:local;
        local_axis={STPair(k==0),STPair(k==1),STPair(k==2)};
      }
      STVec axis=(type==2 || type==3)
          ?stv_load(poses,3*ns+7*nb+3*nj+3*j,width)
          :stv_rotate(poses,quat,width,local_axis);
      if(type==2){result=stv_add(result,axis);continue;}
      STVec anchor=stv_load(poses,3*ns+7*nb+3*j,width);
      result=stv_add(result,stv_cross(axis,stv_sub(point,anchor)));
    }
  }
  return result;
}
kernel void spatial_tendon_site_source(
    device const float* poses [[buffer(0)]],
    device const float* path_words [[buffer(1)]],
    device const float* qvel [[buffer(2)]],
    device const int* types [[buffer(3)]],
    device const int* ids [[buffer(4)]],
    device const int* offset [[buffer(5)]],
    device const int* count [[buffer(6)]],
    device const int* site_body [[buffer(7)]],
    device const int* parent [[buffer(8)]],
    device const int* jadr [[buffer(9)]],
    device const int* jnum [[buffer(10)]],
    device const int* jtype [[buffer(11)]],
    device const int* dadr [[buffer(12)]],
    constant int* dims [[buffer(13)]],
    device float* length_out [[buffer(14)]],
    device float* velocity_out [[buffer(15)]],
    device float* jacobian_out [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
  int nv=dims[0],nt=dims[1],ns=dims[2],nb=dims[4],nj=dims[5],nw=dims[6];
  int world=int(tid/uint(max(nt,1))),t=int(tid%uint(max(nt,1)));
  if(world>=dims[7] || t>=nt || dims[8+world]==0)return;
  int off=offset[t],n=count[t];
  if(n<2)return;
  for(int i=0;i<n;i++)if(types[off+i]!=2 && types[off+i]!=3)return;
  int width=3*ns+7*nb+6*nj, cw=nw;
  device const float* pose=poses+uint(world)*2u*uint(width);
  STPair len(0),vel(0);
  // Compute each column in compensated precision, including accumulation
  // across distinct pulley branches. Length is evaluated only with column0.
  for(int d=0;d<max(nv,1);d++){
    STPair row(0),divisor(1);
    for(int i=0;i<n-1;i++){
      int type0=types[off+i],type1=types[off+i+1];
      if(type0==2 || type1==2){
        if(type0==2)divisor=stp_load(path_words,off+i,cw);
        continue;
      }
      int id0=ids[off+i],id1=ids[off+i+1];
      STVec a=stv_load(pose,3*id0,width),b=stv_load(pose,3*id1,width);
      STVec diff=stv_sub(b,a);
      STPair norm=stp_sqrt(stv_dot(diff,diff));
      if(d==0)len=len+norm/divisor;
      if(nv==0 || site_body[id0]==site_body[id1])continue;
      STVec direction=norm.hi>1e-15f?stv_scale(diff,STPair(1)/norm)
          :STVec{STPair(1),STPair(0),STPair(0)};
      STVec ja=st_source_point_jac(a,site_body[id0],d,pose,
          parent,jadr,jnum,jtype,dadr,ns,nb,nj,width);
      STVec jb=st_source_point_jac(b,site_body[id1],d,pose,
          parent,jadr,jnum,jtype,dadr,ns,nb,nj,width);
      row=row+stv_dot(stv_sub(jb,ja),direction)/divisor;
    }
    jacobian_out[(uint(world)*uint(max(nt,1))+uint(t))*uint(max(nv,1))+uint(d)]=row.hi+row.lo;
    if(nv)vel=vel+row*STPair(qvel[uint(world)*uint(nv)+uint(d)]);
  }
  length_out[uint(world)*uint(max(nt,1))+uint(t)]=len.hi+len.lo;
  velocity_out[uint(world)*uint(max(nt,1))+uint(t)]=vel.hi+vel.lo;
}
