// Copyright 2021 DeepMind Technologies Limited
// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Native source-rounded ten-component inertia operators, pinned MuJoCo 3.10.
struct SR {
  KinDD v;
  SR():v(kin_dd(0)) {} SR(float x):v(kin_dd(x)) {} SR(KinDD x):v(x) {}
};
inline SR operator+(SR a,SR b){return SR(kin_add(a.v,b.v));}
inline SR operator-(SR a,SR b){return SR(kin_sub(a.v,b.v));}
inline SR operator-(SR a){return SR(kin_neg(a.v));}
inline SR operator*(SR a,SR b){return SR(kin_mul(a.v,b.v));}
inline SR operator/(SR a,SR b){return SR(kin_div(a.v,b.v));}
inline SR sfma(SR a,SR b,SR c){return SR(kin_fma(a.v,b.v,c.v));}
inline SR srload(device const float* p,int i){return SR(KinDD{p[3*i],p[3*i+1],p[3*i+2]});}
inline void srstore(device float* p,int i,SR x){p[3*i]=x.v.hi;p[3*i+1]=x.v.lo;p[3*i+2]=x.v.tail;}
inline SR srpose(device const float* h,device const float* l,device const float* t,int i){return SR(KinDD{h[i],l[i],t[i]});}
inline void srpair(SR x,device float* h,device float* l,int i){h[i]=x.v.hi;l[i]=x.v.lo+x.v.tail;}
inline SR srvelocity(device const float* p,int i,int size){return SR(KinDD{p[i],p[size+i],p[2*size+i]});}
inline void srinert(thread SR* r,thread const SR* i,thread const SR* v){
  r[0]=sfma(i[7],v[5],sfma(-i[8],v[4],sfma(i[4],v[2],sfma(i[0],v[0],i[3]*v[1]))));
  r[1]=sfma(-i[6],v[5],sfma(i[8],v[3],sfma(i[5],v[2],sfma(i[3],v[0],i[1]*v[1]))));
  r[2]=sfma(i[6],v[4],sfma(-i[7],v[3],sfma(i[2],v[2],sfma(i[4],v[0],i[5]*v[1]))));
  r[3]=sfma(i[9],v[3],sfma(i[8],v[1],-i[7]*v[2]));
  r[4]=sfma(i[9],v[4],sfma(i[6],v[2],-i[8]*v[0]));
  r[5]=sfma(i[9],v[5],sfma(i[7],v[0],-i[6]*v[1]));
}
inline SR srdot6(thread const SR* a,thread const SR* b){
  return (sfma(a[0],b[0],a[2]*b[2])+sfma(a[1],b[1],a[3]*b[3]))+sfma(a[4],b[4],a[5]*b[5]);
}
inline void srcross(thread SR* r,thread const SR* a,thread const SR* b,bool force){
  for(int k=0;k<3;k++){
    int u=(k+1)%3,v=(k+2)%3;
    r[k]=sfma(a[u],b[v],-a[v]*b[u]);
    r[k+3]=sfma(a[u],b[v+3],-a[v]*b[u+3]);
    if(force)r[k]=r[k]+sfma(a[u+3],b[v+3],-a[v+3]*b[u+3]);
    else r[k+3]=r[k+3]+sfma(a[u+3],b[v],-a[v+3]*b[u]);
  }
}
inline int srmi(int world,int i,int j,int nv,device const int* tree,constant int* layout){
  if(!layout[1])return world*layout[0]+i*nv+j;
  int nc=layout[2],owner=nc+1+max(layout[3],1),local=owner+max(layout[3],1),off=local+max(layout[3],1);
  int c=tree[owner+i];if(c<0 || c!=tree[owner+j])return -1;
  int width=tree[c+1]-tree[c];return world*layout[0]+tree[off+c]+tree[local+i]*width+tree[local+j];
}
kernel void source_rigid_position(
 device const int* meta [[buffer(0)]],device const float* constants [[buffer(1)]],device float* packet [[buffer(2)]],
 device const float* bq [[buffer(3)]],device const float* bql [[buffer(4)]],device const float* bqt [[buffer(5)]],
 device const float* ip [[buffer(6)]],device const float* ipl [[buffer(7)]],device const float* ipt [[buffer(8)]],
 device const float* iq [[buffer(9)]],device const float* iql [[buffer(10)]],device const float* iqt [[buffer(11)]],
 device const float* ja [[buffer(12)]],device const float* jal [[buffer(13)]],device const float* jat [[buffer(14)]],
 device const float* ax [[buffer(15)]],device const float* axl [[buffer(16)]],device const float* axt [[buffer(17)]],
 device float* mass [[buffer(18)]],device float* masslow [[buffer(19)]],
 device const int* bodies [[buffer(20)]],device const int* parents [[buffer(21)]],device const int* dofs [[buffer(22)]],
 device const int* counts [[buffer(23)]],device const int* mask [[buffer(24)]],constant int* awake [[buffer(25)]],
 device const int* tree [[buffer(26)]],constant int* layout [[buffer(27)]],constant int* dims [[buffer(28)]],
 uint world [[thread_position_in_grid]]){
 int nb=dims[0],nv=dims[1],nj=dims[2];if(world>=uint(dims[3]) || !layout[5+world])return;
 int bs=world*max(awake[0],1),ds=world*max(awake[1],1),nbody=counts[3*world],nparent=counts[3*world+1],nd=counts[3*world+2];
 device float* out=packet+3*world*dims[4];int com=0,ci=3*nb,cr=13*nb,cd=23*nb;
 for(int k=0;k<nbody;k++){int b=bodies[bs+k];SR m=srload(constants,b);
  for(int r=0;r<3;r++)srstore(out,3*b+r,m*srpose(ip,ipl,ipt,3*(world*nb+b)+r));}
 for(int k=nparent-1;k>=0;k--){int b=parents[bs+k],p=meta[6*b];if(!b)continue;
  for(int r=0;r<3;r++){SR child=srload(out,3*b+r);if(!mask[world*max(awake[0],1)+b])child=child*srload(constants,nb+b);
   srstore(out,3*p+r,srload(out,3*p+r)+child);}}
 for(int k=0;k<nbody;k++){int b=bodies[bs+k];SR sm=srload(constants,nb+b);
  bool zero=kin_compare(sm.v,KinDD{1.0000000036274937e-15f,-3.627493647833322e-24f,0})<0;
  SR inv=zero?SR(0):SR(1)/sm;
  for(int r=0;r<3;r++)srstore(out,3*b+r,zero?srpose(ip,ipl,ipt,3*(world*nb+b)+r):srload(out,3*b+r)*inv);}
 for(int k=0;k<nbody;k++){int b=bodies[bs+k];SR I[10];
  if(b){int root=meta[6*b+1],qb=4*(world*nb+b);
   KinQ q=kinq(float4(iq[qb],iq[qb+1],iq[qb+2],iq[qb+3]),float4(iql[qb],iql[qb+1],iql[qb+2],iql[qb+3]),float4(iqt[qb],iqt[qb+1],iqt[qb+2],iqt[qb+3]));
   KinMat mat=kinq_matrix(q);SR a[9],tmp[9],r[3],m=srload(constants,b);
   for(int z=0;z<9;z++)a[z]=SR(mat.m[z]);
   for(int u=0;u<3;u++){SR iv=srload(constants,2*nb+3*b+u);for(int v=0;v<3;v++)tmp[3*u+v]=a[3*v+u]*iv;
    r[u]=srpose(ip,ipl,ipt,3*(world*nb+b)+u)-srload(out,3*root+u);}
   int rr[6]={0,1,2,0,0,1},cc[6]={0,1,2,1,2,2};
   for(int z=0;z<6;z++){int u=rr[z],v=cc[z];I[z]=sfma(a[3*u+2],tmp[6+v],sfma(a[3*u],tmp[v],a[3*u+1]*tmp[3+v]));}
   I[0]=I[0]+m*sfma(r[1],r[1],r[2]*r[2]);I[1]=I[1]+m*sfma(r[0],r[0],r[2]*r[2]);I[2]=I[2]+m*sfma(r[0],r[0],r[1]*r[1]);
   I[3]=I[3]-(m*r[0])*r[1];I[4]=I[4]-(m*r[0])*r[2];I[5]=I[5]-(m*r[1])*r[2];
   for(int r0=0;r0<3;r0++)I[6+r0]=m*r[r0];I[9]=m;
  }
  for(int z=0;z<10;z++){srstore(out,ci+10*b+z,I[z]);srstore(out,cr+10*b+z,I[z]);}
  if(!b)continue;int first=meta[6*b+2],count=meta[6*b+3],root=meta[6*b+1];
  for(int jj=0;jj<count;jj++){int j=first+jj,type=meta[6*nb+3*nv+2*j],d=meta[6*nb+3*nv+2*j+1],qa=4*(world*nb+b);
   SR offset[3];for(int z=0;z<3;z++)offset[z]=srload(out,3*root+z)-srpose(ja,jal,jat,3*(world*nj+j)+z);
   int skip=type==0?3:0;
   if(type==0)for(int u=0;u<3;u++)for(int z=0;z<6;z++)srstore(out,cd+6*(d+u)+z,SR(z==u+3?1:0));
   if(type==0 || type==1){KinMat mat=kinq_matrix(kinq(float4(bq[qa],bq[qa+1],bq[qa+2],bq[qa+3]),float4(bql[qa],bql[qa+1],bql[qa+2],bql[qa+3]),float4(bqt[qa],bqt[qa+1],bqt[qa+2],bqt[qa+3])));
    for(int u=0;u<3;u++)for(int z=0;z<3;z++){SR axis=SR(mat.m[3*z+u]);srstore(out,cd+6*(d+skip+u)+z,axis);
     int r=(z+1)%3,s=(z+2)%3;srstore(out,cd+6*(d+skip+u)+3+z,sfma(SR(mat.m[3*r+u]),offset[s],-SR(mat.m[3*s+u])*offset[r]));}}
   else for(int z=0;z<3;z++){SR a=srpose(ax,axl,axt,3*(world*nj+j)+z);srstore(out,cd+6*d+z,type==2?SR(0):a);
    int r=(z+1)%3,s=(z+2)%3;srstore(out,cd+6*d+3+z,type==2?a:sfma(srpose(ax,axl,axt,3*(world*nj+j)+r),offset[s],-srpose(ax,axl,axt,3*(world*nj+j)+s)*offset[r]));}
  }
 }
 for(int k=nparent-1;k>=0;k--){int b=parents[bs+k],p=meta[6*b];if(p>0)for(int z=0;z<10;z++)srstore(out,cr+10*p+z,srload(out,cr+10*p+z)+srload(out,cr+10*b+z));}
 for(int k=0;k<nd;k++){int d=dofs[ds+k],body=meta[6*nb+3*d+1];
  for(int j=0;j<nv;j++){int a=srmi(world,d,j,nv,tree,layout);if(a>=0){mass[a]=0;masslow[a]=0;}}
  if(meta[6*nb+3*d+2]){int a=srmi(world,d,d,nv,tree,layout);srpair(srload(constants,5*nb+nv+d),mass,masslow,a);continue;}
  SR I[10],v[6],buf[6];for(int z=0;z<10;z++)I[z]=srload(out,cr+10*body+z);for(int z=0;z<6;z++)v[z]=srload(out,cd+6*d+z);srinert(buf,I,v);
  for(int j=d;j>=0;j=meta[6*nb+3*j]){for(int z=0;z<6;z++)v[z]=srload(out,cd+6*j+z);SR val=srdot6(v,buf);if(d==j)val=srload(constants,5*nb+d)+val;
   int a=srmi(world,d,j,nv,tree,layout),b=srmi(world,j,d,nv,tree,layout);if(a>=0)srpair(val,mass,masslow,a);if(b>=0)srpair(val,mass,masslow,b);}
 }
}
kernel void source_rigid_publish(device const int* meta [[buffer(0)]],device const float* packet [[buffer(1)]],
 device float* rootcom [[buffer(2)]],device float* cdof [[buffer(3)]],device float* inertia [[buffer(4)]],
 device const int* bodies [[buffer(5)]],device const int* dofs [[buffer(6)]],device const int* counts [[buffer(7)]],
 constant int* awake [[buffer(8)]],constant int* layout [[buffer(9)]],constant int* dims [[buffer(10)]],uint world [[thread_position_in_grid]]){
 if(world>=uint(dims[3]) || !layout[5+world])return;int nb=dims[0],nv=dims[1];device const float* p=packet+3*world*dims[4];
 for(int k=0;k<counts[3*world];k++){int b=bodies[world*max(awake[0],1)+k],root=meta[6*b+1];for(int z=0;z<3;z++)rootcom[3*(world*nb+b)+z]=srload(p,3*root+z).v.hi;
  SR I[10],v[6],out[6];for(int z=0;z<10;z++)I[z]=srload(p,3*nb+10*b+z);
  for(int c=0;c<6;c++){for(int z=0;z<6;z++)v[z]=SR(z==c?1:0);srinert(out,I,v);for(int r=0;r<6;r++)inertia[36*(world*nb+b)+6*r+c]=out[r].v.hi;}}
 for(int k=0;k<counts[3*world+2];k++){int d=dofs[world*max(awake[1],1)+k];for(int z=0;z<6;z++)cdof[6*(world*nv+d)+z]=srload(p,23*nb+6*d+z).v.hi;}
}
kernel void source_rigid_velocity(device const int* meta [[buffer(0)]],device const float* constants [[buffer(1)]],
 device const float* position [[buffer(2)]],device float* velocity [[buffer(3)]],device const float* qvel [[buffer(4)]],
 device const int* bodies [[buffer(5)]],device const int* parents [[buffer(6)]],device const int* dofs [[buffer(7)]],
 device const int* counts [[buffer(8)]],constant int* awake [[buffer(9)]],device const int* mask [[buffer(10)]],constant int* dims [[buffer(11)]],
 device float* bias [[buffer(12)]],device float* biaslow [[buffer(13)]],device float* cvel [[buffer(14)]],device float* cvlow [[buffer(15)]],
 device float* cdot [[buffer(16)]],device float* cdotlow [[buffer(17)]],device float* cacc [[buffer(18)]],device float* calow [[buffer(19)]],
 device float* force [[buffer(20)]],device float* flow [[buffer(21)]],uint world [[thread_position_in_grid]]){
 if(world>=uint(dims[3]) || !mask[world])return;int nb=dims[0],nv=dims[1];int bs=world*max(awake[0],1),ds=world*max(awake[1],1),nbody=counts[3*world],np=counts[3*world+1],nd=counts[3*world+2];
 device const float* p=position+3*world*dims[4];device float* w=velocity+3*world*dims[5];int cd=23*nb,cv=0,dot=6*nb,ca=6*nb+6*nv,fr=12*nb+6*nv;
 // Sleeping child force planes are retained inputs to the parent-awake pass.
 // Preserve the source tail while the public pair is unchanged; import a
 // changed retained pair before accumulating it, including explicit restores.
 for(int b=1;b<nb;b++){
  bool selected=false;for(int k=0;k<nbody;k++)if(bodies[bs+k]==b)selected=true;
  if(!selected)for(int z=0;z<6;z++){
   int i=6*(world*nb+b)+z;SR old=srload(w,fr+6*b+z);
   if(old.v.hi!=force[i] || old.v.lo+old.v.tail!=flow[i])
    srstore(w,fr+6*b+z,SR(KinDD{force[i],flow[i],0}));
  }
 }
 for(int z=0;z<6;z++){srstore(w,z,SR(0));srstore(w,ca+z,(z>=3 && !(dims[6]&128))?-srload(constants,5*nb+2*nv+z-3):SR(0));srstore(w,fr+z,SR(0));}
 for(int k=0;k<nbody;k++){int b=bodies[bs+k];if(!b)continue;int parent=meta[6*b],first=meta[6*b+2],nj=meta[6*b+3],da=meta[6*b+4],dn=meta[6*b+5];SR v[6];for(int z=0;z<6;z++)v[z]=srload(w,6*parent+z);
  for(int z=0;z<6*dn;z++)srstore(w,dot+6*da+z,SR(0));
  for(int jj=0;jj<nj;jj++){int j=first+jj,type=meta[6*nb+3*nv+2*j],d=meta[6*nb+3*nv+2*j+1],skip=type==0?3:0;
   if(type==0)for(int z=0;z<6;z++){SR sum(0);for(int u=0;u<3;u++)sum=sfma(srload(p,cd+6*(d+u)+z),srvelocity(qvel,world*nv+d+u,dims[3]*nv),sum);v[z]=v[z]+sum;}
   int n=(type==0 || type==1)?3:1;
   for(int u=0;u<n;u++){SR axis[6],out[6];for(int z=0;z<6;z++)axis[z]=srload(p,cd+6*(d+skip+u)+z);srcross(out,v,axis,false);for(int z=0;z<6;z++)srstore(w,dot+6*(d+skip+u)+z,out[z]);}
   for(int z=0;z<6;z++){SR sum(0);for(int u=0;u<n;u++)sum=sfma(srload(p,cd+6*(d+skip+u)+z),srvelocity(qvel,world*nv+d+skip+u,dims[3]*nv),sum);v[z]=v[z]+sum;}
  }
  for(int z=0;z<6;z++)srstore(w,6*b+z,v[z]);
  SR acc[6],I[10],f[6],mv[6],cross[6];for(int z=0;z<6;z++){SR sum(0);for(int u=0;u<dn;u++)sum=sfma(srload(w,dot+6*(da+u)+z),srvelocity(qvel,world*nv+da+u,dims[3]*nv),sum);acc[z]=srload(w,ca+6*parent+z)+sum;srstore(w,ca+6*b+z,acc[z]);}
  for(int z=0;z<10;z++)I[z]=srload(p,3*nb+10*b+z);srinert(f,I,acc);srinert(mv,I,v);srcross(cross,v,mv,true);for(int z=0;z<6;z++)srstore(w,fr+6*b+z,f[z]+cross[z]);
 }
 for(int k=np-1;k>=0;k--){int b=parents[bs+k],parent=meta[6*b];if(parent>0)for(int z=0;z<6;z++)srstore(w,fr+6*parent+z,srload(w,fr+6*parent+z)+srload(w,fr+6*b+z));}
 for(int k=0;k<nd;k++){int d=dofs[ds+k],body=meta[6*nb+3*d+1];SR a[6],b[6];for(int z=0;z<6;z++){a[z]=srload(p,cd+6*d+z);b[z]=srload(w,fr+6*body+z);srpair(srload(w,dot+6*d+z),cdot,cdotlow,6*(world*nv+d)+z);}
  // Keep the legacy public high word and supplement its omitted operands.
  SR residual=srdot6(a,b)-SR(bias[world*nv+d]);
  biaslow[world*nv+d]=kin_value(residual.v);
 }
 for(int k=0;k<nbody;k++){int b=bodies[bs+k];for(int z=0;z<6;z++){int i=6*(world*nb+b)+z;srpair(srload(w,6*b+z),cvel,cvlow,i);srpair(srload(w,ca+6*b+z),cacc,calow,i);srpair(srload(w,fr+6*b+z),force,flow,i);}}
}
