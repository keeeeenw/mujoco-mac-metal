// Copyright 2026 The MuJoCo Metal contributors
// Licensed under the Apache License, Version 2.0.
// Source semantics: pinned engine_support.c::mj_geomDistance and
// engine_collision_gjk.c::mjc_ccd (one actual witness, zero geom margins).

static inline void gq_quat_matrix(float4 q,thread FlexDD* m) {
  FlexDD w=flex_dd(q.x),x=flex_dd(q.y),y=flex_dd(q.z),z=flex_dd(q.w);
  FlexDD norm=flex_dd_source_sqrt(flex_dd_source_add(
      flex_dd_source_add(flex_dd_source_mul(w,w),flex_dd_source_mul(x,x)),
      flex_dd_source_add(flex_dd_source_mul(y,y),flex_dd_source_mul(z,z))));
  FlexDD inv=flex_dd_source_div(flex_dd(1),norm);
  w=flex_dd_source_mul(w,inv);x=flex_dd_source_mul(x,inv);
  y=flex_dd_source_mul(y,inv);z=flex_dd_source_mul(z,inv);
  FlexDD xx=flex_dd_source_mul(x,x),yy=flex_dd_source_mul(y,y),zz=flex_dd_source_mul(z,z);
  FlexDD xy=flex_dd_source_mul(x,y),xz=flex_dd_source_mul(x,z),yz=flex_dd_source_mul(y,z);
  FlexDD wx=flex_dd_source_mul(w,x),wy=flex_dd_source_mul(w,y),wz=flex_dd_source_mul(w,z);
  m[0]=flex_dd_source_sub(flex_dd(1),flex_dd_source_mul(flex_dd(2),flex_dd_source_add(yy,zz)));
  m[1]=flex_dd_source_mul(flex_dd(2),flex_dd_source_sub(xy,wz));
  m[2]=flex_dd_source_mul(flex_dd(2),flex_dd_source_add(xz,wy));
  m[3]=flex_dd_source_mul(flex_dd(2),flex_dd_source_add(xy,wz));
  m[4]=flex_dd_source_sub(flex_dd(1),flex_dd_source_mul(flex_dd(2),flex_dd_source_add(xx,zz)));
  m[5]=flex_dd_source_mul(flex_dd(2),flex_dd_source_sub(yz,wx));
  m[6]=flex_dd_source_mul(flex_dd(2),flex_dd_source_sub(xz,wy));
  m[7]=flex_dd_source_mul(flex_dd(2),flex_dd_source_add(yz,wx));
  m[8]=flex_dd_source_sub(flex_dd(1),flex_dd_source_mul(flex_dd(2),flex_dd_source_add(xx,yy)));
}

static __attribute__((noinline)) int gq_source_distance(
    thread CommonCCDSupportObject& a,thread CommonCCDSupportObject& b,
    FlexDD cutoff,device float* hull,device int* info,int header,int query,
    thread FlexDD& distance,thread FlexDD3& wa,thread FlexDD3& wb) {
  int vc=info[header+11],fc=info[header+12],hc=info[header+13],sc=info[header+14];
  int scratch=info[header+7]+query*info[header+8];
  device FlexDDVertex* verts=reinterpret_cast<device FlexDDVertex*>(hull+scratch);
  device int2* ids=reinterpret_cast<device int2*>(info+info[header+24]+2*query*vc);
  device FlexEpaFace* faces=reinterpret_cast<device FlexEpaFace*>(hull+scratch+29*vc);
  device int* map=reinterpret_cast<device int*>(hull+scratch+29*vc+20*fc);
  int horizon_base=scratch+29*vc+20*fc+fc+(29*vc&1);
  device int2* horizon=reinterpret_cast<device int2*>(hull+horizon_base);
  device FlexHorizonFrame* stack=reinterpret_cast<device FlexHorizonFrame*>(hull+horizon_base+2*hc);
  int tb=info[header+10],iterations=info[header+9];
  FlexDD tolerance=FlexDD{hull[tb],hull[tb+1],hull[tb+2]};
  FlexDD3 ca=FlexDD3{a.pos[0],a.pos[1],a.pos[2]},cb=FlexDD3{b.pos[0],b.pos[1],b.pos[2]};
  CommonCCDVertex simplex[4];
  CommonCCDSupportObject sa=a,sb=b;
  FlexDD radius_a=flex_dd(0),radius_b=flex_dd(0);
  bool shrink_a=a.geom_type==2 || a.geom_type==3;
  bool shrink_b=b.geom_type==2 || b.geom_type==3;
  if (shrink_a) {radius_a=a.size[0];sa.kind=a.geom_type==2?COMMON_CCD_POINT:COMMON_CCD_LINE;sa.geom=-1;}
  if (shrink_b) {radius_b=b.size[0];sb.kind=b.geom_type==2?COMMON_CCD_POINT:COMMON_CCD_LINE;sb.geom=-1;}
  if (shrink_a || shrink_b) {
    FlexDD radii=flex_dd_source_add(radius_a,radius_b);
    CommonCCDGjkResult shallow=common_ccd_source_gjk(sa,sb,ca,cb,tolerance,
        flex_dd_source_add(cutoff,radii),iterations,hull,info,simplex,nullptr,0);
    if (shallow.status==1) return 0;
    if (shallow.status!=0) return -shallow.status;
    if (flex_dd_compare_exact(shallow.distance,tolerance)>0) {
      distance=flex_dd_source_sub(shallow.distance,radii);
      FlexDD3 delta=flex_dd3_source_sub(shallow.witness_b,shallow.witness_a);
      FlexDD inverse=flex_dd_source_div(flex_dd(1),flex_dd_source_sqrt(flex_dd_source_dot3(delta,delta)));
      FlexDD3 direction=flex_dd3_scale(delta,inverse);
      wa=flex_dd3_add(shallow.witness_a,flex_dd3_scale(direction,radius_a));
      wb=flex_dd3_sub(shallow.witness_b,flex_dd3_scale(direction,radius_b));
      return 1;
    }
  }
  CommonCCDGjkResult gjk=common_ccd_source_gjk(a,b,ca,cb,tolerance,cutoff,
      iterations,hull,info,simplex,nullptr,0);
  if (gjk.status==1) return 0;
  if (gjk.status!=0) return -gjk.status;
  distance=gjk.distance;wa=gjk.witness_a;wb=gjk.witness_b;
  if (!gjk.needs_intersection) return gjk.simplex_count>0?1:0;
  ContactGeom contact;int support_count=0,face_count=0;
  CommonCCDEpaResult epa=common_ccd_source_epa(simplex,gjk.simplex_count,a,b,
      iterations,tolerance,gjk.distance,hull,info,verts,ids,faces,map,horizon,stack,
      vc,fc,hc,sc,contact,nullptr,nullptr,support_count,face_count);
  if (epa.status==1) {distance=flex_dd(0);return 1;}
  if (epa.status!=0) return -10-epa.status;
  distance=flex_dd_neg(epa.distance);wa=epa.witness_a;wb=epa.witness_b;
  return 1;
}

// Distance queries reduce the same source-ordered plane manifold used by
// physical contacts, preserving the first strict minimum and its witnesses.
static inline int gq_plane_convex(thread CommonCCDSupportObject& plane,
    thread CommonCCDSupportObject& convex,FlexDD cutoff,FlexDD rbound,
    device float* hull,device int* info,thread FlexDD& distance,
    thread FlexDD3& wa,thread FlexDD3& wb) {
  FlexDD depths[3]; FlexDD3 points[3],normal;
  int count=common_ccd_source_plane_convex(plane,convex,cutoff,rbound,hull,info,
                                          depths,points,normal);
  if(count<=0) return count;
  int pick=0;
  for(int k=1;k<count;k++) if(flex_dd_compare_exact(depths[k],depths[pick])<0) pick=k;
  distance=depths[pick];wb=points[pick];
  wa=flex_dd3_sub(wb,flex_dd3_scale(normal,distance));
  return 1;
}

kernel void geometry_query_pairs(
    device const float* pos [[buffer(0)]],device const float* poslo [[buffer(1)]],
    device const float* postail [[buffer(2)]],device const float* quat [[buffer(3)]],
    device const float* mat [[buffer(4)]],device const float* matlo [[buffer(5)]],
    device const float* mattail [[buffer(6)]],device const int* types [[buffer(7)]],
    device const int* pairs [[buffer(8)]],device const float* cutoffs [[buffer(9)]],
    device float* hull [[buffer(10)]],device int* info [[buffer(11)]],
    device float* results [[buffer(12)]],device const int* mask [[buffer(13)]],
    constant int* dims [[buffer(14)]],constant int& has_matrix [[buffer(15)]],
    uint index [[thread_position_in_grid]]) {
  int batch=dims[0],np=dims[1],ng=dims[2],ns=dims[3];
  if (index>=uint(batch*np) || (mask[0]&2)==0 || (dims[5]&8192)!=0) return;
  int world=int(index)/np,pair=int(index)%np,sensor=pairs[3*pair];
  if (mask[1+world*max(ns,1)+sensor]==0 || mask[1+batch*max(ns,1)+world]==0) return;
  int ga=pairs[3*pair+1],gb=pairs[3*pair+2],ta=types[ga],tb=types[gb];
  if (ta==1 || tb==1 || ((ta==8 || tb==8) && ta!=0 && tb!=0)) return;
  int header=9*ng,base=int(index)*10;
  FlexDD bound=flex_dd(cutoffs[pair]),dist=bound;
  FlexDD3 wa=flex_dd3(float3(0)),wb=wa;
  int found=0;
  FlexDD3 pa=common_ccd_load_geom_position(pos,poslo,postail,3*(world*ng+ga));
  FlexDD3 pb=common_ccd_load_geom_position(pos,poslo,postail,3*(world*ng+gb));
  float4 qa=float4(quat[4*(world*ng+ga)],quat[4*(world*ng+ga)+1],quat[4*(world*ng+ga)+2],quat[4*(world*ng+ga)+3]);
  float4 qb=float4(quat[4*(world*ng+gb)],quat[4*(world*ng+gb)+1],quat[4*(world*ng+gb)+2],quat[4*(world*ng+gb)+3]);
  FlexDD ma[9],mb[9];
  if (!has_matrix) gq_quat_matrix(qa,ma);
  else common_ccd_load_geom_matrix(mat,matlo,mattail,9*(world*ng+ga),ma);
  if (!has_matrix) gq_quat_matrix(qb,mb);
  else common_ccd_load_geom_matrix(mat,matlo,mattail,9*(world*ng+gb),mb);
  if(!common_ccd_finite_pose(pa,ma) || !common_ccd_finite_pose(pb,mb)) {
    results[base]=1.0f;results[base+1]=bound.hi;
    for(int k=2;k<10;k++) results[base+k]=0.0f;
    return;
  }
  int records=info[header+1],sizes=info[header+34];
  CommonCCDSupportObject a=common_ccd_production_geom(ga,ta==8?7:ta,records+9*ga,info,hull,pa,ma,float3(0),0);
  CommonCCDSupportObject b=common_ccd_production_geom(gb,tb==8?7:tb,records+9*gb,info,hull,pb,mb,float3(0),0);
  for (int k=0;k<3;k++) {
    a.size[k]=FlexDD{hull[sizes+9*ga+3*k],hull[sizes+9*ga+3*k+1],hull[sizes+9*ga+3*k+2]};
    b.size[k]=FlexDD{hull[sizes+9*gb+3*k],hull[sizes+9*gb+3*k+1],hull[sizes+9*gb+3*k+2]};
  }
  bool convex=common_ccd_production_uses_mjc_convex(ta,tb);
  if ((ta==0 && (tb==4 || tb==7 || tb==8)) || (tb==0 && (ta==4 || ta==7 || ta==8))) {
    int g=ta==0?gb:ga;int rb=info[header+33]+3*g;
    FlexDD rbound=FlexDD{hull[rb],hull[rb+1],hull[rb+2]};
    if(ta==0) found=gq_plane_convex(a,b,bound,rbound,hull,info,dist,wa,wb);
    else {found=gq_plane_convex(b,a,bound,rbound,hull,info,dist,wb,wa);}
  } else if (COMMON_CCD_USE_LIBCCD==0 && (convex || (ta==6 && tb==6))) {
    found=gq_source_distance(a,b,bound,hull,info,header,int(index),dist,wa,wb);
  } else if (COMMON_CCD_USE_LIBCCD!=0 && convex) {
    // The shared producer owns normal correction and ordered restart searches.
    return;
  } else {
    bool flip=ta>tb;int g1=flip?gb:ga,g2=flip?ga:gb;
    float3 s1=flex_dd3_high(FlexDD3{a.size[0],a.size[1],a.size[2]});
    float3 s2=flex_dd3_high(FlexDD3{b.size[0],b.size[1],b.size[2]});
    ContactGeom con[50];
    int bounds=info[header+33];
    int n=collide_pair_without_sdf(flip?tb:ta,flex_dd3_high(flip?pb:pa),flip?qb:qa,flip?s2:s1,hull[bounds+3*g1],
        flip?ta:tb,flex_dd3_high(flip?pa:pb),flip?qa:qb,flip?s1:s2,hull[bounds+3*g2],
        cutoffs[pair],0.0f,(dims[5]&(1<<19))!=0,con,g1,g2,hull,info,50);
    for (int k=0;k<n;k++) if (con[k].dist<dist.hi) {
      dist=flex_dd(con[k].dist);found=1;
      float sign=flip?-1.0f:1.0f;
      wa=flex_dd3(con[k].pos-.5f*sign*con[k].dist*con[k].normal);
      wb=flex_dd3(con[k].pos+.5f*sign*con[k].dist*con[k].normal);
    }
  }
  results[base]=float(found<0?-found:0);
  if (found<=0) {dist=bound;wa=flex_dd3(float3(0));wb=wa;}
  if (flex_dd_compare_exact(dist,bound)>0) dist=bound;
  results[base+1]=dist.hi;results[base+2]=dist.lo;results[base+3]=dist.tail;
  float3 x1=flex_dd3_high(wa),x2=flex_dd3_high(wb);
  results[base+4]=x1.x;results[base+5]=x1.y;results[base+6]=x1.z;
  results[base+7]=x2.x;results[base+8]=x2.y;results[base+9]=x2.z;
}

kernel void geometry_query_reduce(device const float* results [[buffer(0)]],
    device const int* spans [[buffer(1)]],device const int* meta [[buffer(2)]],
    device float* output [[buffer(3)]],device const int* mask [[buffer(4)]],
    constant int* dims [[buffer(5)]],uint index [[thread_position_in_grid]]) {
  int batch=dims[0],np=dims[1],ns=dims[3],ndata=dims[4];
  if (index>=uint(batch*ns) || (mask[0]&2)==0 || (dims[5]&8192)!=0) return;
  int world=int(index)/ns,sensor=int(index)%ns;
  if (mask[1+index]==0 || mask[1+batch*max(ns,1)+world]==0) return;
  int type=meta[10*sensor],adr=meta[10*sensor+6],dim=meta[10*sensor+5];
  if (type<39 || type>41) return;
  float bound=as_type<float>(meta[10*sensor+7]);FlexDD best=flex_dd(bound);
  int pick=-1;bool failure=false;
  for (int k=spans[2*sensor];k<spans[2*sensor]+spans[2*sensor+1];k++) {
    int base=(world*np+k)*10;
    if (results[base]!=0) {failure=true;break;}
    FlexDD d=FlexDD{results[base+1],results[base+2],results[base+3]};
    if (flex_dd_compare_exact(d,best)<0) {best=d;pick=base;}
  }
  int out=world*ndata+adr;
  if (failure) {for(int k=0;k<dim;k++) output[out+k]=NAN;return;}
  if(type==39) {output[out]=best.hi;return;}
  float3 a=pick<0?float3(0):float3(results[pick+4],results[pick+5],results[pick+6]);
  float3 b=pick<0?float3(0):float3(results[pick+7],results[pick+8],results[pick+9]);
  if(type==40) {float3 delta=b-a;float3 n=length_squared(delta)>0?normalize(delta):float3(0);
    output[out]=n.x;output[out+1]=n.y;output[out+2]=n.z;}
  else {output[out]=a.x;output[out+1]=a.y;output[out+2]=a.z;
    output[out+3]=b.x;output[out+4]=b.y;output[out+5]=b.z;}
}


kernel void geometry_query_prepare_terrain(device const float* quat [[buffer(0)]],
    device const float* mat [[buffer(1)]],device const float* matlo [[buffer(2)]],
    device const float* mattail [[buffer(3)]],device float* output [[buffer(4)]],
    device int* logical_map [[buffer(5)]],device const int* pairs [[buffer(6)]],
    device const int* mask [[buffer(7)]],constant int* dims [[buffer(8)]],
    constant int& has_matrix [[buffer(9)]],uint index [[thread_position_in_grid]]) {
  int batch=dims[0],np=dims[1],ng=dims[2],ns=dims[3],stride=batch*ng*9;
  if (index<uint(batch*np)) {
    int world=int(index)/np,pair=int(index)%np,sensor=pairs[3*pair];
    bool active=(mask[0]&2)!=0 && (dims[5]&8192)==0 && mask[1+world*max(ns,1)+sensor]!=0
        && mask[1+batch*max(ns,1)+world]!=0;
    logical_map[index]=active?int(index):-1;
  }
  if (index>=uint(batch*ng)) return;
  int base=9*int(index);FlexDD matrix[9];
  if(!has_matrix)
    gq_quat_matrix(float4(quat[4*index],quat[4*index+1],quat[4*index+2],quat[4*index+3]),matrix);
  else common_ccd_load_geom_matrix(mat,matlo,mattail,base,matrix);
  for(int k=0;k<9;k++) {output[base+k]=matrix[k].hi;output[stride+base+k]=matrix[k].lo;
    output[2*stride+base+k]=matrix[k].tail;}
}

kernel void geometry_query_finish_terrain(device const int* types [[buffer(0)]],
    device const int* pairs [[buffer(1)]],device const float* cutoffs [[buffer(2)]],
    device const float* hull [[buffer(3)]],device const int* info [[buffer(4)]],
    device float* results [[buffer(5)]],device const int* mask [[buffer(6)]],
    constant int* dims [[buffer(7)]],uint index [[thread_position_in_grid]]) {
  int batch=dims[0],np=dims[1],ng=dims[2],ns=dims[3];
  if(index>=uint(batch*np) || (mask[0]&2)==0 || (dims[5]&8192)!=0) return;
  int world=int(index)/np,pair=int(index)%np,sensor=pairs[3*pair];
  if(mask[1+world*max(ns,1)+sensor]==0 || mask[1+batch*max(ns,1)+world]==0) return;
  int a=pairs[3*pair+1],b=pairs[3*pair+2];
  bool terrain=types[a]==1 || types[b]==1;
  bool alternate=COMMON_CCD_USE_LIBCCD!=0 && common_ccd_production_uses_mjc_convex(types[a],types[b]);
  if(!terrain && !alternate) return;
  int header=9*ng,out=info[header+5]+int(index)*info[header+6],base=int(index)*10;
  float best=cutoffs[pair];int pick=-1;
  // Unsupported HField/HField and plane/HField are empty source callbacks.
  bool supported=(types[a]==1 && types[b]>=2 && types[b]<=7) ||
                 (types[b]==1 && types[a]>=2 && types[a]<=7) || alternate;
  int count=supported ? int(hull[out]) : 0;
  results[base]=supported ? hull[out+1] : 0;
  for(int k=0;k<count;k++) {int record=out+2+16*k;
    if(hull[record]<best) {best=hull[record];pick=record;}}
  results[base+1]=best;results[base+2]=results[base+3]=0;
  float3 p=pick<0?float3(0):float3(hull[pick+4],hull[pick+5],hull[pick+6]);
  float3 n=pick<0?float3(0):float3(hull[pick+1],hull[pick+2],hull[pick+3]);
  float sign=types[a]>types[b]?-1.0f:1.0f;
  float3 x1=p-.5f*sign*best*n,x2=p+.5f*sign*best*n;
  results[base+4]=x1.x;results[base+5]=x1.y;results[base+6]=x1.z;
  results[base+7]=x2.x;results[base+8]=x2.y;results[base+9]=x2.z;
}

// Source mjc_SDF / mjc_MeshSDF ignore the distance-query margin and publish
// penetrating minima in callback order. HFieldSDF is a pinned empty callback.
kernel void geometry_query_finish_sdf(device const int* types [[buffer(0)]],
    device const int* pairs [[buffer(1)]],device const float* cutoffs [[buffer(2)]],
    device const int* counts [[buffer(3)]],device const float* contacts [[buffer(4)]],
    device float* results [[buffer(5)]],device const int* mask [[buffer(6)]],
    constant int* dims [[buffer(7)]],uint index [[thread_position_in_grid]]) {
  int batch=dims[0],np=dims[1],ns=dims[3];
  if(index>=uint(batch*np) || (mask[0]&2)==0 || (dims[5]&8192)!=0) return;
  int world=int(index)/np,pair=int(index)%np,sensor=pairs[3*pair];
  if(mask[1+world*max(ns,1)+sensor]==0 || mask[1+batch*max(ns,1)+world]==0) return;
  int a=pairs[3*pair+1],b=pairs[3*pair+2],ta=types[a],tb=types[b];
  if((ta!=8 && tb!=8) || ta==0 || tb==0) return;
  int count=(ta==1 || tb==1)?0:counts[index],base=int(index)*10;
  results[base]=count<0?1:0;
  FlexDD best=flex_dd(cutoffs[pair]);int pick=-1;
  for(int k=0;k<count;k++) {int record=(int(index)*50+k)*37;
    FlexDD dist=FlexDD{contacts[record+12],contacts[record+9],contacts[record+10]};
    if(flex_dd_compare_exact(dist,best)<0) {best=dist;pick=record;}}
  FlexDD3 p=flex_dd3(float3(0)),n=p;
  if(pick>=0) {
    p=FlexDD3{FlexDD{contacts[pick+13],contacts[pick+25],contacts[pick+31]},
             FlexDD{contacts[pick+14],contacts[pick+26],contacts[pick+32]},
             FlexDD{contacts[pick+15],contacts[pick+27],contacts[pick+33]}};
    n=FlexDD3{FlexDD{contacts[pick+16],contacts[pick+28],contacts[pick+34]},
             FlexDD{contacts[pick+17],contacts[pick+29],contacts[pick+35]},
             FlexDD{contacts[pick+18],contacts[pick+30],contacts[pick+36]}};
  }
  FlexDD scale=flex_dd_source_mul(flex_dd(ta>tb?-.5f:.5f),best);
  float3 x1=flex_dd3_high(flex_dd3_sub(p,flex_dd3_scale(n,scale)));
  float3 x2=flex_dd3_high(flex_dd3_add(p,flex_dd3_scale(n,scale)));
  results[base+1]=best.hi;results[base+2]=best.lo;results[base+3]=best.tail;
  results[base+4]=x1.x;results[base+5]=x1.y;results[base+6]=x1.z;
  results[base+7]=x2.x;results[base+8]=x2.y;results[base+9]=x2.z;
}
