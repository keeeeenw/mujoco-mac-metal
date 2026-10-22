static inline FlexDD3 common_mpr_enclosure_portal_direction_012(thread CommonCCDVertex* p) {
  // Private CPU diagnostic switch: only the shared portal determinant convention.
  if (true) {
    FlexDD3 a=mpr_sub(p[2].minkowski,p[1].minkowski);
    FlexDD3 b=mpr_sub(p[3].minkowski,p[1].minkowski);
    return mpr_normalize(FlexDD3{
      flex_dd_source_sub(flex_dd_source_mul(a.y,b.z),flex_dd_source_mul(a.z,b.y)),
      flex_dd_source_sub(flex_dd_source_mul(a.z,b.x),flex_dd_source_mul(a.x,b.z)),
      flex_dd_source_sub(flex_dd_source_mul(a.x,b.y),flex_dd_source_mul(a.y,b.x))});
  }

  return mpr_normalize(mpr_cross(mpr_sub(p[2].minkowski,p[1].minkowski),
      mpr_sub(p[3].minkowski,p[1].minkowski)));
}
static inline int common_mpr_enclosure_support_unit_pair_012(
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD3 raw_direction, device const float* mesh_vertices,
    device const int* mesh_graph, thread CommonCCDVertex& output,
    device float* epa_support_trace, thread int& epa_support_count, int query_budget) {
  if(epa_support_count>=query_budget)return -20;
  FlexDD length=flex_dd_sqrt(flex_dd3_dot(raw_direction,raw_direction));
  FlexDD3 direction=flex_dd_compare_exact(length,flex_dd(1.0e-15f))>0
      ? flex_dd3_div(raw_direction,length)
      : FlexDD3{flex_dd(1.0f),flex_dd(0.0f),flex_dd(0.0f)};
  int status=mpr_support_pair(a,b,direction,mesh_vertices,mesh_graph,
                                     output);
  if (!status)
    common_ccd_diag_epa_support(epa_support_trace,epa_support_count,
                                direction,output);
  return status;
}
static inline int common_mpr_enclosure_initialize_p3_012(
    thread const CommonCCDVertex* seed,
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD gjk_distance, device const float* mesh_vertices,
    device const int* mesh_graph, device FlexDDVertex* vertices,
    device int2* vertex_ids,
    thread FlexDD3& center, thread int3* face_vertices,
    thread int3* adjacency, device float* epa_support_trace,
    thread int& epa_support_count, int query_budget) {
  for (int i=0;i<3;i++) {
    vertices[i]=common_ccd_flex_vertex(seed[i]);
    if (vertex_ids!=nullptr) vertex_ids[i]=int2(seed[i].index_a,seed[i].index_b);
  }
  FlexDD3 d1=flex_dd3_sub(seed[1].minkowski,seed[0].minkowski);
  FlexDD3 d2=flex_dd3_sub(seed[2].minkowski,seed[0].minkowski);
  FlexDD3 normal=flex_dd3_cross(d1,d2);
  FlexDD normal_length=flex_dd_sqrt(flex_dd3_dot(normal,normal));
  if (flex_dd_compare_exact(normal_length,flex_dd(1.0e-15f))<0) return 1;
  FlexDD3 center_sum=flex_dd3_add(seed[0].minkowski,seed[1].minkowski);
  center_sum=flex_dd3_add(center_sum,seed[2].minkowski);
  // Pinned mjc_polytope3 multiplies by the mjtNum (binary64) expression
  // 1.0/3.0. Encoding the rounded float32 reciprocal here changes the
  // centroid by several ULPs of the represented inputs and can alter EPA's
  // source-ordered closest-face tie.
  center=flex_dd3_scale(center_sum,FlexDD{
      0.3333333432674408f,-9.934107758624577e-9f,2.7755575615628914e-16f});
  CommonCCDVertex v5,v4;
  if (common_mpr_enclosure_support_unit_pair_012(a,b,flex_dd3_neg(normal),mesh_vertices,
                                  mesh_graph,v5,epa_support_trace,
                                  epa_support_count,query_budget) ||
      common_mpr_enclosure_support_unit_pair_012(a,b,normal,mesh_vertices,mesh_graph,v4,
                                  epa_support_trace,epa_support_count,query_budget))
    return 2;
  vertices[3]=common_ccd_flex_vertex(v5);
  vertices[4]=common_ccd_flex_vertex(v4);
  if (vertex_ids!=nullptr) {
    vertex_ids[3]=int2(v5.index_a,v5.index_b);
    vertex_ids[4]=int2(v4.index_a,v4.index_b);
  }
  if (common_ccd_tri_point_intersect(seed[0].minkowski,
          seed[1].minkowski,seed[2].minkowski,v4.minkowski) ||
      common_ccd_tri_point_intersect(seed[0].minkowski,
          seed[1].minkowski,seed[2].minkowski,v5.minkowski)) return 3;
  if (flex_dd_compare_exact(gjk_distance,flex_dd(1.0e-14f))>0 &&
      !common_ccd_test_tetra_source(seed[0].minkowski,
          seed[1].minkowski,seed[2].minkowski,v4.minkowski) &&
      !common_ccd_test_tetra_source(seed[0].minkowski,
          seed[1].minkowski,seed[2].minkowski,v5.minkowski))
    return 4;

  face_vertices[0]=int3(4,0,1); face_vertices[1]=int3(4,2,0);
  face_vertices[2]=int3(4,1,2); face_vertices[3]=int3(3,1,0);
  face_vertices[4]=int3(3,0,2); face_vertices[5]=int3(3,2,1);
  adjacency[0]=int3(1,3,2); adjacency[1]=int3(2,4,0);
  adjacency[2]=int3(0,5,1); adjacency[3]=int3(5,0,4);
  adjacency[4]=int3(3,1,5); adjacency[5]=int3(4,2,3);
  for (int i=0;i<6;i++) {
    FlexEpaFace face;
    int3 f=face_vertices[i];
    if (!flex_epa_make_face_dd(vertices,center,f.x,f.y,f.z,adjacency[i],face) ||
        flex_dd_compare_exact(face.dist2_dd,flex_dd(1.0e-30f))<0) return 5;
  }
  return 0;
}
// Private complete-prediction-admitted face/vertex extraction.
static inline bool common_fv_finite_012(FlexDD x) {
  return isfinite(x.hi)&&isfinite(x.lo)&&isfinite(x.tail);
}
static inline bool common_fv_finite3_012(FlexDD3 x) {
  return common_fv_finite_012(x.x)&&common_fv_finite_012(x.y)&&common_fv_finite_012(x.z);
}
static inline FlexDD common_fv_max_012(FlexDD a,FlexDD b) {
  return flex_dd_compare_exact(a,b)>0?a:b;
}
static inline void common_fv_affine_012(FlexDD3 s1,FlexDD3 s2,FlexDD3 s3,FlexDD3 p,thread FlexDD* output) {
  FlexDD m14=flex_epa_affine_minor(FlexDD2{s1.y,s1.z},
      FlexDD2{s2.y,s2.z},FlexDD2{s3.y,s3.z});
  FlexDD m24=flex_epa_affine_minor(FlexDD2{s1.x,s1.z},
      FlexDD2{s2.x,s2.z},FlexDD2{s3.x,s3.z});
  FlexDD m34=flex_epa_affine_minor(FlexDD2{s1.x,s1.y},
      FlexDD2{s2.x,s2.y},FlexDD2{s3.x,s3.y});
  FlexDD abs14=flex_dd_abs(m14), abs24=flex_dd_abs(m24), abs34=flex_dd_abs(m34);
  FlexDD mmax; FlexDD2 a2,b2,c2,q2;
  if (flex_dd_compare(abs14,abs24)>=0 && flex_dd_compare(abs14,abs34)>=0) {
    mmax=m14; a2=FlexDD2{s1.y,s1.z}; b2=FlexDD2{s2.y,s2.z};
    c2=FlexDD2{s3.y,s3.z}; q2=FlexDD2{p.y,p.z};
  } else if (flex_dd_compare(abs24,abs34)>=0) {
    mmax=m24; a2=FlexDD2{s1.x,s1.z}; b2=FlexDD2{s2.x,s2.z};
    c2=FlexDD2{s3.x,s3.z}; q2=FlexDD2{p.x,p.z};
  } else {
    mmax=m34; a2=FlexDD2{s1.x,s1.y}; b2=FlexDD2{s2.x,s2.y};
    c2=FlexDD2{s3.x,s3.y}; q2=FlexDD2{p.x,p.y};
  }
  FlexDD weights[3]={
      flex_dd_source_div(flex_epa_affine_area(q2,b2,c2),mmax),
      flex_dd_source_div(flex_epa_affine_area(q2,c2,a2),mmax),
      flex_dd_source_div(flex_epa_affine_area(q2,a2,b2),mmax)};
  for(int i=0;i<3;i++)output[i]=weights[i];
}
static inline bool common_fv_scaled_unit_012(FlexDD3 value,thread int& exponent,
    thread FlexDD3& scaled,thread FlexDD& squared,thread FlexDD& length,
    thread FlexDD& inverse,thread FlexDD3& unit) {
  exponent=0;scaled=flex_dd3(float3(0));squared=length=inverse=flex_dd(0);unit=scaled;
  if(!common_fv_finite3_012(value))return false;
  bool found=false;
  for(int axis=0;axis<3;axis++) {
    FlexDD c=flex_dd3_component(value,axis);float words[3]={c.hi,c.lo,c.tail};
    for(int limb=0;limb<3;limb++)if((as_type<uint>(words[limb])&0x7fffffffu)!=0u) {
      int e=flex_float_exponent(words[limb]);
      if(!found||e>exponent){found=true;exponent=e;}
    }
  }
  if(!found)return false;
  scaled=FlexDD3{flex_dd_scalbn(value.x,-exponent),
      flex_dd_scalbn(value.y,-exponent),flex_dd_scalbn(value.z,-exponent)};
  FlexDD3 back=FlexDD3{flex_dd_scalbn(scaled.x,exponent),
      flex_dd_scalbn(scaled.y,exponent),flex_dd_scalbn(scaled.z,exponent)};
  if(!common_fv_finite3_012(scaled)||!common_fv_finite3_012(back)||
     flex_dd_compare_exact(back.x,value.x)!=0||
     flex_dd_compare_exact(back.y,value.y)!=0||
     flex_dd_compare_exact(back.z,value.z)!=0)return false;
  squared=flex_dd_source_dot3(scaled,scaled);
  length=flex_dd_source_sqrt(squared);
  if(!common_fv_finite_012(length)||flex_dd_compare_exact(length,flex_dd(0))<=0)return false;
  inverse=flex_dd_source_div(flex_dd(1),length);unit=mpr_scale(scaled,inverse);
  return common_fv_finite_012(inverse)&&common_fv_finite3_012(unit);
}
static inline int common_fv_recover_scaled_012(
    const thread FlexEpaFace& face,device const FlexDDVertex* vertices,int vertex_count,
    thread CommonCCDSupportObject& a,thread CommonCCDSupportObject& b,
    device const float* mesh_vertices,device const int* mesh_graph,
    int max_iterations,FlexDD tolerance,thread int& support_count,
    thread CommonCCDEpaResult& out,thread ContactGeom& contact) {
  if(max_iterations<=0||max_iterations>2147483645||vertex_count<3||
     face.a<0||face.b<0||face.c<0||face.a>=vertex_count||
     face.b>=vertex_count||face.c>=vertex_count)return -20;
  
  int cap=max_iterations+2;
  FlexDD zero=flex_dd(0),one=flex_dd(1),old_depth=out.distance;
  FlexDD3 old_direction=mpr_sub(out.witness_a,out.witness_b);
  FlexDD best_depth=flex_dd(3.402823466e+38f),best_rounding=zero;
  FlexDD3 best1=flex_dd3(float3(0)),best2=best1,best_triangle[3];
  FlexDD best_weights[3]={zero,zero,zero};
  CommonCCDVertex best_query;best_query.point_a=best_query.point_b=best_query.minkowski=best1;
  int best_side=-1,best_qbefore=0,best_qafter=0;
  FlexDD3 empty_triangle[3]={best1,best1,best1};
  
  for(int side=0;side<2;side++) {
    if(cap-support_count<=1) {
      break;
    }
    FlexDD3 triangle[3];int ids[3]={face.a,face.b,face.c};
    for(int i=0;i<3;i++)triangle[i]=side?vertices[ids[i]].b:vertices[ids[i]].a;
    FlexDD3 e=mpr_sub(triangle[1],triangle[0]),f=mpr_sub(triangle[2],triangle[0]);
    FlexDD3 n=mpr_cross(e,f);
    FlexDD3 unscaled_cross=n,scaled,normalized;
    FlexDD squared,length,inverse;int exponent;
    bool scale_ok=common_fv_scaled_unit_012(n,exponent,scaled,squared,length,inverse,normalized);
    
    if(!scale_ok)continue;
    
    
    n=normalized;
    if(flex_dd_compare_exact(flex_dd_source_dot3(n,old_direction),zero)<0)n=flex_dd3_neg(n);
    if(!common_fv_finite3_012(n))continue;
    CommonCCDVertex query;query.point_a=query.point_b=query.minkowski=flex_dd3(float3(0));query.index_a=query.index_b=-1;int before=support_count;support_count++;
    int status=mpr_support_pair(a,b,n,mesh_vertices,mesh_graph,query);
    FlexDD weights[3]={zero,zero,zero};
    
    if(status)return -20;
    FlexDD scale=one;
    for(int k=0;k<3;k++) {
      scale=common_fv_max_012(scale,flex_dd_abs(flex_dd3_component(query.point_a,k)));
      scale=common_fv_max_012(scale,flex_dd_abs(flex_dd3_component(query.point_b,k)));
      for(int i=0;i<3;i++)scale=common_fv_max_012(scale,flex_dd_abs(flex_dd3_component(triangle[i],k)));
    }
    FlexDD rounding=flex_dd_source_mul(flex_dd(0x1p-44f),scale);
    FlexDD plane_error=flex_dd_source_dot3(n,mpr_sub(side?query.point_b:query.point_a,triangle[0]));
    if(!common_fv_finite_012(rounding)||!common_fv_finite_012(plane_error)||
       flex_dd_compare_exact(flex_dd_abs(plane_error),rounding)>0)continue;
    FlexDD3 opposite=side?query.point_a:query.point_b;
    FlexDD h=flex_dd_source_dot3(n,mpr_sub(triangle[0],opposite));
    if(side)h=flex_dd_neg(h);
    if(!common_fv_finite_012(h)||flex_dd_compare_exact(h,zero)<=0||
       flex_dd_compare_exact(flex_dd_abs(flex_dd_source_sub(h,old_depth)),tolerance)>0)continue;
    FlexDD3 projection=mpr_add(opposite,mpr_scale(n,side?flex_dd_neg(h):h));
    common_fv_affine_012(triangle[0],triangle[1],triangle[2],projection,weights);
    if(!common_fv_finite_012(weights[0])||!common_fv_finite_012(weights[1])||!common_fv_finite_012(weights[2])||
       flex_dd_compare_exact(weights[0],zero)<0||flex_dd_compare_exact(weights[1],zero)<0||
       flex_dd_compare_exact(weights[2],zero)<0)continue;
    FlexDD sum=flex_dd_source_add(flex_dd_source_add(weights[0],weights[1]),weights[2]);
    FlexDD partition_error=flex_dd_abs(flex_dd_source_sub(sum,one));
    if(flex_dd_compare_exact(partition_error,flex_dd_source_mul(flex_dd(0x1p-45f),common_fv_max_012(one,flex_dd_abs(sum))))>0)continue;
    FlexDD3 facepoint=flex_dd3_lincomb_source(triangle,weights,3);
    FlexDD3 x1=side?opposite:facepoint,x2=side?facepoint:opposite;
    FlexDD3 delta=mpr_sub(x1,x2);
    FlexDD3 delta_scaled,delta_unit;FlexDD delta_squared,delta_length,delta_inverse;
    int delta_exponent;
    bool delta_ok=common_fv_scaled_unit_012(delta,delta_exponent,delta_scaled,delta_squared,
        delta_length,delta_inverse,delta_unit);
    
    if(!delta_ok)continue;
    FlexDD depth=flex_dd_scalbn(delta_length,delta_exponent);
    if(flex_dd_compare_exact(flex_dd_scalbn(depth,-delta_exponent),delta_length)!=0)continue;
    if(!common_fv_finite_012(depth)||flex_dd_compare_exact(depth,zero)<=0||
       flex_dd_compare_exact(flex_dd_abs(flex_dd_source_sub(depth,old_depth)),tolerance)>0)continue;
    
    if(best_side<0||flex_dd_compare_exact(depth,best_depth)<0) {
      best_side=side;best_depth=depth;best1=x1;best2=x2;best_rounding=rounding;
      best_query=query;best_qbefore=before;best_qafter=support_count;
      for(int i=0;i<3;i++){best_triangle[i]=triangle[i];best_weights[i]=weights[i];}
    }
  }
  if(best_side<0||support_count>=cap)return 0;
  FlexDD3 final_delta=mpr_sub(best1,best2),final_scaled,n;
  FlexDD final_squared,final_length,final_inverse;int final_exponent;
  bool final_ok=common_fv_scaled_unit_012(final_delta,final_exponent,final_scaled,final_squared,
      final_length,final_inverse,n);
  
  if(!final_ok)return 0;
  CommonCCDVertex query;query.point_a=query.point_b=query.minkowski=flex_dd3(float3(0));query.index_a=query.index_b=-1;int before=support_count;support_count++;
  int status=mpr_support_pair(a,b,n,mesh_vertices,mesh_graph,query);
  FlexDD gap=status?zero:flex_dd_source_sub(flex_dd_source_dot3(n,query.minkowski),best_depth);
  
  if(status)return -20;
  if(!common_fv_finite_012(gap)||flex_dd_compare_exact(gap,tolerance)>0||
     flex_dd_compare_exact(gap,flex_dd_neg(best_rounding))<0)return 0;
  out.distance=best_depth;out.witness_a=best1;out.witness_b=best2;out.normal=n;
  out.position=mpr_scale(mpr_add(best1,best2),flex_dd(.5f));
  contact.normal=flex_dd3_high(n);contact.pos=flex_dd3_high(out.position);
  contact.dist=-((best_depth.hi+best_depth.lo)+best_depth.tail);
  contact.t1=float3(0);make_frame(contact.normal,contact.t1,contact.t2);
  
  return 1;
}
// Private complete-prediction-admitted configured face-overlap recovery.
static inline int common_fo_count_012(thread const CommonCCDSupportObject& object,int live) {
  if(!common_fv_finite_012(object.margin))return 0;
  int count=0;
  if(object.kind==COMMON_CCD_HFIELD_PRISM) {
    if(object.vertex_count!=6)return 0;count=6;
  } else if(object.geom>=0 && object.kind==COMMON_CCD_BOX)count=8;
  else if(object.geom>=0 && object.kind==COMMON_CCD_HULL)count=object.vertex_count;
  if(count<3 || count>live || count>2147483647/3)return 0;
  if(object.kind==COMMON_CCD_HFIELD_PRISM || object.kind==COMMON_CCD_HULL) {
    if(object.vertex_offset<0 || object.vertex_offset>2147483647/3-count)return 0;
  }
  return count;
}
static inline bool common_fo_vertex_012(thread FlexDD3& result,
    thread const CommonCCDSupportObject& object,int index,FlexDD3 outward,
    device const float* vertices) {
  FlexDD3 local;
  if(object.kind==COMMON_CCD_BOX) {
    local=FlexDD3{(index&1)?object.size[0]:flex_dd_neg(object.size[0]),
                 (index&2)?object.size[1]:flex_dd_neg(object.size[1]),
                 (index&4)?object.size[2]:flex_dd_neg(object.size[2])};
  } else {
    int address=3*(object.vertex_offset+index);
    if(object.kind==COMMON_CCD_HFIELD_PRISM) {
      local=FlexDD3{
       FlexDD{vertices[address],object.hfield_residual[index].x.hi,object.hfield_residual[index].x.lo},
       FlexDD{vertices[address+1],object.hfield_residual[index].y.hi,object.hfield_residual[index].y.lo},
       FlexDD{vertices[address+2],object.hfield_residual[index].z.hi,object.hfield_residual[index].z.lo}};
    } else local=flex_dd3(float3(vertices[address],vertices[address+1],vertices[address+2]));
  }
  // Preserve this native object's original support-reader association. The
  // prism vertices are already in the shared local collision frame; ordinary
  // geometry transforms then adds its existing world-direction half margin.
  if(object.kind==COMMON_CCD_HFIELD_PRISM)result=local;
  else {
    result=common_ccd_local_to_world(object.mat,object.pos,local);
    if(object.geom>=0 && flex_dd_compare_exact(object.margin,flex_dd(0))>0)
      result=flex_dd3_source_madd(outward,flex_dd_mul(object.margin,flex_dd(.5f)),result);
  }
  return common_fv_finite3_012(result);
}
static inline bool common_fo_member_012(thread FlexDD3& result,
    thread const CommonCCDSupportObject& object,FlexDD3 outward,FlexDD3 target,
    int live,FlexDD rounding,device const float* vertices,
    thread FlexDD* certificate) {
  int count=common_fo_count_012(object,live);
  if(!count || !common_fv_finite_012(rounding) || flex_dd_compare_exact(rounding,flex_dd(0))<0)return false;
  int anchor=0;FlexDD3 a,v;
  if(!common_fo_vertex_012(a,object,0,outward,vertices))return false;
  FlexDD best=flex_dd_source_dot3(outward,a);
  if(!common_fv_finite_012(best))return false;
  for(int i=1;i<count;i++) {
    if(!common_fo_vertex_012(v,object,i,outward,vertices))return false;
    FlexDD score=flex_dd_source_dot3(outward,v);
    if(!common_fv_finite_012(score))return false;
    if(flex_dd_compare_exact(score,best)>0){anchor=i;best=score;a=v;}
  }
  for(int i=0;i<count;i++)if(i!=anchor) {
    FlexDD3 b;if(!common_fo_vertex_012(b,object,i,outward,vertices))return false;
    for(int j=i+1;j<count;j++)if(j!=anchor) {
      FlexDD3 c;
      if(!common_fo_vertex_012(c,object,j,outward,vertices))return false;
      FlexDD3 cross=mpr_cross(mpr_sub(b,a),mpr_sub(c,a));
      if(!common_fv_finite3_012(cross))continue;
      if(flex_dd_compare_exact(cross.x,flex_dd(0))==0 &&
         flex_dd_compare_exact(cross.y,flex_dd(0))==0 &&
         flex_dd_compare_exact(cross.z,flex_dd(0))==0)continue;
      FlexDD weights[3];common_fv_affine_012(a,b,c,target,weights);
      if(!common_fv_finite_012(weights[0])||!common_fv_finite_012(weights[1])||!common_fv_finite_012(weights[2])||
         flex_dd_compare_exact(weights[0],flex_dd(0))<0||
         flex_dd_compare_exact(weights[1],flex_dd(0))<0||
         flex_dd_compare_exact(weights[2],flex_dd(0))<0)continue;
      FlexDD sum=flex_dd_source_add(flex_dd_source_add(weights[0],weights[1]),weights[2]);
      if(!common_fv_finite_012(sum)||flex_dd_compare_exact(flex_dd_abs(flex_dd_source_sub(sum,flex_dd(1))),
         flex_dd_source_mul(flex_dd(0x1p-45f),common_fv_max_012(flex_dd(1),flex_dd_abs(sum))))>0)continue;
      FlexDD3 triangle[3]={a,b,c};FlexDD3 point=flex_dd3_lincomb_source(triangle,weights,3);
      bool valid=common_fv_finite3_012(point);
      for(int k=0;k<3;k++)if(flex_dd_compare_exact(flex_dd_abs(flex_dd_source_sub(
          flex_dd3_component(point,k),flex_dd3_component(target,k))),rounding)>0)valid=false;
      if(!valid)continue;
      result=point;certificate[0]=flex_dd(float(count));certificate[1]=flex_dd(float(anchor));
      certificate[2]=flex_dd(float(i));certificate[3]=flex_dd(float(j));
      for(int k=0;k<3;k++) {
        certificate[4+k]=weights[k];certificate[7+k]=flex_dd3_component(a,k);
        certificate[10+k]=flex_dd3_component(b,k);certificate[13+k]=flex_dd3_component(c,k);
        certificate[16+k]=flex_dd3_component(target,k);certificate[19+k]=flex_dd3_component(point,k);
      }
      certificate[22]=rounding;certificate[23]=flex_dd(float(live));return true;
    }
  }
  return false;
}
static inline int common_fo_configured_recover_012(
    const thread FlexEpaFace& face,device const FlexDDVertex* vertices,int vertex_count,
    thread CommonCCDSupportObject& a,thread CommonCCDSupportObject& b,
    device const float* mesh_vertices,device const int* mesh_graph,
    int max_iterations,FlexDD tolerance,thread int& support_count,
    thread CommonCCDEpaResult& out,thread ContactGeom& contact) {
  if(max_iterations<=0||max_iterations>2147483645||vertex_count<3||
     face.a<0||face.b<0||face.c<0||face.a>=vertex_count||
     face.b>=vertex_count||face.c>=vertex_count)return -20;
  
  int cap=max_iterations+2;
  FlexDD zero=flex_dd(0),one=flex_dd(1),old_depth=out.distance;
  FlexDD3 old_direction=mpr_sub(out.witness_a,out.witness_b);
  FlexDD best_depth=flex_dd(3.402823466e+38f),best_rounding=zero;
  FlexDD3 best1=flex_dd3(float3(0)),best2=best1,best_triangle[3];
  FlexDD best_weights[3]={zero,zero,zero};
  CommonCCDVertex best_query;best_query.point_a=best_query.point_b=best_query.minkowski=best1;
  int best_side=-1,best_qbefore=0,best_qafter=0;
  FlexDD3 empty_triangle[3]={best1,best1,best1};
  
  for(int side=0;side<2;side++) {
    if(cap-support_count<=1) {
      break;
    }
    FlexDD3 triangle[3];int ids[3]={face.a,face.b,face.c};
    for(int i=0;i<3;i++)triangle[i]=side?vertices[ids[i]].b:vertices[ids[i]].a;
    FlexDD3 e=mpr_sub(triangle[1],triangle[0]),f=mpr_sub(triangle[2],triangle[0]);
    FlexDD3 n=mpr_cross(e,f);
    FlexDD3 unscaled_cross=n,scaled,normalized;
    FlexDD squared,length,inverse;int exponent;
    bool scale_ok=common_fv_scaled_unit_012(n,exponent,scaled,squared,length,inverse,normalized);
    
    if(!scale_ok)continue;
    
    
    n=normalized;
    if(flex_dd_compare_exact(flex_dd_source_dot3(n,old_direction),zero)<0)n=flex_dd3_neg(n);
    if(!common_fv_finite3_012(n))continue;
    CommonCCDVertex query;query.point_a=query.point_b=query.minkowski=flex_dd3(float3(0));query.index_a=query.index_b=-1;int before=support_count;support_count++;
    int status=mpr_support_pair(a,b,n,mesh_vertices,mesh_graph,query);
    FlexDD weights[3]={zero,zero,zero};
    
    if(status)return -20;
    FlexDD scale=one;
    for(int k=0;k<3;k++) {
      scale=common_fv_max_012(scale,flex_dd_abs(flex_dd3_component(query.point_a,k)));
      scale=common_fv_max_012(scale,flex_dd_abs(flex_dd3_component(query.point_b,k)));
      for(int i=0;i<3;i++)scale=common_fv_max_012(scale,flex_dd_abs(flex_dd3_component(triangle[i],k)));
    }
    FlexDD rounding=flex_dd_source_mul(flex_dd(0x1p-44f),scale);
    FlexDD plane_error=flex_dd_source_dot3(n,mpr_sub(side?query.point_b:query.point_a,triangle[0]));
    if(!common_fv_finite_012(rounding)||!common_fv_finite_012(plane_error)||
       flex_dd_compare_exact(flex_dd_abs(plane_error),rounding)>0)continue;
    FlexDD3 opposite=side?query.point_a:query.point_b;
    FlexDD h=flex_dd_source_dot3(n,mpr_sub(triangle[0],opposite));
    if(side)h=flex_dd_neg(h);
    if(!common_fv_finite_012(h)||flex_dd_compare_exact(h,zero)<=0||
       flex_dd_compare_exact(flex_dd_abs(flex_dd_source_sub(h,old_depth)),tolerance)>0)continue;
    FlexDD3 projection=mpr_add(opposite,mpr_scale(n,side?flex_dd_neg(h):h));
    common_fv_affine_012(triangle[0],triangle[1],triangle[2],projection,weights);
    if(!common_fv_finite_012(weights[0])||!common_fv_finite_012(weights[1])||!common_fv_finite_012(weights[2])||
       flex_dd_compare_exact(weights[0],zero)<0||flex_dd_compare_exact(weights[1],zero)<0||
       flex_dd_compare_exact(weights[2],zero)<0) {
      FlexDD third=flex_dd_source_div(one,flex_dd(3));
      weights[0]=weights[1]=weights[2]=third;
      FlexDD3 center=flex_dd3_lincomb_source(triangle,weights,3);
      FlexDD3 target=mpr_add(center,mpr_scale(n,side?h:flex_dd_neg(h)));
      FlexDD3 outward=side?n:flex_dd3_neg(n);FlexDD certificate[24];
      if(!common_fo_member_012(opposite,side?a:b,outward,target,cap,
                        rounding,mesh_vertices,certificate))continue;
      
    }
    FlexDD sum=flex_dd_source_add(flex_dd_source_add(weights[0],weights[1]),weights[2]);
    FlexDD partition_error=flex_dd_abs(flex_dd_source_sub(sum,one));
    if(flex_dd_compare_exact(partition_error,flex_dd_source_mul(flex_dd(0x1p-45f),common_fv_max_012(one,flex_dd_abs(sum))))>0)continue;
    FlexDD3 facepoint=flex_dd3_lincomb_source(triangle,weights,3);
    FlexDD3 x1=side?opposite:facepoint,x2=side?facepoint:opposite;
    FlexDD3 delta=mpr_sub(x1,x2);
    FlexDD3 delta_scaled,delta_unit;FlexDD delta_squared,delta_length,delta_inverse;
    int delta_exponent;
    bool delta_ok=common_fv_scaled_unit_012(delta,delta_exponent,delta_scaled,delta_squared,
        delta_length,delta_inverse,delta_unit);
    
    if(!delta_ok)continue;
    FlexDD depth=flex_dd_scalbn(delta_length,delta_exponent);
    if(flex_dd_compare_exact(flex_dd_scalbn(depth,-delta_exponent),delta_length)!=0)continue;
    if(!common_fv_finite_012(depth)||flex_dd_compare_exact(depth,zero)<=0||
       flex_dd_compare_exact(flex_dd_abs(flex_dd_source_sub(depth,old_depth)),tolerance)>0)continue;
    
    if(best_side<0||flex_dd_compare_exact(depth,best_depth)<0) {
      best_side=side;best_depth=depth;best1=x1;best2=x2;best_rounding=rounding;
      best_query=query;best_qbefore=before;best_qafter=support_count;
      for(int i=0;i<3;i++){best_triangle[i]=triangle[i];best_weights[i]=weights[i];}
    }
  }
  if(best_side<0||support_count>=cap)return 0;
  FlexDD3 final_delta=mpr_sub(best1,best2),final_scaled,n;
  FlexDD final_squared,final_length,final_inverse;int final_exponent;
  bool final_ok=common_fv_scaled_unit_012(final_delta,final_exponent,final_scaled,final_squared,
      final_length,final_inverse,n);
  
  if(!final_ok)return 0;
  CommonCCDVertex query;query.point_a=query.point_b=query.minkowski=flex_dd3(float3(0));query.index_a=query.index_b=-1;int before=support_count;support_count++;
  int status=mpr_support_pair(a,b,n,mesh_vertices,mesh_graph,query);
  FlexDD gap=status?zero:flex_dd_source_sub(flex_dd_source_dot3(n,query.minkowski),best_depth);
  
  if(status)return -20;
  if(!common_fv_finite_012(gap)||flex_dd_compare_exact(gap,tolerance)>0||
     flex_dd_compare_exact(gap,flex_dd_neg(best_rounding))<0)return 0;
  out.distance=best_depth;out.witness_a=best1;out.witness_b=best2;out.normal=n;
  out.position=mpr_scale(mpr_add(best1,best2),flex_dd(.5f));
  contact.normal=flex_dd3_high(n);contact.pos=flex_dd3_high(out.position);
  contact.dist=-((best_depth.hi+best_depth.lo)+best_depth.tail);
  contact.t1=float3(0);make_frame(contact.normal,contact.t1,contact.t2);
  
  return 1;
}
static inline bool common_ellipsoid_vertex_certificate_012(
    const thread FlexEpaFace&,device const FlexDDVertex*,
    thread CommonCCDSupportObject&,thread CommonCCDSupportObject&,
    device const float*,device const int*,int,FlexDD,thread int&,
    thread CommonCCDEpaResult&,thread ContactGeom&);
static inline CommonCCDEpaResult common_mpr_enclosure_epa_012(
    thread const CommonCCDVertex* seed, int seed_count,
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    int max_iterations, FlexDD tolerance, FlexDD gjk_distance,
    device const float* mesh_vertices, device const int* mesh_graph,
    device FlexDDVertex* vertices, device int2* vertex_ids,
    device FlexEpaFace* faces,
    device int* face_map, device int2* horizon,
    device FlexHorizonFrame* horizon_stack, int vertex_capacity,
    int face_capacity, int horizon_capacity, int horizon_stack_capacity,
    thread ContactGeom& contact, device float* epa_support_trace,
    device float* epa_face_trace, thread int& epa_support_count,
    thread int& epa_face_count) {
  CommonCCDEpaResult out;
  out.distance=flex_dd(3.402823466e+38f);
  out.status=1;
  out.face_vertices=int3(-1);
  epa_support_count=0;
  epa_face_count=0;
  // engine_collision_gjk.c::epa uses mjMINEPATOL for discrete pairs; the
  // pinned double build defines that as mjMINVAL (1e-15), independent of
  // model.opt.ccd_tolerance. Smooth pairs keep the configured tolerance.
  FlexDD epa_tolerance=tolerance;
  if (max_iterations<=0 || seed_count!=4 ||
      vertex_capacity<6 || face_capacity<6 || horizon_capacity<3 ||
      horizon_stack_capacity<1) {
    out.status=2;  // Invalid invocation is not a source no-contact result.
    return out;
  }
  CommonCCDVertex local_seed[4];
  for (int i=0;i<seed_count;i++) {
    local_seed[i]=seed[i];
    vertices[i]=common_ccd_flex_vertex(local_seed[i]);
    if (vertex_ids!=nullptr) vertex_ids[i]=int2(local_seed[i].index_a,local_seed[i].index_b);
  }
  int vertex_count=seed_count;
  FlexDD3 center=flex_dd3(float3(0.0f));
  for (int i=0;i<seed_count;i++)
    center=flex_dd3_add(center,local_seed[i].minkowski);
  FlexDD center_scale=seed_count==4 ? flex_dd(0.25f) :
      seed_count==3 ? flex_dd_source_div(flex_dd(1.0f),flex_dd(3.0f))
                    : flex_dd(0.5f);
  center=flex_dd3_scale(center,center_scale);
  int face_count=0;
  int map_count=0;
  FlexEpaFace face;
  int3 face_vertices[6];
  int3 adjacency[6];
  int initial_faces=0;
  bool p3_initialized=false;
  if (seed_count==4) {
    face_vertices[0]=int3(0,1,2); face_vertices[1]=int3(0,3,1);
    face_vertices[2]=int3(0,2,3); face_vertices[3]=int3(3,2,1);
    adjacency[0]=int3(1,3,2); adjacency[1]=int3(2,3,0);
    adjacency[2]=int3(0,3,1); adjacency[3]=int3(2,0,1);
    initial_faces=4;
    // polytope4 tests source-ordered faces before testing tetra containment.
    // A face at the origin is replaced by that exact ordered P3 simplex.
    int fallback_face=-1;
    for (int i=0;i<4;i++) {
    if (common_ccd_face_below_source_limit(
              vertices,center,face_vertices[i],adjacency[i],
              flex_dd(1.0e-30f))) { fallback_face=i; break; }
    }
    if (fallback_face>=0) {
      int3 f=face_vertices[fallback_face];
      CommonCCDVertex p3seed[3]={local_seed[f.x],local_seed[f.y],local_seed[f.z]};
      int init=common_mpr_enclosure_initialize_p3_012(p3seed,a,b,gjk_distance,
          mesh_vertices,mesh_graph,vertices,vertex_ids,center,face_vertices,
          adjacency,epa_support_trace,epa_support_count,max_iterations+2);
      if (init) {
        // Source polytope initialization rejects boundary/degenerate seeds
        // without a contact. A support-read failure remains an error.
        out.status=init==2 ? 2 : 1;
        if (out.status==1) out.distance=flex_dd(0.0f);
        return out;
      }
      vertex_count=5; initial_faces=6; p3_initialized=true;
    }
    if (!p3_initialized &&
        !common_ccd_test_tetra_source(local_seed[0].minkowski,
                         local_seed[1].minkowski,
                         local_seed[2].minkowski,
                         local_seed[3].minkowski)) {
      out.status=1; out.distance=flex_dd(0.0f); return out;
    }
  } else if (seed_count==3) {
    int init=common_mpr_enclosure_initialize_p3_012(local_seed,a,b,gjk_distance,
        mesh_vertices,mesh_graph,vertices,vertex_ids,center,face_vertices,
        adjacency,epa_support_trace,epa_support_count,max_iterations+2);
    if (init) {
      out.status=init==2 ? 2 : 1;
      if (out.status==1) out.distance=flex_dd(0.0f);
      return out;
    }
    vertex_count=5;
    initial_faces=6;
    p3_initialized=true;
  } else {
    FlexDD3 directions[3];
    flex_epa_p2_directions_source(local_seed[0].minkowski,local_seed[1].minkowski,directions);
    CommonCCDVertex p2seed[5];
    p2seed[0]=local_seed[0]; p2seed[1]=local_seed[1];
    for (int i=0;i<3;i++) {
      CommonCCDVertex support;
      // The shared P2 helper already performs epaSupport's component
      // divisions. A second normalization perturbs strict support/face ties.
      if (common_ccd_support_pair(a,b,directions[i],mesh_vertices,
                                  mesh_graph,support)) {
        out.status=2; return out;
      }
      common_ccd_diag_epa_support(epa_support_trace,epa_support_count,
                                  directions[i],support);
      p2seed[2+i]=support;
      vertices[vertex_count++]=common_ccd_flex_vertex(support);
      if (vertex_ids!=nullptr) vertex_ids[vertex_count-1]=int2(support.index_a,support.index_b);
    }
    face_vertices[0]=int3(0,2,3); face_vertices[1]=int3(0,4,2);
    face_vertices[2]=int3(0,3,4); face_vertices[3]=int3(1,3,2);
    face_vertices[4]=int3(1,2,4); face_vertices[5]=int3(1,4,3);
    adjacency[0]=int3(1,3,2); adjacency[1]=int3(2,4,0);
    adjacency[2]=int3(0,5,1); adjacency[3]=int3(5,0,4);
    adjacency[4]=int3(3,1,5); adjacency[5]=int3(4,2,3);
    initial_faces=6;
    int fallback_face=-1;
    for (int i=0;i<6;i++) {
      if (common_ccd_face_below_source_limit(
              vertices,center,face_vertices[i],adjacency[i],
              flex_dd(1.0e-30f))) { fallback_face=i; break; }
    }
    if (fallback_face>=0) {
      int3 f=face_vertices[fallback_face];
      CommonCCDVertex p3seed[3]={p2seed[f.x],p2seed[f.y],p2seed[f.z]};
      int init=common_mpr_enclosure_initialize_p3_012(p3seed,a,b,gjk_distance,
          mesh_vertices,mesh_graph,vertices,vertex_ids,center,face_vertices,
          adjacency,epa_support_trace,epa_support_count,max_iterations+2);
      if (init) {
        out.status=init==2 ? 2 : 1;
        if (out.status==1) out.distance=flex_dd(0.0f);
        return out;
      }
      vertex_count=5; initial_faces=6; p3_initialized=true;
    }
    if (!p3_initialized && common_ccd_ray_triangle_source(
                          local_seed[0].minkowski,
                          local_seed[1].minkowski,
                          vertices[2].m,vertices[3].m,vertices[4].m)==0) {
      out.status=1; out.distance=flex_dd(0.0f); return out;
    }
  }
  if (vertex_count>vertex_capacity || initial_faces>face_capacity) {
    out.status=2; return out;
  }
  for (int i=0;i<initial_faces;i++) {
    if (!flex_epa_make_face_dd(vertices,center,
          face_vertices[i].x,face_vertices[i].y,face_vertices[i].z,
          adjacency[i],face) ||
        flex_dd_compare_exact(face.dist2_dd,flex_dd(1.0e-30f))<0) {
      out.status=1; out.distance=flex_dd(0.0f); return out;
    }
    face.index=i;
    faces[face_count]=face;
    face_map[map_count++]=face_count++;
    if (epa_face_trace!=nullptr && i<6) {
      int base=i*COMMON_CCD_DIAG_EPA_INIT_FACE_WORDS;
      epa_face_trace[base+0]=float(i);
      epa_face_trace[base+1]=float(face.a);
      epa_face_trace[base+2]=float(face.b);
      epa_face_trace[base+3]=float(face.c);
      epa_face_trace[base+4]=float(face.adj0);
      epa_face_trace[base+5]=float(face.adj1);
      epa_face_trace[base+6]=float(face.adj2);
      epa_face_trace[base+7]=face.dist2_dd.hi;
      epa_face_trace[base+8]=face.dist2_dd.lo;
      epa_face_trace[base+9]=face.dist2_dd.tail;
      common_ccd_diag_store_dd3(epa_face_trace,base+10,face.v_dd);
      epa_face_trace[base+19]=1.0f;
      epa_face_count++;
    }
  }
  FlexDD upper=flex_dd(3.402823466e+38f);
  FlexDD upper2=upper;
  int best_face=-1;
  int epa_budget=min(max_iterations,max_iterations+2-epa_support_count);
  bool enclosure_converged=false;
  float horizon_trace[133];
  for (int i=0;i<133;i++) horizon_trace[i]=0.0f;
  for (int iteration=0;iteration<epa_budget;iteration++) {
    FlexDD lower2=flex_dd(3.402823466e+38f);
    best_face=-1;
    for (int i=0;i<map_count;i++) {
      int face_id=face_map[i];
      if (face_id<0 || face_id>=face_count) { out.status=2; return out; }
      if (flex_dd_compare_exact(faces[face_id].dist2_dd,lower2)<0) {
        lower2=faces[face_id].dist2_dd;
        best_face=face_id;
      }
    }
    if (best_face<0) { out.status=2; return out; }
    FlexDD lower=flex_dd_sqrt(lower2);
    FlexEpaFace nearest=faces[best_face];
    FlexDD3 direction=flex_dd_compare_exact(lower,flex_dd(1.0e-15f))>0
        ? flex_dd3_div(nearest.v_dd,lower)
        : FlexDD3{flex_dd(1.0f),flex_dd(0.0f),flex_dd(0.0f)};
    CommonCCDVertex support;
    if (epa_support_count>=max_iterations+2 ||
        mpr_support_pair(a,b,direction,mesh_vertices,mesh_graph,support)) {
      out.status=2; return out;
    }
    common_ccd_diag_epa_support(epa_support_trace,epa_support_count,
                                direction,support);
    FlexDD upper_k=flex_dd_div(flex_dd3_dot(nearest.v_dd,support.minkowski),
                               lower);
    if (flex_dd_compare_exact(upper_k,upper)<0) {
      upper=upper_k;
      upper2=flex_dd_mul(upper,upper);
    }
    if (epa_face_trace!=nullptr && iteration<COMMON_CCD_DIAG_EPA_ITERATIONS) {
      int base=COMMON_CCD_DIAG_EPA_ITER_OFFSET
          -COMMON_CCD_DIAG_EPA_FACE_OFFSET
          +iteration*COMMON_CCD_DIAG_EPA_ITER_WORDS;
      epa_face_trace[base+0]=float(iteration);
      epa_face_trace[base+1]=float(best_face);
      epa_face_trace[base+2]=float(nearest.a);
      epa_face_trace[base+3]=float(nearest.b);
      epa_face_trace[base+4]=float(nearest.c);
      epa_face_trace[base+5]=float(nearest.adj0);
      epa_face_trace[base+6]=float(nearest.adj1);
      epa_face_trace[base+7]=float(nearest.adj2);
      common_ccd_diag_store_dd3(epa_face_trace,base+8,nearest.v_dd);
      epa_face_trace[base+17]=lower.hi;
      epa_face_trace[base+18]=lower.lo;
      epa_face_trace[base+19]=lower.tail;
      common_ccd_diag_store_dd3(epa_face_trace,base+20,direction);
      epa_face_trace[base+29]=upper_k.hi;
      epa_face_trace[base+30]=upper_k.lo;
      epa_face_trace[base+31]=upper_k.tail;
      epa_face_trace[base+32]=float(support.index_a);
      epa_face_trace[base+33]=float(support.index_b);
      epa_face_count++;
    }
    if (flex_dd_compare_exact(flex_dd_sub(upper_k,lower),epa_tolerance)<0) {enclosure_converged=true;break;}
    // Pinned mjc_ccd::epa stops after a repeated support feature for discrete
    // pairs (mesh/box/HField with zero margin). This test follows its distance
    // tolerance check and precedes horizon expansion. The rigid production
    // caller supplies the source vertex-ID sidecar; the standalone diagnostic
    // probe may omit it and therefore remains a trace-only path.
    if (false && common_ccd_discrete_pair(a,b) && vertex_ids!=nullptr) {
      // mjc_prism_support does not update mjCCDObj::vertindex (it remains
      // -1 from mjc_initCCDObj). Common support traces retain the selected
      // prism vertex for diagnostics, so normalize only this source-repeat
      // comparison; mesh and box IDs keep their source feature indices.
      const int hfield=1;
      int support_id_a=a.geom_type==hfield ? -1 : support.index_a;
      int support_id_b=b.geom_type==hfield ? -1 : support.index_b;
      bool repeated=false;
      for (int i=0;i<vertex_count;i++) {
        int prior_id_a=a.geom_type==hfield ? -1 : vertex_ids[i].x;
        int prior_id_b=b.geom_type==hfield ? -1 : vertex_ids[i].y;
        if (prior_id_a==support_id_a && prior_id_b==support_id_b) {
          repeated=true;
          break;
        }
      }
      if (repeated) break;
    }
    if (face_count+3>face_capacity ||
        vertex_count>=vertex_capacity || horizon_capacity<3) {
      out.status=2; return out;
    }
    FlexDDVertex support_vertex=common_ccd_flex_vertex(support);
    int new_vertex=vertex_count++;
    vertices[new_vertex]=support_vertex;
    if (vertex_ids!=nullptr) vertex_ids[new_vertex]=int2(support.index_a,support.index_b);
    int horizon_count=0;
    int ok=flex_epa_delete_face(faces,face_map,map_count,best_face,face_count);
    if (!ok) { out.status=2; return out; }
    for (int edge=0;edge<3;edge++) {
      int adjacent=edge==0 ? nearest.adj0 : edge==1 ? nearest.adj1 : nearest.adj2;
      if (adjacent<0 || adjacent>=face_count) { out.status=2; return out; }
      int shared=flex_epa_face_vertex(nearest,(edge+1)%3);
      FlexEpaFace adj=faces[adjacent];
      int adj_edge=flex_epa_get_edge(adj,shared);
      if (adj_edge<0 || adj_edge>=3) { out.status=2; return out; }
      int before=horizon_count;
      int visible=flex_epa_horizon_visit_dd(
          faces,face_map,map_count,support_vertex,horizon,horizon_stack,
          horizon_stack_capacity,face_count,horizon_count,
          adjacent,adj_edge,horizon_trace);
      if (visible<0) { out.status=2; return out; }
      if (visible==0 && horizon_count==before) {
        if (horizon_count>=horizon_capacity) { out.status=2; return out; }
        horizon[horizon_count++]=int2(adjacent,adj_edge);
      }
    }
    if (horizon_count<3 || horizon_count>horizon_capacity ||
        face_count+horizon_count>face_capacity) { out.status=2; return out; }
    int base=face_count;
    for (int h=0;h<horizon_count;h++) {
      int old_id=horizon[h].x, old_edge=horizon[h].y;
      if (old_id<0 || old_id>=face_count || old_edge<0 || old_edge>=3) {
        out.status=2; return out;
      }
      FlexEpaFace old=faces[old_id];
      int v1=flex_epa_face_vertex(old,old_edge);
      int v2=flex_epa_face_vertex(old,(old_edge+1)%3);
      int new_id=face_count;
      int prev=(h==0) ? base+horizon_count-1 : new_id-1;
      int next=(h==horizon_count-1) ? base : new_id+1;
      if (old_edge==0) faces[old_id].adj0=new_id;
      else if (old_edge==1) faces[old_id].adj1=new_id;
      else faces[old_id].adj2=new_id;
      if (!flex_epa_make_face_dd(vertices,center,new_vertex,v2,v1,
                                 int3(prev,old_id,next),face)) {
        out.status=2; return out;
      }
      faces[face_count]=face;
      if (flex_dd_compare_exact(face.dist2_dd,lower2)>=0 &&
          flex_dd_compare_exact(face.dist2_dd,upper2)<=0) {
        face.index=map_count;
        faces[face_count].index=map_count;
        face_map[map_count++]=face_count;
      }
      face_count++;
    }
  }
  // Pinned mjc_ccd's epa() returns its last selected face when the configured
  // iteration budget is exhausted. Non-convergence alone is not an EPA error:
  // after the loop it calls epaWitness whenever `face` is non-null. Capacity,
  // support, and malformed-adjacency failures already returned above. Preserve
  // that source contract and only fail if no source-ordered face was selected.
  if (best_face<0 || best_face>=face_count || (!enclosure_converged && !((a.geom_type==1&&b.geom_type==4)||(a.geom_type==4&&b.geom_type==1)))) {
    out.status=2; return out;
  }
  // Select an actual nearest live triangle: a plane projection can lie
  // outside the triangle and extrapolate its body witnesses.
  FlexDD closest_distance=flex_dd(3.402823466e+38f);
  int closest_face=-1;FlexDD3 closest_point=flex_dd3(float3(0));
  for(int i=0;i<face_count;i++) {
    if(!faces[i].active || faces[i].index<0)continue;
    FlexEpaFace f=faces[i];
    FlexDD3 point=mpr_triangle_closest(vertices[f.a].m,vertices[f.b].m,vertices[f.c].m);
    FlexDD distance=mpr_dot(point,point);
    if(flex_dd_compare_exact(distance,closest_distance)<0) {
      closest_distance=distance;closest_face=i;closest_point=point;
    }
  }
  if(closest_face<0 || flex_dd_compare_exact(closest_distance,flex_dd(0))<=0) {out.status=2;return out;}
  best_face=closest_face;
  FlexEpaFace nearest=faces[best_face];
  nearest.v_dd=closest_point;nearest.dist2_dd=closest_distance;
  flex_epa_write_witness_dd(nearest,vertices,contact,&out.position,&out.normal,
                            &out.witness_a,&out.witness_b);
  out.face_vertices=int3(nearest.a,nearest.b,nearest.c);
  out.distance=flex_dd_sqrt(nearest.dist2_dd);
  out.status=0;
  if(!enclosure_converged) {
    if(!common_ellipsoid_vertex_certificate_012(nearest,vertices,a,b,mesh_vertices,mesh_graph,max_iterations,tolerance,epa_support_count,out,contact))out.status=2;
    return out;
  }
  int recovery=common_fo_configured_recover_012(nearest,vertices,vertex_count,a,b,
      mesh_vertices,mesh_graph,max_iterations,tolerance,epa_support_count,out,contact);
  if(recovery<0){out.status=2;return out;}
  return out;
}
/* Internal syntax wrapper over the EXISTING FlexDD source-53 arithmetic.
 * No tensor, ABI field, arena, storage word or arithmetic precision is added. */
struct SPN012 {
  FlexDD words;
  SPN012():words(flex_dd(0.0f)){}
  SPN012(float x):words(flex_dd(x)){}
  SPN012(int x):words(flex_dd(float(x))){}
  SPN012(FlexDD x):words(x){}
};
inline SPN012 operator+(SPN012 a,SPN012 b){return SPN012(flex_dd_source_add(a.words,b.words));}
inline SPN012 operator-(SPN012 a,SPN012 b){return SPN012(flex_dd_source_sub(a.words,b.words));}
inline SPN012 operator-(SPN012 a){return SPN012(flex_dd_neg(a.words));}
inline SPN012 operator*(SPN012 a,SPN012 b){return SPN012(flex_dd_source_mul(a.words,b.words));}
inline SPN012 operator/(SPN012 a,SPN012 b){return SPN012(flex_dd_source_div(a.words,b.words));}
inline bool spn_nan_012(SPN012 a){return (as_type<uint>(a.words.hi)&0x7fffffffu)>0x7f800000u;}
inline bool operator<(SPN012 a,SPN012 b){
  uint aa=as_type<uint>(a.words.hi),bb=as_type<uint>(b.words.hi),am=aa&0x7fffffffu,bm=bb&0x7fffffffu;
  if(am>0x7f800000u||bm>0x7f800000u)return false;
  if(am==0x7f800000u||bm==0x7f800000u){bool an=(aa>>31)!=0,bn=(bb>>31)!=0;if(am==0x7f800000u&&bm==0x7f800000u)return an&&!bn;return am==0x7f800000u?an:!bn;}
  return flex_dd_compare_exact(a.words,b.words)<0;
}
inline bool operator>(SPN012 a,SPN012 b){return b<a;}
inline bool operator<=(SPN012 a,SPN012 b){return !spn_nan_012(a)&&!spn_nan_012(b)&&!(b<a);}
inline bool operator>=(SPN012 a,SPN012 b){return b<=a;}
inline bool operator==(SPN012 a,SPN012 b){return !spn_nan_012(a)&&!spn_nan_012(b)&&!(a<b)&&!(b<a);}
inline bool operator!=(SPN012 a,SPN012 b){return !(a==b);}
inline SPN012 spn_fabs_012(SPN012 a){return a==SPN012(0)?SPN012(0):SPN012(flex_dd_abs(a.words));}
inline SPN012 spn_fmax_012(SPN012 a,SPN012 b){if(spn_nan_012(a))return b;if(spn_nan_012(b))return a;return a<b?b:a;}
inline SPN012 spn_fmin_012(SPN012 a,SPN012 b){if(spn_nan_012(a))return b;if(spn_nan_012(b))return a;return b<a?b:a;}
inline SPN012 spn_fma_012(SPN012 a,SPN012 b,SPN012 c){return SPN012(flex_dd_source_fma(a.words,b.words,c.words));}
inline SPN012 spn_sqrt_012(SPN012 a){return SPN012(flex_dd_source_sqrt(a.words));}
inline SPN012 spn_scalbn_012(SPN012 a,int exponent){return SPN012(flex_dd_scalbn(a.words,exponent));}
inline bool spn_isfinite_012(SPN012 a){return (as_type<uint>(a.words.hi)&0x7fffffffu)<0x7f800000u&&(as_type<uint>(a.words.lo)&0x7fffffffu)<0x7f800000u&&(as_type<uint>(a.words.tail)&0x7fffffffu)<0x7f800000u;}
inline void spn_frexp_012(SPN012 a,thread int* exponent){
  // frexp exponent of the represented positive value, including a high limb
  // that rounded up to a power of two. No lower precision exponent inference.
  float limb=a.words.hi!=0.0f?a.words.hi:a.words.lo!=0.0f?a.words.lo:a.words.tail;
  int e=flex_float_exponent(limb);
  SPN012 power_threshold=spn_scalbn_012(SPN012(1),e-1);
  if(a<power_threshold)e--;
  else if(e<128&&a>=spn_scalbn_012(SPN012(1),e))e++;
  *exponent=e;
}

/* Exact Boolean sign over the CURRENT stored binary32 limbs.
 * One explicit 10*uint=40 byte transient two's-complement accumulator in units
 * 2^-149. Finite limb max bit276; <=9 limbs sum below2^281, signed320 safe.
 * No floating arithmetic, higher-precision physical value, new representation,
 * tensor/ABI/cache storage, support query, EPA iteration or solver work.
 */
static inline bool common_axis_limb_finite_012(float x) {
  return ((as_type<uint>(x)>>23)&255u)!=255u;
}
static inline bool common_axis_words_finite_012(FlexDD x) {
  return common_axis_limb_finite_012(x.hi)&&common_axis_limb_finite_012(x.lo)&&common_axis_limb_finite_012(x.tail);
}
static inline bool common_axis_add_limb_012(thread uint* words,float x,int coefficient) {
  uint bits=as_type<uint>(x),exponent=(bits>>23)&255u;
  if(exponent==255u)return false;
  uint mantissa=bits&0x7fffffu;if(exponent)mantissa|=0x800000u;
  uint shift=exponent?exponent-1u:0u,slot=shift>>5,offset=shift&31u;
  uint low=mantissa<<offset,high=offset?mantissa>>(32u-offset):0u;
  bool negative=((bits>>31)!=0u)!=(coefficient<0);uint carry=0;
  for(uint i=0;i<10u;i++) {
    uint piece=i==slot?low:(i==slot+1u?high:0u),old=words[i];
    if(negative) {
      uint first=old-piece,borrow1=old<piece;
      uint second=first-carry,borrow2=first<carry;
      words[i]=second;carry=borrow1|borrow2;
    } else {
      uint first=old+piece,carry1=first<old;
      uint second=first+carry,carry2=second<first;
      words[i]=second;carry=carry1|carry2;
    }
  }
  return true;
}
static inline int common_axis_terms_sign_012(thread const FlexDD* terms,
    thread const int* coefficients,int count) {
  if(count<1||count>3)return 2;
  uint words[10];for(int i=0;i<10;i++)words[i]=0u;
  for(int i=0;i<count;i++) {
    if(coefficients[i]!=1&&coefficients[i]!=-1)return 2;
    if(!common_axis_add_limb_012(words,terms[i].hi,coefficients[i])||
       !common_axis_add_limb_012(words,terms[i].lo,coefficients[i])||
       !common_axis_add_limb_012(words,terms[i].tail,coefficients[i]))return 2;
  }
  uint any=0u;for(int i=0;i<10;i++)any|=words[i];
  return any==0u?0:((words[9]&0x80000000u)?-1:1);
}
static inline int common_axis_compare_words_012(FlexDD a,FlexDD b) {
  FlexDD terms[2]={a,b};int coefficients[2]={1,-1};
  return common_axis_terms_sign_012(terms,coefficients,2);
}
static inline FlexDD common_axis_signed_words_012(FlexDD x,int sign) {
  uint mask=sign<0?0x80000000u:0u;
  return FlexDD{as_type<float>(as_type<uint>(x.hi)^mask),
                as_type<float>(as_type<uint>(x.lo)^mask),
                as_type<float>(as_type<uint>(x.tail)^mask)};
}
static inline bool common_box_prism_axis_zero_012(
    thread const CommonCCDSupportObject& first,
    thread const CommonCCDSupportObject& second,
    device const float* vertices,thread FlexDD* certificate=nullptr) {
  bool reverse=first.kind==COMMON_CCD_BOX&&second.kind==COMMON_CCD_HFIELD_PRISM;
  if(!(first.kind==COMMON_CCD_HFIELD_PRISM&&second.kind==COMMON_CCD_BOX)&&!reverse)return false;
  thread const CommonCCDSupportObject* hf=reverse?&second:&first;
  thread const CommonCCDSupportObject* box=reverse?&first:&second;
  if(hf->vertex_count!=6||hf->vertex_offset<0||
     common_axis_compare_words_012(hf->margin,flex_dd(0.0f))!=0||
     common_axis_compare_words_012(box->margin,flex_dd(0.0f))!=0)return false;
  for(int i=0;i<3;i++)if(!common_axis_words_finite_012(box->pos[i])||
      common_axis_compare_words_012(box->size[i],flex_dd(0.0f))!=1)return false;
  for(int i=0;i<9;i++)if(!common_axis_words_finite_012(box->mat[i]))return false;
  for(int v=0;v<6;v++)for(int k=0;k<3;k++) {
    FlexDD r=flex_dd3_component(hf->hfield_residual[v],k);
    if((as_type<uint>(r.tail)&0x7fffffffu)!=0u||
       !common_axis_words_finite_012(FlexDD{vertices[3*(hf->vertex_offset+v)+k],r.hi,r.lo}))return false;
  }
  for(int axis=0;axis<3;axis++) {
    int local=-1;bool axis_row=true;
    for(int k=0;k<3;k++) {
      FlexDD x=box->mat[3*axis+k];
      if(common_axis_compare_words_012(x,flex_dd(1.0f))==0||
         common_axis_compare_words_012(x,flex_dd(-1.0f))==0) {
        if(local>=0)axis_row=false;local=k;
      } else if(common_axis_compare_words_012(x,flex_dd(0.0f))!=0)axis_row=false;
    }
    if(!axis_row||local<0)continue;
    for(int sign=-1;sign<=1;sign+=2) {
      FlexDD upper=flex_dd(0.0f);
      for(int v=0;v<6;v++) {
        FlexDD r=flex_dd3_component(hf->hfield_residual[v],axis);
        FlexDD p=FlexDD{vertices[3*(hf->vertex_offset+v)+axis],r.hi,r.lo};
        if(v==0||sign*common_axis_compare_words_012(p,upper)>0)upper=p;
      }
      FlexDD terms[3]={box->pos[axis],upper,box->size[local]};
      int coefficients[3]={sign,-sign,-1};
      int plane_sign=common_axis_terms_sign_012(terms,coefficients,3);
      if(plane_sign==0||plane_sign==1) {
        if(certificate!=nullptr) {
          certificate[0]=flex_dd(float(axis));certificate[1]=flex_dd(float(sign));
          certificate[2]=common_axis_signed_words_012(upper,sign);
          certificate[3]=common_axis_signed_words_012(box->pos[axis],sign);
          certificate[4]=box->size[local];certificate[5]=flex_dd(float(plane_sign));
          certificate[6]=flex_dd(6.0f);certificate[7]=flex_dd(float(local));
        }
        return true;
      }
    }
  }
  return false;
}

/* Exact bounded Boolean on current F32 words, not physical arithmetic.
 * Normalize nonzero integer mantissas; maxTop-minLSB <=58 implies every
 * magnitude <2^59 and at most9 terms have absolute sum <2^63. Therefore
 * every signed64 partial sum is safe. Wider/nonfinite inputs DECLINE.
 * No multiword accumulator, optional diagnostic output or float arithmetic.
 */
static inline int common_axis_short_sign_012(thread const FlexDD* terms,
    thread const int* coefficients,int count) {
  if(count<1||count>3)return 2;
  int minimum=2048,maximum=-2048;bool any=false;
  for(int i=0;i<count;i++) {
    if(coefficients[i]!=1&&coefficients[i]!=-1)return 2;
    for(int j=0;j<3;j++) {
      float x=j==0?terms[i].hi:(j==1?terms[i].lo:terms[i].tail);
      uint bits=as_type<uint>(x),e=(bits>>23)&255u,m=bits&0x7fffffu;
      if(e==255u)return 2;if(e)m|=0x800000u;if(m==0u)continue;
      int exponent=e?int(e)-150:-149;
      int trailing=31-int(clz(m&(~m+1u)));m>>=trailing;exponent+=trailing;
      minimum=min(minimum,exponent);maximum=max(maximum,exponent+31-int(clz(m)));any=true;
    }
  }
  if(!any)return 0;if(maximum-minimum>58)return 2;
  long total=0;
  for(int i=0;i<count;i++)for(int j=0;j<3;j++) {
    float x=j==0?terms[i].hi:(j==1?terms[i].lo:terms[i].tail);
    uint bits=as_type<uint>(x),e=(bits>>23)&255u,m=bits&0x7fffffu;
    if(e)m|=0x800000u;if(m==0u)continue;
    int exponent=e?int(e)-150:-149;
    int trailing=31-int(clz(m&(~m+1u)));m>>=trailing;exponent+=trailing;
    long magnitude=long(m)<<uint(exponent-minimum);
    if(((bits>>31)!=0u)!=(coefficients[i]<0))total-=magnitude;else total+=magnitude;
  }
  return total==0?0:(total<0?-1:1);
}
static inline int common_axis_short_compare_012(FlexDD a,FlexDD b) {
  FlexDD terms[2]={a,b};int coefficients[2]={1,-1};
  return common_axis_short_sign_012(terms,coefficients,2);
}

/* Exact ellipsoid nonpenetration using extrema streamed from the same six
 * source-rounded prism points at construction, before writing device scratch.
 * No device-vertex readback, optional diagnostic output or physics arithmetic.
 * Bounds occupy six existing-precision FlexDD locals (72 logical bytes plus
 * one validity Boolean); no persistent/ABI/cache/support/EPA/solver growth.
 */
static inline bool common_ellipsoid_axis_short_interval_nonpenetration_012(
    thread const CommonCCDSupportObject& ellipsoid,
    thread const FlexDD* lower,thread const FlexDD* upper,bool valid) {
  if(!valid || ellipsoid.kind!=COMMON_CCD_ELLIPSOID ||
     common_axis_short_compare_012(ellipsoid.margin,flex_dd(0.0f))!=0)return false;
  for(int i=0;i<3;i++)if(!common_axis_words_finite_012(ellipsoid.pos[i])||
      common_axis_short_compare_012(ellipsoid.size[i],flex_dd(0.0f))!=1||
      !common_axis_words_finite_012(lower[i])||!common_axis_words_finite_012(upper[i])||
      common_axis_short_compare_012(lower[i],upper[i])>0)return false;
  for(int i=0;i<9;i++)if(!common_axis_words_finite_012(ellipsoid.mat[i]))return false;
  for(int axis=0;axis<3;axis++) {
    int local=-1;bool axis_row=true;
    for(int k=0;k<3;k++) {
      FlexDD x=ellipsoid.mat[3*axis+k];
      if(common_axis_short_compare_012(x,flex_dd(1.0f))==0||
         common_axis_short_compare_012(x,flex_dd(-1.0f))==0) {
        if(local>=0)axis_row=false;local=k;
      } else if(common_axis_short_compare_012(x,flex_dd(0.0f))!=0)axis_row=false;
    }
    if(!axis_row||local<0)continue;
    for(int sign=-1;sign<=1;sign+=2) {
      FlexDD terms[3]={ellipsoid.pos[axis],sign<0?lower[axis]:upper[axis],ellipsoid.size[local]};
      int coefficients[3]={sign,-sign,-1};
      int plane_sign=common_axis_short_sign_012(terms,coefficients,3);
      if(plane_sign==0||plane_sign==1)return true;
    }
  }
  return false;
}

/* Exact cylinder coordinate projection: radial extent r on localX/Y, h on localZ.
 * Requires actual kindCYLINDER, positive r/h, exactzero unusedsize and both margins.
 * Same actual HField-local six-vertex streamed bounds and exact bounded64 Boolean.
 * No shape object mutation or support/physics arithmetic, no new persistent storage.
 */
static inline bool common_cylinder_axis_short_interval_nonpenetration_012(
    thread const CommonCCDSupportObject& cylinder,
    thread const FlexDD* lower,thread const FlexDD* upper,bool valid,FlexDD prism_margin) {
  if(!valid || cylinder.kind!=COMMON_CCD_CYLINDER ||
     common_axis_short_compare_012(prism_margin,flex_dd(0.0f))!=0 ||
     !common_axis_words_finite_012(cylinder.size[2]) ||
     common_axis_short_compare_012(cylinder.size[2],flex_dd(0.0f))!=0 ||
     common_axis_short_compare_012(cylinder.margin,flex_dd(0.0f))!=0)return false;
  for(int i=0;i<3;i++)if(!common_axis_words_finite_012(cylinder.pos[i])||
      (i<2 && common_axis_short_compare_012(cylinder.size[i],flex_dd(0.0f))!=1)||
      !common_axis_words_finite_012(lower[i])||!common_axis_words_finite_012(upper[i])||
      common_axis_short_compare_012(lower[i],upper[i])>0)return false;
  for(int i=0;i<9;i++)if(!common_axis_words_finite_012(cylinder.mat[i]))return false;
  for(int axis=0;axis<3;axis++) {
    int local=-1;bool axis_row=true;
    for(int k=0;k<3;k++) {
      FlexDD x=cylinder.mat[3*axis+k];
      if(common_axis_short_compare_012(x,flex_dd(1.0f))==0||
         common_axis_short_compare_012(x,flex_dd(-1.0f))==0) {
        if(local>=0)axis_row=false;local=k;
      } else if(common_axis_short_compare_012(x,flex_dd(0.0f))!=0)axis_row=false;
    }
    if(!axis_row||local<0)continue;
    for(int sign=-1;sign<=1;sign+=2) {
      FlexDD terms[3]={cylinder.pos[axis],sign<0?lower[axis]:upper[axis],cylinder.size[local==2?1:0]};
      int coefficients[3]={sign,-sign,-1};
      int plane_sign=common_axis_short_sign_012(terms,coefficients,3);
      if(plane_sign==0||plane_sign==1)return true;
    }
  }
  return false;
}

/* Actual validated F32 hull vertices, never a bounding-box substitute.
 * Exact signed-coordinate row projects conv(vertices) to its actual minimum.
 * IEEE integer ordering retains every finite subnormal and signed zero; no
 * rounded support point or floating sign/min arithmetic enters this Boolean.
 * Caller has already admitted the immutable original backing, finite metadata,
 * capacity and six support queries. Wide/unsupported exact sums decline.
 */
static inline bool common_mesh_axis_short_interval_nonpenetration_012(
    thread const CommonCCDSupportObject& mesh,device const float* vertices,
    thread const FlexDD* lower,thread const FlexDD* upper,bool valid,FlexDD prism_margin) {
  if(!valid || mesh.kind!=COMMON_CCD_HULL || mesh.geom_type!=7 ||
     mesh.vertex_count<4 || mesh.vertex_offset<0 || mesh.vertex_count>715827882 ||
     mesh.vertex_offset>715827882-mesh.vertex_count ||
     common_axis_short_compare_012(mesh.margin,flex_dd(0.0f))!=0 ||
     common_axis_short_compare_012(prism_margin,flex_dd(0.0f))!=0)return false;
  for(int i=0;i<3;i++)if(!common_axis_words_finite_012(mesh.pos[i]) ||
      !common_axis_words_finite_012(lower[i]) || !common_axis_words_finite_012(upper[i]) ||
      common_axis_short_compare_012(lower[i],upper[i])>0)return false;
  for(int i=0;i<9;i++)if(!common_axis_words_finite_012(mesh.mat[i]))return false;
  for(int v=0;v<mesh.vertex_count;v++)for(int k=0;k<3;k++)
    if((as_type<uint>(vertices[3*(mesh.vertex_offset+v)+k])&0x7f800000u)==0x7f800000u)return false;
  for(int axis=0;axis<3;axis++) {
    int local=-1,row_sign=0;bool axis_row=true;
    for(int k=0;k<3;k++) {
      FlexDD x=mesh.mat[3*axis+k];
      int positive=common_axis_short_compare_012(x,flex_dd(1.0f));
      int negative=common_axis_short_compare_012(x,flex_dd(-1.0f));
      if(positive==0||negative==0) {
        if(local>=0)axis_row=false;local=k;row_sign=positive==0?1:-1;
      } else if(common_axis_short_compare_012(x,flex_dd(0.0f))!=0)axis_row=false;
    }
    if(!axis_row||local<0)continue;
    for(int sign=-1;sign<=1;sign+=2) {
      uint minimum_bits=0u,minimum_key=0xffffffffu;
      for(int v=0;v<mesh.vertex_count;v++) {
        uint bits=as_type<uint>(vertices[3*(mesh.vertex_offset+v)+local]);
        if(sign!=row_sign)bits^=0x80000000u;
        uint key=(bits&0x80000000u)?~bits:(bits^0x80000000u);
        if(v==0||key<minimum_key){minimum_bits=bits;minimum_key=key;}
      }
      FlexDD extent=flex_dd(as_type<float>(minimum_bits^0x80000000u));
      if(common_axis_short_compare_012(extent,flex_dd(0.0f))!=1)continue;
      FlexDD terms[3]={mesh.pos[axis],sign<0?lower[axis]:upper[axis],extent};
      int coefficients[3]={sign,-sign,-1};
      int plane_sign=common_axis_short_sign_012(terms,coefficients,3);
      if(plane_sign==0||plane_sign==1)return true;
    }
  }
  return false;
}

/* Internal source53 syntax only; no new precision, arena or ABI. */
static inline bool common_ellipsoid_vertex_certificate_012(
    const thread FlexEpaFace& face,device const FlexDDVertex* vertices,
    thread CommonCCDSupportObject& a,thread CommonCCDSupportObject& b,
    device const float* mesh_vertices,device const int* mesh_graph,
    int max_iterations,FlexDD tolerance_words,thread int& support_count,
    thread CommonCCDEpaResult& out,thread ContactGeom& contact) {
  bool reverse=a.geom_type==4&&b.geom_type==1;
  if(!(a.geom_type==1&&b.geom_type==4)&&!reverse)return false;
  if(max_iterations<=0||max_iterations>2147483645||max_iterations+2-support_count<2)return false;
  thread const CommonCCDSupportObject* ellipsoid=reverse?&a:&b;
  SPN012 margin(ellipsoid->margin),tolerance(tolerance_words);
  if(!spn_isfinite_012(margin)||margin<0)return false;
  int ids[3]={face.a,face.b,face.c};FlexDD3 anchor_words=reverse?vertices[ids[0]].b:vertices[ids[0]].a;
  for(int j=1;j<3;j++) {
    FlexDD3 candidate=reverse?vertices[ids[j]].b:vertices[ids[j]].a;
    for(int k=0;k<3;k++)if(SPN012(flex_dd3_component(candidate,k))!=SPN012(flex_dd3_component(anchor_words,k)))return false;
  }
  SPN012 anchor[3],p[3]={0},y[3]={0},a2[3],old[3],amin=SPN012(as_type<float>(0x7f800000u)),amax=0;
  for(int k=0;k<3;k++){anchor[k]=SPN012(flex_dd3_component(anchor_words,k));old[k]=SPN012(flex_dd3_component(reverse?out.witness_a:out.witness_b,k))-SPN012(ellipsoid->pos[k]);}
  for(int k=0;k<3;k++) {
    SPN012 axis(ellipsoid->size[k]);if(!spn_isfinite_012(axis)||axis<=0)return false;
    amin=spn_fmin_012(amin,axis);amax=spn_fmax_012(amax,axis);a2[k]=axis*axis;
    for(int j=0;j<3;j++){p[k]=p[k]+SPN012(ellipsoid->mat[3*j+k])*(anchor[j]-SPN012(ellipsoid->pos[j]));y[k]=y[k]+SPN012(ellipsoid->mat[3*j+k])*old[j];}
  }
  int axis=0;for(int k=1;k<3;k++)if(spn_fabs_012(y[k]/SPN012(ellipsoid->size[k]))>spn_fabs_012(y[axis]/SPN012(ellipsoid->size[axis])))axis=k;
  if(y[axis]==0)return false;
  SPN012 tau=a2[axis]*(p[axis]/y[axis]-1);
  for(int iteration=0;iteration<2;iteration++) {
    if(!spn_isfinite_012(tau)||tau<=-amin*amin||tau>=0)return false;
    SPN012 f=-1,derivative=0;
    for(int k=0;k<3;k++) {
      SPN012 denominator=a2[k]+tau,term=a2[k]*p[k]*p[k]/(denominator*denominator);
      if(!spn_isfinite_012(term)||denominator<=0)return false;
      f=f+term;derivative=derivative-2*term/denominator;
    }
    if(!spn_isfinite_012(derivative)||derivative>=0)return false;tau=tau-f/derivative;
  }
  if(!spn_isfinite_012(tau)||tau<=-amin*amin||tau>=0)return false;
  SPN012 n[3]={0};for(int k=0;k<3;k++)for(int j=0;j<3;j++)n[j]=n[j]-SPN012(ellipsoid->mat[3*j+k])*p[k]/(a2[k]+tau);
  FlexDD3 nw={n[0].words,n[1].words,n[2].words};SPN012 length(flex_dd_source_sqrt(flex_dd_source_dot3(nw,nw)));if(!spn_isfinite_012(length)||length<=0)return false;
  SPN012 factor=SPN012(reverse?-1:1)/length;for(int k=0;k<3;k++)n[k]=n[k]*factor;nw=FlexDD3{n[0].words,n[1].words,n[2].words};
  CommonCCDVertex first;support_count++;if(mpr_support_pair(a,b,nw,mesh_vertices,mesh_graph,first))return false;
  for(int k=0;k<3;k++)if(SPN012(flex_dd3_component(reverse?first.point_b:first.point_a,k))!=anchor[k])return false;
  SPN012 depth(flex_dd_source_sqrt(flex_dd_source_dot3(first.minkowski,first.minkowski)));if(!spn_isfinite_012(depth)||depth<=0)return false;
  factor=SPN012(1)/depth;for(int k=0;k<3;k++)n[k]=SPN012(flex_dd3_component(first.minkowski,k))*factor;nw=FlexDD3{n[0].words,n[1].words,n[2].words};
  CommonCCDVertex second;support_count++;if(mpr_support_pair(a,b,nw,mesh_vertices,mesh_graph,second))return false;
  for(int k=0;k<3;k++)if(SPN012(flex_dd3_component(reverse?second.point_b:second.point_a,k))!=anchor[k])return false;
  SPN012 radius=amin*(amin/amax)+margin*SPN012(.5f),center[3];
  for(int k=0;k<3;k++)center[k]=SPN012(flex_dd3_component(second.minkowski,k))-radius*n[k];
  FlexDD3 cw={center[0].words,center[1].words,center[2].words};SPN012 upper(flex_dd_source_dot3(nw,second.minkowski)),lower=radius-SPN012(flex_dd_source_sqrt(flex_dd_source_dot3(cw,cw))),gap=upper-depth,scale=1;
  for(int k=0;k<3;k++){scale=spn_fmax_012(scale,spn_fabs_012(SPN012(flex_dd3_component(first.point_a,k))));scale=spn_fmax_012(scale,spn_fabs_012(SPN012(flex_dd3_component(first.point_b,k))));scale=spn_fmax_012(scale,spn_fabs_012(SPN012(flex_dd3_component(second.minkowski,k))));}
  scale=spn_fmax_012(scale,spn_fmax_012(radius,spn_fabs_012(upper)));SPN012 error=SPN012(0x1p-42f)*scale;
  if(!spn_isfinite_012(radius)||radius<=0||!spn_isfinite_012(lower)||lower<=error||!spn_isfinite_012(upper)||upper-lower+2*error>tolerance||!spn_isfinite_012(gap)||gap < -error||gap+error>tolerance||depth+error<lower)return false;
  out.distance=depth.words;out.witness_a=first.point_a;out.witness_b=first.point_b;out.normal=nw;out.position=mpr_scale(mpr_add(first.point_a,first.point_b),flex_dd(.5f));
  contact.normal=flex_dd3_high(nw);contact.pos=flex_dd3_high(out.position);contact.dist=-((out.distance.hi+out.distance.lo)+out.distance.tail);contact.t1=float3(0);make_frame(contact.normal,contact.t1,contact.t2);
  return true;
}

/* Exact prism faces and analytic sphere; no proxy geometry or support loop. */
static SPN012 spn_dot_012(thread const SPN012* a,thread const SPN012* b) {
  return spn_fma_012(a[2],b[2],spn_fma_012(a[0],b[0],a[1]*b[1]));
}
static void spn_sub_012(thread SPN012* out,thread const SPN012* a,thread const SPN012* b) {
  for(int k=0;k<3;k++)out[k]=a[k]-b[k];
}
static void spn_cross_012(thread SPN012* out,thread const SPN012* a,thread const SPN012* b) {
  SPN012 p=a[1]*b[2],q=a[2]*b[1];out[0]=p-q;
  p=a[2]*b[0];q=a[0]*b[2];out[1]=p-q;
  p=a[0]*b[1];q=a[1]*b[0];out[2]=p-q;
}
static int spn_unit_012(thread const SPN012* v,thread SPN012* unit,thread SPN012* length) {
  SPN012 largest=spn_fmax_012(spn_fabs_012(v[0]),spn_fmax_012(spn_fabs_012(v[1]),spn_fabs_012(v[2])));
  if(!spn_isfinite_012(largest)||largest<=0)return 0;
  int exponent;spn_frexp_012(largest,&exponent);SPN012 scaled[3];
  for(int k=0;k<3;k++)scaled[k]=spn_scalbn_012(v[k],-exponent);
  SPN012 norm=spn_sqrt_012(spn_dot_012(scaled,scaled));
  if(!spn_isfinite_012(norm)||norm<=0)return 0;
  for(int k=0;k<3;k++)unit[k]=scaled[k]/norm;
  *length=spn_scalbn_012(norm,exponent);
  return spn_isfinite_012(*length)&&*length>0;
}
static int spn_face_012(thread const SPN012* a,thread const SPN012* b,thread const SPN012* c,
                      thread SPN012* normal,thread SPN012* length) {
  SPN012 e[3],f[3];spn_sub_012(e,b,a);spn_sub_012(f,c,a);
  SPN012 largest=0;for(int k=0;k<3;k++)largest=spn_fmax_012(largest,spn_fmax_012(spn_fabs_012(e[k]),spn_fabs_012(f[k])));
  if(!spn_isfinite_012(largest)||largest<=0)return 0;
  int exponent;spn_frexp_012(largest,&exponent);
  for(int k=0;k<3;k++){e[k]=spn_scalbn_012(e[k],-exponent);f[k]=spn_scalbn_012(f[k],-exponent);}
  SPN012 cross[3];spn_cross_012(cross,e,f);
  return spn_unit_012(cross,normal,length);
}
static void spn_segment_012(thread SPN012* p,thread const SPN012* center,
                          thread const SPN012* a,thread const SPN012* b) {
  SPN012 edge[3],unit[3],length,delta[3];spn_sub_012(edge,b,a);spn_sub_012(delta,center,a);
  SPN012 t=0;if(spn_unit_012(edge,unit,&length))t=spn_dot_012(delta,unit)/length;
  t=spn_fmax_012(0,spn_fmin_012(1,t));for(int k=0;k<3;k++)p[k]=spn_fma_012(t,edge[k],a[k]);
}
static SPN012 spn_distance_012(thread const SPN012* a,thread const SPN012* b) {
  SPN012 v[3],unit[3],length=0;spn_sub_012(v,a,b);
  if(v[0]==0&&v[1]==0&&v[2]==0)return 0;
  return spn_unit_012(v,unit,&length)?length:SPN012(as_type<float>(0x7f800000u));
}
static int spn_triangle_012(thread SPN012* point,thread const SPN012* center,
        thread const SPN012* a,thread const SPN012* b,thread const SPN012* c,thread const SPN012* normal) {
  SPN012 e[3],f[3],delta[3],projected[3],cross[3];
  spn_sub_012(e,b,a);spn_sub_012(f,c,a);spn_sub_012(delta,center,a);
  SPN012 h=spn_dot_012(normal,delta);
  for(int k=0;k<3;k++)projected[k]=spn_fma_012(-h,normal[k],center[k]);
  SPN012 largest=0;for(int k=0;k<3;k++)largest=spn_fmax_012(largest,spn_fmax_012(spn_fabs_012(e[k]),spn_fabs_012(f[k])));
  int exponent;spn_frexp_012(largest,&exponent);
  for(int k=0;k<3;k++){e[k]=spn_scalbn_012(e[k],-exponent);f[k]=spn_scalbn_012(f[k],-exponent);delta[k]=spn_scalbn_012(projected[k]-a[k],-exponent);}
  spn_cross_012(cross,e,f);int axis=0;
  if(spn_fabs_012(cross[1])>spn_fabs_012(cross[axis]))axis=1;
  if(spn_fabs_012(cross[2])>spn_fabs_012(cross[axis]))axis=2;
  SPN012 denominator=cross[axis],x[3],y[3];
  spn_cross_012(x,delta,f);spn_cross_012(y,e,delta);
  SPN012 u=x[axis]/denominator,v=y[axis]/denominator,w=(1-u)-v;
  if(spn_isfinite_012(u)&&spn_isfinite_012(v)&&spn_isfinite_012(w)&&u>=0&&v>=0&&w>=0) {
    for(int k=0;k<3;k++)point[k]=projected[k];return 1;
  }
  spn_segment_012(point,center,a,b);SPN012 best=spn_distance_012(point,center),candidate[3];
  spn_segment_012(candidate,center,a,c);SPN012 d=spn_distance_012(candidate,center);
  if(d<best){best=d;for(int k=0;k<3;k++)point[k]=candidate[k];}
  spn_segment_012(candidate,center,b,c);d=spn_distance_012(candidate,center);
  if(d<best)for(int k=0;k<3;k++)point[k]=candidate[k];
  return 0;
}

static __attribute__((noinline)) int analytic_sphere_prism_012(
    thread CommonCCDSupportObject& a,thread CommonCCDSupportObject& b,
    int iterations,FlexDD tolerance_words,device const float* mesh_vertices,
    thread FlexDD& depth,thread FlexDD3& position,thread FlexDD3& normal) {
  bool reverse=a.geom_type==2&&b.geom_type==1;
  if(!(a.geom_type==1&&b.geom_type==2)&&!reverse)return -20;
  thread const CommonCCDSupportObject* prism=reverse?&b:&a;
  thread const CommonCCDSupportObject* sphere=reverse?&a:&b;
  if(prism->kind!=COMMON_CCD_HFIELD_PRISM||prism->vertex_count!=6||prism->vertex_offset<0)return -20;
  SPN012 vertices[6][3];
  for(int i=0;i<6;i++)for(int k=0;k<3;k++) {
    FlexDD residual=k==0?prism->hfield_residual[i].x:k==1?prism->hfield_residual[i].y:prism->hfield_residual[i].z;
    vertices[i][k]=SPN012(FlexDD{mesh_vertices[3*(prism->vertex_offset+i)+k],residual.hi,residual.lo});
  }
  SPN012 center[3]={SPN012(sphere->pos[0]),SPN012(sphere->pos[1]),SPN012(sphere->pos[2])};
  SPN012 radius=SPN012(sphere->size[0])+SPN012(sphere->margin)*SPN012(.5f);
  SPN012 tolerance(tolerance_words);
  const int triangles[8][3]={{0,2,1},{3,4,5},{0,1,4},{0,4,3},{1,2,5},{1,5,4},{2,0,3},{2,3,5}};
  SPN012 normals[8][3],heights[8],closest[3]={0},minimum=SPN012(as_type<float>(0x7f800000u)),best=SPN012(as_type<float>(0x7f800000u));
  int valid[8]={0},projection[8]={0},inside=1,result=-1,selected=-1;
  SPN012 result_depth=0,result_normal[3]={0},result_position[3]={0};
  if(iterations<=0||!spn_isfinite_012(radius)||radius<=0)return -23;
  for(int i=0;i<8;i++) {
    thread const SPN012 *x=vertices[triangles[i][0]],*y=vertices[triangles[i][1]],*z=vertices[triangles[i][2]];
    SPN012 normlength,delta[3];if(!spn_face_012(x,y,z,normals[i],&normlength))continue;
    SPN012 orientation=0;for(int j=0;j<6;j++){spn_sub_012(delta,vertices[j],x);SPN012 score=spn_dot_012(normals[i],delta);if(spn_fabs_012(score)>spn_fabs_012(orientation))orientation=score;}
    if(orientation>0)for(int k=0;k<3;k++)normals[i][k]=-normals[i][k];
    spn_sub_012(delta,center,x);heights[i]=-spn_dot_012(normals[i],delta);
    if(!spn_isfinite_012(heights[i]))return -23;
    if(heights[i]<0)inside=0;if(heights[i]<minimum)minimum=heights[i];
    SPN012 p[3];projection[i]=spn_triangle_012(p,center,x,y,z,normals[i]);valid[i]=1;
    SPN012 distance=spn_distance_012(p,center);if(distance<best){best=distance;for(int k=0;k<3;k++)closest[k]=p[k];}
  }
  SPN012 n[3],wa[3],wb[3],d;
  if(inside) {
    for(int i=0;i<8;i++)if(valid[i]&&projection[i]&&heights[i]<=minimum+tolerance){selected=i;break;}
    if(selected<0)return -23;
    for(int k=0;k<3;k++){n[k]=normals[selected][k];wa[k]=spn_fma_012(heights[selected],n[k],center[k]);}
    d=radius+heights[selected];
  } else {
    SPN012 delta[3],length;spn_sub_012(delta,center,closest);
    if(!spn_isfinite_012(best)||!spn_unit_012(delta,n,&length))return -23;
    d=radius-length;for(int k=0;k<3;k++)wa[k]=closest[k];
  }
  if(!spn_isfinite_012(d)||d<=0)return -23;
  for(int k=0;k<3;k++){wb[k]=spn_fma_012(-radius,n[k],center[k]);result_normal[k]=reverse?-n[k]:n[k];result_position[k]=(wa[k]+wb[k])*SPN012(.5f);}
  result_depth=d;
  depth=result_depth.words;
  position=FlexDD3{result_position[0].words,result_position[1].words,result_position[2].words};
  normal=FlexDD3{result_normal[0].words,result_normal[1].words,result_normal[2].words};
  return 1;
}

/* Exact capsule core-segment Minkowski sum; Source prototype only.
 * The 34 triangles cover the boundary of conv(P +/- axis). No support loop.
 * All work/storage counts are explicit; no native inputs are produced here. */
static void cp_triangle_ids_012(int i,thread int* ids) {
  const int faces[8][3]={{0,2,1},{3,4,5},{0,1,4},{0,4,3},{1,2,5},{1,5,4},{2,0,3},{2,3,5}};
  const int edges[9][2]={{0,1},{1,2},{2,0},{3,4},{4,5},{5,3},{0,3},{1,4},{2,5}};
  if(i<16){for(int k=0;k<3;k++)ids[k]=faces[i%8][k]+6*(i/8);}
  else {int e=(i-16)/2,a=edges[e][0],b=edges[e][1];ids[0]=a;ids[1]=(i%2)?b+6:b;ids[2]=(i%2)?a+6:b+6;}
}
static int cp_weights_012(thread SPN012* weights,thread const SPN012* point,
                         thread const SPN012* a,thread const SPN012* b,thread const SPN012* c) {
  SPN012 e[3],f[3],v[3],cross[3],x[3],y[3];
  spn_sub_012(e,b,a);spn_sub_012(f,c,a);spn_sub_012(v,point,a);
  SPN012 largest=0;for(int k=0;k<3;k++)largest=spn_fmax_012(largest,spn_fmax_012(spn_fabs_012(e[k]),spn_fabs_012(f[k])));
  int exponent;spn_frexp_012(largest,&exponent);
  for(int k=0;k<3;k++){e[k]=spn_scalbn_012(e[k],-exponent);f[k]=spn_scalbn_012(f[k],-exponent);v[k]=spn_scalbn_012(v[k],-exponent);}
  spn_cross_012(cross,e,f);int axis=0;
  if(spn_fabs_012(cross[1])>spn_fabs_012(cross[axis]))axis=1;
  if(spn_fabs_012(cross[2])>spn_fabs_012(cross[axis]))axis=2;
  spn_cross_012(x,v,f);spn_cross_012(y,e,v);
  weights[1]=x[axis]/cross[axis];weights[2]=y[axis]/cross[axis];weights[0]=(1-weights[1])-weights[2];
  return spn_isfinite_012(weights[0])&&spn_isfinite_012(weights[1])&&spn_isfinite_012(weights[2]);
}

static __attribute__((noinline)) int analytic_capsule_prism_012(
    thread CommonCCDSupportObject& a,thread CommonCCDSupportObject& b,
    int iterations,FlexDD tolerance_words,device const float* mesh_vertices,
    thread FlexDD& depth,thread FlexDD3& position,thread FlexDD3& normal) {
  bool reverse=a.geom_type==3&&b.geom_type==1;
  if(!(a.geom_type==1&&b.geom_type==3)&&!reverse)return -20;
  thread const CommonCCDSupportObject* prism=reverse?&b:&a;
  thread const CommonCCDSupportObject* capsule=reverse?&a:&b;
  if(prism->kind!=COMMON_CCD_HFIELD_PRISM||prism->vertex_count!=6||prism->vertex_offset<0)return -20;
  SPN012 original[6][3];
  for(int i=0;i<6;i++)for(int k=0;k<3;k++) {
    FlexDD residual=k==0?prism->hfield_residual[i].x:k==1?prism->hfield_residual[i].y:prism->hfield_residual[i].z;
    original[i][k]=SPN012(FlexDD{mesh_vertices[3*(prism->vertex_offset+i)+k],residual.hi,residual.lo});
  }
  SPN012 center[3]={SPN012(capsule->pos[0]),SPN012(capsule->pos[1]),SPN012(capsule->pos[2])};
  SPN012 tolerance(tolerance_words);
  SPN012 radius=SPN012(capsule->size[0])+SPN012(capsule->margin)*SPN012(.5f),axis[3],vertices[12][3];
  for(int k=0;k<3;k++)axis[k]=SPN012(capsule->size[1])*SPN012(capsule->mat[3*k+2]);
  for(int i=0;i<12;i++)for(int k=0;k<3;k++)vertices[i][k]=original[i%6][k]-(i<6?-axis[k]:axis[k]);
  SPN012 normals[34][3],heights[34],minimum=SPN012(as_type<float>(0x7f800000u)),best=SPN012(as_type<float>(0x7f800000u)),closest[3]={0};
  int valid[34]={0},projection[34]={0},inside=1,result=-1,selected=-1,nearest=-1,face_count=0;
  SPN012 result_depth=0,result_normal[3]={0},result_position[3]={0};
  if(iterations<=0||!spn_isfinite_012(radius)||radius<=0)return -23;
  for(int i=0;i<34;i++) {
    int ids[3];cp_triangle_ids_012(i,ids);
    thread const SPN012 *x=vertices[ids[0]],*y=vertices[ids[1]],*z=vertices[ids[2]];
    SPN012 length,delta[3];if(!spn_face_012(x,y,z,normals[i],&length))continue;
    SPN012 orientation=0;for(int j=0;j<12;j++){spn_sub_012(delta,vertices[j],x);SPN012 score=spn_dot_012(normals[i],delta);if(spn_fabs_012(score)>spn_fabs_012(orientation))orientation=score;}
    if(orientation>0)for(int k=0;k<3;k++)normals[i][k]=-normals[i][k];
    int supporting=1;
    for(int j=0;j<12;j++) {
      spn_sub_012(delta,vertices[j],x);SPN012 score=spn_dot_012(normals[i],delta),magnitude=0;
      for(int k=0;k<3;k++)magnitude=magnitude+spn_fabs_012(normals[i][k])*(spn_fabs_012(vertices[j][k])+spn_fabs_012(x[k]));
      /* Floating operation error enclosure, not a geometric/CCD tolerance. */
      if(score>SPN012(0x1p-46f)*magnitude)supporting=0;
    }
    if(!supporting)continue;
    face_count++;spn_sub_012(delta,center,x);heights[i]=-spn_dot_012(normals[i],delta);
    if(!spn_isfinite_012(heights[i]))return -23;
    if(heights[i]<0)inside=0;if(heights[i]<minimum)minimum=heights[i];
    SPN012 point[3];projection[i]=spn_triangle_012(point,center,x,y,z,normals[i]);valid[i]=1;
    SPN012 distance=spn_distance_012(point,center);
    if(distance<best){best=distance;nearest=i;for(int k=0;k<3;k++)closest[k]=point[k];}
  }
  SPN012 n[3],wa[3],wb[3],d,point[3],weights[3];
  if(inside) {
    for(int i=0;i<34;i++)if(valid[i]&&projection[i]&&heights[i]<=minimum+tolerance){selected=i;break;}
    if(selected<0)return -23;
    for(int k=0;k<3;k++){n[k]=normals[selected][k];point[k]=spn_fma_012(heights[selected],n[k],center[k]);}
    d=radius+heights[selected];
  } else {
    SPN012 delta[3],length;spn_sub_012(delta,center,closest);
    if(nearest<0||!spn_isfinite_012(best)||!spn_unit_012(delta,n,&length))return -23;
    selected=nearest;d=radius-length;for(int k=0;k<3;k++)point[k]=closest[k];
  }
  if(!spn_isfinite_012(d)||d<=0)return -23;
  int ids[3];cp_triangle_ids_012(selected,ids);
  if(!cp_weights_012(weights,point,vertices[ids[0]],vertices[ids[1]],vertices[ids[2]]))return -23;
  for(int k=0;k<3;k++) {
    SPN012 core=0;wa[k]=0;
    for(int j=0;j<3;j++){wa[k]=spn_fma_012(weights[j],original[ids[j]%6][k],wa[k]);core=spn_fma_012(weights[j],ids[j]<6?-axis[k]:axis[k],core);}
    wb[k]=spn_fma_012(-radius,n[k],center[k]+core);result_normal[k]=reverse?-n[k]:n[k];result_position[k]=(wa[k]+wb[k])*SPN012(.5f);
  }
  result_depth=d;
  depth=result_depth.words;
  position=FlexDD3{result_position[0].words,result_position[1].words,result_position[2].words};
  normal=FlexDD3{result_normal[0].words,result_normal[1].words,result_normal[2].words};
  return 1;
}

static __attribute__((noinline)) int common_mpr_enclosure_penetration_012(
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD3 center_a,FlexDD3 center_b,FlexDD margin,int iterations,FlexDD tolerance,
    device const float* vertices,device const int* graph,
    thread ContactGeom& contact,thread FlexDD3& position,thread FlexDD3& normal,
    thread FlexDD* retained_distance=nullptr,
    device FlexDDVertex* enclosure_vertices=nullptr,device int2* enclosure_ids=nullptr,
    device FlexEpaFace* enclosure_faces=nullptr,device int* enclosure_map=nullptr,
    device int2* enclosure_horizon=nullptr,device FlexHorizonFrame* enclosure_stack=nullptr,
    int enclosure_vertex_capacity=0,int enclosure_face_capacity=0,
    int enclosure_horizon_capacity=0,int enclosure_stack_capacity=0,
    thread bool* enclosure_certificate=nullptr) {
  if(enclosure_certificate!=nullptr)*enclosure_certificate=false;
  CommonCCDVertex portal[4];
  portal[0].point_a=center_a;portal[0].point_b=center_b;
  portal[0].minkowski=mpr_sub(center_a,center_b);
  FlexDD3 origin=flex_dd3(float3(0));
  if (mpr_eq3(portal[0].minkowski,origin))
    portal[0].minkowski.x=flex_dd_source_add(portal[0].minkowski.x,flex_dd(0x1.4p-49f));
  // Pinned libccd MPR discovery seeds the portal from the Minkowski center
  // difference, not from the first_dir callback (first_dir is consumed only
  // by GJK intersect, never by ccdMPRPenetration/discoverPortal).
  FlexDD3 direction=mpr_normalize(flex_dd3_neg(portal[0].minkowski));
  if (mpr_support_pair(a,b,direction,vertices,graph,portal[1])) return -20;
  FlexDD dot=mpr_dot(portal[1].minkowski,direction);
  if (mpr_zero(dot)||flex_dd_compare_exact(dot,flex_dd(0))<0) return 0;
  direction=mpr_cross(portal[0].minkowski,portal[1].minkowski);
  FlexDD depth;
  if (mpr_zero(mpr_dot(direction,direction))) {
    if (mpr_eq3(portal[1].minkowski,origin)) return 0;
    position=mpr_scale(mpr_add(portal[1].point_a,portal[1].point_b),flex_dd(.5f));
    depth=flex_dd_source_sqrt(mpr_dot(portal[1].minkowski,portal[1].minkowski));
    normal=mpr_normalize(portal[1].minkowski);
  } else {
    direction=mpr_normalize(direction);
    if (mpr_support_pair(a,b,direction,vertices,graph,portal[2])) return -20;
    dot=mpr_dot(portal[2].minkowski,direction);
    if (mpr_zero(dot)||flex_dd_compare_exact(dot,flex_dd(0))<0) return 0;
    direction=mpr_normalize(mpr_cross(mpr_sub(portal[1].minkowski,portal[0].minkowski),
        mpr_sub(portal[2].minkowski,portal[0].minkowski)));
    if (flex_dd_compare_exact(mpr_dot(direction,portal[0].minkowski),flex_dd(0))>0) {
      CommonCCDVertex swap=portal[1];portal[1]=portal[2];portal[2]=swap;
      direction=flex_dd3_neg(direction);
    }
    while (true) {
      if (mpr_support_pair(a,b,direction,vertices,graph,portal[3])) return -20;
      dot=mpr_dot(portal[3].minkowski,direction);
      if (mpr_zero(dot)||flex_dd_compare_exact(dot,flex_dd(0))<0) return 0;
      bool changed=false;
      dot=mpr_dot(mpr_cross(portal[1].minkowski,portal[3].minkowski),portal[0].minkowski);
      if (flex_dd_compare_exact(dot,flex_dd(0))<0 && !mpr_zero(dot)) {
        portal[2]=portal[3];changed=true;
      }
      if (!changed) {
        dot=mpr_dot(mpr_cross(portal[3].minkowski,portal[2].minkowski),portal[0].minkowski);
        if (flex_dd_compare_exact(dot,flex_dd(0))<0 && !mpr_zero(dot)) {
          portal[1]=portal[3];changed=true;
        }
      }
      if (!changed) break;
      direction=mpr_normalize(mpr_cross(mpr_sub(portal[1].minkowski,portal[0].minkowski),
          mpr_sub(portal[2].minkowski,portal[0].minkowski)));
    }
    // Discovery and refinement have no artificial iteration caps in libccd.
    while (true) {
      direction=common_mpr_enclosure_portal_direction_012(portal);
      dot=mpr_dot(direction,portal[1].minkowski);
      if (mpr_zero(dot)||flex_dd_compare_exact(dot,flex_dd(0))>0) break;
      CommonCCDVertex next;
      if (mpr_support_pair(a,b,direction,vertices,graph,next)) return -20;
      dot=mpr_dot(next.minkowski,direction);
      if ((!mpr_zero(dot)&&flex_dd_compare_exact(dot,flex_dd(0))<=0)||
          mpr_reached(portal,next,direction,tolerance)) return 0;
      mpr_expand(portal,next);
    }
    if(a.geom_type==3 || b.geom_type==3) {
      int result=analytic_capsule_prism_012(a,b,iterations,tolerance,vertices,depth,position,normal);
      if(result<0)return result;
      if(enclosure_certificate!=nullptr)*enclosure_certificate=true;
    } else if(a.geom_type==2 || b.geom_type==2) {
      int result=analytic_sphere_prism_012(a,b,iterations,tolerance,vertices,depth,position,normal);
      if(result<0)return result;
      if(enclosure_certificate!=nullptr)*enclosure_certificate=true;
    } else if(true) {
      if(enclosure_vertices==nullptr || enclosure_faces==nullptr || enclosure_map==nullptr ||
         enclosure_horizon==nullptr || enclosure_stack==nullptr)return -22;
      int support_count=0,face_count=0;
      CommonCCDEpaResult result=common_mpr_enclosure_epa_012(
          portal,4,a,b,iterations,tolerance,flex_dd(0),vertices,graph,
          enclosure_vertices,nullptr,enclosure_faces,enclosure_map,
          enclosure_horizon,enclosure_stack,enclosure_vertex_capacity,
          enclosure_face_capacity,enclosure_horizon_capacity,enclosure_stack_capacity,
          contact,nullptr,nullptr,support_count,face_count);
      if(result.status!=0) {
        // Geometric rejection is empty only with an exact actual-object proof.
        // Invalid invocation/support/capacity and uncertified seeds still fail.
        if(result.status==1 && SPN012(result.distance)==0 &&
           common_box_prism_axis_zero_012(a,b,vertices))return 0;
        return -23;
      }
      depth=result.distance;position=result.position;normal=result.normal;
      if(enclosure_certificate!=nullptr)*enclosure_certificate=true;
    } else {
    int used=0;
    while (true) {
      direction=common_mpr_enclosure_portal_direction_012(portal);
      CommonCCDVertex next;
      if (mpr_support_pair(a,b,direction,vertices,graph,next)) return -20;
      // Source deliberately tests strictly greater than max_iterations.
      if (mpr_reached(portal,next,direction,tolerance)||used>iterations) {
        normal=mpr_triangle_closest(portal[1].minkowski,portal[2].minkowski,portal[3].minkowski);
        depth=flex_dd_source_sqrt(mpr_dot(normal,normal));
        if (mpr_zero(depth)) return 0;
        normal=mpr_normalize(normal);
        position=mpr_contact_position(portal);
        break;
      }
      mpr_expand(portal,next);used++;
    }
    }
  }
  if (mpr_eq3(normal,origin)) return 0;
  if (!all(isfinite(flex_dd3_high(position))) || !all(isfinite(flex_dd3_high(normal))) ||
      !isfinite(depth.hi)) return -21;
  FlexDD distance=flex_dd_source_sub(margin,depth);
  contact.dist=distance.hi;
  if(retained_distance!=nullptr)*retained_distance=distance;
  contact.pos=flex_dd3_high(position);contact.normal=flex_dd3_high(normal);
  contact.t1=float3(0);make_frame(contact.normal,contact.t1,contact.t2);
  return 1;
}

// Production candidate/witness bridge for pinned HField--compiled-mesh pairs.
// This file calls the shared source-order GJK/EPA implementation; it does not
// contain a second penetration solver.  `mesh_hull` is the combined static
// vertex upload plus bounded per-(world,pair) output/arena tail.  The extended
// `mesh_hull_info` ABI is described in common_ccd_bridge.py.

#define COMMON_CCD_PRODUCTION_MAGIC 1128487732
#define COMMON_CCD_GEOM_INFO_WORDS 9
#define COMMON_CCD_HEADER_WORDS 35
#define COMMON_CCD_CONTACT_PRECISION_BASE 802
#define COMMON_CCD_OUTPUT_WORDS (802 + 50*14)
#define COMMON_CCD_DIAG_PRODUCTION_CONTEXT_OFFSET 7330
#define COMMON_CCD_DIAG_PRODUCTION_CONTEXT_WORDS 232

// Contact-world frame normal follows pinned mju_makeFrame's inlined
// mju_normalize3 association: rounded y product, x FMA, then z FMA.
// MPR's internal normalizers and the existing tangent producer are separate.
static inline FlexDD3 common_ccd_contact_normal_source_012(FlexDD3 a) {
  return mpr_scale(a,flex_dd_source_div(flex_dd(1),
      flex_dd_source_sqrt(flex_dd_source_dot3(a,a))));
}

static inline void common_ccd_production_diag_dd(
    device float* diagnostic, int offset, FlexDD value) {
  diagnostic[offset+0]=value.hi;
  diagnostic[offset+1]=value.lo;
  diagnostic[offset+2]=value.tail;
}

static inline void common_ccd_production_diag_dd3(
    device float* diagnostic, int offset, FlexDD3 value) {
  common_ccd_production_diag_dd(diagnostic,offset+0,value.x);
  common_ccd_production_diag_dd(diagnostic,offset+3,value.y);
  common_ccd_production_diag_dd(diagnostic,offset+6,value.z);
}

static inline void common_ccd_production_diag_context(
    device float* diagnostic, int world, int pair, int hfield_geom,
    int mesh_geom, int row, int col, int tri, int nrow, int ncol,
    FlexDD3 ph, FlexDD3 pm, float4 qh, float4 qm,
    thread const FlexDD* Rh, thread const FlexDD* Rm,
    FlexDD size0, FlexDD size1, FlexDD size2, FlexDD size3,
    FlexDD dx, FlexDD dy, FlexDD margin, FlexDD3 local_pos,
    thread const FlexDD* local_mat,
    thread const CommonCCDSupportObject& mesh) {
  if (diagnostic==nullptr) return;
  device float* out=diagnostic+COMMON_CCD_DIAG_PRODUCTION_CONTEXT_OFFSET;
  out[0]=20261004.0f;
  out[1]=float(world); out[2]=float(pair);
  out[3]=float(hfield_geom); out[4]=float(mesh_geom);
  out[5]=float(row); out[6]=float(col); out[7]=float(tri);
  out[8]=float(nrow); out[9]=float(ncol);
  out[10]=ph.x.hi; out[11]=ph.y.hi; out[12]=ph.z.hi;
  out[13]=pm.x.hi; out[14]=pm.y.hi; out[15]=pm.z.hi;
  out[16]=qh.x; out[17]=qh.y; out[18]=qh.z; out[19]=qh.w;
  out[20]=qm.x; out[21]=qm.y; out[22]=qm.z; out[23]=qm.w;
  for (int i=0;i<9;i++) {
    out[24+i]=Rh[i].hi;
    out[33+i]=Rm[i].hi;
  }
  common_ccd_production_diag_dd(out,42,size0);
  common_ccd_production_diag_dd(out,45,size1);
  common_ccd_production_diag_dd(out,48,size2);
  common_ccd_production_diag_dd(out,51,size3);
  common_ccd_production_diag_dd(out,54,dx);
  common_ccd_production_diag_dd(out,57,dy);
  common_ccd_production_diag_dd(out,60,margin);
  common_ccd_production_diag_dd3(out,63,local_pos);
  for (int i=0;i<9;i++) common_ccd_production_diag_dd(out,72+3*i,local_mat[i]);
  common_ccd_production_diag_dd3(out,99,
      FlexDD3{mesh.pos[0],mesh.pos[1],mesh.pos[2]});
  for (int i=0;i<9;i++) common_ccd_production_diag_dd(out,108+3*i,mesh.mat[i]);
  for (int i=0;i<3;i++) common_ccd_production_diag_dd(out,135+3*i,mesh.size[i]);
  common_ccd_production_diag_dd(out,144,mesh.margin);
  // New xmat transport witness: full input matrix residuals, source-relative
  // transform residuals, and geometry position residuals. These fields occupy
  // the previously unused tail of the fixed 232-word diagnostic record.
  for (int i=0;i<9;i++) {
    out[147+2*i]=Rh[i].lo; out[148+2*i]=Rh[i].tail;
    out[165+2*i]=Rm[i].lo; out[166+2*i]=Rm[i].tail;
    out[195+2*i]=local_mat[i].lo; out[196+2*i]=local_mat[i].tail;
  }
  FlexDD positions[6]={ph.x,ph.y,ph.z,pm.x,pm.y,pm.z};
  for (int i=0;i<6;i++) {
    out[183+2*i]=positions[i].lo;
    out[184+2*i]=positions[i].tail;
  }
  FlexDD relative_position[3]={local_pos.x,local_pos.y,local_pos.z};
  for (int i=0;i<3;i++) {
    out[213+2*i]=relative_position[i].lo;
    out[214+2*i]=relative_position[i].tail;
  }
}

static inline bool common_ccd_production_uses_mjc_convex(int ta, int tb) {
  int lo=min(ta,tb), hi=max(ta,tb);
  return (lo==2 && (hi==4 || hi==7))
      || (lo==3 && (hi==4 || hi==5 || hi==7))
      || (lo==4 && (hi==4 || hi==5 || hi==6 || hi==7))
      || (lo==5 && (hi==5 || hi==6 || hi==7))
      || (lo==6 && hi==7) || (lo==7 && hi==7);
}

static inline FlexDD common_ccd_production_size(
    device const float* mesh_hull, int base, int component) {
  return FlexDD{mesh_hull[base + 3*component],
                mesh_hull[base + 3*component + 1],
                mesh_hull[base + 3*component + 2]};
}

static inline CommonCCDSupportObject common_ccd_production_object() {
  CommonCCDSupportObject object;
  object.kind=COMMON_CCD_POINT; object.geom=-1; object.geom_type=-1;
  object.vertex_offset=-1; object.vertex_count=0; object.cached_vertex=-1;
  object.graph_offset=-1; object.graph_vertex_count=0;
  object.graph_word_count=0;
  for (int i=0;i<3;i++) {
    object.pos[i]=flex_dd(0.0f); object.size[i]=flex_dd(0.0f);
  }
  for (int i=0;i<9;i++) object.mat[i]=flex_dd((i%4)==0 ? 1.0f : 0.0f);
  object.margin=flex_dd(0.0f);
  for (int i=0;i<6;i++)
    object.hfield_residual[i]=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
  return object;
}

// General mesh backing admission; no ABI or persistent storage change.
// Actual production mesh storage admission. Uses existing descriptor/header
// fields and immutable float backing only; no arithmetic geometry replacement,
// support query, solver iteration, new buffer or allocation occurs here.
static inline bool common_ccd_mesh_storage_valid_012(
    int geom,int record,device const int* info,device const float* hull) {
  if(geom<0 || geom>(2147483647-COMMON_CCD_HEADER_WORDS)/COMMON_CCD_GEOM_INFO_WORDS)
    return false;
  int prefix=COMMON_CCD_HEADER_WORDS+COMMON_CCD_GEOM_INFO_WORDS*geom;
  if(record<prefix)return false;
  int header=record-prefix;
  if(header>(2147483647-COMMON_CCD_HEADER_WORDS)/2)return false;
  if(info[header]!=COMMON_CCD_PRODUCTION_MAGIC ||
     info[header+1]!=header+COMMON_CCD_HEADER_WORDS)return false;
  int offset=info[record],count=info[record+1];
  int source_address=info[record+7],duplicate_count=info[record+8];
  int source_count=info[header+4],output_base=info[header+5];
  if(offset<0 || count<=0 || count!=duplicate_count || source_address<0 ||
     source_count<=0 || count>source_count || source_address>source_count-count ||
     offset<source_address || output_base<0)return false;
  // The host appends all source vertices, then twelve words per HField,
  // then polygon-normal planes. Reconstruct this independent boundary from
  // the existing normal-table offset and HField descriptors, never offset.
  if(header%COMMON_CCD_GEOM_INFO_WORDS!=0)return false;
  int ngeom=header/COMMON_CCD_GEOM_INFO_WORDS;
  int normal_base=info[header+22],fields=0;
  if(normal_base<0 || normal_base>output_base)return false;
  for(int g=0;g<ngeom;g++) {
    int r=header+COMMON_CCD_HEADER_WORDS+COMMON_CCD_GEOM_INFO_WORDS*g;
    if(info[r]==-1 && info[r+1]==0 && info[r+2]==-1 &&
       info[r+3]==0 && info[r+4]==0 && info[r+7]>=2 && info[r+8]>=2)
      fields++;
  }
  if(fields>normal_base/12)return false;
  int source_end=normal_base-12*fields;
  if(source_end%3!=0 || source_count>source_end/3)return false;
  int source_base=source_end/3-source_count,ordinal=0;
  for(int g=0;g<ngeom;g++) {
    int r=header+COMMON_CCD_HEADER_WORDS+COMMON_CCD_GEOM_INFO_WORDS*g;
    if(info[r]==-1 && info[r+1]==0 && info[r+2]==-1 &&
       info[r+3]==0 && info[r+4]==0 && info[r+7]>=2 && info[r+8]>=2) {
      if(info[r+5]!=source_end+12*ordinal)return false;
      ordinal++;
    }
  }
  if(offset!=source_base+source_address)return false;
  // This range is now contained in the original static source-vertex region,
  // before output/scratch. IEEE classification preserves finite subnormals
  // and signed zeros; it cannot be optimized into a fast-math assumption.
  for(int i=0;i<count;i++)for(int k=0;k<3;k++) {
    uint bits=as_type<uint>(hull[3*(offset+i)+k]);
    if((bits&0x7f800000u)==0x7f800000u)return false;
  }
  return true;
}

static inline CommonCCDSupportObject common_ccd_production_mesh(
    int geom, int record, device const int* mesh_hull_info,
    device const float* mesh_hull, FlexDD3 local_position,
    thread const FlexDD* local_matrix, float3 size, float margin) {
  CommonCCDSupportObject object=common_ccd_production_object();
  if(!common_ccd_mesh_storage_valid_012(geom,record,mesh_hull_info,mesh_hull)) {
    object.kind=-1;return object;
  }

  object.kind=COMMON_CCD_HULL; object.geom=geom; object.geom_type=7;
  object.vertex_offset=mesh_hull_info[record+0];
  object.vertex_count=mesh_hull_info[record+1];
  object.cached_vertex=-1;
  object.graph_offset=mesh_hull_info[record+2];
  object.graph_vertex_count=mesh_hull_info[record+3];
  object.graph_word_count=mesh_hull_info[record+4];
  object.pos[0]=local_position.x;
  object.pos[1]=local_position.y;
  object.pos[2]=local_position.z;
  // Both the input/output arenas are row-major, matching mjtNum geom_xmat.
  // Keep all three float words; rebuilding a matrix from geom_xquat drops
  // source residuals that affect support projection and the selected witness.
  for (int i=0;i<9;i++) object.mat[i]=local_matrix[i];
  object.size[0]=flex_dd(size.x); object.size[1]=flex_dd(size.y);
  object.size[2]=flex_dd(size.z); object.margin=flex_dd(margin);
  return object;
}

static inline FlexDD3 common_ccd_production_hfield_point(
    float x, float y, float z) {
  return FlexDD3{flex_dd(x),flex_dd(y),flex_dd(z)};
}

// A source scalar has already rounded to binary64 before this storage boundary.
// The public float high can differ from the internal expansion's leading word.
// Rebase BOTH residual words against that actual published high. Scale before
// subtraction so representable subnormal residual storage survives Metal FTZ.
static inline FlexDD common_ccd_publication_rebase_012(FlexDD value,float high) {
  uint nonfinite=(as_type<uint>(value.hi)&0x7f800000u);
  if(nonfinite==0x7f800000u ||
     (as_type<uint>(value.lo)&0x7f800000u)==0x7f800000u ||
     (as_type<uint>(value.tail)&0x7f800000u)==0x7f800000u ||
     (as_type<uint>(high)&0x7f800000u)==0x7f800000u) return value;
  // Remove exact opposite storage words before choosing an arithmetic scale.
  // Otherwise a canceled normal leading pair can flush a remaining subnormal.
  uint ah=as_type<uint>(value.hi),al=as_type<uint>(value.lo),at=as_type<uint>(value.tail);
  if((ah^al)==0x80000000u)value={value.tail,0,0};
  else if((ah^at)==0x80000000u)value={value.lo,0,0};
  else if((al^at)==0x80000000u)value={value.hi,0,0};
  float lead=value.hi;
  if((as_type<uint>(lead)&0x7fffffffu)==0u)lead=value.lo;
  if((as_type<uint>(lead)&0x7fffffffu)==0u)lead=value.tail;
  if((as_type<uint>(lead)&0x7fffffffu)==0u)return {high,value.lo,value.tail};
  int shift=20-flex_float_exponent(lead);
  FlexDD scaled=flex_dd_scalbn(value,shift);
  FlexDD residual=flex_dd_source_sub(scaled,{flex_float_scalbn(high,shift),0,0});
  float low_scaled=(residual.hi+residual.lo)+residual.tail;
  FlexDD remainder=flex_dd_source_sub(residual,{low_scaled,0,0});
  float tail_scaled=(remainder.hi+remainder.lo)+remainder.tail;
  return {high,flex_float_scalbn(low_scaled,-shift),flex_float_scalbn(tail_scaled,-shift)};
}

static inline void common_ccd_production_store_contact(
    device float* mesh_hull, int output, int slot,
    thread const ContactGeom& contact) {
  int base=output+2+16*slot;
  mesh_hull[base+0]=contact.dist;
  mesh_hull[base+1]=contact.normal.x;
  mesh_hull[base+2]=contact.normal.y;
  mesh_hull[base+3]=contact.normal.z;
  mesh_hull[base+4]=contact.pos.x;
  mesh_hull[base+5]=contact.pos.y;
  mesh_hull[base+6]=contact.pos.z;
  mesh_hull[base+7]=contact.t1.x;
  mesh_hull[base+8]=contact.t1.y;
  mesh_hull[base+9]=contact.t1.z;
  mesh_hull[base+10]=contact.t2.x;
  mesh_hull[base+11]=contact.t2.y;
  mesh_hull[base+12]=contact.t2.z;
  // Every ordinary producer initializes its residual payload. A source-wide
  // producer overwrites it only after the high contact has been published.
  int precision=output+COMMON_CCD_CONTACT_PRECISION_BASE+14*slot;
  for(int k=0;k<14;k++) mesh_hull[precision+k]=0.0f;
}

static inline void common_ccd_production_fail(
    device atomic_int* world_status, int world) {
  // The ordinary coupled solver consumes nonzero status before reading its
  // candidate rows. Every bridge failure has the same public meaning: this
  // world's native narrowphase could not produce a trustworthy row set.
  atomic_store_explicit(&world_status[world], 2, memory_order_relaxed);
}

static inline FlexDD3 common_ccd_load_geom_position(
    device const float* high, device const float* low,
    device const float* tail, int base) {
  return FlexDD3{
      FlexDD{high[base],low[base],tail[base]},
      FlexDD{high[base+1],low[base+1],tail[base+1]},
      FlexDD{high[base+2],low[base+2],tail[base+2]}};
}

static inline void common_ccd_load_geom_matrix(
    device const float* high, device const float* low,
    device const float* tail, int base, thread FlexDD* matrix) {
  for (int i=0;i<9;i++)
    matrix[i]=FlexDD{high[base+i],low[base+i],tail[base+i]};
}

static inline bool common_ccd_finite_pose(
    FlexDD3 position, thread const FlexDD* matrix) {
  if (!all(isfinite(float3(position.x.hi,position.y.hi,position.z.hi))) ||
      !all(isfinite(float3(position.x.lo,position.y.lo,position.z.lo))) ||
      !all(isfinite(float3(position.x.tail,position.y.tail,position.z.tail))))
    return false;
  for (int i=0;i<9;i++)
    if (!isfinite(matrix[i].hi) || !isfinite(matrix[i].lo) ||
        !isfinite(matrix[i].tail)) return false;
  return true;
}

static inline void common_ccd_matrix_transpose_product(
    thread const FlexDD* a, thread const FlexDD* b,
    thread FlexDD* result) {
  // The pinned arm64 mji_mulMatTMat3 rounds the middle product, then fuses
  // the first and last products into it. Rounding the first product instead
  // changes near-tied prism support choices even at identical world poses.
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
    result[3*i+j]=flex_dd_source_matvec_row(
        a[i],a[3+i],a[6+i],b[j],b[3+j],b[6+j]);
  }
}

static inline int common_ccd_production_one_prism(
    thread CommonCCDSupportObject& prism,
    thread CommonCCDSupportObject& mesh,
    FlexDD3 center_prism, FlexDD3 center_mesh,
    int iterations, FlexDD tolerance,
    device const float* mesh_vertices, device const int* mesh_graph,
    device FlexDDVertex* epa_vertices, device int2* vertex_ids,
    device FlexEpaFace* epa_faces,
    device int* epa_face_map, device int2* epa_horizon,
    device FlexHorizonFrame* epa_stack,
    int vertex_capacity, int face_capacity, int horizon_capacity,
    int stack_capacity, thread ContactGeom& contact,
    thread int2* out_face_ids, device float* diagnostic,
    int diagnostic_row, int diagnostic_col, int diagnostic_triangle,
    thread FlexDD3* retained_position=nullptr,
    thread FlexDD3* retained_normal=nullptr,
    thread FlexDD* retained_distance=nullptr) {
  CommonCCDVertex simplex[4];
  for (int i=0;i<4;i++) {
    simplex[i].point_a=FlexDD3{flex_dd(0),flex_dd(0),flex_dd(0)};
    simplex[i].point_b=simplex[i].point_a;
    simplex[i].minkowski=simplex[i].point_a;
    simplex[i].index_a=-1; simplex[i].index_b=-1;
  }
  if (diagnostic!=nullptr) {
    diagnostic[0]=1.0f;
    diagnostic[1]=float(diagnostic_row);
    diagnostic[2]=float(diagnostic_col);
    diagnostic[3]=float(diagnostic_triangle);
  }
  CommonCCDGjkResult gjk=common_ccd_source_gjk(
      prism,mesh,center_prism,center_mesh,tolerance,
      flex_dd(0.0f),iterations,mesh_vertices,mesh_graph,simplex,
      diagnostic,diagnostic==nullptr ? 0 : COMMON_CCD_DIAG_SUPPORTS);
  int result_base=COMMON_CCD_DIAG_SUPPORT_OFFSET
      +COMMON_CCD_DIAG_SUPPORTS*COMMON_CCD_DIAG_WORDS_PER_SUPPORT;
  if (diagnostic!=nullptr) {
    device float* context=diagnostic+COMMON_CCD_DIAG_PRODUCTION_CONTEXT_OFFSET;
    common_ccd_production_diag_dd3(context,159,center_prism);
    common_ccd_production_diag_dd3(context,168,center_mesh);
    for (int v=0;v<6;v++) {
      int address=3*(prism.vertex_offset+v);
      FlexDD3 point=FlexDD3{
          FlexDD{mesh_vertices[address],prism.hfield_residual[v].x.hi,
                 prism.hfield_residual[v].x.lo},
          FlexDD{mesh_vertices[address+1],prism.hfield_residual[v].y.hi,
                 prism.hfield_residual[v].y.lo},
          FlexDD{mesh_vertices[address+2],prism.hfield_residual[v].z.hi,
                 prism.hfield_residual[v].z.lo}};
      common_ccd_production_diag_dd3(context,177+9*v,point);
    }
    diagnostic[4]=float(gjk.status);
    diagnostic[5]=float(gjk.iterations);
    diagnostic[6]=float(gjk.simplex_count);
    diagnostic[7]=float(gjk.needs_intersection);
    diagnostic[8]=float(gjk.support_count);
    diagnostic[31]=gjk.support_count>COMMON_CCD_DIAG_SUPPORTS ? 1.0f : 0.0f;
    diagnostic[9]=gjk.distance.hi;
    diagnostic[10]=gjk.distance.lo;
    diagnostic[11]=gjk.distance.tail;
    common_ccd_diag_store_dd3(diagnostic,12,gjk.witness_a);
    common_ccd_diag_store_dd3(diagnostic,21,gjk.witness_b);
    for (int i=0;i<4;i++) {
      int base=result_base+29*i;
      common_ccd_diag_store_dd3(diagnostic,base,simplex[i].point_a);
      common_ccd_diag_store_dd3(diagnostic,base+9,simplex[i].point_b);
      common_ccd_diag_store_dd3(diagnostic,base+18,simplex[i].minkowski);
      diagnostic[base+27]=float(simplex[i].index_a);
      diagnostic[base+28]=float(simplex[i].index_b);
    }
  }
  if (gjk.status==1) return 0;
  if (gjk.status!=0) return -gjk.status;
  if (!gjk.needs_intersection) return 0;
  int epa_support_count=0, epa_face_count=0;
  CommonCCDEpaResult epa=common_ccd_source_epa(
      simplex,gjk.simplex_count,prism,mesh,iterations,tolerance,gjk.distance,
      mesh_vertices,mesh_graph,epa_vertices,vertex_ids,epa_faces,epa_face_map,
      epa_horizon,epa_stack,vertex_capacity,face_capacity,horizon_capacity,
      stack_capacity,contact,
      diagnostic==nullptr ? nullptr :
          diagnostic+COMMON_CCD_DIAG_EPA_SUPPORT_OFFSET,
      diagnostic==nullptr ? nullptr : diagnostic+COMMON_CCD_DIAG_EPA_FACE_OFFSET,
      epa_support_count,epa_face_count);
  if (diagnostic!=nullptr) {
    int epa_base=result_base+116;
    diagnostic[epa_base+0]=float(epa.status);
    diagnostic[epa_base+1]=epa.distance.hi;
    diagnostic[epa_base+2]=epa.distance.lo;
    diagnostic[epa_base+3]=epa.distance.tail;
    diagnostic[epa_base+4]=float(epa.face_vertices.x);
    diagnostic[epa_base+5]=float(epa.face_vertices.y);
    diagnostic[epa_base+6]=float(epa.face_vertices.z);
    diagnostic[epa_base+7]=contact.dist;
    diagnostic[epa_base+8]=contact.normal.x;
    diagnostic[epa_base+9]=contact.normal.y;
    diagnostic[epa_base+10]=contact.normal.z;
    diagnostic[epa_base+11]=contact.pos.x;
    diagnostic[epa_base+12]=contact.pos.y;
    diagnostic[epa_base+13]=contact.pos.z;
    // The 20-word EPA summary is followed immediately by the support trace,
    // so the event counts live in the two-word diagnostic trailer.  Keep the
    // final face's three support-id pairs in summary words 14..19.
    diagnostic[COMMON_CCD_DIAG_COUNTS_OFFSET+0]=float(epa_support_count);
    diagnostic[COMMON_CCD_DIAG_COUNTS_OFFSET+1]=float(epa_face_count);
    if (vertex_ids!=nullptr && epa.status==0) {
      diagnostic[epa_base+14]=float(vertex_ids[epa.face_vertices.x].x);
      diagnostic[epa_base+15]=float(vertex_ids[epa.face_vertices.x].y);
      diagnostic[epa_base+16]=float(vertex_ids[epa.face_vertices.y].x);
      diagnostic[epa_base+17]=float(vertex_ids[epa.face_vertices.y].y);
      diagnostic[epa_base+18]=float(vertex_ids[epa.face_vertices.z].x);
      diagnostic[epa_base+19]=float(vertex_ids[epa.face_vertices.z].y);
    }
  }
  if (epa.status==0 && out_face_ids!=nullptr) {
    int3 fv=epa.face_vertices;
    out_face_ids[0]=vertex_ids[fv.x];
    out_face_ids[1]=vertex_ids[fv.y];
    out_face_ids[2]=vertex_ids[fv.z];
  }
  if (epa.status==0) {
    if (retained_position!=nullptr) *retained_position=epa.position;
    if (retained_normal!=nullptr) *retained_normal=epa.normal;
    if (retained_distance!=nullptr) *retained_distance=flex_dd_neg(epa.distance);
  }
  if (epa.status==1) return 0;  // Source rejected a nonpenetrating seed.
  return epa.status==0 ? 1 : -10-epa.status;
}

static inline CommonCCDSupportObject common_ccd_production_geom(
    int geom, int geom_type, int record,
    device const int* mesh_hull_info, device const float* mesh_hull,
    FlexDD3 position, thread const FlexDD* rotation,
    float3 size, float margin) {
  if (geom_type==7) {
    return common_ccd_production_mesh(geom,record,mesh_hull_info,mesh_hull,
                                      position,rotation,size,margin);
  }
  CommonCCDSupportObject object=common_ccd_production_object();
  object.geom=geom; object.geom_type=geom_type;
  object.vertex_offset=-1; object.vertex_count=0; object.cached_vertex=-1;
  object.graph_offset=-1; object.graph_vertex_count=0;
  object.graph_word_count=0;
  object.kind=geom_type==2 ? COMMON_CCD_SPHERE
      : geom_type==3 ? COMMON_CCD_CAPSULE
      : geom_type==4 ? COMMON_CCD_ELLIPSOID
      : geom_type==5 ? COMMON_CCD_CYLINDER
      : geom_type==6 ? COMMON_CCD_BOX : -1;
  object.pos[0]=position.x;
  object.pos[1]=position.y;
  object.pos[2]=position.z;
  for (int i=0;i<9;i++) object.mat[i]=rotation[i];
  object.size[0]=flex_dd(size.x);
  object.size[1]=flex_dd(size.y);
  object.size[2]=flex_dd(size.z);
  object.margin=flex_dd(margin);
  return object;
}

static inline int common_ccd_production_one_rigid_pair(
    thread CommonCCDSupportObject& a,
    thread CommonCCDSupportObject& b,
    FlexDD3 center_a, FlexDD3 center_b, FlexDD margin,
    int iterations, FlexDD tolerance,
    device const float* mesh_vertices, device const int* mesh_graph,
    device FlexDDVertex* epa_vertices, device int2* vertex_ids,
    device FlexEpaFace* epa_faces,
    device int* epa_face_map, device int2* epa_horizon,
    device FlexHorizonFrame* epa_stack,
    int vertex_capacity, int face_capacity, int horizon_capacity,
    int stack_capacity, thread ContactGeom& contact,
    thread int2* out_face_ids, thread int3& out_face_vertices,
    thread FlexDD3* source_position=nullptr,
    thread FlexDD3* source_normal=nullptr) {
  out_face_vertices=int3(-1);
  CommonCCDSupportObject solve_a=a, solve_b=b;
  bool shrink_a=(a.geom_type==2 || a.geom_type==3);
  bool shrink_b=(b.geom_type==2 || b.geom_type==3);
  FlexDD full_a=flex_dd(0.0f), full_b=flex_dd(0.0f);
  if (shrink_a) {
    full_a=flex_dd_source_add(a.size[0],
        flex_dd_source_mul(a.margin,flex_dd(0.5f)));
    solve_a.kind=a.geom_type==2 ? COMMON_CCD_POINT : COMMON_CCD_LINE;
    solve_a.geom=-1; solve_a.margin=flex_dd(0.0f);
  }
  if (shrink_b) {
    full_b=flex_dd_source_add(b.size[0],
        flex_dd_source_mul(b.margin,flex_dd(0.5f)));
    solve_b.kind=b.geom_type==2 ? COMMON_CCD_POINT : COMMON_CCD_LINE;
    solve_b.geom=-1; solve_b.margin=flex_dd(0.0f);
  }
  if (shrink_a || shrink_b) {
    FlexDD cutoff=flex_dd_add(full_a,full_b);
    CommonCCDVertex shallow_simplex[4];
    for (int i=0;i<4;i++) {
      shallow_simplex[i].point_a=FlexDD3{flex_dd(0),flex_dd(0),flex_dd(0)};
      shallow_simplex[i].point_b=shallow_simplex[i].point_a;
      shallow_simplex[i].minkowski=shallow_simplex[i].point_a;
      shallow_simplex[i].index_a=-1; shallow_simplex[i].index_b=-1;
    }
    CommonCCDGjkResult shallow=common_ccd_source_gjk(
        solve_a,solve_b,center_a,center_b,tolerance,cutoff,iterations,
        mesh_vertices,mesh_graph,shallow_simplex,nullptr,0);
    // The source changes support callbacks on these same CCD objects.
    // Cached mesh vertices survive inflation and the deep fallback.
    a.cached_vertex=solve_a.cached_vertex;
    b.cached_vertex=solve_b.cached_vertex;
    if (shallow.status==2 || shallow.status==3) return -shallow.status;
    if (shallow.status==1) return 0;
    if (flex_dd_compare_exact(shallow.distance,tolerance)>0) {
      FlexDD distance=flex_dd_sub(shallow.distance,cutoff);
      if (flex_dd_compare_exact(distance,flex_dd(0.0f))>=0) return 0;
      // Source inflate rounds each witness subtraction, normalization,
      // margin multiplication and addition before mjc_penetration reads it.
      FlexDD3 delta=flex_dd3_source_sub(shallow.witness_b,shallow.witness_a);
      FlexDD delta_norm=flex_dd_source_sqrt(flex_dd3_dot(delta,delta));
      const FlexDD minval=FlexDD{1e-15f,-3.627493647833322e-24f,0.0f};
      FlexDD3 direction=flex_dd3(float3(1,0,0));
      if (flex_dd_compare_exact(delta_norm,minval)>=0)
        direction=mpr_scale(delta,flex_dd_source_div(flex_dd(1),delta_norm));
      FlexDD3 wa=flex_dd3_source_madd(direction,full_a,shallow.witness_a);
      FlexDD3 wb=flex_dd3_source_madd(direction,flex_dd_neg(full_b),shallow.witness_b);
      FlexDD3 normal_delta=flex_dd3_source_sub(wa,wb);
      FlexDD normal_norm=flex_dd_source_sqrt(flex_dd3_dot(normal_delta,normal_delta));
      // Below mjMINVAL, mju_normalize3 publishes +x even when a tiny
      // negative depth is retained. Dividing it reverses an exact-touch
      // frame and produces a false impulse.
      FlexDD3 wide_normal=flex_dd3(float3(1,0,0));
      if (flex_dd_compare_exact(normal_norm,minval)>=0)
        wide_normal=mpr_scale(normal_delta,flex_dd_source_div(flex_dd(1),normal_norm));
      contact.normal=flex_dd3_high(wide_normal);
      contact.dist=(distance.hi+distance.lo+distance.tail)
          +(margin.hi+margin.lo+margin.tail);
      FlexDD3 wide_position=mpr_scale(mpr_add(wa,wb),flex_dd(0.5f));
      contact.pos=flex_dd3_high(wide_position);
      if (source_position!=nullptr) *source_position=wide_position;
      if (source_normal!=nullptr) *source_normal=wide_normal;
      contact.t1=float3(0.0f); make_frame(contact.normal,contact.t1,contact.t2);
      return 1;
    }
    // Pinned mjc_ccd restores the full sphere/capsule callbacks and margins
    // before the deep-contact GJK/EPA pass.
  }
  CommonCCDVertex simplex[4];
  for (int i=0;i<4;i++) {
    simplex[i].point_a=FlexDD3{flex_dd(0),flex_dd(0),flex_dd(0)};
    simplex[i].point_b=simplex[i].point_a;
    simplex[i].minkowski=simplex[i].point_a;
    simplex[i].index_a=-1; simplex[i].index_b=-1;
  }
  CommonCCDGjkResult gjk=common_ccd_source_gjk(
      a,b,center_a,center_b,tolerance,flex_dd(0.0f),iterations,
      mesh_vertices,mesh_graph,simplex,nullptr,0);
  if (gjk.status==1) return 0;
  if (gjk.status!=0) return -gjk.status;
  if (!gjk.needs_intersection) return 0;
  int ignored_epa_support_count=0, ignored_epa_face_count=0;
  CommonCCDEpaResult epa=common_ccd_source_epa(
      simplex,gjk.simplex_count,a,b,iterations,tolerance,gjk.distance,
      mesh_vertices,mesh_graph,epa_vertices,vertex_ids,epa_faces,epa_face_map,
      epa_horizon,epa_stack,vertex_capacity,face_capacity,horizon_capacity,
      stack_capacity,contact,nullptr,nullptr,
      ignored_epa_support_count,ignored_epa_face_count);
  if (epa.status==1) return 0;
  if (epa.status!=0) return -10-epa.status;
  if (source_position!=nullptr) *source_position=epa.position;
  if (source_normal!=nullptr) *source_normal=epa.normal;
  out_face_vertices=epa.face_vertices;
  if (out_face_ids!=nullptr) {
    int3 fv=epa.face_vertices;
    out_face_ids[0]=vertex_ids[fv.x];
    out_face_ids[1]=vertex_ids[fv.y];
    out_face_ids[2]=vertex_ids[fv.z];
  }
  // mjc_penetration adds the caller's (geom margin + gap) after mjc_ccd has
  // inflated support points by the same amount. Store the source's final
  // physical-geometry distance, while contact activation still uses margin.
  contact.dist += margin.hi+margin.lo+margin.tail;
  return 1;
}

// Grid boundaries use the full source scalar, including a residual that
// crosses an integral high word. The bounded grid coordinate needs at most
// one correction to the float32 estimate, without dropping that residual.
static inline int common_ccd_grid_bound(FlexDD value, bool upper) {
  int result=int(floor(value.hi));
  if (flex_dd_compare_exact(value,flex_dd(float(result)))<0) result--;
  else if (flex_dd_compare_exact(value,flex_dd(float(result+1)))>=0) result++;
  if (upper && flex_dd_compare_exact(value,flex_dd(float(result)))>0) result++;
  return result;
}

kernel void common_ccd_hfield_mesh_candidates(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_pos_low [[buffer(2)]],
    device const float* geom_pos_tail [[buffer(3)]],
    device const float* geom_xmat [[buffer(4)]],
    device const float* geom_xmat_low [[buffer(5)]],
    device const float* geom_xmat_tail [[buffer(6)]],
    device const float* geom_size [[buffer(7)]],
    device const int* geom_type [[buffer(8)]],
    device const float* geom_rbound [[buffer(9)]],
    device const int* pair_geoms [[buffer(10)]],
    device const float* pair_margin_gap [[buffer(11)]],
    device const int* pair_dims [[buffer(12)]],
    device const int* logical_pair_to_packed [[buffer(13)]],
    device float* mesh_hull [[buffer(14)]],
    device int* mesh_hull_info [[buffer(15)]],
    device atomic_int* world_status [[buffer(16)]],
    device const float* pair_margin_source [[buffer(17)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=pair_dims[3], npairs=pair_dims[1], ngeom=pair_dims[6];
  if (npairs<=0 || uint(tid)>=uint(batch*npairs)) return;
  int world=int(tid)/npairs, pair=int(tid)%npairs;
  if (logical_pair_to_packed[world*npairs+pair]<0) return;
  int common_header=9*pair_dims[6];
  int diagnostic_query=mesh_hull_info[common_header+29];
  int diagnostic_row=mesh_hull_info[common_header+30];
  int diagnostic_col=mesh_hull_info[common_header+31];
  int diagnostic_tri=mesh_hull_info[common_header+32];
  bool trace_query=diagnostic_query==world*npairs+pair;
  int diagnostic_base=mesh_hull_info[common_header+27];
  device float* diagnostic=(trace_query && diagnostic_base>=0)
      ? mesh_hull+diagnostic_base : nullptr;
  if (trace_query && diagnostic!=nullptr) diagnostic[0]=0.0f;
  int a=pair_geoms[2*pair], b=pair_geoms[2*pair+1];
  bool reversed=(geom_type[b]==1);
  int hfield_geom=reversed ? b : a;
  int mesh_geom=reversed ? a : b;
  if (geom_type[hfield_geom]!=1 || geom_type[mesh_geom]<2 ||
      geom_type[mesh_geom]>7) return;
  int mask_pointer=10+npairs+1+pair_dims[2];
  int mask_offset=pair_dims[mask_pointer];
  if (mask_offset<0 || pair_dims[mask_offset+world]==0) return;
  int header=9*ngeom;
  if (mesh_hull_info[header+0]!=COMMON_CCD_PRODUCTION_MAGIC) {
    common_ccd_production_fail(world_status,world); return;
  }
  int geom_records=mesh_hull_info[header+1];
  int graph_base=mesh_hull_info[header+2];
  int output_base=mesh_hull_info[header+5];
  int output_stride=mesh_hull_info[header+6];
  int scratch_base=mesh_hull_info[header+7];
  int scratch_stride=mesh_hull_info[header+8];
  int iterations=mesh_hull_info[header+9];
  int tolerance_base=mesh_hull_info[header+10];
  if (geom_records<0 || graph_base<0 || output_base<0 || output_stride<COMMON_CCD_OUTPUT_WORDS ||
      scratch_base<0 || scratch_stride<=0 || iterations<=0 || tolerance_base<0)
    { common_ccd_production_fail(world_status,world); return; }
  int query=world*npairs+pair;
  if (diagnostic_query!=query) diagnostic=nullptr;
  int out=output_base+query*output_stride;
  mesh_hull[out]=0.0f; mesh_hull[out+1]=0.0f;
  int scratch=scratch_base+query*scratch_stride;
  int rec_hf=geom_records+COMMON_CCD_GEOM_INFO_WORDS*hfield_geom;
  int rec_mesh=geom_records+COMMON_CCD_GEOM_INFO_WORDS*mesh_geom;
  int size_base=mesh_hull_info[rec_hf+5];
  int data_base=mesh_hull_info[rec_hf+6];
  int nrow=mesh_hull_info[rec_hf+7], ncol=mesh_hull_info[rec_hf+8];
  if (size_base<0 || data_base<0 || nrow<2 || ncol<2) {
    mesh_hull[out+1]=1.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  FlexDD size0=common_ccd_production_size(mesh_hull,size_base,0);
  FlexDD size1=common_ccd_production_size(mesh_hull,size_base,1);
  FlexDD size2=common_ccd_production_size(mesh_hull,size_base,2);
  FlexDD size3=common_ccd_production_size(mesh_hull,size_base,3);
  // Detection uses source-rounded margin+gap; the solver still consumes
  // the separately stored inclusion margin downstream.
  FlexDD margin_source=FlexDD{pair_margin_source[pair],
      pair_margin_source[npairs+pair],pair_margin_source[2*npairs+pair]};
  float margin=margin_source.hi;
  if (!isfinite(margin_source.hi) || !isfinite(margin_source.lo) ||
      !isfinite(margin_source.tail)) {
    mesh_hull[out+1]=3.0f;
    common_ccd_production_fail(world_status,world); return;
  }

  int go=world*ngeom;
  FlexDD3 ph=common_ccd_load_geom_position(geom_pos,geom_pos_low,
      geom_pos_tail,3*(go+hfield_geom));
  FlexDD3 pm=common_ccd_load_geom_position(geom_pos,geom_pos_low,
      geom_pos_tail,3*(go+mesh_geom));
  float4 qh=float4(geom_quat[4*(go+hfield_geom)],
      geom_quat[4*(go+hfield_geom)+1],geom_quat[4*(go+hfield_geom)+2],
      geom_quat[4*(go+hfield_geom)+3]);
  float4 qm=float4(geom_quat[4*(go+mesh_geom)],
      geom_quat[4*(go+mesh_geom)+1],geom_quat[4*(go+mesh_geom)+2],
      geom_quat[4*(go+mesh_geom)+3]);
  FlexDD Rh[9], Rm[9];
  common_ccd_load_geom_matrix(geom_xmat,geom_xmat_low,geom_xmat_tail,
      9*(go+hfield_geom),Rh);
  common_ccd_load_geom_matrix(geom_xmat,geom_xmat_low,geom_xmat_tail,
      9*(go+mesh_geom),Rm);
  float3 size_mesh=float3(geom_size[3*mesh_geom],geom_size[3*mesh_geom+1],geom_size[3*mesh_geom+2]);
  if (!common_ccd_finite_pose(ph,Rh) || !common_ccd_finite_pose(pm,Rm) ||
      !all(isfinite(size_mesh)) || !all(isfinite(qh)) || !all(isfinite(qm))) {
    mesh_hull[out+1]=4.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  FlexDD3 local_pos=common_ccd_matrix_transpose_vector(
      Rh,flex_dd3_source_sub(pm,ph));
  FlexDD local_mat[9];
  common_ccd_matrix_transpose_product(Rh,Rm,local_mat);
  CommonCCDSupportObject mesh=common_ccd_production_geom(
      mesh_geom,geom_type[mesh_geom],rec_mesh,mesh_hull_info,mesh_hull,
      local_pos,local_mat,size_mesh,margin);
  CommonCCDSupportObject original=common_ccd_production_geom(
      mesh_geom,geom_type[mesh_geom],rec_mesh,mesh_hull_info,mesh_hull,
      pm,Rm,size_mesh,margin);
  original.margin=margin_source;
  int source_sizes=mesh_hull_info[header+34];
  int source_bounds=mesh_hull_info[header+33];
  if (source_sizes<0 || source_bounds<0 || mesh.kind<0 ||
      (geom_type[mesh_geom]==7 && (mesh.vertex_offset<0 || mesh.vertex_count<=0))) {
    mesh_hull[out+1]=3.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  for (int axis=0;axis<3;axis++) {
    mesh.size[axis]=common_ccd_production_size(
        mesh_hull,source_sizes+9*mesh_geom,axis);
    original.size[axis]=mesh.size[axis];
  }
  int vertex_capacity=mesh_hull_info[header+11];
  int face_capacity=mesh_hull_info[header+12];
  int horizon_capacity=mesh_hull_info[header+13];
  int stack_capacity=mesh_hull_info[header+14];
  if (vertex_capacity<6 || face_capacity<6 || horizon_capacity<3 ||
      stack_capacity<1) {
    mesh_hull[out+1]=2.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  int vertex_words=29*vertex_capacity;
  int face_words=20*face_capacity;
  int map_words=face_capacity;
  int horizon_words=2*horizon_capacity;
  int stack_words=6*stack_capacity;
  int horizon_alignment=vertex_words&1;
  int prism_vertex_offset=(scratch+vertex_words+face_words+map_words+
                           horizon_alignment+horizon_words+stack_words)/3;
  // Keep each per-query dynamic prism range globally addressable to the
  // shared HField-prism support callback.
  int dynamic_vertex_base=mesh_hull_info[header+3];
  int query_vertex_offset=dynamic_vertex_base+query*6;
  prism_vertex_offset=query_vertex_offset;

  device FlexDDVertex* epa_vertices=reinterpret_cast<device FlexDDVertex*>(mesh_hull+scratch);
  device FlexEpaFace* epa_faces=reinterpret_cast<device FlexEpaFace*>(mesh_hull+scratch+vertex_words);
  device int* epa_face_map=reinterpret_cast<device int*>(mesh_hull+scratch+vertex_words+face_words);
  device int2* epa_horizon=reinterpret_cast<device int2*>(
      mesh_hull+scratch+vertex_words+face_words+map_words+horizon_alignment);
  device FlexHorizonFrame* epa_stack=reinterpret_cast<device FlexHorizonFrame*>(
      mesh_hull+scratch+vertex_words+face_words+map_words+horizon_alignment+
      horizon_words);
  device float* prism_vertices=mesh_hull+3*prism_vertex_offset;

  FlexDD rb=flex_dd_source_add(
      common_ccd_production_size(mesh_hull,source_bounds,mesh_geom),margin_source);
  if (flex_dd_compare_exact(size0,flex_dd_source_sub(local_pos.x,rb))<0 ||
      flex_dd_compare_exact(flex_dd_neg(size0),flex_dd_source_add(local_pos.x,rb))>0 ||
      flex_dd_compare_exact(size1,flex_dd_source_sub(local_pos.y,rb))<0 ||
      flex_dd_compare_exact(flex_dd_neg(size1),flex_dd_source_add(local_pos.y,rb))>0 ||
      flex_dd_compare_exact(size2,flex_dd_source_sub(local_pos.z,rb))<0 ||
      flex_dd_compare_exact(flex_dd_neg(size3),flex_dd_source_add(local_pos.z,rb))>0) return;
  mesh.margin=flex_dd(0.0f);
  FlexDD3 unit_x=FlexDD3{flex_dd(1),flex_dd(0),flex_dd(0)};
  FlexDD3 unit_y=FlexDD3{flex_dd(0),flex_dd(1),flex_dd(0)};
  FlexDD3 unit_z=FlexDD3{flex_dd(0),flex_dd(0),flex_dd(1)};
  CommonCCDSupportResult sx=common_ccd_support(mesh,unit_x,mesh_hull);
  CommonCCDSupportResult nx=common_ccd_support(mesh,flex_dd3_neg(unit_x),mesh_hull);
  CommonCCDSupportResult sy=common_ccd_support(mesh,unit_y,mesh_hull);
  CommonCCDSupportResult ny=common_ccd_support(mesh,flex_dd3_neg(unit_y),mesh_hull);
  CommonCCDSupportResult sz=common_ccd_support(mesh,unit_z,mesh_hull);
  CommonCCDSupportResult nz=common_ccd_support(mesh,flex_dd3_neg(unit_z),mesh_hull);
  if (sx.status || nx.status || sy.status || ny.status || sz.status || nz.status) {
    mesh_hull[out+1]=3.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  if (flex_dd_compare_exact(flex_dd_source_sub(nx.point.x,margin_source),size0)>0 ||
      flex_dd_compare_exact(flex_dd_source_add(sx.point.x,margin_source),flex_dd_neg(size0))<0 ||
      flex_dd_compare_exact(flex_dd_source_sub(ny.point.y,margin_source),size1)>0 ||
      flex_dd_compare_exact(flex_dd_source_add(sy.point.y,margin_source),flex_dd_neg(size1))<0 ||
      flex_dd_compare_exact(flex_dd_source_sub(nz.point.z,margin_source),size2)>0 ||
      flex_dd_compare_exact(flex_dd_source_add(sz.point.z,margin_source),flex_dd_neg(size3))<0) return;
  FlexDD span_x=flex_dd_source_mul(flex_dd(2),size0);
  FlexDD span_y=flex_dd_source_mul(flex_dd(2),size1);
  int cmin=max(0,common_ccd_grid_bound(flex_dd_source_mul(flex_dd_source_div(
      flex_dd_source_add(nx.point.x,size0),span_x),flex_dd(float(ncol-1))),false));
  int cmax=min(ncol-1,common_ccd_grid_bound(flex_dd_source_mul(flex_dd_source_div(
      flex_dd_source_add(sx.point.x,size0),span_x),flex_dd(float(ncol-1))),true));
  int rmin=max(0,common_ccd_grid_bound(flex_dd_source_mul(flex_dd_source_div(
      flex_dd_source_add(ny.point.y,size1),span_y),flex_dd(float(nrow-1))),false));
  int rmax=min(nrow-1,common_ccd_grid_bound(flex_dd_source_mul(flex_dd_source_div(
      flex_dd_source_add(sy.point.y,size1),span_y),flex_dd(float(nrow-1))),true));
  mesh.margin=margin_source;
  // mjc_ConvexHField forms grid spacing from mjtNum size values, not float
  // geom-size highs. Keep the complete compiled source size expansion and
  // round each operation in the same order as `(2.0*size)/(count-1)`.
  FlexDD dx=flex_dd_source_div(
      flex_dd_source_mul(flex_dd(2.0f),size0),flex_dd(float(ncol-1)));
  FlexDD dy=flex_dd_source_div(
      flex_dd_source_mul(flex_dd(2.0f),size1),flex_dd(float(nrow-1)));
  common_ccd_production_diag_context(
      diagnostic,world,pair,hfield_geom,mesh_geom,diagnostic_row,
      diagnostic_col,diagnostic_tri,nrow,ncol,ph,pm,qh,qm,Rh,Rm,
      size0,size1,size2,size3,dx,dy,margin_source,local_pos,
      local_mat,mesh);
  int ncontact=0, status=0;
  FlexDD tolerance=FlexDD{mesh_hull[tolerance_base],
                          mesh_hull[tolerance_base+1],
                          mesh_hull[tolerance_base+2]};
  for (int r=rmin;r<rmax;r++) {
    for (int c=cmin+1;c<=cmax;c++) {
      for (int tri=0;tri<2;tri++) {
        int rr[3], cc[3];
        if (tri==0) { rr[0]=r+1;cc[0]=c-1;rr[1]=r;cc[1]=c-1;rr[2]=r+1;cc[2]=c; }
        else { rr[0]=r;cc[0]=c-1;rr[1]=r+1;cc[1]=c;rr[2]=r;cc[2]=c; }
        CommonCCDSupportObject prism=common_ccd_production_object();
        prism.kind=COMMON_CCD_HFIELD_PRISM; prism.geom=-1; prism.geom_type=1;
        prism.vertex_offset=prism_vertex_offset; prism.vertex_count=6;
        FlexDD3 center=FlexDD3{flex_dd(0),flex_dd(0),flex_dd(0)};
        FlexDD top_z[3];
        FlexDD prism_lower[3],prism_upper[3];bool prism_bounds_valid=true;
        for (int v=0;v<6;v++) {
          int k=v%3;
          // Pinned engine_collision_convex.c:addPrismVert evaluates
          // `dx*c - size0` and `dy*(r+dr) - size1` as single C expressions.
          // The arm64 build contracts these to fnmsub; preserve that one
          // binary64 rounding boundary instead of rounding product then sum.
          FlexDD x=flex_dd_source_fma(
              dx,flex_dd(float(cc[k])),flex_dd_neg(size0));
          FlexDD y=flex_dd_source_fma(
              dy,flex_dd(float(rr[k])),flex_dd_neg(size1));
          FlexDD z;
          if (v<3) z=flex_dd_neg(size3);
          else z=flex_dd_source_add(
              flex_dd_source_mul(flex_dd(mesh_hull[data_base+rr[k]*ncol+cc[k]]),size2),
              margin_source);
          if (v>=3) top_z[k]=z;
          FlexDD point[3]={x,y,z};
          for(int axis=0;axis<3;axis++) {
            if(!common_axis_words_finite_012(point[axis]))prism_bounds_valid=false;
            if(v==0){prism_lower[axis]=point[axis];prism_upper[axis]=point[axis];}
            else {
              int lower_order=common_axis_short_compare_012(point[axis],prism_lower[axis]);
              int upper_order=common_axis_short_compare_012(point[axis],prism_upper[axis]);
              if(lower_order==2||upper_order==2)prism_bounds_valid=false;
              if(lower_order==-1)prism_lower[axis]=point[axis];
              if(upper_order==1)prism_upper[axis]=point[axis];
            }
          }
          FlexDD3 residual;
          for (int axis=0;axis<3;axis++) {
            float high=point[axis].hi;
            prism_vertices[3*v+axis]=high;
            FlexDD tail=FlexDD{point[axis].lo,point[axis].tail,0.0f};
            if (axis==0) residual.x=tail;
            else if (axis==1) residual.y=tail;
            else residual.z=tail;
          }
          prism.hfield_residual[v]=residual;
          center=mpr_add(center,FlexDD3{x,y,z});
        }
        bool below_prism=true;
        for (int v=3;v<6;v++) {
          if (flex_dd_compare(top_z[v-3],nz.point.z)>=0) below_prism=false;
        }
        if (below_prism) continue;
        if (common_ellipsoid_axis_short_interval_nonpenetration_012(mesh,prism_lower,prism_upper,prism_bounds_valid) ||
            common_cylinder_axis_short_interval_nonpenetration_012(mesh,prism_lower,prism_upper,prism_bounds_valid,margin_source) ||
            common_mesh_axis_short_interval_nonpenetration_012(mesh,mesh_hull,prism_lower,prism_upper,prism_bounds_valid,margin_source)) continue;
        FlexDD reciprocal_six=flex_dd_source_div(flex_dd(1.0f),flex_dd(6.0f));
        center=mpr_scale(center,reciprocal_six);
        ContactGeom contact;
        contact.dist=3.402823466e+38f; contact.pos=float3(0);
        contact.normal=float3(0); contact.t1=float3(0); contact.t2=float3(0);
        int2 face_ids[3];
        bool trace_prism=trace_query && r==diagnostic_row
            && c==diagnostic_col && tri==diagnostic_tri;
        device float* prism_diagnostic=trace_prism ? diagnostic : nullptr;
        device int2* vertex_id_base_ptr=reinterpret_cast<device int2*>(
            mesh_hull_info+mesh_hull_info[common_header+24]
            +2*(world*npairs+pair)*vertex_capacity);
        FlexDD3 retained_position,retained_normal;
        FlexDD retained_distance=flex_dd(0);
        const bool libccd=COMMON_CCD_USE_LIBCCD!=0;
        const bool shrunk_primitive=!libccd && (geom_type[mesh_geom]==2 || geom_type[mesh_geom]==3);
        int3 face_slots=int3(-1);
        int result;
        bool enclosure_certificate=false;
        if (libccd) {
          result=common_mpr_enclosure_penetration_012(
              prism,mesh,center,local_pos,flex_dd(0),iterations,tolerance,
              mesh_hull,mesh_hull_info,contact,retained_position,retained_normal,&retained_distance,
              epa_vertices,vertex_id_base_ptr,epa_faces,epa_face_map,epa_horizon,epa_stack,
              vertex_capacity,face_capacity,horizon_capacity,stack_capacity,&enclosure_certificate);
        } else if (shrunk_primitive) {
          // mjc_ccd shrinks spheres/capsules before prism penetration. The
          // terrain caller has zero output margin, while the convex object's
          // own margin still contributes to its source inflation radius.
          result=common_ccd_production_one_rigid_pair(
              prism,mesh,center,local_pos,flex_dd(0),iterations,tolerance,
              mesh_hull,mesh_hull_info,epa_vertices,vertex_id_base_ptr,
              epa_faces,epa_face_map,epa_horizon,epa_stack,
              vertex_capacity,face_capacity,horizon_capacity,stack_capacity,
              contact,face_ids,face_slots,&retained_position,&retained_normal);
        } else {
          result=common_ccd_production_one_prism(
              prism,mesh,center,local_pos,
              iterations,tolerance,mesh_hull,mesh_hull_info,
              epa_vertices,vertex_id_base_ptr,epa_faces,epa_face_map,epa_horizon,epa_stack,
              vertex_capacity,face_capacity,horizon_capacity,stack_capacity,contact,face_ids,
              prism_diagnostic,r,c,tri,
              &retained_position,&retained_normal,&retained_distance);
        }
        if (result<0) {
          status=-result;
          common_ccd_production_fail(world_status,world);
          break;
        }
        if (result>0) {
          // Contact is expressed in the hfield frame. Preserve the common
          // source witness, then map to world once at the production boundary.
          FlexDD3 world_pos=common_ccd_matrix_vector(Rh,retained_position);
          world_pos=mpr_add(world_pos,ph);
          contact.pos=flex_dd3_high(world_pos);
          FlexDD3 world_normal=common_ccd_matrix_vector(Rh,retained_normal);
          if (libccd && !enclosure_certificate) world_normal=common_libccd_fix_normal(
              prism,original,world_pos,world_normal);
          // Source mj_addContact normalizes the corrected world normal
          // before constructing the remaining axes. Preserve that source
          // normalization rather than normalizing its float32 projection.
          // Native prism EPA has the same retained witness contract as MPR.
          // Do not round either witness before world-frame normalization.
          bool retain_prism_precision=libccd || !shrunk_primitive;
          if(retain_prism_precision) world_normal=common_ccd_contact_normal_source_012(world_normal);
          if(!libccd && !shrunk_primitive) contact.dist=retained_distance.hi;
          contact.normal=flex_dd3_high(world_normal);
          // mjc_ConvexHField publishes only the world normal after
          // mjc_fixNormal. mj_addContact then builds a fresh world frame.
          // Rotating pre-correction prism tangents leaves them inconsistent
          // with the corrected normal, changing the friction Jacobian.
          contact.t1=float3(0.0f);
          make_frame(contact.normal,contact.t1,contact.t2);
          if(retain_prism_precision) {
            FlexDD3 seed=flex_dd3(float3(0,1,0));
            if(flex_dd_compare_exact(world_normal.y,flex_dd(.5f))>=0 ||
               flex_dd_compare_exact(world_normal.y,flex_dd(-.5f))<=0)
              seed=flex_dd3(float3(0,0,1));
            FlexDD3 tangent=mpr_normalize(mpr_sub(seed,mpr_scale(world_normal,mpr_dot(world_normal,seed))));
            contact.t1=flex_dd3_high(tangent);
            contact.t2=flex_dd3_high(mpr_cross(world_normal,tangent));
          }
          common_ccd_production_store_contact(mesh_hull,out,ncontact,contact);
          if(retain_prism_precision) {
            int precision=out+COMMON_CCD_CONTACT_PRECISION_BASE+14*ncontact;
            mesh_hull[precision]=retained_distance.lo;
            mesh_hull[precision+1]=retained_distance.tail;
            FlexDD point[3]={world_pos.x,world_pos.y,world_pos.z};
            FlexDD normal[3]={world_normal.x,world_normal.y,world_normal.z};
            for(int axis=0;axis<3;axis++) {
              float point_high[3]={contact.pos.x,contact.pos.y,contact.pos.z};
              float normal_high[3]={contact.normal.x,contact.normal.y,contact.normal.z};
              FlexDD published_point=common_ccd_publication_rebase_012(point[axis],point_high[axis]);
              FlexDD published_normal=common_ccd_publication_rebase_012(normal[axis],normal_high[axis]);
              mesh_hull[precision+2+axis]=published_point.lo;
              mesh_hull[precision+5+axis]=published_point.tail;
              mesh_hull[precision+8+axis]=published_normal.lo;
              mesh_hull[precision+11+axis]=published_normal.tail;
            }
          }
          // Keep the source-order terrain prism that produced each witness
          // in the three currently unused contact-record words. The canonical
          // row consumer reads only words 0:13; these tags support exact
          // CPU/native candidate-order diagnostics without changing physics.
          int witness=out+2+16*ncontact;
          mesh_hull[witness+13]=float(r);
          mesh_hull[witness+14]=float(c);
          mesh_hull[witness+15]=float(tri);
          ncontact++;
          if (ncontact>=min(pair_dims[10+pair+1]-pair_dims[10+pair],50)) break;
        }
      }
      if (status || ncontact>=50) break;
    }
    if (status || ncontact>=50) break;
  }
  mesh_hull[out]=float(ncontact);
  mesh_hull[out+1]=float(status);
}


// Source feature recovery for the face/face branch of mjc_ccd's native
// `multicontact`. EPA support ids use the pinned box bit encoding and mesh
// vertex indices, retained on every EPA vertex by this candidate's ABI.
static constant int COMMON_CCD_MAX_POLYVERT=150;
static constant float COMMON_CCD_FACE_TOL=0.99999872f;
static constant float COMMON_CCD_EDGE_TOL=0.00159999931f;
static constant float COMMON_CCD_MINVAL=1.0e-15f;

static inline int common_ccd_feature_dimension(int3 ids) {
  if (ids.x!=ids.y) return (ids.z==ids.x || ids.z==ids.y) ? 2 : 3;
  if (ids.x!=ids.z) return 2;
  return 1;
}

static inline float3 common_ccd_box_normal(
    thread const CommonCCDSupportObject& object, float3 world_direction,
    thread int& face_id) {
  // The explicit row-major multiplication mirrors boxNormals2 in the pinned
  // source. Feature selection only needs the represented direction.
  FlexDD3 dd_local=common_ccd_matrix_transpose_vector(object.mat,
                                                       flex_dd3(world_direction));
  float3 local=float3(dd_local.x.hi,dd_local.y.hi,dd_local.z.hi);
  float local_length=length(local);
  if (!(local_length>0.0f) || !isfinite(local_length)) { face_id=-1; return float3(0); }
  local/=local_length;
  const float3 axes[6]={float3(1,0,0),float3(-1,0,0),
                        float3(0,1,0),float3(0,-1,0),
                        float3(0,0,1),float3(0,0,-1)};
  for (int i=0;i<6;i++) {
    if (dot(local,axes[i])>COMMON_CCD_FACE_TOL) {
      face_id=i;
      FlexDD3 world=common_ccd_matrix_vector(object.mat,
                                             flex_dd3(axes[i]));
      return float3(world.x.hi,world.y.hi,world.z.hi);
    }
  }
  face_id=-1;
  return float3(0);
}

static inline int common_ccd_mesh_feature_face(
    int3 ids, int geom_record, int header,
    device const int* info, thread const CommonCCDSupportObject& object,
    thread int& face_id) {
  int dimension=common_ccd_feature_dimension(ids);
  if (dimension!=3) return 0;
  int vertadr=info[geom_record+7];
  int polyadr=info[geom_record+5];
  int mapadr_base=info[header+19], mapnum_base=info[header+20];
  int map_base=info[header+21];
  int map_address[3], map_count[3];
  int vertices[3]={ids.x,ids.y,ids.z};
  for (int i=0;i<3;i++) {
    if (vertices[i]<0 || vertices[i]>=object.vertex_count) return 0;
    int global_vertex=vertadr+vertices[i];
    map_address[i]=info[mapadr_base+global_vertex];
    map_count[i]=info[mapnum_base+global_vertex];
    if (map_address[i]<0 || map_count[i]<0) return 0;
  }
  int first=-1;
  for (int i=0;i<map_count[0];i++) {
    int face=info[map_base+map_address[0]+i];
    bool in_second=false, in_third=false;
    for (int j=0;j<map_count[1];j++)
      if (info[map_base+map_address[1]+j]==face) { in_second=true; break; }
    if (!in_second) continue;
    for (int j=0;j<map_count[2];j++)
      if (info[map_base+map_address[2]+j]==face) { in_third=true; break; }
    if (in_third) { first=face; break; }
  }
  if (first<0 || first>=info[geom_record+6]) return 0;
  face_id=first;
  return 1;
}

static inline FlexDD3 common_ccd_mesh_feature_normal_wide(
    int face_id, int geom_record, int header, device const int* info,
    device const float* hull, thread const CommonCCDSupportObject& object) {
  int normal_base=info[header+22];
  int polyadr=info[geom_record+5];
  int at=normal_base+3*(polyadr+face_id);
  int plane=3*info[header+23];
  FlexDD3 local=FlexDD3{
      FlexDD{hull[at],hull[at+plane],hull[at+2*plane]},
      FlexDD{hull[at+1],hull[at+plane+1],hull[at+2*plane+1]},
      FlexDD{hull[at+2],hull[at+plane+2],hull[at+2*plane+2]}};
  return common_ccd_matrix_vector(object.mat,local);
}

static inline float3 common_ccd_mesh_feature_normal(
    int face_id, int geom_record, int header, device const int* info,
    device const float* hull, thread const CommonCCDSupportObject& object) {
  return flex_dd3_high(common_ccd_mesh_feature_normal_wide(
      face_id,geom_record,header,info,hull,object));
}

static inline FlexDD3 common_ccd_feature_normal_wide(
    int face_id, int geom_record, int header, device const int* info,
    device const float* hull, thread const CommonCCDSupportObject& object) {
  if (object.geom_type==7)
    return common_ccd_mesh_feature_normal_wide(face_id,geom_record,header,info,hull,object);
  const float3 axes[6]={float3(1,0,0),float3(-1,0,0),float3(0,1,0),
                        float3(0,-1,0),float3(0,0,1),float3(0,0,-1)};
  return common_ccd_matrix_vector(object.mat,flex_dd3(axes[face_id]));
}

static inline void common_ccd_multicontact_witness(
    FlexDD3 point, FlexDD3 displacement, bool flip,
    thread ContactGeom& contact, thread FlexDD3* position=nullptr,
    thread FlexDD3* normal=nullptr) {
  // Pinned polygonClip reconstructs x1=x2-approx_dir; mjc_penetration
  // then subtracts the witnesses and normalizes. Retain source words until
  // after those operations. Subtracting rounded absolute world points can
  // corrupt a shallow contact normal even when both witnesses look accurate.
  FlexDD3 x1=flex_dd3_source_sub(point,displacement),x2=point;
  if (flip) {FlexDD3 tmp=x1;x1=x2;x2=tmp;}
  FlexDD midpoint_scale=flex_dd(.5f);
  FlexDD3 wide_position=FlexDD3{
      flex_dd_source_mul(flex_dd_source_add(x1.x,x2.x),midpoint_scale),
      flex_dd_source_mul(flex_dd_source_add(x1.y,x2.y),midpoint_scale),
      flex_dd_source_mul(flex_dd_source_add(x1.z,x2.z),midpoint_scale)};
  FlexDD3 delta=flex_dd3_source_sub(x1,x2);
  FlexDD norm=flex_dd_source_sqrt(flex_dd3_dot(delta,delta));
  const FlexDD minval=FlexDD{1e-15f,-3.627493647833322e-24f,0.0f};
  FlexDD3 wide_normal=flex_dd3(float3(1,0,0));
  if(flex_dd_compare_exact(norm,minval)>=0) {
    FlexDD inverse=flex_dd_source_div(flex_dd(1.0f),norm);
    wide_normal=FlexDD3{flex_dd_source_mul(delta.x,inverse),
                       flex_dd_source_mul(delta.y,inverse),
                       flex_dd_source_mul(delta.z,inverse)};
  }
  contact.pos=flex_dd3_high(wide_position);
  contact.normal=flex_dd3_high(wide_normal);
  contact.t1=float3(0);make_frame(contact.normal,contact.t1,contact.t2);
  if(position!=nullptr)*position=wide_position;
  if(normal!=nullptr)*normal=wide_normal;
}

static inline int common_ccd_face_polygon(
    thread const CommonCCDSupportObject& object, int geom_record,
    int face_id, int header, device const int* info,
    device const float* hull, thread float3* polygon) {
  if (object.geom_type==6) {
    float x=object.size[0].hi, y=object.size[1].hi, z=object.size[2].hi;
    float3 p[4];
    if (face_id==0) { p[0]=float3(x,y,z);p[1]=float3(x,y,-z);p[2]=float3(x,-y,-z);p[3]=float3(x,-y,z); }
    else if (face_id==1) { p[0]=float3(-x,y,-z);p[1]=float3(-x,y,z);p[2]=float3(-x,-y,z);p[3]=float3(-x,-y,-z); }
    else if (face_id==2) { p[0]=float3(-x,y,-z);p[1]=float3(x,y,-z);p[2]=float3(x,y,z);p[3]=float3(-x,y,z); }
    else if (face_id==3) { p[0]=float3(-x,-y,z);p[1]=float3(x,-y,z);p[2]=float3(x,-y,-z);p[3]=float3(-x,-y,-z); }
    else if (face_id==4) { p[0]=float3(-x,y,z);p[1]=float3(x,y,z);p[2]=float3(x,-y,z);p[3]=float3(-x,-y,z); }
    else if (face_id==5) { p[0]=float3(x,y,-z);p[1]=float3(-x,y,-z);p[2]=float3(-x,-y,-z);p[3]=float3(x,-y,-z); }
    else return 0;
    for (int i=0;i<4;i++) {
      FlexDD3 world=common_ccd_local_to_world(object.mat,object.pos,
                                               flex_dd3(p[i]));
      polygon[i]=float3(world.x.hi,world.y.hi,world.z.hi);
    }
    return 4;
  }
  if (object.geom_type!=7 || face_id<0 || face_id>=info[geom_record+6])
    return 0;
  int polyadr=info[geom_record+5];
  int polyvertadr_base=info[header+16], polyvertnum_base=info[header+17];
  int polyvert_base=info[header+18];
  int global_face=polyadr+face_id;
  int count=min(info[polyvertnum_base+global_face],COMMON_CCD_MAX_POLYVERT);
  int address=info[polyvertadr_base+global_face];
  if (count<3 || address<0) return 0;
  for (int i=0;i<count;i++) {
    int local_vertex=info[polyvert_base+address+(count-1-i)];
    if (local_vertex<0 || local_vertex>=object.vertex_count) return 0;
    int v=3*(object.vertex_offset+local_vertex);
    FlexDD3 world=common_ccd_local_to_world(
        object.mat,object.pos,
        flex_dd3(float3(hull[v],hull[v+1],hull[v+2])));
    polygon[i]=float3(world.x.hi,world.y.hi,world.z.hi);
  }
  return count;
}

static inline float common_ccd_plane_intersect(
    float3 normal, float pd, float3 a, float3 b, thread float3& output) {
  float3 ab=b-a;
  float denominator=dot(normal,ab);
  if (denominator==0.0f) return 3.402823466e+38f;
  float t=(pd-dot(normal,a))/denominator;
  if (t>=0.0f && t<=1.0f) output=a+t*ab;
  return t;
}

static inline float common_ccd_quad_area(float3 a,float3 b,float3 c,float3 d) {
  float3 e=cross(d-a,b-d), f=cross(c-b,a-c);
  return 0.5f*length(e+f);
}

static inline int common_ccd_clip_faces(
    thread const float3* face1, int n1, thread const float3* face2, int n2,
    float3 normal1, float3 approximate_direction,
    thread float3* output) {
  if (n1<3 || n2<1 || n1>COMMON_CCD_MAX_POLYVERT ||
      n2>COMMON_CCD_MAX_POLYVERT) return 0;
  float3 first[2*COMMON_CCD_MAX_POLYVERT];
  float3 second[2*COMMON_CCD_MAX_POLYVERT];
  for (int i=0;i<n2;i++) first[i]=face2[i];
  int count=n2;
  for (int edge=0;edge<n1;edge++) {
    float3 a=face1[edge], b=face1[(edge+1)%n1];
    float3 pn=cross(b-a,normal1);
    float pnlen=length(pn);
    if (!(pnlen>0.0f)) return 0;
    pn/=pnlen;
    float pd=dot(pn,a);
    int next_count=0;
    for (int i=0;i<count;i++) {
      float3 p=first[i], q=first[(i+1)%count];
      bool pin=dot(p-a,pn)>-COMMON_CCD_MINVAL;
      bool qin=dot(q-a,pn)>-COMMON_CCD_MINVAL;
      if (!pin && !qin) continue;
      if (pin && qin) {
        if (next_count>=2*COMMON_CCD_MAX_POLYVERT) return 0;
        second[next_count++]=q;
        continue;
      }
      float3 intersection;
      float t=common_ccd_plane_intersect(pn,pd,p,q,intersection);
      if (next_count>=2*COMMON_CCD_MAX_POLYVERT) return 0;
      if (t>=0.0f && t<=1.0f) second[next_count++]=intersection;
      if (qin) {
        if (next_count>=2*COMMON_CCD_MAX_POLYVERT) return 0;
        second[next_count++]=q;
      }
    }
    count=next_count;
    for (int i=0;i<count;i++) first[i]=second[i];
    if (count==0) return 0;
  }
  if (count>4) {
    int ai=0,bi=1,ci=2,di=3;
    float best=common_ccd_quad_area(first[ai],first[bi],first[ci],first[di]);
    for (int a=0;a<count;a++) {
      while (true) {
        int next_d=(di+1)%count;
        float candidate=common_ccd_quad_area(first[a],first[bi],first[ci],first[next_d]);
        if (candidate<=best) break;
        best=candidate;di=next_d;ai=a;
        while (true) {
          int next_c=(ci+1)%count;
          candidate=common_ccd_quad_area(first[a],first[bi],first[next_c],first[di]);
          if (candidate<=best) break;
          best=candidate;ci=next_c;ai=a;
        }
        while (true) {
          int next_b=(bi+1)%count;
          candidate=common_ccd_quad_area(first[a],first[next_b],first[ci],first[di]);
          if (candidate<=best) break;
          best=candidate;bi=next_b;ai=a;
        }
      }
      if (bi==a) {
        bi=(bi+1)%count;
        if (ci==bi) {
          ci=(ci+1)%count;
          if (di==ci) di=(di+1)%count;
        }
      }
    }
    output[0]=first[ai];output[1]=first[bi];
    output[2]=first[ci];output[3]=first[di];
    return 4;
  }
  if (n2==2 && count>2) {
    int best_a=0,best_b=1;
    float best_distance=0.0f;
    for (int i=0;i<count;i++) for (int j=i+1;j<count;j++) {
      float3 delta=first[j]-first[i];
      float distance=dot(delta,delta);
      if (distance>best_distance) { best_distance=distance;best_a=i;best_b=j; }
    }
    output[0]=first[best_a];output[1]=first[best_b];
    return 2;
  }
  for (int i=0;i<count;i++) output[i]=first[i];
  return count;
}

static inline int common_ccd_intersect_faces(
    thread int* output, thread const int* a, int na,
    thread const int* b, int nb) {
  int count=0;
  for (int i=0;i<na;i++) for (int j=0;j<nb;j++) {
    if (a[i]==b[j]) {
      if (count<2) output[count++]=a[i];
      if (count==2) return count;
    }
  }
  return count;
}

static inline int common_ccd_intersect_device_faces(
    thread int* output, device const int* a, int na,
    device const int* b, int nb) {
  int count=0;
  for (int i=0;i<na;i++) for (int j=0;j<nb;j++) {
    if (a[i]==b[j]) {
      if (count<2) output[count++]=a[i];
      if (count==2) return count;
    }
  }
  return count;
}

static inline int common_ccd_intersect_thread_device_faces(
    thread int* output, thread const int* a, int na,
    device const int* b, int nb) {
  int count=0;
  for (int i=0;i<na;i++) for (int j=0;j<nb;j++) {
    if (a[i]==b[j]) {
      if (count<2) output[count++]=a[i];
      if (count==2) return count;
    }
  }
  return count;
}

static inline int common_ccd_mesh_normals(
    int3 ids, int dimension, int geom_record, int header,
    device const int* info, device const float* hull,
    thread const CommonCCDSupportObject& object,
    thread float3* normals, thread int* face_ids) {
  int mapadr_base=info[header+19], mapnum_base=info[header+20];
  int map_base=info[header+21], polyadr=info[geom_record+5];
  int ids_v[3]={ids.x,ids.y,ids.z};
  int local_adr[3], local_num[3];
  int use=dimension==1?1:dimension==2?2:3;
  for (int i=0;i<use;i++) {
    if (ids_v[i]<0 || ids_v[i]>=object.vertex_count) return 0;
    int global_vertex=info[geom_record+7]+ids_v[i];
    local_adr[i]=info[mapadr_base+global_vertex];
    local_num[i]=info[mapnum_base+global_vertex];
    if (local_adr[i]<0 || local_num[i]<0) return 0;
  }
  if (dimension==3) {
    int common01[2];
    int n=common_ccd_intersect_device_faces(common01,
        info+map_base+local_adr[0],local_num[0],
        info+map_base+local_adr[1],local_num[1]);
    if (!n) return 0;
    int found[2];
    n=common_ccd_intersect_thread_device_faces(found,common01,n,
        info+map_base+local_adr[2],local_num[2]);
    if (!n) return 0;
    face_ids[0]=found[0];
    normals[0]=common_ccd_mesh_feature_normal(found[0],geom_record,header,
                                               info,hull,object);
    return 1;
  }
  if (dimension==2) {
    int found[2];
    int n=common_ccd_intersect_device_faces(found,
        info+map_base+local_adr[0],local_num[0],
        info+map_base+local_adr[1],local_num[1]);
    for (int i=0;i<n;i++) {
      face_ids[i]=found[i];
      normals[i]=common_ccd_mesh_feature_normal(found[i],geom_record,header,
                                                 info,hull,object);
    }
    return n;
  }
  int n=min(local_num[0],COMMON_CCD_MAX_POLYVERT);
  for (int i=0;i<n;i++) {
    int face=info[map_base+local_adr[0]+i];
    face_ids[i]=face;
    normals[i]=common_ccd_mesh_feature_normal(face,geom_record,header,
                                              info,hull,object);
  }
  return n;
}

static inline float3 common_ccd_box_local_axis(
    thread const CommonCCDSupportObject& object, float3 local) {
  FlexDD3 world=common_ccd_matrix_vector(object.mat,flex_dd3(local));
  return float3(world.x.hi,world.y.hi,world.z.hi);
}

static inline int common_ccd_box_normals(
    int3 ids, int dimension, thread const CommonCCDSupportObject& object,
    float3 direction, thread float3* normals, thread int* face_ids) {
  if (dimension==1) {
    float3 local[3]={float3((ids.x&1)?1:-1,0,0),
      float3(0,(ids.x&2)?1:-1,0),float3(0,0,(ids.x&4)?1:-1)};
    int face[3]={(ids.x&1)?0:1,(ids.x&2)?2:3,(ids.x&4)?4:5};
    for (int i=0;i<3;i++) { normals[i]=common_ccd_box_local_axis(object,local[i]);face_ids[i]=face[i]; }
    return 3;
  }
  int c=0;
  if (dimension==3) {
    int x=((ids.x&1)&&(ids.y&1)&&(ids.z&1))-(!(ids.x&1)&&!(ids.y&1)&&!(ids.z&1));
    int y=((ids.x&2)&&(ids.y&2)&&(ids.z&2))-(!(ids.x&2)&&!(ids.y&2)&&!(ids.z&2));
    int z=((ids.x&4)&&(ids.y&4)&&(ids.z&4))-(!(ids.x&4)&&!(ids.y&4)&&!(ids.z&4));
    if (x) { normals[c]=common_ccd_box_local_axis(object,float3(x,0,0));face_ids[c++]=x>0?0:1; }
    if (y) { normals[c]=common_ccd_box_local_axis(object,float3(0,y,0));face_ids[c++]=y>0?2:3; }
    if (z) { normals[c]=common_ccd_box_local_axis(object,float3(0,0,z));face_ids[c++]=z>0?4:5; }
    if (c==1) return 1;
  } else {
    int vals[3]={ids.x,ids.y,ids.z};
    int common[3]={1,2,4};
    for (int axis=0;axis<3;axis++) {
      int bit=common[axis],samepos=1,sameneg=1;
      for (int k=0;k<2;k++) { samepos &= (vals[k]&bit)!=0; sameneg &= (vals[k]&bit)==0; }
      if (samepos || sameneg) {
        float sign=samepos?1.0f:-1.0f;
        normals[c]=common_ccd_box_local_axis(object,axis==0?float3(sign,0,0):axis==1?float3(0,sign,0):float3(0,0,sign));
        face_ids[c++]=axis*2+(sign<0.0f);
      }
    }
    if (c==2) return 2;
  }
  FlexDD3 local_dd=common_ccd_matrix_transpose_vector(object.mat,flex_dd3(direction));
  float3 local=float3(local_dd.x.hi,local_dd.y.hi,local_dd.z.hi);
  float length_local=length(local);
  if (!(length_local>0.0f)) return 0;
  local/=length_local;
  const float3 axes[6]={float3(1,0,0),float3(-1,0,0),float3(0,1,0),float3(0,-1,0),float3(0,0,1),float3(0,0,-1)};
  for (int i=0;i<6;i++) if (dot(local,axes[i])>COMMON_CCD_FACE_TOL) {
    normals[0]=common_ccd_box_local_axis(object,axes[i]);face_ids[0]=i;return 1;
  }
  return 0;
}

static inline int common_ccd_box_edge_normals(
    int3 ids,int dimension,thread const CommonCCDSupportObject& object,
    float3 p1,float3 p2,thread float3* normals,thread float3* ends) {
  if (dimension==2) { ends[0]=p2; normals[0]=normalize(p2-p1); return 1; }
  if (dimension!=1) return 0;
  float3 size=float3(object.size[0].hi,object.size[1].hi,object.size[2].hi);
  float x=(ids.x&1)?size.x:-size.x,y=(ids.x&2)?size.y:-size.y,z=(ids.x&4)?size.z:-size.z;
  float3 local[3]={float3(-x,y,z),float3(x,-y,z),float3(x,y,-z)};
  FlexDD3 world[3]={common_ccd_local_to_world(object.mat,object.pos,flex_dd3(local[0])),
    common_ccd_local_to_world(object.mat,object.pos,flex_dd3(local[1])),
    common_ccd_local_to_world(object.mat,object.pos,flex_dd3(local[2]))};
  for (int i=0;i<3;i++) { ends[i]=float3(world[i].x.hi,world[i].y.hi,world[i].z.hi);normals[i]=normalize(ends[i]-p1); }
  return 3;
}

static inline int common_ccd_mesh_edge_normals(
    int3 ids,int dimension,int geom_record,int header,device const int* info,
    device const float* hull,thread const CommonCCDSupportObject& object,
    float3 p1,float3 p2,thread float3* normals,thread float3* ends) {
  if (dimension==2) { ends[0]=p2;normals[0]=normalize(p2-p1);return 1; }
  if (dimension!=1 || ids.x<0 || ids.x>=object.vertex_count) return 0;
  int adr_base=info[header+16],num_base=info[header+17],vert_base=info[header+18];
  int polyadr=info[geom_record+5],mapadr_base=info[header+19],mapnum_base=info[header+20],map_base=info[header+21];
  int global_v=info[geom_record+7]+ids.x;
  int mapadr=info[mapadr_base+global_v],mapnum=min(info[mapnum_base+global_v],COMMON_CCD_MAX_POLYVERT);
  if (mapadr<0 || mapnum<0) return 0;
  int out=0;
  for (int i=0;i<mapnum;i++) {
    int face=info[map_base+mapadr+i];
    int face_global=polyadr+face;
    int start=info[adr_base+face_global],count=min(info[num_base+face_global],COMMON_CCD_MAX_POLYVERT);
    if (start<0 || count<2) continue;
    for (int j=0;j<count;j++) if (info[vert_base+start+j]==ids.x) {
      int prev=(j==0)?count-1:j-1;
      int local=info[vert_base+start+prev];
      if (local<0 || local>=object.vertex_count) break;
      int v=3*(object.vertex_offset+local);
      FlexDD3 end=common_ccd_local_to_world(object.mat,object.pos,
          flex_dd3(float3(hull[v],hull[v+1],hull[v+2])));
      ends[out]=float3(end.x.hi,end.y.hi,end.z.hi);normals[out]=normalize(ends[out]-p1);out++;
      break;
    }
  }
  return out;
}

static inline int common_ccd_aligned_faces(
    thread const float3* a,int na,thread const float3* b,int nb,
    thread int2& selected) {
  for (int i=0;i<na;i++) for (int j=0;j<nb;j++)
    if (dot(a[i],b[j]) < -COMMON_CCD_FACE_TOL) { selected=int2(i,j);return 1; }
  return 0;
}

static inline int common_ccd_aligned_edge_face(
    thread const float3* edges,int ne,thread const float3* faces,int nf,
    thread int2& selected) {
  for (int i=0;i<nf;i++) for (int j=0;j<ne;j++)
    if (abs(dot(edges[j],faces[i]))<COMMON_CCD_EDGE_TOL) { selected=int2(j,i);return 1; }
  return 0;
}

static inline int common_ccd_multicontact_faces(
    int type_a, int type_b, int rec_a, int rec_b, int header,
    device const int* info, device const float* hull,
    thread const CommonCCDSupportObject& a,
    thread const CommonCCDSupportObject& b,
    thread const int2* support_ids, thread const int3& face_slots,
    device const FlexDDVertex* epa_vertices,
    float3 primary_normal, float penetration, thread ContactGeom* contacts) {
  if (!((type_a==6 || type_a==7) && (type_b==6 || type_b==7))) return 1;
  int3 ids_a=int3(support_ids[0].x,support_ids[1].x,support_ids[2].x);
  int3 ids_b=int3(support_ids[0].y,support_ids[1].y,support_ids[2].y);
  int dim_a=common_ccd_feature_dimension(ids_a);
  int dim_b=common_ccd_feature_dimension(ids_b);
  // The EPA face itself carries source-ordered witnesses. They are the
  // endpoints needed when either support feature is an edge or vertex.
  FlexDDVertex va=epa_vertices[face_slots.x];
  FlexDDVertex vb=epa_vertices[face_slots.y];
  FlexDDVertex vc=epa_vertices[face_slots.z];
  float3 a0=float3(va.a.x.hi,va.a.y.hi,va.a.z.hi);
  float3 a1=float3(vb.a.x.hi,vb.a.y.hi,vb.a.z.hi);
  float3 a2=float3(vc.a.x.hi,vc.a.y.hi,vc.a.z.hi);
  float3 b0=float3(va.b.x.hi,va.b.y.hi,va.b.z.hi);
  float3 b1=float3(vb.b.x.hi,vb.b.y.hi,vb.b.z.hi);
  float3 b2=float3(vc.b.x.hi,vc.b.y.hi,vc.b.z.hi);
  // Match simplexDim's in-place ordering when vertices 0 and 1 coincide;
  // its second endpoint then becomes source simplex vertex 2.
  if (ids_a.x==ids_a.y && ids_a.x!=ids_a.z) { ids_a.y=ids_a.z;a1=a2; }
  if (ids_b.x==ids_b.y && ids_b.x!=ids_b.z) { ids_b.y=ids_b.z;b1=b2; }
  // EPA's source status displacement is x2-x1; its norm is the penetration
  // depth and its orientation is opposite contact.normal. The raw support
  // vertices are only used as feature endpoints, never as this displacement.
  float3 dir=-primary_normal*penetration;
  float direction_length=length(dir);
  if (!(direction_length>0.0f)) return 1;
  float3 normals_a[COMMON_CCD_MAX_POLYVERT];
  float3 normals_b[COMMON_CCD_MAX_POLYVERT];
  float3 edge_ends[COMMON_CCD_MAX_POLYVERT];
  float3 edge_normals[COMMON_CCD_MAX_POLYVERT];
  int face_ids_a[COMMON_CCD_MAX_POLYVERT];
  int face_ids_b[COMMON_CCD_MAX_POLYVERT];
  int nn_a=type_a==6 ? common_ccd_box_normals(ids_a,dim_a,a,-dir,normals_a,face_ids_a) :
      common_ccd_mesh_normals(ids_a,dim_a,rec_a,header,info,hull,a,normals_a,face_ids_a);
  int nn_b=type_b==6 ? common_ccd_box_normals(ids_b,dim_b,b,dir,normals_b,face_ids_b) :
      common_ccd_mesh_normals(ids_b,dim_b,rec_b,header,info,hull,b,normals_b,face_ids_b);
  int2 selected=int2(-1);
  bool edge_a=false,edge_b=false;
  if (!common_ccd_aligned_faces(normals_a,nn_a,normals_b,nn_b,selected)) {
    if (dim_a<3 && dim_a<=dim_b) {
      int edge_count=type_a==6 ? common_ccd_box_edge_normals(ids_a,dim_a,a,a0,a1,edge_normals,edge_ends) :
          common_ccd_mesh_edge_normals(ids_a,dim_a,rec_a,header,info,hull,a,a0,a1,edge_normals,edge_ends);
      if (!common_ccd_aligned_edge_face(edge_normals,edge_count,normals_b,nn_b,selected)) return 1;
      edge_a=true;
      normals_a[0]=edge_normals[selected.x];
      edge_ends[0]=edge_ends[selected.x];
    } else if (dim_b<3) {
      int edge_count=type_b==6 ? common_ccd_box_edge_normals(ids_b,dim_b,b,b0,b1,edge_normals,edge_ends) :
          common_ccd_mesh_edge_normals(ids_b,dim_b,rec_b,header,info,hull,b,b0,b1,edge_normals,edge_ends);
      if (!common_ccd_aligned_edge_face(edge_normals,edge_count,normals_a,nn_a,selected)) return 1;
      edge_b=true;
      normals_b[0]=edge_normals[selected.x];
      edge_ends[0]=edge_ends[selected.x];
    } else {
      return 1;
    }
  }

  float3 polygon_a[COMMON_CCD_MAX_POLYVERT];
  float3 polygon_b[COMMON_CCD_MAX_POLYVERT];
  float3 clipped[2*COMMON_CCD_MAX_POLYVERT];
  int na,nb;
  if (edge_a) {
    polygon_a[0]=a0;polygon_a[1]=edge_ends[0];na=2;
  } else if (type_a==6) {
    na=common_ccd_face_polygon(a,rec_a,face_ids_a[edge_b?selected.y:selected.x],header,info,hull,polygon_a);
  } else {
    na=common_ccd_face_polygon(a,rec_a,face_ids_a[edge_b?selected.y:selected.x],header,info,hull,polygon_a);
  }
  if (edge_b) {
    polygon_b[0]=b0;polygon_b[1]=edge_ends[0];nb=2;
  } else {
    // `alignedFaces` returns i into geom1's feature list and j into geom2's;
    // the pinned `multicontact` always recovers geom2 with idx2[j].
    nb=common_ccd_face_polygon(b,rec_b,face_ids_b[selected.y],header,info,hull,polygon_b);
  }
  if (na<2 || nb<2) return 1;
  int count=0;
  bool flip=false;
  if (edge_a) {
    // Pinned multicontact clips the edge against the opposing face, then
    // swaps witnesses because the polygonClip arguments are reversed.
    float3 normal=normals_b[selected.y];
    count=common_ccd_clip_faces(polygon_b,nb,polygon_a,na,normal,
                                -normal*direction_length,clipped);
    flip=true;
  } else if (edge_b) {
    float3 normal=normals_a[selected.y];
    count=common_ccd_clip_faces(polygon_a,na,polygon_b,nb,normal,
                                -normal*direction_length,clipped);
  } else {
    float3 normal1=normals_a[selected.x],normal2=normals_b[selected.y];
    count=common_ccd_clip_faces(polygon_a,na,polygon_b,nb,normal1,
                                normal2*direction_length,clipped);
  }
  if (count<1) return 1;
  FlexDD3 selected_normal=edge_b ?
      common_ccd_feature_normal_wide(face_ids_a[selected.y],rec_a,header,info,hull,a) :
      common_ccd_feature_normal_wide(face_ids_b[selected.y],rec_b,header,info,hull,b);
  FlexDD3 displacement=flex_dd3_scale(selected_normal,
      flex_dd((edge_a || edge_b) ? -direction_length : direction_length));
  for (int i=0;i<count;i++) {
    contacts[i]=contacts[0];
    common_ccd_multicontact_witness(flex_dd3(clipped[i]),displacement,flip,contacts[i]);
  }
  return min(count,4);
}

// Rigid pairs emit at most five witnesses, leaving the remainder of the
// shared 50-contact output record available for source-precision restart
// state. These words are never interpreted as published contacts.
#define COMMON_CCD_RIGID_WIDE_BASE 82
#define COMMON_CCD_RIGID_NORMAL_BASE 127
#define COMMON_CCD_RIGID_CACHE_BASE 136
static inline void common_ccd_store_wide3(device float* values,int base,FlexDD3 value) {
  values[base]=value.x.hi;values[base+1]=value.x.lo;values[base+2]=value.x.tail;
  values[base+3]=value.y.hi;values[base+4]=value.y.lo;values[base+5]=value.y.tail;
  values[base+6]=value.z.hi;values[base+7]=value.z.lo;values[base+8]=value.z.tail;
}
static inline FlexDD3 common_ccd_load_wide3(device const float* values,int base) {
  return FlexDD3{FlexDD{values[base],values[base+1],values[base+2]},
                 FlexDD{values[base+3],values[base+4],values[base+5]},
                 FlexDD{values[base+6],values[base+7],values[base+8]}};
}

// Pinned mjc_Convex's restart frame. Keep the initial witness in source
// precision: rounding its origin or normal before rotation changes GJK ties.
static inline FlexDD3 common_ccd_restart_normalize(FlexDD3 value) {
  FlexDD length=flex_dd_source_sqrt(flex_dd3_dot(value,value));
  FlexDD reciprocal=flex_dd_source_div(flex_dd(1.0f),length);
  return mpr_scale(value,reciprocal);
}

static inline void common_ccd_restart_axes(
    FlexDD3 normal, thread FlexDD3* axes) {
  FlexDD3 x=common_ccd_restart_normalize(normal);
  FlexDD3 y=(flex_dd_compare_exact(x.y,flex_dd(-0.5f))>0 &&
              flex_dd_compare_exact(x.y,flex_dd(0.5f))<0)
      ? flex_dd3(float3(0,1,0)) : flex_dd3(float3(0,0,1));
  y=flex_dd3_source_sub(y,mpr_scale(x,flex_dd3_dot(x,y)));
  axes[0]=common_ccd_restart_normalize(y);
  axes[1]=mpr_cross(x,axes[0]);
}

static inline void common_ccd_restart_matrix(
    FlexDD3 axis, bool positive, thread FlexDD* matrix) {
  // Binary64 sin/cos of +/- 0.001/2, split into three float words.
  FlexDD sine=FlexDD{0.00049999996554106474f,
                    1.3625602382494328e-11f,-2.1684043449710089e-19f};
  if (!positive) sine=flex_dd_neg(sine);
  FlexDD cosine=FlexDD{0.99999988079071045f,
                      -5.7907079131780392e-09f,1.1102230246251565e-16f};
  FlexDD q[4]={cosine,flex_dd_source_mul(axis.x,sine),
              flex_dd_source_mul(axis.y,sine),flex_dd_source_mul(axis.z,sine)};
  FlexDD q00=flex_dd_source_mul(q[0],q[0]);
  FlexDD q01=flex_dd_source_mul(q[0],q[1]);
  FlexDD q02=flex_dd_source_mul(q[0],q[2]);
  FlexDD q03=flex_dd_source_mul(q[0],q[3]);
  FlexDD q11=flex_dd_source_mul(q[1],q[1]);
  FlexDD q12=flex_dd_source_mul(q[1],q[2]);
  FlexDD q13=flex_dd_source_mul(q[1],q[3]);
  FlexDD q22=flex_dd_source_mul(q[2],q[2]);
  FlexDD q23=flex_dd_source_mul(q[2],q[3]);
  FlexDD q33=flex_dd_source_mul(q[3],q[3]);
  FlexDD two=flex_dd(2.0f);
  matrix[0]=flex_dd_source_sub(flex_dd_source_sub(flex_dd_source_add(q00,q11),q22),q33);
  matrix[1]=flex_dd_source_mul(two,flex_dd_source_sub(q12,q03));
  matrix[2]=flex_dd_source_mul(two,flex_dd_source_add(q13,q02));
  matrix[3]=flex_dd_source_mul(two,flex_dd_source_add(q12,q03));
  matrix[4]=flex_dd_source_sub(flex_dd_source_add(flex_dd_source_sub(q00,q11),q22),q33);
  matrix[5]=flex_dd_source_mul(two,flex_dd_source_sub(q23,q01));
  matrix[6]=flex_dd_source_mul(two,flex_dd_source_sub(q13,q02));
  matrix[7]=flex_dd_source_mul(two,flex_dd_source_add(q23,q01));
  matrix[8]=flex_dd_source_add(flex_dd_source_sub(flex_dd_source_sub(q00,q11),q22),q33);
}

static inline CommonCCDSupportObject common_ccd_restart_object(
    thread const CommonCCDSupportObject& original, FlexDD3 origin,
    thread const FlexDD* rotation) {
  CommonCCDSupportObject result=original;
  for (int column=0;column<3;column++) {
    FlexDD3 value=common_ccd_matrix_vector(rotation,
        FlexDD3{original.mat[column],original.mat[3+column],original.mat[6+column]});
    result.mat[column]=value.x;result.mat[3+column]=value.y;result.mat[6+column]=value.z;
  }
  FlexDD3 oldpos=FlexDD3{original.pos[0],original.pos[1],original.pos[2]};
  FlexDD3 relative=flex_dd3_source_sub(origin,oldpos);
  FlexDD3 displacement=flex_dd3_source_sub(common_ccd_matrix_vector(rotation,relative),relative);
  FlexDD3 position=flex_dd3_source_sub(oldpos,displacement);
  result.pos[0]=position.x;result.pos[1]=position.y;result.pos[2]=position.z;
  return result;
}

// Production source-GJK/EPA route for every pinned mjc_Convex dispatch entry.
// Single-pass box/mesh manifolds and rotated convex restart witnesses are
// consumed by contact_normal before row/Jacobian assembly. Dedicated plane/HField/SDF and analytic
// callbacks do not enter this kernel.
kernel void common_ccd_rigid_convex_candidates(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_quat [[buffer(1)]],
    device const float* geom_pos_low [[buffer(2)]],
    device const float* geom_pos_tail [[buffer(3)]],
    device const float* geom_xmat [[buffer(4)]],
    device const float* geom_xmat_low [[buffer(5)]],
    device const float* geom_xmat_tail [[buffer(6)]],
    device const float* geom_size [[buffer(7)]],
    device const int* geom_type [[buffer(8)]],
    device const int* pair_geoms [[buffer(9)]],
    device const float* pair_margin_gap [[buffer(10)]],
    device const int* pair_dims [[buffer(11)]],
    device const int* logical_pair_to_packed [[buffer(12)]],
    device float* mesh_hull [[buffer(13)]],
    device int* mesh_hull_info [[buffer(14)]],
    device atomic_int* world_status [[buffer(15)]],
    constant long& restart_pass [[buffer(16)]],
    device const float* pair_margin_source [[buffer(17)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=pair_dims[3], npairs=pair_dims[1], ngeom=pair_dims[6];
  if (npairs<=0 || uint(tid)>=uint(batch*npairs)) return;
  int world=int(tid)/npairs, pair=int(tid)%npairs;
  int a=pair_geoms[2*pair], b=pair_geoms[2*pair+1];
  if (!common_ccd_production_uses_mjc_convex(geom_type[a],geom_type[b])) return;
  int mask_pointer=10+npairs+1+pair_dims[2];
  int mask_offset=pair_dims[mask_pointer];
  if (mask_offset<0 || pair_dims[mask_offset+world]==0) return;
  if (logical_pair_to_packed[world*npairs+pair]<0) return;
  int header=9*ngeom;
  if (mesh_hull_info[header+0]!=COMMON_CCD_PRODUCTION_MAGIC) {
    common_ccd_production_fail(world_status,world); return;
  }
  int geom_records=mesh_hull_info[header+1];
  int graph_base=mesh_hull_info[header+2];
  int output_base=mesh_hull_info[header+5];
  int output_stride=mesh_hull_info[header+6];
  int scratch_base=mesh_hull_info[header+7];
  int scratch_stride=mesh_hull_info[header+8];
  int iterations=mesh_hull_info[header+9];
  int tolerance_base=mesh_hull_info[header+10];
  int vertex_capacity=mesh_hull_info[header+11];
  int face_capacity=mesh_hull_info[header+12];
  int horizon_capacity=mesh_hull_info[header+13];
  int stack_capacity=mesh_hull_info[header+14];
  if (geom_records<0 || graph_base<0 || output_base<0 || output_stride<COMMON_CCD_OUTPUT_WORDS ||
      scratch_base<0 || scratch_stride<=0 || iterations<=0 ||
      tolerance_base<0 || vertex_capacity<6 || face_capacity<6 ||
      horizon_capacity<3 || stack_capacity<1) {
    common_ccd_production_fail(world_status,world); return;
  }
  int query=world*npairs+pair;
  int out=output_base+query*output_stride;
  int phase=int(restart_pass);
  if (phase<0 || phase>4) { common_ccd_production_fail(world_status,world);return; }
  if (phase==0) { mesh_hull[out]=0.0f;mesh_hull[out+1]=0.0f; }
  else if (mesh_hull[out]<1.0f || mesh_hull[out+1]!=0.0f) return;
  int scratch=scratch_base+query*scratch_stride;
  int go=world*ngeom;
  FlexDD3 pa=common_ccd_load_geom_position(geom_pos,geom_pos_low,
      geom_pos_tail,3*(go+a));
  FlexDD3 pb=common_ccd_load_geom_position(geom_pos,geom_pos_low,
      geom_pos_tail,3*(go+b));
  float4 qa=float4(geom_quat[4*(go+a)],geom_quat[4*(go+a)+1],
                   geom_quat[4*(go+a)+2],geom_quat[4*(go+a)+3]);
  float4 qb=float4(geom_quat[4*(go+b)],geom_quat[4*(go+b)+1],
                   geom_quat[4*(go+b)+2],geom_quat[4*(go+b)+3]);
  FlexDD ma[9], mb[9];
  common_ccd_load_geom_matrix(geom_xmat,geom_xmat_low,geom_xmat_tail,
      9*(go+a),ma);
  common_ccd_load_geom_matrix(geom_xmat,geom_xmat_low,geom_xmat_tail,
      9*(go+b),mb);
  float3 sza=float3(geom_size[3*a],geom_size[3*a+1],geom_size[3*a+2]);
  float3 szb=float3(geom_size[3*b],geom_size[3*b+1],geom_size[3*b+2]);
  if (!common_ccd_finite_pose(pa,ma) || !common_ccd_finite_pose(pb,mb) ||
      !all(isfinite(qa)) || !all(isfinite(qb)) ||
      !all(isfinite(sza)) || !all(isfinite(szb))) {
    mesh_hull[out+1]=4.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  int rec_a=geom_records+COMMON_CCD_GEOM_INFO_WORDS*a;
  int rec_b=geom_records+COMMON_CCD_GEOM_INFO_WORDS*b;
  float margin_value=pair_margin_gap[2*pair]+pair_margin_gap[2*pair+1];
  FlexDD source_margin=FlexDD{pair_margin_source[pair],
      pair_margin_source[npairs+pair],pair_margin_source[2*npairs+pair]};
  // mj_geomDistance passes distmax directly as the collision margin. A
  // negative finite bound is valid and must reach libccd's support mapping
  // and depth correction unchanged, rather than becoming an empty query.
  if (!isfinite(margin_value) || !isfinite(source_margin.hi) ||
      !isfinite(source_margin.lo) || !isfinite(source_margin.tail)) {
    mesh_hull[out+1]=4.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  CommonCCDSupportObject object_a=common_ccd_production_geom(
      a,geom_type[a],rec_a,mesh_hull_info,mesh_hull,pa,ma,
      sza,margin_value);
  CommonCCDSupportObject object_b=common_ccd_production_geom(
      b,geom_type[b],rec_b,mesh_hull_info,mesh_hull,pb,mb,
      szb,margin_value);
  // Preserve the compiled collision margin in the support mapping and
  // depth correction. Its float32 projection can flip EPA face ties and
  // rotated restart normals even when positions agree to nanometers.
  object_a.margin=source_margin;
  object_b.margin=source_margin;
  if (object_a.kind<0 || object_b.kind<0) {
    mesh_hull[out+1]=3.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  if ((geom_type[a]==7 && (object_a.vertex_offset<0 || object_a.vertex_count<=0)) ||
      (geom_type[b]==7 && (object_b.vertex_offset<0 || object_b.vertex_count<=0))) {
    mesh_hull[out+1]=3.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  int source_size_base=mesh_hull_info[header+34];
  if (source_size_base<0) {
    mesh_hull[out+1]=3.0f;common_ccd_production_fail(world_status,world);return;
  }
  for (int coordinate=0;coordinate<3;coordinate++) {
    int ia=source_size_base+9*a+3*coordinate;
    int ib=source_size_base+9*b+3*coordinate;
    object_a.size[coordinate]=FlexDD{mesh_hull[ia],mesh_hull[ia+1],mesh_hull[ia+2]};
    object_b.size[coordinate]=FlexDD{mesh_hull[ib],mesh_hull[ib+1],mesh_hull[ib+2]};
  }
  int vertex_words=29*vertex_capacity;
  int face_words=20*face_capacity;
  int map_words=face_capacity;
  int horizon_words=2*horizon_capacity;
  int horizon_alignment=vertex_words&1;
  device FlexDDVertex* epa_vertices=reinterpret_cast<device FlexDDVertex*>(
      mesh_hull+scratch);
  // The descriptor addresses int32 words, while the typed arena stores
  // int2 support pairs. Each query therefore owns two words per vertex.
  int idbase=mesh_hull_info[header+24]+2*query*vertex_capacity;
  device int2* epa_vertex_ids=reinterpret_cast<device int2*>(mesh_hull_info+idbase);
  device FlexEpaFace* epa_faces=reinterpret_cast<device FlexEpaFace*>(
      mesh_hull+scratch+vertex_words);
  device int* epa_face_map=reinterpret_cast<device int*>(
      mesh_hull+scratch+vertex_words+face_words);
  device int2* epa_horizon=reinterpret_cast<device int2*>(
      mesh_hull+scratch+vertex_words+face_words+map_words+horizon_alignment);
  device FlexHorizonFrame* epa_stack=reinterpret_cast<device FlexHorizonFrame*>(
      mesh_hull+scratch+vertex_words+face_words+map_words+horizon_alignment+
      horizon_words);
  FlexDD tolerance=FlexDD{mesh_hull[tolerance_base],
                          mesh_hull[tolerance_base+1],
                          mesh_hull[tolerance_base+2]};
  // Immutable model options specialize this pipeline before compilation.
  // Keeping both heavy algorithms live in one runtime branch needlessly
  // enlarges the compiled kernel; this selection changes no numerical inputs.
  const bool libccd=COMMON_CCD_USE_LIBCCD!=0;
  bool source_multicontact=!libccd &&
      flex_dd_compare_exact(source_margin,flex_dd(0.0f))==0 && pair_dims[8]==0 &&
      (geom_type[a]==6 || geom_type[a]==7) && (geom_type[b]==6 || geom_type[b]==7);
  if (phase>0) {
    object_a.cached_vertex=int(mesh_hull[out+COMMON_CCD_RIGID_CACHE_BASE]);
    object_b.cached_vertex=int(mesh_hull[out+COMMON_CCD_RIGID_CACHE_BASE+1]);
  }
  if (phase>0) {
    if (source_multicontact || pair_dims[8]!=0 ||
        geom_type[a]==2 || geom_type[a]==4 || geom_type[b]==2 || geom_type[b]==4) return;
    FlexDD3 origin=common_ccd_load_wide3(mesh_hull,out+COMMON_CCD_RIGID_WIDE_BASE);
    FlexDD3 initial_normal=common_ccd_load_wide3(mesh_hull,out+COMMON_CCD_RIGID_NORMAL_BASE);
    FlexDD3 axes[2];common_ccd_restart_axes(initial_normal,axes);
    FlexDD rotation[9],inverse[9];
    common_ccd_restart_matrix(axes[(phase-1)/2],((phase-1)%2)==1,rotation);
    for (int row=0;row<3;row++) for (int column=0;column<3;column++)
      inverse[3*row+column]=rotation[3*column+row];
    object_a=common_ccd_restart_object(object_a,origin,rotation);
    object_b=common_ccd_restart_object(object_b,origin,inverse);
  }
  FlexDD3 center_a=FlexDD3{object_a.pos[0],object_a.pos[1],object_a.pos[2]};
  FlexDD3 center_b=FlexDD3{object_b.pos[0],object_b.pos[1],object_b.pos[2]};
  ContactGeom contact;
  contact.dist=3.402823466e+38f; contact.pos=float3(0);
  contact.normal=float3(0); contact.t1=float3(0); contact.t2=float3(0);
  FlexDD3 contact_position,contact_normal;
  int2 face_ids[3];
  int3 face_slots=int3(-1);
  int found=libccd ? common_libccd_penetration(
      object_a,object_b,center_a,center_b,
      source_margin,iterations,tolerance,mesh_hull,mesh_hull_info,
      contact,contact_position,contact_normal) : common_ccd_production_one_rigid_pair(
      object_a,object_b,center_a,center_b,
      source_margin,iterations,tolerance,
      mesh_hull,mesh_hull_info,epa_vertices,epa_vertex_ids,epa_faces,epa_face_map,
      epa_horizon,epa_stack,vertex_capacity,face_capacity,horizon_capacity,
      stack_capacity,contact,face_ids,face_slots,&contact_position,&contact_normal);
  // mjc_Convex reuses each object's support hint across all four rotated
  // searches, including queries that do not add a distinct contact. These
  // two owned words are outside the five published contact records.
  mesh_hull[out+COMMON_CCD_RIGID_CACHE_BASE]=float(object_a.cached_vertex);
  mesh_hull[out+COMMON_CCD_RIGID_CACHE_BASE+1]=float(object_b.cached_vertex);
  if (found<0) {
    mesh_hull[out+1]=float(-found);
    common_ccd_production_fail(world_status,world); return;
  }
  if (found<=0) return;
  if (libccd) {
    // Pinned mjc_fixNormal uses the original world geometry even on rotated
    // restart queries. Construct those original poses before refining normal.
    CommonCCDSupportObject original_a=object_a,original_b=object_b;
    original_a.pos[0]=pa.x;original_a.pos[1]=pa.y;original_a.pos[2]=pa.z;
    original_b.pos[0]=pb.x;original_b.pos[1]=pb.y;original_b.pos[2]=pb.z;
    for (int i=0;i<9;i++) {original_a.mat[i]=ma[i];original_b.mat[i]=mb[i];}
    contact_normal=common_libccd_fix_normal(original_a,original_b,contact_position,contact_normal);
    contact.normal=flex_dd3_high(contact_normal);
    contact.t1=float3(0);make_frame(contact.normal,contact.t1,contact.t2);
  }
  if (phase==0) {
    ContactGeom contacts[5];contacts[0]=contact;
    int contact_count=1;
    if (source_multicontact && face_slots.x>=0 && face_slots.y>=0 && face_slots.z>=0) {
      contact_count=common_ccd_multicontact_faces(
          geom_type[a],geom_type[b],rec_a,rec_b,header,mesh_hull_info,
          mesh_hull,object_a,object_b,face_ids,face_slots,epa_vertices,contact.normal,
          max(0.0f,-contact.dist),contacts);
    }
    for (int slot=0;slot<contact_count;slot++)
      common_ccd_production_store_contact(mesh_hull,out,slot,contacts[slot]);
    common_ccd_store_wide3(mesh_hull,out+COMMON_CCD_RIGID_WIDE_BASE,contact_position);
    common_ccd_store_wide3(mesh_hull,out+COMMON_CCD_RIGID_NORMAL_BASE,contact_normal);
    mesh_hull[out]=float(contact_count);
    return;
  }
  int count=int(mesh_hull[out]);
  if (count<1 || count>=5) {
    mesh_hull[out+1]=3.0f;common_ccd_production_fail(world_status,world);return;
  }
  int bound_base=mesh_hull_info[header+33];
  if (bound_base<0) {
    mesh_hull[out+1]=3.0f;common_ccd_production_fail(world_status,world);return;
  }
  FlexDD bound_a=FlexDD{mesh_hull[bound_base+3*a],mesh_hull[bound_base+3*a+1],mesh_hull[bound_base+3*a+2]};
  FlexDD bound_b=FlexDD{mesh_hull[bound_base+3*b],mesh_hull[bound_base+3*b+1],mesh_hull[bound_base+3*b+2]};
  FlexDD relative=FlexDD{0.0010000000474974513f,-4.7497450417210985e-11f,-8.6736173798840355e-19f};
  FlexDD distinct_tolerance=flex_dd_source_mul(relative,
      flex_dd_compare_exact(bound_a,bound_b)<0?bound_a:bound_b);
  for (int previous=0;previous<count;previous++) {
    FlexDD3 delta=flex_dd3_source_sub(
        common_ccd_load_wide3(mesh_hull,out+COMMON_CCD_RIGID_WIDE_BASE+9*previous),contact_position);
    if (flex_dd_compare_exact(flex_dd_source_sqrt(flex_dd3_dot(delta,delta)),distinct_tolerance)<=0) return;
  }
  // Every extra point has the primary physical penetration depth; its
  // position and normal remain those found by the rotated source query.
  contact.dist=mesh_hull[out+2];
  common_ccd_production_store_contact(mesh_hull,out,count,contact);
  common_ccd_store_wide3(mesh_hull,out+COMMON_CCD_RIGID_WIDE_BASE+9*count,contact_position);
  mesh_hull[out]=float(count+1);
}

// Pinned engine_collision_convex.c::mjc_PlaneConvex and addplanemesh.
// Depths and unprojected solid witnesses retain source arithmetic. The first
// contact position is the source separation reference for ordered neighbors.
static inline int common_ccd_source_plane_convex(
    thread CommonCCDSupportObject& plane,thread CommonCCDSupportObject& convex,
    FlexDD margin,FlexDD rbound,device float* hull,device int* info,
    thread FlexDD* depths,thread FlexDD3* points,thread FlexDD3& normal) {
  normal=FlexDD3{plane.mat[2],plane.mat[5],plane.mat[8]};
  FlexDD3 origin=FlexDD3{plane.pos[0],plane.pos[1],plane.pos[2]};
  CommonCCDSupportResult support=mpr_object_support(convex,flex_dd3_neg(normal),hull,info);
  if(support.status) return -support.status;
  depths[0]=flex_dd_source_dot3(normal,flex_dd3_source_sub(support.point,origin));
  if(flex_dd_compare_exact(depths[0],margin)>0) return 0;
  points[0]=support.point;
  FlexDD3 first=flex_dd3_sub(points[0],flex_dd3_scale(normal,
      flex_dd_source_mul(flex_dd(.5f),depths[0])));
  if(convex.geom_type!=7) return 1;
  FlexDD3 pos=FlexDD3{convex.pos[0],convex.pos[1],convex.pos[2]};
  FlexDD3 direction=common_ccd_matrix_transpose_vector(convex.mat,flex_dd3_neg(normal));
  FlexDD threshold=flex_dd_source_sub(flex_dd_source_dot3(normal,
      flex_dd3_source_sub(pos,origin)),margin);
  int count=1,edge=0,graph=convex.graph_offset,n=convex.graph_vertex_count;
  if(graph>=0) {
    if(convex.cached_vertex<0) return count;
    edge=info[graph+2+convex.cached_vertex];
  }
  for(int visit=0;visit<convex.vertex_count && count<3;visit++) {
    int candidate_vertex=visit;
    if(graph>=0) {int local=info[graph+2+2*n+edge++];if(local<0) break;
      candidate_vertex=info[graph+2+n+local];}
    else if(candidate_vertex==support.selected_vertex) continue;
    device const float* source=hull+3*(convex.vertex_offset+candidate_vertex);
    FlexDD projection=common_ccd_mesh_projection(direction,source);
    if(flex_dd_compare_exact(projection,threshold)<=0) continue;
    FlexDD3 point=common_ccd_local_to_world(convex.mat,convex.pos,
        flex_dd3(float3(source[0],source[1],source[2])));
    FlexDD3 delta=flex_dd3_source_sub(point,first);
    FlexDD separation=flex_dd_source_sqrt(flex_dd_source_dot3(delta,delta));
    // The pinned decimal is a double, not a rounded float literal.
    FlexDD threshold_distance=flex_dd_source_mul(
        FlexDD{0.30000001192092896f,-1.1920929132713809e-08f,1.6653345369377348e-16f},rbound);
    if(flex_dd_compare_exact(separation,threshold_distance)<0) continue;
    points[count]=point;
    depths[count]=flex_dd_source_dot3(normal,flex_dd3_source_sub(point,origin));
    count++;
  }
  return count;
}

kernel void contact_produce_plane_convex(
    device const float* geom_pos [[buffer(0)]],
    device const float* geom_pos_low [[buffer(1)]],
    device const float* geom_pos_tail [[buffer(2)]],
    device const float* geom_xmat [[buffer(3)]],
    device const float* geom_xmat_low [[buffer(4)]],
    device const float* geom_xmat_tail [[buffer(5)]],
    device const int* types [[buffer(6)]],device const int* pairs [[buffer(7)]],
    device const float* margins [[buffer(8)]],device const int* dims [[buffer(9)]],
    device const int* logical_map [[buffer(10)]],device float* hull [[buffer(11)]],
    device int* info [[buffer(12)]],device int* counts [[buffer(13)]],
    device float* records [[buffer(14)]],device atomic_int* status [[buffer(15)]],
    uint tid [[thread_position_in_grid]]) {
  int np=dims[1],nc=dims[2],batch=dims[3],ng=dims[6];
  if(np<=0 || tid>=uint(batch*np)) return;
  int world=int(tid)/np,pair=int(tid)%np,a=pairs[2*pair],b=pairs[2*pair+1];
  int other=types[a]==0?types[b]:types[a];
  if(!((types[a]==0 || types[b]==0) && (other==4 || other==7 || other==8))) return;
  int ptr=10+np+1+nc,mask_offset=dims[ptr];
  if(dims[mask_offset+world]==0 || logical_map[tid]<0) return;
  counts[tid]=0;
  bool flip=types[a]!=0;int plane=flip?b:a,solid=flip?a:b,header=9*ng;
  FlexDD ma[9],mb[9];
  FlexDD3 pa=common_ccd_load_geom_position(geom_pos,geom_pos_low,geom_pos_tail,3*(world*ng+plane));
  FlexDD3 pb=common_ccd_load_geom_position(geom_pos,geom_pos_low,geom_pos_tail,3*(world*ng+solid));
  common_ccd_load_geom_matrix(geom_xmat,geom_xmat_low,geom_xmat_tail,9*(world*ng+plane),ma);
  common_ccd_load_geom_matrix(geom_xmat,geom_xmat_low,geom_xmat_tail,9*(world*ng+solid),mb);
  if(!common_ccd_finite_pose(pa,ma)||!common_ccd_finite_pose(pb,mb)) {
    common_ccd_production_fail(status,world);return;
  }
  int base=info[header+1],rb=info[header+33]+3*solid;
  CommonCCDSupportObject p=common_ccd_production_geom(plane,0,base+9*plane,info,hull,pa,ma,float3(0),0);
  CommonCCDSupportObject c=common_ccd_production_geom(solid,other==8?7:other,base+9*solid,info,hull,pb,mb,float3(0),0);
  int sizes=info[header+34]+9*solid;
  for(int j=0;j<3;j++) c.size[j]=common_ccd_production_size(hull,sizes,j);
  FlexDD depths[3];FlexDD3 points[3],normal;
  int count=common_ccd_source_plane_convex(p,c,flex_dd(margins[2*pair]+margins[2*pair+1]),
      FlexDD{hull[rb],hull[rb+1],hull[rb+2]},hull,info,depths,points,normal);
  int offset=dims[10+pair],capacity=dims[11+pair]-offset;
  if(count<0 || count>capacity) {common_ccd_production_fail(status,world);return;}
  if(flip) normal=flex_dd3_neg(normal);
  for(int k=0;k<count;k++) {
    int dst=(world*nc+offset+k)*37;
    for(int j=0;j<37;j++) records[dst+j]=0;
    records[dst+12]=depths[k].hi;records[dst+9]=depths[k].lo;records[dst+10]=depths[k].tail;
    FlexDD3 position=flex_dd3_sub(points[k],flex_dd3_scale(flip?flex_dd3_neg(normal):normal,
        flex_dd_source_mul(flex_dd(.5f),depths[k])));
    float3 hi=flex_dd3_high(position),lo=float3(position.x.lo,position.y.lo,position.z.lo),
      tail=float3(position.x.tail,position.y.tail,position.z.tail);
    float3 nh=flex_dd3_high(normal),nl=float3(normal.x.lo,normal.y.lo,normal.z.lo),
      nt=float3(normal.x.tail,normal.y.tail,normal.z.tail),t1=float3(0),t2=float3(0);
    make_frame(nh,t1,t2);
    for(int j=0;j<3;j++) {
      records[dst+13+j]=hi[j];records[dst+25+j]=lo[j];records[dst+31+j]=tail[j];
      records[dst+16+j]=nh[j];records[dst+28+j]=nl[j];records[dst+34+j]=nt[j];
      records[dst+19+j]=t1[j];records[dst+22+j]=t2[j];
    }
  }
  counts[tid]=count;
}
