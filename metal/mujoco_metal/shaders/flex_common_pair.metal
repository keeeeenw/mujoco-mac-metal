// One pinned mjc_ConvexElem contact for a fixed flex-element pair slot.
// The descriptor has one slot per source element pair because MuJoCo calls
// mjc_ConvexElem with max_contacts=1. This producer deliberately fails closed
// if the common source core reports an invalid seed/EPA workspace.
static inline FlexDD3 flex_pair_vertex(device const float* pair,
                                       int env,int nvert,int node) {
  int world_base=env*3*nvert*3;
  int base=world_base+node*3;
  int low_base=world_base+nvert*3+node*3;
  int tail_base=world_base+2*nvert*3+node*3;
  return FlexDD3{
    FlexDD{pair[base],pair[low_base],pair[tail_base]},
    FlexDD{pair[base+1],pair[low_base+1],pair[tail_base+1]},
    FlexDD{pair[base+2],pair[low_base+2],pair[tail_base+2]}};
}

static inline FlexDD3 flex_pair_center(device const float* pair,
                                       int env,int nvert,
                                       device const int* nodes,
                                       int base,int count) {
  FlexDD3 lo=flex_pair_vertex(pair,env,nvert,nodes[base]);
  FlexDD3 hi=lo;
  for (int i=1;i<count;i++) {
    FlexDD3 p=flex_pair_vertex(pair,env,nvert,nodes[base+i]);
    if (flex_dd_compare_exact(p.x,lo.x)<0) lo.x=p.x;
    if (flex_dd_compare_exact(p.y,lo.y)<0) lo.y=p.y;
    if (flex_dd_compare_exact(p.z,lo.z)<0) lo.z=p.z;
    if (flex_dd_compare_exact(p.x,hi.x)>0) hi.x=p.x;
    if (flex_dd_compare_exact(p.y,hi.y)>0) hi.y=p.y;
    if (flex_dd_compare_exact(p.z,hi.z)>0) hi.z=p.z;
  }
  return flex_dd3_scale(flex_dd3_add(lo,hi),flex_dd(0.5f));
}

kernel void flex_contact_detect_common_element_pair(
    device const int* candidate_meta [[buffer(0)]],
    device const int* candidate_nodes1 [[buffer(1)]],
    device const float* flexvert_xpos_pair [[buffer(2)]],
    device const float* geom_pos [[buffer(3)]],
    device const float* geom_quat [[buffer(4)]],
    device const float* geom_size [[buffer(5)]],
    device const int* geom_type [[buffer(6)]],
    device const float* flex_radius [[buffer(7)]],
    device const float* candidate_margin_gap [[buffer(8)]],
    device const int* candidate_nodes2 [[buffer(9)]],
    device const int* geom_hull_info [[buffer(10)]],
    constant int* dims [[buffer(11)]],
    device int* active [[buffer(12)]],
    device float* distance [[buffer(13)]],
    device float* contact_pos [[buffer(14)]],
    device float* contact_normal [[buffer(15)]],
    device float* barycentric1 [[buffer(16)]],
    device float* barycentric2 [[buffer(17)]],
    device float* contact_frame [[buffer(18)]],
    device float* candidate_selection_pos [[buffer(19)]],
    device const int* flex_vertadr [[buffer(20)]],
    device float* jacobian_weights [[buffer(21)]],
    constant float* ccd_tolerance [[buffer(22)]],
    device int* narrowphase_status [[buffer(23)]],
    device float* ccd_trace [[buffer(24)]],
    device float* epa_float_workspace [[buffer(25)]],
    device int* epa_int_workspace [[buffer(26)]],
    device const float* flex_radius_low [[buffer(27)]],
    device const float* geom_size_low [[buffer(28)]],
    device const float* candidate_margin_gap_low [[buffer(29)]],
    constant float* ccd_tolerance_low [[buffer(30)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0], nslot=dims[1], nvert=dims[2];
  if (tid>=uint(batch*nslot) || dims[4]==0) return;
  int env=int(tid)/nslot, slot=int(tid)%nslot;
  if (dims[23+env]==0) return;
  int world_vertex_base=env*3*nvert*3;
  device const float* flexvert_xpos=flexvert_xpos_pair+world_vertex_base;
  int meta=10*slot;
  int kind=candidate_meta[meta];
  int f1=candidate_meta[meta+1], e1=candidate_meta[meta+2];
  int f2=candidate_meta[meta+4], e2=candidate_meta[meta+5];
  if (kind!=FLEX_ELEMENT_PAIR || f1<0 || f2<0 || e1<0 || e2<0) return;
  int base=4*slot, count1=0, count2=0;
  while (count1<4 && candidate_nodes1[base+count1]>=0) count1++;
  while (count2<4 && candidate_nodes2[base+count2]>=0) count2++;
  // The exact 2-by-2 case is already produced by the source raw-capsule
  // manifold path. This source-order support path also handles edge/face
  // combinations in either side order.
  if (count1<2 || count2<2 || (count1==2 && count2==2)) return;
  active[tid]=0;
  distance[tid]=0.0f;
  for (int k=0;k<3;k++) {
    contact_pos[3*tid+k]=0.0f;
    candidate_selection_pos[3*tid+k]=0.0f;
    contact_normal[3*tid+k]=0.0f;
    contact_frame[9*tid+k]=0.0f;
    contact_frame[9*tid+3+k]=0.0f;
    contact_frame[9*tid+6+k]=0.0f;
    barycentric1[4*tid+k]=0.0f;
    barycentric2[4*tid+k]=0.0f;
  }
  barycentric1[4*tid+3]=0.0f;
  barycentric2[4*tid+3]=0.0f;
  for (int k=0;k<4;k++) {
    jacobian_weights[4*tid+k]=0.0f;
    jacobian_weights[4*batch*nslot+4*tid+k]=0.0f;
  }

  float3 center1f=flex_center(flexvert_xpos,env,nvert,candidate_nodes1,base,count1);
  float3 center2f=flex_center(flexvert_xpos,env,nvert,candidate_nodes2,base,count2);
  FlexDD3 center1=flex_pair_center(
      flexvert_xpos_pair,env,nvert,candidate_nodes1,base,count1);
  FlexDD3 center2=flex_pair_center(
      flexvert_xpos_pair,env,nvert,candidate_nodes2,base,count2);
  thread float3 support1[4], support2[4];
  CommonCCDSupportObject a, b;
  a.kind=COMMON_CCD_FLEX_ELEMENT; b.kind=COMMON_CCD_FLEX_ELEMENT;
  a.geom=-1; b.geom=-1; a.geom_type=0; b.geom_type=0;
  a.vertex_offset=0; b.vertex_offset=0;
  a.vertex_count=count1; b.vertex_count=count2;
  a.cached_vertex=-1; b.cached_vertex=-1;
  a.graph_offset=-1; b.graph_offset=-1;
  a.graph_vertex_count=0; b.graph_vertex_count=0;
  for (int i=0;i<9;i++) {
    FlexDD v=(i==0 || i==4 || i==8) ? flex_dd(1.0f) : flex_dd(0.0f);
    a.mat[i]=v; b.mat[i]=v;
  }
  for (int i=0;i<3;i++) {
    a.pos[i]=flex_dd(0.0f); b.pos[i]=flex_dd(0.0f);
  }
  for (int i=0;i<3;i++) {
    a.size[i]=flex_dd(0.0f); b.size[i]=flex_dd(0.0f);
  }
  a.size[0]=FlexDD{flex_radius[f1],flex_radius_low[2*f1],
                   flex_radius_low[2*f1+1]};
  b.size[0]=FlexDD{flex_radius[f2],flex_radius_low[2*f2],
                   flex_radius_low[2*f2+1]};
  FlexDD margin=FlexDD{candidate_margin_gap[2*slot],
      candidate_margin_gap_low[4*slot],candidate_margin_gap_low[4*slot+1]};
  FlexDD gap=FlexDD{candidate_margin_gap[2*slot+1],
      candidate_margin_gap_low[4*slot+2],candidate_margin_gap_low[4*slot+3]};
  FlexDD threshold=flex_dd_source_add(margin,gap);
  a.margin=threshold; b.margin=threshold;
  for (int i=0;i<4;i++) {
    FlexDD3 va=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
    FlexDD3 vb=va;
    if (i<count1) {
      float3 p=flex_world_vertex(flexvert_xpos,env,nvert,candidate_nodes1[base+i]);
      support1[i]=p;
      va=flex_pair_vertex(flexvert_xpos_pair,env,nvert,
                          candidate_nodes1[base+i]);
    } else support1[i]=float3(0.0f);
    if (i<count2) {
      float3 p=flex_world_vertex(flexvert_xpos,env,nvert,candidate_nodes2[base+i]);
      support2[i]=p;
      vb=flex_pair_vertex(flexvert_xpos_pair,env,nvert,
                          candidate_nodes2[base+i]);
    } else support2[i]=float3(0.0f);
    a.flex_vertices[i]=va; b.flex_vertices[i]=vb;
  }
  FlexDD tol=FlexDD{ccd_tolerance[0],ccd_tolerance_low[0],
                    ccd_tolerance_low[1]};
  CommonCCDVertex simplex[4];
  for (int i=0;i<4;i++) {
    simplex[i].point_a=center1;
    simplex[i].point_b=center2;
    simplex[i].minkowski=flex_dd3_sub(center1,center2);
    simplex[i].index_a=-1; simplex[i].index_b=-1;
  }
  CommonCCDGjkResult gjk=common_ccd_source_gjk(
      a,b,center1,center2,tol,flex_dd(0.0f),dims[6],
      reinterpret_cast<device const float*>(candidate_nodes2),
      geom_hull_info,simplex,nullptr,0);
  // The pair's fixed slot is never compacted. A clean GJK separation remains
  // inactive; malformed support/seed/EPA state is surfaced to the solver.
  if (gjk.status==1) return;
  if (gjk.status!=0 || gjk.simplex_count<2) {
    narrowphase_status[tid]=34;
    return;
  }
  if (!gjk.needs_intersection) return;
  ContactGeom contact;
  contact.dist=3.402823466e+38f;
  contact.pos=float3(0.0f); contact.normal=float3(0.0f);
  contact.t1=float3(0.0f); contact.t2=float3(0.0f);
  int float_base=int(tid)*dims[16];
  int int_base=int(tid)*dims[17];
  int vertex_capacity=dims[11], face_capacity=dims[12];
  int horizon_capacity=dims[13], stack_capacity=dims[14];
  device FlexDDVertex* vertices=reinterpret_cast<device FlexDDVertex*>(
      epa_float_workspace+float_base+vertex_capacity*12);
  device FlexEpaFace* faces=reinterpret_cast<device FlexEpaFace*>(
      epa_float_workspace+float_base+vertex_capacity*39);
  device int* face_map=epa_int_workspace+int_base;
  device int2* horizon=reinterpret_cast<device int2*>(
      epa_int_workspace+int_base+dims[15]);
  device FlexHorizonFrame* stack=reinterpret_cast<device FlexHorizonFrame*>(
      epa_int_workspace+int_base+dims[15]+horizon_capacity*2);
  int ignored_epa_support_count=0, ignored_epa_face_count=0;
  CommonCCDEpaResult epa=common_ccd_source_epa(
      simplex,gjk.simplex_count,a,b,dims[6],tol,gjk.distance,
      reinterpret_cast<device const float*>(candidate_nodes2),geom_hull_info,
      vertices,nullptr,faces,face_map,horizon,stack,vertex_capacity,face_capacity,
      horizon_capacity,stack_capacity,contact,nullptr,nullptr,
      ignored_epa_support_count,ignored_epa_face_count);
  if (epa.status!=0 || !isfinite(contact.dist)) {
    narrowphase_status[tid]=35;
    return;
  }
  float contact_dist=contact.dist+(threshold.hi+threshold.lo+threshold.tail);
  float3 q=contact.pos, n=contact.normal, t1=contact.t1, t2=contact.t2;
  float radius1=a.size[0].hi+a.size[0].lo+a.size[0].tail;
  float radius2=b.size[0].hi+b.size[0].lo+b.size[0].tail;
  float3 relative1[4], relative2[4];
  for (int i=0;i<4;i++) {
    relative1[i]=support1[i]-center1f;
    relative2[i]=support2[i]-center2f;
  }
  float3 target1=q-n*(radius1+0.5f*contact_dist);
  float3 target2=q+n*(radius2+0.5f*contact_dist);
  float4 weights1, weights2;
  flex_barycentric(target1-center1f,relative1,count1,weights1);
  flex_barycentric(target2-center2f,relative2,count2,weights2);
  make_frame(n,t1,t2);
  active[tid]=contact_dist<=threshold.hi+threshold.lo+threshold.tail;
  distance[tid]=contact_dist;
  for (int k=0;k<3;k++) {
    contact_pos[3*tid+k]=q[k];
    candidate_selection_pos[3*tid+k]=q[k];
    contact_normal[3*tid+k]=n[k];
    contact_frame[9*tid+k]=n[k];
    contact_frame[9*tid+3+k]=t1[k];
    contact_frame[9*tid+6+k]=t2[k];
    barycentric1[4*tid+k]=weights1[k];
    barycentric2[4*tid+k]=weights2[k];
  }
  barycentric1[4*tid+3]=weights1.w;
  barycentric2[4*tid+3]=weights2.w;
  float4 jw1=flex_contact_jac_weights(f1,e1,-1,f2,-1,q,env,slot,nvert,
      candidate_nodes1,flex_vertadr,flexvert_xpos);
  float4 jw2=flex_contact_jac_weights(f2,e2,-1,f1,-1,q,env,slot,nvert,
      candidate_nodes2,flex_vertadr,flexvert_xpos);
  for (int i=0;i<4;i++) {
    jacobian_weights[4*tid+i]=jw1[i];
    jacobian_weights[4*batch*nslot+4*tid+i]=jw2[i];
  }
}
