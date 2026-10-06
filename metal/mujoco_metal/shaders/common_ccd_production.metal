// Production candidate/witness bridge for pinned HField--compiled-mesh pairs.
// This file calls the shared source-order GJK/EPA implementation; it does not
// contain a second penetration solver.  `mesh_hull` is the combined static
// vertex upload plus bounded per-(world,pair) output/arena tail.  The extended
// `mesh_hull_info` ABI is described in common_ccd_bridge.py.

#define COMMON_CCD_PRODUCTION_MAGIC 1128487732
#define COMMON_CCD_GEOM_INFO_WORDS 9
#define COMMON_CCD_HEADER_WORDS 33
#define COMMON_CCD_OUTPUT_WORDS 802
#define COMMON_CCD_DIAG_PRODUCTION_CONTEXT_OFFSET 7330
#define COMMON_CCD_DIAG_PRODUCTION_CONTEXT_WORDS 232

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

static inline CommonCCDSupportObject common_ccd_production_mesh(
    int geom, int record, device const int* mesh_hull_info,
    device const float* mesh_hull, FlexDD3 local_position,
    thread const FlexDD* local_matrix, float3 size, float margin) {
  CommonCCDSupportObject object=common_ccd_production_object();
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
  // mju_mulMatTMat3 computes result[i,j] from column i of a and column j of
  // b, with source expression order first term, second term, third term.
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
    FlexDD first=flex_dd_source_mul(a[i],b[j]);
    FlexDD second=flex_dd_source_fma(a[3+i],b[3+j],first);
    result[3*i+j]=flex_dd_source_fma(a[6+i],b[6+j],second);
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
    int diagnostic_row, int diagnostic_col, int diagnostic_triangle) {
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
    thread int2* out_face_ids, thread int3& out_face_vertices) {
  out_face_vertices=int3(-1);
  CommonCCDSupportObject solve_a=a, solve_b=b;
  bool shrink_a=(a.geom_type==2 || a.geom_type==3);
  bool shrink_b=(b.geom_type==2 || b.geom_type==3);
  FlexDD full_a=flex_dd(0.0f), full_b=flex_dd(0.0f);
  if (shrink_a) {
    full_a=flex_dd_source_add(a.size[0],
        flex_dd_mul(margin,flex_dd(0.5f)));
    solve_a.kind=a.geom_type==2 ? COMMON_CCD_POINT : COMMON_CCD_LINE;
    solve_a.geom=-1; solve_a.margin=flex_dd(0.0f);
  }
  if (shrink_b) {
    full_b=flex_dd_source_add(b.size[0],
        flex_dd_mul(margin,flex_dd(0.5f)));
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
    if (shallow.status==2 || shallow.status==3) return -shallow.status;
    if (shallow.status==1) return 0;
    if (flex_dd_compare_exact(shallow.distance,tolerance)>0) {
      FlexDD distance=flex_dd_sub(shallow.distance,cutoff);
      if (flex_dd_compare_exact(distance,flex_dd(0.0f))>=0) return 0;
      FlexDD3 delta=flex_dd3_sub(shallow.witness_b,shallow.witness_a);
      FlexDD delta_norm=flex_dd_sqrt(flex_dd3_dot(delta,delta));
      if (flex_dd_compare_exact(delta_norm,flex_dd(1.0e-30f))<=0) return -2;
      FlexDD3 direction=flex_dd3_div(delta,delta_norm);
      FlexDD3 wa=flex_dd3_add(shallow.witness_a,
                                  flex_dd3_scale(direction,full_a));
      FlexDD3 wb=flex_dd3_sub(shallow.witness_b,
                                  flex_dd3_scale(direction,full_b));
      FlexDD3 normal_delta=flex_dd3_sub(wa,wb);
      FlexDD normal_norm=flex_dd_sqrt(flex_dd3_dot(normal_delta,normal_delta));
      if (flex_dd_compare_exact(normal_norm,flex_dd(1.0e-30f))<=0) return -2;
      contact.normal=flex_dd3_high(flex_dd3_div(normal_delta,normal_norm));
      contact.dist=(distance.hi+distance.lo+distance.tail)
          +(margin.hi+margin.lo+margin.tail);
      contact.pos=flex_dd3_high(flex_dd3_scale(
          flex_dd3_add(wa,wb),flex_dd(0.5f)));
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
  if (epa.status!=0) return -10-epa.status;
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
  bool reversed=(geom_type[a]==7 && geom_type[b]==1);
  int hfield_geom=reversed ? b : a;
  int mesh_geom=reversed ? a : b;
  if (geom_type[hfield_geom]!=1 || geom_type[mesh_geom]!=7) return;
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
  if (geom_records<0 || graph_base<0 || output_base<0 || output_stride<802 ||
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
  float margin=pair_margin_gap[2*pair];

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
  CommonCCDSupportObject mesh=common_ccd_production_mesh(
      mesh_geom,rec_mesh,mesh_hull_info,mesh_hull,local_pos,local_mat,size_mesh,margin);
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

  float size0f=mesh_hull[size_base], size1f=mesh_hull[size_base+3];
  float size2f=mesh_hull[size_base+6], size3f=mesh_hull[size_base+9];
  float fx=float(ncol-1)/max(2.0f*size0f,1.0e-30f);
  float fy=float(nrow-1)/max(2.0f*size1f,1.0e-30f);
  float rb=geom_rbound[mesh_geom]+margin;
  float3 local_pos_hi=flex_dd3_high(local_pos);
  if (local_pos_hi.x < -size0f-rb || local_pos_hi.x > size0f+rb ||
      local_pos_hi.y < -size1f-rb || local_pos_hi.y > size1f+rb ||
      local_pos_hi.z < -size3f-rb || local_pos_hi.z > size2f+rb) return;
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
  float xmin=nx.point.x.hi, xmax=sx.point.x.hi;
  float ymin=ny.point.y.hi, ymax=sy.point.y.hi;
  float zmin=nz.point.z.hi, zmax=sz.point.z.hi;
  if (xmin-margin>size0f || xmax+margin< -size0f ||
      ymin-margin>size1f || ymax+margin< -size1f ||
      zmin-margin>size2f || zmax+margin< -size3f) return;
  int cmin=max(0,int(floor((xmin+size0f)*fx)));
  int cmax=min(ncol-1,int(ceil((xmax+size0f)*fx)));
  int rmin=max(0,int(floor((ymin+size1f)*fy)));
  int rmax=min(nrow-1,int(ceil((ymax+size1f)*fy)));
  mesh.margin=flex_dd(margin);
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
      size0,size1,size2,size3,dx,dy,flex_dd(margin),local_pos,
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
              flex_dd(margin));
          if (v>=3) top_z[k]=z;
          FlexDD point[3]={x,y,z};
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
          center=flex_dd3_add(center,FlexDD3{x,y,z});
        }
        bool below_prism=true;
        for (int v=3;v<6;v++) {
          if (flex_dd_compare(top_z[v-3],nz.point.z)>=0) below_prism=false;
        }
        if (below_prism) continue;
        center=flex_dd3_scale(center,flex_dd_source_div(flex_dd(1.0f),flex_dd(6.0f)));
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
        int result=common_ccd_production_one_prism(
            prism,mesh,center,local_pos,
            iterations,tolerance,mesh_hull,mesh_hull_info,
            epa_vertices,vertex_id_base_ptr,epa_faces,epa_face_map,epa_horizon,epa_stack,
            vertex_capacity,face_capacity,horizon_capacity,stack_capacity,contact,face_ids,
            prism_diagnostic,r,c,tri);
        if (result<0) {
          status=-result;
          common_ccd_production_fail(world_status,world);
          break;
        }
        if (result>0) {
          // Contact is expressed in the hfield frame. Preserve the common
          // source witness, then map to world once at the production boundary.
          FlexDD3 world_pos=common_ccd_matrix_vector(Rh,flex_dd3(contact.pos));
          world_pos=flex_dd3_add(world_pos,ph);
          contact.pos=flex_dd3_high(world_pos);
          contact.normal=flex_dd3_high(
              common_ccd_matrix_vector(Rh,flex_dd3(contact.normal)));
          contact.t1=flex_dd3_high(
              common_ccd_matrix_vector(Rh,flex_dd3(contact.t1)));
          contact.t2=flex_dd3_high(
              common_ccd_matrix_vector(Rh,flex_dd3(contact.t2)));
          common_ccd_production_store_contact(mesh_hull,out,ncontact,contact);
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

static inline float3 common_ccd_mesh_feature_normal(
    int face_id, int geom_record, int header, device const int* info,
    device const float* hull, thread const CommonCCDSupportObject& object) {
  int normal_base=info[header+22];
  int polyadr=info[geom_record+5];
  int at=normal_base+3*(polyadr+face_id);
  FlexDD3 local=FlexDD3{flex_dd(hull[at]),flex_dd(hull[at+1]),
                         flex_dd(hull[at+2])};
  FlexDD3 world=common_ccd_matrix_vector(object.mat,local);
  return float3(world.x.hi,world.y.hi,world.z.hi);
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
  for (int i=0;i<count;i++) {
    float3 x2=clipped[i],x1=x2-(edge_a?-normals_b[selected.y]*direction_length:
                                      edge_b?-normals_a[selected.y]*direction_length:
                                      normals_b[selected.y]*direction_length);
    if (flip) { float3 tmp=x1;x1=x2;x2=tmp; }
    contacts[i]=contacts[0];
    contacts[i].pos=0.5f*(x1+x2);
    contacts[i].normal=normalize(x1-x2);
    contacts[i].t1=float3(0);make_frame(contacts[i].normal,contacts[i].t1,contacts[i].t2);
  }
  return min(count,4);
}

// Production source-GJK/EPA route for every pinned mjc_Convex dispatch entry.
// The current kernel emits the primary native witness; contact_normal consumes
// it before row/Jacobian assembly. Dedicated plane/HField/SDF and analytic
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
  if (geom_records<0 || graph_base<0 || output_base<0 || output_stride<802 ||
      scratch_base<0 || scratch_stride<=0 || iterations<=0 ||
      tolerance_base<0 || vertex_capacity<6 || face_capacity<6 ||
      horizon_capacity<3 || stack_capacity<1) {
    common_ccd_production_fail(world_status,world); return;
  }
  int query=world*npairs+pair;
  int out=output_base+query*output_stride;
  mesh_hull[out]=0.0f; mesh_hull[out+1]=0.0f;
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
  if (!isfinite(margin_value) || margin_value<0.0f) {
    mesh_hull[out+1]=4.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  CommonCCDSupportObject object_a=common_ccd_production_geom(
      a,geom_type[a],rec_a,mesh_hull_info,mesh_hull,pa,ma,
      sza,margin_value);
  CommonCCDSupportObject object_b=common_ccd_production_geom(
      b,geom_type[b],rec_b,mesh_hull_info,mesh_hull,pb,mb,
      szb,margin_value);
  if (object_a.kind<0 || object_b.kind<0) {
    mesh_hull[out+1]=3.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  if ((geom_type[a]==7 && (object_a.vertex_offset<0 || object_a.vertex_count<=0)) ||
      (geom_type[b]==7 && (object_b.vertex_offset<0 || object_b.vertex_count<=0))) {
    mesh_hull[out+1]=3.0f;
    common_ccd_production_fail(world_status,world); return;
  }
  int vertex_words=29*vertex_capacity;
  int face_words=20*face_capacity;
  int map_words=face_capacity;
  int horizon_words=2*horizon_capacity;
  int horizon_alignment=vertex_words&1;
  device FlexDDVertex* epa_vertices=reinterpret_cast<device FlexDDVertex*>(
      mesh_hull+scratch);
  int idbase=mesh_hull_info[header+24]+query*vertex_capacity;
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
  FlexDD3 center_a=pa;
  FlexDD3 center_b=pb;
  ContactGeom contact;
  contact.dist=3.402823466e+38f; contact.pos=float3(0);
  contact.normal=float3(0); contact.t1=float3(0); contact.t2=float3(0);
  int2 face_ids[3];
  int3 face_slots=int3(-1);
  int found=common_ccd_production_one_rigid_pair(
      object_a,object_b,center_a,center_b,
      FlexDD{margin_value,0.0f,0.0f},iterations,tolerance,
      mesh_hull,mesh_hull_info,epa_vertices,epa_vertex_ids,epa_faces,epa_face_map,
      epa_horizon,epa_stack,vertex_capacity,face_capacity,horizon_capacity,
      stack_capacity,contact,face_ids,face_slots);
  if (found<0) {
    mesh_hull[out+1]=float(-found);
    common_ccd_production_fail(world_status,world); return;
  }
  if (found>0) {
    ContactGeom contacts[4];
    contacts[0]=contact;
    int contact_count=1;
    bool source_multicontact = margin_value==0.0f && pair_dims[8]==0 &&
        (geom_type[a]==6 || geom_type[a]==7) &&
        (geom_type[b]==6 || geom_type[b]==7);
    if (source_multicontact && face_slots.x>=0 &&
        face_slots.y>=0 && face_slots.z>=0) {
      contact_count=common_ccd_multicontact_faces(
          geom_type[a],geom_type[b],rec_a,rec_b,header,mesh_hull_info,
          mesh_hull,object_a,object_b,face_ids,face_slots,epa_vertices,contact.normal,
          max(0.0f,-contact.dist),contacts);
      contact_count=min(contact_count,4);
    }
    for (int k=0;k<contact_count;k++)
      common_ccd_production_store_contact(mesh_hull,out,k,contacts[k]);
    mesh_hull[out]=float(contact_count);
  }
}
