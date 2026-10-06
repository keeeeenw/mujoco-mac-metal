// Private source-order CCD support adapter prototype.
//
// Private common-CCD candidate concatenated after flex_contact.metal to reuse
// its three-word FlexDD and EPA topology primitives. It contains a generic
// support adapter, source-order GJK/intersection, P2/P3/P4 EPA initialization,
// an EPA loop, and a standalone diagnostic entry. It is not wired to a
// production detector and remains unqualified.

static constant int COMMON_CCD_POINT = 0;
static constant int COMMON_CCD_SPHERE = 1;
static constant int COMMON_CCD_LINE = 2;
static constant int COMMON_CCD_CAPSULE = 3;
static constant int COMMON_CCD_BOX = 4;
static constant int COMMON_CCD_HULL = 5;
static constant int COMMON_CCD_ELLIPSOID = 6;
static constant int COMMON_CCD_CYLINDER = 7;
static constant int COMMON_CCD_HFIELD_PRISM = 8;
static constant int COMMON_CCD_FLEX_ELEMENT = 9;

struct CommonCCDSupportObject {
  int kind;
  int geom;
  int geom_type;
  int vertex_offset;
  int vertex_count;
  int cached_vertex;
  int graph_offset;
  int graph_vertex_count;
  int graph_word_count;
  FlexDD pos[3];
  FlexDD mat[9];
  FlexDD size[3];
  FlexDD margin;
  // Inline simplex vertices let the flex producer call the same source-order
  // GJK/EPA core without staging a dense per-candidate vertex buffer. The
  // producer fills this only for COMMON_CCD_FLEX_ELEMENT.
  FlexDD3 flex_vertices[4];
  // Residual float words for source-double generated prism vertices. Compiled
  // mesh/hull vertices remain exact float32 values with zero residuals.
  FlexDD3 hfield_residual[6];
};

struct CommonCCDSupportResult {
  FlexDD3 point;
  int selected_vertex;
  int status;
};

static inline FlexDD3 common_ccd_matrix_vector(
    thread const FlexDD* matrix, FlexDD3 vector) {
  // mji_mulMatVec3's lexical row/column order. Preserve the first product as
  // the rounded accumulator and contract each following product-plus-sum.
  return FlexDD3{
      flex_dd_source_fma(matrix[2],vector.z,
          flex_dd_source_fma(matrix[1],vector.y,
              flex_dd_source_mul(matrix[0],vector.x))),
      flex_dd_source_fma(matrix[5],vector.z,
          flex_dd_source_fma(matrix[4],vector.y,
              flex_dd_source_mul(matrix[3],vector.x))),
      flex_dd_source_fma(matrix[8],vector.z,
          flex_dd_source_fma(matrix[7],vector.y,
              flex_dd_source_mul(matrix[6],vector.x)))};
}

static inline FlexDD3 common_ccd_matrix_transpose_vector(
    thread const FlexDD* matrix, FlexDD3 vector) {
  // mji_mulMatTVec3 source order: matrix rows are strided by three.
  return FlexDD3{
      flex_dd_source_fma(matrix[6],vector.z,
          flex_dd_source_fma(matrix[3],vector.y,
              flex_dd_source_mul(matrix[0],vector.x))),
      flex_dd_source_fma(matrix[7],vector.z,
          flex_dd_source_fma(matrix[4],vector.y,
              flex_dd_source_mul(matrix[1],vector.x))),
      flex_dd_source_fma(matrix[8],vector.z,
          flex_dd_source_fma(matrix[5],vector.y,
              flex_dd_source_mul(matrix[2],vector.x)))};
}

static inline FlexDD3 common_ccd_local_to_world(
    thread const FlexDD* matrix, thread const FlexDD* pos, FlexDD3 point) {
  FlexDD3 world=common_ccd_matrix_vector(matrix,point);
  return FlexDD3{flex_dd_source_add(world.x,pos[0]),
                 flex_dd_source_add(world.y,pos[1]),
                 flex_dd_source_add(world.z,pos[2])};
}

static inline FlexDD common_ccd_mesh_projection(
    FlexDD3 local_direction, device const float* vertex_data) {
  // Pinned mjc_meshSupport uses dot3f(local_dir, float_vertex), with strict
  // greater-than updates. Vertex coordinates are compiled float mesh data;
  // the accumulated projection and tie comparison remain source precision.
  FlexDD3 v=flex_dd3(float3(vertex_data[0],vertex_data[1],vertex_data[2]));
  return flex_dd_source_dot3(local_direction,v);
}

static inline FlexDD common_ccd_hfield_projection(
    FlexDD3 local_direction, device const float* vertex_data, int address,
    FlexDD3 residual) {
  // Unlike compiled mesh vertices, HField prism coordinates are mjtNum
  // values synthesized from grid spacing and dimensions. Include their
  // retained residual pair when reproducing mjc_prism_support's strict
  // source-precision dot-product ordering.
  FlexDD3 support_vertex=FlexDD3{
      FlexDD{vertex_data[address],residual.x.hi,residual.x.lo},
      FlexDD{vertex_data[address+1],residual.y.hi,residual.y.lo},
      FlexDD{vertex_data[address+2],residual.z.hi,residual.z.lo}};
  return flex_dd3_dot(local_direction,support_vertex);
}

static inline CommonCCDSupportResult common_ccd_support(
    thread const CommonCCDSupportObject& object, FlexDD3 direction,
    device const float* mesh_vertices) {
  CommonCCDSupportResult out;
  out.point=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
  out.selected_vertex=-1;
  out.status=0;
  FlexDD3 pos=FlexDD3{object.pos[0],object.pos[1],object.pos[2]};
  FlexDD3 local_direction=common_ccd_matrix_transpose_vector(object.mat,
                                                              direction);
  FlexDD3 local=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
  FlexDD3 world;

  if (object.kind==COMMON_CCD_POINT) {
    // mjc_pointSupport copies `pos` directly; it does not round-trip through
    // the object's rotation matrix.
    world=pos;
  } else if (object.kind==COMMON_CCD_SPHERE) {
    // mjc_sphereSupport is likewise world-axis invariant and computes
    // radius*dir + pos without a local-frame rotation.
    world=flex_dd3_source_madd(direction,object.size[0],pos);
  } else if (object.kind==COMMON_CCD_LINE) {
    FlexDD sign=flex_dd_compare_exact(local_direction.z,flex_dd(0.0f))>=0
                    ? object.size[1] : flex_dd_neg(object.size[1]);
    local=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),sign};
  } else if (object.kind==COMMON_CCD_CAPSULE) {
    local=flex_dd3_scale(local_direction,object.size[0]);
    FlexDD axial=flex_dd_compare_exact(local_direction.z,flex_dd(0.0f))>=0
                    ? object.size[1] : flex_dd_neg(object.size[1]);
    local.z=flex_dd_source_add(local.z,axial);
  } else if (object.kind==COMMON_CCD_BOX) {
    local=FlexDD3{
        flex_dd_compare_exact(local_direction.x,flex_dd(0.0f))>=0
            ? object.size[0] : flex_dd_neg(object.size[0]),
        flex_dd_compare_exact(local_direction.y,flex_dd(0.0f))>=0
            ? object.size[1] : flex_dd_neg(object.size[1]),
        flex_dd_compare_exact(local_direction.z,flex_dd(0.0f))>=0
            ? object.size[2] : flex_dd_neg(object.size[2])};
    // Match mjc_boxSupport's source bit encoding (zero local direction picks
    // the positive corner because local_supp uses >= 0).
    out.selected_vertex=(flex_dd_compare_exact(local_direction.x,
                            flex_dd(0.0f))>=0 ? 1 : 0)
        | (flex_dd_compare_exact(local_direction.y,flex_dd(0.0f))>=0 ? 2 : 0)
        | (flex_dd_compare_exact(local_direction.z,flex_dd(0.0f))>=0 ? 4 : 0);
  } else if (object.kind==COMMON_CCD_ELLIPSOID) {
    FlexDD3 scaled=FlexDD3{
        flex_dd_mul(local_direction.x,object.size[0]),
        flex_dd_mul(local_direction.y,object.size[1]),
        flex_dd_mul(local_direction.z,object.size[2])};
    FlexDD norm2=flex_dd3_dot(scaled,scaled);
    FlexDD tiny=FlexDD{1.0000000031710769e-30f,-3.171076767634631e-39f,0.0f};
    if (flex_dd_compare_exact(norm2,tiny)<0) {
      // Preserve the pinned degenerate-direction fallback exactly: first
      // rotation column multiplied by size[0], then translated.
      local=FlexDD3{object.size[0],flex_dd(0.0f),flex_dd(0.0f)};
    } else {
      FlexDD norm_inv=flex_dd_source_div(flex_dd(1.0f),
                                          flex_dd_sqrt(norm2));
      local=FlexDD3{
          flex_dd_mul(scaled.x,flex_dd_mul(norm_inv,object.size[0])),
          flex_dd_mul(scaled.y,flex_dd_mul(norm_inv,object.size[1])),
          flex_dd_mul(scaled.z,flex_dd_mul(norm_inv,object.size[2]))};
    }
  } else if (object.kind==COMMON_CCD_CYLINDER) {
    FlexDD norm2=flex_dd_source_add(
        flex_dd_mul(local_direction.x,local_direction.x),
        flex_dd_mul(local_direction.y,local_direction.y));
    FlexDD tiny=FlexDD{1.0000000031710769e-30f,-3.171076767634631e-39f,0.0f};
    FlexDD scl=flex_dd_compare_exact(norm2,tiny)>=0
        ? flex_dd_source_div(object.size[0],flex_dd_sqrt(norm2))
        : flex_dd(0.0f);
    FlexDD axial=flex_dd_compare_exact(local_direction.z,flex_dd(0.0f))>=0
        ? object.size[1] : flex_dd_neg(object.size[1]);
    local=FlexDD3{flex_dd_mul(scl,local_direction.x),
                  flex_dd_mul(scl,local_direction.y),axial};
  } else if (object.kind==COMMON_CCD_FLEX_ELEMENT) {
    // mjc_flexSupport selects the first element vertex attaining the strict
    // maximum projection, then adds flex_radius + margin/2 along the original
    // (already source-selected) direction. The vertex words are world-space
    // `flexvert_xpos`; no geom transform is applied to this object kind.
    if (object.vertex_offset<0 || object.vertex_count<1 ||
        object.vertex_count>4) {
      out.status=1;
      return out;
    }
    int best=0;
    local=object.flex_vertices[0];
    FlexDD best_dot=flex_dd3_dot(local,direction);
    for (int i=1;i<object.vertex_count;i++) {
      FlexDD3 candidate=object.flex_vertices[i];
      FlexDD score=flex_dd3_dot(candidate,direction);
      if (flex_dd_compare_exact(score,best_dot)>0) {
        best=i;
        best_dot=score;
        local=candidate;
      }
    }
    FlexDD radius_margin=flex_dd_source_add(
        object.size[0],flex_dd_mul(object.margin,flex_dd(0.5f)));
    world=flex_dd3_source_madd(direction,radius_margin,local);
    out.selected_vertex=best;
  } else if (object.kind==COMMON_CCD_HFIELD_PRISM) {
    // mjc_prism_support selects one triangular layer by the sign of local z,
    // then keeps the first vertex on equal mju_dot3 scores. The prism vertices
    // are the exact six values produced by pinned addPrismVert; margin is
    // already baked into the rolling top vertex, so callers set geom=-1.
    if (object.vertex_offset<0 || object.vertex_count!=6) {
      out.status=1;
      return out;
    }
    int start=flex_dd_compare_exact(local_direction.z,flex_dd(0.0f))<0 ? 0 : 3;
    int best=start;
    int best_address=3*(object.vertex_offset+best);
    FlexDD best_dot=common_ccd_hfield_projection(
        local_direction,mesh_vertices,best_address,
        object.hfield_residual[best]);
    for (int i=1;i<3;i++) {
      int candidate=start+i;
      int candidate_address=3*(object.vertex_offset+candidate);
      FlexDD score=common_ccd_hfield_projection(
          local_direction,mesh_vertices,candidate_address,
          object.hfield_residual[candidate]);
      if (flex_dd_compare_exact(score,best_dot)>0) {
        best_dot=score;
        best=candidate;
      }
    }
    int address=3*(object.vertex_offset+best);
    local=FlexDD3{
        FlexDD{mesh_vertices[address],object.hfield_residual[best].x.hi,
               object.hfield_residual[best].x.lo},
        FlexDD{mesh_vertices[address+1],object.hfield_residual[best].y.hi,
               object.hfield_residual[best].y.lo},
        FlexDD{mesh_vertices[address+2],object.hfield_residual[best].z.hi,
               object.hfield_residual[best].z.lo}};
    out.selected_vertex=best;
  } else if (object.kind==COMMON_CCD_HULL) {
    if (object.vertex_offset<0 || object.vertex_count<=0 ||
        object.cached_vertex>=object.vertex_count) {
      out.status=1;
      return out;
    }
    int best=object.cached_vertex>=0 ? object.cached_vertex : 0;
    FlexDD best_dot=FlexDD{ -3.402823466e+38f,0.0f,0.0f };
    if (object.cached_vertex>=0) {
      best_dot=common_ccd_mesh_projection(local_direction,
          mesh_vertices+3*(object.vertex_offset+best));
    }
    for (int i=0;i<object.vertex_count;i++) {
      FlexDD score=common_ccd_mesh_projection(local_direction,
          mesh_vertices+3*(object.vertex_offset+i));
      if (flex_dd_compare_exact(score,best_dot)>0) {
        best_dot=score;
        best=i;
      }
    }
    int address=3*(object.vertex_offset+best);
    local=flex_dd3(float3(mesh_vertices[address],mesh_vertices[address+1],
                          mesh_vertices[address+2]));
    out.selected_vertex=best;
  } else {
    out.status=2;
    return out;
  }

  bool already_world=(object.kind==COMMON_CCD_POINT ||
                      object.kind==COMMON_CCD_SPHERE ||
                      object.kind==COMMON_CCD_FLEX_ELEMENT);
  if (!already_world)
    world=common_ccd_local_to_world(object.mat,object.pos,local);
  // `support` adds half of each geom margin along its normalized world input
  // direction after the shape callback. Callers implementing mjc_ccd's
  // sphere/capsule shrink branch pass geom=-1 and margin=0 here, then apply
  // full-margin inflation later.
  if (object.geom>=0 && flex_dd_compare_exact(object.margin,flex_dd(0.0f))>0) {
    FlexDD half_margin=flex_dd_mul(object.margin,flex_dd(0.5f));
    world=flex_dd3_source_madd(direction,half_margin,world);
  }
  out.point=world;
  return out;
}


// Source-order `mjc_hillclimbSupport` adapter for compiled mesh graphs. The
// graph is the exact model.mesh_graph payload and `mesh_index` is the mutable
// per-CCD-object local vertex ID carried across support calls. The strict
// `vdot > max` test preserves warm-start and tie behavior. The walk can make at
// most graph_vertex_count strict improvements; overflow/corrupt adjacency
// returns a nonzero status instead of indexing outside the model arrays.
static inline CommonCCDSupportResult common_ccd_hillclimb_mesh_support(
    thread const CommonCCDSupportObject& object, FlexDD3 direction,
    device const float* mesh_vertices, device const int* mesh_graph,
    thread int& mesh_index) {
  CommonCCDSupportResult out;
  out.point=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
  out.selected_vertex=-1;
  out.status=0;
  if (object.graph_offset<0 || object.graph_vertex_count<=0
      || object.vertex_offset<0 || object.vertex_count<=0
      || object.graph_vertex_count>object.vertex_count
      || object.graph_word_count<2+2*object.graph_vertex_count) {
    out.status=3;
    return out;
  }
  int graph_base=object.graph_offset;
  int ngraph=mesh_graph[graph_base];
  int nface=mesh_graph[graph_base+1];
  if (ngraph!=object.graph_vertex_count || nface<=0
      || object.graph_word_count!=2+3*ngraph+6*nface) {
    out.status=4;
    return out;
  }
  int adjacency_begin=graph_base+2;
  int global_id_begin=adjacency_begin+ngraph;
  int edge_local_begin=global_id_begin+ngraph;
  int edge_local_end=edge_local_begin+ngraph+3*nface;
  int graph_end=graph_base+object.graph_word_count;
  if (edge_local_end+3*nface!=graph_end) { out.status=4; return out; }
  FlexDD3 local_direction=common_ccd_matrix_transpose_vector(
      object.mat,direction);
  FlexDD best=FlexDD{-3.402823466e+38f,0.0f,0.0f};
  int previous=-1;
  int current=(mesh_index>=0) ? mesh_index : 0;
  int steps=0;
  while (current!=previous) {
    if (current<0 || current>=ngraph || steps++>ngraph) {
      out.status=5;
      return out;
    }
    previous=current;
    int edge=mesh_graph[adjacency_begin+current];
    int visits=0;
    while (true) {
      if (edge<0 || edge_local_begin+edge>=edge_local_end) {
        out.status=6;
        return out;
      }
      int subidx=mesh_graph[edge_local_begin+edge];
      if (subidx<0) break;
      if (subidx>=ngraph || visits++>=ngraph) {
        out.status=8;
        return out;
      }
      int global_vertex=mesh_graph[global_id_begin+subidx];
      if (global_vertex<0 || global_vertex>=object.vertex_count) {
        out.status=9;
        return out;
      }
      int vertex_address=3*(object.vertex_offset+global_vertex);
      FlexDD score=common_ccd_mesh_projection(local_direction,
                                               mesh_vertices+vertex_address);
      if (flex_dd_compare_exact(score,best)>0) {
        best=score;
        current=subidx;
      }
      edge++;
    }
  }
  mesh_index=current;
  int global_vertex=mesh_graph[global_id_begin+current];
  if (global_vertex<0 || global_vertex>=object.vertex_count) {
    out.status=10;
    return out;
  }
  int vertex_address=3*(object.vertex_offset+global_vertex);
  FlexDD3 local=flex_dd3(float3(mesh_vertices[vertex_address],
      mesh_vertices[vertex_address+1],mesh_vertices[vertex_address+2]));
  FlexDD3 world=common_ccd_local_to_world(object.mat,object.pos,local);
  if (flex_dd_compare_exact(object.margin,flex_dd(0.0f))>0) {
    world=flex_dd3_source_madd(direction,
        flex_dd_mul(object.margin,flex_dd(0.5f)),world);
  }
  out.point=world;
  out.selected_vertex=global_vertex;
  return out;
}

// Source-order generic Minkowski/GJK stage. This is a caller-usable core
// primitive, rather than a support-only demo: callers provide compiled object
// descriptors and centers, and receive source-ordered simplex/witness output.
// It intentionally stops before EPA; touching/deep-overlap callers must route
// its retained simplex to common_ccd_source_intersection and a compatible EPA.
struct CommonCCDVertex {
  FlexDD3 point_a;
  FlexDD3 point_b;
  FlexDD3 minkowski;
  int index_a;
  int index_b;
};

struct CommonCCDGjkResult {
  FlexDD distance;
  FlexDD3 witness_a;
  FlexDD3 witness_b;
  int simplex_count;
  int iterations;
  int status;  // 0 complete, 1 separated/cutoff, 2 support error, 3 invalid
  int needs_intersection;  // simplex is contact-ready; EPA still required
  int support_count;
};

static_assert(sizeof(CommonCCDVertex)==116,
              "common GJK vertex ABI must be three FlexDD3 plus two ids");

static constant int COMMON_CCD_DIAG_SUPPORTS = 64;
static constant int COMMON_CCD_DIAG_WORDS_PER_SUPPORT = 38;
static constant int COMMON_CCD_DIAG_SUPPORT_OFFSET = 32;
static constant int COMMON_CCD_DIAG_EPA_SUPPORT_OFFSET = 2600;
static constant int COMMON_CCD_DIAG_EPA_SUPPORTS = 64;
static constant int COMMON_CCD_DIAG_EPA_FACE_OFFSET = 5032;
static constant int COMMON_CCD_DIAG_EPA_INIT_FACE_WORDS = 20;
static constant int COMMON_CCD_DIAG_EPA_ITER_OFFSET = 5152;
static constant int COMMON_CCD_DIAG_EPA_ITER_WORDS = 34;
static constant int COMMON_CCD_DIAG_EPA_ITERATIONS = 64;
static constant int COMMON_CCD_DIAG_COUNTS_OFFSET = 7328;
// The first source-GJK support record occupies slot 0 and normal trace records
// occupy slots 0..support_count-1. Slots 48..62 are reserved for five bounded
// tetrahedron-intersection records. Each iteration owns three existing
// 38-word slots (114 words); state uses 38 words and the candidate uses 66
// words beginning at offset 38. Slot 63 remains the truncation marker. Five
// records end exactly at that marker and below the EPA support buffer at 2600.
static constant int COMMON_CCD_DIAG_INTERSECTION_SLOT = 48;
static constant int COMMON_CCD_DIAG_INTERSECTION_RECORD_WORDS = 114;
static constant int COMMON_CCD_DIAG_INTERSECTION_ITERATIONS = 5;

static inline void common_ccd_diag_store_dd3(
    device float* words, int base, FlexDD3 value);

static inline void common_ccd_diag_store_support(
    device float* words, int base, FlexDD3 direction,
    thread const CommonCCDVertex& support) {
  common_ccd_diag_store_dd3(words,base,support.point_a);
  common_ccd_diag_store_dd3(words,base+9,support.point_b);
  common_ccd_diag_store_dd3(words,base+18,support.minkowski);
  words[base+27]=float(support.index_a);
  words[base+28]=float(support.index_b);
  common_ccd_diag_store_dd3(words,base+29,direction);
}

static inline void common_ccd_diag_epa_support(
    device float* trace, thread int& count, FlexDD3 direction,
    thread const CommonCCDVertex& support) {
  int slot=count++;
  if (trace!=nullptr && slot<COMMON_CCD_DIAG_EPA_SUPPORTS)
    common_ccd_diag_store_support(trace,
        slot*COMMON_CCD_DIAG_WORDS_PER_SUPPORT,direction,support);
}

static inline void common_ccd_diag_store_dd3(
    device float* words, int base, FlexDD3 value) {
  words[base+0]=value.x.hi; words[base+1]=value.x.lo;
  words[base+2]=value.x.tail; words[base+3]=value.y.hi;
  words[base+4]=value.y.lo; words[base+5]=value.y.tail;
  words[base+6]=value.z.hi; words[base+7]=value.z.lo;
  words[base+8]=value.z.tail;
}

static inline CommonCCDSupportResult common_ccd_object_support(
    thread CommonCCDSupportObject& object, FlexDD3 direction,
    device const float* mesh_vertices, device const int* mesh_graph) {
  CommonCCDSupportResult out;
  // Match pinned mjc_initCCDObj: meshes with fewer than
  // mjMESH_HILLCLIMB_MIN (10) vertices use exhaustive strict-max search even
  // when a compiled graph is present.  The graph walk can retain a different
  // vertex on support ties, changing the EPA seed topology.
  if (object.kind==COMMON_CCD_HULL && object.graph_offset>=0 &&
      object.vertex_count>=10) {
    int mesh_index=object.cached_vertex;
    out=common_ccd_hillclimb_mesh_support(object,direction,mesh_vertices,
                                          mesh_graph,mesh_index);
    if (out.status==0) object.cached_vertex=mesh_index;
    return out;
  }
  out=common_ccd_support(object,direction,mesh_vertices);
  if (out.status==0 && object.kind==COMMON_CCD_HULL)
    object.cached_vertex=out.selected_vertex;
  return out;
}

static inline int common_ccd_support_pair(
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD3 direction, device const float* mesh_vertices,
    device const int* mesh_graph, thread CommonCCDVertex& result) {
  CommonCCDSupportResult sa=common_ccd_object_support(
      a,direction,mesh_vertices,mesh_graph);
  if (sa.status) return 1;
  CommonCCDSupportResult sb=common_ccd_object_support(
      b,flex_dd3_neg(direction),mesh_vertices,mesh_graph);
  if (sb.status) return 2;
  result.point_a=sa.point;
  result.point_b=sb.point;
  result.minkowski=flex_dd3_sub(sa.point,sb.point);
  result.index_a=sa.selected_vertex;
  result.index_b=sb.selected_vertex;
  return 0;
}

static inline bool common_ccd_discrete_pair(
    thread const CommonCCDSupportObject& a,
    thread const CommonCCDSupportObject& b) {
  const int mesh=7, box=6, hfield=1;
  if (flex_dd_compare_exact(a.margin,flex_dd(0.0f))!=0 ||
      flex_dd_compare_exact(b.margin,flex_dd(0.0f))!=0) return false;
  bool a_discrete=a.geom_type==mesh || a.geom_type==box || a.geom_type==hfield;
  bool b_discrete=b.geom_type==mesh || b.geom_type==box || b.geom_type==hfield;
  return a_discrete && b_discrete;
}

static inline bool common_ccd_same3(FlexDD3 a, FlexDD3 b) {
  FlexDD tol=flex_dd(1.0e-15f);
  return flex_dd_compare_exact(flex_dd_abs(flex_dd_sub(a.x,b.x)),tol)<0 &&
         flex_dd_compare_exact(flex_dd_abs(flex_dd_sub(a.y,b.y)),tol)<0 &&
         flex_dd_compare_exact(flex_dd_abs(flex_dd_sub(a.z,b.z)),tol)<0;
}

static inline void common_ccd_reduced_simplex(
    thread CommonCCDVertex* simplex, thread FlexDD* lambda, int count,
    thread CommonCCDVertex* reduced, thread FlexDD* reduced_lambda,
    thread int& reduced_count) {
  reduced_count=0;
  for (int i=0;i<count;i++) {
    if (flex_dd_compare_exact(lambda[i],flex_dd(0.0f))==0) continue;
    reduced[reduced_count]=simplex[i];
    reduced_lambda[reduced_count]=lambda[i];
    reduced_count++;
  }
}

static inline FlexDD common_ccd_dot3(FlexDD3 a, FlexDD3 b) {
  return flex_dd3_dot(a,b);
}

static inline void common_ccd_subdistance(
    thread FlexDD* lambda, int n, thread const CommonCCDVertex* simplex) {
  for (int i=0;i<4;i++) lambda[i]=flex_dd(0.0f);
  if (n==4) flex_dd_s3d(lambda,simplex[0].minkowski,
      simplex[1].minkowski,simplex[2].minkowski,simplex[3].minkowski);
  else if (n==3) flex_dd_s2d(lambda,simplex[0].minkowski,
      simplex[1].minkowski,simplex[2].minkowski);
  else if (n==2) flex_dd_s1d(lambda,simplex[0].minkowski,
      simplex[1].minkowski);
  else lambda[0]=flex_dd(1.0f);
}

static inline FlexDD common_ccd_signed_face(
    thread FlexDD3& normal, CommonCCDVertex a, CommonCCDVertex b,
    CommonCCDVertex c) {
  FlexDD3 diff1=flex_dd3_sub(c.minkowski,a.minkowski);
  FlexDD3 diff2=flex_dd3_sub(b.minkowski,a.minkowski);
  normal=flex_dd3_cross(diff1,diff2);
  FlexDD n2=common_ccd_dot3(normal,normal);
  if (flex_dd_compare_exact(n2,flex_dd(1.0e-30f))>0 &&
      flex_dd_compare_exact(n2,flex_dd(1.0e30f))<0) {
    FlexDD n=flex_dd_sqrt(n2);
    normal=flex_dd3_div(normal,n);
    return common_ccd_dot3(normal,a.minkowski);
  }
  return flex_dd(3.402823466e+38f);
}

// Pinned gjkIntersect state transition for a source-ordered tetrahedron.
// Return 1 on intersection, 0 on separation, -1 if inconclusive, and -2 for
// malformed support input. The caller keeps the original simplex on -1.
static inline int common_ccd_source_intersection(
    thread const CommonCCDVertex* input, thread CommonCCDVertex* output,
    thread int& output_count, int iteration, int max_iterations,
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    device const float* mesh_vertices, device const int* mesh_graph,
    thread int& last_iteration, device float* support_trace,
    int main_support_count) {
  CommonCCDVertex simplex[4];
  for (int i=0;i<4;i++) simplex[i]=input[i];
  int order[4]={0,1,2,3};
  if (support_trace!=nullptr && main_support_count>
          COMMON_CCD_DIAG_INTERSECTION_SLOT) {
    support_trace[30]=-1.0f;
  }
  for (int k=iteration;k<max_iterations;k++) {
    if (support_trace!=nullptr && main_support_count<=
            COMMON_CCD_DIAG_INTERSECTION_SLOT &&
        k-iteration>=COMMON_CCD_DIAG_INTERSECTION_ITERATIONS) {
      int last=COMMON_CCD_DIAG_SUPPORT_OFFSET
          +(COMMON_CCD_DIAG_INTERSECTION_SLOT+
            (COMMON_CCD_DIAG_INTERSECTION_RECORD_WORDS
             /COMMON_CCD_DIAG_WORDS_PER_SUPPORT)*
            COMMON_CCD_DIAG_INTERSECTION_ITERATIONS)
              *COMMON_CCD_DIAG_WORDS_PER_SUPPORT;
      support_trace[last+37]=1.0f;
    }
    FlexDD3 normal[4];
    FlexDD distance[4];
    distance[0]=common_ccd_signed_face(normal[0],simplex[order[2]],
                                        simplex[order[1]],simplex[order[3]]);
    distance[1]=common_ccd_signed_face(normal[1],simplex[order[0]],
                                        simplex[order[2]],simplex[order[3]]);
    distance[2]=common_ccd_signed_face(normal[2],simplex[order[1]],
                                        simplex[order[0]],simplex[order[3]]);
    distance[3]=common_ccd_signed_face(normal[3],simplex[order[0]],
                                        simplex[order[1]],simplex[order[2]]);
    int trace_base=-1;
    if (support_trace!=nullptr && main_support_count<=
            COMMON_CCD_DIAG_INTERSECTION_SLOT &&
        k-iteration<COMMON_CCD_DIAG_INTERSECTION_ITERATIONS) {
      int slot=COMMON_CCD_DIAG_INTERSECTION_SLOT+
          (COMMON_CCD_DIAG_INTERSECTION_RECORD_WORDS
           /COMMON_CCD_DIAG_WORDS_PER_SUPPORT)*(k-iteration);
      trace_base=COMMON_CCD_DIAG_SUPPORT_OFFSET
          +slot*COMMON_CCD_DIAG_WORDS_PER_SUPPORT;
      support_trace[30]=float(k-iteration+1);
      int first=trace_base;
      support_trace[first+0]=float(k);
      for (int q=0;q<4;q++) support_trace[first+1+q]=float(order[q]);
      for (int q=0;q<4;q++) {
        support_trace[first+5+3*q+0]=distance[q].hi;
        support_trace[first+5+3*q+1]=distance[q].lo;
        support_trace[first+5+3*q+2]=distance[q].tail;
      }
      int second=first+COMMON_CCD_DIAG_WORDS_PER_SUPPORT;
      for (int q=0;q<4;q++)
        common_ccd_diag_store_dd3(support_trace,second+9*q,
                                  simplex[order[q]].minkowski);
    }
    if (flex_dd_compare_exact(distance[3],flex_dd(0.0f))==0 ||
        flex_dd_compare_exact(distance[2],flex_dd(0.0f))==0 ||
        flex_dd_compare_exact(distance[1],flex_dd(0.0f))==0 ||
        flex_dd_compare_exact(distance[0],flex_dd(0.0f))==0) {
      last_iteration=k;
      return -1;
    }
    int i=(flex_dd_compare_exact(distance[0],distance[1])<0) ? 0 : 1;
    int j=(flex_dd_compare_exact(distance[2],distance[3])<0) ? 2 : 3;
    int face=(flex_dd_compare_exact(distance[i],distance[j])<0) ? i : j;
    if (trace_base>=0) {
      support_trace[trace_base+17]=float(face);
      common_ccd_diag_store_dd3(support_trace,trace_base+18,normal[face]);
      for (int q=0;q<4;q++) {
        support_trace[trace_base+27+q]=float(simplex[order[q]].index_a);
        support_trace[trace_base+31+q]=float(simplex[order[q]].index_b);
      }
    }
    if (flex_dd_compare_exact(distance[face],flex_dd(0.0f))>0) {
      for (int q=0;q<4;q++) output[q]=simplex[order[q]];
      output_count=4;
      last_iteration=k;
      return 1;
    }
    CommonCCDVertex candidate;
    int support_status=common_ccd_support_pair(
        a,b,normal[face],mesh_vertices,mesh_graph,candidate);
    if (support_status) return -2;
    if (trace_base>=0) {
      int second=trace_base+COMMON_CCD_DIAG_WORDS_PER_SUPPORT;
      common_ccd_diag_store_dd3(support_trace,second+36,
                                candidate.minkowski);
      common_ccd_diag_store_dd3(support_trace,second+45,candidate.point_a);
      common_ccd_diag_store_dd3(support_trace,second+54,candidate.point_b);
      support_trace[second+63]=float(candidate.index_a);
      support_trace[second+64]=float(candidate.index_b);
      support_trace[second+65]=float(support_status);
    }
    simplex[order[face]]=candidate;
    if (flex_dd_compare_exact(
            common_ccd_dot3(normal[face],candidate.minkowski),
            flex_dd(0.0f))<0) {
      output_count=0;
      last_iteration=k;
      return 0;
    }
    int ia=(face+1)&3, ib=(face+2)&3;
    int swap=order[ia]; order[ia]=order[ib]; order[ib]=swap;
  }
  last_iteration=max_iterations;
  return -1;
}

// Source-order mjc_ccd::gjk distance pass. The caller supplies each object's
// pinned center() result (status.x1/x2), exact option scalars and per-query
// mutable support state. A zero distance cutoff triggers the pinned
// gjkIntersect fallback; this routine returns the retained simplex so the
// matching polytope initializer/EPA stage can consume it without requerying.
static inline CommonCCDGjkResult common_ccd_source_gjk(
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD3 center_a, FlexDD3 center_b, FlexDD tolerance,
    FlexDD distance_cutoff, int max_iterations,
    device const float* mesh_vertices, device const int* mesh_graph,
    thread CommonCCDVertex* output_simplex,
    device float* support_trace, int support_trace_capacity) {
  CommonCCDGjkResult out;
  out.distance=flex_dd(3.402823466e+38f);
  out.witness_a=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
  out.witness_b=out.witness_a;
  out.simplex_count=0;
  out.iterations=0;
  out.status=3;
  out.needs_intersection=0;
  out.support_count=0;
  if (max_iterations<0) return out;
  if (support_trace!=nullptr) support_trace[30]=0.0f;
  CommonCCDVertex simplex[4], reduced[4];
  FlexDD lambda[4]={flex_dd(1.0f),flex_dd(0.0f),
                    flex_dd(0.0f),flex_dd(0.0f)};
  FlexDD3 x=flex_dd3_sub(center_a,center_b);
  FlexDD tol2=flex_dd_mul(tolerance,tolerance);
  FlexDD cutoff2=flex_dd_mul(distance_cutoff,distance_cutoff);
  bool discrete=common_ccd_discrete_pair(a,b);
  FlexDD epsilon=discrete ? flex_dd(0.0f)
                          : flex_dd_mul(flex_dd(0.5f),tol2);
  FlexDD min_norm2=discrete ? flex_dd(1.0e-30f) : tol2;
  bool get_distance=flex_dd_compare_exact(distance_cutoff,flex_dd(0.0f))>0;
  bool backup=!get_distance;
  int count=0;
  FlexDD x_norm=flex_dd(0.0f);
  bool separated=false;
  bool exited=false;
  for (int k=0;k<max_iterations;k++) {
    out.iterations=k;
    FlexDD norm2=common_ccd_dot3(x,x);
    if (flex_dd_compare_exact(norm2,min_norm2)<0) {
      x_norm=flex_dd_sqrt(norm2);
      exited=true;
      break;
    }
    x_norm=flex_dd_sqrt(norm2);
    FlexDD3 direction=flex_dd3_div(x,x_norm);
    direction=flex_dd3_neg(direction);
    int support_status=common_ccd_support_pair(
        a,b,direction,mesh_vertices,mesh_graph,simplex[count]);
    if (support_status) { out.status=2; out.iterations=k; return out; }
    out.support_count++;
    // Once the intersection trace owns the reserved tail slots, later
    // primary iterations must not overwrite it if fallback was inconclusive.
    if (support_trace!=nullptr && k<support_trace_capacity
        && k<COMMON_CCD_DIAG_SUPPORTS
        && !(support_trace[30]>0.0f &&
             k>=COMMON_CCD_DIAG_INTERSECTION_SLOT)) {
      int base=COMMON_CCD_DIAG_SUPPORT_OFFSET
          +k*COMMON_CCD_DIAG_WORDS_PER_SUPPORT;
      common_ccd_diag_store_dd3(support_trace,base,
                                simplex[count].point_a);
      common_ccd_diag_store_dd3(support_trace,base+9,
                                simplex[count].point_b);
      common_ccd_diag_store_dd3(support_trace,base+18,
                                simplex[count].minkowski);
      support_trace[base+27]=float(simplex[count].index_a);
      support_trace[base+28]=float(simplex[count].index_b);
      common_ccd_diag_store_dd3(support_trace,base+29,direction);
    }
    FlexDD3 diff=flex_dd3_sub(x,simplex[count].minkowski);
    if (flex_dd_compare_exact(common_ccd_dot3(x,diff),epsilon)<0) {
      if (k==0) count=1;
      exited=true;
      break;
    }
    FlexDD vs=common_ccd_dot3(x,simplex[count].minkowski);
    if (!get_distance) {
      if (flex_dd_compare_exact(vs,flex_dd(0.0f))>0) {
        separated=true; out.iterations=k; exited=true; break;
      }
    } else if (flex_dd_compare_exact(distance_cutoff,
                                     flex_dd(3.402823466e+38f))<0) {
      if (flex_dd_compare_exact(vs,flex_dd(0.0f))>0 &&
          flex_dd_compare_exact(flex_dd_div(flex_dd_mul(vs,vs),norm2),
                                cutoff2)>=0) {
        separated=true; out.iterations=k; exited=true; break;
      }
    }
    if (count==3 && backup) {
      int intersection_iteration=k;
      CommonCCDVertex intersected[4]; int intersected_count=0;
      int intersection=common_ccd_source_intersection(
          simplex,intersected,intersected_count,k,max_iterations,a,b,
          mesh_vertices,mesh_graph,intersection_iteration,support_trace,
          out.support_count);
      if (intersection!=-1) {
        if (intersection<0) { out.status=2; out.iterations=intersection_iteration; return out; }
        out.iterations=intersection_iteration;
        if (intersection==0) { separated=true; exited=true; break; }
        for (int q=0;q<4;q++) simplex[q]=intersected[q];
        count=4; x_norm=flex_dd(0.0f); exited=true; break;
      }
      k=intersection_iteration;
      backup=false;
    }
    common_ccd_subdistance(lambda,count+1,simplex);
    int reduced_count=0;
    for (int i=0;i<4;i++) {
      if (flex_dd_compare_exact(lambda[i],flex_dd(0.0f))==0) continue;
      reduced[reduced_count]=simplex[i];
      FlexDD reduced_weight=lambda[i];
      if (reduced_count!=i) lambda[reduced_count]=reduced_weight;
      reduced_count++;
    }
    if (reduced_count<1) {
      out.status=1; out.iterations=k; out.simplex_count=0;
      out.distance=flex_dd(3.402823466e+38f); return out;
    }
    for (int i=0;i<reduced_count;i++) simplex[i]=reduced[i];
    count=reduced_count;
    FlexDD3 values[4];
    for (int i=0;i<count;i++) values[i]=simplex[i].minkowski;
    FlexDD3 next=flex_dd3_lincomb_source(values,lambda,count);
    if (common_ccd_same3(next,x)) { x=next; x_norm=flex_dd_sqrt(common_ccd_dot3(x,x)); exited=true; break; }
    x=next;
    if (count==4) { x_norm=flex_dd(0.0f); exited=true; break; }
  }
  if (!exited) out.iterations=max_iterations;
  out.status=separated ? 1 : 0;
  out.distance=separated ? flex_dd(3.402823466e+38f) : x_norm;
  out.simplex_count=separated ? 0 : count;
  out.needs_intersection=(!separated && count>1
      && flex_dd_compare_exact(x_norm,tolerance)<=0) ? 1 : 0;
  if (!separated) {
    FlexDD3 values_a[4], values_b[4];
    for (int i=0;i<count;i++) {
      output_simplex[i]=simplex[i];
      values_a[i]=simplex[i].point_a;
      values_b[i]=simplex[i].point_b;
    }
    out.witness_a=flex_dd3_lincomb_source(values_a,lambda,count);
    out.witness_b=flex_dd3_lincomb_source(values_b,lambda,count);
  }
  return out;
}

static inline FlexDD3 common_ccd_load_dd3(
    device const float* words, int base) {
  return FlexDD3{FlexDD{words[base],words[base+1],words[base+2]},
                 FlexDD{words[base+3],words[base+4],words[base+5]},
                 FlexDD{words[base+6],words[base+7],words[base+8]}};
}

static inline void common_ccd_store_dd3(
    device float* words, int base, FlexDD3 value) {
  words[base+0]=value.x.hi; words[base+1]=value.x.lo;
  words[base+2]=value.x.tail; words[base+3]=value.y.hi;
  words[base+4]=value.y.lo; words[base+5]=value.y.tail;
  words[base+6]=value.z.hi; words[base+7]=value.z.lo;
  words[base+8]=value.z.tail;
}

static inline CommonCCDSupportObject common_ccd_load_object(
    device const int* ints, device const float* words) {
  CommonCCDSupportObject value;
  value.kind=ints[0]; value.geom=ints[1]; value.geom_type=ints[2];
  value.vertex_offset=ints[3]; value.vertex_count=ints[4];
  value.cached_vertex=ints[5]; value.graph_offset=ints[6];
  value.graph_vertex_count=ints[7]; value.graph_word_count=ints[8];
  FlexDD3 pos=common_ccd_load_dd3(words,0);
  value.pos[0]=pos.x; value.pos[1]=pos.y; value.pos[2]=pos.z;
  for (int i=0;i<9;i++)
    value.mat[i]=FlexDD{words[9+3*i],words[10+3*i],words[11+3*i]};
  FlexDD3 size=common_ccd_load_dd3(words,36);
  value.size[0]=size.x; value.size[1]=size.y; value.size[2]=size.z;
  value.margin=FlexDD{words[45],words[46],words[47]};
  for (int i=0;i<6;i++) {
    int base=48+6*i;
    value.hfield_residual[i]=FlexDD3{
        FlexDD{words[base+0],words[base+1],0.0f},
        FlexDD{words[base+2],words[base+3],0.0f},
        FlexDD{words[base+4],words[base+5],0.0f}};
  }
  return value;
}

// Standalone native caller for the common source-GJK stage. Production rigid
// and flex candidate kernels can reuse this exact adapter/output contract;
// this probe entry intentionally does not provide an EPA contact witness.
kernel void common_ccd_source_gjk_probe(
    device const int* dims [[buffer(0)]],
    device const int* object_int [[buffer(1)]],
    device const float* object_words [[buffer(2)]],
    device const float* center_words [[buffer(3)]],
    device const float* option_words [[buffer(4)]],
    device const float* mesh_vertices [[buffer(5)]],
    device const int* mesh_graph [[buffer(6)]],
    device float* result [[buffer(7)]],
    device int* result_int [[buffer(8)]],
    device float* simplex_words [[buffer(9)]],
    device int* simplex_ids [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0];
  if (tid>=uint(batch)) return;
  int env=int(tid);
  CommonCCDSupportObject a=common_ccd_load_object(
      object_int+18*env,object_words+168*env);
  CommonCCDSupportObject b=common_ccd_load_object(
      object_int+18*env+9,object_words+168*env+84);
  FlexDD3 center_a=common_ccd_load_dd3(center_words+18*env,0);
  FlexDD3 center_b=common_ccd_load_dd3(center_words+18*env+9,0);
  FlexDD tolerance=FlexDD{option_words[6*env],option_words[6*env+1],
                          option_words[6*env+2]};
  FlexDD cutoff=FlexDD{option_words[6*env+3],option_words[6*env+4],
                       option_words[6*env+5]};
  CommonCCDVertex simplex[4];
  for (int i=0;i<4;i++) {
    simplex[i].point_a=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
    simplex[i].point_b=simplex[i].point_a;
    simplex[i].minkowski=simplex[i].point_a;
    simplex[i].index_a=-1; simplex[i].index_b=-1;
  }
  CommonCCDGjkResult value=common_ccd_source_gjk(
      a,b,center_a,center_b,tolerance,cutoff,dims[1],mesh_vertices,
      mesh_graph,simplex,nullptr,0);
  int out=21*env;
  result[out+0]=value.distance.hi; result[out+1]=value.distance.lo;
  result[out+2]=value.distance.tail;
  common_ccd_store_dd3(result,out+3,value.witness_a);
  common_ccd_store_dd3(result,out+12,value.witness_b);
  int meta=6*env;
  result_int[meta+0]=value.status; result_int[meta+1]=value.simplex_count;
  result_int[meta+2]=value.iterations;
  result_int[meta+3]=value.needs_intersection;
  result_int[meta+4]=a.cached_vertex; result_int[meta+5]=b.cached_vertex;
  for (int i=0;i<4;i++) {
    int base=(env*4+i)*27;
    CommonCCDVertex vertex_value=simplex[i];
    common_ccd_store_dd3(simplex_words,base,vertex_value.point_a);
    common_ccd_store_dd3(simplex_words,base+9,vertex_value.point_b);
    common_ccd_store_dd3(simplex_words,base+18,vertex_value.minkowski);
    simplex_ids[2*(env*4+i)]=vertex_value.index_a;
    simplex_ids[2*(env*4+i)+1]=vertex_value.index_b;
  }
}

// A contact-producing common EPA stage for retained four-vertex GJK seeds.
// This private adapter deliberately reports unsupported seed cardinalities;
// callers must not reinterpret them as separation. P2/P3 polytope expansion
// and the exact pinned contact dispatcher are still required before use.
struct CommonCCDEpaResult {
  FlexDD distance;
  int status;
  int3 face_vertices;
};

static inline FlexDDVertex common_ccd_flex_vertex(CommonCCDVertex value) {
  return FlexDDVertex{value.point_a,value.point_b,value.minkowski};
}

// Double-source predicates for EPA initialization. The older flex helper
// versions take float3 and can change P2/P3/P4 admission when a nearly
// coplanar simplex has meaningful low words. Preserve the pinned C operation
// sequence while rounding at each mjtNum (binary64) boundary.
static inline FlexDD common_ccd_affine_minor(
    FlexDD t0, FlexDD t1, FlexDD t2,
    FlexDD t3, FlexDD t4, FlexDD t5) {
  FlexDD value=flex_dd_source_add(t0,t1);
  value=flex_dd_source_add(value,t2);
  value=flex_dd_source_sub(value,t3);
  value=flex_dd_source_sub(value,t4);
  return flex_dd_source_sub(value,t5);
}

[[clang::noinline]] static inline bool common_ccd_tri_point_intersect(
    FlexDD3 a, FlexDD3 b, FlexDD3 c, FlexDD3 p) {
  FlexDD m14=flex_dd_source_sub(
      flex_dd_source_mul(b.y,c.z),flex_dd_source_mul(b.z,c.y));
  m14=flex_dd_source_sub(m14,flex_dd_source_mul(a.y,c.z));
  m14=flex_dd_source_add(m14,flex_dd_source_mul(a.z,c.y));
  m14=flex_dd_source_add(m14,flex_dd_source_mul(a.y,b.z));
  m14=flex_dd_source_sub(m14,flex_dd_source_mul(a.z,b.y));
  FlexDD m24=flex_dd_source_sub(
      flex_dd_source_mul(b.x,c.z),flex_dd_source_mul(b.z,c.x));
  m24=flex_dd_source_sub(m24,flex_dd_source_mul(a.x,c.z));
  m24=flex_dd_source_add(m24,flex_dd_source_mul(a.z,c.x));
  m24=flex_dd_source_add(m24,flex_dd_source_mul(a.x,b.z));
  m24=flex_dd_source_sub(m24,flex_dd_source_mul(a.z,b.x));
  FlexDD m34=flex_dd_source_sub(
      flex_dd_source_mul(b.x,c.y),flex_dd_source_mul(b.y,c.x));
  m34=flex_dd_source_sub(m34,flex_dd_source_mul(a.x,c.y));
  m34=flex_dd_source_add(m34,flex_dd_source_mul(a.y,c.x));
  m34=flex_dd_source_add(m34,flex_dd_source_mul(a.x,b.y));
  m34=flex_dd_source_sub(m34,flex_dd_source_mul(a.y,b.x));

  int x, y;
  FlexDD mmax;
  FlexDD am14=flex_dd_abs(m14), am24=flex_dd_abs(m24), am34=flex_dd_abs(m34);
  if (flex_dd_compare_exact(am14,am24)>=0 &&
      flex_dd_compare_exact(am14,am34)>=0) {
    x=1; y=2; mmax=m14;
  } else if (flex_dd_compare_exact(am24,am34)>=0) {
    x=0; y=2; mmax=m24;
  } else {
    x=0; y=1; mmax=m34;
  }
  if (flex_dd_compare_exact(mmax,flex_dd(0.0f))==0) return false;
  FlexDD av[3]={a.x,a.y,a.z}, bv[3]={b.x,b.y,b.z};
  FlexDD cv[3]={c.x,c.y,c.z}, pv[3]={p.x,p.y,p.z};
  FlexDD c31=common_ccd_affine_minor(
      flex_dd_source_mul(pv[x],bv[y]),flex_dd_source_mul(pv[y],cv[x]),
      flex_dd_source_mul(bv[x],cv[y]),flex_dd_source_mul(pv[x],cv[y]),
      flex_dd_source_mul(pv[y],bv[x]),flex_dd_source_mul(cv[x],bv[y]));
  FlexDD c32=common_ccd_affine_minor(
      flex_dd_source_mul(pv[x],cv[y]),flex_dd_source_mul(pv[y],av[x]),
      flex_dd_source_mul(cv[x],av[y]),flex_dd_source_mul(pv[x],av[y]),
      flex_dd_source_mul(pv[y],cv[x]),flex_dd_source_mul(av[x],cv[y]));
  FlexDD c33=common_ccd_affine_minor(
      flex_dd_source_mul(pv[x],av[y]),flex_dd_source_mul(pv[y],bv[x]),
      flex_dd_source_mul(av[x],bv[y]),flex_dd_source_mul(pv[x],bv[y]),
      flex_dd_source_mul(pv[y],av[x]),flex_dd_source_mul(bv[x],av[y]));
  FlexDD lambda0=flex_dd_source_div(c31,mmax);
  FlexDD lambda1=flex_dd_source_div(c32,mmax);
  FlexDD lambda2=flex_dd_source_div(c33,mmax);
  if (flex_dd_compare_exact(lambda0,flex_dd(0.0f))<0 ||
      flex_dd_compare_exact(lambda1,flex_dd(0.0f))<0 ||
      flex_dd_compare_exact(lambda2,flex_dd(0.0f))<0) return false;
  FlexDD3 projected;
  projected.x=flex_dd_source_add(
      flex_dd_source_add(flex_dd_source_mul(a.x,lambda0),
                         flex_dd_source_mul(b.x,lambda1)),
      flex_dd_source_mul(c.x,lambda2));
  projected.y=flex_dd_source_add(
      flex_dd_source_add(flex_dd_source_mul(a.y,lambda0),
                         flex_dd_source_mul(b.y,lambda1)),
      flex_dd_source_mul(c.y,lambda2));
  projected.z=flex_dd_source_add(
      flex_dd_source_add(flex_dd_source_mul(a.z,lambda0),
                         flex_dd_source_mul(b.z,lambda1)),
      flex_dd_source_mul(c.z,lambda2));
  FlexDD3 delta=flex_dd3_source_sub(projected,p);
  FlexDD length=flex_dd_sqrt(flex_dd3_dot(delta,delta));
  return flex_dd_compare_exact(length,flex_dd(1.0e-15f))<0;
}

[[clang::noinline]] static inline FlexDD common_ccd_det3_source(
    FlexDD3 a, FlexDD3 b, FlexDD3 c) {
  FlexDD x=flex_dd_source_mul(a.x,
      flex_dd_source_sub(flex_dd_source_mul(b.y,c.z),
                         flex_dd_source_mul(b.z,c.y)));
  FlexDD y=flex_dd_source_mul(a.y,
      flex_dd_source_sub(flex_dd_source_mul(b.z,c.x),
                         flex_dd_source_mul(b.x,c.z)));
  FlexDD z=flex_dd_source_mul(a.z,
      flex_dd_source_sub(flex_dd_source_mul(b.x,c.y),
                         flex_dd_source_mul(b.y,c.x)));
  return flex_dd_source_add(flex_dd_source_add(x,y),z);
}

[[clang::noinline]] static inline bool common_ccd_same_side_source(
    FlexDD3 p0, FlexDD3 p1, FlexDD3 p2, FlexDD3 p3) {
  FlexDD3 normal=flex_dd3_cross(flex_dd3_source_sub(p1,p0),
                               flex_dd3_source_sub(p2,p0));
  FlexDD d1=flex_dd3_dot(normal,flex_dd3_source_sub(p3,p0));
  FlexDD d2=flex_dd3_dot(normal,flex_dd3_neg(p0));
  return (flex_dd_compare_exact(d1,flex_dd(0.0f))>0 &&
          flex_dd_compare_exact(d2,flex_dd(0.0f))>0) ||
         (flex_dd_compare_exact(d1,flex_dd(0.0f))<0 &&
          flex_dd_compare_exact(d2,flex_dd(0.0f))<0);
}

[[clang::noinline]] static inline bool common_ccd_test_tetra_source(
    FlexDD3 a, FlexDD3 b, FlexDD3 c, FlexDD3 d) {
  return common_ccd_same_side_source(a,b,c,d) &&
         common_ccd_same_side_source(b,c,d,a) &&
         common_ccd_same_side_source(c,d,a,b) &&
         common_ccd_same_side_source(d,a,b,c);
}

[[clang::noinline]] static inline int common_ccd_ray_triangle_source(
    FlexDD3 v1, FlexDD3 v2, FlexDD3 v3,
    FlexDD3 v4, FlexDD3 v5) {
  FlexDD3 d12=flex_dd3_source_sub(v2,v1);
  FlexDD3 d13=flex_dd3_source_sub(v3,v1);
  FlexDD3 d14=flex_dd3_source_sub(v4,v1);
  FlexDD3 d15=flex_dd3_source_sub(v5,v1);
  FlexDD v0=common_ccd_det3_source(d13,d14,d12);
  FlexDD v1d=common_ccd_det3_source(d14,d15,d12);
  FlexDD v2d=common_ccd_det3_source(d15,d13,d12);
  bool nonnegative=flex_dd_compare_exact(v0,flex_dd(0.0f))>=0 &&
                   flex_dd_compare_exact(v1d,flex_dd(0.0f))>=0 &&
                   flex_dd_compare_exact(v2d,flex_dd(0.0f))>=0;
  bool nonpositive=flex_dd_compare_exact(v0,flex_dd(0.0f))<=0 &&
                   flex_dd_compare_exact(v1d,flex_dd(0.0f))<=0 &&
                   flex_dd_compare_exact(v2d,flex_dd(0.0f))<=0;
  return nonnegative ? 1 : nonpositive ? -1 : 0;
}

static inline int common_ccd_support_unit_pair(
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD3 raw_direction, device const float* mesh_vertices,
    device const int* mesh_graph, thread CommonCCDVertex& output,
    device float* epa_support_trace, thread int& epa_support_count) {
  FlexDD length=flex_dd_sqrt(flex_dd3_dot(raw_direction,raw_direction));
  FlexDD3 direction=flex_dd_compare_exact(length,flex_dd(1.0e-15f))>0
      ? flex_dd3_div(raw_direction,length)
      : FlexDD3{flex_dd(1.0f),flex_dd(0.0f),flex_dd(0.0f)};
  int status=common_ccd_support_pair(a,b,direction,mesh_vertices,mesh_graph,
                                     output);
  if (!status)
    common_ccd_diag_epa_support(epa_support_trace,epa_support_count,
                                direction,output);
  return status;
}

// Source-order polytope3 initializer shared by P3 and P2/P4 face fallback.
// It returns the pinned branch code family through a small integer and emits
// the canonical six faces only after the seed passes all admission checks.
static inline int common_ccd_initialize_p3(
    thread const CommonCCDVertex* seed,
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD gjk_distance, device const float* mesh_vertices,
    device const int* mesh_graph, device FlexDDVertex* vertices,
    device int2* vertex_ids,
    thread FlexDD3& center, thread int3* face_vertices,
    thread int3* adjacency, device float* epa_support_trace,
    thread int& epa_support_count) {
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
  if (common_ccd_support_unit_pair(a,b,flex_dd3_neg(normal),mesh_vertices,
                                  mesh_graph,v5,epa_support_trace,
                                  epa_support_count) ||
      common_ccd_support_unit_pair(a,b,normal,mesh_vertices,mesh_graph,v4,
                                  epa_support_trace,epa_support_count))
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

static inline bool common_ccd_face_below_source_limit(
    device const FlexDDVertex* vertices, FlexDD3 center, int3 f, int3 adj,
    FlexDD limit) {
  FlexEpaFace face;
  return !flex_epa_make_face_dd(vertices,center,f.x,f.y,f.z,adj,face) ||
         flex_dd_compare_exact(face.dist2_dd,limit)<0;
}

static inline CommonCCDEpaResult common_ccd_source_epa(
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
  FlexDD epa_tolerance=common_ccd_discrete_pair(a,b)
      ? FlexDD{1.0000000036274937e-15f,-3.627493647833322e-24f,0.0f}
      : tolerance;
  if (max_iterations<=0 || seed_count<2 || seed_count>4 ||
      vertex_capacity<6 || face_capacity<6 || horizon_capacity<3 ||
      horizon_stack_capacity<1) return out;
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
      int init=common_ccd_initialize_p3(p3seed,a,b,gjk_distance,
          mesh_vertices,mesh_graph,vertices,vertex_ids,center,face_vertices,
          adjacency,epa_support_trace,epa_support_count);
      if (init) { out.status=2; return out; }
      vertex_count=5; initial_faces=6; p3_initialized=true;
    }
    if (!p3_initialized &&
        !common_ccd_test_tetra_source(local_seed[0].minkowski,
                         local_seed[1].minkowski,
                         local_seed[2].minkowski,
                         local_seed[3].minkowski)) {
      out.status=2; return out;
    }
  } else if (seed_count==3) {
    int init=common_ccd_initialize_p3(local_seed,a,b,gjk_distance,
        mesh_vertices,mesh_graph,vertices,vertex_ids,center,face_vertices,
        adjacency,epa_support_trace,epa_support_count);
    if (init) { out.status=2; return out; }
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
      if (common_ccd_support_unit_pair(a,b,directions[i],mesh_vertices,
                                       mesh_graph,support,epa_support_trace,
                                       epa_support_count)) {
        out.status=2; return out;
      }
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
      int init=common_ccd_initialize_p3(p3seed,a,b,gjk_distance,
          mesh_vertices,mesh_graph,vertices,vertex_ids,center,face_vertices,
          adjacency,epa_support_trace,epa_support_count);
      if (init) { out.status=2; return out; }
      vertex_count=5; initial_faces=6; p3_initialized=true;
    }
    if (!p3_initialized && common_ccd_ray_triangle_source(
                          local_seed[0].minkowski,
                          local_seed[1].minkowski,
                          vertices[2].m,vertices[3].m,vertices[4].m)==0) {
      out.status=2; return out;
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
      out.status=2; return out;
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
  int epa_budget=min(max_iterations,1000);
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
    if (common_ccd_support_pair(a,b,direction,mesh_vertices,mesh_graph,
                                support)) {
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
    if (flex_dd_compare_exact(flex_dd_sub(upper,lower),epa_tolerance)<0) break;
    // Pinned mjc_ccd::epa stops after a repeated support feature for discrete
    // pairs (mesh/box/HField with zero margin). This test follows its distance
    // tolerance check and precedes horizon expansion. The rigid production
    // caller supplies the source vertex-ID sidecar; the standalone diagnostic
    // probe may omit it and therefore remains a trace-only path.
    if (common_ccd_discrete_pair(a,b) && vertex_ids!=nullptr) {
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
  if (best_face<0 || best_face>=face_count) {
    out.status=2; return out;
  }
  FlexEpaFace nearest=faces[best_face];
  flex_epa_write_witness_dd(nearest,vertices,contact);
  out.face_vertices=int3(nearest.a,nearest.b,nearest.c);
  out.distance=flex_dd_sqrt(nearest.dist2_dd);
  out.status=0;
  return out;
}


// Standalone composed diagnostic entry. The host must allocate each arena
// using the explicit per-world capacities in dims; this probe is not a public
// simulation caller and carries no admission policy.
kernel void common_ccd_source_gjk_epa_probe(
    device const int* dims [[buffer(0)]],
    device const int* object_int [[buffer(1)]],
    device const float* object_words [[buffer(2)]],
    device const float* center_words [[buffer(3)]],
    device const float* option_words [[buffer(4)]],
    device const float* mesh_vertices [[buffer(5)]],
    device const int* mesh_graph [[buffer(6)]],
    device float* result [[buffer(7)]],
    device int* result_int [[buffer(8)]],
    device float* simplex_words [[buffer(9)]],
    device int* simplex_ids [[buffer(10)]],
    device FlexDDVertex* epa_vertices [[buffer(11)]],
    device FlexEpaFace* epa_faces [[buffer(12)]],
    device int* epa_face_map [[buffer(13)]],
    device int2* epa_horizon [[buffer(14)]],
    device FlexHorizonFrame* epa_stack [[buffer(15)]],
    device float* contact_words [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
  int batch=dims[0];
  if (tid>=uint(batch)) return;
  int env=int(tid);
  int vertex_capacity=dims[2], face_capacity=dims[3];
  int horizon_capacity=dims[4], stack_capacity=dims[5];
  if (vertex_capacity<6 || face_capacity<6 || horizon_capacity<3 ||
      stack_capacity<1) {
    result_int[8*env]=3; return;
  }
  CommonCCDSupportObject a=common_ccd_load_object(
      object_int+18*env,object_words+168*env);
  CommonCCDSupportObject b=common_ccd_load_object(
      object_int+18*env+9,object_words+168*env+84);
  FlexDD3 center_a=common_ccd_load_dd3(center_words+18*env,0);
  FlexDD3 center_b=common_ccd_load_dd3(center_words+18*env+9,0);
  FlexDD tolerance=FlexDD{option_words[6*env],option_words[6*env+1],
                          option_words[6*env+2]};
  FlexDD cutoff=FlexDD{option_words[6*env+3],option_words[6*env+4],
                       option_words[6*env+5]};
  CommonCCDVertex simplex[4];
  for (int i=0;i<4;i++) {
    simplex[i].point_a=FlexDD3{flex_dd(0.0f),flex_dd(0.0f),flex_dd(0.0f)};
    simplex[i].point_b=simplex[i].point_a;
    simplex[i].minkowski=simplex[i].point_a;
    simplex[i].index_a=-1; simplex[i].index_b=-1;
  }
  CommonCCDGjkResult gjk=common_ccd_source_gjk(
      a,b,center_a,center_b,tolerance,cutoff,dims[1],mesh_vertices,
      mesh_graph,simplex,nullptr,0);
  int meta=8*env;
  result_int[meta+0]=gjk.status;
  result_int[meta+1]=gjk.simplex_count;
  result_int[meta+2]=gjk.iterations;
  result_int[meta+3]=gjk.needs_intersection;
  result_int[meta+4]=a.cached_vertex;
  result_int[meta+5]=b.cached_vertex;
  result_int[meta+6]=0;
  result_int[meta+7]=0;
  int out=21*env;
  result[out+0]=gjk.distance.hi; result[out+1]=gjk.distance.lo;
  result[out+2]=gjk.distance.tail;
  common_ccd_store_dd3(result,out+3,gjk.witness_a);
  common_ccd_store_dd3(result,out+12,gjk.witness_b);
  for (int i=0;i<4;i++) {
    int base=(env*4+i)*27;
    common_ccd_store_dd3(simplex_words,base,simplex[i].point_a);
    common_ccd_store_dd3(simplex_words,base+9,simplex[i].point_b);
    common_ccd_store_dd3(simplex_words,base+18,simplex[i].minkowski);
    simplex_ids[2*(env*4+i)]=simplex[i].index_a;
    simplex_ids[2*(env*4+i)+1]=simplex[i].index_b;
  }
  ContactGeom contact;
  contact.dist=3.402823466e+38f;
  contact.pos=float3(0.0f); contact.normal=float3(0.0f);
  contact.t1=float3(0.0f); contact.t2=float3(0.0f);
  int epa_status=1;
  int epa_support_count=0, epa_face_count=0;
  if (gjk.status==0 && gjk.needs_intersection && gjk.simplex_count>=2) {
    CommonCCDEpaResult epa=common_ccd_source_epa(
        simplex,gjk.simplex_count,a,b,dims[1],tolerance,gjk.distance,
        mesh_vertices,mesh_graph,
        epa_vertices+env*vertex_capacity,
        nullptr,
        epa_faces+env*face_capacity,
        epa_face_map+env*face_capacity,
        epa_horizon+env*horizon_capacity,
        epa_stack+env*stack_capacity,
        vertex_capacity,face_capacity,horizon_capacity,stack_capacity,
        contact,nullptr,nullptr,epa_support_count,epa_face_count);
    epa_status=epa.status;
    result[out]=epa.distance.hi; result[out+1]=epa.distance.lo;
    result[out+2]=epa.distance.tail;
  }
  result_int[meta+6]=epa_status;
  int cbase=16*env;
  contact_words[cbase+0]=contact.dist;
  contact_words[cbase+1]=contact.normal.x;
  contact_words[cbase+2]=contact.normal.y;
  contact_words[cbase+3]=contact.normal.z;
  contact_words[cbase+4]=contact.pos.x;
  contact_words[cbase+5]=contact.pos.y;
  contact_words[cbase+6]=contact.pos.z;
  contact_words[cbase+7]=contact.t1.x;
  contact_words[cbase+8]=contact.t1.y;
  contact_words[cbase+9]=contact.t1.z;
  contact_words[cbase+10]=contact.t2.x;
  contact_words[cbase+11]=contact.t2.y;
  contact_words[cbase+12]=contact.t2.z;
  contact_words[cbase+13]=gjk.distance.hi;
  contact_words[cbase+14]=gjk.distance.lo;
  contact_words[cbase+15]=gjk.distance.tail;
}
