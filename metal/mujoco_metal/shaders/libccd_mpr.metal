// MPR portal discovery, refinement and witnesses adapt pinned libccd mpr.c
// and vec3.c. Copyright (c) 2010-2012 Daniel Fiser <danfis@danfis.cz> and
// the Intelligent and Mobile Robotics Group, Czech Technical University in
// Prague. All rights reserved. See NOTICE for the retained BSD conditions.
// MuJoCo support callbacks adapt engine_collision_convex.c, Copyright 2021
// DeepMind Technologies Limited, Apache License 2.0.
//
// This alternate collision algorithm shares immutable compiled objects and
// source-precision operands with native CCD. It does not call GJK or EPA.

static inline bool mpr_zero(FlexDD a) {
  return flex_dd_compare_exact(flex_dd_abs(a),flex_dd(0x1p-52f))<0;
}
static inline bool mpr_eq(FlexDD a, FlexDD b) {
  FlexDD difference=flex_dd_abs(flex_dd_source_sub(a,b));
  if (mpr_zero(difference)) return true;
  a=flex_dd_abs(a); b=flex_dd_abs(b);
  return flex_dd_compare_exact(difference,flex_dd_source_mul(
      flex_dd(0x1p-52f),flex_dd_compare_exact(a,b)>0?a:b))<0;
}
static inline bool mpr_eq3(FlexDD3 a, FlexDD3 b) {
  return mpr_eq(a.x,b.x) && mpr_eq(a.y,b.y) && mpr_eq(a.z,b.z);
}
static inline FlexDD3 mpr_add(FlexDD3 a, FlexDD3 b) {
  return FlexDD3{flex_dd_source_add(a.x,b.x),flex_dd_source_add(a.y,b.y),
                 flex_dd_source_add(a.z,b.z)};
}
static inline FlexDD3 mpr_sub(FlexDD3 a, FlexDD3 b) {
  return flex_dd3_source_sub(a,b);
}
static inline FlexDD3 mpr_scale(FlexDD3 a, FlexDD b) {
  return FlexDD3{flex_dd_source_mul(a.x,b),flex_dd_source_mul(a.y,b),
                 flex_dd_source_mul(a.z,b)};
}
static inline FlexDD mpr_dot(FlexDD3 a, FlexDD3 b) {
  return flex_dd_source_fma(a.z,b.z,flex_dd_source_fma(a.y,b.y,
      flex_dd_source_mul(a.x,b.x)));
}
static inline FlexDD3 mpr_cross(FlexDD3 a, FlexDD3 b) {
  return FlexDD3{
      flex_dd_source_fma(a.y,b.z,flex_dd_neg(flex_dd_source_mul(a.z,b.y))),
      flex_dd_source_fma(a.z,b.x,flex_dd_neg(flex_dd_source_mul(a.x,b.z))),
      flex_dd_source_fma(a.x,b.y,flex_dd_neg(flex_dd_source_mul(a.y,b.x)))};
}
static inline FlexDD3 mpr_normalize(FlexDD3 a) {
  return mpr_scale(a,flex_dd_source_div(flex_dd(1),
      flex_dd_source_sqrt(mpr_dot(a,a))));
}
static inline FlexDD mpr_sign(FlexDD a) {
  return flex_dd(float(flex_dd_compare_exact(a,flex_dd(0))));
}

static __attribute__((noinline)) CommonCCDSupportResult mpr_object_support(
    thread CommonCCDSupportObject& object, FlexDD3 direction,
    device const float* vertices, device const int* graph) {
  CommonCCDSupportResult out;
  out.status=0;out.selected_vertex=-1;
  out.point=flex_dd3(float3(0));
  FlexDD3 local_dir=common_ccd_matrix_transpose_vector(object.mat,direction);
  FlexDD3 local=flex_dd3(float3(0));
  if (object.kind==COMMON_CCD_FLEX_ELEMENT || object.kind==COMMON_CCD_HFIELD_PRISM) {
    // These source callbacks are the same for native CCD and libccd.
    return common_ccd_support(object,direction,vertices);
  }
  if (object.kind==COMMON_CCD_SPHERE || object.kind==COMMON_CCD_CAPSULE) {
    local=mpr_scale(local_dir,object.size[0]);
    if (object.kind==COMMON_CCD_CAPSULE)
      local.z=flex_dd_source_add(local.z,flex_dd_source_mul(
          mpr_sign(local_dir.z),object.size[1]));
  } else if (object.kind==COMMON_CCD_ELLIPSOID) {
    local=FlexDD3{flex_dd_source_mul(local_dir.x,object.size[0]),
        flex_dd_source_mul(local_dir.y,object.size[1]),
        flex_dd_source_mul(local_dir.z,object.size[2])};
    FlexDD norm=flex_dd_source_sqrt(flex_dd_source_dot3(local,local));
    if (flex_dd_compare_exact(norm,flex_dd(1e-15f))<0) local=flex_dd3(float3(1,0,0));
    else local=mpr_scale(local,flex_dd_source_div(flex_dd(1),norm));
    local=FlexDD3{flex_dd_source_mul(local.x,object.size[0]),
        flex_dd_source_mul(local.y,object.size[1]),
        flex_dd_source_mul(local.z,object.size[2])};
  } else if (object.kind==COMMON_CCD_CYLINDER) {
    FlexDD norm=flex_dd_source_radial_norm(local_dir.x,local_dir.y);
    FlexDD minimum=FlexDD{1.0000000036274937e-15f,-3.627493647833322e-24f,0.0f};
    if (flex_dd_compare_exact(norm,minimum)>0) {
      local.x=flex_dd_source_mul(flex_dd_source_div(local_dir.x,norm),object.size[0]);
      local.y=flex_dd_source_mul(flex_dd_source_div(local_dir.y,norm),object.size[0]);
    }
    local.z=flex_dd_source_mul(mpr_sign(local_dir.z),object.size[1]);
  } else if (object.kind==COMMON_CCD_BOX) {
    local=FlexDD3{flex_dd_source_mul(mpr_sign(local_dir.x),object.size[0]),
        flex_dd_source_mul(mpr_sign(local_dir.y),object.size[1]),
        flex_dd_source_mul(mpr_sign(local_dir.z),object.size[2])};
  } else if (object.kind==COMMON_CCD_HULL) {
    if (object.vertex_offset<0 || object.vertex_count<1) {out.status=1;return out;}
    int best=0;
    if (object.graph_offset>=0 && object.vertex_count>=10) {
      int begin=object.graph_offset, n=graph[begin], nf=graph[begin+1];
      if (n!=object.graph_vertex_count || n<1 || nf<1 || n>object.vertex_count ||
          object.graph_word_count!=2+3*n+6*nf) {out.status=2;return out;}
      int edges=begin+2+2*n, globals=begin+2+n;
      best=object.cached_vertex<0?0:object.cached_vertex;
      if (best>=n) {out.status=3;return out;}
      int global=graph[globals+best];
      if (global<0 || global>=object.vertex_count) {out.status=4;return out;}
      FlexDD score=common_ccd_mesh_projection(local_dir,
          vertices+3*(object.vertex_offset+global));
      bool changed=true;int improvements=0;
      while (changed) {
        changed=false;int edge=graph[begin+2+best];int visits=0;
        while (true) {
          if (edge<0 || edge>=n+3*nf) {out.status=5;return out;}
          int candidate=graph[edges+edge++];
          if (candidate<0) break;
          if (candidate>=n || ++visits>n) {out.status=6;return out;}
          global=graph[globals+candidate];
          if (global<0 || global>=object.vertex_count) {out.status=7;return out;}
          FlexDD candidate_score=common_ccd_mesh_projection(local_dir,
              vertices+3*(object.vertex_offset+global));
          if (flex_dd_compare_exact(candidate_score,score)>0) {
            score=candidate_score;best=candidate;changed=true;
            if (++improvements>n) {out.status=8;return out;}
          }
        }
      }
      object.cached_vertex=best;
      best=graph[globals+best];
    } else {
      FlexDD score=flex_dd(-1e10f);best=-1;
      for (int candidate=0;candidate<object.vertex_count;candidate++) {
        FlexDD value=common_ccd_mesh_projection(local_dir,
            vertices+3*(object.vertex_offset+candidate));
        if (flex_dd_compare_exact(value,score)>0) {score=value;best=candidate;}
      }
      object.cached_vertex=best;
      if (best<0) {out.status=9;return out;}
    }
    local=flex_dd3(float3(vertices[3*(object.vertex_offset+best)],
        vertices[3*(object.vertex_offset+best)+1],vertices[3*(object.vertex_offset+best)+2]));
    out.selected_vertex=best;
  } else {out.status=10;return out;}
  // Unlike native support, libccd adds margin in the local frame, before
  // applying the original rotation and translation. Preserve both operations.
  local.x=flex_dd_source_add(local.x,flex_dd_source_mul(
      flex_dd_source_mul(local_dir.x,object.margin),flex_dd(.5f)));
  local.y=flex_dd_source_add(local.y,flex_dd_source_mul(
      flex_dd_source_mul(local_dir.y,object.margin),flex_dd(.5f)));
  local.z=flex_dd_source_add(local.z,flex_dd_source_mul(
      flex_dd_source_mul(local_dir.z,object.margin),flex_dd(.5f)));
  out.point=common_ccd_local_to_world(object.mat,object.pos,local);
  return out;
}

static __attribute__((noinline)) int mpr_support_pair(
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD3 direction, device const float* vertices, device const int* graph,
    thread CommonCCDVertex& support_point) {
  CommonCCDSupportResult va=mpr_object_support(a,direction,vertices,graph);
  CommonCCDSupportResult vb=mpr_object_support(b,flex_dd3_neg(direction),vertices,graph);
  if (va.status || vb.status) return -20;
  support_point.point_a=va.point;support_point.point_b=vb.point;
  support_point.minkowski=mpr_sub(va.point,vb.point);
  support_point.index_a=va.selected_vertex;support_point.index_b=vb.selected_vertex;
  return 0;
}
static inline FlexDD3 mpr_portal_direction(thread CommonCCDVertex* p) {
  return mpr_normalize(mpr_cross(mpr_sub(p[2].minkowski,p[1].minkowski),
      mpr_sub(p[3].minkowski,p[1].minkowski)));
}
static inline bool mpr_reached(thread CommonCCDVertex* p,
    thread const CommonCCDVertex& v,FlexDD3 direction,FlexDD tolerance) {
  FlexDD d=mpr_dot(v.minkowski,direction);
  FlexDD difference=flex_dd_source_sub(d,mpr_dot(p[1].minkowski,direction));
  for (int i=2;i<4;i++) {
    FlexDD next=flex_dd_source_sub(d,mpr_dot(p[i].minkowski,direction));
    if (flex_dd_compare_exact(next,difference)<0) difference=next;
  }
  return mpr_eq(difference,tolerance) || flex_dd_compare_exact(difference,tolerance)<0;
}
static inline void mpr_expand(thread CommonCCDVertex* p,
    thread const CommonCCDVertex& support_point) {
  FlexDD3 cross=mpr_cross(support_point.minkowski,p[0].minkowski);
  if (flex_dd_compare_exact(mpr_dot(p[1].minkowski,cross),flex_dd(0))>0) {
    p[flex_dd_compare_exact(mpr_dot(p[2].minkowski,cross),flex_dd(0))>0?1:3]=support_point;
  } else {
    p[flex_dd_compare_exact(mpr_dot(p[3].minkowski,cross),flex_dd(0))>0?2:1]=support_point;
  }
}

static __attribute__((noinline)) FlexDD3 mpr_segment_closest(FlexDD3 a,FlexDD3 b) {
  FlexDD3 direction=mpr_sub(b,a);
  FlexDD t=flex_dd_source_div(flex_dd_neg(mpr_dot(a,direction)),mpr_dot(direction,direction));
  if (mpr_zero(t) || flex_dd_compare_exact(t,flex_dd(0))<0) return a;
  if (mpr_eq(t,flex_dd(1)) || flex_dd_compare_exact(t,flex_dd(1))>0) return b;
  return mpr_add(mpr_scale(direction,t),a);
}
static __attribute__((noinline)) FlexDD3 mpr_triangle_closest(FlexDD3 a,FlexDD3 b,FlexDD3 c) {
  FlexDD3 d1=mpr_sub(b,a),d2=mpr_sub(c,a);
  FlexDD v=mpr_dot(d1,d1),w=mpr_dot(d2,d2),p=mpr_dot(a,d1),q=mpr_dot(a,d2),r=mpr_dot(d1,d2);
  // The pinned arm64 vec3.c contracts these C expressions. Rounding both
  // products separately changes closest-point branches on narrow portals.
  FlexDD determinant=flex_dd_source_fma(w,v,flex_dd_neg(flex_dd_source_mul(r,r)));
  FlexDD s=flex_dd(-1),t=flex_dd(-1);
  if (!mpr_zero(determinant)) {
    s=flex_dd_source_div(flex_dd_source_fma(q,r,flex_dd_neg(flex_dd_source_mul(w,p))),determinant);
    t=flex_dd_source_div(flex_dd_source_fma(flex_dd_neg(s),r,flex_dd_neg(q)),w);
  }
  FlexDD sum=flex_dd_source_add(s,t);
  if ((mpr_zero(s)||flex_dd_compare_exact(s,flex_dd(0))>0) &&
      (mpr_eq(s,flex_dd(1))||flex_dd_compare_exact(s,flex_dd(1))<0) &&
      (mpr_zero(t)||flex_dd_compare_exact(t,flex_dd(0))>0) &&
      (mpr_eq(t,flex_dd(1))||flex_dd_compare_exact(t,flex_dd(1))<0) &&
      (mpr_eq(sum,flex_dd(1))||flex_dd_compare_exact(sum,flex_dd(1))<0))
    return mpr_add(mpr_add(a,mpr_scale(d1,s)),mpr_scale(d2,t));
  FlexDD3 result=mpr_segment_closest(a,b),next=mpr_segment_closest(a,c);
  FlexDD distance=mpr_dot(result,result),next_distance=mpr_dot(next,next);
  if (flex_dd_compare_exact(next_distance,distance)<0) {result=next;distance=next_distance;}
  next=mpr_segment_closest(b,c);
  if (flex_dd_compare_exact(mpr_dot(next,next),distance)<0) result=next;
  return result;
}
static __attribute__((noinline)) FlexDD3 mpr_contact_position(thread CommonCCDVertex* p) {
  FlexDD weights[4];
  weights[0]=mpr_dot(mpr_cross(p[1].minkowski,p[2].minkowski),p[3].minkowski);
  weights[1]=mpr_dot(mpr_cross(p[3].minkowski,p[2].minkowski),p[0].minkowski);
  weights[2]=mpr_dot(mpr_cross(p[0].minkowski,p[1].minkowski),p[3].minkowski);
  weights[3]=mpr_dot(mpr_cross(p[2].minkowski,p[1].minkowski),p[0].minkowski);
  FlexDD sum=flex_dd_source_add(flex_dd_source_add(flex_dd_source_add(weights[0],weights[1]),weights[2]),weights[3]);
  if (mpr_zero(sum) || flex_dd_compare_exact(sum,flex_dd(0))<0) {
    FlexDD3 direction=mpr_portal_direction(p);
    weights[0]=flex_dd(0);
    weights[1]=mpr_dot(mpr_cross(p[2].minkowski,p[3].minkowski),direction);
    weights[2]=mpr_dot(mpr_cross(p[3].minkowski,p[1].minkowski),direction);
    weights[3]=mpr_dot(mpr_cross(p[1].minkowski,p[2].minkowski),direction);
    sum=flex_dd_source_add(flex_dd_source_add(weights[1],weights[2]),weights[3]);
  }
  FlexDD inverse=flex_dd_source_div(flex_dd(1),sum);
  FlexDD3 a=flex_dd3(float3(0)),b=a;
  for (int i=0;i<4;i++) {
    a=mpr_add(a,mpr_scale(p[i].point_a,weights[i]));
    b=mpr_add(b,mpr_scale(p[i].point_b,weights[i]));
  }
  return mpr_scale(mpr_add(mpr_scale(a,inverse),mpr_scale(b,inverse)),flex_dd(.5f));
}

static inline FlexDD3 mpr_mju_normalize(FlexDD3 a) {
  FlexDD norm=flex_dd_source_sqrt(flex_dd_source_dot3(a,a));
  if (flex_dd_compare_exact(norm,flex_dd(1e-15f))<0) return flex_dd3(float3(1,0,0));
  return mpr_scale(a,flex_dd_source_div(flex_dd(1),norm));
}
static inline FlexDD mpr_weighted_dot(FlexDD3 a,FlexDD3 b,thread const FlexDD* w) {
  return flex_dd_source_fma(flex_dd_source_mul(a.z,b.z),w[2],
      flex_dd_source_fma(flex_dd_source_mul(a.y,b.y),w[1],
          flex_dd_source_mul(flex_dd_source_mul(a.x,b.x),w[0])));
}
static __attribute__((noinline)) bool mpr_ellipsoid_normal(
    thread FlexDD3& normal,FlexDD3 position,thread const FlexDD* size) {
  FlexDD s2[3],inverse[3];
  for (int i=0;i<3;i++) {
    if (flex_dd_compare_exact(size[i],flex_dd(1e-15f))<0) return false;
    s2[i]=flex_dd_source_mul(size[i],size[i]);
    inverse[i]=flex_dd_source_div(flex_dd(1),s2[i]);
  }
  FlexDD c=flex_dd_source_sub(mpr_weighted_dot(position,position,inverse),flex_dd(1));
  FlexDD tolerance=flex_dd_source_div(flex_dd(1),flex_dd(1000000));
  if (flex_dd_compare_exact(c,flex_dd(0))<=0) {
    normal=mpr_mju_normalize(normal);
    for (int iteration=0;iteration<30;iteration++) {
      FlexDD a=mpr_weighted_dot(normal,normal,inverse);
      FlexDD b=mpr_weighted_dot(position,normal,inverse);
      FlexDD determinant=flex_dd_source_fma(b,b,flex_dd_neg(flex_dd_source_mul(a,c)));
      if (flex_dd_compare_exact(determinant,flex_dd(1e-15f))<0 ||
          flex_dd_compare_exact(a,flex_dd(1e-15f))<0) return iteration>0;
      FlexDD distance=flex_dd_source_div(flex_dd_source_sub(
          flex_dd_source_sqrt(determinant),b),a);
      if (flex_dd_compare_exact(distance,flex_dd(0))<0) return iteration>0;
      FlexDD3 point=flex_dd3_source_madd(normal,distance,position);
      FlexDD3 next=mpr_mju_normalize(FlexDD3{
          flex_dd_source_mul(point.x,inverse[0]),
          flex_dd_source_mul(point.y,inverse[1]),
          flex_dd_source_mul(point.z,inverse[2])});
      FlexDD3 delta=mpr_sub(normal,next);
      FlexDD change=flex_dd_source_sqrt(flex_dd_source_dot3(delta,delta));
      normal=next;
      if (flex_dd_compare_exact(change,tolerance)<0) break;
    }
  } else {
    FlexDD ps2[3]={flex_dd_source_mul(flex_dd_source_mul(position.x,position.x),s2[0]),
        flex_dd_source_mul(flex_dd_source_mul(position.y,position.y),s2[1]),
        flex_dd_source_mul(flex_dd_source_mul(position.z,position.z),s2[2])};
    FlexDD lambda=flex_dd(0);
    for (int iteration=0;iteration<30;iteration++) {
      FlexDD r[3],terms[3],derivative_terms[3];
      for (int i=0;i<3;i++) {
        r[i]=flex_dd_source_div(flex_dd(1),flex_dd_source_add(s2[i],lambda));
        terms[i]=flex_dd_source_mul(flex_dd_source_mul(ps2[i],r[i]),r[i]);
        derivative_terms[i]=flex_dd_source_mul(terms[i],r[i]);
      }
      FlexDD value=flex_dd_source_sub(flex_dd_source_add(
          flex_dd_source_add(terms[0],terms[1]),terms[2]),flex_dd(1));
      if (flex_dd_compare_exact(value,tolerance)<0) break;
      FlexDD derivative=flex_dd_source_mul(flex_dd(-2),flex_dd_source_add(
          flex_dd_source_add(derivative_terms[0],derivative_terms[1]),derivative_terms[2]));
      if (flex_dd_compare_exact(derivative,flex_dd(-1e-15f))>0) break;
      FlexDD delta=flex_dd_source_div(flex_dd_neg(value),derivative);
      if (flex_dd_compare_exact(delta,tolerance)<0) break;
      lambda=flex_dd_source_add(lambda,delta);
    }
    normal=mpr_mju_normalize(FlexDD3{
        flex_dd_source_div(position.x,flex_dd_source_add(s2[0],lambda)),
        flex_dd_source_div(position.y,flex_dd_source_add(s2[1],lambda)),
        flex_dd_source_div(position.z,flex_dd_source_add(s2[2],lambda))});
  }
  return true;
}
static __attribute__((noinline)) bool mpr_fix_normal_side(
    thread const CommonCCDSupportObject& object,FlexDD3 point,
    thread FlexDD3& normal) {
  int type=object.geom_type;
  if (type<2 || type>5) return false;
  FlexDD3 position=common_ccd_matrix_transpose_vector(object.mat,mpr_sub(point,
      FlexDD3{object.pos[0],object.pos[1],object.pos[2]}));
  FlexDD3 local=common_ccd_matrix_transpose_vector(object.mat,normal);
  if (type==2) local=position;
  else if (type==3) {
    local.x=position.x;local.y=position.y;local.z=flex_dd(0);
    if (flex_dd_compare_exact(position.z,flex_dd_neg(object.size[1]))<0)
      local.z=flex_dd_source_add(position.z,object.size[1]);
    else if (flex_dd_compare_exact(position.z,object.size[1])>0)
      local.z=flex_dd_source_sub(position.z,object.size[1]);
  } else if (type==4) {
    if (!mpr_ellipsoid_normal(local,position,object.size)) return false;
  } else {
    if (flex_dd_compare_exact(flex_dd_abs(position.z),flex_dd_source_mul(
        flex_dd_source_div(flex_dd(19),flex_dd(20)),object.size[1]))>0) return false;
    FlexDD flat=flex_dd_abs(flex_dd_source_sub(object.size[1],flex_dd_abs(position.z)));
    FlexDD round=flex_dd_abs(flex_dd_source_sub(object.size[0],flex_dd_source_sqrt(
        flex_dd_source_fma(position.x,position.x,flex_dd_source_mul(position.y,position.y)))));
    if (flex_dd_compare_exact(flat,flex_dd_source_mul(flex_dd(.25f),round))<0) return false;
    local=FlexDD3{position.x,position.y,flex_dd(0)};
  }
  normal=common_ccd_matrix_vector(object.mat,mpr_mju_normalize(local));
  return true;
}
static inline FlexDD3 common_libccd_fix_normal(
    thread const CommonCCDSupportObject& a,thread const CommonCCDSupportObject& b,
    FlexDD3 position,FlexDD3 normal) {
  FlexDD3 na=normal,nb=flex_dd3_neg(normal);
  bool fixed_a=mpr_fix_normal_side(a,position,na);
  bool fixed_b=mpr_fix_normal_side(b,position,nb);
  if (fixed_a && fixed_b) return mpr_mju_normalize(mpr_sub(na,nb));
  if (fixed_a) return na;
  if (fixed_b) return flex_dd3_neg(nb);
  return normal;
}

static __attribute__((noinline)) int common_libccd_penetration(
    thread CommonCCDSupportObject& a, thread CommonCCDSupportObject& b,
    FlexDD3 center_a,FlexDD3 center_b,FlexDD margin,int iterations,FlexDD tolerance,
    device const float* vertices,device const int* graph,
    thread ContactGeom& contact,thread FlexDD3& position,thread FlexDD3& normal,
    thread FlexDD* retained_distance=nullptr) {
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
      direction=mpr_portal_direction(portal);
      dot=mpr_dot(direction,portal[1].minkowski);
      if (mpr_zero(dot)||flex_dd_compare_exact(dot,flex_dd(0))>0) break;
      CommonCCDVertex next;
      if (mpr_support_pair(a,b,direction,vertices,graph,next)) return -20;
      dot=mpr_dot(next.minkowski,direction);
      if ((!mpr_zero(dot)&&flex_dd_compare_exact(dot,flex_dd(0))<=0)||
          mpr_reached(portal,next,direction,tolerance)) return 0;
      mpr_expand(portal,next);
    }
    int used=0;
    while (true) {
      direction=mpr_portal_direction(portal);
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
