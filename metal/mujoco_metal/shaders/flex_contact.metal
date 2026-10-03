#include <metal_stdlib>
using namespace metal;

// Candidate kinds match flex_contact.py. One thread owns one fixed contact
// feature, so independent obstacles can never overwrite each other's output.
static constant int FLEX_PLANE_VERTEX = 0;
static constant int FLEX_GEOM_VERTEX = 1;
static constant int FLEX_GEOM_ELEMENT = 2;
static constant int FLEX_INTERNAL = 3;
static constant int FLEX_ELEMENT_PAIR_VERTEX = 4;
static constant int FLEX_ELEMENT_PAIR_EDGE = 5;

static inline float3 quat_rotate(float4 q, float3 v) {
  float3 t = 2.0f * cross(q.yzw, v);
  return v + q.x * t + cross(q.yzw, t);
}

static inline float3 quat_unrotate(float4 q, float3 v) {
  q.yzw = -q.yzw;
  return quat_rotate(q, v);
}

static inline float sd_box(float3 p, float3 h, thread float3& normal) {
  float3 q = abs(p) - h;
  float3 outside = max(q, float3(0.0f));
  float out_len = length(outside);
  float inside = min(max(q.x, max(q.y, q.z)), 0.0f);
  if (out_len > 1e-12f) {
    float3 s = select(float3(-1.0f), float3(1.0f), p >= 0.0f);
    normal = normalize(outside * s);
  } else if (q.x >= q.y && q.x >= q.z) {
    normal = float3(p.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
  } else if (q.y >= q.z) {
    normal = float3(0.0f, p.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
  } else {
    normal = float3(0.0f, 0.0f, p.z >= 0.0f ? 1.0f : -1.0f);
  }
  return out_len + inside;
}

static inline float sd_geom(int type, float3 p, float3 size,
                            thread float3& normal) {
  if (type == 2) {  // sphere
    float r = length(p);
    normal = r > 1e-12f ? p / r : float3(0.0f, 0.0f, 1.0f);
    return r - size.x;
  }
  if (type == 3) {  // capsule, local z axis
    float z = clamp(p.z, -size.y, size.y);
    float3 d = p - float3(0.0f, 0.0f, z);
    float r = length(d);
    normal = r > 1e-12f ? d / r : float3(1.0f, 0.0f, 0.0f);
    return r - size.x;
  }
  if (type == 4) {  // ellipsoid approximation: exact on principal axes
    float3 scaled = p / max(size, float3(1e-12f));
    float r = length(scaled);
    normal = normalize(p / max(size*size, float3(1e-12f)));
    return (r - 1.0f) * min(size.x, min(size.y, size.z));
  }
  if (type == 5) {  // cylinder, local z axis
    float2 q = float2(length(p.xy) - size.x, abs(p.z) - size.y);
    float2 outside = max(q, float2(0.0f));
    float d = length(outside) + min(max(q.x, q.y), 0.0f);
    if (q.x >= q.y) {
      float r = max(length(p.xy), 1e-12f);
      normal = float3(p.xy / r, 0.0f);
    } else {
      normal = float3(0.0f, 0.0f, p.z >= 0.0f ? 1.0f : -1.0f);
    }
    return d;
  }
  if (type == 6) return sd_box(p, size, normal);
  normal = float3(0.0f, 0.0f, 1.0f);
  return INFINITY;
}

static inline float3 closest_triangle(float3 p, float3 a, float3 b, float3 c,
                                      thread float3& bary) {
  float3 ab = b-a, ac = c-a, ap = p-a;
  float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) { bary=float3(1,0,0); return a; }
  float3 bp=p-b; float d3=dot(ab,bp), d4=dot(ac,bp);
  if (d3 >= 0.0f && d4 <= d3) { bary=float3(0,1,0); return b; }
  float vc=d1*d4-d3*d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float v=d1/(d1-d3); bary=float3(1-v,v,0); return a+v*ab;
  }
  float3 cp=p-c; float d5=dot(ab,cp), d6=dot(ac,cp);
  if (d6 >= 0.0f && d5 <= d6) { bary=float3(0,0,1); return c; }
  float vb=d5*d2-d1*d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float w=d2/(d2-d6); bary=float3(1-w,0,w); return a+w*ac;
  }
  float va=d3*d6-d5*d4;
  if (va <= 0.0f && (d4-d3) >= 0.0f && (d5-d6) >= 0.0f) {
    float w=(d4-d3)/((d4-d3)+(d5-d6)); bary=float3(0,1-w,w); return b+w*(c-b);
  }
  float denom=1.0f/(va+vb+vc), v=vb*denom, w=vc*denom;
  bary=float3(1-v-w,v,w); return a+ab*v+ac*w;
}

static inline void closest_segments(float3 p1, float3 q1, float3 p2, float3 q2,
                                    thread float& s, thread float& t,
                                    thread float3& c1, thread float3& c2) {
  float3 d1=q1-p1, d2=q2-p2, r=p1-p2;
  float a=dot(d1,d1), e=dot(d2,d2), f=dot(d2,r);
  if (a <= 1e-20f && e <= 1e-20f) { s=0; t=0; c1=p1; c2=p2; return; }
  if (a <= 1e-20f) { s=0; t=clamp(f/max(e,1e-20f),0.0f,1.0f); }
  else {
    float c=dot(d1,r);
    if (e <= 1e-20f) { t=0; s=clamp(-c/a,0.0f,1.0f); }
    else {
      float b=dot(d1,d2), denom=a*e-b*b;
      s=abs(denom)>1e-20f ? clamp((b*f-c*e)/denom,0.0f,1.0f) : 0.0f;
      t=(b*s+f)/e;
      if (t<0.0f) { t=0.0f; s=clamp(-c/a,0.0f,1.0f); }
      else if (t>1.0f) { t=1.0f; s=clamp((b-c)/a,0.0f,1.0f); }
    }
  }
  c1=p1+d1*s; c2=p2+d2*t;
}

kernel void flex_contact_detect(
    device const int* candidate_kind [[buffer(0)]],
    device const int* candidate_flex1 [[buffer(1)]],
    device const int* candidate_elem1 [[buffer(2)]],
    device const int* candidate_vert1 [[buffer(3)]],
    device const int* candidate_flex2 [[buffer(4)]],
    device const int* candidate_elem2 [[buffer(5)]],
    device const int* candidate_vert2 [[buffer(6)]],
    device const int* candidate_geom [[buffer(7)]],
    device const int* candidate_nodes1 [[buffer(8)]],
    device const int* candidate_nodes2 [[buffer(9)]],
    device const float* flexvert_xpos [[buffer(10)]],
    device const float* geom_pos [[buffer(11)]],
    device const float* geom_quat [[buffer(12)]],
    device const float* geom_size [[buffer(13)]],
    device const int* geom_type [[buffer(14)]],
    device const float* flex_radius [[buffer(15)]],
    device const float* flex_radius2 [[buffer(16)]],
    device const float* candidate_margin_gap [[buffer(17)]],
    constant int* dims [[buffer(18)]],
    device int* active [[buffer(19)]],
    device float* distance [[buffer(20)]],
    device float* contact_pos [[buffer(21)]],
    device float* contact_normal [[buffer(22)]],
    device float* barycentric1 [[buffer(23)]],
    device float* barycentric2 [[buffer(24)]],
    uint tid [[thread_position_in_grid]]) {
  int batch = dims[0], nslot = dims[1], nvert = dims[2], ngeom = dims[3];
  if (tid >= uint(batch*nslot)) return;
  int slot = int(tid) % nslot, env = int(tid) / nslot;
  int kind = candidate_kind[slot];
  int f1 = candidate_flex1[slot];
  int v1 = candidate_vert1[slot], v2 = candidate_vert2[slot];
  int geom = candidate_geom[slot];
  float3 p=float3(0), q=float3(0), n=float3(0,0,1);
  float3 w1=float3(0), w2=float3(0);
  float dist=INFINITY;
  if (kind == FLEX_GEOM_ELEMENT && geom >= 0) {
    int nodebase=slot*4;
    int i0=candidate_nodes1[nodebase+0];
    int i1=candidate_nodes1[nodebase+1];
    int i2=candidate_nodes1[nodebase+2];
    if (i0 >= 0 && i1 >= 0 && i2 >= 0) {
      float3 gp=float3(geom_pos[(env*ngeom+geom)*3+0],
                       geom_pos[(env*ngeom+geom)*3+1],
                       geom_pos[(env*ngeom+geom)*3+2]);
      float4 quat=float4(geom_quat[(env*ngeom+geom)*4+0],
                         geom_quat[(env*ngeom+geom)*4+1],
                         geom_quat[(env*ngeom+geom)*4+2],
                         geom_quat[(env*ngeom+geom)*4+3]);
      float3 center_local=float3(0.0f);
      int gt=geom_type[geom];
      if (gt==3) center_local.z=0.0f;
      float3 center=gp+quat_rotate(quat,center_local);
      float3 a=float3(flexvert_xpos[(env*nvert+i0)*3+0],flexvert_xpos[(env*nvert+i0)*3+1],flexvert_xpos[(env*nvert+i0)*3+2]);
      float3 bb=float3(flexvert_xpos[(env*nvert+i1)*3+0],flexvert_xpos[(env*nvert+i1)*3+1],flexvert_xpos[(env*nvert+i1)*3+2]);
      float3 c=float3(flexvert_xpos[(env*nvert+i2)*3+0],flexvert_xpos[(env*nvert+i2)*3+1],flexvert_xpos[(env*nvert+i2)*3+2]);
      float3 bary;
      p=closest_triangle(center,a,bb,c,bary);
      float3 local=quat_unrotate(quat,p-gp);
      float3 nl;
      float3 size=float3(geom_size[geom*3+0],geom_size[geom*3+1],geom_size[geom*3+2]);
      float sd=sd_geom(gt,local,size,nl);
      n=quat_rotate(quat,nl);
      dist=sd-flex_radius[f1];
      q=p-n*(0.5f*dist+flex_radius[f1]);
      w1=bary;
    }
  } else if ((kind == FLEX_PLANE_VERTEX || kind == FLEX_GEOM_VERTEX) && geom >= 0 && v1 >= 0) {
    p=float3(flexvert_xpos[(env*nvert+v1)*3+0],
             flexvert_xpos[(env*nvert+v1)*3+1],
             flexvert_xpos[(env*nvert+v1)*3+2]);
    float3 gp=float3(geom_pos[(env*ngeom+geom)*3+0],
                     geom_pos[(env*ngeom+geom)*3+1],
                     geom_pos[(env*ngeom+geom)*3+2]);
    float4 gq=float4(geom_quat[(env*ngeom+geom)*4+0],
                     geom_quat[(env*ngeom+geom)*4+1],
                     geom_quat[(env*ngeom+geom)*4+2],
                     geom_quat[(env*ngeom+geom)*4+3]);
    float3 local=quat_unrotate(gq,p-gp);
    int gt=geom_type[geom];
    float3 nl;
    if (kind == FLEX_PLANE_VERTEX || gt == 0) {
      nl=float3(0,0,1);
      n=quat_rotate(gq,nl);
      dist=dot(local,nl)-flex_radius[f1];
      q=p-n*(0.5f*dist+flex_radius[f1]);
    } else {
      float3 size=float3(geom_size[geom*3+0],geom_size[geom*3+1],geom_size[geom*3+2]);
      float sd=sd_geom(gt,local,size,nl);
      n=quat_rotate(gq,nl);
      dist=sd-flex_radius[f1];
      q=p-n*(0.5f*dist+flex_radius[f1]);
    }
    w1.x=1.0f;
  } else if ((kind == FLEX_ELEMENT_PAIR_VERTEX || kind == FLEX_INTERNAL)
             && ((v1 >= 0) || (v2 >= 0))) {
    bool reverse=(v1 < 0);
    int point_vert=reverse ? v2 : v1;
    p=float3(flexvert_xpos[(env*nvert+point_vert)*3+0],
             flexvert_xpos[(env*nvert+point_vert)*3+1],
             flexvert_xpos[(env*nvert+point_vert)*3+2]);
    int nodebase=slot*4;
    int i0=reverse ? candidate_nodes1[nodebase+0] : candidate_nodes2[nodebase+0];
    int i1=reverse ? candidate_nodes1[nodebase+1] : candidate_nodes2[nodebase+1];
    int i2=reverse ? candidate_nodes1[nodebase+2] : candidate_nodes2[nodebase+2];
    if (i0 >= 0 && i1 >= 0 && i2 >= 0) {
      float3 a=float3(flexvert_xpos[(env*nvert+i0)*3+0],flexvert_xpos[(env*nvert+i0)*3+1],flexvert_xpos[(env*nvert+i0)*3+2]);
      float3 b=float3(flexvert_xpos[(env*nvert+i1)*3+0],flexvert_xpos[(env*nvert+i1)*3+1],flexvert_xpos[(env*nvert+i1)*3+2]);
      float3 c=float3(flexvert_xpos[(env*nvert+i2)*3+0],flexvert_xpos[(env*nvert+i2)*3+1],flexvert_xpos[(env*nvert+i2)*3+2]);
      float3 bary; q=closest_triangle(p,a,b,c,bary);
      w1.x=1.0f; w2=bary;
      if (reverse) { w1=bary; w2.x=1.0f; }
      float3 delta=p-q; float len=length(delta);
      n=len > 1e-12f ? delta/len : normalize(cross(b-a,c-a));
      dist=len-flex_radius[f1]-flex_radius2[slot];
      q=0.5f*(p+q);
    }
  } else if (kind == FLEX_ELEMENT_PAIR_EDGE) {
    int base=slot*4;
    int a=candidate_nodes1[base], b=candidate_nodes1[base+1];
    int c=candidate_nodes2[base], d=candidate_nodes2[base+1];
    if (a>=0 && b>=0 && c>=0 && d>=0) {
      float3 p1=float3(flexvert_xpos[(env*nvert+a)*3+0],flexvert_xpos[(env*nvert+a)*3+1],flexvert_xpos[(env*nvert+a)*3+2]);
      float3 q1=float3(flexvert_xpos[(env*nvert+b)*3+0],flexvert_xpos[(env*nvert+b)*3+1],flexvert_xpos[(env*nvert+b)*3+2]);
      float3 p2=float3(flexvert_xpos[(env*nvert+c)*3+0],flexvert_xpos[(env*nvert+c)*3+1],flexvert_xpos[(env*nvert+c)*3+2]);
      float3 q2=float3(flexvert_xpos[(env*nvert+d)*3+0],flexvert_xpos[(env*nvert+d)*3+1],flexvert_xpos[(env*nvert+d)*3+2]);
      float s,t; float3 c1,c2;
      closest_segments(p1,q1,p2,q2,s,t,c1,c2);
      float3 delta=c1-c2; float len=length(delta);
      n=len>1e-12f ? delta/len : normalize(cross(q1-p1,q2-p2));
      dist=len-flex_radius[f1]-flex_radius2[slot];
      w1=float3(1.0f-s,s,0.0f); w2=float3(1.0f-t,t,0.0f);
      q=0.5f*((c1-n*flex_radius[f1])+(c2+n*flex_radius2[slot]));
    }
  }
  float threshold=candidate_margin_gap[2*slot]+candidate_margin_gap[2*slot+1];
  int hit=isfinite(dist) && dist <= threshold;
  active[tid]=hit;
  distance[tid]=isfinite(dist) ? dist : 0.0f;
  contact_pos[3*tid+0]=hit ? q.x : 0.0f;
  contact_pos[3*tid+1]=hit ? q.y : 0.0f;
  contact_pos[3*tid+2]=hit ? q.z : 0.0f;
  contact_normal[3*tid+0]=hit ? n.x : 0.0f;
  contact_normal[3*tid+1]=hit ? n.y : 0.0f;
  contact_normal[3*tid+2]=hit ? n.z : 0.0f;
  for (int i=0; i<4; ++i) {
    barycentric1[4*tid+i]=(hit && i<3) ? w1[i] : 0.0f;
    barycentric2[4*tid+i]=(hit && i<3) ? w2[i] : 0.0f;
  }
}

// Fixed-capacity flex-contact wake links. Static lowering deduplicates all
// potential tree pairs; this kernel retains each pair iff any owning slot is
// active. It writes deterministic links and never reads a device count back.
kernel void flex_contact_tree_links(
    device const bool* active [[buffer(0)]],
    device const int* candidate_link_ids [[buffer(1)]],
    device const int* link_tree_pairs [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    device int* links [[buffer(4)]],
    device int* overflow [[buffer(5)]],
    uint world [[thread_position_in_grid]]) {
  int batch=dims[0], nslot=dims[1], max_per_slot=dims[2];
  int nlinks=dims[3], capacity=dims[4];
  if (world >= uint(batch)) return;
  int emitted=0, over=0;
  for (int link=0; link<nlinks; ++link) {
    bool live=false;
    for (int slot=0; slot<nslot && !live; ++slot) {
      if (!active[world*nslot+slot]) continue;
      for (int j=0; j<max_per_slot; ++j) {
        if (candidate_link_ids[slot*max_per_slot+j] == link) {
          live=true;
          break;
        }
      }
    }
    if (!live) continue;
    if (emitted < capacity) {
      links[world*capacity+emitted*2+0]=link_tree_pairs[2*link+0];
      links[world*capacity+emitted*2+1]=link_tree_pairs[2*link+1];
      ++emitted;
    } else {
      over=1;
    }
  }
  for (int i=emitted; i<capacity; ++i) {
    links[world*capacity+i*2+0]=-1;
    links[world*capacity+i*2+1]=-1;
  }
  overflow[world]=over;
}
