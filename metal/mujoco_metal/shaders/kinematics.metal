// Copyright 2026 The MuJoCo Metal contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>
using namespace metal;

inline float4 qmul(float4 a, float4 b) {
  return float4(a.x*b.x-dot(a.yzw, b.yzw),
      a.x*b.yzw+b.x*a.yzw+cross(a.yzw, b.yzw));
}
inline float3 qrot(float4 q, float3 v) {
  return v + 2.0f*cross(q.yzw, cross(q.yzw, v) + q.x*v);
}
inline float4 qnorm(float4 q) { return q * rsqrt(dot(q, q)); }
inline float4 axisq(float3 axis, float angle) {
  axis = normalize(axis);
  return float4(cos(angle*0.5f), axis*sin(angle*0.5f));
}
struct KinDD { float hi; float lo; float tail; };
struct KinDD3 { KinDD x; KinDD y; KinDD z; };
struct KinQ { KinDD w; KinDD x; KinDD y; KinDD z; };
struct KinMat { KinDD m[9]; };
inline KinDD kin_dd(float x) { return KinDD{x, 0.0f, 0.0f}; }
inline KinDD kin_add(KinDD a, KinDD b) {
  float s=a.hi+b.hi;
  float bv=s-a.hi;
  float e=(a.hi-(s-bv))+(b.hi-bv);
  float t=a.lo+b.lo;
  float tv=t-a.lo;
  float te=(a.lo-(t-tv))+(b.lo-tv);
  float u=e+t;
  float uv=u-e;
  float ue=(e-(u-uv))+(t-uv);
  float h=s+u;
  float hv=h-s;
  float he=(s-(h-hv))+(u-hv);
  return KinDD{h,he,ue+te+a.tail+b.tail};
}
inline KinDD kin_neg(KinDD a) { return KinDD{-a.hi,-a.lo,-a.tail}; }
inline KinDD kin_sub(KinDD a, KinDD b) { return kin_add(a,kin_neg(b)); }
inline KinDD kin_mul(KinDD a, KinDD b) {
  KinDD out=kin_dd(0.0f);
  float av[3]={a.hi,a.lo,a.tail};
  float bv[3]={b.hi,b.lo,b.tail};
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
    float p=av[i]*bv[j];
    float e=fma(av[i],bv[j],-p);
    out=kin_add(out,KinDD{p,e,0.0f});
  }
  return out;
}
inline KinDD kin_scale(KinDD a, float b) { return kin_mul(a,kin_dd(b)); }
inline KinDD kin_sub(KinDD a, KinDD b);
inline float kin_value(KinDD a) { return (a.hi+a.lo)+a.tail; }
inline int kin_sign(KinDD a) {
  if (a.hi>0.0f) return 1;
  if (a.hi<0.0f) return -1;
  if (a.lo>0.0f) return 1;
  if (a.lo<0.0f) return -1;
  if (a.tail>0.0f) return 1;
  if (a.tail<0.0f) return -1;
  return 0;
}
inline int kin_compare(KinDD a, KinDD b) { return kin_sign(kin_sub(a,b)); }
inline bool kin_equal(KinDD a, KinDD b) {
  KinDD d=kin_sub(a,b);
  return d.hi==0.0f && d.lo==0.0f && d.tail==0.0f;
}
inline KinDD kin_div(KinDD a, KinDD b) {
  float q0=a.hi/b.hi;
  KinDD r0=kin_sub(a,kin_mul(b,kin_dd(q0)));
  KinDD q1=kin_add(kin_dd(r0.hi/b.hi),kin_dd(r0.lo/b.hi));
  q1=kin_add(q1,kin_dd(r0.tail/b.hi));
  KinDD q=kin_add(kin_dd(q0),q1);
  KinDD r1=kin_sub(a,kin_mul(b,q));
  KinDD q2=kin_add(kin_dd(r1.hi/b.hi),kin_dd(r1.lo/b.hi));
  q2=kin_add(q2,kin_dd(r1.tail/b.hi));
  return kin_add(q,q2);
}
inline KinDD kin_sqrt(KinDD a) {
  float r0=sqrt(max(a.hi,0.0f));
  if (r0==0.0f) return kin_dd(0.0f);
  KinDD root=kin_dd(r0);
  KinDD residual=kin_sub(a,kin_mul(root,root));
  KinDD correction=kin_div(residual,kin_dd(2.0f*r0));
  root=kin_add(root,correction);
  residual=kin_sub(a,kin_mul(root,root));
  correction=kin_div(residual,kin_dd(2.0f*r0));
  return kin_add(root,correction);
}
inline KinDD3 kin3(float3 hi, float3 lo, float3 tail) {
  return KinDD3{KinDD{hi.x,lo.x,tail.x},KinDD{hi.y,lo.y,tail.y},
                KinDD{hi.z,lo.z,tail.z}};
}
inline KinDD3 kin3(float3 hi, float3 lo) {
  return kin3(hi,lo,float3(0.0f));
}
inline KinDD3 kin3_add(KinDD3 a, KinDD3 b) {
  return KinDD3{kin_add(a.x,b.x),kin_add(a.y,b.y),kin_add(a.z,b.z)};
}
inline KinDD3 kin3_sub(KinDD3 a, KinDD3 b) {
  return KinDD3{kin_sub(a.x,b.x),kin_sub(a.y,b.y),kin_sub(a.z,b.z)};
}
inline KinDD3 kin3_scale(KinDD3 a, float b) {
  return KinDD3{kin_scale(a.x,b),kin_scale(a.y,b),kin_scale(a.z,b)};
}
inline KinDD3 kin3_cross_dd(KinDD3 a, KinDD3 b) {
  return KinDD3{
    kin_sub(kin_mul(a.y,b.z),kin_mul(a.z,b.y)),
    kin_sub(kin_mul(a.z,b.x),kin_mul(a.x,b.z)),
    kin_sub(kin_mul(a.x,b.y),kin_mul(a.y,b.x))};
}
inline KinQ kinq(float4 hi, float4 lo, float4 tail) {
  return KinQ{KinDD{hi.x,lo.x,tail.x},KinDD{hi.y,lo.y,tail.y},
              KinDD{hi.z,lo.z,tail.z},KinDD{hi.w,lo.w,tail.w}};
}
inline KinQ kinq(float4 q) { return kinq(q,float4(0.0f),float4(0.0f)); }
inline float4 kinq_value(KinQ q) {
  return float4(kin_value(q.w),kin_value(q.x),kin_value(q.y),kin_value(q.z));
}
inline KinQ kinq_conj(KinQ q) {
  return KinQ{q.w,kin_neg(q.x),kin_neg(q.y),kin_neg(q.z)};
}
inline bool kinq_equal(KinQ a, KinQ b) {
  return kin_equal(a.w,b.w) && kin_equal(a.x,b.x)
      && kin_equal(a.y,b.y) && kin_equal(a.z,b.z);
}
inline KinQ kinq_mul(KinQ a, KinQ b) {
  KinQ out;
  out.w=kin_sub(kin_sub(kin_sub(kin_mul(a.w,b.w),kin_mul(a.x,b.x)),
                        kin_mul(a.y,b.y)),kin_mul(a.z,b.z));
  out.x=kin_sub(kin_add(kin_add(kin_mul(a.w,b.x),kin_mul(a.x,b.w)),
                        kin_mul(a.y,b.z)),kin_mul(a.z,b.y));
  out.y=kin_add(kin_add(kin_sub(kin_mul(a.w,b.y),kin_mul(a.x,b.z)),
                        kin_mul(a.y,b.w)),kin_mul(a.z,b.x));
  out.z=kin_add(kin_sub(kin_add(kin_mul(a.w,b.z),kin_mul(a.x,b.y)),
                        kin_mul(a.y,b.x)),kin_mul(a.z,b.w));
  return out;
}
inline KinQ kinq_norm(KinQ q) {
  // Match engine_util_blas.c:mju_normalize4's left-to-right binary64
  // expression: (((w*w + x*x) + y*y) + z*z). Pairwise grouping changes the
  // final source-rounded norm for compiled quaternions with small residuals.
  KinDD n2=kin_add(kin_add(kin_add(kin_mul(q.w,q.w),kin_mul(q.x,q.x)),
                           kin_mul(q.y,q.y)),kin_mul(q.z,q.z));
  KinDD n=kin_sqrt(n2);
  // mjMINVAL is a binary64 constant. Keep its residual so the source
  // normalize/no-op branch is decided from represented words, not float32.
  KinDD minval=KinDD{1.0000000036274937e-15f,
                     -3.627493647833322e-24f,0.0f};
  if (kin_compare(n,minval)<0) return kinq(float4(1,0,0,0));
  KinDD delta=kin_sub(n,kin_dd(1.0f));
  if (kin_sign(delta)<0) delta=kin_neg(delta);
  if (kin_compare(delta,minval)<=0) return q;
  // mju_normalize4 computes one reciprocal and multiplies every component
  // by that value; four independent divisions do not preserve its operation
  // sequence near source-precision normalization boundaries.
  KinDD norm_inv=kin_div(kin_dd(1.0f),n);
  return KinQ{kin_mul(q.w,norm_inv),kin_mul(q.x,norm_inv),
              kin_mul(q.y,norm_inv),kin_mul(q.z,norm_inv)};
}
inline KinDD3 kin3_scale_dd(KinDD3 a, float b) {
  return KinDD3{kin_scale(a.x,b),kin_scale(a.y,b),kin_scale(a.z,b)};
}
inline KinDD3 kinq_rotate(KinQ q, KinDD3 v) {
  // Keep engine_inline.h:mji_rotVecQuat's component operation order. This
  // helper is used for joint axes/anchors; body and attached-frame placement
  // instead uses the already-published xmat as mj_local2Global does.
  KinDD tx=kin_sub(kin_add(kin_mul(q.w,v.x),kin_mul(q.y,v.z)),
                   kin_mul(q.z,v.y));
  KinDD ty=kin_sub(kin_add(kin_mul(q.w,v.y),kin_mul(q.z,v.x)),
                   kin_mul(q.x,v.z));
  KinDD tz=kin_sub(kin_add(kin_mul(q.w,v.z),kin_mul(q.x,v.y)),
                   kin_mul(q.y,v.x));
  return KinDD3{
    kin_add(v.x,kin_scale(kin_sub(kin_mul(q.y,tz),kin_mul(q.z,ty)),2.0f)),
    kin_add(v.y,kin_scale(kin_sub(kin_mul(q.z,tx),kin_mul(q.x,tz)),2.0f)),
    kin_add(v.z,kin_scale(kin_sub(kin_mul(q.x,ty),kin_mul(q.y,tx)),2.0f))};
}
inline KinMat kinq_matrix(KinQ q) {
  KinMat out;
  // Match mju_quat2Mat's exact null-quaternion test on represented values.
  // Testing only each high word can discard a compiled low/tail residual and
  // incorrectly replace a near-identity source matrix with exact identity.
  bool identity=kinq_equal(q,kinq(float4(1.0f,0.0f,0.0f,0.0f)));
  if (identity) {
    out.m[0]=kin_dd(1); out.m[1]=kin_dd(0); out.m[2]=kin_dd(0);
    out.m[3]=kin_dd(0); out.m[4]=kin_dd(1); out.m[5]=kin_dd(0);
    out.m[6]=kin_dd(0); out.m[7]=kin_dd(0); out.m[8]=kin_dd(1);
    return out;
  }
  KinDD q00=kin_mul(q.w,q.w), q01=kin_mul(q.w,q.x);
  KinDD q02=kin_mul(q.w,q.y), q03=kin_mul(q.w,q.z);
  KinDD q11=kin_mul(q.x,q.x), q12=kin_mul(q.x,q.y);
  KinDD q13=kin_mul(q.x,q.z), q22=kin_mul(q.y,q.y);
  KinDD q23=kin_mul(q.y,q.z), q33=kin_mul(q.z,q.z);
  out.m[0]=kin_sub(kin_sub(kin_add(q00,q11),q22),q33);
  out.m[4]=kin_sub(kin_add(kin_sub(q00,q11),q22),q33);
  out.m[8]=kin_add(kin_sub(kin_sub(q00,q11),q22),q33);
  out.m[1]=kin_scale(kin_sub(q12,q03),2.0f);
  out.m[2]=kin_scale(kin_add(q13,q02),2.0f);
  out.m[3]=kin_scale(kin_add(q12,q03),2.0f);
  out.m[5]=kin_scale(kin_sub(q23,q01),2.0f);
  out.m[6]=kin_scale(kin_sub(q13,q02),2.0f);
  out.m[7]=kin_scale(kin_add(q23,q01),2.0f);
  return out;
}
inline KinDD3 kinmat_mul(KinMat m, KinDD3 v) {
  return KinDD3{
    kin_add(kin_add(kin_mul(m.m[0],v.x),kin_mul(m.m[1],v.y)),kin_mul(m.m[2],v.z)),
    kin_add(kin_add(kin_mul(m.m[3],v.x),kin_mul(m.m[4],v.y)),kin_mul(m.m[5],v.z)),
    kin_add(kin_add(kin_mul(m.m[6],v.x),kin_mul(m.m[7],v.y)),kin_mul(m.m[8],v.z))};
}
inline float3 kin3_value(KinDD3 a) {
  return float3(kin_value(a.x),kin_value(a.y),kin_value(a.z));
}
inline bool kin3_equal(KinDD3 a, KinDD3 b) {
  return kin_equal(a.x,b.x) && kin_equal(a.y,b.y) && kin_equal(a.z,b.z);
}
inline void kinmat_store(device float* hi, device float* low,
                         device float* tail, uint offset, KinMat m) {
  for (uint i=0;i<9;i++) {
    float h=kin_value(m.m[i]);
    KinDD r=kin_sub(m.m[i],kin_dd(h));
    hi[offset+i]=h; low[offset+i]=r.hi; tail[offset+i]=r.lo+r.tail;
  }
}
inline KinQ kinq_load(device const float* pair, uint offset, uint stride) {
  float4 h=float4(pair[offset],pair[offset+1],pair[offset+2],pair[offset+3]);
  float4 l=float4(pair[stride+offset],pair[stride+offset+1],
                  pair[stride+offset+2],pair[stride+offset+3]);
  float4 t=float4(pair[2*stride+offset],pair[2*stride+offset+1],
                  pair[2*stride+offset+2],pair[2*stride+offset+3]);
  return kinq(h,l,t);
}
inline KinDD3 kin3_load(device const float* pair, uint offset, uint stride) {
  return kin3(float3(pair[offset],pair[offset+1],pair[offset+2]),
              float3(pair[stride+offset],pair[stride+offset+1],
                     pair[stride+offset+2]),
              float3(pair[2*stride+offset],pair[2*stride+offset+1],
                     pair[2*stride+offset+2]));
}
inline void kinq_store(device float* hi, device float* low,
                       device float* tail, uint offset, KinQ q) {
  KinDD c[4]={q.w,q.x,q.y,q.z};
  for (uint i=0;i<4;i++) {
    float h=kin_value(c[i]);
    KinDD r=kin_sub(c[i],kin_dd(h));
    hi[offset+i]=h; low[offset+i]=r.hi;
    tail[offset+i]=r.lo+r.tail;
  }
}
inline void kinmat_store(device float* hi, device float* low,
                         device float* tail, uint offset, KinQ q) {
  KinDD q00=kin_mul(q.w,q.w), q01=kin_mul(q.w,q.x);
  KinDD q02=kin_mul(q.w,q.y), q03=kin_mul(q.w,q.z);
  KinDD q11=kin_mul(q.x,q.x), q12=kin_mul(q.x,q.y);
  KinDD q13=kin_mul(q.x,q.z), q22=kin_mul(q.y,q.y);
  KinDD q23=kin_mul(q.y,q.z), q33=kin_mul(q.z,q.z);
  KinDD m[9];
  m[0]=kin_sub(kin_add(q00,q11),kin_add(q22,q33));
  m[4]=kin_sub(kin_add(q00,q22),kin_add(q11,q33));
  m[8]=kin_sub(kin_add(q00,q33),kin_add(q11,q22));
  m[1]=kin_scale(kin_sub(q12,q03),2.0f);
  m[2]=kin_scale(kin_add(q13,q02),2.0f);
  m[3]=kin_scale(kin_add(q12,q03),2.0f);
  m[5]=kin_scale(kin_sub(q23,q01),2.0f);
  m[6]=kin_scale(kin_sub(q13,q02),2.0f);
  m[7]=kin_scale(kin_add(q23,q01),2.0f);
  for (uint i=0;i<9;i++) {
    float h=kin_value(m[i]);
    KinDD r=kin_sub(m[i],kin_dd(h));
    hi[offset+i]=h; low[offset+i]=r.hi; tail[offset+i]=r.lo+r.tail;
  }
}
inline KinDD3 kin3_cross(float3 a, KinDD3 b) {
  return KinDD3{
    kin_sub(kin_scale(b.z,a.y),kin_scale(b.y,a.z)),
    kin_sub(kin_scale(b.x,a.z),kin_scale(b.z,a.x)),
    kin_sub(kin_scale(b.y,a.x),kin_scale(b.x,a.y))};
}
inline KinDD3 qrot_pair(float4 q, KinDD3 v) {
  float3 qv=q.yzw;
  KinDD3 c=kin3_cross(qv,v);
  KinDD3 inner=KinDD3{
      kin_add(c.x,kin_scale(v.x,q.x)),
      kin_add(c.y,kin_scale(v.y,q.x)),
      kin_add(c.z,kin_scale(v.z,q.x))};
  KinDD3 t=kin3_scale(kin3_cross(qv,inner),2.0f);
  return kin3_add(v,t);
}
inline KinDD3 kin_from3(float3 high, float3 low, float3 tail) {
  return kin3(high,low,tail);
}
inline float3 kin_residual_low(KinDD3 value, float3 high) {
  KinDD3 d=kin3_sub(value,kin_from3(high,float3(0.0f),float3(0.0f)));
  return float3(d.x.hi,d.y.hi,d.z.hi);
}
inline float3 kin_residual_tail(KinDD3 value, float3 high) {
  KinDD3 d=kin3_sub(value,kin_from3(high,float3(0.0f),float3(0.0f)));
  return float3(d.x.lo+d.x.tail,d.y.lo+d.y.tail,d.z.lo+d.z.tail);
}
inline float3 load3(device const float* values, uint offset) {
  return float3(values[offset], values[offset+1], values[offset+2]);
}
inline float4 load4(device const float* values, uint offset) {
  return float4(values[offset], values[offset+1], values[offset+2], values[offset+3]);
}
inline void store3(device float* values, uint offset, float3 value) {
  values[offset] = value.x; values[offset+1] = value.y; values[offset+2] = value.z;
}
inline void store4(device float* values, uint offset, float4 value) {
  values[offset] = value.x; values[offset+1] = value.y;
  values[offset+2] = value.z; values[offset+3] = value.w;
}

// Copy only selected rows of external pose inputs into the compact FK cache.
kernel void prepare_fk_rows(
    device const int* world_mask [[buffer(0)]],
    device const float* mocap_pos [[buffer(1)]],
    device const float* mocap_quat [[buffer(2)]],
    device const int* tree_awake [[buffer(3)]],
    device float* auxiliary [[buffer(4)]],
    constant uint* dims [[buffer(5)]],
    uint world [[thread_position_in_grid]]) {
  uint nbody=dims[1], batch=dims[5], nmocap=dims[6], ntree=dims[7];
  if (world >= batch || world_mask[world] == 0) return;
  uint mocap_values = nmocap ? batch*nmocap*7 : 1;
  uint mocap_id_offset = mocap_values+nbody;
  uint awake_offset = mocap_id_offset+nbody;
  uint valid_offset = awake_offset+batch*max(ntree, 1u);
  uint mismatch_offset = valid_offset+batch;
  if (nmocap) {
    uint pos_base=world*nmocap*3;
    uint quat_base=batch*nmocap*3+world*nmocap*4;
    for (uint i=0; i<nmocap*3; ++i)
      auxiliary[pos_base+i]=mocap_pos[pos_base+i];
    for (uint i=0; i<nmocap*4; ++i)
      auxiliary[quat_base+i]=mocap_quat[world*nmocap*4+i];
  }
  uint stride=max(ntree, 1u);
  for (uint i=0; i<stride; ++i) {
    auxiliary[awake_offset+world*stride+i] =
        float(tree_awake[world*stride+i]);
    auxiliary[mismatch_offset+world*stride+i]=0.0f;
  }
}

// One thread computes one world's complete pose tree. This favors a simple
// correctness baseline; parallel body traversal is a separate optimization.
kernel void forward_kinematics(
    device const int* static_int [[buffer(0)]],
    device const float* static_float [[buffer(1)]],
    device const float* qpos [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const int* world_mask [[buffer(4)]],
    device float* auxiliary [[buffer(5)]],
    constant uint* dims [[buffer(6)]],
    uint world [[thread_position_in_grid]]) {
  device const int* parent=static_int+dims[12];
  device const int* jnt_type=static_int+dims[13];
  device const int* jnt_qposadr=static_int+dims[14];
  device const int* jnt_bodyid=static_int+dims[15];
  device const int* body_sameframe=static_int+dims[16];
  device const int* geom_bodyid=static_int+dims[17];
  device const int* geom_sameframe=static_int+dims[18];
  device const int* site_bodyid=static_int+dims[19];
  device const int* site_sameframe=static_int+dims[20];
  device const float* body_pos=static_float+dims[21];
  device const float* body_quat=static_float+dims[22];
  device const float* jnt_pos=static_float+dims[23];
  device const float* jnt_axis=static_float+dims[24];
  device const float* qpos0=static_float+dims[25];
  device const float* geom_pos=static_float+dims[26];
  device const float* geom_quat=static_float+dims[27];
  device const float* site_pos=static_float+dims[28];
  device const float* site_quat=static_float+dims[29];
  device const float* body_ipos=static_float+dims[30];
  device const float* body_iquat=static_float+dims[31];
  device float* xpos=output+dims[32];
  device float* xpos_low=output+dims[33];
  device float* xpos_tail=output+dims[34];
  device float* xquat=output+dims[35];
  device float* xquat_low=output+dims[36];
  device float* xquat_tail=output+dims[37];
  device float* geom_xpos=output+dims[38];
  device float* geom_xpos_low=output+dims[39];
  device float* geom_xpos_tail=output+dims[40];
  device float* geom_xquat=output+dims[41];
  device float* geom_xquat_low=output+dims[42];
  device float* geom_xquat_tail=output+dims[43];
  device float* geom_xmat=output+dims[44];
  device float* geom_xmat_low=output+dims[45];
  device float* geom_xmat_tail=output+dims[46];
  device float* site_xpos=output+dims[47];
  device float* site_xpos_low=output+dims[48];
  device float* site_xpos_tail=output+dims[49];
  device float* site_xquat=output+dims[50];
  device float* site_xquat_low=output+dims[51];
  device float* site_xquat_tail=output+dims[52];
  device float* inertial_xpos=output+dims[53];
  device float* inertial_xpos_low=output+dims[54];
  device float* inertial_xpos_tail=output+dims[55];
  device float* inertial_xquat=output+dims[56];
  device float* inertial_xquat_low=output+dims[57];
  device float* inertial_xquat_tail=output+dims[58];
  device float* joint_anchor=output+dims[59];
  device float* joint_axis=output+dims[60];
  uint nq = dims[0], nbody = dims[1], njnt = dims[2];
  uint ngeom = dims[3], nsite = dims[4], batch = dims[5];
  uint nmocap = dims[6];
  uint ntree = dims[7];
  if (world >= batch || world_mask[world] == 0) return;
  uint mocap_values = nmocap ? batch*nmocap*7 : 1;
  uint tree_offset = mocap_values;
  uint mocap_id_offset = tree_offset+nbody;
  uint awake_offset = mocap_id_offset+nbody;
  uint valid_offset = awake_offset+batch*max(ntree, 1u);
  uint mismatch_offset = valid_offset+batch;
  device const float* geom_pos_low=geom_pos+max(ngeom*3,1u);
  device const float* geom_pos_tail=geom_pos+2*max(ngeom*3,1u);
  device const float* site_pos_low=site_pos+max(nsite*3,1u);
  device const float* site_pos_tail=site_pos+2*max(nsite*3,1u);
  device const float* qpos0_low=qpos0+max(nq,1u);
  device const float* qpos0_tail=qpos0+2*max(nq,1u);
  uint body_q_stride=max(nbody*4,1u);
  uint geom_q_stride=max(ngeom*4,1u);
  uint site_q_stride=max(nsite*4,1u);
  uint inert_q_stride=max(nbody*4,1u);
  uint axis_stride=max(njnt*3,1u);
  device const float* jnt_axis_low=jnt_axis+axis_stride;
  device const float* jnt_axis_tail=jnt_axis+2*axis_stride;
  bool cached = auxiliary[valid_offset+world] != 0.0f;
  uint tree_stride = max(ntree, 1u);
  for (uint t=0; t<tree_stride; ++t) {
    auxiliary[mismatch_offset+world*tree_stride+t] = 0.0f;
  }
  uint qbase = world*nq;
  uint posebase = world*nbody;
  store3(xpos, posebase*3, float3(0));
  store3(xpos_low, posebase*3, float3(0));
  store3(xpos_tail, posebase*3, float3(0));
  store4(xquat, posebase*4, float4(1, 0, 0, 0));
  store4(xquat_low, posebase*4, float4(0));
  store4(xquat_tail, posebase*4, float4(0));
  for (uint b=1; b<nbody; ++b) {
    int tree = int(auxiliary[tree_offset+b]);
    bool sleeping = tree >= 0 && uint(tree) < ntree
        && auxiliary[awake_offset+world*tree_stride+uint(tree)] == 0.0f;
    uint p = uint(parent[b]);
    KinQ parent_q=kinq(load4(xquat,(posebase+p)*4),
                       load4(xquat_low,(posebase+p)*4),
                       load4(xquat_tail,(posebase+p)*4));
    KinDD3 parent_pos=kin_from3(load3(xpos,(posebase+p)*3),
        load3(xpos_low,(posebase+p)*3),load3(xpos_tail,(posebase+p)*3));
    KinDD3 body_local=kin3_load(body_pos,b*3,max(nbody*3,1u));
    KinMat parent_mat=kinq_matrix(parent_q);
    KinDD3 pos_pair=kin3_add(parent_pos,kinmat_mul(parent_mat,body_local));
    float3 pos=kin3_value(pos_pair);
    KinQ quat=kinq_mul(parent_q,kinq_load(body_quat,b*4,body_q_stride));
    int mid = int(auxiliary[mocap_id_offset+b]);
    if (mid >= 0 && uint(mid) < max(nmocap, 1u)) {
      // Pinned MuJoCo 3.10 mj_kinematics: the prescribed mocap pose replaces
      // the compiled body frame, composed with the parent frame (the pinned
      // compiler restricts mocap bodies to jointless world children).
      // Layout: all positions [batch, nmocap, 3] then all quaternions.
      uint nm = max(nmocap, 1u);
      uint mbase = world*nm + uint(mid);
      float3 mpos = load3(auxiliary, mbase*3);
      KinQ mquat=kinq_norm(kinq(load4(auxiliary,batch*nm*3+mbase*4)));
      KinDD3 mocap_local=kin3(mpos,float3(0.0f),float3(0.0f));
      pos_pair=kin3_add(parent_pos,kinmat_mul(parent_mat,mocap_local));
      pos=kin3_value(pos_pair);
      quat=kinq_mul(parent_q,mquat);
    }
    for (uint j=0; j<njnt; ++j) {
      if (uint(jnt_bodyid[j]) != b) continue;
      int typ = jnt_type[j];
      uint qa = qbase + uint(jnt_qposadr[j]);
      KinDD3 joint_local=kin3_load(jnt_pos,j*3,max(njnt*3,1u));
      KinDD3 axis_local=kin3(
          load3(jnt_axis,j*3),load3(jnt_axis_low,j*3),
          load3(jnt_axis_tail,j*3));
      KinDD3 anchor_pair=kin3_add(pos_pair,kinq_rotate(quat,joint_local));
      KinDD3 axis_pair=kinq_rotate(quat,axis_local);
      float3 anchor=kin3_value(anchor_pair);
      float3 axis=kin3_value(axis_pair);
      store3(joint_anchor, (world*njnt+j)*3, anchor);
      store3(joint_axis, (world*njnt+j)*3, axis);
      if (typ == 0) {
        pos = load3(qpos, qa);
        pos_pair=kin3(pos,float3(0.0f),float3(0.0f));
        quat = kinq_norm(kinq(load4(qpos, qa+3)));
        store3(joint_anchor, (world*njnt+j)*3, pos);
        store3(joint_axis, (world*njnt+j)*3, load3(jnt_axis, j*3));
      } else if (typ == 1) {
        quat=kinq_mul(quat,kinq_norm(kinq(load4(qpos,qa))));
        pos_pair=kin3_sub(anchor_pair,kinq_rotate(quat,joint_local));
        pos=kin3_value(pos_pair);
      } else if (typ == 2) {
        KinDD delta=kin_sub(kin_dd(qpos[qa]),
            KinDD{qpos0[jnt_qposadr[j]],qpos0_low[jnt_qposadr[j]],
                  qpos0_tail[jnt_qposadr[j]]});
        // engine_core_smooth.c transforms the local slide axis to xaxis
        // before mji_addToScl3(xpos, xaxis, delta). Keep that operation order
        // rather than rotating a locally scaled vector.
        KinDD3 slide_shift=KinDD3{kin_mul(delta,axis_pair.x),
                                  kin_mul(delta,axis_pair.y),
                                  kin_mul(delta,axis_pair.z)};
        pos_pair=kin3_add(pos_pair,slide_shift);
        pos=kin3_value(pos_pair);
      } else if (typ == 3) {
        float angle=kin_value(kin_sub(kin_dd(qpos[qa]),
            KinDD{qpos0[jnt_qposadr[j]],qpos0_low[jnt_qposadr[j]],
                  qpos0_tail[jnt_qposadr[j]]}));
        float cosine;
        float sine=sincos(0.5f*angle,cosine);
        KinQ local_rotation=KinQ{kin_dd(cosine),kin_scale(axis_local.x,sine),
            kin_scale(axis_local.y,sine),kin_scale(axis_local.z,sine)};
        quat=kinq_mul(quat,local_rotation);
        pos_pair=kin3_sub(anchor_pair,kinq_rotate(quat,joint_local));
        pos=kin3_value(pos_pair);
      }
    }
    // Pinned mj_kinematics1 computes a candidate pose even for a sleeping
    // articulated body, compares it exactly with the retained pose, and marks
    // the tree awake on any change. The scheduler then wakes its sleep cycle
    // before collision/constraint assembly. Do this comparison before the
    // cached-pose early-out; otherwise state writes made before the first
    // forward pass (and later in-place qpos edits) are silently ignored.
    if (sleeping) {
      bool has_joint = false;
      for (uint j=0; j<njnt; ++j) {
        has_joint = has_joint || uint(jnt_bodyid[j]) == b;
      }
      if (has_joint) {
        KinDD3 old_pos=kin_from3(load3(xpos,(posebase+b)*3),
            load3(xpos_low,(posebase+b)*3),load3(xpos_tail,(posebase+b)*3));
        KinQ old_quat=kinq(load4(xquat,(posebase+b)*4),
            load4(xquat_low,(posebase+b)*4),load4(xquat_tail,(posebase+b)*4));
        bool matches=kin3_equal(pos_pair,old_pos) && kinq_equal(quat,old_quat);
        if (!matches) {
          // The host consumes this float storage through an int32 view, so
          // write the integer-one bit pattern rather than float 1.0.
          auxiliary[mismatch_offset+world*tree_stride+uint(tree)] =
              as_type<float>(1u);
        }
        if (cached && matches) continue;
      }
      // A sleeping body without joints cannot differ from its compiled frame
      // under qpos changes. It still retains its validated frame when cached.
      else if (cached) {
        continue;
      }
    }
    store3(xpos, (posebase+b)*3, pos);
    store3(xpos_low,(posebase+b)*3,kin_residual_low(pos_pair,pos));
    store3(xpos_tail,(posebase+b)*3,kin_residual_tail(pos_pair,pos));
    quat=kinq_norm(quat);
    kinq_store(xquat,xquat_low,xquat_tail,(posebase+b)*4,quat);
  }
  // Match mj_kinematics2's inertial frame SAMEFRAME cases before any geom or
  // site asks to reuse the body's inertial frame.
  for (uint b=0; b<nbody; ++b) {
    int tree=int(auxiliary[tree_offset+b]);
    if (cached && tree>=0
        && auxiliary[awake_offset+world*tree_stride+uint(tree)]==0.0f) continue;
    uint boff=(posebase+b)*4, poff=(posebase+b)*3;
    KinQ bq=kinq(load4(xquat,boff),load4(xquat_low,boff),
                 load4(xquat_tail,boff));
    KinDD3 bp=kin_from3(load3(xpos,poff),load3(xpos_low,poff),
                        load3(xpos_tail,poff));
    KinDD3 local=kin3_load(body_ipos,b*3,max(nbody*3,1u));
    int sf=body_sameframe[b];
    KinQ iq;
    KinDD3 ip;
    if (sf==1) {
      iq=bq; ip=bp;
    } else if (sf==3) {
      iq=bq; ip=kin3_add(bp,kinmat_mul(kinq_matrix(bq),local));
    } else if (sf==2 || sf==4) {
      // The compiler emits these only for frames that alias an already
      // materialized inertial frame. Preserve its cached representation.
      if (!cached) { iq=kinq_mul(bq,kinq_load(body_iquat,b*4,inert_q_stride));
                     ip=kin3_add(bp,kinmat_mul(kinq_matrix(bq),local)); }
      else continue;
    } else {
      iq=kinq_mul(bq,kinq_load(body_iquat,b*4,inert_q_stride));
      ip=kin3_add(bp,kinmat_mul(kinq_matrix(bq),local));
    }
    float3 ih=kin3_value(ip);
    store3(inertial_xpos,poff,ih);
    store3(inertial_xpos_low,poff,kin_residual_low(ip,ih));
    store3(inertial_xpos_tail,poff,kin_residual_tail(ip,ih));
    kinq_store(inertial_xquat,inertial_xquat_low,inertial_xquat_tail,boff,iq);
  }
  for (uint g=0; g<ngeom; ++g) {
    uint b = uint(geom_bodyid[g]);
    int tree = int(auxiliary[tree_offset+b]);
    if (cached && tree >= 0 && auxiliary[awake_offset+world*tree_stride+uint(tree)] == 0.0f) continue;
    KinQ bq=kinq(load4(xquat,(posebase+b)*4),
                 load4(xquat_low,(posebase+b)*4),
                 load4(xquat_tail,(posebase+b)*4));
    uint geometry = (world*ngeom+g)*3;
    KinDD3 body_position = kin_from3(
        load3(xpos,(posebase+b)*3), load3(xpos_low,(posebase+b)*3),
        load3(xpos_tail,(posebase+b)*3));
    KinDD3 local_position = kin_from3(
        load3(geom_pos,g*3), load3(geom_pos_low,g*3),
        load3(geom_pos_tail,g*3));
    uint io=(posebase+b)*3;
    KinDD3 inertial_position=kin_from3(load3(inertial_xpos,io),
        load3(inertial_xpos+max(batch*nbody*3,1u),io),
        load3(inertial_xpos+2*max(batch*nbody*3,1u),io));
    KinQ iq=kinq(load4(inertial_xquat,(posebase+b)*4),
        load4(inertial_xquat_low,(posebase+b)*4),
        load4(inertial_xquat_tail,(posebase+b)*4));
    int sf=geom_sameframe[g];
    KinDD3 origin=body_position;
    KinMat frame_mat=kinq_matrix(bq);
    KinQ gq;
    KinDD3 exact_geom_position;
    if (sf==1) { exact_geom_position=body_position; gq=bq; }
    else if (sf==2) { exact_geom_position=inertial_position; gq=iq; }
    else if (sf==4) {
      // mj_local2Global uses body xmat for INERTIAROT position and ximat
      // only for orientation. This differs from INERTIA (code 2).
      origin=body_position; frame_mat=kinq_matrix(bq);
      exact_geom_position=kin3_add(origin,kinmat_mul(frame_mat,local_position));
      gq=iq;
    } else if (sf==3) {
      exact_geom_position=kin3_add(origin,kinmat_mul(frame_mat,local_position));
      gq=bq;
    } else {
      exact_geom_position=kin3_add(origin,kinmat_mul(frame_mat,local_position));
      gq=kinq_mul(bq,kinq_load(geom_quat,g*4,geom_q_stride));
    }
    float3 rounded_geom_position=kin3_value(exact_geom_position);
    store3(geom_xpos, (world*ngeom+g)*3,
           rounded_geom_position);
    store3(geom_xpos_low,geometry,
           kin_residual_low(exact_geom_position,rounded_geom_position));
    store3(geom_xpos_tail,geometry,
           kin_residual_tail(exact_geom_position,rounded_geom_position));
    uint gqoff=(world*ngeom+g)*4;
    kinq_store(geom_xquat,geom_xquat_low,geom_xquat_tail,gqoff,gq);
    KinMat gm=(sf==3) ? kinq_matrix(bq) : (sf==4 ? kinq_matrix(iq) : kinq_matrix(gq));
    kinmat_store(geom_xmat,geom_xmat_low,geom_xmat_tail,
                 (world*ngeom+g)*9,gm);
  }
  for (uint s=0; s<nsite; ++s) {
    uint b = uint(site_bodyid[s]);
    int tree = int(auxiliary[tree_offset+b]);
    if (cached && tree >= 0 && auxiliary[awake_offset+world*tree_stride+uint(tree)] == 0.0f) continue;
    KinQ bq=kinq(load4(xquat,(posebase+b)*4),
                 load4(xquat_low,(posebase+b)*4),
                 load4(xquat_tail,(posebase+b)*4));
    KinDD3 body_position=kin_from3(load3(xpos,(posebase+b)*3),
        load3(xpos_low,(posebase+b)*3),load3(xpos_tail,(posebase+b)*3));
    uint io=(posebase+b)*3;
    KinDD3 inertial_position=kin_from3(load3(inertial_xpos,io),
        load3(inertial_xpos+max(batch*nbody*3,1u),io),
        load3(inertial_xpos+2*max(batch*nbody*3,1u),io));
    KinQ iq=kinq(load4(inertial_xquat,(posebase+b)*4),
        load4(inertial_xquat_low,(posebase+b)*4),
        load4(inertial_xquat_tail,(posebase+b)*4));
    KinDD3 local_position=KinDD3{
        KinDD{load3(site_pos,s*3).x,load3(site_pos_low,s*3).x,load3(site_pos_tail,s*3).x},
        KinDD{load3(site_pos,s*3).y,load3(site_pos_low,s*3).y,load3(site_pos_tail,s*3).y},
        KinDD{load3(site_pos,s*3).z,load3(site_pos_low,s*3).z,load3(site_pos_tail,s*3).z}};
    int sf=site_sameframe[s];
    KinDD3 site_position; KinQ sq;
    if (sf==1) { site_position=body_position; sq=bq; }
    else if (sf==2) { site_position=inertial_position; sq=iq; }
    else if (sf==4) {
      site_position=kin3_add(body_position,
          kinmat_mul(kinq_matrix(bq),local_position)); sq=iq;
    } else if (sf==3) {
      site_position=kin3_add(body_position,
          kinmat_mul(kinq_matrix(bq),local_position)); sq=bq;
    } else {
      site_position=kin3_add(body_position,
          kinmat_mul(kinq_matrix(bq),local_position));
      sq=kinq_mul(bq,kinq_load(site_quat,s*4,site_q_stride));
    }
    uint soff=(world*nsite+s)*3;
    float3 site_high=kin3_value(site_position);
    store3(site_xpos,soff,site_high);
    store3(site_xpos_low,soff,kin_residual_low(site_position,site_high));
    store3(site_xpos_tail,soff,kin_residual_tail(site_position,site_high));
    kinq_store(site_xquat,site_xquat_low,site_xquat_tail,
               (world*nsite+s)*4,sq);
  }
  auxiliary[valid_offset+world] = 1.0f;
}
