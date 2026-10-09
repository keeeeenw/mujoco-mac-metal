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
inline FlexDD kin_source(KinDD value) {
  return FlexDD{value.hi,value.lo,value.tail};
}
inline KinDD kin_retained(FlexDD value) {
  return KinDD{value.hi,value.lo,value.tail};
}
inline KinDD kin_add(KinDD a, KinDD b) {
  return kin_retained(flex_dd_source_add(kin_source(a),kin_source(b)));
}
inline KinDD kin_neg(KinDD a) { return KinDD{-a.hi,-a.lo,-a.tail}; }
inline KinDD kin_sub(KinDD a, KinDD b) { return kin_add(a,kin_neg(b)); }
inline KinDD kin_mul(KinDD a, KinDD b) {
  return kin_retained(flex_dd_source_mul(kin_source(a),kin_source(b)));
}
inline KinDD kin_fma(KinDD a, KinDD b, KinDD c) {
  return kin_retained(flex_dd_source_fma(kin_source(a),kin_source(b),kin_source(c)));
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
  return kin_retained(flex_dd_source_div(kin_source(a),kin_source(b)));
}
inline KinDD kin_sqrt(KinDD a) {
  return kin_retained(flex_dd_source_sqrt(kin_source(a)));
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
  // Match pinned arm64 mju_mulQuat contractions, including the rounded
  // second product that seeds each row. Do not compute in excess precision.
  return KinQ{
    kin_fma(kin_neg(a.z),b.z,kin_fma(kin_neg(a.y),b.y,
        kin_fma(a.w,b.w,kin_neg(kin_mul(a.x,b.x))))),
    kin_fma(kin_neg(a.z),b.y,kin_fma(a.y,b.z,
        kin_fma(a.w,b.x,kin_mul(a.x,b.w)))),
    kin_fma(a.z,b.x,kin_fma(a.y,b.w,
        kin_fma(a.w,b.y,kin_neg(kin_mul(a.x,b.z))))),
    kin_fma(a.z,b.w,kin_fma(kin_neg(a.y),b.x,
        kin_fma(a.w,b.z,kin_mul(a.x,b.y))))};
}
inline KinQ kinq_norm(KinQ q) {
  // The pinned norm rounds x*x, fuses w*w into it, then y*y and z*z.
  KinDD n2=kin_fma(q.z,q.z,kin_fma(q.y,q.y,
      kin_fma(q.w,q.w,kin_mul(q.x,q.x))));
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
  KinDD tx=kin_fma(kin_neg(q.z),v.y,kin_fma(q.w,v.x,kin_mul(q.y,v.z)));
  KinDD ty=kin_fma(kin_neg(q.x),v.z,kin_fma(q.w,v.y,kin_mul(q.z,v.x)));
  KinDD tz=kin_fma(kin_neg(q.y),v.x,kin_fma(q.w,v.z,kin_mul(q.x,v.y)));
  return KinDD3{
    kin_fma(kin_fma(q.y,tz,kin_neg(kin_mul(q.z,ty))),kin_dd(2),v.x),
    kin_fma(kin_fma(q.z,tx,kin_neg(kin_mul(q.x,tz))),kin_dd(2),v.y),
    kin_fma(kin_fma(q.x,ty,kin_neg(kin_mul(q.y,tx))),kin_dd(2),v.z)};
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
  // Pinned arm64 mji_mulMatVec3 rounds the middle product before the
  // first and last products are contracted into it.
  return KinDD3{
    kin_fma(m.m[2],v.z,kin_fma(m.m[0],v.x,kin_mul(m.m[1],v.y))),
    kin_fma(m.m[5],v.z,kin_fma(m.m[3],v.x,kin_mul(m.m[4],v.y))),
    kin_fma(m.m[8],v.z,kin_fma(m.m[6],v.x,kin_mul(m.m[7],v.y)))};
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

// Source-word trigonometry: reduce finite float32-domain arguments using
// a 256-bit fixed-point 2/pi product. Integer range reduction never discards
// the low/tail input; the reduced series is evaluated at source precision.
inline uint kin_product_bit(thread const uint* p,int bit) {
  return bit<0 || bit>=320 ? 0u : (p[bit/16]>>(bit%16))&1u;
}
inline KinDD kin_sincos(KinDD x,bool want_sin) {
  int sign=kin_sign(x);if(sign==0)return want_sin?x:kin_dd(1.0f);
  KinDD a=sign<0?kin_neg(x):x;
  KinDD halfpi=KinDD{1.5707963705062866211f,-4.3711388286737928865e-8f,
                    -1.7763568394002504647e-15f};
  int quadrant=0;KinDD reduced=a;
  if(kin_compare(a,kin_scale(halfpi,0.5f))>0) {
    int exponent=flex_float_exponent(a.hi);
    if(kin_compare(a,kin_dd(flex_float_scalbn(0.5f,exponent)))<0)exponent--;
    FlexDD normalized=flex_dd_scalbn(kin_source(a),53-exponent);
    ulong mantissa=ulong(normalized.hi);
    long correction=long(normalized.lo)+long(normalized.tail);
    mantissa=ulong(long(mantissa)+correction);
    // floor((2/pi)*2^256), little-endian base 2^16 limbs.
    const uint digits[16]={0xc561u,0xdebb,0x63ab,0xfe51,0x9041,0x3c43,0x9599,0xdb62,
                          0xddc0,0xf534,0x57d1,0xfc27,0x1529,0x4e44,0x836e,0xa2f9};
    uint product[20];for(int i=0;i<20;i++)product[i]=0;
    for(int i=0;i<4;i++) {
      ulong digit=(mantissa>>(16*i))&65535ul,carry=0;
      for(int j=0;j<16;j++) {
        ulong term=digit*ulong(digits[j])+ulong(product[i+j])+carry;
        product[i+j]=uint(term&65535ul);carry=term>>16;
      }
      int at=i+16;
      while(carry && at<20) {
        ulong term=ulong(product[at])+carry;
        product[at++]=uint(term&65535ul);carry=term>>16;
      }
    }
    int shift=309-exponent;
    bool round_up=kin_product_bit(product,shift-1)!=0;
    quadrant=int((kin_product_bit(product,shift)
                   +2u*kin_product_bit(product,shift+1)+uint(round_up))&3u);
    uint fraction[3]={0,0,0};
    for(int k=0;k<72;k++)fraction[k/24]=(fraction[k/24]<<1)|kin_product_bit(product,shift-1-k);
    KinDD f=kin_add(kin_dd(ldexp(float(fraction[0]),-24)),
                    kin_add(kin_dd(ldexp(float(fraction[1]),-48)),
                            kin_dd(ldexp(float(fraction[2]),-72))));
    if(round_up)f=kin_sub(f,kin_dd(1));
    reduced=kin_mul(f,halfpi);
  }
  bool series_sine=want_sin ? !(quadrant&1) : bool(quadrant&1);
  KinDD square=kin_mul(reduced,reduced);
  KinDD term=series_sine?reduced:kin_dd(1),sum=term;
  for(int n=1;n<=12;n++) {
    float den=series_sine?float((2*n)*(2*n+1)):float((2*n-1)*(2*n));
    term=kin_neg(kin_div(kin_mul(term,square),kin_dd(den)));
    sum=kin_add(sum,term);
  }
  bool negate=want_sin ? quadrant>=2 : quadrant==1 || quadrant==2;
  if(want_sin && sign<0)negate=!negate;
  return negate?kin_neg(sum):sum;
}
inline KinDD kin_state_load(device const float* high,device const float* words,
                             uint at,uint word_offset,uint tail_offset) {
  return KinDD{high[at],words[word_offset],words[tail_offset]};
}
inline void kin_state_store(device float* high,device float* words,uint at,
                             uint word_offset,uint tail_offset,KinDD value) {
  float h=kin_value(value);KinDD rest=kin_sub(value,kin_dd(h));
  high[at]=h;words[word_offset]=rest.hi;words[tail_offset]=rest.lo+rest.tail;
}
