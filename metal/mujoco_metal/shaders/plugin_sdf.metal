#include <metal_stdlib>
using namespace metal;

// Bundled MuJoCo 3.10.0 analytic plugin-SDF evaluator.  This file deliberately
// exposes the two inline routines so the flex and rigid candidate programs can
// concatenate this source and evaluate points inside their own fixed device
// loops, without a host round trip.

constant float kPluginSdfPi = 3.14159265358979323846f;

inline float plugin_sdf_union(float a, float b) { return min(a, b); }
inline float plugin_sdf_intersection(float a, float b) { return max(a, b); }
inline float plugin_sdf_subtraction(float a, float b) { return max(a, -b); }
inline float plugin_sdf_fract(float x) { return x - floor(x); }
inline float plugin_sdf_mod(float x, float y) { return x - y * floor(x / y); }

inline float plugin_sdf_extrusion(float3 p, float d2, float h) {
  float2 w = float2(d2, fabs(p.z) - h);
  float2 wp = max(w, float2(0.0f));
  return min(max(w.x, w.y), 0.0f) + length(wp);
}

inline float plugin_sdf_bolt(float3 p, thread const float* a) {
  const float screw = 12.0f;
  const float radius = length(p.xy) - a[0];
  const float root2 = sqrt(2.0f) * 0.5f;
  float azimuth = atan2(p.y, p.x);
  float triangle = fabs(plugin_sdf_fract(p.z * screw - azimuth / kPluginSdfPi / 2.0f) - 0.5f);
  float thread_sdf = (radius - triangle / screw) * root2;
  float bolt = plugin_sdf_subtraction(thread_sdf, 0.5f - fabs(p.z + 0.5f));
  float cone = (p.z - radius) * root2;
  bolt = plugin_sdf_subtraction(bolt, cone + root2);
  float k = 6.0f / kPluginSdfPi / 2.0f;
  float angle = -floor(atan2(p.y, p.x) * k + 0.5f) / k;
  float s0 = sin(angle), s1 = sin(angle + kPluginSdfPi * 0.5f);
  float2 rotated = float2(s1 * p.x - s0 * p.y, s0 * p.x + s1 * p.y);
  float head = rotated.x - 0.5f;
  head = plugin_sdf_intersection(head, fabs(p.z + 0.25f) - 0.25f);
  head = plugin_sdf_intersection(head, (p.z + radius - 0.22f) * root2);
  return plugin_sdf_union(bolt, head);
}

inline float plugin_sdf_bowl(float3 p, thread const float* a) {
  float h = a[0], radius = a[1], thickness = a[2];
  float width = sqrt(max(radius * radius - h * h, 0.0f));
  float2 q = float2(length(p.xy), p.z);
  float2 qdiff = q - float2(width, h);
  return ((h * q.x < width * q.y) ? length(qdiff)
                                   : fabs(length(q) - radius)) - thickness;
}

inline float plugin_sdf_circle(float rho, float radius) { return rho - radius; }
inline float plugin_sdf_smooth_union(float a, float b, float k) {
  float h = clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
  return b * (1.0f - h) + a * h - k * h * (1.0f - h);
}
inline float plugin_sdf_smooth_intersection(float a, float b, float k) {
  return plugin_sdf_subtraction(
      plugin_sdf_intersection(a, b),
      plugin_sdf_smooth_union(plugin_sdf_subtraction(a, b),
                              plugin_sdf_subtraction(b, a), k));
}

inline float plugin_sdf_gear_2d(float3 p, thread const float* a) {
  float D = a[1], N = a[2], alpha = a[0];
  float psi = 3.096e-5f * N * N - 6.557e-3f * N + 0.551f;
  float R = D * 0.5f;
  float rho = length(p.xy);
  float Pd = N / D;
  float P = kPluginSdfPi / Pd;
  float addendum = 1.0f / Pd;
  float Do = D + 2.0f * addendum;
  float Ro = Do * 0.5f;
  float h = 2.2f / Pd;
  float innerR = Ro - h - 0.14f * D;
  if (a[4] >= 0.0f) innerR = a[4] * 0.5f;
  if (innerR - rho > 0.0f) return innerR - rho;
  if (Ro - rho < -0.2f) return rho - Ro;
  float Db = D * cos(psi), Rb = Db * 0.5f;
  float fi = atan2(p.y, p.x) + alpha;
  float alphaStride = P / R;
  float invAlpha = acos(clamp(Rb / R, -1.0f, 1.0f));
  float invPhi = tan(invAlpha) - invAlpha;
  float shift = alphaStride * 0.5f - 2.0f * invPhi;
  float fia = plugin_sdf_mod(fi + shift * 0.5f, alphaStride) - shift * 0.5f;
  float fib = plugin_sdf_mod(-fi - shift + shift * 0.5f, alphaStride) - shift * 0.5f;
  float dista = -1.0e6f, distb = -1.0e6f;
  if (Rb < rho) {
    float acos_rbRho = acos(clamp(Rb / rho, -1.0f, 1.0f));
    float thetaa = fia + acos_rbRho;
    float thetab = fib + acos_rbRho;
    float ta = sqrt(max(rho * rho - Rb * Rb, 0.0f));
    dista = ta - Rb * thetaa;
    distb = ta - Rb * thetab;
  }
  float gearOuter = plugin_sdf_circle(rho, Ro);
  float gearLowBase = plugin_sdf_circle(rho, Ro - h);
  float crownBase = plugin_sdf_circle(rho, innerR);
  float cogs = plugin_sdf_intersection(dista, distb);
  float baseWalls = plugin_sdf_intersection(fia - (alphaStride - shift),
                                             fib - (alphaStride - shift));
  cogs = plugin_sdf_intersection(baseWalls, cogs);
  cogs = plugin_sdf_smooth_intersection(gearOuter, cogs, 0.0035f * D);
  cogs = plugin_sdf_smooth_union(gearLowBase, cogs, Rb - Ro + h);
  return plugin_sdf_subtraction(cogs, crownBase);
}

inline float plugin_sdf_gear(float3 p, thread const float* a) {
  return plugin_sdf_extrusion(p, plugin_sdf_gear_2d(p, a), a[3] * 0.5f);
}

inline float plugin_sdf_nut(float3 p, thread const float* a) {
  const float screw = 12.0f;
  const float radius2 = length(p.xy) - a[0];
  const float root2 = sqrt(2.0f) * 0.5f;
  float azimuth = atan2(p.y, p.x);
  float triangle = fabs(plugin_sdf_fract(p.z * screw - azimuth / kPluginSdfPi / 2.0f) - 0.5f);
  float thread2 = (radius2 - triangle / screw) * root2;
  float cone2 = (p.z - radius2) * root2;
  float hole = plugin_sdf_subtraction(thread2, cone2 + 0.5f * root2);
  hole = plugin_sdf_union(hole, -cone2 - 0.05f * root2);
  float k = 6.0f / kPluginSdfPi / 2.0f;
  float angle = -floor(atan2(p.y, p.x) * k + 0.5f) / k;
  float s0 = sin(angle), s1 = sin(angle + kPluginSdfPi * 0.5f);
  float2 rotated = float2(s1 * p.x - s0 * p.y, s0 * p.x + s1 * p.y);
  float head = rotated.x - 0.5f;
  head = plugin_sdf_intersection(head, fabs(p.z + 0.25f) - 0.25f);
  head = plugin_sdf_intersection(head, (p.z + radius2 - 0.22f) * root2);
  return plugin_sdf_subtraction(head, hole);
}

inline float plugin_sdf_torus(float3 p, thread const float* a) {
  float q = length(p.xy) - a[0];
  return length(float2(q, p.z)) - a[1];
}

inline float plugin_sdf_distance(int kind, float3 p, thread const float* a) {
  switch (kind) {
    case 1: return plugin_sdf_bolt(p, a);
    case 2: return plugin_sdf_bowl(p, a);
    case 3: return plugin_sdf_gear(p, a);
    case 4: return plugin_sdf_nut(p, a);
    case 5: return plugin_sdf_torus(p, a);
    default: return 0.0f;
  }
}

// Forward-mode directional derivatives for the double-precision upstream
// callback's one-sided gradient.  A Jet carries the value and three partials;
// min/max/abs select the branch approached by a positive coordinate step at
// exact ties.  This avoids subtracting nearly equal float32 SDF values (which
// destroys the registered callback's gradient at corners and thread creases).
struct SdfWide { float hi; float mid; float lo; };
inline SdfWide sw(float x);
inline SdfWide sw_add(SdfWide a, SdfWide b);
inline SdfWide sw_sub(SdfWide a, SdfWide b);
inline SdfWide sw_mul(SdfWide a, SdfWide b);
inline SdfWide sw_div(SdfWide a, SdfWide b);
inline SdfWide sw_sqrt(SdfWide a);
inline int sw_sign(SdfWide a);
struct SdfJet { float v; float e; float3 d; float tail; };
inline SdfJet sjpair(float hi, float lo, float3 d) { return {hi, lo, d, 0.0f}; }
inline SdfJet sjwide(SdfWide v, float3 d = float3(0.0f)) {
  return {v.hi, v.mid, d, v.lo};
}
inline SdfJet sj(float v, float3 d = float3(0.0f)) { return sjpair(v, 0.0f, d); }
inline SdfWide sjwide_value(SdfJet a) {
  return sw_add(sw_add(sw(a.v), sw(a.e)), sw(a.tail));
}
inline float sjvalue(SdfJet a) { return (a.v + a.e) + a.tail; }
inline int sjcompare(SdfJet a, SdfJet b) {
  return sw_sign(sw_sub(sjwide_value(a), sjwide_value(b)));
}
inline int sjfloor_value(SdfJet a) {
  float f=floor(a.v);
  if (a.v==f && sw_sign(sw_add(sw(a.e),sw(a.tail)))<0) return int(f)-1;
  return int(f);
}
inline bool sjinteger(SdfJet a) {
  return sjcompare(a,sj(float(sjfloor_value(a))))==0;
}
inline SdfJet sjrenorm(float hi, float lo, float3 d) {
  float sum = hi + lo;
  float err = lo - (sum - hi);
  return sjpair(sum, err, d);
}
inline SdfJet sjadd(SdfJet a, SdfJet b) {
  return sjwide(sw_add(sjwide_value(a), sjwide_value(b)), a.d + b.d);
}
inline SdfJet sjsub(SdfJet a, SdfJet b) {
  return sjadd(a, SdfJet{-b.v, -b.e, -b.d, -b.tail});
}
inline SdfJet sjneg(SdfJet a) { return SdfJet{-a.v, -a.e, -a.d, -a.tail}; }
inline SdfJet sjmul(SdfJet a, SdfJet b) {
  return sjwide(sw_mul(sjwide_value(a), sjwide_value(b)),
      a.d*sjvalue(b) + b.d*sjvalue(a));
}
inline SdfJet sjscale(SdfJet a, float b) {
  return sjwide(sw_mul(sjwide_value(a), sw(b)), a.d*b);
}
inline SdfJet sjdiv(SdfJet a, SdfJet b) {
  float av=sjvalue(a), bv=sjvalue(b), den=bv*bv;
  return sjwide(sw_div(sjwide_value(a), sjwide_value(b)),
      (a.d*bv-b.d*av)/den);
}
// Two-word scalar operations used by the source finite-difference callback.
// The registered callbacks evaluate in mjtNum (binary64); native Metal has no
// binary64 arithmetic, so the finite-stencil path carries float expansions.
// In particular, a hardware float sin/cos followed by a first-order low-word
// correction is not accurate enough: its ~1e-8 absolute library error is
// divided by the callback's 1e-8 stencil.
inline float2 sdrenorm(float hi, float lo) {
  float s=hi+lo;
  return float2(s,lo-(s-hi));
}
inline bool sdnegative(float x) { return (as_type<uint>(x) & 0x80000000u) != 0; }
inline float2 sdadd(float2 a,float2 b) {
  float s=a.x+b.x, bv=s-a.x;
  float e=(a.x-(s-bv))+(b.x-bv)+a.y+b.y;
  return sdrenorm(s,e);
}
inline float2 sdneg(float2 a) { return float2(-a.x,-a.y); }
inline float2 sdsub(float2 a,float2 b) { return sdadd(a,sdneg(b)); }
inline int sdcompare(float2 a,float2 b) {
  float2 d=sdsub(a,b);
  if(d.x>0.0f) return 1;
  if(d.x<0.0f) return -1;
  return d.y>0.0f ? 1 : (d.y<0.0f ? -1 : 0);
}
inline float2 sdmul(float2 a,float2 b) {
  float p=a.x*b.x;
  float e=fma(a.x,b.x,-p)+a.x*b.y+a.y*b.x+a.y*b.y;
  return sdrenorm(p,e);
}
inline float2 sdscale(float2 a,float b) {
  float p=a.x*b;
  return sdrenorm(p,fma(a.x,b,-p)+a.y*b);
}
inline float2 sddiv(float2 a,float2 b) {
  float q=a.x/b.x;
  float2 rem=sdsub(a,sdmul(b,float2(q,0.0f)));
  return sdrenorm(q,(rem.x+rem.y)/b.x);
}
inline float2 sdsqrt(float2 a) {
  a=sdrenorm(a.x,a.y);
  if(a.x<0.0f && a.x+a.y>=0.0f) a=sdrenorm(0.0f,a.x+a.y);
  if(a.x<0.0f) return float2(0.0f,0.0f);
  float v=sqrt(max(a.x,0.0f));
  if(v==0.0f) return float2(0.0f,sqrt(max(a.y,0.0f)));
  float2 residual=sdsub(a,sdmul(float2(v,0.0f),float2(v,0.0f)));
  float correction=(residual.x+residual.y)/(2.0f*v);
  float2 root=sdrenorm(v,correction);
  // One Newton correction removes the rounding in the first correction.
  residual=sdsub(a,sdmul(root,root));
  return sdrenorm(root.x,root.y+(residual.x+residual.y)/(2.0f*root.x));
}
constant float kSdfPiHi=3.1415927410125732421875f;
constant float kSdfPiLo=-8.7422776573475857731e-8f;
constant float kSdfHalfPiHi=1.57079637050628662109375f;
constant float kSdfHalfPiLo=-4.3711388286737928865e-8f;
constant float kSdfQuarterPiHi=0.785398185253143310546875f;
constant float kSdfQuarterPiLo=-2.1855694143368964433e-8f;
inline float2 sdatan_series(float2 x) {
  float2 xx=sdmul(x,x), term=x, sum=x;
  for(int n=1;n<=28;n++) {
    term=sdmul(term,xx);
    float2 add=sddiv(term,float2(float(2*n+1),0.0f));
    if(n&1) add=sdneg(add);
    sum=sdadd(sum,add);
  }
  return sum;
}
inline float2 sdatan(float2 x) {
  bool neg=x.x<0.0f || (x.x==0.0f && (sdnegative(x.x) || x.y<0.0f));
  if(neg) x=sdneg(x);
  bool reciprocal=x.x>1.0f;
  if(reciprocal) x=sddiv(float2(1.0f,0.0f),x);
  bool around_one=x.x>0.41421353816986083984375f;
  float2 a;
  if(around_one) {
    float2 z=sddiv(sdsub(x,float2(1.0f,0.0f)),sdadd(x,float2(1.0f,0.0f)));
    a=sdadd(float2(kSdfQuarterPiHi,kSdfQuarterPiLo),sdatan_series(z));
  } else {
    a=sdatan_series(x);
  }
  if(reciprocal) a=sdsub(float2(kSdfHalfPiHi,kSdfHalfPiLo),a);
  return neg ? sdneg(a) : a;
}
inline float2 sdatan2(float2 y,float2 x) {
  bool xzero=x.x==0.0f && x.y==0.0f;
  bool yzero=y.x==0.0f && y.y==0.0f;
  bool xneg=x.x<0.0f || (xzero && (sdnegative(x.x) || x.y<0.0f));
  bool yneg=y.x<0.0f || (yzero && (sdnegative(y.x) || y.y<0.0f));
  if(xzero) {
    if(yzero) {
      if(xneg) return yneg ? sdneg(float2(kSdfPiHi,kSdfPiLo))
                           : float2(kSdfPiHi,kSdfPiLo);
      return float2(yneg ? -0.0f : 0.0f,0.0f);
    }
    return yneg ? sdneg(float2(kSdfHalfPiHi,kSdfHalfPiLo))
                : float2(kSdfHalfPiHi,kSdfHalfPiLo);
  }
  float2 a=sdatan(sddiv(y,x));
  if(xneg) a=sdadd(a,yneg ? sdneg(float2(kSdfPiHi,kSdfPiLo))
                           : float2(kSdfPiHi,kSdfPiLo));
  return a;
}
inline float2 sd_sincos_reduced(float2 r,bool want_sin) {
  float2 rr=sdmul(r,r);
  float2 term=want_sin ? r : float2(1.0f,0.0f), sum=term;
  for(int n=1;n<=12;n++) {
    float den=want_sin ? float((2*n)*(2*n+1)) : float((2*n-1)*(2*n));
    term=sdneg(sddiv(sdmul(term,rr),float2(den,0.0f)));
    sum=sdadd(sum,term);
  }
  return sum;
}
inline float2 sdsin_cos(float2 x,bool want_sin) {
  // Reduce to [-pi/4,pi/4] using a split pi/2 and evaluate a convergent
  // Taylor series there.  The integer reduction preserves the low word.
  float scaled=(x.x+x.y)*0.63661977236758134308f;
  int q=int(rint(scaled));
  float2 r=sdsub(x,sdscale(float2(kSdfHalfPiHi,kSdfHalfPiLo),float(q)));
  const float2 quarter_pi=float2(kSdfQuarterPiHi,kSdfQuarterPiLo);
  if(sdcompare(r,quarter_pi)>0) {
    q+=1;
    r=sdsub(r,float2(kSdfHalfPiHi,kSdfHalfPiLo));
  } else if(sdcompare(r,sdneg(quarter_pi))<0) {
    q-=1;
    r=sdadd(r,float2(kSdfHalfPiHi,kSdfHalfPiLo));
  }
  int quadrant=((q%4)+4)%4;
  if(want_sin) {
    if(quadrant==1) return sd_sincos_reduced(r,false);
    if(quadrant==2) return sdneg(sd_sincos_reduced(r,true));
    if(quadrant==3) return sdneg(sd_sincos_reduced(r,false));
    return sd_sincos_reduced(r,true);
  } else {
    if(quadrant==1) return sdneg(sd_sincos_reduced(r,true));
    if(quadrant==2) return sdneg(sd_sincos_reduced(r,false));
    if(quadrant==3) return sd_sincos_reduced(r,true);
    return sd_sincos_reduced(r,false);
  }
}
inline SdfJet sjsqrt(SdfJet a) {
  SdfWide root = sw_sqrt(sjwide_value(a));
  float v=(root.hi+root.mid)+root.lo;
  return sjwide(root, v>0.0f ? a.d*(0.5f/v) : float3(0.0f));
}
inline SdfJet sjsin(SdfJet a) {
  float2 v=sdsin_cos(float2(a.v,a.e),true);
  float2 c=sdsin_cos(float2(a.v,a.e),false);
  return sjrenorm(v.x,v.y,(c.x+c.y)*a.d);
}
inline SdfJet sjcos(SdfJet a) {
  float2 v=sdsin_cos(float2(a.v,a.e),false);
  float2 s=sdsin_cos(float2(a.v,a.e),true);
  return sjrenorm(v.x,v.y,-(s.x+s.y)*a.d);
}
inline SdfJet sjtan(SdfJet a) {
  float2 s=sdsin_cos(float2(a.v,a.e),true);
  float2 c=sdsin_cos(float2(a.v,a.e),false);
  float2 v=sddiv(s,c);
  float den=(c.x+c.y)*(c.x+c.y);
  return sjrenorm(v.x,v.y,a.d/den);
}
inline SdfJet sjatan2(SdfJet y, SdfJet x) {
  float den=x.v*x.v+y.v*y.v;
  float2 value=sdatan2(float2(y.v,y.e),float2(x.v,x.e));
  float lo=value.y;
  return sjrenorm(value.x,lo,
      den>0.0f ? (x.v*y.d-y.v*x.d)/den : float3(0.0f));
}
inline SdfJet sjacos(SdfJet a) {
  float2 av=float2(a.v,a.e);
  if(sdcompare(av,float2(1.0f,0.0f))>=0) {
    return sdcompare(av,float2(1.0f,0.0f))==0
        ? sj(0.0f,-a.d) : sj(0.0f);
  }
  if(sdcompare(av,float2(-1.0f,0.0f))<=0) {
    return sjrenorm(kSdfPiHi,kSdfPiLo,float3(0.0f));
  }
  float2 one_minus=sdsub(float2(1.0f,0.0f),sdmul(av,av));
  float2 den_pair=sdsqrt(one_minus);
  float den=den_pair.x+den_pair.y;
  float2 value=sdatan2(den_pair,av);
  return sjrenorm(value.x,value.y,
      den>0.0f ? -a.d/den : float3(0.0f));
}
inline SdfJet sjmin(SdfJet a, SdfJet b) {
  if (sjcompare(a,b)<0) return a; if (sjcompare(b,a)<0) return b;
  return SdfJet{a.v,a.e,min(a.d,b.d),a.tail};
}
inline SdfJet sjmax(SdfJet a, SdfJet b) {
  if (sjcompare(a,b)>0) return a; if (sjcompare(b,a)>0) return b;
  return SdfJet{a.v,a.e,max(a.d,b.d),a.tail};
}
inline SdfJet sjabs(SdfJet a) {
  if (sjcompare(a,sj(0.0f))>0) return a;
  if (sjcompare(a,sj(0.0f))<0) return sjneg(a);
  return sjwide(sw(0.0f), abs(a.d));
}
inline SdfJet sjfloor(SdfJet a) { return sj(float(sjfloor_value(a)), float3(0.0f)); }
inline SdfJet sjfract(SdfJet a) { return sjsub(a,sjfloor(a)); }
inline SdfJet sjmod(SdfJet a, SdfJet b) { return sjsub(a,sjmul(b,sjfloor(sjdiv(a,b)))); }
inline SdfJet sjlength2(SdfJet x, SdfJet y) {
  if (sjcompare(x,sj(0.0f))==0 && sjcompare(y,sj(0.0f))==0) {
    // The positive-coordinate directional derivative of a norm at the
    // origin is the norm of that coordinate's component directions.
    return sj(0.0f, sqrt(x.d*x.d + y.d*y.d));
  }
  return sjsqrt(sjadd(sjmul(x,x),sjmul(y,y)));
}
inline SdfJet sjlength3(SdfJet x, SdfJet y, SdfJet z) {
  if (sjcompare(x,sj(0.0f))==0 && sjcompare(y,sj(0.0f))==0 &&
      sjcompare(z,sj(0.0f))==0) {
    return sj(0.0f, sqrt(x.d*x.d + y.d*y.d + z.d*z.d));
  }
  return sjsqrt(sjadd(sjadd(sjmul(x,x),sjmul(y,y)),sjmul(z,z)));
}
inline SdfJet sjunion(SdfJet a,SdfJet b) { return sjmin(a,b); }
inline SdfJet sjintersection(SdfJet a,SdfJet b) { return sjmax(a,b); }
inline SdfJet sjsubtraction(SdfJet a,SdfJet b) { return sjmax(a,sjneg(b)); }
inline SdfJet sjselect_lt(SdfJet test,SdfJet yes,SdfJet no) {
  int tv=sjcompare(test,sj(0.0f));
  if(tv<0) return yes;
  if(tv>0) return no;
  return sjpair(yes.v,yes.e,select(no.d,yes.d,test.d<0.0f));
}
inline SdfJet sjextrusion(SdfJet x,SdfJet y,SdfJet z,SdfJet d2,SdfJet h) {
  SdfJet w0=d2, w1=sjsub(sjabs(z),h);
  SdfJet outside=sjlength2(sjmax(w0,sj(0.0f)),sjmax(w1,sj(0.0f)));
  return sjadd(sjmin(sjmax(w0,w1),sj(0.0f)),outside);
}
inline SdfJet sjbolt_nut(int kind,SdfJet x,SdfJet y,SdfJet z,
                         thread const float* a,thread const float* alow) {
  const float screw=12.0f, root2=0.7071067811865475f;
  SdfJet radius=sjsub(sjlength2(x,y),sjpair(a[0],alow[0],float3(0.0f)));
  SdfJet az=sjatan2(y,x);
  SdfJet phase=sjsub(sjscale(z,screw),sjscale(az,1.0f/kPluginSdfPi/2.0f));
  SdfJet triangle;
  if (sjinteger(phase)) {
    // At an integer thread phase, fract's value jumps between 0 and 1, but
    // the composed triangular profile remains continuous. Its one-sided
    // derivative is -abs(phase') on both directions; differentiating fract
    // alone would invent a 1/eps spike or choose the wrong tangent.
    triangle=sj(0.5f,-abs(phase.d));
  } else {
    triangle=sjabs(sjsub(sjfract(phase),sj(0.5f)));
  }
  SdfJet threadv=sjscale(sjsub(radius,sjscale(triangle,1.0f/screw)),root2);
  SdfJet cone=sjscale(sjsub(z,radius),root2);
  SdfJet k=sj(6.0f/kPluginSdfPi/2.0f);
  SdfJet angle=sjdiv(sjneg(sjfloor(sjadd(sjscale(az,k.v),sj(0.5f)))),k);
  SdfJet s0=sjsin(angle), s1=sjsin(sjadd(angle,sj(kPluginSdfPi*0.5f)));
  SdfJet xr=sjsub(sjmul(s1,x),sjmul(s0,y));
  SdfJet head=sjsub(xr,sj(0.5f));
  head=sjintersection(head,sjsub(sjabs(sjadd(z,sj(0.25f))),sj(0.25f)));
  head=sjintersection(head,sjscale(sjsub(sjadd(z,radius),sj(0.22f)),root2));
  if(kind==1) {
    SdfJet bolt=sjsubtraction(threadv,sjsub(sj(0.5f),sjabs(sjadd(z,sj(0.5f)))));
    bolt=sjsubtraction(bolt,sjadd(cone,sj(root2)));
    return sjunion(bolt,head);
  }
  SdfJet hole=sjsubtraction(threadv,sjadd(cone,sj(0.5f*root2)));
  hole=sjunion(hole,sjsub(sjneg(cone),sj(0.05f*root2)));
  return sjsubtraction(head,hole);
}
inline SdfJet sjbowl(SdfJet x,SdfJet y,SdfJet z,thread const float* a,
                     thread const float* alow) {
  SdfJet h=sjpair(a[0],alow[0],float3(0.0f));
  SdfJet r=sjpair(a[1],alow[1],float3(0.0f));
  SdfJet thick=sjpair(a[2],alow[2],float3(0.0f));
  SdfJet width=sjsqrt(sjmax(sjsub(sjmul(r,r),sjmul(h,h)),sj(0.0f)));
  SdfJet qx=sjlength2(x,y), qy=z;
  SdfJet dx=sjsub(qx,width), dy=sjsub(qy,h);
  SdfJet branch=sjsub(sjmul(qx,h),sjmul(qy,width));
  SdfJet shell=sjabs(sjsub(sjlength2(qx,qy),r));
  SdfJet d=sjselect_lt(branch,sjlength2(dx,dy),shell);
  return sjsub(d,thick);
}
inline SdfJet sjsmooth_union(SdfJet a,SdfJet b,SdfJet k) {
  SdfJet h=sjmax(sjmin(sjadd(sj(0.5f),sjscale(sjdiv(sjsub(b,a),k),0.5f)),sj(1.0f)),sj(0.0f));
  return sjsub(sjadd(sjmul(b,sjsub(sj(1.0f),h)),sjmul(a,h)),
               sjmul(k,sjmul(h,sjsub(sj(1.0f),h))));
}
inline SdfJet sjsmooth_intersection(SdfJet a,SdfJet b,SdfJet k) {
  return sjsubtraction(sjintersection(a,b),sjsmooth_union(sjsubtraction(a,b),sjsubtraction(b,a),k));
}
inline SdfJet sjgear2d(SdfJet x,SdfJet y,thread const float* a,
                       thread const float* alow) {
  SdfJet alpha=sjpair(a[0],alow[0],float3(0.0f));
  SdfJet D=sjpair(a[1],alow[1],float3(0.0f));
  SdfJet N=sjpair(a[2],alow[2],float3(0.0f));
  SdfJet psi=sjadd(sjsub(sjscale(sjmul(N,N),3.096e-5f),
                          sjscale(N,6.557e-3f)),sj(0.551f));
  SdfJet R=sjscale(D,0.5f), rho=sjlength2(x,y);
  SdfJet Pd=sjdiv(N,D), P=sjdiv(sj(kPluginSdfPi),Pd);
  SdfJet addendum=sjdiv(sj(1.0f),Pd);
  SdfJet Do=sjadd(D,sjscale(addendum,2.0f));
  SdfJet Ro=sjscale(Do,0.5f), h=sjdiv(sj(2.2f),Pd);
  SdfJet innerR=sjsub(sjsub(Ro,h),sjscale(D,0.14f));
  SdfJet inner=sjpair(a[4],alow[4],float3(0.0f));
  if(sjcompare(inner,sj(0.0f))>=0) innerR=sjscale(inner,0.5f);
  if(sjcompare(sjsub(innerR,rho),sj(0.0f))>0) return sjsub(innerR,rho);
  if(sjcompare(sjsub(Ro,rho),sj(-0.2f))<0) return sjsub(rho,Ro);
  SdfJet Rb=sjscale(sjmul(D,sjcos(psi)),0.5f);
  SdfJet fi=sjadd(sjatan2(y,x),alpha);
  SdfJet stride=sjdiv(P,R), invAlpha=sjacos(sjdiv(Rb,R));
  SdfJet invPhi=sjsub(sjtan(invAlpha),invAlpha);
  SdfJet shift=sjsub(sjscale(stride,0.5f),sjscale(invPhi,2.0f));
  SdfJet halfShift=sjscale(shift,0.5f);
  SdfJet fia=sjsub(sjmod(sjadd(fi,halfShift),stride),halfShift);
  SdfJet fib=sjsub(sjmod(sjadd(sjneg(fi),sjneg(halfShift)),stride),halfShift);
  SdfJet da=sj(-1.0e6f),db=sj(-1.0e6f);
  if(sjcompare(Rb,rho)<0) {
    SdfJet arc=sjacos(sjdiv(Rb,rho));
    SdfJet ta=sjsqrt(sjmax(sjsub(sjmul(rho,rho),sjmul(Rb,Rb)),sj(0.0f)));
    da=sjsub(ta,sjmul(sjadd(fia,arc),Rb));
    db=sjsub(ta,sjmul(sjadd(fib,arc),Rb));
  }
  SdfJet outer=sjsub(rho,Ro), low=sjsub(rho,sjsub(Ro,h));
  SdfJet crown=sjsub(rho,innerR), cogs=sjintersection(da,db);
  SdfJet wall=sjsub(stride,shift);
  SdfJet walls=sjintersection(sjsub(fia,wall),sjsub(fib,wall));
  cogs=sjintersection(walls,cogs);
  cogs=sjsmooth_intersection(outer,cogs,sjscale(D,0.0035f));
  cogs=sjsmooth_union(low,cogs,sjadd(sjsub(Rb,Ro),h));
  return sjsubtraction(cogs,crown);
}
inline SdfJet sjdistance_xyz(int kind,SdfJet x,SdfJet y,SdfJet z,
                             thread const float* a,
                             thread const float* alow) {
  if(kind==1 || kind==4) return sjbolt_nut(kind,x,y,z,a,alow);
  if(kind==2) return sjbowl(x,y,z,a,alow);
  if(kind==3) {
    SdfJet d=sjgear2d(x,y,a,alow);
    return sjextrusion(x,y,z,d,sjpair(a[3]*0.5f,alow[3]*0.5f,
                                      float3(0.0f)));
  }
  SdfJet q=sjsub(sjlength2(x,y),sjpair(a[0],alow[0],float3(0.0f)));
  return sjsub(sjlength2(q,z),sjpair(a[1],alow[1],float3(0.0f)));
}
inline SdfJet sjdistance(int kind,float3 p,thread const float* a,
                         thread const float* alow) {
  return sjdistance_xyz(kind,sj(p.x),sj(p.y),sj(p.z),a,alow);
}

// Three-float carrier for pinned binary64 operations in the SDF callback and
// descent paths. The expansion is rounded to nearest-even at each source
// operation boundary; retaining three words stores those 53 bits without
// Metal double and preserves low residuals through the 1e-8 bowl stencil.
inline SdfWide sw(float x) { return {x, 0.0f, 0.0f}; }
// Binary64 source constants decomposed into three binary32 words. Keep the
// source's decimal constants exact through the 1e-8 finite stencil and Wolfe
// backtracking instead of silently rounding them to one float word.
inline SdfWide sw_source_eps() {
  return {1.0e-8f, 6.077470660972466e-17f, 3.308722450212111e-24f};
}
inline SdfWide sw_source_minval() {
  return {1.0000000036274937e-15f, -3.627493647833322e-24f, 0.0f};
}
inline SdfWide sw_source_box_eps() {
  return {9.999999974752427e-7f, 2.5247572468697606e-15f, 0.0f};
}
inline SdfWide sw_source_point_one() {
  return {0.10000000149011612f, -1.4901161415892261e-9f,
          2.7755575615628914e-17f};
}
inline SdfWide sw_source_amin() {
  return {9.999999747378752e-5f, 2.5262125290942405e-12f,
          -4.0657581468206416e-20f};
}
struct SdfProjection53 { float hi; float mid; float low; };

// Expansion producers are bounded to 21 terms (nine product/error pairs
// plus three addends); rounding comparison scratch needs at most 22.
// A 24-word capacity leaves two slots of margin and limits per-call
// thread storage for the contact pipeline.
// Grow an exact float expansion by one term. The expansion is kept
// nonoverlapping, from least to most significant component, so a three-term
// dot product can be rounded to the 53-bit significand used by mjtNum.
static inline bool sw_float_is_subnormal(float value) {
  uint bits=as_type<uint>(value);
  return (bits&0x7f800000u)==0u && (bits&0x007fffffu)!=0u;
}

// Scale one binary32 word by an exact power of two without relying on MPS
// subnormal arithmetic.  The integer path implements IEEE round-to-nearest,
// ties-to-even when the scaled value crosses the subnormal boundary.
static inline uint sw_round_shift_rne(uint value, uint shift) {
  if (shift==0u) return value;
  if (shift>=32u) return 0u;
  uint quotient=value>>shift;
  uint mask=(1u<<shift)-1u;
  uint remainder=value&mask;
  uint halfway=1u<<(shift-1u);
  return quotient + uint(remainder>halfway ||
                         (remainder==halfway && (quotient&1u)));
}

static inline float sw_scale_word_pow2(float value, int power) {
  uint bits=as_type<uint>(value);
  uint sign=bits&0x80000000u;
  uint exponent=(bits>>23)&0xffu;
  uint fraction=bits&0x007fffffu;
  if (exponent==0xffu || (exponent==0u && fraction==0u)) return value;
  uint significand=exponent==0u ? fraction : (fraction|0x00800000u);
  int base_power=(exponent==0u ? -149 : int(exponent)-150)+power;
  int highest=base_power+int(31u-clz(significand));
  if (highest>=-126) {
    int right_shift=int(31u-clz(significand))-23;
    uint rounded=right_shift>0
        ? sw_round_shift_rne(significand,uint(right_shift))
        : (significand<<uint(-right_shift));
    if (rounded>=0x01000000u) {
      rounded>>=1;
      ++highest;
    }
    if (highest>127) return as_type<float>(sign|0x7f800000u);
    uint out_exp=uint(highest+127);
    return as_type<float>(sign|(out_exp<<23)|(rounded&0x007fffffu));
  }
  int unit_power=base_power+149;
  uint units=unit_power>=0
      ? (unit_power>=32 ? 0u : significand<<uint(unit_power))
      : sw_round_shift_rne(significand,uint(-unit_power));
  if (units>=0x00800000u)
    return as_type<float>(sign|0x00800000u);
  return as_type<float>(sign|units);
}

static inline SdfWide sw_scale_wide_pow2(SdfWide value,int power) {
  return {sw_scale_word_pow2(value.hi,power),
          sw_scale_word_pow2(value.mid,power),
          sw_scale_word_pow2(value.lo,power)};
}

static inline bool sw_wide_has_subnormal_leading(SdfWide value) {
  return sw_float_is_subnormal(value.hi);
}

// Return e such that |x| < 2^e for every finite nonzero binary32 x.
// Decoding the word also works when the input is subnormal and MPS arithmetic
// would flush it before frexp can inspect its exponent.
static inline int sw_float_power_upper(float value) {
  uint bits=as_type<uint>(value)&0x7fffffffu;
  uint exponent=(bits>>23)&0xffu;
  uint fraction=bits&0x007fffffu;
  if (exponent==0u) {
    if (fraction==0u) return -10000;
    return int(31u-clz(fraction))-148;
  }
  if (exponent==0xffu) return 10000;
  return int(exponent)-126;
}

static inline int sw_float_power_floor(float value) {
  uint bits=as_type<uint>(value)&0x7fffffffu;
  uint exponent=(bits>>23)&0xffu;
  uint fraction=bits&0x007fffffu;
  if (exponent==0u) {
    if (fraction==0u) return -10000;
    return int(31u-clz(fraction))-149;
  }
  if (exponent==0xffu) return 10000;
  return int(exponent)-127;
}

static __attribute__((noinline)) int sw_expansion_add(
    thread float* expansion, int count, float value,
    thread float* scratch) {
  // MPS arithmetic flushes subnormals to zero. Keep such exact input words
  // as low expansion components and only run TwoSum on normal operands.
  // They can still break a binary64 halfway comparison after the normal
  // components cancel, so dropping them here changes the source rounding.
  int out_count=0;
  if (sw_float_is_subnormal(value)) scratch[out_count++]=value;
  for (int i=0;i<count;i++) {
    if (sw_float_is_subnormal(expansion[i]))
      scratch[out_count++]=expansion[i];
  }
  float q=sw_float_is_subnormal(value) ? 0.0f : value;
  for (int i=0;i<count;i++) {
    float component=expansion[i];
    if (sw_float_is_subnormal(component)) continue;
    float sum=q+component;
    float virtual_component=sum-q;
    float error=(q-(sum-virtual_component))+(component-virtual_component);
    if (error!=0.0f) scratch[out_count++]=error;
    q=sum;
  }
  if (q!=0.0f || out_count==0) scratch[out_count++]=q;
  for (int i=0;i<out_count;i++) expansion[i]=scratch[i];
  return out_count;
}

// Add a product with subnormal-safe normalization while keeping both factors
// normal during multiply/FMA. A three-float carrier cannot encode a residual
// smaller than 2^-149 after rescaling; such terms round to the nearest
// representable carrier word here.
static __attribute__((noinline)) int sw_expansion_add_product(
    thread float* expansion,int count,float a,float b,int sign,
    thread float* scratch) {
  int ea=sw_float_power_floor(a);
  int eb=sw_float_power_floor(b);
  if (ea==-10000 || eb==-10000) return count;
  int scale_a=max(0,-126-ea);
  int scale_b=max(0,-126-eb);
  int product_exp=ea+eb+scale_a+scale_b;
  int extra=max(0,-126-product_exp);
  if (extra>0) {
    if (ea<=eb) scale_a+=extra;
    else scale_b+=extra;
  }
  int total_scale=scale_a+scale_b;
  float scaled_a=sw_scale_word_pow2(a,scale_a);
  float scaled_b=sw_scale_word_pow2(b,scale_b);
  float product=scaled_a*scaled_b;
  float error=fma(scaled_a,scaled_b,-product);
  float actual_error=sw_scale_word_pow2(error,-total_scale);
  float actual_product=sw_scale_word_pow2(product,-total_scale);
  if (sign<0) {
    actual_error=-actual_error;
    actual_product=-actual_product;
  }
  count=sw_expansion_add(expansion,count,actual_error,scratch);
  return sw_expansion_add(expansion,count,actual_product,scratch);
}

static __attribute__((noinline)) int sw_expansion_sign(
    thread const float* expansion, int count) {
  int subnormal_sum=0;
  for (int i=count-1;i>=0;i--) {
    uint bits=as_type<uint>(expansion[i]);
    uint magnitude=bits&0x7fffffffu;
    if (magnitude==0u) continue;
    if ((magnitude&0x7f800000u)==0u) {
      int mantissa=int(magnitude&0x007fffffu);
      subnormal_sum += (bits&0x80000000u) ? -mantissa : mantissa;
      continue;
    }
    // A normalized expansion is ordered by magnitude, so this highest
    // normal component dominates every subnormal term below it.
    return (bits&0x80000000u) ? -1 : 1;
  }
  return (subnormal_sum>0) - (subnormal_sum<0);
}

static __attribute__((noinline)) int sw_expansion_compare_value(
    thread const float* expansion, int count, float value) {
  thread float work[24];
  thread float scratch[24];
  int work_count=0;
  for (int i=0;i<count;i++) {
    work_count=sw_expansion_add(work,work_count,expansion[i],scratch);
  }
  work_count=sw_expansion_add(work,work_count,-value,scratch);
  return sw_expansion_sign(work,work_count);
}

// Round an exact expansion to the nearest binary64 significand, ties to even.
// Three float components hold the resulting 53 bits without using unsupported
// Metal `double`. This is for source-order mju_dot3 support selection only.
static inline float sw_float32_from_subnormal_units(int signed_units);
static __attribute__((noinline)) SdfProjection53 sw_expansion_round53(
    thread const float* expansion, int count) {
  SdfProjection53 result=SdfProjection53{0.0f,0.0f,0.0f};
  if (count<=0) return result;
  // Exact sums within the first normal binade are integer multiples of the
  // least-subnormal unit. If the total fits one binary32 significand, publish
  // it directly instead of asking MPS float arithmetic to combine subnormals.
  int boundary_units=0;
  bool boundary_exact=true;
  for (int i=0;i<count;i++) {
    uint bits=as_type<uint>(expansion[i]);
    uint exponent=(bits>>23)&0xffu;
    uint significand=bits&0x007fffffu;
    if (exponent>1u) { boundary_exact=false; break; }
    if (exponent==1u) significand|=0x00800000u;
    int units=int(significand);
    boundary_units+=(bits&0x80000000u) ? -units : units;
  }
  if (boundary_exact && abs(boundary_units)<0x01000000)
    return {sw_float32_from_subnormal_units(boundary_units),0.0f,0.0f};
  result.hi=expansion[count-1];
  if (count==1) return result;
  int exponent=0;
  float significand=frexp(fabs(result.hi),exponent);
  int tail_sign=sw_expansion_sign(expansion,count-1);
  // At an exact power-of-two leading component, values just toward zero lie
  // in the lower binade, whose ulp is half as large. Choose that grid before
  // quantizing the complete tail (including negative projections).
  bool toward_zero=(result.hi>0.0f && tail_sign<0)
                   || (result.hi<0.0f && tail_sign>0);
  if (significand==0.5f && toward_zero) exponent-=1;
  int quantum_exp=exponent-53;
  thread float residual[24];
  thread float scratch[24];
  int residual_count=0;
  float estimate=0.0f;
  // Quantize every component below `hi`, including a widely separated
  // second component. This pass is only an estimate: a much smaller term can
  // still decide which side of an exact halfway value the expansion lies on.
  for (int i=0;i<count-1;i++) {
    float scaled=sw_scale_word_pow2(expansion[i],-quantum_exp);
    residual_count=sw_expansion_add(residual,residual_count,
                                      expansion[i],scratch);
    estimate+=scaled;
  }
  // Split the integer number of 2^(e-53) quanta into a float32 high part and
  // a small exact correction. `rint` supplies a nearby integral estimate.
  float step_hi=rint(estimate);
  residual_count=sw_expansion_add(
      residual,residual_count,-sw_scale_word_pow2(step_hi,quantum_exp),scratch);
  float residual_estimate=0.0f;
  for (int i=residual_count-1;i>=0;i--)
    residual_estimate+=sw_scale_word_pow2(residual[i],-quantum_exp);
  float lower=floor(residual_estimate);
  // Correct the approximate integer using the unscaled exact remainder. This
  // avoids underflow when a tiny expansion word breaks a halfway tie.
  if (sw_expansion_compare_value(
          residual,residual_count,sw_scale_word_pow2(lower,quantum_exp))<0)
    lower-=1.0f;
  if (sw_expansion_compare_value(
          residual,residual_count,sw_scale_word_pow2(lower+1.0f,quantum_exp))>=0)
    lower+=1.0f;
  int half_cmp=sw_expansion_compare_value(
      residual,residual_count,sw_scale_word_pow2(lower+0.5f,quantum_exp));
  bool tie_even=((int(step_hi)+int(lower))&1)==0;
  float step_lo=(half_cmp<0 ? lower
                  : (half_cmp>0 ? lower+1.0f
                     : (tie_even ? lower : lower+1.0f)));
  result.mid=sw_scale_word_pow2(step_hi,quantum_exp);
  result.low=sw_scale_word_pow2(step_lo,quantum_exp);
  return result;
}


static inline float sw_float32_from_subnormal_units(int signed_units) {
  if (signed_units == 0) return 0.0f;
  bool negative = signed_units < 0;
  uint magnitude = uint(negative ? -signed_units : signed_units);
  uint sign_bit = negative ? 0x80000000u : 0u;
  if (magnitude < 0x00800000u)
    return as_type<float>(sign_bit | magnitude);

  uint highest = 31u - clz(magnitude);
  int shift = max(0, int(highest) - 23);
  uint significand = magnitude >> uint(shift);
  if (shift > 0) {
    uint mask = (1u << uint(shift)) - 1u;
    uint remainder = magnitude & mask;
    uint halfway = 1u << uint(shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (significand & 1u))) {
      ++significand;
    }
  }
  if (significand == 0x01000000u) {
    significand >>= 1;
    ++shift;
  }
  uint exponent = uint(shift + 1);
  uint bits = exponent >= 255u ? 0x7f800000u
      : (exponent << 23) | (significand - 0x00800000u);
  return as_type<float>(sign_bit | bits);
}

static __attribute__((noinline)) float sw_expansion_round_float32(
    thread const float* expansion,int count) {
  if (count<=0) return 0.0f;
  float hi=expansion[count-1];
  if (count==1) return hi;
  uint hi_bits=as_type<uint>(hi);
  uint hi_exponent=(hi_bits>>23)&0xffu;
  // For subnormal and minimum-normal leading words, all binary32 expansion
  // components are integer multiples of 2^-149. Sum and round those units
  // with integer operations so MPS flush-to-zero cannot erase a result word.
  if (hi_exponent<=1u) {
    int units=0;
    for (int i=0;i<count;i++) {
      uint bits=as_type<uint>(expansion[i]);
      uint exponent=(bits>>23)&0xffu;
      uint significand=bits&0x007fffffu;
      if (exponent!=0u) significand|=0x00800000u;
      uint shift=exponent==0u ? 0u : exponent-1u;
      int word_units=int(significand<<shift);
      units += (bits&0x80000000u) ? -word_units : word_units;
    }
    return sw_float32_from_subnormal_units(units);
  }
  int exponent=0;
  float significand=frexp(fabs(hi),exponent);
  int tail_sign=sw_expansion_sign(expansion,count-1);
  bool toward_zero=(hi>0.0f && tail_sign<0)
                   || (hi<0.0f && tail_sign>0);
  if (significand==0.5f && toward_zero) exponent-=1;
  int quantum_exp=exponent-24;
  thread float residual[24];
  thread float scratch[24];
  int residual_count=0;
  float estimate=0.0f;
  for (int i=0;i<count-1;i++) {
    residual_count=sw_expansion_add(residual,residual_count,
                                    expansion[i],scratch);
    estimate+=sw_scale_word_pow2(expansion[i],-quantum_exp);
  }
  float step_hi=rint(estimate);
  residual_count=sw_expansion_add(residual,residual_count,
      -sw_scale_word_pow2(step_hi,quantum_exp),scratch);
  float residual_estimate=0.0f;
  for (int i=residual_count-1;i>=0;i--)
    residual_estimate+=sw_scale_word_pow2(residual[i],-quantum_exp);
  float lower=floor(residual_estimate);
  if (sw_expansion_compare_value(
          residual,residual_count,sw_scale_word_pow2(lower,quantum_exp))<0)
    lower-=1.0f;
  if (sw_expansion_compare_value(
          residual,residual_count,sw_scale_word_pow2(lower+1.0f,quantum_exp))>=0)
    lower+=1.0f;
  int half_cmp=sw_expansion_compare_value(
      residual,residual_count,sw_scale_word_pow2(lower+0.5f,quantum_exp));
  // The leading word is an integral number of output quanta. Its significand
  // parity participates in ties along with the rounded tail quanta.
  int leading_quanta=int(sw_scale_word_pow2(hi,-quantum_exp));
  bool tie_even=((leading_quanta+int(step_hi)+int(lower))&1)==0;
  float step_lo=(half_cmp<0 ? lower
      : (half_cmp>0 ? lower+1.0f
         : (tie_even ? lower : lower+1.0f)));
  float rounded_tail=sw_scale_word_pow2(step_hi+step_lo,quantum_exp);
  // A zero rounded tail must not route a preserved subnormal leading word
  // through a flush-to-zero addition.
  if (rounded_tail==0.0f) return hi;
  return hi+rounded_tail;
}
inline float sw_float32(SdfWide value) {
  thread float expansion[3]={value.lo,value.mid,value.hi};
  return sw_expansion_round_float32(expansion,3);
}

inline float2 sw_two_sum(float a, float b) {
  float s = a + b;
  float bb = s - a;
  return float2(s, (a - (s - bb)) + (b - bb));
}
inline SdfWide sw_add_word(SdfWide a, float b) {
  float2 x = sw_two_sum(a.hi, b);
  float2 y = sw_two_sum(a.mid, x.y);
  float2 z = sw_two_sum(a.lo, y.y);
  float2 h = sw_two_sum(x.x, y.x);
  float2 m = sw_two_sum(h.y, z.x);
  float2 n = sw_two_sum(h.x, m.x);
  float2 l = sw_two_sum(m.y, z.y);
  float2 k = sw_two_sum(n.y, l.x);
  float2 q = sw_two_sum(n.x, k.x);
  return {q.x, q.y, k.y + l.y};
}
[[clang::noinline]] inline SdfWide sw_source_add(SdfWide a, SdfWide b) {
  thread float expansion[24];
  thread float scratch[24];
  int count=0;
  count=sw_expansion_add(expansion,count,a.lo,scratch);
  count=sw_expansion_add(expansion,count,a.mid,scratch);
  count=sw_expansion_add(expansion,count,a.hi,scratch);
  count=sw_expansion_add(expansion,count,b.lo,scratch);
  count=sw_expansion_add(expansion,count,b.mid,scratch);
  count=sw_expansion_add(expansion,count,b.hi,scratch);
  SdfProjection53 rounded=sw_expansion_round53(expansion,count);
  return {rounded.hi,rounded.mid,rounded.low};
}
[[clang::noinline]] inline SdfWide sw_source_mul_core(SdfWide a, SdfWide b) {
  thread float expansion[24];
  thread float scratch[24];
  int count=0;
  float av[3]={a.lo,a.mid,a.hi};
  float bv[3]={b.lo,b.mid,b.hi};
  for (int i=0;i<3;i++) {
    for (int j=0;j<3;j++) {
      count=sw_expansion_add_product(expansion,count,av[i],bv[j],1,scratch);
    }
  }
  SdfProjection53 rounded=sw_expansion_round53(expansion,count);
  return {rounded.hi,rounded.mid,rounded.low};
}
[[clang::noinline]] inline SdfWide sw_source_mul(SdfWide a,SdfWide b) {
  int scale_a=sw_wide_has_subnormal_leading(a) ? 23 : 0;
  int scale_b=sw_wide_has_subnormal_leading(b) ? 23 : 0;
  if (scale_a==0 && scale_b==0) return sw_source_mul_core(a,b);
  SdfWide product=sw_source_mul_core(sw_scale_wide_pow2(a,scale_a),
                                      sw_scale_wide_pow2(b,scale_b));
  return sw_scale_wide_pow2(product,-scale_a-scale_b);
}
inline SdfWide sw_neg(SdfWide a) { return {-a.hi,-a.mid,-a.lo}; }
inline SdfWide sw_add(SdfWide a, SdfWide b) { return sw_source_add(a,b); }
inline SdfWide sw_sub(SdfWide a, SdfWide b) {
  return sw_source_add(a,sw_neg(b));
}
inline SdfWide sw_mul(SdfWide a, SdfWide b) { return sw_source_mul(a,b); }
[[clang::noinline]] inline SdfWide sw_source_fma_core(
    SdfWide a,SdfWide b,SdfWide c) {
  thread float expansion[24];
  thread float scratch[24];
  int count=0;
  float av[3]={a.lo,a.mid,a.hi};
  float bv[3]={b.lo,b.mid,b.hi};
  for (int i=0;i<3;i++) {
    for (int j=0;j<3;j++) {
      count=sw_expansion_add_product(expansion,count,av[i],bv[j],1,scratch);
    }
  }
  count=sw_expansion_add(expansion,count,c.lo,scratch);
  count=sw_expansion_add(expansion,count,c.mid,scratch);
  count=sw_expansion_add(expansion,count,c.hi,scratch);
  SdfProjection53 rounded=sw_expansion_round53(expansion,count);
  return {rounded.hi,rounded.mid,rounded.low};
}
[[clang::noinline]] inline SdfWide sw_source_fma(
    SdfWide a,SdfWide b,SdfWide c) {
  int scale_a=sw_wide_has_subnormal_leading(a) ? 23 : 0;
  int scale_b=sw_wide_has_subnormal_leading(b) ? 23 : 0;
  int total_scale=scale_a+scale_b;
  if (total_scale==0) return sw_source_fma_core(a,b,c);
  // If scaling c would overflow, prove that the exact product is far below
  // half a binary64 ulp of this exact float32 c. In that bounded case the
  // fused result rounds to c, without scaling c or dropping a relevant term.
  if (c.mid==0.0f && c.lo==0.0f && c.hi!=0.0f) {
    int a_exp=sw_float_power_upper(a.hi);
    int b_exp=sw_float_power_upper(b.hi);
    int c_exp=sw_float_power_upper(c.hi);
    bool scaling_c_overflows=(c_exp+total_scale)>128;
    bool product_below_half_ulp=(a_exp+b_exp+2<=c_exp-55);
    if (scaling_c_overflows && product_below_half_ulp) return c;
  }
  SdfWide scaled_c=sw_scale_wide_pow2(c,total_scale);
  SdfWide result=sw_source_fma_core(
      sw_scale_wide_pow2(a,scale_a),sw_scale_wide_pow2(b,scale_b),scaled_c);
  return sw_scale_wide_pow2(result,-total_scale);
}
inline int sw_sign(SdfWide a) {
  thread float expansion[3]={a.lo,a.mid,a.hi};
  return sw_expansion_sign(expansion,3);
}
[[clang::noinline]] inline SdfWide sw_division_residual(
    SdfWide a,SdfWide b,float q0,float q1,float q2) {
  thread float expansion[24];
  thread float scratch[24];
  int count=0;
  count=sw_expansion_add(expansion,count,a.lo,scratch);
  count=sw_expansion_add(expansion,count,a.mid,scratch);
  count=sw_expansion_add(expansion,count,a.hi,scratch);
  float denominator[3]={b.lo,b.mid,b.hi};
  float quotient[3]={q0,q1,q2};
  for (int qi=0;qi<3;qi++) {
    for (int bi=0;bi<3;bi++) {
      count=sw_expansion_add_product(expansion,count,denominator[bi],
                                     quotient[qi],-1,scratch);
    }
  }
  SdfProjection53 rounded=sw_expansion_round53(expansion,count);
  return {rounded.hi,rounded.mid,rounded.low};
}
[[clang::noinline]] inline SdfWide sw_product_residual(SdfWide value,SdfWide a,SdfWide b) {
  thread float expansion[24];
  thread float scratch[24];
  int count=0;
  count=sw_expansion_add(expansion,count,value.lo,scratch);
  count=sw_expansion_add(expansion,count,value.mid,scratch);
  count=sw_expansion_add(expansion,count,value.hi,scratch);
  float av[3]={a.lo,a.mid,a.hi};
  float bv[3]={b.lo,b.mid,b.hi};
  for (int i=0;i<3;i++) {
    for (int j=0;j<3;j++) {
      count=sw_expansion_add_product(expansion,count,av[i],bv[j],-1,scratch);
    }
  }
  SdfProjection53 rounded=sw_expansion_round53(expansion,count);
  return {rounded.hi,rounded.mid,rounded.low};
}
[[clang::noinline]] inline SdfWide sw_div_core(SdfWide a, SdfWide b) {
  float q0=a.hi/b.hi;
  SdfWide remainder=sw_division_residual(a,b,q0,0.0f,0.0f);
  float q1=((remainder.hi+remainder.mid)+remainder.lo)/b.hi;
  remainder=sw_division_residual(a,b,q0,q1,0.0f);
  float q2=((remainder.hi+remainder.mid)+remainder.lo)/b.hi;
  remainder=sw_division_residual(a,b,q0,q1,q2);
  float q3=((remainder.hi+remainder.mid)+remainder.lo)/b.hi;
  thread float expansion[24];
  thread float scratch[24];
  int count=0;
  count=sw_expansion_add(expansion,count,q0,scratch);
  count=sw_expansion_add(expansion,count,q1,scratch);
  count=sw_expansion_add(expansion,count,q2,scratch);
  count=sw_expansion_add(expansion,count,q3,scratch);
  SdfProjection53 rounded=sw_expansion_round53(expansion,count);
  return {rounded.hi,rounded.mid,rounded.low};
}
[[clang::noinline]] inline SdfWide sw_div(SdfWide a,SdfWide b) {
  if (b.hi==0.0f && b.mid==0.0f && b.lo==0.0f)
    return sw_div_core(a,b);
  int ea=sw_float_power_floor(a.hi);
  int eb=sw_float_power_floor(b.hi);
  if (ea==-10000) return sw(0.0f);
  int scale_a=max(0,-126-ea);
  int scale_b=max(0,-126-eb);
  int quotient_exp=ea+scale_a-eb-scale_b;
  if (quotient_exp< -126) scale_a+=(-126-quotient_exp);
  if (scale_a==0 && scale_b==0) return sw_div_core(a,b);
  SdfWide scaled=sw_div_core(sw_scale_wide_pow2(a,scale_a),
                             sw_scale_wide_pow2(b,scale_b));
  return sw_scale_wide_pow2(scaled,scale_b-scale_a);
}
[[clang::noinline]] inline SdfWide sw_sqrt_core(SdfWide a) {
  if (a.hi==0.0f && a.mid==0.0f && a.lo==0.0f) return sw(0.0f);
  float q0=sqrt(a.hi);
  SdfWide q0_value=sw(q0);
  SdfWide remainder=sw_product_residual(a,q0_value,q0_value);
  float q1=((remainder.hi+remainder.mid)+remainder.lo)/(2.0f*q0);
  SdfWide approximation=sw_source_add(q0_value,sw(q1));
  remainder=sw_product_residual(a,approximation,approximation);
  float q2=((remainder.hi+remainder.mid)+remainder.lo)/(2.0f*q0);
  thread float expansion[24];
  thread float scratch[24];
  int count=0;
  // q2 corrects the rounded 53-bit approximation, not the unrounded q0+q1
  // pair. Re-summing q0 and q1 here would add back the rounding residual
  // already included when q2 was computed.
  count=sw_expansion_add(expansion,count,approximation.lo,scratch);
  count=sw_expansion_add(expansion,count,approximation.mid,scratch);
  count=sw_expansion_add(expansion,count,approximation.hi,scratch);
  count=sw_expansion_add(expansion,count,q2,scratch);
  SdfProjection53 rounded=sw_expansion_round53(expansion,count);
  return {rounded.hi,rounded.mid,rounded.low};
}
[[clang::noinline]] inline SdfWide sw_sqrt(SdfWide a) {
  // MPS may flush subnormal comparisons to zero. Inspect the leading word's
  // bits before the numeric zero test so a nonzero subnormal reaches scaling.
  bool has_subnormal_leading=sw_wide_has_subnormal_leading(a);
  if (!has_subnormal_leading &&
      a.hi==0.0f && a.mid==0.0f && a.lo==0.0f) return sw(0.0f);
  int exponent=sw_float_power_floor(a.hi);
  // q1 is roughly one float32 ulp below sqrt(a); q1*q1 enters the second
  // residual. Raise small inputs to the even binade near 2^-48 so that this
  // square and the residual division stay normal, then undo half the scale.
  if (exponent < -78) {
    int scale=-48-exponent;
    if (scale&1) ++scale;
    return sw_scale_wide_pow2(
        sw_sqrt_core(sw_scale_wide_pow2(a,scale)),-scale/2);
  }
  return sw_sqrt_core(a);
}
inline SdfWide sw_abs(SdfWide a) { return sw_sign(a) < 0 ? sw_neg(a) : a; }
inline float sw_value(SdfWide a) { return (a.hi + a.mid) + a.lo; }
inline SdfWide sw_norm2(SdfWide x, SdfWide y) {
  // Pinned arm64 mju_norm(n=2) rounds y*y, then fuses x*x into that
  // accumulator before sqrt. Preserve that binary64 FMA boundary.
  return sw_sqrt(sw_source_fma(x, x, sw_mul(y, y)));
}
// Match the pinned arm64 mju_dot3 contraction: round y*y first, then fuse
// x*x and z*z into the accumulator in that source order (as verified in the
// shared source-rounded CCD primitives).
inline SdfWide sw_source_dot3(SdfWide x,SdfWide y,SdfWide z,
                              SdfWide u,SdfWide v,SdfWide w) {
  SdfWide y_product=sw_mul(y,v);
  SdfWide xy=sw_source_fma(x,u,y_product);
  return sw_source_fma(z,w,xy);
}
inline SdfWide sw_bowl_distance(SdfWide x, SdfWide y, SdfWide z,
                                thread const float* a,
                                thread const float* alow,
                                thread const float* atail) {
  SdfWide h = sw_add(sw_add(sw(a[0]), sw(alow[0])), sw(atail[0]));
  SdfWide radius = sw_add(sw_add(sw(a[1]), sw(alow[1])), sw(atail[1]));
  SdfWide thick = sw_add(sw_add(sw(a[2]), sw(alow[2])), sw(atail[2]));
  SdfWide width = sw_sqrt(sw_sub(sw_mul(radius, radius), sw_mul(h, h)));
  SdfWide qx = sw_norm2(x, y), qy = z;
  if (sw_sign(sw_sub(sw_mul(h, qx), sw_mul(width, qy))) < 0) {
    return sw_sub(sw_norm2(sw_sub(qx, width), sw_sub(qy, h)), thick);
  }
  return sw_sub(sw_abs(sw_sub(sw_norm2(qx, qy), radius)), thick);
}
inline SdfWide sw_torus_distance(SdfWide x, SdfWide y, SdfWide z,
                                thread const float* a,
                                thread const float* alow,
                                thread const float* atail) {
  SdfWide radius1 = sw_add(sw_add(sw(a[0]), sw(alow[0])), sw(atail[0]));
  SdfWide radius2 = sw_add(sw_add(sw(a[1]), sw(alow[1])), sw(atail[1]));
  SdfWide q = sw_sub(sw_norm2(x, y), radius1);
  return sw_sub(sw_norm2(q, z), radius2);
}
inline SdfWide sw_torus_gradient_component(SdfWide p, SdfWide q,
                                           SdfWide lenxy,
                                           SdfWide lenqz,
                                           SdfWide denominator) {
  return sw_div(sw_mul(q, sw_div(p, lenxy)), denominator);
}
inline void plugin_sdf_source_gradient_wide_bowl(
    float3 p_hi, float3 p_lo, float3 p_tail, thread const float* a,
    thread const float* alow, thread const float* atail,
    thread float3* output) {
  SdfWide eps = sw_source_eps();
  SdfWide p[3] = {
      sw_add(sw_add(sw(p_hi.x), sw(p_lo.x)), sw(p_tail.x)),
      sw_add(sw_add(sw(p_hi.y), sw(p_lo.y)), sw(p_tail.y)),
      sw_add(sw_add(sw(p_hi.z), sw(p_lo.z)), sw(p_tail.z))};
  SdfWide base = sw_bowl_distance(p[0], p[1], p[2], a, alow, atail);
  for (int axis = 0; axis < 3; ++axis) {
    SdfWide shifted[3] = {p[0], p[1], p[2]};
    shifted[axis] = sw_add(shifted[axis], eps);
    SdfWide value = sw_bowl_distance(shifted[0], shifted[1], shifted[2],
                                     a, alow, atail);
    SdfWide gradient = sw_div(sw_sub(value, base), eps);
    output[axis] = float3(gradient.hi, gradient.mid, gradient.lo);
  }
}
inline void plugin_sdf_torus_gradient_wide(
    float3 p_hi, float3 p_lo, float3 p_tail, thread const float* a,
    thread const float* alow, thread const float* atail,
    thread float3* output) {
  SdfWide x = sw_add(sw_add(sw(p_hi.x), sw(p_lo.x)), sw(p_tail.x));
  SdfWide y = sw_add(sw_add(sw(p_hi.y), sw(p_lo.y)), sw(p_tail.y));
  SdfWide z = sw_add(sw_add(sw(p_hi.z), sw(p_lo.z)), sw(p_tail.z));
  SdfWide radius = sw_add(sw_add(sw(a[0]), sw(alow[0])), sw(atail[0]));
  SdfWide lenxy = sw_norm2(x, y);
  SdfWide q = sw_sub(lenxy, radius);
  SdfWide lenqz = sw_norm2(q, z);
  SdfWide minval = sw_source_minval();
  SdfWide denominator = sw_sign(sw_sub(lenqz, minval)) > 0
      ? lenqz : minval;
  SdfWide gx = sw_div(sw_mul(q, sw_div(x, lenxy)), denominator);
  SdfWide gy = sw_div(sw_mul(q, sw_div(y, lenxy)), denominator);
  SdfWide gz = sw_div(z, denominator);
  output[0] = float3(gx.hi, gx.mid, gx.lo);
  output[1] = float3(gy.hi, gy.mid, gy.lo);
  output[2] = float3(gz.hi, gz.mid, gz.lo);
}
inline float3 plugin_sdf_source_gradient(int kind, float3 p,
                                         thread const float* a,
                                         thread const float* alow) {
  if (kind == 2) {
    float3 wide[3];
    float3 zero = float3(0.0f);
    float atail[5] = {0, 0, 0, 0, 0};
    plugin_sdf_source_gradient_wide_bowl(
        p, zero, zero, a, alow, atail, wide);
    return float3((wide[0].x + wide[0].y) + wide[0].z,
                  (wide[1].x + wide[1].y) + wide[1].z,
                  (wide[2].x + wide[2].y) + wide[2].z);
  }
  // MuJoCo 3.10's bolt/bowl/gear/nut callbacks use a positive, one-sided
  // finite difference with mjtNum eps=1e-8.  Keep the perturbation as a
  // low word: adding it directly to a float point erases the step at normal
  // contact coordinates and changes the callback contract.
  constexpr float kGradientEpsHi = 1.0e-8f;
  SdfJet base = sjdistance(kind,p,a,alow);
  float result[3];
  for (int axis=0;axis<3;axis++) {
    SdfJet x=sj(p.x),y=sj(p.y),z=sj(p.z);
    if (axis==0) { x.e=kGradientEpsHi; }
    else if (axis==1) { y.e=kGradientEpsHi; }
    else { z.e=kGradientEpsHi; }
    SdfJet shifted=sjdistance_xyz(kind,x,y,z,a,alow);
    SdfJet delta=sjsub(shifted,base);
    result[axis]=(delta.v+delta.e)/kGradientEpsHi;
  }
  return float3(result[0],result[1],result[2]);
}
inline float3 plugin_sdf_gradient(int kind, float3 p, thread const float* a,
                                  thread const float* alow) {
  // Torus preserves the upstream singular gradient at its axis; the query
  // status reports non-finite values instead of manufacturing a normal.
  if(kind==5) {
    float3 wide[3];
    float3 zero = float3(0.0f);
    float atail[5] = {0, 0, 0, 0, 0};
    plugin_sdf_torus_gradient_wide(
        p, zero, zero, a, alow, atail, wide);
    return float3((wide[0].x + wide[0].y) + wide[0].z,
                  (wide[1].x + wide[1].y) + wide[1].z,
                  (wide[2].x + wide[2].y) + wide[2].z);
  }
  return plugin_sdf_source_gradient(kind,p,a,alow);
}

inline SdfJet plugin_sdf_device_jet(int kind, float3 p,
                                    device const float* values, int base) {
  float attributes[5], attributes_low[5];
  for (int i = 0; i < 5; ++i) {
    attributes[i] = values[base + i];
    attributes_low[i] = values[base + 5 + i];
  }
  return sjdistance(kind, p, attributes, attributes_low);
}

// Contact descent keeps the source point as a three-word expansion. Evaluate
// the bundled callback at that represented point instead of rounding it to
// float before evaluating its objective or its positive one-sided gradient.
inline SdfJet plugin_sdf_device_jet_pair(int kind, float3 p_hi, float3 p_lo,
                                         float3 p_tail,
                                         device const float* values,
                                         int base, int attribute_tail_base) {
  float attributes[5], attributes_low[5], attributes_tail[5];
  for (int i = 0; i < 5; ++i) {
    attributes[i] = values[base + i];
    attributes_low[i] = values[base + 5 + i];
    attributes_tail[i] = attribute_tail_base >= 0
        ? values[attribute_tail_base + i] : 0.0f;
  }
  if (kind == 2 || kind == 5) {
    SdfWide x = sw_add(sw_add(sw(p_hi.x), sw(p_lo.x)), sw(p_tail.x));
    SdfWide y = sw_add(sw_add(sw(p_hi.y), sw(p_lo.y)), sw(p_tail.y));
    SdfWide z = sw_add(sw_add(sw(p_hi.z), sw(p_lo.z)), sw(p_tail.z));
    SdfWide distance = kind == 2
        ? sw_bowl_distance(x, y, z, attributes, attributes_low,
                           attributes_tail)
        : sw_torus_distance(x, y, z, attributes, attributes_low,
                            attributes_tail);
    return sjwide(distance, float3(0.0f));
  }
  float3 zero = float3(0.0f);
  return sjdistance_xyz(kind,
      sjpair(p_hi.x, p_lo.x + p_tail.x, zero),
      sjpair(p_hi.y, p_lo.y + p_tail.y, zero),
      sjpair(p_hi.z, p_lo.z + p_tail.z, zero), attributes, attributes_low);
}

inline float3 plugin_sdf_device_gradient(int kind, float3 p,
                                         device const float* values,
                                         int base);

inline void plugin_sdf_device_gradient_pair(
    int kind, float3 p_hi, float3 p_lo, float3 p_tail,
    device const float* values, int base, int attribute_tail_base,
    thread float3* output) {
  if (kind == 5) {
    // Preserve the source's analytic formula and singular axis behavior,
    // carrying point and attribute residuals through its binary64 operations.
    float attributes[5], attributes_low[5], attributes_tail[5];
    for (int i = 0; i < 5; ++i) {
      attributes[i] = values[base + i];
      attributes_low[i] = values[base + 5 + i];
      attributes_tail[i] = attribute_tail_base >= 0
          ? values[attribute_tail_base + i] : 0.0f;
    }
    plugin_sdf_torus_gradient_wide(p_hi, p_lo, p_tail, attributes,
                                   attributes_low, attributes_tail, output);
    return;
  }
  if (kind == 2) {
    float attributes[5], attributes_low[5], attributes_tail[5];
    for (int i = 0; i < 5; ++i) {
      attributes[i] = values[base + i];
      attributes_low[i] = values[base + 5 + i];
      attributes_tail[i] = attribute_tail_base >= 0
          ? values[attribute_tail_base + i] : 0.0f;
    }
    plugin_sdf_source_gradient_wide_bowl(p_hi, p_lo, p_tail, attributes,
                                         attributes_low, attributes_tail,
                                         output);
    return;
  }
  SdfWide eps = sw_source_eps();
  float attributes[5], attributes_low[5];
  for (int i = 0; i < 5; ++i) {
    attributes[i] = values[base + i];
    attributes_low[i] = values[base + 5 + i];
  }
  SdfJet base_value = sjdistance_xyz(kind,
      sjpair(p_hi.x, p_lo.x, float3(0.0f)),
      sjpair(p_hi.y, p_lo.y, float3(0.0f)),
      sjpair(p_hi.z, p_lo.z, float3(0.0f)), attributes, attributes_low);
  SdfWide result[3];
  for (int axis = 0; axis < 3; ++axis) {
    SdfJet x = sjpair(p_hi.x, p_lo.x, float3(0.0f));
    SdfJet y = sjpair(p_hi.y, p_lo.y, float3(0.0f));
    SdfJet z = sjpair(p_hi.z, p_lo.z, float3(0.0f));
    if (axis == 0) x = sjadd(x, sjwide(eps));
    else if (axis == 1) y = sjadd(y, sjwide(eps));
    else z = sjadd(z, sjwide(eps));
    SdfJet shifted = sjdistance_xyz(kind, x, y, z, attributes,
                                    attributes_low);
    SdfJet delta = sjsub(shifted, base_value);
    result[axis] = sw_div(sjwide_value(delta), eps);
  }
  output[0] = float3(result[0].hi,result[0].mid,result[0].lo);
  output[1] = float3(result[1].hi,result[1].mid,result[1].lo);
  output[2] = float3(result[2].hi,result[2].mid,result[2].lo);
}

// Rigid contact consumes compact high/low attributes from its geometry
// backing. Keep the plugin query ABI (thread arrays) shared with ray queries.
inline float plugin_sdf_device_distance(int kind, float3 p,
                                       device const float* values, int base) {
  return sjvalue(plugin_sdf_device_jet(kind, p, values, base));
}

inline float3 plugin_sdf_device_gradient(int kind, float3 p,
                                         device const float* values,
                                         int base) {
  float attributes[5], attributes_low[5];
  for (int i = 0; i < 5; ++i) {
    attributes[i] = values[base + i];
    attributes_low[i] = values[base + 5 + i];
  }
  return plugin_sdf_gradient(kind, p, attributes, attributes_low);
}

kernel void query_bundled_plugin_sdf(
    device const float* points [[buffer(0)]],
    device const int* plugin_instance [[buffer(1)]],
    device const int* instance_kind [[buffer(2)]],
    device const float* plugin_attributes [[buffer(3)]],
    device const float* plugin_attributes_low [[buffer(8)]],
    device float* distance [[buffer(4)]],
    device float* gradient [[buffer(5)]],
    device int* status [[buffer(6)]],
    constant int* dims [[buffer(7)]],
  uint gid [[thread_position_in_grid]]) {
  int batch = dims[0], slots = dims[1], nplugin = dims[2];
  int index = int(gid);
  if (index >= batch * slots) return;
  int instance = plugin_instance[index];
  if (instance < 0) {
    distance[index] = 0.0f;
    gradient[3 * index] = gradient[3 * index + 1] = gradient[3 * index + 2] = 0.0f;
    status[index] = 0;
    return;
  }
  if (instance >= nplugin) {
    distance[index] = 0.0f;
    gradient[3 * index] = gradient[3 * index + 1] = gradient[3 * index + 2] = 0.0f;
    status[index] = 1;
    return;
  }
  int kind = instance_kind[instance];
  thread float attr[5];
  for (int i = 0; i < 5; ++i) attr[i] = plugin_attributes[instance * 5 + i];
  float3 point = float3(points[3 * index], points[3 * index + 1], points[3 * index + 2]);
  float d = plugin_sdf_distance(kind, point, attr);
  thread float attr_low[5];
  for (int i = 0; i < 5; ++i) attr_low[i] = plugin_attributes_low[instance * 5 + i];
  float3 g = plugin_sdf_gradient(kind, point, attr, attr_low);
  bool valid = kind >= 1 && kind <= 5 && isfinite(d) && all(isfinite(g));
  distance[index] = valid ? d : 0.0f;
  gradient[3 * index] = valid ? g.x : 0.0f;
  gradient[3 * index + 1] = valid ? g.y : 0.0f;
  gradient[3 * index + 2] = valid ? g.z : 0.0f;
  status[index] = valid ? 0 : 1;
}
