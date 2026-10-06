// Extracted source-safe CCD primitive unit. No detector kernels are included.
#include <metal_stdlib>
using namespace metal;

static constant int FLEX_CCD_EPA_UNRESOLVED = 31;
static constant int FLEX_CCD_EPA_BAD_ADJACENCY = 33;

struct FlexDD {
  float hi;
  float lo;
  float tail;
};

struct FlexProjection53 {
  float hi;
  float mid;
  float low;
};

struct FlexDD3 {
  FlexDD x, y, z;
};

struct FlexDD2 {
  FlexDD x, y;
};

struct FlexDDVertex {
  FlexDD3 a, b, m;
};

static inline FlexDD flex_dd_source_add(FlexDD a, FlexDD b);
static inline FlexDD flex_dd_source_mul(FlexDD a, FlexDD b);
static inline FlexDD flex_dd_source_sqrt(FlexDD value);
static inline FlexDD flex_dd_source_div(FlexDD a, FlexDD b);
static inline FlexDD flex_dd_abs(FlexDD value);
static inline int flex_dd_compare_exact(FlexDD a, FlexDD b);
[[clang::noinline]] static inline FlexDD flex_dd_source_matvec_row(
    FlexDD m0, FlexDD m1, FlexDD m2,
    FlexDD x0, FlexDD x1, FlexDD x2);

static inline FlexDD flex_dd(float value) {
  return FlexDD{value, 0.0f, 0.0f};
}

static inline FlexDD flex_dd_renorm(float high, float low) {
  float sum=high+low;
  float error=low-(sum-high);
  return FlexDD{sum,error,0.0f};
}

static inline FlexDD flex_dd_two_sum(float a, float b) {
  float sum=a+b;
  float virtual_b=sum-a;
  return FlexDD{sum,(a-(sum-virtual_b))+(b-virtual_b),0.0f};
}

static inline FlexDD flex_dd_add(FlexDD a, FlexDD b) {
  return flex_dd_source_add(a,b);
}

static inline FlexDD flex_dd_neg(FlexDD value) {
  return FlexDD{-value.hi,-value.lo,-value.tail};
}

static inline FlexDD flex_dd_sub(FlexDD a, FlexDD b) {
  return flex_dd_add(a,flex_dd_neg(b));
}

static inline FlexDD flex_dd_mul(FlexDD a, FlexDD b) {
  return flex_dd_source_mul(a,b);
}

static inline FlexDD flex_dd_div(FlexDD a, FlexDD b) {
  return flex_dd_source_div(a,b);
}

static inline FlexDD flex_dd_sqrt(FlexDD value) {
  return flex_dd_source_sqrt(value);
}

static inline int flex_dd_compare(FlexDD a, FlexDD b) {
  return flex_dd_compare_exact(a,b);
}

static inline FlexDD3 flex_dd3(float3 value) {
  return FlexDD3{flex_dd(value.x),flex_dd(value.y),flex_dd(value.z)};
}

static inline FlexDD3 flex_dd3_add(FlexDD3 a, FlexDD3 b) {
  return FlexDD3{flex_dd_add(a.x,b.x),flex_dd_add(a.y,b.y),
                 flex_dd_add(a.z,b.z)};
}

static inline FlexDD3 flex_dd3_sub(FlexDD3 a, FlexDD3 b) {
  return FlexDD3{flex_dd_sub(a.x,b.x),flex_dd_sub(a.y,b.y),
                 flex_dd_sub(a.z,b.z)};
}

static inline FlexDD3 flex_dd3_scale(FlexDD3 a, FlexDD b) {
  return FlexDD3{flex_dd_mul(a.x,b),flex_dd_mul(a.y,b),flex_dd_mul(a.z,b)};
}

static inline FlexDD flex_dd_source_dot3(FlexDD3 a, FlexDD3 b);
static __attribute__((noinline)) FlexDD flex_dd_source_fma(
    FlexDD a, FlexDD b, FlexDD c);

static inline FlexDD flex_dd3_dot(FlexDD3 a, FlexDD3 b) {
  return flex_dd_source_dot3(a,b);
}

// Grow an exact float expansion by one term. The expansion is kept
// nonoverlapping, from least to most significant component, so a three-term
// dot product can be rounded to the 53-bit significand used by mjtNum.
static inline bool flex_float_is_subnormal(float value) {
  uint bits=as_type<uint>(value);
  return (bits&0x7f800000u)==0u && (bits&0x007fffffu)!=0u;
}

static __attribute__((noinline)) int flex_expansion_add(
    thread float* expansion, int count, float value,
    thread float* scratch) {
  // MPS arithmetic flushes subnormals to zero. Keep such exact input words
  // as low expansion components and only run TwoSum on normal operands.
  // They can still break a binary64 halfway comparison after the normal
  // components cancel, so dropping them here changes the source rounding.
  int out_count=0;
  if (flex_float_is_subnormal(value)) scratch[out_count++]=value;
  for (int i=0;i<count;i++) {
    if (flex_float_is_subnormal(expansion[i]))
      scratch[out_count++]=expansion[i];
  }
  float q=flex_float_is_subnormal(value) ? 0.0f : value;
  for (int i=0;i<count;i++) {
    float component=expansion[i];
    if (flex_float_is_subnormal(component)) continue;
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

static __attribute__((noinline)) int flex_expansion_sign(
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

static __attribute__((noinline)) int flex_expansion_compare_value(
    thread const float* expansion, int count, float value) {
  thread float work[32];
  thread float scratch[32];
  int work_count=0;
  for (int i=0;i<count;i++) {
    work_count=flex_expansion_add(work,work_count,expansion[i],scratch);
  }
  work_count=flex_expansion_add(work,work_count,-value,scratch);
  return flex_expansion_sign(work,work_count);
}

// Round an exact expansion to the nearest binary64 significand, ties to even.
// Three float components hold the resulting 53 bits without using unsupported
// Metal `double`. This is for source-order mju_dot3 support selection only.
static __attribute__((noinline)) FlexProjection53 flex_expansion_round53(
    thread const float* expansion, int count) {
  FlexProjection53 result=FlexProjection53{0.0f,0.0f,0.0f};
  if (count<=0) return result;
  result.hi=expansion[count-1];
  if (count==1) return result;
  int exponent=0;
  float significand=frexp(fabs(result.hi),exponent);
  int tail_sign=flex_expansion_sign(expansion,count-1);
  // At an exact power-of-two leading component, values just toward zero lie
  // in the lower binade, whose ulp is half as large. Choose that grid before
  // quantizing the complete tail (including negative projections).
  bool toward_zero=(result.hi>0.0f && tail_sign<0)
                   || (result.hi<0.0f && tail_sign>0);
  if (significand==0.5f && toward_zero) exponent-=1;
  int quantum_exp=exponent-53;
  thread float residual[32];
  thread float scratch[32];
  int residual_count=0;
  float estimate=0.0f;
  // Quantize every component below `hi`, including a widely separated
  // second component. This pass is only an estimate: a much smaller term can
  // still decide which side of an exact halfway value the expansion lies on.
  for (int i=0;i<count-1;i++) {
    float scaled=ldexp(expansion[i],-quantum_exp);
    residual_count=flex_expansion_add(residual,residual_count,
                                      expansion[i],scratch);
    estimate+=scaled;
  }
  // Split the integer number of 2^(e-53) quanta into a float32 high part and
  // a small exact correction. `rint` supplies a nearby integral estimate.
  float step_hi=rint(estimate);
  residual_count=flex_expansion_add(
      residual,residual_count,-ldexp(step_hi,quantum_exp),scratch);
  float residual_estimate=0.0f;
  for (int i=residual_count-1;i>=0;i--)
    residual_estimate+=ldexp(residual[i],-quantum_exp);
  float lower=floor(residual_estimate);
  // Correct the approximate integer using the unscaled exact remainder. This
  // avoids underflow when a tiny expansion word breaks a halfway tie.
  if (flex_expansion_compare_value(
          residual,residual_count,ldexp(lower,quantum_exp))<0)
    lower-=1.0f;
  if (flex_expansion_compare_value(
          residual,residual_count,ldexp(lower+1.0f,quantum_exp))>=0)
    lower+=1.0f;
  int half_cmp=flex_expansion_compare_value(
      residual,residual_count,ldexp(lower+0.5f,quantum_exp));
  bool tie_even=((int(step_hi)+int(lower))&1)==0;
  float step_lo=(half_cmp<0 ? lower
                  : (half_cmp>0 ? lower+1.0f
                     : (tie_even ? lower : lower+1.0f)));
  result.mid=ldexp(step_hi,quantum_exp);
  result.low=ldexp(step_lo,quantum_exp);
  return result;
}

static inline int flex_projection53_compare(FlexProjection53 a,
                                            FlexProjection53 b) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  count=flex_expansion_add(expansion,count,-b.low,scratch);
  count=flex_expansion_add(expansion,count,-b.mid,scratch);
  count=flex_expansion_add(expansion,count,-b.hi,scratch);
  count=flex_expansion_add(expansion,count,a.low,scratch);
  count=flex_expansion_add(expansion,count,a.mid,scratch);
  count=flex_expansion_add(expansion,count,a.hi,scratch);
  return flex_expansion_sign(expansion,count);
}

// FlexDD stores a rounded source scalar as three binary32 words.  Those words
// are an expansion, not a lexicographically ordered tuple: near a binary32
// binade boundary, a smaller `hi` can have a positive residual that makes the
// represented scalar larger.  Source mjtNum comparisons compare the scalar
// value, so compare the exact sums of the stored words.
static __attribute__((noinline)) int flex_dd_compare_exact(
    FlexDD a, FlexDD b) {
  return flex_projection53_compare(
      FlexProjection53{a.hi,a.lo,a.tail},
      FlexProjection53{b.hi,b.lo,b.tail});
}

static inline FlexProjection53 flex_mju_dot3_projection53(
    float3 point, FlexDD3 direction) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  FlexDD coordinate[3]={flex_dd(point.x),flex_dd(point.y),flex_dd(point.z)};
  FlexDD axis[3]={direction.x,direction.y,direction.z};
  FlexProjection53 product[3];
  for (int i=0;i<3;i++) {
    float high=coordinate[i].hi*axis[i].hi;
    float high_error=fma(coordinate[i].hi,axis[i].hi,-high);
    float cross1=coordinate[i].hi*axis[i].lo;
    float cross1_error=fma(coordinate[i].hi,axis[i].lo,-cross1);
    float cross2=coordinate[i].lo*axis[i].hi;
    float cross2_error=fma(coordinate[i].lo,axis[i].hi,-cross2);
    float cross3=coordinate[i].hi*axis[i].tail;
    float cross3_error=fma(coordinate[i].hi,axis[i].tail,-cross3);
    float cross4=coordinate[i].lo*axis[i].lo;
    float cross4_error=fma(coordinate[i].lo,axis[i].lo,-cross4);
    float cross5=coordinate[i].lo*axis[i].tail;
    float cross5_error=fma(coordinate[i].lo,axis[i].tail,-cross5);
    float cross6=coordinate[i].tail*axis[i].hi;
    float cross6_error=fma(coordinate[i].tail,axis[i].hi,-cross6);
    float cross7=coordinate[i].tail*axis[i].lo;
    float cross7_error=fma(coordinate[i].tail,axis[i].lo,-cross7);
    float low=coordinate[i].tail*axis[i].tail;
    float low_error=fma(coordinate[i].tail,axis[i].tail,-low);
    count=flex_expansion_add(expansion,count,high_error,scratch);
    count=flex_expansion_add(expansion,count,high,scratch);
    count=flex_expansion_add(expansion,count,cross1_error,scratch);
    count=flex_expansion_add(expansion,count,cross1,scratch);
    count=flex_expansion_add(expansion,count,cross2_error,scratch);
    count=flex_expansion_add(expansion,count,cross2,scratch);
    count=flex_expansion_add(expansion,count,cross3_error,scratch);
    count=flex_expansion_add(expansion,count,cross3,scratch);
    count=flex_expansion_add(expansion,count,cross4_error,scratch);
    count=flex_expansion_add(expansion,count,cross4,scratch);
    count=flex_expansion_add(expansion,count,cross5_error,scratch);
    count=flex_expansion_add(expansion,count,cross5,scratch);
    count=flex_expansion_add(expansion,count,cross6_error,scratch);
    count=flex_expansion_add(expansion,count,cross6,scratch);
    count=flex_expansion_add(expansion,count,cross7_error,scratch);
    count=flex_expansion_add(expansion,count,cross7,scratch);
    count=flex_expansion_add(expansion,count,low_error,scratch);
    count=flex_expansion_add(expansion,count,low,scratch);
    product[i]=flex_expansion_round53(expansion,count);
    count=0;
  }
  // The pinned arm64 binary is:
  //   p1 = round(point.y * direction.y)
  //   acc = round(point.x * direction.x + p1)  // fused multiply-add
  //   dot = round(point.z * direction.z + acc) // fused multiply-add
  // Keep the exact x/z products until their FMA boundaries; the middle
  // product alone is rounded before it enters the first FMA.
  count=0;
  count=flex_expansion_add(expansion,count,product[1].low,scratch);
  count=flex_expansion_add(expansion,count,product[1].mid,scratch);
  count=flex_expansion_add(expansion,count,product[1].hi,scratch);
  // Replace rounded x/z products by their exact expansions for the fused
  // source operations. Regenerate from float32 coordinate and three-word axis
  // to retain every product residual until the corresponding rounding.
  FlexDD scalar=coordinate[0];
  FlexDD d=axis[0];
  float p=scalar.hi*d.hi;
  count=flex_expansion_add(expansion,count,fma(scalar.hi,d.hi,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=scalar.hi*d.lo;
  count=flex_expansion_add(expansion,count,fma(scalar.hi,d.lo,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=scalar.hi*d.tail;
  count=flex_expansion_add(expansion,count,fma(scalar.hi,d.tail,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  FlexProjection53 first_fma=flex_expansion_round53(expansion,count);

  count=0;
  scalar=coordinate[2]; d=axis[2];
  p=scalar.hi*d.hi;
  count=flex_expansion_add(expansion,count,fma(scalar.hi,d.hi,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=scalar.hi*d.lo;
  count=flex_expansion_add(expansion,count,fma(scalar.hi,d.lo,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=scalar.hi*d.tail;
  count=flex_expansion_add(expansion,count,fma(scalar.hi,d.tail,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  count=flex_expansion_add(expansion,count,first_fma.low,scratch);
  count=flex_expansion_add(expansion,count,first_fma.mid,scratch);
  count=flex_expansion_add(expansion,count,first_fma.hi,scratch);
  return flex_expansion_round53(expansion,count);
}

// Round the exact sum of float expansions back to the pinned source's
// binary64 operation boundary, then retain the representable result in the
// native three-word value type. This preserves the source rounding point
// without changing strict comparisons or introducing a tolerance.
static inline FlexDD flex_dd_from_projection53(FlexProjection53 value) {
  return FlexDD{value.hi,value.mid,value.low};
}

static inline FlexProjection53 flex_dd_product53(FlexDD a, FlexDD b) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  float p=a.hi*b.hi;
  count=flex_expansion_add(expansion,count,fma(a.hi,b.hi,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.hi*b.lo;
  count=flex_expansion_add(expansion,count,fma(a.hi,b.lo,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.lo*b.hi;
  count=flex_expansion_add(expansion,count,fma(a.lo,b.hi,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.hi*b.tail;
  count=flex_expansion_add(expansion,count,fma(a.hi,b.tail,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.lo*b.lo;
  count=flex_expansion_add(expansion,count,fma(a.lo,b.lo,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.lo*b.tail;
  count=flex_expansion_add(expansion,count,fma(a.lo,b.tail,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.tail*b.hi;
  count=flex_expansion_add(expansion,count,fma(a.tail,b.hi,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.tail*b.lo;
  count=flex_expansion_add(expansion,count,fma(a.tail,b.lo,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  p=a.tail*b.tail;
  count=flex_expansion_add(expansion,count,fma(a.tail,b.tail,-p),scratch);
  count=flex_expansion_add(expansion,count,p,scratch);
  return flex_expansion_round53(expansion,count);
}

static inline FlexDD flex_dd_source_mulsub(FlexDD a, FlexDD b,
                                           FlexDD c, FlexDD d) {
  // The pinned arm64 `projectOriginPlane` cross uses `fnmul` for the second
  // product followed by `fmadd` for the first product and that rounded
  // negative. Preserve the single-round fused operation boundary.
  FlexDD rounded_negative=flex_dd_neg(flex_dd_source_mul(c,d));
  return flex_dd_source_fma(a,b,rounded_negative);
}

static inline FlexProjection53 flex_projection53_add(
    FlexProjection53 a, FlexProjection53 b) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  count=flex_expansion_add(expansion,count,a.low,scratch);
  count=flex_expansion_add(expansion,count,a.mid,scratch);
  count=flex_expansion_add(expansion,count,a.hi,scratch);
  count=flex_expansion_add(expansion,count,b.low,scratch);
  count=flex_expansion_add(expansion,count,b.mid,scratch);
  count=flex_expansion_add(expansion,count,b.hi,scratch);
  return flex_expansion_round53(expansion,count);
}

static inline FlexDD flex_dd_source_add(FlexDD a, FlexDD b) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  count=flex_expansion_add(expansion,count,a.lo,scratch);
  count=flex_expansion_add(expansion,count,a.tail,scratch);
  count=flex_expansion_add(expansion,count,a.hi,scratch);
  count=flex_expansion_add(expansion,count,b.lo,scratch);
  count=flex_expansion_add(expansion,count,b.tail,scratch);
  count=flex_expansion_add(expansion,count,b.hi,scratch);
  return flex_dd_from_projection53(flex_expansion_round53(expansion,count));
}

// Preserve all three binary32 words representing a rounded binary64 source
// value.  The expansion is rounded at the same operation boundary as the
// pinned mjtNum computation, then stored as three words so the next source
// operation does not collapse back to an effective 48-bit double-float.
static inline FlexDD flex_dd_source_mul(FlexDD a, FlexDD b) {
  return flex_dd_from_projection53(flex_dd_product53(a,b));
}

static inline FlexDD flex_dd_source_sub(FlexDD a, FlexDD b) {
  return flex_dd_source_add(a,flex_dd_neg(b));
}

static inline FlexDD flex_dd_source_muladd(FlexDD a, FlexDD b, FlexDD c) {
  return flex_dd_from_projection53(
      flex_projection53_add(flex_dd_product53(a,b),
          FlexProjection53{c.hi,c.lo,c.tail}));
}

// Form a quotient-refinement residual from the unrounded expansion
// `a - b*q0 - b*q1`.  Rounding either product to the source 53-bit boundary
// before subtraction can erase the residual digits that q1/q2 are meant to
// recover.  Every float product is split with FMA into its exact high/error
// pair, then all terms are accumulated before the residual is rounded once.
static __attribute__((noinline)) FlexDD flex_dd_division_residual(
    FlexDD a, FlexDD b, float q0, float q1) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  count=flex_expansion_add(expansion,count,a.tail,scratch);
  count=flex_expansion_add(expansion,count,a.lo,scratch);
  count=flex_expansion_add(expansion,count,a.hi,scratch);
  float denominator[3]={b.tail,b.lo,b.hi};
  float quotient[2]={q0,q1};
  for (int qi=0;qi<2;qi++) {
    for (int bi=0;bi<3;bi++) {
      float product=denominator[bi]*quotient[qi];
      float error=fma(denominator[bi],quotient[qi],-product);
      count=flex_expansion_add(expansion,count,-error,scratch);
      count=flex_expansion_add(expansion,count,-product,scratch);
    }
  }
  return flex_dd_from_projection53(flex_expansion_round53(expansion,count));
}

// Return a source-rounded residual without first rounding the product. This is
// used by sqrt refinement, where value - q*q can be much smaller than either
// operand and the discarded product bits otherwise change the final binary64
// result. Expand all nine products of the three-word operands and subtract
// their exact high/error pairs before one final source rounding.
static __attribute__((noinline)) FlexDD flex_dd_product_residual(
    FlexDD value, FlexDD a, FlexDD b) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  count=flex_expansion_add(expansion,count,value.tail,scratch);
  count=flex_expansion_add(expansion,count,value.lo,scratch);
  count=flex_expansion_add(expansion,count,value.hi,scratch);
  float av[3]={a.tail,a.lo,a.hi};
  float bv[3]={b.tail,b.lo,b.hi};
  for (int i=0;i<3;i++) {
    for (int j=0;j<3;j++) {
      float product=av[i]*bv[j];
      float error=fma(av[i],bv[j],-product);
      count=flex_expansion_add(expansion,count,-error,scratch);
      count=flex_expansion_add(expansion,count,-product,scratch);
    }
  }
  return flex_dd_from_projection53(flex_expansion_round53(expansion,count));
}

// Match the arm64 contraction used by pinned 3.10 support constructors:
// `fmadd(d, scale, base)` rounds the exact product-plus-add once to mjtNum.
// Keep this separate from flex_dd_source_muladd, whose two source rounding
// boundaries remain appropriate for non-fused multiply-then-add expressions.
static __attribute__((noinline)) FlexDD flex_dd_source_fma(FlexDD a, FlexDD b, FlexDD c) {
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  float product=a.hi*b.hi;
  count=flex_expansion_add(expansion,count,fma(a.hi,b.hi,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.hi*b.lo;
  count=flex_expansion_add(expansion,count,fma(a.hi,b.lo,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.lo*b.hi;
  count=flex_expansion_add(expansion,count,fma(a.lo,b.hi,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.hi*b.tail;
  count=flex_expansion_add(expansion,count,fma(a.hi,b.tail,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.lo*b.lo;
  count=flex_expansion_add(expansion,count,fma(a.lo,b.lo,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.lo*b.tail;
  count=flex_expansion_add(expansion,count,fma(a.lo,b.tail,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.tail*b.hi;
  count=flex_expansion_add(expansion,count,fma(a.tail,b.hi,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.tail*b.lo;
  count=flex_expansion_add(expansion,count,fma(a.tail,b.lo,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  product=a.tail*b.tail;
  count=flex_expansion_add(expansion,count,fma(a.tail,b.tail,-product),scratch);
  count=flex_expansion_add(expansion,count,product,scratch);
  count=flex_expansion_add(expansion,count,c.tail,scratch);
  count=flex_expansion_add(expansion,count,c.lo,scratch);
  count=flex_expansion_add(expansion,count,c.hi,scratch);
  return flex_dd_from_projection53(flex_expansion_round53(expansion,count));
}

static inline FlexDD flex_dd_source_dot3(FlexDD3 a, FlexDD3 b) {
  // Match the pinned arm64 contraction in mju_dot3: y*y is rounded first,
  // then x*x and z*z are fused into the running sum in that order.
  FlexDD y=flex_dd_source_mul(a.y,b.y);
  FlexDD xy=flex_dd_source_fma(a.x,b.x,y);
  return flex_dd_source_fma(a.z,b.z,xy);
}

static inline FlexDD3 flex_dd3_source_madd(FlexDD3 a, FlexDD scale,
                                           FlexDD3 b) {
  return FlexDD3{flex_dd_source_fma(a.x,scale,b.x),
                 flex_dd_source_fma(a.y,scale,b.y),
                 flex_dd_source_fma(a.z,scale,b.z)};
}

static inline FlexDD3 flex_dd3_source_sub(FlexDD3 a, FlexDD3 b) {
  return FlexDD3{flex_dd_source_sub(a.x,b.x),
                 flex_dd_source_sub(a.y,b.y),
                 flex_dd_source_sub(a.z,b.z)};
}

// Three quotient digits approximate the source binary64 result while keeping
// exact product residuals between refinements. The final expansion is rounded
// once to the source 53-bit boundary and retained in three float words.
static inline FlexDD flex_dd_source_div(FlexDD a, FlexDD b) {
  float q0=a.hi/b.hi;
  FlexDD remainder=flex_dd_division_residual(a,b,q0,0.0f);
  float q1=((remainder.hi+remainder.lo)+remainder.tail)/b.hi;
  remainder=flex_dd_division_residual(a,b,q0,q1);
  float q2=((remainder.hi+remainder.lo)+remainder.tail)/b.hi;
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  count=flex_expansion_add(expansion,count,q0,scratch);
  count=flex_expansion_add(expansion,count,q1,scratch);
  count=flex_expansion_add(expansion,count,q2,scratch);
  return flex_dd_from_projection53(flex_expansion_round53(expansion,count));
}

static inline FlexDD flex_dd_source_sqrt(FlexDD value) {
  // Pinned sqrt(0) is exactly zero. Avoid the zero-residual 0/(2*q0)
  // refinement, which otherwise creates NaNs in degenerate capsule witnesses.
  if (value.hi == 0.0f && value.lo == 0.0f && value.tail == 0.0f)
    return flex_dd(0.0f);
  float q0=sqrt(value.hi);
  FlexDD q0_value=flex_dd(q0);
  FlexDD remainder=flex_dd_product_residual(value,q0_value,q0_value);
  float q1=((remainder.hi+remainder.lo)+remainder.tail)/(2.0f*q0);
  FlexDD approximation=flex_dd_add(q0_value,flex_dd(q1));
  remainder=flex_dd_product_residual(value,approximation,approximation);
  float q2=((remainder.hi+remainder.lo)+remainder.tail)/(2.0f*q0);
  thread float expansion[32];
  thread float scratch[32];
  int count=0;
  count=flex_expansion_add(expansion,count,q0,scratch);
  count=flex_expansion_add(expansion,count,q1,scratch);
  count=flex_expansion_add(expansion,count,q2,scratch);
  return flex_dd_from_projection53(flex_expansion_round53(expansion,count));
}

static inline FlexDD3 flex_dd3_cross(FlexDD3 a, FlexDD3 b) {
  return FlexDD3{
      flex_dd_source_mulsub(a.y,b.z,a.z,b.y),
      flex_dd_source_mulsub(a.z,b.x,a.x,b.z),
      flex_dd_source_mulsub(a.x,b.y,a.y,b.x)};
}

static inline FlexDD3 flex_dd3_normalize(FlexDD3 value) {
  FlexDD length=flex_dd_sqrt(flex_dd3_dot(value,value));
  return FlexDD3{flex_dd_source_div(value.x,length),
                 flex_dd_source_div(value.y,length),
                 flex_dd_source_div(value.z,length)};
}

static inline float3 flex_dd3_high(FlexDD3 value) {
  return float3((value.x.hi+value.x.lo)+value.x.tail,
                (value.y.hi+value.y.lo)+value.y.tail,
                (value.z.hi+value.z.lo)+value.z.tail);
}

static inline FlexDD3 flex_dd3_neg(FlexDD3 value) {
  return FlexDD3{flex_dd_neg(value.x),flex_dd_neg(value.y),
                 flex_dd_neg(value.z)};
}

static inline FlexDD3 flex_dd3_div(FlexDD3 value, FlexDD divisor) {
  return FlexDD3{flex_dd_source_div(value.x,divisor),
                 flex_dd_source_div(value.y,divisor),
                 flex_dd_source_div(value.z,divisor)};
}

// Pinned engine_collision_primitive.c:mjraw_SphereSphere and
// mjraw_CapsuleCapsule on source-rounded geometry.  The capsule-axis vectors
// are built directly from compiled endpoints, avoiding the lossy quaternion
// round trip used by the earlier float-only detector route.

[[clang::noinline]] static inline FlexDD flex_dd_source_matvec_row(
    FlexDD m0, FlexDD m1, FlexDD m2,
    FlexDD x0, FlexDD x1, FlexDD x2) {
  FlexDD value=flex_dd_source_mul(m1,x1);
  value=flex_dd_source_fma(m0,x0,value);
  return flex_dd_source_fma(m2,x2,value);
}


[[clang::noinline]] static inline void flex_epa_p2_directions_source(
    FlexDD3 a, FlexDD3 b, thread FlexDD3* directions) {
  FlexDD3 diff=flex_dd3_source_sub(b,a);
  FlexDD least=flex_dd_abs(diff.x);
  int index=0;
  FlexDD ay=flex_dd_abs(diff.y), az=flex_dd_abs(diff.z);
  if (flex_dd_compare(ay,least)<0) { least=ay; index=1; }
  if (flex_dd_compare(az,least)<0) { least=az; index=2; }

  FlexDD axis_norm=flex_dd_sqrt(flex_dd3_dot(diff,diff));
  FlexDD3 axis=flex_dd3_div(diff,axis_norm);
  FlexDD3 basis=flex_dd3(float3(0.0f));
  if (index==0) basis.x=flex_dd(1.0f);
  else if (index==1) basis.y=flex_dd(1.0f);
  else basis.z=flex_dd(1.0f);
  FlexDD3 d1=flex_dd3_cross(basis,diff);

  // The decimal constant is the exact source literal rounded to mjtNum,
  // represented as a three-word scalar.  cos=-0.5 and (1-cos)=1.5 are exact.
  // These words sum to the nearest binary64 value of the source decimal
  // literal (not the unrounded decimal itself).
  FlexDD sine=FlexDD{0.8660253882408142f,1.553918593799608e-8f,
                     -1.1102230246251565e-16f};
  FlexDD one_minus_cos=flex_dd(1.5f);
  FlexDD u1=axis.x, u2=axis.y, u3=axis.z;
  FlexDD rotation[9];
  // The pinned source expressions are compiled with arm64 contraction at the
  // final multiply/add boundary. Preserve the first pairwise product as a
  // rounded source scalar, then contract its scale and the signed other term.
  // This follows the installed libmujoco polytope2 instruction sequence;
  // independently rounded products and sums can change the support direction.
  rotation[0]=flex_dd_source_fma(
      flex_dd_source_mul(u1,u1),one_minus_cos,flex_dd(-0.5f));
  rotation[1]=flex_dd_source_fma(
      flex_dd_source_mul(u1,u2),one_minus_cos,
      flex_dd_neg(flex_dd_source_mul(u3,sine)));
  rotation[2]=flex_dd_source_fma(
      flex_dd_source_mul(u1,u3),one_minus_cos,flex_dd_source_mul(u2,sine));
  rotation[3]=flex_dd_source_fma(
      flex_dd_source_mul(u2,u1),one_minus_cos,flex_dd_source_mul(u3,sine));
  rotation[4]=flex_dd_source_fma(
      flex_dd_source_mul(u2,u2),one_minus_cos,flex_dd(-0.5f));
  rotation[5]=flex_dd_source_fma(
      flex_dd_source_mul(u2,u3),one_minus_cos,
      flex_dd_neg(flex_dd_source_mul(u1,sine)));
  rotation[6]=flex_dd_source_fma(
      flex_dd_source_mul(u1,u3),one_minus_cos,
      flex_dd_neg(flex_dd_source_mul(u2,sine)));
  rotation[7]=flex_dd_source_fma(
      flex_dd_source_mul(u2,u3),one_minus_cos,flex_dd_source_mul(u1,sine));
  rotation[8]=flex_dd_source_fma(
      flex_dd_source_mul(u3,u3),one_minus_cos,flex_dd(-0.5f));

  FlexDD3 d2, d3;
  d2.x=flex_dd_source_matvec_row(rotation[0],rotation[1],rotation[2],
                                  d1.x,d1.y,d1.z);
  d2.y=flex_dd_source_matvec_row(rotation[3],rotation[4],rotation[5],
                                  d1.x,d1.y,d1.z);
  d2.z=flex_dd_source_matvec_row(rotation[6],rotation[7],rotation[8],
                                  d1.x,d1.y,d1.z);
  d3.x=flex_dd_source_matvec_row(rotation[0],rotation[1],rotation[2],
                                  d2.x,d2.y,d2.z);
  d3.y=flex_dd_source_matvec_row(rotation[3],rotation[4],rotation[5],
                                  d2.x,d2.y,d2.z);
  d3.z=flex_dd_source_matvec_row(rotation[6],rotation[7],rotation[8],
                                  d2.x,d2.y,d2.z);

  FlexDD3 raw[3]={d1,d2,d3};
  for (int i=0;i<3;i++) {
    FlexDD norm=flex_dd_sqrt(flex_dd3_dot(raw[i],raw[i]));
    // epaSupport uses its +x fallback when dnorm <= mjMINVAL.
    directions[i]=flex_dd_compare(norm,flex_dd(1.0e-15f))>0
        ? flex_dd3_div(raw[i],norm) : flex_dd3(float3(1.0f,0.0f,0.0f));
  }
}


static inline FlexDD3 flex_dd3_madd(FlexDD3 a, FlexDD scale, FlexDD3 b) {
  // Match source expressions `base + scale * vector` under the installed
  // 3.10 arm64 build, which contracts these pairs to FMA (including the
  // GJK lincomb and projected-origin helpers).
  return flex_dd3_source_madd(a,scale,b);
}


static inline FlexDD3 flex_dd3_lincomb_source(
    thread const FlexDD3* values, thread const FlexDD* weights, int count) {
  // Match the pinned arm64 `lincomb` lowering, not a left fold of the C
  // expression. For n>=2 the compiler rounds term 1, fuses term 0 into it,
  // then fuses each remaining term in increasing order. This is visible in
  // engine_collision_gjk.c:lincomb assembly and changes sub-ulp simplex
  // projections near the origin.
  if (count<=0) return flex_dd3(float3(0.0f));
  if (count==1) return flex_dd3_scale(values[0],weights[0]);
  FlexDD3 result=flex_dd3_scale(values[1],weights[1]);
  result=flex_dd3_madd(values[0],weights[0],result);
  for (int i=2;i<count;i++)
    result=flex_dd3_madd(values[i],weights[i],result);
  return result;
}


static inline FlexDD flex_dd3_component(FlexDD3 value, int index) {
  return index==0 ? value.x : (index==1 ? value.y : value.z);
}


static inline FlexDD flex_dd_abs(FlexDD value) {
  return value.hi<0.0f || (value.hi==0.0f
      && (value.lo<0.0f || (value.lo==0.0f && value.tail<0.0f)))
      ? flex_dd_neg(value) : value;
}


static inline int flex_dd_same_sign(FlexDD a, FlexDD b) {
  int ap=a.hi>0.0f || (a.hi==0.0f && (a.lo>0.0f || (a.lo==0.0f && a.tail>0.0f)));
  int an=a.hi<0.0f || (a.hi==0.0f && (a.lo<0.0f || (a.lo==0.0f && a.tail<0.0f)));
  int bp=b.hi>0.0f || (b.hi==0.0f && (b.lo>0.0f || (b.lo==0.0f && b.tail>0.0f)));
  int bn=b.hi<0.0f || (b.hi==0.0f && (b.lo<0.0f || (b.lo==0.0f && b.tail<0.0f)));
  return (ap && bp) || (an && bn);
}


static inline FlexDD flex_dd_det3(FlexDD3 a, FlexDD3 b, FlexDD3 c) {
  FlexDD3 cross_bc=flex_dd3_cross(b,c);
  return flex_dd3_dot(a,cross_bc);
}


static inline FlexDD3 flex_dd_project_origin_line(FlexDD3 a, FlexDD3 b) {
  FlexDD3 diff=flex_dd3_sub(b,a);
  FlexDD scale=flex_dd_neg(flex_dd_div(flex_dd3_dot(b,diff),
                                        flex_dd3_dot(diff,diff)));
  return flex_dd3_madd(diff,scale,b);
}


static inline int flex_dd_project_origin_plane(thread FlexDD3& result,
                                                FlexDD3 a, FlexDD3 b,
                                                FlexDD3 c) {
  FlexDD3 diff_ba=flex_dd3_sub(b,a), diff_ca=flex_dd3_sub(c,a);
  FlexDD3 diff_cb=flex_dd3_sub(c,b);
  FlexDD3 normal=flex_dd3_cross(diff_cb,diff_ba);
  FlexDD nv=flex_dd3_dot(normal,b), nn=flex_dd3_dot(normal,normal);
  if (flex_dd_compare(nn,flex_dd(0.0f))==0) return 1;
  if (flex_dd_compare(nv,flex_dd(0.0f))!=0
      && flex_dd_compare(nn,flex_dd(1.0e-15f))>0) {
    result=flex_dd3_scale(normal,flex_dd_div(nv,nn));
    return 0;
  }
  normal=flex_dd3_cross(diff_ba,diff_ca);
  nv=flex_dd3_dot(normal,a); nn=flex_dd3_dot(normal,normal);
  if (flex_dd_compare(nn,flex_dd(0.0f))==0) return 1;
  if (flex_dd_compare(nv,flex_dd(0.0f))!=0
      && flex_dd_compare(nn,flex_dd(1.0e-15f))>0) {
    result=flex_dd3_scale(normal,flex_dd_div(nv,nn));
    return 0;
  }
  normal=flex_dd3_cross(diff_ca,diff_cb);
  nv=flex_dd3_dot(normal,c); nn=flex_dd3_dot(normal,normal);
  if (flex_dd_compare(nn,flex_dd(0.0f))==0) return 1;
  result=flex_dd3_scale(normal,flex_dd_div(nv,nn));
  return 0;
}


static inline void flex_dd_s1d(thread FlexDD* lambda, FlexDD3 a, FlexDD3 b) {
  FlexDD3 projected=flex_dd_project_origin_line(a,b);
  FlexDD maximum=flex_dd_sub(flex_dd3_component(a,0),
                             flex_dd3_component(b,0));
  int index=0;
  for (int axis=1;axis<3;axis++) {
    FlexDD candidate=flex_dd_sub(flex_dd3_component(a,axis),
                                 flex_dd3_component(b,axis));
    if (flex_dd_compare(flex_dd_abs(candidate),flex_dd_abs(maximum))>=0) {
      maximum=candidate;
      index=axis;
    }
  }
  FlexDD c1=flex_dd_sub(flex_dd3_component(projected,index),
                        flex_dd3_component(b,index));
  FlexDD c2=flex_dd_sub(flex_dd3_component(a,index),
                        flex_dd3_component(projected,index));
  bool same=flex_dd_same_sign(maximum,c1) && flex_dd_same_sign(maximum,c2);
  lambda[0]=same ? flex_dd_div(c1,maximum) : flex_dd(0.0f);
  lambda[1]=same ? flex_dd_div(c2,maximum) : flex_dd(1.0f);
}


static inline FlexDD flex_dd_area2(FlexDD ax, FlexDD ay,
                                    FlexDD bx, FlexDD by,
                                    FlexDD cx, FlexDD cy) {
  // Match the pinned arm64 contraction of the lexical S2D area expression:
  // the first product is rounded, then each remaining product is fused into
  // the accumulator in source order. Separately rounding all six products
  // changes the final barycentric weight by one mjtNum ULP for the captured
  // near-symmetric simplex, which then perturbs GJK's next support point.
  FlexDD area=flex_dd_source_mul(ax,by);
  area=flex_dd_source_fma(ay,cx,area);
  area=flex_dd_source_fma(bx,cy,area);
  area=flex_dd_source_fma(flex_dd_neg(ax),cy,area);
  area=flex_dd_source_fma(flex_dd_neg(ay),bx,area);
  area=flex_dd_source_fma(flex_dd_neg(cx),by,area);
  return area;
}


static inline void flex_dd_s2d(thread FlexDD* lambda, FlexDD3 a,
                               FlexDD3 b, FlexDD3 c) {
  FlexDD3 origin_projection;
  if (flex_dd_project_origin_plane(origin_projection,a,b,c)) {
    flex_dd_s1d(lambda,a,b); lambda[2]=flex_dd(0.0f); return;
  }
  FlexDD m14=flex_dd_det3(a,b,c); // overwritten below with pinned minors
  m14=flex_dd_area2(b.y,b.z,c.y,c.z,a.y,a.z);
  FlexDD m24=flex_dd_area2(b.x,b.z,c.x,c.z,a.x,a.z);
  FlexDD m34=flex_dd_area2(b.x,b.y,c.x,c.y,a.x,a.y);
  FlexDD abs14=flex_dd_abs(m14), abs24=flex_dd_abs(m24), abs34=flex_dd_abs(m34);
  int xaxis,yaxis;
  FlexDD mmax;
  if (flex_dd_compare(abs14,abs24)>=0 && flex_dd_compare(abs14,abs34)>=0) {
    mmax=m14; xaxis=1; yaxis=2;
  } else if (flex_dd_compare(abs24,abs34)>=0) {
    mmax=m24; xaxis=0; yaxis=2;
  } else {
    mmax=m34; xaxis=0; yaxis=1;
  }
  FlexDD px=flex_dd3_component(origin_projection,xaxis);
  FlexDD py=flex_dd3_component(origin_projection,yaxis);
  FlexDD ax=flex_dd3_component(a,xaxis), ay=flex_dd3_component(a,yaxis);
  FlexDD bx=flex_dd3_component(b,xaxis), by=flex_dd3_component(b,yaxis);
  FlexDD cx=flex_dd3_component(c,xaxis), cy=flex_dd3_component(c,yaxis);
  FlexDD c31=flex_dd_area2(px,py,bx,by,cx,cy);
  FlexDD c32=flex_dd_area2(px,py,cx,cy,ax,ay);
  FlexDD c33=flex_dd_area2(px,py,ax,ay,bx,by);
  bool comp1=flex_dd_same_sign(mmax,c31), comp2=flex_dd_same_sign(mmax,c32);
  bool comp3=flex_dd_same_sign(mmax,c33);
  if (comp1 && comp2 && comp3) {
    lambda[0]=flex_dd_div(c31,mmax); lambda[1]=flex_dd_div(c32,mmax);
    lambda[2]=flex_dd_div(c33,mmax); return;
  }
  FlexDD dmin=flex_dd(3.402823466e+38f), edge[2];
  if (!comp1) {
    flex_dd_s1d(edge,b,c);
    FlexDD3 values[2]={b,c};
    FlexDD3 p=flex_dd3_lincomb_source(values,edge,2);
    lambda[0]=flex_dd(0.0f); lambda[1]=edge[0]; lambda[2]=edge[1];
    dmin=flex_dd3_dot(p,p);
  }
  if (!comp2) {
    flex_dd_s1d(edge,a,c);
    FlexDD3 values[2]={a,c};
    FlexDD3 p=flex_dd3_lincomb_source(values,edge,2);
    FlexDD d=flex_dd3_dot(p,p);
    if (flex_dd_compare(d,dmin)<0) {
      lambda[0]=edge[0]; lambda[1]=flex_dd(0.0f); lambda[2]=edge[1]; dmin=d;
    }
  }
  if (!comp3) {
    flex_dd_s1d(edge,a,b);
    FlexDD3 values[2]={a,b};
    FlexDD3 p=flex_dd3_lincomb_source(values,edge,2);
    FlexDD d=flex_dd3_dot(p,p);
    if (flex_dd_compare(d,dmin)<0) {
      lambda[0]=edge[0]; lambda[1]=edge[1]; lambda[2]=flex_dd(0.0f);
    }
  }
}


static inline void flex_dd_s3d(thread FlexDD* lambda, FlexDD3 a,
                               FlexDD3 b, FlexDD3 c, FlexDD3 d) {
  FlexDD c41=flex_dd_neg(flex_dd_det3(b,c,d));
  FlexDD c42=flex_dd_det3(a,c,d), c43=flex_dd_neg(flex_dd_det3(a,b,d));
  FlexDD c44=flex_dd_det3(a,b,c);
  FlexDD det=flex_dd_add(flex_dd_add(c41,c42),flex_dd_add(c43,c44));
  bool comp1=flex_dd_same_sign(det,c41), comp2=flex_dd_same_sign(det,c42);
  bool comp3=flex_dd_same_sign(det,c43), comp4=flex_dd_same_sign(det,c44);
  if (comp1 && comp2 && comp3 && comp4) {
    lambda[0]=flex_dd_div(c41,det); lambda[1]=flex_dd_div(c42,det);
    lambda[2]=flex_dd_div(c43,det); lambda[3]=flex_dd_div(c44,det); return;
  }
  FlexDD dmin=flex_dd(3.402823466e+38f), face[3];
  if (!comp1) {
    flex_dd_s2d(face,b,c,d); FlexDD3 values[3]={b,c,d};
    FlexDD3 p=flex_dd3_lincomb_source(values,face,3);
    lambda[0]=flex_dd(0.0f); lambda[1]=face[0]; lambda[2]=face[1]; lambda[3]=face[2];
    dmin=flex_dd3_dot(p,p);
  }
  if (!comp2) {
    flex_dd_s2d(face,a,c,d); FlexDD3 values[3]={a,c,d};
    FlexDD3 p=flex_dd3_lincomb_source(values,face,3);
    FlexDD dv=flex_dd3_dot(p,p);
    if (flex_dd_compare(dv,dmin)<0) {
      lambda[0]=face[0]; lambda[1]=flex_dd(0.0f); lambda[2]=face[1]; lambda[3]=face[2]; dmin=dv;
    }
  }
  if (!comp3) {
    flex_dd_s2d(face,a,b,d); FlexDD3 values[3]={a,b,d};
    FlexDD3 p=flex_dd3_lincomb_source(values,face,3);
    FlexDD dv=flex_dd3_dot(p,p);
    if (flex_dd_compare(dv,dmin)<0) {
      lambda[0]=face[0]; lambda[1]=face[1]; lambda[2]=flex_dd(0.0f); lambda[3]=face[2]; dmin=dv;
    }
  }
  if (!comp4) {
    flex_dd_s2d(face,a,b,c); FlexDD3 values[3]={a,b,c};
    FlexDD3 p=flex_dd3_lincomb_source(values,face,3);
    FlexDD dv=flex_dd3_dot(p,p);
    if (flex_dd_compare(dv,dmin)<0) {
      lambda[0]=face[0]; lambda[1]=face[1]; lambda[2]=face[2]; lambda[3]=flex_dd(0.0f);
    }
  }
}


static inline void flex_dd_subdistance(thread FlexDD* lambda, int n,
                                       thread const FlexDDVertex* simplex) {
  for (int i=0;i<4;i++) lambda[i]=flex_dd(0.0f);
  if (n==4) flex_dd_s3d(lambda,simplex[0].m,simplex[1].m,
                        simplex[2].m,simplex[3].m);
  else if (n==3) flex_dd_s2d(lambda,simplex[0].m,simplex[1].m,simplex[2].m);
  else if (n==2) flex_dd_s1d(lambda,simplex[0].m,simplex[1].m);
  else lambda[0]=flex_dd(1.0f);
}


struct FlexEpaFace {
  int a, b, c;
  int adj0, adj1, adj2;
  int index;
  // int (rather than bool) fixes the device-scratch ABI below.
  int active;
  // The sphere-flex path mirrors MuJoCo's mjtNum (double) EPA state in a
  // three-word lane; diagnostics convert only when writing their float ABI.
  FlexDD3 v_dd;
  FlexDD dist2_dd;
};
struct FlexHorizonFrame {
  int face;
  int edge;
  int phase;
  int boundary_face;
  int boundary_edge;
  int waiting;
};
static_assert(sizeof(FlexDD)==12, "FlexDD ABI");
static_assert(sizeof(FlexDDVertex)==108, "FlexDDVertex remains the d8d8 27-float ABI");
static_assert(sizeof(FlexEpaFace)==80, "FlexEpaFace ABI");
static_assert(sizeof(FlexHorizonFrame)==24, "FlexHorizonFrame ABI");

static inline FlexDD3 flex_dd3_average3(FlexDD3 a, FlexDD3 b, FlexDD3 c) {
  FlexDD third=FlexDD{0.3333333432674408f,-9.934107481068821e-9f,0.0f};
  return flex_dd3_scale(flex_dd3_add(flex_dd3_add(a,b),c),third);
}


static inline bool flex_project_origin_plane_dd(
    thread FlexDD3& result, FlexDD3 v1, FlexDD3 v2, FlexDD3 v3) {
  // Match projectOriginPlane's source boundaries: sub3 rounds each component,
  // cross3 uses the installed arm64 contraction, dot3 rounds each contracted
  // sum, division returns one mjtNum, and scl3 rounds each final product.
  FlexDD3 diff21=flex_dd3_source_sub(v2,v1);
  FlexDD3 diff31=flex_dd3_source_sub(v3,v1);
  FlexDD3 diff32=flex_dd3_source_sub(v3,v2);
  FlexDD3 n=flex_dd3_cross(diff32,diff21);
  FlexDD nv=flex_dd3_dot(n,v2), nn=flex_dd3_dot(n,n);
  if (flex_dd_compare(nn,flex_dd(0.0f))==0) return true;
  if (flex_dd_compare(nv,flex_dd(0.0f))!=0
      && flex_dd_compare(nn,flex_dd(1.0e-15f))>0) {
    FlexDD scale=flex_dd_source_div(nv,nn);
    result=FlexDD3{flex_dd_source_mul(n.x,scale),
                   flex_dd_source_mul(n.y,scale),
                   flex_dd_source_mul(n.z,scale)};
    return false;
  }
  n=flex_dd3_cross(diff21,diff31);
  nv=flex_dd3_dot(n,v1); nn=flex_dd3_dot(n,n);
  if (flex_dd_compare(nn,flex_dd(0.0f))==0) return true;
  if (flex_dd_compare(nv,flex_dd(0.0f))!=0
      && flex_dd_compare(nn,flex_dd(1.0e-15f))>0) {
    FlexDD scale=flex_dd_source_div(nv,nn);
    result=FlexDD3{flex_dd_source_mul(n.x,scale),
                   flex_dd_source_mul(n.y,scale),
                   flex_dd_source_mul(n.z,scale)};
    return false;
  }
  n=flex_dd3_cross(diff31,diff32);
  nv=flex_dd3_dot(n,v3); nn=flex_dd3_dot(n,n);
  FlexDD scale=flex_dd_source_div(nv,nn);
  result=FlexDD3{flex_dd_source_mul(n.x,scale),
                 flex_dd_source_mul(n.y,scale),
                 flex_dd_source_mul(n.z,scale)};
  return false;
}


[[clang::noinline]] static inline bool flex_epa_make_face_dd(
    device const FlexDDVertex* vertices, FlexDD3 center,
    int a, int b, int c, int3 adj, thread FlexEpaFace& face) {
  FlexDD3 projection;
  if (flex_project_origin_plane_dd(projection,vertices[c].m,
                                   vertices[b].m,vertices[a].m)) return false;
  FlexDD3 outward=flex_dd3_sub(vertices[a].m,center);
  if (flex_dd_compare(flex_dd3_dot(projection,outward),flex_dd(0.0f))<0)
    projection=flex_dd3_neg(projection);
  FlexDD dist2=flex_dd3_dot(projection,projection);
  face.a=a; face.b=b; face.c=c;
  face.adj0=adj.x; face.adj1=adj.y; face.adj2=adj.z;
  face.index=-1;
  face.v_dd=projection; face.dist2_dd=dist2;
  face.active=true;
  return true;
}


static inline void flex_epa_write_witness_dd(
    const thread FlexEpaFace& face, device const FlexDDVertex* vertices,
    thread ContactGeom& contact) {
  FlexDD3 s1=vertices[face.a].m, s2=vertices[face.b].m;
  FlexDD3 s3=vertices[face.c].m, p=face.v_dd;
  FlexDD m14=flex_dd_add(
      flex_dd_sub(flex_dd_mul(s2.y,s3.z),flex_dd_mul(s2.z,s3.y)),
      flex_dd_add(flex_dd_sub(flex_dd_mul(s1.z,s3.y),flex_dd_mul(s1.y,s3.z)),
                  flex_dd_sub(flex_dd_mul(s1.y,s2.z),flex_dd_mul(s1.z,s2.y))));
  FlexDD m24=flex_dd_add(
      flex_dd_sub(flex_dd_mul(s2.x,s3.z),flex_dd_mul(s2.z,s3.x)),
      flex_dd_add(flex_dd_sub(flex_dd_mul(s1.z,s3.x),flex_dd_mul(s1.x,s3.z)),
                  flex_dd_sub(flex_dd_mul(s1.x,s2.z),flex_dd_mul(s1.z,s2.x))));
  FlexDD m34=flex_dd_add(
      flex_dd_sub(flex_dd_mul(s2.x,s3.y),flex_dd_mul(s2.y,s3.x)),
      flex_dd_add(flex_dd_sub(flex_dd_mul(s1.y,s3.x),flex_dd_mul(s1.x,s3.y)),
                  flex_dd_sub(flex_dd_mul(s1.x,s2.y),flex_dd_mul(s1.y,s2.x))));
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
  FlexDD c31=flex_dd_add(
      flex_dd_sub(flex_dd_add(flex_dd_mul(q2.x,b2.y),
                              flex_dd_mul(q2.y,c2.x)),
                  flex_dd_mul(q2.x,c2.y)),
      flex_dd_sub(flex_dd_mul(b2.x,c2.y),flex_dd_mul(q2.y,b2.x)));
  c31=flex_dd_sub(c31,flex_dd_mul(c2.x,b2.y));
  FlexDD c32=flex_dd_add(
      flex_dd_sub(flex_dd_add(flex_dd_mul(q2.x,c2.y),
                              flex_dd_mul(q2.y,a2.x)),
                  flex_dd_mul(q2.x,a2.y)),
      flex_dd_sub(flex_dd_mul(c2.x,a2.y),flex_dd_mul(q2.y,c2.x)));
  c32=flex_dd_sub(c32,flex_dd_mul(a2.x,c2.y));
  FlexDD c33=flex_dd_add(
      flex_dd_sub(flex_dd_add(flex_dd_mul(q2.x,a2.y),
                              flex_dd_mul(q2.y,b2.x)),
                  flex_dd_mul(q2.x,b2.y)),
      flex_dd_sub(flex_dd_mul(a2.x,b2.y),flex_dd_mul(q2.y,a2.x)));
  c33=flex_dd_sub(c33,flex_dd_mul(b2.x,a2.y));
  FlexDD u=flex_dd_div(c31,mmax), v=flex_dd_div(c32,mmax);
  FlexDD w=flex_dd_div(c33,mmax);
  FlexDD3 xa=flex_dd3_add(flex_dd3_madd(vertices[face.a].a,u,
                                        flex_dd3(float3(0.0f))),
      flex_dd3_add(flex_dd3_scale(vertices[face.b].a,v),
                   flex_dd3_scale(vertices[face.c].a,w)));
  FlexDD3 xb=flex_dd3_add(flex_dd3_scale(vertices[face.a].b,u),
      flex_dd3_add(flex_dd3_scale(vertices[face.b].b,v),
                   flex_dd3_scale(vertices[face.c].b,w)));
  FlexDD3 witness_delta=flex_dd3_sub(xa,xb);
  FlexDD witness_norm=flex_dd_sqrt(flex_dd3_dot(witness_delta,witness_delta));
  contact.normal=flex_dd_compare(witness_norm,flex_dd(0.0f))>0
      ? flex_dd3_high(flex_dd3_div(witness_delta,witness_norm))
      : float3(0.0f);
  FlexDD distance=flex_dd_sqrt(face.dist2_dd);
  contact.dist=-((distance.hi+distance.lo)+distance.tail);
  contact.pos=flex_dd3_high(flex_dd3_scale(flex_dd3_add(xa,xb),flex_dd(0.5f)));
  contact.t1=float3(0.0f);
  make_frame(contact.normal,contact.t1,contact.t2);
}


static inline int flex_epa_face_vertex(thread const FlexEpaFace& face,
                                       int index) {
  return index==0 ? face.a : index==1 ? face.b : face.c;
}


static inline int flex_epa_get_edge(thread const FlexEpaFace& face,
                                    int vertex_id) {
  if (face.a==vertex_id) return 0;
  if (face.b==vertex_id) return 1;
  return 2;
}


static inline bool flex_epa_delete_face(device FlexEpaFace* faces,
                                        device int* map,
                                        thread int& map_count,
                                        int face_id, int face_count) {
  if (face_id<0 || face_id>=face_count || map_count<0
      || map_count>face_count) return false;
  int index=faces[face_id].index;
  if (index>=0) {
    if (index>=map_count) return false;
    int replacement=map[map_count-1];
    if (replacement<0 || replacement>=face_count) return false;
    map_count--;
    map[index]=replacement;
    faces[replacement].index=index;
  }
  faces[face_id].index=-2;
  faces[face_id].active=false;
  return true;
}


static inline int flex_epa_horizon_visit_dd(
    device FlexEpaFace* faces, device int* map, thread int& map_count,
    FlexDDVertex support,
    device int2* out_edges, device FlexHorizonFrame* stack,
    int stack_capacity, int face_count, thread int& out_count,
    int start_face, int start_edge, thread float* trace) {
  if (face_count<=0 || start_face<0 || start_face>=face_count
      || start_edge<0 || start_edge>=3 || stack_capacity<=0)
    return -2;
  int top=0;
  trace[69]=max(trace[69],1.0f);
  stack[0]={start_face,start_edge,0,-1,-1,false};
  bool has_result=false, result=false;
  ulong steps=0;
  while (top>=0) {
    // A valid walk marks every visited visible face deleted on entry. The
    // source polytope is finite; this larger bound protects corrupt cycles
    // without narrowing any valid 3.10 face topology.
    if (++steps>ulong(face_count)*4ul+ulong(stack_capacity)) return -2;
    if (top>=stack_capacity) return -1;
    if (stack[top].face<0 || stack[top].face>=face_count
        || stack[top].edge<0 || stack[top].edge>=3) return -2;
    if (has_result && stack[top].waiting) {
      if (!result) {
        if (out_count>=24) return -1;
        out_edges[out_count++]=int2(stack[top].boundary_face,
                                    stack[top].boundary_edge);
        int out_index=out_count-1;
        trace[85+2*out_index]=float(stack[top].boundary_face);
        trace[86+2*out_index]=float(stack[top].boundary_edge);
        trace[70]=float(out_count);
      }
      stack[top].waiting=false;
      has_result=false;
    }
    FlexHorizonFrame frame=stack[top];
    if (frame.phase==0) {
      FlexDD visibility=flex_dd_sub(
          flex_dd3_dot(faces[frame.face].v_dd,support.m),
          faces[frame.face].dist2_dd);
      // Pinned horizonRec uses `visibility > mjMINVAL`, not mjMINVAL2.
      // This is a signed distance test, so preserve the unsquared 1e-15
      // threshold even though the face's stored distance is squared.
      if (flex_dd_compare(visibility,flex_dd(1.0e-15f))<=0) {
        top--; result=false; has_result=true; continue;
      }
      if (!flex_epa_delete_face(faces,map,map_count,frame.face,face_count))
        return -2;
      stack[top].phase=1;
      continue;
    }
    if (frame.phase<=2) {
      int i=(frame.edge+frame.phase)%3;
      stack[top].phase=frame.phase+1;
      FlexEpaFace current_face=faces[frame.face];
      int next_face=i==0 ? current_face.adj0 :
                    i==1 ? current_face.adj1 : current_face.adj2;
      if (next_face<0 || next_face>=face_count) return -2;
      if (faces[next_face].index<=-2) continue;
      FlexEpaFace current_data=faces[frame.face];
      FlexEpaFace next_data=faces[next_face];
      int shared_vertex=flex_epa_face_vertex(current_data,(i+1)%3);
      int next_edge=flex_epa_get_edge(next_data,shared_vertex);
      stack[top].boundary_face=next_face;
      stack[top].boundary_edge=next_edge;
      stack[top].waiting=true;
      if (top+1>=stack_capacity) return -1;
      stack[++top]={next_face,next_edge,0,-1,-1,false};
      trace[69]=max(trace[69],float(top+1));
      continue;
    }
    top--; result=true; has_result=true;
  }
  return result ? 1 : 0;
}
