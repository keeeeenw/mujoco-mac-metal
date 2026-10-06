"""Native guard witness for the valid small-RHS Newton rescale boundary."""
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest


def _values(case="normal_high"):
  if case == "normal_high":
    # The scaled iterate is O(1); after a ~1e-36 RHS rescale its low-word
    # correction is subnormal while the represented solution is normal.
    hi=np.asarray([.5,-.375,.25,.5,-.75,.625],np.float32)
    lo=np.asarray([2.0**-26,-2.0**-25,2.0**-27,0.,-2.0**-26,2.0**-27],np.float32)
    scale=np.float32(2.0**-120)
  elif case == "subnormal_component":
    # Captured slope-primary Newton solution: a normal RHS scale multiplies
    # one O(2.4e-6) normalized component into the subnormal range.
    hi=np.asarray([2.436889e-6,0.1725926,.5,-.125,.75,-.5],np.float32)
    lo=np.asarray([2.0**-28,-7.301569e-9,-2.0**-27,2.0**-28,0.,2.0**-29],np.float32)
    scale=np.float32(1.128474576789396e-36)
  elif case == "legitimate_tiny_zero_component":
    # This is the captured 50-degree solver component whose exact physical
    # product is below half a min-subnormal. A second nonzero component proves
    # that the whole vector remains a valid tiny, nonzero RHS solution.
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    hi[0]=np.float32(-2.661650283080001e-18)
    lo[0]=np.float32(4.211651952074789e-26)
    hi[1]=np.float32(0.5)
    scale=np.float32(7.573064690121713e-29)
  elif case == "all_zero_rhs":
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    scale=np.float32(2.0**-120)
  elif case == "invalid_zero_scale":
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    scale=np.float32(0.0)
  elif case == "invalid_subnormal_scale":
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    hi[0]=np.float32(0.5)
    scale=np.nextafter(np.float32(0.0),np.float32(1.0))
  elif case == "invalid_infinite_scale":
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    hi[0]=np.float32(0.5)
    scale=np.float32(np.inf)
  elif case.startswith("half_"):
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    magnitude=np.float32(2.0**-24)
    if case.endswith("below_pos") or case.endswith("below_neg"):
      magnitude=np.nextafter(magnitude,np.float32(0.0),dtype=np.float32)
    elif case.endswith("above_pos") or case.endswith("above_neg"):
      magnitude=np.nextafter(magnitude,np.float32(np.inf),dtype=np.float32)
    if case.endswith("_neg"):
      magnitude=-magnitude
    hi[0]=magnitude
    scale=np.float32(2.0**-126)
  elif case == "split_half_plus_low_pos" or case == "split_half_plus_low_neg":
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    sign=np.float32(-1.0 if case.endswith("_neg") else 1.0)
    hi[0]=sign*np.float32(2.0**-30)
    lo[0]=sign*np.float32(2.0**-55)
    scale=np.float32(2.0**-120)
  elif case.startswith("negative_low_") or case.startswith("positive_low_"):
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    hi[0]=np.float32(2.0**-29)
    magnitude=np.float32(2.0**-30)
    if case.endswith("below"):
      magnitude=np.nextafter(magnitude,np.float32(0.0))
    elif case.endswith("above"):
      magnitude=np.nextafter(magnitude,np.float32(np.inf))
    lo[0]=magnitude if case.startswith("positive") else -magnitude
    scale=np.float32(2.0**-120)
  elif case.startswith("integer_half_"):
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    _,_,parity,side,offset=case.split("_")
    integer=2 if parity=="even" else 3
    magnitude=np.float32(2.0**-30)
    if offset=="below":
      magnitude=np.nextafter(magnitude,np.float32(0.0))
    elif offset=="above":
      magnitude=np.nextafter(magnitude,np.float32(np.inf))
    hi[0]=np.float32(integer*2.0**-29)
    lo[0]=magnitude if side=="pos" else -magnitude
    scale=np.float32(2.0**-120)
  elif case.startswith("minnormal_midpoint_") or case=="largest_subnormal_exact":
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    hi[0]=np.float32(1.0)
    half=np.float32(2.0**-24)
    if case.endswith("below"):
      half=np.nextafter(half,np.float32(np.inf))
    elif case.endswith("above"):
      half=np.nextafter(half,np.float32(0.0))
    lo[0]=-np.float32(2.0**-23) if case=="largest_subnormal_exact" else -half
    scale=np.float32(2.0**-126)
  elif case.startswith("normal_midpoint_"):
    hi=np.zeros(6,np.float32); lo=np.zeros(6,np.float32)
    hi[0]=np.float32(1.0)
    if case.endswith("_neg"):
      half=np.float32(2.0**-25)
      if case.endswith("below_neg"):
        half=np.nextafter(half,np.float32(0.0))
      elif case.endswith("above_neg"):
        half=np.nextafter(half,np.float32(np.inf))
      lo[0]=-half
    else:
      half=np.float32(2.0**-24)
      if case.endswith("below_pos"):
        half=np.nextafter(half,np.float32(0.0))
      elif case.endswith("above_pos"):
        half=np.nextafter(half,np.float32(np.inf))
      lo[0]=half
    scale=np.float32(1.0)
  else:
    raise ValueError(case)
  return hi,lo,scale


def _f32_fraction(value):
  return Fraction.from_float(float(np.float32(value)))


def _rne_integer(numerator,denominator):
  whole,remainder=divmod(numerator,denominator)
  twice=2*remainder
  if twice>denominator or (twice==denominator and whole&1):
    whole+=1
  return whole


def _exact_f32_bits(value):
  """Independent exact-rational binary32 RNE oracle, including subnormals."""
  value=Fraction(value)
  sign=0x80000000 if value<0 else 0
  value=abs(value)
  if value==0: return sign
  numerator,denominator=value.numerator,value.denominator
  exponent=numerator.bit_length()-denominator.bit_length()
  if ((exponent>=0 and numerator<(denominator<<exponent))
      or (exponent<0 and (numerator<<-exponent)<denominator)):
    exponent-=1
  if exponent>=-126:
    shift=23-exponent
    significand=(_rne_integer(numerator<<shift,denominator) if shift>=0
                 else _rne_integer(numerator,denominator<<-shift))
    if significand>=1<<24:
      significand>>=1; exponent+=1
    if exponent>127: return sign|0x7f800000
    return sign|((exponent+127)<<23)|(significand-(1<<23))
  subnormal=_rne_integer(numerator<<149,denominator)
  if subnormal==0: return sign
  if subnormal>=1<<23: return sign|0x00800000
  return sign|subnormal


def _exact_expected(hi,lo,scale):
  scales=np.broadcast_to(np.asarray(scale,dtype=np.float32),np.shape(hi))
  return np.asarray([
      _exact_f32_bits((_f32_fraction(a)+_f32_fraction(b))
                      *_f32_fraction(s))
      for a,b,s in zip(hi,lo,scales)],dtype=np.uint32).view(np.float32)


def _arbitrary_scale_half_vectors():
  """Build 64 exact pair products near half-minsubnormal at four scales."""
  scales=np.asarray([
      7.573064690121713e-29,  # captured original 50-degree scale
      1.128474576789396e-36,  # captured small-RHS witness scale
      3.141592653589793e-28,
      5.75e-30,
  ],dtype=np.float32)
  offsets=[-2.0**-2,-2.0**-16,-2.0**-32,-2.0**-48,
            2.0**-48,2.0**-32,2.0**-16,2.0**-2]
  hi=[]; lo=[]; scale_index=[]; signs=[]
  min_subnormal=Fraction(1,1<<149)
  for si,scale in enumerate(scales):
    scale_q=_f32_fraction(scale)
    for offset in offsets:
      target_units=Fraction(1,2)+Fraction.from_float(offset)
      target=target_units*min_subnormal/scale_q
      high=np.float32(float(target))
      residual=target-_f32_fraction(high)
      low=np.float32(float(residual))
      for sign in (1,-1):
        hi.append(np.float32(sign*high))
        lo.append(np.float32(sign*low))
        scale_index.append(si)
        signs.append(sign)
  hi=np.asarray(hi,np.float32); lo=np.asarray(lo,np.float32)
  packed=np.empty((len(hi),3),np.float32)
  packed[:,0]=hi; packed[:,1]=lo; packed[:,2]=scales[np.asarray(scale_index)]
  expected=_exact_expected(hi,lo,packed[:,2])
  return packed,expected,scales,np.asarray(scale_index),np.asarray(signs)


_ARBITRARY_SCALE_VECTOR_KERNEL=r"""
kernel void primal_tiny_rhs_arbitrary_scale_vector_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  for (uint i = 0; i < 64; ++i) {
    PrimalFloatPair value = {input[3 * i], input[3 * i + 1]};
    PrimalFloatPair scaled = primal_pair_multiply_scaled(value, input[3 * i + 2]);
    output[2 * i] = scaled.hi;
    output[2 * i + 1] = scaled.lo;
  }
}
"""


def test_cpu_arbitrary_mantissa_half_boundary_vectors_use_exact_fraction_oracle():
  packed,expected,scales,scale_index,signs=_arbitrary_scale_half_vectors()
  assert packed.shape==(64,3)
  assert len(np.unique(packed[:,2].view(np.uint32)))==4
  assert np.any(scales==np.float32(7.573064690121713e-29))
  expected_words=expected.view(np.uint32)
  for si in range(len(scales)):
    selected=scale_index==si
    # Each arbitrary mantissa has values on both sides of half a quantum;
    # the sign bit must survive correctly rounded underflow.
    assert set(expected_words[selected& (signs>0)]&0x7fffffff)=={0,1}
    assert set(expected_words[selected& (signs<0)]&0x7fffffff)=={0,1}
    assert np.all((expected_words[selected]&0x7fffffff)<=1)


def test_native_arbitrary_mantissa_half_boundary_vector_matches_exact_oracle():
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native MPS execution is opt-in via MUJOCO_METAL_RUN_GPU=1")
  torch=pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps,"compile_shader"):
    pytest.skip("requires native MPS shader execution")
  packed,expected,_,_,_=_arbitrary_scale_half_vectors()
  root=Path(__file__).parents[1]/"mujoco_metal"/"shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source=_coupled_shader_source(expected_root=root)+"\n"+_ARBITRARY_SCALE_VECTOR_KERNEL
  lib=torch.mps.compile_shader(source)
  output=torch.zeros((128,),dtype=torch.float32,device="mps")
  lib.primal_tiny_rhs_arbitrary_scale_vector_witness(
      torch.tensor(packed.reshape(-1),dtype=torch.float32,device="mps"),
      output,threads=(1,),group_size=(1,))
  actual=output.cpu().numpy().reshape((64,2))
  # These products occupy less than one subnormal quantum: there is no finer
  # representable low limb. Fraction(0) has no sign, so inspect the retained
  # high-word bits directly rather than erase signed zero by summing limbs.
  assert np.all((actual[:,1].copy().view(np.uint32)&0x7fffffff)==0)
  np.testing.assert_array_equal(actual[:,0].copy().view(np.uint32),
                                expected.view(np.uint32))


@pytest.mark.parametrize("case",["normal_high","subnormal_component"])
def test_cpu_small_rhs_pair_collapse_is_the_correct_float32_rounding(case):
  hi,lo,scale=_values(case)
  exact=(hi.astype(np.float64)+lo.astype(np.float64))*float(scale)
  expected=_exact_expected(hi,lo,scale)
  high_only=(hi*scale).astype(np.float32)
  assert np.any(np.abs(exact.astype(np.float64)-high_only.astype(np.float64))
                > 0.0)
  assert np.all(np.isfinite(expected))
  ulp=np.spacing(np.abs(expected).astype(np.float32)).astype(np.float64)
  assert np.all(np.abs(expected.astype(np.float64)-exact) <= .5*ulp)
  if case == "subnormal_component":
    assert np.any((np.abs(expected) > 0.0)
                  & (np.abs(expected) < np.finfo(np.float32).tiny))
    # Preserve a concrete normalized-pair negative control: directly
    # multiplying one source component by scale loses one output ULP.
    assert high_only[1].view(np.uint32) != expected[1].view(np.uint32)
  assert np.any((lo.astype(np.float64)*float(scale)) <
                np.finfo(np.float32).tiny)


@pytest.mark.parametrize("case,expected_word",[
    ("all_zero_rhs",0x00000000),
    ("legitimate_tiny_zero_component",0x80000000),
    ("half_below_pos",0x00000000),
    ("half_at_pos",0x00000000),
    ("half_above_pos",0x00000001),
    ("half_below_neg",0x80000000),
    ("half_at_neg",0x80000000),
    ("half_above_neg",0x80000001),
    ("split_half_plus_low_pos",0x00000001),
    ("split_half_plus_low_neg",0x80000001),
    ("negative_low_below",0x00000001),
    ("negative_low_at",0x00000000),
    ("negative_low_above",0x00000000),
    ("positive_low_below",0x00000001),
    ("positive_low_at",0x00000002),
    ("positive_low_above",0x00000002),
    ("integer_half_even_pos_below",0x00000002),
    ("integer_half_even_pos_at",0x00000002),
    ("integer_half_even_pos_above",0x00000003),
    ("integer_half_odd_pos_below",0x00000003),
    ("integer_half_odd_pos_at",0x00000004),
    ("integer_half_odd_pos_above",0x00000004),
    ("integer_half_even_neg_below",0x00000002),
    ("integer_half_even_neg_at",0x00000002),
    ("integer_half_even_neg_above",0x00000001),
    ("integer_half_odd_neg_below",0x00000003),
    ("integer_half_odd_neg_at",0x00000002),
    ("integer_half_odd_neg_above",0x00000002),
    ("minnormal_midpoint_below",0x007fffff),
    ("minnormal_midpoint_at",0x00800000),
    ("minnormal_midpoint_above",0x00800000),
    ("largest_subnormal_exact",0x007fffff),
    ("normal_midpoint_below_pos",0x3f800000),
    ("normal_midpoint_at_pos",0x3f800000),
    ("normal_midpoint_above_pos",0x3f800001),
    ("normal_midpoint_below_neg",0x3f800000),
    ("normal_midpoint_at_neg",0x3f800000),
    ("normal_midpoint_above_neg",0x3f7fffff),
])
def test_cpu_exact_rhs_pair_rne_boundaries(case,expected_word):
  hi,lo,scale=_values(case)
  expected=_exact_expected(hi,lo,scale)
  assert int(expected.view(np.uint32)[0])==expected_word
  if case=="legitimate_tiny_zero_component":
    assert expected.view(np.uint32)[1]&0x7fffffff
  if case=="all_zero_rhs":
    assert not np.any(expected.view(np.uint32)&0x7fffffff)


@pytest.mark.parametrize("case",[
    "normal_high","subnormal_component","legitimate_tiny_zero_component",
    "all_zero_rhs","half_below_pos","half_at_pos","half_above_pos",
    "half_below_neg","half_at_neg","half_above_neg",
    "split_half_plus_low_pos","split_half_plus_low_neg",
    "negative_low_below","negative_low_at","negative_low_above",
    "positive_low_below","positive_low_at","positive_low_above",
    "integer_half_even_pos_below","integer_half_even_pos_at",
    "integer_half_even_pos_above","integer_half_odd_pos_below",
    "integer_half_odd_pos_at","integer_half_odd_pos_above",
    "integer_half_even_neg_below","integer_half_even_neg_at",
    "integer_half_even_neg_above","integer_half_odd_neg_below",
    "integer_half_odd_neg_at","integer_half_odd_neg_above",
    "minnormal_midpoint_below","minnormal_midpoint_at",
    "minnormal_midpoint_above","largest_subnormal_exact",
    "normal_midpoint_below_pos","normal_midpoint_at_pos",
    "normal_midpoint_above_pos","normal_midpoint_below_neg",
    "normal_midpoint_at_neg","normal_midpoint_above_neg",
    "invalid_zero_scale","invalid_subnormal_scale",
    "invalid_infinite_scale"])
def test_native_tiny_rhs_rescale_keeps_a_finite_solution(case):
  if os.environ.get("MUJOCO_METAL_RUN_GPU") != "1":
    pytest.skip("native MPS execution is opt-in via MUJOCO_METAL_RUN_GPU=1")
  torch=pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps,"compile_shader"):
    pytest.skip("requires native MPS shader execution")
  hi,lo,scale=_values(case)
  inp=np.concatenate((hi,lo,np.asarray([scale],np.float32)))
  high_only=(hi*scale).astype(np.float32).astype(np.float64)
  dims=np.zeros(24,dtype=np.int32)
  dims[0]=0; dims[22]=0; dims[23]=0
  awake=np.ones(1,dtype=np.int32)
  root=Path(__file__).parents[1]/"mujoco_metal"/"shaders"
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  source = _coupled_shader_source(expected_root=root)
  lib=torch.mps.compile_shader(source)
  out=torch.zeros((13,),dtype=torch.float32,device="mps")
  lib.primal_tiny_rhs_rescale_witness(
      torch.tensor(inp,dtype=torch.float32,device="mps"),
      torch.tensor(dims,dtype=torch.int32,device="mps"),
      torch.tensor(awake,dtype=torch.int32,device="mps"),out,
      threads=(1,),group_size=(1,))
  actual=out.cpu().numpy().astype(np.float64)
  if case.startswith("invalid_"):
    print("PRIMAL_TINY_RHS_INVALID_SCALE", "case=",case,
          "status=",actual[0],flush=True)
    assert actual[0] == 0.0
    return
  expected=((hi.astype(np.float64)+lo.astype(np.float64))*float(scale))
  rounded=_exact_expected(hi,lo,scale).astype(np.float64)
  reconstructed=actual[1:7]+actual[7:13]
  print("PRIMAL_TINY_RHS_RESCALE", "status=",actual[0],
        "actual_hi=",actual[1:7].tolist(),
        "actual_lo=",actual[7:13].tolist(),
        "reconstructed=",reconstructed.tolist(),
        "source_float32=",rounded.tolist(),flush=True)
  assert actual[0] == 1.0
  assert np.array_equal(reconstructed.astype(np.float32),
                        rounded.astype(np.float32))
  low_ulp=np.spacing(np.abs(actual[7:13]).astype(np.float32)).astype(np.float64)
  low_ulp=np.maximum(low_ulp,
                     float(np.nextafter(np.float32(0.0),np.float32(1.0))))
  assert np.all(np.abs(reconstructed-expected) <= low_ulp)
  if case == "subnormal_component":
    # The second captured pair has a subnormal low product that ordinary
    # float multiplication flushes. The internal Newton pair must retain it.
    assert actual[8] != 0.0
    assert abs(reconstructed[1]-expected[1]) < abs(float(high_only[1])-expected[1])
  if case in {"legitimate_tiny_zero_component", "all_zero_rhs"}:
    assert np.array_equal(reconstructed.astype(np.float32), rounded.astype(np.float32))
    if case == "legitimate_tiny_zero_component":
      # Check the retained high word directly: summing -0 and the canonical
      # +0 low word on the host would erase the negative-zero sign.
      assert int(np.asarray(actual[1],np.float32).view(np.uint32)) == 0x80000000
      assert int(np.asarray(actual[7],np.float32).view(np.uint32)) & 0x7fffffff == 0
      assert reconstructed[1] != 0.0
    else:
      assert not np.any(reconstructed)
  if (case.startswith("half_") or case.startswith("split_half_plus_low_")
      or case.startswith("negative_low_")
      or case.startswith("positive_low_")
      or case.startswith("integer_half_")
      or case.startswith("minnormal_midpoint_")
      or case=="largest_subnormal_exact"
      or case.startswith("normal_midpoint_")):
    np.testing.assert_array_equal(reconstructed.astype(np.float32),rounded.astype(np.float32))


def test_subnormal_pair_rounding_keeps_the_low_limb_on_the_half_boundary():
  shader=(Path(__file__).parents[1]/"mujoco_metal"/"shaders"
          /"coupled_constraints.metal").read_text()
  rne=shader[shader.index("inline float primal_pair_scale_pow2_rne"):
             shader.index("inline float primal_float_scale_pow2_rne")]
  assert shader.index("inline bool primal_pair_negative(") < shader.index(
      "inline float primal_pair_scale_pow2_rne(")
  assert "float fraction_delta = (units_hi - integral) - 0.5f;" in rne
  assert "{fraction_delta, 0.0f}, {units_lo, 0.0f}" in rne
  assert "float remainder = (units_hi - integral) + units_lo;" not in rne
  scaled=shader[shader.index("inline PrimalFloatPair primal_pair_multiply_scaled"):
                shader.index("inline void primal_rescale_trace_failure")]
  assert "exponent_field == 0 || exponent_field == 0xff" in scaled
  assert "primal_pair_scale_pow2_rne(product, exponent)" in scaled
  assert "bool lost_high =" in scaled and "bool lost_low =" in scaled
  assert "if (lost_high || lost_low)" in scaled
  assert "return {combined, 0.0f};" in scaled
