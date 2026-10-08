"""Oracle and opt-in native witness for the production SdfWide helpers."""
from fractions import Fraction
import math
import os
from pathlib import Path
import struct
import warnings

import numpy as np
import pytest


_KERNEL = r"""
kernel void sdf_source_arithmetic_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= uint(count)) return;
  uint ib = index * 9;
  uint ob = index * 17;
  SdfWide a = {input[ib], input[ib + 1], input[ib + 2]};
  SdfWide b = {input[ib + 3], input[ib + 4], input[ib + 5]};
  SdfWide c = {input[ib + 6], input[ib + 7], input[ib + 8]};
  SdfWide values[5] = {
      sw_add(a, b), sw_mul(a, b), sw_div(a, b), sw_sqrt(sw_abs(a)),
      sw_source_fma(a, b, c)};
  for (uint operation = 0; operation < 5; ++operation) {
    output[ob + 3 * operation] = values[operation].hi;
    output[ob + 3 * operation + 1] = values[operation].mid;
    output[ob + 3 * operation + 2] = values[operation].lo;
  }
  output[ob + 15] = sw_float32(a);
  output[ob + 16] = sw_float32(sw_source_fma(a, b, c));
}
"""


def _f32(value):
  return struct.unpack("=f", struct.pack("=f", float(value)))[0]


def _words(value):
  high = _f32(value)
  residual = value - high
  middle = _f32(residual)
  low = _f32(residual - middle)
  assert sum((Fraction.from_float(word) for word in
              (high, middle, low)), Fraction()) == Fraction.from_float(value)
  return high, middle, low


def _exact(words):
  return sum((Fraction.from_float(float(word)) for word in words), Fraction())


def _cases():
  tiny = _f32(float.fromhex("0x1p-149"))
  return [
      # Halfway and just-over-halfway add boundaries.
      (1.0, 2.0 ** -53, 0.0),
      (1.0, math.nextafter(2.0 ** -53, math.inf), 0.0),
      # Exact cancellation with a retained low term in the fused operation.
      (1.0 + 2.0 ** -27, 1.0 - 2.0 ** -27, -1.0),
      # Nontrivial product/division/root and float32 halfway publication.
      (2.0, 0.1, -0.2),
      (1.0 + 2.0 ** -24, 1.0, 0.0),
      (1.0 + 3.0 * 2.0 ** -24, 1.0, 0.0),
      # Halfway with an odd leading float32 significand must round upward.
      (1.0 + 2.0 ** -23 + 2.0 ** -24, 1.0, 0.0),
      # The least positive binary32 subnormal is carried through each helper.
      (tiny, 1.0, tiny),
      # Negative cancellation and a large exponent spread.
      (-1.0, math.nextafter(2.0 ** -53, math.inf), 1.0),
      (2.0 ** 40, 2.0 ** -40, -1.0),
      # Normal operands whose exact product is the least subnormal.
      (2.0 ** -100, 2.0 ** -49, 0.0),
      # The finite addend must not overflow merely to scale the tiny product.
      (tiny, 1.0, 2.0 ** 110),
      # Opposite signs produce exact zero; all intermediates remain finite.
      (tiny, -1.0, tiny),
      (0.0, 1.0, -0.0),
      # The largest subnormal lies one unit below the minimum normal.
      (2.0 ** -126, 1.0, -tiny),
      (2.0 ** -126, 1.0, tiny),
      # Normal leading words with subnormal mid/low tails in the input ABI.
      ((1.0, tiny, tiny), 1.0, -1.0),
      # Normal inputs can also produce a subnormal quotient.
      (2.0 ** -100, 2.0 ** 49, 0.0),
      # Non-power neighbors ensure sqrt scaling is based on the exponent, not
      # on exact powers or one hand-picked subnormal bit pattern.
      (3.0 * tiny, 1.0, 0.0),
      (2.0 ** -126 + tiny, 1.0, 0.0),
      # Production razor operand triple from the mesh-SDF Frank-Wolfe update
      # (face5/sp8 step1): fused single rounding must hold on GPU words.
      ((0.6666666865348816, -1.9868215517249155e-08, 5.551115123125783e-16),
       (0.03999999910593033, 0.0, 0.0),
       (-0.019999999552965164, 3.469446951953614e-18, 0.0)),
      # Fused-vs-separate discriminators: naive separate multiply-add rounds
      # these differently, so they fail closed on an unfused implementation.
      (0.2750700585440605, -0.22040888733484526, -0.14404877411885747),
      (916312.6616423248, 753211.6602122856, 1171857.4824565682),
      (2368.6215373227114, -13574.021118237719, 11486.609269564708),
      (-6569828.860219568, 10999784.937490845, -6120873.1286308),
  ]


def _case_words(value):
  if isinstance(value, tuple):
    assert len(value) == 3
    return tuple(_f32(word) for word in value)
  return _words(value)


def _input_array():
  rows = []
  for av, bv, cv in _cases():
    rows.extend(_case_words(av) + _case_words(bv) + _case_words(cv))
  return np.asarray(rows, dtype=np.float32).reshape(-1, 9)


def _expected(case):
  av, bv, cv = case
  a, b, c = map(_case_words, case)
  A, B, C = map(_exact, (a, b, c))
  # Python converts an exact Fraction to IEEE binary64 nearest-even.
  wide = (float(A + B), float(A * B), float(A / B),
          math.sqrt(abs(float(A))), float(A * B + C))
  return tuple(_words(value) for value in wide), _f32(float(A)), _f32(float(A * B + C))


def test_cpu_oracle_cases_cover_rounding_and_admission_boundaries():
  cases = _cases()
  assert len(cases) >= 8
  for case in cases:
    wide, float_a, float_fma = _expected(case)
    assert all(len(words) == 3 for words in wide)
    assert math.isfinite(float_a) and math.isfinite(float_fma)
  # Explicit nearest-even float32 ties are part of the witness payload.
  assert _expected(cases[4])[1] == 1.0
  assert _expected(cases[5])[1] == _f32(1.0 + 2.0 ** -22)
  assert _expected(cases[6])[1] == _f32(1.0 + 2.0 * 2.0 ** -23)
  assert _exact(_expected(cases[0])[0][0]) == Fraction(1)
  assert _exact(_expected(cases[1])[0][0]) == Fraction.from_float(
      1.0 + 2.0 ** -52)
  assert _exact(_expected(cases[0])[0][2]) == Fraction(2 ** 53)
  assert _exact(_expected(cases[2])[0][1]) == Fraction(1)
  assert _exact(_expected(cases[2])[0][4]) == -Fraction(1, 2 ** 54)
  # The division quotient has a nonzero term below q2 that resolves an exact
  # halfway result; the residual must survive through q3 before 53-bit rounding.
  assert _exact(_expected(cases[2])[0][2]) == (
      Fraction(1) + Fraction(1, 2 ** 26) + Fraction(1, 2 ** 52))
  assert _exact(_expected(cases[7])[0][1]) == Fraction.from_float(
      _f32(float.fromhex("0x1p-149")))
  assert _exact(_expected(cases[7])[0][2]) == Fraction.from_float(
      _f32(float.fromhex("0x1p-149")))
  assert _exact(_expected(cases[7])[0][4]) == Fraction.from_float(
      _f32(2.0 * float.fromhex("0x1p-149")))
  tiny_root = _expected(cases[7])[0][3]
  assert tiny_root[1] != 0.0 and tiny_root[2] != 0.0
  # A normal-normal product can still underflow to a representable subnormal.
  tiny = Fraction.from_float(_f32(float.fromhex("0x1p-149")))
  assert _exact(_expected(cases[10])[0][1]) == tiny
  # Scaling the subnormal multiplicand and a large finite addend together
  # would overflow; exact fused rounding leaves this addend unchanged.
  assert _exact(_expected(cases[11])[0][4]) == Fraction(2 ** 110)
  assert _exact(_expected(cases[12])[0][4]) == 0
  assert _exact(_expected(cases[13])[0][4]) == 0
  largest_subnormal = Fraction.from_float(_f32(float.fromhex("0x1p-126"))) - tiny
  assert _exact(_expected(cases[14])[0][4]) == largest_subnormal
  assert _exact(_expected(cases[15])[0][4]) == (
      Fraction.from_float(_f32(float.fromhex("0x1p-126"))) + tiny)
  assert _exact(_expected(cases[16])[0][4]) == 2 * tiny
  assert _exact(_expected(cases[17])[0][2]) == tiny
  assert _expected(cases[18])[0][3][1] != 0.0
  assert _expected(cases[19])[0][3][1] != 0.0
  # Production razor triple (face5/sp8 step1 update): fused single rounding
  # lands on 0x3f7b4e81aaaaaaad, matching the installed oracle; the naive
  # separate multiply-add double-rounds to ...aaac.
  import struct as _struct

  def _fl64(value):
    return _struct.unpack("<d", _struct.pack("<d", float(value)))[0]

  razor = cases[-5]
  assert _fl64(_exact(_expected(razor)[0][4])) == _fl64(
      Fraction("0.006666666517655057"))
  razor_bits = hex(_struct.unpack(
      "<Q", _struct.pack("<d", _fl64(_exact(_expected(razor)[0][4]))))[0])
  assert razor_bits == "0x3f7b4e81aaaaaaad"
  for case in cases[-5:]:
    av, bv, cv = case
    a, b, c = map(_case_words, (av, bv, cv))
    A, B, C = map(_exact, (a, b, c))
    fused = float(A * B + C)
    separate = float(float(A * B) + float(C))
    assert fused != separate, case
    assert _fl64(_exact(_expected(case)[0][4])) == fused, case

  source = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/plugin_sdf.metal").read_text()
  assert "inline SdfWide sw_add(SdfWide a, SdfWide b)" in source
  assert "inline SdfWide sw_mul(SdfWide a, SdfWide b)" in source
  assert "inline SdfWide sw_div(SdfWide a, SdfWide b)" in source
  assert "inline SdfWide sw_sqrt(SdfWide a)" in source
  assert "inline SdfWide sw_source_fma(" in source
  assert "inline float sw_float32(SdfWide value)" in source
  assert "inline float sw_project_float32(SdfWide value)" in source
  assert "sw_float32_from_subnormal_units" in source
  assert "leading_quanta+int(step_hi)+int(lower)" in source
  assert "return hi+rounded_tail;" in source
  assert "sw_scale_word_pow2" in source and "sw_scale_wide_pow2" in source
  assert "sw_source_mul_core" in source and "sw_source_fma_core" in source
  assert "sw_division_residual(a,b,q0,q1,q2)" in source
  assert "count=sw_expansion_add(expansion,count,q3,scratch);" in source
  assert "int scale=-48-exponent;" in source
  assert "if (exponent < -78)" in source
  sqrt_source = source[source.index("inline SdfWide sw_sqrt(SdfWide a) {",
                                   source.index("sw_sqrt_core")):]
  assert sqrt_source.index("sw_wide_has_subnormal_leading(a)") < sqrt_source.index("a.hi==0.0f")
  assert "if (!has_subnormal_leading &&" in sqrt_source
  assert "sw_sqrt_core(sw_scale_wide_pow2(a,scale)),-scale/2" in source
  assert "sw_float_power_upper" in source
  assert "scaling_c_overflows && product_below_half_ulp" in source
  assert "sw_expansion_add_product" in source
  assert "if (quotient_exp< -126) scale_a+=(-126-quotient_exp);" in source
  assert "boundary_exact && abs(boundary_units)<0x01000000" in source
  assert "cannot encode a residual" in source
  assert "ldexp(" not in source
  assert "isfinite(attrs)" in source or "isfinite" in source


def test_existing_plugin_input_rejection_contract_is_unchanged():
  from mujoco_metal.bundled_plugins import BundledPluginError, _sdf_attributes
  from mujoco_metal.plugin_sdf import _check_query
  with pytest.raises(BundledPluginError, match="finite"):
    _sdf_attributes("mujoco.sdf.bowl", {"height": "nan",
        "radius": "0.9", "thickness": "0.05"})
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    with pytest.raises(BundledPluginError, match="float32 device range"):
      _sdf_attributes("mujoco.sdf.bowl", {"height": "1e100",
          "radius": "0.9", "thickness": "0.05"})
  with pytest.raises(ValueError, match="finite"):
    _check_query(2, [0.0, float("inf"), 0.0], [0.32, 0.9, 0.05, 0, 0])


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native helper witness is parent-dispatched")
def test_native_witness_calls_production_sdf_arithmetic_helpers():
  import torch
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("MPS compile_shader is unavailable")
  shader = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/plugin_sdf.metal").read_text()
  library = torch.mps.compile_shader(shader + "\n" + _KERNEL)
  kernel = library.sdf_source_arithmetic_witness
  host = _input_array()
  inputs = torch.as_tensor(host.copy(), dtype=torch.float32, device="mps").contiguous()
  output = torch.empty((len(host), 17), dtype=torch.float32, device="mps")
  count = torch.tensor([len(host)], dtype=torch.int32, device="mps")
  kernel(inputs.reshape(-1), output.reshape(-1), count,
         threads=(len(host),), group_size=(1,))
  actual = output.cpu().numpy()
  failures = []
  for i, case in enumerate(_cases()):
    wide, float_a, float_fma = _expected(case)
    for op, expected in enumerate(wide):
      actual_words = tuple(float(v) for v in actual[i, 3 * op:3 * op + 3])
      if not all(math.isfinite(v) for v in actual_words):
        failures.append(f"case={i} op={op} nonfinite={actual_words} expected={expected}")
        continue
      if _exact(actual_words) != _exact(expected):
        failures.append(
            f"case={i} op={op} got={actual_words} ({_exact(actual_words)}) "
            f"expected={expected} ({_exact(expected)})")
    if actual[i, 15] != float_a:
      failures.append(
          f"case={i} op=float32(a) got={actual[i, 15]} expected={float_a}")
    if actual[i, 16] != float_fma:
      failures.append(
          f"case={i} op=float32(fma) got={actual[i, 16]} expected={float_fma}")
  assert not failures, "native SDF arithmetic mismatches:\n" + "\n".join(failures)


_PROJECTION_KERNEL = r"""
kernel void sdf_float32_projection_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= uint(count)) return;
  SdfWide triple = {input[3 * index], input[3 * index + 1],
                    input[3 * index + 2]};
  output[2 * index] = sw_project_float32(triple);
  output[2 * index + 1] = (input[3 * index] + input[3 * index + 1]) +
      input[3 * index + 2];
}
"""


def _projection_split(value):
  # Mirror the host octree uploader: hi is the direct binary64 rounding,
  # lo/tail are float32 residuals. The split must be exact as reals.
  high = _f32(value)
  middle = _f32(value - high)
  low = _f32(value - high - middle)
  assert sum((Fraction.from_float(word) for word in
              (high, middle, low)), Fraction()) == Fraction.from_float(value)
  return high, middle, low


def _projection_cases():
  # (source double, expect naive sequential float32 sum to differ)
  base = float.fromhex("0x1.000002fffffffp+0")
  below = float.fromhex("0x1.000002fffffffp+0") - 2.0 ** -53
  above = float.fromhex("0x1.0000038000000p+0")
  exact = float.fromhex("0x1.000004p+0")
  return [
      # Double-rounding boundary: direct rounds to 0x3f800001, the naive
      # float32 re-sum ties halfway (even) and lands on 0x3f800002.
      (base, True),
      # Sign control: the same hazard mirrored below zero.
      (-base, True),
      # Just below the boundary: direct still rounds to 0x3f800001 while
      # the naive re-sum stays on 0x3f800002.
      (below, True),
      # Clearly above: both agree on 0x3f800002.
      (above, False),
      # Exact zero and exactly representable values agree trivially.
      (0.0, False),
      (exact, False),
  ]


def _f32_bits(value):
  return struct.unpack("=I", struct.pack("=f", float(value)))[0]


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native helper witness is parent-dispatched")
def test_native_float32_projection_boundary_gpu():
  """Correctly rounded float32 projection must not double-round residuals.

  The scalar octree readers project retained (hi, lo, tail) triples with
  sw_project_float32 (the high word, i.e. the direct binary64 rounding).
  Re-adding residuals in float32 double-rounds on real upload data: for
  0x1.000002fffffffp+0, direct rounding gives 0x3f800001 while sequential
  float32 (hi+lo)+tail gives 0x3f800002. The naive-sum assertions below
  prove this test is genuine: they fail if the projection is implemented
  as a float32 re-sum. Tails are nonzero on the boundary cases, so the
  hazard is really exercised, not padded with zeros.
  """
  import torch
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("MPS compile_shader is unavailable")
  cases = _projection_cases()
  words = []
  for value, _ in cases:
    words.extend(_projection_split(value))
  assert any(word != 0.0 for word in words[2::3]), "tails must be nonzero"
  shader = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/plugin_sdf.metal").read_text()
  assert "inline float sw_project_float32(SdfWide value)" in shader
  library = torch.mps.compile_shader(shader + "\n" + _PROJECTION_KERNEL)
  kernel = library.sdf_float32_projection_witness
  host = np.asarray(words, dtype=np.float32)
  inputs = torch.as_tensor(host.copy(), dtype=torch.float32, device="mps").contiguous()
  output = torch.empty((len(cases), 2), dtype=torch.float32, device="mps")
  count = torch.tensor([len(cases)], dtype=torch.int32, device="mps")
  kernel(inputs.reshape(-1), output.reshape(-1), count,
         threads=(len(cases),), group_size=(1,))
  torch.mps.synchronize()
  actual = output.cpu().numpy()
  failures = []
  for i, (value, expect_naive_differs) in enumerate(cases):
    got_helper = _f32_bits(actual[i, 0])
    want_helper = _f32_bits(np.float32(value))
    if got_helper != want_helper:
      failures.append(
          f"case={i} projection got={got_helper:#x} expected={want_helper:#x}")
    hi, lo, tail = (float(words[3 * i]), float(words[3 * i + 1]),
                    float(words[3 * i + 2]))
    want_naive = _f32_bits(np.float32(np.float32(hi + lo) + np.float32(tail)))
    got_naive = _f32_bits(actual[i, 1])
    if got_naive != want_naive:
      failures.append(
          f"case={i} naive got={got_naive:#x} expected={want_naive:#x}")
    if expect_naive_differs and got_naive == got_helper:
      failures.append(
          f"case={i} hazard not exercised: naive matches projection")
    if not expect_naive_differs and got_naive != got_helper:
      failures.append(
          f"case={i} control diverged: naive={got_naive:#x} "
          f"projection={got_helper:#x}")
  assert not failures, "native projection mismatches:\n" + "\n".join(failures)
