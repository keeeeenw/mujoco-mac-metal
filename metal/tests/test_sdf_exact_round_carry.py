# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Limb-carry regressions for exact expansion rounding + 64-case device fixture.

The first exact rounder accumulated limb magnitudes in `uint`, so duplicate
bit contributions at a high limb bit wrapped to zero before normalization
(e.g. two words sharing limb-bit 31 lose a full 2**32). The repair adds
every set bit value-exactly with ripple-carry propagation. This module
commits:

- literal helper-level word rows (the independently reproduced table plus
  multi-carry, cross-limb, borrow-chain, full-count and sign controls)
  executed through BOTH the actual helper and its `round53` wrapper, with
  `ok == True` and exact triple sums required (no silent fallback);
- real add/mul/FMA producer cases with duplicate-heavy and subnormal-word
  expansions establishing reachable coverage separately;
- all 64 admitted-domain differential failures as a deterministic literal
  fixture with exact-oracle expectations, executed on device (not only in
  a CPU replica sharing the algorithm).

Every device case runs on BOTH the helper-only unit (`plugin_sdf.metal`)
and the COMPLETE production unit (`_coupled_shader_source()`), with
source ownership verified (the complete unit carries contact kernels the
helper unit lacks; both carry the exact rounder). Scope note, stated
honestly: the table rows are helper-bounds stress cases. A 55k-shape
brute force over subnormal/max-mantissa-heavy add/mul/FMA producer
triples (old vs new shader, exact-oracle comparison) found zero
producer-reachable wraps — TwoSum growth merges normal duplicates and
surviving subnormal duplicates stay far below wrap magnitude — so the
helper proof (valid for every in-bounds input regardless of reachability)
is the load-bearing argument, and the producer cases below pin exactness
on reachable shapes.
"""
import os
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

_F32 = np.float32


def _f32(value):
  return struct.unpack("=f", struct.pack("=f", float(value)))[0]


def _split3(value):
  high = _f32(value)
  residual = value - high
  middle = _f32(residual)
  low = _f32(residual - middle)
  assert sum((Fraction(word) for word in (high, middle, low)),
             Fraction(0)) == Fraction(value)
  return high, middle, low


def _exact(words):
  return sum((Fraction(float(word)) for word in words), Fraction())


_HELPER_KERNEL = r"""
kernel void carry_helper_probe(
    device const float* words [[buffer(0)]],
    device int* out_ok [[buffer(1)]],
    device float* out_triple [[buffer(2)]],
    constant int& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= uint(count)) return;
  thread float expansion[24];
  for (int i = 0; i < 24; i++) expansion[i] = words[24 * index + i];
  SdfProjection53 out = {0.0f, 0.0f, 0.0f};
  bool ok = sw_expansion_round53_exact(expansion, 24, &out);
  out_ok[index] = ok ? 1 : 0;
  out_triple[3 * index] = out.hi;
  out_triple[3 * index + 1] = out.mid;
  out_triple[3 * index + 2] = out.low;
}
"""

_WRAPPER_KERNEL = r"""
kernel void carry_wrapper_probe(
    device const float* words [[buffer(0)]],
    device float* out_triple [[buffer(1)]],
    constant int& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= uint(count)) return;
  thread float expansion[24];
  for (int i = 0; i < 24; i++) expansion[i] = words[24 * index + i];
  SdfProjection53 out = sw_expansion_round53(expansion, 24);
  out_triple[3 * index] = out.hi;
  out_triple[3 * index + 1] = out.mid;
  out_triple[3 * index + 2] = out.low;
}
"""

_PRODUCER_KERNEL = r"""
kernel void carry_producer_probe(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& count [[buffer(2)]],
    constant int& op [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= uint(count)) return;
  uint ib = index * 9;
  SdfWide a = {input[ib], input[ib + 1], input[ib + 2]};
  SdfWide b = {input[ib + 3], input[ib + 4], input[ib + 5]};
  SdfWide c = {input[ib + 6], input[ib + 7], input[ib + 8]};
  SdfWide v = (op == 0) ? sw_add(a, b)
      : ((op == 1) ? sw_mul(a, b) : sw_source_fma(a, b, c));
  output[3 * index] = v.hi;
  output[3 * index + 1] = v.mid;
  output[3 * index + 2] = v.lo;
}
"""

_H = "0x1.FFFFFEp+0"
_HBIG = "0x1.FFFFFEp+10"
_HX = "0x1.FFFFFEp+5"
_TINY = "0x1p-149"

# (tag, word-hex list, oracle-hex). Rows 1-4 are the independently
# reproduced native-probe table (limb offsets: dup rows share limb-bit 31
# of limb 18 at quantum scale 2**-23; sub16 shares limb-bit 31 of limb 17
# at subnormal scale); the rest audit multiple carries (8x/4x max
# mantissa), cross-limb spans, borrow chains, the permitted 24-word
# count, exact cancellation to zero, and single-word controls.
_LIMB_WORD_ROWS = (
    ("dup-pos", ("0x1p-23", "0x1p-23", "0x1p+0"), "0x1.0000040000000p+0"),
    ("dup-neg", ("-0x1p-23", "-0x1p-23", "-0x1p+0"), "-0x1.0000040000000p+0"),
    ("sub16", ("0x1p-149",) * 16 + ("0x1p-94",), "0x1.0000000000002p-94"),
    ("control", ("0x1p-23", "0x1p+0"), "0x1.0000020000000p+0"),
    ("dup8-max", (_H,) * 8 + ("0x1p+0",), "0x1.0fffff0000000p+4"),
    ("dup4-max", (_H,) * 4 + ("0x1p+0",), "0x1.1fffff0000000p+3"),
    ("dup2-big", (_HBIG,) * 2 + ("0x1p+0",), "0x1.000fff0000000p+12"),
    ("borrow-chain", (_H,) * 8 + ("-0x1.FFFFFEp+0",) * 8 + ("0x1p-20",),
     "0x1.0000000000000p-20"),
    ("zero-cancel", (_H, "-0x1.FFFFFEp+0"), "0x0.0p+0"),
    ("count24", ("0x1p-30",) * 23 + ("0x1p-1",), "0x1.000000b800000p-1"),
    ("cross-limb", (_HX,) * 3 + ("0x1p+0",), "0x1.81fffe8000000p+7"),
    ("control-single", (_H,), "0x1.fffffe0000000p+0"),
)

# (tag, op[0=add,1=mul,2=fma], a/b/c (hi,mid,lo) triples). Duplicate-heavy
# and subnormal-word producer shapes; oracles derive from exact rationals.
_PRODUCER_CASES = (
    ("add-dup-max", 0, (_H, "0x0p+0", "0x0p+0"), (_H, "0x0p+0", "0x0p+0"),
     ("0x0p+0", "0x0p+0", "0x0p+0")),
    ("add-dup-maxC", 0, (_H, "0x1p-24", "0x0p+0"), (_H, "0x1p-24", "0x0p+0"),
     ("0x0p+0", "0x0p+0", "0x0p+0")),
    ("mul-dup-X", 1, ("0x1.8p+0", "0x1.8p+0", "0x0p+0"),
     ("0x1.8p+0", "0x1.8p+0", "0x0p+0"), ("0x0p+0", "0x0p+0", "0x0p+0")),
    ("mul-dup-H", 1, (_H, _H, "0x0p+0"), (_H, _H, "0x0p+0"),
     ("0x0p+0", "0x0p+0", "0x0p+0")),
    ("fma-dup", 2, ("0x1.8p+0", "0x1.8p+0", "0x0p+0"),
     ("0x1.8p+0", "0x1.8p+0", "0x0p+0"), ("0x1.8p+0", "0x1.8p+0", "0x0p+0")),
    ("add-subnorm", 0, ("0x1p+0", _TINY, _TINY), ("0x1p+0", _TINY, "0x0p+0"),
     ("0x0p+0", "0x0p+0", "0x0p+0")),
    ("mul-subnorm", 1, ("0x1p+0", _TINY, "0x0p+0"), ("0x1p+0", _TINY, "0x0p+0"),
     ("0x0p+0", "0x0p+0", "0x0p+0")),
    ("fma-subnorm", 2, ("0x1p+0", _TINY, "0x0p+0"), ("0x1p+0", "0x0p+0", "0x0p+0"),
     ("-0x1p+0", "0x0p+0", "0x0p+0")),
    ("add-ctrl", 0, ("0x1p+0", "0x0p+0", "0x0p+0"), ("0x1p-23", "0x0p+0", "0x0p+0"),
     ("0x0p+0", "0x0p+0", "0x0p+0")),
)

_FMA64 = (
    (float.fromhex("0x1.4c1be6fb45fefp+0"), float.fromhex("0x1.9dc69b20f4dbap-1"), float.fromhex("-0x1.0c6556d7d3314p+0"),),
    (float.fromhex("0x1.a82df2cacba22p-1"), float.fromhex("0x1.195135776dcdfp+0"), float.fromhex("-0x1.d220fd793c975p-1"),),
    (float.fromhex("0x1.8d12214c4d94cp-1"), float.fromhex("0x1.8133a0b6acc3ep+0"), float.fromhex("-0x1.2abc2ac6b4475p+0"),),
    (float.fromhex("0x1.a69aa6d9ff394p+0"), float.fromhex("0x1.867636e8e2226p+0"), float.fromhex("-0x1.42496118f8be3p+1"),),
    (float.fromhex("0x1.3e5e46bc9de4bp-1"), float.fromhex("0x1.74bf1914d7334p+0"), float.fromhex("-0x1.cf8ea7cc716cep-1"),),
    (float.fromhex("0x1.48423ab442b63p-1"), float.fromhex("0x1.5aa9d627b9545p+0"), float.fromhex("-0x1.bc834b576c0b4p-1"),),
    (float.fromhex("0x1.b5320547d7374p+0"), float.fromhex("0x1.1326520553c44p+0"), float.fromhex("-0x1.d5e62a9de2665p+0"),),
    (float.fromhex("0x1.4e9f47d531338p-1"), float.fromhex("0x1.fb102cad07a9cp+0"), float.fromhex("-0x1.4b654d96cb335p+0"),),
    (float.fromhex("0x1.d7c0d82bb8162p+0"), float.fromhex("0x1.375a73a18c5a2p+0"), float.fromhex("-0x1.1ee0fc0d29baep+1"),),
    (float.fromhex("0x1.42b4ccf1e7b63p-1"), float.fromhex("0x1.5e771c6530896p-1"), float.fromhex("-0x1.b9c95851b0ab5p-2"),),
    (float.fromhex("0x1.a78aaffd8cc90p+0"), float.fromhex("0x1.acb89cbd58fe6p+0"), float.fromhex("-0x1.62a6a58d16ab9p+1"),),
    (float.fromhex("0x1.2805b6eb4df0ap+0"), float.fromhex("0x1.d4911ba5092c6p+0"), float.fromhex("-0x1.0ee9203c0d5fbp+1"),),
    (float.fromhex("0x1.f59cf6623b4f8p+0"), float.fromhex("0x1.a79afc1e08f6dp+0"), float.fromhex("-0x1.9f03064be72cfp+1"),),
    (float.fromhex("0x1.5f2509f2b5a08p-1"), float.fromhex("0x1.67564a9c77cb1p+0"), float.fromhex("-0x1.ece34c0dcc96cp-1"),),
    (float.fromhex("0x1.9d69a52d8a768p-1"), float.fromhex("0x1.002cb9e12626dp+0"), float.fromhex("-0x1.9db1e0d050064p-1"),),
    (float.fromhex("0x1.161aea0b1eb8fp-1"), float.fromhex("0x1.862fc8b9315ddp+0"), float.fromhex("-0x1.a7e0ec0afbbb7p-1"),),
    (float.fromhex("0x1.f6651fb6d7496p+0"), float.fromhex("0x1.8eea22e2e851cp-1"), float.fromhex("-0x1.876e58b8bedb7p+0"),),
    (float.fromhex("0x1.5b3159346984cp+0"), float.fromhex("0x1.e7bf99128a38ep-1"), float.fromhex("-0x1.4abf5e8595a1bp+0"),),
    (float.fromhex("0x1.028f80291e38ap+0"), float.fromhex("0x1.340944131c7d6p+0"), float.fromhex("-0x1.371e0346a10d3p+0"),),
    (float.fromhex("0x1.26f428128a33ap+0"), float.fromhex("0x1.9576ec1cc2073p+0"), float.fromhex("-0x1.d329498dccb66p+0"),),
    (float.fromhex("0x1.27f42951aa828p+0"), float.fromhex("0x1.20bf49160452ap-1"), float.fromhex("-0x1.4dcfd352cd45bp-1"),),
    (float.fromhex("0x1.6bd58b5f72072p-1"), float.fromhex("0x1.fc5704b59f276p+0"), float.fromhex("-0x1.693bb794ddbfap+0"),),
    (float.fromhex("0x1.07ef81f5c38cap+0"), float.fromhex("0x1.b3ac422965c92p-1"), float.fromhex("-0x1.c12d905b349f2p-1"),),
    (float.fromhex("0x1.749db549cb7d8p-1"), float.fromhex("0x1.bc40fa7e68776p+0"), float.fromhex("-0x1.43500f104cc3bp+0"),),
    (float.fromhex("0x1.7d73902cdebf6p-1"), float.fromhex("0x1.a7565b3a51b3ep-1"), float.fromhex("-0x1.3b65517b14af4p-1"),),
    (float.fromhex("0x1.83229524ba946p+0"), float.fromhex("0x1.531e8311b9254p+0"), float.fromhex("-0x1.006a76d817fe5p+1"),),
    (float.fromhex("0x1.13d4deaec2cf2p+0"), float.fromhex("0x1.ec9b38e155386p+0"), float.fromhex("-0x1.09622c70ac234p+1"),),
    (float.fromhex("0x1.110c96366fd9bp+0"), float.fromhex("0x1.ec83361c6b4a3p-1"), float.fromhex("-0x1.06a812fac9f30p+0"),),
    (float.fromhex("0x1.ab893ad5b6c32p+0"), float.fromhex("0x1.f4f39d49d2c16p-1"), float.fromhex("-0x1.a24f719042666p+0"),),
    (float.fromhex("0x1.47d61e02071ddp-1"), float.fromhex("0x1.3608a9bfbc6ccp+0"), float.fromhex("-0x1.8d085e1be8963p-1"),),
    (float.fromhex("0x1.fef35b467efbap-1"), float.fromhex("0x1.1cee8c56adcf0p+0"), float.fromhex("-0x1.1c590d968fe2ep+0"),),
    (float.fromhex("0x1.0018627443b2fp+0"), float.fromhex("0x1.e2531b9fe2ec3p-1"), float.fromhex("-0x1.e2810df786170p-1"),),
    (float.fromhex("0x1.53d61b52e9199p+0"), float.fromhex("0x1.d4d33f5a7966ap-1"), float.fromhex("-0x1.372dea110dec8p+0"),),
    (float.fromhex("0x1.fcd460fd0d7a0p-1"), float.fromhex("0x1.6e4b5118d503dp-1"), float.fromhex("-0x1.6c06a9cb2ba10p-1"),),
    (float.fromhex("0x1.6c88e10ee9e2cp+0"), float.fromhex("0x1.22389add36778p+0"), float.fromhex("-0x1.9d43a8187351dp+0"),),
    (float.fromhex("0x1.376d2ab6b1078p+0"), float.fromhex("0x1.85aed07fb550ep+0"), float.fromhex("-0x1.da0d8d3cdc076p+0"),),
    (float.fromhex("0x1.1c1df828ce370p+0"), float.fromhex("0x1.7e1c242892f59p+0"), float.fromhex("-0x1.a813f24db0d85p+0"),),
    (float.fromhex("0x1.d88cb025b6e98p-1"), float.fromhex("0x1.69e547a83ac1dp+0"), float.fromhex("-0x1.4e02ce21924d9p+0"),),
    (float.fromhex("0x1.3fe317b02ee63p-1"), float.fromhex("0x1.ae35c240c222cp+0"), float.fromhex("-0x1.0cc94e2499799p+0"),),
    (float.fromhex("0x1.aa83289a54f2bp+0"), float.fromhex("0x1.b9670e7290036p+0"), float.fromhex("-0x1.6fb3d3563b7d9p+1"),),
    (float.fromhex("0x1.f2d5be7a909fap+0"), float.fromhex("0x1.2b49af5b06cd4p+0"), float.fromhex("-0x1.23979b705a6cbp+1"),),
    (float.fromhex("0x1.aa462ab2c2c68p+0"), float.fromhex("0x1.ee5d39a55a453p+0"), float.fromhex("-0x1.9b97531423fc9p+1"),),
    (float.fromhex("0x1.e0d0a7eb5c99ep-1"), float.fromhex("0x1.ffc05c3b020fep+0"), float.fromhex("-0x1.e094e74413e82p+0"),),
    (float.fromhex("0x1.2f3ca4d60da09p+0"), float.fromhex("0x1.29e2a25ed2e46p+0"), float.fromhex("-0x1.60d9d098fc6edp+0"),),
    (float.fromhex("0x1.f781806d50e8ep+0"), float.fromhex("0x1.b5982b4f1ee40p-1"), float.fromhex("-0x1.ae55aae2e924cp+0"),),
    (float.fromhex("0x1.52aad270b1068p-1"), float.fromhex("0x1.62e75b36cfb30p+0"), float.fromhex("-0x1.d5824517394ccp-1"),),
    (float.fromhex("0x1.4c41bd69404acp+0"), float.fromhex("0x1.9d69a61b4a748p+0"), float.fromhex("-0x1.0c47957c0e024p+1"),),
    (float.fromhex("0x1.a2400df92946ap+0"), float.fromhex("0x1.3273edba5659bp+0"), float.fromhex("-0x1.f4adf5694b2afp+0"),),
    (float.fromhex("0x1.4fb2246f3b0a4p+0"), float.fromhex("0x1.864f142a917fcp+0"), float.fromhex("-0x1.ffd11942dc596p+0"),),
    (float.fromhex("0x1.3c600a7620f6bp+0"), float.fromhex("0x1.429c85afb2e1fp-1"), float.fromhex("-0x1.8eb23e66884fdp-1"),),
    (float.fromhex("0x1.e755bc2cba3b0p-1"), float.fromhex("0x1.5c3774416a6b9p+0"), float.fromhex("-0x1.4b71101abcdf6p+0"),),
    (float.fromhex("0x1.abe13bed027a3p-1"), float.fromhex("0x1.af72eee869528p+0"), float.fromhex("-0x1.6890285c31f68p+0"),),
    (float.fromhex("0x1.911ba24fc2be6p-1"), float.fromhex("0x1.32c09f9f02aa0p+0"), float.fromhex("-0x1.e0a0d3fca5995p-1"),),
    (float.fromhex("0x1.f0abaf91b252bp+0"), float.fromhex("0x1.0f445cc717b4ap+0"), float.fromhex("-0x1.07253173d9483p+1"),),
    (float.fromhex("0x1.baea0df8170dfp+0"), float.fromhex("0x1.819c61a80d221p+0"), float.fromhex("-0x1.4d94486d457f1p+1"),),
    (float.fromhex("0x1.97d28d07e08d6p-1"), float.fromhex("0x1.741affa037a8ep+0"), float.fromhex("-0x1.28647d2bc4218p+0"),),
    (float.fromhex("0x1.2c84afdd704fap-1"), float.fromhex("0x1.bb3d2d98e341ap+0"), float.fromhex("-0x1.0428b7be7d5d6p+0"),),
    (float.fromhex("0x1.a85bc8e22b309p+0"), float.fromhex("0x1.46108f9af9feap+0"), float.fromhex("-0x1.0e402c295a292p+1"),),
    (float.fromhex("0x1.9cc33b846d7ebp+0"), float.fromhex("0x1.3d09cf6ba2b2dp+0"), float.fromhex("-0x1.ff2d8fc9faee6p+0"),),
    (float.fromhex("0x1.350931f46c22bp+0"), float.fromhex("0x1.68fbe80eed670p-1"), float.fromhex("-0x1.b3c504a9b16f8p-1"),),
    (float.fromhex("0x1.4ad0d737e7a36p+0"), float.fromhex("0x1.8602f08adea2cp+0"), float.fromhex("-0x1.f7fdf6fa3d2cep+0"),),
    (float.fromhex("0x1.95f623e6b978cp+0"), float.fromhex("0x1.946ab9bfc0adap+0"), float.fromhex("-0x1.40a8d61abd92ap+1"),),
    (float.fromhex("0x1.260fd38ba11b2p+0"), float.fromhex("0x1.9cf05901f01fbp-1"), float.fromhex("-0x1.da559042efef2p-1"),),
    (float.fromhex("0x1.0acb1e3491a7ap-1"), float.fromhex("0x1.74173c1d8e5e1p+0"), float.fromhex("-0x1.83c7611b609cbp-1"),),
)


def _row_oracle(words):
  total = sum((Fraction(_w(w)) for w in words), Fraction(0))
  return float(total)


def _w(word):
  return float.fromhex(word) if isinstance(word, str) else float(word)


def _producer_oracle(op, a, b, c):
  A = sum((Fraction(_w(w)) for w in a), Fraction(0))
  B = sum((Fraction(_w(w)) for w in b), Fraction(0))
  C = sum((Fraction(_w(w)) for w in c), Fraction(0))
  if op == 0:
    return float(A + B)
  if op == 1:
    return float(A * B)
  return float(A * B + C)


def _helper_source():
  return (Path(__file__).parents[1] / "mujoco_metal/shaders" /
          "plugin_sdf.metal").read_text()


def _complete_source():
  from mujoco_metal.coupled_constraints import _coupled_shader_source
  return _coupled_shader_source()


def _sources():
  return {"helper-only": _helper_source(), "complete": _complete_source()}


def test_limb_carry_oracle_contract_cpu():
  assert len(_LIMB_WORD_ROWS) == 12
  for tag, words, oracle_hex in _LIMB_WORD_ROWS:
    assert len(words) <= 24, tag
    assert _row_oracle(words).hex() == oracle_hex, tag
  # The independently reproduced table rows pin the reported values.
  table = dict((tag, oracle) for tag, _, oracle in _LIMB_WORD_ROWS)
  assert table["dup-pos"] == "0x1.0000040000000p+0"
  assert table["dup-neg"] == "-0x1.0000040000000p+0"
  assert table["sub16"] == "0x1.0000000000002p-94"
  for tag, op, a, b, c in _PRODUCER_CASES:
    assert len(a) == len(b) == len(c) == 3, tag
    assert all(len(t) == 3 for t in (a, b, c))
  assert len(_FMA64) == 64
  for triple in _FMA64:
    assert len(triple) == 3
    _split3(triple[0]), _split3(triple[1]), _split3(triple[2])


def test_shader_source_ownership_cpu():
  helper, complete = _sources()["helper-only"], _sources()["complete"]
  assert "sw_expansion_round53_exact" in helper
  assert "sw_expansion_round53_exact" in complete
  assert "sw_limbs_add_pow2" in helper and "sw_limbs_add_pow2" in complete
  assert "kernel void contact_produce_mesh_sdf" in complete
  assert "kernel void contact_produce_mesh_sdf" not in helper
  assert "kernel void mask_failed_candidate_worlds" in complete
  assert "kernel void mask_failed_candidate_worlds" not in helper


def _compile(library_source):
  import torch
  if not torch.backends.mps.is_available() or not hasattr(
      torch.mps, "compile_shader"):
    pytest.skip("MPS compile_shader is unavailable")
  return torch.mps.compile_shader(
      library_source + "\n" + _HELPER_KERNEL + "\n" + _WRAPPER_KERNEL + "\n"
      + _PRODUCER_KERNEL)


def _run_words(lib, rows):
  import torch
  n = len(rows)
  host = np.zeros((n, 24), dtype=np.float32)
  for i, (_, words, _) in enumerate(rows):
    for j, w in enumerate(words):
      host[i, j] = _f32(_w(w))
  inp = torch.as_tensor(host.copy(), dtype=torch.float32, device="mps")
  ok = torch.empty((n,), dtype=torch.int32, device="mps")
  tri_h = torch.empty((n, 3), dtype=torch.float32, device="mps")
  tri_w = torch.empty((n, 3), dtype=torch.float32, device="mps")
  cnt = torch.tensor([n], dtype=torch.int32, device="mps")
  lib.carry_helper_probe(inp.reshape(-1), ok, tri_h.reshape(-1), cnt,
                         threads=(n,), group_size=(1,))
  lib.carry_wrapper_probe(inp.reshape(-1), tri_w.reshape(-1), cnt,
                          threads=(n,), group_size=(1,))
  torch.mps.synchronize()
  return (ok.cpu().numpy().tolist(), tri_h.cpu().numpy(),
          tri_w.cpu().numpy())


def _run_producer(lib, cases, op_override=None):
  import torch
  n = len(cases)
  host = np.zeros((n, 9), dtype=np.float32)
  ops = []
  for i, (tag, op, a, b, c) in enumerate(cases):
    words = ([_w(w) for w in a] + [_w(w) for w in b]
             + [_w(w) for w in c])
    host[i, :] = [_f32(w) for w in words]
    ops.append(op if op_override is None else op_override)
  failures = []
  for op in sorted(set(ops)):
    idx = [i for i, o in enumerate(ops) if o == op]
    sub = host[idx]
    m = len(sub)
    inp = torch.as_tensor(sub.copy(), dtype=torch.float32, device="mps")
    out = torch.empty((m, 3), dtype=torch.float32, device="mps")
    cnt = torch.tensor([m], dtype=torch.int32, device="mps")
    opc = torch.tensor([op], dtype=torch.int32, device="mps")
    lib.carry_producer_probe(inp.reshape(-1), out.reshape(-1), cnt, opc,
                             threads=(m,), group_size=(1,))
    torch.mps.synchronize()
    actual = out.cpu().numpy()
    for k, i in enumerate(idx):
      tag, _, a, b, c = cases[i]
      oracle = _producer_oracle(ops[i], a, b, c)
      got = (float(actual[k, 0]), float(actual[k, 1]),
             float(actual[k, 2]))
      if _exact(got) != _exact(_split3(oracle)):
        failures.append(f"{tag}: got exact {float(_exact(got)).hex()} "
                        f"want {oracle.hex()}")
  return failures


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native carry witness is parent-dispatched")
@pytest.mark.parametrize("unit", ["helper-only", "complete"])
def test_native_limb_carry_helper_and_wrapper_gpu(unit):
  lib = _compile(_sources()[unit])
  oks, tri_h, tri_w = _run_words(lib, _LIMB_WORD_ROWS)
  failures = []
  for i, (tag, words, oracle_hex) in enumerate(_LIMB_WORD_ROWS):
    oracle = float.fromhex(oracle_hex)
    if not oks[i]:
      failures.append(f"{tag}: helper returned false (silent fallback)")
      continue
    for got, label in (((float(tri_h[i, 0]), float(tri_h[i, 1]),
                               float(tri_h[i, 2])), "helper"),
                       ((float(tri_w[i, 0]), float(tri_w[i, 1]),
                                float(tri_w[i, 2])), "wrapper")):
      if float(_exact(got)) != oracle:
        failures.append(
            f"{tag}[{label}]: got {float(_exact(got)).hex()} "
            f"want {oracle.hex()}")
  assert not failures, "carry mismatches:\n" + "\n".join(failures)


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native carry witness is parent-dispatched")
@pytest.mark.parametrize("unit", ["helper-only", "complete"])
def test_native_producer_duplicates_gpu(unit):
  lib = _compile(_sources()[unit])
  failures = _run_producer(lib, _PRODUCER_CASES)
  assert not failures, "producer mismatches:\n" + "\n".join(failures)


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native carry witness is parent-dispatched")
@pytest.mark.parametrize("unit", ["helper-only", "complete"])
def test_native_fma64_device_fixture_gpu(unit):
  lib = _compile(_sources()[unit])
  cases = [("fma64-%d" % i, 2, _split3(a), _split3(b), _split3(c))
           for i, (a, b, c) in enumerate(_FMA64)]
  failures = _run_producer(lib, cases)
  assert not failures, "fma64 mismatches:\n" + "\n".join(failures[:12])
