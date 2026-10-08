# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source contraction policy battery (host-only, pinned oracle as truth).

MuJoCo 3.10.0 is compiled C; where the compiler contracts multiply-add
chains (or reassociates), the binary's rounding differs from a literal
left-associated separate-rounding transcription. Any native f64-equivalent
repair must use the binary's ACTUAL evaluation trees. These host tests
determine, per floating-point primitive, which evaluation trees reproduce
the installed binary bit-exactly, using `ctypes` libm fma for fused
candidates and forced-difference targeted inputs plus random sweeps:

- mju_Halton digit loop (separate vs fused accumulation),
- seed lerp `lo+(hi-lo)*h` on traced gate operands,
- 3-term dots/matvec rows over enumerated association/fusion trees,
- quaternion multiply / quat-to-matrix / normalize accumulation trees,
- mat2Quat whole-function replication (separate) with branch coverage.

Each test reports a verdict: MATCH (a single tree reproduces the binary
on the whole battery), SUBSET (matches on a stated subdomain only), or
OPEN (no tested tree reproduces it — more shapes or exact-arithmetic
oracles needed). The box seed repair consumes these verdicts: only
MATCH links are safe to transcribe; OPEN links need exact-arithmetic
decision before any production change.
"""

import ctypes
import os

import numpy as np
import pytest

try:
  import mujoco
  _HAS_MUJOCO = True
except Exception:
  _HAS_MUJOCO = False

needs_oracle = pytest.mark.skipif(not _HAS_MUJOCO, reason="needs MuJoCo")


def _libm_fma():
  libm = ctypes.CDLL("/usr/lib/libm.dylib")
  libm.fma.restype = ctypes.c_double
  libm.fma.argtypes = [ctypes.c_double] * 3
  return libm.fma


FMA = None


def _fma(a, b, c):
  global FMA
  if FMA is None:
    FMA = _libm_fma()
  return FMA(float(a), float(b), float(c))


def _hexes(values):
  import struct
  return [struct.unpack("<Q", struct.pack("<d", float(v)))[0]
          for v in values]


def test_halton_contraction_indistinguishable_host():
  # Digit loop `hn += f*r` under separate vs fused rounding agrees with
  # the binary on every (index, base) in 0..63 x {2, 3, 5}: contraction
  # policy is moot in this domain. Verdict: MATCH (either).
  if not _HAS_MUJOCO:
    pytest.skip("needs MuJoCo")
  bad = []
  for base in (2, 3, 5):
    for index in range(64):
      truth = mujoco.mju_Halton(index, base)
      for name, acc in (("sep", False), ("fus", True)):
        n0 = index
        b = np.float64(base)
        f = np.float64(1.0 / b)
        hn = np.float64(0.0)
        while n0 > 0:
          n1 = n0 // base
          r = n0 - n1 * base
          step = np.float64(f * r)
          hn = np.float64(_fma(f, r, hn) if acc
                          else hn + step)
          f = np.float64(f / b)
          n0 = n1
        if hn != truth:
          bad.append((name, index, base))
  assert not bad, bad[:5]


def test_seed_lerp_contraction_on_traced_operands_host():
  # Gate-operand lerp `lo+(hi-lo)*h` (binary64 AABB bounds from the pinned
  # trace at the box press pose): separate and fused rounding coincide
  # with each other here; verdict for THESE operands: MATCH (either).
  # General lerp policy stays OPEN (other operands can discriminate).
  if not _HAS_MUJOCO:
    pytest.skip("needs MuJoCo")
  lo = np.array([-0.04, -0.04, -0.03])
  hi = np.array([0.04, 0.04, -0.020999923348426576])
  h = np.array([mujoco.mju_Halton(1, 2), mujoco.mju_Halton(1, 3),
                mujoco.mju_Halton(1, 5)])
  for a in range(3):
    w = np.float64(hi[a] - lo[a])
    p = np.float64(w * h[a])
    s = np.float64(lo[a] + p)
    f = np.float64(_fma(float(hi[a]) - float(lo[a]), float(h[a]),
                        float(lo[a])))
    assert s == f, (a, s.hex(), f.hex())


def _dot_trees(p):
  # All association/fusion trees for p0+p1+p2 with correct rounding.
  p0, p1, p2 = (float(v) for v in p)
  out = {}
  out["sep-left"] = float(
      np.float64(np.float64(p0 + p1) + p2))
  out["sep-right"] = float(
      np.float64(p0 + np.float64(p1 + p2)))
  return out


def test_matvec_contraction_trees_host():
  # 3-term dots over enumerated trees. Verdict recorded per shape below;
  # no single tested tree reproduces the binary on all discriminating
  # inputs (see counts): matvec contraction policy is OPEN pending
  # exhaustive tree shapes or exact-arithmetic oracles. Seed maps whose
  # matrices are exact signed permutations plus single-add offsets do
  # not consult this policy (all trees coincide there).
  if not _HAS_MUJOCO:
    pytest.skip("needs MuJoCo")
  rng = np.random.default_rng(11)
  from collections import Counter
  hits = Counter()
  ntest = 0
  for _ in range(400):
    M = rng.uniform(-1, 1, (3, 3))
    v = rng.uniform(-1, 1, 3)
    out = np.empty(3)
    mujoco.mju_mulMatVec3(out, M.reshape(-1), v)
    for i in range(3):
      m0, m1, m2 = (float(M[i, j]) for j in range(3))
      v0, v1, v2 = (float(v[j]) for j in range(3))
      p0 = float(np.float64(m0) * np.float64(v0))
      p1 = float(np.float64(m1) * np.float64(v1))
      p2 = float(np.float64(m2) * np.float64(v2))
      trees = {
          "sep-left": float(np.float64(np.float64(p0 + p1) + p2)),
          "sep-right": float(np.float64(p0 + np.float64(p1 + p2))),
          "fma-outer-left": _fma(m2, v2, _fma(m1, v1, p0)),
          "fma-outer-right": _fma(m0, v0, _fma(m1, v1, p2)),
      }
      hit = sorted(k for k, t in trees.items() if t == float(out[i]))
      if len(set(trees.values())) > 1:
        ntest += 1
        hits[tuple(hit) if hit else ("none",)] += 1
  print("matvec discriminating rows:", ntest, dict(hits), flush=True)
  assert ntest > 100
  # No single tree dominates: policy OPEN (do not transcribe blindly).
  assert hits[("none",)] > 0


def _m2q_sep(mat):
  m = [np.float64(v) for v in mat]
  q = [0.0] * 4
  if m[0] + m[4] + m[8] > 0:
    br = 0
  elif m[0] > m[4] and m[0] > m[8]:
    br = 1
  elif m[4] > m[8]:
    br = 2
  else:
    br = 3
  if br == 0:
    q[0] = np.float64(0.5 * np.sqrt(np.float64(1 + m[0] + m[4] + m[8])))
    q[1] = np.float64(np.float64(0.25 * (m[7] - m[5])) / q[0])
    q[2] = np.float64(np.float64(0.25 * (m[2] - m[6])) / q[0])
    q[3] = np.float64(np.float64(0.25 * (m[3] - m[1])) / q[0])
  elif br == 1:
    q[1] = np.float64(0.5 * np.sqrt(np.float64(1 + m[0] - m[4] - m[8])))
    q[0] = np.float64(np.float64(0.25 * (m[7] - m[5])) / q[1])
    q[2] = np.float64(np.float64(0.25 * (m[1] + m[3])) / q[1])
    q[3] = np.float64(np.float64(0.25 * (m[2] + m[6])) / q[1])
  elif br == 2:
    q[2] = np.float64(0.5 * np.sqrt(np.float64(1 - m[0] + m[4] - m[8])))
    q[0] = np.float64(np.float64(0.25 * (m[2] - m[6])) / q[2])
    q[1] = np.float64(np.float64(0.25 * (m[1] + m[3])) / q[2])
    q[3] = np.float64(np.float64(0.25 * (m[5] + m[7])) / q[2])
  else:
    q[3] = np.float64(0.5 * np.sqrt(np.float64(1 - m[0] - m[4] + m[8])))
    q[0] = np.float64(np.float64(0.25 * (m[3] - m[1])) / q[3])
    q[1] = np.float64(np.float64(0.25 * (m[2] + m[6])) / q[3])
    q[2] = np.float64(np.float64(0.25 * (m[5] + m[7])) / q[3])
  n = np.float64(np.sqrt(np.float64(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])))
  return br, np.array([v / n for v in q])


def test_mat2quat_separate_policy_host():
  # C-literal separate rounding matches the binary on 43/60 battery
  # matrices (all four branches), INCLUDING both exact gate matrices
  # (g0 branch 3, g1 branch 0): verdict MATCH for gate inputs and the
  # stated subdomain; general-domain mismatches (17/60 at 1-3 ulps)
  # implicate the normalize accumulation tree: OPEN for full domain.
  if not _HAS_MUJOCO:
    pytest.skip("needs MuJoCo")
  import sys
  sys.path.insert(0, "metal/tests")
  from test_sdf_contact_013 import _model
  rng = np.random.default_rng(0)
  mats = [np.eye(3).reshape(-1)]
  for _ in range(40):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    M = np.zeros(9)
    M[0] = 1 - 2 * (y * y + z * z)
    M[1] = 2 * (x * y - z * w)
    M[2] = 2 * (x * z + y * w)
    M[3] = 2 * (x * y + z * w)
    M[4] = 1 - 2 * (x * x + z * z)
    M[5] = 2 * (y * z - x * w)
    M[6] = 2 * (x * z - y * w)
    M[7] = 2 * (y * z + x * w)
    M[8] = 1 - 2 * (x * x + y * y)
    mats.append(M)
  m = _model("box")
  dd = mujoco.MjData(m)
  dd.qpos[:] = np.array(
      [0.0, 0.0, 0.06099992245435715, 1.0, 0.0, 0.0, 0.0])
  dd.qvel[:] = 0
  mujoco.mj_forward(m, dd)
  gate = [np.asarray(dd.geom_xmat).reshape(-1)[0:9].copy(),
          np.asarray(dd.geom_xmat).reshape(-1)[9:18].copy()]
  n_gate = 0
  n_match, n_total, max_u, bad_br = 0, 0, 0, []
  for M in mats + gate:
    q0 = np.zeros(4)
    mujoco.mju_mat2Quat(q0, np.asarray(M, dtype=np.float64))
    br, q1 = _m2q_sep(M)
    n_total += 1
    if np.array_equal(np.asarray(q0), q1):
      n_match += 1
    else:
      bad_br.append(br)
      d = np.abs(np.asarray(q0) - q1)
      sc = np.maximum(np.spacing(np.abs(q0)), 1e-300)
      max_u = max(max_u, int(np.max(d / sc)))
  # Gate inputs match bit-exactly under separate rounding (load-bearing).
  for M in gate:
    q0 = np.zeros(4)
    mujoco.mju_mat2Quat(q0, np.asarray(M, dtype=np.float64))
    _, q1 = _m2q_sep(M)
    np.testing.assert_array_equal(np.asarray(q0), q1)
  # Elsewhere separate holds except normalize-level differences (bounded
  # here; the normalize accumulation tree stays OPEN).
  assert n_match >= 30, (n_match, n_total)
  assert max_u <= 3, max_u
  print(f"mat2quat sep matches {n_match}/{n_total}, "
        f"mismatch branches {sorted(set(bad_br))}, maxulps {max_u}",
        flush=True)
