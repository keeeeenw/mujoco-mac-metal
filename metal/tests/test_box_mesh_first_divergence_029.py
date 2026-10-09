# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Box mesh-SDF parity regression at the exact 013 press pose.

Bounded reproducer plus structured same-input stage packets for
`box@(0.0,0.0)/press`. Methodology (same represented float32 gate pose on
both sides, one producer epoch, position-matched slots) is asserted.

PRODUCTION REPAIR (mat2Quat route): the native SDF seed path now derives
its quats from FK geom_xmat via pinned mju_mat2Quat (source order,
correctly rounded sw_*), matching mjc_SDF which does
mju_mat2Quat(geom_xmat) then mapPose. Previously it used FK quat
composition directly (1.67e-15 quat gap -> 1.33e-15 T21 tilt -> seed
transverse sign flip -> x-mirror). T21 is now bit-exact; seeds agree to
1e-12 absolute (9/12 bit-exact, rest 1-ulp dot noise); press contact
agrees directly (no mirror). Fails without the repair (mirror + T tilt).
"""

import os

import numpy as np
import pytest


def _needs_gpu():
  return pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                            reason="opt-in GPU")


_GATE_Z = 0.06099992245435715
_GATE_QP = np.array([0.0, 0.0, _GATE_Z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _gate_model():
  import sys
  sys.path.insert(0, "metal/tests")
  from test_sdf_contact_013 import _model, _touch
  m = _model("box")
  zt = _touch(m, 0.0, 0.0)
  z = zt - 0.004
  # Touch height is float64-stable; the gate qp is its float32 cast.
  assert abs(float(zt) - 0.0649999237060543) < 1e-12, zt
  assert _GATE_QP[2] == np.float32(float(z)), (_GATE_QP[2], z)
  return m


@_needs_gpu()
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_box_press_first_divergence_packet_gpu():
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from test_sdf_seed_source_trace_013 import (
      _capture_staged_workspace, _collapse_words, _collapse_wide,
      _STAGE, _STATE_WORDS)
  from mujoco_metal.simulation import MetalSimulation
  m = _gate_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=_GATE_QP.reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  epochs, _ = _capture_staged_workspace(sim)
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  # Methodology: exactly one complete producer epoch at this pose.
  assert len(epochs) == 1
  ep = epochs[-1]
  assert ep[0][0] == "_contact_sdf_seed_init"
  assert ep[-1][0] == "_contact_sdf_finalize_producer"
  # Same-input oracle at the identical represented pose.
  d = mujoco.MjData(m)
  d.qpos[:] = np.asarray(_GATE_QP, dtype=np.float64)
  d.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d)
  assert int(d.ncon) == 2
  mask = np.asarray(s["contact_mask"][0].cpu().numpy())
  npos = np.asarray(s["contact_position"][0].cpu().numpy())
  ndist = np.asarray(s["contact_distance"][0].cpu().numpy())
  nnrm = np.asarray(s["contact_normal"][0].cpu().numpy())
  slots = [int(x) for x in range(len(mask)) if mask[x] > 0.5]
  assert len(slots) == 2
  # Position-matched slots; press contact is the deeper one on both sides.
  used = set()
  assignment = {}
  for i in slots:
    c = min((x for x in range(d.ncon) if x not in used),
            key=lambda x: float(np.linalg.norm(
                np.asarray(d.contact[x].pos) - npos[i])))
    used.add(c)
    assert float(np.linalg.norm(np.asarray(d.contact[c].pos) - npos[i])) < 1e-3
    assignment[c] = i
  cpu_press = min(range(d.ncon),
                  key=lambda c: float(d.contact[c].dist))
  assert float(d.contact[cpu_press].dist) < -1e-3
  nat_press = assignment[cpu_press]
  # Depths agree tightly; this is not a depth/area tie.
  assert abs(float(ndist[nat_press]) - float(d.contact[cpu_press].dist)) < 1e-9
  cpu_nrm = np.asarray(d.contact[cpu_press].frame[:3], dtype=np.float64)
  cpu_pos = np.asarray(d.contact[cpu_press].pos, dtype=np.float64)
  nat_nrm = np.asarray(nnrm[nat_press], dtype=np.float64)
  nat_pos = np.asarray(npos[nat_press], dtype=np.float64)
  # CPU ground truth at the gate pose (robust over +-1e-3 x shifts).
  np.testing.assert_allclose(
      cpu_nrm, np.array([0.12222085138013072, -0.18293366315697751,
                         -0.97549850762155732]),
      rtol=0, atol=1e-9)
  # DIRECT AGREEMENT (load-bearing: fails as x-mirror without the mat2Quat
  # repair). Press contact matches CPU on the same side, far below the
  # 013 tiers (12e-3 pos, 0.2 nrm) and below float32 publication noise.
  np.testing.assert_allclose(nat_pos, cpu_pos, rtol=0, atol=1e-7,
                             err_msg=f"pos {nat_pos} vs {cpu_pos}")
  np.testing.assert_allclose(nat_nrm, cpu_nrm, rtol=0, atol=1e-7,
                             err_msg=f"nrm {nat_nrm} vs {cpu_nrm}")
  assert abs(float(ndist[nat_press]) - float(d.contact[cpu_press].dist)) < 1e-9
  # Staged seed-1 INTERSECTION terminal: accepted alpha and dist agree
  # (same basin depth); only the transverse sign differs.
  terms = []
  for name, _, state, _, _, _ in ep:
    if "line_search" not in name:
      continue
    st = state.cpu().numpy().reshape(-1, _STATE_WORDS)
    x = _collapse_words(st[1, _STAGE["x"]:][:9])
    terms.append((float(_collapse_wide(st[1, _STAGE["alpha"]:][:3])),
                  float(_collapse_wide(st[1, _STAGE["dist"]:][:3])),
                  [float(v) for v in x]))
  assert terms
  print(f"BOXDIVERGE cpu_press_dist={float(d.contact[cpu_press].dist):.12f} "
        f"cpu_nrm={cpu_nrm.tolist()} nat_nrm={nat_nrm.tolist()} "
        f"cpu_pos={cpu_pos.tolist()} nat_pos={nat_pos.tolist()} "
        f"seed1_terms={terms[-3:]}", flush=True)



def _f64_bits(x):
  """Raw binary64 word with explicit little-endian byte order."""
  import struct
  return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _ordered_key(x):
  """Monotone ulp-order key: finite values map preserving order, with
  -0.0 and +0.0 sharing one key; callers handle nonfinites explicitly."""
  b = _f64_bits(x)
  sign = 0x8000000000000000
  magnitude = b & (sign - 1)
  return sign - magnitude if b & sign else sign + magnitude


def _ulp_distance(a, b):
  """Ordered ulp distance, or None for nonfinite inputs. Returns 0 for
  numerically equal values (including -0.0 vs +0.0)."""
  import math
  fa, fb = float(a), float(b)
  if not (math.isfinite(fa) and math.isfinite(fb)):
    return None
  if fa == fb:
    return 0
  return abs(_ordered_key(fa) - _ordered_key(fb))


def _triple_value(hi, lo, tail):
  """Reconstruct one coordinate from staged (hi, lo, tail) planes with a
  validated exact sum (math.fsum: correctly rounded infinite-precision
  sum), NOT naive left-to-right addition and NOT the hi word alone."""
  import math
  return math.fsum([float(hi), float(lo), float(tail)])


def test_seed_word_ulp_machinery_host():
  # Host unit checks for the comparison machinery (no GPU needed):
  # adjacent values, signs, powers of two, signed zero, subnormals,
  # nonfinites and exact-sum reconstruction.
  import math
  assert _ulp_distance(1.0, np.nextafter(1.0, 2.0)) == 1
  assert _ulp_distance(1.0, 1.0) == 0
  assert _ulp_distance(0.5, 1.0) == 2**52
  assert _ulp_distance(0.0, -0.0) == 0
  assert _f64_bits(-1.0) >> 63 == 1
  assert _f64_bits(1.0) >> 63 == 0
  tiny = np.nextafter(0.0, 1.0)
  assert tiny > 0.0 and _ulp_distance(tiny, 0.0) == 1
  assert _ulp_distance(float("inf"), 1.0) is None
  assert _ulp_distance(float("inf"), float("inf")) is None
  assert _ulp_distance(float("nan"), float("nan")) is None
  assert _ulp_distance(-1.0, np.nextafter(-1.0, -2.0)) == 1
  assert _ordered_key(-2.0) < _ordered_key(-1.0) < _ordered_key(0.0)
  assert _ulp_distance(-tiny, 0.0) == 1
  assert _ulp_distance(-tiny, tiny) == 2
  assert _ordered_key(-0.0) == _ordered_key(0.0)
  # Across zero, distance is the sum of the two positive magnitudes'
  # distances to zero; this is independent of the negative-key transform.
  a = float.fromhex("-0x1.ea999835accffp-63")
  b = float.fromhex("0x1.e3cb80ddeaacap-56")
  assert _ulp_distance(a, b) == _f64_bits(abs(a)) + _f64_bits(abs(b))
  # math.fsum is exactly rounded: catastrophic cancellation case where
  # naive left-to-right addition loses the small word entirely.
  assert math.fsum([1e16, 1.0, -1e16]) == 1.0
  assert float(np.float64(1e16) + np.float64(1.0) + np.float64(-1e16)) == 0.0
  # Triple reconstruction of a real staged seed coordinate.
  assert _triple_value(0.03279993683099747, 9.536744727256519e-10,
                       2.7755575615628914e-17) == math.fsum(
                           [0.03279993683099747, 9.536744727256519e-10,
                            2.7755575615628914e-17])


def _trace_build_paths():
  import os
  root = os.environ.get("MUJOCO_METAL_TRACE_ROOT") or os.path.dirname(
      os.path.dirname(
          os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
  helper = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "trace_sdf.py")
  lib = os.path.join(root, ".upstream", "mujoco-trace", "build-fixed",
                     "lib", "libmujoco.dylib")
  return helper, lib


@_needs_gpu()
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
def test_box_seed_source_indexed_comparison_gpu(tmp_path):
  # Executable same-input, source-indexed CPU/native comparison at the
  # exact press pose (mjc_SDF octree-versus-analytic route, NOT per-face
  # Frank-Wolfe). CPU raw binary64 words come from a live trace-build run
  # (pinned binary truth, not a replay approximation; explicit skip names
  # the missing artifact when the trace build is absent); native raw words
  # come from staged triples and hull constants. Stages, in producer order:
  #   1. input anchoring (hull AABB triple sums vs model f64),
  #   2. seed values per seed/coordinate (raw hex, ulp distances),
  #   3. decisive INTERSECTION gradpick operands/side,
  #   4. precontact admission.
  # Methodology is asserted; magnitudes are reported. The seed-stage ulp
  # gap (stage 2) localizes the earliest observable mismatch; input
  # anchoring (stage 1) holds bit-exactly, so divergence enters in seed
  # arithmetic (bounds/Halton/lerp/maps chain), not in inputs.
  import json
  import os
  import subprocess
  import sys
  sys.path.insert(0, "metal/tests")
  import mujoco
  from test_sdf_seed_source_trace_013 import (
      _capture_staged_workspace, _STAGE, _STATE_WORDS)
  from mujoco_metal.simulation import MetalSimulation
  helper, lib = _trace_build_paths()
  if not os.path.exists(lib):
    pytest.skip("trace build absent: " + lib)
  m = _gate_model()
  sim = MetalSimulation(m, batch_size=1, profile="integrated_euler_v1")
  sim.reset(qpos=_GATE_QP.reshape(1, -1),
            qvel=np.zeros((1, m.nv), dtype=np.float32))
  epochs, _ = _capture_staged_workspace(sim)
  s = sim.assembled_system(recompute=True)
  assert int(np.asarray(sim.state.status.cpu().numpy()).ravel()[0]) == 0
  assert len(epochs) == 1
  # Stage 1: input anchoring bit-exact (hull triple sums == model f64).
  # The low plane alone does not suffice (float32 residual rounding);
  # the tail plane completes the expansion: hi+lo+tail reproduces the
  # pinned entry bitwise (center tail at +0, half tail at +6).
  cc = sim._coupled_constraints
  hull = cc._constants["mesh_hull"].cpu().numpy().astype(np.float64)
  info = cc._constants["mesh_hull_info"].cpu().numpy().astype(np.int64)
  info = info.reshape(-1, 9)
  ga = np.asarray(m.geom_aabb, dtype=np.float64).reshape(-1)
  base = int(info[0, 3])
  tbase = int(info[0, 6])
  assert base >= 0 and tbase >= 0
  for g in (0, 1):
    triple = (hull[base + 18 * g:base + 18 * g + 6]
              + hull[base + 18 * g + 6:base + 18 * g + 12]
              + np.concatenate([hull[tbase + 15 * g:tbase + 15 * g + 3],
                                hull[tbase + 15 * g + 6:tbase + 15 * g + 9]]))
    np.testing.assert_array_equal(triple, ga[6 * g:6 * g + 6])
  # Stage 1b: relative-transform comparison (staged T21 vs pip mapPose).
  # Direction is identified by the offset: it equals o21 bit-exactly
  # (not o12, which differs macroscopically), so T21 is the SDF-frame
  # relative transform. Load-bearing parity: rotation AND offset are
  # bit-exact after the mat2Quat repair (fails with 1.33e-15 tilt when
  # FK quats feed T directly).
  import math as _math
  from test_sdf_seed_source_trace_013 import _pose_map
  d_pose = mujoco.MjData(m)
  d_pose.qpos[:] = np.asarray(_GATE_QP, dtype=np.float64)
  d_pose.qvel[:] = np.zeros(m.nv)
  mujoco.mj_forward(m, d_pose)
  o21, R21 = _pose_map(d_pose, 0, 1)
  o12, R12 = _pose_map(d_pose, 1, 0)
  R21 = np.asarray(R21).reshape(3, 3)
  st0 = epochs[-1][0][2].cpu().numpy().reshape(-1, _STATE_WORDS)
  w0 = st0[1].astype(np.float64)
  nat_R = np.empty((3, 3))
  nat_o = np.empty(3)
  for _c in range(3):
    for _r in range(3):
      _k = 42 + 9 * _c + 3 * _r
      nat_R[_r, _c] = _math.fsum(
          [float(w0[_k]), float(w0[_k + 1]), float(w0[_k + 2])])
    _k = 69 + 3 * _c
    nat_o[_c] = _math.fsum(
        [float(w0[_k]), float(w0[_k + 1]), float(w0[_k + 2])])
  np.testing.assert_array_equal(nat_o, np.asarray(o21))
  # The other direction is macroscopically different, proving the
  # identification above (not a near-miss between close candidates).
  assert float(np.max(np.abs(nat_o - np.asarray(o12)))) > 1e-3
  rot_gap = float(np.max(np.abs(nat_R - R21)))
  assert rot_gap == 0.0, rot_gap
  print(f"TRANSFORM off==o21 exact; rot bit-exact", flush=True)
  # Stage 1c: FK quaternion inputs vs pinned mat2Quat consumption.
  # Documents the repaired input fork: native FK quats preserve model
  # values while the SDF path consumes mju_mat2Quat(FK xmat) quats
  # (previously 1.67e-15 apart, now routed correctly for T).
  fk = sim._smooth._fk
  fws = fk._workspace["outputs"]
  qh = np.asarray(fws["geom_quat"].cpu().numpy(), dtype=np.float64)
  ql = np.asarray(fws["geom_quat_low"].cpu().numpy(), dtype=np.float64)
  qt = np.asarray(fws["geom_quat_tail"].cpu().numpy(), dtype=np.float64)
  for _g in (0, 1):
    nq = [_math.fsum([float(qh[4 * _g + i]), float(ql[4 * _g + i]),
                      float(qt[4 * _g + i])]) for i in range(4)]
    cq = np.zeros(4)
    mujoco.mju_mat2Quat(
        cq, np.asarray(d_pose.geom_xmat).reshape(-1)[9 * _g:9 * _g + 9])
    print(f"QUAT g{_g} maxabsdiff="
          f"{float(np.max(np.abs(np.asarray(nq) - cq))):.2e}", flush=True)
  # Live CPU trace at gate-baked XML.
  zf = repr(float(np.float32(float(_GATE_QP[2]))))
  xml_text = f"""<mujoco>
<asset><mesh name="b" vertex="-0.05 -0.05 -0.04  0.05 -0.05 -0.04 0.05 0.05 -0.04  -0.05 0.05 -0.04  -0.05 -0.05 0.04  0.05 -0.05 0.04  0.05 0.05 0.04  -0.05 0.05 0.04"/></asset>
<option timestep="0.002" integrator="Euler" iterations="100" tolerance="1e-8" gravity="0 0 0" sdf_initpoints="4"/>
<worldbody>
<geom name="sdf" type="sdf" mesh="b" contype="1" conaffinity="1"/>
<body pos="0 0 {zf}"><freejoint/><geom name="o" type="box" size="0.04 0.04 0.03" contype="1" conaffinity="1"/></body>
</worldbody>
</mujoco>"""
  p = tmp_path / "box_gate.xml"
  p.write_text(xml_text)
  env = dict(os.environ, MUJOCO_SDF_TRACE="1")
  proc = subprocess.run(
      [sys.executable, helper, str(p)], capture_output=True, text=True,
      env=env, timeout=120)
  assert proc.returncode == 0, proc.stderr[-2000:]
  numbers = json.loads(proc.stdout)
  assert numbers["ncon"] == 2
  tlines = [ln for ln in proc.stderr.splitlines() if "MJCSDFTRACE" in ln]
  cpu_seeds = {}
  for ln in tlines:
    if "MJCSDFTRACE seed" in ln:
      i = int(ln.split("i=")[1].split()[0])
      cpu_seeds[i] = tuple(float(v) for v in ln.split("x=")[1].split())
  assert sorted(cpu_seeds) == [0, 1, 2, 3]
  # Stage 2: raw-word seed comparison with full triple reconstruction
  # (math.fsum over hi/lo/tail planes, NOT the float32 hi word alone)
  # and explicit-byte-order ordered ulp distances. Admitted parity
  # contract after the mat2Quat repair: seeds agree to 1e-12 absolute
  # (9/12 bit-exact; remaining 1-ulp dot noise); the load-bearing gates
  # are T21 bit-exactness above and direct contact agreement below.
  # Fails without the repair via the contact gate (mirror), not via a
  # bogus ulp threshold.
  st = epochs[-1][0][2].cpu().numpy().reshape(-1, _STATE_WORDS)
  max_ulp, loc, detail = 0, None, ""
  max_abs = 0.0
  for seed in range(4):
    for axis in range(3):
      words = [float(st[seed, axis]), float(st[seed, 3 + axis]),
               float(st[seed, 6 + axis])]
      assert all(np.isfinite(words))
      a = _triple_value(*words)
      b = float(cpu_seeds[seed][axis])
      gap = _ulp_distance(a, b)
      assert gap is not None
      max_abs = max(max_abs, abs(a - b))
      if gap > max_ulp:
        max_ulp, loc = gap, (seed, axis)
        detail = (f"seed={seed} axis={axis} "
                  f"hi={words[0].hex()} lo={words[1].hex()} "
                  f"tail={words[2].hex()} fsum={a.hex()} cpu={b.hex()}")
  print(f"SEEDCOMPARE max_ulp={max_ulp} max_abs={max_abs:.3e} at={loc} {detail}",
        flush=True)
  assert max_abs < 1e-12, (max_abs, loc, detail)
  # Stage 3: press-seed INTERSECTION pick is decisive on the CPU side
  # (analytic side wins by a margin no rounding can flip).
  picks = [ln for ln in tlines if "gradpick" in ln]
  assert len(picks) == 4
  a = float(picks[1].split("a=")[1].split()[0])
  b = float(picks[1].split("b=")[1].split()[0])
  side = int(picks[1].split("side=")[1].split()[0])
  assert side == 1 and (a - b) < -1e-3, (a, b, side)
  # Stage 4: two admissions, no spurious rejects at this pose.
  decisions = [ln.split("decision=")[1].split()[0]
               for ln in tlines if "precontact" in ln]
  assert decisions.count("admit") == 2
