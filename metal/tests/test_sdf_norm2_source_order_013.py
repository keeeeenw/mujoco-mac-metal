# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Source-order witness for the shared rigid-SDF two-term norm helper."""

import ctypes
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from test_bundled_sdf_rigid_oracle_019 import _plugin_sphere_model
from test_flex_sdf_gradient_fma_018 import _runtime_library
from test_sdf_arithmetic_native_witness import _exact, _words
from test_sdf_seed_source_trace_013 import (
    _plugin_callbacks, _source_pair_replay)


# These fixtures leave plugin attributes unspecified. Keep the pinned 3.10
# defaults from plugin/sdf/{bowl,torus}.h explicit in this witness.
_BOWL_HEIGHT, _BOWL_RADIUS, _BOWL_THICKNESS = 0.4, 1.0, 0.02
_TORUS_RADIUS1, _TORUS_RADIUS2 = 0.35, 0.15
# First native bowl mismatch from native-sdf-source-0bc5.log. Keep this exact
# source point even if future fixture trajectories stop reaching it.
_SQRT_REGRESSION = (float.fromhex("0x1.55e85d08b96e0p-4"),
                    float.fromhex("-0x1.99a201d7d5cd5p-2"))


_KERNEL = r"""
kernel void sdf_norm2_source_order_witness(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= uint(count)) return;
  uint ib = index * 6, ob = index * 9;
  SdfWide x = {input[ib], input[ib + 1], input[ib + 2]};
  SdfWide y = {input[ib + 3], input[ib + 4], input[ib + 5]};
  SdfWide source = sw_norm2(x, y);
  SdfWide unfused = sw_sqrt(sw_add(sw_mul(x, x), sw_mul(y, y)));
  SdfWide source_squared = sw_source_fma(x, x, sw_mul(y, y));
  output[ob] = source.hi;
  output[ob + 1] = source.mid;
  output[ob + 2] = source.lo;
  output[ob + 3] = unfused.hi;
  output[ob + 4] = unfused.mid;
  output[ob + 5] = unfused.lo;
  output[ob + 6] = source_squared.hi;
  output[ob + 7] = source_squared.mid;
  output[ob + 8] = source_squared.lo;
}
"""


def _source_norm_cases():
  """Collect real pinned descent/finite-difference points for both plugins."""
  cases = {"bowl": [], "torus": []}
  for name in cases:
    model = _plugin_sphere_model(name)
    data = mujoco.MjData(model)
    # Native fixtures use the public float32 state representation. Replay from
    # those represented qpos words so the sampled source points share it.
    data.qpos[:] = model.qpos0.astype(np.float32).astype(np.float64)
    mujoco.mj_forward(model, data)
    sdf_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
        int(mujoco.mjtGeom.mjGEOM_SDF))[0])
    sphere_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
        int(mujoco.mjtGeom.mjGEOM_SPHERE))[0])
    trace = _source_pair_replay(model, data, sdf_geom, sphere_geom)
    for seed in trace:
      points = [seed["x_init"]]
      for mode in ("collision", "intersection"):
        for step in seed[mode]:
          points.extend((step["x0"], step["x"]))
      for point in points:
        if name == "bowl":
          # Bowl::Gradient samples all three positive one-sided perturbations.
          for axis in range(3):
            shifted = point.copy()
            shifted[axis] += 1e-8
            for sample in (point, shifted):
              qx = mujoco.mju_norm(np.asarray(sample[:2], dtype=np.float64))
              cases[name].append((float(sample[0]), float(sample[1])))
              cases[name].append((float(qx), float(sample[2])))
              width = math.sqrt(_BOWL_RADIUS * _BOWL_RADIUS -
                                _BOWL_HEIGHT * _BOWL_HEIGHT)
              cases[name].append((float(qx - width),
                                  float(sample[2] - _BOWL_HEIGHT)))
        else:
          # Torus::Distance and ::Gradient use these two nested 2D norms.
          lenxy = mujoco.mju_norm(np.asarray(point[:2], dtype=np.float64))
          q = float(lenxy - _TORUS_RADIUS1)
          cases[name].append((float(point[0]), float(point[1])))
          cases[name].append((q, float(point[2])))
  cases["bowl"].append(_SQRT_REGRESSION)
  return cases


def _runtime_norm_and_fma():
  library = _runtime_library()
  norm = library.mju_norm
  norm.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
  norm.restype = ctypes.c_double
  fma = ctypes.CDLL(None).fma
  fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
  fma.restype = ctypes.c_double
  return norm, fma


def _norm2(norm, x, y):
  vector = (ctypes.c_double * 2)(float(x), float(y))
  return float(norm(vector, 2))


def _shader_source():
  return (Path(__file__).parents[1] /
          "mujoco_metal/shaders/plugin_sdf.metal").read_text()


def test_cpu_runtime_norm2_matches_pinned_bowl_and_torus_points():
  norm, fma = _runtime_norm_and_fma()
  cases = _source_norm_cases()
  for name, points in cases.items():
    assert len(points) >= 8
    negative_control_differences = 0
    for x, y in points:
      source = _norm2(norm, x, y)
      # This checks the installed mju_norm primitive itself. Torus::Distance
      # spells its norm inline; its actual callback is checked separately.
      source_order = math.sqrt(fma(x, x, y * y))
      assert source == source_order, (name, x.hex(), y.hex(),
                                      source.hex(), source_order.hex())
      unfused = math.sqrt(x * x + y * y)
      negative_control_differences += source != unfused
    assert negative_control_differences > 0, (
        name, "unfused control did not diverge")


def test_installed_plugin_distance_callbacks_match_pinned_source_formulas():
  norm, fma = _runtime_norm_and_fma()
  for name in ("bowl", "torus"):
    model = _plugin_sphere_model(name)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0.astype(np.float32).astype(np.float64)
    mujoco.mj_forward(model, data)
    sdf_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
        int(mujoco.mjtGeom.mjGEOM_SDF))[0])
    sphere_geom = int(np.flatnonzero(np.asarray(model.geom_type) ==
        int(mujoco.mjtGeom.mjGEOM_SPHERE))[0])
    distance, _ = _plugin_callbacks(model, data, sdf_geom)
    trace = _source_pair_replay(model, data, sdf_geom, sphere_geom)
    points = []
    for seed in trace:
      points.append(seed["x_init"])
      for mode in ("collision", "intersection"):
        for step in seed[mode]:
          points.extend((step["x0"], step["x"]))
    assert points
    fused_matches = unfused_matches = 0
    for point in points:
      x, y, z = (float(v) for v in point)
      if name == "bowl":
        # Bowl::Distance uses mju_norm for both norms and the source branch.
        width = math.sqrt(_BOWL_RADIUS * _BOWL_RADIUS -
                          _BOWL_HEIGHT * _BOWL_HEIGHT)
        qx = _norm2(norm, x, y)
        qz = z
        if _BOWL_HEIGHT * qx < width * qz:
          expected = (_norm2(norm, qx - width,
                             qz - _BOWL_HEIGHT) - _BOWL_THICKNESS)
        else:
          expected = (abs(_norm2(norm, qx, qz) - _BOWL_RADIUS)
                      - _BOWL_THICKNESS)
        assert distance(point) == expected, (name, point, distance(point), expected)
      else:
        # Torus::Distance has two inline sqrt(a*a + b*b) expressions. Test
        # both possible contraction outcomes against the actual plugin ABI.
        radial_sq_fused = fma(x, x, y * y)
        radial_sq_unfused = x * x + y * y
        radial_fused = math.sqrt(radial_sq_fused)
        radial_unfused = math.sqrt(radial_sq_unfused)
        q_fused = radial_fused - _TORUS_RADIUS1
        q_unfused = radial_unfused - _TORUS_RADIUS1
        full_fused = math.sqrt(fma(q_fused, q_fused, z * z)) - _TORUS_RADIUS2
        full_unfused = (math.sqrt(q_unfused * q_unfused + z * z)
                        - _TORUS_RADIUS2)
        actual = distance(point)
        fused_matches += actual == full_fused
        unfused_matches += actual == full_unfused
    if name == "torus":
      assert fused_matches == len(points), (
          "torus callback differs from source-order fused formula",
          fused_matches, len(points))
      assert unfused_matches < len(points), (
          "unfused control unexpectedly reproduced callback",
          unfused_matches, len(points))


def test_rigid_sdf_norm2_uses_pinned_fma_order():
  source = _shader_source()
  start = source.index("inline SdfWide sw_norm2(SdfWide x, SdfWide y) {")
  end = source.index("\n}", start) + 2
  helper = source[start:end]
  assert "sw_sqrt(sw_source_fma(x, x, sw_mul(y, y)))" in helper
  assert "sw_sqrt(sw_add(sw_mul(x, x), sw_mul(y, y)))" not in helper


def test_sqrt_rounds_the_same_approximation_used_by_second_residual():
  source = _shader_source()
  start = source.index("[[clang::noinline]] inline SdfWide sw_sqrt_core(")
  end = source.index("\n}", start) + 2
  helper = source[start:end]
  assert "SdfWide approximation=sw_source_add(q0_value,sw(q1));" in helper
  assert "sw_product_residual(a,approximation,approximation)" in helper
  assert "sw_expansion_add(expansion,count,approximation.hi,scratch)" in helper
  assert "sw_expansion_add(expansion,count,q0,scratch)" not in helper
  assert "sw_expansion_add(expansion,count,q1,scratch)" not in helper


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="parent-dispatched native norm2 helper witness")
def test_native_norm2_helper_matches_pinned_runtime_output_words():
  import torch

  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("MPS compile_shader is unavailable")
  norm, fma = _runtime_norm_and_fma()
  source = _shader_source()
  library = torch.mps.compile_shader(source + "\n" + _KERNEL)
  cases = _source_norm_cases()
  failures = []
  for name, points in cases.items():
    host = np.asarray([word for x, y in points
                       for word in (*_words(x), *_words(y))], dtype=np.float32)
    input_device = torch.as_tensor(host.copy(), dtype=torch.float32,
                                   device="mps").contiguous()
    output = torch.empty((len(points), 9), dtype=torch.float32, device="mps")
    count = torch.tensor([len(points)], dtype=torch.int32, device="mps")
    library.sdf_norm2_source_order_witness(
        input_device, output.reshape(-1), count,
        threads=(len(points),), group_size=(1,))
    actual = output.cpu().numpy()
    negative_control_differences = 0
    for index, (x, y) in enumerate(points):
      expected = _norm2(norm, x, y)
      source_words = tuple(float(v) for v in actual[index, :3])
      unfused_words = tuple(float(v) for v in actual[index, 3:6])
      squared_words = tuple(float(v) for v in actual[index, 6:9])
      if _exact(source_words) != _exact(_words(expected)):
        failures.append((name, index, x.hex(), y.hex(), "source", source_words,
                         _words(expected), _exact(source_words), expected.hex()))
      expected_squared = fma(x, x, y * y)
      if _exact(squared_words) != _exact(_words(expected_squared)):
        failures.append((name, index, x.hex(), y.hex(), "squared norm",
                         squared_words, _words(expected_squared)))
      unfused_expected = math.sqrt(x * x + y * y)
      if _exact(unfused_words) != _exact(_words(unfused_expected)):
        failures.append((name, index, x.hex(), y.hex(), "unfused control",
                         unfused_words, _words(unfused_expected)))
      negative_control_differences += (
          _exact(unfused_words) != _exact(_words(expected)))
    if negative_control_differences == 0:
      failures.append((name, "unfused native control did not diverge"))
  assert not failures, "native SDF norm2 source mismatches: " + repr(failures[:12])
