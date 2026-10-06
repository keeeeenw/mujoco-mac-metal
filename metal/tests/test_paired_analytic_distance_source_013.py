"""Execute paired analytic distance against the pinned, compiled C formula.

The C fixture retains upstream function bodies. This isolates helper arithmetic;
full public contacts, force and lifecycle are separate qualification gates.
"""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
import pytest


@pytest.fixture(scope="module")
def source_distance(tmp_path_factory):
  compiler = shutil.which("clang")
  if compiler is None:
    pytest.skip("C compiler required for pinned distance reference")
  package = Path(mujoco.__file__).resolve().parent
  libraries = list(package.glob("libmujoco*.dylib")) or list(package.glob("libmujoco*.so*"))
  assert libraries, "installed MuJoCo library unavailable"
  source = Path(__file__).parent / "reference/sdf_analytic_distance_3_10.c"
  output = tmp_path_factory.mktemp("analytic-distance") / "source.dylib"
  subprocess.run([compiler, "-shared", "-fPIC", "-O0", "-fno-fast-math",
                  "-I", str(package / "include"), str(source),
                  str(libraries[0]), "-Wl,-rpath," + str(package),
                  "-o", str(output)], check=True, capture_output=True, text=True)
  library = ctypes.CDLL(str(output))
  ptr = ctypes.POINTER(ctypes.c_double)
  function = library.source_analytic_distance
  function.argtypes = [ctypes.c_int, ptr, ptr]
  function.restype = ctypes.c_double

  def evaluate(kind, size, point):
    size = np.asarray(size, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    return function(kind, size.ctypes.data_as(ptr), point.ctypes.data_as(ptr))
  return evaluate


def _cases():
  rng = np.random.default_rng(2941)
  cases = []
  for name, native_id in (("plane", 1), ("sphere", 2), ("capsule", 3),
                           ("ellipsoid", 4), ("cylinder", 5), ("box", 6)):
    cpu_id = int(getattr(mujoco.mjtGeom, "mjGEOM_" + name.upper()))
    for index in range(64):
      size64 = rng.uniform(.05, .7, 3)
      size_high = size64.astype(np.float32)
      size_low = (size64 - size_high.astype(np.float64)).astype(np.float32)
      size = size_high.astype(np.float64) + size_low.astype(np.float64)
      point = rng.uniform(-.8, .8, 3)
      if index < 32:
        point = rng.uniform(-.8, .8, 3) * size.astype(np.float64)
      high = point.astype(np.float32)
      low = (point - high.astype(np.float64)).astype(np.float32)
      represented = high.astype(np.float64) + low.astype(np.float64)
      cases.append((name, native_id, cpu_id, size, high, low, represented))
  return cases


def test_compiled_box_distance_uses_normalized_radial_field(source_distance):
  size = np.asarray([.5, .3, .2], dtype=np.float64)
  point = np.asarray([.08, -.04, .03], dtype=np.float64)
  gap = np.abs(point) - size
  raw = -size / gap
  field = raw / np.linalg.norm(raw)
  field *= np.where(point < 0, -1, 1)
  expected = -np.min(-gap / np.abs(field)) * np.linalg.norm(field)
  actual = source_distance(int(mujoco.mjtGeom.mjGEOM_BOX), size, point)
  np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-16)
  assert abs(actual - expected * np.linalg.norm(raw)) > .1


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native paired analytic distance")
def test_native_paired_analytic_distances_match_pinned_c(source_distance):
  import torch
  shader = Path(__file__).parents[1] / "mujoco_metal/shaders"
  source = "\n".join((shader / name).read_text() for name in
      ("collision_primitives.metal", "convex_narrowphase.metal",
       "plugin_sdf.metal", "sdf_narrowphase.metal"))
  source += r"""
kernel void analytic_distance_source_probe(
    device const int* kinds [[buffer(0)]], device const float* sizes [[buffer(1)]],
    device const float* points [[buffer(2)]], device float* output [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  SdfPointPair point = {float3(points[6*i], points[6*i+1], points[6*i+2]),
                       float3(points[6*i+3], points[6*i+4], points[6*i+5])};
  float3 size(sizes[6*i], sizes[6*i+1], sizes[6*i+2]);
  float3 size_low(sizes[6*i+3], sizes[6*i+4], sizes[6*i+5]);
  SdfJet distance = sdf_analytic_pair_jet(kinds[i], size, point, size_low);
  output[2*i] = distance.v; output[2*i+1] = distance.e;
}
"""
  library = torch.mps.compile_shader(source)
  cases = _cases()
  expected = np.asarray([source_distance(c[2], c[3], c[6]) for c in cases])
  kinds = torch.tensor([c[1] for c in cases], dtype=torch.int32, device="mps")
  size_high = np.stack([c[3] for c in cases]).astype(np.float32)
  size_low = (np.stack([c[3] for c in cases]) - size_high.astype(np.float64)).astype(np.float32)
  assert np.any(size_low != 0)
  sizes = torch.tensor(np.concatenate([size_high, size_low], axis=1), device="mps")
  points = torch.tensor(np.stack([np.r_[c[4], c[5]] for c in cases]), device="mps")
  output = torch.empty((len(cases), 2), dtype=torch.float32, device="mps")
  library.analytic_distance_source_probe(kinds, sizes, points, output,
                                       threads=(len(cases),), group_size=(64,))
  pairs = output.cpu().numpy().astype(np.float64)
  actual = pairs.sum(axis=1)
  np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
