# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned arm64 FMA witness for the plugin-free flex SDF gradient."""
from __future__ import annotations

import ctypes
import hashlib
import platform
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest


_RUNTIME_SHA256 = (
    "9d0f4b7fd202d51f6a126912ae913b99ca6b7ee8f901887e897525041c0a79dd")


def _axis_fixture():
  xml = """<mujoco><asset>
    <mesh name="sdfbox" vertex="-.05 -.05 -.05 .05 -.05 -.05
        .05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05
        .05 .05 .05 -.05 .05 .05"/>
    </asset><option gravity="0 0 0" sdf_initpoints="4"/>
    <worldbody>
      <geom name="sdf" type="sdf" mesh="sdfbox"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 .04"
                quat="1 0 0 0" spacing=".04 .04 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _runtime_library():
  package = Path(mujoco.__file__).resolve().parent
  libraries = sorted(package.glob("libmujoco.3.10.0.dylib"))
  assert len(libraries) == 1
  path = libraries[0]
  assert hashlib.sha256(path.read_bytes()).hexdigest() == _RUNTIME_SHA256
  return ctypes.CDLL(str(path))


def _find_leaf_and_derivatives(model, octadr: int, point: np.ndarray):
  """Replay pinned findOct at one interior point, preserving source order."""
  node = 0
  for _ in range(100):
    box = np.asarray(model.oct_aabb[octadr + node], np.float64)
    lo = box[:3] - box[3:]
    hi = box[:3] + box[3:]
    assert all(lo[k] - 1.0e-8 <= point[k] <= hi[k] + 1.0e-8
               for k in range(3))
    coord = np.asarray([
        (point[k] - lo[k]) / (hi[k] - lo[k]) for k in range(3)], np.float64)
    children = np.asarray(model.oct_child[octadr + node], np.int32)
    if np.all(children == -1):
      break
    child = (4 * int(coord[2] >= 0.5) + 2 * int(coord[1] >= 0.5)
             + int(coord[0] >= 0.5))
    node = int(children[child])
  else:
    raise AssertionError("pinned source findOct did not reach a leaf")

  derivatives = np.empty((8, 3), np.float64)
  for j in range(8):
    derivatives[j, 0] = (
        (1.0 if j & 1 else -1.0)
        * (coord[1] if j & 2 else 1.0 - coord[1])
        * (coord[2] if j & 4 else 1.0 - coord[2]))
    derivatives[j, 1] = (
        (coord[0] if j & 1 else 1.0 - coord[0])
        * (1.0 if j & 2 else -1.0)
        * (coord[2] if j & 4 else 1.0 - coord[2]))
    derivatives[j, 2] = (
        (coord[0] if j & 1 else 1.0 - coord[0])
        * (coord[1] if j & 2 else 1.0 - coord[1])
        * (1.0 if j & 4 else -1.0))
  return node, lo, hi, coord, derivatives


def _hex_vector(values):
  return tuple(float(value).hex() for value in values)


def _gradient_with_runtime(model, data, geom_id, point):
  """Call installed pinned mjc_gradient for a single-SDF descriptor."""
  library = _runtime_library()

  class MjSDF(ctypes.Structure):
    _fields_ = [
        ("plugin", ctypes.c_void_p),
        ("id", ctypes.POINTER(ctypes.c_int)),
        ("type", ctypes.c_int),
        ("relpos", ctypes.POINTER(ctypes.c_double)),
        ("relmat", ctypes.POINTER(ctypes.c_double)),
        ("geomtype", ctypes.POINTER(ctypes.c_int)),
    ]

  geom_data_id = ctypes.c_int(int(model.geom_dataid[geom_id]))
  plugin = ctypes.c_void_p(None)
  relpos = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
  relmat = (ctypes.c_double * 9)(*([0.0] * 9))
  geomtype = ctypes.c_int(int(mujoco.mjtGeom.mjGEOM_SDF))
  sdf = MjSDF(
      ctypes.cast(ctypes.pointer(plugin), ctypes.c_void_p),
      ctypes.pointer(geom_data_id),
      int(mujoco.mjtSDFType.mjSDFTYPE_SINGLE),
      relpos,
      relmat,
      ctypes.pointer(geomtype))

  gradient = (ctypes.c_double * 3)()
  point_words = (ctypes.c_double * 3)(*map(float, point))
  source_gradient = library.mjc_gradient
  source_gradient.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(MjSDF),
      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
  source_gradient.restype = None
  source_gradient(
      ctypes.c_void_p(model._address), ctypes.c_void_p(data._address),
      ctypes.byref(sdf), gradient, point_words)
  return np.asarray(gradient, np.float64), library


def _replay_fw_seed(library, model, data, sdf, corners, seed, source_order):
  """Replay one pinned direct-route seed through exported runtime leaves."""
  d3 = ctypes.c_double * 3
  halton = library.mju_Halton
  halton.argtypes = [ctypes.c_int, ctypes.c_int]
  halton.restype = ctypes.c_double
  gradient = library.mjc_gradient
  gradient.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(type(sdf)),
      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
  gradient.restype = None
  dot3 = library.mju_dot3
  dot3.argtypes = [ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double)]
  dot3.restype = ctypes.c_double
  sub_from3 = library.mju_subFrom3
  sub_from3.argtypes = [ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double)]
  sub_from3.restype = None
  add_to_scl3 = library.mju_addToScl3
  add_to_scl3.argtypes = [ctypes.POINTER(ctypes.c_double),
                          ctypes.POINTER(ctypes.c_double), ctypes.c_double]
  add_to_scl3.restype = None
  distance = library.mjc_distance
  distance.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(type(sdf)),
      ctypes.POINTER(ctypes.c_double)]
  distance.restype = ctypes.c_double
  source_fma = ctypes.CDLL(None).fma
  source_fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
  source_fma.restype = ctypes.c_double

  u = halton(seed + 1, 2)
  v = halton(seed + 1, 3)
  if u + v > 1.0:
    u, v = 1.0 - u, 1.0 - v
  b0 = 1.0 - u - v
  if source_order:
    # _processSdfCorners at arm64 0x37a60..0x37a8c does fmul(c1,u),
    # fmla(c0,b0), fmla(c2,v). This mirrors pinned binary 9d0f4b7f...
    # (the exact SHA is checked by _runtime_library).
    point = np.asarray([
        source_fma(b0, corners[0, axis], u * corners[1, axis])
        for axis in range(3)], np.float64)
  else:
    # Negative control: the previous shader order rounded b0*c0 first, then
    # fused u*c1 and v*c2. It must recreate the two spurious direct contacts.
    point = np.asarray([
        source_fma(u, corners[1, axis], b0 * corners[0, axis])
        for axis in range(3)], np.float64)
  point = np.asarray([
      source_fma(v, corners[2, axis], point[axis])
      for axis in range(3)], np.float64)
  start = point.copy()

  for step in range(int(model.opt.sdf_iterations)):
    grad = (ctypes.c_double * 3)()
    point_words = d3(*map(float, point))
    gradient(ctypes.c_void_p(model._address), ctypes.c_void_p(data._address),
             ctypes.byref(sdf), grad, point_words)
    scores = [dot3(d3(*map(float, corner)), grad) for corner in corners]
    best = 0
    for i in range(1, 3):
      if scores[i] < scores[best]:
        best = i
    selected = d3(*map(float, corners[best]))
    sub_from3(selected, point_words)
    add_to_scl3(point_words, selected, 2.0 / (step + 2.0))
    point = np.asarray(point_words, np.float64)

  point_words = d3(*map(float, point))
  depth = distance(ctypes.c_void_p(model._address), ctypes.c_void_p(data._address),
                   ctypes.byref(sdf), point_words)
  return start, point, depth


def _source_deduplicate(library, candidates):
  """Apply pinned isknown's strict mju_dist3 < mjMINVAL rule."""
  dist3 = library.mju_dist3
  dist3.argtypes = [ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double)]
  dist3.restype = ctypes.c_double
  accepted = []
  for point, depth in candidates:
    if depth >= 0.0:
      continue
    if any(dist3((ctypes.c_double * 3)(*map(float, point)),
                 (ctypes.c_double * 3)(*map(float, prior))) < 1.0e-15
           for prior, _ in accepted):
      continue
    accepted.append((point, depth))
  return accepted


@pytest.mark.skipif(sys.platform != "darwin" or platform.machine() != "arm64",
                    reason="requires the pinned arm64 MuJoCo 3.10 oracle")
def test_axis_flex_sdf_gradient_uses_pinned_per_coefficient_fma():
  assert mujoco.__version__ == "3.10.0"
  model, data = _axis_fixture()
  # Exercise the direct route and obtain its installed-CPU contact oracle.
  model.flex_bvhadr[0] = -1
  mujoco.mj_forward(model, data)
  geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sdf"))
  octadr = int(model.mesh_octadr[int(model.geom_dataid[geom])])

  elem_adr = int(model.flex_elemdataadr[0])
  triangle_nodes = np.asarray(model.flex_elem[elem_adr:elem_adr + 3], np.int32)
  np.testing.assert_array_equal(triangle_nodes, [0, 2, 3])
  # FW step 1 starts at the first triangle corner selected on step 0.
  point = np.asarray(data.flexvert_xpos[triangle_nodes[0]], np.float64)
  assert _hex_vector(point) == (
      "-0x1.47ae147ae147bp-6", "-0x1.47ae147ae147bp-6",
      "0x1.47ae147ae147bp-5")

  leaf, _, _, coord, derivatives = _find_leaf_and_derivatives(
      model, octadr, point)
  assert leaf == 62952
  assert _hex_vector(coord) == (
      "0x1.745d22e8ba2cfp-4", "0x1.745d22e8ba2cfp-4",
      "0x1.a2e8b745d174dp-1")
  coeff = np.asarray(model.oct_coeff[octadr + leaf], np.float64)
  assert _hex_vector(coeff) == (
      "-0x1.0000004000000p-6", "-0x1.0000004000000p-6",
      "-0x1.0000004000000p-6", "-0x1.0000004000000p-6",
      "-0x1.1eb8523333330p-7", "-0x1.1eb8523333330p-7",
      "-0x1.1eb8523333330p-7", "-0x1.1eb8523333330p-7")
  expected_dw = (
      ("-0x1.528335f34e3e0p-3", "-0x1.528335f34e3e0p-3",
       "-0x1.a723f4e47ef2ap-1"),
      ("0x1.528335f34e3e0p-3", "-0x1.0ecf67ab5f762p-6",
       "-0x1.528335f34e3e2p-4"),
      ("-0x1.0ecf67ab5f762p-6", "0x1.528335f34e3e0p-3",
       "-0x1.528335f34e3e2p-4"),
      ("0x1.0ecf67ab5f762p-6", "0x1.0ecf67ab5f762p-6",
       "-0x1.0ecf67ab5f764p-7"),
      ("-0x1.7cd38e26152aep-1", "-0x1.7cd38e26152aep-1",
       "0x1.a723f4e47ef2ap-1"),
      ("0x1.7cd38e26152aep-1", "-0x1.30a948fde24f7p-4",
       "0x1.528335f34e3e2p-4"),
      ("-0x1.30a948fde24f7p-4", "0x1.7cd38e26152aep-1",
       "0x1.528335f34e3e2p-4"),
      ("0x1.30a948fde24f7p-4", "0x1.30a948fde24f7p-4",
       "0x1.0ecf67ab5f764p-7"),
  )
  assert tuple(_hex_vector(row) for row in derivatives) == expected_dw

  library = ctypes.CDLL(None)
  source_fma = library.fma
  source_fma.argtypes = (
      ctypes.c_double, ctypes.c_double, ctypes.c_double)
  source_fma.restype = ctypes.c_double
  separate = np.zeros(3, np.float64)
  fused = np.zeros(3, np.float64)
  for i in range(8):
    for axis in range(3):
      separate[axis] += derivatives[i, axis] * coeff[i]
      fused[axis] = source_fma(
          derivatives[i, axis], coeff[i], fused[axis])

  runtime_gradient, mujoco_library = _gradient_with_runtime(
      model, data, geom, point)
  assert _hex_vector(separate) == (
      "0x0.0p+0", "-0x1.8000000000000p-62",
      "0x1.c28f5c999999fp-8")
  expected_gradient = (
      "-0x1.047132e9c91a8p-62", "-0x1.42389974e48d4p-61",
      "0x1.c28f5c999999fp-8")
  assert _hex_vector(fused) == expected_gradient
  assert _hex_vector(runtime_gradient) == expected_gradient

  dot3 = mujoco_library.mju_dot3
  dot3.argtypes = [ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double)]
  dot3.restype = ctypes.c_double
  triangle = np.asarray(data.flexvert_xpos[triangle_nodes], np.float64)

  def corner_scores(gradient):
    words = (ctypes.c_double * 3)(*map(float, gradient))
    return tuple(dot3((ctypes.c_double * 3)(*map(float, corner)), words)
                 for corner in triangle)

  separate_scores = corner_scores(separate)
  fused_scores = corner_scores(fused)
  assert _hex_vector(separate_scores) == (
      "0x1.205bc0624dd33p-12", "0x1.205bc0624dd33p-12",
      "0x1.205bc0624dd33p-12")
  assert _hex_vector(fused_scores) == (
      "0x1.205bc0624dd33p-12", "0x1.205bc0624dd33p-12",
      "0x1.205bc0624dd32p-12")
  # Pinned strict '<' retains corner 0 on the separate-rounding negative
  # control, but selects corner 2 after the source FMA gradient accumulation.
  strict_best = lambda scores: next(
      (i for i in range(1, 3) if scores[i] < scores[0]), 0)
  assert strict_best(separate_scores) == 0
  assert strict_best(fused_scores) == 2

  # Replay all four Halton starts on both triangles through the exact pinned
  # arm64 runtime leaves. The fixture's direct route has one contact on elem0
  # and two on elem1; the prior initial-term order instead creates two extra
  # per-element attractors despite identical gradient/FW/dedup code.
  class MjSDF(ctypes.Structure):
    _fields_ = [
        ("plugin", ctypes.c_void_p),
        ("id", ctypes.POINTER(ctypes.c_int)),
        ("type", ctypes.c_int),
        ("relpos", ctypes.POINTER(ctypes.c_double)),
        ("relmat", ctypes.POINTER(ctypes.c_double)),
        ("geomtype", ctypes.POINTER(ctypes.c_int)),
    ]

  geom_data_id = ctypes.c_int(int(model.geom_dataid[geom]))
  plugin = ctypes.c_void_p(None)
  relpos = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
  relmat = (ctypes.c_double * 9)(*([0.0] * 9))
  geomtype = ctypes.c_int(int(mujoco.mjtGeom.mjGEOM_SDF))
  sdf = MjSDF(
      ctypes.cast(ctypes.pointer(plugin), ctypes.c_void_p),
      ctypes.pointer(geom_data_id), int(mujoco.mjtSDFType.mjSDFTYPE_SINGLE),
      relpos, relmat, ctypes.pointer(geomtype))
  runtime = _runtime_library()
  expected_source_hex = (
      ("-0x1.2fd94630c7963p-6", "-0x1.2fd94630c7963p-6",
       "0x1.47ae147ae147bp-5"),
      ("-0x1.2fd94630c7963p-6", "-0x1.2fd94630c7963p-6",
       "0x1.47ae147ae147bp-5"),
      ("-0x1.2fd94630c7963p-6", "-0x1.2fd94630c7963p-6",
       "0x1.47ae147ae147bp-5"),
      ("-0x1.2fd94630c7963p-6", "-0x1.2fd94630c7963p-6",
       "0x1.47ae147ae147bp-5"),
  )
  expected_source_hex_elem1 = (
      ("-0x1.2fd94630c7963p-6", "-0x1.d0b5b6a4f503fp-7",
       "0x1.47ae147ae147bp-5"),
      ("-0x1.2fd94630c7963p-6", "-0x1.d0b5b6a4f503fp-7",
       "0x1.47ae147ae147bp-5"),
      ("-0x1.d0b5b6a4f503ep-7", "-0x1.d0b5b6a4f503ep-7",
       "0x1.47ae147ae147bp-5"),
      ("-0x1.2fd94630c7963p-6", "-0x1.d0b5b6a4f503fp-7",
       "0x1.47ae147ae147bp-5"),
  )
  expected_old_unique_hex = (
      (
          ("-0x1.2fd94630c7963p-6", "-0x1.2fd94630c7963p-6",
           "0x1.47ae147ae147bp-5"),
          ("-0x1.89374bc6a7efap-7", "-0x1.47ae147ae147bp-6",
           "0x1.47ae147ae147bp-5"),
          ("-0x1.d0b5b6a4f503ep-7", "-0x1.d0b5b6a4f503ep-7",
           "0x1.47ae147ae147bp-5"),
      ),
      (
          ("-0x1.2fd94630c7963p-6", "-0x1.d0b5b6a4f503fp-7",
           "0x1.47ae147ae147bp-5"),
          ("-0x1.23eedf0bbabd8p-6", "-0x1.23eedf0bbabd8p-6",
           "0x1.47ae147ae147bp-5"),
      ),
  )
  correct_by_element = []
  old_by_element = []
  for elem in range(int(model.flex_elemnum[0])):
    elem_adr = int(model.flex_elemdataadr[0]) + 3 * elem
    nodes = np.asarray(model.flex_elem[elem_adr:elem_adr + 3], np.int32)
    corners = np.asarray(data.flexvert_xpos[nodes], np.float64)
    correct = [_replay_fw_seed(runtime, model, data, sdf, corners, seed, True)
               for seed in range(4)]
    old = [_replay_fw_seed(runtime, model, data, sdf, corners, seed, False)
           for seed in range(4)]
    expected = expected_source_hex if elem == 0 else expected_source_hex_elem1
    assert tuple(_hex_vector(point) for _, point, _ in correct) == expected
    correct_by_element.append(_source_deduplicate(
        runtime, [(point, depth) for _, point, depth in correct]))
    old_by_element.append(_source_deduplicate(
        runtime, [(point, depth) for _, point, depth in old]))

  assert [len(points) for points in correct_by_element] == [1, 2]
  assert [len(points) for points in old_by_element] == [3, 2]
  assert tuple(
      tuple(_hex_vector(point) for point, _ in elem_points)
      for elem_points in old_by_element) == expected_old_unique_hex
  cpu = [c for c in data.contact[:data.ncon]
         if int(c.geom[0]) == geom and int(c.flex[1]) == 0]
  assert [int(c.elem[1]) for c in cpu] == [0, 1, 1]
  expected_cpu_xy = np.asarray([
      [point[0], point[1]]
      for elem_points in correct_by_element
      for point, _ in elem_points], np.float64)
  actual_cpu_xy = np.asarray([c.pos[:2] for c in cpu], np.float64)
  np.testing.assert_array_equal(actual_cpu_xy, expected_cpu_xy)

  shader = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/flex_contact.metal").read_text()
  compact_shader = "".join(shader.split())
  start = shader.index("static inline FlexDD3 flex_sdf_source_gradient(")
  stop = shader.index("return grad;", start)
  gradient_source = shader[start:stop]
  for axis in "xyz":
    assert (f"grad.{axis}=flex_dd_source_fma(derivatives[i].{axis},c,"
            f"grad.{axis});") in gradient_source
  assert "flex_dd_source_mul(derivatives[i].x,c)" not in gradient_source
  assert "flex_dd_source_mul(derivatives[i].y,c)" not in gradient_source
  assert "flex_dd_source_mul(derivatives[i].z,c)" not in gradient_source
  for axis in "xyz":
    assert (f"x.{axis}=flex_dd_source_fma(b0,tri_exact[0].{axis},"
            f"flex_dd_source_mul(u,tri_exact[1].{axis}));") in compact_shader
    assert (f"x.{axis}=flex_dd_source_fma(v,tri_exact[2].{axis},x.{axis});") \
        in compact_shader
