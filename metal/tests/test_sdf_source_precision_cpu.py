"""Offline source-order checks for the bundled SDF callback precision path."""
import ctypes
from fractions import Fraction
import math
import random
import struct
from pathlib import Path
import importlib.util
import sys

import mujoco
import numpy as np


def _source_function(source, signature):
  start = source.index(signature)
  brace = source.index("{", start)
  depth = 0
  for pos in range(brace, len(source)):
    if source[pos] == "{":
      depth += 1
    elif source[pos] == "}":
      depth -= 1
      if depth == 0:
        return source[start:pos + 1]
  raise AssertionError(f"unterminated function: {signature}")


def _f32(x):
  return np.float32(x).item()


def _two_sum(a, b):
  s = _f32(a + b)
  bb = _f32(s - a)
  return s, _f32(_f32(a - _f32(s - bb)) + _f32(b - bb))


def _wadd_word(a, b):
  x = _two_sum(a[0], b)
  y = _two_sum(a[1], x[1])
  z = _two_sum(a[2], y[1])
  h = _two_sum(x[0], y[0])
  m = _two_sum(h[1], z[0])
  n = _two_sum(h[0], m[0])
  l = _two_sum(m[1], z[1])
  k = _two_sum(n[1], l[0])
  q = _two_sum(n[0], k[0])
  return q[0], q[1], _f32(k[1] + l[1])


def _wadd(a, b):
  return _wadd_word(_wadd_word(_wadd_word(a, b[0]), b[1]), b[2])


def _wneg(a):
  return tuple(-x for x in a)


def _wsub(a, b):
  return _wadd(a, _wneg(b))


def _wmul(a, b):
  result = (0.0, 0.0, 0.0)
  for x, y in ((a[0], b[0]), (a[0], b[1]), (a[1], b[0]),
               (a[0], b[2]), (a[1], b[1]), (a[2], b[0])):
    product = _f32(x * y)
    # For binary32 operands, Python binary64 retains the exact product and
    # sum needed to model the shader's correctly rounded fma residual.
    result = _wadd_word(result, _f32(float(x) * float(y) - product))
    result = _wadd_word(result, product)
  return result


def _wdiv(a, b):
  q0 = _f32(a[0] / b[0])
  result = _wsub(a, _wmul(b, (q0, 0.0, 0.0)))
  q1 = _f32((result[0] + result[1] + result[2]) / b[0])
  q = _wadd((q0, 0.0, 0.0), (q1, 0.0, 0.0))
  result = _wsub(a, _wmul(b, q))
  q2 = _f32((result[0] + result[1] + result[2]) / b[0])
  return _wadd(q, (q2, 0.0, 0.0))


def _wsqrt(a):
  q0 = _f32(np.sqrt(_f32(max(a[0], 0.0))))
  q = (q0, 0.0, 0.0)
  den = _wmul((2.0, 0.0, 0.0), q)
  rem = _wsub(a, _wmul(q, q))
  q = _wadd(q, _wdiv(rem, den))
  rem = _wsub(a, _wmul(q, q))
  return _wadd(q, _wdiv(rem, den))


def _wnorm2(x, y):
  return _wsqrt(_wadd(_wmul(x, x), _wmul(y, y)))


def _wbowl(point):
  h, radius, thick = (0.32, 0.9, 0.05)
  h, radius, thick = ((h, 0.0, 0.0), (radius, 0.0, 0.0),
                      (thick, 0.0, 0.0))
  width = _wsqrt(_wsub(_wmul(radius, radius), _wmul(h, h)))
  qx = _wnorm2(point[0], point[1])
  qy = point[2]
  branch = _wsub(_wmul(h, qx), _wmul(width, qy))
  if branch[0] < 0.0 or (branch[0] == 0.0 and branch[1] < 0.0):
    dx, dy = _wsub(qx, width), _wsub(qy, h)
    norm = _wnorm2(dx, dy)
    return _wsub(norm, thick)
  shell = _wsub(_wnorm2(qx, qy), radius)
  if shell[0] < 0.0 or (shell[0] == 0.0 and shell[1] < 0.0):
    shell = _wneg(shell)
  return _wsub(shell, thick)


def _wide_bowl_gradient(point):
  p = []
  for coordinate in point:
    hi = _f32(coordinate)
    lo = _f32(coordinate - hi)
    p.append((hi, lo, 0.0))
  base = _wbowl(p)
  result = []
  for axis in range(3):
    shifted = list(p)
    shifted[axis] = _wadd(shifted[axis], (1e-8, 0.0, 0.0))
    grad = _wdiv(_wsub(_wbowl(shifted), base), (1e-8, 0.0, 0.0))
    result.append((grad[0] + grad[1]) + grad[2])
  return np.asarray(result)


def _wide_torus_gradient(point, radius1=.28):
  x, y, z = [(float(c), 0.0, 0.0) for c in point]
  lenxy = _wnorm2(x, y)
  q = _wsub(lenxy, (radius1, 0.0, 0.0))
  lenqz = _wnorm2(q, z)
  denominator = (lenqz if lenqz[0] > np.finfo(np.float32).tiny
                 else (np.finfo(np.float32).tiny, 0.0, 0.0))
  result = []
  for component in (_wdiv(_wmul(q, _wdiv(x, lenxy)), denominator),
                    _wdiv(_wmul(q, _wdiv(y, lenxy)), denominator),
                    _wdiv(z, denominator)):
    result.append((component[0] + component[1]) + component[2])
  return np.asarray(result)


def _plugin_descriptor(slot):
  fields = [
      ("name", ctypes.c_char_p), ("nattribute", ctypes.c_int),
      ("attributes", ctypes.POINTER(ctypes.c_char_p)),
      ("capabilityflags", ctypes.c_int), ("needstage", ctypes.c_int),
  ]
  fields.extend((name, ctypes.c_void_p) for name in (
      "nstate", "nsensordata", "init", "destroy", "copy", "reset",
      "compute", "advance", "visualize", "actuator_act_dot",
      "sdf_distance", "sdf_gradient", "sdf_staticdistance",
      "sdf_attribute", "sdf_aabb"))
  descriptor_type = type("_RegisteredPlugin", (ctypes.Structure,),
                         {"_fields_": fields})
  library = ctypes.CDLL(mujoco._structs.__file__)
  getter = library.mjp_getPluginAtSlot
  getter.argtypes = [ctypes.c_int]
  getter.restype = ctypes.POINTER(descriptor_type)
  descriptor = getter(int(slot))
  assert descriptor
  return descriptor.contents


def _model(name):
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco><extension><plugin plugin="mujoco.sdf.{name}">
      <instance name="shape"/>
    </plugin></extension><asset>
      <mesh name="sdfmesh"><plugin instance="shape"/></mesh>
    </asset><worldbody><geom type="sdf" mesh="sdfmesh">
      <plugin instance="shape"/>
    </geom></worldbody></mujoco>
  """)


def _source_bowl_gradient(point, height=.32, radius=.9, thick=.05):
  """Pinned bowl.cc distance and positive mjtNum finite-difference order."""
  def distance(p):
    width = np.sqrt(radius * radius - height * height)
    q = [np.sqrt(p[0] * p[0] + p[1] * p[1]), p[2]]
    qdiff = [q[0] - width, q[1] - height]
    if height * q[0] < width * q[1]:
      return np.sqrt(qdiff[0] * qdiff[0] + qdiff[1] * qdiff[1]) - thick
    return abs(np.sqrt(q[0] * q[0] + q[1] * q[1]) - radius) - thick
  eps = 1e-8
  d0 = distance(point)
  result = []
  for axis in range(3):
    shifted = list(point)
    shifted[axis] += eps
    result.append((distance(shifted) - d0) / eps)
  return np.asarray(result)


def test_bowl_hidden_point_source_callback_and_shader_use_same_stencil():
  # First hidden point from the CPU/GPU witness capture that exposed the
  # residual. This runs the installed pinned plugin callback on CPU.
  point = np.asarray([0.8000487209889151, -0.6000087813390214,
                      1.5871044851766924e-5], dtype=np.float64)
  model = _model("bowl")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  callback = _plugin_descriptor(model.plugin[0]).sdf_gradient
  gradient_fn = ctypes.CFUNCTYPE(
      None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
      ctypes.c_void_p, ctypes.c_int)(callback)
  p = (ctypes.c_double * 3)(*point)
  out = (ctypes.c_double * 3)()
  gradient_fn(out, p, ctypes.c_void_p(data._address), 0)
  np.testing.assert_allclose(np.asarray(out), _source_bowl_gradient(point),
                             rtol=0, atol=2e-15)
  np.testing.assert_allclose(_wide_bowl_gradient(point), np.asarray(out),
                             rtol=0, atol=2e-7)

  shader = Path(__file__).parents[1] / "mujoco_metal/shaders/plugin_sdf.metal"
  source = shader.read_text()
  assert "plugin_sdf_source_gradient_wide_bowl" in source
  assert "plugin_sdf_source_gradient_wide_bowl(p_hi, p_lo" in source
  assert "SdfWide eps = sw_source_eps();" in source
  assert "sw_add(shifted[axis], eps)" in source
  assert "sw_div(sw_sub(value, base), eps)" in source


def test_torus_source_analytic_gradient_keeps_pair_residuals():
  point = np.asarray([0.30000001192092896, -0.20000000298023224, 0.0])
  model = _model("torus")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  callback = _plugin_descriptor(model.plugin[0]).sdf_gradient
  gradient_fn = ctypes.CFUNCTYPE(
      None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
      ctypes.c_void_p, ctypes.c_int)(callback)
  p = (ctypes.c_double * 3)(*point)
  out = (ctypes.c_double * 3)()
  gradient_fn(out, p, ctypes.c_void_p(data._address), 0)
  np.testing.assert_allclose(_wide_torus_gradient(point), np.asarray(out),
                             rtol=0, atol=2e-7)

  source = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/plugin_sdf.metal").read_text()
  assert "plugin_sdf_torus_gradient_wide(p_hi, p_lo" in source
  assert "sw_norm2(x, y)" in source
  assert "sw_norm2(q, z)" in source
  assert "sw_sign(sw_sub(lenqz, minval)) > 0" in source


def test_compiled_plugin_attributes_keep_third_source_residual():
  module_path = (Path(__file__).parents[1] /
                 "mujoco_metal/bundled_plugins.py")
  spec = importlib.util.spec_from_file_location("sdf_bundle_delta", module_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  config = {
      "height": "0.3212345678901234",
      "radius": "0.9123456789012345",
      "thickness": "0.0543210987654321",
  }
  xml_attrs = "".join(
      f'<config key="{key}" value="{value}"/>'
      for key, value in config.items())
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><extension><plugin plugin="mujoco.sdf.bowl">
      <instance name="shape">{xml_attrs}</instance>
    </plugin></extension><asset>
      <mesh name="sdfmesh"><plugin instance="shape"/></mesh>
    </asset><worldbody><geom type="sdf" mesh="sdfmesh">
      <plugin instance="shape"/>
    </geom></worldbody></mujoco>
  """)
  lowered = module.lower_bundled_plugins(model)
  high = lowered.sdf_geom_attributes[0]
  low = lowered.sdf_geom_attributes_low[0]
  tail = lowered.sdf_geom_attributes_tail[0]
  expected = np.asarray([float(config[k]) for k in
                         ("height", "radius", "thickness")])
  represented = (high[:3].astype(np.float64) + low[:3].astype(np.float64)
                 + tail[:3].astype(np.float64))
  np.testing.assert_array_equal(represented, expected)
  assert tail.shape == (5,)
  assert np.any(tail[:3] != 0.0)


def _f32_exact(value):
  return struct.unpack("f", struct.pack("f", float(value)))[0]


def _split_source_double(value):
  """Represent one source binary64 result as three binary32 words."""
  high = _f32_exact(value)
  residual = value - high
  middle = _f32_exact(residual)
  low = _f32_exact(residual - middle)
  words = (high, middle, low)
  exact_words = sum((Fraction.from_float(word) for word in words), Fraction())
  assert exact_words == Fraction.from_float(value)
  return words


def _exact_float_sum(words):
  return sum((Fraction.from_float(float(word)) for word in words), Fraction())


def _source_add_words(a, b):
  return float(_exact_float_sum(a + b))


def _source_mul_words(a, b):
  return float(_exact_float_sum(a) * _exact_float_sum(b))


def _source_fma_words(a, b, c):
  return float(_exact_float_sum(a) * _exact_float_sum(b)
               + _exact_float_sum(c))


def test_source_rounding_expansions_match_binary64_operation_boundaries():
  rng = random.Random(0x5310)
  cases = [
      (1.0, math.ldexp(1.0, -53)),
      (1.0, math.nextafter(math.ldexp(1.0, -53), math.inf)),
      (-1.0, -math.nextafter(math.ldexp(1.0, -53), math.inf)),
      (math.ldexp(1.0, 30), math.ldexp(1.0, -23)),
      (math.ldexp(1.0, -30), math.ldexp(1.0, -83)),
  ]
  for _ in range(256):
    exponent_a = rng.randrange(-30, 31)
    exponent_b = rng.randrange(-30, 31)
    left = math.ldexp(rng.uniform(-1.0, 1.0), exponent_a)
    right = math.ldexp(rng.uniform(-1.0, 1.0), exponent_b)
    cases.append((left, right))

  for left, right in cases:
    a = _split_source_double(left)
    b = _split_source_double(right)
    exact_a, exact_b = _exact_float_sum(a), _exact_float_sum(b)
    # Fraction->float is nearest-even binary64 rounding of the exact expansion,
    # exactly the boundary each SdfWide add/multiply must reproduce.
    expected_sum = float(exact_a + exact_b)
    expected_product = float(exact_a * exact_b)
    c = _split_source_double(math.ldexp(rng.uniform(-1.0, 1.0),
                                        rng.randrange(-30, 31)))
    expected_fma = float(exact_a * exact_b + _exact_float_sum(c))
    assert _source_add_words(a, b) == expected_sum
    assert _source_mul_words(a, b) == expected_product
    assert _source_fma_words(a, b, c) == expected_fma
    assert sum((Fraction.from_float(v) for v in _split_source_double(expected_sum)),
               Fraction()) == Fraction.from_float(expected_sum)
    assert sum((Fraction.from_float(v) for v in _split_source_double(expected_product)),
               Fraction()) == Fraction.from_float(expected_product)

  source = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/plugin_sdf.metal").read_text()
  product = _source_function(
      source, "static __attribute__((noinline)) int sw_expansion_add_product(")
  multiply_core = _source_function(
      source, "[[clang::noinline]] inline SdfWide sw_source_mul_core(")
  multiply = _source_function(
      source, "[[clang::noinline]] inline SdfWide sw_source_mul(")
  assert "static __attribute__((noinline)) int sw_expansion_add(" in source
  # Product residuals are computed after scaling both factors into a safe
  # exponent range, then rescaled before insertion into the expansion.
  assert "float scaled_a=sw_scale_word_pow2(a,scale_a);" in product
  assert "float scaled_b=sw_scale_word_pow2(b,scale_b);" in product
  assert "float error=fma(scaled_a,scaled_b,-product);" in product
  assert "float actual_error=sw_scale_word_pow2(error,-total_scale);" in product
  assert "float actual_product=sw_scale_word_pow2(product,-total_scale);" in product
  assert "count=sw_expansion_add_product(expansion,count,av[i],bv[j],1,scratch);" in multiply_core
  assert "SdfProjection53 rounded=sw_expansion_round53(expansion,count);" in multiply_core
  assert "sw_source_mul_core(sw_scale_wide_pow2(a,scale_a)," in multiply
  assert "return sw_scale_wide_pow2(product,-scale_a-scale_b);" in multiply
  assert "sw_source_dot3" in source
  assert "sw_source_add(a,b)" in source
  assert "sw_source_mul(a,b)" in source
  assert "return sw_source_fma(z,w,xy);" in source


def test_rigid_sdf_keeps_source_point_distance_and_witness_tails():
  source = (Path(__file__).parents[1] /
            "mujoco_metal/shaders/sdf_narrowphase.metal").read_text()
  assert "SdfWide sdf_descend(" in source
  assert "SdfPointPair accepted_points[50]" in source
  assert "sdf_pair_distance(x,accepted_points[k])" in source
  assert "sw_source_minval()" in source
  assert "sdf_pair_add_scaled(world_point,world_normal" in source
  assert "sw_float32(dint)" in source
  assert "float2 pts[24]" not in source
  assert "local_normal.tail=-local_normal.tail" in source
