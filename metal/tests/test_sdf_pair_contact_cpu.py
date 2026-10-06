"""CPU arithmetic checks for the source-precision SDF contact path.

These checks validate only the represented-value arithmetic contract.  The
public collision tests remain the physics oracle and are run separately by the
serialized native qualification gate.
"""

import numpy as np


def _f32(value):
  return np.float32(value)


def _renorm(hi, lo):
  total = _f32(hi + lo)
  return total, _f32(lo - _f32(total - hi))


def _add(a, b):
  s = _f32(a[0] + b[0])
  bv = _f32(s - a[0])
  e = _f32(_f32(a[0] - _f32(s - bv)) + _f32(b[0] - bv))
  e = _f32(e + a[1] + b[1])
  return _renorm(s, e)


def _mul(a, b):
  product = _f32(a[0] * b[0])
  error = _f32(float(a[0]) * float(b[0]) - float(product))
  error = _f32(error + _f32(a[0] * b[1]) + _f32(a[1] * b[0])
               + _f32(a[1] * b[1]))
  return _renorm(product, error)


def test_sdf_pair_affine_transform_matches_binary64_reference():
  matrix = np.asarray([[0.731, -0.414, 0.541],
                       [0.613, 0.887, -0.229],
                       [-0.298, 0.211, 0.931]], dtype=np.float32)
  point = np.asarray([0.47, -0.29, 0.11], dtype=np.float64)
  point_hi = point.astype(np.float32)
  point_lo = (point - point_hi.astype(np.float64)).astype(np.float32)
  translation = np.asarray([0.031, -0.027, 0.019], dtype=np.float32)
  for row in range(3):
    value = (0.0, 0.0)
    for col in range(3):
      value = _add(value, _mul((matrix[row, col], 0.0),
                               (point_hi[col], point_lo[col])))
    value = _add(value, (translation[row], 0.0))
    expected = sum(float(matrix[row, col]) * point[col] for col in range(3))
    expected += float(translation[row])
    assert abs(float(value[0]) + float(value[1]) - expected) < 2e-8


def test_sdf_pair_descent_retains_sub_ulp_source_update():
  x = 0.47
  high = _f32(x)
  low = _f32(x - float(high))
  grad = _f32(1.0e-8)
  alpha = _f32(0.5)
  naive = _f32(high - _f32(grad * alpha))
  paired = _add((high, low), _mul((_f32(-grad), 0.0), (alpha, 0.0)))
  expected = x - float(grad) * float(alpha)
  assert naive == high
  assert paired[1] != 0.0
  assert abs(float(paired[0]) + float(paired[1]) - expected) < 1e-8
