"""CPU oracle for source-precision equality acceleration references."""

import mujoco
import numpy as np
import pytest


def _renormalize(hi, lo):
  hi = np.float32(hi)
  lo = np.float32(lo)
  total = np.float32(hi + lo)
  virtual = np.float32(total - hi)
  error = np.float32((hi - np.float32(total - virtual))
                     + (lo - virtual))
  return np.float32(total), error


def _add(a, b):
  total = np.float32(a[0] + b[0])
  virtual = np.float32(total - a[0])
  error = np.float32((a[0] - np.float32(total - virtual))
                     + (b[0] - virtual))
  error = np.float32(error + a[1] + b[1])
  return _renormalize(total, error)


def _neg(a):
  return np.float32(-a[0]), np.float32(-a[1])


def _mul(a, b):
  product = np.float32(a[0] * b[0])
  error = np.float32(float(a[0]) * float(b[0]) - float(product))
  error = np.float32(error + float(a[0]) * float(b[1]))
  error = np.float32(error + float(a[1]) * float(b[0]))
  error = np.float32(error + float(a[1]) * float(b[1]))
  return _renormalize(product, error)


def _div(a, b):
  q0 = np.float32(a[0] / b[0])
  rem = _add(a, _neg(_mul(b, (q0, np.float32(0.0)))))
  q1 = np.float32((float(rem[0]) + float(rem[1])) / float(b[0]))
  return _renormalize(q0, q1)


def _param_pair(value):
  high = np.float32(value)
  low = np.float32(float(value) - float(high))
  return high, low


def _vec_pair_add(a, b):
  return [_add(x, y) for x, y in zip(a, b)]


def _vec_pair_scale(a, scale):
  return [_mul(value, (np.float32(scale), np.float32(0.0)))
          for value in a]


def _cross_pair(a, b):
  return [
      _add(_mul((a[1], np.float32(0.0)), b[2]),
           _neg(_mul((a[2], np.float32(0.0)), b[1]))),
      _add(_mul((a[2], np.float32(0.0)), b[0]),
           _neg(_mul((a[0], np.float32(0.0)), b[2]))),
      _add(_mul((a[0], np.float32(0.0)), b[1]),
           _neg(_mul((a[1], np.float32(0.0)), b[0]))),
  ]


def _qrot_translate_pair(q, local, translation):
  q = np.asarray(q, dtype=np.float32)
  local = np.asarray(local, dtype=np.float32)
  translation = np.asarray(translation, dtype=np.float32)
  local_pair = [(value, np.float32(0.0)) for value in local]
  inner = _vec_pair_add(
      _cross_pair(q[1:], local_pair),
      _vec_pair_scale(local_pair, q[0]))
  outer = _cross_pair(q[1:], inner)
  rotated = _vec_pair_add(local_pair, _vec_pair_scale(outer, 2.0))
  return _vec_pair_add(
      rotated, [(value, np.float32(0.0)) for value in translation])


def _source_aref_pair(model, qpos, qvel):
  """Match the joint-equality shader's two-word reference arithmetic."""
  eq = 0
  j1, j2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
  q1, q2 = int(model.jnt_qposadr[j1]), int(model.jnt_qposadr[j2])
  d1, d2 = int(model.jnt_dofadr[j1]), int(model.jnt_dofadr[j2])
  pos = _add((qpos[q1], np.float32(0.0)), _neg((np.float32(model.qpos0[q1]), 0.0)))
  pos = _add(pos, _neg(_param_pair(model.eq_data[eq, 0])))
  dif = _add((qpos[q2], 0.0), _neg((np.float32(model.qpos0[q2]), 0.0)))
  coeff = _param_pair(model.eq_data[eq, 1])
  poly = _mul(coeff, dif)
  pos = _add(pos, _neg(poly))
  vel = _add((qvel[d1], 0.0), _neg(_mul(coeff, (qvel[d2], 0.0))))

  solref = model.eq_solref[eq]
  solimp = model.eq_solimp[eq]
  r0, r1, d0, d1imp, imp_width = (
      _param_pair(x) for x in [*solref, *solimp[:3]])
  # This fixture is outside the impedance transition width, so pinned
  # getimpedance selects solimp[1] exactly.
  impedance = d1imp
  B = _div((np.float32(2.0), 0.0), _mul(d1imp, r0))
  den = _mul(_mul(_mul(d1imp, d1imp), _mul(r0, r0)), _mul(r1, r1))
  K = _div((np.float32(1.0), 0.0), den)
  return _add(_neg(_mul(B, vel)),
              _neg(_mul(_mul(K, impedance), pos)))


def test_joint_equality_aref_pair_matches_pinned_source_at_float32_state():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option timestep=".0005" gravity="0 0 -9.81"/>
    <worldbody>
      <body pos="-1 0 .4"><joint name="chain_slide" type="slide" axis="1 0 0"/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
        <body pos=".35 0 0"><joint name="chain_hinge" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".06" mass=".7"/>
        </body>
      </body>
      <body pos="1 0 .3"><joint name="coupled_slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".08" mass=".8"/></body>
    </worldbody>
    <equality><joint joint1="chain_hinge" joint2="coupled_slide" polycoef="0 1 0 0 0"
      solref=".02 1"/></equality>
  </mujoco>""")
  qpos = np.asarray(model.qpos0, dtype=np.float32).copy()
  qvel = np.asarray([0.2, -0.3, 0.1], dtype=np.float32)
  qpos[1] = np.float32(0.18)
  qpos[2] = np.float32(0.01)
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  row = int(np.flatnonzero(
      data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))[0])
  high, low = _source_aref_pair(model, qpos, qvel)
  np.testing.assert_allclose(float(high) + float(low), data.efc_aref[row],
                             rtol=0.0, atol=2e-6)
  # Reproduce the old all-float32 expression, including its rounded K/B and
  # products. The low-word source constants and paired product recover the
  # pinned result where that expression loses several float32 ULPs.
  params = np.hstack([model.eq_solref[0], model.eq_solimp[0]]).astype(np.float32)
  width, r0, r1 = params[3], params[0], params[1]
  denominator = np.float32(width * width)
  denominator = np.float32(denominator * r0)
  denominator = np.float32(denominator * r0)
  denominator = np.float32(denominator * r1)
  denominator = np.float32(denominator * r1)
  stiffness = np.float32(1.0) / denominator
  damping = np.float32(2.0) / np.float32(width * r0)
  imp = params[3]
  pos = np.float32(qpos[1] - qpos[2])
  vel = np.float32(qvel[1] - qvel[2])
  naive = np.float32(-np.float32(damping * vel)
                     - np.float32(np.float32(stiffness * imp) * pos))
  assert abs(float(naive) - data.efc_aref[row]) > 4e-5


def test_cached_aref_pair_transports_reference_across_velocity_change():
  model = mujoco.MjModel.from_xml_string("""<mujoco>
    <worldbody>
      <body><joint name="a" type="hinge"/><geom type="sphere" size=".1"/></body>
      <body><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1"/></body>
    </worldbody>
    <equality><joint joint1="a" joint2="b" polycoef=".013 1.00000003 0 0 0"
      solref=".020000001 1.00000003"/></equality>
  </mujoco>""")
  qpos = np.asarray(model.qpos0, dtype=np.float32).copy()
  qpos[:] = np.asarray([.12345679, -.2345679], dtype=np.float32)
  old_qvel = np.asarray([.12345679, -.2345679], dtype=np.float32)
  new_qvel = np.asarray([.1234568, -.23456786], dtype=np.float32)
  old_hi, old_lo = _source_aref_pair(model, qpos, old_qvel)
  new_hi, new_lo = _source_aref_pair(model, qpos, new_qvel)

  # Match refresh_cached_constraint_aref_low: retain the POS expansion,
  # apply -B*(J*delta_qvel), then re-center the expansion around the refreshed
  # public high word. B's static residual is relative to the context's high B.
  eq = 0
  b_scale, r0 = float(model.eq_solimp[eq, 1]), float(model.eq_solref[eq, 0])
  exact_b = 2.0 / (b_scale * r0)
  high_b = np.float32(exact_b)
  low_b = np.float32(exact_b - float(high_b))
  coeff = _param_pair(model.eq_data[eq, 1])
  old_vel = _add(_param_pair(old_qvel[0]), _neg(_mul(coeff, _param_pair(old_qvel[1]))))
  new_vel = _add(_param_pair(new_qvel[0]), _neg(_mul(coeff, _param_pair(new_qvel[1]))))
  delta_vel = _add(new_vel, _neg(old_vel))
  delta_ar = _neg(_mul((high_b, low_b), delta_vel))
  transported = _add((old_hi, old_lo), delta_ar)
  recentered = _add(transported, _neg((new_hi, np.float32(0.0))))
  transported_low = np.float32(recentered[0] + recentered[1])
  np.testing.assert_allclose(float(new_hi) + float(transported_low),
                             float(new_hi) + float(new_lo),
                             rtol=0.0, atol=3e-7)


def test_fresh_assembly_clears_only_aref_low_tail_for_each_world():
  torch = pytest.importorskip("torch")
  from mujoco_metal.coupled_constraints import _clear_current_aref_low

  batch, stride, rows, offset = 2, 64, 5, 47
  debug = torch.arange(batch * stride, dtype=torch.float32).reshape(batch, stride)
  before = debug.clone()
  _clear_current_aref_low(debug, batch, rows, stride, offset)
  for world in range(batch):
    torch.testing.assert_close(debug[world, offset:offset + rows],
                               torch.zeros(rows))
    assert torch.equal(debug[world, :offset], before[world, :offset])
    assert torch.equal(debug[world, offset + rows:],
                       before[world, offset + rows:])


def test_fresh_assembly_resets_low_words_before_equality_and_contact_producers():
  import inspect

  from mujoco_metal.coupled_constraints import MetalCoupledConstraints

  source = inspect.getsource(MetalCoupledConstraints.run_device)
  clear_at = source.index("_clear_current_aref_low(")
  equality_at = source.index("# 1b. Equality assembly kernel")
  contact_at = source.index("# 1. Contact normal kernel")
  assert clear_at < contact_at < equality_at

  cached_source = inspect.getsource(MetalCoupledConstraints.run_velocity_device)
  assert "_clear_current_aref_low(" not in cached_source


def test_body_anchor_point_transform_retains_sub_ulp_pair():
  """The connect/weld anchor transform keeps product error for aref."""
  q = np.asarray([0.9635211, 0.12187155, -0.08712346, 0.21987654],
                 dtype=np.float32)
  local = np.asarray([0.13765432, -0.09345679, 0.21765433],
                     dtype=np.float32)
  translation = np.asarray([0.1345679, -0.28765434, 0.19345678],
                           dtype=np.float32)

  vector = q[1:].astype(np.float64)
  point = local.astype(np.float64)
  exact = (point + 2.0 * np.cross(
      vector, np.cross(vector, point) + float(q[0]) * point)
      + translation.astype(np.float64))
  pair = _qrot_translate_pair(q, local, translation)
  reconstructed = np.asarray([float(hi) + float(lo) for hi, lo in pair])
  rounded = np.asarray([hi for hi, _ in pair])
  assert np.max(np.abs(reconstructed - exact)) < 2e-15
  assert np.max(np.abs(rounded - exact)) > 1e-9

  from pathlib import Path
  shader = (Path(__file__).parents[1] / "mujoco_metal" / "shaders"
            / "equality_assembly.metal").read_text()
  assert "inline EqVec3Pair eq_qrot_pair" in shader
  assert shader.count("eq_qrot_translate_pair(") >= 3
  assert "eq_vec_pair_component(cpos_pair, k)" in shader
  assert "eq_vec_pair_component(cpos_t_pair, k)" in shader
