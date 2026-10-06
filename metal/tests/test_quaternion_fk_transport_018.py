"""Source-faithful quaternion/FK transport qualification for MuJoCo 3.10."""

import os
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace
from dataclasses import replace

import mujoco
from mujoco_metal.metal_kinematics import _FK_FLOAT_INPUTS
from mujoco_metal.metal_kinematics import _FK_INT_INPUTS
from mujoco_metal.metal_kinematics import _FK_OUTPUTS
from mujoco_metal.metal_kinematics import _fk_output_layout
from mujoco_metal.metal_kinematics import _pack_fk_constants
from mujoco_metal.metal_kinematics import _prepare_host_arrays
from mujoco_metal.capacity import _fk_auxiliary_words
from mujoco_metal.capacity import _runtime_buffer_sizes
from mujoco_metal.model import load_model
from mujoco_metal.model import snapshot_descriptor


_XML = b"""
<mujoco model="sameframe-fk">
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <body name="hinge" pos=".13 -.21 .07" quat=".9238795325112867 .3826834323650898 0 0">
      <joint name="hinge-joint" type="hinge" axis=".2 .9 .3" pos=".03 -.02 .04"/>
      <inertial pos=".031 .017 -.022" quat=".9238795325112867 0 .3826834323650898 0"
                mass="1.2" diaginertia=".08 .11 .13"/>
      <geom name="same-body" type="sphere" size=".03"/>
      <geom name="body-rot" type="sphere" pos=".07 .01 -.03" size=".02"/>
      <geom name="same-inertia" type="sphere" pos=".031 .017 -.022"
            quat=".9238795325112867 0 .3826834323650898 0" size=".02"/>
      <geom name="inertia-rot" type="sphere" pos="-.06 .02 .03"
            quat=".9238795325112867 0 .3826834323650898 0" size=".02"/>
      <geom name="general" type="sphere" pos=".014 -.037 .052"
            quat=".8660254037844386 .2886751345948129 .2886751345948129 .2886751345948129"
            size=".02"/>
      <site name="site-body"/>
      <site name="site-bodyrot" pos=".02 .03 -.01"/>
      <site name="site-inertia" pos=".031 .017 -.022"
            quat=".9238795325112867 0 .3826834323650898 0"/>
      <site name="site-inertiarot" pos="-.02 .01 .03"
            quat=".9238795325112867 0 .3826834323650898 0"/>
      <site name="site-general" pos=".013 -.027 .041"
            quat=".8660254037844386 .2886751345948129 .2886751345948129 .2886751345948129"/>
      <body name="ball" pos=".09 .04 -.02">
        <joint name="ball-joint" type="ball"/>
        <geom name="ball-geom" type="capsule" fromto="0 0 0 .04 .02 .01" size=".01"/>
        <site name="ball-site" pos=".03 -.01 .02"/>
      </body>
    </body>
    <body name="free">
      <joint name="free-joint" type="free"/>
      <geom name="free-geom" type="box" size=".02 .03 .04" pos=".01 .02 -.03"
            quat=".9238795325112867 .3826834323650898 0 0"/>
      <site name="free-site" pos="-.01 .02 .03"/>
    </body>
    <body name="slide" pos=".04 -.03 .02">
      <joint name="slide-joint" type="slide" axis=".3 .4 .8" pos=".02 -.01 .03"/>
      <geom name="slide-geom" type="sphere" size=".012" pos=".01 .02 -.01"/>
      <site name="slide-site" pos="-.02 .03 .01"/>
    </body>
    <body name="body-same" pos="-.08 .03 .11">
      <inertial pos="0 0 0" mass=".7" diaginertia=".03 .04 .05"/>
      <geom name="body-same-geom" type="sphere" size=".014"/>
    </body>
    <body name="body-rot-inertia" pos=".09 -.04 .03">
      <inertial pos=".021 -.013 .017" mass=".8" diaginertia=".04 .05 .06"/>
      <geom name="body-rot-inertia-geom" type="sphere" size=".016"/>
    </body>
    <body name="mocap" mocap="true" pos="-.2 .1 .3">
      <geom name="mocap-geom" type="sphere" size=".015" pos=".02 -.01 .03"/>
      <site name="mocap-site" pos="-.03 .01 .02"/>
    </body>
  </worldbody>
</mujoco>
"""


def _expand_pair(host, name):
  pair = host[name + "_pair"]
  stride = max(host[name].size, 1)
  high, low, tail = (pair[:stride], pair[stride:2*stride], pair[2*stride:])
  return high.astype(np.float64) + low.astype(np.float64) + tail.astype(np.float64)


def _quat_matrix(q):
  w, x, y, z = np.asarray(q, dtype=np.float64).T
  return np.stack((
      w*w + x*x - y*y - z*z, 2*(x*y-w*z), 2*(x*z+w*y),
      2*(x*y+w*z), w*w - x*x + y*y - z*z, 2*(y*z-w*x),
      2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z,
  ), axis=-1).reshape(np.asarray(q).shape[:-1] + (3, 3))


def _rot_vec_quat_source(q, v):
  """Literal engine_inline.h:mji_rotVecQuat binary64 operation sequence."""
  w, x, y, z = q
  vx, vy, vz = v
  tmp = (w * vx + y * vz - z * vy,
         w * vy + z * vx - x * vz,
         w * vz + x * vy - y * vx)
  return (vx + 2 * (y * tmp[2] - z * tmp[1]),
          vy + 2 * (z * tmp[0] - x * tmp[2]),
          vz + 2 * (x * tmp[1] - y * tmp[0]))


def test_fk_constant_packing_retains_quaternion_and_axis_residuals():
  descriptor = load_model(_XML)
  host = _prepare_host_arrays(descriptor)
  for name in ("body_quat", "body_iquat", "geom_quat", "site_quat", "jnt_axis"):
    expected = np.asarray(getattr(descriptor, name))
    np.testing.assert_array_equal(_expand_pair(host, name).reshape(expected.shape),
                                  expected)
    assert host[name + "_pair"].size == 3 * max(
        np.asarray(getattr(descriptor, name)).size, 1)
  assert set(np.unique(descriptor.geom_sameframe)) == {0, 1, 2, 3, 4}
  assert set(np.unique(descriptor.site_sameframe)) == {0, 1, 2, 3, 4}
  assert set(np.unique(descriptor.body_sameframe)) == {0, 1, 3}


def test_fk_constant_arena_offsets_cover_each_packed_field():
  descriptor = load_model(_XML)
  host = _prepare_host_arrays(descriptor)
  # This executes the same host packing contract used before any device
  # initialization and proves the metadata names cover the shader's fields.
  assert set(_FK_INT_INPUTS) == {
      "body_parentid", "jnt_type", "jnt_qposadr", "jnt_bodyid",
      "body_sameframe", "geom_bodyid", "geom_sameframe", "site_bodyid",
      "site_sameframe",
  }
  assert set(_FK_FLOAT_INPUTS) == {
      "body_pos_pair", "body_quat_pair", "jnt_pos_pair", "jnt_axis_pair",
      "qpos0_pair", "geom_pos_pair", "geom_quat_pair", "site_pos_pair",
      "site_quat_pair", "body_ipos_pair", "body_iquat_pair",
  }
  assert all(name in host for name in _FK_FLOAT_INPUTS)
  assert all(name in host for name in _FK_INT_INPUTS)
  packed, offsets = _pack_fk_constants(host, _FK_FLOAT_INPUTS, np.float32)
  for name in _FK_FLOAT_INPUTS:
    values = np.asarray(host[name], dtype=np.float32).reshape(-1)
    np.testing.assert_array_equal(packed[offsets[name]:offsets[name] + values.size],
                                  values)
  packed_int, int_offsets = _pack_fk_constants(host, _FK_INT_INPUTS, np.int32)
  for name in _FK_INT_INPUTS:
    values = np.asarray(host[name], dtype=np.int32).reshape(-1)
    np.testing.assert_array_equal(
        packed_int[int_offsets[name]:int_offsets[name] + values.size], values)
  source = mujoco.MjModel.from_xml_string(_XML.decode())
  runtime = dict(_runtime_buffer_sizes(source, 2))
  output_offsets, output_words = _fk_output_layout(descriptor, 2)
  del output_offsets
  assert runtime["fk.pose_output_arena"] == output_words
  assert runtime["model.fk.static_int"] == packed_int.size
  assert runtime["model.fk.static_float"] == packed.size
  assert runtime["fk.dims"] == 61
  assert runtime["model.smooth.tendon_j_colind"] == max(
      np.asarray(source.ten_J_colind).size, 1)
  ntree = int(np.max(np.asarray(source.body_treeid), initial=-1)) + 1
  expected_auxiliary = max(
      (2 * source.nmocap * 7 if source.nmocap else 1)
      + 2 * source.nbody
      + 2 * max(ntree, 1)
      + 2
      + 2 * max(ntree, 1), 1)
  assert _fk_auxiliary_words(2, source.nmocap, source.nbody, ntree) == expected_auxiliary
  assert runtime["fk.auxiliary"] == expected_auxiliary
  assert _fk_auxiliary_words(1, 0, 0, 0) == 4
  assert _fk_auxiliary_words(2, 2, 3, 4) == 52


def test_source_quaternion_matrix_keeps_residual_and_exact_null_contract():
  # mju_quat2Mat has an exact null-quaternion special case. A source
  # quaternion whose x component is below float32 precision must take the
  # general formula and retain its small, source-visible matrix terms.
  q = np.array([1.0 + 2.0 ** -30, 0.0, 0.0, 0.0], dtype=np.float64)
  high = q.astype(np.float32).astype(np.float64)
  low = (q - high.astype(np.float64)).astype(np.float32).astype(np.float64)
  tail = (q - high - low).astype(np.float32).astype(np.float64)
  represented = high + low + tail
  np.testing.assert_array_equal(represented, q)
  source_matrix = _quat_matrix(represented)
  high_only_matrix = _quat_matrix(high)
  assert np.any(source_matrix != high_only_matrix)
  qnorm = np.array([0.3840708785782809, 0.00285780492673382,
                    0.8967529972208935, 0.4271721884233767])
  source_n2 = (((qnorm[0] * qnorm[0] + qnorm[1] * qnorm[1])
                + qnorm[2] * qnorm[2]) + qnorm[3] * qnorm[3])
  pairwise_n2 = ((qnorm[0] * qnorm[0] + qnorm[1] * qnorm[1])
                 + (qnorm[2] * qnorm[2] + qnorm[3] * qnorm[3]))
  assert source_n2.hex() == "0x1.225859c1f2bbfp+0"
  assert pairwise_n2.hex() == "0x1.225859c1f2bbep+0"
  q = (0.7616097563914078, 0.7765762452580611,
       0.21528334335525923, 0.7641806758531811)
  axis = (0.4830719610483487, 0.3218305290972474, 0.5215556125228837)
  delta = 0.3200420778946067
  source_slide = tuple(delta * value for value in _rot_vec_quat_source(q, axis))
  rotate_scaled_axis = _rot_vec_quat_source(
      q, tuple(delta * value for value in axis))
  assert source_slide != rotate_scaled_axis
  shader = Path(__file__).parents[1] / "mujoco_metal" / "shaders" / "kinematics.metal"
  text = shader.read_text()
  assert "bool identity=kinq_equal(q,kinq(float4(1.0f,0.0f,0.0f,0.0f)))" in text
  assert "kin_add(kin_add(kin_add(kin_mul(q.w,q.w),kin_mul(q.x,q.x))," in text
  assert "kin_mul(delta,axis_pair.x)" in text
  assert "kinq_rotate(quat,axis_shift)" not in text
  assert "KinDD norm_inv=kin_div(kin_dd(1.0f),n);" in text
  assert "kin_mul(q.w,norm_inv),kin_mul(q.x,norm_inv)" in text


def test_snapshot_recomputes_sameframe_codes_after_compiled_pose_mutation():
  descriptor = load_model(_XML)
  geom_pos = np.array(descriptor.geom_pos, copy=True)
  body_ids = np.asarray(descriptor.geom_bodyid)
  index = next(i for i, code in enumerate(descriptor.geom_sameframe)
               if int(code) == 1 and int(body_ids[i]) != 0)
  geom_pos[index, 0] += .013
  changed = snapshot_descriptor(replace(descriptor, geom_pos=geom_pos))
  assert int(changed.geom_sameframe[index]) == 3

  body_iquat = np.array(descriptor.body_iquat, copy=True)
  body = int(np.flatnonzero(descriptor.body_sameframe == 3)[0])
  body_iquat[body] = np.array([.9238795325112867, .3826834323650898, 0, 0])
  changed_body = snapshot_descriptor(replace(descriptor, body_iquat=body_iquat))
  assert int(changed_body.body_sameframe[body]) == 0


def test_fk_output_arena_exact_field_offsets_and_int32_admission():
  model = SimpleNamespace(nbody=3, ngeom=2, nsite=1, njnt=4)
  offsets, words = _fk_output_layout(model, 2)
  cursor = 0
  for name, count_name, width in _FK_OUTPUTS:
    count = getattr(model, count_name)
    assert offsets[name] == cursor
    cursor += max(2 * count * width, 1)
  assert words == cursor
  assert offsets["geom_xmat"] < offsets["site_pos"]
  assert offsets["inertial_quat_tail"] < words
  huge = SimpleNamespace(nbody=1 << 29, ngeom=0, nsite=0, njnt=0)
  with pytest.raises(ValueError, match="int32"):
    _fk_output_layout(huge, 2)


def _full_qpos_cases(model):
  rows = np.tile(np.asarray(model.qpos0, dtype=np.float32), (2, 1))
  for j, typ in enumerate(model.jnt_type):
    qa = int(model.jnt_qposadr[j])
    if typ == mujoco.mjtJoint.mjJNT_SLIDE:
      rows[:, qa] += np.array([0.031, -0.047], dtype=np.float32)
    elif typ == mujoco.mjtJoint.mjJNT_BALL:
      rows[0, qa:qa + 4] = np.array([.93, .17, -.22, .25], np.float32)
      rows[1, qa:qa + 4] = np.array([.81, -.31, .37, .2], np.float32)
    elif typ == mujoco.mjtJoint.mjJNT_FREE:
      rows[0, qa:qa + 3] = np.array([.27, -.14, .31], np.float32)
      rows[1, qa:qa + 3] = np.array([-.19, .22, -.08], np.float32)
      rows[0, qa + 3:qa + 7] = np.array([.91, .12, -.2, .34], np.float32)
      rows[1, qa + 3:qa + 7] = np.array([.78, -.24, .31, .39], np.float32)
  return np.ascontiguousarray(rows)


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="set MUJOCO_METAL_RUN_GPU=1 for native Metal")
def test_native_fk_sameframe_all_modes_and_dynamic_joint_frames():
  import torch
  if not torch.backends.mps.is_available():
    pytest.skip("requires native Metal")
  from mujoco_metal.metal_kinematics import MetalKinematics

  model = mujoco.MjModel.from_xml_string(_XML.decode())
  assert set(np.unique(model.geom_sameframe)) == {0, 1, 2, 3, 4}
  assert set(np.unique(model.site_sameframe)) == {0, 1, 2, 3, 4}
  assert set(np.unique(model.body_sameframe)) == {0, 1, 3}
  qpos = _full_qpos_cases(model)
  # Public state inputs are float32. The independent source run therefore
  # receives precisely the values uploaded to the MPS kernel.
  mocap_pos = np.array([[[.14, -.06, .23]], [[-.1, .18, .07]]], np.float32)
  mocap_quat = np.array([[[.91, .1, -.2, .32]], [[.82, -.23, .31, .39]]], np.float32)
  stage = MetalKinematics(load_model(model), batch_size=2)
  observed = stage.run_device(
      torch.as_tensor(qpos, device="mps"),
      torch.as_tensor(mocap_pos, device="mps"),
      torch.as_tensor(mocap_quat, device="mps"))
  expected = []
  for w in range(2):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[w].astype(np.float64)
    data.mocap_pos[:] = mocap_pos[w].astype(np.float64)
    data.mocap_quat[:] = mocap_quat[w].astype(np.float64)
    mujoco.mj_forward(model, data)
    expected.append(data)

  def represented(name):
    return sum(observed[name + suffix].detach().cpu().numpy().astype(np.float64)
               for suffix in ("", "_low", "_tail"))

  for name, source in (("body_pos", "xpos"), ("geom_pos", "geom_xpos"),
                       ("site_pos", "site_xpos"),
                       ("inertial_pos", "xipos")):
    ref = np.stack([getattr(data, source) for data in expected])
    np.testing.assert_allclose(represented(name), ref, rtol=0, atol=6e-6)
  for kind, source, count in (
      ("body", "xmat", model.nbody), ("geom", "geom_xmat", model.ngeom),
      ("site", "site_xmat", model.nsite), ("inertial", "ximat", model.nbody),
  ):
    ref = np.stack([getattr(data, source).reshape(count, 3, 3)
                    for data in expected])
    values = np.stack([_quat_matrix(row) for row in represented(kind + "_quat")])
    if kind == "geom":
      values = sum(observed[f"geom_xmat{suffix}"].detach().cpu().numpy()
                   .astype(np.float64) for suffix in ("", "_low", "_tail"))
      values = values.reshape(2, count, 3, 3)
    np.testing.assert_allclose(values, ref, rtol=0, atol=6e-6)


@pytest.mark.skipif(os.environ.get("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="set MUJOCO_METAL_RUN_GPU=1 for native Metal")
def test_native_fk_world_mask_and_cache_invalidation_preserve_other_world():
  import torch
  if not torch.backends.mps.is_available():
    pytest.skip("requires native Metal")
  from mujoco_metal.metal_kinematics import MetalKinematics

  model = mujoco.MjModel.from_xml_string(_XML.decode())
  stage = MetalKinematics(load_model(model), batch_size=2)
  first = _full_qpos_cases(model)
  stage.run_device(torch.as_tensor(first, device="mps"),
                   torch.zeros((2, 1, 3), device="mps"),
                   torch.tensor([[[1., 0, 0, 0]], [[1., 0, 0, 0]]], device="mps"))
  before = {k: v.clone() for k, v in stage.run_device(
      torch.as_tensor(first, device="mps"),
      torch.zeros((2, 1, 3), device="mps"),
      torch.tensor([[[1., 0, 0, 0]], [[1., 0, 0, 0]]], device="mps")).items()}
  changed = first.copy()
  changed[0, :] += np.float32(.01)
  updated = stage.run_device(
      torch.as_tensor(changed, device="mps"),
      torch.zeros((2, 1, 3), device="mps"),
      torch.tensor([[[1., 0, 0, 0]], [[1., 0, 0, 0]]], device="mps"),
      world_mask=torch.tensor([0, 1], dtype=torch.int32, device="mps"))
  for key in before:
    torch.testing.assert_close(updated[key][0], before[key][0], rtol=0, atol=0)
  stage.invalidate_cache([0])
  refreshed = stage.run_device(
      torch.as_tensor(changed, device="mps"),
      torch.zeros((2, 1, 3), device="mps"),
      torch.tensor([[[1., 0, 0, 0]], [[1., 0, 0, 0]]], device="mps"))
  assert not torch.equal(refreshed["geom_xmat"][0], before["geom_xmat"][0])
