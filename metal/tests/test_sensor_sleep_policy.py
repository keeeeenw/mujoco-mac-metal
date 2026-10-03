"""Pinned sensor sleep-state output-retention policy fixtures."""

import numpy as np
import mujoco
import pytest

from mujoco_metal.sensor_sleep import (
    lower_sensor_sleep_policy,
    sensor_awake_mask_cpu,
)


class _Model:
  nsensor = 6
  nbody = 3
  ntree = 2
  ntendon = 0
  neq = 0
  nu = 0
  nflex = 0
  body_treeid = np.array([-1, 0, 1], dtype=np.int32)
  jnt_bodyid = np.array([1], dtype=np.int32)
  dof_bodyid = np.array([1, 2], dtype=np.int32)
  site_bodyid = np.array([1, 2], dtype=np.int32)
  geom_bodyid = np.array([], dtype=np.int32)
  cam_bodyid = np.array([], dtype=np.int32)
  light_bodyid = np.array([], dtype=np.int32)
  sensor_type = np.array([
      mujoco.mjtSensor.mjSENS_JOINTPOS,
      mujoco.mjtSensor.mjSENS_FRAMEPOS,
      mujoco.mjtSensor.mjSENS_FRAMEPOS,
      mujoco.mjtSensor.mjSENS_USER,
      mujoco.mjtSensor.mjSENS_RANGEFINDER,
      mujoco.mjtSensor.mjSENS_CONTACT,
  ], dtype=np.int32)
  sensor_objtype = np.array([
      mujoco.mjtObj.mjOBJ_JOINT,
      mujoco.mjtObj.mjOBJ_BODY,
      mujoco.mjtObj.mjOBJ_UNKNOWN,
      mujoco.mjtObj.mjOBJ_UNKNOWN,
      mujoco.mjtObj.mjOBJ_UNKNOWN,
      mujoco.mjtObj.mjOBJ_SITE,
  ], dtype=np.int32)
  sensor_objid = np.array([0, 1, -1, -1, -1, 0], dtype=np.int32)
  sensor_reftype = np.array([
      mujoco.mjtObj.mjOBJ_UNKNOWN,
      mujoco.mjtObj.mjOBJ_BODY,
      mujoco.mjtObj.mjOBJ_BODY,
      mujoco.mjtObj.mjOBJ_UNKNOWN,
      mujoco.mjtObj.mjOBJ_UNKNOWN,
      mujoco.mjtObj.mjOBJ_UNKNOWN,
  ], dtype=np.int32)
  sensor_refid = np.array([-1, 2, 2, -1, -1, -1], dtype=np.int32)


def test_pinned_sensor_wake_rules_and_unknown_object_combinations():
  treeids, counts, always, sensor_refs = lower_sensor_sleep_policy(_Model())
  np.testing.assert_array_equal(always, [0, 0, 0, 1, 1, 1])
  np.testing.assert_array_equal(counts[:3], [[1, -1], [1, 1], [-1, 1]])

  # MuJoCo holds output only when the sensor sleepState is ASLEEP. A reference
  # object wakes a frame sensor; an UNKNOWN object delegates to its peer.
  np.testing.assert_array_equal(
      sensor_awake_mask_cpu([[0, 1]], treeids, counts, always),
      [[0, 1, 1, 1, 1, 1]])
  np.testing.assert_array_equal(
      sensor_awake_mask_cpu([[0, 0]], treeids, counts, always),
      [[0, 0, 0, 1, 1, 1]])


def test_static_sensor_object_and_unsupported_objects_fail_open_awake():
  model = _Model()
  model.nsensor = 2
  model.sensor_type = np.array([mujoco.mjtSensor.mjSENS_FRAMEPOS] * 2, dtype=np.int32)
  model.sensor_objtype = np.array([
      mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_EQUALITY], dtype=np.int32)
  model.sensor_objid = np.array([0, 0], dtype=np.int32)
  model.sensor_reftype = np.array([
      mujoco.mjtObj.mjOBJ_UNKNOWN, mujoco.mjtObj.mjOBJ_UNKNOWN], dtype=np.int32)
  model.sensor_refid = np.array([-1, -1], dtype=np.int32)
  treeids, counts, always, sensor_refs = lower_sensor_sleep_policy(model)
  np.testing.assert_array_equal(counts, [[0, -1], [0, 0]])
  np.testing.assert_array_equal(always, [0, 1])
  # A known STATIC target paired with UNKNOWN is awake under pinned's special
  # case; an unsupported equality target is always awake conservatively.
  np.testing.assert_array_equal(
      sensor_awake_mask_cpu([[0, 0]], treeids, counts, always), [[1, 1]])


def test_sensor_object_references_propagate_sleep_state_recursively():
  model = _Model()
  model.nsensor = 3
  model.sensor_type = np.array([
      mujoco.mjtSensor.mjSENS_JOINTPOS,
      mujoco.mjtSensor.mjSENS_FRAMEPOS,
      mujoco.mjtSensor.mjSENS_FRAMEPOS,
  ], dtype=np.int32)
  model.sensor_objtype = np.array([
      mujoco.mjtObj.mjOBJ_JOINT,
      mujoco.mjtObj.mjOBJ_SENSOR,
      mujoco.mjtObj.mjOBJ_SENSOR,
  ], dtype=np.int32)
  model.sensor_objid = np.array([0, 0, 1], dtype=np.int32)
  model.sensor_reftype = np.full((3,), mujoco.mjtObj.mjOBJ_UNKNOWN,
                                 dtype=np.int32)
  model.sensor_refid = np.full((3,), -1, dtype=np.int32)
  treeids, counts, always, sensor_refs = lower_sensor_sleep_policy(model)
  np.testing.assert_array_equal(counts[:, 0], [1, -4, -4])
  np.testing.assert_array_equal(sensor_refs[:, 0], [-1, 0, 1])
  np.testing.assert_array_equal(
      sensor_awake_mask_cpu([[0, 0]], treeids, counts, always, sensor_refs),
      [[0, 0, 0]])
  np.testing.assert_array_equal(
      sensor_awake_mask_cpu([[1, 0]], treeids, counts, always, sensor_refs),
      [[1, 1, 1]])


def test_sensor_lowering_covers_pinned_object_families():
  model = _Model()
  model.nsensor = 10
  model.ntendon = model.nflex = model.nu = model.neq = 1
  model.sensor_type = np.full(
      (10,), mujoco.mjtSensor.mjSENS_FRAMEPOS, dtype=np.int32)
  model.sensor_objtype = np.array([
      mujoco.mjtObj.mjOBJ_XBODY,
      mujoco.mjtObj.mjOBJ_DOF,
      mujoco.mjtObj.mjOBJ_SITE,
      mujoco.mjtObj.mjOBJ_GEOM,
      mujoco.mjtObj.mjOBJ_CAMERA,
      mujoco.mjtObj.mjOBJ_LIGHT,
      mujoco.mjtObj.mjOBJ_TENDON,
      mujoco.mjtObj.mjOBJ_FLEX,
      mujoco.mjtObj.mjOBJ_ACTUATOR,
      mujoco.mjtObj.mjOBJ_EQUALITY,
  ], dtype=np.int32)
  model.sensor_objid = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                dtype=np.int32)
  model.sensor_reftype = np.full(
      (10,), mujoco.mjtObj.mjOBJ_UNKNOWN, dtype=np.int32)
  model.sensor_refid = np.full((10,), -1, dtype=np.int32)
  model.geom_bodyid = np.array([1], dtype=np.int32)
  model.cam_bodyid = np.array([1], dtype=np.int32)
  model.light_bodyid = np.array([2], dtype=np.int32)
  model.tendon_treenum = np.array([2], dtype=np.int32)
  model.tendon_treeid = np.array([[0, 1]], dtype=np.int32)
  model.flex_interp = np.array([0], dtype=np.int32)
  model.flex_vertadr = np.array([0], dtype=np.int32)
  model.flex_vertnum = np.array([2], dtype=np.int32)
  model.flex_vertbodyid = np.array([0, 2], dtype=np.int32)
  model.actuator_trntype = np.array([mujoco.mjtTrn.mjTRN_JOINT], dtype=np.int32)
  model.actuator_trnid = np.array([[0, -1]], dtype=np.int32)
  model.eq_type = np.array([mujoco.mjtEq.mjEQ_CONNECT], dtype=np.int32)
  model.eq_objtype = np.array([mujoco.mjtObj.mjOBJ_BODY], dtype=np.int32)
  model.eq_obj1id = np.array([1], dtype=np.int32)
  model.eq_obj2id = np.array([2], dtype=np.int32)

  treeids, counts, always, sensor_refs = lower_sensor_sleep_policy(model)
  assert not np.any(always)
  np.testing.assert_array_equal(counts[:, 0], [1, 1, 1, 1, 1, 1, 2, 1, 1, 2])
  np.testing.assert_array_equal(
      treeids[:, 0, :], [[0, -1], [1, -1], [0, -1], [0, -1], [0, -1],
                        [1, -1], [0, 1], [1, -1], [0, -1], [0, 1]])
  assert np.all(sensor_refs == -1)


def test_sensor_awake_mask_native_matches_pinned_cpu_oracle():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")
  from mujoco_metal.sensor_sleep import SensorSleepPolicy

  model = _Model()
  policy = SensorSleepPolicy(model, batch_size=3)
  tree_awake = torch.tensor(
      [[0, 0], [1, 0], [0, 1]], dtype=torch.int32, device="mps")
  expected = sensor_awake_mask_cpu(
      tree_awake.cpu().numpy(), *lower_sensor_sleep_policy(model))
  actual = policy.run_device(tree_awake).cpu().numpy()
  np.testing.assert_array_equal(actual, expected)

  referenced = _Model()
  referenced.nsensor = 3
  referenced.sensor_type = np.array([
      mujoco.mjtSensor.mjSENS_JOINTPOS,
      mujoco.mjtSensor.mjSENS_FRAMEPOS,
      mujoco.mjtSensor.mjSENS_FRAMEPOS,
  ], dtype=np.int32)
  referenced.sensor_objtype = np.array([
      mujoco.mjtObj.mjOBJ_JOINT,
      mujoco.mjtObj.mjOBJ_SENSOR,
      mujoco.mjtObj.mjOBJ_SENSOR,
  ], dtype=np.int32)
  referenced.sensor_objid = np.array([0, 0, 1], dtype=np.int32)
  referenced.sensor_reftype = np.full((3,), mujoco.mjtObj.mjOBJ_UNKNOWN,
                                      dtype=np.int32)
  referenced.sensor_refid = np.full((3,), -1, dtype=np.int32)
  reference_policy = SensorSleepPolicy(referenced, batch_size=3)
  expected = sensor_awake_mask_cpu(
      tree_awake.cpu().numpy(), *lower_sensor_sleep_policy(referenced))
  actual = reference_policy.run_device(tree_awake).cpu().numpy()
  np.testing.assert_array_equal(actual, expected)
