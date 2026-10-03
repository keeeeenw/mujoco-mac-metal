"""Narrowphase active-slot links used by MuJoCo tree sleep scheduling."""

import numpy as np
import pytest

from mujoco_metal.active_contact_links import (
    active_contact_tree_links_cpu,
    equality_tree_links,
    equality_tree_ids,
    pair_tree_ids,
)


def test_only_active_narrowphase_slots_emit_stable_cross_tree_links():
  # Pair slot ranges: p0=[0,2), p1=[2,3), p2=[3,4), p3=[4,5).
  offsets = np.array([0, 2, 3, 4, 5], dtype=np.int32)
  trees = np.array([[0, 1], [1, 2], [2, 2], [-1, 0]], dtype=np.int32)
  flags = np.array([[0, 1, 0, 1, 1],
                    [0, 0, 1, 0, 1]], dtype=np.float32)
  links, overflow = active_contact_tree_links_cpu(flags, offsets, trees)
  np.testing.assert_array_equal(overflow, [0, 0])
  np.testing.assert_array_equal(
      links,
      [[[0, 1], [-1, -1], [-1, -1], [-1, -1]],
       [[-1, -1], [1, 2], [-1, -1], [-1, -1]]],
  )


def test_link_capacity_overflow_is_fail_open_not_truncated():
  flags = np.array([[1, 1, 0]], dtype=np.int32)
  offsets = np.array([0, 1, 2, 3], dtype=np.int32)
  trees = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
  links, overflow = active_contact_tree_links_cpu(
      flags, offsets, trees, capacity=1)
  np.testing.assert_array_equal(overflow, [1])
  np.testing.assert_array_equal(links, [[[-1, -1]]])


def test_zero_capacity_overflows_only_when_a_cross_tree_contact_is_active():
  offsets = np.array([0, 1, 2], dtype=np.int32)
  trees = np.array([[0, 0], [0, 1]], dtype=np.int32)
  empty, no_overflow = active_contact_tree_links_cpu(
      np.array([[0, 0]], dtype=np.int32), offsets, trees, capacity=0)
  assert empty.shape == (1, 0, 2)
  np.testing.assert_array_equal(no_overflow, [0])
  _, overflow = active_contact_tree_links_cpu(
      np.array([[0, 1]], dtype=np.int32), offsets, trees, capacity=0)
  np.testing.assert_array_equal(overflow, [1])


def test_active_connect_and_weld_equalities_join_distinct_trees():
  offsets = np.array([0], dtype=np.int32)
  pair_trees = np.zeros((0, 2), dtype=np.int32)
  equality_trees = np.array([[0, 1], [1, 2], [3, 3], [-1, 4]], dtype=np.int32)
  # Connect/weld activity is world-local; invalid and same-tree descriptors
  # never create scheduler edges.
  active = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int32)
  links, overflow = active_contact_tree_links_cpu(
      np.zeros((2, 0), dtype=np.int32), offsets, pair_trees,
      equality_trees=equality_trees, equality_active=active)
  np.testing.assert_array_equal(overflow, [0, 0])
  np.testing.assert_array_equal(
      links,
      [[[0, 1], [-1, -1], [-1, -1], [-1, -1]],
       [[-1, -1], [1, 2], [-1, -1], [-1, -1]]])


def test_equality_overflow_is_fail_open_and_nonfinite_activity_is_rejected():
  _, overflow = active_contact_tree_links_cpu(
      np.zeros((1, 0), dtype=np.int32), [0], np.zeros((0, 2), dtype=np.int32),
      capacity=0, equality_trees=[[0, 1]], equality_active=[[1]])
  np.testing.assert_array_equal(overflow, [1])
  with pytest.raises(ValueError, match="finite"):
    active_contact_tree_links_cpu(
        np.array([[np.nan]], dtype=np.float32), [0, 1], [[0, 1]])


def test_active_hyperedge_equality_fails_open_and_never_drops_sleep_links():
  links, overflow = active_contact_tree_links_cpu(
      np.zeros((1, 0), dtype=np.int32), [0], np.zeros((0, 2), dtype=np.int32),
      capacity=1, equality_trees=[[-2, -2], [1, 1]],
      equality_active=[[1, 1]])
  np.testing.assert_array_equal(overflow, [1])
  np.testing.assert_array_equal(links, [[[-1, -1]]])


def test_equality_tree_ids_cover_body_site_joint_and_mark_hyperedges():
  mujoco = pytest.importorskip("mujoco")

  class Model:
    neq = 5
    body_treeid = np.array([-1, 0, 1, 2], dtype=np.int32)
    eq_type = np.array([
        mujoco.mjtEq.mjEQ_CONNECT,
        mujoco.mjtEq.mjEQ_WELD,
        mujoco.mjtEq.mjEQ_JOINT,
        mujoco.mjtEq.mjEQ_TENDON,
        mujoco.mjtEq.mjEQ_FLEX,
    ], dtype=np.int32)
    eq_objtype = np.array([
        mujoco.mjtObj.mjOBJ_BODY,
        mujoco.mjtObj.mjOBJ_SITE,
        mujoco.mjtObj.mjOBJ_JOINT,
        mujoco.mjtObj.mjOBJ_TENDON,
        mujoco.mjtObj.mjOBJ_FLEX,
    ], dtype=np.int32)
    eq_obj1id = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    eq_obj2id = np.array([2, 1, 1, -1, -1], dtype=np.int32)
    site_bodyid = np.array([1, 2], dtype=np.int32)
    jnt_bodyid = np.array([1, 2], dtype=np.int32)

  np.testing.assert_array_equal(
      equality_tree_ids(Model()),
      [[0, 1], [0, 1], [0, 1], [-2, -2], [-2, -2]])


def test_flex_equality_expands_all_dynamic_tree_members_as_one_star():
  mujoco = pytest.importorskip("mujoco")

  class Model:
    neq = 1
    nflex = 1
    body_treeid = np.array([-1, 0, 1, 2], dtype=np.int32)
    eq_type = np.array([mujoco.mjtEq.mjEQ_FLEX], dtype=np.int32)
    eq_objtype = np.array([mujoco.mjtObj.mjOBJ_FLEX], dtype=np.int32)
    eq_obj1id = np.array([0], dtype=np.int32)
    eq_obj2id = np.array([-1], dtype=np.int32)
    flex_interp = np.array([0], dtype=np.int32)
    flex_vertadr = np.array([0], dtype=np.int32)
    flex_vertnum = np.array([3], dtype=np.int32)
    flex_vertbodyid = np.array([1, 2, 3], dtype=np.int32)
    flex_nodeadr = np.array([0], dtype=np.int32)
    flex_nodenum = np.array([0], dtype=np.int32)
    flex_nodebodyid = np.zeros((0,), dtype=np.int32)

  links, owners = equality_tree_links(Model())
  np.testing.assert_array_equal(links, [[0, 1], [0, 2]])
  np.testing.assert_array_equal(owners, [0, 0])
  mapped, overflow = active_contact_tree_links_cpu(
      np.zeros((1, 0), dtype=np.int32), [0], np.zeros((0, 2), dtype=np.int32),
      capacity=2, equality_trees=links, equality_activity_ids=owners,
      equality_active=[[1]])
  np.testing.assert_array_equal(mapped, [[[0, 1], [0, 2]]])
  np.testing.assert_array_equal(overflow, [0])


def test_invalid_slot_partition_rejected():
  with pytest.raises(ValueError, match="partition all logical slots"):
    active_contact_tree_links_cpu(
        np.zeros((1, 2), dtype=np.int32), [0, 1], [[0, 1]])


def test_pair_tree_ids_follow_compiled_geom_body_and_tree_maps():
  class Model:
    geom_bodyid = np.array([0, 1, 2, 3], dtype=np.int32)
    body_treeid = np.array([-1, 0, 1, 1], dtype=np.int32)

  np.testing.assert_array_equal(
      pair_tree_ids(Model(), [[0, 1], [1, 2], [2, 3]]),
      [[-1, 0], [0, 1], [1, 1]])


def test_mps_link_map_native_batch_local():
  torch = pytest.importorskip("torch")
  from mujoco_metal.row_compaction import compact_flags
  if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
    pytest.skip("requires native MPS shader execution")
  flags = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0]],
                       dtype=torch.float32, device="mps")
  slots = compact_flags(flags, capacity=4)
  # Pair ranges [0,2), [2,3), [3,4) with pair trees 0-1, 1-2, 2-3.
  from mujoco_metal.active_contact_links import ActiveContactLinkWorkspace
  workspace = ActiveContactLinkWorkspace(
      2, [0, 2, 3, 4], [[0, 1], [1, 2], [2, 3]], capacity=3)
  links, overflow = workspace.run(slots)
  torch.mps.synchronize()
  np.testing.assert_array_equal(
      links.cpu().numpy(),
      [[[0, 1], [-1, -1], [2, 3]],
       [[-1, -1], [1, 2], [-1, -1]]])
  np.testing.assert_array_equal(overflow.cpu().numpy(), [0, 0])
