# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Topology/row-capacity qualification for fixed flex contact candidates."""

import mujoco
import numpy as np

from mujoco_metal.flex_contact import (
    _KIND_ELEMENT_PAIR_VERTEX,
    _KIND_GEOM_ELEMENT,
    _KIND_PLANE_VERTEX,
    lower_flex_contacts,
)


def _model(geom_xml, flex_xml, cone="pyramidal"):
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" cone="{cone}"/><worldbody>
      {geom_xml}
      {flex_xml}
    </worldbody></mujoco>
  """)


def _flex(name, pos, contype, conaffinity):
  return f"""
    <flexcomp name="{name}" type="grid" count="2 2 1"
              pos="{pos}" spacing=".1 .1 .1" mass="1" dim="2">
      <contact contype="{contype}" conaffinity="{conaffinity}"
               selfcollide="none"/>
      <edge stiffness="0" damping="0"/>
      <elasticity young="100" poisson=".2" thickness=".01"
                  elastic2d="stretch"/>
    </flexcomp>
  """


def test_plane_candidates_keep_simultaneous_obstacles_and_exact_row_spans():
  model = _model("""
    <geom name="floor_a" type="plane" size="0 0 .1"
          contype="0" conaffinity="1" condim="3" friction=".8 .02 .01"/>
    <geom name="floor_b" type="plane" pos="0 0 -.2" size="0 0 .1"
          contype="0" conaffinity="1" condim="4" friction=".6 .03 .02"/>
    <geom name="filtered" type="plane" pos="0 0 -.4" size="0 0 .1"
          contype="0" conaffinity="0"/>
  """, _flex("cloth", "0 0 .1", 1, 0))
  descriptor = lower_flex_contacts(model)

  nvert = int(model.flex_vertnum[0])
  assert descriptor.slot_count == 2 * nvert
  assert np.all(descriptor.kind == _KIND_PLANE_VERTEX)
  assert np.all(descriptor.geom[:nvert] == 0)
  assert np.all(descriptor.geom[nvert:] == 1)
  np.testing.assert_array_equal(
      descriptor.row_span,
      np.r_[np.full(nvert, 4), np.full(nvert, 6)])
  np.testing.assert_array_equal(
      descriptor.row_start,
      np.r_[np.arange(nvert) * 4,
            nvert * 4 + np.arange(nvert) * 6])
  assert descriptor.row_capacity == nvert * (4 + 6)
  np.testing.assert_allclose(descriptor.friction[0], [1., 1., .02, .01, .01])
  np.testing.assert_allclose(descriptor.friction[nvert], [1., 1., .03, .02, .02])


def test_elliptic_rows_follow_mixed_condim_without_overwriting_features():
  model = _model("""
    <geom name="floor" type="plane" size="0 0 .1"
          contype="0" conaffinity="1" condim="4"/>
  """, _flex("cloth", "0 0 .1", 1, 0), cone="elliptic")
  descriptor = lower_flex_contacts(model)
  nvert = int(model.flex_vertnum[0])
  assert descriptor.slot_count == nvert
  assert np.all(descriptor.condim == 4)
  assert np.all(descriptor.row_span == 4)
  np.testing.assert_array_equal(
      descriptor.row_start, np.arange(nvert, dtype=np.int32) * 4)


def test_cross_flex_candidates_retain_both_elements_and_side_vertex_ids():
  first = _flex("left", "0 0 .1", 1, 0)
  second = _flex("right", ".5 0 .1", 0, 1)
  model = _model("", first + second)
  descriptor = lower_flex_contacts(model)
  pair_mask = ((descriptor.kind == _KIND_ELEMENT_PAIR_VERTEX)
               & (descriptor.flex1 == 0) & (descriptor.flex2 == 1))
  pair_ids = np.flatnonzero(pair_mask)
  assert pair_ids.size == (2 * int(model.flex_elemnum[0])
                           * int(model.flex_elemnum[1]) * 3)
  assert np.all(descriptor.elem1[pair_ids] >= 0)
  assert np.all(descriptor.elem2[pair_ids] >= 0)
  assert np.all((descriptor.vert1[pair_ids] >= 0)
                ^ (descriptor.vert2[pair_ids] >= 0))
  assert np.all(descriptor.nodes1[pair_ids, :3] >= 0)
  assert np.all(descriptor.nodes2[pair_ids, :3] >= 0)
  second_vertadr = int(model.flex_vertadr[1])
  assert np.all(descriptor.vert2[pair_ids][descriptor.vert2[pair_ids] >= 0]
                >= second_vertadr)
  assert np.all(descriptor.nodes2[pair_ids, :3] >= second_vertadr)


def test_geom_element_slots_and_tree_link_expansion_retain_face_support():
  model = _model("""
    <body pos=".05 .05 .01"><freejoint/>
      <geom name="sphere" type="sphere" size=".08" condim="4"/>
    </body>
  """, _flex("cloth", "0 0 .1", 1, 0))
  descriptor = lower_flex_contacts(model)
  element = np.flatnonzero(descriptor.kind == _KIND_GEOM_ELEMENT)
  assert element.size == int(model.flex_elemnum[0])
  assert np.all(descriptor.vert1[element] == -1)
  assert np.all(descriptor.nodes1[element, :3] >= 0)
  assert descriptor.link_capacity > 0
  assert descriptor.candidate_link_ids.shape == (
      descriptor.slot_count, descriptor.max_links_per_candidate)
  assert np.all(descriptor.link_tree_pairs[:, 0]
                != descriptor.link_tree_pairs[:, 1])
  assert np.all(descriptor.link_tree_pairs[:, 0] >= 0)
  assert np.all(descriptor.link_tree_pairs[:, 1] >= 0)


def test_disabled_global_contacts_have_no_candidate_slots():
  model = _model("""
    <geom name="floor" type="plane" size="0 0 .1"
          contype="0" conaffinity="1"/>
  """, _flex("cloth", "0 0 .1", 1, 0))
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  descriptor = lower_flex_contacts(model)
  assert descriptor.slot_count > 0
  assert not descriptor.global_enabled
