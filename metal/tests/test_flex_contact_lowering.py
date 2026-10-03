# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Topology/row-capacity qualification for fixed flex contact candidates."""

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    _KIND_ELEMENT_PAIR,
    _KIND_GEOM_ELEMENT,
    _KIND_INTERNAL_VERTEX_ELEMENT,
    _KIND_PLANE_VERTEX,
    FlexContactProgram,
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


def _volume_flex(name, selfcollide, contype=1, conaffinity=0,
                 count="2 2 2"):
  return f"""
    <flexcomp name="{name}" type="grid" count="{count}"
              pos="0 0 .1" spacing=".1 .1 .1" mass="1" dim="3">
      <contact contype="{contype}" conaffinity="{conaffinity}"
               selfcollide="{selfcollide}"/>
      <edge stiffness="0" damping="0"/>
      <elasticity young="100" poisson=".2"/>
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


def test_cross_flex_candidates_retain_one_complete_simplex_pair():
  first = _flex("left", "0 0 .1", 1, 0)
  second = _flex("right", ".5 0 .1", 0, 1)
  model = _model("", first + second)
  descriptor = lower_flex_contacts(model)
  pair_mask = ((descriptor.kind == _KIND_ELEMENT_PAIR)
               & (descriptor.flex1 == 0) & (descriptor.flex2 == 1))
  pair_ids = np.flatnonzero(pair_mask)
  assert pair_ids.size == (int(model.flex_elemnum[0])
                           * int(model.flex_elemnum[1]))
  assert np.all(descriptor.elem1[pair_ids] >= 0)
  assert np.all(descriptor.elem2[pair_ids] >= 0)
  assert np.all(descriptor.nodes1[pair_ids, :3] >= 0)
  assert np.all(descriptor.nodes2[pair_ids, :3] >= 0)
  second_vertadr = int(model.flex_vertadr[1])
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
  for flag in (mujoco.mjtDisableBit.mjDSBL_CONTACT,
               mujoco.mjtDisableBit.mjDSBL_CONSTRAINT):
    model = _model("""
      <geom name="floor" type="plane" size="0 0 .1"
            contype="0" conaffinity="1"/>
    """, _flex("cloth", "0 0 .1", 1, 0))
    model.opt.disableflags |= int(flag)
    descriptor = lower_flex_contacts(model)
    assert descriptor.slot_count > 0
    assert not descriptor.global_enabled


def test_contact_margin_override_uses_pinned_assign_margin():
  model = _model(
      '<geom type="plane" size="0 0 .1" margin=".02" '
      'contype="0" conaffinity="1"/>', _flex("cloth", "0 0 .1", 1, 0))
  model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
  model.opt.o_margin = .037
  descriptor = lower_flex_contacts(model)
  np.testing.assert_allclose(descriptor.margin, .037)


def test_geom_flex_candidates_apply_bodyflex_and_geom_bitmasks():
  model = _model(
      '<body pos="0 0 .1"><freejoint/>'
      '<geom type="sphere" size=".2" contype="0" conaffinity="1"/>'
      '</body>', _flex("cloth", "0 0 .1", 1, 0))
  geom_body = int(model.geom_bodyid[0])
  # The geom-level directed mask would allow this pair. Pinned broadphase
  # first applies canCollide2 to the body's aggregate mask, so clearing it
  # must eliminate the candidate before the geom-level test.
  model.body_contype[geom_body] = 0
  model.body_conaffinity[geom_body] = 0
  descriptor = lower_flex_contacts(model)
  assert not np.any(descriptor.geom == 0)


def test_geom_candidates_include_inactive_tetrahedra_like_pinned_source():
  model = _model(
      '<geom type="sphere" size=".2" contype="0" conaffinity="1"/>',
      _volume_flex("volume", "none"))
  assert int(model.flex_dim[0]) == 3
  assert int(model.flex_elemnum[0]) > 0
  descriptor = lower_flex_contacts(model)
  assert np.count_nonzero(descriptor.kind == _KIND_GEOM_ELEMENT) == int(
      model.flex_elemnum[0])
  # Pinned geom/flex broadphase does not call mj_isElemActive; that filter is
  # used only by direct self-collision traversal.
  model.flex_activelayers[0] = 0
  still_collidable = lower_flex_contacts(model)
  assert np.count_nonzero(still_collidable.kind == _KIND_GEOM_ELEMENT) == int(
      model.flex_elemnum[0])


def test_self_collision_requires_matching_contype_and_conaffinity():
  model = _model(
      '', _volume_flex("volume", "narrow", contype=1, conaffinity=0))
  descriptor = lower_flex_contacts(model)
  assert not np.any(descriptor.kind == _KIND_ELEMENT_PAIR)
  assert descriptor.row_start.shape == (descriptor.slot_count,)
  assert descriptor.row_capacity == 0


def test_cross_flex_pairs_are_not_gated_by_flex_rigid_flag():
  model = _model("", _flex("left", "0 0 .1", 1, 0)
                  + _flex("right", ".5 0 .1", 0, 1))
  # The broadphase's flex/flex branch has no flex_rigid exclusion; only the
  # separate self/internal loop checks that flag.
  model.flex_rigid[0] = True
  descriptor = lower_flex_contacts(model)
  pair_mask = ((descriptor.kind == _KIND_ELEMENT_PAIR)
               & (descriptor.flex1 == 0) & (descriptor.flex2 == 1))
  assert np.any(pair_mask)


def test_direct_self_collision_filters_inactive_tetrahedra():
  model = _model(
      '', _volume_flex("volume", "narrow", contype=1, conaffinity=1,
                       count="3 3 3"))
  descriptor = lower_flex_contacts(model)
  assert np.any(descriptor.kind == _KIND_ELEMENT_PAIR)
  model.flex_activelayers[0] = 0
  inactive = lower_flex_contacts(model)
  assert not np.any(inactive.kind == _KIND_ELEMENT_PAIR)


def test_predefined_internal_vertex_pairs_keep_the_full_element():
  xml = _flex("cloth", "0 0 .1", 1, 1).replace(
      'selfcollide="none"', 'selfcollide="none" internal="true"')
  model = _model("", xml)
  assert bool(model.flex_internal[0])
  assert int(model.flex_evpairnum[0]) > 0
  descriptor = lower_flex_contacts(model)
  slots = np.flatnonzero(
      descriptor.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
  assert slots.size == int(model.flex_evpairnum[0])
  for slot in slots:
    elem = int(descriptor.elem1[slot])
    expected = np.asarray(model.flex_elem[
        int(model.flex_elemdataadr[0]) + elem*3:
        int(model.flex_elemdataadr[0]) + elem*3 + 3])
    np.testing.assert_array_equal(descriptor.nodes1[slot, :3], expected)
    assert descriptor.margin[slot] == 0.0
    assert descriptor.gap[slot] == 0.0


def test_enabled_nonplane_contacts_fail_closed_until_ccd_is_admitted():
  torch = pytest.importorskip("torch")
  model = _model(
      '<geom type="sphere" size=".2" contype="0" conaffinity="1"/>',
      _volume_flex("volume", "none"))
  program = FlexContactProgram(model, device="cpu")
  with pytest.raises(NotImplementedError, match="support/CCD/manifold"):
    program.run_device(
        torch.zeros((1, model.nflexvert, 3)),
        torch.zeros((1, model.ngeom, 3)),
        torch.tensor([[[1., 0., 0., 0.]]]))


def test_globally_disabled_contacts_return_zero_fixed_slots_without_dispatch():
  torch = pytest.importorskip("torch")
  model = _model(
      '<geom type="sphere" size=".2" contype="0" conaffinity="1"/>',
      _volume_flex("volume", "none"))
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  program = FlexContactProgram(model, device="cpu")
  result = program.run_device(
      torch.zeros((1, model.nflexvert, 3)),
      torch.zeros((1, model.ngeom, 3)),
      torch.tensor([[[1., 0., 0., 0.]]]))
  assert tuple(result["active"].shape) == (1, program.descriptor.slot_count)
  assert not torch.any(result["active"])
