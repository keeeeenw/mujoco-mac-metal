# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Topology/row-capacity qualification for fixed flex contact candidates."""

import os
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from mujoco_metal.flex_contact import (
    _candidate_filter_modes,
    _KIND_ELEMENT_PAIR,
    _KIND_GEOM_ELEMENT,
    _KIND_INTERNAL_VERTEX_ELEMENT,
    _KIND_PLANE_VERTEX,
    _flex_mesh_hull,
    _flex_bvh_element_order,
    _validate_contact_program_capacity,
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


def _position_after_filter_swaps(source_position, selected_positions):
  """Map an original contact index through filterFlexContacts swaps."""
  position = int(source_position)
  for pick, selected_position in enumerate(selected_positions):
    selected_position = int(selected_position)
    if position == pick:
      position = selected_position
    elif position == selected_position:
      position = pick
  return position


def test_filter_position_replay_applies_both_directions_of_source_swap():
  """Pinned FPS selection swaps both array occupants at every pick."""
  original = list(range(7))
  selected_positions = (5, 2, 6)
  for pick, selected_position in enumerate(selected_positions):
    original[pick], original[selected_position] = (
        original[selected_position], original[pick])

  for source_position, identity in enumerate(range(7)):
    assert original[_position_after_filter_swaps(
        source_position, selected_positions)] == identity
  assert _position_after_filter_swaps(5, selected_positions) == 0


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


def _assert_cpu_geom_flex_contacts_covered(model, descriptor, geom_id):
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  contacts = [contact for contact in data.contact[:data.ncon]
              if int(contact.geom[0]) == geom_id and int(contact.flex[1]) >= 0]
  assert contacts
  for contact in contacts:
    elem = int(contact.elem[1])
    assert np.any((descriptor.kind == _KIND_GEOM_ELEMENT)
                  & (descriptor.geom == geom_id)
                  & (descriptor.flex1 == int(contact.flex[1]))
                  & (descriptor.elem1 == elem))


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
  assert np.all(descriptor.filterable)
  assert np.all(descriptor.geom[:nvert] == 0)
  assert np.all(descriptor.geom[nvert:] == 1)
  body = int(model.geom_bodyid[0])
  midphase = (not (int(model.opt.disableflags)
                   & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE))
              and int(model.body_bvhadr[body]) >= 0
              and int(model.flex_bvhadr[0]) >= 0)
  assert descriptor.filter_group_count == (1 if midphase else 2)
  assert np.unique(descriptor.filter_group[:nvert]).size == 1
  assert np.unique(descriptor.filter_group[nvert:]).size == 1
  assert (descriptor.filter_group[0] == descriptor.filter_group[nvert]) == midphase
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


def test_zero_candidate_runtime_reuses_typed_empty_workspaces():
  torch = pytest.importorskip("torch")
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body name='static'/></worldbody></mujoco>")
  program = FlexContactProgram(model, batch_size=2, device="cpu")
  flexvert_xpos = torch.empty((2, 0, 3), dtype=torch.float32)
  geom_pos = torch.empty((2, 0, 3), dtype=torch.float32)
  geom_quat = torch.empty((2, 0, 4), dtype=torch.float32)
  first = program.run_device(flexvert_xpos, geom_pos, geom_quat)
  second = program.run_device(flexvert_xpos, geom_pos, geom_quat)
  assert first["active"].dtype == torch.bool
  assert first["raw_active"].dtype == torch.bool
  assert first["active"].shape == (2, 0)
  assert first["frame"].shape == (2, 0, 3, 3)
  for name in ("active", "raw_active", "dist", "pos", "normal", "frame",
               "barycentric1", "barycentric2"):
    assert first[name] is second[name]


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


def test_cpu_contact_cap_and_group_route_follow_midphase_selection():
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom type="plane" size="0 0 .1" contype="0" conaffinity="1"/>
      <geom type="plane" pos="0 0 .2" size="0 0 .1"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="8 8 1" pos="0 0 -.1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  records = {}
  for midphase in (False, True):
    model = mujoco.MjModel.from_xml_string(xml)
    if not midphase:
      model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    descriptor = lower_flex_contacts(model)
    body = int(model.geom_bodyid[0])
    can_use_tree = (int(model.body_bvhadr[body]) >= 0
                    and int(model.flex_bvhadr[0]) >= 0)
    expected_groups = 1 if midphase and can_use_tree else 2
    assert descriptor.filter_group_count == expected_groups
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    identities = [(tuple(int(x) for x in c.geom),
                   tuple(int(x) for x in c.flex),
                   tuple(int(x) for x in c.elem)) for c in data.contact]
    records[midphase] = identities
    counts = {geom: sum(1 for c in data.contact if int(c.geom[0]) == geom)
              for geom in (0, 1)}
    if midphase and can_use_tree:
      assert data.ncon == 50
      assert 0 < counts[0] < 50 and 0 < counts[1] < 50
      assert [int(c.geom[0]) for c in data.contact] == (
          [0] * counts[0] + [1] * counts[1])
    else:
      assert counts == {0: 50, 1: 50}
      assert [int(c.geom[0]) for c in data.contact] == [0] * 50 + [1] * 50
    repeat = mujoco.MjData(model)
    mujoco.mj_forward(model, repeat)
    assert identities == [(tuple(int(x) for x in c.geom),
                           tuple(int(x) for x in c.flex),
                           tuple(int(x) for x in c.elem)) for c in repeat.contact]
  assert len(records[True]) == 50
  assert len(records[False]) == 100


def _split_source_mjt_num(values):
  """Split source binary64 inputs into the detector's high/low/tail ABI."""
  source = np.asarray(values, dtype=np.float64)
  high = source.astype(np.float32)
  remainder = source - high.astype(np.float64)
  low = remainder.astype(np.float32)
  tail = (remainder - low.astype(np.float64)).astype(np.float32)
  reconstructed = (high.astype(np.float64) + low.astype(np.float64)
                   + tail.astype(np.float64))
  scale = max(1.0, float(np.max(np.abs(source), initial=0.0)))
  assert np.all(np.isfinite(reconstructed))
  assert np.max(np.abs(reconstructed-source), initial=0.0) <= (
      np.finfo(np.float64).eps * scale)
  return high, low, tail


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native flex contact cap requires explicit GPU opt-in")
@pytest.mark.parametrize("midphase", [False, True])
def test_native_plane_contact_filter_matches_pinned_cpu_contact_identity_set(
    midphase):
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native flex contact filter requires MPS")
  from mujoco_metal.flex_contact import FlexContactProgram

  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom type="plane" size="0 0 .1" contype="0" conaffinity="1"/>
      <geom type="plane" pos="0 0 .2" size="0 0 .1"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="8 8 1" pos="0 0 -.1"
                spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  if not midphase:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  descriptor = lower_flex_contacts(model)
  geom_body = int(model.geom_bodyid[0])
  can_use_tree = (int(model.body_bvhadr[geom_body]) >= 0
                  and int(model.flex_bvhadr[0]) >= 0)
  expected_group_count = 1 if midphase and can_use_tree else 2
  assert descriptor.filter_group_count == expected_group_count
  expected = {(int(c.geom[0]), int(c.vert[1]))
              for c in data.contact[:data.ncon]}
  assert len(expected) == data.ncon
  geom_quat = np.empty((int(model.ngeom), 4), np.float32)
  for geom in range(int(model.ngeom)):
    quat64 = np.empty(4, np.float64)
    mujoco.mju_mat2Quat(
        quat64, np.asarray(data.geom_xmat[geom], np.float64).reshape(9))
    geom_quat[geom] = quat64.astype(np.float32)
  mps = lambda values: torch.as_tensor(
      np.asarray(values, np.float32)[None].copy(), dtype=torch.float32,
      device="mps").contiguous()
  program = FlexContactProgram(model, device="mps")
  assert program._narrowphase_admitted
  # This oracle retains CPU binary64 vertices and geometry positions. Pass
  # their residual words too: narrowing only the native input changes tied
  # farthest-point selections even when the contact count stays unchanged.
  # Raw high-only input parity is tested separately against represented CPU
  # inputs in test_flex_fps_input_contract_018.py.
  def source_words(values):
    source = np.asarray(values, np.float64)
    high = source.astype(np.float32)
    low = (source - high.astype(np.float64)).astype(np.float32)
    tail = (source - high.astype(np.float64)
            - low.astype(np.float64)).astype(np.float32)
    return mps(high), mps(low), mps(tail)

  vertex_high, vertex_low, vertex_tail = source_words(data.flexvert_xpos)
  geom_high, geom_low, geom_tail = source_words(data.geom_xpos)
  result = program.run_device(
      vertex_high, geom_high, mps(geom_quat),
      flexvert_xpos_low=vertex_low, flexvert_xpos_tail=vertex_tail,
      geom_pos_low=geom_low, geom_pos_tail=geom_tail)
  active = result["active"].cpu().numpy()[0]
  expected_feature = ((descriptor.kind == _KIND_PLANE_VERTEX)
                      & (descriptor.geom >= 0))
  actual = {(int(descriptor.geom[slot]), int(descriptor.vert1[slot]))
            for slot in np.flatnonzero(active & expected_feature)}
  assert actual == expected, (midphase, len(actual), len(expected),
                              sorted(actual ^ expected))


@pytest.mark.parametrize("missing_bvh", ["body", "flex"])
def test_midphase_filter_group_requires_both_compiled_bvhs(missing_bvh):
  model = _model(
      '<body><geom type="sphere" size=".2"/></body>',
      _flex("cloth", "0 0 .1", 1, 0))
  body = int(model.geom_bodyid[0])
  assert int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE) == 0
  assert int(model.body_bvhadr[body]) >= 0
  assert int(model.flex_bvhadr[0]) >= 0
  if missing_bvh == "body":
    model.body_bvhadr[body] = -1
  else:
    model.flex_bvhadr[0] = -1
  descriptor = lower_flex_contacts(model)
  assert descriptor.filter_group_count == 1


def test_capsule_pair_lowering_reserves_each_pinned_raw_witness():
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <flexcomp name="a" type="grid" count="2 1 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" dim="1">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
      <flexcomp name="b" type="grid" count="2 1 1" pos="0 .1 0"
                spacing=".1 .1 .1" mass="1" dim="1">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  assert descriptor.slot_count == 4
  np.testing.assert_array_equal(descriptor.flex1, np.zeros(4, np.int32))
  np.testing.assert_array_equal(descriptor.flex2, np.ones(4, np.int32))
  np.testing.assert_array_equal(descriptor.contact_ordinal, np.arange(4))
  np.testing.assert_array_equal(descriptor.row_start, np.arange(4) * 4)
  assert np.all(descriptor.row_span == 4)


def test_flex_capsule_vs_geom_capsule_reserves_raw_manifold_capacity():
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="obstacle" pos="0 .04 0">
        <geom name="capsule" type="capsule" size=".04 .1"
              contype="0" conaffinity="1"/>
      </body>
      <flexcomp name="line" type="grid" count="2 1 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" dim="1">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  capsule = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "capsule"))
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == capsule))
  assert slots.size == 4
  np.testing.assert_array_equal(descriptor.contact_ordinal[slots], np.arange(4))
  np.testing.assert_array_equal(descriptor.elem1[slots], np.zeros(4, np.int32))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert 0 < data.ncon <= 4
  assert any(int(contact.geom[0]) == capsule
             and int(contact.flex[1]) == 0
             and int(contact.elem[1]) == 0
             for contact in data.contact[:data.ncon])
  _assert_cpu_geom_flex_contacts_covered(model, descriptor, capsule)


def test_geom_capsule_triangle_candidates_reserve_five_raw_ordinals():
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="obstacle" pos="-.05 -.05 0">
        <geom name="capsule" type="capsule" size=".04 .1"
              contype="0" conaffinity="1"/>
      </body>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" radius=".01" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  capsule = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "capsule"))
  for elem in range(int(model.flex_elemnum[0])):
    slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                           & (descriptor.geom == capsule)
                           & (descriptor.elem1 == elem))
    assert slots.size == 5
    np.testing.assert_array_equal(descriptor.contact_ordinal[slots],
                                  np.arange(5))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  _assert_cpu_geom_flex_contacts_covered(model, descriptor, capsule)


def test_geom_box_capsule_candidates_reserve_two_raw_ordinals():
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="obstacle" pos="0 .04 0">
        <geom name="box" type="box" size=".04 .04 .04"
              contype="0" conaffinity="1"/>
      </body>
      <flexcomp name="line" type="grid" count="2 1 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" radius=".01" dim="1">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  box = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box"))
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == box))
  assert slots.size == 2
  np.testing.assert_array_equal(descriptor.contact_ordinal[slots], np.arange(2))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  _assert_cpu_geom_flex_contacts_covered(model, descriptor, box)


def test_sphere_capsule_candidate_uses_single_raw_slot():
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="obstacle" pos="0 .04 0">
        <geom name="sphere" type="sphere" size=".04"
              contype="0" conaffinity="1"/>
      </body>
      <flexcomp name="line" type="grid" count="2 1 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" radius=".01" dim="1">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  sphere = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "sphere"))
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == sphere))
  assert slots.size == 1
  assert descriptor.contact_ordinal[slots[0]] == 0
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon == 1
  assert int(data.contact[0].geom[0]) == sphere
  _assert_cpu_geom_flex_contacts_covered(model, descriptor, sphere)


def test_geom_box_triangle_candidates_reserve_eleven_raw_ordinals():
  xml = """
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="obstacle" pos="-.05 -.05 0">
        <geom name="box" type="box" size=".04 .04 .04"
              contype="0" conaffinity="1"/>
      </body>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 0"
                spacing=".1 .1 .1" mass="1" radius=".01" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  box = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box"))
  for elem in range(int(model.flex_elemnum[0])):
    slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                           & (descriptor.geom == box)
                           & (descriptor.elem1 == elem))
    assert slots.size == 11
    np.testing.assert_array_equal(descriptor.contact_ordinal[slots],
                                  np.arange(11))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  _assert_cpu_geom_flex_contacts_covered(model, descriptor, box)


@pytest.mark.parametrize("kind,size", [
    ("ellipsoid", ".04 .05 .04"),
    ("cylinder", ".04 .05"),
])
def test_generic_convex_geom_families_lower_complete_flex_elements(kind, size):
  model = _model(
      f'<geom name="obstacle" type="{kind}" pos="0 .04 0" '
      f'size="{size}" contype="0" conaffinity="1"/>',
      _flex("cloth", "0 0 0", 1, 0))
  descriptor = lower_flex_contacts(model)
  geom = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle"))
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == geom))
  assert slots.size == int(model.flex_elemnum[0])
  assert np.all(descriptor.contact_ordinal[slots] == 0)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  for contact in data.contact[:data.ncon]:
    elem = int(contact.elem[1])
    assert np.any((descriptor.kind == _KIND_GEOM_ELEMENT)
                  & (descriptor.geom == geom)
                  & (descriptor.flex1 == int(contact.flex[1]))
                  & (descriptor.elem1 == elem))


def test_mesh_geom_flex_candidates_pack_compiled_convex_hull():
  pytest.importorskip("torch")
  xml = """<mujoco><option gravity="0 0 0"/><asset>
    <mesh name="tet" vertex="-.1 -.1 -.1 .1 -.1 -.1 0 .1 -.1 0 0 .1"
          face="0 2 1 0 1 3 1 2 3 2 0 3"/>
    </asset><worldbody>
      <geom name="mesh" type="mesh" mesh="tet"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1"
                pos="0 0 0" spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  for contact in data.contact[:data.ncon]:
    elem = int(contact.elem[1])
    assert np.any((descriptor.kind == _KIND_GEOM_ELEMENT)
                  & (descriptor.geom == int(contact.geom[0]))
                  & (descriptor.elem1 == elem))
  program = FlexContactProgram(model, device="cpu")
  geom = int(contact.geom[0])
  offset, count = map(
      int, program._geom_hull_info.reshape(-1, 9)[geom, :2].tolist())
  mesh = int(model.geom_dataid[geom])
  adr = int(model.mesh_vertadr[mesh])
  expected = np.asarray(model.mesh_vert[
      adr:adr + int(model.mesh_vertnum[mesh])], np.float32)
  actual = program._geom_hull[offset*3:(offset + count)*3].numpy().reshape(-1, 3)
  np.testing.assert_array_equal(actual, expected)


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


def test_hfield_element_slots_preserve_raw_cap_and_compiled_payload():
  torch = pytest.importorskip("torch")
  xml = """<mujoco><asset>
    <hfield name="terrain" nrow="3" ncol="3" size="1 1 .2 .1"
            elevation="0 0 0 0 0 0 0 0 0"/>
    </asset><worldbody>
      <geom name="terrain" type="hfield" hfield="terrain"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1"
                pos="0 0 -.005" spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  mask = descriptor.kind == _KIND_GEOM_ELEMENT
  assert np.any(mask)
  assert np.all(descriptor.geom[mask] == 0)
  assert np.unique(descriptor.contact_ordinal[mask]).size == 50
  assert set(descriptor.contact_ordinal[mask]) == set(range(50))
  for elem in np.unique(descriptor.elem1[mask]):
    rows = np.flatnonzero(mask & (descriptor.elem1 == elem))
    np.testing.assert_array_equal(descriptor.contact_ordinal[rows],
                                  np.arange(50, dtype=np.int32))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert data.ncon > 0
  for contact in data.contact[:data.ncon]:
    elem = int(contact.elem[1])
    assert np.any(mask & (descriptor.geom == 0)
                  & (descriptor.flex1 == int(contact.flex[1]))
                  & (descriptor.elem1 == elem))
  program = FlexContactProgram(model, device="cpu")
  payload = program._geom_hull_info.reshape(-1, 9)[0]
  hfield = int(model.geom_dataid[0])
  assert payload[5] >= 0
  assert payload[6:8].tolist() == [int(model.hfield_nrow[hfield]),
                                   int(model.hfield_ncol[hfield])]
  data_offset = int(payload[5])
  data_count = int(payload[6] * payload[7])
  data_adr = int(model.hfield_adr[hfield])
  np.testing.assert_array_equal(
      program._geom_hull[data_offset:data_offset + data_count].numpy(),
      model.hfield_data[data_adr:data_adr + data_count])
  size_offset = int(payload[8])
  np.testing.assert_allclose(
      program._geom_hull[size_offset:size_offset + 4].numpy(),
      model.hfield_size[hfield], rtol=1e-7, atol=1e-7)


def test_mixed_mesh_and_hfield_payloads_keep_distinct_index_units():
  pytest.importorskip("torch")
  xml = """<mujoco><asset>
    <hfield name="terrain" nrow="2" ncol="2" size="1 1 .2 .1"
            elevation="0 0 0 0"/>
    <mesh name="block" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"
          face="0 2 1 0 1 3 1 2 3 2 0 3"/>
    </asset><worldbody>
      <geom name="terrain" type="hfield" hfield="terrain"
            contype="0" conaffinity="1"/>
      <geom name="block" type="mesh" mesh="block" pos="0 0 .2"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1"
                pos="0 0 .05" spacing=".1 .1 .1" mass="1" dim="2">
        <contact contype="1" conaffinity="0" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>"""
  model = mujoco.MjModel.from_xml_string(xml)
  descriptor = lower_flex_contacts(model)
  program = FlexContactProgram(model, device="cpu")
  info = program._geom_hull_info.reshape(-1, 9).numpy()
  hfield = int(model.geom_dataid[0])
  assert info[0, 5] == 0
  assert info[0, 8] == int(model.hfield_nrow[hfield]
                           * model.hfield_ncol[hfield])
  mesh_id = int(model.geom_dataid[1])
  mesh_offset, mesh_count = map(int, info[1, :2])
  assert mesh_offset * 3 >= info[0, 8] + 4
  mesh_adr = int(model.mesh_vertadr[mesh_id])
  expected_mesh = np.asarray(model.mesh_vert[
      mesh_adr:mesh_adr + int(model.mesh_vertnum[mesh_id])], np.float32)
  actual_mesh = program._geom_hull[
      mesh_offset*3:(mesh_offset + mesh_count)*3].numpy().reshape(-1, 3)
  np.testing.assert_array_equal(actual_mesh, expected_mesh)
  assert descriptor.slot_count > 0


def test_flex_sdf_payload_matches_compiled_plugin_free_octree():
  xml = """
    <mujoco><asset>
      <mesh name="sdfbox" vertex="-.05 -.05 -.05 .05 -.05 -.05
          .05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05
          .05 .05 .05 -.05 .05 .05"/>
    </asset><option gravity="0 0 0" sdf_initpoints="4"/>
    <worldbody>
      <geom name="sdf" type="sdf" mesh="sdfbox"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 .04"
                spacing=".04 .04 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  geom = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "sdf"))
  mesh = int(model.geom_dataid[geom])
  assert int(model.geom_plugin[geom]) == -1
  assert int(model.mesh_octadr[mesh]) >= 0
  descriptor = lower_flex_contacts(model)
  sdf_slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                             & (descriptor.geom == geom))
  assert int(model.flex_bvhadr[0]) >= 0
  assert sdf_slots.size == int(model.flex_elemnum[0])
  np.testing.assert_array_equal(descriptor.contact_ordinal[sdf_slots], -1)
  active_elements = set(int(e) for e in descriptor.elem1[
      (descriptor.kind == _KIND_GEOM_ELEMENT) & (descriptor.geom == geom)])
  expected_order = [e for e in _flex_bvh_element_order(
      model, 0, int(model.flex_elemnum[0])) if e in active_elements]
  actual_order = descriptor.elem1[
      (descriptor.kind == _KIND_GEOM_ELEMENT) & (descriptor.geom == geom)]
  np.testing.assert_array_equal(actual_order, expected_order)
  assert np.any((descriptor.kind == _KIND_GEOM_ELEMENT)
                & (descriptor.geom == geom))
  modes = _candidate_filter_modes(model, descriptor)
  np.testing.assert_array_equal(modes[sdf_slots], 2)
  payload, raw_info = _flex_mesh_hull(model, descriptor)
  info = raw_info.reshape(model.ngeom, 9)
  offset = int(info[geom, 0])
  count = int(info[geom, 1])
  assert count == int(model.mesh_octnum[mesh])
  assert int(info[geom, 2]) == int(model.opt.sdf_iterations)
  assert int(info[geom, 3]) >= offset + 22 * count
  oct_adr = int(model.mesh_octadr[mesh])
  np.testing.assert_array_equal(
      payload[offset:offset + 8 * count],
      np.asarray(model.oct_child[oct_adr:oct_adr + count], np.float32).reshape(-1))
  np.testing.assert_array_equal(
      payload[offset + 8 * count:offset + 14 * count],
      np.asarray(model.oct_aabb[oct_adr:oct_adr + count], np.float32).reshape(-1))
  np.testing.assert_array_equal(
      payload[offset + 14 * count:offset + 22 * count],
      np.asarray(model.oct_coeff[oct_adr:oct_adr + count], np.float32).reshape(-1))
  aabb_offset = int(info[geom, 3]) + 6 * geom
  np.testing.assert_array_equal(
      payload[aabb_offset:aabb_offset + 6],
      np.asarray(model.geom_aabb[geom], np.float32))
  # The upstream no-BVH route emits each penetrating Halton seed rather than
  # reducing to one deepest candidate per flex element.
  model.flex_bvhadr[0] = -1
  direct_descriptor = lower_flex_contacts(model)
  direct_slots = np.flatnonzero((direct_descriptor.kind == _KIND_GEOM_ELEMENT)
                                & (direct_descriptor.geom == geom))
  np.testing.assert_array_equal(
      _candidate_filter_modes(model, direct_descriptor)[direct_slots], 3)
  initpoints = min(max(1, int(model.opt.sdf_initpoints)), 50)
  assert direct_slots.size == int(model.flex_elemnum[0]) * initpoints
  expected_ordinals = np.tile(np.arange(initpoints, dtype=np.int32),
                              int(model.flex_elemnum[0]))
  np.testing.assert_array_equal(
      direct_descriptor.contact_ordinal[direct_slots], expected_ordinals)


def test_flex_plugin_sdf_payload_uses_bundled_instance_attributes():
  """Lower plugin-backed SDF candidates without treating them as mesh octrees."""
  from mujoco_metal.bundled_plugins import lower_bundled_plugins

  model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <extension>
        <plugin plugin="mujoco.sdf.torus">
          <instance name="torus"/>
        </plugin>
      </extension>
      <asset>
        <mesh name="torus_mesh"><plugin instance="torus"/></mesh>
      </asset>
      <option gravity="0 0 0" sdf_initpoints="4"/>
      <worldbody>
        <geom name="torus_geom" type="sdf" mesh="torus_mesh"
              contype="0" conaffinity="1">
          <plugin instance="torus"/>
        </geom>
        <flexcomp name="cloth" type="grid" count="2 2 1"
                  pos="0 0 .04" spacing=".04 .04 .01" mass="1" dim="2">
          <contact contype="1" conaffinity="1" selfcollide="none"/>
          <edge stiffness="0" damping="0"/>
          <elasticity young="100" poisson=".2" thickness=".01"
                      elastic2d="stretch"/>
        </flexcomp>
      </worldbody>
    </mujoco>
  """)
  geom = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "torus_geom"))
  descriptor = lower_flex_contacts(model)
  slots = np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                         & (descriptor.geom == geom))
  assert slots.size > 0

  plugin = lower_bundled_plugins(model)
  assert int(plugin.geom_plugin_instance[geom]) == 0
  assert int(plugin.instance_kind[0]) == 5
  payload, raw_info = _flex_mesh_hull(model, descriptor)
  info = raw_info.reshape(model.ngeom, 9)[geom]
  offset = int(info[0])
  assert int(info[1]) == -2
  assert int(info[2]) == int(model.opt.sdf_iterations)
  assert int(info[3]) == 5
  assert int(info[4]) == 5
  np.testing.assert_array_equal(
      payload[offset:offset + 5], plugin.plugin_attributes[0])


def test_two_sdf_geom_cpu_contacts_use_route_specific_filter_groups():
  xml = """
    <mujoco><asset>
      <mesh name="sdfbox" vertex="-.05 -.05 -.05 .05 -.05 -.05
          .05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05
          .05 .05 .05 -.05 .05 .05"/>
    </asset><option gravity="0 0 0" sdf_initpoints="4"/>
    <worldbody>
      <geom name="sdf0" type="sdf" mesh="sdfbox"
            contype="0" conaffinity="1"/>
      <geom name="sdf1" type="sdf" mesh="sdfbox" pos=".04 0 0"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="5 5 1" pos="0 0 .04"
                spacing=".02 .02 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  for midphase in (True, False):
    model = mujoco.MjModel.from_xml_string(xml)
    if not midphase:
      model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    descriptor = lower_flex_contacts(model)
    assert data.ncon > 16
    counts = [sum(int(c.geom[0]) == geom
                  for c in data.contact[:data.ncon]) for geom in (0, 1)]
    assert all(count > 8 for count in counts)
    expected_groups = (1 if midphase
                       and int(model.body_bvhadr[0]) >= 0
                       and int(model.flex_bvhadr[0]) >= 0 else 2)
    assert descriptor.filter_group_count == expected_groups
    slots = [np.flatnonzero((descriptor.kind == _KIND_GEOM_ELEMENT)
                            & (descriptor.geom == geom)) for geom in (0, 1)]
    groups = [np.unique(descriptor.filter_group[indices]) for indices in slots]
    if expected_groups == 1:
      np.testing.assert_array_equal(groups[0], groups[1])
    else:
      assert groups[0][0] != groups[1][0]
    assert [int(c.geom[0]) for c in data.contact[:data.ncon]] == (
        [0] * counts[0] + [1] * counts[1])
    repeated = mujoco.MjData(model)
    mujoco.mj_forward(model, repeated)
    identities = lambda d: [
        (tuple(int(x) for x in c.geom), tuple(int(x) for x in c.flex),
         tuple(int(x) for x in c.elem)) for c in d.contact[:d.ncon]]
    assert identities(data) == identities(repeated)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="set MUJOCO_METAL_RUN_GPU=1 for native Metal")
@pytest.mark.parametrize(("orientation", "quat", "no_flex_bvh"), [
    pytest.param("combined_xy", "0.9292857777 0.1890653283 "
                 "0.3109346717 -0.0632603739", False,
                 id="combined-xy-bvh"),
    pytest.param("combined_xy", "0.9292857777 0.1890653283 "
                 "0.3109346717 -0.0632603739", True,
                 id="combined-xy-direct"),
    pytest.param("axis", "1 0 0 0", False,
                 id="axis-bvh"),
    pytest.param("axis", "1 0 0 0", True,
                 id="axis-direct"),
    pytest.param("y_only", "0.9238795 0 0.3826834 0", False,
                 id="y-only-bvh"),
    pytest.param("y_only", "0.9238795 0 0.3826834 0", True,
                 id="y-only-direct"),
])
def test_native_plugin_free_sdf_flex_kernel_matches_cpu_feature_witnesses(
    orientation, quat, no_flex_bvh):
  import torch

  xml = f"""
    <mujoco><asset>
      <mesh name="sdfbox" vertex="-.05 -.05 -.05 .05 -.05 -.05
          .05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05
          .05 .05 .05 -.05 .05 .05"/>
    </asset><option gravity="0 0 0" sdf_initpoints="4"/>
    <worldbody>
      <geom name="sdf" type="sdf" mesh="sdfbox"
            contype="0" conaffinity="1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 .04"
                quat="{quat}"
                spacing=".04 .04 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
                    elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  if no_flex_bvh:
    model.flex_bvhadr[0] = -1
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  geom = int(mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "sdf"))
  cpu = [c for c in data.contact[:data.ncon]
         if int(c.geom[0]) == geom and int(c.flex[1]) == 0]
  assert cpu, "fixture must generate CPU plugin-free flex-SDF contacts"
  if orientation == "combined_xy":
    # Avoid a non-unique Frank-Wolfe corner projection: each triangle has a
    # measurable gap between its two least-projected vertices along the witness
    # normal, including contacts whose SDF normal is nearly +Z.
    elem_data = np.asarray(model.flex_elem[
        int(model.flex_elemdataadr[0]):
        int(model.flex_elemdataadr[0] + 3*model.flex_elemnum[0])], np.int32)
    for contact in cpu:
      elem = int(contact.elem[1])
      nodes = elem_data[3*elem:3*elem+3]
      projection = (np.asarray(data.flexvert_xpos[nodes], np.float64)
                    @ np.asarray(contact.frame[:3], np.float64))
      assert np.diff(np.sort(projection))[0] > 0.005

  # This deliberately exercises the still-gated kernel in isolation. It does
  # not change the production admission guard or qualify solver integration.
  program = FlexContactProgram(model, device="mps")
  assert not program._narrowphase_admitted
  program._narrowphase_admitted = True
  geom_xquat64 = np.empty((int(model.ngeom), 4), np.float64)
  for geom_id in range(int(model.ngeom)):
    mujoco.mju_mat2Quat(geom_xquat64[geom_id],
                        np.asarray(data.geom_xmat[geom_id], np.float64).reshape(9))
  geom_xquat = geom_xquat64.astype(np.float32)
  flex_hi, flex_lo, flex_tail = _split_source_mjt_num(data.flexvert_xpos)
  geom_hi, geom_lo, geom_tail = _split_source_mjt_num(data.geom_xpos)
  result = program.run_device(
      torch.as_tensor(flex_hi[None].copy(), device="mps"),
      torch.as_tensor(geom_hi[None].copy(), device="mps"),
      torch.as_tensor(geom_xquat[None].copy(),
                      device="mps"),
      flexvert_xpos_low=torch.as_tensor(flex_lo[None].copy(), device="mps"),
      flexvert_xpos_tail=torch.as_tensor(flex_tail[None].copy(), device="mps"),
      geom_pos_low=torch.as_tensor(geom_lo[None].copy(), device="mps"),
      geom_pos_tail=torch.as_tensor(geom_tail[None].copy(), device="mps"))
  active = result["active"][0].cpu().numpy().astype(bool)
  assert active.any()
  slots = np.flatnonzero(active)
  native_elem_ids = [int(x) for x in program.descriptor.elem1[slots]]
  cpu_elem_ids = [int(c.elem[1]) for c in cpu]
  raw = result["raw_active"][0].cpu().numpy().astype(bool)
  raw_elem_ids = [int(x) for x in
                  program.descriptor.elem1[np.flatnonzero(raw)]]
  if orientation == "axis":
    import json
    raw_slots = np.flatnonzero(raw)
    selection_high = program._selection_pos[0].cpu().numpy()
    selection_words = program._contact_pos_low_tail[:, 0].cpu().numpy()
    print("FLEX_SDF_AXIS_RAW_SOURCE_WORDS", json.dumps({
        "direct": bool(no_flex_bvh),
        "slots": raw_slots.tolist(),
        "elements": program.descriptor.elem1[raw_slots].tolist(),
        "ordinals": program.descriptor.contact_ordinal[raw_slots].tolist(),
        "active": active[raw_slots].tolist(),
        "distance": result["dist"][0].cpu().numpy()[raw_slots].tolist(),
        "selection_high": selection_high[raw_slots].tolist(),
        "selection_low": selection_words[2, raw_slots].tolist(),
        "selection_tail": selection_words[3, raw_slots].tolist(),
        "midpoint_high": result["pos"][0].cpu().numpy()[raw_slots].tolist(),
        "midpoint_low": selection_words[0, raw_slots].tolist(),
        "midpoint_tail": selection_words[1, raw_slots].tolist(),
    }))
  positions = result["pos"][0].cpu().numpy()
  normals = result["normal"][0].cpu().numpy()
  distances = result["dist"][0].cpu().numpy()
  frames = result["frame"][0].cpu().numpy()
  barycentric = result["barycentric1"][0].cpu().numpy()
  if orientation != "combined_xy":
    native_points = positions[slots]
    native_normals = normals[slots]
    cpu_points = np.asarray([c.pos for c in cpu], np.float32)
    cpu_normals = np.asarray([c.frame[:3] for c in cpu], np.float32)
    print("SDF flex regression probe", orientation, "no_bvh", no_flex_bvh,
          "raw ids", raw_elem_ids, "active ids", native_elem_ids,
          "CPU ids", cpu_elem_ids,
          "native points", native_points.tolist(),
          "CPU points", cpu_points.tolist(),
          "native distances", distances[slots].tolist(),
          "CPU distances", [float(c.dist) for c in cpu],
          "normal sums", native_normals.sum(axis=0).tolist(),
          cpu_normals.sum(axis=0).tolist(),
          "unit-normal moment sums",
          np.cross(native_points, native_normals).sum(axis=0).tolist(),
          np.cross(cpu_points, cpu_normals).sum(axis=0).tolist())
  assert (len(slots) == len(cpu)
          and sorted(native_elem_ids) == sorted(cpu_elem_ids)), (
              orientation, no_flex_bvh, "feature multiplicity",
              native_elem_ids, cpu_elem_ids,
              "raw candidate identities", raw_elem_ids)
  nodes = program.descriptor.nodes1
  unmatched = list(cpu)
  for slot in slots:
    elem = int(program.descriptor.elem1[slot])
    candidates = [c for c in unmatched if int(c.elem[1]) == elem]
    assert candidates, (no_flex_bvh, elem)
    ref = min(candidates, key=lambda c: float(np.linalg.norm(
        np.asarray(c.pos, np.float32) - positions[slot])))
    unmatched.remove(ref)
    assert abs(float(distances[slot]) - float(ref.dist)) < 5e-5
    np.testing.assert_allclose(
        normals[slot],
        np.asarray(ref.frame[:3], np.float32), atol=5e-4, rtol=0)
    np.testing.assert_allclose(
        positions[slot],
        np.asarray(ref.pos, np.float32), atol=1e-4, rtol=0)
    np.testing.assert_allclose(
        frames[slot].reshape(-1),
        np.asarray(ref.frame, np.float32), atol=5e-4, rtol=0)
    weights = barycentric[slot]
    np.testing.assert_allclose(weights.sum(), 1.0, atol=2e-4, rtol=0)
    assert np.all(weights[:3] >= -2e-4) and abs(float(weights[3])) < 2e-4
    tri_nodes = nodes[slot, :3]
    surface = np.asarray(data.flexvert_xpos[tri_nodes], np.float32)
    recovered = (np.asarray(ref.pos, np.float32)
                 + np.asarray(ref.frame[:3], np.float32)
                 * (0.5 * float(ref.dist)))
    np.testing.assert_allclose(weights[:3] @ surface, recovered,
                               atol=5e-5, rtol=0)
  assert not unmatched


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
  assert np.all(descriptor.filterable[slots])
  for slot in slots:
    elem = int(descriptor.elem2[slot])
    vertex = int(descriptor.vert1[slot])
    assert elem >= 0 and vertex >= 0
    assert int(descriptor.elem1[slot]) == -1
    assert int(descriptor.vert2[slot]) == -1
    expected = np.asarray(model.flex_elem[
        int(model.flex_elemdataadr[0]) + elem*3:
        int(model.flex_elemdataadr[0]) + elem*3 + 3])
    np.testing.assert_array_equal(descriptor.nodes1[slot, 0], vertex)
    np.testing.assert_array_equal(descriptor.nodes2[slot, :3], expected)
    assert descriptor.margin[slot] == 0.0
    assert descriptor.gap[slot] == 0.0


def test_tetra_internal_face_point_slots_preserve_contact_side_order():
  model = _model(
      '', _volume_flex("volume", "none", contype=1, conaffinity=1))
  model.flex_internal[0] = True
  descriptor = lower_flex_contacts(model)
  slots = np.flatnonzero(
      (descriptor.kind == _KIND_INTERNAL_VERTEX_ELEMENT)
      & (descriptor.feature1 >= 0))
  assert slots.size == 4 * int(model.flex_elemnum[0])
  assert np.all(descriptor.filterable[slots])
  for slot in slots:
    assert descriptor.elem1[slot] >= 0
    assert descriptor.vert1[slot] == -1
    assert descriptor.elem2[slot] == -1
    vertex = int(descriptor.vert2[slot])
    assert vertex >= 0
    assert descriptor.nodes1[slot, 3] == -1
    assert np.count_nonzero(descriptor.nodes1[slot] >= 0) == 3
    np.testing.assert_array_equal(descriptor.nodes2[slot, 0], vertex)


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


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="native opt-in")
def test_native_plane_contact_jacobians_match_node_weighted_reference():
  torch = pytest.importorskip("torch")
  if not torch.backends.mps.is_available():
    pytest.skip("native Metal contact Jacobian qualification requires MPS")
  model, data, program, flex_jac = _actual_plane_contact_jacobians()
  descriptor = program.descriptor
  nslot, nv = descriptor.slot_count, int(model.nv)
  assert data.ncon == nslot
  flex_xpos = np.repeat(
      np.asarray(data.flexvert_xpos, np.float32)[None, :, :], 2, axis=0)
  flex_xpos[1, :, 2] += 0.02  # separated negative control
  geom_pos = np.repeat(np.asarray(data.geom_xpos, np.float32)[None, :, :],
                       2, axis=0)
  geom_quat = np.zeros((2, int(model.ngeom), 4), np.float32)
  geom_quat[..., 0] = 1.0
  program = FlexContactProgram(model, batch_size=2, device="mps")
  result = program.run_device(
      torch.as_tensor(flex_xpos, device="mps"),
      torch.as_tensor(geom_pos, device="mps"),
      torch.as_tensor(geom_quat, device="mps"))
  active = result["active"].cpu().numpy()
  assert active[0].all()
  assert not active[1].any()
  np.testing.assert_allclose(result["frame"].cpu().numpy()[1], 0, atol=0)
  frames = result["frame"].cpu().numpy()[0]
  cpu_contacts = {int(c.vert[1]): c for c in data.contact[:data.ncon]}
  for slot, vertex in enumerate(descriptor.vert1):
    expected_frame = np.asarray(
        cpu_contacts[int(vertex)].frame, np.float32).reshape(3, 3)
    np.testing.assert_allclose(frames[slot], expected_frame,
                               rtol=1e-6, atol=1e-7)
  flex_jac = np.repeat(flex_jac, 2, axis=0)
  geom_point_jac = np.zeros((2, nslot, 3, nv), np.float32)
  jac = program.run_contact_jacobians(
      result, torch.as_tensor(flex_jac, device="mps"),
      torch.as_tensor(geom_point_jac, device="mps"))
  relative = jac["relative"].cpu().numpy()[0]
  for slot, contact in enumerate(data.contact[:data.ncon]):
    normal = np.asarray(contact.frame[:3], np.float32)
    projected = normal @ relative[slot]
    pinned = np.asarray(data.efc_J, np.float32).reshape(-1, nv)[
        int(contact.efc_address)]
    np.testing.assert_allclose(projected, pinned, rtol=1e-5, atol=1e-6)


def _actual_plane_contact_jacobians():
  model = _model(
      '<geom name="ground" type="plane" size="0 0 .1" '
      'contype="0" conaffinity="1" condim="1"/>',
      _flex("cloth", "0 0 0", 1, 0).replace(
          'conaffinity="0"', 'conaffinity="0" condim="1"'),
      cone="elliptic")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  program = FlexContactProgram(model, device="cpu")
  descriptor = program.descriptor
  assert np.all(descriptor.kind == _KIND_PLANE_VERTEX)
  nv = int(model.nv)
  assert data.ncon == descriptor.slot_count
  assert data.nefc >= data.ncon
  assert np.all(descriptor.condim == 1)
  flex_jac = np.zeros((1, int(model.nflexvert), 3, nv), np.float32)
  for vertex in range(int(model.nflexvert)):
    body = int(model.flex_vertbodyid[vertex])
    jacp = np.zeros((3, nv), np.float64)
    jacr = np.zeros((3, nv), np.float64)
    mujoco.mj_jac(model, data, jacp, jacr,
                  np.asarray(data.flexvert_xpos[vertex]), body)
    flex_jac[0, vertex] = jacp
  for contact in data.contact[:data.ncon]:
    vertex = int(contact.vert[1])
    assert int(contact.efc_address) >= 0
    jac = flex_jac[0, vertex]
    assert np.linalg.norm(jac) > 0.0
    pinned = np.asarray(data.efc_J, np.float32).reshape(-1, nv)[
        int(contact.efc_address)]
    np.testing.assert_allclose(
        np.asarray(contact.frame[:3], np.float32) @ jac,
        pinned, rtol=1e-6, atol=1e-7)
  return model, data, program, flex_jac


def test_actual_flex_node_jacobians_match_pinned_plane_efc_jacobian():
  pytest.importorskip("torch")
  _actual_plane_contact_jacobians()


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


def test_contact_program_capacity_guards_run_before_torch_allocation():
  model = _model(
      '<geom type="plane" size="0 0 .1" contype="0" conaffinity="1"/>',
      _volume_flex("volume", "none"))
  descriptor = lower_flex_contacts(model)
  assert _validate_contact_program_capacity(model, descriptor, np.int64(2)) == 2
  # Constructor rejects the invalid batch before attempting to import or
  # allocate Torch; this assertion also runs in the Torch-free preflight.
  with pytest.raises(ValueError, match="positive integer"):
    FlexContactProgram(model, batch_size=0, device="cpu")
  for invalid in (0, -1, True, 1.0):
    with pytest.raises(ValueError, match="positive integer"):
      _validate_contact_program_capacity(model, descriptor, invalid)
  with pytest.raises(ValueError, match="signed 32-bit"):
    _validate_contact_program_capacity(model, descriptor, 2**31)
  # Use a synthetic extreme nv with a nonempty static slot set to exercise the
  # flattened spatial-J overflow check without allocating a corresponding
  # model or touching a device.
  with pytest.raises(ValueError, match="workspace exceeds"):
    _validate_contact_program_capacity(
        model, replace(descriptor, nv=np.iinfo(np.int32).max), 2)
