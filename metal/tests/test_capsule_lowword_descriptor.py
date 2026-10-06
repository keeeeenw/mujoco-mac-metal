"""CPU coverage for the exact model scalars consumed by flex capsule rows."""
import numpy as np
import mujoco
import pytest

from mujoco_metal.flex_contact import lower_flex_contacts, _KIND_ELEMENT_PAIR


def _model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <flexcomp name="a" type="grid" count="3 1 1" spacing=".05 .05 .05"
                mass="1" dim="1" radius=".00500000012345">
        <contact contype="1" conaffinity="1" selfcollide="none"
                 margin=".000100000012345" gap=".000200000023456"/>
      </flexcomp>
      <flexcomp name="b" type="grid" count="3 1 1" pos=".03 0 0"
                spacing=".05 .05 .05" mass="1" dim="1"
                radius=".00700000023456">
        <contact contype="1" conaffinity="1" selfcollide="none"
                 margin=".000300000034567" gap=".000400000045678"/>
      </flexcomp>
    </worldbody></mujoco>""")


def _expand(high, middle, low):
    return (high.astype(np.float64) + middle.astype(np.float64)
            + low.astype(np.float64))


def test_capsule_descriptor_keeps_each_flex_radius_as_three_float_words():
    model = _model()
    descriptor = lower_flex_contacts(model)
    restored = _expand(descriptor.flex_radius_hi,
                       descriptor.flex_radius_mid,
                       descriptor.flex_radius_low)
    np.testing.assert_array_equal(restored, model.flex_radius)
    assert np.any(descriptor.flex_radius_mid != 0)
    assert np.any(descriptor.flex_radius_low != 0)
    for slot in range(descriptor.slot_count):
        flex_id = int(descriptor.flex1[slot])
        assert restored[flex_id] == model.flex_radius[flex_id]


def test_capsule_descriptor_keeps_mixed_margin_and_gap_as_separate_triples():
    model = _model()
    descriptor = lower_flex_contacts(model)
    slots = np.flatnonzero((descriptor.kind == _KIND_ELEMENT_PAIR)
                           & (descriptor.flex2 >= 0))
    assert len(slots)
    f1, f2 = descriptor.flex1[slots], descriptor.flex2[slots]
    expected_margin = model.flex_margin[f1] + model.flex_margin[f2]
    expected_gap = model.flex_gap[f1] + model.flex_gap[f2]
    restored_margin = _expand(descriptor.margin[slots],
                              descriptor.margin_mid[slots],
                              descriptor.margin_low[slots])
    restored_gap = _expand(descriptor.gap[slots],
                           descriptor.gap_mid[slots],
                           descriptor.gap_low[slots])
    np.testing.assert_array_equal(restored_margin, expected_margin)
    np.testing.assert_array_equal(restored_gap, expected_gap)
    assert np.any(descriptor.margin_mid[slots] != 0)
    assert np.any(descriptor.gap_mid[slots] != 0)


def test_capsule_lowword_fixture_has_full_pinned_cross_manifold_reference():
    """The native low-word fixture retains all pinned raw capsule ordinals."""
    from test_capsule_lowword_native import _lowword_fixture, _expected_contacts

    model, data = _lowword_fixture()
    descriptor = lower_flex_contacts(model)
    expected = _expected_contacts(model, data, descriptor, "lowword-cross")
    assert set(expected) == {0, 1, 8, 9, 12, 13}
    assert len(expected) == 6
    for contact in expected.values():
        f1, f2 = map(int, contact.flex)
        centerline_distance = contact.dist + model.flex_radius[f1] + model.flex_radius[f2]
        assert centerline_distance == pytest.approx(0.0, abs=2e-8)
        assert contact.dist == pytest.approx(
            -(model.flex_radius[f1] + model.flex_radius[f2]), abs=2e-8)
