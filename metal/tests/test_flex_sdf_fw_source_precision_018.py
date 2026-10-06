"""Source-order CPU witnesses for plugin-free flex SDF Frank-Wolfe."""

import numpy as np
import mujoco
import pytest

from test_flex_contact_lowering import _split_source_mjt_num
import mujoco_metal.flex_contact as flex_contact_module
from mujoco_metal.flex_contact import _flex_mesh_hull, lower_flex_contacts
from mujoco_metal.capacity import _runtime_buffer_sizes


def test_halton_barycentric_scalars_match_pinned_mju_halton():
  # Exercise the non-binary base-3 path as well as the exact base-2 values.
  for index in range(1, 65):
    for base in (2, 3):
      n = index
      factor = 1.0 / float(base)
      value = 0.0
      while n > 0:
        next_n = n // base
        digit = n - next_n * base
        value += factor * float(digit)
        factor /= float(base)
        n = next_n
      assert value == mujoco.mju_Halton(index, base)


def test_full_source_vertices_and_geom_pose_fit_owned_three_word_input():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <asset><mesh name="sdfbox" vertex="-.05 -.05 -.05 .05 -.05 -.05
      .05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05
      .05 .05 .05 -.05 .05 .05"/></asset>
    <option gravity="0 0 0" sdf_initpoints="4"/>
    <worldbody>
      <geom name="sdf" type="sdf" mesh="sdfbox"/>
      <flexcomp name="cloth" type="grid" count="2 2 1"
        pos="0 0 .04" quat="0.9238795 0 0.3826834 0"
        spacing=".04 .04 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
          elastic2d="stretch"/>
      </flexcomp>
    </worldbody></mujoco>''')
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for values in (data.flexvert_xpos, data.geom_xpos):
    hi, lo, tail = _split_source_mjt_num(values)
    represented = (hi.astype(np.float64) + lo.astype(np.float64)
                   + tail.astype(np.float64))
    np.testing.assert_array_equal(represented, np.asarray(values, np.float64))


def test_compiled_sdf_octree_operands_retain_all_source_words_and_capacity():
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <asset><mesh name="sdfbox" vertex="-.05 -.05 -.05 .05 -.05 -.05
      .05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05
      .05 .05 .05 -.05 .05 .05"/></asset>
    <option gravity="0 0 0" sdf_initpoints="4"/>
    <worldbody><geom name="sdf" type="sdf" mesh="sdfbox"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 .04"
        spacing=".04 .04 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
          elastic2d="stretch"/>
      </flexcomp></worldbody></mujoco>''')
  descriptor = lower_flex_contacts(model)
  hull, info_flat = _flex_mesh_hull(model, descriptor)
  info = info_flat.reshape(int(model.ngeom), 9)
  sdf_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sdf"))
  mesh = int(model.geom_dataid[sdf_geom])
  adr, count = int(model.mesh_octadr[mesh]), int(model.mesh_octnum[mesh])
  row = info[sdf_geom]
  high_aabb = int(row[0]) + 8 * count
  high_coeff = int(row[0]) + 14 * count
  low_aabb, tail_aabb, low_coeff, tail_coeff = map(int, row[4:8])
  assert row[1] == count and all(x >= 0 for x in
                                  (low_aabb, tail_aabb, low_coeff, tail_coeff))
  assert low_aabb == high_coeff + 8 * count
  assert tail_aabb == low_aabb + 6 * count
  assert low_coeff == tail_aabb + 6 * count
  assert tail_coeff == low_coeff + 8 * count
  assert tail_coeff + 8 * count <= hull.size

  def restore(low, tail, high, width):
    return (hull[high:high + width].astype(np.float64)
            + hull[low:low + width].astype(np.float64)
            + hull[tail:tail + width].astype(np.float64))

  source_aabb = np.asarray(
      model.oct_aabb[adr:adr + count], dtype=np.float64).reshape(-1)
  source_coeff = np.asarray(
      model.oct_coeff[adr:adr + count], dtype=np.float64).reshape(-1)
  np.testing.assert_array_equal(
      restore(low_aabb, tail_aabb, high_aabb, 6 * count), source_aabb)
  np.testing.assert_array_equal(
      restore(low_coeff, tail_coeff, high_coeff, 8 * count), source_coeff)

  inventory = dict(_runtime_buffer_sizes(model, 1))
  assert inventory["flex_contact.geom_hull_payload"] == hull.size
  assert inventory["flex_contact.geom_hull_info"] == info_flat.size


def test_compiled_sdf_octree_payload_fails_before_int32_offset_overflow(
    monkeypatch):
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <asset><mesh name="sdfbox" vertex="-.05 -.05 -.05 .05 -.05 -.05
      .05 .05 -.05 -.05 .05 -.05 -.05 -.05 .05 .05 -.05 .05
      .05 .05 .05 -.05 .05 .05"/></asset>
    <worldbody><geom name="sdf" type="sdf" mesh="sdfbox"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" pos="0 0 .04"
        spacing=".04 .04 .01" mass="1" dim="2">
        <contact contype="1" conaffinity="1" selfcollide="none"/>
        <edge stiffness="0" damping="0"/>
        <elasticity young="100" poisson=".2" thickness=".01"
          elastic2d="stretch"/>
      </flexcomp></worldbody></mujoco>''')
  descriptor = lower_flex_contacts(model)
  # Force an intentionally small arena limit so the real append/offset guard
  # executes without allocating a pathological model-sized host payload.
  monkeypatch.setattr(flex_contact_module, "_INT32_MAX", 16)
  with pytest.raises(ValueError, match="signed int32 offsets"):
    _flex_mesh_hull(model, descriptor)
