import mujoco


def test_runtime_capacity_counts_flex_ccd_arenas_and_three_word_scalars():
  from mujoco_metal.capacity import _runtime_buffer_sizes
  from mujoco_metal.flex_contact import (
      _flex_ccd_workspace_capacity, lower_flex_contacts)

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><geom type="plane" size="0 0 .1"/>
    <flexcomp name="sheet" type="grid" count="2 2 1"
      spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
      <contact contype="1" conaffinity="1" selfcollide="none"/>
      <edge stiffness="0" damping="0"/>
      <elasticity young="100" poisson=".2" thickness=".01"
        elastic2d="stretch"/>
    </flexcomp></worldbody></mujoco>''')
  batch = 3
  slots = int(lower_flex_contacts(model).slot_count)
  assert slots > 0
  vertices, faces, horizon, stack = _flex_ccd_workspace_capacity(
      model.opt.ccd_iterations, batch, slots)
  float_stride = (vertices * 39 + faces * 20 + 3) & ~3
  int_stride = ((faces + 1) & ~1) + horizon * 2 + stack * 6

  inventory = dict(_runtime_buffer_sizes(model, batch))
  assert inventory["flex_contact.radius_high"] == model.nflex
  assert inventory["flex_contact.radius_mid"] == model.nflex
  assert inventory["flex_contact.radius_low"] == model.nflex
  assert inventory["flex_contact.margin"] == slots
  assert inventory["flex_contact.gap"] == slots
  assert inventory["flex_contact.margin_gap_high"] == slots * 2
  assert inventory["flex_contact.geom_size_mid"] == model.ngeom * 3
  assert inventory["flex_contact.geom_size_low"] == model.ngeom * 3
  assert inventory["flex_contact.margin_gap_mid"] == slots * 2
  assert inventory["flex_contact.margin_gap_low"] == slots * 2
  assert inventory["flex_contact.ccd_tolerance_mid"] == 1
  assert inventory["flex_contact.ccd_tolerance_low"] == 1
  assert inventory["flex_contact.epa_float_workspace"] == (
      batch * max(slots, 1) * float_stride)
  assert inventory["flex_contact.epa_int_workspace"] == (
      batch * max(slots, 1) * int_stride)
  assert inventory["flex_contact.filter_permutation"] == batch * slots
  assert inventory["flex_contact.filter_selected_position"] == batch * slots
  assert inventory["flex_contact.filter_min_distance_words"] == batch * slots * 3
  assert inventory["flex_contact.contact_pos_low_tail"] == batch * slots * 12


def test_fw_and_midpoint_residual_planes_match_owned_host_allocation():
  import torch
  from mujoco_metal.flex_contact import FlexContactProgram, lower_flex_contacts

  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <worldbody><geom type="plane" size="0 0 .1"/>
    <flexcomp name="sheet" type="grid" count="2 2 1"
      spacing=".1 .1 .1" mass="1" dim="2" radius=".005">
      <contact contype="1" conaffinity="1" selfcollide="none"/>
      <edge stiffness="0" damping="0"/>
      <elasticity young="100" poisson=".2" thickness=".01"
        elastic2d="stretch"/>
    </flexcomp></worldbody></mujoco>''')
  batch = 3
  slots = int(lower_flex_contacts(model).slot_count)
  program = FlexContactProgram(model, batch_size=batch, device="cpu")
  # Four plane-major residuals: contact midpoint low/tail and raw SDF FW
  # point low/tail. Check the actual owned allocation, not only capacity math.
  assert tuple(program._contact_pos_low_tail.shape) == (4, batch, slots, 3)
  assert program._contact_pos_low_tail.dtype == torch.float32
  assert program._contact_pos_low_tail.is_contiguous()
  expected = 4 * batch * slots * 3
  assert program._contact_pos_low_tail.numel() == expected
  assert dict(__import__("mujoco_metal.capacity", fromlist=["_runtime_buffer_sizes"])
              ._runtime_buffer_sizes(model, batch))["flex_contact.contact_pos_low_tail"] == expected
