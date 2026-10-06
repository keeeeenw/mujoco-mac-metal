# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Isolated production-selector replay against pinned filterFlexContacts.

This drives the actual compiled flex_contact_select MSL kernel with explicit
B=3 contact arrays. The arrays are float32 inputs to the kernel; the CPU
reference runs pinned engine_collision_driver's mutable-array state machine
using those exact represented values.
"""

import os

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mujoco_metal.flex_contact import FlexContactProgram
from mujoco_metal.flex_contact import _candidate_filter_modes
from mujoco_metal.flex_contact import lower_flex_contacts


def _model():
  # 128 fixed candidate slots and two real filter groups. The selector test
  # supplies actual stage inputs so every world's active subset and group
  # geometry is deliberately distinct.
  model = mujoco.MjModel.from_xml_string("""
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
  """)
  model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
  return model


def _pinned_filter_slots(slots, dist, pos, pos_low_tail=None, maxkeep=50):
  """Pinned filterFlexContacts with contact arrays in their exact input order."""
  if pos_low_tail is not None:
    pos = (np.asarray(pos, dtype=np.float64)
           + np.asarray(pos_low_tail[0], dtype=np.float64)
           + np.asarray(pos_low_tail[1], dtype=np.float64))
  else:
    pos = np.asarray(pos, dtype=np.float64)
  contacts = [int(slot) for slot in slots]
  n = len(contacts)
  if n <= maxkeep:
    return contacts
  selected = np.zeros(n, dtype=np.bool_)
  min_dist = np.full(n, 1.0e10, dtype=np.float64)
  best = 0
  bestval = -float(dist[contacts[0]])
  for i in range(1, n):
    value = -float(dist[contacts[i]])
    if value > bestval:
      bestval, best = value, i
  nselected = 0
  while nselected < maxkeep and best >= 0:
    selected[best] = True
    best_pos = np.asarray(pos[contacts[best]], dtype=np.float64)
    nextbest = -1
    nextbestdist = -1.0
    for i in range(n):
      if selected[i]:
        continue
      delta = (np.asarray(pos[contacts[i]], dtype=np.float64) - best_pos)
      d2 = float((delta[0] * delta[0] + delta[1] * delta[1])
                 + delta[2] * delta[2])
      if d2 < min_dist[i]:
        min_dist[i] = d2
      if min_dist[i] > nextbestdist:
        nextbestdist, nextbest = min_dist[i], i
    if nselected < maxkeep - 1:
      contacts[nselected], contacts[best] = contacts[best], contacts[nselected]
      if nextbest == nselected:
        nextbest = best
    nselected += 1
    best = nextbest
  return contacts[:nselected]


def _stage_inputs(metadata=None):
  batch, nslot = 3, 128
  if metadata is None:
    group = np.repeat(np.arange(2, dtype=np.int32), nslot // 2)
    filterable = np.ones(nslot, dtype=np.int32)
    # Exercise nonfilterable pass-through in the CPU contract fixture.
    filterable[np.asarray([3, 19, 67, 91, 111])] = 0
    mode = np.where(filterable != 0, 1, 0).astype(np.int32)
    element = np.arange(nslot, dtype=np.int32)
    geom = 200 + np.arange(nslot, dtype=np.int32)
  else:
    group, filterable, mode, element, geom = metadata
  active = np.zeros((batch, nslot), dtype=np.int32)
  # Each world has different active groups and counts: world 0 runs FPS in
  # both groups, world 1 runs FPS only in group 0, and world 2 is masked.
  group_slots = [np.flatnonzero(group == value) for value in range(2)]
  if tuple(map(len, group_slots)) != (64, 64):
    raise AssertionError("selector stage fixture needs two 64-slot groups")
  active[0, group_slots[0]] = 1
  active[0, group_slots[1][:53]] = 1
  active[1, group_slots[0][:58]] = 1
  active[1, group_slots[1][:18]] = 1
  distance = np.empty((batch, nslot), dtype=np.float32)
  pos = np.empty((batch, nslot, 3), dtype=np.float32)
  for env in range(batch):
    for slot in range(nslot):
      # Nonuniform depths and noncollinear positions exercise tie ordering,
      # position-indexed min_dist, and all three squared-distance terms.
      distance[env, slot] = np.float32(
          -0.001 * (1 + ((slot * 17 + env * 23) % 97)))
      pos[env, slot] = np.asarray([
          np.float32((slot * 13 % 37) * .017 + env * .11),
          np.float32((slot * 7 % 29) * .023 - env * .07),
          np.float32((slot * 11 % 31) * .019 + env * .031)], np.float32)
  source_pos = pos.astype(np.float64)
  for env in range(batch):
    for slot in range(nslot):
      source_pos[env, slot, (slot + env) % 3] += (
          (-1.0 if (slot + env) & 1 else 1.0)
          * (1 + (slot % 5)) * 2.0 ** -29)
  high = source_pos.astype(np.float32)
  low = (source_pos - high.astype(np.float64)).astype(np.float32)
  tail = (source_pos - high.astype(np.float64)
          - low.astype(np.float64)).astype(np.float32)
  pos_low_tail = np.stack((low, tail), axis=1)
  return (batch, nslot, active, distance, high, pos_low_tail, group,
          filterable, mode, element, geom)


def _expected_filtered(active, distance, pos, pos_low_tail, group, filterable,
                       ngroup=2, maxkeep=50):
  expected = np.zeros_like(active, dtype=np.bool_)
  for env in range(active.shape[0] - 1):
    expected[env, (active[env] != 0) & (filterable == 0)] = True
    for group_id in range(ngroup):
      slots = np.flatnonzero((active[env] != 0) & (filterable != 0)
                             & (group == group_id))
      expected_slots = _pinned_filter_slots(
          slots, distance[env], pos[env], pos_low_tail[env], maxkeep=maxkeep)
      expected[env, expected_slots] = True
  return expected


def test_cpu_b3_selector_fixture_replays_pinned_mutable_filter():
  (batch, nslot, active, distance, pos, pos_low_tail, group, filterable, mode,
   element, geom) = _stage_inputs()
  expected = _expected_filtered(active, distance, pos, pos_low_tail, group,
                                filterable)
  assert expected[0].sum() == 105  # 50+50 plus five pass-through slots
  assert expected[1].sum() == 70   # 50+17 plus three pass-through slots
  assert not expected[2].any()
  assert not np.array_equal(expected[0], expected[1])
  assert group.shape == filterable.shape == mode.shape == element.shape == geom.shape == (nslot,)


def test_contact_midpoint_tail_capacity_is_budgeted_for_each_world():
  from mujoco_metal.capacity import _runtime_buffer_sizes

  model = _model()
  descriptor = lower_flex_contacts(model)
  sizes = dict(_runtime_buffer_sizes(model, 3))
  assert sizes["flex_contact.contact_pos_low_tail"] == (
      3 * descriptor.slot_count * 12)


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="production selector needs explicit MPS opt-in")
def test_native_production_selector_matches_pinned_mutable_filter_B3():
  if not torch.backends.mps.is_available():
    pytest.skip("production flex selector requires MPS")
  batch = 3
  model = _model()
  program = FlexContactProgram(model, batch_size=batch, device="mps")
  nslot = program.descriptor.slot_count
  assert program.descriptor.filter_group_count == 2
  metadata = (
      np.asarray(program.descriptor.filter_group, dtype=np.int32).copy(),
      np.asarray(program.descriptor.filterable, dtype=np.int32).copy(),
      _candidate_filter_modes(model, program.descriptor),
      np.asarray(program.descriptor.elem1, dtype=np.int32).copy(),
      np.asarray(program.descriptor.geom, dtype=np.int32).copy())
  # Exercise source nonfilterable passthrough in the same production kernel;
  # all remaining slot-to-group/feature maps stay the compiled model maps.
  metadata[1][np.asarray([3, 19, 67, 91, 111])] = 0
  (batch, nslot, active, distance, pos, pos_low_tail, group, filterable, mode,
   element, geom) = _stage_inputs(metadata)
  assert program.descriptor.slot_count == nslot
  # The group/filter/feature vectors are the actual compiled production
  # descriptors. The stage inputs below vary world activity and exact float32
  # contact values while preserving the slot identity maps.
  def mps(array, dtype=None):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype,
                           device="mps").contiguous()
  dims = mps(np.asarray([batch, nslot, 2, 50, 1, 1, 0], np.int32),
             torch.int32)
  source_active = mps(np.full((batch, nslot), -17, np.int32), torch.int32)
  raw_mask = mps(np.ones((batch, nslot), np.bool_), torch.bool)
  filtered = mps(np.ones((batch, nslot), np.bool_), torch.bool)
  permutation = mps(np.full((batch, nslot), -23, np.int32), torch.int32)
  selected_position = mps(np.full((batch, nslot), -29, np.int32), torch.int32)
  min_distance = mps(np.full((batch, nslot, 3), -31.0, np.float32),
                     torch.float32)
  world_mask = np.asarray([1, 1, 0], np.int32)
  program._shader.flex_contact_select(
      mps(active, torch.int32), mps(distance, torch.float32),
      mps(pos, torch.float32), mps(pos, torch.float32),
      mps(group, torch.int32), mps(filterable, torch.int32),
      mps(mode, torch.int32), mps(element, torch.int32), mps(geom, torch.int32),
      mps(np.asarray([batch, nslot, 2, 50, *world_mask], np.int32),
          torch.int32),
      source_active, raw_mask, filtered, permutation, selected_position,
      min_distance, mps(pos_low_tail, torch.float32),
      threads=(batch,), group_size=(1,))
  expected = _expected_filtered(active, distance, pos, pos_low_tail, group,
                                filterable)
  actual = filtered.cpu().numpy()
  np.testing.assert_array_equal(actual[:2], expected[:2])
  # The masked third world's every output/scratch buffer must remain untouched.
  np.testing.assert_array_equal(source_active.cpu().numpy()[2], -17)
  np.testing.assert_array_equal(raw_mask.cpu().numpy()[2], True)
  np.testing.assert_array_equal(filtered.cpu().numpy()[2], True)
  np.testing.assert_array_equal(permutation.cpu().numpy()[2], -23)
  np.testing.assert_array_equal(selected_position.cpu().numpy()[2], -29)
  np.testing.assert_array_equal(min_distance.cpu().numpy()[2], -31.0)
