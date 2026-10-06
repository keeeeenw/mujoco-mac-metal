# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Canonical fixed flex-contact row allocation and metadata contracts."""

import mujoco
import numpy as np
import pytest

from mujoco_metal.coupled_constraints import lower_coupled_constraints
from mujoco_metal.capacity import _runtime_buffer_sizes
from mujoco_metal.flex_contact import lower_flex_contacts
from mujoco_metal.solver_islands import lower_solver_row_metadata


def _model(condim, cone):
  return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 0" cone="{cone}"/>
      <worldbody>
        <geom name="ground" type="plane" size="0 0 .1"
              contype="0" conaffinity="1" condim="{condim}"
              friction=".7 .4 .03"/>
        <flexcomp name="cloth" type="grid" count="2 2 1"
                  pos="0 0 .02" spacing=".1 .1 .1" mass="1" dim="2">
          <contact contype="1" conaffinity="0" selfcollide="none"
                   condim="{condim}" friction=".5 .6 .02"/>
          <elasticity young="100" poisson=".2" thickness=".01"
                      elastic2d="stretch"/>
        </flexcomp>
      </worldbody>
    </mujoco>
  """)


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("condim", [1, 3, 4, 6])
def test_coupled_allocator_reserves_exact_flex_candidate_row_spans(condim, cone):
  model = _model(condim, cone)
  flex = lower_flex_contacts(model)
  descriptor = lower_coupled_constraints(model)

  assert descriptor.flex_contact_descriptor.row_capacity == flex.row_capacity
  assert descriptor.n_flex_contact_rows == flex.row_capacity
  assert descriptor.flex_contact_base == descriptor.nr_joint + (
      descriptor.nr - descriptor.nr_joint - descriptor.n_flex_contact_rows)
  assert descriptor.nr == descriptor.flex_contact_base + flex.row_capacity
  if flex.slot_count:
    assert flex.row_start[0] == 0
    assert flex.row_start[-1] + flex.row_span[-1] == flex.row_capacity
    expected_span = (1 if condim == 1 else condim if cone == "elliptic"
                     else 2 * (condim - 1))
    np.testing.assert_array_equal(flex.row_span,
                                  np.full(flex.slot_count, expected_span))


def test_flex_candidate_rows_keep_stable_contact_identity_in_island_metadata():
  model = _model(4, "pyramidal")
  descriptor = lower_coupled_constraints(model)
  flex = descriptor.flex_contact_descriptor
  metadata = lower_solver_row_metadata(model, descriptor)
  assert flex is not None and flex.slot_count > 0

  contact_types = {
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
      int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
  }
  for slot in range(flex.slot_count):
    start = descriptor.flex_contact_base + int(flex.row_start[slot])
    stop = start + int(flex.row_span[slot])
    rows = metadata[start:stop]
    assert np.all(np.isin(rows[:, 0], tuple(contact_types)))
    assert np.all(rows[:, 1] == descriptor.ncontacts_max + slot)
    # A flex contact can depend on every node in an interpolated element.
    # Its rigid two-tree shortcut metadata is therefore disabled; the island
    # builder scans the assembled row Jacobian for actual contributors.
    assert np.all(rows[:, 5] == 0)


def test_solver_flex_cone_constant_capacity_matches_descriptor():
  model = _model(6, "elliptic")
  descriptor = lower_flex_contacts(model)
  sizes = dict(_runtime_buffer_sizes(model, 2))
  assert sizes["model.solver_flex_contact_friction"] == descriptor.slot_count * 5
  assert sizes["model.solver_flex_contact_condim"] == descriptor.slot_count * 3


def test_solver_flex_cone_constant_capacity_is_empty_without_flex():
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
  sizes = dict(_runtime_buffer_sizes(model, 1))
  assert sizes["model.solver_flex_contact_friction"] == 0
  assert sizes["model.solver_flex_contact_condim"] == 0
