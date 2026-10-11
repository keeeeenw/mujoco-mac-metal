# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Exact admission for compensated-accumulation scratch (host, CPU-only).

The TwoSum scratch is pre-allocated once against admitted
simulation.twosum_float / simulation.twosum_mask budget entries and never
grown. These tests prove the inventory bytes, the exact-budget boundary,
zero-DOF/contact-free inclusion and the one-byte-under failure occurring
before any device allocation.
"""

import mujoco
import pytest

from mujoco_metal.capacity import (
    CapacityLimits,
    CapacityOverflow,
    check_capacity,
    estimate_capacity,
)


def _slide_model():
  bodies = "".join(
      f'<body pos="{0.2 * k} 0 0.3"><joint name="s{k}" type="slide" axis="1 0 0"/>'
      f'<geom type="sphere" size="0.05"/></body>' for k in range(4))
  return mujoco.MjModel.from_xml_string(
      f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")


def _zero_dof_model():
  return mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')


def _breakdown(est):
  return dict(est.memory_breakdown)


def test_twosum_inventory_has_exact_bytes():
  model = _slide_model()
  batch, nv = 2, int(model.nv)
  est = estimate_capacity(model, batch, npairs=3, nslots=5, nr=20)
  table = _breakdown(est)
  assert table["simulation.twosum_float"] == 8 * max(batch * nv, 1) * 4
  # Bool selectors occupy one physical byte per element.
  assert table["simulation.twosum_mask"] == max(batch, 1)
  assert est.memory_bytes == sum(table.values())


def test_exact_budget_succeeds_one_byte_under_fails_before_allocation():
  model = _slide_model()
  est = estimate_capacity(model, 2, npairs=3, nslots=5, nr=20)
  exact = CapacityLimits(memory_budget_bytes=int(est.memory_bytes))
  assert check_capacity(est, exact) is est
  short = CapacityLimits(memory_budget_bytes=int(est.memory_bytes) - 1)
  with pytest.raises(CapacityOverflow, match="memory"):
    check_capacity(est, short)


def test_zero_dof_and_contact_free_models_carry_guarded_scratch():
  for model in (_zero_dof_model(), _slide_model()):
    est = estimate_capacity(model, 3, npairs=0, nslots=0, nr=0)
    table = _breakdown(est)
    nv = int(model.nv)
    assert table["simulation.twosum_float"] == 8 * max(3 * nv, 1) * 4
    assert table["simulation.twosum_mask"] == 3
    assert check_capacity(est) is est


def test_twosum_growth_is_bounded_by_batch_and_dof():
  model = _slide_model()
  small = estimate_capacity(model, 1, npairs=1, nslots=1, nr=4)
  big = estimate_capacity(model, 4, npairs=1, nslots=1, nr=4)
  table_small, table_big = _breakdown(small), _breakdown(big)
  nv = int(model.nv)
  assert table_big["simulation.twosum_float"] == 8 * 4 * nv * 4
  assert table_big["simulation.twosum_mask"] == 4
  assert table_small["simulation.twosum_float"] == 8 * 1 * nv * 4
  # A one-world budget cannot admit a four-world scratch.
  with pytest.raises(CapacityOverflow, match="memory"):
    check_capacity(big, CapacityLimits(
        memory_budget_bytes=int(small.memory_bytes)))
