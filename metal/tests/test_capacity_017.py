# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Milestone 017: explicit capacity architecture (CPU-only part).

Deterministic estimation, user limits, memory budgeting, overflow
diagnostics, and growth-at-host-boundaries. No physics changes here.
"""

import mujoco
import numpy as np
import pytest

from mujoco_metal.capacity import (
    BASE_MAX_PAIRS,
    BASE_MAX_ROWS,
    BASE_MAX_SLOTS,
    BASE_NVIDIA_NV,
    AUTO_JACOBIAN_DENSE_NV,
    DENSE_ROW_THRESHOLD,
    CapacityLimits,
    CapacityOverflow,
    check_capacity,
    estimate_capacity,
    estimate_workspace,
    _check_i32_elements,
    SIGNED_I32_ELEMENT_LIMIT,
    NATIVE_MAX_NV,
    NATIVE_MAX_ROWS,
    selected_jacobian_kind,
    primal_scratch_floats,
)


def _model(nv_hint="slide-chain"):
  if nv_hint == "slide-chain":
    bodies = "".join(
        f'<body pos="{0.2 * k} 0 0.3"><joint name="s{k}" type="slide" axis="1 0 0"/>'
        f'<geom type="sphere" size="0.05"/></body>' for k in range(4))
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")
  return mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body><joint/><geom type='sphere' size='.1'/></body></worldbody></mujoco>")


def test_estimate_is_deterministic_and_counts_memory():
  model = _model()
  est1 = estimate_capacity(model, 2, npairs=3, nslots=5, nr=20)
  est2 = estimate_capacity(model, 2, npairs=3, nslots=5, nr=20)
  assert est1 == est2
  assert est1.dense_path == (20 <= DENSE_ROW_THRESHOLD)
  assert est1.memory_bytes == sum(v for _, v in est1.memory_breakdown)
  assert est1.memory_bytes > 0
  names = [k for k, _ in est1.memory_breakdown]
  # R08a: parts mirror prepare_workspace buffer-for-buffer.
  assert "workspace_debug" in names and "contact_row_data" in names
  assert "workspace_J" in names and "out_contact_force" in names
  assert "eq_active_default" in names
  solver_parts = estimate_workspace_expected(model, 2, 3, 5, 20)
  assert est1.memory_breakdown[:len(solver_parts)] == solver_parts
  runtime = dict(est1.memory_breakdown[len(solver_parts):])
  assert runtime["smooth.crb"] == 2 * model.nbody * 36 * 4
  assert runtime["smooth.local_inertia"] == 2 * model.nbody * 36 * 4
  assert runtime["smooth.cdof_dot"] == 2 * model.nv * 6 * 4
  assert runtime["simulation.body_wrench"] == 2 * model.nbody * 6 * 4
  assert runtime["sensor.sensor_meta"] == max(model.nsensor, 1) * 10 * 4
  assert runtime["sensor.subtree_runtime"] == (
      4 + 2 * model.nbody * 32 + 2 * max(model.nsensor, 1)) * 4


def test_default_limits_preserve_historical_ceilings():
  assert (BASE_NVIDIA_NV, BASE_MAX_PAIRS, BASE_MAX_SLOTS, BASE_MAX_ROWS) == (32, 32, 48, 96)
  lim = CapacityLimits()
  assert (lim.max_nv, lim.max_pairs, lim.max_slots, lim.max_rows) == (32, 32, 48, 96)


def test_overflow_messages_preserve_historical_text():
  import dataclasses
  model = _model()
  with pytest.raises(ValueError, match="bounds nv to 32"):
    check_capacity(dataclasses.replace(estimate_capacity(model, 1, 0, 0, 0), nv=33))
  with pytest.raises(ValueError, match=r"candidate contact pairs \(33\) exceeds capacity 32"):
    check_capacity(estimate_capacity(model, 1, 33, 0, 0))
  with pytest.raises(ValueError, match=r"total candidate contact slots \(49\) exceeds capacity 48"):
    check_capacity(estimate_capacity(model, 1, 0, 49, 0))
  with pytest.raises(ValueError, match=r"total candidate constraint rows \(97\) exceeds capacity 96"):
    check_capacity(estimate_capacity(model, 1, 0, 0, 97))


def test_user_limits_and_memory_budget_are_honored():
  model = _model()
  est = estimate_capacity(model, 1, 2, 2, 10)
  tight = CapacityLimits(max_pairs=1)
  with pytest.raises(CapacityOverflow, match="pairs"):
    check_capacity(est, tight)
  tiny_mem = CapacityLimits(memory_budget_bytes=1)
  with pytest.raises(CapacityOverflow, match="memory"):
    check_capacity(est, tiny_mem)
  assert check_capacity(est) is est


def test_user_limits_cannot_exceed_compiled_solver_abi():
  import dataclasses
  model = _model()
  with pytest.raises(ValueError, match=f"bounds nv to {NATIVE_MAX_NV}"):
    check_capacity(
        dataclasses.replace(estimate_capacity(model, 1, 0, 0, 0),
                            nv=NATIVE_MAX_NV + 1),
        CapacityLimits(max_nv=NATIVE_MAX_NV + 100))
  with pytest.raises(ValueError, match=f"capacity {NATIVE_MAX_ROWS}"):
    check_capacity(
        dataclasses.replace(estimate_capacity(model, 1, 0, 0, 0),
                            nr=NATIVE_MAX_ROWS + 1),
        CapacityLimits(max_rows=NATIVE_MAX_ROWS + 100))


def test_overflow_is_valueerror_and_batch_checked():
  model = _model()
  est = estimate_capacity(model, 17, 0, 0, 0)
  with pytest.raises(ValueError, match="batch"):
    check_capacity(est)
  with pytest.raises(TypeError):
    estimate_capacity("not-a-model", 1, 0, 0, 0)
  with pytest.raises(ValueError, match="positive"):
    estimate_capacity(model, 0, 0, 0, 0)


def test_signed_int32_buffer_addressing_precedes_user_memory_budget():
  # Arithmetic only: these dimensions must fail before any device allocation
  # even when a caller configures a very large byte budget.
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
  with pytest.raises(CapacityOverflow, match="workspace_debug.*signed 32-bit"):
    estimate_capacity(model, 32768, npairs=0, nslots=0, nr=256)

  with pytest.raises(CapacityOverflow, match="contact_jacobian.*signed 32-bit"):
    estimate_workspace(
        nv=64, npairs=0, nslots=100, nr=1, nr_joint=0, neq=0,
        batch=100000, solver_type=int(mujoco.mjtSolver.mjSOL_PGS))
  _check_i32_elements("boundary", SIGNED_I32_ELEMENT_LIMIT)
  with pytest.raises(CapacityOverflow, match="boundary.*signed 32-bit"):
    _check_i32_elements("boundary", SIGNED_I32_ELEMENT_LIMIT + 1)


@pytest.mark.parametrize("field,value", [
    ("batch", True), ("batch", 1.5), ("nv", -1), ("npairs", 1.5),
    ("nslots", -1), ("nr", True), ("nr_joint", -1), ("neq", 2.0),
])
def test_workspace_dimensions_reject_bool_fractional_and_negative(field, value):
  args = dict(nv=1, npairs=0, nslots=0, nr=0, nr_joint=0, neq=0, batch=1)
  args[field] = value
  with pytest.raises(ValueError):
    estimate_workspace(**args)


def test_public_capacity_batch_rejects_fractional_and_boolean_values():
  model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
  for batch in (True, 1.5, 0, -1):
    with pytest.raises(ValueError):
      estimate_capacity(model, batch, 0, 0, 0)


def estimate_workspace_expected(model, batch, npairs, nslots, nr):
  """Independent exact byte count for the allocated solver/map buffers."""
  b, v = batch, max(int(model.nv), 1)
  nc = nslots
  sizes = (
      max(b * nc * 36, 1), max(b * nc * 12, 1),
      max(b * nc * 6 * v, 1), max(b * max(npairs, 1), 1),
      max(b * nr * v, 1), max(b * (nr * nr + 7 * nr
                                    + primal_scratch_floats(v, nr, model.opt.solver)), 1),
      max(b * v, 1), max(b * v, 1), b, max(b * 10, 1),
      max(b * nc * 11, 1), max(b, 1), max(b, 1), max(b, 1),
      b * npairs, b * npairs, b * nc, b * nc, b * nr, b * nr, b * 4,
      b * max(npairs, 1), b * max((npairs + 255) // 256, 1) * 2,
      b * max(nc, 1), b * max((nc + 255) // 256, 1) * 2,
      b * max(nr, 1), b * max((nr + 255) // 256, 1) * 2,
  )
  names = (
      "contact_row_data", "contact_frame", "contact_jacobian", "pair_mask",
      "workspace_J", "workspace_debug", "out_force", "out_acc", "out_status",
      "out_diagnostics", "out_contact_force", "out_joint_force", "eq_active",
      "eq_active_default", "packed_to_pair", "pair_to_packed", "packed_to_slot",
      "slot_to_packed", "packed_to_logical_row", "logical_to_packed_row",
      "compaction_counts_overflow", "pair_scan_prefix", "pair_scan_blocks",
      "slot_scan_prefix", "slot_scan_blocks", "row_scan_prefix", "row_scan_blocks",
  )
  return tuple(zip(names, (n * 4 for n in sizes)))


def _jacobian_model(nv, setting="auto"):
  bodies = "".join(
      '<body><joint type="slide" axis="1 0 0"/><geom type="sphere" size=".01"/></body>'
      for _ in range(nv))
  return mujoco.MjModel.from_xml_string(
      f'<mujoco><option jacobian="{setting}"/><worldbody>{bodies}</worldbody></mujoco>')


@pytest.mark.parametrize("nv, expected", [(60, "dense"), (61, "sparse")])
def test_auto_jacobian_uses_pinned_sixty_dof_dispatch(nv, expected):
  model = _jacobian_model(nv)
  assert selected_jacobian_kind(model) == expected
  estimate = estimate_capacity(model, 1, 0, 0, 0)
  assert estimate.jacobian_kind == expected
  assert estimate.jacobian_auto_threshold == AUTO_JACOBIAN_DENSE_NV == 60


@pytest.mark.parametrize("setting, expected", [("dense", "dense"), ("sparse", "sparse")])
def test_explicit_jacobian_setting_overrides_auto_dispatch(setting, expected):
  model = _jacobian_model(61, setting)
  assert selected_jacobian_kind(model) == expected
  assert estimate_capacity(model, 1, 0, 0, 0).jacobian_kind == expected
