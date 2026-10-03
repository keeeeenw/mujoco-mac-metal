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
    DENSE_ROW_THRESHOLD,
    CapacityLimits,
    CapacityOverflow,
    check_capacity,
    estimate_capacity,
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
  assert "delassus_W" in names and "warmstart" in names


def test_default_limits_preserve_historical_ceilings():
  assert (BASE_NVIDIA_NV, BASE_MAX_PAIRS, BASE_MAX_SLOTS, BASE_MAX_ROWS) == (32, 16, 24, 96)
  lim = CapacityLimits()
  assert (lim.max_nv, lim.max_pairs, lim.max_slots, lim.max_rows) == (32, 16, 24, 96)


def test_overflow_messages_preserve_historical_text():
  import dataclasses
  model = _model()
  with pytest.raises(ValueError, match="bounds nv to 32"):
    check_capacity(dataclasses.replace(estimate_capacity(model, 1, 0, 0, 0), nv=33))
  with pytest.raises(ValueError, match=r"candidate contact pairs \(17\) exceeds capacity 16"):
    check_capacity(estimate_capacity(model, 1, 17, 0, 0))
  with pytest.raises(ValueError, match=r"total candidate contact slots \(25\) exceeds capacity 24"):
    check_capacity(estimate_capacity(model, 1, 0, 25, 0))
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


def test_overflow_is_valueerror_and_batch_checked():
  model = _model()
  est = estimate_capacity(model, 17, 0, 0, 0)
  with pytest.raises(ValueError, match="batch"):
    check_capacity(est)
  with pytest.raises(TypeError):
    estimate_capacity("not-a-model", 1, 0, 0, 0)
  with pytest.raises(ValueError, match="positive"):
    estimate_capacity(model, 0, 0, 0, 0)
