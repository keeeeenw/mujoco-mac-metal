"""CPU oracle gate for the scalable marble feed demo."""

from __future__ import annotations

import os

import pytest

from examples import scalable_marble_factory as demo


def test_marble_factory_exceeds_old_workspace_caps_with_real_contacts():
  result = demo.run(steps=30, mode="cpu", check=True)
  assert result["nv"] == 90
  assert result["candidate_pairs"] == 45
  assert result["allocated_slots"] == 45
  assert result["allocated_rows"] == 390
  assert result["estimated_workspace_bytes"] < 16 * 1024 * 1024
  assert result["peak_active_pairs"] == 45
  assert result["peak_contacts"] == 45
  assert result["peak_active_rows"] == 270
  assert result["finite"]
  assert result["max_qpos_error"] == 0.0
  assert result["max_qvel_error"] == 0.0
  assert result["max_qacc_error"] == 0.0
  assert result["peak_native_contact_slots"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1",
                    reason="serialized native scalable marble qualification")
def test_native_marble_factory_has_real_contacts_and_checkpoint_replay():
  result = demo.run(steps=30, mode="metal", check=True, restore_check=True)
  assert result["peak_native_broadphase_pairs"] == 45
  assert result["peak_native_contact_slots"] == 45
  assert result["peak_native_active_rows"] == 270
  assert result["checkpoint_replay_bitwise"] is True
