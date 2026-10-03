# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned mjtState selector bitmask ABI checks (CPU-safe)."""

from mujoco_metal.native_api import StateSpec


def test_state_spec_matches_pinned_mujoco_310_constants():
  assert StateSpec.TIME == 1
  assert StateSpec.QPOS == 2
  assert StateSpec.QVEL == 4
  assert StateSpec.ACT == 8
  assert StateSpec.HISTORY == 16
  assert StateSpec.WARMSTART == 32
  assert StateSpec.CTRL == 64
  assert StateSpec.QFRC_APPLIED == 128
  assert StateSpec.XFRC_APPLIED == 256
  assert StateSpec.EQ_ACTIVE == 512
  assert StateSpec.MOCAP_POS == 1024
  assert StateSpec.MOCAP_QUAT == 2048
  assert StateSpec.USERDATA == 4096
  assert StateSpec.PLUGIN == 8192
  assert StateSpec.PHYSICS == 30
  assert StateSpec.FULLPHYSICS == 8223
  assert StateSpec.USER == 8128
  assert StateSpec.INTEGRATION == StateSpec.ALL == 16383
