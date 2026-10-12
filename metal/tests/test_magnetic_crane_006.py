# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""The lifecycle demo uses real native contacts, replay and working viewer flags."""

import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def demo(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
  return importlib.import_module("magnetic_crane")


def test_slot_pair_labels_follow_actual_compiled_contact_offsets(demo):
  from mujoco_metal.coupled_constraints import lower_coupled_constraints
  model = demo._load_model()
  descriptor = lower_coupled_constraints(model)
  labels = demo._slot_pairs(SimpleNamespace(
      _coupled_constraints=SimpleNamespace(descriptor=descriptor)))
  assert len(labels) == descriptor.ncontacts_max
  for i in range(descriptor.npairs):
    wanted = (int(descriptor.geom1[i]), int(descriptor.geom2[i]))
    assert all(pair == wanted for pair in labels[
        descriptor.pair_contact_offset[i]:descriptor.pair_contact_offset[i+1]])


@pytest.mark.parametrize("args", [
    ["--headless", "--check", "--steps", "100"],
    ["--viewer-seconds", "nan"], ["--viewer-seconds", "0"],
    ["--headless", "--viewer"], ["--viewer", "--mode", "cpu"],
])
def test_invalid_demo_invocations_reject_before_physics_or_viewer(demo, args, monkeypatch):
  def unexpected(*_args, **_kwargs):
    raise AssertionError("invalid invocation must not launch physics/viewer")
  monkeypatch.setattr(demo, "run", unexpected)
  monkeypatch.setattr(demo, "play", unexpected)
  monkeypatch.setattr(sys, "argv", ["magnetic_crane.py", *args])
  with pytest.raises(SystemExit):
    demo.main()


def test_valid_viewer_command_reaches_viewer_without_headless_run(demo, monkeypatch):
  calls = []
  monkeypatch.setattr(demo, "play", lambda seconds: calls.append(seconds) or {"viewer_steps": 2})
  monkeypatch.setattr(sys, "argv", ["magnetic_crane.py", "--viewer", "--viewer-seconds", "3"])
  demo.main()
  assert calls == [3.]


def test_replay_comparator_rejects_dtype_changes_and_signed_zero(demo):
  with pytest.raises(AssertionError):
    demo._same_checkpoint(np.asarray([1.], np.float64), np.asarray([1.], np.float32))
  with pytest.raises(AssertionError, match="literal bits"):
    demo._same_checkpoint(np.asarray([-0.], np.float32), np.asarray([0.], np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native full crane")
def test_complete_crane_native_contacts_source_release_and_checkpoint_replay(demo):
  result = demo.run(steps=900, mode="metal", check=True,
                    record=os.getenv("MUJOCO_METAL_CRANE_RECORD"))
  if os.getenv("MUJOCO_METAL_CRANE_REPORT"):
    Path(os.environ["MUJOCO_METAL_CRANE_REPORT"]).write_text(json.dumps(result, indent=2)+"\n")
  assert result["checkpoint_replay_exact"] is True
  assert result["native_bin_hits"] > 20
  assert result["native_latched_bin_hits"] == 0
  assert result["native_weld_before"] > 1
  assert result["native_weld_after"] < 1e-4
