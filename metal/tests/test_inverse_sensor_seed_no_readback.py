# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Low-plane ownership uses structural metadata, never tensor truth values."""
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize("value", [0.0, .125])
def test_lazy_inverse_sensor_seed_never_branches_on_tensor_contents(value, monkeypatch):
  torch = pytest.importorskip("torch")
  from mujoco_metal.finite_difference import _inverse_fd_seed_sensor_low
  seed = torch.full((2, 3), value, dtype=torch.float32)
  program = SimpleNamespace(_s_state_out_low=None)
  sim = SimpleNamespace(_raw_sensor_program=program)
  def reject_truth(self):
    raise AssertionError("tensor truth causes numerical host synchronization")
  monkeypatch.setattr(torch.Tensor, "__bool__", reject_truth)
  _inverse_fd_seed_sensor_low(sim, seed)
  assert program._s_state_out_low is seed
