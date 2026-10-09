# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Public flex contact trajectories must not depend on observer readbacks."""
import os

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUJOCO_METAL_RUN_GPU") != "1", reason="native opt-in")
@pytest.mark.parametrize(("dof", "shell"), [
    ("trilinear", False), ("quadratic", False),
    ("trilinear", True), ("quadratic", True),
])
def test_full_friction_lifecycle_without_contact_host_reads(monkeypatch, dof, shell):
  import torch
  from mujoco_metal.flex_contact import FlexContactProgram
  from test_flex_contact_rows import (
      test_native_public_interpolated_sphere_flex_cc_friction_trajectory_B2)

  original_run = FlexContactProgram.run_device
  original_cpu = torch.Tensor.cpu
  in_pipeline = False

  def forbid_contact_readback(tensor, *args, **kwargs):
    if in_pipeline and tensor.device.type == "mps":
      raise AssertionError("contact pipeline read physics data onto the host")
    return original_cpu(tensor, *args, **kwargs)

  def run_contact(self, *args, **kwargs):
    nonlocal in_pipeline
    in_pipeline = True
    try:
      return original_run(self, *args, **kwargs)
    finally:
      in_pipeline = False

  monkeypatch.setattr(torch.Tensor, "cpu", forbid_contact_readback)
  monkeypatch.setattr(FlexContactProgram, "run_device", run_contact)
  # Execute the unchanged canonical J/R/aref, force, four-step trajectory,
  # bitwise checkpoint replay and cold-reset checks. No numeric relaxation.
  test_native_public_interpolated_sphere_flex_cc_friction_trajectory_B2(dof, shell)
