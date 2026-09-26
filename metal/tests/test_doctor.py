# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Install and CPU-only behavior checks for the doctor command."""

import builtins
import json

from mujoco_metal import __main__ as cli


def test_default_doctor_skips_torch_import(monkeypatch):
  real_import = builtins.__import__

  def reject_torch_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
      raise AssertionError("default doctor must not import torch")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", reject_torch_import)
  result = cli.doctor()
  assert result["command"] == "doctor"
  assert result["python"]["version"]
  assert result["platform"]["system"]
  assert result["mujoco"]["version"]
  assert result["torch"]["imported"] is False
  assert result["gpu"] == {
      "requested": False,
      "tested": False,
      "passed": None,
      "status": "not_requested",
  }


def test_doctor_json_command_is_parseable(capsys):
  assert cli.main(["doctor", "--json"]) == 0
  result = json.loads(capsys.readouterr().out)
  assert result["command"] == "doctor"
  assert result["gpu"]["tested"] is False


def test_gpu_request_rejects_torch_mps_fallback_before_import(monkeypatch):
  monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
  real_import = builtins.__import__

  def reject_torch_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
      raise AssertionError("fallback precheck must run before torch import")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", reject_torch_import)
  result = cli.doctor(gpu=True)
  assert result["gpu"]["status"] == "mps_fallback_enabled"
  assert result["gpu"]["tested"] is False
  assert result["gpu"]["passed"] is False
  assert "set it to 0" in result["gpu"]["action"]


def test_gpu_json_failure_is_actionable_and_nonzero(monkeypatch, capsys):
  monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
  assert cli.main(["doctor", "--gpu", "--json"]) == 1
  result = json.loads(capsys.readouterr().out)
  assert result["gpu"]["tested"] is False
  assert result["gpu"]["passed"] is False
  assert result["gpu"]["action"]


def test_preflight_command_is_preserved(capsys):
  assert cli.main(["preflight", "--json"]) == 0
  result = json.loads(capsys.readouterr().out)
  assert "actual_mujoco_version" in result
  assert result["gpu_qualified"] is False
  assert result["actuation_shader"]["sha256"]
  assert result["stages"]["contact_free_forces_euler_v1"]
  assert result["stages"]["scalar_motor_force"]
  assert (
      "narrowly GPU-qualified"
      in result["stages"]["contact_free_motor_euler_v1"]
  )
