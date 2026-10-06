# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Install and CPU-only behavior checks for the doctor command."""

import builtins
import json
import hashlib
from pathlib import Path

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


def test_preflight_inventories_actual_installed_shaders_and_requirements(monkeypatch):
  from mujoco_metal.registry import REQUIREMENTS
  real_import = builtins.__import__

  def reject_torch_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
      raise AssertionError("shader and requirement inventory must remain host-only")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", reject_torch_import)
  result = cli.preflight(include_inventory=True)
  package = Path(cli.__file__).resolve().parent
  shaders = sorted((package / "shaders").glob("*.metal"))
  assert set(result["shaders"]) == {path.stem for path in shaders}
  assert len(shaders) > 20  # Include newer runtime operators, not only the old list.
  for path in shaders:
    actual = result["shaders"][path.stem]
    assert Path(actual["path"]).resolve() == path.resolve()
    assert actual["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
  assert [row["id"] for row in result["requirements"]] == [
      requirement.id for requirement in REQUIREMENTS]
  assert sum(result["requirement_counts"].values()) == len(REQUIREMENTS)
  assert result["diagnostic_execution"] == "host_inventory_only"
  assert result["gpu_qualified"] is False
  assert "not a physics completion percentage" in result["requirement_counts_scope"]
