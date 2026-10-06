# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Regression tests for demo scripts executed from installed package paths."""

import sys
from types import ModuleType
from pathlib import Path

import pytest

from mujoco_metal.demo_cli import run_demo
from mujoco_metal import demo_cli


def test_record_pendulum_demo_help_uses_package_sibling_and_optional_pillow(
    monkeypatch,
):
  """Installed dispatch must not depend on cwd or optional Pillow for help."""
  monkeypatch.setitem(sys.modules, "PIL", None)
  with pytest.raises(SystemExit) as result:
    run_demo("record_pendulum", ["--help"])
  assert result.value.code == 0


def test_record_pendulum_runtime_reports_missing_optional_pillow(monkeypatch):
  monkeypatch.setitem(sys.modules, "PIL", None)
  with pytest.raises(RuntimeError, match="requires the optional Pillow"):
    run_demo("record_pendulum", ["unused.gif"])


def test_demo_dispatch_makes_packaged_sibling_helpers_importable(tmp_path,
                                                                  monkeypatch):
  (tmp_path / "fixture_helper.py").write_text(
      'VALUE = "selected sibling"\n')
  script = tmp_path / "fixture_demo.py"
  script.write_text(
      'if __name__ == "__main__":\n'
      '  import fixture_helper\n'
      '  from pathlib import Path\n'
      '  assert fixture_helper.VALUE == "selected sibling"\n'
      '  assert Path(fixture_helper.__file__).parent == Path(__file__).parent\n')
  monkeypatch.setattr(demo_cli, "demo_directory", lambda: Path(tmp_path))

  # A top-level helper from another checkout must not shadow the sibling
  # packaged alongside this selected demo, and its exact cache entry must be
  # restored after execution.
  cached = ModuleType("fixture_helper")
  cached.VALUE = "wrong cached copy"
  monkeypatch.setitem(sys.modules, "fixture_helper", cached)
  previous_path = sys.path[:]
  run_demo("fixture_demo", [])
  assert sys.path == previous_path
  assert sys.modules["fixture_helper"] is cached


def test_demo_dispatch_restores_path_and_module_cache_after_failure(
    tmp_path, monkeypatch,
):
  (tmp_path / "failing_helper.py").write_text('VALUE = "loaded"\n')
  script = tmp_path / "failing_demo.py"
  script.write_text(
      'if __name__ == "__main__":\n'
      '  from failing_helper import VALUE\n'
      '  assert VALUE == "loaded"\n'
      '  raise RuntimeError("fixture failure")\n')
  monkeypatch.setattr(demo_cli, "demo_directory", lambda: Path(tmp_path))
  sys.modules.pop("failing_helper", None)
  previous_path = sys.path[:]
  with pytest.raises(RuntimeError, match="fixture failure"):
    run_demo("failing_demo", [])
  assert sys.path == previous_path
  assert "failing_helper" not in sys.modules
