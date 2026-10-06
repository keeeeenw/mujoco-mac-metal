# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Host-only dispatcher and source/wheel asset-location contracts."""

from pathlib import Path
import sys

import pytest

from mujoco_metal import __main__ as cli
from mujoco_metal import demo_cli


def test_demo_listing_contains_real_executables_without_importing_torch(capsys, monkeypatch):
  import builtins
  original = builtins.__import__

  def reject_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
      raise AssertionError("listing demos must not import Torch or compile Metal")
    return original(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", reject_torch)
  assert cli.main(["demo", "--list"]) == 0
  names = capsys.readouterr().out.splitlines()
  assert "haptic_calligraphy" in names
  assert "marble_music_machine" in names
  assert "__init__" not in names
  assert names == sorted(names)


def test_demo_dispatch_preserves_arguments_and_restores_process_argv(monkeypatch):
  calls = []
  previous = sys.argv

  def observe(path, *, run_name):
    calls.append((Path(path), run_name, sys.argv.copy()))
    raise RuntimeError("demo failure")

  monkeypatch.setattr(demo_cli.runpy, "run_path", observe)
  with pytest.raises(RuntimeError, match="demo failure"):
    cli.main(["demo", "haptic_calligraphy", "--mode", "cpu", "--headless", "--steps", "3"])
  assert sys.argv is previous
  assert calls[0][0] == demo_cli.demo_directory() / "haptic_calligraphy.py"
  assert calls[0][1] == "__main__"
  assert calls[0][2][1:] == ["--mode", "cpu", "--headless", "--steps", "3"]


@pytest.mark.parametrize("name", ["../haptic_calligraphy", "/tmp/demo", "unknown"])
def test_demo_rejects_paths_and_unknown_scripts(name):
  with pytest.raises(ValueError, match="Unknown demo"):
    demo_cli.run_demo(name, [])


def test_demo_prefers_installed_assets_and_does_not_require_checkout(tmp_path, monkeypatch):
  package = tmp_path / "site-packages" / "mujoco_metal"
  assets = package / "examples"
  assets.mkdir(parents=True)
  (assets / "fixture.py").write_text('if __name__ == "__main__": pass\n')
  monkeypatch.setattr(demo_cli, "__file__", str(package / "demo_cli.py"))
  assert demo_cli.demo_directory() == assets
  assert demo_cli.available_demos() == ("fixture",)
  (assets / "fixture.py").unlink()
  assets.rmdir()
  with pytest.raises(RuntimeError, match="missing"):
    demo_cli.demo_directory()
