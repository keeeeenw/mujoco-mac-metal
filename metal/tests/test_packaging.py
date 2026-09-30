# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Packaging test: every runtime-required Metal shader exists (CPU only).

Collects shader references in both literal (`shaders/name.metal`) and
path-joined (`Path(...) / "shaders" / "name.metal"`) forms from the package
sources, then checks each file exists. A fixture-tree negative case proves
removal is detected. No GPU work is executed here.
"""

import re
from pathlib import Path

import mujoco_metal

_LITERAL_RE = re.compile(r"""shaders/([A-Za-z0-9_]+\.metal)""")
_JOINED_RE = re.compile(r"""["']shaders["']\s*/\s*["']([A-Za-z0-9_]+\.metal)["']""")


def _package_dir():
  return Path(mujoco_metal.__file__).resolve().parent


def referenced_shaders(package_dir=None):
  """All `*.metal` shader basenames referenced by package sources."""
  pkg = Path(package_dir) if package_dir is not None else _package_dir()
  refs = set()
  for path in sorted(pkg.glob("*.py")):
    text = path.read_text()
    for match in _LITERAL_RE.finditer(text):
      refs.add(match.group(1))
    for match in _JOINED_RE.finditer(text):
      refs.add(match.group(1))
  return refs


def find_missing_shaders(package_dir=None):
  """Subset of referenced shaders absent from `<package>/shaders/`."""
  pkg = Path(package_dir) if package_dir is not None else _package_dir()
  return sorted(
      name for name in referenced_shaders(pkg)
      if not (pkg / "shaders" / name).is_file())


def test_all_referenced_shaders_exist_in_source():
  refs = referenced_shaders()
  assert refs, "expected shader references in package sources"
  assert find_missing_shaders() == []


def test_missing_shader_is_detected(tmp_path):
  # Negative control: a reference without its file must be reported.
  # Mirrors the cb3d1bd defect class (unconditional shader read, file absent).
  pkg = tmp_path / "mujoco_metal"
  (pkg / "shaders").mkdir(parents=True)
  (pkg / "mod.py").write_text(
      'X = Path(__file__).parent / "shaders" / "ghost.metal"\n'
      'Y = "shaders/other.metal"\n')
  (pkg / "shaders" / "other.metal").write_text("// placeholder\n")
  assert referenced_shaders(pkg) == {"ghost.metal", "other.metal"}
  assert find_missing_shaders(pkg) == ["ghost.metal"]


def test_manifest_covers_shaders_directory():
  # The build must ship every shader in the directory, not just the ones
  # referenced today: compare the manifest glob against directory contents.
  # NOTE: only an editable install exists in this environment and no build
  # backend (setuptools/wheel/build) is installed, so no distributable
  # wheel can be inspected here; the manifest string is static evidence.
  pyproject = _package_dir().parent / "pyproject.toml"
  assert pyproject.is_file(), pyproject
  text = pyproject.read_text()
  assert "shaders/*.metal" in text
  pkg = _package_dir()
  on_disk = sorted(p.name for p in (pkg / "shaders").glob("*.metal"))
  assert on_disk, "shaders directory must not be empty"
  refs = referenced_shaders()
  assert set(refs) <= set(on_disk)
