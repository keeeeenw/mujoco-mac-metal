# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Run installed demo scripts without a repository checkout.

This dispatcher selects a packaged script. It does not infer physics support:
each demonstration retains its own native/CPU/check/viewer arguments and the
backend's model admission and failure handling.
"""

from pathlib import Path
import runpy
import sys


def demo_directory():
  package = Path(__file__).resolve().parent
  installed = package / "examples"
  if installed.is_dir():
    return installed
  # Direct PYTHONPATH source runs predate the editable installation hook that
  # maps mujoco_metal.examples. Limit this route to the known source layout.
  source = package.parent / "examples"
  project = package.parent / "pyproject.toml"
  if source.is_dir() and project.is_file():
    import tomllib
    if tomllib.loads(project.read_text()).get("project", {}).get("name") == "mujoco-mac-metal":
      return source
  raise RuntimeError(
      "Installed demo scripts are missing. Reinstall a development wheel that "
      "includes examples, or use the documented source installation.")


def available_demos():
  """Return executable scripts only, excluding helpers and package files."""
  root = demo_directory()
  return tuple(path.stem for path in sorted(root.glob("*.py"))
               if not path.name.startswith("_")
               and '__name__ == "__main__"' in path.read_text())


def run_demo(name, arguments):
  # Exact inventory membership excludes traversal, arbitrary paths and modules.
  if name not in available_demos():
    raise ValueError(f"Unknown demo {name!r}; use 'mujoco-metal demo --list'")
  script = demo_directory() / (name + ".py")
  previous = sys.argv
  previous_path = sys.path
  # These demos also support direct script execution and use sibling helper
  # imports. run_path does not provide Python's script-directory lookup.
  # Scope it to this invocation, including any previously cached sibling
  # names from another checkout: the selected installation owns its helpers.
  sibling_names = {path.stem for path in script.parent.glob("*.py")
                   if not path.name.startswith("_")}
  previous_modules = {key: sys.modules[key] for key in sibling_names
                      if key in sys.modules}
  try:
    for key in sibling_names:
      sys.modules.pop(key, None)
    sys.path = [str(script.parent), *previous_path]
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")
  finally:
    sys.argv = previous
    sys.path = previous_path
    for key in sibling_names:
      sys.modules.pop(key, None)
    sys.modules.update(previous_modules)
