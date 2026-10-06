# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Individual pinned bundled-plugin inventory, separate from custom extensions."""
import json
import subprocess
import sys

from mujoco_metal.registry import (
    Execution, Implementation, Qualification, REQUIREMENTS,
)


def test_pinned_bundled_plugins_match_clean_upstream_registration_table():
  # A clean process excludes plugins loaded by other tests or applications.
  # mjpPlugin.name is the first field in pinned mjplugin.h; only that field is
  # read. We do not call any plugin's simulation callbacks or create GPU state.
  script = r'''
import ctypes
import json
from pathlib import Path
import mujoco
assert mujoco.__version__ == "3.10.0", mujoco.__version__
root = Path(mujoco.__file__).parent
libraries = [p for p in root.iterdir() if p.is_file()
             and p.name.startswith(("libmujoco", "mujoco"))
             and (p.name.endswith((".dylib", ".dll")) or ".so" in p.name)]
assert len(libraries) == 1, [p.name for p in libraries]
library = ctypes.CDLL(str(libraries[0]))
library.mjp_pluginCount.argtypes = []
library.mjp_pluginCount.restype = ctypes.c_int
library.mjp_getPluginAtSlot.argtypes = [ctypes.c_int]
library.mjp_getPluginAtSlot.restype = ctypes.c_void_p
names = []
for slot in range(library.mjp_pluginCount()):
    plugin = library.mjp_getPluginAtSlot(slot)
    assert plugin
    name = ctypes.cast(plugin, ctypes.POINTER(ctypes.c_char_p))[0]
    assert name
    names.append(name.decode("utf-8"))
print(json.dumps(sorted(names)))
'''
  result = subprocess.run([sys.executable, "-c", script], check=True,
                          capture_output=True, text=True, timeout=20)
  registered = json.loads(result.stdout)
  rows = [r for r in REQUIREMENTS if r.id.startswith("REQ-PLUG-")]
  assert sorted(r.name for r in rows) == registered
  assert len(rows) == 8
  for row in rows:
    assert row.implementation == Implementation.NATIVE_GPU
    assert row.qualification == Qualification.UNQUALIFIED
    assert row.execution == Execution.DEVICE
    assert row.milestone == "019"
    assert row.source_ref.startswith("plugin/")
    assert "reject unknown plugins" in row.admission
    assert row.entry_point != "none"
    assert len(row.tests) >= 2
