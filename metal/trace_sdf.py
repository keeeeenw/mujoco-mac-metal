#!/usr/bin/env python3
# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Run an SDF XML fixture through the workspace-local instrumented pinned
MuJoCo build and print QACC/contact summary as JSON on stdout.

The instrumented build is compiled from the pinned 3.10.0 public source with
an additive env-gated per-seed descent trace (see the MJCSDFTRACE statements
in the workspace-local trace source). With MUJOCO_SDF_TRACE set, per-seed
lines (seed, line-search trials with alpha/dist/Wolfe operands, stage
terminals) go to stderr. Unset, no trace output is produced and no
computation changes: every trace statement is a guarded read of existing
locals, and production physics never enables this path.

Usage: trace_sdf.py <model-xml>

Gate poses are baked into the XML (body placement), so the model-default
forward runs at the exact represented gate state.
"""

import ctypes
import json
import os
import sys
import tempfile

_ROOT = os.environ.get("MUJOCO_METAL_TRACE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRACE_LIB = os.path.join(
    _ROOT, ".upstream", "mujoco-trace", "build-fixed", "lib",
    "libmujoco.dylib")
_TRACE_PLUGIN = os.path.join(
    _ROOT, ".upstream", "mujoco-trace", "build-fixed", "lib",
    "libsdf_plugin.dylib")


def _load():
  if not os.path.exists(_TRACE_LIB):
    raise FileNotFoundError("trace libmujoco not built: " + _TRACE_LIB)
  lib = ctypes.CDLL(_TRACE_LIB)
  lib.mj_version.restype = ctypes.c_int
  lib.mj_loadPluginLibrary.argtypes = [ctypes.c_char_p]
  lib.mj_loadPluginLibrary.restype = None
  lib.mj_loadXML.argtypes = [
      ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
  lib.mj_loadXML.restype = ctypes.c_void_p
  lib.mj_makeData.argtypes = [ctypes.c_void_p]
  lib.mj_makeData.restype = ctypes.c_void_p
  lib.mj_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  lib.mj_forward.restype = None
  lib.mj_printData.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
  lib.mj_printData.restype = None
  lib.mj_printFormattedData.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
  lib.mj_printFormattedData.restype = None
  lib.mj_deleteData.argtypes = [ctypes.c_void_p]
  lib.mj_deleteData.restype = None
  lib.mj_deleteModel.argtypes = [ctypes.c_void_p]
  lib.mj_deleteModel.restype = None
  if os.path.exists(_TRACE_PLUGIN):
    lib.mj_loadPluginLibrary(_TRACE_PLUGIN.encode())
  return lib


def _parse_print(path):
  qacc, contacts, cur = [], [], None
  section = None
  with open(path) as f:
    for line in f:
      s = line.strip()
      if s == "QACC":
        section = "qacc"
        continue
      if s == "CONTACT":
        section = "contact"
        continue
      if s.startswith("EFC_TYPE"):
        section = None
        continue
      if section == "qacc":
        if s and (s[0].isdigit() or s[0] in "+-."):
          qacc.extend(float(x) for x in s.split())
        elif s:
          section = None
      elif section == "contact":
        if s.endswith(":") and s[:-1].strip().isdigit():
          cur = {}
          contacts.append(cur)
        elif cur is not None:
          if s.startswith("dist"):
            cur["dist"] = float(s.split()[-1])
          elif s.startswith("pos"):
            cur["pos"] = [float(x) for x in s.split()[-3:]]
          elif s.startswith("frame"):
            vals = [float(x) for x in s.split()[-9:]]
            cur["normal"] = vals[:3]
  return qacc, contacts


def main(xml_path):
  lib = _load()
  err = ctypes.create_string_buffer(1000)
  model = lib.mj_loadXML(xml_path.encode(), None, err, 1000)
  if not model:
    sys.stderr.write("mj_loadXML failed: " + err.value.decode() + "\n")
    return 2
  data = lib.mj_makeData(model)
  if not data:
    sys.stderr.write("mj_makeData failed\n")
    return 2
  lib.mj_forward(model, data)
  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".txt", delete=False) as f:
    out_path = f.name
  # Full-precision dump: the default print format keeps only 3 significant
  # digits, which would mask build-identity comparison.
  lib.mj_printFormattedData(model, data, out_path.encode(), b"%.17g")
  qacc, contacts = _parse_print(out_path)
  os.unlink(out_path)
  lib.mj_deleteData(data)
  lib.mj_deleteModel(model)
  print(json.dumps({
      "version": int(lib.mj_version()),
      "ncon": len(contacts),
      "qacc": qacc,
      "contacts": contacts,
  }))
  return 0


if __name__ == "__main__":
  if len(sys.argv) == 2 and sys.argv[1] == "--version":
    lib = _load()
    print(json.dumps({"version": int(lib.mj_version())}))
    sys.exit(0)
  sys.exit(main(sys.argv[1]))
