# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Pinned instrumented trace build: equivalence, inertness, per-seed traces.

The workspace-local trace build compiles the pinned 3.10.0 public source
with an additive env-gated per-seed SDF descent trace (MJCSDFTRACE lines on
stderr when MUJOCO_SDF_TRACE is set; see the trace source). These host-only
tests prove, on committed fixtures:

- the trace library loads and reports the pinned version (3010000);
- the trace source differs from pristine only by the guarded trace
  statements (no computation change);
- with the trace env unset, the trace build reproduces the installed pip
  oracle bit-identically on the torus SDF fixture (ncon, qacc, every
  contact dist) and emits no trace output;
- with the trace env set, the same run emits ordered per-seed lines (seed
  order/count preserved: 8 seeds for sdf_initpoints=8) while the numbers
  stay identical to the unset run (inert instrumentation);
- the helper runner resolves workspace-local paths without hardcoded
  home directories.
"""

import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_HELPER = os.path.join(_ROOT, "project", "metal", "trace_sdf.py")
_TRACE_SRC = os.path.join(
    _ROOT, ".upstream", "mujoco-trace", "src", "engine",
    "engine_collision_sdf.c")
_PRISTINE_SRC = os.path.join(
    _ROOT, ".upstream", "mujoco-3.10.0", "src", "engine",
    "engine_collision_sdf.c")
_TRACE_LIB = os.path.join(
    _ROOT, ".upstream", "mujoco-trace", "build-fixed", "lib",
    "libmujoco.dylib")

_TORUS_XML = """
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="100" tolerance="1e-8" sdf_initpoints="8"/>
      <extension><plugin plugin="mujoco.sdf.torus">
        <instance name="shape"/>
      </plugin></extension>
      <asset><mesh name="shape_mesh"><plugin instance="shape"/></mesh></asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="probe" pos="0.3 -0.2 0.0">
          <freejoint/><geom name="probe_sphere" type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <contact><pair geom1="plugin_sdf" geom2="probe_sphere" condim="1"/></contact>
    </mujoco>
  """


def _needs_trace_build():
  return pytest.mark.skipif(
      not os.path.exists(_TRACE_LIB), reason="trace build absent")


def _write_xml(tmp_path):
  p = tmp_path / "torus_trace.xml"
  p.write_text(_TORUS_XML)
  return str(p)


def _run(xml, trace=False):
  env = dict(os.environ)
  if trace:
    env["MUJOCO_SDF_TRACE"] = "1"
  else:
    env.pop("MUJOCO_SDF_TRACE", None)
  proc = subprocess.run(
      [sys.executable, _HELPER, xml], capture_output=True, text=True,
      env=env, timeout=120)
  assert proc.returncode == 0, proc.stderr[-2000:]
  return json.loads(proc.stdout), proc.stderr


@_needs_trace_build()
def test_trace_lib_reports_pinned_version():
  # Subprocess-isolated: loading the trace libmujoco in-process would place
  # a second libmujoco image in this process and break the installed
  # oracle's own plugin dlopen (@rpath) later in the session.
  proc = subprocess.run(
      [sys.executable, _HELPER, "--version"], capture_output=True, text=True,
      timeout=120)
  assert proc.returncode == 0, proc.stderr[-2000:]
  assert json.loads(proc.stdout)["version"] == 3010000


@_needs_trace_build()
def test_trace_source_diff_is_guarded_output_only():
  with open(_TRACE_SRC) as f:
    trace = f.read()
  with open(_PRISTINE_SRC) as f:
    pristine = f.read()
  # The only added executable statements are env-gated fprintf lines and
  # the enablement helper; no branch predicate, tolerance, seed selection,
  # iteration count or dynamics line is altered.
  assert "MUJOCO_SDF_TRACE" in trace
  assert trace.count('"MJCSDFTRACE ') == 6
  for marker in ("amin = 1e-4", "dist - dist0 > wolfe", "sdf_initpoints",
                 "mju_Halton(j, 2)"):
    assert marker in trace and marker in pristine
  # Pristine lines are preserved verbatim inside the trace file: every
  # pristine line appears in order in the trace source.
  it = iter(trace.splitlines())
  preserved = 0
  for line in pristine.splitlines():
    for tline in it:
      if tline == line:
        preserved += 1
        break
    else:
      raise AssertionError("pristine line lost: " + line[:80])
  assert preserved == len(pristine.splitlines())


@_needs_trace_build()
def test_trace_build_matches_installed_oracle_torus(tmp_path):
  import mujoco
  import numpy as np
  xml = _write_xml(tmp_path)
  numbers, stderr = _run(xml, trace=False)
  assert stderr == ""
  assert numbers["version"] == 3010000
  model = mujoco.MjModel.from_xml_string(_TORUS_XML)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert numbers["ncon"] == int(data.ncon) == 8
  np.testing.assert_array_equal(
      np.asarray(numbers["qacc"], dtype=np.float64),
      np.asarray(data.qacc, dtype=np.float64))
  for tc, c in zip(numbers["contacts"], range(data.ncon)):
    assert tc["dist"] == float(data.contact[c].dist)
    np.testing.assert_array_equal(
        np.asarray(tc["pos"], dtype=np.float64),
        np.asarray(data.contact[c].pos, dtype=np.float64))
    np.testing.assert_array_equal(
        np.asarray(tc["normal"], dtype=np.float64),
        np.asarray(data.contact[c].frame[:3], dtype=np.float64))


@_needs_trace_build()
def test_trace_env_emits_ordered_per_seed_lines_without_changing_numbers(
    tmp_path):
  xml = _write_xml(tmp_path)
  off, off_err = _run(xml, trace=False)
  on, on_err = _run(xml, trace=True)
  assert off == on
  lines = [ln for ln in on_err.splitlines() if "MJCSDFTRACE" in ln]
  assert len(lines) > 8
  seeds = [ln for ln in lines if "MJCSDFTRACE seed" in ln]
  assert len(seeds) == 8
  # Seed order/count preserved: i=0..7 with matching Halton indices.
  for k, ln in enumerate(seeds):
    assert f"i={k} halton={k}" in ln
  stages = [ln for ln in lines if "MJCSDFTRACE stage" in ln]
  assert len(stages) == 16  # COLLISION + INTERSECTION per seed
  assert any("type=COLLISION" in ln for ln in stages)
  assert any("type=INTERSECTION" in ln for ln in stages)
  trials = [ln for ln in lines if "MJCSDFTRACE trial" in ln]
  assert trials
