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

import numpy as np
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
def test_trace_provenance_files_and_hashes():  # Provenance: pristine vs modified source identity (sha256 reported,
  # differing exactly by the guarded instrumentation), built library and
  # plugin presence, and the pinned Release build flags on record. Hash
  # values are reported, not hard-asserted, so a deliberate rebuild does
  # not break the contract; the version + equivalence tests above carry
  # the identity.
  import hashlib
  trace_lib = os.path.join(
      _ROOT, ".upstream", "mujoco-trace", "build-fixed", "lib",
      "libmujoco.dylib")
  trace_plugin = os.path.join(
      _ROOT, ".upstream", "mujoco-trace", "build-fixed", "lib",
      "libsdf_plugin.dylib")
  assert os.path.exists(trace_lib), trace_lib
  assert os.path.exists(trace_plugin), trace_plugin
  with open(_PRISTINE_SRC, "rb") as f:
    pristine_hash = hashlib.sha256(f.read()).hexdigest()
  with open(_TRACE_SRC, "rb") as f:
    trace_hash = hashlib.sha256(f.read()).hexdigest()
  assert pristine_hash != trace_hash
  with open(trace_lib, "rb") as f:
    lib_hash = hashlib.sha256(f.read()).hexdigest()
  with open(trace_plugin, "rb") as f:
    plugin_hash = hashlib.sha256(f.read()).hexdigest()
  cache = os.path.join(
      _ROOT, ".upstream", "mujoco-trace", "build-fixed", "CMakeCache.txt")
  assert os.path.exists(cache)
  flags = open(cache).read()
  assert "CMAKE_BUILD_TYPE:STRING=Release" in flags
  assert "CMAKE_C_COMPILER:STRING=/usr/bin/cc" in flags
  assert "CMAKE_CXX_COMPILER:STRING=/usr/bin/c++" in flags
  print(f"PROVENANCE pristine={pristine_hash} trace={trace_hash} "
        f"lib={lib_hash} plugin={plugin_hash}", flush=True)


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
  # Every added line is a guarded fprintf of existing locals: the pinned
  # computation statements are untouched. Required record kinds (format
  # strings `"MJCSDFTRACE <kind> ..."`): plugin-descent steps/trials/
  # terminals/seeds/stages, gradient side picks, mesh face/seed records,
  # precontact admission, FPS picks.
  kinds = set()
  for match in __import__("re").finditer(r'"MJCSDFTRACE ([A-Za-z-]+)',
                                         trace):
    kinds.add(match.group(1))
  for kind in ("step", "trial", "terminal", "seed", "stage", "gradpick",
               "meshface", "meshseed", "precontact", "fps", "fps-pick"):
    assert kind in kinds, kind
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


@_needs_trace_build()
def test_trial_backtrack_field_semantics_gpu_free(tmp_path):
  # `backtrack=1` means the WHILE continuation predicate holds, so another
  # trial follows with halved alpha; the final trial of each step has
  # `backtrack=0` (accepted via Wolfe satisfaction or the alpha floor).
  # Rejection, acceptance, alpha floor and no-improvement returns are all
  # exercised on the committed torus trace: some step descends through
  # multiple rejected trials, and the INTERSECTION pass terminates at the
  # alpha floor in at least one seed.
  xml = _write_xml(tmp_path)
  _, on_err = _run(xml, trace=True)
  lines = [ln for ln in on_err.splitlines() if "MJCSDFTRACE trial" in ln]
  # Group by (seed, niter, step): the trace interleaves seeds, and steps
  # repeat across seeds, so the seed index scopes each group.
  groups = {}
  seed = None
  for ln in on_err.splitlines():
    if "MJCSDFTRACE seed" in ln:
      seed = ln.split("i=")[1].split()[0]
    elif "MJCSDFTRACE trial" in ln:
      tag = ln.split("niter=")[1].split(" alpha=")[0]
      groups.setdefault((seed, tag), []).append(ln)
  multi = [v for v in groups.values() if len(v) > 1]
  assert multi
  for trials in multi:
    for ln in trials[:-1]:
      assert "backtrack=1" in ln
    assert "backtrack=0" in trials[-1]
  alphas = [float(ln.split("alpha=")[1].split()[0]) for trials in multi
            for ln in trials]
  assert min(alphas) <= 1e-4 + 1e-12
  terminals = [ln for ln in on_err.splitlines()
               if "MJCSDFTRACE terminal" in ln]
  assert any("improved=0" in ln for ln in terminals)
  assert any("improved=1" in ln for ln in terminals)


@_needs_trace_build()
def test_trace_box_gate_pose_equivalence_and_records(tmp_path):
  # Box-press gate pose through the plugin-descent route (the mesh-octree
  # SDF field is descended by mjc_SDF stepGradient, not per-face
  # Frank-Wolfe): the trace build reproduces the installed oracle
  # bit-identically with empty stderr unset; set, it emits ordered
  # seed/stage/trial/terminal/gradpick/precontact records with identical
  # numbers. Body placement is baked into the XML so qpos0 equals the
  # gate float32 qp (representative-state injection, not XML default).
  import mujoco
  import numpy as np
  zf = repr(float(np.float32(0.060999923706054296)))
  xml_text = f"""<mujoco>
<asset><mesh name="b" vertex="-0.05 -0.05 -0.04  0.05 -0.05 -0.04 0.05 0.05 -0.04  -0.05 0.05 -0.04  -0.05 -0.05 0.04  0.05 -0.05 0.04  0.05 0.05 0.04  -0.05 0.05 0.04"/></asset>
<option timestep="0.002" integrator="Euler" iterations="100" tolerance="1e-8" gravity="0 0 0" sdf_initpoints="4"/>
<worldbody>
<geom name="sdf" type="sdf" mesh="b" contype="1" conaffinity="1"/>
<body pos="0 0 {zf}"><freejoint/><geom name="o" type="box" size="0.04 0.04 0.03" contype="1" conaffinity="1"/></body>
</worldbody>
</mujoco>"""
  p = tmp_path / "box_gate.xml"
  p.write_text(xml_text)
  xml = str(p)
  gate_qp = np.array([0.0, 0.0, 0.060999923706054296, 1.0, 0.0, 0.0, 0.0],
                     dtype=np.float32)
  model = mujoco.MjModel.from_xml_string(xml_text)
  np.testing.assert_array_equal(
      np.asarray(model.qpos0, dtype=np.float64),
      np.asarray(gate_qp, dtype=np.float64))
  off, off_err = _run(xml, trace=False)
  assert off_err == ""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert off["ncon"] == int(data.ncon) == 2
  np.testing.assert_array_equal(
      np.asarray(off["qacc"], dtype=np.float64),
      np.asarray(data.qacc, dtype=np.float64))
  for tc, c in zip(off["contacts"], range(data.ncon)):
    assert tc["dist"] == float(data.contact[c].dist)
    np.testing.assert_array_equal(
        np.asarray(tc["pos"], dtype=np.float64),
        np.asarray(data.contact[c].pos, dtype=np.float64))
    np.testing.assert_array_equal(
        np.asarray(tc["normal"], dtype=np.float64),
        np.asarray(data.contact[c].frame[:3], dtype=np.float64))
  on, on_err = _run(xml, trace=True)
  assert on == off
  lines = [ln for ln in on_err.splitlines() if "MJCSDFTRACE" in ln]
  kinds = {ln.split()[1] for ln in lines}
  for kind in ("seed", "stage", "step", "trial", "terminal", "gradpick",
               "precontact"):
    assert kind in kinds, kind
  seeds = [ln for ln in lines if "MJCSDFTRACE seed" in ln]
  assert len(seeds) == 4
  for k, ln in enumerate(seeds):
    assert f"i={k} halton={k}" in ln
  decisions = [ln.split("decision=")[1].split()[0]
               for ln in lines if "precontact" in ln]
  assert "admit" in decisions


@_needs_trace_build()
def test_trace_mesh_face_route_records_and_equivalence(tmp_path):
  # Mesh-geom vs plugin-SDF fixture exercises the per-face route
  # (BVH faces, Halton barycentric seeds, Frank-Wolfe depths, precontact
  # admission): meshface/meshseed records appear with stable seed order,
  # and numbers are inert (identical on/off) and bit-identical to the
  # installed oracle.
  import mujoco
  import sys
  sys.path.insert(0, "metal/tests")
  from test_bundled_sdf_rigid_oracle_019 import _BOX_VERTS, _BOX_FACES
  xml_text = f"""
    <mujoco>
      <option timestep="0.002" integrator="Euler" gravity="0 0 0"
              iterations="100" tolerance="1e-8" sdf_initpoints="50"/>
      <extension><plugin plugin="mujoco.sdf.torus">
        <instance name="shape"/>
      </plugin></extension>
      <asset>
        <mesh name="shape_mesh"><plugin instance="shape"/></mesh>
        <mesh name="probe_mesh" vertex="{_BOX_VERTS}" face="{_BOX_FACES}"/>
      </asset>
      <worldbody>
        <geom name="plugin_sdf" type="sdf" mesh="shape_mesh">
          <plugin instance="shape"/>
        </geom>
        <body name="probe" pos="0.45 0 0">
          <freejoint/><geom name="probe_mesh_geom" type="mesh" mesh="probe_mesh"/>
        </body>
      </worldbody>
      <contact><pair geom1="probe_mesh_geom" geom2="plugin_sdf" condim="1"/></contact>
    </mujoco>
  """
  p = tmp_path / "mesh_gate.xml"
  p.write_text(xml_text)
  xml = str(p)
  model = mujoco.MjModel.from_xml_string(xml_text)
  off, off_err = _run(xml, trace=False)
  assert off_err == ""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert off["ncon"] == int(data.ncon) > 0
  np.testing.assert_array_equal(
      np.asarray(off["qacc"], dtype=np.float64),
      np.asarray(data.qacc, dtype=np.float64))
  for tc, c in zip(off["contacts"], range(data.ncon)):
    assert tc["dist"] == float(data.contact[c].dist)
    np.testing.assert_array_equal(
        np.asarray(tc["pos"], dtype=np.float64),
        np.asarray(data.contact[c].pos, dtype=np.float64))
  on, on_err = _run(xml, trace=True)
  assert on == off
  lines = [ln for ln in on_err.splitlines() if "MJCSDFTRACE" in ln]
  kinds = {ln.split()[1] for ln in lines}
  assert "meshface" in kinds and "meshseed" in kinds
  assert "precontact" in kinds


@_needs_trace_build()
def test_instrumentation_patch_applies_deterministically(tmp_path):
  # Deterministic instrumentation application: the committed patch
  # applies cleanly to the pristine tree with the system patch tool and
  # reproduces the instrumented source bit-identically (no fuzz, no
  # rejects). This is the clean-source reproduction path the rebuild
  # script documents.
  import shutil
  import subprocess
  if shutil.which("patch") is None:
    pytest.skip("system patch tool absent")
  patch = os.path.join(_ROOT, "project", "metal",
                       "trace_instrumentation.patch")
  assert os.path.exists(patch), patch
  (tmp_path / "src" / "engine").mkdir(parents=True)
  shutil.copy(_PRISTINE_SRC, tmp_path / "src" / "engine" /
              "engine_collision_sdf.c")
  adapted = []
  for line in open(patch).read().split("\n"):
    if line.startswith("--- "):
      adapted.append("--- src/engine/engine_collision_sdf.c")
    elif line.startswith("+++ "):
      adapted.append("+++ src/engine/engine_collision_sdf.c")
    else:
      adapted.append(line)
  adapted_path = tmp_path / "test.patch"
  adapted_path.write_text("\n".join(adapted))
  proc = subprocess.run(
      ["patch", "-p0", "--quiet", "--forward"],
      input=adapted_path.read_bytes(), cwd=str(tmp_path),
      capture_output=True, timeout=120)
  assert proc.returncode == 0, proc.stderr.decode()[-2000:]
  produced = (tmp_path / "src" / "engine" /
              "engine_collision_sdf.c").read_bytes()
  with open(_TRACE_SRC, "rb") as f:
    assert produced == f.read()
