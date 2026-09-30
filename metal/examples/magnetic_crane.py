# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Mocap-driven magnetic crane demo with weld attach/release.

A mocap hook carries a welded cargo box from above the left staging area to
above the right bin, then releases the weld through the public
`sim.set_equality_active` API; the cargo falls under gravity into the bin.
Hook motion is prescribed per-environment mocap input via the public
`sim.set_mocap` API on an identical deterministic schedule in CPU/native runs.
A paired always-attached run shows the physical effect of release.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"
RELEASE_STEP = 500
MOVE_START = 100
MOVE_END = 400
HOOK_START = np.array([0.0, 0.0, 1.0], dtype=np.float64)
HOOK_END = np.array([0.7, 0.0, 1.0], dtype=np.float64)


def _load_model():
  model_path = Path(__file__).with_suffix(".xml")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if model.neq != 1 or model.nv != 6 or model.nmocap != 1:
    raise RuntimeError(
        f"Expected magnetic crane with 1 equality, 6 DOFs, 1 mocap, "
        f"got neq={model.neq} nv={model.nv} nmocap={model.nmocap}"
    )
  return model


def _hook_target(step):
  if step < MOVE_START:
    return HOOK_START.copy()
  if step < MOVE_END:
    t = (step - MOVE_START) / float(MOVE_END - MOVE_START)
    return HOOK_START + (HOOK_END - HOOK_START) * t
  return HOOK_END.copy()


def run(steps=900, mode="metal", check=False, record=None, release_step=RELEASE_STEP):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    native.reset_to_keyframe(0)

  cpu_rel = mujoco.MjData(model)
  cpu_lat = mujoco.MjData(model)
  for d in (cpu_rel, cpu_lat):
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)

  recorder = None
  if record:
    from demo_recording import ComparisonRecorder
    recorder = ComparisonRecorder(model, record, "Magnetic Crane | Mocap Carry and Release",
                                  [0.2, 0, 0.5], 2.4, azimuth=135, elevation=-20)

  max_qpos_err = 0.0; max_qvel_err = 0.0
  pre_qpos_err = 0.0; pre_qvel_err = 0.0
  native_weld_before = 0.0; native_weld_after = 0.0
  native_hook_before = 0.0; native_hook_after = 0.0
  hook_track_err = 0.0
  anchor_before = 0.0
  anchor_parity_err = 0.0
  cargo_contacts = 0
  bin_hits = 0
  latched_bin_hits = 0
  hook_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hook")
  cargo_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cargo_geom")
  binR_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "binR_geom")

  for step in range(steps):
    target = _hook_target(step)
    target32 = target.astype(np.float32)
    if native is not None:
      native.set_mocap(target32.reshape(1, 3), np.array([[1, 0, 0, 0]], dtype=np.float32))
    cpu_rel.mocap_pos[0] = target
    cpu_lat.mocap_pos[0] = target
    if step == release_step:
      if native is not None:
        native.set_equality_active(np.array([0], dtype=np.int32))
      cpu_rel.eq_active[0] = 0
    if native is not None:
      native.step(1)
    mujoco.mj_step(model, cpu_rel)
    mujoco.mj_step(model, cpu_lat)
    attached = step < release_step
    if native is not None:
      gq = native.state.qpos[0].cpu().numpy()
      gv = native.state.qvel[0].cpu().numpy()
      step_qpos_err = float(np.max(np.abs(gq - cpu_rel.qpos)))
      step_qvel_err = float(np.max(np.abs(gv - cpu_rel.qvel)))
      max_qpos_err = max(max_qpos_err, step_qpos_err)
      max_qvel_err = max(max_qvel_err, step_qvel_err)
      if step < MOVE_START:
        pre_qpos_err = max(pre_qpos_err, step_qpos_err)
        pre_qvel_err = max(pre_qvel_err, step_qvel_err)
      # Hook tracking: native hook body pos vs prescribed target.
      hook_pos = np.asarray(native.state._mpos[0].cpu().numpy())[0]
      hook_track_err = max(hook_track_err, float(np.linalg.norm(hook_pos - target32)))
      if step in (50, 700):
        try:
          asm = native.assembled_system(recompute=True)
          jf = asm.get("joint_force", None)
          if jf is not None:
            w = float(np.linalg.norm(jf[0].cpu().numpy()[:6]))
            if step == 50:
              native_weld_before = w
            else:
              native_weld_after = w
        except Exception:
          pass
        # Weld anchor coincidence via CPU forward on native state.
        native_mj = mujoco.MjData(model)
        native_mj.qpos[:] = gq
        native_mj.mocap_pos[0] = target
        mujoco.mj_forward(model, native_mj)
        hook_x = np.asarray(native_mj.xpos[hook_id])
        cargo_x = np.asarray(native_mj.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cargo")])
        err = float(np.linalg.norm(hook_x - np.asarray(native_mj.mocap_pos)[0]))
        cpu_hook = np.asarray(cpu_rel.xpos[hook_id])
        anchor_parity_err = max(anchor_parity_err, abs(err - float(np.linalg.norm(cpu_hook - target))))
        if step == 50:
          anchor_before = err
          native_hook_before = float(np.linalg.norm(hook_x - target))
        if step == 700:
          native_hook_after = float(np.linalg.norm(hook_x - target))
    if cpu_rel.ncon > 0:
      cargo_contacts += 1
    for c in range(cpu_rel.ncon):
      g1, g2 = int(cpu_rel.contact[c].geom[0]), int(cpu_rel.contact[c].geom[1])
      if (g1 == cargo_geom and g2 == binR_geom) or (g2 == cargo_geom and g1 == binR_geom):
        bin_hits += 1
        break
    for c in range(cpu_lat.ncon):
      g1, g2 = int(cpu_lat.contact[c].geom[0]), int(cpu_lat.contact[c].geom[1])
      if (g1 == cargo_geom and g2 == binR_geom) or (g2 == cargo_geom and g1 == binR_geom):
        latched_bin_hits += 1
        break
    if recorder and native is not None:
      actual = mujoco.MjData(model)
      actual.qpos[:] = native.state.qpos[0].cpu().numpy()
      actual.qvel[:] = native.state.qvel[0].cpu().numpy()
      actual.mocap_pos[0] = target
      actual.time = float(native.state.time[0].cpu().numpy())
      extra = f"{'ATTACHED' if attached else 'RELEASED'} | t={actual.time:.2f}s | cargo x={float(actual.qpos[0]):+.2f}"
      recorder.frame(step, actual, cpu_rel, extra=extra)

  if recorder:
    recorder.close()

  if mode == "metal" and native is not None:
    released = native.state.qpos[0].cpu().numpy()
    released_x, released_z = float(released[0]), float(released[2])
    from mujoco_metal import MetalSimulation as MS
    lat_sim = MS(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    lat_sim.reset_to_keyframe(0)
    for step in range(steps):
      lat_sim.set_mocap(_hook_target(step).astype(np.float32).reshape(1, 3),
                        np.array([[1, 0, 0, 0]], dtype=np.float32))
      lat_sim.step(1)
    latched = lat_sim.state.qpos[0].cpu().numpy()
    latched_x, latched_z = float(latched[0]), float(latched[2])
  else:
    released_x, released_z = float(cpu_rel.qpos[0]), float(cpu_rel.qpos[2])
    latched_x, latched_z = float(cpu_lat.qpos[0]), float(cpu_lat.qpos[2])

  result = {
      "max_qpos_err": max_qpos_err, "max_qvel_err": max_qvel_err,
      "pre_qpos_err": pre_qpos_err, "pre_qvel_err": pre_qvel_err,
      "native_weld_before": native_weld_before, "native_weld_after": native_weld_after,
      "hook_track_err": hook_track_err,
      "native_hook_before": native_hook_before, "native_hook_after": native_hook_after,
      "anchor_before": anchor_before, "anchor_parity_err": anchor_parity_err,
      "cargo_contacts": cargo_contacts, "bin_hits": bin_hits,
      "latched_bin_hits": latched_bin_hits,
      "released_x": released_x, "released_z": released_z,
      "latched_x": latched_x, "latched_z": latched_z,
      "release_step": release_step, "steps": steps,
  }
  if check and mode == "metal":
    assert native_weld_before > 1.0, native_weld_before
    assert native_weld_after < 1e-4, native_weld_after
    assert hook_track_err < 1e-5, hook_track_err
    assert anchor_before < 5e-3, anchor_before
    assert anchor_parity_err < 5e-5, anchor_parity_err
    assert pre_qpos_err < 5e-5, pre_qpos_err
    assert pre_qvel_err < 5e-3, pre_qvel_err
    assert cargo_contacts > 20, cargo_contacts
    assert bin_hits > 20, bin_hits
    assert latched_bin_hits == 0, latched_bin_hits
    assert 0.52 <= released_x <= 0.88, released_x
    assert released_z < 0.35, released_z
    assert latched_z > 0.6, latched_z
    assert (latched_z - released_z) > 0.4
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=900)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true", help="check native/CPU rollout")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  parser.add_argument("--json", help="write numerical report to JSON")
  args = parser.parse_args()
  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")
  out = run(steps=args.steps, mode=args.mode, check=args.check, record=args.record)
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
