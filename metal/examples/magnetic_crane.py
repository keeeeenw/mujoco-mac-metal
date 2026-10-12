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
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import time

import mujoco
import numpy as np

PROFILE = "integrated_euler_v1"
RELEASE_STEP = 500
MOVE_START = 100
MOVE_END = 400
HOOK_START = np.array([0.0, 0.0, 1.0], dtype=np.float64)
HOOK_END = np.array([0.7, 0.0, 1.0], dtype=np.float64)


def _same_checkpoint(actual, expected):
  """Compare the complete public replay checkpoint, including retained seeds."""
  if isinstance(expected, np.ndarray):
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert actual.tobytes() == expected.tobytes(), "checkpoint replay must preserve literal bits"
  elif is_dataclass(expected):
    assert type(actual) is type(expected)
    for field in fields(expected):
      _same_checkpoint(getattr(actual, field.name), getattr(expected, field.name))
  elif isinstance(expected, dict):
    assert actual.keys() == expected.keys()
    for key in expected:
      _same_checkpoint(actual[key], expected[key])
  elif isinstance(expected, (tuple, list)):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
      _same_checkpoint(left, right)
  else:
    assert actual == expected


def _slot_pairs(sim):
  descriptor = sim._coupled_constraints.descriptor
  return [(int(descriptor.geom1[i]), int(descriptor.geom2[i]))
          for i in range(descriptor.npairs)
          for _ in range(int(descriptor.pair_contact_offset[i+1]
                             - descriptor.pair_contact_offset[i]))]


def _active_pairs(sim, slots):
  contacts = sim._last_coupled
  if contacts is None or "contact_mask" not in contacts:
    raise RuntimeError("native contact observation metadata is unavailable")
  mask = contacts["contact_mask"][0].detach().cpu().numpy()
  return [slots[index] for index in np.flatnonzero(mask > .5)]


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
  if check and steps < 900:
    raise ValueError("complete crane qualification requires at least 900 steps")
  model = _load_model()
  qpos0 = model.qpos0.copy().astype(np.float32)
  qvel0 = np.zeros(model.nv, dtype=np.float32)

  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation
    native = MetalSimulation(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    native.reset_to_keyframe(0)
  native_slots = _slot_pairs(native) if native is not None else []

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
  native_bin_hits = 0
  native_latched_bin_hits = 0
  replay_start = min(MOVE_END, steps // 2)
  replay_checkpoint = None
  hook_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hook")
  cargo_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cargo_geom")
  binR_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "binR_geom")

  for step in range(steps):
    if check and native is not None and step == replay_start:
      replay_checkpoint = native.snapshot()
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
      np.testing.assert_array_equal(native.state._status.detach().cpu().numpy(), [0])
      native_bin_hits += int(any(
          set(pair) == {cargo_geom, binR_geom}
          for pair in _active_pairs(native, native_slots)))
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
        asm = native.assembled_system(recompute=True)
        jf = asm["joint_force"]
        w = float(np.linalg.norm(jf[0].cpu().numpy()[:6]))
        if step == 50:
          native_weld_before = w
        else:
          native_weld_after = w
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
    finished_checkpoint = native.snapshot() if check else None
    released = native.state.qpos[0].cpu().numpy()
    released_x, released_z = float(released[0]), float(released[2])
    from mujoco_metal import MetalSimulation as MS
    lat_sim = MS(model, batch_size=1, qpos=qpos0[None, :], qvel=qvel0[None, :], profile=PROFILE)
    lat_sim.reset_to_keyframe(0)
    latched_slots = _slot_pairs(lat_sim)
    for step in range(steps):
      lat_sim.set_mocap(_hook_target(step).astype(np.float32).reshape(1, 3),
                        np.array([[1, 0, 0, 0]], dtype=np.float32))
      lat_sim.step(1)
      np.testing.assert_array_equal(lat_sim.state._status.detach().cpu().numpy(), [0])
      native_latched_bin_hits += int(any(
          set(pair) == {cargo_geom, binR_geom}
          for pair in _active_pairs(lat_sim, latched_slots)))
    latched = lat_sim.state.qpos[0].cpu().numpy()
    latched_x, latched_z = float(latched[0]), float(latched[2])
    if check:
      native.restore(replay_checkpoint)
      for step in range(replay_start, steps):
        native.set_mocap(_hook_target(step).astype(np.float32).reshape(1, 3),
                          np.array([[1, 0, 0, 0]], np.float32))
        if step == release_step:
          native.set_equality_active(np.array([0], np.int32))
        native.step(1)
        # Match observational operations in the original execution as well.
        if step in (50, 700):
          native.assembled_system(recompute=True)
      _same_checkpoint(native.snapshot(), finished_checkpoint)
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
      "native_bin_hits": native_bin_hits,
      "native_latched_bin_hits": native_latched_bin_hits,
      "checkpoint_replay_exact": bool(check and native is not None),
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
    assert native_bin_hits > 20, native_bin_hits
    assert native_latched_bin_hits == 0, native_latched_bin_hits
    assert 0.52 <= released_x <= 0.88, released_x
    assert released_z < 0.35, released_z
    assert latched_z > 0.6, latched_z
    assert (latched_z - released_z) > 0.4
  return result


def play(viewer_seconds=None):
  """Display actual native physics; MuJoCo only mirrors it for OpenGL rendering."""
  import glfw
  from mujoco_metal import MetalSimulation
  model = _load_model()
  sim = MetalSimulation(model, batch_size=1, profile=PROFILE)
  sim.reset_to_keyframe(0)
  display = mujoco.MjData(model)
  completed = 0
  if not glfw.init():
    raise RuntimeError("GLFW could not initialize the Mac viewer")
  window = None
  context = None
  try:
    window = glfw.create_window(1000, 700, "Magnetic crane — native Metal", None, None)
    if window is None:
      raise RuntimeError("GLFW could not create an OpenGL viewer window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.2, 0, .5]
    camera.distance, camera.azimuth, camera.elevation = 2.4, 135, -20
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    options = mujoco.MjvOption()
    previous = list(glfw.get_cursor_pos(window))
    def keyboard(win, key, scancode, action, mods):
      if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(win, True)
    def cursor(win, x, y):
      dx, dy = x-previous[0], y-previous[1]
      previous[:] = [x, y]
      left = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
      right = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
      middle = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
      if not (left or right or middle):
        return
      action = (mujoco.mjtMouse.mjMOUSE_MOVE_V if right else
                mujoco.mjtMouse.mjMOUSE_ROTATE_V if left else mujoco.mjtMouse.mjMOUSE_ZOOM)
      height = max(1, glfw.get_window_size(win)[1])
      mujoco.mjv_moveCamera(model, action, dx/height, dy/height, scene, camera)
    def scroll(win, x, y):
      mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05*y, scene, camera)
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, cursor)
    glfw.set_scroll_callback(window, scroll)
    started = time.monotonic()
    for step in range(900):
      if glfw.window_should_close(window) or (viewer_seconds is not None and time.monotonic()-started >= viewer_seconds):
        break
      before = time.monotonic()
      target = _hook_target(step).astype(np.float32)
      sim.set_mocap(target.reshape(1, 3), np.asarray([[1, 0, 0, 0]], np.float32))
      if step == RELEASE_STEP:
        sim.set_equality_active(np.asarray([0], np.int32))
      sim.step(1)
      np.testing.assert_array_equal(sim.state._status.detach().cpu().numpy(), [0])
      display.qpos[:] = sim.state.qpos[0].cpu().numpy()
      display.qvel[:] = sim.state.qvel[0].cpu().numpy()
      display.mocap_pos[0] = target
      display.eq_active[:] = step < RELEASE_STEP
      display.time = float(sim.state.time[0].cpu())
      mujoco.mj_forward(model, display)
      mujoco.mjv_updateScene(model, display, options, None, camera,
                            mujoco.mjtCatBit.mjCAT_ALL, scene)
      width, height = glfw.get_framebuffer_size(window)
      mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
      glfw.swap_buffers(window)
      glfw.poll_events()
      completed += 1
      time.sleep(max(0, float(model.opt.timestep) - (time.monotonic()-before)))
  finally:
    if context is not None:
      context.free()
    if window is not None:
      glfw.destroy_window(window)
    glfw.terminate()
  return {"viewer_steps": completed, "backend": "native Metal physics; upstream OpenGL visualization"}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=900)
  parser.add_argument("--mode", choices=["metal", "cpu"], default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--check", action="store_true", help="check native/CPU rollout")
  parser.add_argument("--record", help="record native/CPU renders to a GIF")
  parser.add_argument("--json", help="write numerical report to JSON")
  parser.add_argument("--viewer", action="store_true", help="show actual native physics in a GLFW window on the main thread")
  parser.add_argument("--viewer-seconds", type=float, help="close the viewer after this many wall seconds")
  args = parser.parse_args()
  if args.steps <= 0 or (args.check and not args.headless):
    raise SystemExit("use --headless --check for verification")
  if args.check and args.steps < 900:
    raise SystemExit("complete crane qualification requires at least 900 steps")
  if args.viewer_seconds is not None and (not np.isfinite(args.viewer_seconds) or args.viewer_seconds <= 0):
    raise SystemExit("--viewer-seconds must be positive and finite")
  if args.viewer or args.viewer_seconds is not None:
    if args.headless or args.check or args.record or args.mode != "metal":
      raise SystemExit("viewer requires Metal mode and cannot be combined with headless/check/record")
    print(json.dumps(play(args.viewer_seconds), indent=2))
    return
  out = run(steps=args.steps, mode=args.mode, check=args.check, record=args.record)
  print(json.dumps(out, indent=2))
  if args.json:
    Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
