# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A three-craft space approach scene using externally computed thruster wrenches.

The colored spheres are visual target markers. No docking contacts or actuators
are modeled: a host-side PD controller computes forces and local torques, maps
them into generalized force, and sends the identical input to both backends.
"""

import argparse
import ctypes
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

_DAMPING = (0.08, 0.65, 2.4)
_CRAFT = ("craft_low", "craft_mid", "craft_high")
_COLORS = np.array([[.18, .8, 1, 1], [.55, 1, .42, 1], [1, .42, .58, 1]])
_DARK_PANEL_COLORS = ((28, 192, 232), (112, 224, 72), (245, 82, 132))
_STARTS = np.array([[-2.5, 0, 0], [0, 0, -.5], [2.5, 0, .5]], dtype=float)


class SpaceDocking:
  """Three free bodies controlled by one shared external wrench stream."""

  def __init__(self, mode="metal"):
    self.model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).with_name("space_docking.xml"))
    )
    self.mode = mode
    if (self.model.nq, self.model.nv, self.model.nu) != (21, 18, 0):
      raise RuntimeError("Expected three free bodies and no actuators")
    if mode not in ("metal", "cpu"):
      raise ValueError(f"Unknown mode {mode!r}")
    self.initial_qpos = self.model.qpos0.copy()
    self.initial_qvel = np.zeros(self.model.nv)
    for i in range(3):
      adr = int(self.model.jnt_qposadr[i])
      self.initial_qpos[adr:adr + 3] = _STARTS[i]
    self.native = None
    if mode == "metal":
      from mujoco_metal import MetalSimulation

      self.native = MetalSimulation(
          self.model,
          batch_size=1,
          profile="contact_free_forces_euler_v1",
          qpos=self.initial_qpos[None, :],
          qvel=self.initial_qvel[None, :],
      )
    self.actual = mujoco.MjData(self.model)
    self.reference = mujoco.MjData(self.model)
    self.time = 0.0
    self.max_qpos_error = self.max_qvel_error = 0.0
    self.reset()

  def reset(self, display_lock=None):
    if self.native is not None:
      self.native.state.reset(
          qpos=self.initial_qpos[None, :], qvel=self.initial_qvel[None, :]
      )

    def reset_host():
      for data in (self.actual, self.reference):
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.initial_qpos
        data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, data)
      self.time = 0.0

    if display_lock is None:
      reset_host()
    else:
      with display_lock():
        reset_host()
    self.max_qpos_error = self.max_qvel_error = 0.0

  def targets(self, at_time):
    """Smooth, distinct 3D target paths; markers have no collision geometry."""
    p = np.empty((3, 3), dtype=float)
    for i in range(3):
      p[i] = _STARTS[i] + [0.55 + .16 * np.sin(.7 * at_time + i),
                            .55 * np.sin(.55 * at_time + 1.8 * i),
                            .45 * np.sin(.8 * at_time + 2.1 * i)]
    return p

  def compute_force(self):
    """Host-side PD translation and gentle body-local tumble torques."""
    mujoco.mj_forward(self.model, self.actual)
    generalized = np.zeros(self.model.nv, dtype=np.float64)
    target = self.targets(self.time)
    for i, name in enumerate(_CRAFT):
      body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
      dof = int(self.model.jnt_dofadr[i])
      position_error = target[i] - self.actual.xpos[body]
      velocity = self.actual.qvel[dof:dof + 3]
      world_force = np.clip(5.0 * position_error - 3.0 * velocity, -8.0, 8.0)
      local_torque = np.array([
          .34 * np.sin(.8 * self.time + i),
          .26 * np.cos(.55 * self.time + i),
          .30 * np.sin(.6 * self.time + 1.3 * i),
      ])
      world_torque = np.empty(3)
      mujoco.mju_rotVecQuat(world_torque, local_torque, self.actual.xquat[body])
      mujoco.mj_applyFT(
          self.model, self.actual, world_force, world_torque,
          self.actual.xipos[body], body, generalized,
      )
    return generalized.astype(np.float32), target

  def step(self, display_lock=None):
    if display_lock is None:
      applied, targets = self.compute_force()
    else:
      with display_lock():
        applied, targets = self.compute_force()
    if self.native is not None:
      self.native.step(qfrc_applied=applied[None, :])
      snapshot = self.native.state.snapshot()
      if np.any(snapshot.status != 0):
        raise RuntimeError(f"Native Metal step failed: {snapshot.status.tolist()}")
      def copy_snapshot():
        self.actual.qpos[:] = snapshot.qpos[0]
        self.actual.qvel[:] = snapshot.qvel[0]
        self.actual.time = float(snapshot.time[0])

      if display_lock is None:
        copy_snapshot()
      else:
        with display_lock():
          copy_snapshot()
      self.time = self.actual.time
    else:
      def step_actual():
        self.actual.qfrc_applied[:] = applied
        mujoco.mj_step(self.model, self.actual)

      if display_lock is None:
        step_actual()
      else:
        with display_lock():
          step_actual()
      self.time = self.actual.time
    # The very same host-computed generalized force drives the reference.
    self.reference.qfrc_applied[:] = applied
    mujoco.mj_step(self.model, self.reference)
    self.max_qpos_error = max(
        self.max_qpos_error,
        float(np.max(np.abs(self.actual.qpos - self.reference.qpos))),
    )
    self.max_qvel_error = max(
        self.max_qvel_error,
        float(np.max(np.abs(self.actual.qvel - self.reference.qvel))),
    )
    for state in (self.actual, self.reference):
      if not np.all(np.isfinite(state.qpos)) or not np.all(np.isfinite(state.qvel)):
        raise RuntimeError("Nonfinite simulation state")
    return targets

  def prepare_render(self):
    # Forward kinematics is display preparation only; both states advance above.
    mujoco.mj_forward(self.model, self.actual)
    mujoco.mj_forward(self.model, self.reference)

  def _add_marker(self, scene, position, color, x_offset=0):
    if scene.ngeom >= scene.maxgeom:
      raise RuntimeError("Insufficient viewer geometry capacity")
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE),
        [.13, .13, .13], np.asarray(position) + [x_offset, 0, 0],
        np.eye(3).ravel(), color,
    )
    scene.ngeom += 1

  def add_scene_overlays(self, scene, targets):
    """Add target markers for both copies; marker spheres are visual only."""
    for i in range(3):
      self._add_marker(scene, targets[i], _COLORS[i])
      self._add_marker(scene, targets[i], _COLORS[i] * [.72, .72, .72, .6], 6)
    for geom_id in range(self.model.ngeom):
      if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Insufficient viewer geometry capacity")
      color = self.model.geom_rgba[geom_id].copy()
      color[3] = .68
      mujoco.mjv_initGeom(
          scene.geoms[scene.ngeom], int(self.model.geom_type[geom_id]),
          self.model.geom_size[geom_id], self.reference.geom_xpos[geom_id] + [6, 0, 0],
          self.reference.geom_xmat[geom_id], color,
      )
      scene.ngeom += 1

  def add_target_markers(self, scene, targets):
    for i in range(3):
      self._add_marker(scene, targets[i], _COLORS[i])

  def report(self):
    return {
        "mode": self.mode,
        "profile": "contact_free_forces_euler_v1" if self.native else "CPU MuJoCo mj_step",
        "simulated_seconds": self.time,
        "control": "host PD translation plus body-local torque, same qfrc_applied on both sides",
        "joint_linear_damping": list(_DAMPING),
        "max_qpos_error": self.max_qpos_error,
        "max_qvel_error": self.max_qvel_error,
        "target_markers_are_visual_only": True,
    }


def _active_display_count():
  lib = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
  fn = lib.CGGetActiveDisplayList
  ids, count = (ctypes.c_uint32 * 8)(), ctypes.c_uint32()
  fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
  fn.restype = ctypes.c_int32
  error = fn(8, ids, ctypes.byref(count))
  if error:
    raise RuntimeError(f"CoreGraphics display query failed with code {error}")
  return count.value


def camera():
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.lookat[:] = [0, 0, 0]
  cam.distance, cam.azimuth, cam.elevation = 8.6, 110, 18
  return cam


def render_panels(sim, targets, renderers=None):
  """Render three full-size craft in separate, consistently framed panels."""
  from PIL import Image, ImageDraw, ImageFont

  sim.prepare_render()
  cam = camera()
  panels = []
  font = ImageFont.load_default(size=17)
  if renderers is None:
    with ExitStack() as stack:
      renderers = [
          stack.enter_context(mujoco.Renderer(sim.model, height=540, width=600))
          for _ in range(2)
      ]
      panels = _render_panel_pair(sim, targets, cam, renderers)
  else:
    panels = _render_panel_pair(sim, targets, cam, renderers)
  for panel in panels:
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, 600, 42), fill=(8, 16, 27, 225))
    draw.text((16, 13), "LOW  .08", font=font, fill=_DARK_PANEL_COLORS[0])
    draw.text((220, 13), "MID  .65", font=font, fill=_DARK_PANEL_COLORS[1])
    draw.text((430, 13), "HIGH  2.4", font=font, fill=_DARK_PANEL_COLORS[2])

  # Sparse, deterministic stars add a space setting without covering the craft.
  rng = np.random.default_rng(17)
  for panel in panels:
    pixels = panel.load()
    for x, y in rng.integers([5, 50], [595, 535], size=(65, 2)):
      if max(pixels[int(x), int(y)]) < 34:
        pixels[int(x), int(y)] = (90, 115, 148)
  return panels


def _render_panel_pair(sim, targets, cam, renderers):
  from PIL import Image

  panels = []
  for state, renderer in zip((sim.actual, sim.reference), renderers):
    renderer.update_scene(state, camera=cam)
    sim.add_target_markers(renderer.scene, targets)
    panels.append(Image.fromarray(renderer.render()).convert("RGB"))
  return panels


def compose_frame(sim, targets, renderers=None):
  from PIL import Image, ImageDraw, ImageFont

  left, right = render_panels(sim, targets, renderers)
  canvas = Image.new("RGB", (1200, 600), (7, 12, 22))
  canvas.paste(left, (0, 60))
  canvas.paste(right, (600, 60))
  draw = ImageDraw.Draw(canvas, "RGBA")
  draw.rectangle((0, 0, 1200, 60), fill=(8, 16, 27, 255))
  font = ImageFont.load_default(size=20)
  left_name = "NATIVE METAL" if sim.mode == "metal" else "CPU MUJOCO"
  draw.text((18, 18), f"{left_name}  |  THREE-CRAFT SPACE APPROACH", font=font, fill=(235, 245, 255))
  draw.text((635, 18), "CPU REFERENCE  |  SAME WRENCH  |  1x", font=font, fill=(235, 245, 255))
  return canvas


def record_gif(sim, path, seconds=4):
  """Record 1 simulated second per playback second (bounded by the CLI)."""
  fps = 20
  frames = []
  with ExitStack() as stack:
    renderers = [
        stack.enter_context(mujoco.Renderer(sim.model, height=540, width=600))
        for _ in range(2)
    ]
    for _ in range(round(seconds * fps)):
      for _ in range(round(1 / (fps * sim.model.opt.timestep))):
        targets = sim.step()
      frames.append(compose_frame(sim, targets, renderers))
  frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--check", action="store_true", help="Check a short CPU comparison rollout (up to 200 steps)")
  parser.add_argument("--image", type=Path, help="Save a comparison PNG (requires Pillow and OpenGL)")
  parser.add_argument("--record", type=Path, help="Record a bounded comparison GIF (requires Pillow and OpenGL)")
  parser.add_argument("--record-seconds", type=float, default=4, help="GIF duration in simulated seconds (0.5 to 12; default 4)")
  parser.add_argument("--viewer-seconds", type=float, default=0, help="Stop viewer after this wall time; 0 keeps it open")
  args = parser.parse_args(argv)
  if args.steps < 1 or args.check and (not args.headless or args.steps > 200):
    parser.error("Use positive --steps; --check requires --headless and at most 200 steps")
  if not .5 <= args.record_seconds <= 12:
    parser.error("--record-seconds must be between 0.5 and 12")
  if not args.headless and sys.platform == "darwin" and _active_display_count() == 0:
    parser.error("No active macOS display; use --headless for a display-free run")
  sim = SpaceDocking(args.mode)
  left_label = "Native Metal" if args.mode == "metal" else "CPU MuJoCo"
  print(f"LEFT: {left_label} | RIGHT: CPU MuJoCo reference | target spheres are visual only", flush=True)
  if args.headless:
    for _ in range(args.steps):
      sim.step()
    if args.check and args.mode == "metal":
      assert sim.max_qpos_error < 1e-3, sim.report()
      assert sim.max_qvel_error < 1e-2, sim.report()
    if args.image:
      compose_frame(sim, sim.targets(sim.time)).save(args.image)
    if args.record:
      record_gif(sim, args.record, args.record_seconds)
      print(f"Saved {args.record} ({args.record_seconds:g}s at 20 fps, 1x playback)")
    print(json.dumps(sim.report(), indent=2))
    return
  from mujoco import viewer

  with viewer.launch_passive(sim.model, sim.actual) as view:
    cam = camera()
    view.cam.lookat[:] = cam.lookat
    view.cam.distance, view.cam.azimuth, view.cam.elevation = cam.distance, cam.azimuth, cam.elevation
    deadline = time.monotonic() + args.viewer_seconds
    while view.is_running():
      if args.viewer_seconds and time.monotonic() >= deadline:
        break
      start = time.monotonic()
      for _ in range(5):
        targets = sim.step(display_lock=view.lock)
      with view.lock():
        sim.prepare_render()
        view.user_scn.ngeom = 0
        sim.add_scene_overlays(view.user_scn, targets)
      view.sync()
      time.sleep(max(0, .01 - (time.monotonic() - start)))
  print(json.dumps(sim.report(), indent=2))


if __name__ == "__main__":
  main()
