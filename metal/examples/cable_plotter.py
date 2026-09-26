# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A tiny crossed-cable XY plotter with native scalar transmission forces.

Two fixed-joint tendons route orthogonal carriage coordinates as x+y and x-y.
Stateless affine position servos drive those lengths along a bounded looping
calligraphy path. The native example composes Metal smooth dynamics, the
transmission force stage, a dense solve, and semi-implicit Euler; MuJoCo's CPU
``mj_step`` advances an independent reference.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


class _PlotterRecorder:
  """Side-by-side recorder with world-space cable and measured pen trails."""

  def __init__(self, model, path):
    from PIL import Image, ImageDraw

    self.Image, self.ImageDraw = Image, ImageDraw
    self.path = Path(path)
    self.model = model
    self.renderer = mujoco.Renderer(model, 420, 620)
    self.camera = mujoco.MjvCamera()
    self.camera.lookat[:] = [0, 0, .68]
    self.camera.distance = 1.55
    self.camera.azimuth = 135
    self.camera.elevation = -32
    self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip")
    self.trails = [[], []]
    self.every = max(1, round(1 / (25 * model.opt.timestep)))
    self.duration = round(1000 * self.every * model.opt.timestep)
    self.frames = []

  def _connector(self, scene, start, end, color, width):
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
        np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(start, dtype=np.float64),
                         np.asarray(end, dtype=np.float64))
    scene.ngeom += 1

  def _draw(self, data, panel):
    mujoco.mj_forward(self.model, data)
    point = data.site_xpos[self.site_id].copy()
    trail = self.trails[panel]
    trail.append(point)
    if len(trail) > 250:
      del trail[:-250]
    self.renderer.update_scene(data, self.camera)
    scene = self.renderer.scene
    # Crossed cable runs terminate at the moving pen carriage; their anchors
    # and end points follow the model state in world coordinates.
    self._connector(scene, [-.46, -.34, .79], point, [1, .12, .16, 1], 3)
    self._connector(scene, [.46, -.34, .79], point, [.08, .88, .95, 1], 3)
    for previous, current in zip(trail, trail[1:]):
      self._connector(scene, previous, current, [1, .2, .58, .92], 4)
    return self.renderer.render().copy()

  def frame(self, step, actual, reference):
    if step % self.every:
      return
    panels = [self._draw(actual, 0), self._draw(reference, 1)]
    image = self.Image.fromarray(np.concatenate(panels, axis=1))
    draw = self.ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(20, 26, 36))
    draw.text((14, 8), "Native Metal | crossed cable paths", fill="white")
    draw.text((634, 8), "CPU MuJoCo | measured pen trails", fill="white")
    self.frames.append(image)

  def close(self):
    self.renderer.close()
    if self.frames:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      self.frames[0].save(self.path, save_all=True,
                          append_images=self.frames[1:],
                          duration=self.duration, loop=0)


def _path(t):
  """Smooth figure-eight calligraphy command in carriage coordinates."""
  return .22 * np.sin(.72 * t), .17 * np.sin(1.44 * t + .35)


def run(steps=200, mode="metal", check=False, record=None):
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if model.nu != 2 or model.ntendon != 2 or model.nq != 2 or model.nv != 2:
    raise RuntimeError("Expected a two-axis carriage driven by two fixed tendons")
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")

  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  actual = mujoco.MjData(model)
  reference = mujoco.MjData(model)
  actual.qpos[:] = reference.qpos[:] = qpos
  actual.qvel[:] = reference.qvel[:] = qvel
  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    native = MetalSimulation(
        model,
        qpos=qpos[None, :],
        qvel=qvel[None, :],
        profile="contact_free_transmission_euler_v1",
    )
  recorder = None
  if record:
    recorder = _PlotterRecorder(model, record)

  max_qpos_error = max_qvel_error = 0.0
  dt = float(model.opt.timestep)
  try:
    for index in range(steps):
      command_x, command_y = _path(index * dt)
      # Tendon lengths are x+y and x-y, matching the fixed-joint wrap maps.
      control = np.array([command_x + command_y, command_x - command_y])
      actual.ctrl[:] = reference.ctrl[:] = control
      mujoco.mj_step(model, reference)
      if native is None:
        mujoco.mj_step(model, actual)
      else:
        native.step(ctrl=control[None, :])
        snapshot = native.state.snapshot()
        if np.any(snapshot.status):
          raise RuntimeError(f"native simulation status: {snapshot.status.tolist()}")
        actual.qpos[:] = snapshot.qpos[0]
        actual.qvel[:] = snapshot.qvel[0]
        actual.time = snapshot.time[0]
      max_qpos_error = max(max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos))))
      max_qvel_error = max(max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel))))
      if recorder:
        recorder.frame(index, actual, reference)
  finally:
    if recorder:
      recorder.close()

  report = {
      "demo": "cable_plotter",
      "mode": mode,
      "steps": steps,
      "transmissions": ["x + y", "x - y"],
      "profile": "contact_free_transmission_euler_v1" if native else "MuJoCo CPU",
      "actuators": "stateless fixed-gain affine-bias position servos",
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "recording": str(record) if record else None,
  }
  if check and mode == "metal" and (max_qpos_error > 1e-3 or max_qvel_error > 1e-2):
    raise AssertionError(json.dumps(report, indent=2))
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true", help="Run without an interactive viewer")
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--check", action="store_true", help="Check a short native-versus-CPU rollout (at most 200 steps)")
  parser.add_argument("--record", type=Path, help="Record actual native/CPU renders as a GIF (requires OpenGL and Pillow)")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and (not args.headless or args.steps > 200):
    parser.error("Use positive --steps; --check requires --headless and at most 200 steps")
  if args.check and args.mode != "metal":
    parser.error("--check compares native Metal against the independent CPU reference; use --mode metal")
  print(json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2))


if __name__ == "__main__":
  main()
