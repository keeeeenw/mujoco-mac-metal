# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A rotating inspection carousel with native perspective/orthographic depth images.

Depth tiles are actual stored RANGEFINDER samples, not renderer depth or a host
physics substitute. OpenGL renders the scene; a separate CPU run is the oracle.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np


class InspectionCarousel:
  def __init__(self, mode="metal"):
    self.mode = mode
    self.model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).with_suffix(".xml")))
    self.actual, self.reference = (mujoco.MjData(self.model) for _ in range(2))
    self.native = None
    if mode == "metal":
      from mujoco_metal import MetalSimulation
      self.native = MetalSimulation(self.model, batch_size=1,
          qvel=np.array([[.85]], np.float32), profile="integrated_euler_v1")
    elif mode != "cpu":
      raise ValueError(f"Unknown mode {mode!r}")
    for data in (self.actual, self.reference):
      data.qvel[:] = np.float32(.85)
      mujoco.mj_forward(self.model, data)
    self.steps = 0
    self.max_qpos_error = self.max_qvel_error = self.max_depth_error = 0.
    self.minimum_depth = np.inf
    self.maximum_depth = -np.inf
    self.max_image_change = 0.
    self.first_images = None
    self.images = np.zeros((2, 24, 32), np.float32)

  def step(self):
    ctrl = np.array([.015*np.sin(self.steps*self.model.opt.timestep)], np.float32)
    if self.native is not None:
      status = self.native.step(ctrl=ctrl[None])
      if status.cpu().tolist() != [0]:
        raise RuntimeError(f"Native physics failed: {status.cpu().tolist()}")
      # Stored step samples retain the exact step-stage sampling semantics.
      values = self.native.step_sensordata()[0]
      snapshot = self.native.state.snapshot()
      self.actual.qpos[:] = snapshot.qpos[0]
      self.actual.qvel[:] = snapshot.qvel[0]
      self.actual.time = float(snapshot.time[0])
    else:
      self.actual.ctrl[:] = ctrl
      mujoco.mj_step(self.model, self.actual)
      values = self.actual.sensordata.copy()
    self.reference.ctrl[:] = ctrl
    mujoco.mj_step(self.model, self.reference)
    self.images = np.asarray(values).reshape(2, 24, 32).copy()
    if not np.all(np.isfinite(self.images)) or np.min(self.images) <= 0:
      raise RuntimeError("Inspection cameras must contain finite positive hits")
    self.max_depth_error = max(self.max_depth_error,
        float(np.max(np.abs(values-self.reference.sensordata))))
    self.max_qpos_error = max(self.max_qpos_error,
        float(np.max(np.abs(self.actual.qpos-self.reference.qpos))))
    self.max_qvel_error = max(self.max_qvel_error,
        float(np.max(np.abs(self.actual.qvel-self.reference.qvel))))
    self.minimum_depth = min(self.minimum_depth, float(np.min(self.images)))
    self.maximum_depth = max(self.maximum_depth, float(np.max(self.images)))
    if self.first_images is None:
      self.first_images = self.images.copy()
    self.max_image_change = max(self.max_image_change,
        float(np.max(np.abs(self.images-self.first_images))))
    self.steps += 1
    return self.images

  def report(self):
    return dict(mode=self.mode, steps=self.steps, pixels_per_camera=768,
        simulated_seconds=float(self.actual.time), max_qpos_error=self.max_qpos_error,
        max_qvel_error=self.max_qvel_error, max_depth_error=self.max_depth_error,
        minimum_depth=self.minimum_depth, maximum_depth=self.maximum_depth,
        max_image_change=self.max_image_change)

  def check(self):
    if self.max_qpos_error > 2e-5 or self.max_qvel_error > 2e-5:
      raise AssertionError(self.report())
    if self.max_depth_error > 2e-4:
      raise AssertionError(self.report())
    if self.maximum_depth-self.minimum_depth < .3 or self.max_image_change < .01:
      raise AssertionError("Vacuous depth image or stationary inspection scene")


class InspectionRecorder:
  def __init__(self, model, path):
    from PIL import Image, ImageDraw
    self.Image, self.ImageDraw = Image, ImageDraw
    self.path = Path(path)
    self.renderer = mujoco.Renderer(model, height=440, width=960)
    self.camera = mujoco.MjvCamera()
    self.camera.lookat[:] = [0, 0, .85]
    self.camera.distance, self.camera.azimuth, self.camera.elevation = 4.7, 130, -26
    self.frames = []

  def frame(self, scene):
    # Host kinematics are exclusively for visualization of the read-back pose.
    mujoco.mj_forward(scene.model, scene.actual)
    self.renderer.update_scene(scene.actual, camera=self.camera)
    image = self.Image.new("RGB", (960, 730), (24, 34, 48))
    image.paste(self.Image.fromarray(self.renderer.render().copy()), (0, 30))
    draw = self.ImageDraw.Draw(image)
    draw.text((16, 10), f"OPTICAL INSPECTION | {scene.mode.upper()} PHYSICS", fill="white")
    for index, depth in enumerate(scene.images):
      # A fixed shared scale makes motion and lens differences comparable.
      t = np.clip((depth-1.5)/.75, 0, 1)
      colors = np.stack([255*(1-t), 190-80*t, 70+185*t], axis=-1).astype(np.uint8)
      tile = self.Image.fromarray(colors).resize((288, 216), self.Image.Resampling.NEAREST)
      x = 120+index*480
      image.paste(tile, (x, 490))
      label = "Perspective" if index == 0 else "Orthographic"
      draw.text((x, 475), label+" | stored optical depth [m]", fill="white")
    draw.text((16, 713), "32 x 24 rays per lens | near: gold, far: blue | OpenGL scene display", fill="white")
    self.frames.append(image)

  def close(self, duration=50):
    self.renderer.close()
    self.path.parent.mkdir(parents=True, exist_ok=True)
    if self.frames:
      if self.path.suffix.lower() == ".gif":
        self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:],
                            duration=duration, loop=0, optimize=True)
      else:
        self.frames[-1].save(self.path)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=160)
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", type=Path)
  parser.add_argument("--record-seconds", type=float, default=4.)
  parser.add_argument("--image", type=Path)
  parser.add_argument("--viewer-seconds", type=float, default=0.)
  args = parser.parse_args(argv)
  if args.steps < 1 or (args.check and not 20 <= args.steps <= 200):
    parser.error("--steps must be positive; --check requires 20..200 steps")
  if not .5 <= args.record_seconds <= 10 or args.viewer_seconds < 0:
    parser.error("record duration must be .5..10 seconds; viewer duration must be nonnegative")
  scene = InspectionCarousel(args.mode)
  for _ in range(args.steps):
    scene.step()
  if args.check:
    scene.check()
  if args.image or args.record:
    recorder = InspectionRecorder(scene.model, args.record or args.image)
    try:
      if args.record:
        stride = max(1, round(.05/scene.model.opt.timestep))
        for step in range(round(args.record_seconds/scene.model.opt.timestep)):
          scene.step()
          if step % stride == 0:
            recorder.frame(scene)
      else:
        recorder.frame(scene)
    finally:
      recorder.close(round(1000*stride*scene.model.opt.timestep) if args.record else 50)
  elif not args.headless:
    from mujoco import viewer
    deadline = time.monotonic()+args.viewer_seconds
    with viewer.launch_passive(scene.model, scene.actual) as view:
      view.cam.lookat[:] = [0, 0, .85]
      view.cam.distance, view.cam.azimuth, view.cam.elevation = 4.7, 130, -26
      while view.is_running() and (not args.viewer_seconds or time.monotonic() < deadline):
        start = time.monotonic()
        scene.step()
        mujoco.mj_forward(scene.model, scene.actual)
        view.sync()
        time.sleep(max(0, scene.model.opt.timestep-(time.monotonic()-start)))
  if args.check:
    scene.check()
  print(json.dumps(scene.report(), indent=2))


if __name__ == "__main__":
  main()
