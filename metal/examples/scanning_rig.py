# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""A pan/tilt scanning rig that makes native pose and velocity sensors visible.

The scanner reads a fixed reference site's relative pose and velocity. Its
traces are frame-sensor measurements; this example does not cast rays or
estimate surface depth.
"""

import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


class ScanningRig:
  """Drive a two-axis sensor rig with identical controls in both solvers."""

  def __init__(self, mode="metal"):
    self.mode = mode
    self.model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).with_name("scanning_rig.xml"))
    )
    self.native = None
    self.qpos0 = self.model.qpos0.copy()
    self.qvel0 = np.array([.48, .72], dtype=np.float32)
    if mode == "metal":
      from mujoco_metal import MetalSimulation

      self.native = MetalSimulation(
          self.model,
          batch_size=1,
          qpos=self.qpos0[None, :],
          qvel=self.qvel0[None, :],
          profile="contact_free_sensor_euler_v1",
      )
    elif mode != "cpu":
      raise ValueError(f"Unknown mode {mode!r}")
    self.actual = mujoco.MjData(self.model)
    self.reference = mujoco.MjData(self.model)
    self.position_adr = self._sensor_address("tip_position")
    self.axis_adr = self._sensor_address("tip_forward")
    self.gyro_adr = self._sensor_address("tip_gyro")
    self.velocity_adr = self._sensor_address("tip_relative_velocity")
    self.position_history = []
    self.axis_history = []
    self.max_qpos_error = self.max_qvel_error = self.max_sensor_error = 0.0
    self.reset()

  def _sensor_address(self, name):
    sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
      raise RuntimeError(f"Missing sensor {name}")
    return int(self.model.sensor_adr[sid])

  def reset(self):
    if self.native is not None:
      self.native.state.reset(qpos=self.qpos0[None, :], qvel=self.qvel0[None, :])
    for data in (self.actual, self.reference):
      mujoco.mj_resetData(self.model, data)
      data.qpos[:] = self.qpos0
      data.qvel[:] = self.qvel0
      mujoco.mj_forward(self.model, data)
    self.position_history.clear()
    self.axis_history.clear()
    self.max_qpos_error = self.max_qvel_error = self.max_sensor_error = 0.0

  def control(self, at_time):
    # Distinct, smooth drive signals give the two axes a looping scan pattern.
    return np.array([
        .55 * np.sin(.82 * at_time + .25),
        .62 * np.cos(.61 * at_time - .4),
    ], dtype=np.float32)

  def step(self):
    ctrl = self.control(self.actual.time)
    if self.native is not None:
      self.native.step(ctrl=ctrl[None, :])
      state = self.native.state.snapshot()
      if np.any(state.status):
        raise RuntimeError(f"Native Metal step failed: {state.status.tolist()}")
      self.actual.qpos[:] = state.qpos[0]
      self.actual.qvel[:] = state.qvel[0]
      self.actual.time = float(state.time[0])
    else:
      self.actual.ctrl[:] = ctrl
      mujoco.mj_step(self.model, self.actual)
    self.reference.ctrl[:] = ctrl
    mujoco.mj_step(self.model, self.reference)

    # Sensor values are deliberately queried at the current post-step state.
    # mj_step sensors are sampled inside its stage sequence, so mj_forward gives
    # the matching current-state query semantics used by MetalSimulation.
    mujoco.mj_forward(self.model, self.actual)
    mujoco.mj_forward(self.model, self.reference)
    if self.native is not None:
      values = self.native.sensor_values()[0].cpu().numpy().astype(np.float64)
    else:
      values = self.actual.sensordata.copy()
    sensor_error = float(np.max(np.abs(values - self.reference.sensordata)))
    self.max_sensor_error = max(self.max_sensor_error, sensor_error)
    self.max_qpos_error = max(
        self.max_qpos_error, float(np.max(np.abs(self.actual.qpos-self.reference.qpos)))
    )
    self.max_qvel_error = max(
        self.max_qvel_error, float(np.max(np.abs(self.actual.qvel-self.reference.qvel)))
    )
    self.position_history.append(values[self.position_adr:self.position_adr+3].copy())
    self.axis_history.append(values[self.axis_adr:self.axis_adr+3].copy())
    return values

  def report(self):
    return {
        "mode": self.mode,
        "simulated_seconds": float(self.actual.time),
        "max_qpos_error": self.max_qpos_error,
        "max_qvel_error": self.max_qvel_error,
        "max_current_sensor_error": self.max_sensor_error,
        "samples": len(self.position_history),
    }


class ScanRecorder:
  """Render native/CPU poses and two measured sensor traces into a GIF."""

  def __init__(self, model, destination):
    from PIL import Image, ImageDraw, ImageFont

    self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
    self.destination = Path(destination)
    self.renderers = [
        mujoco.Renderer(model, height=340, width=580) for _ in range(2)
    ]
    self.camera = mujoco.MjvCamera()
    self.camera.lookat[:] = [1.0, .05, .78]
    self.camera.distance = 4.7
    self.camera.azimuth, self.camera.elevation = 136, -22
    self.frames = []

  def _chart(self, draw, values, box, title, color, axis_labels):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=(13, 23, 37), outline=(43, 67, 91))
    draw.text((x0+14, y0+11), title, fill=(225, 238, 250), font=self.ImageFont.load_default(size=17))
    draw.line((x0+34, y0+56, x0+34, y1-28), fill=(70, 91, 113), width=1)
    draw.line((x0+34, y1-28, x1-18, y1-28), fill=(70, 91, 113), width=1)
    if len(values) < 2:
      return
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
      return
    if array.shape[1] == 3:
      a, b = array[:, 0], array[:, 1]
      scale = max(float(np.max(np.abs(np.concatenate((a, b))))), .05)
      lim = min(scale, 3.0)
      coords = [
          (x0+34+(x1-x0-60)*(float(x)/lim+1)/2,
           y0+56+(y1-y0-84)*(1-(float(y)/lim+1)/2)) for x, y in zip(a, b)
      ]
    else:
      coords = []
    if len(coords) > 1:
      draw.line(coords, fill=color, width=3, joint="curve")
      draw.ellipse((coords[-1][0]-5, coords[-1][1]-5, coords[-1][0]+5, coords[-1][1]+5), fill=color)
    draw.text((x0+13, y1-22), axis_labels, fill=(137, 162, 189), font=self.ImageFont.load_default())

  def frame(self, sim, values):
    for renderer, data in zip(self.renderers, (sim.actual, sim.reference)):
      renderer.update_scene(data, camera=self.camera)
    top = self.Image.new("RGB", (1160, 340), (7, 13, 23))
    top.paste(self.Image.fromarray(self.renderers[0].render().copy()), (0, 0))
    top.paste(self.Image.fromarray(self.renderers[1].render().copy()), (580, 0))
    canvas = self.Image.new("RGB", (1160, 720), (7, 13, 23))
    canvas.paste(top, (0, 0))
    draw = self.ImageDraw.Draw(canvas)
    font = self.ImageFont.load_default(size=18)
    draw.rectangle((0, 0, 1159, 44), fill=(8, 18, 31))
    draw.text((18, 14), "NATIVE SENSOR SCAN  |  METAL", fill=(211, 235, 255), font=font)
    draw.text((600, 14), "CPU MUJOCO REFERENCE", fill=(211, 235, 255), font=font)
    draw.rectangle((0, 340, 1159, 719), fill=(7, 13, 23))
    self._chart(draw, sim.position_history, (20, 360, 565, 700),
                "Measured scanner position relative to target site",
                (255, 174, 66), "x [m]  /  y [m]")
    self._chart(draw, sim.axis_history, (595, 360, 1140, 700),
                "Measured scanner forward axis in world frame",
                (50, 224, 208), "x-axis component  /  y-axis component")
    p = values[sim.position_adr:sim.position_adr+3]
    g = values[sim.gyro_adr:sim.gyro_adr+3]
    text = f"tip relative position {p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f} m    gyro {np.linalg.norm(g):.2f} rad/s    frame sensors only; no ray casting"
    draw.text((24, 326), text, fill=(255, 220, 154), font=self.ImageFont.load_default(size=15))
    self.frames.append(canvas)

  def close(self, seconds, save=True):
    for renderer in self.renderers:
      renderer.close()
    if self.frames and save:
      self.destination.parent.mkdir(parents=True, exist_ok=True)
      self.frames[0].save(
          self.destination, save_all=True, append_images=self.frames[1:],
          duration=max(20, round(1000*seconds/len(self.frames))), loop=0,
          optimize=True,
      )


def _record(sim, path, seconds):
  sim.reset()
  recorder = ScanRecorder(sim.model, path)
  fps = 20
  strides = max(1, round(1/(fps*sim.model.opt.timestep)))
  steps = round(seconds*fps)*strides
  for index in range(steps):
    values = sim.step()
    if index % strides == 0:
      recorder.frame(sim, values)
  recorder.close(seconds)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--check", action="store_true", help="verify a short current-state CPU sensor comparison")
  parser.add_argument("--image", type=Path, help="save a sensor visualization PNG")
  parser.add_argument("--record", type=Path, help="save a bounded comparison GIF")
  parser.add_argument("--record-seconds", type=float, default=4)
  parser.add_argument("--viewer-seconds", type=float, default=0)
  args = parser.parse_args(argv)
  if args.steps < 1 or (args.check and (not args.headless or args.steps > 200)):
    parser.error("--check requires --headless and 1..200 steps")
  if not .5 <= args.record_seconds <= 10:
    parser.error("--record-seconds must be between 0.5 and 10")
  if not args.headless and sys.platform == "darwin" and not args.record and not args.image:
    lib = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    fn = lib.CGGetActiveDisplayList
    ids, count = (ctypes.c_uint32 * 8)(), ctypes.c_uint32()
    fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    if fn(8, ids, ctypes.byref(count)) or count.value == 0:
      parser.error("No active macOS display; use --headless")
  sim = ScanningRig(args.mode)
  if args.headless:
    for _ in range(args.steps):
      sim.step()
    if args.check and args.mode == "metal":
      assert sim.max_qpos_error < 1e-3, sim.report()
      assert sim.max_qvel_error < 1e-2, sim.report()
      assert sim.max_sensor_error < 1e-3, sim.report()
    if args.image:
      recorder = ScanRecorder(sim.model, args.image)
      values = sim.step()
      recorder.frame(sim, values)
      recorder.close(.5, save=False)
      recorder.frames[-1].save(args.image)
    if args.record:
      _record(sim, args.record, args.record_seconds)
    print(json.dumps(sim.report(), indent=2))
    return
  from mujoco import viewer

  deadline = time.monotonic() + args.viewer_seconds
  with viewer.launch_passive(sim.model, sim.actual) as view:
    view.cam.lookat[:] = [1.0, .05, .78]
    view.cam.distance, view.cam.azimuth, view.cam.elevation = 4.7, 136, -22
    while view.is_running() and (not args.viewer_seconds or time.monotonic() < deadline):
      start = time.monotonic()
      for _ in range(5):
        sim.step()
      view.sync()
      time.sleep(max(0, .01-(time.monotonic()-start)))
  print(json.dumps(sim.report(), indent=2))


if __name__ == "__main__":
  main()
