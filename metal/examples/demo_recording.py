# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Small side-by-side recorder for genuine native/CPU simulation frames."""

from pathlib import Path
import mujoco
import numpy as np


class ComparisonRecorder:

  def __init__(
      self,
      model,
      path,
      title,
      lookat,
      distance,
      azimuth=135,
      elevation=-25,
      fps=25,
  ):
    from PIL import Image, ImageDraw

    self.Image, self.ImageDraw = Image, ImageDraw
    self.path, self.title = path, title
    self.renderer = mujoco.Renderer(model, 360, 600)
    self.model = model
    self.camera = mujoco.MjvCamera()
    self.camera.lookat[:] = lookat
    self.camera.distance = distance
    self.camera.azimuth = azimuth
    self.camera.elevation = elevation
    self.every = max(1, round(1 / (fps * model.opt.timestep)))
    self.duration = round(1000 * self.every * model.opt.timestep)
    self.frames = []

  def frame(self, step, actual, reference, extra=''):
    if step % self.every:
      return
    panels = []
    for data in (actual, reference):
      mujoco.mj_forward(self.model, data)
      self.renderer.update_scene(data, self.camera)
      panels.append(self.renderer.render().copy())
    # Dedicated caption bands keep labels readable without covering geometry.
    image = self.Image.new("RGB", (1200, 412), (24, 34, 48))
    image.paste(self.Image.fromarray(np.concatenate(panels, axis=1)), (0, 26))
    draw = self.ImageDraw.Draw(image)
    draw.text((12, 8), self.title + ' | Native Metal', fill='white')
    draw.text((612, 8), 'CPU MuJoCo reference', fill='white')
    draw.text(
        (12, 394), extra or 'Actual simulation | 1x playback', fill='white'
    )
    self.frames.append(image)

  def close(self):
    self.renderer.close()
    if self.frames:
      Path(self.path).parent.mkdir(parents=True, exist_ok=True)
      self.frames[0].save(
          self.path,
          save_all=True,
          append_images=self.frames[1:],
          duration=self.duration,
          loop=0,
      )
