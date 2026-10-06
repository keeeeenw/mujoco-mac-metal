# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed in writing, software is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Render the actual hybrid rollout to a GIF (requires Pillow and OpenGL)."""

import argparse
import importlib
import json
from pathlib import Path

import mujoco

try:
  # Installed wheels expose the sibling as a real package module. This is
  # independent of the caller's working directory and the script directory.
  _pendulum = importlib.import_module("mujoco_metal.examples.pendulum")
except ModuleNotFoundError as error:
  if error.name != "mujoco_metal.examples":
    raise
  # Preserve the documented source-checkout invocation, where examples are
  # loose files rather than an importable package directory.
  _sibling = Path(__file__).with_name("pendulum.py")
  _spec = importlib.util.spec_from_file_location(
      "_mujoco_metal_pendulum_example", _sibling)
  if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load sibling demo {_sibling}") from error
  _pendulum = importlib.util.module_from_spec(_spec)
  _spec.loader.exec_module(_pendulum)

camera = _pendulum.camera
Comparison = _pendulum.Comparison


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  try:
    from PIL import Image
    from PIL import ImageDraw
  except ImportError as error:
    raise RuntimeError(
        "record_pendulum requires the optional Pillow package") from error
  sim = Comparison("metal-hybrid")
  frames = []
  with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
    for _ in range(75):
      sim.update_poses()
      renderer.update_scene(sim.actual, camera=camera())
      sim.add_reference(renderer.scene)
      frame = Image.new("RGB", (640, 540), (15, 18, 24))
      frame.paste(Image.fromarray(renderer.render()), (0, 30))
      draw = ImageDraw.Draw(frame)
      draw.text((20, 8), "Metal M/bias + CPU solve/integration", fill="white")
      draw.text((380, 8), "CPU MuJoCo reference", fill="white")
      draw.text(
          (20, 516),
          f"Simulation: {sim.actual.time:.2f}s | OpenGL display | Playback is not a speed benchmark",
          fill="white",
      )
      frames.append(frame)
      for _ in range(40):
        sim.step()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  frames[0].save(
      args.output,
      save_all=True,
      append_images=frames[1:],
      duration=40,
      loop=0,
      optimize=True,
  )
  print(json.dumps(sim.report(), indent=2))


if __name__ == "__main__":
  main()
