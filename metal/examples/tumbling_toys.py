# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Three asymmetric toys spinning freely with native RK4 and a CPU oracle."""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np


def run(steps=1000, record=None, mode='metal'):
  m = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix('.xml')))
  q = m.qpos0.copy().astype('float32')
  v = np.zeros(m.nv, dtype='float32')
  v[3:6] = [4, 0.08, 0.1]
  v[9:12] = [0.1, 4, 0.08]
  v[15:18] = [0.08, 0.1, 4]
  refs = mujoco.MjData(m)
  refs.qpos[:] = q
  refs.qvel[:] = v
  display = mujoco.MjData(m)
  display.qpos[:] = q
  display.qvel[:] = v
  sim = None
  if mode == 'metal':
    from mujoco_metal import MetalSimulation

    sim = MetalSimulation(
        m, qpos=q[None], qvel=v[None], profile='contact_free_rk4_v1'
    )
  frames = []
  errors = np.zeros(2)
  renderer = None
  if record:
    from PIL import Image, ImageDraw

    renderer = mujoco.Renderer(m, 300, 600)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1]
    camera.distance = 3.7
    camera.azimuth = 90
    camera.elevation = -18
  try:
    for step in range(steps):
      mujoco.mj_step(m, refs)
      if sim:
        sim.step()
        snap = sim.state.snapshot()
        if np.any(snap.status):
          raise RuntimeError(snap.status)
        display.qpos[:] = snap.qpos[0]
        display.qvel[:] = snap.qvel[0]
      else:
        mujoco.mj_step(m, display)
      errors = np.maximum(
          errors,
          [
              np.max(abs(display.qpos - refs.qpos)),
              np.max(abs(display.qvel - refs.qvel)),
          ],
      )
      if renderer and step % 20 == 0:
        panels = []
        for data in (display, refs):
          mujoco.mj_forward(m, data)
          renderer.update_scene(data, camera)
          panels.append(renderer.render().copy())
        image = Image.fromarray(np.concatenate(panels, axis=1))
        draw = ImageDraw.Draw(image)
        draw.text(
            (12, 10), 'Native Metal RK4' if sim else 'CPU MuJoCo', fill='white'
        )
        draw.text((612, 10), 'CPU MuJoCo RK4 reference', fill='white')
        draw.text(
            (12, 280),
            'Free asymmetric toys: three spin axes | 1x playback',
            fill='white',
        )
        frames.append(image)
  finally:
    if renderer:
      renderer.close()
  if frames:
    Path(record).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        record, save_all=True, append_images=frames[1:], duration=40, loop=0
    )
  if errors[0] > 1e-3 or errors[1] > 3e-3:
    raise AssertionError(f'trajectory errors {errors}')
  return dict(
      mode=mode,
      steps=steps,
      max_qpos_error=float(errors[0]),
      max_qvel_error=float(errors[1]),
      integrator='RK4',
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--steps', type=int, default=1000)
  parser.add_argument('--record')
  parser.add_argument('--mode', choices=['metal', 'cpu'], default='metal')
  args = parser.parse_args()
  print(json.dumps(run(args.steps, args.record, args.mode), indent=2))
