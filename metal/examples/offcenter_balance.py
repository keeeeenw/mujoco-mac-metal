# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Zero-gravity balance workshop showing midpoint motion of offset COMs.

Three bright L-shaped rigid tools spin freely with inertial centers offset
from their free-joint origins. The overlaid chart traces each measured world
COM in the native and independent CPU panels. Contacts are disabled; the
colored shapes and paths are visual aids, not collision or force sensors.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROFILE = "contact_free_implicitfast_v1"


def _model():
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if (
      model.nv != 18
      or model.nbody != 4
      or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  ):
    raise RuntimeError(
        "Expected three standalone free bodies under implicitfast"
    )
  if not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
    raise RuntimeError(
        "This contact-free example requires geom contacts disabled"
    )
  return model


def _initial(model):
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  angular = ([0.2, 2.8, 0.4], [2.1, -0.3, 0.2], [-0.4, 2.3, 1.1])
  probe = mujoco.MjData(model)
  probe.qpos[:] = qpos
  mujoco.mj_forward(model, probe)
  for body, omega_body in enumerate(angular, start=1):
    dof = 6 * (body - 1)
    omega_body = np.asarray(omega_body)
    omega_world = probe.xmat[body].reshape(3, 3) @ omega_body
    com_offset_world = probe.xmat[body].reshape(3, 3) @ model.body_ipos[body]
    qvel[dof : dof + 3] = -np.cross(omega_world, com_offset_world)
    qvel[dof + 3 : dof + 6] = omega_body
  return qpos, qvel


def run(steps=1800, mode="metal", check=False, record=None):
  if mode not in ("metal", "cpu"):
    raise ValueError(f"Unknown mode {mode!r}")
  model = _model()
  qpos0, qvel0 = _initial(model)
  actual, reference = mujoco.MjData(model), mujoco.MjData(model)
  for data in (actual, reference):
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)
  native = None
  if mode == "metal":
    from mujoco_metal import MetalSimulation

    native = MetalSimulation(
        model, qpos=qpos0[None], qvel=qvel0[None], profile=PROFILE
    )
  recorder = None
  if record:
    from demo_recording import ComparisonRecorder

    recorder = ComparisonRecorder(
        model,
        record,
        "Off-center balance workshop | implicitfast midpoint",
        [0, 0, 0],
        4.5,
        azimuth=135,
        elevation=-18,
    )
  max_qpos_error = max_qvel_error = 0.0
  origin_path = []
  com_path = []
  reference_origin_path = []
  reference_com_path = []
  dt = float(model.opt.timestep)
  try:
    for step in range(steps):
      mujoco.mj_step(model, reference)
      if native is None:
        mujoco.mj_step(model, actual)
      else:
        native.step()
        state = native.state.snapshot()
        if np.any(state.status):
          raise RuntimeError(
              f"native implicitfast status: {state.status.tolist()}"
          )
        actual.qpos[:] = state.qpos[0]
        actual.qvel[:] = state.qvel[0]
        actual.time = float(state.time[0])
      mujoco.mj_forward(model, actual)
      mujoco.mj_forward(model, reference)
      max_qpos_error = max(
          max_qpos_error, float(np.max(np.abs(actual.qpos - reference.qpos)))
      )
      max_qvel_error = max(
          max_qvel_error, float(np.max(np.abs(actual.qvel - reference.qvel)))
      )
      origin_path.append(actual.xpos[1:4, :].copy())
      com_path.append(actual.xipos[1:4, :].copy())
      reference_origin_path.append(reference.xpos[1:4, :].copy())
      reference_com_path.append(reference.xipos[1:4, :].copy())
      if recorder:
        old_frame_count = len(recorder.frames)
        recorder.frame(
            step,
            actual,
            reference,
            extra="Colored trails: free-joint origins (thin) and inertial COMs (bright)",
        )
        if len(recorder.frames) > old_frame_count:
          _draw_trails(
              recorder.frames[-1],
              origin_path,
              com_path,
              reference_origin_path,
              reference_com_path,
          )
  finally:
    if recorder:
      recorder.close()
  origin_path, com_path = np.asarray(origin_path), np.asarray(com_path)
  result = {
      "demo": "offcenter_balance",
      "mode": mode,
      "profile": PROFILE if native is not None else "MuJoCo CPU",
      "steps": steps,
      "simulated_seconds": steps * dt,
      "features": [
          "eligible standalone free bodies with off-center inertial COMs",
          "source-derived implicitfast Newton rotational midpoint and analytic COM translation",
          "independent CPU MuJoCo reference; geom contacts disabled",
          "measured COM and free-joint-origin path overlays when recording",
      ],
      "max_qpos_error": max_qpos_error,
      "max_qvel_error": max_qvel_error,
      "com_path_length": np.linalg.norm(np.diff(com_path, axis=0), axis=-1)
      .sum(axis=0)
      .tolist(),
      "origin_path_length": np.linalg.norm(
          np.diff(origin_path, axis=0), axis=-1
      )
      .sum(axis=0)
      .tolist(),
  }
  if (
      check
      and mode == "metal"
      and (max_qpos_error > 2e-4 or max_qvel_error > 2e-4)
  ):
    raise AssertionError(json.dumps(result, indent=2))
  return result


def _draw_trails(
    image, origin_history, com_history, ref_origin_history, ref_com_history
):
  from PIL import ImageDraw

  draw = ImageDraw.Draw(image)
  colors = [(255, 176, 55), (60, 238, 255), (232, 124, 255)]
  # Give each body its own COM-centered view in a labeled world-coordinate
  # plane. Independent native/CPU histories remain in their own panels.
  histories = (
      (np.asarray(origin_history), np.asarray(com_history)),
      (np.asarray(ref_origin_history), np.asarray(ref_com_history)),
  )
  planes = ((0, 2, "XZ"), (1, 2, "YZ"), (0, 2, "XZ"))
  scale = 150  # pixels per meter; the faint rings have radius 0.1 m
  for panel, x0 in enumerate((18, 618)):
    draw.rounded_rectangle(
        (x0, 226, x0 + 260, 330),
        radius=8,
        fill=(8, 15, 25),
        outline=(80, 98, 120),
    )
    draw.text(
        (x0 + 8, 232),
        "Origin paths / COM dots | rings: 0.1 m",
        fill=(225, 235, 245),
    )
    origins, centers = histories[panel]
    for body, (color, plane) in enumerate(zip(colors, planes)):
      cx, cy = x0 + 46 + 80 * body, 293
      draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), outline=(45, 62, 78))
      draw.text((cx - 6, 250), plane[2], fill=color)
      anchor = centers[0, body]
      for samples, width in ((origins[:, body], 1), (centers[:, body], 3)):
        relative = samples - anchor
        xy = [
            (cx + round(p[plane[0]] * scale), cy - round(p[plane[1]] * scale))
            for p in relative
        ]
        if len(xy) > 1:
          draw.line(xy, fill=color, width=width)
        if width == 3:
          x, y = xy[-1]
          draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("metal", "cpu"), default="metal")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--steps", type=int, default=1800)
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--record", help="save native/CPU comparison GIF")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.check and not args.headless:
    parser.error("--check requires --headless and a positive step count")
  if args.check and args.mode != "metal":
    parser.error(
        "--check compares native Metal with the independent CPU reference"
    )
  print(
      json.dumps(run(args.steps, args.mode, args.check, args.record), indent=2)
  )


if __name__ == "__main__":
  main()
