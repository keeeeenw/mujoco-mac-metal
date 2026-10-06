# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Two-link haptic calligraphy arm driven by a native Cartesian FORCE plugin.

The pen site follows an analytic planar Lissajous target. World-space PD
feedback is projected through the articulated site's translational Jacobian;
the XML's frame sensors independently expose measured world position/velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from mujoco_metal.site_feedback import SiteFeedbackPlugin, register_site_feedback
from mujoco_metal.extensions import PluginType, default_registry


PROFILE = "integrated_euler_v1"
PLUGIN_NAME = "haptic_calligraphy_feedback"
SITE_NAME = "pen_tip"
KP, KD = 1.8, 0.65
CENTER = np.array([1.0, 0.0, 0.90], dtype=np.float32)
AMPLITUDE = np.array([0.06, 0.0, 0.055], dtype=np.float32)
FREQUENCY = np.array([0.9, 0.0, 1.3], dtype=np.float32)
PHASE = np.array([0.0, 0.0, 1.1], dtype=np.float32)


def _load_model():
  model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_suffix(".xml")))
  if (model.nq, model.nv, model.nu, model.nsite) != (2, 2, 0, 1):
    raise RuntimeError("Expected a free-standing, two-hinge drawing arm")
  if model.nsensor != 2:
    raise RuntimeError("Expected world frame-position and frame-linear-velocity sensors")
  return model


def _initial_state(model):
  qpos = model.qpos0.copy().astype(np.float32)
  qvel = np.zeros(model.nv, dtype=np.float32)
  qpos[:] = (0.2, -0.4)
  qvel[:] = (0.0, 0.0)
  return qpos, qvel


def _target_numpy(at_time):
  phase = FREQUENCY.astype(np.float64) * float(at_time) + PHASE
  target = CENTER + AMPLITUDE * np.sin(phase)
  velocity = AMPLITUDE * FREQUENCY * np.cos(phase)
  return target.astype(np.float64), velocity.astype(np.float64)


def _cpu_feedback(model, data, site_id):
  """Independent host oracle: measured site velocity plus pinned mj_applyFT."""
  point = data.site_xpos[site_id].copy()
  site_velocity = np.empty(6, dtype=np.float64)
  mujoco.mj_objectVelocity(
      model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, site_velocity, 0)
  target, target_velocity = _target_numpy(data.time)
  force = KP * (target - point) + KD * (target_velocity - site_velocity[3:])
  qfrc = np.zeros(model.nv, dtype=np.float64)
  body = int(model.site_bodyid[site_id])
  mujoco.mj_applyFT(model, data, force, np.zeros(3), point, body, qfrc)
  return {
      "site_position": point,
      "site_velocity": site_velocity[3:].copy(),
      "target_position": target,
      "target_velocity": target_velocity,
      "force": force,
      "qfrc": qfrc,
  }


def _register(model):
  register_site_feedback(
      PLUGIN_NAME, site=SITE_NAME, kp=KP, kd=KD, center=CENTER,
      amplitude=AMPLITUDE, frequency=FREQUENCY, phase=PHASE)
  try:
    from mujoco_metal import MetalSimulation
    qpos, qvel = _initial_state(model)
    simulation = MetalSimulation(
        model, batch_size=1, qpos=qpos[None, :], qvel=qvel[None, :],
        profile=PROFILE)
  except Exception:
    default_registry.unregister(PLUGIN_NAME, PluginType.FORCE)
    raise
  plugin = next(p for p in simulation._native_plugins
                if p.name == PLUGIN_NAME)
  return simulation, plugin


class _Recorder:
  def __init__(self, model, destination):
    try:
      from PIL import Image
    except ImportError as error:
      raise RuntimeError("Haptic GIF recording requires the optional Pillow package") from error
    self.Image = Image
    self.path = Path(destination)
    self.model = model
    self.renderer = mujoco.Renderer(model, height=620, width=850)
    self.camera = mujoco.MjvCamera()
    self.camera.lookat[:] = [0.54, 0.0, 0.91]
    self.camera.distance = 2.35
    self.camera.azimuth, self.camera.elevation = 92, -9
    self.site_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME))
    self.trail = []
    self.every = max(1, round(1.0 / (30.0 * model.opt.timestep)))
    self.frame_rate = 1.0 / (self.every * model.opt.timestep)
    self.frames = []
    self._closed = False

  @staticmethod
  def _connector(scene, start, end, color, width):
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
        np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
        np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64))
    scene.ngeom += 1

  def render(self, data, *, mode, force, feedback=None, sensor_values=None):
    mujoco.mj_forward(self.model, data)
    point = (data.site_xpos[self.site_id].copy() if feedback is None else
             np.asarray(feedback["site_position"]).copy())
    target = (_target_numpy(data.time)[0] if feedback is None else
              np.asarray(feedback["target_position"]).copy())
    self.trail.append(point)
    if len(self.trail) > 240:
      del self.trail[:-240]
    self.renderer.update_scene(data, self.camera)
    scene = self.renderer.scene
    if scene.ngeom < scene.maxgeom:
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
          geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.035, 0, 0]),
          target, np.eye(3).reshape(-1), np.array([1, .26, .45, .92]))
      scene.ngeom += 1
    self._connector(scene, point, point + .16 * force,
                    [1.0, .76, .18, .98], .009)
    for a, b in zip(self.trail, self.trail[1:]):
      self._connector(scene, a, b, [.95, .23, .57, .8], .0035)
    pos_adr, vel_adr = (int(self.model.sensor_adr[0]),
                        int(self.model.sensor_adr[1]))
    sensor_values = (data.sensordata if sensor_values is None else
                     np.asarray(sensor_values))
    position = sensor_values[pos_adr:pos_adr+3]
    velocity = sensor_values[vel_adr:vel_adr+3]
    _add_scene_label(scene, point, position, velocity, force)
    self.frames.append(self.Image.fromarray(self.renderer.render().copy()))

  def close(self):
    if self._closed:
      return
    self._closed = True
    self.renderer.close()
    try:
      if self.frames:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(
            self.path, save_all=True, append_images=self.frames[1:],
            duration=round(1000 / self.frame_rate), loop=0)
    finally:
      for frame in self.frames:
        frame.close()
      self.frames.clear()


def run(steps=200, mode="metal", check=False, render=None):
  if mode not in ("cpu", "metal"):
    raise ValueError("mode must be 'cpu' or 'metal'")
  if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
    raise ValueError("steps must be a positive integer")
  if check and (mode != "metal" or steps > 200):
    raise ValueError("--check requires metal mode and at most 200 steps")
  model = _load_model()
  site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME))
  qpos0, qvel0 = _initial_state(model)
  actual = mujoco.MjData(model)
  actual.qpos[:] = qpos0
  actual.qvel[:] = qvel0
  mujoco.mj_forward(model, actual)
  reference = None
  if check:
    reference = mujoco.MjData(model)
    reference.qpos[:] = qpos0
    reference.qvel[:] = qvel0
    mujoco.mj_forward(model, reference)
  native = plugin = None
  if mode == "metal":
    native, plugin = _register(model)

  recorder = None
  max_qpos = max_qvel = max_qacc = max_sensor = 0.0
  max_force = max_tracking = 0.0
  trajectory = []
  position_adr = int(model.sensor_adr[0])
  velocity_adr = int(model.sensor_adr[1])
  try:
    recorder = _Recorder(model, render) if render else None
    for step in range(steps):
      if native is not None:
        if reference is not None:
          # The independent CPU oracle runs only for --check. Its current VEL
          # site state and wrench are recomputed by pinned MuJoCo.
          reference.qfrc_applied[:] = 0
          mujoco.mj_forward(model, reference)
          ref_force = _cpu_feedback(model, reference, site_id)
          reference.qfrc_applied[:] = ref_force["qfrc"]
          mujoco.mj_step(model, reference)
          reference_qacc = reference.qacc.copy()
          mujoco.mj_forward(model, reference)
          reference_sensor = reference.sensordata.copy()
        status = native.step()
        if np.any(status.detach().cpu().numpy()):
          raise RuntimeError(f"native simulation status: {status.tolist()}")
        snapshot = native.state.snapshot()
        native_qpos = snapshot.qpos[0].copy()
        native_qvel = snapshot.qvel[0].copy()
        native_qacc = snapshot.qacc[0].copy()
        native_time = float(snapshot.time[0])
        native_sensor = native.sensor_values()[0].detach().cpu().numpy().copy()
        if reference is not None:
          max_qpos = max(max_qpos, float(np.max(np.abs(
              native_qpos-reference.qpos))))
          max_qvel = max(max_qvel, float(np.max(np.abs(
              native_qvel-reference.qvel))))
          max_qacc = max(max_qacc, float(np.max(np.abs(
              native_qacc-reference_qacc))))
          max_sensor = max(max_sensor, float(np.max(np.abs(
              native_sensor-reference_sensor))))
        # Fetch current-stage native feedback at a host visualization/query
        # boundary. The plugin still computes the force entirely on device.
        prepared = native.forward_skip(mujoco.mjtStage.mjSTAGE_NONE)
        feedback_device = plugin.evaluate(
            native.state, prepared["velocity"]["dynamics"])
        feedback = {key: value[0].detach().cpu().numpy().copy()
                    for key, value in feedback_device.items()}
        actual.qpos[:] = native_qpos
        actual.qvel[:] = native_qvel
        actual.qacc[:] = native_qacc
        actual.time = native_time
        if recorder is not None:
          # Host FK is only for OpenGL scene geometry. Preserve the native
          # qacc and sensor samples captured above for every reported metric.
          mujoco.mj_forward(model, actual)
          actual.qacc[:] = native_qacc
      else:
        actual.qfrc_applied[:] = 0
        mujoco.mj_forward(model, actual)
        feedback = _cpu_feedback(model, actual, site_id)
        actual.qfrc_applied[:] = feedback["qfrc"]
        mujoco.mj_step(model, actual)
        mujoco.mj_forward(model, actual)
        feedback = _cpu_feedback(model, actual, site_id)
      if native is None:
        measured = feedback
      else:
        # FK and sensors here are native stage measurements; only the CPU
        # renderer's scene data is refreshed independently.
        measured = feedback
      target_residual = float(np.linalg.norm(
          measured["target_position"]-measured["site_position"]))
      max_tracking = max(max_tracking, target_residual)
      max_force = max(max_force, float(np.linalg.norm(measured["force"])))
      trajectory.append(np.asarray(measured["site_position"]).copy())
      if recorder and step % recorder.every == 0:
        recorder.render(actual, mode=mode, force=measured["force"],
                        feedback=measured,
                        sensor_values=(native_sensor if native is not None else None))
    result = {
        "demo": "haptic_calligraphy",
        "mode": mode,
        "profile": PROFILE if native is not None else "MuJoCo CPU mj_step",
        "steps": steps,
        "simulated_seconds": float(actual.time),
        "max_qpos_error": max_qpos if reference is not None else None,
        "max_qvel_error": max_qvel if reference is not None else None,
        "max_qacc_error": max_qacc if reference is not None else None,
        "max_world_sensor_error": max_sensor if reference is not None else None,
        "max_tracking_residual": max_tracking,
        "max_cartesian_force": max_force,
        "trajectory_samples": len(trajectory),
        "force_is_jacobian_transpose_projected": True,
        "cpu_oracle_enabled": reference is not None,
    }
    if check:
      if max_qpos > 8e-4 or max_qvel > 5e-3 or max_qacc > 2e-2:
        raise AssertionError(json.dumps(result, indent=2))
      if max_sensor > 5e-3 or not (max_force > 1e-4):
        raise AssertionError(json.dumps(result, indent=2))
    return result
  finally:
    if recorder:
      recorder.close()
    if native is not None:
      default_registry.unregister(PLUGIN_NAME, PluginType.FORCE)


def _add_viewer_overlays(viewer, model, data, site_id, force, trail, *,
                         feedback=None, sensor_values=None):
  scene = viewer.user_scn
  scene.ngeom = 0
  point = (data.site_xpos[site_id] if feedback is None else
           np.asarray(feedback["site_position"]))
  trail.append(point.copy())
  if len(trail) > 240:
    del trail[:-240]
  target = (_target_numpy(data.time)[0] if feedback is None else
            np.asarray(feedback["target_position"]))
  if scene.ngeom < scene.maxgeom:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.035, 0, 0]), target,
        np.eye(3).reshape(-1), np.array([1, .26, .45, .92]))
    scene.ngeom += 1
  _Recorder._connector(scene, point, point + .16 * force,
                       [1.0, .76, .18, .98], .009)
  for start, end in zip(trail, trail[1:]):
    _Recorder._connector(scene, start, end, [.95, .23, .57, .8], .0035)
  sensors = (data.sensordata if sensor_values is None else
             np.asarray(sensor_values))
  pos_adr, vel_adr = (int(model.sensor_adr[i]) for i in range(2))
  _add_scene_label(scene, point, sensors[pos_adr:pos_adr+3],
                   sensors[vel_adr:vel_adr+3], force)


def _add_scene_label(scene, point, position, velocity, force):
  # Keep three short measurements over the shoulder rather than anchoring a
  # long line at the pen, where it extends past the right edge of the viewer.
  for line, (name, value) in enumerate((
      ("framepos", position), ("framelinvel", velocity), ("force", force))):
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
        np.array([.15, -.035, 1.34 - .07 * line]), np.eye(3).reshape(-1),
        np.array([.86, .93, 1, 1], dtype=np.float32))
    geom.label = name + " " + np.array2string(value, precision=2)
    scene.ngeom += 1


def run_viewer(mode="metal", seconds=0):
  from mujoco import viewer
  model = _load_model()
  site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME))
  native = plugin = None
  data = mujoco.MjData(model)
  qpos, qvel = _initial_state(model)
  data.qpos[:], data.qvel[:] = qpos, qvel
  mujoco.mj_forward(model, data)
  if mode == "metal":
    native, plugin = _register(model)
  elif mode != "cpu":
    raise ValueError("mode must be 'cpu' or 'metal'")
  deadline = time.monotonic() + seconds if seconds > 0 else None
  trail = []
  try:
    with viewer.launch_passive(model, data) as view:
      view.cam.lookat[:] = [.54, 0, .91]
      view.cam.distance, view.cam.azimuth, view.cam.elevation = 2.35, 92, -9
      while view.is_running() and (deadline is None or time.monotonic() < deadline):
        if native is not None:
          status = native.step()
          if np.any(status.detach().cpu().numpy()):
            raise RuntimeError(f"native simulation status: {status.tolist()}")
          snap = native.state.snapshot()
          data.qpos[:] = snap.qpos[0]
          data.qvel[:] = snap.qvel[0]
          data.time = float(snap.time[0])
          native_sensor = native.sensor_values()[0].detach().cpu().numpy().copy()
          prepared = native.forward_skip(mujoco.mjtStage.mjSTAGE_NONE)
          feedback_device = plugin.evaluate(
              native.state, prepared["velocity"]["dynamics"])
          feedback = {key: value[0].detach().cpu().numpy().copy()
                      for key, value in feedback_device.items()}
        else:
          data.qfrc_applied[:] = 0
          mujoco.mj_forward(model, data)
          feedback = _cpu_feedback(model, data, site_id)
          data.qfrc_applied[:] = feedback["qfrc"]
          mujoco.mj_step(model, data)
          native_sensor = None
        mujoco.mj_forward(model, data)
        if native is None:
          feedback = _cpu_feedback(model, data, site_id)
        with view.lock():
          _add_viewer_overlays(view, model, data, site_id,
                               feedback["force"], trail, feedback=feedback,
                               sensor_values=(native_sensor if native is not None
                                              else None))
        view.sync()
  finally:
    if native is not None:
      default_registry.unregister(PLUGIN_NAME, PluginType.FORCE)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("cpu", "metal"), default="metal")
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--viewer-seconds", type=float, default=0.0)
  parser.add_argument("--render", "--record", dest="render", type=Path,
                      help="save a host-rendered GIF with target and force overlays")
  args = parser.parse_args(argv)
  if args.steps <= 0 or args.viewer_seconds < 0:
    parser.error("steps must be positive and viewer-seconds nonnegative")
  if args.check and (args.mode != "metal" or not args.headless or args.steps > 200):
    parser.error("--check requires --mode metal --headless and 1..200 steps")
  if not args.headless and args.render is None:
    run_viewer(args.mode, args.viewer_seconds)
    return
  print(json.dumps(run(args.steps, args.mode, args.check, args.render), indent=2))


if __name__ == "__main__":
  main()
