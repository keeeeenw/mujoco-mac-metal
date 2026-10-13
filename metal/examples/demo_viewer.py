# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Main-thread OpenGL visualization of an actual native Metal simulation."""
import threading
import time
import mujoco
import numpy as np


def run_native_viewer(model, sim, *, steps, seconds=0, title="Native Metal",
                      lookat=(0, 0, 0), distance=3, azimuth=135, elevation=-15,
                      before_step=None):
  """Advance Metal only; CPU forward queries mirror native poses for display."""
  import glfw
  if threading.current_thread() is not threading.main_thread():
    raise RuntimeError("The Mac GLFW viewer requires the main thread")
  if not np.isfinite(seconds) or seconds<0:
    raise ValueError("viewer seconds must be finite and nonnegative")
  if isinstance(steps,bool) or not isinstance(steps,int) or steps<=0:
    raise ValueError("viewer steps must be a positive integer")
  if sim.state._device.type!="mps":
    raise ValueError("The native viewer requires a Metal simulation")
  display=mujoco.MjData(model)
  completed = 0
  if not glfw.init():
    raise RuntimeError("GLFW could not initialize the Mac viewer")
  window = None
  context = None
  try:
    window = glfw.create_window(1000, 700, title, None, None)
    if window is None:
      raise RuntimeError("GLFW could not create an OpenGL viewer window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
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
    for step in range(steps):
      if glfw.window_should_close(window) or (seconds > 0 and time.monotonic()-started >= seconds):
        break
      before = time.monotonic()
      commands = before_step(sim, step) if before_step else {}
      sim.step(1, **(commands or {}))
      np.testing.assert_array_equal(sim.state._status.detach().cpu().numpy(), [0])
      display.qpos[:] = sim.state.qpos[0].cpu().numpy()
      display.qvel[:] = sim.state.qvel[0].cpu().numpy()
      if model.na:
        display.act[:] = sim.state.act[0].cpu().numpy()
      if model.nu:
        display.ctrl[:] = sim._control[0].cpu().numpy()
      if model.nmocap:
        display.mocap_pos[:] = sim.state.mocap_pos[0].cpu().numpy()
        display.mocap_quat[:] = sim.state.mocap_quat[0].cpu().numpy()
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
  return {"viewer_steps": completed, "steps": completed, "backend": "native Metal physics; upstream OpenGL visualization"}

