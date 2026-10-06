# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Device implementation of MuJoCo 3.10's bundled ``mujoco.pid`` actuator."""

from pathlib import Path

import numpy as np


_SHADER = Path(__file__).parent / "shaders" / "bundled_pid.metal"


def lower_pid_actuators(model, bundled_plugins):
  """Return per-actuator PID constants, state flags, and plugin mask.

  Plugin activation slots are validated by MuJoCo's pinned plugin init. This
  lowering independently checks their contiguous addresses so the native
  kernel never indexes outside ``MjData.act``.
  """
  nu, na = int(model.nu), int(model.na)
  params = np.zeros((max(nu, 1), 5), dtype=np.float32)
  params[:, 3] = -1.0  # absent imax means no integral clamp
  flags = np.zeros((max(nu, 1), 2), dtype=np.int32)
  mask = np.zeros((max(nu, 1),), dtype=np.int32)
  plugin_ids = np.asarray(model.actuator_plugin, dtype=np.int32).reshape(-1)
  actadr = np.asarray(model.actuator_actadr, dtype=np.int32).reshape(-1)
  actnum = np.asarray(model.actuator_actnum, dtype=np.int32).reshape(-1)
  dyntype = np.asarray(model.actuator_dyntype, dtype=np.int32).reshape(-1)
  cursor = 0
  for actuator in range(nu):
    count = int(actnum[actuator])
    address = int(actadr[actuator])
    if count and address != cursor:
      raise ValueError("actuator activation slots must densely cover [0, na)")
    cursor += count
    instance_id = int(plugin_ids[actuator])
    if instance_id < 0:
      continue
    instance = bundled_plugins.instance(instance_id)
    if instance.name != "mujoco.pid":
      raise ValueError(
          f"actuator {actuator} uses unsupported bundled plugin {instance.name!r}")
    config = instance.config
    kp, ki, kd = (float(config[key]) for key in ("kp", "ki", "kd"))
    imax = config.get("imax") or None
    slewmax = config.get("slewmax") or None
    int_enabled = ki != 0.0
    slew_enabled = slewmax is not None
    expected = int(int_enabled) + int(slew_enabled)
    if int(dyntype[actuator]) in (1, 2, 3):  # INTEGRATOR/FILTER/FILTEREXACT
      expected += 1
    if count != expected:
      raise ValueError(
          f"PID actuator {actuator} has {count} activation slots, expected {expected}")
    int_limit = -1.0 if imax is None or not int_enabled else float(imax) / ki
    row = (kp, ki, kd, int_limit,
           -1.0 if slewmax is None else float(slewmax))
    if not np.all(np.isfinite(row)):
      raise ValueError("PID plugin parameters must be finite float32 values")
    if not np.all(np.isfinite(np.asarray(row, dtype=np.float32))):
      raise ValueError("PID plugin parameters exceed float32 range")
    params[actuator] = row
    flags[actuator] = (int(int_enabled), int(slew_enabled))
    mask[actuator] = 1
  if cursor != na:
    raise ValueError("activation slots must densely cover [0, na)")
  return params, flags, mask


def validate_pid_profile_plugins(model, bundled_plugins=None):
  """Validate bundled plugin force terms used by implicit profiles.

  MuJoCo 3.10's PID plugin intentionally registers no actuator force
  derivative callback (see ``plugin/actuator/pid.cc`` TODO), so its force is
  explicit with respect to the implicit velocity solve. The cable callback is
  pose/Jacobian dependent and has zero velocity derivative; touch-grid only
  writes ACC sensor outputs; registered SDFs contribute position-dependent
  collision geometry. These source contracts make them safe for the implicit
  velocity derivative, while unsupported plugin classes remain rejected.
  """
  if not int(model.nplugin):
    return bundled_plugins
  if bundled_plugins is None:
    from mujoco_metal.bundled_plugins import lower_bundled_plugins
    bundled_plugins = lower_bundled_plugins(model)
  supported = {"mujoco.pid", "mujoco.elasticity.cable",
               "mujoco.sensor.touch_grid"}
  unsupported = sorted({instance.name for instance in bundled_plugins.instances
                        if instance.name not in supported
                        and not instance.name.startswith("mujoco.sdf.")})
  if unsupported:
    raise ValueError(
        "integrated plugin force/derivative support does not include "
        f"{unsupported}")
  names = {instance.name for instance in bundled_plugins.instances}
  if "mujoco.pid" in names:
    lower_pid_actuators(model, bundled_plugins)
  if "mujoco.elasticity.cable" in names:
    from mujoco_metal.bundled_cable import lower_cable
    lower_cable(model, bundled_plugins)
  if "mujoco.sensor.touch_grid" in names:
    from mujoco_metal.bundled_touch_grid import lower_touch_grid_sensors
    lower_touch_grid_sensors(model, bundled_plugins)
  return bundled_plugins


class MetalBundledPID:
  """Reusable fixed-shape PID actuator stage on the simulation device."""

  def __init__(self, model, bundled_plugins, *, batch_size, device,
               actuator_device_data):
    import torch
    from mujoco_metal.stateful_actuation import _INT32_MAX

    b, nu, na = int(batch_size), int(model.nu), int(model.na)
    if max(b * max(nu, 1), nu * 5, nu * 2, max(na, 1)) > _INT32_MAX:
      raise ValueError("bundled PID workspace exceeds int32 shader indexing")
    params, flags, mask = lower_pid_actuators(model, bundled_plugins)
    self._torch = torch
    self._device = torch.device(device)
    if self._device.type != "mps":
      raise ValueError("MetalBundledPID requires an MPS device")
    self.batch_size, self.nu, self.na = b, nu, na
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.bundled_pid_stage
    self._params = torch.as_tensor(params.copy(), dtype=torch.float32, device=device)
    self._flags = torch.as_tensor(flags.copy(), dtype=torch.int32, device=device)
    self._mask = torch.as_tensor(mask.copy(), dtype=torch.int32, device=device)
    self._actadr = actuator_device_data["actadr"]
    self._actnum = actuator_device_data["actnum"]
    self._dyntype = actuator_device_data["dyntype"]
    self._actearly = actuator_device_data["actearly"]
    self._actlimited = actuator_device_data["actlimited"]
    self._actrange = actuator_device_data["actrange"]
    self._ctrllimited = actuator_device_data["ctrllimited"]
    self._ctrlrange = actuator_device_data["ctrlrange"]
    self._dynprm = actuator_device_data["dynprm"]
    self._dims = torch.tensor([
        nu, na, b,
        int(bool(int(model.opt.disableflags) & int(__import__("mujoco").mjtDisableBit.mjDSBL_ACTUATION))),
        int(bool(int(model.opt.disableflags) & int(__import__("mujoco").mjtDisableBit.mjDSBL_CLAMPCTRL))),
    ], dtype=torch.int32, device=device)
    self._dt = torch.tensor([float(model.opt.timestep)], dtype=torch.float32, device=device)
    self._force = torch.zeros((b, max(nu, 1)), dtype=torch.float32, device=device)
    self._dummy = torch.zeros((1,), dtype=torch.float32, device=device)
    self._has_actuators = bool(np.any(mask))

  @property
  def has_actuators(self):
    return self._has_actuators

  @property
  def force_buffer(self):
    return self._force[:, :self.nu]

  @property
  def actuator_mask(self):
    return self._mask

  def run_device(self, control, activation, act_dot, kin, time):
    torch = self._torch
    b, nu, na = self.batch_size, self.nu, self.na
    for name, value, shape in (
        ("control", control, (b, nu)), ("act_dot", act_dot, (b, max(na, 1))),
        ("length", kin["length"], (b, nu)),
        ("velocity", kin["velocity"], (b, nu)), ("time", time, (b,)),
    ):
      if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"PID {name} must be a tensor with shape {shape}")
      if value.dtype != torch.float32 or value.device.type != "mps" or not value.is_contiguous():
        raise ValueError(f"PID {name} must be contiguous float32 on MPS")
    if na and (not isinstance(activation, torch.Tensor)
               or tuple(activation.shape) != (b, na)
               or activation.dtype != torch.float32 or activation.device.type != "mps"
               or not activation.is_contiguous()):
      raise ValueError("PID activation must be contiguous float32 [B,na] on MPS")
    if not nu or not self.has_actuators:
      self._force.zero_()
      return self.force_buffer
    self._kernel(
        control.reshape(-1), activation.reshape(-1) if na else self._dummy,
        act_dot.reshape(-1), kin["length"].reshape(-1), kin["velocity"].reshape(-1),
        self._mask, self._params.reshape(-1), self._flags.reshape(-1),
        self._actadr, self._actnum, self._dyntype, self._actearly,
        self._actlimited, self._actrange.reshape(-1), self._dynprm.reshape(-1),
        time.reshape(-1), self._dims, self._dt, self._force.reshape(-1),
        self._ctrllimited, self._ctrlrange.reshape(-1),
        threads=(b,), group_size=(1,),
    )
    return self.force_buffer
