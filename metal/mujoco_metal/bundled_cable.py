# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Device implementation of MuJoCo 3.10's bundled elasticity.cable plugin.

The cable callback has no plugin state.  Model topology, material constants,
rest lengths, and reference curvature are lowered once; each call evaluates
the current relative quaternion stress and projects the resulting body torque
through the current body angular Jacobians.
"""

from pathlib import Path

import mujoco
import numpy as np


_SHADER = Path(__file__).parent / "shaders" / "bundled_cable.metal"


def _geom_stiffness(model, geom, twist, bend):
  typ = int(model.geom_type[geom])
  size = np.asarray(model.geom_size[geom], dtype=np.float64)
  if typ in (int(mujoco.mjtGeom.mjGEOM_CYLINDER),
             int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
    radius = float(size[0])
    polar = np.pi * radius ** 4 / 2.0
    iy = iz = np.pi * radius ** 4 / 4.0
  elif typ == int(mujoco.mjtGeom.mjGEOM_BOX):
    h, w = float(size[1]), float(size[2])
    a, b = max(h, w), min(h, w)
    if a <= 0.0:
      raise ValueError("cable box cross-section must have positive dimensions")
    polar = a * b ** 3 * (16.0 / 3.0 - 3.36 * b / a *
                          (1.0 - b ** 4 / (a ** 4 * 12.0)))
    iy = (2.0 * w) ** 3 * 2.0 * h / 12.0
    iz = (2.0 * h) ** 3 * 2.0 * w / 12.0
  else:
    # The pinned callback leaves J, Iy, Iz at zero for all other geom types.
    # Such a segment may still be present in the cable chain, but its own
    # Compute body is skipped when all three values are zero.
    return np.zeros((3,), dtype=np.float64)
  result = np.asarray([polar * twist, iy * bend, iz * bend], dtype=np.float64)
  if not np.all(np.isfinite(result)):
    raise ValueError("cable stiffness is not finite")
  return result


def lower_cable(model, bundled_plugins):
  """Compile exact static cable-chain arrays from the pinned MjModel."""
  body_ids, previous, following, qadr = [], [], [], []
  stiffness, omega0, local_quat, rest_length = [], [], [], []
  data = mujoco.MjData(model)
  if int(model.nmocap):
    data.mocap_quat[:] = 0
  data.qpos[:] = np.asarray(model.qpos0)
  mujoco.mj_kinematics(model, data)
  parent = np.asarray(model.body_parentid, dtype=np.int32).copy()
  dof_body = np.asarray(model.dof_bodyid, dtype=np.int32).copy()

  for plugin in bundled_plugins.instances:
    if plugin.name != "mujoco.elasticity.cable":
      continue
    members = np.flatnonzero(
        np.asarray(model.body_plugin, dtype=np.int32) == int(plugin.instance_id))
    if not members.size:
      raise ValueError("cable plugin instance must be attached to at least one body")
    if not np.array_equal(members, np.arange(members[0], members[-1] + 1)):
      raise ValueError("cable plugin bodies must form the pinned contiguous chain")
    twist, bend = float(plugin.get("twist")), float(plugin.get("bend"))
    flat = str(plugin.get("flat", "")).strip().lower() == "true"
    start = len(body_ids)
    for ordinal, body_value in enumerate(members.tolist()):
      body = int(body_value)
      geom = int(model.body_geomadr[body])
      if geom < 0:
        raise ValueError("each cable body requires a cross-section geom")
      joint = int(model.body_jntadr[body])
      ndof = int(model.body_dofnum[body])
      if ordinal == 0 and ndof == 0:
        # The first body is never passed to LocalStress by the pinned callback:
        # it has no previous link, so its qpos quaternion is never read.
        address = 0
      else:
        if joint < 0 or ndof not in (3, 6):
          raise ValueError("non-root cable segments require a ball or free joint")
        joint_type = int(model.jnt_type[joint])
        if (int(model.body_jntnum[body]) != 1
            or joint_type not in (int(mujoco.mjtJoint.mjJNT_BALL),
                                  int(mujoco.mjtJoint.mjJNT_FREE))
            or (joint_type == int(mujoco.mjtJoint.mjJNT_BALL) and ndof != 3)
            or (joint_type == int(mujoco.mjtJoint.mjJNT_FREE) and ndof != 6)):
          raise ValueError(
              "non-root cable segments require exactly one ball or free joint")
        address = int(model.jnt_qposadr[joint]) + ndof - 3
        if address < 0 or address + 4 > int(model.nq):
          raise ValueError("cable joint quaternion address is outside qpos")
      local = np.asarray(model.body_quat[body], dtype=np.float64)
      reference = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
      if ordinal and not flat:
        mujoco.mju_subQuat(reference, local,
                           np.asarray(data.qpos[address:address + 4]))
      prior = int(members[ordinal - 1]) if ordinal else -1
      nxt = int(members[ordinal + 1]) if ordinal + 1 < len(members) else -1
      length = (float(np.linalg.norm(data.xpos[body] - data.xpos[prior]))
                if prior >= 0 else 0.0)
      if ordinal and (not np.isfinite(length) or length <= 0.0):
        raise ValueError("cable chain rest lengths must be finite and positive")
      k = _geom_stiffness(model, geom, twist, bend)
      row = np.asarray([k[0], k[1], k[2], length], dtype=np.float32)
      if not np.all(np.isfinite(row)):
        raise ValueError("cable constants exceed float32 device range")
      body_ids.append(body)
      previous.append(prior)
      following.append(nxt)
      qadr.append(address)
      stiffness.append(row)
      omega0.append(np.asarray(reference, dtype=np.float32))
      local_quat.append(np.asarray(local, dtype=np.float32))
      rest_length.append(length)

  n = len(body_ids)
  arrays = {
      "body_ids": np.asarray(body_ids, dtype=np.int32),
      "previous": np.asarray(previous, dtype=np.int32),
      "following": np.asarray(following, dtype=np.int32),
      "qadr": np.asarray(qadr, dtype=np.int32),
      "stiffness": np.asarray(stiffness, dtype=np.float32).reshape(n, 4),
      "omega0": np.asarray(omega0, dtype=np.float32).reshape(n, 3),
      "local_quat": np.asarray(local_quat, dtype=np.float32).reshape(n, 4),
      "body_parent": parent,
      "dof_body": dof_body,
      "rest_length": np.asarray(rest_length, dtype=np.float64),
  }
  return arrays


def cable_workspace_sizes(model, bundled_plugins=None):
  """Return exact persistent native metadata sizes, in scalar words."""
  if bundled_plugins is None:
    from mujoco_metal.bundled_plugins import lower_bundled_plugins
    bundled_plugins = lower_bundled_plugins(model)
  arrays = lower_cable(model, bundled_plugins)
  n = int(arrays["body_ids"].size)
  if not n:
    return {}
  return {
      "body_ids": n,
      "previous": n,
      "following": n,
      "qadr": n,
      "stiffness": 4 * n,
      "omega0": 3 * n,
      "local_quat": 4 * n,
      "body_parent": int(model.nbody),
      "dof_body": int(model.nv),
      "dims": 5,
      "flags": 1,
  }


def _quat_mul(a, b):
  w, x, y, z = a
  W, X, Y, Z = b
  return np.asarray([w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
                     w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W])


def _quat_rotate(q, v):
  qv = q[1:]
  t = 2.0 * np.cross(qv, v)
  return v + q[0] * t + np.cross(qv, t)


def _quat_velocity(q):
  axis = np.asarray(q[1:], dtype=np.float64).copy()
  sine = np.linalg.norm(axis)
  if sine:
    axis /= sine
  speed = 2.0 * np.arctan2(sine, float(q[0]))
  if speed > np.pi:
    speed -= 2.0 * np.pi
  return axis * speed


def cable_force_reference(model, qpos, body_quat_world, cdof, cable):
  """Independent NumPy transcription for oracle and producer tests."""
  qpos = np.asarray(qpos, dtype=np.float64)
  body_quat_world = np.asarray(body_quat_world, dtype=np.float64)
  cdof = np.asarray(cdof, dtype=np.float64)
  out = np.zeros((int(model.nv),), dtype=np.float64)
  arrays = cable
  n = int(arrays["body_ids"].size)
  for s in range(n):
    if not np.any(np.asarray(arrays["stiffness"][s, :3]) != 0.0):
      continue
    body = int(arrays["body_ids"][s])
    local_q = _quat_mul(arrays["local_quat"][s],
                        qpos[int(arrays["qadr"][s]):int(arrays["qadr"][s])+4])
    omega = _quat_velocity(local_q)
    tmp = -arrays["stiffness"][s, :3] * (omega - arrays["omega0"][s])
    if arrays["previous"][s] >= 0:
      tmp /= arrays["stiffness"][s, 3]
      cur = _quat_rotate(local_q * np.asarray([1., -1., -1., -1.]), tmp)
    else:
      cur = np.zeros(3)
    nxt_body = int(arrays["following"][s])
    if nxt_body >= 0:
      next_s = s + 1
      next_q = _quat_mul(arrays["local_quat"][next_s],
                         qpos[int(arrays["qadr"][next_s]):int(arrays["qadr"][next_s])+4])
      next_omega = _quat_velocity(next_q)
      next_tmp = -arrays["stiffness"][next_s, :3] * (
          next_omega - arrays["omega0"][next_s]) / arrays["stiffness"][next_s, 3]
      cur -= next_tmp
    torque = _quat_rotate(body_quat_world[body], cur)
    ancestor = body
    while ancestor > 0:
      for dof in np.flatnonzero(arrays["dof_body"] == ancestor):
        out[dof] += np.dot(cdof[dof, :3], torque)
      ancestor = int(arrays["body_parent"][ancestor])
  return out


class MetalBundledCable:
  """Persistent native cable metadata and additive passive-force stage."""

  def __init__(self, model, bundled_plugins, *, batch_size, device):
    import torch
    arrays = lower_cable(model, bundled_plugins)
    self.nsegment = int(arrays["body_ids"].size)
    self.batch_size, self.nv, self.nq = (
        int(batch_size), int(model.nv), int(model.nq))
    if not self.nsegment:
      raise ValueError("MetalBundledCable requires a cable plugin instance")
    self._torch = torch
    self._device = torch.device(device)
    if self._device.type != "mps":
      raise ValueError("MetalBundledCable requires an MPS device")
    self._body_ids = torch.as_tensor(arrays["body_ids"].copy(), dtype=torch.int32, device=device)
    self._previous = torch.as_tensor(arrays["previous"].copy(), dtype=torch.int32, device=device)
    self._following = torch.as_tensor(arrays["following"].copy(), dtype=torch.int32, device=device)
    self._qadr = torch.as_tensor(arrays["qadr"].copy(), dtype=torch.int32, device=device)
    self._stiffness = torch.as_tensor(arrays["stiffness"].copy(), dtype=torch.float32, device=device)
    self._omega0 = torch.as_tensor(arrays["omega0"].copy(), dtype=torch.float32, device=device)
    self._local_quat = torch.as_tensor(arrays["local_quat"].copy(), dtype=torch.float32, device=device)
    self._body_parent = torch.as_tensor(arrays["body_parent"].copy(), dtype=torch.int32, device=device)
    self._dof_body = torch.as_tensor(arrays["dof_body"].copy(), dtype=torch.int32, device=device)
    self._dims = torch.tensor([self.batch_size, self.nv, int(model.nbody),
                               self.nsegment, self.nq] + [1] * self.batch_size,
                              dtype=torch.int32,
                              device=device)
    self._flags = torch.zeros((1,), dtype=torch.int32, device=device)
    self._dummy = torch.zeros((1,), dtype=torch.float32, device=device)
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.bundled_cable_force

  def run_device(self, qpos, poses, cdof, force_target, *, capture_target=None,
                 world_mask=None):
    torch = self._torch
    b, nv, nb, ns = self.batch_size, self.nv, int(self._body_parent.numel()), self.nsegment
    def on_device(value):
      return (value.device.type == self._device.type
              and (self._device.index is None
                   or value.device.index == self._device.index))
    if not nv:
      return force_target
    if (not isinstance(qpos, torch.Tensor) or qpos.ndim != 2
        or tuple(qpos.shape) != (b, self.nq)
        or qpos.dtype != torch.float32 or not qpos.is_contiguous()
        or not on_device(qpos)):
      raise ValueError("cable qpos must be contiguous float32 MPS [B,nq]")
    body_quat = poses.get("body_quat") if isinstance(poses, dict) else None
    if (not isinstance(body_quat, torch.Tensor)
        or tuple(body_quat.shape) != (b, nb, 4)
        or body_quat.dtype != torch.float32 or not body_quat.is_contiguous()
        or not on_device(body_quat)):
      raise ValueError("cable body_quat must be contiguous float32 MPS [B,nbody,4]")
    if (not isinstance(cdof, torch.Tensor)
        or tuple(cdof.shape) != (b, nv, 6) or cdof.dtype != torch.float32
        or not cdof.is_contiguous() or not on_device(cdof)):
      raise ValueError("cable cdof must be float32 MPS [B,nv,6]")
    for name, value in (("force_target", force_target),):
      if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (b, nv)
          or value.dtype != torch.float32 or not value.is_contiguous()
          or not on_device(value)):
        raise ValueError(f"cable {name} must be float32 MPS [B,nv]")
    capture = capture_target is not None
    if capture and (not isinstance(capture_target, torch.Tensor)
                    or tuple(capture_target.shape) != (b, nv)
                    or capture_target.dtype != torch.float32
                    or not capture_target.is_contiguous()
                    or not on_device(capture_target)):
      raise ValueError("cable capture_target must be float32 MPS [B,nv]")
    if world_mask is None:
      self._dims[5:5 + b].fill_(1)
    else:
      if (not isinstance(world_mask, torch.Tensor)
          or world_mask.dtype != torch.int32 or tuple(world_mask.shape) != (b,)
          or world_mask.device.type != "mps" or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous int32 MPS [B]")
      self._dims[5:5 + b].copy_(world_mask)
    self._flags.fill_(int(capture))
    self._kernel(
        qpos.reshape(-1), body_quat.reshape(-1), cdof.reshape(-1),
        self._body_ids, self._previous, self._following, self._qadr,
        self._stiffness.reshape(-1), self._omega0.reshape(-1),
        self._local_quat.reshape(-1), self._body_parent, self._dof_body,
        force_target.reshape(-1),
        capture_target.reshape(-1) if capture else self._dummy,
        self._dims, self._flags,
        threads=(b * nv,), group_size=(min(128, b * nv),))
    return force_target
