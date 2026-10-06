# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
"""Device spatial queries ported from pinned MuJoCo 3.10 core utilities.

Compiled topology is prepared on the host. Runtime transforms, Jacobians,
velocity, acceleration and wrench projection use device tensors exclusively.
The explicit CPU device is for numerical validation, not a physics fallback.
Sources: engine_core_util.c mj_jac/mj_jacDot/mj_objectVelocity/
mj_objectAcceleration and engine_core_smooth.c mj_camlight/mj_rnePostConstraint.
"""
import numbers
from pathlib import Path

import mujoco
import numpy as np
import torch


_PLUGIN_FORCE_LIBRARY = None


def _plugin_force_library():
  global _PLUGIN_FORCE_LIBRARY
  if _PLUGIN_FORCE_LIBRARY is None:
    _PLUGIN_FORCE_LIBRARY = torch.mps.compile_shader(
        (Path(__file__).with_name("shaders") / "spatial_plugin_force.metal").read_text())
  return _PLUGIN_FORCE_LIBRARY


def _rotate(q, value):
  return value + 2*torch.cross(q[..., 1:],
      torch.cross(q[..., 1:], value, dim=-1) + q[..., :1]*value, dim=-1)


def _qmul(a, b):
  return torch.cat((a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True),
                    a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] +
                    torch.cross(a[..., 1:], b[..., 1:], dim=-1)), -1)


def _rotation(q):
  basis = torch.eye(3, dtype=q.dtype, device=q.device).expand(q.shape[:-1] + (3, 3))
  return _rotate(q.unsqueeze(-2).expand(q.shape[:-1] + (3, 4)), basis).transpose(-1, -2)


def _quat_matrix(q):
  """Pinned mju_quat2Mat polynomial, without silently normalizing input."""
  w, x, y, z = q.unbind(-1)
  return torch.stack((w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y),
                      2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x),
                      2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z),
                     dim=-1).reshape(q.shape[:-1] + (3, 3))


def _unit(v):
  length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
  fallback = torch.zeros_like(v)
  fallback[..., 0] = 1
  return torch.where(length < 1e-15, fallback, v/length.clamp_min(1e-15))


class DeviceSpatialQueries:
  """Owned query outputs from a complete smooth stage, in rot:lin convention."""

  def __init__(self, model, batch_size, device):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError("spatial queries require pinned MuJoCo 3.10.0")
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
    b, nb, nv = int(batch_size), int(model.nbody), int(model.nv)
    _validate_workspace_index_capacity(b, {"body_dof": nb*max(nv, 1),
        "jacobian": b*3*nv, "body_motion": b*nb*6},
        {"nbody": nb, "nv": nv})
    self.model, self.batch_size, self.device = model, b, torch.device(device)
    self.nv = nv
    chain = np.zeros((nb, nv), dtype=np.float32)
    for body in range(nb):
      weld = int(model.body_weldid[body])
      if weld:
        dof = int(model.body_dofadr[weld] + model.body_dofnum[weld] - 1)
        while dof >= 0:
          chain[body, dof] = 1
          dof = int(model.dof_parentid[dof])
    def tensor(value, dtype=torch.float32):
      return torch.tensor(np.asarray(value).copy(), dtype=dtype, device=self.device)
    self._chain = tensor(chain)
    self._parent = tuple(int(x) for x in model.body_parentid)
    self._root = tuple(int(x) for x in model.body_rootid)
    self._root_device = tensor(model.body_rootid, torch.int32)
    self._body_mass = tensor(model.body_mass)
    self._body_inertia = tensor(model.body_inertia)
    self._subtree_mass = tensor(model.body_subtreemass)
    self._gravity = tensor(model.opt.gravity)
    self._gravity_enabled = not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
    self._quat_dof = []
    for dof in range(nv):
      joint = int(model.dof_jntid[dof])
      kind = int(model.jnt_type[joint])
      if kind == int(mujoco.mjtJoint.mjJNT_BALL) or (kind == int(mujoco.mjtJoint.mjJNT_FREE)
          and dof >= int(model.jnt_dofadr[joint])+3):
        self._quat_dof.append((dof, int(model.dof_bodyid[dof])))
    self._camera = {name: tensor(getattr(model, name)) for name in
        ("cam_pos", "cam_quat", "cam_pos0", "cam_poscom0", "cam_mat0")}
    # Reused scratch for guarded FORCE plugins. Unselected rows are deliberately
    # left untouched; callers consume only selected rows through the same mask.
    self._plugin_force = torch.empty(
        (b, max(nv, 1)), dtype=torch.float32, device=self.device)
    self._plugin_mask = torch.empty((b,), dtype=torch.int32, device=self.device)
    self._plugin_dims = torch.tensor(
        [b, nv, nb, 0, 0, 0], dtype=torch.int32, device=self.device)
    self._magnetic_charge = torch.zeros((1,), dtype=torch.float32, device=self.device)
    self._site_feedback_gains = torch.zeros((2,), dtype=torch.float32, device=self.device)

  def _prepare_plugin_mask(self, compute_mask):
    if (not isinstance(compute_mask, torch.Tensor)
        or tuple(compute_mask.shape) != (self.batch_size,)
        or compute_mask.dtype != torch.bool
        or compute_mask.device != self._plugin_mask.device or not compute_mask.is_contiguous()):
      raise ValueError("compute_mask must be contiguous bool[batch] on the query device")
    self._plugin_mask.copy_(compute_mask)
    return self._plugin_mask

  def masked_magnetic_force(self, dynamics, *, body_ids, field, charge,
                            compute_mask):
    """Project built-in magnetic FORCE rows through native point Jacobians.

    Each Metal thread returns before touching stage inputs or the output when
    its selector is zero. The output is a borrowed contiguous ``[B,nv]`` view.
    """
    if self.device.type != "mps":
      raise RuntimeError("guarded spatial FORCE execution requires MPS")
    mask = self._prepare_plugin_mask(compute_mask)
    poses = dynamics.get("poses") if isinstance(dynamics, dict) else None
    if not isinstance(poses, dict):
      raise ValueError("magnetic force requires complete smooth-stage poses")
    cvel = self._tensor(dynamics.get("cvel"),
                        (self.batch_size, self.model.nbody, 6), "cvel")
    root_com = self._tensor(dynamics.get("root_com"),
                            (self.batch_size, self.model.nbody, 3), "root_com")
    cdof = self._tensor(dynamics.get("cdof"),
                        (self.batch_size, self.nv, 6), "cdof")
    point = self._tensor(poses.get("inertial_pos"),
                         (self.batch_size, self.model.nbody, 3), "inertial_pos")
    field = self._tensor(field, (3,), "magnetic field")
    bodies = body_ids
    if (not isinstance(bodies, torch.Tensor) or bodies.dtype != torch.int32
        or bodies.device != self._plugin_mask.device or bodies.ndim != 1
        or not bodies.is_contiguous()):
      raise ValueError("body_ids must be contiguous int32 on the query device")
    if any(not value.is_contiguous() for value in
           (cvel, root_com, cdof, point, field, self._chain,
            self._root_device, bodies, self._plugin_force)):
      raise ValueError("guarded magnetic inputs must be contiguous")
    self._plugin_dims[3] = int(bodies.numel())
    self._plugin_dims[4] = 0
    self._plugin_dims[5] = 0
    self._magnetic_charge[0] = float(charge)
    library = _plugin_force_library()
    library.masked_magnetic_force(
        cvel.reshape(-1), root_com.reshape(-1), cdof.reshape(-1),
        point.reshape(-1), self._chain.reshape(-1), self._root_device,
        bodies, field, self._plugin_force.reshape(-1), self._plugin_dims,
        mask, self._magnetic_charge, threads=(self.batch_size,), group_size=(1,))
    return self._plugin_force[:, :self.nv]

  def masked_site_feedback_force(self, dynamics, *, time, site_id, body_id,
                                 center, amplitude, frequency, phase, kp, kd,
                                 compute_mask):
    """Guarded current-stage site feedback force and point-Jacobian path."""
    if self.device.type != "mps":
      raise RuntimeError("guarded spatial FORCE execution requires MPS")
    mask = self._prepare_plugin_mask(compute_mask)
    poses = dynamics.get("poses") if isinstance(dynamics, dict) else None
    if not isinstance(poses, dict):
      raise ValueError("site feedback requires complete smooth-stage poses")
    cvel = self._tensor(dynamics.get("cvel"),
                        (self.batch_size, self.model.nbody, 6), "cvel")
    root_com = self._tensor(dynamics.get("root_com"),
                            (self.batch_size, self.model.nbody, 3), "root_com")
    cdof = self._tensor(dynamics.get("cdof"),
                        (self.batch_size, self.nv, 6), "cdof")
    site_pos = self._tensor(poses.get("site_pos"),
                            (self.batch_size, self.model.nsite, 3), "site_pos")
    time = self._tensor(time, (self.batch_size,), "time")
    vectors = []
    for value, name in ((center, "center"), (amplitude, "amplitude"),
                        (frequency, "frequency"), (phase, "phase")):
      vectors.append(self._tensor(value, (3,), name))
    if not 0 <= int(site_id) < int(self.model.nsite):
      raise ValueError("site_id is outside the compiled model")
    if not 0 <= int(body_id) < int(self.model.nbody):
      raise ValueError("body_id is outside the compiled model")
    if any(not value.is_contiguous() for value in
           (cvel, root_com, cdof, site_pos, time, *vectors, self._chain,
            self._root_device, self._plugin_force)):
      raise ValueError("guarded site-feedback inputs must be contiguous")
    self._plugin_dims[3] = int(self.model.nsite)
    self._plugin_dims[4] = int(site_id)
    self._plugin_dims[5] = int(body_id)
    self._site_feedback_gains[0] = float(kp)
    self._site_feedback_gains[1] = float(kd)
    library = _plugin_force_library()
    library.masked_site_feedback_force(
        cvel.reshape(-1), root_com.reshape(-1), cdof.reshape(-1),
        site_pos.reshape(-1), self._chain.reshape(-1), self._root_device,
        time, *vectors, self._plugin_force.reshape(-1), self._plugin_dims,
        mask, self._site_feedback_gains,
        threads=(self.batch_size,), group_size=(1,))
    return self._plugin_force[:, :self.nv]

  def _id(self, value, count, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
      raise TypeError(f"{name} must be an integer")
    if not 0 <= int(value) < count:
      raise ValueError(f"{name} is outside the compiled model")
    return int(value)

  def _tensor(self, value, shape, name):
    if (not isinstance(value, torch.Tensor)
        or value.device.type != self.device.type
        or (self.device.index is not None
            and value.device.index != self.device.index)
        or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape)):
      raise ValueError(f"{name} must be float32{tuple(shape)} on {self.device}")
    return value

  def jac(self, dynamics, point, body):
    """World translation/rotation Jacobians, each [batch,3,nv]."""
    body = self._id(body, self.model.nbody, "body")
    point = self._tensor(point, (self.batch_size, 3), "point")
    cdof = self._tensor(dynamics["cdof"], (self.batch_size, self.nv, 6), "cdof")
    offset = point-dynamics["root_com"][:, self._root[body]]
    angular = cdof[..., :3]
    linear = cdof[..., 3:] + torch.cross(angular, offset[:, None].expand_as(angular), dim=-1)
    mask = self._chain[body][None, :, None]
    # where, rather than multiplication, keeps unrelated invalid DOFs out.
    return (torch.where(mask != 0, linear, 0).transpose(-1, -2).contiguous(),
            torch.where(mask != 0, angular, 0).transpose(-1, -2).contiguous())

  def local_to_global(self, dynamics, position, quaternion, body, sameframe=0):
    """Owned position/matrix pair for all pinned mjtSameFrame alignments.

    A missing local position or quaternion produces None for that output,
    like the corresponding null source input/output. Rotation-only alignment
    still translates the local position using the regular body frame.
    Local quaternions are used as supplied, including nonunit quaternions.
    """
    body = self._id(body, self.model.nbody, "body")
    sameframe = self._id(sameframe, 5, "sameframe")
    if position is not None:
      position = self._tensor(position, (self.batch_size, 3), "position")
    if quaternion is not None:
      quaternion = self._tensor(quaternion, (self.batch_size, 4), "quaternion")
    poses = dynamics["poses"]
    transformed_position = transformed_matrix = None
    if position is not None:
      if sameframe == int(mujoco.mjtSameFrame.mjSAMEFRAME_BODY):
        transformed_position = poses["body_pos"][:, body].clone()
      elif sameframe == int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA):
        transformed_position = poses["inertial_pos"][:, body].clone()
      else:
        rotation = _quat_matrix(poses["body_quat"][:, body])
        transformed_position = (torch.bmm(rotation, position.unsqueeze(-1)).squeeze(-1)
                                + poses["body_pos"][:, body])
    if quaternion is not None:
      if sameframe == int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE):
        transformed_matrix = _quat_matrix(_qmul(poses["body_quat"][:, body], quaternion))
      elif sameframe in (int(mujoco.mjtSameFrame.mjSAMEFRAME_BODY),
                         int(mujoco.mjtSameFrame.mjSAMEFRAME_BODYROT)):
        transformed_matrix = _quat_matrix(poses["body_quat"][:, body])
      else:
        transformed_matrix = _quat_matrix(poses["inertial_quat"][:, body])
    return transformed_position, transformed_matrix

  def jac_dot(self, dynamics, point, body):
    """Pinned mj_jacDot, including quaternion DOF motion-cross correction."""
    body = self._id(body, self.model.nbody, "body")
    point = self._tensor(point, (self.batch_size, 3), "point")
    cdof = dynamics["cdof"]
    derivative = dynamics["cdof_dot"].clone()
    for dof, owner in self._quat_dof:
      velocity = dynamics["cvel"][:, owner]
      derivative[:, dof, :3] = torch.cross(velocity[:, :3], cdof[:, dof, :3], dim=-1)
      derivative[:, dof, 3:] = (torch.cross(velocity[:, :3], cdof[:, dof, 3:], dim=-1)
          + torch.cross(velocity[:, 3:], cdof[:, dof, :3], dim=-1))
    offset = point-dynamics["root_com"][:, self._root[body]]
    velocity = dynamics["cvel"][:, body]
    point_velocity = velocity[:, 3:] + torch.cross(velocity[:, :3], offset, dim=-1)
    linear = (derivative[..., 3:] + torch.cross(derivative[..., :3],
        offset[:, None].expand(-1, self.nv, -1), dim=-1) + torch.cross(cdof[..., :3],
        point_velocity[:, None].expand(-1, self.nv, -1), dim=-1))
    mask = self._chain[body][None, :, None] != 0
    return (torch.where(mask, linear, 0).transpose(-1, -2).contiguous(),
            torch.where(mask, derivative[..., :3], 0).transpose(-1, -2).contiguous())

  def jac_point_axis(self, dynamics, point, axis, body):
    """Pinned point and axis Jacobians, each owned [batch,3,nv].

    The axis is expressed in world coordinates and is not normalized: the
    source computes each axis column as angular_jacobian_column cross axis.
    """
    axis = self._tensor(axis, (self.batch_size, 3), "axis")
    translation, rotation = self.jac(dynamics, point, body)
    columns = rotation.transpose(-1, -2)
    axis_jacobian = torch.cross(
        columns, axis[:, None, :].expand_as(columns), dim=-1)
    return translation, axis_jacobian.transpose(-1, -2).contiguous()

  def subtree_com(self, dynamics):
    positions = dynamics["poses"]["inertial_pos"]
    weighted = positions*self._body_mass[None, :, None]
    for body in range(self.model.nbody-1, 0, -1):
      weighted[:, self._parent[body]].add_(weighted[:, body])
    return torch.where(self._subtree_mass[None, :, None] >= 1e-15,
        weighted/self._subtree_mass[None, :, None].clamp_min(1e-15),
        positions)

  def kinematics(self, dynamics):
    """Owned pinned Cartesian kinematics fields from the native pose stage."""
    poses = dynamics["poses"]
    result = {field: poses[key].clone() for field, key in (
        ("xpos", "body_pos"), ("xquat", "body_quat"),
        ("xipos", "inertial_pos"), ("geom_xpos", "geom_pos"),
        ("site_xpos", "site_pos"), ("xanchor", "joint_anchor"),
        ("xaxis", "joint_axis"))}
    for field, key in (("xmat", "body_quat"),
                       ("ximat", "inertial_quat"),
                       ("geom_xmat", "geom_quat"),
                       ("site_xmat", "site_quat")):
      result[field] = _quat_matrix(poses[key])
    return result

  def center_of_mass_position(self, dynamics):
    """Pinned mj_comPos fields with root-COM-referenced ten-word inertia."""
    centers = self.subtree_com(dynamics)
    poses = dynamics["poses"]
    rotation = _quat_matrix(poses["inertial_quat"])
    inertia = torch.matmul(
        rotation * self._body_inertia[None, :, None, :],
        rotation.transpose(-1, -2))
    offset = poses["inertial_pos"] - centers[:, list(self._root)]
    mass = self._body_mass[None, :, None]
    x, y, z = offset.unbind(-1)
    m = mass[..., 0]
    # mju_inertCom: rotated principal inertia plus parallel-axis terms,
    # followed by first mass moments and mass. Spatial order is rot:lin.
    cinert = torch.stack((
        inertia[..., 0, 0] + m * (y*y + z*z),
        inertia[..., 1, 1] + m * (x*x + z*z),
        inertia[..., 2, 2] + m * (x*x + y*y),
        inertia[..., 0, 1] - m*x*y,
        inertia[..., 0, 2] - m*x*z,
        inertia[..., 1, 2] - m*y*z,
        m*x, m*y, m*z, m.expand(self.batch_size, -1)), dim=-1)
    cinert[:, 0].zero_()  # World inertia is explicitly zero in the source.
    return {"subtree_com": centers, "cinert": cinert,
            "cdof": dynamics["cdof"].clone()}

  def center_of_mass_velocity(self, dynamics):
    """Owned complete cvel and cdof_dot from the native smooth VEL stage."""
    return {name: dynamics[name].clone() for name in ("cvel", "cdof_dot")}

  def jac_subtree_com(self, dynamics, body):
    body = self._id(body, self.model.nbody, "body")
    output = torch.zeros((self.batch_size, 3, self.nv), dtype=torch.float32, device=self.device)
    for child in range(body, self.model.nbody):
      if child > body and int(self.model.body_parentid[child]) < body:
        break
      jacp, _ = self.jac(dynamics, dynamics["poses"]["inertial_pos"][:, child], child)
      output.add_(jacp*self._body_mass[child])
    return output/self._subtree_mass[body]

  def angmom_matrix(self, dynamics, body):
    """World angular momentum about subtree COM per unit generalized velocity.

    Pinned ``mj_angmomMat`` combines each body's rotated principal inertia
    with the moment of its linear momentum about the selected subtree COM.
    Compiled preorder determines subtree membership; runtime poses/Jacobians
    remain on device. Returns owned [batch,3,nv], including empty DOF models.
    """
    body = self._id(body, self.model.nbody, "body")
    output = torch.zeros((self.batch_size, 3, self.nv),
                         dtype=torch.float32, device=self.device)
    center = self.subtree_com(dynamics)[:, body]
    for child in range(body, self.model.nbody):
      if child > body and self._parent[child] < body:
        break
      position = dynamics["poses"]["inertial_pos"][:, child]
      jacp, jacr = self.jac(dynamics, position, child)
      rotation = _rotation(dynamics["poses"]["inertial_quat"][:, child])
      inertia = torch.bmm(rotation * self._body_inertia[child][None, None, :],
                          rotation.transpose(-1, -2))
      output.add_(torch.bmm(inertia, jacr))
      moment = torch.cross((position-center)[:, None, :].expand(-1, self.nv, -1),
                           jacp.transpose(-1, -2), dim=-1)
      output.add_(moment.transpose(-1, -2) * self._body_mass[child])
    return output

  def subtree_velocity(self, dynamics):
    """Owned full-state subtree COM velocity and COM angular momentum.

    This nonmutating query uses all supplied body motion values, rather than
    updating upstream's sleep-filtered persistent subtree sensor cache.
    The source-order momentum accumulation follows pinned mj_subtreeVel.
    """
    velocities = torch.stack([self.object_velocity(
        dynamics, mujoco.mjtObj.mjOBJ_BODY, body, False)
        for body in range(self.model.nbody)], dim=1)
    linear = velocities[..., 3:] * self._body_mass[None, :, None]
    rotation = _rotation(dynamics["poses"]["inertial_quat"])
    principal_velocity = torch.matmul(rotation.transpose(-1, -2),
                                      velocities[..., :3, None])
    angular = torch.matmul(rotation,
        principal_velocity * self._body_inertia[None, :, :, None]).squeeze(-1)
    centers = self.subtree_com(dynamics)
    positions = dynamics["poses"]["inertial_pos"]
    for body in range(self.model.nbody-1, -1, -1):
      if body:
        linear[:, self._parent[body]].add_(linear[:, body])
      linear[:, body].div_(self._subtree_mass[body].clamp_min(1e-15))
    for body in range(self.model.nbody-1, 0, -1):
      parent = self._parent[body]
      relative_body_momentum = (velocities[:, body, 3:]-linear[:, body]) * self._body_mass[body]
      angular[:, body].add_(torch.cross(positions[:, body]-centers[:, body],
                                      relative_body_momentum, dim=-1))
      angular[:, parent].add_(angular[:, body])
      relative_subtree_momentum = (linear[:, body]-linear[:, parent]) * self._subtree_mass[body]
      angular[:, parent].add_(torch.cross(centers[:, body]-centers[:, parent],
                                        relative_subtree_momentum, dim=-1))
    return {"subtree_linvel": linear, "subtree_angmom": angular}

  def recursive_newton_euler(self, dynamics, qvel, qacc, include_acceleration):
    """Owned full-state pinned RNE force, excluding joint/tendon armature.

    Spatial inertia and motion share each root subtree's COM reference.
    Unlike upstream's sleep-filtered in-place output, this query evaluates
    every body and returns all DOFs without mutating retained sleep state.
    """
    self._tensor(qvel, (self.batch_size, self.nv), "qvel")
    self._tensor(qacc, (self.batch_size, self.nv), "qacc")
    poses = dynamics["poses"]
    rotations = _rotation(poses["inertial_quat"])
    inertia = torch.matmul(
        rotations * self._body_inertia[None, :, None, :],
        rotations.transpose(-1, -2))
    acceleration = [qvel.new_zeros((self.batch_size, 6))]
    if self._gravity_enabled:
      acceleration[0][:, 3:] = -self._gravity
    force = [qvel.new_zeros((self.batch_size, 6))]

    def multiply_inertia(body, motion):
      offset = (poses["inertial_pos"][:, body]
                - dynamics["root_com"][:, self._root[body]])
      angular, linear = motion[:, :3], motion[:, 3:]
      momentum = (linear + torch.cross(angular, offset, dim=-1)) * self._body_mass[body]
      torque = torch.matmul(inertia[:, body], angular[..., None]).squeeze(-1)
      torque = torque + torch.cross(offset, momentum, dim=-1)
      return torch.cat((torque, momentum), -1)

    for body in range(1, self.model.nbody):
      first = int(self.model.body_dofadr[body])
      count = int(self.model.body_dofnum[body])
      current = acceleration[self._parent[body]].clone()
      if count:
        owned = slice(first, first+count)
        current.add_((dynamics["cdof_dot"][:, owned]
                      * qvel[:, owned, None]).sum(1))
        if include_acceleration:
          current.add_((dynamics["cdof"][:, owned]
                        * qacc[:, owned, None]).sum(1))
      acceleration.append(current)
      velocity = dynamics["cvel"][:, body]
      momentum = multiply_inertia(body, velocity)
      cross = torch.cat((
          torch.cross(velocity[:, :3], momentum[:, :3], dim=-1)
          + torch.cross(velocity[:, 3:], momentum[:, 3:], dim=-1),
          torch.cross(velocity[:, :3], momentum[:, 3:], dim=-1)), -1)
      force.append(multiply_inertia(body, current) + cross)
    for body in range(self.model.nbody-1, 0, -1):
      parent = self._parent[body]
      if parent:
        force[parent] = force[parent] + force[body]
    if not self.nv:
      return qvel.new_zeros((self.batch_size, 0))
    return torch.stack([
        (dynamics["cdof"][:, dof] * force[int(self.model.dof_bodyid[dof])]).sum(-1)
        for dof in range(self.nv)], -1)

  def object_frame(self, dynamics, objtype, objid):
    poses = dynamics["poses"]
    if isinstance(objtype, bool) or not isinstance(objtype, (numbers.Integral, mujoco.mjtObj)):
      raise TypeError("objtype must be an integer object type")
    kind = int(objtype)
    O = mujoco.mjtObj
    mapping = {int(O.mjOBJ_BODY): (self.model.nbody, None, "inertial"),
               int(O.mjOBJ_XBODY): (self.model.nbody, None, "body"),
               int(O.mjOBJ_GEOM): (self.model.ngeom, self.model.geom_bodyid, "geom"),
               int(O.mjOBJ_SITE): (self.model.nsite, self.model.site_bodyid, "site")}
    if kind in mapping:
      count, owners, prefix = mapping[kind]
      objid = self._id(objid, count, "objid")
      body = objid if owners is None else int(owners[objid])
      return body, poses[prefix+"_pos"][:, objid], _rotation(poses[prefix+"_quat"][:, objid])
    if kind != int(O.mjOBJ_CAMERA):
      raise ValueError("object type has no spatial frame")
    objid = self._id(objid, self.model.ncam, "objid")
    body = int(self.model.cam_bodyid[objid])
    bquat = poses["body_quat"][:, body]
    pos = poses["body_pos"][:, body] + _rotate(bquat, self._camera["cam_pos"][objid].expand(self.batch_size, -1))
    rot = _rotation(_qmul(bquat, self._camera["cam_quat"][objid].expand(self.batch_size, -1)))
    mode = int(self.model.cam_mode[objid])
    if mode in (1, 2):
      rot = self._camera["cam_mat0"][objid].reshape(3, 3).expand(self.batch_size, -1, -1)
      pos = (poses["body_pos"][:, body] + self._camera["cam_pos0"][objid]
             if mode == 1 else self.subtree_com(dynamics)[:, body] + self._camera["cam_poscom0"][objid])
    elif mode in (3, 4):
      target = int(self.model.cam_targetbodyid[objid])
      if target >= 0:
        look = poses["body_pos"][:, target] if mode == 3 else self.subtree_com(dynamics)[:, target]
        z = _unit(pos-look)
        up = torch.zeros_like(z)
        up[:, 2] = 1
        x = _unit(torch.cross(up, z, dim=-1))
        y = _unit(torch.cross(z, x, dim=-1))
        rot = torch.stack((x, y, z), -1)
    return body, pos, rot

  @staticmethod
  def _motion_at(motion, offset):
    return torch.cat((motion[:, :3], motion[:, 3:] +
                      torch.cross(motion[:, :3], offset, dim=-1)), -1)

  def object_velocity(self, dynamics, objtype, objid, local=False):
    body, pos, rotation = self.object_frame(dynamics, objtype, objid)
    if not int(self.model.body_weldid[body]):
      return torch.zeros((self.batch_size, 6), dtype=torch.float32, device=self.device)
    result = self._motion_at(dynamics["cvel"][:, body], pos-dynamics["root_com"][:, self._root[body]])
    if local:
      result = torch.matmul(rotation.transpose(-1, -2), result.reshape(-1, 2, 3).transpose(-1, -2)).transpose(-1, -2).reshape(-1, 6)
    return result

  def object_acceleration(self, dynamics, qvel, qacc, objtype, objid, local=False):
    body, pos, rotation = self.object_frame(dynamics, objtype, objid)
    self._tensor(qvel, (self.batch_size, self.nv), "qvel")
    self._tensor(qacc, (self.batch_size, self.nv), "qacc")
    if not int(self.model.body_weldid[body]):
      return torch.zeros((self.batch_size, 6), dtype=torch.float32, device=self.device)
    terms = dynamics["cdof_dot"]*qvel[..., None] + dynamics["cdof"]*qacc[..., None]
    cacc = torch.where(self._chain[body][None, :, None] != 0, terms, 0).sum(1)
    if self._gravity_enabled:
      cacc[:, 3:] -= self._gravity
    offset = pos-dynamics["root_com"][:, self._root[body]]
    result = self._motion_at(cacc, offset)
    velocity = self._motion_at(dynamics["cvel"][:, body], offset)
    result[:, 3:] += torch.cross(velocity[:, :3], velocity[:, 3:], dim=-1)
    if local:
      result = torch.matmul(rotation.transpose(-1, -2), result.reshape(-1, 2, 3).transpose(-1, -2)).transpose(-1, -2).reshape(-1, 6)
    return result

  def apply_ft(self, dynamics, force, torque, point, body):
    """Generalized wrench contribution Jp.T*force + Jr.T*torque."""
    force = self._tensor(force, (self.batch_size, 3), "force")
    torque = self._tensor(torque, (self.batch_size, 3), "torque")
    jp, jr = self.jac(dynamics, point, body)
    return torch.bmm(jp.transpose(-1, -2), force[..., None]).squeeze(-1) + \
        torch.bmm(jr.transpose(-1, -2), torque[..., None]).squeeze(-1)
