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

import mujoco
import numpy as np
import torch


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
    self._body_mass = tensor(model.body_mass)
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

  def _id(self, value, count, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
      raise TypeError(f"{name} must be an integer")
    if not 0 <= int(value) < count:
      raise ValueError(f"{name} is outside the compiled model")
    return int(value)

  def _tensor(self, value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.device != self.device
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

  def subtree_com(self, dynamics):
    positions = dynamics["poses"]["inertial_pos"]
    weighted = positions*self._body_mass[None, :, None]
    for body in range(self.model.nbody-1, 0, -1):
      weighted[:, self._parent[body]].add_(weighted[:, body])
    return torch.where(self._subtree_mass[None, :, None] > 1e-15,
        weighted/self._subtree_mass[None, :, None].clamp_min(1e-15),
        dynamics["poses"]["body_pos"])

  def jac_subtree_com(self, dynamics, body):
    body = self._id(body, self.model.nbody, "body")
    output = torch.zeros((self.batch_size, 3, self.nv), dtype=torch.float32, device=self.device)
    for child in range(body, self.model.nbody):
      if child > body and int(self.model.body_parentid[child]) < body:
        break
      jacp, _ = self.jac(dynamics, dynamics["poses"]["inertial_pos"][:, child], child)
      output.add_(jacp*self._body_mass[child])
    return output/self._subtree_mass[body]

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
