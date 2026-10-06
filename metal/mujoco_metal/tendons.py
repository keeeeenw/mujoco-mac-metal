# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.

"""Fixed-joint tendon passive forces, damping tangent, and armature matrix."""

from pathlib import Path

import mujoco
import numpy as np

_SHADER = Path(__file__).parent / "shaders" / "tendons.metal"
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1


def _frozen(value, dtype=np.float32):
  array = np.array(value, dtype=dtype, order="C", copy=True)
  if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
    raise ValueError("tendon constants must be finite and float32-representable")
  return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class FixedTendonModel:
  """CPU reference lowered from one pinned MuJoCo fixed-joint-tendon model.

  Tendon lengths use the MuJoCo 3.10 definition, ``sum(coef * qpos)``. Spatial
  wraps, tendon limits/friction, actuator-inherited armature/damping, and other
  passive subsystems are rejected explicitly. With ``spatial_ok=True`` (mixed
  pipeline use only), site-led spatial paths produce zero rows here instead of
  raising; the spatial stage owns those tendon ids.
  """

  def __init__(self, model, spatial_ok=False):
    if mujoco.__version__ != "3.10.0":
      raise RuntimeError(f"fixed tendon lowering requires MuJoCo 3.10.0; found {mujoco.__version__}")
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    self.nq, self.nv, self.ntendon = int(model.nq), int(model.nv), int(model.ntendon)
    if any(value < 0 or value > _INT32_MAX for value in (self.nq, self.nv, self.ntendon)):
      raise ValueError("tendon dimensions exceed int32")
    if self.ntendon * self.nq > _UINT32_MAX or self.ntendon * self.nv > _UINT32_MAX:
      raise ValueError("tendon model buffers exceed uint32 indexing")
    # R06: actuator-inherited tendon armature and damping fold into tendon fields below.
    if np.any(np.asarray(model.tendon_limited)) and not spatial_ok:
      raise ValueError("tendon limits are unsupported")
    if np.any(np.asarray(model.tendon_frictionloss) != 0) and not spatial_ok:
      raise ValueError("tendon friction loss is unsupported")
    if np.any(np.asarray(model.tendon_actfrclimited)) and not spatial_ok:
      raise ValueError("tendon actuator-force limits are unsupported")

    hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    wrap_joint = int(mujoco.mjtWrap.mjWRAP_JOINT)
    wrap_site = int(mujoco.mjtWrap.mjWRAP_SITE)
    wrap_pulley = int(mujoco.mjtWrap.mjWRAP_PULLEY)
    wrap_sphere = int(mujoco.mjtWrap.mjWRAP_SPHERE)
    wrap_cyl = int(mujoco.mjtWrap.mjWRAP_CYLINDER)
    length_map = np.zeros((self.ntendon, self.nq), dtype=np.float64)
    moment_map = np.zeros((self.ntendon, self.nv), dtype=np.float64)
    for tendon in range(self.ntendon):
      start, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
      if count <= 0:
        raise ValueError("tendons must contain at least one fixed joint wrap")
      first = int(model.wrap_type[start])
      if first != wrap_joint:
        if not spatial_ok:
          raise ValueError("only fixed tendons made of joint wraps are supported")
        # Spatial path: zero rows here; the spatial stage owns this tendon.
        for wrap in range(start, start+count):
          if int(model.wrap_type[wrap]) not in (
              wrap_site, wrap_pulley, wrap_sphere, wrap_cyl):
            raise ValueError("only fixed tendons made of joint wraps are supported")
        continue
      for wrap in range(start, start+count):
        if int(model.wrap_type[wrap]) != wrap_joint:
          raise ValueError("only fixed tendons made of joint wraps are supported")
        joint = int(model.wrap_objid[wrap])
        if joint < 0 or joint >= int(model.njnt) or int(model.jnt_type[joint]) not in (hinge, slide):
          raise ValueError("fixed tendon wraps must target hinge or slide joints")
        coefficient = float(model.wrap_prm[wrap])
        length_map[tendon, int(model.jnt_qposadr[joint])] += coefficient
        moment_map[tendon, int(model.jnt_dofadr[joint])] += coefficient

    self.length_map = _frozen(length_map)
    self.moment_map = _frozen(moment_map)
    self.stiffness = _frozen(model.tendon_stiffness)
    self.stiffnesspoly = _frozen(np.asarray(model.tendon_stiffnesspoly).reshape(self.ntendon, 2))
    from mujoco_metal.model import actuator_tendon_inheritance
    _t_arm, _t_damp, _t_dpoly = actuator_tendon_inheritance(model)
    self.damping = _frozen(np.asarray(model.tendon_damping, dtype=np.float64) + _t_damp)
    self.dampingpoly = _frozen(
        np.asarray(model.tendon_dampingpoly, dtype=np.float64).reshape(self.ntendon, 2) + _t_dpoly)
    self.spring_range = _frozen(np.asarray(model.tendon_lengthspring).reshape(self.ntendon, 2))
    self.armature = _frozen(np.asarray(model.tendon_armature, dtype=np.float64) + _t_arm)
    if np.any(self.armature < 0) or np.any(self.damping < 0):
      raise ValueError("tendon armature and damping must be nonnegative")
    self.disable_spring = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_SPRING))
    self.disable_damper = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))
    ancestor_mask = np.zeros((self.nv, self.nv), dtype=np.float32)
    for dof in range(self.nv):
      anc = dof
      while anc >= 0:
        ancestor_mask[dof, anc] = 1.0
        ancestor_mask[anc, dof] = 1.0
        anc = int(model.dof_parentid[anc])
    self.ancestor_mask = _frozen(ancestor_mask)

  def _state(self, qpos, qvel):
    qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != self.nq or qvel.shape != (qpos.shape[0], self.nv) or not qpos.shape[0]:
      raise ValueError(f"qpos/qvel must have shapes (B, {self.nq}) and (B, {self.nv})")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
      raise ValueError("qpos and qvel must be finite")
    return qpos, qvel

  def run(self, qpos, qvel):
    """Return ``(qfrc_tendon, damping_matrix, armature_matrix)`` on CPU."""
    qpos, qvel = self._state(qpos, qvel)
    batch = len(qpos)
    lengths = qpos @ self.length_map.T
    velocities = qvel @ self.moment_map.T
    force = np.zeros((batch, self.nv), dtype=np.float64)
    damping_matrix = np.zeros((batch, self.nv, self.nv), dtype=np.float64)
    armature_matrix = np.zeros((self.nv, self.nv), dtype=np.float64)
    for tendon in range(self.ntendon):
      jac = self.moment_map[tendon].astype(np.float64)
      if not self.disable_spring:
        displacement = np.where(
            lengths[:, tendon] > self.spring_range[tendon, 1],
            lengths[:, tendon] - self.spring_range[tendon, 1],
            np.where(
                lengths[:, tendon] < self.spring_range[tendon, 0],
                lengths[:, tendon] - self.spring_range[tendon, 0],
                0.0,
            ),
        )
        spring_coef = (
            self.stiffness[tendon]
            + self.stiffnesspoly[tendon, 0] * displacement
            + self.stiffnesspoly[tendon, 1] * displacement * displacement
        )
        spring_force = -displacement * spring_coef
      else:
        spring_force = np.zeros(batch, dtype=np.float64)
      if not self.disable_damper:
        velocity = velocities[:, tendon]
        speed = np.abs(velocity)
        damper_coef = (
            self.damping[tendon]
            + self.dampingpoly[tendon, 0] * speed
            + self.dampingpoly[tendon, 1] * velocity * velocity
        )
        damper_force = -velocity * damper_coef
        damper_tangent = (
            self.damping[tendon]
            + 2 * self.dampingpoly[tendon, 0] * speed
            + 3 * self.dampingpoly[tendon, 1] * velocity * velocity
        )
      else:
        damper_force = np.zeros(batch, dtype=np.float64)
        damper_tangent = np.zeros(batch, dtype=np.float64)
      force += (spring_force + damper_force)[:, None] * jac[None, :]
      damping_matrix += damper_tangent[:, None, None] * np.outer(jac, jac)[None, :, :]
      armature_matrix += self.armature[tendon] * np.outer(jac, jac)
    armature_matrix *= self.ancestor_mask
    return force, damping_matrix, armature_matrix


class MetalFixedTendonDynamics:
  """Native MPS fixed-tendon passive-force and rank-one dynamics stage."""

  def __init__(self, model, batch_size=1, spatial_ok=False, *,
               armature_storage="dense", velocity_derivative_layout=None):
    self._meta = FixedTendonModel(model, spatial_ok=spatial_ok)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
      raise ValueError("batch_size must be a positive integer")
    self.batch_size = batch_size
    if armature_storage not in ("dense", "external"):
      raise ValueError("armature_storage must be 'dense' or 'external'")
    self.armature_storage = armature_storage
    for value in (self._meta.nq, self._meta.nv, self._meta.ntendon, batch_size):
      if value > _INT32_MAX:
        raise ValueError("tendon dimensions exceed int32")
    if batch_size * max(self._meta.nq, self._meta.nv, self._meta.ntendon, 1) > _UINT32_MAX:
      raise ValueError("tendon workspace exceeds uint32 indexing")
    if batch_size * self._meta.nv * self._meta.nv > _UINT32_MAX:
      raise ValueError("tendon matrix workspace exceeds uint32 indexing")
    if batch_size * max(self._meta.ntendon, 1) > _INT32_MAX:
      raise ValueError("cached tendon lengths exceed int32 indexing")
    import torch
    self._torch = torch
    self._device = torch.device("mps")
    self._library = torch.mps.compile_shader(_SHADER.read_text())
    self._kernel = self._library.fixed_tendon_dynamics
    self._coo_damping_kernel = self._library.fixed_tendon_damping_coo
    def tensor(value, dtype=torch.float32):
      npdtype = np.int32 if dtype == torch.int32 else np.float32
      array = np.array(value, dtype=npdtype, order="C", copy=True)
      if array.dtype == np.float32 and not np.all(np.isfinite(array)):
        raise ValueError("tendon constants must be finite float32")
      if array.size == 0:
        array = np.zeros(1, dtype=npdtype)
      return torch.as_tensor(array, dtype=dtype, device=self._device)
    meta = self._meta
    self._length_map = tensor(meta.length_map.reshape(-1))
    self._moment_map = tensor(meta.moment_map.reshape(-1))
    self._stiffness = tensor(meta.stiffness)
    self._stiffnesspoly = tensor(meta.stiffnesspoly.reshape(-1))
    self._damping = tensor(meta.damping)
    self._dampingpoly = tensor(meta.dampingpoly.reshape(-1))
    self._spring_range = tensor(meta.spring_range.reshape(-1))
    self._armature = tensor(meta.armature)
    self._ancestor_mask = tensor(meta.ancestor_mask.reshape(-1))
    self._dims = tensor([meta.nq, meta.nv, meta.ntendon, batch_size,
                         int(meta.disable_spring), int(meta.disable_damper),
                         int(armature_storage == "dense"), 0] +
                        [1] * batch_size, torch.int32)
    self._velocity_derivative_layout = velocity_derivative_layout
    if velocity_derivative_layout is not None:
      if int(velocity_derivative_layout.diagonal_slots.size) != meta.nv:
        raise ValueError("velocity derivative layout nv does not match tendon model")
      from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
      _validate_workspace_index_capacity(
          batch_size,
          {"fixed_tendon.coo_dims": 5,
           "fixed_tendon.coo_edges": max(int(velocity_derivative_layout.edge_count), 1)},
          {"nv": meta.nv,
           "edges": int(velocity_derivative_layout.edge_count)})
      self._coo_damping_dims = tensor(
          [meta.nv, meta.ntendon, batch_size,
           int(velocity_derivative_layout.edge_count),
           int(meta.disable_damper)], torch.int32)
    else:
      self._coo_damping_dims = None
    self._velocity_dims = self._dims.clone()
    self._velocity_dims[7] = 1
    self._last_length = torch.zeros((batch_size, max(meta.ntendon, 1)),
                                    dtype=torch.float32, device=self._device)
    self._qfrc = torch.empty((batch_size, meta.nv), dtype=torch.float32, device=self._device)
    self._damping_matrix = torch.empty((batch_size, meta.nv, meta.nv), dtype=torch.float32, device=self._device)
    self._armature_matrix = (
        torch.empty((batch_size, meta.nv, meta.nv), dtype=torch.float32,
                    device=self._device)
        if armature_storage == "dense" else torch.zeros(1, dtype=torch.float32,
                                                         device=self._device))
    self._dummy = torch.zeros(1, dtype=torch.float32, device=self._device)

  def run_device(self, qpos, qvel, *, length_override=None, world_mask=None):
    """Return borrowed `(qfrc, damping matrix, armature matrix)` MPS tensors."""
    torch, meta = self._torch, self._meta
    for name, value, shape in (
        ("qpos", qpos, (self.batch_size, meta.nq)),
        ("qvel", qvel, (self.batch_size, meta.nv)),
    ):
      if not isinstance(value, torch.Tensor) or value.device.type != "mps" or value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32 MPS with shape {shape}")
    if length_override is not None:
      shape = (self.batch_size, max(meta.ntendon, 1))
      if (not isinstance(length_override, torch.Tensor)
          or tuple(length_override.shape) != shape
          or length_override.dtype != torch.float32
          or length_override.device.type != "mps"
          or not length_override.is_contiguous()):
        raise ValueError(f"cached tendon lengths must be contiguous float32 MPS {shape}")
    if world_mask is not None:
      if (not isinstance(world_mask, torch.Tensor)
          or world_mask.dtype != torch.int32
          or tuple(world_mask.shape) != (self.batch_size,)
          or world_mask.device.type != "mps" or not world_mask.is_contiguous()):
        raise ValueError("world_mask must be contiguous int32 MPS [batch_size]")
      self._dims[8:8 + self.batch_size].copy_(world_mask)
    else:
      self._dims[8:8 + self.batch_size].fill_(1)
    dims = self._dims if length_override is None else self._velocity_dims
    self._kernel(
        qpos.reshape(-1) if meta.nq else self._dummy,
        qvel.reshape(-1) if meta.nv else self._dummy,
        self._length_map, self._moment_map, self._stiffness,
        self._stiffnesspoly, self._damping, self._dampingpoly,
        self._spring_range, self._armature, dims,
        self._qfrc.reshape(-1) if meta.nv else self._dummy,
        self._damping_matrix.reshape(-1) if meta.nv else self._dummy,
        self._armature_matrix.reshape(-1) if meta.nv else self._dummy,
        self._ancestor_mask,
        length_override.reshape(-1) if length_override is not None else self._dummy,
        self._last_length.reshape(-1),
        threads=(self.batch_size,), group_size=(1,),
    )
    armature = (self._armature_matrix if self.armature_storage == "dense"
                else None)
    return self._qfrc, self._damping_matrix, armature

  def run_damping_derivative_coo_device(self, qvel, edge_writer):
    """Accumulate fixed-tendon qDeriv directly into compiled COO slots.

    The physical damping matrix remains available from ``run_device`` for
    legacy dense consumers. Sparse implicit assembly uses this producer and
    avoids allocating or gathering a global nv-by-nv tendon derivative.
    """
    torch, meta = self._torch, self._meta
    if self._velocity_derivative_layout is None:
      raise ValueError("compiled velocity derivative layout is required for COO tendons")
    expected = (self.batch_size, meta.nv)
    if (not isinstance(qvel, torch.Tensor) or qvel.dtype != torch.float32
        or qvel.device.type != "mps" or tuple(qvel.shape) != expected
        or not qvel.is_contiguous()):
      raise ValueError(f"qvel must be contiguous float32 MPS {expected}")
    edge_count = int(self._velocity_derivative_layout.edge_count)
    if (getattr(edge_writer, "nv", None) != meta.nv
        or getattr(edge_writer, "edge_count", None) != edge_count
        or tuple(edge_writer.values.shape) !=
        (self.batch_size, max(edge_count, 1))):
      raise ValueError("edge_writer does not match fixed-tendon compiled COO layout")
    if edge_count:
      self._coo_damping_kernel(
          qvel.reshape(-1), self._moment_map, self._damping,
          self._dampingpoly, edge_writer._edge_rows,
          edge_writer._edge_cols, self._coo_damping_dims,
          edge_writer.values.reshape(-1),
          threads=(self.batch_size * edge_count,), group_size=(1,))
    return edge_writer.values

  @property
  def fixed_jacobian_template(self):
    """Borrow the immutable ``[ntendon,nv]`` fixed-tendon moment map.

    Spatial tendon rows are zero in this table and are supplied by the spatial
    kinematics stage. Callers building one canonical runtime ``tendon_J`` may
    copy this map into each world before overlaying spatial rows.
    """
    return self._moment_map
