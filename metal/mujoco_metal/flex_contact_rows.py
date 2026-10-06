# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0.

"""Native canonical row assembly for fixed-slot MuJoCo flex contacts."""

from pathlib import Path

import numpy as np
import mujoco

_SHADER = Path(__file__).parent / "shaders" / "flex_contact_rows.metal"


class FlexContactRows:
  """Assemble MuJoCo contact Jacobian rows and cached position context.

  The contact candidate descriptor fixes each slot's row span at lowering
  time. Runtime narrowphase only supplies its per-world frame, distance and
  live mask. ``diagA`` is the solver's row diagonal approximation, in the
  same canonical row order as ``descriptor.row_start``.
  """

  _CONTEXT_FIELDS = ("K", "B", "imp", "pos", "margin")
  _INT32_MAX = np.iinfo(np.int32).max

  def __init__(self, model, descriptor, batch_size=1, device="mps",
               sparse_only=False):
    # Check all host metadata before importing/allocating through Torch.  The
    # Metal kernels use signed 32-bit offsets, so silent NumPy int32 wrap here
    # would corrupt otherwise valid buffers.
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
      raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    nv = int(descriptor.nv)
    nslot = int(descriptor.slot_count)
    nrow = int(descriptor.row_capacity)
    if min(nv, nslot, nrow) < 0 or max(nv, nslot, nrow, batch_size) > self._INT32_MAX:
      raise ValueError("flex contact dimensions must fit signed 32-bit indexing")
    if batch_size * max(nrow, 1) * max(nv, 1) > self._INT32_MAX:
      raise ValueError("flex contact row workspace exceeds signed 32-bit indexing")
    if batch_size * max(nslot, 1) * 6 * max(nv, 1) > self._INT32_MAX:
      raise ValueError("flex contact spatial workspace exceeds signed 32-bit indexing")
    self._validate_descriptor(descriptor, nslot, nrow, nv)
    try:
      import torch
    except ImportError as exc:
      raise RuntimeError("flex contact row assembly requires PyTorch") from exc
    if device != "mps":
      raise ValueError("flex contact row assembly requires the native MPS device")
    self._torch = torch
    self._model = model
    self._descriptor = descriptor
    self.batch_size = int(batch_size)
    self.device = torch.device(device)
    self.nv = nv
    self.nslot = nslot
    self.nrow = nrow
    self.sparse_only = bool(sparse_only)

    def static(value, dtype):
      return torch.as_tensor(np.array(value, copy=True), dtype=dtype,
                             device=self.device).contiguous()

    self._row_start = static(descriptor.row_start, torch.int32)
    self._row_span = static(descriptor.row_span, torch.int32)
    self._condim = static(descriptor.condim, torch.int32)
    self._cone = static(descriptor.cone, torch.int32)
    self._friction = static(descriptor.friction, torch.float32)
    self._solref = static(descriptor.solref, torch.float32)
    # MuJoCo 3.10 flex/geom and flex/flex generated contacts use zero
    # solreffriction (the collision driver initializes it to {0, 0}).
    self._solreffriction = torch.zeros(
        (self.batch_size, self.nslot, 2), dtype=torch.float32,
        device=self.device)
    self._solimp = static(descriptor.solimp, torch.float32)
    self._margin = static(descriptor.margin, torch.float32)
    self._active_i32 = torch.empty(
        (self.batch_size, self.nslot), dtype=torch.int32, device=self.device)
    # Recovery masks are copied into the dimension record suffix so the
    # shader can return before reading a masked world's contact inputs.
    self._world_mask = torch.ones(
        (self.batch_size,), dtype=torch.int32, device=self.device)
    self._surface_velocity_input = torch.zeros(
        (self.batch_size, self.nslot, 6), dtype=torch.float32,
        device=self.device)
    self._assemble_dims = torch.tensor(
        [self.batch_size, self.nslot, self.nv, self.nrow, 0, 0, 0]
        + [1] * self.batch_size,
        dtype=torch.int32, device=self.device)
    self._refresh_dims = torch.tensor(
        [self.batch_size, self.nrow, self.nv, 0, 0] + [1] * self.batch_size,
        dtype=torch.int32, device=self.device)
    self._params = torch.tensor(
        [float(model.opt.timestep), float(model.opt.impratio),
         float(not (int(model.opt.disableflags)
                    & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)))],
        dtype=torch.float32, device=self.device)
    shape = (self.batch_size, self.nrow)
    # Keep a one-float backing stride for nv==0: MPS custom-shader bindings
    # must never receive a zero-byte buffer.  The public zero-width view is
    # created once and remains borrowed alongside the other workspaces.
    self._storage_nv = max(self.nv, 1)
    self._spatial_J_input = (torch.zeros(1, dtype=torch.float32,
                                          device=self.device)
                             if self.sparse_only else torch.zeros(
        (self.batch_size, self.nslot, 6, self._storage_nv),
        dtype=torch.float32, device=self.device))
    self._qvel_input = torch.zeros(
        (self.batch_size, self._storage_nv), dtype=torch.float32,
        device=self.device)
    self.workspace_J = (torch.zeros(1, dtype=torch.float32, device=self.device)
                        if self.sparse_only else torch.empty(
                            (*shape, self._storage_nv), dtype=torch.float32,
                            device=self.device))
    self._workspace_J_public = (None if self.sparse_only else
                                self.workspace_J[..., :self.nv])
    self._packed_jacobian = None
    self._packed_row_offset = 0
    self.R = torch.empty(shape, dtype=torch.float32, device=self.device)
    self.aref = torch.empty_like(self.R)
    self.lo = torch.empty_like(self.R)
    self.hi = torch.empty_like(self.R)
    self.active = torch.empty(shape, dtype=torch.int32, device=self.device)
    self.position_context = torch.empty(
        (*shape, len(self._CONTEXT_FIELDS)), dtype=torch.float32,
        device=self.device)
    self.row_owner = torch.empty(shape, dtype=torch.int32, device=self.device)
    self.row_local = torch.empty_like(self.row_owner)
    self.row_cone = torch.empty_like(self.row_owner)
    self.row_friction = torch.empty(
        (*shape, 5), dtype=torch.float32, device=self.device)
    self.row_surface_velocity = torch.empty(
        shape, dtype=torch.float32, device=self.device)
    self._shader = torch.mps.compile_shader(_SHADER.read_text()) if self.nslot else None
    self._has_assembly = False

  @classmethod
  def _validate_descriptor(cls, descriptor, nslot, nrow, nv):
    """Validate immutable slot identity and complete, disjoint row coverage."""
    vector_names = ("kind", "flex1", "elem1", "vert1", "flex2", "elem2",
                    "vert2", "geom", "row_start", "row_span", "condim",
                    "cone", "margin", "gap", "contact_ordinal")
    for name in vector_names:
      arr = np.asarray(getattr(descriptor, name))
      if arr.ndim != 1 or arr.size != nslot:
        raise ValueError(f"descriptor.{name} must have shape ({nslot},)")
    material_shapes = {"friction": (nslot, 5), "solref": (nslot, 2),
                       "solimp": (nslot, 5), "margin": (nslot,),
                       "gap": (nslot,)}
    for name, shape in material_shapes.items():
      if np.asarray(getattr(descriptor, name)).shape != shape:
        raise ValueError(f"descriptor.{name} must have shape {shape}")
    start = np.asarray(descriptor.row_start, dtype=np.int64)
    span = np.asarray(descriptor.row_span, dtype=np.int64)
    condim = np.asarray(descriptor.condim, dtype=np.int64)
    cone = np.asarray(descriptor.cone, dtype=np.int64)
    if np.any(start < 0) or np.any(span <= 0):
      if nslot:
        raise ValueError("contact row starts must be nonnegative and spans positive")
    if np.any(start > cls._INT32_MAX) or np.any(span > cls._INT32_MAX):
      raise ValueError("contact row addresses must fit signed 32-bit indexing")
    if np.any(~np.isin(condim, (1, 3, 4, 6))):
      raise ValueError("flex contacts require condim 1, 3, 4, or 6")
    if np.any(~np.isin(cone, (0, 1))):
      raise ValueError("flex contacts require an elliptic or pyramidal cone")
    # MuJoCo's enum values are pyramidal=0, elliptic=1.
    expected = np.where((cone == 1) | (condim == 1), condim,
                        2 * (condim - 1))
    if np.any(span != expected):
      raise ValueError("contact row span does not match condim and cone")
    if nslot:
      order = np.argsort(start, kind="stable")
      cursor = 0
      for slot in order:
        if int(start[slot]) != cursor:
          raise ValueError("contact row spans must be disjoint and cover rows contiguously")
        cursor += int(span[slot])
      if cursor != nrow:
        raise ValueError("contact row spans do not cover row_capacity")
    elif nrow:
      raise ValueError("an empty contact descriptor must have zero row capacity")
    if int(getattr(descriptor, "feature_capacity", nslot)) < nslot:
      raise ValueError("contact slot count exceeds feature capacity")
    if nv > cls._INT32_MAX:
      raise ValueError("nv exceeds signed 32-bit indexing")

  def _validate(self, name, value, shape, dtype):
    torch = self._torch
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
        or value.dtype != dtype or value.device.type != "mps"
        or not value.is_contiguous()):
      raise ValueError(f"{name} must be contiguous MPS {dtype} with shape {shape}")

  def run_device(self, contact_result, relative_spatial_jacobian, qvel,
                 diagA, relative_surface_velocity=None, *,
                 packed_jacobian=None, canonical_row_offset=0,
                 world_mask=None):
    """Write ``J, R, aref, bounds, active`` and replayable position context.

    ``relative_spatial_jacobian`` is `[B,slot,6,nv]` in world coordinates;
    its first three rows are relative point translation and its last three
    rows are relative angular velocity. ``relative_surface_velocity`` is an
    optional externally prescribed relative spatial velocity `[B,slot,6]`;
    its row projection is kept separately so it can be replayed at later
    velocity stages. ``diagA`` is `[B,row_capacity]`. A zero in the optional
    int32 ``world_mask`` preserves that world's row and packed-J state.
    Returned tensors are borrowed workspaces overwritten by the next call.
    """
    torch = self._torch
    b, s, nv, nr = self.batch_size, self.nslot, self.nv, self.nrow
    if world_mask is None:
      self._world_mask.fill_(1)
    else:
      self._validate("world_mask", world_mask, (b,), torch.int32)
      self._world_mask.copy_(world_mask)
    self._assemble_dims[7:].copy_(self._world_mask)
    sparse = packed_jacobian is not None
    canonical_row_offset = int(canonical_row_offset)
    if canonical_row_offset < 0 or canonical_row_offset > self._INT32_MAX:
      raise ValueError("canonical row offset must fit signed int32")
    if sparse:
      if (not isinstance(packed_jacobian, torch.Tensor)
          or packed_jacobian.ndim != 1 or packed_jacobian.numel() < 12
          or packed_jacobian.dtype != torch.float32
          or packed_jacobian.device.type != "mps"
          or not packed_jacobian.is_contiguous()):
        raise ValueError("packed_jacobian must be a flat contiguous MPS float32 record")
      self._packed_jacobian = packed_jacobian
      self._packed_row_offset = canonical_row_offset
    elif self.sparse_only:
      raise ValueError("sparse-only flex rows require a packed Jacobian target")
    else:
      self._packed_jacobian = None
      self._packed_row_offset = 0
    self._assemble_dims[4] = 1 if sparse else 0
    self._assemble_dims[5] = canonical_row_offset if sparse else 0
    prewritten_packed = bool(sparse and relative_spatial_jacobian is None)
    if relative_spatial_jacobian is None:
      if not sparse:
        raise ValueError("relative_spatial_jacobian is required for dense flex rows")
    else:
      self._validate("relative_spatial_jacobian", relative_spatial_jacobian,
                     (b, s, 6, nv), torch.float32)
    self._assemble_dims[6] = 1 if prewritten_packed else 0
    self._validate("qvel", qvel, (b, nv), torch.float32)
    self._validate("diagA", diagA, (b, nr), torch.float32)
    dist = contact_result.get("dist")
    active = contact_result.get("active")
    frame = contact_result.get("frame")
    self._validate("contact_result.dist", dist, (b, s), torch.float32)
    self._validate("contact_result.active", active, (b, s), torch.bool)
    self._validate("contact_result.frame", frame, (b, s, 3, 3), torch.float32)
    solreffriction = contact_result.get("solreffriction")
    if solreffriction is None:
      self._solreffriction.zero_()
    else:
      self._validate("contact_result.solreffriction", solreffriction,
                     (b, s, 2), torch.float32)
      self._solreffriction.copy_(solreffriction)
    if relative_surface_velocity is None:
      self._surface_velocity_input.zero_()
    else:
      self._validate("relative_surface_velocity", relative_surface_velocity,
                     (b, s, 6), torch.float32)
      self._surface_velocity_input.copy_(relative_surface_velocity)
    if nr == 0:
      self._has_assembly = True
      return self._result()
    self._active_i32.copy_(active)
    if self._shader is None:
      raise RuntimeError("flex row shader is unavailable for nonempty contacts")
    if s:
      if nv:
        if not prewritten_packed:
          self._spatial_J_input.copy_(relative_spatial_jacobian)
        self._qvel_input.copy_(qvel)
      else:
        self._spatial_J_input.zero_()
        self._qvel_input.zero_()
      self._shader.flex_contact_assemble_rows(
          (self._spatial_J_input.reshape(-1) if not prewritten_packed
           else self._packed_jacobian[:1]), frame.reshape(-1),
          dist.reshape(-1), self._active_i32.reshape(-1),
          self._row_start, self._row_span, self._condim, self._cone,
          self._friction.reshape(-1), self._solref.reshape(-1),
          self._solreffriction.reshape(-1), self._solimp.reshape(-1),
          self._margin, diagA.reshape(-1), self._qvel_input.reshape(-1),
          self._assemble_dims, self._params,
          (packed_jacobian if sparse else self.workspace_J).reshape(-1), self.R.reshape(-1),
          self.aref.reshape(-1), self.lo.reshape(-1), self.hi.reshape(-1),
          self.active.reshape(-1), self.position_context.reshape(-1),
          self.row_owner.reshape(-1), self.row_local.reshape(-1),
          self.row_cone.reshape(-1), self.row_friction.reshape(-1),
          self._surface_velocity_input.reshape(-1),
          self.row_surface_velocity.reshape(-1),
          threads=(b*s,), group_size=(128,))
    self._has_assembly = True
    return self._result()

  def refresh_aref(self, qvel, world_mask=None):
    """Refresh reference accelerations from cached J/context after qvel changes."""
    b, nr, nv = self.batch_size, self.nrow, self.nv
    if world_mask is None:
      self._world_mask.fill_(1)
    else:
      self._validate("world_mask", world_mask, (b,), self._torch.int32)
      self._world_mask.copy_(world_mask)
    self._refresh_dims[5:].copy_(self._world_mask)
    self._validate("qvel", qvel, (b, nv), self._torch.float32)
    if not self._has_assembly:
      raise RuntimeError("contact rows must be assembled before refreshing aref")
    if nr:
      if nv:
        self._qvel_input.copy_(qvel)
      else:
        self._qvel_input.zero_()
      self._refresh_dims[3] = 1 if self._packed_jacobian is not None else 0
      self._refresh_dims[4] = self._packed_row_offset
      self._shader.flex_contact_refresh_aref(
          (self._packed_jacobian if self._packed_jacobian is not None
           else self.workspace_J).reshape(-1), self._qvel_input.reshape(-1),
          self.position_context.reshape(-1), self.active.reshape(-1),
          self._refresh_dims, self.row_surface_velocity.reshape(-1),
          self.aref.reshape(-1),
          threads=(b*nr,), group_size=(128,))
    return self.aref

  def _result(self):
    result = {
        "R": self.R,
        "aref": self.aref,
        "lo": self.lo,
        "hi": self.hi,
        "active": self.active,
        "position_context": self.position_context,
        # Static, slot-relative logical layout. This is the source of truth
        # for a parent allocator reserving the complete flex row block.
        "row_start": self._row_start,
        "row_span": self._row_span,
        "row_owner": self.row_owner,
        "row_local": self.row_local,
        "row_cone": self.row_cone,
        "row_friction": self.row_friction,
        "surface_velocity": self.row_surface_velocity,
    }
    if self._packed_jacobian is None:
      result["workspace_J"] = self._workspace_J_public
    else:
      result["jacobian_packed"] = self._packed_jacobian
      result["canonical_row_offset"] = self._packed_row_offset
    return result
