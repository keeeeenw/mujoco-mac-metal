# Copyright 2026 The MuJoCo Metal contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All-device contact-free semi-implicit Euler simulation orchestration."""

import numbers

import numpy as np

import mujoco

from mujoco_metal.device_state import DeviceState
from mujoco_metal.integration import MetalEulerIntegration
from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
from mujoco_metal.smooth_metal import MetalSmoothDynamics
from mujoco_metal.smooth_solve import MetalDenseSolve
from mujoco_metal.stepping import validate_stepping_profile


class MetalSimulation:
  """Batched native MPS stepping for ``contact_free_euler_v1`` models.

  The profile currently assumes zero applied generalized force. It uses native
  smooth dynamics, a dense SPD solve and semi-implicit Euler; unsupported
  model features are rejected during construction before MPS initialization.
  ``state`` owns persistent qpos, qvel, qacc, time and status. Its reset and
  restore operations replace state tensors safely; every step reads the current
  tensors, so those operations do not leave stale cached references.
  A failed row keeps its first nonzero status and does not advance again until
  that row is reset or the state is restored.

  ``step`` returns a borrowed MPS int32 status tensor, valid until the next
  step call. Read or copy it before stepping again if it must be retained.
  State properties return detached copies, and snapshots perform host
  readback; use them outside the hot stepping loop.
  """

  def __init__(
      self,
      model,
      batch_size=1,
      qpos=None,
      qvel=None,
      *,
      profile="contact_free_euler_v1",
  ):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    if isinstance(batch_size, bool) or not isinstance(
        batch_size, numbers.Integral
    ):
      raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    batch_size = int(batch_size)
    # This CPU-only contract check must finish before any constructor can
    # initialize MPS or compile a shader.
    profile = validate_stepping_profile(model, profile=profile)
    _validate_workspace_index_capacity(
        batch_size,
        {
            "qpos": batch_size * profile.nq,
            "qvel": batch_size * profile.nv,
            "mass": batch_size * profile.nv * profile.nv,
            "body_pos": batch_size * model.nbody * 3,
            "body_quat": batch_size * model.nbody * 4,
            "geom_pos": batch_size * model.ngeom * 3,
            "geom_quat": batch_size * model.ngeom * 4,
            "site_pos": batch_size * model.nsite * 3,
            "site_quat": batch_size * model.nsite * 4,
            "inertial_pos": batch_size * model.nbody * 3,
            "inertial_quat": batch_size * model.nbody * 4,
            "joint_anchor": batch_size * model.njnt * 3,
            "joint_axis": batch_size * model.njnt * 3,
            "root_com": batch_size * model.nbody * 3,
            "cdof": batch_size * profile.nv * 6,
            "crb": batch_size * model.nbody * 36,
            "local_inertia": batch_size * model.nbody * 36,
            "cvel": batch_size * model.nbody * 6,
            "cdof_dot": batch_size * profile.nv * 6,
            "cacc": batch_size * model.nbody * 6,
            "body_force": batch_size * model.nbody * 6,
        },
        {
            "nq": profile.nq,
            "nv": profile.nv,
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
        },
    )
    self._state = DeviceState(model, profile, batch_size, qpos=qpos, qvel=qvel)
    descriptor = self._state._model
    self.profile = profile
    self._smooth = MetalSmoothDynamics(descriptor, batch_size=batch_size)
    self._solver = MetalDenseSolve(descriptor.nv, batch_size)
    self._euler_solver = (
        MetalDenseSolve(descriptor.nv, batch_size)
        if profile.implicit_euler_damping
        else None
    )
    self._integrator = MetalEulerIntegration(
        descriptor, batch_size, profile.timestep
    )

    torch = self._state._torch
    self._rhs = torch.empty(
        (batch_size, descriptor.nv),
        dtype=torch.float32,
        device=self._state._device,
    )
    self._applied_force = torch.zeros_like(self._rhs)
    self._damping = torch.tensor(
        descriptor.dof_damping
        if profile.passive_damping_enabled
        else np.zeros(descriptor.nv),
        dtype=torch.float32,
        device=self._state._device,
    )
    self._effective_mass = (
        torch.empty(
            (batch_size, descriptor.nv, descriptor.nv),
            dtype=torch.float32,
            device=self._state._device,
        )
        if profile.implicit_euler_damping
        else None
    )
    self._implicit_damping = (
        self.profile.timestep * self._damping
        if profile.implicit_euler_damping
        else None
    )
    self._success = torch.empty(
        (batch_size,), dtype=torch.bool, device=self._state._device
    )
    self._combined_status = torch.empty_like(self._state._status)
    self._next_qpos = torch.empty_like(self._state._qpos)
    self._next_qvel = torch.empty_like(self._state._qvel)
    self._next_qacc = torch.empty_like(self._state._qacc)
    self._next_time = torch.empty_like(self._state._time)
    self._next_status = torch.empty_like(self._state._status)

  @property
  def state(self):
    """The owned :class:`DeviceState` lifecycle and checkpoint interface."""
    return self._state

  def _prepare_force(self, qfrc_applied):
    torch = self._state._torch
    shape = (self._state.batch_size, self._state._model.nv)
    if qfrc_applied is None:
      if self.profile.name == "contact_free_forces_euler_v1":
        self._applied_force.zero_()
      return
    if self.profile.name == "contact_free_euler_v1":
      raise ValueError("qfrc_applied requires contact_free_forces_euler_v1")
    if isinstance(qfrc_applied, torch.Tensor):
      if qfrc_applied.dtype != torch.float32:
        raise TypeError("qfrc_applied tensor must have dtype torch.float32")
      if tuple(qfrc_applied.shape) != shape:
        raise ValueError(f"qfrc_applied must have shape {shape}")
      if qfrc_applied.device.type != "mps":
        raise ValueError("qfrc_applied tensor must be on MPS")
      if not qfrc_applied.is_contiguous():
        raise ValueError("qfrc_applied tensor must be contiguous")
      self._applied_force.copy_(qfrc_applied)
      return
    array = np.asarray(qfrc_applied)
    if array.shape != shape or array.dtype.kind not in "fiu":
      raise ValueError(f"qfrc_applied must be numeric with shape {shape}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      array = np.asarray(array, dtype=np.float32, order="C")
    if not np.all(np.isfinite(array)):
      raise ValueError(
          "qfrc_applied must be finite and representable as float32"
      )
    self._applied_force.copy_(torch.tensor(array, device=self._state._device))

  def step(self, steps=1, *, qfrc_applied=None, ctrl=None):
    """Advance all worlds by a positive number of native contact-free steps.

    The fixed workspace is reused. Failed worlds retain their previous state
    and acceleration while status records the failure. Failure remains sticky
    until reset or restore. No CPU physics, solve, integration, or readback
    occurs in this method.
    """
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral):
      raise TypeError("steps must be a positive integer")
    if steps <= 0:
      raise ValueError("steps must be a positive integer")
    if ctrl is not None:
      raise ValueError(
          "ctrl is unsupported until scalar motor lowering is qualified"
      )
    self._prepare_force(qfrc_applied)
    torch = self._state._torch
    state = self._state
    for _ in range(int(steps)):
      dynamics = self._smooth.run_device(state._qpos, state._qvel)
      torch.neg(dynamics["qfrc_bias"], out=self._rhs)
      if self.profile.name == "contact_free_forces_euler_v1":
        if self.profile.passive_damping_enabled and self._rhs.numel():
          self._rhs.addcmul_(
              state._qvel, self._damping.unsqueeze(0), value=-1.0
          )
        self._rhs.add_(self._applied_force)
      acceleration, solve_status = self._solver.run_device(
          dynamics["mass_matrix"], self._rhs
      )
      integration_acceleration = acceleration
      if self._euler_solver is not None:
        self._effective_mass.copy_(dynamics["mass_matrix"])
        self._effective_mass.diagonal(dim1=1, dim2=2).add_(
            self._implicit_damping
        )
        integration_acceleration, euler_status = self._euler_solver.run_device(
            self._effective_mass, self._rhs
        )
        torch.where(
            (solve_status == 0),
            euler_status,
            solve_status,
            out=self._combined_status,
        )
        solve_status = self._combined_status
      torch.eq(state._status, 0, out=self._success)
      torch.where(
          self._success, solve_status, state._status, out=self._combined_status
      )
      qpos, qvel, time, status = self._integrator.run_device(
          state._qpos,
          state._qvel,
          integration_acceleration,
          state._time,
          self._combined_status,
      )
      torch.eq(status, 0, out=self._success)
      torch.where(
          self._success.unsqueeze(1),
          acceleration,
          state._qacc,
          out=self._next_qacc,
      )
      self._next_qpos.copy_(qpos)
      self._next_qvel.copy_(qvel)
      self._next_time.copy_(time)
      self._next_status.copy_(status)

      # Ping-pong owned tensors keep live state disjoint from borrowed stage
      # outputs and make each new current-state reference safe for the next
      # invocation. DeviceState reset/restore may replace these references.
      state._qpos, self._next_qpos = self._next_qpos, state._qpos
      state._qvel, self._next_qvel = self._next_qvel, state._qvel
      state._qacc, self._next_qacc = self._next_qacc, state._qacc
      state._time, self._next_time = self._next_time, state._time
      state._status, self._next_status = self._next_status, state._status
      state._generation += 1
    return state._status
