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

from dataclasses import replace

import mujoco

from mujoco_metal.device_state import DeviceState
from mujoco_metal.actuation import MetalScalarMotorForce
from mujoco_metal.actuation import ScalarMotorModel
from mujoco_metal.integration import MetalEulerIntegration
from mujoco_metal.metal_kinematics import _validate_workspace_index_capacity
from mujoco_metal.smooth_metal import MetalSmoothDynamics
from mujoco_metal.smooth_solve import MetalDenseSolve
from mujoco_metal.stepping import validate_stepping_profile


class MetalSimulation:
  """Batched native MPS stepping for explicit contact-free Euler profiles.

  The default ``contact_free_euler_v1`` profile uses native smooth dynamics,
  a dense SPD solve and semi-implicit Euler with no applied forces. The opt-in
  ``contact_free_forces_euler_v1`` profile adds per-call generalized force and
  linear joint damping. ``contact_free_motor_euler_v1`` additionally supports
  the bounded scalar joint motor subset. Unsupported model features are
  rejected during construction before MPS initialization.
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
    with_transmissions = "transmission" in profile.name
    motor_model = (
        ScalarMotorModel.from_model(model)
        if profile.name.replace("rk4", "euler")
        in (
            "contact_free_motor_euler_v1",
            "joint_constraints_euler_v1",
            "contact_free_fluid_euler_v1",
            "contact_free_implicitfast_v1",
            "contact_free_passive_euler_v1",
            "contact_free_sensor_euler_v1",
            "normal_contact_euler_v1",
            "friction_contact_euler_v1",
        )
        else None
    )
    if motor_model is not None:
      maximum = (2**31 - 1) // max(1, motor_model.nv, motor_model.nu)
      if batch_size > maximum:
        raise ValueError(
            "batch exceeds scalar motor int32 kernel index capacity"
        )
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
            "ctrl": batch_size * model.nu,
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
    smooth_descriptor = descriptor
    if with_transmissions:
      # Fixed tendon armature is assembled separately as J.T @ armature @ J.
      smooth_descriptor = replace(
          descriptor, tendon_armature=np.zeros_like(descriptor.tendon_armature)
      )
    self._smooth = MetalSmoothDynamics(
        smooth_descriptor,
        batch_size=batch_size,
    )
    self._motor = (
        MetalScalarMotorForce(motor_model, batch_size=batch_size)
        if motor_model is not None
        else None
    )
    self._transmissions = None
    if with_transmissions:
      from mujoco_metal.transmissions import MetalTransmissions

      self._transmissions = MetalTransmissions(model)
    self._tendons = None
    self._tendon_damping = None
    if with_transmissions and model.ntendon:
      from mujoco_metal.tendons import MetalFixedTendonDynamics

      self._tendons = MetalFixedTendonDynamics(model, batch_size)
    self._passive = None
    self._damping_tangent = None
    if (
        with_transmissions
        or "joint_constraints" in profile.name
        or "fluid" in profile.name
        or "implicitfast" in profile.name
        or "passive" in profile.name
        or "sensor" in profile.name
        or profile.name
        in ("normal_contact_euler_v1", "friction_contact_euler_v1")
    ):
      from mujoco_metal.passive import MetalPassiveForces

      self._passive = MetalPassiveForces(model)
    self._fluid = None
    if "fluid" in profile.name:
      from mujoco_metal.fluid import MetalInertiaBoxFluid

      self._fluid = MetalInertiaBoxFluid(model, batch_size)
    self._sensors = None
    self._sensordata = None
    if "sensor" in profile.name:
      from mujoco_metal.sensors import SensorProgram

      self._sensors = SensorProgram(model, batch_size)
    self._joint_constraints = None
    if profile.name == "joint_constraints_euler_v1":
      from mujoco_metal.joint_constraints import JointConstraintProgram

      self._joint_constraints = JointConstraintProgram(model, batch_size)
    self._contact = None
    if profile.name in (
        "normal_contact_euler_v1",
        "friction_contact_euler_v1",
    ) and not (
        int(model.opt.disableflags)
        & (
            int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        )
    ):
      from mujoco_metal.contact import MetalContact

      self._contact = MetalContact(model, batch_size)
    self._solver = MetalDenseSolve(descriptor.nv, batch_size)
    self._implicitfast = None
    if profile.name == "contact_free_implicitfast_v1":
      from mujoco_metal.implicit import ImplicitFastProgram

      self._implicitfast = ImplicitFastProgram(model, batch_size)
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
    self._applied_force = (
        torch.zeros_like(self._rhs)
        if profile.name.replace("rk4", "euler")
        in (
            "contact_free_forces_euler_v1",
            "contact_free_motor_euler_v1",
            "contact_free_transmission_euler_v1",
            "joint_constraints_euler_v1",
            "contact_free_fluid_euler_v1",
            "contact_free_implicitfast_v1",
            "contact_free_passive_euler_v1",
            "contact_free_sensor_euler_v1",
            "normal_contact_euler_v1",
            "friction_contact_euler_v1",
        )
        else None
    )
    self._body_wrench = (
        torch.zeros(
            (batch_size, int(model.nbody), 6),
            dtype=torch.float32,
            device=self._state._device,
        )
        if self._passive is not None
        else None
    )
    self._control = (
        torch.zeros(
            (batch_size, model.nu),
            dtype=torch.float32,
            device=self._state._device,
        )
        if motor_model is not None or with_transmissions
        else None
    )
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
    if "rk4" in profile.name:
      from mujoco_metal.runge_kutta import MetalRungeKutta

      self._rk4 = MetalRungeKutta(descriptor, batch_size, profile.timestep)
    else:
      self._rk4 = None

  @property
  def state(self):
    """The owned :class:`DeviceState` lifecycle and checkpoint interface."""
    return self._state

  def sensor_values(self):
    """Evaluate supported stateless sensors at CURRENT state on MPS.

    This is an explicit forward-stage query, not the pre-integration sample
    left by MuJoCo mj_step. Re-evaluation makes reset/restore immediately visible.
    Returned values are detached copies owned by the caller.
    """
    if self._sensors is None:
      raise ValueError("sensor_values requires a sensor stepping profile")
    state = self._state
    dynamics = self._smooth.run_device(state._qpos, state._qvel)
    poses = dict(
        dynamics["poses"], cvel=dynamics["cvel"], root_com=dynamics["root_com"]
    )
    return self._sensors.run_device(
        state._qpos, state._qvel, state._time, poses
    ).clone()

  def _prepare_force(self, qfrc_applied):
    torch = self._state._torch
    shape = (self._state.batch_size, self._state._model.nv)
    if qfrc_applied is None:
      if self._applied_force is not None:
        self._applied_force.zero_()
      return
    if self._applied_force is None:
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

  def _prepare_wrench(self, xfrc_applied):
    if self._body_wrench is None:
      if xfrc_applied is not None:
        raise ValueError("xfrc_applied requires a passive or sensor profile")
      return
    if xfrc_applied is None:
      self._body_wrench.zero_()
      return
    torch = self._state._torch
    shape = tuple(self._body_wrench.shape)
    if isinstance(xfrc_applied, torch.Tensor):
      if (
          xfrc_applied.device.type != "mps"
          or xfrc_applied.dtype != torch.float32
          or not xfrc_applied.is_contiguous()
          or tuple(xfrc_applied.shape) != shape
      ):
        raise ValueError(
            f"xfrc_applied must be contiguous float32 MPS with shape {shape}"
        )
      self._body_wrench.copy_(xfrc_applied)
    else:
      array = np.asarray(xfrc_applied)
      if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"xfrc_applied must be numeric with shape {shape}")
      with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float32)
      if not np.all(np.isfinite(array)):
        raise ValueError("xfrc_applied must be finite float32")
      self._body_wrench.copy_(torch.tensor(array, device=self._state._device))

  def _prepare_control(self, ctrl):
    if self._control is None:
      if ctrl is not None:
        raise ValueError("ctrl requires contact_free_motor_euler_v1")
      return
    torch = self._state._torch
    shape = tuple(self._control.shape)
    if ctrl is None:
      self._control.zero_()
      return
    if isinstance(ctrl, torch.Tensor):
      if ctrl.dtype != torch.float32:
        raise TypeError("ctrl tensor must have dtype torch.float32")
      if tuple(ctrl.shape) != shape:
        raise ValueError(f"ctrl must have shape {shape}")
      if ctrl.device.type != "mps":
        raise ValueError("ctrl tensor must be on MPS")
      if not ctrl.is_contiguous():
        raise ValueError("ctrl tensor must be contiguous")
      self._control.copy_(ctrl)
      return
    array = np.asarray(ctrl)
    if array.shape != shape or array.dtype.kind not in "fiu":
      raise ValueError(f"ctrl must be numeric with shape {shape}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      array = np.asarray(array, dtype=np.float32, order="C")
    if not np.all(np.isfinite(array)):
      raise ValueError("ctrl must be finite and representable as float32")
    self._control.copy_(torch.tensor(array, device=self._state._device))

  def _acceleration(self, qpos, qvel):
    dynamics = self._smooth.run_device(qpos, qvel)
    self._state._torch.neg(dynamics["qfrc_bias"], out=self._rhs)
    if self._applied_force is not None:
      if (
          self._passive is None
          and self.profile.passive_damping_enabled
          and self._rhs.numel()
      ):
        self._rhs.addcmul_(qvel, self._damping.unsqueeze(0), value=-1.0)
      self._rhs.add_(self._applied_force)
    if self._passive is not None:
      passive, self._damping_tangent = self._passive.run_device(
          qpos, qvel, xfrc_applied=self._body_wrench, return_damping=True
      )
      self._rhs.add_(passive)
    if self._fluid is not None:
      self._rhs.add_(self._fluid.run_device(qpos, qvel, dynamics))
    if self._tendons is not None:
      tendon_force, self._tendon_damping, tendon_armature = (
          self._tendons.run_device(qpos, qvel)
      )
      self._rhs.add_(tendon_force)
      dynamics["mass_matrix"].add_(tendon_armature)
    if self._transmissions is not None:
      self._rhs.add_(self._transmissions.run_device(qpos, qvel, self._control))
    if self._motor is not None:
      self._rhs.add_(self._motor.run_device(self._control))
    acceleration, status = self._solver.run_device(
        dynamics["mass_matrix"], self._rhs
    )
    if self._contact is not None:
      contact = self._contact.run_device(
          dynamics["poses"], dynamics["mass_matrix"], acceleration, qvel
      )
      torch = self._state._torch
      status = torch.where(status == 0, contact["status"], status)
      acceleration = contact["qacc"]
      self._rhs.add_(contact["qfrc_contact"])
    if self._joint_constraints is not None:
      constrained = self._joint_constraints.run_device(
          dynamics["mass_matrix"], self._rhs, qpos, qvel
      )
      status = self._state._torch.where(
          status == 0, constrained["status"], status
      )
      acceleration = constrained["qacc"]
      self._rhs.add_(constrained["qfrc_constraint"])
    return acceleration, status, dynamics

  def step(self, steps=1, *, qfrc_applied=None, ctrl=None, xfrc_applied=None):
    """Advance all worlds by a positive number of native contact-free steps.

    ``qfrc_applied`` and ``ctrl`` are optional per-call host arrays or
    contiguous float32 MPS tensors with shapes ``[batch_size, nv]`` and
    ``[batch_size, nu]``. Each supplied input is held constant for all steps in
    this call; omitting one supplies zeros. Controls require the scalar motor
    profile. Host arrays are validated and copied before device stepping;
    MPS tensors stay on device and nonfinite rows fail independently.

    The fixed workspace is reused. Failed worlds retain their previous state
    and acceleration while status records the failure. Failure remains sticky
    until reset or restore. Device stepping performs no CPU physics, solve,
    integration or readback.
    """
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral):
      raise TypeError("steps must be a positive integer")
    if steps <= 0:
      raise ValueError("steps must be a positive integer")
    self._prepare_wrench(xfrc_applied)
    self._prepare_control(ctrl)
    self._prepare_force(qfrc_applied)
    torch = self._state._torch
    state = self._state
    for _ in range(int(steps)):
      if self._rk4 is not None:
        qpos, qvel, acceleration, time, status = self._rk4.run_device(
            state._qpos,
            state._qvel,
            state._time,
            state._status,
            lambda q, v: self._acceleration(q, v)[:2],
        )
        success = status == 0
        state._qpos = qpos.clone()
        state._qvel = qvel.clone()
        state._qacc = torch.where(success[:, None], acceleration, state._qacc)
        state._time = time.clone()
        state._status = status.clone()
        state._generation += 1
        continue
      acceleration, solve_status, dynamics = self._acceleration(
          state._qpos, state._qvel
      )
      integration_acceleration = acceleration
      if self._implicitfast is not None:
        implicit = self._implicitfast.run_device(
            dynamics["mass_matrix"], self._rhs
        )
        integration_acceleration = implicit["qacc"]
        solve_status = torch.where(
            solve_status == 0, implicit["status"], solve_status
        )
      if self._euler_solver is not None:
        self._effective_mass.copy_(dynamics["mass_matrix"])
        self._effective_mass.diagonal(dim1=1, dim2=2).add_(
            self.profile.timestep * self._damping_tangent
            if self._passive is not None
            else self._implicit_damping
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
