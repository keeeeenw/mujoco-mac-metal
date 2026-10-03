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

"""All-device orchestration for explicitly validated native physics profiles."""

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


def _clone_system_dict(d):
  """Clone system tensor dictionary for immutable accepted-step retention."""
  if d is None:
    return None
  import torch
  cloned = {}
  for k, v in d.items():
    if isinstance(v, torch.Tensor):
      cloned[k] = v.detach().clone()
    elif isinstance(v, dict):
      cloned[k] = _clone_system_dict(v)
    else:
      cloned[k] = v
  return cloned


class MetalSimulation:
  """Batched native MPS stepping for bounded, explicitly selected profiles.

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
    is_integrated = profile.name == "integrated_euler_v1"
    with_transmissions = "transmission" in profile.name or is_integrated
    motor_model = (
        ScalarMotorModel.from_model(model)
        if (
            not is_integrated
            and profile.name.replace("rk4", "euler")
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
    self.batch_size = int(batch_size)
    self._mjmodel = model
    descriptor = self._state._model
    self.profile = profile
    smooth_descriptor = descriptor
    if with_transmissions and model.ntendon:
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
    self._actuators = None
    self._tendons = None
    self._spatial_tendons = None
    self._spatial_kin = None
    self._spatial_cache_key = None
    self._tendon_damping = None
    self._passive = None
    self._damping_tangent = None
    self._fluid = None
    self._sensors = None
    self._sensordata = None
    self._joint_constraints = None
    self._contact = None
    self._coupled_constraints = None
    self._accepted_step = None
    self._coupled_solve_dispatches = 0

    if is_integrated:
      import mujoco as _mj_init
      plan = profile.execution_plan
      # Actuation state ownership requires the actuator objects whenever the
      # model has actuators, even with mjDSBL_ACTUATION: the disable bit
      # gates force output (kernels emit zeros) and freezes the advance
      # (next carries live state), matching pinned skip semantics while
      # keeping activation owned end-to-end (R02).
      _act_disabled = bool(int(model.opt.disableflags)
                           & int(_mj_init.mjtDisableBit.mjDSBL_ACTUATION))
      if plan.is_stage_enabled("actuation") or (_act_disabled and int(model.na) > 0):
        from mujoco_metal.stateful_actuation import ActuatorModel
        from mujoco_metal.stateful_actuation import MetalActuators
        actuator_meta = ActuatorModel(model, allow_inherited=True)
        if actuator_meta.needs_general_path:
          self._actuators = MetalActuators(model, batch_size)
        elif plan.is_stage_enabled("actuation"):
          from mujoco_metal.transmissions import MetalTransmissions
          self._transmissions = MetalTransmissions(model)
      if plan.is_stage_enabled("fixed_tendons"):
        from mujoco_metal.tendons import MetalFixedTendonDynamics
        self._tendons = MetalFixedTendonDynamics(model, batch_size, spatial_ok=True)
        from mujoco_metal.spatial_tendons import SpatialTendonModel
        from mujoco_metal.spatial_tendons import MetalSpatialTendonDynamics
        if SpatialTendonModel(model).has_spatial:
          self._spatial_tendons = MetalSpatialTendonDynamics(model, batch_size)
      if plan.is_stage_enabled("passive_forces"):
        from mujoco_metal.passive import MetalPassiveForces
        self._passive = MetalPassiveForces(model)
      if plan.is_stage_enabled("fluid_forces"):
        from mujoco_metal.fluid import MetalInertiaBoxFluid
        self._fluid = MetalInertiaBoxFluid(model, batch_size)
      if plan.is_stage_enabled("coupled_constraints"):
        from mujoco_metal.coupled_constraints import MetalCoupledConstraints
        self._coupled_constraints = MetalCoupledConstraints(model, batch_size)
      if plan.is_stage_enabled("sensor_query"):
        from mujoco_metal.sensors import SensorProgram
        self._sensors = SensorProgram(model, batch_size)
    else:
      if with_transmissions and model.nu:
        from mujoco_metal.transmissions import MetalTransmissions
        self._transmissions = MetalTransmissions(model)
      if with_transmissions and model.ntendon:
        from mujoco_metal.tendons import MetalFixedTendonDynamics
        self._tendons = MetalFixedTendonDynamics(model, batch_size, spatial_ok=True)
      if (
          with_transmissions
          or "joint_constraints" in profile.name
          or "fluid" in profile.name
          or "implicitfast" in profile.name
          or "passive" in profile.name
          or "sensor" in profile.name
          or profile.name in ("normal_contact_euler_v1", "friction_contact_euler_v1")
      ):
        from mujoco_metal.passive import MetalPassiveForces
        self._passive = MetalPassiveForces(model)
      if "fluid" in profile.name:
        from mujoco_metal.fluid import MetalInertiaBoxFluid
        self._fluid = MetalInertiaBoxFluid(model, batch_size)
      if "sensor" in profile.name:
        from mujoco_metal.sensors import SensorProgram
        self._sensors = SensorProgram(model, batch_size)
      if profile.name == "joint_constraints_euler_v1":
        from mujoco_metal.joint_constraints import JointConstraintProgram
        self._joint_constraints = JointConstraintProgram(model, batch_size)
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
    self._midpoint = None
    if profile.name == "contact_free_implicitfast_v1":
      from mujoco_metal.implicit import ImplicitFastProgram

      self._implicitfast = ImplicitFastProgram(model, batch_size)
      from mujoco_metal.implicit_midpoint import FreeBodyMidpointProgram
      from mujoco_metal.implicit_midpoint import lower_free_body_midpoints

      if lower_free_body_midpoints(model).nfree:
        self._midpoint = FreeBodyMidpointProgram(model, batch_size)
    self._euler_solver = (
        MetalDenseSolve(descriptor.nv, batch_size)
        if (
            profile.execution_plan.is_stage_enabled("euler_damping")
            if is_integrated
            else profile.implicit_euler_damping
        )
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
        if (
            is_integrated
            or profile.name.replace("rk4", "euler")
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
    self._cached_act_force = None
    self._cached_qfrc_act = None
    self._sen_act_force = None
    self._sen_qfrc_act = None
    self._sen_jnt_map = None
    self._sen_ten_map = None
    self._sen_pair_live = None
    self._sen_slot_pair = None
    self._has_acc_sensors = False
    self._has_contact_sensors = False
    self._has_ray_sensors = False
    self._has_geomdist_sensors = False
    if self._sensors is not None:
      ST = mujoco.mjtSensor
      need = np.asarray(self._sensors.descriptor.sensor_needstage)
      types = np.asarray(self._sensors.descriptor.sensor_type)
      ACC = int(mujoco.mjtStage.mjSTAGE_ACC)
      self._has_acc_sensors = bool(np.any(need == ACC))
      self._has_contact_sensors = bool(np.any(np.isin(types, [
          int(ST.mjSENS_CONTACT), int(ST.mjSENS_TOUCH)])))
      self._has_ray_sensors = bool(np.any(types == int(ST.mjSENS_RANGEFINDER)))
      self._has_geomdist_sensors = bool(np.any(np.isin(types, [
          int(ST.mjSENS_GEOMDIST), int(ST.mjSENS_GEOMNORMAL),
          int(ST.mjSENS_GEOMFROMTO)])))
      if (self._has_ray_sensors or self._has_geomdist_sensors) and profile.name != "integrated_euler_v1":
        GT = mujoco.mjtGeom
        stock = {int(GT.mjGEOM_MESH), int(GT.mjGEOM_HFIELD), int(GT.mjGEOM_SDF)}
        if bool(np.any(np.isin(np.asarray(model.geom_type), list(stock)))):
          raise ValueError(
              f"{profile.name} supports rays/geom-distance only for analytic "
              "scenes; mesh/heightfield/SDF spatial queries require "
              "integrated_euler_v1")
      cc = getattr(self, "_coupled_constraints", None)
      if cc is not None:
        d = cc.descriptor
        g1 = np.asarray(d.geom1).reshape(-1)
        g2 = np.asarray(d.geom2).reshape(-1)
        gb = np.asarray(model.geom_bodyid, dtype=np.int64)
        jn = np.asarray(model.body_jntnum, dtype=np.int64)
        par = np.asarray(model.body_parentid, dtype=np.int64)
        live = []
        for a, b in zip(g1.tolist(), g2.tolist()):
          ok = False
          for bb in (int(gb[int(a)]), int(gb[int(b)])):
            while bb > 0:
              if int(jn[bb]) > 0:
                ok = True
                break
              bb = int(par[bb])
            if ok:
              break
          live.append(1 if ok else 0)
        self._sen_pair_live = self._state._torch.as_tensor(
            np.array(live if live else [1], dtype=np.int32),
            dtype=self._state._torch.int32, device=self._state._device)
        off = np.asarray(d.pair_contact_offset, dtype=np.int64)
        slot = np.zeros(int(d.ncontacts_max), dtype=np.int32)
        for p in range(int(d.npairs)):
          slot[off[p]:off[p + 1]] = p
        self._sen_slot_pair = self._state._torch.as_tensor(
            slot.copy(), dtype=self._state._torch.int32,
            device=self._state._device)
      if profile.name != "integrated_euler_v1":
        S = mujoco.mjtSensor
        gated = {int(S.mjSENS_TOUCH), int(S.mjSENS_CONTACT),
                 int(S.mjSENS_JOINTLIMITFRC), int(S.mjSENS_TENDONLIMITFRC)}
        if bool(np.any(np.isin(np.asarray(
                self._sensors.descriptor.sensor_type), list(gated)))):
          raise ValueError(
              f"{profile.name} supports touch/contact/limit-force sensors "
              "only under integrated_euler_v1")
    self._combined_status = torch.empty_like(self._state._status)
    self._next_qpos = torch.empty_like(self._state._qpos)
    self._next_qvel = torch.empty_like(self._state._qvel)
    self._next_qacc = torch.empty_like(self._state._qacc)
    self._next_time = torch.empty_like(self._state._time)
    self._next_status = torch.empty_like(self._state._status)
    na = int(getattr(self._state, "_na", 0))
    nu = int(model.nu)
    if na > 0 and self._actuators is None:
      raise ValueError("activation state requires the general actuator stage")
    self._act_dot = torch.zeros(
        (batch_size, max(na, 1)), dtype=torch.float32, device=self._state._device)
    self._act_vel = torch.zeros(
        (batch_size, max(nu, 1)), dtype=torch.float32, device=self._state._device)
    self._next_act = (torch.empty_like(self._state._act)
                      if na > 0 else None)
    if "rk4" in profile.name:
      from mujoco_metal.runge_kutta import MetalRungeKutta

      self._rk4 = MetalRungeKutta(descriptor, batch_size, profile.timestep)
    else:
      self._rk4 = None

    self._assembled_system_valid = False

    def _handle_state_snapshot(snap):
      # Attached solver seeds ride the documented StateSnapshot field
      # (validated by its __post_init__). The raw-device snapshot alone
      # owns kinematics/equality/mocap/activation only.
      if self._coupled_constraints is not None:
        warm = self._coupled_constraints.get_warmstart()
        object.__setattr__(snap, "warmstart_multiplier", warm)

    def _handle_state_validate_restore(snap, env_ids=None):
      # Runs BEFORE any live tensor is touched: a bad seed payload raises
      # with all state unchanged. None (unknown/old snapshots) is valid
      # and restores cold seeds.
      if self._coupled_constraints is None:
        if getattr(snap, "warmstart_multiplier", None) is not None:
          raise ValueError("snapshot seeds do not match this simulation")
        return
      nr = int(self._coupled_constraints.descriptor.nr)
      warm = getattr(snap, "warmstart_multiplier", None)
      if warm is None:
        return
      import numpy as _np
      arr = _np.asarray(warm, dtype=_np.float64)
      if arr.shape != (self.batch_size, nr) or not _np.all(_np.isfinite(arr)):
        raise ValueError("snapshot seeds have an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if not _np.all(_np.isfinite(arr.astype(_np.float32))):
          raise ValueError("snapshot seeds must be float32-representable")

    def _handle_state_restore(snap, env_ids=None):
      self._assembled_system_valid = False
      if hasattr(self, "_last_coupled"):
        self._last_coupled = None
      if hasattr(self, "_last_coupled_generation"):
        self._last_coupled_generation = None
      self._accepted_step = None
      if self._coupled_constraints is not None:
        warm = getattr(snap, "warmstart_multiplier", None)
        if warm is not None:
          import numpy as _np
          self._coupled_constraints.set_warmstart(
              _np.asarray(warm, dtype=_np.float32)
              if env_ids is None else _np.asarray(warm, dtype=_np.float32)[_np.asarray(env_ids)],
              env_ids=env_ids)
        else:
          self._coupled_constraints.clear_warmstart(env_ids=env_ids)

    def _handle_state_reset(env_ids=None):
      self._assembled_system_valid = False
      if hasattr(self, "_last_coupled"):
        self._last_coupled = None
      if hasattr(self, "_last_coupled_generation"):
        self._last_coupled_generation = None
      self._accepted_step = None

    self._state._on_snapshot = _handle_state_snapshot
    self._state._on_validate_restore = _handle_state_validate_restore
    self._state._on_restore = _handle_state_restore
    self._state._on_reset = _handle_state_reset
    # R06/D1 actuator delay line: ring buffers owned here (model-level
    # config, per-world rows), threaded through the force path with
    # pre-step times and recorded after integration (pinned mj_readCtrl /
    # mj_advance order). None when no actuator configures a history
    # buffer; nsample==0 actuators always read live control.
    self._delay = None
    if int(model.nu) > 0:
      from mujoco_metal.stateful_actuation import actuator_delay_config as _delay_cfg
      from mujoco_metal.stateful_actuation import MetalDelayLine as _DelayLine
      _ns, _ip, _dl = _delay_cfg(model)
      if bool(np.any(_ns > 0)):
        self._delay = _DelayLine(_ns, _ip, _dl, batch_size=batch_size)

  @property
  def state(self):
    """The owned :class:`DeviceState` lifecycle and checkpoint interface."""
    return self._state

  @property
  def execution_plan(self):
    """The model-derived execution plan for the active stepping profile."""
    return self.profile.execution_plan

  @property
  def solver_settings(self):
    """The explicit solver configuration for the coupled constraint solve."""
    if self._coupled_constraints is not None:
      return self._coupled_constraints.solver_settings
    return None

  def get_warmstart(self):
    """Return a host copy of retained constraint multipliers, shape `(batch, nr)`.

    These seed the next coupled solve (warm start) unless the WARMSTART
    disable flag is set or the cost check prefers a cold start. Empty
    `(batch, 0)` when the model has no constraint rows.
    """
    if self._coupled_constraints is None:
      raise ValueError("get_warmstart requires a coupled constraint profile")
    return self._coupled_constraints.get_warmstart()

  def set_warmstart(self, values, env_ids=None):
    """Store retained constraint multipliers for selected environments.

    `values` accepts host arrays shaped `(nr,)` (broadcast) or
    `(len(env_ids), nr)`, finite float32-representable. Validation is
    atomic: bad input leaves all worlds unchanged. Takes effect on the next
    solve and invalidates cached assembly; sticky failure status is
    preserved. The cost check still rejects vectors that lose to cold.
    """
    if self._coupled_constraints is None:
      raise ValueError("set_warmstart requires a coupled constraint profile")
    self._coupled_constraints.set_warmstart(values, env_ids=env_ids)
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return self._state.generation

  def clear_warmstart(self, env_ids=None):
    """Zero retained multipliers (cold start) for selected environments."""
    if self._coupled_constraints is None:
      return self._state.generation
    self._coupled_constraints.clear_warmstart(env_ids=env_ids)
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return self._state.generation

  def set_equality_active(self, values, env_ids=None):
    """Set persistent equality activity for selected environments.

    `values` accepts host boolean/integer arrays with shape `(neq,)` (broadcast
    to selected worlds) or `(len(env_ids), neq)`, or contiguous int32/bool MPS
    tensors with the same shapes (copied, never borrowed). `env_ids=None`
    selects all worlds. Validation is atomic: bad input leaves all worlds
    unchanged. Changes take effect on the next step/assembly, invalidate cached
    assembly, and do not clear sticky failure status. Works for joint, connect
    and weld equalities; reattaching uses the compiled reference, not a
    recaptured pose.
    """
    if self._coupled_constraints is None:
      # No coupled stage (e.g., constraints disabled): only allow empty/no-op?
      # Preserve no-equality support: raise unless model truly has no equalities.
      neq = int(self._mjmodel.neq) if isinstance(self._mjmodel, mujoco.MjModel) else 0
      if neq == 0:
        # Accept empty calls for uniformity? Reject non-empty to avoid silent loss.
        try:
          arr = values.detach().cpu().numpy() if hasattr(values, "detach") else __import__("numpy").asarray(values)
        except Exception:
          raise ValueError("model has no equalities")
        if np.asarray(arr).size != 0:
          raise ValueError("model has no equalities")
        return self._state.generation
      raise ValueError("set_equality_active requires a coupled constraint profile")
    gen = self._state.set_equality_active(values, env_ids=env_ids)
    # Invalidate cached assembly so next assembled_system recomputes with new
    # activity and fresh diagnostics. Do not touch failure status.
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return gen

  def set_mocap(self, pos, quat=None, env_ids=None):
    """Set prescribed mocap poses for selected environments.

    Mirrors `DeviceState.set_mocap`: host arrays or contiguous float32 MPS
    tensors, `(nmocap, 3/4)` broadcast or `(len(env_ids), nmocap, 3/4)` per
    world, or a single `(…, nmocap, 7)` posquat array with `quat=None`.
    Validation is atomic; sticky failure status is preserved; changes take
    effect on the next step/assembly and invalidate cached assembly.
    """
    if getattr(self._state, "_nmocap", 0) == 0:
      raise ValueError("model has no mocap bodies")
    gen = self._state.set_mocap(pos, quat, env_ids=env_ids)
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return gen

  def get_mocap(self):
    """Return `(pos, quat)` copies of prescribed mocap poses, or `(None, None)`."""
    state = self._state
    if getattr(state, "_nmocap", 0) == 0:
      return None, None
    return state.mocap_pos, state.mocap_quat

  def copy_environment(self, src, dst):
    """Copy all state rows (qpos/qvel/qacc/time/status/eq/mocap) src -> dst."""
    gen = self._state.copy_environment(src, dst)
    for name, value in (("src", src), ("dst", dst)):
      raw = np.asarray(value)
      if raw.shape != () or raw.dtype.kind not in "iu":
        raise ValueError("src/dst must be integer environment indices")
    src_i, dst_i = int(np.asarray(src)), int(np.asarray(dst))
    for index in (src_i, dst_i):
      if not 0 <= index < self.batch_size:
        raise ValueError("environment index out of range")
    # Full simulation state follows the device rows: retained warmstarts,
    # stored sensor samples and held per-call inputs. Caches invalidate.
    cc = getattr(self, "_coupled_constraints", None)
    if cc is not None and int(cc.descriptor.nr) > 0:
      w = cc._workspace["workspace_debug"].reshape(
          self.batch_size, cc.descriptor.nr * cc.descriptor.nr + 7 * cc.descriptor.nr)
      nr = cc.descriptor.nr
      w[dst_i, nr * nr + 3 * nr:nr * nr + 4 * nr] = w[src_i, nr * nr + 3 * nr:nr * nr + 4 * nr].clone()
    if getattr(self, "_sensordata", None) is not None:
      self._sensordata[dst_i] = self._sensordata[src_i].clone()
    for held in ("_control", "_body_wrench", "_applied_force"):
      tensor = getattr(self, held, None)
      if tensor is not None:
        tensor[dst_i] = tensor[src_i].clone()
    if getattr(self, "_delay", None) is not None:
      self._delay.copy_row(src_i, dst_i)
    self._assembled_system_valid = False
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return gen

  def snapshot(self):
    """Capture a versioned simulation-level checkpoint (host, immutable).

    Covers everything needed for exact replay: the device state, retained
    constraint multipliers, held per-call inputs, stored sensor samples
    and actuator delay rings. The narrower :meth:`DeviceState.snapshot`
    owns only kinematic/equality/mocap/activation rows; use this method
    for full trajectory replay.
    """
    import numpy as _np
    torch = self._state._torch
    snap = {
        "schema_version": 1,
        "model_fingerprint": self._state._model_fingerprint,
        "profile_fingerprint": self._state._profile_fingerprint,
        "batch_size": self.batch_size,
        "nr": 0,
        "nsensordata": int(self._mjmodel.nsensordata),
        "nu": int(self._mjmodel.nu),
        "nv": int(self._mjmodel.nv),
        "nbody": int(self._mjmodel.nbody),
        "device": self._state.snapshot(),
    }
    cc = getattr(self, "_coupled_constraints", None)
    if cc is not None and int(cc.descriptor.nr) > 0:
      nr = int(cc.descriptor.nr)
      snap["nr"] = nr
      snap["warmstart"] = cc.get_warmstart().astype(_np.float32, copy=True)
    else:
      snap["warmstart"] = None
    held = {}
    for name, tensor in (("control", self._control),
                         ("wrench", self._body_wrench),
                         ("force", self._applied_force)):
      held[name] = (tensor.detach().cpu().numpy().copy() if tensor is not None else None)
    snap["held"] = held
    snap["sensordata"] = (self._sensordata.detach().cpu().numpy().copy()
                          if getattr(self, "_sensordata", None) is not None else None)
    snap["delay"] = (self._delay.snapshot()
                     if getattr(self, "_delay", None) is not None else None)
    return snap

  def restore(self, snapshot, env_ids=None):
    """Restore a simulation snapshot, fully or for selected worlds.

    Every field is validated before anything is committed: a bad snapshot
    (or bad env selection) raises with all live state untouched. Warm
    starts, held inputs and stored sensor samples follow the same rows as
    the device state. Restoring never clears retained seeds beyond what the
    snapshot carries; use :meth:`reset` for a cold start.
    """
    import numpy as _np
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
      raise ValueError("unsupported simulation snapshot schema")
    if (snapshot.get("model_fingerprint") != self._state._model_fingerprint
            or snapshot.get("profile_fingerprint") != self._state._profile_fingerprint
            or snapshot.get("batch_size") != self.batch_size):
      raise ValueError("snapshot model, profile, or batch size do not match")
    ids = self._state._env_ids(env_ids) if env_ids is not None else None
    if ids is not None and ids.size == 0:
      raise ValueError("env_ids must select at least one world")
    nr = int(snapshot.get("nr", 0))
    cc = getattr(self, "_coupled_constraints", None)
    cc_nr = int(cc.descriptor.nr) if cc is not None else 0
    if nr != cc_nr:
      raise ValueError("snapshot constraint-row dimensions do not match")
    warm = snapshot.get("warmstart")
    warm32 = None
    if nr > 0:
      warm = _np.asarray(warm, dtype=_np.float64)
      if warm.shape != (self.batch_size, nr) or not _np.all(_np.isfinite(warm)):
        raise ValueError("snapshot warmstart has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        warm32 = _np.asarray(warm, dtype=_np.float32)
      if not _np.all(_np.isfinite(warm32)):
        raise ValueError("snapshot warmstart values overflow float32")
    elif warm is not None:
      raise ValueError("snapshot warmstart must be None without constraint rows")
    if int(snapshot.get("nsensordata", -1)) != int(self._mjmodel.nsensordata):
      raise ValueError("snapshot sensor dimensions do not match")
    if (int(snapshot.get("nu", -1)) != int(self._mjmodel.nu)
            or int(snapshot.get("nv", -1)) != int(self._mjmodel.nv)
            or int(snapshot.get("nbody", -1)) != int(self._mjmodel.nbody)):
      raise ValueError("snapshot model dimensions do not match")
    held = snapshot.get("held", {})
    held_checked = {}
    for name, tensor in (("control", self._control),
                         ("wrench", self._body_wrench),
                         ("force", self._applied_force)):
      value = held.get(name)
      if tensor is None:
        if value is not None:
          raise ValueError(f"snapshot held {name} does not match this simulation")
        held_checked[name] = None
        continue
      if value is None:
        raise ValueError(f"snapshot held {name} is missing")
      arr = _np.asarray(value, dtype=_np.float64)
      if arr.shape != tuple(tensor.shape) or not _np.all(_np.isfinite(arr)):
        raise ValueError(f"snapshot held {name} has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        arr32 = _np.asarray(arr, dtype=_np.float32)
      if not _np.all(_np.isfinite(arr32)):
        raise ValueError(f"snapshot held {name} values overflow float32")
      held_checked[name] = arr32.copy()
    sens = snapshot.get("sensordata")
    sens_checked = None
    if sens is not None:
      sens_arr = _np.asarray(sens, dtype=_np.float64)
      if sens_arr.shape != (self.batch_size, int(self._mjmodel.nsensordata)) or not _np.all(_np.isfinite(sens_arr)):
        raise ValueError("snapshot sensor sample has an invalid shape or nonfinite values")
      with _np.errstate(over="ignore", under="ignore", invalid="ignore"):
        sens32 = _np.asarray(sens_arr, dtype=_np.float32)
      if not _np.all(_np.isfinite(sens32)):
        raise ValueError("snapshot sensor sample values overflow float32")
      sens_checked = sens32.copy()

    # Delay rings (R06/D1): validated pre-commit like everything else. A
    # snapshot without delay state restores cold rings; a model without
    # delay lines must not carry ring state.
    delay_checked = None
    delay_snap = snapshot.get("delay", None)
    if getattr(self, "_delay", None) is None:
      if delay_snap is not None:
        raise ValueError("snapshot delay state does not match this simulation")
    else:
      if delay_snap is None:
        raise ValueError("snapshot delay state is missing")
      delay_checked = self._delay.checked_snapshot(delay_snap)

    # Commit boundary: DeviceState validates its fields atomically before mutating.
    self._state.restore(snapshot["device"], env_ids=ids)
    torch = self._state._torch
    if nr > 0 and warm32 is not None:
      if ids is None:
        cc.set_warmstart(warm32)
      else:
        cc.set_warmstart(warm32[ids], env_ids=ids)
    for name, tensor in (("control", self._control),
                         ("wrench", self._body_wrench),
                         ("force", self._applied_force)):
      if tensor is not None:
        val_t = torch.as_tensor(held_checked[name], dtype=torch.float32, device=self._state._device)
        if ids is None:
          tensor.copy_(val_t)
        else:
          tensor[ids] = val_t[ids]
    if sens_checked is not None:
      if self._sensordata is None:
        self._sensordata = torch.zeros(
            (self.batch_size, int(self._mjmodel.nsensordata)),
            dtype=torch.float32, device=self._state._device)
      val_s = torch.as_tensor(sens_checked, dtype=torch.float32, device=self._state._device)
      if ids is None:
        self._sensordata.copy_(val_s)
      else:
        self._sensordata[ids] = val_s[ids]
    else:
      if ids is None:
        self._sensordata = None
      elif self._sensordata is not None:
        self._sensordata[ids] = 0.0
    if delay_checked is not None:
      if ids is None:
        self._delay.restore(delay_checked)
      else:
        cur = self._delay.snapshot()
        nmax = self._delay.nmax
        nu = self._delay.nu
        for row in np.asarray(ids).tolist():
          cur["cursor"].reshape(self.batch_size, nu)[row] = \
              delay_checked["cursor"].reshape(self.batch_size, nu)[row].copy()
          cur["times"].reshape(self.batch_size, nu, nmax)[row] = \
              delay_checked["times"].reshape(self.batch_size, nu, nmax)[row].copy()
          cur["values"].reshape(self.batch_size, nu, nmax)[row] = \
              delay_checked["values"].reshape(self.batch_size, nu, nmax)[row].copy()
        self._delay.restore(cur)
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    self._assembled_system_valid = False
    self._accepted_step = None
    return self._state.generation

  def reset(self, env_ids=None, qpos=None, qvel=None, eq_active=None,
            mocap_pos=None, mocap_quat=None, act=None):
    """Reset selected worlds, clear held per-call inputs, invalidate cache.

    Reset is cold: retained warmstart multipliers and delay rings are
    cleared alongside state, matching pinned reset semantics.
    """
    gen = self._state.reset(
        env_ids=env_ids, qpos=qpos, qvel=qvel, eq_active=eq_active,
        mocap_pos=mocap_pos, mocap_quat=mocap_quat, act=act,
    )
    self._clear_held_inputs(env_ids)
    if getattr(self, "_coupled_constraints", None) is not None:
      self._coupled_constraints.clear_warmstart(env_ids=env_ids)
    if getattr(self, "_delay", None) is not None:
      self._delay.reset(env_ids=env_ids)
    if getattr(self, "_sensordata", None) is not None:
      # Stored step samples are state: reset zeroes selected worlds.
      if env_ids is None:
        self._sensordata.zero_()
      else:
        ids = self._state._env_ids(env_ids)
        if ids.size:
          self._sensordata[ids.tolist()] = 0.0
    self._assembled_system_valid = False
    self._accepted_step = None
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return gen

  def reset_to_keyframe(self, key_id, env_ids=None):
    """Reset selected worlds from a compiled keyframe (pinned mj_resetDataKeyframe).

    Sets time to `key_time`, qpos/qvel/mocap to the key values, qacc/status to
    zero, equality activity to compiled defaults, and held controls to `key_ctrl`
    for the selected worlds. Invalid key ids fail atomically before any mutation.
    """
    model = self._mjmodel
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("keyframe reset requires a compiled MjModel")
    if isinstance(key_id, bool) or not isinstance(key_id, (int, np.integer)):
      raise TypeError("key_id must be an integer")
    key_id = int(key_id)
    if not 0 <= key_id < int(model.nkey):
      raise ValueError(f"key_id {key_id} out of range for nkey={int(model.nkey)}")
    nq, nv, nmocap, nu = int(model.nq), int(model.nv), int(model.nmocap), int(model.nu)
    key_qpos = np.asarray(model.key_qpos).reshape(int(model.nkey), nq)[key_id]
    key_qvel = np.asarray(model.key_qvel).reshape(int(model.nkey), nv)[key_id] if nv else np.zeros(0)
    if nmocap:
      key_mpos = np.asarray(model.key_mpos).reshape(int(model.nkey), nmocap, 3)[key_id]
      key_mquat = np.asarray(model.key_mquat).reshape(int(model.nkey), nmocap, 4)[key_id]
    else:
      key_mpos, key_mquat = None, None
    key_time = float(np.asarray(model.key_time).reshape(int(model.nkey))[key_id])
    if not np.isfinite(key_time) or key_time < 0:
      raise ValueError("keyframe time must be finite and nonnegative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      key_time32 = np.float32(key_time)
    if not np.isfinite(key_time32):
      raise ValueError("keyframe time must be float32-representable")
    # R4: validate held-control payload BEFORE any state mutation so that an
    # invalid keyframe fails atomically (state, held inputs, generation and
    # caches all untouched). _apply_key_ctrl re-checks defensively.
    if nu and self._control is not None:
      pre_ctrl = np.asarray(model.key_ctrl).reshape(int(model.nkey), nu)[key_id]
      if not np.all(np.isfinite(pre_ctrl)):
        raise ValueError("keyframe ctrl must be finite")
      with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        pre_ctrl32 = np.asarray(pre_ctrl, dtype=np.float32)
      if not np.all(np.isfinite(pre_ctrl32)):
        raise ValueError("keyframe ctrl must be float32-representable")
    na = int(model.na)
    if na:
      key_act = np.asarray(model.key_act).reshape(int(model.nkey), na)[key_id]
      if not np.all(np.isfinite(np.asarray(key_act, dtype=np.float32))):
        raise ValueError("keyframe act must be finite float32")
    else:
      key_act = None
    # Validate key payloads through the same paths as reset before mutating.
    gen = self._state.reset(
        env_ids=env_ids, qpos=key_qpos, qvel=key_qvel,
        mocap_pos=key_mpos, mocap_quat=key_mquat, act=key_act,
    )
    # mj_resetDataKeyframe sets time after _resetData; mirror per selected world.
    if env_ids is None:
      self._state._time.zero_().add_(key_time)
    else:
      ids = self._state._env_ids(env_ids)
      if ids.size:
        index = self._state._torch.as_tensor(ids, dtype=self._state._torch.int64, device=self._state._device)
        time_tensor = self._state._torch.full((ids.size,), key_time, dtype=self._state._torch.float32, device=self._state._device)
        next_time = self._state._time.clone()
        next_time.index_copy_(0, index, time_tensor)
        self._state._time = next_time
    self._state._generation += 1
    # Held controls follow key_ctrl for selected worlds; other held inputs zeroed.
    self._apply_key_ctrl(key_id, env_ids)
    # Keyframe reset is cold like reset.
    if getattr(self, "_coupled_constraints", None) is not None:
      self._coupled_constraints.clear_warmstart(env_ids=env_ids)
    if getattr(self, "_delay", None) is not None:
      self._delay.reset(env_ids=env_ids)
    if getattr(self, "_sensordata", None) is not None:
      if env_ids is None:
        self._sensordata.zero_()
      else:
        ids = self._state._env_ids(env_ids)
        if ids.size:
          self._sensordata[ids.tolist()] = 0.0
    self._assembled_system_valid = False
    self._accepted_step = None
    if hasattr(self, "_last_coupled"):
      self._last_coupled = None
    if hasattr(self, "_last_coupled_generation"):
      self._last_coupled_generation = None
    return self._state._generation

  def apply_lifecycle(self, lifecycle):
    """Adopt a rebuilt model into this live simulation, atomically (R7).

    `lifecycle` is a host-preparation `ModelLifecycle` (or a compiled
    `MjModel`) whose model must be structurally compatible: identical
    nq/nv/neq/nmocap/na counts, body/joint/geom/site/tendon/actuator counts
    and the same stepping profile (revalidated). All device stages are
    rebuilt from the new model first (raising before any mutation), then
    swapped in; live state (qpos/qvel/time/activity/held inputs) is
    preserved, fingerprints/generation advance, and assembled-system/spatial
    caches are invalidated. The model (and its batch-sharing: one model for
    all environments) applies to every world. On failure the simulation is
    untouched and keeps stepping.
    """
    from mujoco_metal.lifecycle import ModelLifecycle as _LC
    if isinstance(lifecycle, _LC):
      new_model = lifecycle._model
    elif isinstance(lifecycle, mujoco.MjModel):
      new_model = lifecycle
    else:
      raise TypeError("apply_lifecycle requires a ModelLifecycle or MjModel")
    old = self._mjmodel
    for attr in ("nq", "nv", "nbody", "njnt", "ngeom", "nsite", "ntendon",
                 "nu", "na", "neq", "nmocap"):
      if int(getattr(new_model, attr)) != int(getattr(old, attr)):
        raise ValueError(
            f"apply_lifecycle: structural count {attr} differs "
            f"({int(getattr(old, attr))} -> {int(getattr(new_model, attr))}); "
            f"rebuild the simulation instead")
    # Full validation + stage construction happens here, before any mutation
    # of this simulation (atomicity).
    fresh = type(self)(new_model, self.batch_size, profile=self.profile.name)
    keep = {"_state", "batch_size", "_success", "_combined_status",
            "_next_qpos", "_next_qvel", "_next_qacc", "_next_time",
            "_next_status", "_next_act", "_control", "_applied_force",
            "_body_wrench", "_tendon_damping", "_damping_tangent",
            "_act_dot", "_act_vel", "_spatial_kin", "_spatial_cache_key",
            "_last_coupled", "_last_coupled_generation"}
    for key, value in fresh.__dict__.items():
      if key not in keep:
        setattr(self, key, value)
    self._state.adopt_descriptor(fresh._state)
    self._assembled_system_valid = False
    self._accepted_step = None
    self._last_coupled = None
    self._last_coupled_generation = None
    self._spatial_cache_key = None
    self._spatial_kin = None
    # Delay rings are model-owned: a rebuilt model starts cold history
    # (fresh construction above replaces them; not in `keep`).
    self._state._generation += 1
    return self._state.generation

  def _clear_held_inputs(self, env_ids):
    ids = None if env_ids is None else self._state._env_ids(env_ids)
    if self._applied_force is not None:
      if ids is None:
        self._applied_force.zero_()
      elif ids.size:
        self._applied_force[ids] = 0
    if self._control is not None:
      if ids is None:
        self._control.zero_()
      elif ids.size:
        self._control[ids] = 0
    if self._body_wrench is not None:
      if ids is None:
        self._body_wrench.zero_()
      elif ids.size:
        self._body_wrench[ids] = 0

  def _apply_key_ctrl(self, key_id, env_ids):
    model = self._mjmodel
    nu = int(model.nu)
    key_ctrl = np.asarray(model.key_ctrl).reshape(int(model.nkey), nu)[key_id] if nu else None
    if key_ctrl is not None and self._control is not None:
      # R4: validate before clearing/applying so failures leave held inputs
      # and state untouched.
      if not np.all(np.isfinite(key_ctrl)):
        raise ValueError("keyframe ctrl must be finite")
      with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ctrl32 = np.asarray(key_ctrl, dtype=np.float32)
      if not np.all(np.isfinite(ctrl32)):
        raise ValueError("keyframe ctrl must be float32-representable")
    self._clear_held_inputs(env_ids)
    if key_ctrl is not None and self._control is not None:
      ids = None if env_ids is None else self._state._env_ids(env_ids)
      ctrl32 = np.asarray(key_ctrl, dtype=np.float32)
      if ids is None:
        self._control.copy_(
            self._state._torch.as_tensor(
                np.broadcast_to(ctrl32, (self.batch_size, nu)).copy(),
                dtype=self._state._torch.float32, device=self._state._device,
            )
        )
      elif ids.size:
        index = self._state._torch.as_tensor(ids, dtype=self._state._torch.int64, device=self._state._device)
        vals = self._state._torch.as_tensor(
            np.broadcast_to(ctrl32, (ids.size, nu)).copy(),
            dtype=self._state._torch.float32, device=self._state._device,
        )
        nxt = self._control.clone()
        nxt.index_copy_(0, index, vals)
        self._control = nxt

  def buffer_audit(self):
    """Return a tuple of buffer specifications derived from real device allocations."""
    b = self.batch_size
    entries = [
        {"name": "state.qpos", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._model.nq})", "dtype": str(self._state._qpos.dtype).replace("torch.", "")},
        {"name": "state.qvel", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._model.nv})", "dtype": str(self._state._qvel.dtype).replace("torch.", "")},
        {"name": "state.status", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b},)", "dtype": str(self._state._status.dtype).replace("torch.", "")},
        {"name": "state.time", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b},)", "dtype": str(self._state._time.dtype).replace("torch.", "")},
        {"name": "mass_matrix", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv}, {self._state._model.nv})", "dtype": "float32"},
        {"name": "qfrc_bias", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv})", "dtype": "float32"},
        {"name": "qfrc_smooth", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv})", "dtype": "float32"},
    ]
    if getattr(self._state, "_neq", 0) > 0 and getattr(self._state, "_eq_active", None) is not None:
      entries.append({"name": "state.eq_active", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._neq})", "dtype": str(self._state._eq_active.dtype).replace("torch.", "")})
    if getattr(self._state, "_nmocap", 0) > 0:
      entries.append({"name": "state.mocap_pos", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._nmocap}, 3)", "dtype": "float32"})
      entries.append({"name": "state.mocap_quat", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._nmocap}, 4)", "dtype": "float32"})
    if getattr(self._state, "_na", 0) > 0:
      entries.append({"name": "state.act", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._state._na})", "dtype": "float32"})
      entries.append({"name": "act_dot", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._na})", "dtype": "float32"})
    if self._euler_solver is not None:
      entries.append({"name": "effective_mass", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"({b}, {self._state._model.nv}, {self._state._model.nv})", "dtype": "float32"})
    if self._coupled_constraints is not None:
      d = self._coupled_constraints.descriptor
      entries.extend([
          {"name": "_eq_active_default", "residency": "MPS device-resident", "lifetime": "persistent preallocated", "shape": f"({b}, {max(d.neq, 1)})", "dtype": str(self._coupled_constraints._eq_active_default.dtype).replace("torch.", "")},
          {"name": "workspace_J", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.nr}, {d.nv})", "dtype": "float32"},
          {"name": "workspace_debug", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.nr * d.nr + 7 * d.nr})", "dtype": "float32"},
          {"name": "out_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.nv})", "dtype": "float32"},
          {"name": "out_acc", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.nv})", "dtype": "float32"},
          {"name": "out_status", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b},)", "dtype": str(self._coupled_constraints._workspace["out_status"].dtype).replace("torch.", "")},
          {"name": "out_diagnostics", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, 10)", "dtype": "float32"},
      ])
      if d.nc > 0:
        entries.extend([
            {"name": "contact_row_data", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.ncontacts_max}, 5, 6)", "dtype": "float32"},
            {"name": "contact_jacobian", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.ncontacts_max}, 5, {d.nv})", "dtype": "float32"},
            {"name": "out_contact_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {d.ncontacts_max * 5})", "dtype": "float32"},
        ])

      if d.nr_joint > 0:
        entries.append({"name": "out_joint_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"({b}, {max(d.nr_joint, 1)})", "dtype": "float32"})
    if self._sensors is not None:
      entries.append({"name": "sensordata", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"({b}, {self._mjmodel.nsensordata})", "dtype": "float32"})
    return tuple(entries)

  def sensor_values(self):
    """Evaluate supported stateless sensors at CURRENT state on MPS.

    This is an explicit forward-stage query, not the pre-integration sample
    left by MuJoCo mj_step (see step_sensordata). Re-evaluation makes
    reset/restore immediately visible. Returned values are detached copies
    owned by the caller.
    """
    if self._sensors is None:
      raise ValueError("sensor_values requires a sensor stepping profile")
    state = self._state
    mpos = getattr(state, "_mpos", None)
    mquat = getattr(state, "_mquat", None)
    dynamics = self._smooth.run_device(state._qpos, state._qvel, mpos, mquat)
    poses = dict(
        dynamics["poses"], cvel=dynamics["cvel"], root_com=dynamics["root_com"]
    )
    torch = state._torch
    b = state.batch_size
    base = torch.zeros((b, self._mjmodel.nsensordata), dtype=torch.float32,
                       device=state._device)
    out = self._sensors.run_device(
        state._qpos, state._qvel, state._time, poses, sensordata=base)
    out = self._run_state_sensors(state._qpos, state._qvel, poses, dynamics,
                                  out=out)
    need_acc_or_contact = self._has_acc_sensors or self._has_contact_sensors
    if need_acc_or_contact:
      # ACC and CONTACT families need the solved forward state (pinned sensorAcc runs
      # post-constraint); evaluate the full forward, then the ACC/contact kernels.
      # Queries must not perturb future trajectory: the forward solve
      # refreshes retained warmstarts and the coupled cache, so both are
      # saved and restored around the query (R03 isolation).
      cc = getattr(self, "_coupled_constraints", None)
      lam_saved = cc.get_warmstart() if cc is not None else None
      last_coupled = getattr(self, "_last_coupled", None)
      last_gen = getattr(self, "_last_coupled_generation", None)
      accepted_step = getattr(self, "_accepted_step", None)
      asm_valid = getattr(self, "_assembled_system_valid", True)
      ws_saved = {k: v.clone() for k, v in cc._workspace.items()} if cc is not None else None
      rhs_saved = self._rhs.clone() if getattr(self, "_rhs", None) is not None else None
      try:
        acceleration, _, acc_dyn = self._acceleration(state._qpos, state._qvel)
        acc_poses = dict(acc_dyn["poses"], cvel=acc_dyn["cvel"],
                         root_com=acc_dyn["root_com"])
        if self._has_acc_sensors:
          out = self._run_acc_into(state._qpos, state._qvel, acceleration,
                                   acc_poses, acc_dyn, out)
        if self._has_contact_sensors or self._has_ray_sensors or self._has_geomdist_sensors:
          out = self._run_spatial_into(acc_poses, out)
      finally:
        if ws_saved is not None:
          for k, v in ws_saved.items():
            cc._workspace[k].copy_(v)
        if rhs_saved is not None:
          self._rhs.copy_(rhs_saved)
        if cc is not None and lam_saved is not None:
          import numpy as _np
          if bool(_np.all(_np.isfinite(lam_saved))):
            cc.set_warmstart(lam_saved)
          else:
            cc.clear_warmstart()
        if hasattr(self, "_last_coupled"):
          self._last_coupled = last_coupled
        if hasattr(self, "_last_coupled_generation"):
          self._last_coupled_generation = last_gen
        self._accepted_step = accepted_step
        self._assembled_system_valid = asm_valid
    elif self._has_ray_sensors or self._has_geomdist_sensors:
      out = self._run_spatial_into(poses, out)
    return out.clone()

  def _state_sensor_inputs(self, qpos, qvel, poses, dynamics):
    """Borrowed (ten_spa_len, ten_spa_vel, act_dyn_len, act_dyn_vel, kind).

    Computes spatial/general kinematics scratch synchronously (no readback)
    so tendon/actuator state sensors observe the given state. Returns
    ``None`` views when the driving stage is absent.
    """
    ten_l = ten_v = act_l = act_v = None
    kind = 0
    if getattr(self, "_spatial_tendons", None) is not None:
      self._spatial_jacobian(qvel, poses)
      kin = self._spatial_kin
      ten_l = kin["length"].reshape(self._state.batch_size, -1)
      ten_v = kin["velocity"].reshape(self._state.batch_size, -1)
    if getattr(self, "_actuators", None) is not None:
      kind = 1
      actuators = self._actuators
      contacts = None
      if actuators.meta.has_body_transmission:
        if self._coupled_constraints is None:
          raise ValueError("BODY transmissions require the coupled constraint stage")
        self._coupled_constraints._constants["body_dims"][3] = actuators.meta.nu
        self._coupled_constraints.generate_candidates(poses, qvel)
        contacts = self._coupled_constraints.contact_buffers()
      kin = actuators.run_kinematics(qpos, qvel, poses, contacts)
      act_l = kin["length"].reshape(self._state.batch_size, -1)
      act_v = kin["velocity"].reshape(self._state.batch_size, -1)
    return ten_l, ten_v, act_l, act_v, kind

  def _run_state_sensors(self, qpos, qvel, poses, dynamics, out):
    """Evaluate milestone-016 state families into ``out`` (borrowed)."""
    ten_l, ten_v, act_l, act_v, kind = self._state_sensor_inputs(
        qpos, qvel, poses, dynamics)
    mm = dynamics.get("mass_matrix", None)
    return self._sensors.run_state_device(
        qpos, qvel, poses, mass_matrix=mm,
        ten_spa_len=ten_l, ten_spa_vel=ten_v,
        act_dyn_len=act_l, act_dyn_vel=act_v, act_kind=kind, out=out)

  def assembled_system(self, *, ctrl=None, qfrc_applied=None, recompute=False):
    """Return the coupled constraint system tensors on MPS.

    If `recompute` is True, `ctrl` is given, `qfrc_applied` is given, or no step
    has been run yet, evaluates the forward smooth dynamics and coupled constraint
    assembly at the current device state with the provided control and applied forces.
    Returns dict containing 'J', 'W', 'W_regularized', 'R', 'ar', 'rhs', 'lambda',
    'qacc', 'qfrc_constraint', 'mass_matrix', 'status'.
    Changing equality activity, reset or restore invalidates the cache.
    """
    if self._coupled_constraints is None:
      raise ValueError("assembled_system requires a coupled constraint stepping profile")
    if ctrl is not None:
      self._prepare_control(ctrl)
      recompute = True
    if qfrc_applied is not None:
      self._prepare_force(qfrc_applied)
      recompute = True
    if not recompute and getattr(self, "_assembled_system_valid", False):
      if (
          getattr(self, "_accepted_step", None) is not None
          and self._accepted_step["generation"] == self._state.generation
      ):
        return self._accepted_step["system"]
      if (
          getattr(self, "_last_coupled", None) is not None
          and getattr(self, "_last_coupled_generation", None) == self._state.generation
      ):
        return self._last_coupled
    state = self._state
    qpos, qvel = state._qpos, state._qvel
    mpos = getattr(state, "_mpos", None)
    mquat = getattr(state, "_mquat", None)
    dynamics = self._smooth.run_device(qpos, qvel, mpos, mquat)
    torch = state._torch
    rhs = torch.neg(dynamics["qfrc_bias"])
    if self._applied_force is not None:
      rhs.add_(self._applied_force)
    if self._passive is not None:
      passive, _ = self._passive.run_device(
          qpos, qvel, xfrc_applied=self._body_wrench, return_damping=True,
          mocap_pos=mpos, mocap_quat=mquat,
      )
      rhs.add_(passive)
    if self._fluid is not None:
      rhs.add_(self._fluid.run_device(qpos, qvel, dynamics))
    if self._tendons is not None:
      tendon_force, _, tendon_armature = self._tendons.run_device(qpos, qvel)
      rhs.add_(tendon_force)
      dynamics["mass_matrix"].add_(tendon_armature)
    if self._spatial_tendons is not None:
      sjac = self._spatial_jacobian(qvel, dynamics["poses"])
      skin = self._spatial_kin
      sforce, _, sarm = self._spatial_tendons.run_forces(skin)
      rhs.add_(sforce.reshape(rhs.shape))
      sbias, _ = self._spatial_tendons.run_armature_bias(
          skin, qvel, dynamics["poses"], dynamics.get("cvel", None),
          dynamics.get("root_com", None), dynamics.get("cdof", None),
          dynamics.get("cdof_dot", None))
      # Armature bias is a bias force (pinned mj_tendonBias accumulates into
      # qfrc_bias), so it subtracts from rhs like Coriolis/gravity.
      rhs.sub_(sbias.reshape(rhs.shape))
      dynamics["mass_matrix"].add_(sarm.reshape(dynamics["mass_matrix"].shape))
    if self._transmissions is not None:
      rhs.add_(self._transmissions.run_device(
          qpos, qvel, self._delayed_control(state._time)))
    if self._actuators is not None:
      rhs.add_(self._actuation_force(qpos, qvel, dynamics["poses"]))
    if self._motor is not None:
      rhs.add_(self._motor.run_device(self._delayed_control(state._time)))
    eq_active = getattr(state, "_eq_active", None)
    _ten_J, _ten_L = self._spatial_for_coupled(qvel, dynamics["poses"])
    self._coupled_solve_dispatches += 1
    coupled = self._coupled_constraints.run_device(
        dynamics["poses"], dynamics["mass_matrix"], rhs, qpos, qvel,
        eq_active=eq_active, cvel=dynamics.get("cvel", None),
        tendon_J_spatial=_ten_J, tendon_length_spatial=_ten_L,
    )
    coupled["mass_matrix"] = dynamics["mass_matrix"]
    self._last_coupled = coupled
    self._last_coupled_generation = self._state.generation
    self._assembled_system_valid = True
    return coupled

  @property
  def accepted_step(self):
    """The explicit accepted-step record from the most recent trajectory advancement."""
    if self._accepted_step is None:
      return None
    return {
        "system": _clone_system_dict(self._accepted_step["system"]),
        "generation": self._accepted_step["generation"],
        "input_generation": self._accepted_step["input_generation"],
        "input_time": self._accepted_step["input_time"],
        "status": self._accepted_step["status"].clone(),
        "acceleration": self._accepted_step["acceleration"].clone(),
        "accepted_mask": (self._accepted_step["accepted_mask"].clone()
                          if "accepted_mask" in self._accepted_step else None),
    }

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

  def _spatial_jacobian(self, qvel, poses):
    """Run spatial tendon kinematics once per assembly; stash for reuse.

    Returns the borrowed dense Jacobian [B, nt, nv] or None. Both the
    actuator-moment overlay and the spatial passive forces share this result
    within one assembly (pinned mj_tendon runs once in mj_fwdPosition).
    """
    if self._spatial_tendons is None:
      return None
    gen = (self._state.generation, id(qvel), id(poses.get("body_pos", None)))
    if getattr(self, "_spatial_cache_key", None) != gen:
      kin = self._spatial_tendons.run_kinematics(qvel, poses)
      self._spatial_cache_key = gen
      self._spatial_kin = kin
    return self._spatial_kin["jacobian"]

  def _spatial_for_coupled(self, qvel, poses):
    """Borrowed (J, length) spatial tendon views for the coupled stage."""
    if self._spatial_tendons is None:
      return None, None
    self._spatial_jacobian(qvel, poses)
    kin = self._spatial_kin
    return kin["jacobian"], kin["length"]

  def _delayed_control(self, time):
    """Held control with per-actuator delay applied (pinned mj_fwdActuation).

    ``time`` is a scalar or per-world stamp vector. Without a delay line
    this is the live held control (borrowed); with one, each actuator
    reads its ring at ``time - delay`` (live passthrough where nsample
    is 0). Read-only: recording happens in :meth:`step` after integration.
    """
    if self._delay is None or self._control is None:
      return self._control
    return self._delay.read(self._control, time)

  def _record_delay(self, time, success):
    """Record held control into delay rings for successful worlds."""
    if self._delay is None or self._control is None:
      return
    import numpy as _np
    mask = _np.ones((self.batch_size, int(self._mjmodel.nu)), dtype=_np.int32)
    if success is not None:
      ok = _np.asarray(success.detach().cpu().numpy()).reshape(-1)
      mask = (ok.astype(_np.int32).reshape(-1, 1) * _np.ones(
          (1, int(self._mjmodel.nu)), dtype=_np.int32))
    t = time.detach().cpu().numpy() if hasattr(time, "detach") else _np.asarray(time)
    self._delay.record(self._control, t.reshape(-1), mask=mask)

  def _actuation_force(self, qpos, qvel, poses):
    """General actuator force stage with pinned mj_fwdActuation ordering.

    Runs candidate-contact generation first when BODY adhesion transmissions
    exist (same-step contacts, mirroring mj_fwdPosition order), then
    transmission kinematics, gravity-compensation routing, and the fused
    dynamics/force kernel. Stashes per-step act_dot/velocity for the
    activation advance. Returns borrowed MPS qfrc_actuator.
    """
    actuators = self._actuators
    state = self._state
    contacts = None
    if actuators.meta.has_body_transmission:
      if self._coupled_constraints is None:
        raise ValueError("BODY transmissions require the coupled constraint stage")
      self._coupled_constraints._constants["body_dims"][3] = actuators.meta.nu
      self._coupled_constraints.generate_candidates(poses, qvel)
      contacts = self._coupled_constraints.contact_buffers()
    kin = actuators.run_kinematics(qpos, qvel, poses, contacts)
    spatial_jac = self._spatial_jacobian(qvel, poses)
    if spatial_jac is not None:
      # Complete spatial-tendon actuator inputs (R1): the actuator
      # kinematics stage only holds fixed-tendon maps, so overwrite
      # length/velocity rows and add gear*ten_J moment rows for actuators
      # targeting spatial tendons.
      self._spatial_tendons.apply_spatial_tendon_state(
          actuators.meta, self._spatial_kin, kin)
    gravcomp = None
    if self._passive is not None and bool(
        np.any(np.asarray(actuators.meta.jnt_actgravcomp))):
      gravcomp = self._passive.gravcomp_device(
          qpos, getattr(state, "_mpos", None), getattr(state, "_mquat", None))
    act = getattr(state, "_act", None)
    if actuators.meta.na > 0 and act is None:
      raise ValueError("activation state is missing")
    out = actuators.run_forces(self._delayed_control(state._time), act, kin, gravcomp)
    if self._sensors is not None and self._has_acc_sensors:
      self._lazy_sensor_scratch()
      self._sen_act_force.add_(out["force"].reshape(self._sen_act_force.shape))
      self._sen_qfrc_act.add_(out["qfrc"].reshape(self._sen_qfrc_act.shape))
    na, nu = actuators.meta.na, actuators.meta.nu
    if na > 0:
      self._act_dot.copy_(out["act_dot"].reshape(self._act_dot.shape))
    if nu > 0:
      self._act_vel.copy_(kin["velocity"].reshape(self._act_vel.shape))
    if actuators.meta.nv == 0:
      return self._state._torch.zeros_like(self._rhs)
    return out["qfrc"].reshape(self._rhs.shape)

  def _lazy_sensor_scratch(self):
    """Allocate (no-op after first call) ACC capture scratch."""
    if self._sen_act_force is None:
      torch = self._state._torch
      dev = self._state._device
      b, nu, nv = self.batch_size, int(self._mjmodel.nu), int(self._mjmodel.nv)
      self._sen_act_force = torch.zeros((b, max(nu, 1)), dtype=torch.float32, device=dev)
      self._sen_qfrc_act = torch.zeros((b, max(nv, 1)), dtype=torch.float32, device=dev)
      self._sen_jnt_map = None
      self._sen_ten_map = None

  def _limit_row_maps(self):
    """Host (nj,3)/(nt,2) candidate limit-row maps for ACC sensors.

    Joint rows follow the coupled layout (ball joints use the first
    reserved row only); tendon rows follow ten_base + friction-count
    ordering. Models without the integrated coupled stage get all -1
    (limit-force sensors there are rejected at construction).
    """
    if self._sen_jnt_map is not None:
      return self._sen_jnt_map, self._sen_ten_map
    import mujoco as _mj
    torch = self._state._torch
    dev = self._state._device
    m = self._mjmodel
    nj, nt = int(m.njnt), int(m.ntendon)
    jmap = np.full((max(nj, 1), 3), -1, dtype=np.int32)
    tmap = np.full((max(nt, 1), 2), -1, dtype=np.int32)
    cc = getattr(self, "_coupled_constraints", None)
    if cc is not None:
      d = cc.descriptor
      HINGE, SLIDE, BALL = (int(_mj.mjtJoint.mjJNT_HINGE),
                            int(_mj.mjtJoint.mjJNT_SLIDE),
                            int(_mj.mjtJoint.mjJNT_BALL))
      for j in range(nj):
        t = int(np.asarray(m.jnt_type)[j])
        if t in (HINGE, SLIDE):
          jmap[j] = [d.n_eq_rows + d.nv + 2 * j, d.n_eq_rows + d.nv + 2 * j + 1,
                     int(np.asarray(m.jnt_dofadr)[j])]
        elif t == BALL:
          jmap[j] = [d.n_eq_rows + d.nv + 2 * j, -1,
                     int(np.asarray(m.jnt_dofadr)[j])]
      nfric = int(d.ten_friction_rows)
      pre = 0
      for t in range(nt):
        if bool(np.asarray(m.tendon_limited)[t]):
          tmap[t] = [d.ten_base + nfric + 2 * pre, d.ten_base + nfric + 2 * pre + 1]
          pre += 1
    self._sen_jnt_map = torch.as_tensor(jmap.copy(), dtype=torch.int32, device=dev)
    self._sen_ten_map = torch.as_tensor(tmap.copy(), dtype=torch.int32, device=dev)
    return self._sen_jnt_map, self._sen_ten_map

  def _contact_views(self):
    """Borrowed coupled contact views for ACC sensors, or None."""
    cc = getattr(self, "_coupled_constraints", None)
    if cc is None:
      return None
    d = cc.descriptor
    nc = int(d.ncontacts_max)
    if nc == 0:
      return None
    w = cc._workspace
    b = self.batch_size
    return {
        "frame": w["contact_frame"][: b * nc * 12].reshape(b, nc, 12),
        "force": w["out_contact_force"][: b * nc * 11].reshape(b, nc, 11),
        "row": w["contact_row_data"][: b * nc * 6 * 6].reshape(b, nc, 6, 6)[:, :, 0, 0],
        "packed": cc._constants["contact_condim"].reshape(-1),
        "mu": cc._constants["contact_friction"].reshape(nc, 5),
        "pair_geoms": cc._constants["pair_geoms"].reshape(-1),
        "pair_offset": cc._constants["pair_contact_offset"].reshape(-1),
        "slot_pair": self._sen_slot_pair,
        "pair_live": self._sen_pair_live,
        "npairs": int(d.npairs),
    }

  def _run_spatial_into(self, poses, out):
    """Evaluate CONTACT/ray/geomdist families into ``out`` (borrowed)."""
    # Rays/geomdist use the sensor hull store (all scene meshes/hfields,
    # including contype-0 geoms that never appear in contact pairs).
    hull = self._sensors._sp_hull.reshape(-1)
    hinfo = self._sensors._sp_hull_info.reshape(-1)
    if self._has_contact_sensors:
      out = self._sensors.run_contact_device(
          poses, self._contact_views(), out=out)
    if self._has_ray_sensors or self._has_geomdist_sensors:
      if self._has_ray_sensors:
        out = self._sensors.run_rays_device(poses, hull, hinfo, out=out)
      if self._has_geomdist_sensors:
        out = self._sensors.run_geomdist_device(poses, hull, hinfo, out=out)
    return out

  def _run_acc_into(self, qpos, qvel, qacc, poses, dynamics, out):
    """Evaluate ACC force families into ``out`` (borrowed, merged)."""
    cc = getattr(self, "_coupled_constraints", None)
    lam = None
    lam_nr = lam_s = 0
    eqr = None
    if cc is not None:
      d = cc.descriptor
      lam_nr, lam_s = int(d.nr), int(d.nr * d.nr + 7 * d.nr)
      lam = cc._workspace["workspace_debug"][: self.batch_size * lam_s].reshape(
          self.batch_size, lam_s)
      eqr = cc._constants["eq_rowadr"].reshape(-1)
    jm, tm = self._limit_row_maps()
    contact = self._contact_views()
    return self._sensors.run_acc_device(
        qpos, qvel, qacc, poses, xfrc=self._body_wrench, contact=contact,
        eq_rowadr=eqr, jnt_map=jm, ten_map=tm, slot_pair=self._sen_slot_pair,
        lam_raw=lam, lam_nr=lam_nr, lam_stride=lam_s,
        act_force=self._sen_act_force, qfrc_act=self._sen_qfrc_act, out=out)

  def _acceleration(self, qpos, qvel):
    mpos = getattr(self._state, "_mpos", None)
    mquat = getattr(self._state, "_mquat", None)
    dynamics = self._smooth.run_device(qpos, qvel, mpos, mquat)
    if self._sensors is not None and self._has_acc_sensors:
      self._lazy_sensor_scratch()
      self._sen_act_force.zero_()
      self._sen_qfrc_act.zero_()
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
          qpos, qvel, xfrc_applied=self._body_wrench, return_damping=True,
          mocap_pos=mpos, mocap_quat=mquat,
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
    if self._spatial_tendons is not None:
      sjac = self._spatial_jacobian(qvel, dynamics["poses"])
      skin = self._spatial_kin
      sforce, sdamp, sarm = self._spatial_tendons.run_forces(skin)
      self._rhs.add_(sforce.reshape(self._rhs.shape))
      sbias, _ = self._spatial_tendons.run_armature_bias(
          skin, qvel, dynamics["poses"], dynamics.get("cvel", None),
          dynamics.get("root_com", None), dynamics.get("cdof", None),
          dynamics.get("cdof_dot", None))
      # Armature bias is a bias force (pinned mj_tendonBias accumulates into
      # qfrc_bias), so it subtracts from rhs like Coriolis/gravity.
      self._rhs.sub_(sbias.reshape(self._rhs.shape))
      dynamics["mass_matrix"].add_(sarm.reshape(dynamics["mass_matrix"].shape))
    if self._transmissions is not None:
      t_qfrc = self._transmissions.run_device(
          qpos, qvel, self._delayed_control(self._state._time))
      self._rhs.add_(t_qfrc)
      if self._sensors is not None and self._has_acc_sensors:
        self._lazy_sensor_scratch()
        self._sen_act_force.add_(self._transmissions._last_force.reshape(
            self._sen_act_force.shape))
        self._sen_qfrc_act.add_(t_qfrc.reshape(self._sen_qfrc_act.shape))
    if self._actuators is not None:
      self._rhs.add_(self._actuation_force(qpos, qvel, dynamics["poses"]))
    if self._motor is not None:
      m_qfrc = self._motor.run_device(self._delayed_control(self._state._time))
      self._rhs.add_(m_qfrc)
      if self._sensors is not None and self._has_acc_sensors:
        self._lazy_sensor_scratch()
        self._sen_act_force.add_(self._motor._last_force.reshape(
            self._sen_act_force.shape) if self._motor._last_force is not None
            else 0.0)
        self._sen_qfrc_act.add_(m_qfrc.reshape(self._sen_qfrc_act.shape))
    acceleration, status = self._solver.run_device(
        dynamics["mass_matrix"], self._rhs
    )
    if self._coupled_constraints is not None:
      self._coupled_solve_dispatches += 1
      eq_active = getattr(self._state, "_eq_active", None)
      _ten_J, _ten_L = self._spatial_for_coupled(qvel, dynamics["poses"])
      coupled = self._coupled_constraints.run_device(
          dynamics["poses"], dynamics["mass_matrix"], self._rhs, qpos, qvel,
          eq_active=eq_active, cvel=dynamics.get("cvel", None),
          tendon_J_spatial=_ten_J, tendon_length_spatial=_ten_L,
      )
      coupled["mass_matrix"] = dynamics["mass_matrix"]
      self._last_coupled = coupled
      self._last_coupled_generation = self._state.generation
      status = self._state._torch.where(
          status == 0, coupled["status"], status
      )
      acceleration = coupled["qacc"]
      self._rhs.add_(coupled["qfrc_constraint"])
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
    """Advance all worlds by a positive number of native profile steps.

    ``qfrc_applied`` and ``ctrl`` are optional per-call host arrays or
    contiguous float32 MPS tensors with shapes ``[batch_size, nv]`` and
    ``[batch_size, nu]``. Each supplied input is held constant for all steps in
    this call; omitting one supplies zeros. Controls require a profile supporting the model
    actuator family. Host arrays are validated and copied before device stepping;
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
    cc = getattr(self, "_coupled_constraints", None)
    for _ in range(int(steps)):
      pre_step_gen = state.generation
      pre_step_time = state._time.clone()
      pre_warm = None
      if cc is not None and int(cc.descriptor.nr) > 0:
        nr = int(cc.descriptor.nr)
        w_dbg = cc._workspace["workspace_debug"].reshape(self.batch_size, nr * nr + 7 * nr)
        pre_warm = w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr].clone()
      pre_sens = self._sensordata.clone() if self._sensordata is not None else None
      if self._rk4 is not None:
        if self._sensors is not None:
          # Pinned RK4 evaluates the outer forward (with sensors) at the
          # step-start state; intermediate stages skip sensing.
          fwd_acc, _, fwd_dyn = self._acceleration(state._qpos, state._qvel)
          self._store_step_sensors(state._qpos, state._qvel, fwd_dyn,
                                   qacc=fwd_acc)
        qpos, qvel, acceleration, time, status = self._rk4.run_device(
            state._qpos,
            state._qvel,
            state._time,
            state._status,
            lambda q, v: self._acceleration(q, v)[:2],
        )
        success = status == 0
        state._qpos = torch.where(success.unsqueeze(1), qpos, state._qpos)
        state._qvel = torch.where(success.unsqueeze(1), qvel, state._qvel)
        state._qacc = torch.where(success[:, None], acceleration, state._qacc)
        state._time = torch.where(success, time, state._time)
        state._status = status.clone()
        self._record_delay(pre_step_time, success)
        state._generation += 1
        if not bool(torch.all(success)):
          if self._sensordata is not None:
            rollback_sens = pre_sens if pre_sens is not None else torch.zeros_like(self._sensordata)
            self._sensordata.copy_(torch.where(success.unsqueeze(1), self._sensordata, rollback_sens))
          if pre_warm is not None and cc is not None:
            failed_mask = ~success
            nr = int(cc.descriptor.nr)
            w_dbg = cc._workspace["workspace_debug"].reshape(self.batch_size, nr * nr + 7 * nr)
            slice_w = w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr]
            slice_w.copy_(torch.where(failed_mask.unsqueeze(1), pre_warm, slice_w))
        if bool(torch.any(success)):
          self._assembled_system_valid = True
          if getattr(self, "_last_coupled", None) is not None:
            self._accepted_step = {
                "system": _clone_system_dict(self._last_coupled),
                "generation": state.generation,
                "input_generation": pre_step_gen,
                "input_time": pre_step_time,
                "status": state._status.clone(),
                "acceleration": state._qacc.clone(),
                "accepted_mask": success.clone(),
            }
        else:
          self._assembled_system_valid = False
        continue
      acceleration, solve_status, dynamics = self._acceleration(
          state._qpos, state._qvel
      )
      solved_acceleration = acceleration
      if self._sensors is not None:
        # Stored step-stage sample at the pre-integration (forward) state,
        # matching pinned mj_step sensor timing; see step_sensordata.
        # ACC sensors observe the solved acceleration (pre-midpoint).
        self._store_step_sensors(state._qpos, state._qvel, dynamics,
                                 qacc=solved_acceleration)
      integration_acceleration = acceleration
      next_velocity = position_velocity = None
      if self._implicitfast is not None:
        implicit = self._implicitfast.run_device(
            dynamics["mass_matrix"], self._rhs
        )
        integration_acceleration = implicit["qacc"]
        solve_status = torch.where(
            solve_status == 0, implicit["status"], solve_status
        )
      if self._midpoint is not None:
        midpoint = self._midpoint.run_device(
            state._qvel,
            integration_acceleration,
            acceleration,
            self._rhs + dynamics["qfrc_bias"],
            dynamics["poses"]["body_quat"],
        )
        next_velocity = midpoint["qvel_next"]
        position_velocity = midpoint["position_velocity"]
        acceleration = midpoint["qacc"]
        solve_status = torch.where(
            solve_status == 0, midpoint["status"], solve_status
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
          next_velocity=next_velocity,
          position_velocity=position_velocity,
      )
      torch.eq(status, 0, out=self._success)
      torch.where(
          self._success.unsqueeze(1),
          acceleration,
          state._qacc,
          out=self._next_qacc,
      )
      torch.where(
          self._success.unsqueeze(1),
          qpos,
          state._qpos,
          out=self._next_qpos,
      )
      torch.where(
          self._success.unsqueeze(1),
          qvel,
          state._qvel,
          out=self._next_qvel,
      )
      torch.where(
          self._success,
          time,
          state._time,
          out=self._next_time,
      )
      self._next_status.copy_(status)
      if self._actuators is not None and getattr(state, "_na", 0) > 0:
        self._advance_activations(state)

      # Ping-pong owned tensors keep live state disjoint from borrowed stage
      # outputs and make each new current-state reference safe for the next
      # invocation. DeviceState reset/restore may replace these references.
      state._qpos, self._next_qpos = self._next_qpos, state._qpos
      state._qvel, self._next_qvel = self._next_qvel, state._qvel
      state._qacc, self._next_qacc = self._next_qacc, state._qacc
      state._time, self._next_time = self._next_time, state._time
      state._status, self._next_status = self._next_status, state._status
      if self._actuators is not None and getattr(state, "_na", 0) > 0:
        state._act, self._next_act = self._next_act, state._act
      # Pinned history advance (mj_advance records ctrl at the pre-step
      # time); successful worlds only, failed rows freeze (R02).
      self._record_delay(pre_step_time, self._success)
      state._generation += 1

      # Failure atomicity (R02): rollback sensor samples and warmstarts for failed worlds
      if not bool(torch.all(self._success)):
        if self._sensordata is not None:
          rollback_sens = pre_sens if pre_sens is not None else torch.zeros_like(self._sensordata)
          self._sensordata.copy_(torch.where(self._success.unsqueeze(1), self._sensordata, rollback_sens))
        if pre_warm is not None and cc is not None:
          failed_mask = ~self._success
          nr = int(cc.descriptor.nr)
          w_dbg = cc._workspace["workspace_debug"].reshape(self.batch_size, nr * nr + 7 * nr)
          slice_w = w_dbg[:, nr * nr + 3 * nr:nr * nr + 4 * nr]
          slice_w.copy_(torch.where(failed_mask.unsqueeze(1), pre_warm, slice_w))

      # Accepted step record (R03/R04): exact immutable snapshot of the solved system
      if bool(torch.any(self._success)):
        self._assembled_system_valid = True
        if getattr(self, "_last_coupled", None) is not None:
          self._accepted_step = {
              "system": _clone_system_dict(self._last_coupled),
              "generation": state.generation,
              "input_generation": pre_step_gen,
              "input_time": pre_step_time,
              "status": state._status.clone(),
              "acceleration": state._qacc.clone(),
              "accepted_mask": self._success.clone(),
          }
      else:
        self._assembled_system_valid = False
    return state._status

  def step_sensordata(self):
    """Return a host copy of the stored step-stage sensor sample.

    The sample is taken at every step's pre-integration forward state for
    the model's position/velocity stages (acceleration stages join in the
    force-family milestone), exactly the values MuJoCo's ``mj_step`` leaves
    in ``d->sensordata``. This differs from :meth:`sensor_values`, which
    re-evaluates at the current (post-step) state. Reset zeroes the sample;
    a state restore leaves it untouched until the next step recomputes it.
    """
    if self._sensors is None or self._sensordata is None:
      raise ValueError("step_sensordata requires a sensor stepping profile")
    return self._sensordata.detach().cpu().numpy().copy()

  def _store_step_sensors(self, qpos, qvel, dynamics, qacc=None):
    """Evaluate position/velocity (+ACC, when present) stages into storage."""
    import mujoco as _mj
    torch = self._state._torch
    if self._sensordata is None:
      self._sensordata = torch.zeros(
          (self._state.batch_size, self._mjmodel.nsensordata),
          dtype=torch.float32, device=self._state._device)
    poses = dict(dynamics["poses"], cvel=dynamics["cvel"],
                 root_com=dynamics["root_com"])
    stages = [_mj.mjtStage.mjSTAGE_POS, _mj.mjtStage.mjSTAGE_VEL]
    old = self._sensors.run_device(
        qpos, qvel, self._state._time, poses, stages=tuple(stages))
    merged = self._run_state_sensors(qpos, qvel, poses, dynamics, out=old)
    if self._has_acc_sensors and qacc is not None:
      # _acceleration just ran at this exact state; its cached actuator
      # rows plus the solved constraint rows feed the ACC kernel.
      merged = self._run_acc_into(qpos, qvel, qacc, poses, dynamics, merged)
    if self._has_contact_sensors or self._has_ray_sensors or self._has_geomdist_sensors:
      merged = self._run_spatial_into(poses, merged)
    self._sensordata.copy_(merged)

  def _advance_activations(self, state):
    """Advance activation state with pinned mj_nextActivation semantics.

    Under mjDSBL_ACTUATION the advance is skipped but the next buffer is
    still defined (live state carried over), so the end-of-step transaction
    preserves activation exactly. Failed worlds keep their previous
    activation via the final accepted-step mask, matching the qpos/qvel
    rollback contract.
    """
    import mujoco as _mj
    torch = state._torch
    if int(self._mjmodel.opt.disableflags) & int(_mj.mjtDisableBit.mjDSBL_ACTUATION):
      self._next_act.copy_(state._act)
      return
    new_act = self._actuators.advance(state._act, self._act_dot, self._act_vel)
    torch.where(
        self._success.unsqueeze(1),
        new_act,
        state._act,
        out=self._next_act,
    )
