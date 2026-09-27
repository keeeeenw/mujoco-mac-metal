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

"""CPU-only contracts for the supported native stepping profiles."""

import copy
from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np

from mujoco_metal.lifecycle import _fingerprint
from mujoco_metal.model import load_model
from mujoco_metal.registry import TARGET_MUJOCO_VERSION


@dataclass(frozen=True)
class PipelineStageSpec:
  """Explicit model-derived specification of one pipeline stage."""
  name: str
  subsystem: str
  enabled: bool
  dependencies: tuple[str, ...]
  inputs: tuple[str, ...]
  outputs: tuple[str, ...]
  buffer_lifetimes: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPlan:
  """Model-derived execution plan and buffer audit."""
  profile_name: str
  timestep: float
  stages: tuple[PipelineStageSpec, ...]
  on_demand_queries: tuple[PipelineStageSpec, ...] = ()
  buffer_audit: tuple[dict, ...] = ()

  def is_stage_enabled(self, name: str) -> bool:
    for stage in self.stages:
      if stage.name == name:
        return stage.enabled
    for query in self.on_demand_queries:
      if query.name == name:
        return query.enabled
    return False

  def get_stage(self, name: str) -> PipelineStageSpec | None:
    for stage in self.stages:
      if stage.name == name:
        return stage
    for query in self.on_demand_queries:
      if query.name == name:
        return query
    return None

  def validate_dependencies(self) -> None:
    """Validate that the execution plan forms a topologically valid closed graph."""
    available = {
        "qpos", "qvel", "time", "status", "ctrl", "qfrc_applied", "xfrc_applied",
        "eq_active", "implicit_damping"
    }
    enabled_stages = set()
    for stage in self.stages:
      if not stage.enabled:
        continue
      for dep in stage.dependencies:
        if dep not in enabled_stages:
          raise ValueError(
              f"Stage '{stage.name}' requires dependency '{dep}' to be enabled and executed before it."
          )
      for inp in stage.inputs:
        if inp not in available:
          raise ValueError(
              f"Stage '{stage.name}' requires input '{inp}' which is not produced by any preceding stage."
          )
      available.update(stage.outputs)
      enabled_stages.add(stage.name)


@dataclass(frozen=True)
class SteppingProfile:
  """Immutable validation result for one compiled model and Euler timestep."""

  name: str
  timestep: float
  model_fingerprint: str
  descriptor_fingerprint: str
  nq: int
  nv: int
  joint_types: tuple[int, ...]
  supported: tuple[str, ...]
  irrelevant: tuple[str, ...]
  rejected: tuple[str, ...]
  passive_damping_enabled: bool = False
  implicit_euler_damping: bool = False
  execution_plan: object = None


_SUPPORTED = (
    "rigid hinge, slide, free, and ball joints",
    "gravity (including the MuJoCo gravity disable flag)",
    "joint armature",
    "semi-implicit Euler",
    "contact disabled by mjDSBL_CONTACT",
    "finite positive per-step timestep",
)

_IRRELEVANT = (
    "constraint-solver options because this profile has no constraints",
    "energy diagnostics flag; stepping does not produce energy diagnostics",
    "visual, rendering, naming, keyframe, and user-data metadata",
    "disable flags for absent or rejected subsystems",
)

_REJECTED = (
    "actuators and actuator state",
    "tendons, including tendon armature",
    "equality constraints",
    "joint or tendon limits, even if not active at the initial state",
    "joint or tendon friction loss",
    "flex/deformable elements",
    "MuJoCo plugins",
    "mocap bodies",
    "polynomial damping, joint stiffness, and springs",
    "tendon damping and stiffness",
    "fluid forces and nonzero density, viscosity, or wind",
    "body gravity compensation",
    "sensors",
    "sleep mode",
    "non-Euler integrators",
    "global passive or control callbacks",
    "unsupported disable/enable flags",
)


def _build_integrated_execution_plan(
    model, timestep, implicit_euler_damping, passive_damping_enabled, coupled_desc
) -> ExecutionPlan:
  dis = int(model.opt.disableflags)
  passive_enabled = bool(model.njnt > 0 or model.nbody > 0)
  fluid_enabled = bool(
      model.opt.density > 0 or model.opt.viscosity > 0 or np.any(model.opt.wind != 0)
  )
  tendons_enabled = bool(model.ntendon > 0)
  actuation_enabled = bool(
      model.nu > 0
      and not (dis & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))
  )
  coupled_constraints_enabled = bool(
      (coupled_desc.nc > 0 or coupled_desc.nr_joint > 0)
      and not (dis & int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT))
  )
  euler_damping_enabled = bool(implicit_euler_damping)

  smooth_assembly_deps = ["smooth_dynamics"]
  smooth_assembly_inputs = ["qfrc_bias", "qfrc_applied", "mass_matrix"]
  if passive_enabled:
    smooth_assembly_deps.append("passive_forces")
    smooth_assembly_inputs.append("qfrc_passive")
  if fluid_enabled:
    smooth_assembly_deps.append("fluid_forces")
    smooth_assembly_inputs.append("qfrc_fluid")
  if tendons_enabled:
    smooth_assembly_deps.append("fixed_tendons")
    smooth_assembly_inputs.extend(["qfrc_tendon", "tendon_armature"])
  if actuation_enabled:
    smooth_assembly_deps.append("actuation")
    smooth_assembly_inputs.append("qfrc_actuator")

  stages = [
      PipelineStageSpec(
          name="smooth_dynamics",
          subsystem="smooth",
          enabled=True,
          dependencies=(),
          inputs=("qpos", "qvel"),
          outputs=("poses", "mass_matrix", "qfrc_bias"),
          buffer_lifetimes=("scratch per step: poses, mass_matrix (batch, nv, nv), qfrc_bias (batch, nv)",),
      ),
      PipelineStageSpec(
          name="passive_forces",
          subsystem="passive",
          enabled=passive_enabled,
          dependencies=("smooth_dynamics",),
          inputs=("qpos", "qvel", "poses", "xfrc_applied"),
          outputs=("qfrc_passive", "damping_tangent"),
          buffer_lifetimes=("scratch per step: qfrc_passive (batch, nv), damping_tangent (batch, nv)",),
      ),
      PipelineStageSpec(
          name="fluid_forces",
          subsystem="fluid",
          enabled=fluid_enabled,
          dependencies=("smooth_dynamics",),
          inputs=("poses", "qvel"),
          outputs=("qfrc_fluid",),
          buffer_lifetimes=("scratch per step: qfrc_fluid (batch, nv)",),
      ),
      PipelineStageSpec(
          name="fixed_tendons",
          subsystem="tendons",
          enabled=tendons_enabled,
          dependencies=("smooth_dynamics",),
          inputs=("qpos", "qvel"),
          outputs=("qfrc_tendon", "tendon_damping", "tendon_armature"),
          buffer_lifetimes=("constant: _ancestor_mask; scratch per step: qfrc_tendon, tendon_damping, tendon_armature",),
      ),
      PipelineStageSpec(
          name="actuation",
          subsystem="transmissions",
          enabled=actuation_enabled,
          dependencies=("smooth_dynamics",),
          inputs=("ctrl", "qpos", "qvel"),
          outputs=("qfrc_actuator",),
          buffer_lifetimes=("scratch per step: qfrc_actuator (batch, nv)",),
      ),
      PipelineStageSpec(
          name="smooth_assembly",
          subsystem="smooth",
          enabled=True,
          dependencies=tuple(smooth_assembly_deps),
          inputs=tuple(smooth_assembly_inputs),
          outputs=("qfrc_smooth", "effective_mass"),
          buffer_lifetimes=("scratch per step: qfrc_smooth (batch, nv), effective_mass (batch, nv, nv)",),
      ),
      PipelineStageSpec(
          name="unconstrained_solve",
          subsystem="dense_solve",
          enabled=True,
          dependencies=("smooth_assembly",),
          inputs=("effective_mass", "qfrc_smooth"),
          outputs=("qacc_unconstrained", "unconstrained_status"),
          buffer_lifetimes=("scratch per step: qacc_unconstrained (batch, nv)",),
      ),
      PipelineStageSpec(
          name="coupled_constraints",
          subsystem="coupled_constraints",
          enabled=coupled_constraints_enabled,
          dependencies=("smooth_dynamics", "smooth_assembly", "unconstrained_solve"),
          inputs=("poses", "effective_mass", "qfrc_smooth", "qpos", "qvel", "eq_active"),
          outputs=("qacc", "qfrc_constraint", "coupled_status", "solver_diagnostics", "contact_force", "joint_force"),
          buffer_lifetimes=(
              f"persistent MPS: _eq_active_default; preallocated MPS workspace: workspace_J (batch, {coupled_desc.nr}, {model.nv}), "
              "contact_row_data, contact_jacobian, out_force, out_acc",
          ),
      ),
      PipelineStageSpec(
          name="euler_damping",
          subsystem="dense_solve",
          enabled=euler_damping_enabled,
          dependencies=(("coupled_constraints",) if coupled_constraints_enabled else ("unconstrained_solve",)),
          inputs=("effective_mass", "qfrc_smooth", "damping_tangent" if passive_enabled else "implicit_damping"),
          outputs=("integration_acceleration", "euler_status"),
          buffer_lifetimes=("scratch per step: effective_mass (batch, nv, nv), qrhs (batch, nv)",),
      ),
      PipelineStageSpec(
          name="euler_integration",
          subsystem="integration",
          enabled=True,
          dependencies=(("euler_damping",) if euler_damping_enabled else (("coupled_constraints",) if coupled_constraints_enabled else ("unconstrained_solve",))),
          inputs=("qpos", "qvel", "integration_acceleration" if euler_damping_enabled else ("qacc" if coupled_constraints_enabled else "qacc_unconstrained"), "time", "status"),
          outputs=("qpos_next", "qvel_next", "time_next", "status_next"),
          buffer_lifetimes=("persistent MPS: state.qpos, state.qvel, state.time, state.status",),
      ),
  ]

  on_demand_queries = (
      PipelineStageSpec(
          name="sensor_query",
          subsystem="sensors",
          enabled=bool(model.nsensor > 0),
          dependencies=(),
          inputs=("qpos", "qvel", "poses"),
          outputs=("sensordata",),
          buffer_lifetimes=("persistent MPS: sensordata (batch, nsensordata)",),
      ),
  )

  audit = [
      {"name": "state.qpos", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"(batch, {model.nq})", "dtype": "float32"},
      {"name": "state.qvel", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"(batch, {model.nv})", "dtype": "float32"},
      {"name": "state.status", "residency": "MPS device-resident", "lifetime": "persistent", "shape": "(batch,)", "dtype": "int32"},
      {"name": "state.time", "residency": "MPS device-resident", "lifetime": "persistent", "shape": "(batch,)", "dtype": "float32"},
      {"name": "mass_matrix", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"(batch, {model.nv}, {model.nv})", "dtype": "float32"},
      {"name": "qfrc_bias", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"(batch, {model.nv})", "dtype": "float32"},
      {"name": "qfrc_smooth", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"(batch, {model.nv})", "dtype": "float32"},
  ]
  if euler_damping_enabled:
    audit.append({"name": "effective_mass", "residency": "MPS device-resident", "lifetime": "scratch/step", "shape": f"(batch, {model.nv}, {model.nv})", "dtype": "float32"})
  if coupled_constraints_enabled:
    audit.extend([
        {"name": "_eq_active_default", "residency": "MPS device-resident", "lifetime": "persistent preallocated", "shape": f"(batch, {max(model.neq, 1)})", "dtype": "int32"},
        {"name": "workspace_J", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"(batch, {coupled_desc.nr}, {model.nv})", "dtype": "float32"},
        {"name": "out_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"(batch, {model.nv})", "dtype": "float32"},
        {"name": "out_acc", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"(batch, {model.nv})", "dtype": "float32"},
        {"name": "out_status", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": "(batch,)", "dtype": "int32"},
        {"name": "out_diagnostics", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": "(batch, 2)", "dtype": "float32"},
    ])
    if coupled_desc.nc > 0:
      audit.extend([
          {"name": "contact_row_data", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"(batch, {coupled_desc.nc}, 5, 6)", "dtype": "float32"},
          {"name": "contact_jacobian", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"(batch, {coupled_desc.nc}, 5, {model.nv})", "dtype": "float32"},
          {"name": "out_contact_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"(batch, {coupled_desc.nc * 5})", "dtype": "float32"},
      ])
    if coupled_desc.nr_joint > 0:
      audit.append({"name": "out_joint_force", "residency": "MPS device-resident", "lifetime": "preallocated workspace", "shape": f"(batch, {max(coupled_desc.nr_joint, 1)})", "dtype": "float32"})
  if model.nsensor > 0:
    audit.append({"name": "sensordata", "residency": "MPS device-resident", "lifetime": "persistent", "shape": f"(batch, {model.nsensordata})", "dtype": "float32"})

  plan = ExecutionPlan(
      profile_name="integrated_euler_v1",
      timestep=float(timestep),
      stages=tuple(stages),
      on_demand_queries=on_demand_queries,
      buffer_audit=tuple(audit),
  )
  plan.validate_dependencies()
  return plan


def validate_stepping_profile(
    model, timestep=None, profile="contact_free_euler_v1"
):
  """Validate a compiled ``MjModel`` without allocating device state.

  The model must have contact explicitly disabled. This check is deliberately
  separate from ``load_model``: generalized mass and bias queries support a
  broader model set than this stepping profile.
  """
  if profile == "integrated_euler_v1":
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    if mujoco.__version__ != TARGET_MUJOCO_VERSION:
      raise RuntimeError(
          f"requires MuJoCo {TARGET_MUJOCO_VERSION}; found {mujoco.__version__}"
      )
    dt = model.opt.timestep if timestep is None else timestep
    try:
      dt = float(dt)
    except (TypeError, ValueError, OverflowError) as error:
      raise ValueError("timestep must be finite and positive") from error
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError("timestep must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
      native_dt = np.float32(dt)
    if not np.isfinite(native_dt) or native_dt <= 0:
      raise ValueError(
          "timestep must be representable as a finite positive float32"
      )

    opt = model.opt
    if int(opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
      raise ValueError("integrated_euler_v1 requires the Euler integrator")
    if not math.isfinite(float(opt.timestep)) or opt.timestep <= 0:
      raise ValueError("compiled model timestep must be finite and positive")

    if model.nv > 32:
      raise ValueError(f"integrated_euler_v1 bounds nv to 32; found {model.nv}")

    enable = int(opt.enableflags)
    sleep = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    if enable & sleep:
      raise ValueError("sleep mode is unsupported by integrated_euler_v1")
    energy = int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    unknown_enable = enable & ~(energy | sleep)
    if unknown_enable:
      raise ValueError(f"unsupported enable flags: 0x{unknown_enable:x}")

    allowed_joints = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
        int(mujoco.mjtJoint.mjJNT_FREE),
        int(mujoco.mjtJoint.mjJNT_BALL),
    }
    joint_types = tuple(int(value) for value in model.jnt_type)
    if any(value not in allowed_joints for value in joint_types):
      raise ValueError("unsupported joint type in integrated_euler_v1")

    if mujoco.get_mjcb_contactfilter() is not None:
      raise ValueError("global contact filter callback is unsupported")
    if mujoco.get_mjcb_passive() is not None:
      raise ValueError("global passive callback is unsupported")
    if mujoco.get_mjcb_control() is not None:
      raise ValueError("global control callback is unsupported")

    if hasattr(model, "nflex") and model.nflex > 0:
      raise ValueError("flex/deformable elements are unsupported")
    if model.nplugin > 0:
      raise ValueError("MuJoCo plugins are unsupported")
    if model.nmocap > 0:
      raise ValueError("mocap bodies are unsupported")

    supported_list = [
        "rigid hinge, slide, free, and ball joints",
        "semi-implicit Euler integration",
        "gravity compensation and MuJoCo disable flags",
    ]

    if model.nu > 0:
      from mujoco_metal.transmissions import TransmissionModel
      TransmissionModel(model)
      supported_list.append("stateless scalar actuators and transmissions")
    if model.ntendon > 0:
      from mujoco_metal.tendons import FixedTendonModel
      FixedTendonModel(model)
      supported_list.append("fixed-joint tendons with spring, damping, and armature")

    from mujoco_metal.passive import PassiveForceModel
    PassiveForceModel(model)
    supported_list.append("passive joint springs, damping, gravcomp, and body wrenches")

    if model.opt.density > 0 or model.opt.viscosity > 0 or np.any(model.opt.wind != 0):
      from mujoco_metal.fluid import InertiaBoxFluidModel
      InertiaBoxFluidModel(model)
      supported_list.append("inertia-box fluid forces, wind, and viscosity")

    if model.nsensor > 0:
      from mujoco_metal.sensors import lower_sensors
      lower_sensors(model)
      supported_list.append("stateless current-state sensor queries")

    from mujoco_metal.coupled_constraints import lower_coupled_constraints
    coupled_desc = lower_coupled_constraints(model)
    if coupled_desc.nc > 0 or coupled_desc.nr_joint > 0:
      supported_list.append("coupled constraint solve for contacts, joint limits, dry friction, and equalities")

    implicit_euler_damping = not bool(int(opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP))
    passive_damping_enabled = not bool(int(opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_DAMPER))

    execution_plan = _build_integrated_execution_plan(
        model, float(native_dt), implicit_euler_damping, passive_damping_enabled, coupled_desc
    )

    return SteppingProfile(
        name="integrated_euler_v1",
        timestep=float(native_dt),
        model_fingerprint=_fingerprint(model),
        descriptor_fingerprint=_fingerprint(load_model(model)),
        nq=int(model.nq),
        nv=int(model.nv),
        joint_types=joint_types,
        supported=tuple(supported_list),
        irrelevant=(
            "energy diagnostics flag; stepping does not produce energy diagnostics",
            "visual, rendering, naming, keyframe, and user-data metadata",
            "warmstart disable flag",
        ),
        rejected=(
            "flex/deformable elements",
            "MuJoCo plugins",
            "mocap bodies",
            "spatial/wrapping tendons, tendon limits, and tendon frictionloss",
            "non-scalar/non-fixed-tendon actuators, activation state, and muscles",
            "non-sphere collision geoms",
            "sleep mode",
            "non-Euler integrators",
            "global callbacks",
        ),
        passive_damping_enabled=passive_damping_enabled,
        implicit_euler_damping=implicit_euler_damping,
        execution_plan=execution_plan,
    )

  if profile == "contact_free_implicitfast_v1":
    from mujoco_metal.implicit import lower_implicitfast

    lower_implicitfast(model)
    reference = copy.copy(model)
    reference.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    base = validate_stepping_profile(
        reference, timestep, "contact_free_passive_euler_v1"
    )
    return replace(
        base,
        name=profile,
        model_fingerprint=_fingerprint(model),
        descriptor_fingerprint=_fingerprint(load_model(model)),
        implicit_euler_damping=False,
        supported=base.supported
        + (
            "bounded implicitfast velocity solve and eligible free-body midpoint",
        ),
        rejected=tuple(
            item for item in base.rejected if item != "non-Euler integrators"
        )
        + ("full implicit integrator, nonconstant velocity derivatives",),
    )
  if profile in ("normal_contact_euler_v1", "friction_contact_euler_v1"):
    from mujoco_metal.contact import lower_contacts

    contacts = lower_contacts(model)
    if contacts.nv > 32 or contacts.pair_count > 16:
      raise ValueError(
          "native contact requires nv<=32 and at most16 candidate pairs"
      )
    if profile == "normal_contact_euler_v1" and np.any(contacts.condim != 1):
      raise ValueError(
          "normal_contact_euler_v1 requires condim1; use friction_contact_euler_v1"
      )
    reference = copy.copy(model)
    reference.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    reference.opt.disableflags &= ~(
        int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        | int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    )
    base = validate_stepping_profile(
        reference, timestep, "contact_free_passive_euler_v1"
    )
    return replace(
        base,
        name=profile,
        model_fingerprint=_fingerprint(model),
        descriptor_fingerprint=_fingerprint(load_model(model)),
        supported=tuple(
            item for item in base.supported if "contact disabled" not in item
        )
        + (
            ("normal-only sphere/plane and sphere/sphere contact",)
            if profile == "normal_contact_euler_v1"
            else ("sphere/plane and sphere/sphere condim1/3 pyramidal contact",)
        ),
        irrelevant=tuple(
            item for item in base.irrelevant if "constraint-solver" not in item
        ),
    )
  if profile in (
      "contact_free_rk4_v1",
      "contact_free_forces_rk4_v1",
      "contact_free_motor_rk4_v1",
      "contact_free_transmission_rk4_v1",
      "contact_free_fluid_rk4_v1",
      "contact_free_passive_rk4_v1",
      "contact_free_sensor_rk4_v1",
  ):
    if not isinstance(model, mujoco.MjModel):
      raise TypeError("model must be a compiled mujoco.MjModel")
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_RK4):
      raise ValueError(f"{profile} requires the RK4 integrator")
    reference = copy.copy(model)
    reference.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    result = validate_stepping_profile(
        reference, timestep, profile.replace("rk4", "euler")
    )
    return replace(
        result,
        name=profile,
        model_fingerprint=_fingerprint(model),
        implicit_euler_damping=False,
        supported=tuple(
            "four-stage Runge-Kutta" if item == "semi-implicit Euler" else item
            for item in result.supported
        ),
        rejected=tuple(
            "non-RK4 integrators" if item == "non-Euler integrators" else item
            for item in result.rejected
        ),
    )
  with_fluid = profile == "contact_free_fluid_euler_v1"
  if with_fluid:
    from mujoco_metal.fluid import InertiaBoxFluidModel

    InertiaBoxFluidModel(model)
  with_joint_constraints = profile == "joint_constraints_euler_v1"
  if with_joint_constraints:
    from mujoco_metal.joint_constraints import lower_joint_constraints

    lower_joint_constraints(model)
  with_transmissions = profile == "contact_free_transmission_euler_v1"
  with_sensors = profile == "contact_free_sensor_euler_v1"
  advanced_passive = profile in (
      "contact_free_transmission_euler_v1",
      "joint_constraints_euler_v1",
      "contact_free_fluid_euler_v1",
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  )
  if advanced_passive:
    from mujoco_metal.passive import PassiveForceModel

    PassiveForceModel(model)
  if profile not in (
      "contact_free_transmission_euler_v1",
      "joint_constraints_euler_v1",
      "contact_free_fluid_euler_v1",
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
      "contact_free_euler_v1",
      "contact_free_forces_euler_v1",
      "contact_free_motor_euler_v1",
      "contact_free_transmission_euler_v1",
      "joint_constraints_euler_v1",
      "contact_free_fluid_euler_v1",
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  ):
    raise ValueError(f"unsupported stepping profile: {profile!r}")
  if not isinstance(model, mujoco.MjModel):
    raise TypeError("model must be a compiled mujoco.MjModel")
  if mujoco.__version__ != TARGET_MUJOCO_VERSION:
    raise RuntimeError(
        f"requires MuJoCo {TARGET_MUJOCO_VERSION}; found {mujoco.__version__}"
    )

  dt = model.opt.timestep if timestep is None else timestep
  try:
    dt = float(dt)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("timestep must be finite and positive") from error
  if not math.isfinite(dt) or dt <= 0:
    raise ValueError("timestep must be finite and positive")
  with np.errstate(over="ignore", under="ignore"):
    native_dt = np.float32(dt)
  if not np.isfinite(native_dt) or native_dt <= 0:
    raise ValueError(
        "timestep must be representable as a finite positive float32"
    )

  opt = model.opt
  if int(opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
    raise ValueError(f"{profile} requires the Euler integrator")
  if not math.isfinite(float(opt.timestep)) or opt.timestep <= 0:
    raise ValueError("compiled model timestep must be finite and positive")

  disable = int(opt.disableflags)
  enable = int(opt.enableflags)
  contact = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  if not disable & contact:
    raise ValueError(
        f"{profile} requires contact explicitly disabled " "with mjDSBL_CONTACT"
    )

  # These bits affect absent or explicitly rejected subsystems. Gravity is a
  # supported switch because smooth bias honors mjDSBL_GRAVITY. Contact must
  # be disabled above; unrelated switches are rejected to avoid silent modes.
  irrelevant_disable = 0
  for name in (
      "mjDSBL_CONSTRAINT",
      "mjDSBL_EQUALITY",
      "mjDSBL_FRICTIONLOSS",
      "mjDSBL_LIMIT",
      "mjDSBL_SENSOR",
      "mjDSBL_SPRING",
      "mjDSBL_DAMPER",
      "mjDSBL_WARMSTART",
      "mjDSBL_ACTUATION",
      "mjDSBL_AUTORESET",
      "mjDSBL_CLAMPCTRL",
      "mjDSBL_EULERDAMP",
      "mjDSBL_GRAVITY",
  ):
    if profile in (
        "contact_free_forces_euler_v1",
        "contact_free_motor_euler_v1",
        "contact_free_transmission_euler_v1",
        "joint_constraints_euler_v1",
        "contact_free_fluid_euler_v1",
        "contact_free_passive_euler_v1",
        "contact_free_sensor_euler_v1",
    ) and name in (
        "mjDSBL_DAMPER",
        "mjDSBL_EULERDAMP",
    ):
      continue
    irrelevant_disable |= int(getattr(mujoco.mjtDisableBit, name))
  unknown_disable = disable & ~(contact | irrelevant_disable)
  if profile in (
      "contact_free_forces_euler_v1",
      "contact_free_motor_euler_v1",
      "contact_free_transmission_euler_v1",
      "joint_constraints_euler_v1",
      "contact_free_fluid_euler_v1",
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  ):
    unknown_disable &= ~(
        int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
        | int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    )
  if with_joint_constraints:
    unknown_disable &= ~int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
  if unknown_disable:
    raise ValueError(f"unsupported disable flags: 0x{unknown_disable:x}")

  energy = int(mujoco.mjtEnableBit.mjENBL_ENERGY)
  known_enable = energy
  sleep = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  if enable & sleep:
    raise ValueError(f"sleep mode is unsupported by {profile}")
  unknown_enable = enable & ~(known_enable | sleep)
  if unknown_enable:
    raise ValueError(f"unsupported enable flags: 0x{unknown_enable:x}")

  allowed_joints = {
      int(mujoco.mjtJoint.mjJNT_HINGE),
      int(mujoco.mjtJoint.mjJNT_SLIDE),
      int(mujoco.mjtJoint.mjJNT_FREE),
      int(mujoco.mjtJoint.mjJNT_BALL),
  }
  joint_types = tuple(int(value) for value in model.jnt_type)
  if any(value not in allowed_joints for value in joint_types):
    raise ValueError(f"unsupported joint type in {profile}")

  motor_supported = ()
  if model.nu and profile not in (
      "contact_free_motor_euler_v1",
      "contact_free_transmission_euler_v1",
      "joint_constraints_euler_v1",
      "contact_free_fluid_euler_v1",
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  ):
    raise ValueError(f"actuators are unsupported by {profile}")
  if profile in (
      "contact_free_motor_euler_v1",
      "contact_free_transmission_euler_v1",
      "joint_constraints_euler_v1",
      "contact_free_fluid_euler_v1",
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  ):
    from mujoco_metal.actuation import ScalarMotorModel

    if with_transmissions:
      from mujoco_metal.transmissions import TransmissionModel

      TransmissionModel(model)
      from mujoco_metal.tendons import FixedTendonModel

      FixedTendonModel(model)
    else:
      ScalarMotorModel.from_model(model)
    motor_supported = (
        "fixed-gain hinge and slide joint motors",
        "per-call controls with MuJoCo control/force clipping and disable flags",
    )
  if with_transmissions:
    motor_supported = (
        "stateless scalar fixed/affine gain and affine bias actuators",
        "hinge/slide and fixed joint tendon transmissions",
        "per-call controls with control/force clipping and disable flags",
    )
  if model.ntendon:
    if not with_transmissions:
      raise ValueError("tendons, including tendon armature, are unsupported")
    if np.any(model.tendon_limited) or np.any(model.tendon_actfrclimited):
      raise ValueError("tendon armature and tendon limits are unsupported")
    if np.any(model.wrap_type != int(mujoco.mjtWrap.mjWRAP_JOINT)):
      raise ValueError("only fixed joint tendons are supported")
  if model.neq and not with_joint_constraints:
    raise ValueError("equality constraints are unsupported")
  if np.any(model.jnt_limited) and not with_joint_constraints:
    raise ValueError("joint limits are unsupported, including delayed limits")
  if (np.any(model.dof_frictionloss) and not with_joint_constraints) or np.any(
      model.tendon_frictionloss
  ):
    raise ValueError("friction loss is unsupported")
  if model.nflex or model.nflexvert or model.nflexelem:
    raise ValueError("flex/deformable elements are unsupported")
  if (
      model.nplugin
      or np.any(model.body_plugin >= 0)
      or np.any(model.geom_plugin >= 0)
  ):
    raise ValueError("MuJoCo plugins are unsupported")
  if model.nmocap or np.any(model.body_mocapid >= 0):
    raise ValueError("mocap bodies are unsupported")
  if model.nsensor:
    if not with_sensors:
      raise ValueError(
          "sensors are unsupported by this profile; use contact_free_sensor_euler_v1 or RK4"
      )
    from mujoco_metal.sensors import lower_sensors

    lower_sensors(model)
  if np.any(model.dof_dampingpoly) and not advanced_passive:
    raise ValueError("polynomial damping is unsupported")
  damping = np.asarray(model.dof_damping, dtype=np.float64)
  if np.any(~np.isfinite(damping)) or np.any(damping < 0):
    raise ValueError("linear joint damping must be finite and nonnegative")
  with np.errstate(over="ignore", invalid="ignore"):
    damping32 = np.asarray(damping, dtype=np.float32)
  if np.any(~np.isfinite(damping32)):
    raise ValueError("linear joint damping must be representable as float32")
  if profile == "contact_free_euler_v1" and np.any(model.dof_damping):
    raise ValueError("joint damping is unsupported by contact_free_euler_v1")
  if (
      np.any(model.jnt_stiffness) or np.any(model.jnt_stiffnesspoly)
  ) and not advanced_passive:
    raise ValueError("joint stiffness and springs are unsupported")
  if (
      np.any(model.tendon_damping) or np.any(model.tendon_dampingpoly)
  ) and not with_transmissions:
    raise ValueError("tendon damping is unsupported")
  if (
      np.any(model.tendon_stiffness) or np.any(model.tendon_stiffnesspoly)
  ) and not with_transmissions:
    raise ValueError("tendon stiffness is unsupported")
  if (
      opt.density != 0 or opt.viscosity != 0 or np.any(opt.wind)
  ) and not with_fluid:
    raise ValueError("fluid forces are unsupported")
  if np.any(model.geom_fluid):
    raise ValueError("geom fluid interaction is unsupported")
  if np.any(model.body_gravcomp) and not advanced_passive:
    raise ValueError("body gravity compensation is unsupported")
  if (
      mujoco.get_mjcb_passive() is not None
      or mujoco.get_mjcb_control() is not None
  ):
    raise ValueError("global passive and control callbacks are unsupported")

  return SteppingProfile(
      name=profile,
      timestep=dt,
      model_fingerprint=_fingerprint(model),
      descriptor_fingerprint=_fingerprint(load_model(model)),
      nq=int(model.nq),
      nv=int(model.nv),
      joint_types=joint_types,
      supported=(
          _SUPPORTED
          + (
              ("per-call applied generalized forces", "linear joint damping")
              if profile
              in (
                  "contact_free_forces_euler_v1",
                  "contact_free_motor_euler_v1",
                  "contact_free_transmission_euler_v1",
                  "joint_constraints_euler_v1",
                  "contact_free_fluid_euler_v1",
                  "contact_free_passive_euler_v1",
                  "contact_free_sensor_euler_v1",
              )
              else ()
          )
          + motor_supported
          + (
              ("inertia-box body fluid drag, viscosity and wind",)
              if with_fluid
              else ()
          )
          + (
              (
                  "scalar joint limits, DOF frictionloss and polynomial joint equality",
              )
              if with_joint_constraints
              else ()
          )
          + (
              ("rigid joint linear/polynomial springs and damping",)
              if advanced_passive
              else ()
          )
      ),
      irrelevant=(
          tuple(
              item
              for item in _IRRELEVANT
              if "disable flags" not in item
              and not (with_joint_constraints and "constraint-solver" in item)
          )
          if profile
          in (
              "contact_free_forces_euler_v1",
              "contact_free_motor_euler_v1",
              "contact_free_transmission_euler_v1",
              "joint_constraints_euler_v1",
              "contact_free_fluid_euler_v1",
              "contact_free_passive_euler_v1",
              "contact_free_sensor_euler_v1",
          )
          else _IRRELEVANT
      ),
      rejected=(
          tuple(
              item
              for item in _REJECTED
              if not (
                  (
                      "joint damping" in item
                      or (advanced_passive and "stiffness" in item)
                  )
                  and profile
                  in (
                      "contact_free_forces_euler_v1",
                      "contact_free_motor_euler_v1",
                      "contact_free_transmission_euler_v1",
                      "joint_constraints_euler_v1",
                      "contact_free_fluid_euler_v1",
                      "contact_free_passive_euler_v1",
                      "contact_free_sensor_euler_v1",
                  )
              )
              and not (advanced_passive and item == "body gravity compensation")
              and not (with_sensors and item == "sensors")
              and not (
                  with_fluid
                  and item
                  == "fluid forces and nonzero density, viscosity, or wind"
              )
              and not (
                  with_joint_constraints
                  and item
                  in (
                      "equality constraints",
                      "joint or tendon limits, even if not active at the initial state",
                      "joint or tendon friction loss",
                  )
              )
              and not (
                  with_transmissions
                  and item
                  in (
                      "tendons, including tendon armature",
                      "tendon damping and stiffness",
                  )
              )
              and not (
                  "actuators and actuator state" in item
                  and profile
                  in (
                      "contact_free_motor_euler_v1",
                      "contact_free_transmission_euler_v1",
                      "joint_constraints_euler_v1",
                      "contact_free_fluid_euler_v1",
                      "contact_free_passive_euler_v1",
                      "contact_free_sensor_euler_v1",
                  )
              )
          )
          + (
              (
                  "non-joint equalities, ball/tendon limits, tendon frictionloss, contacts, warmstart and general solver configuration",
              )
              if with_joint_constraints
              else ()
          )
          + (
              (
                  (
                      "stateful actuators, spatial tendons, wrapping, "
                      "non-affine gains/biases, actuator damping and force-limit routing"
                      if with_transmissions
                      else "unsupported actuator state, plugins, non-joint transmissions, "
                      "non-fixed gains, biases, dynamics, actuator armature or damping, "
                      "and joint-level actuator force limits"
                  ),
              )
              if profile
              in (
                  "contact_free_motor_euler_v1",
                  "contact_free_transmission_euler_v1",
                  "joint_constraints_euler_v1",
                  "contact_free_fluid_euler_v1",
                  "contact_free_passive_euler_v1",
                  "contact_free_sensor_euler_v1",
              )
              else ()
          )
      ),
      passive_damping_enabled=(
          profile
          in (
              "contact_free_forces_euler_v1",
              "contact_free_motor_euler_v1",
              "contact_free_transmission_euler_v1",
              "joint_constraints_euler_v1",
              "contact_free_fluid_euler_v1",
              "contact_free_passive_euler_v1",
              "contact_free_sensor_euler_v1",
          )
          and not disable & int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
          and bool(
              np.any(damping32 > 0)
              or (advanced_passive and np.any(model.dof_dampingpoly))
          )
      ),
      implicit_euler_damping=(
          profile
          in (
              "contact_free_forces_euler_v1",
              "contact_free_motor_euler_v1",
              "contact_free_transmission_euler_v1",
              "joint_constraints_euler_v1",
              "contact_free_fluid_euler_v1",
              "contact_free_passive_euler_v1",
              "contact_free_sensor_euler_v1",
          )
          and not disable & int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
          and not disable & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
          and bool(
              np.any(damping32 > 0)
              or (advanced_passive and np.any(model.dof_dampingpoly))
          )
      ),
  )
