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


def validate_stepping_profile(
    model, timestep=None, profile="contact_free_euler_v1"
):
  """Validate a compiled ``MjModel`` without allocating device state.

  The model must have contact explicitly disabled. This check is deliberately
  separate from ``load_model``: generalized mass and bias queries support a
  broader model set than this stepping profile.
  """
  if profile in (
      "contact_free_rk4_v1",
      "contact_free_forces_rk4_v1",
      "contact_free_motor_rk4_v1",
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
  with_sensors = profile == "contact_free_sensor_euler_v1"
  advanced_passive = profile in (
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  )
  if advanced_passive:
    from mujoco_metal.passive import PassiveForceModel

    PassiveForceModel(model)
  if profile not in (
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
      "contact_free_euler_v1",
      "contact_free_forces_euler_v1",
      "contact_free_motor_euler_v1",
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
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  ):
    unknown_disable &= ~(
        int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
        | int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    )
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
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  ):
    raise ValueError(f"actuators are unsupported by {profile}")
  if profile in (
      "contact_free_motor_euler_v1",
      "contact_free_passive_euler_v1",
      "contact_free_sensor_euler_v1",
  ):
    from mujoco_metal.actuation import ScalarMotorModel

    ScalarMotorModel.from_model(model)
    motor_supported = (
        "fixed-gain hinge and slide joint motors",
        "per-call controls with MuJoCo control/force clipping and disable flags",
    )
  if model.ntendon:
    raise ValueError("tendons, including tendon armature, are unsupported")
  if model.neq:
    raise ValueError("equality constraints are unsupported")
  if np.any(model.jnt_limited):
    raise ValueError("joint limits are unsupported, including delayed limits")
  if np.any(model.dof_frictionloss) or np.any(model.tendon_frictionloss):
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
  if np.any(model.tendon_damping) or np.any(model.tendon_dampingpoly):
    raise ValueError("tendon damping is unsupported")
  if np.any(model.tendon_stiffness) or np.any(model.tendon_stiffnesspoly):
    raise ValueError("tendon stiffness is unsupported")
  if opt.density != 0 or opt.viscosity != 0 or np.any(opt.wind):
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
                  "contact_free_passive_euler_v1",
                  "contact_free_sensor_euler_v1",
              )
              else ()
          )
          + motor_supported
          + (
              ("rigid joint linear/polynomial springs and damping",)
              if advanced_passive
              else ()
          )
      ),
      irrelevant=(
          tuple(item for item in _IRRELEVANT if "disable flags" not in item)
          if profile
          in (
              "contact_free_forces_euler_v1",
              "contact_free_motor_euler_v1",
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
                      "contact_free_passive_euler_v1",
                      "contact_free_sensor_euler_v1",
                  )
              )
              and not (advanced_passive and item == "body gravity compensation")
              and not (with_sensors and item == "sensors")
              and not (
                  "actuators and actuator state" in item
                  and profile
                  in (
                      "contact_free_motor_euler_v1",
                      "contact_free_passive_euler_v1",
                      "contact_free_sensor_euler_v1",
                  )
              )
          )
          + (
              (
                  "unsupported actuator state, plugins, non-joint transmissions, "
                  "non-fixed gains, biases, dynamics, actuator armature or damping, "
                  "and joint-level actuator force limits",
              )
              if profile
              in (
                  "contact_free_motor_euler_v1",
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
