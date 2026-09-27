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

"""Optional MuJoCo Metal experiments; import does not initialize a GPU."""

__version__ = "0.4.0"


def __getattr__(name):
  if name in ("load_model", "ModelDescriptor"):
    from mujoco_metal import model

    return getattr(model, name)
  if name in ("ModelLifecycle", "KinematicsBatchState", "BatchedConstants"):
    from mujoco_metal import lifecycle

    return getattr(lifecycle, name)
  if name == "MetalKinematics":
    from mujoco_metal.metal_kinematics import MetalKinematics

    return MetalKinematics
  if name == "smooth_dynamics":
    from mujoco_metal.smooth import smooth_dynamics

    return smooth_dynamics
  if name in ("SteppingProfile", "validate_stepping_profile"):
    from mujoco_metal import stepping

    return getattr(stepping, name)
  if name in ("DeviceState", "StateSnapshot"):
    from mujoco_metal import device_state

    return getattr(device_state, name)
  if name == "MetalSmoothDynamics":
    from mujoco_metal.smooth_metal import MetalSmoothDynamics

    return MetalSmoothDynamics
  if name == "MetalDenseSolve":
    from mujoco_metal.smooth_solve import MetalDenseSolve

    return MetalDenseSolve
  if name == "MetalEulerIntegration":
    from mujoco_metal.integration import MetalEulerIntegration

    return MetalEulerIntegration
  if name in ("ScalarMotorModel", "MetalScalarMotorForce"):
    from mujoco_metal import actuation

    return getattr(actuation, name)
  if name == "MetalSimulation":
    from mujoco_metal.simulation import MetalSimulation

    return MetalSimulation
  if name in (
      "CoupledConstraintDescriptor",
      "lower_coupled_constraints",
      "coupled_constraint_oracle",
      "MetalCoupledConstraints",
  ):
    from mujoco_metal import coupled_constraints

    return getattr(coupled_constraints, name)
  raise AttributeError(name)
