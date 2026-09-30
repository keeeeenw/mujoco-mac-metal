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

"""Machine-readable pinned-version support inventory; import is host-only."""

from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum

import mujoco

TARGET_MUJOCO_VERSION = "3.10.0"
SOURCE_TREE_VERSION = "3.14.1 (not a target)"


class Stage(str, Enum):
  MODEL = "model"
  KINEMATICS = "kinematics"
  DYNAMICS = "dynamics"
  COLLISION = "collision"
  CONSTRAINTS = "constraints"
  INTEGRATION = "integration"
  SENSOR = "sensor"
  RENDERING = "rendering"
  API = "api"


class Implementation(str, Enum):
  LOWERED = "lowered"
  CPU_REFERENCE = "cpu_reference"
  NATIVE_GPU = "native_gpu"
  UPSTREAM_CPU_ONLY = "upstream_cpu_only"
  NOT_IMPLEMENTED = "not_implemented"


class Qualification(str, Enum):
  UNQUALIFIED = "unqualified"
  CPU_ORACLE = "cpu_oracle"
  GPU_QUALIFIED = "gpu_qualified"


class Execution(str, Enum):
  HOST = "host"
  DEVICE = "device"
  UPSTREAM_HOST = "upstream_host"
  NONE = "none"


@dataclass(frozen=True)
class Feature:
  name: str
  stage: Stage
  implementation: Implementation
  qualification: Qualification
  execution: Execution
  limitation: str


# Every member is an explicit inventory row, but remains unsupported until the
# corresponding stage gains an implementation and qualification record.
_ENUMS = (
    "mjtJoint",
    "mjtGeom",
    "mjtIntegrator",
    "mjtCone",
    "mjtJacobian",
    "mjtSolver",
    "mjtEq",
    "mjtTrn",
    "mjtDyn",
    "mjtGain",
    "mjtBias",
    "mjtSensor",
    "mjtState",
    "mjtDisableBit",
    "mjtEnableBit",
)


def _enum_inventory():
  rows = []
  for enum_name in _ENUMS:
    enum_type = getattr(mujoco, enum_name, None)
    if enum_type is None:
      rows.append(
          Feature(
              f"enum:{enum_name}",
              Stage.MODEL,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "not exposed by pinned Python bindings",
          )
      )
      continue
    for member_name in enum_type.__members__:
      rows.append(
          Feature(
              f"{enum_name}.{member_name}",
              Stage.MODEL,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "inventory only; not implemented by this package",
          )
      )
  return rows


_API_FAMILIES = (
    "mj_forward/mj_step/mj_step1/mj_step2",
    "mj_inverse/mj_compareFwdInv",
    "mj_resetData/mj_resetDataKeyframe/mj_copyData",
    "mj_getState/mj_setState",
    "mj_fullM/mj_mulM/mj_solveM/mj_factorM",
    "mj_jac/mj_jacBody/mj_jacSite/mj_jacGeom",
    "mj_ray/mj_ray flex and geom query APIs",
    "mj_energyPos/mj_energyVel",
    "mj_sensorAcc/mj_objectVelocity/mj_objectAcceleration",
    "mj_addPlugin/mj_plugin APIs",
    "mjv_* visualization and renderer APIs",
    "mj_saveModel/mj_loadModel/mj_printModel",
)


def _inventory():
  result = [
      Feature(
          "compiled model lowering",
          Stage.MODEL,
          Implementation.LOWERED,
          Qualification.CPU_ORACLE,
          Execution.HOST,
          "requires MuJoCo 3.10.0",
      )
  ]
  result.append(
      Feature(
          "generic CPU joint FK",
          Stage.KINEMATICS,
          Implementation.CPU_REFERENCE,
          Qualification.CPU_ORACLE,
          Execution.HOST,
          "kinematics only; no physics stepping",
      )
  )
  result.append(
      Feature(
          "generic Metal joint FK",
          Stage.KINEMATICS,
          Implementation.NATIVE_GPU,
          Qualification.GPU_QUALIFIED,
          Execution.DEVICE,
          "narrow Apple M1 FK cases only; one immutable model per batch; no per-row model randomization or physics stepping",
      )
  )
  result.append(
      Feature(
          "native Metal smooth mass matrix and inertial bias",
          Stage.DYNAMICS,
          Implementation.NATIVE_GPU,
          Qualification.GPU_QUALIFIED,
          Execution.DEVICE,
          "narrow Apple M1 fixtures only; one immutable model per batch; no per-row model randomization, actuation, passive forces, contacts, constraints, or stepping",
      )
  )
  result.append(
      Feature(
          "CPU smooth mass matrix and inertial bias reference",
          Stage.DYNAMICS,
          Implementation.CPU_REFERENCE,
          Qualification.CPU_ORACLE,
          Execution.HOST,
          "zero actuator/tendon armature only; no actuator/passive force computation, contacts, constraints, or stepping",
      )
  )
  result.extend(_enum_inventory())
  result.extend(
      Feature(
          f"api:{name}",
          Stage.API,
          Implementation.UPSTREAM_CPU_ONLY,
          Qualification.UNQUALIFIED,
          Execution.UPSTREAM_HOST,
          "not exposed as a Metal API",
      )
      for name in _API_FAMILIES
  )
  result.extend(
      Feature(
          f"python-api:{name}",
          Stage.API,
          Implementation.UPSTREAM_CPU_ONLY,
          Qualification.UNQUALIFIED,
          Execution.UPSTREAM_HOST,
          "upstream Python binding only; not exposed as a Metal API",
      )
      for name in sorted(dir(mujoco))
      if name.startswith(("mj_", "mju_")) and callable(getattr(mujoco, name))
  )
  result.extend(
      (
          Feature(
              "mj_setConst parameter recomputation",
              Stage.MODEL,
              Implementation.CPU_REFERENCE,
              Qualification.CPU_ORACLE,
              Execution.HOST,
              "host-only transactional model constant update",
          ),
          Feature(
              "batched qpos cache/reset/restore",
              Stage.API,
              Implementation.CPU_REFERENCE,
              Qualification.CPU_ORACLE,
              Execution.HOST,
              "CPU state lifecycle only; no stepping",
          ),
          Feature(
              "collision pipeline",
              Stage.COLLISION,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "general collision coverage remains incomplete; bounded normal-contact stage inventoried separately",
          ),
          Feature(
              "constraint assembly and solvers",
              Stage.CONSTRAINTS,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "general constraints remain incomplete; bounded normal-contact stage inventoried separately",
          ),
          Feature(
              "actuator evaluation",
              Stage.DYNAMICS,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "unrestricted actuation remains incomplete; bounded motor and servo stages inventoried separately",
          ),
          Feature(
              "forward/inverse dynamics",
              Stage.DYNAMICS,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "general forward/inverse dynamics APIs and unrestricted force evaluation are not implemented; the bounded profile is inventoried separately",
          ),
          Feature(
              "integrators/stepping",
              Stage.INTEGRATION,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "general-purpose integration and mj_step/mj_step1/mj_step2 are not implemented; see the separately inventoried bounded profile",
          ),
          Feature(
              "native dense SPD factorization and multiple-RHS solve",
              Stage.DYNAMICS,
              Implementation.NATIVE_GPU,
              Qualification.GPU_QUALIFIED,
              Execution.DEVICE,
              "narrow M1 correctness qualification against conditioned/scaled synthetic systems; dense float32 Cholesky, per-world failure status, no jitter; not a general dynamics solver qualification",
          ),
          Feature(
              "native semi-implicit Euler joint-coordinate integration",
              Stage.INTEGRATION,
              Implementation.NATIVE_GPU,
              Qualification.GPU_QUALIFIED,
              Execution.DEVICE,
              "narrow M1 comparison against MuJoCo 3.10 for hinge/slide/ball/free position, velocity, quaternion, and time updates; component qualification only",
          ),
          Feature(
              "persistent batched MPS simulation state",
              Stage.API,
              Implementation.NATIVE_GPU,
              Qualification.GPU_QUALIFIED,
              Execution.DEVICE,
              "qpos/qvel/qacc/time/status/eq_active storage with reset, snapshots (schema v2 with activity), restore and row reset; host checkpoint readback; does not establish end-to-end stepping qualification",
          ),
          Feature(
              "contact_free_euler_v1 native simulation pipeline",
              Stage.INTEGRATION,
              Implementation.NATIVE_GPU,
              Qualification.GPU_QUALIFIED,
              Execution.DEVICE,
              "narrow M1 qualification: four rigid-body model fixtures, three initial states, and 1,000-step 1 ms rollouts with reset/restore resume; profile requires contact-disabled Euler models with hinge/slide/free/ball joints and zero applied forces; actuators, tendons, limits, passive forces, sensors and other unsupported features are rejected",
          ),
          Feature(
              "contact_free_forces_euler_v1 native force/damping pipeline",
              Stage.INTEGRATION,
              Implementation.NATIVE_GPU,
              Qualification.GPU_QUALIFIED,
              Execution.DEVICE,
              "narrow M1 CPU-reference qualification: applied generalized force and linear joint damping, separate physical and Euler-effective acceleration solves; no Cartesian-force projection, springs, polynomial damping, contacts or general actuation",
          ),
          Feature(
              "scalar motor force mapping and contact_free_motor_euler_v1",
              Stage.INTEGRATION,
              Implementation.NATIVE_GPU,
              Qualification.GPU_QUALIFIED,
              Execution.DEVICE,
              "narrow M1 qualification: fixed-gain, no-bias, stateless hinge/slide joint motors; control/force clipping and actuator disable flags; no nonzero actuator armature/damping, joint actuator-force limits or other transmissions; host control upload or device controls",
          ),
          Feature(
              "built-in sensors",
              Stage.SENSOR,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "full sensor family/timing semantics remain incomplete; bounded current-state queries inventoried separately",
          ),
          Feature(
              "deformables and plugins",
              Stage.DYNAMICS,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "no device deformables or plugins",
          ),
          Feature(
              "rendering",
              Stage.RENDERING,
              Implementation.NOT_IMPLEMENTED,
              Qualification.UNQUALIFIED,
              Execution.NONE,
              "no Metal renderer",
          ),
      )
  )
  for name, stage, scope in (
      (
          "contact-free RK4",
          Stage.INTEGRATION,
          "quaternion-aware native RK4, mixed-joint trajectories; no stateful actuators or contact RK4",
      ),
      (
          "rigid passive forces",
          Stage.DYNAMICS,
          "joint springs, polynomial damping, gravcomp and Cartesian body forces; wider force families are separate stages",
      ),
      (
          "stateless scalar transmissions",
          Stage.DYNAMICS,
          "fixed/affine gains and affine biases, scalar joints and fixed joint tendons with spring/damping/armature; no activation state or spatial wrapping",
      ),
      (
          "normal contact slice",
          Stage.CONSTRAINTS,
          "plane-sphere/sphere-sphere condim1; nv<=32, <=16 candidates, Euler only; no broad geometry/constraint coverage",
      ),
      (
          "joint constraints",
          Stage.CONSTRAINTS,
          "scalar limits, DOF frictionloss, polynomial joint equality plus connect/weld equalities with per-environment activity; tendon/ball limits and other equality families remain separate stages",
      ),
      (
          "inertia-box fluid",
          Stage.DYNAMICS,
          "body drag, viscosity and wind; no geom ellipsoid, lift or buoyancy",
      ),
      (
          "bounded implicitfast",
          Stage.INTEGRATION,
          "rigid joints, constant damping, scalar motors and eligible free-body midpoint; full implicit remains unsupported",
      ),
      (
          "current-state sensor queries",
          Stage.SENSOR,
          "joint/frame/clock/gyro/velocity subset; explicit query, not stored mj_step timing; no history/noise/delay",
      ),
      (
          "integrated Euler pipeline",
          Stage.INTEGRATION,
          "coupled constraints (sphere contacts, scalar limits, dry friction, joint/connect/weld equalities with per-environment activity) with actuation, tendons, passive/fluid forces, and sensors in integrated_euler_v1",
      ),
  ):
    result.append(
        Feature(
            name,
            stage,
            Implementation.NATIVE_GPU,
            Qualification.GPU_QUALIFIED,
            Execution.DEVICE,
            "narrow local M1 CPU-reference qualification: " + scope,
        )
    )
  return tuple(result)


FEATURES = _inventory()
INVENTORY_COMPLETE = False


SOURCE_TAG = "mujoco-3.10.0"

MILESTONE_IDS = (
    "baseline",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
    "out-of-scope",
)


@dataclass(frozen=True)
class Requirement:
  """One auditable support-contract row with an owning milestone."""

  id: str
  name: str
  stage: Stage
  implementation: Implementation
  qualification: Qualification
  execution: Execution
  limitation: str
  source_ref: str
  admission: str
  entry_point: str
  milestone: str
  tests: tuple[str, ...] = ()
  enums: tuple[str, ...] = ()


REQUIREMENTS = (
    # Pinned model lowering and version gate (pre-existing baseline).
    Requirement(
        "REQ-MOD-001", "pinned MuJoCo 3.10.0 model lowering",
        Stage.MODEL, Implementation.LOWERED, Qualification.CPU_ORACLE,
        Execution.HOST, "requires MuJoCo 3.10.0; other versions rejected",
        "mujoco_metal/model.py:load_model", "accept 3.10.0, reject others",
        "load_model", "baseline", ("test_model.py",),
    ),
    Requirement(
        "REQ-MOD-002", "bounded stepping profile validation",
        Stage.MODEL, Implementation.LOWERED, Qualification.CPU_ORACLE,
        Execution.HOST, "unknown profiles and unsupported combinations rejected",
        "mujoco_metal/stepping.py:validate_stepping_profile",
        "accept listed profiles, reject others",
        "MetalSimulation", "baseline", ("test_stepping.py",),
    ),
    # Joint types.
    Requirement(
        "REQ-JNT-001", "hinge/slide/ball/free kinematics",
        Stage.KINEMATICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "one immutable model per batch",
        "mujoco_metal/metal_kinematics.py", "accept all four joint types",
        "MetalKinematics", "baseline", ("test_model.py",),
        enums=("mjtJoint.mjJNT_FREE", "mjtJoint.mjJNT_BALL",
               "mjtJoint.mjJNT_SLIDE", "mjtJoint.mjJNT_HINGE"),
    ),
    Requirement(
        "REQ-JNT-002", "scalar hinge/slide joint limits",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "ball/tendon limits rejected",
        "mujoco_metal/coupled_constraints.py:lower_coupled_constraints",
        "accept limited hinge/slide, reject limited ball",
        "MetalSimulation(profile='integrated_euler_v1')", "baseline",
        ("test_coupled_constraints.py",),
    ),
    Requirement(
        "REQ-JNT-003", "ball joint limits",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "pinned axis-angle rows in reserved slots (integrated profile; legacy joint-constraints profile stays scalar-only)",
        "mujoco_metal/shaders/coupled_constraints.metal", "ball branch in joint-limit loop",
        "MetalSimulation", "009", ("test_rigid_constraints_009.py",),
    ),
    # Geometry families.
    Requirement(
        "REQ-GEO-001", "plane/sphere/capsule/box collision",
        Stage.COLLISION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "all 9 valid unordered pairs; plane-plane yields nothing",
        "mujoco_metal/shaders/collision_primitives.metal",
        "accept the four primitives, reject others",
        "MetalSimulation(profile='integrated_euler_v1')", "baseline",
        ("test_primitive_collision_qualification.py",),
        enums=("mjtGeom.mjGEOM_PLANE", "mjtGeom.mjGEOM_SPHERE",
               "mjtGeom.mjGEOM_CAPSULE", "mjtGeom.mjGEOM_BOX"),
    ),
    Requirement(
        "REQ-GEO-002", "cylinder/ellipsoid collision",
        Stage.COLLISION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "rejected at lowering",
        "engine/engine_collision_primitive.c", "reject cylinders and ellipsoids",
        "none", "010", (),
        enums=("mjtGeom.mjGEOM_CYLINDER", "mjtGeom.mjGEOM_ELLIPSOID"),
    ),
    Requirement(
        "REQ-GEO-003", "convex mesh collision",
        Stage.COLLISION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "rejected at lowering; visual meshes unaffected",
        "engine/engine_collision_convex.c", "reject meshes",
        "none", "011", (),
        enums=("mjtGeom.mjGEOM_MESH",),
    ),
    Requirement(
        "REQ-GEO-004", "heightfield collision",
        Stage.COLLISION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "rejected at lowering",
        "engine/engine_collision_driver.c", "reject heightfields",
        "none", "012", (),
        enums=("mjtGeom.mjGEOM_HFIELD",),
    ),
    Requirement(
        "REQ-GEO-005", "SDF collision",
        Stage.COLLISION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "rejected at lowering; third-party SDF plugins per 019 contract",
        "engine/engine_collision_sdf.c", "reject SDFs",
        "none", "013", (),
        enums=("mjtGeom.mjGEOM_SDF",),
    ),
    Requirement(
        "REQ-GEO-006", "visual-only geometry decorations",
        Stage.RENDERING, Implementation.UPSTREAM_CPU_ONLY, Qualification.UNQUALIFIED,
        Execution.UPSTREAM_HOST, "renderer labels/decorations, never collision geometry",
        "doc/XMLreference.rst", "never admitted as collision geometry",
        "mujoco.Renderer (upstream host)", "out-of-scope", (),
        enums=("mjtGeom.mjGEOM_ARROW", "mjtGeom.mjGEOM_ARROW1",
               "mjtGeom.mjGEOM_ARROW2", "mjtGeom.mjGEOM_LINE",
               "mjtGeom.mjGEOM_LINEBOX", "mjtGeom.mjGEOM_FLEX",
               "mjtGeom.mjGEOM_SKIN", "mjtGeom.mjGEOM_LABEL",
               "mjtGeom.mjGEOM_TRIANGLE", "mjtGeom.mjGEOM_NONE"),
    ),
    Requirement(
        "REQ-GEO-007", "mjtGeom count sentinel",
        Stage.MODEL, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "mjNGEOMTYPES is a count, not a selectable type",
        "mujoco/mjmodel.h", "never admitted",
        "none", "005", (),
        enums=("mjtGeom.mjNGEOMTYPES",),
    ),
    # Contact condim/cone matrix.
    Requirement(
        "REQ-CON-001", "condim 1/3/4/6 with pyramidal/elliptic cones",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "other condim values rejected",
        "mujoco_metal/coupled_constraints.py",
        "accept 1/3/4/6, reject others",
        "MetalSimulation(profile='integrated_euler_v1')", "baseline",
        ("test_coupled_constraints.py",),
        enums=("mjtCone.mjCONE_PYRAMIDAL", "mjtCone.mjCONE_ELLIPTIC"),
    ),
    Requirement(
        "REQ-CON-002", "nine valid primitive contact pairs",
        Stage.COLLISION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "plane-plane yields nothing; cylinder/ellipsoid/mesh/hfield/SDF pairs per owning milestones",
        "mujoco_metal/coupled_constraints.py:pair_max_contacts",
        "accept the 9 primitive pairs",
        "MetalSimulation(profile='integrated_euler_v1')", "baseline",
        ("test_primitive_collision_qualification.py",),
    ),
    Requirement(
        "REQ-CON-003", "anisotropic sliding/torsional/rolling friction",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "five-coefficient friction expansion with pair mixing",
        "mujoco_metal/coupled_constraints.py", "accept finite friction rows",
        "MetalSimulation(profile='integrated_euler_v1')", "baseline",
        ("test_coupled_constraints.py",),
    ),
    # Integrators.
    Requirement(
        "REQ-INT-001", "Euler/RK4/implicitfast bounded profiles",
        Stage.INTEGRATION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "per-profile supported combinations only",
        "mujoco_metal/integration.py", "accept listed profile/integrator pairs",
        "MetalSimulation", "baseline", ("test_integration.py",),
        enums=("mjtIntegrator.mjINT_EULER", "mjtIntegrator.mjINT_RK4",
               "mjtIntegrator.mjINT_IMPLICITFAST"),
    ),
    Requirement(
        "REQ-INT-002", "full implicit integrator",
        Stage.INTEGRATION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "constant-damping implicitfast shortcut must not cover it",
        "engine/engine_forward.c", "reject mjINT_IMPLICIT outside guards",
        "none", "015", (),
        enums=("mjtIntegrator.mjINT_IMPLICIT",),
    ),
    # Jacobian/sparse options.
    Requirement(
        "REQ-JAC-001", "dense Jacobians",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "native pipeline is dense",
        "mujoco_metal/shaders/coupled_constraints.metal",
        "accept dense",
        "MetalSimulation(profile='integrated_euler_v1')", "baseline",
        ("test_coupled_constraints.py",),
        enums=("mjtJacobian.mjJAC_DENSE",),
    ),
    Requirement(
        "REQ-JAC-002", "sparse/auto Jacobians",
        Stage.CONSTRAINTS, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "no sparse path; auto must not silently select one",
        "engine/engine_util_sparse.c", "reject sparse",
        "none", "017", (),
        enums=("mjtJacobian.mjJAC_SPARSE", "mjtJacobian.mjJAC_AUTO"),
    ),
    # Solver selection.
    Requirement(
        "REQ-SOL-001", "PGS/Newton selector mapping",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "accepted names run the native projected solver, documented as a mapping",
        "mujoco_metal/coupled_constraints.py", "accept PGS/Newton as mapped",
        "MetalSimulation(profile='integrated_euler_v1')", "baseline",
        ("test_coupled_constraints.py",),
        enums=("mjtSolver.mjSOL_PGS", "mjtSolver.mjSOL_NEWTON"),
    ),
    Requirement(
        "REQ-SOL-002", "CG solver selection",
        Stage.CONSTRAINTS, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "rejected at lowering",
        "engine/engine_solver.c", "reject CG",
        "none", "014", (),
        enums=("mjtSolver.mjSOL_CG",),
    ),
    # Equalities.
    Requirement(
        "REQ-EQ-001", "joint/connect/weld equalities with per-env activity",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "1/3/6 rows, body/world/site forms incl. mocap anchors, torquescale, schema-2/3 activity state",
        "mujoco_metal/shaders/equality_assembly.metal",
        "accept joint/connect/weld body/site/world/mocap, reject others even if inactive",
        "sim.set_equality_active", "004",
        ("test_connect_weld.py", "test_equality_activity.py",
         "test_equality_qualification.py", "test_mocap_state.py"),
        enums=("mjtEq.mjEQ_JOINT", "mjtEq.mjEQ_CONNECT", "mjtEq.mjEQ_WELD"),
    ),
    Requirement(
        "REQ-EQ-002", "tendon equality",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "cubic tendon-length coupling as coupled rows (single or paired tendons)",
        "mujoco_metal/coupled_constraints.py", "tendon_constraint_rows equality branch",
        "MetalSimulation", "008", ("test_spatial_tendons_008.py",),
        enums=("mjtEq.mjEQ_TENDON",),
    ),
    Requirement(
        "REQ-EQ-003", "flex equalities",
        Stage.CONSTRAINTS, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "rejected at lowering",
        "engine/engine_core_constraint.c", "reject flex/flexvert/flexstrain",
        "none", "018", (),
        enums=("mjtEq.mjEQ_FLEX", "mjtEq.mjEQ_FLEXVERT",
               "mjtEq.mjEQ_FLEXSTRAIN"),
    ),
    Requirement(
        "REQ-EQ-004", "removed distance equality",
        Stage.CONSTRAINTS, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "removed upstream in MuJoCo 2.2.2; rejected",
        "doc/XMLreference.rst", "reject distance equalities",
        "none", "009", (),
        enums=("mjtEq.mjEQ_DISTANCE",),
    ),
    # Transmissions.
    Requirement(
        "REQ-TEN-001", "fixed-joint tendons with spring/damping/armature",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "fixed joint paths only; no wrapping, limits, friction or equality",
        "mujoco_metal/tendons.py", "accept fixed joint tendons",
        "MetalSimulation", "baseline", ("test_tendons.py",),
    ),
    Requirement(
        "REQ-TEN-002", "spatial tendons, wrapping, limits, friction, equality",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "pulley/site/sphere/cylinder paths; tendon limits, friction loss and tendon equality as coupled rows",
        "mujoco_metal/spatial_tendons.py", "spatial kinematics + tendon_constraint_rows",
        "MetalSimulation", "008", ("test_spatial_tendons_008.py",),
    ),
    Requirement(
        "REQ-TRN-001", "joint/jointinparent/tendon transmissions",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "stateless fixed/affine scalar scope",
        "mujoco_metal/transmissions.py", "accept listed transmissions",
        "MetalSimulation", "baseline", ("test_transmissions.py",),
        enums=("mjtTrn.mjTRN_JOINT", "mjtTrn.mjTRN_JOINTINPARENT",
               "mjtTrn.mjTRN_TENDON"),
    ),
    Requirement(
        "REQ-TRN-002", "slidercrank/site/body transmissions",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "rigid transmissions incl. ball/free gears; spatial-tendon paths stay in 008",
        "mujoco_metal/stateful_actuation.py", "general kinematics + BODY adhesion from same-step candidates",
        "MetalSimulation", "007", ("test_actuators_007.py",),
        enums=("mjtTrn.mjTRN_SLIDERCRANK", "mjtTrn.mjTRN_SITE",
               "mjtTrn.mjTRN_BODY"),
    ),
    Requirement(
        "REQ-TRN-003", "undefined transmission sentinel",
        Stage.MODEL, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "never admitted",
        "mujoco/mjmodel.h", "reject undefined",
        "none", "005", (),
        enums=("mjtTrn.mjTRN_UNDEFINED",),
    ),
    # Actuator dynamics/gains/biases.
    Requirement(
        "REQ-DYN-001", "stateless (none) actuator dynamics",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "no activation state",
        "mujoco_metal/actuation.py", "accept dyntype none",
        "MetalSimulation", "baseline", ("test_actuation.py",),
        enums=("mjtDyn.mjDYN_NONE",),
    ),
    Requirement(
        "REQ-DYN-002", "stateful activation dynamics",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "built-in integrator/filter/filterexact/muscle/dcmotor; USER callbacks stay in 019",
        "mujoco_metal/stateful_actuation.py", "act_dot switch + exact-slot advance, schema-4 state",
        "MetalSimulation", "007", ("test_actuators_007.py",),
        enums=("mjtDyn.mjDYN_INTEGRATOR", "mjtDyn.mjDYN_FILTER",
               "mjtDyn.mjDYN_FILTEREXACT", "mjtDyn.mjDYN_MUSCLE",
               "mjtDyn.mjDYN_DCMOTOR", "mjtDyn.mjDYN_USER"),
    ),
    Requirement(
        "REQ-GAIN-001", "fixed/affine gains",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "stateless scalar scope",
        "mujoco_metal/actuation.py", "accept fixed/affine",
        "MetalSimulation", "baseline", ("test_actuation.py",),
        enums=("mjtGain.mjGAIN_FIXED", "mjtGain.mjGAIN_AFFINE"),
    ),
    Requirement(
        "REQ-GAIN-002", "muscle/DC-motor/user gains",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "built-in muscle/dcmotor; USER callbacks stay in 019",
        "mujoco_metal/stateful_actuation.py", "muscle FLV + DC resistance/voltage paths",
        "MetalSimulation", "007", ("test_actuators_007.py",),
        enums=("mjtGain.mjGAIN_MUSCLE", "mjtGain.mjGAIN_DCMOTOR",
               "mjtGain.mjGAIN_USER"),
    ),
    Requirement(
        "REQ-BIAS-001", "none/affine biases",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "stateless scalar scope",
        "mujoco_metal/actuation.py", "accept none/affine",
        "MetalSimulation", "baseline", ("test_actuation.py",),
        enums=("mjtBias.mjBIAS_NONE", "mjtBias.mjBIAS_AFFINE"),
    ),
    Requirement(
        "REQ-BIAS-002", "muscle/DC-motor/user biases",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "built-in muscle passive + DC back-EMF/cogging/LuGre; USER stays in 019",
        "mujoco_metal/stateful_actuation.py", "bias switch + post-clamp DC mechanics",
        "MetalSimulation", "007", ("test_actuators_007.py",),
        enums=("mjtBias.mjBIAS_MUSCLE", "mjtBias.mjBIAS_DCMOTOR",
               "mjtBias.mjBIAS_USER"),
    ),
    # Sensors: 14 supported, 35 deferred to 016.
    Requirement(
        "REQ-SENS-001", "current-state kinematic/clock/gyro/velocimeter queries",
        Stage.SENSOR, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "explicit query, not stored mj_step timing; no delay/noise/history",
        "mujoco_metal/sensors.py", "accept the 14 listed types",
        "sim.sensor_values()", "baseline",
        ("test_sensors.py",),
        enums=("mjtSensor.mjSENS_JOINTPOS", "mjtSensor.mjSENS_JOINTVEL",
               "mjtSensor.mjSENS_BALLQUAT", "mjtSensor.mjSENS_BALLANGVEL",
               "mjtSensor.mjSENS_FRAMEPOS", "mjtSensor.mjSENS_FRAMEQUAT",
               "mjtSensor.mjSENS_FRAMEXAXIS", "mjtSensor.mjSENS_FRAMEYAXIS",
               "mjtSensor.mjSENS_FRAMEZAXIS", "mjtSensor.mjSENS_CLOCK",
               "mjtSensor.mjSENS_FRAMELINVEL", "mjtSensor.mjSENS_FRAMEANGVEL",
               "mjtSensor.mjSENS_GYRO", "mjtSensor.mjSENS_VELOCIMETER"),
    ),
    Requirement(
        "REQ-SENS-002", "remaining sensor families",
        Stage.SENSOR, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "touch/acceleration/force/torque/magnetic/range/camera/tendon/actuator/limit/energy/subtree/site/geom/contact/user/plugin/tactile rejected",
        "mujoco_metal/sensors.py:lower_sensors", "reject unlisted sensor types",
        "none", "016", (),
        enums=("mjtSensor.mjSENS_TOUCH", "mjtSensor.mjSENS_ACCELEROMETER",
               "mjtSensor.mjSENS_FORCE", "mjtSensor.mjSENS_TORQUE",
               "mjtSensor.mjSENS_MAGNETOMETER", "mjtSensor.mjSENS_RANGEFINDER",
               "mjtSensor.mjSENS_CAMPROJECTION", "mjtSensor.mjSENS_TENDONPOS",
               "mjtSensor.mjSENS_TENDONVEL", "mjtSensor.mjSENS_ACTUATORPOS",
               "mjtSensor.mjSENS_ACTUATORVEL", "mjtSensor.mjSENS_ACTUATORFRC",
               "mjtSensor.mjSENS_JOINTACTFRC", "mjtSensor.mjSENS_TENDONACTFRC",
               "mjtSensor.mjSENS_JOINTLIMITPOS", "mjtSensor.mjSENS_JOINTLIMITVEL",
               "mjtSensor.mjSENS_JOINTLIMITFRC", "mjtSensor.mjSENS_TENDONLIMITPOS",
               "mjtSensor.mjSENS_TENDONLIMITVEL", "mjtSensor.mjSENS_TENDONLIMITFRC",
               "mjtSensor.mjSENS_FRAMELINACC", "mjtSensor.mjSENS_FRAMEANGACC",
               "mjtSensor.mjSENS_SUBTREECOM", "mjtSensor.mjSENS_SUBTREELINVEL",
               "mjtSensor.mjSENS_SUBTREEANGMOM", "mjtSensor.mjSENS_INSIDESITE",
               "mjtSensor.mjSENS_GEOMDIST", "mjtSensor.mjSENS_GEOMNORMAL",
               "mjtSensor.mjSENS_GEOMFROMTO", "mjtSensor.mjSENS_CONTACT",
               "mjtSensor.mjSENS_E_POTENTIAL", "mjtSensor.mjSENS_E_KINETIC",
               "mjtSensor.mjSENS_TACTILE", "mjtSensor.mjSENS_PLUGIN",
               "mjtSensor.mjSENS_USER"),
    ),
    # Runtime state fields (mjtState).
    Requirement(
        "REQ-STATE-001", "time/qpos/qvel/eq_active/mocap state ownership",
        Stage.API, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "persistent per-env poses/masks, schema-3 snapshots, keyframe reset, copy",
        "mujoco_metal/device_state.py", "accept matching snapshots, reject v1-into-neq and <v3-into-mocap",
        "sim.state", "006", ("test_device_state.py", "test_equality_activity.py",
         "test_mocap_state.py"),
        enums=("mjtState.mjSTATE_TIME", "mjtState.mjSTATE_QPOS",
               "mjtState.mjSTATE_QVEL", "mjtState.mjSTATE_EQ_ACTIVE"),
    ),
    Requirement(
        "REQ-STATE-002", "control/applied-force state ownership",
        Stage.API, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "per-call held inputs validated before the device loop",
        "mujoco_metal/simulation.py", "accept finite host/device inputs",
        "sim.step", "baseline", ("test_simulation.py",),
        enums=("mjtState.mjSTATE_CTRL", "mjtState.mjSTATE_QFRC_APPLIED",
               "mjtState.mjSTATE_XFRC_APPLIED"),
    ),
    Requirement(
        "REQ-STATE-003", "mocap position/quaternion inputs",
        Stage.KINEMATICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "jointless world-child mocap bodies; per-env poses, keyframe reset, schema-3 snapshots",
        "mujoco_metal/simulation.py:MetalSimulation.set_mocap", "accept valid mocap, reject non-world-child",
        "sim.set_mocap", "006", ("test_mocap_state.py",),
        enums=("mjtState.mjSTATE_MOCAP_POS", "mjtState.mjSTATE_MOCAP_QUAT"),
    ),
    Requirement(
        "REQ-STATE-004", "actuator activation state",
        Stage.DYNAMICS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "schema-4 act storage, reset/keyframe/restore/copy, exact-slot advance",
        "mujoco_metal/device_state.py", "na rows with actearly/actrange semantics",
        "MetalSimulation", "007", ("test_actuators_007.py",),
        enums=("mjtState.mjSTATE_ACT",),
    ),
    Requirement(
        "REQ-STATE-005", "warmstart state",
        Stage.CONSTRAINTS, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "cold starts only",
        "engine/engine_solver.c", "ignore warmstart content",
        "none", "014", (),
        enums=("mjtState.mjSTATE_WARMSTART",),
    ),
    Requirement(
        "REQ-STATE-006", "history state",
        Stage.INTEGRATION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "no sensor/actuator history storage",
        "engine/engine_forward.c", "reject history-dependent models",
        "none", "015", (),
        enums=("mjtState.mjSTATE_HISTORY",),
    ),
    Requirement(
        "REQ-STATE-007", "userdata/plugin state",
        Stage.API, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "no userdata/plugin state ownership",
        "engine/engine_plugin.cc", "reject stateful plugins",
        "none", "019", (),
        enums=("mjtState.mjSTATE_USERDATA", "mjtState.mjSTATE_PLUGIN",
               "mjtState.mjSTATE_USER"),
    ),
    Requirement(
        "REQ-STATE-008", "getState group selectors",
        Stage.API, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "no mj_getState/mj_setState group API",
        "engine/engine_io.c", "not exposed",
        "none", "019", (),
        enums=("mjtState.mjSTATE_PHYSICS", "mjtState.mjSTATE_FULLPHYSICS",
               "mjtState.mjSTATE_INTEGRATION"),
    ),
    Requirement(
        "REQ-STATE-009", "state count sentinel",
        Stage.MODEL, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "mjNSTATE is a count, not selectable state",
        "mujoco/mjmodel.h", "never admitted",
        "none", "005", (),
        enums=("mjtState.mjNSTATE",),
    ),
    # Disable flags: honored vs deferred.
    Requirement(
        "REQ-DSBL-001", "honored disable flags",
        Stage.API, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "constraint/equality/frictionloss/limit/contact/spring/damper/gravity/clampctrl/warmstart/filterparent/actuation/refsafe/sensor/midphase/eulerdamp/autoreset honored per stage",
        "mujoco_metal/stepping.py", "accept listed flags",
        "MetalSimulation", "baseline",
        ("test_simulation.py",),
        enums=("mjtDisableBit.mjDSBL_CONSTRAINT", "mjtDisableBit.mjDSBL_EQUALITY",
               "mjtDisableBit.mjDSBL_FRICTIONLOSS", "mjtDisableBit.mjDSBL_LIMIT",
               "mjtDisableBit.mjDSBL_CONTACT", "mjtDisableBit.mjDSBL_SPRING",
               "mjtDisableBit.mjDSBL_DAMPER", "mjtDisableBit.mjDSBL_GRAVITY",
               "mjtDisableBit.mjDSBL_CLAMPCTRL", "mjtDisableBit.mjDSBL_WARMSTART",
               "mjtDisableBit.mjDSBL_FILTERPARENT", "mjtDisableBit.mjDSBL_ACTUATION",
               "mjtDisableBit.mjDSBL_REFSAFE", "mjtDisableBit.mjDSBL_SENSOR",
               "mjtDisableBit.mjDSBL_MIDPHASE", "mjtDisableBit.mjDSBL_EULERDAMP",
               "mjtDisableBit.mjDSBL_AUTORESET"),
    ),
    Requirement(
        "REQ-DSBL-002", "island/ccd disable flags",
        Stage.INTEGRATION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "no island/ccd execution paths",
        "engine/engine_island.c", "reject nativeccd/island/multiccd",
        "none", "017", (),
        enums=("mjtDisableBit.mjDSBL_NATIVECCD", "mjtDisableBit.mjDSBL_ISLAND",
               "mjtDisableBit.mjDSBL_MULTICCD"),
    ),
    Requirement(
        "REQ-DSBL-003", "disable-bit count sentinel",
        Stage.MODEL, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "mjNDISABLE is a count, not a flag",
        "mujoco/mjmodel.h", "never admitted",
        "none", "005", (),
        enums=("mjtDisableBit.mjNDISABLE",),
    ),
    # Enable flags.
    Requirement(
        "REQ-ENBL-001", "override/energy enable flags",
        Stage.API, Implementation.CPU_REFERENCE, Qualification.CPU_ORACLE,
        Execution.HOST, "override rejected for contacts; energy is metadata-only",
        "mujoco_metal/coupled_constraints.py", "reject override",
        "load_model", "baseline", ("test_coupled_constraints.py",),
        enums=("mjtEnableBit.mjENBL_OVERRIDE", "mjtEnableBit.mjENBL_ENERGY"),
    ),
    Requirement(
        "REQ-ENBL-002", "forward/inverse enable flags",
        Stage.DYNAMICS, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "no fwdinv/invdiscrete execution paths",
        "engine/engine_inverse.c", "reject fwdinv/invdiscrete",
        "none", "019", (),
        enums=("mjtEnableBit.mjENBL_FWDINV", "mjtEnableBit.mjENBL_INVDISCRETE"),
    ),
    Requirement(
        "REQ-ENBL-003", "sleep enable flag",
        Stage.INTEGRATION, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "sleep rejected; no sleeping execution path",
        "engine/engine_sleep.h", "reject sleep",
        "none", "017", (),
        enums=("mjtEnableBit.mjENBL_SLEEP",),
    ),
    Requirement(
        "REQ-ENBL-004", "exact-diagonal enable flag",
        Stage.CONSTRAINTS, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "diagapprox used; diagexact not executed",
        "engine/engine_core_constraint.c", "ignore diagexact",
        "none", "014", (),
        enums=("mjtEnableBit.mjENBL_DIAGEXACT",),
    ),
    Requirement(
        "REQ-ENBL-005", "enable-bit count sentinel",
        Stage.MODEL, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "mjNENABLE is a count, not a flag",
        "mujoco/mjmodel.h", "never admitted",
        "none", "005", (),
        enums=("mjtEnableBit.mjNENABLE",),
    ),
    # Stepping profiles.
    Requirement(
        "REQ-PROF-001", "contact_free_euler_v1 baseline",
        Stage.INTEGRATION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "unforced Euler baseline",
        "mujoco_metal/stepping.py", "accept contact-disabled Euler models",
        "MetalSimulation", "baseline", ("test_simulation.py",),
    ),
    Requirement(
        "REQ-PROF-002", "forces/motor/transmission/fluid/passive/sensor Euler profiles",
        Stage.INTEGRATION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "per-profile bounded subsets",
        "mujoco_metal/stepping.py", "accept listed profile/model pairs",
        "MetalSimulation", "baseline",
        ("test_forces.py", "test_transmissions.py", "test_fluid.py",
         "test_passive.py", "test_sensors.py"),
    ),
    Requirement(
        "REQ-PROF-003", "joint_constraints_euler_v1 scalar stage",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "scalar-only stage; connect/weld explicitly not admitted",
        "mujoco_metal/joint_constraints.py", "accept scalar joint content only",
        "MetalSimulation", "baseline", ("test_joint_constraints.py",),
    ),
    Requirement(
        "REQ-PROF-004", "normal/friction contact Euler profiles",
        Stage.CONSTRAINTS, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "bounded normal/pyramidal sphere slices",
        "mujoco_metal/contact.py", "accept listed contact content only",
        "MetalSimulation", "baseline", ("test_contact.py",),
    ),
    Requirement(
        "REQ-PROF-005", "integrated_euler_v1 unified pipeline",
        Stage.INTEGRATION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "nv<=32, pairs<=16, slots<=24, rows<=96",
        "mujoco_metal/simulation.py", "accept validated integrated models",
        "MetalSimulation(profile='integrated_euler_v1')", "004",
        ("test_integrated_simulation.py", "test_connect_weld.py",
         "test_equality_activity.py", "test_equality_qualification.py"),
    ),
    Requirement(
        "REQ-PROF-006", "RK4 variants",
        Stage.INTEGRATION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "contact-free RK4 subsets; integrated RK4 deferred",
        "mujoco_metal/runge_kutta.py", "accept listed RK4 profile/model pairs",
        "MetalSimulation", "baseline", ("test_runge_kutta.py",),
    ),
    Requirement(
        "REQ-PROF-007", "implicitfast profile",
        Stage.INTEGRATION, Implementation.NATIVE_GPU, Qualification.GPU_QUALIFIED,
        Execution.DEVICE, "constant damping + eligible free-body midpoint",
        "mujoco_metal/implicit.py", "accept eligible models",
        "MetalSimulation", "baseline", ("test_implicit.py",),
    ),
    # Upstream-host API routes (explicit hybrid/host utilities, not GPU rewrites).
    Requirement(
        "REQ-API-001", "host XML compilation/model editing/serialization",
        Stage.API, Implementation.UPSTREAM_CPU_ONLY, Qualification.UNQUALIFIED,
        Execution.UPSTREAM_HOST, "upstream host route, not a GPU rewrite",
        "mujoco.MjSpec", "host only",
        "mujoco.MjModel.from_xml_string", "005", (),
    ),
    Requirement(
        "REQ-API-002", "host OpenGL visualization",
        Stage.RENDERING, Implementation.UPSTREAM_CPU_ONLY, Qualification.UNQUALIFIED,
        Execution.UPSTREAM_HOST, "OpenGL only; no Metal renderer",
        "mujoco.Renderer", "host only",
        "demo_recording.ComparisonRecorder", "out-of-scope", (),
    ),
    Requirement(
        "REQ-REL-001", "wheel build/install qualification and coverage closure",
        Stage.API, Implementation.NOT_IMPLEMENTED, Qualification.UNQUALIFIED,
        Execution.NONE, "clean workspace-local wheel install, bundled shaders/assets, CLI, viewer startup, reconciled matrix",
        "metal/pyproject.toml", "build/install in workspace-local env",
        "pip install ./metal", "020", (),
    ),
)


def requirement_ids():
  """Return all requirement IDs in definition order."""
  return tuple(req.id for req in REQUIREMENTS)


def enum_coverage():
  """Map each covered `Enum.member` string to its requirement IDs."""
  covered = {}
  for req in REQUIREMENTS:
    for member in req.enums:
      covered.setdefault(member, []).append(req.id)
  return covered


def unclassified_enum_members():
  """Return pinned enum members covered by no requirement (drift signal)."""
  covered = enum_coverage()
  missing = []
  for enum_name in _ENUMS:
    enum_type = getattr(mujoco, enum_name, None)
    if enum_type is None:
      missing.append(enum_name)
      continue
    for member_name in enum_type.__members__:
      if f"{enum_name}.{member_name}" not in covered:
        missing.append(f"{enum_name}.{member_name}")
  return tuple(missing)


# Model-field classification: longest-prefix match against FIELD_RULES, with
# exact-name exceptions checked first. Every rule must match at least one field
# (dead rules fail the drift test); every field must match exactly one rule.
FIELD_RULES = (
    # Named-object accessors (not physics constants) and counts.
    ("actuator", "007", "actuator objects"),
    ("body", "baseline", "body objects"),
    ("cam", "out-of-scope", "camera objects are visualization"),
    ("eq", "004", "equality objects"),
    ("exclude", "baseline", "exclusion objects"),
    ("geom", "baseline", "geom objects"),
    ("jnt", "baseline", "joint objects"),
    ("mat", "011", "material objects"),
    ("pair", "baseline", "explicit contact pairs"),
    ("sensor", "baseline", "sensor objects; unadmitted types in 016"),
    ("site", "baseline", "site objects"),
    ("tendon", "008", "tendon objects"),
    ("nbvh", "011", "BVH counts"),
    ("nbvhdynamic", "011", "BVH counts"),
    ("nbvhstatic", "011", "BVH counts"),
    ("nemax", "004", "equality sizing"),
    ("nexclude", "baseline", "exclusion counts"),
    ("nflex", "018", "flex counts"),
    ("nflexbending", "018", "flex counts"),
    ("nflexedge", "018", "flex counts"),
    ("nflexelem", "018", "flex counts"),
    ("nflexelemdata", "018", "flex counts"),
    ("nflexelemedge", "018", "flex counts"),
    ("nflexevpair", "018", "flex counts"),
    ("nflexnode", "018", "flex counts"),
    ("nflexshelldata", "018", "flex counts"),
    ("nflexstiffness", "018", "flex counts"),
    ("nflextexcoord", "018", "flex counts"),
    ("nflexvert", "018", "flex counts"),
    ("ngravcomp", "baseline", "gravity-compensation counts"),
    ("nhistory", "015", "history counts"),
    ("nmat", "011", "material counts"),
    ("nnames", "out-of-scope", "naming metadata"),
    ("nnames_map", "out-of-scope", "naming metadata"),
    ("noct", "011", "mesh octree counts"),
    ("npair", "baseline", "pair counts"),
    ("ntendon", "008", "tendon counts"),
    # Actuation families.
    ("actuator_", "007", "actuator dynamics/gains/biases; zero-armature subset in baseline"),
    # Tendons: fixed subset baseline, spatial/limits/friction/equality later.
    ("tendon_", "008", "tendon mechanics; fixed-joint subset in baseline"),
    ("ten_", "008", "tendon derived quantities"),
    ("wrap_", "008", "tendon path/wrapping"),
    # Deformables.
    ("flex", "018", "flex/deformable subsystem"),
    ("flexedge", "018", "flex edges"),
    ("flexvert", "018", "flex vertices"),
    ("mesh", "011", "mesh assets and collision hulls"),
    ("skin", "011", "skins follow meshes"),
    ("hfield", "012", "heightfields"),
    ("tex", "011", "mesh textures"),
    ("texture", "011", "mesh textures"),
    ("mat_", "011", "mesh materials"),
    ("material", "011", "mesh materials"),
    ("bvh", "011", "collision acceleration structures"),
    # Cameras/lights/visuals/naming: upstream host, out of scope for physics.
    ("cam_", "out-of-scope", "cameras are visualization"),
    ("camera", "out-of-scope", "cameras are visualization"),
    ("light", "out-of-scope", "lights are visualization"),
    ("vis", "out-of-scope", "visualization only"),
    ("name", "out-of-scope", "naming metadata"),
    ("names", "out-of-scope", "naming metadata"),
    ("text", "out-of-scope", "custom text metadata"),
    ("tuple", "out-of-scope", "custom tuple metadata"),
    ("numeric", "out-of-scope", "custom numeric metadata"),
    ("nuser", "out-of-scope", "custom allocation counts"),
    ("ntext", "out-of-scope", "custom text counts"),
    ("ntex", "out-of-scope", "asset counts"),
    ("nmesh", "out-of-scope", "asset counts"),
    ("nskin", "out-of-scope", "asset counts"),
    ("nhfield", "out-of-scope", "asset counts"),
    ("nlight", "out-of-scope", "visual counts"),
    ("ncam", "out-of-scope", "visual counts"),
    ("nkey", "out-of-scope", "keyframes are host-side"),
    ("key", "out-of-scope", "keyframes are host-side"),
    ("keyframe", "out-of-scope", "keyframes are host-side"),
    ("plugin", "019", "plugin lifecycle"),
    ("nplugin", "019", "plugin counts"),
    # Solver/island/sleep internals.
    ("map", "017", "sparse/index maps"),
    ("B_", "017", "sparse structures"),
    ("D_", "017", "sparse structures"),
    ("M_", "017", "sparse structures"),
    ("nB", "017", "sparse counts"),
    ("nC", "017", "sparse counts"),
    ("nD", "017", "sparse counts"),
    ("nM", "017", "sparse counts"),
    ("nJmom", "007", "actuator moment counts"),
    ("nJten", "008", "tendon Jacobian counts"),
    ("nJfe", "018", "flex Jacobian counts"),
    ("nJfv", "018", "flex Jacobian counts"),
    ("oct", "011", "mesh octree"),
    ("tree", "017", "kinematic tree/island bookkeeping"),
    ("ntree", "017", "tree counts"),
    ("stat", "out-of-scope", "model statistics"),
    ("from", "out-of-scope", "internal source ranges"),
    ("signature", "out-of-scope", "pair/exclude signatures"),
    ("bind", "out-of-scope", "skin binding metadata"),
    ("equality", "004", "equality count metadata"),
    ("joint", "baseline", "joint count metadata"),
    ("nq", "baseline", "dimensions"),
    ("nv", "baseline", "dimensions"),
    ("nu", "baseline", "dimensions"),
    ("na", "007", "activation dimensions"),
    ("nbody", "baseline", "dimensions"),
    ("njnt", "baseline", "dimensions"),
    ("ngeom", "baseline", "dimensions"),
    ("nsite", "baseline", "dimensions"),
    ("neq", "004", "equality dimensions"),
    ("nmocap", "006", "mocap counts"),
    ("nconmax", "baseline", "contact workspace sizing"),
    ("njmax", "baseline", "constraint workspace sizing"),
    ("narena", "017", "arena sizing"),
    ("nbuffer", "017", "buffer counts"),
    ("nuserdata", "out-of-scope", "custom counts"),
    ("ntuple", "out-of-scope", "custom counts"),
    ("nwrap", "008", "tendon wrap counts"),
    ("nnumeric", "out-of-scope", "custom counts"),
    ("ntextdata", "out-of-scope", "custom counts"),
    ("ntupledata", "out-of-scope", "custom counts"),
    ("ntexdata", "011", "asset data counts"),
    ("npaths", "out-of-scope", "asset path counts"),
    ("paths", "out-of-scope", "asset paths"),
    # Core rigid-body/dynamics/contact/constraint fields used by lowering.
    ("body_", "baseline", "rigid-body constants"),
    ("jnt_", "baseline", "joint constants"),
    ("dof_", "baseline", "dof constants"),
    ("geom_", "baseline", "geom constants"),
    ("site_", "baseline", "site constants"),
    ("eq_", "004", "equality constants"),
    ("pair_", "baseline", "explicit contact pairs"),
    ("exclude_", "baseline", "exclusions"),
    ("sensor_", "baseline", "admitted sensor subset; rest in 016"),
    ("nsensor", "baseline", "sensor counts"),
    ("nsensordata", "baseline", "sensor counts"),
    ("qpos", "baseline", "reference configuration"),
    ("opt", "baseline", "options; unlisted option handling per milestone"),
)


def classify_model_field(name):
  """Return the (milestone, note) rule for one MjModel attribute name."""
  best = None
  for rule, milestone, note in FIELD_RULES:
    if name == rule or name.startswith(rule):
      if best is None or len(rule) > len(best[0]):
        best = (rule, milestone, note)
  if best is None:
    return None
  return best[1], best[2]


def coverage_table():
  """Render the requirement inventory as a Markdown table."""
  lines = [
      "| ID | Capability | Execution | Status | Owner | Admission | Tests |",
      "|---|---|---|---|---|---|---|",
  ]
  for req in REQUIREMENTS:
    status = f"{req.implementation.value}/{req.qualification.value}"
    tests = ", ".join(req.tests) if req.tests else "—"
    lines.append(
        f"| {req.id} | {req.name} | {req.execution.value} | {status} | "
        f"{req.milestone} | {req.admission} | {tests} |"
    )
  return "\n".join(lines) + "\n"


def _jsonable_requirement(req):
  data = asdict(req)
  data["stage"] = req.stage.value
  data["implementation"] = req.implementation.value
  data["qualification"] = req.qualification.value
  data["execution"] = req.execution.value
  data["tests"] = list(req.tests)
  data["enums"] = list(req.enums)
  return data


def feature_status():
  """Return the immutable pinned-version inventory without loading Torch."""
  return FEATURES
